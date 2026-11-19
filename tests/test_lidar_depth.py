"""Raw depth decoding, registration, fusion and dataset publication."""
from pathlib import Path
from types import SimpleNamespace
import json
import struct
import sys
import tempfile
import unittest
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lidar_depth as ld
from prior_pose_refine import cmd_export_depth


def read_png(path):
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    while offset < len(data):
        size, kind = struct.unpack_from(">I4s", data, offset)
        chunk = data[offset+8:offset+8+size]
        assert zlib.crc32(kind+chunk) == struct.unpack_from(">I", data, offset+8+size)[0]
        if kind == b"IHDR":
            w, h, bits, color, *_ = struct.unpack(">2I5B", chunk)
            assert bits == 16 and color in (0, 2)
            channels = 1 if color == 0 else 3
        if kind == b"IDAT":
            rows = np.frombuffer(zlib.decompress(chunk), dtype="u1").reshape(h, 2*w*channels+1)
            assert not rows[:, 0].any()
            samples = rows[:, 1:].copy().view(">u2").reshape((h, w) if channels == 1 else (h, w, 3))
        offset += size + 12
    return samples


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.scan = root/"scan"
        self.dataset = root/"dataset"
        self.model = self.dataset/"sparse/0"
        self.model.mkdir(parents=True)
        self.opts = SimpleNamespace(scan=self.scan, model=self.model, dataset=self.dataset)
        self.scan.mkdir()
        (self.scan/"depth").mkdir()
        (self.scan/"capture-meta.json").write_text(json.dumps({
            "app": "fable-gs-capture", "depthFormat": "float32_raw_little_endian"}))
        row = dict(imageFile="cam/photo.jpg", timestamp=12.04, depthWidth=32, depthHeight=24,
                   depthFile="z.bin", confidenceFile="c.bin")
        (self.scan/"poses.jsonl").write_text(json.dumps(row))
        self.conf = np.zeros((24, 32), "u1")
        self.conf[:, :18] = 2
        self.conf.tofile(self.scan/"depth/c.bin")
        np.full(self.conf.shape, 2, "<f4").tofile(self.scan/"depth/z.bin")
        (self.model/"cameras.bin").write_bytes(struct.pack("<QiiQQ4d", 1, 5, 1, 32, 24, 20, 20, 16, 12))
        (self.model/"images.bin").write_bytes(struct.pack("<QI7dI", 1, 23, 1, 0, 0, 0, 0, 0, 0, 5)
                                               + b"cam/photo.jpg\0" + struct.pack("<Q", 0))

    def make_stream(self):
        directory = self.scan/"mid360/continuous"
        directory.mkdir(parents=True)
        cal = dict(id="rig", serial="both", units="meters", T_cameraCV_lidar=np.eye(4).ravel().tolist())
        (directory.parent/"calibration.json").write_text(json.dumps(cal))
        session = dict(calibrationID="rig", serial="both", poseLog="poses.jsonl",
                       segments=[dict(recording=dict(chunks=["p.mid360"]))])
        (directory/"session.json").write_text(json.dumps(session))
        m = [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0]
        (directory/"poses.jsonl").write_text("\n".join(json.dumps(dict(t=t, m=m)) for t in (12, 12.05, 12.1)))
        points = [(100*x+50, 100*y+50, 2000, 50, 0) for x in range(-10, 12) for y in range(-8, 8)]
        # A farther return shares a pixel and a tagged outlier would lie in front of the plane.
        points += [(1150*2, 50*2, 4000, 50, 0), (575, 25, 1000, 50, 1)]
        payload = b"".join(struct.pack("<iiiBB", *p) for p in points)
        packet = bytearray(36 + len(payload))
        struct.pack_into("<HHH", packet, 1, len(packet), 1000, len(points))
        packet[10] = 1
        struct.pack_into("<Q", packet, 28, 2_000_000_000)
        packet[36:] = payload
        path = directory/"p.mid360"
        path.write_bytes(b"MID360R1" + struct.pack("<HHd", 56301, len(packet), 12.0401) + packet)
        return path

    def test_normal_entry_point_fuses_raw_inputs_and_preserves_names(self):
        self.make_stream()
        cmd_export_depth(self.opts)
        depth = read_png(self.dataset/"depth/cam/photo.png")
        np.testing.assert_array_equal(depth[:, :18], 2000)
        self.assertEqual(depth[12, 27], 2000)
        self.assertEqual(depth[0, 31], 0)
        normal = read_png(self.dataset/"normals/cam/photo.png")
        np.testing.assert_array_equal(normal[12, 10], [32768, 32768, 0])
        np.testing.assert_array_equal(normal[0, 31], [32768, 32768, 32768])
        # Repeat or resume an interrupted publication, preserving original maps.
        (self.dataset/"normals/cam/photo.png").unlink()
        (self.dataset/"normals/cam").rmdir()
        (self.dataset/"normals").rmdir()
        cmd_export_depth(self.opts)
        np.testing.assert_array_equal(normal, read_png(self.dataset/"normals/cam/photo.png"))
        path = self.dataset/"depth/cam/photo.png"
        path.write_bytes(b"external depth")
        with self.assertRaisesRegex(ValueError, "different maps"):
            cmd_export_depth(self.opts)
        self.assertEqual(path.read_bytes(), b"external depth")

    def test_single_source_and_generic_scan(self):
        cmd_export_depth(self.opts)
        np.testing.assert_array_equal(read_png(self.dataset/"depth/cam/photo.png"), (self.conf == 2)*2000)
        np.zeros_like(self.conf).tofile(self.scan/"depth/c.bin")
        with self.assertRaisesRegex(ValueError, "different maps"):
            cmd_export_depth(self.opts)
        (self.scan/"capture-meta.json").unlink()
        cmd_export_depth(self.opts)

    def test_damaged_stream_cannot_publish_partial_maps(self):
        path = self.make_stream()
        with path.open("ab") as f:
            f.write(b"bad")
        with self.assertRaisesRegex(ValueError, "truncated"):
            cmd_export_depth(self.opts)
        self.assertFalse((self.dataset/"depth").exists())
        self.assertFalse(list(self.dataset.glob(".depth-*")))

    def test_calibration_mismatch_and_reordered_udp_packets(self):
        path = self.make_stream()
        record = path.read_bytes()[8:]
        later = bytearray(record)
        struct.pack_into("<Q", later, 12+28, 2_010_000_000)
        struct.pack_into("<d", later, 4, 12.0501)
        path.write_bytes(b"MID360R1" + later + record)
        packets = list(ld.ordered_packets([path]))
        np.testing.assert_allclose([p[0] for p in packets], [2, 2.01])
        self.assertEqual(len(packets), 2)
        cal = self.scan/"mid360/calibration.json"
        data = json.loads(cal.read_text())
        data["serial"] = "unrelated"
        cal.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "do not match"):
            cmd_export_depth(self.opts)
        self.assertFalse((self.dataset/"depth").exists())
        self.assertFalse((self.dataset/"normals").exists())

    def test_mid_only_and_moving_camera(self):
        self.make_stream()
        directory = self.scan/"mid360/continuous"
        # Camera moves 10 m/s to the right. Between measurement and photo, its
        # center moves 0.2 m, or two pixels for a point at 2 m.
        poses = [json.loads(s) for s in (directory/"poses.jsonl").read_text().splitlines()]
        for pose in poses:
            pose["m"][3] = 10*(pose["t"]-12)
        (directory/"poses.jsonl").write_text("\n".join(map(json.dumps, poses)))
        (self.scan/"poses.jsonl").write_text(json.dumps(dict(imageFile="cam/photo.jpg", timestamp=12.06)))
        cmd_export_depth(self.opts)
        depth = read_png(self.dataset/"depth/cam/photo.png")
        self.assertEqual(depth[12, 25], 2000)
        self.assertEqual(depth[12, 27], 0)

    def test_disagreement_and_edges(self):
        phone = np.full((32, 32), 2.0, "f")
        phone[:, 16:] = 0
        mid = np.full_like(phone, 4.0)
        np.testing.assert_array_equal(ld.fuse(phone, mid)[0], phone)
        mid[:, :16] = 2.0
        fused, error = ld.fuse(phone, mid)
        self.assertEqual(error, 0)
        self.assertEqual(fused[16, 16], 0)
        self.assertEqual(fused[16, 25], 4)

    def test_sloped_surface_normals_and_discontinuity(self):
        y, x = np.indices((48, 64))
        k = [80, 90, 32, 24]
        n = np.array([-0.3, 0.2, -1.])
        n /= np.linalg.norm(n)
        z = -2/(n[0]*(x+0.5-k[2])/k[0] + n[1]*(y+0.5-k[3])/k[1] + n[2])
        z[:, 32:] *= 2
        z[16, 16] = 0
        normal, _ = ld.camera_normals(z, k)
        np.testing.assert_allclose(normal[18, 18]/32767.5-1, n, atol=3e-5)
        np.testing.assert_array_equal(normal[18, 31], [32768]*3)
        np.testing.assert_array_equal(normal[16, 16], [32768]*3)

    def test_refined_projection_matches_colmap(self):
        try:
            import pycolmap
        except ImportError:
            self.skipTest("pycolmap not installed")
        for model, extra in (("OPENCV", [0.03, 0.02, -0.01, 0.015]),
                             ("OPENCV_FISHEYE", [0.3, 0.1, -0.02, -0.01]),
                             ("RAD_TAN_THIN_PRISM_FISHEYE", [0.3, 0.1, -0.02, -0.01, 0.12, -0.08,
                                                             0.015, -0.01, 0.01, 0.03, -0.02, 0.01])):
            p = [690, 712, 450, 289, *extra]
            ref = pycolmap.Camera(model=model, width=900, height=600, params=p)
            xy = np.array([[0, 0], [0.65, -0.34], [-0.31, 0.22]])
            actual = ld.distorted_pixels((model, 900, 600, p), *xy.T)
            np.testing.assert_allclose(np.column_stack(actual), ref.img_from_cam(np.column_stack([xy, np.ones(3)])),
                                       rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
