"""Regress the camera convention, supported surfaces, tile boundaries and holdout."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/tiled_mesh"))
from export_mesh import main
from fusion import surface
from views import filter_splats, lidar_depth


class MeshTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.views = self.root / "views"
        self.views.mkdir()
        self.frames = []
        for i in range(4):
            w2c = np.eye(4)[:3]
            w2c[0, 3] = 0.02 * i
            self.frames.append(dict(name=f"{i}.jpg", excluded=i == 3, size=[16, 16],
                                    K=[[20, 0, 8], [0, 20, 8], [0, 0, 1]], w2c=w2c.tolist()))
            z = np.full((16, 16), 0.63 if i < 3 else 3, np.float32)
            np.save(self.views / f"lidar-{i}.npy", z)
        (self.views / "views.json").write_text(json.dumps(dict(views=self.frames, sources=["lidar"])))

    def run_mesh(self, dest, cells, *extra):
        path = self.root / dest
        main(["build-mesh", "--views", str(self.views), "--output", str(path), "--source", "lidar",
              "--voxel", "0.025", "--tile-cells", str(cells), *extra])
        manifest = json.loads((path / "mesh.json").read_text())
        self.assertEqual(manifest["fusion_views"], 3)
        triangles = []
        for tile in manifest["tiles"]:
            mesh = PlyData.read(path / tile["file"])
            xyz = np.column_stack([mesh["vertex"][axis] for axis in "xyz"])
            faces = np.stack(mesh["face"]["vertex_indices"])
            points = xyz[faces]
            self.assertTrue((np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])[:, 2] < 0).all())
            for tri in points.round(7):
                triangles.append(tuple(sorted(map(tuple, tri))))
        return manifest, triangles

    def test_plane_seams_winding_and_independent_tile(self):
        a, small = self.run_mesh("small", 8)
        _, large = self.run_mesh("large", 16)
        self.assertEqual(sorted(small), sorted(large))
        self.assertEqual(len(small), len(set(small)))
        tile = a["tiles"][0]
        _, subset = self.run_mesh("part", 8, "--tile", *map(str, tile["tile"]))
        self.assertEqual(len(subset), tile["faces"])
        self.assertEqual((self.root / "part" / tile["file"]).read_bytes(),
                         (self.root / "small" / tile["file"]).read_bytes())

    def test_unknown_voxels_do_not_close_a_hole(self):
        field = (2.5 - np.indices((6, 6, 6))[2]).astype(np.float32)
        valid = np.ones(field.shape, bool)
        valid[3:, :, :] = False
        # Unknown voxels with opposite signs across the boundary must not generate faces.
        field[~valid] = -1
        vertices, _ = surface(field, valid, np.zeros(3), 1)
        self.assertTrue((vertices[:, 0] <= 2).all())
        np.testing.assert_allclose(vertices[:, 2], 2.5)

    def test_lidar_confidence_and_camera_rays(self):
        (self.root / "depth").mkdir()
        depth = np.arange(1, 17, dtype="<f4").reshape(4, 4)
        conf = np.full((4, 4), 2, np.uint8)
        conf[1, 1] = 1
        depth.tofile(self.root / "depth" / "d")
        conf.tofile(self.root / "depth" / "c")
        row = dict(intrinsics=dict(cx=1.5, cy=1.5, fx=2, fy=2, width=4, height=4),
                   depthWidth=4, depthHeight=4, depthFile="d", confidenceFile="c")
        view = dict(size=[4, 4], K=[[2, 0, 3], [0, 2, 2], [0, 0, 1]])
        actual = lidar_depth(self.root, row, view, 100)
        expected = np.zeros_like(depth)
        expected[:, 1:] = depth[:, :-1]
        expected[1, 2] = 0
        np.testing.assert_array_equal(actual, expected)

    def test_pose_optimized_text_cameras_and_missing_final_poses(self):
        trained = self.root / "trained"
        model = trained / "cameras_refined"
        model.mkdir(parents=True)
        (trained / "training_config.json").write_text(json.dumps({"optimization": {"pose_opt": True}}))
        (model / "cameras.txt").write_text("1 PINHOLE 4 4 2 2 2 2\n")
        (model / "images.txt").write_text("1 1 0 0 0 0.25 0 0 1 measured.jpg\n\n")
        (model / "points3D.txt").touch()
        scan = self.root / "scan"
        depth = scan / "depth"
        depth.mkdir(parents=True)
        (scan / "capture-meta.json").write_text(json.dumps({"depthFormat": "float32_raw_little_endian"}))
        rows = []
        for image, value in [("unused.jpg", 9), ("measured.jpg", 3)]:
            rows.append(dict(imageFile=image, depthWidth=4, depthHeight=4, depthFile=image+".bin",
                             confidenceFile="confidence.bin",
                             intrinsics=dict(width=4, height=4, fx=2, fy=2, cx=1.5, cy=1.5)))
            np.full((4, 4), value, "<f4").tofile(depth / (image+".bin"))
        np.full((4, 4), 2, "u1").tofile(depth / "confidence.bin")
        (scan / "poses.jsonl").write_text("\n".join(map(json.dumps, rows)))
        out = self.root / "prepared"
        cmd = ["prepare-views", "--trained", str(trained), "--scan", str(scan), "--output"]
        main([*cmd, str(out)])
        view, = json.loads((out / "views.json").read_text())["views"]
        self.assertEqual(view["w2c"][0][3], 0.25)
        np.testing.assert_array_equal(np.load(out / "lidar-0.npy"), np.full((4, 4), 3))
        model.rename(trained / "unavailable")
        with self.assertRaisesRegex(ValueError, "final world-aligned"):
            main([*cmd, str(self.root / "missing")])
        self.assertFalse((self.root / "missing").exists())

    def test_splat_support_and_lidar_free_space(self):
        for i in range(3):
            z = np.full((16, 16), 0.5, np.float32)
            if i == 0:
                z[8, 8] = 8
            np.save(self.views / f"raw-{i}.npy", z)
        filter_splats(self.frames, self.views, True)
        depth = np.load(self.views / "splat-0.npy")
        self.assertEqual(depth[8, 8], 0)
        self.assertEqual(depth[8, 7], 0.5)
        self.assertFalse(np.load(self.views / "fill-0.npy").any())
        self.assertFalse((self.views / "splat-3.npy").exists())


if __name__ == "__main__":
    unittest.main()
