"""Regression tests for input protection, geometry and COLMAP pose consistency."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prior_pose_refine", ROOT / "prior_pose_refine.py")
ppr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ppr)


def write_prior(path, centers, points=()):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<QiiQQ4d", 1, 1, 1, 640, 480, 500, 500, 320, 240))
    with (path / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(centers)))
        for i, center in enumerate(centers, 1):
            f.write(struct.pack("<I7dI", i, 1, 0, 0, 0, *(-np.asarray(center)), 1))
            f.write(f"frame_{i:05d}.jpg".encode() + b"\0" + struct.pack("<Q", 0))
    with (path / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for i, point in enumerate(points, 1):
            f.write(struct.pack("<Q3d3BdQ", i, *point, 10, 20, 30, 0, 0))


class PipelineInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prior pose refine ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scan = self.root / "scan"
        (self.scan / "images").mkdir(parents=True)
        write_prior(self.scan / "sparse/0", [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.out = self.root / "scan_opt"
        self.out.mkdir()
        self.marker = self.out / "preserve.txt"
        self.marker.write_text("existing output")
        self.calls = self.root / "colmap-called"
        bindir = self.root / "bin"
        bindir.mkdir()
        colmap = bindir / "colmap"
        colmap.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$PPR_TEST_COLMAP_CALLED"\nexit 99\n')
        colmap.chmod(0o755)
        self.env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
                        PPR_TEST_COLMAP_CALLED=str(self.calls))

    def run_pipeline(self, *options, out=None):
        return subprocess.run(["bash", str(ROOT / "prior-pose-refine"), str(self.scan),
                               str(self.out if out is None else out), "--python", sys.executable,
                               "--force", *options], env=self.env, text=True, capture_output=True, timeout=30)

    def assert_rejected(self, options, message, out=None):
        result = self.run_pipeline(*options, out=out)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, result.stderr)
        self.assertNotIn("unbound variable", result.stderr)
        self.assertFalse(self.calls.exists(), "COLMAP must not run for invalid inputs")
        self.assertEqual(self.marker.read_text(), "existing output")
        return result

    def test_overlapping_scan_paths_are_rejected(self):
        for out in (self.scan, self.root, self.scan / "new_output"):
            with self.subTest(out=out):
                self.assert_rejected([], "OUT_DIR overlaps an input", out=out)
                self.assertTrue((self.scan / "sparse/0/images.bin").is_file())
                self.assertFalse((self.scan / "new_output").exists())

    def test_symlink_alias_of_input_is_rejected(self):
        alias = self.root / "scan-alias"
        alias.symlink_to(self.scan, target_is_directory=True)
        self.assert_rejected([], "OUT_DIR overlaps an input", out=alias)
        self.assertTrue((self.scan / "sparse/0/images.bin").is_file())

    def test_symlinked_default_images_are_protected(self):
        external = self.out / "external-images"
        external.mkdir()
        (self.scan / "images").rmdir()
        (self.scan / "images").symlink_to(external, target_is_directory=True)
        self.assert_rejected([], "OUT_DIR overlaps an input")
        self.assertTrue(external.is_dir())

    def test_custom_images_and_prior_are_protected(self):
        external_images = self.out / "images"
        external_images.mkdir()
        self.assert_rejected(["--images", str(external_images)], "OUT_DIR overlaps an input")
        external_prior = self.out / "prior"
        write_prior(external_prior, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.assert_rejected(["--prior-model", str(external_prior)], "OUT_DIR overlaps an input")
        self.assertTrue((external_prior / "images.bin").is_file())

    def test_missing_option_values_are_rejected(self):
        for option in ("--rounds", "--images", "--python", "--min-obs"):
            for suffix in ([], ["--force"], [""]):
                with self.subTest(option=option, suffix=suffix):
                    self.assert_rejected([option, *suffix], f"{option} requires a value")

    def test_invalid_integer_ranges_are_rejected(self):
        cases = [("--rounds", "0"), ("--ba-iterations", "0"), ("--gpu-workers", "0"),
                 ("--max-features", "0"), ("--overlap", "0"), ("--min-obs", "-1"),
                 ("--top-k", "-1"), ("--seq", "-1"), ("--gpu", "-1"),
                 ("--rounds", "1.5"), ("--rounds", "bad"), ("--rounds", "2147483648"),
                 ("--rounds", "99999999999999999999")]
        for option, value in cases:
            with self.subTest(option=option, value=value):
                self.assert_rejected([option, value], f"{option} must be an integer")

    def test_nonpositive_and_nonfinite_alignment_thresholds_are_rejected(self):
        for value in ("0", "-0.1", "nan", "inf", "1e999", "1e-999", "bad"):
            with self.subTest(value=value):
                self.assert_rejected(["--align-max-error", value], "--align-max-error must be a finite number")

    def test_empty_pair_configuration_is_rejected(self):
        self.assert_rejected(["--top-k", "0", "--seq", "0"], "cannot both be zero")

    def test_decimal_integers_and_disabled_restoration_reach_extraction(self):
        result = self.run_pipeline("--rounds", "08", "--gpu-workers", "01", "--min-obs", "0",
                                   "--top-k", "0", "--seq", "1", "--align-max-error", "1e-1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("feature_extraction", result.stdout)
        self.assertIn("feature_extractor", self.calls.read_text())
        self.assertIn("rounds=8", result.stdout)


class GeometryTests(unittest.TestCase):
    def test_similarity_recovers_scale_rotation_and_translation(self):
        rng = np.random.default_rng(7)
        src = rng.normal(size=(30, 3))
        rotation = Rotation.from_euler("xyz", [20, -10, 35], degrees=True).as_matrix()
        dst = 1.7 * (src @ rotation.T) + [1, 2, 3]
        scale, fitted, translation = ppr.umeyama(src, dst)
        np.testing.assert_allclose(scale * (src @ fitted.T) + translation, dst, atol=1e-12)

    def test_planar_centers_are_supported(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        dst = 2 * src + [1, 2, 3]
        scale, rotation, translation = ppr.umeyama(src, dst)
        np.testing.assert_allclose(scale * (src @ rotation.T) + translation, dst, atol=1e-12)

    def test_degenerate_and_nonfinite_centers_are_rejected(self):
        cases = [np.ones((3, 3)), np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
                 np.zeros((2, 3)), np.full((3, 3), np.nan), np.full((3, 3), np.inf)]
        for src in cases:
            with self.subTest(src=src):
                with self.assertRaises(ValueError):
                    ppr.umeyama(src, src)

    def test_collapsed_destination_is_rejected(self):
        with self.assertRaises(ValueError):
            ppr.umeyama(np.eye(3), np.ones((3, 3)))

    def test_rotation_round_trip(self):
        for matrix in Rotation.random(20, random_state=5).as_matrix():
            np.testing.assert_allclose(ppr.quat_to_rot(ppr.rot_to_quat(matrix)), matrix, atol=1e-12)

    def test_invalid_gravity_is_rejected_before_database_access(self):
        frame = {"R": np.eye(3)}
        with patch.object(ppr, "read_images", return_value=[frame]), patch.object(ppr.sqlite3, "connect") as connect:
            for direction in ("0,0,0", "1,2", "nan,0,0", "inf,0,0"):
                with self.subTest(direction=direction):
                    with self.assertRaises(ValueError):
                        ppr.cmd_write_pose_priors(SimpleNamespace(prior="unused", world_down=direction))
            connect.assert_not_called()


class PairSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prior pairs ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prior = self.root / "prior"
        self.out = self.root / "pairs.txt"
        self.args = SimpleNamespace(prior=str(self.prior), out=str(self.out), top_k=1, seq=1,
                                    cell=20, max_points=400000)

    def test_invisible_points_give_actionable_error_without_writing(self):
        for point in ([0, 0, -2], [1000, 1000, 2]):
            with self.subTest(point=point):
                write_prior(self.prior, [[0, 0, 0], [0.1, 0, 0]], [point] * 1000)
                self.out.write_text("preserve pairs")
                with self.assertRaisesRegex(SystemExit, "no prior points are visible.*--matcher sequential"):
                    ppr.cmd_select_pairs(self.args)
                self.assertEqual(self.out.read_text(), "preserve pairs")

    def test_visible_points_keep_pairs_unique_and_add_sequential_neighbors(self):
        write_prior(self.prior, [[0, 0, 0], [0.1, 0, 0], [100, 0, 0]], [[0, 0, 2]] * 1000)
        with contextlib.redirect_stdout(io.StringIO()):
            ppr.cmd_select_pairs(self.args)
        self.assertEqual(self.out.read_text().splitlines(), ["frame_00001.jpg frame_00002.jpg",
                                                           "frame_00002.jpg frame_00003.jpg"])

    def test_too_few_images_are_rejected(self):
        write_prior(self.prior, [[0, 0, 0]], [[0, 0, 2]] * 1000)
        with self.assertRaisesRegex(SystemExit, "at least two prior images"):
            ppr.cmd_select_pairs(self.args)
        self.assertFalse(self.out.exists())

    def test_too_few_points_are_rejected(self):
        write_prior(self.prior, [[0, 0, 0], [0.1, 0, 0]])
        with self.assertRaisesRegex(SystemExit, "only 0 points.*--matcher sequential"):
            ppr.cmd_select_pairs(self.args)

    def test_no_shared_points_and_no_sequential_neighbors_are_rejected(self):
        write_prior(self.prior, [[0, 0, 0], [100, 0, 0]], [[0, 0, 2]] * 1000)
        self.args.seq = 0
        self.out.write_text("preserve pairs")
        with self.assertRaisesRegex(SystemExit, "no matching pairs.*increase --seq"):
            ppr.cmd_select_pairs(self.args)
        self.assertEqual(self.out.read_text(), "preserve pairs")


class HelperArgumentTests(unittest.TestCase):
    def test_invalid_numeric_arguments_fail_before_opening_models(self):
        cases = [("select-pairs", ["--prior", "missing", "--out", "unused", "--cell", "0"]),
                 ("select-pairs", ["--prior", "missing", "--out", "unused", "--top-k", "-1"]),
                 ("select-pairs", ["--prior", "missing", "--out", "unused", "--max-points", "-1"]),
                 ("restore-prior-poses", ["--prior", "missing", "--model", "missing", "--min-obs", "-1"]),
                 ("write-pose-priors", ["--prior", "missing", "--database", "unused", "--std", "nan"]),
                 ("report-agreement", ["--prior", "missing", "--model", "missing", "--flag-cm", "inf"])]
        for command, args in cases:
            with self.subTest(command=command, args=args):
                result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "prior_pose_refine.py"),
                                         command, *args], text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)

    def test_degenerate_similarity_has_clear_cli_error(self):
        with tempfile.TemporaryDirectory(prefix="prior similarity ") as tmp:
            model = Path(tmp)
            write_prior(model, [[0, 0, 0]] * 3)
            result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "prior_pose_refine.py"),
                                     "report-agreement", "--prior", tmp, "--model", tmp],
                                    text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("camera centers are coincident or collinear", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


@unittest.skipUnless(importlib.util.find_spec("pycolmap"), "pycolmap is needed for COLMAP binary round-trip")
class ColmapPoseTests(unittest.TestCase):
    def test_pose_update_round_trips_images_and_frames(self):
        import pycolmap

        with tempfile.TemporaryDirectory(prefix="prior colmap ") as tmp:
            root = Path(tmp)
            text, binary, exported = (root / name for name in ("text", "binary", "exported"))
            for directory in (text, binary, exported):
                directory.mkdir()
            (text / "cameras.txt").write_text("1 PINHOLE 640 480 500 500 320 240\n")
            (text / "rigs.txt").write_text("1 1 CAMERA 1\n")
            (text / "frames.txt").write_text("70 1 1 0 0 0 0 0 0 1 CAMERA 1 7\n"
                                            "420 1 1 0 0 0 1 0 0 1 CAMERA 1 42\n")
            (text / "images.txt").write_text("7 1 0 0 0 0 0 0 1 first.jpg\n\n"
                                            "42 1 0 0 0 1 0 0 1 second.jpg\n\n")
            (text / "points3D.txt").write_text("")
            pycolmap.Reconstruction(str(text)).write_binary(str(binary))
            preserved = {name: (binary / name).read_bytes() for name in ("cameras.bin", "rigs.bin", "points3D.bin")}
            q = np.array([np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0])
            t = np.array([1.5, -2, 3])
            ppr.write_poses(binary, {42: (q, t)})
            updated = pycolmap.Reconstruction(str(binary))
            updated.write_text(str(exported))
            round_trip = pycolmap.Reconstruction(str(exported))
            expected_rotation = Rotation.from_euler("y", 45, degrees=True).as_matrix()
            for reconstruction in (updated, round_trip):
                pose = reconstruction.images[42].cam_from_world()
                np.testing.assert_allclose(pose.rotation.matrix(), expected_rotation, atol=1e-12)
                np.testing.assert_allclose(pose.translation, t, atol=1e-12)
                np.testing.assert_allclose(reconstruction.images[7].cam_from_world().translation, [0, 0, 0])
            poses = {f["image_id"]: f for f in ppr.read_images(binary)}
            np.testing.assert_allclose(poses[42]["q"], q)
            np.testing.assert_allclose(poses[42]["t"], t)
            for name, content in preserved.items():
                self.assertEqual((binary / name).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
