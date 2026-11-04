# prior-pose-refine

**English** | [繁體中文](README.zh-TW.md)

Refine approximate camera poses, together with the focal length and lens distortion, with COLMAP, and get a dataset that is ready for 3D Gaussian Splatting (3DGS) training. The poses and the 3D points can come from any source that provides both: ARKit with LiDAR, a LiDAR-inertial or visual SLAM system, a motion-capture rig, and so on.

Tracked poses are close but not exact, and trackers such as ARKit describe the camera as a pinhole with no lens distortion, so splats trained on their data come out doubled and blurry. prior-pose-refine starts from the prior poses, uses the 3D points to choose which images to match, bundle-adjusts the poses together with a camera model that can describe a phone lens, and aligns the result back to the prior model's metric frame. It is developed and tested on iPhone ARKit scans; see [where the prior can come from](#31-where-the-poses-and-points-can-come-from).

## Results

An indoor office scan with iPhone ARKit poses and Livox Mid-360 points (729 posed images), trained with LichtFeld Studio for 30,000 iterations at 1M Gaussians, with every 8th image held out for testing:

| Poses used for training | COLMAP time | Test PSNR |
|---|---|---|
| ARKit poses as exported by the capture app | — | 23.46 dB |
| COLMAP from scratch (vocab tree + global_mapper) | 7:36 | 28.23 dB |
| prior-pose-refine with `--camera-model OPENCV` | 3:03 | 28.41 dB |
| **prior-pose-refine (default)** | **3:30** | **30.84 dB** |

A second scan (1,505 images, iPhone LiDAR) agrees: 30.55 dB against 28.28 dB for COLMAP from scratch and 23.14 dB for the exported ARKit poses, with a COLMAP time of 7:30 against 17:47. How the scans were captured and evaluated is described in [docs/benchmark.md](docs/benchmark.md#about-this-data).

Most of the gain comes from the camera model: the default RAD_TAN_THIN_PRISM_FISHEYE model is 2.4 dB better than OPENCV on the same pipeline. Times were measured on a Core Ultra 7 265K with an RTX 5070 (12 GB) and an idle GPU. The full data and comparison images are in [docs/benchmark.md](docs/benchmark.md).

## Contents

1. [Requirements](#1-requirements)
2. [Setup](#2-setup)
3. [Input](#3-input)
4. [Quick start](#4-quick-start)
5. [Output](#5-output)
6. [Training](#6-training)
7. [Options](#7-options)
8. [Checking the result](#8-checking-the-result)
9. [When to use `--mode global`](#9-when-to-use---mode-global)
10. [Troubleshooting](#10-troubleshooting)
11. [How it works](#11-how-it-works)
12. [Files](#12-files)
13. [License](#13-license)

## 1. Requirements

| Item | Notes |
|---|---|
| OS | Linux with bash 5 or later |
| COLMAP | 4.x built with CUDA (tested with 4.3.0.dev), on `PATH` |
| Python | Python 3 with numpy and scipy |
| GPU | NVIDIA GPU, used for SIFT extraction, matching and bundle adjustment |
| Training | Any 3DGS trainer that reads a COLMAP dataset; tested with LichtFeld Studio |

## 2. Setup

```bash
git clone https://github.com/KuoFengYuan/prior-pose-refine.git
cd prior-pose-refine
pip install -r requirements.txt     # or point PYTHON at an existing Python with numpy and scipy
cp local.env.example local.env      # then fill in the paths for this machine
source local.env                    # run this in every new shell
```

Paths differ from machine to machine, so they live in `local.env`, which is not committed. `prior-pose-refine` needs only the first two variables:

| Variable | Meaning |
|---|---|
| `PATH` | must include the directory that holds `colmap` |
| `PYTHON` | the Python to use (default: `python3`); `--python` overrides it |
| `LICHTFELD`, `SCAN_ROOT`, `EXP_ROOT` | optional shortcuts for the trainer, your scans and your experiments |

## 3. Input

The input, `SCAN_DIR`, is a folder of images plus a COLMAP model that holds the prior poses, intrinsics and points:

```
SCAN_DIR/
├── images/              images, e.g. frame_00001.jpg …
└── sparse/0/            COLMAP binary model
    ├── images.bin       prior pose of each image (world-to-camera, COLMAP convention)
    ├── cameras.bin      prior intrinsics (one PINHOLE camera per image is fine)
    └── points3D.bin     3D points (for example LiDAR), in the same frame as the poses
```

Requirements:

- **Poses and points are in metres.** Several thresholds are in metres (pair selection, the alignment inlier threshold, the report flags), so a model in another unit, such as a scale-free SfM reconstruction, does not work as is.
- **Poses use the COLMAP convention:** world-to-camera, with the camera's x axis to the right, y down and z forward. Many trackers, ARKit included, report camera-to-world poses with y up and z backward; convert those before writing `images.bin`.
- **Poses are roughly right.** The default `fast` mode makes local corrections; on the test scans it fixed ARKit drift of up to 25 cm. For larger errors, use `--mode global` (section 9).
- **Only images in `images.bin` are used.** Images without a pose are ignored.
- **All images come from one camera.** The tool fits one shared camera (RAD_TAN_THIN_PRISM_FISHEYE by default), starting from the median prior intrinsics.
- **The prior intrinsics belong to the image size recorded in `cameras.bin`.** If the images in `--images` have another size (for example, downscaled), the intrinsics are scaled to match.
- **File names sort in capture order.** Each image is also matched with its 3 neighbours on each side in name order.
- **`points3D.bin` should cover what the cameras see.** The points are used to choose the matching pairs and as part of the initial point cloud for training. Without points, add `--matcher sequential`; the initial point cloud then uses the SfM points only.

In this documentation and in the report, the poses, intrinsics and points of this model are called the *prior*.

### 3.1 Where the poses and points can come from

| Source | Status |
|---|---|
| ARKit poses with iPhone LiDAR points | tested |
| ARKit poses with points from a separate LiDAR (Livox Mid-360) | tested |
| Poses already refined on the device (the capture app's own refinement) | tested |
| LiDAR-inertial odometry (for example FAST-LIO2), visual or visual-inertial SLAM, motion capture, robot odometry, RTK-GNSS with a separate point cloud | should work if the requirements above hold; not tested yet |

Scans exported by [Splatify-Anything](https://github.com/KuoFengYuan/splatify-anything) already have this layout. For another source, write its poses, intrinsics and points as a COLMAP model, for example in COLMAP's text format followed by `colmap model_converter --output_type BIN`.

## 4. Quick start

```bash
source local.env
./prior-pose-refine path/to/scan path/to/scan_opt
```

`prior-pose-refine` prints each stage and its time:

```
prior-pose-refine: mode=fast matcher=covis rounds=3 camera=RAD_TAN_THIN_PRISM_FISHEYE init=merge
  scan:   …/scan
  prior:  …/scan/sparse/0
  images: …/scan/images
  out:    …/scan_opt
list_images           0.1 s
export_centers        0.1 s
feature_extraction    8.7 s
select_pairs          1.7 s
matching              13.8 s
init_model            0.2 s
triangulate_r0        11.4 s
bundle_adjust_r0      10.0 s
…
align                 0.4 s
restore_prior_poses   0.1 s
undistort             5.5 s
merge_points          0.3 s
== prior-pose-refine report
…
dataset ready: …/scan_opt/dataset
```

- On an RTX 5070, a 390-image scan took 1:46 and an 805-image scan 3:11.
- If the output folder already exists, `prior-pose-refine` stops instead of overwriting it; add `--force` to overwrite.
- Before writing output, the tool validates option values and resolves symbolic links. `OUT_DIR` must be separate from `SCAN_DIR`, the image folder and the prior model: equal, parent and child paths are rejected, including with `--force`. An output folder containing the pipeline itself is also rejected.
- Each run writes 1.2 to 2 GB, so check the free disk space first.

## 5. Output

```
OUT_DIR/
├── dataset/                 ← train on this
│   ├── images/              undistorted images (PINHOLE)
│   └── sparse/0/            cameras.bin, images.bin, points3D.bin (SfM points + prior points)
├── model/                   the refined model before undistortion, aligned to the prior model (with distortion parameters)
├── report.txt               summary of the result, see section 8
├── timing.tsv               time of each stage
└── work/                    intermediate files, safe to delete
    ├── database.db          COLMAP database (features, matches)
    ├── image_list.txt       images that have a prior pose
    ├── prior_centers.txt    prior camera centers, for model_aligner
    ├── pairs.txt            the selected matching pairs
    ├── init/                initial model with the prior poses
    ├── tri*/ ba*/           triangulation and bundle-adjustment result of each round
    └── logs/                full log of each stage
```

- `dataset/` is in the prior model's coordinate frame, in metres, so it lines up with the prior points and poses.
- `model/` keeps the original resolution and the distortion parameters. Use it to undistort differently or to export to other software. Not every program supports RAD_TAN_THIN_PRISM_FISHEYE; if you need an OPENCV camera (for example, to train directly on distorted images), rerun with `--camera-model OPENCV`, at a cost of about 2.4 dB.
- `work/` takes about 0.6 GB and can be deleted once the result looks right.

## 6. Training

`dataset/` is a standard COLMAP dataset with PINHOLE cameras. With LichtFeld Studio:

```bash
"$LICHTFELD" -d OUT_DIR/dataset --max-cap 1000000
```

- `--max-cap 1000000` caps the number of Gaussians at 1M, the setting used for all results above; adjust it to your GPU memory (LichtFeld's default is 5M).
- **Initial point cloud:** `dataset/sparse/0/points3D.bin` holds the SfM points plus the LiDAR points, mapped into the model with the similarity between the camera centers. On two scans, compared with SfM points only, PSNR was unchanged (−0.03 / +0.02 dB), LPIPS improved, and floaters in the room dropped by 41% / 67%. LiDAR points alone lost about 1.4 dB. Add `--init-points sfm` for SfM points only.

To measure quality on held-out images:

```bash
"$LICHTFELD" --headless -d OUT_DIR/dataset -o OUT_DIR/train \
    --max-cap 1000000 --eval --test-every 8
```

The metrics are written to `OUT_DIR/train/metrics_report.txt`, and ground-truth/render pairs for each test image to `eval_step_*/`.

## 7. Options

```
prior-pose-refine SCAN_DIR OUT_DIR [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--images DIR` | `SCAN_DIR/images` | image folder |
| `--prior-model DIR` | `SCAN_DIR/sparse/0` | prior COLMAP model (poses, intrinsics, points) |
| `--mode fast\|global` | `fast` | `fast`: bundle-adjust starting from the prior poses (recommended). `global`: COLMAP global_mapper with the gravity direction, then alignment; see section 9 |
| `--rounds N` | `3` | triangulation + bundle-adjustment rounds (fast mode only). 2 rounds are about 60 s faster but lose about 0.2 dB; a 4th round does not help |
| `--camera-model M` | fast: `RAD_TAN_THIN_PRISM_FISHEYE`<br>global: `OPENCV` | camera model for bundle adjustment and undistortion. `OPENCV_FISHEYE` has fewer parameters and is more conservative (about 0.08 dB lower); `OPENCV` is about 2.4 dB lower. `--mode global` supports only `OPENCV` |
| `--ba-iterations N` | `15` | iteration cap of each bundle_adjuster run; results stop changing after 15 |
| `--init-points sfm\|merge` | `merge` (`sfm` when the prior model has no `points3D.bin`) | initial point cloud of the dataset. `merge` = SfM points plus the points of the prior model; `sfm` = SfM points only. See section 6 |
| `--matcher covis\|sequential` | `covis` | `covis`: choose matching pairs from the prior poses and point covisibility. `sequential`: COLMAP matching in capture order, for scans without points |
| `--top-k N` | `25` | covisible partners per image |
| `--seq N` | `3` | neighbours in capture order added per image |
| `--overlap N` | `20` | neighbours matched on each side in `sequential` mode |
| `--gpu N` | `0` | GPU index |
| `--gpu-workers N` | `3` | SIFT extractors running on that GPU |
| `--max-features N` | `4096` | SIFT features per image |
| `--align-max-error M` | `0.1` | inlier threshold of the alignment to the prior poses, in metres |
| `--min-obs N` | `20` | images with fewer than N observations keep their prior pose; see [how it works](docs/method.md#step-8-images-with-too-few-observations). 0 turns this off |
| `--python PATH` | `$PYTHON` or `python3` | Python with numpy and scipy |
| `--force` | | overwrite an existing output folder; overlapping input paths are always rejected |

The defaults are the best settings found in the experiments; change them only to solve a problem.

`--rounds`, `--ba-iterations`, `--overlap`, `--gpu-workers` and `--max-features` must be positive integers. `--gpu`, `--top-k`, `--seq` and `--min-obs` accept zero; all integer options are limited to 2147483647. With `--matcher covis`, `--top-k` and `--seq` cannot both be zero. `--align-max-error` must be finite and greater than zero. Missing values and invalid ranges stop the run before an existing output is removed.

## 8. Checking the result

Read `OUT_DIR/report.txt`. This is the report of a 390-image scan rerun on 2026-11-04, as a reference for normal values:

```
Registered images: 390            ← should equal the number of prior images
Points: 120528
Mean reprojection error: 0.690695px
refined camera (with distortion):
camera 1: RAD_TAN_THIN_PRISM_FISHEYE 1920x1440 1363.97 1363.82 963.985 724.283 0.622796 -0.103451 -8.14841 37.5694 -62.5652 36.0914 …
agreement with the prior poses:
390/390 prior frames in the model | distance to the prior camera centers: median 1.8 cm, max 3.6 cm | after a similarity
(scale 1.0000, rotation 0.00 deg): median 1.8 cm, p95 3.3 cm
frames with fewer than 20 observations (prior pose kept): 0
frames more than 10 cm from their prior pose after the similarity: 0
```

The rerun took 103.4 s on an RTX 5070, with GPU utilization recorded from an idle start. These are reference values, not exact regression targets: GPU nondeterminism can change the point count by a few hundred and reprojection error by about 0.001 px. Input checks do not establish image quality; improvements in PSNR still require 3DGS evaluation.

| Field | Normal | When it is not |
|---|---|---|
| Registered images | equals the number of prior images | far fewer: some images lack matches; check `work/logs/matching.log` |
| Mean reprojection error | about 0.7–0.8 px (about 1.15 px with `--camera-model OPENCV`) | 1.5 px or more: poses or intrinsics did not converge; try `--rounds 4`, `--camera-model OPENCV_FISHEYE` or `--mode global` |
| fx | within 1% of the prior fx (for ARKit) | far off: the camera model may not suit the lens (for example, an ultra-wide lens), or the prior intrinsics are wrong |
| Distortion coefficients | large high-order terms with alternating signs are normal; they cancel within the image | judge by the reprojection error, not the coefficients |
| Distance to the prior poses after the similarity | median 1–2 cm (for ARKit) | 10 cm or more: tracking may have failed; try `--mode global` |
| `frames with fewer than 20 observations` | 0 | the listed images have too few features (usually blur or blank walls) and keep their prior pose. A few are fine; many mean that part of the capture is poor |
| `frames more than 10 cm from their prior pose` | 0 or a few | the listed images were moved more than 10 cm, with their observation counts in brackets. Many observations (hundreds) usually means drift in the prior poses was corrected; with few, look at the images |

## 9. When to use `--mode global`

The default `fast` mode assumes the prior poses are roughly right and only need local correction. Use `--mode global` when:

- Tracking jumped, relocalized or drifted over a large range during the capture.
- The report shows a median distance to the prior poses above 10 cm, or a high reprojection error.

`global` mode does not use the prior poses themselves, only:

1. The gravity direction, to help global_mapper estimate rotations. It is derived from the cameras' image-down direction, which is right when the camera is held roughly upright (handheld or on a ground robot), but not for, say, a downward-looking drone camera.
2. The prior camera centers, to align the model back to the metric frame at the end.

`global` mode supports only the OPENCV camera model. On the test scan its quality matched COLMAP from scratch (28.26 dB), about 2.6 dB below `fast`, and it took longer (global_mapper alone ran for about 245 s).

```bash
./prior-pose-refine SCAN_DIR OUT_DIR --mode global
```

## 10. Troubleshooting

| Message or symptom | What to do |
|---|---|
| `lacks numpy/scipy` | set `PYTHON` in `local.env`, pass `--python PATH`, or `pip install -r requirements.txt` |
| `colmap is not on PATH` | add the COLMAP directory to `PATH` in `local.env`, then `source local.env` |
| `OUT_DIR exists` | choose another output folder, or add `--force` |
| `OUT_DIR overlaps an input` or `OUT_DIR contains the pipeline directory` | choose a separate output folder, conventionally `<SCAN_DIR>_opt`; symbolic links are resolved, and `--force` does not bypass these checks |
| `requires a value`, `must be an integer` or `must be a finite number` | provide the option value and use the ranges in section 7; `--min-obs 0` remains valid |
| `--top-k and --seq cannot both be zero` | enable covisible partners or sequential neighbours, or use `--matcher sequential` |
| `select-pairs: only N points` | the scan has no LiDAR points; add `--matcher sequential` |
| `no prior points are visible` or `no matching pairs were selected` | check the prior coordinate convention, metre units and point coverage; increase `--seq` if needed, or use `--matcher sequential` |
| `camera centers are coincident or collinear` | a similarity cannot determine rotation from these centers; use a scan with camera motion in at least two independent directions. Planar camera paths are supported |
| `world down` has invalid components or no stable direction | check the prior rotations; when calling `write-pose-priors` directly, pass a finite nonzero `--world-down x,y,z` |
| `prior frames are not in the database` | images are missing from the `--images` folder, or their names do not match `images.bin` |
| `--mode global supports only --camera-model OPENCV` | do not pass another camera model in `global` mode |
| `Points: 0` in the report | triangulation produced no points. If you changed the pipeline: extraction and matching must use an OPENCV camera (otherwise two-view verification fails for every pair), and point_triangulator needs `--Mapper.max_extra_param 1e9` (otherwise the large high-order coefficients are treated as invalid) |
| a stage prints `FAILED` | the last 20 lines of its log are printed; the full log is `OUT_DIR/work/logs/<stage>.log` |
| GPU out of memory | lower `--gpu-workers`, for example to 1 |
| disk full | put `OUT_DIR` on a disk with more space; delete `work/` once the result is checked |

## 11. How it works

In short: SIFT features → matching pairs chosen from the prior poses and point covisibility → three rounds of triangulation and bundle adjustment with a RAD_TAN_THIN_PRISM_FISHEYE camera, starting from the prior poses → alignment to the prior model's metric frame → undistortion. Each step and the reasons behind it are in [docs/method.md](docs/method.md).

## 12. Files

| File | Contents |
|---|---|
| `prior-pose-refine` | the pipeline |
| `prior_pose_refine.py` | helper subcommands: read the prior model, choose matching pairs, build the initial model, write priors, restore the prior pose of images with too few observations, check the alignment, merge the initial point cloud. Each runs on its own; see `--help` |
| `requirements.txt` | Python packages |
| `tests/test_prior_pose_refine.py` | regression tests for input protection, numeric ranges, geometry, pair selection and COLMAP pose consistency |
| `.github/workflows/checks.yml` | GitHub Actions checks on Python 3.11, 3.12 and 3.13, including the pycolmap round-trip |
| `local.env.example` | template for the machine-specific paths; copy it to `local.env` (not committed) |
| `docs/method.md` | how the pipeline works and why |
| `docs/benchmark.md` | full experiment data and comparison images |
| `AGENTS.md` | contributor working agreement and delivery workflow |
| `LICENSE` | MIT licence |

Run the regression tests with the same Python as the pipeline:

```bash
source local.env
"$PYTHON" -I -B tests/test_prior_pose_refine.py -v
```

The tests use numpy, scipy and the standard library. The COLMAP binary round-trip test also uses pycolmap and is skipped if it is unavailable; pycolmap is not a pipeline dependency. Synthetic tests do not replace a full scan run or 3DGS quality evaluation.

GitHub Actions installs pycolmap and runs the regression suite on pushes to `main` and task branches, and on pull requests. It checks CPU behavior and binary model compatibility. Full CUDA pipeline runs and 3DGS evaluation use local scans and remain separate validation steps.

## 13. License

MIT; see [LICENSE](LICENSE).
