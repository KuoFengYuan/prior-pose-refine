# How it works

**English** | [繁體中文](method.zh-TW.md)

This page describes what `prior-pose-refine` does in the default `fast` mode, and why. The *prior* means the poses, intrinsics and points in `SCAN_DIR/sparse/0`. The numbers come from the test scans in [benchmark.md](benchmark.md).

## Input checks

Before removing an existing output or starting a stage, the pipeline checks option values and resolves input and output paths, including symbolic links. The output must not equal, contain or lie inside the scan, image folder or prior model. It also cannot contain the pipeline directory. These checks apply with `--force`, so an overwrite cannot remove the source data or the program. Counts and iteration limits use the ranges in the [README](../README.md#7-options); the alignment threshold must be finite and positive.

## Steps

| Step | Command | What it does |
|---|---|---|
| 1 | `prior_pose_refine.py list-images`, `export-centers` | lists the images that have an prior pose and writes their camera centers |
| 2 | `colmap feature_extractor` | extracts SIFT features. All images share one camera; 3 workers run on one GPU. Extraction and matching always use an OPENCV camera, because the two-view verification of step 4 uses the database camera |
| 3 | `prior_pose_refine.py select-pairs` | chooses the matching pairs from the prior poses and points (see below) |
| 4 | `colmap matches_importer` | matches only the chosen pairs, without guided matching |
| 5 | `prior_pose_refine.py init-model` | builds the initial model: prior poses, median prior intrinsics, the `--camera-model` camera started as a distortion-free pinhole, no 3D points. It switches the database camera to the same model, because point_triangulator requires the two to agree |
| 6 | `colmap point_triangulator` → `bundle_adjuster`, 3 rounds | triangulation keeps the poses fixed and turns matches into 3D points (2 internal refinements); bundle adjustment refines poses, focal length and distortion together (at most 15 iterations) |
| 7 | `colmap model_aligner` | aligns the model back to the prior frame with a similarity fitted to the camera centers |
| 8 | `prior_pose_refine.py restore-prior-poses` | gives images with fewer than `--min-obs` observations (default 20) their prior pose (see below) |
| 9 | `colmap image_undistorter` | undistorts the images and writes the training dataset |
| 10 | `prior_pose_refine.py merge-points` | maps the prior points into the model and adds them to the dataset's initial point cloud (`--init-points merge`, the default) |

## Choosing matching pairs (step 3)

Pair selection is the main speed-up of the pipeline:

1. Project the points into every camera and keep those in front of the camera and inside the image.
2. Divide the image into 20 px cells and keep, in each cell, only the points near the closest one, so that points behind a wall do not count as visible.
3. Count the points each pair of cameras sees in common, and pair every image with its 25 most covisible images plus its 3 neighbours on each side in capture order.

On the first test scan this gave 12,706 pairs, 96% of which passed geometric verification, and matching took 18 s instead of 164 s with a vocabulary tree. COLMAP's built-in `spatial_matcher` looks only at camera positions, not at viewing directions, and lost 0.5 dB.

At least two prior images and 1,000 prior points are required for this selector. If no point is visible in any camera, it stops with a coordinate-convention and units hint instead of building an empty visibility matrix. If the settings select no pairs, it stops before writing the pair list. Use `--matcher sequential` for a prior without usable point coverage; this is an explicit choice, not an automatic fallback.

## Camera model (steps 5 and 6)

- The distortion of the iPhone main lens does not simply grow towards the edge. On the test scan, compared with a pinhole camera, it is +0.6% at a radius of 200 px, peaks at +2.4% at 600 px, and falls back to +1.4% in the corners.
- OPENCV has only two radial terms (k1, k2). It cannot follow a curve that rises and then falls, and leaves a systematic error that oscillates with the radius: +1.3 px near the centre, −2.1 px in the corners. That error is about as large as the whole reprojection error, and it blurs the trained images.
- RAD_TAN_THIN_PRISM_FISHEYE has 6 radial terms, plus 2 tangential and 4 thin-prism terms. The systematic error drops below 0.1 px, the reprojection error from 1.15 px to 0.82 px, and PSNR rises by 2.4 dB.
- "FISHEYE" in the name means that the distortion polynomial is written in the ray angle θ; the lens does not have to be a fisheye. A normal lens images at f·tan θ, so filling the coefficients with the Taylor series of tan θ (1/3, 2/15, 17/315, …) gives a distortion-free pinhole camera (within 0.02 px). `init-model` starts from the prior intrinsics this way.
- The fitted high-order coefficients are large and alternate in sign (for example −9.9, 44.9, −75.8, 44.9). They cancel within the image; this is normal as long as the reprojection error is normal.
- With this model the centre focal length is 1366.8, within 0.35% of ARKit's 1371.6. With OPENCV the fitted fx comes out about 1.3% larger; that is the model bending to fit the distortion curve, not an error in ARKit's focal length.

## Three rounds (step 6)

The first triangulation uses the prior poses, so some matches have large errors and are dropped as outliers. Each round of bundle adjustment corrects the poses and intrinsics, and the next triangulation keeps more correct points: the observations grew from 1.4 million in the first round to 1.99 million. Each round cuts the remaining distance to the converged result to about a third: 2 rounds lose about 0.2 dB, and a 4th round changes nothing. On the test scan fx converged as 1371.6 → 1366.3 → 1366.4 → 1366.8.

## Iteration caps (step 6)

These two caps are the other main speed-up:

- COLMAP's default bundle adjustment runs 30 to 40 iterations before it stops, but the cost and the intrinsics stop changing after about 10. With a cap of 15 the poses are identical to a full run, and each round saves about 15 s.
- point_triangulator runs up to 5 more points-only adjustments internally, which the following bundle_adjuster redoes anyway. Capping them at 2 saves about 13 s per round and changes the final poses by 0.025 px.

## Alignment (step 7)

Bundle adjustment does not fix the overall scale and orientation, so after three rounds the model has drifted relative to the prior: by 1.3% in scale and 0.9° in rotation on the test scan. After the alignment, the output lines up with the prior poses and points again.

The helper's similarity fits used for pose restoration, agreement reporting and point merging require at least three matching finite camera centers. Coincident or collinear centers do not determine a unique rotation and are rejected before poses or points are written. Planar trajectories remain valid. In global mode, the gravity prior must also have three finite components and a nonzero, stable direction before the database is changed.

## Step 8: images with too few observations

Bundle adjustment moves the whole model, but an image without any 3D-point observations is not constrained at all, so bundle adjustment never moves it. The alignment then transforms that image together with the rest of the model, and it ends up off by the model's drift. On scan_20261103_223455_C9BFD7, frames 00299–00304 are textureless and blurred and have no observations; the model drifted 21 cm before the alignment, and these 6 images ended up about 25 cm from their prior poses. Such images have no image evidence to correct their pose, so they keep their prior pose, mapped into the model with the similarity fitted to the other images.

- Why 20: on the 7 earlier scans, every image had at least 96 observations, so normal images are never reset.
- Only the observation count decides, never the size of the correction. A large correction is not necessarily wrong: on another scan ARKit drifted by up to 25 cm in three stretches, and bundle adjustment corrected them. Large corrections are only listed in the report.
