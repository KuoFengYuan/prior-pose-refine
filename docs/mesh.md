# Tiled mesh export for LichtFeld Studio

**English** | [繁體中文](mesh.zh-TW.md)

Export triangle PLY tiles with world coordinates in metres from an existing training run. Each tile can be loaded on its own or placed alongside its neighbors without a transform. The output contains geometry; continue to use the splat for view-dependent appearance.

## Install and use

Use LichtFeld Studio 0.5.5 with plugin API v1 and an external Python environment. For splat or hybrid depths the worker needs CUDA, PyTorch and fVDB Reality Capture with `GaussianSplat3d.render_depths`. Follow the [fVDB installation instructions](https://fvdb-reality-capture.readthedocs.io/latest/installation.html) for your CUDA environment. Sensor-only fusion runs on the CPU.

From the repository, using the worker's Python:

```bash
python -m pip install -r plugins/tiled_mesh/requirements.txt
export MESH_PYTHON="$(command -v python)"
export LICHTFELD="$(command -v LichtFeld-Studio)"  # or source local.env
mkdir -p ~/.lichtfeld/plugins
cp -R plugins/tiled_mesh ~/.lichtfeld/plugins/  # first install
"$LICHTFELD" plugin check tiled_mesh
```

Load the plugin in LichtFeld's plugin manager. In **Mesh Export**, choose **Worker Python** if the app was launched without `MESH_PYTHON`. Select a depth source and voxel size, then **Export Saved Model**. **Export after training** runs the same worker when a training run completes and its bound, saved checkpoint matches the final iteration. Enable saving the final project in LichtFeld. For a saved model from another run, set the model and its matching dataset explicitly. Each launch writes a separate directory with `export.log`, `mesh/mesh.json`, PLY tiles and reusable depth views.

To rebuild part of a saved export in the panel, select its `views/` directory under **Prepared views**, match the original voxel size and cells per tile edge, enter the three indices from `mesh.json`, and click **Rebuild from Prepared Views**. A blank index rebuilds the whole scene.

## Command line

```bash
"$MESH_PYTHON" -I plugins/tiled_mesh/export_mesh.py export-mesh \
  --trained OUT_DIR/train --dataset OUT_DIR/dataset \
  --splat OUT_DIR/train/project.licht --scan SCAN_DIR \
  --source hybrid --voxel 0.04 --output OUT_DIR/mesh
```

- `--source splat` requires a trained Gaussian PLY or a saved training project and matching final cameras. The `.licht` path uses LichtFeld's converter. Use a training scene in the same metric frame as the camera model. A transformed, scaled or edited multi-node project needs a matching export of cameras and splats.
- `--source lidar` uses `--scan` (omit `--splat`). It reads the capture's registered float32 camera-z depth and uint8 confidence by image name, using the per-frame RGB intrinsics scaled to the depth sensor resolution. It assumes co-located RGB-aligned depth and uses confidence level 2. Unregistered sensors require prior calibration and reprojection.
- `--source hybrid` requires both. Measured depth takes priority. A separate fill map rejects splat estimates in pixels with valid measured depth and estimates in free space observed by nearby sensor views.
- `--cameras DIR` selects an explicit final COLMAP model. With `--trained`, the worker looks for `cameras_refined/`, then the undistorted dataset's `sparse/0`. Use final, metric PINHOLE or SIMPLE_PINHOLE cameras. The refined distorted model in `model/` is unsuitable.
- Defaults are a 4 cm voxel, 64 cells per tile edge, 320 pixels maximum depth width, 20 m maximum camera-z and 250,000 splats per rendering batch. Adjust `--max-depth` to the reliable capture range, `--width` to the detail supported by the views, and `--batch-splats` to the available VRAM. Halving voxel size makes a tile cover half the distance along each axis.
- `--exclude FILE` takes one registered image name per line. Excluded views can retain sensor reference depth for evaluation. Their rendered depths and all their measurements are excluded from fusion and multi-view checks.

To change resolution, compare sources or rebuild one tile, reuse the depth cache. The three integers after `--tile` are a tile key in `mesh.json`. Omit them to export the complete scene. Each invocation writes to a new directory.

```bash
"$MESH_PYTHON" -I plugins/tiled_mesh/export_mesh.py build-mesh \
  --views OUT_DIR/mesh/views --source hybrid --voxel 0.04 \
  --tile 0 0 0 --output OUT_DIR/mesh-rebuilt-tile
```

Use `prepare-views` with the same input options to create a standalone cache. An interrupted export leaves previously published directories intact. Sparse coverage can leave openings in glass, thin structures and surfaces observed from a single direction. Inspect those areas before using a mesh for collision or measurement.

## Acceptance checks

1. Export **LiDAR** with a 4 cm voxel. In the output directory, open the PLY files together in Blender or CloudCompare. Walls that cross a tile boundary should align in world coordinates.
2. Select a tile from `mesh.json`. Rebuild it with the same prepared views, depth source, voxel size and cells per edge. Compare the two PLY files with `cmp`. A matching tile is byte-identical in the same worker environment.
3. For a training-end check, select **Export after training** and enable the final project save before starting a training run. On successful completion, check the export log and the new mesh manifest. Stopping a run should leave the export folder unchanged.

To reproduce the benchmark split, extract the image names with a test evaluation from `per_image_metrics.json`, one per line, and pass the file to `prepare-views --exclude`. Reuse the resulting views for all three sources and both voxel sizes.

## How the exporter scales

The worker renders camera-z and opacity in depth-sorted batches, then normalizes the accumulated depth by total opacity. It keeps pixels with opacity of at least 0.5 and support from two additional views with distinct camera positions. Splat preparation keeps model parameters and per-view ordering in host memory. Converting a project to PLY also uses LichtFeld's own model-loading resources; an existing PLY bypasses conversion.

The CPU fusion pass indexes only tiles touched by the depth footprints and their four-voxel truncation bands. Each tile allocates its own grid, shares endpoint samples with adjacent tiles and owns a disjoint set of Marching Cubes cells. A cell needs at least two observations at every corner. The voxel working set scales with the selected tile dimensions; the view index grows with observed coverage. Prepared depths and tile meshes are stored on disk. Compare held-out depth error and coverage together to distinguish missing surfaces from misplaced ones. See the [mesh measurements](benchmark.md#mesh-export).
