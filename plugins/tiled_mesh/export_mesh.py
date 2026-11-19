"""Export tiled metric meshes and baked photographic textures from LichtFeld."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

# Allow an isolated worker interpreter to import the bundled modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from views import prepare
from fusion import build


def run_stages(args, out):
    timings = {}

    def stage(name, function, directory):
        start = time.perf_counter()
        function(args, directory)
        timings[name] = time.perf_counter() - start

    if args.command in {"prepare-views", "export-mesh"}:
        args.views = out if args.command == "prepare-views" else out / "views"
        args.views.mkdir(exist_ok=True)
        stage("prepare_views", prepare, args.views)
    if args.command in {"build-mesh", "export-mesh"}:
        stage("build_mesh", build, out)
        args.mesh = out
    if args.command == "bake-textures" or getattr(args, "bake_textures", False):
        from textures import bake
        stage("bake_textures", bake, out)
    (out / "timings.json").write_text(json.dumps(timings, indent=2))
    print("Wall seconds: " + ", ".join(f"{k}={v:.2f}" for k, v in timings.items()), flush=True)


def publish(output, args):
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output exists: {output}. Choose a new directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        run_stages(args, stage)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-views", "build-mesh", "export-mesh", "bake-textures"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True, help="New output directory")
        p.add_argument("--images", type=Path, help="Undistorted RGB photo directory for baking")
        if name in {"prepare-views", "export-mesh"}:
            p.add_argument("--dataset", type=Path, help="Undistorted COLMAP dataset")
            p.add_argument("--trained", type=Path, help="LichtFeld training output, including final cameras")
            p.add_argument("--cameras", type=Path, help="Explicit final COLMAP model")
            p.add_argument("--scan", type=Path, help="Original RGB-aligned Splatify-Anything capture")
            p.add_argument("--merged-depths", type=Path,
                           help="Registered 16-bit camera-z PNGs in millimeters, for sensor hole filling")
            p.add_argument("--splat", type=Path, help="Final Gaussian .ply or saved .licht project")
            p.add_argument("--model-version", help=argparse.SUPPRESS)
            p.add_argument("--lichtfeld", default=os.environ.get("LICHTFELD", "LichtFeld-Studio"))
            p.add_argument("--batch-splats", type=int, default=250000, help="Maximum splats per GPU render batch")
            p.add_argument("--width", type=int, default=320, help="Maximum depth image width")
            p.add_argument("--max-depth", type=float, default=20, help="Maximum camera-z in metres")
            p.add_argument("--exclude", type=Path, help="One image name per line to hold out of fusion and baking")
        else:
            p.add_argument("--views", type=Path, required=name == "build-mesh", help="Prepared depth views")
        if name in {"build-mesh", "export-mesh"}:
            p.add_argument("--voxel", type=float, default=0.02, help="Voxel edge in metres")
            p.add_argument("--tile-cells", type=int, default=64, help="Cells per tile edge (8 to 256)")
            p.add_argument("--source", choices=["lidar", "splat", "hybrid", "sensor-priority", "sensor-priority-fill"],
                           help="Defaults to sensor-priority with merged depths, or the available single source")
            p.add_argument("--bake-textures", action="store_true", help="Also export textured GLB tiles")
        if name == "bake-textures":
            p.add_argument("--mesh", type=Path, required=True, help="Existing mesh directory containing mesh.json")
        if name != "prepare-views":
            p.add_argument("--tile", type=int, nargs=3, metavar=("X", "Y", "Z"))
            p.add_argument("--atlas-size", type=int, choices=[512, 1024, 2048, 4096], default=2048,
                           help="Maximum width and height of each texture page")
            p.add_argument("--texture-downsample", type=int, choices=[1, 2, 4], default=1,
                           help="Photograph downsampling factor for texture storage")
            p.add_argument("--bake-views", type=int, default=8, help="Maximum selected photographs per tile per side")
            p.add_argument("--two-sided-textures", action="store_true",
                           help="Also color unsupported back faces with a tighter depth check")
    args = parser.parse_args(argv)
    for name in ("voxel", "width", "max_depth", "batch_splats", "bake_views"):
        value = getattr(args, name, 1)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be positive and finite")
    if not 8 <= getattr(args, "tile_cells", 64) <= 256:
        parser.error("--tile-cells must be in [8, 256]")
    output = args.output.resolve()
    if any((p / "capture-meta.json").is_file() for p in (output, *output.parents)):
        parser.error("Choose an output outside the original scan")
    for name in ("scan", "dataset", "views", "cameras", "images", "mesh", "merged_depths"):
        path = getattr(args, name, None)
        if path and output.is_relative_to(path.resolve()):
            parser.error(f"Choose an output outside the {name} directory")
    if getattr(args, "merged_depths", None):
        if not args.scan:
            parser.error("--merged-depths requires --scan")
        if not args.merged_depths.is_dir():
            parser.error("--merged-depths must be an existing directory")
    if args.command == "export-mesh":
        if args.source is None:
            args.source = "lidar" if args.scan and not args.splat else "splat"
            if args.merged_depths:
                args.source = "sensor-priority"
        if args.source in {"sensor-priority", "sensor-priority-fill"} and not args.merged_depths:
            parser.error(f"{args.source} requires --merged-depths")
        if args.source in {"hybrid", "lidar", "sensor-priority", "sensor-priority-fill"} and not args.scan:
            parser.error(f"{args.source} requires --scan")
        if args.source in {"hybrid", "splat", "sensor-priority-fill"} and not args.splat:
            parser.error(f"{args.source} requires --splat")
    publish(args.output, args)


if __name__ == "__main__":
    main()
