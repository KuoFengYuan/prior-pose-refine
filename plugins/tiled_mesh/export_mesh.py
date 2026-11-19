"""Export tiled metric meshes from LichtFeld and aligned RGB-D captures."""
import argparse
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

# Allow an isolated worker interpreter to import the two bundled modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from views import prepare
from fusion import build


def publish(output, function, args):
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output exists: {output}. Choose a new directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        function(args, stage)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def export(args, directory):
    args.views = directory / "views"
    args.views.mkdir()
    prepare(args, args.views)
    build(args, directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in [("prepare-views", prepare), ("build-mesh", build), ("export-mesh", export)]:
        p = sub.add_parser(name)
        p.set_defaults(function=func)
        p.add_argument("--output", type=Path, required=True, help="New output directory")
        if name != "build-mesh":
            p.add_argument("--dataset", type=Path, help="Undistorted COLMAP dataset")
            p.add_argument("--trained", type=Path, help="LichtFeld training output, including final cameras")
            p.add_argument("--cameras", type=Path, help="Explicit final COLMAP model")
            p.add_argument("--scan", type=Path, help="Original RGB-aligned Splatify-Anything capture")
            p.add_argument("--splat", type=Path, help="Final Gaussian .ply or saved .licht project")
            p.add_argument("--model-version", help=argparse.SUPPRESS)
            p.add_argument("--lichtfeld", default=os.environ.get("LICHTFELD", "LichtFeld-Studio"))
            p.add_argument("--batch-splats", type=int, default=250000, help="Maximum splats per GPU render batch")
            p.add_argument("--width", type=int, default=320, help="Maximum depth image width")
            p.add_argument("--max-depth", type=float, default=20, help="Maximum camera-z in metres")
            p.add_argument("--exclude", type=Path, help="One image name per line to hold out of fusion")
        else:
            p.add_argument("--views", type=Path, required=True)
        if name != "prepare-views":
            p.add_argument("--voxel", type=float, default=0.04, help="Voxel edge in metres")
            p.add_argument("--tile-cells", type=int, default=64, help="Cells per tile edge (8 to 256)")
            p.add_argument("--tile", type=int, nargs=3, metavar=("X", "Y", "Z"))
            p.add_argument("--source", choices=["lidar", "splat", "hybrid"], default="splat")
    args = parser.parse_args(argv)
    for name in ("voxel", "width", "max_depth", "batch_splats"):
        value = getattr(args, name, 1)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be positive and finite")
    if not 8 <= getattr(args, "tile_cells", 64) <= 256:
        parser.error("--tile-cells must be in [8, 256]")
    output = args.output.resolve()
    if any((p / "capture-meta.json").is_file() for p in (output, *output.parents)):
        parser.error("Choose an output outside the original scan")
    for name in ("scan", "dataset", "views", "cameras"):
        path = getattr(args, name, None)
        if path and output.is_relative_to(path.resolve()):
            parser.error(f"Choose an output outside the {name} directory")
    if args.command == "export-mesh":
        if args.source in {"hybrid", "lidar"} and not args.scan:
            parser.error(f"{args.source} requires --scan")
        if args.source in {"hybrid", "splat"} and not args.splat:
            parser.error(f"{args.source} requires --splat")
    publish(args.output, args.function, args)


if __name__ == "__main__":
    main()
