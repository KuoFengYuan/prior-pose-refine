"""LichtFeld Studio controls for the external tiled mesh worker."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import lichtfeld as lf
from lfs_plugins.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PropertyGroup, StringProperty

_hooks = None
_job = None
_status = ""
_log = None


def lichtfeld_binary():
    configured = os.environ.get("LICHTFELD") or shutil.which("LichtFeld-Studio")
    if configured:
        return configured
    # In embedded Python sys.executable can be a standalone interpreter.
    process = Path("/proc/self/exe")
    if process.is_symlink():
        binary = process.resolve()
        if binary.name.lower() == "lichtfeld-studio":
            return str(binary)
    raise ValueError("Set LICHTFELD to the LichtFeld Studio executable for project conversion")


def default_python():
    configured = os.environ.get("MESH_PYTHON")
    if configured:
        return configured
    local = Path.home() / ".local/share/lichtfeld-tiled-mesh/worker/bin/python"
    return str(local) if local.is_file() else ""


class Settings(PropertyGroup):
    python = StringProperty(name="Worker Python", subtype="FILE_PATH", default=default_python())
    model = StringProperty(name="Saved model (manual override)", subtype="FILE_PATH")
    dataset = StringProperty(name="Dataset (manual override)", subtype="DIR_PATH")
    cameras = StringProperty(name="Final cameras (override)", subtype="DIR_PATH")
    scan = StringProperty(name="Original depth scan", subtype="DIR_PATH")
    destination = StringProperty(name="Export folder", subtype="DIR_PATH")
    source = EnumProperty(name="Depth source", default="splat", items=[
        ("splat", "3DGS", "Multi-view checked Gaussian depth"),
        ("lidar", "LiDAR", "High-confidence RGB-aligned sensor depth"),
        ("hybrid", "LiDAR + 3DGS", "Use sensor evidence first")])
    voxel = FloatProperty(name="Voxel size (m)", default=0.02, min=0.001, step=0.01)
    cells = IntProperty(name="Cells per tile edge", default=64, min=8, max=256)
    width = IntProperty(name="Depth width (px)", default=320, min=8)
    bake = BoolProperty(name="Bake photo textures", default=True)
    two_sided = BoolProperty(name="Bake both sides", default=False)
    images = StringProperty(name="Undistorted images (override)", subtype="DIR_PATH")
    photo_scale = EnumProperty(name="Texture resolution", default="1", items=[
        ("1", "Native photos", "Retain input photo resolution"),
        ("2", "Half resolution", "Smaller textures for previews"),
        ("4", "Quarter resolution", "Compact textures")])
    cache = StringProperty(name="Prepared views", subtype="DIR_PATH")
    mesh = StringProperty(name="Existing mesh to rebake", subtype="DIR_PATH")
    tile = StringProperty(name="Tile X Y Z (blank for all)")
    automatic = BoolProperty(name="Export after training", default=False)


def model_command(s, iteration):
    manual = iteration is None
    params = lf.dataset_params()
    override = manual and bool(s.model)
    use_active = params.has_params() and not override
    output_path = params.output_path if use_active else ""
    project = s.model if override else lf.project_path()
    if not project and output_path:
        project = str(Path(output_path) / "project.licht")
    if not project and not params.has_params():
        project = s.model
    if not project or not Path(project).is_file():
        raise ValueError("Choose a saved training project or Gaussian PLY")
    project = Path(project).resolve()
    info = project.stat()
    if not manual:
        details = lf.io.inspect_project_details(project)
        if not any(c.binds_scene_graph and c.header_reachable and c.iteration == iteration
                   for c in details.retained_checkpoints):
            raise ValueError(f"Save the terminal checkpoint at iteration {iteration} before exporting")
    trained = Path(output_path) if output_path else project.parent
    dataset = params.data_path if use_active else ""
    if s.dataset and (manual or not params.has_params()):
        dataset = s.dataset
    if not dataset and project.suffix.lower() == ".licht":
        references = lf.io.inspect_project_details(project, checkpoint_byte_budget=0).references
        dataset = next((str(ref.path) for ref in references if ref.kind == "dataset" and ref.reachable), "")
    cameras = s.cameras
    if not cameras and use_active:
        if lf.optimization_params().pose_opt or params.centralize_dataset != "off":
            cameras = str(trained / "cameras_refined")
            if not Path(cameras).is_dir():
                raise ValueError("Choose the final COLMAP camera model in the Mesh Export tab")
    cmd = ["export-mesh", "--trained", str(trained)]
    if s.source != "lidar":
        cmd += ["--splat", str(project), "--model-version", f"{info.st_size}-{info.st_mtime_ns}"]
        if project.suffix.lower() == ".licht":
            cmd += ["--lichtfeld", lichtfeld_binary()]
    if s.source != "splat":
        if not s.scan:
            raise ValueError("Choose the original depth scan")
        cmd += ["--scan", s.scan]
    if dataset:
        cmd += ["--dataset", dataset]
    if cameras:
        cmd += ["--cameras", cameras]
    return cmd, trained


def launch(iteration=None, cached=False, rebake=False):
    global _job, _status, _log
    try:
        if _job is not None and _job.poll() is None:
            raise ValueError("Mesh export already running")
        s = Settings.get_instance()
        if not s.python:
            raise ValueError("Choose Worker Python in the Mesh Export tab")
        if cached or rebake:
            mesh = Path(s.mesh).resolve() if rebake else None
            if rebake and not (mesh / "mesh.json").is_file():
                raise ValueError("Choose an existing mesh directory containing mesh.json")
            cache = Path(s.cache).resolve() if s.cache else None
            if cache is None and rebake:
                cache = mesh / "views"
            if cache is None or not (cache / "views.json").is_file():
                raise ValueError("Choose a prepared views directory containing views.json")
            if rebake:
                cmd, base = ["bake-textures", "--mesh", str(mesh)], mesh.parent
            else:
                cmd, base = ["build-mesh"], cache.parent
            cmd += ["--views", str(cache)]
            if s.tile.strip():
                coords = [int(t) for t in s.tile.split()]
                if len(coords) != 3:
                    raise ValueError("Enter three tile indices as X Y Z")
                cmd += ["--tile", *map(str, coords)]
        else:
            cmd, base = model_command(s, iteration)
            cmd += ["--width", str(s.width)]
        if not rebake:
            cmd += ["--source", s.source, "--voxel", str(s.voxel), "--tile-cells", str(s.cells)]
            if s.bake:
                cmd += ["--bake-textures"]
        if rebake or s.bake:
            cmd += ["--texture-downsample", s.photo_scale]
            if s.two_sided:
                cmd += ["--two-sided-textures"]
            if s.images:
                cmd += ["--images", s.images]
            elif (cached or rebake) and s.dataset:
                cmd += ["--images", str(Path(s.dataset) / "images")]
        root = Path(s.destination).expanduser().resolve() if s.destination else base.resolve() / "mesh_exports"
        if any((p / "capture-meta.json").is_file() for p in (root, *root.parents)):
            raise ValueError("Choose an export folder outside the original scan")
        root.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="mesh-", dir=root))
        cmd += ["--output", str(run / "mesh")]
        script = Path(__file__).resolve().with_name("export_mesh.py")
        _log = run / "export.log"
        with _log.open("w") as log:
            _job = subprocess.Popen([s.python, "-I", str(script), *cmd], stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        _status = f"Exporting to {run / 'mesh'}"
        lf.log.info(_status)
    except (OSError, ValueError, RuntimeError) as error:
        _status = str(error)
        lf.log.error(_status)


def training_end(hook):
    if hook.get("terminal_state") == "completed" and Settings.get_instance().automatic:
        launch(hook["iteration"])


class MeshPanel(lf.ui.Panel):
    id = "tiled_mesh.export"
    label = "Mesh Export"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB

    def draw(self, ui):
        global _job, _status
        s = Settings.get_instance()
        for name in ("python", "model", "dataset", "cameras", "source"):
            ui.prop(s, name)
        if s.source != "splat":
            ui.prop(s, "scan")
        for name in ("destination", "voxel", "cells", "width", "bake"):
            ui.prop(s, name)
        if s.bake or s.mesh:
            ui.prop(s, "images")
            ui.prop(s, "photo_scale")
            ui.prop(s, "two_sided")
        ui.prop(s, "automatic")
        if ui.button("Export Saved Model"):
            launch()
        ui.prop(s, "cache")
        ui.prop(s, "mesh")
        if s.cache or s.mesh:
            ui.prop(s, "tile")
            if s.cache and ui.button("Rebuild from Prepared Views"):
                launch(cached=True)
            if s.mesh and ui.button("Rebake Textures"):
                launch(rebake=True)
        if _job is not None and _job.poll() is not None:
            _status = f"Mesh saved in {_log.parent / 'mesh'}" if _job.returncode == 0 else f"Export failed. See {_log}"
            if _job.returncode == 0:
                times = json.loads((_log.parent / "mesh" / "timings.json").read_text())
                _status += " (" + ", ".join(f"{name.replace('_', ' ')} {t:.1f}s" for name, t in times.items()) + ")"
            _job = None
        if _status:
            ui.label(_status)


def on_load():
    global _hooks
    lf.register_class(MeshPanel)
    _hooks = lf.ScopedHandler()
    _hooks.on_training_end(training_end)


def on_unload():
    global _hooks
    if _hooks is not None:
        _hooks.clear()
        _hooks = None
    lf.unregister_class(MeshPanel)
