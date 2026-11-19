"""Projective narrow-band TSDF fusion on a shared metric lattice."""
import itertools
import json
from collections import defaultdict

import numpy as np
from numba import njit
from plyfile import PlyData, PlyElement
from skimage.measure import marching_cubes


@njit
def index_view(depth, K, pose, voxel, cells):
    """Conservative world bounds of each depth pixel and its truncation band."""
    bounds = {}
    R = pose[:, :3]
    center = -R.T @ pose[:, 3].copy()
    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            z = depth[y, x]
            if z <= 0:
                continue
            rx = (x + 0.5 - K[0, 2]) / K[0, 0]
            ry = (y + 0.5 - K[1, 2]) / K[1, 1]
            band = 4 * voxel
            # Enclose the pixel cone at both endpoints in a world-space cube.
            radius = (band * np.sqrt(1 + rx * rx + ry * ry)
                      + (z + band) * np.hypot(0.5 / K[0, 0], 0.5 / K[1, 1])) / voxel
            px = (center[0] + z * (R[0, 0]*rx + R[1, 0]*ry + R[2, 0])) / voxel
            py = (center[1] + z * (R[0, 1]*rx + R[1, 1]*ry + R[2, 1])) / voxel
            pz = (center[2] + z * (R[0, 2]*rx + R[1, 2]*ry + R[2, 2])) / voxel
            x0, y0, z0 = int(np.ceil(px-radius)), int(np.ceil(py-radius)), int(np.ceil(pz-radius))
            x1, y1, z1 = int(np.floor(px+radius)), int(np.floor(py+radius)), int(np.floor(pz+radius))
            # Each tile owns cells [origin, origin+cells), and both endpoint samples.
            for a in range((x0-1)//cells, x1//cells+1):
                for b in range((y0-1)//cells, y1//cells+1):
                    for c in range((z0-1)//cells, z1//cells+1):
                        key = (a, b, c)
                        if key in bounds:
                            old = bounds[key]
                            bounds[key] = (min(old[0], x0), min(old[1], y0), min(old[2], z0),
                                           max(old[3], x1), max(old[4], y1), max(old[5], z1))
                        else:
                            bounds[key] = (x0, y0, z0, x1, y1, z1)
    return bounds


@njit
def integrate(depth, K, pose, voxel, origin, bounds, total, count):
    n = total.shape[0]
    lo = np.maximum(np.asarray(bounds[:3]) - origin, 0)
    hi = np.minimum(np.asarray(bounds[3:]) - origin + 1, n)
    for x in range(lo[0], hi[0]):
        xx = (origin[0] + x) * voxel
        for y in range(lo[1], hi[1]):
            yy = (origin[1] + y) * voxel
            cx = pose[0, 0]*xx + pose[0, 1]*yy + pose[0, 3]
            cy = pose[1, 0]*xx + pose[1, 1]*yy + pose[1, 3]
            cz = pose[2, 0]*xx + pose[2, 1]*yy + pose[2, 3]
            for z in range(lo[2], hi[2]):
                zz = (origin[2] + z) * voxel
                d = cz + pose[2, 2]*zz
                if d <= 0:
                    continue
                u = int(np.floor(K[0, 0]*(cx + pose[0, 2]*zz)/d + K[0, 2]))
                v = int(np.floor(K[1, 1]*(cy + pose[1, 2]*zz)/d + K[1, 2]))
                if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
                    measured = depth[v, u]
                    sdf = measured - d
                    if measured > 0 and abs(sdf) <= 4*voxel:
                        total[x, y, z] += sdf
                        count[x, y, z] += 1


def surface(field, known, origin, voxel):
    # Lewiner's mask selects a cell by its upper corner. Every corner must have evidence.
    cells = np.ones(np.array(known.shape) - 1, bool)
    for delta in itertools.product((0, 1), repeat=3):
        cells &= known[tuple(slice(i, n-1+i) for i, n in zip(delta, known.shape))]
    mask = np.zeros_like(known)
    mask[1:, 1:, 1:] = cells
    if not cells.any() or field.min() >= 0 or field.max() <= 0:
        return None
    try:
        verts, faces, _, _ = marching_cubes(field, 0, mask=mask, allow_degenerate=False,
                                            gradient_direction="descent")
    except RuntimeError as exc:
        if "No surface found" in str(exc):
            return None
        raise
    return (verts.astype(np.float64) + origin) * voxel, faces


def write_ply(path, vertices, faces):
    v = np.empty(len(vertices), dtype=[(c, "<f8") for c in "xyz"])
    for axis, name in enumerate("xyz"):
        v[name] = vertices[:, axis]
    f = np.empty(len(faces), dtype=[("vertex_indices", "<i4", (3,))])
    f["vertex_indices"] = faces
    PlyData([PlyElement.describe(v, "vertex"), PlyElement.describe(f, "face")],
            byte_order="<").write(str(path))


def build(args, out):
    meta = json.loads((args.views / "views.json").read_text())
    if args.source is None:
        args.source = "splat" if "splat" in meta["sources"] else "lidar"
        if "patch" in meta["sources"]:
            args.source = "sensor-priority"
    sources = {"hybrid": ["lidar", "splat"],
               "sensor-priority": ["lidar", "patch"],
               "sensor-priority-fill": ["lidar", "patch", "guard"]}.get(args.source, [args.source])
    if set(sources) - set(meta["sources"]):
        raise ValueError(f"Missing prepared depths for {args.source}")
    tiles = defaultdict(list)
    for i, view in enumerate(meta["views"]):
        if view["excluded"]:
            continue
        K = np.asarray(view["K"])
        pose = np.asarray(view["w2c"])
        for source in sources:
            prefix = "fill" if args.source == "hybrid" and source == "splat" else source
            path = args.views / f"{prefix}-{i}.npy"
            depth = np.load(path, mmap_mode="r")
            for key, bounds in index_view(depth, K, pose, args.voxel, args.tile_cells).items():
                if args.tile is None or key == tuple(args.tile):
                    tiles[key].append((source, path, K, pose, bounds))
    entries = []
    shape = (args.tile_cells + 1,) * 3
    for index, (key, observations) in enumerate(sorted(tiles.items())):
        if index % 100 == 0:
            print(f"Fusing tile {index+1}/{len(tiles)}", flush=True)
        origin = np.array(key) * args.tile_cells
        field = np.zeros(shape, np.float32)
        known = np.zeros(shape, bool)
        for source in reversed(sources):
            total = np.zeros(shape, np.float32)
            count = np.zeros(shape, np.uint32)
            for kind, path, K, pose, bounds in observations:
                if kind == source:
                    integrate(np.load(path, mmap_mode="r"), K, pose, args.voxel, origin, bounds, total, count)
            supported = count >= 2
            field[supported] = total[supported] / count[supported]
            known |= supported
        mesh = surface(field, known, origin, args.voxel)
        if mesh is not None:
            name = "tile_{}_{}_{}.ply".format(*key)
            write_ply(out / name, *mesh)
            entries.append(dict(tile=key, file=name, vertices=len(mesh[0]), faces=len(mesh[1])))
    if not entries:
        raise ValueError("No observed surface in the selected tiles")
    manifest = dict(units="metres", source=args.source, voxel=args.voxel, tile_cells=args.tile_cells,
                    fusion_views=sum(not v["excluded"] for v in meta["views"]), tiles=entries)
    (out / "mesh.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(entries)} tiles, {sum(e['faces'] for e in entries)} triangles", flush=True)
