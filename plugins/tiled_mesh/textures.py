"""Bake calibrated photographs into portable base-color atlases, one mesh tile at a time."""
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
import trimesh


class Photos:
    def __init__(self, views, images, source):
        self.directory = views
        metadata = json.loads((views / "views.json").read_text())
        root = images or metadata.get("image_root")
        if not root or not Path(root).is_dir():
            raise ValueError("Choose the undistorted dataset's images directory with --images")
        self.root = Path(root).resolve()
        self.frames = metadata["views"]
        self.indices = np.array([i for i, v in enumerate(self.frames) if not v["excluded"]], dtype=int)
        self.source = source
        self.K = np.array([v["K"] for v in self.frames])
        self.poses = np.array([v["w2c"] for v in self.frames])
        self.centers = -np.einsum("nji,nj->ni", self.poses[:, :, :3], self.poses[:, :, 3])
        sizes = np.array([v["size"] for v in self.frames])
        px = self.K @ self.poses
        self.planes = np.stack([px[:, 0], sizes[:, 1, None]*px[:, 2]-px[:, 0],
                                px[:, 1], sizes[:, 0, None]*px[:, 2]-px[:, 1], px[:, 2]], axis=1)

    def candidates(self, points):
        """Cull cameras whose frusta do not intersect the tile surface bounds."""
        lo, hi = points.min(axis=0), points.max(axis=0)
        planes = self.planes[self.indices]
        extent = np.abs(planes[..., :3]) @ ((hi-lo)/2)
        inside = planes[..., :3] @ ((hi+lo)/2) + planes[..., 3] + extent >= 0
        return self.indices[inside.all(axis=1)]

    @lru_cache(maxsize=32)
    def depth(self, index, kind):
        return np.load(self.directory / f"{kind}-{index}.npy", mmap_mode="r")

    @lru_cache(maxsize=8)
    def image(self, index):
        name = Path(self.frames[index]["name"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Expected a dataset-relative photo name: {name}")
        path = self.root / name
        with Image.open(path) as image:
            if "rgb_size" in self.frames[index] and list(image.size[::-1]) != self.frames[index]["rgb_size"]:
                raise ValueError(f"{name}: photo dimensions differ from the calibrated undistorted camera")
            return image.convert("RGB")

    def project(self, points, index):
        pose = self.poses[index]
        K = self.K[index]
        camera = np.einsum("...j,kj->...k", points, pose[:, :3]) + pose[:, 3]
        z = camera[..., 2]
        uv = camera[..., :2] / np.maximum(z[..., None], 1e-6)
        uv *= [K[0, 0], K[1, 1]]
        uv += K[:2, 2]
        return uv, z

    def scores(self, centers, normals, index, voxel, reverse):
        uv, z = self.project(centers, index)
        h, w = self.frames[index]["size"]
        ray = self.centers[index] - centers
        distance = np.linalg.norm(ray, axis=1).clip(1e-6)
        cosine = np.einsum("ij,ij->i", normals, ray) / distance
        visible = (z > 0) & (cosine > 0.15) & (uv[:, 0] >= 1) & (uv[:, 0] < w-1)
        visible &= (uv[:, 1] >= 1) & (uv[:, 1] < h-1)
        score = np.zeros(len(z), dtype=np.float32)
        if not visible.any():
            return score
        measured = self.measured(index, uv[visible])
        tolerance = 2 * voxel + 0.02 * z[visible]
        if reverse:
            tolerance *= 0.5
        agrees = (measured > 0) & (np.abs(measured - z[visible]) <= tolerance)
        # Favor frontal, high sampling density views while rejecting occluders.
        score[visible] = agrees * cosine[visible] ** 2 * self.K[index, 0, 0] / distance[visible]
        return score

    def measured(self, index, uv):
        u, v = np.moveaxis(uv.astype(int), -1, 0)
        secondary = {"hybrid": "splat", "sensor-priority": "patch", "sensor-priority-fill": "patch"}.get(self.source)
        primary = "lidar" if secondary else self.source
        measured = self.depth(index, primary)[v, u]
        if secondary:
            measured = np.where(measured > 0, measured, self.depth(index, secondary)[v, u])
        if self.source == "sensor-priority-fill":
            measured = np.where(measured > 0, measured, self.depth(index, "guard")[v, u])
        return measured

    def corners_visible(self, tri, index, voxel):
        uv, z = self.project(tri, index)
        h, w = self.frames[index]["size"]
        valid = ((uv >= 0) & (uv < [w, h])).all(axis=(1, 2)) & (z > 0).all(axis=1)
        depth = self.measured(index, uv[valid])
        blocked = (depth > 0) & (depth + 2*voxel + 0.02*z[valid] < z[valid])
        valid[valid] = ~blocked.any(axis=1)
        return valid


def choose_views(photos, tri, normals, voxel, limit, reverse=False):
    centers = tri.mean(axis=1)
    camera = np.full(len(centers), -1)
    candidates = photos.candidates(centers)
    if not len(candidates):
        return camera
    sample = np.linspace(0, len(centers)-1, min(512, len(centers)), dtype=int)
    scores = np.stack([photos.scores(centers[sample], normals[sample], i, voxel, reverse)
                       for i in candidates])
    current = np.zeros(len(sample), dtype=np.float32)
    choices = []
    for _ in range(min(limit, len(scores))):
        gain = np.maximum(scores-current, 0).sum(axis=1)
        j = gain.argmax()
        if gain[j] <= 0:
            break
        choices.append(candidates[j])
        current = np.maximum(current, scores[j])
    # Prefer a common photograph when the gain in sampling density is small.
    best = np.zeros(len(centers), dtype=np.float32)
    for i in choices:
        value = photos.scores(centers, normals, i, voxel, reverse)
        better = np.flatnonzero(value > best * 1.12)
        better = better[photos.corners_visible(tri[better], i, voxel)]
        camera[better] = i
        best[better] = value[better]
    return camera


def charts(photos, vertices, faces, camera, downsample, atlas_size):
    """Group neighboring projections and copy native photograph detail with gutters."""
    patches = []
    for i in np.unique(camera):
        if i < 0:
            continue
        indices = np.flatnonzero(camera == i)
        uv, z = photos.project(vertices[faces[indices]], i)
        image = photos.image(int(i))
        h, w = photos.frames[i]["size"]
        uv *= [image.width / w, image.height / h]
        valid = ((uv >= 0) & (uv < [image.width, image.height])).all(axis=(1, 2)) & (z > 0).all(axis=1)
        camera[indices[~valid]] = -1
        uv, indices = uv[valid], indices[valid]
        if not len(indices):
            continue
        # Fixed image-space buckets bound crop waste for disconnected patches.
        cell = max(64, min(256, atlas_size // 4)) * downsample
        grid = (uv.mean(axis=1) / cell).astype(int)
        keys = grid[:, 1] * (1 + image.width // cell) + grid[:, 0]
        order = np.argsort(keys, kind="stable")
        for group in np.split(order, np.flatnonzero(np.diff(keys[order])) + 1):
            coords = uv[group]
            lo = np.maximum(np.floor(coords.min(axis=(0, 1))).astype(int)-2*downsample, 0)
            hi = np.minimum(np.ceil(coords.max(axis=(0, 1))).astype(int)+2*downsample, image.size)
            size = hi-lo
            scale = min(1/downsample, (atlas_size-4)/size.max())
            dest = np.maximum(1, np.ceil(size*scale).astype(int))
            crop = image.crop((*lo, *hi))
            if crop.size != tuple(dest):
                crop = crop.resize(tuple(dest), Image.Resampling.LANCZOS)
            # Replicate outer samples into a 2-pixel gutter, including image borders.
            rgb = np.pad(np.asarray(crop), ((2, 2), (2, 2), (0, 0)), mode="edge")
            local = (coords-lo) * (dest/size) + 2
            patches.append((indices[group], local, Image.fromarray(rgb)))
    missing = np.flatnonzero(camera < 0)
    if len(missing):
        patches.append((missing, np.full((len(missing), 3, 2), 2.5), Image.new("RGB", (5, 5), (128, 128, 128))))
    return patches


def pack(patches, maximum):
    patches.sort(key=lambda p: -p[2].height)
    area = sum(p[2].width*p[2].height for p in patches)
    width = min(maximum, max(max(p[2].width for p in patches), math.ceil(math.sqrt(area))))
    pages, row, x, y, row_height = [], [], 0, 0, 0
    for indices, uv, image in patches:
        if x + image.width > width:
            y += row_height
            x, row_height = 0, 0
        if y + image.height > maximum:
            pages.append(row)
            row, x, y, row_height = [], 0, 0, 0
        row.append((indices, uv, image, x, y))
        x += image.width
        row_height = max(row_height, image.height)
    if row:
        pages.append(row)
    return pages


def textured_tile(path, destination, photos, voxel, args):
    mesh = PlyData.read(path, known_list_len={"face": {"vertex_indices": 3}})
    verts = np.column_stack([mesh["vertex"][axis] for axis in "xyz"])
    faces = np.stack(mesh["face"]["vertex_indices"])
    tri = verts[faces]
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(areas[:, None], 1e-20)
    camera = choose_views(photos, tri, normals, voxel, args.bake_views)
    front = camera >= 0
    if args.two_sided_textures and not front.all():
        camera[~front] = choose_views(photos, tri[~front], -normals[~front], voxel,
                                      args.bake_views, reverse=True)
    patches = charts(photos, verts, faces, camera, args.texture_downsample, args.atlas_size)
    pages = pack(patches, args.atlas_size)
    scene = trimesh.Scene()
    # Preserve smooth vertex normals across UV and material boundaries within a tile.
    vn = np.zeros_like(verts)
    for column in faces.T:
        np.add.at(vn, column, cross)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20)
    for page, group in enumerate(pages):
        width = max(x + img.width for _, _, img, x, _ in group)
        height = max(y + img.height for _, _, img, _, y in group)
        image = Image.new("RGB", (width, height))
        pos, normal, coords, triangles, offset = [], [], [], [], 0
        for ids, uv, img, x, y in group:
            image.paste(img, (x, y))
            # Within one chart each original vertex has a single photographic UV.
            unique, first, inverse = np.unique(faces[ids].ravel(), return_index=True, return_inverse=True)
            pos.append(verts[unique])
            normal.append(vn[unique])
            coords.append((uv.reshape(-1, 2)[first]+[x, y])/[width, height])
            triangles.append(inverse.reshape(-1, 3)+offset)
            offset += len(unique)
        uv = np.concatenate(coords)
        uv[:, 1] = 1-uv[:, 1]
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicFactor=0,
                                                       roughnessFactor=1, doubleSided=args.two_sided_textures)
        visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
        part = trimesh.Trimesh(vertices=np.concatenate(pos), faces=np.concatenate(triangles),
                               vertex_normals=np.concatenate(normal), visual=visual, process=False)
        scene.add_geometry(part, geom_name=f"{path.stem}_{page}")
    destination.write_bytes(scene.export(file_type="glb"))
    return dict(file=destination.name, pages=len(pages), faces=len(faces),
                observed_area_m2=float(areas[camera >= 0].sum()/2), area_m2=float(areas.sum()/2),
                reverse_area_m2=float(areas[~front & (camera >= 0)].sum()/2))


def bake(args, out):
    meta = json.loads((args.mesh / "mesh.json").read_text())
    views = args.views or args.mesh / "views"
    photos = Photos(views, args.images, meta["source"])
    tiles = [t for t in meta["tiles"] if args.tile is None or tuple(t["tile"]) == tuple(args.tile)]
    if not tiles:
        raise ValueError("The selected tile is absent from mesh.json")
    entries = []
    for n, tile in enumerate(tiles):
        if n % 10 == 0:
            print(f"Baking tile {n+1}/{len(tiles)}", flush=True)
        path = args.mesh / tile["file"]
        result = textured_tile(path, out / path.with_suffix(".glb").name, photos, meta["voxel"], args)
        entries.append(dict(tile=tile["tile"], **result))
    area = sum(tile["area_m2"] for tile in entries)
    observed = sum(tile["observed_area_m2"] for tile in entries)
    coverage = observed / area if area else 0.
    report = dict(source=meta["source"], voxel=meta["voxel"], atlas_size=args.atlas_size,
                  texture_downsample=args.texture_downsample, bake_views=args.bake_views,
                  two_sided_textures=args.two_sided_textures,
                  observed_area_fraction=coverage, tiles=entries)
    (out / "textures.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {len(entries)} textured GLB tiles, {coverage:.1%} of surface area textured", flush=True)
    photos.image.cache_clear()
    photos.depth.cache_clear()
