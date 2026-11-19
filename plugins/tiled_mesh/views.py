"""Prepare calibrated, camera-z depth images for independent tile jobs."""
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def cameras(model, width):
    import pycolmap

    reconstruction = pycolmap.Reconstruction(model)
    views = []
    for image in sorted(reconstruction.images.values(), key=lambda i: i.name):
        camera = reconstruction.cameras[image.camera_id]
        if camera.model_name not in {"PINHOLE", "SIMPLE_PINHOLE"}:
            raise ValueError(f"{image.name}: use the undistorted PINHOLE dataset")
        w = min(width, camera.width)
        h = max(1, round(camera.height * w / camera.width))
        K = camera.calibration_matrix().copy()
        K[:2] *= np.array([w / camera.width, h / camera.height])[:, None]
        views.append(dict(name=image.name, K=K.tolist(), size=[h, w], rgb_size=[camera.height, camera.width],
                          w2c=image.cam_from_world().matrix().tolist()))
    if not views:
        raise ValueError("The camera model contains no registered images")
    return views


def lidar_rows(scan):
    meta = json.loads((scan / "capture-meta.json").read_text())
    if meta.get("depthFormat") != "float32_raw_little_endian":
        raise ValueError("Expected float32_raw_little_endian RGB-aligned depth")
    rows = {}
    for line in (scan / "poses.jsonl").read_text().splitlines():
        row = json.loads(line)
        name = row["imageFile"]
        if name in rows:
            raise ValueError(f"Duplicate capture image name: {name}")
        rows[name] = row
    return rows


def lidar_depth(scan, row, view, far):
    """Resample co-located, RGB-aligned optical-axis depth by its calibrated rays."""
    h, w = view["size"]
    result = np.zeros((h, w), dtype=np.float32)
    if not row or not row.get("depthFile") or not row.get("confidenceFile"):
        return result
    shape = (row["depthHeight"], row["depthWidth"])
    depth = np.fromfile(scan / "depth" / row["depthFile"], "<f4").reshape(shape)
    conf = np.fromfile(scan / "depth" / row["confidenceFile"], "u1").reshape(shape)
    k = row["intrinsics"]
    K = np.asarray(view["K"])
    # COLMAP locates pixel centers at (0.5, 0.5). ARKit calibration uses
    # pixel indices, so convert the native principal point before scaling.
    def indices(n, focal, principal, source_size, reference_size, axis):
        rays = (np.arange(n) + 0.5 - K[axis, 2]) / K[axis, axis]
        return np.floor((rays * focal + principal + 0.5) * source_size / reference_size).astype(int)
    x = indices(w, k["fx"], k["cx"], shape[1], k["width"], 0)
    y = indices(h, k["fy"], k["cy"], shape[0], k["height"], 1)
    inside = ((y >= 0) & (y < shape[0]))[:, None] & ((x >= 0) & (x < shape[1]))
    ij = np.ix_(y.clip(0, shape[0] - 1), x.clip(0, shape[1] - 1))
    z = depth[ij]
    valid = inside & (conf[ij] == 2) & (z > 0) & (z <= far)
    result[valid] = z[valid]
    return result


def file_version(path):
    info = path.stat()
    return f"{info.st_size}-{info.st_mtime_ns}"


def render_depths(model_path, views, directory, far, binary, batch_size, expected_version=None):
    import torch
    from fvdb_reality_capture import GaussianSplat3d

    version = file_version(model_path)
    if expected_version and expected_version != version:
        raise ValueError("The saved model changed after the export was requested")
    ply = model_path
    if model_path.suffix.lower() == ".licht":
        ply = directory / "converted.ply"
        subprocess.run([binary, "convert", str(model_path), str(ply), "--sh-degree", "0"], check=True)
    host, _ = GaussianSplat3d.from_ply(ply, device="cpu")
    if file_version(model_path) != version:
        raise ValueError("The saved model changed during loading. Retry with the saved version.")
    if ply != model_path:
        ply.unlink()
    torch.set_num_threads(1)
    resident = host.to("cuda") if host.num_gaussians <= batch_size else None
    centers = host.means.detach().numpy()
    with torch.inference_mode():
        for i, view in enumerate(views):
            if i % 100 == 0:
                print(f"Rendering depth views {i+1}/{len(views)}", flush=True)
            if view["excluded"]:
                continue
            h, w = view["size"]
            pose = np.eye(4, dtype=np.float32)
            pose[:3] = view["w2c"]
            matrix = torch.as_tensor(pose, device="cuda")[None]
            K = torch.tensor(view["K"], dtype=torch.float32, device="cuda")[None]
            depth_sum = torch.zeros((h, w), device="cuda")
            transmission = torch.ones_like(depth_sum)
            if resident is None:
                camera_z = np.einsum("ij,j->i", centers, pose[2, :3]) + pose[2, 3]
                visible = np.flatnonzero(camera_z > 0.01)
                order = visible[np.argsort(camera_z[visible], kind="stable")]
            count = host.num_gaussians if resident is not None else len(order)
            for start in range(0, count, batch_size):
                # Composite depth-ordered batches with their residual transmittance.
                # Unnormalized depth is additive this way even at partial opacity.
                part = resident
                if part is None:
                    indices = torch.from_numpy(order[start:start + batch_size])
                    part = host[indices].to("cuda")
                depth, alpha = part.render_depths(matrix, K, w, h, near=0.01, far=1e10, min_radius_2d=0)
                depth_sum += transmission * depth[0, :, :, 0]
                transmission *= 1 - alpha[0, :, :, 0]
                del part
            opacity = 1 - transmission
            z = depth_sum / opacity.clamp_min(1e-8)
            valid = (opacity >= 0.5) & (z > 0) & (z <= far)
            np.save(directory / f"raw-{i}.npy", torch.where(valid, z, 0).cpu().numpy())


def filter_splats(views, directory, has_lidar):
    """Require agreement in two other cameras with a nonzero physical baseline."""
    indices = [i for i, view in enumerate(views) if not view["excluded"]]
    if len(indices) < 3:
        raise ValueError("Splat fusion requires at least three non-excluded cameras")
    poses = np.array([views[i]["w2c"] for i in indices])
    centers = -np.einsum("nji,nj->ni", poses[:, :, :3], poses[:, :, 3])
    tree = cKDTree(centers)
    for j, i in enumerate(indices):
        if j % 100 == 0:
            print(f"Checking multi-view depth {j+1}/{len(indices)}", flush=True)
        z = np.load(directory / f"raw-{i}.npy")
        y, x = np.nonzero(z)
        K = np.asarray(views[i]["K"])
        xyz = np.stack(((x + 0.5 - K[0, 2]) / K[0, 0],
                        (y + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(x))), axis=1)
        world = np.einsum("ij,jk->ik", xyz * z[y, x, None], poses[j, :, :3]) + centers[j]
        votes = np.zeros(len(x), dtype=np.uint8)
        _, nearest = tree.query(centers[j], k=min(24, len(indices)))
        veto = np.zeros(len(x), dtype=bool)
        if has_lidar:
            sensor = np.load(directory / f"lidar-{i}.npy", mmap_mode="r")
            veto |= sensor[y, x] > 0
        neighbors = []
        for n in np.atleast_1d(nearest):
            if np.linalg.norm(centers[j] - centers[n]) < 0.02 or poses[j, 2, :3] @ poses[n, 2, :3] < 0.3:
                continue
            if any(np.linalg.norm(centers[n] - centers[m]) < 0.02 for m in neighbors):
                continue
            k = indices[n]
            other = np.load(directory / f"raw-{k}.npy", mmap_mode="r")
            K2 = np.asarray(views[k]["K"])
            camera = np.einsum("ij,kj->ik", world, poses[n, :, :3]) + poses[n, :, 3]
            predicted = camera[:, 2]
            uv = camera[:, :2] / np.maximum(predicted[:, None], 1e-6)
            uv = np.floor(uv * [K2[0, 0], K2[1, 1]] + K2[:2, 2]).astype(np.int64)
            h, w = other.shape
            u, v = uv.T
            inside = (predicted > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            d = other[v[inside], u[inside]]
            votes[inside] += (d > 0) & (np.abs(d - predicted[inside]) <= 0.03 + 0.015 * predicted[inside])
            if has_lidar:
                sensor = np.load(directory / f"lidar-{k}.npy", mmap_mode="r")
                observed = sensor[v[inside], u[inside]]
                # A return behind this candidate means measured free space here.
                veto[inside] |= observed > predicted[inside] * 1.015 + 0.03
            neighbors.append(n)
            if len(neighbors) >= 8:
                break
        z[y[votes < 2], x[votes < 2]] = 0
        np.save(directory / f"splat-{i}.npy", z)
        if has_lidar:
            z[y[veto], x[veto]] = 0
            np.save(directory / f"fill-{i}.npy", z)
    for i in indices:
        (directory / f"raw-{i}.npy").unlink()


def prepare(args, directory):
    dataset = args.dataset
    trained = args.trained
    config_file = trained / "training_config.json" if trained else None
    config = json.loads(config_file.read_text()) if config_file and config_file.exists() else {}
    if not dataset and config.get("dataset", {}).get("data_path"):
        dataset = Path(config["dataset"]["data_path"])
    refined = trained / "cameras_refined" if trained else None
    optimization = config.get("optimization", {})
    pose_opt = optimization.get("pose_opt", optimization.get("use_pose_opt"))
    if args.cameras:
        model = args.cameras
    elif refined and pose_opt is not False and any((refined / f"cameras.{ext}").is_file() for ext in ("bin", "txt")):
        model = refined
    elif pose_opt or config.get("dataset", {}).get("centralize_dataset", "off") != "off":
        raise ValueError("Supply final world-aligned --cameras for this transformed or pose-refined run")
    elif dataset:
        model = dataset / "sparse" / "0"
    else:
        raise ValueError("Supply --dataset or --cameras")
    views = cameras(model, args.width)
    excluded = set(args.exclude.read_text().splitlines()) if args.exclude else set()
    unknown = excluded.difference(v["name"] for v in views)
    if unknown or len(excluded) == len(views):
        raise ValueError("Excluded names must match registered cameras and leave fusion views")
    for v in views:
        v["excluded"] = v["name"] in excluded
    sources = []
    if args.scan:
        rows = lidar_rows(args.scan)
        valid = 0
        for i, v in enumerate(views):
            z = lidar_depth(args.scan, rows.get(v["name"]), v, args.max_depth)
            valid += bool(z.any()) and not v["excluded"]
            np.save(directory / f"lidar-{i}.npy", z)
        if not valid:
            raise ValueError("No matched non-excluded frames contain high-confidence depth")
        sources.append("lidar")
    if args.splat:
        render_depths(args.splat, views, directory, args.max_depth, args.lichtfeld, args.batch_splats, args.model_version)
        filter_splats(views, directory, bool(args.scan))
        sources.append("splat")
    if not sources:
        raise ValueError("Supply a --scan, --splat or both")
    image_root = args.images
    if not image_root and dataset:
        image_root = dataset / config.get("dataset", {}).get("images", "images")
    metadata = dict(views=views, sources=sources)
    if image_root:
        metadata["image_root"] = str(image_root.resolve())
    (directory / "views.json").write_text(json.dumps(metadata))
