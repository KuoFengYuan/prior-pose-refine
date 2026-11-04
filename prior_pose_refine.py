#!/usr/bin/env python3
"""Helper subcommands of prior-pose-refine: COLMAP model I/O and the steps between the COLMAP commands.

The prior model is a COLMAP binary model holding the approximate (prior) poses, intrinsics and points,
in metres. See README.md and docs/method.md.

Subcommands
  list-images          names of the frames that have a prior pose
  export-centers       prior camera centers ("name x y z") for colmap model_aligner
  select-pairs         matching pairs from the prior poses and point covisibility
  init-model           COLMAP text model with the prior poses, ids taken from a database
  write-pose-priors    prior positions and gravity written into a database, for colmap mappers
  restore-prior-poses  give the frames of a model with too few observations their prior pose
  report-agreement     camera-center agreement between a model and the prior poses, and the frames to inspect
  merge-points         add the prior points, mapped into a model's frame, to its SfM points
  print-camera         print the camera(s) of a COLMAP model
"""
import argparse
import os
from pathlib import Path
import sqlite3
import struct
import sys
import time

import numpy as np

# COLMAP camera model id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16), 12: ("SIMPLE_DIVISION", 4), 13: ("DIVISION", 5),
    14: ("SIMPLE_FISHEYE", 3), 15: ("FISHEYE", 4), 16: ("EUCM", 6),
}
SINGLE_FOCAL = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE",
                "SIMPLE_DIVISION", "SIMPLE_FISHEYE"}

# Distortion params that make a camera model a plain pinhole, used to start from the prior intrinsics.
# The "fisheye" models map the ray angle t to t (1 + k1 t^2 + k2 t^4 + ...); with the Taylor series of
# tan(t) they are a pinhole (within 0.02 px for 6 terms, 0.4 px for 4 terms at the corners of an iPhone image).
TAN_SERIES = [1 / 3, 2 / 15, 17 / 315, 62 / 2835, 1382 / 155925, 21844 / 6081075]
PINHOLE_DISTORTION = {"OPENCV": [0.0] * 4, "OPENCV_FISHEYE": TAN_SERIES[:4],
                      "RAD_TAN_THIN_PRISM_FISHEYE": TAN_SERIES + [0.0] * 6}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(R):
    """(w, x, y, z) of a rotation matrix, with w >= 0."""
    m = R
    k = np.array([[m[0, 0] - m[1, 1] - m[2, 2], 0, 0, 0],
                  [m[1, 0] + m[0, 1], m[1, 1] - m[0, 0] - m[2, 2], 0, 0],
                  [m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], m[2, 2] - m[0, 0] - m[1, 1], 0],
                  [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]]]) / 3
    w, v = np.linalg.eigh(k)  # uses the lower triangle
    x, y, z, qw = v[:, np.argmax(w)]
    q = np.array([qw, x, y, z])
    return q if qw >= 0 else -q


def umeyama(src, dst):
    """Similarity (scale, R, t) with dst ≈ scale * R @ src + t."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if (src.ndim != 2 or src.shape != dst.shape or src.shape[1] != 3 or len(src) < 3
            or not np.isfinite(src).all() or not np.isfinite(dst).all()):
        raise ValueError("similarity requires at least three matching finite 3D camera centers")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    if sig[0] == 0 or sig[1] <= sig[0] * 3 * np.finfo(float).eps:
        raise ValueError("camera centers are coincident or collinear; similarity rotation is underconstrained")
    e = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        e[2, 2] = -1
    r = u @ e @ vt
    scale = np.trace(np.diag(sig) @ e) / s.var(0).sum()
    translation = mu_d - scale * r @ mu_s
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(translation).all():
        raise ValueError("camera centers do not define a finite positive-scale similarity")
    return scale, r, translation


def rotation_deg(r):
    return np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))


def camera_centers(frames):
    return {f["name"]: -f["R"].T @ f["t"] for f in frames}


# ---------------------------------------------------------------------------
# COLMAP binary model I/O
# ---------------------------------------------------------------------------

def read_images(model_dir):
    """Frames sorted by name: dicts with name, image_id, q (w,x,y,z), t, R (world-to-camera), camera_id
    and n_obs (keypoints with a 3D point)."""
    data = Path(model_dir, "images.bin").read_bytes()
    count = struct.unpack_from("<Q", data, 0)[0]
    off, frames = 8, []
    for _ in range(count):
        image_id, qw, qx, qy, qz, tx, ty, tz, cam = struct.unpack_from("<I7dI", data, off)
        off += struct.calcsize("<I7dI")
        end = data.index(b"\0", off)
        name = data[off:end].decode()
        off = end + 1
        n2d = struct.unpack_from("<Q", data, off)[0]
        point_ids = np.frombuffer(data, "<i8", 3 * n2d, off + 8)[2::3]  # (x, y, point3D_id) per keypoint
        off += 8 + 24 * n2d
        q = np.array([qw, qx, qy, qz])
        frames.append({"name": name, "image_id": image_id, "q": q, "t": np.array([tx, ty, tz]), "R": quat_to_rot(q),
                       "camera_id": cam, "n_obs": int((point_ids != -1).sum())})
    frames.sort(key=lambda f: f["name"])
    return frames


def write_poses(model_dir, poses):
    """Replaces the world-to-camera poses of some images (image_id -> (q, t)) of a binary model in place.

    COLMAP 4 keeps the pose in frames.bin (rig_from_world) as well as in images.bin, so both are rewritten.
    Only single-camera frames are supported, where rig_from_world is cam_from_world.
    """
    images_path = Path(model_dir, "images.bin")
    images = bytearray(images_path.read_bytes())
    off = 8
    for _ in range(struct.unpack_from("<Q", images, 0)[0]):
        image_id = struct.unpack_from("<I", images, off)[0]
        if image_id in poses:
            struct.pack_into("<7d", images, off + 4, *poses[image_id][0], *poses[image_id][1])
        end = images.index(b"\0", off + struct.calcsize("<I7dI"))
        off = end + 9 + 24 * struct.unpack_from("<Q", images, end + 1)[0]
    frames_path = Path(model_dir, "frames.bin")
    frames = bytearray(frames_path.read_bytes()) if frames_path.exists() else None
    if frames is not None:
        # frame_id, rig_id, rig_from_world (qw qx qy qz tx ty tz), num_data_ids, (sensor_type, sensor_id, data_id)[]
        off = 8
        for _ in range(struct.unpack_from("<Q", frames, 0)[0]):
            n_data = struct.unpack_from("<I", frames, off + 64)[0]
            data_ids = [struct.unpack_from("<iIQ", frames, off + 68 + 16 * k) for k in range(n_data)]
            hit = [data_id for sensor_type, _, data_id in data_ids if sensor_type == 0 and data_id in poses]
            if hit:
                if n_data != 1:
                    sys.exit(f"write_poses: frame with {n_data} sensors in {frames_path}; only single-camera rigs are supported")
                struct.pack_into("<7d", frames, off + 8, *poses[hit[0]][0], *poses[hit[0]][1])
            off += 68 + 16 * n_data
        if off != len(frames):
            sys.exit(f"write_poses: {frames_path} does not have the expected layout; nothing written")
    images_path.write_bytes(images)
    if frames is not None:
        frames_path.write_bytes(frames)


def read_cameras(model_dir):
    """camera_id -> (model name, width, height, params)."""
    data = Path(model_dir, "cameras.bin").read_bytes()
    count = struct.unpack_from("<Q", data, 0)[0]
    off, cams = 8, {}
    for _ in range(count):
        cid, model, width, height = struct.unpack_from("<iiQQ", data, off)
        off += 24
        name, n = CAMERA_MODELS[model]
        cams[cid] = (name, width, height, struct.unpack_from(f"<{n}d", data, off))
        off += 8 * n
    return cams


def pinhole(model, params):
    """fx, fy, cx, cy of any COLMAP camera model (distortion ignored)."""
    if model in SINGLE_FOCAL:
        return params[0], params[0], params[1], params[2]
    return params[0], params[1], params[2], params[3]


def read_points(model_dir, max_points, seed=0):
    path = os.path.join(model_dir, "points3D.bin")
    if not os.path.exists(path):
        return np.empty((0, 3))
    data = Path(path).read_bytes()
    count = struct.unpack_from("<Q", data, 0)[0]
    off, pts = 8, np.empty((count, 3))
    for i in range(count):
        pts[i] = struct.unpack_from("<3d", data, off + 8)
        off += 51 + 8 * struct.unpack_from("<Q", data, off + 43)[0]
    if max_points and len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    return pts


def point_records(path):
    """The raw records of a points3D.bin, one bytes object per point (id, xyz, rgb, error, track)."""
    if not os.path.exists(path):
        return []
    data = Path(path).read_bytes()
    off, records = 8, []
    for _ in range(struct.unpack_from("<Q", data, 0)[0]):
        size = 51 + 8 * struct.unpack_from("<Q", data, off + 43)[0]
        records.append(data[off:off + size])
        off += size
    return records


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list_images(a):
    names = [f["name"] for f in read_images(a.prior)]
    Path(a.out).write_text("\n".join(names) + "\n")
    print(f"{len(names)} frames with a prior pose")


def cmd_export_centers(a):
    lines = [f"{n} {float(c[0])!r} {float(c[1])!r} {float(c[2])!r}"
             for n, c in camera_centers(read_images(a.prior)).items()]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} camera centers")


def cmd_select_pairs(a):
    """Pairs images that see many of the same prior points.

    Every point is projected into every camera; a coarse z-buffer (a.cell px) keeps
    only the points near the closest depth of each cell, so points behind walls do
    not count as visible. Each image is paired with its a.top_k most covisible images
    and with its a.seq neighbours in name (capture) order.
    """
    from scipy import sparse as sp

    t0 = time.time()
    frames = read_images(a.prior)
    if len(frames) < 2:
        sys.exit("select-pairs: at least two prior images are required")
    if a.top_k == 0 and a.seq == 0:
        sys.exit("select-pairs: --top-k and --seq cannot both be zero")
    cams = read_cameras(a.prior)
    pts = read_points(a.prior, a.max_points)
    if len(pts) < 1000:
        sys.exit(f"select-pairs: only {len(pts)} points in {a.prior}/points3D.bin; use --matcher sequential")
    rows, cols = [], []
    for ci, f in enumerate(frames):
        model, width, height, params = cams[f["camera_id"]]
        fx, fy, cx, cy = pinhole(model, params)
        pc = pts @ f["R"].T + f["t"]
        z = pc[:, 2]
        front = np.nonzero(z > 0.05)[0]
        u = fx * pc[front, 0] / z[front] + cx
        v = fy * pc[front, 1] / z[front] + cy
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        idx, zz = front[inside], z[front][inside]
        if len(idx) == 0:
            continue
        per_row = int(width) // a.cell + 1
        cell = (v[inside] // a.cell).astype(np.int64) * per_row + (u[inside] // a.cell).astype(np.int64)
        nearest = np.full(cell.max() + 1, np.inf)
        np.minimum.at(nearest, cell, zz)
        visible = idx[zz <= nearest[cell] * 1.05 + 0.02]
        rows.append(np.full(len(visible), ci))
        cols.append(visible)
    if not rows:
        sys.exit("select-pairs: no prior points are visible in any camera; check the prior coordinate convention "
                 "and units, or use --matcher sequential")
    vis = sp.csr_matrix((np.ones(sum(map(len, rows)), dtype=np.float32),
                         (np.concatenate(rows), np.concatenate(cols))), shape=(len(frames), len(pts)))
    covis = (vis @ vis.T).toarray()
    np.fill_diagonal(covis, 0)
    pairs = set()
    for i in range(len(frames)):
        for j in np.argsort(-covis[i])[:a.top_k]:
            if covis[i, j] > 0:
                pairs.add((min(i, j), max(i, j)))
        for j in range(max(0, i - a.seq), min(len(frames), i + a.seq + 1)):
            if j != i:
                pairs.add((min(i, j), max(i, j)))
    if not pairs:
        sys.exit("select-pairs: no matching pairs were selected; increase --seq or use --matcher sequential")
    with open(a.out, "w") as f:
        for i, j in sorted(pairs):
            f.write(f"{frames[i]['name']} {frames[j]['name']}\n")
    isolated = int((covis.max(1) == 0).sum())
    print(f"{len(frames)} images, {len(pts)} points, {vis.sum(1).mean():.0f} visible/image, "
          f"{len(pairs)} pairs (top {a.top_k} + seq ±{a.seq}), {isolated} images without covisible partners, "
          f"{time.time() - t0:.1f} s")


def cmd_init_model(a):
    """Text model: prior poses, one a.camera_model camera with the median prior intrinsics and no distortion.

    The database camera is switched to the same model and params: point_triangulator requires them to agree.
    Matching must already be done, as two-view verification uses the database camera.
    """
    frames = read_images(a.prior)
    cams = read_cameras(a.prior)
    model = a.camera_model
    if model not in PINHOLE_DISTORTION:
        sys.exit(f"init-model: camera model {model} is not supported, use one of {', '.join(PINHOLE_DISTORTION)}")
    db = sqlite3.connect(a.database)
    db_cams = db.execute("SELECT camera_id, width, height FROM cameras").fetchall()
    if len(db_cams) != 1:
        sys.exit(f"init-model: expected one camera in the database, found {len(db_cams)}")
    cam_id, width, height = db_cams[0]
    ids = dict(db.execute("SELECT name, image_id FROM images"))
    frame_of = dict(db.execute("SELECT data_id, frame_id FROM frame_data WHERE sensor_type = 0"))
    rig_of = dict(db.execute("SELECT frame_id, rig_id FROM frames"))
    missing = [f["name"] for f in frames if f["name"] not in ids]
    if missing:
        sys.exit(f"init-model: {len(missing)} prior frames are not in the database, e.g. {missing[:3]}")

    # The prior intrinsics belong to the image size of the prior model; scale them to the images used.
    k = np.array([[*pinhole(m, p), w, h] for m, w, h, p in (cams[f["camera_id"]] for f in frames)])
    fx, fy, cx, cy, aw, ah = np.median(k, axis=0)
    sx, sy = width / aw, height / ah
    fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy

    params = np.array([fx, fy, cx, cy, *PINHOLE_DISTORTION[model]], dtype=np.float64)
    model_id = next(i for i, (name, _) in CAMERA_MODELS.items() if name == model)
    db.execute("UPDATE cameras SET model = ?, params = ? WHERE camera_id = ?", (model_id, params.tobytes(), cam_id))
    db.commit()

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "cameras.txt"), "w") as f:
        f.write(f"{cam_id} {model} {width} {height} " + " ".join(repr(float(v)) for v in params) + "\n")
    with open(os.path.join(a.out, "rigs.txt"), "w") as f:
        for rig_id in sorted(set(rig_of[frame_of[ids[fr['name']]]] for fr in frames)):
            f.write(f"{rig_id} 1 CAMERA {cam_id}\n")
    with open(os.path.join(a.out, "frames.txt"), "w") as ff, open(os.path.join(a.out, "images.txt"), "w") as fi:
        for fr in frames:
            image_id = ids[fr["name"]]
            frame_id = frame_of[image_id]
            pose = " ".join(repr(float(v)) for v in (*fr["q"], *fr["t"]))
            ff.write(f"{frame_id} {rig_of[frame_id]} {pose} 1 CAMERA {cam_id} {image_id}\n")
            fi.write(f"{image_id} {pose} {cam_id} {fr['name']}\n\n")
    Path(a.out, "points3D.txt").write_text("")
    print(f"init model: {len(frames)} frames, {model} {width}x{height} fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")


def cmd_write_pose_priors(a):
    """Position (Cartesian, prior frame) and gravity (camera frame) priors for colmap mappers."""
    frames = read_images(a.prior)
    if a.world_down == "auto":
        # Cameras are usually held roughly upright: the image-down axis (+y) of the
        # cameras, averaged in world coordinates, points along gravity.
        down = np.mean([f["R"].T @ np.array([0.0, 1.0, 0.0]) for f in frames], axis=0)
    else:
        down = np.array([float(v) for v in a.world_down.split(",")])
    if down.shape != (3,) or not np.isfinite(down).all():
        raise ValueError("world down must be three finite components, e.g. --world-down 0,1,0")
    norm = np.linalg.norm(down)
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("world down has no stable direction; pass a nonzero --world-down x,y,z")
    down /= norm
    db = sqlite3.connect(a.database)
    ids = dict(db.execute("SELECT name, image_id FROM images"))
    cam_of = dict(db.execute("SELECT image_id, camera_id FROM images"))
    db.execute("DELETE FROM pose_priors")
    cov = (np.eye(3) * a.std ** 2).astype(np.float64)
    written = 0
    for f in frames:
        if f["name"] not in ids:
            continue
        image_id = ids[f["name"]]
        center = -f["R"].T @ f["t"]
        gravity = f["R"] @ down
        db.execute("INSERT INTO pose_priors (corr_data_id, corr_sensor_id, corr_sensor_type, position, "
                   "position_covariance, coordinate_system, gravity) VALUES (?, ?, 0, ?, ?, 1, ?)",
                   (image_id, cam_of[image_id], center.astype(np.float64).tobytes(), cov.T.tobytes(),
                    gravity.astype(np.float64).tobytes()))
        written += 1
    db.commit()
    print(f"{written} pose priors (std {a.std} m), world down = {np.round(down, 3).tolist()}")


def cmd_restore_prior_poses(a):
    """Gives the frames of a.model with fewer than a.min_obs observations their prior pose, in place.

    Bundle adjustment has no fixed gauge, so it moves the whole model (by 21 cm on scan_20261103), but it
    cannot move a frame without observations. model_aligner then maps such a frame along with the rest,
    and it ends up off by the model's drift (24 cm). The prior pose is mapped into the model with the
    similarity between the prior and the model camera centers of the other frames.
    """
    prior = {f["name"]: f for f in read_images(a.prior)}
    frames = [f for f in read_images(a.model) if f["name"] in prior]
    weak = [f for f in frames if f["n_obs"] < a.min_obs]
    if not weak:
        print(f"all {len(frames)} frames have at least {a.min_obs} observations; no pose restored")
        return
    good = [f["name"] for f in frames if f["n_obs"] >= a.min_obs]
    if len(good) < 3:
        sys.exit(f"restore-prior-poses: only {len(good)} frames have at least {a.min_obs} observations")
    prior_c, model_c = camera_centers(prior.values()), camera_centers(frames)
    scale, r, t = umeyama(np.array([prior_c[n] for n in good]), np.array([model_c[n] for n in good]))
    poses, moved = {}, []
    for f in weak:
        R = prior[f["name"]]["R"] @ r.T
        c = scale * r @ prior_c[f["name"]] + t
        poses[f["image_id"]] = (rot_to_quat(R), -R @ c)
        moved.append(np.linalg.norm(c - model_c[f["name"]]))
    write_poses(a.model, poses)
    listed = ", ".join(f"{f['name']} ({f['n_obs']} obs, moved {d * 100:.1f} cm)" for f, d in zip(weak[:20], moved))
    print(f"{len(weak)}/{len(frames)} frames with fewer than {a.min_obs} observations restored to their prior pose: "
          f"{listed}{', …' if len(weak) > 20 else ''}")


def cmd_report_agreement(a):
    prior = camera_centers(read_images(a.prior))
    frames = {f["name"]: f for f in read_images(a.model)}
    model = camera_centers(frames.values())
    names = sorted(set(prior) & set(model))
    if len(names) < 3:
        sys.exit(f"report-agreement: only {len(names)} frames shared between the prior model and {a.model}")
    p = np.array([prior[n] for n in names])
    m = np.array([model[n] for n in names])
    raw = np.linalg.norm(m - p, axis=1)
    scale, r, t = umeyama(m, p)
    res = np.linalg.norm((scale * (r @ m.T)).T + t - p, axis=1)
    print(f"{len(names)}/{len(prior)} prior frames in the model | distance to the prior camera centers: "
          f"median {np.median(raw) * 100:.1f} cm, max {raw.max() * 100:.1f} cm | after a similarity "
          f"(scale {scale:.4f}, rotation {rotation_deg(r):.2f} deg): median {np.median(res) * 100:.1f} cm, "
          f"p95 {np.percentile(res, 95) * 100:.1f} cm")
    weak = [n for n in names if frames[n]["n_obs"] < a.min_obs]
    print(f"frames with fewer than {a.min_obs} observations (prior pose kept): {len(weak)}"
          + (": " + ", ".join(f"{n} ({frames[n]['n_obs']})" for n in weak[:20]) + (", …" if len(weak) > 20 else "")
             if weak else ""))
    far = [i for i in np.argsort(-res) if res[i] > a.flag_cm / 100]
    print(f"frames more than {a.flag_cm:g} cm from their prior pose after the similarity: {len(far)}"
          + (" | largest: " + ", ".join(f"{names[i]} {res[i] * 100:.1f} cm ({frames[names[i]]['n_obs']} obs)"
                                         for i in far[:10]) if far else ""))


def cmd_merge_points(a):
    """SfM points of a.model plus the points of the prior model, written to a.out.

    The prior points are in the prior frame. The model is aligned to it, but only up to the alignment's
    tolerance, so the points are mapped with the similarity between the prior and the model camera
    centers. They get new ids and empty tracks.
    """
    prior = camera_centers(read_images(a.prior))
    model = camera_centers(read_images(a.model))
    names = sorted(set(prior) & set(model))
    if len(names) < 3:
        sys.exit(f"merge-points: only {len(names)} frames shared between the prior model and {a.model}")
    src, dst = np.array([prior[n] for n in names]), np.array([model[n] for n in names])
    scale, r, t = umeyama(src, dst)
    res = np.linalg.norm((scale * (r @ src.T)).T + t - dst, axis=1)
    sfm = point_records(os.path.join(a.model, "points3D.bin"))
    prior_pts = point_records(os.path.join(a.prior, "points3D.bin"))
    if not prior_pts:
        sys.exit(f"merge-points: no points in {a.prior}/points3D.bin")
    xyz = np.array([struct.unpack_from("<3d", p, 8) for p in prior_pts])
    xyz = (scale * (r @ xyz.T)).T + t
    next_id = max((struct.unpack_from("<Q", p, 0)[0] for p in sfm), default=0) + 1
    added = [struct.pack("<Q3d3BdQ", next_id + i, *xyz[i], *struct.unpack_from("<3B", p, 32), 0.0, 0)
             for i, p in enumerate(prior_pts)]
    with open(a.out, "wb") as f:
        f.write(struct.pack("<Q", len(sfm) + len(added)))
        f.write(b"".join(sfm) + b"".join(added))
    print(f"{len(sfm)} SfM + {len(prior_pts)} prior points -> {a.out} | prior points mapped with scale {scale:.4f}, "
          f"rotation {rotation_deg(r):.2f} deg; camera-center residual median {np.median(res) * 100:.1f} cm")


def cmd_print_camera(a):
    for cid, (model, w, h, params) in read_cameras(a.model).items():
        print(f"camera {cid}: {model} {w}x{h} " + " ".join(f"{v:.6g}" for v in params))


def nonnegative_int(value):
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer greater than or equal to zero") from None
    if result < 0:
        raise argparse.ArgumentTypeError("must be an integer greater than or equal to zero")
    return result


def positive_int(value):
    result = nonnegative_int(value)
    if result == 0:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return result


def positive_float(value):
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero") from None
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    prior_help = "prior COLMAP model directory (images.bin, cameras.bin, points3D.bin)"

    p = sub.add_parser("list-images"); p.add_argument("--prior", required=True, help=prior_help)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_list_images)
    p = sub.add_parser("export-centers"); p.add_argument("--prior", required=True, help=prior_help)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_export_centers)
    p = sub.add_parser("select-pairs"); p.add_argument("--prior", required=True, help=prior_help)
    p.add_argument("--out", required=True)
    p.add_argument("--top-k", type=nonnegative_int, default=25); p.add_argument("--seq", type=nonnegative_int, default=3)
    p.add_argument("--cell", type=positive_int, default=20); p.add_argument("--max-points", type=nonnegative_int, default=400000)
    p.set_defaults(fn=cmd_select_pairs)
    p = sub.add_parser("init-model"); p.add_argument("--prior", required=True, help=prior_help)
    p.add_argument("--database", required=True); p.add_argument("--out", required=True)
    p.add_argument("--camera-model", default="OPENCV", choices=list(PINHOLE_DISTORTION))
    p.set_defaults(fn=cmd_init_model)
    p = sub.add_parser("write-pose-priors"); p.add_argument("--prior", required=True, help=prior_help)
    p.add_argument("--database", required=True)
    p.add_argument("--std", type=positive_float, default=0.05); p.add_argument("--world-down", default="auto")
    p.set_defaults(fn=cmd_write_pose_priors)
    p = sub.add_parser("restore-prior-poses"); p.add_argument("--prior", required=True, help=prior_help)
    p.add_argument("--model", required=True); p.add_argument("--min-obs", type=nonnegative_int, default=20)
    p.set_defaults(fn=cmd_restore_prior_poses)
    p = sub.add_parser("report-agreement"); p.add_argument("--prior", required=True, help=prior_help)
    p.add_argument("--model", required=True)
    p.add_argument("--min-obs", type=nonnegative_int, default=20); p.add_argument("--flag-cm", type=positive_float, default=10.0)
    p.set_defaults(fn=cmd_report_agreement)
    p = sub.add_parser("merge-points"); p.add_argument("--prior", required=True, help=prior_help)
    p.add_argument("--model", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_merge_points)
    p = sub.add_parser("print-camera"); p.add_argument("--model", required=True)
    p.set_defaults(fn=cmd_print_camera)
    a = ap.parse_args()
    try:
        a.fn(a)
    except ValueError as exc:
        sys.exit(f"{a.cmd}: {exc}")


if __name__ == "__main__":
    main()
