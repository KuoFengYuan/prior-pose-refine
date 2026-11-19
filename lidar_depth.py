"""Registered camera-z depth from fable-gs-capture iPhone maps and continuous Mid-360 packets."""
from collections import deque
from itertools import islice
import heapq
import filecmp
import fcntl
import os
import json
from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
from scipy.spatial.transform import Rotation, Slerp


def child(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.name:
        raise ValueError(f"expected a relative capture path, got {name!r}")
    return root / path


def write_png(path, samples):
    """Grayscale or RGB 16-bit PNG. PNG stores samples and chunk fields big-endian."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    h, w = samples.shape[:2]
    channels = 1 if samples.ndim == 2 else 3
    rows = np.zeros((h, 1 + w * channels * 2), dtype=np.uint8)
    rows[:, 1:] = samples.astype(">u2").view(np.uint8).reshape(h, -1)
    header = struct.pack(">2I5B", w, h, 16, 0 if channels == 1 else 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                + chunk(b"IDAT", zlib.compress(rows.tobytes(), 3)) + chunk(b"IEND", b""))


def distorted_pixels(camera, x, y):
    """Forward projection of pinhole rays into the original RGB image."""
    model, _, _, p = camera
    fx, fy, cx, cy = p[:4]
    r2 = x*x + y*y
    if model == "OPENCV":
        k1, k2, p1, p2 = p[4:]
        radial = 1 + r2 * (k1 + r2 * k2)
        u = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        v = y*radial + 2*p2*x*y + p1*(r2 + 2*y*y)
    elif model in {"OPENCV_FISHEYE", "RAD_TAN_THIN_PRISM_FISHEYE"}:
        radius = np.sqrt(r2)
        angle = np.arctan(radius)
        n = 4 if model == "OPENCV_FISHEYE" else 6
        radial = np.polynomial.polynomial.polyval(angle*angle, [1, *p[4:4+n]])
        radial *= np.divide(angle, radius, out=np.ones_like(angle), where=radius > 0)
        u, v = x * radial, y * radial
        if n == 6:
            p0, p1, s0, s1, s2, s3 = p[10:]
            r2 = u*u + v*v
            du = 2*p1*u*v + p0*(r2 + 2*u*u) + r2*(s0 + s1*r2)
            dv = 2*p0*u*v + p1*(r2 + 2*v*v) + r2*(s2 + s3*r2)
            u, v = u + du, v + dv
    elif model == "PINHOLE":
        u, v = x, y
    else:
        raise ValueError(f"unsupported refined RGB camera {model}")
    return fx*u + cx, fy*v + cy


def training_grid(camera):
    model, width, height, params = camera
    if model != "PINHOLE":
        raise ValueError("depth export needs the undistorted PINHOLE camera in dataset/sparse/0")
    w = min(width, 512)
    h = max(1, round(height*w/width))
    return (h, w), np.array(params) * [w/width, h/height, w/width, h/height]


def iphone_lookup(raw, shape, k, dh, dw):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    u, v = distorted_pixels(raw, (x + 0.5 - k[2])/k[0], (y + 0.5 - k[3])/k[1])
    ix = np.floor(u * dw/raw[1]).astype(int)
    iy = np.floor(v * dh/raw[2]).astype(int)
    inside = (ix >= 0) & (ix < dw) & (iy >= 0) & (iy < dh)
    return iy.clip(0, dh-1), ix.clip(0, dw-1), inside


def iphone_depth(scan, row, lookup):
    shape = row["depthHeight"], row["depthWidth"]
    z = np.fromfile(child(scan/"depth", row["depthFile"]), "<f4").reshape(shape)
    conf = np.fromfile(child(scan/"depth", row["confidenceFile"]), "u1").reshape(shape)
    valid = np.isfinite(z) & (z > 0.1) & (z <= 65.535) & ((conf == 1) | (conf == 2))
    # Mixed foreground/background pixels give poor supervision on depth edges.
    near = minimum_filter(np.where(valid, z, np.inf), size=3)
    far = maximum_filter(np.where(valid, z, 0), size=3)
    valid &= far - near <= 0.01 + 0.03*z
    y, x, inside = lookup
    return np.where(inside & valid[y, x], z[y, x], 0)


POINT = np.dtype([("xyz", "<i4", 3), ("reflectivity", "u1"), ("tag", "u1")])


def packets(paths):
    for path in paths:
        with path.open("rb") as f:
            if f.read(8) != b"MID360R1":
                raise ValueError(f"{path.name}: expected a MID360R1 recording")
            while header := f.read(12):
                if len(header) != 12:
                    raise ValueError(f"{path.name}: truncated record")
                _, size, receipt = struct.unpack("<HHd", header)
                data = f.read(size)
                if (size < 36 or len(data) != size or data[0] != 0
                        or struct.unpack_from("<H", data, 1)[0] != size or not np.isfinite(receipt)):
                    raise ValueError(f"{path.name}: invalid Livox packet")
                if data[10] == 0:  # IMU
                    continue
                interval, count = struct.unpack_from("<HH", data, 3)
                if data[10] != 1 or not count or size != 36 + count*POINT.itemsize:
                    raise ValueError(f"{path.name}: expected Cartesian32 Mid-360 returns (type 1)")
                stamp = struct.unpack_from("<Q", data, 28)[0] * 1e-9
                yield stamp, receipt, interval*1e-7, count, data[36:]


def ordered_packets(paths):
    """Restore acquisition order across a short burst of reordered UDP packets."""
    pending = []
    newest = -np.inf
    for seq, packet in enumerate(packets(paths)):
        stamp = packet[0]
        if stamp < newest - 0.1:
            raise ValueError("Mid-360 clock reset or late packet exceeding 100 ms in one segment")
        newest = max(newest, stamp)
        heapq.heappush(pending, (stamp, seq, packet))
        while pending[0][0] < newest - 0.1:
            yield heapq.heappop(pending)[2]
    while pending:
        yield heapq.heappop(pending)[2]


def clock_fit(paths):
    """Fit offset and drift to the earliest packet arrivals in two-second windows."""
    best = {}
    for stamp, receipt, duration, _, _ in packets(paths):
        lag = receipt - duration - stamp
        key = int(stamp // 2)
        if key not in best or lag < best[key][1]:
            best[key] = (stamp, lag)
    if not best:
        raise ValueError("Mid-360 recording contains no Cartesian returns")
    stamps, lags = np.asarray(list(best.values())).T
    slope = 0.0
    offset = lags.min()
    if np.ptp(stamps) >= 2:
        slope, offset = np.polyfit(stamps - stamps[0], lags, 1)
    return stamps[0], slope, offset


class Trajectory:
    def __init__(self, path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.t = np.array([row["t"] for row in rows])
        m = np.array([row["m"] for row in rows]).reshape(-1, 3, 4)
        if len(rows) < 2 or not np.isfinite(m).all() or not np.isfinite(self.t).all():
            raise ValueError("Mid-360 requires a finite continuous ARKit trajectory")
        self.center = m[:, :, 3]
        self.valid = np.array([row.get("n", True) for row in rows], dtype=bool)
        # ARKit uses right/up/back. The camera-LiDAR extrinsic uses right/down/front.
        self.rotation = Slerp(self.t, Rotation.from_matrix(m[:, :, :3] @ np.diag([1, -1, -1])))

    def sample(self, times):
        t = np.clip(times, self.t[0], self.t[-1])
        i = np.searchsorted(self.t, t, side="right").clip(1, len(self.t)-1)
        dt = self.t[i] - self.t[i-1]
        alpha = ((t - self.t[i-1])/dt)[..., None]
        c = self.center[i-1] + alpha*(self.center[i] - self.center[i-1])
        valid = (times == t) & (dt < 0.1) & self.valid[i] & self.valid[i-1]
        return self.rotation(t), c, valid


def world_blocks(directory, session, trajectory, ext):
    previous_end = -np.inf
    for segment in session["segments"]:
        paths = [child(directory, s) for s in segment["recording"]["chunks"]]
        origin, drift, offset = clock_fit(paths)
        stream = ordered_packets(paths)
        while batch := list(islice(stream, 1024)):
            stamps, _, durations, counts, payloads = zip(*batch)
            starts = np.asarray(stamps)
            starts = starts + offset + drift*(starts-origin)
            stops = starts + np.array(durations)*(1+drift)
            if starts[0] < previous_end - 0.001:
                raise ValueError("overlapping Mid-360 packet batches or recording segments")
            previous_end = stops[-1]
            counts = np.array(counts)
            indices = np.repeat(np.arange(len(batch)), counts)
            within = np.arange(len(indices)) - np.repeat(np.cumsum(counts)-counts, counts)
            alpha = within / np.maximum(counts[indices]-1, 1)
            pts = np.frombuffer(b"".join(payloads), POINT)
            xyz = pts["xyz"].astype(float)*0.001
            r0, c0, ok0 = trajectory.sample(starts)
            r1, c1, ok1 = trajectory.sample(stops)
            valid = ((pts["tag"] & 0x3f) == 0) & (np.linalg.norm(xyz, axis=1) >= 0.3)
            valid &= ok0[indices] & ok1[indices]
            if not np.any(valid):
                continue
            i, a = indices[valid], alpha[valid]
            p = xyz[valid] @ ext[:3, :3].T + ext[:3, 3]
            # Interpolate each return between poses at the two ends of its sub-ms packet.
            first = np.einsum("nij,nj->ni", r0.as_matrix()[i], p) + c0[i]
            last = np.einsum("nij,nj->ni", r1.as_matrix()[i], p) + c1[i]
            times = starts[i] + a*(stops[i]-starts[i])
            order = np.argsort(times, kind="stable")
            yield times[order], (first + a[:, None]*(last-first))[order].astype(np.float32)


class Mid360:
    def __init__(self, scan):
        directory = scan/"mid360/continuous"
        session = json.loads((directory/"session.json").read_bytes())
        calibration = json.loads((scan/"mid360/calibration.json").read_bytes())
        if not session.get("segments"):
            raise ValueError("Mid-360 session has no recording segments")
        if (not calibration.get("id") or not calibration.get("serial")
                or session.get("calibrationID") != calibration.get("id") or calibration.get("units") != "meters"
                or session.get("serial") != calibration.get("serial")):
            raise ValueError("Mid-360 session and metric calibration do not match")
        ext = calibration["T_cameraCV_lidar"]
        report = directory/"refusion-report.json"
        if report.exists():
            refined = json.loads(report.read_bytes()).get("extrinsic", {})
            if refined.get("status") == "applied":
                ext = refined["T_cameraCV_lidar"]
        ext = np.asarray(ext, dtype=float).reshape(4, 4)
        if (not np.isfinite(ext).all() or not np.allclose(ext[3], [0, 0, 0, 1])
                or not np.allclose(ext[:3, :3].T @ ext[:3, :3], np.eye(3), atol=1e-4)
                or np.linalg.det(ext[:3, :3]) < 0):
            raise ValueError("invalid Mid-360 camera extrinsic")
        self.poses = Trajectory(child(directory, session["poseLog"]))
        self.blocks = world_blocks(directory, session, self.poses, ext)
        self.buffer = deque()

    def project(self, stamp, shape, k):
        lo, hi = stamp - 0.5, stamp + 0.5
        while not self.buffer or self.buffer[-1][0][-1] < hi:
            block = next(self.blocks, None)
            if block is None:
                break
            self.buffer.append(block)
            while self.buffer and self.buffer[0][0][-1] < lo:
                self.buffer.popleft()
        while self.buffer and self.buffer[0][0][-1] < lo:
            self.buffer.popleft()
        nearest = np.full(shape, np.inf, dtype=np.float32)
        rot, center, valid = self.poses.sample(np.array([stamp]))
        if valid[0]:
            for times, points in self.buffer:
                a, b = np.searchsorted(times, [lo, hi])
                p = (points[a:b] - center[0]) @ rot.as_matrix()[0]
                p = p[(p[:, 2] > 0.1) & (p[:, 2] <= 65.535)]
                x = np.floor(p[:, 0]/p[:, 2]*k[0] + k[2]).astype(int)
                y = np.floor(p[:, 1]/p[:, 2]*k[1] + k[3]).astype(int)
                inside = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
                np.minimum.at(nearest, (y[inside], x[inside]), p[inside, 2])
        nearest[~np.isfinite(nearest)] = 0
        return nearest


def mid_surface(mid):
    """Fill short interior gaps between returns on a locally continuous surface."""
    valid = mid > 0
    if not np.any(valid):
        return mid
    near = minimum_filter(np.where(valid, mid, np.inf), size=9, mode="constant", cval=np.inf)
    far = maximum_filter(mid, size=9, mode="constant", cval=0)
    density = uniform_filter(valid.astype(np.float32), size=9, mode="constant")
    mean = uniform_filter(mid, size=9, mode="constant") / np.maximum(density, 1/81)
    # Opposing quadrants must both contain a return within four pixels of the hole.
    occupied = [maximum_filter(valid, size=5, origin=origin, mode="constant")
                for origin in [(-2, -2), (-2, 2), (2, 2), (2, -2)]]
    interior = (occupied[0] & occupied[2]) | (occupied[1] & occupied[3])
    fill = ~valid & interior & (density > 3.5/81) & (far-near <= 0.03 + 0.02*near)
    return np.where(fill, mean, mid)


def fuse(iphone, mid):
    overlap = (iphone > 0) & (mid > 0)
    error = np.median(np.abs(mid[overlap]/iphone[overlap]-1)) if overlap.any() else np.nan
    if iphone.any() and (overlap.sum() < 64 or error > 0.05):
        return iphone, error
    mid = mid_surface(mid)
    # Reject fills across an occlusion boundary of a nearby phone measurement.
    near = minimum_filter(np.where(iphone > 0, iphone, np.inf), size=5)
    far = maximum_filter(iphone, size=5)
    tol = 0.03 + 0.03*mid
    fill = (iphone == 0) & (mid > 0) & ((far == 0) | ((mid <= near+tol) & (mid >= far-tol)))
    return np.where(fill, mid, iphone), error


def camera_normals(z, intrinsics):
    """Fit a plane in inverse depth over each supported 7x7 neighborhood."""
    fx, fy, cx, cy = intrinsics
    y, x = np.ogrid[:z.shape[0], :z.shape[1]]
    x, y = x + 0.5 - cx, y + 0.5 - cy
    mask = (z > 0).astype(np.float64)
    inverse = np.divide(1., z.astype(np.float64), out=np.zeros_like(mask), where=mask > 0)
    density = uniform_filter(mask, size=7, mode="constant")
    denominator = np.maximum(density, 1/49)

    def mean(values):
        return uniform_filter(values, size=7, mode="constant") / denominator

    mx, my, mz = mean(mask*x), mean(mask*y), mean(inverse)
    xx = mean(mask*x*x) - mx*mx
    xy = mean(mask*x*y) - mx*my
    yy = mean(mask*y*y) - my*my
    xz = mean(x*inverse) - mx*mz
    yz = mean(y*inverse) - my*mz
    det = xx*yy - xy*xy
    # On a plane, 1/Z = a*(u-cx) + b*(v-cy) + c.
    a = (yy*xz - xy*yz) / np.maximum(det, 1e-8)
    b = (xx*yz - xy*xz) / np.maximum(det, 1e-8)
    c = mz - a*mx - b*my
    normal = np.stack([fx*a, fy*b, c], axis=-1)
    length = np.linalg.norm(normal, axis=-1)
    variance = mean(inverse*inverse) - mz*mz - a*xz - b*yz
    rms = np.sqrt(np.maximum(variance, 0)) * z*z
    near = minimum_filter(np.where(mask > 0, z, np.inf), size=7)
    far = maximum_filter(z, size=7)
    valid = ((mask > 0) & (density >= 0.6) & (det > 1e-8) & (length > 0)
             & (far-near <= 0.03 + 0.08*z) & (rms <= 0.005 + 0.005*z))
    # The plane coefficients point away from the camera. Flip before RGB encoding.
    normal = np.divide(-normal, length[..., None], out=np.zeros_like(normal), where=valid[..., None])
    return np.rint((np.clip(normal, -1, 1) + 1)*32767.5).astype(np.uint16), int(valid.sum())


def publish_maps(staged, dataset):
    """Publish complete layers. A retry can reuse a matching layer left by a previous export."""
    layers = [staged/"depth", staged/"normals"]
    fd = os.open(dataset, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        for layer in layers:
            dest = dataset/layer.name
            if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
                raise ValueError(f"{dest} is not a local directory")
            if dest.exists():
                expected = {p.relative_to(layer) for p in layer.rglob("*.png")}
                existing = {p.relative_to(dest) for p in dest.rglob("*.png")}
                if expected != existing or not all(filecmp.cmp(layer/p, dest/p, shallow=False) for p in expected):
                    raise ValueError(f"existing {dest} has different maps; use a separate dataset")
        for layer in layers:
            dest = dataset/layer.name
            if layer.is_dir() and not dest.exists():
                layer.rename(dest)
    finally:
        os.close(fd)


def export_depth(scan, model, dataset, read_images, read_cameras):
    meta_path = scan/"capture-meta.json"
    has_mid = (scan/"mid360/continuous/session.json").is_file()
    if not meta_path.exists() or not (has_mid or (scan/"depth").is_dir()):
        print("LiDAR depth export skipped (no supported capture data)")
        return
    meta = json.loads(meta_path.read_bytes())
    if meta.get("app") != "fable-gs-capture":
        print("LiDAR depth export skipped (unrecognized capture format)")
        return
    if dataset.resolve().is_relative_to(scan.resolve()):
        raise ValueError("depth output must be outside the original scan")
    rows = {}
    for line in (scan/"poses.jsonl").read_text().splitlines():
        row = json.loads(line)
        name = row["imageFile"]
        if name in rows:
            raise ValueError(f"duplicate capture image {name}")
        rows[name] = row
    images = read_images(dataset/"sparse/0")
    frames = [f for f in images if f["name"] in rows]
    frames.sort(key=lambda f: rows[f["name"]]["timestamp"])
    if not has_mid and not any(rows[f["name"]].get("depthFile") for f in frames):
        print("LiDAR depth export skipped (no registered frames with sensor depth)")
        return
    raw = read_cameras(model)
    final = read_cameras(dataset/"sparse/0")
    # Undistortion preserves the image and camera IDs.
    raw_ids = {f["name"]: f["camera_id"] for f in read_images(model)}
    mid = Mid360(scan) if has_mid else None
    lookups = {}
    coverage, normal_coverage = [], []
    phone_frames = mid_frames = 0
    with tempfile.TemporaryDirectory(prefix=".depth-", dir=dataset) as tmp:
        staged = Path(tmp)
        print("image\tiphone_pixels\tmid360_fill_pixels\tvalid_pixels\ttotal_pixels\toverlap_median_relative_error\tvalid_normal_pixels")
        for frame in frames:
            name = frame["name"]
            row = rows[name]
            cid = frame["camera_id"]
            shape, k = training_grid(final[cid])
            phone = np.zeros(shape, dtype=np.float32)
            if row.get("depthFile") or row.get("confidenceFile"):
                if not (row.get("depthFile") and row.get("confidenceFile")):
                    raise ValueError(f"{name}: need both depthFile and confidenceFile")
                if meta.get("depthFormat") != "float32_raw_little_endian":
                    raise ValueError(f"{name}: unsupported iPhone depth format")
                key = cid, raw_ids[name], row["depthHeight"], row["depthWidth"]
                if key not in lookups:
                    lookups[key] = iphone_lookup(raw[key[1]], shape, k, *key[2:])
                phone = iphone_depth(scan, row, lookups[key])
            depth, error = phone, np.nan
            if mid:
                depth, error = fuse(phone, mid.project(row["timestamp"], shape, k))
            n_phone, n_valid = np.count_nonzero(phone), np.count_nonzero(depth)
            n_normal = 0
            if n_valid:
                relative = child(Path(), name).with_suffix(".png")
                mm = np.rint(depth*1000).astype(np.uint16)
                normal, n_normal = camera_normals(mm.astype(np.float32)*0.001, k)
                write_png(staged/"depth"/relative, mm)
                write_png(staged/"normals"/relative, normal)
                coverage.append(n_valid/depth.size)
                normal_coverage.append(n_normal/depth.size)
            phone_frames += int(n_phone > 0)
            mid_frames += int(n_valid > n_phone)
            print(f"{name}\t{n_phone}\t{n_valid-n_phone}\t{n_valid}\t{depth.size}\t{error:.5f}\t{n_normal}")
        publish_maps(staged, dataset)

    median = np.median(coverage)*100 if coverage else 0
    normal_median = np.median(normal_coverage)*100 if coverage else 0
    print(f"LiDAR depth and normal maps: {len(coverage)}/{len(images)} registered images; iPhone in {phone_frames}, "
          f"Mid-360 in {mid_frames}; median valid pixels: depth {median:.1f}%, normals {normal_median:.1f}%")
