from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.interpolate import splprep, splev


def as_xy(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected [N,2] points, got {arr.shape}")
    return arr


def arc_lengths(points: np.ndarray) -> np.ndarray:
    points = as_xy(points)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    deltas = np.diff(points, axis=0)
    seg = np.linalg.norm(deltas, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_arc_length(points: np.ndarray, n_points: int = 50) -> np.ndarray:
    points = as_xy(points)
    if len(points) == 0:
        raise ValueError("cannot resample empty trajectory")
    if len(points) == 1:
        return np.repeat(points, n_points, axis=0)
    s = arc_lengths(points)
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    points = points[keep]
    s = s[keep]
    if len(points) == 1 or s[-1] <= 1e-9:
        return np.repeat(points[:1], n_points, axis=0)
    target = np.linspace(0.0, s[-1], n_points)
    x = np.interp(target, s, points[:, 0])
    y = np.interp(target, s, points[:, 1])
    return np.stack([x, y], axis=1)


def fit_centerline_from_control_points(control_points: np.ndarray, n_samples: int = 100, smoothing: float = 0.0) -> dict:
    control_points = as_xy(control_points)
    if len(control_points) < 2:
        raise ValueError("at least two control points are required")
    if len(control_points) == 2:
        samples = resample_arc_length(control_points, n_samples)
    else:
        k = min(3, len(control_points) - 1)
        try:
            tck, _ = splprep(control_points.T, s=smoothing, k=k)
            u = np.linspace(0.0, 1.0, n_samples)
            samples = np.array(splev(u, tck)).T
        except Exception:
            samples = resample_arc_length(control_points, n_samples)
    return path_geometry(samples)


def path_geometry(centerline: np.ndarray) -> dict:
    centerline = as_xy(centerline)
    s = arc_lengths(centerline)
    tangent = compute_tangent(centerline)
    curvature = compute_curvature(centerline, s)
    return {
        "centerline": centerline,
        "arc_length": s,
        "tangent": tangent,
        "curvature": curvature,
    }


def compute_tangent(points: np.ndarray) -> np.ndarray:
    points = as_xy(points)
    if len(points) < 2:
        return np.zeros_like(points)
    grad = np.gradient(points, axis=0)
    norm = np.linalg.norm(grad, axis=1, keepdims=True)
    return grad / np.maximum(norm, 1e-9)


def compute_curvature(points: np.ndarray, s: np.ndarray | None = None) -> np.ndarray:
    points = as_xy(points)
    if len(points) < 3:
        return np.zeros(len(points), dtype=np.float64)
    if s is None:
        s = arc_lengths(points)
    ds = np.gradient(s)
    ds = np.maximum(ds, 1e-6)
    dx = np.gradient(points[:, 0]) / ds
    dy = np.gradient(points[:, 1]) / ds
    ddx = np.gradient(dx) / ds
    ddy = np.gradient(dy) / ds
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    return (dx * ddy - dy * ddx) / denom


def mean_resampled_distance(a: np.ndarray, b: np.ndarray, n_points: int = 100) -> float:
    aa = resample_arc_length(a, n_points)
    bb = resample_arc_length(b, n_points)
    return float(np.linalg.norm(aa - bb, axis=1).mean())


def min_distance_to_tracks(centerline: np.ndarray, tracks: list, n_points: int = 100) -> tuple[float, str | None]:
    best = math.inf
    best_id = None
    for tr in tracks:
        try:
            d = mean_resampled_distance(centerline, tr.xy, n_points=n_points)
        except ValueError:
            continue
        if d < best:
            best = d
            best_id = tr.track_id
    return float(best), best_id


def is_too_similar_to_raw_track(centerline: np.ndarray, tracks: list, threshold: float = 0.10) -> tuple[bool, float, str | None]:
    best, track_id = min_distance_to_tracks(centerline, tracks)
    return bool(best < threshold), best, track_id


def project_point_to_polyline(point: np.ndarray, polyline: np.ndarray) -> tuple[float, float, int]:
    point = np.asarray(point, dtype=np.float64)
    polyline = as_xy(polyline)
    if len(polyline) == 0:
        return 0.0, math.inf, -1
    dists = np.linalg.norm(polyline - point[None, :], axis=1)
    idx = int(np.argmin(dists))
    s = arc_lengths(polyline)[idx]
    return float(s), float(dists[idx]), idx


def assign_track_to_paths(track_xy: np.ndarray, paths: list[dict], n_points: int = 100) -> tuple[str | None, float]:
    best_id = None
    best_dist = math.inf
    for path in paths:
        centerline = np.asarray(path["centerline"], dtype=np.float64)
        dist = mean_resampled_distance(track_xy, centerline, n_points=n_points)
        if dist < best_dist:
            best_dist = dist
            best_id = path["path_id"]
    return best_id, float(best_dist)


def compute_assignment_coverage(tracks: list, paths: list[dict], distance_threshold: float = 3.5) -> dict:
    if not tracks or not paths:
        return {"coverage": 0.0, "assigned": 0, "total": len(tracks), "distances": []}
    n_points = 100
    path_samples = np.stack(
        [resample_arc_length(np.asarray(path["centerline"], dtype=np.float64), n_points).astype(np.float32) for path in paths],
        axis=0,
    )
    distances: list[float] = []
    batch_size = 512
    for start in range(0, len(tracks), batch_size):
        batch = tracks[start : start + batch_size]
        track_samples = np.stack([resample_arc_length(tr.xy, n_points).astype(np.float32) for tr in batch], axis=0)
        diff = track_samples[:, None, :, :] - path_samples[None, :, :, :]
        mean_dist = np.linalg.norm(diff, axis=-1).mean(axis=-1)
        distances.extend(np.min(mean_dist, axis=1).astype(np.float64).tolist())
    assigned = int(np.sum(np.asarray(distances, dtype=np.float64) <= distance_threshold))
    return {
        "coverage": float(assigned / max(1, len(tracks))),
        "assigned": int(assigned),
        "total": int(len(tracks)),
        "mean_distance": float(np.mean(distances)) if distances else math.inf,
        "p95_distance": float(np.percentile(distances, 95)) if distances else math.inf,
        "distances": distances,
    }


def has_self_intersection(points: np.ndarray) -> bool:
    points = as_xy(points)
    if len(points) < 4:
        return False

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def intersects(a, b, c, d):
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    for i in range(len(points) - 1):
        for j in range(i + 2, len(points) - 1):
            if j == i + 1:
                continue
            if intersects(points[i], points[i + 1], points[j], points[j + 1]):
                return True
    return False


def precompute_inter_path_geometry(paths: list[dict]) -> dict[str, np.ndarray]:
    k = len(paths)
    p = max([len(path["centerline"]) for path in paths], default=0)
    t_kl = np.zeros((k, k, p, 4), dtype=np.float32)
    if k == 0 or p == 0:
        return {"T_kl": t_kl}
    centers = [resample_arc_length(np.asarray(path["centerline"], dtype=np.float64), p) for path in paths]
    tangents = [compute_tangent(c) for c in centers]
    lengths = [arc_lengths(c) for c in centers]
    for a in range(k):
        for b in range(k):
            for i in range(p):
                point = centers[a][i]
                dists = np.linalg.norm(centers[b] - point[None, :], axis=1)
                j = int(np.argmin(dists))
                rel = centers[b][j] - point
                cross = tangents[a][i, 0] * rel[1] - tangents[a][i, 1] * rel[0]
                signed_d = dists[j] * (1.0 if cross >= 0 else -1.0)
                dot = np.clip(float(np.dot(tangents[a][i], tangents[b][j])), -1.0, 1.0)
                dtheta = math.acos(dot)
                topo = 0.0 if a == b else 4.0
                t_kl[a, b, i] = [lengths[b][j], signed_d, dtheta, topo]
    return {"T_kl": t_kl}
