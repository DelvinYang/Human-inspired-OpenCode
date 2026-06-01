from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache import read_scene_cache
from .cluster import is_vehicle_track
from .geometry import arc_lengths, compute_tangent, resample_arc_length


STATUS_CODES = {
    "assigned": 1,
    "rejected_ignore_track": 2,
    "unassigned": 3,
    "excluded_non_vehicle": 4,
    "invalid_too_short": 5,
}


@dataclass(frozen=True)
class Thresholds:
    segment_mean: float
    lateral_mean: float
    lateral_p90: float

    def scaled(self, factor: float) -> "Thresholds":
        return Thresholds(
            segment_mean=self.segment_mean * factor,
            lateral_mean=self.lateral_mean * factor,
            lateral_p90=self.lateral_p90 * factor,
        )


@dataclass
class DirectedPath:
    path_id: str
    centerline: np.ndarray
    invert: bool
    source: dict[str, Any]
    reject_id: str | None = None
    arc_length: np.ndarray | None = None
    segments: np.ndarray | None = None
    segment_starts: np.ndarray | None = None
    segment_lengths: np.ndarray | None = None
    segment_units: np.ndarray | None = None

    def __post_init__(self) -> None:
        center = np.asarray(self.centerline, dtype=np.float64)
        if self.invert:
            center = center[::-1].copy()
        self.centerline = _dedupe_polyline(center)
        self.arc_length = arc_lengths(self.centerline)
        if len(self.centerline) >= 2:
            starts = self.centerline[:-1]
            ends = self.centerline[1:]
            seg = ends - starts
            seg_len = np.linalg.norm(seg, axis=1)
            good = seg_len > 1e-9
            self.segment_starts = starts[good]
            self.segments = seg[good]
            self.segment_lengths = seg_len[good]
            self.segment_units = self.segments / self.segment_lengths[:, None]
        else:
            self.segment_starts = np.zeros((0, 2), dtype=np.float64)
            self.segments = np.zeros((0, 2), dtype=np.float64)
            self.segment_lengths = np.zeros(0, dtype=np.float64)
            self.segment_units = np.zeros((0, 2), dtype=np.float64)


@dataclass
class MatchResult:
    path_id: str | None
    invert: bool | None
    score: float | None
    segment_mean_distance: float | None
    lateral_mean: float | None
    lateral_p90: float | None
    frechet_distance: float | None
    direction_mean_dot: float | None
    monotonic_ratio: float | None
    reject_id: str | None = None
    passed: bool = False
    ambiguous: bool = False
    score_gap: float | None = None
    best_alt_path_id: str | None = None
    reason: str | None = None
    variant: DirectedPath | None = None
    match_mode: str = "shape"
    lane_change_count: int | None = None
    initial_lateral_mean: float | None = None
    initial_lateral_p90: float | None = None
    bundle_lateral_mean: float | None = None
    bundle_lateral_p90: float | None = None
    lane_change_path_sequence: list[str] | None = None


def default_thresholds(unit: str, dataset: str | None = None) -> Thresholds:
    if unit == "pixel":
        return Thresholds(segment_mean=120.0, lateral_mean=80.0, lateral_p90=160.0)
    if dataset == "NGSIM":
        return Thresholds(segment_mean=10.0, lateral_mean=3.5, lateral_p90=8.0)
    return Thresholds(segment_mean=6.0, lateral_mean=3.5, lateral_p90=8.0)


def load_paths(path_file: Path) -> list[dict]:
    if not path_file.exists():
        return []
    return json.loads(path_file.read_text(encoding="utf-8")).get("paths", [])


def load_rejects(reject_file: Path) -> list[dict]:
    if not reject_file.exists():
        return []
    return json.loads(reject_file.read_text(encoding="utf-8")).get("rejects", [])


def directed_path_variants(paths: list[dict], rejects: bool = False) -> list[DirectedPath]:
    variants: list[DirectedPath] = []
    for item in paths:
        centerline = np.asarray(item.get("centerline", []), dtype=np.float64)
        if len(centerline) < 2:
            continue
        path_id = item.get("path_id") or item.get("reject_id")
        if path_id is None:
            continue
        source = {
            "path_id": item.get("path_id"),
            "reject_id": item.get("reject_id"),
            "source_candidate": item.get("source_candidate", {}),
            "reason": item.get("reason"),
        }
        for invert in (False, True):
            variants.append(
                DirectedPath(
                    path_id=str(path_id),
                    centerline=centerline,
                    invert=invert,
                    source=source,
                    reject_id=str(item.get("reject_id")) if rejects and item.get("reject_id") is not None else None,
                )
            )
    return variants


def assign_scene(
    scene,
    paths: list[dict],
    rejects: list[dict],
    reject_scale: float = 0.7,
    refine_top_k: int = 4,
) -> tuple[list[dict], dict[str, Any]]:
    thresholds = default_thresholds(scene.unit, scene.dataset)
    path_variants = directed_path_variants(paths)
    reject_variants = directed_path_variants(rejects, rejects=True)
    allow_lane_change_fallback = scene.dataset == "NGSIM"
    assignments: list[dict] = []
    for track in scene.tracks:
        assignments.append(
            assign_track_record(
                track,
                path_variants,
                reject_variants,
                thresholds,
                reject_scale,
                refine_top_k,
                allow_lane_change_fallback=allow_lane_change_fallback,
            )
        )
    summary = summarize_assignments(scene, assignments, len(paths), len(rejects), thresholds, reject_scale)
    return assignments, summary


def assign_track_record(
    track,
    path_variants: list[DirectedPath],
    reject_variants: list[DirectedPath],
    thresholds: Thresholds,
    reject_scale: float,
    refine_top_k: int,
    allow_lane_change_fallback: bool = False,
) -> dict[str, Any]:
    base = {
        "track_id": str(track.track_id),
        "status": None,
        "path_id": None,
        "invert": None,
        "score": None,
        "segment_mean_distance": None,
        "lateral_mean": None,
        "lateral_p90": None,
        "frechet_distance": None,
        "direction_mean_dot": None,
        "monotonic_ratio": None,
        "reject_id": None,
        "best_alt_path_id": None,
        "best_candidate_path_id": None,
        "match_mode": None,
        "lane_change_count": None,
        "initial_lateral_mean": None,
        "initial_lateral_p90": None,
        "bundle_lateral_mean": None,
        "bundle_lateral_p90": None,
        "lane_change_path_sequence": None,
        "score_gap": None,
        "ambiguous": False,
    }
    if not is_vehicle_track(track):
        base["status"] = "excluded_non_vehicle"
        return base

    xy = _valid_xy(track.xy)
    if len(xy) < 2 or _polyline_length(xy) <= 1e-9:
        base["status"] = "invalid_too_short"
        return base
    match_xy = resample_arc_length(xy, 80)

    if reject_variants:
        reject_match = best_match(match_xy, reject_variants, thresholds.scaled(reject_scale), refine_top_k=refine_top_k)
        if reject_match.passed:
            return {**base, **match_to_record(reject_match, "rejected_ignore_track")}

    match = best_match(match_xy, path_variants, thresholds, refine_top_k=refine_top_k)
    if match.passed:
        return {**base, **match_to_record(match, "assigned")}
    if allow_lane_change_fallback:
        lane_change_match = best_initial_path_lane_change_match(match_xy, path_variants, thresholds)
        if lane_change_match is not None:
            return {**base, **match_to_record(lane_change_match, "assigned")}
    return {**base, **match_to_record(match, "unassigned")}


def match_to_record(match: MatchResult, status: str) -> dict[str, Any]:
    path_id = match.path_id if status == "assigned" else None
    best_candidate_path_id = match.path_id if status == "unassigned" else None
    return {
        "status": status,
        "path_id": path_id,
        "invert": match.invert,
        "score": match.score,
        "segment_mean_distance": match.segment_mean_distance,
        "lateral_mean": match.lateral_mean,
        "lateral_p90": match.lateral_p90,
        "frechet_distance": match.frechet_distance,
        "direction_mean_dot": match.direction_mean_dot,
        "monotonic_ratio": match.monotonic_ratio,
        "reject_id": match.reject_id,
        "best_candidate_path_id": best_candidate_path_id,
        "best_alt_path_id": match.best_alt_path_id,
        "match_mode": match.match_mode if status == "assigned" else None,
        "lane_change_count": match.lane_change_count,
        "initial_lateral_mean": match.initial_lateral_mean,
        "initial_lateral_p90": match.initial_lateral_p90,
        "bundle_lateral_mean": match.bundle_lateral_mean,
        "bundle_lateral_p90": match.bundle_lateral_p90,
        "lane_change_path_sequence": match.lane_change_path_sequence,
        "score_gap": match.score_gap,
        "ambiguous": match.ambiguous,
    }


def best_match(
    track_xy: np.ndarray,
    variants: list[DirectedPath],
    thresholds: Thresholds,
    refine_top_k: int = 4,
) -> MatchResult:
    if not variants:
        return MatchResult(None, None, None, None, None, None, None, None, None, reason="no_variants")

    coarse_top_n = min(len(variants), max(12, refine_top_k * 3))
    coarse = sorted(((coarse_variant_score(track_xy, variant), variant) for variant in variants), key=lambda x: x[0])
    candidates = [variant for _, variant in coarse[:coarse_top_n]]
    prelim = [evaluate_variant(track_xy, variant, thresholds, with_frechet=False) for variant in candidates]
    prelim.sort(key=lambda x: math.inf if x.score is None else x.score)
    top = prelim[: max(1, min(refine_top_k, len(prelim)))]
    refined = [evaluate_variant(track_xy, result.variant, thresholds, with_frechet=True) for result in top if result.variant is not None]
    if not refined:
        return prelim[0]
    refined.sort(key=lambda x: math.inf if x.score is None else x.score)
    best = refined[0]
    alt = refined[1] if len(refined) > 1 else (prelim[1] if len(prelim) > 1 else None)
    if alt is not None and best.score is not None and alt.score is not None and math.isfinite(alt.score):
        best.best_alt_path_id = alt.path_id
        best.score_gap = float((alt.score - best.score) / max(abs(best.score), 1e-6))
        best.ambiguous = bool(best.score_gap < 0.10)
    return best


def coarse_variant_score(track_xy: np.ndarray, variant: DirectedPath) -> float:
    track = track_xy[:: max(1, len(track_xy) // 12)]
    path = variant.centerline[:: max(1, len(variant.centerline) // 24)]
    if len(track) == 0 or len(path) == 0:
        return math.inf
    d = np.linalg.norm(track[:, None, :] - path[None, :, :], axis=2).min(axis=1)
    return float(np.mean(d) + 0.3 * np.percentile(d, 90))


def evaluate_variant(
    track_xy: np.ndarray,
    variant: DirectedPath,
    thresholds: Thresholds,
    with_frechet: bool,
) -> MatchResult:
    if variant.segment_starts is None or len(variant.segment_starts) == 0:
        return MatchResult(
            variant.path_id,
            variant.invert,
            math.inf,
            None,
            None,
            None,
            None,
            None,
            None,
            reason="empty_path",
            variant=variant,
        )

    s, lateral, path_tangent = project_points_to_path(track_xy, variant)
    track_tangent = compute_tangent(track_xy)
    dots = np.sum(track_tangent * path_tangent, axis=1)
    finite = np.isfinite(dots)
    direction_mean = float(np.mean(dots[finite])) if np.any(finite) else -1.0
    start_dot = float(dots[0]) if len(dots) else -1.0
    end_dot = float(dots[-1]) if len(dots) else -1.0
    ds = np.diff(s)
    monotonic = float(np.mean(ds >= -1e-6)) if len(ds) else 1.0
    lateral_mean = float(np.mean(lateral))
    lateral_p90 = float(np.percentile(lateral, 90))

    s0 = float(np.percentile(s, 2))
    s1 = float(np.percentile(s, 98))
    if s1 <= s0 + 1e-9:
        s0 = float(np.min(s))
        s1 = float(np.max(s))
    if s1 <= s0 + 1e-9:
        return MatchResult(
            variant.path_id,
            variant.invert,
            math.inf,
            None,
            lateral_mean,
            lateral_p90,
            None,
            direction_mean,
            monotonic,
            variant.reject_id,
            reason="zero_projected_span",
            variant=variant,
        )

    track_rs = resample_arc_length(track_xy, 80)
    local_rs = sample_path_segment(variant.centerline, s0, s1, 80)
    segment_mean = float(np.linalg.norm(track_rs - local_rs, axis=1).mean())

    frechet = discrete_frechet(track_rs[::4], local_rs[::4]) if with_frechet else segment_mean
    unit_scale = max(1.0, thresholds.segment_mean / 6.0)
    direction_penalty = unit_scale * 2.0 * max(0.0, 1.0 - direction_mean)
    monotonic_penalty = unit_scale * 10.0 * max(0.0, 0.95 - monotonic)
    score = float(segment_mean + 0.5 * lateral_p90 + 0.25 * frechet + direction_penalty + monotonic_penalty)

    passed = (
        segment_mean <= thresholds.segment_mean
        and lateral_mean <= thresholds.lateral_mean
        and lateral_p90 <= thresholds.lateral_p90
        and direction_mean >= 0.35
        and start_dot >= -0.25
        and end_dot >= -0.25
        and monotonic >= 0.75
    )
    result = MatchResult(
        path_id=variant.path_id,
        invert=variant.invert,
        score=score,
        segment_mean_distance=segment_mean,
        lateral_mean=lateral_mean,
        lateral_p90=lateral_p90,
        frechet_distance=float(frechet),
        direction_mean_dot=direction_mean,
        monotonic_ratio=monotonic,
        reject_id=variant.reject_id,
        passed=bool(passed),
        reason=None if passed else "threshold_or_gate_failed",
        variant=variant,
    )
    return result


def best_initial_path_lane_change_match(
    track_xy: np.ndarray,
    variants: list[DirectedPath],
    thresholds: Thresholds,
) -> MatchResult | None:
    if len(track_xy) < 8 or not variants:
        return None

    initial_n = max(8, min(len(track_xy), int(math.ceil(len(track_xy) * 0.2))))
    initial_xy = track_xy[:initial_n]
    initial_candidates: list[tuple[float, DirectedPath, float, float, float, float]] = []
    for variant in variants:
        if variant.segment_starts is None or len(variant.segment_starts) == 0:
            continue
        s_init, lateral_init, tangent_init = project_points_to_path(initial_xy, variant)
        dots_init = np.sum(compute_tangent(initial_xy) * tangent_init, axis=1)
        initial_direction = float(np.mean(dots_init[np.isfinite(dots_init)])) if np.any(np.isfinite(dots_init)) else -1.0
        initial_monotonic = float(np.mean(np.diff(s_init) >= -1e-6)) if len(s_init) > 1 else 1.0
        initial_mean = float(np.mean(lateral_init))
        initial_p90 = float(np.percentile(lateral_init, 90))
        if (
            initial_mean <= thresholds.lateral_mean
            and initial_p90 <= thresholds.lateral_p90
            and initial_direction >= 0.70
            and initial_monotonic >= 0.85
        ):
            initial_score = initial_mean + 0.3 * initial_p90 + 2.0 * max(0.0, 1.0 - initial_direction)
            initial_candidates.append((initial_score, variant, initial_mean, initial_p90, initial_direction, initial_monotonic))
    if not initial_candidates:
        return None

    initial_candidates.sort(key=lambda x: x[0])
    initial_score, initial_variant, initial_mean, initial_p90, _, _ = initial_candidates[0]
    same_direction_variants = [variant for _, variant, *_ in initial_candidates if variant.invert == initial_variant.invert]
    if len(same_direction_variants) < 2:
        return None

    bundle = evaluate_lane_bundle(track_xy, same_direction_variants, thresholds)
    if bundle is None:
        return None
    lane_change_count, sequence, bundle_mean, bundle_p90, bundle_direction, bundle_monotonic = bundle
    if lane_change_count < 1:
        return None
    if not (
        bundle_mean <= thresholds.lateral_mean
        and bundle_p90 <= thresholds.lateral_p90
        and bundle_direction >= 0.70
        and bundle_monotonic >= 0.85
    ):
        return None
    if sequence and sequence[0] != initial_variant.path_id:
        sequence = [initial_variant.path_id, *sequence]
        lane_change_count = max(0, len(sequence) - 1)

    full_on_initial = evaluate_variant(track_xy, initial_variant, thresholds, with_frechet=True)
    score = float(initial_score + bundle_mean + 0.5 * bundle_p90 + 1.5 * lane_change_count)
    full_on_initial.score = score
    full_on_initial.path_id = initial_variant.path_id
    full_on_initial.invert = initial_variant.invert
    full_on_initial.passed = True
    full_on_initial.reason = None
    full_on_initial.match_mode = "initial_ref_path_lane_change"
    full_on_initial.lane_change_count = lane_change_count
    full_on_initial.initial_lateral_mean = initial_mean
    full_on_initial.initial_lateral_p90 = initial_p90
    full_on_initial.bundle_lateral_mean = bundle_mean
    full_on_initial.bundle_lateral_p90 = bundle_p90
    full_on_initial.direction_mean_dot = bundle_direction
    full_on_initial.monotonic_ratio = bundle_monotonic
    full_on_initial.lane_change_path_sequence = sequence
    full_on_initial.variant = initial_variant
    return full_on_initial


def evaluate_lane_bundle(
    track_xy: np.ndarray,
    variants: list[DirectedPath],
    thresholds: Thresholds,
) -> tuple[int, list[str], float, float, float, float] | None:
    laterals = []
    dots = []
    monotonic_flags = []
    path_ids = []
    track_tangent = compute_tangent(track_xy)
    for variant in variants:
        if variant.segment_starts is None or len(variant.segment_starts) == 0:
            continue
        s, lateral, path_tangent = project_points_to_path(track_xy, variant)
        dot = np.sum(track_tangent * path_tangent, axis=1)
        finite = np.isfinite(dot)
        if not np.any(finite):
            continue
        mean_dot = float(np.mean(dot[finite]))
        monotonic = float(np.mean(np.diff(s) >= -1e-6)) if len(s) > 1 else 1.0
        if mean_dot < 0.70 or monotonic < 0.85:
            continue
        laterals.append(lateral)
        dots.append(dot)
        monotonic_flags.append(np.diff(s, prepend=s[0]) >= -1e-6)
        path_ids.append(variant.path_id)
    if len(laterals) < 2:
        return None

    lateral_stack = np.stack(laterals, axis=1)
    nearest_idx = np.argmin(lateral_stack, axis=1)
    rows = np.arange(len(track_xy))
    bundle_lateral = lateral_stack[rows, nearest_idx]
    dot_stack = np.stack(dots, axis=1)
    monotonic_stack = np.stack(monotonic_flags, axis=1)
    bundle_dots = dot_stack[rows, nearest_idx]
    bundle_monotonic_flags = monotonic_stack[rows, nearest_idx]
    sequence = compress_path_sequence([path_ids[int(i)] for i in nearest_idx])
    lane_change_count = max(0, len(sequence) - 1)
    return (
        int(lane_change_count),
        sequence,
        float(np.mean(bundle_lateral)),
        float(np.percentile(bundle_lateral, 90)),
        float(np.mean(bundle_dots[np.isfinite(bundle_dots)])),
        float(np.mean(bundle_monotonic_flags)),
    )


def compress_path_sequence(path_ids: list[str], min_run: int = 3) -> list[str]:
    runs: list[tuple[str, int]] = []
    for path_id in path_ids:
        if runs and runs[-1][0] == path_id:
            runs[-1] = (path_id, runs[-1][1] + 1)
        else:
            runs.append((path_id, 1))
    filtered = [path_id for path_id, count in runs if count >= min_run]
    if not filtered and runs:
        filtered = [max(runs, key=lambda x: x[1])[0]]
    compressed: list[str] = []
    for path_id in filtered:
        if not compressed or compressed[-1] != path_id:
            compressed.append(path_id)
    return compressed


def project_points_to_path(points: np.ndarray, path: DirectedPath) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = path.segment_starts
    seg = path.segments
    seg_len = path.segment_lengths
    seg_unit = path.segment_units
    assert starts is not None and seg is not None and seg_len is not None and seg_unit is not None and path.arc_length is not None
    rel = points[:, None, :] - starts[None, :, :]
    denom = np.maximum(seg_len**2, 1e-12)
    u = np.clip(np.sum(rel * seg[None, :, :], axis=2) / denom[None, :], 0.0, 1.0)
    proj = starts[None, :, :] + u[:, :, None] * seg[None, :, :]
    dist = np.linalg.norm(points[:, None, :] - proj, axis=2)
    idx = np.argmin(dist, axis=1)
    rows = np.arange(len(points))
    s = path.arc_length[idx] + u[rows, idx] * seg_len[idx]
    lateral = dist[rows, idx]
    tangent = seg_unit[idx]
    return s.astype(np.float64), lateral.astype(np.float64), tangent.astype(np.float64)


def sample_path_segment(centerline: np.ndarray, s0: float, s1: float, n_points: int) -> np.ndarray:
    s = arc_lengths(centerline)
    s0 = float(np.clip(s0, 0.0, s[-1]))
    s1 = float(np.clip(s1, 0.0, s[-1]))
    if s1 <= s0:
        s1 = min(float(s[-1]), s0 + 1e-6)
    target = np.linspace(s0, s1, n_points)
    x = np.interp(target, s, centerline[:, 0])
    y = np.interp(target, s, centerline[:, 1])
    return np.stack([x, y], axis=1)


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    ca = np.empty((n, m), dtype=np.float64)
    ca[0, 0] = d[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, n):
        prev = ca[i - 1]
        row = ca[i]
        for j in range(1, m):
            row[j] = max(min(prev[j], prev[j - 1], row[j - 1]), d[i, j])
    return float(ca[n - 1, m - 1])


def summarize_assignments(
    scene,
    assignments: list[dict],
    n_paths: int,
    n_rejects: int,
    thresholds: Thresholds,
    reject_scale: float,
) -> dict[str, Any]:
    counts = {key: 0 for key in STATUS_CODES}
    for item in assignments:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    vehicle_total = counts.get("assigned", 0) + counts.get("rejected_ignore_track", 0) + counts.get("unassigned", 0)
    assigned = counts.get("assigned", 0)
    rejected = counts.get("rejected_ignore_track", 0)
    ambiguous_assigned = sum(1 for x in assignments if x.get("status") == "assigned" and x.get("ambiguous"))
    invert_assigned = sum(1 for x in assignments if x.get("status") == "assigned" and x.get("invert"))
    match_modes: dict[str, int] = {}
    for item in assignments:
        if item.get("status") == "assigned":
            mode = item.get("match_mode") or "shape"
            match_modes[mode] = match_modes.get(mode, 0) + 1
    failed = [x for x in assignments if x.get("status") == "unassigned"]
    failed_sorted = sorted(failed, key=lambda x: math.inf if x.get("score") is None else float(x["score"]))[:20]
    assigned_ratio = assigned / max(1, vehicle_total)
    usable_ratio = (assigned + rejected) / max(1, vehicle_total)
    ambiguous_ratio = ambiguous_assigned / max(1, assigned)
    invert_ratio = invert_assigned / max(1, assigned)
    return {
        "scene_id": scene.scene_id,
        "dataset": scene.dataset,
        "unit": scene.unit,
        "n_tracks": len(assignments),
        "n_reference_paths": n_paths,
        "n_reject_prototypes": n_rejects,
        "counts": counts,
        "match_modes": match_modes,
        "assigned_ratio": float(assigned_ratio),
        "usable_ratio_assigned_or_rejected": float(usable_ratio),
        "ambiguous_assigned": int(ambiguous_assigned),
        "ambiguous_ratio": float(ambiguous_ratio),
        "invert_ratio": float(invert_ratio),
        "flags": {
            "low_assigned_ratio": bool(assigned_ratio < 0.90),
            "high_ambiguous_ratio": bool(ambiguous_ratio > 0.10),
            "high_invert_ratio": bool(invert_ratio > 0.60),
        },
        "thresholds": {
            "segment_mean": thresholds.segment_mean,
            "lateral_mean": thresholds.lateral_mean,
            "lateral_p90": thresholds.lateral_p90,
            "reject_scale": reject_scale,
        },
        "top_unassigned_by_score": failed_sorted,
    }


def write_assignment_outputs(work_dir: Path, scene, assignments: list[dict], summary: dict[str, Any]) -> tuple[Path, Path, Path]:
    json_path = work_dir / "assignments" / scene.dataset / f"{scene.scene_id}.json"
    npz_path = work_dir / "assignments_npz" / scene.dataset / f"{scene.scene_id}.npz"
    qc_path = work_dir / "assignments_qc" / scene.dataset / f"{scene.scene_id}_summary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene_id": scene.scene_id,
        "dataset": scene.dataset,
        "source": "full_raw_trajectory_scene_level_hard_assignment",
        "assignments": assignments,
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    qc_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    track_ids = np.asarray([x.get("track_id", "") for x in assignments], dtype="U128")
    status = np.asarray([STATUS_CODES.get(x.get("status"), 0) for x in assignments], dtype=np.int16)
    path_ids = np.asarray([x.get("path_id") or "" for x in assignments], dtype="U64")
    reject_ids = np.asarray([x.get("reject_id") or "" for x in assignments], dtype="U64")
    best_candidate_path_ids = np.asarray([x.get("best_candidate_path_id") or "" for x in assignments], dtype="U64")
    invert = np.asarray([bool(x.get("invert")) for x in assignments], dtype=np.bool_)
    score = np.asarray([_float_or_nan(x.get("score")) for x in assignments], dtype=np.float32)
    segment = np.asarray([_float_or_nan(x.get("segment_mean_distance")) for x in assignments], dtype=np.float32)
    lateral_mean = np.asarray([_float_or_nan(x.get("lateral_mean")) for x in assignments], dtype=np.float32)
    lateral_p90 = np.asarray([_float_or_nan(x.get("lateral_p90")) for x in assignments], dtype=np.float32)
    frechet = np.asarray([_float_or_nan(x.get("frechet_distance")) for x in assignments], dtype=np.float32)
    direction = np.asarray([_float_or_nan(x.get("direction_mean_dot")) for x in assignments], dtype=np.float32)
    monotonic = np.asarray([_float_or_nan(x.get("monotonic_ratio")) for x in assignments], dtype=np.float32)
    ambiguous = np.asarray([bool(x.get("ambiguous")) for x in assignments], dtype=np.bool_)
    match_modes = np.asarray([x.get("match_mode") or "" for x in assignments], dtype="U48")
    lane_change_count = np.asarray([int(x.get("lane_change_count") or 0) for x in assignments], dtype=np.int16)
    initial_lateral_mean = np.asarray([_float_or_nan(x.get("initial_lateral_mean")) for x in assignments], dtype=np.float32)
    initial_lateral_p90 = np.asarray([_float_or_nan(x.get("initial_lateral_p90")) for x in assignments], dtype=np.float32)
    bundle_lateral_mean = np.asarray([_float_or_nan(x.get("bundle_lateral_mean")) for x in assignments], dtype=np.float32)
    bundle_lateral_p90 = np.asarray([_float_or_nan(x.get("bundle_lateral_p90")) for x in assignments], dtype=np.float32)
    np.savez(
        npz_path,
        track_ids=track_ids,
        status=status,
        path_ids=path_ids,
        reject_ids=reject_ids,
        best_candidate_path_ids=best_candidate_path_ids,
        invert=invert,
        score=score,
        segment_mean_distance=segment,
        lateral_mean=lateral_mean,
        lateral_p90=lateral_p90,
        frechet_distance=frechet,
        direction_mean_dot=direction,
        monotonic_ratio=monotonic,
        ambiguous=ambiguous,
        match_modes=match_modes,
        lane_change_count=lane_change_count,
        initial_lateral_mean=initial_lateral_mean,
        initial_lateral_p90=initial_lateral_p90,
        bundle_lateral_mean=bundle_lateral_mean,
        bundle_lateral_p90=bundle_lateral_p90,
    )
    return json_path, npz_path, qc_path


def iter_saved_path_files(work_dir: Path, dataset: str | None, scene_id: str | None) -> list[Path]:
    root = work_dir / "reference_paths" / "paths"
    if dataset and scene_id:
        return [root / dataset / f"{scene_id}.json"]
    if dataset:
        return sorted((root / dataset).glob("*.json"))
    return sorted(root.glob("*/*.json"))


def run_assignment(
    cache_root: Path,
    work_dir: Path,
    dataset: str | None = None,
    scene_id: str | None = None,
    reject_scale: float = 0.7,
    refine_top_k: int = 4,
) -> dict[str, Any]:
    scene_summaries = []
    errors = []
    for path_file in iter_saved_path_files(work_dir, dataset, scene_id):
        if not path_file.exists():
            errors.append({"scene_id": scene_id, "dataset": dataset, "error": f"missing paths file: {path_file}"})
            continue
        scene_dataset = path_file.parent.name
        scene_name = path_file.stem
        cache_file = cache_root / scene_dataset / f"{scene_name}.pkl"
        if not cache_file.exists():
            errors.append({"scene_id": scene_name, "dataset": scene_dataset, "error": f"missing scene cache: {cache_file}"})
            continue
        scene = read_scene_cache(cache_file)
        paths = load_paths(path_file)
        rejects = load_rejects(work_dir / "reference_paths" / "rejects" / scene_dataset / f"{scene_name}.json")
        assignments, summary = assign_scene(scene, paths, rejects, reject_scale=reject_scale, refine_top_k=refine_top_k)
        json_path, npz_path, qc_path = write_assignment_outputs(work_dir, scene, assignments, summary)
        scene_summaries.append({**summary, "json_path": str(json_path), "npz_path": str(npz_path), "qc_path": str(qc_path)})
        print(
            f"{scene.dataset}/{scene.scene_id}: assigned={summary['counts'].get('assigned', 0)} "
            f"rejected={summary['counts'].get('rejected_ignore_track', 0)} "
            f"unassigned={summary['counts'].get('unassigned', 0)} "
            f"excluded={summary['counts'].get('excluded_non_vehicle', 0)} "
            f"invalid={summary['counts'].get('invalid_too_short', 0)} "
            f"invert_ratio={summary['invert_ratio']:.3f} ambiguous={summary['ambiguous_ratio']:.3f}",
            flush=True,
        )
    batch = summarize_batch(scene_summaries, errors)
    batch_path = work_dir / "assignments_qc" / "batch_summary.json"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    return batch


def summarize_batch(scene_summaries: list[dict], errors: list[dict]) -> dict[str, Any]:
    totals: dict[str, int] = {key: 0 for key in STATUS_CODES}
    by_dataset: dict[str, dict[str, Any]] = {}
    flagged = []
    for summary in scene_summaries:
        dataset = summary["dataset"]
        row = by_dataset.setdefault(dataset, {"scenes": 0, "tracks": 0, "counts": {key: 0 for key in STATUS_CODES}})
        row["scenes"] += 1
        row["tracks"] += int(summary["n_tracks"])
        for key, value in summary["counts"].items():
            totals[key] = totals.get(key, 0) + int(value)
            row["counts"][key] = row["counts"].get(key, 0) + int(value)
        if any(summary["flags"].values()):
            flagged.append(
                {
                    "dataset": dataset,
                    "scene_id": summary["scene_id"],
                    "assigned_ratio": summary["assigned_ratio"],
                    "ambiguous_ratio": summary["ambiguous_ratio"],
                    "invert_ratio": summary["invert_ratio"],
                    "flags": summary["flags"],
                }
            )
    return {
        "n_scenes": len(scene_summaries),
        "totals": totals,
        "by_dataset": by_dataset,
        "flagged_scenes": flagged,
        "errors": errors,
    }


def _valid_xy(xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float64)
    return arr[np.all(np.isfinite(arr), axis=1)]


def _dedupe_polyline(points: np.ndarray) -> np.ndarray:
    points = _valid_xy(points)
    if len(points) <= 1:
        return points
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9])
    return points[keep]


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(arc_lengths(points)[-1])


def _float_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign all raw trajectories to saved reference paths.")
    parser.add_argument("--cache-root", required=True, help="SceneRaw pkl cache root.")
    parser.add_argument("--work-dir", required=True, help="Reference path work directory.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--reject-scale", type=float, default=0.7)
    parser.add_argument("--refine-top-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch = run_assignment(
        cache_root=Path(args.cache_root),
        work_dir=Path(args.work_dir),
        dataset=args.dataset,
        scene_id=args.scene_id,
        reject_scale=args.reject_scale,
        refine_top_k=args.refine_top_k,
    )
    print(json.dumps({"n_scenes": batch["n_scenes"], "totals": batch["totals"], "errors": batch["errors"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
