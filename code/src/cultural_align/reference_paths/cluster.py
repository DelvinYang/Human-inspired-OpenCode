from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import arc_lengths, mean_resampled_distance, resample_arc_length
from .models import CandidatePath, SceneRaw, TrackRaw


VEHICLE_TYPES = {"car", "truck", "bus", "vehicle", "motorcycle", "van"}
CANDIDATE_SEED_GENERATION = "cluster_mean_resampled_average_v8_prune_overlap_keep_longest"


def is_vehicle_track(track: TrackRaw) -> bool:
    agent = (track.agent_type or track.meta.get("agent_type") or track.meta.get("class") or "vehicle")
    agent = str(agent).strip().lower()
    return agent in VEHICLE_TYPES or agent in {"2", "3"}


def filter_candidate_tracks(
    scene: SceneRaw,
    min_duration_s: float = 3.0,
    vehicles_only: bool = True,
    require_boundary: bool = False,
) -> list[TrackRaw]:
    min_frames = max(2, int(round(scene.frame_rate * min_duration_s)))
    tracks = []
    for tr in scene.tracks:
        if vehicles_only and not is_vehicle_track(tr):
            continue
        if len(tr.xy) < min_frames:
            continue
        if not np.all(np.isfinite(np.asarray(tr.xy, dtype=np.float64))):
            continue
        if require_boundary and not _looks_boundary_to_boundary(scene, tr):
            continue
        tracks.append(tr)
    return tracks


def _looks_boundary_to_boundary(scene: SceneRaw, track: TrackRaw, margin_ratio: float = 0.08) -> bool:
    all_xy = np.concatenate([t.xy for t in scene.tracks if len(t.xy) >= 2], axis=0)
    lo = np.nanpercentile(all_xy, 1, axis=0)
    hi = np.nanpercentile(all_xy, 99, axis=0)
    margin = (hi - lo) * margin_ratio
    start = track.xy[0]
    end = track.xy[-1]
    start_boundary = np.any((start <= lo + margin) | (start >= hi - margin))
    end_boundary = np.any((end <= lo + margin) | (end >= hi - margin))
    return bool(start_boundary and end_boundary and np.linalg.norm(end - start) > np.linalg.norm(hi - lo) * 0.15)


def mine_candidates(
    scene: SceneRaw,
    n_points: int = 50,
    min_duration_s: float = 3.0,
    min_cluster_size: int | None = None,
    min_samples: int = 5,
    shape_normalized: bool = False,
    max_tracks: int | None = None,
    require_boundary: bool = False,
) -> tuple[list[CandidatePath], dict[str, Any]]:
    tracks = filter_candidate_tracks(scene, min_duration_s=min_duration_s, require_boundary=require_boundary)
    if max_tracks and len(tracks) > max_tracks:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(len(tracks), size=max_tracks, replace=False).tolist())
        tracks = [tracks[i] for i in idx]
    if not tracks:
        return [], {"scene_id": scene.scene_id, "n_tracks": 0, "method": "none", "warning": "no eligible tracks"}

    resampled = np.stack([resample_arc_length(t.xy, n_points=n_points) for t in tracks], axis=0)
    features = resampled.reshape(len(tracks), -1)
    if shape_normalized:
        centered = resampled - resampled.mean(axis=1, keepdims=True)
        scale = np.maximum(np.linalg.norm(centered.reshape(len(tracks), -1), axis=1, keepdims=True), 1e-9)
        features = centered.reshape(len(tracks), -1) / scale

    min_cluster_size = min_cluster_size or max(10, int(round(0.01 * len(tracks))))
    labels, persistence, method = _cluster_features(features, min_cluster_size=min_cluster_size, min_samples=min_samples)
    raw_candidates = _labels_to_candidates(labels, persistence, tracks, resampled)
    raw_candidate_payload = [candidate_to_dict(c) for c in raw_candidates]
    candidates, rejects = split_short_ignore_candidates(scene, raw_candidates)
    candidates, duplicate_pruned, duplicate_prune_meta = prune_overlapping_candidates(scene, candidates)
    postprocess_meta = {
        "enabled": True,
        "mode": "manual_delete_with_short_ignore_reject_and_overlap_prune",
        "reject_actions": {"short_ignore_turn": "ignore_track"},
        "duplicate_prune": duplicate_prune_meta,
    }
    meta = {
        "scene_id": scene.scene_id,
        "dataset": scene.dataset,
        "n_tracks_raw": len(scene.tracks),
        "n_tracks_eligible": len(tracks),
        "n_clusters_raw": len(raw_candidates),
        "n_clusters": len(candidates),
        "n_rejects": len(rejects),
        "n_outliers": int(np.sum(labels == -1)),
        "method": method,
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "shape_normalized": bool(shape_normalized),
        "candidate_seed_generation": CANDIDATE_SEED_GENERATION,
        "postprocess": postprocess_meta,
        "_rejects": rejects,
        "_raw_candidates": raw_candidate_payload,
        "_duplicate_pruned": duplicate_pruned,
        "label_counts": {str(k): int(v) for k, v in Counter(labels.tolist()).items()},
    }
    return candidates, meta


def split_short_ignore_candidates(scene: SceneRaw, candidates: list[CandidatePath]) -> tuple[list[CandidatePath], list[dict]]:
    accepted: list[CandidatePath] = []
    rejects: list[dict] = []
    for cand in candidates:
        should_reject, reason = _is_short_ignore_turn(cand.medoid, scene)
        if should_reject:
            rejects.append(_candidate_reject(cand, len(rejects), reason, "ignore_track"))
        else:
            accepted.append(cand)
    return accepted, rejects


def prune_overlapping_candidates(
    scene: SceneRaw,
    candidates: list[CandidatePath],
) -> tuple[list[CandidatePath], list[dict], dict[str, Any]]:
    thresholds = _overlap_prune_thresholds(scene)
    if not thresholds["enabled"] or len(candidates) < 2:
        return candidates, [], thresholds

    kept: list[CandidatePath] = []
    pruned: list[dict] = []
    ordered = sorted(candidates, key=lambda c: (-_candidate_length(c), -c.size, c.cluster_id))
    for cand in ordered:
        duplicate_of: CandidatePath | None = None
        match_stats: dict[str, float] = {}
        for base in kept:
            is_duplicate, stats = _is_overlapping_duplicate(cand, base, thresholds)
            if is_duplicate:
                duplicate_of = base
                match_stats = stats
                break
        if duplicate_of is None:
            kept.append(cand)
            continue
        pruned.append(_candidate_pruned_duplicate(cand, duplicate_of, len(pruned), match_stats))

    kept.sort(key=lambda c: c.cluster_id)
    thresholds = dict(thresholds)
    thresholds["n_pruned"] = len(pruned)
    return kept, pruned, thresholds


def _overlap_prune_thresholds(scene: SceneRaw) -> dict[str, Any]:
    enabled = scene.dataset != "CitySim"
    if scene.unit == "pixel":
        mean_distance = 60.0
        p90_distance = 110.0
    else:
        mean_distance = 1.6
        p90_distance = 3.0
    return {
        "enabled": bool(enabled),
        "mode": "keep_longest_near_overlap",
        "mean_nearest_distance": float(mean_distance),
        "p90_nearest_distance": float(p90_distance),
        "direction_dot_min": 0.85,
        "min_length_ratio": 0.55,
        "resample_short_points": 100,
        "resample_long_points": 200,
    }


def _is_overlapping_duplicate(
    a: CandidatePath,
    b: CandidatePath,
    thresholds: dict[str, Any],
) -> tuple[bool, dict[str, float]]:
    a_seed = np.asarray(a.medoid, dtype=np.float64)
    b_seed = np.asarray(b.medoid, dtype=np.float64)
    if len(a_seed) < 4 or len(b_seed) < 4:
        return False, {}

    a_len = _polyline_length(a_seed)
    b_len = _polyline_length(b_seed)
    if min(a_len, b_len) <= 1e-9:
        return False, {}
    short_seed, long_seed = (a_seed, b_seed) if a_len <= b_len else (b_seed, a_seed)
    length_ratio = min(a_len, b_len) / max(a_len, b_len)
    if length_ratio < float(thresholds["min_length_ratio"]):
        return False, {"length_ratio": float(length_ratio)}

    start_dot = _direction_dot(short_seed, long_seed, from_start=True)
    end_dot = _direction_dot(short_seed, long_seed, from_start=False)
    direction_dot = min(start_dot, end_dot)
    if direction_dot < float(thresholds["direction_dot_min"]):
        return False, {
            "length_ratio": float(length_ratio),
            "start_direction_dot": float(start_dot),
            "end_direction_dot": float(end_dot),
        }

    short_rs = resample_arc_length(short_seed, n_points=int(thresholds["resample_short_points"]))
    long_rs = resample_arc_length(long_seed, n_points=int(thresholds["resample_long_points"]))
    dists = np.linalg.norm(short_rs[:, None, :] - long_rs[None, :, :], axis=-1).min(axis=1)
    mean_dist = float(np.mean(dists))
    p90_dist = float(np.percentile(dists, 90))
    stats = {
        "mean_nearest_distance": mean_dist,
        "p90_nearest_distance": p90_dist,
        "length_ratio": float(length_ratio),
        "start_direction_dot": float(start_dot),
        "end_direction_dot": float(end_dot),
    }
    is_duplicate = (
        mean_dist <= float(thresholds["mean_nearest_distance"])
        and p90_dist <= float(thresholds["p90_nearest_distance"])
    )
    return bool(is_duplicate), stats


def _direction_dot(a_seed: np.ndarray, b_seed: np.ndarray, from_start: bool) -> float:
    if from_start:
        a_vec = a_seed[min(5, len(a_seed) - 1)] - a_seed[0]
        b_vec = b_seed[min(5, len(b_seed) - 1)] - b_seed[0]
    else:
        a_vec = a_seed[-1] - a_seed[max(0, len(a_seed) - 6)]
        b_vec = b_seed[-1] - b_seed[max(0, len(b_seed) - 6)]
    denom = max(float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec)), 1e-9)
    return float(np.dot(a_vec, b_vec) / denom)


def _candidate_length(candidate: CandidatePath) -> float:
    return _polyline_length(np.asarray(candidate.medoid, dtype=np.float64))


def _polyline_length(seed: np.ndarray) -> float:
    if len(seed) < 2:
        return 0.0
    return float(arc_lengths(seed)[-1])


def _candidate_pruned_duplicate(
    cand: CandidatePath,
    kept: CandidatePath,
    prune_index: int,
    stats: dict[str, float],
) -> dict[str, Any]:
    return {
        "prune_id": f"d{prune_index:03d}",
        "reason": "overlapping_duplicate_candidate",
        "action": "ui_prune_keep_longest",
        "cluster_id": int(cand.cluster_id),
        "kept_cluster_id": int(kept.cluster_id),
        "size": int(cand.size),
        "kept_size": int(kept.size),
        "length": _candidate_length(cand),
        "kept_length": _candidate_length(kept),
        "stats": {k: float(v) for k, v in stats.items()},
        "member_track_ids": cand.member_track_ids,
    }


def postprocess_candidates(
    scene: SceneRaw,
    candidates: list[CandidatePath],
    duplicate_mean_threshold: float | None = None,
    duplicate_endpoint_threshold: float | None = None,
) -> tuple[list[CandidatePath], list[dict], dict[str, Any]]:
    accepted: list[CandidatePath] = []
    rejects: list[dict] = []
    thresholds = _postprocess_thresholds(
        scene,
        candidates,
        duplicate_mean_threshold=duplicate_mean_threshold,
        duplicate_endpoint_threshold=duplicate_endpoint_threshold,
    )
    for cand in sorted(candidates, key=lambda c: (-c.size, c.cluster_id)):
        u_turn, u_reason = _is_short_ignore_turn(cand.medoid, scene)
        if u_turn:
            rejects.append(_candidate_reject(cand, len(rejects), u_reason, "ignore_track"))
            continue
        duplicate = next((base for base in accepted if _is_duplicate_candidate(base, cand, thresholds)), None)
        if duplicate is not None:
            rejects.append(_candidate_reject(cand, len(rejects), "duplicate_candidate", "covered_by_accepted_path"))
            _merge_candidate(duplicate, cand)
            continue
        accepted.append(cand)
    accepted.sort(key=lambda c: c.cluster_id)
    for new_id, cand in enumerate(accepted):
        original = cand.meta.setdefault("source_cluster_ids", [cand.cluster_id])
        cand.meta["original_cluster_id"] = cand.cluster_id
        cand.meta["source_cluster_ids"] = sorted({int(x) for x in original})
        cand.cluster_id = new_id
    meta = {
        "enabled": True,
        "duplicate_mean_distance_threshold": thresholds["mean"],
        "duplicate_endpoint_distance_threshold": thresholds["endpoint"],
        "duplicate_threshold_source": thresholds["source"],
        "reject_actions": {
            "short_ignore_turn": "ignore_track",
            "duplicate_candidate": "covered_by_accepted_path",
        },
    }
    return accepted, rejects, meta


def _postprocess_thresholds(
    scene: SceneRaw,
    candidates: list[CandidatePath],
    duplicate_mean_threshold: float | None = None,
    duplicate_endpoint_threshold: float | None = None,
) -> dict[str, float | str]:
    arrays = [c.medoid for c in candidates if len(c.medoid) >= 2]
    if arrays:
        pts = np.concatenate(arrays, axis=0)
        lo = np.nanpercentile(pts, 1, axis=0)
        hi = np.nanpercentile(pts, 99, axis=0)
        diag = float(np.linalg.norm(hi - lo))
    else:
        diag = 1.0
    if scene.unit == "pixel":
        defaults = {"mean": max(80.0, 0.025 * diag), "endpoint": max(150.0, 0.05 * diag)}
    else:
        defaults = {"mean": max(2.5, 0.01 * diag), "endpoint": max(5.0, 0.02 * diag)}
    configured = duplicate_mean_threshold is not None or duplicate_endpoint_threshold is not None
    return {
        "mean": float(duplicate_mean_threshold if duplicate_mean_threshold is not None else defaults["mean"]),
        "endpoint": float(duplicate_endpoint_threshold if duplicate_endpoint_threshold is not None else defaults["endpoint"]),
        "source": "scene_config" if configured else "auto_default",
    }


def _is_duplicate_candidate(a: CandidatePath, b: CandidatePath, thresholds: dict[str, float]) -> bool:
    a_seed = np.asarray(a.medoid, dtype=np.float64)
    b_seed = np.asarray(b.medoid, dtype=np.float64)
    if len(a_seed) < 2 or len(b_seed) < 2:
        return False
    entry_dist = float(np.linalg.norm(a_seed[0] - b_seed[0]))
    exit_dist = float(np.linalg.norm(a_seed[-1] - b_seed[-1]))
    if entry_dist > thresholds["endpoint"] or exit_dist > thresholds["endpoint"]:
        return False
    return mean_resampled_distance(a_seed, b_seed, n_points=100) <= thresholds["mean"]


def _is_short_ignore_turn(seed: np.ndarray, scene: SceneRaw) -> tuple[bool, str]:
    if scene.dataset != "CitySim" or scene.scene_id != "CitySim_IntersectionA":
        return False, ""
    seed = np.asarray(seed, dtype=np.float64)
    if len(seed) < 4:
        return False, ""
    length = float(arc_lengths(seed)[-1])
    max_ignore_length = 350.0 if scene.unit == "pixel" else 12.0
    if length > max_ignore_length:
        return False, ""
    displacement = float(np.linalg.norm(seed[-1] - seed[0]))
    direct_ratio = displacement / max(length, 1e-9)
    start_vec = seed[min(5, len(seed) - 1)] - seed[0]
    end_vec = seed[-1] - seed[max(0, len(seed) - 6)]
    denom = max(float(np.linalg.norm(start_vec) * np.linalg.norm(end_vec)), 1e-9)
    direction_dot = float(np.dot(start_vec, end_vec) / denom)
    if direction_dot < -0.35 or direct_ratio < 0.65:
        return True, "short_ignore_turn"
    return False, ""


def _merge_candidate(base: CandidatePath, other: CandidatePath) -> None:
    total = max(1, base.size + other.size)
    base.medoid = (base.medoid * base.size + other.medoid * other.size) / total
    base.size = int(total)
    base.member_track_ids = sorted(set(base.member_track_ids + other.member_track_ids))
    base.entry = base.medoid[0].tolist()
    base.exit = base.medoid[-1].tolist()
    source = set(base.meta.get("source_cluster_ids", [base.cluster_id]))
    source.update(other.meta.get("source_cluster_ids", [other.cluster_id]))
    base.meta["source_cluster_ids"] = sorted(int(x) for x in source)
    base.meta.setdefault("merged_duplicate_cluster_ids", []).append(int(other.cluster_id))


def _candidate_reject(cand: CandidatePath, reject_index: int, reason: str, action: str) -> dict[str, Any]:
    source_ids = cand.meta.get("source_cluster_ids", [cand.cluster_id])
    return {
        "reject_id": f"r{reject_index:03d}",
        "reason": reason,
        "assignment_action": action,
        "source_cluster_ids": [int(x) for x in source_ids],
        "size": int(cand.size),
        "candidate_seed": cand.medoid.tolist(),
        "member_track_ids": cand.member_track_ids,
        "entry": cand.entry,
        "exit": cand.exit,
        "meta": dict(cand.meta),
    }


def _cluster_features(features: np.ndarray, min_cluster_size: int, min_samples: int) -> tuple[np.ndarray, dict[int, float], str]:
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(features)
        persistence = {
            int(label): float(clusterer.cluster_persistence_[i])
            for i, label in enumerate(sorted(set(labels) - {-1}))
            if i < len(clusterer.cluster_persistence_)
        }
        return labels.astype(int), persistence, "hdbscan"
    except ImportError:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import pairwise_distances

        sample_size = min(len(features), 1000)
        if sample_size < len(features):
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(features), size=sample_size, replace=False)
            sample = features[sample_idx]
        else:
            sample = features
        dist = pairwise_distances(sample)
        nonzero = dist[dist > 0]
        eps = float(np.percentile(nonzero, 10)) if nonzero.size else 1.0
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(features)
        return labels.astype(int), {}, "dbscan_fallback_missing_hdbscan_sampled_eps"


def _labels_to_candidates(
    labels: np.ndarray,
    persistence: dict[int, float],
    tracks: list[TrackRaw],
    resampled: np.ndarray,
) -> list[CandidatePath]:
    out: list[CandidatePath] = []
    for label in sorted(set(labels.tolist())):
        if label == -1:
            continue
        idx = np.flatnonzero(labels == label)
        if len(idx) == 0:
            continue
        cluster = resampled[idx]
        seed = cluster.mean(axis=0)
        member_tracks = [tracks[i].track_id for i in idx.tolist()]
        out.append(
            CandidatePath(
                cluster_id=int(label),
                size=int(len(idx)),
                persistence=persistence.get(int(label)),
                medoid=seed,
                member_track_ids=member_tracks,
                entry=seed[0].tolist(),
                exit=seed[-1].tolist(),
                meta={"candidate_seed_only": True, "seed_generation": CANDIDATE_SEED_GENERATION},
            )
        )
    return out


def _medoid_index(cluster: np.ndarray) -> int:
    flat = cluster.reshape(len(cluster), -1)
    dist = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    return int(np.argmin(dist.sum(axis=1)))


def candidate_to_dict(candidate: CandidatePath) -> dict[str, Any]:
    return {
        "cluster_id": candidate.cluster_id,
        "size": candidate.size,
        "persistence": candidate.persistence,
        "candidate_seed": candidate.medoid.tolist(),
        "member_track_ids": candidate.member_track_ids,
        "entry": candidate.entry,
        "exit": candidate.exit,
        "meta": candidate.meta,
    }


def candidates_from_json(entries: list[dict]) -> list[CandidatePath]:
    out: list[CandidatePath] = []
    for item in entries:
        seed = np.asarray(item["candidate_seed"], dtype=np.float64)
        out.append(
            CandidatePath(
                cluster_id=int(item["cluster_id"]),
                size=int(item.get("size", 0)),
                persistence=item.get("persistence"),
                medoid=seed,
                member_track_ids=[str(x) for x in item.get("member_track_ids", [])],
                entry=item.get("entry", seed[0].tolist() if len(seed) else []),
                exit=item.get("exit", seed[-1].tolist() if len(seed) else []),
                meta=dict(item.get("meta", {})),
            )
        )
    return out


def candidates_to_json(scene: SceneRaw, candidates: list[CandidatePath], meta: dict[str, Any]) -> dict:
    public_meta = {k: v for k, v in meta.items() if not k.startswith("_")}
    return {
        "scene_id": scene.scene_id,
        "dataset": scene.dataset,
        "source": "raw_dataset",
        "meta": public_meta,
        "candidates": [candidate_to_dict(c) for c in candidates],
        "rejects": meta.get("_rejects", []),
        "raw_candidates": meta.get("_raw_candidates", []),
        "duplicate_pruned": meta.get("_duplicate_pruned", []),
    }


def write_candidates(path: str | Path, scene: SceneRaw, candidates: list[CandidatePath], meta: dict[str, Any]) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidates_to_json(scene, candidates, meta), indent=2), encoding="utf-8")
