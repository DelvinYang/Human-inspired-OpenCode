from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import fit_centerline_from_control_points, resample_arc_length


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidate_by_cluster_id(candidates: list[dict]) -> dict[int, dict]:
    return {int(item["cluster_id"]): item for item in candidates}


def apply_manual_selection(
    candidate_payload: dict[str, Any],
    manual_selection: dict[str, Any] | None = None,
    allow_missing_ids: bool = False,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    candidates = list(candidate_payload.get("candidates", []))
    lookup = candidate_by_cluster_id(candidates)
    source_rejects = list(candidate_payload.get("rejects", []))
    if manual_selection is None:
        return candidates, source_rejects, {
            "selection_source": "none_all_candidates_accepted",
            "accepted_cluster_ids": [int(c["cluster_id"]) for c in candidates],
            "deleted_cluster_ids": [],
            "rejected_cluster_ids": [],
        }

    accepted_ids = [int(x) for x in manual_selection.get("accepted_cluster_ids", [])]
    deleted_ids = [int(x) for x in manual_selection.get("deleted_cluster_ids", [])]
    rejected_ids = [int(x) for x in manual_selection.get("rejected_cluster_ids", [])]
    requested = set(accepted_ids + deleted_ids + rejected_ids)
    missing = sorted(requested.difference(lookup))
    if missing and not allow_missing_ids:
        raise ValueError(f"manual selection references missing cluster ids: {missing}")

    accepted = [lookup[x] for x in accepted_ids if x in lookup]
    rejects = list(source_rejects)
    for cluster_id in rejected_ids:
        if cluster_id in lookup:
            rejects.append(reject_from_candidate(lookup[cluster_id], f"r{len(rejects):03d}"))

    return accepted, rejects, {
        "selection_source": "manual_selection",
        "accepted_cluster_ids": accepted_ids,
        "deleted_cluster_ids": deleted_ids,
        "rejected_cluster_ids": rejected_ids,
        "missing_cluster_ids": missing,
    }


def final_path_from_candidate(candidate: dict, path_id: str) -> dict:
    seed = np.asarray(candidate["candidate_seed"], dtype=np.float64)
    controls = resample_arc_length(seed, 12)
    geom = fit_centerline_from_control_points(controls, n_samples=100, smoothing=0.0)
    return {
        "path_id": path_id,
        "type": "unknown",
        "entry": {"id": ""},
        "exit": {"id": ""},
        "width": 3.5,
        "speed_limit": -1,
        "centerline": geom["centerline"].tolist(),
        "arc_length": geom["arc_length"].tolist(),
        "tangent": geom["tangent"].tolist(),
        "curvature": geom["curvature"].tolist(),
        "source_candidate": {"cluster_id": candidate.get("cluster_id"), "size": candidate.get("size")},
        "generation": "auto_spline_from_cluster_mean_seed_no_raw_uniqueness_enforcement",
    }


def reject_from_candidate(candidate: dict, reject_id: str, reason: str = "manual_layer_mismatch") -> dict:
    return {
        "reject_id": reject_id,
        "reason": reason,
        "assignment_action": "ignore_track",
        "source_cluster_ids": [int(candidate["cluster_id"])],
        "size": int(candidate.get("size", 0)),
        "candidate_seed": candidate["candidate_seed"],
        "member_track_ids": candidate.get("member_track_ids", []),
        "entry": candidate.get("entry", []),
        "exit": candidate.get("exit", []),
        "meta": {
            **dict(candidate.get("meta", {})),
            "manual_reject": True,
            "manual_reject_reason": reason,
        },
    }


def reject_prototype_from_candidate(reject: dict) -> dict:
    seed = np.asarray(reject.get("candidate_seed") or reject.get("centerline"), dtype=np.float64)
    controls = resample_arc_length(seed, 12)
    geom = fit_centerline_from_control_points(controls, n_samples=100, smoothing=0.0)
    return {
        "reject_id": reject["reject_id"],
        "reason": reject.get("reason", "unknown"),
        "assignment_action": reject.get("assignment_action", "ignore_track"),
        "source_cluster_ids": reject.get("source_cluster_ids", []),
        "size": reject.get("size", 0),
        "centerline": geom["centerline"].tolist(),
        "arc_length": geom["arc_length"].tolist(),
        "tangent": geom["tangent"].tolist(),
        "curvature": geom["curvature"].tolist(),
        "member_track_ids": reject.get("member_track_ids", []),
        "generation": "reject_spline_from_cluster_mean_seed",
    }


def reject_payload(dataset: str, scene_id: str, rejects: list[dict]) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "dataset": dataset,
        "source": "raw_dataset",
        "assignment_usage": "match ignore_track prototypes before normal reference paths; drop matched samples",
        "rejects": [reject_prototype_from_candidate(item) for item in rejects],
    }
