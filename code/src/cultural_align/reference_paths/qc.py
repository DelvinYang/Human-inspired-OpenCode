from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .cluster import filter_candidate_tracks
from .geometry import (
    arc_lengths,
    compute_assignment_coverage,
    has_self_intersection,
    is_too_similar_to_raw_track,
    path_geometry,
    precompute_inter_path_geometry,
)
from .models import SceneRaw


def enrich_path_geometry(path: dict) -> dict:
    geom = path_geometry(np.asarray(path["centerline"], dtype=np.float64))
    path = dict(path)
    path["centerline"] = geom["centerline"].tolist()
    path["arc_length"] = geom["arc_length"].tolist()
    path["tangent"] = geom["tangent"].tolist()
    path["curvature"] = geom["curvature"].tolist()
    return path


def validate_paths(
    scene: SceneRaw,
    paths: list[dict],
    uniqueness_threshold: float = 0.10,
    assignment_threshold: float = 3.5,
    enforce_uniqueness: bool = False,
) -> dict[str, Any]:
    if scene.unit == "pixel" and uniqueness_threshold == 0.10:
        uniqueness_threshold = 1.0
    if scene.unit == "pixel" and assignment_threshold == 3.5:
        assignment_threshold = 20.0
    eligible = filter_candidate_tracks(scene)
    path_reports = []
    valid = True
    for path in paths:
        centerline = np.asarray(path["centerline"], dtype=np.float64)
        if enforce_uniqueness:
            too_similar, min_dist, raw_track_id = is_too_similar_to_raw_track(centerline, eligible, threshold=uniqueness_threshold)
        else:
            too_similar, min_dist, raw_track_id = False, None, None
        s = arc_lengths(centerline)
        geom_ok = bool(len(centerline) >= 2 and np.all(np.diff(s) >= -1e-9) and not has_self_intersection(centerline))
        meta_ok = all(path.get(key) not in (None, "") for key in ("path_id", "type", "width"))
        if (too_similar and enforce_uniqueness) or not geom_ok or not meta_ok:
            valid = False
        path_reports.append(
            {
                "path_id": path.get("path_id"),
                "not_identical_to_raw_track": not too_similar,
                "min_mean_distance_to_raw_track": min_dist,
                "nearest_raw_track_id": raw_track_id,
                "geometry_ok": geom_ok,
                "metadata_ok": meta_ok,
            }
        )
    coverage = compute_assignment_coverage(eligible, paths, distance_threshold=assignment_threshold)
    if coverage["coverage"] < 0.90:
        valid = False
    return {
        "scene_id": scene.scene_id,
        "dataset": scene.dataset,
        "valid": valid,
        "assignment": coverage,
        "paths": path_reports,
        "thresholds": {
            "uniqueness_mean_distance": uniqueness_threshold,
            "assignment_distance": assignment_threshold,
            "uniqueness_enforced": enforce_uniqueness,
        },
    }


def write_paths_and_qc(
    work_dir: str | Path,
    scene: SceneRaw,
    paths: list[dict],
    uniqueness_threshold: float = 0.10,
    assignment_threshold: float = 3.5,
    require_coverage: bool = True,
    enforce_uniqueness: bool = False,
    write_visual: bool = False,
) -> tuple[Path, Path, Path]:
    work_dir = Path(work_dir)
    enriched = [enrich_path_geometry(p) for p in paths]
    report = validate_paths(scene, enriched, uniqueness_threshold, assignment_threshold, enforce_uniqueness=enforce_uniqueness)
    payload = {
        "scene_id": scene.scene_id,
        "dataset": scene.dataset,
        "source": "raw_dataset",
        "paths": _attach_path_qc(enriched, report),
        "qc_summary": {
            "valid": report["valid"],
            "assignment_coverage": report["assignment"]["coverage"],
            "coverage_required_for_save": require_coverage,
            "raw_uniqueness_enforced": enforce_uniqueness,
        },
    }
    path_file = work_dir / "reference_paths" / "paths" / scene.dataset / f"{scene.scene_id}.json"
    qc_file = work_dir / "reference_paths" / "qc" / scene.dataset / f"{scene.scene_id}_report.json"
    inter_file = work_dir / "reference_paths" / "inter_path_geom" / scene.dataset / f"{scene.scene_id}.npz"
    path_file.parent.mkdir(parents=True, exist_ok=True)
    qc_file.parent.mkdir(parents=True, exist_ok=True)
    inter_file.parent.mkdir(parents=True, exist_ok=True)
    hard_path_failure = any(
        ((not p.get("not_identical_to_raw_track", False)) and enforce_uniqueness)
        or (not p.get("geometry_ok", False))
        or (not p.get("metadata_ok", False))
        for p in report.get("paths", [])
    )
    can_save = report["valid"] if require_coverage else not hard_path_failure
    if not can_save:
        raise ValueError(f"QC failed for {scene.scene_id}; refusing to save final paths. Report: {report}")
    path_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    qc_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(inter_file, **precompute_inter_path_geometry(enriched))
    if write_visual:
        from .visualize import write_qc_overlay

        visual_file = work_dir / "reference_paths" / "qc" / scene.dataset / f"{scene.scene_id}_overlay.png"
        write_qc_overlay(visual_file, scene, enriched)
    return path_file, qc_file, inter_file


def _attach_path_qc(paths: list[dict], report: dict[str, Any]) -> list[dict]:
    by_id = {p["path_id"]: p for p in report.get("paths", [])}
    out = []
    for path in paths:
        path = dict(path)
        r = by_id.get(path.get("path_id"), {})
        path["qc"] = {
            "not_identical_to_raw_track": r.get("not_identical_to_raw_track", False),
            "min_mean_distance_to_raw_track": r.get("min_mean_distance_to_raw_track"),
            "nearest_raw_track_id": r.get("nearest_raw_track_id"),
            "assignment_coverage": report.get("assignment", {}).get("coverage"),
        }
        out.append(path)
    return out
