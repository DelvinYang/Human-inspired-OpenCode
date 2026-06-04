#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cultural_align.reference_paths.cache import read_scene_cache
from cultural_align.reference_paths.qc import write_paths_and_qc
from cultural_align.reference_paths.selection import (
    apply_manual_selection,
    final_path_from_candidate,
    load_json,
    reject_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert candidate centerlines into reference path JSON files and QC outputs.")
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--path-type", default="unknown")
    parser.add_argument("--width", type=float, default=3.5)
    parser.add_argument("--manual-selection-file", type=Path, default=None)
    parser.add_argument("--allow-missing-selection-ids", action="store_true")
    parser.add_argument("--assignment-threshold", type=float, default=3.5)
    parser.add_argument("--uniqueness-threshold", type=float, default=0.10)
    parser.add_argument("--require-coverage", action="store_true")
    parser.add_argument("--enforce-uniqueness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.candidate_file)
    dataset = payload["dataset"]
    scene_id = payload["scene_id"]
    scene = read_scene_cache(args.cache_root / dataset / f"{scene_id}.pkl")
    selection_file = args.manual_selection_file or args.work_dir / "reference_paths" / "manual_selection" / dataset / f"{scene_id}.json"
    manual_selection = load_json(selection_file) if selection_file.exists() else None
    accepted, rejects, selection_meta = apply_manual_selection(payload, manual_selection, allow_missing_ids=args.allow_missing_selection_ids)
    paths = [final_path_from_candidate(candidate, f"p{index:03d}") for index, candidate in enumerate(accepted)]
    for path in paths:
        path["type"] = args.path_type
        path["width"] = args.width

    reject_file = args.work_dir / "reference_paths" / "rejects" / dataset / f"{scene_id}.json"
    reject_file.parent.mkdir(parents=True, exist_ok=True)
    reject_file.write_text(json.dumps(reject_payload(dataset, scene_id, rejects), indent=2), encoding="utf-8")
    path_file, qc_file, inter_file = write_paths_and_qc(
        args.work_dir,
        scene,
        paths,
        uniqueness_threshold=args.uniqueness_threshold,
        assignment_threshold=args.assignment_threshold,
        require_coverage=args.require_coverage,
        enforce_uniqueness=args.enforce_uniqueness,
    )
    print(
        json.dumps(
            {
                "event": "done",
                "dataset": dataset,
                "scene_id": scene_id,
                "n_paths": len(paths),
                "selection": selection_meta,
                "paths": str(path_file),
                "qc": str(qc_file),
                "inter_path_geom": str(inter_file),
                "rejects": str(reject_file),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
