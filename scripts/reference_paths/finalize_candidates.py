#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cultural_align.reference_paths.cache import read_scene_cache
from cultural_align.reference_paths.qc import write_paths_and_qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert candidate centerlines into reference path JSON files and QC outputs.")
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--path-type", default="vehicle_path")
    parser.add_argument("--width", type=float, default=3.5)
    parser.add_argument("--assignment-threshold", type=float, default=3.5)
    parser.add_argument("--uniqueness-threshold", type=float, default=0.10)
    parser.add_argument("--require-coverage", action="store_true")
    parser.add_argument("--enforce-uniqueness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.candidate_file.read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    scene_id = payload["scene_id"]
    scene = read_scene_cache(args.cache_root / dataset / f"{scene_id}.pkl")
    paths = []
    for index, candidate in enumerate(payload.get("candidates", [])):
        paths.append(
            {
                "path_id": f"p{index:03d}",
                "type": args.path_type,
                "width": args.width,
                "centerline": candidate["candidate_seed"],
                "source_candidate": {
                    "cluster_id": candidate.get("cluster_id"),
                    "size": candidate.get("size"),
                    "persistence": candidate.get("persistence"),
                },
            }
        )

    reject_file = args.work_dir / "reference_paths" / "rejects" / dataset / f"{scene_id}.json"
    reject_file.parent.mkdir(parents=True, exist_ok=True)
    rejects = []
    for item in payload.get("rejects", []):
        reject = dict(item)
        if "centerline" not in reject and "candidate_seed" in reject:
            reject["centerline"] = reject["candidate_seed"]
        rejects.append(reject)
    reject_file.write_text(json.dumps({"dataset": dataset, "scene_id": scene_id, "rejects": rejects}, indent=2), encoding="utf-8")
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
