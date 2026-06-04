#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cultural_align.reference_paths.cache import discover_cached_scenes, read_scene_cache
from cultural_align.reference_paths.cluster import mine_candidates, write_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine reference-path candidate centerlines from cached scenes.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--n-points", type=int, default=50)
    parser.add_argument("--min-duration-s", type=float, default=3.0)
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--shape-normalized", action="store_true")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--require-boundary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = discover_cached_scenes(args.cache_root, dataset=args.dataset)
    if args.scene_id:
        rows = [row for row in rows if row["scene_id"] == args.scene_id]
    outputs = []
    for row in rows:
        scene = read_scene_cache(row["path"])
        candidates, meta = mine_candidates(
            scene,
            n_points=args.n_points,
            min_duration_s=args.min_duration_s,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            shape_normalized=args.shape_normalized,
            max_tracks=args.max_tracks,
            require_boundary=args.require_boundary,
        )
        out = args.work_dir / "reference_paths" / "candidates" / scene.dataset / f"{scene.scene_id}.json"
        write_candidates(out, scene, candidates, meta)
        outputs.append({"dataset": scene.dataset, "scene_id": scene.scene_id, "n_candidates": len(candidates), "path": str(out)})
        print(json.dumps({"event": "scene_done", **outputs[-1]}), flush=True)
    print(json.dumps({"event": "done", "n_scenes": len(outputs), "outputs": outputs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
