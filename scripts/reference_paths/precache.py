#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cultural_align.reference_paths.cache import build_scene_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable SceneRaw pkl caches from local raw datasets.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Top-level root containing raw dataset folders.")
    parser.add_argument("--cache-root", type=Path, required=True, help="Output cache root.")
    parser.add_argument("--dataset", default=None, help="Optional dataset name: HighD, inD, CitySim, sinD, NGSIM, or DJI.")
    parser.add_argument("--scene-id", default=None, help="Optional exact scene id.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = build_scene_cache(args.dataset_root, args.cache_root, dataset=args.dataset, scene_id=args.scene_id, force=args.force)
    print(json.dumps({"event": "done", "n_scenes": len(written), "files": [str(p) for p in written]}, indent=2))


if __name__ == "__main__":
    main()
