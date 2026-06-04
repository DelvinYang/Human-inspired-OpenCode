#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cultural_align.training.engine import evaluate_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on NPZ test splits.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--domain", choices=["DE", "US", "CN", "INTERACTION_EXT", "SIX_DATASETS"])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--max-test-samples-per-dataset", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--rollout-mode", choices=["semi", "teacher"], default="semi")
    parser.add_argument("--min-trajectory-steps", type=int, default=50)
    parser.add_argument("--cr-rel-threshold", type=float, default=0.05)
    parser.add_argument("--cr-abs-threshold", type=float, default=2.0)
    parser.add_argument("--max-trajectories-per-dataset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    evaluate_checkpoint(vars(parse_args()))


if __name__ == "__main__":
    main()
