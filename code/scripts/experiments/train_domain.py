#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from cultural_align.training.engine import train_supervised  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train source-domain or pooled TrajVista models.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--domain", choices=["DE", "US", "CN", "INTERACTION_EXT", "SIX_DATASETS"])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--max-train-samples-per-dataset", type=int, default=0)
    parser.add_argument("--max-val-samples-per-dataset", type=int, default=0)
    parser.add_argument("--max-test-samples-per-dataset", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bc-coef", type=float, default=0.1)
    parser.add_argument("--itd-coef", type=float, default=0.1)
    parser.add_argument("--acc-coef", type=float, default=1.0)
    parser.add_argument("--k-neg", type=int, default=32)
    parser.add_argument("--neg-std", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    train_supervised(vars(parse_args()))


if __name__ == "__main__":
    main()
