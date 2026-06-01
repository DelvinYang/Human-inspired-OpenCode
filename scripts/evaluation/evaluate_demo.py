#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cultural_align.training.engine import evaluate_checkpoint  # noqa: E402


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the demo inD checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/demo_ind.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    config = load_config(resolve(args.config))
    dataset_config = load_config(resolve(config["dataset_config"]))
    output_dir = resolve(config.get("output_dir", "artifacts/demo_ind")) / "ours"
    payload = {
        "checkpoint": str(output_dir / "best_model.pt"),
        "dataset_dir": dataset_config["processed_root"],
        "datasets": [dataset_config.get("dataset", "inD")],
        "out_dir": str(output_dir),
        "eval_batch_size": config.get("training", {}).get("eval_batch_size", 4096),
        "device": config.get("training", {}).get("device", "auto"),
    }
    evaluate_checkpoint(payload)


if __name__ == "__main__":
    main()
