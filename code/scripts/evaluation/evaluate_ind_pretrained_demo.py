#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

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
    parser = argparse.ArgumentParser(description="Evaluate the released inD pretrained checkpoint on the inD demo/test split.")
    parser.add_argument("--config", type=Path, default=Path("code/configs/experiments/demo_evaluate_ind_pretrained.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    config = load_config(resolve(args.config))
    dataset_config = load_config(resolve(config["dataset_config"]))
    evaluation = config.get("evaluation", {})
    payload = {
        "checkpoint": str(resolve(config["checkpoint"])),
        "dataset_dir": dataset_config["processed_root"],
        "datasets": [dataset_config.get("dataset", "inD")],
        "out_dir": str(resolve(config.get("out_dir", "results/demo_ind_pretrained_eval"))),
        "eval_batch_size": evaluation.get("eval_batch_size", 4096),
        "device": evaluation.get("device", "auto"),
    }
    evaluate_checkpoint(payload)


if __name__ == "__main__":
    main()
