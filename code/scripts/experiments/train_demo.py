#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from cultural_align.training.engine import train_supervised  # noqa: E402


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"expected mapping in {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the demo inD PsiPhi model.")
    parser.add_argument("--config", type=Path, default=Path("code/configs/experiments/demo_ind.yaml"))
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    config_path = resolve(args.config)
    config = load_config(config_path)
    dataset_config = load_config(resolve(config["dataset_config"]))
    output_dir = resolve(config.get("output_dir", "code/artifacts/demo_ind"))
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    training = config.get("training", {})
    payload = {
        **training,
        "seed": config.get("seed", 42),
        "dataset_dir": dataset_config["processed_root"],
        "datasets": [dataset_config.get("dataset", "inD")],
        "out_dir": str(output_dir / "ours"),
    }
    train_supervised(payload)


if __name__ == "__main__":
    main()
