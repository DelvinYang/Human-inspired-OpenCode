#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cultural_align.training.engine import train_transfer  # noqa: E402


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the released metatype on the inD demo subset.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/demo_transfer_from_cn_us_metatype.yaml"))
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    config = load_config(resolve(args.config))
    dataset_config = load_config(resolve(config["dataset_config"]))
    output_dir = resolve(config.get("output_dir", "artifacts/demo_ind_transfer_from_cn_us_metatype"))
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    training = dict(config.get("training", {}))
    dataset_root = dataset_config["processed_root"]
    common = {
        **{key: value for key, value in training.items() if key not in {"calibrate_epochs", "finetune_epochs"}},
        "seed": config.get("seed", 20260515),
        "dataset_dir": dataset_root,
        "target_dataset_dir": dataset_root,
        "target_datasets": [dataset_config.get("dataset", "inD")],
        "source_checkpoint": str(resolve(config["source_checkpoint"])),
    }
    calibrate_dir = output_dir / "calibrate_w"
    train_transfer({
        **common,
        "phase": "calibrate_w",
        "epochs": int(training.get("calibrate_epochs", 1)),
        "out_dir": str(calibrate_dir),
    })
    train_transfer({
        **common,
        "phase": "finetune_with_target_w",
        "epochs": int(training.get("finetune_epochs", training.get("epochs", 2))),
        "target_w_path": str(calibrate_dir / "w_best.npy"),
        "out_dir": str(output_dir / "ours_transfer"),
    })


if __name__ == "__main__":
    main()
