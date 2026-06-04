#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

CODE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from cultural_align.data.build_ind import build  # noqa: E402


DEFAULTS = {
    "max_samples": 0,
    "shard_size": 1000,
    "t_hist": 12,
    "stride": 12,
    "max_radius": 60.0,
    "abs_ax_limit": 10.0,
    "abs_ay_limit": 10.0,
    "delta_ax_limit": 8.0,
    "delta_ay_limit": 8.0,
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_runtime_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else path


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in config: {path}")
    return payload


def namespace_from_config(config: dict) -> SimpleNamespace:
    merged = {**DEFAULTS, **config}
    required = ["raw_root", "assignment_dir", "processed_root"]
    missing = [key for key in required if key not in merged]
    if missing:
        raise KeyError(f"missing required config keys: {', '.join(missing)}")
    return SimpleNamespace(
        raw_root=resolve_runtime_path(merged["raw_root"]),
        assignment_dir=resolve_runtime_path(merged["assignment_dir"]),
        out_dir=resolve_runtime_path(merged["processed_root"]),
        max_samples=int(merged["max_samples"]),
        shard_size=int(merged["shard_size"]),
        t_hist=int(merged["t_hist"]),
        stride=int(merged["stride"]),
        max_radius=float(merged["max_radius"]),
        abs_ax_limit=float(merged["abs_ax_limit"]),
        abs_ay_limit=float(merged["abs_ay_limit"]),
        delta_ax_limit=float(merged["delta_ax_limit"]),
        delta_ay_limit=float(merged["delta_ay_limit"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the included inD demo training tensors.")
    parser.add_argument("--config", type=Path, default=Path("code/configs/datasets/ind_demo.yaml"))
    parser.add_argument("--clean", action="store_true", help="Remove the configured processed_root before building.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    ns = namespace_from_config(load_config(config_path))
    os.chdir(PROJECT_ROOT)
    if args.clean and ns.out_dir.exists():
        shutil.rmtree(ns.out_dir)
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build(ns)
    print(json.dumps({"event": "demo_summary", "samples": summary["samples"], "split_counts": summary["split_counts"]}))


if __name__ == "__main__":
    main()
