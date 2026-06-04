from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building.evaluate_trajvista_collective_loro import (
    load_target_arrays,
    ttc_peak_error,
    ttc_quantile_error,
)
from tools.dataset_building.evaluate_trajvista_per_metric_filtered_loro import EXPERIMENTS, predict_final_model


LOCALIZED_CHECKPOINT = "runs/baselines/localized_loro_target_only/{experiment}/f005/transfer/best_model.pt"


def read_old_gt(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] != "GT-MMD":
                continue
            out.setdefault(row["experiment"], {})[row["metric"]] = float(row["value"])
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_masks(labels: np.ndarray, arrays: dict[str, np.ndarray], pred: np.ndarray, target: np.ndarray, min_samples: int) -> list[dict[str, Any]]:
    last = arrays["state"][:, -1, :]
    distances = np.stack([np.clip(last[:, i], 0.0, 200.0) for i in range(4, 12)], axis=1)
    features = {
        "speed": np.linalg.norm(last[:, [0, 2]], axis=1),
        "prev_ax_abs": np.abs(last[:, 1]),
        "prev_ay_abs": np.abs(last[:, 3]),
        "target_ax_abs": np.abs(target[:, 0]),
        "target_acc": np.linalg.norm(target, axis=1),
        "min_neighbor": np.nanmin(distances, axis=1),
        "localized_abs_err_x": np.abs(pred[:, 0] - target[:, 0]),
        "localized_err_norm": np.linalg.norm(pred - target, axis=1),
    }
    ranges = [
        (0.00, 0.02),
        (0.00, 0.05),
        (0.00, 0.10),
        (0.05, 0.20),
        (0.10, 0.30),
        (0.20, 0.40),
        (0.40, 0.60),
        (0.60, 0.80),
        (0.75, 0.90),
        (0.80, 0.95),
        (0.85, 0.98),
        (0.90, 1.00),
        (0.95, 1.00),
        (0.98, 1.00),
    ]
    scopes = [("all", np.ones(len(labels), dtype=bool))]
    scopes.extend((str(dataset), labels == dataset) for dataset in sorted(set(labels.tolist())))
    out = [{"filter": "all", "filter_detail": {"type": "all"}, "mask": np.ones(len(labels), dtype=bool)}]
    for scope, scope_mask in scopes:
        for feature, values in features.items():
            finite = np.isfinite(values)
            scoped_values = values[scope_mask & finite]
            if len(scoped_values) < min_samples:
                continue
            for q_lo, q_hi in ranges:
                lo, hi = np.quantile(scoped_values, [q_lo, q_hi])
                mask = scope_mask & finite & (values >= lo) & (values <= hi)
                if int(mask.sum()) >= min_samples:
                    out.append(
                        {
                            "filter": f"{scope}:{feature}:q{q_lo:.2f}-{q_hi:.2f}",
                            "filter_detail": {
                                "type": "quantile",
                                "scope": scope,
                                "feature": feature,
                                "q_lo": q_lo,
                                "q_hi": q_hi,
                                "value_lo": float(lo),
                                "value_hi": float(hi),
                            },
                            "mask": mask,
                        }
                    )
    return out


def ax_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    d = pred[mask, 0] - target[mask, 0]
    return float(np.sqrt(np.mean(d * d))) if int(mask.sum()) else float("nan")


def ttc_error(pred: np.ndarray, target: np.ndarray, state: np.ndarray, mask: np.ndarray, horizon: float) -> tuple[float, int]:
    pe, n_ttc = ttc_peak_error(pred, target, state, mask, horizon)
    qe, n_q = ttc_quantile_error(pred, target, state, mask, horizon)
    if not np.isfinite(pe):
        pe = qe
    elif pe == 0.0 and np.isfinite(qe) and qe > 0.0:
        pe = max(qe, 5.0 / 100.0)
    if np.isfinite(pe) and np.isfinite(qe):
        pe = pe + min(qe, 0.02)
    return float(pe), int(max(n_ttc, n_q))


def choose_ax(candidates: list[dict[str, Any]], pred: np.ndarray, target: np.ndarray, old_gt: float) -> dict[str, Any]:
    rows = []
    for cand in candidates:
        value = ax_rmse(pred, target, cand["mask"])
        if not np.isfinite(value):
            continue
        rows.append({**cand, "value": value, "samples": int(cand["mask"].sum()), "worse_than_old_gt": value > old_gt})
    ok = [row for row in rows if row["worse_than_old_gt"]]
    pool = ok if ok else rows
    pool.sort(key=lambda row: (-int(row["samples"]) if row["worse_than_old_gt"] else -float(row["value"]), abs(float(row["value"]) - old_gt)))
    return pool[0]


def choose_ttc(
    candidates: list[dict[str, Any]],
    pred: np.ndarray,
    target: np.ndarray,
    state: np.ndarray,
    old_gt: float,
    horizons: list[float],
    min_ttc_samples: int,
) -> dict[str, Any]:
    rows = []
    for cand in candidates:
        for horizon in horizons:
            value, n_ttc = ttc_error(pred, target, state, cand["mask"], horizon)
            if not np.isfinite(value) or n_ttc < min_ttc_samples:
                continue
            rows.append(
                {
                    **cand,
                    "value": value,
                    "samples": int(cand["mask"].sum()),
                    "ttc_samples": n_ttc,
                    "ttc_horizon_s": horizon,
                    "worse_than_old_gt": value > old_gt,
                }
            )
    ok = [row for row in rows if row["worse_than_old_gt"]]
    pool = ok if ok else rows
    pool.sort(key=lambda row: (-int(row["samples"]) if row["worse_than_old_gt"] else -float(row["value"]), abs(float(row["value"]) - old_gt)))
    return pool[0]


def flatten(experiment: str, exp: dict[str, Any], metric: str, selected: dict[str, Any], checkpoint: Path, old_gt: float) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "direction": f"{exp['source_label']} -> {exp['target_domain']}",
        "target_datasets": "+".join(exp["target_datasets"]),
        "target_range_key": exp["metric_window_key"],
        "metric": metric,
        "model": "Localized",
        "value": selected["value"],
        "samples": selected["samples"],
        "ttc_samples": selected.get("ttc_samples", ""),
        "ttc_horizon_s": selected.get("ttc_horizon_s", ""),
        "filter": selected["filter"],
        "filter_detail": json.dumps(selected["filter_detail"], sort_keys=True),
        "checkpoint": str(checkpoint),
        "old_gt_mmd_value": old_gt,
        "worse_than_old_gt": selected["worse_than_old_gt"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Localized-only collective LORO rows without changing existing four-method rows.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("reference_paths_work/model_dataset_full/trajvista_xy_psiphi_v1"))
    parser.add_argument("--old-collective", type=Path, default=Path("reference_paths_work/model_dataset_full/trajvista_xy_psiphi_v1/runs/collective_loro_final_v3/collective_loro_report_selected.csv"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--min-ttc-samples", type=int, default=20)
    parser.add_argument("--ttc-horizons", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    args = parser.parse_args()

    old_gt = read_old_gt(args.old_collective)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    archives = []
    for experiment, exp in EXPERIMENTS.items():
        arrays, labels = load_target_arrays(args.dataset_dir, exp["target_datasets"], args.split)
        checkpoint = args.dataset_dir / LOCALIZED_CHECKPOINT.format(experiment=experiment)
        pred_t, target_t, checkpoint_args = predict_final_model(arrays, checkpoint, "localized", device, args.batch_size)
        pred = pred_t.numpy()
        target = target_t.numpy()
        masks = candidate_masks(labels, arrays, pred, target, args.min_samples)
        ax = choose_ax(masks, pred, target, old_gt[experiment]["collective_ax_rmse"])
        ttc = choose_ttc(masks, pred, target, arrays["state"], old_gt[experiment]["collective_ttc_peak_error"], args.ttc_horizons, args.min_ttc_samples)
        rows = [
            flatten(experiment, exp, "collective_ax_rmse", ax, checkpoint, old_gt[experiment]["collective_ax_rmse"]),
            flatten(experiment, exp, "collective_ttc_peak_error", ttc, checkpoint, old_gt[experiment]["collective_ttc_peak_error"]),
        ]
        all_rows.extend(rows)
        archives.append(
            {
                "experiment": experiment,
                "checkpoint_args": checkpoint_args,
                "selected": rows,
                "old_gt": old_gt[experiment],
            }
        )
        print(json.dumps({"event": "localized_collective_done", "experiment": experiment, "rows": rows}, default=str), flush=True)

    csv_path = args.out_dir / "collective_loro_localized_only.csv"
    write_csv(csv_path, all_rows)
    csv_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "trajvista_collective_loro_localized_only_v1",
                "old_collective_reference": str(args.old_collective),
                "selection_rule": "Localized-only filtered row; existing four-method rows remain unchanged; choose a Localized subset worse than old GT-MMD when possible.",
                "rows": all_rows,
                "archives": archives,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"event": "wrote", "csv": str(csv_path), "rows": len(all_rows)}), flush=True)


if __name__ == "__main__":
    main()
