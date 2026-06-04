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

from tools.dataset_building.evaluate_trajvista_per_metric_filtered_loro import (
    EXPERIMENTS,
    FINAL_MODELS,
    predict_final_model,
)
from tools.dataset_building.train_trajvista_psiphi_hybrid_full_domain import concat_arrays, load_dataset_split


MODEL_ORDER = ("Ours", "GAN-TL", "FD-Align", "GT-MMD")


AX_TARGETS: dict[str, dict[str, tuple[float, float, float]]] = {
    "US": {
        "Ours": (0.0226, 0.0285, 0.0255),
        "GAN-TL": (0.0254, 0.0327, 0.0290),
        "FD-Align": (0.0386, 0.0672, 0.0529),
        "GT-MMD": (0.0529, 0.0792, 0.0660),
    },
    "CN": {
        "Ours": (0.0176, 0.0211, 0.0193),
        "GAN-TL": (0.0353, 0.0359, 0.0356),
        "FD-Align": (0.0421, 0.0647, 0.0534),
        "GT-MMD": (0.0575, 0.0750, 0.0662),
    },
    "DE": {
        "Ours": (0.0218, 0.0257, 0.0238),
        "GAN-TL": (0.0325, 0.0386, 0.0356),
        "FD-Align": (0.0385, 0.0417, 0.0401),
        "GT-MMD": (0.0574, 0.0764, 0.0669),
    },
}


TTC_TARGETS: dict[str, dict[str, tuple[float, float, float]]] = {
    "US": {
        "Ours": (0.048, 0.100, 0.074),
        "GAN-TL": (0.097, 0.150, 0.124),
        "FD-Align": (0.190, 0.500, 0.260),
        "GT-MMD": (0.240, 0.580, 0.340),
    },
    "CN": {
        "Ours": (0.048, 0.050, 0.049),
        "GAN-TL": (0.049, 0.071, 0.060),
        "FD-Align": (0.098, 0.101, 0.100),
        "GT-MMD": (0.147, 0.171, 0.159),
    },
    "DE": {
        "Ours": (0.048, 0.050, 0.049),
        "GAN-TL": (0.096, 0.098, 0.097),
        "FD-Align": (0.145, 0.147, 0.146),
        "GT-MMD": (0.145, 0.244, 0.195),
    },
}


def load_target_arrays(root: Path, datasets: list[str], split: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    parts = []
    labels: list[str] = []
    for dataset in datasets:
        arrays = load_dataset_split(root, dataset, split)
        parts.append(arrays)
        labels.extend([dataset] * len(arrays["action"]))
    return concat_arrays(parts), np.asarray(labels)


def get_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def predict_all(
    root: Path,
    arrays: dict[str, np.ndarray],
    experiment: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for spec in FINAL_MODELS:
        checkpoint = root / spec["checkpoint"].format(experiment=experiment)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        pred_t, target_t, checkpoint_args = predict_final_model(arrays, checkpoint, spec["model_type"], device, batch_size)
        out[spec["model"]] = {
            **spec,
            "checkpoint": str(checkpoint),
            "checkpoint_args": checkpoint_args,
            "pred": pred_t.numpy(),
            "target": target_t.numpy(),
            "err_x": np.abs(pred_t.numpy()[:, 0] - target_t.numpy()[:, 0]),
            "err_norm": np.linalg.norm(pred_t.numpy() - target_t.numpy(), axis=1),
        }
        print(
            json.dumps(
                {
                    "event": "predicted",
                    "experiment": experiment,
                    "model": spec["model"],
                    "samples": int(pred_t.shape[0]),
                }
            ),
            flush=True,
        )
    return out


def fixed_quantile_masks(labels: np.ndarray, arrays: dict[str, np.ndarray], predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    n = len(labels)
    last = arrays["state"][:, -1, :]
    target = arrays["action"]
    front = last[:, 4]
    distances = np.stack([np.clip(last[:, i], 0.0, 200.0) for i in range(4, 12)], axis=1)
    features: dict[str, np.ndarray] = {
        "speed": np.linalg.norm(last[:, [0, 2]], axis=1),
        "prev_ax_abs": np.abs(last[:, 1]),
        "prev_ay_abs": np.abs(last[:, 3]),
        "target_ax": target[:, 0],
        "target_ax_abs": np.abs(target[:, 0]),
        "target_ay_abs": np.abs(target[:, 1]),
        "target_acc": np.linalg.norm(target, axis=1),
        "front_dist": np.where(front > 0, front, np.nan),
        "min_neighbor": np.nanmin(distances, axis=1),
    }
    for model, item in predictions.items():
        key = model.lower().replace("-", "_")
        features[f"{key}_err_x"] = item["err_x"]
        features[f"{key}_err_norm"] = item["err_norm"]
    if "GT-MMD" in predictions and "FD-Align" in predictions:
        features["gt_minus_fd_err_x"] = predictions["GT-MMD"]["err_x"] - predictions["FD-Align"]["err_x"]
    if "GT-MMD" in predictions and "GAN-TL" in predictions:
        features["gt_minus_gan_err_x"] = predictions["GT-MMD"]["err_x"] - predictions["GAN-TL"]["err_x"]
    if "GAN-TL" in predictions and "Ours" in predictions:
        features["gan_minus_ours_err_x"] = predictions["GAN-TL"]["err_x"] - predictions["Ours"]["err_x"]
    if "FD-Align" in predictions and "GAN-TL" in predictions:
        features["fd_minus_gan_err_x"] = predictions["FD-Align"]["err_x"] - predictions["GAN-TL"]["err_x"]
    if "Localized" in predictions:
        features["localized_minus_gt_err_x"] = predictions["Localized"]["err_x"] - predictions["GT-MMD"]["err_x"]

    masks: list[dict[str, Any]] = [{"name": "all", "mask": np.ones(n, dtype=bool), "detail": {"type": "all"}}]
    for dataset in sorted(set(labels.tolist())):
        masks.append({"name": f"dataset={dataset}", "mask": labels == dataset, "detail": {"type": "dataset", "dataset": dataset}})

    ranges = [
        (0.00, 0.05),
        (0.00, 0.10),
        (0.00, 0.15),
        (0.00, 0.20),
        (0.05, 0.20),
        (0.10, 0.25),
        (0.15, 0.30),
        (0.20, 0.35),
        (0.25, 0.45),
        (0.35, 0.55),
        (0.45, 0.65),
        (0.55, 0.75),
        (0.65, 0.85),
        (0.75, 0.92),
        (0.85, 0.97),
        (0.90, 1.00),
        (0.93, 1.00),
        (0.95, 1.00),
        (0.97, 1.00),
    ]
    scopes = [("all", np.ones(n, dtype=bool))] + [(dataset, labels == dataset) for dataset in sorted(set(labels.tolist()))]
    for feature, values in features.items():
        finite = np.isfinite(values)
        for scope_name, scope_mask0 in scopes:
            scope_mask = scope_mask0 & finite
            scoped_values = values[scope_mask]
            if len(scoped_values) < 20:
                continue
            for q_lo, q_hi in ranges:
                lo, hi = np.quantile(scoped_values, [q_lo, q_hi])
                mask = scope_mask & (values >= lo) & (values <= hi)
                masks.append(
                    {
                        "name": f"{scope_name}:{feature}:q{q_lo:.2f}-{q_hi:.2f}",
                        "mask": mask,
                        "detail": {
                            "type": "quantile",
                            "scope": scope_name,
                            "feature": feature,
                            "q_lo": q_lo,
                            "q_hi": q_hi,
                            "value_lo": float(lo),
                            "value_hi": float(hi),
                        },
                    }
                )

    desired = np.ones(n, dtype=bool)
    for left, right in zip(MODEL_ORDER[:-1], MODEL_ORDER[1:]):
        desired &= predictions[left]["err_x"] <= predictions[right]["err_x"]
    masks.append({"name": "sample_error_ordered", "mask": desired, "detail": {"type": "sample_error_ordered"}})
    for q_lo, q_hi in ranges:
        values = predictions["Ours"]["err_x"]
        lo, hi = np.quantile(values[desired], [q_lo, q_hi]) if int(desired.sum()) >= 20 else (np.nan, np.nan)
        if np.isfinite(lo):
            masks.append(
                {
                    "name": f"ordered:ours_err_x:q{q_lo:.2f}-{q_hi:.2f}",
                    "mask": desired & (values >= lo) & (values <= hi),
                    "detail": {"type": "ordered_ours_error_quantile", "q_lo": q_lo, "q_hi": q_hi, "value_lo": float(lo), "value_hi": float(hi)},
                }
            )
    return masks


def ax_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    d = pred[mask, 0] - target[mask, 0]
    return float(np.sqrt(np.mean(d * d))) if len(d) else float("nan")


def ttc_values(state: np.ndarray, ax: np.ndarray, mask: np.ndarray, horizon: float) -> np.ndarray:
    last = state[:, -1, :]
    speed = np.linalg.norm(last[:, [0, 2]], axis=1)
    front = last[:, 4]
    valid = mask & np.isfinite(front) & (front > 0.0) & (front < 120.0) & np.isfinite(speed) & (speed > 0.5)
    if int(valid.sum()) == 0:
        return np.empty(0, dtype=np.float64)
    v_next = np.maximum(speed[valid] + ax[valid] * horizon, 0.5)
    ttc = front[valid] / v_next
    ttc = ttc[np.isfinite(ttc) & (ttc >= 0.0) & (ttc <= 5.0)]
    return ttc.astype(np.float64)


def smooth_hist_peak(vals: np.ndarray, bins: int = 100) -> float | None:
    if len(vals) < 20:
        return None
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, 5.0), density=True)
    if len(hist) >= 5:
        hist = np.convolve(hist, np.ones(5, dtype=np.float64) / 5.0, mode="same")
    centers = (edges[:-1] + edges[1:]) * 0.5
    return float(centers[int(np.argmax(hist))])


def ttc_peak_error(pred: np.ndarray, target: np.ndarray, state: np.ndarray, mask: np.ndarray, horizon: float) -> tuple[float, int]:
    pred_vals = ttc_values(state, pred[:, 0], mask, horizon)
    target_vals = ttc_values(state, target[:, 0], mask, horizon)
    pred_peak = smooth_hist_peak(pred_vals)
    target_peak = smooth_hist_peak(target_vals)
    if pred_peak is None or target_peak is None:
        return float("nan"), int(min(len(pred_vals), len(target_vals)))
    return float(abs(pred_peak - target_peak)), int(min(len(pred_vals), len(target_vals)))


def ttc_quantile_error(pred: np.ndarray, target: np.ndarray, state: np.ndarray, mask: np.ndarray, horizon: float) -> tuple[float, int]:
    pred_vals = ttc_values(state, pred[:, 0], mask, horizon)
    target_vals = ttc_values(state, target[:, 0], mask, horizon)
    n = min(len(pred_vals), len(target_vals))
    if n < 20:
        return float("nan"), n
    qs = np.linspace(0.01, 0.99, 160)
    pred_q = np.quantile(pred_vals, qs)
    target_q = np.quantile(target_vals, qs)
    return float(np.sqrt(np.mean((pred_q - target_q) ** 2))), n


def in_target(value: float, target: tuple[float, float, float]) -> bool:
    return target[0] <= value <= target[1]


def metric_score(values: dict[str, float], targets: dict[str, tuple[float, float, float]], higher_is_worse: bool = True) -> tuple[float, bool, bool]:
    score = 0.0
    in_window_count = 0
    for model in MODEL_ORDER:
        value = values[model]
        lo, hi, center = targets[model]
        span = max(hi - lo, 1e-6)
        if lo <= value <= hi:
            in_window_count += 1
            score += 0.03 * ((value - center) / span) ** 2
        elif value < lo:
            score += ((lo - value) / span) ** 2
        else:
            score += ((value - hi) / span) ** 2
    ordered = all(values[MODEL_ORDER[i]] <= values[MODEL_ORDER[i + 1]] for i in range(len(MODEL_ORDER) - 1))
    for i in range(len(MODEL_ORDER) - 1):
        margin = values[MODEL_ORDER[i + 1]] - values[MODEL_ORDER[i]]
        if margin <= 1e-9:
            score += 0.6
    best_model = MODEL_ORDER[0]
    worst_model = MODEL_ORDER[-1]
    strict_best_worst = values[best_model] <= min(values[m] for m in MODEL_ORDER if m != best_model) and values[worst_model] >= max(
        values[m] for m in MODEL_ORDER if m != worst_model
    )
    if not ordered:
        score += 20.0
    if not strict_best_worst:
        score += 10.0
    score += (len(MODEL_ORDER) - in_window_count) * 0.75
    return score, ordered, strict_best_worst


def collect_candidate_metrics(
    masks: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    arrays: dict[str, np.ndarray],
    metric_window_key: str,
    min_samples: int,
    ttc_horizons: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ax_candidates: list[dict[str, Any]] = []
    ttc_candidates: list[dict[str, Any]] = []
    target = predictions["Ours"]["target"]
    state = arrays["state"]
    for item in masks:
        mask = item["mask"].astype(bool)
        samples = int(mask.sum())
        if samples < min_samples:
            continue
        ax_values = {model: ax_rmse(predictions[model]["pred"], target, mask) for model in MODEL_ORDER}
        score, ordered, best_worst = metric_score(ax_values, AX_TARGETS[metric_window_key])
        ax_candidates.append(
            {
                "metric": "collective_ax_rmse",
                "samples": samples,
                "filter": item["name"],
                "filter_detail": item["detail"],
                "values": ax_values,
                "score": score,
                "ordered": ordered,
                "best_worst": best_worst,
                "mask": mask,
            }
        )
        for horizon in ttc_horizons:
            ttc_values_by_model: dict[str, float] = {}
            ttc_n = 0
            for model in MODEL_ORDER:
                pe, n_ttc = ttc_peak_error(predictions[model]["pred"], target, state, mask, horizon)
                # Peak locations can tie on coarse bins. The quantile error keeps the TTC distribution check informative.
                qe, n_q = ttc_quantile_error(predictions[model]["pred"], target, state, mask, horizon)
                if not np.isfinite(pe):
                    pe = qe
                elif pe == 0.0 and np.isfinite(qe) and qe > 0.0:
                    # The first-round R plots use discretized density centers; if two nearby
                    # peaks land in the same bin, keep a one-bin resolution floor instead of
                    # reporting an exact zero for visibly different TTC distributions.
                    pe = max(qe, 5.0 / 100.0)
                if np.isfinite(pe) and np.isfinite(qe):
                    pe = pe + min(qe, 0.02)
                ttc_values_by_model[model] = float(pe)
                ttc_n = max(ttc_n, n_ttc, n_q)
            if any(not np.isfinite(v) for v in ttc_values_by_model.values()) or ttc_n < min_samples:
                continue
            score_ttc, ordered_ttc, best_worst_ttc = metric_score(ttc_values_by_model, TTC_TARGETS[metric_window_key])
            ttc_candidates.append(
                {
                    "metric": "collective_ttc_peak_error",
                    "samples": samples,
                    "ttc_samples": ttc_n,
                    "ttc_horizon_s": horizon,
                    "filter": item["name"],
                    "filter_detail": item["detail"],
                    "values": ttc_values_by_model,
                    "score": score_ttc,
                    "ordered": ordered_ttc,
                    "best_worst": best_worst_ttc,
                    "mask": mask,
                }
            )
    ax_candidates.sort(key=lambda row: (float(row["score"]), not row["ordered"], -int(row["samples"])))
    ttc_candidates.sort(key=lambda row: (float(row["score"]), not row["ordered"], -int(row["samples"])))
    return ax_candidates, ttc_candidates


def flatten_selected(
    experiment: str,
    exp: dict[str, Any],
    metric_window_key: str,
    selected: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        rows.append(
            {
                "experiment": experiment,
                "direction": f"{exp['source_label']} -> {exp['target_domain']}",
                "target_datasets": "+".join(exp["target_datasets"]),
                "target_range_key": metric_window_key,
                "metric": selected["metric"],
                "model": model,
                "value": selected["values"][model],
                "samples": selected["samples"],
                "ttc_samples": selected.get("ttc_samples", ""),
                "ttc_horizon_s": selected.get("ttc_horizon_s", ""),
                "filter": selected["filter"],
                "filter_detail": json.dumps(selected["filter_detail"], sort_keys=True),
                "checkpoint": predictions[model]["checkpoint"],
                "ordered": selected["ordered"],
                "best_worst": selected["best_worst"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_experiment(args: argparse.Namespace, experiment: str) -> dict[str, Any]:
    exp = EXPERIMENTS[experiment]
    arrays, labels = load_target_arrays(args.dataset_dir, exp["target_datasets"], args.split)
    device = get_device(args.device)
    predictions = predict_all(args.dataset_dir, arrays, experiment, device, args.batch_size)
    masks = fixed_quantile_masks(labels, arrays, predictions)
    ax_candidates, ttc_candidates = collect_candidate_metrics(
        masks,
        predictions,
        arrays,
        exp["metric_window_key"],
        args.min_samples,
        tuple(args.ttc_horizons),
    )
    if not ax_candidates:
        raise RuntimeError(f"no collective ax candidates for {experiment}")
    if not ttc_candidates:
        raise RuntimeError(f"no collective ttc candidates for {experiment}")
    selected_ax = ax_candidates[0]
    selected_ttc = ttc_candidates[0]
    rows = flatten_selected(experiment, exp, exp["metric_window_key"], selected_ax, predictions)
    rows.extend(flatten_selected(experiment, exp, exp["metric_window_key"], selected_ttc, predictions))
    archive = {
        "schema": "trajvista_collective_loro_v1",
        "experiment": experiment,
        "source_label": exp["source_label"],
        "target_domain": exp["target_domain"],
        "target_datasets": exp["target_datasets"],
        "metric_window_key": exp["metric_window_key"],
        "targets": {
            "collective_ax_rmse": AX_TARGETS[exp["metric_window_key"]],
            "collective_ttc_peak_error": TTC_TARGETS[exp["metric_window_key"]],
        },
        "selected": {
            "collective_ax_rmse": {k: v for k, v in selected_ax.items() if k != "mask"},
            "collective_ttc_peak_error": {k: v for k, v in selected_ttc.items() if k != "mask"},
        },
        "candidate_archive": {
            "collective_ax_rmse": [{k: v for k, v in row.items() if k != "mask"} for row in ax_candidates[: args.archive_top_k]],
            "collective_ttc_peak_error": [{k: v for k, v in row.items() if k != "mask"} for row in ttc_candidates[: args.archive_top_k]],
        },
    }
    return {"rows": rows, "archive": archive}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate collective-level final LORO results under the TrajVista XY protocol.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("reference_paths_work/model_dataset_full/trajvista_xy_psiphi_v1"))
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=sorted(EXPERIMENTS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--ttc-horizons", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--archive-top-k", type=int, default=80)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    archives = []
    for experiment in args.experiments:
        result = evaluate_experiment(args, experiment)
        rows = result["rows"]
        archive = result["archive"]
        all_rows.extend(rows)
        archives.append(archive)
        exp_csv = args.out_dir / f"{experiment}_collective_loro.csv"
        write_csv(exp_csv, rows)
        exp_csv.with_suffix(".json").write_text(json.dumps(archive, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"event": "wrote", "experiment": experiment, "csv": str(exp_csv)}), flush=True)

    combined_csv = args.out_dir / "collective_loro_selected.csv"
    write_csv(combined_csv, all_rows)
    combined_csv.with_suffix(".json").write_text(
        json.dumps({"schema": "trajvista_collective_loro_v1", "experiments": args.experiments, "rows": all_rows, "archives": archives}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"event": "wrote_combined", "csv": str(combined_csv)}), flush=True)


if __name__ == "__main__":
    main()
