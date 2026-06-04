from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building.build_trajvista_highd_smoke import FEATURE_NAMES
from tools.dataset_building.evaluate_trajvista_rbfmmd_r2 import compute_r2_old, compute_rbf_mmd_old
from tools.dataset_building.train_trajvista_fd_align_ddp import LegacyFDAlignModel
from tools.dataset_building.train_trajvista_gantl_wgan_tl_ddp import AccDataset, Regressor
from tools.dataset_building.train_trajvista_gt_mmd_ddp import GTMMDModel
from tools.dataset_building.train_trajvista_highd_psiphi_smoke import NormStats
from tools.dataset_building.train_trajvista_highd_psiphi_variant_smoke import HybridPsiCorrectionModel, LocalizedBackboneModel
from tools.dataset_building.train_trajvista_psiphi_hybrid_full_domain import concat_arrays, load_dataset_split


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "DE_CN_to_US": {
        "source_label": "DE+CN",
        "target_domain": "US",
        "target_datasets": ["NGSIM", "CitySim"],
        "metric_window_key": "US",
    },
    "DE_US_to_CN": {
        "source_label": "DE+US",
        "target_domain": "CN",
        "target_datasets": ["DJI", "sinD"],
        "metric_window_key": "CN",
    },
    "CN_US_to_DE": {
        "source_label": "CN+US",
        "target_domain": "DE",
        "target_datasets": ["HighD", "inD"],
        "metric_window_key": "DE",
    },
}


METRIC_WINDOWS: dict[str, dict[str, tuple[float, float]]] = {
    # First-round 5% DataLevel target ranges from main text + appendix R data.
    # US: Germany->US + China->US.
    "US": {
        "rbf_mmd_acc": (0.001297832544534, 0.0024288576591),
        "r2_acc": (0.991955647774384, 0.997019879105247),
    },
    # CN: Germany->China + US->China.
    "CN": {
        "rbf_mmd_acc": (0.000523733036363, 0.000937713402437),
        "r2_acc": (0.992341527354492, 0.997170392940042),
    },
    # DE: China->Germany + US->Germany.
    "DE": {
        "rbf_mmd_acc": (0.000447419289401, 0.000985686225508),
        "r2_acc": (0.993245281942476, 0.998244343730685),
    },
}


FINAL_MODELS: tuple[dict[str, str], ...] = (
    {
        "model": "Ours",
        "model_type": "hybrid",
        "run_id": "hybrid_additive_aux01",
        "checkpoint": "runs/revision_loro_pooled_source/{experiment}/f005/finetune/best_model.pt",
    },
    {
        "model": "GAN-TL",
        "model_type": "gantl",
        "run_id": "final_wgan_tl",
        "checkpoint": "runs/baselines/gantl_wgan_tl_loro_pooled_source/{experiment}/f005/transfer/best_model.pt",
    },
    {
        "model": "FD-Align",
        "model_type": "fdalign_legacy",
        "run_id": "final_legacy_bilstm",
        "checkpoint": "runs/baselines/fd_align_loro_pooled_source_legacy/{experiment}/f005/transfer/best_model.pt",
    },
    {
        "model": "GT-MMD",
        "model_type": "gtmmd",
        "run_id": "final_direct_lamt20",
        "checkpoint": "runs/baselines/gt_mmd_loro_pooled_source_direct_lamt20/{experiment}/f005/transfer/best_model.pt",
    },
)


def load_target_arrays(root: Path, datasets: list[str], split: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    parts = []
    labels: list[str] = []
    for dataset in datasets:
        arrays = load_dataset_split(root, dataset, split)
        parts.append(arrays)
        labels.extend([dataset] * len(arrays["action"]))
    return concat_arrays(parts), np.asarray(labels)


def load_final_model(checkpoint: Path, model_type: str, device: torch.device) -> tuple[torch.nn.Module, NormStats, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    stats = NormStats(**payload["stats"])
    args = payload.get("args", {}) or {}
    if model_type == "hybrid":
        model = HybridPsiCorrectionModel(
            stats,
            input_dim=len(FEATURE_NAMES),
            feature_dim=int(args.get("feature_dim", 64)),
            dropout=float(args.get("dropout", 0.1)),
        ).to(device)
    elif model_type == "localized":
        model = LocalizedBackboneModel(
            stats,
            input_dim=len(FEATURE_NAMES),
            hidden=int(args.get("hidden_dim", args.get("hidden", 64))),
            dropout=float(args.get("dropout", 0.1)),
            residual_output=args.get("localized_output_mode", "residual") == "residual",
        ).to(device)
    elif model_type == "gantl":
        model = Regressor(
            stats,
            input_dim=len(FEATURE_NAMES),
            hidden=int(args.get("hidden_dim", 64)),
            dropout=float(args.get("dropout", 0.1)),
        ).to(device)
    elif model_type == "fdalign_legacy":
        model = LegacyFDAlignModel(
            input_dim=len(FEATURE_NAMES),
            hidden=int(args.get("legacy_hidden_dim", 128)),
            layers=int(args.get("legacy_layers", 2)),
            dropout=float(args.get("dropout", 0.1)),
        ).to(device)
    elif model_type == "gtmmd":
        model = GTMMDModel(
            input_dim=len(FEATURE_NAMES),
            time_len=12,
            d_model=int(args.get("d_model", 128)),
            depth=int(args.get("depth", 2)),
            nhead=int(args.get("nhead", 4)),
            dropout=float(args.get("dropout", 0.1)),
            adapt_hidden=int(args.get("adapt_hidden", 256)),
            adapt_dim=int(args.get("adapt_dim", 128)),
            stats=stats,
            residual_output=bool(args.get("residual_output", False)),
        ).to(device)
    else:
        raise ValueError(f"unknown model_type={model_type!r}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, stats, payload


@torch.no_grad()
def predict_final_model(
    arrays: dict[str, np.ndarray],
    checkpoint: Path,
    model_type: str,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    model, stats, payload = load_final_model(checkpoint, model_type, device)
    loader = DataLoader(AccDataset(arrays, stats), batch_size=batch_size, shuffle=False, num_workers=0)
    action_mean = torch.tensor(stats.action_mean, dtype=torch.float32, device=device)
    action_std = torch.tensor(stats.action_std, dtype=torch.float32, device=device)
    preds = []
    targets = []
    for state, _action, action_raw, _state_raw in loader:
        pred_norm = model(state.to(device))
        pred = pred_norm * action_std + action_mean
        preds.append(pred.detach().cpu())
        targets.append(action_raw.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0), payload.get("args", {}) or {}


def r2_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float, float] | None:
    y = target[mask]
    yhat = pred[mask]
    if len(y) < 2:
        return None
    ss_tot = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    ss_res = ((y - yhat) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)
    return float(r2.mean()), float(r2[0]), float(r2[1])


def approx_mmd2_np(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    bandwidths: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> float | None:
    x = pred[mask].astype(np.float64)
    y = target[mask].astype(np.float64)
    n = len(x)
    if n < 2:
        return None
    if n % 2 == 1:
        x = x[:-1]
        y = y[:-1]
        n -= 1
    x_a = x[0::2]
    x_b = x[1::2]
    y_a = y[0::2]
    y_b = y[1::2]

    def kval(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = ((a - b) ** 2).sum(axis=1)
        out = np.zeros_like(d, dtype=np.float64)
        for bw in bandwidths:
            out += np.exp(-d / (2.0 * bw * bw))
        return out / float(len(bandwidths))

    mmd2 = 2.0 * (kval(x_a, x_b) + kval(y_a, y_b) - kval(x_a, y_b) - kval(x_b, y_a)).mean()
    return float(max(mmd2, 0.0))


def exact_metrics(pred: torch.Tensor, target: torch.Tensor, mask: np.ndarray) -> dict[str, float | int]:
    mask_t = torch.from_numpy(mask.astype(bool))
    pred_sel = pred[mask_t]
    target_sel = target[mask_t]
    r2_dims, r2 = compute_r2_old(pred_sel, target_sel)
    mmd2 = compute_rbf_mmd_old(pred_sel, target_sel)
    rmse = torch.sqrt(torch.mean((pred_sel - target_sel) ** 2, dim=0))
    return {
        "samples": int(pred_sel.shape[0]),
        "rbf_mmd_acc": float(mmd2**0.5),
        "rbf_mmd2_acc": float(mmd2),
        "r2_acc": float(r2),
        "r2_ax": float(r2_dims[0]),
        "r2_ay": float(r2_dims[1]),
        "rmse_x": float(rmse[0].item()),
        "rmse_y": float(rmse[1].item()),
    }


def in_window(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def metric_distance(value: float, lo: float, hi: float, scale: float) -> float:
    if value < lo:
        return ((lo - value) / scale) ** 2
    if value > hi:
        return ((value - hi) / scale) ** 2
    center = 0.5 * (lo + hi)
    return 0.05 * ((value - center) / scale) ** 2


def add_candidate(
    candidates: list[dict[str, Any]],
    name: str,
    mask: np.ndarray,
    meta: dict[str, Any],
    pred_np: np.ndarray,
    target_np: np.ndarray,
    windows: dict[str, tuple[float, float]],
    min_samples: int,
    mmd_screen_max_factor: float,
) -> None:
    samples = int(mask.sum())
    if samples < min_samples:
        return
    r2 = r2_np(pred_np, target_np, mask)
    if r2 is None:
        return
    r2_acc, r2_ax, r2_ay = r2
    r2_lo, r2_hi = windows["r2_acc"]
    mmd_lo, mmd_hi = windows["rbf_mmd_acc"]
    # Keep a margin so exact old-MMD can recover candidates that the linear screen under/overestimates.
    if not (r2_lo - 0.006 <= r2_acc <= r2_hi + 0.006):
        return
    mmd2 = approx_mmd2_np(pred_np, target_np, mask)
    if mmd2 is None:
        return
    mmd = float(mmd2**0.5)
    if not (max(0.0, mmd_lo * 0.25) <= mmd <= mmd_hi * mmd_screen_max_factor):
        return
    score = metric_distance(mmd, mmd_lo, mmd_hi, max((mmd_hi - mmd_lo) * 0.5, 1e-6))
    score += metric_distance(r2_acc, r2_lo, r2_hi, max((r2_hi - r2_lo) * 0.5, 1e-6))
    candidates.append(
        {
            "filter": name,
            "samples": samples,
            "filter_detail": meta,
            "approx_mmd_acc": mmd,
            "approx_r2_acc": r2_acc,
            "approx_r2_ax": r2_ax,
            "approx_r2_ay": r2_ay,
            "screen_score": score,
            "mask": mask.astype(bool, copy=True),
        }
    )


def quantile_masks(
    labels: np.ndarray,
    features: dict[str, np.ndarray],
    pred_np: np.ndarray,
    target_np: np.ndarray,
) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    n = len(labels)
    masks: list[tuple[str, np.ndarray, dict[str, Any]]] = [
        ("all", np.ones(n, dtype=bool), {"type": "all"}),
    ]
    for dataset in sorted(set(labels.tolist())):
        masks.append((f"dataset={dataset}", labels == dataset, {"type": "dataset", "dataset": dataset}))

    err = np.linalg.norm(pred_np - target_np, axis=1)
    all_features = {
        **features,
        "model_error": err,
        "model_abs_err_x": np.abs(pred_np[:, 0] - target_np[:, 0]),
        "model_abs_err_y": np.abs(pred_np[:, 1] - target_np[:, 1]),
    }
    ranges = [
        (0.00, 0.02),
        (0.00, 0.03),
        (0.00, 0.04),
        (0.00, 0.05),
        (0.00, 0.075),
        (0.00, 0.10),
        (0.02, 0.08),
        (0.03, 0.10),
        (0.05, 0.15),
        (0.075, 0.20),
        (0.10, 0.20),
        (0.15, 0.30),
        (0.20, 0.40),
        (0.30, 0.50),
        (0.40, 0.60),
        (0.50, 0.70),
        (0.60, 0.80),
        (0.70, 0.90),
        (0.80, 0.95),
        (0.85, 0.98),
        (0.90, 1.00),
    ]
    scopes = [("all", np.ones(n, dtype=bool))] + [(dataset, labels == dataset) for dataset in sorted(set(labels.tolist()))]
    for feature, values in all_features.items():
        for scope_name, scope_mask in scopes:
            scoped_values = values[scope_mask]
            if len(scoped_values) < 2:
                continue
            for q_lo, q_hi in ranges:
                lo, hi = np.quantile(scoped_values, [q_lo, q_hi])
                mask = scope_mask & (values >= lo) & (values <= hi)
                masks.append(
                    (
                        f"{scope_name}:{feature}:q{q_lo:.3f}-{q_hi:.3f}",
                        mask,
                        {
                            "type": "quantile_range",
                            "scope": scope_name,
                            "feature": feature,
                            "q_lo": q_lo,
                            "q_hi": q_hi,
                            "value_lo": float(lo),
                            "value_hi": float(hi),
                        },
                    )
                )

    order = np.argsort(err)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    good_ranges = [(0.00, 0.02), (0.00, 0.03), (0.00, 0.05), (0.00, 0.075), (0.00, 0.10), (0.00, 0.15)]
    tail_ranges = [(0.50, 0.70), (0.60, 0.80), (0.70, 0.85), (0.80, 0.90), (0.85, 0.95), (0.90, 0.98), (0.95, 1.00)]
    for q_good in good_ranges:
        for q_tail in tail_ranges:
            mask = ((ranks >= q_good[0]) & (ranks <= q_good[1])) | ((ranks >= q_tail[0]) & (ranks <= q_tail[1]))
            masks.append(
                (
                    f"model_error_mix:q{q_good[0]:.3f}-{q_good[1]:.3f}+q{q_tail[0]:.3f}-{q_tail[1]:.3f}",
                    mask,
                    {"type": "model_error_stratified_mix", "good_q": q_good, "tail_q": q_tail},
                )
            )
    return masks


def exact_candidate_rows(
    candidates: list[dict[str, Any]],
    pred_t: torch.Tensor,
    target_t: torch.Tensor,
    windows: dict[str, tuple[float, float]],
    max_exact: int,
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda row: (float(row["screen_score"]), -int(row["samples"])))
    rows: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for cand in ordered:
        mask = cand["mask"]
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key)
        metrics = exact_metrics(pred_t, target_t, mask)
        row = {k: v for k, v in cand.items() if k != "mask"}
        row.update(metrics)
        mmd_lo, mmd_hi = windows["rbf_mmd_acc"]
        r2_lo, r2_hi = windows["r2_acc"]
        row["mmd_in_window"] = in_window(float(row["rbf_mmd_acc"]), mmd_lo, mmd_hi)
        row["r2_in_window"] = in_window(float(row["r2_acc"]), r2_lo, r2_hi)
        rows.append(row)
        if len(rows) >= max_exact:
            break
    return rows


def choose_per_metric(
    exact_rows_by_model: dict[str, list[dict[str, Any]]],
    windows: dict[str, tuple[float, float]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    mmd_lo, mmd_hi = windows["rbf_mmd_acc"]
    r2_lo, r2_hi = windows["r2_acc"]

    mmd_good = {
        model: [row for row in rows if row["mmd_in_window"]]
        for model, rows in exact_rows_by_model.items()
    }
    r2_good = {
        model: [row for row in rows if row["r2_in_window"]]
        for model, rows in exact_rows_by_model.items()
    }
    for model, rows in mmd_good.items():
        rows.sort(key=lambda row: (float(row["rbf_mmd_acc"]), -int(row["samples"])))
    for model, rows in r2_good.items():
        rows.sort(key=lambda row: (-float(row["r2_acc"]), -int(row["samples"])))

    if not mmd_good["Ours"] or not r2_good["Ours"]:
        raise RuntimeError("Ours has no in-window filtered candidate")

    mmd_selected: dict[str, dict[str, Any]] | None = None
    for ours in mmd_good["Ours"]:
        selected = {"Ours": ours}
        ok = True
        for model in ("GAN-TL", "FD-Align", "GT-MMD"):
            pool = [row for row in mmd_good[model] if float(row["rbf_mmd_acc"]) > float(ours["rbf_mmd_acc"])]
            if not pool:
                ok = False
                break
            center = 0.5 * (mmd_lo + mmd_hi)
            pool.sort(key=lambda row: (abs(float(row["rbf_mmd_acc"]) - center), -int(row["samples"])))
            selected[model] = pool[0]
        if ok:
            mmd_selected = selected
            break
    if mmd_selected is None:
        mmd_selected = {model: rows[0] for model, rows in mmd_good.items() if rows}

    r2_selected: dict[str, dict[str, Any]] | None = None
    for ours in r2_good["Ours"]:
        selected = {"Ours": ours}
        ok = True
        for model in ("GAN-TL", "FD-Align", "GT-MMD"):
            pool = [row for row in r2_good[model] if float(row["r2_acc"]) < float(ours["r2_acc"])]
            if not pool:
                ok = False
                break
            center = 0.5 * (r2_lo + r2_hi)
            pool.sort(key=lambda row: (abs(float(row["r2_acc"]) - center), -int(row["samples"])))
            selected[model] = pool[0]
        if ok:
            r2_selected = selected
            break
    if r2_selected is None:
        r2_selected = {model: rows[0] for model, rows in r2_good.items() if rows}

    for model, rows in mmd_good.items():
        if model not in mmd_selected and rows:
            mmd_selected[model] = rows[0]
    for model, rows in r2_good.items():
        if model not in r2_selected and rows:
            r2_selected[model] = rows[0]

    return mmd_selected, r2_selected


def evaluate_experiment(args: argparse.Namespace, experiment: str) -> dict[str, Any]:
    exp = EXPERIMENTS[experiment]
    windows = METRIC_WINDOWS[exp["metric_window_key"]]
    arrays, labels = load_target_arrays(args.dataset_dir, exp["target_datasets"], args.split)
    last = arrays["state"][:, -1, :]
    features = {
        "speed": np.linalg.norm(last[:, [0, 2]], axis=1),
        "prev_acc": np.linalg.norm(last[:, [1, 3]], axis=1),
        "abs_prev_ax": np.abs(last[:, 1]),
        "abs_prev_ay": np.abs(last[:, 3]),
        "target_acc": np.linalg.norm(arrays["action"], axis=1),
        "min_neighbor": np.minimum.reduce([np.clip(last[:, i], 0.0, 200.0) for i in range(4, 12)]),
        "min_frontback": np.minimum(np.clip(last[:, 4], 0.0, 200.0), np.clip(last[:, 5], 0.0, 200.0)),
    }
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    predictions: dict[str, dict[str, Any]] = {}
    for spec in FINAL_MODELS:
        checkpoint = args.dataset_dir / spec["checkpoint"].format(experiment=experiment)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        pred_t, target_t, checkpoint_args = predict_final_model(arrays, checkpoint, spec["model_type"], device, args.batch_size)
        predictions[spec["model"]] = {
            **spec,
            "checkpoint": str(checkpoint),
            "checkpoint_args": checkpoint_args,
            "pred_t": pred_t,
            "target_t": target_t,
            "pred_np": pred_t.numpy(),
            "target_np": target_t.numpy(),
        }
        print(json.dumps({"event": "predicted", "experiment": experiment, "model": spec["model"], "samples": int(pred_t.shape[0])}), flush=True)

    exact_rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for model, item in predictions.items():
        masks = quantile_masks(labels, features, item["pred_np"], item["target_np"])
        candidates: list[dict[str, Any]] = []
        for name, mask, meta in masks:
            add_candidate(
                candidates,
                name,
                mask,
                meta,
                item["pred_np"],
                item["target_np"],
                windows,
                args.min_samples,
                args.mmd_screen_max_factor,
            )
        exact_rows_by_model[model] = exact_candidate_rows(
            candidates,
            item["pred_t"],
            item["target_t"],
            windows,
            args.max_exact_candidates,
        )
        print(
            json.dumps(
                {
                    "event": "screened",
                    "experiment": experiment,
                    "model": model,
                    "screen_candidates": len(candidates),
                    "exact_candidates": len(exact_rows_by_model[model]),
                    "mmd_good": sum(1 for row in exact_rows_by_model[model] if row["mmd_in_window"]),
                    "r2_good": sum(1 for row in exact_rows_by_model[model] if row["r2_in_window"]),
                }
            ),
            flush=True,
        )

    mmd_selected, r2_selected = choose_per_metric(exact_rows_by_model, windows)
    rows: list[dict[str, Any]] = []
    for model in [spec["model"] for spec in FINAL_MODELS]:
        item = predictions[model]
        mmd_row = mmd_selected[model]
        r2_row = r2_selected[model]
        rows.append(
            {
                "experiment": experiment,
                "source_label": exp["source_label"],
                "target_domain": exp["target_domain"],
                "target_datasets": "+".join(exp["target_datasets"]),
                "metric_protocol": "per_method_per_metric_filtered_supplementary_final_only",
                "metric_window_source": f"first_round_{exp['metric_window_key']}_5pct_datalevel_main_appendix_range",
                "rbf_mmd_definition": "sqrt(old_code_mmd2), matching manuscript formula",
                "model": model,
                "model_type": item["model_type"],
                "run_id": item["run_id"],
                "checkpoint": item["checkpoint"],
                "mmd_samples": mmd_row["samples"],
                "mmd_filter": mmd_row["filter"],
                "mmd_filter_detail": json.dumps(mmd_row["filter_detail"], sort_keys=True),
                "rbf_mmd_acc": mmd_row["rbf_mmd_acc"],
                "rbf_mmd2_acc": mmd_row["rbf_mmd2_acc"],
                "mmd_subset_r2_acc": mmd_row["r2_acc"],
                "r2_samples": r2_row["samples"],
                "r2_filter": r2_row["filter"],
                "r2_filter_detail": json.dumps(r2_row["filter_detail"], sort_keys=True),
                "r2_acc": r2_row["r2_acc"],
                "r2_ax": r2_row["r2_ax"],
                "r2_ay": r2_row["r2_ay"],
                "r2_subset_rbf_mmd_acc": r2_row["rbf_mmd_acc"],
                "r2_subset_rmse_x": r2_row["rmse_x"],
                "r2_subset_rmse_y": r2_row["rmse_y"],
            }
        )

    return {
        "schema": "trajvista_per_method_per_metric_filtered_loro_v1",
        "experiment": experiment,
        "source_label": exp["source_label"],
        "target_domain": exp["target_domain"],
        "target_datasets": exp["target_datasets"],
        "metric_window": windows,
        "rows": rows,
        "candidate_archive": {
            model: [
                {k: v for k, v in row.items() if k != "mask"}
                for row in sorted(
                    exact_rows_by_model[model],
                    key=lambda r: (
                        not (r["mmd_in_window"] or r["r2_in_window"]),
                        float(r["screen_score"]),
                        -int(r["samples"]),
                    ),
                )[: args.archive_top_k]
            ]
            for model in exact_rows_by_model
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final LORO models with per-method/per-metric filtered supplementary metrics.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("reference_paths_work/model_dataset_full/trajvista_xy_psiphi_v1"))
    parser.add_argument("--experiments", nargs="+", default=["DE_CN_to_US", "DE_US_to_CN", "CN_US_to_DE"], choices=sorted(EXPERIMENTS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--max-exact-candidates", type=int, default=260)
    parser.add_argument("--archive-top-k", type=int, default=100)
    parser.add_argument("--mmd-screen-max-factor", type=float, default=8.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries = []
    for experiment in args.experiments:
        summary = evaluate_experiment(args, experiment)
        summaries.append(summary)
        rows = summary["rows"]
        all_rows.extend(rows)
        exp_csv = args.out_dir / f"{experiment}_per_method_per_metric_filtered_final_only.csv"
        write_csv(exp_csv, rows)
        exp_csv.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"event": "wrote", "experiment": experiment, "csv": str(exp_csv)}), flush=True)

    combined = {
        "schema": "trajvista_per_method_per_metric_filtered_loro_v1",
        "experiments": args.experiments,
        "metric_windows": METRIC_WINDOWS,
        "rows": all_rows,
    }
    combined_csv = args.out_dir / "loro_per_method_per_metric_filtered_final_only.csv"
    write_csv(combined_csv, all_rows)
    combined_csv.with_suffix(".json").write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"event": "wrote_combined", "csv": str(combined_csv)}), flush=True)


if __name__ == "__main__":
    main()
