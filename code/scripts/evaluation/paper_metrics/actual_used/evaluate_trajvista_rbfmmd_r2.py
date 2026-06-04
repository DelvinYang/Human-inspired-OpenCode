from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building.build_trajvista_highd_smoke import FEATURE_NAMES
from tools.dataset_building.train_trajvista_highd_psiphi_smoke import NormStats, TransitionDataset
from tools.dataset_building.train_trajvista_highd_psiphi_variant_smoke import HybridPsiCorrectionModel, LocalizedBackboneModel
from tools.dataset_building.train_trajvista_fd_align_ddp import LegacyFDAlignModel
from tools.dataset_building.train_trajvista_gantl_wgan_tl_ddp import AccDataset, Regressor
from tools.dataset_building.train_trajvista_gt_mmd_ddp import GTMMDModel
from tools.dataset_building.train_trajvista_psiphi_hybrid_full_domain import concat_arrays, load_dataset_split


def compute_r2_old(pred: torch.Tensor, target: torch.Tensor) -> tuple[list[float], float]:
    target_mean = torch.mean(target, dim=0)
    ss_tot = torch.sum((target - target_mean) ** 2, dim=0)
    ss_res = torch.sum((target - pred) ** 2, dim=0)
    safe_ss_tot = torch.where(ss_tot > 0, ss_tot, torch.ones_like(ss_tot))
    r2 = 1.0 - ss_res / safe_ss_tot
    zero_var_mask = ss_tot <= 0
    perfect_mask = zero_var_mask & (ss_res <= 0)
    r2 = torch.where(zero_var_mask, torch.zeros_like(r2), r2)
    r2 = torch.where(perfect_mask, torch.ones_like(r2), r2)
    return [float(x) for x in r2.tolist()], float(torch.mean(r2).item())


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=-1)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, bandwidths: Sequence[float]) -> torch.Tensor:
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    dists = pairwise_sq_dists(x, y)
    kernel = torch.zeros_like(dists, dtype=torch.float64)
    for bw in bandwidths:
        gamma = 1.0 / (2.0 * (bw**2))
        kernel += torch.exp(-gamma * dists)
    return kernel / float(len(bandwidths))


def rbf_kernel_values(x: torch.Tensor, y: torch.Tensor, bandwidths: Sequence[float]) -> torch.Tensor:
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    dists = torch.sum((x - y) ** 2, dim=-1)
    vals = torch.zeros_like(dists, dtype=torch.float64)
    for bw in bandwidths:
        gamma = 1.0 / (2.0 * (bw**2))
        vals += torch.exp(-gamma * dists)
    return vals / float(len(bandwidths))


def compute_rbf_mmd_old(
    pred: torch.Tensor,
    target: torch.Tensor,
    bandwidths: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    exact_threshold: int = 4096,
    fallback_sample: int = 2048,
    eps: float = 1e-12,
) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape: {pred.shape} vs {target.shape}")
    n = pred.shape[0]
    if n == 0:
        raise ValueError("pred and target must contain at least one sample")
    pred = pred.to(torch.float64)
    target = target.to(torch.float64)
    if n <= exact_threshold:
        if n < 2:
            raise ValueError("Need at least two samples to estimate MMD")
        k_xx = gaussian_kernel(pred, pred, bandwidths)
        k_yy = gaussian_kernel(target, target, bandwidths)
        k_xy = gaussian_kernel(pred, target, bandwidths)
        mmd_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
        mmd_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n * (n - 1))
        mmd = mmd_xx + mmd_yy - 2.0 * k_xy.mean()
        if mmd < -eps:
            mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
        return float(max(mmd.item(), 0.0))
    if n % 2 == 1:
        pred = pred[:-1]
        target = target[:-1]
        n -= 1
    x_a = pred[0::2]
    x_b = pred[1::2]
    y_a = target[0::2]
    y_b = target[1::2]
    k_xx = rbf_kernel_values(x_a, x_b, bandwidths)
    k_yy = rbf_kernel_values(y_a, y_b, bandwidths)
    k_xy = rbf_kernel_values(x_a, y_b, bandwidths)
    k_yx = rbf_kernel_values(x_b, y_a, bandwidths)
    mmd = 2.0 * (k_xx + k_yy - k_xy - k_yx).mean()
    if mmd < -eps:
        subset = min(n, fallback_sample)
        idx = torch.randperm(n, device=pred.device)[:subset]
        pred_sub = pred[idx]
        target_sub = target[idx]
        k_xx_b = gaussian_kernel(pred_sub, pred_sub, bandwidths)
        k_yy_b = gaussian_kernel(target_sub, target_sub, bandwidths)
        k_xy_b = gaussian_kernel(pred_sub, target_sub, bandwidths)
        mmd = k_xx_b.mean() + k_yy_b.mean() - 2.0 * k_xy_b.mean()
    return float(max(mmd.item(), 0.0))


def load_checkpoint_model(checkpoint_path: Path, device: torch.device, model_type: str) -> tuple[torch.nn.Module, NormStats, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stats = NormStats(**payload["stats"])
    args = payload.get("args", {})
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
def predict_arrays(
    model: torch.nn.Module,
    stats: NormStats,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    model_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = AccDataset(arrays, stats) if model_type in {"gantl", "localized"} else TransitionDataset(arrays, stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    action_mean = torch.tensor(stats.action_mean, dtype=torch.float32, device=device)
    action_std = torch.tensor(stats.action_std, dtype=torch.float32, device=device)
    preds = []
    targets = []
    for batch in loader:
        if model_type in {"gantl", "localized"}:
            state, _action, action_raw, _state_raw = batch
        else:
            state, _action, _ns, _na, action_raw, _state_raw = batch
        state = state.to(device)
        pred_norm = model(state)
        pred = pred_norm * action_std + action_mean
        preds.append(pred.detach().cpu())
        targets.append(action_raw.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def evaluate_arrays(
    model: torch.nn.Module,
    stats: NormStats,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    model_type: str,
) -> dict[str, Any]:
    pred, target = predict_arrays(model, stats, arrays, device, batch_size, model_type)
    r2_dims, r2_mean = compute_r2_old(pred, target)
    return {
        "samples": int(pred.shape[0]),
        "r2_acc_dims": r2_dims,
        "r2_acc": r2_mean,
        "rbf_mmd_acc": compute_rbf_mmd_old(pred, target),
    }


def infer_datasets(summary: dict[str, Any]) -> list[str]:
    if "target_datasets" in summary:
        return list(summary["target_datasets"])
    if "datasets" in summary:
        return list(summary["datasets"])
    if "source_datasets" in summary:
        return list(summary["source_datasets"])
    raise ValueError("could not infer datasets from summary")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TrajVista checkpoints with the old R2/RBF-MMD algorithms.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--dataset-dir", type=Path, default=Path("reference_paths_work/model_dataset_full/trajvista_xy_psiphi_v1"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-type", choices=["hybrid", "localized", "gantl", "fdalign_legacy", "gtmmd"], default="hybrid")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    summary = json.loads(args.summary.read_text(encoding="utf-8")) if args.summary else {}
    datasets = args.datasets or infer_datasets(summary)
    model, stats, payload = load_checkpoint_model(args.checkpoint, device, args.model_type)

    by_dataset = {}
    parts = []
    for dataset in datasets:
        arrays = load_dataset_split(args.dataset_dir, dataset, args.split)
        by_dataset[dataset] = evaluate_arrays(model, stats, arrays, device, args.batch_size, args.model_type)
        parts.append(arrays)
    overall = evaluate_arrays(model, stats, concat_arrays(parts), device, args.batch_size, args.model_type)
    result = {
        "schema": "trajvista_old_rbfmmd_r2_eval_v1",
        "checkpoint": str(args.checkpoint),
        "source_summary": str(args.summary) if args.summary else "",
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "datasets": datasets,
        "model_type": args.model_type,
        "algorithm": {
            "r2": "old TrajVista compute_r2: per-dim 1 - SS_res / SS_tot, reported mean over ax/ay",
            "rbf_mmd": "old TrajVista compute_rbf_mmd: bandwidths=(1,2,4,8), exact U-statistic <=4096 samples, linear unbiased estimator above 4096",
        },
        "checkpoint_args": payload.get("args", {}),
        "overall": overall,
        "by_dataset": by_dataset,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
