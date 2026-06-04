from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cultural_align.data.schema import FEATURE_NAMES
from cultural_align.models.trajvista import FEATURE_DIM, HIDDEN_DIM, make_model
from cultural_align.training.data import (
    NormStats,
    compute_stats,
    concat_arrays,
    load_dataset_split,
    load_multi_split,
    make_transition_loader,
    set_seed,
)
from cultural_align.training.losses import bc_q_loss_from_hs, itd_loss_from_hs
from cultural_align.training.metrics import regression_metrics
from cultural_align.training.trajectory_metrics import complete_trajectory_metrics


DOMAIN_DATASETS = {
    "DE": ["HighD", "inD"],
    "US": ["NGSIM", "CitySim"],
    "CN": ["DJI", "sinD"],
    "INTERACTION_EXT": ["INTERACTION_CHN", "INTERACTION_DEU", "INTERACTION_USA"],
    "SIX_DATASETS": ["HighD", "inD", "CitySim", "sinD", "NGSIM", "DJI"],
}


DEFAULTS = {
    "seed": 42,
    "epochs": 20,
    "early_stop_patience": 0,
    "early_stop_min_delta": 0.0,
    "batch_size": 1024,
    "eval_batch_size": 4096,
    "num_workers": 0,
    "pin_memory": False,
    "device": "auto",
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "gamma": 0.9,
    "k_neg": 32,
    "neg_std": 0.5,
    "acc_coef": 1.0,
    "bc_coef": 0.1,
    "itd_coef": 0.1,
    "grad_clip": 5.0,
    "max_train_samples_per_dataset": 0,
    "max_val_samples_per_dataset": 0,
    "max_test_samples_per_dataset": 0,
    "target_fraction": 1.0,
    "matmul_precision": "high",
}


def to_namespace(config: dict[str, Any] | SimpleNamespace) -> SimpleNamespace:
    payload = vars(config).copy() if isinstance(config, SimpleNamespace) else dict(config)
    forbidden = {"hidden_dim", "feature_dim"} & set(payload)
    if forbidden:
        keys = ", ".join(sorted(forbidden))
        raise ValueError(f"{keys} are fixed by the released architecture and must not be configured")
    return SimpleNamespace(**{**DEFAULTS, **payload})


def resolve_datasets(domain: str | None = None, datasets: list[str] | None = None) -> list[str]:
    if datasets:
        return list(datasets)
    if domain:
        if domain not in DOMAIN_DATASETS:
            raise ValueError(f"unknown domain {domain!r}; choices={sorted(DOMAIN_DATASETS)}")
        return list(DOMAIN_DATASETS[domain])
    raise ValueError("provide a domain or explicit datasets")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def model_to_raw_action(pred_norm: torch.Tensor, stats: NormStats, device: torch.device) -> torch.Tensor:
    action_mean = torch.tensor(stats.action_mean, dtype=torch.float32, device=device)
    action_std = torch.tensor(stats.action_std, dtype=torch.float32, device=device)
    return pred_norm * action_std + action_mean


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, stats: NormStats) -> dict[str, Any]:
    model.eval()
    pred_rows = []
    target_rows = []
    loss_sum = 0.0
    loss_count = 0
    for state, action, _next_state, _next_action, action_raw, _state_raw in loader:
        state = state.to(device)
        action = action.to(device)
        pred_norm = model(state)
        loss_sum += float(F.mse_loss(pred_norm, action, reduction="sum").item())
        loss_count += int(action.numel())
        pred_rows.append(model_to_raw_action(pred_norm, stats, device).cpu().numpy())
        target_rows.append(action_raw.numpy())
    pred = np.concatenate(pred_rows, axis=0)
    target = np.concatenate(target_rows, axis=0)
    metrics = regression_metrics(pred, target)
    metrics["loss_standardized"] = float(loss_sum / max(loss_count, 1))
    return metrics


@torch.no_grad()
def w_sensitivity(model: nn.Module, sample_state: torch.Tensor, device: torch.device) -> float:
    model.eval()
    state = sample_state[: min(256, len(sample_state))].to(device)
    pred_a = model(state).detach().clone()
    old_w = model.w.detach().clone()
    model.w.add_(0.5)
    pred_b = model(state).detach().clone()
    model.w.copy_(old_w)
    return float(torch.mean(torch.abs(pred_b - pred_a)).item())


def _make_model_from_args(args: SimpleNamespace, stats: NormStats) -> nn.Module:
    return make_model(
        stats,
        input_dim=len(FEATURE_NAMES),
        dropout=float(args.dropout),
    )


def _prediction_loss(model: nn.Module, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, next_action: torch.Tensor, args: SimpleNamespace) -> tuple[torch.Tensor, dict[str, float]]:
    hidden = model.encode_state(state)
    pred = model.forward_from_hs(hidden, state)
    loss_acc = F.mse_loss(pred, action)
    loss_bc = bc_q_loss_from_hs(model, hidden, action, int(args.k_neg), float(args.neg_std))
    loss_itd = itd_loss_from_hs(model, hidden, action, next_state, next_action, float(args.gamma))
    loss = float(args.acc_coef) * loss_acc + float(args.bc_coef) * loss_bc + float(args.itd_coef) * loss_itd
    return loss, {"loss_acc": float(loss_acc.item()), "loss_bc": float(loss_bc.item()), "loss_itd": float(loss_itd.item())}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_splits(args: SimpleNamespace, datasets: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    root = Path(args.dataset_dir)
    train, train_meta = load_multi_split(
        root,
        datasets,
        "train",
        max_samples_per_dataset=int(args.max_train_samples_per_dataset),
        random_seed=int(args.seed),
        fraction=float(getattr(args, "train_fraction", 1.0)),
    )
    val, val_meta = load_multi_split(root, datasets, "val", max_samples_per_dataset=int(args.max_val_samples_per_dataset))
    test_parts = {}
    test_meta = {}
    for dataset in datasets:
        arr, meta = load_dataset_split(root, dataset, "test", max_samples=int(args.max_test_samples_per_dataset))
        test_parts[dataset] = arr
        test_meta[dataset] = meta
    return train, val, concat_arrays(list(test_parts.values())), {
        "train": train_meta,
        "val": val_meta,
        "test": test_meta,
        "test_counts": {dataset: int(len(arr["action"])) for dataset, arr in test_parts.items()},
    }


def train_supervised(config: dict[str, Any] | SimpleNamespace) -> dict[str, Any]:
    args = to_namespace(config)
    set_seed(int(args.seed))
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
    datasets = resolve_datasets(getattr(args, "domain", None), getattr(args, "datasets", None))
    train_arrays, val_arrays, test_arrays, load_meta = _load_splits(args, datasets)
    stats = compute_stats(train_arrays)
    train_loader = make_transition_loader(train_arrays, stats, int(args.batch_size), True, int(args.num_workers), bool(args.pin_memory))
    val_loader = make_transition_loader(val_arrays, stats, int(args.eval_batch_size), False, int(args.num_workers), bool(args.pin_memory))
    test_loader = make_transition_loader(test_arrays, stats, int(args.eval_batch_size), False, int(args.num_workers), bool(args.pin_memory))
    device = resolve_device(args.device)
    model = _make_model_from_args(args, stats).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(args.epochs), 1), eta_min=float(args.lr) * 0.05)
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    stop_reason = "max_epochs_reached"

    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.time()
        model.train()
        sums = np.zeros(6, dtype=np.float64)
        for state, action, next_state, next_action, _raw, _state_raw in train_loader:
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)
            next_action = next_action.to(device)
            loss, parts = _prediction_loss(model, state, action, next_state, next_action, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            n = int(action.shape[0])
            elems = int(action.numel())
            sums[0] += float(loss.item()) * elems
            sums[1] += parts["loss_acc"] * elems
            sums[2] += parts["loss_bc"] * elems
            sums[3] += parts["loss_itd"] * elems
            sums[4] += elems
            sums[5] += n
        scheduler.step()
        val = evaluate(model, val_loader, device, stats)
        improved = val["loss_standardized"] < best_val - float(args.early_stop_min_delta)
        if improved:
            best_val = val["loss_standardized"]
            best_epoch = epoch
            bad = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
        row = {
            "epoch": epoch,
            "train_loss": float(sums[0] / max(sums[4], 1.0)),
            "train_acc": float(sums[1] / max(sums[4], 1.0)),
            "train_bc": float(sums[2] / max(sums[4], 1.0)),
            "train_itd": float(sums[3] / max(sums[4], 1.0)),
            "train_samples": int(sums[5]),
            "epoch_elapsed_sec": time.time() - epoch_started,
            "val_loss": val["loss_standardized"],
            "val_rmse": val["raw_rmse"],
            "val_r2_mean": val["r2_mean"],
            "lr": scheduler.get_last_lr()[0],
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "elapsed_sec": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if int(args.early_stop_patience) > 0 and bad >= int(args.early_stop_patience):
            stop_reason = f"early_stop_patience_{args.early_stop_patience}"
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_test = evaluate(model, test_loader, device, stats)
    sens = w_sensitivity(model, next(iter(test_loader))[0], device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "stats": asdict(stats), "args": vars(args)}, out_dir / "best_model.pt")
    summary = {
        "schema": "trajvista_xy_psiphi_v1",
        "mode": "supervised",
        "model": "ours",
        "model_dimensions": {"hidden_dim": HIDDEN_DIM, "feature_dim": FEATURE_DIM},
        "datasets": datasets,
        "dataset_dir": str(args.dataset_dir),
        "load_meta": load_meta,
        "stats": asdict(stats),
        "history": history,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "final_test": final_test,
        "fixed_checks": {
            "same_state_action_schema": True,
            "strict_adjacent_transition_pairs": True,
            "uses_psi_phi_and_w": True,
            "w_sensitivity_mean_abs_norm_output": sens,
        },
    }
    _save_json(out_dir / "summary.json", summary)
    print(json.dumps({"model": "ours", "best_val_loss": best_val, "final_test": final_test}, indent=2), flush=True)
    return summary


def _load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], NormStats, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    return payload["model_state_dict"], NormStats(**payload["stats"]), payload


def _inherit_checkpoint_model_args(args: SimpleNamespace, payload: dict[str, Any]) -> None:
    checkpoint_args = payload.get("args", {})
    for key in ("dropout",):
        if key in checkpoint_args:
            setattr(args, key, checkpoint_args[key])


def train_transfer(config: dict[str, Any] | SimpleNamespace) -> dict[str, Any]:
    args = to_namespace(config)
    set_seed(int(args.seed))
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
    device = resolve_device(args.device)
    if not getattr(args, "source_checkpoint", None):
        raise ValueError("train_transfer requires source_checkpoint; use train_domain for source-domain training")
    source_state, source_stats, source_payload = _load_checkpoint(Path(args.source_checkpoint), device)
    _inherit_checkpoint_model_args(args, source_payload)

    target_datasets = resolve_datasets(getattr(args, "target_domain", None), getattr(args, "target_datasets", None))
    root = Path(args.target_dataset_dir or args.dataset_dir)
    train_arrays, train_meta = load_multi_split(
        root,
        target_datasets,
        "train",
        max_samples_per_dataset=int(args.max_train_samples_per_dataset),
        fraction=float(args.target_fraction),
    )
    val_arrays, val_meta = load_multi_split(root, target_datasets, "val", max_samples_per_dataset=int(args.max_val_samples_per_dataset))
    test_parts = {
        dataset: load_dataset_split(root, dataset, "test", max_samples=int(args.max_test_samples_per_dataset))[0]
        for dataset in target_datasets
    }
    stats = source_stats
    train_loader = make_transition_loader(train_arrays, stats, int(args.batch_size), True, int(args.num_workers), bool(args.pin_memory))
    val_loader = make_transition_loader(val_arrays, stats, int(args.eval_batch_size), False, int(args.num_workers), bool(args.pin_memory))
    test_loader = make_transition_loader(concat_arrays(list(test_parts.values())), stats, int(args.eval_batch_size), False, int(args.num_workers), bool(args.pin_memory))
    model = _make_model_from_args(args, stats).to(device)
    load_report = model.load_state_dict(source_state, strict=False)
    load_report = {"missing": list(load_report.missing_keys), "unexpected": list(load_report.unexpected_keys)}
    phase = getattr(args, "phase", None)
    if phase == "calibrate_w":
        for name, param in model.named_parameters():
            param.requires_grad = name == "w"
    elif phase == "finetune_with_target_w":
        if not getattr(args, "target_w_path", None):
            raise ValueError("phase=finetune_with_target_w requires target_w_path")
        w = np.load(args.target_w_path).astype(np.float32).reshape(-1)
        with torch.no_grad():
            model.w.copy_(torch.from_numpy(w).to(model.w.device))
        for name, param in model.named_parameters():
            param.requires_grad = name != "w"
    else:
        raise ValueError("train_transfer phase must be 'calibrate_w' or 'finetune_with_target_w'")
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(args.epochs), 1), eta_min=float(args.lr) * 0.05)
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        sums = np.zeros(5, dtype=np.float64)
        for state, action, next_state, next_action, _raw, _state_raw in train_loader:
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)
            next_action = next_action.to(device)
            loss, parts = _prediction_loss(model, state, action, next_state, next_action, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            elems = int(action.numel())
            sums[0] += float(loss.item()) * elems
            sums[1] += parts["loss_acc"] * elems
            sums[2] += parts["loss_bc"] * elems
            sums[3] += parts["loss_itd"] * elems
            sums[4] += elems
        scheduler.step()
        val = evaluate(model, val_loader, device, stats)
        if val["loss_standardized"] < best_val:
            best_val = val["loss_standardized"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row = {
            "epoch": epoch,
            "train_loss": float(sums[0] / max(sums[4], 1.0)),
            "train_acc": float(sums[1] / max(sums[4], 1.0)),
            "train_bc": float(sums[2] / max(sums[4], 1.0)),
            "train_itd": float(sums[3] / max(sums[4], 1.0)),
            "val_loss": val["loss_standardized"],
            "val_rmse": val["raw_rmse"],
            "lr": scheduler.get_last_lr()[0],
            "best_epoch": best_epoch,
            "elapsed_sec": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    final_test = evaluate(model, test_loader, device, stats)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "stats": asdict(stats), "args": vars(args)}, out_dir / "best_model.pt")
    w_best = model.w.detach().cpu().numpy().astype(np.float32)
    np.save(out_dir / "w_best.npy", w_best)
    summary = {
        "schema": "trajvista_xy_psiphi_v1",
        "mode": "target_finetune",
        "model": "ours",
        "model_dimensions": {"hidden_dim": HIDDEN_DIM, "feature_dim": FEATURE_DIM},
        "phase": phase,
        "target_datasets": target_datasets,
        "target_fraction": float(args.target_fraction),
        "target_train_meta": train_meta,
        "val_meta": val_meta,
        "test_counts": {dataset: int(len(arr["action"])) for dataset, arr in test_parts.items()},
        "source_checkpoint": getattr(args, "source_checkpoint", None),
        "target_w_path": getattr(args, "target_w_path", None),
        "w_file": "w_best.npy",
        "w_l2_norm": float(np.linalg.norm(w_best)),
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
        "load_state_dict": load_report,
        "source_checkpoint_args": {} if source_payload is None else source_payload.get("args", {}),
        "stats": asdict(stats),
        "history": history,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "final_test": final_test,
        "fixed_checks": {
            "target_fraction_by_track_hash": True,
            "target_w_frozen": phase == "finetune_with_target_w",
            "only_w_trainable": phase == "calibrate_w",
            "strict_adjacent_transition_pairs": True,
            "next_action_teacher_forcing": True,
        },
    }
    _save_json(out_dir / "summary.json", summary)
    print(json.dumps({"mode": "target_finetune", "model": "ours", "best_val_loss": best_val, "final_test": final_test}, indent=2), flush=True)
    return summary


def evaluate_checkpoint(config: dict[str, Any] | SimpleNamespace) -> dict[str, Any]:
    args = to_namespace(config)
    device = resolve_device(args.device)
    state_dict, stats, _payload = _load_checkpoint(Path(args.checkpoint), device)
    _inherit_checkpoint_model_args(args, _payload)
    datasets = resolve_datasets(getattr(args, "domain", None), getattr(args, "datasets", None))
    test_parts = {
        dataset: load_dataset_split(Path(args.dataset_dir), dataset, "test", max_samples=int(args.max_test_samples_per_dataset))[0]
        for dataset in datasets
    }
    loader = make_transition_loader(concat_arrays(list(test_parts.values())), stats, int(args.eval_batch_size), False, int(args.num_workers), bool(args.pin_memory))
    model = _make_model_from_args(args, stats).to(device)
    model.load_state_dict(state_dict, strict=False)
    metrics = evaluate(model, loader, device, stats)
    metrics.update(
        complete_trajectory_metrics(
            model,
            stats,
            concat_arrays(list(test_parts.values())),
            device,
            batch_size=int(args.eval_batch_size),
            mode=getattr(args, "rollout_mode", "semi"),
            min_pred_steps=int(getattr(args, "min_trajectory_steps", 50)),
            cr_rel_threshold=float(getattr(args, "cr_rel_threshold", 0.05)),
            cr_abs_threshold=float(getattr(args, "cr_abs_threshold", 2.0)),
            max_trajectories_per_dataset=int(getattr(args, "max_trajectories_per_dataset", 0)),
            seed=int(getattr(args, "seed", 20260517)),
        )
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "model": "ours",
        "model_dimensions": {"hidden_dim": HIDDEN_DIM, "feature_dim": FEATURE_DIM},
        "datasets": datasets,
        "metrics": metrics,
        "test_counts": {dataset: int(len(arr["action"])) for dataset, arr in test_parts.items()},
    }
    if getattr(args, "out_dir", None):
        _save_json(Path(args.out_dir) / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
