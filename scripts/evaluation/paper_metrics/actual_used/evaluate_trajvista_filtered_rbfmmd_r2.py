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

from tools.dataset_building.evaluate_trajvista_rbfmmd_r2 import (
    compute_r2_old,
    compute_rbf_mmd_old,
    load_checkpoint_model,
)
from tools.dataset_building.train_trajvista_gantl_wgan_tl_ddp import AccDataset
from tools.dataset_building.train_trajvista_highd_psiphi_smoke import TransitionDataset
from tools.dataset_building.train_trajvista_psiphi_hybrid_full_domain import concat_arrays, load_dataset_split


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "DE_CN_to_US": {
        "target_datasets": ["NGSIM", "CitySim"],
        "source_label": "DE+CN",
        "target_domain": "US",
    },
    "DE_US_to_CN": {
        "target_datasets": ["DJI", "sinD"],
        "source_label": "DE+US",
        "target_domain": "CN",
    },
    "CN_US_to_DE": {
        "target_datasets": ["HighD", "inD"],
        "source_label": "CN+US",
        "target_domain": "DE",
    },
}


MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "model": "Ours",
        "model_type": "hybrid",
        "checkpoint": "runs/revision_loro_pooled_source/{experiment}/f005/finetune/best_model.pt",
    },
    {
        "model": "GAN-TL",
        "model_type": "gantl",
        "checkpoint": "runs/baselines/gantl_wgan_tl_loro_pooled_source/{experiment}/f005/transfer/best_model.pt",
    },
    {
        "model": "FD-Align",
        "model_type": "fdalign_legacy",
        "checkpoint": "runs/baselines/fd_align_loro_pooled_source_legacy/{experiment}/f005/transfer/best_model.pt",
    },
    {
        "model": "GT-MMD",
        "model_type": "gtmmd",
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


def make_subset_mask(arrays: dict[str, np.ndarray], labels: np.ndarray, subset: str, top_frac: float) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(arrays["action"])
    if subset == "all":
        return np.ones(n, dtype=bool), {"subset": subset}
    if subset.startswith("dataset:"):
        dataset = subset.split(":", 1)[1]
        return labels == dataset, {"subset": subset, "dataset": dataset}
    if subset == "high_speed_top":
        if not (0.0 < top_frac < 1.0):
            raise ValueError("--top-frac must be in (0, 1) for high_speed_top")
        last = arrays["state"][:, -1, :]
        speed = np.linalg.norm(last[:, [0, 2]], axis=1)
        threshold = float(np.quantile(speed, 1.0 - top_frac))
        return speed >= threshold, {"subset": subset, "top_frac": top_frac, "speed_threshold": threshold}
    raise ValueError(f"unsupported subset={subset!r}")


@torch.no_grad()
def predict_for_model(
    arrays: dict[str, np.ndarray],
    checkpoint: Path,
    model_type: str,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model, stats, _payload = load_checkpoint_model(checkpoint, device, model_type)
    dataset = AccDataset(arrays, stats) if model_type == "gantl" else TransitionDataset(arrays, stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    action_mean = torch.tensor(stats.action_mean, dtype=torch.float32, device=device)
    action_std = torch.tensor(stats.action_std, dtype=torch.float32, device=device)
    preds = []
    targets = []
    for batch in loader:
        if model_type == "gantl":
            state, _action, action_raw, _state_raw = batch
        else:
            state, _action, _next_state, _next_action, action_raw, _state_raw = batch
        pred_norm = model(state.to(device))
        pred = pred_norm * action_std + action_mean
        preds.append(pred.detach().cpu())
        targets.append(action_raw.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def evaluate_prediction(pred: torch.Tensor, target: torch.Tensor, mask: np.ndarray) -> dict[str, Any]:
    mask_t = torch.from_numpy(mask.astype(bool))
    pred_sel = pred[mask_t]
    target_sel = target[mask_t]
    r2_dims, r2 = compute_r2_old(pred_sel, target_sel)
    mmd2 = compute_rbf_mmd_old(pred_sel, target_sel)
    rmse = torch.sqrt(torch.mean((pred_sel - target_sel) ** 2, dim=0))
    return {
        "samples": int(pred_sel.shape[0]),
        "r2_acc": float(r2),
        "r2_ax": float(r2_dims[0]),
        "r2_ay": float(r2_dims[1]),
        "rbf_mmd2_acc": float(mmd2),
        "rbf_mmd_acc": float(mmd2**0.5),
        "rmse_x": float(rmse[0].item()),
        "rmse_y": float(rmse[1].item()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "experiment",
        "source_label",
        "target_domain",
        "target_datasets",
        "subset",
        "subset_detail",
        "model",
        "model_type",
        "samples",
        "rbf_mmd_acc",
        "rbf_mmd2_acc",
        "r2_acc",
        "r2_ax",
        "r2_ay",
        "rmse_x",
        "rmse_y",
        "checkpoint",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LORO checkpoints on a transparent filtered target subset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("reference_paths_work/model_dataset_full/trajvista_xy_psiphi_v1"))
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="DE_CN_to_US")
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", choices=["all", "high_speed_top"], default="high_speed_top")
    parser.add_argument("--top-frac", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment]
    arrays, labels = load_target_arrays(args.dataset_dir, exp["target_datasets"], args.split)
    mask, subset_meta = make_subset_mask(arrays, labels, args.subset, args.top_frac)
    if int(mask.sum()) < 2:
        raise ValueError(f"subset selected too few samples: {int(mask.sum())}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        checkpoint = args.dataset_dir / spec["checkpoint"].format(experiment=args.experiment)
        if not checkpoint.exists():
            continue
        pred, target = predict_for_model(arrays, checkpoint, spec["model_type"], device, args.batch_size)
        metrics = evaluate_prediction(pred, target, mask)
        rows.append(
            {
                "experiment": args.experiment,
                "source_label": exp["source_label"],
                "target_domain": exp["target_domain"],
                "target_datasets": "+".join(exp["target_datasets"]),
                "subset": args.subset,
                "subset_detail": json.dumps(subset_meta, sort_keys=True),
                "model": spec["model"],
                "model_type": spec["model_type"],
                "checkpoint": str(checkpoint),
                **metrics,
            }
        )

    write_csv(args.out, rows)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
