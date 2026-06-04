from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building.evaluate_trajvista_complete_trajectory_rollout import (
    ACTION_FEATURE_INDICES,
    DATASET_DT,
    EXPERIMENTS,
    HISTORY,
    VELOCITY_FEATURE_INDICES,
    default_model_rows,
    reconstruct_dataset_trajectories,
)
from tools.dataset_building.evaluate_trajvista_rbfmmd_r2 import load_checkpoint_model


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_traj_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty trajectory rows")
    ade = np.asarray([float(row["ade"]) for row in rows], dtype=np.float64)
    fde = np.asarray([float(row["fde"]) for row in rows], dtype=np.float64)
    complete = np.asarray([int(row["complete"]) for row in rows], dtype=np.float64)
    pred_steps = np.asarray([int(row["pred_steps"]) for row in rows], dtype=np.float64)
    path_len = np.asarray([float(row["true_path_len"]) for row in rows], dtype=np.float64)
    return {
        "trajectories": int(len(rows)),
        "pred_steps": int(pred_steps.sum()),
        "ade_mean": float(ade.mean()),
        "ade_median": float(np.median(ade)),
        "ade_p90": float(np.percentile(ade, 90)),
        "fde_mean": float(fde.mean()),
        "fde_median": float(np.median(fde)),
        "fde_p90": float(np.percentile(fde, 90)),
        "fde_p95": float(np.percentile(fde, 95)),
        "fde_p99": float(np.percentile(fde, 99)),
        "cr": float(complete.mean()),
        "true_path_len_mean": float(path_len.mean()),
        "steps_per_traj_mean": float(pred_steps.mean()),
    }


@torch.no_grad()
def rollout_model_trajectory_metrics(
    model: torch.nn.Module,
    stats: Any,
    trajectories: list[Any],
    dt: float,
    device: torch.device,
    batch_size: int,
    mode: str,
    cr_rel_threshold: float,
    cr_abs_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in {"semi", "teacher"}:
        raise ValueError("mode must be semi or teacher")
    if not trajectories:
        raise ValueError("no trajectories to evaluate")

    state_mean = torch.tensor(stats.state_mean, dtype=torch.float32, device=device)
    state_std = torch.tensor(stats.state_std, dtype=torch.float32, device=device)
    action_mean = torch.tensor(stats.action_mean, dtype=torch.float32, device=device)
    action_std = torch.tensor(stats.action_std, dtype=torch.float32, device=device)

    hist = np.stack([seg.features[:HISTORY] for seg in trajectories]).astype(np.float32)
    prev_v = hist[:, -1, list(VELOCITY_FEATURE_INDICES)].astype(np.float32).copy()
    pred_disp = np.zeros((len(trajectories), 2), dtype=np.float64)
    true_disp = np.zeros((len(trajectories), 2), dtype=np.float64)
    ade_sum = np.zeros((len(trajectories),), dtype=np.float64)
    pred_steps = np.zeros((len(trajectories),), dtype=np.int64)

    max_steps = max(seg.pred_steps for seg in trajectories)
    sum_sq = np.zeros(2, dtype=np.float64)
    sum_abs = np.zeros(2, dtype=np.float64)
    total_steps = 0

    for step in range(max_steps):
        active = np.asarray([i for i, seg in enumerate(trajectories) if step < seg.pred_steps], dtype=np.int64)
        if active.size == 0:
            continue

        preds = []
        for start in range(0, active.size, batch_size):
            idx = active[start : start + batch_size]
            state_raw = torch.from_numpy(hist[idx]).to(device=device, dtype=torch.float32)
            state_norm = (state_raw - state_mean) / state_std
            pred_norm = model(state_norm)
            pred_raw = pred_norm * action_std + action_mean
            preds.append(pred_raw.detach().cpu().numpy().astype(np.float32))
        pred_acc = np.concatenate(preds, axis=0)

        true_feat = np.stack([trajectories[int(i)].features[HISTORY + step] for i in active]).astype(np.float32)
        true_acc = true_feat[:, list(ACTION_FEATURE_INDICES)]
        true_v = true_feat[:, list(VELOCITY_FEATURE_INDICES)]

        err = pred_acc - true_acc
        sum_sq += np.sum(err.astype(np.float64) ** 2, axis=0)
        sum_abs += np.sum(np.abs(err).astype(np.float64), axis=0)
        total_steps += int(active.size)

        pred_v = prev_v[active] + pred_acc * float(dt)
        pred_disp[active] += pred_v.astype(np.float64) * float(dt)
        true_disp[active] += true_v.astype(np.float64) * float(dt)
        disp_err = np.linalg.norm(pred_disp[active] - true_disp[active], axis=1)
        ade_sum[active] += disp_err
        pred_steps[active] += 1

        if mode == "semi":
            next_feat = true_feat.copy()
            next_feat[:, VELOCITY_FEATURE_INDICES[0]] = pred_v[:, 0]
            next_feat[:, VELOCITY_FEATURE_INDICES[1]] = pred_v[:, 1]
            next_feat[:, ACTION_FEATURE_INDICES[0]] = true_acc[:, 0]
            next_feat[:, ACTION_FEATURE_INDICES[1]] = true_acc[:, 1]
            prev_v[active] = pred_v
        else:
            next_feat = true_feat
            prev_v[active] = true_v

        hist[active, :-1] = hist[active, 1:]
        hist[active, -1] = next_feat

    fde = np.linalg.norm(pred_disp - true_disp, axis=1)
    ade = ade_sum / np.maximum(pred_steps, 1)
    path_len = np.linalg.norm(true_disp, axis=1)
    cr_threshold = np.maximum(float(cr_abs_threshold), float(cr_rel_threshold) * path_len)
    complete = fde <= cr_threshold

    rows: list[dict[str, Any]] = []
    for i, seg in enumerate(trajectories):
        rows.append(
            {
                "dataset": seg.dataset,
                "scene": seg.scene,
                "track_id": seg.track_id,
                "frame_k_start": int(seg.frame_k_start),
                "frame_k_end": int(seg.frame_k_end),
                "windows": int(seg.windows),
                "pred_steps": int(pred_steps[i]),
                "ade": float(ade[i]),
                "fde": float(fde[i]),
                "true_path_len": float(path_len[i]),
                "complete": int(bool(complete[i])),
                "cr_threshold": float(cr_threshold[i]),
            }
        )

    rmse = np.sqrt(sum_sq / max(total_steps, 1))
    mae = sum_abs / max(total_steps, 1)
    aggregate = aggregate_traj_rows(rows)
    aggregate.update(
        {
            "acc_rmse_x": float(rmse[0]),
            "acc_rmse_y": float(rmse[1]),
            "acc_mae_x": float(mae[0]),
            "acc_mae_y": float(mae[1]),
        }
    )
    return rows, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Individual-level LORO rollout metrics on reconstructed complete test trajectories.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("reference_paths_work/model_dataset_full/trajvista_xy_psiphi_v1"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_metrics/individual_loro_rollout"))
    parser.add_argument("--mode", choices=["semi", "teacher"], default="semi")
    parser.add_argument("--min-pred-steps", type=int, default=50)
    parser.add_argument("--max-trajectories-per-dataset", type=int, default=0, help="0 means use all reconstructed trajectories.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiments", nargs="*", default=[exp["experiment"] for exp in EXPERIMENTS])
    parser.add_argument("--cr-rel-threshold", type=float, default=0.05)
    parser.add_argument("--cr-abs-threshold", type=float, default=2.0)
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected_experiments = set(args.experiments)
    experiment_meta = {exp["experiment"]: exp for exp in EXPERIMENTS}
    needed_datasets = sorted(
        {
            ds
            for exp in EXPERIMENTS
            if exp["experiment"] in selected_experiments
            for ds in exp["target_datasets"]
        }
    )

    trajectory_cache: dict[str, list[Any]] = {}
    data_meta: dict[str, Any] = {}
    for dataset in needed_datasets:
        segments, meta = reconstruct_dataset_trajectories(
            args.dataset_dir,
            dataset,
            args.split,
            args.min_pred_steps,
            args.max_trajectories_per_dataset,
            args.seed,
        )
        trajectory_cache[dataset] = segments
        data_meta[dataset] = meta
        print(json.dumps({"event": "dataset_ready", **meta}), flush=True)

    dataset_rows: list[dict[str, Any]] = []
    experiment_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    grouped_for_experiment: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in default_model_rows(args.dataset_dir):
        experiment = row["experiment"]
        if experiment not in selected_experiments:
            continue
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model, stats, payload = load_checkpoint_model(checkpoint, device, row["model_type"])
        model.eval()

        for dataset in row["target_datasets"]:
            segments = trajectory_cache[dataset]
            per_traj, metrics = rollout_model_trajectory_metrics(
                model,
                stats,
                segments,
                DATASET_DT[dataset],
                device,
                args.batch_size,
                args.mode,
                args.cr_rel_threshold,
                args.cr_abs_threshold,
            )
            annotated = []
            for item in per_traj:
                out = {
                    "experiment": experiment,
                    "target_country": "+".join(row["target_datasets"]),
                    "model": row["model_name"],
                    "model_type": row["model_type"],
                    **item,
                }
                annotated.append(out)
            trajectory_rows.extend(annotated)
            grouped_for_experiment.setdefault((experiment, row["model_name"]), []).extend(annotated)

            out_row = {
                "experiment": experiment,
                "target_country": "+".join(row["target_datasets"]),
                "target_dataset": dataset,
                "model": row["model_name"],
                "model_type": row["model_type"],
                "mode": args.mode,
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": payload.get("epoch", ""),
                **metrics,
            }
            dataset_rows.append(out_row)
            print(json.dumps({"event": "model_dataset_done", **out_row}), flush=True)

    for (experiment, model_name), rows in grouped_for_experiment.items():
        agg = aggregate_traj_rows(rows)
        meta = experiment_meta[experiment]
        model_type = next(row["model_type"] for row in dataset_rows if row["experiment"] == experiment and row["model"] == model_name)
        experiment_rows.append(
            {
                "experiment": experiment,
                "source_target": f"{experiment.replace('_to_', ' -> ')}",
                "target_datasets": "+".join(meta["target_datasets"]),
                "model": model_name,
                "model_type": model_type,
                "mode": args.mode,
                **agg,
            }
        )

    model_order = {"Ours": 0, "GAN-TL": 1, "FD-Align": 2, "GT-MMD": 3, "Localized": 4}
    experiment_rows.sort(key=lambda r: (str(r["experiment"]), model_order.get(str(r["model"]), 99)))
    dataset_rows.sort(key=lambda r: (str(r["experiment"]), str(r["target_dataset"]), model_order.get(str(r["model"]), 99)))

    summary = {
        "schema": "trajvista_individual_loro_rollout_v1",
        "created_at_unix": time.time(),
        "elapsed_sec": time.time() - started,
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "mode": args.mode,
        "trajectory_policy": {
            "reconstruction": "stride=12 windows tiled by dataset/scene/track_id/meta_frame_k into complete test trajectories",
            "semi": "model-predicted acceleration integrates velocity; non-velocity context and acceleration slots are teacher-forced from the next frame",
            "ade": "mean Euclidean displacement error over rollout steps",
            "fde": "final Euclidean displacement error after the reconstructed trajectory rollout",
            "cr": "fraction of trajectories with FDE <= max(cr_abs_threshold, cr_rel_threshold * true_path_len)",
            "cr_abs_threshold": float(args.cr_abs_threshold),
            "cr_rel_threshold": float(args.cr_rel_threshold),
        },
        "dt_by_dataset": DATASET_DT,
        "data_meta": data_meta,
        "experiment_rows": experiment_rows,
        "dataset_rows": dataset_rows,
    }

    json_path = args.out_dir / f"individual_loro_rollout_{args.mode}.json"
    experiment_csv = args.out_dir / f"individual_loro_rollout_{args.mode}_experiment.csv"
    dataset_csv = args.out_dir / f"individual_loro_rollout_{args.mode}_dataset.csv"
    traj_csv = args.out_dir / f"individual_loro_rollout_{args.mode}_per_trajectory.csv"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_csv(experiment_csv, experiment_rows)
    write_csv(dataset_csv, dataset_rows)
    write_csv(traj_csv, trajectory_rows)
    print(
        json.dumps(
            {
                "event": "done",
                "json": str(json_path),
                "experiment_csv": str(experiment_csv),
                "dataset_csv": str(dataset_csv),
                "trajectory_csv": str(traj_csv),
                "experiment_rows": len(experiment_rows),
                "dataset_rows": len(dataset_rows),
                "trajectory_rows": len(trajectory_rows),
                "elapsed_sec": summary["elapsed_sec"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
