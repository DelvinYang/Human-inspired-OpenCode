from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building.build_trajvista_highd_smoke import FEATURE_NAMES
from tools.dataset_building.evaluate_trajvista_rbfmmd_r2 import load_checkpoint_model


HISTORY = 12
ACTION_FEATURE_INDICES = (FEATURE_NAMES.index("xAcceleration"), FEATURE_NAMES.index("yAcceleration"))
VELOCITY_FEATURE_INDICES = (FEATURE_NAMES.index("xVelocity"), FEATURE_NAMES.index("yVelocity"))

DATASET_DT = {
    "HighD": 1.0 / 25.0,
    "inD": 1.0 / 25.0,
    "CitySim": 1.0 / 30.0,
    "DJI": 1.0 / 30.0,
    "sinD": 1.0 / 10.0,
    "NGSIM": 1.0 / 10.0,
}

EXPERIMENTS = [
    {"experiment": "DE_CN_to_US", "target_datasets": ["NGSIM", "CitySim"]},
    {"experiment": "DE_US_to_CN", "target_datasets": ["DJI", "sinD"]},
    {"experiment": "CN_US_to_DE", "target_datasets": ["HighD", "inD"]},
]

MODEL_SPECS = [
    {
        "model_name": "Ours",
        "model_type": "hybrid",
        "checkpoint": "runs/revision_loro_pooled_source/{experiment}/f005/finetune/best_model.pt",
    },
    {
        "model_name": "GAN-TL",
        "model_type": "gantl",
        "checkpoint": "runs/baselines/gantl_wgan_tl_loro_pooled_source/{experiment}/f005/transfer/best_model.pt",
    },
    {
        "model_name": "FD-Align",
        "model_type": "fdalign_legacy",
        "checkpoint": "runs/baselines/fd_align_loro_pooled_source_legacy/{experiment}/f005/transfer/best_model.pt",
    },
    {
        "model_name": "GT-MMD",
        "model_type": "gtmmd",
        "checkpoint": "runs/baselines/gt_mmd_loro_pooled_source_direct_lamt20/{experiment}/f005/transfer/best_model.pt",
    },
]


@dataclass
class TrajectorySegment:
    dataset: str
    scene: str
    track_id: str
    frame_k_start: int
    frame_k_end: int
    windows: int
    features: np.ndarray

    @property
    def pred_steps(self) -> int:
        return int(self.features.shape[0] - HISTORY)


def split_contiguous_stride12(frames: np.ndarray) -> list[tuple[int, int]]:
    if len(frames) == 0:
        return []
    cuts = np.where(np.diff(frames) != HISTORY)[0] + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(frames)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def reconstruct_dataset_trajectories(
    root: Path,
    dataset: str,
    split: str,
    min_pred_steps: int,
    max_trajectories: int,
    seed: int,
) -> tuple[list[TrajectorySegment], dict[str, Any]]:
    paths = sorted((root / split / dataset).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"missing {split}/{dataset} under {root}")

    states, next_states, scenes, tracks, frames = [], [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            states.append(z["state"].astype(np.float32, copy=False))
            next_states.append(z["next_state"].astype(np.float32, copy=False))
            scenes.append(z["meta_scene"].astype(str))
            tracks.append(z["meta_track_id"].astype(str))
            frames.append(z["meta_frame_k"].astype(np.int64))
    state = np.concatenate(states, axis=0)
    next_state = np.concatenate(next_states, axis=0)
    scene_arr = np.concatenate(scenes, axis=0)
    track_arr = np.concatenate(tracks, axis=0)
    frame_arr = np.concatenate(frames, axis=0)

    groups: dict[tuple[str, str], list[int]] = {}
    for idx, key in enumerate(zip(scene_arr.tolist(), track_arr.tolist())):
        groups.setdefault(key, []).append(idx)

    descriptors: list[tuple[str, str, np.ndarray]] = []
    total_segments = 0
    short_segments = 0
    tiling_checks = 0
    tiling_bad = 0
    tiling_max_abs_diff = 0.0
    for (scene, track_id), idxs in groups.items():
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        order = np.argsort(frame_arr[idxs_arr], kind="stable")
        idxs_arr = idxs_arr[order]
        group_frames = frame_arr[idxs_arr]
        for s, e in split_contiguous_stride12(group_frames):
            total_segments += 1
            seg_idxs = idxs_arr[s:e]
            pred_steps = HISTORY * len(seg_idxs) - HISTORY + 1
            if pred_steps < min_pred_steps:
                short_segments += 1
                continue
            if len(seg_idxs) > 1:
                left = next_state[seg_idxs[:-1], -1]
                right = state[seg_idxs[1:], 0]
                diff = np.max(np.abs(left - right), axis=1)
                tiling_checks += int(diff.size)
                tiling_bad += int(np.count_nonzero(diff > 1e-4))
                if diff.size:
                    tiling_max_abs_diff = max(tiling_max_abs_diff, float(diff.max()))
            descriptors.append((scene, track_id, seg_idxs))

    rng = np.random.default_rng(seed)
    if max_trajectories > 0 and len(descriptors) > max_trajectories:
        keep = rng.choice(len(descriptors), size=max_trajectories, replace=False)
        descriptors = [descriptors[int(i)] for i in keep]

    segments: list[TrajectorySegment] = []
    for scene, track_id, seg_idxs in descriptors:
        seg_frames = frame_arr[seg_idxs]
        parts = [state[seg_idxs[0]]]
        for idx in seg_idxs[1:]:
            parts.append(state[idx])
        parts.append(next_state[seg_idxs[-1], -1:])  # one final frame after the last history window
        features = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        segments.append(
            TrajectorySegment(
                dataset=dataset,
                scene=scene,
                track_id=track_id,
                frame_k_start=int(seg_frames[0]),
                frame_k_end=int(seg_frames[-1]),
                windows=int(len(seg_idxs)),
                features=features,
            )
        )

    pred_steps = np.asarray([seg.pred_steps for seg in segments], dtype=np.int64)
    meta = {
        "dataset": dataset,
        "split": split,
        "npz_files": len(paths),
        "samples": int(len(frame_arr)),
        "tracks": int(len(groups)),
        "segments_stride12_total": int(total_segments),
        "segments_shorter_than_min_pred_steps": int(short_segments),
        "segments_selected": int(len(segments)),
        "max_trajectories": int(max_trajectories),
        "min_pred_steps": int(min_pred_steps),
        "tiling_checks": int(tiling_checks),
        "tiling_bad_gt_1e-4": int(tiling_bad),
        "tiling_max_abs_diff": float(tiling_max_abs_diff),
        "pred_steps_p50_p90_p99_max": (
            [float(np.percentile(pred_steps, q)) for q in (50, 90, 99)] + [int(pred_steps.max())]
            if pred_steps.size
            else []
        ),
    }
    return segments, meta


@torch.no_grad()
def rollout_model(
    model: torch.nn.Module,
    stats: Any,
    trajectories: list[TrajectorySegment],
    dt: float,
    device: torch.device,
    batch_size: int,
    mode: str,
) -> dict[str, Any]:
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
    fde_steps = np.zeros((len(trajectories),), dtype=np.int64)

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
        fde_steps[active] += 1

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
    path_len = np.linalg.norm(true_disp, axis=1)
    rmse = np.sqrt(sum_sq / max(total_steps, 1))
    mae = sum_abs / max(total_steps, 1)
    return {
        "mode": mode,
        "trajectories": int(len(trajectories)),
        "pred_steps": int(total_steps),
        "acc_rmse": rmse.astype(float).tolist(),
        "acc_mae": mae.astype(float).tolist(),
        "fde_mean": float(np.mean(fde)),
        "fde_median": float(np.median(fde)),
        "fde_p90": float(np.percentile(fde, 90)),
        "fde_p95": float(np.percentile(fde, 95)),
        "fde_p99": float(np.percentile(fde, 99)),
        "fde_max": float(np.max(fde)),
        "true_path_len_mean": float(np.mean(path_len)),
        "pred_path_len_mean": float(np.mean(np.linalg.norm(pred_disp, axis=1))),
        "steps_per_traj_mean": float(np.mean(fde_steps)),
        "steps_per_traj_p90": float(np.percentile(fde_steps, 90)),
    }


def default_model_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for exp in EXPERIMENTS:
        for spec in MODEL_SPECS:
            ck = dataset_dir / spec["checkpoint"].format(experiment=exp["experiment"])
            rows.append(
                {
                    "experiment": exp["experiment"],
                    "target_datasets": exp["target_datasets"],
                    "model_name": spec["model_name"],
                    "model_type": spec["model_type"],
                    "checkpoint": ck,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Old-style semi rollout on reconstructed complete test trajectories.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("reference_paths_work/model_dataset_full/trajvista_xy_psiphi_v1"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", type=Path, default=Path("reference_paths_work/model_dataset_full/trajvista_xy_psiphi_v1/runs/trajectory_rollout"))
    parser.add_argument("--mode", choices=["semi", "teacher"], default="semi")
    parser.add_argument("--min-pred-steps", type=int, default=50)
    parser.add_argument("--max-trajectories-per-dataset", type=int, default=0, help="0 means use all reconstructed trajectories.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiments", nargs="*", default=[exp["experiment"] for exp in EXPERIMENTS])
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    needed_datasets = sorted({ds for exp in EXPERIMENTS if exp["experiment"] in set(args.experiments) for ds in exp["target_datasets"]})
    trajectory_cache: dict[str, list[TrajectorySegment]] = {}
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

    result_rows = []
    detail: dict[str, Any] = {}
    for row in default_model_rows(args.dataset_dir):
        if row["experiment"] not in set(args.experiments):
            continue
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model, stats, payload = load_checkpoint_model(checkpoint, device, row["model_type"])
        model.eval()
        exp_key = row["experiment"]
        for dataset in row["target_datasets"]:
            segments = trajectory_cache[dataset]
            metrics = rollout_model(model, stats, segments, DATASET_DT[dataset], device, args.batch_size, args.mode)
            out_row = {
                "experiment": exp_key,
                "target_dataset": dataset,
                "model": row["model_name"],
                "model_type": row["model_type"],
                "mode": args.mode,
                "trajectories": metrics["trajectories"],
                "pred_steps": metrics["pred_steps"],
                "acc_rmse_x": metrics["acc_rmse"][0],
                "acc_rmse_y": metrics["acc_rmse"][1],
                "acc_rmse_mean": float((metrics["acc_rmse"][0] + metrics["acc_rmse"][1]) / 2.0),
                "acc_mae_x": metrics["acc_mae"][0],
                "acc_mae_y": metrics["acc_mae"][1],
                "fde_mean": metrics["fde_mean"],
                "fde_median": metrics["fde_median"],
                "fde_p90": metrics["fde_p90"],
                "fde_p95": metrics["fde_p95"],
                "fde_p99": metrics["fde_p99"],
                "fde_max": metrics["fde_max"],
                "true_path_len_mean": metrics["true_path_len_mean"],
                "pred_path_len_mean": metrics["pred_path_len_mean"],
                "checkpoint": str(checkpoint),
            }
            result_rows.append(out_row)
            detail.setdefault(exp_key, {})[f"{row['model_name']}::{dataset}"] = metrics
            print(json.dumps({"event": "model_dataset_done", **out_row}), flush=True)

    summary = {
        "schema": "trajvista_complete_trajectory_rollout_v1",
        "created_at_unix": time.time(),
        "elapsed_sec": time.time() - started,
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "mode": args.mode,
        "rollout_policy": {
            "source": "old TrajVista evaluate_acc_new semi mode",
            "semi": "model predicts ax/ay; predicted acceleration integrates x/y velocity; history keeps true acceleration and all non-velocity context features",
            "teacher": "history remains fully ground truth; included only for acceleration-sequence diagnosis",
            "trajectory_reconstruction": "current stride=12 windows are tiled by same track and meta_frame_k diff=12 into frame-level trajectories",
        },
        "dt_by_dataset": DATASET_DT,
        "data_meta": data_meta,
        "rows": result_rows,
        "detail": detail,
    }
    json_path = args.out_dir / f"complete_trajectory_rollout_{args.mode}.json"
    csv_path = args.out_dir / f"complete_trajectory_rollout_{args.mode}.csv"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_csv(csv_path, result_rows)
    print(json.dumps({"event": "done", "json": str(json_path), "csv": str(csv_path), "rows": len(result_rows), "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
