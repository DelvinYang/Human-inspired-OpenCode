from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cultural_align.data.schema import FEATURE_NAMES
from cultural_align.training.data import NormStats


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
    "INTERACTION_CHN": 1.0 / 10.0,
    "INTERACTION_DEU": 1.0 / 10.0,
    "INTERACTION_USA": 1.0 / 10.0,
}


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


def compute_va95(v_arr: np.ndarray, a_arr: np.ndarray) -> float:
    if v_arr.size == 0 or a_arr.size == 0:
        return float("nan")
    v_norm = np.linalg.norm(v_arr, axis=1) + 1e-9
    a_norm = np.linalg.norm(a_arr, axis=1)
    product = v_norm * a_norm
    return float(np.quantile(product, 0.95)) if product.size else float("nan")


def compute_rpa(v_arr: np.ndarray, a_arr: np.ndarray, dt: float) -> float:
    if v_arr.size == 0 or a_arr.size == 0:
        return float("nan")
    v_norm = np.linalg.norm(v_arr, axis=1)
    a_norm = np.linalg.norm(a_arr, axis=1)
    numerator = float(np.sum(v_norm * np.maximum(a_norm, 0.0) * float(dt)))
    denominator = float(np.sum(v_norm * float(dt)))
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _rmse(pred: np.ndarray, truth: np.ndarray) -> tuple[float, int]:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(truth)
    if not np.any(mask):
        return float("nan"), 0
    diff = pred[mask] - truth[mask]
    return float(np.sqrt(np.mean(diff * diff))), int(mask.sum())


def split_contiguous_stride12(frames: np.ndarray) -> list[tuple[int, int]]:
    if len(frames) == 0:
        return []
    cuts = np.where(np.diff(frames) != HISTORY)[0] + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(frames)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def reconstruct_trajectory_segments(
    arrays: dict[str, np.ndarray],
    min_pred_steps: int = 50,
    max_trajectories_per_dataset: int = 0,
    seed: int = 20260517,
) -> tuple[list[TrajectorySegment], dict[str, Any]]:
    required = {"state", "next_state", "meta_scene", "meta_track_id", "meta_frame_k"}
    missing = sorted(required - set(arrays))
    if missing:
        return [], {"available": False, "reason": f"missing metadata arrays: {', '.join(missing)}"}

    state = arrays["state"].astype(np.float32, copy=False)
    next_state = arrays["next_state"].astype(np.float32, copy=False)
    if state.ndim != 3 or state.shape[1:] != next_state.shape[1:]:
        return [], {"available": False, "reason": "state and next_state must have matching [N,T,F] shapes"}

    dataset_arr = arrays.get("meta_dataset")
    if dataset_arr is None:
        dataset_arr = np.asarray(["UNKNOWN"] * len(state), dtype="U32")
    dataset_arr = dataset_arr.astype(str)
    scene_arr = arrays["meta_scene"].astype(str)
    track_arr = arrays["meta_track_id"].astype(str)
    frame_arr = arrays["meta_frame_k"].astype(np.int64)

    groups: dict[tuple[str, str, str], list[int]] = {}
    for idx, key in enumerate(zip(dataset_arr.tolist(), scene_arr.tolist(), track_arr.tolist())):
        groups.setdefault(key, []).append(idx)

    descriptors_by_dataset: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    total_segments = 0
    short_segments = 0
    tiling_checks = 0
    tiling_bad = 0
    tiling_max_abs_diff = 0.0
    for (dataset, scene, track_id), idxs in groups.items():
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        order = np.argsort(frame_arr[idxs_arr], kind="stable")
        idxs_arr = idxs_arr[order]
        group_frames = frame_arr[idxs_arr]
        for start, end in split_contiguous_stride12(group_frames):
            total_segments += 1
            seg_idxs = idxs_arr[start:end]
            pred_steps = HISTORY * len(seg_idxs) - HISTORY + 1
            if pred_steps < int(min_pred_steps):
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
            descriptors_by_dataset.setdefault(dataset, []).append((scene, track_id, seg_idxs))

    rng = np.random.default_rng(seed)
    selected_descriptors: list[tuple[str, str, str, np.ndarray]] = []
    for dataset, descriptors in sorted(descriptors_by_dataset.items()):
        if max_trajectories_per_dataset > 0 and len(descriptors) > max_trajectories_per_dataset:
            keep = rng.choice(len(descriptors), size=max_trajectories_per_dataset, replace=False)
            descriptors = [descriptors[int(i)] for i in keep]
        selected_descriptors.extend((dataset, scene, track_id, seg_idxs) for scene, track_id, seg_idxs in descriptors)

    segments: list[TrajectorySegment] = []
    for dataset, scene, track_id, seg_idxs in selected_descriptors:
        seg_frames = frame_arr[seg_idxs]
        parts = [state[seg_idxs[0]]]
        for idx in seg_idxs[1:]:
            parts.append(state[idx])
        parts.append(next_state[seg_idxs[-1], -1:])
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
        "available": bool(segments),
        "samples": int(len(state)),
        "tracks": int(len(groups)),
        "segments_stride12_total": int(total_segments),
        "segments_shorter_than_min_pred_steps": int(short_segments),
        "segments_selected": int(len(segments)),
        "max_trajectories_per_dataset": int(max_trajectories_per_dataset),
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


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    ade = np.asarray([float(row["ade"]) for row in rows], dtype=np.float64)
    fde = np.asarray([float(row["fde"]) for row in rows], dtype=np.float64)
    complete = np.asarray([int(row["complete"]) for row in rows], dtype=np.float64)
    pred_steps = np.asarray([int(row["pred_steps"]) for row in rows], dtype=np.float64)
    path_len = np.asarray([float(row["true_path_len"]) for row in rows], dtype=np.float64)
    va95_pred = np.asarray([float(row["va95_pred"]) for row in rows], dtype=np.float64)
    va95_true = np.asarray([float(row["va95_true"]) for row in rows], dtype=np.float64)
    rpa_pred = np.asarray([float(row["rpa_pred"]) for row in rows], dtype=np.float64)
    rpa_true = np.asarray([float(row["rpa_true"]) for row in rows], dtype=np.float64)
    va95_rmse, va95_n = _rmse(va95_pred, va95_true)
    rpa_rmse, rpa_n = _rmse(rpa_pred, rpa_true)
    return {
        "trajectories": int(len(rows)),
        "pred_steps": int(pred_steps.sum()),
        "ade": float(ade.mean()),
        "ade_mean": float(ade.mean()),
        "ade_median": float(np.median(ade)),
        "ade_p90": float(np.percentile(ade, 90)),
        "fde": float(fde.mean()),
        "fde_mean": float(fde.mean()),
        "fde_median": float(np.median(fde)),
        "fde_p90": float(np.percentile(fde, 90)),
        "fde_p95": float(np.percentile(fde, 95)),
        "cr": float(complete.mean()),
        "true_path_len_mean": float(path_len.mean()),
        "steps_per_traj_mean": float(pred_steps.mean()),
        "va95": float(va95_rmse),
        "va95_rmse": float(va95_rmse),
        "va95_count": int(va95_n),
        "va95_pred_mean": float(np.nanmean(va95_pred)),
        "va95_true_mean": float(np.nanmean(va95_true)),
        "rpa": float(rpa_rmse),
        "rpa_rmse": float(rpa_rmse),
        "rpa_count": int(rpa_n),
        "rpa_pred_mean": float(np.nanmean(rpa_pred)),
        "rpa_true_mean": float(np.nanmean(rpa_true)),
    }


@torch.no_grad()
def _rollout_dataset(
    model: torch.nn.Module,
    stats: NormStats,
    trajectories: list[TrajectorySegment],
    dt: float,
    device: torch.device,
    batch_size: int,
    mode: str,
    cr_rel_threshold: float,
    cr_abs_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in {"semi", "teacher"}:
        raise ValueError("mode must be 'semi' or 'teacher'")
    if not trajectories:
        return [], {}

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
    pred_v_list: list[list[np.ndarray]] = [[] for _ in trajectories]
    pred_a_list: list[list[np.ndarray]] = [[] for _ in trajectories]
    true_v_list: list[list[np.ndarray]] = [[] for _ in trajectories]
    true_a_list: list[list[np.ndarray]] = [[] for _ in trajectories]

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

        for row_idx, traj_idx in enumerate(active.tolist()):
            pred_v_list[traj_idx].append(pred_v[row_idx].copy())
            pred_a_list[traj_idx].append(pred_acc[row_idx].copy())
            true_v_list[traj_idx].append(true_v[row_idx].copy())
            true_a_list[traj_idx].append(true_acc[row_idx].copy())

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
        pv = np.asarray(pred_v_list[i], dtype=np.float32)
        pa = np.asarray(pred_a_list[i], dtype=np.float32)
        tv = np.asarray(true_v_list[i], dtype=np.float32)
        ta = np.asarray(true_a_list[i], dtype=np.float32)
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
                "va95_pred": compute_va95(pv, pa),
                "va95_true": compute_va95(tv, ta),
                "rpa_pred": compute_rpa(pv, pa, dt),
                "rpa_true": compute_rpa(tv, ta, dt),
            }
        )

    aggregate = _aggregate_rows(rows)
    rmse = np.sqrt(sum_sq / max(total_steps, 1))
    mae = sum_abs / max(total_steps, 1)
    aggregate.update(
        {
            "acc_rmse_x": float(rmse[0]),
            "acc_rmse_y": float(rmse[1]),
            "acc_mae_x": float(mae[0]),
            "acc_mae_y": float(mae[1]),
        }
    )
    return rows, aggregate


@torch.no_grad()
def complete_trajectory_metrics(
    model: torch.nn.Module,
    stats: NormStats,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 8192,
    mode: str = "semi",
    min_pred_steps: int = 50,
    cr_rel_threshold: float = 0.05,
    cr_abs_threshold: float = 2.0,
    max_trajectories_per_dataset: int = 0,
    seed: int = 20260517,
) -> dict[str, Any]:
    segments, reconstruction_meta = reconstruct_trajectory_segments(
        arrays,
        min_pred_steps=min_pred_steps,
        max_trajectories_per_dataset=max_trajectories_per_dataset,
        seed=seed,
    )
    if not segments:
        return {
            "trajectory_metrics_available": False,
            "trajectory_reconstruction": reconstruction_meta,
        }

    rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for dataset in sorted({seg.dataset for seg in segments}):
        if dataset not in DATASET_DT:
            continue
        dataset_segments = [seg for seg in segments if seg.dataset == dataset]
        dataset_traj_rows, dataset_metrics = _rollout_dataset(
            model,
            stats,
            dataset_segments,
            DATASET_DT[dataset],
            device,
            batch_size,
            mode,
            cr_rel_threshold,
            cr_abs_threshold,
        )
        rows.extend(dataset_traj_rows)
        dataset_rows.append({"dataset": dataset, "dt": DATASET_DT[dataset], **dataset_metrics})

    if not rows:
        return {
            "trajectory_metrics_available": False,
            "trajectory_reconstruction": reconstruction_meta,
            "reason": "no selected trajectories had a configured dataset dt",
        }

    aggregate = _aggregate_rows(rows)
    aggregate.update(
        {
            "trajectory_metrics_available": True,
            "trajectory_reconstruction": reconstruction_meta,
            "trajectory_metric_policy": {
                "mode": mode,
                "reconstruction": "stride=12 windows tiled by dataset/scene/track_id/meta_frame_k into complete test trajectories",
                "va95": "RMSE between predicted and true trajectory-level 95th percentile of |v|*|a|",
                "rpa": "RMSE between predicted and true trajectory-level relative positive acceleration",
                "ade": "mean Euclidean displacement error over semi-rollout steps",
                "fde": "final Euclidean displacement error after the reconstructed trajectory rollout",
                "cr": "fraction of trajectories with FDE <= max(cr_abs_threshold, cr_rel_threshold * true_path_len)",
                "cr_abs_threshold": float(cr_abs_threshold),
                "cr_rel_threshold": float(cr_rel_threshold),
            },
            "dataset_rows": dataset_rows,
        }
    )
    return aggregate
