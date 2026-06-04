"""Shared helpers for computing VA95 and RPA metrics on trajectory datasets."""

from __future__ import annotations

import math
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from scripts.common.model_utils import choose_fixed_model, load_model, set_seed

# ---------------------------------------------------------------------------
# Trajectory metrics
# ---------------------------------------------------------------------------

def compute_va95(v_arr: np.ndarray, a_arr: np.ndarray) -> float:
    """Compute VA95 = 95th percentile of |v| * |a|."""

    if v_arr.size == 0 or a_arr.size == 0:
        return float("nan")

    v_norm = np.linalg.norm(v_arr, axis=1) + 1e-9
    a_norm = np.linalg.norm(a_arr, axis=1)
    product = v_norm * a_norm
    return float(np.quantile(product, 0.95)) if product.size else float("nan")


def compute_rpa(v_arr: np.ndarray, a_arr: np.ndarray, dt: float) -> float:
    """Compute Relative Positive Acceleration (RPA) over an entire trajectory."""

    if v_arr.size == 0 or a_arr.size == 0:
        return float("nan")

    v_norm = np.linalg.norm(v_arr, axis=1)
    a_norm = np.linalg.norm(a_arr, axis=1)
    a_pos = np.maximum(a_norm, 0.0)

    numerator = float(np.sum(v_norm * a_pos * dt))
    denominator = float(np.sum(v_norm * dt))
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


@torch.no_grad()
def evaluate_vehicle(
    features: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    *,
    model: torch.nn.Module,
    device: torch.device,
    dt: float,
    min_frames: int,
    acc_limit: float = 10.0,
    mode: str = "semi",
) -> Optional[Tuple[float, float, float, float]]:
    """Run open-loop prediction for a single vehicle and return predicted/true (VA95, RPA)."""

    if features.ndim != 2 or features.shape[1] != 12:
        logger.warning("Unexpected feature shape: {}", features.shape)
        return None

    features = features.to(dtype=torch.float32, device="cpu")
    num_frames = features.shape[0]

    history = 12
    if num_frames < max(history + 1, min_frames):
        return None

    hist_seq: deque[torch.Tensor] = deque((features[idx].clone() for idx in range(history)), maxlen=history)

    a_pred_list: List[List[float]] = []
    v_pred_list: List[List[float]] = []

    for frame_idx in range(history, num_frames):
        true_ax = float(labels[frame_idx, 0].item()) if labels is not None and labels.shape[0] > frame_idx else 0.0
        true_ay = float(labels[frame_idx, 1].item()) if labels is not None and labels.shape[0] > frame_idx else 0.0

        hist_tensor = torch.stack(list(hist_seq), dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)
        accel_pred = model(hist_tensor).squeeze(0).to("cpu")
        ax_pred = float(accel_pred[0])
        ay_pred = float(accel_pred[1])

        vx_tm1 = float(hist_seq[-1][0].item())
        vy_tm1 = float(hist_seq[-1][2].item())

        v_pred_lon = vx_tm1 + ax_pred * dt
        v_pred_lat = vy_tm1 + ay_pred * dt

        a_pred_list.append([ax_pred, ay_pred])
        v_pred_list.append([v_pred_lon, v_pred_lat])

        next_feat = features[frame_idx].clone()
        if mode == "semi":
            next_feat[0] = v_pred_lon
            next_feat[1] = true_ax  # keep true acc
            next_feat[2] = v_pred_lat
            next_feat[3] = true_ay
        elif mode == "gt":
            # teacher forcing inputs; keep history as ground truth
            pass
        else:  # closed
            next_feat[0] = v_pred_lon
            next_feat[1] = ax_pred
            next_feat[2] = v_pred_lat
            next_feat[3] = ay_pred
        hist_seq.append(next_feat)

    if not a_pred_list:
        return None

    a_pred_arr = np.asarray(a_pred_list, dtype=np.float32)
    v_pred_arr = np.asarray(v_pred_list, dtype=np.float32)

    if np.any(np.abs(a_pred_arr) > acc_limit):
        return None

    va95_pred = compute_va95(v_pred_arr, a_pred_arr)
    rpa_pred = compute_rpa(v_pred_arr, a_pred_arr, dt)

    va95_true = float("nan")
    rpa_true = float("nan")
    if labels is not None:
        if labels.ndim == 2 and labels.shape[1] >= 2 and labels.shape[0] == num_frames:
            true_v = features[history:, [0, 2]].to("cpu").numpy()
            true_a = labels[history:, :2].to("cpu").numpy()
            if np.any(np.abs(true_a) > acc_limit):
                return None
            va95_true = compute_va95(true_v, true_a)
            rpa_true = compute_rpa(true_v, true_a, dt)
        else:
            logger.warning("Skipping true metrics due to label shape mismatch: {}", labels.shape)

    return va95_pred, rpa_pred, va95_true, rpa_true


def iter_available_prefixes(trajectory_dir: Path) -> Iterator[str]:
    """Yield all prefix identifiers inferred from trajectory pickle names."""

    pattern = "trajectory_samples_"
    for path in sorted(trajectory_dir.glob("trajectory_samples_*.pkl")):
        suffix = path.stem
        if suffix.startswith(pattern):
            yield suffix[len(pattern):]


def run_evaluation(
    *,
    prefixes: Sequence[str],
    trajectory_dir: Path,
    data_loader,
    model_paths: Sequence[Path],
    device: torch.device,
    min_frames: int,
    output_pkl: Path,
    test_size: int = 1000,
    acc_limit: float = 10.0,
    model_limit: int = 50,
    seed: Optional[int] = None,
    mode: str = "semi",
) -> None:
    """Evaluate VA95/RPA on a fixed sampled subset and save true + multi-model predictions."""

    history = 12
    rng = np.random.RandomState(seed if seed is not None else 42)
    model_paths = list(model_paths)
    if model_limit and len(model_paths) > model_limit:
        keep_indices = rng.choice(len(model_paths), size=model_limit, replace=False)
        model_paths = [model_paths[idx] for idx in keep_indices]
        logger.info(
            "Model cap enabled: randomly selected {} models out of {}",
            model_limit,
            len(keep_indices),
        )

    # Step 1: enumerate all available (prefix, ego) pairs
    all_pairs: List[Tuple[str, object]] = []
    for prefix in tqdm(prefixes, desc="Scan prefixes"):
        traj_path = trajectory_dir / f"trajectory_samples_{prefix}.pkl"
        if not traj_path.exists():
            logger.warning("Trajectory file missing for prefix {}: {}", prefix, traj_path)
            continue
        with traj_path.open("rb") as handle:
            trajectory_data = pickle.load(handle)
        if not trajectory_data:
            continue
        for ego_id in trajectory_data.keys():
            all_pairs.append((prefix, ego_id))

    if not all_pairs:
        raise RuntimeError("No trajectories found for evaluation.")
    if mode not in {"closed", "semi", "gt"}:
        raise ValueError("mode must be one of: closed, semi, gt")

    if len(all_pairs) > test_size:
        selected_indices = rng.choice(len(all_pairs), size=test_size, replace=False)
        selected_pairs = [all_pairs[idx] for idx in selected_indices]
    else:
        selected_pairs = all_pairs

    logger.info("Selected {} trajectories (requested {}).", len(selected_pairs), test_size)

    # Step 2: cache needed features/labels and dt per prefix
    needed_per_prefix: Dict[str, set] = {}
    for prefix, ego_id in selected_pairs:
        needed_per_prefix.setdefault(prefix, set()).add(ego_id)

    data_cache: Dict[str, Dict[object, Tuple[torch.Tensor, torch.Tensor]]] = {}
    dt_cache: Dict[str, float] = {}
    for prefix, ego_set in needed_per_prefix.items():
        traj_path = trajectory_dir / f"trajectory_samples_{prefix}.pkl"
        try:
            dt_value = data_loader(prefix)
            if isinstance(dt_value, tuple):
                dt = float(dt_value[1])
            else:
                dt = float(dt_value)
            dt_cache[prefix] = dt
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to load dt for prefix {}: {}", prefix, exc)
            continue

        with traj_path.open("rb") as handle:
            trajectory_data = pickle.load(handle)

        subset: Dict[object, Tuple[torch.Tensor, torch.Tensor]] = {}
        for ego_id in ego_set:
            if ego_id in trajectory_data:
                subset[ego_id] = trajectory_data[ego_id]
        data_cache[prefix] = subset

    # Step 3: compute true metrics once for the selected set
    def _true_metrics(features: torch.Tensor, labels: torch.Tensor, dt: float) -> Optional[Tuple[float, float]]:
        if features.ndim != 2 or features.shape[1] != 12:
            return None
        if labels.ndim != 2 or labels.shape[1] < 2:
            return None
        num_frames = features.shape[0]
        if num_frames < max(history + 1, min_frames):
            return None
        true_v = features[history:, [0, 2]].to("cpu").numpy()
        true_a = labels[history:, :2].to("cpu").numpy()
        if np.any(np.abs(true_a) > acc_limit):
            return None
        return compute_va95(true_v, true_a), compute_rpa(true_v, true_a, dt)

    valid_samples: List[Tuple[str, object]] = []
    all_ids: List[str] = []
    all_va95_true: List[float] = []
    all_rpa_true: List[float] = []

    for prefix, ego_id in tqdm(selected_pairs, desc="True metrics"):
        if prefix not in data_cache or ego_id not in data_cache[prefix]:
            continue
        features, labels = data_cache[prefix][ego_id]
        dt = dt_cache.get(prefix)
        if dt is None:
            continue
        true_result = _true_metrics(features, labels, dt)
        if true_result is None or not all(math.isfinite(val) for val in true_result):
            continue
        va95_true, rpa_true = true_result
        valid_samples.append((prefix, ego_id))
        all_ids.append(f"{prefix}:{ego_id}")
        all_va95_true.append(va95_true)
        all_rpa_true.append(rpa_true)

    if not valid_samples:
        raise RuntimeError("No valid trajectories after filtering.")

    logger.info("Valid trajectories for evaluation: {}", len(valid_samples))

    # Step 4: evaluate each model on the same subset
    model_results: Dict[str, Dict[str, np.ndarray]] = {}
    for model_path in model_paths:
        model_name = model_path.stem
        logger.info("Evaluating model: {}", model_path)
        model = load_model(model_path, device)
        va95_pred_list: List[float] = []
        rpa_pred_list: List[float] = []

        for prefix, ego_id in tqdm(valid_samples, desc=f"Model {model_name}", leave=False):
            features, labels = data_cache[prefix][ego_id]
            dt = dt_cache[prefix]
            try:
                result = evaluate_vehicle(
                    features,
                    labels,
                    model=model,
                    device=device,
                    dt=dt,
                    min_frames=min_frames,
                    acc_limit=acc_limit,
                    mode=mode,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Evaluation failed for ego {} in prefix {} with model {}: {}", ego_id, prefix, model_name, exc)
                result = None

            if result is None:
                va95_pred, rpa_pred = float("nan"), float("nan")
            else:
                va95_pred, rpa_pred, _va_true, _ra_true = result
            va95_pred_list.append(va95_pred)
            rpa_pred_list.append(rpa_pred)

        model_results[model_name] = {
            "VA95_pred": np.array(va95_pred_list, dtype=float),
            "RPA_pred": np.array(rpa_pred_list, dtype=float),
        }

    # Step 5: save consolidated results
    save_dict = {
        "ego_id": np.array(all_ids),
        "VA95_true": np.array(all_va95_true, dtype=float),
        "RPA_true": np.array(all_rpa_true, dtype=float),
        "models": model_results,
        "meta": {
            "test_size": test_size,
            "requested_prefixes": list(prefixes),
            "selected_samples": len(valid_samples),
            "mode": mode,
        },
    }

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as handle:
        pickle.dump(save_dict, handle)
    logger.info("Saved metrics to {}", output_pkl)


__all__ = [
    "set_seed",
    "choose_fixed_model",
    "load_model",
    "compute_va95",
    "compute_rpa",
    "evaluate_vehicle",
    "iter_available_prefixes",
    "run_evaluation",
]
