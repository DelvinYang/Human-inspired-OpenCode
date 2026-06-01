"""Shared helpers for computing Final Displacement Error (FDE) from velocity/acc sequences."""

from __future__ import annotations

import math
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from scripts.common.model_utils import load_model


HISTORY = 12


def _extract_dt(data_loader, prefix: str) -> Optional[float]:
    try:
        result = data_loader(prefix)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load dt for prefix {}: {}", prefix, exc)
        return None
    if isinstance(result, tuple):
        if len(result) >= 2:
            return float(result[1])
        if len(result) == 1:
            return float(result[0])
        return None
    try:
        return float(result)
    except Exception:  # pragma: no cover - defensive
        logger.warning("Unable to parse dt from data_loader result: {}", result)
        return None


def _true_displacement(
    features: torch.Tensor,
    dt: float,
    min_frames: int,
) -> Optional[Tuple[float, float, int, np.ndarray]]:
    if dt is None or dt <= 0.0:
        return None
    if features.ndim != 2 or features.shape[1] != 12:
        return None
    num_frames = features.shape[0]
    if num_frames < max(HISTORY + 1, min_frames):
        return None

    dx = 0.0
    dy = 0.0
    path: List[Tuple[float, float]] = []
    for frame_idx in range(HISTORY, num_frames):
        v_lon = float(features[frame_idx, 0].item())
        v_lat = float(features[frame_idx, 2].item())
        if not math.isfinite(v_lon) or not math.isfinite(v_lat):
            return None
        dx += v_lon * dt
        dy += v_lat * dt
        path.append((dx, dy))
    horizon = num_frames - HISTORY
    return dx, dy, horizon, np.asarray(path, dtype=np.float32)


@torch.no_grad()
def _predict_displacement(
    features: torch.Tensor,
    *,
    model: torch.nn.Module,
    device: torch.device,
    dt: float,
    min_frames: int,
    acc_limit: float,
    mode: str,
) -> Optional[Tuple[float, float, np.ndarray]]:
    if dt is None or dt <= 0.0:
        return None
    if features.ndim != 2 or features.shape[1] != 12:
        return None
    num_frames = features.shape[0]
    if num_frames < max(HISTORY + 1, min_frames):
        return None

    hist_seq: deque[torch.Tensor] = deque((features[idx].clone() for idx in range(HISTORY)), maxlen=HISTORY)

    vx_prev = float(features[HISTORY - 1, 0].item())
    vy_prev = float(features[HISTORY - 1, 2].item())
    pred_dx = 0.0
    pred_dy = 0.0
    path: List[Tuple[float, float]] = []

    for frame_idx in range(HISTORY, num_frames):
        hist_tensor = torch.stack(list(hist_seq), dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)
        accel_pred = model(hist_tensor).squeeze(0).to("cpu")
        ax_pred = float(accel_pred[0].detach())
        ay_pred = float(accel_pred[1].detach())
        if not math.isfinite(ax_pred) or not math.isfinite(ay_pred):
            return None
        if acc_limit > 0 and (abs(ax_pred) > acc_limit or abs(ay_pred) > acc_limit):
            return None

        true_ax = float(features[frame_idx, 1].item())
        true_ay = float(features[frame_idx, 3].item())
        true_vlon = float(features[frame_idx, 0].item())
        true_vlat = float(features[frame_idx, 2].item())

        vx_pred = vx_prev + ax_pred * dt
        vy_pred = vy_prev + ay_pred * dt

        pred_dx += vx_pred * dt
        pred_dy += vy_pred * dt
        path.append((pred_dx, pred_dy))

        next_feat = features[frame_idx].clone()
        if mode == "semi":
            next_feat[0] = vx_pred
            next_feat[1] = true_ax  # keep true acc channel
            next_feat[2] = vy_pred
            next_feat[3] = true_ay
        elif mode == "gt":
            # teacher forcing inputs; keep history as ground truth
            pass
        else:  # closed
            next_feat[0] = vx_pred
            next_feat[1] = ax_pred
            next_feat[2] = vy_pred
            next_feat[3] = ay_pred

        hist_seq.append(next_feat)
        if mode == "gt":
            vx_prev = true_vlon
            vy_prev = true_vlat
        else:
            vx_prev = vx_pred
            vy_prev = vy_pred

    return pred_dx, pred_dy, np.asarray(path, dtype=np.float32)


def run_fde_evaluation(
    *,
    prefixes: Sequence[str],
    trajectory_dir: Path,
    data_loader,
    model_paths: Sequence[Path],
    device: torch.device,
    min_frames: int,
    output_pkl: Path,
    percent_tag: str,
    test_size: int = 500,
    model_limit: int = 50,
    seed: Optional[int] = None,
    reuse_records_from: Optional[Path] = None,
    append: bool = False,
    acc_limit: float = 20.0,
    test_mode: bool = False,
    mode: str = "semi",
) -> List[Dict[str, object]]:
    """Evaluate Final Displacement Error for multiple models on a shared sample set."""

    trajectory_dir = Path(trajectory_dir)
    output_pkl = Path(output_pkl)

    if not model_paths:
        raise ValueError("No model paths provided for FDE evaluation.")
    if mode not in {"closed", "semi", "gt"}:
        raise ValueError("mode must be one of: closed, semi, gt")

    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)
    model_paths = list(model_paths)
    if model_limit and len(model_paths) > model_limit:
        keep_indices = rng.choice(len(model_paths), size=model_limit, replace=False)
        model_paths = [model_paths[idx] for idx in keep_indices]
        logger.info("Model cap enabled: selected {} models out of {}", len(model_paths), len(keep_indices))

    existing_payload: Dict[str, object] | None = None

    # Always merge with existing results to avoid overwriting other percents.
    if output_pkl.exists():
        try:
            with output_pkl.open("rb") as handle:
                existing_payload = pickle.load(handle)
            logger.info("Loaded existing results from {} for merging.", output_pkl)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load existing payload (will overwrite): {}", exc)

    # Step 1: build cache of features and true displacements
    data_cache: Dict[str, Dict[object, torch.Tensor]] = {}
    true_cache: Dict[str, Dict[object, Tuple[float, float, int, np.ndarray]]] = {}
    dt_cache: Dict[str, float] = {}
    valid_pairs: List[Tuple[str, object]] = []

    def _load_features(prefix: str, needed: set | None = None) -> Dict[object, Tuple[torch.Tensor, torch.Tensor]]:
        traj_path = trajectory_dir / f"trajectory_samples_{prefix}.pkl"
        with traj_path.open("rb") as handle:
            trajectory_data = pickle.load(handle)
        if needed is None:
            return trajectory_data
        return {ego_key: trajectory_data[ego_key] for ego_key in needed if ego_key in trajectory_data}

    scan_prefixes = list(prefixes)
    rng.shuffle(scan_prefixes)
    for prefix in tqdm(scan_prefixes, desc="Scan prefixes"):
        traj_path = trajectory_dir / f"trajectory_samples_{prefix}.pkl"
        if not traj_path.exists():
            logger.warning("Trajectory file missing for prefix {}: {}", prefix, traj_path)
            continue
        dt = _extract_dt(data_loader, prefix)
        if dt is None:
            continue
        dt_cache[prefix] = dt
        trajectory_data = _load_features(prefix)
        if not trajectory_data:
            continue
        ego_items = list(trajectory_data.items())
        rng.shuffle(ego_items)
        for ego_key, (features, _labels) in ego_items:
            features = features.to(dtype=torch.float32, device="cpu")
            true_disp = _true_displacement(features, dt, min_frames)
            if true_disp is None:
                continue
            data_cache.setdefault(prefix, {})[ego_key] = features
            true_cache.setdefault(prefix, {})[ego_key] = true_disp
            valid_pairs.append((prefix, ego_key))

    if not valid_pairs:
        logger.warning("No FDE results to save.")
        return []

    # Step 3: evaluate models (each model samples its own subset and only stores ADE/FDE)
    model_results: Dict[str, List[Dict[str, float]]] = {}
    best_record: Optional[Dict[str, float]] = None
    best_model: Optional[str] = None
    best_horizon: Optional[int] = None
    child_seeds = seed_seq.spawn(len(model_paths)) if model_paths else []

    for model_idx, model_path in enumerate(model_paths):
        model_name = model_path.stem
        logger.info("Evaluating model: {}", model_path)
        model = load_model(model_path, device)
        preds_for_model: List[Dict[str, float]] = []
        rng_model = np.random.default_rng(child_seeds[model_idx]) if child_seeds else np.random.default_rng()

        sample_size = 1 if test_mode else min(test_size, len(valid_pairs))
        selected_indices = rng_model.choice(len(valid_pairs), size=sample_size, replace=False)
        final_pairs = [valid_pairs[idx] for idx in selected_indices]

        for prefix, ego_key in tqdm(
            final_pairs,
            desc=f"Model {model_name}",
            total=len(final_pairs),
        ):
            features = data_cache[prefix][ego_key]
            dt = dt_cache[prefix]
            pred_disp = _predict_displacement(
                features,
                model=model,
                device=device,
                dt=dt,
                min_frames=min_frames,
                acc_limit=acc_limit,
                mode=mode,
            )
            true_dx, true_dy, _horizon, true_path = true_cache[prefix][ego_key]
            if pred_disp is None:
                preds_for_model.append(
                    {
                        "pred_dx": float("nan"),
                        "pred_dy": float("nan"),
                        "fde": float("nan"),
                        "ade": float("nan"),
                    }
                )
                continue
            pred_dx, pred_dy, pred_path = pred_disp
            fde_val = math.hypot(pred_dx - true_dx, pred_dy - true_dy)
            ade_val = float("nan")
            if isinstance(pred_path, np.ndarray) and isinstance(true_path, np.ndarray):
                if pred_path.shape == true_path.shape and pred_path.size > 0:
                    diffs = pred_path - true_path
                    ade_val = float(np.mean(np.hypot(diffs[:, 0], diffs[:, 1])))

            try:
                ego_id = int(ego_key)
            except (TypeError, ValueError):
                ego_id = ego_key
            preds_for_model.append(
                {
                    "prefix": prefix,
                    "ego_id": ego_id,
                    "ego_key": ego_key,
                    "fde": float(fde_val),
                    "ade": ade_val,
                }
            )
            if test_mode:
                if best_record is None or fde_val < best_record.get("fde", float("inf")):
                    best_record = {
                        "fde": float(fde_val),
                        "ade": ade_val,
                    }
                    best_model = model_name
                    best_horizon = int(_horizon)
        logger.info("Model {} sampled {} trajectories (available {}).", model_name, len(final_pairs), len(valid_pairs))
        model_results[model_name] = preds_for_model

    if test_mode and best_record is not None and best_model is not None:
        logger.info(
            "Test mode best {}: FDE={:.3f} | ADE={:.3f} | horizon={} frames",
            best_model,
            best_record["fde"],
            best_record["ade"] if math.isfinite(best_record["ade"]) else float("nan"),
            best_horizon if best_horizon is not None else "n/a",
        )

    # Step 4: assemble payload (only ADE/FDE per model/trajectory)
    payload: Dict[str, object] = existing_payload if existing_payload is not None else {}

    models_field = payload.setdefault("models", {})
    models_field.setdefault(percent_tag, {})
    models_field[percent_tag].update(model_results)

    meta = payload.setdefault("meta", {})
    meta["available_samples"] = len(valid_pairs)
    meta["model_paths"] = [str(p) for p in model_paths]
    meta["trajectory_dir"] = str(trajectory_dir)
    meta["min_frames"] = min_frames
    meta["test_size"] = test_size
    meta["percents"] = sorted(models_field.keys())
    meta["acc_limit"] = acc_limit
    meta["mode"] = mode
    meta["schema"] = "fde_v2"

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved FDE results to {}", output_pkl)
    logger.info("Stored per-model ADE/FDE for {} models.", len(model_results))
    return []


__all__ = ["run_fde_evaluation"]
