"""Shared helpers for computing per-trajectory acceleration sequences."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch  # Needed so deserialising pickles with tensors works.
from loguru import logger
from tqdm import tqdm

from scripts.common.acc_utils import extract_longitudinal_acc
from scripts.common.open_loop import predict_longitudinal_acc_sequences, find_vehicle_state_generic
from scripts.common.model_utils import load_model


def run_acc_evaluation(
    *,
    prefixes: Sequence[str],
    trajectory_dir: Path,
    data_loader,
    model_paths: Sequence[Path],
    device: torch.device,
    min_frames: int,
    output_pkl: Path,
    test_size: int = 500,
    model_limit: int = 50,
    seed: Optional[int] = None,
    acc_limit: float = 10.0,
    model_workers: int = 1,
) -> List[Dict[str, object]]:
    """Compute acceleration sequences for a fixed subset of trajectories across multiple models."""

    trajectory_dir = Path(trajectory_dir)
    output_pkl = Path(output_pkl)

    if not model_paths:
        raise ValueError("No model paths provided for acceleration evaluation.")

    rng = np.random.RandomState(seed if seed is not None else 42)
    model_paths = list(model_paths)
    if model_limit and len(model_paths) > model_limit:
        keep_indices = rng.choice(len(model_paths), size=model_limit, replace=False)
        model_paths = [model_paths[idx] for idx in keep_indices]
        logger.info("Model cap enabled: selected {} models out of {}", len(model_paths), len(keep_indices))


    # Prepare device pool
    # Step 1: enumerate and filter valid trajectories (length + true acc limit)
    data_cache: Dict[str, Dict[object, torch.Tensor]] = {}
    true_cache: Dict[str, Dict[object, np.ndarray]] = {}
    scenario_cache: Dict[str, object] = {}
    dt_cache: Dict[str, float] = {}
    valid_pairs: List[Tuple[str, object]] = []

    def _true_acc_sequence(features: torch.Tensor, scenario, ego_id: int) -> Optional[np.ndarray]:
        history = 12
        if features.ndim != 2 or features.shape[1] != 12:
            return None
        if features.shape[0] < max(history + 1, min_frames):
            return None
        try:
            ego_vehicle = scenario.find_vehicle_by_id(ego_id)
        except Exception:
            return None
        start_frame = getattr(ego_vehicle, "initial_frame", None)
        if start_frame is None:
            return None
        acc_list: List[float] = []
        for frame_idx in range(history, features.shape[0]):
            frame_num = start_frame + frame_idx
            ego_state = find_vehicle_state_generic(scenario, ego_id, frame_num)
            true_acc = None if ego_state is None else extract_longitudinal_acc(ego_state)
            if true_acc is None:
                return None
            acc_list.append(float(true_acc))
        if not acc_list:
            return None
        acc_arr = np.asarray(acc_list, dtype=np.float32)
        if np.any(np.abs(acc_arr) > acc_limit):
            return None
        return acc_arr

    for prefix in prefixes:
        traj_path = trajectory_dir / f"trajectory_samples_{prefix}.pkl"
        if not traj_path.exists():
            logger.warning("Missing trajectory file for prefix {}: {}", prefix, traj_path)
            continue
        try:
            scenario, dt = data_loader(prefix)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load scenario for prefix {}: {}", prefix, exc)
            continue
        with traj_path.open("rb") as handle:
            trajectory_data = pickle.load(handle)
        if not trajectory_data:
            continue
        for ego_id, (features, _labels) in trajectory_data.items():
            true_acc = _true_acc_sequence(features, scenario, ego_id)
            if true_acc is None:
                continue
            data_cache.setdefault(prefix, {})[ego_id] = features
            true_cache.setdefault(prefix, {})[ego_id] = true_acc
            scenario_cache[prefix] = scenario
            dt_cache[prefix] = dt
            valid_pairs.append((prefix, ego_id))

    if not valid_pairs:
        logger.warning("No trajectories found for acceleration evaluation.")
        return []

    if len(valid_pairs) > test_size:
        selected_indices = rng.choice(len(valid_pairs), size=test_size, replace=False)
        selected_pairs = [valid_pairs[idx] for idx in selected_indices]
    else:
        selected_pairs = valid_pairs

    logger.info("Selected {} trajectories (requested {}).", len(selected_pairs), test_size)

    # Step 2: build records with true sequences
    sample_records: List[Dict[str, object]] = []
    for prefix, ego_key in selected_pairs:
        acc_true = true_cache[prefix][ego_key]
        try:
            ego_id = int(ego_key)
        except (TypeError, ValueError):
            ego_id = ego_key
        sample_records.append(
            {
                "prefix": prefix,
                "ego_id": ego_id,
                "ego_key": ego_key,
                "acc_true": acc_true.astype(np.float32, copy=False),
                "dt": float(dt_cache.get(prefix, 0.0)),
                "samples_used": int(acc_true.shape[0]),
            }
        )

    # Step 3: evaluate all models on the same set (parallel threads)
    model_results: Dict[str, List[np.ndarray]] = {}

    def _eval_single_model(model_path: Path) -> Tuple[str, List[np.ndarray]]:
        model_name = model_path.stem
        logger.info("Evaluating model: {}", model_path)
        model = load_model(model_path, device)
        preds_for_model: List[np.ndarray] = []

        for prefix, ego_key in tqdm(
            selected_pairs,
            desc=f"Model {model_name}",
            total=len(selected_pairs),
        ):
            features = data_cache[prefix][ego_key]
            try:
                ego_id = int(ego_key)
            except (TypeError, ValueError):
                ego_id = ego_key

            seq = predict_longitudinal_acc_sequences(
                features,
                scenario=scenario_cache[prefix],
                ego_id=ego_id,
                model=model,
                device=device,
                min_frames=min_frames,
            )
            if seq is None:
                preds_for_model.append(np.empty((0,), dtype=np.float32))
            else:
                pred_acc, _true_acc = seq
                preds_for_model.append(pred_acc.astype(np.float32, copy=False))

        return model_name, preds_for_model

    for mp in model_paths:
        model_name, preds_for_model = _eval_single_model(Path(mp))
        model_results[model_name] = preds_for_model

    payload = {
        "records": sample_records,
        "models": model_results,
        "meta": {
            "min_frames": min_frames,
            "trajectory_dir": str(trajectory_dir),
            "num_prefixes": len(prefixes),
            "selected_samples": len(sample_records),
            "test_size": test_size,
            "model_paths": [str(p) for p in model_paths],
            "schema": "acc_sequence_v2",
            "acc_limit": acc_limit,
        },
    }

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved acceleration results to {}", output_pkl)
    logger.info("Stored {} trajectory summaries.", len(sample_records))
    return sample_records


__all__ = ["run_acc_evaluation"]
