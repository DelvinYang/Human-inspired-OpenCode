"""Shared helpers for computing per-frame TTC sequences on trajectory datasets."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch  # Required so that pickle can deserialize torch.Tensor entries.
from loguru import logger
from tqdm import tqdm

from scripts.common.acc_utils import compute_speed, extract_position
from scripts.common.open_loop import find_vehicle_state_generic, predict_ego_trajectory
from scripts.common.model_utils import load_model


def _compute_frame_ttc_from_values(
    ego_pos: Optional[Tuple[float, float]],
    ego_speed: float,
    neighbours,
    *,
    distance_scale: float,
    speed_epsilon: float,
) -> Optional[float]:
    """Compute TTC for the closest neighbour given explicit ego position/speed."""

    if not neighbours or ego_pos is None:
        return None

    nearest_distance = math.inf
    rel_speed = None

    for sv_state in neighbours:
        if sv_state is None:
            continue
        sv_pos = extract_position(sv_state)
        if sv_pos is None:
            continue

        dist = math.hypot(ego_pos[0] - sv_pos[0], ego_pos[1] - sv_pos[1])
        dist *= distance_scale
        if dist < nearest_distance:
            sv_speed = compute_speed(sv_state)
            nearest_distance = dist
            rel_speed = abs(ego_speed - sv_speed)

    if not math.isfinite(nearest_distance) or rel_speed is None:
        return None
    if rel_speed <= speed_epsilon:
        return float("inf")

    return nearest_distance / rel_speed


def _compute_frame_ttc(
    ego_state,
    neighbours,
    *,
    distance_scale: float,
    speed_epsilon: float,
) -> Optional[float]:
    ego_pos = extract_position(ego_state)
    ego_speed = compute_speed(ego_state)
    return _compute_frame_ttc_from_values(
        ego_pos,
        ego_speed,
        neighbours,
        distance_scale=distance_scale,
        speed_epsilon=speed_epsilon,
    )


def _normalise_vehicle_id(raw_id) -> Optional[int]:
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        logger.warning("Skip ego with non-integer id: {}", raw_id)
        return None


def compute_vehicle_ttc(
    *,
    scenario,
    ego_id: int,
    num_frames: int,
    distance_scale: float,
    speed_epsilon: float,
) -> Optional[np.ndarray]:
    """Compute the TTC series for a single vehicle."""

    try:
        ego_vehicle = scenario.find_vehicle_by_id(ego_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to locate ego {}: {}", ego_id, exc)
        return None

    start_frame = getattr(ego_vehicle, "initial_frame", None)
    final_frame = getattr(ego_vehicle, "final_frame", None)
    if start_frame is None or final_frame is None:
        return None

    ttc_values: List[float] = []

    for frame_offset in range(num_frames):
        frame_num = start_frame + frame_offset
        if frame_num > final_frame:
            break

        ego_state = find_vehicle_state_generic(scenario, ego_id, frame_num)
        if ego_state is None:
            continue
        neighbours = scenario.find_svs_state(frame_num, ego_id)
        frame_ttc = _compute_frame_ttc(
            ego_state,
            neighbours,
            distance_scale=distance_scale,
            speed_epsilon=speed_epsilon,
        )
        if frame_ttc is not None:
            ttc_values.append(frame_ttc)

    if not ttc_values:
        return None
    return np.asarray(ttc_values, dtype=np.float32)


def _compute_true_frame_ttc(
    *,
    scenario,
    ego_id: int,
    frame_num: int,
    neighbours,
    distance_scale: float,
    speed_epsilon: float,
) -> Optional[float]:
    """Compute true TTC for a specific frame using the scenario state."""

    ego_state = find_vehicle_state_generic(scenario, ego_id, frame_num)
    if ego_state is None:
        return None

    return _compute_frame_ttc(
        ego_state,
        neighbours,
        distance_scale=distance_scale,
        speed_epsilon=speed_epsilon,
    )


def _summarise_ttc(values: np.ndarray) -> Dict[str, float]:
    finite_values = values[np.isfinite(values)]
    summary: Dict[str, float] = {}

    if finite_values.size == 0:
        summary["ttc_min"] = float("inf")
        summary["ttc_mean"] = float("inf")
        summary["ttc_median"] = float("inf")
        summary["ttc_p05"] = float("inf")
        summary["ttc_p95"] = float("inf")
    else:
        summary["ttc_min"] = float(np.min(finite_values))
        summary["ttc_mean"] = float(np.mean(finite_values))
        summary["ttc_median"] = float(np.median(finite_values))
        summary["ttc_p05"] = float(np.quantile(finite_values, 0.05))
        summary["ttc_p95"] = float(np.quantile(finite_values, 0.95))

    summary["ttc_finite_ratio"] = float(finite_values.size / max(1, values.size))
    return summary


def prepare_ttc_cache(
    *,
    prefixes: Sequence[str],
    trajectory_dir: Path,
    data_loader,
    min_frames: int,
) -> Dict[str, object]:
    """预加载轨迹特征、场景和 dt，避免多次重复读取。"""

    trajectory_dir = Path(trajectory_dir)
    data_cache: Dict[str, Dict[object, torch.Tensor]] = {}
    scenario_cache: Dict[str, object] = {}
    dt_cache: Dict[str, float] = {}
    valid_pairs: List[Tuple[str, object]] = []

    for prefix in prefixes:
        traj_path = trajectory_dir / f"trajectory_samples_{prefix}.pkl"
        if not traj_path.exists():
            logger.warning("Trajectory file missing for prefix {}: {}", prefix, traj_path)
            continue

        try:
            scenario, dt = data_loader(prefix)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load scenario for prefix {}: {}", prefix, exc)
            continue

        with traj_path.open("rb") as handle:
            trajectory_data: Dict[object, Tuple[torch.Tensor, torch.Tensor]] = pickle.load(handle)

        if not trajectory_data:
            logger.info("No trajectories stored for prefix {}", prefix)
            continue

        for ego_key, (features, _labels) in trajectory_data.items():
            if features.shape[0] < min_frames:
                continue
            vehicle_id = _normalise_vehicle_id(ego_key)
            if vehicle_id is None:
                continue
            data_cache.setdefault(prefix, {})[ego_key] = features
            scenario_cache[prefix] = scenario
            dt_cache[prefix] = dt
            valid_pairs.append((prefix, ego_key))

    return {
        "data_cache": data_cache,
        "scenario_cache": scenario_cache,
        "dt_cache": dt_cache,
        "valid_pairs": valid_pairs,
    }


def run_ttc_evaluation(
    *,
    prefixes: Sequence[str],
    trajectory_dir: Path,
    data_loader,
    model_paths: Sequence[Path],
    device: torch.device,
    min_frames: int,
    output_pkl: Path,
    percent_tag: str,
    reuse_records_from: Optional[Path] = None,
    append: bool = False,
    cache: Optional[Dict[str, object]] = None,
    test_size: int = 500,
    model_limit: int = 50,
    seed: Optional[int] = None,
    distance_scale: float = 1.0,
    speed_epsilon: float = 1e-3,
) -> List[Dict[str, object]]:
    """Iterate over trajectories and store TTC sequences for multiple models.

    percent_tag:       Label for the current percent (used to group model outputs).
    reuse_records_from:Reuse previously sampled records to avoid re-sampling and recomputing ground truth.
    append:            Append model outputs to an existing output_pkl instead of overwriting.
    """

    trajectory_dir = Path(trajectory_dir)
    output_pkl = Path(output_pkl)

    def _record_signature(rec_list: Sequence[Dict[str, object]]) -> List[Tuple[object, object, int]]:
        """Compare records without touching heavy numpy arrays."""

        signature: List[Tuple[object, object, int]] = []
        for rec in rec_list:
            signature.append(
                (
                    rec.get("prefix"),
                    rec.get("ego_key"),
                    int(rec.get("samples_used", 0)),
                )
            )
        return signature

    if not model_paths:
        raise ValueError("No model paths provided for TTC evaluation.")

    existing_payload: Dict[str, object] | None = None
    reference_records: List[Dict[str, object]] | None = None
    reference_source: Optional[str] = None

    if reuse_records_from:
        try:
            with Path(reuse_records_from).open("rb") as handle:
                reuse_payload = pickle.load(handle)
            reference_records = list(reuse_payload.get("records", []))
            reference_source = str(Path(reuse_records_from))
            logger.info("Reusing {} records from {}", len(reference_records), reference_source)
        except FileNotFoundError:
            logger.warning("reuse_records_from not found: {}", reuse_records_from)

    if append and output_pkl.exists():
        with output_pkl.open("rb") as handle:
            existing_payload = pickle.load(handle)
        if reference_records is None:
            reference_records = list(existing_payload.get("records", []))
            reference_source = str(output_pkl)
            logger.info("Appending using {} existing records from {}", len(reference_records), output_pkl)
    rng = np.random.RandomState(seed if seed is not None else 42)
    model_paths = list(model_paths)
    if model_limit and len(model_paths) > model_limit:
        keep_indices = rng.choice(len(model_paths), size=model_limit, replace=False)
        model_paths = [model_paths[idx] for idx in keep_indices]
        logger.info("Model cap enabled: selected {} models out of {}", len(model_paths), len(keep_indices))

    if cache is None:
        cache = prepare_ttc_cache(
            prefixes=prefixes,
            trajectory_dir=trajectory_dir,
            data_loader=data_loader,
            min_frames=min_frames,
        )

    data_cache: Dict[str, Dict[object, torch.Tensor]] = cache.get("data_cache", {})  # type: ignore[assignment]
    scenario_cache: Dict[str, object] = cache.get("scenario_cache", {})  # type: ignore[assignment]
    dt_cache: Dict[str, float] = cache.get("dt_cache", {})  # type: ignore[assignment]
    valid_pairs: List[Tuple[str, object]] = cache.get("valid_pairs", [])  # type: ignore[assignment]

    if not valid_pairs:
        logger.warning("No TTC results to save.")
        return []

    # 如果已有参考记录，则基于参考记录筛选同样的样本；否则按 test_size 随机抽样。
    records: List[Dict[str, object]] = []
    final_pairs: List[Tuple[str, object]] = []
    if reference_records:
        logger.info("Using reference records to keep sample set fixed.")
        for rec in reference_records:
            prefix = rec.get("prefix")
            ego_key = rec.get("ego_key")
            if prefix not in data_cache or ego_key not in data_cache[prefix]:
                logger.warning("Skip missing cached features for pair ({}, {})", prefix, ego_key)
                continue
            records.append(rec)
            final_pairs.append((prefix, ego_key))
        logger.info("Kept {} / {} reference records", len(records), len(reference_records))
    else:
        if len(valid_pairs) > test_size:
            selected_indices = rng.choice(len(valid_pairs), size=test_size, replace=False)
            selected_pairs = [valid_pairs[idx] for idx in selected_indices]
        else:
            selected_pairs = valid_pairs

        logger.info("Selected {} trajectories (requested {}).", len(selected_pairs), test_size)

        # records with true TTC (compute now to avoid heavy upfront scanning)
        for prefix, ego_key in selected_pairs:
            features = data_cache[prefix][ego_key]
            try:
                ego_id = int(ego_key)
            except (TypeError, ValueError):
                ego_id = ego_key
            try:
                true_arr = compute_vehicle_ttc(
                    scenario=scenario_cache[prefix],
                    ego_id=int(ego_id) if isinstance(ego_id, int) else ego_id,
                    num_frames=int(features.shape[0]),
                    distance_scale=distance_scale,
                    speed_epsilon=speed_epsilon,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("TTC evaluation failed for ego {} in {}: {}", ego_key, prefix, exc)
                continue
            if true_arr is None or len(true_arr) == 0:
                continue
            true_arr = np.asarray(true_arr, dtype=np.float32)
            records.append(
                {
                    "prefix": prefix,
                    "ego_id": ego_id,
                    "ego_key": ego_key,
                    "ttc_true": true_arr,
                    "samples_used": int(true_arr.size),
                    "dt": float(dt_cache.get(prefix, 0.0)),
                }
            )
            final_pairs.append((prefix, ego_key))

    # evaluate models
    model_results: Dict[str, List[np.ndarray]] = {}
    for model_path in model_paths:
        model_name = model_path.stem
        logger.info("Evaluating model: {}", model_path)
        model = load_model(model_path, device)
        preds_for_model: List[np.ndarray] = []

        for prefix, ego_key in tqdm(
            final_pairs,
            desc=f"Model {model_name}",
            total=len(final_pairs),
        ):
            features = data_cache[prefix][ego_key]
            try:
                ego_id = int(ego_key)
            except (TypeError, ValueError):
                ego_id = ego_key

            trajectory = predict_ego_trajectory(
                features,
                scenario=scenario_cache[prefix],
                ego_id=ego_id,
                model=model,
                device=device,
                dt=dt_cache[prefix],
                min_frames=min_frames,
            )

            paired_pred_values: List[float] = []

            if trajectory is not None:
                for idx, frame_num in enumerate(trajectory.frames):
                    frame_int = int(frame_num)
                    neighbours = scenario_cache[prefix].find_svs_state(frame_int, ego_id)
                    ego_pos = tuple(map(float, trajectory.positions[idx]))
                    vx, vy = trajectory.velocities[idx]
                    ego_speed = float(math.hypot(vx, vy))

                    frame_ttc_pred = _compute_frame_ttc_from_values(
                        ego_pos,
                        ego_speed,
                        neighbours,
                        distance_scale=distance_scale,
                        speed_epsilon=speed_epsilon,
                    )
                    if frame_ttc_pred is None:
                        continue
                    paired_pred_values.append(frame_ttc_pred)

            pred_array = (
                np.asarray(paired_pred_values, dtype=np.float32)
                if paired_pred_values
                else np.asarray([], dtype=np.float32)
            )
            model_results.setdefault(model_name, []).append(pred_array)
    payload: Dict[str, object]
    if existing_payload is not None and append and reference_records:
        payload = existing_payload
        payload.setdefault("models", {})
        logger.info("Appending results to existing payload with {} records", len(payload.get("records", [])))
    else:
        payload = {
            "records": records,
            "models": {},
            "meta": {
                "distance_scale": distance_scale,
                "speed_epsilon": speed_epsilon,
                "min_frames": min_frames,
                "trajectory_dir": str(trajectory_dir),
                "num_prefixes": len(prefixes),
                "selected_samples": len(records),
                "test_size": test_size,
                "schema": "ttc_sequence_v2",
            },
        }

    # 确保 records 一致
    payload_records = payload.get("records")
    if records and isinstance(payload_records, list):
        if _record_signature(payload_records) != _record_signature(records):
            logger.warning(
                "Payload records differ from computed/reference records; using reference to keep consistency."
            )
            payload["records"] = records
    elif records and payload_records is None:
        payload["records"] = records

    models_field = payload.setdefault("models", {})
    models_field.setdefault(percent_tag, {})
    models_field[percent_tag].update(model_results)

    meta = payload.setdefault("meta", {})
    meta["selected_samples"] = len(payload.get("records", []))
    meta["model_paths"] = [str(p) for p in model_paths]
    meta["distance_scale"] = distance_scale
    meta["speed_epsilon"] = speed_epsilon
    meta["min_frames"] = min_frames
    meta["trajectory_dir"] = str(trajectory_dir)
    meta["num_prefixes"] = len(prefixes)
    meta["test_size"] = test_size
    meta["percents"] = sorted(models_field.keys())
    if reference_source:
        meta["reference_records"] = reference_source

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved TTC results to {}", output_pkl)
    logger.info("Stored {} trajectory records.", len(payload.get("records", [])))
    return records


__all__ = ["compute_vehicle_ttc", "run_ttc_evaluation", "prepare_ttc_cache"]
