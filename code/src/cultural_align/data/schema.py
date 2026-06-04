from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


SPLITS = ("train", "val", "test")
FULL_LIMIT = 10**18

EIGHT_ID_COLUMNS = [
    "precedingId",
    "followingId",
    "leftPrecedingId",
    "leftAlongsideId",
    "leftFollowingId",
    "rightPrecedingId",
    "rightAlongsideId",
    "rightFollowingId",
]

DIR_NAMES = (
    "preceding",
    "following",
    "leftPreceding",
    "leftAlongside",
    "leftFollowing",
    "rightPreceding",
    "rightAlongside",
    "rightFollowing",
)

FEATURE_NAMES = [
    "xVelocity",
    "xAcceleration",
    "yVelocity",
    "yAcceleration",
    "dist_preceding",
    "dist_following",
    "dist_leftPreceding",
    "dist_leftAlongside",
    "dist_leftFollowing",
    "dist_rightPreceding",
    "dist_rightAlongside",
    "dist_rightFollowing",
]

TARGET_NAMES = ["xAcceleration", "yAcceleration"]


def stable_unit_hash(value: str) -> float:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) / float(2**64)


def split_for_track(
    dataset: str,
    scene: str,
    track_id: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> str:
    value = stable_unit_hash(f"{dataset}/{scene}/{track_id}")
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def sample_limit_value(samples: int) -> int:
    return FULL_LIMIT if samples <= 0 else int(samples)


def safe_unit_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def finite_rows(df, columns: Iterable[str], skip: set[str] | None = None):
    skip = skip or set()
    mask = np.ones(len(df), dtype=bool)
    for col in columns:
        if col in skip:
            continue
        mask &= np.isfinite(df[col].to_numpy(dtype=np.float64))
    return df.loc[mask].copy()


def physical_pair(
    action: np.ndarray,
    next_action: np.ndarray,
    abs_ax_limit: float,
    abs_ay_limit: float,
    delta_ax_limit: float,
    delta_ay_limit: float,
) -> bool:
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(next_action)):
        return False
    if abs(float(action[0])) > abs_ax_limit or abs(float(action[1])) > abs_ay_limit:
        return False
    if abs(float(next_action[0])) > abs_ax_limit or abs(float(next_action[1])) > abs_ay_limit:
        return False
    delta = next_action - action
    return abs(float(delta[0])) <= delta_ax_limit and abs(float(delta[1])) <= delta_ay_limit


def savgol_window(n: int, preferred: int, poly: int = 3) -> int | None:
    if n <= poly:
        return None
    win = min(preferred, n if n % 2 == 1 else n - 1)
    return win if win > poly else None


def empty_bucket() -> dict[str, list[Any]]:
    return {
        "state": [],
        "action": [],
        "next_state": [],
        "next_action": [],
        "meta_dataset": [],
        "meta_scene": [],
        "meta_track_id": [],
        "meta_frame_k": [],
        "meta_frame_next": [],
    }


def pack_bucket(bucket: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    return {
        "state": np.stack(bucket["state"]).astype(np.float32),
        "action": np.stack(bucket["action"]).astype(np.float32),
        "next_state": np.stack(bucket["next_state"]).astype(np.float32),
        "next_action": np.stack(bucket["next_action"]).astype(np.float32),
        "meta_dataset": np.asarray(bucket["meta_dataset"], dtype="U32"),
        "meta_scene": np.asarray(bucket["meta_scene"], dtype="U160"),
        "meta_track_id": np.asarray(bucket["meta_track_id"], dtype="U160"),
        "meta_frame_k": np.asarray(bucket["meta_frame_k"], dtype=np.int64),
        "meta_frame_next": np.asarray(bucket["meta_frame_next"], dtype=np.int64),
        "feature_names": np.asarray(FEATURE_NAMES, dtype="U32"),
        "target_names": np.asarray(TARGET_NAMES, dtype="U32"),
    }


class DatasetWriter:
    def __init__(self, out_dir: Path, dataset: str, shard_size: int, file_prefix: str | None = None):
        self.out_dir = Path(out_dir)
        self.dataset = dataset
        self.file_prefix = safe_unit_id(file_prefix or dataset)
        self.shard_size = int(shard_size)
        self.buckets = {split: empty_bucket() for split in SPLITS}
        self.shard_index = {split: 0 for split in SPLITS}
        self.counts = {split: 0 for split in SPLITS}
        self.files: dict[str, list[str]] = {split: [] for split in SPLITS}

    def append(self, split: str, sample: dict[str, Any]) -> None:
        bucket = self.buckets[split]
        for key, value in sample.items():
            bucket[key].append(value)
        self.counts[split] += 1
        if self.shard_size > 0 and len(bucket["action"]) >= self.shard_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        bucket = self.buckets[split]
        if not bucket["action"]:
            return
        path = self.out_dir / split / self.dataset / f"{self.file_prefix}_psiphi_xy_{self.shard_index[split]:05d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **pack_bucket(bucket))
        self.files[split].append(str(path))
        self.shard_index[split] += 1
        self.buckets[split] = empty_bucket()

    def close(self) -> None:
        for split in SPLITS:
            self.flush(split)


def transition_sample(
    dataset: str,
    scene: str,
    track_id: str,
    frame_k: int,
    frame_next: int,
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    next_action: np.ndarray,
) -> dict[str, Any]:
    return {
        "state": state,
        "action": action,
        "next_state": next_state,
        "next_action": next_action,
        "meta_dataset": dataset,
        "meta_scene": scene,
        "meta_track_id": str(track_id),
        "meta_frame_k": int(frame_k),
        "meta_frame_next": int(frame_next),
    }


def append_transition(
    writer: DatasetWriter,
    dataset: str,
    scene: str,
    track_id: str,
    frame_k: int,
    frame_next: int,
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    next_action: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    split = split_for_track(dataset, scene, str(track_id), train_ratio=train_ratio, val_ratio=val_ratio)
    writer.append(split, transition_sample(dataset, scene, str(track_id), frame_k, frame_next, state, action, next_state, next_action))


def iter_track_windows(records: list[Any], frame_getter: Callable[[Any], int], t_hist: int, stride: int):
    starts = np.arange(0, len(records) - t_hist - 1, max(1, stride), dtype=np.int64)
    for start in starts:
        start = int(start)
        state_rows = records[start : start + t_hist]
        next_state_rows = records[start + 1 : start + t_hist + 1]
        action_row = records[start + t_hist]
        next_action_row = records[start + t_hist + 1]
        frames = np.asarray([frame_getter(r) for r in [*state_rows, action_row, next_action_row]], dtype=np.int64)
        yield state_rows, next_state_rows, action_row, next_action_row, frames


def load_assigned_track_ids(assignment_dir: Path, dataset: str, scene_id: str, require_path_id: bool = False) -> set[str]:
    path = Path(assignment_dir) / dataset / f"{scene_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing assignment file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = set()
    for item in payload.get("assignments", []):
        if item.get("status") != "assigned":
            continue
        if require_path_id and not item.get("path_id"):
            continue
        out.add(str(item["track_id"]))
    return out


def maybe_assignment_filter(assignment_dir: Path | None, dataset: str, scene_id: str, require_path_id: bool = False) -> set[str] | None:
    if assignment_dir is None:
        return None
    return load_assigned_track_ids(Path(assignment_dir), dataset, scene_id, require_path_id=require_path_id)


def base_summary(
    dataset: str,
    writer: DatasetWriter,
    started: float,
    extra: dict[str, Any],
    schema: str = "trajvista_xy_psiphi_v1",
) -> dict[str, Any]:
    writer.close()
    samples = int(sum(writer.counts.values()))
    return {
        "dataset": dataset,
        "samples": samples,
        "split_counts": {k: int(v) for k, v in writer.counts.items()},
        "files": writer.files,
        "elapsed_sec": time.time() - started,
        "schema": schema,
        "state": ["N", 12, len(FEATURE_NAMES)],
        "action": ["N", 2],
        "next_state": ["N", 12, len(FEATURE_NAMES)],
        "next_action": ["N", 2],
        "meta": ["meta_dataset", "meta_scene", "meta_track_id", "meta_frame_k", "meta_frame_next"],
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        **extra,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def ensure_consecutive(frames: np.ndarray) -> bool:
    return bool(np.all(np.diff(frames) == 1))


def heading_sector_distances(
    rel_xy: np.ndarray,
    dist: np.ndarray,
    theta: float,
    half_width: float,
    half_length: float,
) -> list[float]:
    bucket = {name: math.inf for name in DIR_NAMES}
    c = math.cos(theta)
    s = math.sin(theta)
    longitudinal = c * rel_xy[:, 0] + s * rel_xy[:, 1]
    lateral_left = -s * rel_xy[:, 0] + c * rel_xy[:, 1]
    for long_v, lat_v, d in zip(longitudinal, lateral_left, dist):
        if abs(float(lat_v)) <= half_width:
            if long_v > half_length:
                name = "preceding"
            elif long_v < -half_length:
                name = "following"
            else:
                continue
        elif abs(float(long_v)) <= half_length:
            name = "leftAlongside" if lat_v > 0 else "rightAlongside"
        elif long_v > 0:
            name = "leftPreceding" if lat_v > 0 else "rightPreceding"
        else:
            name = "leftFollowing" if lat_v > 0 else "rightFollowing"
        if float(d) < bucket[name]:
            bucket[name] = float(d)
    return [bucket[name] if math.isfinite(bucket[name]) else -1.0 for name in DIR_NAMES]
