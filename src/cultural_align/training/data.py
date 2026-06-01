from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ARRAY_KEYS = ("state", "action", "next_state", "next_action")


@dataclass
class NormStats:
    state_mean: list[float]
    state_std: list[float]
    action_mean: list[float]
    action_std: list[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_unit_hash(value: str) -> float:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) / float(2**64)


def fraction_mask(dataset: str, scene: np.ndarray, track: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 0.999999:
        return np.ones(track.shape[0], dtype=bool)
    if fraction <= 0.0:
        return np.zeros(track.shape[0], dtype=bool)
    keep = np.zeros(track.shape[0], dtype=bool)
    for i, (scene_id, track_id) in enumerate(zip(scene.astype(str), track.astype(str))):
        keep[i] = stable_unit_hash(f"{dataset}/{scene_id}/{track_id}") < fraction
    return keep


def _track_keys(z: np.lib.npyio.NpzFile) -> list[str]:
    if "meta_scene" in z.files and "meta_track_id" in z.files:
        return [f"{scene}/{track}" for scene, track in zip(z["meta_scene"].astype(str), z["meta_track_id"].astype(str))]
    return [str(i) for i in range(len(z["action"]))]


def load_dataset_split(
    root: Path,
    dataset: str,
    split: str,
    max_samples: int = 0,
    random_seed: int | None = None,
    fraction: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paths = sorted((Path(root) / split / dataset).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"missing split={split} dataset={dataset} under {root}")

    chunks: dict[str, list[np.ndarray]] = {key: [] for key in ARRAY_KEYS}
    rows_seen = rows_kept = loaded = 0
    tracks_seen: set[str] = set()
    tracks_kept: set[str] = set()
    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    for path in paths:
        with np.load(path) as z:
            n = int(len(z["action"]))
            rows_seen += n
            keys = _track_keys(z)
            tracks_seen.update(keys)
            if "meta_scene" in z.files and "meta_track_id" in z.files:
                mask = fraction_mask(dataset, z["meta_scene"], z["meta_track_id"], fraction)
            else:
                mask = np.ones(n, dtype=bool) if fraction >= 0.999999 else np.asarray(
                    [stable_unit_hash(f"{dataset}/{path.name}/{i}") < fraction for i in range(n)],
                    dtype=bool,
                )
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            if max_samples > 0 and rng is None:
                remaining = max_samples - loaded
                if remaining <= 0:
                    break
                idx = idx[:remaining]
            for key in ARRAY_KEYS:
                chunks[key].append(z[key][idx].astype(np.float32))
            loaded += int(idx.size)
            rows_kept += int(idx.size)
            tracks_kept.update(keys[int(i)] for i in idx.tolist())

    if not chunks["action"]:
        raise ValueError(f"split={split} dataset={dataset} fraction={fraction} produced zero samples")

    arrays = {key: np.concatenate(parts, axis=0) for key, parts in chunks.items()}
    if max_samples > 0 and rng is not None and len(arrays["action"]) > max_samples:
        idx = np.sort(rng.choice(len(arrays["action"]), size=max_samples, replace=False))
        arrays = {key: value[idx] for key, value in arrays.items()}
        rows_kept = int(max_samples)

    return arrays, {
        "dataset": dataset,
        "split": split,
        "files": [str(path) for path in paths],
        "rows_seen": int(rows_seen),
        "rows_kept": int(rows_kept),
        "tracks_seen": int(len(tracks_seen)),
        "tracks_kept": int(len(tracks_kept)),
        "fraction": float(fraction),
    }


def concat_arrays(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("cannot concatenate an empty dataset list")
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in ARRAY_KEYS}


def load_multi_split(
    root: Path,
    datasets: list[str],
    split: str,
    max_samples_per_dataset: int = 0,
    random_seed: int | None = None,
    fraction: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = []
    meta: dict[str, Any] = {}
    for idx, dataset in enumerate(datasets):
        seed = None if random_seed is None else int(random_seed + idx * 1009)
        part, part_meta = load_dataset_split(
            root,
            dataset,
            split,
            max_samples=max_samples_per_dataset,
            random_seed=seed,
            fraction=fraction,
        )
        arrays.append(part)
        meta[dataset] = part_meta
    return concat_arrays(arrays), meta


def compute_stats(train: dict[str, np.ndarray]) -> NormStats:
    state = train["state"].reshape(-1, train["state"].shape[-1])
    action = train["action"]
    state_mean = state.mean(axis=0)
    state_std = state.std(axis=0)
    state_std[state_std < 1e-6] = 1.0
    action_mean = action.mean(axis=0)
    action_std = action.std(axis=0)
    action_std[action_std < 1e-6] = 1.0
    return NormStats(
        state_mean.astype(np.float32).tolist(),
        state_std.astype(np.float32).tolist(),
        action_mean.astype(np.float32).tolist(),
        action_std.astype(np.float32).tolist(),
    )


class TransitionDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], stats: NormStats):
        state_mean = np.asarray(stats.state_mean, dtype=np.float32)
        state_std = np.asarray(stats.state_std, dtype=np.float32)
        action_mean = np.asarray(stats.action_mean, dtype=np.float32)
        action_std = np.asarray(stats.action_std, dtype=np.float32)
        self.state_raw = arrays["state"].astype(np.float32)
        self.action_raw = arrays["action"].astype(np.float32)
        self.next_state_raw = arrays["next_state"].astype(np.float32)
        self.next_action_raw = arrays["next_action"].astype(np.float32)
        self.state = (self.state_raw - state_mean) / state_std
        self.next_state = (self.next_state_raw - state_mean) / state_std
        self.action = (self.action_raw - action_mean) / action_std
        self.next_action = (self.next_action_raw - action_mean) / action_std

    def __len__(self) -> int:
        return int(self.action.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.state[idx]),
            torch.from_numpy(self.action[idx]),
            torch.from_numpy(self.next_state[idx]),
            torch.from_numpy(self.next_action[idx]),
            torch.from_numpy(self.action_raw[idx]),
            torch.from_numpy(self.state_raw[idx]),
        )


def make_transition_loader(
    arrays: dict[str, np.ndarray],
    stats: NormStats,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        TransitionDataset(arrays, stats),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
