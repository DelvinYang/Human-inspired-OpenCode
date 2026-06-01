from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .schema import (
    DatasetWriter,
    FEATURE_NAMES,
    TARGET_NAMES,
    stable_unit_hash,
    transition_sample,
)


COUNTRY_DATASETS = {
    "CHN": "INTERACTION_CHN",
    "DEU": "INTERACTION_DEU",
    "USA": "INTERACTION_USA",
}
DIR_NAMES = [
    "preceding",
    "following",
    "leftPreceding",
    "leftAlongside",
    "leftFollowing",
    "rightPreceding",
    "rightAlongside",
    "rightFollowing",
]
NUMERIC_COLUMNS = ["case_id", "track_id", "frame_id", "timestamp_ms", "x", "y", "vx", "vy", "psi_rad", "length", "width"]


def resolve_interaction_root(path: Path) -> Path:
    if list(path.glob("DR_*_*.csv")):
        return path
    nested = path / "recorded_trackfiles"
    if list(nested.glob("DR_*_*.csv")):
        return nested
    raise FileNotFoundError(f"no INTERACTION DR_* csv files under {path}")


def parse_file(path: Path) -> tuple[str, str, str]:
    parts = path.stem.split("_")
    if len(parts) < 5 or parts[0] != "DR":
        raise ValueError(f"unexpected INTERACTION filename: {path.name}")
    country = parts[1]
    split = parts[-1]
    scene = "_".join(parts[:-1])
    if country not in COUNTRY_DATASETS:
        raise ValueError(f"unsupported INTERACTION country in {path.name}")
    if split not in {"train", "val"}:
        raise ValueError(f"unsupported INTERACTION split in {path.name}")
    return country, scene, split


def split_for_track(country: str, scene: str, case_id: int, track_id: int, file_split: str, val_test_ratio: float) -> str:
    if file_split == "train":
        return "train"
    value = stable_unit_hash(f"INTERACTION/{country}/{scene}/{case_id}/{track_id}")
    return "test" if value < val_test_ratio else "val"


def savgol_window(n: int, preferred: int, poly: int) -> int | None:
    if n <= poly + 1:
        return None
    win = min(preferred, n if n % 2 == 1 else n - 1)
    return win if win > poly else None


def velocity_acceleration(vx: np.ndarray, vy: np.ndarray, timestamps_ms: np.ndarray, preferred_window: int, poly: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(vx))
    if n < 3:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    t = timestamps_ms.astype(np.float64) / 1000.0
    diffs = np.diff(t)
    valid_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    dt = float(np.median(valid_diffs)) if len(valid_diffs) else 0.1
    win = savgol_window(n, preferred_window, poly)
    if win is not None:
        ax = savgol_filter(vx.astype(np.float64), window_length=win, polyorder=poly, deriv=1, delta=dt, mode="interp")
        ay = savgol_filter(vy.astype(np.float64), window_length=win, polyorder=poly, deriv=1, delta=dt, mode="interp")
    elif len(valid_diffs) == n - 1:
        ax = np.gradient(vx.astype(np.float64), t)
        ay = np.gradient(vy.astype(np.float64), t)
    else:
        ax = np.gradient(vx.astype(np.float64), dt)
        ay = np.gradient(vy.astype(np.float64), dt)
    return ax.astype(np.float64), ay.astype(np.float64)


def load_car_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[*NUMERIC_COLUMNS, "agent_type"])
    df = df[df["agent_type"].astype(str).str.lower().eq("car")].copy()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = np.ones(len(df), dtype=bool)
    for col in NUMERIC_COLUMNS:
        mask &= np.isfinite(df[col].to_numpy(dtype=np.float64))
    df = df.loc[mask].copy()
    df["case_i"] = np.rint(df["case_id"].to_numpy(dtype=np.float64)).astype(np.int64)
    df["track_i"] = np.rint(df["track_id"].to_numpy(dtype=np.float64)).astype(np.int64)
    df["frame_i"] = np.rint(df["frame_id"].to_numpy(dtype=np.float64)).astype(np.int64)
    return df


def build_frame_lookup(df: pd.DataFrame) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    lookup: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for (case_id, frame_id), rows in df.groupby(["case_i", "frame_i"], sort=False):
        lookup[(int(case_id), int(frame_id))] = {
            "track_id": rows["track_i"].to_numpy(dtype=np.int64),
            "xy": rows[["x", "y"]].to_numpy(dtype=np.float64),
        }
    return lookup


def direction_distances(
    case_id: int,
    frame_id: int,
    track_id: int,
    x: float,
    y: float,
    psi_rad: float,
    length: float,
    width: float,
    frame_lookup: dict[tuple[int, int], dict[str, np.ndarray]],
    max_radius: float,
) -> list[float]:
    bucket = {name: math.inf for name in DIR_NAMES}
    same = frame_lookup.get((int(case_id), int(frame_id)))
    if not same:
        return [-1.0] * len(DIR_NAMES)
    ego_xy = np.asarray([x, y], dtype=np.float64)
    keep = same["track_id"] != int(track_id)
    if not np.any(keep):
        return [-1.0] * len(DIR_NAMES)
    rel = same["xy"][keep] - ego_xy[None, :]
    dist = np.linalg.norm(rel, axis=1)
    keep = (dist > 1e-6) & (dist <= max_radius)
    if not np.any(keep):
        return [-1.0] * len(DIR_NAMES)
    rel = rel[keep]
    dist = dist[keep]
    c = math.cos(float(psi_rad))
    s = math.sin(float(psi_rad))
    longitudinal = c * rel[:, 0] + s * rel[:, 1]
    lateral_left = -s * rel[:, 0] + c * rel[:, 1]
    half_width = max(float(width) * 0.5, 0.75)
    half_length = max(float(length) * 0.5, 1.5)
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


def physical_pair(action: np.ndarray, next_action: np.ndarray, abs_acc_limit: float, delta_acc_limit: float) -> bool:
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(next_action)):
        return False
    if np.max(np.abs(action)) > abs_acc_limit or np.max(np.abs(next_action)) > abs_acc_limit:
        return False
    return bool(np.max(np.abs(next_action - action)) <= delta_acc_limit)


def process_file(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    country, scene, file_split = parse_file(path)
    dataset = COUNTRY_DATASETS[country]
    writer = DatasetWriter(args.out_dir, dataset, args.shard_size, file_prefix=path.stem)
    started = time.time()
    df = load_car_rows(path)
    frame_lookup = build_frame_lookup(df)
    kept = candidates = tracks_car = skipped_short = skipped_nonconsecutive = skipped_physical = 0

    for (case_id, track_id), rows in df.groupby(["case_i", "track_i"], sort=False):
        tracks_car += 1
        rows = rows.sort_values("frame_i")
        n = int(len(rows))
        if n < args.t_hist + 2:
            skipped_short += 1
            continue
        frames = rows["frame_i"].to_numpy(dtype=np.int64)
        timestamps = rows["timestamp_ms"].to_numpy(dtype=np.float64)
        vx = rows["vx"].to_numpy(dtype=np.float64)
        vy = rows["vy"].to_numpy(dtype=np.float64)
        ax, ay = velocity_acceleration(vx, vy, timestamps, args.accel_window, args.accel_polyorder)
        xy = rows[["x", "y"]].to_numpy(dtype=np.float64)
        psi = rows["psi_rad"].to_numpy(dtype=np.float64)
        length = rows["length"].to_numpy(dtype=np.float64)
        width = rows["width"].to_numpy(dtype=np.float64)

        features = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
        for i in range(n):
            features[i] = np.asarray(
                [
                    vx[i],
                    ax[i],
                    vy[i],
                    ay[i],
                    *direction_distances(
                        int(case_id),
                        int(frames[i]),
                        int(track_id),
                        float(xy[i, 0]),
                        float(xy[i, 1]),
                        float(psi[i]),
                        float(length[i]),
                        float(width[i]),
                        frame_lookup,
                        args.max_radius,
                    ),
                ],
                dtype=np.float32,
            )
        actions = features[:, [FEATURE_NAMES.index("xAcceleration"), FEATURE_NAMES.index("yAcceleration")]]
        starts = np.arange(0, n - args.t_hist - 1, max(1, args.stride), dtype=np.int64)
        for start in starts:
            candidates += 1
            end_state = int(start + args.t_hist)
            next_action_idx = int(start + args.t_hist + 1)
            frame_block = frames[start : next_action_idx + 1]
            if not np.all(np.diff(frame_block) == 1):
                skipped_nonconsecutive += 1
                continue
            state = features[start:end_state]
            next_state = features[start + 1 : end_state + 1]
            action = actions[end_state]
            next_action = actions[next_action_idx]
            if not physical_pair(action, next_action, args.abs_acc_limit, args.delta_acc_limit):
                skipped_physical += 1
                continue
            split = split_for_track(str(country), path.stem, int(case_id), int(track_id), str(file_split), args.val_file_test_ratio)
            writer.append(
                split,
                transition_sample(dataset, f"{path.stem}/case_{int(case_id)}", f"{int(case_id)}:{int(track_id)}", int(frames[end_state - 1]), int(frames[end_state]), state, action, next_state, next_action),
            )
            kept += 1
    writer.close()
    return {
        "file": path.name,
        "country": country,
        "dataset": dataset,
        "scene": scene,
        "file_split": file_split,
        "rows_car": int(len(df)),
        "tracks_car": int(tracks_car),
        "candidates": int(candidates),
        "kept": int(kept),
        "skipped_short_tracks": int(skipped_short),
        "skipped_nonconsecutive": int(skipped_nonconsecutive),
        "skipped_physical": int(skipped_physical),
        "split_counts": {key: int(value) for key, value in writer.counts.items()},
        "files": writer.files,
        "elapsed_sec": time.time() - started,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve_interaction_root(args.raw_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    files = sorted(root.glob(args.file_glob))
    if not files:
        raise FileNotFoundError(f"no files matching {args.file_glob!r} under {root}")
    file_summaries = []
    if args.workers <= 1:
        for path in files:
            result = process_file(path, args)
            file_summaries.append(result)
            print(json.dumps({"event": "file_done", **result}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_file, path, args): path for path in files}
            for future in as_completed(futures):
                result = future.result()
                file_summaries.append(result)
                print(json.dumps({"event": "file_done", **result}), flush=True)

    summaries = []
    for country, dataset in COUNTRY_DATASETS.items():
        country_files = [item for item in file_summaries if item["dataset"] == dataset]
        split_counts = {split: int(sum(item["split_counts"].get(split, 0) for item in country_files)) for split in ("train", "val", "test")}
        written_files: dict[str, list[str]] = {split: [] for split in ("train", "val", "test")}
        for item in country_files:
            for split, paths in item["files"].items():
                written_files[split].extend(paths)
        summaries.append({
            "dataset": dataset,
            "samples": int(sum(split_counts.values())),
            "split_counts": split_counts,
            "files": written_files,
            "elapsed_sec": time.time() - started,
            "schema": "trajvista_xy_psiphi_external_interaction_v1",
            "state": ["N", 12, len(FEATURE_NAMES)],
            "action": ["N", 2],
            "next_state": ["N", 12, len(FEATURE_NAMES)],
            "next_action": ["N", 2],
            "meta": ["meta_dataset", "meta_scene", "meta_track_id", "meta_frame_k", "meta_frame_next"],
            "feature_names": FEATURE_NAMES,
            "target_names": TARGET_NAMES,
            "country": country,
            "source_dataset": "INTERACTION",
            "label_source": "INTERACTION CSV vx/vy used directly; x/y acceleration from Savitzky-Golay first derivative of vx/vy",
            "neighbor_policy": "car-only nearest same-frame agents in ego heading sectors within max_radius; missing=-1",
            "split_policy": "official train files -> train; official val files hash-split by case/track into val/test",
            "val_file_test_ratio": args.val_file_test_ratio,
            "smoothing": {"velocity_derivative_window": args.accel_window, "polyorder": args.accel_polyorder},
            "files_processed": country_files,
            "candidates": int(sum(item["candidates"] for item in country_files)),
            "kept": int(sum(item["kept"] for item in country_files)),
            "skipped_short_tracks": int(sum(item["skipped_short_tracks"] for item in country_files)),
            "skipped_nonconsecutive": int(sum(item["skipped_nonconsecutive"] for item in country_files)),
            "skipped_physical": int(sum(item["skipped_physical"] for item in country_files)),
        })
    manifest = {
        "schema": "trajvista_xy_psiphi_external_interaction_v1",
        "source_root": str(root),
        "out_dir": str(args.out_dir),
        "datasets": list(COUNTRY_DATASETS.values()),
        "state": ["N", 12, len(FEATURE_NAMES)],
        "action": ["N", 2],
        "next_state": ["N", 12, len(FEATURE_NAMES)],
        "next_action": ["N", 2],
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "elapsed_sec": time.time() - started,
        "summaries": summaries,
    }
    summary_path = args.out_dir / "interaction_external_summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_path": str(summary_path), "elapsed_sec": manifest["elapsed_sec"]}), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TrajVista XY/PsiPhi shards from INTERACTION CSV files.")
    parser.add_argument("--raw-root", type=Path, required=True, help="INTERACTION root or its recorded_trackfiles directory.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--t-hist", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=100000)
    parser.add_argument("--max-radius", type=float, default=80.0)
    parser.add_argument("--accel-window", type=int, default=9)
    parser.add_argument("--accel-polyorder", type=int, default=2)
    parser.add_argument("--abs-acc-limit", type=float, default=10.0)
    parser.add_argument("--delta-acc-limit", type=float, default=10.0)
    parser.add_argument("--val-file-test-ratio", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--file-glob", default="DR_*_*.csv")
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
