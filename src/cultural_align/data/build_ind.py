from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    DatasetWriter,
    DIR_NAMES,
    append_transition,
    base_summary,
    ensure_consecutive,
    finite_rows,
    heading_sector_distances,
    iter_track_windows,
    load_assigned_track_ids,
    physical_pair,
    sample_limit_value,
    write_summary,
)


DATASET = "inD"
VEHICLE_CLASSES = {"car", "truck_bus"}
TRACK_COLUMNS = [
    "recordingId",
    "trackId",
    "frame",
    "xCenter",
    "yCenter",
    "heading",
    "width",
    "length",
    "xVelocity",
    "yVelocity",
    "xAcceleration",
    "yAcceleration",
]


def vehicle_ids(meta_path: Path) -> set[int]:
    meta = pd.read_csv(meta_path, usecols=["trackId", "class"])
    return set(int(v) for v in meta.loc[meta["class"].isin(VEHICLE_CLASSES), "trackId"].to_numpy())


def build_frame_lookup(df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    lookup: dict[int, dict[str, np.ndarray]] = {}
    for frame, rows in df.groupby("frame", sort=False):
        lookup[int(frame)] = {
            "track_id": rows["trackId"].to_numpy(dtype=np.int64),
            "xy": rows[["xCenter", "yCenter"]].to_numpy(dtype=np.float64),
        }
    return lookup


def direction_distances(row: Any, frame_lookup: dict[int, dict[str, np.ndarray]], max_radius: float) -> list[float]:
    same = frame_lookup.get(int(row.frame))
    if not same:
        return [-1.0] * len(DIR_NAMES)
    ego_xy = np.asarray([float(row.xCenter), float(row.yCenter)], dtype=np.float64)
    keep = same["track_id"] != int(row.trackId)
    if not np.any(keep):
        return [-1.0] * len(DIR_NAMES)
    rel = same["xy"][keep] - ego_xy[None, :]
    dist = np.linalg.norm(rel, axis=1)
    keep = (dist > 1e-6) & (dist <= max_radius)
    if not np.any(keep):
        return [-1.0] * len(DIR_NAMES)
    return heading_sector_distances(
        rel[keep],
        dist[keep],
        math.radians(float(row.heading)),
        max(float(row.width) * 0.5, 0.75),
        max(float(row.length) * 0.5, 1.5),
    )


def row_feature(row: Any, frame_lookup: dict[int, dict[str, np.ndarray]], max_radius: float) -> np.ndarray:
    return np.asarray(
        [float(row.xVelocity), float(row.xAcceleration), float(row.yVelocity), float(row.yAcceleration), *direction_distances(row, frame_lookup, max_radius)],
        dtype=np.float32,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    writer = DatasetWriter(args.out_dir, DATASET, args.shard_size)
    tracks_files = sorted(Path(args.raw_root).glob("*_tracks.csv"))
    if not tracks_files:
        raise FileNotFoundError(f"no inD *_tracks.csv under {args.raw_root}")
    limit = sample_limit_value(args.max_samples)
    total_kept = total_candidates = skipped_nonvehicle = skipped_unassigned = skipped_nonconsecutive = skipped_physical = 0
    scenes = []

    for tracks_path in tracks_files:
        if total_kept >= limit:
            break
        prefix = tracks_path.name.split("_")[0]
        scene_id = f"inD_{prefix}"
        assigned = load_assigned_track_ids(args.assignment_dir, DATASET, scene_id)
        vehicle_set = vehicle_ids(Path(args.raw_root) / f"{prefix}_tracksMeta.csv")
        df = finite_rows(pd.read_csv(tracks_path, usecols=TRACK_COLUMNS), TRACK_COLUMNS, skip={"recordingId"})
        before = len(df)
        df = df[df["trackId"].isin(vehicle_set)].copy()
        skipped_nonvehicle += int(before - len(df))
        before = len(df)
        df = df[df["trackId"].astype(str).isin(assigned)].copy()
        skipped_unassigned += int(before - len(df))
        frame_lookup = build_frame_lookup(df)
        scene_kept = scene_candidates = 0
        for track_id, rows in df.groupby("trackId", sort=False):
            if total_kept >= limit:
                break
            records = list(rows.sort_values("frame").itertuples(index=False))
            if len(records) < args.t_hist + 2:
                continue
            for state_rows, next_state_rows, action_row, next_action_row, frames in iter_track_windows(records, lambda r: int(r.frame), args.t_hist, args.stride):
                if total_kept >= limit:
                    break
                total_candidates += 1
                scene_candidates += 1
                if not ensure_consecutive(frames):
                    skipped_nonconsecutive += 1
                    continue
                state = np.stack([row_feature(r, frame_lookup, args.max_radius) for r in state_rows]).astype(np.float32)
                next_state = np.stack([row_feature(r, frame_lookup, args.max_radius) for r in next_state_rows]).astype(np.float32)
                action = np.asarray([float(action_row.xAcceleration), float(action_row.yAcceleration)], dtype=np.float32)
                next_action = np.asarray([float(next_action_row.xAcceleration), float(next_action_row.yAcceleration)], dtype=np.float32)
                if not physical_pair(action, next_action, args.abs_ax_limit, args.abs_ay_limit, args.delta_ax_limit, args.delta_ay_limit):
                    skipped_physical += 1
                    continue
                append_transition(writer, DATASET, scene_id, str(track_id), int(state_rows[-1].frame), int(action_row.frame), state, action, next_state, next_action)
                total_kept += 1
                scene_kept += 1
        scenes.append({"scene": scene_id, "assigned_tracks": len(assigned), "kept": scene_kept, "candidates": scene_candidates})

    summary = base_summary(DATASET, writer, started, {
        "source_root": str(args.raw_root),
        "assignment_dir": str(args.assignment_dir),
        "label_source": "inD CSV xAcceleration/yAcceleration; no resampling and no differencing",
        "assignment_filter": "vehicle class plus status=assigned only",
        "neighbor_policy": "ego-heading XY eight-sector nearest vehicle distances within max_radius; missing=-1",
        "max_radius": args.max_radius,
        "skipped_nonvehicle_rows": skipped_nonvehicle,
        "skipped_unassigned_or_rejected_rows": skipped_unassigned,
        "skipped_nonconsecutive": skipped_nonconsecutive,
        "skipped_physical": skipped_physical,
        "candidates": total_candidates,
        "scenes": scenes,
    })
    write_summary(Path(args.out_dir) / f"{DATASET}_summary.json", summary)
    print(json.dumps({"event": "done", "dataset": DATASET, "samples": summary["samples"], "split_counts": summary["split_counts"]}))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build inD TrajVista XY/PsiPhi NPZ shards.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing inD *_tracks.csv and *_tracksMeta.csv files.")
    parser.add_argument("--assignment-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=200000)
    parser.add_argument("--t-hist", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-radius", type=float, default=60.0)
    parser.add_argument("--abs-ax-limit", type=float, default=10.0)
    parser.add_argument("--abs-ay-limit", type=float, default=10.0)
    parser.add_argument("--delta-ax-limit", type=float, default=8.0)
    parser.add_argument("--delta-ay-limit", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
