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


DATASET = "sinD"
VEHICLE_TYPES = {"car", "truck", "bus", "van", "motorcycle", "tricycle"}
TRACK_COLUMNS = [
    "track_id",
    "frame_id",
    "agent_type",
    "x",
    "y",
    "vx",
    "vy",
    "ax",
    "ay",
    "heading_rad",
]


def scene_id_from_csv(path: Path) -> str:
    city = path.parents[1].name
    scene = path.parent.name
    return f"sinD_{city}_{scene}"


def vehicle_rows(df: pd.DataFrame) -> pd.DataFrame:
    lowered = df["agent_type"].astype(str).str.lower()
    return df.loc[lowered.isin(VEHICLE_TYPES)].copy()


def build_frame_lookup(df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    lookup: dict[int, dict[str, np.ndarray]] = {}
    for frame, rows in df.groupby("frame_id", sort=False):
        lookup[int(frame)] = {
            "track_id": rows["track_id"].to_numpy(dtype=np.int64),
            "xy": rows[["x", "y"]].to_numpy(dtype=np.float64),
        }
    return lookup


def direction_distances(row: Any, frame_lookup: dict[int, dict[str, np.ndarray]], max_radius: float) -> list[float]:
    same = frame_lookup.get(int(row.frame_id))
    if not same:
        return [-1.0] * 8
    ego_xy = np.asarray([float(row.x), float(row.y)], dtype=np.float64)
    keep = same["track_id"] != int(row.track_id)
    if not np.any(keep):
        return [-1.0] * 8
    rel = same["xy"][keep] - ego_xy[None, :]
    dist = np.linalg.norm(rel, axis=1)
    keep = (dist > 1e-6) & (dist <= max_radius)
    if not np.any(keep):
        return [-1.0] * 8
    return heading_sector_distances(rel[keep], dist[keep], float(row.heading_rad), half_width=1.0, half_length=2.5)


def row_feature(row: Any, frame_lookup: dict[int, dict[str, np.ndarray]], max_radius: float) -> np.ndarray:
    return np.asarray(
        [float(row.vx), float(row.ax), float(row.vy), float(row.ay), *direction_distances(row, frame_lookup, max_radius)],
        dtype=np.float32,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    writer = DatasetWriter(args.out_dir, DATASET, args.shard_size)
    csv_files = sorted(Path(args.raw_root).glob("*/*/Veh_smoothed_tracks.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no sinD Veh_smoothed_tracks.csv under {args.raw_root}")
    limit = sample_limit_value(args.max_samples)
    total_kept = total_candidates = skipped_nonvehicle = skipped_unassigned = skipped_nonconsecutive = skipped_physical = 0
    scenes = []

    for csv_path in csv_files:
        if total_kept >= limit:
            break
        scene_id = scene_id_from_csv(csv_path)
        assigned = load_assigned_track_ids(args.assignment_dir, DATASET, scene_id)
        df = finite_rows(pd.read_csv(csv_path, usecols=TRACK_COLUMNS), TRACK_COLUMNS, skip={"agent_type"})
        before = len(df)
        df = vehicle_rows(df)
        skipped_nonvehicle += int(before - len(df))
        before = len(df)
        df = df[df["track_id"].astype(str).isin(assigned)].copy()
        skipped_unassigned += int(before - len(df))
        if df.empty:
            scenes.append({"scene": scene_id, "assigned_tracks": len(assigned), "kept": 0, "candidates": 0})
            continue
        frame_lookup = build_frame_lookup(df)
        scene_kept = scene_candidates = 0
        for track_id, rows in df.groupby("track_id", sort=False):
            if total_kept >= limit:
                break
            records = list(rows.sort_values("frame_id").itertuples(index=False))
            if len(records) < args.t_hist + 2:
                continue
            for state_rows, next_state_rows, action_row, next_action_row, frames in iter_track_windows(records, lambda r: int(r.frame_id), args.t_hist, args.stride):
                if total_kept >= limit:
                    break
                total_candidates += 1
                scene_candidates += 1
                if not ensure_consecutive(frames):
                    skipped_nonconsecutive += 1
                    continue
                state = np.stack([row_feature(r, frame_lookup, args.max_radius) for r in state_rows]).astype(np.float32)
                next_state = np.stack([row_feature(r, frame_lookup, args.max_radius) for r in next_state_rows]).astype(np.float32)
                action = np.asarray([float(action_row.ax), float(action_row.ay)], dtype=np.float32)
                next_action = np.asarray([float(next_action_row.ax), float(next_action_row.ay)], dtype=np.float32)
                if not physical_pair(action, next_action, args.abs_ax_limit, args.abs_ay_limit, args.delta_ax_limit, args.delta_ay_limit):
                    skipped_physical += 1
                    continue
                append_transition(writer, DATASET, scene_id, str(track_id), int(state_rows[-1].frame_id), int(action_row.frame_id), state, action, next_state, next_action)
                total_kept += 1
                scene_kept += 1
        scenes.append({"scene": scene_id, "assigned_tracks": len(assigned), "kept": scene_kept, "candidates": scene_candidates})

    summary = base_summary(DATASET, writer, started, {
        "source_root": str(args.raw_root),
        "assignment_dir": str(args.assignment_dir),
        "label_source": "sinD Veh_smoothed_tracks.csv vx/vy/ax/ay; no resampling and no differencing",
        "assignment_filter": "vehicle type plus status=assigned only",
        "neighbor_policy": "ego-heading XY eight-sector nearest assigned vehicle distances within max_radius; missing=-1",
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
    parser = argparse.ArgumentParser(description="Build sinD TrajVista XY/PsiPhi NPZ shards.")
    parser.add_argument("--raw-root", type=Path, required=True, help="sinD root containing city/scene/Veh_smoothed_tracks.csv files.")
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
