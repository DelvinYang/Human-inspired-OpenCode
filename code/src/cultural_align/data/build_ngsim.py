from __future__ import annotations

import argparse
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
    append_transition,
    base_summary,
    ensure_consecutive,
    finite_rows,
    iter_track_windows,
    load_assigned_track_ids,
    physical_pair,
    sample_limit_value,
    savgol_window,
    write_summary,
)


DATASET = "NGSIM"
FT_TO_M = 0.3048
FRAME_RATE = 10.0
DT = 1.0 / FRAME_RATE
TRACK_COLUMNS = [
    "Vehicle_ID",
    "Frame_ID",
    "Local_X",
    "Local_Y",
    "v_Length",
    "v_Width",
    "v_Class",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
]


def scene_id_from_csv(path: Path) -> str:
    return f"NGSIM_{path.stem.replace('trajectories-', '')}"


def add_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _track_id, rows in df.sort_values(["Vehicle_ID", "Frame_ID"]).groupby("Vehicle_ID", sort=False):
        rows = rows.sort_values("Frame_ID").drop_duplicates("Frame_ID", keep="first").copy()
        speed = rows["v_Vel"].to_numpy(dtype=np.float64) * FT_TO_M
        moving = np.flatnonzero(np.isfinite(speed) & (np.abs(speed) > 1e-6))
        if moving.size == 0:
            continue
        rows = rows.iloc[int(moving[0]) :].copy()
        n = len(rows)
        if n < 2:
            continue

        frames = rows["Frame_ID"].to_numpy(dtype=np.float64)
        x_raw = rows["Local_Y"].to_numpy(dtype=np.float64) * FT_TO_M
        y_raw = rows["Local_X"].to_numpy(dtype=np.float64) * FT_TO_M
        pos_win = savgol_window(n, preferred=51, poly=3)
        if pos_win is not None:
            x = savgol_filter(x_raw, window_length=pos_win, polyorder=3, deriv=0, mode="interp")
            y = savgol_filter(y_raw, window_length=pos_win, polyorder=3, deriv=0, mode="interp")
        else:
            x = x_raw
            y = y_raw

        dt = float(np.median(np.diff(frames)) / FRAME_RATE) if n > 1 else DT
        if not np.isfinite(dt) or dt <= 0:
            dt = DT
        deriv_win = savgol_window(n, preferred=31, poly=3)
        if deriv_win is not None:
            vx = savgol_filter(x, window_length=deriv_win, polyorder=3, deriv=1, delta=dt, mode="interp")
            vy = savgol_filter(y, window_length=deriv_win, polyorder=3, deriv=1, delta=dt, mode="interp")
            ax = savgol_filter(x, window_length=deriv_win, polyorder=3, deriv=2, delta=dt, mode="interp")
            ay = savgol_filter(y, window_length=deriv_win, polyorder=3, deriv=2, delta=dt, mode="interp")
        elif n >= 3:
            t = frames / FRAME_RATE
            vx = np.gradient(x, t)
            vy = np.gradient(y, t)
            ax = np.gradient(vx, t)
            ay = np.gradient(vy, t)
        else:
            vx = np.gradient(x, dt)
            vy = np.gradient(y, dt)
            ax = np.zeros_like(vx)
            ay = np.zeros_like(vy)

        rows["x"] = x
        rows["y"] = y
        rows["vx"] = vx
        rows["vy"] = vy
        rows["ax"] = ax
        rows["ay"] = ay
        rows["length_m"] = rows["v_Length"].to_numpy(dtype=np.float64) * FT_TO_M
        rows["width_m"] = rows["v_Width"].to_numpy(dtype=np.float64) * FT_TO_M
        parts.append(rows)
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def build_frame_lookup(df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    lookup: dict[int, dict[str, np.ndarray]] = {}
    for frame, rows in df.groupby("Frame_ID", sort=False):
        lookup[int(frame)] = {
            "track_id": rows["Vehicle_ID"].to_numpy(dtype=np.int64),
            "xy": rows[["x", "y"]].to_numpy(dtype=np.float64),
            "lane": rows["Lane_ID"].to_numpy(dtype=np.int64),
        }
    return lookup


def direction_distances(row: Any, frame_lookup: dict[int, dict[str, np.ndarray]], max_radius: float) -> list[float]:
    bucket = {
        "preceding": math.inf,
        "following": math.inf,
        "leftPreceding": math.inf,
        "leftAlongside": math.inf,
        "leftFollowing": math.inf,
        "rightPreceding": math.inf,
        "rightAlongside": math.inf,
        "rightFollowing": math.inf,
    }
    same = frame_lookup.get(int(row.Frame_ID))
    if not same:
        return [-1.0] * 8
    ego_xy = np.asarray([float(row.x), float(row.y)], dtype=np.float64)
    ids = same["track_id"]
    xy = same["xy"]
    lanes = same["lane"]
    keep = ids != int(row.Vehicle_ID)
    if not np.any(keep):
        return [-1.0] * 8
    other_xy = xy[keep]
    other_lanes = lanes[keep]
    dist = np.linalg.norm(other_xy - ego_xy[None, :], axis=1)
    keep = (dist > 1e-6) & (dist <= max_radius)
    if not np.any(keep):
        return [-1.0] * 8
    other_xy = other_xy[keep]
    other_lanes = other_lanes[keep]
    dist = dist[keep]

    ego_lane = int(row.Lane_ID)
    half_length = max(float(row.length_m) * 0.5, 1.5)
    for xy_i, lane_i, d in zip(other_xy, other_lanes, dist):
        dx = float(xy_i[0] - ego_xy[0])
        lane_delta = int(lane_i) - ego_lane
        if lane_delta == 0:
            if dx > half_length:
                name = "preceding"
            elif dx < -half_length:
                name = "following"
            else:
                continue
        elif lane_delta == -1:
            if abs(dx) <= half_length:
                name = "leftAlongside"
            elif dx > 0:
                name = "leftPreceding"
            else:
                name = "leftFollowing"
        elif lane_delta == 1:
            if abs(dx) <= half_length:
                name = "rightAlongside"
            elif dx > 0:
                name = "rightPreceding"
            else:
                name = "rightFollowing"
        else:
            continue
        if float(d) < bucket[name]:
            bucket[name] = float(d)
    return [bucket[name] if math.isfinite(bucket[name]) else -1.0 for name in bucket]


def row_feature(row: Any, frame_lookup: dict[int, dict[str, np.ndarray]], max_radius: float) -> np.ndarray:
    return np.asarray(
        [float(row.vx), float(row.ax), float(row.vy), float(row.ay), *direction_distances(row, frame_lookup, max_radius)],
        dtype=np.float32,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    writer = DatasetWriter(args.out_dir, DATASET, args.shard_size)
    csv_files = sorted(Path(args.raw_root).glob("trajectories-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no NGSIM trajectories-*.csv under {args.raw_root}")
    limit = sample_limit_value(args.max_samples)
    total_kept = total_candidates = skipped_class1 = skipped_unassigned = skipped_nonconsecutive = skipped_physical = 0
    scenes = []

    for csv_path in csv_files:
        if total_kept >= limit:
            break
        scene_id = scene_id_from_csv(csv_path)
        assigned = load_assigned_track_ids(args.assignment_dir, DATASET, scene_id)
        df = finite_rows(pd.read_csv(csv_path, usecols=TRACK_COLUMNS), TRACK_COLUMNS)
        before = len(df)
        df = df[df["v_Class"].astype(int) != 1].copy()
        skipped_class1 += int(before - len(df))
        before = len(df)
        df = df[df["Vehicle_ID"].astype(str).isin(assigned)].copy()
        skipped_unassigned += int(before - len(df))
        if df.empty:
            scenes.append({"scene": scene_id, "assigned_tracks": len(assigned), "kept": 0, "candidates": 0})
            continue
        df = add_kinematics(df)
        df = df[np.isfinite(df[["x", "y", "vx", "vy", "ax", "ay", "length_m", "width_m", "Lane_ID"]]).all(axis=1)].copy()
        frame_lookup = build_frame_lookup(df)
        scene_kept = scene_candidates = 0
        for track_id, rows in df.groupby("Vehicle_ID", sort=False):
            if total_kept >= limit:
                break
            records = list(rows.sort_values("Frame_ID").itertuples(index=False))
            if len(records) < args.t_hist + 2:
                continue
            for state_rows, next_state_rows, action_row, next_action_row, frames in iter_track_windows(records, lambda r: int(r.Frame_ID), args.t_hist, args.stride):
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
                append_transition(writer, DATASET, scene_id, str(track_id), int(state_rows[-1].Frame_ID), int(action_row.Frame_ID), state, action, next_state, next_action)
                total_kept += 1
                scene_kept += 1
        scenes.append({"scene": scene_id, "assigned_tracks": len(assigned), "kept": scene_kept, "candidates": scene_candidates})

    summary = base_summary(DATASET, writer, started, {
        "source_root": str(args.raw_root),
        "assignment_dir": str(args.assignment_dir),
        "label_source": "NGSIM SG derivatives from x=Local_Y*ft_to_m, y=Local_X*ft_to_m; no Frenet labels",
        "assignment_filter": "status=assigned only; v_Class=1 dropped",
        "neighbor_policy": "Lane_ID-based eight-sector nearest vehicle distances within max_radius; missing=-1",
        "max_radius": args.max_radius,
        "smoothing": {"position_window": 51, "derivative_window": 31, "polyorder": 3},
        "skipped_class1_rows": skipped_class1,
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
    parser = argparse.ArgumentParser(description="Build NGSIM TrajVista XY/PsiPhi NPZ shards.")
    parser.add_argument("--raw-root", type=Path, required=True, help="NGSIM root containing trajectories-*.csv files.")
    parser.add_argument("--assignment-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=200000)
    parser.add_argument("--t-hist", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-radius", type=float, default=100.0)
    parser.add_argument("--abs-ax-limit", type=float, default=8.0)
    parser.add_argument("--abs-ay-limit", type=float, default=5.0)
    parser.add_argument("--delta-ax-limit", type=float, default=6.0)
    parser.add_argument("--delta-ay-limit", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
