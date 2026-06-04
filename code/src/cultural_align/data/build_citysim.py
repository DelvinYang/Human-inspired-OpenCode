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
    heading_sector_distances,
    iter_track_windows,
    load_assigned_track_ids,
    physical_pair,
    sample_limit_value,
    savgol_window,
    write_summary,
)


DATASET = "CitySim"
PIXEL_TO_M = 0.0421
FRAME_RATE = 30.0
DT = 1.0 / FRAME_RATE
CITYSIM_INTERSECTIONS = ("IntersectionA", "IntersectionB", "IntersectionD", "IntersectionE")
TRACK_COLUMNS = [
    "frameNum",
    "carId",
    "carCenterX",
    "carCenterY",
    "carCenterXft",
    "carCenterYft",
    "speed",
    "course",
    "heading",
]


def trajectory_dir(scene_dir: Path) -> Path:
    for name in ("Trajectories", "Trajectory"):
        path = scene_dir / name
        if path.exists():
            return path
    return scene_dir / "Trajectories"


def add_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _track_uid, rows in df.sort_values(["track_uid", "frameNum"]).groupby("track_uid", sort=False):
        rows = rows.sort_values("frameNum").drop_duplicates("frameNum", keep="first").copy()
        speed = rows["speed"].to_numpy(dtype=np.float64)
        moving = np.flatnonzero(np.isfinite(speed) & (np.abs(speed) > 1e-6))
        if moving.size == 0:
            continue
        rows = rows.iloc[int(moving[0]) :].copy()
        n = len(rows)
        if n < 2:
            continue

        frames = rows["frameNum"].to_numpy(dtype=np.float64)
        x_raw = rows["carCenterX"].to_numpy(dtype=np.float64) * PIXEL_TO_M
        y_raw = rows["carCenterY"].to_numpy(dtype=np.float64) * PIXEL_TO_M
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

        theta = np.arctan2(vy, vx)
        slow = np.hypot(vx, vy) < 1e-3
        if np.any(slow):
            theta[slow] = np.deg2rad(rows["course"].to_numpy(dtype=np.float64)[slow])
        rows["x_m"] = x
        rows["y_m"] = y
        rows["vx"] = vx
        rows["vy"] = vy
        rows["ax"] = ax
        rows["ay"] = ay
        rows["theta"] = theta
        parts.append(rows)
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def mostly_stationary(history_rows: list[Any]) -> bool:
    speed = np.asarray([math.hypot(float(r.vx), float(r.vy)) for r in history_rows], dtype=np.float64)
    return int(np.count_nonzero(speed < 0.1)) > (len(history_rows) // 2)


def build_frame_lookup(df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    lookup: dict[int, dict[str, np.ndarray]] = {}
    for frame, rows in df.groupby("frameNum", sort=False):
        lookup[int(frame)] = {
            "track_uid": rows["track_uid"].to_numpy(dtype=str),
            "xy": rows[["x_m", "y_m"]].to_numpy(dtype=np.float64),
        }
    return lookup


def direction_distances(row: Any, frame_lookup: dict[int, dict[str, np.ndarray]], max_radius: float) -> list[float]:
    same = frame_lookup.get(int(row.frameNum))
    if not same:
        return [-1.0] * 8
    ego_xy = np.asarray([float(row.x_m), float(row.y_m)], dtype=np.float64)
    keep = same["track_uid"] != str(row.track_uid)
    if not np.any(keep):
        return [-1.0] * 8
    rel = same["xy"][keep] - ego_xy[None, :]
    dist = np.linalg.norm(rel, axis=1)
    keep = (dist > 1e-6) & (dist <= max_radius)
    if not np.any(keep):
        return [-1.0] * 8
    return heading_sector_distances(rel[keep], dist[keep], float(row.theta), half_width=1.0, half_length=2.5)


def row_feature(row: Any, frame_lookup: dict[int, dict[str, np.ndarray]], max_radius: float) -> np.ndarray:
    return np.asarray(
        [float(row.vx), float(row.ax), float(row.vy), float(row.ay), *direction_distances(row, frame_lookup, max_radius)],
        dtype=np.float32,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    writer = DatasetWriter(args.out_dir, DATASET, args.shard_size)
    limit = sample_limit_value(args.max_samples)
    total_kept = total_candidates = skipped_unassigned = skipped_nonconsecutive = skipped_physical = skipped_stationary = 0
    scenes = []

    for scene in CITYSIM_INTERSECTIONS:
        if total_kept >= limit:
            break
        scene_id = f"CitySim_{scene}"
        assigned = load_assigned_track_ids(args.assignment_dir, DATASET, scene_id)
        csv_files = sorted(trajectory_dir(Path(args.raw_root) / scene).glob("*.csv"))
        if not csv_files:
            scenes.append({"scene": scene_id, "assigned_tracks": len(assigned), "kept": 0, "candidates": 0, "warning": "no trajectory csv files"})
            continue
        scene_kept = scene_candidates = 0
        for csv_path in csv_files:
            if total_kept >= limit:
                break
            recording = csv_path.stem
            df = finite_rows(pd.read_csv(csv_path, usecols=TRACK_COLUMNS), TRACK_COLUMNS)
            df["track_uid"] = [f"{recording}:{int(v)}" for v in df["carId"].to_numpy()]
            before = len(df)
            df = df[df["track_uid"].astype(str).isin(assigned)].copy()
            skipped_unassigned += int(before - len(df))
            if df.empty:
                continue
            df = add_kinematics(df)
            df = df[np.isfinite(df[["vx", "vy", "ax", "ay", "x_m", "y_m", "theta", "course"]]).all(axis=1)].copy()
            frame_lookup = build_frame_lookup(df)
            for track_id, rows in df.groupby("track_uid", sort=False):
                if total_kept >= limit:
                    break
                records = list(rows.sort_values("frameNum").itertuples(index=False))
                if len(records) < args.t_hist + 2:
                    continue
                for state_rows, next_state_rows, action_row, next_action_row, frames in iter_track_windows(records, lambda r: int(r.frameNum), args.t_hist, args.stride):
                    if total_kept >= limit:
                        break
                    total_candidates += 1
                    scene_candidates += 1
                    if not ensure_consecutive(frames):
                        skipped_nonconsecutive += 1
                        continue
                    if mostly_stationary(state_rows):
                        skipped_stationary += 1
                        continue
                    state = np.stack([row_feature(r, frame_lookup, args.max_radius) for r in state_rows]).astype(np.float32)
                    next_state = np.stack([row_feature(r, frame_lookup, args.max_radius) for r in next_state_rows]).astype(np.float32)
                    action = np.asarray([float(action_row.ax), float(action_row.ay)], dtype=np.float32)
                    next_action = np.asarray([float(next_action_row.ax), float(next_action_row.ay)], dtype=np.float32)
                    if not physical_pair(action, next_action, args.abs_ax_limit, args.abs_ay_limit, args.delta_ax_limit, args.delta_ay_limit):
                        skipped_physical += 1
                        continue
                    append_transition(writer, DATASET, scene_id, str(track_id), int(state_rows[-1].frameNum), int(action_row.frameNum), state, action, next_state, next_action)
                    total_kept += 1
                    scene_kept += 1
        scenes.append({"scene": scene_id, "assigned_tracks": len(assigned), "kept": scene_kept, "candidates": scene_candidates})

    summary = base_summary(DATASET, writer, started, {
        "source_root": str(args.raw_root),
        "assignment_dir": str(args.assignment_dir),
        "label_source": "CitySim TrajVista-style SG derivatives from smoothed XY centers",
        "position_source": "carCenterX/carCenterY pixel centers converted to meters with PIXEL_TO_M=0.0421",
        "assignment_filter": "status=assigned only; ABDE intersections only",
        "neighbor_policy": "smoothed XY heading eight-sector nearest vehicle distances within max_radius; missing=-1",
        "max_radius": args.max_radius,
        "smoothing": {"position_window": 51, "derivative_window": 31, "polyorder": 3},
        "skipped_unassigned_or_rejected_rows": skipped_unassigned,
        "skipped_nonconsecutive": skipped_nonconsecutive,
        "skipped_stationary_history": skipped_stationary,
        "skipped_physical": skipped_physical,
        "candidates": total_candidates,
        "scenes": scenes,
    })
    write_summary(Path(args.out_dir) / f"{DATASET}_summary.json", summary)
    print(json.dumps({"event": "done", "dataset": DATASET, "samples": summary["samples"], "split_counts": summary["split_counts"]}))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CitySim TrajVista XY/PsiPhi NPZ shards.")
    parser.add_argument("--raw-root", type=Path, required=True, help="CitySim root containing IntersectionA/B/D/E directories.")
    parser.add_argument("--assignment-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=200000)
    parser.add_argument("--t-hist", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-radius", type=float, default=60.0)
    parser.add_argument("--abs-ax-limit", type=float, default=12.0)
    parser.add_argument("--abs-ay-limit", type=float, default=12.0)
    parser.add_argument("--delta-ax-limit", type=float, default=12.0)
    parser.add_argument("--delta-ay-limit", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
