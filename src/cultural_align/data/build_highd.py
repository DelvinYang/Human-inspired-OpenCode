from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    DatasetWriter,
    EIGHT_ID_COLUMNS,
    append_transition,
    base_summary,
    ensure_consecutive,
    finite_rows,
    iter_track_windows,
    load_assigned_track_ids,
    physical_pair,
    sample_limit_value,
    write_summary,
)


DATASET = "HighD"
TRACK_COLUMNS = [
    "frame",
    "id",
    "x",
    "y",
    "xVelocity",
    "yVelocity",
    "xAcceleration",
    "yAcceleration",
    *EIGHT_ID_COLUMNS,
]


def build_frame_lookup(df: pd.DataFrame) -> dict[int, dict[int, tuple[float, float]]]:
    lookup: dict[int, dict[int, tuple[float, float]]] = {}
    for row in df[["frame", "id", "x", "y"]].itertuples(index=False):
        lookup.setdefault(int(row.frame), {})[int(row.id)] = (float(row.x), float(row.y))
    return lookup


def eight_distances(row: Any, frame_lookup: dict[int, dict[int, tuple[float, float]]]) -> list[float]:
    same = frame_lookup.get(int(row.frame), {})
    ego_xy = np.asarray([float(row.x), float(row.y)], dtype=np.float64)
    out: list[float] = []
    for col in EIGHT_ID_COLUMNS:
        other_id = int(getattr(row, col))
        if other_id == 0 or other_id not in same:
            out.append(-1.0)
            continue
        out.append(float(np.linalg.norm(np.asarray(same[other_id], dtype=np.float64) - ego_xy)))
    return out


def row_feature(row: Any, frame_lookup: dict[int, dict[int, tuple[float, float]]]) -> np.ndarray:
    return np.asarray(
        [
            float(row.xVelocity),
            float(row.xAcceleration),
            float(row.yVelocity),
            float(row.yAcceleration),
            *eight_distances(row, frame_lookup),
        ],
        dtype=np.float32,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    writer = DatasetWriter(args.out_dir, DATASET, args.shard_size)
    tracks_files = sorted(Path(args.raw_root).glob("*_tracks.csv"))
    if not tracks_files:
        raise FileNotFoundError(f"no HighD *_tracks.csv under {args.raw_root}")

    limit = sample_limit_value(args.max_samples)
    total_kept = total_candidates = skipped_unassigned = skipped_nonconsecutive = skipped_physical = 0
    scenes = []

    for tracks_path in tracks_files:
        if total_kept >= limit:
            break
        recording = tracks_path.name.split("_")[0]
        scene_id = f"HighD_{recording}"
        assigned = load_assigned_track_ids(args.assignment_dir, DATASET, scene_id)
        df = finite_rows(pd.read_csv(tracks_path, usecols=TRACK_COLUMNS), TRACK_COLUMNS)
        before = len(df)
        df = df[df["id"].astype(str).isin(assigned)].copy()
        skipped_unassigned += int(before - len(df))
        frame_lookup = build_frame_lookup(df)
        scene_kept = scene_candidates = 0
        for track_id, rows in df.groupby("id", sort=False):
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
                state = np.stack([row_feature(r, frame_lookup) for r in state_rows]).astype(np.float32)
                next_state = np.stack([row_feature(r, frame_lookup) for r in next_state_rows]).astype(np.float32)
                action = np.asarray([float(action_row.xAcceleration), float(action_row.yAcceleration)], dtype=np.float32)
                next_action = np.asarray([float(next_action_row.xAcceleration), float(next_action_row.yAcceleration)], dtype=np.float32)
                if not physical_pair(action, next_action, args.abs_ax_limit, args.abs_ay_limit, args.delta_ax_limit, args.delta_ay_limit):
                    skipped_physical += 1
                    continue
                append_transition(writer, DATASET, scene_id, str(track_id), int(state_rows[-1].frame), int(action_row.frame), state, action, next_state, next_action)
                total_kept += 1
                scene_kept += 1
        scenes.append({"scene": scene_id, "assigned_tracks": len(assigned), "kept": scene_kept, "candidates": scene_candidates})

    summary = base_summary(
        DATASET,
        writer,
        started,
        {
            "source_root": str(args.raw_root),
            "assignment_dir": str(args.assignment_dir),
            "label_source": "HighD CSV xAcceleration/yAcceleration; no resampling and no differencing",
            "assignment_filter": "status=assigned only",
            "skipped_unassigned_or_rejected_rows": skipped_unassigned,
            "skipped_nonconsecutive": skipped_nonconsecutive,
            "skipped_physical": skipped_physical,
            "candidates": total_candidates,
            "scenes": scenes,
        },
    )
    write_summary(Path(args.out_dir) / f"{DATASET}_summary.json", summary)
    print(json.dumps({"event": "done", "dataset": DATASET, "samples": summary["samples"], "split_counts": summary["split_counts"]}))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HighD TrajVista XY/PsiPhi NPZ shards.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing HighD *_tracks.csv files.")
    parser.add_argument("--assignment-dir", type=Path, required=True, help="Directory containing assignment JSON files by dataset.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=200000)
    parser.add_argument("--t-hist", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--abs-ax-limit", type=float, default=10.0)
    parser.add_argument("--abs-ay-limit", type=float, default=5.0)
    parser.add_argument("--delta-ax-limit", type=float, default=5.0)
    parser.add_argument("--delta-ay-limit", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
