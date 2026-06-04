#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RECORDING = "04"
DEFAULT_FRAME_START = 2028
DEFAULT_FRAME_END = 3027
DEFAULT_MIN_FRAMES = 14
VEHICLE_CLASSES = {"car", "truck_bus"}
TRACK_COLUMNS = [
    "recordingId",
    "trackId",
    "frame",
    "trackLifetime",
    "xCenter",
    "yCenter",
    "heading",
    "width",
    "length",
    "xVelocity",
    "yVelocity",
    "xAcceleration",
    "yAcceleration",
    "lonVelocity",
    "latVelocity",
    "lonAcceleration",
    "latAcceleration",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_track_meta(meta: pd.DataFrame, subset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for track_id, track_rows in subset.groupby("trackId", sort=True):
        item = meta.loc[meta["trackId"] == track_id].iloc[0].copy()
        item["initialFrame"] = int(track_rows["frame"].min())
        item["finalFrame"] = int(track_rows["frame"].max())
        item["numFrames"] = int(len(track_rows))
        rows.append(item)
    return pd.DataFrame(rows, columns=meta.columns)


def update_recording_meta(
    recording_meta: pd.DataFrame,
    track_meta: pd.DataFrame,
    frame_start: int,
    frame_end: int,
) -> pd.DataFrame:
    out = recording_meta.copy()
    if len(out) != 1:
        raise ValueError("expected exactly one row in inD recordingMeta file")
    frame_rate = float(out.loc[out.index[0], "frameRate"])
    frame_count = max(0, int(frame_end) - int(frame_start) + 1)
    out.loc[out.index[0], "duration"] = frame_count / frame_rate if frame_rate > 0 else 0.0
    out.loc[out.index[0], "numTracks"] = int(len(track_meta))
    out.loc[out.index[0], "numVehicles"] = int(track_meta["class"].isin(VEHICLE_CLASSES).sum())
    out.loc[out.index[0], "numVRUs"] = int((~track_meta["class"].isin(VEHICLE_CLASSES)).sum())
    return out


def create_subset(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    assignment_dir = Path(args.assignment_dir)
    recording = str(args.recording).zfill(2)
    scene_id = f"inD_{recording}"

    tracks_path = source_root / f"{recording}_tracks.csv"
    tracks_meta_path = source_root / f"{recording}_tracksMeta.csv"
    recording_meta_path = source_root / f"{recording}_recordingMeta.csv"
    assignment_path = assignment_dir / "inD" / f"{scene_id}.json"
    for path in [tracks_path, tracks_meta_path, recording_meta_path, assignment_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    tracks = pd.read_csv(tracks_path, usecols=TRACK_COLUMNS)
    tracks_meta = pd.read_csv(tracks_meta_path)
    recording_meta = pd.read_csv(recording_meta_path)
    assignment = read_json(assignment_path)

    assigned_items = [item for item in assignment.get("assignments", []) if item.get("status") == "assigned"]
    assigned_ids = {int(item["track_id"]) for item in assigned_items}
    vehicle_ids = set(
        int(v)
        for v in tracks_meta.loc[tracks_meta["class"].isin(VEHICLE_CLASSES), "trackId"].to_numpy()
    )
    subset = tracks.loc[
        (tracks["frame"] >= int(args.frame_start))
        & (tracks["frame"] <= int(args.frame_end))
        & tracks["trackId"].isin(assigned_ids & vehicle_ids)
    ].copy()
    if subset.empty:
        raise ValueError("frame window did not contain assigned vehicle tracks")

    counts = subset.groupby("trackId").size()
    kept_ids = set(int(v) for v in counts.loc[counts >= int(args.min_frames)].index.to_numpy())
    subset = subset.loc[subset["trackId"].isin(kept_ids)].copy()
    if subset.empty:
        raise ValueError("no selected tracks are long enough for demo windowing")

    subset.sort_values(["trackId", "frame"], inplace=True)
    subset["trackLifetime"] = subset.groupby("trackId").cumcount()
    track_meta_subset = update_track_meta(tracks_meta, subset)
    recording_meta_subset = update_recording_meta(
        recording_meta,
        track_meta_subset,
        int(args.frame_start),
        int(args.frame_end),
    )

    raw_out = out_root / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    subset.to_csv(raw_out / f"{recording}_tracks.csv", index=False, float_format="%.6f")
    track_meta_subset.to_csv(raw_out / f"{recording}_tracksMeta.csv", index=False, float_format="%.6f")
    recording_meta_subset.to_csv(raw_out / f"{recording}_recordingMeta.csv", index=False, float_format="%.8f")

    assignment_items = [item for item in assigned_items if int(item["track_id"]) in kept_ids]
    subset_assignment = {
        "scene_id": scene_id,
        "dataset": "inD",
        "source": {
            "recording": recording,
            "frame_start": int(args.frame_start),
            "frame_end": int(args.frame_end),
            "source_assignment": f"data/reference_paths/assignments/inD/{scene_id}.json",
        },
        "assignments": assignment_items,
        "summary": {
            "scene_id": scene_id,
            "dataset": "inD",
            "demo_subset": True,
            "n_tracks": int(len(assignment_items)),
            "counts": {"assigned": int(len(assignment_items))},
        },
    }
    assignment_out = out_root / "assignments" / "inD" / f"{scene_id}.json"
    write_json(assignment_out, subset_assignment)

    manifest = {
        "dataset": "inD",
        "scene_id": scene_id,
        "recording": recording,
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "min_frames_per_track": int(args.min_frames),
        "raw_rows": int(len(subset)),
        "tracks": [int(v) for v in sorted(kept_ids)],
        "source_files": [
            f"{recording}_tracks.csv",
            f"{recording}_tracksMeta.csv",
            f"{recording}_recordingMeta.csv",
        ],
        "assignment_file": str(assignment_out.relative_to(out_root)),
        "notes": "Small inD recording-04 excerpt for repository demos.",
    }
    write_json(out_root / "demo_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the small inD demo subset used by this release.")
    parser.add_argument("--source-root", type=Path, required=True, help="Directory containing full local inD CSV files.")
    parser.add_argument("--assignment-dir", type=Path, default=Path("data/reference_paths/assignments"))
    parser.add_argument("--out-root", type=Path, default=Path("data/demo/ind"))
    parser.add_argument("--recording", default=DEFAULT_RECORDING)
    parser.add_argument("--frame-start", type=int, default=DEFAULT_FRAME_START)
    parser.add_argument("--frame-end", type=int, default=DEFAULT_FRAME_END)
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    return parser.parse_args()


def main() -> None:
    manifest = create_subset(parse_args())
    print(json.dumps({"event": "done", **manifest}, indent=2))


if __name__ == "__main__":
    main()
