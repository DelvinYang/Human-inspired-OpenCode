from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


Unit = Literal["meter", "pixel"]


@dataclass
class Polyline:
    polyline_id: str
    points: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackRaw:
    track_id: str
    xy: np.ndarray
    frames: np.ndarray
    heading: np.ndarray | None = None
    speed: np.ndarray | None = None
    lane_id: np.ndarray | None = None
    agent_type: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneRaw:
    scene_id: str
    dataset: str
    tracks: list[TrackRaw]
    map_image_path: Path | None = None
    lane_geometry: list[Polyline] | None = None
    frame_rate: float = 10.0
    unit: Unit = "meter"
    transform_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePath:
    cluster_id: int
    size: int
    persistence: float | None
    medoid: np.ndarray
    member_track_ids: list[str]
    entry: list[float]
    exit: list[float]
    meta: dict[str, Any] = field(default_factory=dict)
