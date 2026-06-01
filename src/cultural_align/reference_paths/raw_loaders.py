from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Proj

from .models import Polyline, SceneRaw, TrackRaw

CITYSIM_INTERSECTIONS = {"IntersectionA", "IntersectionB", "IntersectionD", "IntersectionE"}
DEGREES_TO_METERS = 111_320.0
FT_TO_M = 0.3048
IND_BACKGROUND_DOWNSAMPLE = 12.0


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _group_tracks(
    df: pd.DataFrame,
    id_col: str,
    frame_col: str,
    x_col: str,
    y_col: str,
    heading_col: str | None = None,
    speed_col: str | None = None,
    lane_col: str | None = None,
    agent_col: str | None = None,
    meta_by_id: dict[str, dict] | None = None,
) -> list[TrackRaw]:
    tracks: list[TrackRaw] = []
    for tid, rows in df.groupby(id_col, sort=False):
        rows = rows.sort_values(frame_col)
        xy = rows[[x_col, y_col]].to_numpy(dtype=np.float64)
        frames = rows[frame_col].to_numpy()
        heading = rows[heading_col].to_numpy(dtype=np.float64) if heading_col and heading_col in rows else None
        speed = rows[speed_col].to_numpy(dtype=np.float64) if speed_col and speed_col in rows else None
        lane = rows[lane_col].to_numpy() if lane_col and lane_col in rows else None
        agent = str(rows[agent_col].iloc[0]) if agent_col and agent_col in rows else None
        meta = (meta_by_id or {}).get(str(tid), {})
        tracks.append(TrackRaw(str(tid), xy, frames, heading, speed, lane, agent, meta))
    return tracks


def _meta_by_id(path: Path, id_col: str, class_col: str | None = "class") -> dict[str, dict]:
    if not path.exists():
        return {}
    df = _read_csv(path)
    out = {}
    for _, row in df.iterrows():
        key = str(row[id_col])
        meta = row.to_dict()
        if class_col and class_col in row:
            meta["agent_type"] = row[class_col]
        out[key] = meta
    return out


def _recording_frame_rate(path: Path, default: float = 10.0) -> float:
    if not path.exists():
        return default
    df = _read_csv(path)
    for col in ("frameRate", "frame_rate"):
        if col in df.columns and len(df):
            return float(df[col].iloc[0])
    return default


def _recording_row(path: Path) -> dict:
    if not path.exists():
        return {}
    df = _read_csv(path)
    if not len(df):
        return {}
    return df.iloc[0].to_dict()


def _image_size(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    with Image.open(path) as im:
        return im.size


def _parse_float_list(value: object) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [float(x) for x in str(value).split(";") if str(x).strip()]


def _background_extent(x0: float, x1: float, y0: float, y1: float, y_axis: str, source: str) -> dict:
    return {
        "x0": float(x0),
        "x1": float(x1),
        "y0": float(y0),
        "y1": float(y1),
        "y_axis": y_axis,
        "source": source,
    }


def _highd_transform_meta(prefix: str, rec_path: Path, bg_path: Path) -> dict:
    row = _recording_row(rec_path)
    meta = {"raw_prefix": prefix}
    upper = _parse_float_list(row.get("upperLaneMarkings"))
    lower = _parse_float_list(row.get("lowerLaneMarkings"))
    size = _image_size(bg_path) if bg_path.exists() else None
    if upper and lower and size:
        width_px, height_px = size
        world_height = max(lower) + min(upper)
        if world_height > 0:
            px_per_meter = height_px / world_height
            world_width = width_px / px_per_meter
            meta["background_extent"] = _background_extent(
                0.0,
                world_width,
                0.0,
                world_height,
                "down",
                "highd_lane_markings_plus_image_aspect",
            )
            meta["image_scale"] = {"px_per_meter": px_per_meter, "meter_per_px": 1.0 / px_per_meter}
    meta["recording_meta"] = {
        "upperLaneMarkings": upper,
        "lowerLaneMarkings": lower,
        "speedLimit": row.get("speedLimit"),
        "locationId": row.get("locationId"),
    }
    return meta


def _ind_transform_meta(prefix: str, rec: pd.DataFrame, bg_path: Path) -> dict:
    location_id = int(rec["locationId"].iloc[0]) if "locationId" in rec and len(rec) else None
    px_to_meter = float(rec["orthoPxToMeter"].iloc[0]) if "orthoPxToMeter" in rec and len(rec) else None
    meta = {"raw_prefix": prefix, "location_id": location_id, "orthoPxToMeter": px_to_meter}
    size = _image_size(bg_path) if bg_path.exists() else None
    if size and px_to_meter:
        width_px, height_px = size
        meter_per_display_px = px_to_meter * IND_BACKGROUND_DOWNSAMPLE
        meta["background_extent"] = _background_extent(
            0.0,
            width_px * meter_per_display_px,
            -height_px * meter_per_display_px,
            0.0,
            "up",
            "ind_recordingMeta_orthoPxToMeter_image_size",
        )
        meta["image_scale"] = {
            "meter_per_original_px": px_to_meter,
            "display_downsample": IND_BACKGROUND_DOWNSAMPLE,
            "meter_per_display_px": meter_per_display_px,
        }
    return meta


def _utm_projector(lat_origin: float = 0.0, lon_origin: float = 0.0):
    zone = int(np.floor((lon_origin + 180.0) / 6.0) + 1)
    projector = Proj(proj="utm", ellps="WGS84", zone=zone, datum="WGS84")
    x_origin, y_origin = projector(lon_origin, lat_origin)

    def latlon2xy(lat: float, lon: float) -> np.ndarray:
        x, y = projector(lon, lat)
        return np.asarray([x - x_origin, y - y_origin], dtype=np.float64)

    return latlon2xy


def _sind_osm_geometry(osm_path: Path | None) -> tuple[list[Polyline] | None, dict | None]:
    if not osm_path or not osm_path.exists():
        return None, None
    root = ET.parse(osm_path).getroot()
    project = _utm_projector(0.0, 0.0)
    nodes = {}
    for node in root.findall("node"):
        lat = float(node.attrib["lat"])
        lon = float(node.attrib["lon"])
        nodes[node.attrib["id"]] = project(lat, lon)
    if not nodes:
        return None, None
    points = np.asarray(list(nodes.values()), dtype=np.float64)
    polylines: list[Polyline] = []
    for way in root.findall("way"):
        tags = {tag.attrib.get("k"): tag.attrib.get("v") for tag in way.findall("tag")}
        refs = [nd.attrib["ref"] for nd in way.findall("nd") if nd.attrib["ref"] in nodes]
        if len(refs) < 2:
            continue
        pts = np.vstack([nodes[ref] for ref in refs])
        poly_id = tags.get("name") or tags.get("link") or way.attrib.get("id") or str(len(polylines))
        polylines.append(Polyline(str(poly_id), pts, tags))
    pts = np.asarray(points, dtype=np.float64)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    extent = _background_extent(lo[0], hi[0], lo[1], hi[1], "up", "sind_official_utm_projector_osm_map")
    return polylines or None, extent


def _dji_transform_meta(scene_name: str, prefix: str, rec_path: Path, bg_path: Path) -> dict:
    row = _recording_row(rec_path)
    meta = {"raw_scene": scene_name, "raw_prefix": prefix}
    scale_match = re.search(r"1\s*pixel\s*=\s*([0-9.]+)\s*m", str(row.get("scale", "")))
    size = _image_size(bg_path) if bg_path.exists() else None
    if scale_match and size:
        meter_per_px = float(scale_match.group(1))
        width_px, height_px = size
        meta["background_extent"] = _background_extent(
            0.0,
            width_px * meter_per_px,
            0.0,
            height_px * meter_per_px,
            "down",
            "dji_recordingMeta_scale",
        )
        meta["image_scale"] = {"meter_per_px": meter_per_px}
    meta["recording_meta"] = {
        "scale": row.get("scale"),
        "laneMarkings": row.get("laneMarkings"),
        "speedLimit": row.get("speedLimit"),
    }
    return meta


def _citysim_lane_geometry(scene_dir: Path) -> list[Polyline] | None:
    lanes_path = next(iter(sorted(scene_dir.glob("*lanes_center.json"))), None)
    if not lanes_path:
        return None
    data = json.loads(lanes_path.read_text(encoding="utf-8"))
    polylines = []
    for item in data:
        pts = np.asarray(item.get("center_points", []), dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 2:
            polylines.append(Polyline(str(item.get("center_id", len(polylines))), pts, item))
    return polylines or None


def _citysim_background_path(scene_dir: Path) -> Path:
    exact = scene_dir / "background.png"
    if exact.exists():
        return exact
    matches = sorted(
        [
            p
            for p in scene_dir.glob("*.png")
            if "background" in p.name.lower() and "lane" not in p.name.lower() and "signal" not in p.name.lower()
        ]
    )
    return matches[0] if matches else exact


def _dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []


def _first_root(candidates: list[Path], predicate) -> Path:
    for path in candidates:
        if path.exists() and predicate(path):
            return path
    return candidates[0]


def _highd_root(dataset_root: Path) -> Path:
    return _first_root(
        [dataset_root, dataset_root / "highdrawdata", dataset_root / "HighD", dataset_root / "highD"],
        lambda p: bool(list(p.glob("*_tracks.csv"))),
    )


def _ind_root(dataset_root: Path) -> Path:
    return _first_root(
        [dataset_root, dataset_root / "data", dataset_root / "inD" / "data", dataset_root / "inD"],
        lambda p: bool(list(p.glob("*_tracks.csv"))),
    )


def _citysim_root(dataset_root: Path) -> Path:
    def looks_like_citysim(path: Path) -> bool:
        return any((path / scene).is_dir() for scene in CITYSIM_INTERSECTIONS)

    return _first_root(
        [dataset_root, dataset_root / "Citysim", dataset_root / "CitySim", dataset_root / "citysim"],
        looks_like_citysim,
    )


def _sind_root(dataset_root: Path) -> Path:
    return _first_root(
        [dataset_root, dataset_root / "sinD", dataset_root / "SIND", dataset_root / "SinD"],
        lambda p: bool(list(p.glob("*/*/Veh_smoothed_tracks.csv"))),
    )


def _ngsim_root(dataset_root: Path) -> Path:
    return _first_root(
        [dataset_root, dataset_root / "NGSIM", dataset_root / "ngsim"],
        lambda p: bool(list(p.glob("trajectories-*.csv"))),
    )


def _dji_root(dataset_root: Path) -> Path:
    return _first_root(
        [dataset_root, dataset_root / "DJI", dataset_root / "dji"],
        lambda p: bool(list(p.glob("*/*_tracks.csv")) or list(p.glob("*_tracks.csv"))),
    )


def _dji_scene_dirs(root: Path) -> list[Path]:
    if list(root.glob("*_tracks.csv")):
        return [root]
    return _dirs(root)


def load_highd(dataset_root: Path) -> Iterable[SceneRaw]:
    root = _highd_root(dataset_root)
    for tracks_path in sorted(root.glob("*_tracks.csv")):
        prefix = tracks_path.name.split("_")[0]
        meta_path = root / f"{prefix}_tracksMeta.csv"
        rec_path = root / f"{prefix}_recordingMeta.csv"
        bg_path = root / f"{prefix}_highway.png"
        meta = _meta_by_id(meta_path, "id")
        df = pd.read_csv(tracks_path, usecols=["frame", "id", "x", "y", "xVelocity", "laneId"])
        tracks = _group_tracks(df, "id", "frame", "x", "y", None, "xVelocity", "laneId", None, meta)
        for tr in tracks:
            tr.agent_type = str(tr.meta.get("class", "vehicle"))
        yield SceneRaw(
            scene_id=f"HighD_{prefix}",
            dataset="HighD",
            tracks=tracks,
            map_image_path=bg_path if bg_path.exists() else None,
            frame_rate=_recording_frame_rate(rec_path, 25.0),
            unit="meter",
            transform_meta=_highd_transform_meta(prefix, rec_path, bg_path),
        )


def load_ind(dataset_root: Path) -> Iterable[SceneRaw]:
    root = _ind_root(dataset_root)
    for tracks_path in sorted(root.glob("*_tracks.csv")):
        prefix = tracks_path.name.split("_")[0]
        meta_path = root / f"{prefix}_tracksMeta.csv"
        rec_path = root / f"{prefix}_recordingMeta.csv"
        bg_path = root / f"{prefix}_background.png"
        meta = _meta_by_id(meta_path, "trackId")
        df = pd.read_csv(tracks_path, usecols=["trackId", "frame", "xCenter", "yCenter", "heading"])
        tracks = _group_tracks(df, "trackId", "frame", "xCenter", "yCenter", "heading", None, None, None, meta)
        for tr in tracks:
            tr.agent_type = str(tr.meta.get("class", "vehicle"))
        rec = _read_csv(rec_path) if rec_path.exists() else pd.DataFrame()
        yield SceneRaw(
            scene_id=f"inD_{prefix}",
            dataset="inD",
            tracks=tracks,
            map_image_path=bg_path if bg_path.exists() else None,
            frame_rate=_recording_frame_rate(rec_path, 25.0),
            unit="meter",
            transform_meta=_ind_transform_meta(prefix, rec, bg_path),
        )


def load_citysim(dataset_root: Path) -> Iterable[SceneRaw]:
    root = _citysim_root(dataset_root)
    for scene_dir in _dirs(root):
        if scene_dir.name not in CITYSIM_INTERSECTIONS:
            continue
        traj_dir = scene_dir / "Trajectories"
        if not traj_dir.exists():
            traj_dir = scene_dir / "Trajectory"
        if not traj_dir.exists():
            continue
        tracks: list[TrackRaw] = []
        wanted = {"carId", "frameNum", "carCenterX", "carCenterY", "heading", "speed", "laneId"}
        for csv_path in sorted(traj_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, usecols=lambda c: c in wanted)
            if not {"carId", "frameNum", "carCenterX", "carCenterY"}.issubset(df.columns):
                continue
            part = _group_tracks(
                df,
                "carId",
                "frameNum",
                "carCenterX",
                "carCenterY",
                "heading" if "heading" in df.columns else None,
                "speed" if "speed" in df.columns else None,
                "laneId" if "laneId" in df.columns else None,
            )
            for tr in part:
                tr.track_id = f"{csv_path.stem}:{tr.track_id}"
                tr.agent_type = "vehicle"
            tracks.extend(part)
        bg_path = _citysim_background_path(scene_dir)
        size = _image_size(bg_path) if bg_path.exists() else None
        transform_meta = {"raw_scene": scene_dir.name, "coordinate_columns": ["carCenterX", "carCenterY"]}
        if size:
            width_px, height_px = size
            transform_meta["background_extent"] = _background_extent(
                0.0, width_px, 0.0, height_px, "down", "citysim_pixel_background"
            )
        yield SceneRaw(
            scene_id=f"CitySim_{scene_dir.name}",
            dataset="CitySim",
            tracks=tracks,
            map_image_path=bg_path if bg_path.exists() else None,
            lane_geometry=_citysim_lane_geometry(scene_dir),
            frame_rate=30.0,
            unit="pixel",
            transform_meta=transform_meta,
        )


def load_sind(dataset_root: Path) -> Iterable[SceneRaw]:
    root = _sind_root(dataset_root)
    for city_dir in _dirs(root):
        osm_path = next(iter(sorted(city_dir.glob("*.osm"))), None)
        lane_geometry, osm_extent = _sind_osm_geometry(osm_path)
        unregistered_map = next(iter(sorted(city_dir.glob("*.png"))), None)
        for scene_dir in _dirs(city_dir):
            veh_path = scene_dir / "Veh_smoothed_tracks.csv"
            if not veh_path.exists():
                continue
            keep = {"track_id", "frame_id", "agent_type", "x", "y", "heading_rad"}
            df = pd.read_csv(veh_path, usecols=lambda c: c in keep)
            tracks = _group_tracks(
                df,
                "track_id",
                "frame_id",
                "x",
                "y",
                "heading_rad" if "heading_rad" in df.columns else None,
                None,
                None,
                "agent_type" if "agent_type" in df.columns else None,
            )
            transform_meta = {"city": city_dir.name, "raw_scene": scene_dir.name}
            if osm_path:
                transform_meta["osm_path"] = str(osm_path)
            if unregistered_map:
                transform_meta["unregistered_map_image_path"] = str(unregistered_map)
            if osm_extent:
                transform_meta["plot_extent_hint"] = osm_extent
                transform_meta["background_extent"] = osm_extent
            transform_meta["map_image_status"] = "official_osm_map_background"
            yield SceneRaw(
                scene_id=f"sinD_{city_dir.name}_{scene_dir.name}",
                dataset="sinD",
                tracks=tracks,
                map_image_path=None,
                lane_geometry=lane_geometry,
                frame_rate=10.0,
                unit="meter",
                transform_meta=transform_meta,
            )


def load_ngsim(dataset_root: Path) -> Iterable[SceneRaw]:
    root = _ngsim_root(dataset_root)
    for csv_path in sorted(root.glob("trajectories-*.csv")):
        df = pd.read_csv(csv_path, usecols=["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y", "v_Vel", "Lane_ID"])
        df["plot_x_m"] = df["Local_Y"] * FT_TO_M
        df["plot_y_m"] = df["Local_X"] * FT_TO_M
        df["speed_mps"] = df["v_Vel"] * FT_TO_M
        tracks = _group_tracks(df, "Vehicle_ID", "Frame_ID", "plot_x_m", "plot_y_m", None, "speed_mps", "Lane_ID")
        for tr in tracks:
            tr.agent_type = "vehicle"
        yield SceneRaw(
            scene_id=f"NGSIM_{csv_path.stem.replace('trajectories-', '')}",
            dataset="NGSIM",
            tracks=tracks,
            frame_rate=10.0,
            unit="meter",
            transform_meta={
                "raw_file": csv_path.name,
                "background": "none",
                "plot_extent": "trajectory_quantile",
                "coordinate_transform": "plot_x=Local_Y*ft_to_m, plot_y=Local_X*ft_to_m",
                "source_columns": ["Local_X", "Local_Y"],
            },
        )


def load_dji(dataset_root: Path) -> Iterable[SceneRaw]:
    root = _dji_root(dataset_root)
    for scene_dir in _dji_scene_dirs(root):
        tracks_files = sorted(scene_dir.glob("*_tracks.csv"))
        if not tracks_files:
            continue
        for tracks_path in tracks_files:
            prefix = tracks_path.name.split("_")[0]
            meta_path = scene_dir / f"{prefix}_tracksMeta.csv"
            rec_path = scene_dir / f"{prefix}_recordingMeta.csv"
            bg_path = scene_dir / f"{prefix}_backgroundpics.jpg"
            meta = _meta_by_id(meta_path, "id")
            keep = {"frame", "id", "x", "y", "orientation", "laneId"}
            df = pd.read_csv(tracks_path, usecols=lambda c: c in keep)
            tracks = _group_tracks(df, "id", "frame", "x", "y", "orientation" if "orientation" in df.columns else None, None, "laneId", None, meta)
            for tr in tracks:
                tr.agent_type = str(tr.meta.get("class", "vehicle"))
            yield SceneRaw(
                scene_id=f"DJI_{scene_dir.name}_{prefix}",
                dataset="DJI",
                tracks=tracks,
                map_image_path=bg_path if bg_path.exists() else None,
                frame_rate=_recording_frame_rate(rec_path, 30.0),
                unit="meter",
                transform_meta=_dji_transform_meta(scene_dir.name, prefix, rec_path, bg_path),
            )


LOADERS = {
    "HighD": load_highd,
    "inD": load_ind,
    "CitySim": load_citysim,
    "sinD": load_sind,
    "NGSIM": load_ngsim,
    "DJI": load_dji,
}


def discover_scenes(dataset_root: str | Path, dataset: str | None = None) -> list[dict]:
    root = Path(dataset_root)
    rows: list[dict] = []
    selected = [dataset] if dataset else list(LOADERS)
    if "HighD" in selected:
        for p in sorted(_highd_root(root).glob("*_tracks.csv")):
            prefix = p.name.split("_")[0]
            rows.append({"dataset": "HighD", "scene_id": f"HighD_{prefix}"})
    if "inD" in selected:
        for p in sorted(_ind_root(root).glob("*_tracks.csv")):
            prefix = p.name.split("_")[0]
            rows.append({"dataset": "inD", "scene_id": f"inD_{prefix}"})
    if "CitySim" in selected:
        city_root = _citysim_root(root)
        for p in _dirs(city_root):
            if p.name not in CITYSIM_INTERSECTIONS:
                continue
            if (p / "Trajectories").exists() or (p / "Trajectory").exists():
                rows.append({"dataset": "CitySim", "scene_id": f"CitySim_{p.name}"})
    if "sinD" in selected:
        sind_root = _sind_root(root)
        for city in _dirs(sind_root):
            for scene in _dirs(city):
                if (scene / "Veh_smoothed_tracks.csv").exists():
                    rows.append({"dataset": "sinD", "scene_id": f"sinD_{city.name}_{scene.name}"})
    if "NGSIM" in selected:
        for p in sorted(_ngsim_root(root).glob("trajectories-*.csv")):
            rows.append({"dataset": "NGSIM", "scene_id": f"NGSIM_{p.stem.replace('trajectories-', '')}"})
    if "DJI" in selected:
        dji_root = _dji_root(root)
        for scene_dir in _dji_scene_dirs(dji_root):
            for p in sorted(scene_dir.glob("*_tracks.csv")):
                prefix = p.name.split("_")[0]
                rows.append({"dataset": "DJI", "scene_id": f"DJI_{scene_dir.name}_{prefix}"})
    return rows


def iter_scenes(dataset_root: str | Path, dataset: str | None = None, scene_id: str | None = None) -> Iterable[SceneRaw]:
    root = Path(dataset_root)
    selected = [dataset] if dataset else list(LOADERS)
    for name in selected:
        if name not in LOADERS:
            raise ValueError(f"unknown dataset {name}; choices={sorted(LOADERS)}")
        for scene in LOADERS[name](root):
            if scene_id and scene.scene_id != scene_id:
                continue
            yield scene
