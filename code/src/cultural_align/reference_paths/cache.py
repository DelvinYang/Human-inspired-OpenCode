from __future__ import annotations

import pickle
from pathlib import Path
import sys
import types
from typing import Iterable

from .models import SceneRaw
from . import models
from .raw_loaders import iter_scenes


def cache_scene_path(cache_root: str | Path, dataset: str, scene_id: str) -> Path:
    return Path(cache_root) / dataset / f"{scene_id}.pkl"


def write_scene_cache(scene: SceneRaw, cache_root: str | Path) -> Path:
    path = cache_scene_path(cache_root, scene.dataset, scene.scene_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(scene, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def read_scene_cache(path: str | Path) -> SceneRaw:
    _install_legacy_pickle_aliases()
    with Path(path).open("rb") as f:
        return pickle.load(f)


def build_scene_cache(
    dataset_root: str | Path,
    cache_root: str | Path,
    dataset: str | None = None,
    scene_id: str | None = None,
    force: bool = False,
) -> list[Path]:
    written: list[Path] = []
    for scene in iter_scenes(dataset_root, dataset, scene_id):
        out = cache_scene_path(cache_root, scene.dataset, scene.scene_id)
        if out.exists() and not force:
            written.append(out)
            continue
        written.append(write_scene_cache(scene, cache_root))
    return written


def discover_cached_scenes(cache_root: str | Path, dataset: str | None = None) -> list[dict]:
    root = Path(cache_root)
    if not root.exists():
        return []
    paths: Iterable[Path]
    if dataset:
        paths = sorted((root / dataset).glob("*.pkl"))
    else:
        paths = sorted(root.glob("*/*.pkl"))
    rows = []
    for p in paths:
        rows.append({"dataset": p.parent.name, "scene_id": p.stem, "path": str(p)})
    return rows


def _install_legacy_pickle_aliases() -> None:
    sys.modules.setdefault("tools", types.ModuleType("tools"))
    sys.modules.setdefault("tools.path_library", types.ModuleType("tools.path_library"))
    sys.modules.setdefault("tools.path_library.models", models)
