from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from .project import write_json


FEATURE_CACHE_VERSION = 1
FEATURE_CACHE_DIR_ENV = "OSRGEN_FEATURE_CACHE_DIR"
FEATURE_CACHE_DISABLE_ENV = "OSRGEN_DISABLE_FEATURE_CACHE"


@dataclass(frozen=True)
class FeatureCacheEntry:
    key: str
    directory: Path
    features: Path
    metadata: Path


def default_feature_cache_dir() -> Path | None:
    if os.environ.get(FEATURE_CACHE_DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    override = os.environ.get(FEATURE_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "OSRGen" / "feature_cache"
    try:
        return Path.home() / ".cache" / "osrgen" / "feature_cache"
    except RuntimeError:
        return None


def feature_cache_entry(request: dict[str, object], cache_dir: str | Path | None = None) -> FeatureCacheEntry | None:
    root = Path(cache_dir).expanduser() if cache_dir is not None else default_feature_cache_dir()
    if root is None:
        return None
    key = feature_cache_key(request)
    directory = root / key[:2] / key
    return FeatureCacheEntry(
        key=key,
        directory=directory,
        features=directory / "features.csv",
        metadata=directory / "metadata.json",
    )


def feature_cache_key(request: dict[str, object]) -> str:
    payload = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_cached_features_path(
    request: dict[str, object],
    cache_dir: str | Path | None = None,
) -> Path | None:
    entry = feature_cache_entry(request, cache_dir)
    if entry is None or not entry.features.is_file() or not entry.metadata.is_file():
        return None
    try:
        metadata = json.loads(entry.metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("request") != request:
        return None
    return entry.features


def store_cached_features(
    request: dict[str, object],
    features_path: str | Path,
    cache_dir: str | Path | None = None,
) -> Path | None:
    entry = feature_cache_entry(request, cache_dir)
    if entry is None:
        return None
    source = Path(features_path)
    if not source.is_file():
        return None
    entry.directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, entry.features)
    write_json(
        entry.metadata,
        {
            "version": FEATURE_CACHE_VERSION,
            "key": entry.key,
            "request": request,
        },
    )
    return entry.features


def feature_cache_request(
    *,
    input_path: str | Path,
    input_fingerprint: dict[str, Any],
    analysis_fps: float,
    max_width: int,
    scene_threshold: float,
    roi: tuple[float, float, float, float],
    region_features: bool,
    region_regions: tuple[str, ...],
    region_signals: tuple[str, ...],
) -> dict[str, object]:
    return {
        "version": FEATURE_CACHE_VERSION,
        "input_path": str(Path(input_path).expanduser().resolve()).lower(),
        "input_fingerprint": input_fingerprint,
        "analysis_fps": float(analysis_fps),
        "max_width": int(max_width),
        "scene_threshold": float(scene_threshold),
        "roi": [float(value) for value in roi],
        "region_features": bool(region_features),
        "region_regions": list(region_regions),
        "region_signals": list(region_signals),
    }
