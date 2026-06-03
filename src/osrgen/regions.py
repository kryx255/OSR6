from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .features import MotionFeature, motion_feature_from_stats
from .flow import flow_statistics, roi_to_pixels
from .video import iter_gray_frames, require_cv2


BASE_REGION_NAME = "__base__"
DEFAULT_REGION_SIGNALS = [
    "mean_dx",
    "mean_dy",
    "radial",
    "mean_mag",
    "active_mean_dx",
    "active_mean_dy",
    "active_mean_mag",
    "active_center_x",
    "active_center_y",
    "active_center_x_delta",
    "active_center_y_delta",
    "center_edge_mag_delta",
]
DEFAULT_REGION_FEATURE_SIGNALS = tuple(DEFAULT_REGION_SIGNALS)


@dataclass(frozen=True)
class CandidateRegion:
    name: str
    roi: tuple[float, float, float, float]

    def to_json(self) -> dict[str, object]:
        return {"name": self.name, "roi": list(self.roi)}


DEFAULT_REGIONS = [
    CandidateRegion("full", (0.00, 0.00, 1.00, 1.00)),
    CandidateRegion("center_70", (0.15, 0.15, 0.70, 0.70)),
    CandidateRegion("center_50", (0.25, 0.20, 0.50, 0.60)),
    CandidateRegion("center_wide", (0.05, 0.15, 0.90, 0.70)),
    CandidateRegion("upper_center", (0.20, 0.08, 0.60, 0.45)),
    CandidateRegion("lower_center", (0.20, 0.45, 0.60, 0.45)),
    CandidateRegion("lower_narrow", (0.30, 0.50, 0.40, 0.40)),
    CandidateRegion("top_band", (0.10, 0.05, 0.80, 0.35)),
    CandidateRegion("bottom_band", (0.10, 0.60, 0.80, 0.35)),
    CandidateRegion("left_center", (0.05, 0.20, 0.45, 0.60)),
    CandidateRegion("right_center", (0.50, 0.20, 0.45, 0.60)),
    CandidateRegion("mid_left", (0.05, 0.35, 0.45, 0.45)),
    CandidateRegion("mid_right", (0.50, 0.35, 0.45, 0.45)),
]


def region_feature_column(region_name: str, signal: str) -> str:
    return f"region_{region_name}_{signal}"


def region_feature_columns(
    regions: Iterable[CandidateRegion],
    signals: Iterable[str] = DEFAULT_REGION_FEATURE_SIGNALS,
) -> list[str]:
    return [region_feature_column(region.name, signal) for region in regions for signal in signals]


def extract_motion_features_with_region_features(
    video_path: str | Path,
    *,
    roi: tuple[float, float, float, float],
    regions: Iterable[CandidateRegion],
    signals: Iterable[str] = DEFAULT_REGION_FEATURE_SIGNALS,
    analysis_fps: float = 8.0,
    max_width: int = 360,
    max_frames: int | None = None,
    scene_threshold: float = 35.0,
) -> list[MotionFeature]:
    selected_regions = list(regions)
    features_by_region = extract_region_motion_features(
        video_path,
        regions=[CandidateRegion(BASE_REGION_NAME, roi), *selected_regions],
        analysis_fps=analysis_fps,
        max_width=max_width,
        max_frames=max_frames,
        scene_threshold=scene_threshold,
    )
    base_features = features_by_region.pop(BASE_REGION_NAME, [])
    return add_region_features(base_features, features_by_region, signals)


def add_region_features(
    features: Iterable[MotionFeature],
    features_by_region: dict[str, list[MotionFeature]],
    signals: Iterable[str] = DEFAULT_REGION_FEATURE_SIGNALS,
) -> list[MotionFeature]:
    selected_signals = tuple(signals)
    validate_region_signals(selected_signals)
    columns = [
        region_feature_column(region_name, signal)
        for region_name in features_by_region
        for signal in selected_signals
    ]
    extras_by_time: dict[int, dict[str, float]] = {}

    for region_name, region_features in features_by_region.items():
        for signal in selected_signals:
            column = region_feature_column(region_name, signal)
            values = signal_values(region_features, signal)
            for feature, value in zip(region_features, values):
                extras_by_time.setdefault(feature.time_ms, {})[column] = float(value)

    enriched: list[MotionFeature] = []
    for feature in features:
        extra = dict(feature.extra)
        for column in columns:
            extra.setdefault(column, 0.0)
        extra.update(extras_by_time.get(feature.time_ms, {}))
        enriched.append(replace(feature, extra=extra))
    return enriched


def extract_region_motion_features(
    video_path: str | Path,
    *,
    regions: Iterable[CandidateRegion],
    analysis_fps: float = 8.0,
    max_width: int = 360,
    max_frames: int | None = None,
    scene_threshold: float = 35.0,
) -> dict[str, list[MotionFeature]]:
    cv2 = require_cv2()

    selected_regions = list(regions)
    features_by_region: dict[str, list[MotionFeature]] = {region.name: [] for region in selected_regions}
    previous_gray = None
    for sample in iter_gray_frames(
        video_path,
        analysis_fps=analysis_fps,
        max_width=max_width,
        max_frames=max_frames,
    ):
        gray = sample.gray
        height, width = gray.shape[:2]
        if previous_gray is None:
            previous_gray = gray
            continue
        if previous_gray.shape != gray.shape:
            previous_gray = cv2.resize(previous_gray, (gray.shape[1], gray.shape[0]))

        frame_diff = float(np.mean(np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32))))
        scene_cut = scene_threshold > 0 and frame_diff >= scene_threshold
        if scene_cut:
            for region in selected_regions:
                x, y, w, h = roi_to_pixels(region.roi, width, height)
                features_by_region[region.name].append(
                    motion_feature_from_stats(sample.time_ms, (x, y, w, h), (width, height), scene_cut=1)
                )
            previous_gray = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        dx_all = flow[..., 0]
        dy_all = flow[..., 1]
        mag_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
        for region in selected_regions:
            x, y, w, h = roi_to_pixels(region.roi, width, height)
            dx = dx_all[y : y + h, x : x + w]
            dy = dy_all[y : y + h, x : x + w]
            mag = mag_all[y : y + h, x : x + w]
            stats = flow_statistics(dx, dy, mag)
            features_by_region[region.name].append(
                motion_feature_from_stats(sample.time_ms, (x, y, w, h), (width, height), stats, scene_cut=0)
            )
        previous_gray = gray
    return features_by_region


def select_regions(names: tuple[str, ...] | None) -> list[CandidateRegion]:
    if names is None:
        return list(DEFAULT_REGIONS)
    by_name = {region.name: region for region in DEFAULT_REGIONS}
    selected: list[CandidateRegion] = []
    invalid: list[str] = []
    for name in names:
        region = by_name.get(name)
        if region is None:
            invalid.append(name)
        else:
            selected.append(region)
    if invalid:
        raise RuntimeError(f"Unknown regions: {', '.join(invalid)}")
    if not selected:
        raise RuntimeError("At least one region is required.")
    return selected


def validate_region_signals(signals: Iterable[str]) -> None:
    valid = set(DEFAULT_REGION_SIGNALS)
    invalid = [signal for signal in signals if signal not in valid]
    if invalid:
        raise RuntimeError(f"Unknown region signals: {', '.join(invalid)}")


def signal_values(features: list[MotionFeature], signal: str) -> list[float]:
    if signal == "active_center_x_delta":
        return delta_values([feature.active_center_x for feature in features])
    if signal == "active_center_y_delta":
        return delta_values([feature.active_center_y for feature in features])
    if not features or not hasattr(features[0], signal):
        raise RuntimeError(f"Unknown region signal: {signal}")
    return [float(getattr(feature, signal)) for feature in features]


def delta_values(values: list[float]) -> list[float]:
    if not values:
        return []
    result = [0.0]
    result.extend(current - previous for previous, current in zip(values, values[1:]))
    return result
