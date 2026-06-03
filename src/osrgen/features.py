from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
import csv
from pathlib import Path
from typing import Any, Iterable


BASE_MOTION_STAT_COLUMNS = (
    "mean_dx",
    "mean_dy",
    "mean_mag",
    "radial",
    "std_dx",
    "std_dy",
    "std_mag",
    "p90_mag",
    "mean_abs_dx",
    "mean_abs_dy",
    "vertical_ratio",
    "horizontal_ratio",
    "center_mag",
    "edge_mag",
    "center_edge_mag_delta",
    "divergence",
    "curl",
    "direction_x",
    "direction_y",
    "active_ratio",
    "active_center_x",
    "active_center_y",
    "active_spread_x",
    "active_spread_y",
    "active_mean_dx",
    "active_mean_dy",
    "active_mean_mag",
)


@dataclass(frozen=True)
class MotionFeature:
    time_ms: int
    mean_dx: float
    mean_dy: float
    mean_mag: float
    radial: float
    scene_cut: int
    roi_x: float
    roi_y: float
    roi_w: float
    roi_h: float
    std_dx: float = 0.0
    std_dy: float = 0.0
    std_mag: float = 0.0
    p90_mag: float = 0.0
    mean_abs_dx: float = 0.0
    mean_abs_dy: float = 0.0
    vertical_ratio: float = 0.0
    horizontal_ratio: float = 0.0
    center_mag: float = 0.0
    edge_mag: float = 0.0
    center_edge_mag_delta: float = 0.0
    divergence: float = 0.0
    curl: float = 0.0
    direction_x: float = 0.0
    direction_y: float = 0.0
    active_ratio: float = 0.0
    active_center_x: float = 0.0
    active_center_y: float = 0.0
    active_spread_x: float = 0.0
    active_spread_y: float = 0.0
    active_mean_dx: float = 0.0
    active_mean_dy: float = 0.0
    active_mean_mag: float = 0.0
    # Compatibility-only CSV fields. Raw-video inference does not extract pose/track signals.
    track_detected: int = 0
    track_confidence: float = 0.0
    track_x: float = 0.0
    track_y: float = 0.0
    track_dx: float = 0.0
    track_dy: float = 0.0
    track_speed: float = 0.0
    track_area: float = 0.0
    track_spread_x: float = 0.0
    track_spread_y: float = 0.0
    track_mean_dx: float = 0.0
    track_mean_dy: float = 0.0
    track_mean_mag: float = 0.0
    track_stability: float = 0.0
    pose_detected: int = 0
    pose_confidence: float = 0.0
    pose_box_x: float = 0.0
    pose_box_y: float = 0.0
    pose_box_w: float = 0.0
    pose_box_h: float = 0.0
    pose_box_area: float = 0.0
    pose_box_aspect: float = 0.0
    pose_center_x: float = 0.0
    pose_center_y: float = 0.0
    pose_center_dx: float = 0.0
    pose_center_dy: float = 0.0
    pose_shoulder_x: float = 0.0
    pose_shoulder_y: float = 0.0
    pose_hip_x: float = 0.0
    pose_hip_y: float = 0.0
    pose_hip_dx: float = 0.0
    pose_hip_dy: float = 0.0
    pose_torso_dx: float = 0.0
    pose_torso_dy: float = 0.0
    pose_torso_angle: float = 0.0
    pose_torso_length: float = 0.0
    pose_torso_angle_delta: float = 0.0
    pose_torso_length_delta: float = 0.0
    pose_shoulder_width: float = 0.0
    pose_hip_width: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, float | int]:
        row = {field_info.name: getattr(self, field_info.name) for field_info in motion_feature_fields()}
        for column, value in self.extra.items():
            if column not in row:
                row[column] = value
        return row


INT_FEATURE_COLUMNS = {"time_ms", "scene_cut", "track_detected", "pose_detected"}


def motion_feature_fields():
    return [field_info for field_info in fields(MotionFeature) if field_info.name != "extra"]


FEATURE_COLUMNS = [field_info.name for field_info in motion_feature_fields()]
MODEL_INPUT_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column != "time_ms" and not column.startswith(("track_", "pose_"))
]


def motion_feature_from_stats(
    time_ms: int,
    roi_px: tuple[int, int, int, int],
    frame_size: tuple[int, int],
    stats: dict[str, float] | None = None,
    *,
    scene_cut: int = 0,
) -> MotionFeature:
    x, y, w, h = roi_px
    width, height = frame_size
    values = {column: 0.0 for column in BASE_MOTION_STAT_COLUMNS}
    if stats:
        values.update({column: float(stats.get(column, 0.0)) for column in BASE_MOTION_STAT_COLUMNS})
    return MotionFeature(
        time_ms=time_ms,
        scene_cut=scene_cut,
        roi_x=x / max(1, width),
        roi_y=y / max(1, height),
        roi_w=w / max(1, width),
        roi_h=h / max(1, height),
        **values,
    )


def save_features_csv(features: Iterable[MotionFeature], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [feature.to_row() for feature in features]
    extra_columns = sorted({column for row in rows for column in row if column not in FEATURE_COLUMNS})
    columns = [*FEATURE_COLUMNS, *extra_columns]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, 0.0 if column in extra_columns else "") for column in columns})


def load_features_csv(path: str | Path) -> list[MotionFeature]:
    rows: list[MotionFeature] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(MotionFeature(**parse_feature_row(row)))
    return rows


def parse_feature_row(row: dict[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    known = {field_info.name for field_info in motion_feature_fields()}
    for field_info in motion_feature_fields():
        raw = row.get(field_info.name, "")
        if raw == "" or raw is None:
            if field_info.default is not MISSING:
                values[field_info.name] = field_info.default
                continue
            raise RuntimeError(f"Missing required feature column: {field_info.name}")
        if field_info.name in INT_FEATURE_COLUMNS:
            values[field_info.name] = int(float(raw))
        else:
            values[field_info.name] = float(raw)
    values["extra"] = parse_extra_feature_columns(row, known)
    return values


def parse_extra_feature_columns(row: dict[str, str], known: set[str]) -> dict[str, float]:
    extra: dict[str, float] = {}
    for column, raw in row.items():
        if column is None or column in known:
            continue
        if raw == "" or raw is None:
            extra[column] = 0.0
            continue
        try:
            extra[column] = float(raw)
        except ValueError as exc:
            raise RuntimeError(f"Invalid numeric feature value for {column}: {raw}") from exc
    return extra
