from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from .axes import SUPPORTED_AXES


@dataclass(frozen=True)
class PostprocessDetail:
    smoothing_ms: int
    min_interval_ms: int
    min_delta: float
    deadband: float
    max_actions_per_second: float | None
    source: str | None = None
    reason: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        data = dict(self.extra)
        data["smoothing_ms"] = self.smoothing_ms
        data["min_interval_ms"] = self.min_interval_ms
        data["min_delta"] = self.min_delta
        data["deadband"] = self.deadband
        data["max_actions_per_second"] = self.max_actions_per_second
        if self.source:
            data.setdefault("source", self.source)
        if self.reason:
            data.setdefault("reason", self.reason)
        return data


def load_axis_scale_profile_data(path: str | Path) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    data = read_json_file(path)
    raw = data.get("axis_scales", data.get("axes", {}))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Axis scale profile has no axis mapping: {path}")
    scales: dict[str, float] = {}
    details: dict[str, dict[str, object]] = {}
    for axis, value in raw.items():
        axis_name = str(axis).lower()
        validate_axis(axis_name)
        if isinstance(value, dict):
            scale = optional_profile_float(value.get("scale"))
            detail = dict(value)
        else:
            scale = optional_profile_float(value)
            detail = {"scale": scale}
        if scale is None or scale < 0:
            raise RuntimeError(f"Invalid axis scale for {axis_name}: {value}")
        detail["scale"] = scale
        detail.setdefault("source", str(path))
        scales[axis_name] = scale
        details[axis_name] = detail
    return scales, details


def load_postprocess_profile(path: str | Path) -> dict[str, dict[str, object]]:
    data = read_json_file(path)
    raw_axes = data.get("axes")
    if not isinstance(raw_axes, dict):
        raise RuntimeError(f"Postprocess profile has no axes mapping: {path}")
    profile: dict[str, dict[str, object]] = {}
    for axis, value in raw_axes.items():
        axis_name = str(axis).lower()
        validate_axis(axis_name)
        if not isinstance(value, dict):
            raise RuntimeError(f"Invalid postprocess profile entry for {axis_name}: {path}")
        profile[axis_name] = normalize_postprocess_detail(value, source=str(path))
    return profile


def normalize_postprocess_detail(value: dict[str, object], *, source: str | None) -> dict[str, object]:
    detail = parse_postprocess_detail(value, source=source)
    return detail.to_json()


def parse_postprocess_detail(value: dict[str, object], *, source: str | None) -> PostprocessDetail:
    smoothing_ms = required_profile_int(value.get("smoothing_ms"), "smoothing_ms")
    min_interval_ms = required_profile_int(value.get("min_interval_ms"), "min_interval_ms")
    min_delta = profile_float_or_default(value.get("min_delta"), "min_delta", 0.0)
    deadband = profile_float_or_default(value.get("deadband"), "deadband", 0.0)
    max_actions_per_second = optional_profile_float_or_error(
        value.get("max_actions_per_second"),
        "max_actions_per_second",
    )
    if smoothing_ms < 0:
        raise RuntimeError(f"Invalid smoothing_ms: {value.get('smoothing_ms')}")
    if min_interval_ms <= 0:
        raise RuntimeError(f"Invalid min_interval_ms: {value.get('min_interval_ms')}")
    if min_delta < 0:
        raise RuntimeError(f"Invalid min_delta: {value.get('min_delta')}")
    if deadband < 0:
        raise RuntimeError(f"Invalid deadband: {value.get('deadband')}")
    if max_actions_per_second is not None and max_actions_per_second <= 0:
        raise RuntimeError(f"Invalid max_actions_per_second: {value.get('max_actions_per_second')}")
    extra = dict(value)
    return PostprocessDetail(
        smoothing_ms=smoothing_ms,
        min_interval_ms=min_interval_ms,
        min_delta=min_delta,
        deadband=deadband,
        max_actions_per_second=max_actions_per_second,
        source=source,
        reason=str(value["reason"]) if "reason" in value else None,
        extra=extra,
    )


def read_json_file(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return data


def required_profile_int(value: object, field_name: str) -> int:
    parsed = optional_profile_int(value)
    if parsed is None:
        raise RuntimeError(f"Invalid {field_name}: {value}")
    return parsed


def profile_float_or_default(value: object, field_name: str, default: float) -> float:
    if value is None or value == "":
        return default
    parsed = optional_profile_float(value)
    if parsed is None:
        raise RuntimeError(f"Invalid {field_name}: {value}")
    return parsed


def optional_profile_float_or_error(value: object, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    parsed = optional_profile_float(value)
    if parsed is None:
        raise RuntimeError(f"Invalid {field_name}: {value}")
    return parsed


def optional_profile_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def optional_profile_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def optional_float_value(value: object) -> float | None:
    return optional_profile_float(value)


def validate_axis(axis: str) -> None:
    if axis not in SUPPORTED_AXES:
        raise RuntimeError(f"Unsupported axis: {axis}")
