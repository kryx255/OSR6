from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable

from .axes import AXIS_ORDER, SUPPORTED_AXES
from .modeling import checkpoint_feature_columns, load_axis_scale_profile_data, load_checkpoint
from .modeling import load_postprocess_profile, require_torch
from .project import write_json


@dataclass(frozen=True)
class ModelPresetValidationConfig:
    preset: str
    output: str | None = None
    check_device: bool = True

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def validate_model_preset(config: ModelPresetValidationConfig) -> dict[str, object]:
    preset_path = Path(config.preset)
    preset = read_preset(preset_path)
    errors: list[str] = []
    warnings: list[str] = []
    resources: list[dict[str, object]] = []
    runtime: dict[str, object] = {}

    axes = preset_axes(preset.get("axes", AXIS_ORDER), errors)
    checkpoint_dir = preset_path_value(preset, "checkpoint_dir", errors)
    checkpoints: dict[str, dict[str, Any]] = {}

    if checkpoint_dir is None:
        errors.append("Preset must define checkpoint_dir.")
    elif not checkpoint_dir.is_dir():
        add_resource(resources, "checkpoint_dir", checkpoint_dir, "error", "directory does not exist")
        errors.append(f"Checkpoint directory does not exist: {checkpoint_dir}")
    else:
        add_resource(resources, "checkpoint_dir", checkpoint_dir, "ok", f"{len(axes)} selected axes")
        for axis in axes:
            checkpoint = inspect_checkpoint_resource(
                resources,
                errors,
                path=checkpoint_dir / f"{axis}.pt",
                kind=f"checkpoint:{axis}",
                expected_axis=axis,
            )
            if checkpoint is not None:
                checkpoints[axis] = checkpoint

    inspect_optional_json_profile(
        resources,
        errors,
        warnings,
        preset=preset,
        key="axis_scale_profile",
        kind="axis_scale_profile",
        axes=axes,
        loader=lambda path: load_axis_scale_profile_data(path)[0],
    )
    inspect_optional_json_profile(
        resources,
        errors,
        warnings,
        preset=preset,
        key="postprocess_profile",
        kind="postprocess_profile",
        axes=axes,
        loader=load_postprocess_profile,
    )

    if config.check_device:
        inspect_runtime(runtime, errors)

    report: dict[str, object] = {
        "config": config.to_json(),
        "preset": str(preset_path),
        "name": preset.get("name"),
        "status": "ready" if not errors else "error",
        "axes": axes,
        "checkpoint_count": len(checkpoints),
        "resources": resources,
        "runtime": runtime,
        "warnings": warnings,
        "errors": errors,
    }
    if config.output:
        write_json(Path(config.output) / "summary.json", report)
    return report


def read_preset(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read preset: {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Preset must contain a JSON object: {path}")
    return data


def preset_axes(value: object, errors: list[str]) -> list[str]:
    if isinstance(value, str):
        axes = [axis.strip().lower() for axis in value.split(",") if axis.strip()]
    elif isinstance(value, list) and all(isinstance(axis, str) for axis in value):
        axes = [axis.strip().lower() for axis in value if axis.strip()]
    else:
        errors.append("Preset axes must be a comma-separated string or string list.")
        return []
    invalid = [axis for axis in axes if axis not in SUPPORTED_AXES]
    if invalid:
        errors.append(f"Preset has unsupported axes: {', '.join(invalid)}")
    if len(set(axes)) != len(axes):
        errors.append("Preset axes must not contain duplicates.")
    if not axes:
        errors.append("Preset must select at least one axis.")
    return [axis for axis in axes if axis in SUPPORTED_AXES]


def preset_path_value(preset: dict[str, object], key: str, errors: list[str]) -> Path | None:
    value = preset.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        errors.append(f"Preset {key} must be a path string.")
        return None
    return Path(value)


def inspect_checkpoint_resource(
    resources: list[dict[str, object]],
    errors: list[str],
    *,
    path: Path,
    kind: str,
    expected_axis: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        add_resource(resources, kind, path, "error", "file does not exist")
        errors.append(f"Missing {kind}: {path}")
        return None
    try:
        checkpoint = load_checkpoint(path)
        axis = str(checkpoint.get("axis", "")).lower()
        columns = checkpoint_feature_columns(checkpoint)
        input_dim = int(checkpoint["model_config"]["input_dim"])
        normalization_dim = len(checkpoint["normalization"]["mean"])
        if axis != expected_axis:
            raise RuntimeError(f"expected axis {expected_axis}, checkpoint contains {axis or '<missing>'}")
        if input_dim != len(columns) or normalization_dim != len(columns):
            raise RuntimeError(
                f"dimension mismatch: model={input_dim}, normalization={normalization_dim}, columns={len(columns)}"
            )
        add_resource(resources, kind, path, "ok", f"axis={axis}, features={len(columns)}")
        return checkpoint
    except Exception as exc:
        add_resource(resources, kind, path, "error", str(exc))
        errors.append(f"Invalid {kind}: {path}: {exc}")
        return None


def inspect_optional_json_profile(
    resources: list[dict[str, object]],
    errors: list[str],
    warnings: list[str],
    *,
    preset: dict[str, object],
    key: str,
    kind: str,
    axes: list[str],
    loader: Callable[[Path], dict[str, object]],
) -> None:
    path = preset_path_value(preset, key, errors)
    if path is None:
        return
    if not path.is_file():
        add_resource(resources, kind, path, "error", "file does not exist")
        errors.append(f"Missing {kind}: {path}")
        return
    try:
        profile = loader(path)
        missing = [axis for axis in axes if axis not in profile]
        detail = f"{len(profile)} configured axes"
        if missing:
            detail += f"; fallback defaults for {','.join(missing)}"
            warnings.append(f"{kind} has no entries for: {', '.join(missing)}")
        add_resource(resources, kind, path, "ok", detail)
    except Exception as exc:
        add_resource(resources, kind, path, "error", str(exc))
        errors.append(f"Invalid {kind}: {path}: {exc}")


def inspect_runtime(runtime: dict[str, object], errors: list[str]) -> None:
    try:
        torch, _ = require_torch()
        runtime["torch_version"] = str(torch.__version__)
        runtime["cuda_available"] = bool(torch.cuda.is_available())
        runtime["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            runtime["cuda_devices"] = [
                str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        errors.append(f"Runtime device check failed: {exc}")


def add_resource(resources: list[dict[str, object]], kind: str, path: Path, status: str, detail: str) -> None:
    resources.append(
        {
            "kind": kind,
            "path": str(path),
            "status": status,
            "detail": detail,
        }
    )
