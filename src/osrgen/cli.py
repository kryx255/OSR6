from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .axes import AXIS_ORDER, SUPPORTED_AXES
from .batch_predict import BatchPredictConfig, run_batch_prediction
from .modeling import ModelPredictAllConfig, load_axis_scale_profile_data, load_postprocess_profile
from .modeling import predict_all_models
from .preset_validation import ModelPresetValidationConfig, validate_model_preset
from .video import inspect_video


DEFAULT_PRESET = Path("configs/presets/region_hybrid_experience_95.json")


def main(argv: list[str] | None = None) -> int:
    configure_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osrgen",
        description="Generate OSR6/SR6 funscripts from videos with the packaged OSRGen model.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="Show basic video metadata.")
    inspect_parser.add_argument("video", type=Path)
    inspect_parser.set_defaults(func=cmd_inspect)

    gui_parser = subparsers.add_parser("gui", help="Open the Windows GUI.")
    gui_parser.set_defaults(func=cmd_gui)

    model_parser = subparsers.add_parser("model", help="Model-based generation commands.")
    model_subparsers = model_parser.add_subparsers(dest="model_command")

    predict_all = model_subparsers.add_parser("predict-all", help="Generate all packaged OSR6 axes for one video.")
    predict_all.add_argument("input", type=Path, help="Video file, features.csv, or a directory containing features.csv.")
    add_predict_all_config_args(predict_all, output_default=Path("outputs/model_predict_all"))
    predict_all.set_defaults(func=cmd_model_predict_all)

    batch = model_subparsers.add_parser("batch-predict", help="Generate scripts for every video in a folder.")
    batch.add_argument("--input-dir", type=Path, required=True)
    batch.add_argument("--output", type=Path, default=Path("outputs/batch_predict"))
    batch.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument("--force", action="store_true", help="Regenerate even when prediction_all.json matches.")
    batch.add_argument("--workers", type=int, default=1, help="Number of videos to generate in parallel.")
    add_predict_runtime_overrides(batch)
    batch.set_defaults(func=cmd_model_batch_predict)

    validate = model_subparsers.add_parser("validate-preset", help="Check packaged checkpoint and profile files.")
    validate.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    validate.add_argument("--output", type=Path, default=None)
    validate.add_argument("--skip-device-check", action="store_true")
    validate.add_argument("--device", default=None, help="Validation inference device: auto, cpu, cuda, or cuda:0.")
    validate.set_defaults(func=cmd_model_validate_preset)

    return parser


def add_predict_all_config_args(parser: argparse.ArgumentParser, *, output_default: Path) -> None:
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--axes", default=None, help="Comma-separated axes, default comes from preset.")
    parser.add_argument("--output", type=Path, default=output_default)
    add_predict_runtime_overrides(parser)


def add_predict_runtime_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--analysis-fps", type=float, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    parser.add_argument("--scene-threshold", type=float, default=None)
    parser.add_argument("--roi", default=None, help="ROI as x,y,w,h normalized floats.")
    parser.add_argument("--axis-scales", default=None, help="Example: l0=1.25,l1=1.0,r2=0.8")
    parser.add_argument("--axis-scale-profile", type=Path, default=None)
    parser.add_argument("--postprocess-profile", type=Path, default=None)
    parser.add_argument("--quality-gate", choices=["warn", "neutralize", "omit"], default=None)
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--device", default=None, help="Inference device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--no-feature-cache", action="store_true", help="Disable the local extracted-feature cache.")
    parser.add_argument("--feature-cache-dir", type=Path, default=None, help="Override the extracted-feature cache directory.")


def cmd_inspect(args: argparse.Namespace) -> int:
    info = inspect_video(args.video)
    print(json.dumps(info.to_json(), indent=2, ensure_ascii=False))
    return 0


def cmd_gui(_args: argparse.Namespace) -> int:
    from .gui import main as gui_main

    return gui_main()


def cmd_model_predict_all(args: argparse.Namespace) -> int:
    config = build_predict_all_config(args, input_path=args.input, output=args.output)
    paths = predict_all_models(config)
    for path in paths:
        print(path)
    print(f"Generated {len(paths)} scripts in {Path(config.output) / Path(config.input_path).stem}")
    return 0


def cmd_model_batch_predict(args: argparse.Namespace) -> int:
    predict = build_predict_all_config(args, input_path=Path(""), output=args.output)
    results = run_batch_prediction(
        BatchPredictConfig(
            input_dir=str(args.input_dir),
            output=str(args.output),
            predict=predict,
            recursive=bool(args.recursive),
            resume=not bool(args.force),
            workers=max(1, int(args.workers)),
        )
    )
    ok = sum(1 for result in results if result.status in {"ok", "skipped_existing"})
    errors = sum(1 for result in results if result.status == "error")
    print(f"Processed {len(results)} videos: {ok} ok/skipped, {errors} errors.")
    print(Path(args.output) / "batch_summary.json")
    return 1 if errors else 0


def cmd_model_validate_preset(args: argparse.Namespace) -> int:
    report = validate_model_preset(
        ModelPresetValidationConfig(
            preset=str(args.preset),
            output=str(args.output) if args.output else None,
            check_device=not bool(args.skip_device_check),
            device=str(args.device) if args.device else None,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "ready" else 1


def build_predict_all_config(
    args: argparse.Namespace,
    *,
    input_path: Path,
    output: Path,
) -> ModelPredictAllConfig:
    preset = load_preset(args.preset)
    checkpoint_dir = preset_path(args, preset, "checkpoint_dir", Path("models/region_all_profile_95_e20"))
    axes = parse_axes(preset_value(args, preset, "axes", list(AXIS_ORDER)))
    analysis_fps = float(preset_value(args, preset, "analysis_fps", 8.0))
    max_width = int(preset_value(args, preset, "max_width", 360))
    scene_threshold = float(preset_value(args, preset, "scene_threshold", 35.0))
    roi = parse_float_list(preset_value(args, preset, "roi", [0.15, 0.15, 0.70, 0.70]), expected=4)
    quality_gate = str(preset_value(args, preset, "quality_gate", "warn"))
    quality_threshold = float(preset_value(args, preset, "quality_threshold", 50.0))
    device = str(preset_value(args, preset, "device", "auto"))
    feature_cache = not bool(getattr(args, "no_feature_cache", False))
    feature_cache_dir = getattr(args, "feature_cache_dir", None)
    postprocess_profile = preset_path(args, preset, "postprocess_profile", None)
    axis_scale_profile = preset_path(args, preset, "axis_scale_profile", None)

    axis_postprocess = load_postprocess_profile(postprocess_profile) if postprocess_profile else None
    axis_scales = None
    axis_scale_details = None
    if axis_scale_profile:
        axis_scales, axis_scale_details = load_axis_scale_profile_data(axis_scale_profile)
    manual_axis_scales = parse_axis_scales(preset_value(args, preset, "axis_scales", None))
    if manual_axis_scales:
        axis_scales = {**(axis_scales or {}), **manual_axis_scales}
        axis_scale_details = manual_axis_scale_details(axis_scale_details, manual_axis_scales)

    return ModelPredictAllConfig(
        input_path=str(input_path),
        checkpoint_dir=str(checkpoint_dir),
        output=str(output),
        preset=str(args.preset) if args.preset else None,
        axes=tuple(axes),
        analysis_fps=analysis_fps,
        max_width=max_width,
        scene_threshold=scene_threshold,
        roi=(roi[0], roi[1], roi[2], roi[3]),
        min_interval_ms=int(preset.get("min_interval_ms", 50)),
        smoothing_ms=int(preset.get("smoothing_ms", 250)),
        min_delta=float(preset.get("min_delta", 0.0)),
        deadband=float(preset.get("deadband", 0.0)),
        max_actions_per_second=optional_float(preset.get("max_actions_per_second")),
        axis_scales=axis_scales,
        axis_scale_details=axis_scale_details,
        axis_scale_profile=str(axis_scale_profile) if axis_scale_profile else None,
        axis_postprocess=axis_postprocess,
        postprocess_profile=str(postprocess_profile) if postprocess_profile else None,
        quality_gate=quality_gate,
        quality_threshold=quality_threshold,
        device=device,
        feature_cache=feature_cache,
        feature_cache_dir=str(feature_cache_dir) if feature_cache_dir else None,
    )


def load_preset(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Preset does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid preset JSON: {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Preset must contain a JSON object: {path}")
    return data


def preset_value(args: argparse.Namespace, preset: dict[str, object], key: str, default: object) -> object:
    arg_name = key.replace("-", "_")
    value = getattr(args, arg_name, None)
    if value is not None:
        return value
    return preset.get(key, default)


def preset_path(
    args: argparse.Namespace,
    preset: dict[str, object],
    key: str,
    default: Path | None,
) -> Path | None:
    value = preset_value(args, preset, key, default)
    if value is None or value == "":
        return None
    return Path(value)


def parse_axes(value: object) -> list[str]:
    if isinstance(value, str):
        axes = [axis.strip().lower() for axis in value.split(",") if axis.strip()]
    elif isinstance(value, (list, tuple)) and all(isinstance(axis, str) for axis in value):
        axes = [axis.strip().lower() for axis in value if axis.strip()]
    else:
        raise RuntimeError("Axes must be a comma-separated string or string list.")
    invalid = [axis for axis in axes if axis not in SUPPORTED_AXES]
    if invalid:
        raise RuntimeError(f"Unsupported axes: {', '.join(invalid)}")
    if not axes:
        raise RuntimeError("At least one axis is required.")
    return axes


def parse_float_list(value: object, *, expected: int | None = None) -> list[float]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise RuntimeError(f"Expected a comma-separated float list, got {value!r}")
    floats = [float(item) for item in items]
    if expected is not None and len(floats) != expected:
        raise RuntimeError(f"Expected {expected} float values, got {len(floats)}.")
    return floats


def parse_axis_scales(value: str | dict[str, object] | None) -> dict[str, float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        raw_items = value.items()
    elif isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        raw_items = []
        for part in parts:
            if "=" not in part:
                raise RuntimeError(f"Axis scale must use axis=value syntax: {part}")
            axis, scale = part.split("=", 1)
            raw_items.append((axis.strip(), scale.strip()))
    else:
        raise RuntimeError("Axis scales must be a dict or comma-separated axis=value string.")

    scales: dict[str, float] = {}
    for axis, raw_scale in raw_items:
        axis_name = str(axis).lower()
        if axis_name not in SUPPORTED_AXES:
            raise RuntimeError(f"Unsupported axis in scale list: {axis_name}")
        scale = float(raw_scale)
        if scale < 0:
            raise RuntimeError(f"Axis scale must be non-negative for {axis_name}.")
        scales[axis_name] = scale
    return scales


def manual_axis_scale_details(
    existing: dict[str, dict[str, object]] | None,
    manual_axis_scales: dict[str, float],
) -> dict[str, dict[str, object]]:
    details = {axis: dict(value) for axis, value in (existing or {}).items()}
    for axis, scale in manual_axis_scales.items():
        details[axis] = {
            **details.get(axis, {}),
            "scale": scale,
            "source": "manual",
            "reason": "configured by preset or CLI",
        }
    return details


def optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
