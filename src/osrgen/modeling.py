from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .axes import AXIS_ORDER, AXIS_SCRIPT_SUFFIXES, SUPPORTED_AXES
from .features import MODEL_INPUT_COLUMNS, MotionFeature, load_features_csv, save_features_csv
from .flow import extract_motion_features
from .funscript import Funscript
from .generator import estimate_fps, moving_average, ms_to_samples, positions_to_funscript
from .project import file_fingerprint, write_json
from .regions import DEFAULT_REGIONS, DEFAULT_REGION_FEATURE_SIGNALS
from .regions import extract_motion_features_with_region_features, select_regions


MODEL_FEATURE_COLUMNS = MODEL_INPUT_COLUMNS


@dataclass(frozen=True)
class ModelPredictAllConfig:
    input_path: str
    checkpoint_dir: str
    output: str
    preset: str | None = None
    axes: tuple[str, ...] = AXIS_ORDER
    analysis_fps: float = 8.0
    max_width: int = 360
    scene_threshold: float = 35.0
    roi: tuple[float, float, float, float] = (0.15, 0.15, 0.70, 0.70)
    region_features: bool = False
    region_regions: tuple[str, ...] | None = None
    region_signals: tuple[str, ...] = tuple(DEFAULT_REGION_FEATURE_SIGNALS)
    min_interval_ms: int = 50
    smoothing_ms: int = 250
    min_delta: float = 0.0
    deadband: float = 0.0
    max_actions_per_second: float | None = None
    axis_scales: dict[str, float] | None = None
    axis_scale_details: dict[str, dict[str, object]] | None = None
    axis_scale_profile: str | None = None
    axis_postprocess: dict[str, dict[str, object]] | None = None
    postprocess_profile: str | None = None
    quality_gate: str = "warn"
    quality_threshold: float = 50.0

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["axes"] = list(self.axes)
        data["region_regions"] = list(self.region_regions) if self.region_regions is not None else None
        data["region_signals"] = list(self.region_signals)
        return data


def predict_all_models(config: ModelPredictAllConfig) -> list[Path]:
    selected_axes = tuple(axis.lower() for axis in config.axes)
    for axis in selected_axes:
        validate_axis(axis)
    validate_quality_gate(config.quality_gate, config.quality_threshold)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoints: list[dict[str, Any]] = []
    missing: list[str] = []
    for axis in selected_axes:
        path = checkpoint_dir / f"{axis}.pt"
        if not path.exists():
            missing.append(axis)
            continue
        checkpoint = load_checkpoint(path)
        if str(checkpoint.get("axis", "")).lower() != axis:
            raise RuntimeError(f"Checkpoint {path} must target {axis}, got {checkpoint.get('axis')}.")
        checkpoints.append(checkpoint)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {checkpoint_dir} for axes: {', '.join(selected_axes)}")

    all_columns = [column for checkpoint in checkpoints for column in checkpoint_feature_columns(checkpoint)]
    unsupported = unsupported_online_feature_columns(all_columns)
    if unsupported and not feature_input_path(Path(config.input_path)):
        raise RuntimeError(
            "This runtime build can extract optical-flow and region features only. "
            f"Unsupported checkpoint columns for raw-video prediction: {', '.join(sorted(unsupported))}"
        )

    input_path = Path(config.input_path)
    output_root = Path(config.output)
    output_name = input_path.parent.name if input_path.name.lower() == "features.csv" else input_path.stem
    out_dir = output_root / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = config_with_inferred_region_request(config, all_columns)
    features = load_or_extract_features(
        runtime_config,
        input_path,
        out_dir,
        region_features=runtime_config.region_features or checkpoint_needs_region(all_columns),
    )
    if not features:
        raise RuntimeError(f"No motion features were produced for input: {input_path}")

    times = [feature.time_ms for feature in features]
    generated: list[dict[str, object]] = []
    script_paths: list[Path] = []
    prediction_dir = out_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint in checkpoints:
        axis = str(checkpoint["axis"]).lower()
        columns = checkpoint_feature_columns(checkpoint)
        matrix = feature_matrix_from_motion_features(features, columns)
        postprocess_detail = axis_postprocess_detail_for(config, axis)
        smoothing_ms = int(postprocess_detail["smoothing_ms"])
        min_interval_ms = int(postprocess_detail["min_interval_ms"])
        min_delta = float(postprocess_detail["min_delta"])
        deadband = float(postprocess_detail["deadband"])
        max_actions_per_second = optional_profile_float(postprocess_detail.get("max_actions_per_second"))

        positions = smooth_prediction(times, predict_positions(checkpoint, matrix), smoothing_ms)
        raw_stats = position_stats(times, positions)
        scale = axis_scale_for(config, axis)
        scale_detail = axis_scale_detail_for(config, axis, scale)
        if scale != 1.0:
            positions = scale_axis_positions(positions, scale)

        write_prediction_csv(times, positions, prediction_dir / f"{axis}.csv")
        script = positions_to_funscript(
            times,
            positions.tolist(),
            min_interval_ms=min_interval_ms,
            min_delta=min_delta,
            deadband=deadband,
            max_actions_per_second=max_actions_per_second,
        )
        final_stats = position_stats(times, positions)
        quality = prediction_quality_for_axis(
            axis=axis,
            action_count=len(script.actions),
            times=times,
            prediction_stats=final_stats,
            raw_prediction_stats=raw_stats,
            axis_scale_detail=scale_detail,
        )
        script_path = out_dir / axis_script_filename(output_name, axis)
        pre_gate_action_count = len(script.actions)
        gate_detail = quality_gate_detail(
            mode=config.quality_gate,
            threshold=config.quality_threshold,
            quality=quality,
        )
        gate_action = str(gate_detail["action"])
        if gate_action == "neutralize":
            script = neutral_funscript(times)
        if gate_action == "omit":
            script_path.unlink(missing_ok=True)
        else:
            script.save(script_path)
            script_paths.append(script_path)

        generated.append(
            {
                "axis": axis,
                "script_path": script_path.name if gate_action != "omit" else None,
                "planned_script_path": script_path.name,
                "action_count": len(script.actions) if gate_action != "omit" else 0,
                "pre_gate_action_count": pre_gate_action_count,
                "axis_scale": scale,
                "axis_scale_detail": scale_detail,
                "postprocess_detail": postprocess_detail,
                "prediction_stats": final_stats,
                "raw_prediction_stats": raw_stats,
                "quality": quality,
                "quality_gate": gate_detail,
                "feature_columns": columns,
            }
        )

    write_json(
        out_dir / "prediction_all.json",
        {
            "config": config.to_json(),
            "input": str(input_path),
            "input_fingerprint": file_fingerprint(input_path) if input_path.is_file() else None,
            "checkpoint_dir": str(checkpoint_dir),
            "preset": config.preset,
            "feature_count": len(features),
            "axis_scale_profile": config.axis_scale_profile,
            "postprocess_profile": config.postprocess_profile,
            "quality_summary": summarize_prediction_quality(generated),
            "quality_gate_summary": summarize_quality_gates(generated),
            "generated": generated,
            "missing_axes": missing,
        },
    )
    return script_paths


def load_or_extract_features(
    config: ModelPredictAllConfig,
    input_path: Path,
    out_dir: Path,
    *,
    region_features: bool,
) -> list[MotionFeature]:
    if input_path.is_dir():
        features_path = input_path / "features.csv"
        if not features_path.exists():
            raise RuntimeError(f"Feature directory has no features.csv: {input_path}")
        return load_features_csv(features_path)
    if input_path.suffix.lower() == ".csv":
        return load_features_csv(input_path)
    if region_features:
        features = extract_motion_features_with_region_features(
            input_path,
            roi=config.roi,
            regions=select_regions(config.region_regions),
            signals=config.region_signals,
            analysis_fps=config.analysis_fps,
            max_width=config.max_width,
            scene_threshold=config.scene_threshold,
        )
    else:
        features = extract_motion_features(
            input_path,
            analysis_fps=config.analysis_fps,
            max_width=config.max_width,
            roi=config.roi,
            scene_threshold=config.scene_threshold,
        )
    save_features_csv(features, out_dir / "features.csv")
    return features


def require_torch():
    try:
        import torch  # type: ignore
        from torch import nn  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for model prediction. Run install.bat or use: uv sync --extra model"
        ) from exc
    return torch, nn


def create_model(input_dim: int, channels: int, layers: int, kernel_size: int, dropout: float):
    torch, nn = require_torch()

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            padding = dilation * (kernel_size - 1) // 2
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.act = nn.ReLU()
            self.drop = nn.Dropout(dropout)
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        def forward(self, x):  # type: ignore[no-untyped-def]
            residual = x if self.proj is None else self.proj(x)
            return self.drop(self.act(self.conv(x))) + residual

    class TinyTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = []
            in_channels = input_dim
            for index in range(layers):
                blocks.append(ConvBlock(in_channels, channels, dilation=2**index))
                in_channels = channels
            self.net = nn.Sequential(*blocks)
            self.head = nn.Conv1d(channels, 1, kernel_size=1)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return torch.sigmoid(self.head(self.net(x))).squeeze(1)

    return TinyTCN()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    torch, _ = require_torch()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model_config = checkpoint["model_config"]
    model = create_model(
        input_dim=int(model_config["input_dim"]),
        channels=int(model_config["channels"]),
        layers=int(model_config["layers"]),
        kernel_size=int(model_config["kernel_size"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    checkpoint["model"] = model
    checkpoint["normalization"]["mean"] = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32)
    checkpoint["normalization"]["std"] = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32)
    return checkpoint


def predict_positions(checkpoint: dict[str, Any], features: np.ndarray) -> np.ndarray:
    torch, _ = require_torch()
    model = checkpoint["model"]
    mean = checkpoint["normalization"]["mean"]
    std = checkpoint["normalization"]["std"]
    if features.shape[1] != len(mean):
        raise RuntimeError(
            f"Feature dimension mismatch: checkpoint expects {len(mean)} columns "
            f"({', '.join(checkpoint_feature_columns(checkpoint))}), got {features.shape[1]}."
        )
    x = normalize_features(features, mean, std)
    with torch.no_grad():
        tensor = torch.as_tensor(x.T[None, :, :], dtype=torch.float32)
        pred = model(tensor).detach().cpu().numpy()[0]
    return np.clip(pred * 100.0, 0.0, 100.0)


def smooth_prediction(times: list[int], positions: np.ndarray, smoothing_ms: int) -> np.ndarray:
    if smoothing_ms <= 0 or len(positions) < 3:
        return positions
    fps = estimate_fps(times, fallback=30.0)
    samples = ms_to_samples(smoothing_ms, fps)
    smoothed = moving_average([float(value) for value in positions], window_samples=samples)
    return np.asarray(smoothed, dtype=np.float32)


def feature_matrix_from_motion_features(
    features: list[MotionFeature],
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    columns = feature_columns or MODEL_FEATURE_COLUMNS
    rows = [[feature_value(feature, column) for column in columns] for feature in features]
    return np.asarray(rows, dtype=np.float32)


def checkpoint_feature_columns(checkpoint: dict[str, Any]) -> list[str]:
    columns = checkpoint.get("feature_columns")
    if not columns:
        return MODEL_FEATURE_COLUMNS
    return [str(column) for column in columns]


def config_with_inferred_region_request(
    config: ModelPredictAllConfig,
    feature_columns: Iterable[str],
) -> ModelPredictAllConfig:
    if config.region_features or not checkpoint_needs_region(feature_columns):
        return config
    request = infer_region_request_from_feature_columns(feature_columns)
    if request is None:
        return config
    regions, signals = request
    return replace(config, region_features=True, region_regions=tuple(regions), region_signals=tuple(signals))


def infer_region_request_from_feature_columns(feature_columns: Iterable[str]) -> tuple[list[str], list[str]] | None:
    region_names: set[str] = set()
    signal_names: set[str] = set()
    region_order = [region.name for region in DEFAULT_REGIONS]
    signal_order = list(DEFAULT_REGION_FEATURE_SIGNALS)
    for column in feature_columns:
        if not column.startswith("region_"):
            continue
        for region_name in region_order:
            prefix = f"region_{region_name}_"
            if not column.startswith(prefix):
                continue
            signal = column[len(prefix) :]
            if signal in signal_order:
                region_names.add(region_name)
                signal_names.add(signal)
            break
    if not region_names or not signal_names:
        return None
    return (
        [name for name in region_order if name in region_names],
        [name for name in signal_order if name in signal_names],
    )


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


def axis_postprocess_detail_for(config: ModelPredictAllConfig, axis: str) -> dict[str, object]:
    if config.axis_postprocess and axis in config.axis_postprocess:
        return normalize_postprocess_detail(config.axis_postprocess[axis], source=config.postprocess_profile)
    return {
        "smoothing_ms": int(config.smoothing_ms),
        "min_interval_ms": int(config.min_interval_ms),
        "min_delta": float(config.min_delta),
        "deadband": float(config.deadband),
        "max_actions_per_second": config.max_actions_per_second,
        "source": "config",
        "reason": "global predict-all settings",
    }


def normalize_postprocess_detail(value: dict[str, object], *, source: str | None) -> dict[str, object]:
    smoothing_ms = optional_profile_int(value.get("smoothing_ms"))
    min_interval_ms = optional_profile_int(value.get("min_interval_ms"))
    min_delta = optional_profile_float(value.get("min_delta")) or 0.0
    deadband = optional_profile_float(value.get("deadband")) or 0.0
    max_actions_per_second = optional_profile_float(value.get("max_actions_per_second"))
    if smoothing_ms is None or smoothing_ms < 0:
        raise RuntimeError(f"Invalid smoothing_ms: {value.get('smoothing_ms')}")
    if min_interval_ms is None or min_interval_ms <= 0:
        raise RuntimeError(f"Invalid min_interval_ms: {value.get('min_interval_ms')}")
    if min_delta < 0:
        raise RuntimeError(f"Invalid min_delta: {value.get('min_delta')}")
    if deadband < 0:
        raise RuntimeError(f"Invalid deadband: {value.get('deadband')}")
    if max_actions_per_second is not None and max_actions_per_second <= 0:
        raise RuntimeError(f"Invalid max_actions_per_second: {value.get('max_actions_per_second')}")
    detail = dict(value)
    detail["smoothing_ms"] = smoothing_ms
    detail["min_interval_ms"] = min_interval_ms
    detail["min_delta"] = min_delta
    detail["deadband"] = deadband
    detail["max_actions_per_second"] = max_actions_per_second
    if source:
        detail.setdefault("source", source)
    return detail


def axis_scale_for(config: ModelPredictAllConfig, axis: str) -> float:
    if not config.axis_scales:
        return 1.0
    scale = float(config.axis_scales.get(axis, 1.0))
    if scale < 0:
        raise RuntimeError(f"Axis scale must be non-negative for {axis}: {scale}")
    return scale


def axis_scale_detail_for(config: ModelPredictAllConfig, axis: str, scale: float) -> dict[str, object]:
    details = dict(config.axis_scale_details.get(axis, {})) if config.axis_scale_details else {}
    details["scale"] = scale
    details.setdefault("source", config.axis_scale_profile or "config")
    return details


def scale_axis_positions(positions: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(50.0 + ((positions.astype(np.float32) - 50.0) * float(scale)), 0.0, 100.0)


def position_stats(times: list[int], positions: np.ndarray) -> dict[str, float | int | None]:
    if len(positions) == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "range": None,
            "duration_s": prediction_duration_seconds(times),
        }
    return {
        "count": int(len(positions)),
        "min": float(np.min(positions)),
        "max": float(np.max(positions)),
        "mean": float(np.mean(positions)),
        "std": float(np.std(positions)),
        "range": float(np.max(positions) - np.min(positions)),
        "duration_s": prediction_duration_seconds(times),
    }


def prediction_quality_for_axis(
    *,
    axis: str,
    action_count: int,
    times: list[int],
    prediction_stats: dict[str, float | int | None],
    raw_prediction_stats: dict[str, float | int | None],
    axis_scale_detail: dict[str, object],
) -> dict[str, object]:
    duration_s = prediction_duration_seconds(times)
    actions_per_second = float(action_count / duration_s) if duration_s and duration_s > 0 else None
    prediction_range = optional_float_value(prediction_stats.get("range"))
    raw_range = optional_float_value(raw_prediction_stats.get("range"))
    scale = optional_float_value(axis_scale_detail.get("scale"))
    confidence = optional_float_value(axis_scale_detail.get("confidence"))

    score = 100
    tags: list[str] = []
    if scale is not None and scale < 0.95:
        tags.append("intentionally_scaled_down")
        if scale < 0.50:
            tags.append("weak_axis_scaled")
            score -= 10
    if confidence is not None:
        if confidence < 0.20:
            tags.append("low_axis_confidence")
            score -= 25
        elif confidence < 0.40:
            tags.append("moderate_axis_confidence")
            score -= 10
    if prediction_range is not None:
        if prediction_range < 5.0:
            tags.append("near_flat_output")
            score -= 15
        elif prediction_range < 10.0:
            tags.append("low_amplitude_output")
            score -= 8
    if raw_range is not None and prediction_range is not None and scale is not None:
        if raw_range >= 15.0 and prediction_range < raw_range * 0.60:
            tags.append("amplitude_limited_by_scale")
    if actions_per_second is not None:
        if actions_per_second > 5.0:
            tags.append("very_high_action_density")
            score -= 20
        elif actions_per_second > 4.0:
            tags.append("high_action_density")
            score -= 10
        if prediction_range is not None and actions_per_second > 3.0 and prediction_range < 8.0:
            tags.append("many_small_actions")
            score -= 20

    score = max(0, min(100, score))
    if score < 50:
        status = "weak"
    elif score < 70 or "many_small_actions" in tags:
        status = "review"
    else:
        status = "ok"
    return {
        "axis": axis,
        "status": status,
        "score": score,
        "tags": tags,
        "actions_per_second": actions_per_second,
        "duration_s": duration_s,
        "range": prediction_range,
        "raw_range": raw_range,
        "axis_scale": scale,
        "axis_confidence": confidence,
    }


def summarize_prediction_quality(generated: list[dict[str, object]]) -> dict[str, object]:
    by_status: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    review_axes: list[str] = []
    scores: list[float] = []
    for item in generated:
        quality = item.get("quality")
        if not isinstance(quality, dict):
            continue
        status = str(quality.get("status", "unknown"))
        axis = str(item.get("axis", ""))
        by_status[status] = by_status.get(status, 0) + 1
        if status != "ok" and axis:
            review_axes.append(axis)
        score = optional_float_value(quality.get("score"))
        if score is not None:
            scores.append(score)
        tags = quality.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                tag_text = str(tag)
                tag_counts[tag_text] = tag_counts.get(tag_text, 0) + 1
    return {
        "axis_count": len(generated),
        "mean_score": float(np.mean(scores)) if scores else None,
        "by_status": by_status,
        "tag_counts": dict(sorted(tag_counts.items())),
        "review_axes": review_axes,
    }


def validate_quality_gate(mode: str, threshold: float) -> None:
    if mode not in {"warn", "neutralize", "omit"}:
        raise RuntimeError(f"Unsupported quality gate mode: {mode}")
    if not 0 <= float(threshold) <= 100:
        raise RuntimeError(f"Quality threshold must be between 0 and 100: {threshold}")


def quality_gate_detail(*, mode: str, threshold: float, quality: dict[str, object]) -> dict[str, object]:
    validate_quality_gate(mode, threshold)
    score = optional_float_value(quality.get("score"))
    triggered = score is None or score < float(threshold)
    action = mode if triggered else "keep"
    return {
        "mode": mode,
        "threshold": float(threshold),
        "triggered": triggered,
        "action": action,
        "quality_status": quality.get("status"),
        "quality_score": score,
    }


def summarize_quality_gates(generated: list[dict[str, object]]) -> dict[str, object]:
    by_action: dict[str, int] = {}
    triggered_axes: list[str] = []
    for item in generated:
        detail = item.get("quality_gate")
        if not isinstance(detail, dict):
            continue
        action = str(detail.get("action", "unknown"))
        by_action[action] = by_action.get(action, 0) + 1
        if detail.get("triggered"):
            triggered_axes.append(str(item.get("axis", "")))
    return {
        "axis_count": len(generated),
        "triggered_count": len(triggered_axes),
        "triggered_axes": triggered_axes,
        "by_action": by_action,
    }


def neutral_funscript(times: list[int]) -> Funscript:
    if not times:
        return Funscript.from_actions([])
    if times[0] == times[-1]:
        return Funscript.from_actions([(times[0], 50)])
    return Funscript.from_actions([(times[0], 50), (times[-1], 50)])


def prediction_duration_seconds(times: list[int]) -> float | None:
    if len(times) < 2:
        return None
    duration_ms = max(times) - min(times)
    if duration_ms <= 0:
        return None
    return duration_ms / 1000.0


def unsupported_online_feature_columns(columns: Iterable[str]) -> set[str]:
    unsupported: set[str] = set()
    for column in columns:
        if column.startswith(("pose_", "track_", "semantic_pca_", "weak_track_")):
            unsupported.add(column)
    return unsupported


def checkpoint_needs_region(feature_columns: Iterable[str]) -> bool:
    return any(column.startswith("region_") for column in feature_columns)


def feature_input_path(path: Path) -> bool:
    return path.name.lower() == "features.csv" or (path.is_dir() and (path / "features.csv").exists())


def feature_value(feature: MotionFeature, column: str) -> float:
    if hasattr(feature, column):
        return float(getattr(feature, column))
    return float(feature.extra.get(column, 0.0))


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.where(np.abs(std) < 1e-9, 1.0, std)
    return (features.astype(np.float32) - mean) / safe_std


def axis_script_filename(output_name: str, axis: str) -> str:
    suffix = AXIS_SCRIPT_SUFFIXES.get(axis)
    if suffix is None:
        raise RuntimeError(f"Unsupported axis {axis}.")
    if not suffix:
        return f"{output_name}.funscript"
    return f"{output_name}.{suffix}.funscript"


def write_prediction_csv(times: list[int], positions: np.ndarray, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_ms", "pos"])
        writer.writeheader()
        for time_ms, pos in zip(times, positions):
            writer.writerow({"time_ms": time_ms, "pos": round(float(pos), 6)})


def read_json_file(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return data


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
