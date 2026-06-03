from __future__ import annotations

from dataclasses import dataclass

from .profiles import optional_float_value


@dataclass(frozen=True)
class PredictionQuality:
    axis: str
    status: str
    score: int
    tags: tuple[str, ...]
    actions_per_second: float | None
    duration_s: float | None
    range: float | None
    raw_range: float | None
    axis_scale: float | None
    axis_confidence: float | None

    def to_json(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "status": self.status,
            "score": self.score,
            "tags": list(self.tags),
            "actions_per_second": self.actions_per_second,
            "duration_s": self.duration_s,
            "range": self.range,
            "raw_range": self.raw_range,
            "axis_scale": self.axis_scale,
            "axis_confidence": self.axis_confidence,
        }


@dataclass(frozen=True)
class QualityGateDetail:
    mode: str
    threshold: float
    triggered: bool
    action: str
    quality_status: object
    quality_score: float | None

    def to_json(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "threshold": self.threshold,
            "triggered": self.triggered,
            "action": self.action,
            "quality_status": self.quality_status,
            "quality_score": self.quality_score,
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
    return score_prediction_quality(
        axis=axis,
        action_count=action_count,
        times=times,
        prediction_stats=prediction_stats,
        raw_prediction_stats=raw_prediction_stats,
        axis_scale_detail=axis_scale_detail,
    ).to_json()


def score_prediction_quality(
    *,
    axis: str,
    action_count: int,
    times: list[int],
    prediction_stats: dict[str, float | int | None],
    raw_prediction_stats: dict[str, float | int | None],
    axis_scale_detail: dict[str, object],
) -> PredictionQuality:
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
    return PredictionQuality(
        axis=axis,
        status=status,
        score=score,
        tags=tuple(tags),
        actions_per_second=actions_per_second,
        duration_s=duration_s,
        range=prediction_range,
        raw_range=raw_range,
        axis_scale=scale,
        axis_confidence=confidence,
    )


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
        "mean_score": float(sum(scores) / len(scores)) if scores else None,
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
    return build_quality_gate_detail(mode=mode, threshold=threshold, quality=quality).to_json()


def build_quality_gate_detail(*, mode: str, threshold: float, quality: dict[str, object]) -> QualityGateDetail:
    validate_quality_gate(mode, threshold)
    score = optional_float_value(quality.get("score"))
    triggered = score is None or score < float(threshold)
    action = mode if triggered else "keep"
    return QualityGateDetail(
        mode=mode,
        threshold=float(threshold),
        triggered=triggered,
        action=action,
        quality_status=quality.get("status"),
        quality_score=score,
    )


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


def prediction_duration_seconds(times: list[int]) -> float | None:
    if len(times) < 2:
        return None
    duration_ms = max(times) - min(times)
    if duration_ms <= 0:
        return None
    return duration_ms / 1000.0
