from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .features import MotionFeature
from .funscript import Action, Funscript, clamp_pos


@dataclass(frozen=True)
class GenerateConfig:
    mode: str = "pov"
    analysis_fps: float = 30.0
    smoothing_window_ms: int = 250
    detrend_window_ms: int = 1200
    min_interval_ms: int = 50
    amplitude: float = 100.0
    invert: bool = False

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def generate_l0(features: Iterable[MotionFeature], config: GenerateConfig) -> Funscript:
    items = list(features)
    if len(items) < 3:
        return Funscript.from_actions([])

    times = [item.time_ms for item in items]
    raw = []
    for item in items:
        if item.scene_cut:
            raw.append(0.0)
            continue
        # mean_dy is the strongest cheap signal for POV; radial flow catches scale-like motion.
        value = (item.mean_dy * 0.70) - (item.radial * 0.30)
        if config.invert:
            value = -value
        raw.append(value)

    velocity = zscore(raw)
    displacement = cumulative(velocity)
    fps = estimate_fps(times, fallback=config.analysis_fps)
    detrended = subtract_moving_average(displacement, window_samples=ms_to_samples(config.detrend_window_ms, fps))
    smoothed = moving_average(detrended, window_samples=ms_to_samples(config.smoothing_window_ms, fps))
    positions = normalize_positions(smoothed, amplitude=config.amplitude)
    return positions_to_funscript(times, positions, min_interval_ms=config.min_interval_ms)


def positions_to_funscript(
    times: list[int],
    positions: list[float],
    *,
    min_interval_ms: int = 50,
    min_delta: float = 0.0,
    deadband: float = 0.0,
    max_actions_per_second: float | None = None,
) -> Funscript:
    if len(times) != len(positions):
        raise ValueError("times and positions must have the same length")
    if not times:
        return Funscript.from_actions([])
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")
    if deadband < 0:
        raise ValueError("deadband must be non-negative")
    effective_interval_ms = effective_min_interval_ms(min_interval_ms, max_actions_per_second)
    processed_positions = apply_deadband(positions, deadband=deadband)
    action_indices = pick_keyframes(times, processed_positions, min_interval_ms=effective_interval_ms)
    actions = [Action(times[index], clamp_pos(processed_positions[index])) for index in action_indices]
    actions = filter_actions_by_min_delta(actions, min_delta=min_delta)
    return Funscript.from_actions(actions)


def effective_min_interval_ms(min_interval_ms: int, max_actions_per_second: float | None) -> int:
    effective = int(min_interval_ms)
    if max_actions_per_second is None:
        return effective
    max_rate = float(max_actions_per_second)
    if max_rate <= 0:
        raise ValueError("max_actions_per_second must be positive")
    return max(effective, int(math.ceil(1000.0 / max_rate)))


def apply_deadband(positions: list[float], *, deadband: float) -> list[float]:
    if deadband <= 0:
        return positions[:]
    return [50.0 if abs(float(position) - 50.0) < deadband else float(position) for position in positions]


def filter_actions_by_min_delta(actions: list[Action], *, min_delta: float) -> list[Action]:
    if min_delta <= 0 or len(actions) <= 2:
        return actions
    filtered = [actions[0]]
    for action in actions[1:-1]:
        if abs(action.pos - filtered[-1].pos) >= min_delta:
            filtered.append(action)
    if actions[-1].at != filtered[-1].at:
        filtered.append(actions[-1])
    return filtered


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-9:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def cumulative(values: list[float]) -> list[float]:
    out: list[float] = []
    total = 0.0
    for value in values:
        total += value
        out.append(total)
    return out


def moving_average(values: list[float], *, window_samples: int) -> list[float]:
    if window_samples <= 1 or len(values) <= 2:
        return values[:]
    window_samples = max(1, int(window_samples))
    radius = window_samples // 2
    out: list[float] = []
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        out.append((prefix[right] - prefix[left]) / (right - left))
    return out


def subtract_moving_average(values: list[float], *, window_samples: int) -> list[float]:
    avg = moving_average(values, window_samples=window_samples)
    return [value - baseline for value, baseline in zip(values, avg)]


def normalize_positions(values: list[float], *, amplitude: float = 100.0) -> list[float]:
    if not values:
        return []
    low = percentile(values, 5.0)
    high = percentile(values, 95.0)
    if abs(high - low) < 1e-9:
        return [50.0 for _ in values]
    amp = max(0.0, min(100.0, float(amplitude))) / 100.0
    out: list[float] = []
    for value in values:
        normalized = (value - low) / (high - low)
        pos = 100.0 - (normalized * 100.0)
        pos = max(0.0, min(100.0, pos))
        out.append(50.0 + ((pos - 50.0) * amp))
    return out


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    alpha = rank - low
    return ordered[low] * (1.0 - alpha) + ordered[high] * alpha


def pick_keyframes(times: list[int], positions: list[float], *, min_interval_ms: int) -> list[int]:
    if len(times) != len(positions) or not times:
        return []
    selected = [0]
    last_time = times[0]
    for index in range(1, len(positions) - 1):
        prev_delta = positions[index] - positions[index - 1]
        next_delta = positions[index + 1] - positions[index]
        is_extreme = (prev_delta >= 0 > next_delta) or (prev_delta <= 0 < next_delta)
        if not is_extreme:
            continue
        if times[index] - last_time < min_interval_ms:
            if abs(positions[index] - 50.0) > abs(positions[selected[-1]] - 50.0):
                selected[-1] = index
                last_time = times[index]
            continue
        selected.append(index)
        last_time = times[index]

    if selected[-1] != len(times) - 1:
        selected.append(len(times) - 1)

    if len(selected) <= 2:
        selected = fallback_keyframes(times, min_interval_ms=max(100, min_interval_ms))
    return selected


def fallback_keyframes(times: list[int], *, min_interval_ms: int) -> list[int]:
    selected: list[int] = []
    last_time = -10**12
    for index, time_ms in enumerate(times):
        if time_ms - last_time >= min_interval_ms:
            selected.append(index)
            last_time = time_ms
    if times and (not selected or selected[-1] != len(times) - 1):
        selected.append(len(times) - 1)
    return selected


def estimate_fps(times: list[int], *, fallback: float) -> float:
    if len(times) < 2:
        return fallback
    deltas = [right - left for left, right in zip(times, times[1:]) if right > left]
    if not deltas:
        return fallback
    avg_delta = sum(deltas) / len(deltas)
    if avg_delta <= 0:
        return fallback
    return 1000.0 / avg_delta


def ms_to_samples(ms: int, fps: float) -> int:
    return max(1, int(round((ms / 1000.0) * fps)))
