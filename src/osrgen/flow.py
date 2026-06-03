from __future__ import annotations

from pathlib import Path

from .features import MotionFeature
from .video import iter_gray_frames, require_cv2


def extract_motion_features(
    video_path: str | Path,
    *,
    analysis_fps: float = 30.0,
    max_width: int = 1280,
    roi: tuple[float, float, float, float] = (0.15, 0.15, 0.70, 0.70),
    scene_threshold: float = 35.0,
    max_frames: int | None = None,
) -> list[MotionFeature]:
    cv2 = require_cv2()
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "NumPy is required for optical-flow analysis. Install dependencies with: "
            "python -m pip install -e ."
        ) from exc

    previous_gray = None
    previous_time_ms = 0
    features: list[MotionFeature] = []

    for sample in iter_gray_frames(
        video_path,
        analysis_fps=analysis_fps,
        max_width=max_width,
        max_frames=max_frames,
    ):
        gray = sample.gray
        height, width = gray.shape[:2]
        x, y, w, h = roi_to_pixels(roi, width, height)
        current_roi = gray[y : y + h, x : x + w]

        if previous_gray is None:
            previous_gray = current_roi
            previous_time_ms = sample.time_ms
            continue

        if previous_gray.shape != current_roi.shape:
            previous_gray = cv2.resize(previous_gray, (current_roi.shape[1], current_roi.shape[0]))

        frame_diff = float(np.mean(np.abs(current_roi.astype(np.float32) - previous_gray.astype(np.float32))))
        if scene_threshold > 0 and frame_diff >= scene_threshold:
            features.append(
                MotionFeature(
                    time_ms=sample.time_ms,
                    mean_dx=0.0,
                    mean_dy=0.0,
                    mean_mag=0.0,
                    radial=0.0,
                    scene_cut=1,
                    roi_x=x / max(1, width),
                    roi_y=y / max(1, height),
                    roi_w=w / max(1, width),
                    roi_h=h / max(1, height),
                )
            )
            previous_gray = current_roi
            previous_time_ms = sample.time_ms
            continue

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_roi,
            None,
            0.5,
            3,
            21,
            3,
            5,
            1.2,
            0,
        )
        dx = flow[..., 0]
        dy = flow[..., 1]
        mag = np.sqrt(dx * dx + dy * dy)
        stats = flow_statistics(dx, dy, mag)
        features.append(
            MotionFeature(
                time_ms=sample.time_ms,
                mean_dx=float(stats["mean_dx"]),
                mean_dy=float(stats["mean_dy"]),
                mean_mag=float(stats["mean_mag"]),
                radial=float(stats["radial"]),
                scene_cut=0,
                roi_x=x / max(1, width),
                roi_y=y / max(1, height),
                roi_w=w / max(1, width),
                roi_h=h / max(1, height),
                std_dx=float(stats["std_dx"]),
                std_dy=float(stats["std_dy"]),
                std_mag=float(stats["std_mag"]),
                p90_mag=float(stats["p90_mag"]),
                mean_abs_dx=float(stats["mean_abs_dx"]),
                mean_abs_dy=float(stats["mean_abs_dy"]),
                vertical_ratio=float(stats["vertical_ratio"]),
                horizontal_ratio=float(stats["horizontal_ratio"]),
                center_mag=float(stats["center_mag"]),
                edge_mag=float(stats["edge_mag"]),
                center_edge_mag_delta=float(stats["center_edge_mag_delta"]),
                divergence=float(stats["divergence"]),
                curl=float(stats["curl"]),
                direction_x=float(stats["direction_x"]),
                direction_y=float(stats["direction_y"]),
                active_ratio=float(stats["active_ratio"]),
                active_center_x=float(stats["active_center_x"]),
                active_center_y=float(stats["active_center_y"]),
                active_spread_x=float(stats["active_spread_x"]),
                active_spread_y=float(stats["active_spread_y"]),
                active_mean_dx=float(stats["active_mean_dx"]),
                active_mean_dy=float(stats["active_mean_dy"]),
                active_mean_mag=float(stats["active_mean_mag"]),
            )
        )

        previous_gray = current_roi
        previous_time_ms = sample.time_ms

    _ = previous_time_ms
    return features


def flow_statistics(dx, dy, mag) -> dict[str, float]:
    import numpy as np  # type: ignore

    eps = 1e-6
    mean_abs_dx = float(np.mean(np.abs(dx)))
    mean_abs_dy = float(np.mean(np.abs(dy)))
    total_abs = mean_abs_dx + mean_abs_dy + eps
    center_mag, edge_mag = center_edge_magnitude(mag)
    divergence, curl = flow_derivatives(dx, dy)
    active = active_motion_stats(dx, dy, mag)
    mean_mag = float(np.mean(mag))
    direction_x = float(np.mean(dx / (mag + eps)))
    direction_y = float(np.mean(dy / (mag + eps)))
    return {
        "mean_dx": float(np.mean(dx)),
        "mean_dy": float(np.mean(dy)),
        "mean_mag": mean_mag,
        "radial": float(radial_flow(dx, dy)),
        "std_dx": float(np.std(dx)),
        "std_dy": float(np.std(dy)),
        "std_mag": float(np.std(mag)),
        "p90_mag": float(np.percentile(mag, 90)),
        "mean_abs_dx": mean_abs_dx,
        "mean_abs_dy": mean_abs_dy,
        "vertical_ratio": mean_abs_dy / total_abs,
        "horizontal_ratio": mean_abs_dx / total_abs,
        "center_mag": center_mag,
        "edge_mag": edge_mag,
        "center_edge_mag_delta": center_mag - edge_mag,
        "divergence": divergence,
        "curl": curl,
        "direction_x": direction_x,
        "direction_y": direction_y,
        **active,
    }


def center_edge_magnitude(mag) -> tuple[float, float]:
    import numpy as np  # type: ignore

    height, width = mag.shape[:2]
    y0 = max(0, int(height * 0.25))
    y1 = min(height, int(height * 0.75))
    x0 = max(0, int(width * 0.25))
    x1 = min(width, int(width * 0.75))
    if y1 <= y0 or x1 <= x0:
        mean = float(np.mean(mag))
        return mean, mean
    center = mag[y0:y1, x0:x1]
    edge_mask = np.ones(mag.shape, dtype=bool)
    edge_mask[y0:y1, x0:x1] = False
    edge = mag[edge_mask]
    center_mag = float(np.mean(center)) if center.size else float(np.mean(mag))
    edge_mag = float(np.mean(edge)) if edge.size else float(np.mean(mag))
    return center_mag, edge_mag


def flow_derivatives(dx, dy) -> tuple[float, float]:
    import numpy as np  # type: ignore

    if dx.shape[0] < 2 or dx.shape[1] < 2:
        return 0.0, 0.0
    ddx_dy, ddx_dx = np.gradient(dx)
    ddy_dy, ddy_dx = np.gradient(dy)
    divergence = float(np.mean(ddx_dx + ddy_dy))
    curl = float(np.mean(ddy_dx - ddx_dy))
    return divergence, curl


def active_motion_stats(dx, dy, mag) -> dict[str, float]:
    import numpy as np  # type: ignore

    height, width = mag.shape[:2]
    if height == 0 or width == 0:
        return empty_active_motion_stats()

    threshold = max(float(np.percentile(mag, 80)), float(np.mean(mag) + np.std(mag)))
    mask = mag >= threshold
    active_count = int(np.count_nonzero(mask))
    total = int(mask.size)
    if active_count <= 0 or total <= 0:
        return empty_active_motion_stats()

    weights = mag[mask].astype(np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-9:
        return empty_active_motion_stats()

    yy, xx = np.mgrid[0:height, 0:width]
    active_x = xx[mask].astype(np.float64)
    active_y = yy[mask].astype(np.float64)
    center_x = float(np.sum(active_x * weights) / weight_sum)
    center_y = float(np.sum(active_y * weights) / weight_sum)
    norm_x = center_x / max(1.0, width - 1.0)
    norm_y = center_y / max(1.0, height - 1.0)
    spread_x = float(np.sqrt(np.sum(((active_x - center_x) ** 2) * weights) / weight_sum) / max(1.0, width))
    spread_y = float(np.sqrt(np.sum(((active_y - center_y) ** 2) * weights) / weight_sum) / max(1.0, height))
    return {
        "active_ratio": active_count / total,
        "active_center_x": norm_x,
        "active_center_y": norm_y,
        "active_spread_x": spread_x,
        "active_spread_y": spread_y,
        "active_mean_dx": float(np.mean(dx[mask])),
        "active_mean_dy": float(np.mean(dy[mask])),
        "active_mean_mag": float(np.mean(mag[mask])),
    }


def empty_active_motion_stats() -> dict[str, float]:
    return {
        "active_ratio": 0.0,
        "active_center_x": 0.0,
        "active_center_y": 0.0,
        "active_spread_x": 0.0,
        "active_spread_y": 0.0,
        "active_mean_dx": 0.0,
        "active_mean_dy": 0.0,
        "active_mean_mag": 0.0,
    }


def roi_to_pixels(roi: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    rx, ry, rw, rh = roi
    x = max(0, min(width - 1, int(round(rx * width))))
    y = max(0, min(height - 1, int(round(ry * height))))
    w = max(1, min(width - x, int(round(rw * width))))
    h = max(1, min(height - y, int(round(rh * height))))
    return x, y, w, h


def radial_flow(dx, dy) -> float:
    import numpy as np  # type: ignore

    height, width = dx.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    vx = xx - cx
    vy = yy - cy
    norm = np.sqrt(vx * vx + vy * vy)
    norm[norm == 0] = 1.0
    ux = vx / norm
    uy = vy / norm
    return float(np.mean(dx * ux + dy * uy))
