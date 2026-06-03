from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrameSample:
    index: int
    source_frame: int
    time_ms: int
    gray: object


def require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for video analysis. Install dependencies with: "
            "python -m pip install -e ."
        ) from exc
    return cv2


def inspect_video(path: str | Path) -> VideoInfo:
    cv2 = require_cv2()
    src = Path(path)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoInfo(
            path=str(src),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
        )
    finally:
        cap.release()


def iter_gray_frames(
    path: str | Path,
    *,
    analysis_fps: float = 30.0,
    max_width: int = 1280,
    max_frames: int | None = None,
) -> Iterator[FrameSample]:
    cv2 = require_cv2()
    src = Path(path)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")

    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = analysis_fps
        sample_interval_ms = 1000.0 / max(1.0, analysis_fps)
        next_sample_ms = 0.0
        source_frame = 0
        sample_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            current_ms = source_frame * 1000.0 / source_fps
            source_frame += 1
            if current_ms + 0.001 < next_sample_ms:
                continue

            next_sample_ms += sample_interval_ms
            if max_width > 0 and frame.shape[1] > max_width:
                scale = max_width / float(frame.shape[1])
                new_height = max(1, int(round(frame.shape[0] * scale)))
                frame = cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield FrameSample(
                index=sample_index,
                source_frame=source_frame - 1,
                time_ms=int(round(current_ms)),
                gray=gray,
            )
            sample_index += 1
            if max_frames is not None and sample_index >= max_frames:
                break
    finally:
        cap.release()
