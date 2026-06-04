from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import html
import json
from pathlib import Path
from typing import Callable

from .modeling import ModelPredictAllConfig, predict_all_models
from .project import file_fingerprint, write_json


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}


@dataclass(frozen=True)
class BatchPredictConfig:
    input_dir: str
    output: str
    predict: ModelPredictAllConfig
    recursive: bool = False
    resume: bool = True
    workers: int = 1

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BatchPredictResult:
    sample_id: str
    status: str
    video: str
    output_dir: str
    prediction_summary: str | None
    qc_report: str | None
    axis_count: int
    mean_score: float | None
    review_axes: list[str]
    gated_axes: list[str]
    omitted_axes: list[str]
    warnings: list[str]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def run_batch_prediction(
    config: BatchPredictConfig,
    *,
    predictor: Callable[[ModelPredictAllConfig], list[Path]] = predict_all_models,
) -> list[BatchPredictResult]:
    input_root = Path(config.input_dir)
    if not input_root.exists():
        raise RuntimeError(f"Batch input directory does not exist: {input_root}")
    if not input_root.is_dir():
        raise RuntimeError(f"Batch input path is not a directory: {input_root}")

    videos = scan_batch_videos(input_root, recursive=config.recursive)
    output_root = Path(config.output)
    output_root.mkdir(parents=True, exist_ok=True)
    worker_count = normalized_worker_count(config.workers, len(videos))
    if worker_count <= 1:
        results = [
            run_batch_prediction_for_video(
                video,
                input_root=input_root,
                output_root=output_root,
                config=config,
                predictor=predictor,
            )
            for video in videos
        ]
    else:
        ordered: list[BatchPredictResult | None] = [None for _ in videos]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    run_batch_prediction_for_video,
                    video,
                    input_root=input_root,
                    output_root=output_root,
                    config=config,
                    predictor=predictor,
                ): index
                for index, video in enumerate(videos)
            }
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        results = [result for result in ordered if result is not None]

    write_json(
        output_root / "batch_summary.json",
        {
            "config": config.to_json(),
            "video_count": len(videos),
            "result_count": len(results),
            "aggregate": aggregate_batch_results(results),
            "results": [result.to_json() for result in results],
        },
    )
    write_batch_summary_csv(results, output_root / "batch_summary.csv")
    return results


def normalized_worker_count(requested: int, item_count: int) -> int:
    if item_count <= 1:
        return 1
    return max(1, min(int(requested), item_count))


def run_batch_prediction_for_video(
    video: Path,
    *,
    input_root: Path,
    output_root: Path,
    config: BatchPredictConfig,
    predictor: Callable[[ModelPredictAllConfig], list[Path]],
) -> BatchPredictResult:
    relative = video.relative_to(input_root)
    sample_id = relative.with_suffix("").as_posix()
    sample_output_root = output_root / relative.parent
    prediction_dir = sample_output_root / video.stem
    prediction_summary = prediction_dir / "prediction_all.json"
    qc_output = output_root / "_qc" / relative.parent / video.stem
    try:
        prediction_config = replace(
            config.predict,
            input_path=str(video),
            output=str(sample_output_root),
        )
        regeneration_warnings: list[str] = []
        if config.resume and prediction_summary.exists():
            reusable, regeneration_warnings = prediction_summary_matches_request(
                prediction_summary,
                prediction_config,
                video,
            )
            if reusable:
                return batch_result_from_prediction(
                    sample_id=sample_id,
                    video=video,
                    prediction_dir=prediction_dir,
                    prediction_summary=prediction_summary,
                    qc_output=qc_output,
                    status="skipped_existing",
                    warnings=["existing prediction_all.json reused"],
                )
        predictor(prediction_config)
        return batch_result_from_prediction(
            sample_id=sample_id,
            video=video,
            prediction_dir=prediction_dir,
            prediction_summary=prediction_summary,
            qc_output=qc_output,
            status="ok",
            warnings=regeneration_warnings,
        )
    except Exception as exc:
        return BatchPredictResult(
            sample_id=sample_id,
            status="error",
            video=str(video),
            output_dir=str(prediction_dir),
            prediction_summary=None,
            qc_report=None,
            axis_count=0,
            mean_score=None,
            review_axes=[],
            gated_axes=[],
            omitted_axes=[],
            warnings=[str(exc)],
        )


def scan_batch_videos(input_root: Path, *, recursive: bool) -> list[Path]:
    candidates = input_root.rglob("*") if recursive else input_root.iterdir()
    videos = [path for path in candidates if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos, key=lambda path: path.relative_to(input_root).as_posix().lower())


def prediction_summary_matches_request(
    path: Path,
    predict: ModelPredictAllConfig,
    video: Path,
) -> tuple[bool, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    existing_config = data.get("config")
    if not isinstance(existing_config, dict):
        return False, ["existing prediction config missing; regenerated"]
    if json_key(existing_config) != json_key(predict.to_json()):
        return False, ["prediction config changed; regenerated"]
    if data.get("input_fingerprint") != file_fingerprint(video):
        return False, ["input video changed; regenerated"]
    if not prediction_outputs_exist(data, path.parent):
        return False, ["prediction output files missing; regenerated"]
    return True, []


def prediction_outputs_exist(data: dict[str, object], output_dir: Path) -> bool:
    generated = data.get("generated")
    if not isinstance(generated, list):
        return False
    for item in generated:
        if not isinstance(item, dict):
            return False
        script_path = item.get("script_path")
        if script_path is None:
            gate = item.get("quality_gate")
            if not isinstance(gate, dict) or gate.get("action") != "omit":
                return False
            continue
        if not isinstance(script_path, str) or not (output_dir / script_path).is_file():
            return False
    return True


def json_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def batch_result_from_prediction(
    *,
    sample_id: str,
    video: Path,
    prediction_dir: Path,
    prediction_summary: Path,
    qc_output: Path,
    status: str,
    warnings: list[str],
) -> BatchPredictResult:
    if not prediction_summary.exists():
        raise RuntimeError(f"Prediction did not write summary: {prediction_summary}")
    data = json.loads(prediction_summary.read_text(encoding="utf-8"))
    quality_summary = data.get("quality_summary", {})
    gate_summary = data.get("quality_gate_summary", {})
    generated = data.get("generated", [])
    if not isinstance(quality_summary, dict):
        quality_summary = {}
    if not isinstance(gate_summary, dict):
        gate_summary = {}
    if not isinstance(generated, list):
        generated = []
    write_lightweight_qc_report(data, qc_output)
    omitted_axes = [
        str(item.get("axis", ""))
        for item in generated
        if isinstance(item, dict)
        and isinstance(item.get("quality_gate"), dict)
        and item["quality_gate"].get("action") == "omit"
    ]
    return BatchPredictResult(
        sample_id=sample_id,
        status=status,
        video=str(video),
        output_dir=str(prediction_dir),
        prediction_summary=str(prediction_summary),
        qc_report=str(qc_output / "index.html"),
        axis_count=int(quality_summary.get("axis_count", len(generated))),
        mean_score=optional_float(quality_summary.get("mean_score")),
        review_axes=string_list(quality_summary.get("review_axes")),
        gated_axes=string_list(gate_summary.get("triggered_axes")),
        omitted_axes=omitted_axes,
        warnings=warnings,
    )


def aggregate_batch_results(results: list[BatchPredictResult]) -> dict[str, object]:
    scores = [result.mean_score for result in results if result.mean_score is not None]
    return {
        "ok_count": sum(1 for result in results if result.status == "ok"),
        "skipped_existing_count": sum(1 for result in results if result.status == "skipped_existing"),
        "error_count": sum(1 for result in results if result.status == "error"),
        "mean_score": sum(scores) / len(scores) if scores else None,
        "review_video_count": sum(1 for result in results if result.review_axes),
        "gated_video_count": sum(1 for result in results if result.gated_axes),
        "omitted_video_count": sum(1 for result in results if result.omitted_axes),
    }


def write_lightweight_qc_report(prediction: dict[str, object], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    generated = prediction.get("generated", [])
    if not isinstance(generated, list):
        generated = []
    rows = []
    for item in generated:
        if not isinstance(item, dict):
            continue
        quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
        gate = item.get("quality_gate") if isinstance(item.get("quality_gate"), dict) else {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('axis', '')))}</td>"
            f"<td>{html.escape(str(item.get('script_path') or 'omitted'))}</td>"
            f"<td>{html.escape(str(quality.get('status', 'unknown')))}</td>"
            f"<td>{html.escape(str(quality.get('score', '')))}</td>"
            f"<td>{html.escape(str(gate.get('action', '')))}</td>"
            f"<td>{html.escape(', '.join(str(tag) for tag in quality.get('tags', []) if tag is not None))}</td>"
            "</tr>"
        )
    body = "\n".join(rows) or "<tr><td colspan=\"6\">No generated axes.</td></tr>"
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OSRGen Prediction QC</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>OSRGen Prediction QC</h1>
  <table>
    <thead><tr><th>Axis</th><th>Script</th><th>Status</th><th>Score</th><th>Gate</th><th>Tags</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
"""
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def write_batch_summary_csv(results: list[BatchPredictResult], path: Path) -> None:
    columns = [
        "sample_id",
        "status",
        "mean_score",
        "axis_count",
        "review_axes",
        "gated_axes",
        "omitted_axes",
        "video",
        "output_dir",
        "prediction_summary",
        "qc_report",
        "warnings",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            row = result.to_json()
            for key in ("review_axes", "gated_axes", "omitted_axes", "warnings"):
                row[key] = "; ".join(str(value) for value in row[key])
            writer.writerow({column: row.get(column) for column in columns})


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
