from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osrgen.batch_predict import BatchPredictConfig, run_batch_prediction, scan_batch_videos
from osrgen.modeling import ModelPredictAllConfig
from osrgen.project import file_fingerprint, write_json


class BatchPredictTests(unittest.TestCase):
    def test_scan_batch_videos_respects_recursive_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "movie.mp4").write_bytes(b"fake")
            nested = root / "nested"
            nested.mkdir()
            (nested / "clip.mkv").write_bytes(b"fake")
            (nested / "ignore.txt").write_text("x", encoding="utf-8")

            direct = scan_batch_videos(root, recursive=False)
            recursive = scan_batch_videos(root, recursive=True)

        self.assertEqual([path.name for path in direct], ["movie.mp4"])
        self.assertEqual([path.name for path in recursive], ["movie.mp4", "clip.mkv"])

    def test_batch_predict_writes_summary_qc_and_resumes(self) -> None:
        calls: list[str] = []

        def fake_predictor(config: ModelPredictAllConfig) -> list[Path]:
            calls.append(config.input_path)
            video = Path(config.input_path)
            out_dir = Path(config.output) / video.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                out_dir / "prediction_all.json",
                {
                    "config": config.to_json(),
                    "input_fingerprint": file_fingerprint(video),
                    "quality_summary": {
                        "axis_count": 1,
                        "mean_score": 42.0,
                        "review_axes": ["r2"],
                    },
                    "quality_gate_summary": {
                        "triggered_axes": ["r2"],
                    },
                    "generated": [
                        {
                            "axis": "r2",
                            "script_path": None,
                            "action_count": 0,
                            "axis_scale": 0.25,
                            "axis_scale_detail": {"confidence": 0.15},
                            "prediction_stats": {"range": 4.0},
                            "quality": {
                                "status": "weak",
                                "score": 42,
                                "tags": ["near_flat_output"],
                                "actions_per_second": 3.0,
                            },
                            "quality_gate": {
                                "action": "omit",
                                "triggered": True,
                            },
                        }
                    ],
                },
            )
            return []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "videos"
            output = root / "output"
            input_dir.mkdir()
            (input_dir / "movie.mp4").write_bytes(b"fake")
            config = BatchPredictConfig(
                input_dir=str(input_dir),
                output=str(output),
                predict=ModelPredictAllConfig(input_path="", checkpoint_dir="models", output=""),
            )

            first = run_batch_prediction(config, predictor=fake_predictor)
            second = run_batch_prediction(config, predictor=fake_predictor)

            self.assertEqual(first[0].status, "ok")
            self.assertEqual(first[0].review_axes, ["r2"])
            self.assertEqual(first[0].omitted_axes, ["r2"])
            self.assertEqual(second[0].status, "skipped_existing")
            self.assertEqual(len(calls), 1)
            self.assertTrue((output / "batch_summary.csv").exists())
            self.assertTrue((output / "_qc" / "movie" / "index.html").exists())

    def test_batch_predict_regenerates_when_config_or_video_changes(self) -> None:
        calls: list[ModelPredictAllConfig] = []

        def fake_predictor(config: ModelPredictAllConfig) -> list[Path]:
            calls.append(config)
            video = Path(config.input_path)
            out_dir = Path(config.output) / video.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                out_dir / "prediction_all.json",
                {
                    "config": config.to_json(),
                    "input_fingerprint": file_fingerprint(video),
                    "quality_summary": {"axis_count": 0, "mean_score": None, "review_axes": []},
                    "quality_gate_summary": {"triggered_axes": []},
                    "generated": [],
                },
            )
            return []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "videos"
            output = root / "output"
            input_dir.mkdir()
            video = input_dir / "movie.mp4"
            video.write_bytes(b"first")
            baseline = BatchPredictConfig(
                input_dir=str(input_dir),
                output=str(output),
                predict=ModelPredictAllConfig(input_path="", checkpoint_dir="models", output=""),
            )
            changed_config = BatchPredictConfig(
                input_dir=str(input_dir),
                output=str(output),
                predict=ModelPredictAllConfig(input_path="", checkpoint_dir="models", output="", smoothing_ms=100),
            )

            first = run_batch_prediction(baseline, predictor=fake_predictor)
            config_changed = run_batch_prediction(changed_config, predictor=fake_predictor)
            video.write_bytes(b"second version")
            video_changed = run_batch_prediction(changed_config, predictor=fake_predictor)

            self.assertEqual(first[0].warnings, [])
            self.assertEqual(config_changed[0].warnings, ["prediction config changed; regenerated"])
            self.assertEqual(video_changed[0].warnings, ["input video changed; regenerated"])
            self.assertEqual(len(calls), 3)

    def test_batch_predict_regenerates_when_script_output_is_missing(self) -> None:
        calls: list[str] = []

        def fake_predictor(config: ModelPredictAllConfig) -> list[Path]:
            calls.append(config.input_path)
            video = Path(config.input_path)
            out_dir = Path(config.output) / video.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            script = out_dir / "movie.funscript"
            script.write_text('{"actions":[]}', encoding="utf-8")
            write_json(
                out_dir / "prediction_all.json",
                {
                    "config": config.to_json(),
                    "input_fingerprint": file_fingerprint(video),
                    "quality_summary": {"axis_count": 1, "mean_score": 100.0, "review_axes": []},
                    "quality_gate_summary": {"triggered_axes": []},
                    "generated": [{"axis": "l0", "script_path": script.name, "quality_gate": {"action": "keep"}}],
                },
            )
            return [script]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "videos"
            output = root / "output"
            input_dir.mkdir()
            (input_dir / "movie.mp4").write_bytes(b"fake")
            config = BatchPredictConfig(
                input_dir=str(input_dir),
                output=str(output),
                predict=ModelPredictAllConfig(input_path="", checkpoint_dir="models", output=""),
            )

            first = run_batch_prediction(config, predictor=fake_predictor)
            (output / "movie" / "movie.funscript").unlink()
            second = run_batch_prediction(config, predictor=fake_predictor)

            self.assertEqual(first[0].warnings, [])
            self.assertEqual(second[0].warnings, ["prediction output files missing; regenerated"])
            self.assertEqual(len(calls), 2)

    def test_batch_predict_reports_corrupt_resumed_summary_as_video_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "videos"
            output = root / "output"
            input_dir.mkdir()
            (input_dir / "movie.mp4").write_bytes(b"fake")
            prediction_dir = output / "movie"
            prediction_dir.mkdir(parents=True)
            (prediction_dir / "prediction_all.json").write_text("{broken", encoding="utf-8")
            config = BatchPredictConfig(
                input_dir=str(input_dir),
                output=str(output),
                predict=ModelPredictAllConfig(input_path="", checkpoint_dir="models", output=""),
            )

            results = run_batch_prediction(config)

            self.assertEqual(results[0].status, "error")
            self.assertIn("Expecting property name", results[0].warnings[0])


if __name__ == "__main__":
    unittest.main()
