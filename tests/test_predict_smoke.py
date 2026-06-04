from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from osrgen.axes import AXIS_ORDER
from osrgen.features import MODEL_INPUT_COLUMNS
from osrgen.modeling import ModelPredictAllConfig, predict_all_models
from osrgen.tcn import create_model


class PredictSmokeTests(unittest.TestCase):
    def test_predict_all_models_runs_from_synthetic_video(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            self.skipTest(f"Optional runtime dependency is not installed: {exc.name}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "motion.avi"
            checkpoint_dir = root / "checkpoints"
            output = root / "output"
            checkpoint_dir.mkdir()

            write_synthetic_video(video, cv2, np)
            for axis in AXIS_ORDER:
                write_test_checkpoint(checkpoint_dir / f"{axis}.pt", axis, torch)

            script_paths = predict_all_models(
                ModelPredictAllConfig(
                    input_path=str(video),
                    checkpoint_dir=str(checkpoint_dir),
                    output=str(output),
                    analysis_fps=10.0,
                    max_width=64,
                    axes=AXIS_ORDER,
                    smoothing_ms=0,
                    min_interval_ms=50,
                    quality_gate="warn",
                    feature_cache=False,
                )
            )

            summary_path = output / "motion" / "prediction_all.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(len(script_paths), 6)
            self.assertEqual(summary["feature_count"], 9)
            self.assertEqual(len(summary["generated"]), 6)
            for script_path in script_paths:
                script = json.loads(script_path.read_text(encoding="utf-8"))
                self.assertIn("actions", script)
                self.assertGreaterEqual(len(script["actions"]), 2)


def write_synthetic_video(path: Path, cv2, np) -> None:  # type: ignore[no-untyped-def]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV could not create a synthetic test video.")
    try:
        for index in range(10):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            x = 6 + index * 3
            cv2.rectangle(frame, (x, 20), (x + 18, 42), (255, 255, 255), thickness=-1)
            writer.write(frame)
    finally:
        writer.release()


def write_test_checkpoint(path: Path, axis: str, torch) -> None:  # type: ignore[no-untyped-def]
    model_config = {
        "input_dim": len(MODEL_INPUT_COLUMNS),
        "channels": 4,
        "layers": 1,
        "kernel_size": 3,
        "dropout": 0.0,
    }
    model = create_model(**model_config)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    torch.save(
        {
            "axis": axis,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "normalization": {
                "mean": [0.0 for _ in MODEL_INPUT_COLUMNS],
                "std": [1.0 for _ in MODEL_INPUT_COLUMNS],
            },
            "feature_columns": list(MODEL_INPUT_COLUMNS),
        },
        path,
    )


if __name__ == "__main__":
    unittest.main()
