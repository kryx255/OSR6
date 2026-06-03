from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osrgen.preset_validation import ModelPresetValidationConfig, validate_model_preset


class PresetValidationTests(unittest.TestCase):
    def test_validate_model_preset_accepts_consistent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "models"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "l0.pt").touch()
            preset = root / "preset.json"
            preset.write_text(
                json.dumps({"name": "test", "checkpoint_dir": str(checkpoint_dir), "axes": ["l0"]}),
                encoding="utf-8",
            )

            with patch("osrgen.preset_validation.load_checkpoint", return_value=fake_checkpoint("l0")):
                report = validate_model_preset(
                    ModelPresetValidationConfig(
                        preset=str(preset),
                        output=str(root / "report"),
                        check_device=False,
                    )
                )

            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["checkpoint_count"], 1)
            self.assertTrue((root / "report" / "summary.json").is_file())

    def test_validate_model_preset_rejects_axis_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "models"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "l0.pt").touch()
            preset = root / "preset.json"
            preset.write_text(
                json.dumps({"checkpoint_dir": str(checkpoint_dir), "axes": ["l0"]}),
                encoding="utf-8",
            )

            with patch("osrgen.preset_validation.load_checkpoint", return_value=fake_checkpoint("r0")):
                report = validate_model_preset(
                    ModelPresetValidationConfig(preset=str(preset), check_device=False)
                )

            self.assertEqual(report["status"], "error")
            self.assertIn("expected axis l0", " ".join(report["errors"]))

    def test_validate_model_preset_accepts_postprocess_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "models"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "l0.pt").touch()
            profile = root / "post.json"
            profile.write_text(
                json.dumps(
                    {
                        "axes": {
                            "l0": {
                                "smoothing_ms": 0,
                                "min_interval_ms": 33,
                                "min_delta": 0,
                                "deadband": 0,
                                "max_actions_per_second": None,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            preset = root / "preset.json"
            preset.write_text(
                json.dumps(
                    {
                        "checkpoint_dir": str(checkpoint_dir),
                        "axes": ["l0"],
                        "postprocess_profile": str(profile),
                    }
                ),
                encoding="utf-8",
            )

            with patch("osrgen.preset_validation.load_checkpoint", return_value=fake_checkpoint("l0")):
                report = validate_model_preset(
                    ModelPresetValidationConfig(preset=str(preset), check_device=False)
                )

            self.assertEqual(report["status"], "ready")
            self.assertTrue(any(resource["kind"] == "postprocess_profile" for resource in report["resources"]))


def fake_checkpoint(axis: str, columns: list[str] | None = None) -> dict[str, object]:
    active_columns = columns or ["mean_dx"]
    return {
        "axis": axis,
        "feature_columns": active_columns,
        "model_config": {"input_dim": len(active_columns)},
        "normalization": {"mean": [0.0 for _ in active_columns]},
    }


if __name__ == "__main__":
    unittest.main()
