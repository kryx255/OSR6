from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from osrgen.modeling import ModelPredictAllConfig, axis_script_filename, config_with_inferred_region_request
from osrgen.modeling import infer_region_request_from_feature_columns, load_axis_scale_profile_data
from osrgen.modeling import load_postprocess_profile, neutral_funscript, position_stats
from osrgen.modeling import prediction_quality_for_axis, quality_gate_detail, scale_axis_positions
from osrgen.modeling import summarize_prediction_quality


class ModelingRuntimeTests(unittest.TestCase):
    def test_config_to_json_serializes_tuples(self) -> None:
        config = ModelPredictAllConfig(
            input_path="video.mp4",
            checkpoint_dir="models",
            output="out",
            axes=("l0", "r2"),
            region_regions=("lower_center",),
            region_signals=("mean_dy",),
        )

        data = config.to_json()

        self.assertEqual(data["axes"], ["l0", "r2"])
        self.assertEqual(data["region_regions"], ["lower_center"])
        self.assertEqual(data["region_signals"], ["mean_dy"])

    def test_region_request_infers_checkpoint_subset(self) -> None:
        request = infer_region_request_from_feature_columns(
            [
                "mean_dx",
                "region_mid_left_active_mean_dy",
                "region_lower_center_radial",
            ]
        )

        self.assertEqual(request, (["lower_center", "mid_left"], ["radial", "active_mean_dy"]))

    def test_config_enables_inferred_regions(self) -> None:
        config = ModelPredictAllConfig(input_path="video.mp4", checkpoint_dir="models", output="out")

        inferred = config_with_inferred_region_request(config, ["region_center_50_mean_dy"])

        self.assertTrue(inferred.region_features)
        self.assertEqual(inferred.region_regions, ("center_50",))
        self.assertEqual(inferred.region_signals, ("mean_dy",))

    def test_axis_scale_profile_loader_accepts_axis_scales(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scales.json"
            path.write_text(
                json.dumps({"axis_scales": {"l0": 1.25, "r2": {"scale": 0.8, "confidence": 0.6}}}),
                encoding="utf-8",
            )

            scales, details = load_axis_scale_profile_data(path)

        self.assertEqual(scales["l0"], 1.25)
        self.assertEqual(scales["r2"], 0.8)
        self.assertEqual(details["r2"]["confidence"], 0.6)

    def test_postprocess_profile_loader_normalizes_axis_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "post.json"
            path.write_text(
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

            profile = load_postprocess_profile(path)

        self.assertEqual(profile["l0"]["smoothing_ms"], 0)
        self.assertEqual(profile["l0"]["min_interval_ms"], 33)
        self.assertIsNone(profile["l0"]["max_actions_per_second"])

    def test_scale_axis_positions_scales_around_neutral(self) -> None:
        positions = np.asarray([0.0, 50.0, 100.0], dtype=np.float32)

        scaled = scale_axis_positions(positions, 0.5)

        self.assertEqual(scaled.tolist(), [25.0, 50.0, 75.0])

    def test_position_stats_include_range_and_duration(self) -> None:
        stats = position_stats([0, 100, 200], np.asarray([50.0, 60.0, 55.0], dtype=np.float32))

        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["range"], 10.0)
        self.assertEqual(stats["duration_s"], 0.2)

    def test_prediction_quality_flags_weak_no_reference_axes(self) -> None:
        quality = prediction_quality_for_axis(
            axis="r2",
            action_count=500,
            times=[0, 100_000],
            prediction_stats={"range": 4.0},
            raw_prediction_stats={"range": 20.0},
            axis_scale_detail={"scale": 0.25, "confidence": 0.15},
        )

        self.assertEqual(quality["status"], "weak")
        self.assertIn("low_axis_confidence", quality["tags"])
        self.assertIn("near_flat_output", quality["tags"])
        self.assertIn("many_small_actions", quality["tags"])

    def test_quality_gate_marks_low_score_axis(self) -> None:
        detail = quality_gate_detail(mode="neutralize", threshold=60.0, quality={"status": "weak", "score": 45})

        self.assertTrue(detail["triggered"])
        self.assertEqual(detail["action"], "neutralize")

    def test_summarize_prediction_quality_collects_review_axes(self) -> None:
        summary = summarize_prediction_quality(
            [
                {"axis": "l0", "quality": {"status": "ok", "score": 100, "tags": []}},
                {"axis": "r2", "quality": {"status": "weak", "score": 40, "tags": ["near_flat_output"]}},
            ]
        )

        self.assertEqual(summary["axis_count"], 2)
        self.assertEqual(summary["review_axes"], ["r2"])
        self.assertEqual(summary["tag_counts"], {"near_flat_output": 1})

    def test_neutral_funscript_keeps_video_duration(self) -> None:
        script = neutral_funscript([0, 100, 200])

        self.assertEqual([(action.at, action.pos) for action in script.actions], [(0, 50), (200, 50)])

    def test_axis_script_filename_uses_sr6_suffixes(self) -> None:
        self.assertEqual(axis_script_filename("movie", "l0"), "movie.funscript")
        self.assertEqual(axis_script_filename("movie", "l1"), "movie.surge.funscript")
        self.assertEqual(axis_script_filename("movie", "r2"), "movie.pitch.funscript")


if __name__ == "__main__":
    unittest.main()
