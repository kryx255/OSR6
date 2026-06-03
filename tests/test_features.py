from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osrgen.features import FEATURE_COLUMNS, MODEL_INPUT_COLUMNS, MotionFeature, load_features_csv, save_features_csv


class FeatureTests(unittest.TestCase):
    def test_load_old_feature_csv_defaults_pose_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            path.write_text(
                "time_ms,mean_dx,mean_dy,mean_mag,radial,scene_cut,roi_x,roi_y,roi_w,roi_h\n"
                "33,1,2,3,4,0,0.1,0.2,0.7,0.8\n",
                encoding="utf-8",
            )

            features = load_features_csv(path)

        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].time_ms, 33)
        self.assertEqual(features[0].pose_detected, 0)
        self.assertEqual(features[0].pose_center_x, 0.0)

    def test_save_feature_csv_includes_pose_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            save_features_csv(
                [
                    MotionFeature(
                        time_ms=33,
                        mean_dx=0,
                        mean_dy=0,
                        mean_mag=0,
                        radial=0,
                        scene_cut=0,
                        roi_x=0,
                        roi_y=0,
                        roi_w=1,
                        roi_h=1,
                        pose_detected=1,
                        pose_center_x=0.4,
                    )
                ],
                path,
            )

            text = path.read_text(encoding="utf-8")
            loaded = load_features_csv(path)

        self.assertIn("pose_detected", text.splitlines()[0])
        self.assertEqual(loaded[0].pose_detected, 1)
        self.assertEqual(loaded[0].pose_center_x, 0.4)
        self.assertIn("pose_center_x", FEATURE_COLUMNS)
        self.assertIn("pose_center_x", MODEL_INPUT_COLUMNS)
        self.assertNotIn("time_ms", MODEL_INPUT_COLUMNS)

    def test_save_feature_csv_round_trips_extra_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            save_features_csv(
                [
                    MotionFeature(
                        time_ms=33,
                        mean_dx=0,
                        mean_dy=0,
                        mean_mag=0,
                        radial=0,
                        scene_cut=0,
                        roi_x=0,
                        roi_y=0,
                        roi_w=1,
                        roi_h=1,
                        extra={"region_center_50_mean_dy": 0.25},
                    )
                ],
                path,
            )

            text = path.read_text(encoding="utf-8")
            loaded = load_features_csv(path)

        self.assertIn("region_center_50_mean_dy", text.splitlines()[0])
        self.assertEqual(loaded[0].extra["region_center_50_mean_dy"], 0.25)


if __name__ == "__main__":
    unittest.main()
