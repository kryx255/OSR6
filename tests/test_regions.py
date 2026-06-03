from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osrgen.features import MotionFeature
from osrgen.regions import DEFAULT_REGIONS, add_region_features
from osrgen.regions import region_feature_column, select_regions, signal_values


class RegionTests(unittest.TestCase):
    def test_default_regions_are_valid(self) -> None:
        names = set()
        for region in DEFAULT_REGIONS:
            self.assertNotIn(region.name, names)
            names.add(region.name)
            x, y, w, h = region.roi
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertLessEqual(x + w, 1.0)
            self.assertLessEqual(y + h, 1.0)

    def test_select_regions_rejects_unknown_name(self) -> None:
        with self.assertRaises(RuntimeError):
            select_regions(("missing",))

    def test_signal_values_active_delta(self) -> None:
        features = [
            motion_feature(0, active_center_x=0.2),
            motion_feature(33, active_center_x=0.5),
            motion_feature(67, active_center_x=0.4),
        ]

        values = signal_values(features, "active_center_x_delta")
        self.assertEqual(values[:2], [0.0, 0.3])
        self.assertAlmostEqual(values[2], -0.1)

    def test_add_region_features_merges_extra_columns_by_time(self) -> None:
        base = [motion_feature(33), motion_feature(67)]
        region_features = {
            "center_50": [
                motion_feature(33, mean_dy=0.1),
                motion_feature(67, mean_dy=0.3),
            ]
        }

        enriched = add_region_features(base, region_features, ["mean_dy"])

        column = region_feature_column("center_50", "mean_dy")
        self.assertEqual(enriched[0].extra[column], 0.1)
        self.assertEqual(enriched[1].extra[column], 0.3)


def motion_feature(time_ms: int, *, active_center_x: float = 0.0, mean_dy: float = 0.0) -> MotionFeature:
    return MotionFeature(
        time_ms=time_ms,
        mean_dx=0,
        mean_dy=mean_dy,
        mean_mag=0,
        radial=0,
        scene_cut=0,
        roi_x=0,
        roi_y=0,
        roi_w=1,
        roi_h=1,
        active_center_x=active_center_x,
    )


if __name__ == "__main__":
    unittest.main()
