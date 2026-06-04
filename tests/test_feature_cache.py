from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from osrgen.feature_cache import feature_cache_request, load_cached_features_path, store_cached_features
from osrgen.project import file_fingerprint


class FeatureCacheTests(unittest.TestCase):
    def test_store_and_load_cached_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            features = root / "features.csv"
            cache = root / "cache"
            video.write_bytes(b"video")
            features.write_text("time_ms,mean_dx\n0,0\n", encoding="utf-8")
            request = feature_cache_request(
                input_path=video,
                input_fingerprint=file_fingerprint(video),
                analysis_fps=8.0,
                max_width=360,
                scene_threshold=35.0,
                roi=(0.15, 0.15, 0.7, 0.7),
                region_features=True,
                region_regions=("full", "center_70"),
                region_signals=("mean_dx", "mean_dy"),
            )

            stored = store_cached_features(request, features, cache)
            loaded = load_cached_features_path(request, cache)

            self.assertIsNotNone(stored)
            self.assertEqual(loaded, stored)
            self.assertEqual(loaded.read_text(encoding="utf-8"), features.read_text(encoding="utf-8"))  # type: ignore[union-attr]

    def test_cache_request_changes_with_speed_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"video")
            fingerprint = file_fingerprint(video)

            quality = feature_cache_request(
                input_path=video,
                input_fingerprint=fingerprint,
                analysis_fps=8.0,
                max_width=360,
                scene_threshold=35.0,
                roi=(0.15, 0.15, 0.7, 0.7),
                region_features=True,
                region_regions=("full",),
                region_signals=("mean_dx",),
            )
            fast = feature_cache_request(
                input_path=video,
                input_fingerprint=fingerprint,
                analysis_fps=4.0,
                max_width=256,
                scene_threshold=35.0,
                roi=(0.15, 0.15, 0.7, 0.7),
                region_features=True,
                region_regions=("full",),
                region_signals=("mean_dx",),
            )

            self.assertNotEqual(quality, fast)


if __name__ == "__main__":
    unittest.main()
