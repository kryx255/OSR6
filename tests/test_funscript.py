from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osrgen.funscript import Action, Funscript
from osrgen.generator import positions_to_funscript


class FunscriptTests(unittest.TestCase):
    def test_roundtrip_and_interpolation(self) -> None:
        script = Funscript.from_actions([Action(0, 0), Action(1000, 100)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.funscript"
            script.save(path)
            loaded = Funscript.load(path)

        self.assertEqual(len(loaded.actions), 2)
        self.assertEqual(loaded.sample_at(0), 0)
        self.assertEqual(loaded.sample_at(500), 50)
        self.assertEqual(loaded.sample_at(1000), 100)

    def test_positions_to_funscript_filters_small_actions(self) -> None:
        times = [0, 100, 200, 300, 400, 500]
        positions = [50, 52, 49, 53, 48, 50]

        script = positions_to_funscript(times, positions, min_interval_ms=50, min_delta=5.0, deadband=2.0)

        self.assertLess(len(script.actions), len(times))
        self.assertEqual(script.actions[0].at, 0)
        self.assertEqual(script.actions[-1].at, 500)

    def test_positions_to_funscript_limits_action_rate(self) -> None:
        times = [0, 100, 200, 300, 400, 500]
        positions = [50, 80, 20, 80, 20, 50]

        script = positions_to_funscript(times, positions, min_interval_ms=50, max_actions_per_second=2.0)

        self.assertLessEqual(len(script.actions), 2)


if __name__ == "__main__":
    unittest.main()
