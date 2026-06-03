from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osrgen.cli import build_parser, parse_axis_paths, parse_axis_scales, parse_float_list, parse_int_list
from osrgen.cli import parse_optional_float_list


class CliTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows batch wrapper test")
    def test_batch_wrapper_runs_from_outside_project_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                ["cmd", "/d", "/c", "call", str(ROOT / "run_osrgen.bat"), "--help"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Generate OSR6/SR6 funscripts", result.stdout)

    def test_parse_axis_scales(self) -> None:
        self.assertEqual(parse_axis_scales("l1=0.5,r0=0.25"), {"l1": 0.5, "r0": 0.25})
        self.assertEqual(parse_axis_scales({"l0": 1.25}), {"l0": 1.25})
        self.assertIsNone(parse_axis_scales(""))

    def test_parse_int_list(self) -> None:
        self.assertEqual(parse_int_list("0,100,250"), [0, 100, 250])

    def test_parse_float_lists(self) -> None:
        self.assertEqual(parse_float_list("0,1.5"), [0.0, 1.5])
        self.assertEqual(parse_optional_float_list("2.5"), [2.5])
        self.assertIsNone(parse_optional_float_list(""))

    def test_parse_axis_paths(self) -> None:
        self.assertEqual(parse_axis_paths("l0=a.json,r1=b.json"), {"l0": "a.json", "r1": "b.json"})

    def test_predict_all_parser_accepts_runtime_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "model",
                "predict-all",
                "video.mp4",
                "--preset",
                "configs/presets/region_hybrid_experience_95.json",
                "--axis-scales",
                "l1=0.5,r0=0.25",
                "--axis-scale-profile",
                "scales.json",
                "--postprocess-profile",
                "post.json",
                "--quality-gate",
                "neutralize",
                "--quality-threshold",
                "60",
            ]
        )

        self.assertEqual(str(args.input), "video.mp4")
        self.assertEqual(args.axis_scales, "l1=0.5,r0=0.25")
        self.assertEqual(str(args.axis_scale_profile), "scales.json")
        self.assertEqual(str(args.postprocess_profile), "post.json")
        self.assertEqual(args.quality_gate, "neutralize")
        self.assertEqual(args.quality_threshold, 60.0)

    def test_validate_preset_parser(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "model",
                "validate-preset",
                "--preset",
                "configs/presets/region_hybrid_experience_95.json",
                "--skip-device-check",
            ]
        )

        self.assertEqual(args.preset, Path("configs/presets/region_hybrid_experience_95.json"))
        self.assertTrue(args.skip_device_check)

    def test_batch_predict_parser(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "model",
                "batch-predict",
                "--input-dir",
                "videos",
                "--preset",
                "configs/presets/region_hybrid_experience_95.json",
                "--quality-gate",
                "warn",
                "--recursive",
                "--force",
            ]
        )

        self.assertEqual(str(args.input_dir), "videos")
        self.assertEqual(args.quality_gate, "warn")
        self.assertTrue(args.recursive)
        self.assertTrue(args.force)


if __name__ == "__main__":
    unittest.main()
