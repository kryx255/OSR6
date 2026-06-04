from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from osrgen.gui import OUTPUT_SAME_DIR, OUTPUT_SAME_NAME_FOLDER
from osrgen.gui import available_worker_options
from osrgen.gui import build_predict_command, collect_generated_scripts, copy_scripts_to_video_directory
from osrgen.gui import final_output_dir_for
from osrgen.gui import format_device_options
from osrgen.gui import format_speed_options, speed_overrides
from osrgen.gui import normalize_video_paths, prediction_dir_for, scan_video_folder
from osrgen.gui import parse_worker_count
from osrgen.gui import should_clear_video_queue_after_run


class GuiHelperTests(unittest.TestCase):
    def test_normalize_video_paths_filters_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            other = root / "notes.txt"
            video.write_bytes(b"video")
            other.write_text("x", encoding="utf-8")

            paths = normalize_video_paths([video, video, other, root / "missing.mp4"])

            self.assertEqual(paths, [video.resolve()])

    def test_scan_video_folder_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            (root / "a.mp4").write_bytes(b"a")
            (nested / "b.mkv").write_bytes(b"b")

            paths = scan_video_folder(root, recursive=True)

            self.assertEqual([path.name for path in paths], ["a.mp4", "b.mkv"])

    def test_output_dir_modes(self) -> None:
        video = Path("C:/videos/demo.mp4")

        self.assertEqual(final_output_dir_for(video, OUTPUT_SAME_DIR), Path("C:/videos"))
        self.assertEqual(final_output_dir_for(video, OUTPUT_SAME_NAME_FOLDER), Path("C:/videos/demo"))

    def test_prediction_dir_uses_video_stem(self) -> None:
        self.assertEqual(prediction_dir_for("C:/videos/demo.mp4", "C:/out"), Path("C:/out/demo"))

    def test_copy_scripts_to_video_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            generated = root / "tmp" / "clip"
            generated.mkdir(parents=True)
            video.write_bytes(b"video")
            (generated / "clip.funscript").write_text('{"actions":[]}', encoding="utf-8")
            (generated / "clip.surge.funscript").write_text('{"actions":[]}', encoding="utf-8")
            (generated / "features.csv").write_text("time_ms,pos\n", encoding="utf-8")

            copied = copy_scripts_to_video_directory(video, generated)

            self.assertEqual([path.name for path in copied], ["clip.funscript", "clip.surge.funscript"])
            self.assertTrue((root / "clip.funscript").is_file())
            self.assertTrue((root / "clip.surge.funscript").is_file())

    def test_collect_generated_scripts_sorts_scripts_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.funscript").write_text("{}", encoding="utf-8")
            (root / "a.funscript").write_text("{}", encoding="utf-8")
            (root / "features.csv").write_text("", encoding="utf-8")

            scripts = collect_generated_scripts(root)

            self.assertEqual([path.name for path in scripts], ["a.funscript", "b.funscript"])

    def test_queue_clears_only_after_successful_full_run(self) -> None:
        self.assertTrue(should_clear_video_queue_after_run(stopped=False, success=2, total=2))
        self.assertFalse(should_clear_video_queue_after_run(stopped=True, success=1, total=2))
        self.assertFalse(should_clear_video_queue_after_run(stopped=False, success=1, total=2))

    def test_predict_command_includes_device(self) -> None:
        command = build_predict_command(
            Path("C:/videos/demo.mp4"),
            output_root=Path("C:/out"),
            preset=Path("preset.json"),
            device="cuda:0",
        )

        self.assertIn("--device", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")

    def test_predict_command_includes_speed_overrides(self) -> None:
        command = build_predict_command(
            Path("C:/videos/demo.mp4"),
            output_root=Path("C:/out"),
            preset=Path("preset.json"),
            analysis_fps=4.0,
            max_width=256,
        )

        self.assertEqual(command[command.index("--analysis-fps") + 1], "4.0")
        self.assertEqual(command[command.index("--max-width") + 1], "256")

    def test_device_options_show_hardware_labels(self) -> None:
        options = format_device_options("zh", {"cuda:0": "NVIDIA GeForce RTX 5090"})

        self.assertEqual([option.value for option in options], ["auto", "cpu", "cuda:0"])
        self.assertIn("NVIDIA GeForce RTX 5090", options[0].label)
        self.assertIn("CPU", options[1].label)
        self.assertEqual(options[2].label, "GPU 0: NVIDIA GeForce RTX 5090 (cuda:0)")

    def test_speed_options_map_to_runtime_overrides(self) -> None:
        options = format_speed_options("zh")

        self.assertEqual([option.value for option in options], ["quality", "balanced", "fast"])
        self.assertEqual(speed_overrides("quality"), (None, None))
        self.assertEqual(speed_overrides("balanced"), (6.0, 320))
        self.assertEqual(speed_overrides("fast"), (4.0, 256))

    def test_worker_options_are_bounded(self) -> None:
        self.assertEqual(available_worker_options(2), ("1", "2"))
        self.assertEqual(available_worker_options(16), ("1", "2", "3", "4"))
        self.assertEqual(parse_worker_count("3"), 3)
        self.assertEqual(parse_worker_count("bad"), 1)


if __name__ == "__main__":
    unittest.main()
