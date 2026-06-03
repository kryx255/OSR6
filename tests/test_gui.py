from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from osrgen.gui import OUTPUT_SAME_DIR, OUTPUT_SAME_NAME_FOLDER
from osrgen.gui import collect_generated_scripts, copy_scripts_to_video_directory, final_output_dir_for
from osrgen.gui import normalize_video_paths, prediction_dir_for, scan_video_folder
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


if __name__ == "__main__":
    unittest.main()
