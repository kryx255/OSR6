from __future__ import annotations

from dataclasses import dataclass
import locale
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Iterable

from .batch_predict import VIDEO_EXTENSIONS


DEFAULT_PRESET = Path("configs/presets/region_hybrid_experience_95.json")
OUTPUT_SAME_DIR = "same_dir"
OUTPUT_SAME_NAME_FOLDER = "same_name_folder"
LOG_POLL_MS = 100
DEFAULT_DEVICE = "auto"
SPEED_QUALITY = "quality"
SPEED_BALANCED = "balanced"
SPEED_FAST = "fast"


LANGUAGE_NAMES = {
    "zh": "简体中文",
    "en": "English",
    "ja": "日本語",
}

LANGUAGE_PACKS = {
    "zh": {
        "title": "OSR6 脚本生成器",
        "language": "语言",
        "add_videos": "添加视频",
        "add_folder": "添加文件夹",
        "recursive": "递归",
        "remove_selected": "移除选中",
        "clear": "清空",
        "output_settings": "输出设置",
        "output_same_dir": "脚本直接生成在视频同目录",
        "output_same_name_folder": "在视频同目录创建同名文件夹",
        "preset": "模型 preset",
        "device": "推理设备",
        "speed": "速度/质量",
        "speed_quality": "质量优先（8fps / 360px）",
        "speed_balanced": "均衡（6fps / 320px）",
        "speed_fast": "快速（4fps / 256px）",
        "device_auto_cpu": "自动选择（当前可用：CPU）",
        "device_auto_gpu": "自动选择（推荐：{name}）",
        "device_cpu": "CPU（处理器）",
        "device_cuda_index": "GPU {index}: {name} ({device})",
        "browse": "浏览",
        "video": "视频",
        "output": "输出位置",
        "start": "开始生成",
        "stop": "停止",
        "choose_videos": "选择视频",
        "choose_folder": "选择视频文件夹",
        "choose_preset": "选择 preset JSON",
        "status_select_videos": "请选择视频。",
        "status_added": "已添加 {count} 个视频，共 {total} 个。",
        "status_cleared": "列表已清空。",
        "no_videos_title": "没有视频",
        "no_videos_message": "请先添加一个或多个视频。",
        "preset_missing_title": "Preset 不存在",
        "preset_missing_message": "找不到 preset:\n{path}",
        "status_generating": "正在生成...",
        "status_stopping": "正在停止当前任务...",
        "status_stopped": "已停止，完成 {success} / {total}。",
        "status_done": "完成 {success} / {total}。",
        "log_preset": "使用 preset: {preset}",
        "log_device": "推理设备: {device}",
        "log_speed": "速度/质量: {speed}",
        "log_output_mode": "输出模式: {mode}",
        "log_start": "[{index}/{total}] 开始: {video}",
        "log_done": "[{index}/{total}] 完成: {count} 个脚本 -> {output} ({elapsed:.1f}s)",
        "error_prefix": "错误: {message}",
        "generation_failed": "生成失败",
    },
    "en": {
        "title": "OSR6 Script Generator",
        "language": "Language",
        "add_videos": "Add Videos",
        "add_folder": "Add Folder",
        "recursive": "Recursive",
        "remove_selected": "Remove Selected",
        "clear": "Clear",
        "output_settings": "Output Settings",
        "output_same_dir": "Write scripts next to video",
        "output_same_name_folder": "Create a same-name folder next to video",
        "preset": "Model preset",
        "device": "Inference device",
        "speed": "Speed/quality",
        "speed_quality": "Quality (8fps / 360px)",
        "speed_balanced": "Balanced (6fps / 320px)",
        "speed_fast": "Fast (4fps / 256px)",
        "device_auto_cpu": "Auto (available: CPU)",
        "device_auto_gpu": "Auto (recommended: {name})",
        "device_cpu": "CPU (processor)",
        "device_cuda_index": "GPU {index}: {name} ({device})",
        "browse": "Browse",
        "video": "Video",
        "output": "Output",
        "start": "Generate",
        "stop": "Stop",
        "choose_videos": "Choose videos",
        "choose_folder": "Choose video folder",
        "choose_preset": "Choose preset JSON",
        "status_select_videos": "Choose videos.",
        "status_added": "Added {count} videos, {total} total.",
        "status_cleared": "Queue cleared.",
        "no_videos_title": "No videos",
        "no_videos_message": "Add one or more videos first.",
        "preset_missing_title": "Preset not found",
        "preset_missing_message": "Could not find preset:\n{path}",
        "status_generating": "Generating...",
        "status_stopping": "Stopping current job...",
        "status_stopped": "Stopped, completed {success} / {total}.",
        "status_done": "Completed {success} / {total}.",
        "log_preset": "Preset: {preset}",
        "log_device": "Device: {device}",
        "log_speed": "Speed/quality: {speed}",
        "log_output_mode": "Output mode: {mode}",
        "log_start": "[{index}/{total}] Start: {video}",
        "log_done": "[{index}/{total}] Done: {count} scripts -> {output} ({elapsed:.1f}s)",
        "error_prefix": "Error: {message}",
        "generation_failed": "Generation failed",
    },
    "ja": {
        "title": "OSR6 スクリプト生成",
        "language": "言語",
        "add_videos": "動画を追加",
        "add_folder": "フォルダーを追加",
        "recursive": "再帰",
        "remove_selected": "選択を削除",
        "clear": "クリア",
        "output_settings": "出力設定",
        "output_same_dir": "動画と同じフォルダーに出力",
        "output_same_name_folder": "動画と同名のフォルダーを作成",
        "preset": "モデル preset",
        "device": "推論デバイス",
        "speed": "速度/品質",
        "speed_quality": "品質優先（8fps / 360px）",
        "speed_balanced": "バランス（6fps / 320px）",
        "speed_fast": "高速（4fps / 256px）",
        "device_auto_cpu": "自動選択（利用可能: CPU）",
        "device_auto_gpu": "自動選択（推奨: {name}）",
        "device_cpu": "CPU（プロセッサ）",
        "device_cuda_index": "GPU {index}: {name} ({device})",
        "browse": "参照",
        "video": "動画",
        "output": "出力先",
        "start": "生成",
        "stop": "停止",
        "choose_videos": "動画を選択",
        "choose_folder": "動画フォルダーを選択",
        "choose_preset": "preset JSON を選択",
        "status_select_videos": "動画を選択してください。",
        "status_added": "{count} 件追加、合計 {total} 件。",
        "status_cleared": "リストをクリアしました。",
        "no_videos_title": "動画がありません",
        "no_videos_message": "先に動画を追加してください。",
        "preset_missing_title": "Preset が見つかりません",
        "preset_missing_message": "Preset が見つかりません:\n{path}",
        "status_generating": "生成中...",
        "status_stopping": "現在の処理を停止しています...",
        "status_stopped": "停止しました。完了 {success} / {total}。",
        "status_done": "完了 {success} / {total}。",
        "log_preset": "Preset: {preset}",
        "log_device": "デバイス: {device}",
        "log_speed": "速度/品質: {speed}",
        "log_output_mode": "出力モード: {mode}",
        "log_start": "[{index}/{total}] 開始: {video}",
        "log_done": "[{index}/{total}] 完了: {count} scripts -> {output} ({elapsed:.1f}s)",
        "error_prefix": "エラー: {message}",
        "generation_failed": "生成に失敗しました",
    },
}


def default_language() -> str:
    language = (locale.getlocale()[0] or "").lower()
    if language.startswith("zh"):
        return "zh"
    if language.startswith("ja"):
        return "ja"
    return "en"


@dataclass(frozen=True)
class GuiGenerationResult:
    video: Path
    output_dir: Path
    scripts: tuple[Path, ...]


@dataclass(frozen=True)
class DeviceOption:
    value: str
    label: str


@dataclass(frozen=True)
class SpeedOption:
    value: str
    label: str
    analysis_fps: float | None
    max_width: int | None


def translated_text(language: str, key: str, **kwargs: object) -> str:
    pack = LANGUAGE_PACKS.get(language, LANGUAGE_PACKS["en"])
    text = pack.get(key, LANGUAGE_PACKS["en"].get(key, key))
    return text.format(**kwargs)


def format_speed_options(language: str) -> tuple[SpeedOption, ...]:
    return (
        SpeedOption(SPEED_QUALITY, translated_text(language, "speed_quality"), None, None),
        SpeedOption(SPEED_BALANCED, translated_text(language, "speed_balanced"), 6.0, 320),
        SpeedOption(SPEED_FAST, translated_text(language, "speed_fast"), 4.0, 256),
    )


def speed_overrides(value: str) -> tuple[float | None, int | None]:
    for option in format_speed_options("en"):
        if option.value == value:
            return option.analysis_fps, option.max_width
    return None, None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_video_paths(paths: Iterable[str | Path]) -> list[Path]:
    seen: set[str] = set()
    videos: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file():
            continue
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        videos.append(path)
    return videos


def scan_video_folder(folder: str | Path, *, recursive: bool = True) -> list[Path]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        return []
    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(normalize_video_paths(path for path in candidates if path.is_file()), key=lambda path: str(path).lower())


def prediction_dir_for(video: str | Path, output_root: str | Path) -> Path:
    return Path(output_root) / Path(video).stem


def final_output_dir_for(video: str | Path, output_mode: str) -> Path:
    src = Path(video)
    if output_mode == OUTPUT_SAME_NAME_FOLDER:
        return src.parent / src.stem
    if output_mode == OUTPUT_SAME_DIR:
        return src.parent
    raise ValueError(f"Unsupported output mode: {output_mode}")


def should_clear_video_queue_after_run(*, stopped: bool, success: int, total: int) -> bool:
    return not stopped and success == total


def collect_generated_scripts(prediction_dir: str | Path) -> tuple[Path, ...]:
    root = Path(prediction_dir)
    return tuple(sorted(root.glob("*.funscript"), key=lambda path: path.name.lower()))


def copy_scripts_to_video_directory(video: str | Path, prediction_dir: str | Path) -> tuple[Path, ...]:
    target_dir = Path(video).parent
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for script in collect_generated_scripts(prediction_dir):
        target = target_dir / script.name
        shutil.copy2(script, target)
        copied.append(target)
    return tuple(copied)


def python_executable() -> str:
    return os.environ.get("OSRGEN_PYTHON_EXE") or sys.executable


def detected_cuda_devices() -> dict[str, str]:
    devices: dict[str, str] = {}
    try:
        from .tcn import require_torch

        torch, _ = require_torch()
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                devices[f"cuda:{index}"] = str(torch.cuda.get_device_name(index))
    except Exception:
        pass
    return devices


def format_device_options(language: str, cuda_devices: dict[str, str]) -> tuple[DeviceOption, ...]:
    sorted_cuda = sorted(cuda_devices.items(), key=lambda item: int(item[0].split(":", 1)[1]))
    options: list[DeviceOption] = []
    if sorted_cuda:
        _device, name = sorted_cuda[0]
        options.append(DeviceOption(DEFAULT_DEVICE, translated_text(language, "device_auto_gpu", name=name)))
    else:
        options.append(DeviceOption(DEFAULT_DEVICE, translated_text(language, "device_auto_cpu")))
    options.append(DeviceOption("cpu", translated_text(language, "device_cpu")))
    for device, name in sorted_cuda:
        index = device.split(":", 1)[1]
        options.append(
            DeviceOption(
                device,
                translated_text(language, "device_cuda_index", index=index, name=name, device=device),
            )
        )
    return tuple(options)


def available_device_options(language: str = "en") -> tuple[DeviceOption, ...]:
    return format_device_options(language, detected_cuda_devices())


def build_predict_command(
    video: Path,
    *,
    output_root: Path,
    preset: Path,
    device: str = DEFAULT_DEVICE,
    analysis_fps: float | None = None,
    max_width: int | None = None,
) -> list[str]:
    command = [
        python_executable(),
        "-m",
        "osrgen",
        "model",
        "predict-all",
        str(video),
        "--preset",
        str(preset),
        "--output",
        str(output_root),
        "--device",
        device,
    ]
    if analysis_fps is not None:
        command.extend(["--analysis-fps", str(analysis_fps)])
    if max_width is not None:
        command.extend(["--max-width", str(max_width)])
    return command


def run_generation_for_video(
    video: Path,
    *,
    preset: Path,
    device: str,
    analysis_fps: float | None,
    max_width: int | None,
    output_mode: str,
    log: Callable[[str], None],
    should_stop: Callable[[], bool],
) -> GuiGenerationResult:
    root = project_root()
    if output_mode == OUTPUT_SAME_NAME_FOLDER:
        output_root = video.parent
        prediction_dir = prediction_dir_for(video, output_root)
        scripts = run_predict_subprocess(
            video,
            preset=preset,
            device=device,
            analysis_fps=analysis_fps,
            max_width=max_width,
            output_root=output_root,
            prediction_dir=prediction_dir,
            log=log,
            should_stop=should_stop,
            cwd=root,
        )
        return GuiGenerationResult(video=video, output_dir=prediction_dir, scripts=scripts)

    with tempfile.TemporaryDirectory(prefix="osrgen_gui_") as tmp:
        output_root = Path(tmp)
        prediction_dir = prediction_dir_for(video, output_root)
        run_predict_subprocess(
            video,
            preset=preset,
            device=device,
            analysis_fps=analysis_fps,
            max_width=max_width,
            output_root=output_root,
            prediction_dir=prediction_dir,
            log=log,
            should_stop=should_stop,
            cwd=root,
        )
        scripts = copy_scripts_to_video_directory(video, prediction_dir)
        return GuiGenerationResult(video=video, output_dir=video.parent, scripts=scripts)


def run_predict_subprocess(
    video: Path,
    *,
    preset: Path,
    device: str,
    analysis_fps: float | None,
    max_width: int | None,
    output_root: Path,
    prediction_dir: Path,
    log: Callable[[str], None],
    should_stop: Callable[[], bool],
    cwd: Path,
) -> tuple[Path, ...]:
    command = build_predict_command(
        video,
        output_root=output_root,
        preset=preset,
        device=device,
        analysis_fps=analysis_fps,
        max_width=max_width,
    )
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=creationflags,
    )
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if line:
            log(line.rstrip())
        if should_stop():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError("Generation stopped by user.")
        if line == "" and process.poll() is not None:
            break
    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"生成失败，退出码 {exit_code}: {video}")
    scripts = collect_generated_scripts(prediction_dir)
    if not scripts:
        raise RuntimeError(f"生成完成但没有找到 funscript: {prediction_dir}")
    return scripts


class OsrGeneratorGui:
    def __init__(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk
            from tkinter.scrolledtext import ScrolledText
        except ImportError as exc:
            raise RuntimeError("当前 Python 环境缺少 Tkinter，无法启动 Windows GUI。") from exc

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.ttk = ttk
        self.ScrolledText = ScrolledText

        self.root = tk.Tk()
        self.root.geometry("980x680")
        self.root.minsize(860, 560)

        self.videos: list[Path] = []
        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.language = tk.StringVar(value=default_language())
        self.translatable_widgets: list[tuple[object, str, str]] = []
        self.output_mode = tk.StringVar(value=OUTPUT_SAME_DIR)
        self.recursive_folder = tk.BooleanVar(value=True)
        self.preset_path = tk.StringVar(value=str((project_root() / DEFAULT_PRESET).resolve()))
        self.device = tk.StringVar(value=DEFAULT_DEVICE)
        self.device_label = tk.StringVar(value="")
        self.device_label_to_value: dict[str, str] = {}
        self.speed_profile = tk.StringVar(value=SPEED_QUALITY)
        self.speed_label = tk.StringVar(value="")
        self.speed_label_to_value: dict[str, str] = {}
        self.status_text = tk.StringVar(value=self.t("status_select_videos"))
        self.progress_text = tk.StringVar(value="0 / 0")
        self.output_mode.trace_add("write", lambda *_args: self._refresh_table())
        self.language.trace_add("write", lambda *_args: self.apply_language())

        self._build_ui()
        self.apply_language()
        self.root.after(LOG_POLL_MS, self._drain_log_queue)

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        top = ttk.Frame(self.root, padding=(12, 12, 12, 6))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(5, weight=1)

        self._track(ttk.Button(top, command=self.add_videos), "add_videos").grid(row=0, column=0, padx=(0, 8))
        self._track(ttk.Button(top, command=self.add_folder), "add_folder").grid(row=0, column=1, padx=(0, 8))
        self._track(ttk.Checkbutton(top, variable=self.recursive_folder), "recursive").grid(row=0, column=2, padx=(0, 14))
        self._track(ttk.Button(top, command=self.remove_selected), "remove_selected").grid(row=0, column=3, padx=(0, 8))
        self._track(ttk.Button(top, command=self.clear_videos), "clear").grid(row=0, column=4, sticky="w")
        language_frame = ttk.Frame(top)
        language_frame.grid(row=0, column=5, sticky="e")
        self._track(ttk.Label(language_frame), "language").grid(row=0, column=0, padx=(0, 6))
        language_combo = ttk.Combobox(
            language_frame,
            state="readonly",
            width=12,
            values=[LANGUAGE_NAMES[key] for key in ("zh", "en", "ja")],
        )
        language_combo.grid(row=0, column=1)
        language_combo.set(LANGUAGE_NAMES[self.language.get()])
        language_combo.bind("<<ComboboxSelected>>", lambda event: self._set_language_from_name(language_combo.get()))
        self.language_combo = language_combo

        options = self._track(ttk.LabelFrame(self.root, padding=(12, 8)), "output_settings")
        options.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        options.columnconfigure(1, weight=1)

        self._track(ttk.Radiobutton(
            options,
            variable=self.output_mode,
            value=OUTPUT_SAME_DIR,
        ), "output_same_dir").grid(row=0, column=0, sticky="w", padx=(0, 20))
        self._track(ttk.Radiobutton(
            options,
            variable=self.output_mode,
            value=OUTPUT_SAME_NAME_FOLDER,
        ), "output_same_name_folder").grid(row=0, column=1, sticky="w")

        self._track(ttk.Label(options), "preset").grid(row=1, column=0, sticky="w", pady=(8, 0))
        preset_entry = ttk.Entry(options, textvariable=self.preset_path)
        preset_entry.grid(row=1, column=1, sticky="ew", pady=(8, 0), padx=(8, 8))
        self._track(ttk.Button(options, command=self.choose_preset), "browse").grid(row=1, column=2, pady=(8, 0))

        self._track(ttk.Label(options), "device").grid(row=2, column=0, sticky="w", pady=(8, 0))
        device_combo = ttk.Combobox(
            options,
            textvariable=self.device_label,
            state="readonly",
            width=42,
        )
        device_combo.grid(row=2, column=1, sticky="w", pady=(8, 0), padx=(8, 8))
        device_combo.bind("<<ComboboxSelected>>", lambda _event: self._set_device_from_label())
        self.device_combo = device_combo
        self._refresh_device_options()

        self._track(ttk.Label(options), "speed").grid(row=3, column=0, sticky="w", pady=(8, 0))
        speed_combo = ttk.Combobox(
            options,
            textvariable=self.speed_label,
            state="readonly",
            width=28,
        )
        speed_combo.grid(row=3, column=1, sticky="w", pady=(8, 0), padx=(8, 8))
        speed_combo.bind("<<ComboboxSelected>>", lambda _event: self._set_speed_from_label())
        self.speed_combo = speed_combo
        self._refresh_speed_options()

        center = ttk.Frame(self.root, padding=(12, 0, 12, 6))
        center.grid(row=2, column=0, sticky="nsew")
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=3)
        center.rowconfigure(1, weight=2)

        columns = ("path", "output")
        self.video_table = ttk.Treeview(center, columns=columns, show="headings", selectmode="extended")
        self.video_table.column("path", width=560, anchor="w")
        self.video_table.column("output", width=320, anchor="w")
        self.video_table.grid(row=0, column=0, sticky="nsew")
        table_scroll = ttk.Scrollbar(center, orient="vertical", command=self.video_table.yview)
        self.video_table.configure(yscrollcommand=table_scroll.set)
        table_scroll.grid(row=0, column=1, sticky="ns")

        self.log_box = self.ScrolledText(center, height=12, wrap="word")
        self.log_box.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        self.log_box.configure(state="disabled")

        bottom = ttk.Frame(self.root, padding=(12, 4, 12, 12))
        bottom.grid(row=3, column=0, sticky="ew")
        bottom.columnconfigure(2, weight=1)

        self.start_button = self._track(ttk.Button(bottom, command=self.start_generation), "start")
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_button = self._track(ttk.Button(bottom, command=self.stop_generation, state="disabled"), "stop")
        self.stop_button.grid(row=0, column=1, sticky="w", padx=(0, 10))
        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.grid(row=0, column=2, sticky="ew", padx=(0, 10))
        ttk.Label(bottom, textvariable=self.progress_text).grid(row=0, column=3, padx=(0, 12))
        ttk.Label(bottom, textvariable=self.status_text).grid(row=0, column=4, sticky="e")

    def add_videos(self) -> None:
        paths = self.filedialog.askopenfilenames(
            title=self.t("choose_videos"),
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.webm *.avi *.mov *.m4v"),
                ("All files", "*.*"),
            ],
        )
        self._add_paths(paths)

    def add_folder(self) -> None:
        folder = self.filedialog.askdirectory(title=self.t("choose_folder"))
        if not folder:
            return
        self._add_paths(scan_video_folder(folder, recursive=bool(self.recursive_folder.get())))

    def choose_preset(self) -> None:
        path = self.filedialog.askopenfilename(
            title=self.t("choose_preset"),
            initialdir=str(project_root() / "configs" / "presets"),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.preset_path.set(path)

    def _add_paths(self, paths: Iterable[str | Path]) -> None:
        existing = {str(path).lower() for path in self.videos}
        added = 0
        for path in normalize_video_paths(paths):
            key = str(path).lower()
            if key in existing:
                continue
            self.videos.append(path)
            existing.add(key)
            added += 1
        if added:
            self._refresh_table()
            self.status_text.set(self.t("status_added", count=added, total=len(self.videos)))

    def remove_selected(self) -> None:
        selected = set(self.video_table.selection())
        if not selected:
            return
        self.videos = [path for index, path in enumerate(self.videos) if str(index) not in selected]
        self._refresh_table()

    def clear_videos(self) -> None:
        self.videos.clear()
        self._refresh_table()
        self.status_text.set(self.t("status_cleared"))

    def _refresh_table(self) -> None:
        for item in self.video_table.get_children():
            self.video_table.delete(item)
        mode = self.output_mode.get()
        for index, video in enumerate(self.videos):
            self.video_table.insert(
                "",
                "end",
                iid=str(index),
                values=(str(video), str(final_output_dir_for(video, mode))),
            )
        total = len(self.videos)
        self.progress.configure(maximum=max(total, 1), value=0)
        self.progress_text.set(f"0 / {total}")

    def start_generation(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if not self.videos:
            self.messagebox.showwarning(self.t("no_videos_title"), self.t("no_videos_message"))
            return
        preset = Path(self.preset_path.get()).expanduser().resolve()
        if not preset.is_file():
            self.messagebox.showerror(
                self.t("preset_missing_title"),
                self.t("preset_missing_message", path=preset),
            )
            return
        self.stop_event.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.configure(maximum=len(self.videos), value=0)
        self.progress_text.set(f"0 / {len(self.videos)}")
        self.status_text.set(self.t("status_generating"))
        self._append_log("")
        self._append_log(self.t("log_preset", preset=preset))
        self._set_device_from_label()
        self._set_speed_from_label()
        device = self.device.get()
        analysis_fps, max_width = speed_overrides(self.speed_profile.get())
        device_label = self.device_label.get() or device
        self._append_log(self.t("log_device", device=device_label))
        self._append_log(self.t("log_speed", speed=self.speed_label.get() or self.speed_profile.get()))
        self._append_log(self.t("log_output_mode", mode=self.output_mode.get()))
        videos = list(self.videos)
        mode = self.output_mode.get()
        self.worker = threading.Thread(
            target=self._generation_worker,
            args=(videos, preset, device, analysis_fps, max_width, mode),
            daemon=True,
        )
        self.worker.start()

    def stop_generation(self) -> None:
        self.stop_event.set()
        self.status_text.set(self.t("status_stopping"))

    def _generation_worker(
        self,
        videos: list[Path],
        preset: Path,
        device: str,
        analysis_fps: float | None,
        max_width: int | None,
        mode: str,
    ) -> None:
        success = 0
        try:
            for index, video in enumerate(videos, start=1):
                if self.stop_event.is_set():
                    break
                self.log_queue.put(("log", self.t("log_start", index=index, total=len(videos), video=video)))
                started = time.time()
                result = run_generation_for_video(
                    video,
                    preset=preset,
                    device=device,
                    analysis_fps=analysis_fps,
                    max_width=max_width,
                    output_mode=mode,
                    log=lambda message: self.log_queue.put(("log", f"  {message}")),
                    should_stop=self.stop_event.is_set,
                )
                elapsed = time.time() - started
                success += 1
                self.log_queue.put(
                    (
                        "log",
                        self.t(
                            "log_done",
                            index=index,
                            total=len(videos),
                            count=len(result.scripts),
                            output=result.output_dir,
                            elapsed=elapsed,
                        ),
                    )
                )
                self.log_queue.put(("progress", index))
        except Exception as exc:
            self.log_queue.put(("error", str(exc)))
        finally:
            stopped = self.stop_event.is_set()
            self.log_queue.put(("done", (success, len(videos), stopped)))

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, value = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(value))
                elif kind == "progress":
                    current = int(value)
                    self.progress.configure(value=current)
                    self.progress_text.set(f"{current} / {len(self.videos)}")
                elif kind == "error":
                    self._append_log(self.t("error_prefix", message=value))
                    self.messagebox.showerror(self.t("generation_failed"), str(value))
                elif kind == "done":
                    success, total, stopped = value  # type: ignore[misc]
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    if stopped:
                        self.status_text.set(self.t("status_stopped", success=success, total=total))
                    else:
                        if should_clear_video_queue_after_run(stopped=stopped, success=success, total=total):
                            self.videos.clear()
                            self._refresh_table()
                        self.status_text.set(self.t("status_done", success=success, total=total))
                    self.progress_text.set(f"{success} / {total}")
        except queue.Empty:
            pass
        self.root.after(LOG_POLL_MS, self._drain_log_queue)

    def _append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def run(self) -> None:
        self.root.mainloop()

    def t(self, key: str, **kwargs: object) -> str:
        language = self.language.get() if hasattr(self, "language") else default_language()
        return translated_text(language, key, **kwargs)

    def _track(self, widget: object, key: str, option: str = "text"):
        self.translatable_widgets.append((widget, key, option))
        return widget

    def apply_language(self) -> None:
        language = self.language.get()
        self.root.title(self.t("title"))
        for widget, key, option in self.translatable_widgets:
            widget.configure(**{option: self.t(key)})  # type: ignore[attr-defined]
        self.video_table.heading("path", text=self.t("video"))
        self.video_table.heading("output", text=self.t("output"))
        if hasattr(self, "language_combo"):
            self.language_combo.set(LANGUAGE_NAMES.get(language, LANGUAGE_NAMES["en"]))
        self._refresh_device_options()
        self._refresh_speed_options()

    def _set_language_from_name(self, name: str) -> None:
        for key, label in LANGUAGE_NAMES.items():
            if label == name:
                self.language.set(key)
                return

    def _refresh_device_options(self) -> None:
        if not hasattr(self, "device_combo"):
            return
        current_value = self.device.get() or DEFAULT_DEVICE
        options = available_device_options(self.language.get())
        labels = [option.label for option in options]
        self.device_label_to_value = {option.label: option.value for option in options}
        self.device_combo.configure(values=labels)
        selected = next((option.label for option in options if option.value == current_value), labels[0])
        self.device_label.set(selected)
        self.device.set(self.device_label_to_value.get(selected, DEFAULT_DEVICE))

    def _set_device_from_label(self) -> None:
        self.device.set(self.device_label_to_value.get(self.device_label.get(), DEFAULT_DEVICE))

    def _refresh_speed_options(self) -> None:
        if not hasattr(self, "speed_combo"):
            return
        current_value = self.speed_profile.get() or SPEED_QUALITY
        options = format_speed_options(self.language.get())
        labels = [option.label for option in options]
        self.speed_label_to_value = {option.label: option.value for option in options}
        self.speed_combo.configure(values=labels)
        selected = next((option.label for option in options if option.value == current_value), labels[0])
        self.speed_label.set(selected)
        self.speed_profile.set(self.speed_label_to_value.get(selected, SPEED_QUALITY))

    def _set_speed_from_label(self) -> None:
        self.speed_profile.set(self.speed_label_to_value.get(self.speed_label.get(), SPEED_QUALITY))


def main() -> int:
    app = OsrGeneratorGui()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
