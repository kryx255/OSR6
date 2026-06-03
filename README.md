# OSRGen

[English](#english) | [中文简体](#中文简体) | [日本語](#日本語)

## English

OSRGen is a Windows-friendly video-to-OSR6/SR6 funscript generator. This repository contains the GUI, CLI, packaged checkpoints, and runtime config needed to generate scripts.

### Documentation

- [Architecture](ARCHITECTURE.md#english)
- [Experiments](EXPERIMENTS.md#english)

### Install

1. Install Python 3.10 or newer.
2. Run `install.bat`.
3. Run `launch_osrgen_gui.bat`.

Manual install with `uv`:

```powershell
uv sync --extra model
.\launch_osrgen_gui.bat
```

### GUI

- Add one or many videos.
- Add a folder of videos, optionally recursive.
- Write scripts next to each video, or create a same-name output folder next to each video.
- Choose inference device: `auto`, `cpu`, `cuda`, or a specific device such as `cuda:0`.
- The queue is cleared after a fully successful run, so repeated clicks do not regenerate previous videos.
- The GUI supports Simplified Chinese, English, and Japanese.

### CLI

Single video:

```powershell
.\run_osrgen.bat model predict-all .\videos\sample.mp4 --preset .\configs\presets\region_hybrid_experience_95.json --output .\outputs
```

Use a specific inference device:

```powershell
.\run_osrgen.bat model predict-all .\videos\sample.mp4 --device cuda:0
```

Batch:

```powershell
.\run_osrgen.bat model batch-predict --input-dir .\videos --preset .\configs\presets\region_hybrid_experience_95.json --recursive --output .\outputs\batch_predict
```

Preset validation:

```powershell
.\run_osrgen.bat model validate-preset --preset .\configs\presets\region_hybrid_experience_95.json
```

### Release Contents

- `src/osrgen/gui.py`: Windows GUI.
- `src/osrgen/cli.py`: runtime CLI.
- `src/osrgen/modeling.py`: final TCN checkpoint inference.
- `models/region_all_profile_95_e20/`: final six-axis model.
- `configs/presets/region_hybrid_experience_95.json`: default experience-first preset.
- `configs/postprocess/region_all_quality_95.json`: postprocess profile.

## 中文简体

OSRGen 是一个 Windows 友好的视频转 OSR6/SR6 funscript 生成器。这个仓库包含 GUI、命令行入口、模型 checkpoint 和生成脚本所需的运行配置。

### 文档

- [架构](ARCHITECTURE.md#中文简体)
- [实验摘要](EXPERIMENTS.md#中文简体)

### 安装

1. 安装 Python 3.10 或更新版本。
2. 双击 `install.bat`。
3. 双击 `launch_osrgen_gui.bat` 打开 GUI。

如果已经安装了 `uv`，也可以手动运行：

```powershell
uv sync --extra model
.\launch_osrgen_gui.bat
```

### GUI 使用

- `添加视频`：选择一个或多个视频。
- `添加文件夹`：批量加入文件夹里的视频，可勾选递归。
- 输出方式可以选择直接生成在视频同目录，或在视频同目录创建同名文件夹。
- 可选择推理设备：`auto`、`cpu`、`cuda`，或指定 `cuda:0`。
- 生成成功后，当前待生成列表会自动清空，避免重复生成同一批视频。
- GUI 支持中文、English、日本語，可在右上角切换。

### 命令行

生成单个视频：

```powershell
.\run_osrgen.bat model predict-all .\videos\sample.mp4 --preset .\configs\presets\region_hybrid_experience_95.json --output .\outputs
```

指定推理设备：

```powershell
.\run_osrgen.bat model predict-all .\videos\sample.mp4 --device cuda:0
```

批量生成：

```powershell
.\run_osrgen.bat model batch-predict --input-dir .\videos --preset .\configs\presets\region_hybrid_experience_95.json --recursive --output .\outputs\batch_predict
```

检查模型和运行环境：

```powershell
.\run_osrgen.bat model validate-preset --preset .\configs\presets\region_hybrid_experience_95.json
```

### 发布包内容

- `src/osrgen/gui.py`：Windows GUI。
- `src/osrgen/cli.py`：运行版 CLI。
- `src/osrgen/modeling.py`：最终 TCN checkpoint 推理。
- `models/region_all_profile_95_e20/`：最终六轴模型。
- `configs/presets/region_hybrid_experience_95.json`：默认体验优先 preset。
- `configs/postprocess/region_all_quality_95.json`：动作后处理 profile。

## 日本語

OSRGen は Windows 向けの動画から OSR6/SR6 funscript を生成するツールです。このリポジトリには GUI、CLI、checkpoint、実行用設定が含まれます。

### ドキュメント

- [アーキテクチャ](ARCHITECTURE.md#日本語)
- [実験サマリー](EXPERIMENTS.md#日本語)

### インストール

1. Python 3.10 以降をインストールします。
2. `install.bat` を実行します。
3. `launch_osrgen_gui.bat` を実行します。

`uv` を使う場合：

```powershell
uv sync --extra model
.\launch_osrgen_gui.bat
```

### GUI

- 1つまたは複数の動画を追加できます。
- フォルダー内の動画を一括追加できます。
- スクリプトを動画と同じフォルダーに出力するか、動画と同名のフォルダーを作って出力できます。
- 推論デバイスを選択できます: `auto`、`cpu`、`cuda`、または `cuda:0`。
- 生成がすべて成功するとキューを自動でクリアします。
- GUI は簡体字中国語、英語、日本語に対応しています。

### CLI

単一動画：

```powershell
.\run_osrgen.bat model predict-all .\videos\sample.mp4 --preset .\configs\presets\region_hybrid_experience_95.json --output .\outputs
```

推論デバイス指定：

```powershell
.\run_osrgen.bat model predict-all .\videos\sample.mp4 --device cuda:0
```

一括生成：

```powershell
.\run_osrgen.bat model batch-predict --input-dir .\videos --preset .\configs\presets\region_hybrid_experience_95.json --recursive --output .\outputs\batch_predict
```

preset 検証：

```powershell
.\run_osrgen.bat model validate-preset --preset .\configs\presets\region_hybrid_experience_95.json
```
