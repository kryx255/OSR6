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
- Add and organize selected videos into same-name folders next to the original files.
- Add a folder of videos, optionally recursive.
- Write scripts next to each video, or create a same-name output folder next to each video.
- Choose an available inference device by name, such as CPU or a detected NVIDIA GPU.
- Choose speed/quality mode: Quality, Balanced, or Fast.
- Choose parallel jobs for batch generation.
- Optionally move each video into its output folder after successful generation.
- The queue is cleared after a fully successful run, so repeated clicks do not regenerate previous videos.
- The GUI supports Simplified Chinese, English, and Japanese.

### Speed

- OSRGen caches extracted video features locally and reuses them when the same video and analysis settings are generated again.
- The default Quality mode keeps the packaged preset settings. Balanced and Fast reduce optical-flow input size for faster first-time generation.
- Batch generation can process multiple videos at once. Use a modest worker count to avoid CPU/GPU oversubscription.
- The PyTorch device speeds up the TCN model step. OpenCV video decoding and optical-flow extraction are still mostly CPU-bound.
- Disable the local cache with `--no-feature-cache`, or set it with `--feature-cache-dir`.

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

Batch with parallel jobs:

```powershell
.\run_osrgen.bat model batch-predict --input-dir .\videos --workers 2 --recursive --output .\outputs\batch_predict
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
- `添加并整理`：选择视频后，按文件名移动到同目录的同名文件夹，并加入列表。
- `添加文件夹`：批量加入文件夹里的视频，可勾选递归。
- 输出方式可以选择直接生成在视频同目录，或在视频同目录创建同名文件夹。
- 可按名称选择本机可用推理设备，例如 CPU 或检测到的 NVIDIA 显卡。
- 可选择速度/质量档位：质量优先、均衡、快速。
- 批量生成时可选择并行任务数。
- 可选择生成成功后把视频移动到对应输出文件夹。
- 生成成功后，当前待生成列表会自动清空，避免重复生成同一批视频。
- GUI 支持中文、English、日本語，可在右上角切换。

### 加速

- OSRGen 会在本地缓存已抽取的视频特征；同一视频、同一分析设置再次生成时会复用缓存。
- 默认“质量优先”保持打包 preset 设置。“均衡”和“快速”会降低光流输入规模，加快第一次生成。
- 批量生成可以同时处理多个视频。建议使用适中的并行数，避免 CPU/GPU 过载。
- 推理设备会加速 TCN 模型步骤；OpenCV 视频解码和光流特征提取仍主要受 CPU 影响。
- 命令行可用 `--no-feature-cache` 关闭缓存，或用 `--feature-cache-dir` 指定缓存目录。

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

并行批量生成：

```powershell
.\run_osrgen.bat model batch-predict --input-dir .\videos --workers 2 --recursive --output .\outputs\batch_predict
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
- 選択した動画を元フォルダー内の同名フォルダーへ整理して追加できます。
- フォルダー内の動画を一括追加できます。
- スクリプトを動画と同じフォルダーに出力するか、動画と同名のフォルダーを作って出力できます。
- CPU や検出された NVIDIA GPU など、利用可能な推論デバイスを名前で選択できます。
- 速度/品質モードを選択できます: 品質、バランス、高速。
- 一括生成では並列ジョブ数を選択できます。
- 生成成功後に動画を対応する出力フォルダーへ移動できます。
- 生成がすべて成功するとキューを自動でクリアします。
- GUI は簡体字中国語、英語、日本語に対応しています。

### 高速化

- OSRGen は抽出済みの動画特徴をローカルにキャッシュし、同じ動画と解析設定で再生成すると再利用します。
- 既定の品質モードは packaged preset を維持します。バランスと高速は光フロー入力サイズを下げ、初回生成を速くします。
- 一括生成では複数の動画を同時に処理できます。CPU/GPU の過負荷を避けるため、控えめな worker 数を推奨します。
- PyTorch デバイスは TCN モデル部分を高速化します。OpenCV の動画デコードと光フロー抽出は主に CPU で処理されます。
- CLI では `--no-feature-cache` でキャッシュを無効化し、`--feature-cache-dir` で保存先を指定できます。

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

並列一括生成：

```powershell
.\run_osrgen.bat model batch-predict --input-dir .\videos --workers 2 --recursive --output .\outputs\batch_predict
```

preset 検証：

```powershell
.\run_osrgen.bat model validate-preset --preset .\configs\presets\region_hybrid_experience_95.json
```
