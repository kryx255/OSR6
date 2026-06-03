# OSRGen Experiments

[English](#english) | [中文简体](#中文简体) | [日本語](#日本語)

## English

This page records practical notes for the packaged runtime model.

### Default Model

- Preset: `configs/presets/region_hybrid_experience_95.json`
- Checkpoints: `models/region_all_profile_95_e20/`
- Postprocess profile: `configs/postprocess/region_all_quality_95.json`
- Runtime target: six-axis OSR6/SR6 script generation
- Device default: `auto`, using CUDA for TCN inference when PyTorch reports an available CUDA device

### Why This Preset

Manual review preferred weak but active secondary-axis motion over nearly static conservative output. The final preset is experience-first:

- `L0` amplitude scale: `1.25`
- Secondary axes: `1.0`
- Quality gate: `warn`, so scripts are not automatically removed or neutralized

### Known Limits

- `L0` is the strongest axis; secondary axes can be less consistent.
- The packaged model is motion-feature based, not a semantic video-understanding model.
- Generated scripts should still be tested by a user, especially multi-axis output.

## 中文简体

这个页面记录当前打包模型的实际使用备注。

### 当前默认模型

- Preset：`configs/presets/region_hybrid_experience_95.json`
- Checkpoint：`models/region_all_profile_95_e20/`
- Profile：`configs/postprocess/region_all_quality_95.json`
- 运行目标：生成六轴 OSR6/SR6 脚本
- 设备默认值：`auto`，当 PyTorch 检测到 CUDA 可用时，TCN 推理使用 CUDA

### 选择原因

人工试片反馈显示，弱轴“有动作”优于过度保守的近静态输出。最终 preset 因此采用体验优先策略：

- `L0` 振幅放大到 `1.25`
- 副轴保留 `1.0`
- 质量门控保持 `warn`，不自动删除或中和弱轴脚本

### 已知限制

- 主轴 `L0` 相对稳定，副轴稳定性仍弱一些。
- 模型不是语义理解模型，主要依靠光流和固定区域运动信号。
- 输出仍建议人工回听/测试，尤其是副轴。

## 日本語

このページは同梱モデルの実用メモです。

### 既定モデル

- Preset: `configs/presets/region_hybrid_experience_95.json`
- Checkpoint: `models/region_all_profile_95_e20/`
- 後処理 profile: `configs/postprocess/region_all_quality_95.json`
- 実行対象: 六軸 OSR6/SR6 スクリプト生成
- デバイス既定値: `auto`。PyTorch が CUDA を検出した場合、TCN 推論に CUDA を使います。

### 採用理由

手動テストでは、ほぼ静止した保守的な副軸よりも、弱くても動きがある副軸の方が体験として良好でした。そのため既定 preset は体験優先です。

- `L0` は `1.25` に拡大
- 副軸は `1.0`
- quality gate は `warn`

### 制限

- `L0` が最も安定しています。
- 副軸はまだ安定しない場合があります。
- 生成結果は実機またはプレイヤーで確認することを推奨します。
