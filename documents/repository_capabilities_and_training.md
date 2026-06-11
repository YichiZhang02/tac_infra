# VTLA 仓库能力、Quick Start 与训练说明

本文档按当前仓库实现整理，覆盖：支持的模型、视觉/触觉/state 路由、触觉 backbone 预训练、policy 训练，以及常用命令。

## 1. 仓库定位

这个仓库是一个基于 LeRobot 数据格式的 VTLA 训练框架。主训练入口是 `vtla.train`，数据从 `playground/data/<dataset_id>` 读取，训练结果默认写到 `playground/results`，日志默认写到 `playground/logs`。

核心模块：

- `vtla/train.py`：离线 supervised fine-tuning 训练主入口，使用 `accelerate`。
- `vtla/datasets/`：LeRobot dataset 读取、metadata、采样、统计。
- `vtla/frameworks/`：policy/model 实现。
- `vtla/frameworks/sensor_routing.py`：四类 policy 共用的视觉、触觉、state 路由配置。
- `vtla/frameworks/tactile_encode.py`：policy 训练时使用的 Tactile-MAE 触觉 token encoder。
- `vtla/tac_encoder/tactile_mae/`：触觉 backbone 预训练、评估和推理封装。
- `train.sh`：policy 训练启动器。
- `train_enc.sh`：触觉 backbone 预训练启动器。

## 2. 当前支持的 policy / 模型

仓库内通过 `--policy.type` 支持以下四类 policy：

| policy.type | 模型路线 | 主要文件 | 说明 |
|---|---|---|---|
| `act` | ACT / Action Chunking Transformer | `vtla/frameworks/act/` | ResNet 视觉 backbone + Transformer encoder/decoder + action chunk。 |
| `diffusion` | Diffusion Policy | `vtla/frameworks/diffusion/` | ResNet 视觉 encoder + Conditional 1D U-Net；支持 DDPM/DDIM。 |
| `pi05` | PI0.5 / PaliGemma + action expert | `vtla/frameworks/pi05/` | PaliGemma/SigLIP 视觉语言 prefix + flow matching action expert；可从 `pi05_base` 加载。 |
| `starvla_groot` | Qwen-VL + GR00T action head | `vtla/frameworks/starvla_groot/` | Qwen-VL prefix encoder + GR00T flow-matching DiT action head。 |

默认启动器里的 pretrained / base 模型约定：

- `act`、`diffusion`：默认从零训练。
- `pi05`：默认 `--policy.pretrained_path=playground/pretrained_models/pi05_base`，即完整 PI05 policy checkpoint。
- `starvla_groot`：默认 `--policy.base_vlm=playground/pretrained_models/Qwen3.5-0.8B`，policy/action head 从当前配置训练。

## 3. 统一传感器路由能力

四类 policy 共享 `SensorRoutingMixin`，核心参数如下。

### 3.1 相机选择

```bash
--policy.wrist_only=false
--policy.top_camera_key=observation.images.cam_top
--policy.wrist_camera_key=observation.images.cam_right_wrist
```

- `wrist_only=false`：使用 top + wrist 两路 RGB 相机。
- `wrist_only=true`：只使用 wrist 相机。
- 未被选中的视觉 feature 会在 config validation 阶段从 `input_features` 里剪掉。

### 3.2 触觉引入方式

```bash
--policy.tactile_mode=none      # none | as_image | encode
--policy.tactile_keys='["observation.images.cam_finger0","observation.images.cam_finger1"]'
```

支持三种模式：

| tactile_mode | 行为 |
|---|---|
| `none` | 不使用触觉图像。 |
| `as_image` | 把 finger tactile 图像当作额外视觉图像输入模型的视觉/VLM 路径。 |
| `encode` | finger tactile 图像不进入普通视觉路径，而是进入独立 Tactile-MAE encoder，输出 query tokens，再注入 policy。 |

`encode` 模式需要额外提供：

```bash
--policy.tactile_encoder_path=<tactile-mae checkpoint 或 HF dir>
--policy.tactile_insert_location=encoder    # encoder | decoder
--policy.tactile_num_tokens=8
--policy.freeze_tactile_encoder=false
```

`tactile_num_tokens` 是每张触觉图输出的 query token 数。默认两路 finger 图像时，总触觉 token 数是 `2 * tactile_num_tokens`。

当前各模型的 `encode` 接入方式：

| policy | `encode` 接入位置 |
|---|---|
| `act` | `encoder`：触觉 token 进入 ACT observation encoder；`decoder`：触觉 token 追加到 decoder cross-attention memory。 |
| `diffusion` | 触觉 token 展平后加入 global conditioning；`tactile_insert_location` 对 Diffusion 无实际区别。 |
| `pi05` | 触觉 token 投影到 PaliGemma width；`encoder` 时放在 image 与 language prefix 之间，`decoder` 时作为 trailing condition block。 |
| `starvla_groot` | 触觉 token 追加到 Qwen-VL hidden states，供 GR00T action head cross-attend；`encoder` 当前会 warning，并走与 `decoder` 相同的 action-head 条件路径。 |

### 3.3 state 选择

```bash
--policy.state_mode=joint    # none | joint | ee
```

| state_mode | 行为 |
|---|---|
| `joint` | 使用 `observation.state` 关节状态。默认模式。 |
| `none` | 不使用 `observation.state`。 |
| `ee` | 预留给末端位姿 conditioning，当前未实现，会抛 `NotImplementedError`。 |

注意：

- `pi05` 在 `state_mode=joint` 下会把 state padding 到 `max_state_dim`，并通过 processor 进入 prompt/tokenizer 流程。
- `pi05 --policy.use_relative_actions=true` 依赖 state，因此不能和 `state_mode=none` 同时使用。

## 4. 触觉 backbone：Tactile-MAE

触觉 backbone 位于 `vtla/tac_encoder/tactile_mae/`，是 AnyTouch stage-1 风格的 tactile-image masked autoencoder。

支持能力：

- 直接读取 LeRobot 数据集，不需要先转换数据格式。
- 支持 `vit_l` 和 `vit_b`。
- 支持三种初始化：
  - `scratch`：随机初始化。
  - `clip`：从 CLIP ViT 初始化 encoder/projection。
  - `anytouch`：从 AnyTouch ViT-L 权重 strict load。
- 支持触觉接触帧筛选：基于 per-channel std，默认启动器中开启。
- 支持多数据集联合训练。
- 训练出的 checkpoint 可直接作为 policy 的 `--policy.tactile_encoder_path`。

训练启动器：

```bash
bash train_enc.sh <dataset_ids> <init_mode> <arch> <num_processes> <batch_size> <epochs>
```

示例：

```bash
# 单数据集，AnyTouch ViT-L 初始化
bash train_enc.sh rm_nist_260320_strawberry anytouch vit_l 4 128 100

# 多数据集，CLIP ViT-B 初始化
bash train_enc.sh "rm_nist_260520_usb rm_nist_260323_plug2_raw" clip vit_b 8 128 100

# 从零训练 ViT-L
bash train_enc.sh "rm_nist_260520_usb rm_nist_260323_plug2_raw" scratch vit_l 8 128 100
```

常用环境变量：

```bash
GPU_ID=0,1,2,3
RUN_NAME=2026_0610_test
FINGER_CAMS="observation.images.cam_finger0 observation.images.cam_finger1"
CONTACT_FILTER=1
CONTACT_STD_THRESHOLD=0.5
NONCONTACT_KEEP_RATIO=0.05
AMP_DTYPE=bfloat16
```

输出位置：

- checkpoint：`playground/results/backbones/<run_tag>_tacmae_<arch>_from_<init_mode>/checkpoints/`
- 日志：`playground/logs/backbones/`
- 使用过的数据集列表：`datasets.txt`

## 5. Policy 训练 Quick Start

### 5.1 准备环境

建议在 conda 环境中安装依赖：

```bash
conda create -n vtla python=3.10 -y
conda activate vtla
pip install -r requirement_260608.txt
```

训练脚本默认假设：

- 仓库路径：`/mnt/data/xidong_data/tac_infra`
- 数据路径：`playground/data/<dataset_id>`
- 预训练模型路径：`playground/pretrained_models/`

如果路径不同，先修改 `train.sh` 和 `train_enc.sh` 顶部的 `REPO_ROOT`。

### 5.2 最小 policy 训练命令

```bash
bash train.sh <dataset_id> <policy_type> <num_processes> <batch_size> <steps> <wrist_only> <tactile_mode> <state_mode>
```

示例：

```bash
# Diffusion，top+wrist+finger 触觉图像，使用 joint state
bash train.sh rm_nist_260320_strawberry diffusion 8 32 5000 false as_image joint

# ACT，只用 RGB，不用触觉，使用 joint state
bash train.sh rm_nist_260320_strawberry act 8 32 5000 false none joint

# PI05，触觉作为额外图像输入
bash train.sh rm_nist_260320_strawberry pi05 8 6 6500 false as_image joint

# StarVLA-GR00T，触觉作为额外图像输入
bash train.sh rm_nist_260320_strawberry starvla_groot 8 8 5000 false as_image joint
```

### 5.3 只用 wrist 相机

```bash
bash train.sh rm_nist_260320_strawberry diffusion 8 32 5000 true none joint
```

等价于：

- RGB 输入只保留 `observation.images.cam_right_wrist`。
- `cam_top` 和未选中的视觉 feature 会被剪掉。

### 5.4 不使用 state

```bash
bash train.sh rm_nist_260320_strawberry act 8 32 5000 false as_image none
```

注意：`pi05` 如果开启 `use_relative_actions=true`，不能使用 `state_mode=none`。

### 5.5 使用 Tactile-MAE encoder 触觉分支

先训练或准备 tactile-MAE checkpoint，然后：

```bash
export TACTILE_ENCODER_PATH=playground/results/backbones/2026_0610_tacmae_vit_l_from_anytouch/checkpoints/best.pth
export TACTILE_INSERT_LOCATION=encoder
export TACTILE_NUM_TOKENS=8

bash train.sh rm_nist_260320_strawberry act 8 32 5000 false encode joint
```

也可以把 checkpoint 作为第 9 个位置参数传入：

```bash
bash train.sh rm_nist_260320_strawberry diffusion 8 32 5000 false encode joint \
  playground/results/backbones/2026_0610_tacmae_vit_b_from_clip/checkpoints/best.pth \
  decoder
```

## 6. 训练输出和恢复

`train.sh` 默认输出：

- 模型 checkpoint：`playground/results/models/<dataset>_<policy>_<routing>/`
- 训练日志：`playground/logs/models/<dataset>_<policy>_<routing>.log`

`vtla.train` 会保存：

- policy 权重。
- `train_config.json`。
- preprocessor / postprocessor。
- optimizer / scheduler training state。

恢复训练可直接使用 `vtla.train --resume=true --config_path=<.../train_config.json>`。当前 `train.sh` 没有封装 resume 参数，如需恢复建议直接调用 `accelerate launch -m vtla.train` 并传入对应配置。

## 7. 常用批量训练脚本

仓库提供了两个批量示例：

```bash
bash train_all.sh
bash train_enc_all.sh
```

`train_all.sh` 覆盖 `act`、`diffusion`、`starvla_groot`、`pi05` 的 `as_image` 和 `none` 触觉模式示例。

`train_enc_all.sh` 覆盖 tactile-MAE 的 `vit_b/vit_l` 和 `scratch/clip/anytouch` 初始化示例。

## 8. Smoke tests

仓库里有两个离线 smoke test：

```bash
python smoke_test_frameworks.py
python smoke_test_tactile_encode.py <tactile_mae_checkpoint>
```

用途：

- `smoke_test_frameworks.py`：检查四类 framework 在不同 `wrist_only/tactile_mode/state_mode` 组合下能 forward/backward。
- `smoke_test_tactile_encode.py`：检查 `tactile_mode=encode`、Tactile-MAE query tokens、冻结/非冻结、插入位置等路径。

## 9. 推荐实验矩阵

先跑低成本 baseline：

```bash
bash train.sh <dataset> diffusion 8 32 5000 false none joint
bash train.sh <dataset> diffusion 8 32 5000 false as_image joint
bash train.sh <dataset> act       8 32 5000 false none joint
bash train.sh <dataset> act       8 32 5000 false as_image joint
```

再训练 tactile backbone 并接入：

```bash
bash train_enc.sh "<dataset1> <dataset2>" clip vit_b 8 128 100
export TACTILE_ENCODER_PATH=<best.pth>
bash train.sh <dataset> diffusion 8 32 5000 false encode joint
bash train.sh <dataset> act       8 32 5000 false encode joint
```

最后评估 VLM policy：

```bash
bash train.sh <dataset> pi05          8 6 6500 false as_image joint
bash train.sh <dataset> starvla_groot 8 8 5000 false as_image joint
```

如果显存紧张，优先尝试：

- `wrist_only=true`
- 降低 `batch_size`
- 对 `pi05/starvla_groot` 开启 `train_expert_only=true` 或 `freeze_vision_encoder=true`
- 先用 `act/diffusion` 验证数据和路由正确性

## 10. 当前限制

- `state_mode=ee` 当前未实现。
- `starvla_groot tactile_insert_location=encoder` 还没有真正插入 Qwen-VL 输入 embedding；目前会 warning，并将触觉 token 追加到 action head cross-attention memory。
- `diffusion` 没有 encoder/decoder token split，因此 `tactile_insert_location` 不影响实际路径。
- `train.sh` 顶部 `REPO_ROOT` 是硬编码路径，迁移机器时需要修改。
- 多数据集 policy 训练在 `TrainPipelineConfig.validate()` 中标记为未实现；多数据集已用于 tactile-MAE 预训练。
