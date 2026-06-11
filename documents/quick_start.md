# Quick Start

完整说明见 [repository_capabilities_and_training.md](repository_capabilities_and_training.md)。

## 1. 环境

```bash
conda create -n vtla python=3.10 -y
conda activate vtla
pip install -r requirement_260608.txt
```

如果仓库不在 `/mnt/data/xidong_data/tac_infra`，先修改 `train.sh` 和 `train_enc.sh` 顶部的 `REPO_ROOT`。

## 2. 训练 tactile backbone

```bash
bash train_enc.sh rm_nist_260320_strawberry anytouch vit_l 4 128 100
```

输出默认在：

```text
playground/results/backbones/
playground/logs/backbones/
```

## 3. 训练 policy

```bash
# Diffusion baseline: top+wrist，触觉不用，joint state
bash train.sh rm_nist_260320_strawberry diffusion 8 32 5000 false none joint

# Diffusion + tactile as image
bash train.sh rm_nist_260320_strawberry diffusion 8 32 5000 false as_image joint

# ACT + tactile as image
bash train.sh rm_nist_260320_strawberry act 8 32 5000 false as_image joint
```

使用 Tactile-MAE encoder：

```bash
export TACTILE_ENCODER_PATH=playground/results/backbones/<run>/checkpoints/best.pth
export TACTILE_INSERT_LOCATION=encoder
export TACTILE_NUM_TOKENS=8

bash train.sh rm_nist_260320_strawberry act 8 32 5000 false encode joint
```

## 4. 关键开关

- `policy_type`: `act`、`diffusion`、`pi05`、`starvla_groot`
- `wrist_only`: `true` 只用 wrist；`false` 用 top + wrist
- `tactile_mode`: `none`、`as_image`、`encode`
- `state_mode`: `joint`、`none`；`ee` 预留未实现
