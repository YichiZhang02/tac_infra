# Usage

完整能力说明见 [repository_capabilities_and_training.md](repository_capabilities_and_training.md)。

## Policy 训练命令格式

```bash
bash train.sh <dataset_id> <policy_type> <num_processes> <batch_size> <steps> <wrist_only> <tactile_mode> <state_mode>
```

示例：

```bash
bash train.sh rm_nist_260320_strawberry diffusion 8 32 5000 false as_image joint
bash train.sh rm_nist_260320_strawberry act 8 32 5000 true none joint
bash train.sh rm_nist_260320_strawberry pi05 8 6 6500 false as_image joint
bash train.sh rm_nist_260320_strawberry starvla_groot 8 8 5000 false none joint
```

## Tactile-MAE 训练命令格式

```bash
bash train_enc.sh <dataset_ids> <init_mode> <arch> <num_processes> <batch_size> <epochs>
```

示例：

```bash
bash train_enc.sh "rm_nist_260520_usb rm_nist_260323_plug2_raw" clip vit_b 8 128 100
bash train_enc.sh "rm_nist_260520_usb rm_nist_260323_plug2_raw" anytouch vit_l 8 128 100
```

## 路由参数

```bash
--policy.wrist_only=false
--policy.tactile_mode=none|as_image|encode
--policy.state_mode=joint|none|ee
```

`tactile_mode=encode` 还需要：

```bash
--policy.tactile_encoder_path=<checkpoint>
--policy.tactile_insert_location=encoder|decoder
--policy.tactile_num_tokens=8
```

## 输出目录

- policy checkpoint：`playground/results/models/`
- policy 日志：`playground/logs/models/`
- tactile backbone checkpoint：`playground/results/backbones/`
- tactile backbone 日志：`playground/logs/backbones/`
