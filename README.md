# tac_infra

A private infra for VTLA.

包含三块: **backbone 预训练**(触觉 MAE encoder)、**VTLA 策略训练**(act / diffusion / pi05 / starvla_groot)、以及在睿尔曼双臂上的 **deployment**(数据采集 + 推理)。

所有运行期路径均为**相对路径**, 统一挂在 `playground/` 下, 跨机器可移植:

```
playground/
├── pretrained_models/   # 预训练权重 (CLIP / AnyTouch / pi05_base / 底座 VLM)
├── data/                # 训练 / 预训练数据集 (LeRobot 格式或裸 frame_cache)
├── results/
│   ├── backbones/       # train_enc.sh 输出 (触觉 MAE)
│   └── models/          # train.sh 输出 (VTLA 策略 checkpoint)
├── logs/                # 训练日志 (backbones/ 与 models/ 分目录)
└── eval/                # 推理录制的评测数据集
```

---

# Quick start

## 1) 环境

```bash
conda create -n vtla python=3.10 -y
conda activate vtla
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 2) 下载预训练权重 -> `playground/pretrained_models/`

训练前需把以下权重放到 `playground/pretrained_models/`(目录名保持下表一致, 脚本按名引用)。
推荐用 `huggingface-cli download <repo_id> --local-dir playground/pretrained_models/<目录名>`。

| 目录名 | 用途 | 来源 |
| --- | --- | --- |
| `CLIP-ViT-L-14-DataComp.XL-s13B-b90K` | 触觉 MAE 的 CLIP 初始化 (ViT-L) | https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K |
| `CLIP-ViT-B-16-DataComp.XL-s13B-b90K` | 触觉 MAE 的 CLIP 初始化 (ViT-B, 需 HF 格式) | https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K |
| `AnyTouch-ViT-L-16` | 触觉 MAE 的 AnyTouch 初始化 / VTLA 触觉 encoder (仅 ViT-L) | AnyTouch 官方发布 (见 https://github.com/GeWu-Lab/AnyTouch ); 下载后转成 HF 目录或直接用 `.pth` |
| `pi05_base` | pi05 策略底座 (完整 policy 检查点) | https://huggingface.co/lerobot/pi05_base |
| `Qwen3.5-0.8B` | starvla_groot 底座 VLM (HF 模型目录) | https://huggingface.co/Qwen/Qwen3.5-0.8B |

> 说明: `pi05_base` 是**完整 policy 检查点**(含 `model.safetensors` + 预/后处理器), 通过 `--policy.pretrained_path` 加载整个 policy; `Qwen3.5-0.8B` 是 starvla_groot 的**底座 VLM**, 通过 `--policy.base_vlm` 传入, policy 本身从零训练。
> AnyTouch 只有 ViT-L 权重; CLIP-B-16 需要 HF 格式(bundled 的 open_clip 权重需先转换)。

## 3) 放置数据集 -> `playground/data/`

每个数据集放成 `playground/data/<dataset_id>/`, 训练脚本按 `dataset_id` 引用。支持两类:

- **LeRobot 数据集**: 含 `meta/info.json`(VTLA 训练 / 触觉 MAE 都可用), 如 `rm_umi_dual_pen_open`。
- **裸 frame_cache**: 无 LeRobot meta、单路相机(仅触觉 MAE 预训练用), 如 `pretrained_data`。

```
playground/data/
├── rm_umi_dual_pen_open/      # LeRobot 数据集
│   └── meta/info.json
└── pretrained_data/           # 裸 frame_cache (无 meta)
```

---

# Git usage

```bash
# 开始写代码前
git pull --rebase origin main

# 写完后
git add .
git commit -m "..."
git push origin main

# 另一台服务器同步
git pull origin main

# 强制同步 (丢弃本地改动)
git fetch origin
git reset --hard origin/main
```

---

# Training

## Backbone 预训练: 触觉 MAE (`train_enc.sh`)

基于 `vtla/tac_encoder/tactile_mae`, 直接吃 `playground/data/` 下的数据训练 AnyTouch stage1 风格的触觉 MAE encoder, 输出到 `playground/results/backbones/`。

```bash
# 用法: bash train_enc.sh <dataset_ids> <init_mode> <arch> <num_processes> <batch_size> <epochs>
bash train_enc.sh rm_umi_dual_pen_open clip vit_l
```

位置参数:

| 序号 | 参数 | 默认 | 说明 |
| --- | --- | --- | --- |
| 1 | `dataset_ids` | `pretrained_data` | 一个或多个数据集; 多个用引号包成空格分隔串, 如 `"ds1 ds2"`。**LeRobot 与裸 frame_cache 不能混跑** |
| 2 | `init_mode` | `clip` | `scratch`(随机) / `clip`(CLIP 初始化) / `anytouch`(AnyTouch 完整 MAE, 仅 ViT-L) |
| 3 | `arch` | `vit_b` | `vit_l` / `vit_b` |
| 4 | `num_processes` | `4` | >1 用 torchrun 多卡, =1 用单卡 python |
| 5 | `batch_size` | `128` | |
| 6 | `epochs` | `100` | |

常用环境变量: `GPU_ID`(默认 `0,1,2,3`)、`TACTILE_KEYS`(触觉图 key 列表 `[k1,k2,...]`)、`SENSOR_ID`、`MASK_RATIO`、`CONTACT_FILTER`(接触帧筛选, 默认开)、`IMAGE_SIZE`、`RAW_FRAME_CACHE`(强制裸缓存模式)。

> 数据模式自动识别: 含 `meta/info.json` 的为 LeRobot(先 warm-up 缓存再训练); 否则为裸 frame_cache(跳过 warm-up、关闭 contact_filter, 须用与构建缓存时一致的 `IMAGE_SIZE`)。

## VTLA 策略训练 (`train.sh`)

训练 act / diffusion / pi05 / starvla_groot 策略, 输出到 `playground/results/models/`, 日志到 `playground/logs/models/`。

```bash
# 用法: bash train.sh <dataset_id> <policy_type> <num_processes> <batch_size> <steps> \
#                      <wrist_only> <tactile_mode> <state_mode> <action_mode>
bash train.sh rm_umi_dual_pen_open diffusion
```

位置参数:

| 序号 | 参数 | 默认 | 说明 |
| --- | --- | --- | --- |
| 1 | `dataset_id` | `rm_umi_dual_pen_open` | `playground/data/<dataset_id>` |
| 2 | `policy_type` | `diffusion` | `act` / `diffusion` (从零) ・ `pi05` (载 `pi05_base`) ・ `starvla_groot` (载底座 VLM) |
| 3 | `num_processes` | `1` | accelerate 进程数 |
| 4 | `batch_size` | `32` | |
| 5 | `steps` | `10_000` | |
| 6 | `wrist_only` | `false` | `true` 只用 wrist 相机; `false` 用 top + wrist |
| 7 | `tactile_mode` | `none` | `none`(触觉不进模型) / `as_image`(触觉当图像输入) / `encode`(触觉 encoder) |
| 8 | `state_mode` | `joint` | `none` / `joint`(关节角) / `episode_ee`(末端位姿, 相对每个 episode 首帧) |
| 9 | `action_mode` | `joint` | `joint`(关节角) / `relative_ee`(末端位姿, 相对当前观测) |

触觉 encoder (仅 `tactile_mode=encode`): 通过 `TACTILE_ENCODER_PATH`(默认 `playground/pretrained_models/AnyTouch-ViT-L-16`)指定 tactile-MAE 权重作为 encoder 初始化, `arch`/`sensor_id`/`image_size` 从 checkpoint 自动读取。`encode` 模式下 encoder + query token 会**随 policy 一起训练**(非冻结)。

相机 / 触觉 key (draccus 列表 `[k1,k2,...]`, 按数据集命名用环境变量覆盖):

- `TOP_CAM` (默认 `[observation.images.cam_top]`)
- `WRIST_CAM` (默认 `[observation.images.left_cam_wrist,observation.images.right_cam_wrist]`)
- `TACTILE_KEYS` (默认四路 finger: `left/right × finger0/1`)

示例 (双臂 + 触觉作为图像输入, 4 卡):

```bash
bash train.sh rm_umi_dual_pen_open diffusion 4 32 20000 false as_image joint
```

### 末端位姿 (EE) 模式

除关节角外, state / action 还支持末端位姿 (end-effector), 与 UMI 数据对齐:

- `state_mode=episode_ee`: state 是**相对每个 episode 首帧**的末端位姿 `T0⁻¹·Tt` (rot6d 表示)。
- `action_mode=relative_ee`: action 是**相对当前观测**的末端位姿 `St⁻¹·S_{t+k}` (数据集存绝对的 episode_ee, 训练时在线转相对)。
- 布局 20 维 (双臂, **right 在前**): 每臂 `[xyz(3), rot6d(6), gripper(1)]`; gripper 始终保持绝对值。
- 约束: `action_mode=relative_ee` 必须搭配 `state_mode=episode_ee`。四个 policy (act/diffusion/pi05/starvla_groot) 均支持。

**前置 (一次性)**: 先把关节数据集离线转出 EE 列 (正运动学 FK, 用睿尔曼算法库, 无需连机械臂)。会给原数据集**新增** `observation.state_episode_ee` / `action_episode_ee` 两列及 `action_relative_ee` 归一化统计, 原关节列保持不变 (joint 模式照常可用):

```bash
python -m vtla.datasets.convert_joints_to_eepose --root playground/data/rm_umi_dual_pen_open
#   --horizon 32   # action_relative_ee 统计的最大 chunk 步长; 训练 chunk_size 须 <= 该值
```

EE 模式训练示例 (pi05, 双臂):

```bash
bash train.sh rm_umi_dual_pen_open pi05 1 32 10000 false none episode_ee relative_ee
```

> 推理 (EE 模式) 的硬件下发链路 (FK 读 state + IK/`rm_movep_canfd` 发 action + 首帧绝对位姿缓存) 仍在接入中, 见 `TODO`。训练侧已全部可用。

---

# Deployment

在睿尔曼 RM75b 双臂上做数据采集与策略推理。所有命令从仓库根运行(脚本会自动 `cd` 到根)。

## 1) 厂商 SDK 位置

硬件封装在 `import` 厂商库前统一调用 `deployment/hardware/_sdk_paths.py` 的 `ensure_*` 把 SDK 加到 `sys.path`。SDK 统一放在 `deployment/sdk/` 下:

```
deployment/sdk/
├── Robotic_Arm/          # 睿尔曼机械臂 SDK     -> import Robotic_Arm.*   (已内置)
│   └── libs/linux_x86/libapi_c.so
├── dm_lingkong_grip/     # 领控电爪客户端       -> import dm_lingkong_grip_sdk.*
├── fish_camera_client/   # 鱼眼相机 gRPC 客户端 -> 扁平 import
└── dmrobotics/           # Flux 触觉传感器 SDK  -> import dmrobotics.*
```

> 仓库已内置 `Robotic_Arm`(含 linux_x86 的 `libapi_c.so`)。其余三个 SDK(电爪 / 鱼眼相机 / 触觉)需按上面的目录名放入 `deployment/sdk/`; 目录缺失时 `_sdk_paths` 静默跳过, 由对应硬件 import 时报错。

## 2) Robot config

机器人配置类在 `deployment/robots/<robot_type>/config_<robot_type>.py`, 通过 `@RobotConfig.register_subclass("<robot_type>")` 注册, 用 `--robot.type=<robot_type>` 选择。已注册的 type 见 `deployment/robots/`(如 `realman_ugripper_dual`、`realman_ugripper_dual_notac`、`realman_ugripper_dual_notop_notac` 等)。

以 `realman_ugripper_dual` 为例, 上机前**重点核对**(详见 [config_realman_ugripper_dual.py](deployment/robots/realman_ugripper_dual/config_realman_ugripper_dual.py)):

- **从臂 IP/端口**: `left_follower_ip=192.168.1.200` / `right_follower_ip=192.168.1.201`, `follower_tcp_port=8080`
- **每臂板子 IP**(鱼眼/触觉/夹爪代理): `left_board_ip=192.168.1.10` / `right_board_ip=192.168.1.11`
- **本机 IP**: `pc_host=192.168.1.120`(触觉 UDP 回传用)
- **夹爪满行程**: `left_gripper_itinerary`/`right_gripper_itinerary`(左右传动比不同, 已实测覆盖)
- **触觉/鱼眼输出尺寸与编码范围**: `tactile_width/height`、`tactile_depth/deform_min/max`(须与训练时一致)
- **启用的手臂与硬件开关**: `arms=["left","right"]`、`use_tactile`、`cameras`(顶部相机, 空字典即无 top)

config 字段都可在命令行用 `--robot.<字段>=<值>` 覆盖, 如 `--robot.use_tactile=false`、`--robot.left_follower_ip=...`。`--robot.id` 用于区分同型号的不同机器人(影响标定文件目录)。

## 3) Pre-test (硬件自检)

```bash
# 1) 存在性检查 (默认, 最安全, 不连硬件)
python -m deployment.tools.hardware_check

# 2) 图像检查 (实际连相机/触觉抓一帧, 可存图)
python -m deployment.tools.hardware_check --stage camera --show

# 3) 主从同步 (⚠️ 会驱动从臂, 必须显式确认)
python -m deployment.tools.hardware_check --stage teleop --confirm-move
```

## 4) 数据采集

推荐用 `collect.sh`(自动按时间命名 `local/<时间戳>_<name>`, 默认存到 `playground/data/<repo_id 末段>`):

```bash
# 用法: bash collect.sh <name> <single_task> <num_episodes>
bash collect.sh rm_tactile_demo "抓笔" 30
```

或直接调脚本:

```bash
python -m deployment.collect \
    --robot.type=realman_ugripper_dual \
    --robot.id=realman_dual \
    --teleop.type=bi_realman_ugripper_leader \
    --dataset.repo_id=local/$(date +%Y%m%d_%H%M%S)_grab_pen \
    --dataset.single_task="抓笔" \
    --dataset.num_episodes=50 \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=15 \
    --robot.use_tactile=true \
    --dataset.push_to_hub=false
```

> 采集中可中途按右键提前保存当前 episode; `episode_time_s` 为单集最长录制秒数, `reset_time_s` 为集间复位场景时间。

## 5) 模型推理

推荐用 `inference.sh`(按 `pretrained_id` + `step` 自动拼 checkpoint 路径, `--match_policy` 自动对齐硬件 + 任务):

```bash
# 用法: bash inference.sh <pretrained_id> <step>
bash inference.sh rm_umi_dual_pen_open_diffusion_wristonly_false_tactile_none_state_joint 5000
```

`pretrained_id` 即 `playground/results/models/` 下的目录名, 实际加载 `playground/results/models/<pretrained_id>/checkpoints/<step 补零6位>/pretrained_model`。

或直接调脚本:

```bash
# 自动对齐 (match_policy 从 checkpoint 读取硬件/任务配置)
python -m deployment.inference \
    --robot.type=realman_ugripper_dual \
    --policy.path=<path to pretrained_model> \
    --dataset.repo_id=local/eval_$(date +%Y%m%d_%H%M%S)_pen \
    --match_policy=true

# 手动模式 (match_policy=false 时, 硬件/任务需自己给)
python -m deployment.inference \
    --robot.type=realman_ugripper_dual \
    --policy.path=<path to pretrained_model> \
    --dataset.repo_id=local/eval_pen \
    --match_policy=false \
    --robot.use_tactile=false \
    --dataset.single_task="Grasp the cap and pull it off the pen."
```

> 推理录制的评测数据集默认落到 `playground/eval/<repo_id 末段>`。
