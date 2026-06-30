# 方案 A：新增 `absolute_ee` state_mode（落地计划）

> 状态：计划，尚未实现。后续按本文件改动。
> 命名约定：`state_mode="absolute_ee"`；新列 `observation.state_absolute_ee` / `action_absolute_ee`
> （与现有 `state_episode_ee` 平行、不混淆）。

## 背景与动机

- 现有 EE 组合是 `state_mode=episode_ee`（`S=T0⁻¹·Tt`，相对每个 episode 首帧）+ `action_mode=relative_ee`。
- 真机实测（在 send_action 按内容路由修复**之后**）：`episode_ee/relative_ee` 效果**不如** `joint/joint`。
  判定为 **state 表示问题**，非部署 bug。
- episode_ee 的问题：第一帧恒为单位阵、整段只编码"离起点多远"，**丢绝对位置**、随起始姿态漂移；
  对"物体在工作空间绝对位置"类任务信息不足。
- 约束：动作空间**必须保持 `relative_ee`**（未来要兼容 relative_ee 预训练模型）。

## 关键不变量（为什么换 state 不影响动作 / 不影响预训练兼容）

relative 动作目标与 anchor 选 episode 还是 absolute **无关**：

```
episode:  a = S_t⁻¹·S_{t+k} = (T0⁻¹Tt)⁻¹(T0⁻¹T_{t+k}) = Tt⁻¹·T_{t+k}
absolute: a = S_t⁻¹·S_{t+k} = Tt⁻¹·T_{t+k}            （S=T，无 T0）
```

T0 自动消掉 → **动作目标完全一致**。因此：
- `action_relative_ee` 的 stats **直接复用**（绝对/相对同值）；
- `ee_transforms`、Relative/AbsoluteActionsProcessorStep、robot movep 下发 **全部不动**；
- relative_ee 预训练模型的**动作 expert 可直接迁移**，只需 state 编码路径适应 absolute_ee 输入。

absolute_ee 仅改变"喂给模型的 state"和"relativize 的 anchor"：`T0⁻¹·T` → `T`。

---

## 落地清单

### 1. 命名与校验 — `vtla/frameworks/sensor_routing.py`
- `VALID_STATE_MODES` 加 `"absolute_ee"`。
- 放宽 ~line 202 交叉校验：`relative_ee` 允许 `state_mode ∈ {episode_ee, absolute_ee}`。
- 新常量：`OBS_STATE_ABSOLUTE_EE = OBS_STATE+"_absolute_ee"`，`ACTION_ABSOLUTE_EE = ACTION+"_absolute_ee"`。
- `apply_state_mode`：加 `absolute_ee` 分支（选 `state_absolute_ee` 作 canonical `observation.state`，prune 其它），照抄 `episode_ee` 分支。
- `apply_action_mode`/校验：`relative_ee` 时确认对应动作列存在（episode→`action_episode_ee`，absolute→`action_absolute_ee`）。

### 2. 离线列 + stats — `tools/convert_joints_to_eepose.py`
- 新增 **`observation.state_absolute_ee`**（20维）= FK(state 关节) 打包 `[pos,rot6d,grip]`，**不减 T0**（原始基座系 FK，比 episode 版少一步）。布局 right-first，与 `build_names()` 一致。
- 新增 **`action_absolute_ee`**（20维）= `state_absolute_ee` 的副本（与 `action_episode_ee` 同套路，单独成列以独立挂 action horizon）。
- stats：`state_absolute_ee` 算全局（`meta/stats.json`）+ 每集（`meta/episodes/*.parquet`）；`action_absolute_ee` 顺手算（规范，不用于归一化）。
- **`action_relative_ee` stats 复用，不重算。**
- 更新 `meta/info.json` features 加这两列。
- 迁移：对数据集**原地重跑**脚本；joint / episode_ee / absolute_ee 三套列共存于同一数据集。

### 3. 训练路由
- `vtla/engine/processor/relative_action_processor.py` `route_ee_batch`：
  - `state_mode=="absolute_ee"` → `observation.state ← state_absolute_ee`；
  - `action_mode=="relative_ee" and state_mode=="absolute_ee"` → `action ← action_absolute_ee`（否则仍 `action_episode_ee`）。
- `vtla/datasets/factory.py` `resolve_delta_timestamps`：`action_absolute_ee` 也挂 action horizon（同 `action_episode_ee`）。
- `vtla/frameworks/ee_processor_utils.py` `remap_ee_dataset_stats`：`state_mode=="absolute_ee"` → `OBS_STATE ← state_absolute_ee` stats；**action 仍 `← action_relative_ee`**（复用）。
- 各策略 config（pi05/act/diffusion/starvla_groot）：`action_delta_indices` 对 `relative_ee` 不变（`range(1,chunk+1)`）；`apply_state_mode/apply_action_mode` 由 mixin 统一处理。
- processor 里写死的 state_mode 白名单加 `absolute_ee`：如 `processor_pi05.py` 的列表、`Pi05PrepareStateTokenizerProcessorStep`（当普通归一化 state，同 joint/episode_ee）。

### 4. 推理路径（比 episode_ee 更简单）
- 复用/泛化 state 预处理：给 `EpisodeEEPreprocessorStep` 加 `relative_to_baseline: bool`（或新建 `AbsoluteEEPreprocessorStep`）——absolute 时**只 FK 打包、不减 T0、不缓存 A0**。
- `vtla/frameworks/factory.py` 推理分支：
  - `state_mode=="absolute_ee"` → prepend 该绝对预处理步骤；
  - **不 append `EpisodeEEToWorldStep`**。因为 `relative_step` 缓存的 anchor 即绝对 `Tt`，后处理
    `ee_to_absolute(Tt, a_rel)=T_{t+k}` 直接是**世界绝对 flange 位姿**，可直接下发。
- 即相比 episode_ee：**砍掉 A0 缓存 + 砍掉 world 转换步**。

### 5. 部署 robot — 无改动
- `realman_ugripper_dual` 的 `action_space=ee` / `_send_action_ee`(movep) / 内容路由 / 坐标系自检照用——
  它本就期望"世界绝对 flange 位姿"，absolute_ee 后处理正好直接给这个。
- `inference.py` 的 `_resolve_action_space`（relative_ee→ee）不变。

### 6. 验证
- 离线：扩展 `tests/test_ee_deploy_roundtrip.py` 加 absolute 链路断言：`Tt=FK(joints)`（无 T0）、
  `a_rel=ee_to_relative(Tt,T_{t+k})`、`ee_to_absolute(Tt,a_rel)==T_{t+k}`（世界系），证明无需 A0。
- 训练 ablation：`absolute_ee/relative_ee` vs `joint/joint` vs `episode_ee/relative_ee` 三方对比。
- 真机前：坐标系自检通过、`max_ee_pos_step_m` 小值、单臂降速。

---

## 改动量小结

| 类别 | 内容 |
|---|---|
| **新增** | 1 个 state_mode、2 个离线列 + state stats、1 个绝对 state 预处理（或加 flag）|
| **小改** | sensor_routing 校验/路由、route_ee_batch、delta_timestamps、remap stats、各 config 白名单、factory 推理分支 |
| **复用不动** | ee_transforms、Relative/Absolute 处理步骤、robot movep/路由、`action_relative_ee` stats、match_policy |
| **反而砍掉** | 推理侧 A0 缓存 + EpisodeEEToWorldStep |

## 建议起点
从**第 2 步离线列**起（其余都依赖数据集先有 `state_absolute_ee`）：先只加列 + stats，跑通转换验证，
再往上接训练 / 推理。

## 相关文件参考
- 现有 episode_ee 部署修复见记忆 `ee-deployment-fix`；EE 整体设计见 `ee-pose-support-design`。
- 数学：`vtla/engine/utils/ee_transforms.py`（ee_to_relative/absolute）、`vtla/engine/utils/ee_kinematics.py`（FK）。
