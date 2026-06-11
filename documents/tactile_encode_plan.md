# tactile_mode=encode 规划

## 目标

让所有 policy framework 都支持：

```bash
--policy.tactile_mode=encode
--policy.tactile_encoder_path=/path/to/checkpoint_or_hf_dir
```

其中 tactile encoder 复用现有 `vtla/tac_encoder/tactile_mae`，不在 `vtla/frameworks`
里另写一套 encoder wrapper。`arch`、`sensor_id` 等信息尽量从 checkpoint 自动读取，
用户训练 policy 时只需要指定 encoder 权重路径。

## 已确定原则

1. `LeRobot` batch 中的 tactile image key 仍来自 `tactile_keys`，默认：
   - `observation.images.cam_finger0`
   - `observation.images.cam_finger1`
2. `tactile_mode=none` 和 `tactile_mode=as_image` 保持现状。
3. `tactile_mode=encode` 时，finger images 不走普通 RGB image path，而是走 tactile MAE encoder。
4. 每个 policy 都把 tactile 特征插入自己的 VLM / observation encoder / prefix encoder 中。
5. 对 ACT / PI05 / StarVLA-GR00T，第一版就支持选择 tactile token 插入到 encoder 侧或 decoder
   条件侧。
6. 不把 tactile 特征直接作为动作输出的一部分；即使选择 decoder 侧，也只是作为 policy decoder
   查询的条件 token。
7. `train.sh` 只新增 `tactile_encoder_path` 和 `tactile_insert_location` 的传参，不要求显式输入
   `arch`、`sensor_id`。

## tactile_mae 侧改动

在 `vtla/tac_encoder/tactile_mae` 内新增推理入口，例如：

```text
vtla/tac_encoder/tactile_mae/inference.py
```

提供一个轻量 feature extractor：

```python
class TactileMAEFeatureExtractor(nn.Module):
    @classmethod
    def from_pretrained(cls, path, freeze=True):
        ...

    @property
    def feature_dim(self):
        ...

    def forward(self, images):
        ...
```

行为：

- 复用现有 `build_model()`、`load_pretrained()`、`extract_features()`。
- 支持输入 `[B, C, H, W]` 和 `[B, T, C, H, W]`。
- 自动 resize 到 encoder 需要的 `image_size`，默认 `224`。
- 自动做与 tactile MAE 训练一致的 ImageNet normalize。
- 输出：
  - `[B, D]` for `[B, C, H, W]`
  - `[B, T, D]` for `[B, T, C, H, W]`
- 默认冻结 encoder：`requires_grad_(False)` + `eval()`。

### 自动读取配置

优先从我们训练出的 MAE checkpoint 中读取 `args`：

```python
arch
use_sensor_token
use_same_patchemb
sensor_id
image_size
```

如果 checkpoint 没有 `args`，例如旧 AnyTouch 权重或 HF dir：

- 从权重 shape 推断 `arch`：
  - hidden size 1024 / patch 14 => `vit_l`
  - hidden size 768 / patch 16 => `vit_b`
- `sensor_id` 默认 `-1`
- `use_sensor_token` 默认 `True`
- `use_same_patchemb` 默认 `True`
- `image_size` 默认 `224`

## 公共配置改动

在 `vtla/frameworks/sensor_routing.py` 中：

```python
tactile_encoder_path: str | None = None
tactile_insert_location: str = "decoder"  # encoder | decoder, only active when tactile_mode="encode"
freeze_tactile_encoder: bool = True
```

保留：

```python
tactile_mode: str  # none | as_image | encode
tactile_keys: list[str]
```

修改校验：

- 移除当前 `tactile_mode='encode'` 的 `NotImplementedError`。
- 当 `tactile_mode='encode'` 且 `tactile_encoder_path` 为空时报错。
- `tactile_insert_location` 只允许 `"encoder"` 或 `"decoder"`。
- `tactile_insert_location` 仅在 `tactile_mode='encode'` 时生效，其他 mode 下忽略。
- `prune_unselected_visual_features()` 继续保留 tactile keys，使 batch 中能取到 tactile images。

## tactile token 插入位置

对于 ACT / PI05 / StarVLA-GR00T，可以统一看成：

```text
observation encoder -> policy decoder / action module
```

其中 PI05 和 StarVLA-GR00T 的 observation encoder 是 VLM，decoder 是动作 expert / action head。
因此 `tactile_mode=encode` 时有两个可选插入位置。

### encoder 侧

结构：

```text
RGB / language / state / tactile_tokens
        -> encoder
        -> encoded condition
        -> decoder / action module
```

含义：

- tactile token 与视觉、语言、状态 token 一起进入 encoder。
- tactile 可以在 encoder 内部与其他模态深度交互。
- 更接近 `tactile_mode=as_image` 的语义，只是 tactile image 先经过 tactile MAE 编码。

风险：

- 对 PI05 / StarVLA-GR00T，VLM 原本没有见过 tactile MAE embedding 分布。
- 如果 VLM 冻结，陌生 tactile token 只能靠 projection 对齐，效果不一定稳定。
- 工程上需要改 VLM / prefix 输入构造。

### decoder 侧

结构：

```text
RGB / language / state
        -> encoder
        -> encoded condition

tactile images
        -> tactile MAE
        -> tactile_tokens

encoded condition + tactile_tokens
        -> decoder / action module
```

含义：

- tactile token 不再经过 VLM / observation encoder。
- tactile token 作为 policy decoder 查询的额外条件。
- 对 PI05，可以理解为 action suffix tokens 查询原 prefix KV cache 外，还能查询 tactile condition。
- 对 StarVLA-GR00T，可以理解为 GR00T action head 的 `encoder_hidden_states` 中额外拼入 tactile token。

优点：

- tactile MAE 表示不需要被 VLM 直接理解。
- 更适合 VLM 冻结或半冻结的情况。
- tactile 作为控制相关条件直接给 action module，路径更短。

风险：

- tactile 与语言/视觉的深层交互减少。
- 多模态融合主要发生在 action decoder / policy module 中。

### Diffusion 的特殊情况

Diffusion policy 没有上述显式 encoder-decoder token 结构。它只有：

```text
observation condition -> diffusion policy
```

因此 `tactile_insert_location` 对 Diffusion 不产生结构分歧。无论设置为 `"encoder"` 还是
`"decoder"`，第一版都将 tactile feature 加入 global conditioning：

```text
global_cond = concat(state, rgb_features, tactile_features, env_state)
```

## 各 Framework 接入方向

### ACT

当 `tactile_insert_location="encoder"`：

tactile MAE 输出作为 ACT transformer encoder 的额外 observation token。

候选 token 序列：

```text
[latent, state?, env_state?, image_tokens..., tactile_tokens...]
```

每个 tactile key 产生一个 CLS feature：

```python
feat = tactile_encoder(batch[key])   # [B, D]
token = tactile_proj(feat)           # [B, dim_model]
```

`tactile_proj` 属于 ACT 的 encoder 输入投影层，不属于 action head。

当 `tactile_insert_location="decoder"`：

- ACT encoder 仍只编码原本的视觉、状态、环境状态。
- tactile MAE 输出投影成 decoder memory token。
- 在 ACT decoder cross-attention 的 memory 侧，将 tactile tokens 拼到 encoder output 后面：

```text
decoder memory = concat(act_encoder_out, tactile_tokens)
```

此时 tactile token 作为 action query 可查询的额外条件，不经过 ACT encoder。

### Diffusion

tactile MAE 输出加入 diffusion observation encoder 的 global conditioning。

候选结构：

```text
global_cond = concat(state, rgb_encoder_features, tactile_encoder_features, env_state)
```

支持训练时的 `[B, n_obs_steps, C, H, W]` tactile 输入，输出 `[B, n_obs_steps, D]`，
再与其他 observation feature 一起 flatten。

`tactile_insert_location` 对 Diffusion 不改变实现路径。

### PI05

当 `tactile_insert_location="encoder"`：

tactile MAE 输出加入 PI05 prefix encoder。

候选 prefix 顺序：

```text
[image tokens..., tactile tokens..., language tokens...]
```

需要修改 `PI05Pytorch.embed_prefix()`，允许额外 tactile prefix embeddings。
`PI05Policy` 负责从 batch 中取 tactile images、编码、投影到 PaliGemma hidden dim，
再传给 core model。

当 `tactile_insert_location="decoder"`：

- PaliGemma/Gemma prefix encoder 仍只处理 image + language。
- tactile MAE 输出投影到 action expert 可消费的 hidden dim。
- 在 action suffix tokens 查询条件时，把 tactile tokens 作为额外 condition 拼到 prefix KV / prefix
  hidden context 中。
- 语义上是：

```text
action tokens query concat(vlm_prefix_condition, tactile_condition)
```

实现时需要在 `PI05Pytorch.forward()` 和 `sample_actions()` 的 prefix-cache / prefix-mask 构造处支持
额外 tactile condition。

### StarVLA-GR00T

当 `tactile_insert_location="encoder"`：

tactile MAE 输出作为额外 token 进入 Qwen-VL prefix encoder 的输入构造。

候选结构：

```text
[image tokens..., tactile tokens..., language tokens...] -> Qwen-VL
```

这条路径需要更深地改 Qwen-VL input embedding / attention mask 构造，因为 tactile token 不是原生
image patch，也不是 text token。

当 `tactile_insert_location="decoder"`：

tactile MAE 输出加入 Qwen-VL prefix context 之后、GR00T action head 之前。

候选结构：

```text
qwen_last_hidden = Qwen-VL(image, text)
tactile_tokens = tactile_proj(tactile_encoder(finger_images))
last_hidden = concat(qwen_last_hidden, tactile_tokens)
attention_mask = concat(attention_mask, tactile_mask)
```

GR00T flow-matching action head 继续只消费 `last_hidden + attention_mask`。

## train.sh 改动

根目录 `train.sh` 增加一个参数或环境变量：

```bash
tactile_encoder_path=${TACTILE_ENCODER_PATH:-${9:-}}
tactile_insert_location=${TACTILE_INSERT_LOCATION:-${10:-decoder}}
```

当 `tactile_mode=encode` 时追加：

```bash
--policy.tactile_encoder_path=${tactile_encoder_path}
--policy.tactile_insert_location=${tactile_insert_location}
```

示例：

```bash
TACTILE_ENCODER_PATH=playground/results/backbones/xxx/checkpoints/best.pth \
bash train.sh rm_nist_260320_strawberry diffusion 8 32 5000 false encode joint
```

选择 encoder 侧：

```bash
TACTILE_ENCODER_PATH=playground/results/backbones/xxx/checkpoints/best.pth \
TACTILE_INSERT_LOCATION=encoder \
bash train.sh rm_nist_260320_strawberry pi05 8 32 5000 false encode joint
```

选择 decoder 侧：

```bash
TACTILE_ENCODER_PATH=playground/results/backbones/xxx/checkpoints/best.pth \
TACTILE_INSERT_LOCATION=decoder \
bash train.sh rm_nist_260320_strawberry starvla_groot 8 32 5000 false encode joint
```

## 待讨论细节

下面这些位置还需要进一步讨论后再定：

1. ACT encoder 侧：tactile token 放在 image tokens 前还是后。
2. ACT decoder 侧：tactile token 是否直接拼到 encoder memory 后面，还是单独加 type embedding。
3. 每个 framework 是否每个 finger 一个 token，还是先融合两个 finger 后一个 token。
4. Diffusion 中 tactile feature 与 RGB feature concat 的顺序，以及是否需要单独 projection。
5. PI05 encoder 侧：tactile token 放在 image tokens 后、language tokens 前，还是放在所有 prefix 末尾。
6. PI05 decoder 侧：tactile condition 是拼进 prefix KV cache，还是作为单独 condition stream。
7. StarVLA encoder 侧：tactile token 如何进入 Qwen-VL，作为 pseudo-image token 还是独立 embedding token。
8. StarVLA decoder 侧：tactile token concat 到 Qwen hidden 的前面还是后面。
9. tactile encoder 是否始终冻结，还是允许后续通过 `freeze_tactile_encoder=false` 微调。

## 测试计划

1. `tactile_mae.inference` smoke test：
   - 能从 `best.pth` 读取 `args`
   - 能自动推断 `vit_l/vit_b`
   - `[B,3,H,W]` 输出 `[B,D]`
   - `[B,T,3,H,W]` 输出 `[B,T,D]`
2. config 测试：
   - 四个 framework 都能接受 `tactile_mode=encode`
   - `encode` 无 `tactile_encoder_path` 时报清晰错误
   - `none/as_image` 行为不变
3. framework smoke test：
   - ACT forward 不报错
   - Diffusion forward 不报错
   - PI05 prefix 长度增加且 forward/sample path 不报错
   - StarVLA attention mask shape 正确
4. 最小真实训练：
   - 每个 framework 跑 `steps=2`
   - `tactile_mode=encode`
   - 指定同一个 `TACTILE_ENCODER_PATH`
   - 确认能 forward/backward/save checkpoint
