#!/usr/bin/env python

# Copyright 2025 starVLA community & The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""QwenGR00T policy ported from starVLA into the vtla (LeRobot-style) framework.

Adapts the starVLA ``Qwen_GR00T`` framework (Qwen-VL prefix encoder + GR00T
flow-matching DiT action head) to the vtla ``PreTrainedPolicy`` contract:
a LeRobot ``batch: dict[str, Tensor]`` in, ``(loss, dict)`` / action chunk out.
"""

from collections import deque

import numpy as np
import torch
from torch import Tensor

from vtla.engine.utils.constants import ACTION, OBS_STATE
from vtla.engine.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from ..tactile_encode import TactileEncoder
from .action_head.flow_matching_head import ActionHeadConfig, FlowmatchingActionHead
from .configuration_starvla_groot import StarvlaGrootConfig
from .qwen_vl_interface import QwenVLInterface


class StarvlaGrootPolicy(PreTrainedPolicy):
    """Qwen-VL + GR00T flow-matching action head."""

    config_class = StarvlaGrootConfig
    name = "starvla_groot"

    def __init__(self, config: StarvlaGrootConfig, **kwargs):
        require_package("transformers", extra="starvla_groot")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Resolve action / state dims from dataset features (real, un-padded dims).
        action_dim = config.action_dim or int(config.output_features[ACTION].shape[0])
        if config.state_mode == "none":
            state_dim = 0
        elif config.state_dim is not None:
            state_dim = int(config.state_dim)
        elif OBS_STATE in config.input_features:
            state_dim = int(config.input_features[OBS_STATE].shape[0])
        else:
            state_dim = 0
        self.action_dim = action_dim
        self.state_dim = state_dim

        # VLM prefix encoder.
        load_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.qwen_vl = QwenVLInterface(
            base_vlm=config.base_vlm,
            attn_implementation=config.attn_implementation,
            dtype=load_dtype,
            image_resolution=config.image_resolution,
        )

        # Align the DiT cross-attention dim to the VLM hidden size.
        diffusion_model_cfg = dict(config.diffusion_model_cfg)
        diffusion_model_cfg["cross_attention_dim"] = self.qwen_vl.hidden_size

        head_cfg = ActionHeadConfig(
            action_model_type=config.action_model_type,
            hidden_size=config.action_head_hidden_size,
            action_dim=action_dim,
            state_dim=state_dim,
            action_horizon=config.chunk_size,
            num_inference_timesteps=config.num_inference_timesteps,
            num_target_vision_tokens=config.num_target_vision_tokens,
            add_pos_embed=config.add_pos_embed,
            max_seq_len=config.max_seq_len,
            noise_beta_alpha=config.noise_beta_alpha,
            noise_beta_beta=config.noise_beta_beta,
            noise_s=config.noise_s,
            num_timestep_buckets=config.num_timestep_buckets,
            diffusion_model_cfg=diffusion_model_cfg,
        )
        self.action_head = FlowmatchingActionHead(head_cfg)

        # Tactile-encode branch (tactile_mode="encode"). Tactile tokens are projected to
        # the Qwen-VL hidden size, then either:
        #   - "encoder": injected into the Qwen-VL input embeddings so they pass through
        #     all LLM layers (deep fusion with image/language tokens), or
        #   - "decoder": appended to the VLM output hidden states as an extra condition
        #     the GR00T action head cross-attends to.
        self.tactile_encoder = None
        if config.tactile_mode == "encode":
            self.tactile_encoder = TactileEncoder(config, self.qwen_vl.hidden_size)
            if config.tactile_insert_location == "encoder" and not self.qwen_vl.supports_prefix_injection():
                raise RuntimeError(
                    "StarVLA-GR00T tactile_insert_location='encoder' requires a Qwen3.5-style "
                    "VLM backbone that exposes get_rope_index / get_image_features / "
                    "get_placeholder_mask. Use tactile_insert_location='decoder' for this backbone."
                )

        self._set_requires_grad()
        self.to(config.device)
        self.reset()

    # ------------------------------------------------------------------
    # Freezing / training-mode helpers
    # ------------------------------------------------------------------
    def _set_requires_grad(self):
        if self.config.train_expert_only:
            self.qwen_vl.eval()
            for param in self.qwen_vl.parameters():
                param.requires_grad = False
        elif self.config.freeze_vision_encoder:
            visual = getattr(self.qwen_vl.model, "visual", None)
            if visual is not None:
                visual.eval()
                for param in visual.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.train_expert_only:
            self.qwen_vl.eval()
        elif self.config.freeze_vision_encoder:
            visual = getattr(self.qwen_vl.model, "visual", None)
            if visual is not None:
                visual.eval()
        return self

    # ------------------------------------------------------------------
    # Batch -> model-input bridging
    # ------------------------------------------------------------------
    def _build_images(self, batch: dict[str, Tensor]) -> Tensor:
        """Collect multi-view camera tensors into a GPU stack ``[B, V, C, H, W]``.

        Each view is resized on GPU to the Qwen ``smart_resize`` target (a constant for a
        square ``image_resolution``, e.g. 256), so all views share a resolution and can be
        stacked. Downsizing here keeps the per-frame Qwen vision-token count bounded: raw
        frames (e.g. 896x896) would otherwise emit ~1 token per merged 32x32 px and blow up
        the VLM self-attention (O(L^2)) and the action-head cross-attention KV length.

        No tensor->CPU->PIL round-trip: the heavy preprocessing stays on GPU.
        """
        import torch.nn.functional as F

        vlm_image_keys = self.config.vlm_image_keys()
        present_keys = [key for key in vlm_image_keys if key in batch]
        if not present_keys:
            raise ValueError(
                f"No VLM image keys present in batch. Expected one of {vlm_image_keys}, "
                f"got batch keys {list(batch.keys())}"
            )

        device = next(self.parameters()).device
        target_h, target_w = self.qwen_vl._target_h, self.qwen_vl._target_w

        views: list[Tensor] = []
        for key in present_keys:
            cam = batch[key]  # [B, C, H, W], float in [0, 1] (LeRobot VISUAL=IDENTITY)
            if cam.dim() != 4:
                raise ValueError(f"Expected camera '{key}' as [B, C, H, W], got {tuple(cam.shape)}")
            cam = cam.to(device=device, dtype=torch.float32, non_blocking=True)
            if cam.shape[-2:] != (target_h, target_w):
                cam = F.interpolate(
                    cam, size=(target_h, target_w), mode="bilinear",
                    align_corners=False, antialias=True,
                )
            views.append(cam)

        # [B, V, C, H, W], view order == present_keys order (matches input_ids placeholders).
        return torch.stack(views, dim=1)

    def _get_instructions(self, batch: dict[str, Tensor], bsize: int) -> list[str]:
        tasks = batch.get("task")
        if tasks is None:
            return [""] * bsize
        if isinstance(tasks, str):
            tasks = [tasks]
        return list(tasks)

    def _get_state(self, batch: dict[str, Tensor], device, dtype) -> Tensor | None:
        if self.config.state_mode == "none" or self.state_dim == 0:
            return None
        if OBS_STATE not in batch:
            return None
        state = batch[OBS_STATE].to(device=device, dtype=dtype)  # [B, state_dim]
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [B, 1, state_dim]
        return state

    def _encode_prefix(self, batch: dict[str, Tensor]):
        """Run the VLM and return (last_hidden_state, attention_mask)."""
        images = self._build_images(batch)
        instructions = self._get_instructions(batch, len(images))

        tactile_tokens = None
        if self.tactile_encoder is not None:
            tactile_tokens = self.tactile_encoder(batch)  # [B, n_tac, H]

        # Encoder side: inject tactile tokens into the Qwen-VL *input* embeddings so they
        # flow through every LLM layer alongside image/language tokens (deep fusion).
        if tactile_tokens is not None and self.config.tactile_insert_location == "encoder":
            return self.qwen_vl.forward_with_prefix_tokens(images, instructions, tactile_tokens)

        qwen_inputs = self.qwen_vl.build_qwenvl_inputs(images=images, instructions=instructions)
        attention_mask = qwen_inputs.get("attention_mask", None)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]  # [B, L, H]

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)

        # Decoder side: append tactile tokens to the VLM *output* hidden states (and extend
        # the mask) as an extra condition the GR00T action head cross-attends to. They do
        # not pass through the Qwen-VL transformer.
        if tactile_tokens is not None:
            tactile_tokens = tactile_tokens.to(device=last_hidden.device, dtype=last_hidden.dtype)
            last_hidden = torch.cat([last_hidden, tactile_tokens], dim=1)
            if attention_mask is not None:
                tac_mask = torch.ones(
                    tactile_tokens.shape[:2], dtype=torch.bool, device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, tac_mask], dim=1)

        return last_hidden, attention_mask

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        last_hidden, attention_mask = self._encode_prefix(batch)

        with torch.autocast("cuda", dtype=torch.float32):
            actions = batch[ACTION].to(device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.config.chunk_size :, :]

            # Single noise sample per element. (Variance reduction via larger batch_size,
            # not by replicating the action head — replication is reserved for inference.)
            state = self._get_state(batch, last_hidden.device, last_hidden.dtype)

            per_sample_loss = self.action_head(
                last_hidden, actions_target, state, encoder_attention_mask=attention_mask
            )  # (B,)

        loss_dict = {"action_loss": per_sample_loss.mean().item()}
        if reduction == "none":
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        loss = per_sample_loss.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        last_hidden, attention_mask = self._encode_prefix(batch)

        # Inference-time ensemble: denoise `repeated_diffusion_steps` independent noise
        # initializations in parallel (the VLM prefix is encoded once and reused), then
        # average them into a single action chunk. Under no_grad this costs no training
        # memory. reps=1 falls back to a single sample.
        reps = max(1, int(self.config.repeated_diffusion_steps))

        with torch.autocast("cuda", dtype=torch.float32):
            state = self._get_state(batch, last_hidden.device, last_hidden.dtype)

            if reps > 1:
                bsize = last_hidden.shape[0]
                last_hidden_rep = last_hidden.repeat(reps, 1, 1)
                attn_rep = attention_mask.repeat(reps, 1) if attention_mask is not None else None
                state_rep = state.repeat(reps, 1, 1) if state is not None else None
                pred_rep = self.action_head.predict_action(
                    last_hidden_rep, state_rep, encoder_attention_mask=attn_rep
                )  # [B * reps, chunk, action_dim]
                pred_actions = pred_rep.view(reps, bsize, *pred_rep.shape[1:]).mean(dim=0)
            else:
                pred_actions = self.action_head.predict_action(
                    last_hidden, state, encoder_attention_mask=attention_mask
                )  # [B, chunk, action_dim]

        return pred_actions.to(dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # (B, n_action_steps, action_dim) -> queue of (B, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_default_peft_targets(self) -> dict:
        # Train the action head fully and adapt the VLM attention projections.
        return {
            "target_modules": r"(.*self_attn\.(q|v)_proj)",
            "modules_to_save": ["action_head"],
        }
