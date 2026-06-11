# Copyright 2025 starVLA community. All rights reserved.
# Ported into vtla from starVLA/model/modules/vlm/QWen3.py
#
# A thin, framework-agnostic wrapper around a HuggingFace image-text-to-text VLM
# (Qwen2.5-VL / Qwen3-VL). It centralizes:
#   - loading the backbone + processor
#   - building chat-template inputs from (images, instructions)
#   - exposing the language hidden size for action-head alignment
#
# Unlike the starVLA original it does not depend on OmegaConf configs, the
# overwatch logger, the fast-token vocabulary, or CoT prompt templating.

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QwenVLInterface(nn.Module):
    """Wrapper around a Qwen-VL backbone used as the VLA prefix encoder."""

    def __init__(
        self,
        base_vlm: str,
        attn_implementation: str = "sdpa",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Fall back to sdpa if flash_attention_2 is requested but unavailable.
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                logger.warning("flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = AutoModelForImageTextToText.from_pretrained(
            base_vlm,
            attn_implementation=attn_implementation,
            dtype=dtype,
            ignore_mismatched_sizes=True,
        )
        processor = AutoProcessor.from_pretrained(base_vlm)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor

        # Align the hidden_size accessor across Qwen2.5-VL / Qwen3-VL configs.
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            self.model.config.hidden_size = text_config.hidden_size

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs)

    def supports_prefix_injection(self) -> bool:
        """Whether the backbone exposes the Qwen3.5-style API needed to inject extra
        prefix tokens into the LLM input embeddings."""
        core = getattr(self.model, "model", None)
        return core is not None and all(
            hasattr(core, m)
            for m in ("get_rope_index", "get_image_features", "get_placeholder_mask",
                      "get_input_embeddings")
        )

    def forward_with_prefix_tokens(self, images: List[list], instructions: List[str], extra_embeds):
        """Run the VLM with ``extra_embeds`` injected into the LLM input sequence.

        ``extra_embeds`` (``[B, N, H]``) are placed at the end of the prompt as ``N``
        extra tokens that go through *all* Qwen-VL transformer layers (genuine
        encoder-side fusion), as opposed to being appended to the VLM output. Image
        features are merged by the model's own helpers and the extra tokens receive
        proper (text-continuation) M-RoPE positions.

        Returns ``(last_hidden_state, attention_mask)`` over the full
        ``image + language + extra`` sequence.
        """
        if not self.supports_prefix_injection():
            raise RuntimeError(
                "This VLM backbone does not expose the Qwen3.5-style API "
                "(get_rope_index / get_image_features / get_placeholder_mask) required for "
                "tactile_insert_location='encoder'. Use 'decoder' for this backbone."
            )

        core = self.model.model  # Qwen3_5Model
        inputs = self.build_qwenvl_inputs(images=images, instructions=instructions)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        device = input_ids.device
        bsize, seq_len = input_ids.shape
        n_extra = extra_embeds.shape[1]

        image_token_id = self.model.config.image_token_id
        video_token_id = getattr(self.model.config, "video_token_id", -1)

        # --- extend the token sequence with N filler *text* tokens at the end ---
        pad_id = self.processor.tokenizer.pad_token_id or 0
        filler = torch.full((bsize, n_extra), pad_id, dtype=input_ids.dtype, device=device)
        input_ids_ext = torch.cat([input_ids, filler], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attn_ext = torch.cat(
            [attention_mask, torch.ones(bsize, n_extra, dtype=attention_mask.dtype, device=device)], dim=1
        )

        # mm_token_type_ids: image=1, video=2, text=0 (filler tokens are text).
        mm_tt = torch.zeros_like(input_ids_ext)
        mm_tt[input_ids_ext == image_token_id] = 1
        if video_token_id >= 0:
            mm_tt[input_ids_ext == video_token_id] = 2

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # M-RoPE positions for the extended sequence (filler -> text positions).
            position_ids = core.get_rope_index(
                input_ids_ext,
                mm_tt,
                image_grid_thw=image_grid_thw,
                attention_mask=attn_ext,
            )[0]

            # Merge text + image embeddings exactly as the model would, then override
            # the filler slots with the tactile embeddings.
            inputs_embeds = core.get_input_embeddings()(input_ids_ext)
            if pixel_values is not None:
                image_out = core.get_image_features(pixel_values, image_grid_thw, return_dict=True)
                image_embeds = torch.cat(image_out.pooler_output, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                image_mask, _ = core.get_placeholder_mask(
                    input_ids_ext, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[:, seq_len:, :] = extra_embeds.to(inputs_embeds.dtype)

            outputs = core(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_ext,
                position_ids=position_ids,
                use_cache=False,
            )

        last_hidden = outputs.last_hidden_state
        return last_hidden, attn_ext.to(dtype=torch.bool)

    def build_qwenvl_inputs(self, images: List[list], instructions: List[str], **kwargs):
        """Build batched model inputs from raw (images, instructions).

        Args:
            images: list over batch; each element is a list of PIL.Image (multi-view).
            instructions: list of instruction strings, one per batch element.
        Returns:
            BatchFeature on the model device (input_ids, attention_mask, pixel_values, ...).
        """
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            # processor-specific kwargs (e.g. padding) must go here, not in **kwargs,
            # otherwise transformers warns and ignores the intended location.
            processor_kwargs={"padding": True},
        )
        return batch_inputs.to(self.model.device)
