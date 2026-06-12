#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Shared tactile-encode token builder used by all vtla policies.

When ``tactile_mode="encode"``, each policy owns one ``TactileEncoder``. It wraps the
tactile-MAE feature extractor (``vtla.tac_encoder.tactile_mae.inference``), which emits
``N = tactile_num_tokens`` learnable query tokens per tactile image, plus a trainable
projection that maps them into the policy's token space. All tactile keys (fingers) are
encoded in a single batched forward, so the total number of tactile tokens is
``n_keys * N``.

The encoder weights / arch / sensor_id / image_size are loaded automatically from
``config.tactile_encoder_path``; the user only specifies that path. By default the MAE
encoder + query tokens are fine-tuned end-to-end during policy training
(``freeze_tactile_encoder=False``).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from vtla.tac_encoder.tactile_mae.inference import TactileMAEFeatureExtractor


class TactileEncoder(nn.Module):
    """Tactile-MAE query-token encoder + projection that yields tactile tokens.

    forward(batch) returns tactile tokens projected to ``output_dim``:
      * ``[B, n_keys * N, output_dim]``         for ``[B, C, H, W]`` tactile inputs
      * ``[B, T, n_keys * N, output_dim]``      for ``[B, T, C, H, W]`` tactile inputs

    where ``N = tactile_num_tokens`` is the number of query tokens per tactile image.
    """

    def __init__(self, config, output_dim: int):
        super().__init__()
        self.tactile_keys = list(config.tactile_encoder_keys())
        if not self.tactile_keys:
            raise ValueError(
                "TactileEncoder requires tactile_mode='encode' with non-empty tactile_keys."
            )

        self.extractor = TactileMAEFeatureExtractor.from_pretrained(
            config.tactile_encoder_path,
            freeze=config.freeze_tactile_encoder,
            num_query_tokens=config.tactile_num_tokens,
        )
        self.output_dim = int(output_dim)
        self.proj = nn.Linear(self.extractor.feature_dim, self.output_dim)
        if self.extractor.compute_dtype is not None:
            self.proj.to(dtype=self.extractor.compute_dtype)

    @property
    def feature_dim(self) -> int:
        return self.extractor.feature_dim

    @property
    def num_tokens(self) -> int:
        """Total tactile tokens per (time) step: ``n_keys * num_query_tokens``."""
        return len(self.tactile_keys) * self.extractor.num_query_tokens

    def _missing_keys(self, batch: dict[str, Tensor]) -> list[str]:
        return [k for k in self.tactile_keys if k not in batch]

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        missing = self._missing_keys(batch)
        if missing:
            raise ValueError(
                f"tactile_mode='encode' expected tactile keys missing from the batch: {missing}. "
                f"Batch keys: {list(batch.keys())}"
            )

        device = self.proj.weight.device
        n_keys = len(self.tactile_keys)

        imgs = []
        for key in self.tactile_keys:
            img = batch[key]
            if img.device != device:
                img = img.to(device)
            imgs.append(img)

        # Stack every tactile key and run the MAE encoder a *single* time (keys folded
        # into the batch dim) instead of one forward per key. Token ordering matches the
        # old per-key concat: key0's N query tokens, then key1's, ... along the token dim.
        sample = imgs[0]
        if sample.dim() == 4:                              # [B, C, H, W] per key
            stacked = torch.stack(imgs, dim=1)             # [B, n_keys, C, H, W]
            feat = self.extractor(stacked)                 # [B, n_keys, N, D]
            b, _, n, d = feat.shape
            feat = feat.reshape(b, n_keys * n, d)          # [B, n_keys*N, D]
        elif sample.dim() == 5:                            # [B, T, C, H, W] per key
            b, t = sample.shape[:2]
            stacked = torch.stack(imgs, dim=1)             # [B, n_keys, T, C, H, W]
            flat = stacked.reshape(b * n_keys, t, *sample.shape[2:])
            feat = self.extractor(flat)                    # [B*n_keys, T, N, D]
            n, d = feat.shape[-2:]
            feat = feat.reshape(b, n_keys, t, n, d)
            feat = feat.permute(0, 2, 1, 3, 4).reshape(b, t, n_keys * n, d)  # [B, T, n_keys*N, D]
        else:
            raise ValueError(
                f"TactileEncoder expects 4D or 5D tactile tensors, got shape {tuple(sample.shape)}"
            )

        return self.proj(feat)                              # [B, n_keys*N, P] or [B, T, n_keys*N, P]
