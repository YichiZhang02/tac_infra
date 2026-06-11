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
"""Shared sensor-routing config knobs for all vtla policies.

Three knobs, unified across act / diffusion / pi05 / starvla_groot:

- ``wrist_only``: ``True`` uses only the wrist camera; ``False`` uses top + wrist.
- ``tactile_mode``: ``none`` (tactile unused) / ``as_image`` (tactile fed as
  extra image inputs) / ``encode`` (tactile through a dedicated encoder — reserved).
- ``state_mode``: ``none`` (no proprio state) / ``joint`` (joint angles) /
  ``ee`` (end-effector pose — reserved).

The bulk of the routing is feature selection performed at ``validate_features()``
time via the composable helpers below, so the model code only needs to consume
whatever VISUAL / STATE features survive. ``encode`` and ``ee`` are reserved and
raise ``NotImplementedError`` consistently across all policies.
"""

from dataclasses import dataclass, field

from vtla.engine.configs import FeatureType, PolicyFeature
from vtla.engine.utils.constants import OBS_IMAGES, OBS_STATE

VALID_TACTILE_MODES = ("none", "as_image", "encode")
VALID_STATE_MODES = ("none", "joint", "ee")
VALID_TACTILE_ENCODERS = (None, "anytouch2", "native")
VALID_TACTILE_INSERT_LOCATIONS = ("encoder", "decoder")


@dataclass
class SensorRoutingMixin:
    """Mixin holding the shared sensor-routing fields + helper methods.

    Intended to be mixed in *before* ``PreTrainedConfig`` so its (all-defaulted)
    fields combine cleanly:

        @PreTrainedConfig.register_subclass("act")
        @dataclass
        class ACTConfig(SensorRoutingMixin, PreTrainedConfig): ...
    """

    # --- camera routing ---
    # top_camera_keys / wrist_camera_keys 均为列表，支持多路相机（如双臂 left/right wrist）。
    wrist_only: bool = False
    top_camera_keys: list[str] = field(
        default_factory=lambda: ["observation.images.cam_top"]
    )
    wrist_camera_keys: list[str] = field(
        default_factory=lambda: ["observation.images.cam_right_wrist"]
    )

    # --- tactile routing ---
    tactile_mode: str = "none"  # none | as_image | encode
    tactile_encoder_type: str | None = None  # None | anytouch2 | native (encode only)
    tactile_keys: list[str] = field(
        default_factory=lambda: [
            "observation.images.cam_finger0",
            "observation.images.cam_finger1",
        ]
    )
    # --- tactile encoder (tactile_mode="encode" only) ---
    # Path to a trained tactile-MAE checkpoint (.pth) or HF dir. The encoder arch /
    # sensor_id / image_size are read from the checkpoint automatically.
    tactile_encoder_path: str | None = None
    # Where the tactile tokens are injected, relative to each policy's
    # observation-encoder -> action-decoder structure:
    #   "encoder": tactile tokens enter the observation encoder with the other
    #              modalities (deep multimodal interaction).
    #   "decoder": tactile tokens are an extra condition queried by the action
    #              decoder only (does not pass through the VLM / obs encoder).
    # Ignored for Diffusion (no explicit encoder/decoder split) and only active
    # when tactile_mode="encode".
    tactile_insert_location: str = "decoder"  # encoder | decoder
    # Number of learnable query tokens emitted by the tactile-MAE encoder per tactile
    # image. Total tactile tokens = len(tactile_keys) * tactile_num_tokens.
    tactile_num_tokens: int = 8
    # By default the tactile-MAE encoder + query tokens are fine-tuned end-to-end with
    # the policy (the checkpoint is used as initialization). Set True to freeze the MAE
    # backbone and train only the query tokens + projection.
    freeze_tactile_encoder: bool = False

    # --- state routing ---
    state_mode: str = "joint"  # none | joint | ee

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _dedupe(keys: list[str]) -> list[str]:
        seen, out = set(), []
        for k in keys:
            if k not in seen:
                out.append(k)
                seen.add(k)
        return out

    def selected_camera_keys(self) -> list[str]:
        """RGB cameras selected by ``wrist_only`` (top/wrist 各可多路)。"""
        keys = (
            list(self.wrist_camera_keys)
            if self.wrist_only
            else list(self.top_camera_keys) + list(self.wrist_camera_keys)
        )
        return self._dedupe(keys)

    def image_keys(self) -> list[str]:
        """All image keys fed to the model's vision path (cameras + tactile-as-image)."""
        keys = self.selected_camera_keys()
        if self.tactile_mode == "as_image":
            keys = keys + list(self.tactile_keys)
        return self._dedupe(keys)

    # Alias used by the VLM policies (pi05 / starvla_groot).
    def vlm_image_keys(self) -> list[str]:
        return self.image_keys()

    def tactile_encoder_keys(self) -> list[str]:
        """Tactile keys reserved for a dedicated encoder branch (``encode`` mode)."""
        return self._dedupe(list(self.tactile_keys)) if self.tactile_mode == "encode" else []

    @property
    def image_features(self) -> dict:
        """RGB image features fed to the policy's vision backbone.

        Overrides ``PreTrainedConfig.image_features`` (the mixin precedes it in the
        MRO) to drop tactile-encoder keys in ``encode`` mode: those tactile images go
        through the dedicated tactile-MAE encoder, not the RGB vision path.
        """
        from vtla.engine.configs import FeatureType

        if not self.input_features:
            return {}
        tactile_keys = set(self.tactile_encoder_keys())
        return {
            key: ft
            for key, ft in self.input_features.items()
            if ft.type is FeatureType.VISUAL and key not in tactile_keys
        }

    def normalizer_input_features(self) -> dict:
        """``input_features`` for the dataset normalizer.

        In ``encode`` mode the tactile-encoder keys are dropped here so they are *not*
        normalized with dataset mean/std: the tactile-MAE encoder consumes raw [0, 1]
        images and applies its own (ImageNet) normalization internally.
        """
        feats = dict(self.input_features)
        if self.tactile_mode == "encode":
            for key in self.tactile_encoder_keys():
                feats.pop(key, None)
        return feats

    # ------------------------------------------------------------------
    # Validation building blocks (call these from each config)
    # ------------------------------------------------------------------
    def validate_sensor_modes(self) -> None:
        """Enum validation + reserved-mode gating. Call from ``__post_init__``."""
        if self.tactile_mode not in VALID_TACTILE_MODES:
            raise ValueError(
                f"Invalid tactile_mode '{self.tactile_mode}'. Expected one of {VALID_TACTILE_MODES}."
            )
        if self.state_mode not in VALID_STATE_MODES:
            raise ValueError(f"Invalid state_mode '{self.state_mode}'. Expected one of {VALID_STATE_MODES}.")
        if self.tactile_encoder_type not in VALID_TACTILE_ENCODERS:
            raise ValueError(
                f"Invalid tactile_encoder_type '{self.tactile_encoder_type}'. "
                f"Expected one of {VALID_TACTILE_ENCODERS}."
            )
        if self.tactile_insert_location not in VALID_TACTILE_INSERT_LOCATIONS:
            raise ValueError(
                f"Invalid tactile_insert_location '{self.tactile_insert_location}'. "
                f"Expected one of {VALID_TACTILE_INSERT_LOCATIONS}."
            )
        if self.tactile_mode == "encode" and not self.tactile_encoder_path:
            raise ValueError(
                "tactile_mode='encode' requires --policy.tactile_encoder_path to point at a "
                "trained tactile-MAE checkpoint (.pth) or HF directory."
            )
        if self.tactile_mode == "encode" and self.tactile_num_tokens < 1:
            raise ValueError(
                f"tactile_num_tokens must be >= 1, got {self.tactile_num_tokens}."
            )

    def require_visual_feature(self, key: str, purpose: str) -> None:
        if key not in self.input_features:
            available = [n for n, ft in self.input_features.items() if ft.type is FeatureType.VISUAL]
            raise ValueError(
                f"{type(self).__name__}: {purpose} key '{key}' is not present in input_features. "
                f"Available visual keys: {available}"
            )
        if self.input_features[key].type is not FeatureType.VISUAL:
            raise ValueError(
                f"{type(self).__name__}: {purpose} key '{key}' must be a visual feature, "
                f"got {self.input_features[key].type}."
            )

    def validate_routed_keys(self) -> None:
        """Check that selected cameras and (if used) tactile keys exist as VISUAL features."""
        for key in self.selected_camera_keys():
            self.require_visual_feature(key, "camera")
        if self.tactile_mode in ("as_image", "encode"):
            if not self.tactile_keys:
                raise ValueError(f"{type(self).__name__}: tactile_mode='{self.tactile_mode}' requires tactile_keys.")
            for key in self.tactile_keys:
                self.require_visual_feature(key, "tactile")

    def prune_unselected_visual_features(self, extra_keep: tuple[str, ...] = ()) -> None:
        """Drop VISUAL input features that are not part of the active routing.

        Keeps: selected cameras, tactile (as_image/encode), and any ``extra_keep``
        (e.g. empty-camera placeholders). Everything else VISUAL is removed so the
        model only sees the cameras the knobs selected.
        """
        keep = set(self.image_keys()) | set(self.tactile_encoder_keys()) | set(extra_keep)
        for key in list(self.input_features):
            ft = self.input_features[key]
            if ft.type is FeatureType.VISUAL and key not in keep:
                self.input_features.pop(key)

    def apply_state_mode(self, padded_state_dim: int | None = None) -> None:
        """Route the proprioceptive state according to ``state_mode``.

        - ``none``: remove ``observation.state``.
        - ``ee``:   reserved → ``NotImplementedError``.
        - ``joint``: keep; if ``padded_state_dim`` is given and state is missing,
                     materialise a padded state feature (pi05-style).
        """
        if self.state_mode == "none":
            self.input_features.pop(OBS_STATE, None)
        elif self.state_mode == "ee":
            raise NotImplementedError(
                "state_mode='ee' (end-effector pose conditioning) is reserved and not implemented yet."
            )
        elif self.state_mode == "joint":
            if padded_state_dim is not None and OBS_STATE not in self.input_features:
                self.input_features[OBS_STATE] = PolicyFeature(
                    type=FeatureType.STATE, shape=(padded_state_dim,)
                )

    def add_empty_cameras(self, num: int, image_resolution: tuple[int, int]) -> list[str]:
        """Add ``num`` zero-padded placeholder cameras; return their keys."""
        keys = []
        for i in range(num):
            key = OBS_IMAGES + f".empty_camera_{i}"
            self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_resolution))
            keys.append(key)
        return keys
