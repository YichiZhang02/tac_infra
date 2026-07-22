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
from vtla.engine.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

VALID_TACTILE_MODES = ("none", "as_image", "encode")
# joint = joint angles; episode_ee = EE pose relative to each episode's FIRST frame (T0^-1·Tt);
# absolute_ee = EE pose in the robot base frame (Tt, no T0) — keeps absolute workspace position.
VALID_STATE_MODES = ("none", "joint", "episode_ee", "absolute_ee")
# joint = joint targets; relative_ee = EE pose relative to the CURRENT observation (delta).
VALID_ACTION_MODES = ("joint", "relative_ee")
VALID_TACTILE_ENCODERS = (None, "anytouch2", "native")
VALID_TACTILE_INSERT_LOCATIONS = ("encoder", "decoder")

# Dataset columns / stats keys added offline by tools/convert_joints_to_eepose.py.
OBS_STATE_EPISODE_EE = OBS_STATE + "_episode_ee"  # observation.state_episode_ee
ACTION_EPISODE_EE = ACTION + "_episode_ee"  # action_episode_ee (per-frame absolute-in-episode)
OBS_STATE_ABSOLUTE_EE = OBS_STATE + "_absolute_ee"  # observation.state_absolute_ee (base-frame Tt)
ACTION_ABSOLUTE_EE = ACTION + "_absolute_ee"  # action_absolute_ee (per-frame base-frame copy)
ACTION_RELATIVE_EE = ACTION + "_relative_ee"  # stats-only key: relative action St^-1·S_{t+k}


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
    # --- tactile temporal window (independent of the RGB observation window) ---
    # Number of tactile frames fed to the policy per step, INCLUDING the current frame.
    # ``1`` (default) = current frame only = exact legacy behaviour. ``F>1`` stacks the
    # current frame plus ``F-1`` earlier frames along a leading time axis, giving the
    # policy short-horizon tactile history. Applies to both tactile_mode="as_image"
    # and "encode". Decoupled from the shared observation window / n_obs_steps.
    tactile_num_frames: int = 1
    # Spacing, in dataset frames, between two consecutive tactile frames. ``1`` = adjacent
    # frames; ``k`` = every k-th frame (wider temporal receptive field, same F frames).
    # The sampled tactile delta indices are ``[-(F-1)*offset, ..., -offset, 0]``.
    tactile_frame_offset: int = 1
    # By default the tactile-MAE encoder + query tokens are fine-tuned end-to-end with
    # the policy (the checkpoint is used as initialization). Set True to freeze the MAE
    # backbone and train only the query tokens + projection.
    freeze_tactile_encoder: bool = False

    # --- state / action routing ---
    state_mode: str = "joint"  # none | joint | episode_ee
    action_mode: str = "joint"  # joint | relative_ee
    # Number of arms packed in the EE vectors (per arm = 3 pos + 6 rot6d + 1 gripper = 10 dims).
    ee_num_arms: int = 2
    # Ordered names of the observation.state joints (populated by make_policy from ds_meta).
    # Required for EpisodeEEPreprocessorStep to locate joint/gripper indices at inference time.
    state_feature_names: list[str] | None = None

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

    def tactile_windowed(self) -> bool:
        """True when a multi-frame tactile history is requested (``tactile_num_frames > 1``)."""
        return self.tactile_mode in ("as_image", "encode") and self.tactile_num_frames > 1

    def tactile_delta_indices(self) -> list[int]:
        """Frame offsets sampled for each tactile key, oldest → current.

        ``[-(F-1)*offset, ..., -offset, 0]`` where ``F = tactile_num_frames`` and
        ``offset = tactile_frame_offset``. ``F == 1`` yields ``[0]`` (current frame only).
        These override the shared ``observation_delta_indices`` for tactile keys so the
        tactile temporal window is independent of the RGB observation window.
        """
        f = int(self.tactile_num_frames)
        off = int(self.tactile_frame_offset)
        return [-(f - 1 - i) * off for i in range(f)]

    def tactile_windowed_keys(self) -> list[str]:
        """Tactile image keys that receive the temporal window (both as_image and encode)."""
        if self.tactile_mode in ("as_image", "encode"):
            return self._dedupe(list(self.tactile_keys))
        return []

    def image_feature_keys_expanded(self) -> list[str]:
        """``image_features`` keys with windowed tactile-as-image keys expanded per frame.

        Preserves the ``image_features`` insertion order (what the vision models iterate),
        replacing each windowed tactile key with ``F`` per-frame keys ``<key>.f{i}``. When
        the tactile window is inactive (F == 1 or not as_image) this returns the plain
        ``image_features`` keys — i.e. exact legacy behaviour.
        """
        keys = list(self.image_features)
        if not (self.tactile_mode == "as_image" and self.tactile_num_frames > 1):
            return keys
        windowed = set(self.tactile_windowed_keys())
        f = int(self.tactile_num_frames)
        out: list[str] = []
        for key in keys:
            if key in windowed:
                out.extend(f"{key}.f{i}" for i in range(f))
            else:
                out.append(key)
        return out

    def decoded_video_keys(self) -> list[str]:
        """All camera videos this policy actually consumes (RGB + tactile-as-image + tactile-encode).

        Used to tell the dataset which video streams to decode, so unselected cameras (e.g. finger
        cams when ``tactile_mode='none'``) are not decoded every sample — a large data-loading win
        for fast models that would otherwise starve the GPU. Cameras not in this list are skipped.
        """
        return self._dedupe(self.image_keys() + self.tactile_encoder_keys())

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
        if self.action_mode not in VALID_ACTION_MODES:
            raise ValueError(f"Invalid action_mode '{self.action_mode}'. Expected one of {VALID_ACTION_MODES}.")
        if self.action_mode == "relative_ee" and self.state_mode not in ("episode_ee", "absolute_ee"):
            raise ValueError(
                "action_mode='relative_ee' requires state_mode in {'episode_ee', 'absolute_ee'}: the "
                "relative action is computed against the current EE observation (the relative target "
                "St^-1·S_{t+k} is anchor-independent, so either EE state encoding works)."
            )
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
        # Tactile temporal window (both as_image and encode).
        if self.tactile_num_frames < 1:
            raise ValueError(
                f"tactile_num_frames must be >= 1, got {self.tactile_num_frames}."
            )
        if self.tactile_frame_offset < 1:
            raise ValueError(
                f"tactile_frame_offset must be >= 1, got {self.tactile_frame_offset}."
            )
        if self.tactile_num_frames > 1 and self.tactile_mode == "none":
            raise ValueError(
                "tactile_num_frames > 1 requires tactile_mode in {'as_image', 'encode'} "
                f"(got tactile_mode='none')."
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

        The dataset carries the joint ``observation.state`` plus the EE variants
        ``observation.state_episode_ee`` / ``observation.state_absolute_ee``. This selects one
        as the canonical ``observation.state`` the model consumes and drops the unselected ones.

        - ``none``: remove ``observation.state`` (all variants).
        - ``joint``: keep joint ``observation.state``, drop the EE variants; if
                     ``padded_state_dim`` is given and state is missing, materialise a
                     padded state feature (pi05-style).
        - ``episode_ee``: use ``observation.state_episode_ee`` as ``observation.state``.
        - ``absolute_ee``: use ``observation.state_absolute_ee`` as ``observation.state``.
        """
        if self.state_mode == "none":
            self.input_features.pop(OBS_STATE, None)
            self.input_features.pop(OBS_STATE_EPISODE_EE, None)
            self.input_features.pop(OBS_STATE_ABSOLUTE_EE, None)
        elif self.state_mode == "joint":
            self.input_features.pop(OBS_STATE_EPISODE_EE, None)
            self.input_features.pop(OBS_STATE_ABSOLUTE_EE, None)
            if padded_state_dim is not None and OBS_STATE not in self.input_features:
                self.input_features[OBS_STATE] = PolicyFeature(
                    type=FeatureType.STATE, shape=(padded_state_dim,)
                )
        elif self.state_mode in ("episode_ee", "absolute_ee"):
            ee_key = OBS_STATE_ABSOLUTE_EE if self.state_mode == "absolute_ee" else OBS_STATE_EPISODE_EE
            other_key = OBS_STATE_EPISODE_EE if self.state_mode == "absolute_ee" else OBS_STATE_ABSOLUTE_EE
            self.input_features.pop(other_key, None)
            ee_ft = self.input_features.pop(ee_key, None)
            if ee_ft is not None:
                # Dataset has the pre-computed column; rename it to canonical OBS_STATE.
                self.input_features.pop(OBS_STATE, None)
                self.input_features[OBS_STATE] = ee_ft
            elif OBS_STATE not in self.input_features:
                raise ValueError(
                    f"state_mode='{self.state_mode}' requires either '{ee_key}' in the dataset "
                    "(run tools/convert_joints_to_eepose.py for offline datasets) or "
                    f"'{OBS_STATE}' for real-time inference (an EpisodeEEPreprocessorStep converts "
                    "joint angles to EE pose at runtime)."
                )
            # else: OBS_STATE present but the EE column absent → inference mode.
            # EpisodeEEPreprocessorStep converts observation.state joints → EE pose before the model.

    def apply_action_mode(self) -> None:
        """Route the action according to ``action_mode`` (mirrors :meth:`apply_state_mode`).

        The dataset carries the joint ``action`` plus the EE variants ``action_episode_ee`` /
        ``action_absolute_ee``. This selects one as the canonical ``action`` output and drops the
        unselected variants. For ``relative_ee`` the EE variant is chosen to match ``state_mode``
        (episode→``action_episode_ee``, absolute→``action_absolute_ee``); the relative target is
        anchor-independent so either yields the same trained action.
        """
        if self.output_features is None:
            return
        if self.action_mode == "joint":
            self.output_features.pop(ACTION_EPISODE_EE, None)
            self.output_features.pop(ACTION_ABSOLUTE_EE, None)
        elif self.action_mode == "relative_ee":
            ee_key = ACTION_ABSOLUTE_EE if self.state_mode == "absolute_ee" else ACTION_EPISODE_EE
            other_key = ACTION_EPISODE_EE if self.state_mode == "absolute_ee" else ACTION_ABSOLUTE_EE
            self.output_features.pop(other_key, None)
            ee_ft = self.output_features.pop(ee_key, None)
            if ee_ft is not None:
                self.output_features.pop(ACTION, None)
                self.output_features[ACTION] = ee_ft
            elif ACTION not in self.output_features:
                raise ValueError(
                    f"action_mode='relative_ee' (state_mode='{self.state_mode}') requires '{ee_key}' "
                    "in the dataset. Run tools/convert_joints_to_eepose.py first."
                )

    def add_empty_cameras(self, num: int, image_resolution: tuple[int, int]) -> list[str]:
        """Add ``num`` zero-padded placeholder cameras; return their keys."""
        keys = []
        for i in range(num):
            key = OBS_IMAGES + f".empty_camera_{i}"
            self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_resolution))
            keys.append(key)
        return keys
