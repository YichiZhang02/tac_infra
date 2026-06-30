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

"""Real-time EE-pose computation for inference with state_mode='episode_ee'.

When a policy is trained with state_mode='episode_ee' the model expects
``observation.state`` to contain the 20-dim end-effector pose relative to each
episode's FIRST frame (T0^{-1}·Tt), NOT raw joint angles.

This preprocessor step bridges the gap at inference time:
  1. On episode reset it records the first joint observation and computes T0 via FK.
  2. At every subsequent step it runs FK on the current joints and expresses the result
     relative to T0, replacing ``observation.state`` in the transition dict.

The step is injected into the preprocessor pipeline by
``make_pre_post_processors`` when ``policy_cfg.state_mode == 'episode_ee'`` and
a pretrained checkpoint is being loaded (inference path only — during training the
dataset already supplies the pre-computed ``observation.state_episode_ee`` column).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vtla.engine.configs import PipelineFeatureType, PolicyFeature
from vtla.engine.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from vtla.engine.utils.constants import OBS_STATE
from vtla.engine.utils.ee_kinematics import (
    EE_DIM,
    compute_baseline,
    joint_indices,
    make_realman_algo,
    mat_to_rot6d,
    to_absolute_ee,
    to_episode_ee,
)


@dataclass
@ProcessorStepRegistry.register(name="episode_ee_state_processor")
class EpisodeEEPreprocessorStep(ObservationProcessorStep):
    """Convert ``observation.state`` from joint angles to EE pose at inference time.

    Args:
        state_feature_names: Ordered names of each dimension of ``observation.state``
            (e.g. ``["right_joint_1", ..., "right_gripper", "left_joint_1", ...]``).
            Used to locate the per-arm joint and gripper indices.
        relative_to_baseline: If ``True`` (state_mode='episode_ee') the EE pose is expressed
            relative to each episode's FIRST frame (T0^-1·Tt) and the episode-start world pose A0 is
            cached for the paired :class:`EpisodeEEToWorldStep`. If ``False``
            (state_mode='absolute_ee') the pose is the raw base-frame FK (Tt, no T0) and no A0 is
            cached — the model output already decodes straight to world flange poses.
    """

    state_feature_names: list[str] = field(default_factory=list)
    relative_to_baseline: bool = True

    def __post_init__(self) -> None:
        self._algo = make_realman_algo()
        self._jidx: dict = joint_indices(self.state_feature_names)
        self._baseline: tuple | None = None  # ((R_p0, R_R0), (L_p0, L_R0))
        self._a0_packed: torch.Tensor | None = None  # (1, 20) world-flange EE at episode start

    def reset(self) -> None:
        """Clear the episode-start baseline; called at the start of each episode."""
        self._baseline = None
        self._a0_packed = None

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Replace ``observation.state`` (joints) with the 20-dim EE pose.

        episode_ee (``relative_to_baseline=True``): pose relative to the episode's first frame.
        absolute_ee (``relative_to_baseline=False``): raw base-frame FK, no baseline.
        """
        raw = observation.get(OBS_STATE)
        if raw is None:
            return observation

        if isinstance(raw, torch.Tensor):
            vec16 = raw.detach().cpu().numpy().astype(np.float64).flatten()
        else:
            vec16 = np.asarray(raw, dtype=np.float64).flatten()

        if not self.relative_to_baseline:
            ee_vec = to_absolute_ee(self._algo, vec16, self._jidx)  # float32 (20,)
            observation[OBS_STATE] = torch.from_numpy(ee_vec)
            return observation

        if self._baseline is None:
            self._baseline = compute_baseline(self._algo, vec16, self._jidx)
            self._a0_packed = self._pack_baseline(self._baseline)

        ee_vec = to_episode_ee(self._algo, vec16, self._jidx, self._baseline)  # float32 (20,)
        observation[OBS_STATE] = torch.from_numpy(ee_vec)
        return observation

    @staticmethod
    def _pack_baseline(baseline: tuple) -> torch.Tensor:
        """Pack the episode-start FK baseline into a (1, 20) world-flange EE vector.

        Layout matches the offline ``build_names``: RIGHT arm first then LEFT, per arm
        ``[pos(3), rot6d(6), gripper(1)]``. The gripper slot is filled with 0.0 because the
        absolute-pose composition (``ee_to_absolute``) carries the gripper from the action side,
        never from this reference baseline.
        """
        (Rp0, RR0), (Lp0, LR0) = baseline
        vec = np.concatenate([
            Rp0, mat_to_rot6d(RR0), [0.0],
            Lp0, mat_to_rot6d(LR0), [0.0],
        ]).astype(np.float32)
        return torch.from_numpy(vec).unsqueeze(0)  # (1, 20)

    def get_baseline_ee(self) -> torch.Tensor | None:
        """Return the cached (1, 20) world-frame EE pose of the episode's FIRST frame (A0).

        Used by the inference-time :class:`EpisodeEEToWorldStep` to lift the model's
        episode-relative action ``S_{t+k}`` back into the world frame: ``A_{t+k} = A0 · S_{t+k}``.
        ``None`` until the first observation of an episode has been seen.
        """
        return self._a0_packed

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Update the declared shape of ``observation.state`` from joints to EE dim."""
        from vtla.engine.configs import FeatureType

        for bucket in features.values():
            if OBS_STATE in bucket:
                ft = bucket[OBS_STATE]
                if ft.type is FeatureType.STATE:
                    bucket[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(EE_DIM,))
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "state_feature_names": self.state_feature_names,
            "relative_to_baseline": self.relative_to_baseline,
        }
