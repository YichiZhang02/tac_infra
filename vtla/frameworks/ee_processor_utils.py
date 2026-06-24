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

"""Shared EE-mode processor wiring used by all policy processor factories (pi05 / act / diffusion).

EE modes (state_mode='episode_ee', action_mode='relative_ee') reuse the existing relative-action
machinery: route_ee_batch (in train.py) puts the episode_ee state and action_episode_ee chunk under
the canonical observation.state / action keys, then these helpers (a) remap the normalization stats
to those canonical keys and (b) build the pose-aware relative/absolute steps.
"""

from vtla.engine.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from vtla.engine.utils.constants import ACTION, OBS_STATE

from .sensor_routing import ACTION_RELATIVE_EE, OBS_STATE_EPISODE_EE


def remap_ee_dataset_stats(dataset_stats, config):
    """Return ``dataset_stats`` with EE stats placed under the canonical keys (shallow copy).

    - state_mode='episode_ee':  observation.state  <- observation.state_episode_ee stats
    - action_mode='relative_ee': action            <- action_relative_ee stats (relative St^-1·S_{t+k})

    A no-op (returns the input) for joint modes.
    """
    if dataset_stats is None:
        return dataset_stats
    state_ee = getattr(config, "state_mode", "joint") == "episode_ee"
    action_ee = getattr(config, "action_mode", "joint") == "relative_ee"
    if not (state_ee or action_ee):
        return dataset_stats

    dataset_stats = dict(dataset_stats)
    if state_ee and OBS_STATE_EPISODE_EE in dataset_stats:
        dataset_stats[OBS_STATE] = dataset_stats[OBS_STATE_EPISODE_EE]
    if action_ee:
        if ACTION_RELATIVE_EE not in dataset_stats:
            raise KeyError(
                f"action_mode='relative_ee' needs '{ACTION_RELATIVE_EE}' stats. Re-run "
                "tools/convert_joints_to_eepose.py to (re)generate them."
            )
        dataset_stats[ACTION] = dataset_stats[ACTION_RELATIVE_EE]
    return dataset_stats


def make_ee_relative_steps(config):
    """Build the paired (relative, absolute) action steps for a policy processor.

    In EE mode (action_mode='relative_ee') they run in SE(3) ``pose`` mode; otherwise they fall back
    to the joint element-wise behaviour gated by the pi05-only ``use_relative_actions`` flag (a no-op
    for act/diffusion, which lack that flag). The relative step (preprocess) caches the reference
    state; the absolute step (postprocess) reverses the conversion at inference.
    """
    action_ee = getattr(config, "action_mode", "joint") == "relative_ee"
    enabled = getattr(config, "use_relative_actions", False) or action_ee
    relative_step = RelativeActionsProcessorStep(
        enabled=enabled,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
        mode="pose" if action_ee else "joint",
        n_arms=getattr(config, "ee_num_arms", 2),
    )
    absolute_step = AbsoluteActionsProcessorStep(enabled=enabled, relative_step=relative_step)
    return relative_step, absolute_step
