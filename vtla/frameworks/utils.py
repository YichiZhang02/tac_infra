#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
from collections import deque

import numpy as np
import torch
from torch import nn

from vtla.engine.configs import FeatureType, PolicyFeature, PreTrainedConfig
from vtla.engine.utils.constants import ACTION, OBS_STR
from vtla.engine.utils.feature_utils import build_dataset_frame
from vtla.engine.types import PolicyAction, RobotAction, RobotObservation


def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
):
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module (nn.Module): a PyTorch module
        input_shape (tuple): A tuple representing the input shape, e.g., (batch_size, channels, height, width)

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


def log_model_loading_keys(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    model_name: str | None = None,
    max_display: int = 20,
) -> None:
    """Log missing and unexpected keys when loading a model checkpoint.

    Always emits a one-line summary with the counts (so that a clean load is
    visible too, not just a silent success), followed by the actual key names
    (truncated to ``max_display``) whenever any are present.

    Args:
        missing_keys (list[str]): Keys expected by the model but not found in the checkpoint.
        unexpected_keys (list[str]): Keys present in the checkpoint but not used by the model.
        model_name (str | None): Optional model/policy name used to tag the log lines.
        max_display (int): Maximum number of keys to list before truncating.
    """
    tag = f"[{model_name}] " if model_name else ""
    missing_keys = list(missing_keys)
    unexpected_keys = list(unexpected_keys)

    def _fmt(keys: list[str]) -> str:
        shown = keys[:max_display]
        out = "".join(f"\n  - {k}" for k in shown)
        if len(keys) > max_display:
            out += f"\n  ... and {len(keys) - max_display} more"
        return out

    logging.info(
        f"{tag}Loaded checkpoint weights: "
        f"{len(missing_keys)} missing key(s), {len(unexpected_keys)} unexpected key(s)."
    )
    if missing_keys:
        logging.warning(f"{tag}Missing key(s) when loading model:{_fmt(missing_keys)}")
    if unexpected_keys:
        logging.warning(f"{tag}Unexpected key(s) when loading model:{_fmt(unexpected_keys)}")
    if not missing_keys and not unexpected_keys:
        logging.info(f"{tag}All keys matched: model weights fully loaded from checkpoint.")


# TODO(Steven): Move this function to a proper preprocessor step
def prepare_observation_for_inference(
    observation: dict[str, np.ndarray],
    device: torch.device,
    task: str | None = None,
    robot_type: str | None = None,
) -> RobotObservation:
    """Converts observation data to model-ready PyTorch tensors.

    This function takes a dictionary of NumPy arrays, performs necessary
    preprocessing, and prepares it for model inference. The steps include:
    1. Converting NumPy arrays to PyTorch tensors.
    2. Normalizing and permuting image data (if any).
    3. Adding a batch dimension to each tensor.
    4. Moving all tensors to the specified compute device.
    5. Adding task and robot type information to the dictionary.

    Args:
        observation: A dictionary mapping observation names (str) to NumPy
            array data. For images, the format is expected to be (H, W, C).
        device: The PyTorch device (e.g., 'cpu' or 'cuda') to which the
            tensors will be moved.
        task: An optional string identifier for the current task.
        robot_type: An optional string identifier for the robot being used.

    Returns:
        A dictionary where values are PyTorch tensors preprocessed for
        inference, residing on the target device. Image tensors are reshaped
        to (C, H, W) and normalized to a [0, 1] range.
    """
    for name in observation:
        observation[name] = torch.from_numpy(observation[name])
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""

    return observation


def build_inference_frame(
    observation: RobotObservation,
    device: torch.device,
    ds_features: dict[str, dict],
    task: str | None = None,
    robot_type: str | None = None,
) -> RobotObservation:
    """Constructs a model-ready observation tensor dict from a raw observation.

    This utility function orchestrates the process of converting a raw,
    unstructured observation from an environment into a structured,
    tensor-based format suitable for passing to a policy model.

    Args:
        observation: The raw observation dictionary, which may contain
            superfluous keys.
        device: The target PyTorch device for the final tensors.
        ds_features: A configuration dictionary that specifies which features
            to extract from the raw observation.
        task: An optional string identifier for the current task.
        robot_type: An optional string identifier for the robot being used.

    Returns:
        A dictionary of preprocessed tensors ready for model inference.
    """
    # Extracts the correct keys from the incoming raw observation
    observation = build_dataset_frame(ds_features, observation, prefix=OBS_STR)

    # Performs the necessary conversions to the observation
    observation = prepare_observation_for_inference(observation, device, task, robot_type)

    return observation


def make_robot_action(action_tensor: PolicyAction, ds_features: dict[str, dict]) -> RobotAction:
    """Converts a policy's output tensor into a dictionary of named actions.

    This function translates the numerical output from a policy model into a
    human-readable and robot-consumable format, where each dimension of the
    action tensor is mapped to a named motor or actuator command.

    Args:
        action_tensor: A PyTorch tensor representing the policy's action,
            typically with a batch dimension (e.g., shape [1, action_dim]).
        ds_features: A configuration dictionary containing metadata, including
            the names corresponding to each index of the action tensor.

    Returns:
        A dictionary mapping action names (e.g., "joint_1_motor") to their
        corresponding floating-point values, ready to be sent to a robot
        controller.
    """
    # TODO(Steven): Check if these steps are already in all postprocessor policies
    action_tensor = action_tensor.squeeze(0)
    action_tensor = action_tensor.to("cpu")

    action_names = ds_features[ACTION]["names"]
    act_processed_policy: RobotAction = {
        f"{name}": float(action_tensor[i]) for i, name in enumerate(action_names)
    }
    return act_processed_policy


def raise_feature_mismatch_error(
    provided_features: set[str],
    expected_features: set[str],
) -> None:
    """
    Raises a standardized ValueError for feature mismatches between dataset/environment and policy config.
    """
    missing = expected_features - provided_features
    extra = provided_features - expected_features
    # TODO (jadechoghari): provide a dynamic rename map suggestion to the user.
    raise ValueError(
        f"Feature mismatch between dataset/environment and policy config.\n"
        f"- Missing features: {sorted(missing) if missing else 'None'}\n"
        f"- Extra features: {sorted(extra) if extra else 'None'}\n\n"
        f"Please ensure your dataset and policy use consistent feature names.\n"
        f"If your dataset uses different observation keys (e.g., cameras named differently), "
        f"use the `--rename_map` argument, for example:\n"
        f'  --rename_map=\'{{"observation.images.left": "observation.images.camera1", '
        f'"observation.images.top": "observation.images.camera2"}}\''
    )


def validate_visual_features_consistency(
    cfg: PreTrainedConfig,
    features: dict[str, PolicyFeature],
) -> None:
    """
    Validates visual feature consistency between a policy config and provided dataset/environment features.

    Validation passes if EITHER:
    - Policy's expected visuals are a subset of dataset (policy uses some cameras, dataset has more)
    - Dataset's provided visuals are a subset of policy (policy declares extras for flexibility)

    Args:
        cfg (PreTrainedConfig): The model or policy configuration containing input_features and type.
        features (Dict[str, PolicyFeature]): A mapping of feature names to PolicyFeature objects.
    """
    expected_visuals = {k for k, v in cfg.input_features.items() if v.type == FeatureType.VISUAL}
    provided_visuals = {k for k, v in features.items() if v.type == FeatureType.VISUAL}

    # Accept if either direction is a subset
    policy_subset_of_dataset = expected_visuals.issubset(provided_visuals)
    dataset_subset_of_policy = provided_visuals.issubset(expected_visuals)

    if not (policy_subset_of_dataset or dataset_subset_of_policy):
        raise_feature_mismatch_error(provided_visuals, expected_visuals)
