from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias, TypedDict

import numpy as np
import torch


class TransitionKey(str, Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    REWARD = "reward"
    DONE = "done"
    TRUNCATED = "truncated"
    INFO = "info"
    COMPLEMENTARY_DATA = "complementary_data"


PolicyAction: TypeAlias = torch.Tensor
RobotAction: TypeAlias = dict[str, Any]
EnvAction: TypeAlias = np.ndarray
RobotObservation: TypeAlias = dict[str, Any]

EnvTransition = TypedDict(
    "EnvTransition",
    {
        TransitionKey.OBSERVATION.value: dict[str, Any] | None,
        TransitionKey.ACTION.value: PolicyAction | RobotAction | EnvAction | None,
        TransitionKey.REWARD.value: float | torch.Tensor | None,
        TransitionKey.DONE.value: bool | torch.Tensor | None,
        TransitionKey.TRUNCATED.value: bool | torch.Tensor | None,
        TransitionKey.INFO.value: dict[str, Any] | None,
        TransitionKey.COMPLEMENTARY_DATA.value: dict[str, Any] | None,
    },
)
