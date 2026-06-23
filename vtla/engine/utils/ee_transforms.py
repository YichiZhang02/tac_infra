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

"""Batched (torch) end-effector pose transforms for the EE action/state modes.

A "packed" EE pose vector concatenates, per arm, ``[pos(3), rot6d(6), gripper(1)]`` = 10 dims.
For a dual-arm robot the full vector is 20 dims, ordered ``right`` arm first then ``left``
(matching the offline ``convert_joints_to_eepose`` layout).

``rot6d`` is the first two columns of the rotation matrix (Zhou et al. 2019). It is recovered
to a full rotation matrix with Gram-Schmidt, so ``matrix_to_rot6d`` / ``rot6d_to_matrix`` round-trip.

Two conversions (per arm; positions and rotations in the reference's *local* frame, gripper kept
absolute — mirrors the ``T0^{-1}·Tt`` convention of the offline script):

- relative  (action -> relative-to-reference):  p_rel = R_s^T (p_a - p_s),  R_rel = R_s^T R_a
- absolute  (relative -> back to reference frame): p_a = p_s + R_s p_rel,  R_a = R_s R_rel

where ``s`` is the reference pose (the current observation EE pose) and ``a`` is the action pose.
The two are exact inverses.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Per-arm packed layout: [pos(0:3), rot6d(3:9), gripper(9:10)].
PER_ARM_DIM = 10
_POS = slice(0, 3)
_ROT6D = slice(3, 9)
_GRIP = slice(9, 10)


def matrix_to_rot6d(matrix: Tensor) -> Tensor:
    """``(..., 3, 3)`` rotation matrix -> ``(..., 6)`` rot6d (first two columns)."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """``(..., 6)`` rot6d -> ``(..., 3, 3)`` rotation matrix via Gram-Schmidt (Zhou 2019)."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    # Remove the b1 component from a2, then normalize.
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    # Columns of the rotation matrix are b1, b2, b3.
    return torch.stack([b1, b2, b3], dim=-1)


def _unpack(x: Tensor, n_arms: int) -> tuple[Tensor, Tensor, Tensor]:
    """Packed ``(..., n_arms*10)`` -> (pos ``(..., n_arms, 3)``, R ``(..., n_arms, 3, 3)``, grip ``(..., n_arms, 1)``)."""
    if x.shape[-1] != n_arms * PER_ARM_DIM:
        raise ValueError(f"Expected last dim {n_arms * PER_ARM_DIM} for {n_arms} arms, got {x.shape[-1]}")
    blocks = x.reshape(*x.shape[:-1], n_arms, PER_ARM_DIM)
    pos = blocks[..., _POS]
    rot = rot6d_to_matrix(blocks[..., _ROT6D])
    grip = blocks[..., _GRIP]
    return pos, rot, grip


def _pack(pos: Tensor, rot: Tensor, grip: Tensor) -> Tensor:
    """Inverse of :func:`_unpack`: (pos, R, grip) -> packed ``(..., n_arms*10)``."""
    blocks = torch.cat([pos, matrix_to_rot6d(rot), grip], dim=-1)
    return blocks.reshape(*blocks.shape[:-2], blocks.shape[-2] * PER_ARM_DIM)


def _align_reference(reference: Tensor, other: Tensor) -> Tensor:
    """Broadcast the per-sample ``reference`` over any extra (e.g. chunk) dims that ``other`` has."""
    while reference.ndim < other.ndim:
        reference = reference.unsqueeze(-2)
    return reference


def ee_to_relative(reference_ee: Tensor, action_ee: Tensor, n_arms: int = 2) -> Tensor:
    """Convert absolute(-in-reference-frame) action poses to poses relative to ``reference_ee``.

    Args:
        reference_ee: ``(B, n_arms*10)`` reference EE pose (the current observation).
        action_ee: ``(B, n_arms*10)`` or ``(B, T, n_arms*10)`` action EE pose(s).
        n_arms: Number of arms packed in the vector.

    Returns:
        Relative EE pose(s), same shape as ``action_ee``. Gripper dims are passed through (absolute).
    """
    reference_ee = _align_reference(reference_ee, action_ee)
    p_s, R_s, _ = _unpack(reference_ee, n_arms)
    p_a, R_a, grip_a = _unpack(action_ee, n_arms)

    R_s_T = R_s.transpose(-1, -2)
    p_rel = torch.matmul(R_s_T, (p_a - p_s).unsqueeze(-1)).squeeze(-1)
    R_rel = torch.matmul(R_s_T, R_a)
    return _pack(p_rel, R_rel, grip_a)


def ee_to_absolute(reference_ee: Tensor, relative_ee: Tensor, n_arms: int = 2) -> Tensor:
    """Inverse of :func:`ee_to_relative`: convert relative poses back into ``reference_ee``'s frame.

    Args:
        reference_ee: ``(B, n_arms*10)`` reference EE pose (the current observation).
        relative_ee: ``(B, n_arms*10)`` or ``(B, T, n_arms*10)`` relative EE pose(s).
        n_arms: Number of arms packed in the vector.

    Returns:
        Absolute(-in-reference-frame) EE pose(s), same shape as ``relative_ee``.
    """
    reference_ee = _align_reference(reference_ee, relative_ee)
    p_s, R_s, _ = _unpack(reference_ee, n_arms)
    p_rel, R_rel, grip = _unpack(relative_ee, n_arms)

    p_a = p_s + torch.matmul(R_s, p_rel.unsqueeze(-1)).squeeze(-1)
    R_a = torch.matmul(R_s, R_rel)
    return _pack(p_a, R_a, grip)
