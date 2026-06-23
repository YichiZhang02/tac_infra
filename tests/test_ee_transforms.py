#!/usr/bin/env python
"""Standalone verification for vtla.engine.utils.ee_transforms.

No pytest dependency: run directly ``python tests/test_ee_transforms.py``.
Cross-checks the torch implementation against scipy where applicable.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vtla.engine.utils.ee_transforms import (  # noqa: E402
    ee_to_absolute,
    ee_to_relative,
    matrix_to_rot6d,
    rot6d_to_matrix,
)

torch.manual_seed(0)
np.random.seed(0)
ATOL = 1e-5


def _rand_rotmats(n: int) -> torch.Tensor:
    """n random valid rotation matrices (via scipy) as a (n, 3, 3) tensor."""
    mats = R.random(n).as_matrix()
    return torch.tensor(mats, dtype=torch.float64)


def _rand_pose_vec(shape_arms) -> torch.Tensor:
    """Random packed EE vector with VALID rotations. shape_arms: (..., n_arms)."""
    *batch, n_arms = shape_arms
    total = int(np.prod(batch)) * n_arms if batch else n_arms
    mats = _rand_rotmats(total)
    rot6d = matrix_to_rot6d(mats).reshape(*batch, n_arms, 6)
    pos = torch.randn(*batch, n_arms, 3, dtype=torch.float64)
    grip = torch.rand(*batch, n_arms, 1, dtype=torch.float64)
    vec = torch.cat([pos, rot6d, grip], dim=-1)  # (..., n_arms, 10)
    return vec.reshape(*batch, n_arms * 10)


def test_rot6d_roundtrip():
    mats = _rand_rotmats(64)
    rt = rot6d_to_matrix(matrix_to_rot6d(mats))
    assert torch.allclose(rt, mats, atol=ATOL), "rot6d->matrix->rot6d must be identity"
    # rot6d_to_matrix must yield valid rotations (orthonormal, det +1) even from noisy input.
    noisy = torch.randn(32, 6, dtype=torch.float64)
    m = rot6d_to_matrix(noisy)
    eye = torch.eye(3, dtype=torch.float64).expand_as(torch.matmul(m.transpose(-1, -2), m))
    assert torch.allclose(torch.matmul(m.transpose(-1, -2), m), eye, atol=ATOL), "must be orthonormal"
    assert torch.allclose(torch.det(m), torch.ones(32, dtype=torch.float64), atol=ATOL), "det must be +1"
    print("  ✓ rot6d roundtrip + orthonormalization")


def test_relative_absolute_roundtrip():
    for shape in [(8, 2), (8, 1)]:  # dual-arm and single-arm
        n_arms = shape[-1]
        ref = _rand_pose_vec(shape)
        act = _rand_pose_vec(shape)
        rel = ee_to_relative(ref, act, n_arms=n_arms)
        back = ee_to_absolute(ref, rel, n_arms=n_arms)
        assert torch.allclose(back, act, atol=ATOL), f"absolute∘relative must be identity (shape={shape})"
    print("  ✓ absolute∘relative == identity (single + dual arm)")


def test_chunk_broadcast():
    # reference is (B, D); action is a chunk (B, T, D) sharing the same reference.
    B, T, n_arms = 4, 5, 2
    ref = _rand_pose_vec((B, n_arms))
    act = _rand_pose_vec((B, T, n_arms))
    rel = ee_to_relative(ref, act, n_arms=n_arms)
    assert rel.shape == act.shape
    back = ee_to_absolute(ref, rel, n_arms=n_arms)
    assert torch.allclose(back, act, atol=ATOL), "chunk roundtrip must hold"
    # Each chunk element must be relativized against the SAME reference: compare to per-slice call.
    for t in range(T):
        rel_t = ee_to_relative(ref, act[:, t], n_arms=n_arms)
        assert torch.allclose(rel_t, rel[:, t], atol=ATOL), "broadcast must match per-slice"
    print("  ✓ chunk broadcast (shared reference) matches per-slice")


def test_against_scipy_se3():
    # Independently verify the SE(3) math with scipy on a single sample / single arm.
    p_s = np.random.randn(3)
    p_a = np.random.randn(3)
    R_s = R.random().as_matrix()
    R_a = R.random().as_matrix()
    grip = 0.42

    def vec(p, Rm, g):
        r6 = np.concatenate([Rm[:, 0], Rm[:, 1]])
        return torch.tensor(np.concatenate([p, r6, [g]]), dtype=torch.float64).unsqueeze(0)

    ref = vec(p_s, R_s, 0.0)
    act = vec(p_a, R_a, grip)
    rel = ee_to_relative(ref, act, n_arms=1)[0].numpy()

    # Expected (reference local frame): p = R_s^T (p_a - p_s), R = R_s^T R_a.
    exp_p = R_s.T @ (p_a - p_s)
    exp_R = R_s.T @ R_a
    exp_r6 = np.concatenate([exp_R[:, 0], exp_R[:, 1]])
    assert np.allclose(rel[:3], exp_p, atol=ATOL), "relative position mismatch vs scipy"
    assert np.allclose(rel[3:9], exp_r6, atol=ATOL), "relative rotation mismatch vs scipy"
    assert np.allclose(rel[9], grip, atol=ATOL), "gripper must stay absolute"
    print("  ✓ SE(3) relative matches independent scipy computation")


def test_identity_reference():
    # If action == reference, relative pose must be identity (zero pos, identity rot), gripper kept.
    ref = _rand_pose_vec((6, 2))
    rel = ee_to_relative(ref, ref, n_arms=2).reshape(6, 2, 10)
    assert torch.allclose(rel[..., :3], torch.zeros_like(rel[..., :3]), atol=ATOL), "self-relative pos = 0"
    R_rel = rot6d_to_matrix(rel[..., 3:9])
    eye = torch.eye(3, dtype=torch.float64).expand_as(R_rel)
    assert torch.allclose(R_rel, eye, atol=ATOL), "self-relative rot = identity"
    print("  ✓ self-relative is identity (gripper preserved)")


if __name__ == "__main__":
    print("Running ee_transforms verification:")
    test_rot6d_roundtrip()
    test_relative_absolute_roundtrip()
    test_chunk_broadcast()
    test_against_scipy_se3()
    test_identity_reference()
    print("ALL PASSED ✅")
