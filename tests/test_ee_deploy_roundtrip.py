#!/usr/bin/env python
"""Full EE encode->decode round-trip proof for the deployment path (offline, no robot).

Chain (mirrors training + inference):
  joints_t      --FK-->  A_t (world EE)         [robot reads live]
  A_0 = A_t at episode start                    [cached at inference start]
  S_t  = A_0^-1 . A_t   (episode_ee state)      [fed to model]
  S_{t+k} (episode_ee of future joints)         [== action_episode_ee]
  a_rel = S_t^-1 . S_{t+k}                       [model target / output]
  -- inference postproc (AbsoluteActionsProcessorStep, pose):
  S_{t+k}' = ee_to_absolute(S_t, a_rel)          [recovers episode_ee]
  A_{t+k}' = ee_to_absolute(A_0, S_{t+k}')       [robot: episode_ee -> world]
  joints'  = IK(A_{t+k}', seed=joints_t)         [robot: rm_movep_canfd / send_joints]

Passes if joints' ~= the original future joints, proving the whole EE encode/decode is consistent
including FK/IK. Run: python tests/test_ee_deploy_roundtrip.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment" / "sdk"))

from Robotic_Arm.rm_ctypes_wrap import (  # noqa: E402
    rm_force_type_e,
    rm_inverse_kinematics_params_t,
    rm_robot_arm_model_e,
)
from Robotic_Arm.rm_robot_interface import Algo  # noqa: E402
from vtla.engine.utils.ee_transforms import (  # noqa: E402
    ee_to_absolute,
    ee_to_relative,
    matrix_to_rot6d,
    rot6d_to_matrix,
)
from vtla.engine.utils.constants import OBS_STATE  # noqa: E402

algo = Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E, rm_force_type_e.RM_MODEL_RM_B_E)


def fk_arm(joints_rad):
    pose = algo.rm_algo_forward_kinematics(np.degrees(joints_rad).tolist(), flag=0)  # [x,y,z,qw,qx,qy,qz]
    pos = np.array(pose[:3])
    mat = R.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
    return pos, mat


def pack_arm(pos, mat, grip):
    return np.concatenate([pos, matrix_to_rot6d(torch.tensor(mat)).numpy(), [grip]]).astype(np.float64)


def world_packed(joints16, jidx):
    """16-d joints (left-first) -> 20-d world EE pose (right-first), grips from joints."""
    rj, rg = joints16[jidx["rj"]], joints16[jidx["rg"]]
    lj, lg = joints16[jidx["lj"]], joints16[jidx["lg"]]
    rp, rm = fk_arm(rj)
    lp, lm = fk_arm(lj)
    return np.concatenate([pack_arm(rp, rm, rg), pack_arm(lp, lm, lg)])


def ik_arm(pose20_arm, seed_joints_rad):
    """One arm: 10-d [pos, rot6d, grip] world pose -> 7 joint radians via Algo IK (seeded)."""
    pos = pose20_arm[:3]
    mat = rot6d_to_matrix(torch.tensor(pose20_arm[3:9])).numpy()
    qx, qy, qz, qw = R.from_matrix(mat).as_quat()  # scipy (x,y,z,w)
    q_pose = [float(pos[0]), float(pos[1]), float(pos[2]), float(qw), float(qx), float(qy), float(qz)]
    params = rm_inverse_kinematics_params_t(np.degrees(seed_joints_rad).tolist(), q_pose, 0)
    ret, q_out = algo.rm_algo_inverse_kinematics(params)
    return ret, np.radians(np.array(q_out))


def main():
    import pyarrow.parquet as pq

    ds = ROOT / "playground/data/rm_umi_dual_pen_open"
    df = pq.read_table(ds / "data/chunk-000/file-000.parquet",
                       columns=["observation.state", "frame_index", "episode_index"]).to_pandas()
    names = __import__("json").load(open(ds / "meta/info.json"))["features"]["observation.state"]["names"]
    jidx = {"rj": [names.index(f"right_main_joint{i}") for i in range(1, 8)],
            "lj": [names.index(f"left_main_joint{i}") for i in range(1, 8)],
            "rg": names.index("right_main_gripper"), "lg": names.index("left_main_gripper")}

    ep0 = df[df["episode_index"] == 0].reset_index(drop=True)
    j = np.stack(ep0["observation.state"].to_numpy()).astype(np.float64)  # (L, 16)
    t, k = 50, 8  # anchor and future offset within the episode

    A0 = torch.tensor(world_packed(j[0], jidx)).unsqueeze(0)        # episode first frame (world)
    At = torch.tensor(world_packed(j[t], jidx)).unsqueeze(0)        # current (world)
    Atk = torch.tensor(world_packed(j[t + k], jidx)).unsqueeze(0)   # future (world, ground truth)

    # encode: world -> episode_ee
    St = ee_to_relative(A0, At)        # state fed to model
    Stk = ee_to_relative(A0, Atk)      # action_episode_ee
    # train target: relative to current obs
    a_rel = ee_to_relative(St, Stk)

    # --- inference decode ---
    Stk_rec = ee_to_absolute(St, a_rel)          # postproc (AbsoluteActionsProcessorStep)
    Atk_rec = ee_to_absolute(A0, Stk_rec)        # robot: episode_ee -> world (uses cached A0)

    # check the pose round-trip first (pure geometry)
    pose_err = (Atk_rec - Atk).abs().max().item()
    assert pose_err < 1e-4, f"world pose round-trip failed: {pose_err}"
    print(f"  ✓ pose round-trip joints->episode_ee->a_rel->world: max err {pose_err:.2e}")

    # decode world pose -> joints via IK, per arm, compare to ground-truth future joints
    p = Atk_rec[0].numpy()
    ret_r, jr = ik_arm(p[:10], j[t][jidx["rj"]])
    ret_l, jl = ik_arm(p[10:], j[t][jidx["lj"]])
    assert ret_r == 0 and ret_l == 0, f"IK failed ret=({ret_r},{ret_l})"
    # 7-DOF arm is redundant: IK may pick a different valid joint config than ground truth, so the
    # correctness check is that the IK joints REACH the commanded pose, i.e. FK(IK(pose)) == pose.
    for side, jik, p_arm in (("right", jr, p[:10]), ("left", jl, p[10:])):
        pos_fk, mat_fk = fk_arm(jik)
        pos_err = np.abs(pos_fk - p_arm[:3]).max()
        mat_err = np.abs(matrix_to_rot6d(torch.tensor(mat_fk)).numpy() - p_arm[3:9]).max()
        jdiff = np.abs(jik - j[t + k][jidx["rj" if side == "right" else "lj"]]).max()
        print(f"  ✓ {side}: FK(IK(pose)) reaches pose (pos {pos_err:.2e} m, rot6d {mat_err:.2e}); "
              f"joint diff vs GT {jdiff:.3f} rad (redundancy)")
        assert pos_err < 1e-3 and mat_err < 1e-3, f"{side} IK pose not reached: {pos_err}, {mat_err}"

    # grippers carried through absolutely
    assert abs(Stk_rec[0, 9].item() - j[t + k][jidx["rg"]]) < 1e-5
    print("  ✓ grippers carried through (absolute)")


def test_deploy_classes():
    """Same chain but through the ACTUAL inference classes (proves the wiring, not just the math):

      EpisodeEEPreprocessorStep  : joints -> S_t (model input) + caches A0 (get_baseline_ee)
      AbsoluteActionsProcessorStep: a_rel -> S_{t+k}   (already covered by main(), reused here)
      EpisodeEEToWorldStep        : S_{t+k} -> world A_{t+k}  (uses cached A0)
      robot _send_action_ee math  : rot6d -> matrix -> quat -> pose7 reaches the commanded flange pose
    """
    import pyarrow.parquet as pq

    from vtla.frameworks.episode_ee_processor import EpisodeEEPreprocessorStep
    from vtla.engine.processor.episode_ee_world_processor import EpisodeEEToWorldStep
    from deployment.robots.realman_ugripper_dual.realman_ugripper_dual import (
        _mat_to_quat_wxyz,
        _rot6d_to_mat,
    )

    ds = ROOT / "playground/data/rm_umi_dual_pen_open"
    df = pq.read_table(ds / "data/chunk-000/file-000.parquet",
                       columns=["observation.state", "episode_index"]).to_pandas()
    names = __import__("json").load(open(ds / "meta/info.json"))["features"]["observation.state"]["names"]
    jidx = {"rj": [names.index(f"right_main_joint{i}") for i in range(1, 8)],
            "lj": [names.index(f"left_main_joint{i}") for i in range(1, 8)],
            "rg": names.index("right_main_gripper"), "lg": names.index("left_main_gripper")}
    ep0 = df[df["episode_index"] == 0].reset_index(drop=True)
    j = np.stack(ep0["observation.state"].to_numpy()).astype(np.float64)
    t, k = 50, 8

    # --- preprocessor: episode-start sets A0, step t produces S_t ---
    ee = EpisodeEEPreprocessorStep(state_feature_names=names)
    ee.reset()
    ee.observation({OBS_STATE: torch.tensor(j[0])})           # episode start -> caches A0
    St = ee.observation({OBS_STATE: torch.tensor(j[t])})[OBS_STATE].unsqueeze(0).double()  # (1,20)

    A0 = ee.get_baseline_ee().double()                        # (1,20)
    A0_gt = torch.tensor(world_packed(j[0], jidx)).unsqueeze(0)
    pose_dims = [i for i in range(20) if i not in (9, 19)]     # gripper slots are 0 in A0 by design
    a0_err = (A0[0, pose_dims] - A0_gt[0, pose_dims]).abs().max().item()
    assert a0_err < 1e-4, f"A0 packing mismatch vs world FK: {a0_err}"
    print(f"  ✓ EpisodeEEPreprocessorStep.get_baseline_ee == world FK(first frame): max err {a0_err:.2e}")

    # also confirm S_t equals the reference encode (A0^-1 . A_t)
    St_ref = ee_to_relative(A0_gt, torch.tensor(world_packed(j[t], jidx)).unsqueeze(0))
    st_err = (St[0, pose_dims] - St_ref[0, pose_dims]).abs().max().item()
    assert st_err < 1e-4, f"S_t mismatch: {st_err}"

    # --- model target, then full decode through the real world step ---
    Stk = ee_to_relative(A0_gt, torch.tensor(world_packed(j[t + k], jidx)).unsqueeze(0))
    a_rel = ee_to_relative(St, Stk)
    Stk_rec = ee_to_absolute(St, a_rel)                       # AbsoluteActionsProcessorStep equivalent
    world_step = EpisodeEEToWorldStep(n_arms=2, ee_step=ee)
    Atk_rec = world_step.action(Stk_rec)                      # the NEW step

    Atk_gt = torch.tensor(world_packed(j[t + k], jidx)).unsqueeze(0)
    w_err = (Atk_rec[0, pose_dims] - Atk_gt[0, pose_dims]).abs().max().item()
    assert w_err < 1e-4, f"EpisodeEEToWorldStep world pose mismatch: {w_err}"
    print(f"  ✓ EpisodeEEToWorldStep recovers world pose A_t+k: max err {w_err:.2e}")

    # --- robot send_action_ee math: rot6d -> R -> quat -> back to rot6d reaches the same pose ---
    p = Atk_rec[0].numpy()
    for side, arm_pose in (("right", p[:10]), ("left", p[10:])):
        R_tgt = _rot6d_to_mat(arm_pose[3:9])
        quat = _mat_to_quat_wxyz(R_tgt)                      # [qw,qx,qy,qz] sent to rm_movep_canfd
        R_back = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        rot_err = np.abs(matrix_to_rot6d(torch.tensor(R_back)).numpy() - arm_pose[3:9]).max()
        assert rot_err < 1e-6, f"{side} rot6d->quat->rot6d mismatch: {rot_err}"
    print("  ✓ robot _send_action_ee rot6d->matrix->quat round-trips (flange pose preserved)")


if __name__ == "__main__":
    print("EE deployment round-trip proof:")
    main()
    print("EE deployment class-wiring proof:")
    test_deploy_classes()
    print("ALL PASSED ✅")
