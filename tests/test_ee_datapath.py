#!/usr/bin/env python
"""Smoke test for the EE (episode_ee / relative_ee) training data path on a real converted dataset.

Exercises: config feature selection -> dataloader chunking (delta_indices) -> route_ee_batch ->
pose-aware RelativeActionsProcessorStep, and checks shapes + relative correctness against an
independent ee_to_relative call. Does NOT load the heavy pi05 model.

Run: python tests/test_ee_datapath.py [dataset_root]
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deployment" / "sdk"))

from vtla.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from vtla.datasets.factory import resolve_delta_timestamps  # noqa: E402
import vtla.datasets.lerobot_dataset as _ld  # noqa: E402

# Local dataset (not on the hub): skip the hub version check + download.
_ld.get_safe_version = lambda repo_id, rev: rev
LeRobotDataset = _ld.LeRobotDataset
LeRobotDataset._download = lambda self, *a, **k: None

# This test only verifies the state/action data path; stub video decode (the env's torchcodec
# backend is broken) so fetching a batch doesn't touch the real videos.
import vtla.datasets.dataset_reader as _dr  # noqa: E402
_dr.decode_video_frames = lambda video_path, timestamps, *a, **k: torch.zeros(len(timestamps), 3, 2, 2)
from vtla.engine.processor.relative_action_processor import RelativeActionsProcessorStep, route_ee_batch  # noqa: E402
from vtla.engine.types import TransitionKey  # noqa: E402
from vtla.engine.utils.constants import ACTION, OBS_STATE  # noqa: E402
from vtla.engine.utils.ee_transforms import ee_to_relative  # noqa: E402
from vtla.engine.utils.feature_utils import dataset_to_policy_features  # noqa: E402
from vtla.frameworks.pi05.configuration_pi05 import PI05Config  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ee_test_ds"
ATOL = 1e-4


def build_ee_config(meta) -> PI05Config:
    cfg = PI05Config(
        state_mode="episode_ee", action_mode="relative_ee", chunk_size=8, n_action_steps=8,
        top_camera_keys=["observation.images.cam_top"],
        wrist_camera_keys=["observation.images.left_cam_wrist", "observation.images.right_cam_wrist"],
    )
    feats = dataset_to_policy_features(meta.features)
    cfg.output_features = {k: ft for k, ft in feats.items() if ft.type.name == "ACTION"}
    cfg.input_features = {k: ft for k, ft in feats.items() if k not in cfg.output_features}
    cfg.validate_features()
    return cfg


def main():
    meta = LeRobotDatasetMetadata(repo_id=Path(ROOT).name, root=ROOT)
    cfg = build_ee_config(meta)

    # --- config feature selection ---
    assert OBS_STATE in cfg.input_features and cfg.input_features[OBS_STATE].shape == (20,), \
        f"observation.state must be the 20-d episode_ee feature, got {cfg.input_features.get(OBS_STATE)}"
    assert "observation.state_episode_ee" not in cfg.input_features, "raw episode_ee key must be dropped"
    assert cfg.output_features[ACTION].shape == (20,), f"action must be 20-d, got {cfg.output_features[ACTION]}"
    assert "action_episode_ee" not in cfg.output_features, "raw action_episode_ee key must be dropped"
    assert cfg.action_delta_indices == list(range(1, cfg.chunk_size + 1)), "relative_ee chunk must start at t+1"
    print("  ✓ config: observation.state(20) + action(20), chunk starts t+1")

    # --- dataloader chunking ---
    delta = resolve_delta_timestamps(cfg, meta)
    assert "action_episode_ee" in delta and len(delta["action_episode_ee"]) == cfg.chunk_size, \
        "action_episode_ee must receive the action horizon"
    ds = LeRobotDataset(repo_id=Path(ROOT).name, root=ROOT, delta_timestamps=delta)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    assert batch["action_episode_ee"].shape[1:] == (cfg.chunk_size, 20), batch["action_episode_ee"].shape
    print(f"  ✓ dataloader: action_episode_ee chunked to {tuple(batch['action_episode_ee'].shape)},"
          f" state_episode_ee {tuple(batch['observation.state_episode_ee'].shape)}")

    # keep raw copies for the independent check, then route
    state_ee_raw = batch["observation.state_episode_ee"].clone().float()
    action_ee_raw = batch["action_episode_ee"].clone().float()
    batch = route_ee_batch(batch, cfg.state_mode, cfg.action_mode)
    assert OBS_STATE in batch and ACTION in batch, "routing must populate canonical keys"
    assert "action_episode_ee" not in batch, "routing must consume the ee action column"
    print("  ✓ route_ee_batch: episode_ee -> observation.state, action_episode_ee -> action")

    # --- pose-aware relative step ---
    step = RelativeActionsProcessorStep(enabled=True, mode="pose", n_arms=2)
    transition = {TransitionKey.OBSERVATION: {OBS_STATE: batch[OBS_STATE].float()},
                  TransitionKey.ACTION: batch[ACTION].float()}
    out = step(transition)
    rel = out[TransitionKey.ACTION]
    assert rel.shape == batch[ACTION].shape, rel.shape

    # state reference: squeeze the obs-history dim if present (B,1,20)->(B,20)
    ref = state_ee_raw.squeeze(1) if state_ee_raw.ndim == 3 else state_ee_raw
    expected = ee_to_relative(ref, action_ee_raw, n_arms=2)
    err = (rel - expected).abs().max().item()
    assert err < ATOL, f"pose relative mismatch vs independent ee_to_relative: {err}"
    print(f"  ✓ pose relative step matches independent ee_to_relative (max err {err:.2e})")

    # gripper dims (idx 9 and 19) must stay absolute (== raw action gripper)
    g_err = (rel[..., [9, 19]] - action_ee_raw[..., [9, 19]]).abs().max().item()
    assert g_err < 1e-6, f"gripper must be passthrough, err {g_err}"
    print(f"  ✓ grippers kept absolute (err {g_err:.1e})")


if __name__ == "__main__":
    print("EE data-path smoke test:")
    main()
    print("ALL PASSED ✅")
