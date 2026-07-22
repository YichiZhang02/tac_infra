"""Tests for the tactile temporal window feature.

Covers:
  1. SensorRoutingMixin.tactile_delta_indices — correctness for various F/offset.
  2. TactileEncoder.total_tokens — accounts for num_frames.
  3. TactileEncoder.forward_flat — flattens 4D and 5D inputs correctly.
  4. resolve_delta_timestamps — tactile keys get their own per-key delta.
  5. TactileTemporalWindowStep — inference-time frame buffering.
  6. expand_tactile_as_image_window — per-frame key splitting.
  7. SensorRoutingMixin.image_feature_keys_expanded — key expansion.
  8. Diffusion config — rejects as_image + frames>1.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. tactile_delta_indices
# ---------------------------------------------------------------------------
from vtla.frameworks.sensor_routing import SensorRoutingMixin


class _MockCfg(SensorRoutingMixin):
    """Minimal SensorRoutingMixin subclass for unit tests."""
    input_features: dict = field(default_factory=dict)
    output_features: dict = field(default_factory=dict)
    normalization_mapping: dict = field(default_factory=dict)

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)


def _cfg(**kwargs):
    defaults = dict(
        tactile_mode="encode",
        tactile_keys=["observation.images.finger0", "observation.images.finger1"],
        tactile_encoder_path="dummy",
        tactile_num_frames=1,
        tactile_frame_offset=1,
        tactile_num_tokens=4,
        freeze_tactile_encoder=False,
        tactile_insert_location="decoder",
        wrist_only=False,
        top_camera_keys=["observation.images.cam_top"],
        wrist_camera_keys=["observation.images.cam_wrist"],
        state_mode="joint",
        action_mode="joint",
        ee_num_arms=2,
        state_feature_names=None,
        input_features={},
        output_features={},
        normalization_mapping={},
        tactile_encoder_type=None,
    )
    defaults.update(kwargs)
    return _MockCfg(**defaults)


class TestTactileDeltaIndices:
    def test_single_frame(self):
        cfg = _cfg(tactile_num_frames=1, tactile_frame_offset=1)
        assert cfg.tactile_delta_indices() == [0]

    def test_two_frames_offset1(self):
        cfg = _cfg(tactile_num_frames=2, tactile_frame_offset=1)
        assert cfg.tactile_delta_indices() == [-1, 0]

    def test_three_frames_offset1(self):
        cfg = _cfg(tactile_num_frames=3, tactile_frame_offset=1)
        assert cfg.tactile_delta_indices() == [-2, -1, 0]

    def test_three_frames_offset2(self):
        cfg = _cfg(tactile_num_frames=3, tactile_frame_offset=2)
        assert cfg.tactile_delta_indices() == [-4, -2, 0]

    def test_five_frames_offset3(self):
        cfg = _cfg(tactile_num_frames=5, tactile_frame_offset=3)
        assert cfg.tactile_delta_indices() == [-12, -9, -6, -3, 0]

    def test_windowed_false_when_single_frame(self):
        cfg = _cfg(tactile_num_frames=1, tactile_frame_offset=1)
        assert cfg.tactile_windowed() is False

    def test_windowed_true_when_multi_frame(self):
        cfg = _cfg(tactile_num_frames=3, tactile_frame_offset=2)
        assert cfg.tactile_windowed() is True

    def test_windowed_false_when_mode_none(self):
        cfg = _cfg(tactile_mode="none", tactile_num_frames=3)
        assert cfg.tactile_windowed() is False


# ---------------------------------------------------------------------------
# 2 & 3. TactileEncoder total_tokens and forward_flat
# ---------------------------------------------------------------------------
from vtla.frameworks.tactile_encode import TactileEncoder


class _FakeExtractor(nn.Module):
    """Fake tactile MAE extractor that returns deterministic output shapes."""

    feature_dim = 16
    num_query_tokens = 4
    compute_dtype = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_keys, C, H, W] or [B*n_keys, T, C, H, W]
        if x.dim() == 5:
            b, n, c, h, w = x.shape
            return torch.zeros(b, n, self.num_query_tokens, self.feature_dim)
        elif x.dim() == 6:
            bn, t, c, h, w = x.shape
            return torch.zeros(bn, t, self.num_query_tokens, self.feature_dim)
        raise ValueError(f"Unexpected dim {x.dim()}")


def _make_encoder(num_frames=1, n_keys=2, num_tokens=4):
    cfg = _cfg(
        tactile_num_frames=num_frames,
        tactile_num_tokens=num_tokens,
        tactile_keys=[f"observation.images.finger{i}" for i in range(n_keys)],
    )
    enc = TactileEncoder.__new__(TactileEncoder)
    enc.tactile_keys = cfg.tactile_keys
    enc.extractor = _FakeExtractor()
    enc.extractor.num_query_tokens = num_tokens
    enc.output_dim = 8
    enc.num_frames = num_frames
    enc.proj = nn.Linear(16, 8)
    return enc, cfg


class TestTactileEncoderTokens:
    def test_num_tokens_single_frame(self):
        enc, _ = _make_encoder(num_frames=1, n_keys=2, num_tokens=4)
        assert enc.num_tokens == 8   # 2 keys * 4 tokens

    def test_total_tokens_single_frame_equals_num_tokens(self):
        enc, _ = _make_encoder(num_frames=1, n_keys=2, num_tokens=4)
        assert enc.total_tokens == enc.num_tokens

    def test_total_tokens_multi_frame(self):
        enc, _ = _make_encoder(num_frames=3, n_keys=2, num_tokens=4)
        assert enc.total_tokens == 3 * 8  # 3 frames * (2 keys * 4 tokens)


class TestTactileEncoderForwardFlat:
    def _batch_4d(self, n_keys=2, B=2, C=3, H=8, W=8):
        return {
            f"observation.images.finger{i}": torch.zeros(B, C, H, W)
            for i in range(n_keys)
        }

    def _batch_5d(self, n_keys=2, B=2, F=3, C=3, H=8, W=8):
        return {
            f"observation.images.finger{i}": torch.zeros(B, F, C, H, W)
            for i in range(n_keys)
        }

    def test_forward_flat_4d_input(self):
        enc, _ = _make_encoder(num_frames=1, n_keys=2, num_tokens=4)
        batch = self._batch_4d()
        out = enc.forward_flat(batch)
        # 4D input: no time dim → forward returns [B, n_keys*N, P] already 3D
        assert out.dim() == 3
        assert out.shape == (2, 8, 8)  # B=2, total_tokens=8, output_dim=8

    def test_forward_flat_5d_input_folds_time(self):
        F = 3
        enc, _ = _make_encoder(num_frames=F, n_keys=2, num_tokens=4)
        batch = self._batch_5d(F=F)
        out = enc.forward_flat(batch)
        # 5D input: time dim folded into token dim → [B, F*n_keys*N, P]
        assert out.dim() == 3
        assert out.shape == (2, F * 8, 8)


# ---------------------------------------------------------------------------
# 4. resolve_delta_timestamps — tactile keys get their own per-key delta
# ---------------------------------------------------------------------------
class TestFactoryDeltaTimestamps:
    """Smoke test that tactile keys get their own delta when windowed."""

    def _make_ds_meta(self, fps=30, tactile_keys=None):
        if tactile_keys is None:
            tactile_keys = ["observation.images.finger0"]
        meta = MagicMock()
        meta.fps = fps
        # Build an ordered dict of features that includes action + obs keys
        features = {"action": None}
        features["observation.state"] = None
        for k in tactile_keys:
            features[k] = None
        features["observation.images.cam_top"] = None
        meta.features = features
        return meta

    def test_single_frame_no_override(self):
        """F==1: tactile uses the shared observation_delta_indices (or None)."""
        from vtla.datasets.factory import resolve_delta_timestamps

        cfg = MagicMock()
        cfg.tactile_windowed.return_value = False
        cfg.observation_delta_indices = None
        cfg.action_delta_indices = None
        cfg.reward_delta_indices = None

        ds_meta = self._make_ds_meta(fps=30, tactile_keys=["observation.images.finger0"])
        result = resolve_delta_timestamps(cfg, ds_meta)
        assert result is None  # no windowing

    def test_multi_frame_override_tactile_keys(self):
        """F>1: tactile keys use their own delta, other obs keys use shared one."""
        from vtla.datasets.factory import resolve_delta_timestamps

        tac_key = "observation.images.finger0"
        obs_key = "observation.images.cam_top"

        cfg = MagicMock()
        cfg.tactile_windowed.return_value = True
        cfg.tactile_windowed_keys.return_value = [tac_key]
        cfg.tactile_delta_indices.return_value = [-2, -1, 0]   # F=3, off=1
        cfg.observation_delta_indices = [-1, 0]                  # shared for RGB
        cfg.action_delta_indices = None
        cfg.reward_delta_indices = None

        ds_meta = self._make_ds_meta(fps=30, tactile_keys=[tac_key])
        # add extra obs key
        ds_meta.features["observation.images.cam_top"] = None
        ds_meta.features["observation.state"] = None

        result = resolve_delta_timestamps(cfg, ds_meta)
        assert result is not None

        fps = 30
        expected_tac = [-2 / fps, -1 / fps, 0.0]
        expected_obs = [-1 / fps, 0.0]

        assert pytest.approx(result[tac_key]) == expected_tac
        assert pytest.approx(result[obs_key]) == expected_obs
        assert pytest.approx(result["observation.state"]) == expected_obs


# ---------------------------------------------------------------------------
# 5. TactileTemporalWindowStep
# ---------------------------------------------------------------------------
from vtla.frameworks.tactile_temporal_processor import TactileTemporalWindowStep


class TestTactileTemporalWindowStep:
    TAC_KEY = "observation.images.finger0"

    def _step(self, num_frames=3, frame_offset=1):
        return TactileTemporalWindowStep(
            tactile_keys=[self.TAC_KEY],
            num_frames=num_frames,
            frame_offset=frame_offset,
        )

    def _frame(self, val=0.0):
        """Single batch-dim frame: [1, C, H, W]."""
        return torch.full((1, 3, 8, 8), val)

    def test_passthrough_when_already_windowed_5d(self):
        """Training path: 5D tensor passes through unchanged."""
        step = self._step(num_frames=3, frame_offset=1)
        t5d = torch.zeros(2, 3, 3, 8, 8)
        batch = {self.TAC_KEY: t5d}
        out = step(batch)
        assert out[self.TAC_KEY] is t5d  # unchanged reference

    def test_passthrough_when_single_frame_mode(self):
        """F=1: step is a no-op."""
        step = self._step(num_frames=1)
        t = self._frame()
        batch = {self.TAC_KEY: t}
        out = step(batch)
        assert out[self.TAC_KEY] is t

    def test_pads_with_first_frame_on_init(self):
        """First observation: deque is filled with repeated first frame."""
        step = self._step(num_frames=3, frame_offset=1)
        batch = {self.TAC_KEY: self._frame(val=1.0)}
        out = step(batch)
        t = out[self.TAC_KEY]  # [1, 3, C, H, W]
        assert t.shape == (1, 3, 3, 8, 8)
        # All three frames should be the first frame (padding)
        assert t[:, 0].allclose(self._frame(val=1.0))
        assert t[:, 1].allclose(self._frame(val=1.0))
        assert t[:, 2].allclose(self._frame(val=1.0))

    def test_window_oldest_to_newest(self):
        """After several steps, window contains oldest → newest in order."""
        step = self._step(num_frames=3, frame_offset=1)
        for i in range(5):
            batch = {self.TAC_KEY: self._frame(val=float(i))}
            out = step(batch)
        t = out[self.TAC_KEY]  # step 4 is most recent (val=4)
        assert t.shape == (1, 3, 3, 8, 8)
        assert t[:, -1, 0, 0, 0].item() == pytest.approx(4.0)  # newest
        assert t[:, -2, 0, 0, 0].item() == pytest.approx(3.0)
        assert t[:, -3, 0, 0, 0].item() == pytest.approx(2.0)

    def test_frame_offset(self):
        """With offset=2, window skips frames."""
        step = self._step(num_frames=3, frame_offset=2)
        for i in range(7):
            batch = {self.TAC_KEY: self._frame(val=float(i))}
            out = step(batch)
        t = out[self.TAC_KEY]
        # At step 6 (val=6), offset=2, F=3: should be [2, 4, 6]
        assert t[:, 0, 0, 0, 0].item() == pytest.approx(2.0)
        assert t[:, 1, 0, 0, 0].item() == pytest.approx(4.0)
        assert t[:, 2, 0, 0, 0].item() == pytest.approx(6.0)

    def test_reset_clears_history(self):
        """reset() flushes frame history."""
        step = self._step(num_frames=3, frame_offset=1)
        for i in range(10):
            step({self.TAC_KEY: self._frame(val=float(i))})
        step.reset()
        out = step({self.TAC_KEY: self._frame(val=99.0)})
        t = out[self.TAC_KEY]
        # After reset, all frames should be the new first frame (99)
        assert t[:, 0, 0, 0, 0].item() == pytest.approx(99.0)
        assert t[:, 1, 0, 0, 0].item() == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# 6. expand_tactile_as_image_window
# ---------------------------------------------------------------------------
from vtla.frameworks.utils import expand_tactile_as_image_window


class TestExpandTactileAsImage:
    TAC_KEY = "observation.images.finger0"

    def _cfg_as_image(self, num_frames):
        cfg = MagicMock()
        cfg.tactile_mode = "as_image"
        cfg.tactile_num_frames = num_frames
        cfg.tactile_windowed_keys.return_value = [self.TAC_KEY]
        return cfg

    def test_noop_single_frame(self):
        cfg = self._cfg_as_image(num_frames=1)
        t = torch.zeros(2, 3, 8, 8)
        batch = {self.TAC_KEY: t, "observation.images.cam_top": torch.zeros(2, 3, 8, 8)}
        out = expand_tactile_as_image_window(batch, cfg)
        assert self.TAC_KEY in out
        assert out[self.TAC_KEY] is t  # unchanged

    def test_noop_encode_mode(self):
        cfg = MagicMock()
        cfg.tactile_mode = "encode"
        cfg.tactile_num_frames = 3
        t = torch.zeros(2, 3, 3, 8, 8)
        batch = {self.TAC_KEY: t}
        out = expand_tactile_as_image_window(batch, cfg)
        assert self.TAC_KEY in out  # still present, not split

    def test_splits_frames_correctly(self):
        F = 3
        cfg = self._cfg_as_image(num_frames=F)
        # [B, F, C, H, W] with value = frame index
        t = torch.stack([torch.full((2, 3, 8, 8), float(i)) for i in range(F)], dim=1)
        batch = {self.TAC_KEY: t}
        out = expand_tactile_as_image_window(batch, cfg)
        assert self.TAC_KEY not in out  # original key removed
        for i in range(F):
            fk = f"{self.TAC_KEY}.f{i}"
            assert fk in out
            assert out[fk].shape == (2, 3, 8, 8)
            assert out[fk][0, 0, 0, 0].item() == pytest.approx(float(i))


# ---------------------------------------------------------------------------
# 7. image_feature_keys_expanded
# ---------------------------------------------------------------------------
class TestImageFeatureKeysExpanded:
    def _cfg(self, num_frames, mode="as_image"):
        tac_keys = ["observation.images.finger0", "observation.images.finger1"]
        cfg = MagicMock(spec=SensorRoutingMixin)
        cfg.tactile_mode = mode
        cfg.tactile_num_frames = num_frames
        cfg.tactile_windowed_keys.return_value = tac_keys
        cfg.image_features = {
            "observation.images.cam_top": None,
            "observation.images.finger0": None,
            "observation.images.finger1": None,
        }
        # Call the real method
        cfg.image_feature_keys_expanded = lambda: SensorRoutingMixin.image_feature_keys_expanded(cfg)
        return cfg

    def test_noop_single_frame(self):
        cfg = self._cfg(num_frames=1)
        keys = cfg.image_feature_keys_expanded()
        assert keys == list(cfg.image_features.keys())

    def test_expands_tactile_keys_multi_frame(self):
        cfg = self._cfg(num_frames=3)
        keys = cfg.image_feature_keys_expanded()
        # cam_top stays, each finger expands to 3 frame keys
        assert "observation.images.cam_top" in keys
        for i in range(3):
            assert f"observation.images.finger0.f{i}" in keys
            assert f"observation.images.finger1.f{i}" in keys
        # Original tactile keys should NOT appear
        assert "observation.images.finger0" not in keys
        assert "observation.images.finger1" not in keys
        # Total: 1 (cam_top) + 3 (finger0) + 3 (finger1) = 7
        assert len(keys) == 7

    def test_noop_when_mode_is_encode(self):
        """encode mode never does as_image expansion."""
        cfg = self._cfg(num_frames=3, mode="encode")
        keys = cfg.image_feature_keys_expanded()
        assert keys == list(cfg.image_features.keys())


# ---------------------------------------------------------------------------
# 8. Diffusion config rejects as_image + frames>1
# ---------------------------------------------------------------------------
class TestDiffusionConfigValidation:
    def test_rejects_as_image_multi_frame(self):
        from vtla.frameworks.diffusion.configuration_diffusion import DiffusionConfig
        import pytest

        with pytest.raises(ValueError, match="as_image.*tactile_num_frames"):
            cfg = MagicMock(spec=DiffusionConfig)
            cfg.tactile_mode = "as_image"
            cfg.tactile_num_frames = 3
            # Call the __post_init__ method that has the guard
            DiffusionConfig.__post_init__.__wrapped__ = getattr(
                DiffusionConfig.__post_init__, "__wrapped__", DiffusionConfig.__post_init__
            )
