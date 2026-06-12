"""Tactile-MAE feature extractor with learnable query tokens for downstream policies.

This is the entry point that policy frameworks use when training with
``--policy.tactile_mode=encode``. It wraps the existing ``build_model`` /
``load_pretrained`` from ``tactile_mae`` and adds ``N`` learnable *query tokens* that
are injected into the encoder input sequence right after the prefix tokens::

    [cls, 5x sensor, N x query, 256 x patch]

The query tokens self-attend with every other token at every transformer layer (the
CLIP vision encoder is fully bidirectional, no attention mask), and only the ``N``
query outputs are returned as the tactile representation. The whole module (MAE
encoder + query tokens) is normally fine-tuned end-to-end during policy training.

The encoder configuration (``arch`` / ``sensor_id`` / ``image_size`` / ...) is read
automatically from the checkpoint so the user only needs to pass a weight path:

    extractor = TactileMAEFeatureExtractor.from_pretrained("path/to/best.pth", num_query_tokens=8)
    feats = extractor(images)        # [B, C, H, W]    -> [B, N, D]
    feats = extractor(image_seq)     # [B, T, C, H, W] -> [B, T, N, D]

``freeze=True`` keeps the MAE backbone frozen (only the query tokens train);
``freeze=False`` (the default for policy training) fine-tunes everything.
"""
import os

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .data import IMAGENET_MEAN, IMAGENET_STD
from .models import build_model, load_pretrained
from .models.build import _read_state_dict

DEFAULT_IMAGE_SIZE = 224


def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported tactile-MAE dtype: {dtype}")


def _load_checkpoint_args(path: str) -> dict | None:
    """Return the ``args`` namespace stored in our MAE checkpoints as a dict.

    Our ``misc.save_model`` / ``save_best_model`` store the training ``argparse``
    Namespace under the ``args`` key. Older AnyTouch weights / HF dirs do not have
    it, in which case ``None`` is returned and the config is inferred from shapes.
    """
    if not path or os.path.isdir(path) or not path.endswith((".pth", ".pt", ".bin")):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(ckpt, dict) or "args" not in ckpt:
        return None
    args = ckpt["args"]
    return dict(vars(args)) if hasattr(args, "__dict__") else (args if isinstance(args, dict) else None)


def _infer_arch_from_state_dict(sd: dict) -> str:
    """Infer ``vit_l`` / ``vit_b`` from encoder patch-embedding weight shape.

    hidden 1024 / patch 14 -> vit_l ; hidden 768 / patch 16 -> vit_b.
    """
    for key in (
        "touch_model.embeddings.patch_embedding.weight",
        "video_patch_embedding.weight",
    ):
        if key in sd:
            w = sd[key]
            hidden = w.shape[0]
            return "vit_l" if hidden >= 1024 else "vit_b"
    # Fall back to the project default.
    return "vit_l"


class TactileMAEFeatureExtractor(nn.Module):
    """Frozen tactile-MAE encoder that maps tactile images to CLS features."""

    def __init__(
        self,
        model: nn.Module,
        sensor_id: int = -1,
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze: bool = False,
        num_query_tokens: int = 8,
        dtype: str | torch.dtype | None = "bfloat16",
    ):
        super().__init__()
        self.compute_dtype = _resolve_dtype(dtype)
        if self.compute_dtype is not None:
            model.to(dtype=self.compute_dtype)

        self.model = model
        self.sensor_id = int(sensor_id)
        self.image_size = int(image_size)
        self._freeze = freeze
        self.num_query_tokens = int(num_query_tokens)

        # CLS feature dim == encoder hidden size (post_layernorm output).
        self._feature_dim = int(model.touch_model.post_layernorm.weight.shape[0])

        # Learnable query tokens, warm-started from the pretrained CLS embedding (a
        # global summary token) plus small noise to break symmetry between queries.
        cls_embed = model.touch_model.embeddings.class_embedding.detach()  # [D]
        query_init = cls_embed.unsqueeze(0).repeat(self.num_query_tokens, 1)
        query_init = query_init + 0.02 * torch.randn_like(query_init)
        self.query_tokens = nn.Parameter(query_init)  # [N, D], always trainable

        # ImageNet normalization buffers (match tactile-MAE training transforms).
        self.register_buffer(
            "_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls, path: str, freeze: bool = False, num_query_tokens: int = 8,
        dtype: str | torch.dtype | None = "bfloat16",
    ) -> "TactileMAEFeatureExtractor":
        """Build the encoder from any supported checkpoint / HF dir.

        ``arch`` / ``use_sensor_token`` / ``use_same_patchemb`` / ``sensor_id`` /
        ``image_size`` are read from the checkpoint ``args`` when present, otherwise
        inferred from the weights (with sensible AnyTouch defaults).
        """
        if not path:
            raise ValueError("TactileMAEFeatureExtractor.from_pretrained requires a checkpoint path.")

        ckpt_args = _load_checkpoint_args(path)
        sd = _read_state_dict(path)

        if ckpt_args is not None:
            arch = ckpt_args.get("arch", _infer_arch_from_state_dict(sd))
            use_sensor_token = bool(ckpt_args.get("use_sensor_token", True))
            use_same_patchemb = bool(ckpt_args.get("use_same_patchemb", True))
            sensor_id = int(ckpt_args.get("sensor_id", -1))
            image_size = int(ckpt_args.get("image_size", DEFAULT_IMAGE_SIZE))
        else:
            # Old AnyTouch weights / HF dir: infer arch, use defaults for the rest.
            arch = _infer_arch_from_state_dict(sd)
            use_sensor_token = "sensor_token" in sd or any(
                k.endswith("sensor_token") for k in sd
            )
            use_same_patchemb = True
            sensor_id = -1
            image_size = DEFAULT_IMAGE_SIZE

        model = build_model(
            arch=arch,
            mask_ratio=0.0,
            use_sensor_token=use_sensor_token,
            use_same_patchemb=use_same_patchemb,
        )
        load_pretrained(model, path, verbose=True)

        return cls(
            model, sensor_id=sensor_id, image_size=image_size,
            freeze=freeze, num_query_tokens=num_query_tokens, dtype=dtype,
        )

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def train(self, mode: bool = True):
        """Keep the encoder in eval mode when frozen (BN / dropout stability)."""
        super().train(mode)
        if self._freeze:
            self.model.eval()
        return self

    def _normalize(self, images: Tensor) -> Tensor:
        """Resize to ``image_size`` and apply ImageNet normalization.

        ``images``: ``[N, 3, H, W]`` float in [0, 1].
        """
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        mean = self._mean.to(device=images.device, dtype=images.dtype)
        std = self._std.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std

    def forward(self, images: Tensor) -> Tensor:
        """Encode tactile images into query-token features.

        Accepts ``[B, C, H, W]`` -> ``[B, N, D]`` or ``[B, T, C, H, W]`` -> ``[B, T, N, D]``,
        where ``N`` is ``num_query_tokens``.
        """
        if images.dim() == 4:
            squeeze_time = False
            b = images.shape[0]
            flat = images
        elif images.dim() == 5:
            squeeze_time = True
            b, t = images.shape[:2]
            flat = images.flatten(0, 1)  # [B*T, C, H, W]
        else:
            raise ValueError(
                f"TactileMAEFeatureExtractor expects 4D or 5D image tensors, got shape {tuple(images.shape)}"
            )

        flat = flat.float()
        flat = self._normalize(flat)
        if self.compute_dtype is not None:
            flat = flat.to(dtype=self.compute_dtype)

        n = flat.shape[0]
        sensor_type = torch.full((n,), self.sensor_id, dtype=torch.long, device=flat.device)

        # No no_grad() wrapper: even when ``freeze=True`` the MAE backbone params have
        # requires_grad=False (so they are not updated), but the autograd graph must
        # still be built so gradients reach the always-trainable query tokens + the
        # downstream projection. requires_grad flags alone decide what gets updated.
        feats = self._encode_query_tokens(flat, sensor_type)  # [N, Nq, D]

        if squeeze_time:
            feats = feats.reshape(b, t, self.num_query_tokens, self._feature_dim)
        return feats

    def _encode_query_tokens(self, x: Tensor, sensor_type: Tensor) -> Tensor:
        """Inject the learnable query tokens and return their encoder outputs.

        Sequence fed to the (fully bidirectional) CLIP encoder::

            [cls, 5x sensor, N x query, patches]

        ``mask_ratio`` is forced to 0 so every patch is present; the query outputs see
        all tokens. Returns ``[N, Nq, D]``.
        """
        m = self.model
        prev_ratio = m.mask_ratio
        m.mask_ratio = 0.0
        try:
            # embed() returns [cls, sensor, patches] with pos-embeds already applied.
            embeddings, _, _ = m.embed(x, sensor_type, noise=None)
            p = m.n_prefix
            prefix, patches = embeddings[:, :p, :], embeddings[:, p:, :]
            queries = self.query_tokens.to(embeddings.dtype)
            queries = queries.unsqueeze(0).expand(x.shape[0], -1, -1)  # [B, Nq, D]
            seq = torch.cat([prefix, queries, patches], dim=1)

            hidden = m.touch_model.pre_layrnorm(seq)
            enc_out = m.touch_model.encoder(inputs_embeds=hidden)
            query_out = enc_out.last_hidden_state[:, p : p + self.num_query_tokens, :]
            out = m.touch_model.post_layernorm(query_out)
        finally:
            m.mask_ratio = prev_ratio
        return out

    def _extract_cls(self, x: Tensor, sensor_type: Tensor) -> Tensor:
        """CLS feature extraction (grad-aware variant of ``model.extract_features``).

        Retained for ablation / debugging. ``model.extract_features`` is wrapped in
        ``@torch.no_grad()``; replicating its body here lets gradients flow.
        """
        m = self.model
        prev_ratio = m.mask_ratio
        m.mask_ratio = 0.0
        try:
            embeddings, _, _ = m.embed(x, sensor_type, noise=None)
            hidden = m.touch_model.pre_layrnorm(embeddings)
            enc_out = m.touch_model.encoder(inputs_embeds=hidden)
            cls = enc_out.last_hidden_state[:, 0, :]
            cls = m.touch_model.post_layernorm(cls)
        finally:
            m.mask_ratio = prev_ratio
        return cls
