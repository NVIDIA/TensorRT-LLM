# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flux family adapter for the auto path.

Targets Diffusers `FluxTransformer2DModel` (Flux.1).
`torch.export.export(strict=False)` succeeds on this transformer with
dynamic batch / image_seq / txt_seq dims; the dim choices and example
shapes below were validated against full Flux.1-dev (19 dual-stream +
38 single-stream blocks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..adapter import VisGenFamilyAdapter
from ..policy import RewritePolicy

if TYPE_CHECKING:
    from ...config import DiffusionModelConfig

# Tiny example dims for export tracing. These are unrelated to runtime — the
# captured graph uses dynamic dims, so these only serve to seed shape-symbol
# resolution. Picked to be > 1 so `torch.export` does not specialize the
# batch dim to 1.
_EXAMPLE_BATCH = 2
_EXAMPLE_IMG_SEQ = 64
_EXAMPLE_TXT_SEQ = 16

# Dynamic-shape range for Flux1 image/text sequences. Bounds are loose; the
# rewriter must not depend on them for correctness.
_DIM_BATCH_MAX = 8
_DIM_IMG_SEQ_MAX = 16384
_DIM_TXT_SEQ_MAX = 1024


def _arch_dim(cfg: "DiffusionModelConfig", attr: str, default: int) -> int:
    """Read an architectural dim from `cfg.pretrained_config`, with a fallback.

    `pretrained_config` is a `SimpleNamespace` populated from the transformer's
    `config.json` (see `DiffusionModelConfig.from_pretrained`), so attributes
    correspond to the keys diffusers writes there.
    """
    pc = getattr(cfg, "pretrained_config", None)
    return int(getattr(pc, attr, default)) if pc is not None else default


class FluxAdapter(VisGenFamilyAdapter):
    family = "Flux"
    diffusers_transformer_cls = "FluxTransformer2DModel"

    def example_inputs(
        self,
        cfg: "DiffusionModelConfig",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple[tuple, dict[str, Any]]:
        in_channels = _arch_dim(cfg, "in_channels", 64)
        joint_dim = _arch_dim(cfg, "joint_attention_dim", 4096)
        pooled_dim = _arch_dim(cfg, "pooled_projection_dim", 768)
        # FLUX.1-dev sets `guidance_embeds=True` → the pipeline passes a
        # `guidance` tensor at runtime. FLUX.1-schnell omits it.
        pc = getattr(cfg, "pretrained_config", None)
        guidance_embeds = bool(getattr(pc, "guidance_embeds", False))

        B = _EXAMPLE_BATCH
        S_img = _EXAMPLE_IMG_SEQ
        S_txt = _EXAMPLE_TXT_SEQ

        kwargs: dict[str, Any] = {
            "hidden_states": torch.randn(B, S_img, in_channels, device=device, dtype=dtype),
            "encoder_hidden_states": torch.randn(B, S_txt, joint_dim, device=device, dtype=dtype),
            "pooled_projections": torch.randn(B, pooled_dim, device=device, dtype=dtype),
            # Diffusers' FluxPipeline passes `timestep / 1000` with dtype matching
            # `latents.dtype` (BF16 in our pipeline runs). Match that here so the
            # captured graph's Long→BF16 cast doesn't drift the BF16→BF16 runtime path.
            "timestep": torch.full((B,), 0.5, device=device, dtype=dtype),
            "img_ids": torch.zeros(S_img, 3, device=device, dtype=dtype),
            "txt_ids": torch.zeros(S_txt, 3, device=device, dtype=dtype),
            "return_dict": False,
        }
        if guidance_embeds:
            # FLUX.1-dev passes guidance as float32 (Diffusers' FluxPipeline
            # uses torch.full((B,), guidance_scale, dtype=torch.float32)).
            kwargs["guidance"] = torch.full((B,), 3.5, device=device, dtype=torch.float32)
        return (), kwargs

    def dynamic_shapes(self, cfg: "DiffusionModelConfig") -> dict[str, Any]:
        pc = getattr(cfg, "pretrained_config", None)
        guidance_embeds = bool(getattr(pc, "guidance_embeds", False))

        B = torch.export.Dim("B", min=1, max=_DIM_BATCH_MAX)
        S_img = torch.export.Dim("S_img", min=16, max=_DIM_IMG_SEQ_MAX)
        S_txt = torch.export.Dim("S_txt", min=1, max=_DIM_TXT_SEQ_MAX)
        spec: dict[str, Any] = {
            "hidden_states": {0: B, 1: S_img},
            "encoder_hidden_states": {0: B, 1: S_txt},
            "pooled_projections": {0: B},
            "timestep": {0: B},
            "img_ids": {0: S_img},
            "txt_ids": {0: S_txt},
            "return_dict": None,
        }
        if guidance_embeds:
            spec["guidance"] = {0: B}
        return spec

    def rewrite_policy(self, cfg: "DiffusionModelConfig") -> RewritePolicy:
        return RewritePolicy(attention_backend=cfg.attention.backend)
