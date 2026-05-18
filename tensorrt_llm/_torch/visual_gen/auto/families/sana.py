# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sana family adapter (Efficient-Large-Model/Sana_*).

Targets Diffusers `SanaTransformer2DModel`. Architecture (per the 1.6B
config):
- `in_channels=32`, `patch_size=1` (no spatial patchify — operates directly
  on VAE latents at 32× compression).
- `num_attention_heads=70`, `attention_head_dim=32` → `inner_dim=2240`.
- Separate self-attention (`num_attention_heads=70`) and cross-attention
  (`num_cross_attention_heads=20`, `cross_attention_head_dim=112`).
- `caption_channels=2304` (Gemma-2-2B text encoder output).
- 4-D `hidden_states (B, C, H, W)` input.
- Uses **linear attention** in self-attention (not standard quadratic SDPA).
  This is the Sana paper's headline trick — auto path's `visgen_auto.sdpa`
  pattern matcher only catches `F.scaled_dot_product_attention`, so the
  linear-attn blocks remain in eager. Cross-attention is regular SDPA and
  gets rewritten normally.
- No QK-RMSNorm, no RoPE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..adapter import VisGenFamilyAdapter
from ..policy import RewritePolicy

if TYPE_CHECKING:
    from ...config import DiffusionModelConfig


_EX_B = 2  # CFG-doubled batch
_DEFAULT_LATENT_H = 32  # 1024 / 32 (VAE compression) = 32
_DEFAULT_LATENT_W = 32
_EX_TXT_SEQ = 300  # Sana uses Gemma-2 with similar default text seq


def _arch_dim(cfg: "DiffusionModelConfig", attr: str, default: int) -> int:
    pc = getattr(cfg, "pretrained_config", None)
    return int(getattr(pc, attr, default)) if pc is not None else default


class SanaAdapter(VisGenFamilyAdapter):
    family = "Sana"
    diffusers_transformer_cls = "SanaTransformer2DModel"

    def example_inputs(
        self,
        cfg: "DiffusionModelConfig",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple[tuple, dict[str, Any]]:
        in_channels = _arch_dim(cfg, "in_channels", 32)
        caption_channels = _arch_dim(cfg, "caption_channels", 2304)

        B = _EX_B
        H = _DEFAULT_LATENT_H
        W = _DEFAULT_LATENT_W
        S_txt = _EX_TXT_SEQ

        # SanaPipeline calls the transformer with `hidden_states` POSITIONAL.
        hidden_states = torch.randn(B, in_channels, H, W, device=device, dtype=dtype)
        kwargs: dict[str, Any] = {
            "encoder_hidden_states": torch.randn(
                B, S_txt, caption_channels, device=device, dtype=dtype
            ),
            "timestep": torch.full((B,), 500, device=device, dtype=torch.long),
            "encoder_attention_mask": torch.ones(B, S_txt, device=device, dtype=torch.long),
            "return_dict": False,
        }
        return (hidden_states,), kwargs

    def dynamic_shapes(self, cfg: "DiffusionModelConfig") -> dict[str, Any]:
        # All-static for first-light (same rationale as PixArt). Sana's
        # linear-attention blocks have internal reshapes that don't admit
        # cleanly to Dim() constraints; specialize at example shape.
        return {
            "hidden_states": None,
            "encoder_hidden_states": None,
            "timestep": None,
            "encoder_attention_mask": None,
            "return_dict": None,
        }

    def rewrite_policy(self, cfg: "DiffusionModelConfig") -> RewritePolicy:
        # No QK-RMSNorm, no RoPE → fuse_qk_rope=False.
        #
        # NOTE: `attention.backend` (`VANILLA` / `TRTLLM` / `FA4`) is a no-op
        # for Sana self-attention sites — Sana's linear attention blocks don't
        # call `F.scaled_dot_product_attention`, so the SDPA rewriter's
        # pattern matcher skips them entirely. The backend still applies to
        # cross-attention sites (which use quadratic SDPA), but the
        # headline perf characteristic of Sana is the linear self-attn,
        # which stays in eager. Document this asymmetry vs. handwritten Sana.
        return RewritePolicy(
            attention_backend=cfg.attention.backend,
            fuse_qk_rope=False,
        )
