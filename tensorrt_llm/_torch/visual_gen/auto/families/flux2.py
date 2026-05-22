# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flux2 family adapter for the auto path.

Targets Diffusers `Flux2Transformer2DModel` (FLUX.2-dev/-pro). Differences vs
Flux1:
- No `pooled_projections` — Flux2 uses `time_guidance_embed(timestep, guidance)`
  for modulation, not the combined timestep+pooled embedding.
- `axes_dims_rope` is 4-axis (Flux1: 3-axis).
- Supports optional KV-cache reference-image flow (`kv_cache_mode`,
  `num_ref_tokens`). This adapter targets the *no-ref-tokens* path
  (`kv_cache_mode=None`); ref-token support is a follow-up.
- `joint_attention_dim` is much larger (15360 in FLUX.2-dev — T5 + Mistral).
- 8 double-stream + 48 single-stream blocks (vs Flux1: 19 + 38).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..adapter import VisGenFamilyAdapter
from ..policy import RewritePolicy

if TYPE_CHECKING:
    from ...config import DiffusionModelConfig

# Tiny example dims for export tracing.
_EXAMPLE_BATCH = 2
_EXAMPLE_IMG_SEQ = 64
_EXAMPLE_TXT_SEQ = 16

_DIM_BATCH_MAX = 8
_DIM_IMG_SEQ_MAX = 16384
_DIM_TXT_SEQ_MAX = 1024


def _arch_dim(cfg: "DiffusionModelConfig", attr: str, default: int) -> int:
    pc = getattr(cfg, "pretrained_config", None)
    return int(getattr(pc, attr, default)) if pc is not None else default


class Flux2Adapter(VisGenFamilyAdapter):
    family = "Flux2"
    diffusers_transformer_cls = "Flux2Transformer2DModel"

    def example_inputs(
        self,
        cfg: "DiffusionModelConfig",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple[tuple, dict[str, Any]]:
        in_channels = _arch_dim(cfg, "in_channels", 128)
        joint_dim = _arch_dim(cfg, "joint_attention_dim", 15360)
        # `guidance_embeds=True` for FLUX.2-dev (CFG-distilled, takes guidance scalar);
        # `guidance_embeds=False` for FLUX.2-klein (fully distilled, pipeline passes
        # `guidance=None`). Match the pipeline's call shape at capture time.
        pc = getattr(cfg, "pretrained_config", None)
        guidance_embeds = bool(getattr(pc, "guidance_embeds", True)) if pc is not None else True

        B = _EXAMPLE_BATCH
        S_img = _EXAMPLE_IMG_SEQ
        S_txt = _EXAMPLE_TXT_SEQ

        # KV-cache / ref-token kwargs (kv_cache, kv_cache_mode, num_ref_tokens,
        # ref_fixed_timestep) are NOT included — they have safe defaults
        # (None / 0 / 0.0) and the Diffusers pipeline doesn't pass them.
        # Including them in `example_inputs` would force runtime to pass them
        # too (torch.export's kwarg-set is strict). Ref-token support is a
        # follow-up adapter mode.
        kwargs: dict[str, Any] = {
            "hidden_states": torch.randn(B, S_img, in_channels, device=device, dtype=dtype),
            "encoder_hidden_states": torch.randn(B, S_txt, joint_dim, device=device, dtype=dtype),
            "timestep": torch.full((B,), 0.5, device=device, dtype=dtype),
            # 4-axis RoPE ids (vs Flux1's 3-axis), passed as 3D `(B, S, 4)` by
            # Diffusers' Flux2Pipeline (Flux1 used 2D).
            "img_ids": torch.zeros(B, S_img, 4, device=device, dtype=dtype),
            "txt_ids": torch.zeros(B, S_txt, 4, device=device, dtype=dtype),
            "guidance": (
                torch.full((B,), 3.5, device=device, dtype=torch.float32)
                if guidance_embeds
                else None
            ),
            "return_dict": False,
        }
        return (), kwargs

    def dynamic_shapes(self, cfg: "DiffusionModelConfig") -> dict[str, Any]:
        pc = getattr(cfg, "pretrained_config", None)
        guidance_embeds = bool(getattr(pc, "guidance_embeds", True)) if pc is not None else True
        B = torch.export.Dim("B", min=1, max=_DIM_BATCH_MAX)
        S_img = torch.export.Dim("S_img", min=16, max=_DIM_IMG_SEQ_MAX)
        S_txt = torch.export.Dim("S_txt", min=1, max=_DIM_TXT_SEQ_MAX)
        return {
            "hidden_states": {0: B, 1: S_img},
            "encoder_hidden_states": {0: B, 1: S_txt},
            "timestep": {0: B},
            "img_ids": {0: B, 1: S_img},
            "txt_ids": {0: B, 1: S_txt},
            "guidance": {0: B} if guidance_embeds else None,
            "return_dict": None,
        }

    def rewrite_policy(self, cfg: "DiffusionModelConfig") -> RewritePolicy:
        return RewritePolicy(attention_backend=cfg.attention.backend)

    # No `default_quant_exclude_modules` override: the structural heuristic in
    # `auto/sensitivity.py` auto-detects `time_guidance_embed.*` from the
    # module tree (its `timestep_embedder.linear_1` has `in_features=256`).
    # Override here only if a family has Linears the heuristic misses.
