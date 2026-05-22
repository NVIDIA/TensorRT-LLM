# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-detect quantization-sensitive Linears via module-tree structure.

The goal: replace hand-curated per-family exclude lists (e.g. the
`["*time_guidance_embed*"]` we had on `Flux2Adapter`) with a structural
heuristic that applies to any Diffusers transformer.

Heuristic ("narrow-dim embedder, either direction"):

- Skip per-step block stacks (children that are `nn.ModuleList`) — Linears
  in those run once per token site and have block-local fan-out, so their
  per-tensor FP8 error doesn't propagate model-wide.
- For each remaining top-level child module, scan its descendant `nn.Linear`
  instances. If *any* of them has a narrow input dim (`in_features <=
  SMALL_DIM_THRESHOLD`, i.e., it's a conditioning embedder reading a low-dim
  scalar/code) *or* a narrow output dim (`out_features <= SMALL_DIM_THRESHOLD`,
  i.e., it projects to a small latent channel count), treat the whole
  subtree as quant-sensitive and emit a `*<name>*` glob pattern that
  excludes all its Linears.

Why this works on DiT-family models:

- Timestep / guidance / label embedders project a 1-D or low-dim scalar
  (256 sinusoidal channels, 128 patch latents, 768 pooled-text features)
  into the model hidden dim. The first Linear in that MLP always has small
  `in_features`. Its output (`temb`) feeds every modulation linear in every
  block, so per-tensor FP8 rounding error there compounds system-wide
  (LPIPS 0.10 vs 0.04 on FLUX.2 when the modulation MLP's first Linear is
  NOT excluded).
- Final output projections (`proj_out`, `audio_proj_out`) compress the full
  inner_dim *down* to the tiny VAE latent channel count (out_features=128
  for LTX-2, ~64 for FLUX/SD3 patches). Weight-side rounding there feeds
  directly into the latent the VAE/vocoder upsamples, amplifying drift
  into pixels/waveform (LTX-2 NVFP4 LPIPS 0.066 → 0.050 once excluded).
- Per-block attention/MLP Linears (`to_q`, `to_k`, etc.) all read the
  hidden-dim activation (6144 for FLUX.2) and write back to the same
  hidden dim, which is well above the threshold, so they're not falsely
  excluded.

This catches the canonical sensitive layers without any family-specific
configuration:

- FLUX.1: `time_text_embed.timestep_embedder.linear_1` (256→3072) triggers
  in-side exclusion of the whole `time_text_embed.*` subtree
- FLUX.2: `time_guidance_embed.timestep_embedder.linear_1` (256→6144) triggers
  in-side exclusion of the whole `time_guidance_embed.*` subtree
- SD3:    same pattern under `time_text_embed.*`
- LTX-2:  `proj_out` (4096→128) + `audio_proj_out` (2048→128) trigger
  out-side exclusion — replaces the need for an LTX2Adapter
  `default_quant_exclude_modules` override

Users can still pass their own `quant_config.exclude_modules` to add layers;
the auto-detected patterns are unioned in.
"""

from __future__ import annotations

import os

import torch.nn as nn

# Conservative default. The smallest activation a DiT block sees is the
# hidden dim (typically 1024+); anything ≤512 in_features or ≤512 out_features
# almost certainly comes from a low-dim conditioning projection or a
# latent-channel compression.
SMALL_DIM_THRESHOLD = 512

# Back-compat alias — older call sites pass `max_embed_in_dim=`.
SMALL_INPUT_DIM_THRESHOLD = SMALL_DIM_THRESHOLD


def _all_linear_descendants(module: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Return every `nn.Linear` reachable under `module`, with relative names."""
    return [(n, m) for n, m in module.named_modules() if isinstance(m, nn.Linear)]


def find_quantization_sensitive_linears(
    root: nn.Module,
    max_embed_in_dim: int = SMALL_DIM_THRESHOLD,
    max_embed_out_dim: int | None = None,
) -> list[str]:
    """Return glob patterns covering Linears that produce conditioning signals
    or compress to latent channels.

    Args:
        root: The transformer module to analyse.
        max_embed_in_dim: A Linear with `in_features <= this` triggers
            "conditioning embedder" detection on its containing subtree.
        max_embed_out_dim: A Linear with `out_features <= this` triggers
            "latent compression" detection on its containing subtree.

    Returns:
        Glob patterns of the form ``["*<top_level_child_name>*", ...]`` suitable
        for `QuantConfig.exclude_modules`. Empty list if nothing triggers.

    Env override: setting `VISGEN_SENSITIVITY_OUT_DIM` overrides the
    `max_embed_out_dim` default. Set to `0` to disable the out-side rule
    entirely (useful for regression-testing against the old in-only
    behavior).
    """
    if max_embed_out_dim is None:
        env = os.environ.get("VISGEN_SENSITIVITY_OUT_DIM")
        max_embed_out_dim = int(env) if env is not None else SMALL_DIM_THRESHOLD
    patterns: list[str] = []
    for name, child in root.named_children():
        if isinstance(child, nn.ModuleList):
            continue
        child_linears = _all_linear_descendants(child)
        if not child_linears:
            continue
        if any(
            lin.in_features <= max_embed_in_dim or lin.out_features <= max_embed_out_dim
            for _, lin in child_linears
        ):
            patterns.append(f"*{name}*")
    return patterns
