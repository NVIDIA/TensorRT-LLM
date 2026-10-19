# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped gated RMS normalization with the gate applied before the norm --
Mamba2's `norm_before_gate=False` -- on the mamba_ssm single-pass Triton kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.mamba.layernorm_gated import _layer_norm_fwd

#: Per rank at tep4: d_inner 16384 / 4 = 4096 channels, n_groups 8 / 4 = two
#: whole groups of 2048.
WIDTH = 4096
GROUP_SIZE = 2048
#: The gate z is the first 4096 columns of the rank's in_proj output row,
#: laid out z 4096 | xBC 4608 | dt 64.
PROJ_WIDTH = 8768
#: At dep4 (attention DP) every rank runs the whole mixer: all 16384 channels,
#: the 8 groups of 2048; z the first 16384 columns of the in_proj row
#: z 16384 | xBC 18432 | dt 256.
DEP4_WIDTH = 16384
DEP4_PROJ_WIDTH = 35072
#: At tep8 (the BF16 checkpoint over 8 ranks) a rank holds exactly one group: 16384 / 8 =
#: 2048 channels, z the first 2048 columns of the in_proj row z 2048 | xBC 2304 | dt 32.
TEP8_WIDTH = 2048
TEP8_PROJ_WIDTH = 4384
#: `layer_norm_epsilon` of the checkpoint.
EPS = 1e-5

#: The call site: z read in place from the projection, the checkpoint's eps.
_CALL_SITE = dict(width=WIDTH, group_size=GROUP_SIZE, eps=EPS, z="proj", proj_width=PROJ_WIDTH)
#: The same call at dep4 geometry.
_DEP4_CALL_SITE = dict(_CALL_SITE, width=DEP4_WIDTH, proj_width=DEP4_PROJ_WIDTH)
#: ... and at tep8 geometry.
_TEP8_CALL_SITE = dict(_CALL_SITE, width=TEP8_WIDTH, proj_width=TEP8_PROJ_WIDTH)


class RmsNormGatedGrouped(OpWrapper):
    """`weight * rmsnorm_per_group(x * silu(z))`, new tensor out.

    `g = x * z * sigmoid(z)`; for every row and every group `G` of
    `group_size` consecutive channels,
    `out[G] = g[G] * rsqrt(mean(g[G]^2) + eps) * weight[G]` -- the gate before
    the norm, each group normalized on its own statistics, all in fp32 and
    rounded once to `x.dtype`. The weight multiply is part of that fp32
    product, before the single rounding (mamba_ssm's convention; see
    `reference` for how HF's differs).

    Fusion boundary: gate, grouped normalization and weight. The scan that
    produces `x`, the projection that produces `z` and the quantization of
    out_proj's input are the caller's.

    Fixed by the entry, not arguments: RMS (no mean), gate before the norm,
    SiLU gate, no bias, plain `weight` (not `1 + weight`), output in
    `x.dtype`. `x` and `z` are not mutated.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_107})

    # spec keys: tokens, width, group_size, eps, z ("proj": the first `width`
    # columns of a token-major `[tokens, proj_width]` buffer | "contiguous"),
    # x_scale (multiplies x), group_scales (one factor per group, applied to x).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, 1 token: z the projection's first 4096 columns",
            spec=dict(_CALL_SITE, tokens=1),
        ),
        Cell(why="decode batch 32, the CUDA-graph ceiling", spec=dict(_CALL_SITE, tokens=32)),
        Cell(why="prefill 4101 tokens: past 4096, odd", spec=dict(_CALL_SITE, tokens=4101)),
        Cell(why="prefill at the 8192-token step ceiling", spec=dict(_CALL_SITE, tokens=8192)),
        Cell(
            why="z a contiguous [T, 4096] rather than a projection slice",
            spec=dict(_CALL_SITE, tokens=32, z="contiguous"),
        ),
        Cell(
            why="rows whose mean square is near eps: eps inside the root",
            spec=dict(_CALL_SITE, tokens=32, x_scale=1e-3),
        ),
        Cell(
            why="group scales 100x apart: each group normalized independently",
            spec=dict(_CALL_SITE, tokens=32, group_scales=(1.0, 100.0)),
        ),
        Cell(
            why="dep4 decode, 1 token: 16384 wide, z the projection's first columns",
            spec=dict(_DEP4_CALL_SITE, tokens=1),
        ),
        Cell(why="dep4 decode batch 32", spec=dict(_DEP4_CALL_SITE, tokens=32)),
        Cell(why="dep4 prefill 4101 tokens", spec=dict(_DEP4_CALL_SITE, tokens=4101)),
        Cell(why="dep4 prefill at the 8192-token ceiling", spec=dict(_DEP4_CALL_SITE, tokens=8192)),
        Cell(
            why="dep4: 8 group scales spanning 1e4, each group normalized independently",
            spec=dict(
                _DEP4_CALL_SITE,
                tokens=32,
                group_scales=(1.0, 100.0, 0.01, 3.0, 30.0, 0.3, 10.0, 0.1),
            ),
        ),
        Cell(
            why="tep8 decode, 1 token: 2048 wide, one group, z the projection's first columns",
            spec=dict(_TEP8_CALL_SITE, tokens=1),
        ),
        Cell(why="tep8 decode batch 32", spec=dict(_TEP8_CALL_SITE, tokens=32)),
        Cell(why="tep8 prefill 4101 tokens", spec=dict(_TEP8_CALL_SITE, tokens=4101)),
        Cell(why="tep8 prefill at the 8192-token ceiling", spec=dict(_TEP8_CALL_SITE, tokens=8192)),
        Cell(
            why="tep8: rows whose mean square is near eps: eps inside the root",
            spec=dict(_TEP8_CALL_SITE, tokens=32, x_scale=1e-3),
        ),
    )

    note = """
    Metadata consumed: none.

    Call site (every Mamba2 layer, prefill and decode, one call over all T
    rows): `x` the scan output as a contiguous `[T, 4096]`; `z = proj[:, 0:4096]`
    -- the rank's gate columns of the in_proj output, row stride 8768, read in
    place; `weight` the rank's 4096-slice of `norm.weight`; `eps` 1e-5;
    `group_size` 2048 (n_groups 8 over 4 ranks: two whole groups per rank).
    Returns a fresh contiguous `[T, 4096]` bf16 tensor, out_proj's input.
    At dep4 the same call unsharded: `x` `[T, 16384]`, `z = proj[:, 0:16384]`
    (row stride 35072), the whole `norm.weight`, `group_size` 2048 (8 groups).

    Kernel: `_layer_norm_fwd`'s single-pass kernel, grid (T, groups), one
    program per row-group holding the whole 2048-wide group (BLOCK_N 2048, 8
    warps). Its multi-row variant serves only norm-before-gate and is never
    selected here. Each call allocates the output and a `[groups * T]` fp32
    rstd scratch from the caching allocator; it captures into a CUDA graph and
    replays against new inputs (tested).

    Numerics: against `reference`, one output ulp on at most 2.3e-5 of the
    elements, the rest bitwise -- reduction order only; relative RMS at most
    0.0045 x 2^-8. HF's bf16-after-cast weight multiply is a different
    function at that resolution (see `reference`), and the test's RMS gate
    tells the two apart.

    The op checks itself, loudly: `x.stride(-1) == 1`, `z.stride(-1) == 1`,
    `z.shape == x.shape`, `weight` of shape `(N,)` with unit stride, `N %
    group_size == 0` (bare AssertionError); a group wider than 64 KB
    (RuntimeError). Every one is driven in the test, so no `is_valid`.

    Not certified here, though the kernel takes them: `fp8_scale` (a
    float8_e4m3fn output for out_proj straight from the norm), a bias,
    norm-before-gate, the sigmoid gate, `1 + weight`.

    Hard limits
      row * row stride of x and out  int32 offsets in the kernel  boundary
        2^31 elements; the call sites reach 8192 * 35072 = 2.9e8 (z at dep4).
        Not probed.
    """

    def __call__(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        group_size: int,
    ) -> torch.Tensor:
        out, _, _ = _layer_norm_fwd(
            x,
            weight,
            None,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=False,
            is_rms_norm=True,
            fp8_scale=None,
            gate_activation="silu",
            weight_is_delta=False,
        )
        return out

    def reference(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        group_size: int,
    ) -> torch.Tensor:
        """fp32 throughout and one rounding, weight included: what the kernel
        computes, and mamba_ssm's `rms_norm_ref(norm_before_gate=False)`.

        HF's NemotronH / Zamba2 gated norm is not this function: it rounds the
        normalized value to the activation dtype first and multiplies by the
        bf16 weight in bf16, a second rounding. Measured against this kernel
        that convention is one output ulp off on 26% of the elements -- a
        relative RMS of 0.64-0.72 x 2^-8 where this reference sits at 0.0045
        -- outside the test's band; the test drives it to prove the band can
        tell the two apart.
        """
        tokens, width = x.shape
        zf = z.float()
        g = (x.float() * (zf * torch.sigmoid(zf))).view(tokens, width // group_size, group_size)
        g = g * torch.rsqrt(g.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (g.view(tokens, width) * weight.float()).to(x.dtype)


rms_norm_gated_grouped = RmsNormGatedGrouped()
