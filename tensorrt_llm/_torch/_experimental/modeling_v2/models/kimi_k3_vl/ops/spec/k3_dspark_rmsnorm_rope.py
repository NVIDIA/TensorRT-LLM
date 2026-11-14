# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark RMSNorm + RoPE of BF16 rows read in place from a row-strided view (CuTe DSL)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._dspark_mla_rope import k3r_dspark_rmsnorm_rope


def rmsnorm_rope_reference(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
    norm_dim: Optional[int] = None,
) -> torch.Tensor:
    """FP64 RMSNorm over the first ``norm_dim`` columns (all when None), scaled by ``weight``,
    then the interleaved-pair RoPE of the last ``rope_dim`` columns with row ``r``'s
    ``(cos, sin)`` from ``freqs[r // num_heads]``; rounded once to ``x``'s dtype, compact."""
    rows = x.reshape(-1, x.shape[-1]).double()
    n = rows.shape[-1] if norm_dim is None else norm_dim
    head = rows[:, :n]
    if apply_rmsnorm:
        head = head * torch.rsqrt(head.square().mean(dim=-1, keepdim=True) + eps)
    if apply_weight:
        head = head * weight.double()
    out = torch.cat([head, rows[:, n:]], dim=-1)
    if rope_dim > 0:
        row_freqs = freqs.double().repeat_interleave(num_heads, dim=0)[: out.shape[0]]
        cos, sin = row_freqs[..., 0], row_freqs[..., 1]
        if inverse_rope:
            sin = -sin
        real = out[:, -rope_dim::2].clone()
        imag = out[:, -rope_dim + 1 :: 2].clone()
        out[:, -rope_dim::2] = real * cos - imag * sin
        out[:, -rope_dim + 1 :: 2] = imag * cos + real * sin
    return out.to(x.dtype).view(x.shape)


class K3DsparkRmsnormRope(OpWrapper):
    """``[..., hidden]`` BF16, compact and new: RMSNorm of each row's first ``norm_dim``
    columns (all when None; skipped without ``apply_rmsnorm``), times ``weight`` (with
    ``apply_weight``), then RoPE of the last ``rope_dim`` columns, pairs interleaved,
    ``(cos, sin)`` of row ``r`` from ``freqs[r // num_heads]`` (``[P, rope_dim // 2, 2]`` FP32),
    the inverse rotation with ``inverse_rope``. Columns between the norm and the rope pass
    through.

    ``x`` is BF16 with a unit inner stride and any regular row stride (read in place);
    ``weight`` BF16 ``[norm_dim]``, or None without ``apply_weight``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS = (
        Cell(
            why="dep16 draft block latents, 16 requests x 6 rows (DSpark 5): compact [512 latent "
            "| 64 rope] rows, the latent normalized with its weight",
            spec=dict(rows=96, row_stride=576),
        ),
        Cell(
            why="helix8 c240 draft block latents, 20 requests x 6 rows: compact rows",
            spec=dict(rows=120, row_stride=576),
        ),
        Cell(
            why="helix8 c24 / c42 draft block latents, 2 requests x 8 rows (DSpark 7), read in "
            "place from the fused [q_a | kv_a] GEMM output (row stride 2112)",
            spec=dict(rows=16, row_stride=2112),
        ),
    )

    note = """
    ``cute_dsl_dspark_rmsnorm_rope``'s kernel and support check, compiled for row-strided input:
    the MLA drafter normalizes column slices of its fused [q_a | kv_a] GEMM output without a
    contiguous copy. FP32 throughout, one rounding per output. A process-wide compile cache holds
    one kernel per (hidden, rope_dim, heads, eps, weight, norm, inverse, norm_dim); compile before
    capture. Raises ValueError on input the support check refuses.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        freqs: torch.Tensor,
        num_heads: int,
        rope_dim: int,
        eps: float,
        apply_weight: bool,
        apply_rmsnorm: bool,
        inverse_rope: bool,
        norm_dim: Optional[int] = None,
    ) -> torch.Tensor:
        return k3r_dspark_rmsnorm_rope(
            x,
            weight,
            freqs,
            num_heads,
            rope_dim,
            eps,
            apply_weight,
            apply_rmsnorm,
            inverse_rope,
            norm_dim,
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        freqs: torch.Tensor,
        num_heads: int,
        rope_dim: int,
        eps: float,
        apply_weight: bool,
        apply_rmsnorm: bool,
        inverse_rope: bool,
        norm_dim: Optional[int] = None,
    ) -> torch.Tensor:
        return rmsnorm_rope_reference(
            x,
            weight,
            freqs,
            num_heads,
            rope_dim,
            eps,
            apply_weight,
            apply_rmsnorm,
            inverse_rope,
            norm_dim,
        )

    def is_valid(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        freqs: torch.Tensor,
        num_heads: int,
        rope_dim: int,
        eps: float,
        apply_weight: bool,
        apply_rmsnorm: bool,
        inverse_rope: bool,
        norm_dim: Optional[int] = None,
    ) -> None:
        rows = x.numel() // x.shape[-1]
        # The kernel indexes freqs by row // num_heads without a bound.
        assert freqs.shape[0] * num_heads >= rows, "fewer frequency rows than rows"
        if apply_weight:
            n = x.shape[-1] if norm_dim is None else norm_dim
            assert weight.numel() == n, "weight must span the normalized columns"


k3_dspark_rmsnorm_rope = K3DsparkRmsnormRope()
