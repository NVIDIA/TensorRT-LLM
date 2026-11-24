# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused KDA ``[f_a | b] -> f_b`` decode projection on Rubin (one launch, CuTe DSL)."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ...kernels.ssm.rubin_kda_fab import k3_kda_fab_fa as _fab_fa
from ...kernels.ssm.rubin_kda_fab import k3_kda_fab_proj as _fab_proj

#: Width of f_a (the f_b reduction size).
FA_DIM = 128


class RubinKdaFab(OpWrapper):
    """The decode gate projections of one KDA layer: `bfa = bf16(x @ w_bfa^T)`, then `f_b`.

    `w_bfa` [N1, K] bf16 stacks `f_a_proj` (rows [0, 128)), `b_proj` (rows [128, 128 + H)) and
    zero rows up to N1 (a multiple of 8, at most 144). With `fa_only` the op returns `bfa`
    [M, N1] (the verify then applies `w_fb` itself); otherwise `(beta, forget_gate)`:
    `bfa[:, 128:128 + H]` and `bf16(bfa[:, :128] @ w_fb^T)` [M, N2]. New tensors.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{point}: {rows} verify rows at 12 KDA heads, both outputs",
                spec=dict(rows=rows, heads=12, hidden=7168, fa_only=False, multi_tile=False),
            )
            for point, rows in (("TEP8 b1k7", 8), ("helix8 b16k5", 96))
        ),
        Cell(
            why="helix8 b16k5 in f_a mode (the verify computes the forget gate)",
            spec=dict(rows=96, heads=12, hidden=7168, fa_only=True, multi_tile=False),
        ),
        Cell(
            why="helix8 b24k5 (144 rows): several token tiles per tail worker",
            spec=dict(rows=144, heads=12, hidden=7168, fa_only=False, multi_tile=True),
        ),
        Cell(
            why="one decode row (M = 1) at 12 heads",
            spec=dict(rows=1, heads=12, hidden=7168, fa_only=False, multi_tile=False),
        ),
    )
    note = """
    x [M, K] bf16 contiguous, 1 <= M <= 256, K a multiple of 64; w_fb
    [N2, 128] bf16 (N2 = H * 128). Takes only BF16 f_a / b / f_b weights;
    no target calls it (the specialized targets' f_a / b are MXFP8).
    multi_tile lets each tail worker take several token tiles where one
    tile does not fit (the helix8 mid-M batch); tilings that fit are the
    same either way. In f_a mode w_fb is only warmed in L2.

    fp32 accumulation, bfa rounded to bf16 before the f_b product, one
    rounding per output. The kernel keeps a per-device workspace and
    counter: launches on one device must be stream ordered. It runs next to
    the qkvg GEMM on the aux stream; warm it before graph capture (the
    workspace is created on first use).
    """

    def __call__(
        self,
        x: torch.Tensor,
        w_bfa: torch.Tensor,
        w_fb: Optional[torch.Tensor],
        num_heads: int,
        fa_only: bool,
        multi_tile: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if fa_only:
            return _fab_fa(x, w_bfa, num_heads, w_fb, multi_tile=multi_tile)
        return _fab_proj(x, w_bfa, w_fb, num_heads, multi_tile=multi_tile)

    def reference(
        self,
        x: torch.Tensor,
        w_bfa: torch.Tensor,
        w_fb: Optional[torch.Tensor],
        num_heads: int,
        fa_only: bool,
        multi_tile: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """fp32 products, bfa rounded to bf16 as the kernel hands it on; `multi_tile` only tiles."""
        with true_fp32_matmul():
            bfa = (x.float() @ w_bfa.float().t()).to(torch.bfloat16)
            if fa_only:
                return bfa
            gate = (bfa[:, :FA_DIM].float() @ w_fb.float().t()).to(torch.bfloat16)
        return bfa[:, FA_DIM : FA_DIM + num_heads].contiguous(), gate


rubin_kda_fab = RubinKdaFab()
