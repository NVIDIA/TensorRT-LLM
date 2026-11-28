# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The critical half of a decode-sized Hyper-Connections boundary: post update, collapse, RMSNorm; optionally
with the tensor-parallel all-reduce of the previous sub-layer's output in the same launch."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post_mapping import (
    hc_post_mapping,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre import hc_pre

MAX_ROWS = 4096  # the split boundary's range (`split_max_tokens`; an attention entry with a post update stops at 1023)


def _norm(h: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float) -> torch.Tensor:
    li = h.double()
    return (
        li * torch.rsqrt(li.square().mean(-1, keepdim=True) + norm_eps) * norm_weight.double()
    ).to(h.dtype)


class HcPostCollapseNorm(OpWrapper):
    """`(x_prev | None, residual [n, hc, d], post_prev, comb_prev, pre_mix [n, hc], norm_weight)` ->
    `(residual_cur [n, hc, d], h [n, d])`, with `mxfp8` also `h_mx`.

    The engine's `mhc_post_collapse_norm` (one CuTe DSL launch): `residual_cur = hc_post_mapping(x_prev, ...)`
    (or `residual` as is without `x_prev`), `h = rmsnorm(hc_pre(residual_cur, pre_mix), norm_weight)`. This
    boundary's own mixes are not computed here (`hc_coefficients` does, off the critical path). With `mxfp8` the same
    launch also writes `h` as `mxfp8_quantize(h, True)` gives it (E4M3 rows, swizzled UE8M0 scales; the scale tiles'
    padding rows, which no product reads, are left unwritten), for the quantized q_a | kv GEMM past the low-M rows.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="decode rows 6 with the post update", spec=dict(n=6, d=5120, post=True)),
        Cell(why="decode rows 24 with the post update", spec=dict(n=24, d=5120, post=True)),
        Cell(why="decode rows 64: the largest split boundary", spec=dict(n=64, d=5120, post=True)),
        Cell(why="collapse only, rows 6", spec=dict(n=6, d=5120, post=False)),
        Cell(
            why="dep32 decode rows 24 with the MXFP8 copy",
            spec=dict(n=24, d=5120, post=True, mxfp8=True),
        ),
        Cell(
            why="rows 64 with the MXFP8 copy, collapse only",
            spec=dict(n=64, d=5120, post=False, mxfp8=True),
        ),
        Cell(
            why="dep32 c7600 decode rows 150 with the MXFP8 copy",
            spec=dict(n=150, d=5120, post=True, mxfp8=True),
        ),
        Cell(
            why="dep8 c7000 decode rows 160 with the post update",
            spec=dict(n=160, d=5120, post=True),
        ),
        Cell(
            why="rows 1023 with the post update, read after the wait",
            spec=dict(n=1023, d=5120, post=True, residual_after_wait=True),
        ),
        Cell(
            why="dep8 batch 320: an FFN boundary of 1280 rows",
            spec=dict(n=1280, d=5120, post=True),
        ),
    )
    note = "Allocates its outputs only and syncs nothing; the kernel is JIT-compiled per row count on first use."

    def __call__(
        self,
        x_prev: Optional[torch.Tensor],
        residual: torch.Tensor,
        post_prev: Optional[torch.Tensor],
        comb_prev: Optional[torch.Tensor],
        pre_mix: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        mxfp8: bool = False,
        residual_after_wait: bool = False,
    ) -> tuple:
        from tensorrt_llm._torch.modules.mhc.mhc_lowm import mhc_post_collapse_norm

        return mhc_post_collapse_norm(
            x_prev,
            residual,
            post_prev,
            comb_prev,
            pre_mix,
            norm_weight,
            norm_eps,
            mxfp8=mxfp8,
            residual_after_wait=residual_after_wait,
        )

    def reference(
        self,
        x_prev: Optional[torch.Tensor],
        residual: torch.Tensor,
        post_prev: Optional[torch.Tensor],
        comb_prev: Optional[torch.Tensor],
        pre_mix: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        mxfp8: bool = False,
        residual_after_wait: bool = False,
    ) -> tuple:
        cur = (
            residual
            if x_prev is None
            else hc_post_mapping.reference(x_prev, residual, post_prev, comb_prev)
        )
        h = _norm(hc_pre.reference(cur, pre_mix), norm_weight, norm_eps)
        if mxfp8:
            return cur, h, torch.ops.trtllm.mxfp8_quantize(h.contiguous(), True)
        return cur, h

    def is_valid(
        self,
        x_prev: Optional[torch.Tensor],
        residual: torch.Tensor,
        post_prev: Optional[torch.Tensor],
        comb_prev: Optional[torch.Tensor],
        pre_mix: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        mxfp8: bool = False,
        residual_after_wait: bool = False,
    ) -> None:
        assert residual.dim() == 3 and residual.shape[1] == 4, "residual is [n, 4, d]"
        n, _, d = residual.shape
        assert 0 < n <= MAX_ROWS, "the split boundary's rows"
        assert residual.dtype == torch.bfloat16 and residual.is_contiguous() and residual.is_cuda
        assert d % 256 == 0 and d // 8 <= 1024, "the kernel's hidden-size range"
        assert (x_prev is None) == (post_prev is None) == (comb_prev is None), (
            "x_prev, post_prev, comb_prev go together"
        )
        if x_prev is not None:
            assert x_prev.shape == (n, d) and x_prev.dtype == torch.bfloat16
        assert (
            pre_mix.shape == (n, 4)
            and norm_weight.shape == (d,)
            and norm_weight.dtype == torch.bfloat16
        )


hc_post_collapse_norm = HcPostCollapseNorm()
