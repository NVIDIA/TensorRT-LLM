# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 x BF16 -> BF16 low-M GEMM of the Eagle3 drafter's linears at fixed FlashInfer tactics (SM10x)."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: (N, K) of the TP8 drafter shards -> ((max rows, kind, tactic), ...), rows ascending. Kind "d" is FlashInfer's direct
#: SIMT kernel with tactic (block, outputs per block, rows per block); kind "s" is its split-K tensor-core kernel with
#: tactic (mma_m, mma_n, split_k, ab_stages). A row count is served by the first entry whose max rows holds it; the
#: rows modeling's DRAFT_LOWM covers are only the ones measured (the fastest tactic at no larger error vs fp64 than
#: cuBLAS). TP8 shards (64 query / 4 KV heads of 128, KV heads replicated across rank pairs; intermediate 18432):
#: qkv 1280 x 12288, o_proj 6144 x 1024, gate_up 4608 x 6144, down 6144 x 2304. Rubin SM107 TP8 sweep (mclk
#: locked, rotating weights >= 3x L2, CUDA-graph timing), us cuBLAS -> low-M at 1 / 4 rows: qkv 6.15 -> 3.94 / 6.26 ->
#: 5.69, o 4.97 -> 2.42 / 4.48 -> 2.68, gate_up 8.09 -> 5.78 / 8.11 -> 6.99, down 5.62 -> 4.00 / 5.74 -> 3.94. The fc is
#: replicated (the TP2 shape): its TP2 tactic and row.
TACTICS: Mapping[Tuple[int, int], tuple] = {
    # qkv
    (1280, 12288): ((1, "d", (256, 2, 1)), (4, "d", (128, 4, 4))),
    # o_proj
    (6144, 1024): ((1, "s", (64, 8, 1, 10)), (4, "s", (64, 8, 2, 6))),
    # gate_up
    (4608, 6144): ((1, "d", (128, 2, 1)), (4, "d", (64, 4, 4))),
    # down
    (6144, 2304): ((1, "s", (64, 8, 2, 6)), (4, "s", (64, 8, 1, 10))),
    # the Eagle3 fc over the three captured hidden states
    (6144, 18432): ((1, "d", (384, 2, 1)),),
}
#: Rows each (N, K) is certified at (a subset of what its table would serve).
ROWS: Mapping[Tuple[int, int], Tuple[int, ...]] = {
    (1280, 12288): (1, 4),
    (6144, 1024): (1, 4),
    (4608, 6144): (1, 4),
    (6144, 2304): (1, 4),
    (6144, 18432): (1,),
}


def tactic_for(n: int, k: int, m: int) -> Optional[Tuple[str, tuple]]:
    """(kind, tactic) of an [m, k] x [n, k]^T call, or None when the table does not serve it."""
    for rows, kind, tactic in TACTICS.get((n, k), ()):
        if m <= rows:
            return kind, tactic
    return None


class M3DraftLowM(OpWrapper):
    """``out`` [M, N] BF16 = ``x`` [M, K] BF16 @ ``weight`` [N, K]^T BF16, FP32 accumulation.

    ``kind`` / ``tactic`` pick FlashInfer's kernel: "d" runs ``run_direct_dense`` (SIMT, one dot product per output in
    K order), "s" runs ``run_splitk_dense`` (tcgen05 MMA, K split over ``split_k`` CTAs and summed in FP32). Both
    round once to BF16. The kernel is compiled on its first call per (M, N, K, tactic, PDL); the target compiles every
    covered call before warmup (``M3Eagle3Draft.prebuild_linears``).
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="drafter qkv, 1 rows: d tactic", spec=dict(nk=(1280, 12288), m=1)),
        Cell(why="drafter qkv, 4 rows: d tactic", spec=dict(nk=(1280, 12288), m=4)),
        Cell(why="drafter o_proj, 1 rows: s tactic", spec=dict(nk=(6144, 1024), m=1)),
        Cell(why="drafter o_proj, 4 rows: s tactic", spec=dict(nk=(6144, 1024), m=4)),
        Cell(why="drafter gate_up, 1 rows: d tactic", spec=dict(nk=(4608, 6144), m=1)),
        Cell(why="drafter gate_up, 4 rows: d tactic", spec=dict(nk=(4608, 6144), m=4)),
        Cell(why="drafter down, 1 rows: s tactic", spec=dict(nk=(6144, 2304), m=1)),
        Cell(why="drafter down, 4 rows: s tactic", spec=dict(nk=(6144, 2304), m=4)),
        Cell(why="drafter fc, 1 rows: d tactic", spec=dict(nk=(6144, 18432), m=1)),
    )

    note = """
    The table is the TP8 drafter's: its shapes are the per-rank shards of the Eagle3 layer at 8-way tensor
    parallelism (64 query and 4 KV heads, the KV heads replicated across rank pairs; intermediate 18432) and the
    replicated fc, at the b1 decode rows (1 row in draft passes 1-2, 4 in pass 0). Certified at those shapes and
    rows only; FlashInfer's kernels take other shapes, which are not measured here.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        kind: str,
        tactic: tuple,
        use_pdl: bool,
        own_splitk: bool = False,
        ar: Optional[tuple] = None,
    ) -> torch.Tensor:
        if kind == "d":
            from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
                DirectTactic,
                run_direct_dense,
            )

            run_direct_dense(x, weight.t(), out, use_pdl, DirectTactic(*tactic))
        elif own_splitk:
            # This target's copy of the split-K kernel (kernels/m3_lowm_splitk): bit for bit FlashInfer's; with ``ar``
            # it also pushes the rows into a drafter boundary all-reduce (modeling's DRAFT_AR_PUSH).
            from ...kernels.m3_lowm_splitk import (
                SplitKTactic,
                run_splitk_dense,
                run_splitk_dense_push,
            )

            if ar is not None:
                run_splitk_dense_push(x, weight.t(), out, use_pdl, SplitKTactic(*tactic), ar)
            else:
                run_splitk_dense(x, weight.t(), None, out, use_pdl, SplitKTactic(*tactic))
        else:
            from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
                SplitKTactic,
                run_splitk_dense,
            )

            run_splitk_dense(x, weight.t(), None, out, use_pdl, SplitKTactic(*tactic))
        return out

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        kind: str,
        tactic: tuple,
        use_pdl: bool,
        own_splitk: bool = False,
        ar: Optional[tuple] = None,
    ) -> torch.Tensor:
        """bf16(x @ weight^T) summed in fp64 and rounded once; ``out``, ``kind``, ``tactic`` and ``use_pdl`` only
        choose where and how the op writes."""
        return (x.double() @ weight.double().t()).to(torch.bfloat16)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        kind: str,
        tactic: tuple,
        use_pdl: bool,
        own_splitk: bool = False,
        ar: Optional[tuple] = None,
    ) -> None:
        """The kernels index ``x`` and ``out`` densely: a strided view is read as if it were packed."""
        assert x.dim() == 2 and x.is_contiguous(), "x must be a contiguous [M, K] matrix"
        assert weight.dim() == 2 and weight.is_contiguous(), (
            "weight must be a contiguous [N, K] matrix"
        )
        assert out.is_contiguous() and tuple(out.shape) == (x.shape[0], weight.shape[0]), (
            "out must be a contiguous [M, N] matrix"
        )
        assert ar is None or (own_splitk and kind == "s" and len(ar) == 5), (
            "ar (the boundary all-reduce push) runs only on this target's split-K copy"
        )


m3_draft_lowm = M3DraftLowM()
