# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 x BF16 -> BF16 low-M GEMM of the Eagle3 drafter's linears at fixed FlashInfer tactics (SM10x)."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: (N, K) of the TP8 drafter shards -> ((max rows, kind, tactic), ...), rows ascending. Kind "d" is FlashInfer's direct
#: SIMT kernel with tactic (block, outputs per block, rows per block); kind "s" is its split-K tensor-core kernel with
#: tactic (mma_m, mma_n, split_k, ab_stages). A row count is served by the first entry whose max rows holds it; rows
#: past the last entry are not served. GB300 cold L2, CUDA-graph timing, us cuBLAS -> low-M at 1 / 4 rows: qkv 11.9 ->
#: 6.4 / 12.0 -> 7.8, o 3.6 -> 2.3 / -, gate_up 11.9 -> 9.4 / 12.1 -> 11.0, down 6.8 -> 5.6 / -, fc 37.9 -> 31.9 /
#: 39.2 -> 35.5.
TACTICS: Mapping[Tuple[int, int], tuple] = {
    # qkv
    (1280, 12288): (
        (1, "d", (384, 2, 1)),
        (2, "d", (128, 4, 2)),
        (3, "d", (128, 4, 3)),
        (8, "s", (64, 8, 4, 12)),
    ),
    # o_proj
    (6144, 1024): ((1, "d", (32, 4, 1)), (2, "d", (32, 4, 2))),
    # gate_up
    (4608, 6144): (
        (1, "d", (128, 1, 1)),
        (2, "d", (256, 2, 2)),
        (3, "s", (128, 8, 3, 6)),
        (4, "s", (128, 8, 3, 5)),
        (8, "s", (64, 8, 2, 10)),
    ),
    # down
    (6144, 2304): ((1, "d", (96, 1, 1)), (2, "d", (96, 2, 2))),
    # the Eagle3 fc over the three captured hidden states
    (6144, 18432): (
        (1, "d", (384, 2, 1)),
        (2, "d", (64, 4, 2)),
        (3, "d", (32, 2, 3)),
        (8, "s", (64, 8, 1, 12)),
    ),
}


def tactic_for(n: int, k: int, m: int) -> Optional[Tuple[str, tuple]]:
    """(kind, tactic) of an [m, k] x [n, k]^T call, or None when the table does not serve it."""
    for rows, kind, tactic in TACTICS.get((n, k), ()):
        if m <= rows:
            return kind, tactic
    return None


#: The drafter linear each (N, K) of the table is.
_NAMES: Mapping[Tuple[int, int], str] = {
    (1280, 12288): "qkv",
    (6144, 1024): "o_proj",
    (4608, 6144): "gate_up",
    (6144, 2304): "down",
    (6144, 18432): "fc",
}


class M3DraftLowM(OpWrapper):
    """``out`` [M, N] BF16 = ``x`` [M, K] BF16 @ ``weight`` [N, K]^T BF16, FP32 accumulation.

    ``kind`` / ``tactic`` pick FlashInfer's kernel: "d" runs ``run_direct_dense`` (SIMT, one dot product per output in
    K order), "s" runs ``run_splitk_dense`` (tcgen05 MMA, K split over ``split_k`` CTAs and summed in FP32). Both
    round once to BF16. The kernel is compiled on its first call per (M, N, K, tactic, PDL); the target compiles every
    covered call before warmup (modeling's ``_prebuild_draft_lowm``).
    """

    ARCHS = frozenset({Arch.SM_103})
    # Every row count of every table entry.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"drafter {_NAMES[nk]}, {m} rows: {tactic_for(nk[0], nk[1], m)[0]} tactic",
            spec=dict(nk=nk, m=m),
        )
        for nk, table in TACTICS.items()
        for m in range(1, table[-1][0] + 1)
    )

    note = """
    The table is the TP8 drafter's: its shapes are the per-rank shards of the Eagle3 layer at 8-way tensor
    parallelism (64 query and 4 KV heads, intermediate 18432) and the replicated fc. Certified at those shapes and
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
    ) -> torch.Tensor:
        if kind == "d":
            from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
                DirectTactic,
                run_direct_dense,
            )

            run_direct_dense(x, weight.t(), out, use_pdl, DirectTactic(*tactic))
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
    ) -> None:
        """The kernels index ``x`` and ``out`` densely: a strided view is read as if it were packed."""
        assert x.dim() == 2 and x.is_contiguous(), "x must be a contiguous [M, K] matrix"
        assert weight.dim() == 2 and weight.is_contiguous(), (
            "weight must be a contiguous [N, K] matrix"
        )
        assert out.is_contiguous() and tuple(out.shape) == (x.shape[0], weight.shape[0]), (
            "out must be a contiguous [M, N] matrix"
        )


m3_draft_lowm = M3DraftLowM()
