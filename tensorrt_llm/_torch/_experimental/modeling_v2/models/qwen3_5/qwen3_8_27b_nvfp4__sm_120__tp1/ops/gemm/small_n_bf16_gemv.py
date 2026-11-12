# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN in_proj_ba at decode rows: a Triton CUDA-core GEMV for a 96-wide BF16 projection.

For an output width this small cuBLAS on SM120 picks a 16x16-tile WMMA kernel
that launches N/16 CTAs walking K serially (M 2-4), a dot + reduce pair (M = 1),
or split-K plus a separate reduce (M 5-16) -- all latency-bound on a 1 MB weight.
One program per (weight row, activation row) with the whole K unrolled
(``_small_n_gemv``) is one short launch. Above 16 rows in_proj_ba stays on cuBLAS.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ._small_n_gemv import small_n_gemv


def _cell(why: str, m: int) -> Cell:
    return Cell(why=why, spec=dict(m=m, n=96, k=5120))


class SmallNBf16Gemv(OpWrapper):
    """``x @ weight.T`` in BF16 with fp32 accumulation, one rounding: ``x`` ``[M, K]``,
    ``weight`` ``[N, K]`` (the stored weight), a fresh ``[M, N]`` BF16 result."""

    ARCHS = frozenset({Arch.SM_120})

    # GDN in_proj_ba (2 x 48 value heads = 96 outputs, hidden 5120) at rows 1-16; it runs
    # on the auxiliary stream beside in_proj_qkvz when the step is captured.
    CELLS: tuple[Cell, ...] = (
        _cell("decode M=1", 1),
        _cell("decode M=2", 2),
        _cell("decode M=4", 4),
        _cell("decode M=9: split-K territory for cuBLAS", 9),
        _cell("decode M=16: the route's last row", 16),
    )

    note = """
    Metadata consumed: none. Routing (modeling.py): rows 1..16; above, cuBLAS.

    Preconditions: BF16 operands, unit inner stride on both (`weight` the stored
    `[N, K]` tensor, not its transpose), K a multiple of 256 (the launcher asserts).

    Numerics: products and sums in fp32 -- per-thread partials over the unrolled K
    steps, then a fixed tree -- one rounding; reordered against cuBLAS, bitwise
    repeatable.

    Compilation: Triton JIT on the first call per (K, dtypes); make it before CUDA-graph
    capture (the engine's warmup does).

    Hard limits
      M   any (one program row per activation row)   driven 1..16
      K   multiple of 256, unrolled statically       5120 driven
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return small_n_gemv(x, weight)

    def reference(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """fp32-accumulated (TF32 off), one rounding."""
        with true_fp32_matmul():
            return (x.float() @ weight.float().t()).to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor) -> None:
        # The kernel addresses rows by their stride and assumes unit inner stride: a
        # transposed or column-strided operand is read as dense rows, silently.
        assert x.dim() == 2 and x.stride(-1) == 1, "x must be [M, K] with unit inner stride"
        assert weight.dim() == 2 and weight.stride(-1) == 1, (
            "weight must be the stored [N, K] tensor with unit inner stride"
        )


small_n_bf16_gemv = SmallNBf16Gemv()
