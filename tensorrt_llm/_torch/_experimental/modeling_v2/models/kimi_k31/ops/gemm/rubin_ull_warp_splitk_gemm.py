# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM main's warp-specialized split-K BF16 GEMM, run through its prepared engine
(``kernels/vendor/vllm_main/.../gemm/warp_splitk.py``'s ``PreparedWarpSplitK``)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _cell(why: str, rows: int) -> Cell:
    return Cell(why=why, spec=dict(rows=rows))


class WarpSplitKGemm(OpWrapper):
    """``out = x @ engine.weight^T``: BF16 ``x`` [M, K] packed row-major, ``out`` packed BF16
    [M, N], both 32-byte aligned; ``engine`` a ``PreparedWarpSplitK`` holding the weight's view and
    one compiled kernel per prepared ``M`` (its tactic). FP32 accumulation rounded once. Returns
    ``out``."""

    ARCHS = frozenset({Arch.SM_103})

    # The DSA layer's packed [hidden | gate] bank (2272 + 768 = 3040 rows x 7168) at the row counts
    # the in-projection prepares (3..16), each with the device's ``dsa_front.warp_tactic``.
    CELLS: tuple[Cell, ...] = (
        _cell("b1 verify: 8 rows of the packed DSA bank", 8),
        _cell("b2 verify: 16 rows (token tile 16)", 16),
        _cell("3 rows, the smallest the in-projection serves", 3),
        _cell("12 rows: a partial 16-row token tile", 12),
    )

    note = """
    The engine is prepared per layer before CUDA-graph capture (``DsaInProjection.prepare``: the
    tactics, the compiled code shared through its cache, the weight's view); an ``M`` it was not
    prepared for raises KeyError. The kernel never reads its bias operand (a row of ``out`` fills
    it). Not bitwise the planned split-K projection or cuBLAS (another split of K).
    """

    def __call__(self, engine, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return engine.run(x, out)

    def reference(self, engine, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The FP64 product with the engine's weight, rounded once to BF16; ``out`` is not
        read."""
        return (x.double() @ engine.weight.double().t()).to(torch.bfloat16)

    def is_valid(self, engine, x: torch.Tensor, out: torch.Tensor) -> None:
        """The kernel reads ``x`` and writes ``out`` as packed rows from their base pointers and
        does not check either."""
        if not x.is_contiguous() or not out.is_contiguous():
            raise ValueError("warp_splitk_gemm: x and out must be packed row-major")
        if x.data_ptr() % 32 or out.data_ptr() % 32:
            raise ValueError("warp_splitk_gemm: x and out must be 32-byte aligned")
        if x.shape[1] != engine.weight.shape[1] or out.shape != (
            x.shape[0],
            engine.weight.shape[0],
        ):
            raise ValueError("warp_splitk_gemm: x [M, K] and out [M, N] must match the weight")


warp_splitk_gemm = WarpSplitKGemm()
