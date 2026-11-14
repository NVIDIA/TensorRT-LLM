# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mHC pre-mapping's mixer product and row sum of squares, on CUDA-core FP32 FMA.

The first of the two kernels of an unfused pre-mapping (``mHC.pre_mapping``):
``torch.ops.trtllm.mhc_gemm_sqrsum_fma`` (``mhcGemmSqrsumFmaKernel``). For the ``M`` rows of the
residual streams flattened stream-major, ``x = residual.view(M, K)`` with ``K = mult * hidden``
(BF16), and the next sublayer's mixer ``fn`` (FP32 ``[mix, K]``, ``mix = (2 + mult) * mult = 24``)::

    mixes[t, n] = sum_k x[t, k] * fn[n, k]     # [M, 24] FP32, before the RMS scaling
    sqrsum[t]   = sum_k x[t, k] ** 2           # [M] FP32

``mhc_big_fuse`` turns the two into the coefficients and the layer input. One CTA of 256 threads
computes ``tile_n`` columns of one row over all of ``K``: each thread runs a sequential FMA chain over
its 64 elements (4 of every 1024), then a warp butterfly and a sum over the 8 warps in warp order.
Each column is computed the same way whatever the tile width, so ``tile_n`` changes the grid and not
the result. A BF16-by-FP32 product is exact in FP32, so the result differs from any other FP32
summation of the same products only in rounding order.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: GLM-5.3-Flash's hyper-connection: 4 streams of 4096, 24 mixer outputs.
MULT = 4
HIDDEN = 4096
MIX = (2 + MULT) * MULT

#: The tile widths the kernel is instantiated for: the divisors of 24.
TILE_N_OPTIONS = (1, 2, 3, 4, 6, 8, 12, 24)


def gemm_sqrsum_tile_n(rows: int) -> int:
    """The kernel's own default tile width for ``rows`` rows (``selectFmaTileN`` at 24 columns).

    One column per CTA up to 32 rows, eight above. It is what ``mHC.pre_mapping`` runs when the
    autotuner has no entry for the step.
    """
    return 1 if rows <= 32 else 8


class MhcGemmSqrsum(OpWrapper):
    """``(x @ fn.T, (x * x).sum(-1))`` in FP32 for BF16 ``x [M, K]`` and FP32 ``fn [24, K]``.

    Args:
        x: ``[M, K]`` BF16, contiguous: the residual streams ``[M, 4, hidden]`` viewed as
            ``[M, 4 * hidden]``.
        fn: ``[24, K]`` FP32, contiguous: the next sublayer's mixer weights as stored
            (``hc_attn_fn`` / ``hc_ffn_fn``; the checkpoint's BF16 values held as FP32).
        tile_n: output columns per CTA, one of ``TILE_N_OPTIONS``; ``gemm_sqrsum_tile_n(M)`` is the
            kernel's default. It does not change the result.

    Returns:
        ``(mixes, sqrsum)``: new ``[M, 24]`` and ``[M]`` FP32 tensors.
    """

    ARCHS = frozenset({Arch.SM_100})

    _GLM = dict(mult=MULT, hidden=HIDDEN)
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode of one request without MTP", spec=dict(rows=1, **_GLM)),
        Cell(why="MTP-3 verify of one request (4 rows)", spec=dict(rows=4, **_GLM)),
        Cell(why="MTP-3 verify of two requests (8 rows)", spec=dict(rows=8, **_GLM)),
        Cell(
            why="MTP-3 verify of 8 requests (32 rows): the default tile's last one-column step",
            spec=dict(rows=32, **_GLM),
        ),
        Cell(why="MTP-3 verify of 64 requests (256 rows)", spec=dict(rows=256, **_GLM)),
        Cell(
            why="a quarter of a 13,312-token prompt: one rank's rows under sequence parallelism",
            spec=dict(rows=3328, **_GLM),
        ),
    )

    note = """
    The target runs the pre-mapping once per forward step, on the embedding's expanded streams;
    every later boundary goes through `mhc_fused_hc`. Only the CUDA-core kernel is wrapped: the
    DeepGEMM TF32 product that `mHC.pre_mapping`'s autotuner can also pick is not offered. Each CTA
    reads its row of `x` and `tile_n` rows of `fn` in full, so every row of `x` is read
    `24 / tile_n` times. Launched without programmatic dependent launch; a call of no rows launches
    nothing. Outputs are allocated per call, from the graph's private pool under CUDA graph capture.
    """

    def __call__(
        self, x: torch.Tensor, fn: torch.Tensor, tile_n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, k = x.shape
        mix = fn.shape[0]
        mixes = torch.empty((rows, mix), dtype=torch.float32, device=x.device)
        sqrsum = torch.empty((rows,), dtype=torch.float32, device=x.device)
        # tile_m is reserved by the launcher and ignored.
        torch.ops.trtllm.mhc_gemm_sqrsum_fma(x, fn, mixes, sqrsum, rows, mix, k, tile_n, 0)
        return mixes, sqrsum

    def reference(
        self, x: torch.Tensor, fn: torch.Tensor, tile_n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The same sums in FP32: a true-FP32 matmul and a sum of squares. ``tile_n`` is ignored."""
        xf = x.float()
        with true_fp32_matmul():
            mixes = xf @ fn.float().t()
        return mixes, xf.square().sum(-1)

    def is_valid(self, x: torch.Tensor, fn: torch.Tensor, tile_n: int) -> None:
        assert x.dim() == 2 and fn.dim() == 2 and fn.shape[1] == x.shape[1], (
            f"x is [M, K] and fn [mix, K]; got {tuple(x.shape)} and {tuple(fn.shape)}"
        )
        assert x.is_contiguous() and fn.is_contiguous(), (
            "the kernel reads x and fn row-major with a row stride of K elements"
        )


mhc_gemm_sqrsum = MhcGemmSqrsum()
