# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma fused residual add + RMS normalization at one row per CTA, 320 threads a row: the
decode-sized add+norm, bitwise the stock kernel's output.

The kernel is flashinfer's CuTe-DSL ``FusedAddRMSNormKernel`` subclassed with a one-row
geometry that loads and stores with 320 threads and replays the stock 64-thread row sum
from shared memory; it lives in ``_gemma_add_rmsnorm_1row``. At 16..128 rows the stock
grid (two rows a CTA, 64 threads a row) puts 8..64 CTAs of 128 threads on the GPU; this
one puts 16..128 CTAs of 320.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ._gemma_add_rmsnorm_1row import gemma_fused_add_rmsnorm_1row
from .flashinfer_gemma_fused_add_rmsnorm import flashinfer_gemma_fused_add_rmsnorm

#: The checkpoint's residual stream: hidden 5120, bf16, `rms_norm_eps` 1e-6.
_CKPT = dict(hidden=5120, dtype=torch.bfloat16, eps=1e-6)


class GemmaAddRmsnorm1Row(OpWrapper):
    """`residual = x + residual; x = residual / sqrt(mean(residual^2) + eps) * (1 + w)`, in
    place; returns `(x, residual)`, the same two tensors.

    Bitwise the stock gemma add+norm (`flashinfer_gemma_fused_add_rmsnorm`): the residual
    is the same rounding of the same fp32 sum, and the row's sum of squares is accumulated
    in the stock kernel's order (64 partials, each sequential over its 80 columns, then
    the butterfly warp reduce and warp 0 + warp 1), so `rstd` and `x` match bit for bit.

    Fusion boundary: as the stock add+norm.
    """

    ARCHS = frozenset({Arch.SM_120})

    # The target runs it at 16..128 rows: the post-attention norm at 16..32 and
    # 113..128, the next layer's input norm at 33..128, the final norm at 16..128.
    CELLS: tuple[Cell, ...] = (
        Cell(why="16 rows: the lowest row count routed here", spec=dict(rows=16, **_CKPT)),
        Cell(why="decode 32 rows (c32): the post-attention norm", spec=dict(rows=32, **_CKPT)),
        Cell(
            why="33 rows: the next layer's input norm above the early-trigger band",
            spec=dict(rows=33, **_CKPT),
        ),
        Cell(
            why="113 rows: the post-attention norm between the two NVFP4-epilogue bands",
            spec=dict(rows=113, **_CKPT),
        ),
        Cell(
            why="decode 128 rows (concurrency 128): the top of the band",
            spec=dict(rows=128, **_CKPT),
        ),
    )

    note = """
    Metadata consumed: none.

    Kernel: grid `M` CTAs of 320 threads (10 warps); each thread loads and stores two
    16-byte vectors of x, residual and weight (cp.async into shared memory), stages
    the fp32 `h` row in shared memory (20 KB), and warps 0-1 replay the stock
    64-thread sum. Shared memory ~41 KB a CTA. Compiled once per (dtype, hidden, PDL,
    SM) on first use. PDL from `TRTLLM_ENABLE_PDL` (default on): waits before its
    loads, triggers after its stores.

    Preconditions, enforced loudly (AssertionError before launch or at compile):
    contiguous 2-D `x` and `residual` of one shape, contiguous `weight`, and a hidden
    size whose 8-wide vector count is divisible by 320 and by 64 (5120: 640 vectors).

    Numerics: bitwise the stock gemma add+norm on `x` and the residual (measured:
    every row count 1..32 and 40, 48, 56, 64, 80, 96, 112, 128 under CUDA graphs; 33,
    127, 129, 2048 and 8192 eagerly); against the fp32 reference, the stock norm's
    band.

    Certified: hidden 5120 bf16 at 16, 32, 33, 113 and 128 rows. The kernel serves
    any row count (one CTA a row), but below 16 and above 128 rows the stock grid is
    faster and the target does not call it there. Not certified: fp16 / fp32, other
    widths, strided rows.
    """

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes `x` (the normalized output) and `residual` (the sum) in place and
        returns both, in that order."""
        gemma_fused_add_rmsnorm_1row(x, residual, weight, eps, get_env_enable_pdl())
        return x, residual

    def reference(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(x_out, residual_out)`: the stock gemma add+norm's fp32 reference."""
        return flashinfer_gemma_fused_add_rmsnorm.reference(x, residual, weight, eps)

    def is_valid(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> None:
        # The op writes both tensors: one buffer passed twice ends up holding the
        # normalized output, and the returned residual is not the sum.
        assert x.data_ptr() != residual.data_ptr(), (
            "x and residual must be distinct buffers; the op writes both"
        )


gemma_add_rmsnorm_1row = GemmaAddRmsnorm1Row()
