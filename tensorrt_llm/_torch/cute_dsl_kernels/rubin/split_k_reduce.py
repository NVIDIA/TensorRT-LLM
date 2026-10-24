# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduction of split-K GEMM partials: ``out[M, N] = bf16(sum_s partial[s, M, N])``.

The Rubin dense block-scaled GEMM computes split ``s`` of a split-K tactic as batch ``s``
of an FP32 ``[S, M, N]`` partial tensor (``Sm107BlockScaledPersistentDenseGemmKernel``,
``split_k``). This kernel sums the splits in a fixed order (0, 1, ..., S-1) in FP32 and
rounds once to BF16 (round-to-nearest-even, as the GEMM epilogue does), so the result is
deterministic and differs from the unsplit GEMM only by the FP32 summation order.

With ``use_pdl`` the kernel is launched with programmatic dependent launch: it can be
scheduled while the GEMM still runs (the PDL GEMM triggers its dependents early) and waits
on ``griddepcontrol.wait`` before its first load, which returns once the GEMM grid has
completed and its partials are visible.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4


@triton.jit
def _gdc_wait_with_memory_clobber():
    # The memory clobber keeps the compiler from hoisting the partial loads above the wait.
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _split_k_reduce_bf16_kernel(
    partial_ptr,
    out_ptr,
    numel,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        _gdc_wait_with_memory_clobber()
        tl.extra.cuda.gdc_launch_dependents()
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.load(partial_ptr + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(partial_ptr + s * numel + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


def split_k_reduce_bf16(
    partial: torch.Tensor, out: torch.Tensor, use_pdl: bool = False
) -> torch.Tensor:
    """Sum the contiguous FP32 ``partial[S, M, N]`` over S into the contiguous BF16 ``out[M, N]``.

    ``use_pdl`` launches the kernel with programmatic dependent launch (see the module
    docstring); it is safe after any kernel because the kernel waits before touching memory.
    """
    if partial.dtype != torch.float32 or partial.dim() != 3 or not partial.is_contiguous():
        raise ValueError("split-K partials must be a contiguous float32 [S, M, N] tensor")
    split_k, m, n = partial.shape
    if out.dtype != torch.bfloat16 or tuple(out.shape) != (m, n) or not out.is_contiguous():
        raise ValueError(f"split-K output must be a contiguous bfloat16 [{m}, {n}] tensor")
    numel = m * n
    if numel == 0:
        return out
    grid = (triton.cdiv(numel, _BLOCK),)
    _split_k_reduce_bf16_kernel[grid](
        partial,
        out,
        numel,
        SPLIT_K=split_k,
        BLOCK=_BLOCK,
        USE_PDL=use_pdl,
        num_warps=_NUM_WARPS,
        launch_pdl=use_pdl,
    )
    return out
