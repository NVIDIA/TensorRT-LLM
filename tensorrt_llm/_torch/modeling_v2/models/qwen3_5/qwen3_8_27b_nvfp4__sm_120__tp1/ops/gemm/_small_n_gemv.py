# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton CUDA-core GEMV for small-N, small-M BF16 projections (``y = x @ w.T``).

Each program owns ``BLOCK_N`` weight rows and ``BLOCK_M`` activation rows, issues
all of its K loads in one statically unrolled pass, accumulates the elementwise
products in FP32 registers and reduces them once with a fixed tree. The reduction
order depends only on the static launch configuration, so repeated calls, eager
execution and CUDA-graph replay are bitwise identical.

Launch: one weight row and one activation row per program, 8 warps, one BF16
element per thread per 256-wide K step -- the fastest configuration at decode rows
for the GDN in_proj_ba (N=96, K=5120) on SM120 next to in_proj_qkvz on the other
stream (more, shorter programs fill the SMs the partner leaves free).
"""

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

BLOCK_M = 1
BLOCK_N = 1
BLOCK_K = 256
NUM_WARPS = 8


@triton.jit
def _small_n_gemv_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_wn,
    stride_ym,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    x_rows = x_ptr + offs_m[:, None].to(tl.int64) * stride_xm
    w_rows = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_K), dtype=tl.float32)
    # K % BLOCK_K == 0 is guaranteed by the caller.
    for k0 in tl.static_range(0, K, BLOCK_K):
        kk = k0 + offs_k
        w = tl.load(w_rows + kk[None, :], mask=mask_n[:, None], other=0.0)
        x = tl.load(x_rows + kk[None, :], mask=mask_m[:, None], other=0.0)
        acc += x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :]
    y = tl.sum(acc, axis=2)
    y_ptrs = y_ptr + offs_m[:, None].to(tl.int64) * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def small_n_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``y = x @ weight.T`` for BF16 ``x`` ``[M, K]`` and ``weight`` ``[N, K]``, both with
    unit inner stride; a fresh contiguous ``[M, N]`` BF16 tensor."""
    m, k = x.shape
    n = weight.shape[0]
    assert k % BLOCK_K == 0, f"K={k} must be a multiple of {BLOCK_K}"
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return y
    grid = (triton.cdiv(n, BLOCK_N), triton.cdiv(m, BLOCK_M))
    _small_n_gemv_kernel[grid](
        x,
        weight,
        y,
        m,
        n,
        x.stride(0),
        weight.stride(0),
        y.stride(0),
        K=k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
    )
    return y
