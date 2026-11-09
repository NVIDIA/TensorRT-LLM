# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 linear with a fixed K reduction, independent of the token count."""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(
    X,
    W,
    Y,
    BIAS,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SXM: tl.constexpr,
    SXK: tl.constexpr,
    SWN: tl.constexpr,
    SWK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    cols = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        inner = block * BK + k
        a = tl.load(
            X + rows[:, None] * SXM + inner[None, :] * SXK,
            (rows[:, None] < M) & (inner[None, :] < K),
            other=0,
        )
        b = tl.load(
            W + inner[:, None] * SWK + cols[None, :] * SWN,
            (inner[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    if HAS_BIAS:
        acc += tl.load(BIAS + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


def deterministic_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """BF16 linear with FP32 accumulation and a fixed 128-element K tile.

    Output tiles may vary with M; the reduction tile and accumulation order
    are independent of M. Bias is added before rounding to output_dtype
    (the input dtype by default). Routers can retain FP32 accumulation.
    """
    shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    m, k = x.shape
    n = weight.shape[0]
    bm, bn = (16, 64) if m < 32 else ((64, 64) if m < 256 else (128, 128))
    output = torch.empty((m, n), device=x.device, dtype=output_dtype or x.dtype)
    if m and n:
        _linear_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            x,
            weight,
            output,
            bias,
            m,
            n,
            k,
            *x.stride(),
            *weight.stride(),
            bm,
            bn,
            128,
            bias is not None,
            num_warps=4,
            num_stages=3,
        )
    return output.view(*shape, n)
