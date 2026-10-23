# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-order summation of rank-major collective buffers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_rank_ordered_kernel(X, Y, N, RANKS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    accumulator = tl.full((BLOCK,), 0, tl.float32)
    for rank in range(RANKS):
        value = tl.load(X + rank * N.to(tl.int64) + offsets, offsets < N, other=0).to(tl.float32)
        accumulator = accumulator + value
    tl.store(Y + offsets, accumulator, offsets < N)


def sum_rank_ordered(gathered: torch.Tensor) -> torch.Tensor:
    """Sum a contiguous [ranks, elements] buffer in rank order using FP32.

    Inputs and output may be BF16, FP16 or FP32. Each output element uses the
    same sequence of additions regardless of the collective message size.
    """
    ranks, elements = gathered.shape
    output = torch.empty((elements,), dtype=gathered.dtype, device=gathered.device)
    if elements:
        _sum_rank_ordered_kernel[(triton.cdiv(elements, 256),)](
            gathered, output, elements, ranks, 256, num_warps=4
        )
    return output


@triton.jit
def _rms_norm_kernel(
    X,
    RESIDUAL,
    WEIGHT,
    BIAS,
    Y,
    RESIDUAL_OUT,
    H: tl.constexpr,
    EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    value = tl.load(X + row * H + cols, mask, other=0)
    if HAS_BIAS:
        bias = tl.load(BIAS + cols, mask, other=0)
        value = (value.to(tl.float32) + bias.to(tl.float32)).to(X.dtype.element_ty)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + row * H + cols, mask, other=0)
        value = (value.to(tl.float32) + residual.to(tl.float32)).to(X.dtype.element_ty)
        tl.store(RESIDUAL_OUT + row * H + cols, value, mask)
    value_f32 = value.to(tl.float32)
    variance = tl.sum(value_f32 * value_f32, axis=0) / H
    normalized = value_f32 * tl.rsqrt(variance + EPS)
    if HAS_WEIGHT:
        normalized = normalized * tl.load(WEIGHT + cols, mask, other=0).to(tl.float32)
    tl.store(Y + row * H + cols, normalized, mask)


def rms_norm_after_allreduce(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the unquantized AllReduce epilogue without changing its rounding.

    Bias and residual additions each round to the input dtype before the
    FP32 RMS reduction. Inputs are contiguous; weights and bias have H elements.
    """
    hidden = x.shape[-1]
    output = torch.empty_like(x)
    residual_out = torch.empty_like(x) if residual is not None else x
    rows = x.numel() // hidden
    if rows:
        _rms_norm_kernel[(rows,)](
            x,
            residual,
            weight,
            bias,
            output,
            residual_out,
            hidden,
            eps,
            residual is not None,
            weight is not None,
            bias is not None,
            triton.next_power_of_2(hidden),
            num_warps=4,
        )
    return output, residual_out
