# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-order summation of rank-major collective buffers."""

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

# Upper bound on the gathered [tp_size, chunk] buffer per collective call.
# 16 MiB keeps the transient allocation small for long prefills (a 16k-token,
# hidden 8192 BF16 activation at TP8 would otherwise gather 2 GiB at once)
# while leaving decode-sized messages in a single call.
DETERMINISTIC_ALLREDUCE_CHUNK_BYTES = 16 << 20


@triton.jit
def _sum_rank_ordered_kernel(X, Y, N, RANKS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    accumulator = tl.full((BLOCK,), 0, tl.float32)
    for rank in range(RANKS):
        value = tl.load(X + rank * N.to(tl.int64) + offsets, offsets < N, other=0).to(tl.float32)
        accumulator = accumulator + value
    tl.store(Y + offsets, accumulator, offsets < N)


def allreduce_rank_ordered(
    input: torch.Tensor,
    mapping: "Mapping",
    chunk_bytes: int | None = None,
) -> torch.Tensor:
    """All-reduce `input` over the TP group with a fixed, rank-ordered sum.

    Each rank's contribution to an element is added in rank order in FP32,
    independent of the message size, the collective algorithm and the chunk
    size: chunking only bounds the transient gathered buffer, it never
    changes which additions an element sees. The returned tensor has the
    shape and dtype of `input`.
    """
    from .ops import allgather

    flat = input.reshape(-1)
    if chunk_bytes is None:
        chunk_bytes = DETERMINISTIC_ALLREDUCE_CHUNK_BYTES
    chunk = max(1, chunk_bytes // (flat.element_size() * mapping.tp_size))
    if flat.numel() <= chunk:
        gathered = allgather(flat.reshape(1, -1), mapping, dim=0)
        return sum_rank_ordered(gathered).view(input.shape)
    output = torch.empty_like(flat)
    for start in range(0, flat.numel(), chunk):
        piece = flat[start : start + chunk]
        gathered = allgather(piece.reshape(1, -1), mapping, dim=0)
        output[start : start + piece.numel()] = sum_rank_ordered(gathered)
    return output.view(input.shape)


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
