# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 split-half RoPE with eager BF16 rounding boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _minimax_h3_rope_kernel(
    hidden_states_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    NUM_ELEMENTS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    stride_batch: tl.constexpr,
    stride_sequence: tl.constexpr,
    stride_head: tl.constexpr,
    stride_dim: tl.constexpr,
    cos_stride_sequence: tl.constexpr,
    cos_stride_dim: tl.constexpr,
    sin_stride_sequence: tl.constexpr,
    sin_stride_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    d = i % HEAD_DIM
    h = i // HEAD_DIM % NUM_HEADS
    s = i // (HEAD_DIM * NUM_HEADS) % SEQ_LEN
    b = i // (HEAD_DIM * NUM_HEADS * SEQ_LEN)
    x = tl.load(
        hidden_states_ptr
        + b * stride_batch
        + s * stride_sequence
        + h * stride_head
        + d * stride_dim,
        i < NUM_ELEMENTS,
        0,
    ).to(tl.float32)
    partner = tl.where(d < ROTARY_DIM // 2, d + ROTARY_DIM // 2, d - ROTARY_DIM // 2)
    xp = tl.load(
        hidden_states_ptr
        + b * stride_batch
        + s * stride_sequence
        + h * stride_head
        + partner * stride_dim,
        (i < NUM_ELEMENTS) & (d < ROTARY_DIM),
        0,
    ).to(tl.float32)
    rotated = tl.where(d < ROTARY_DIM // 2, -xp, xp)
    c = tl.load(
        cos_ptr + s * cos_stride_sequence + d * cos_stride_dim,
        (i < NUM_ELEMENTS) & (d < ROTARY_DIM),
        0,
    )
    sn = tl.load(
        sin_ptr + s * sin_stride_sequence + d * sin_stride_dim,
        (i < NUM_ELEMENTS) & (d < ROTARY_DIM),
        0,
    )
    # Eager casts the tables and materializes both products in BF16 before
    # adding them. Keeping these boundaries prevents output drift from fusion.
    c = c.to(tl.bfloat16).to(tl.float32)
    sn = sn.to(tl.bfloat16).to(tl.float32)
    left = (x * c).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sn).to(tl.bfloat16).to(tl.float32)
    y = tl.where(d < ROTARY_DIM, left + right, x)
    tl.store(output_ptr + i, y, i < NUM_ELEMENTS)


def apply_minimax_h3_rope_bf16(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply partial split-half RoPE with eager BF16 rounding.

    Args:
        hidden_states: CUDA BF16 tensor shaped ``[batch, sequence, heads, dim]``.
        cos: FP32 or BF16 table shaped ``[sequence, rotary_dim]`` on the same device.
        sin: FP32 or BF16 table with the same shape/device as cos.

    Returns:
        Contiguous BF16 output with the unrotated tail copied unchanged.
        Arbitrary input/table strides are supported. Tables and both products
        round to BF16 before addition; multiply-add contraction is disabled.

    Gradient-bearing inputs are rejected when autograd is enabled; the model
    dispatcher retains eager execution for these inputs.
    """
    if hidden_states.ndim != 4:
        raise ValueError("H3 fused RoPE expects [batch, sequence, heads, dim]")
    if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
        raise ValueError("H3 fused RoPE requires CUDA BF16 hidden states")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != hidden_states.shape[1]:
        raise ValueError("H3 fused RoPE tables must have shape [sequence, rotary_dim]")
    if cos.shape[1] > hidden_states.shape[-1] or cos.shape[1] % 2:
        raise ValueError("H3 rotary dimension must be even and no larger than the head dimension")
    for table in (cos, sin):
        if table.device != hidden_states.device or table.dtype not in (
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError("H3 RoPE tables must be FP32 or BF16 on the hidden-state device")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (hidden_states, cos, sin)):
        raise ValueError("H3 fused RoPE does not support autograd")
    _, seq, heads, dim = hidden_states.shape
    out = torch.empty(hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)
    if out.numel() == 0:
        return out
    with torch.cuda.device(hidden_states.device):
        _minimax_h3_rope_kernel[(triton.cdiv(out.numel(), 1024),)](
            hidden_states,
            cos,
            sin,
            out,
            out.numel(),
            seq,
            heads,
            dim,
            cos.shape[-1],
            *hidden_states.stride(),
            *cos.stride(),
            *sin.stride(),
            BLOCK_SIZE=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _minimax_h3_qk_norm_rope_kernel(
    hidden_states_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    NUM_ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    stride_batch: tl.constexpr,
    stride_sequence: tl.constexpr,
    stride_head: tl.constexpr,
    stride_dim: tl.constexpr,
    cos_stride_sequence: tl.constexpr,
    cos_stride_dim: tl.constexpr,
    sin_stride_sequence: tl.constexpr,
    sin_stride_dim: tl.constexpr,
    eps: tl.constexpr,
    TORCH_REDUCTION_ORDER: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    d = tl.arange(0, BLOCK_DIM)
    h = row % NUM_HEADS
    s = row // NUM_HEADS % SEQ_LEN
    b = row // (NUM_HEADS * SEQ_LEN)
    row_base = b * stride_batch + s * stride_sequence + h * stride_head
    row_ok = row < NUM_ROWS
    dim_ok = d < HEAD_DIM
    mask = row_ok[:, None] & dim_ok[None, :]
    x = tl.load(hidden_states_ptr + row_base[:, None] + d[None, :] * stride_dim, mask, 0).to(
        tl.float32
    )

    if TORCH_REDUCTION_ORDER:
        # Reproduce torch's CUDA reduction over a contiguous 128-wide row: each
        # of 32 lanes squares and sums its four contiguous elements in order,
        # then a shuffle-down tree combines lanes at offsets 16, 8, 4, 2, 1.
        # tl.split peels exact elements, so squaring first changes nothing.
        squares = tl.reshape(x * x, [ROWS_PER_PROGRAM, 32, 2, 2])
        even, odd = tl.split(squares)  # elements (0, 2) and (1, 3) of each lane
        e0, e2 = tl.split(even)
        e1, e3 = tl.split(odd)
        partial = ((e0 + e1) + e2) + e3
        partial = tl.sum(tl.reshape(partial, [ROWS_PER_PROGRAM, 2, 16]), axis=1)
        partial = tl.sum(tl.reshape(partial, [ROWS_PER_PROGRAM, 2, 8]), axis=1)
        partial = tl.sum(tl.reshape(partial, [ROWS_PER_PROGRAM, 2, 4]), axis=1)
        partial = tl.sum(tl.reshape(partial, [ROWS_PER_PROGRAM, 2, 2]), axis=1)
        sum_sq = tl.sum(partial, axis=1)
    else:
        sum_sq = tl.sum(x * x, axis=1)
    variance = sum_sq / HEAD_DIM
    inv_rms = tl.rsqrt(variance + eps)
    # Eager RMSNormTPAware rounds x * rsqrt to BF16, then multiplies by the
    # BF16 weight and rounds again. Keep both boundaries.
    normalized = (x * inv_rms[:, None]).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(weight_ptr + d, dim_ok, 0).to(tl.float32)
    normalized = (weight[None, :] * normalized).to(tl.bfloat16).to(tl.float32)

    partner = tl.where(d < ROTARY_DIM // 2, d + ROTARY_DIM // 2, d - ROTARY_DIM // 2)
    partner = tl.where(d < ROTARY_DIM, partner, d)
    swapped = tl.gather(
        normalized, tl.broadcast_to(partner[None, :], [ROWS_PER_PROGRAM, BLOCK_DIM]), 1
    )
    rotated = tl.where(d[None, :] < ROTARY_DIM // 2, -swapped, swapped)
    rot_mask = row_ok[:, None] & (d[None, :] < ROTARY_DIM)
    c = tl.load(
        cos_ptr + s[:, None] * cos_stride_sequence + d[None, :] * cos_stride_dim, rot_mask, 0
    )
    sn = tl.load(
        sin_ptr + s[:, None] * sin_stride_sequence + d[None, :] * sin_stride_dim, rot_mask, 0
    )
    c = c.to(tl.bfloat16).to(tl.float32)
    sn = sn.to(tl.bfloat16).to(tl.float32)
    left = (normalized * c).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sn).to(tl.bfloat16).to(tl.float32)
    y = tl.where(d[None, :] < ROTARY_DIM, left + right, normalized)
    tl.store(output_ptr + row[:, None] * HEAD_DIM + d[None, :], y, mask)


def apply_minimax_h3_qk_norm_rope_bf16(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    torch_reduction_order: bool = True,
    rows_per_program: int = 4,
    num_warps: int = 1,
) -> torch.Tensor:
    """Apply per-head RMSNorm followed by partial split-half RoPE in one kernel.

    Reproduces ``RMSNormTPAware`` (FP32 variance, ``x * rsqrt`` rounded to BF16,
    BF16 weight multiply rounded to BF16) followed by
    :func:`apply_minimax_h3_rope_bf16`. ``weight`` is a BF16 ``[dim]`` tensor.
    ``torch_reduction_order`` reproduces torch's CUDA reduction order for the
    variance of a 128-wide row, which makes the result bit-identical to the
    eager module; a plain block sum flips about one BF16 rounding per 3e7
    elements. Launch parameters default to the fastest measured configuration
    on B200 for ``[1, 20400, 56, 128]``.
    """
    if hidden_states.ndim != 4:
        raise ValueError("H3 fused QK-norm RoPE expects [batch, sequence, heads, dim]")
    if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
        raise ValueError("H3 fused QK-norm RoPE requires CUDA BF16 hidden states")
    dim = hidden_states.shape[-1]
    if (
        weight.shape != (dim,)
        or weight.dtype != torch.bfloat16
        or weight.device != hidden_states.device
    ):
        raise ValueError("H3 QK-norm weight must be a BF16 [dim] tensor on the hidden-state device")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != hidden_states.shape[1]:
        raise ValueError("H3 fused RoPE tables must have shape [sequence, rotary_dim]")
    if cos.shape[1] > dim or cos.shape[1] % 2:
        raise ValueError("H3 rotary dimension must be even and no larger than the head dimension")
    for table in (cos, sin):
        if table.device != hidden_states.device or table.dtype not in (
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError("H3 RoPE tables must be FP32 or BF16 on the hidden-state device")
    if torch_reduction_order and dim != 128:
        raise ValueError("torch reduction order is implemented for head dimension 128 only")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (hidden_states, weight, cos, sin)):
        raise ValueError("H3 fused QK-norm RoPE does not support autograd")
    batch, seq, heads, _ = hidden_states.shape
    out = torch.empty(hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)
    rows = batch * seq * heads
    if rows == 0:
        return out
    with torch.cuda.device(hidden_states.device):
        _minimax_h3_qk_norm_rope_kernel[(triton.cdiv(rows, rows_per_program),)](
            hidden_states,
            weight,
            cos,
            sin,
            out,
            rows,
            seq,
            heads,
            dim,
            cos.shape[-1],
            *hidden_states.stride(),
            *cos.stride(),
            *sin.stride(),
            float(eps),
            bool(torch_reduction_order),
            rows_per_program,
            triton.next_power_of_2(dim),
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    return out
