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
