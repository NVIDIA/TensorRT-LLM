# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py
# (Apache-2.0), by way of tensorrt_llm/_torch/attention/kernels/fused_qk_norm_rope_gate.py.
"""Qwen3.5 gated full-attention prologue in one Triton kernel.

The projection produces per-head interleaved ``[q0, g0, q1, g1, ..., K, V]``. The kernel
reads that layout directly and writes packed ``[Q, K, V]`` while applying per-head Gemma
RMSNorm and NeoX interleaved-MRoPE to Q/K. The output gate is not copied: the caller
reads it as a strided view of the projection.

The arithmetic is the in-tree ``_fused_qkv_gemma_rmsnorm_rope_gate_kernel``'s, with two
differences in what it reads and writes, not in what it computes: the MRoPE section rows
are read through a runtime section stride (so a ``[3, 1, T]`` slice of a wider position
buffer is read in place), and the gate store is gone.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_gemma_rmsnorm_mrope_kernel(
    qkv_ptr,
    qkv_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    positions_ptr,
    positions_stride_section,
    max_positions,
    qkv_stride_token,
    qkv_out_stride_token,
    cos_sin_stride_position,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    head_block: tl.constexpr,
    rotary_block: tl.constexpr,
    eps: tl.constexpr,
    has_pass_through: tl.constexpr,
    mrope_section1: tl.constexpr,
    mrope_section2: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    output_dtype = tl.bfloat16

    if head < num_q_heads + num_kv_heads:
        is_k = head >= num_q_heads
        local_head = tl.where(is_k, head - num_q_heads, head)
        if is_k:
            input_base = qkv_ptr + token * qkv_stride_token + 2 * q_size + local_head * head_dim
            output_base = (
                qkv_out_ptr + token * qkv_out_stride_token + q_size + local_head * head_dim
            )
            weight_ptr = k_weight_ptr
        else:
            input_base = qkv_ptr + token * qkv_stride_token + local_head * 2 * head_dim
            output_base = qkv_out_ptr + token * qkv_out_stride_token + local_head * head_dim
            weight_ptr = q_weight_ptr

        head_offsets = tl.arange(0, head_block)
        head_mask = head_offsets < head_dim
        x = tl.load(input_base + head_offsets, mask=head_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + head_offsets, mask=head_mask, other=0.0).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(x * x, axis=0) / head_dim + eps)
        normalized = (x * inverse_rms * (weight + 1.0)).to(output_dtype).to(tl.float32)

        if has_pass_through:
            pass_mask = head_mask & (head_offsets >= rotary_dim)
            tl.store(output_base + head_offsets, normalized, mask=pass_mask)

        rotary_offsets = tl.arange(0, rotary_block)
        rotary_mask = rotary_offsets < half_rotary
        x_first = tl.load(input_base + rotary_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        x_second = tl.load(
            input_base + half_rotary + rotary_offsets, mask=rotary_mask, other=0.0
        ).to(tl.float32)
        weight_first = tl.load(weight_ptr + rotary_offsets, mask=rotary_mask, other=0.0).to(
            tl.float32
        )
        weight_second = tl.load(
            weight_ptr + half_rotary + rotary_offsets, mask=rotary_mask, other=0.0
        ).to(tl.float32)
        x_first = (x_first * inverse_rms * (weight_first + 1.0)).to(output_dtype).to(tl.float32)
        x_second = (x_second * inverse_rms * (weight_second + 1.0)).to(output_dtype).to(tl.float32)

        section = tl.where(
            (rotary_offsets % 3 == 1) & (rotary_offsets < mrope_section1 * 3),
            1,
            tl.where(
                (rotary_offsets % 3 == 2) & (rotary_offsets < mrope_section2 * 3),
                2,
                0,
            ),
        )
        position = tl.load(
            positions_ptr + section * positions_stride_section + token,
            mask=rotary_mask,
            other=0,
        ).to(tl.int64)
        position_is_valid = (position >= 0) & (position < max_positions)
        tl.device_assert(position_is_valid, "position is outside the RoPE table")
        safe_position = tl.where(position_is_valid, position, 0)
        cos_sin_base = cos_sin_ptr + safe_position * cos_sin_stride_position
        cos = tl.load(cos_sin_base + rotary_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        sin = tl.load(cos_sin_base + half_rotary + rotary_offsets, mask=rotary_mask, other=0.0).to(
            tl.float32
        )
        cos = tl.where(position_is_valid, cos, float("nan"))
        sin = tl.where(position_is_valid, sin, float("nan"))
        tl.store(
            output_base + rotary_offsets,
            x_first * cos - x_second * sin,
            mask=rotary_mask,
        )
        tl.store(
            output_base + half_rotary + rotary_offsets,
            x_second * cos + x_first * sin,
            mask=rotary_mask,
        )
    else:
        local_head = head - num_q_heads - num_kv_heads
        head_offsets = tl.arange(0, head_block)
        head_mask = head_offsets < head_dim
        input_base = (
            qkv_ptr + token * qkv_stride_token + 2 * q_size + kv_size + local_head * head_dim
        )
        output_base = (
            qkv_out_ptr + token * qkv_out_stride_token + q_size + kv_size + local_head * head_dim
        )
        value = tl.load(input_base + head_offsets, mask=head_mask, other=0.0)
        tl.store(output_base + head_offsets, value, mask=head_mask)


def qkv_gemma_rmsnorm_mrope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: Tuple[int, int, int],
) -> torch.Tensor:
    """Packed ``[T, q_size + 2 * kv_size]`` Q (normed, roped) | K (normed, roped) | V.

    ``positions`` holds ``3 * T`` MRoPE positions as ``[3, ..., T]``; a slice of a wider
    ``[3, 1, max_tokens]`` buffer is read through its section stride.
    """
    assert qkv.dim() == 2 and qkv.stride(-1) == 1
    assert qkv.dtype == torch.bfloat16
    assert q_weight.shape == (head_dim,) and k_weight.shape == (head_dim,)
    assert q_weight.device == qkv.device and k_weight.device == qkv.device
    assert cos_sin.dtype == torch.float32 and cos_sin.is_contiguous()
    assert cos_sin.device == qkv.device and positions.device == qkv.device
    assert rotary_dim > 0 and rotary_dim <= head_dim and rotary_dim % 2 == 0
    assert cos_sin.shape == (cos_sin.shape[0], 2, rotary_dim // 2)
    assert cos_sin.shape[0] > 0
    assert len(mrope_section) == 3 and sum(mrope_section) == rotary_dim // 2
    assert mrope_section[1] > 0 and mrope_section[2] > 0

    num_tokens = qkv.shape[0]
    assert positions.numel() == 3 * num_tokens
    assert positions.dtype in (torch.int32, torch.int64)
    # reshape keeps a [3, 1, max_tokens] buffer slice a strided view; the kernel reads
    # section rows through positions_stride_section.
    positions = positions.reshape(3, num_tokens)
    if num_tokens > 1 and positions.stride(-1) != 1:
        positions = positions.contiguous()
    positions_stride_section = positions.stride(0)

    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    assert qkv.shape[1] == 2 * q_size + 2 * kv_size

    qkv_out = torch.empty((num_tokens, q_size + 2 * kv_size), dtype=qkv.dtype, device=qkv.device)
    if num_tokens == 0:
        return qkv_out

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rotary_block = triton.next_power_of_2(half_rotary)
    grid = (num_tokens, num_q_heads + 2 * num_kv_heads)
    _qkv_gemma_rmsnorm_mrope_kernel[grid](
        qkv,
        qkv_out,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        positions_stride_section,
        cos_sin.shape[0],
        qkv.stride(0),
        qkv_out.stride(0),
        cos_sin.stride(0),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        half_rotary=half_rotary,
        head_block=head_block,
        rotary_block=rotary_block,
        eps=eps,
        has_pass_through=rotary_dim < head_dim,
        mrope_section1=mrope_section[1],
        mrope_section2=mrope_section[2],
        num_warps=4,
    )
    return qkv_out
