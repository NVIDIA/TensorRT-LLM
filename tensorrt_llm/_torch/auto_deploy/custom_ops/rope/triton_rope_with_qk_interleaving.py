# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton kernel for DeepSeek-style interleaved RoPE (torch_rope_with_qk_interleaving).

The interleaved RoPE first permutes q/k channels so that even indices come first
and odd indices second, then applies the standard rotate_half rotation using the
provided cos/sin tensors. Mathematically, for pair index i in [0, D/2):

    a = x[..., 2*i]     (even-indexed original element)
    b = x[..., 2*i + 1] (odd-indexed original element)

    out[..., i]       = a * cos[..., i] - b * sin[..., i]
    out[..., D/2 + i] = b * cos[..., D/2 + i] + a * sin[..., D/2 + i]

Since cos and sin are typically duplicated ([cos_half, cos_half]), cos[i] == cos[D/2+i],
so both halves use the same rotation angle but with different signs.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rope_with_qk_interleaving_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    stride_x_row: tl.constexpr,
    stride_cos_row: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    """Triton kernel for interleaved RoPE.

    Grid: (num_rows,) where num_rows = product of all dims except the last.
    Each program handles one row of D elements.
    """
    row_idx = tl.program_id(0)

    D_HALF: tl.constexpr = D // 2
    BLOCK_D_HALF: tl.constexpr = BLOCK_D // 2

    pair_offsets = tl.arange(0, BLOCK_D_HALF)
    pair_mask = pair_offsets < D_HALF

    # Load even and odd elements from x: x[..., 2*i] and x[..., 2*i+1]
    x_row_ptr = x_ptr + row_idx * stride_x_row
    a = tl.load(x_row_ptr + pair_offsets * 2, mask=pair_mask, other=0.0).to(tl.float32)
    b = tl.load(x_row_ptr + pair_offsets * 2 + 1, mask=pair_mask, other=0.0).to(tl.float32)

    # Load cos and sin values for the first half (indices 0..D/2-1)
    cos_row_ptr = cos_ptr + row_idx * stride_cos_row
    sin_row_ptr = sin_ptr + row_idx * stride_cos_row
    cos_first = tl.load(cos_row_ptr + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
    sin_first = tl.load(sin_row_ptr + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)

    # Load cos and sin values for the second half (indices D/2..D-1)
    second_half_offsets = pair_offsets + D_HALF
    second_half_mask = second_half_offsets < D
    cos_second = tl.load(
        cos_row_ptr + second_half_offsets, mask=second_half_mask, other=0.0
    ).to(tl.float32)
    sin_second = tl.load(
        sin_row_ptr + second_half_offsets, mask=second_half_mask, other=0.0
    ).to(tl.float32)

    # Compute output first half: out[i] = a*cos[i] - b*sin[i]
    out_first = a * cos_first - b * sin_first

    # Compute output second half: out[D/2+i] = b*cos[D/2+i] + a*sin[D/2+i]
    out_second = b * cos_second + a * sin_second

    # Store results cast back to input dtype
    out_row_ptr = output_ptr + row_idx * stride_x_row
    tl.store(out_row_ptr + pair_offsets, out_first.to(INPUT_DTYPE), mask=pair_mask)
    tl.store(
        out_row_ptr + second_half_offsets,
        out_second.to(INPUT_DTYPE),
        mask=second_half_mask,
    )


def rope_with_qk_interleaving(
    x: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
) -> Tensor:
    """Python launcher for the interleaved RoPE Triton kernel.

    Applies DeepSeek-style interleaved RoPE to a single tensor (q or k).

    Args:
        x: Input tensor, shape [..., D] where D is the head dimension (must be even).
        cos: Cosine frequencies, shape broadcastable to x after unsqueeze.
        sin: Sine frequencies, shape broadcastable to x after unsqueeze.
        unsqueeze_dim: Dimension along which cos/sin are unsqueezed for broadcasting.

    Returns:
        Tensor with same shape as x, with interleaved RoPE applied.
    """
    assert x.shape[-1] % 2 == 0, "Last dimension D must be even for interleaved RoPE."
    x = x.contiguous()

    # Unsqueeze cos/sin for broadcasting (same as the torch reference)
    cos = cos.unsqueeze(unsqueeze_dim).contiguous()
    sin = sin.unsqueeze(unsqueeze_dim).contiguous()

    # Broadcast cos/sin to match x shape
    cos = cos.expand_as(x).contiguous()
    sin = sin.expand_as(x).contiguous()

    D = x.shape[-1]
    num_rows = x.numel() // D

    # Choose block size (power of 2, >= D/2 since we process pairs)
    BLOCK_D = triton.next_power_of_2(D)

    # Strides (in elements) for the row dimension
    stride_x_row = x.stride(-2) if x.ndim >= 2 else D
    stride_cos_row = cos.stride(-2) if cos.ndim >= 2 else D

    # Map input dtype to Triton constexpr
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    input_dtype = dtype_map[x.dtype]

    out = torch.empty_like(x)

    grid = (num_rows,)
    _rope_with_qk_interleaving_kernel[grid](
        x,
        cos,
        sin,
        out,
        stride_x_row=stride_x_row,
        stride_cos_row=stride_cos_row,
        D=D,
        BLOCK_D=BLOCK_D,
        INPUT_DTYPE=input_dtype,
        num_warps=4,
        num_stages=3,
    )

    return out


def rope_with_qk_interleaving_fused(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
) -> Tuple[Tensor, Tensor]:
    """Apply interleaved RoPE to both q and k tensors.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine frequencies.
        sin: Sine frequencies.
        unsqueeze_dim: Dimension for unsqueezing cos/sin.

    Returns:
        Tuple of (q_embed, k_embed) with interleaved RoPE applied.
    """
    q_embed = rope_with_qk_interleaving(q, cos, sin, unsqueeze_dim)
    k_embed = rope_with_qk_interleaving(k, cos, sin, unsqueeze_dim)
    return q_embed, k_embed
