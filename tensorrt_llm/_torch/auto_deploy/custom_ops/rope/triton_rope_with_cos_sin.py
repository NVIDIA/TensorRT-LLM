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

"""Triton kernel and launcher for HF-style RoPE with explicit cos/sin tensors.

Computes:
    rotate_half(x) = cat(-x[..., D//2:], x[..., :D//2])
    output = x * cos + rotate_half(x) * sin

for both q and k tensors in a single fused kernel launch (two output tensors).
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rope_cos_sin_kernel(
    # Input tensor pointer (q or k)
    x_ptr,
    # cos and sin pointers (already unsqueezed and broadcast-ready)
    cos_ptr,
    sin_ptr,
    # Output pointer
    out_ptr,
    # Strides for x/out (same layout)
    stride_x_row: tl.constexpr,
    # Strides for cos/sin
    stride_cs_row: tl.constexpr,
    # Problem dimensions
    HALF_D: tl.constexpr,  # D // 2
    BLOCK_SIZE: tl.constexpr,  # >= HALF_D, power of 2
):
    """Triton kernel for RoPE with explicit cos/sin.

    Grid: (num_rows,) where num_rows = total elements / D.
    Each program processes one "row" of dimension D = 2 * HALF_D.

    The rotate_half operation splits x into [x1, x2] each of size HALF_D,
    then computes:
        out[..., :HALF_D]  = x1 * cos[..., :HALF_D] + (-x2) * sin[..., :HALF_D]
        out[..., HALF_D:]  = x2 * cos[..., HALF_D:] + x1 * sin[..., HALF_D:]

    Since cos and sin are duplicated (cos = [c, c], sin = [s, s] where c and s
    are each of size HALF_D), this simplifies to loading only HALF_D values of
    cos/sin.
    """
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < HALF_D

    # Pointers for the first half and second half of x
    x_row_ptr = x_ptr + row_idx * stride_x_row
    x1 = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)
    x2 = tl.load(x_row_ptr + HALF_D + col_offsets, mask=mask, other=0.0)

    # Load cos and sin (only need HALF_D values since they are duplicated)
    cs_row_ptr_base = cos_ptr + row_idx * stride_cs_row
    c = tl.load(cs_row_ptr_base + col_offsets, mask=mask, other=0.0)

    ss_row_ptr_base = sin_ptr + row_idx * stride_cs_row
    s = tl.load(ss_row_ptr_base + col_offsets, mask=mask, other=0.0)

    # Upcast for numerical stability
    x1f = x1.to(tl.float32)
    x2f = x2.to(tl.float32)
    cf = c.to(tl.float32)
    sf = s.to(tl.float32)

    # rotate_half: [-x2, x1]
    # out_first_half  = x1 * cos - x2 * sin
    # out_second_half = x2 * cos + x1 * sin
    y1 = x1f * cf - x2f * sf
    y2 = x2f * cf + x1f * sf

    # Store results, casting back to input dtype
    out_row_ptr = out_ptr + row_idx * stride_x_row
    tl.store(out_row_ptr + col_offsets, y1.to(x1.dtype), mask=mask)
    tl.store(out_row_ptr + HALF_D + col_offsets, y2.to(x2.dtype), mask=mask)


def rope_with_cos_sin(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Python launcher for the Triton RoPE kernel with explicit cos/sin.

    Args:
        q: Query tensor, e.g. [B, N, S, D] or [B, S, N, D].
        k: Key tensor, same layout as q.
        cos: Cosine tensor of shape [B, S, D], to be unsqueezed at unsqueeze_dim.
        sin: Sine tensor of shape [B, S, D], to be unsqueezed at unsqueeze_dim.
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting.

    Returns:
        Tuple of (q_embed, k_embed) with the same shape and dtype as q, k.
    """
    # Ensure contiguous for pointer arithmetic
    q = q.contiguous()
    k = k.contiguous()

    # Cast cos/sin to match q dtype (matching the torch reference behavior)
    cos = cos.to(dtype=q.dtype).contiguous()
    sin = sin.to(dtype=q.dtype).contiguous()

    # Unsqueeze cos/sin to broadcast with q/k
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    D = q.shape[-1]
    assert D % 2 == 0, "RoPE requires even head dimension"
    HALF_D = D // 2

    BLOCK_SIZE = triton.next_power_of_2(HALF_D)

    # Expand cos/sin separately for q and k (they may have different num_heads in GQA)
    cos_q = cos.expand_as(q).contiguous()
    sin_q = sin.expand_as(q).contiguous()

    # Flatten all dimensions except the last one into "rows"
    num_rows_q = q.numel() // D

    # Allocate outputs
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    # Launch for q
    grid_q = (num_rows_q,)
    _rope_cos_sin_kernel[grid_q](
        q,
        cos_q,
        sin_q,
        q_out,
        stride_x_row=D,
        stride_cs_row=D,
        HALF_D=HALF_D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=3,
    )

    # Expand cos/sin for k shape (may differ from q in GQA)
    cos_k = cos.expand_as(k).contiguous()
    sin_k = sin.expand_as(k).contiguous()

    num_rows_k = k.numel() // D

    # Launch for k
    grid_k = (num_rows_k,)
    _rope_cos_sin_kernel[grid_k](
        k,
        cos_k,
        sin_k,
        k_out,
        stride_x_row=D,
        stride_cs_row=D,
        HALF_D=HALF_D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=3,
    )

    return q_out, k_out
