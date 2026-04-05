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

"""Triton kernel for complex-multiplication RoPE (Rotary Position Embedding).

This implements the interleaved complex RoPE variant where frequencies are provided
as a single complex-valued tensor ``freqs_cis`` of shape ``[B, S, D//2]``.

The mathematical operation is:
    xq_ = view_as_complex(xq.reshape(..., -1, 2))   # pairs -> complex
    xq_out = view_as_real(xq_ * freqs_cis).flatten() # complex mul -> pairs
    # equivalently: (a, b) * (cos, sin) = (a*cos - b*sin, a*sin + b*cos)
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def rope_complex_kernel(
    # Pointers to input/output tensors
    x_ptr,
    freqs_real_ptr,
    freqs_imag_ptr,
    out_ptr,
    # Strides for x (and out, which has the same layout)
    stride_x_row,
    # Strides for freqs (after unsqueeze, effectively [B, S, 1, D//2])
    # We pass the stride for the "row" dimension of freqs (B*S flattened)
    stride_f_row,
    # Problem sizes
    N_HEADS: tl.constexpr,  # number of heads
    D_HALF: tl.constexpr,  # D // 2 (number of complex pairs)
    BLOCK_D: tl.constexpr,  # block size for D_HALF dimension (power of 2)
):
    """Triton kernel for complex-multiplication RoPE.

    Grid: (num_rows, N_HEADS) where num_rows = B * S.
    Each program handles one (row, head) combination, applying the complex
    rotation to D_HALF pairs of elements.
    """
    row_idx = tl.program_id(0)  # which (batch, seq) position
    head_idx = tl.program_id(1)  # which head

    # Offsets within the D_HALF dimension
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D_HALF

    # Compute base pointer for this row and head in x
    # x layout after flatten to 2D: [B*S, N*D] with stride_x_row = N*D
    # Within a row: head_idx * D elements, then pairs at 2*d_offsets and 2*d_offsets+1
    x_base = x_ptr + row_idx * stride_x_row + head_idx * (D_HALF * 2)

    # Load the real and imaginary parts of input (interleaved pairs)
    a = tl.load(x_base + d_offsets * 2, mask=d_mask, other=0.0).to(tl.float32)
    b = tl.load(x_base + d_offsets * 2 + 1, mask=d_mask, other=0.0).to(tl.float32)

    # Load freqs for this row (shared across heads)
    # freqs layout: [B*S, D//2] for real and imag separately
    f_base = row_idx * stride_f_row
    cos_val = tl.load(freqs_real_ptr + f_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    sin_val = tl.load(freqs_imag_ptr + f_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    # Complex multiplication: (a + bi) * (cos + sin*i) = (a*cos - b*sin) + (a*sin + b*cos)*i
    out_real = a * cos_val - b * sin_val
    out_imag = a * sin_val + b * cos_val

    # Store back as interleaved pairs, cast back to input dtype
    out_base = out_ptr + row_idx * stride_x_row + head_idx * (D_HALF * 2)
    tl.store(out_base + d_offsets * 2, out_real, mask=d_mask)
    tl.store(out_base + d_offsets * 2 + 1, out_imag, mask=d_mask)


def rope_with_complex_freqs(
    xq: Tensor,
    xk: Tensor,
    freqs_cis: Tensor,
    unsqueeze_dim: int = 2,
) -> Tuple[Tensor, Tensor]:
    """Python launcher for the complex-multiplication RoPE Triton kernel.

    Args:
        xq: Query tensor of shape [B, S, N, D] (when unsqueeze_dim=2) or
            [B, N, S, D] (when unsqueeze_dim=1).
        xk: Key tensor, same layout as xq (may have different N for GQA).
        freqs_cis: Complex frequency tensor of shape [B, S, D//2].
        unsqueeze_dim: Dimension along which to unsqueeze freqs_cis to broadcast
            over heads. 2 for BSND layout, 1 for BNSD layout.

    Returns:
        Tuple of rotated (xq_out, xk_out) with the same shapes and dtype as inputs.
    """
    # Determine layout
    if unsqueeze_dim == 2:
        # BSND layout: [B, S, N, D]
        B, S = xq.shape[0], xq.shape[1]
        Nq, D = xq.shape[2], xq.shape[3]
        Nk = xk.shape[2]
    else:
        # BNSD layout: [B, N, S, D]
        B, S = xq.shape[0], xq.shape[2]
        Nq, D = xq.shape[1], xq.shape[3]
        Nk = xk.shape[1]

    D_HALF = D // 2
    assert D % 2 == 0, "RoPE requires an even head dimension."

    # Ensure contiguous for correct stride arithmetic
    xq = xq.contiguous()
    xk = xk.contiguous()

    # Reshape to [B*S, N, D] for uniform processing
    if unsqueeze_dim == 2:
        # BSND -> [B*S, Nq, D]
        xq_flat = xq.reshape(B * S, Nq, D)
        xk_flat = xk.reshape(B * S, Nk, D)
    else:
        # BNSD -> transpose to BSND first, then flatten
        xq_flat = xq.transpose(1, 2).contiguous().reshape(B * S, Nq, D)
        xk_flat = xk.transpose(1, 2).contiguous().reshape(B * S, Nk, D)

    # Decompose complex freqs_cis into real and imaginary parts
    # freqs_cis shape: [B, S, D//2] complex
    freqs_real = freqs_cis.real.contiguous().reshape(B * S, D_HALF)
    freqs_imag = freqs_cis.imag.contiguous().reshape(B * S, D_HALF)

    # Allocate outputs
    xq_out_flat = torch.empty_like(xq_flat)
    xk_out_flat = torch.empty_like(xk_flat)

    # Block size for D_HALF dimension
    BLOCK_D = triton.next_power_of_2(D_HALF)

    num_rows = B * S
    stride_x_row_q = xq_flat.stride(0)  # N * D
    stride_x_row_k = xk_flat.stride(0)
    stride_f_row = freqs_real.stride(0)  # D_HALF

    # Launch kernel for queries
    grid_q = (num_rows, Nq)
    rope_complex_kernel[grid_q](
        xq_flat,
        freqs_real,
        freqs_imag,
        xq_out_flat,
        stride_x_row=stride_x_row_q,
        stride_f_row=stride_f_row,
        N_HEADS=Nq,
        D_HALF=D_HALF,
        BLOCK_D=BLOCK_D,
        num_warps=min(4, max(1, BLOCK_D // 32)),
        num_stages=3,
    )

    # Launch kernel for keys
    grid_k = (num_rows, Nk)
    rope_complex_kernel[grid_k](
        xk_flat,
        freqs_real,
        freqs_imag,
        xk_out_flat,
        stride_x_row=stride_x_row_k,
        stride_f_row=stride_f_row,
        N_HEADS=Nk,
        D_HALF=D_HALF,
        BLOCK_D=BLOCK_D,
        num_warps=min(4, max(1, BLOCK_D // 32)),
        num_stages=3,
    )

    # Reshape outputs back to original layout and cast to input dtype
    if unsqueeze_dim == 2:
        xq_out = xq_out_flat.reshape(B, S, Nq, D).to(xq.dtype)
        xk_out = xk_out_flat.reshape(B, S, Nk, D).to(xk.dtype)
    else:
        # Reshape to BSND then transpose back to BNSD
        xq_out = xq_out_flat.reshape(B, S, Nq, D).transpose(1, 2).contiguous().to(xq.dtype)
        xk_out = xk_out_flat.reshape(B, S, Nk, D).transpose(1, 2).contiguous().to(xk.dtype)

    return xq_out, xk_out
