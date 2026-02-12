# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fused elementwise operations for Mamba2 prefill optimization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _extract_transpose_prefill_kernel(
    src_ptr,
    dst_ptr,
    num_prefill_tokens,
    d_in_proj,
    d_inner,
    conv_dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_CONV: tl.constexpr,
):
    """Extract src[0:num_prefill_tokens, d_inner:d_inner+conv_dim] and
    transpose to dst[conv_dim, num_prefill_tokens]."""
    pid_seq = tl.program_id(0)
    pid_conv = tl.program_id(1)

    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    conv_offsets = pid_conv * BLOCK_CONV + tl.arange(0, BLOCK_CONV)

    seq_mask = seq_offsets < num_prefill_tokens
    conv_mask = conv_offsets < conv_dim
    mask = seq_mask[:, None] & conv_mask[None, :]

    src_offsets = seq_offsets[:, None] * d_in_proj + (d_inner + conv_offsets[None, :])
    data = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

    dst_offsets = conv_offsets[:, None] * num_prefill_tokens + seq_offsets[None, :]
    tl.store(dst_ptr + dst_offsets, tl.trans(data), mask=conv_mask[:, None] & seq_mask[None, :])


def extract_transpose_xbc_prefill(
    zxbcdt: torch.Tensor,
    num_prefill_tokens: int,
    d_inner: int,
    conv_dim: int,
) -> torch.Tensor:
    """
    Extract and transpose xbc slice from zxbcdt for causal_conv1d_fn.

    Input:  zxbcdt[num_tokens, d_in_proj]
    Output: [conv_dim, num_prefill_tokens]
    """
    out = torch.empty(conv_dim, num_prefill_tokens, dtype=zxbcdt.dtype, device=zxbcdt.device)

    BLOCK_SEQ, BLOCK_CONV = 32, 128
    grid = (triton.cdiv(num_prefill_tokens, BLOCK_SEQ), triton.cdiv(conv_dim, BLOCK_CONV))

    _extract_transpose_prefill_kernel[grid](
        zxbcdt,
        out,
        num_prefill_tokens,
        zxbcdt.shape[1],
        d_inner,
        conv_dim,
        BLOCK_SEQ,
        BLOCK_CONV,
    )
    return out


@triton.jit
def _fused_conv_output_transpose_kernel(
    src_ptr,
    out_x_ptr,
    out_B_ptr,
    out_C_ptr,
    num_prefill_tokens,
    d_inner,
    bc_size,
    x_tiles,
    bc_tiles,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Transpose and split conv1d output into x, B, C using linear grid mapping.

    Grid: tiles [0, x_tiles) -> x, [x_tiles, x_tiles+bc_tiles) -> B, rest -> C
    """
    tile_id = tl.program_id(0)

    is_x = tile_id < x_tiles
    is_B = (tile_id >= x_tiles) & (tile_id < x_tiles + bc_tiles)

    local_tile = tl.where(
        is_x, tile_id, tl.where(is_B, tile_id - x_tiles, tile_id - x_tiles - bc_tiles)
    )
    dim_size = tl.where(is_x, d_inner, bc_size)
    num_dim_blocks = tl.cdiv(dim_size, BLOCK_DIM)

    pid_seq = local_tile // num_dim_blocks
    pid_dim = local_tile % num_dim_blocks

    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dim_offsets = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    seq_mask = seq_offsets < num_prefill_tokens
    dim_mask = dim_offsets < dim_size

    src_offset = tl.where(is_x, 0, tl.where(is_B, d_inner, d_inner + bc_size))
    src_indices = (src_offset + dim_offsets[:, None]) * num_prefill_tokens + seq_offsets[None, :]
    data = tl.load(src_ptr + src_indices, mask=dim_mask[:, None] & seq_mask[None, :], other=0.0)

    out_ptr = tl.where(is_x, out_x_ptr, tl.where(is_B, out_B_ptr, out_C_ptr))
    dst_indices = seq_offsets[:, None] * dim_size + dim_offsets[None, :]
    tl.store(out_ptr + dst_indices, tl.trans(data), mask=seq_mask[:, None] & dim_mask[None, :])


def fused_split_rearrange_after_conv1d(
    xbc: torch.Tensor,
    d_inner: int,
    n_groups: int,
    d_state: int,
    nheads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split and rearrange causal_conv1d output into contiguous x, B, C tensors.

    Input:  xbc[conv_dim, num_prefill_tokens]
    Output: x[1, num_prefill_tokens, nheads, head_dim],
            B[1, num_prefill_tokens, n_groups, d_state],
            C[1, num_prefill_tokens, n_groups, d_state]
    """
    conv_dim, num_prefill_tokens = xbc.shape
    bc_size = n_groups * d_state

    x_flat = torch.empty(num_prefill_tokens, d_inner, dtype=xbc.dtype, device=xbc.device)
    B_flat = torch.empty(num_prefill_tokens, bc_size, dtype=xbc.dtype, device=xbc.device)
    C_flat = torch.empty(num_prefill_tokens, bc_size, dtype=xbc.dtype, device=xbc.device)

    BLOCK_SEQ, BLOCK_DIM = 64, 64
    num_seq_blocks = triton.cdiv(num_prefill_tokens, BLOCK_SEQ)
    x_tiles = num_seq_blocks * triton.cdiv(d_inner, BLOCK_DIM)
    bc_tiles = num_seq_blocks * triton.cdiv(bc_size, BLOCK_DIM)

    _fused_conv_output_transpose_kernel[(x_tiles + 2 * bc_tiles,)](
        xbc,
        x_flat,
        B_flat,
        C_flat,
        num_prefill_tokens,
        d_inner,
        bc_size,
        x_tiles,
        bc_tiles,
        BLOCK_SEQ,
        BLOCK_DIM,
    )

    return (
        x_flat.view(1, num_prefill_tokens, nheads, head_dim),
        B_flat.view(1, num_prefill_tokens, n_groups, d_state),
        C_flat.view(1, num_prefill_tokens, n_groups, d_state),
    )
