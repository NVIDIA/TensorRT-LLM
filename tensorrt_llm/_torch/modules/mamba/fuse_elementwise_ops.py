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

    # Cast to int64 to avoid overflow: seq_offsets * d_in_proj can exceed INT32_MAX
    # (e.g., 131071 * 22656 = 2,969,544,576 > 2,147,483,647)
    src_offsets = seq_offsets[:, None].to(tl.int64) * d_in_proj + d_inner + conv_offsets[None, :]
    data = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

    dst_offsets = conv_offsets[:, None] * num_prefill_tokens + seq_offsets[None, :]
    tl.store(dst_ptr + dst_offsets, tl.trans(data), mask=conv_mask[:, None] & seq_mask[None, :])


def extract_transpose_prefill_slice(
    src: torch.Tensor,
    num_prefill_tokens: int,
    start_col: int,
    width: int,
) -> torch.Tensor:
    """
    Extract and transpose a contiguous prefill slice for causal_conv1d_fn.

    Input:  src[num_tokens, num_cols]
    Output: [width, num_prefill_tokens]
    """
    out = torch.empty(width, num_prefill_tokens, dtype=src.dtype, device=src.device)

    BLOCK_SEQ, BLOCK_CONV = 32, 128
    grid = (triton.cdiv(num_prefill_tokens, BLOCK_SEQ), triton.cdiv(width, BLOCK_CONV))

    _extract_transpose_prefill_kernel[grid](
        src,
        out,
        num_prefill_tokens,
        src.shape[1],
        start_col,
        width,
        BLOCK_SEQ,
        BLOCK_CONV,
    )
    return out


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
    return extract_transpose_prefill_slice(
        zxbcdt,
        num_prefill_tokens,
        d_inner,
        conv_dim,
    )


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


@triton.jit
def _transpose_and_split_qkv_kernel(
    prefill_t_ptr,
    decode_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    num_prefill,
    num_decode,
    num_cols,
    q_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Fused transpose-prefill + split-decode into contiguous q, k, v.

    Reads prefill from transposed layout [D, T_p] and decode from
    row-major layout [T_d, D], writes both into contiguous q/k/v outputs.
    Grid: (num_seq_blocks_total, num_dim_blocks, 3)
    program_id(2): 0=Q, 1=K, 2=V
    """
    pid_seq = tl.program_id(0)
    pid_dim = tl.program_id(1)
    pid_out = tl.program_id(2)

    total_seq = num_prefill + num_decode
    out_dim = tl.where(pid_out == 2, v_dim, tl.where(pid_out == 1, k_dim, q_dim))
    src_col_offset = tl.where(pid_out == 0, 0, tl.where(pid_out == 1, q_dim, q_dim + k_dim))

    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dim_offsets = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    seq_mask = seq_offsets < total_seq
    dim_mask = dim_offsets < out_dim
    mask = seq_mask[:, None] & dim_mask[None, :]

    # Determine if each row is prefill or decode
    is_prefill = seq_offsets < num_prefill
    is_decode = ~is_prefill & (seq_offsets < total_seq)

    # Prefill: read from prefill_t[D, T_p] transposed — index [col, row]
    prefill_src_col = src_col_offset + dim_offsets
    prefill_indices = prefill_src_col[None, :].to(tl.int64) * num_prefill + seq_offsets[:, None].to(
        tl.int64
    )
    prefill_data = tl.load(
        prefill_t_ptr + prefill_indices, mask=is_prefill[:, None] & dim_mask[None, :], other=0.0
    )

    # Decode: read from decode_ptr[T_d, D] row-major
    decode_row = seq_offsets - num_prefill
    decode_indices = decode_row[:, None].to(tl.int64) * num_cols + (
        src_col_offset + dim_offsets[None, :]
    ).to(tl.int64)
    decode_data = tl.load(
        decode_ptr + decode_indices, mask=is_decode[:, None] & dim_mask[None, :], other=0.0
    )

    # Merge
    data = tl.where(is_prefill[:, None], prefill_data, decode_data)

    # Write to output [total_seq, out_dim]
    out_ptr = tl.where(pid_out == 0, q_ptr, tl.where(pid_out == 1, k_ptr, v_ptr))
    dst_indices = seq_offsets[:, None].to(tl.int64) * out_dim + dim_offsets[None, :]
    tl.store(out_ptr + dst_indices, data, mask=mask)


def transpose_and_split_qkv(
    prefill_t: torch.Tensor,
    decode: torch.Tensor,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    num_q_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused transpose prefill [D, T_p] + split decode [T_d, D] into contiguous q, k, v.

    Replaces separate transpose_copy_back + split_qkv_contiguous for mixed batches.
    """
    num_cols, num_prefill = prefill_t.shape
    num_decode = decode.shape[0]
    total_seq = num_prefill + num_decode

    q_flat = torch.empty(total_seq, q_dim, dtype=prefill_t.dtype, device=prefill_t.device)
    k_flat = torch.empty(total_seq, k_dim, dtype=prefill_t.dtype, device=prefill_t.device)
    v_flat = torch.empty(total_seq, v_dim, dtype=prefill_t.dtype, device=prefill_t.device)

    BLOCK_SEQ, BLOCK_DIM = 32, 128
    max_dim = max(q_dim, k_dim, v_dim)
    grid = (triton.cdiv(total_seq, BLOCK_SEQ), triton.cdiv(max_dim, BLOCK_DIM), 3)

    _transpose_and_split_qkv_kernel[grid](
        prefill_t,
        decode,
        q_flat,
        k_flat,
        v_flat,
        num_prefill,
        num_decode,
        num_cols,
        q_dim,
        k_dim,
        v_dim,
        BLOCK_SEQ,
        BLOCK_DIM,
    )

    return (
        q_flat.view(1, total_seq, num_q_heads, head_k_dim),
        k_flat.view(1, total_seq, num_q_heads, head_k_dim),
        v_flat.view(1, total_seq, num_v_heads, head_v_dim),
    )


@triton.jit
def _split_qkv_contiguous_kernel(
    src_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    seq_len,
    src_stride_seq,
    src_stride_dim,
    q_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Split mixed_qkv [T, qkv_dim] into 3 contiguous output tensors.

    Supports arbitrary source strides to handle both contiguous and
    transposed inputs (e.g. from causal_conv1d_fn(...).transpose(0, 1)).

    Grid: (num_seq_blocks, num_dim_blocks_for_max_dim, 3)
    program_id(2): 0=Q, 1=K, 2=V
    """
    pid_seq = tl.program_id(0)
    pid_dim = tl.program_id(1)
    pid_out = tl.program_id(2)

    # Determine output dim and source column offset for this output
    out_dim = tl.where(pid_out == 2, v_dim, tl.where(pid_out == 1, k_dim, q_dim))
    src_col_offset = tl.where(pid_out == 0, 0, tl.where(pid_out == 1, q_dim, q_dim + k_dim))

    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dim_offsets = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    seq_mask = seq_offsets < seq_len
    dim_mask = dim_offsets < out_dim
    mask = seq_mask[:, None] & dim_mask[None, :]

    # Read from src [T, qkv_dim] using actual strides
    src_indices = (
        seq_offsets[:, None].to(tl.int64) * src_stride_seq
        + (src_col_offset + dim_offsets[None, :]).to(tl.int64) * src_stride_dim
    )
    data = tl.load(src_ptr + src_indices, mask=mask, other=0.0)

    # Write to output [T, out_dim] — contiguous, stride = (out_dim, 1)
    out_ptr = tl.where(pid_out == 0, q_ptr, tl.where(pid_out == 1, k_ptr, v_ptr))
    dst_indices = seq_offsets[:, None].to(tl.int64) * out_dim + dim_offsets[None, :]
    tl.store(out_ptr + dst_indices, data, mask=mask)


def split_qkv_contiguous(
    mixed_qkv: torch.Tensor,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    num_q_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split mixed_qkv [T, qkv_dim] into 3 contiguous tensors and reshape.

    Returns:
        q_out [1, T, num_q_heads, head_k_dim]  — contiguous
        k_out [1, T, num_q_heads, head_k_dim]  — contiguous
        v_out [1, T, num_v_heads, head_v_dim]  — contiguous
    """
    seq_len = mixed_qkv.shape[0]
    src_stride_seq, src_stride_dim = mixed_qkv.stride()

    q_flat = torch.empty(seq_len, q_dim, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    k_flat = torch.empty(seq_len, k_dim, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    v_flat = torch.empty(seq_len, v_dim, dtype=mixed_qkv.dtype, device=mixed_qkv.device)

    BLOCK_SEQ = 32
    BLOCK_DIM = 128
    max_dim = max(q_dim, k_dim, v_dim)
    grid = (
        triton.cdiv(seq_len, BLOCK_SEQ),
        triton.cdiv(max_dim, BLOCK_DIM),
        3,
    )

    _split_qkv_contiguous_kernel[grid](
        mixed_qkv,
        q_flat,
        k_flat,
        v_flat,
        seq_len,
        src_stride_seq,
        src_stride_dim,
        q_dim,
        k_dim,
        v_dim,
        BLOCK_SEQ,
        BLOCK_DIM,
    )

    return (
        q_flat.view(1, seq_len, num_q_heads, head_k_dim),
        k_flat.view(1, seq_len, num_q_heads, head_k_dim),
        v_flat.view(1, seq_len, num_v_heads, head_v_dim),
    )


@triton.jit
def _ssd_output_transpose_kernel(
    src_ptr,
    dst_ptr,
    num_prefill_tokens,
    H,
    D,
    NC,
    CS,
    stride_h,
    stride_d,
    stride_nc,
    HD,
    BLOCK_L: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    # (1, H, D, NC, CS) → (L, H*D); BLOCK_L | CS keeps each tile in one chunk.
    pid_l = tl.program_id(0)
    pid_hd = tl.program_id(1)

    l_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    hd_offs = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD)

    l_mask = l_offs < num_prefill_tokens
    hd_mask = hd_offs < HD

    nc = l_offs // CS
    cs = l_offs % CS
    h = hd_offs // D
    d = hd_offs % D

    # int64 offsets: H*D*NC*CS can exceed INT32_MAX for long seqlens.
    src_off = (
        h.to(tl.int64)[None, :] * stride_h
        + d.to(tl.int64)[None, :] * stride_d
        + nc.to(tl.int64)[:, None] * stride_nc
        + cs.to(tl.int64)[:, None]
    )
    mask = l_mask[:, None] & hd_mask[None, :]
    data = tl.load(src_ptr + src_off, mask=mask, other=0.0)

    dst_off = l_offs.to(tl.int64)[:, None] * HD + hd_offs[None, :]
    tl.store(dst_ptr + dst_off, data, mask=mask)


def ssd_output_transpose(
    out_contig: torch.Tensor,
    dst: torch.Tensor,
    num_prefill_tokens: int,
) -> None:
    """Transpose (1, H, D, NC, CS) bf16 to (num_prefill_tokens, H*D) bf16 in dst."""
    # Tuned on B200; bandwidth-bound, so a wider tile (two heads, num_warps=2)
    # beats more warps. BLOCK_L must divide CS so each tile stays in one chunk.
    BLOCK_L = 128
    BLOCK_HD = 128
    NUM_WARPS = 2

    assert out_contig.is_contiguous(), "out_contig must be contiguous in (B, H, D, NC, CS)"
    assert out_contig.ndim == 5 and out_contig.shape[0] == 1, (
        f"expected (1, H, D, NC, CS), got {tuple(out_contig.shape)}"
    )
    _, H, D, NC, CS = out_contig.shape
    HD = H * D
    assert dst.is_contiguous()
    assert dst.numel() == num_prefill_tokens * HD, (
        f"dst numel {dst.numel()} != L_p*H*D = {num_prefill_tokens}*{HD}"
    )
    assert NC * CS >= num_prefill_tokens, (
        f"padded seqlen {NC * CS} < num_prefill_tokens {num_prefill_tokens}"
    )
    assert CS % BLOCK_L == 0, f"chunk_size {CS} must be a multiple of BLOCK_L={BLOCK_L}"

    stride_h = D * NC * CS
    stride_d = NC * CS
    stride_nc = CS

    grid = (triton.cdiv(num_prefill_tokens, BLOCK_L), triton.cdiv(HD, BLOCK_HD))
    _ssd_output_transpose_kernel[grid](
        out_contig,
        dst,
        num_prefill_tokens,
        H,
        D,
        NC,
        CS,
        stride_h,
        stride_d,
        stride_nc,
        HD,
        BLOCK_L=BLOCK_L,
        BLOCK_HD=BLOCK_HD,
        num_warps=NUM_WARPS,
    )
