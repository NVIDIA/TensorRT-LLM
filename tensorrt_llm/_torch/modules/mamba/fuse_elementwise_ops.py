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
    src_stride_seq,
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

    # Cast to int64 to avoid overflow: seq_offsets * src_stride_seq can exceed
    # INT32_MAX (e.g., 131071 * 22656 = 2,969,544,576 > 2,147,483,647)
    src_offsets = (
        seq_offsets[:, None].to(tl.int64) * src_stride_seq + d_inner + conv_offsets[None, :]
    )
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
    Extract and transpose a prefill slice for causal_conv1d_fn.

    Input:  src[num_tokens, num_cols], rows contiguous (arbitrary row stride,
            so column-slice views of a wider tensor work in place)
    Output: [width, num_prefill_tokens]
    """
    assert src.stride(1) == 1
    out = torch.empty(width, num_prefill_tokens, dtype=src.dtype, device=src.device)

    BLOCK_SEQ, BLOCK_CONV = 32, 128
    grid = (triton.cdiv(num_prefill_tokens, BLOCK_SEQ), triton.cdiv(width, BLOCK_CONV))

    _extract_transpose_prefill_kernel[grid](
        src,
        out,
        num_prefill_tokens,
        src.stride(0),
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
def _fused_gdn_post_conv_kernel(
    prefill_ptr,
    decode_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    num_prefill_tokens,
    num_decode_tokens,
    prefill_stride_token,
    prefill_stride_dim,
    decode_stride_token,
    decode_stride_dim,
    a_stride_token,
    a_stride_head,
    b_stride_token,
    b_stride_head,
    l2_norm_eps,
    softplus_threshold,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_K_DIM: tl.constexpr,
    HEAD_V_DIM: tl.constexpr,
    HAS_DECODE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Prepare contiguous Q/K/V and GDN gates from causal-conv output."""
    token_block = tl.program_id(0)
    head_idx = tl.program_id(1)

    token_offsets = token_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    total_tokens = num_prefill_tokens + num_decode_tokens
    token_mask = token_offsets < total_tokens
    is_prefill = token_offsets < num_prefill_tokens
    if HAS_DECODE:
        is_decode = ~is_prefill & token_mask

    if head_idx < NUM_K_HEADS:
        dim_offsets = tl.arange(0, BLOCK_K)
        dim_mask = dim_offsets < HEAD_K_DIM
        load_mask_prefill = is_prefill[:, None] & dim_mask[None, :]
        store_mask = token_mask[:, None] & dim_mask[None, :]

        q_feature_offsets = head_idx * HEAD_K_DIM + dim_offsets

        prefill_q_offsets = (
            token_offsets[:, None].to(tl.int64) * prefill_stride_token
            + q_feature_offsets[None, :].to(tl.int64) * prefill_stride_dim
        )
        prefill_k_offsets = prefill_q_offsets + NUM_K_HEADS * HEAD_K_DIM * prefill_stride_dim
        q_prefill = tl.load(prefill_ptr + prefill_q_offsets, mask=load_mask_prefill, other=0.0)
        k_prefill = tl.load(prefill_ptr + prefill_k_offsets, mask=load_mask_prefill, other=0.0)

        if HAS_DECODE:
            load_mask_decode = is_decode[:, None] & dim_mask[None, :]
            decode_rows = token_offsets - num_prefill_tokens
            decode_q_offsets = (
                decode_rows[:, None].to(tl.int64) * decode_stride_token
                + q_feature_offsets[None, :].to(tl.int64) * decode_stride_dim
            )
            decode_k_offsets = decode_q_offsets + NUM_K_HEADS * HEAD_K_DIM * decode_stride_dim
            q_decode = tl.load(decode_ptr + decode_q_offsets, mask=load_mask_decode, other=0.0)
            k_decode = tl.load(decode_ptr + decode_k_offsets, mask=load_mask_decode, other=0.0)
            q_values = tl.where(is_prefill[:, None], q_prefill, q_decode).to(tl.float32)
            k_values = tl.where(is_prefill[:, None], k_prefill, k_decode).to(tl.float32)
        else:
            q_values = q_prefill.to(tl.float32)
            k_values = k_prefill.to(tl.float32)

        q_inv_norm = 1.0 / tl.sqrt(tl.sum(q_values * q_values, axis=1) + l2_norm_eps)
        k_inv_norm = 1.0 / tl.sqrt(tl.sum(k_values * k_values, axis=1) + l2_norm_eps)
        q_values *= q_inv_norm[:, None]
        k_values *= k_inv_norm[:, None]

        output_offsets = (
            token_offsets[:, None].to(tl.int64) * (NUM_K_HEADS * HEAD_K_DIM)
            + q_feature_offsets[None, :]
        )
        tl.store(q_ptr + output_offsets, q_values, mask=store_mask)
        tl.store(k_ptr + output_offsets, k_values, mask=store_mask)
    else:
        value_head_idx = head_idx - NUM_K_HEADS
        dim_offsets = tl.arange(0, BLOCK_V)
        dim_mask = dim_offsets < HEAD_V_DIM
        load_mask_prefill = is_prefill[:, None] & dim_mask[None, :]
        store_mask = token_mask[:, None] & dim_mask[None, :]

        value_feature_offsets = (
            2 * NUM_K_HEADS * HEAD_K_DIM + value_head_idx * HEAD_V_DIM + dim_offsets
        )
        prefill_v_offsets = (
            token_offsets[:, None].to(tl.int64) * prefill_stride_token
            + value_feature_offsets[None, :].to(tl.int64) * prefill_stride_dim
        )
        v_prefill = tl.load(prefill_ptr + prefill_v_offsets, mask=load_mask_prefill, other=0.0)
        if HAS_DECODE:
            load_mask_decode = is_decode[:, None] & dim_mask[None, :]
            decode_rows = token_offsets - num_prefill_tokens
            decode_v_offsets = (
                decode_rows[:, None].to(tl.int64) * decode_stride_token
                + value_feature_offsets[None, :].to(tl.int64) * decode_stride_dim
            )
            v_decode = tl.load(decode_ptr + decode_v_offsets, mask=load_mask_decode, other=0.0)
            v_values = tl.where(is_prefill[:, None], v_prefill, v_decode)
        else:
            v_values = v_prefill

        output_offsets = (
            token_offsets[:, None].to(tl.int64) * (NUM_V_HEADS * HEAD_V_DIM)
            + value_head_idx * HEAD_V_DIM
            + dim_offsets[None, :]
        )
        tl.store(v_ptr + output_offsets, v_values, mask=store_mask)

        a_offsets = token_offsets.to(tl.int64) * a_stride_token + value_head_idx * a_stride_head
        b_offsets = token_offsets.to(tl.int64) * b_stride_token + value_head_idx * b_stride_head
        a_values = tl.load(a_ptr + a_offsets, mask=token_mask, other=0.0).to(tl.float32)
        b_values = tl.load(b_ptr + b_offsets, mask=token_mask, other=0.0).to(tl.float32)
        A_log = tl.load(A_log_ptr + value_head_idx).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + value_head_idx).to(tl.float32)

        gate_input = a_values + dt_bias
        softplus = tl.where(
            gate_input <= softplus_threshold,
            tl.log(1.0 + tl.exp(gate_input)),
            gate_input,
        )
        g_values = -tl.exp(A_log) * softplus
        beta_values = tl.sigmoid(b_values)
        gate_offsets = token_offsets.to(tl.int64) * NUM_V_HEADS + value_head_idx
        tl.store(g_ptr + gate_offsets, g_values, mask=token_mask)
        tl.store(beta_ptr + gate_offsets, beta_values, mask=token_mask)


def fused_gdn_post_conv(
    prefill: torch.Tensor,
    decode: torch.Tensor | None,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    l2_norm_eps: float = 1e-6,
    softplus_threshold: float = 20.0,
    beta_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse GDN post-conv split, Q/K normalization, and gate preparation.

    Args:
        prefill: Prefill causal-conv output with shape ``[qkv_dim, num_prefill_tokens]``.
        decode: Optional decode causal-conv output with shape ``[num_decode_tokens, qkv_dim]``.
        a: Gate input with shape ``[num_tokens, num_v_heads]``.
        b: Beta input with shape ``[num_tokens, num_v_heads]``.
        A_log: Log decay with shape ``[num_v_heads]``.
        dt_bias: Time-step bias with shape ``[num_v_heads]``.
        num_k_heads: Number of local query/key heads.
        head_k_dim: Query/key dimension per head.
        num_v_heads: Number of local value heads.
        head_v_dim: Value dimension per head.
        l2_norm_eps: Epsilon added to the Q/K squared norm.
        softplus_threshold: Linear fallback threshold for the decay gate softplus.
        beta_dtype: Output dtype for beta. Target verification uses the input dtype for parity.
    Returns:
        Contiguous Q, K, V, log-space G, and beta tensors with shapes
        ``[1, num_tokens, num_k_heads, head_k_dim]``,
        ``[1, num_tokens, num_k_heads, head_k_dim]``,
        ``[1, num_tokens, num_v_heads, head_v_dim]``, and twice
        ``[1, num_tokens, num_v_heads]``.
    """
    num_prefill_tokens = prefill.shape[1]
    num_decode_tokens = 0 if decode is None else decode.shape[0]
    num_tokens = num_prefill_tokens + num_decode_tokens

    q = torch.empty(
        (1, num_tokens, num_k_heads, head_k_dim), dtype=prefill.dtype, device=prefill.device
    )
    k = torch.empty_like(q)
    v = torch.empty(
        (1, num_tokens, num_v_heads, head_v_dim), dtype=prefill.dtype, device=prefill.device
    )
    g = torch.empty((1, num_tokens, num_v_heads), dtype=torch.float32, device=prefill.device)
    beta = torch.empty((1, num_tokens, num_v_heads), dtype=beta_dtype, device=prefill.device)
    if num_tokens == 0:
        return q, k, v, g, beta

    has_decode = decode is not None
    # Triton requires a tensor for every pointer argument. HAS_DECODE=False
    # compile-time eliminates all decode address arithmetic and loads, so this
    # pointer and its strides are never interpreted as a decode tensor.
    decode_input = prefill if decode is None else decode
    block_tokens = 16
    block_k = triton.next_power_of_2(head_k_dim)
    block_v = triton.next_power_of_2(head_v_dim)
    grid = (triton.cdiv(num_tokens, block_tokens), num_k_heads + num_v_heads)
    _fused_gdn_post_conv_kernel[grid](
        prefill,
        decode_input,
        a,
        b,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        num_prefill_tokens,
        num_decode_tokens,
        prefill.stride(1),
        prefill.stride(0),
        decode_input.stride(0),
        decode_input.stride(1),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        l2_norm_eps,
        softplus_threshold,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        has_decode,
        block_tokens,
        block_k,
        block_v,
        num_warps=4,
        num_stages=2,
    )
    return q, k, v, g, beta


@triton.jit
def _pack_gdn_decode_qkv_kernel(
    mixed_qkv_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    num_tokens,
    stride_token,
    stride_dim,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_K_DIM: tl.constexpr,
    HEAD_V_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Pack target-verification decode Q/K/V without normalization or gates."""
    token_offsets = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    head_idx = tl.program_id(1)
    token_mask = token_offsets < num_tokens

    if head_idx < NUM_K_HEADS:
        dim_offsets = tl.arange(0, BLOCK_K)
        dim_mask = dim_offsets < HEAD_K_DIM
        mask = token_mask[:, None] & dim_mask[None, :]
        q_feature_offsets = head_idx * HEAD_K_DIM + dim_offsets
        q_offsets = (
            token_offsets[:, None].to(tl.int64) * stride_token
            + q_feature_offsets[None, :].to(tl.int64) * stride_dim
        )
        k_offsets = q_offsets + NUM_K_HEADS * HEAD_K_DIM * stride_dim
        output_offsets = (
            token_offsets[:, None].to(tl.int64) * (NUM_K_HEADS * HEAD_K_DIM)
            + q_feature_offsets[None, :]
        )
        tl.store(q_ptr + output_offsets, tl.load(mixed_qkv_ptr + q_offsets, mask=mask), mask=mask)
        tl.store(k_ptr + output_offsets, tl.load(mixed_qkv_ptr + k_offsets, mask=mask), mask=mask)
    else:
        value_head_idx = head_idx - NUM_K_HEADS
        dim_offsets = tl.arange(0, BLOCK_V)
        dim_mask = dim_offsets < HEAD_V_DIM
        mask = token_mask[:, None] & dim_mask[None, :]
        value_feature_offsets = (
            2 * NUM_K_HEADS * HEAD_K_DIM + value_head_idx * HEAD_V_DIM + dim_offsets
        )
        value_offsets = (
            token_offsets[:, None].to(tl.int64) * stride_token
            + value_feature_offsets[None, :].to(tl.int64) * stride_dim
        )
        output_offsets = (
            token_offsets[:, None].to(tl.int64) * (NUM_V_HEADS * HEAD_V_DIM)
            + value_head_idx * HEAD_V_DIM
            + dim_offsets[None, :]
        )
        tl.store(
            v_ptr + output_offsets, tl.load(mixed_qkv_ptr + value_offsets, mask=mask), mask=mask
        )


def pack_gdn_decode_qkv(
    mixed_qkv: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack target-verification decode Q/K/V while keeping them unnormalized.

    Args:
        mixed_qkv: Decode causal-conv output with shape ``[num_tokens, qkv_dim]``.
        num_k_heads: Number of local query/key heads.
        head_k_dim: Query/key dimension per head.
        num_v_heads: Number of local value heads.
        head_v_dim: Value dimension per head.

    Returns:
        Contiguous Q, K, and V tensors with shapes
        ``[1, num_tokens, num_k_heads, head_k_dim]``,
        ``[1, num_tokens, num_k_heads, head_k_dim]``, and
        ``[1, num_tokens, num_v_heads, head_v_dim]``.
    """
    num_tokens = mixed_qkv.shape[0]
    q = torch.empty(
        (1, num_tokens, num_k_heads, head_k_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    k = torch.empty_like(q)
    v = torch.empty(
        (1, num_tokens, num_v_heads, head_v_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    if num_tokens == 0:
        return q, k, v

    block_tokens = 16
    block_k = triton.next_power_of_2(head_k_dim)
    block_v = triton.next_power_of_2(head_v_dim)
    grid = (triton.cdiv(num_tokens, block_tokens), num_k_heads + num_v_heads)
    _pack_gdn_decode_qkv_kernel[grid](
        mixed_qkv,
        q,
        k,
        v,
        num_tokens,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        block_tokens,
        block_k,
        block_v,
        num_warps=4,
        num_stages=2,
    )
    return q, k, v


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
