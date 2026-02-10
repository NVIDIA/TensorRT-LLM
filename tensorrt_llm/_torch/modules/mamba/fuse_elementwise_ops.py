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


# ---------------------------------------------------------------------------
# Plan C: Fused extract + conv1d + split kernel
# Eliminates extract_transpose (12ms) + split_transpose (12ms) by reading
# directly from zxbcdt[tokens, d_in_proj] and writing split x, B, C.
# Based on _causal_conv1d_fwd_kernel from causal_conv1d_triton.py.
# ---------------------------------------------------------------------------

PAD_SLOT_ID = -1


@triton.jit
def _fused_extract_conv1d_split_kernel(
    # Input
    zxbcdt_ptr,  # [num_tokens, d_in_proj]
    w_ptr,  # [conv_dim, width]
    bias_ptr,  # [conv_dim]
    conv_states_ptr,  # [num_cache_lines, conv_dim, width-1]
    cache_indices_ptr,  # [num_seqs]
    has_initial_states_ptr,  # [num_seqs]
    query_start_loc_ptr,  # [num_seqs + 1]
    # Output (3 separate buffers)
    out_x_ptr,  # [num_tokens, d_inner]
    out_B_ptr,  # [num_tokens, bc_size]
    out_C_ptr,  # [num_tokens, bc_size]
    # Dimensions
    conv_dim,
    d_in_proj,
    d_inner,
    bc_size,
    num_cache_lines: tl.constexpr,
    # Strides
    stride_cs_seq: tl.constexpr,
    stride_cs_dim: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    # Meta
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused: extract xbc from zxbcdt + causal_conv1d + SiLU + split to x,B,C.

    Reads directly from zxbcdt in (tokens, d_in_proj) layout.
    Writes split outputs: x[tokens, d_inner], B[tokens, bc_size], C[tokens, bc_size].
    """
    state_len = KERNEL_WIDTH - 1

    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    seq_start = tl.load(query_start_loc_ptr + idx_seq)
    seq_end = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = seq_end - seq_start

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)
    if segment_len <= 0:
        return

    # Input base: zxbcdt[seq_start, d_inner + idx_feats]
    # stride along tokens = d_in_proj, stride along features = 1
    x_base = zxbcdt_ptr + seq_start * d_in_proj + d_inner + idx_feats

    # Conv state setup
    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(cache_indices_ptr + idx_seq).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return

    conv_states_base = (
        conv_states_ptr + conv_state_batch_coord * stride_cs_seq + idx_feats * stride_cs_dim
    )

    mask_feat = idx_feats < conv_dim

    # ---- Load initial sliding-window state (chunk_offset == 0) ----
    if chunk_offset == 0:
        load_init = False
        if HAS_INITIAL_STATES:
            load_init = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init:
            prior = conv_states_base + (state_len - 1) * stride_cs_tok
            if KERNEL_WIDTH >= 4:
                col2 = tl.load(prior, mask_feat, 0.0)
                col1 = tl.load(prior - stride_cs_tok, mask_feat, 0.0)
                col0 = tl.load(prior - 2 * stride_cs_tok, mask_feat, 0.0)
            elif KERNEL_WIDTH >= 3:
                col1 = tl.load(prior, mask_feat, 0.0)
                col0 = tl.load(prior - stride_cs_tok, mask_feat, 0.0)
            elif KERNEL_WIDTH >= 2:
                col0 = tl.load(prior, mask_feat, 0.0)
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N,), dtype=zxbcdt_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N,), dtype=zxbcdt_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N,), dtype=zxbcdt_ptr.dtype.element_ty)

        # Update conv_states with last state_len tokens
        if state_len <= seqlen:
            idx_tok = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            src_ptrs = (
                zxbcdt_ptr
                + (seq_start + idx_tok)[:, None] * d_in_proj
                + (d_inner + idx_feats)[None, :]
            )
            src_mask = (
                (idx_tok >= 0)[:, None]
                & (idx_tok < seqlen)[:, None]
                & (idx_feats < conv_dim)[None, :]
            )
            new_state = tl.load(src_ptrs, src_mask, 0.0)
            dst_tok = tl.arange(0, NP2_STATELEN)
            dst_ptrs = conv_states_base[None, :] + (dst_tok * stride_cs_tok)[:, None]
            dst_mask = (dst_tok < state_len)[:, None] & (idx_feats < conv_dim)[None, :]
            tl.store(dst_ptrs, new_state, dst_mask)
        else:
            if load_init:
                idx_tok = tl.arange(0, NP2_STATELEN)
                cs_src = (
                    conv_states_ptr
                    + conv_state_batch_coord * stride_cs_seq
                    + idx_feats[None, :] * stride_cs_dim
                    + ((idx_tok + seqlen) * stride_cs_tok)[:, None]
                )
                cs_mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tok + seqlen) < state_len)[:, None]
                    & (idx_feats < conv_dim)[None, :]
                )
                conv_state = tl.load(cs_src, cs_mask, 0.0)
                VAL = state_len - seqlen
                x_src = (
                    zxbcdt_ptr
                    + (seq_start + idx_tok - VAL)[:, None] * d_in_proj
                    + (d_inner + idx_feats)[None, :]
                )
                x_mask = (
                    (idx_tok - VAL >= 0)[:, None]
                    & (idx_tok - VAL < seqlen)[:, None]
                    & (idx_feats < conv_dim)[None, :]
                )
                loaded_x = tl.load(x_src, x_mask, 0.0)
                new_state = tl.where(cs_mask, conv_state, loaded_x)
                dst_ptrs = conv_states_base[None, :] + (idx_tok * stride_cs_tok)[:, None]
                dst_mask = (idx_tok < state_len)[:, None] & (idx_feats < conv_dim)[None, :]
                tl.store(dst_ptrs, new_state, dst_mask)
            else:
                idx_tok = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_src = (
                    zxbcdt_ptr
                    + (seq_start + idx_tok - VAL)[:, None] * d_in_proj
                    + (d_inner + idx_feats)[None, :]
                )
                x_mask = (
                    (idx_tok - VAL >= 0)[:, None]
                    & (idx_tok - VAL < seqlen)[:, None]
                    & (idx_feats < conv_dim)[None, :]
                )
                new_state = tl.load(x_src, x_mask, 0.0)
                dst_ptrs = conv_states_base[None, :] + (idx_tok * stride_cs_tok)[:, None]
                dst_mask = (idx_tok < state_len)[:, None] & (idx_feats < conv_dim)[None, :]
                tl.store(dst_ptrs, new_state, dst_mask)
    else:
        # chunk_offset > 0: load prior tokens from input
        prior = x_base + (token_offset - 1) * d_in_proj
        if KERNEL_WIDTH >= 4:
            col2 = tl.load(prior, mask_feat, 0.0)
            col1 = tl.load(prior - d_in_proj, mask_feat, 0.0)
            col0 = tl.load(prior - 2 * d_in_proj, mask_feat, 0.0)
        elif KERNEL_WIDTH >= 3:
            col1 = tl.load(prior, mask_feat, 0.0)
            col0 = tl.load(prior - d_in_proj, mask_feat, 0.0)
        elif KERNEL_WIDTH >= 2:
            col0 = tl.load(prior, mask_feat, 0.0)

    # ---- Preload bias and weights ----
    if HAS_BIAS:
        acc_preload = tl.load(bias_ptr + idx_feats, mask=mask_feat, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Weight layout: [conv_dim, width] with stride [width, 1]
    # So w[f, k] is at w_ptr + f * width + k
    w_stride_dim = KERNEL_WIDTH  # width
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_ptr + idx_feats * w_stride_dim + 0, mask_feat, 0.0)
        w_col1 = tl.load(w_ptr + idx_feats * w_stride_dim + 1, mask_feat, 0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_ptr + idx_feats * w_stride_dim + 2, mask_feat, 0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_ptr + idx_feats * w_stride_dim + 3, mask_feat, 0.0)

    # ---- Output addressing: determine which buffer ----
    is_x = idx_feats < d_inner
    is_B = (idx_feats >= d_inner) & (idx_feats < d_inner + bc_size)
    local_feat = tl.where(
        is_x, idx_feats, tl.where(is_B, idx_feats - d_inner, idx_feats - d_inner - bc_size)
    )
    local_stride = tl.where(is_x, d_inner, bc_size)
    out_base = tl.where(is_x, out_x_ptr, tl.where(is_B, out_B_ptr, out_C_ptr))

    # ---- Main loop: conv1d + SiLU + store ----
    x_ptr_1d = x_base + token_offset * d_in_proj
    for idx_token in range(segment_len):
        acc = acc_preload

        # Width-4 convolution (most common case)
        if KERNEL_WIDTH == 4:
            acc += col0 * w_col0
            acc += col1 * w_col1
            acc += col2 * w_col2
            cur_x = tl.load(x_ptr_1d + idx_token * d_in_proj, mask=mask_feat)
            acc += cur_x * w_col3
            col0 = col1
            col1 = col2
            col2 = cur_x
        elif KERNEL_WIDTH == 3:
            acc += col0 * w_col0
            acc += col1 * w_col1
            cur_x = tl.load(x_ptr_1d + idx_token * d_in_proj, mask=mask_feat)
            acc += cur_x * w_col2
            col0 = col1
            col1 = cur_x
        elif KERNEL_WIDTH == 2:
            acc += col0 * w_col0
            cur_x = tl.load(x_ptr_1d + idx_token * d_in_proj, mask=mask_feat)
            acc += cur_x * w_col1
            col0 = cur_x

        # SiLU activation
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))

        # Store to split output
        token_abs = seq_start + token_offset + idx_token
        o_ptrs = out_base + token_abs * local_stride + local_feat
        mask_out = (idx_token < segment_len) & mask_feat
        tl.store(o_ptrs, acc, mask=mask_out)


def fused_extract_conv1d_split(
    zxbcdt: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    num_prefill_tokens: int,
    d_inner: int,
    conv_dim: int,
    n_groups: int,
    d_state: int,
    nheads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused: extract xbc from zxbcdt + causal_conv1d + SiLU + split to x, B, C.

    Replaces: extract_transpose + causal_conv1d_fn + fused_split_rearrange_after_conv1d

    Input:  zxbcdt[num_tokens, d_in_proj]
    Output: x[1, num_prefill_tokens, nheads, head_dim],
            B[1, num_prefill_tokens, n_groups, d_state],
            C[1, num_prefill_tokens, n_groups, d_state]
    """
    d_in_proj = zxbcdt.shape[1]
    bc_size = n_groups * d_state
    _, width = weight.shape
    num_seqs = cu_seqlens.shape[0] - 1

    x_flat = torch.empty(num_prefill_tokens, d_inner, dtype=zxbcdt.dtype, device=zxbcdt.device)
    B_flat = torch.empty(num_prefill_tokens, bc_size, dtype=zxbcdt.dtype, device=zxbcdt.device)
    C_flat = torch.empty(num_prefill_tokens, bc_size, dtype=zxbcdt.dtype, device=zxbcdt.device)

    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    num_cache_lines = conv_states.shape[0] if conv_states is not None else 0

    # Compute max_seq_len for grid sizing
    max_seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    BLOCK_M = 8
    BLOCK_N = 256

    grid = (
        num_seqs,
        (max_seq_len + BLOCK_M - 1) // BLOCK_M,
        triton.cdiv(conv_dim, BLOCK_N),
    )

    _fused_extract_conv1d_split_kernel[grid](
        zxbcdt,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        cu_seqlens,
        x_flat,
        B_flat,
        C_flat,
        conv_dim,
        d_in_proj,
        d_inner,
        bc_size,
        num_cache_lines,
        conv_states.stride(0) if conv_states is not None else 0,
        conv_states.stride(1) if conv_states is not None else 0,
        conv_states.stride(2) if conv_states is not None else 0,
        PAD_SLOT_ID,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=True,
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=True,
        NP2_STATELEN=np2_statelen,
        KERNEL_WIDTH=width,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return (
        x_flat.view(1, num_prefill_tokens, nheads, head_dim),
        B_flat.view(1, num_prefill_tokens, n_groups, d_state),
        C_flat.view(1, num_prefill_tokens, n_groups, d_state),
    )
