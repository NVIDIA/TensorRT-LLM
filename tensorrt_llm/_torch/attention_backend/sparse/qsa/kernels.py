# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for QSA sparse attention."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_gemma_norm_rope(
    x,
    pos_t,
    pos_h,
    pos_w,
    cos_sin,
    cos_sin_stride,
    norm_weight,
    eps,
    NUM_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    """Apply per-row Gemma RMSNorm and NeoX RoPE to register values."""
    dims = tl.arange(0, HEAD_DIM)
    x_f32 = x.to(tl.float32)
    weight = tl.load(norm_weight + dims).to(tl.float32) + 1.0
    reciprocal_rms = tl.rsqrt(tl.sum(x_f32 * x_f32, axis=1) / HEAD_DIM + eps)
    normalized = (x_f32 * reciprocal_rms[:, None] * weight[None, :]).to(cos_sin.dtype.element_ty)

    pairs = tl.arange(0, ROTARY_PAIRS)
    if IS_MROPE:
        height_pair = (pairs % 3 == 1) & (pairs < 3 * MROPE_H)
        width_pair = (pairs % 3 == 2) & (pairs < 3 * MROPE_W)
        position = tl.where(height_pair, pos_h, tl.where(width_pair, pos_w, pos_t))
    else:
        position = pos_t
    cosine = tl.load(cos_sin + position * cos_sin_stride + pairs)
    sine = tl.load(cos_sin + position * cos_sin_stride + ROTARY_PAIRS + pairs)
    rotated, passthrough = tl.split(
        tl.permute(
            tl.reshape(normalized, (NUM_ROWS, 2, HEAD_DIM // 2)),
            (0, 2, 1),
        )
    )
    first, second = tl.split(
        tl.permute(
            tl.reshape(rotated, (NUM_ROWS, 2, ROTARY_PAIRS)),
            (0, 2, 1),
        )
    )
    output_first = first * cosine[None, :] - second * sine[None, :]
    output_second = second * cosine[None, :] + first * sine[None, :]
    rotated = tl.reshape(
        tl.permute(tl.join(output_first, output_second), (0, 2, 1)),
        (NUM_ROWS, HEAD_DIM // 2),
    )
    return tl.reshape(
        tl.permute(tl.join(rotated, passthrough), (0, 2, 1)),
        (NUM_ROWS, HEAD_DIM),
    ).to(tl.bfloat16)


@triton.jit
def _qsa_decode_pre_indexer_kernel(
    q,
    token_k,
    position_coordinates,
    request_indices,
    logical_positions,
    block_table,
    index_cache,
    position_cache,
    q_norm_weight,
    k_norm_weight,
    cos_sin,
    eps,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_row: tl.constexpr,
    k_stride_dim: tl.constexpr,
    position_stride_row: tl.constexpr,
    position_stride_axis: tl.constexpr,
    block_table_stride_request: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    cache_stride_page: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    position_cache_stride_page: tl.constexpr,
    position_cache_stride_token: tl.constexpr,
    position_cache_stride_axis: tl.constexpr,
    cos_sin_stride: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    QUERY_HEADS_PER_CTA: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
    WORK_PER_ROW: tl.constexpr,
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    """Fuse index Q normalization/RoPE and raw K cache updates."""
    row = tl.program_id(0) // WORK_PER_ROW
    work = tl.program_id(0) % WORK_PER_ROW
    dims = tl.arange(0, HEAD_DIM)
    pos_t = tl.load(position_coordinates + row * position_stride_row)
    pos_h = tl.load(position_coordinates + row * position_stride_row + position_stride_axis)
    pos_w = tl.load(position_coordinates + row * position_stride_row + 2 * position_stride_axis)

    if work < NUM_QUERY_TILES:
        heads = work * QUERY_HEADS_PER_CTA + tl.arange(0, QUERY_HEADS_PER_CTA)
        valid_heads = heads < NUM_QUERY_HEADS
        q_ptrs = (
            q + row * q_stride_row + heads[:, None] * q_stride_head + dims[None, :] * q_stride_dim
        )
        q_values = tl.load(q_ptrs, mask=valid_heads[:, None], other=0.0)
        normalized_q = _qsa_gemma_norm_rope(
            q_values,
            pos_t,
            pos_h,
            pos_w,
            cos_sin,
            cos_sin_stride,
            q_norm_weight,
            eps,
            NUM_ROWS=QUERY_HEADS_PER_CTA,
            HEAD_DIM=HEAD_DIM,
            ROTARY_PAIRS=ROTARY_PAIRS,
            IS_MROPE=IS_MROPE,
            MROPE_H=MROPE_H,
            MROPE_W=MROPE_W,
        )
        tl.store(q_ptrs, normalized_q, mask=valid_heads[:, None])

    if work == NUM_QUERY_TILES:
        request = tl.load(request_indices + row).to(tl.int64)
        logical = tl.load(logical_positions + row).to(tl.int64)
        logical_page = logical // TOKENS_PER_BLOCK
        token_in_page = logical % TOKENS_PER_BLOCK
        physical_page = tl.load(
            block_table
            + request * block_table_stride_request
            + logical_page * block_table_stride_page
        ).to(tl.int64)
        current_k = tl.load(token_k + row * k_stride_row + dims * k_stride_dim)

        axes = tl.arange(0, 4)
        tl.store(
            position_cache
            + physical_page * position_cache_stride_page
            + token_in_page * position_cache_stride_token
            + axes * position_cache_stride_axis,
            tl.load(
                position_coordinates + row * position_stride_row + axes * position_stride_axis,
                mask=axes < 3,
                other=0,
            ),
            mask=axes < 3,
        )

        stored_k = current_k
        if (logical + 1) % COMPRESS_RATIO == 0:
            group_offsets = tl.arange(0, COMPRESS_RATIO)
            group_positions = logical - (COMPRESS_RATIO - 1) + group_offsets
            group_logical_pages = group_positions // TOKENS_PER_BLOCK
            group_tokens_in_page = group_positions % TOKENS_PER_BLOCK
            group_physical_pages = tl.load(
                block_table
                + request * block_table_stride_request
                + group_logical_pages * block_table_stride_page
            ).to(tl.int64)
            group_values = tl.load(
                index_cache
                + group_physical_pages[:, None] * cache_stride_page
                + group_tokens_in_page[:, None] * cache_stride_token
                + dims[None, :] * cache_stride_dim
            )
            group_values = tl.where(
                group_offsets[:, None] == COMPRESS_RATIO - 1,
                current_k[None, :],
                group_values,
            )
            # Preserve the reference's BF16 materialization after FP32 pooling.
            pooled = (tl.sum(group_values.to(tl.float32), axis=0) / COMPRESS_RATIO).to(tl.bfloat16)
            first_position = logical - (COMPRESS_RATIO - 1)
            first_logical_page = first_position // TOKENS_PER_BLOCK
            first_token = first_position % TOKENS_PER_BLOCK
            first_page = tl.load(
                block_table
                + request * block_table_stride_request
                + first_logical_page * block_table_stride_page
            ).to(tl.int64)
            first_pos_t = tl.load(
                position_cache
                + first_page * position_cache_stride_page
                + first_token * position_cache_stride_token
            )
            first_pos_h = tl.load(
                position_cache
                + first_page * position_cache_stride_page
                + first_token * position_cache_stride_token
                + position_cache_stride_axis
            )
            first_pos_w = tl.load(
                position_cache
                + first_page * position_cache_stride_page
                + first_token * position_cache_stride_token
                + 2 * position_cache_stride_axis
            )
            stored_k = tl.reshape(
                _qsa_gemma_norm_rope(
                    pooled[None, :],
                    first_pos_t,
                    first_pos_h,
                    first_pos_w,
                    cos_sin,
                    cos_sin_stride,
                    k_norm_weight,
                    eps,
                    NUM_ROWS=1,
                    HEAD_DIM=HEAD_DIM,
                    ROTARY_PAIRS=ROTARY_PAIRS,
                    IS_MROPE=IS_MROPE,
                    MROPE_H=MROPE_H,
                    MROPE_W=MROPE_W,
                ),
                (HEAD_DIM,),
            )
        tl.store(
            index_cache
            + physical_page * cache_stride_page
            + token_in_page * cache_stride_token
            + dims * cache_stride_dim,
            stored_k,
        )


def triton_qsa_decode_pre_indexer(
    *,
    q: torch.Tensor,
    token_k: torch.Tensor,
    position_coordinates: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    block_table: torch.Tensor,
    index_cache: torch.Tensor,
    position_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    tokens_per_block: int,
    compress_ratio: int,
    mrope_section: tuple[int, int, int] | None,
) -> torch.Tensor:
    """Run the ordinary-decode QSA pre-indexer in one Triton launch."""
    rows, num_query_heads, head_dim = q.shape
    if token_k.shape != (rows, 1, head_dim):
        raise ValueError("QSA decode index K shape does not match index Q")
    if position_coordinates.shape != (rows, 3):
        raise ValueError("QSA decode positions must have shape [rows, 3]")
    if cos_sin.ndim != 2 or cos_sin.shape[1] % 2 != 0:
        raise ValueError("QSA RoPE cache must be a packed 2D cos/sin tensor")
    rotary_pairs = cos_sin.shape[1] // 2
    if 4 * rotary_pairs != head_dim:
        raise ValueError("QSA fused decode requires half-width rotary embedding")
    head_dim_is_power_of_two = head_dim > 0 and (head_dim & (head_dim - 1)) == 0
    ratio_is_power_of_two = compress_ratio > 0 and (compress_ratio & (compress_ratio - 1)) == 0
    if not head_dim_is_power_of_two or not ratio_is_power_of_two:
        raise ValueError("QSA fused decode requires power-of-two head and compression widths")
    section = mrope_section if mrope_section is not None else (0, 0, 0)
    query_heads_per_cta = 2 if num_query_heads > 1 else 1
    work_per_row = triton.cdiv(num_query_heads, query_heads_per_cta) + 1
    _qsa_decode_pre_indexer_kernel[(rows * work_per_row,)](
        q,
        token_k,
        position_coordinates,
        request_indices,
        logical_positions,
        block_table,
        index_cache,
        position_cache,
        q_norm_weight,
        k_norm_weight,
        cos_sin,
        eps,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        token_k.stride(0),
        token_k.stride(2),
        position_coordinates.stride(0),
        position_coordinates.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        index_cache.stride(0),
        index_cache.stride(1),
        index_cache.stride(3),
        position_cache.stride(0),
        position_cache.stride(1),
        position_cache.stride(2),
        cos_sin.stride(0),
        NUM_QUERY_HEADS=num_query_heads,
        HEAD_DIM=head_dim,
        ROTARY_PAIRS=rotary_pairs,
        TOKENS_PER_BLOCK=tokens_per_block,
        COMPRESS_RATIO=compress_ratio,
        QUERY_HEADS_PER_CTA=query_heads_per_cta,
        NUM_QUERY_TILES=work_per_row - 1,
        WORK_PER_ROW=work_per_row,
        IS_MROPE=mrope_section is not None,
        MROPE_H=section[1],
        MROPE_W=section[2],
        num_warps=1,
    )
    return q


@triton.jit
def _expand_qsa_block_indices_kernel(
    block_indices,
    query_positions,
    sequence_lengths,
    output,
    block_stride: tl.constexpr,
    output_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    FINAL_TOPK: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, OUTPUT_BLOCK_SIZE)
    sequence_length = tl.load(sequence_lengths + row)

    source_columns = columns // COMPRESS_RATIO
    offsets = columns % COMPRESS_RATIO
    blocks = tl.load(
        block_indices + row * block_stride + source_columns,
        mask=(columns < TOKEN_TOPK) & (source_columns < BLOCK_TOPK),
        other=-1,
    )
    expanded = blocks * COMPRESS_RATIO + offsets
    expanded_valid = (columns < TOKEN_TOPK) & (blocks >= 0) & (expanded < sequence_length)

    valid_blocks = tl.load(
        block_indices + row * block_stride + columns,
        mask=columns < BLOCK_TOPK,
        other=-1,
    )
    valid_token_count = tl.minimum(
        tl.sum(((columns < BLOCK_TOPK) & (valid_blocks >= 0)).to(tl.int32), axis=0)
        * COMPRESS_RATIO,
        TOKEN_TOPK,
    )

    query_position = tl.load(query_positions + row)
    visible_tokens = query_position + 1
    tail_start = (visible_tokens // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_offset = columns - valid_token_count
    tail_count = visible_tokens - tail_start
    tail = tail_start + tail_offset
    tail_valid = (
        (tail_offset >= 0)
        & (tail_offset < COMPRESS_RATIO - 1)
        & (tail_offset < tail_count)
        & (tail < sequence_length)
    )

    result = tl.where(
        expanded_valid & (columns < valid_token_count),
        expanded,
        tl.where(tail_valid, tail, -1),
    )
    tl.store(
        output + row * output_stride + columns,
        result,
        mask=columns < FINAL_TOPK,
    )


def triton_expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand compressed block indices without materializing Torch temporaries."""
    rows, block_topk = block_indices.shape
    final_topk = token_topk + compress_ratio - 1
    output = torch.empty(
        (rows, final_topk),
        dtype=torch.int32,
        device=block_indices.device,
    )
    if rows == 0:
        return output
    _expand_qsa_block_indices_kernel[(rows,)](
        block_indices,
        query_positions,
        sequence_lengths,
        output,
        block_indices.stride(0),
        output.stride(0),
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        FINAL_TOPK=final_topk,
        OUTPUT_BLOCK_SIZE=triton.next_power_of_2(final_topk),
        num_warps=8,
    )
    return output


@triton.jit
def _qsa_paged_index_scores_kernel(
    q,
    index_cache,
    block_table,
    query_positions,
    request_indices,
    output,
    score_scale,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    cache_stride_page: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    block_table_stride_request: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_block: tl.constexpr,
    NUM_INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    MAX_COMPRESSED_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    block_columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, INDEX_HEAD_DIM)
    request = tl.load(request_indices + row).to(tl.int64)
    query_position = tl.load(query_positions + row)
    visible_blocks = (query_position + 1) // COMPRESS_RATIO
    valid_blocks = (block_columns < MAX_COMPRESSED_BLOCKS) & (block_columns < visible_blocks)

    anchor_positions = block_columns * COMPRESS_RATIO + COMPRESS_RATIO - 1
    logical_pages = anchor_positions // TOKENS_PER_BLOCK
    token_in_page = anchor_positions % TOKENS_PER_BLOCK
    physical_pages = tl.load(
        block_table
        + request * block_table_stride_request
        + logical_pages * block_table_stride_page,
        mask=valid_blocks,
        other=0,
    ).to(tl.int64)

    query_values = tl.load(
        q
        + row * q_stride_row
        + head_offsets[:, None] * q_stride_head
        + dim_offsets[None, :] * q_stride_dim,
        mask=(head_offsets < NUM_INDEX_HEADS)[:, None],
        other=0.0,
    )
    keys = tl.load(
        index_cache
        + physical_pages[None, :] * cache_stride_page
        + token_in_page[None, :] * cache_stride_token
        + dim_offsets[:, None] * cache_stride_dim,
        mask=valid_blocks[None, :],
        other=0.0,
    )
    per_head_scores = tl.dot(query_values, keys) * score_scale
    scores = tl.sum(tl.maximum(per_head_scores, 0.0), axis=0)
    scores = tl.where(valid_blocks, scores, -float("inf"))
    tl.store(
        output + row * output_stride_row + block_columns * output_stride_block,
        scores,
        mask=block_columns < MAX_COMPRESSED_BLOCKS,
    )


def triton_qsa_paged_index_scores(
    *,
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Score every visible compressed key directly in the paged side cache."""
    rows, num_index_heads, index_head_dim = q.shape
    max_compressed_blocks = block_table.shape[1] * tokens_per_block // compress_ratio
    output = torch.empty(
        (rows, max_compressed_blocks),
        dtype=torch.float32,
        device=q.device,
    )
    # A wider key tile amortizes the query/cache setup for packed prefill and
    # batched decode. Keep the smallest decode graph on a narrow tile to avoid
    # adding latency to the single-request path.
    block_n = 32 if rows == 1 else 128
    grid = (rows, triton.cdiv(max_compressed_blocks, block_n))
    _qsa_paged_index_scores_kernel[grid](
        q,
        index_cache,
        block_table,
        query_positions,
        request_indices,
        output,
        index_head_dim**-0.5,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        index_cache.stride(0),
        index_cache.stride(1),
        index_cache.stride(2),
        index_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        output.stride(0),
        output.stride(1),
        NUM_INDEX_HEADS=num_index_heads,
        INDEX_HEAD_DIM=index_head_dim,
        TOKENS_PER_BLOCK=tokens_per_block,
        COMPRESS_RATIO=compress_ratio,
        MAX_COMPRESSED_BLOCKS=max_compressed_blocks,
        BLOCK_H=max(16, triton.next_power_of_2(num_index_heads)),
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=2,
    )
    return output


@triton.jit
def _qsa_paged_sparse_gqa_kernel(
    q,
    k_cache,
    v_cache,
    block_table,
    selected_tokens,
    request_indices,
    output,
    softmax_scale,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_page: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_page: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_dim: tl.constexpr,
    block_table_stride_request: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    selected_stride_row: tl.constexpr,
    selected_stride_token: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_head: tl.constexpr,
    output_stride_dim: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.load(request_indices + row).to(tl.int64)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    query_heads = kv_head * GROUP_SIZE + head_offsets
    query_values = tl.load(
        q
        + row * q_stride_row
        + query_heads[:, None] * q_stride_head
        + dim_offsets[None, :] * q_stride_dim,
        mask=(head_offsets < GROUP_SIZE)[:, None],
        other=0.0,
    )
    query_values = (query_values * softmax_scale * 1.4426950408889634).to(query_values.dtype)

    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    token_offsets = tl.arange(0, BLOCK_N)

    for start in range(0, TOPK, BLOCK_N):
        selected_columns = start + token_offsets
        logical_tokens = tl.load(
            selected_tokens + row * selected_stride_row + selected_columns * selected_stride_token,
            mask=selected_columns < TOPK,
            other=-1,
        )
        valid = (selected_columns < TOPK) & (logical_tokens >= 0)
        safe_tokens = tl.where(valid, logical_tokens, 0)
        logical_pages = safe_tokens // TOKENS_PER_BLOCK
        token_in_page = safe_tokens % TOKENS_PER_BLOCK
        physical_pages = tl.load(
            block_table
            + request * block_table_stride_request
            + logical_pages * block_table_stride_page,
            mask=valid,
            other=0,
        ).to(tl.int64)

        keys = tl.load(
            k_cache
            + physical_pages[None, :] * k_stride_page
            + kv_head * k_stride_head
            + token_in_page[None, :] * k_stride_token
            + dim_offsets[:, None] * k_stride_dim,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache
            + physical_pages[:, None] * v_stride_page
            + kv_head * v_stride_head
            + token_in_page[:, None] * v_stride_token
            + dim_offsets[None, :] * v_stride_dim,
            mask=valid[:, None],
            other=0.0,
        )
        # TRT-LLM uses unit-scale FP8 KV cache in the PyTorch backend.  Cast
        # loaded cache values to the query compute type because Triton does
        # not permit mixed BF16-by-FP8 dot operands.
        keys = keys.to(query_values.dtype)
        values = values.to(query_values.dtype)
        scores = tl.where(
            valid[None, :],
            tl.dot(query_values, keys),
            -float("inf"),
        )
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        correction = tl.math.exp2(running_max - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            accumulator * correction[:, None],
        )
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        running_max = next_max

    normalized = tl.where(
        running_sum[:, None] > 0,
        accumulator / running_sum[:, None],
        0.0,
    )
    tl.store(
        output
        + row * output_stride_row
        + query_heads[:, None] * output_stride_head
        + dim_offsets[None, :] * output_stride_dim,
        normalized,
        mask=(head_offsets < GROUP_SIZE)[:, None],
    )


@triton.jit
def _qsa_paged_sparse_gqa_splitk_kernel(
    q,
    k_cache,
    v_cache,
    block_table,
    selected_tokens,
    request_indices,
    partial_output,
    partial_lse,
    output,
    softmax_scale,
    num_rows,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_page: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_page: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_dim: tl.constexpr,
    block_table_stride_request: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    selected_stride_row: tl.constexpr,
    selected_stride_token: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_head: tl.constexpr,
    output_stride_dim: tl.constexpr,
    NUM_CACHE_PAGES: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    request = tl.load(request_indices + row).to(tl.int64)
    valid_request = (request >= 0) & (request < NUM_REQUESTS)
    safe_request = tl.minimum(tl.maximum(request, 0), NUM_REQUESTS - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    token_offsets = tl.arange(0, BLOCK_N)
    query_heads = kv_head * GROUP_SIZE + head_offsets
    query_values = tl.load(
        q
        + row * q_stride_row
        + query_heads[:, None] * q_stride_head
        + dim_offsets[None, :] * q_stride_dim,
        mask=(head_offsets < GROUP_SIZE)[:, None],
        other=0.0,
    )

    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    split_tile_start = split * NUM_TILES // NUM_SPLITS
    split_tile_end = (split + 1) * NUM_TILES // NUM_SPLITS

    for tile in range(split_tile_start, split_tile_end):
        selected_columns = tile * BLOCK_N + token_offsets
        logical_tokens = tl.load(
            selected_tokens + row * selected_stride_row + selected_columns * selected_stride_token,
            mask=selected_columns < TOPK,
            other=-1,
        )
        safe_tokens = tl.maximum(logical_tokens, 0)
        logical_pages = safe_tokens // TOKENS_PER_BLOCK
        token_in_page = safe_tokens % TOKENS_PER_BLOCK
        valid = (
            valid_request
            & (selected_columns < TOPK)
            & (logical_tokens >= 0)
            & (logical_pages < PAGE_TABLE_WIDTH)
        )
        physical_pages = tl.load(
            block_table
            + safe_request * block_table_stride_request
            + tl.minimum(logical_pages, PAGE_TABLE_WIDTH - 1) * block_table_stride_page,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid &= (physical_pages >= 0) & (physical_pages < NUM_CACHE_PAGES)
        safe_pages = tl.maximum(physical_pages, 0)

        keys = tl.load(
            k_cache
            + safe_pages[None, :] * k_stride_page
            + kv_head * k_stride_head
            + token_in_page[None, :] * k_stride_token
            + dim_offsets[:, None] * k_stride_dim,
            mask=valid[None, :],
            other=0.0,
        ).to(query_values.dtype)
        values = tl.load(
            v_cache
            + safe_pages[:, None] * v_stride_page
            + kv_head * v_stride_head
            + token_in_page[:, None] * v_stride_token
            + dim_offsets[None, :] * v_stride_dim,
            mask=valid[:, None],
            other=0.0,
        ).to(query_values.dtype)
        scores = tl.dot(query_values, keys)
        scores *= softmax_scale * 1.4426950408889634
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        correction = tl.math.exp2(running_max - next_max)
        probabilities = tl.where(
            valid[None, :],
            tl.math.exp2(scores - next_max[:, None]),
            0.0,
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            accumulator * correction[:, None],
        )
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        running_max = next_max

    has_values = running_sum > 0
    normalized = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(running_sum[:, None], 1.0e-20),
        0.0,
    )
    output_mask = (head_offsets < GROUP_SIZE)[:, None]
    if NUM_SPLITS == 1:
        tl.store(
            output
            + row * output_stride_row
            + query_heads[:, None] * output_stride_head
            + dim_offsets[None, :] * output_stride_dim,
            normalized,
            mask=output_mask,
        )
    else:
        partial_lse_values = tl.where(
            has_values,
            running_max + tl.math.log2(tl.maximum(running_sum, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output
            + ((split * num_rows + row) * NUM_QUERY_HEADS + query_heads[:, None]) * HEAD_DIM
            + dim_offsets[None, :],
            normalized,
            mask=output_mask,
        )
        tl.store(
            partial_lse + (split * num_rows + row) * NUM_QUERY_HEADS + query_heads,
            partial_lse_values,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output,
    partial_lse,
    output,
    num_rows,
    output_stride_row: tl.constexpr,
    output_stride_head: tl.constexpr,
    output_stride_dim: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse, axis=0)
    has_values = max_lse > -float("inf")
    weights = tl.math.exp2(tl.where(split_mask & has_values, lse - max_lse, -float("inf")))
    denominator = tl.sum(weights, axis=0)
    partials = tl.load(
        partial_output
        + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partials * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output
        + row * output_stride_row
        + head * output_stride_head
        + dim_offsets * output_stride_dim,
        merged,
    )


def triton_qsa_paged_sparse_gqa(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run fused sparse GQA directly over the HND paged K/V cache."""
    rows, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    output = torch.empty_like(q)
    if rows == 0:
        return output

    if os.environ.get("TRTLLM_QSA_SPARSE_SPLITK", "1") != "0":
        base_programs = rows * num_kv_heads
        if base_programs <= 4:
            block_n, target_splits, num_warps = 16, 64, 4
        elif base_programs < 32:
            block_n, target_splits, num_warps = 16, 32, 4
        elif base_programs <= 256:
            block_n, target_splits, num_warps = 64, 8, 2
        elif base_programs <= 512:
            block_n, target_splits, num_warps = 64, 4, 2
        else:
            block_n, target_splits, num_warps = 64, 1, 2

        num_tiles = triton.cdiv(selected_tokens.shape[1], block_n)
        max_useful_splits = 1 << (num_tiles.bit_length() - 1)
        num_splits = min(max_useful_splits, target_splits)
        if num_splits == 1:
            partial_output = output
            partial_lse = output
        else:
            partial_output = torch.empty(
                (num_splits, *q.shape),
                dtype=torch.float32,
                device=q.device,
            )
            partial_lse = torch.empty(
                (num_splits, rows, num_q_heads),
                dtype=torch.float32,
                device=q.device,
            )
        _qsa_paged_sparse_gqa_splitk_kernel[(rows, num_kv_heads, num_splits)](
            q,
            k_cache,
            v_cache,
            block_table,
            selected_tokens,
            request_indices,
            partial_output,
            partial_lse,
            output,
            softmax_scale,
            rows,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            selected_tokens.stride(0),
            selected_tokens.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            NUM_CACHE_PAGES=k_cache.shape[0],
            NUM_REQUESTS=block_table.shape[0],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            NUM_QUERY_HEADS=num_q_heads,
            TOKENS_PER_BLOCK=tokens_per_block,
            TOPK=selected_tokens.shape[1],
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim,
            num_warps=num_warps,
            num_stages=2,
        )
        if num_splits == 1:
            return output
        _qsa_merge_splitk_kernel[(rows, num_q_heads)](
            partial_output,
            partial_lse,
            output,
            rows,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            NUM_QUERY_HEADS=num_q_heads,
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            HEAD_DIM=head_dim,
            num_warps=2,
            num_stages=1,
        )
        return output

    block_n = 64 if rows <= 64 else 32
    _qsa_paged_sparse_gqa_kernel[(rows, num_kv_heads)](
        q,
        k_cache,
        v_cache,
        block_table,
        selected_tokens,
        request_indices,
        output,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        selected_tokens.stride(0),
        selected_tokens.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        GROUP_SIZE=group_size,
        TOKENS_PER_BLOCK=tokens_per_block,
        TOPK=selected_tokens.shape[1],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = [
    "triton_qsa_decode_pre_indexer",
    "triton_qsa_paged_index_scores",
    "triton_qsa_paged_sparse_gqa",
    "triton_expand_qsa_block_indices",
]
