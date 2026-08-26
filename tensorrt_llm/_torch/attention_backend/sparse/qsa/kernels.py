# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for QSA sparse attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
    block_n = 64
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
    block_n = 64 if rows <= 64 else 32
    output = torch.empty_like(q)
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
    "triton_qsa_paged_index_scores",
    "triton_qsa_paged_sparse_gqa",
    "triton_expand_qsa_block_indices",
]
