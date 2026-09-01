# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for QSA sparse attention."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger


def _qsa_pdl_enabled(rows: int) -> bool:
    # PDL helps the two- and four-row speculative target graphs. Larger IFB
    # batches already expose enough split-K parallelism to hide the merge.
    return rows <= 4 and os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1" and get_sm_version() >= 90


@triton.jit
def _qsa_unscale_block_table_kernel(
    scaled_block_table,
    block_table,
    num_rows,
    scaled_stride_row: tl.constexpr,
    scaled_stride_column: tl.constexpr,
    block_stride_row: tl.constexpr,
    block_stride_column: tl.constexpr,
    PAGE_INDEX_SCALE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = (row < num_rows) & (columns < NUM_COLUMNS)
    scaled = tl.load(
        scaled_block_table + row * scaled_stride_row + columns * scaled_stride_column,
        mask=valid,
        other=0,
    )
    tl.store(
        block_table + row * block_stride_row + columns * block_stride_column,
        scaled // PAGE_INDEX_SCALE,
        mask=valid,
    )


def triton_qsa_unscale_block_table(
    *,
    scaled_block_table: torch.Tensor,
    block_table: torch.Tensor,
    page_index_scale: int,
) -> None:
    """Recover lifecycle slot IDs from V2's scaled attention page table."""
    if scaled_block_table.shape != block_table.shape:
        raise ValueError("QSA scaled and slot block tables must have matching shapes")
    if scaled_block_table.ndim != 2:
        raise ValueError("QSA block tables must be two-dimensional")
    if scaled_block_table.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise ValueError("QSA block tables must use int32 storage")
    if not scaled_block_table.is_cuda or not block_table.is_cuda:
        raise ValueError("QSA block-table conversion requires CUDA tensors")
    if page_index_scale <= 0:
        raise ValueError(f"QSA page-index scale must be positive, got {page_index_scale}")
    num_rows, num_columns = scaled_block_table.shape
    if num_rows == 0 or num_columns == 0:
        return
    block_size = triton.next_power_of_2(num_columns)
    _qsa_unscale_block_table_kernel[(num_rows,)](
        scaled_block_table,
        block_table,
        num_rows,
        scaled_block_table.stride(0),
        scaled_block_table.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        PAGE_INDEX_SCALE=page_index_scale,
        NUM_COLUMNS=num_columns,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )


@triton.jit
def _qsa_decode_token_mapping_kernel(
    kv_lens,
    seq_lens,
    request_indices,
    logical_positions,
    sequence_lengths,
    visible_blocks,
    num_rows,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_rows
    kv_len = tl.load(kv_lens + offsets, mask=valid, other=0)
    cached_lens = kv_len - tl.load(
        seq_lens + offsets,
        mask=valid,
        other=0,
    )
    tl.store(request_indices + offsets, offsets, mask=valid)
    tl.store(logical_positions + offsets, cached_lens.to(tl.int64), mask=valid)
    tl.store(sequence_lengths + offsets, kv_len, mask=valid)
    tl.store(
        visible_blocks + offsets,
        ((cached_lens + 1) // COMPRESS_RATIO).to(tl.int32),
        mask=valid,
    )


def triton_qsa_decode_token_mapping(
    *,
    kv_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    visible_blocks: torch.Tensor,
    compress_ratio: int,
) -> None:
    """Build request and position metadata for one-token-per-request decode."""
    rows = kv_lens.numel()
    if rows == 0:
        return
    if seq_lens.numel() != rows:
        raise ValueError("QSA decode KV and sequence lengths must have matching shapes")
    outputs = (
        request_indices,
        logical_positions,
        sequence_lengths,
        visible_blocks,
    )
    if any(output.numel() < rows for output in outputs):
        raise ValueError("QSA decode token-mapping outputs are too small")
    if not all(tensor.is_cuda for tensor in (kv_lens, seq_lens, *outputs)):
        raise ValueError("QSA decode token mapping requires CUDA tensors")
    if compress_ratio <= 0:
        raise ValueError(f"QSA compression ratio must be positive, got {compress_ratio}")

    # One fixed specialization covers all decode graph batch sizes. It avoids
    # cold-start compilation per graph shape and is faster even for small rows.
    block_size = 256
    _qsa_decode_token_mapping_kernel[(triton.cdiv(rows, block_size),)](
        kv_lens,
        seq_lens,
        request_indices,
        logical_positions,
        sequence_lengths,
        visible_blocks,
        rows,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


@triton.jit
def _qsa_paged_kv_store_kernel(
    k,
    v,
    k_cache,
    v_cache,
    request_indices,
    logical_positions,
    block_table,
    num_rows,
    k_stride_row: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_row: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_dim: tl.constexpr,
    k_cache_stride_page: tl.constexpr,
    k_cache_stride_head: tl.constexpr,
    k_cache_stride_token: tl.constexpr,
    k_cache_stride_dim: tl.constexpr,
    v_cache_stride_page: tl.constexpr,
    v_cache_stride_head: tl.constexpr,
    v_cache_stride_token: tl.constexpr,
    v_cache_stride_dim: tl.constexpr,
    block_table_stride_request: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_DIM)
    valid = (row < num_rows) & (head < NUM_KV_HEADS) & (dims < HEAD_DIM)

    request = tl.load(request_indices + row).to(tl.int32)
    logical = tl.load(logical_positions + row).to(tl.int32)
    page_column = logical // TOKENS_PER_BLOCK
    token_in_page = logical % TOKENS_PER_BLOCK
    valid_page_lookup = (
        (request >= 0)
        & (request < NUM_REQUESTS)
        & (logical >= 0)
        & (page_column < PAGE_TABLE_WIDTH)
    )
    page = tl.load(
        block_table + request * block_table_stride_request + page_column * block_table_stride_page,
        mask=valid_page_lookup,
        other=-1,
    ).to(tl.int64)
    valid &= (page >= 0) & (page < NUM_PAGES)

    k_values = tl.load(
        k + row * k_stride_row + head * k_stride_head + dims * k_stride_dim,
        mask=valid,
        other=0.0,
    )
    v_values = tl.load(
        v + row * v_stride_row + head * v_stride_head + dims * v_stride_dim,
        mask=valid,
        other=0.0,
    )
    tl.store(
        k_cache
        + page * k_cache_stride_page
        + head * k_cache_stride_head
        + token_in_page * k_cache_stride_token
        + dims * k_cache_stride_dim,
        k_values,
        mask=valid,
    )
    tl.store(
        v_cache
        + page * v_cache_stride_page
        + head * v_cache_stride_head
        + token_in_page * v_cache_stride_token
        + dims * v_cache_stride_dim,
        v_values,
        mask=valid,
    )


def triton_qsa_paged_kv_store(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    block_table: torch.Tensor,
    tokens_per_block: int,
) -> None:
    """Store one or more QSA K/V rows into an HND paged cache."""
    if k.shape != v.shape or k.ndim != 3:
        raise ValueError("QSA K and V inputs must have matching [tokens, heads, dim] shapes")
    if k_cache.shape != v_cache.shape or k_cache.ndim != 4:
        raise ValueError("QSA K and V caches must have matching [pages, heads, tokens, dim] shapes")
    rows, num_kv_heads, head_dim = k.shape
    if k_cache.shape[1] != num_kv_heads or k_cache.shape[3] != head_dim:
        raise ValueError("QSA K/V inputs and caches have incompatible head geometry")
    if k_cache.shape[2] != tokens_per_block:
        raise ValueError("QSA K/V cache token dimension does not match tokens_per_block")
    if request_indices.numel() < rows or logical_positions.numel() < rows:
        raise ValueError("QSA K/V store request metadata is shorter than its token inputs")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("QSA K/V store requires a two-dimensional int32 block table")
    tensors = (
        k,
        v,
        k_cache,
        v_cache,
        request_indices,
        logical_positions,
        block_table,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("QSA paged K/V store requires CUDA tensors")
    if rows == 0:
        return

    block_dim = triton.next_power_of_2(head_dim)
    _qsa_paged_kv_store_kernel[(rows, num_kv_heads)](
        k,
        v,
        k_cache,
        v_cache,
        request_indices,
        logical_positions,
        block_table,
        rows,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
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
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        TOKENS_PER_BLOCK=tokens_per_block,
        NUM_PAGES=k_cache.shape[0],
        NUM_REQUESTS=block_table.shape[0],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        BLOCK_DIM=block_dim,
        num_warps=8 if block_dim >= 256 else 4,
    )


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
def _qsa_prefill_compress_kernel(
    logical_positions,
    request_indices,
    block_table,
    index_cache,
    position_cache,
    k_norm_weight,
    cos_sin,
    eps,
    block_table_stride_request: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    cache_stride_page: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    position_cache_stride_page: tl.constexpr,
    position_cache_stride_token: tl.constexpr,
    position_cache_stride_axis: tl.constexpr,
    cos_sin_stride: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    """Compress completed QSA groups without data-dependent Torch indexing."""
    row = tl.program_id(0)
    logical = tl.load(logical_positions + row).to(tl.int64)
    if (logical + 1) % COMPRESS_RATIO == 0:
        request = tl.load(request_indices + row).to(tl.int64)
        dims = tl.arange(0, HEAD_DIM)
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
        # Match the eager reference's BF16 materialization between FP32
        # pooling and Gemma RMSNorm.
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
        compressed = tl.reshape(
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
        anchor_logical_page = logical // TOKENS_PER_BLOCK
        anchor_token = logical % TOKENS_PER_BLOCK
        anchor_page = tl.load(
            block_table
            + request * block_table_stride_request
            + anchor_logical_page * block_table_stride_page
        ).to(tl.int64)
        tl.store(
            index_cache
            + anchor_page * cache_stride_page
            + anchor_token * cache_stride_token
            + dims * cache_stride_dim,
            compressed,
        )


def triton_qsa_prefill_compress(
    *,
    logical_positions: torch.Tensor,
    request_indices: torch.Tensor,
    block_table: torch.Tensor,
    index_cache: torch.Tensor,
    position_cache: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    tokens_per_block: int,
    compress_ratio: int,
    mrope_section: tuple[int, int, int] | None,
) -> None:
    """Compress QSA prefill groups with a fixed-shape, graph-safe launch."""
    rows = logical_positions.numel()
    if request_indices.shape != logical_positions.shape:
        raise ValueError("QSA prefill request indices must match logical positions")
    if index_cache.ndim != 4 or index_cache.shape[2] != 1:
        raise ValueError("QSA prefill index cache must have one KV head")
    if position_cache.ndim != 3 or position_cache.shape[2] < 3:
        raise ValueError("QSA prefill position cache must contain three axes")
    if cos_sin.ndim != 2 or cos_sin.shape[1] % 2 != 0:
        raise ValueError("QSA RoPE cache must be a packed 2D cos/sin tensor")
    head_dim = index_cache.shape[3]
    rotary_pairs = cos_sin.shape[1] // 2
    if 4 * rotary_pairs != head_dim:
        raise ValueError("QSA fused prefill requires half-width rotary embedding")
    head_dim_is_power_of_two = head_dim > 0 and (head_dim & (head_dim - 1)) == 0
    ratio_is_power_of_two = compress_ratio > 0 and (compress_ratio & (compress_ratio - 1)) == 0
    if not head_dim_is_power_of_two or not ratio_is_power_of_two:
        raise ValueError("QSA fused prefill requires power-of-two head and compression widths")
    if rows == 0:
        return
    section = mrope_section if mrope_section is not None else (0, 0, 0)
    _qsa_prefill_compress_kernel[(rows,)](
        logical_positions,
        request_indices,
        block_table,
        index_cache,
        position_cache,
        k_norm_weight,
        cos_sin,
        eps,
        block_table.stride(0),
        block_table.stride(1),
        index_cache.stride(0),
        index_cache.stride(1),
        index_cache.stride(3),
        position_cache.stride(0),
        position_cache.stride(1),
        position_cache.stride(2),
        cos_sin.stride(0),
        HEAD_DIM=head_dim,
        ROTARY_PAIRS=rotary_pairs,
        TOKENS_PER_BLOCK=tokens_per_block,
        COMPRESS_RATIO=compress_ratio,
        IS_MROPE=mrope_section is not None,
        MROPE_H=section[1],
        MROPE_W=section[2],
        num_warps=1,
    )


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
    COLUMN_TILE: tl.constexpr,
    COUNT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_TILE + tl.arange(0, COLUMN_TILE)
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

    # The tail's placement depends on the whole row's valid block count, so a
    # column tile narrower than the row re-derives it over its own range rather
    # than reusing the output columns.
    count_columns = tl.arange(0, COUNT_BLOCK)
    valid_blocks = tl.load(
        block_indices + row * block_stride + count_columns,
        mask=count_columns < BLOCK_TOPK,
        other=-1,
    )
    valid_token_count = tl.minimum(
        tl.sum(((count_columns < BLOCK_TOPK) & (valid_blocks >= 0)).to(tl.int32), axis=0)
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


# One program per row leaves a single CTA on the device at decode row counts,
# where the expansion is grid-limited rather than internally bound. Splitting a
# row's output columns across programs costs one extra read of that row's block
# indices per program, so it only pays while the row count alone cannot fill the
# device.
_EXPAND_COLUMN_TILE = 256
_EXPAND_ROWS_FILLING_DEVICE = 128


def _expand_launch(rows: int, final_topk: int) -> tuple[int, int]:
    """Output columns handled by one expansion program, and its warp count."""
    whole_row = triton.next_power_of_2(final_topk)
    if rows >= _EXPAND_ROWS_FILLING_DEVICE:
        return whole_row, 8
    return min(_EXPAND_COLUMN_TILE, whole_row), 4


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
    column_tile, num_warps = _expand_launch(rows, final_topk)
    _expand_qsa_block_indices_kernel[(rows, triton.cdiv(final_topk, column_tile))](
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
        COLUMN_TILE=column_tile,
        COUNT_BLOCK=triton.next_power_of_2(block_topk),
        num_warps=num_warps,
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
    TILES_PER_PROGRAM: tl.constexpr,
    ONLY_VISIBLE_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(request_indices + row).to(tl.int64)
    query_position = tl.load(query_positions + row)
    visible_blocks = (query_position + 1) // COMPRESS_RATIO
    first_tile = tl.program_id(1) * TILES_PER_PROGRAM
    if ONLY_VISIBLE_BLOCKS and first_tile * BLOCK_N >= visible_blocks:
        return

    last_tile = tl.minimum(
        first_tile + TILES_PER_PROGRAM,
        tl.cdiv(
            visible_blocks if ONLY_VISIBLE_BLOCKS else MAX_COMPRESSED_BLOCKS,
            BLOCK_N,
        ),
    )
    dim_offsets = tl.arange(0, INDEX_HEAD_DIM)
    head_offsets = tl.arange(0, BLOCK_H)
    query_values = tl.load(
        q
        + row * q_stride_row
        + dim_offsets[:, None] * q_stride_dim
        + head_offsets[None, :] * q_stride_head,
        mask=head_offsets[None, :] < NUM_INDEX_HEADS,
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(first_tile, last_tile, num_stages=2):
        block_columns = tile * BLOCK_N + column_offsets
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
        keys = tl.load(
            index_cache
            + physical_pages[:, None] * cache_stride_page
            + token_in_page[:, None] * cache_stride_token
            + dim_offsets[None, :] * cache_stride_dim,
            mask=valid_blocks[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        per_head_scores = tl.dot(keys, query_values, out_dtype=tl.float32)
        scores = tl.sum(
            tl.where(
                head_offsets[None, :] < NUM_INDEX_HEADS,
                tl.maximum(per_head_scores, 0.0),
                0.0,
            ),
            axis=1,
        )
        output_ptrs = output + row * output_stride_row + block_columns * output_stride_block
        if ONLY_VISIBLE_BLOCKS:
            tl.store(output_ptrs, scores * score_scale, mask=valid_blocks)
        else:
            tl.store(
                output_ptrs,
                tl.where(valid_blocks, scores * score_scale, -float("inf")),
                mask=block_columns < MAX_COMPRESSED_BLOCKS,
            )


@triton.jit
def _qsa_paged_index_scores_qtile_kernel(
    q,
    index_cache,
    block_table,
    query_positions,
    request_indices,
    output,
    score_scale,
    num_rows,
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
    QUERY_TILE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    ONLY_VISIBLE_BLOCKS: tl.constexpr,
):
    row_base = tl.program_id(0) * QUERY_TILE
    first_tile = tl.program_id(1) * TILES_PER_PROGRAM

    # One dot column per (query row, index head) pair, so the tile carries
    # QUERY_TILE whole rows instead of one row and NUM_INDEX_HEADS of padding.
    dot_columns = tl.arange(0, BLOCK_H)
    column_row = dot_columns // NUM_INDEX_HEADS
    column_head = dot_columns % NUM_INDEX_HEADS
    rows = row_base + column_row
    present = rows < num_rows

    requests = tl.load(request_indices + rows, mask=present, other=-1).to(tl.int64)
    positions = tl.load(query_positions + rows, mask=present, other=0)
    visible_blocks = tl.where(present, (positions + 1) // COMPRESS_RATIO, 0)
    if ONLY_VISIBLE_BLOCKS and first_tile * BLOCK_N >= tl.max(visible_blocks):
        return

    dim_offsets = tl.arange(0, INDEX_HEAD_DIM)
    query_values = tl.load(
        q
        + rows[None, :] * q_stride_row
        + dim_offsets[:, None] * q_stride_dim
        + column_head[None, :] * q_stride_head,
        mask=present[None, :],
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)

    # A gather is only shareable across rows that page through the same block
    # table, so score one request at a time. Context rows arrive packed per
    # request, which leaves a single group for all but the boundary tiles.
    previous_requests = tl.load(
        request_indices + rows - 1,
        mask=present & (rows > 0),
        other=-1,
    ).to(tl.int64)
    group_leaders = (
        present & (column_head == 0) & ((column_row == 0) | (requests != previous_requests))
    )
    pending = present
    for _ in tl.range(0, tl.sum(group_leaders.to(tl.int32))):
        leader = tl.min(tl.where(pending, dot_columns, BLOCK_H))
        request = tl.max(tl.where(dot_columns == leader, requests, 0))
        group = pending & (requests == request)
        pending = pending & (requests != request)
        group_visible = tl.max(tl.where(group, visible_blocks, 0))
        last_tile = tl.minimum(
            first_tile + TILES_PER_PROGRAM,
            tl.cdiv(
                group_visible if ONLY_VISIBLE_BLOCKS else MAX_COMPRESSED_BLOCKS,
                BLOCK_N,
            ),
        )
        for tile in tl.range(first_tile, last_tile, num_stages=2):
            block_columns = tile * BLOCK_N + column_offsets
            valid_blocks = (block_columns < MAX_COMPRESSED_BLOCKS) & (block_columns < group_visible)
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
            keys = tl.load(
                index_cache
                + physical_pages[:, None] * cache_stride_page
                + token_in_page[:, None] * cache_stride_token
                + dim_offsets[None, :] * cache_stride_dim,
                mask=valid_blocks[:, None],
                other=0.0,
                eviction_policy="evict_first",
            )
            positive_scores = tl.maximum(
                tl.dot(keys, query_values, out_dtype=tl.float32),
                0.0,
            )
            for lane in tl.static_range(QUERY_TILE):
                row_columns = group & (column_row == lane)
                scored = tl.max(row_columns.to(tl.int32)) > 0
                row_visible = tl.max(tl.where(row_columns, visible_blocks, 0))
                scores = tl.sum(
                    tl.where(row_columns[None, :], positive_scores, 0.0),
                    axis=1,
                )
                output_ptrs = (
                    output
                    + (row_base + lane) * output_stride_row
                    + block_columns * output_stride_block
                )
                row_valid = block_columns < row_visible
                if ONLY_VISIBLE_BLOCKS:
                    tl.store(
                        output_ptrs,
                        scores * score_scale,
                        mask=scored & row_valid,
                    )
                else:
                    tl.store(
                        output_ptrs,
                        tl.where(row_valid, scores * score_scale, -float("inf")),
                        mask=scored & (block_columns < MAX_COMPRESSED_BLOCKS),
                    )


# ``tl.dot`` needs at least 16 columns on SM90+, so a narrower model dimension
# leaves the rest of every tensor-core tile computed and discarded.
_INDEX_SCORE_TILE_WIDTH = 16


def _qsa_index_scores_query_tile(num_index_heads: int) -> int:
    """Query rows packed into the columns of one paged index-scoring dot.

    Neighbouring context rows of a request read overlapping causal prefixes of
    the same compressed keys, so filling the columns ``num_index_heads`` leaves
    idle with whole query rows both fills the tensor-core tile and lets one
    gather serve every packed row. Only head counts that divide the tile width
    keep a column index separable into ``(row, head)``.
    """
    if num_index_heads <= 0 or _INDEX_SCORE_TILE_WIDTH % num_index_heads != 0:
        return 1
    return _INDEX_SCORE_TILE_WIDTH // num_index_heads


def triton_qsa_paged_index_scores(
    *,
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
    only_visible_blocks: bool = False,
    context_rows: bool = False,
) -> torch.Tensor:
    """Score compressed keys directly in the paged side cache.

    ``only_visible_blocks`` leaves columns at and beyond each row's causal
    boundary unspecified. Callers may enable it only when their consumer uses
    the same per-row boundary and cannot inspect those columns.

    ``context_rows`` declares that consecutive rows mostly share a request, so
    one gather of compressed keys can serve a tile of query rows. Generation
    batches hold one row per request and would only lose parallelism.
    """
    rows, num_index_heads, index_head_dim = q.shape
    max_compressed_blocks = block_table.shape[1] * tokens_per_block // compress_ratio
    output = torch.empty(
        (rows, max_compressed_blocks),
        dtype=torch.float32,
        device=q.device,
    )
    # GB300 tuning uses fine-grained parallelism for decode-sized batches and
    # reuses each query across multiple score tiles for prefill-sized batches.
    # The 256-row boundary also covers the largest decode CUDA graph.
    #
    # A decode row scores its whole causal prefix through one program per score
    # tile, so the tile width sets both the grid and the length of the block
    # table -> compressed key gather each program waits on. At 32 columns the
    # grid covers half the device and the kernel is grid-limited; narrower tiles
    # are monotonically faster down to 8 and then flatten, so 8 takes almost all
    # of the win at a quarter of the programs a 4-column tile would launch.
    is_prefill_sized = rows > 256
    block_n = 64 if is_prefill_sized else 8
    tiles_per_program = 8 if is_prefill_sized else 1
    query_tile = 1
    if (
        context_rows
        and is_prefill_sized
        and os.environ.get("TRTLLM_QSA_INDEX_SCORE_QUERY_TILE", "1") != "0"
    ):
        query_tile = _qsa_index_scores_query_tile(num_index_heads)
    if query_tile > 1:
        # Packing rows leaves QUERY_TILE times fewer programs, so each one
        # walks a deeper column range and takes more warps to keep the GPU
        # filled. GB300 is flat from 24 to 96 tiles per program at four warps
        # and falls off a cliff at eight.
        tiles_per_program = 32
        grid = (
            triton.cdiv(rows, query_tile),
            triton.cdiv(max_compressed_blocks, block_n * tiles_per_program),
        )
        logger.info_once(
            f"QSA paged index scoring packs {query_tile} context rows per dot tile",
            key="qsa_paged_index_scores_query_tile_active",
        )
        _qsa_paged_index_scores_qtile_kernel[grid](
            q,
            index_cache,
            block_table,
            query_positions,
            request_indices,
            output,
            index_head_dim**-0.5,
            rows,
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
            QUERY_TILE=query_tile,
            BLOCK_H=query_tile * num_index_heads,
            BLOCK_N=block_n,
            TILES_PER_PROGRAM=tiles_per_program,
            ONLY_VISIBLE_BLOCKS=only_visible_blocks,
            num_warps=4,
        )
        return output
    grid = (
        rows,
        triton.cdiv(max_compressed_blocks, block_n * tiles_per_program),
    )
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
        TILES_PER_PROGRAM=tiles_per_program,
        ONLY_VISIBLE_BLOCKS=only_visible_blocks,
        num_warps=2,
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
    query_positions,
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
    NUM_CACHE_PAGES: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ONLY_VISIBLE_TOKENS: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.load(request_indices + row).to(tl.int64)
    valid_request = (request >= 0) & (request < NUM_REQUESTS)
    safe_request = tl.minimum(tl.maximum(request, 0), NUM_REQUESTS - 1)

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

    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    token_offsets = tl.arange(0, BLOCK_N)
    valid_tiles = tl.cdiv(TOPK, BLOCK_N)
    if ONLY_VISIBLE_TOKENS:
        visible_tokens = tl.load(query_positions + row) + 1
        selected_blocks = tl.minimum(visible_tokens // COMPRESS_RATIO, BLOCK_TOPK)
        valid_tokens = selected_blocks * COMPRESS_RATIO + visible_tokens % COMPRESS_RATIO
        valid_tiles = tl.cdiv(tl.minimum(valid_tokens, TOPK), BLOCK_N)

    for tile in tl.range(0, valid_tiles, num_stages=2):
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
        )
        values = tl.load(
            v_cache
            + safe_pages[:, None] * v_stride_page
            + kv_head * v_stride_head
            + token_in_page[:, None] * v_stride_token
            + dim_offsets[None, :] * v_stride_dim,
            mask=valid[:, None],
            other=0.0,
        )
        keys = keys.to(query_values.dtype)
        values = values.to(query_values.dtype)
        scores = tl.dot(query_values, keys) * softmax_scale * 1.4426950408889634
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

    normalized = tl.where(
        running_sum[:, None] > 0,
        accumulator / tl.maximum(running_sum[:, None], 1.0e-20),
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
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
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
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


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
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
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
    query_positions: torch.Tensor | None = None,
    compress_ratio: int | None = None,
) -> torch.Tensor:
    """Run fused sparse GQA directly over the HND paged K/V cache."""
    rows, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    output = torch.empty_like(q)
    if rows == 0:
        return output

    if (query_positions is None) != (compress_ratio is None):
        raise ValueError("QSA causal-prefix bounds require positions and compression ratio")
    if query_positions is not None:
        if query_positions.shape != (rows,):
            raise ValueError("QSA query positions must match sparse-attention rows")
        if compress_ratio is None or compress_ratio <= 0:
            raise ValueError("QSA compression ratio must be positive")

    base_programs = rows * num_kv_heads
    if query_positions is not None and base_programs > 512:
        final_topk = selected_tokens.shape[1]
        token_topk = final_topk - compress_ratio + 1
        if token_topk <= 0 or token_topk % compress_ratio:
            raise ValueError("QSA selected-token width is incompatible with its compression ratio")
        block_topk = token_topk // compress_ratio
        block_n = 64
        _qsa_paged_sparse_gqa_kernel[(rows, num_kv_heads)](
            q,
            k_cache,
            v_cache,
            block_table,
            selected_tokens,
            request_indices,
            query_positions.contiguous(),
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
            NUM_CACHE_PAGES=k_cache.shape[0],
            NUM_REQUESTS=block_table.shape[0],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            TOKENS_PER_BLOCK=tokens_per_block,
            TOPK=final_topk,
            BLOCK_TOPK=block_topk,
            COMPRESS_RATIO=compress_ratio,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim,
            ONLY_VISIBLE_TOKENS=True,
            num_warps=2,
            num_stages=2,
        )
        return output

    if os.environ.get("TRTLLM_QSA_SPARSE_SPLITK", "1") != "0":
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
        use_pdl = num_splits > 1 and _qsa_pdl_enabled(rows)
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
            USE_PDL=use_pdl,
            num_warps=num_warps,
            num_stages=2,
            launch_pdl=use_pdl,
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
            USE_PDL=use_pdl,
            num_warps=2,
            num_stages=1,
            launch_pdl=use_pdl,
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
        NUM_CACHE_PAGES=k_cache.shape[0],
        NUM_REQUESTS=block_table.shape[0],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        TOKENS_PER_BLOCK=tokens_per_block,
        TOPK=selected_tokens.shape[1],
        BLOCK_TOPK=selected_tokens.shape[1],
        COMPRESS_RATIO=1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        ONLY_VISIBLE_TOKENS=False,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = [
    "triton_qsa_decode_pre_indexer",
    "triton_qsa_paged_index_scores",
    "triton_qsa_paged_kv_store",
    "triton_qsa_paged_sparse_gqa",
    "triton_qsa_prefill_compress",
    "triton_expand_qsa_block_indices",
]
