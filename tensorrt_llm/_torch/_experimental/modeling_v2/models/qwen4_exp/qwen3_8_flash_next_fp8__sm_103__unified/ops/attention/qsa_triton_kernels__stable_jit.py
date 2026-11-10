# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's copies of the QSA Triton kernels for sparse GQA and the K/V append
(`tensorrt_llm._torch.attention.backends.sparse.qsa.kernels`), compiled once per head geometry.

Extents that follow the engine or the step (pool pages, block-table shape and row stride, row
count) are `do_not_specialize` and the metadata pointers `do_not_specialize_on_alignment`;
geometry, strides and launch tiles stay constexpr. Outputs are bitwise equal to the shared
launchers': the bodies differ only in unsigned range compares under `tl.assume` (decode and
store kernels) and row-major split-K partials in the 64-split bucket.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.jit_variants import (
    compile_only,
    first_time,
)
from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber
from tensorrt_llm._utils import get_sm_version

_INTEGER_DTYPES = (torch.int32, torch.int64)
# Scores use base-2 exponentials for the online softmax (the shared kernels' constant).
_LOG2_E = tl.constexpr(1.4426950408889634)


def _require_same_cuda_device(name: str, *tensors: torch.Tensor) -> None:
    if not tensors or not all(tensor.is_cuda for tensor in tensors):
        raise ValueError(f"{name} requires CUDA tensors")
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError(f"{name} tensors must be on the same CUDA device")


def _pdl_enabled(rows: int) -> bool:
    """The shared launcher's PDL rule: two- to four-row speculative graphs only."""
    return rows <= 4 and os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1" and get_sm_version() >= 90


#: Warps of a K/V append program (a store is a move: the warp count cannot change the result).
_KV_STORE_WARPS = 4


def splitk_launch_config(rows: int, num_kv_heads: int, selected_width: int) -> tuple[int, int, int]:
    """(BLOCK_N, splits, warps) by `rows * num_kv_heads`: the shared launcher's five buckets,
    kept as they are so every launch reduces in the shared kernel's order."""
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
    num_tiles = triton.cdiv(selected_width, block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    return block_n, min(max_useful_splits, target_splits), num_warps


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "num_pages",
        "num_requests",
        "page_table_width",
        "block_table_stride_request",
    ],
    do_not_specialize_on_alignment=["request_indices", "logical_positions", "block_table"],
)
def _qsa_kv_store_stable_kernel(
    k,
    v,
    k_cache,
    v_cache,
    request_indices,
    logical_positions,
    block_table,
    num_rows,
    num_pages,
    num_requests,
    page_table_width,
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
    block_table_stride_request,
    block_table_stride_page: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    # Extents are positive and below 2^31 (the launcher refuses empty ones), so each
    # `0 <= x < extent` test is one unsigned compare.
    tl.assume(num_pages > 0)
    tl.assume(num_requests > 0)
    tl.assume(page_table_width > 0)
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_DIM)
    valid = (row < num_rows) & (head < NUM_KV_HEADS) & (dims < HEAD_DIM)

    request = tl.load(request_indices + row).to(tl.int32)
    logical = tl.load(logical_positions + row).to(tl.int32)
    page_column = logical // TOKENS_PER_BLOCK
    token_in_page = logical % TOKENS_PER_BLOCK
    # `request` in [0, num_requests), `logical >= 0` and `page_column < page_table_width`.
    valid_page_lookup = (request.to(tl.uint32) < num_requests.to(tl.uint32)) & (
        logical.to(tl.uint32) < (page_table_width * TOKENS_PER_BLOCK).to(tl.uint32)
    )
    page32 = tl.load(
        block_table + request * block_table_stride_request + page_column * block_table_stride_page,
        mask=valid_page_lookup,
        other=-1,
    )
    valid &= page32.to(tl.uint32) < num_pages.to(tl.uint32)
    page = page32.to(tl.int64)

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


def qsa_paged_kv_store(
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
    """`triton_qsa_paged_kv_store`'s contract and checks, on `_qsa_kv_store_stable_kernel`."""
    if k.shape != v.shape or k.ndim != 3:
        raise ValueError("QSA K and V inputs must have matching [tokens, heads, dim] shapes")
    if k_cache.shape != v_cache.shape or k_cache.ndim != 4:
        raise ValueError("QSA K and V caches must have matching [pages, heads, tokens, dim] shapes")
    rows, num_kv_heads, head_dim = k.shape
    if num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError("QSA K/V store requires non-empty head geometry")
    if k_cache.shape[1] != num_kv_heads or k_cache.shape[3] != head_dim:
        raise ValueError("QSA K/V inputs and caches have incompatible head geometry")
    if k_cache.shape[2] != tokens_per_block:
        raise ValueError("QSA K/V cache token dimension does not match tokens_per_block")
    if tokens_per_block <= 0:
        raise ValueError("QSA tokens_per_block must be positive")
    if k.dtype != v.dtype or k_cache.dtype != v_cache.dtype:
        raise ValueError("QSA K/V pairs must use matching dtypes")
    if k.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("QSA K/V inputs must use BF16 or FP16")
    if k_cache.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        raise ValueError("QSA K/V cache must use BF16, FP16, or FP8 E4M3")
    if request_indices.shape != (rows,) or logical_positions.shape != (rows,):
        raise ValueError("QSA K/V store metadata must have one entry per input row")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("QSA K/V store requires a two-dimensional int32 block table")
    if (
        request_indices.dtype not in _INTEGER_DTYPES
        or logical_positions.dtype not in _INTEGER_DTYPES
    ):
        raise ValueError("QSA K/V store request metadata must use integer storage")
    _require_same_cuda_device(
        "QSA paged K/V store",
        k,
        v,
        k_cache,
        v_cache,
        request_indices,
        logical_positions,
        block_table,
    )
    if rows == 0:
        return
    if k_cache.shape[0] == 0 or block_table.shape[0] == 0 or block_table.shape[1] == 0:
        raise ValueError("QSA K/V store requires non-empty cache and block-table geometry")

    block_dim = triton.next_power_of_2(head_dim)
    _qsa_kv_store_stable_kernel[(rows, num_kv_heads)](
        k,
        v,
        k_cache,
        v_cache,
        request_indices,
        logical_positions,
        block_table,
        rows,
        k_cache.shape[0],
        block_table.shape[0],
        block_table.shape[1],
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
        BLOCK_DIM=block_dim,
        num_warps=_KV_STORE_WARPS,
    )


@triton.jit(
    do_not_specialize=[
        "num_cache_pages",
        "num_requests",
        "page_table_width",
        "block_table_stride_request",
    ],
    do_not_specialize_on_alignment=[
        "block_table",
        "selected_tokens",
        "request_indices",
        "query_positions",
    ],
)
def _qsa_prefill_pair_stable_kernel(
    q,
    k_cache,
    v_cache,
    block_table,
    selected_tokens,
    request_indices,
    query_positions,
    output,
    softmax_scale,
    num_cache_pages,
    num_requests,
    page_table_width,
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
    block_table_stride_request,
    block_table_stride_page: tl.constexpr,
    selected_stride_row: tl.constexpr,
    selected_stride_token: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_head: tl.constexpr,
    output_stride_dim: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    tl.assume(num_cache_pages > 0)
    tl.assume(num_requests > 0)
    tl.assume(page_table_width > 0)
    request = tl.load(request_indices + row).to(tl.int64)
    # Signed range tests here, as in the shared kernel (the unsigned form is slower over e4m3).
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

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
            & (logical_pages < page_table_width)
        )
        pages32 = tl.load(
            block_table
            + safe_request * block_table_stride_request
            + tl.minimum(logical_pages, page_table_width - 1) * block_table_stride_page,
            mask=valid,
            other=-1,
        )
        physical_pages = pages32.to(tl.int64)
        valid &= (physical_pages >= 0) & (physical_pages < num_cache_pages)
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
        scores = tl.dot(query_values, keys) * softmax_scale * _LOG2_E
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


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "num_cache_pages",
        "num_requests",
        "page_table_width",
        "block_table_stride_request",
    ],
    do_not_specialize_on_alignment=["block_table", "selected_tokens", "request_indices"],
)
def _qsa_splitk_stable_kernel(
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
    num_cache_pages,
    num_requests,
    page_table_width,
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
    block_table_stride_request,
    block_table_stride_page: tl.constexpr,
    selected_stride_row: tl.constexpr,
    selected_stride_token: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_head: tl.constexpr,
    output_stride_dim: tl.constexpr,
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
    ROW_MAJOR: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    if USE_PDL:
        gdc_wait_with_memory_clobber()
    tl.assume(num_cache_pages > 0)
    tl.assume(num_requests > 0)
    tl.assume(page_table_width > 0)
    request = tl.load(request_indices + row).to(tl.int64)
    valid_request = request.to(tl.uint64) < num_requests.to(tl.uint64)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    # A selected token is resident-addressable when `0 <= t` and `t // TOKENS_PER_BLOCK <
    # page_table_width`: one unsigned compare against the table's token span.
    token_span = (page_table_width * TOKENS_PER_BLOCK).to(tl.uint32)

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
            valid_request & (selected_columns < TOPK) & (logical_tokens.to(tl.uint32) < token_span)
        )
        pages32 = tl.load(
            block_table
            + safe_request * block_table_stride_request
            + tl.minimum(logical_pages, page_table_width - 1) * block_table_stride_page,
            mask=valid,
            other=-1,
        )
        valid &= pages32.to(tl.uint32) < num_cache_pages.to(tl.uint32)
        safe_pages = tl.maximum(pages32.to(tl.int64), 0)

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
        scores *= softmax_scale * _LOG2_E
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
        # Partial slot of (split, row): row-major in the 64-split bucket, split-major (the shared
        # layout) otherwise. Same values, merged in the same order.
        if ROW_MAJOR:
            slot = row * NUM_SPLITS + split
        else:
            slot = split * num_rows + row
        tl.store(
            partial_output
            + (slot * NUM_QUERY_HEADS + query_heads[:, None]) * HEAD_DIM
            + dim_offsets[None, :],
            normalized,
            mask=output_mask,
        )
        tl.store(
            partial_lse + slot * NUM_QUERY_HEADS + query_heads,
            partial_lse_values,
            mask=head_offsets < GROUP_SIZE,
        )
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_stable_kernel(
    partial_output,
    partial_lse,
    output,
    output_gate,
    num_rows,
    gate_stride_row: tl.constexpr,
    gate_stride_head: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_head: tl.constexpr,
    output_stride_dim: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_PDL: tl.constexpr,
    FUSE_OUTPUT_GATE: tl.constexpr,
    ROW_MAJOR: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    if USE_PDL:
        gdc_wait_with_memory_clobber()
        tl.extra.cuda.gdc_launch_dependents()
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    if ROW_MAJOR:
        slots = row * NUM_SPLITS + split_offsets
    else:
        slots = split_offsets * num_rows + row
    lse = tl.load(
        partial_lse + slots * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse, axis=0)
    has_values = max_lse > -float("inf")
    weights = tl.math.exp2(tl.where(split_mask & has_values, lse - max_lse, -float("inf")))
    denominator = tl.sum(weights, axis=0)
    partials = tl.load(
        partial_output
        + (slots[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partials * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    if FUSE_OUTPUT_GATE:
        # The merge-then-gate contract: the attention result is rounded to its output dtype
        # before the fp32 gate.
        attention = merged.to(output.dtype.element_ty).to(tl.float32)
        gate = tl.load(
            output_gate + row * gate_stride_row + head * gate_stride_head + dim_offsets
        ).to(tl.float32)
        merged = attention * tl.sigmoid(gate)
    tl.store(
        output
        + row * output_stride_row
        + head * output_stride_head
        + dim_offsets * output_stride_dim,
        merged,
    )


def can_fuse_splitk_output_gate(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    selected_tokens: torch.Tensor,
    output_gate: torch.Tensor,
) -> bool:
    """Whether the split-K merge can consume an attention output gate (the shared rule)."""
    if q.ndim != 3 or selected_tokens.ndim != 2 or selected_tokens.shape[1] <= 0:
        return False
    rows, num_q_heads, head_dim = q.shape
    if selected_tokens.shape[0] != rows or k_cache.ndim != 4:
        return False
    if not output_gate.is_cuda or output_gate.device != q.device or output_gate.dtype != q.dtype:
        return False
    if output_gate.stride(-1) != 1:
        return False
    if output_gate.ndim == 3:
        if output_gate.shape != q.shape:
            return False
    elif output_gate.ndim == 2:
        if output_gate.shape != (rows, num_q_heads * head_dim):
            return False
    else:
        return False
    _, num_splits, _ = splitk_launch_config(rows, k_cache.shape[1], selected_tokens.shape[1])
    return num_splits > 1


def qsa_paged_sparse_gqa(
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
    output_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """`triton_qsa_paged_sparse_gqa`'s contract, checks and branch choice, on these kernels."""
    if q.ndim != 3:
        raise ValueError("QSA queries must have shape [rows, query_heads, head_dim]")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA K/V caches must have matching HND paged shapes")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("QSA sparse GQA requires a two-dimensional int32 block table")
    if selected_tokens.ndim != 2 or selected_tokens.dtype != torch.int32:
        raise ValueError("QSA selected tokens must be a two-dimensional int32 tensor")
    rows, num_q_heads, head_dim = q.shape
    if selected_tokens.shape[0] != rows or selected_tokens.shape[1] <= 0:
        raise ValueError("QSA selected-token rows must match non-empty query rows")
    if request_indices.shape != (rows,) or request_indices.dtype not in _INTEGER_DTYPES:
        raise ValueError("QSA request indices must have one integer entry per row")
    if not request_indices.is_contiguous():
        raise ValueError("QSA request indices must be contiguous")
    if tokens_per_block <= 0 or k_cache.shape[2] != tokens_per_block:
        raise ValueError("QSA cache token dimension must match positive tokens_per_block")
    if k_cache.shape[3] != head_dim or k_cache.dtype != v_cache.dtype:
        raise ValueError("QSA query and K/V cache head geometry is incompatible")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("QSA queries must use BF16 or FP16")
    if k_cache.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        raise ValueError("QSA K/V cache must use BF16, FP16, or FP8 E4M3")
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError("QSA sparse GQA requires a power-of-two head dimension")
    num_kv_heads = k_cache.shape[1]
    if num_kv_heads <= 0 or num_q_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("QSA query heads must be divisible by non-zero KV heads")
    if k_cache.shape[0] <= 0 or block_table.shape[0] <= 0 or block_table.shape[1] <= 0:
        raise ValueError("QSA sparse GQA requires non-empty cache and block-table geometry")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("QSA sparse GQA softmax scale must be finite and positive")
    _require_same_cuda_device(
        "QSA paged sparse GQA",
        q,
        k_cache,
        v_cache,
        block_table,
        selected_tokens,
        request_indices,
    )
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    # Contiguous at every row count: the output strides are kernel constexprs, and `empty_like`
    # would keep a one-row q's (dense) strides and compile a variant of its own.
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if rows == 0:
        return output
    if output_gate is not None and not can_fuse_splitk_output_gate(
        q, k_cache, selected_tokens, output_gate
    ):
        raise ValueError("QSA output-gate fusion requires a compatible split-K merge")

    if (query_positions is None) != (compress_ratio is None):
        raise ValueError("QSA causal-prefix bounds require positions and compression ratio")
    if query_positions is not None:
        if query_positions.shape != (rows,):
            raise ValueError("QSA query positions must match sparse-attention rows")
        if query_positions.dtype not in _INTEGER_DTYPES:
            raise ValueError("QSA query positions must use integer storage")
        _require_same_cuda_device("QSA causal-prefix bounds", q, query_positions)
        if compress_ratio <= 0:
            raise ValueError("QSA compression ratio must be positive")

    base_programs = rows * num_kv_heads
    if query_positions is not None and base_programs > 512:
        final_topk = selected_tokens.shape[1]
        token_topk = final_topk - compress_ratio + 1
        if token_topk <= 0 or token_topk % compress_ratio:
            raise ValueError("QSA selected-token width is incompatible with its compression ratio")
        _qsa_prefill_pair_stable_kernel[(rows, num_kv_heads)](
            *_pair_args(
                q,
                k_cache,
                v_cache,
                block_table,
                selected_tokens,
                request_indices,
                query_positions.contiguous(),
                output,
                softmax_scale,
            ),
            **_pair_constexprs(q, k_cache, selected_tokens, group_size, block_m, compress_ratio),
        )
        return output

    block_n, num_splits, num_warps = splitk_launch_config(
        rows, num_kv_heads, selected_tokens.shape[1]
    )
    use_pdl = num_splits > 1 and _pdl_enabled(rows)
    # Row-major partials in the 64-split bucket (`splitk_launch_config`: rows x KV heads <= 4).
    row_major = rows * num_kv_heads <= 4
    if num_splits == 1:
        partial_output = output
        partial_lse = output
    else:
        lead = (rows, num_splits) if row_major else (num_splits, rows)
        partial_output = torch.empty(
            (*lead, num_q_heads, head_dim), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty((*lead, num_q_heads), dtype=torch.float32, device=q.device)
    gate = output if output_gate is None else output_gate
    _warm_decode_buckets(
        q,
        k_cache,
        v_cache,
        block_table,
        selected_tokens,
        request_indices,
        output,
        gate,
        output_gate,
        softmax_scale,
        group_size,
        block_m,
        tokens_per_block,
        None if query_positions is None else (compress_ratio, query_positions.dtype),
    )
    _qsa_splitk_stable_kernel[(rows, num_kv_heads, num_splits)](
        *_splitk_args(
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
        ),
        **_splitk_constexprs(
            q,
            selected_tokens,
            group_size,
            block_m,
            tokens_per_block,
            block_n,
            num_splits,
            num_warps,
            use_pdl,
            row_major,
        ),
    )
    if num_splits == 1:
        return output
    _qsa_merge_stable_kernel[(rows, num_q_heads)](
        *_merge_args(partial_output, partial_lse, output, gate, output_gate, rows, head_dim),
        **_merge_constexprs(num_q_heads, head_dim, num_splits, use_pdl, row_major, output_gate),
    )
    return output


def _pair_args(
    q,
    k_cache,
    v_cache,
    block_table,
    selected_tokens,
    request_indices,
    query_positions,
    output,
    softmax_scale,
) -> tuple:
    return (
        q,
        k_cache,
        v_cache,
        block_table,
        selected_tokens,
        request_indices,
        query_positions,
        output,
        softmax_scale,
        k_cache.shape[0],
        block_table.shape[0],
        block_table.shape[1],
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
    )


def _pair_constexprs(q, k_cache, selected_tokens, group_size, block_m, compress_ratio) -> dict:
    final_topk = selected_tokens.shape[1]
    token_topk = final_topk - compress_ratio + 1
    return dict(
        GROUP_SIZE=group_size,
        TOKENS_PER_BLOCK=k_cache.shape[2],
        TOPK=final_topk,
        BLOCK_TOPK=token_topk // compress_ratio,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_M=block_m,
        BLOCK_N=64,
        HEAD_DIM=q.shape[2],
        num_warps=2,
        num_stages=2,
    )


def _splitk_args(
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
) -> tuple:
    return (
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
        k_cache.shape[0],
        block_table.shape[0],
        block_table.shape[1],
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
    )


def _splitk_constexprs(
    q,
    selected_tokens,
    group_size,
    block_m,
    tokens_per_block,
    block_n,
    num_splits,
    num_warps,
    use_pdl,
    row_major,
) -> dict:
    return dict(
        GROUP_SIZE=group_size,
        NUM_QUERY_HEADS=q.shape[1],
        TOKENS_PER_BLOCK=tokens_per_block,
        TOPK=selected_tokens.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=triton.cdiv(selected_tokens.shape[1], block_n),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=q.shape[2],
        USE_PDL=use_pdl,
        ROW_MAJOR=row_major,
        num_warps=num_warps,
        num_stages=2,
        launch_pdl=use_pdl,
    )


def _merge_args(partial_output, partial_lse, output, gate, output_gate, rows, head_dim) -> tuple:
    if output_gate is not None and output_gate.ndim == 3:
        gate_stride_row = output_gate.stride(0)
        gate_stride_head = output_gate.stride(1)
    else:
        gate_stride_row = gate.stride(0)
        gate_stride_head = head_dim
    return (
        partial_output,
        partial_lse,
        output,
        gate,
        rows,
        gate_stride_row,
        gate_stride_head,
        output.stride(0),
        output.stride(1),
        output.stride(2),
    )


def _merge_constexprs(num_q_heads, head_dim, num_splits, use_pdl, row_major, output_gate) -> dict:
    return dict(
        NUM_QUERY_HEADS=num_q_heads,
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        HEAD_DIM=head_dim,
        USE_PDL=use_pdl,
        FUSE_OUTPUT_GATE=output_gate is not None,
        ROW_MAJOR=row_major,
        num_warps=2,
        num_stages=1,
        launch_pdl=use_pdl,
    )


#: Row counts reaching every split-K launch variant at one and two KV heads.
_BUCKET_ROWS = (1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 128, 129, 256, 257, 512, 513)


def decode_bucket_variants(num_kv_heads: int, selected_width: int) -> list[tuple]:
    """`(block_n, num_splits, num_warps, use_pdl, row_major)` of every split-K launch, any rows."""
    seen = []
    for rows in _BUCKET_ROWS:
        block_n, num_splits, num_warps = splitk_launch_config(rows, num_kv_heads, selected_width)
        variant = (
            block_n,
            num_splits,
            num_warps,
            num_splits > 1 and _pdl_enabled(rows),
            rows * num_kv_heads <= 4,
        )
        if variant not in seen:
            seen.append(variant)
    return seen


def _warm_decode_buckets(
    q,
    k_cache,
    v_cache,
    block_table,
    selected_tokens,
    request_indices,
    output,
    gate,
    output_gate,
    softmax_scale,
    group_size,
    block_m,
    tokens_per_block,
    pair=None,
) -> None:
    """Compile (not run) the split-K and merge kernels for every row bucket of this call's
    geometry the first time it is seen, so no serving step compiles one."""
    key = (
        q.dtype,
        q.shape[1:],
        q.stride(),
        q.data_ptr() % 16 == 0,
        k_cache.dtype,
        k_cache.shape[1:],
        k_cache.stride(),
        v_cache.stride(),
        k_cache.data_ptr() % 16 == 0 and v_cache.data_ptr() % 16 == 0,
        block_table.stride(1),
        selected_tokens.shape[1],
        selected_tokens.stride(),
        request_indices.dtype,
        None if output_gate is None else (output_gate.shape[1:], output_gate.stride()),
        tokens_per_block,
        pair,
    )
    if not first_time("qsa_stable_decode_buckets", key):
        return
    if output is None:
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        gate = output
    rows = q.shape[0]
    num_q_heads, head_dim = q.shape[1], q.shape[2]
    scratch = torch.empty(16, dtype=torch.float32, device=q.device)
    variants = decode_bucket_variants(k_cache.shape[1], selected_tokens.shape[1])
    splitk, merge = [], []
    for block_n, num_splits, num_warps, use_pdl, row_major in variants:
        partials = (output, output) if num_splits == 1 else (scratch, scratch)
        splitk.append(
            (
                _splitk_args(
                    q,
                    k_cache,
                    v_cache,
                    block_table,
                    selected_tokens,
                    request_indices,
                    *partials,
                    output,
                    softmax_scale,
                    rows,
                ),
                _splitk_constexprs(
                    q,
                    selected_tokens,
                    group_size,
                    block_m,
                    tokens_per_block,
                    block_n,
                    num_splits,
                    num_warps,
                    use_pdl,
                    row_major,
                ),
            )
        )
        if num_splits > 1:
            merge.append(
                (
                    _merge_args(scratch, scratch, output, gate, output_gate, rows, head_dim),
                    _merge_constexprs(
                        num_q_heads, head_dim, num_splits, use_pdl, row_major, output_gate
                    ),
                )
            )
    for args, constexprs in splitk:
        compile_only(_qsa_splitk_stable_kernel, args, **constexprs)
    for args, constexprs in merge:
        compile_only(_qsa_merge_stable_kernel, args, **constexprs)
    if pair is not None:
        # A context launch past 512 programs over the same pool (the prefill pair kernel).
        compress_ratio, positions_dtype = pair
        positions = torch.empty(1, dtype=positions_dtype, device=q.device)
        compile_only(
            _qsa_prefill_pair_stable_kernel,
            _pair_args(
                q,
                k_cache,
                v_cache,
                block_table,
                selected_tokens,
                request_indices,
                positions,
                output,
                softmax_scale,
            ),
            **_pair_constexprs(q, k_cache, selected_tokens, group_size, block_m, compress_ratio),
        )


def warm_decode_buckets(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    softmax_scale: float,
    compress_ratio: int,
) -> None:
    """Compile (not run) the split-K composition for every row bucket of a decode call's
    geometry, and the prefill pair kernel (int64 positions), for a caller that serves part of
    the row range with another kernel."""
    num_q_heads = q.shape[1]
    group_size = num_q_heads // k_cache.shape[1]
    block_m = max(16, triton.next_power_of_2(group_size))
    _warm_decode_buckets(
        q,
        k_cache,
        v_cache,
        block_table,
        selected_tokens,
        request_indices,
        None,
        None,
        None,
        softmax_scale,
        group_size,
        block_m,
        tokens_per_block,
        (compress_ratio, torch.int64),
    )
