# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA over an NVFP4 paged pool: attention over explicitly selected tokens."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Approved exception (as for `qsa_paged_sparse_gqa`): QSA's sparse path has no torch.ops
# surface; its split-K buckets and row-count PDL policy are reached as Python.
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
    _qsa_pdl_enabled,
    _qsa_splitk_launch_config,
)

from .._op import Arch, Cell, OpWrapper

#: Base-2 online softmax: scores are scaled by log2(e) once.
_LOG2_E = tl.constexpr(1.4426950408889634)
#: Values per e4m3 block scale, e2m1 values per byte, tokens per V scale group.
_SF_VEC = 16
_PER_BYTE = 2
_V_SF_QUAD = 4
#: e2m1 codes 0..7 -> magnitude; bit 3 is the sign.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: This checkpoint's geometry and the engine's QSA table. `global_k` / `global_v` are
#: `kv_scale_quant_orig[1]` / `[2]`; `slot_pages` role pages of one lifecycle slot hold
#: the call's planes, layer `layer`'s K and V among them (the trunk pool holds 24).
_CHECKPOINT = dict(
    q_heads=24,
    kv_heads=2,
    head_dim=256,
    tokens_per_block=32,
    compress_ratio=4,
    token_topk=2048,
    q_dtype=torch.bfloat16,
    # q is the first 24 heads of the packed [T, 24 + 2 + 2, 256] QKV buffer.
    q_layout="qkv_slice",
    request_dtype=torch.int32,
    # The engine's table: 260 pages per slot, -1 past each sequence.
    pages=260,
    global_k=1.0,
    global_v=1.0,
    slot_pages=2,
    layer=0,
)

_SPLIT_K = ("splitk", "merge")
_PREFILL = ("prefill",)


def _spec(**overrides) -> dict:
    """A cell spec: this checkpoint's geometry unless the cell says otherwise.

    `segments` is `(first position, rows)` per request, in row order; `prefill_pair` says whether `query_positions`
    / `compress_ratio` are passed, which the target does exactly when the step has a
    context request; `pool_pages` fixes the pool's page count (the requests then sit
    on its last pages); `capture` sweeps the decode site under a CUDA graph.
    """
    return {**_CHECKPOINT, **overrides}


def _decode(segments, launches=_SPLIT_K, **overrides) -> dict:
    base = dict(segments=segments, slots=32, prefill_pair=False, launches=launches)
    return _spec(**{**base, **overrides})


# Step-5-Preview's sparse layers under the kv_nvfp4 variant: 64 q / 4 kv heads
# under attention DP, 16 / 1 under TP4, head_dim 192 zero-padded to 256 (the pad
# dims are all-zero 16-value blocks: e2m1 code 0 under an e4m3 scale of 0), one
# call per KV group on single-head views of the data and block-scale planes. The
# selection holds the top-512 complete regions of 8 tokens (score order, -1
# padded), each expanded to its 8 logical positions, and the current region's 8
# columns, -1 past the query. `layout` says where the current region sits:
# "regions_then_tail" (columns 4096..4103) or "tail_first" (columns 0..7).
_STEP5 = dict(
    q_heads=16,
    kv_heads=1,
    head_dim=256,
    real_head_dim=192,
    tokens_per_block=32,
    region_size=8,
    region_topk=512,
    width=512 * 8 + 8,
    softmax_scale=192**-0.5,
    q_dtype=torch.bfloat16,
    request_dtype=torch.int32,
    selection="step5_regions",
    prefill_pair=False,
    # max_seq_len 9216 / tokens_per_block 32.
    pages=288,
    # max_batch_size 128, plus one so slot and row never coincide.
    slots=129,
    global_k=1.0,
    global_v=1.0,
    # K and V of one layer per slot; the trunk-layout cell holds all 23 sparse layers.
    slot_pages=2,
    layer=0,
)

#: The trunk layout: the 23 sparse layers' K and V pages of one slot, coalesced.
_STEP5_TRUNK = dict(slot_pages=46, layer=22)

#: The pool kv_nvfp4@tep4 allocated: 4243648 KV tokens = 132614 slots of 32, each 46
#: data pages of 4 KiB and 46 block-scale pages of 512 B (one head per rank).
#: `read_last_slot` maps the pool's last slot under a page the selection reads.
_STEP5_MEASURED_POOL = dict(pool_pages=132614, slot_pages=46, read_last_slot=True)

#: The slot layout kv_nvfp4@dep4 allocated -- 46 data pages of 16 KiB and 46 block-scale
#: pages of 2 KiB per slot, four heads per rank -- at 34830 slots, half the 69660 it
#: measured (2229120 KV tokens): 24.4 GiB of data and 3.1 GiB of scales.
_STEP5_DEP4_POOL = dict(pool_pages=34830, slot_pages=46, read_last_slot=True)


def _spread(n: int) -> tuple:
    """`n` decode requests, one row each, spread over every position 1..9215."""
    return tuple((1 + (9214 * i) // max(n - 1, 1), 1) for i in range(n))


_SPREAD_128 = _spread(128)


def _step5(
    segments, pool_kv_heads, head_view, splits, layout="regions_then_tail", **overrides
) -> dict:
    """A Step-5 cell: the split-K kernel, `splits` ways, merged when more than one."""
    launches = _SPLIT_K if splits > 1 else ("splitk",)
    base = dict(
        segments=segments,
        pool_kv_heads=pool_kv_heads,
        head_view=head_view,
        splits=splits,
        launches=launches,
        layout=layout,
    )
    return {**_STEP5, **base, **overrides}


@triton.jit
def _e2m1_to_f32(code):
    """e2m1 code (0..15, bit 3 the sign) -> its value, exactly."""
    code = code.to(tl.int32)
    magnitude_code = code & 7
    exponent = magnitude_code >> 1
    mantissa = (magnitude_code & 1).to(tl.float32)
    magnitude = tl.where(
        exponent == 0,
        0.5 * mantissa,
        (2.0 + mantissa) * (1 << exponent).to(tl.float32) * 0.25,
    )
    return tl.where((code & 8) != 0, -magnitude, magnitude)


@triton.jit
def _load_keys(
    k_cache,
    k_scale_cache,
    pages,
    kv_head,
    tokens,
    valid,
    dims,
    k_global,
    k_stride_page,
    k_stride_head,
    k_stride_token,
    ks_stride_page,
    ks_stride_head,
    SCALE_GROUPS: tl.constexpr,
):
    """K for the selected tokens as an fp32 [HEAD_DIM, BLOCK_N] tile: e2m1 x scale x global."""
    packed = tl.load(
        k_cache
        + pages[None, :] * k_stride_page
        + kv_head * k_stride_head
        + tokens[None, :] * k_stride_token
        + dims[:, None] // 2,
        mask=valid[None, :],
        other=0,
    ).to(tl.int32)
    codes = (packed >> ((dims[:, None] % 2) * 4)) & 0xF
    scales = (
        tl.load(
            k_scale_cache
            + pages[None, :] * ks_stride_page
            + kv_head * ks_stride_head
            + tokens[None, :] * SCALE_GROUPS
            + dims[:, None] // 16,
            mask=valid[None, :],
            other=0,
        )
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    return _e2m1_to_f32(codes) * scales * k_global


@triton.jit
def _load_values(
    v_cache,
    v_scale_cache,
    pages,
    kv_head,
    tokens,
    valid,
    dims,
    v_global,
    v_stride_page,
    v_stride_head,
    v_stride_token,
    vs_stride_page,
    vs_stride_head,
    SCALE_GROUPS: tl.constexpr,
):
    """V for the selected tokens as an fp32 [BLOCK_N, HEAD_DIM] tile; V's scales sit in
    four-token groups, `[tpb / 4, D / 16, 4]` per head page."""
    packed = tl.load(
        v_cache
        + pages[:, None] * v_stride_page
        + kv_head * v_stride_head
        + tokens[:, None] * v_stride_token
        + dims[None, :] // 2,
        mask=valid[:, None],
        other=0,
    ).to(tl.int32)
    codes = (packed >> ((dims[None, :] % 2) * 4)) & 0xF
    scales = (
        tl.load(
            v_scale_cache
            + pages[:, None] * vs_stride_page
            + kv_head * vs_stride_head
            + (tokens[:, None] // 4) * (4 * SCALE_GROUPS)
            + (dims[None, :] // 16) * 4
            + tokens[:, None] % 4,
            mask=valid[:, None],
            other=0,
        )
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    return _e2m1_to_f32(codes) * scales * v_global


@triton.jit
def _nvfp4_sparse_gqa_prefill_kernel(
    q,
    k_cache,
    v_cache,
    k_scale_cache,
    v_scale_cache,
    kv_scale_quant_orig,
    block_table,
    selected_tokens,
    request_indices,
    query_positions,
    output,
    softmax_scale,
    num_cache_pages,
    num_requests,
    page_table_width,
    q_stride_row,
    q_stride_head,
    k_stride_page,
    k_stride_head,
    k_stride_token,
    v_stride_page,
    v_stride_head,
    v_stride_token,
    ks_stride_page,
    ks_stride_head,
    vs_stride_page,
    vs_stride_head,
    block_table_stride_request,
    selected_stride_row,
    output_stride_row,
    output_stride_head,
    GROUP_SIZE: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    request = tl.load(request_indices + row).to(tl.int64)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    k_global = tl.load(kv_scale_quant_orig + 1)
    v_global = tl.load(kv_scale_quant_orig + 2)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    query_heads = kv_head * GROUP_SIZE + head_offsets
    query_values = tl.load(
        q + row * q_stride_row + query_heads[:, None] * q_stride_head + dim_offsets[None, :],
        mask=(head_offsets < GROUP_SIZE)[:, None],
        other=0.0,
    )

    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    token_offsets = tl.arange(0, BLOCK_N)
    visible_tokens = tl.load(query_positions + row).to(tl.int64) + 1
    selected_blocks = tl.minimum(visible_tokens // COMPRESS_RATIO, BLOCK_TOPK)
    valid_tokens = selected_blocks * COMPRESS_RATIO + visible_tokens % COMPRESS_RATIO
    valid_tiles = tl.cdiv(tl.minimum(valid_tokens, TOPK), BLOCK_N)

    for tile in tl.range(0, valid_tiles, num_stages=2):
        selected_columns = tile * BLOCK_N + token_offsets
        logical_tokens = tl.load(
            selected_tokens + row * selected_stride_row + selected_columns,
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
        physical_pages = tl.load(
            block_table
            + safe_request * block_table_stride_request
            + tl.minimum(logical_pages, page_table_width - 1),
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid &= (physical_pages >= 0) & (physical_pages < num_cache_pages)
        safe_pages = tl.maximum(physical_pages, 0)

        keys = _load_keys(
            k_cache,
            k_scale_cache,
            safe_pages,
            kv_head,
            token_in_page,
            valid,
            dim_offsets,
            k_global,
            k_stride_page,
            k_stride_head,
            k_stride_token,
            ks_stride_page,
            ks_stride_head,
            HEAD_DIM // 16,
        ).to(query_values.dtype)
        values = _load_values(
            v_cache,
            v_scale_cache,
            safe_pages,
            kv_head,
            token_in_page,
            valid,
            dim_offsets,
            v_global,
            v_stride_page,
            v_stride_head,
            v_stride_token,
            vs_stride_page,
            vs_stride_head,
            HEAD_DIM // 16,
        ).to(query_values.dtype)
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
        + dim_offsets[None, :],
        normalized,
        mask=(head_offsets < GROUP_SIZE)[:, None],
    )


@triton.jit
def _nvfp4_sparse_gqa_splitk_kernel(
    q,
    k_cache,
    v_cache,
    k_scale_cache,
    v_scale_cache,
    kv_scale_quant_orig,
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
    q_stride_row,
    q_stride_head,
    k_stride_page,
    k_stride_head,
    k_stride_token,
    v_stride_page,
    v_stride_head,
    v_stride_token,
    ks_stride_page,
    ks_stride_head,
    vs_stride_page,
    vs_stride_head,
    block_table_stride_request,
    selected_stride_row,
    output_stride_row,
    output_stride_head,
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
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    request = tl.load(request_indices + row).to(tl.int64)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    k_global = tl.load(kv_scale_quant_orig + 1)
    v_global = tl.load(kv_scale_quant_orig + 2)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    token_offsets = tl.arange(0, BLOCK_N)
    query_heads = kv_head * GROUP_SIZE + head_offsets
    query_values = tl.load(
        q + row * q_stride_row + query_heads[:, None] * q_stride_head + dim_offsets[None, :],
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
            selected_tokens + row * selected_stride_row + selected_columns,
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
        physical_pages = tl.load(
            block_table
            + safe_request * block_table_stride_request
            + tl.minimum(logical_pages, page_table_width - 1),
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid &= (physical_pages >= 0) & (physical_pages < num_cache_pages)
        safe_pages = tl.maximum(physical_pages, 0)

        keys = _load_keys(
            k_cache,
            k_scale_cache,
            safe_pages,
            kv_head,
            token_in_page,
            valid,
            dim_offsets,
            k_global,
            k_stride_page,
            k_stride_head,
            k_stride_token,
            ks_stride_page,
            ks_stride_head,
            HEAD_DIM // 16,
        ).to(query_values.dtype)
        values = _load_values(
            v_cache,
            v_scale_cache,
            safe_pages,
            kv_head,
            token_in_page,
            valid,
            dim_offsets,
            v_global,
            v_stride_page,
            v_stride_head,
            v_stride_token,
            vs_stride_page,
            vs_stride_head,
            HEAD_DIM // 16,
        ).to(query_values.dtype)
        scores = tl.dot(query_values, keys)
        scores *= softmax_scale * _LOG2_E
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        # Until the split meets a resident token both maxima are -inf, and
        # exp2(-inf - -inf) would be NaN; the accumulator and sum are still zero.
        correction = tl.where(next_max == -float("inf"), 1.0, tl.math.exp2(running_max - next_max))
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
            + dim_offsets[None, :],
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
def _nvfp4_merge_splitk_kernel(
    partial_output,
    partial_lse,
    output,
    num_rows,
    output_stride_row,
    output_stride_head,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
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
    tl.store(output + row * output_stride_row + head * output_stride_head + dim_offsets, merged)


def _sparse_gqa_nvfp4(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    kv_scale_quant_orig: torch.Tensor,
    block_table: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    softmax_scale: float,
    query_positions: Optional[torch.Tensor] = None,
    compress_ratio: Optional[int] = None,
) -> torch.Tensor:
    """Sparse GQA over the selected tokens of an NVFP4 HND paged pool (see the entry)."""
    if q.ndim != 3:
        raise ValueError("NVFP4 sparse GQA: q must be [rows, query_heads, head_dim]")
    rows, num_q_heads, head_dim = q.shape
    if q.dtype not in (torch.bfloat16, torch.float16) or q.stride(2) != 1:
        raise ValueError("NVFP4 sparse GQA: q must be BF16 or FP16 with a unit last-dim stride")
    if head_dim < 32 or head_dim & (head_dim - 1):
        raise ValueError("NVFP4 sparse GQA: head_dim must be a power of two of at least 32")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("NVFP4 sparse GQA: k_cache and v_cache must be matching 4-D planes")
    num_pages, num_kv_heads = k_cache.shape[0], k_cache.shape[1]
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("NVFP4 sparse GQA: query heads must be a multiple of KV heads")
    if tokens_per_block <= 0 or tokens_per_block % _V_SF_QUAD:
        raise ValueError("NVFP4 sparse GQA: tokens_per_block must be a positive multiple of 4")
    data_shape = (num_pages, num_kv_heads, tokens_per_block, head_dim // _PER_BYTE)
    scale_shape = (num_pages, num_kv_heads, tokens_per_block, head_dim // _SF_VEC)
    for name, plane, shape in (
        ("k_cache", k_cache, data_shape),
        ("v_cache", v_cache, data_shape),
        ("k_scale_cache", k_scale_cache, scale_shape),
        ("v_scale_cache", v_scale_cache, scale_shape),
    ):
        if plane.dtype != torch.uint8 or tuple(plane.shape) != shape or plane.stride(3) != 1:
            raise ValueError(
                f"NVFP4 sparse GQA: {name} must be uint8 {shape} with a unit last-dim stride, "
                f"got {plane.dtype} {tuple(plane.shape)}"
            )
    for name, plane in (("k_scale_cache", k_scale_cache), ("v_scale_cache", v_scale_cache)):
        if plane.stride(2) != head_dim // _SF_VEC:
            raise ValueError(f"NVFP4 sparse GQA: each head page of {name} must be contiguous")
    if (
        kv_scale_quant_orig.dtype != torch.float32
        or kv_scale_quant_orig.shape != (3,)
        or not kv_scale_quant_orig.is_contiguous()
    ):
        raise ValueError("NVFP4 sparse GQA: kv_scale_quant_orig must be a contiguous fp32 [3]")
    if block_table.ndim != 2 or block_table.dtype != torch.int32 or block_table.stride(1) != 1:
        raise ValueError("NVFP4 sparse GQA: block_table must be a row-contiguous 2-D int32 tensor")
    if block_table.shape[0] <= 0 or block_table.shape[1] <= 0 or num_pages <= 0:
        raise ValueError("NVFP4 sparse GQA: the pool and the table must be non-empty")
    if (
        selected_tokens.ndim != 2
        or selected_tokens.dtype != torch.int32
        or selected_tokens.shape[0] != rows
        or selected_tokens.shape[1] <= 0
        or selected_tokens.stride(1) != 1
    ):
        raise ValueError(
            "NVFP4 sparse GQA: selected_tokens must be a row-contiguous int32 [rows, width]"
        )
    if (
        request_indices.shape != (rows,)
        or request_indices.dtype not in (torch.int32, torch.int64)
        or not request_indices.is_contiguous()
    ):
        raise ValueError("NVFP4 sparse GQA: request_indices must be one contiguous int per row")
    if (query_positions is None) != (compress_ratio is None):
        raise ValueError(
            "NVFP4 sparse GQA: pass both query_positions and compress_ratio, or neither"
        )
    if query_positions is not None and (
        query_positions.shape != (rows,)
        or query_positions.dtype not in (torch.int32, torch.int64)
        or not query_positions.is_contiguous()
        or compress_ratio <= 0
    ):
        raise ValueError(
            "NVFP4 sparse GQA: query_positions must be one contiguous int per row and "
            "compress_ratio positive"
        )
    if not (softmax_scale > 0 and softmax_scale < float("inf")):
        raise ValueError("NVFP4 sparse GQA: softmax_scale must be finite and positive")
    operands = (q, k_cache, v_cache, k_scale_cache, v_scale_cache, kv_scale_quant_orig)
    operands += (block_table, selected_tokens, request_indices)
    operands += () if query_positions is None else (query_positions,)
    if not all(t.is_cuda and t.device == q.device for t in operands):
        raise ValueError("NVFP4 sparse GQA: every operand must be on q's CUDA device")

    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    if rows == 0:
        return output
    plane_strides = (
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        k_scale_cache.stride(0),
        k_scale_cache.stride(1),
        v_scale_cache.stride(0),
        v_scale_cache.stride(1),
    )
    if query_positions is not None and rows * num_kv_heads > 512:
        final_topk = selected_tokens.shape[1]
        token_topk = final_topk - compress_ratio + 1
        if token_topk <= 0 or token_topk % compress_ratio:
            raise ValueError(
                "NVFP4 sparse GQA: the selection width does not come from this compression ratio"
            )
        _nvfp4_sparse_gqa_prefill_kernel[(rows, num_kv_heads)](
            q,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            kv_scale_quant_orig,
            block_table,
            selected_tokens,
            request_indices,
            query_positions,
            output,
            softmax_scale,
            num_pages,
            block_table.shape[0],
            block_table.shape[1],
            q.stride(0),
            q.stride(1),
            *plane_strides,
            block_table.stride(0),
            selected_tokens.stride(0),
            output.stride(0),
            output.stride(1),
            GROUP_SIZE=group_size,
            TOKENS_PER_BLOCK=tokens_per_block,
            TOPK=final_topk,
            BLOCK_TOPK=token_topk // compress_ratio,
            COMPRESS_RATIO=compress_ratio,
            BLOCK_M=block_m,
            BLOCK_N=64,
            HEAD_DIM=head_dim,
            num_warps=2,
            num_stages=2,
        )
        return output

    block_n, num_splits, num_warps = _qsa_splitk_launch_config(
        rows, num_kv_heads, selected_tokens.shape[1]
    )
    num_tiles = triton.cdiv(selected_tokens.shape[1], block_n)
    use_pdl = num_splits > 1 and _qsa_pdl_enabled(rows)
    if num_splits == 1:
        partial_output = partial_lse = output
    else:
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty(
            (num_splits, rows, num_q_heads), dtype=torch.float32, device=q.device
        )
    _nvfp4_sparse_gqa_splitk_kernel[(rows, num_kv_heads, num_splits)](
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        kv_scale_quant_orig,
        block_table,
        selected_tokens,
        request_indices,
        partial_output,
        partial_lse,
        output,
        softmax_scale,
        rows,
        num_pages,
        block_table.shape[0],
        block_table.shape[1],
        q.stride(0),
        q.stride(1),
        *plane_strides,
        block_table.stride(0),
        selected_tokens.stride(0),
        output.stride(0),
        output.stride(1),
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
    _nvfp4_merge_splitk_kernel[(rows, num_q_heads)](
        partial_output,
        partial_lse,
        output,
        rows,
        output.stride(0),
        output.stride(1),
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


def _dequantize_tokens(data, scales, pages, slots, global_scale, grouped: bool) -> torch.Tensor:
    """[..., Hkv, D] fp32 values of the tokens at (pages, slots) of one NVFP4 role plane."""
    heads, tokens_per_block, packed = data.shape[1], data.shape[2], data.shape[3]
    head_dim = packed * _PER_BYTE
    groups = head_dim // _SF_VEC
    raw = data[pages, :, slots, :].long()  # [..., H, D/2]
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=data.device)
    lut = torch.cat([lut, -lut])
    values = torch.stack([lut[raw & 0xF], lut[raw >> 4]], dim=-1).flatten(-2)
    if grouped:
        flat = scales.reshape(scales.shape[0], heads, tokens_per_block * groups)
        base = (slots // _V_SF_QUAD) * (_V_SF_QUAD * groups) + slots % _V_SF_QUAD
        columns = base[..., None] + _V_SF_QUAD * torch.arange(groups, device=data.device)
        sf = torch.stack([flat[pages[..., None], h, columns] for h in range(heads)], dim=-2)
    else:
        sf = scales[pages, :, slots, :]
    sf = sf.view(torch.float8_e4m3fn).float().repeat_interleave(_SF_VEC, dim=-1)
    return values * sf * global_scale


class QsaPagedSparseGqaNvfp4(OpWrapper):
    """Sparse GQA over the selected tokens of an NVFP4 HND paged pool.

    Triton kernels defined in this file. For row `r`, query head `h`, KV head
    `h // (Hq // Hkv)` (contiguous groups), over the columns `j` of
    `selected_tokens[r]` that name a resident token::

        t      = selected_tokens[r, j]                       # logical position
        page   = block_table[request_indices[r], t // tokens_per_block]
        K[t]   = bf16(e2m1(k_cache) * e4m3(k_scale_cache) * kv_scale_quant_orig[1])
        V[t]   = bf16(e2m1(v_cache) * e4m3(v_scale_cache) * kv_scale_quant_orig[2])
        s_j    = (q[r, h] . K[t]) * softmax_scale
        out[r, h] = sum_j softmax(s)_j * V[t]

    The pool layout is the attention op's own NVFP4 append: e2m1 nibbles per
    head page `[tpb, D/2]` (low nibble = even element), K's e4m3 scales
    token-major `[tpb, D/16]`, V's in four-token groups `[tpb/4, D/16, 4]`.
    Each value is dequantized in fp32 and rounded once to `q`'s dtype; at unit
    second-level scales every NVFP4 value is exact in bf16, so a call is the
    bf16 sparse GQA over the bf16 values the pool holds. After the dequantize the
    kernels are QSA's bf16 sparse GQA kernels operation for operation: the same
    tiles, online softmax and split-K merge -- except that the split-K kernel
    holds its online-softmax correction at 1 while nothing resident has been
    seen, so a split may open on tiles that name no token.

    A negative index, a page past the table's width, a negative or out-of-pool
    table entry, and a request index outside the table's rows contribute
    nothing (on the split-K kernel wherever they sit in the row; the prefill
    kernel's bound is below); a row left with nothing returns zeros. Returns a
    fresh, contiguous `[rows, Hq, D]` tensor of `q.dtype`.

    Two kernels, chosen inside the call: with `query_positions` and
    `compress_ratio` given and `rows * Hkv > 512`, a causal-prefix prefill
    kernel; otherwise split-K, plus a merge kernel when it used more than one
    split. **The prefill pair is a bound, not a mask**, exactly as on the bf16
    entry: the prefill kernel reads only `min(visible // cr, block_topk) * cr +
    visible % cr` leading columns of a row.

    Fusion boundary: the block-table gather, the NVFP4 dequantize, scale,
    online softmax, P.V and the split-K merge are inside. Selection, appending
    K/V to the pool, and q's projection, norm, RoPE and output gate are not.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: 4 requests x 4 rows, 8-way split-K + merge, in the engine's trunk "
            "layout: layer 11's planes among the 24 role pages of a slot",
            spec=_decode(
                tuple((4000 + 37 * i, 4) for i in range(4)), slot_pages=24, layer=11, pool_pages=600
            ),
        ),
        Cell(
            why="context 4096 rows from 0: rows see 1..4096 tokens",
            spec=_decode(((0, 4096),), launches=_PREFILL, prefill_pair=True),
        ),
        Cell(
            why="captured decode: G 1, 8 and 32 requests x 1 and 8 rows, replay == eager",
            spec=_decode((), slots=32, capture=dict(groups=(1, 8, 32), rows_per_request=(1, 8))),
        ),
        # ---- Step-5-Preview: split-K only, width 4104, single-KV-head views --
        # Regions then the current region: rows with fewer than 449 regions
        # (positions below 3592) meet a split whose leading tile is all -1.
        Cell(
            why="Step-5 dp 128 decode rows at pos 1..9215, head view 1 of 4, regions then "
            "tail: 8-way split-K, merged",
            spec=_step5(_SPREAD_128, pool_kv_heads=4, head_view=1, splits=8, pool_pages=20000),
        ),
        Cell(
            why="Step-5 dp 8192-row prefill from 0, head view 3 of 4, regions then tail: one split",
            spec=_step5(((0, 8192),), pool_kv_heads=4, head_view=3, splits=1, pool_pages=20000),
        ),
        Cell(
            why="Step-5 tp4 1-head pool, 512-row chunk at 4096 over a reused prefix + a "
            "decode row at region end 8703: 513 rows, one split",
            spec=_step5(
                ((4096, 512), (8703, 1)),
                pool_kv_heads=1,
                head_view=None,
                splits=1,
                pool_pages=20000,
            ),
        ),
        Cell(
            why="Step-5 envelope: dp 8192 rows at pos 1024..9215 after a reused 1024-token "
            "prefix, head view 2: one split",
            spec=_step5(((1024, 8192),), pool_kv_heads=4, head_view=2, splits=1, pool_pages=20000),
        ),
        Cell(
            why="Step-5 dp 1 row at pos 9215 in the trunk layout (46 planes per slot, plane "
            "44/45), head view 2: 64-way split-K under PDL",
            spec=_step5(
                ((9215, 1),),
                pool_kv_heads=4,
                head_view=2,
                splits=64,
                pdl=True,
                pool_pages=600,
                **_STEP5_TRUNK,
            ),
        ),
        # Current region first: every resident token precedes every -1.
        Cell(
            why="Step-5 tail-first dp 128 decode rows at pos 1..9215, head view 1 of 4: 8-way "
            "split-K, merged",
            spec=_step5(
                _SPREAD_128,
                pool_kv_heads=4,
                head_view=1,
                splits=8,
                layout="tail_first",
                pool_pages=20000,
            ),
        ),
        Cell(
            why="Step-5 tail-first dp 8192-row prefill from 0, head view 3 of 4: one split",
            spec=_step5(
                ((0, 8192),),
                pool_kv_heads=4,
                head_view=3,
                splits=1,
                layout="tail_first",
                pool_pages=20000,
            ),
        ),
        # The split-K classes the capture batches and the chunks reach, tail first;
        # `capture=True` also replays the call from a CUDA graph on refilled state.
        Cell(
            why="Step-5 tail-first dp 4 decode rows at pos 1..9215, head view 0 of 4: 64-way "
            "split-K under PDL, merged, captured",
            spec=_step5(
                _spread(4),
                pool_kv_heads=4,
                head_view=0,
                splits=64,
                pdl=True,
                layout="tail_first",
                pool_pages=20000,
                capture=True,
            ),
        ),
        Cell(
            why="Step-5 tail-first tp4 16 decode rows at pos 1..9215, 1-head pool: 32-way "
            "split-K of 16-token tiles, merged, captured",
            spec=_step5(
                _spread(16),
                pool_kv_heads=1,
                head_view=None,
                splits=32,
                layout="tail_first",
                pool_pages=20000,
                capture=True,
            ),
        ),
        Cell(
            why="Step-5 tail-first dp 32 decode rows at pos 1..9215, head view 2 of 4: 8-way "
            "split-K, merged, captured",
            spec=_step5(
                _spread(32),
                pool_kv_heads=4,
                head_view=2,
                splits=8,
                layout="tail_first",
                pool_pages=20000,
                capture=True,
            ),
        ),
        Cell(
            why="Step-5 tail-first tp4 120 decode rows at pos 1..9215, 1-head pool: 8-way "
            "split-K, merged, captured",
            spec=_step5(
                _spread(120),
                pool_kv_heads=1,
                head_view=None,
                splits=8,
                layout="tail_first",
                pool_pages=20000,
                capture=True,
            ),
        ),
        Cell(
            why="Step-5 tail-first dp first 512-token chunk at pos 0..511, head view 3 of 4: "
            "rows of 0..63 regions, 4-way split-K, merged",
            spec=_step5(
                ((0, 512),),
                pool_kv_heads=4,
                head_view=3,
                splits=4,
                layout="tail_first",
                pool_pages=20000,
            ),
        ),
        Cell(
            why="Step-5 tail-first tp4 512-row chunk at 4096 over a reused prefix, 1-head "
            "pool: 4-way split-K, merged",
            spec=_step5(
                ((4096, 512),),
                pool_kv_heads=1,
                head_view=None,
                splits=4,
                layout="tail_first",
                pool_pages=20000,
            ),
        ),
        Cell(
            why="Step-5 tail-first dp mixed step: a 384-row context from 0, then 128 decode "
            "rows at pos 1..9215, head view 1: 512 rows, 4-way split-K, merged",
            spec=_step5(
                ((0, 384),) + _SPREAD_128,
                pool_kv_heads=4,
                head_view=1,
                splits=4,
                layout="tail_first",
                pool_pages=20000,
            ),
        ),
        Cell(
            why="Step-5 captured decode: 1, 2, 24 and 128 rows at pos 1..9215, dp head view "
            "0, regions then tail, replay == eager",
            spec=_step5(
                (),
                pool_kv_heads=4,
                head_view=0,
                splits=8,
                pool_pages=20000,
                # (rows, split-K ways) per captured batch; each launch is merged.
                capture=dict(batches=((1, 64), (2, 64), (24, 32), (128, 8))),
            ),
        ),
        # The measured tp4/tep4 pool in the trunk layout: the last slot's planes sit
        # about 2.5e10 data bytes and 3.1e9 scale bytes into their pools.
        Cell(
            why="Step-5 measured pool, tp4 128 decode rows at pos 1..9215: 132614 slots x 46 "
            "pages, layer 22, regions then tail, 8-way split-K, merged",
            spec=_step5(
                _SPREAD_128,
                pool_kv_heads=1,
                head_view=None,
                splits=8,
                layer=22,
                **_STEP5_MEASURED_POOL,
            ),
        ),
        Cell(
            why="Step-5 measured pool, tp4 1 row at pos 9215: 132614 slots x 46 pages, layer "
            "22, regions then tail, 64-way split-K under PDL, merged",
            spec=_step5(
                ((9215, 1),),
                pool_kv_heads=1,
                head_view=None,
                splits=64,
                pdl=True,
                layer=22,
                **_STEP5_MEASURED_POOL,
            ),
        ),
        Cell(
            why="Step-5 measured pool, tp4 8192 rows from 0: 132614 slots x 46 pages, layer 22, "
            "regions then tail, one split",
            spec=_step5(
                ((0, 8192),),
                pool_kv_heads=1,
                head_view=None,
                splits=1,
                layer=22,
                **_STEP5_MEASURED_POOL,
            ),
        ),
        Cell(
            why="Step-5 measured pool, tail-first tp4 513 rows (a 512-row chunk at 4096 + a "
            "decode row at 8703): 132614 slots x 46 pages, layer 0, one split",
            spec=_step5(
                ((4096, 512), (8703, 1)),
                pool_kv_heads=1,
                head_view=None,
                splits=1,
                layout="tail_first",
                layer=0,
                **_STEP5_MEASURED_POOL,
            ),
        ),
        # The dep4 slot layout (four KV heads per rank) at half its measured slot count:
        # the last slot's planes sit about 2.6e10 data bytes (past 2^34) and 3.3e9 scale
        # bytes into their pools, read through one head's view.
        Cell(
            why="Step-5 dep4 slot pool, 128 decode rows at pos 1..9215, head view 3 of 4: "
            "34830 slots x 46 pages, layer 22, regions then tail, 8-way split-K, merged",
            spec=_step5(
                _SPREAD_128,
                pool_kv_heads=4,
                head_view=3,
                splits=8,
                layer=22,
                **_STEP5_DEP4_POOL,
            ),
        ),
        Cell(
            why="Step-5 dep4 slot pool, 1 row at pos 9215, head view 0 of 4: 34830 slots x 46 "
            "pages, layer 22, regions then tail, 64-way split-K under PDL, merged",
            spec=_step5(
                ((9215, 1),),
                pool_kv_heads=4,
                head_view=0,
                splits=64,
                pdl=True,
                layer=22,
                **_STEP5_DEP4_POOL,
            ),
        ),
        Cell(
            why="Step-5 dep4 slot pool, 8192 rows at pos 1024..9215, head view 2 of 4: 34830 "
            "slots x 46 pages, layer 22, regions then tail, one split",
            spec=_step5(
                ((1024, 8192),),
                pool_kv_heads=4,
                head_view=2,
                splits=1,
                layer=22,
                **_STEP5_DEP4_POOL,
            ),
        ),
        Cell(
            why="Step-5 dep4 slot pool, tail-first 512-row chunk at pos 0..511, head view 1 of "
            "4: 34830 slots x 46 pages, layer 0, 4-way split-K, merged",
            spec=_step5(
                ((0, 512),),
                pool_kv_heads=4,
                head_view=1,
                splits=4,
                layout="tail_first",
                layer=0,
                **_STEP5_DEP4_POOL,
            ),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. What the target
    passes: `block_table` is `qsa_block_table` (physical page ids, the unscaled
    table); `selected_tokens` is `qsa_expand_block_indices`' output, logical
    positions, `-1`-padded; `k_cache` / `v_cache` and `k_scale_cache` /
    `v_scale_cache` are the layer's planes of the QSA manager's `get_buffers(i,
    "HND")` and `get_block_scale_buffers(i, "HND")`; `kv_scale_quant_orig` is the
    fp32 [3] tensor its dense attention call binds (index 1 for K, 2 for V).

    Call sites (the target's `modeling.py`): the context/mixed forward passes the
    prefill pair on every step, the decode forward never does; both pass `q` as
    the 24-head slice of the packed QKV row (row stride 7168).

    Launch shapes at Hkv 2 (programs = rows x 2), as on the bf16 entry: split-K
    64-way up to 2 rows, 32-way up to 15 (16-token tiles), 8-way up to 128 and
    4-way up to 256 (64-token tiles), each followed by the merge; past 256 rows
    the prefill kernel when the pair is passed, else one split and no merge.
    The decode site reaches 256 rows at max_draft_len 7 (32 requests x 8).

    Equivalence: the test also runs QSA's bf16 sparse GQA over a bf16 pool
    holding the dequantized values; on every cell but the capture sweeps the two
    outputs agree within the test's band, except on rows where a split opens on
    a tile naming no token (below), which the comparison leaves out. They are
    not bitwise equal: the dequantizing load changes the compiled kernel around
    the same arithmetic.

    CUDA graphs: the decode site captures at G requests x 1..8 rows over the
    [32, 260] table, and Step-5's over the [129, 288] table at 1, 2, 24 and 128
    rows regions-then-tail and 4, 16, 32 and 120 rows tail-first (64-way
    split-K with PDL at 1, 2 and 4 rows, 32-way at 16 and 24, 8-way at 32, 120
    and 128, each merged); a replay after new operands are written in place is
    bitwise the eager call on them. The split-K partials are allocated inside
    the call.

    Compiled variants: the kernels take the pool's page count (`num_pages`), the
    table's row count (`num_requests`) and the split-K kernel the step's row
    count as plain ints, and Triton compiles each apart at 1, at multiples of 16
    and otherwise. The QSA cells run a 32-row table (a multiple of 16), the
    Step-5 cells a 129-row one (neither); a pool page count of 1 is not
    certified. Strides are plain ints too, specialised by divisibility by 16.

    Step-5-Preview (the `Step-5` cells; its kv_nvfp4 serving configuration).
    One call per KV group: q is the group's 16 heads sliced from the `[T, 64 |
    16, 256]` projection; the caches are the single-head views `planes[:, 2l +
    0 | 1, g:g+1]` of the cache manager's `get_kv_slot_buffers(l, "HND")`
    (`[slots, 2, 4 | 1, 32, 128]` uint8) and `get_kv_block_scale_slot_buffers(l,
    "HND")` (`[slots, 2, 4 | 1, 32, 16]`), indexed by slot ids; head_dim 192 is
    zero-padded to 256 in q and the pool (all-zero 16-value blocks, code 0
    under scale 0, which come back as exactly zero output columns);
    kv_scale_quant_orig ones(3); scale 192^-0.5; no prefill pair. Selection
    width 4104: 512 region slots of 8 logical positions and the current
    region's 8, `-1` past the query, the current region last
    (regions-then-tail) or first (tail-first); both layouts are cells. The
    split-K launch at Hkv 1 is chosen from the row count: up to 4 rows 64
    splits of 16-token tiles under PDL; 5..31 rows 32 splits; 32..256 rows 8
    splits of 64-token tiles; 257..512 rows 4 splits; past 512 one split and
    no merge. Regions-then-tail cells drive 1, 2, 24, 128, 513 and 8192 rows;
    tail-first cells (the order the target emits) drive 4, 16, 32, 120 and 128
    decode rows, the 512-row chunks from 0 (rows of 0..63 regions) and from
    4096, a 512-row step of a 384-row context from 0 and 128 decode rows, and
    8192 rows; at positions up to 9215 (max_seq_len 9216, every one of the 288
    table columns).

    The measured pool (the `Step-5 measured pool` cells): the pool the
    kv_nvfp4 tep4 engine allocated, 4243648 KV tokens = 132614 slots of 32,
    each 46 data pages of 4 KiB and 46 block-scale pages of 512 B (one KV head
    per rank, the 23 sparse layers' K and V coalesced per slot; 23.3 GiB of
    data and 2.9 GiB of scales). Driven at layer 22: 128 decode rows (8
    splits) and 1 row at 9215 (64 splits, PDL), regions then tail, and 8192
    rows from 0 (one split); at layer 0, tail-first 513 rows (the 512-row chunk
    at 4096 and a decode row at 8703, one split). Each reads the pool's last
    slot, whose layer-22 V data plane starts 24986415104 bytes and whose K / V
    scale planes start 3123301376 / 3123301888 bytes into their pools (both
    past 2^31). Measured against the reference, element / RMS ulp: 2.000 /
    0.433, 1.969 / 0.567, 2.000 / 0.568 and 2.000 / 0.655; against the bf16
    twin 1.012 / 0.018, 0.427 / 0.037, 1.882 / 0.010 and 1.138 / 0.018; no NaN,
    with every slot and plane the requests do not own carrying e4m3 NaN
    scales. Peak device memory 30.65 GiB.

    The dep4 slot pool (the `Step-5 dep4 slot pool` cells): the kv_nvfp4 dep4
    engine allocated 2229120 KV tokens per rank = 69660 slots of 32, each 46
    data pages of 16 KiB (4 KV heads x 32 tokens x 128 bytes) and 46
    block-scale pages of 2 KiB (48.9 GiB of data and 6.1 GiB of scales, past
    the 40 GiB cell budget). The cells hold that slot layout at 34830 slots,
    half the measured count (24.4 GiB of data, 3.1 GiB of scales), and call on
    one head's view of the four, each reading the pool's last slot: 128 decode
    rows at layer 22, head view 3 (8 splits); 1 row at 9215, head view 0 (64
    splits, PDL); 8192 rows at 1024..9215, head view 2 (one split); and,
    tail-first at layer 0, the 512-row chunk at 0..511, head view 1 (4
    splits). The last slot's planes start 26250084352..26250113024 data bytes
    (past 2^34) and 3281170944..3281264128 scale bytes into their pools.
    Measured against the reference, element / RMS ulp: 1.969 / 0.474, 1.012 /
    0.610, 2.000 / 0.653 and 2.000 / 0.552; against the bf16 twin 0.837 /
    0.017, 0.421 / 0.036, 1.984 / 0.016 and 1.153 / 0.004; no NaN, with every
    slot and plane the requests do not own carrying e4m3 NaN scales. Peak
    device memory 33.22 GiB. The measured 69660-slot dep4 pool reaches about
    5.25e10 data bytes: the same 64-bit offset class, not a cell.

    Empty leading tiles (measured on this machine, fixed in this file): in the
    regions-then-tail layout a row with fewer than 449 regions (position below
    3592) has `-1` region slots between its regions and its current region, so
    a split can open on a 64-token tile naming no token and reach the current
    region later. Without the guarded correction, `exp2(-inf - -inf)` is NaN,
    the split reports no values, and every token after that tile is dropped
    silently. The unguarded kernel
    measured, on the 128-row decode cell, 50 rows opening such a split
    (positions 1..3556), 25 of them more than 5% off and one all zero, the 50
    equal to the reference without those splits' tokens (1.98 element / 0.63
    RMS ulp); on the 8192-row one-split cell rows 0..7 came back zero. With the
    guard both cells pass at 2.0 / 0.51 and 2.0 / 0.61 ulp, and every other row
    and every QSA cell is bitwise the unguarded kernel's output.
    `test_step5_split_after_an_empty_leading_tile` finds those rows from the
    launch's split boundaries and fails if they are dropped.

    Band (Step-5 128-row decode cell, element / RMS ulp of bf16): the op 2.0 /
    0.51; wrong references land outside it -- the neighbouring KV group's views
    641 / 376, scale 256^-0.5 144 / 12.2, current region dropped 7.2e11 / 680,
    K and V swapped 668 / 391, `-1` read as token 0 293 / 248, nibbles swapped
    632 / 345, V's scales read token-major 515 / 227. The tail-first split-K
    cells, worst per launch class: 64-way 1.855 / 0.398, 32-way 1.984 / 0.407,
    8-way 2.000 / 0.439, 4-way 2.000 / 0.630. In the tail-first layout a split
    that opens on a tile naming no token holds none after it either, so no row
    of those cells depends on the empty-leading-tile guard; in the first
    512-token chunk the 4-way launch's later splits hold no token and the
    merge gives them zero weight.

    Preconditions:
      - Whenever the prefill kernel can run, each selection row is packed at the
        front with exactly the bound's count of valid entries (QSA's producer
        does this; see the bf16 entry's note).
      - The pool holds what the attention op's NVFP4 append writes; the
        `qsa_paged_kv_store_nvfp4` entry writes the same bits.
      - `D` a power of two >= 32; `Hq % Hkv == 0`; `tokens_per_block % 4 == 0`.

    The callable rejects loudly (ValueError) every operand whose dtype, shape or
    layout it cannot read (the test pins the half-passed prefill pair); `rows ==
    0` returns an empty `[0, Hq, D]` tensor (not certified). Calls are bitwise
    deterministic.

    Hard limits: every row, request, page and position index is widened to 64
    bits before any product, so a page offset past 2^31 bytes is formed in 64
    bits: the measured-pool cells read data 2.5e10 and scales 3.1e9 bytes into
    their pools and the dep4 slot pool cells data 2.6e10 and scales 3.3e9
    bytes, and pass (the largest certified extent is 8192 rows x 4104
    selected tokens through a [129, 288] table into the 132614-slot one-head
    trunk pool and the 34830-slot four-head one); positions (<= 9215) never
    pass through fp32.

    Not certified: head geometries other than 24/2 x 256 and Step-5's 16/1 x
    256 views; `tokens_per_block` other than 32; more than 8192 rows; capture of
    the context/mixed site; a pool of one slot; second-level scales other than
    1.0; the prefill pair on the split-K kernel (256 context rows or fewer), and
    at Step-5's geometry at all; one split without a merge on the QSA geometry;
    the trunk layout on the prefill kernel; Step-5 rows 3, 5..31 other than 16
    and 24, 33..256 other than 120 and 128, and 257..511 (4 splits), eager
    or captured; tail-first 1, 2 and 24 rows and regions-then-tail 4, 16,
    32, 120 and 512 rows; capture of the other padded batches (8 and 40..112
    rows) and of any 4-split launch;
    Step-5 tables other than 129 x 288 and pools other than 20000, 600,
    132614 and 34830 slots (the 132614-slot pool only at one KV head and 46
    pages per slot, at 1, 128, 513 and 8192 rows; the 34830-slot pool only at
    four KV heads and 46 pages per slot, at 1, 128, 512 and 8192 rows; both
    uncaptured, at layers 0 and 22 only); the measured 69660-slot dep4 pool
    itself (offsets up to about 5.25e10 bytes, in the same class, past the
    cell budget); rows whose selection resolves no token (all padding, a foreign
    request, unmapped pages); an fp16 or unsliced q; int64 request indices.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale_cache: torch.Tensor,
        v_scale_cache: torch.Tensor,
        kv_scale_quant_orig: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        query_positions: Optional[torch.Tensor] = None,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        return _sparse_gqa_nvfp4(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            kv_scale_quant_orig=kv_scale_quant_orig,
            block_table=block_table,
            selected_tokens=selected_tokens,
            request_indices=request_indices,
            tokens_per_block=tokens_per_block,
            softmax_scale=softmax_scale,
            query_positions=query_positions,
            compress_ratio=compress_ratio,
        )

    def reference(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale_cache: torch.Tensor,
        v_scale_cache: torch.Tensor,
        kv_scale_quant_orig: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        query_positions: Optional[torch.Tensor] = None,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        """fp32 sparse GQA over every resident selected token, rounded once.

        K and V are dequantized in fp32 and rounded to `q`'s dtype, as the kernel
        does. `query_positions` and `compress_ratio` are ignored: on a
        prefix-packed selection the prefill kernel's bound reads exactly the valid
        entries. Walked 32 rows at a time to bound the gathered K/V.
        """
        rows, num_q_heads, head_dim = q.shape
        num_pages, num_kv_heads = k_cache.shape[0], k_cache.shape[1]
        groups = num_q_heads // num_kv_heads
        num_requests, table_width = block_table.shape
        requests = request_indices.long()
        out = torch.empty(rows, num_q_heads, head_dim, dtype=torch.float32, device=q.device)
        for start in range(0, rows, 32):
            stop = min(start + 32, rows)
            chunk = selected_tokens[start:stop].long()
            logical_page = chunk.clamp_min(0) // tokens_per_block
            in_page = chunk.clamp_min(0) % tokens_per_block
            request = requests[start:stop]
            owned = (request >= 0) & (request < num_requests)
            physical = block_table[
                request.clamp(0, num_requests - 1)[:, None].expand_as(logical_page),
                logical_page.clamp(max=table_width - 1),
            ].long()
            resident = (
                (chunk >= 0)
                & owned[:, None]
                & (logical_page < table_width)
                & (physical >= 0)
                & (physical < num_pages)
            )
            physical = physical.clamp(0, num_pages - 1)
            keys = _dequantize_tokens(
                k_cache, k_scale_cache, physical, in_page, kv_scale_quant_orig[1], grouped=False
            )
            values = _dequantize_tokens(
                v_cache, v_scale_cache, physical, in_page, kv_scale_quant_orig[2], grouped=True
            )
            keys = keys.to(q.dtype).float()
            values = values.to(q.dtype).float()
            grouped = q[start:stop].reshape(-1, num_kv_heads, groups, head_dim).float()
            scores = torch.einsum("bhgd,bkhd->bhgk", grouped, keys) * softmax_scale
            scores = scores.masked_fill(~resident[:, None, None, :], float("-inf"))
            weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            pooled = torch.einsum("bhgk,bkhd->bhgd", weights, values)
            out[start:stop] = pooled.reshape(stop - start, num_q_heads, head_dim)
        return out.to(q.dtype)


qsa_paged_sparse_gqa_nvfp4 = QsaPagedSparseGqaNvfp4()
