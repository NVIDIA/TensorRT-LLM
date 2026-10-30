# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA over an NVFP4 paged pool: attention over explicitly selected tokens."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Approved exception (as for `qsa_paged_sparse_gqa`): QSA's sparse path has no torch.ops
# surface; its row-count PDL policy is reached as Python. The split-K buckets are this
# entry's own: an NVFP4 tile needs 16 tokens where the bf16 one takes 64.
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import _qsa_pdl_enabled

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

#: The prefill kernel's (BLOCK_N, warps): on GB300 8192 context rows 6.5 ms, 4096 rows 2.9 ms.
_PREFILL_TILE = (32, 2)

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
def _nvfp4_halves(packed, scales):
    """`[N, D/2]` packed e2m1 bytes (low nibble the even element) and one e4m3 scale per byte
    -> the even and the odd elements as two `[N, D/2]` f16 tensors, each `e2m1 * scale`:
    `cvt.rn.f16x2.e2m1x2` and `cvt.rn.f16x2.e4m3x2`, then one f16x2 multiply, exact (two by
    four significant bits, inside f16's range)."""
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 b0, b1, b2, b3, s0, s1, s2, s3;
        .reg .b16 t0, t1, t2, t3;
        .reg .b32 h0, h1, h2, h3, c0, c1, c2, c3;
        mov.b32 {b0, b1, b2, b3}, $4;
        mov.b32 {s0, s1, s2, s3}, $5;
        mov.b16 t0, {s0, s0};
        mov.b16 t1, {s1, s1};
        mov.b16 t2, {s2, s2};
        mov.b16 t3, {s3, s3};
        cvt.rn.f16x2.e4m3x2 c0, t0;
        cvt.rn.f16x2.e4m3x2 c1, t1;
        cvt.rn.f16x2.e4m3x2 c2, t2;
        cvt.rn.f16x2.e4m3x2 c3, t3;
        cvt.rn.f16x2.e2m1x2 h0, b0;
        cvt.rn.f16x2.e2m1x2 h1, b1;
        cvt.rn.f16x2.e2m1x2 h2, b2;
        cvt.rn.f16x2.e2m1x2 h3, b3;
        mul.rn.f16x2 h0, h0, c0;
        mul.rn.f16x2 h1, h1, c1;
        mul.rn.f16x2 h2, h2, c2;
        mul.rn.f16x2 h3, h3, c3;
        prmt.b32 $0, h0, h1, 0x5410;
        prmt.b32 $1, h2, h3, 0x5410;
        prmt.b32 $2, h0, h1, 0x7632;
        prmt.b32 $3, h2, h3, 0x7632;
        }
        """,
        constraints="=r,=r,=r,=r,r,r",
        args=[packed, scales],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=4,
    )


@triton.jit
def _load_halves(
    cache,
    scale_rows,
    pages,
    kv_head,
    tokens,
    valid,
    stride_page,
    stride_head,
    stride_token,
    SCALE_STEP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """A tile's even and odd elements of one role plane. `scale_rows` points at each token's
    first scale; its D/16 scales lie SCALE_STEP apart (K's token-major rows: 1; V's
    four-token groups `[tpb/4, D/16, 4]`: 4). Each byte and each scale is read once."""
    HALF: tl.constexpr = HEAD_DIM // 2
    GROUPS: tl.constexpr = HEAD_DIM // 16
    packed = tl.load(
        cache
        + pages[:, None] * stride_page
        + kv_head * stride_head
        + tokens[:, None] * stride_token
        + tl.arange(0, HALF)[None, :],
        mask=valid[:, None],
        other=0,
    )
    scales = tl.load(
        scale_rows[:, None] + tl.arange(0, GROUPS)[None, :] * SCALE_STEP,
        mask=valid[:, None],
        other=0,
    )
    scales = tl.reshape(tl.broadcast_to(scales[:, :, None], [BLOCK_N, GROUPS, 8]), [BLOCK_N, HALF])
    return _nvfp4_halves(packed, scales)


@triton.jit
def _attend_tile(
    q_even,
    q_odd,
    running_max,
    running_sum,
    acc_even,
    acc_odd,
    columns,
    row,
    safe_request,
    valid_request,
    kv_head,
    score_scale,
    selected_tokens,
    selected_stride_row,
    block_table,
    block_table_stride_request,
    page_table_width,
    num_cache_pages,
    k_cache,
    v_cache,
    k_scale_cache,
    v_scale_cache,
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
    TOKENS_PER_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One online-softmax step over the selected columns `columns` of `row`: the even and odd
    halves of K against q's, scores in the log2 domain, P rounded to f16 against V's halves."""
    GROUPS: tl.constexpr = HEAD_DIM // 16
    logical_tokens = tl.load(
        selected_tokens + row * selected_stride_row + columns, mask=columns < TOPK, other=-1
    )
    safe_tokens = tl.maximum(logical_tokens, 0)
    logical_pages = safe_tokens // TOKENS_PER_BLOCK
    tokens = safe_tokens % TOKENS_PER_BLOCK
    valid = (
        valid_request
        & (columns < TOPK)
        & (logical_tokens >= 0)
        & (logical_pages < page_table_width)
    )
    pages = tl.load(
        block_table
        + safe_request * block_table_stride_request
        + tl.minimum(logical_pages, page_table_width - 1),
        mask=valid,
        other=-1,
    ).to(tl.int64)
    valid &= (pages >= 0) & (pages < num_cache_pages)
    pages = tl.maximum(pages, 0)

    k_even, k_odd = _load_halves(
        k_cache,
        k_scale_cache + pages * ks_stride_page + kv_head * ks_stride_head + tokens * GROUPS,
        pages,
        kv_head,
        tokens,
        valid,
        k_stride_page,
        k_stride_head,
        k_stride_token,
        1,
        BLOCK_N,
        HEAD_DIM,
    )
    scores = tl.dot(q_even, tl.trans(k_even))
    scores = tl.dot(q_odd, tl.trans(k_odd), scores)
    scores = tl.where(valid[None, :], scores * score_scale, -float("inf"))
    next_max = tl.maximum(running_max, tl.max(scores, axis=1))
    correction = tl.math.exp2(running_max - next_max)
    probabilities = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)

    v_even, v_odd = _load_halves(
        v_cache,
        v_scale_cache
        + pages * vs_stride_page
        + kv_head * vs_stride_head
        + (tokens // 4) * (4 * GROUPS)
        + tokens % 4,
        pages,
        kv_head,
        tokens,
        valid,
        v_stride_page,
        v_stride_head,
        v_stride_token,
        4,
        BLOCK_N,
        HEAD_DIM,
    )
    weights = probabilities.to(tl.float16)
    acc_even = tl.dot(weights, v_even, acc_even * correction[:, None])
    acc_odd = tl.dot(weights, v_odd, acc_odd * correction[:, None])
    running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
    return next_max, running_sum, acc_even, acc_odd


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
    # The global scales scale the scores and the normalized output, not each element.
    score_scale = softmax_scale * _LOG2_E * tl.load(kv_scale_quant_orig + 1)
    v_global = tl.load(kv_scale_quant_orig + 2)

    HALF: tl.constexpr = HEAD_DIM // 2
    head_offsets = tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HALF)
    token_offsets = tl.arange(0, BLOCK_N)
    query_heads = kv_head * GROUP_SIZE + head_offsets
    head_mask = (head_offsets < GROUP_SIZE)[:, None]
    # q's even and odd elements as f16, so both MMA operands are f16 (q is bf16 within
    # f16's range after its norm; an NVFP4 value times its scale is exact in f16).
    q_even_ptr = (
        q + row * q_stride_row + query_heads[:, None] * q_stride_head + 2 * half_offsets[None, :]
    )
    q_even = tl.load(q_even_ptr, mask=head_mask, other=0.0).to(tl.float16)
    q_odd = tl.load(q_even_ptr + 1, mask=head_mask, other=0.0).to(tl.float16)

    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    acc_even = tl.zeros([BLOCK_M, HALF], tl.float32)
    acc_odd = tl.zeros([BLOCK_M, HALF], tl.float32)
    visible_tokens = tl.load(query_positions + row).to(tl.int64) + 1
    selected_blocks = tl.minimum(visible_tokens // COMPRESS_RATIO, BLOCK_TOPK)
    valid_tokens = selected_blocks * COMPRESS_RATIO + visible_tokens % COMPRESS_RATIO
    valid_tiles = tl.cdiv(tl.minimum(valid_tokens, TOPK), BLOCK_N)
    for tile in tl.range(0, valid_tiles, num_stages=2):
        columns = tile * BLOCK_N + token_offsets
        running_max, running_sum, acc_even, acc_odd = _attend_tile(
            q_even,
            q_odd,
            running_max,
            running_sum,
            acc_even,
            acc_odd,
            columns,
            row,
            safe_request,
            valid_request,
            kv_head,
            score_scale,
            selected_tokens,
            selected_stride_row,
            block_table,
            block_table_stride_request,
            page_table_width,
            num_cache_pages,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
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
            TOKENS_PER_BLOCK,
            TOPK,
            BLOCK_N,
            HEAD_DIM,
        )

    denominator = tl.maximum(running_sum[:, None], 1.0e-20)
    has_values = running_sum[:, None] > 0
    out_ptr = (
        output
        + row * output_stride_row
        + query_heads[:, None] * output_stride_head
        + 2 * half_offsets[None, :]
    )
    tl.store(out_ptr, tl.where(has_values, acc_even / denominator, 0.0) * v_global, mask=head_mask)
    tl.store(
        out_ptr + 1, tl.where(has_values, acc_odd / denominator, 0.0) * v_global, mask=head_mask
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
    # The global scales scale the scores and the normalized output, not each element.
    score_scale = softmax_scale * _LOG2_E * tl.load(kv_scale_quant_orig + 1)
    v_global = tl.load(kv_scale_quant_orig + 2)

    HALF: tl.constexpr = HEAD_DIM // 2
    head_offsets = tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HALF)
    token_offsets = tl.arange(0, BLOCK_N)
    query_heads = kv_head * GROUP_SIZE + head_offsets
    head_mask = (head_offsets < GROUP_SIZE)[:, None]
    # q's even and odd elements as f16, so both MMA operands are f16 (q is bf16 within
    # f16's range after its norm; an NVFP4 value times its scale is exact in f16).
    q_even_ptr = (
        q + row * q_stride_row + query_heads[:, None] * q_stride_head + 2 * half_offsets[None, :]
    )
    q_even = tl.load(q_even_ptr, mask=head_mask, other=0.0).to(tl.float16)
    q_odd = tl.load(q_even_ptr + 1, mask=head_mask, other=0.0).to(tl.float16)

    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    acc_even = tl.zeros([BLOCK_M, HALF], tl.float32)
    acc_odd = tl.zeros([BLOCK_M, HALF], tl.float32)
    split_tile_start = split * NUM_TILES // NUM_SPLITS
    split_tile_end = (split + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + token_offsets
        running_max, running_sum, acc_even, acc_odd = _attend_tile(
            q_even,
            q_odd,
            running_max,
            running_sum,
            acc_even,
            acc_odd,
            columns,
            row,
            safe_request,
            valid_request,
            kv_head,
            score_scale,
            selected_tokens,
            selected_stride_row,
            block_table,
            block_table_stride_request,
            page_table_width,
            num_cache_pages,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
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
            TOKENS_PER_BLOCK,
            TOPK,
            BLOCK_N,
            HEAD_DIM,
        )

    has_values = running_sum > 0
    denominator = tl.maximum(running_sum[:, None], 1.0e-20)
    even = tl.where(has_values[:, None], acc_even / denominator, 0.0) * v_global
    odd = tl.where(has_values[:, None], acc_odd / denominator, 0.0) * v_global
    if NUM_SPLITS == 1:
        out_ptr = (
            output
            + row * output_stride_row
            + query_heads[:, None] * output_stride_head
            + 2 * half_offsets[None, :]
        )
    else:
        out_ptr = (
            partial_output
            + ((split * num_rows + row) * NUM_QUERY_HEADS + query_heads[:, None]) * HEAD_DIM
            + 2 * half_offsets[None, :]
        )
        partial_lse_values = tl.where(
            has_values,
            running_max + tl.math.log2(tl.maximum(running_sum, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_lse + (split * num_rows + row) * NUM_QUERY_HEADS + query_heads,
            partial_lse_values,
            mask=head_offsets < GROUP_SIZE,
        )
    tl.store(out_ptr, even, mask=head_mask)
    tl.store(out_ptr + 1, odd, mask=head_mask)
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


def _splitk_launch_config(
    rows: int, num_kv_heads: int, selected_width: int
) -> tuple[int, int, int]:
    """(BLOCK_N, splits, warps) by programs = rows x Hkv, the fastest of a sweep on GB300 at
    2051 selected tokens a row (1 row 12.6 us, 128 rows 160 us, 512 rows 560 us)."""
    base_programs = rows * num_kv_heads
    if base_programs <= 4:
        block_n, target_splits, num_warps = 32, 64, 4
    elif base_programs <= 16:
        block_n, target_splits, num_warps = 32, 16, 4
    elif base_programs <= 32:
        block_n, target_splits, num_warps = 32, 8, 4
    elif base_programs <= 64:
        block_n, target_splits, num_warps = 32, 4, 4
    elif base_programs <= 128:
        block_n, target_splits, num_warps = 64, 4, 2
    elif base_programs <= 256:
        block_n, target_splits, num_warps = 32, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, num_warps = 32, 4, 2
    elif base_programs <= 1024:
        block_n, target_splits, num_warps = 32, 2, 2
    else:
        block_n, target_splits, num_warps = 32, 1, 2
    num_tiles = triton.cdiv(selected_width, block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    return block_n, min(max_useful_splits, target_splits), num_warps


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
            BLOCK_N=_PREFILL_TILE[0],
            HEAD_DIM=head_dim,
            num_warps=_PREFILL_TILE[1],
            num_stages=2,
        )
        return output

    block_n, num_splits, num_warps = _splitk_launch_config(
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
        K[t]   = e2m1(k_cache) * e4m3(k_scale_cache)            # exact in f16
        V[t]   = e2m1(v_cache) * e4m3(v_scale_cache)
        s_j    = (q[r, h] . K[t]) * softmax_scale * kv_scale_quant_orig[1]
        out[r, h] = kv_scale_quant_orig[2] * sum_j softmax(s)_j * V[t]

    The pool layout is the attention op's own NVFP4 append: e2m1 nibbles per
    head page `[tpb, D/2]` (low nibble = even element), K's e4m3 scales
    token-major `[tpb, D/16]`, V's in four-token groups `[tpb/4, D/16, 4]`.
    The kernels decode each byte with `cvt.rn.f16x2.e2m1x2` and scale it in f16,
    which is exact, and run the scores and P.V on f16 MMAs with fp32
    accumulation (q converted to f16, P rounded to f16 before P.V), with the
    online softmax and split-K merge of QSA's bf16 kernels. The second-level
    scales scale the scores and the normalized output, not each element.

    A negative index, a page past the table's width, a negative or out-of-pool
    table entry, and a request index outside the table's rows contribute
    nothing; a row left with nothing returns zeros. Returns a fresh, contiguous
    `[rows, Hq, D]` tensor of `q.dtype`.

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

    Launch shapes at Hkv 2 (programs = rows x 2): split-K at 32-token tiles
    64-way up to 2 rows, 16-way up to 8, 8-way up to 16 and 4-way up to 32
    (4 warps); 64-token tiles 4-way up to 64 rows; 32-token tiles 8-way up to
    128, 4-way up to 256 and 2-way up to 512 rows (2 warps); each followed by
    the merge. Past 256 rows the prefill kernel (32-token tiles, 2 warps) when the pair
    is passed; past 512 rows otherwise one split and no merge. The decode site
    reaches 256 rows at max_draft_len 7 (32 requests x 8).

    Each byte is decoded once, into the even and the odd halves of the head
    dimension, which the kernels keep apart: q's halves against K's on two MMAs,
    P against V's halves into two accumulators, the output stored back
    interleaved. On GB300 at 2051 selected tokens a row: 1 row 12.6 us, 128 rows
    160 us, against 10.4 / 108 us for QSA's bf16 kernel over the same values.

    Equivalence: the test also runs QSA's bf16 sparse GQA over a bf16 pool
    holding the dequantized values; on every cell the two outputs agree within
    the test's band. They are not bitwise equal: P is rounded to f16 here and to
    bf16 there, and the second-level scales act on the scores and the output.

    CUDA graphs: the decode site captures at G requests x 1..8 rows over the
    [32, 260] table; a replay after new operands are written in place is
    bitwise the eager call on them.

    Compiled variants: the kernels take the pool's page count (`num_pages`), the
    table's row count (`num_requests`) and the split-K kernel the step's row
    count as plain ints, and Triton compiles each apart at 1, at multiples of 16
    and otherwise. Every cell runs a 32-row table (a multiple of 16); the other
    table-height classes are not certified. Every other plain int -- the strides, the table width 260 and
    the selection's row stride 2051 -- is fixed at the served value the cells use.

    Preconditions:
      - Whenever the prefill kernel can run, each selection row is packed at the
        front with exactly the bound's count of valid entries (QSA's producer
        does this; see the bf16 entry's note).
      - The pool holds what the attention op's NVFP4 append writes; the
        `qsa_paged_kv_store_nvfp4` entry writes the same bits.
      - `D` a power of two >= 32; `Hq % Hkv == 0`; `tokens_per_block % 4 == 0`.
      - `q` within f16's range: the kernels convert it (the target's q follows its
        RMS norm).
      - An SM 100-family GPU: the byte decode is `cvt.rn.f16x2.e2m1x2`.

    The callable rejects loudly (ValueError) every operand whose dtype, shape or
    layout it cannot read (the test pins the half-passed prefill pair); `rows ==
    0` returns an empty `[0, Hq, D]` tensor (not certified). Calls are bitwise
    deterministic.

    Every row, request, page and position index is widened to 64 bits before any
    product. Not certified: head geometries other than 24/2 x 256;
    `tokens_per_block` other than 32; more than 8192 rows; capture of the
    context/mixed site; a pool of one slot; second-level scales other than 1.0;
    the prefill pair on the split-K kernel (256 context rows or fewer); one split
    without a merge; the trunk layout on the prefill kernel; rows whose selection
    resolves no token (all padding, a foreign request, unmapped pages); an fp16
    or unsliced q; int64 request indices.
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
