# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal paged attention with a phase-independent per-query reduction.

Arithmetic of one query row, identical in prefill, chunked prefill and decode and
independent of the other rows of the batch:

* Keys are walked in fixed tiles aligned to position 0 and grouped into fixed segments,
  also aligned to position 0.
* Inside a segment the softmax is accumulated online in base 2 with an integer running
  exponent, so every rescale multiplies by an exact power of two. QK and PV are Tensor
  Core dots; the probabilities are rounded to BF16 before PV, and their row sum is taken
  through a Tensor Core dot against a one-hot column, so every reduction over keys has an
  order that does not depend on which rows share a program.
* Segments are merged in ascending order. The merge multiplies by exact powers of two and
  adds, so it does not depend on FMA contraction.

One program serves a block of query positions of one sequence for all query heads that
share one KV head. Prefill walks every segment of its rows in one program. Decode can
instead launch one program per segment, which run in parallel, and merge the segment
results with a second kernel; both layouts give the same bits.
"""

import torch
import triton as tr
import triton.language as tl

_LOG2E = 1.4426950408889634
# Keys per segment. Part of the arithmetic: changing it changes the output bits.
_SEGMENT_KEYS = 1024


@tr.jit
def _pow2(k):
    """2**k for a non-positive integer-valued (or -inf) float tensor; 0 below 2**-126."""
    clipped = tl.maximum(k, -127.0).to(tl.int32)
    return tl.where(k < -126.0, 0.0, ((clipped + 127) << 23).to(tl.float32, bitcast=True))


@tr.jit
def _attention(
    Q,
    K,
    V,
    OUTPUT,
    WS,
    PAGES,
    LENGTHS,
    CU_Q,
    NSEG,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE2: tl.constexpr,
    SHARED: tl.constexpr,
    FIXED_Q: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    BQ: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    SEGMENT: tl.constexpr,
    SPLIT: tl.constexpr,
):
    if SPLIT:
        segment_id, kv_head, seq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        tile = seq - seq
    else:
        tile, kv_head, seq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        segment_id = 0
    if FIXED_Q:
        q_start, q_len = seq * FIXED_Q, FIXED_Q
    else:
        q_start = tl.load(CU_Q + seq)
        q_len = tl.load(CU_Q + seq + 1) - q_start
    if tile * BQ >= q_len:
        return
    kv_len = tl.load(LENGTHS + seq)
    rows = tl.arange(0, BQ * GP)
    pos = tile * BQ + rows // GP
    slot = rows % GP
    head = kv_head * G + slot
    row_valid = (pos < q_len) & (slot < G)
    end = kv_len - q_len + pos + 1
    tile_end = kv_len - q_len + tl.minimum((tile + 1) * BQ, q_len)
    dims = tl.arange(0, BD)
    keys = tl.arange(0, BN)
    out_mask = row_valid[:, None] & (dims < D)[None, :]
    out_row = (q_start + pos).to(tl.int64) * H + head
    out_ptr = OUTPUT + out_row[:, None] * D + dims[None, :]
    if tile_end <= 0:
        if not SPLIT:
            tl.store(out_ptr, 0, out_mask)
        return
    num_segments = (tile_end + SEGMENT - 1) // SEGMENT
    if SPLIT:
        if segment_id >= num_segments:
            return
        first_segment = segment_id
        stop_segment = segment_id + 1
    else:
        first_segment = 0
        stop_segment = num_segments
    q = tl.load(
        Q + (q_start + pos).to(tl.int64)[:, None] * QS0 + head[:, None] * QS1 + dims[None, :],
        out_mask,
        0,
    )
    onehot = ((tl.arange(0, 16) == 0)[None, :] & (keys >= 0)[:, None]).to(tl.bfloat16)
    if not SPLIT:
        merged_exponent = tl.full((BQ * GP,), -float("inf"), tl.float32)
        merged_denom = tl.full((BQ * GP,), 0, tl.float32)
        merged_acc = tl.full((BQ * GP, BD), 0, tl.float32)
    for segment in range(first_segment, stop_segment):
        exponent = tl.full((BQ * GP,), -float("inf"), tl.float32)
        denom = tl.full((BQ * GP,), 0, tl.float32)
        acc = tl.full((BQ * GP, BD), 0, tl.float32)
        segment_stop = tl.minimum((segment + 1) * SEGMENT, tile_end)
        for start in range(segment * SEGMENT, segment_stop, BN):
            tokens = start + keys
            valid = tokens < tile_end
            k_page = tl.load(PAGES + seq * PS0 + tokens // PAGE, valid, 0).to(tl.int64)
            if SHARED:
                v_page = k_page
            else:
                v_page = tl.load(PAGES + seq * PS0 + PS1 + tokens // PAGE, valid, 0).to(tl.int64)
            k = tl.load(
                K
                + k_page[None, :] * KS0
                + kv_head * KS1
                + (tokens % PAGE)[None, :] * KS2
                + dims[:, None],
                valid[None, :] & (dims < D)[:, None],
                0,
            )
            v = tl.load(
                V
                + v_page[:, None] * VS0
                + kv_head * VS1
                + (tokens % PAGE)[:, None] * VS2
                + dims[None, :],
                valid[:, None] & (dims < D)[None, :],
                0,
            )
            scores = tl.dot(q, k) * SCALE2
            scores = tl.where(
                (tokens[None, :] < end[:, None]) & row_valid[:, None], scores, -float("inf")
            )
            new_exponent = tl.maximum(exponent, tl.ceil(tl.max(scores, 1)))
            safe_exponent = tl.where(new_exponent == -float("inf"), 0.0, new_exponent)
            alpha = _pow2(exponent - safe_exponent)
            p = tl.exp2(scores - safe_exponent[:, None]).to(tl.bfloat16)
            acc = tl.dot(p, v, acc * alpha[:, None])
            denom = denom * alpha + tl.sum(tl.dot(p, onehot), 1)
            exponent = new_exponent
        if SPLIT:
            ws_row = ((seq.to(tl.int64) * HK + kv_head) * NSEG + segment) * (BQ * GP) + rows
            ws_ptr = WS + ws_row * (BD + 2)
            tl.store(ws_ptr[:, None] + dims[None, :], acc)
            tl.store(ws_ptr + BD, exponent)
            tl.store(ws_ptr + BD + 1, denom)
        else:
            total_exponent = tl.maximum(merged_exponent, exponent)
            safe_total = tl.where(total_exponent == -float("inf"), 0.0, total_exponent)
            scale_merged = _pow2(merged_exponent - safe_total)
            scale_segment = _pow2(exponent - safe_total)
            merged_denom = merged_denom * scale_merged + denom * scale_segment
            merged_acc = merged_acc * scale_merged[:, None] + acc * scale_segment[:, None]
            merged_exponent = total_exponent
    if not SPLIT:
        result = merged_acc / tl.where(merged_denom > 0, merged_denom, 1.0)[:, None]
        tl.store(out_ptr, result, out_mask)


@tr.jit
def _merge_segments(
    OUTPUT,
    WS,
    LENGTHS,
    CU_Q,
    NSEG,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    FIXED_Q: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    BQ: tl.constexpr,
    BD: tl.constexpr,
    SEGMENT: tl.constexpr,
):
    kv_head, seq = tl.program_id(0), tl.program_id(1)
    if FIXED_Q:
        q_start, q_len = seq * FIXED_Q, FIXED_Q
    else:
        q_start = tl.load(CU_Q + seq)
        q_len = tl.load(CU_Q + seq + 1) - q_start
    if q_len <= 0:
        return
    kv_len = tl.load(LENGTHS + seq)
    rows = tl.arange(0, BQ * GP)
    pos = rows // GP
    slot = rows % GP
    head = kv_head * G + slot
    row_valid = (pos < q_len) & (slot < G)
    tile_end = kv_len - q_len + tl.minimum(seq - seq + BQ, q_len)
    dims = tl.arange(0, BD)
    out_mask = row_valid[:, None] & (dims < D)[None, :]
    out_row = (q_start + pos).to(tl.int64) * H + head
    out_ptr = OUTPUT + out_row[:, None] * D + dims[None, :]
    if tile_end <= 0:
        tl.store(out_ptr, 0, out_mask)
        return
    num_segments = (tile_end + SEGMENT - 1) // SEGMENT
    merged_exponent = tl.full((BQ * GP,), -float("inf"), tl.float32)
    merged_denom = tl.full((BQ * GP,), 0, tl.float32)
    merged_acc = tl.full((BQ * GP, BD), 0, tl.float32)
    for segment in range(0, num_segments):
        ws_row = ((seq.to(tl.int64) * HK + kv_head) * NSEG + segment) * (BQ * GP) + rows
        ws_ptr = WS + ws_row * (BD + 2)
        acc = tl.load(ws_ptr[:, None] + dims[None, :])
        exponent = tl.load(ws_ptr + BD)
        denom = tl.load(ws_ptr + BD + 1)
        new_exponent = tl.maximum(merged_exponent, exponent)
        safe_exponent = tl.where(new_exponent == -float("inf"), 0.0, new_exponent)
        scale_merged = _pow2(merged_exponent - safe_exponent)
        scale_segment = _pow2(exponent - safe_exponent)
        merged_denom = merged_denom * scale_merged + denom * scale_segment
        merged_acc = merged_acc * scale_merged[:, None] + acc * scale_segment[:, None]
        merged_exponent = new_exponent
    result = merged_acc / tl.where(merged_denom > 0, merged_denom, 1.0)[:, None]
    tl.store(out_ptr, result, out_mask)


def _decode_rows(group_padded: int) -> int:
    """Query positions per decode program; a Tensor Core dot needs at least 16 rows."""
    return max(1, 16 // group_padded)


def split_workspace_bytes(
    num_seqs: int, num_heads: int, num_kv_heads: int, head_dim: int, max_kv_len: int
) -> int:
    """FP32 workspace that lets decode run one program per key segment."""
    group_padded = tr.next_power_of_2(num_heads // num_kv_heads)
    rows = _decode_rows(group_padded) * group_padded
    num_segments = tr.cdiv(max_kv_len, _SEGMENT_KEYS)
    return num_seqs * num_kv_heads * num_segments * rows * (tr.next_power_of_2(head_dim) + 2) * 4


def deterministic_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_query_lens: torch.Tensor | None,
    fixed_query_len: int,
    max_query_len: int,
    scale: float,
    shared_page_indices: bool,
    output: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> None:
    """Causal paged attention with the same arithmetic for every query in every phase.

    Q is token-major, K/V caches are HND, and output is contiguous. Separate
    K/V page tables have shape [batch, 2, pages]; shared tables [batch, pages].
    The last query in each sequence aligns with its last KV token.

    ``workspace`` is an optional byte buffer. When it holds at least
    ``split_workspace_bytes`` for this call, decode runs its key segments in
    parallel; otherwise each program walks its segments serially. The output
    bits are the same either way.
    """
    if query.numel() == 0:
        return
    heads, kv_heads, dim = query.shape[1], key_cache.shape[1], query.shape[-1]
    page = key_cache.shape[2]
    group = heads // kv_heads
    group_padded = tr.next_power_of_2(group)
    decode_bq = _decode_rows(group_padded)
    padded_dim = tr.next_power_of_2(dim)
    # The key tile is part of the arithmetic and depends only on the head
    # dimension. The row block is not: every block size gives the same bits.
    key_tile = 128 if dim <= 128 else 64
    num_stages = 3 if dim <= 128 else 2
    num_seqs = seq_lens.numel()
    max_kv_len = block_tables.shape[-1] * page
    split = max_query_len <= decode_bq and workspace is not None
    if split:
        needed = split_workspace_bytes(num_seqs, heads, kv_heads, dim, max_kv_len)
        split = workspace.numel() * workspace.element_size() >= needed
    if split:
        bq = decode_bq
        num_segments = tr.cdiv(max_kv_len, _SEGMENT_KEYS)
        partials = workspace.view(-1)[: needed // workspace.element_size()].view(torch.float32)
        grid = (num_segments, kv_heads, num_seqs)
    else:
        prefill_rows = 128 if dim <= 128 else 64
        bq = (
            decode_bq
            if max_query_len <= decode_bq
            else max(decode_bq, prefill_rows // group_padded)
        )
        num_segments = 1
        partials = output
        grid = (tr.cdiv(max_query_len, bq), kv_heads, num_seqs)
    _attention[grid](
        query,
        key_cache,
        value_cache,
        output,
        partials,
        block_tables,
        seq_lens,
        cu_query_lens,
        num_segments,
        *query.stride()[:2],
        *key_cache.stride()[:3],
        *value_cache.stride()[:3],
        block_tables.stride(0),
        0 if shared_page_indices else block_tables.stride(1),
        heads,
        kv_heads,
        dim,
        page,
        scale * _LOG2E,
        shared_page_indices,
        fixed_query_len,
        group,
        group_padded,
        bq,
        key_tile,
        padded_dim,
        _SEGMENT_KEYS,
        split,
        num_warps=4 if bq * group_padded <= 16 else 8,
        num_stages=num_stages,
    )
    if split:
        _merge_segments[(kv_heads, num_seqs)](
            output,
            partials,
            seq_lens,
            cu_query_lens,
            num_segments,
            heads,
            kv_heads,
            dim,
            fixed_query_len,
            group,
            group_padded,
            bq,
            padded_dim,
            _SEGMENT_KEYS,
            num_warps=4,
        )
