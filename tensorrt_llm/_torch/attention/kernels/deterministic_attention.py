# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal attention with a phase-independent per-query reduction."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _attention(
    Q,
    K,
    V,
    OUTPUT,
    PAGES,
    LENGTHS,
    CU_Q,
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
    SCALE: tl.constexpr,
    SHARED: tl.constexpr,
    FIXED_Q: tl.constexpr,
    B: tl.constexpr,
    BD: tl.constexpr,
):
    row, head, seq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if FIXED_Q:
        q_start, q_len = seq * FIXED_Q, FIXED_Q
    else:
        q_start = tl.load(CU_Q + seq)
        q_len = tl.load(CU_Q + seq + 1) - q_start
    if row >= q_len:
        return
    kv_len = tl.load(LENGTHS + seq)
    end = kv_len - q_len + row + 1
    dims = tl.arange(0, BD)
    positions = tl.arange(0, B)
    output_offset = ((q_start + row).to(tl.int64) * H + head) * D + dims
    if end <= 0:
        tl.store(OUTPUT + output_offset, 0, dims < D)
        return
    q = tl.load(Q + (q_start + row).to(tl.int64) * QS0 + head * QS1 + dims, dims < D, 0).to(
        tl.float32
    )
    kv_head = head // (H // HK)
    maximum = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0, tl.float32)
    acc = tl.full((BD,), 0, tl.float32)
    for start in range(0, end, B):
        tokens = start + positions
        valid = tokens < end
        page = tl.load(PAGES + seq * PS0 + tokens // PAGE, valid, 0).to(tl.int64)
        if SHARED:
            v_page = page
        else:
            v_page = tl.load(PAGES + seq * PS0 + PS1 + tokens // PAGE, valid, 0).to(tl.int64)
        k = tl.load(
            K
            + page[:, None] * KS0
            + kv_head * KS1
            + (tokens % PAGE)[:, None] * KS2
            + dims[None, :],
            valid[:, None] & (dims[None, :] < D),
            0,
        ).to(tl.float32)
        v = tl.load(
            V
            + v_page[:, None] * VS0
            + kv_head * VS1
            + (tokens % PAGE)[:, None] * VS2
            + dims[None, :],
            valid[:, None] & (dims[None, :] < D),
            0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 0))
        alpha = tl.exp(maximum - new_maximum)
        p = tl.exp(scores - new_maximum)
        denom = denom * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        maximum = new_maximum
    tl.store(OUTPUT + output_offset, acc / denom, dims < D)


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
) -> None:
    """Causal paged attention with a fixed FP32 reduction for every query.

    Q is token-major, K/V caches are HND, and output is contiguous. Separate
    K/V page tables have shape [batch, 2, pages]; shared tables [batch, pages].
    The last query in each sequence aligns with its last KV token.
    """
    if query.numel() == 0:
        return
    _attention[(max_query_len, query.shape[1], seq_lens.numel())](
        query,
        key_cache,
        value_cache,
        output,
        block_tables,
        seq_lens,
        cu_query_lens,
        *query.stride()[:2],
        *key_cache.stride()[:3],
        *value_cache.stride()[:3],
        block_tables.stride(0),
        0 if shared_page_indices else block_tables.stride(1),
        query.shape[1],
        key_cache.shape[1],
        query.shape[-1],
        key_cache.shape[2],
        scale,
        shared_page_indices,
        fixed_query_len,
        32,
        tr.next_power_of_2(query.shape[-1]),
        num_warps=4,
    )
