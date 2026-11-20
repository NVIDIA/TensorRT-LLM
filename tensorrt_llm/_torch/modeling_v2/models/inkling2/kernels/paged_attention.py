# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of Inkling 2's paged attention: K/V append, and attention with a relative bias,
an attention sink and an output gate.

The KV pool is one layer's `KVCacheManagerV2.get_buffers(layer, "HND")` view, `[pages, 2, kv
heads, page size, head dim]`; `page_table[r, i]` is the pool page of request `r`'s tokens
`i * page_size ..`. A step holds the new tokens of several requests, request-major
(`query_start`); request `r`'s new tokens sit at positions `kv_len[r] - n_r .. kv_len[r] - 1`.

`append_kv` writes the new tokens' keys and values to their pages. `attention` then computes, per
query token `t` at position `p` and query head `h` (KV head `h // group`):

    s_j = q . k_j * sm_scale + rel_logits[t, h, p - j]   (0 <= p - j < rel_extent, else no bias)
    over keys j <= p, and p - j < window on a sliding layer
    o   = softmax(s) . v,  lse = logsumexp(s)
    out = o * sigmoid(lse - sink[t, h]) * sigmoid(gate[t, h])

in fp32, rounded once. `sigmoid(lse - sink)` is the probability mass the keys keep against one
extra logit `sink` of zero value. A program covers `BLOCK_T` tokens of one request and the `group`
query heads of one KV head, as `BLOCK_T * GROUP_PAD` rows of one tile.

The kernels rely on the KV cache keeping every page they touch: `append_kv` writes only the step's
new tokens, and `attention` reads only keys `p - window + 1 .. p` on a sliding layer (`0 .. p` on a
global one). `KVCacheManagerV2` keeps a sliding layer's pages from `history + 1 - window` on, which
covers that range, and the model checks each step's tables for it on the host. `attention` also
masks a key whose page index is negative (a recycled page), so a broken contract cannot read
before the pool.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Masked scores. Finite, so a row whose first key block is fully masked keeps a finite running
# maximum; the first valid key then rescales what the masked keys added to zero.
_MASKED = tl.constexpr(-1.0e30)


@triton.jit
def _append_kv_kernel(
    K,
    V,
    POOL,
    PAGE_TABLE,
    QSTART,
    KV_LEN,
    TOKEN_REQ,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_pool_page,
    stride_pool_kv,
    stride_pool_head,
    stride_pool_slot,
    stride_pt,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    t = tl.program_id(0)
    head = tl.program_id(1)
    req = tl.load(TOKEN_REQ + t)
    start = tl.load(QSTART + req)
    n = tl.load(QSTART + req + 1) - start
    pos = tl.load(KV_LEN + req) - n + (t - start)
    page = tl.load(PAGE_TABLE + req.to(tl.int64) * stride_pt + pos // PAGE_SIZE).to(tl.int64)
    dst = (
        POOL
        + page * stride_pool_page
        + head * stride_pool_head
        + (pos % PAGE_SIZE) * stride_pool_slot
    )
    offs = tl.arange(0, HEAD_DIM)
    t64 = t.to(tl.int64)
    tl.store(dst + offs, tl.load(K + t64 * stride_kt + head * stride_kh + offs))
    tl.store(dst + stride_pool_kv + offs, tl.load(V + t64 * stride_vt + head * stride_vh + offs))


@triton.jit
def _attention_kernel(
    Q,
    POOL,
    PAGE_TABLE,
    QSTART,
    KV_LEN,
    TILE_REQ,
    TILE_START,
    REL,
    SINK,
    GATE,
    OUT,
    stride_qt,
    stride_qh,
    stride_pool_page,
    stride_pool_kv,
    stride_pool_head,
    stride_pool_slot,
    stride_pt,
    stride_rt,
    stride_rh,
    stride_st,
    stride_gt,
    stride_gh,
    stride_ot,
    stride_oh,
    sm_scale,
    window,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    req = tl.load(TILE_REQ + tile)
    q_begin = tl.load(TILE_START + tile)
    start = tl.load(QSTART + req)
    end = tl.load(QSTART + req + 1)
    past = tl.load(KV_LEN + req) - (end - start)

    BLOCK_M: tl.constexpr = BLOCK_T * GROUP_PAD
    rows = tl.arange(0, BLOCK_M)
    token = q_begin + rows // GROUP_PAD
    head_in_group = rows % GROUP_PAD
    head = kv_head * GROUP + head_in_group
    row_ok = (head_in_group < GROUP) & (token < end)
    q_pos = past + (token - start)
    token64 = token.to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + token64[:, None] * stride_qt + head[:, None] * stride_qh + offs_d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )

    first_pos = past + (q_begin - start)
    last_pos = past + (tl.minimum(q_begin + BLOCK_T, end) - start) - 1
    # The oldest key any row of the tile reads: the first row's window start on a sliding layer.
    lo = tl.where(window > 0, tl.maximum(first_pos - window + 1, 0), 0)
    hi = last_pos + 1

    m_i = tl.full([BLOCK_M], _MASKED, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    pt_row = PAGE_TABLE + req.to(tl.int64) * stride_pt
    kv_base = POOL + kv_head * stride_pool_head

    for block in range(lo // BLOCK_N * BLOCK_N, hi, BLOCK_N):
        k_pos = block + tl.arange(0, BLOCK_N)
        key_ok = (k_pos >= lo) & (k_pos < hi)
        page = tl.load(pt_row + k_pos // PAGE_SIZE, mask=key_ok, other=0).to(tl.int64)
        key_ok = key_ok & (page >= 0)
        slot_ptr = kv_base + page * stride_pool_page + (k_pos % PAGE_SIZE) * stride_pool_slot
        k = tl.load(slot_ptr[:, None] + offs_d[None, :], mask=key_ok[:, None], other=0.0)
        v = tl.load(
            slot_ptr[:, None] + stride_pool_kv + offs_d[None, :], mask=key_ok[:, None], other=0.0
        )

        # Scores in log2 units: (q . k * sm_scale + bias) * log2(e).
        s = tl.dot(q, tl.trans(k)) * qk_scale
        dist = q_pos[:, None] - k_pos[None, :]
        allowed = (
            row_ok[:, None] & key_ok[None, :] & (dist >= 0) & ((window <= 0) | (dist < window))
        )
        biased = allowed & (dist < REL_EXTENT)
        bias = tl.load(
            REL + token64[:, None] * stride_rt + head[:, None] * stride_rh + dist,
            mask=biased,
            other=0.0,
        )
        s = tl.where(allowed, s + bias * 1.4426950408889634, _MASKED)

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Natural-log logsumexp of the scores, then the sink and the output gate.
    lse = (m_i + tl.math.log2(l_i)) / 1.4426950408889634
    sink = tl.load(SINK + token64 * stride_st + head, mask=row_ok, other=0.0).to(tl.float32)
    keep = tl.sigmoid(lse - sink)
    gate = tl.load(
        GATE + token64[:, None] * stride_gt + head[:, None] * stride_gh + offs_d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    ).to(tl.float32)
    out = acc / l_i[:, None] * keep[:, None] * tl.sigmoid(gate)
    tl.store(
        OUT + token64[:, None] * stride_ot + head[:, None] * stride_oh + offs_d[None, :],
        out.to(OUT.dtype.element_ty),
        mask=row_ok[:, None],
    )


def append_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_pool: torch.Tensor,
    page_table: torch.Tensor,
    query_start: torch.Tensor,
    kv_len: torch.Tensor,
    token_request: torch.Tensor,
) -> None:
    """Write `k`, `v` `[T, kv heads, head dim]` to their pages of `kv_pool`."""
    num_tokens, num_kv_heads, head_dim = k.shape
    if num_tokens == 0:
        return
    _append_kv_kernel[(num_tokens, num_kv_heads)](
        k,
        v,
        kv_pool,
        page_table,
        query_start,
        kv_len,
        token_request,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        kv_pool.stride(0),
        kv_pool.stride(1),
        kv_pool.stride(2),
        kv_pool.stride(3),
        page_table.stride(0),
        PAGE_SIZE=kv_pool.shape[3],
        HEAD_DIM=head_dim,
    )


def tile_plan(
    query_lengths: list[int], block_t: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per program: its request and first token, from the host-side query lengths of one step."""
    tile_request, tile_start = [], []
    start = 0
    for r, n in enumerate(query_lengths):
        for first in range(start, start + n, block_t):
            tile_request.append(r)
            tile_start.append(first)
        start += n
    as_tensor = lambda values: torch.tensor(values, dtype=torch.int32).to(device, non_blocking=True)  # noqa: E731
    return as_tensor(tile_request), as_tensor(tile_start)


def block_tokens(max_query_length: int) -> int:
    """Query tokens per program: 2 for a decode-only step (16 MMA rows with a group of up to 8
    heads), 16 otherwise."""
    return 2 if max_query_length <= 1 else 16


def attention(
    q: torch.Tensor,
    kv_pool: torch.Tensor,
    page_table: torch.Tensor,
    query_start: torch.Tensor,
    kv_len: torch.Tensor,
    tile_request: torch.Tensor,
    tile_start: torch.Tensor,
    block_t: int,
    rel_logits: torch.Tensor,
    sink: torch.Tensor,
    gate: torch.Tensor,
    window: int,
    sm_scale: float,
) -> torch.Tensor:
    """Launch the attention kernel; see the module docstring. Returns `[T, heads, head dim]`."""
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = kv_pool.shape[2]
    group = num_heads // num_kv_heads
    out = torch.empty_like(q)
    num_tiles = tile_request.shape[0]
    if num_tokens == 0 or num_tiles == 0:
        return out
    _attention_kernel[(num_tiles, num_kv_heads)](
        q,
        kv_pool,
        page_table,
        query_start,
        kv_len,
        tile_request,
        tile_start,
        rel_logits,
        sink,
        gate,
        out,
        q.stride(0),
        q.stride(1),
        kv_pool.stride(0),
        kv_pool.stride(1),
        kv_pool.stride(2),
        kv_pool.stride(3),
        page_table.stride(0),
        rel_logits.stride(0),
        rel_logits.stride(1),
        sink.stride(0),
        gate.stride(0),
        gate.stride(1),
        out.stride(0),
        out.stride(1),
        float(sm_scale),
        int(window),
        GROUP=group,
        GROUP_PAD=triton.next_power_of_2(group),
        REL_EXTENT=rel_logits.shape[2],
        PAGE_SIZE=kv_pool.shape[3],
        HEAD_DIM=head_dim,
        BLOCK_T=block_t,
        BLOCK_N=64,
        num_warps=8 if block_t * triton.next_power_of_2(group) >= 64 else 4,
    )
    return out
