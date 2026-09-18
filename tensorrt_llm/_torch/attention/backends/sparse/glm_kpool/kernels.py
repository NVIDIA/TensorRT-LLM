# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton kernels of the glm_kpool sparse backend: pool-key refresh, pool scoring and selection
expansion over the paged indexer cache. Fixed shapes, no host synchronization, CUDA-graph safe.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_fp8_kv_rows_kernel(
    rows,
    indices,
    scale,
    output,
    local_indices,
    row_stride,
    index_stride,
    column_stride,
    NUM_ROWS: tl.constexpr,
    DIM: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    index = tl.load(indices + (token // TOPK) * index_stride + (token % TOPK) * column_stride)
    valid = (index >= 0) & (index < NUM_ROWS)
    dim = tl.arange(0, BLOCK)
    values = tl.load(
        rows + index.to(tl.int64) * row_stride + dim,
        mask=valid & (dim < DIM),
        other=0.0,
    ).to(tl.float32)
    values *= tl.load(scale)
    tl.store(output + token.to(tl.int64) * DIM + dim, values, mask=dim < DIM)
    tl.store(local_indices + token, tl.where(valid, token, -1))


def gather_fp8_kv_rows(
    rows: torch.Tensor, indices: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather selected FP8 rows as BF16 and remap sparse indices in one pass.

    ``rows`` is ``[N, 1, D]`` with contiguous features, ``indices`` is
    ``[queries, topk]`` and ``scale`` is a device FP32 scalar. Invalid indices
    remain -1; no unselected cache payload is read or dequantized.
    """
    selected = torch.empty(
        (indices.numel(), 1, rows.shape[-1]), dtype=torch.bfloat16, device=rows.device
    )
    local_indices = torch.empty(indices.shape, dtype=torch.int32, device=indices.device)
    if indices.numel():
        _gather_fp8_kv_rows_kernel[(indices.numel(),)](
            rows,
            indices,
            scale,
            selected,
            local_indices,
            rows.stride(0),
            indices.stride(0),
            indices.stride(1),
            NUM_ROWS=rows.shape[0],
            DIM=rows.shape[-1],
            TOPK=indices.shape[1],
            BLOCK=triton.next_power_of_2(rows.shape[-1]),
        )
    return selected, local_indices


_FP32_MIN = torch.finfo(torch.float32).min


@triton.jit
def _kpool_update_kernel(
    POOL,
    BT,
    POS,
    APE,
    REQ,
    slot_stride,
    row_stride,
    bt_stride,
    tpb,
    HD: tl.constexpr,
    KPOOL: tl.constexpr,
    HAS_REQ: tl.constexpr,
):
    """Recompute the pool key of the pool containing ``POS[b]``.

    Members beyond ``POS[b]`` are masked (they are not visible yet), so an
    incomplete pool carries the key of its visible members and is finalized by
    the write of its last member. Row ``b`` reads block table ``REQ[b]`` when
    ``HAS_REQ`` (several rows of one request share a table), else table ``b``. Numerics follow ``build_pools``: fp32
    softmax over ``gate + ape``, bf16 probabilities and products, fp32 sum,
    bf16 result.
    """
    b = tl.program_id(0)
    if HAS_REQ:
        tbl = tl.load(REQ + b).to(tl.int64)
    else:
        tbl = b.to(tl.int64)
    pos = tl.load(POS + b).to(tl.int64)
    start = (pos // KPOOL) * KPOOL
    m = tl.arange(0, KPOOL)
    d = tl.arange(0, HD)
    mem_pos = start + m
    valid = mem_pos <= pos
    page = mem_pos // tpb
    off = mem_pos - page * tpb
    slot = tl.load(BT + tbl * bt_stride + page, mask=valid, other=0).to(tl.int64)
    row = POOL + slot * slot_stride + off * row_stride
    k = tl.load(row[:, None] + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    g = tl.load(row[:, None] + HD + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    ape = tl.load(APE + m[:, None] * HD + d[None, :]).to(tl.float32)
    logits = tl.where(valid[:, None], g + ape, float("-inf"))
    mx = tl.max(logits, axis=0)
    e = tl.exp(logits - mx[None, :])
    probs = e / tl.sum(e, axis=0)[None, :]
    probs = probs.to(tl.bfloat16).to(tl.float32)
    prod = (probs * k).to(tl.bfloat16).to(tl.float32)
    pool_key = tl.sum(prod, axis=0)
    page0 = start // tpb
    off0 = start - page0 * tpb
    slot0 = tl.load(BT + tbl * bt_stride + page0).to(tl.int64)
    tl.store(POOL + slot0 * slot_stride + off0 * row_stride + 2 * HD + d, pool_key.to(tl.bfloat16))


def kpool_update(
    index_pool: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    ape: torch.Tensor,
    tokens_per_block: int,
    *,
    head_dim: int,
    kpool: int,
    request_ids: torch.Tensor | None = None,
) -> None:
    """Refresh the pool keys of the pools containing ``positions`` (one per row).

    ``index_pool`` is the slot-major ``[slots, tokens_per_block, 3 * head_dim]``
    view, ``block_tables`` ``[N, max_pages]`` int64, ``positions`` ``[N]``.
    With ``request_ids`` (``[N]`` int32) row ``i`` reads block table
    ``block_tables[request_ids[i]]`` instead of row ``i`` -- the packed rows of
    several context requests. In-place; no host sync, fixed shapes:
    CUDA-graph safe.
    """
    n = positions.shape[0]
    if n == 0:
        return
    _kpool_update_kernel[(n,)](
        index_pool,
        block_tables,
        positions,
        ape,
        positions if request_ids is None else request_ids,
        index_pool.stride(0),
        index_pool.stride(1),
        block_tables.stride(0),
        tokens_per_block,
        HD=head_dim,
        KPOOL=kpool,
        HAS_REQ=request_ids is not None,
        num_warps=4,
    )


@triton.jit
def _kpool_score_kernel(
    Q,
    W,
    POOL,
    BT,
    KV_LENS,
    REQ,
    OUT,
    num_rows,
    w_row_stride,
    w_head_stride,
    slot_stride,
    row_stride,
    bt_stride,
    tpb,
    num_pools_max,
    q_scale,
    w_scale,
    min_value,
    HD: tl.constexpr,
    KPOOL: tl.constexpr,
    H: tl.constexpr,
    HP: tl.constexpr,
    BP: tl.constexpr,
    ROWS: tl.constexpr,
    PRECISION: tl.constexpr,
    HAS_REQ: tl.constexpr,
):
    """Score ``BP`` pools for ``ROWS`` consecutive rows: ``sum_h w[h] * relu(q_h . key_j * s)``.

    Pools whose last member is not yet visible to a row (``j >= kv_len //
    KPOOL``) get ``min_value`` so a following top-k never picks them ahead of a
    candidate. Row ``r`` reads block table ``REQ[r]`` when ``HAS_REQ`` (packed
    context rows of several requests), else table ``r``. ``ROWS > 1`` is for
    query tokens that share a block table (one request, or one request per
    group under ``HAS_REQ``): the ``BP`` pool keys are gathered once and reused
    for every row of the program. A group straddling two requests (only at
    request boundaries) gathers per row instead.
    """
    r0 = tl.program_id(0) * ROWS
    j0 = tl.program_id(1) * BP
    j = j0 + tl.arange(0, BP)
    d = tl.arange(0, HD)
    h = tl.arange(0, HP)
    hmask = h < H
    r_last = tl.minimum(r0 + ROWS - 1, num_rows - 1)
    pos = j.to(tl.int64) * KPOOL
    page = pos // tpb
    off = pos - page * tpb
    if HAS_REQ:
        t0 = tl.load(REQ + r0).to(tl.int64)
        t_last = tl.load(REQ + r_last).to(tl.int64)
        shared = t0 == t_last
    else:
        t0 = r0.to(tl.int64)
        shared = True
    if shared:
        # The last row of the group has the largest visible length (rows of
        # one request are in position order); a block past its candidates is
        # past every row's.
        kv_last = tl.load(KV_LENS + r_last).to(tl.int64)
        if j0 >= kv_last // KPOOL:
            for i in tl.static_range(ROWS):
                r = r0 + i
                tl.store(
                    OUT + r * num_pools_max + j,
                    tl.full([BP], min_value, tl.float32),
                    mask=(j < num_pools_max) & (r < num_rows),
                )
        else:
            valid_any = j < kv_last // KPOOL
            slot = tl.load(BT + t0 * bt_stride + page, mask=valid_any, other=0).to(tl.int64)
            kptr = POOL + slot * slot_stride + off * row_stride + 2 * HD
            keys = tl.load(kptr[:, None] + d[None, :], mask=valid_any[:, None], other=0.0).to(
                tl.float32
            )
            for i in tl.static_range(ROWS):
                r = r0 + i
                in_range = r < num_rows
                kv_len = tl.load(KV_LENS + r, mask=in_range, other=0).to(tl.int64)
                valid = j < kv_len // KPOOL
                q = tl.load(
                    Q + r * H * HD + h[:, None] * HD + d[None, :],
                    mask=hmask[:, None] & in_range,
                    other=0.0,
                ).to(tl.float32)
                scores = tl.dot(q, tl.trans(keys), input_precision=PRECISION)  # [HP, BP]
                scores = tl.maximum(scores * q_scale, 0.0)
                w = tl.load(
                    W + r * w_row_stride + h * w_head_stride, mask=hmask & in_range, other=0.0
                ).to(tl.float32)
                mixed = tl.sum(scores * (w * w_scale)[:, None], axis=0)
                mixed = tl.where(valid, mixed, min_value)
                tl.store(OUT + r * num_pools_max + j, mixed, mask=(j < num_pools_max) & in_range)
    else:
        for i in tl.static_range(ROWS):
            r = r0 + i
            in_range = r < num_rows
            kv_len = tl.load(KV_LENS + r, mask=in_range, other=0).to(tl.int64)
            tbl = tl.load(REQ + r, mask=in_range, other=0).to(tl.int64)
            valid = j < kv_len // KPOOL
            slot = tl.load(BT + tbl * bt_stride + page, mask=valid, other=0).to(tl.int64)
            kptr = POOL + slot * slot_stride + off * row_stride + 2 * HD
            keys = tl.load(kptr[:, None] + d[None, :], mask=valid[:, None], other=0.0).to(
                tl.float32
            )
            q = tl.load(
                Q + r * H * HD + h[:, None] * HD + d[None, :],
                mask=hmask[:, None] & in_range,
                other=0.0,
            ).to(tl.float32)
            scores = tl.dot(q, tl.trans(keys), input_precision=PRECISION)  # [HP, BP]
            scores = tl.maximum(scores * q_scale, 0.0)
            w = tl.load(
                W + r * w_row_stride + h * w_head_stride, mask=hmask & in_range, other=0.0
            ).to(tl.float32)
            mixed = tl.sum(scores * (w * w_scale)[:, None], axis=0)
            mixed = tl.where(valid, mixed, min_value)
            tl.store(OUT + r * num_pools_max + j, mixed, mask=(j < num_pools_max) & in_range)


def kpool_score(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_pool: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    tokens_per_block: int,
    *,
    num_pools_max: int,
    head_dim: int,
    kpool: int,
    q_scale: float,
    w_scale: float,
    precision: str = "ieee",
    rows_per_program: int = 1,
    request_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool scores ``[N, num_pools_max]`` fp32.

    ``q`` is ``[N, H, head_dim]`` (bf16 or fp32, contiguous), ``weights``
    ``[N, H]`` (possibly strided), ``kv_lens[i]`` row ``i``'s visible length. Only the
    ``kv_len // kpool`` complete pools of each row are scored; the rest hold
    the fp32 minimum. ``rows_per_program > 1`` requires rows that share a
    block table to be consecutive with non-decreasing ``kv_lens``: either all
    rows (``block_tables`` broadcast with stride 0, one context request) or,
    with ``request_ids`` (``[N]`` int32, row -> block-table row), the packed
    query tokens of several context requests in position order. Fixed shapes,
    no host sync.
    """
    n, num_heads, hd = q.shape
    if hd != head_dim:
        raise ValueError(f"kpool_score: q head_dim {hd} != {head_dim}")
    out = torch.empty(n, num_pools_max, dtype=torch.float32, device=q.device)
    if n == 0:
        return out
    rows = max(1, int(rows_per_program))
    if rows > 1 and n > 1 and request_ids is None and block_tables.stride(0) != 0:
        raise ValueError(
            "kpool_score: rows_per_program > 1 needs a broadcast block table or request_ids"
        )
    bp = 64
    grid = (triton.cdiv(n, rows), triton.cdiv(num_pools_max, bp))
    _kpool_score_kernel[grid](
        q,
        weights,
        index_pool,
        block_tables,
        kv_lens,
        kv_lens if request_ids is None else request_ids,
        out,
        n,
        weights.stride(0),
        weights.stride(1),
        index_pool.stride(0),
        index_pool.stride(1),
        block_tables.stride(0),
        tokens_per_block,
        num_pools_max,
        q_scale,
        w_scale,
        _FP32_MIN,
        HD=head_dim,
        KPOOL=kpool,
        H=num_heads,
        HP=max(16, triton.next_power_of_2(num_heads)),
        BP=bp,
        ROWS=rows,
        PRECISION=precision,
        HAS_REQ=request_ids is not None,
        num_warps=4,
    )
    return out


@triton.jit
def _kpool_expand_kernel(
    SEL,
    KV_LENS,
    BT,
    REQ,
    OUT,
    sel_stride,
    bt_stride,
    tpb,
    base_row,
    rows_per_slot,
    out_width,
    SELECT_K: tl.constexpr,
    KPOOL: tl.constexpr,
    KPOOL_P2: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
    HAS_REQ: tl.constexpr,
):
    """Expand selected pools into latent-cache row ids and append the tail.

    Output row ``b`` is ``[SELECT_K * KPOOL expanded members | KPOOL - 1 tail
    positions | -1 padding]``; every invalid entry is ``-1`` (the FlashMLA
    sparse kernel's own invalid marker). Row id of cache position ``p`` is
    ``base_row + block_table[p // tpb] * rows_per_slot + p % tpb``.
    """
    b = tl.program_id(0)
    if HAS_REQ:
        tbl = tl.load(REQ + b).to(tl.int64)
    else:
        tbl = b.to(tl.int64)
    kv_len = tl.load(KV_LENS + b).to(tl.int64)
    num_cand = kv_len // KPOOL

    i = tl.arange(0, SELECT_K)
    j = tl.load(SEL + b * sel_stride + i).to(tl.int64)
    # A radix top-k pads short rows with negative ids; those are invalid too.
    valid = (j >= 0) & (j < num_cand)
    m = tl.arange(0, KPOOL_P2)
    mmask = m < KPOOL
    pos = j[:, None] * KPOOL + m[None, :]
    page = pos // tpb
    off = pos - page * tpb
    ok = valid[:, None] & mmask[None, :]
    slot = tl.load(BT + tbl * bt_stride + page, mask=ok, other=0).to(tl.int64)
    rows = base_row + slot * rows_per_slot + off
    rows = tl.where(ok, rows, -1)
    tl.store(
        OUT + b * out_width + i[:, None] * KPOOL + m[None, :],
        rows.to(tl.int32),
        mask=mmask[None, :] & (i[:, None] >= 0),
    )

    # Tail: the incomplete trailing group, always visible to its own query.
    tail_count = kv_len - num_cand * KPOOL
    tail_start = kv_len - tail_count
    t = tl.arange(0, KPOOL_P2)
    tpos = tail_start + t
    tvalid = t < tail_count
    tpage = tpos // tpb
    toff = tpos - tpage * tpb
    tslot = tl.load(BT + tbl * bt_stride + tpage, mask=tvalid, other=0).to(tl.int64)
    trows = tl.where(tvalid, base_row + tslot * rows_per_slot + toff, -1)
    tidx = SELECT_K * KPOOL + t
    tl.store(OUT + b * out_width + tidx, trows.to(tl.int32), mask=tidx < out_width)

    # -1 padding up to the kernel-aligned width.
    pad0 = SELECT_K * KPOOL + KPOOL_P2
    for start in range(pad0, out_width, PAD_BLOCK):
        p = start + tl.arange(0, PAD_BLOCK)
        tl.store(OUT + b * out_width + p, tl.full([PAD_BLOCK], -1, tl.int32), mask=p < out_width)


def kpool_expand(
    selected: torch.Tensor,
    kv_lens: torch.Tensor,
    block_tables: torch.Tensor,
    tokens_per_block: int,
    *,
    base_row: int,
    rows_per_slot: int,
    kpool: int,
    out_width: int,
    request_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Selected pools ``[N, select_k]`` -> latent row ids ``[N, out_width]`` int32.

    ``out_width`` must cover ``select_k * kpool + (kpool - 1)``; extra columns
    are ``-1`` padding (FlashMLA tiles the top-k axis in 64s). ``request_ids``
    (``[N]`` int32) maps rows to block-table rows as in :func:`kpool_score`.
    """
    n, select_k = selected.shape
    if out_width < select_k * kpool + kpool - 1:
        raise ValueError(f"kpool_expand: out_width {out_width} < {select_k * kpool + kpool - 1}")
    out = torch.empty(n, out_width, dtype=torch.int32, device=selected.device)
    if n == 0:
        return out
    _kpool_expand_kernel[(n,)](
        selected,
        kv_lens,
        block_tables,
        kv_lens if request_ids is None else request_ids,
        out,
        selected.stride(0),
        block_tables.stride(0),
        tokens_per_block,
        base_row,
        rows_per_slot,
        out_width,
        SELECT_K=select_k,
        KPOOL=kpool,
        KPOOL_P2=triton.next_power_of_2(kpool),
        PAD_BLOCK=64,
        HAS_REQ=request_ids is not None,
        num_warps=4,
    )
    return out
