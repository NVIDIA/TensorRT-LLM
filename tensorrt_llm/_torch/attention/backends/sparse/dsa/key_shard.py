# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key-sharded DSA decode indexer for tensor parallelism.

Without attention DP every TP rank holds the whole index-K cache and would score all keys of
every sequence. Here rank ``r`` of ``W`` scores only the 64-token cache blocks ``b`` with
``b % W == r``, takes its local best ``L`` keys by score (``L`` = ``key_shard_local_k``, an exact
local Top-K) and the ranks merge them into the global Top-K.

Selection order: ``(score desc, token index asc)`` on the order-preserving bit key of the fp32
score (the self-sampling GVR deterministic order: ``-0.0`` ranks below ``+0.0``). Each candidate
is one int64 ``(ordered score << 32) | (2^32 - 1 - global index)``, so a larger key is a better
candidate and keys are unique.

Merge: the local logits kernel writes this rank's scores straight into its symmetric (MNNVL/IPC)
workspace. Each rank pushes its ``L`` candidates (ascending local position by default), their
count, its local key count and its smallest sent key into every rank's receive slot, then publishes
a per-row flag on every rank. Each rank selects the ``K`` best of the ``W * L`` received candidates
and checks a certificate: if some rank had more than ``L`` keys and its smallest sent score is not
below the selection threshold's score, its unsent keys could belong to the global Top-K, so the
row is re-selected exactly from every rank's local scores (read remotely). The receive slots
alternate per call (two slots); a call-count epoch (advanced by the merge) tags the flags, so a
rank reuses a receive slot only after every peer has read it. The local scores have one slot: a
rank overwrites them in its next call, which it reaches only after the all-reduce of this call's
attention output, i.e. after every peer has finished this call's merge (the TP layers between two
indexer calls always all-reduce).
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import triton
import triton.language as tl

from .params import disagg_role, is_kimi_k31_dsa

KEY_SHARD_TOKENS_PER_BLOCK = 64
_INT64_MIN = tl.constexpr(-(2**63))
_INT64_MIN_PLUS1 = tl.constexpr(-(2**63) + 1)


# Fewest decode sequences for which a step takes the key-sharded path. The pack / merge add a near-constant ~20 us
# per layer, so the sharded path only wins once the batch holds enough index keys. The decision depends only on the
# batch size, so every rank takes the same path and each CUDA graph captures one of them.
KEY_SHARD_MIN_SEQS = 11
# Helix key-sharded merge: fewest decode sequences, decode rows of the merge workspace and keys each rank sends.
HELIX_KEY_SHARD_MIN_SEQS = 8
HELIX_KEY_SHARD_MAX_ROWS = 128
HELIX_KEY_SHARD_LOCAL_K = 512


def key_shard_enabled() -> bool:
    """Whether the key-sharded decode indexer may run in this process: Kimi K3.1 on a disaggregated generation
    server (the static layout gates are in ``DSAtrtllmAttentionMetadata._key_shard_eligible``)."""
    return is_kimi_k31_dsa() and disagg_role() == "generation"


def key_shard_local_k(top_k: int, world: int) -> int:
    """Keys each rank takes from its shard: about 4x the fair share ``top_k / world``, within [256, 512].

    Every rank pushes all of them, so this bounds the merge's work (world x local_k keys per row) and the exchanged
    bytes. Rows where a rank holds more than this many of the global Top-K are re-selected exactly from the local
    scores, so any value gives the same rows.
    """
    return min(512, max(256, 4 * top_k // world))


def key_shard_owner_rows() -> bool:
    """Whether Helix (owned-local) merges reduce each row on one owner rank (TRTLLM_DSA_HELIX_KEY_SHARD_OWNER_MERGE=1,
    off).

    Row ``r`` belongs to rank ``r % W``: every rank pushes its candidates of the row to the owner only (1/W of the
    all-to-all traffic), the owner selects the row's threshold key and publishes it, and every rank keeps its own
    candidates at or above it. The kept rows and counts are the same as ``merge_owned`` without it."""
    return os.environ.get("TRTLLM_DSA_HELIX_KEY_SHARD_OWNER_MERGE", "0") == "1"


def key_shard_local_count(
    n: torch.Tensor, rank: int, world: int, tokens_per_block: int = KEY_SHARD_TOKENS_PER_BLOCK
) -> torch.Tensor:
    """Positions below ``n`` owned by ``rank`` when cache block ``b`` belongs to rank ``b % world``."""
    full_blocks = n // tokens_per_block
    partial = n % tokens_per_block
    owned_full = torch.clamp((full_blocks - rank + world - 1) // world, min=0)
    return owned_full * tokens_per_block + torch.where(
        full_blocks % world == rank, partial, torch.zeros_like(partial)
    )


@triton.jit
def _ks_local_count(n, rank, W: tl.constexpr, TPB: tl.constexpr):
    """``key_shard_local_count`` for non-negative ``n``."""
    full = n // TPB
    owned = tl.maximum((full - rank + W - 1) // W, 0)
    return owned * TPB + tl.where(full % W == rank, n % TPB, 0)


@triton.jit
def _ks_rows_kernel(
    kv_lens_ptr,
    src_ptr,
    src_stride,
    dst_ptr,
    dst_stride,
    ctx_ptr,
    rows_ptr,
    rank,
    next_n,
    W: tl.constexpr,
    TPB: tl.constexpr,
    NN: tl.constexpr,
    BLK: tl.constexpr,
):
    """Per sequence: the local context length, the ``next_n`` local row lengths and the compacted block table
    (columns of this rank's blocks, clamped to >= 0) up to the local context."""
    seq = tl.program_id(0).to(tl.int64)
    n = tl.load(kv_lens_ptr + seq).to(tl.int32)
    local = _ks_local_count(n, rank, W, TPB)
    if tl.program_id(1) == 0:
        tl.store(ctx_ptr + seq, local)
        j = tl.arange(0, NN)
        tl.store(
            rows_ptr + seq * next_n + j,
            _ks_local_count(n - next_n + 1 + j, rank, W, TPB),
            mask=j < next_n,
        )
    lb = tl.program_id(1) * BLK + tl.arange(0, BLK)
    m = lb < (local + TPB - 1) // TPB
    v = tl.load(src_ptr + seq * src_stride + lb * W + rank, mask=m)
    tl.store(dst_ptr + seq * dst_stride + lb, tl.maximum(v, 0), mask=m)


def key_shard_prepare_rows(
    kv_lens: torch.Tensor,
    block_table: torch.Tensor,
    local_block_table: torch.Tensor,
    ctx_lens: torch.Tensor,
    row_lens: torch.Tensor,
    rank: int,
    world: int,
    next_n: int,
) -> None:
    """One launch for the per-step key-shard views of ``kv_lens.shape[0]`` decode sequences (graph-capturable).

    Writes ``ctx_lens[s]`` (this rank's keys of sequence ``s``), ``row_lens[s * next_n + j]`` (its keys visible to
    verify row ``j``) and ``local_block_table[s, :c]`` = ``max(block_table[s, rank::world][:c], 0)`` where ``c``
    covers ``ctx_lens[s]``; columns past ``c`` are left as they are (no kernel reads them).
    """
    num_seqs = kv_lens.shape[0]
    if num_seqs == 0:
        return
    assert block_table.stride(1) == 1 and local_block_table.stride(1) == 1
    blk = 256
    _ks_rows_kernel[(num_seqs, triton.cdiv(local_block_table.shape[1], blk))](
        kv_lens,
        block_table,
        block_table.stride(0),
        local_block_table,
        local_block_table.stride(0),
        ctx_lens,
        row_lens,
        rank,
        next_n,
        W=world,
        TPB=KEY_SHARD_TOKENS_PER_BLOCK,
        NN=triton.next_power_of_2(max(next_n, 1)),
        BLK=blk,
        num_warps=1,
    )


@triton.jit
def _order_key(score, gidx):
    bits = score.to(tl.int32, bitcast=True)
    ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    low = 4294967295 - gidx.to(tl.int64)
    return (ordered.to(tl.int64) << 32) | low


@triton.jit
def _kth_largest(keys, valid, target):
    """Signed threshold ``t``: the ``target`` largest valid keys are exactly the valid keys ``>= t``.

    Keys are unique and at least ``target`` are valid. Most-significant-byte-first radix select on the unsigned view
    ``key ^ INT64_MIN``, one 256-bin histogram per byte; stops as soon as the boundary byte group is taken whole
    (the low bytes - the token index - only matter for score ties at the boundary).
    """
    sign = tl.full([], -9223372036854775808, tl.int64)
    u = keys ^ sign
    bins = tl.arange(0, 512)
    real = bins < 256
    prefix = tl.full([], 0, tl.int64)
    remaining = tl.zeros([], tl.int32) + target
    active = valid
    done = tl.zeros([], tl.int32)
    for p in tl.static_range(8):
        if done == 0:
            d = ((u >> (56 - 8 * p)) & 0xFF).to(tl.int32)
            hist = tl.histogram(tl.where(active, d, 257), 512)
            hist = tl.where(real, hist, 0)
            suffix = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist
            bstar = tl.max(tl.where(real & (suffix >= remaining), bins, -1), axis=0)
            above = tl.sum(tl.where(bins > bstar, hist, 0), axis=0)
            equal = tl.sum(tl.where(bins == bstar, hist, 0), axis=0)
            remaining = remaining - above
            prefix = prefix | (bstar.to(tl.int64) << (56 - 8 * p))
            active = active & (d == bstar)
            done = tl.where(equal == remaining, 1, done)
    return prefix ^ sign


@triton.jit
def _ks_pack_row(
    row,
    e,
    logits_ptr,
    logits_stride,
    topk_ptr,
    rowlen_ptr,
    keys_ptr,
    i32_ptr,
    dk_ptr,
    d32_ptr,
    rank,
    rows_max,
    W: tl.constexpr,
    L: tl.constexpr,
    TPB: tl.constexpr,
    SORT: tl.constexpr,
    OWNER: tl.constexpr,
):
    """Push this rank's ``L`` candidate keys of one row and the row header into every rank's receive slot and
    publish one flag per rank (the row's local scores already sit in this rank's workspace).

    ``OWNER``: push the keys, the header and the flag to the row's owner rank (``row % W``) only, and the keys also
    into this rank's own receive cell, where its select step reads them back."""
    slot = (e & 1).to(tl.int64)
    n = tl.load(rowlen_ptr + row)
    src = logits_ptr + row * logits_stride
    j = tl.arange(0, L)
    idx = tl.load(topk_ptr + row * L + j)
    if SORT:
        # Ascending local positions (-1 padding last): the merge then emits every row rank-major and ascending
        # within each rank, an order that depends only on the selected set (the verify rows of a sequence share
        # most of their keys, so the sparse attention gathers the same cache rows in the same order).
        idx = tl.sort(tl.where(idx >= 0, idx, 2147483647))
        idx = tl.where(idx == 2147483647, -1, idx)
    valid = idx >= 0
    safe = tl.where(valid, idx, 0)
    score = tl.load(src + safe.to(tl.int64), mask=valid, other=0.0)
    gidx = ((safe // TPB) * W + rank) * TPB + (safe % TPB)
    key = tl.where(valid, _order_key(score, gidx), _INT64_MIN)
    sent = tl.sum(valid.to(tl.int32), axis=0)
    # Smallest sent key: a rank whose every sent key is selected may hold more of the global Top-K.
    boundary = tl.min(tl.where(valid, key, 9223372036854775807), axis=0)
    rw = rows_max * W
    cell = slot * rw + row * W + rank
    recv = keys_ptr + slot * rw * L + (row * W + rank) * L
    if OWNER:
        owner = row % W
        dko = tl.load(dk_ptr + owner)
        d32o = tl.load(d32_ptr + owner)
        tl.store(recv + dko + j, key)
        if owner != rank:
            tl.store(recv + j, key)
        tl.store(keys_ptr + 2 * rw * L + cell + dko, boundary)
        tl.store(i32_ptr + cell + d32o, sent)
        tl.store(i32_ptr + 2 * rw + cell + d32o, n)
        tl.debug_barrier()
        tl.atomic_xchg(i32_ptr + 4 * rw + cell + d32o, e, sem="release", scope="sys")
    else:
        for p in tl.static_range(W):
            tl.store(recv + tl.load(dk_ptr + p) + j, key)
        wv = tl.arange(0, W)
        dk = tl.load(dk_ptr + wv)
        d32 = tl.load(d32_ptr + wv)
        tl.store(keys_ptr + 2 * rw * L + cell + dk, tl.zeros([W], tl.int64) + boundary)
        tl.store(i32_ptr + cell + d32, tl.zeros([W], tl.int32) + sent)
        tl.store(i32_ptr + 2 * rw + cell + d32, tl.zeros([W], tl.int32) + n)
        tl.debug_barrier()
        tl.atomic_xchg(
            i32_ptr + 4 * rw + cell + d32, tl.zeros([W], tl.int32) + e, sem="release", scope="sys"
        )


@triton.jit
def _ks_scan_local(
    f32_ptr,
    d32_ptr,
    base,
    n,
    w,
    prefix,
    p: tl.constexpr,
    W: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
):
    """Byte-``p`` histogram (256 bins) of rank ``w``'s keys of one row, restricted to the keys matching ``prefix`` on
    the bytes above ``p``, read from that rank's local scores."""
    sign = tl.full([], -9223372036854775808, tl.int64)
    bins = tl.arange(0, 256)
    hist = tl.zeros([256], tl.int32)
    row_scores = f32_ptr + tl.load(d32_ptr + w) + base
    for c0 in range(0, n, CH):
        cc = c0 + tl.arange(0, CH)
        m = cc < n
        s = tl.load(row_scores + cc, mask=m, other=0.0, cache_modifier=".cv")
        g = ((cc // TPB) * W + w) * TPB + (cc % TPB)
        u = _order_key(s, g) ^ sign
        act = m
        if p > 0:
            act = act & ((u >> (64 - 8 * p)) == (prefix >> (64 - 8 * p)))
        h = tl.histogram(tl.where(act, ((u >> (56 - 8 * p)) & 0xFF).to(tl.int32), 0), 256)
        hist += tl.where(bins == 0, h - tl.sum((act == 0).to(tl.int32), axis=0), h)
    return hist


@triton.jit
def _ks_wait_flags(i32_ptr, state_ptr, cell, rw, e, max_spin, W: tl.constexpr):
    """Spin until the ``W`` flags of a row's receive cells carry epoch ``e`` (counts a timeout in ``state[3]``)."""
    zero = tl.zeros([W], tl.int32)
    v = tl.atomic_add(i32_ptr + 4 * rw + cell, zero, sem="acquire", scope="sys")
    n = 0
    while (tl.sum((v != e).to(tl.int32), axis=0) > 0) & (n < max_spin):
        v = tl.atomic_add(i32_ptr + 4 * rw + cell, zero, sem="acquire", scope="sys")
        n += 1
    if n >= max_spin:
        tl.atomic_add(state_ptr + 3, 1)
    tl.debug_barrier()


@triton.jit
def _ks_exact_threshold(
    i32_ptr,
    f32_ptr,
    d32_ptr,
    row,
    slot,
    rw,
    LW,
    total_all,
    W: tl.constexpr,
    K: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
):
    """Threshold key of a row's global Top-``K`` re-selected from every rank's local scores (byte-wise radix select
    streaming them); the counts come from this rank's receive cells."""
    base = row * LW
    sign_f = tl.full([], -9223372036854775808, tl.int64)
    bins_f = tl.arange(0, 256)
    prefix_f = tl.full([], 0, tl.int64)
    remaining_f = tl.zeros([], tl.int32) + tl.minimum(total_all, K)
    done_f = tl.zeros([], tl.int32)
    for p in tl.static_range(8):
        if done_f == 0:
            hist_f = tl.zeros([256], tl.int32)
            for w in tl.static_range(W):
                n_w = tl.load(i32_ptr + 2 * rw + slot * rw + row * W + w, volatile=True)
                hist_f += _ks_scan_local(f32_ptr, d32_ptr, base, n_w, w, prefix_f, p, W, TPB, CH)
            suffix_f = tl.sum(hist_f, axis=0) - tl.cumsum(hist_f, axis=0) + hist_f
            bstar_f = tl.max(tl.where(suffix_f >= remaining_f, bins_f, -1), axis=0)
            above_f = tl.sum(tl.where(bins_f > bstar_f, hist_f, 0), axis=0)
            equal_f = tl.sum(tl.where(bins_f == bstar_f, hist_f, 0), axis=0)
            remaining_f = remaining_f - above_f
            prefix_f = prefix_f | (bstar_f.to(tl.int64) << (56 - 8 * p))
            done_f = tl.where(equal_f == remaining_f, 1, done_f)
    thr_all = prefix_f ^ sign_f
    if total_all < K:
        thr_all = tl.full([], _INT64_MIN_PLUS1, tl.int64)
    return thr_all


@triton.jit
def _ks_select_own_scores(
    f32_ptr, base, n_w, w, thr, orow, W: tl.constexpr, TPB: tl.constexpr, CH: tl.constexpr
):
    """Write rank ``w``'s local positions whose key is ``>= thr``, ascending, to ``orow``; returns their count."""
    off_f = tl.zeros([], tl.int32)
    row_scores = f32_ptr + base
    for c0 in range(0, n_w, CH):
        cc = c0 + tl.arange(0, CH)
        m = cc < n_w
        s_f = tl.load(row_scores + cc, mask=m, other=0.0, cache_modifier=".cv")
        g_f = ((cc // TPB) * W + w) * TPB + (cc % TPB)
        sel_f = m & (_order_key(s_f, g_f) >= thr)
        pos_f = off_f + tl.cumsum(sel_f.to(tl.int32), axis=0) - 1
        tl.store(orow + pos_f, cc.to(tl.int32), mask=sel_f)
        off_f += tl.sum(sel_f.to(tl.int32), axis=0)
    return off_f


@triton.jit
def _ks_merge_row(
    row,
    e,
    keys_ptr,
    i32_ptr,
    f32_ptr,
    d32_ptr,
    out_ptr,
    out_stride,
    state_ptr,
    rows_max,
    LW,
    max_spin,
    rank,
    count_ptr,
    W: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
    OWNED: tl.constexpr,
):
    """Global Top-K of one row from the ``W`` pushed candidate lists in this rank's receive slot.

    ``OWNED``: write only this rank's winners, as local token indices, kept entries first in output order and the
    rest ``-1``, plus their count to ``count_ptr[row]`` (Helix CP, where a rank's sparse attention reads its own
    pages only). Rank ``w``'s candidates are its own keys, so its winners are the selected entries of its segment.
    """
    slot = (e & 1).to(tl.int64)
    wv = tl.arange(0, W)
    rw = rows_max * W
    cell = slot * rw + row * W + wv
    _ks_wait_flags(i32_ptr, state_ptr, cell, rw, e, max_spin, W)
    sent = tl.load(i32_ptr + cell, volatile=True)
    cnt = tl.load(i32_ptr + 2 * rw + cell, volatile=True)
    boundary = tl.load(keys_ptr + 2 * rw * L + cell, volatile=True)
    c = tl.arange(0, W * L)
    flat = tl.load(keys_ptr + slot * rw * L + row * W * L + c, cache_modifier=".cv")
    valid = flat > _INT64_MIN
    total = tl.sum(sent, axis=0)
    total_all = tl.sum(cnt, axis=0)
    if total >= K:
        thr = _kth_largest(flat, valid, K)
    else:
        thr = tl.full([], _INT64_MIN_PLUS1, tl.int64)
    # Scores only (not the index bits): exact for any local Top-L whose score multiset is right, i.e. whatever the
    # tie order at a rank's boundary score.
    fail = tl.sum(((cnt > sent) & ((boundary >> 32) >= (thr >> 32))).to(tl.int32), axis=0) > 0
    kk = tl.arange(0, K)
    orow = out_ptr + row * out_stride
    if fail:
        # Re-select exactly from every rank's local scores (rare): byte-wise radix select streaming them.
        tl.atomic_add(state_ptr + 2, 1)
        base = row * LW
        thr_all = _ks_exact_threshold(
            i32_ptr, f32_ptr, d32_ptr, row, slot, rw, LW, total_all, W, K, TPB, CH
        )
        off_f = tl.zeros([], tl.int32)
        for w in tl.static_range(W):
            run_w = True
            if OWNED:
                run_w = w == rank
            if run_w:
                n_w = tl.load(i32_ptr + 2 * rw + slot * rw + row * W + w, volatile=True)
                row_scores = f32_ptr + tl.load(d32_ptr + w) + base
                for c0 in range(0, n_w, CH):
                    cc = c0 + tl.arange(0, CH)
                    m = cc < n_w
                    s_f = tl.load(row_scores + cc, mask=m, other=0.0, cache_modifier=".cv")
                    g_f = ((cc // TPB) * W + w) * TPB + (cc % TPB)
                    sel_f = m & (_order_key(s_f, g_f) >= thr_all)
                    pos_f = off_f + tl.cumsum(sel_f.to(tl.int32), axis=0) - 1
                    if OWNED:
                        tl.store(orow + pos_f, cc.to(tl.int32), mask=sel_f)
                    else:
                        tl.store(orow + pos_f, g_f.to(tl.int32), mask=sel_f)
                    off_f += tl.sum(sel_f.to(tl.int32), axis=0)
        tl.store(orow + kk, tl.full([K], -1, tl.int32), mask=kk >= off_f)
        if OWNED:
            tl.store(count_ptr + row, off_f)
    else:
        sel = valid & (flat >= thr)
        g = 4294967295 - (flat & 4294967295)
        if OWNED:
            sel = sel & ((c // L) == rank)
            g = (g // TPB) // W * TPB + g % TPB
        pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(orow + pos, g.to(tl.int32), mask=sel)
        nsel = tl.sum(sel.to(tl.int32), axis=0)
        tl.store(orow + kk, tl.full([K], -1, tl.int32), mask=kk >= nsel)
        if OWNED:
            tl.store(count_ptr + row, nsel)


@triton.jit
def _ks_advance_epoch(state_ptr, e, num_rows):
    """The last row program of a merge advances the epoch (every program read it at its start)."""
    ticket = tl.atomic_add(state_ptr + 1, 1, sem="acq_rel", scope="gpu")
    if ticket == num_rows - 1:
        tl.store(state_ptr + 1, 0)
        tl.atomic_xchg(state_ptr, e, sem="release", scope="gpu")


@triton.jit
def _ks_pack_kernel(
    logits_ptr,
    logits_stride,
    topk_ptr,
    rowlen_ptr,
    keys_ptr,
    i32_ptr,
    dk_ptr,
    d32_ptr,
    state_ptr,
    rank,
    rows_max,
    W: tl.constexpr,
    L: tl.constexpr,
    TPB: tl.constexpr,
    SORT: tl.constexpr,
    OWNER: tl.constexpr,
    PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    e = tl.load(state_ptr) + 1
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    _ks_pack_row(
        row,
        e,
        logits_ptr,
        logits_stride,
        topk_ptr,
        rowlen_ptr,
        keys_ptr,
        i32_ptr,
        dk_ptr,
        d32_ptr,
        rank,
        rows_max,
        W,
        L,
        TPB,
        SORT,
        OWNER,
    )


@triton.jit
def _ks_merge_kernel(
    keys_ptr,
    i32_ptr,
    f32_ptr,
    d32_ptr,
    out_ptr,
    out_stride,
    state_ptr,
    num_rows,
    rows_max,
    LW,
    max_spin,
    rank,
    count_ptr,
    W: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
    OWNED: tl.constexpr,
    PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    e = tl.load(state_ptr) + 1
    _ks_merge_row(
        row,
        e,
        keys_ptr,
        i32_ptr,
        f32_ptr,
        d32_ptr,
        out_ptr,
        out_stride,
        state_ptr,
        rows_max,
        LW,
        max_spin,
        rank,
        count_ptr,
        W,
        K,
        L,
        TPB,
        CH,
        OWNED,
    )
    if PDL:
        tl.extra.cuda.gdc_wait()
    _ks_advance_epoch(state_ptr, e, num_rows)


@triton.jit
def _ks_owner_kernel(
    keys_ptr,
    i32_ptr,
    f32_ptr,
    thr_ptr,
    tf_ptr,
    dk_ptr,
    d32_ptr,
    state_ptr,
    num_rows,
    rows_max,
    LW,
    max_spin,
    rank,
    W: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
    PDL: tl.constexpr,
):
    """Owner-row merge, owner step: program ``q`` reduces row ``q * W + rank`` (owned by this rank) from the ``W``
    candidate lists in its receive slot and publishes the row's threshold key (and whether it is the exact
    re-selection's) to every rank. Same selection as ``_ks_merge_row``."""
    e = tl.load(state_ptr) + 1
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64) * W + rank
    if row < num_rows:
        slot = (e & 1).to(tl.int64)
        wv = tl.arange(0, W)
        rw = rows_max * W
        cell = slot * rw + row * W + wv
        _ks_wait_flags(i32_ptr, state_ptr, cell, rw, e, max_spin, W)
        sent = tl.load(i32_ptr + cell, volatile=True)
        cnt = tl.load(i32_ptr + 2 * rw + cell, volatile=True)
        boundary = tl.load(keys_ptr + 2 * rw * L + cell, volatile=True)
        c = tl.arange(0, W * L)
        flat = tl.load(keys_ptr + slot * rw * L + row * W * L + c, cache_modifier=".cv")
        valid = flat > _INT64_MIN
        total = tl.sum(sent, axis=0)
        total_all = tl.sum(cnt, axis=0)
        if total >= K:
            thr = _kth_largest(flat, valid, K)
        else:
            thr = tl.full([], _INT64_MIN_PLUS1, tl.int64)
        fail = tl.sum(((cnt > sent) & ((boundary >> 32) >= (thr >> 32))).to(tl.int32), axis=0) > 0
        if fail:
            tl.atomic_add(state_ptr + 2, 1)
            thr = _ks_exact_threshold(
                i32_ptr, f32_ptr, d32_ptr, row, slot, rw, LW, total_all, W, K, TPB, CH
            )
        t = slot * rows_max + row
        dk = tl.load(dk_ptr + wv)
        d32 = tl.load(d32_ptr + wv)
        tl.store(thr_ptr + t + dk, tl.zeros([W], tl.int64) + thr)
        tl.store(tf_ptr + 2 * rows_max + t + d32, tl.zeros([W], tl.int32) + fail.to(tl.int32))
        tl.debug_barrier()
        tl.atomic_xchg(tf_ptr + t + d32, tl.zeros([W], tl.int32) + e, sem="release", scope="sys")
    if PDL:
        tl.extra.cuda.gdc_wait()


@triton.jit
def _ks_select_kernel(
    keys_ptr,
    f32_ptr,
    thr_ptr,
    tf_ptr,
    rowlen_ptr,
    out_ptr,
    out_stride,
    count_ptr,
    state_ptr,
    num_rows,
    rows_max,
    LW,
    max_spin,
    rank,
    W: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
    PDL: tl.constexpr,
):
    """Owner-row merge, select step: this rank's winners of row ``program_id(0)`` (its candidates, or on the exact
    path its local scores, at or above the owner's threshold) as local token indices, kept first and the rest
    ``-1``, and their count; the last program advances the epoch."""
    row = tl.program_id(0).to(tl.int64)
    e = tl.load(state_ptr) + 1
    slot = (e & 1).to(tl.int64)
    t = slot * rows_max + row
    v = tl.atomic_add(tf_ptr + t, 0, sem="acquire", scope="sys")
    n = 0
    while (v != e) & (n < max_spin):
        v = tl.atomic_add(tf_ptr + t, 0, sem="acquire", scope="sys")
        n += 1
    if n >= max_spin:
        tl.atomic_add(state_ptr + 3, 1)
    tl.debug_barrier()
    thr = tl.load(thr_ptr + t, volatile=True)
    fail = tl.load(tf_ptr + 2 * rows_max + t, volatile=True)
    kk = tl.arange(0, K)
    orow = out_ptr + row * out_stride
    if fail != 0:
        n_w = tl.load(rowlen_ptr + row)
        nsel = _ks_select_own_scores(f32_ptr, row * LW, n_w, rank, thr, orow, W, TPB, CH)
    else:
        rw = rows_max * W
        j = tl.arange(0, L)
        key = tl.load(keys_ptr + slot * rw * L + (row * W + rank) * L + j, cache_modifier=".cv")
        sel = (key > _INT64_MIN) & (key >= thr)
        g = 4294967295 - (key & 4294967295)
        g = (g // TPB) // W * TPB + g % TPB
        pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(orow + pos, g.to(tl.int32), mask=sel)
        nsel = tl.sum(sel.to(tl.int32), axis=0)
    tl.store(orow + kk, tl.full([K], -1, tl.int32), mask=kk >= nsel)
    tl.store(count_ptr + row, nsel)
    if PDL:
        tl.extra.cuda.gdc_wait()
    _ks_advance_epoch(state_ptr, e, num_rows)


class KeyShardTopKMerge:
    """Workspace and launches of the pack / merge steps of the key-sharded decode Top-K.

    ``buffers`` are this rank's view of every rank's symmetric buffer (rank order). They come from an MNNVL / IPC
    allocation in a TP group, or from plain local tensors to emulate ``world`` ranks on one GPU (tests).

    Buffer layout (identical on every rank; the receive side has two slots alternating by epoch parity):
      int64 recv[2][rows][W][L]        candidate keys pushed by each rank (INT64_MIN padded)
      int64 boundary[2][rows][W]       smallest key each rank sent
      int32 sent[2][rows][W], count[2][rows][W] (local keys of the row), flag[2][rows][W]
      fp32  scores[rows][width]        this rank's local logits (``local_logits``; read by peers only on fallback)
      int64 thr[2][rows], int32 tflag[2][rows], tfail[2][rows]   owner-row mode: the owner's threshold per row

    ``owner_rows`` (Helix, ``merge_owned`` only): row ``r`` is reduced on rank ``r % W`` alone; see
    ``key_shard_owner_rows``.
    """

    def __init__(
        self,
        buffers: List[torch.Tensor],
        rank: int,
        rows_max: int,
        top_k: int,
        local_width: int,
        local_k: Optional[int] = None,
        max_spin: int = 1 << 26,
        owner_rows: bool = False,
        pdl: bool = True,
    ):
        self.world = len(buffers)
        assert self.world & (self.world - 1) == 0, "the merge needs a power-of-two group size"
        self.rank = rank
        self.rows_max = rows_max
        self.top_k = top_k
        self.local_k = local_k or key_shard_local_k(top_k, self.world)
        assert self.local_k <= top_k
        # The logits kernel also writes its 256-column alignment padding.
        self.local_width = -(-local_width // 256) * 256
        self.max_spin = max_spin
        self.warps = 8
        # Each rank pushes its local candidates in ascending position order: the merged rows are then rank-major
        # and ascending within each rank, so a row's order depends only on its selected set (otherwise it follows
        # the local Top-K's output order).
        self.sorted_rows = True
        # ``pdl``: the pack and merge launch with programmatic dependent launch. The pack waits for its predecessor
        # (the local Top-K) after its first loads and lets the merge launch at once; the merge needs no wait before
        # its flags (they carry the pushes), only before it ends, so the next kernel never overlaps the pack.
        self.pdl = pdl
        self.owner_rows = owner_rows
        need = self.bytes_needed(rows_max, top_k, self.world, self.local_k, self.local_width)
        mine = buffers[rank]
        assert mine.numel() * mine.element_size() >= need
        # The kernels address the peers' buffers through pointer deltas: keep them alive.
        self.buffers = list(buffers)
        base = mine.data_ptr()
        deltas = [b.data_ptr() - base for b in buffers]
        assert all(d % 8 == 0 for d in deltas) and base % 16 == 0
        dev = mine.device
        byte_off = mine.storage_offset() * mine.element_size()
        n_keys = 2 * rows_max * self.world * (self.local_k + 1)
        n_i32 = 6 * rows_max * self.world
        storage = mine.untyped_storage()
        self.keys = torch.empty(0, dtype=torch.int64, device=dev).set_(
            storage, byte_off // 8, (n_keys,)
        )
        self.i32 = torch.empty(0, dtype=torch.int32, device=dev).set_(
            storage, (byte_off + 8 * n_keys) // 4, (n_i32,)
        )
        f32_off = byte_off + 8 * n_keys + 4 * n_i32
        f32_off = -(-f32_off // 16) * 16
        self.scores = torch.empty(0, dtype=torch.float32, device=dev).set_(
            storage, f32_off // 4, (rows_max * self.local_width,)
        )
        # Output of the local logits kernel: [rows_max, local_width] fp32 in the symmetric workspace.
        self.local_logits = self.scores.view(rows_max, self.local_width)
        thr_off = f32_off + 4 * rows_max * self.local_width
        thr_off = -(-thr_off // 16) * 16
        self.thr = torch.empty(0, dtype=torch.int64, device=dev).set_(
            storage, thr_off // 8, (2 * rows_max,)
        )
        self.tflag = torch.empty(0, dtype=torch.int32, device=dev).set_(
            storage, (thr_off + 16 * rows_max) // 4, (4 * rows_max,)
        )
        self._row_lens: Optional[torch.Tensor] = None
        self.dk = torch.tensor([d // 8 for d in deltas], dtype=torch.int64, device=dev)
        self.d32 = torch.tensor([d // 4 for d in deltas], dtype=torch.int64, device=dev)
        # [epoch, ticket, fallback rows, spin timeouts]
        self.state = torch.zeros(4, dtype=torch.int32, device=dev)

    @staticmethod
    def bytes_needed(rows_max: int, top_k: int, world: int, local_k: int, local_width: int) -> int:
        local_width = -(-local_width // 256) * 256
        keys_i32 = 8 * 2 * rows_max * world * (local_k + 1) + 4 * 6 * rows_max * world
        scores_end = -(-keys_i32 // 16) * 16 + 4 * rows_max * local_width
        return -(-scores_end // 16) * 16 + 8 * 2 * rows_max + 4 * 4 * rows_max

    def pack(self, local_topk: torch.Tensor, row_lens: torch.Tensor) -> None:
        """Publish this rank's local Top-``local_k`` (local column indices, -1 padded) of the ``local_logits`` rows
        whose valid local lengths are ``row_lens``."""
        rows = local_topk.shape[0]
        if rows == 0:
            return
        assert (
            rows <= self.rows_max
            and local_topk.shape[1] == self.local_k
            and local_topk.is_contiguous()
            and row_lens.dtype == torch.int32
        )
        _ks_pack_kernel[(rows,)](
            self.local_logits,
            self.local_width,
            local_topk,
            row_lens,
            self.keys,
            self.i32,
            self.dk,
            self.d32,
            self.state,
            self.rank,
            self.rows_max,
            W=self.world,
            L=self.local_k,
            TPB=KEY_SHARD_TOKENS_PER_BLOCK,
            SORT=self.sorted_rows,
            OWNER=self.owner_rows,
            PDL=self.pdl,
            num_warps=self.warps,
            launch_pdl=self.pdl,
        )
        self._row_lens = row_lens

    def merge(self, out: torch.Tensor) -> None:
        """Write the global Top-K (token indices, -1 padded) of every row into ``out`` [rows, top_k]."""
        assert not self.owner_rows, "an owner-row workspace serves merge_owned only"
        self._merge(out, None)

    def merge_owned(self, out: torch.Tensor, counts: Optional[torch.Tensor] = None) -> None:
        """Write this rank's share of every row's global Top-K into ``out`` [rows, top_k] as local token indices.

        The kept entries come first, in the order ``merge`` would list them, the rest is ``-1``: the same rows as
        ``merge`` followed by ``helix_key_shard.keep_owned_local_rows``. ``counts`` [rows] int32 (optional) receives
        the kept count of each row.
        """
        counts = counts if counts is not None else self._counts_scratch(out.shape[0])
        if self.owner_rows:
            self.merge_owned_begin(out.shape[0])
            self.merge_owned_end(out, counts)
            return
        self._merge(out, counts)

    def merge_owned_begin(self, rows: int) -> None:
        """Owner-row mode, first step: reduce this rank's rows (``r % W == rank``) and publish their thresholds."""
        assert self.owner_rows
        if rows == 0:
            return
        _ks_owner_kernel[(triton.cdiv(rows, self.world),)](
            self.keys,
            self.i32,
            self.scores,
            self.thr,
            self.tflag,
            self.dk,
            self.d32,
            self.state,
            rows,
            self.rows_max,
            self.local_width,
            self.max_spin,
            self.rank,
            W=self.world,
            K=self.top_k,
            L=self.local_k,
            TPB=KEY_SHARD_TOKENS_PER_BLOCK,
            CH=2048,
            PDL=self.pdl,
            num_warps=self.warps,
            launch_pdl=self.pdl,
        )

    def merge_owned_end(self, out: torch.Tensor, counts: torch.Tensor) -> None:
        """Owner-row mode, second step: this rank's winners of every row from the owners' thresholds (the
        ``merge_owned`` outputs)."""
        assert self.owner_rows and self._row_lens is not None
        rows = out.shape[0]
        if rows == 0:
            return
        assert out.stride(1) == 1 and out.shape[1] == self.top_k
        assert counts.dtype == torch.int32 and counts.is_contiguous() and counts.shape[0] >= rows
        _ks_select_kernel[(rows,)](
            self.keys,
            self.scores,
            self.thr,
            self.tflag,
            self._row_lens,
            out,
            out.stride(0),
            counts,
            self.state,
            rows,
            self.rows_max,
            self.local_width,
            self.max_spin,
            self.rank,
            W=self.world,
            K=self.top_k,
            L=self.local_k,
            TPB=KEY_SHARD_TOKENS_PER_BLOCK,
            CH=2048,
            PDL=self.pdl,
            num_warps=self.warps,
            launch_pdl=self.pdl,
        )

    def _counts_scratch(self, rows: int) -> torch.Tensor:
        scratch = getattr(self, "_owned_counts", None)
        if scratch is None or scratch.shape[0] < rows:
            scratch = torch.empty(self.rows_max, dtype=torch.int32, device=self.state.device)
            self._owned_counts = scratch
        return scratch

    def _merge(self, out: torch.Tensor, counts: Optional[torch.Tensor]) -> None:
        rows = out.shape[0]
        if rows == 0:
            return
        assert out.stride(1) == 1 and out.shape[1] == self.top_k
        owned = counts is not None
        if owned:
            assert (
                counts.dtype == torch.int32 and counts.is_contiguous() and counts.shape[0] >= rows
            )
        _ks_merge_kernel[(rows,)](
            self.keys,
            self.i32,
            self.scores,
            self.d32,
            out,
            out.stride(0),
            self.state,
            rows,
            self.rows_max,
            self.local_width,
            self.max_spin,
            self.rank,
            counts if owned else self.state,
            W=self.world,
            K=self.top_k,
            L=self.local_k,
            TPB=KEY_SHARD_TOKENS_PER_BLOCK,
            CH=2048,
            OWNED=owned,
            PDL=self.pdl,
            num_warps=self.warps,
            launch_pdl=self.pdl,
        )

    def stats(self) -> List[int]:
        """[epoch, ticket, fallback rows, spin timeouts] (host read; syncs)."""
        return self.state.tolist()


_KEY_SHARD_MERGES: dict = {}


def get_key_shard_merge(
    mapping,
    rows_max: int,
    top_k: int,
    local_width: int,
    local_k: int,
    owner_rows: bool = False,
    pdl: bool = True,
) -> KeyShardTopKMerge:
    """The TP group's merge workspace (collective on first use; call outside CUDA graph capture).

    ``owner_rows``: reduce each row on one owner rank (Helix ``merge_owned`` only, ``key_shard_owner_rows``).
    ``pdl``: launch the pack and merge with programmatic dependent launch."""
    key = (
        mapping.tp_size,
        mapping.tp_rank,
        mapping.pp_rank,
        rows_max,
        top_k,
        local_width,
        local_k,
        owner_rows,
        pdl,
    )
    merge = _KEY_SHARD_MERGES.get(key)
    if merge is not None:
        return merge
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )

    granularity = 1 << 21
    size = KeyShardTopKMerge.bytes_needed(rows_max, top_k, mapping.tp_size, local_k, local_width)
    size = -(-size // granularity) * granularity
    comm = _get_mnnvl_workspace_comm(mapping)
    use_fabric_handle = (
        mapping.is_multi_node() or os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"
    )
    handle = _make_mnnvl_mcast_buffer(comm, size, mapping, use_fabric_handle)
    buffers = [handle.get_uc_buffer(r, (size,), torch.uint8, 0) for r in range(mapping.tp_size)]
    buffers[mapping.tp_rank].zero_()
    torch.cuda.synchronize()
    # Every rank has zeroed its flags before any peer publishes into the slots.
    if not _mnnvl_workspace_all_succeeded(comm, True):
        raise RuntimeError("key-sharded indexer workspace setup failed on a peer")
    merge = KeyShardTopKMerge(
        buffers,
        mapping.tp_rank,
        rows_max,
        top_k,
        local_width,
        local_k,
        owner_rows=owner_rows,
        pdl=pdl,
    )
    merge.handle = handle
    merge.comm = comm
    _KEY_SHARD_MERGES[key] = merge
    return merge
