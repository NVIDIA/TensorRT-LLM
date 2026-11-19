# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key-sharded DSA decode Top-K merge over the Helix CP group: Triton kernels, their launchers and the
workspace layout. The helix8 target calls them through ``ops/attention/rubin_dsa_key_shard.py``.

Under Helix CP rank ``r`` of ``W`` holds only the 64-token pages ``g`` of a sequence with
``g % W == r``. It scores its own keys, takes its local best ``L`` keys by score (``L`` = 512, an
exact local Top-K) and the ranks merge them into the global Top-K, of which each rank keeps its
own winners as local token indices (:func:`key_shard_merge`).

Selection order: ``(score desc, token index asc)`` on the order-preserving bit key of the fp32
score (the self-sampling GVR deterministic order: ``-0.0`` ranks below ``+0.0``). Each candidate
is one int64 ``(ordered score << 32) | (2^32 - 1 - global index)``, so a larger key is a better
candidate and keys are unique.

Merge: the local logits kernel writes this rank's scores straight into its symmetric (MNNVL/IPC)
workspace. Each rank pushes its ``L`` candidates (in ascending local position), their count, its
local key count and its smallest sent key into every rank's receive slot, then publishes
a per-row flag on every rank. Each rank selects the ``K`` best of the ``W * L`` received candidates
and checks a certificate: if some rank had more than ``L`` keys and its smallest sent score is not
below the selection threshold's score, its unsent keys could belong to the global Top-K, so the
row is re-selected exactly from every rank's local scores (read remotely). With ``failing_only``
only the failing ranks' local scores are read, and the other ranks' received candidates stand in for
theirs: a rank whose certificate holds has no unsent key at or above the candidates' threshold, so the
re-selected rows are the same (a row where every rank failed reads them all, as without). The receive
slots alternate per call (two slots); a call-count epoch (advanced by the merge) tags the flags, so a
rank reuses a receive slot only after every peer has read it. The local scores have one slot: a
rank overwrites them in its next call, which it reaches only after the Helix exchange of this
call's attention outputs, i.e. after every peer has finished this call's merge.

The step metadata (``_key_shard_config`` of ``ops/attention/rubin_dsa_metadata.py``) sizes the merge
workspace from the server's limits (max batch x (max draft + 1) rows); a step with more rows is an error. The
workspace (:class:`KeyShardWorkspace`) is built once, outside CUDA graph capture, and passed to every launch:
the launchers keep no state of their own. The epoch and the merge counters live in its ``state``
tensor on the device.
"""

from __future__ import annotations

from typing import Any, List, NamedTuple, Optional

import torch
import triton
import triton.language as tl

KEY_SHARD_TOKENS_PER_BLOCK = 64
_INT64_MIN = tl.constexpr(-(2**63))
_INT64_MIN_PLUS1 = tl.constexpr(-(2**63) + 1)


# Keys each rank takes from its shard. Every rank pushes all of them, so this bounds the merge's work
# (world x local_k keys per row) and the exchanged bytes; 512 is 2x the fair share of a 2048 Top-K at 8 ranks.
# Rows where a rank holds more than this many of the global Top-K are re-selected exactly from the local
# scores, so the value does not change the rows.
KEY_SHARD_LOCAL_K = 512


def key_shard_local_k() -> int:
    """Keys each rank takes from its shard (:data:`KEY_SHARD_LOCAL_K`)."""
    return KEY_SHARD_LOCAL_K


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
):
    """Push this rank's ``L`` candidate keys of one row and the row header into every rank's receive slot and
    publish one flag per rank (the row's local scores already sit in this rank's workspace)."""
    slot = (e & 1).to(tl.int64)
    n = tl.load(rowlen_ptr + row)
    src = logits_ptr + row * logits_stride
    j = tl.arange(0, L)
    idx = tl.load(topk_ptr + row * L + j)
    # Ascending local positions (-1 padding last): each rank's merge then writes its winners in ascending local
    # position, an order that depends only on the selected set (the verify rows of a sequence share most of their
    # keys, so the sparse attention gathers the same cache rows in the same order).
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
def _ks_exact_threshold_failing(
    i32_ptr,
    f32_ptr,
    d32_ptr,
    row,
    slot,
    rw,
    LW,
    flat,
    keep,
    fbits,
    total_c,
    W: tl.constexpr,
    K: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
):
    """Threshold key of a row's global Top-``K`` re-selected from the received candidates ``flat`` that ``keep``
    marks (the ranks whose certificate holds) and every local score of the ranks in ``fbits`` (bit ``w``: rank
    ``w``'s certificate failed), streamed from them; ``total_c`` keys in all. A rank whose certificate holds has no
    unsent key at or above the candidates' threshold, which is at most the row's global ``K``-th key, so the ``K``
    largest keys of this set are the row's global Top-``K``. The byte-wise radix select of
    :func:`_ks_exact_threshold` over that set."""
    base = row * LW
    sign_f = tl.full([], -9223372036854775808, tl.int64)
    bins_f = tl.arange(0, 256)
    u_c = flat ^ sign_f
    prefix_f = tl.full([], 0, tl.int64)
    remaining_f = tl.zeros([], tl.int32) + tl.minimum(total_c, K)
    done_f = tl.zeros([], tl.int32)
    for p in tl.static_range(8):
        if done_f == 0:
            act = keep
            if p > 0:
                act = act & ((u_c >> (64 - 8 * p)) == (prefix_f >> (64 - 8 * p)))
            h = tl.histogram(tl.where(act, ((u_c >> (56 - 8 * p)) & 0xFF).to(tl.int32), 0), 256)
            hist_f = tl.where(bins_f == 0, h - tl.sum((act == 0).to(tl.int32), axis=0), h)
            for w in tl.static_range(W):
                if ((fbits >> w) & 1) != 0:
                    n_w = tl.load(i32_ptr + 2 * rw + slot * rw + row * W + w, volatile=True)
                    hist_f += _ks_scan_local(
                        f32_ptr, d32_ptr, base, n_w, w, prefix_f, p, W, TPB, CH
                    )
            suffix_f = tl.sum(hist_f, axis=0) - tl.cumsum(hist_f, axis=0) + hist_f
            bstar_f = tl.max(tl.where(suffix_f >= remaining_f, bins_f, -1), axis=0)
            above_f = tl.sum(tl.where(bins_f > bstar_f, hist_f, 0), axis=0)
            equal_f = tl.sum(tl.where(bins_f == bstar_f, hist_f, 0), axis=0)
            remaining_f = remaining_f - above_f
            prefix_f = prefix_f | (bstar_f.to(tl.int64) << (56 - 8 * p))
            done_f = tl.where(equal_f == remaining_f, 1, done_f)
    thr_c = prefix_f ^ sign_f
    if total_c < K:
        thr_c = tl.full([], _INT64_MIN_PLUS1, tl.int64)
    return thr_c


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
    bounds_ptr,
    mask_ptr,
    W: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
    MASK: tl.constexpr,
    FAILING: tl.constexpr,
):
    """This rank's share of the global Top-K of one row, from the ``W`` pushed candidate lists in its receive slot.

    Writes only this rank's winners, as local token indices, kept entries first in output order and the rest ``-1``,
    plus their count to ``count_ptr[row]`` (a Helix rank's sparse attention reads its own pages only). Rank ``w``'s
    candidates are its own keys, so this rank's winners are the selected entries of its segment. ``MASK``: also
    write the row's Helix combine mask, ``(bounds_ptr[row] == 0) | (count == 0)``, to ``mask_ptr[row]``, so the
    attention needs no separate mask launch. ``FAILING``: a row whose certificate fails on some but not every rank
    is re-selected from those ranks' local scores and the other ranks' candidates (the same rows, fewer reads).
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
    fail_v = (cnt > sent) & ((boundary >> 32) >= (thr >> 32))
    fail = tl.sum(fail_v.to(tl.int32), axis=0) > 0
    kk = tl.arange(0, K)
    orow = out_ptr + row * out_stride
    if fail:
        # Re-select exactly (rare): byte-wise radix select streaming local scores.
        tl.atomic_add(state_ptr + 2, 1)
        # The ranks whose certificate failed, one bit each; all of them unless FAILING.
        if FAILING:
            fbits = tl.sum(tl.where(fail_v, 1 << wv, 0), axis=0)
        else:
            fbits = tl.zeros([], tl.int32) + ((1 << W) - 1)
        if fbits != (1 << W) - 1:
            # From the failing ranks' local scores and the other ranks' candidates (FAILING).
            seg = c // L
            keep = valid & (((fbits >> seg) & 1) == 0)
            total_c = tl.sum(keep.to(tl.int32), axis=0) + tl.sum(tl.where(fail_v, cnt, 0), axis=0)
            thr_c = _ks_exact_threshold_failing(
                i32_ptr,
                f32_ptr,
                d32_ptr,
                row,
                slot,
                rw,
                LW,
                flat,
                keep,
                fbits,
                total_c,
                W,
                K,
                TPB,
                CH,
            )
            if ((fbits >> rank) & 1) != 0:
                # This rank's certificate failed: its winners from its own local scores, in column order.
                n_r = tl.load(i32_ptr + 2 * rw + slot * rw + row * W + rank, volatile=True)
                own_scores = f32_ptr + tl.load(d32_ptr + rank) + row * LW
                off_c = tl.zeros([], tl.int32)
                for c0 in range(0, n_r, CH):
                    cc = c0 + tl.arange(0, CH)
                    m = cc < n_r
                    s_c = tl.load(own_scores + cc, mask=m, other=0.0, cache_modifier=".cv")
                    g_c = ((cc // TPB) * W + rank) * TPB + (cc % TPB)
                    sel_c = m & (_order_key(s_c, g_c) >= thr_c)
                    pos_c = off_c + tl.cumsum(sel_c.to(tl.int32), axis=0) - 1
                    tl.store(orow + pos_c, cc.to(tl.int32), mask=sel_c)
                    off_c += tl.sum(sel_c.to(tl.int32), axis=0)
                nsel_c = off_c
            else:
                # Its certificate holds: its winners are among its candidates, in ascending local position.
                sel_c = keep & (flat >= thr_c) & (seg == rank)
                g_c = 4294967295 - (flat & 4294967295)
                g_c = (g_c // TPB) // W * TPB + g_c % TPB
                pos_c = tl.cumsum(sel_c.to(tl.int32), axis=0) - 1
                tl.store(orow + pos_c, g_c.to(tl.int32), mask=sel_c)
                nsel_c = tl.sum(sel_c.to(tl.int32), axis=0)
            tl.store(orow + kk, tl.full([K], -1, tl.int32), mask=kk >= nsel_c)
            tl.store(count_ptr + row, nsel_c)
            if MASK:
                tl.store(mask_ptr + row, (tl.load(bounds_ptr + row) == 0) | (nsel_c == 0))
        else:
            # From every rank's local scores.
            base = row * LW
            thr_all = _ks_exact_threshold(
                i32_ptr, f32_ptr, d32_ptr, row, slot, rw, LW, total_all, W, K, TPB, CH
            )
            off_f = tl.zeros([], tl.int32)
            for w in tl.static_range(W):
                # The threshold counted every rank's keys; this rank writes its own winners.
                if w == rank:
                    n_w = tl.load(i32_ptr + 2 * rw + slot * rw + row * W + w, volatile=True)
                    row_scores = f32_ptr + tl.load(d32_ptr + w) + base
                    for c0 in range(0, n_w, CH):
                        cc = c0 + tl.arange(0, CH)
                        m = cc < n_w
                        s_f = tl.load(row_scores + cc, mask=m, other=0.0, cache_modifier=".cv")
                        g_f = ((cc // TPB) * W + w) * TPB + (cc % TPB)
                        sel_f = m & (_order_key(s_f, g_f) >= thr_all)
                        pos_f = off_f + tl.cumsum(sel_f.to(tl.int32), axis=0) - 1
                        tl.store(orow + pos_f, cc.to(tl.int32), mask=sel_f)
                        off_f += tl.sum(sel_f.to(tl.int32), axis=0)
            tl.store(orow + kk, tl.full([K], -1, tl.int32), mask=kk >= off_f)
            tl.store(count_ptr + row, off_f)
            if MASK:
                tl.store(mask_ptr + row, (tl.load(bounds_ptr + row) == 0) | (off_f == 0))
    else:
        sel = valid & (flat >= thr)
        g = 4294967295 - (flat & 4294967295)
        # This rank's segment, as local token indices.
        sel = sel & ((c // L) == rank)
        g = (g // TPB) // W * TPB + g % TPB
        pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(orow + pos, g.to(tl.int32), mask=sel)
        nsel = tl.sum(sel.to(tl.int32), axis=0)
        tl.store(orow + kk, tl.full([K], -1, tl.int32), mask=kk >= nsel)
        tl.store(count_ptr + row, nsel)
        if MASK:
            tl.store(mask_ptr + row, (tl.load(bounds_ptr + row) == 0) | (nsel == 0))


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
):
    row = tl.program_id(0).to(tl.int64)
    e = tl.load(state_ptr) + 1
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
    bounds_ptr,
    mask_ptr,
    W: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
    MASK: tl.constexpr,
    FAILING: tl.constexpr,
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
        bounds_ptr,
        mask_ptr,
        W,
        K,
        L,
        TPB,
        CH,
        MASK,
        FAILING,
    )
    tl.extra.cuda.gdc_wait()
    _ks_advance_epoch(state_ptr, e, num_rows)


# Launch settings of the pack / merge (chosen by measurement on the helix8 generation server):
# - 8 warps per program;
# - programmatic dependent launch: the pack waits for the local Top-K after its first loads and lets
#   the merge launch at once; the merge waits only before it ends, so the next kernel never overlaps
#   the pack.
_WARPS = 8
# Spin bound of the merge's wait for a peer's flags; a timeout is counted in state[3].
_MAX_SPIN = 1 << 26
# Columns the exact fallback streams per step.
_FALLBACK_CHUNK = 2048


class KeyShardWorkspace(NamedTuple):
    """One rank's view of the merge workspace (:func:`key_shard_workspace`).

    Buffer layout (identical on every rank; the receive side has two slots alternating by epoch parity):
      int64 recv[2][rows][W][L]        candidate keys pushed by each rank (INT64_MIN padded)
      int64 boundary[2][rows][W]       smallest key each rank sent
      int32 sent[2][rows][W], count[2][rows][W] (local keys of the row), flag[2][rows][W]
      fp32  scores[rows][width]        this rank's local logits (``local_logits``; read by peers only on fallback)
    """

    world: int
    rank: int
    rows_max: int
    top_k: int
    local_k: int
    local_width: int  # the local logits' row width, rounded up to the logits kernel's 256 columns
    max_spin: int
    keys: torch.Tensor  # int64 receive slots and boundaries
    i32: torch.Tensor  # int32 sent / count / flag cells
    scores: torch.Tensor  # fp32 local logits, flat
    # [rows_max, local_width] view of ``scores``: the local logits' output
    local_logits: torch.Tensor
    local_topk: torch.Tensor  # [rows_max, local_k] int32: the local Top-L rows
    counts: torch.Tensor  # [rows_max] int32: kept counts of a merge whose caller passes none
    dk: torch.Tensor  # int64 pointer deltas of the peers' buffers (8-byte units)
    d32: torch.Tensor  # the same in 4-byte units
    state: torch.Tensor  # int32 [epoch, ticket, fallback rows, spin timeouts]
    buffers: tuple  # every rank's buffer, in rank order (keeps the peers' memory alive)
    owner: Any  # the allocation's owner (MNNVL handle and communicator), or None
    mapping: Any  # the mapping whose TP group is the merge's group, or None


def key_shard_bytes_needed(
    rows_max: int, top_k: int, world: int, local_k: int, local_width: int
) -> int:
    """Bytes of one rank's workspace buffer."""
    local_width = -(-local_width // 256) * 256
    keys_i32 = 8 * 2 * rows_max * world * (local_k + 1) + 4 * 6 * rows_max * world
    return -(-keys_i32 // 16) * 16 + 4 * rows_max * local_width


def key_shard_workspace(
    buffers: List[torch.Tensor],
    rank: int,
    rows_max: int,
    top_k: int,
    local_width: int,
    local_k: Optional[int] = None,
    max_spin: int = _MAX_SPIN,
    owner: Any = None,
    mapping: Any = None,
) -> KeyShardWorkspace:
    """The workspace over ``buffers``: this rank's view of every rank's symmetric buffer, in rank order.

    They come from an MNNVL / IPC allocation over the group, or are plain local tensors that emulate
    ``len(buffers)`` ranks on one GPU (tests). Allocates the local Top-L rows, the counts scratch and
    the state; call it outside CUDA graph capture.
    """
    world = len(buffers)
    assert world & (world - 1) == 0, "the merge needs a power-of-two group size"
    local_k = local_k or key_shard_local_k()
    assert local_k <= top_k
    # The logits kernel also writes its 256-column alignment padding.
    local_width = -(-local_width // 256) * 256
    need = key_shard_bytes_needed(rows_max, top_k, world, local_k, local_width)
    mine = buffers[rank]
    assert mine.numel() * mine.element_size() >= need
    base = mine.data_ptr()
    deltas = [b.data_ptr() - base for b in buffers]
    assert all(d % 8 == 0 for d in deltas) and base % 16 == 0
    dev = mine.device
    byte_off = mine.storage_offset() * mine.element_size()
    n_keys = 2 * rows_max * world * (local_k + 1)
    n_i32 = 6 * rows_max * world
    storage = mine.untyped_storage()
    keys = torch.empty(0, dtype=torch.int64, device=dev).set_(storage, byte_off // 8, (n_keys,))
    i32 = torch.empty(0, dtype=torch.int32, device=dev).set_(
        storage, (byte_off + 8 * n_keys) // 4, (n_i32,)
    )
    f32_off = -(-(byte_off + 8 * n_keys + 4 * n_i32) // 16) * 16
    scores = torch.empty(0, dtype=torch.float32, device=dev).set_(
        storage, f32_off // 4, (rows_max * local_width,)
    )
    return KeyShardWorkspace(
        world=world,
        rank=rank,
        rows_max=rows_max,
        top_k=top_k,
        local_k=local_k,
        local_width=local_width,
        max_spin=max_spin,
        keys=keys,
        i32=i32,
        scores=scores,
        local_logits=scores.view(rows_max, local_width),
        local_topk=torch.full((rows_max, local_k), -1, dtype=torch.int32, device=dev),
        counts=torch.zeros(rows_max, dtype=torch.int32, device=dev),
        dk=torch.tensor([d // 8 for d in deltas], dtype=torch.int64, device=dev),
        d32=torch.tensor([d // 4 for d in deltas], dtype=torch.int64, device=dev),
        state=torch.zeros(4, dtype=torch.int32, device=dev),
        buffers=tuple(buffers),
        owner=owner,
        mapping=mapping,
    )


def key_shard_pack(ws: KeyShardWorkspace, local_topk: torch.Tensor, row_lens: torch.Tensor) -> None:
    """Publish this rank's local Top-``local_k`` (local column indices, -1 padded) of the ``ws.local_logits``
    rows whose valid local lengths are ``row_lens``."""
    rows = local_topk.shape[0]
    if rows == 0:
        return
    assert (
        rows <= ws.rows_max
        and local_topk.shape[1] == ws.local_k
        and local_topk.is_contiguous()
        and row_lens.dtype == torch.int32
    )
    _ks_pack_kernel[(rows,)](
        ws.local_logits,
        ws.local_width,
        local_topk,
        row_lens,
        ws.keys,
        ws.i32,
        ws.dk,
        ws.d32,
        ws.state,
        ws.rank,
        ws.rows_max,
        W=ws.world,
        L=ws.local_k,
        TPB=KEY_SHARD_TOKENS_PER_BLOCK,
        num_warps=_WARPS,
        launch_pdl=True,
    )


def key_shard_merge(
    ws: KeyShardWorkspace,
    out: torch.Tensor,
    counts: Optional[torch.Tensor] = None,
    bounds: Optional[torch.Tensor] = None,
    mask_out: Optional[torch.Tensor] = None,
    failing_only: bool = False,
) -> None:
    """Write this rank's share of every row's global Top-K into ``out`` [rows, top_k] as local token indices.

    The kept entries come first, the rest is ``-1``; they are in ascending local position (the pushes are
    sorted), so a row's order depends only on its set. Mapped back to global positions, the ranks' rows
    partition the global Top-K. ``counts`` [rows] int32 receives the kept count of each row (``ws.counts`` when
    None). With ``bounds`` [rows] (each row's rank-local KV bound) ``mask_out`` [rows] bool receives the Helix
    combine mask ``(bounds == 0) | (count == 0)``, the values ``helix_zero_or_empty_rows`` computes.
    ``failing_only``: a row whose certificate fails is re-selected from the failing ranks' local scores and the
    other ranks' candidates instead of every rank's local scores, unless every rank failed (the same values).
    """
    rows = out.shape[0]
    if mask_out is not None:
        assert bounds is not None and bounds.dim() == 1 and bounds.is_contiguous()
        assert bounds.dtype in (torch.int32, torch.int64) and bounds.shape[0] >= rows
        assert (
            mask_out.dtype == torch.bool and mask_out.is_contiguous() and mask_out.shape[0] >= rows
        )
    if counts is None:
        counts = ws.counts
    if rows == 0:
        return
    assert rows <= ws.rows_max and out.stride(1) == 1 and out.shape[1] == ws.top_k
    assert counts.dtype == torch.int32 and counts.is_contiguous() and counts.shape[0] >= rows
    _ks_merge_kernel[(rows,)](
        ws.keys,
        ws.i32,
        ws.scores,
        ws.d32,
        out,
        out.stride(0),
        ws.state,
        rows,
        ws.rows_max,
        ws.local_width,
        ws.max_spin,
        ws.rank,
        counts,
        bounds if mask_out is not None else ws.state,
        mask_out if mask_out is not None else ws.state,
        W=ws.world,
        K=ws.top_k,
        L=ws.local_k,
        TPB=KEY_SHARD_TOKENS_PER_BLOCK,
        CH=_FALLBACK_CHUNK,
        MASK=mask_out is not None,
        FAILING=bool(failing_only),
        num_warps=_WARPS,
        launch_pdl=True,
    )
