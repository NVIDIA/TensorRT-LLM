# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key-sharded DSA decode Top-K merge over the Helix CP group (see ``helix_key_shard.py``).

Under Helix CP rank ``r`` of ``W`` holds only the 64-token pages ``g`` of a sequence with
``g % W == r``. It scores its own keys, takes its local best ``L`` keys by score (``L`` = 512, an
exact local Top-K) and the ranks merge them into the global Top-K, of which each rank keeps its
own winners as local token indices (:meth:`KeyShardTopKMerge.merge_owned`).

Selection order: ``(score desc, token index asc)`` on the order-preserving bit key of the fp32
score (the self-sampling GVR deterministic order: ``-0.0`` ranks below ``+0.0``). Each candidate
is one int64 ``(ordered score << 32) | (2^32 - 1 - global index)``, so a larger key is a better
candidate and keys are unique.

Merge: the local logits kernel writes this rank's scores straight into its symmetric (MNNVL/IPC)
workspace. Each rank pushes its ``L`` candidates (in ascending local position), their count, its
local key count and its smallest sent key into every rank's receive slot, then publishes
a per-row flag on every rank. Each rank selects the ``K`` best of the ``W * L`` received candidates
(threshold ``thr``) and checks a certificate per rank: a rank is *short* if it had more than ``L``
keys and its smallest sent key could be followed by an unsent key of the global Top-K, i.e. its
smallest sent score is not below ``thr``'s score, or, when every rank's candidates are its exact
Top-``L`` by key (``ties_exact``: score desc, index asc), its smallest sent key is above ``thr``.
A row with a short rank is re-selected exactly, reading only the short ranks' local scores
(remotely): every unsent key of a rank that is not short is below ``thr``, and ``thr`` is at most
the global ``K``-th key, so the global Top-K is the Top-K of the other ranks' received keys and the
short ranks' local keys that are ``>= thr``. Those are collected once into a per-row scratch list
and radix-selected there; if a row's list overflows, the radix select streams the short ranks'
scores instead (still exact). The receive slots
alternate per call (two slots); a call-count epoch (advanced by the merge) tags the flags, so a
rank reuses a receive slot only after every peer has read it. The local scores have one slot: a
rank overwrites them in its next call, which it reaches only after the Helix exchange of this
call's attention outputs, i.e. after every peer has finished this call's merge.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import triton
import triton.language as tl

KEY_SHARD_TOKENS_PER_BLOCK = 64
_INT64_MIN = tl.constexpr(-(2**63))
_INT64_MIN_PLUS1 = tl.constexpr(-(2**63) + 1)


# Keys each rank takes from its shard. Every rank pushes all of them, so this bounds the merge's work
# (world x local_k keys per row) and the exchanged bytes. The default is 2x the fair share of the Top-K
# (512 for a 2048 Top-K at 8 ranks, 256 at 16), never below KEY_SHARD_MIN_LOCAL_K.
# Rows where a rank holds more than this many of the global Top-K are re-selected exactly from the local
# scores, so the value does not change the rows. ``TRTLLM_DSA_KEY_SHARD_LOCAL_K`` overrides it.
KEY_SHARD_LOCAL_K = 512
KEY_SHARD_MIN_LOCAL_K = 256


def key_shard_local_k(world: Optional[int] = None, top_k: int = 2048) -> int:
    """Keys each rank takes from its shard: ``TRTLLM_DSA_KEY_SHARD_LOCAL_K`` if set, else 2x the
    fair share of ``top_k`` over ``world`` ranks (at least :data:`KEY_SHARD_MIN_LOCAL_K`), else
    :data:`KEY_SHARD_LOCAL_K` when the world is not known."""
    override = os.environ.get("TRTLLM_DSA_KEY_SHARD_LOCAL_K")
    if override:
        return min(int(override), top_k)
    if world is None:
        return KEY_SHARD_LOCAL_K
    return min(top_k, max(KEY_SHARD_MIN_LOCAL_K, 2 * top_k // world))


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
    hist = tl.zeros([256], tl.int32)
    row_scores = f32_ptr + tl.load(d32_ptr + w) + base
    for c0 in range(0, n, CH):
        cc = c0 + tl.arange(0, CH)
        m = cc < n
        s = tl.load(row_scores + cc, mask=m, other=0.0, cache_modifier=".cv")
        g = ((cc // TPB) * W + w) * TPB + (cc % TPB)
        hist += _ks_digit_hist(_order_key(s, g) ^ sign, m, prefix, p)
    return hist


@triton.jit
def _ks_digit_hist(u, act, prefix, p: tl.constexpr):
    """Byte-``p`` histogram (256 bins) of the unsigned key images ``u`` where ``act``, restricted to the images
    matching ``prefix`` on the bytes above ``p``."""
    bins = tl.arange(0, 256)
    if p > 0:
        act = act & ((u >> (64 - 8 * p)) == (prefix >> (64 - 8 * p)))
    h = tl.histogram(tl.where(act, ((u >> (56 - 8 * p)) & 0xFF).to(tl.int32), 0), 256)
    return tl.where(bins == 0, h - tl.sum((act == 0).to(tl.int32), axis=0), h)


@triton.jit
def _ks_radix_step(hist, remaining, prefix, p: tl.constexpr):
    """One byte of an MSB-first radix select: the boundary byte of the ``remaining`` largest keys counted in
    ``hist``; returns the keys still to take from the boundary bucket, the extended prefix and whether the
    boundary bucket is taken whole."""
    bins = tl.arange(0, 256)
    suffix = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist
    bstar = tl.max(tl.where(suffix >= remaining, bins, -1), axis=0)
    above = tl.sum(tl.where(bins > bstar, hist, 0), axis=0)
    equal = tl.sum(tl.where(bins == bstar, hist, 0), axis=0)
    remaining = remaining - above
    prefix = prefix | (bstar.to(tl.int64) << (56 - 8 * p))
    return remaining, prefix, equal == remaining


@triton.jit
def _ks_short(sent, cnt, boundary, thr, TIES: tl.constexpr):
    """The certificate per rank: whether a rank's unsent keys could belong to the global Top-K, given the
    ``K``-th largest received key ``thr`` (a lower bound of the global ``K``-th key) and the rank's smallest sent
    key ``boundary``. Its unsent keys score at most ``boundary``'s score (any local Top-L by score); with ``TIES``
    (the candidates are the exact local Top-L by key) they are also below ``boundary`` as keys."""
    if TIES:
        reaches = boundary > thr
    else:
        reaches = (boundary >> 32) >= (thr >> 32)
    return (cnt > sent) & reaches


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
def _ks_reselect(
    flat,
    valid,
    thr,
    short,
    cnt,
    f32_ptr,
    d32_ptr,
    list_ptr,
    state_ptr,
    row,
    rank,
    LW,
    cap,
    W: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
    CR: tl.constexpr,
):
    """Threshold key ``t`` of a row's global Top-``K`` (the Top-K are exactly the keys ``>= t``) when some rank is
    ``short`` ([W]); the row has at least ``K`` keys. ``cnt`` [W]: the ranks' local key counts.

    Candidates: the received keys ``>= thr`` of the ranks that are not short and the local keys ``>= thr`` of the
    short ranks, read once from their local scores (remotely, ``CR`` at a time) into this row's list of ``cap``
    keys, rank by rank in ascending local position. Every global Top-K key is a candidate (``thr`` is at most the
    global ``K``-th key, and a rank that is not short has no unsent key ``>= thr``), so the ``K``-th largest
    candidate is the global ``K``-th key; it is found by a byte-wise radix select over the received keys
    (registers) and the list. A row whose list overflows streams the short ranks' local scores in every radix
    pass instead (all their keys; counted in ``counters[4]``).

    Returns ``(t, listed, lo, hi)``: ``listed`` when the list held every candidate, and then this rank's own
    candidates are ``list[lo:hi]``.
    """
    sign = tl.full([], -9223372036854775808, tl.int64)
    wv = tl.arange(0, W)
    base = row * LW
    short_i = short.to(tl.int32)
    d32v = tl.load(d32_ptr + wv)
    seg_short = tl.reshape(tl.broadcast_to(short_i[:, None], (W, L)), (W * L,))
    keep = valid & (flat >= thr) & (seg_short == 0)
    u = flat ^ sign
    lst = list_ptr + row * cap
    n_list = tl.zeros([], tl.int32)
    lo = tl.zeros([], tl.int32)
    hi = tl.zeros([], tl.int32)
    for w in range(W):
        at = wv == w
        n_w = tl.sum(tl.where(at, cnt, 0), axis=0)
        lo = tl.where(w == rank, n_list, lo)
        if (tl.sum(tl.where(at, short_i, 0), axis=0) > 0) & (n_list <= cap):
            row_scores = f32_ptr + tl.sum(tl.where(at, d32v, 0), axis=0) + base
            for c0 in range(0, n_w, CR):
                cc = c0 + tl.arange(0, CR)
                m = cc < n_w
                s = tl.load(row_scores + cc, mask=m, other=0.0, cache_modifier=".cv")
                g = ((cc // TPB) * W + w) * TPB + (cc % TPB)
                key = _order_key(s, g)
                sel = m & (key >= thr)
                taken = tl.sum(sel.to(tl.int32), axis=0)
                if taken > 0:
                    pos = n_list + tl.cumsum(sel.to(tl.int32), axis=0) - 1
                    tl.store(lst + pos, key, mask=sel & (pos < cap))
                n_list += taken
        hi = tl.where(w == rank, n_list, hi)
    spilled = n_list > cap
    if spilled:
        tl.atomic_add(state_ptr + 4, 1)
    # The list is read back by other threads of the program.
    tl.debug_barrier()
    prefix = tl.full([], 0, tl.int64)
    remaining = tl.zeros([], tl.int32) + K
    done = tl.zeros([], tl.int32)
    for p in tl.static_range(8):
        if done == 0:
            hist = _ks_digit_hist(u, keep, prefix, p)
            if spilled:
                for w in range(W):
                    at = wv == w
                    if tl.sum(tl.where(at, short_i, 0), axis=0) > 0:
                        n_w = tl.sum(tl.where(at, cnt, 0), axis=0)
                        hist += _ks_scan_local(
                            f32_ptr, d32_ptr, base, n_w, w, prefix, p, W, TPB, CH
                        )
            else:
                for c0 in range(0, n_list, CH):
                    cc = c0 + tl.arange(0, CH)
                    m = cc < n_list
                    key = tl.load(lst + cc, mask=m, other=0, cache_modifier=".cg")
                    hist += _ks_digit_hist(key ^ sign, m, prefix, p)
            remaining, prefix, whole = _ks_radix_step(hist, remaining, prefix, p)
            done = tl.where(whole, 1, done)
    # The Top-K are the candidates >= the radix threshold; every key >= thr is a candidate.
    return tl.maximum(prefix ^ sign, thr), spilled == 0, lo, hi


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
    list_ptr,
    list_cap,
    W: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
    MASK: tl.constexpr,
    TIES: tl.constexpr,
):
    """This rank's share of the global Top-K of one row, from the ``W`` pushed candidate lists in its receive slot.

    Writes only this rank's winners, as local token indices, kept entries first in output order and the rest ``-1``,
    plus their count to ``count_ptr[row]`` (a Helix rank's sparse attention reads its own pages only). Rank ``w``'s
    candidates are its own keys, so this rank's winners are the selected entries of its segment, unless it is short
    (then they come from the re-selection's candidate list, or its local scores). ``MASK``: also write the row's
    Helix combine mask,
    ``(bounds_ptr[row] == 0) | (count == 0)``, to ``mask_ptr[row]``, so the attention needs no separate mask launch.
    ``TIES``: every rank's candidates are its exact local Top-L by key (score desc, index asc), which sharpens the
    certificate (:func:`_ks_short`). ``list_ptr`` [rows_max, list_cap] int64: the re-selection's candidate lists.
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
    short = _ks_short(sent, cnt, boundary, thr, TIES)
    fail = tl.sum(short.to(tl.int32), axis=0) > 0
    kk = tl.arange(0, K)
    orow = out_ptr + row * out_stride
    own_short = fail & (tl.sum((short & (wv == rank)).to(tl.int32), axis=0) > 0)
    listed = tl.zeros([], tl.int1)
    lo = tl.zeros([], tl.int32)
    hi = tl.zeros([], tl.int32)
    if fail:
        # Re-select exactly from the short ranks' local scores (rare).
        tl.atomic_add(state_ptr + 2, 1)
        if total_all < K:
            thr = tl.full([], _INT64_MIN_PLUS1, tl.int64)
        else:
            thr, listed, lo, hi = _ks_reselect(
                flat,
                valid,
                thr,
                short,
                cnt,
                f32_ptr,
                d32_ptr,
                list_ptr,
                state_ptr,
                row,
                rank,
                LW,
                list_cap,
                W,
                K,
                L,
                TPB,
                CH,
                4096,
            )
    if own_short & listed:
        # This rank is short and its keys >= the re-selection's lower bound are list[lo:hi], in ascending local
        # position: its winners are those >= thr.
        nsel = tl.zeros([], tl.int32)
        lst = list_ptr + row * list_cap
        for c0 in range(lo, hi, CH):
            cc = c0 + tl.arange(0, CH)
            key = tl.load(lst + cc, mask=cc < hi, other=_INT64_MIN, cache_modifier=".cg")
            sel_l = key >= thr
            g_l = 4294967295 - (key & 4294967295)
            pos_l = nsel + tl.cumsum(sel_l.to(tl.int32), axis=0) - 1
            tl.store(orow + pos_l, ((g_l // TPB) // W * TPB + g_l % TPB).to(tl.int32), mask=sel_l)
            nsel += tl.sum(sel_l.to(tl.int32), axis=0)
    elif own_short:
        # This rank is short: its winners are among its local keys, in ascending local position.
        nsel = tl.zeros([], tl.int32)
        n_r = tl.load(i32_ptr + 2 * rw + slot * rw + row * W + rank, volatile=True)
        row_scores = f32_ptr + row * LW
        for c0 in range(0, n_r, CH):
            cc = c0 + tl.arange(0, CH)
            m = cc < n_r
            s_f = tl.load(row_scores + cc, mask=m, other=0.0, cache_modifier=".cv")
            g_f = ((cc // TPB) * W + rank) * TPB + (cc % TPB)
            sel_f = m & (_order_key(s_f, g_f) >= thr)
            pos_f = nsel + tl.cumsum(sel_f.to(tl.int32), axis=0) - 1
            tl.store(orow + pos_f, cc.to(tl.int32), mask=sel_f)
            nsel += tl.sum(sel_f.to(tl.int32), axis=0)
    else:
        # Every winner of this rank was sent: the selected entries of its segment, as local token indices.
        sel = valid & (flat >= thr)
        g = 4294967295 - (flat & 4294967295)
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
    list_ptr,
    list_cap,
    W: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    TPB: tl.constexpr,
    CH: tl.constexpr,
    MASK: tl.constexpr,
    TIES: tl.constexpr,
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
        bounds_ptr,
        mask_ptr,
        list_ptr,
        list_cap,
        W,
        K,
        L,
        TPB,
        CH,
        MASK,
        TIES,
    )
    if PDL:
        tl.extra.cuda.gdc_wait()
    _ks_advance_epoch(state_ptr, e, num_rows)


class KeyShardTopKMerge:
    """Workspace and launches of the pack / merge steps of the key-sharded decode Top-K.

    ``buffers`` are this rank's view of every rank's symmetric buffer (rank order). They come from an MNNVL / IPC
    allocation over the Helix CP group, or from plain local tensors to emulate ``world`` ranks on one GPU (tests).

    Buffer layout (identical on every rank; the receive side has two slots alternating by epoch parity):
      int64 recv[2][rows][W][L]        candidate keys pushed by each rank (INT64_MIN padded)
      int64 boundary[2][rows][W]       smallest key each rank sent
      int32 sent[2][rows][W], count[2][rows][W] (local keys of the row), flag[2][rows][W]
      fp32  scores[rows][width]        this rank's local logits (``local_logits``; read by peers only on fallback)
      LL    ll[2][rows][ll_pairs]      the LL exchange of the flat indexer (``ll``; see :meth:`ll_pairs`)

    The LL region carries the same information without flags: every 16-byte pair holds two 32-bit words, each
    next to the call's epoch in its own 8 bytes (written and read as relaxed system-scope 64-bit elements), so a
    reader takes a word once its epoch matches. Per (slot, row): ``W * L`` key pairs (rank ``w``'s ``j``-th
    candidate at ``w * L + j``: high, low word of the unsigned key image ``key ^ INT64_MIN``; 0 when unsent),
    then two header pairs per rank (``(count, boundary high)``, ``(boundary low, 0)``, the boundary as an unsigned
    image), then ``(K - 1) // L`` rescue slots of ``K`` key pairs (a short rank's exact local Top-K, in ascending
    local position).
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
        reselect_cap: Optional[int] = None,
    ):
        self.world = len(buffers)
        assert self.world & (self.world - 1) == 0, "the merge needs a power-of-two group size"
        self.rank = rank
        self.rows_max = rows_max
        self.top_k = top_k
        self.local_k = local_k or key_shard_local_k()
        assert self.local_k <= top_k
        # The logits kernel also writes its 256-column alignment padding.
        self.local_width = -(-local_width // 256) * 256
        self.max_spin = max_spin
        # Launch settings of the pack / merge (chosen by measurement on the helix8 generation server):
        # - 8 warps per program;
        # - programmatic dependent launch: the pack waits for the local Top-K after its first loads and lets
        #   the merge launch at once; the merge waits only before it ends, so the next kernel never overlaps
        #   the pack.
        self.warps = 8
        self.pdl = True
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
        ll_off = -(-(f32_off + 4 * rows_max * self.local_width) // 16) * 16
        self.ll_block = self.ll_pairs(self.world, self.local_k, top_k)
        # LL exchange [2][rows_max][ll_block] 16-byte pairs (two int64 halves each).
        self.ll = torch.empty(0, dtype=torch.int64, device=dev).set_(
            storage, ll_off // 8, (2 * rows_max * self.ll_block * 2,)
        )
        self.dk = torch.tensor([d // 8 for d in deltas], dtype=torch.int64, device=dev)
        self.d32 = torch.tensor([d // 4 for d in deltas], dtype=torch.int64, device=dev)
        # [epoch, ticket, fallback rows, spin timeouts, fallback rows whose candidate list overflowed]; ``state``
        # (the kernels' argument) is the first four, the merge also counts into the fifth.
        self.counters = torch.zeros(5, dtype=torch.int32, device=dev)
        self.state = self.counters[:4]
        # The re-selection's candidate list per row (local; written and read only by the row's merge program):
        # the short ranks' keys >= the received K-th key, usually the global Top-K's few hundred keys there.
        self.reselect_cap = reselect_cap or 4 * top_k
        self.reselect = torch.empty((rows_max, self.reselect_cap), dtype=torch.int64, device=dev)
        self._stats_every = int(os.environ.get("TRTLLM_DSA_KEY_SHARD_STATS", "0"))
        self._stats_steps = 0

    @staticmethod
    def ll_pairs(world: int, local_k: int, top_k: int) -> int:
        """16-byte LL pairs of one (slot, row) block of the LL exchange: ``world * local_k`` candidates, two header
        pairs per rank, and ``(top_k - 1) // local_k`` rescue slots of ``top_k`` keys (a short rank's ``local_k``
        sent keys all lie above the ``top_k``-th selected key, so at most that many ranks are short)."""
        return world * (local_k + 2) + ((top_k - 1) // local_k) * top_k

    @staticmethod
    def bytes_needed(rows_max: int, top_k: int, world: int, local_k: int, local_width: int) -> int:
        local_width = -(-local_width // 256) * 256
        keys_i32 = 8 * 2 * rows_max * world * (local_k + 1) + 4 * 6 * rows_max * world
        scores_end = -(-keys_i32 // 16) * 16 + 4 * rows_max * local_width
        ll = 16 * 2 * rows_max * KeyShardTopKMerge.ll_pairs(world, local_k, top_k)
        return -(-scores_end // 16) * 16 + ll

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
            PDL=self.pdl,
            num_warps=self.warps,
            launch_pdl=self.pdl,
        )

    def merge_owned(
        self,
        out: torch.Tensor,
        counts: Optional[torch.Tensor] = None,
        bounds: Optional[torch.Tensor] = None,
        mask_out: Optional[torch.Tensor] = None,
        ties_exact: bool = False,
    ) -> None:
        """Write this rank's share of every row's global Top-K into ``out`` [rows, top_k] as local token indices.

        The kept entries come first, the rest is ``-1``; they are in ascending local position (the pushes are
        sorted), so a row's order depends only on its set. Mapped back to global positions, the ranks' rows
        partition the global Top-K. ``counts`` [rows] int32 (optional) receives the kept count of each row. With
        ``bounds`` [rows] (each row's rank-local KV bound) ``mask_out`` [rows] bool receives the Helix combine mask
        ``(bounds == 0) | (count == 0)``, the values ``helix_zero_or_empty_rows`` computes.

        ``ties_exact``: every rank's pushed candidates are its exact local Top-``local_k`` by key (score desc,
        token index asc, as ``KeyShardSelectPack`` selects them), not just a Top-``local_k`` by score. Every rank
        must then push such candidates; it lets the certificate compare whole keys, so score ties at a rank's
        boundary need no re-selection. The rows are the same either way.
        """
        if mask_out is not None:
            rows = out.shape[0]
            assert bounds is not None and bounds.dim() == 1 and bounds.is_contiguous()
            assert bounds.dtype in (torch.int32, torch.int64) and bounds.shape[0] >= rows
            assert (
                mask_out.dtype == torch.bool
                and mask_out.is_contiguous()
                and mask_out.shape[0] >= rows
            )
        if counts is None:
            counts = self._counts_scratch(out.shape[0])
        rows = out.shape[0]
        if rows == 0:
            return
        assert out.stride(1) == 1 and out.shape[1] == self.top_k
        assert counts.dtype == torch.int32 and counts.is_contiguous() and counts.shape[0] >= rows
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
            counts,
            bounds if mask_out is not None else self.state,
            mask_out if mask_out is not None else self.state,
            self.reselect,
            self.reselect_cap,
            W=self.world,
            K=self.top_k,
            L=self.local_k,
            TPB=KEY_SHARD_TOKENS_PER_BLOCK,
            CH=2048,
            MASK=mask_out is not None,
            TIES=ties_exact,
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

    def stats(self) -> List[int]:
        """[epoch, ticket, fallback rows, spin timeouts, overflowed fallback rows] (host read; syncs)."""
        return self.counters.tolist()

    def maybe_log_stats(self) -> None:
        """Every ``TRTLLM_DSA_KEY_SHARD_STATS`` host steps (0 = never), log the merge counters (one device sync)."""
        if self._stats_every <= 0:
            return
        self._stats_steps += 1
        if self._stats_steps % self._stats_every == 0:
            from tensorrt_llm.logger import logger

            epoch, _, fallback_rows, timeouts, overflowed = self.stats()
            logger.info(
                f"[dsa key shard] rank {self.rank}: {epoch} merges, {fallback_rows} fallback rows, "
                f"{timeouts} spin timeouts, {overflowed} fallback rows overflowed the candidate list"
            )


_KEY_SHARD_MERGES: dict = {}


def get_key_shard_merge(mapping, rows_max: int, top_k: int, local_width: int) -> KeyShardTopKMerge:
    """The merge workspace over ``mapping``'s TP group: under Helix the repurposed mapping, whose TP group is the CP
    group (collective on first use; call outside CUDA graph capture)."""
    key = (mapping.tp_size, mapping.tp_rank, mapping.pp_rank, rows_max, top_k, local_width)
    merge = _KEY_SHARD_MERGES.get(key)
    if merge is not None:
        return merge
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )

    granularity = 1 << 21
    local_k = key_shard_local_k(mapping.tp_size, top_k)
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
    merge = KeyShardTopKMerge(buffers, mapping.tp_rank, rows_max, top_k, local_width, local_k)
    merge.handle = handle
    merge.comm = comm
    _KEY_SHARD_MERGES[key] = merge
    return merge
