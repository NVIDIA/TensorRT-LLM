# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MLA decode under decode context parallelism (DCP = TP, rank d owns head block d)
without collectives, three kernels per layer. Each rank's KV cache holds 1 / DCP of
every context.

  MlaQuery     FP8 query rows [B, H, 576] (+ the KV insert where this rank owns the
               slot); each CTA multicasts its bytes into every rank's Lamport q
               mailbox [B, DCP*H, 576], then polls its share until all landed
  tokenspeed   all DCP*H heads over the local KV shard -> BF16 rows + LSE (log2)
  DcpValue     re-arms the q mailbox; rows pushed to their head owner's Lamport
               mailbox (peer stores); the owner LSE-merges its H heads over the DCP
               sources, then W_UV and the gate (padding from the global lengths)
"""

import math
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake
from ..common.policy import batch_table
from ..common.primitives import (
    alloc,
    asm,
    at,
    bf16x2,
    bits_float,
    float_bits,
    launch,
    load_f32x4,
    load_volatile,
    load_words,
    mma_into,
    no_neg_zero,
    pack_pair,
    published,
    sigmoid,
    store_relaxed,
    store_shared_words,
    store_words,
    unpack_bf16,
    wait,
    warp_sum,
    zeros,
)
from .qvalid import ARMED

# Rubin-class devices (more SMs than GB300's 152): DcpValue push CTAs per head and
# 16-request tile (more peer-store issue; GB300 keeps >= 4 16 B chunks per thread), and
# the tokenspeed split count of the DCP core at q = 1 (<= 32 splits keep its reducer
# short; one CTA pair per split, one wave). Batches not listed keep the GB300 rules.
GB300_SMS = 152
WIDE_PUSHERS = batch_table("1-8:16")
WIDE_SPLITS = batch_table("1-3:32,5-6:16")
# mla_core="tokenspeed_rubin" on sm_107 devices (DCP, q = 1): batches that run
# tokenspeed's Rubin decode kernel (``rubin_decode``) with (KV splits, cluster shape:
# "preferred" 4-CTA clusters, "fallback" 2-CTA clusters, "auto" 4-CTA with 2-CTA
# fallback). Others keep the generic tokenspeed kernel (stock_decode). B8, 96 heads,
# page 32, 1 GPU after a 48-CTA predecessor, us at local 512 / 2048 / 8192 tokens:
# (4, preferred) 9.5 / 11.0 / 18.6; (3, fallback) 8.0 / 10.7 / 20.2; tokenspeed's own
# policy (5, auto) 10.4 / 12.1 / 19.4; the generic kernel (13 splits) 7.6 / 9.3 / 14.0.
RUBIN_TS = {8: (4, "preferred")}
# Rubin-class devices, DCP q = 1, the generic tokenspeed core (mla_core="tokenspeed"):
# batches whose tokenspeed split partials (FP32 O / l and base-2 LSE per split) are
# LSE-merged by DcpValue's pushers (``lse_split``) instead of tokenspeed's reducer
# kernel; value: the exact split count. B8, 2-rank chain (12 heads), us per layer at
# local 550 / 4096 / 8192: reducer (13 splits) 28.63 / 32.70 / 34.50; folded 13
# splits 28.17 / 31.85 / 34.10, 8 splits 27.65 / 30.91 / 33.85, 6: 27.56 / 31.09 /
# 34.57. With the polled fork and FORK_BALANCE, 16 ranks, B1 rows chain (in-proj ->
# MlaQuery -> fork -> DcpValue), us/layer at ISL 8.8k / 64k: 8 splits 21.22 / 24.48,
# 7: 20.83 / 23.92, 6: 20.50 / 24.02, 5: 20.83 / 24.54, 4: 20.77 / 24.91.
LSE_FOLD = {8: 6}
# Rubin-class devices, DCP (AgentX long-context decode): (token rows, gathered heads) of
# the split cores (the B1 rows fork, LSE_FOLD; the packed verify core,
# VERIFY_FOLD_SPLITS) that launch LONG_SPLIT[...][1] KV splits and partition each
# request by its own local length: a request of at most LONG_SPLIT[...][2] 128-token
# tiles keeps the partition of the short split LONG_SPLIT[...][0] (the split count
# these rows ran before: the same partials, and DcpValue's fold only adds weight-0
# slots, so the merged rows are bitwise the same); a longer one spreads over every
# launched split (its FP32 LSE merge sums in another order). One kernel and graph; the
# empty splits' CTAs exit after the prologue. 1 GPU, the core alone (q8, 96 heads, cold
# KV), us at local 2k / 8k / 32k / 62.5k tokens (context 32k / 128k / 512k / 1M): B1
# (packed) S4 6.7 / 13.4 / 39.5 / 70.9, S8 6.4 / 9.6 / 23.5 / 39.3, S16 6.6 / 8.8 /
# 17.1 / 25.0; B2 S4 7.3 / 14.4 / 40.8 / 72.2, S8 7.5 / 11.7 / 25.6 / 41.7. The rows
# fork runs 8 rows x splits CTA pairs: 13 splits = 104 pairs, one wave. (8, 64): the
# drafter's B1 rows (draft.attention ROWS_MODE 7 -> 8 on the same fork, 64 gathered heads).
LONG_SPLIT = {(8, 96): (6, 13, 64), (8, 64): (6, 13, 64), (16, 96): (4, 8, 64)}
# Mixed-context batches (B8 and the wide rows): a request past 64 local tiles
# spreads over more splits (as in an earlier prototype, plus a quantum at 72-128
# rows), and their persistent core places the items by the per-step work list
# (mla.wlist), so the longest requests' items no longer set the makespan. 1 GPU, the
# core alone (q8, 96 heads, AgentX mixes; same node per row), us, today -> the
# prototype's static order -> this: mix8 89.4 (S2) -> 64.9 -> 41.8 (64: 2 / 8, bitwise
# to the prototype's partition); mix14 -> 94.7 -> 62.7, mix16 170.0 (S1) -> 114.6 ->
# 68.4 (2 / 8, quantum 48; 72-128 run 1 / 8, quantum 48: verify_fold_splits switches to
# the long split only from today's split); mix24 -> 121.7 -> 92.2, mix32 203.6 (S1) ->
# 151.6 -> 120.3 (136-256: 1 / 4). Uniform local 2k (vs the prototype): B8 14.7 ->
# 14.9, B16 24.6 -> 25.4, B24 30.9 -> 31.2, B32 34.1 -> 33.9.
LONG_SPLIT.update({(64, 96): (2, 8, 64)})
# B4 (32 rows, today S4): mix4 36.3 -> 26.3 us (4 / 8 + the list); uniform local 2k 9.8
# -> 10.6. B2 (16 rows) keeps its earlier (4, 8, 64) rule without the list (mix2: no
# gain).
LONG_SPLIT.update({(32, 96): (4, 8, 64)})
# B3 / B5 / B6 / B7 (24 / 40 / 48 / 56 rows; today the stock core, its split
# from tokenspeed) join the polled packed core and the work list (as in a later
# prototype: the short split is today's at max_model_len 1M). 1 GPU, the core alone,
# static order, mixed batch us per layer: B3 26.3 -> 24.2, B5 48.2 -> 35.3, B6 77.4
# -> 42.5, B7 83.6 -> 47.9; uniform 32k -0.1 / +0.8 / -0.3 / +1.2.
LONG_SPLIT.update({(24, 96): (5, 8, 64), (40, 96): (3, 8, 64)})
LONG_SPLIT.update({(48, 96): (2, 8, 64), (56, 96): (2, 8, 64)})
# All-long uniform batches pay for the extra splits (1 GPU, local 22.5k = 360k, us,
# the earlier rule -> this): B4 34.0 -> 38.5, B8 59.4 -> 71.8, B16 109.8 -> 116.7,
# B32 213.1 -> 227.5; the split steps at 72-256 rows trade mix16 66.6 -> 68.6 for u16 126.0
# -> 116.7, and win both at 136-256 (mix32 121.9 -> 115.9, u32 239.1 -> 227.5). P(all of
# 8 past 131k) ~0.3%.
LONG_SPLIT.update({(r, 96): (1, 4, 64, 96) for r in range(72, 129, 8)})
# B9-B13 (72-104 rows, tokenspeed's split 1) take B8's rule instead: short split 2
# (VERIFY_FOLD_SPLITS) and 8 long splits, no quantum. 1 GPU, the polled core with the
# work list alone, us per MLA layer, today -> this, AgentX mixes: B9 57.3 -> 42.0, B10
# 61.6 -> 48.4, B12 57.9 -> 54.6, B13 61.6 -> 58.6 (B14 58.1 -> 62.9: kept); uniform
# 360k B9 109.9 -> 84.5; uniform 32k B9 16.5 -> 18.5 (all 9 below 131k: ~0.3%).
LONG_SPLIT.update({(r, 96): (2, 8, 64) for r in range(72, 105, 8)})
LONG_SPLIT.update({(r, 96): (1, 4, 64, 128) for r in range(136, 257, 8)})
# DL3's verify (q4) at odd B: rows 4 mod 8 had no rule (tokenspeed's static split 1:
# one CTA pair per request's whole KV). 1 GPU, the core alone, AgentX draws, us per
# MLA layer, today -> B8's (2, 8, 64) at 68-124 (B17-B31): B17 112 -> 53, B19 199 -> 54,
# B25 219 -> 64, B31 218 -> 61; 132-252 (B33-B63) take their even neighbours' rule: B33
# 191 -> 86 (R2.8 79, R2.4.96 74).
LONG_SPLIT.update({(r, 96): (2, 8, 64) for r in range(68, 125, 8)})
LONG_SPLIT.update({(r, 96): (1, 4, 64, 128) for r in range(132, 253, 8)})
# The drafter's mixed-context cores (DL7: 7 rows per request, 64 gathered
# heads, non-causal) on the same work list (draft.attention DRAFT_WLIST). 1 GPU, the
# core alone, us, today -> this: mix8 62.8 (S3) -> 32.5 (56: 2 / 8 measured; 3 / 8 keeps
# today's short partition), mix16 201.1 (S1) -> 52.5 (63-112: 1 / 8), mix24 209.6 ->
# 65.6, mix32 223.8 -> 86.3 (119-224: 1 / 8, quantum 48). Uniform local 2k: B8 11.1 ->
# 12.9, B16 17.9 -> 20.0, B24 17.1 -> 19.6, B32 28.1 -> 30.7.
LONG_SPLIT.update({(56, 64): (3, 8, 64)})
# The drafter's B6 / B7 (42 / 49 rows, tokenspeed's split 4 / 3) as B8: their long split
# and the work list (DRAFT_WLIST). 1 GPU, the core alone, AgentX mixes, us per layer:
# B6 41.6 (S4) -> 26.2, B7 57.6 (S3) -> 28.1; short splits 2-4 within 0.3 us.
LONG_SPLIT.update({(42, 64): (4, 8, 64), (49, 64): (3, 8, 64)})
# B4 / B5 (28 / 35 rows, split 6 / 5) the same, on the validity-polled core
# (VALID_CORE): B4 with short split 2 (VERIFY_FOLD_SPLITS), B5 keeps its 5. 1 GPU, the
# core alone, us per layer, mean delta over 3 i.i.d. AgentX draws [min, max]: B4 -4.2
# [-6.5, -2.0] (short 6: -1.0 [-3.5, +1.2]), B5 -5.0 [-11.6, -1.3].
LONG_SPLIT.update({(28, 64): (2, 8, 64), (35, 64): (5, 8, 64)})
# The rule fires only from today's split (verify_fold_splits), and on Rubin tokenspeed
# gives B9-B13 (63-91 rows) split 2, not 1 (sms // requests // 8): they kept a static
# 2-way split with no work list, so the longest request set the core's makespan
# (in serving, slower at B9-B13 than at B8 and B14). Their rule starts at 2.
LONG_SPLIT.update({(r, 64): (2, 8, 64) for r in range(63, 92, 7)})
LONG_SPLIT.update({(r, 64): (1, 8, 64) for r in range(98, 113, 7)})
LONG_SPLIT.update({(r, 64): (1, 8, 64, 48) for r in range(119, 225, 7)})
# DL3 (3 drafter rows per request) owns B33-B64: its 99-192 rows had no rule where they
# miss the DL7 keys above, so tokenspeed's split 1 put each request's whole KV on one
# CTA pair. 1 GPU, the core alone, AgentX draws, us per layer, today -> (1, 8, 64) +
# the work list: B33 190 -> 57, B36 228 -> 65, B40 203 -> 62, B48 162 -> 67, B64 252 ->
# 90 (short splits 1-3 within 2 us; B56's 168 rows keep their DL7 rule: flat).
LONG_SPLIT.update(
    {(r, 64): (1, 8, 64) for r in range(99, 193, 3) if (r, 64) not in LONG_SPLIT}
)
# DL3 at max-num-seqs 192: DL3's verify rows 260-768 (q4, B65-B192) and drafter
# rows 195-576 (q3, 64 heads) had no rule: tokenspeed's split 1, persistent, so the
# longest request's whole KV on one CTA pair set the makespan (AgentX contexts reach
# ~840k). Rule (1, 4, 64, 128) + the work list (WLIST, DRAFT_WLIST): a request past 64
# local tiles takes min(4, ceil(tiles / 128)) splits. 1 GPU, the core alone, AgentX D
# mixes (mean 188k), us per layer, today -> this: verify B96 300.7 -> 168.9, B128 338.6
# -> 219.2, B192 435.3 -> 324.6; drafter B96 260.7 -> 113.8, B128 286.5 -> 157.5,
# B192 338.2 -> 219.9.
# Split 1 on the work list (no long splits) stays bound by the longest item (B96 259.5,
# B192 359.3); (1, 2, 64, 128) is within 2% of this at B96-B192.
LONG_SPLIT.update({(r, 96): (1, 4, 64, 128) for r in range(260, 769, 4)})
# B129-B192 (516-768 rows): 2 long splits (fewer FP32 partials for DcpValue's fold).
# 1 GPU, us per layer, (1, 4, 64, 128) -> this: mix192 324.6 -> 319.6, two i.i.d.
# AgentX draws at B192 311.9 / 329.4 -> 305.8 / 323.6, mix160 271.5 -> 265.9; at B96-B128
# within +-2% either way (B48-B64: +12..+25%, so not below B129).
LONG_SPLIT.update({(r, 96): (1, 2, 64, 128) for r in range(516, 769, 4)})
LONG_SPLIT.update(
    {(r, 64): (1, 4, 64, 128) for r in range(195, 577, 3) if (r, 64) not in LONG_SPLIT}
)


def long_split(rows, heads, split, sms):
    """(short split, long tiles, quantum) of LONG_SPLIT's length-keyed partition when
    ``rows`` of ``heads`` gathered heads launch its long split on a Rubin-class device
    (``sms``), else None. An optional 4th LONG_SPLIT field, the quantum q (tiles): a
    long request of T tiles runs min(launched, ceil(T / q)) equal splits instead of
    all launched ones (``long_per``); 0: every launched split."""
    rule = LONG_SPLIT.get((rows, heads))
    if sms <= GB300_SMS or rule is None or split != rule[1]:
        return None
    return rule[0], rule[2], rule[3] if len(rule) > 3 else 0


@cute.jit
def long_per(tiles, split, long):
    """Tiles per split of a request of ``tiles`` K tiles under ``long``
    (``long_split``) launched at ``split``: the short split's partition up to the long
    tiles, past them ``split`` (quantum 0) or min(split, ceil(tiles / quantum)) equal
    splits. The core's partition and DcpValue's fold both read this."""
    short = max((tiles + long[0] - 1) // long[0], 1)
    splits = split
    if cutlass.const_expr(long[2] > 0):
        splits = min(split, (tiles + long[2] - 1) // long[2])
    per = max((tiles + splits - 1) // splits, 1)
    return Int32(cutlass.select_(tiles > long[1], per, short))


@cute.jit
def filled_splits(n, split, long):
    """KV splits a core fills for a request of ``n`` local tokens (128-token tiles)
    launched at ``split`` (``long``: ``long_split``'s rule, else tokenspeed's equal
    partition): contiguous from split 0, every one full but the last. The cores that
    write the split partials and DcpValue's fold read this (and nothing past it: the
    other launched splits are never written)."""
    tiles = (n + 127) // 128
    per = max((tiles + split - 1) // split, 1)
    if cutlass.const_expr(long is not None):
        per = long_per(tiles, split, long)
    return (tiles + per - 1) // per


# Lockstep page-range chunking (LPC; DS-T2P's partition behind DS_LPC): the batch's
# requests, laid end to end in request order as their KV pages plus LPC_OVER pages of
# per-chunk cost (an item's fixed cost: Q intake, FP32 partial stores; after the pages),
# are cut into ``groups`` equal ranges of T; a range's overlap with a request's pages is
# a chunk, and chunk k of a request (its index within the request) writes partial slot
# k. A group of ``items`` clusters runs
# its chunks in order, one item stream (q tile pair x value half) each, in lockstep, so
# the streams of a page share it in L2. T >= ceil(max pages / (slots - 1)) keeps every
# request within ``slots`` chunks. A pure function of the lengths: the core and
# DcpValue's fold both derive it (lpc_table on device, lpc_host for the gates).


# a chunk's cost in page units: max(pages, LPC_MIN) + LPC_OVER (1 GPU, AgentX mix8-32:
# 16 / 0 best of 13-28 / 0-20)
LPC_OVER = 16
LPC_MIN = 0


def lpc_host(lengths, groups, slots, page=64, over=LPC_OVER, least=LPC_MIN):
    """Host twin of ``lpc_table``: (C, T, nf) -- the exclusive prefix C[0..R] of the
    requests' units (max(pages, ``least``) + ``over``, pages first; 0 for an empty
    request), the units T per group range, and each request's chunk count nf[r] (its
    filled slots 0 .. nf[r] - 1)."""
    pages = [-(-int(n) // page) for n in lengths]
    prefix = [0]
    for p in pages:
        prefix.append(prefix[-1] + (max(p, least) + over if p else 0))
    total = prefix[-1]
    widest = -(-max(pages, default=0) // (slots - 1))
    per = max(-(-total // groups), widest, 1)
    counts = [
        (prefix[r] + pages[r] - 1) // per - prefix[r] // per + 1 if pages[r] else 0
        for r in range(len(pages))
    ]
    return prefix, per, counts


def lpc_chunks_host(lengths, groups, slots, page=64, over=LPC_OVER, least=LPC_MIN):
    """Every chunk as (group, request, slot, first page, pages), in group order."""
    prefix, per, _ = lpc_host(lengths, groups, slots, page, over, least)
    pages = [-(-int(n) // page) for n in lengths]
    chunks = []
    for g in range(groups):
        lo, hi = g * per, (g + 1) * per
        for r in range(len(lengths)):
            a, b = max(lo, prefix[r]), min(hi, prefix[r] + pages[r])
            if b > a:
                chunks.append((g, r, g - prefix[r] // per, a - prefix[r], b - a))
    return chunks


@asm(Int32)
def _ld_shared_s32(address):
    return "ld.shared.s32 $0, [$1];", "=r,r,~{memory}", (Uint32(address),)


@asm()
def _st_shared_s32(address, value):
    return "st.shared.s32 [$0], $1;", "r,r,~{memory}", (Uint32(address), Int32(value))


@cute.jit
def lpc_table(
    lengths,
    reqs: cutlass.Constexpr,
    groups: cutlass.Constexpr,
    slots: cutlass.Constexpr,
    table,
    lane,
    page: cutlass.Constexpr = 64,
    over: cutlass.Constexpr = LPC_OVER,
    least: cutlass.Constexpr = LPC_MIN,
):
    """One warp: LPC's unit prefix C[0..reqs] (Int32 words at shared ``table``) and T
    (``table`` word reqs + 1) from the requests' local ``lengths`` (reqs <= 64). The
    readers sync with the writing warp before ``lpc_chunk``."""
    assert reqs <= 64, "lpc_table scans up to 64 requests"
    p0 = Int32(0)
    p1 = Int32(0)
    if lane < reqs:
        p0 = (Int32(lengths[lane]) + page - 1) // page
    if lane + 32 < reqs:
        p1 = (Int32(lengths[lane + 32]) + page - 1) // page
    w0 = cutlass.select_(
        p0 > 0, cutlass.max(p0, Int32(least)) + over, Int32(0)
    )  # units
    w1 = cutlass.select_(p1 > 0, cutlass.max(p1, Int32(least)) + over, Int32(0))
    s0, s1 = w0, w1
    for k in cutlass.range_constexpr(5):  # inclusive scans (Hillis-Steele)
        u0 = cute.arch.shuffle_sync_up(s0, offset=1 << k)
        u1 = cute.arch.shuffle_sync_up(s1, offset=1 << k)
        if lane >= (1 << k):
            s0 = s0 + u0
            s1 = s1 + u1
    low = cute.arch.shuffle_sync(s0, 31)
    s1 = s1 + low
    total = cute.arch.shuffle_sync(s1, 31)
    widest = cutlass.max(p0, p1)
    for k in cutlass.range_constexpr(5):
        widest = cutlass.max(widest, cute.arch.shuffle_sync_bfly(widest, offset=1 << k))
    per = cutlass.max(
        (total + groups - 1) // groups, (widest + slots - 2) // (slots - 1)
    )
    per = cutlass.max(per, Int32(1))
    if lane < reqs:
        _st_shared_s32(table + 4 * lane, s0 - w0)
    if lane + 32 < reqs:
        _st_shared_s32(table + 4 * (lane + 32), s1 - w1)
    if lane == 0:
        _st_shared_s32(table + 4 * reqs, total)
        _st_shared_s32(table + 4 * (reqs + 1), per)


@cute.jit
def lpc_chunk(
    table, lengths, reqs: cutlass.Constexpr, group, n, page: cutlass.Constexpr = 64
):
    """Chunk n of group ``group``'s range: (request, slot, first page, pages); 0 pages
    past the range's last request or on its fixed-cost units only."""
    per = _ld_shared_s32(table + 4 * (reqs + 1))
    lo = group * per
    first = Int32(0)  # the last request starting at or before the range
    step = 32
    while step > 0:
        if (first + step < reqs) & (_ld_shared_s32(table + 4 * (first + step)) <= lo):
            first = first + step
        step = step // 2
    r = first + n
    slot = Int32(0)
    page0 = Int32(0)
    pages = Int32(0)
    if r < reqs:
        start = _ld_shared_s32(table + 4 * r)
        a = cutlass.max(lo, start)
        b = cutlass.min(lo + per, start + (Int32(lengths[r]) + page - 1) // page)
        if b > a:
            slot = group - start // per
            page0 = a - start
            pages = b - a
    return r, slot, page0, pages


@cute.jit
def lpc_filled(table, n, r, reqs: cutlass.Constexpr, page: cutlass.Constexpr = 64):
    """Request r's LPC chunk count from ``lpc_table``'s words: its filled partial
    slots 0 .. nf - 1 (``lpc_host``'s nf; 0 for an empty request)."""
    start = _ld_shared_s32(table + 4 * r)
    per = _ld_shared_s32(table + 4 * (reqs + 1))
    pages = (Int32(n) + page - 1) // page
    nf = (start + pages - 1) // per - start // per + 1
    return cutlass.select_(pages > 0, nf, Int32(0))


# The DL7 verify rows (q 8) whose DS-T2P core runs LPC instead of the LONG_SPLIT work
# list (VLLM_KIMI_K3_DS_LPC), at LPC_SLOTS partial slots per request. 16 ranks, whole
# step, LPC vs the list (before the prologue-after-wait fix, ~1 us per MLA layer): only
# B8 (rows 64) wins clearly (mix8); mix16 / mix24 and uniform 32k B8 / B16 / B24 change
# little. 112-192 keep DS-T2P's list.
LPC_ROWS = batch_table("64:1")
LPC_SLOTS = 8
LPC_LONG = (1, 64, 0)  # (P2Core's ``long`` under LPC: unused, its compile key only)


# LSE_FOLD batches whose tokenspeed core is the polled fork (mla.core_ts): MlaQuery
# does not gather and polls x, the core polls q and re-arms x (``fork_partials``).
POLLED_TS = batch_table("8:1")
# POLLED_TS batches whose fork splits the heads evenly over its 2-CTA pairs (mla.core_ts
# ``balanced``: 48 / 48 instead of 64 / 32 at 96 heads), so no CTA stores more than
# half of a tile's FP32 split partials (the epilogue is SM-store-bandwidth bound). 16
# ranks, B1 rows chain (in-proj -> MlaQuery -> fork -> DcpValue), us/layer at ISL 8.8k
# / 64k: 21.68 -> 21.17 / 24.18 -> 23.91, bitwise the same output.
FORK_BALANCE = batch_table("8:1")
# POLLED_TS batches whose fork's q pollers re-poll all their late chunks together (one
# round trip per pass) instead of spinning on each chunk in turn (mla.core_ts
# ``repoll``); the same reads. 16 ranks, B1 rows chain, us/layer at ISL 8.8k / 64k:
# 20.52 -> 20.20 / 23.61 -> 23.50, bitwise the same output.
FORK_REPOLL = batch_table("8:1")
# POLLED_TS batches whose fork launches clusters of its 2-CTA pairs x all KV splits of
# a request (mla.core_ts ``cluster``: (2, 1, splits), one 12-CTA cluster per request at
# 6 splits); the same kernel and results otherwise. 16 ranks, B1 rows chain, us/layer
# at ISL 8.8k / 64k, three items: 20.18 -> 19.94, 20.16 -> 19.98, 20.20 -> 19.97 /
# 23.22 -> 23.30, 23.22 -> 23.39, 23.26 -> 23.32 (-0.22 / +0.10), bitwise the same.
# Off: with the attention-residual front ahead of the in-projection holding SMs (the
# serving shape of the layer), the cluster launch delays the fork: 16 ranks, us/layer
# at ISL 8.8k / 64k, without -> with: 27.03 -> 27.13 / 30.74 -> 31.90 (LSU q poll);
# 26.62 -> 26.79 / 30.39 -> 31.15 (FORK_VALID 2). The rows chain alone (nothing else
# resident) over-rewards cluster launch shapes.
FORK_CLUSTER = batch_table("")
# POLLED_TS batches whose fork polls q with validity-reporting copies (mla.core_ts
# ``valid``, mla.qvalid): no LSU pollers; the V-load warp copies the CTA's q rows until
# no 16 B chunk is armed, then the Q TMA. MlaQuery writes these batches' q rows as one
# 16 B store per chunk (``MlaQuery(valid=True)``); the mailbox is armed with
# (0x80000000, Q_SENT x 3) chunks on Rubin. Bitwise the same output. 16 ranks, B1 rows
# chain (FORK_CLUSTER), us/layer at ISL 8.8k / 64k: LSU poll 19.97 / 23.33, 1: 19.69 /
# 23.11, 2: 19.47 / 22.80. With the attention-residual front and no FORK_CLUSTER: LSU
# poll 27.08 / 30.84, 2: 26.62 / 30.39.
FORK_VALID = batch_table("8:2")
# FORK_VALID values: 1 = the V-load warp's validity copies into the Q operand's shared
# memory, then the Q TMA; 2 = the Q load itself is the validity copy (CTA-local tensor
# TMAs on grid-constant descriptors of the q mailbox, the pair leader's Q barrier gets
# one 4 B signal per CTA).
# FORK_VALID batches whose validity reader spins this many ns after its PDL wait before
# its first round (readers that poll while the q rows land slow their landing).
FORK_VALID_DELAY: dict = {}
# Rubin-class devices: batches whose DcpValue merges one (head, request) row per CTA
# (push CTA z of a head merges request z, so the 16-source sweep runs on B x H SMs
# instead of H), with the LSE box laid out [CAP, H, DCP] (the 16 source LSEs of a row
# share one line) and the LSEs pushed before the rows, so they gate the sweep. The
# push CTAs run in clusters of 8 along z: the merge cluster all-gathers its merged rows
# through DSMEM and CTA z projects W_UV output tiles 2z, 2z + 1 for every request.
WIDE_MERGE = batch_table("8:1")
# Rubin-class devices, spec verify (q > 1) token-row batches whose merge is spread over
# every CTA: CTA z of a (head, 16-row tile) merges rows R z .. R z + R - 1 (R = 16 /
# pushers) and projects them through W_UV itself (warp w: output tile w), instead of
# one merge CTA per (head, tile) sweeping 16 sources x 16 rows. The LSE box is laid
# out as the wide merge's (the sources of a row share one line, pushed first). TP16,
# 16 ranks, B x q8 in-proj -> DcpValue at 8.8k, us/MLA layer per-head -> spread: B2
# 34.72 -> 30.48, B3 39.75 -> 35.92, B4 43.09 -> 38.85, B5 45.65 -> 43.96, B6 47.95
# -> 45.83, B7 51.27 -> 48.90, B8 45.49 -> 39.80 (64k 55.69 -> 49.93). Rows 14, 21, 28,
# 35: the DSpark drafter's B2-B5 query rows (q = 7, folded splits); 16 ranks, the
# drafter forward, per-head -> spread: faster at B2 and B4; final rows bitwise
# identical. Row 56 is also the drafter's B8 (a faster forward with spread, 16 ranks).
# Rows 42, 49, 63: the drafter's B6 / B7 / B9 (local 2k, a faster forward with spread at
# each; final rows bitwise identical).
# Above 64 rows (the wide verify shapes on DcpBox.widened) every batch spreads.
SPREAD_MERGE = batch_table(
    "14:1,16:1,21:1,24:1,28:1,32:1,35:1,40:1,42:1,48:1,49:1,56:1,63:1,64:1"
)
# Rubin-class devices, wide merge: batches whose output rows are Lamport-polled by the
# o-projection (``OprojFused(polled_x=True)``) instead of read after its PDL wait. The
# output is one of two per-parity buffers armed with PAIR_SENT words (no negative zero
# halves are written, so no output word equals it); each layer's DcpValue re-arms the
# other parity's buffer, whose last reader (the previous MLA layer's o-projection) has
# completed by then (every kernel between them runs griddepcontrol.wait). 4 ranks, 1-GPU
# chain lead -> DcpValue -> o-proj: 11.28 -> 11.10-11.13 us.
POLLED_OUTPUT = batch_table("8:1")
# Wide merge: source rows polled per pass by each merging thread; the late ones of a
# pass are re-polled together (one round trip per pass, not one per late source). 4
# ranks, lead -> DcpValue -> polled o-proj, us: one-by-one
# spinning, 8 per pass 11.14 (16: 11.83, 4: 10.85); re-polled together, 8 per pass
# 10.77, 12 per pass 10.57, 6 per pass 10.61, 16 per pass 11.42.
WIDE_ROW_POLLS = 12
# Rubin, wide merge: batches whose merge warp polls its source LSEs in one warp-uniform
# loop (a vote ends it; each pass re-polls only the lanes whose LSE is still the
# sentinel) instead of each lane spinning on its own source. With per-lane spins the
# warp reconverges for the max / sum shuffles only after the last lane, ~1500 cycles
# after it (4 ranks, one rank 3 us late; 325 without divergence). The same loads,
# re-arms and values. 16 ranks, B1 rows, us/layer: rows chain 8.8k 19.53 -> 19.46;
# with the attention-residual front 8.8k 26.45 -> 26.19, 64k 30.03 -> 29.60.
DV_LSE_UNIFORM = batch_table("8:1")
POLL_LSE = True  # False only in the tests' positive control (the LSEs read once)
SPLIT_SENT = 0xFFFFFFFF  # DcpCoreUmma's armed split workspace (mla.core_umma)
# Polled split partials: the pushers poll the row's (max, sum) words (written first by
# the core) until all are published before loading its 5 x 32 B partials, so a pass
# of 80 KB per CTA is not spent on partials that have not landed yet.
SPLIT_GATE = 1
LSE_SENT = 0xFFFFFFFF  # a NaN pattern the merge never sends
# The LSE of an empty or invalid source: finite, so the merge never forms inf - inf.
EMPTY_LSE = -3.0e38
Q_SENT = 0xFFFFFFFF  # four FP8 0xFF bytes: never written (satfinite, canonical NaN)
PAIR_SENT = 0x80000000  # BF16 pair (+0, -0); sent pairs have no negative zero


@asm()
def load_f32x8(pointer):
    """8 FP32 values as one 32 B (256-bit) load (sm_100+)."""
    registers = ",".join(f"${i}" for i in range(8))
    code = "ld.global.cg.v8.f32 {" + registers + "}, [$8];"
    return code, ",".join(["=f"] * 8 + ["l"]), (pointer,), (Float32,) * 8


@asm()
def load_f32x8_if(pointer, live):
    """``load_f32x8`` when ``live`` != 0, else 8 zeros: a predicated load, no branch,
    so a caller's loads of several slots stay in flight together."""
    registers = ",".join(f"${i}" for i in range(8))
    zero = " ".join(f"mov.f32 ${i}, 0f00000000;" for i in range(8))
    code = "{ .reg .pred p; setp.ne.s32 p, $9, 0; " + zero
    code += " @p ld.global.cg.v8.f32 {" + registers + "}, [$8]; }"
    return code, ",".join(["=f"] * 8 + ["l", "r"]), (pointer, live), (Float32,) * 8


@asm()
def load_f32_if(pointer, live, default):
    """One FP32 word when ``live`` != 0, else ``default`` (a predicated load), with the
    default cache policy (L1 allocating): the 64 chunk threads of a row re-read its
    few split LSE words, as the reference's tensor loads do; ``.cg`` sent each to L2
    (the filled fold measured +1.3..1.8% at B1 with it, 16-rank ABBA)."""
    code = "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.f32 $0, $3; "
    code += "@p ld.global.f32 $0, [$1]; }"
    return code, "=f,l,r,f", (pointer, live, default), (Float32,)


@asm()
def poll_b32x8(pointer):
    """8 words as one 32 B volatile load (a polled split partial)."""
    registers = ",".join(f"${i}" for i in range(8))
    code = "ld.volatile.global.v8.b32 {" + registers + "}, [$8];"
    return code, ",".join(["=r"] * 8 + ["l", "~{memory}"]), (pointer,), (Uint32,) * 8


@asm()
def push_words(pointer, words, barrier, peer):
    """DSMEM store of 4 b32 ``words`` into cluster CTA ``peer``'s shared ``pointer``,
    completing on that CTA's ``barrier``."""
    code = "{ .reg .b32 dst,bar; mapa.shared::cluster.u32 dst,$0,$6; "
    code += "mapa.shared::cluster.u32 bar,$5,$6; "
    code += "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
    code += "[dst],{$1,$2,$3,$4},[bar]; }"
    operands = (pointer, *(Uint32(w) for w in words), barrier, Int32(peer))
    return code, "r,r,r,r,r,r,r,~{memory}", operands


class DcpBox:
    """Model-owned DCP state over ``group``, Lamport mailboxes (sentinel-filled,
    re-armed by their reader): q [2, CAP, DCP*H, 576] FP8 bytes (multicast stores),
    partials [2, DCP, CAP, H, 512] BF16 and LSE [2, DCP, CAP, H] FP32 (peer stores),
    and the local tokenspeed rows / LSE. Parity alternates by layer. ``interleave``:
    the KV cache's DCP interleave (global masking of spec verify rows)."""

    def __init__(self, group, device, size, rank, heads, cap=64, interleave=1):
        self.size, self.rank, self.heads, self.cap = size, rank, heads, cap
        self.interleave, self.group = interleave, group
        self.wide: dict[int, DcpBox] = {}  # ``widened``
        empty = symm_mem.empty
        self.q = empty((2, cap, size * heads, 576), dtype=torch.uint8, device=device)
        shape = (2, size, cap, heads)
        self.partial = empty((*shape, 512), dtype=torch.bfloat16, device=device)
        self.lse = empty(shape, dtype=torch.float32, device=device)
        tensors = [self.q, self.partial, self.lse]
        # Rubin: a second q mailbox for validity-polled readers (mla.qvalid), armed with
        # (0x80000000, Q_SENT x 3) chunks. Kept apart from ``q`` (armed with Q_SENT
        # words, LSU-polled): an LSU reader of a chunk written by separate 4 B stores
        # could see words 1-3 landed beside an unwritten 0x80000000 word 0 and pass,
        # and a validity reader sees no 0x80000000 in a Q_SENT chunk. Each (batch, q)
        # shape uses one box for its whole life (PreparedMla.valid_box).
        rubin = torch.cuda.get_device_properties(device).multi_processor_count
        self.qv = None
        if rubin > GB300_SMS:
            self.qv = empty(self.q.shape, dtype=torch.uint8, device=device)
            tensors.append(self.qv)
        handles = [symm_mem.rendezvous(t, group) for t in tensors]
        self.q_mc = int(handles[0].multicast_ptr or 0)
        self.qv_mc = int(handles[3].multicast_ptr or 0) if self.qv is not None else 0
        if not self.q_mc or (self.qv is not None and not self.qv_mc):
            raise RuntimeError("Kimi K3 specialized MLA DCP needs multicast memory")
        i64 = dict(dtype=torch.int64, device=device)
        self.peers = torch.tensor(handles[1].buffer_ptrs, **i64)
        self.lse_peers = torch.tensor(handles[2].buffer_ptrs, **i64)
        # (host copies: core_q.push_table)
        self.peer_ptrs = list(handles[1].buffer_ptrs)
        self.lse_peer_ptrs = list(handles[2].buffer_ptrs)
        self.q.fill_(0xFF)
        if self.qv is not None:
            chunks = self.qv.view(torch.int32).view(-1, 4)
            chunks[:, 0].fill_(-0x80000000)
            chunks[:, 1:].fill_(-1)
        self.partial.view(torch.int32).fill_(-0x80000000)
        self.lse.view(torch.int32).fill_(-1)
        rows = (cap, 1, size * heads)
        self.rows = torch.empty((*rows, 512), dtype=torch.bfloat16, device=device)
        self.rows_lse = torch.empty(rows, dtype=torch.float32, device=device)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)

    def widened(self, cap):
        """The model's mailboxes for verify shapes above this box's rows: one more box
        set of ``cap`` rows over the same group (every MLA layer shares it; created
        collectively, at the first layer that prepares such a shape)."""
        if cap not in self.wide:
            device = self.q.device
            args = self.group, device, self.size, self.rank, self.heads, cap
            self.wide[cap] = DcpBox(*args, interleave=self.interleave)
        return self.wide[cap]

    def mc(self, parity, valid=False):
        """The q mailbox (``valid``: the validity-polled one) multicast address of
        ``parity``."""
        base = self.qv_mc if valid else self.q_mc
        return Int64(base + parity * self.q[0].numel())

    def mailbox(self, valid=False):
        """The q mailboxes [2, CAP, DCP*H, 576] (``valid``: the validity-polled
        ones)."""
        return self.qv if valid else self.q


class DcpValue:
    """The DCP tail of one MLA layer, one 512-thread CTA per (head, 16 requests).
    Every rank re-arms this parity's q mailbox (tokenspeed has read it) and pushes all
    its local tokenspeed rows (+ LSE) to their head owner's Lamport mailbox (empty
    local shards: zeros, LSE -inf). Warp r then LSE-weights request t0 + r of this
    head over the DCP sources (exp2(lse - max) / sum in FP32), the CTA merges the
    polled rows in FP32
    (one BF16 rounding, re-arming what it read) into shared memory, and W_UV (BF16
    MMA, FP32 accumulation) and the sigmoid gate follow as in MlaValue: warp w owns
    outputs 32 (w // 4) + 8 (w % 4) .. + 7. Rows are tokens, ``q`` per request."""

    def __init__(
        self,
        heads,
        size,
        batch,
        sms=148,
        q=1,
        fold=0,
        polled_out=False,
        polled_split=False,
        lse_split=False,
        out_scale=1.0,
        valid_box=False,
        lse_uniform=False,
        pushed=False,
        filled=True,
        lpc=None,
    ):
        self.heads, self.size, self.batch = heads, size, batch
        # LPC (DS-T2P's lockstep page-range chunks, ``lpc`` = (groups, slots)): a
        # request's filled slots are its chunk count, the same pure function of the
        # step's lengths the core used (lpc_table per CTA, lpc_filled); the fold is the
        # filled fold below with that count instead of filled_splits'.
        self.lpc = tuple(lpc) if lpc else None
        # The split fold (``lse_split``) loads and weights only the request's filled
        # splits (``filled_splits``), not every launched one: the fold costs ~0.17 us
        # per launched split per layer (a chain harness), and today's fork fills
        # 6 of its 13 below 64 local tiles. Same FP32 order and values (the skipped
        # slots carried weight 0 and re-read the last filled split): bitwise.
        # filled=False: every launched slot (the reference).
        self.filled = filled
        assert self.lpc is None or (lse_split and q > 1 and fold == self.lpc[1])
        assert self.lpc is None or not polled_split  # (its wait follows the pushes)
        # The core pushed every row and LSE into the owners' boxes itself (core_q's
        # ``push``, CORE_PUSH): this kernel only re-arms the q mailbox and merges.
        self.pushed = pushed
        # the q mailbox read is DcpBox.qv (re-armed with mla.qvalid.ARMED chunks)
        self.valid_box = valid_box
        self.q = q
        # tokenspeed's split partials (``split_partials``): normalized O per split and
        # one base-2 LSE per split ([B, DCP*H, fold]); a request of n local tokens
        # filled ceil(T / ceil(T / fold)) splits (T = ceil(n / 128)), the others are
        # unwritten; the merged rows are scaled by ``out_scale`` (tokenspeed's reducer
        # applies the output scale).
        self.lse_split = lse_split and fold > 0
        # LONG_SPLIT: the core partitions each request by its own length (valid_splits)
        self.long = (
            long_split(batch, size * heads, fold, sms) if self.lse_split else None
        )
        self.out_scale = out_scale
        # Rubin DcpCore (mla.core): the pushers LSE-merge its ``fold`` split partials
        # (FP32 O and (max, sum) per split) into each row first.
        self.fold = fold
        # DcpCoreUmma's split workspace is polled (mla.core_umma.POLLED_SPLIT): the
        # pushers start before this kernel's PDL wait.
        self.polled_split = polled_split and fold > 0
        # Push-only CTAs (z > 0) spread the NVLink stores over more SMs: up to one
        # wave, at least 4 16 B chunks per thread.
        merge = heads * ((batch + 15) // 16)
        chunks = batch * size * heads * 64
        pushers = min(max(1, sms // merge), -(-chunks // (merge * 512 * 4)))
        if sms > GB300_SMS and batch in WIDE_PUSHERS:
            pushers = min(max(1, sms // merge), WIDE_PUSHERS[batch])
        if self.lse_split and q > 1:  # each pushed chunk folds ``fold`` FP32 partials
            pushers = max(1, sms // merge)
        # Rubin-class devices: only the merge CTAs load W_UV fragments, the push loads
        # its rows unconditionally (empty requests still push zeros: one L2 round trip
        # instead of two) and pushes before re-arming the q mailbox, and the merge
        # reads the output rows' global lengths before its polls. Same arithmetic.
        self.lean = sms > GB300_SMS
        # spec verify rows (SPREAD_MERGE): one wave of CTAs (one per SM: the merges
        # wait for every CTA's pushes), a power of two per (head, tile) up to 16
        self.spread = self.lean and q > 1 and (batch in SPREAD_MERGE or batch > 64)
        if self.spread:
            pushers = 1 << (min(16, max(1, sms // merge)).bit_length() - 1)
        self.merge_rows = 16 // pushers if self.spread else 16
        self.grid = (heads, (batch + 15) // 16, pushers)
        # The spread merges spin until every CTA's grid-stride pushes land, so the grid
        # must be co-resident: one 512-thread CTA per SM (103 registers). Above one wave
        # (6 heads x 36+ row tiles: DL3 B144+) CTA y merges row tiles y, y + G, ..
        # (G row-tile CTAs per head) instead: the same rows, sums and MMAs (bitwise).
        self.tile_stride = 0
        if self.spread and heads * self.grid[1] * pushers > sms:
            self.tile_stride = max(1, sms // (heads * pushers))
            self.grid = (heads, self.tile_stride, pushers)
        # (decode only: spec verify rows, q > 1, keep the per-head merge)
        self.wide = (
            self.lean
            and q == 1
            and batch in WIDE_MERGE
            and batch <= 8
            and pushers % 8 == 0
        )
        # args[14]: the other parity's output buffer (re-armed here)
        self.polled_out = polled_out and self.wide
        self.lse_uniform = bool(lse_uniform) and self.wide  # DV_LSE_UNIFORM

    @cute.jit
    def __call__(self, *args):
        cluster = (1, 1, 8) if self.wide else None
        launch(self.kernel, args, self.grid, 512, cluster=cluster)

    @cute.kernel
    def kernel(self, args):
        rows, rows_lse, lens, box, lbox, peers, lpeers, mailbox = args[0:8]
        w_uv, gate, totals, output, parity, rank = args[8:14]
        H, D, B, cap = self.heads, self.size, self.batch, box.shape[1]
        smem = cutlass.utils.SmemAllocator()
        latent = alloc(smem, BFloat16, (16, 544), 128)
        scales = alloc(smem, Float32, (16, D))
        tid, _, _ = cute.arch.thread_idx()
        head, tile, pusher = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        v = 32 * (warp // 4) + 8 * (warp % 4) + group
        if cutlass.const_expr(self.lpc is not None):
            # this step's LPC prefix and range (lpc_table, as the core): built after the
            # PDL wait below, read by every thread's valid_splits (lpc_filled)
            ltab = smem.allocate_tensor(Int32, cute.make_layout(B // self.q + 2), 16)
            self.lpc_words = Uint32(ltab.iterator.toint())
        bar = None
        if cutlass.const_expr(self.wide):
            # merge cluster CTA z projects output tiles w = 2 z + warp (warps 0, 1)
            bar = smem.allocate(cutlass.Uint64, 8)
            v = 32 * ((2 * pusher + warp) // 4) + 8 * ((2 * pusher + warp) % 4) + group
            if pusher < 8:
                if tid == 0:
                    cute.arch.mbarrier_init(bar, 1)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(bar, B * 1024)
                cute.arch.cluster_arrive_relaxed()
        # B fragments: 8 latent values per 32-wide chunk; A uses the same permutation.
        wb = None
        if cutlass.const_expr(self.wide):
            wb = [(Uint32(0),) * 4 for c in range(16)]
            if (pusher < 8) & (warp < 2):
                wb = [
                    load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
                    for c in range(16)
                ]
        elif cutlass.const_expr(self.spread):
            pass  # every CTA merges: merge_spread loads its W_UV fragments
        elif cutlass.const_expr(self.lean):  # push-only CTAs skip W_UV
            wb = [(Uint32(0),) * 4 for c in range(16)]
            if self.merges(pusher):
                wb = [
                    load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
                    for c in range(16)
                ]
        else:
            wb = [
                load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
                for c in range(16)
            ]
        if cutlass.const_expr(not self.polled_split):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.lpc is not None):
            # (after the wait: the step's lengths are read like the push loop's, never
            # through a line an earlier step may have left in L1; ~1 us: one warp scan
            # and a CTA barrier)
            if warp == 0:
                lpc_table(lens, B // self.q, *self.lpc, self.lpc_words, lane)
            cute.arch.sync_threads()
        threads = H * self.grid[1] * self.grid[2] * 512
        me = ((pusher * self.grid[1] + tile) * H + head) * 512 + tid
        if cutlass.const_expr(not self.lean):
            for c in range(me, B * D * H * 36, threads):  # 16 B q mailbox chunks
                store_words(mailbox.iterator + c * 16, (Uint32(Q_SENT),) * 4)
        if cutlass.const_expr((self.wide or self.spread) and not self.pushed):
            self.push_lse(args, me, threads, cap)  # the LSEs gate the sweep
        for i in range(me, B * D * H * 64 * (not self.pushed), threads):
            r, k = i // 64, i % 64  # r = b * D * H + global head
            b, gh = r // (D * H), r % (D * H)
            words = (Uint32(0),) * 4
            if cutlass.const_expr(self.lse_split):
                fb, fgh = self.fold_row(b, gh)
                loaded = self.folded(rows, rows_lse, fb, fgh, k, lens[b // self.q])
                if lens[b // self.q] > 0:
                    words = loaded
            elif cutlass.const_expr(self.fold > 0):
                loaded = self.folded(rows, rows_lse, b, gh, k)
                if lens[b // self.q] > 0:
                    words = loaded
            elif cutlass.const_expr(self.lean):
                loaded = load_words(at(rows, b, 0, gh, 8 * k), count=4, mutable=True)
                if lens[b // self.q] > 0:
                    words = loaded
            elif lens[b // self.q] > 0:
                words = load_words(at(rows, b, 0, gh, 8 * k), count=4, mutable=True)
            words = tuple(no_neg_zero(w) for w in words)
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            store_relaxed(peers[gh // H] + (Int64(slot) * 512 + 8 * k) * 2, words)
        if cutlass.const_expr(not (self.wide or self.spread or self.pushed)):
            self.push_lse(args, me, threads, cap)
        if cutlass.const_expr(self.polled_split):
            # Polled split partials: the core's grid is waited for only here, before
            # re-arming the q mailbox it read (and before the merge reads the gate).
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.polled_out):
            # Arm / consume / re-arm (POLLED_OUTPUT): the other parity's output
            # buffer is armed here, after this kernel's PDL wait. Its last reader,
            # the previous MLA layer's o-projection, has completed: it ends with
            # griddepcontrol.wait and every kernel between them waits before it
            # completes. Its next writer, the next MLA layer's DcpValue, stores only
            # after its own PDL wait, so after this grid has completed; its next
            # reader, that layer's o-projection, is launched at that DcpValue's
            # trigger, after its wait. This layer's buffer is consumed only by this
            # layer's o-projection, which never writes it. Allocation arms both
            # buffers once; the MLA layer count is even (providers.prepare_layout).
            for c in range(me, B * H * 16, threads):  # 16 B chunks
                store_words(args[14].iterator + c * 8, (Uint32(PAIR_SENT),) * 4)
        # The trigger follows this kernel's PDL wait and its pushes (Rubin: the q
        # mailbox is re-armed after the merges, ``rearm_rows``): DcpCoreUmma
        # ``prewait`` reads the q mailbox and writes the split workspace before its
        # own wait, relying on the next core starting only after this.
        cute.arch.griddepcontrol_launch_dependents()
        R = self.merge_rows
        if cutlass.const_expr(self.tile_stride > 0):  # (spread above one wave)
            for y in range(tile, (self.batch + 15) // 16, self.tile_stride):
                if 16 * y + R * pusher < self.batch:
                    self.merge_spread(args, latent, scales, tid, y)
                    self.rearm_rows(mailbox, 16 * y + R * pusher, R, head, tid)
        elif self.merges(pusher):
            if cutlass.const_expr(self.spread):
                self.merge_spread(args, latent, scales, tid)
                self.rearm_rows(mailbox, 16 * tile + R * pusher, R, head, tid)
            elif cutlass.const_expr(self.wide):
                self.merge_row(args, latent, scales, wb, tid, bar)
                self.rearm_rows(mailbox, pusher, 1, head, tid)
            else:
                self.merge(args, latent, scales, wb, tid)
                self.rearm_rows(mailbox, 16 * tile, 16, head, tid)

    @cute.jit
    def rearm_rows(self, mailbox, first, rows, head, tid):
        """Rubin (``lean``): re-arm this head's share (gathered heads head x DCP ..) of
        the q mailbox rows ``first`` .. ``first`` + ``rows`` - 1 (< B) this CTA merged,
        after the merge. Its polls saw every source's pushes of this layer, which each
        source sent after its MlaQuery completed, so every rank's q publish of these
        rows has landed before the re-arm. (Re-armed right after the PDL wait, a row
        this rank's core never polled, an empty request's or every row of a rank with
        no local KV, could re-arm before a peer's publish lands: that stale row then
        passes as published q in the parity's next layer.) Each row is merged by one
        CTA per head, so the heads' shares cover the batch's rows."""
        if cutlass.const_expr(self.lean):
            D, H, B = self.size, self.heads, self.batch
            armed = tuple(map(Uint32, ARMED if self.valid_box else (Q_SENT,) * 4))
            cute.arch.sync_threads()  # (every thread's polls of the merge first)
            per: cutlass.Constexpr = D * 36  # 16 B chunks of this head's share of a row
            for i in range(tid, rows * per, 512):
                b = first + i // per
                if b < B:
                    c = (b * D * H + head * D) * 36 + i % per
                    store_words(mailbox.iterator + c * 16, armed)

    @cute.jit
    def merges(self, pusher):
        """Whether this CTA merges: z = 0 (one CTA per head and 16 requests), wide
        z < B (request z), or spread: rows R z .. of its tile exist."""
        result = pusher == 0
        if cutlass.const_expr(self.spread):
            _, tile, _ = cute.arch.block_idx()
            result = 16 * tile + self.merge_rows * pusher < self.batch
        elif cutlass.const_expr(self.wide):
            result = pusher < self.batch
        return result

    @cute.jit
    def lse_slot(self, parity, source, cap, b, gh):
        """Index of source's LSE of (request b, global head gh) in the owner's LSE box
        (both parities): [2, DCP, CAP, H], or wide [2, CAP, H, DCP]."""
        H, D = self.heads, self.size
        slot = 0
        if cutlass.const_expr(self.wide or self.spread):
            slot = ((parity * cap + b) * H + gh % H) * D + source
        else:
            slot = ((parity * D + source) * cap + b) * H + gh % H
        return slot

    @cute.jit
    def push_lse(self, args, me, threads, cap):
        """This thread's share of the B x DCP*H LSEs -> their head owners' LSE boxes."""
        rows_lse, lens, lpeers, parity, rank = (args[i] for i in (1, 2, 6, 12, 13))
        H, D, B = self.heads, self.size, self.batch
        for i in range(me, B * D * H, threads):
            b, gh = i // (D * H), i % (D * H)
            if cutlass.const_expr(self.fold > 0):
                n = lens[b // self.q]
                if cutlass.const_expr(self.lse_split):
                    fb, fgh = self.fold_row(b, gh)
                    value = self.folded_lse(rows_lse, fb, fgh, n)
                else:
                    value = self.folded_lse(rows_lse, b, gh)
            else:
                value, n = rows_lse[b, 0, gh], lens[b // self.q]
            bad = (value != value) | (value <= Float32(EMPTY_LSE))  # NaN, -inf
            if (n <= 0) | bad | (value >= Float32(-EMPTY_LSE)):  # or +inf
                value = Float32(EMPTY_LSE)
            slot = self.lse_slot(parity, rank, cap, b, gh)
            store_relaxed(lpeers[gh // H] + Int64(slot) * 4, (float_bits(value),))

    def fold_row(self, b, gh):
        """(request, packed row) of output row b, global head gh in the split
        workspace: q = 1 rows are requests; packed verify rows (q > 1) put query row
        t of request r at packed row t * DCP * H + gh (``split_partials_q``)."""
        if self.q == 1:
            return b, gh
        return b // self.q, (b % self.q) * (self.size * self.heads) + gh

    @cute.jit
    def valid_splits(self, n, r=None):
        """Tokenspeed's nonempty splits of n local tokens (128-token tiles); LONG_SPLIT:
        the short split's partition up to its long tiles (core_ts.get_k_tile_count).
        LPC: request r's chunk count (lpc_filled)."""
        if cutlass.const_expr(self.lpc is not None):
            return lpc_filled(self.lpc_words, n, r, self.batch // self.q)
        return filled_splits(n, self.fold, self.long)

    @cute.jit
    def fold_sources(self, n, r=None):
        """The split each of the ``fold`` slots of a request of n local tokens reads."""
        last = max(self.valid_splits(n, r) - 1, 0)
        return [min(Int32(s), last) for s in range(self.fold)]

    @cute.jit
    def lse_weights(self, stats, b, gh, n):
        """(max, weights, sum of weights) of the row's split LSEs (``lse_split``),
        and the split each of the ``fold`` slots reads: slots past the request's
        nonempty splits re-read its last one (finite data) with weight 0."""
        S = self.fold
        if cutlass.const_expr(self.filled):
            return self.filled_weights(stats, b, gh, n)
        last = max(self.valid_splits(n, b) - 1, 0)
        src = self.fold_sources(n, b)
        ms = [stats[b, gh, src[s]] for s in range(S)]
        top = ms[0]
        for s in cutlass.range_constexpr(1, S):
            top = max(top, ms[s])
        if cutlass.const_expr(self.q > 1):
            # causal verify rows: a split may hold no visible key of a row (LSE -inf);
            # a row with none at all folds to zero weights
            top = max(top, Float32(EMPTY_LSE))
        weights = []
        for s in cutlass.range_constexpr(S):
            live = Float32(min(max(last + 1 - s, 0), 1))
            weights.append(cute.math.exp2(ms[s] - top, fastmath=True) * live)
        total = Float32(0)
        for s in cutlass.range_constexpr(S):
            total = total + weights[s]
        return top, weights, total, src

    @cute.jit
    def filled_weights(self, stats, b, gh, n, nv=None):
        """``lse_weights`` over the request's filled splits only (``filled``): the
        stats of slots s >= filled_splits(n) are not loaded (predicated loads, no
        branch); they read EMPTY_LSE, whose weight exp2(EMPTY_LSE - max) is +0, as the
        reference's 0-weighted re-reads of the last filled split. The same max and
        the same FP32 sum in slot order."""
        S = self.fold
        if cutlass.const_expr(nv is None):
            nv = self.valid_splits(n, b)
        ms = [
            load_f32_if(
                at(stats, b, gh, s),
                cutlass.select_(nv > s, Int32(1), Int32(0)),
                Float32(EMPTY_LSE),
            )[0]
            for s in range(S)
        ]
        top = ms[0]
        for s in cutlass.range_constexpr(1, S):
            top = max(top, ms[s])
        if cutlass.const_expr(self.q > 1):
            top = max(top, Float32(EMPTY_LSE))
        weights = [cute.math.exp2(ms[s] - top, fastmath=True) for s in range(S)]
        total = Float32(0)
        for s in cutlass.range_constexpr(S):
            total = total + weights[s]
        return top, weights, total, [Int32(s) for s in range(S)]

    @cute.jit
    def split_weights(self, stats, b, gh, polled=None):
        """(max, weights, weighted sum) of the row's split (max, sum) pairs, read as
        16 B vectors when the fold is even (the 2 * fold FP32 stats of a row are
        contiguous and 16 B aligned), else as scalars; ``polled``: already polled
        (max, sum) lists (``polled_split``)."""
        S = self.fold
        ms, ls = [], []
        if cutlass.const_expr(polled is not None):
            ms, ls = polled
        elif cutlass.const_expr(self.polled_split):
            _, ms, ls = self.poll_split(None, stats, b, gh, 0, False)
        elif cutlass.const_expr(S % 2 == 0):
            base = at(stats, b, gh, 0, 0)
            values = []
            for q in cutlass.range_constexpr(2 * S // 4):
                values += list(load_f32x4(base + 4 * q, mutable=True))
            ms, ls = values[:S], values[S : 2 * S]
        else:
            ms = [stats[b, gh, 0, s] for s in range(S)]
            ls = [stats[b, gh, 1, s] for s in range(S)]
        top = ms[0]
        for s in cutlass.range_constexpr(1, S):
            top = max(top, ms[s])
        weights = [cute.math.exp2(ms[s] - top, fastmath=True) for s in range(S)]
        total = Float32(0)
        for s in cutlass.range_constexpr(S):
            total = total + weights[s] * ls[s]
        return top, weights, total

    @cute.jit
    def folded(self, part, stats, b, gh, k, n=None):
        """BF16 chunk k (8 columns) of the row merged over its splits (one rounding).
        Each split's 8 FP32 partials are one 32 B load, issued before the stats loads
        (none depends on them)."""
        xs, polled = [], None
        if cutlass.const_expr(self.lse_split):
            S = self.fold
            weights = total = src = None
            if cutlass.const_expr(self.filled):
                xs, weights, total = self.filled_partials(part, stats, b, gh, k, n)
            elif cutlass.const_expr(self.q > 1):  # the partials' loads first
                src = self.fold_sources(n, b)
                xs = [load_f32x8(at(part, b, gh, src[s], 8 * k)) for s in range(S)]
                _, weights, total, _ = self.lse_weights(stats, b, gh, n)
            else:
                _, weights, total, src = self.lse_weights(stats, b, gh, n)
                xs = [load_f32x8(at(part, b, gh, src[s], 8 * k)) for s in range(S)]
            acc = zeros(8, Float32)
            for s in cutlass.range_constexpr(self.fold):
                for e in cutlass.range_constexpr(8):
                    term = weights[s] * xs[s][e]
                    if cutlass.const_expr(self.q > 1):  # masked splits' O may be NaN
                        term = Float32(
                            cutlass.select_(weights[s] > 0, term, Float32(0))
                        )
                    acc[e] = acc[e] + term
            inv = Float32(0)
            if total > Float32(0):
                inv = Float32(self.out_scale) / total
            return tuple(
                pack_pair(acc[2 * e] * inv, acc[2 * e + 1] * inv, policy=None)
                for e in range(4)
            )
        if cutlass.const_expr(self.polled_split):  # partials and stats together
            xs, ms, ls = self.poll_split(part, stats, b, gh, k, True)
            polled = (ms, ls)
        else:
            xs = [load_f32x8(at(part, b, gh, s, 8 * k)) for s in range(self.fold)]
        _, weights, total = self.split_weights(stats, b, gh, polled)
        acc = zeros(8, Float32)
        for s in cutlass.range_constexpr(self.fold):
            for e in cutlass.range_constexpr(8):
                acc[e] = acc[e] + weights[s] * xs[s][e]
        inv = Float32(0)
        if total > Float32(0):
            inv = Float32(1) / total
        return tuple(
            pack_pair(acc[2 * e] * inv, acc[2 * e + 1] * inv, policy=None)
            for e in range(4)
        )

    @cute.jit
    def filled_partials(self, part, stats, b, gh, k, n):
        """(partials, weights, sum of weights) of chunk k over the request's filled
        splits (``filled``): the other slots load nothing (predicated: 8 zeros and
        weight +0, so their terms add +0 to a sum that is never -0, as the
        reference's). The reference's order: q = 1 the stats and weights first, then
        the partials (the 13 x 8 partial registers are not live across the stats
        loads and exp2s: with them first the B1 kernel spilled, 312 B of stack vs the
        reference's 64); q > 1 the partials first."""
        S = self.fold
        nv = self.valid_splits(n, b)
        xs, weights, total = [], None, None
        if cutlass.const_expr(self.q == 1):
            _, weights, total, _ = self.filled_weights(stats, b, gh, n, nv)
        xs = [
            load_f32x8_if(
                at(part, b, gh, s, 8 * k), cutlass.select_(nv > s, Int32(1), Int32(0))
            )
            for s in range(S)
        ]
        if cutlass.const_expr(self.q > 1):
            _, weights, total, _ = self.filled_weights(stats, b, gh, n, nv)
        return xs, weights, total

    @cute.jit
    def poll_split(self, part, stats, b, gh, k, parts: cutlass.Constexpr[bool]):
        """The row's ``fold`` (max, sum) pairs and (``parts``) its ``fold`` 32 B split
        partials of chunk k, polled until no word is SPLIT_SENT: every load in
        flight, then the late ones re-polled together (one round trip per pass)."""
        S = self.fold
        P = S if parts else 0
        raw = zeros((S, 8), Uint32)
        sraw = zeros(2 * S, Uint32)
        if cutlass.const_expr(SPLIT_GATE and parts):  # the stats gate first
            _, ms, ls = self.poll_split(None, stats, b, gh, k, False)
            for i in cutlass.range_constexpr(S):
                sraw[i] = Uint32(float_bits(ms[i]))
                sraw[S + i] = Uint32(float_bits(ls[i]))
        for s in cutlass.range_constexpr(P):
            loaded = poll_b32x8(at(part, b, gh, s, 8 * k))
            for e in cutlass.range_constexpr(8):
                raw[s, e] = loaded[e]
        if cutlass.const_expr(not (SPLIT_GATE and parts)):
            for i in cutlass.range_constexpr(2 * S):
                sraw[i] = load_volatile(at(stats, b, gh, i // S, i % S))[0]
        pending = Int32(0)
        for s in cutlass.range_constexpr(P):
            if not published((raw[s, e] for e in range(8)), sentinel=SPLIT_SENT):
                pending |= 1 << s
        for i in cutlass.range_constexpr(2 * S):
            if sraw[i] == Uint32(SPLIT_SENT):
                pending |= 1 << (S + i)
        while pending != 0:
            for s in cutlass.range_constexpr(P):
                if (pending & (1 << s)) != 0:
                    loaded = poll_b32x8(at(part, b, gh, s, 8 * k))
                    for e in cutlass.range_constexpr(8):
                        raw[s, e] = loaded[e]
            for i in cutlass.range_constexpr(2 * S):
                if (pending & (1 << (S + i))) != 0:
                    sraw[i] = load_volatile(at(stats, b, gh, i // S, i % S))[0]
            for s in cutlass.range_constexpr(P):
                if (pending & (1 << s)) != 0:  # noqa: SIM102
                    if published((raw[s, e] for e in range(8)), sentinel=SPLIT_SENT):
                        pending &= ~(1 << s)
            for i in cutlass.range_constexpr(2 * S):
                if (pending & (1 << (S + i))) != 0:  # noqa: SIM102
                    if sraw[i] != Uint32(SPLIT_SENT):
                        pending &= ~(1 << (S + i))
        xs = [tuple(bits_float(raw[s, e]) for e in range(8)) for s in range(P)]
        values = [bits_float(sraw[i]) for i in range(2 * S)]
        return xs, values[:S], values[S:]

    @cute.jit
    def folded_lse(self, stats, b, gh, n=None):
        top = total = None
        if cutlass.const_expr(self.lse_split):
            top, _, total, _ = self.lse_weights(stats, b, gh, n)
        else:
            top, _, total = self.split_weights(stats, b, gh)
        value = Float32(-math.inf)
        if total > Float32(0):
            value = top + cute.math.log2(total, fastmath=False)
        return value

    @cute.jit
    def merge(self, args, latent, scales, wb, tid):
        """The owner's merge of request rows t0.. of this head, then W_UV + gate."""
        box, lbox, gate, totals, output = args[3], args[4], args[9], args[10], args[11]
        D, B = self.size, self.batch
        head, tile, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        t0 = tile * 16
        col = head * 128 + 32 * (warp // 4) + 8 * (warp % 4) + 2 * quarter
        gates = []
        for r in cutlass.range_constexpr(2):
            word = Uint32(0)
            if t0 + group + 8 * r < B:
                word = load_words(
                    at(gate, t0 + group + 8 * r, col), count=1, mutable=True
                )[0]
            gates.append(unpack_bf16(word))
        alive = []  # Rubin: the global lengths of the output rows, read early
        if cutlass.const_expr(self.lean):
            for r in cutlass.range_constexpr(2):
                row_total = Int32(0)
                if t0 + group + 8 * r < B:
                    row_total = totals[(t0 + group + 8 * r) // self.q]
                alive.append(row_total)
        if t0 + warp < B:  # warp r: the source weights of request t0 + r
            lse = Float32(EMPTY_LSE)
            if lane < D:
                source = at(lbox, lane, t0 + warp, head)
                word = load_volatile(source)[0]
                while word == Uint32(LSE_SENT):
                    word = load_volatile(source)[0]
                store_words(source, (Uint32(LSE_SENT),))
                lse = bits_float(word)
            top = lse
            for shift in cutlass.range_constexpr(4, -1, -1):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            weight = Float32(0)
            if lse > Float32(EMPTY_LSE):
                weight = cute.math.exp2(lse - top, fastmath=True)
            total = warp_sum(weight)
            scale = Float32(0)
            if total > 0:
                scale = weight / total
            if lane < D:
                scales[warp, lane] = scale
        cute.arch.sync_threads()
        chunk = tid % 64
        for half in cutlass.range_constexpr(2):  # rows r and r + 8, 16 B chunk
            r = tid // 64 + 8 * half
            merged = zeros(8, Float32)
            if t0 + r < B:
                for s0 in cutlass.range_constexpr(0, D, 8):  # 8 polls in flight
                    sources = [
                        at(box, s, t0 + r, head, 8 * chunk)
                        for s in range(s0, min(D, s0 + 8))
                    ]
                    loads = [load_volatile(source, count=4) for source in sources]
                    for i in cutlass.range_constexpr(len(sources)):
                        words = loads[i]
                        while not published(words):
                            words = load_volatile(sources[i], count=4)
                        store_words(sources[i], (Uint32(PAIR_SENT),) * 4)
                        w = scales[r, s0 + i]
                        for e in cutlass.range_constexpr(4):
                            lo, hi = unpack_bf16(words[e])
                            merged[2 * e] += w * lo
                            merged[2 * e + 1] += w * hi
            pairs = tuple(
                pack_pair(merged[2 * e], merged[2 * e + 1], policy=None)
                for e in range(4)
            )
            store_shared_words(latent.iterator + r * 544 + 8 * chunk, pairs)
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    latent.iterator + r * 544 + 32 * c + 8 * quarter,
                    count=4,
                    shared=True,
                )
                for r in (group, group + 8)
            )
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                b = (wb[c][2 * s], wb[c][2 * s + 1])
                mma_into(parts[s, None], a, b, bf16=True)
        acc = [parts[0, k] + parts[1, k] for k in range(4)]
        for r in cutlass.range_constexpr(2):
            t = t0 + group + 8 * r
            if t < B:
                # Padded rows may hold a NaN gate; keep inactive rows zero.
                lo, hi = Float32(0), Float32(0)
                early = cutlass.const_expr(self.lean)  # Rubin: the length read early
                if (alive[r] if early else totals[t // self.q]) > 0:
                    lo, hi = (
                        Float32(BFloat16(acc[2 * r + e])) * sigmoid(gates[r][e], False)
                        for e in range(2)
                    )
                store_words(at(output, t, col), (bf16x2(lo, hi),))

    @cute.jit
    def merge_spread(self, args, latent, scales, tid, row_tile=None):
        """spread: CTA z of (head, row tile y) merges rows 16 y + R z + r (r < R) into
        rows r of its latent tile, then warp w projects output tile w of them: the
        weights, sums and per-tile MMAs of merge() (MMA rows are independent, so the
        same values). R = 16 (one pusher: H x row tiles fill the SMs) also merges and
        stores the MMA tile's rows 8-15. ``row_tile``: y (``tile_stride``), else the
        CTA's own. Back-to-back calls need no barrier: the next call writes the
        scales after the last read (before this call's second barrier) and the
        latent tile only after its first barrier (every warp's MMA reads done)."""
        box, lbox, gate, totals, output = args[3], args[4], args[9], args[10], args[11]
        D, B, R = self.size, self.batch, self.merge_rows
        head, tile, z = cute.arch.block_idx()
        if cutlass.const_expr(row_tile is not None):
            tile = row_tile
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        t0 = 16 * tile + R * z  # this CTA's first row
        col = head * 128 + 32 * (warp // 4) + 8 * (warp % 4) + 2 * quarter
        v = 32 * (warp // 4) + 8 * (warp % 4) + group  # the warp's W_UV output tile
        gates = (Float32(0), Float32(0))
        alive = Int32(0)
        if (group < R) & (t0 + group < B):  # Rubin: the global lengths read early
            word = load_words(at(gate, t0 + group, col), count=1, mutable=True)[0]
            gates = unpack_bf16(word)
            alive = totals[(t0 + group) // self.q]
        if cutlass.const_expr(R > 8):  # rows group + 8
            gates8 = (Float32(0), Float32(0))
            alive8 = Int32(0)
            if t0 + group + 8 < B:
                word = load_words(at(gate, t0 + group + 8, col), count=1, mutable=True)
                gates8 = unpack_bf16(word[0])
                alive8 = totals[(t0 + group + 8) // self.q]
        if (warp < R) & (t0 + warp < B):  # warp r: the source weights of row t0 + r
            lse = Float32(EMPTY_LSE)
            if lane < D:  # the row's sources share one line
                source = lbox.iterator + ((t0 + warp) * self.heads + head) * D + lane
                word = load_volatile(source)[0]
                while word == Uint32(LSE_SENT):
                    word = load_volatile(source)[0]
                store_words(source, (Uint32(LSE_SENT),))
                lse = bits_float(word)
            top = lse
            for shift in cutlass.range_constexpr(4, -1, -1):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            weight = Float32(0)
            if lse > Float32(EMPTY_LSE):
                weight = cute.math.exp2(lse - top, fastmath=True)
            total = warp_sum(weight)
            scale = Float32(0)
            if total > 0:
                scale = weight / total
            if lane < D:
                scales[warp, lane] = scale
        cute.arch.sync_threads()
        chunk = tid % 64  # 16 B chunk of row t0 + r
        for h in cutlass.range_constexpr((R + 7) // 8):  # r = tid // 64 (+ 8: R = 16)
            r = tid // 64 if h == 0 else tid // 64 + 8
            if (r < R) & (t0 + r < B):
                merged = zeros(8, Float32)
                P = WIDE_ROW_POLLS
                for s0 in cutlass.range_constexpr(0, D, P):
                    sources = [
                        at(box, s, t0 + r, head, 8 * chunk)
                        for s in range(s0, min(D, s0 + P))
                    ]
                    raw = self.poll_rows(sources)
                    for i in cutlass.range_constexpr(len(sources)):
                        store_words(sources[i], (Uint32(PAIR_SENT),) * 4)
                        w = scales[r, s0 + i]
                        for e in cutlass.range_constexpr(4):
                            lo, hi = unpack_bf16(raw[i, e])
                            merged[2 * e] += w * lo
                            merged[2 * e + 1] += w * hi
                pairs = tuple(
                    pack_pair(merged[2 * e], merged[2 * e + 1], policy=None)
                    for e in range(4)
                )
                store_shared_words(latent.iterator + r * 544 + 8 * chunk, pairs)
        wb = [  # loaded after the polls: not live across them (no spills)
            load_words(at(args[8], head, 32 * c + 8 * quarter, v), count=4)
            for c in range(16)
        ]
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains; rows >= R unused
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    latent.iterator + row * 544 + 32 * c + 8 * quarter,
                    count=4,
                    shared=True,
                )
                for row in (group, group + 8)
            )
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                b = (wb[c][2 * s], wb[c][2 * s + 1])
                mma_into(parts[s, None], a, b, bf16=True)
        if (group < R) & (t0 + group < B):
            lo, hi = Float32(0), Float32(0)
            if alive > 0:  # padded rows may hold a NaN gate; keep inactive rows zero
                lo, hi = (
                    Float32(BFloat16(parts[0, e] + parts[1, e]))
                    * sigmoid(gates[e], False)
                    for e in range(2)
                )
            store_words(at(output, t0 + group, col), (bf16x2(lo, hi),))
        if cutlass.const_expr(R > 8):  # noqa: SIM102 (compile-time): MMA rows 8-15
            if t0 + group + 8 < B:
                lo, hi = Float32(0), Float32(0)
                if alive8 > 0:
                    lo, hi = (
                        Float32(BFloat16(parts[0, 2 + e] + parts[1, 2 + e]))
                        * sigmoid(gates8[e], False)
                        for e in range(2)
                    )
                store_words(at(output, t0 + group + 8, col), (bf16x2(lo, hi),))

    @cute.jit
    def poll_rows(self, sources):
        """The 16 B chunks at ``sources``, every not-yet-published one re-polled
        together (one round trip per pass, not one per late source)."""
        n = len(sources)
        raw = zeros((n, 4), Uint32)
        for i in cutlass.range_constexpr(n):
            words = load_volatile(sources[i], count=4)
            for j in cutlass.range_constexpr(4):
                raw[i, j] = words[j]
        pending = Int32(0)
        for i in cutlass.range_constexpr(n):
            if not published(raw[i, j] for j in range(4)):
                pending |= 1 << i
        while pending != 0:
            for i in cutlass.range_constexpr(n):
                if (pending & (1 << i)) != 0:
                    words = load_volatile(sources[i], count=4)
                    for j in cutlass.range_constexpr(4):
                        raw[i, j] = words[j]
            for i in cutlass.range_constexpr(n):
                if (pending & (1 << i)) != 0:  # noqa: SIM102
                    if published(raw[i, j] for j in range(4)):
                        pending &= ~(1 << i)
        return raw

    @cute.jit
    def merge_row(self, args, latent, scales, wb, tid, bar):
        """wide: merge cluster CTA z merges request z of this head into row z of every
        cluster CTA's latent tile (DSMEM), then projects output tiles 2z, 2z + 1 of all
        requests: the weights, sums and per-tile MMAs of merge(), so the same values."""
        box, lbox, gate, totals, output = args[3], args[4], args[9], args[10], args[11]
        D, B = self.size, self.batch
        head, _, t = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        tile = 2 * t + warp  # output tile of warps 0, 1
        col = head * 128 + 32 * (tile // 4) + 8 * (tile % 4) + 2 * quarter
        gates = (Float32(0), Float32(0))
        alive = Int32(0)
        if (warp < 2) & (group < B):
            word = load_words(at(gate, group, col), count=1, mutable=True)[0]
            gates = unpack_bf16(word)
            alive = totals[group // self.q]
        cute.arch.cluster_wait()  # every cluster CTA's row barrier is initialized
        if warp == 0:  # lane s: source s's LSE (the 16 sources share one line)
            lse = Float32(EMPTY_LSE)
            if cutlass.const_expr(self.lse_uniform):  # one warp-uniform poll loop
                source = lbox.iterator + (t * self.heads + head) * D + lane
                word = Uint32(0)  # (lanes >= D: never the sentinel)
                if lane < D:
                    word = load_volatile(source)[0]
                if cutlass.const_expr(POLL_LSE):
                    ready = cute.arch.vote_all_sync(word != Uint32(LSE_SENT))
                    while not ready:
                        if word == Uint32(LSE_SENT):  # (lanes < D only)
                            word = load_volatile(source)[0]
                        ready = cute.arch.vote_all_sync(word != Uint32(LSE_SENT))
                if lane < D:
                    store_words(source, (Uint32(LSE_SENT),))
                    lse = bits_float(word)
            elif lane < D:
                source = lbox.iterator + (t * self.heads + head) * D + lane
                word = load_volatile(source)[0]
                while word == Uint32(LSE_SENT):
                    word = load_volatile(source)[0]
                store_words(source, (Uint32(LSE_SENT),))
                lse = bits_float(word)
            top = lse
            for shift in cutlass.range_constexpr(4, -1, -1):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            weight = Float32(0)
            if lse > Float32(EMPTY_LSE):
                weight = cute.math.exp2(lse - top, fastmath=True)
            total = warp_sum(weight)
            scale = Float32(0)
            if total > 0:
                scale = weight / total
            if lane < D:
                scales[0, lane] = scale
        cute.arch.sync_threads()
        if tid < 64:  # 16 B chunk tid of the row, WIDE_ROW_POLLS sources per pass
            merged = zeros(8, Float32)
            P = WIDE_ROW_POLLS
            for s0 in cutlass.range_constexpr(0, D, P):
                sources = [
                    at(box, s, t, head, 8 * tid) for s in range(s0, min(D, s0 + P))
                ]
                raw = self.poll_rows(sources)
                for i in cutlass.range_constexpr(len(sources)):
                    store_words(sources[i], (Uint32(PAIR_SENT),) * 4)
                    w = scales[0, s0 + i]
                    for e in cutlass.range_constexpr(4):
                        lo, hi = unpack_bf16(raw[i, e])
                        merged[2 * e] += w * lo
                        merged[2 * e + 1] += w * hi
            pairs = tuple(
                pack_pair(merged[2 * e], merged[2 * e + 1], policy=None)
                for e in range(4)
            )
            for peer in cutlass.range_constexpr(8):
                push_words(latent.iterator + t * 544 + 8 * tid, pairs, bar, peer)
        wait(bar)
        cute.arch.cluster_arrive_relaxed()  # every row landed here: peers may exit
        if warp < 2:
            parts = zeros((2, 4), Float32)  # two accumulation chains
            for c in cutlass.range_constexpr(16):
                x, y = (
                    load_words(
                        latent.iterator + r * 544 + 32 * c + 8 * quarter,
                        count=4,
                        shared=True,
                    )
                    for r in (group, group + 8)
                )
                for s in cutlass.range_constexpr(2):
                    a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                    b = (wb[c][2 * s], wb[c][2 * s + 1])
                    mma_into(parts[s, None], a, b, bf16=True)
            if group < B:  # rows group + 8 are unused (B <= 8)
                lo, hi = Float32(0), Float32(0)
                if alive > 0:
                    lo, hi = (
                        Float32(BFloat16(parts[0, e] + parts[1, e]))
                        * sigmoid(gates[e], False)
                        for e in range(2)
                    )
                word = bf16x2(lo, hi)
                if cutlass.const_expr(self.polled_out):  # never the PAIR_SENT pattern
                    word = no_neg_zero(word)
                store_words(at(output, group, col), (word,))
        cute.arch.cluster_wait()


def polled_output(heads, size, batch, sms):
    """Whether DcpValue's decode output (q = 1) of ``batch`` is Lamport-polled by the
    o-projection (POLLED_OUTPUT, wide merge only)."""
    if sms <= GB300_SMS or not POLLED_OUTPUT.get(batch):
        return False
    return DcpValue(heads, size, batch, sms, 1, 0, polled_out=True).polled_out


@cache
def compile_value(
    heads,
    size,
    batch,
    cap,
    uv,
    pitch,
    device,
    sms=148,
    q=1,
    fold=0,
    polled_out=False,
    polled_split=False,
    lse_split=False,
    out_scale=1.0,
    valid_box=False,
    lse_uniform=False,
    pushed=False,
    lpc=None,
):
    """DcpValue of one batch (``q`` rows per request); ``uv``: the (shape, stride,
    dtype) of W_UV, ``pitch``: the projected rows' (the gate follows the 2112 query /
    KV columns); ``fold``: the DcpCore split count (0: tokenspeed rows); ``polled_out``:
    the output is Lamport-polled by the o-projection (POLLED_OUTPUT; the last argument
    before the stream is the other parity's output buffer, re-armed); ``polled_split``:
    the split workspace is polled (DcpCoreUmma, POLLED_SPLIT); ``lse_uniform``: the
    wide merge's LSE poll is warp-uniform (DV_LSE_UNIFORM); ``pushed``: the core
    pushed the rows and LSEs (merge only); ``lpc``: (groups, slots) of DS-T2P's LPC
    partition (DcpValue's ``lpc``)."""
    f32, i32 = torch.float32, torch.int32
    w_uv, gate = fake(*uv), fake((batch, heads * 128), (pitch, 1))
    rows = fake((batch, 1, size * heads, 512))
    rows_lse = fake((batch, 1, size * heads), None, f32, 4)
    if fold:
        rows = fake((batch, size * heads, fold, 512), None, f32)
        rows_lse = fake((batch, size * heads, 2, fold), None, f32, 4)
    if fold and lse_split:  # tokenspeed's split partials (split_partials)
        rows_lse = fake((batch, size * heads, fold), None, f32, 4)
    if fold and lse_split and q > 1:  # packed verify rows (split_partials_q)
        packed = -(-q * size * heads // 128) * 128
        rows = fake((batch // q, packed, fold, 512), None, f32)
        rows_lse = fake((batch // q, packed, fold), None, f32, 4)
    lens = fake((batch // q,), None, i32, 4)
    box = fake((size, cap, heads, 512))
    lbox = fake((size, cap, heads), None, f32, 4)
    peers = fake((size,), None, torch.int64, 8)
    mailbox = fake((cap, size * heads, 576), None, torch.uint8)
    output = fake((batch, heads * 128), (heads * 128, 1))
    args: tuple = (rows, rows_lse, lens, box, lbox, peers, peers, mailbox, w_uv)
    args += (gate, lens, output, Int32(0), Int32(0))
    value = DcpValue(
        heads,
        size,
        batch,
        sms,
        q,
        fold,
        polled_out,
        polled_split,
        lse_split,
        out_scale,
        valid_box,
        lse_uniform,
        pushed,
        lpc=lpc,
    )
    if value.polled_out:
        args += (fake((batch, heads * 128), (heads * 128, 1)),)
    with torch.accelerator.device_index(device):
        return cute.compile(value, *args, make_fake_stream())


_WORKSPACE: dict[tuple[torch.device, int], torch.Tensor] = {}


def workspace(device, heads, size, sms):
    """The split-KV workspace of (``device``, ``heads``), at least ``size`` bytes."""
    # One per (device, heads), shared by every MLA layer: the calls are stream
    # ordered. PreparedMla.prepare_cache runs each prepared (batch, q) before any
    # capture, so the buffer reaches its final size first; growing it during a capture
    # would leave the graphs captured before with the freed buffer.
    key = device, heads
    if key not in _WORKSPACE or _WORKSPACE[key].numel() < size:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the MLA split-KV workspace must be sized before capture"
            )
        bound = max(size, sms * heads * 513 * 4)
        _WORKSPACE[key] = torch.empty(bound, dtype=torch.int8, device=device)
    return _WORKSPACE[key][:size] if size else None


def stock_decode(
    query,
    cache,
    block_table,
    lengths,
    max_seq,
    scales,
    out,
    lse=None,
    min_split=1,
    pdl=True,
    cp=(1, 0, 1),
    totals=None,
    pack=False,
    blocks=None,
):
    """tokenspeed_mla_decode's FP8 path with caller-owned ``out`` and ``lse`` (log2)
    buffers and no per-call allocation: the split count follows from (B, max_seq,
    min_split); ``lengths`` bound the work, empty splits are skipped by its reducer.
    The q rows of a request mask causally; under DCP in global positions (``cp``:
    world, rank, interleave; ``totals``: the global lengths). ``pack``: M128 tiles hold
    consecutive (query row, head) rows across the q rows. ``blocks``: tokenspeed's
    occupancy bound for the split count (default: the SM count; 2 * batch makes
    ``min_split`` the exact split count)."""
    import tvm_ffi
    from tokenspeed_mla import mla_decode as md

    batch, q_len, heads, _ = query.shape
    fp8, device = torch.float8_e4m3fn, query.device
    arch = torch.cuda.get_device_capability(device)
    tiler, _ = md.select_mla_decode_tilers(
        heads, q_len, is_fp8=True, compute_capability=arch
    )
    pack = pack and tiler[0] == 128
    fold = 1 if pack else md.get_mla_decode_fold_sq_factor(heads, q_len, tiler[0])
    rows = (q_len // fold, heads * fold)
    if pack:
        rows = (md.compute_q_tile_layout(heads, q_len, 128)[1], 128)
    sms = md.get_num_sm(device)
    split, size = md._get_split_kv_and_workspace_size(
        batch, *rows, 512, blocks or sms, max_seq, fp8, tiler, min_split
    )
    scratch = workspace(device, heads, size, sms)
    kernel = md._get_compiled_mla_kernel(
        torch_dtype=fp8,
        page_size=cache.shape[1],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=False,
        compute_capability=arch,
        has_local_visible_lens=False,
        skip_correction_threshold=8.0,
        is_workspace_size_zero=size == 0,
        fold_sq_factor=fold,
        causal_mask=True,
        num_heads=heads,
        seq_len_q=q_len,
        window_left=-1,
        cp_world=cp[0],
        cp_rank=cp[1],
        cp_interleave_size=cp[2],
        use_pdl=pdl,
        return_lse=lse is not None,
        reducer_d_tiles=md._get_reducer_d_tiles(batch, q_len, heads, sms, split),
        reducer_max_splits=md._get_reducer_max_splits(split),
        pack_q=pack,
        partial_fp16=False,
    )
    cache = cache.view(fp8)
    softmax, output_scale = scales
    args = [query[..., :512], query[..., 512:], cache[..., :512], cache[..., 512:]]
    args += [block_table, out, lse, scratch]
    args += [Int32(split), lengths, lengths if totals is None else totals, None]
    args += [Float32(softmax), Float32(output_scale)]
    with tvm_ffi.use_torch_stream():
        kernel(*args)


@cache
def compile_rubin(heads, page, split, branch, device):
    """Tokenspeed's Rubin decode kernel (``rubin_decode``) for ``heads`` query heads at
    q = 1 over pages of ``page`` tokens, ``split`` KV splits, cluster ``branch``."""
    from tokenspeed_mla import mla_decode as md
    from tokenspeed_mla.mla_decode_fp8_sm107 import SM107_MLA_M_TILE as M
    from tokenspeed_mla.mla_decode_fp8_sm107 import (
        Sm107MultiHeadLatentAttentionForwardFP8 as Kernel,
    )

    kernel = Kernel(
        acc_dtype=Float32,
        lse_dtype=Float32,
        mma_qk_tiler_mn=(M, 128),
        mma_pv_tiler_mn=(M, 256),
        max_active_clusters=md._get_sm107_max_active_clusters(),
        page_size=page,
        # tokenspeed's thresholds (P <= 16 unsplit, <= 256 per split; FP8 max 448)
        skip_correction_threshold=4.0 if split == 1 else 8.0,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=False,
        num_heads=heads,
        seq_len_q=1,
        is_causal=True,
        use_fp16_softmax=False,  # FP32 exp2 and row sums
        force_branch=branch,
        reducer_max_splits=md._get_reducer_max_splits(split),
    )
    fp8, sym = cutlass.Float8E4M3FN, cute.sym_int
    rows, seq, heads_, pages, tokens, table = (sym() for _ in range(6))
    width, rope = sym(divisibility=16), sym(divisibility=16)
    fake_q = cute.runtime.make_fake_tensor
    compact = cute.runtime.make_fake_compact_tensor
    queries = [
        fake_q(
            fp8,
            (rows, seq, heads_, d),
            stride=(sym(), sym(), sym(), 1),
            assumed_align=16,
        )
        for d in (width, rope)
    ]
    kv = [
        fake_q(fp8, (pages, tokens, d), stride=(sym(), sym(), 1), assumed_align=16)
        for d in (width, rope)
    ]
    shape = (rows, seq, heads_)
    out = compact(
        BFloat16, (*shape, width), stride_order=(3, 2, 1, 0), assumed_align=16
    )
    lse = compact(Float32, shape, stride_order=(2, 1, 0), assumed_align=4)
    scratch = None if split == 1 else compact(cutlass.Int8, (sym(),), assumed_align=32)
    blocks = compact(Int32, (rows, table), stride_order=(1, 0), assumed_align=4)
    lengths = compact(Int32, (rows,), assumed_align=4)
    args = (*queries, *kv, blocks, out, lse, scratch, Int32(1), lengths, None)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    options = "--enable-tvm-ffi --opt-level 2 --gpu-arch=sm_107a"
    with torch.accelerator.device_index(device):
        return cute.compile(
            kernel, *args, Float32(1.0), Float32(1.0), stream, True, options=options
        )


def rubin_decode(query, cache, block_table, lengths, scales, out, lse, policy):
    """Tokenspeed's Rubin (sm_107) MLA decode kernel (mla_decode_fp8_sm107, from
    FlashInfer's rubin_mtp kernel) at q = 1: ``query`` [B, 1, H, 576] FP8 over the
    FP8 paged cache into BF16 ``out`` [B, 1, H, 512] and base-2 ``lse`` [B, 1, H], as
    stock_decode writes them (the stock DcpValue merge consumes either). ``policy``:
    (KV splits, cluster shape); split partials FP32 merged by its reducer kernel.
    Rows are packed into 256-row M tiles (H = 96: one tile per request, 96 rows live);
    a work item is a 4-CTA cluster (QK pair + PV pair, P over DSMEM) or two 2-CTA
    clusters; FP8 P, FP32 softmax, accumulation and partials; PDL chained."""
    import tvm_ffi
    from tokenspeed_mla import get_num_sm

    split, branch = policy
    batch, q_len, heads, _ = query.shape
    assert q_len == 1 and heads <= 256, (q_len, heads)
    device = query.device
    size = 0 if split == 1 else batch * 256 * split * 513 * 4
    scratch = workspace(device, heads, size, get_num_sm(device))
    kernel = compile_rubin(heads, cache.shape[1], split, branch, device.index)
    cache = cache.view(torch.float8_e4m3fn)
    softmax, output_scale = scales
    args = [query[..., :512], query[..., 512:], cache[..., :512], cache[..., 512:]]
    args += [block_table, out, lse, scratch, Int32(split), lengths, None]
    with tvm_ffi.use_torch_stream():
        kernel(*args, Float32(softmax), Float32(output_scale))


class _NoLaunch:
    def launch(self, *args, **kwargs):
        pass


def _without_reducer(base):
    class SplitPartials(base):
        """tokenspeed's generic FP8 decode kernel without its reduction kernel: the
        split partials (FP32 O / l) and per-split LSEs stay in the workspace."""

        def reduction_kernel(self, *args):
            return _NoLaunch()

    return SplitPartials


@cache
def compile_split_partials(heads, page, split, device):
    """Tokenspeed's generic decode kernel (stock_decode's configuration at q = 1)
    without its reducer, ``split`` KV splits."""
    from tokenspeed_mla import mla_decode as md

    base = md.BlackwellMultiHeadLatentAttentionForwardFP8
    md.BlackwellMultiHeadLatentAttentionForwardFP8 = _without_reducer(base)
    try:
        with torch.accelerator.device_index(device):
            return md._get_compiled_mla_kernel.__wrapped__(
                torch_dtype=torch.float8_e4m3fn,
                page_size=page,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                is_persistent=False,
                is_var_seq=True,
                is_var_split_kv=False,
                compute_capability=torch.cuda.get_device_capability(device),
                has_local_visible_lens=False,
                skip_correction_threshold=8.0,
                is_workspace_size_zero=False,
                fold_sq_factor=1,
                causal_mask=True,
                num_heads=heads,
                seq_len_q=1,
                window_left=-1,
                cp_world=1,
                cp_rank=0,
                cp_interleave_size=1,
                use_pdl=True,
                return_lse=True,
                reducer_d_tiles=1,
                reducer_max_splits=md._get_reducer_max_splits(split),
                pack_q=False,
                partial_fp16=False,
            )
    finally:
        md.BlackwellMultiHeadLatentAttentionForwardFP8 = base


def lse_fold_splits(batch, heads, max_seq, device):
    """The exact split count of LSE_FOLD[batch] after tokenspeed drops the empty
    trailing partitions of ``max_seq``."""
    from tokenspeed_mla import mla_decode as md

    tiler = (128, 128)
    args = batch, 1, heads, 512, 2 * batch, max_seq, torch.float8_e4m3fn, tiler
    split = md._get_split_kv_and_workspace_size(*args, LSE_FOLD[batch])[0]
    rule = LONG_SPLIT.get((batch, heads))
    if rule and split == rule[0] and md.get_num_sm(device) > GB300_SMS:
        # LONG_SPLIT only where max_seq fills all of its long splits (short max_seq: the
        # short split, as before)
        if md._get_split_kv_and_workspace_size(*args, rule[1])[0] == rule[1]:
            split = rule[1]
    return split


_PARTIALS: dict[tuple, torch.Tensor] = {}


def split_workspace_ts(device, heads, batch, split):
    """(workspace, partials [B, heads, split, 512], LSEs [B, heads, split]) of the
    LSE-folded tokenspeed core of one batch: tokenspeed's split-KV workspace layout,
    one per (device, heads, batch, split), shared by every MLA layer (stream ordered;
    as the shared rows of the stock path)."""
    key = device, heads, batch, split
    if key not in _PARTIALS:
        size = batch * heads * split * 513
        _PARTIALS[key] = torch.empty(size, dtype=torch.float32, device=device)
    buf, cut = _PARTIALS[key], batch * heads * split * 512
    part = buf[:cut].view(batch, heads, split, 512)
    return buf.view(torch.int8), part, buf[cut:].view(batch, heads, split)


# Rubin-class devices, DCP spec verify fold: (token rows, gathered heads) whose packed
# verify core runs this many KV splits instead of tokenspeed's rule. 16 ranks, 6 heads
# per rank, local 2000: M16 (DL7 B2 q8, 96 heads) rule 8 -> 4: MLA 21.85 -> 20.72
# us/layer (core -0.19, DcpValue -1.03); the split fold sums in another order (FP32).
# M32 4 -> 2 loses (+0.82).
VERIFY_FOLD_SPLITS = {(16, 96): 4}
# B9-B13: the short split of their LONG_SPLIT rule (above), tokenspeed's 1 -> 2.
VERIFY_FOLD_SPLITS.update({(r, 96): 2 for r in range(72, 105, 8)})
# DL3's odd-B verify rows 68-124: the short split of their LONG_SPLIT rule, 1 -> 2.
VERIFY_FOLD_SPLITS.update({(r, 96): 2 for r in range(68, 125, 8)})
# The drafter's B4 (28 rows, tokenspeed's 6): the short split of its LONG_SPLIT rule.
VERIFY_FOLD_SPLITS[28, 64] = 2


def verify_fold_splits(batch, q, heads, max_seq, device):
    """Tokenspeed's split count of the packed verify core (``stock_decode``'s rule at
    q > 1; ``VERIFY_FOLD_SPLITS`` on Rubin) and its packed rows (tiles x 128)."""
    from tokenspeed_mla import mla_decode as md

    tiles = md.compute_q_tile_layout(heads, q, 128)[1]
    sms = md.get_num_sm(device)
    args = batch // q, tiles, 128, 512, sms, max_seq, torch.float8_e4m3fn, (128, 128)
    split = md._get_split_kv_and_workspace_size(*args, fill_splits(batch, device))[0]
    if sms > GB300_SMS:
        split = VERIFY_FOLD_SPLITS.get((batch, heads), split)
        rule = LONG_SPLIT.get((batch, heads))
        if rule and split == rule[0] and max_seq > 128 * rule[2]:
            split = rule[1]  # (only where some request can pass the long tiles)
    return split, tiles * 128


@cache
def compile_split_partials_q(heads, page, split, reqs, q, cp, device, causal=True):
    """Tokenspeed's generic decode kernel in stock_decode's packed verify configuration
    (q > 1, causal in global positions ``cp`` = (world, rank, interleave)) without its
    reducer: the FP32 split partials and split LSEs stay in the workspace. ``causal``
    False: every query row sees the whole (local) context (the DSpark drafter)."""
    from tokenspeed_mla import mla_decode as md

    base = md.BlackwellMultiHeadLatentAttentionForwardFP8
    md.BlackwellMultiHeadLatentAttentionForwardFP8 = _without_reducer(base)
    sms = md.get_num_sm(torch.device("cuda", device))
    d_tiles = md._get_reducer_d_tiles(reqs, q, heads, sms, split)
    try:
        with torch.accelerator.device_index(device):
            return md._get_compiled_mla_kernel.__wrapped__(
                torch_dtype=torch.float8_e4m3fn,
                page_size=page,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                is_persistent=False,
                is_var_seq=True,
                is_var_split_kv=False,
                compute_capability=torch.cuda.get_device_capability(device),
                has_local_visible_lens=False,
                skip_correction_threshold=8.0,
                is_workspace_size_zero=False,
                fold_sq_factor=1,
                causal_mask=causal,
                num_heads=heads,
                seq_len_q=q,
                window_left=-1,
                cp_world=cp[0],
                cp_rank=cp[1],
                cp_interleave_size=cp[2],
                use_pdl=True,
                return_lse=True,
                reducer_d_tiles=d_tiles,
                reducer_max_splits=md._get_reducer_max_splits(split),
                pack_q=True,
                partial_fp16=False,
            )
    finally:
        md.BlackwellMultiHeadLatentAttentionForwardFP8 = base


def split_partials_q(
    query, cache, block_table, lengths, totals, scales, out, lse, split, cp, causal=True
):
    """``split_partials`` for the packed verify core (``query`` [requests, q, H, 576]):
    FP32 partials and base-2 LSEs per split in ``split_workspace_ts(device, tiles x
    128, requests, split)`` for DcpValue's LSE fold (``out`` / ``lse`` not written)."""
    import tvm_ffi
    from tokenspeed_mla import mla_decode as md

    reqs, q, heads, _ = query.shape
    device = query.device
    kernel = compile_split_partials_q(
        heads, cache.shape[1], split, reqs, q, cp, device.index, causal
    )
    rows = md.compute_q_tile_layout(heads, q, 128)[1] * 128
    buf = split_workspace_ts(device, rows, reqs, split)[0]
    cache = cache.view(torch.float8_e4m3fn)
    softmax, output_scale = scales
    args = [query[..., :512], query[..., 512:], cache[..., :512], cache[..., 512:]]
    args += [block_table, out, lse, buf, Int32(split), lengths, totals, None]
    args += [Float32(softmax), Float32(output_scale)]
    with tvm_ffi.use_torch_stream():
        kernel(*args)


@cache
def compile_fork(
    heads,
    page,
    split,
    batch,
    device,
    balanced=False,
    repoll=False,
    cluster=False,
    valid=False,
    valid_delay=0,
):
    """mla.core_ts's polled fork of tokenspeed's generic kernel (``split_partials``'s
    configuration; block_split_kvs carries its address table)."""
    from tokenspeed_mla import mla_decode as md

    from .core_ts import polled_core

    sms = md.get_num_sm(torch.device("cuda", device))
    base = md.BlackwellMultiHeadLatentAttentionForwardFP8
    md.BlackwellMultiHeadLatentAttentionForwardFP8 = _without_reducer(
        polled_core(
            batch,
            balanced=balanced,
            repoll=repoll,
            cluster=split if cluster else 0,
            valid=valid,
            valid_delay=valid_delay,
            long=long_split(batch, heads, split, sms),
        )
    )
    try:
        with torch.accelerator.device_index(device):
            return md._get_compiled_mla_kernel.__wrapped__(
                torch_dtype=torch.float8_e4m3fn,
                page_size=page,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                is_persistent=False,
                is_var_seq=True,
                is_var_split_kv=True,  # the table's (unused split) argument
                compute_capability=torch.cuda.get_device_capability(device),
                has_local_visible_lens=False,
                skip_correction_threshold=8.0,
                is_workspace_size_zero=False,
                fold_sq_factor=1,
                causal_mask=True,
                num_heads=heads,
                seq_len_q=1,
                window_left=-1,
                cp_world=1,
                cp_rank=0,
                cp_interleave_size=1,
                use_pdl=True,
                return_lse=True,
                reducer_d_tiles=1,
                reducer_max_splits=md._get_reducer_max_splits(split),
                pack_q=False,
                partial_fp16=False,
            )
    finally:
        md.BlackwellMultiHeadLatentAttentionForwardFP8 = base


def fork_table(query_box, x_rows, batch, device):
    """mla.core_ts's address table: [q mailbox lo, hi, x lo, hi, x pitch (bytes)]."""
    words = [0] * max(8, batch)
    for i, address in ((0, query_box.data_ptr()), (2, x_rows.data_ptr())):
        words[i], words[i + 1] = address & 0xFFFFFFFF, address >> 32
    words[4] = x_rows.stride(0) * x_rows.element_size()
    table = torch.tensor(words, dtype=torch.int64).to(torch.int32)  # wraps
    return table.to(device)


def fork_partials(
    query,
    cache,
    block_table,
    lengths,
    scales,
    out,
    lse,
    split,
    table,
    balanced=False,
    repoll=False,
    cluster=False,
    valid=False,
    valid_delay=0,
):
    """``split_partials`` with mla.core_ts's polled fork: q polled from the mailbox
    (MlaQuery does not gather), x re-armed; ``table``: ``fork_table``."""
    import tvm_ffi

    batch, _, heads, _ = query.shape
    index = query.device.index
    options = (balanced, repoll, cluster, valid, valid_delay)
    kernel = compile_fork(heads, cache.shape[1], split, batch, index, *options)
    buf = split_workspace_ts(query.device, heads, batch, split)[0]
    cache = cache.view(torch.float8_e4m3fn)
    softmax, output_scale = scales
    args = [query[..., :512], query[..., 512:], cache[..., :512], cache[..., 512:]]
    args += [block_table, out, lse, buf, Int32(split), lengths, lengths, table[:batch]]
    args += [Float32(softmax), Float32(output_scale)]
    with tvm_ffi.use_torch_stream():
        kernel(*args)


def split_partials(query, cache, block_table, lengths, scales, out, lse, split):
    """Tokenspeed's generic decode kernel at q = 1 without its reducer: FP32 split
    partials and base-2 split LSEs in ``split_workspace_ts`` for DcpValue's LSE fold
    (``out`` / ``lse`` are not written)."""
    import tvm_ffi

    batch, _, heads, _ = query.shape
    kernel = compile_split_partials(heads, cache.shape[1], split, query.device.index)
    buf = split_workspace_ts(query.device, heads, batch, split)[0]
    cache = cache.view(torch.float8_e4m3fn)
    softmax, output_scale = scales
    args = [query[..., :512], query[..., 512:], cache[..., :512], cache[..., 512:]]
    args += [block_table, out, lse, buf, Int32(split), lengths, lengths, None]
    args += [Float32(softmax), Float32(output_scale)]
    with tvm_ffi.use_torch_stream():
        kernel(*args)


def fill_splits(batch, device):
    """Splits filling the SMs (one CTA pair each); tokenspeed caps itself at 32."""
    from tokenspeed_mla import get_num_sm

    return min(256, max(1, get_num_sm(device) // (2 * batch)))


def dcp_splits(batch, device):
    """(min_split, blocks) of stock_decode for the DCP core at q = 1: WIDE_SPLITS on
    Rubin-class devices (exact), else fill_splits."""
    from tokenspeed_mla import get_num_sm

    if get_num_sm(device) > GB300_SMS and batch in WIDE_SPLITS:
        return WIDE_SPLITS[batch], 2 * batch
    return fill_splits(batch, device), None


# Rubin-class devices, DCP spec verify (q > 1, packed query tiles): token-row batches
# whose tokenspeed core runs exactly this many KV splits (tokenspeed's occupancy rule
# gives 2 at B8 q8: 96 heads x 8 rows = 6 M128 tiles per request, and its reducer then
# moves 2 x [64, 96, 512] FP32 partials). Not listed: tokenspeed's rule. B8 q8, 16
# ranks, in-proj -> MlaQuery -> core -> DcpValue, us/layer at ISL 8.8k / 64k: split 1
# (no workspace, no reducer) 52.28 / 68.23, the rule (2) 62.42 / 72.19, 3-4 slower.
# Not set: split 1 loses at long contexts; VERIFY_FOLD drops the reducer instead.
VERIFY_SPLITS: dict = {}
# Rubin-class devices, DCP spec verify (q > 1, packed query tiles): token-row batches
# whose tokenspeed core keeps tokenspeed's split count but skips its reducer; DcpValue's
# pushers LSE-fold the FP32 split partials (``lse_split`` at q > 1, as LSE_FOLD at
# q = 1). The packed workspace [requests, tiles x 128, splits, 512] holds query row t,
# global head g at packed row t * DCP * H + g. Rows 16 / 32 (DL7 B2 / B4 at q8, DL3
# B8 at q4; tokenspeed's rule gives 8 / 4 splits at 96 heads): 4 ranks, 6 heads per
# rank, local 8k, serving's split count, in-proj -> DcpValue us/layer, reducer vs fold
# (with core_q's polled / valid q): B8 q4 37.93 / 33.03, B4 q8 37.69 / 32.85, B2 q8
# 29.40 / 25.92.
VERIFY_FOLD = batch_table("16:1,24:1,32:1,40:1,48:1,56:1,64:1")


def verify_splits(batch, device):
    """(min_split, blocks) of stock_decode for the DCP core at q > 1: VERIFY_SPLITS on
    Rubin-class devices (exact: blocks 1 makes tokenspeed's occupancy rule give 1, so
    min_split decides), else fill_splits."""
    from tokenspeed_mla import get_num_sm

    if get_num_sm(device) > GB300_SMS and batch in VERIFY_SPLITS:
        return VERIFY_SPLITS[batch], 1
    return fill_splits(batch, device), None


def local_max(max_context, size, interleave=1):
    """Longest local shard of a ``max_context`` sequence (block-interleaved), in
    64-token units. ``interleave``: the KV cache's DCP interleave; a rank owns
    whole ``interleave``-token chunks, so its shard can exceed ``max_context /
    size`` by up to one chunk (e.g. 896 for a hetero-DCP P/D decoder)."""
    shard = math.ceil(max_context / (size * interleave)) * interleave
    return max(math.ceil(max_context / (size * 64)) * 64, math.ceil(shard / 64) * 64)
