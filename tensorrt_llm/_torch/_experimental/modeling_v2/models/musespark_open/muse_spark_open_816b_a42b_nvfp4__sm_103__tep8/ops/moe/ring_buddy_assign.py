# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-step expert placement for ring-buddy replicated experts (``replicated_experts``).

Under expert parallelism over ``RANKS`` ranks (``RING_SIZES``: 8 ranks of 32-expert
blocks at tep8, 4 ranks of 64-expert blocks at tep4), block ``b`` (global experts
``[BLOCK*b, BLOCK*(b+1))``) lives on rank ``b``. With ring-buddy replication
rank ``b + 1`` (mod ``RANKS``) holds a second copy of block ``b``, so every
selected expert can run on its owner or on the owner's successor. This entry
decides, from the routing of one step, which of the two runs each expert and
rewrites the step's top-k ids into the id space of one rank's stacked expert
tensor ``[own block | predecessor's block]``:

* an expert this rank runs from its own block       -> local index ``i``
* an expert this rank runs for its predecessor      -> ``BLOCK + i``
* anything else                                     -> an id in
  ``[2*BLOCK, EXPERTS)``, i.e. outside the local window of a
  ``local_expert_offset=0, local_num_experts=2*BLOCK`` expert call.

The decision depends only on the set of selected experts, which every rank
computes identically, so every rank derives the same placement and the
``(token, slot)`` pairs are partitioned across ranks: each is executed by
exactly one rank.

Placement: let ``load[b]`` be the number of distinct selected experts in block
``b`` and ``x[b]`` how many of them move to rank ``b + 1``; rank ``r`` then runs
``load[r] - x[r] + x[r-1]`` experts. A target maximum ``T`` is feasible exactly
when every arc of ``n + 1`` consecutive ranks (``n < RANKS``) can absorb the
load of its first ``n`` blocks, ``sum <= (n + 1) * T``, and the whole ring
holds ``D <= RANKS * T`` (the cut conditions of the ring flow). The kernel
takes the smallest such ``T`` and the pointwise smallest flow,
``x[r] = max(0, max_n(sum of the n blocks ending at r) - n * T))``, which every
feasible flow dominates. Within a block the ``x[b]`` present experts with the
highest local index move.

The kernel is one program that sits on the decode step's MoE critical path
when the main stream is the longer branch, so it is kept to ``[slots, RANKS]``
work: the census is one presence word per block, as wide as the block (32 bits
at tep8, 64 bits at tep4), an OR-reduction of ``1 << i`` over the slots of that
block, which also counts a repeated expert once; a block's load is the word's
population count, and an expert's rank within its block is the population count
of the bits above its own.

With ``balance=False`` and ``static_from`` at the block size every expert stays on
its owner. Those ids address a one-window call as well: the rank's own block
alone (``local_num_experts = BLOCK``), the owner-only window
``ring_buddy.install(windows=1)`` gives ``persistent_experts`` and
``bf16_routed_experts`` without replication.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from ...sm107_defaults import env as _sm107_env

#: Model geometry: 256 experts over the 8 expert-parallel ranks of tep8 (the
#: defaults), or over the 4 of tep4 (``ranks=4``).
EXPERTS = 256
RANKS = 8
BLOCK = EXPERTS // RANKS
TOP_K = 8
#: The ring sizes the kernel serves: expert-parallel ranks -> experts per block.
RING_SIZES = {8: EXPERTS // 8, 4: EXPERTS // 4}


#: Opt-in: with
#: ``MUSESPARK_OPEN_TEP8_RB_PDL=1`` the 8-rank ring's table-writing call (the TEP8 decode prologue) launches
#: as a PDL dependent of the routing kernel before it: the kernel's first statement is ``griddepcontrol.wait`` (nothing
#: of its producer's is read before it), so its launch and prologue overlap the routing kernel's tail instead of
#: following it (the top-8 -> ring buddy gap is short). Launch timing only: values unchanged.
_RB_PDL = _sm107_env("MUSESPARK_OPEN_TEP8_RB_PDL", "0") == "1"


def block_of(ranks: int) -> int:
    """Experts per block on a ring of ``ranks``.

    Raises:
        ValueError: for a ring size not in ``RING_SIZES``.
    """
    if ranks == 8:
        return RING_SIZES[8]
    elif ranks == 4:
        return RING_SIZES[4]
    raise ValueError(f"ring_buddy_assign serves rings of {sorted(RING_SIZES)} ranks, got {ranks}")


#: Largest ``M * TOP_K`` one program handles (M <= 32 decode rows: four requests'
#: K = 7 verify steps). Opt-in: 64 rows (eight requests), so ring-buddy balance
#: (``MUSESPARK_OPEN_TEP8_RING_BUDDY_MAX_ROWS=64``) also places the 33..64-row TEP8 steps; the kernel is the
#: same [SLOTS, RANKS] program at a 512-slot tile. 128 rows (16 requests,
#: ``MUSESPARK_OPEN_TEP8_RING_BUDDY_MAX_ROWS=128``) at a 1024-slot tile; smaller calls keep their tile. 256 rows
#: (``MUSESPARK_OPEN_TEP8_WIDE_BALANCE_ROWS``) on the wide program (``wide=True``).
MAX_SLOTS = 2048
#: Tiles above this many slots (more than 128 rows) run the balanced tep8 placement on the wide program when the
#: caller asks for it (``wide=True``; no routing tables). Per-expert program:
#: per-expert instead of ``[SLOTS, RANKS]`` work, about 2 us at 1024 / 2048 slots on GB300.
WIDE_MIN_SLOTS = 1024
WIDE_NUM_WARPS = 8

#: The half window (``half_window_experts``, tep4): a block's upper half starts at this
#: local index, and the stack ``[own block | the predecessor's upper half]`` holds
#: this many local experts.
HALF_FROM = 32
HALF_SPAN = 96


#: Warps of the one program at every slot tile. The kernel is a chain of small
#: reductions, so more warps shorten each step's per-thread serial work more
#: than they add cross-warp exchanges up to four (GB300, CUDA graph of
#: back-to-back launches, us per launch at 1 / 2 / 4 / 8 warps: 16 slots
#: 1.52 / 1.52 / 1.52 / 1.60, 64 slots 2.21 / 1.86 / 1.65 / 1.72, 128 slots
#: 3.08 / 2.37 / 1.73 / 2.15; a one-kernel BF16 cast of the weights is 0.8).
NUM_WARPS = 4

#: Opt-in: a call with
#: ``meta_tile`` also writes the routing tables TRTLLM-gen's expert op reads for the two-window
#: ``[own block | predecessor's block]`` call (``local_expert_offset=0``, ``local_num_experts=2*BLOCK``):
#: expanded->permuted (``e2p``, -1 where the slot is not this rank's), permuted->token (``p2t``), every
#: CTA's expert and row limit (``c2b`` / ``c2m``), the CTA count (``nnc``) and the padded row count
#: (``tnp``), for ``meta_tile``-token tiles. An expert's rows are ordered by token (the routing kernel
#: orders them by arrival); a row's top-k ids are distinct, so a 64-bit presence word per row gives
#: every expert's count and every slot's position. ``ring_buddy`` (``MUSESPARK_OPEN_TEP8_PREROUTED=1``)
#: then runs FC1 / FC2 on these tables without the routing kernel.
META_FIELDS = ("e2p", "p2t", "c2b", "c2m", "nnc", "tnp")
#: With ``MUSESPARK_OPEN_TEP8_PREROUTED_TRIGGER=1`` the program fences its table writes and lets the
#: expert call's FC1 launch (PDL) at its end instead of at its completion.
_META_TRIGGER = _sm107_env("MUSESPARK_OPEN_TEP8_PREROUTED_TRIGGER", "") == "1"
#: Warps of the program when it also writes the tables (lab, cold chain at 64 / 48 / 32 / 16 rows: 8 warps
#: -3.09 / -2.86 / -2.54 / -1.73 us against the routing-kernel chain, 4 warps -2.43 / -2.48 / -2.83 / -1.63).
_META_WARPS = int(_sm107_env("MUSESPARK_OPEN_TEP8_PREROUTED_WARPS", "8"))


#: Opt-in: ``MUSESPARK_OPEN_TEP8_TILE_PLACEMENT=1``
#: places the TEP8 prerouted calls (``meta_tile``, balanced, two windows) of at least
#: ``MUSESPARK_OPEN_TEP8_TILE_PLACEMENT_MIN_ROWS`` rows (default 33: bs5-8) by expert cost instead of expert count.
#: An expert's cost is ``A + B * (tiles - 1)`` (tiles of ``meta_tile`` tokens;
#: ``MUSESPARK_OPEN_TEP8_TILE_PLACEMENT_COST`` = ``A,B`` in 0.1 us, default 20,11: FP4 FC1 + FC2 stream an expert in
#: ~2.0 us and add ~1.1 us per extra tile). The ring flow is the count rule's on those costs; block ``b`` then moves its
#: present experts in descending local index while the moved cost stays short of its outflow, the last one only if it
#: lands closer. Every rank derives the same placement (integer arithmetic on the same ids), so each slot still runs on
#: exactly one rank; only which experts move changes, so it is accuracy-equivalent to the count rule, not bitwise
#: (per-rank finalize partials). Unset: the count rule.
_TILE_PLACEMENT = _sm107_env("MUSESPARK_OPEN_TEP8_TILE_PLACEMENT", "") == "1"
_TILE_MIN_ROWS = int(_sm107_env("MUSESPARK_OPEN_TEP8_TILE_PLACEMENT_MIN_ROWS", "33") or 33)
_TILE_COST = tuple(
    int(v) for v in _sm107_env("MUSESPARK_OPEN_TEP8_TILE_PLACEMENT_COST", "20,11").split(",")
)
if _TILE_PLACEMENT:
    try:
        from tensorrt_llm.logger import logger as _logger

        _logger.info(
            f"[musespark_open] tep8 tile placement (MUSESPARK_OPEN_TEP8_TILE_PLACEMENT=1): prerouted calls of "
            f">= {_TILE_MIN_ROWS} rows place experts by cost {_TILE_COST[0]} + {_TILE_COST[1]} x (tiles - 1) "
            "(0.1 us) on the two-window ring"
        )
    except Exception:  # noqa: BLE001 - logging only
        pass


def expert_cost(tokens: int, tile: int, cost=_TILE_COST) -> int:
    """Tile placement: an expert's placement cost (0.1 us) for ``tokens`` tokens in ``tile``-token tiles.

    0 when absent.
    """
    if tokens <= 0:
        return 0
    return cost[0] + cost[1] * ((tokens + tile - 1) // tile - 1)


def meta_sizes(rows: int, window: int, tile: int) -> Dict[str, int]:
    """Element counts of the routing tables of a ``rows``-row call over ``window`` experts with ``tile``-token
    tiles (TRT-LLM's ``getMaxNumCtasInBatchDim``; the permuted rows stay below CTAs x tile)."""
    slots = rows * TOP_K
    filled = min(window, slots)
    ctas = filled + (slots - filled) // tile
    return dict(e2p=slots, p2t=ctas * tile + 1, c2b=ctas, c2m=ctas, nnc=1, tnp=1)


@triton.jit
def _fence_acq_rel_gpu(x):
    """``fence.acq_rel.gpu`` on every thread (``x``: any int32 scalar, returned)."""
    return tl.inline_asm_elementwise(
        "fence.acq_rel.gpu; mov.u32 $0, $1;", "=r,r", [x], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _bit_or(a, b):
    return a | b


@triton.jit
def _popc32(v):
    """Set bits of each uint32 lane (SWAR population count), as int32."""
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _popc(v, WIDE: tl.constexpr):
    """Set bits of each presence word (uint32, or uint64 when ``WIDE``), as int32."""
    if WIDE:
        return _popc32(v.to(tl.uint32)) + _popc32((v >> 32).to(tl.uint32))
    else:
        return _popc32(v)


@triton.jit
def _ring_buddy_assign_kernel(
    selected_ptr,  # [num_slots] int32 global expert ids
    weights_ptr,  # [num_slots] float32 routing weights
    local_ids_ptr,  # [num_slots] int32 out
    weights_bf16_ptr,  # [num_slots] bfloat16 out
    stats_ptr,  # [2 * RANKS] int32 out: per-rank loads before, after
    rank,
    num_slots,
    e2p_ptr,  # META: [num_slots] int32 out, expanded -> permuted row (-1: not this rank's)
    p2t_ptr,  # [CTAS_MAX * TILE + 1] int32 out, permuted row -> token
    c2b_ptr,  # [CTAS_MAX] int32 out, CTA -> local expert
    c2m_ptr,  # [CTAS_MAX] int32 out, CTA -> row limit (exclusive)
    nnc_ptr,  # [1] int32 out, CTAs with rows
    tnp_ptr,  # [1] int32 out, padded permuted rows
    scratch_ptr,  # [4 * BLOCK] int32, the per-expert counters (zeroed here)
    SLOTS: tl.constexpr,
    EXPERTS: tl.constexpr,
    RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
    BALANCE: tl.constexpr,
    STATIC_FROM: tl.constexpr,
    HALF: tl.constexpr,
    META: tl.constexpr,
    TILE: tl.constexpr,
    CTAS: tl.constexpr,
    TRIGGER: tl.constexpr,
    TILEW: tl.constexpr = False,
    COST_A: tl.constexpr = 20,
    COST_B: tl.constexpr = 11,
    PDL: tl.constexpr = False,
):
    if PDL:
        # RB_PDL: wait for the routing kernel (a PDL predecessor) before reading anything it wrote
        gdc_wait()
    tl.static_assert(BLOCK == 32 or BLOCK == 64, "one 32- or 64-bit presence word per block")
    tl.static_assert(not HALF or (BLOCK == 64 and BALANCE), "the half window: tep4, balanced")
    WIDE: tl.constexpr = BLOCK == 64
    UPPER: tl.constexpr = BLOCK // 2  # HALF_FROM
    SPAN: tl.constexpr = BLOCK + BLOCK // 2  # HALF_SPAN
    s = tl.arange(0, SLOTS)
    smask = s < num_slots
    g = tl.load(selected_ptr + s, mask=smask, other=0)
    w = tl.load(weights_ptr + s, mask=smask, other=0.0)
    blk = g // BLOCK
    i = g % BLOCK
    ranks = tl.arange(0, RANKS)

    # Census: one presence word per block, bit i set when local expert i is
    # selected by any slot; a duplicate id sets the same bit.
    if WIDE:
        shift = i.to(tl.uint64)
        bit = tl.full([SLOTS], 1, tl.uint64) << shift
        no_bits = tl.zeros([SLOTS, RANKS], dtype=tl.uint64)
    else:
        shift = i.to(tl.uint32)
        bit = tl.full([SLOTS], 1, tl.uint32) << shift
        no_bits = tl.zeros([SLOTS, RANKS], dtype=tl.uint32)
    in_rank = (blk[:, None] == ranks[None, :]) & smask[:, None]  # [SLOTS, RANKS]
    word = tl.reduce(tl.where(in_rank, bit[:, None], no_bits), 0, _bit_or)  # [RANKS]
    load = _popc(word, WIDE)  # [RANKS]

    if BALANCE:
        # arc[r, n-1]: load of the n blocks ending at r (n = 1..RANKS).
        span = ranks + 1
        back = (ranks[:, None] - ranks[None, :] + RANKS) % RANKS  # [r, k]
        inarc = back[:, None, :] < span[None, :, None]  # [r, n, k]
        arc = tl.sum(tl.where(inarc, load[None, None, :], 0), axis=2)  # [r, n]
        need = tl.where(
            span[None, :] == RANKS,
            (arc + RANKS - 1) // RANKS,
            (arc + span[None, :]) // (span[None, :] + 1),
        )
        if HALF:
            # Only block r's upper half crosses the edge r -> r + 1, so every arc of n
            # blocks ending at r also needs T >= ceil((arc - movable[r]) / n)
            # (``ring_windows.half_smallest_max``); arc >= movable[r].
            movable = _popc(word >> UPPER, WIDE)  # [RANKS]
            need = tl.maximum(need, (arc - movable[:, None] + span[None, :] - 1) // span[None, :])
        t = tl.max(tl.max(need, axis=1), axis=0)
        x = tl.max(tl.maximum(arc - span[None, :] * t, 0), axis=1)  # [RANKS]
    elif STATIC_FROM < BLOCK:
        # Fixed placement: local experts STATIC_FROM.. of every block run on
        # the successor, whatever the step selects.
        x = _popc(word >> STATIC_FROM, WIDE)
    else:
        x = tl.zeros([RANKS], dtype=tl.int32)

    if TILEW:
        # Tile placement: the same ring flow on expert costs (A + B * (tiles - 1), 0.1 us) and the movers of each block
        # in descending local index while the moved cost stays short of the block's outflow (the last one only if the
        # outflow is then overshot by no more than it was missed). Pure integer arithmetic on the step's ids.
        tl.static_assert(BALANCE and not HALF, "tile placement: the balanced two-window placement")
        hist = tl.histogram(
            tl.where(smask, g, EXPERTS), 2 * EXPERTS
        )  # [2E]; bin E takes the masked slots
        cnt2 = tl.reshape(hist, (2, EXPERTS))
        first = tl.arange(0, 2)[:, None] == 0
        cnt = tl.sum(tl.where(first, cnt2, 0), axis=0)  # [EXPERTS] tokens per expert
        tiles_e = (cnt + TILE - 1) // TILE
        cost_e = tl.where(cnt > 0, COST_A + COST_B * (tiles_e - 1), 0)  # [EXPERTS]
        cost_rb = tl.reshape(cost_e, (RANKS, BLOCK))  # [block, local index]
        wload = tl.sum(cost_rb, axis=1)  # [RANKS]
        wspan = ranks + 1
        wback = (ranks[:, None] - ranks[None, :] + RANKS) % RANKS
        winarc = wback[:, None, :] < wspan[None, :, None]
        warc = tl.sum(tl.where(winarc, wload[None, None, :], 0), axis=2)  # [r, n]
        wneed = tl.where(
            wspan[None, :] == RANKS,
            (warc + RANKS - 1) // RANKS,
            (warc + wspan[None, :]) // (wspan[None, :] + 1),
        )
        wt = tl.max(tl.max(wneed, axis=1), axis=0)
        wx = tl.max(
            tl.maximum(warc - wspan[None, :] * wt, 0), axis=1
        )  # [RANKS] cost to move out of each block
        after_c = tl.cumsum(
            cost_rb, axis=1, reverse=True
        )  # cost of the experts at or above each local index
        before_c = after_c - cost_rb
        wxb = wx[:, None]
        move_t = (
            (cost_rb > 0) & (before_c < wxb) & (2 * wxb >= before_c + after_c)
        )  # [RANKS, BLOCK]
        # The count rule's movers on the same grid (present experts with fewer than x[b] present above them); the
        # step takes the tile rule only where its busiest rank costs less than the count rule's (ties: count).
        pres = (cost_rb > 0).to(tl.int32)
        above_n = tl.cumsum(pres, axis=1, reverse=True) - pres
        move_c = (pres > 0) & (above_n < x[:, None])
        from_prev_r = ((ranks[:, None] + RANKS - 1) % RANKS) == ranks[
            None, :
        ]  # [r, b]: block b moves onto r
        stay_t = tl.sum(tl.where(move_t, 0, cost_rb), axis=1)
        out_t = tl.sum(tl.where(move_t, cost_rb, 0), axis=1)
        stay_c = tl.sum(tl.where(move_c, 0, cost_rb), axis=1)
        out_c = tl.sum(tl.where(move_c, cost_rb, 0), axis=1)
        max_t = tl.max(stay_t + tl.sum(tl.where(from_prev_r, out_t[None, :], 0), axis=1), axis=0)
        max_c = tl.max(stay_c + tl.sum(tl.where(from_prev_r, out_c[None, :], 0), axis=1), axis=0)
        move = tl.where(max_t < max_c, move_t, move_c)
        moved_e = tl.reshape(move.to(tl.int32), (EXPERTS,))
        mv = tl.gather(moved_e, g, 0)  # [SLOTS]
        er = (blk + mv) % RANKS
        x = tl.sum(move.to(tl.int32), axis=1)  # [RANKS] experts moved (for the stats)
    else:
        # The x[b] present experts of block b with the highest local index move:
        # a slot's expert moves when fewer than x[b] present experts of its block
        # sit above it.
        word_s = tl.reduce(tl.where(in_rank, word[None, :], no_bits), 1, _bit_or)  # [SLOTS]
        above = _popc((word_s >> shift) >> 1, WIDE)
        x_blk = tl.sum(tl.where(in_rank, x[None, :], 0), axis=1)  # [SLOTS]
        er = (blk + (above < x_blk).to(tl.int32)) % RANKS

    if HALF:
        # A moved expert lies in its block's upper half, whose copy sits at local ids
        # BLOCK..HALF_SPAN-1 (``ring_windows.place_half``).
        local = tl.where(
            er == rank,
            tl.where(blk == rank, i, BLOCK - UPPER + i),
            SPAN + g % (EXPERTS - SPAN),
        )
    else:
        local = tl.where(
            er == rank,
            tl.where(blk == rank, i, BLOCK + i),
            2 * BLOCK + g % (EXPERTS - 2 * BLOCK),
        )
    tl.store(local_ids_ptr + s, local, mask=smask)
    tl.store(weights_bf16_ptr + s, w.to(tl.bfloat16), mask=smask)

    if META:
        # The routing tables of the two-window call (WIN local experts), TILE-token tiles. Counts by
        # histogram; a slot's position within its expert from one counter per expert (atomics, so an
        # expert's rows are in arrival order, as the routing kernel orders them); CTA tables by expert.
        WIN: tl.constexpr = 2 * BLOCK
        BINS: tl.constexpr = 2 * WIN  # bin WIN takes every slot that is not this rank's
        MAXT: tl.constexpr = (SLOTS // 8 + TILE - 1) // TILE  # tiles of an expert every row selects
        valid = smask & (local < WIN)
        eid = tl.where(valid, local, WIN)
        ev = tl.arange(0, BINS)
        cnt = tl.where(ev < WIN, tl.histogram(eid, BINS), 0)
        tiles = (cnt + TILE - 1) // TILE
        cta_end = tl.cumsum(tiles, axis=0)
        cta_off = cta_end - tiles
        nnc = tl.sum(tiles, axis=0)
        tl.store(scratch_ptr + ev, tl.zeros([BINS], dtype=tl.int32))
        tl.debug_barrier()
        pos = tl.atomic_add(scratch_ptr + eid, 1, mask=valid, sem="relaxed", scope="cta")
        e2p = tl.where(valid, tl.gather(cta_off, eid, 0) * TILE + pos, -1)
        tl.store(e2p_ptr + s, e2p, mask=smask)
        tl.store(p2t_ptr + e2p, s // 8, mask=valid)
        tt = tl.arange(0, MAXT)
        cta = cta_off[:, None] + tt[None, :]  # [BINS, MAXT]
        cmask = tt[None, :] < tiles[:, None]
        tl.store(c2b_ptr + cta, ev[:, None] + tt[None, :] * 0, mask=cmask)
        tl.store(
            c2m_ptr + cta, tl.minimum((cta + 1) * TILE, (cta_off * TILE + cnt)[:, None]), mask=cmask
        )
        tl.store(nnc_ptr, nnc)
        tl.store(tnp_ptr, nnc * TILE)
        if TRIGGER:
            # Every table write is visible GPU-wide before FC1, whose prologue reads the tables before it
            # waits on this grid, may launch.
            tl.debug_barrier()
            _fence_acq_rel_gpu(nnc)
            gdc_launch_dependents()

    from_prev = ((ranks[:, None] + RANKS - 1) % RANKS) == ranks[None, :]
    x_prev = tl.sum(tl.where(from_prev, x[None, :], 0), axis=1)
    tl.store(stats_ptr + ranks, load)
    tl.store(stats_ptr + RANKS + ranks, load - x + x_prev)


def place_tile_reference(flat: List[int], ranks: int, tile: int, cost=_TILE_COST) -> List[int]:
    """The tile placement in plain Python: for each slot's expert, the rank that runs it (the kernel's integer rule)."""
    size = block_of(ranks)
    tokens = {}
    for e in flat:
        tokens[e] = tokens.get(e, 0) + 1
    cost_e = [expert_cost(tokens.get(e, 0), tile, cost) for e in range(EXPERTS)]
    wload = [sum(cost_e[b * size : (b + 1) * size]) for b in range(ranks)]
    arc = [
        [sum(wload[(r - k) % ranks] for k in range(n)) for n in range(1, ranks + 1)]
        for r in range(ranks)
    ]
    t = 0
    for r in range(ranks):
        for n in range(1, ranks + 1):
            need = -(-arc[r][n - 1] // ranks) if n == ranks else -(-arc[r][n - 1] // (n + 1))
            t = max(t, need)
    x = [max(0, max(arc[r][n - 1] - n * t for n in range(1, ranks + 1))) for r in range(ranks)]
    moved_t = set()
    for b in range(ranks):
        acc = 0
        for i in range(size - 1, -1, -1):
            e = b * size + i
            c = cost_e[e]
            if c == 0:
                continue
            before, after = acc, acc + c
            if before < x[b] and 2 * x[b] >= before + after:
                moved_t.add(e)
            acc = after
    # the count rule (balance_ring on distinct counts, highest local indexes move)
    load = [sum(1 for e in range(b * size, (b + 1) * size) if cost_e[e]) for b in range(ranks)]
    xc = balance_ring(load)[1]
    moved_c = set()
    for b in range(ranks):
        in_block = [e for e in range(b * size, (b + 1) * size) if cost_e[e]]
        moved_c.update(sorted(in_block, reverse=True)[: xc[b]])

    def busiest(moved):
        c = [0] * ranks
        for e in range(EXPERTS):
            if cost_e[e]:
                c[(e // size + (e in moved)) % ranks] += cost_e[e]
        return max(c)

    moved = moved_t if busiest(moved_t) < busiest(moved_c) else moved_c
    return [(e // size + (e in moved)) % ranks for e in flat]


def balance_ring(load: List[int]) -> Tuple[int, List[int]]:
    """The kernel's placement in plain Python: ``(max load, x)``.

    ``x[b]`` experts of block ``b`` move to rank ``b + 1``: the smallest
    feasible maximum and the smallest flow that reaches it.
    """
    ranks = len(load)
    arc = [
        [sum(load[(r - k) % ranks] for k in range(n)) for n in range(1, ranks + 1)]
        for r in range(ranks)
    ]
    t = -(-sum(load) // ranks)
    for r in range(ranks):
        for n in range(1, ranks):
            t = max(t, -(-arc[r][n - 1] // (n + 1)))
    x = [max(0, max(arc[r][n - 1] - n * t for n in range(1, ranks + 1))) for r in range(ranks)]
    return t, x


@triton.jit
def _ring_buddy_assign_wide_kernel(
    selected_ptr,  # [num_slots] int32 global expert ids
    weights_ptr,  # [num_slots] float32 routing weights
    local_ids_ptr,  # [num_slots] int32 out
    weights_bf16_ptr,  # [num_slots] bfloat16 out
    stats_ptr,  # [2 * RANKS] int32 out: per-rank loads before, after
    rank,
    num_slots,
    SLOTS: tl.constexpr,
    EXPERTS: tl.constexpr,
    RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Wide program: ``_ring_buddy_assign_kernel``'s balanced two-window
    tep8 placement with per-expert instead of ``[SLOTS, RANKS]`` work. The census is a histogram of the step's ids, the
    move decision is made once per expert, and each slot gathers its expert's local id: every output is the same
    integer function of ``selected`` and ``rank`` (device gate: == ``reference``)."""
    tl.static_assert(BLOCK == 32 and RANKS * BLOCK == EXPERTS, "the tep8 ring of 32-expert blocks")
    s = tl.arange(0, SLOTS)
    smask = s < num_slots
    g = tl.load(selected_ptr + s, mask=smask, other=0)
    w = tl.load(weights_ptr + s, mask=smask, other=0.0)
    experts = tl.arange(0, EXPERTS)
    ranks = tl.arange(0, RANKS)
    # Census: per-expert slot counts (the masked-off slots load expert 0); a block's presence word sets bit i for each
    # of its selected experts.
    counts = tl.histogram(g, EXPERTS) - tl.where(experts == 0, SLOTS - num_slots, 0)
    i_e = experts % BLOCK
    bit_e = tl.where(
        counts > 0,
        tl.full([EXPERTS], 1, tl.uint32) << i_e.to(tl.uint32),
        tl.zeros([EXPERTS], dtype=tl.uint32),
    )
    word = tl.reduce(tl.reshape(bit_e, [RANKS, BLOCK]), 1, _bit_or)  # [RANKS] uint32
    load = _popc(word, False)  # [RANKS]
    # The balanced placement's smallest maximum T and smallest flow x (as the narrow program).
    span = ranks + 1
    back = (ranks[:, None] - ranks[None, :] + RANKS) % RANKS  # [r, k]
    inarc = back[:, None, :] < span[None, :, None]  # [r, n, k]
    arc = tl.sum(tl.where(inarc, load[None, None, :], 0), axis=2)  # [r, n]
    need = tl.where(
        span[None, :] == RANKS,
        (arc + RANKS - 1) // RANKS,
        (arc + span[None, :]) // (span[None, :] + 1),
    )
    t = tl.max(tl.max(need, axis=1), axis=0)
    x = tl.max(tl.maximum(arc - span[None, :] * t, 0), axis=1)  # [RANKS]
    # Per expert: it moves to its block's successor when fewer than x[b] selected experts of its block sit above it.
    word_e = tl.reshape(tl.broadcast_to(word[:, None], [RANKS, BLOCK]), [EXPERTS])
    x_e = tl.reshape(tl.broadcast_to(x[:, None], [RANKS, BLOCK]), [EXPERTS])
    blk_e = experts // BLOCK
    above_e = _popc((word_e >> i_e.to(tl.uint32)) >> 1, False)
    er_e = (blk_e + (above_e < x_e).to(tl.int32)) % RANKS
    local_e = tl.where(
        er_e == rank,
        tl.where(blk_e == rank, i_e, BLOCK + i_e),
        2 * BLOCK + experts % (EXPERTS - 2 * BLOCK),
    )
    local = tl.gather(local_e, g, 0)  # [SLOTS]
    tl.store(local_ids_ptr + s, local, mask=smask)
    tl.store(weights_bf16_ptr + s, w.to(tl.bfloat16), mask=smask)
    from_prev = ((ranks[:, None] + RANKS - 1) % RANKS) == ranks[None, :]
    x_prev = tl.sum(tl.where(from_prev, x[None, :], 0), axis=1)
    tl.store(stats_ptr + ranks, load)
    tl.store(stats_ptr + RANKS + ranks, load - x + x_prev)


class RingBuddyAssign(OpWrapper):
    """Place one step's selected experts on owner or successor; rewrite ids for one rank.

    Takes the routing method's ``selected`` ids ``[M, 8]`` (int32, global) and
    its fp32 ``weights``, this rank's index and the ring size ``ranks`` (8, the
    default, or 4; the block is ``256 / ranks`` experts). Returns ``(local_ids
    [M, 8] int32, weights_bf16 [M, 8], stats [2 * ranks] int32)``: the ids in
    this rank's stacked expert space (see the module docstring), the weights
    cast to the BF16 the TRTLLM-gen expert op requires, and the per-rank
    distinct-expert loads before (``stats[:ranks]``) and after
    (``stats[ranks:]``) placement. ``balance=False`` places every expert on its
    owner, which reproduces plain expert parallelism through the stacked layout
    (or the owner-only window); with ``balance=False`` and ``static_from`` below
    the block size, local experts ``static_from..`` of every block run on the
    successor at every step instead (a routing-independent regrouping, the
    static placement). ``static_from`` defaults to the block size. ``half=True``
    (tep4, ``half_window_experts``) is the balanced placement onto the successor's
    copy of each block's upper half only (``ring_windows.place_half``): its ids
    address the 96-expert ``[own block | predecessor's experts 32..63]`` stack.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Pure function of ``selected`` and ``rank``; one program, so it is meant for
    decode rows (``M * 8 <= 256``). Duplicate ids within a row are not expected
    (top-k selection is distinct) and are counted once in the census.

    The rewritten ids are meaningful only for an expert call with
    ``local_expert_offset=0`` and ``local_num_experts=2*BLOCK`` over a stacked
    ``[own block | predecessor's block]`` weight tensor (64 experts at tep8, 128
    at tep4), or, for the owner-only placement, ``local_num_experts=BLOCK`` over
    the rank's own block; any other window would execute experts this rank was
    not assigned.
    """

    def __call__(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        rank: int,
        balance: bool = True,
        static_from: Optional[int] = None,
        ranks: int = RANKS,
        half: bool = False,
        meta_tile: Optional[int] = None,
        wide: bool = False,
    ):
        block = block_of(ranks)
        static_from = block if static_from is None else static_from
        num_slots = selected.numel()
        local_ids = torch.empty_like(selected, dtype=torch.int32)
        weights_bf16 = torch.empty(selected.shape, dtype=torch.bfloat16, device=selected.device)
        stats = torch.empty(2 * ranks, dtype=torch.int32, device=selected.device)
        slots = max(16, triton.next_power_of_2(num_slots))
        if (
            wide
            and meta_tile is None
            and slots > WIDE_MIN_SLOTS
            and ranks == 8
            and balance
            and not half
            and static_from == block
        ):
            # The balanced tep8 placement of a > 128-row call on the per-expert program.
            _ring_buddy_assign_wide_kernel[(1,)](
                selected,
                weights,
                local_ids,
                weights_bf16,
                stats,
                rank,
                num_slots,
                SLOTS=slots,
                EXPERTS=EXPERTS,
                RANKS=ranks,
                BLOCK=block,
                num_warps=WIDE_NUM_WARPS,
            )
            return local_ids, weights_bf16, stats
        meta = None
        ctas = 16
        if meta_tile is not None:
            # The two-window call's routing tables, written by the same program.
            rows = selected.shape[0]
            sizes = meta_sizes(rows, 2 * block, meta_tile)
            meta = {
                name: torch.empty(count, dtype=torch.int32, device=selected.device)
                for name, count in sizes.items()
            }
            scratch = torch.empty(4 * block, dtype=torch.int32, device=selected.device)
            meta["e2p"] = meta["e2p"].view(rows, TOP_K)
            meta["rows"] = rows
            meta["tile"] = meta_tile
            ctas = max(16, triton.next_power_of_2(sizes["c2b"]))
        dummy = stats
        tilew = (
            _TILE_PLACEMENT
            and meta is not None
            and balance
            and not half
            and ranks == 8
            and selected.shape[0] >= _TILE_MIN_ROWS
        )
        pdl = _RB_PDL and meta is not None and ranks == 8
        _ring_buddy_assign_kernel[(1,)](
            selected,
            weights,
            local_ids,
            weights_bf16,
            stats,
            rank,
            num_slots,
            *(
                (meta[name] for name in META_FIELDS)
                if meta is not None
                else (dummy,) * len(META_FIELDS)
            ),
            scratch if meta is not None else dummy,
            SLOTS=slots,
            EXPERTS=EXPERTS,
            RANKS=ranks,
            BLOCK=block,
            BALANCE=balance or half,
            STATIC_FROM=static_from,
            HALF=half,
            META=meta is not None,
            TILE=meta_tile if meta is not None else 8,
            CTAS=ctas,
            TRIGGER=meta is not None and _META_TRIGGER,
            TILEW=tilew,
            COST_A=_TILE_COST[0],
            COST_B=_TILE_COST[1],
            PDL=pdl,
            num_warps=_META_WARPS if meta is not None else NUM_WARPS,
            launch_pdl=pdl,
        )
        if meta is not None:
            return local_ids, weights_bf16, stats, meta
        return local_ids, weights_bf16, stats

    def reference(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        rank: int,
        balance: bool = True,
        static_from: Optional[int] = None,
        ranks: int = RANKS,
        half: bool = False,
        meta_tile: Optional[int] = None,
        wide: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The placement in plain Python over the host copy of ``selected``
        (``ring_windows.place_half`` with ``half``)."""
        if half:
            from ...kernels.replicated_experts import windows as _windows

            flat = [int(v) for v in selected.reshape(-1).tolist()]
            local, stats = _windows.place_half(flat, rank)
            device = selected.device
            return (
                torch.tensor(local, dtype=torch.int32, device=device).reshape(selected.shape),
                weights.to(torch.bfloat16),
                torch.tensor(stats, dtype=torch.int32, device=device),
            )
        size = block_of(ranks)
        static_from = size if static_from is None else static_from
        flat = [int(v) for v in selected.reshape(-1).tolist()]
        present = sorted(set(flat))
        load = [0] * ranks
        for expert in present:
            load[expert // size] += 1
        if balance:
            x = balance_ring(load)[1]
        else:
            x = [
                sum(1 for e in present if e // size == b and e % size >= static_from)
                for b in range(ranks)
            ]
        moved = set()
        for b in range(ranks):
            in_block = [e for e in present if e // size == b]
            moved.update(sorted(in_block, reverse=True)[: x[b]])
        local = []
        for expert in flat:
            owner, index = divmod(expert, size)
            runs_on = (owner + (expert in moved)) % ranks
            if runs_on != rank:
                local.append(2 * size + expert % (EXPERTS - 2 * size))
            elif owner == rank:
                local.append(index)
            else:
                local.append(size + index)
        after = [load[r] - x[r] + x[(r - 1) % ranks] for r in range(ranks)]
        device = selected.device
        return (
            torch.tensor(local, dtype=torch.int32, device=device).reshape(selected.shape),
            weights.to(torch.bfloat16),
            torch.tensor(load + after, dtype=torch.int32, device=device),
        )

    def is_valid(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        rank: int,
        balance: bool = True,
        static_from: Optional[int] = None,
        ranks: int = RANKS,
        half: bool = False,
        meta_tile: Optional[int] = None,
        wide: bool = False,
    ) -> None:
        """Inputs the kernel would read past or misplace without an error."""
        assert meta_tile is None or (not half and meta_tile in (8, 16, 32, 64)), (
            "prerouted tables: the two-window call, 8..64-token tiles"
        )
        assert not half or ranks == 4, "the half window is a tep4 placement"
        assert ranks in RING_SIZES, f"ranks must be one of {sorted(RING_SIZES)}, got {ranks}"
        block = block_of(ranks)
        static_from = block if static_from is None else static_from
        assert selected.is_cuda and selected.dtype == torch.int32 and selected.is_contiguous(), (
            "selected must be contiguous int32 on the device; the kernel reads raw words"
        )
        assert selected.ndim == 2 and selected.shape[1] == TOP_K, (
            f"selected must be [M, {TOP_K}], got {tuple(selected.shape)}"
        )
        assert 0 < selected.numel() <= MAX_SLOTS, (
            f"one program covers at most {MAX_SLOTS} slots (M <= {MAX_SLOTS // TOP_K}), got {selected.numel()}"
        )
        assert weights.dtype == torch.float32 and weights.is_contiguous(), (
            "weights must be contiguous float32"
        )
        assert weights.shape == selected.shape, "weights and selected must have one shape"
        assert 0 <= static_from <= block, f"static_from must be in [0, {block}]"
        assert 0 <= rank < ranks, f"rank must be in [0, {ranks}), got {rank}"


ring_buddy_assign = RingBuddyAssign()
