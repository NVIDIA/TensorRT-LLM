# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The packed verify core's per-step work list (``core_ts.polled_core`` ``wlist``).

The persistent core's CTA pair p runs work blocks p, p + P, p + 2P, ... (P pairs). With
the work list, block j runs the item in word j of a per-step list instead of
tokenspeed's static item j: q tile | request << 8 | split << 16 (WLIST_EMPTY: none).
The items and their K-tile partitions are the static schedule's (``dcp.long_per``), so
every partial and the merged rows are bitwise the same; only the order and the pair each
item runs on move.

An AgentX batch mixes 3k- and 760k-token contexts. The static order puts a long
request's items on whatever pairs its linear indices hit, and the core waits for the
most loaded pair. The list is a longest-processing-time schedule in rounds: items
sorted by cost (K tiles + a fixed per-item overhead), each round of P items placed onto
the pairs in ascending load order (the largest item onto the least loaded pair). On the
AgentX mixes it matches the sequential greedy LPT. 1 GPU, the core alone (q8, 96
heads, a prototype's long splits, static order -> list), us: mix8 64.9 -> 41.8, mix16
114.6 -> 73.7; the list itself costs 3-10 us per step (one kernel, ~22 us at 1536
items).

All MLA layers of a step share the request lengths, so the first MLA layer builds the
list (one small kernel) and every layer's core reads it."""

import torch
import triton
import triton.language as tl

from ..common.policy import batch_table

# Packed verify rows whose core runs the work list where mla.dcp.LONG_SPLIT launches its
# long split (persistent). Rubin, DCP16.
# (68: DL3 B17; 260-768: DL3 B65-B192 at max-num-seqs 192)
WLIST = batch_table("24:1,32:1,40:1,48:1,56:1,64:1,68-768:1")

# The empty item: split 0x7FFF of request 0 (no K tiles in any partition).
WLIST_EMPTY = 0x7FFF << 16
# Cost of an item in quarter K tiles: 4 per K tile plus WLIST_OVERHEAD (the prologue and
# the FP32 partial store, ~2.5 tiles of the mainloop's 0.74 us).
WLIST_OVERHEAD = 10


@triton.jit
def _wlist_kernel(
    lens,
    out,
    scratch,
    NT: tl.constexpr,
    NREQ: tl.constexpr,
    SPLIT: tl.constexpr,
    SHORT: tl.constexpr,
    THR: tl.constexpr,
    QUANT: tl.constexpr,
    P: tl.constexpr,
    PB: tl.constexpr,
    NSLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUNDS: tl.constexpr,
    OVER: tl.constexpr,
    EMPTY: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    in_range = i < NSLOTS
    tl.store(out + i, tl.full([BLOCK], EMPTY, tl.int32), mask=in_range)
    # tokenspeed's static order: q tile fastest, then request, then split
    b = (i // NT) % NREQ
    k = i // (NT * NREQ)
    n = tl.load(lens + b, mask=in_range, other=0)
    tiles = (n + 127) // 128
    # dcp.long_per: the short split's partition up to THR tiles, else SPLIT (QUANT 0) or
    # min(SPLIT, ceil(tiles / QUANT)) equal splits
    short = tl.maximum((tiles + SHORT - 1) // SHORT, 1)
    splits = SPLIT
    if QUANT > 0:
        splits = tl.minimum(SPLIT, (tiles + QUANT - 1) // QUANT)
        splits = tl.maximum(splits, 1)
    per = tl.maximum((tiles + splits - 1) // splits, 1)
    per = tl.where(tiles > THR, per, short)
    count = tl.minimum(tiles, (k + 1) * per) - k * per
    count = tl.where(in_range, tl.maximum(count, 0), 0)
    cost = tl.where(count > 0, count * 4 + OVER, 0)
    # longest first: (cost, slot) packed, slot in the low 12 bits (NSLOTS <= 4096)
    packed = tl.sort(cost * 4096 + i, descending=True)
    tl.store(scratch + i, packed)
    live = tl.sum((count > 0).to(tl.int32))
    p = tl.arange(0, PB)
    load = tl.where(p < P, 0, 1 << 20)  # padded pairs never chosen
    bins = scratch + BLOCK
    tl.debug_barrier()
    for r in tl.static_range(ROUNDS):
        j = r * P + p
        ok = (p < P) & (j < live)
        item = tl.load(scratch + j, mask=ok, other=0)
        # pairs by ascending load; a pair whose slot of this round is past the list
        # (the persistent loop visits blocks < NSLOTS only) is never chosen
        late = tl.where(r * P + p < NSLOTS, 0, 1 << 21)
        order = tl.sort((load + late) * PB + p) % PB
        slot = item % 4096
        word = (
            (slot % NT) | (((slot // NT) % NREQ) << 8) | ((slot // (NT * NREQ)) << 16)
        )
        tl.store(out + r * P + order, word, mask=ok)
        tl.store(bins + order, tl.where(ok, item // 4096, 0), mask=p < PB)
        tl.debug_barrier()
        load += tl.load(bins + p)
        tl.debug_barrier()


_LISTS: dict = {}


def shared_list(reqs, tiles, split, long, pairs, device):
    """The WorkList of this shape, one per device and shape for every layer."""
    key = reqs, tiles, split, long, pairs, device
    if key not in _LISTS:
        _LISTS[key] = WorkList(reqs, tiles, split, long, pairs, device)
    return _LISTS[key]


class WorkList:
    """The per-step work list of one packed verify shape: ``reqs`` requests of ``tiles``
    M128 q tiles at ``split`` launched splits under ``long`` (``dcp.long_split``), on
    ``pairs`` persistent CTA pairs. ``buffer``: the list words (``core_ts`` reads them
    through the address table's words 6-7, ``table``)."""

    def __init__(self, reqs, tiles, split, long, pairs, device):
        self.slots = tiles * reqs * split
        assert self.slots <= 4096, "the work list packs slots in 12 bits"
        self.block = triton.next_power_of_2(self.slots)
        self.shape = dict(
            NT=tiles,
            NREQ=reqs,
            SPLIT=split,
            SHORT=long[0],
            THR=long[1],
            QUANT=long[2],
            P=pairs,
            PB=triton.next_power_of_2(pairs),
            NSLOTS=self.slots,
            BLOCK=self.block,
            ROUNDS=-(-self.slots // pairs),
            OVER=WLIST_OVERHEAD,
            EMPTY=WLIST_EMPTY,
        )
        words = self.slots + self.block + self.shape["PB"]
        self.buffer = torch.full(
            (words,), WLIST_EMPTY, dtype=torch.int32, device=device
        )

    def build(self, lengths):
        """Write this step's list from the requests' local lengths (int32 [reqs])."""
        scratch = self.buffer[self.slots :]
        _wlist_kernel[(1,)](lengths, self.buffer, scratch, **self.shape, num_warps=4)

    def table(self, table):
        """``table`` (``core_q.verify_table``) with the list's address in words 6-7."""
        address = self.buffer.data_ptr()
        words = torch.tensor([address & 0xFFFFFFFF, address >> 32], dtype=torch.int64)
        table = table.clone()
        table[6:8] = words.to(torch.int32).to(table.device)
        return table
