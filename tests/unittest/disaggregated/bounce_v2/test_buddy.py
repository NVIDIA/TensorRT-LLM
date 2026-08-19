# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Buddy allocator tests: port of buddyAllocatorTest.cpp.

Right-sizing, no overlap, coalescing on free, exhaustion/backpressure, and the
"many small + one larger-than-buffer (recycled)" regimes.
"""

from __future__ import annotations

import pytest
from conftest import load_bounce_v2

_b = load_bounce_v2()
BuddyAllocator = _b.BuddyAllocator


def check_no_overlap(live: list[tuple[int, int]]) -> None:
    """Assert no two live [offset, offset+size) ranges overlap."""
    ordered = sorted(live)
    for (prev_off, prev_size), (off, _size) in zip(ordered, ordered[1:]):
        assert prev_off + prev_size <= off, f"overlap: [{prev_off},+{prev_size}) vs [{off},...)"


# C++: BuddyAllocator.RoundsUpToPow2MultipleOfMinBlock
def test_rounds_up_to_pow2_multiple_of_min_block() -> None:
    a = BuddyAllocator(capacity=1024, min_block=64)
    assert a.capacity == 1024
    assert a.min_block == 64
    assert a.max_block_bytes == 1024  # whole arena free
    # 1 byte -> a min_block (64). 65 bytes -> 128. 64 -> 64.
    o1 = a.allocate(1)
    o2 = a.allocate(65)
    o3 = a.allocate(64)
    assert o1 is not None and o2 is not None and o3 is not None
    assert o1 % 64 == 0
    # free_bytes accounts for the rounded-up block sizes (64 + 128 + 64 used).
    assert a.free_bytes == 1024 - (64 + 128 + 64)
    assert a.live_blocks == 3


# C++: BuddyAllocator.ManySmallRegionsDistinctNoOverlap
def test_many_small_regions_distinct_no_overlap() -> None:
    # capacity 16 * min_block -> up to 16 concurrent min_block allocations.
    a = BuddyAllocator(capacity=16 * 256, min_block=256)
    live: list[tuple[int, int]] = []
    seen: set[int] = set()
    for i in range(16):
        off = a.allocate(200)  # < 256 -> one min_block
        assert off is not None, f"small alloc {i} failed"
        assert off not in seen, "offset reused while live"
        seen.add(off)
        live.append((off, 256))
    assert a.live_blocks == 16
    assert a.free_bytes == 0
    assert a.allocate(1) is None  # full -> backpressure (None, no overcommit)
    check_no_overlap(live)

    for off, _size in live:
        a.free(off)
    assert a.free_bytes == a.capacity  # all coalesced back
    assert a.max_block_bytes == a.capacity
    assert a.live_blocks == 0


# C++: BuddyAllocator.FreeCoalescesBuddies
def test_free_coalesces_buddies() -> None:
    a = BuddyAllocator(capacity=1024, min_block=256)  # 4 min_blocks
    offs = [a.allocate(256) for _ in range(4)]
    assert all(o is not None for o in offs)
    assert a.max_block_bytes == 0  # fully split into min_blocks
    # Free all -> must coalesce back to one 1024 block (max_block == capacity).
    for off in offs:
        assert off is not None
        a.free(off)
    assert a.max_block_bytes == 1024, "buddies did not coalesce to the top order"
    # And a full-size alloc now succeeds.
    big = a.allocate(1024)
    assert big == 0


# C++: BuddyAllocator.TooLargeRejected
def test_too_large_rejected() -> None:
    a = BuddyAllocator(capacity=1024, min_block=256)
    assert a.allocate(2048) is None  # larger than the whole arena
    assert a.allocate(1024) is not None


# C++: BuddyAllocator.ZeroAndOverflowSizeRejectedNoHang
def test_zero_and_overflow_size_rejected_no_hang() -> None:
    a = BuddyAllocator(capacity=1024, min_block=256)
    assert a.allocate(0) is None
    # A near-SIZE_MAX request must be rejected WITHOUT hanging — the
    # power-of-two rounding loop would otherwise spin; the early
    # `bytes > usable` bound prevents it.
    assert a.allocate(2**64 - 1) is None
    assert a.allocate(2**64 - 2) is None
    assert a.free_bytes == a.capacity  # nothing consumed by rejections
    assert a.allocate(256) is not None  # still usable afterwards


# C++: BuddyAllocator.LargeRecycledStreamLargerThanArena
def test_large_recycled_stream_larger_than_arena() -> None:
    # A transfer larger than the arena streams through with recycling.
    min_block = 1024
    chunk = 4 * 1024  # max_chunk_size_bytes
    a = BuddyAllocator(capacity=4 * chunk, min_block=min_block)
    streamed = 0
    inflight: list[int] = []
    for i in range(1000):  # 1000 chunks * 4KiB = 4MiB through a 16KiB arena
        if len(inflight) == 4:  # keep up to 4 chunks in flight
            a.free(inflight.pop(0))
        off = a.allocate(chunk)
        assert off is not None, f"chunk {i} did not fit despite recycling"
        inflight.append(off)
        streamed += chunk
    assert streamed > a.capacity * 10  # streamed far more than the arena holds
    for off in inflight:
        a.free(off)
    assert a.free_bytes == a.capacity
    assert a.live_blocks == 0


# C++: BuddyAllocator.MixedSmallAndLargeShareArena
def test_mixed_small_and_large_share_arena() -> None:
    a = BuddyAllocator(capacity=8 * 1024, min_block=1024)  # 8 blocks
    big = a.allocate(4 * 1024)  # 4 blocks
    assert big is not None
    smalls = [a.allocate(1024) for _ in range(4)]
    assert all(o is not None for o in smalls)
    assert a.free_bytes == 0
    assert a.allocate(1024) is None  # full
    # Free the smalls -> they coalesce -> a second 4*1024 chunk now fits.
    for off in smalls:
        assert off is not None
        a.free(off)
    big2 = a.allocate(4 * 1024)
    assert big2 is not None
    a.free(big)
    a.free(big2)
    assert a.max_block_bytes == a.capacity


# C++: BuddyAllocator.DoubleFreeIgnored
def test_double_free_ignored() -> None:
    a = BuddyAllocator(1024, 256)
    off = a.allocate(256)
    assert off is not None
    a.free(off)
    a.free(off)  # ignored, no corruption
    assert a.free_bytes == a.capacity
    assert a.live_blocks == 0


# ---- boundary cases ----


# C++: BuddyAllocator.CapacityRoundedDownToMinBlockMultiple
def test_capacity_rounded_down_to_min_block_multiple() -> None:
    # capacity is NOT a power-of-two multiple of min_block -> usable rounds
    # DOWN to 512 (256<<1); the trailing 1000-512 bytes are unusable.
    a = BuddyAllocator(capacity=1000, min_block=256)
    assert a.capacity == 512
    assert a.min_block == 256
    assert a.max_block_bytes == 512
    whole = a.allocate(512)
    assert whole == 0
    assert a.allocate(1) is None  # the rounded-off tail is not allocatable


# C++: BuddyAllocator.MinBlockRoundedUpToPow2
def test_min_block_rounded_up_to_pow2() -> None:
    # min_block 100 -> rounded up to 128; capacity 512 -> orders 0..2.
    a = BuddyAllocator(capacity=512, min_block=100)
    assert a.min_block == 128
    assert a.capacity == 512
    o1 = a.allocate(1)  # -> 128 (one min_block)
    o2 = a.allocate(130)  # -> 256 (next power of two)
    assert o1 is not None and o2 is not None
    assert o1 % 128 == 0
    assert o2 % 256 == 0  # 256-block is 256-aligned
    assert a.free_bytes == 512 - (128 + 256)


# C++: BuddyAllocator.SingleBlockArena
def test_single_block_arena() -> None:
    # capacity == min_block -> exactly one order-0 block (max order 0).
    a = BuddyAllocator(capacity=256, min_block=256)
    assert a.capacity == 256
    assert a.max_block_bytes == 256
    off = a.allocate(10)
    assert off == 0
    assert a.allocate(1) is None  # only one block -> full
    assert a.max_block_bytes == 0
    a.free(off)
    assert a.max_block_bytes == 256
    assert a.allocate(256) is not None  # reusable


# C++: BuddyAllocator.FreeUnknownOffsetIgnored
def test_free_unknown_offset_ignored() -> None:
    a = BuddyAllocator(capacity=1024, min_block=256)
    off = a.allocate(256)
    assert off is not None
    free_before = a.free_bytes
    a.free(999999)  # wildly out of range
    a.free(off + 64)  # in range but not a block start / not allocated
    assert a.free_bytes == free_before
    assert a.live_blocks == 1  # the one real allocation is untouched
    a.free(off)  # the genuine free still works exactly once
    assert a.free_bytes == a.capacity
    assert a.live_blocks == 0


# C++: BuddyAllocator.BlockBytesReportsRoundedUpSize
def test_block_bytes_reports_rounded_up_size() -> None:
    a = BuddyAllocator(capacity=1024, min_block=64)
    off = a.allocate(65)  # 65 -> rounded up to 128
    assert off is not None
    assert a.block_bytes(off) == 128
    assert a.block_bytes(off + 999) == 0  # not a live block start
    a.free(off)
    assert a.block_bytes(off) == 0  # freed -> no longer live


# C++: BuddyAllocator.FragmentationBlocksLargeAllocDespiteFreeBytes
def test_fragmentation_blocks_large_alloc_despite_free_bytes() -> None:
    # Freeing every OTHER min_block leaves half the arena free in bytes, but
    # as scattered order-0 blocks with no free buddy pair -> a 2-block alloc
    # must FAIL; freeing the rest coalesces so it succeeds.
    k_min = 256
    a = BuddyAllocator(capacity=8 * k_min, min_block=k_min)  # 8 order-0 blocks
    all_offs: list[int] = []
    for _ in range(8):
        off = a.allocate(k_min)
        assert off is not None
        all_offs.append(off)
    assert a.free_bytes == 0
    # Free exactly the lower buddy of each pair -> no two freed are buddies.
    held: list[int] = []
    for off in all_offs:
        if off % (2 * k_min) == 0:
            a.free(off)
        else:
            held.append(off)
    assert a.free_bytes == 4 * k_min  # half the arena is free...
    assert a.max_block_bytes == k_min  # ...but only as isolated order-0 blocks
    assert a.allocate(2 * k_min) is None  # 2-block request fails (fragmented)
    extra = a.allocate(k_min)  # a 1-block request still works
    assert extra is not None
    a.free(extra)
    for off in held:
        a.free(off)
    assert a.live_blocks == 0
    assert a.max_block_bytes == a.capacity


# Python-specific constructor validation (the C++ asserted via TLLM_CHECK).
@pytest.mark.parametrize(
    "capacity,min_block",
    [(0, 256), (-1, 256), (1024, 0), (1024, -8), (100, 256)],
    ids=["zero-cap", "neg-cap", "zero-min", "neg-min", "cap-lt-min"],
)
def test_constructor_rejects_bad_sizes(capacity: int, min_block: int) -> None:
    with pytest.raises(ValueError):
        BuddyAllocator(capacity, min_block)
