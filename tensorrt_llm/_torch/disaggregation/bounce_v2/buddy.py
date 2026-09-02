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
"""Power-of-two buddy allocator over a logical byte space.

Python port of the C++ ``BuddyAllocator``
(cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BuddyAllocator.{h,cpp}).
Pure logic, no GPU / threads / IO — fully unit-testable. It is the data-region
allocator for the bounce v2 arena (one shared registered buffer): instead of
fixed full slots, each chunk gets a region whose buddy block is sized to its
actual bytes, so many small transfers fit with less internal waste while a
transfer larger than the whole buffer still streams through chunk by chunk.

Properties (same invariants as the C++ implementation):
  - Block sizes are exactly ``min_block << order``; a block of size S always
    starts at an offset that is a multiple of S (the buddy alignment
    invariant, which makes the buddy of a block ``offset XOR S``).
  - Structured external fragmentation: only free buddies of the same order
    coalesce, so total free bytes may exceed a request while no sufficiently
    large block is currently available.
  - Internal fragmentation is under 2x because requests round up to a
    power-of-two block.
  - ``allocate`` returns ``None`` when no block of the needed order is free
    (the caller applies backpressure; a later free+coalesce releases a
    high-order block — no deadlock, since frees come from
    independently-completing flows).

Sizing: ``capacity`` is rounded DOWN to the largest ``min_block * 2**L`` that
fits (the remainder is unused); ``min_block`` is rounded UP to a power of two.
Always query the properties instead of assuming the constructor arguments.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["BuddyAllocator"]


def _round_up_pow2(value: int) -> int:
    power = 1
    while power < value:
        power <<= 1
    return power


class BuddyAllocator:
    """Buddy allocator managing byte OFFSETS in ``[0, capacity)``.

    The actual GPU buffer (base pointer) lives elsewhere; the
    ``CreditScheduler`` wraps this allocator to hand out region offsets over
    that buffer.
    """

    def __init__(self, capacity: int, min_block: int) -> None:
        """Create the allocator.

        Args:
            capacity: Total bytes (rounded DOWN to the largest
                ``min_block * 2**L`` that fits).
            min_block: Smallest allocatable block (rounded UP to a power of
                two); the order-0 block size.

        Raises:
            ValueError: If ``capacity``/``min_block`` are not positive or the
                capacity cannot fit a single minimum block.
        """
        if capacity <= 0 or min_block <= 0:
            raise ValueError("BuddyAllocator: capacity/min_block must be > 0")
        self._min_block = _round_up_pow2(min_block)
        if capacity < self._min_block:
            raise ValueError("BuddyAllocator: capacity < min_block")

        # Largest order L with min_block << L <= capacity; usable = min_block << L.
        order = 0
        while (self._min_block << (order + 1)) <= capacity:
            order += 1
        self._max_order = order
        self._usable = self._min_block << order

        # _free[order] = set of free block offsets at that order.
        self._free: list[set[int]] = [set() for _ in range(self._max_order + 1)]
        self._free[self._max_order].add(0)  # one block covering the whole usable space
        # allocated offset -> order (so free() knows the block size to coalesce)
        self._alloc_order: dict[int, int] = {}

    def _order_for_bytes(self, nbytes: int) -> int:
        need = _round_up_pow2(max(nbytes, self._min_block))
        order = 0
        while (self._min_block << order) < need:
            order += 1
        return order

    def allocate(self, nbytes: int) -> Optional[int]:
        """Allocate at least ``nbytes`` (> 0).

        Returns the block's OFFSET only (not its size), or ``None`` if no free
        block of the required order exists right now.

        Offset-only is sufficient because ``free`` is offset-keyed (the size
        is looked up internally), and the caller must use its OWN requested
        length for the actual transfer — NOT the rounded-up block size: the
        slack between ``nbytes`` and the block size contains no logical
        payload. Use :meth:`block_bytes` if the rounded size is genuinely
        needed (e.g. metrics).
        """
        # Reject empty and anything larger than the usable arena up front
        # (mirrors the C++ overflow guard on the power-of-two rounding).
        if nbytes <= 0 or nbytes > self._usable:
            return None
        want = self._order_for_bytes(nbytes)
        if want > self._max_order:
            return None
        # Find the smallest order >= want that has a free block.
        cur = want
        while cur <= self._max_order and not self._free[cur]:
            cur += 1
        if cur > self._max_order:
            return None  # no block big enough is free (fragmented / full)
        # Take a block at `cur` and split down to `want`, keeping the upper
        # half free at each split and descending into the lower half.
        block = next(iter(self._free[cur]))
        self._free[cur].discard(block)
        while cur > want:
            cur -= 1
            self._free[cur].add(block + (self._min_block << cur))
        self._alloc_order[block] = want
        return block

    def free(self, offset: int) -> None:
        """Free a block previously returned by :meth:`allocate`.

        Coalesces with the buddy whenever it is also free at the same order.
        An offset that is not a live allocation (double free / bad offset) is
        ignored — robustness over strictness, matching the C++ behavior.
        """
        order = self._alloc_order.pop(offset, None)
        if order is None:
            return
        while order < self._max_order:
            buddy = offset ^ (self._min_block << order)
            if buddy not in self._free[order]:
                break  # buddy not free -> stop merging
            self._free[order].discard(buddy)
            offset = min(offset, buddy)  # merged block starts at the lower address
            order += 1
        self._free[order].add(offset)

    def block_bytes(self, offset: int) -> int:
        """Rounded (power-of-two) size of the live block at ``offset``, or 0
        if ``offset`` is not a live allocation."""
        order = self._alloc_order.get(offset)
        return 0 if order is None else (self._min_block << order)

    @property
    def capacity(self) -> int:
        """Usable capacity in bytes (``min_block << max_order``)."""
        return self._usable

    @property
    def min_block(self) -> int:
        """Order-0 block size (a power of two)."""
        return self._min_block

    @property
    def free_bytes(self) -> int:
        """Sum of all free block sizes in bytes."""
        return sum(
            len(offsets) * (self._min_block << order) for order, offsets in enumerate(self._free)
        )

    @property
    def max_block_bytes(self) -> int:
        """Largest single allocation that can succeed right now (0 if none).

        Lets callers detect temporary allocation backpressure.
        """
        for order in range(self._max_order, -1, -1):
            if self._free[order]:
                return self._min_block << order
        return 0

    @property
    def live_blocks(self) -> int:
        """Number of currently-allocated blocks (tests / metrics / leak checks)."""
        return len(self._alloc_order)
