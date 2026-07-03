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

"""Per-sequence contiguous VA arenas for the KV cache (prototype).

This is the storage foundation for the *contiguous-in-VA active KV cache*: a
single large ``cuMemAddressReserve`` per (pool group, pool) — the **arena** —
out of which each sequence gets a contiguous run of block indices, backed by
physical pages mapped on demand and recycled through the shared physical pool.
See ``contiguous_primary_kvcache/DESIGN.md`` for the full design; section
references below (§4.1, §4.2, ...) point into it.

Contents (bottom-up):

* :func:`check_index_width` -- the int31 kernel-offset ceiling check (§4.1),
  meant to run at startup.
* :class:`BlockRangeAllocator` -- allocates page-aligned, contiguous block-index
  ranges from the arena's index space (§4.1). Pure logic, no CUDA.
* :class:`SequenceArena` -- ties a sparse :class:`VirtMem` reservation to a
  :class:`BlockRangeAllocator` and implements demand paging with map-ahead
  margin (§4.2) and an event-gated deferred-reclaim queue.

The pieces that wire this into the rest of v2 (per-sequence growth in
``_KVCache``, capacity accounting in the scheduler, active<->stale copies) are
NOT here yet; see the "Integration points" stubs at the bottom of the file.
"""

from math import gcd

from ._common import MemAddress
from ._cuda_virt_mem import PooledPhysMemAllocator, VirtMem
from ._utils import CachedCudaEvent, div_up

# Kernel-facing offsets are int32, and v1's KVCacheIndex steals the high bit for
# primary/secondary selection, leaving int31. The emitted value is
# ``block_index * num_coalesced_subbuffers + field``; its maximum must fit. See
# DESIGN.md §4.1.
INT31_MAX: int = (1 << 31) - 1


def check_index_width(block_capacity: int, num_coalesced_subbuffers: int) -> None:
    """Assert that every kernel-facing offset an arena of ``block_capacity``
    blocks can emit fits in int31.

    ``num_coalesced_subbuffers`` is the per-slot ``scale`` factor (number of
    coalesced sub-buffers per block, e.g. ``num_layers * kv_factor`` plus any
    extra buffers) -- the same value the existing offset-table converter uses.

    Raises ``ValueError`` (a clear startup-time error, per DESIGN.md §4.1)
    rather than letting the kernels silently read garbage.
    """
    assert block_capacity > 0 and num_coalesced_subbuffers > 0
    max_emitted = block_capacity * num_coalesced_subbuffers - 1
    if max_emitted > INT31_MAX:
        raise ValueError(
            f"contiguous KV arena would emit block offsets up to {max_emitted}, exceeding the "
            f"int31 kernel-offset ceiling ({INT31_MAX}). block_capacity={block_capacity}, "
            f"scale={num_coalesced_subbuffers}. Mitigations (DESIGN.md §4.1): larger "
            f"tokens_per_block, per-size-class arenas, or a wider offset dtype."
        )


class BlockRangeAllocator:
    """First-fit free-list allocator over an arena's block-index space.

    Hands out contiguous ``[base_block, base_block + length)`` ranges. Every
    allocation is aligned and padded to a physical-page boundary so that two
    sequences never share a physical page (which would couple their map/unmap
    lifetimes). Alignment is expressed in blocks:
    ``align = phys_page_size / gcd(record_stride, phys_page_size)`` -- the
    smallest block count whose byte extent is a whole number of physical pages.

    Live allocations are tracked (``base_block -> padded length``) so ranges can
    be freed by base alone. Pure Python logic (no CUDA); unit-testable on its own.
    """

    __slots__ = ("_capacity", "_align", "_free", "_live")
    _capacity: int  # total blocks in the arena
    _align: int  # allocation granularity in blocks
    # sorted, coalesced, non-overlapping free extents as [start, length] in blocks
    _free: list[list[int]]
    _live: dict[int, int]  # base_block -> padded length in blocks

    def __init__(self, capacity_blocks: int, record_stride: int, phys_page_size: int) -> None:
        assert capacity_blocks > 0 and record_stride > 0 and phys_page_size > 0
        align = phys_page_size // gcd(record_stride, phys_page_size)
        # The arena's usable capacity is floored to a whole number of alignment units.
        capacity = (capacity_blocks // align) * align
        assert capacity >= align, "arena too small to hold even one page-aligned range"
        self._capacity = capacity
        self._align = align
        self._free = [[0, capacity]]
        self._live = {}

    @property
    def align_blocks(self) -> int:
        return self._align

    @property
    def capacity_blocks(self) -> int:
        return self._capacity

    def reserved_len(self, base_block: int) -> int:
        """Padded length (in blocks) of the live allocation starting at
        ``base_block``."""
        return self._live[base_block]

    def _round_up(self, num_blocks: int) -> int:
        align = self._align
        return div_up(num_blocks, align) * align

    def allocate(self, num_blocks: int) -> int:
        """Reserve a contiguous range of at least ``num_blocks`` (padded to the
        alignment) and return its aligned ``base_block``.

        Raises ``MemoryError`` if the arena has no free extent large enough
        (this is VA exhaustion / fragmentation, not physical OOM).
        """
        assert num_blocks > 0
        want = self._round_up(num_blocks)
        for extent in self._free:
            start, length = extent[0], extent[1]
            if length >= want:
                base = start
                if length == want:
                    self._free.remove(extent)
                else:
                    extent[0] = start + want
                    extent[1] = length - want
                self._live[base] = want
                return base
        raise MemoryError(
            f"contiguous KV arena has no free range of {want} blocks "
            f"(largest free extent is {self.largest_free_blocks()})"
        )

    def free(self, base_block: int) -> None:
        """Return a previously allocated range (identified by its ``base_block``)
        to the free list, coalescing with adjacent free extents."""
        length = self._live.pop(base_block)
        assert base_block % self._align == 0, "base_block must be page-aligned"
        assert 0 <= base_block and base_block + length <= self._capacity
        end = base_block + length
        # Insert sorted by start, then coalesce with neighbours.
        free = self._free
        idx = 0
        while idx < len(free) and free[idx][0] < base_block:
            idx += 1
        # No overlap with existing free extents (double-free guard).
        if idx > 0:
            assert free[idx - 1][0] + free[idx - 1][1] <= base_block, "double free / overlap"
        if idx < len(free):
            assert end <= free[idx][0], "double free / overlap"
        free.insert(idx, [base_block, length])
        self._coalesce_at(idx)

    def _coalesce_at(self, idx: int) -> None:
        free = self._free
        # merge with right neighbour
        if idx + 1 < len(free) and free[idx][0] + free[idx][1] == free[idx + 1][0]:
            free[idx][1] += free[idx + 1][1]
            free.pop(idx + 1)
        # merge with left neighbour
        if idx > 0 and free[idx - 1][0] + free[idx - 1][1] == free[idx][0]:
            free[idx - 1][1] += free[idx][1]
            free.pop(idx)

    def largest_free_blocks(self) -> int:
        return max((extent[1] for extent in self._free), default=0)

    def free_blocks(self) -> int:
        return sum(extent[1] for extent in self._free)


class _ReclaimEntry:
    """A freed arena range awaiting event-gated unmap (DESIGN.md §4.2)."""

    __slots__ = ("base_block", "event")
    base_block: int
    event: CachedCudaEvent

    def __init__(self, base_block: int, event: CachedCudaEvent) -> None:
        self.base_block = base_block
        self.event = event


class SequenceArena:
    """One arena (one ``cuMemAddressReserve``) shared by all sequences of a
    (pool group, pool). Carves page-aligned block-index ranges and maps physical
    pages on demand, recycling them through the shared physical pool on free.

    The arena base pointer is fixed for the process lifetime, preserving the
    kernel/CUDA-graph contract ``pool_base + page_index * page_stride``
    (DESIGN.md §3). ``record_stride`` is the byte size of one block's coalesced
    record; the physical mapping granularity (super-page) is the shared
    allocator's ``phys_mem_size`` -- see :class:`ContiguousArenaConfig`.
    """

    __slots__ = (
        "_vm",
        "_alloc",
        "_record_stride",
        "_phys_page_size",
        "_map_ahead_pages",
        "_reclaim",
    )
    _vm: VirtMem
    _alloc: BlockRangeAllocator
    _record_stride: int
    _phys_page_size: int
    _map_ahead_pages: int
    _reclaim: list[_ReclaimEntry]

    def __init__(
        self,
        block_capacity: int,
        record_stride: int,
        phys_mem_allocator: PooledPhysMemAllocator,
        map_ahead_pages: int = 1,
    ) -> None:
        phys_page_size = phys_mem_allocator.phys_mem_size
        va_size = div_up(block_capacity * record_stride, phys_page_size) * phys_page_size
        self._vm = VirtMem(va_size, phys_mem_allocator)
        self._alloc = BlockRangeAllocator(block_capacity, record_stride, phys_page_size)
        self._record_stride = record_stride
        self._phys_page_size = phys_page_size
        self._map_ahead_pages = map_ahead_pages
        self._reclaim = []

    @property
    def base_address(self) -> MemAddress:
        return self._vm.address

    @property
    def mapped_pages(self) -> int:
        return self._vm.num_sparse_chunks

    def reserve(self, max_blocks: int) -> int:
        """Reserve VA for a sequence's maximum block count; returns base_block.
        No physical memory is mapped yet (that happens in :meth:`ensure_mapped`)."""
        return self._alloc.allocate(max_blocks)

    def _chunk_range(self, base_block: int, num_blocks: int) -> tuple[int, int]:
        """Half-open physical-chunk index range covering ``num_blocks`` blocks
        starting at ``base_block``. Ranges are page-aligned per sequence, so a
        chunk belongs to exactly one sequence."""
        stride = self._record_stride
        page = self._phys_page_size
        lo = (base_block * stride) // page
        hi = div_up((base_block + num_blocks) * stride, page)
        return lo, hi

    def ensure_mapped(self, base_block: int, num_valid_blocks: int) -> None:
        """Map physical pages covering ``[base_block, base_block + num_valid_blocks)``
        plus ``map_ahead_pages`` of margin, skipping already-mapped pages.

        Maps are issued as contiguous runs (one :meth:`VirtMem.map_range` per
        run) so a single ``cuMemSetAccess`` covers each run (DESIGN.md §4.2).
        Safe to call concurrently with running kernels: it only touches pages
        strictly ahead of the write frontier.
        """
        assert num_valid_blocks >= 0
        lo, hi = self._chunk_range(base_block, num_valid_blocks)
        hi += self._map_ahead_pages
        # Clamp to this sequence's reserved extent so map-ahead never spills into
        # a neighbour's pages.
        max_hi = self._chunk_range(base_block, self._alloc.reserved_len(base_block))[1]
        if hi > max_hi:
            hi = max_hi
        page = self._phys_page_size
        run_start = -1
        chunk = lo
        while chunk < hi:
            if not self._vm.is_mapped(chunk * page):
                if run_start < 0:
                    run_start = chunk
            elif run_start >= 0:
                self._vm.map_range(run_start * page, chunk - run_start)
                run_start = -1
            chunk += 1
        if run_start >= 0:
            self._vm.map_range(run_start * page, hi - run_start)

    def enqueue_free(self, base_block: int, last_consumer: CachedCudaEvent) -> None:
        """Queue a freed range for deferred unmap once ``last_consumer`` (the
        forward step / D2H copy / transfer agent) completes. Unmapping a range
        still referenced by in-flight work is an IMA (DESIGN.md §4.2, risk #3).

        The block-index range stays reserved (not returned to the allocator)
        until :meth:`drain_reclaim` actually unmaps it, so it cannot be reissued
        to another sequence while pages are still live."""
        self._reclaim.append(_ReclaimEntry(base_block, last_consumer))

    def drain_reclaim(self) -> int:
        """Unmap and recycle every queued range whose gating event has
        completed. Returns the number of ranges reclaimed. Call at iteration
        boundaries."""
        remaining: list[_ReclaimEntry] = []
        reclaimed = 0
        for entry in self._reclaim:
            if entry.event.query_complete():
                num_blocks = self._alloc.reserved_len(entry.base_block)
                lo, hi = self._chunk_range(entry.base_block, num_blocks)
                # Unmap only the pages we actually mapped for this range.
                self._unmap_mapped_run(lo, hi)
                self._alloc.free(entry.base_block)
                reclaimed += 1
            else:
                remaining.append(entry)
        self._reclaim = remaining
        return reclaimed

    def _unmap_mapped_run(self, lo: int, hi: int) -> None:
        page = self._phys_page_size
        run_start = -1
        chunk = lo
        while chunk < hi:
            if self._vm.is_mapped(chunk * page):
                if run_start < 0:
                    run_start = chunk
            elif run_start >= 0:
                self._vm.unmap_range(run_start * page, chunk - run_start)
                run_start = -1
            chunk += 1
        if run_start >= 0:
            self._vm.unmap_range(run_start * page, hi - run_start)

    def destroy(self) -> None:
        self._vm.destroy()


# ---------------------------------------------------------------------------
# Integration points (NOT yet implemented -- P0 remaining work, DESIGN.md §5)
#
# The following seams connect SequenceArena to the rest of v2. They are called
# out here so the scaffold documents the full P0 surface; each needs GPU
# integration testing to build for real.
#
# * _storage/_core.py: `GpuSlotPool` -> `GpuArenaPool` holding a SequenceArena
#   instead of a tail-stack VirtMem; `SlotAllocator` retired for the GPU level.
# * _core/_kv_cache.py `resize`: grow = extend own range + `ensure_mapped`;
#   block-index emission = `base_block + ordinal` (replaces scattered slot ids);
#   suspend/resume = write-out/copy-in + range release/realloc.
# * _storage_manager.py: GPU-level eviction becomes reclaim/preempt;
#   `_batched_migrate` gains an explicit-destination mode for reuse placement
#   (§4.4) and the write-through-on-commit path (§4.3).
# * pyexecutor/kv_cache_manager_v2.py: capacity API in physical pages (§4.6);
#   batched `ensure_mapped` sweep in the prepare step; call `drain_reclaim` at
#   iteration boundaries. Offset-table plumbing is otherwise unchanged.
# * Startup: call `check_index_width(block_capacity, scale)` when building the
#   arena for each pool (§4.1).
# ---------------------------------------------------------------------------
