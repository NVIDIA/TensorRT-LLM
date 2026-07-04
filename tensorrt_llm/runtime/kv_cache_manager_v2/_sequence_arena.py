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
* :class:`PageBudget` -- the GPU-level physical-page quota (§4.6). Shared by
  every arena of a cache level; replaces per-pool-group slot partitioning.
* :class:`BlockRangeAllocator` -- allocates page-aligned, contiguous
  block-index ranges from an arena's index space (§4.1). Pure logic, no CUDA.
* :class:`SequenceArena` -- one block-index space for a whole *pool group*:
  a shared :class:`BlockRangeAllocator` plus one sparse :class:`VirtMem` per
  pool (a block index addresses all pools of its group, exactly like a slot id
  does today). Implements demand paging with map-ahead margin (§4.2) and an
  event-gated deferred-reclaim queue.

The pieces that wire this into the rest of v2 (per-sequence growth in
``_KVCache``, capacity accounting in the scheduler, active<->stale copies) are
NOT here; the storage-side seam (``ArenaPoolGroup``) lives in
``_storage/_core.py``.
"""

from collections.abc import Sequence
from math import gcd

from ._common import MemAddress
from ._cuda_virt_mem import PooledPhysMemAllocator, VirtMem
from ._exceptions import OutOfPagesError
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
    For a pool group whose pools have different scales, pass the maximum.

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


class PageBudget:
    """GPU-level physical-page quota (DESIGN.md §4.6).

    In arena mode the KV GPU quota is enforced here, at *mapping* time, instead
    of by pre-partitioned per-pool-group slot counts: all arenas of a cache
    level share one budget and compete for pages. Pure bookkeeping (the actual
    handles live in the shared :class:`PooledPhysMemAllocator`).
    """

    __slots__ = ("_total", "_used", "_retained")
    _total: int
    _used: int
    # Subset of _used held by lazily retained ranges (§4.4 phase 2): mapped
    # but reclaimable at will, analogous to classic "evictable" blocks.
    _retained: int

    def __init__(self, total_pages: int) -> None:
        assert total_pages > 0
        self._total = total_pages
        self._used = 0
        self._retained = 0

    @property
    def total_pages(self) -> int:
        return self._total

    @property
    def used_pages(self) -> int:
        return self._used

    @property
    def retained_pages(self) -> int:
        return self._retained

    @property
    def free_pages(self) -> int:
        return self._total - self._used

    def retain(self, num_pages: int) -> None:
        assert 0 <= num_pages and self._retained + num_pages <= self._used
        self._retained += num_pages

    def unretain(self, num_pages: int) -> None:
        assert 0 <= num_pages <= self._retained
        self._retained -= num_pages

    def consume(self, num_pages: int) -> None:
        """Take ``num_pages`` from the budget; raises ``OutOfPagesError``
        (without side effects) if the budget cannot cover them. This is the
        signal for the caller to reclaim/preempt (§4.6)."""
        assert num_pages >= 0
        if self._used + num_pages > self._total:
            raise OutOfPagesError(
                f"KV page budget exhausted: need {num_pages} pages, "
                f"{self.free_pages}/{self._total} free"
            )
        self._used += num_pages

    def release(self, num_pages: int) -> None:
        assert 0 <= num_pages <= self._used
        self._used -= num_pages


def _normalize_strides(record_stride: "int | Sequence[int]") -> tuple[int, ...]:
    if isinstance(record_stride, int):
        return (record_stride,)
    strides = tuple(record_stride)
    assert strides, "at least one pool record stride is required"
    return strides


class BlockRangeAllocator:
    """First-fit free-list allocator over a pool group's block-index space.

    Hands out contiguous ``[base_block, base_block + length)`` ranges. Every
    allocation is aligned and padded to a physical-page boundary *in every
    pool of the group* so that two sequences never share a physical page
    (which would couple their map/unmap lifetimes). Per-pool alignment,
    expressed in blocks, is ``phys_page_size / gcd(record_stride,
    phys_page_size)`` -- the smallest block count whose byte extent is a whole
    number of physical pages; the group alignment is the lcm across pools.

    Live allocations are tracked (``base_block -> padded length``) so ranges can
    be freed by base alone. Pure Python logic (no CUDA); unit-testable on its own.
    """

    __slots__ = ("_capacity", "_align", "_free", "_live")
    _capacity: int  # total blocks in the arena
    _align: int  # allocation granularity in blocks
    # sorted, coalesced, non-overlapping free extents as [start, length] in blocks
    _free: list[list[int]]
    _live: dict[int, int]  # base_block -> padded length in blocks

    def __init__(
        self,
        capacity_blocks: int,
        record_stride: "int | Sequence[int]",
        phys_page_size: int,
    ) -> None:
        strides = _normalize_strides(record_stride)
        assert capacity_blocks > 0 and phys_page_size > 0
        assert all(s > 0 for s in strides)
        align = 1
        for stride in strides:
            pool_align = phys_page_size // gcd(stride, phys_page_size)
            align = align * pool_align // gcd(align, pool_align)  # lcm
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

    def is_live(self, base_block: int) -> bool:
        return base_block in self._live

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
    """The contiguous block-index space of one *pool group*: one shared
    :class:`BlockRangeAllocator` and one sparse ``cuMemAddressReserve`` per
    pool. Carves page-aligned block-index ranges for sequences and maps
    physical pages on demand (in every pool), recycling them through the
    shared physical pool on free.

    A block index addresses *all* pools of the group -- byte address in pool
    ``p`` is ``base_address(p) + block_index * record_stride[p]`` -- mirroring
    how one slot id addresses every pool today. Arena base pointers are fixed
    for the process lifetime, preserving the kernel/CUDA-graph contract
    ``pool_base + page_index * page_stride`` (DESIGN.md §3).

    ``record_stride`` is the byte size of one block's coalesced record per
    pool (a single int for one-pool groups); the physical mapping granularity
    (super-page) is the shared allocator's ``phys_mem_size`` -- see
    :class:`ContiguousArenaConfig`. If ``page_budget`` is given, every page
    mapped/unmapped is accounted against it (§4.6); ``ensure_mapped`` raises
    ``OutOfPagesError`` (mapping nothing) when the budget is exhausted.
    """

    __slots__ = (
        "_vms",
        "_alloc",
        "_strides",
        "_phys_page_size",
        "_map_ahead_pages",
        "_budget",
        "_reclaim",
    )
    _vms: list[VirtMem]
    _alloc: BlockRangeAllocator
    _strides: tuple[int, ...]
    _phys_page_size: int
    _map_ahead_pages: int
    _budget: "PageBudget | None"
    _reclaim: list[_ReclaimEntry]

    def __init__(
        self,
        block_capacity: int,
        record_stride: "int | Sequence[int]",
        phys_mem_allocator: PooledPhysMemAllocator,
        map_ahead_pages: int = 1,
        page_budget: "PageBudget | None" = None,
    ) -> None:
        strides = _normalize_strides(record_stride)
        phys_page_size = phys_mem_allocator.phys_mem_size
        self._vms = [
            VirtMem(
                div_up(block_capacity * stride, phys_page_size) * phys_page_size,
                phys_mem_allocator,
            )
            for stride in strides
        ]
        self._alloc = BlockRangeAllocator(block_capacity, strides, phys_page_size)
        self._strides = strides
        self._phys_page_size = phys_page_size
        self._map_ahead_pages = map_ahead_pages
        self._budget = page_budget
        self._reclaim = []

    @property
    def num_pools(self) -> int:
        return len(self._vms)

    def base_address(self, pool: int = 0) -> MemAddress:
        return self._vms[pool].address

    @property
    def mapped_pages(self) -> int:
        total = 0
        for vm in self._vms:
            total += vm.num_sparse_chunks
        return total

    @property
    def capacity_blocks(self) -> int:
        return self._alloc.capacity_blocks

    @property
    def free_blocks(self) -> int:
        return self._alloc.free_blocks()

    def reserved_len(self, base_block: int) -> int:
        return self._alloc.reserved_len(base_block)

    def mapped_pages_in_range(self, base_block: int) -> int:
        """Number of physical pages currently mapped inside the range starting
        at ``base_block``, across all pools."""
        num_blocks = self._alloc.reserved_len(base_block)
        page = self._phys_page_size
        total = 0
        for pool in range(len(self._vms)):
            lo, hi = self._chunk_range(pool, base_block, num_blocks)
            vm = self._vms[pool]
            total += sum(1 for chunk in range(lo, hi) if vm.is_mapped(chunk * page))
        return total

    def reserve(self, max_blocks: int) -> int:
        """Reserve VA for a sequence's maximum block count; returns base_block.
        No physical memory is mapped yet (that happens in :meth:`ensure_mapped`)."""
        return self._alloc.allocate(max_blocks)

    def _chunk_range(self, pool: int, base_block: int, num_blocks: int) -> tuple[int, int]:
        """Half-open physical-chunk index range in ``pool`` covering
        ``num_blocks`` blocks starting at ``base_block``. Ranges are
        page-aligned per sequence in every pool, so a chunk belongs to exactly
        one sequence."""
        stride = self._strides[pool]
        page = self._phys_page_size
        lo = (base_block * stride) // page
        hi = div_up((base_block + num_blocks) * stride, page)
        return lo, hi

    def _missing_runs(self, pool: int, lo: int, hi: int) -> list[tuple[int, int]]:
        """Contiguous runs of unmapped chunks in ``pool`` within ``[lo, hi)``,
        as (start_chunk, num_chunks) pairs."""
        vm = self._vms[pool]
        page = self._phys_page_size
        runs: list[tuple[int, int]] = []
        run_start = -1
        chunk = lo
        while chunk < hi:
            if not vm.is_mapped(chunk * page):
                if run_start < 0:
                    run_start = chunk
            elif run_start >= 0:
                runs.append((run_start, chunk - run_start))
                run_start = -1
            chunk += 1
        if run_start >= 0:
            runs.append((run_start, hi - run_start))
        return runs

    def ensure_mapped(self, base_block: int, num_valid_blocks: int) -> int:
        """Map physical pages covering ``[base_block, base_block + num_valid_blocks)``
        plus ``map_ahead_pages`` of margin, in every pool, skipping
        already-mapped pages. Returns the number of newly mapped pages.

        Maps are issued as contiguous runs (one :meth:`VirtMem.map_range` per
        run) so a single ``cuMemSetAccess`` covers each run (DESIGN.md §4.2).
        Safe to call concurrently with running kernels: it only touches pages
        strictly ahead of the write frontier.

        If a page budget is attached and cannot cover the new pages, raises
        ``OutOfPagesError`` *before mapping anything* — the caller reclaims or
        preempts and retries (§4.6).
        """
        assert num_valid_blocks >= 0
        reserved = self._alloc.reserved_len(base_block)
        margin = self._map_ahead_pages
        runs_per_pool: list[list[tuple[int, int]]] = []
        total_new = 0
        for pool in range(len(self._vms)):
            lo, hi = self._chunk_range(pool, base_block, num_valid_blocks)
            hi += margin
            # Clamp to this sequence's reserved extent so map-ahead never
            # spills into a neighbour's pages.
            max_hi = self._chunk_range(pool, base_block, reserved)[1]
            if hi > max_hi:
                hi = max_hi
            runs = self._missing_runs(pool, lo, hi)
            runs_per_pool.append(runs)
            for _, num_chunks in runs:
                total_new += num_chunks
        if total_new == 0:
            return 0
        budget = self._budget
        if budget is not None:
            budget.consume(total_new)  # raises OutOfPagesError; nothing mapped yet
        page = self._phys_page_size
        num_mapped = 0
        try:
            for pool in range(len(self._vms)):
                vm = self._vms[pool]
                for start_chunk, num_chunks in runs_per_pool[pool]:
                    vm.map_range(start_chunk * page, num_chunks)
                    num_mapped += num_chunks
        except Exception:
            # map_range rolls itself back; return the unmapped remainder to
            # the budget (already-mapped runs stay mapped and accounted).
            if budget is not None:
                budget.release(total_new - num_mapped)
            raise
        return total_new

    def enqueue_free(self, base_block: int, last_consumer: CachedCudaEvent) -> None:
        """Queue a freed range for deferred unmap once ``last_consumer`` (the
        forward step / D2H copy / transfer agent) completes. Unmapping a range
        still referenced by in-flight work is an IMA (DESIGN.md §4.2, risk #3).

        The block-index range stays reserved (not returned to the allocator)
        until :meth:`drain_reclaim` actually unmaps it, so it cannot be reissued
        to another sequence while pages are still live."""
        assert self._alloc.is_live(base_block)
        self._reclaim.append(_ReclaimEntry(base_block, last_consumer))

    def drain_reclaim(self) -> int:
        """Unmap and recycle every queued range whose gating event has
        completed. Returns the number of ranges reclaimed. Call at iteration
        boundaries."""
        remaining: list[_ReclaimEntry] = []
        reclaimed = 0
        for entry in self._reclaim:
            if entry.event.query_complete():
                self._reclaim_now(entry.base_block)
                reclaimed += 1
            else:
                remaining.append(entry)
        self._reclaim = remaining
        return reclaimed

    def reclaim(self, base_block: int) -> None:
        """Immediately unmap a range's pages and return its block indices to
        the allocator. The caller must itself guarantee that no in-flight GPU
        work references the range (e.g. ``ArenaPoolGroup`` gates on slot
        release and CUDA events before calling this). For simple single-event
        gating use :meth:`enqueue_free` + :meth:`drain_reclaim` instead."""
        self._reclaim_now(base_block)

    def _reclaim_now(self, base_block: int) -> None:
        """Unmap a range's pages in every pool and return the block range to
        the allocator. Caller guarantees no in-flight work references it."""
        num_blocks = self._alloc.reserved_len(base_block)
        num_unmapped = 0
        for pool in range(len(self._vms)):
            lo, hi = self._chunk_range(pool, base_block, num_blocks)
            # Unmap only the pages we actually mapped for this range.
            num_unmapped += self._unmap_mapped_run(pool, lo, hi)
        self._alloc.free(base_block)
        if self._budget is not None:
            self._budget.release(num_unmapped)

    def _unmap_mapped_run(self, pool: int, lo: int, hi: int) -> int:
        vm = self._vms[pool]
        page = self._phys_page_size
        num_unmapped = 0
        run_start = -1
        chunk = lo
        while chunk < hi:
            if vm.is_mapped(chunk * page):
                if run_start < 0:
                    run_start = chunk
            elif run_start >= 0:
                vm.unmap_range(run_start * page, chunk - run_start)
                num_unmapped += chunk - run_start
                run_start = -1
            chunk += 1
        if run_start >= 0:
            vm.unmap_range(run_start * page, hi - run_start)
            num_unmapped += hi - run_start
        return num_unmapped

    def destroy(self) -> None:
        if self._budget is not None:
            self._budget.release(self.mapped_pages)
        for vm in self._vms:
            vm.destroy()
        self._reclaim = []
