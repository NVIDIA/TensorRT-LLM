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

import abc
import errno
import os
import sys
import tempfile
import warnings
from bisect import bisect_right, insort
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, NewType, cast, final

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from .. import rawref
from .._common import (
    BAD_FILE_DESCRIPTOR,
    NDEBUG,
    Address,
    CacheTier,
    DiskAddress,
    FileDescriptor,
    MemAddress,
)
from .._cuda_virt_mem import PooledPhysMemAllocator, SharedPhysMem, VirtMem
from .._exceptions import LogicError, OutOfPagesError
from .._sequence_arena import PageBudget, SequenceArena, check_index_width, quiesce_before_unmap
from .._utils import (
    CachedCudaEvent,
    DynamicBitset,
    HomoTuple,
    HostMem,
    TypedIndexList,
    assert_critical,
    div_up,
    filled_list,
    make_typed,
    query_total_gpu_memory,
    resize_file,
    round_down,
    round_up,
    typed_enumerate,
    typed_len,
    typed_map,
    typed_range,
)

PoolGroupIndex = NewType("PoolGroupIndex", int)
PoolIndex = NewType("PoolIndex", int)
SlotId = NewType("SlotId", int)

# Freed-range adoption (default on): freed arena ranges are parked still
# mapped and handed whole to the next admitted sequence instead of being
# unmapped and remapped. Concurrent cuMemMap/cuMemUnmap churn against
# in-flight kernels reading *other* ranges of the same VA reservation
# triggers device-side MMU faults (driver TLB interference — see
# contiguous_primary_kvcache/WORK_LOG.md, 2026-07-07 discriminator matrix);
# adoption removes that churn from the steady state entirely. "0" disables
# (debugging only).
_RANGE_ADOPTION = os.getenv("TRTLLM_KV_ARENA_RANGE_ADOPTION") != "0"
# P3 prefix aliasing: map the SAME physical pages holding a shared,
# fully-committed prefix into multiple sequences' VA ranges (zero-copy,
# zero-charge reuse). Default OFF while the prototype bakes.
_PREFIX_ALIASING = os.getenv("TRTLLM_KV_ARENA_PREFIX_ALIASING") == "1"
# Span-spill protection override (see ContiguousArenaConfig
# .protected_span_fraction); negative = defer to the config value.
_SPAN_PROTECT_FRACTION = float(os.getenv("TRTLLM_KV_ARENA_SPAN_PROTECT_FRACTION", "-1"))


class _CanonicalSpan:
    """A canonical owner's resident prefix pages, pinned for aliasing (P3).
    ``page_refs`` guard identity/liveness of every canonical page in the
    span -- any mismatch invalidates the entry, the same discipline as the
    ghost registry. ``shared_per_pool`` holds one registry reference per
    physical chunk (dropped on spill/invalidation); ``ready_event`` covers
    the bytes' last writes (alias consumers wait it).

    ``owner_base`` >= 0 marks an OWNER-LIVE span (P3 v2): registered at the
    owner's context-end commit, while the pages are still the owner's
    active KV in the range at that base block. Owner-live spans hold no
    registry budget charge (the owner's range charge covers the pages) and
    are exempt from spilling (dropping the pin frees nothing). The owner's
    ``free_sequence`` flips the span to the registry-charged state
    (owner_base = -1) -- the same accounting close-time registration
    produces directly.
    """

    __slots__ = (
        "page_refs",
        "shared_per_pool",
        "num_blocks",
        "num_pages",
        "ready_event",
        "owner_base",
    )
    page_refs: "list[rawref.ref[Any]]"
    shared_per_pool: "list[list[SharedPhysMem]]"
    num_blocks: int
    owner_base: int

    def is_live(self) -> bool:
        """Whether every pinned handle is still referenced. A registry hit is
        held UNPINNED across the admission->first-resume window, and pressure
        can spill the registry in between (dropping the pins); a dead span
        must fall back to the copy path instead of alias-mapping freed
        handles. Adopted ranges are immune -- their mappings hold their own
        references."""
        for shared in self.shared_per_pool:
            for s in shared:
                if s.num_refs == 0:
                    return False
        return True

    num_pages: int
    ready_event: CachedCudaEvent

    def __init__(
        self,
        page_refs: "list[rawref.ref[Any]]",
        shared_per_pool: "list[list[SharedPhysMem]]",
        num_blocks: int,
        ready_event: CachedCudaEvent,
        owner_base: int = -1,
    ) -> None:
        self.page_refs = page_refs
        self.shared_per_pool = shared_per_pool
        self.num_blocks = num_blocks
        num_pages = 0
        for shared in shared_per_pool:
            num_pages += len(shared)
        self.num_pages = num_pages
        self.ready_event = ready_event
        self.owner_base = owner_base


# A temporary work-around while migrating to new page index API.
# To be removed later.
PoolIndex0 = PoolIndex(0)


class SlotPoolBase(abc.ABC):
    _slot_size: int

    @property
    def slot_size(self) -> int:
        return self._slot_size

    @property
    @abc.abstractmethod
    def num_slots(self) -> int: ...

    @property
    def num_bytes(self) -> int:
        return self.slot_size * self.num_slots

    def __init__(self, slot_size: int) -> None:
        self._slot_size = slot_size

    @abc.abstractmethod
    def destroy(self) -> None:
        pass

    @abc.abstractmethod
    def resize(self, new_num_slots: int) -> None:
        pass

    @abc.abstractmethod
    def slot_address(self, slot: int) -> Address:
        pass

    def __del__(self) -> None:
        self.destroy()


@final
class GpuSlotPool(SlotPoolBase):
    __slots__ = ("_vm",)
    _vm: VirtMem

    def __init__(
        self,
        slot_size: int,
        vm_size: int,
        shared_phys_mem_pool: PooledPhysMemAllocator,
        num_slots: int,
    ):
        super().__init__(slot_size)
        assert vm_size % shared_phys_mem_pool.phys_mem_size == 0
        self._vm = VirtMem(vm_size, shared_phys_mem_pool)
        self.resize(num_slots)

    @override
    def destroy(self) -> None:
        self._vm.destroy()

    @override
    def resize(self, new_num_slots: int) -> None:
        new_num_phys_mem = self._compute_num_phys_mem(
            self.slot_size, new_num_slots, self._vm.phys_mem_size
        )
        self._vm.realloc(self._vm.phys_mem_size * new_num_phys_mem)

    def extend_by_one_phys_mem(self) -> int:
        self._vm.extend(1)
        return self.num_slots

    @override
    def slot_address(self, slot: int) -> MemAddress:
        return MemAddress(int(self._vm.address) + self.slot_size * slot)

    @property
    @override
    def num_slots(self) -> int:
        return self._compute_num_slots(
            self.slot_size, self._vm.num_phys_mem, self._vm.phys_mem_size
        )

    @staticmethod
    def _compute_num_phys_mem(slot_size: int, num_slots: int, phys_mem_size: int) -> int:
        return div_up(num_slots * slot_size, phys_mem_size)

    @staticmethod
    def _compute_num_slots(slot_size: int, num_phys_mem: int, phys_mem_size: int) -> int:
        return num_phys_mem * phys_mem_size // slot_size


@final
class ArenaSlotPool(SlotPoolBase):
    """Per-pool *view* over one plane of a :class:`SequenceArena` (contiguous
    primary KV cache, DESIGN.md §4.1). Exists so the ``PoolGroupBase``
    machinery (addresses, sizes, statistics) works unchanged in arena mode.

    The arena owns the VA reservation and all mappings; this view owns
    nothing. ``num_slots``/``num_bytes`` describe the *VA* extent (block-index
    capacity), not physical residency -- physical capacity is governed by the
    shared :class:`PageBudget`.
    """

    __slots__ = ("_arena", "_pool_index")
    _arena: SequenceArena
    _pool_index: int

    def __init__(self, arena: SequenceArena, pool_index: int, slot_size: int) -> None:
        super().__init__(slot_size)
        self._arena = arena
        self._pool_index = pool_index

    @override
    def destroy(self) -> None:
        pass  # the owning ArenaPoolGroup destroys the arena

    @override
    def resize(self, new_num_slots: int) -> None:
        raise LogicError(
            "arena-backed pools have a fixed VA extent; physical capacity is "
            "governed by the shared page budget"
        )

    @override
    def slot_address(self, slot: int) -> MemAddress:
        return MemAddress(int(self._arena.base_address(self._pool_index)) + self.slot_size * slot)

    @property
    @override
    def num_slots(self) -> int:
        return self._arena.capacity_blocks


class HostSlotPool(SlotPoolBase):
    __slots__ = ("_host_mem",)
    _host_mem: HostMem

    def __init__(self, slot_size: int, num_slots: int) -> None:
        super().__init__(slot_size)
        self._host_mem = HostMem(self.aligned_size(num_slots))

    @override
    def destroy(self) -> None:
        self._host_mem.destroy()

    @override
    def resize(self, new_num_slots: int) -> None:
        self._host_mem.resize(self.aligned_size(new_num_slots))

    @override
    def slot_address(self, slot: int) -> MemAddress:
        return MemAddress(self._host_mem._address + self.slot_size * slot)

    @property
    @override
    def num_slots(self) -> int:
        return self._host_mem.size // self.slot_size

    def aligned_size(self, num_slots: int) -> int:
        return round_up(num_slots * self.slot_size, HostMem.ALIGNMENT)


class DiskSlotPool(SlotPoolBase):
    __slots__ = ("_filename", "_fd")
    # Currently only used to get the parent folder where we create temporary files.
    # You won't find file with this name.
    filename: str
    _fd: FileDescriptor

    def __init__(self, filename: str, slot_size: int, num_slots: int) -> None:
        super().__init__(slot_size)
        self.filename = filename
        folder = os.path.dirname(filename)
        assert os.path.isdir(folder), f"Folder {folder} does not exist"
        try:
            fd = os.open(folder, os.O_TMPFILE | os.O_RDWR | os.O_EXCL, 0o664)
        except OSError as e:
            if e.errno != errno.EOPNOTSUPP:
                raise
            # Fallback for filesystems/architectures not supporting O_TMPFILE
            fd, path = tempfile.mkstemp(dir=folder)
            try:
                os.unlink(path)
            except OSError:
                os.close(fd)
                raise
        self._fd = FileDescriptor(fd)
        self.resize(num_slots)

    @override
    def destroy(self) -> None:
        if self.fd == BAD_FILE_DESCRIPTOR:
            return
        os.close(self.fd)
        self._fd = BAD_FILE_DESCRIPTOR

    @property
    def fd(self) -> FileDescriptor:
        return self._fd

    @property
    def file_size(self) -> int:
        return os.lseek(self.fd, 0, os.SEEK_END)

    @override
    def resize(self, new_num_slots: int) -> None:
        file_size = new_num_slots * self.slot_size
        resize_file(self.fd, file_size)

    @override
    def slot_address(self, slot: int) -> DiskAddress:
        assert slot < self.num_slots
        return DiskAddress(self.fd, slot * self.slot_size)

    @property
    @override
    def num_slots(self) -> int:
        return self.file_size // self.slot_size


@dataclass(slots=True)
class Slot:
    # ready_event indicates whether the slot is ready for use.
    #  For newly allocated BlockData, it indicates finish of the last usage by the previous owners of the
    #  slot (who returned the slot to the pool).
    #  After data migration, it indicates finish of data migration.
    #  When passed to release(), it indicates finish of usage by the current owners of the slot.
    _slot_id: SlotId | None
    ready_event: CachedCudaEvent

    @property
    def slot_id(self) -> SlotId:
        assert self._slot_id is not None
        return self._slot_id

    def query_ready(self) -> bool:
        ev = self.ready_event
        if ev is CachedCudaEvent.NULL:
            return True
        ret = ev.query_complete()
        if ret:
            self.ready_event = CachedCudaEvent.NULL
        return ret

    @property
    def has_valid_slot(self) -> bool:
        return self._slot_id is not None

    def move_to_new_slot(self) -> "Slot":
        ret = Slot(None, CachedCudaEvent.NULL)
        ret.set_slot(self)
        return ret

    def set_slot(self, slot: "Slot") -> None:
        if self.has_valid_slot:
            raise LogicError("Slot is already set.")
        self._slot_id = slot.slot_id
        self.ready_event = slot.ready_event
        slot._slot_id = None
        slot.ready_event = CachedCudaEvent.NULL

    def __del__(self) -> None:
        if self.has_valid_slot:
            warnings.warn("[KVCacheManager] slot is not freed before deletion")


class SlotAllocator:
    __slots__ = (
        "_capacity",
        "_num_active_slots",
        "_recycled_slots",
        "_num_ready_recycled_slots",
        "_occupied_mask",
        "_target_capacity",
        "_overflow_slots",
    )
    _capacity: int
    _num_active_slots: int  # active slots are either in use or recycled.
    _recycled_slots: deque[
        Slot
    ]  # only store recycled slots to avoid excessive memory usage on program start
    _num_ready_recycled_slots: int  # number of recycled slots that are ready to be used immediately
    # (no need for sync or wait in stream), i.e. their ready events are triggered.
    _occupied_mask: DynamicBitset

    # for scheduled shrinking resize
    _target_capacity: (
        int  # _target_capacity <= _capacity. Inequal if a shrinking resize is in progress.
    )
    _overflow_slots: list[
        Slot
    ]  # slots that will be out-of-range after a in-progress resize. scheduled for removal.

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._num_active_slots = 0
        self._recycled_slots = deque[Slot]()
        self._num_ready_recycled_slots = 0
        self._occupied_mask = DynamicBitset(capacity)
        self._target_capacity = capacity
        self._overflow_slots = []

    def __del__(self) -> None:
        assert_critical(
            self._num_ready_recycled_slots == len(self._recycled_slots),
            "did you call synchronize()?",
        )
        assert_critical(
            self._target_capacity == self._capacity and not self._overflow_slots,
            "resize is in progress",
        )
        assert_critical(self._occupied_mask.num_set_bits == 0, "some slots are still in use")
        assert_critical(
            len(self._recycled_slots) == self._num_active_slots, "some slots are not free"
        )

    @property
    def num_free_slots(self) -> int:
        return len(self._recycled_slots) + max(self._target_capacity - self._num_active_slots, 0)

    @property
    def num_occupied_slots(self) -> int:
        return self._occupied_mask.num_set_bits

    def allocate(self) -> Slot:
        if self.num_free_slots == 0:
            raise OutOfPagesError("No free slots")
        self._scrub_events()
        # prefererence: ready recycled slots > new slots > recycled slots that are not ready
        if self._num_ready_recycled_slots > 0:
            assert self._recycled_slots
            slot = self._recycled_slots.popleft()
            assert slot.has_valid_slot
            self._num_ready_recycled_slots -= 1
            assert slot.ready_event is CachedCudaEvent.NULL
        elif self._num_active_slots < min(self.num_slots, self._target_capacity):
            slot = Slot(SlotId(self._num_active_slots), CachedCudaEvent.NULL)
            self._num_active_slots += 1
        else:
            slot = self._recycled_slots.popleft()
            assert slot.has_valid_slot
        self._occupied_mask.set(slot.slot_id)
        return slot

    # The reason why we don't use allocate() multiple times is that if what user need is all or none,
    # and when we don't have enough free slots, we will free these newly allocated slots by appending
    # them to the back of the recycled slot queue, which may impact perf.
    def allocate_multiple(self, num_slots: int) -> list[Slot]:
        if self.num_free_slots < num_slots:
            raise OutOfPagesError("Not enough free slots")
        return [self.allocate() for _ in range(num_slots)]

    def release(self, slot: Slot) -> None:
        assert slot.has_valid_slot
        slot = slot.move_to_new_slot()
        if slot.slot_id >= self._capacity or not self._occupied_mask.get(slot.slot_id):
            raise LogicError(f"Slot {slot.slot_id} is not occupied")
        assert type(slot) is Slot and slot.has_valid_slot
        if slot.slot_id < self._target_capacity:
            self._recycled_slots.append(slot)
        else:
            self._overflow_slots.append(slot)
        self._occupied_mask.clear(slot.slot_id)
        self._scrub_events()
        assert NDEBUG or self._check()

    @property
    def num_slots(self) -> int:
        return self._capacity

    def expand(self, new_num_slots: int) -> None:
        assert NDEBUG or self._check()
        assert self._target_capacity == self._capacity
        old_num_slots = self._capacity
        assert new_num_slots > old_num_slots
        self._occupied_mask.resize(new_num_slots)
        self._capacity = new_num_slots
        self._target_capacity = self._capacity
        assert NDEBUG or self._check()

    def prepare_for_shrink(self, new_num_slots: int) -> None:
        assert NDEBUG or self._check()
        assert self._target_capacity == self._capacity
        assert new_num_slots < self._capacity
        new_recycled_slots = deque[Slot]()
        new_num_ready_recycled_slots = 0
        old_num_ready_recycled_slots = self._num_ready_recycled_slots
        for i, slot in enumerate(self._recycled_slots):
            if slot.slot_id < new_num_slots:
                new_recycled_slots.append(slot)
                if i < old_num_ready_recycled_slots:
                    new_num_ready_recycled_slots += 1
            else:
                self._overflow_slots.append(slot)
        self._recycled_slots = new_recycled_slots
        self._num_ready_recycled_slots = new_num_ready_recycled_slots
        self._target_capacity = new_num_slots
        assert NDEBUG or self._check()

    @property
    def shrink_in_progress(self) -> bool:
        "Indicates if a scheduled shrink is in progress."
        assert self._target_capacity <= self._capacity
        return self._target_capacity < self._capacity

    def finish_shrink(self) -> bool:
        assert NDEBUG or self._check()
        # Overflow-range IDs that were ever issued are exactly
        # max(0, _num_active_slots - _target_capacity); the underused case
        # (_num_active_slots <= _target_capacity) collapses to zero.
        expected_overflow = max(0, self._num_active_slots - self._target_capacity)
        if self.shrink_in_progress and len(self._overflow_slots) == expected_overflow:
            assert len(set(s.slot_id for s in self._overflow_slots)) == len(self._overflow_slots), (
                "Some slots are still in use."
            )
            for ev in set(s.ready_event for s in self._overflow_slots):
                ev.synchronize()
            for slot in self._overflow_slots:
                slot.ready_event = CachedCudaEvent.NULL
                slot._slot_id = None
            self._overflow_slots.clear()
            self._capacity = self._target_capacity
            self._num_active_slots = min(self._num_active_slots, self._capacity)
            self._scrub_events()
            assert NDEBUG or self._check()
            return True
        raise RuntimeError("shrink can't be finished")

    def get_slots_blocking_shrink(self) -> HomoTuple[SlotId]:
        return tuple(
            SlotId(id)
            for id in range(self._target_capacity, self._capacity)
            if self._occupied_mask.get(id)
        )

    def _scrub_events(self) -> None:
        self._num_ready_recycled_slots = self._scrub_events_impl(
            self._recycled_slots, self._num_ready_recycled_slots
        )

    def _check(self) -> bool:
        return (
            self._num_active_slots <= self._capacity
            and self._target_capacity <= self._capacity
            and (self.shrink_in_progress or len(self._overflow_slots) == 0)
            and all(
                self._target_capacity <= slot.slot_id < self._capacity
                for slot in self._overflow_slots
            )
            and len(self._recycled_slots) + len(self._overflow_slots) + self.num_occupied_slots
            == self._num_active_slots
        )

    @staticmethod
    def _scrub_events_impl(slots: Sequence[Slot], num_ready: int) -> int:
        assert num_ready <= len(slots)
        for i in range(num_ready, len(slots)):
            slot = slots[i]
            if slot.ready_event.query_complete():
                slot.ready_event = CachedCudaEvent.NULL
                num_ready += 1
            else:
                break
        return num_ready

    def _synchronize(self) -> None:
        "synchronize the events of all unused slots"
        while self._num_ready_recycled_slots != len(self._recycled_slots):
            self._scrub_events()


class PoolGroupBase:
    __slots__ = ("_slot_allocator", "_pools", "_destroyed")

    _slot_allocator: SlotAllocator
    _pools: TypedIndexList[PoolIndex, SlotPoolBase]
    _destroyed: bool

    def __init__(self, num_slots: int) -> None:
        self._slot_allocator = SlotAllocator(num_slots)
        self._destroyed = False

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        if self._destroyed:
            return
        allocator = self._slot_allocator
        if allocator._capacity != 0:
            allocator._synchronize()
            allocator.prepare_for_shrink(0)
            allocator.finish_shrink()
        for pool in self._pools:
            pool.destroy()
        self._destroyed = True

    @property
    def num_pools(self) -> PoolIndex:
        return PoolIndex(len(self._pools))

    @property
    def num_slots(self) -> int:
        num_slots = self._slot_allocator._capacity
        assert num_slots <= self._get_num_slots_from_pools()
        return num_slots

    @property
    def num_free_slots(self) -> int:
        return self._slot_allocator.num_free_slots

    @property
    def num_bytes(self) -> int:
        return sum(pool.num_bytes for pool in self._pools)

    def resize_pools(self, new_num_slots: int | None) -> None:
        """
        Resize the pools, but not the slot allocator. If new_num_slots is None, make pool sizes match
        the slot allocator.
        If exception is raised, size of pools may be imbalanced. Call resize_pools() again with None or
        self._get_num_slots_from_pools() to fix it.
        """
        if new_num_slots is None:
            new_num_slots = self._slot_allocator.num_slots
        for pool in self._pools:
            pool.resize(new_num_slots)
        assert NDEBUG or self._check(True)

    def allocate(self) -> Slot:
        return self._slot_allocator.allocate()

    def allocate_multiple(self, num_slots: int) -> list[Slot]:
        return self._slot_allocator.allocate_multiple(num_slots)

    def release(self, slot: Slot) -> None:
        self._slot_allocator.release(slot)

    def slot_address(self, slot_id: SlotId) -> HomoTuple[Address]:
        return tuple(pool.slot_address(slot_id) for pool in self._pools)

    @property
    def slot_size(self) -> TypedIndexList[PoolIndex, int]:
        return typed_map(self._pools, lambda pg: pg.slot_size)

    def _check(self, allow_mismatch: bool = False) -> bool:
        pool_num_slots = self._get_num_slots_from_pools()
        return (
            self._slot_allocator.num_slots <= pool_num_slots
            if allow_mismatch
            else self._slot_allocator.num_slots == pool_num_slots
        )

    def _get_num_slots_from_pools(self) -> int:
        return min(p.num_slots for p in self._pools)

    @staticmethod
    def _compute_num_phys_mem(
        slot_size_list: Sequence[int], num_slots: int, phys_mem_size: int
    ) -> HomoTuple[int]:
        return tuple(
            GpuSlotPool._compute_num_phys_mem(slot_size, num_slots, phys_mem_size)
            for slot_size in slot_size_list
        )


class GpuPoolGroup(PoolGroupBase):
    __slots__ = ()

    def __init__(
        self,
        num_slots: int,
        slot_size_list: TypedIndexList[PoolIndex, int],
        shared_phys_mem_pool: PooledPhysMemAllocator,
    ):
        super().__init__(num_slots)
        total_gpu_memory = query_total_gpu_memory()
        max_slot_size = max(slot_size_list)
        phys_mem_size = shared_phys_mem_pool.phys_mem_size
        self._pools = typed_map(
            slot_size_list,
            lambda slot_size: GpuSlotPool(
                slot_size,
                round_down(int(total_gpu_memory * slot_size / max_slot_size), phys_mem_size),
                shared_phys_mem_pool,
                num_slots,
            ),
        )


class SequenceRange:
    """Handle for one sequence's contiguous block-index range in an
    :class:`ArenaPoolGroup` (DESIGN.md §4.1).

    Tracks the slots issued from the range and the CUDA events that must
    complete before the range's physical pages can be unmapped (deferred
    reclaim, §4.2): the ready events of released slots (e.g. D2H write-out
    copies) plus the sequence's last-consumer event passed to
    ``free_sequence``.
    """

    __slots__ = (
        "base_block",
        "num_blocks",
        "_outstanding",
        "_freed",
        "_gate_events",
        "_free_event",
        "_reclaim_fence",
        "_ghost_keys",
        "_alias_span_key",
    )
    base_block: int
    num_blocks: int  # reserved (padded) length in blocks
    _outstanding: int  # slots issued and not yet released
    _freed: bool
    _gate_events: list[CachedCudaEvent]  # pending ready events of released slots
    _free_event: CachedCudaEvent
    # Execution-stream fence recorded at the first drain point after the free
    # (DESIGN.md §4.2, risk #3). The overlap scheduler may have speculatively
    # enqueued a step that still reads this range when free_sequence runs; the
    # free/gate events do not order against that step. A fence recorded on the
    # execution stream at a later drain point covers every launch that could
    # reference the range (steps enqueued after it no longer contain the
    # sequence). None until the first fenced drain sees this range.
    _reclaim_fence: CachedCudaEvent | None
    # Ghost registry keys (§4.4 phase 2): ids of host pages whose bytes still
    # live in this (retained) range; purged when the range is reclaimed.
    _ghost_keys: list[int]
    # P3 prefix aliasing: (registry key, aliased block extent) of the span
    # alias-mapped at this range's start (None = unaliased). Aliased ranges
    # may only be adopted by a sequence whose reuse match carries the SAME
    # signature -- the span's pages are shared, so an adopter writing past a
    # SHORTER matched prefix (or any other-key adopter writing the range
    # head) would corrupt every alias of the span.
    _alias_span_key: "tuple[int, int] | None"

    def __init__(self, base_block: int, num_blocks: int) -> None:
        self.base_block = base_block
        self.num_blocks = num_blocks
        self._outstanding = 0
        self._freed = False
        self._gate_events = []
        self._free_event = CachedCudaEvent.NULL
        self._reclaim_fence = None
        self._ghost_keys = []
        self._alias_span_key = None

    @property
    def end_block(self) -> int:
        return self.base_block + self.num_blocks

    def _scrub_gate_events(self) -> None:
        self._gate_events = [ev for ev in self._gate_events if not ev.query_complete()]

    def _ready_to_reclaim(self) -> bool:
        if not self._freed or self._outstanding != 0:
            return False
        if self._reclaim_fence is None or not self._reclaim_fence.query_complete():
            return False
        if not self._free_event.query_complete():
            return False
        self._scrub_gate_events()
        return not self._gate_events


@final
class ArenaPoolGroup(PoolGroupBase):
    """GPU pool group backed by a :class:`SequenceArena` instead of a
    ``SlotAllocator`` (contiguous primary KV cache, DESIGN.md §4.1-§4.2).

    Allocation is *per-sequence*: :meth:`reserve_sequence` carves a contiguous,
    page-aligned block-index range sized for the sequence's maximum block
    count; :meth:`take_slot` then issues ``Slot`` objects with
    ``slot_id = base_block + ordinal``, so every downstream consumer of slot
    ids (pages, offset tables, ``slot_address``) works unchanged -- but the ids
    a sequence sees are consecutive. Physical pages are mapped on demand via
    :meth:`ensure_mapped` against the level-wide :class:`PageBudget`.

    Scattered allocation (``allocate``/``allocate_multiple``) is retired at the
    GPU level in arena mode and raises ``LogicError``: growth goes through the
    owning sequence's range, and reuse onboarding copies into explicit
    destinations inside that range (§4.4).
    """

    __slots__ = (
        "_arena",
        "_ranges",
        "_range_bases",
        "_pending_reclaim",
        "_budget",
        "_lazy_retention",
        "_retained",
        "_ghosts",
        "ghost_hits",
        "ghost_misses",
        "spilled_ranges",
        "_pending_maps",
        "_range_adoption",
        "_prefix_aliasing",
        "_canonical_spans",
        "alias_hits",
        "alias_misses",
        "spilled_spans",
        "dedup_remapped_pages",
    )
    _arena: SequenceArena
    _ranges: dict[int, SequenceRange]  # base_block -> live range
    _range_bases: list[int]  # sorted, for slot_id -> range lookup on release
    _pending_reclaim: list[SequenceRange]
    _budget: PageBudget
    _lazy_retention: bool
    # Lazily retained freed ranges (§4.4 phase 2), insertion-ordered = LRU by
    # retire time. Pages stay mapped; reclaimed only under pressure.
    _retained: dict[int, SequenceRange]
    # id(host page) -> (identity ref, retained-range base, ordinal): where a
    # committed block's bytes still live on GPU after its owner closed.
    _ghosts: "dict[int, tuple[rawref.ref[Any], int, int]]"
    # Freed-range adoption (module default from TRTLLM_KV_ARENA_RANGE_ADOPTION;
    # instance attribute so tests can pin either behavior).
    _range_adoption: bool
    # P3 prefix aliasing (module default from TRTLLM_KV_ARENA_PREFIX_ALIASING,
    # default OFF for the prototype; instance attribute for tests).
    _prefix_aliasing: bool
    # Canonical-span registry (P3): id(head canonical page) -> _CanonicalSpan.
    # A span pins the refcounted physical pages holding a closed canonical
    # owner's fully-committed prefix, so same-prefix admissions alias them
    # (zero-copy, zero-charge) instead of copying from the host tier.
    _canonical_spans: "dict[int, _CanonicalSpan]"

    def __init__(
        self,
        block_capacity: int,
        slot_size_list: TypedIndexList[PoolIndex, int],
        shared_phys_mem_pool: PooledPhysMemAllocator,
        page_budget: PageBudget,
        map_ahead_pages: int,
        page_index_scale: int,
        lazy_retention: bool = False,
        prefix_aliasing: bool = False,
    ) -> None:
        # The int31 kernel-offset ceiling must hold for every offset this
        # arena can emit (DESIGN.md §4.1); fail loudly at startup.
        check_index_width(block_capacity, page_index_scale)
        # SlotAllocator is retired at the GPU level in arena mode; the base
        # class gets an empty one so its teardown logic is a no-op.
        super().__init__(0)
        self._arena = SequenceArena(
            block_capacity,
            tuple(slot_size_list),
            shared_phys_mem_pool,
            map_ahead_pages,
            page_budget,
        )
        arena = self._arena
        self._pools = make_typed(
            lambda pool_idx: ArenaSlotPool(arena, pool_idx, slot_size_list[pool_idx]),
            typed_len(slot_size_list),
        )
        self._ranges = {}
        self._range_bases = []
        self._pending_reclaim = []
        self._budget = page_budget
        self._lazy_retention = lazy_retention
        self._retained = {}
        self._ghosts = {}
        self.ghost_hits = 0
        self.ghost_misses = 0
        self.spilled_ranges = 0
        self._range_adoption = _RANGE_ADOPTION
        self._prefix_aliasing = prefix_aliasing or _PREFIX_ALIASING
        self._canonical_spans = {}
        self.alias_hits = 0
        self.alias_misses = 0
        self.spilled_spans = 0
        self.dedup_remapped_pages = 0
        # base_block -> [range, target frontier (blocks), pages charged]:
        # growth maps deferred to the per-iteration batched sweep (§4.2).
        self._pending_maps: dict[int, list[Any]] = {}

    @override
    def destroy(self) -> None:
        # The int31 check makes __init__ raise before base construction; a
        # partially constructed group has nothing to tear down.
        if not hasattr(self, "_destroyed") or self._destroyed:
            return
        # Deferred maps that never ran only hold budget; return it.
        for _, (_, _, charged, _) in self._pending_maps.items():
            self._budget.release(charged)
        self._pending_maps.clear()
        # Teardown can run right after sequences finished, with their ranges
        # still in the event-gated deferred-reclaim queue (§4.2). Teardown is
        # allowed to block: synchronize the gates and drain, then anything
        # left besides retained ranges is a genuine leak.
        for rng in self._pending_reclaim:
            rng._free_event.synchronize()
            for ev in rng._gate_events:
                ev.synchronize()
            # Teardown blocks until the device is quiet, so the speculative-step
            # fence is moot; substitute the (complete) null event for unfenced
            # ranges and synchronize real ones.
            if rng._reclaim_fence is None:
                rng._reclaim_fence = CachedCudaEvent.NULL
            else:
                rng._reclaim_fence.synchronize()
        self.drain_reclaim()
        for rng in list(self._retained.values()):
            for ev in rng._gate_events:
                ev.synchronize()
        assert self.spill_retained(1 << 62) >= 0
        assert not self._retained
        assert not self._ranges and not self._pending_reclaim, (
            "destroying ArenaPoolGroup with live sequence ranges"
        )
        # Unpin canonical spans (P3) so their handles return to the pool.
        for key in list(self._canonical_spans):
            self._unpin_span(self._canonical_spans.pop(key))
        super().destroy()
        self._arena.destroy()

    # -- per-sequence API (replaces scattered slot allocation) --------------

    def reserve_sequence(
        self, max_blocks: int, alias_key: "tuple[int, int] | None" = None
    ) -> SequenceRange:
        """Reserve a contiguous block-index range for a sequence's maximum
        block count, preferring the adoption of a parked (freed but still
        mapped) range over fresh VA: the adopted range's mapped prefix is
        reused as-is, so the steady state issues no ``cuMemMap``/``cuMemUnmap``
        at all (see ``_try_adopt``). A fresh reservation is pure VA
        bookkeeping; no physical memory is mapped.

        ``alias_key`` is the admission's canonical-span registry key (P3):
        adoption is prefix-affine -- a parked range whose head alias-maps span
        ``K`` may only go to an admission with the same key (its prefix
        content and mappings are exactly what the adopter wants, and any
        other adopter writing the range head would corrupt every alias of the
        shared pages).
        Raises ``MemoryError`` on VA exhaustion (see DESIGN.md §4.8 sizing)."""
        adopted = self._try_adopt(max_blocks, alias_key)
        if adopted is not None:
            return adopted
        base_block = self._arena.reserve(max_blocks)
        rng = SequenceRange(base_block, self._arena.reserved_len(base_block))
        self._ranges[base_block] = rng
        insort(self._range_bases, base_block)
        return rng

    def _try_adopt(
        self, max_blocks: int, alias_key: "tuple[int, int] | None" = None
    ) -> "SequenceRange | None":
        """Hand a parked freed range (pages still mapped, reclaim gates all
        complete) to a new sequence. LRU-first, so the most recently retired
        ranges keep their ghost entries (D2D reuse, §4.4 phase 2) longest.
        The adopted range keeps its mapped prefix and budget charge; only its
        ghost entries are purged (its bytes are about to be overwritten).
        Alias-signature filter (P3): see :meth:`reserve_sequence`.
        Returns None when nothing parked fits ``max_blocks``."""
        if not self._range_adoption:
            return None
        for base in self._retained:
            rng = self._retained[base]
            if rng.num_blocks < max_blocks:
                continue  # heterogeneous sizing: this range cannot hold it
            if rng._alias_span_key != alias_key:
                continue  # prefix-affine: shared span pages must not be overwritten
            if not rng._ready_to_reclaim():
                continue  # e.g. an in-flight D2D onboard still reads from it
            # With lazy retention, a range whose bytes still serve the ghost
            # registry is worth more as a D2D reuse source than as adopted
            # VA; keep it parked (pressure spill trims it eventually). Prune
            # superseded keys so stale entries don't block adoption forever.
            if rng._ghost_keys:
                live: list[int] = []
                for key in rng._ghost_keys:
                    entry = self._ghosts.get(key)
                    if entry is not None and entry[1] == base:
                        live.append(key)
                rng._ghost_keys = live
                if live:
                    continue
            del self._retained[base]
            self._budget.unretain(self._arena.charged_pages_in_range(base))
            # Fresh handle, same extent: the allocator reservation and the
            # arena's mapped frontier (including any alias-mapped span, whose
            # signature carries over) stay untouched.
            fresh = SequenceRange(base, rng.num_blocks)
            fresh._alias_span_key = rng._alias_span_key
            self._ranges[base] = fresh
            return fresh
        return None

    def take_slot(self, rng: SequenceRange, ordinal: int) -> Slot:
        """Issue the slot at ``rng.base_block + ordinal``. The caller (the
        sequence growth path) guarantees each ordinal is taken at most once
        while the range is live."""
        assert not rng._freed, "cannot take slots from a freed range"
        assert 0 <= ordinal < rng.num_blocks
        rng._outstanding += 1
        # Newly mapped arena pages have no previous owner, so no ready event.
        return Slot(SlotId(rng.base_block + ordinal), CachedCudaEvent.NULL)

    def ensure_mapped(self, rng: SequenceRange, num_valid_blocks: int) -> int:
        """Map physical pages (in every pool) covering the first
        ``num_valid_blocks`` of the range plus the map-ahead margin. Returns
        the number of newly mapped pages; raises ``OutOfPagesError`` (mapping
        nothing) if the page budget is exhausted (§4.6)."""
        assert num_valid_blocks <= rng.num_blocks
        return self._arena.ensure_mapped(rng.base_block, num_valid_blocks)

    # -- batched map sweep (§4.2): defer growth maps to one pass/iteration --

    def queue_mapping(self, rng: SequenceRange, num_valid_blocks: int) -> int:
        """Charge the page budget for the pages a target frontier needs (so
        admission control behaves exactly like an immediate map) but defer the
        actual ``cuMemMap``/``cuMemSetAccess`` calls to :meth:`flush_mappings`.
        The map-ahead margin degrades to zero before the charge fails (the
        margin is a latency-hiding hint -- see ``SequenceArena.ensure_mapped``);
        the margin actually charged is recorded on the entry so the flush
        replays the exact same plan. Raises ``OutOfPagesError`` (charging
        nothing) only if the frontier alone does not fit. Returns the pages
        charged. The caller MUST run the flush before any GPU work touches
        the new blocks."""
        assert not rng._freed and num_valid_blocks <= rng.num_blocks
        base = rng.base_block
        entry = self._pending_maps.get(base)
        if entry is None:
            margin = -1
            pages = self._arena.pending_pages(base, num_valid_blocks)
            if not pages:
                return 0
            try:
                self._budget.consume(pages)
            except OutOfPagesError:
                margin = 0
                pages = self._arena.pending_pages(base, num_valid_blocks, 0)
                if not pages:
                    return 0
                self._budget.consume(pages)  # raises; nothing recorded yet
            self._pending_maps[base] = [rng, num_valid_blocks, pages, margin]
            return pages
        if num_valid_blocks <= entry[1]:
            return 0
        # Nothing was mapped since the first queue call, so pending_pages is
        # monotone in the frontier (at a fixed margin) and the delta charge
        # is exact.
        charged: int = entry[2]
        margin: int = entry[3]
        pages = self._arena.pending_pages(base, num_valid_blocks, margin)
        delta = pages - charged
        assert delta >= 0
        if delta:
            try:
                self._budget.consume(delta)
            except OutOfPagesError:
                if margin == 0:
                    raise
                # Degrade this entry's margin. The margin-less plan for the
                # new target can even be SMALLER than what is already charged
                # (margin wider than the added blocks): release the excess so
                # the invariant charged == pending_pages(target, margin)
                # holds exactly for the flush.
                pages = self._arena.pending_pages(base, num_valid_blocks, 0)
                delta = pages - charged
                if delta > 0:
                    self._budget.consume(delta)  # raises; entry unchanged
                elif delta < 0:
                    self._budget.release(-delta)
                entry[3] = 0
        entry[1] = num_valid_blocks
        entry[2] = pages
        return delta

    def flush_mappings(self) -> int:
        """Execute all deferred maps back-to-back (§4.2's batched
        per-iteration sweep). Entries whose range was freed in the meantime
        release their charge instead. Returns pages mapped."""
        if not self._pending_maps:
            return 0
        mapped_total = 0
        for base, (rng, target, charged, margin) in self._pending_maps.items():
            if rng._freed or base not in self._ranges:
                self._budget.release(charged)
                continue
            mapped = self._arena.ensure_mapped(base, target, precharged=True, margin=margin)
            mapped_total += mapped
            if mapped != charged:
                # Defensive: the plan is deterministic, so this indicates
                # concurrent mutation; reconcile the budget either way.
                if mapped < charged:
                    self._budget.release(charged - mapped)
                else:
                    self._budget.consume(mapped - charged)
        self._pending_maps.clear()
        return mapped_total

    def free_sequence(self, rng: SequenceRange, last_consumer: CachedCudaEvent) -> None:
        """Queue the whole range for deferred reclaim (§4.2). The range is
        unmapped and its block indices reused only once ``last_consumer``, the
        ready events of all released slots, and the release of every
        outstanding slot have all happened -- see :meth:`drain_reclaim`."""
        assert not rng._freed, "double free of a sequence range"
        if rng._alias_span_key is not None:
            # Single choke point for the owner-live -> registry-charged flip
            # (P3 v2): close, evacuate, and error unwinds all free the range
            # here, and an owner-live span must not outlive its range's
            # charge (the registry pin keeps the physical handles alive, so
            # unflipped it would hold pages the budget no longer counts).
            self._commit_live_span_charge(rng)
        rng._freed = True
        rng._free_event = last_consumer
        self._pending_reclaim.append(rng)

    def drain_reclaim(
        self, fence: CachedCudaEvent | None = None, wait: bool = False, quiesce: bool = True
    ) -> int:
        """Process every freed range whose gating conditions have all
        completed: unmap and recycle it, or -- with lazy retention (§4.4
        phase 2) -- keep its pages mapped and park it on the retained LRU,
        to be reclaimed only under pressure (:meth:`spill_retained`).
        Returns the number of ranges processed. Call at iteration
        boundaries. ``quiesce=False`` skips the per-range device quiesce
        (hard-reclaim path, P3 v3: the caller quiesced once for the batch).

        ``fence`` must be an event recorded on the *execution* stream at the
        call site; it is attached to ranges freed since the previous fenced
        drain and gates their reclaim (risk #3: a speculatively enqueued step
        may still read a just-freed range — see ``SequenceRange._reclaim_fence``).
        Unfenced ranges are never reclaimed, so fenced drains must happen
        periodically (the pyexecutor adapter fences every iteration).

        With ``wait=True`` the call BLOCKS until each pending range's gating
        events complete (fence, free event, released-slot gates) instead of
        skipping ranges whose events are still in flight. This is the §4.6
        preemption-headroom escape hatch: the scheduler's eviction path frees
        ranges in the middle of a scheduling pass — before the iteration-
        boundary fenced drain could possibly run — and without a blocking
        drain those pages stay charged to the budget, admission keeps
        failing, and a fully suspended batch deadlocks (every resume refused
        by the utilization gate while nothing is left to evict). The wait is
        bounded: the gates are the just-enqueued evacuation copies plus
        already-enqueued forward steps. Ranges with outstanding slots are
        skipped (their release is host-side state, not an event)."""
        if fence is not None:
            self.fence_pending_frees(fence)
        if wait:
            for rng in self._pending_reclaim:
                if rng._outstanding != 0 or rng._reclaim_fence is None:
                    continue
                rng._reclaim_fence.synchronize()
                rng._free_event.synchronize()
                for ev in rng._gate_events:
                    ev.synchronize()
        remaining: list[SequenceRange] = []
        processed = 0
        for rng in self._pending_reclaim:
            if rng._ready_to_reclaim():
                if self._lazy_retention or self._range_adoption:
                    # Park the range with its pages mapped: adoption hands it
                    # whole to the next admitted sequence (no unmap+remap
                    # churn against in-flight kernels); with lazy retention
                    # its ghosts additionally serve D2D reuse. Unmapped only
                    # under pressure (:meth:`spill_retained`).
                    self._retained[rng.base_block] = rng
                    self._budget.retain(self._arena.charged_pages_in_range(rng.base_block))
                else:
                    self._reclaim_range(rng, quiesce)
                processed += 1
            else:
                remaining.append(rng)
        self._pending_reclaim = remaining
        return processed

    def fence_pending_frees(self, fence: CachedCudaEvent) -> None:
        """Attach ``fence`` (an event recorded on the execution stream at an
        iteration boundary) to every pending freed range that has none yet.
        Assign-only variant of the ``fence`` parameter of
        :meth:`drain_reclaim`."""
        for rng in self._pending_reclaim:
            if rng._reclaim_fence is None:
                rng._reclaim_fence = fence

    def _reclaim_range(self, rng: SequenceRange, quiesce: bool = True) -> None:
        """Unmap a range's pages, return its block indices, purge its ghosts."""
        for key in rng._ghost_keys:
            entry = self._ghosts.get(key)
            # A newer registration (e.g. a later sequence's private copy of
            # the same canonical block) may have superseded this range's
            # entry; only purge entries that still point here.
            if entry is not None and entry[1] == rng.base_block:
                self._ghosts.pop(key)
        rng._ghost_keys.clear()
        self._arena.reclaim(rng.base_block, quiesce)
        del self._ranges[rng.base_block]
        idx = bisect_right(self._range_bases, rng.base_block) - 1
        assert self._range_bases[idx] == rng.base_block
        self._range_bases.pop(idx)

    def spill_retained(self, min_pages: int, quiesce: bool = True) -> int:
        """Reclaim retained ranges (LRU first, skipping ranges with pending
        gate events) until at least ``min_pages`` physical pages have been
        freed or nothing more is spillable. Returns pages freed.
        ``quiesce=False`` skips the per-range device quiesce (hard-reclaim
        path, P3 v3: the caller quiesced once for the whole batch -- fixes
        the per-range re-sync of the old ladder)."""
        freed = 0
        for base in list(self._retained):
            if freed >= min_pages:
                break
            rng = self._retained[base]
            if not rng._ready_to_reclaim():
                continue  # e.g. an in-flight D2D onboard still reads from it
            pages = self._arena.charged_pages_in_range(base)
            del self._retained[base]
            self._budget.unretain(pages)
            self._reclaim_range(rng, quiesce)
            self.spilled_ranges += 1
            freed += pages
        return freed

    @property
    def retained_pages(self) -> int:
        return sum(self._arena.charged_pages_in_range(base) for base in self._retained)

    # -- ghost registry (§4.4 phase 2): D2D reuse from retained ranges ------

    def register_ghosts(self, rng: SequenceRange, entries: "Sequence[tuple[int, object]]") -> None:
        """Record that the bytes of ``page`` (now living in the stale tier)
        are still present at ``rng.base_block + ordinal`` for each
        ``(ordinal, page)`` entry. Valid until the range is reclaimed."""
        if not self._lazy_retention:
            return
        for ordinal, page in entries:
            key = id(page)
            self._ghosts[key] = (rawref.ref(page), rng.base_block, ordinal)
            rng._ghost_keys.append(key)

    def lookup_ghost(self, page: object) -> "tuple[SlotId, SequenceRange] | None":
        """If ``page``'s bytes are still resident in a *retained* range,
        return (slot id of the resident copy, the retained range) -- the
        caller must gate the range's reclaim on its copy (append to
        ``_gate_events``). Returns None on any staleness."""
        entry = self._ghosts.get(id(page))
        if entry is None:
            self.ghost_misses += 1
            return None
        ref, base, ordinal = entry
        if ref() is not page:
            self._ghosts.pop(id(page), None)  # id was reused by a new object
            self.ghost_misses += 1
            return None
        rng = self._retained.get(base)
        if rng is None:
            self.ghost_misses += 1
            return None  # still pending (bytes not settled) or already spilled
        self.ghost_hits += 1
        return SlotId(base + ordinal), rng

    # -- canonical-span registry (P3): zero-copy prefix aliasing ------------

    def register_canonical_span(
        self,
        rng: SequenceRange,
        pages: "Sequence[object]",
        ready_event: CachedCudaEvent,
    ) -> None:
        """Pin the resident physical pages holding a closing canonical
        owner's committed prefix (``pages``: the canonical -- by now
        host-tier -- page objects of blocks ``[0, len(pages))``), so future
        same-prefix admissions alias them instead of copying (P3).

        Trims to the chunk-aligned aliasable span; no-op if aliasing is off,
        the span is empty, or this range's head is itself an alias (the
        registry already pins those pages under the original key -- the
        adoption signature keeps serving them). The budget charge for the
        span moves from the range to the registry (release+consume nets zero;
        the range's later reclaim then correctly skips them) and is marked
        retained so the pressure/resume gates count it reclaimable."""
        if not self._prefix_aliasing or not pages:
            return
        if rng._alias_span_key is not None:
            return
        arena = self._arena
        num_blocks = arena.aliasable_prefix_blocks(rng.base_block, len(pages))
        if num_blocks <= 0:
            return
        shared_per_pool = arena.shared_prefix_chunks(rng.base_block, num_blocks)
        span = _CanonicalSpan(
            [rawref.ref(p) for p in pages[:num_blocks]],
            shared_per_pool,
            num_blocks,
            ready_event,
        )
        if span.num_pages == 0:
            return
        key = id(pages[0])
        old = self._canonical_spans.get(key)
        if old is not None:
            self._unpin_span(old)
        for shared in shared_per_pool:
            for s in shared:
                s.add_ref()
        # Transfer the span's budget charge from the owner range to the
        # registry: mark the chunks alias-accounted in the arena (the range's
        # park/adopt/reclaim math then excludes them) and re-charge them here,
        # retained (reclaimable at will via spill_canonical_spans).
        arena._aliased[rng.base_block] = arena._aliased.get(rng.base_block, 0) + span.num_pages
        self._budget.release(span.num_pages)
        self._budget.consume(span.num_pages)
        self._budget.retain(span.num_pages)
        # The owner's parked range aliases the FULL span; typical adopters
        # alias a shorter matched prefix, so this range is usually adopted
        # only by a full-span match (or spilled under pressure).
        rng._alias_span_key = (key, num_blocks)
        self._canonical_spans[key] = span

    def register_live_canonical_span(
        self,
        rng: SequenceRange,
        pages: "Sequence[object]",
        ready_event: CachedCudaEvent,
    ) -> None:
        """Pin a LIVE owner's committed leading prefix at context-end commit
        (P3 v2 owner-live aliasing), so same-prefix admissions alias it while
        the owner is still generating -- close-time registration makes the
        whole first concurrent wave copy instead.

        Differences from :meth:`register_canonical_span`: the pages are the
        owner's active KV, so the span stays OWNER-CHARGED -- no budget
        transfer, no retain, no alias-accounting mark on the range, and the
        spill path skips it (dropping the pin frees nothing). The owner's
        ``free_sequence`` (close, evacuate, and error unwinds all pass
        through it) flips the entry to the registry-charged state via
        :meth:`_commit_live_span_charge`."""
        if not self._prefix_aliasing or not pages:
            return
        if rng._alias_span_key is not None:
            return
        arena = self._arena
        num_blocks = arena.aliasable_prefix_blocks(rng.base_block, len(pages))
        if num_blocks <= 0:
            return
        shared_per_pool = arena.shared_prefix_chunks(rng.base_block, num_blocks)
        span = _CanonicalSpan(
            [rawref.ref(p) for p in pages[:num_blocks]],
            shared_per_pool,
            num_blocks,
            ready_event,
            owner_base=rng.base_block,
        )
        if span.num_pages == 0:
            return
        key = id(pages[0])
        old = self._canonical_spans.get(key)
        if old is not None:
            self._unpin_span(old)
        for shared in shared_per_pool:
            for s in shared:
                s.add_ref()
        rng._alias_span_key = (key, num_blocks)
        self._canonical_spans[key] = span

    def _commit_live_span_charge(self, rng: SequenceRange) -> None:
        """Flip an owner-live span to the registry-charged state as its
        owner's range is freed: transfer the span's budget charge from the
        range to the registry (mark the chunks alias-accounted so the
        range's park/adopt/reclaim math excludes them; release+consume nets
        zero) and mark it retained -- exactly the accounting close-time
        registration produces. No-op for alias CONSUMERS (their span belongs
        to another owner) and for spans already flipped, superseded, or
        invalidated."""
        key_extent = rng._alias_span_key
        assert key_extent is not None
        span = self._canonical_spans.get(key_extent[0])
        if span is None or span.owner_base != rng.base_block:
            return
        arena = self._arena
        arena._aliased[rng.base_block] = arena._aliased.get(rng.base_block, 0) + span.num_pages
        self._budget.release(span.num_pages)
        self._budget.consume(span.num_pages)
        self._budget.retain(span.num_pages)
        span.owner_base = -1

    def lookup_canonical_span(
        self, head_page: object, matched_pages: "Sequence[object]", count_stats: bool = True
    ) -> "tuple[int, _CanonicalSpan, int] | None":
        """Registry hit for an admission whose reuse match starts at
        ``head_page`` and covers ``matched_pages`` (canonical page objects in
        ordinal order): returns ``(key, span, usable_blocks)`` where
        ``usable_blocks`` is the identity-verified matched prefix trimmed to
        whole chunks in every pool -- the safely aliasable extent. A match
        SHORTER than the span aliases only its own covered chunks (the
        adopter writes strictly past them; the shared-prompt benchmark's
        matches are the system prompt, while spans carry the whole committed
        chain of their owner). A mismatch anywhere in the verified prefix
        invalidates and unpins the entry. ``count_stats=False`` keeps the
        hit/miss counters clean for non-admission callers (the dedup-remap
        scan probes every live sequence)."""
        if not self._prefix_aliasing:
            return None
        key = id(head_page)
        span = self._canonical_spans.get(key)
        if span is None:
            if count_stats:
                self.alias_misses += 1
            return None
        verified = min(len(matched_pages), span.num_blocks)
        for i in range(verified):
            if span.page_refs[i]() is not matched_pages[i]:
                self._invalidate_span(key, span)
                if count_stats:
                    self.alias_misses += 1
                return None
        usable = self._arena.aliasable_span_blocks(verified)
        if usable <= 0:
            if count_stats:
                self.alias_misses += 1
            return None
        if count_stats:
            self.alias_hits += 1
        return key, span, usable

    def alias_span_into_range(
        self, rng: SequenceRange, key: int, span: "_CanonicalSpan", num_blocks: int
    ) -> int:
        """Alias-map the first ``num_blocks`` blocks' worth of ``span``'s
        physical pages at the start of ``rng`` (a FRESH, unmapped range) and
        stamp its signature. Signature-matched adopted ranges already carry
        the mappings and skip this. Returns the chunks aliased."""
        assert rng._alias_span_key is None
        trimmed = self._arena.trim_shared_to_blocks(span.shared_per_pool, num_blocks)
        aliased = self._arena.alias_prefix(rng.base_block, trimmed)
        rng._alias_span_key = (key, num_blocks)
        return aliased

    def span_pages_for_blocks(self, span: "_CanonicalSpan", num_blocks: int) -> int:
        """Physical pages a dedup remap of ``num_blocks`` onto ``span`` would
        free (the whole-chunk trim across pools) -- the scan's sort key."""
        trimmed = self._arena.trim_shared_to_blocks(span.shared_per_pool, num_blocks)
        pages = 0
        for shared in trimmed:
            pages += len(shared)
        return pages

    def remap_range_onto_span(
        self, rng: SequenceRange, key: int, span: "_CanonicalSpan", num_blocks: int
    ) -> int:
        """Pressure-time dedup remap (P3 v2): replace the physical backing
        of a LIVE range's head chunks -- currently sequence-private duplicate
        copies of ``span``'s canonical prefix -- with aliases of the span's
        handles, freeing the duplicates' pages. Same-VA swap: slot ids,
        block offsets, page holders, and frontiers are all untouched; the
        range gains the alias signature so it parks prefix-affinely at
        close. The CALLER quiesces once per remap batch. Returns pages
        freed."""
        assert not rng._freed and rng._alias_span_key is None
        trimmed = self._arena.trim_shared_to_blocks(span.shared_per_pool, num_blocks)
        freed = self._arena.remap_prefix_to_alias(rng.base_block, trimmed)
        if freed:
            rng._alias_span_key = (key, num_blocks)
            self.dedup_remapped_pages += freed
        return freed

    def _unpin_span(self, span: "_CanonicalSpan") -> None:
        for shared in span.shared_per_pool:
            for s in shared:
                s.drop_ref()
        if span.owner_base < 0:
            # Owner-live spans hold no registry charge (the owner's range
            # charge covers the pages); only flipped spans release budget.
            self._budget.unretain(span.num_pages)
            self._budget.release(span.num_pages)

    def _invalidate_span(self, key: int, span: "_CanonicalSpan") -> None:
        self._canonical_spans.pop(key, None)
        self._unpin_span(span)

    def spill_canonical_spans(self, min_pages: int) -> int:
        """Drop canonical-span pins (insertion order ≈ LRU) until at least
        ``min_pages`` budget pages are freed. Live aliases keep their handles
        (refcounts); future admissions fall back to host-copy reuse.
        OWNER-LIVE spans are skipped: they hold no registry charge, so
        dropping the pin would forfeit future aliasing and free nothing."""
        freed = 0
        for key in list(self._canonical_spans):
            if freed >= min_pages:
                break
            span = self._canonical_spans[key]
            if span.owner_base >= 0:
                continue
            del self._canonical_spans[key]
            self._unpin_span(span)
            self.spilled_spans += 1
            freed += span.num_pages
        return freed

    @property
    def canonical_span_pages(self) -> int:
        """Registry-CHARGED span pages (the spillable/protected population).
        Owner-live spans are excluded: their pages are charged to their
        owner's range, not the registry."""
        return sum(s.num_pages for s in self._canonical_spans.values() if s.owner_base < 0)

    def _find_range(self, slot_id: int) -> SequenceRange:
        idx = bisect_right(self._range_bases, slot_id) - 1
        assert idx >= 0, f"slot {slot_id} does not belong to any live range"
        rng = self._ranges[self._range_bases[idx]]
        assert rng.base_block <= slot_id < rng.end_block, (
            f"slot {slot_id} does not belong to any live range"
        )
        return rng

    # -- PoolGroupBase interface overrides ----------------------------------

    @override
    def allocate(self) -> Slot:
        raise LogicError(
            "scattered slot allocation is retired in arena mode; use reserve_sequence()/take_slot()"
        )

    @override
    def allocate_multiple(self, num_slots: int) -> list[Slot]:
        raise LogicError(
            "scattered slot allocation is retired in arena mode; use reserve_sequence()/take_slot()"
        )

    @override
    def release(self, slot: Slot) -> None:
        """Return one issued slot. The block index stays reserved for the
        owning sequence; the slot's ready event (if still pending) gates the
        range's eventual reclaim."""
        slot = slot.move_to_new_slot()
        rng = self._find_range(slot.slot_id)
        assert rng._outstanding > 0
        rng._outstanding -= 1
        ev = slot.ready_event
        if ev is not CachedCudaEvent.NULL and not ev.query_complete():
            rng._gate_events.append(ev)
        # Detach so Slot.__del__ does not warn; the range owns the index.
        slot._slot_id = None
        slot.ready_event = CachedCudaEvent.NULL

    @property
    @override
    def num_slots(self) -> int:
        return self._arena.capacity_blocks

    @property
    @override
    def num_free_slots(self) -> int:
        """Free *VA* blocks. Physical availability is governed by the shared
        :class:`PageBudget`, not per-group slot counts (§4.6)."""
        return self._arena.free_blocks

    @property
    def mapped_pages(self) -> int:
        return self._arena.mapped_pages

    def _check(self, allow_mismatch: bool = False) -> bool:
        return True  # the base invariant (SlotAllocator vs pools) does not apply


class HostPoolGroup(PoolGroupBase):
    __slots__ = ()

    def __init__(self, num_slots: int, slot_size_list: TypedIndexList[PoolIndex, int]):
        super().__init__(num_slots)
        self._pools = typed_map(
            slot_size_list, lambda slot_size: HostSlotPool(slot_size, num_slots)
        )


class DiskPoolGroup(PoolGroupBase):
    __slots__ = ()

    def __init__(
        self, num_slots: int, slot_size_list: TypedIndexList[PoolIndex, int], filename_template: str
    ):
        super().__init__(num_slots)
        num_pools = typed_len(slot_size_list)
        self._pools = make_typed(
            lambda pool_idx: DiskSlotPool(
                filename_template.format(pool_idx), slot_size_list[pool_idx], num_slots
            ),
            num_pools,
        )


class CacheLevelStorage:
    TIER: ClassVar[CacheTier]
    __slots__ = "_pool_groups"
    # _total_quota: int  # fixme: remove _total_quota and _ratio_list and compute from _pool_groups
    # _ratio_list: TypedIndexList[PoolGroupIndex, float]
    _pool_groups: TypedIndexList[PoolGroupIndex, PoolGroupBase]

    def __init__(self) -> None:
        if not hasattr(self.__class__, "TIER"):
            raise ValueError(f"{self.__class__.__name__} must define 'TIER' as a class variable")

    def __del__(self) -> None:
        self.destroy()

    @property
    def cache_tier(self) -> CacheTier:
        return self.TIER

    def destroy(self) -> None:
        for pg in self._pool_groups:
            pg.destroy()

    def allocate(self, pool_group_index: PoolGroupIndex) -> Slot:
        return self._pool_groups[pool_group_index].allocate()

    def allocate_multiple(self, pool_group_index: PoolGroupIndex, num_slots: int) -> list[Slot]:
        return self._pool_groups[pool_group_index].allocate_multiple(num_slots)

    def release(self, pool_group_index: PoolGroupIndex, slot: Slot) -> None:
        self._pool_groups[pool_group_index].release(slot)

    @property
    def total_quota(self) -> int:
        granularity = self.pool_size_granularity
        quota = 0
        for pg in self._pool_groups:
            for p in pg._pools:
                quota += round_up(p.num_bytes, granularity)
        return quota

    @property
    def ratio_list(self) -> TypedIndexList[PoolGroupIndex, float]:
        num_pool_groups = self.num_pool_groups
        ret = filled_list(0.0, num_pool_groups)
        total = 0
        for i, pg in typed_enumerate(self._pool_groups):
            size = pg.num_bytes
            total += size
            ret[i] = size
        assert total > 0
        for i in typed_range(num_pool_groups):
            ret[i] /= total
        return ret

    def num_slots(self, pool_group_index: PoolGroupIndex) -> int:
        return self._pool_groups[pool_group_index].num_slots

    def get_num_free_slots(self, pool_group_index: PoolGroupIndex) -> int:
        return self._pool_groups[pool_group_index].num_free_slots

    @property
    def slot_count_list(self) -> TypedIndexList[PoolGroupIndex, int]:
        """
        The number of slots in each pool group.
        """
        return typed_map(self._pool_groups, lambda pg: pg.num_slots)

    def slot_size(self, pool_group_index: PoolGroupIndex) -> TypedIndexList[PoolIndex, int]:
        """
        The slot sizes of each pool in the pool group.
        """
        return self._pool_groups[pool_group_index].slot_size

    @property
    def slot_size_lists(self) -> TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]]:
        """
        A tuple of tuples, each containing the slot sizes for a pool group.
        """
        return typed_map(self._pool_groups, lambda pg: typed_map(pg._pools, lambda p: p.slot_size))

    @property
    def num_pool_groups(self) -> PoolGroupIndex:
        return typed_len(self._pool_groups)

    def slot_address(
        self, pool_group_index: PoolGroupIndex, pool_index: PoolIndex, slot_id: SlotId
    ) -> Address:
        return self._pool(pool_group_index, pool_index).slot_address(slot_id)

    def post_resize(self) -> None:
        pass

    def _pool(self, pool_group_index: PoolGroupIndex, pool_index: PoolIndex) -> SlotPoolBase:
        return self._pool_groups[pool_group_index]._pools[pool_index]

    # Calculate how many slots will there be in each pool group with the given total_quota and
    # ratio_list. Use ratio_to_slot_count_list for initialization.
    def compute_slot_count_list(
        self,
        ratio_list: TypedIndexList[PoolGroupIndex, float],
        min_slots: TypedIndexList[PoolGroupIndex, int],
        total_quota: int | None = None,
    ) -> TypedIndexList[PoolGroupIndex, int]:
        if total_quota is None:
            total_quota = self.total_quota
        assert len(ratio_list) == len(self._pool_groups), (
            f"Wrong ratio_list length. Expected {len(self._pool_groups)}, got {len(ratio_list)}"
        )
        return self.ratio_to_slot_count_list(
            total_quota, self.slot_size_lists, ratio_list, self.pool_size_granularity, min_slots
        )

    @staticmethod
    def _grains_to_slots(
        pg_grains: int,
        slot_size_list: TypedIndexList[PoolIndex, int],
        granularity: int,
    ) -> tuple[int, int]:
        """Compute the maximum slots that fit in a pool group grain budget.

        Returns (num_slots, grains_consumed).
        """
        num_pools = typed_len(slot_size_list)
        min_pool_grains = typed_map(slot_size_list, lambda s: div_up(s, granularity))
        if pg_grains < sum(min_pool_grains):
            return (0, 0)
        num_slots: int = 1 << 63
        remaining_pg_grains = pg_grains
        pool_idx_lst = sorted(typed_range(num_pools), key=lambda i: slot_size_list[i])
        for j, pool in enumerate(pool_idx_lst):
            slot_size = slot_size_list[pool]
            pool_grains = max(
                min_pool_grains[pool],
                round(
                    remaining_pg_grains
                    * (slot_size / sum(slot_size_list[k] for k in pool_idx_lst[j:]))
                ),
            )
            num_slots = min(num_slots, pool_grains * granularity // slot_size)
            remaining_pg_grains -= pool_grains
        assert remaining_pg_grains == 0
        assert num_slots > 0
        _s2g = CacheLevelStorage._grains_for_slots
        lo = num_slots
        step = 1
        hi = lo + step
        while _s2g(hi, slot_size_list, granularity) <= pg_grains:
            lo = hi
            step *= 2
            hi = lo + step
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _s2g(mid, slot_size_list, granularity) <= pg_grains:
                lo = mid
            else:
                hi = mid
        used = _s2g(lo, slot_size_list, granularity)
        assert used <= pg_grains
        assert _s2g(lo + 1, slot_size_list, granularity) > pg_grains
        return lo, used

    @staticmethod
    def _grains_for_slots(
        num_slots: int,
        slot_size_list: TypedIndexList[PoolIndex, int],
        granularity: int,
    ) -> int:
        """Compute the minimum grains needed for num_slots in a pool group."""
        return sum(div_up(num_slots * s, granularity) for s in slot_size_list)

    @staticmethod
    def ratio_to_slot_count_list(
        total_quota: int,
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        ratio_list: TypedIndexList[PoolGroupIndex, float],
        pool_size_granularity: int,
        min_slots: TypedIndexList[PoolGroupIndex, int],
    ) -> TypedIndexList[PoolGroupIndex, int]:
        num_pool_groups = typed_len(ratio_list)
        assert all(x > 0 for x in ratio_list)
        assert num_pool_groups == typed_len(slot_size_lists)
        assert total_quota % pool_size_granularity == 0
        total_grains = total_quota // pool_size_granularity
        assert total_grains >= sum(len(sizes) for sizes in slot_size_lists)
        g = pool_size_granularity
        _g2s = CacheLevelStorage._grains_to_slots
        _s2g = CacheLevelStorage._grains_for_slots

        slot_cnt_list = filled_list(0, num_pool_groups)
        remaining_grains = total_grains
        active_pgs = list(typed_range(num_pool_groups))

        # Iteratively peel off constrained PGs until all active PGs are
        # unconstrained:
        #   1. Distribute remaining quota among active PGs by ratio.
        #   2. Any PG with slots <= min_slots is constrained — pin it to
        #      min_slots and subtract its grains from the budget.
        #   3. Repeat with the remaining PGs and re-normalized ratios.
        # Each iteration removes at least one PG, so this terminates.
        while active_pgs:
            # Distribute remaining_grains among active PGs by ratio.
            active_ratio = [ratio_list[pg] for pg in active_pgs]
            slots_for_active = filled_list(0, len(active_pgs))
            grains_for_active = filled_list(0, len(active_pgs))
            budget = remaining_grains
            idx_lst = sorted(range(len(active_pgs)), key=lambda i: active_ratio[i])
            for i, idx in enumerate(idx_lst):
                pct = active_ratio[idx] / sum(active_ratio[j] for j in idx_lst[i:])
                slots, used = _g2s(round(budget * pct), slot_size_lists[active_pgs[idx]], g)
                slots_for_active[idx] = slots
                grains_for_active[idx] = used
                budget -= used
            assert budget >= 0

            # Identify constrained PGs (slots <= min_slots).
            constrained = []
            unconstrained = []
            for idx in range(len(active_pgs)):
                pg = active_pgs[idx]
                if slots_for_active[idx] <= min_slots[pg]:
                    constrained.append(idx)
                else:
                    unconstrained.append(idx)

            if not constrained:
                # All active PGs are unconstrained — accept their allocations.
                for idx in range(len(active_pgs)):
                    slot_cnt_list[active_pgs[idx]] = slots_for_active[idx]
                break

            # Pin constrained PGs to min_slots and subtract from budget.
            for idx in constrained:
                pg = active_pgs[idx]
                min_grains = _s2g(min_slots[pg], slot_size_lists[pg], g)
                slots, used = _g2s(min_grains, slot_size_lists[pg], g)
                slot_cnt_list[pg] = slots
                remaining_grains -= used

            if not unconstrained:
                # All PGs are constrained — nothing left to redistribute.
                break

            if remaining_grains <= 0:
                raise ValueError("Insufficient quota to satisfy min_slots constraints")

            # Continue with unconstrained PGs only.
            active_pgs = [active_pgs[idx] for idx in unconstrained]

        # _g2s may under-count slots due to imperfect grain distribution
        # across pools. Try bumping each PG's slot count while it still fits
        # within the same grain budget.
        for pg in typed_range(num_pool_groups):
            grains_now = _s2g(slot_cnt_list[pg], slot_size_lists[pg], g)
            while _s2g(slot_cnt_list[pg] + 1, slot_size_lists[pg], g) <= grains_now:
                slot_cnt_list[pg] += 1

        return slot_cnt_list

    @property
    def pool_size_granularity(self) -> int:
        return 2 << 20


class GpuCacheLevelStorage(CacheLevelStorage):
    TIER: ClassVar[CacheTier] = CacheTier.GPU_MEM
    __slots__ = ("shared_phys_mem_pool",)
    shared_phys_mem_pool: PooledPhysMemAllocator

    def __init__(
        self,
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        slot_count_list: TypedIndexList[PoolGroupIndex, int],
        phys_mem_size: int,
    ):
        num_pool_groups = typed_len(slot_size_lists)
        assert num_pool_groups == typed_len(slot_count_list), (
            "slot_size_lists and slot_count_list must have the same length"
        )
        super().__init__()
        self.shared_phys_mem_pool = PooledPhysMemAllocator(phys_mem_size)
        self._pool_groups = make_typed(
            lambda pg_idx: GpuPoolGroup(
                slot_count_list[pg_idx], slot_size_lists[pg_idx], self.shared_phys_mem_pool
            ),
            num_pool_groups,
        )

    @override
    def post_resize(self) -> None:
        super().post_resize()
        self.shared_phys_mem_pool.clear()  # clear cached unused phys mem

    @property
    def pool_size_granularity(self) -> int:
        return self.shared_phys_mem_pool.phys_mem_size

    @override
    def destroy(self) -> None:
        super().destroy()
        self.shared_phys_mem_pool.clear()


class GpuArenaCacheLevelStorage(CacheLevelStorage):
    """GPU cache level backed by per-sequence contiguous arenas
    (contiguous primary KV cache, DESIGN.md §4). Flag-gated alternative to
    :class:`GpuCacheLevelStorage`; selected by
    ``KVCacheManagerConfig.contiguous_arena``.

    Capacity is a level-wide physical :class:`PageBudget` (``quota //
    phys_page_size`` pages) that every pool group's arena maps against --
    per-pool-group slot partitioning and copy-based GPU defrag are retired
    (§4.2/§4.6). ``block_capacity_list`` sizes each pool group's *VA* extent in
    blocks (see §4.8 sizing); ``page_index_scale_list`` carries each group's
    maximum kernel page-index scale for the int31 startup check (§4.1).
    """

    TIER: ClassVar[CacheTier] = CacheTier.GPU_MEM
    __slots__ = ("shared_phys_mem_pool", "page_budget", "_span_spill_floor")
    shared_phys_mem_pool: PooledPhysMemAllocator
    page_budget: PageBudget
    # Canonical-span pages exempt from pressure spills (P3 span-spill
    # protection): ``protected_span_fraction`` of the page budget.
    _span_spill_floor: int

    def __init__(
        self,
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        block_capacity_list: TypedIndexList[PoolGroupIndex, int],
        page_index_scale_list: TypedIndexList[PoolGroupIndex, int],
        quota: int,
        phys_page_size: int,
        map_ahead_pages: int,
        lazy_retention: bool = False,
        prefix_aliasing: bool = False,
        protected_span_fraction: float = 0.05,
    ):
        num_pool_groups = typed_len(slot_size_lists)
        assert num_pool_groups == typed_len(block_capacity_list), (
            "slot_size_lists and block_capacity_list must have the same length"
        )
        assert num_pool_groups == typed_len(page_index_scale_list), (
            "slot_size_lists and page_index_scale_list must have the same length"
        )
        assert quota >= phys_page_size, "GPU quota must cover at least one physical page"
        # Run the int31 startup check (§4.1) for every group before touching
        # CUDA, so a mis-sized arena fails cleanly with nothing constructed.
        for pg_idx in typed_range(num_pool_groups):
            check_index_width(block_capacity_list[pg_idx], page_index_scale_list[pg_idx])
        super().__init__()
        self.shared_phys_mem_pool = PooledPhysMemAllocator(phys_page_size)
        self.page_budget = PageBudget(quota // phys_page_size)
        if _SPAN_PROTECT_FRACTION >= 0.0:
            protected_span_fraction = _SPAN_PROTECT_FRACTION
        assert 0.0 <= protected_span_fraction <= 1.0
        self._span_spill_floor = int(self.page_budget.total_pages * protected_span_fraction)
        self._pool_groups = make_typed(
            lambda pg_idx: ArenaPoolGroup(
                block_capacity_list[pg_idx],
                slot_size_lists[pg_idx],
                self.shared_phys_mem_pool,
                self.page_budget,
                map_ahead_pages,
                page_index_scale_list[pg_idx],
                lazy_retention,
                prefix_aliasing,
            ),
            num_pool_groups,
        )

    def pool_group(self, pool_group_index: PoolGroupIndex) -> ArenaPoolGroup:
        pg = self._pool_groups[pool_group_index]
        assert type(pg) is ArenaPoolGroup
        return pg

    def drain_reclaim(
        self, fence: CachedCudaEvent | None = None, wait: bool = False, quiesce: bool = True
    ) -> int:
        """Drain every pool group's deferred-reclaim queue (call at iteration
        boundaries). ``fence`` gates newly freed ranges against speculatively
        enqueued steps; ``wait=True`` blocks on in-flight gating events
        instead of skipping their ranges (see
        :meth:`ArenaPoolGroup.drain_reclaim`); ``quiesce=False`` skips the
        per-range device quiesce (:meth:`hard_reclaim` quiesced once for the
        batch). Returns the number of sequence ranges reclaimed."""
        reclaimed = 0
        for pg in self._pool_groups:
            assert type(pg) is ArenaPoolGroup
            reclaimed += pg.drain_reclaim(fence, wait, quiesce)
        return reclaimed

    def fence_pending_frees(self, fence: CachedCudaEvent) -> None:
        """Assign the iteration-boundary reclaim fence in every pool group
        without draining (see :meth:`ArenaPoolGroup.fence_pending_frees`)."""
        for pg in self._pool_groups:
            assert type(pg) is ArenaPoolGroup
            pg.fence_pending_frees(fence)

    def spill_retained(self, min_pages: int, spill_spans: bool = True, quiesce: bool = True) -> int:
        """Reclaim lazily retained ranges (§4.4 phase 2), LRU-first per pool
        group, until ``min_pages`` physical pages are freed or nothing more
        is spillable; canonical-span pins (P3) are spilled last -- shared
        prefixes are the hottest bytes on the level -- and only above the
        protected floor (``protected_span_fraction``): spilling-last alone
        still forfeits the whole registry under sustained pressure, and with
        it every zero-copy alias hit, to cover a transient need worth a
        handful of pages. Protected callers back off like any failed
        allocation instead (§4.6). ``spill_spans=False`` skips the span pass
        entirely; ``quiesce=False`` skips the per-range device quiesce
        (:meth:`hard_reclaim` quiesced once for the batch). Returns pages
        freed."""
        freed = 0
        for pg in self._pool_groups:
            if freed >= min_pages:
                break
            assert type(pg) is ArenaPoolGroup
            freed += pg.spill_retained(min_pages - freed, quiesce)
        if not spill_spans:
            return freed
        spillable = -self._span_spill_floor
        for pg in self._pool_groups:
            assert type(pg) is ArenaPoolGroup
            spillable += pg.canonical_span_pages
        for pg in self._pool_groups:
            if freed >= min_pages or spillable <= 0:
                break
            assert type(pg) is ArenaPoolGroup
            got = pg.spill_canonical_spans(min(min_pages - freed, spillable))
            freed += got
            spillable -= got
        return freed

    def dedup_remap(
        self, items: "list[tuple[int, SequenceRange, int, object, int]]", quiesce: bool = True
    ) -> int:
        """Execute a dedup-remap batch (P3 v2) behind ONE device quiesce:
        each item ``(pg_idx, rng, key, span, num_blocks)`` replaces the
        range's private duplicate prefix backing with aliases of the span's
        handles. The caller (the manager's pressure-time scan) has already
        verified eligibility host-side; nothing here enqueues GPU work, so
        the batch is atomic with respect to the executor's enqueue thread.
        ``quiesce=False`` skips the quiesce (:meth:`hard_reclaim` already
        quiesced). Returns total pages freed."""
        if not items:
            return 0
        if quiesce:
            quiesce_before_unmap()
        freed = 0
        for pg_idx, rng, key, span, num_blocks in items:
            pg = self._pool_groups[PoolGroupIndex(pg_idx)]
            assert type(pg) is ArenaPoolGroup
            freed += pg.remap_range_onto_span(rng, key, cast("_CanonicalSpan", span), num_blocks)
        return freed

    def hard_reclaim(
        self,
        min_pages: int,
        dedup_items: "list[tuple[int, SequenceRange, int, object, int]]",
        fence: CachedCudaEvent | None = None,
    ) -> int:
        """Pressure-time hard reclaim (P3 v3): every rung that needs the
        device quiesced, behind ONE quiesce, harvesting maximally
        (destruction-ordered -- see the manager's ``hard_reclaim_gpu``).

        Sync-free exit first: if there is nothing to reclaim anywhere (no
        dedup candidates, nothing pending, nothing parked, no registry
        excess above the protection floor), returns 0 WITHOUT syncing.
        Otherwise quiesces once, then:

        1. fenced drain -- every pending free's gates are complete
           post-sync, so all of them park (adoption/retention) or unmap
           (plain mode) now;
        2. dedup remap of ALL ``dedup_items`` (duplicates have zero
           retention value: this frees pages from live sequences without
           touching the parked cache);
        3. parked-range spill for the REMAINDER still needed, then registry
           excess above the floor last (both destroy reuse value).

        ``fence``: an execution-stream event recorded by the caller BEFORE
        this call (it completes under the quiesce); attached to pending
        frees that have none yet so the drain can take them. Returns pages
        freed (budget delta)."""
        budget = self.page_budget
        free_before = budget.free_pages
        has_work = bool(dedup_items)
        if not has_work:
            for pg in self._pool_groups:
                assert type(pg) is ArenaPoolGroup
                # Pending ranges with outstanding slot refs can never drain
                # (their release is host-side state, not an event) -- do not
                # pay a quiesce for them.
                if pg._retained or any(r._outstanding == 0 for r in pg._pending_reclaim):
                    has_work = True
                    break
        if not has_work:
            spillable = -self._span_spill_floor
            for pg in self._pool_groups:
                assert type(pg) is ArenaPoolGroup
                spillable += pg.canonical_span_pages
            has_work = spillable > 0
        if not has_work:
            return 0
        quiesce_before_unmap()
        if fence is not None:
            self.fence_pending_frees(fence)
        self.drain_reclaim(quiesce=False)
        if dedup_items:
            self.dedup_remap(dedup_items, quiesce=False)
        needed = min_pages - (budget.free_pages - free_before)
        if needed > 0:
            self.spill_retained(needed, spill_spans=True, quiesce=False)
        return budget.free_pages - free_before

    @property
    def protected_span_pages(self) -> int:
        """Canonical-span pages currently under the spill-protection floor.
        These are pinned for the registry's benefit and NOT reclaimable
        under pressure, so utilization gates must not count them available
        the way plain retained pages are."""
        if self._span_spill_floor == 0:
            return 0
        span_pages = 0
        for pg in self._pool_groups:
            assert type(pg) is ArenaPoolGroup
            span_pages += pg.canonical_span_pages
        return min(span_pages, self._span_spill_floor)

    def flush_mappings(self) -> int:
        """Execute every pool group's deferred growth maps back-to-back
        (§4.2 batched per-iteration sweep). Returns pages mapped."""
        mapped = 0
        for pg in self._pool_groups:
            assert type(pg) is ArenaPoolGroup
            mapped += pg.flush_mappings()
        return mapped

    @property
    def pool_size_granularity(self) -> int:
        return self.shared_phys_mem_pool.phys_mem_size

    @property
    def total_quota(self) -> int:
        """The level's PHYSICAL byte quota (page budget x super-page size).

        The base implementation sums pool byte sizes, which for arena pools
        is the VA reservation extent -- virtual address space, not memory.
        Consumers of ``total_quota`` (kv stats ``allocated_bytes`` and,
        through it, the KV memory estimator's temporary-pool credit) need
        physical bytes: crediting VA back over-grants the final KV budget by
        the VA-vs-physical difference (observed: +8 GiB on an 80 GiB H100,
        leaving <0.1 GiB of device headroom at full budget fill).
        """
        return self.page_budget.total_pages * self.shared_phys_mem_pool.phys_mem_size

    @override
    def destroy(self) -> None:
        # __init__ raises by design on an int31 violation (§4.1); a partially
        # constructed level has nothing to tear down.
        if not hasattr(self, "_pool_groups"):
            return
        super().destroy()
        self.shared_phys_mem_pool.clear()


class HostCacheLevelStorage(CacheLevelStorage):
    TIER: ClassVar[CacheTier] = CacheTier.HOST_MEM
    POOL_SIZE_GRANULARITY: ClassVar[int] = HostMem.ALIGNMENT
    __slots__ = ()

    def __init__(
        self,
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        slot_count_list: TypedIndexList[PoolGroupIndex, int],
    ):
        num_pool_groups = typed_len(slot_size_lists)
        assert num_pool_groups == typed_len(slot_count_list), (
            "slot_size_lists and slot_count_list must have the same length"
        )
        super().__init__()
        self._pool_groups = make_typed(
            lambda pg_idx: HostPoolGroup(slot_count_list[pg_idx], slot_size_lists[pg_idx]),
            num_pool_groups,
        )

    @property
    def pool_size_granularity(self) -> int:
        return self.POOL_SIZE_GRANULARITY


class DiskCacheLevelStorage(CacheLevelStorage):
    __slots__ = ()
    TIER: ClassVar[CacheTier] = CacheTier.DISK
    POOL_SIZE_GRANULARITY: ClassVar[int] = 2 << 20

    def __init__(
        self,
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        slot_count_list: TypedIndexList[PoolGroupIndex, int],
        filename_template: str,
    ):
        num_pool_groups = typed_len(slot_size_lists)
        assert num_pool_groups == typed_len(slot_count_list), (
            "slot_size_lists and slot_count_list must have the same length"
        )
        super().__init__()
        self._pool_groups = make_typed(
            lambda pg_idx: DiskPoolGroup(
                slot_count_list[pg_idx],
                slot_size_lists[pg_idx],
                filename_template.format(pg_idx, "{}"),
            ),
            num_pool_groups,
        )

    @property
    def pool_size_granularity(self) -> int:
        return self.POOL_SIZE_GRANULARITY
