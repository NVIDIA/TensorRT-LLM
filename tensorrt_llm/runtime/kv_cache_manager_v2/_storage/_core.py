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
from typing import ClassVar, NewType, final

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from .._common import (
    BAD_FILE_DESCRIPTOR,
    NDEBUG,
    Address,
    CacheTier,
    DiskAddress,
    FileDescriptor,
    MemAddress,
)
from .._cuda_virt_mem import PooledPhysMemAllocator, VirtMem
from .._exceptions import LogicError, OutOfPagesError
from .._sequence_arena import PageBudget, SequenceArena, check_index_width
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
    )
    base_block: int
    num_blocks: int  # reserved (padded) length in blocks
    _outstanding: int  # slots issued and not yet released
    _freed: bool
    _gate_events: list[CachedCudaEvent]  # pending ready events of released slots
    _free_event: CachedCudaEvent

    def __init__(self, base_block: int, num_blocks: int) -> None:
        self.base_block = base_block
        self.num_blocks = num_blocks
        self._outstanding = 0
        self._freed = False
        self._gate_events = []
        self._free_event = CachedCudaEvent.NULL

    @property
    def end_block(self) -> int:
        return self.base_block + self.num_blocks

    def _scrub_gate_events(self) -> None:
        self._gate_events = [ev for ev in self._gate_events if not ev.query_complete()]

    def _ready_to_reclaim(self) -> bool:
        if not self._freed or self._outstanding != 0:
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

    __slots__ = ("_arena", "_ranges", "_range_bases", "_pending_reclaim")
    _arena: SequenceArena
    _ranges: dict[int, SequenceRange]  # base_block -> live range
    _range_bases: list[int]  # sorted, for slot_id -> range lookup on release
    _pending_reclaim: list[SequenceRange]

    def __init__(
        self,
        block_capacity: int,
        slot_size_list: TypedIndexList[PoolIndex, int],
        shared_phys_mem_pool: PooledPhysMemAllocator,
        page_budget: PageBudget,
        map_ahead_pages: int,
        page_index_scale: int,
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

    @override
    def destroy(self) -> None:
        # The int31 check makes __init__ raise before base construction; a
        # partially constructed group has nothing to tear down.
        if not hasattr(self, "_destroyed") or self._destroyed:
            return
        assert not self._ranges and not self._pending_reclaim, (
            "destroying ArenaPoolGroup with live sequence ranges"
        )
        super().destroy()
        self._arena.destroy()

    # -- per-sequence API (replaces scattered slot allocation) --------------

    def reserve_sequence(self, max_blocks: int) -> SequenceRange:
        """Reserve a contiguous block-index range for a sequence's maximum
        block count. Pure VA bookkeeping; no physical memory is mapped.
        Raises ``MemoryError`` on VA exhaustion (see DESIGN.md §4.8 sizing)."""
        base_block = self._arena.reserve(max_blocks)
        rng = SequenceRange(base_block, self._arena.reserved_len(base_block))
        self._ranges[base_block] = rng
        insort(self._range_bases, base_block)
        return rng

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

    def free_sequence(self, rng: SequenceRange, last_consumer: CachedCudaEvent) -> None:
        """Queue the whole range for deferred reclaim (§4.2). The range is
        unmapped and its block indices reused only once ``last_consumer``, the
        ready events of all released slots, and the release of every
        outstanding slot have all happened -- see :meth:`drain_reclaim`."""
        assert not rng._freed, "double free of a sequence range"
        rng._freed = True
        rng._free_event = last_consumer
        self._pending_reclaim.append(rng)

    def drain_reclaim(self) -> int:
        """Unmap and recycle every freed range whose gating conditions have
        all completed. Returns the number of ranges reclaimed. Call at
        iteration boundaries."""
        remaining: list[SequenceRange] = []
        reclaimed = 0
        for rng in self._pending_reclaim:
            if rng._ready_to_reclaim():
                self._arena.reclaim(rng.base_block)
                del self._ranges[rng.base_block]
                idx = bisect_right(self._range_bases, rng.base_block) - 1
                assert self._range_bases[idx] == rng.base_block
                self._range_bases.pop(idx)
                reclaimed += 1
            else:
                remaining.append(rng)
        self._pending_reclaim = remaining
        return reclaimed

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
    __slots__ = ("shared_phys_mem_pool", "page_budget")
    shared_phys_mem_pool: PooledPhysMemAllocator
    page_budget: PageBudget

    def __init__(
        self,
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        block_capacity_list: TypedIndexList[PoolGroupIndex, int],
        page_index_scale_list: TypedIndexList[PoolGroupIndex, int],
        quota: int,
        phys_page_size: int,
        map_ahead_pages: int,
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
        self._pool_groups = make_typed(
            lambda pg_idx: ArenaPoolGroup(
                block_capacity_list[pg_idx],
                slot_size_lists[pg_idx],
                self.shared_phys_mem_pool,
                self.page_budget,
                map_ahead_pages,
                page_index_scale_list[pg_idx],
            ),
            num_pool_groups,
        )

    def pool_group(self, pool_group_index: PoolGroupIndex) -> ArenaPoolGroup:
        pg = self._pool_groups[pool_group_index]
        assert type(pg) is ArenaPoolGroup
        return pg

    def drain_reclaim(self) -> int:
        """Drain every pool group's deferred-reclaim queue (call at iteration
        boundaries). Returns the number of sequence ranges reclaimed."""
        reclaimed = 0
        for pg in self._pool_groups:
            assert type(pg) is ArenaPoolGroup
            reclaimed += pg.drain_reclaim()
        return reclaimed

    @property
    def pool_size_granularity(self) -> int:
        return self.shared_phys_mem_pool.phys_mem_size

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
