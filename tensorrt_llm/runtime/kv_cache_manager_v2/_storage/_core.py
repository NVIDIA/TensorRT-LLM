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
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, NewType

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
from .._cuda_virt_mem import LOCALIZATION_OFFSET, PooledPhysMemAllocator, VirtMem
from .._exceptions import LogicError, OutOfPagesError
from .._utils import (
    CachedCudaEvent,
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
    def slot_address(self, slot: SlotId) -> Address:
        pass

    def __del__(self) -> None:
        self.destroy()


class GpuSlotPool:
    """GPU slot pool supporting 1 or N locality domains within a single VirtMem reservation.

    For a single locality domain (the default), construction accepts scalar ``vm_size``
    and ``num_slots`` arguments — identical to the original non-localized API.
    For multiple locality domains, pass lists of per-locality domain values instead.

    All operations that touch physical memory or VA addressing accept an
    optional ``locality_domain_id`` (default 0) so that single-locality domain call sites remain
    unchanged.

    ``slot_address`` expects a local 0-based slot index for the selected
    locality domain and converts it into a byte address::

        address = locality_domain_address(locality_domain_id) + slot_size * slot

    For ``locality_domain_id == 0`` this reduces to ``base + slot_size * slot``.
    """

    __slots__ = ("_slot_size", "_vm")
    _slot_size: int
    _vm: VirtMem

    def __init__(
        self,
        slot_size: int,
        vm_sizes: int | list[int],
        shared_phys_mem_pool: PooledPhysMemAllocator,
        num_slots: int | list[int],
    ) -> None:
        num_locality_domains = shared_phys_mem_pool.num_locality_domains
        # Normalise scalar args to per-locality domain lists.
        if isinstance(vm_sizes, int):
            vm_sizes = [vm_sizes]
        if isinstance(num_slots, int):
            num_slots = [num_slots] * num_locality_domains
        assert len(vm_sizes) == num_locality_domains and len(num_slots) == num_locality_domains, (
            f"Expected {num_locality_domains} elements (num_locality_domains={num_locality_domains}), "
            f"got vm_sizes={len(vm_sizes)}, num_slots={len(num_slots)}"
        )

        phys_mem_size = shared_phys_mem_pool.phys_mem_size
        for vs in vm_sizes:
            assert vs % phys_mem_size == 0

        self._slot_size = slot_size
        self._vm = VirtMem(vm_sizes, shared_phys_mem_pool)
        for uid, ns in enumerate(num_slots):
            self.resize(ns, locality_domain_id=uid)

    @property
    def slot_size(self) -> int:
        return self._slot_size

    @property
    def phys_mem_size(self) -> int:
        return self._vm.phys_mem_size

    @property
    def num_locality_domains(self) -> int:
        return self._vm.num_locality_domains

    def destroy(self) -> None:
        self._vm.destroy()

    def __del__(self) -> None:
        self.destroy()

    def resize(self, new_num_slots: int, locality_domain_id: int = 0) -> None:
        new_num_phys_mem = self._compute_num_phys_mem(
            self.slot_size, new_num_slots, self._vm.phys_mem_size
        )
        self._vm.realloc(self._vm.phys_mem_size * new_num_phys_mem, locality_domain_id)

    def extend_by_one_phys_mem(self, locality_domain_id: int = 0) -> int:
        self._vm.extend(1, locality_domain_id)
        return self.num_slots(locality_domain_id)

    def slot_address(self, slot: SlotId, locality_domain_id: int = 0) -> MemAddress:
        """Return the memory address for local slot index ``slot`` in ``locality_domain_id``'s VA region."""
        return MemAddress(
            self._vm.locality_domain_address(locality_domain_id) + self._slot_size * slot
        )

    def num_slots(self, locality_domain_id: int = 0) -> int:
        return self._compute_num_slots(
            self.slot_size, self._vm.num_phys_mem(locality_domain_id), self._vm.phys_mem_size
        )

    def num_bytes(self, locality_domain_id: int = 0) -> int:
        return self.slot_size * self.num_slots(locality_domain_id)

    @staticmethod
    def _compute_num_phys_mem(slot_size: int, num_slots: int, phys_mem_size: int) -> int:
        return div_up(num_slots * slot_size, phys_mem_size)

    @staticmethod
    def _compute_num_slots(slot_size: int, num_phys_mem: int, phys_mem_size: int) -> int:
        return num_phys_mem * phys_mem_size // slot_size


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
    def slot_address(self, slot: SlotId) -> MemAddress:
        return MemAddress(self._host_mem._address + self.slot_size * int(slot))

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
    def slot_address(self, slot: SlotId) -> DiskAddress:
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
    # locality_domain_id is None for non-localized (regular GPU) slots.
    # For localized locality domain slots it must be 0 or 1, identifying which locality domain pool owns this slot.
    # In localized mode slot_id carries a canonical per-locality domain slot-space stride
    # rather than the raw byte-space LOCALIZATION_OFFSET. The VA address is
    # reconstructed later from (slot_id, locality_domain_id) using the pool's slot size.
    locality_domain_id: int | None

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
        ret = Slot(None, CachedCudaEvent.NULL, None)
        ret.set_slot(self)
        return ret

    def set_slot(self, slot: "Slot") -> None:
        if self.has_valid_slot:
            raise LogicError("Slot is already set.")
        self._slot_id = slot.slot_id
        self.locality_domain_id = slot.locality_domain_id
        self.ready_event = slot.ready_event
        slot._slot_id = None
        slot.locality_domain_id = None
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
        "_occupied_slot_ids",
        "_slot_id_offset",
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
    # Set of slot_ids currently handed out to callers.  A set (vs. DynamicBitset) is used so that
    # localized slot_ids with canonical per-locality domain offsets can be tracked
    # without allocating a multi-terabyte bitset.
    _occupied_slot_ids: set[SlotId]
    # Additive offset applied to every slot_id this allocator creates.
    # In localized mode this is a canonical slot-id stride derived from the
    # pool-group's largest slot size, so slot_ids remain unique across locality domains
    # while LOCALIZATION_OFFSET itself stays byte-based.
    _slot_id_offset: int

    # for scheduled shrinking resize
    _target_capacity: (
        int  # _target_capacity <= _capacity. Inequal if a shrinking resize is in progress.
    )
    _overflow_slots: list[
        Slot
    ]  # slots that will be out-of-range after a in-progress resize. scheduled for removal.

    def __init__(self, capacity: int, slot_id_offset: int = 0) -> None:
        self._capacity = capacity
        self._num_active_slots = 0
        self._recycled_slots = deque[Slot]()
        self._num_ready_recycled_slots = 0
        self._occupied_slot_ids = set()
        self._slot_id_offset = slot_id_offset
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
        assert_critical(len(self._occupied_slot_ids) == 0, "some slots are still in use")
        assert_critical(
            len(self._recycled_slots) == self._num_active_slots, "some slots are not free"
        )

    @property
    def num_free_slots(self) -> int:
        return len(self._recycled_slots) + max(self._target_capacity - self._num_active_slots, 0)

    @property
    def num_occupied_slots(self) -> int:
        return len(self._occupied_slot_ids)

    def _local_idx(self, slot_id: SlotId) -> int:
        """Convert a slot_id (which may carry an offset) back to the local 0-based index."""
        return int(slot_id) - self._slot_id_offset

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
            slot = Slot(
                SlotId(self._num_active_slots + self._slot_id_offset), CachedCudaEvent.NULL, None
            )
            self._num_active_slots += 1
        else:
            slot = self._recycled_slots.popleft()
            assert slot.has_valid_slot
        self._occupied_slot_ids.add(slot.slot_id)
        return slot

    # The reason why we don't use allocate() multiple times is that if what user need is all or none,
    # and when we don't have enough free slots, we will free these newly allocated slots by appending
    # them to the back of the recycled slot queue, which may impact perf.
    def allocate_multiple(self, num_slots: int) -> list[Slot]:
        if num_slots < 0:
            raise LogicError("SlotAllocator.allocate_multiple: slot count must be non-negative")
        if self.num_free_slots < num_slots:
            raise OutOfPagesError("Not enough free slots")
        return [self.allocate() for _ in range(num_slots)]

    def release(self, slot: Slot) -> None:
        assert slot.has_valid_slot
        slot = slot.move_to_new_slot()
        if (
            self._local_idx(slot.slot_id) >= self._capacity
            or slot.slot_id not in self._occupied_slot_ids
        ):
            raise LogicError(f"Slot {slot.slot_id} is not occupied")
        assert type(slot) is Slot and slot.has_valid_slot
        if self._local_idx(slot.slot_id) < self._target_capacity:
            self._recycled_slots.append(slot)
        else:
            self._overflow_slots.append(slot)
        self._occupied_slot_ids.discard(slot.slot_id)
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
            if self._local_idx(slot.slot_id) < new_num_slots:
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
            for id in range(
                self._slot_id_offset + self._target_capacity,
                self._slot_id_offset + self._capacity,
            )
            if SlotId(id) in self._occupied_slot_ids
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
                self._target_capacity <= self._local_idx(slot.slot_id) < self._capacity
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
        if allocator.num_slots != 0:
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
        num_slots = self._slot_allocator.num_slots
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


class GpuPoolGroup:
    """GPU pool group supporting 1 or N locality domains.

    Manages a list of ``SlotAllocator`` instances (one per locality domain) over shared
    ``GpuSlotPool`` instances.  For a single locality domain (the default) the allocator
    list has one entry with offset 0.  For N locality domains, allocator k has a
    canonical slot-id stride derived from ``max(slot_size_list)`` so that the
    byte-space ``LOCALIZATION_OFFSET`` remains 1 TiB while slot_ids stay in a
    compact int32-friendly range.

    All slot operations accept an optional ``locality_domain_id`` (default 0) so that
    single-locality domain call sites remain unchanged.
    """

    __slots__ = ("_slot_allocators", "_pools", "_destroyed", "_slot_id_offset")

    _slot_allocators: list[SlotAllocator]
    _pools: TypedIndexList[PoolIndex, GpuSlotPool]
    _destroyed: bool
    _slot_id_offset: int

    @staticmethod
    def _query_localized_gpu_memory(locality_domain_id: int) -> int:
        # Placeholder — will be replaced with a real per-locality domain capacity API.
        # Assuming there are only 2 locality domains for now.
        assert locality_domain_id in (0, 1)
        return query_total_gpu_memory() // 2

    def __init__(
        self,
        num_slots: int | list[int],
        slot_size_list: TypedIndexList[PoolIndex, int],
        shared_phys_mem_pool: PooledPhysMemAllocator,
    ) -> None:
        num_locality_domains = shared_phys_mem_pool.num_locality_domains
        # Normalise scalar to per-locality domain list.
        if isinstance(num_slots, int):
            num_slots = [num_slots] * num_locality_domains
        assert len(num_slots) == num_locality_domains

        max_slot_size = max(slot_size_list)
        self._slot_id_offset = (
            div_up(LOCALIZATION_OFFSET, max_slot_size) if num_locality_domains > 1 else 0
        )

        # One SlotAllocator per locality domain; slot ids use a canonical per-locality domain stride
        # derived from the largest pool slot size in the group.
        self._slot_allocators = [
            SlotAllocator(num_slots[k], slot_id_offset=k * self._slot_id_offset)
            for k in range(num_locality_domains)
        ]
        self._destroyed = False

        phys_mem_size = shared_phys_mem_pool.phys_mem_size

        # Compute per-locality domain VM sizes.
        if num_locality_domains == 1:
            gpu_memory = [query_total_gpu_memory()]
        else:
            gpu_memory = [self._query_localized_gpu_memory(k) for k in range(num_locality_domains)]

        self._pools = typed_map(
            slot_size_list,
            lambda slot_size: GpuSlotPool(
                slot_size,
                [
                    round_down(int(gpu_memory[k] * slot_size / max_slot_size), phys_mem_size)
                    for k in range(num_locality_domains)
                ],
                shared_phys_mem_pool,
                num_slots,
            ),
        )

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        if self._destroyed:
            return
        for allocator in self._slot_allocators:
            if allocator._capacity != 0:
                # Best-effort teardown: when destroy() runs during an error
                # unwind, slots may still be occupied by never-freed requests
                # (or a shrink may already be in flight), so the shrink cannot
                # complete. Raising here aborts shutdown mid-way and turns a
                # per-rank error into a whole-instance hang (peer ranks block
                # until the 300s watchdog); warn and release the pools instead.
                try:
                    allocator._synchronize()
                    if not allocator.shrink_in_progress:
                        allocator.prepare_for_shrink(0)
                    allocator.finish_shrink()
                except (RuntimeError, LogicError, AssertionError) as e:
                    warnings.warn(
                        f"KV cache slot allocator teardown incomplete ({e}); "
                        "releasing pools anyway."
                    )
        for pool in self._pools:
            pool.destroy()
        self._destroyed = True

    # ------------------------------------------------------------------
    # Properties (locality_domain-agnostic)
    # ------------------------------------------------------------------

    @property
    def num_locality_domains(self) -> int:
        return len(self._slot_allocators)

    @property
    def num_pools(self) -> PoolIndex:
        return PoolIndex(len(self._pools))

    @property
    def slot_size(self) -> TypedIndexList[PoolIndex, int]:
        return typed_map(self._pools, lambda p: p.slot_size)

    # ------------------------------------------------------------------
    # Slot allocator + pool operations (all take locality_domain_id, defaulting to 0)
    # ------------------------------------------------------------------

    def num_slots(self, locality_domain_id: int = 0) -> int:
        num_slots = self._slot_allocators[locality_domain_id]._capacity
        assert num_slots <= self._get_num_slots_from_pools(locality_domain_id)
        return num_slots

    def num_free_slots(self, locality_domain_id: int = 0) -> int:
        return self._slot_allocators[locality_domain_id].num_free_slots

    def allocate(self, locality_domain_id: int = 0) -> Slot:
        slot = self._slot_allocators[locality_domain_id].allocate()
        # Stamp locality_domain_id onto the slot so that any holder can release it to the
        # correct locality domain allocator without needing to track locality_domain_id separately.
        slot.locality_domain_id = locality_domain_id
        return slot

    def allocate_multiple(self, num_slots: int, locality_domain_id: int = 0) -> list[Slot]:
        slots = self._slot_allocators[locality_domain_id].allocate_multiple(num_slots)
        for slot in slots:
            slot.locality_domain_id = locality_domain_id
        return slots

    def release(self, slot: Slot, locality_domain_id: int = 0) -> None:
        self._slot_allocators[locality_domain_id].release(slot)

    def num_bytes(self, locality_domain_id: int = 0) -> int:
        return sum(pool.num_bytes(locality_domain_id) for pool in self._pools)

    def resize_pools(self, new_num_slots: int | None, locality_domain_id: int = 0) -> None:
        """Resize all pools for the given locality domain, but not its slot allocator.

        If new_num_slots is None, resize to match that locality domain's slot allocator
        capacity.  If an exception is raised, pool sizes may be imbalanced;
        call resize_pools() again with None to recover.
        """
        if new_num_slots is None:
            new_num_slots = self._slot_allocators[locality_domain_id].num_slots
        for pool in self._pools:
            pool.resize(new_num_slots, locality_domain_id)
        assert NDEBUG or self._check(locality_domain_id, allow_mismatch=True)

    def slot_address(self, slot_id: SlotId, locality_domain_id: int = 0) -> HomoTuple[Address]:
        """Return addresses across all pools for ``slot_id`` in ``locality_domain_id``'s VA region."""
        if (
            self.num_locality_domains > 1
            and self._slot_id_offset > 0
            and int(slot_id) >= self._slot_id_offset
        ):
            locality_domain_id = self.get_locality_domain_id(slot_id)
        local_idx = self.get_local_slot_index(slot_id, locality_domain_id)
        return tuple(pool.slot_address(local_idx, locality_domain_id) for pool in self._pools)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_locality_domain_id(self, slot_id: SlotId) -> int:
        if self.num_locality_domains == 1 or self._slot_id_offset == 0:
            return 0
        locality_domain_id = int(slot_id) // self._slot_id_offset
        if not (0 <= locality_domain_id < self.num_locality_domains):
            raise LogicError(
                f"Slot {slot_id} is out of range for {self.num_locality_domains} locality domains"
            )
        return locality_domain_id

    def get_local_slot_index(self, slot_id: SlotId, locality_domain_id: int | None = None) -> int:
        if locality_domain_id is None:
            locality_domain_id = self.get_locality_domain_id(slot_id)
        if self.num_locality_domains == 1 or self._slot_id_offset == 0:
            return int(slot_id)
        raw_slot_id = int(slot_id)
        lower = locality_domain_id * self._slot_id_offset
        upper = lower + self._slot_id_offset
        if lower <= raw_slot_id < upper:
            return raw_slot_id - lower
        # Callers that already know the target locality domain may pass a local slot id
        # (for example SlotId(0) when querying that locality domain's base address).
        return raw_slot_id

    def _get_num_slots_from_pools(self, locality_domain_id: int = 0) -> int:
        return min(p.num_slots(locality_domain_id) for p in self._pools)

    def _check(self, locality_domain_id: int = 0, allow_mismatch: bool = False) -> bool:
        pool_num_slots = self._get_num_slots_from_pools(locality_domain_id)
        allocator_num_slots = self._slot_allocators[locality_domain_id].num_slots
        return (
            allocator_num_slots <= pool_num_slots
            if allow_mismatch
            else allocator_num_slots == pool_num_slots
        )

    @staticmethod
    def _compute_num_phys_mem(
        slot_size_list: Sequence[int], num_slots: int, phys_mem_size: int
    ) -> HomoTuple[int]:
        return tuple(
            GpuSlotPool._compute_num_phys_mem(slot_size, num_slots, phys_mem_size)
            for slot_size in slot_size_list
        )


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
        num_slots = 1 << 63
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
    """GPU cache storage tier supporting 1 or N locality domains.

    The number of locality domains is determined by whether locality domain is enabled in the GPU
    tier config and localization is supported on the current device.  For
    N > 1, total_quota is split evenly across locality domains and the allocator is
    created via ``PooledPhysMemAllocator.create_localized``.

    All slot operations accept an optional ``locality_domain_id`` (default 0) so that
    single-locality domain call sites remain unchanged.
    """

    TIER: ClassVar[CacheTier] = CacheTier.GPU_MEM
    __slots__ = ("shared_phys_mem_pool",)
    shared_phys_mem_pool: PooledPhysMemAllocator

    _pool_groups: TypedIndexList[PoolGroupIndex, GpuPoolGroup]

    @staticmethod
    def _quota_per_locality_domain(
        total_quota: int,
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        phys_mem_size: int,
        num_locality_domains: int,
    ) -> int:
        if num_locality_domains == 1:
            return total_quota

        total_num_pools = sum(len(slot_sizes) for slot_sizes in slot_size_lists)
        min_total_quota = phys_mem_size * total_num_pools * num_locality_domains
        adjusted_total_quota = max(
            min_total_quota, round_up(total_quota, phys_mem_size * num_locality_domains)
        )
        return adjusted_total_quota // num_locality_domains

    def __init__(
        self,
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        slot_count_list: TypedIndexList[PoolGroupIndex, int],
        phys_mem_size: int,
        localized: bool = False,
    ):
        num_pool_groups = typed_len(slot_size_lists)
        assert num_pool_groups == typed_len(slot_count_list), (
            "slot_size_lists and slot_count_list must have the same length"
        )
        super().__init__()
        if localized:
            self.shared_phys_mem_pool = PooledPhysMemAllocator.create_localized(phys_mem_size)
        else:
            self.shared_phys_mem_pool = PooledPhysMemAllocator(phys_mem_size)

        num_locality_domains = self.shared_phys_mem_pool.num_locality_domains
        if num_locality_domains > 1:
            slot_count_list = typed_map(
                slot_count_list, lambda count: max(1, count // num_locality_domains)
            )
        self._pool_groups = make_typed(
            lambda pg_idx: GpuPoolGroup(
                [slot_count_list[pg_idx]] * num_locality_domains,
                slot_size_lists[pg_idx],
                self.shared_phys_mem_pool,
            ),
            num_pool_groups,
        )

    # ------------------------------------------------------------------
    # locality domain info
    # ------------------------------------------------------------------

    @property
    def num_locality_domains(self) -> int:
        return self.shared_phys_mem_pool.num_locality_domains

    # ------------------------------------------------------------------
    # Slot operations — all accept locality_domain_id (default 0)
    # ------------------------------------------------------------------

    @override
    def allocate(self, pool_group_index: PoolGroupIndex, locality_domain_id: int = 0) -> Slot:
        return self._pool_groups[pool_group_index].allocate(locality_domain_id)

    @override
    def allocate_multiple(
        self, pool_group_index: PoolGroupIndex, num_slots: int, locality_domain_id: int = 0
    ) -> list[Slot]:
        return self._pool_groups[pool_group_index].allocate_multiple(num_slots, locality_domain_id)

    @override
    def release(
        self, pool_group_index: PoolGroupIndex, slot: Slot, locality_domain_id: int = 0
    ) -> None:
        self._pool_groups[pool_group_index].release(slot, locality_domain_id)

    @override
    def num_slots(self, pool_group_index: PoolGroupIndex, locality_domain_id: int = 0) -> int:
        return self._pool_groups[pool_group_index].num_slots(locality_domain_id)

    @override
    def get_num_free_slots(
        self, pool_group_index: PoolGroupIndex, locality_domain_id: int = 0
    ) -> int:
        return self._pool_groups[pool_group_index].num_free_slots(locality_domain_id)

    @override
    def slot_address(
        self,
        pool_group_index: PoolGroupIndex,
        pool_index: PoolIndex,
        slot_id: SlotId,
        locality_domain_id: int = 0,
    ) -> Address:
        pool_group = self._pool_groups[pool_group_index]
        local_idx = int(slot_id)
        if pool_group.num_locality_domains > 1 and pool_group._slot_id_offset > 0:
            if int(slot_id) >= pool_group._slot_id_offset:
                locality_domain_id = pool_group.get_locality_domain_id(slot_id)
            local_idx = pool_group.get_local_slot_index(slot_id, locality_domain_id)
        return pool_group._pools[pool_index].slot_address(local_idx, locality_domain_id)

    # ------------------------------------------------------------------
    # Aggregated quota / ratio across all locality domains
    # ------------------------------------------------------------------

    @property
    @override
    def total_quota(self) -> int:
        granularity = self.pool_size_granularity
        quota = 0
        for pg in self._pool_groups:
            for p in pg._pools:
                for uid in range(self.num_locality_domains):
                    quota += round_up(p.num_bytes(uid), granularity)
        return quota

    @property
    @override
    def ratio_list(self) -> TypedIndexList[PoolGroupIndex, float]:
        num_pool_groups = self.num_pool_groups
        ret = filled_list(0.0, num_pool_groups)
        total = 0
        for i, pg in typed_enumerate(self._pool_groups):
            size = sum(pg.num_bytes(uid) for uid in range(self.num_locality_domains))
            total += size
            ret[i] = size
        assert total > 0
        for i in typed_range(num_pool_groups):
            ret[i] /= total
        return ret

    @property
    @override
    def slot_count_list(self) -> TypedIndexList[PoolGroupIndex, int]:
        """Per-locality domain slot count (symmetric allocation: same for all locality domains)."""
        return typed_map(self._pool_groups, lambda pg: pg.num_slots(0))

    # ------------------------------------------------------------------
    # Granularity, post_resize, destroy
    # ------------------------------------------------------------------

    @property
    def pool_size_granularity(self) -> int:
        return self.shared_phys_mem_pool.phys_mem_size

    @override
    def post_resize(self) -> None:
        super().post_resize()
        self.shared_phys_mem_pool.clear()

    @override
    def destroy(self) -> None:
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
