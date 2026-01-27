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
from .._exceptions import LogicError, OutOfPagesError, ResourceBusyError
from .._utils import (
    CachedCudaEvent,
    DynamicBitset,
    HomoTuple,
    HostMem,
    assert_critical,
    div_up,
    query_total_gpu_memory,
    remove_if,
    resize_file,
    round_down,
    round_up,
)

PoolGroupIndex = NewType("PoolGroupIndex", int)
PoolIndex = NewType("PoolIndex", int)
SlotId = NewType("SlotId", int)


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
        "_num_ready_overflow_slots",
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
    _num_ready_overflow_slots: int  # similar to _num_ready_recycled_slots, but for _overflow_slots.

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._num_active_slots = 0
        self._recycled_slots = deque[Slot]()
        self._num_ready_recycled_slots = 0
        self._occupied_mask = DynamicBitset(capacity)
        self._target_capacity = capacity
        self._overflow_slots = []
        self._num_ready_overflow_slots = 0

    def __del__(self) -> None:
        assert_critical(
            self._num_ready_recycled_slots == len(self._recycled_slots)
            and self._num_ready_overflow_slots == len(self._overflow_slots),
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
        elif self._num_active_slots < self.num_slots:
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
            self._try_trigger_shrink()
        self._occupied_mask.clear(slot.slot_id)
        self._scrub_events()
        assert NDEBUG or self._check()

    @property
    def num_slots(self) -> int:
        return self._capacity

    def resize(self, new_num_slots: int) -> None:
        if self._target_capacity != self._capacity:
            self.cancel_scheduled_resize()
        assert NDEBUG or self._check()
        old_num_slots = self.num_slots
        if new_num_slots < self.num_slots and self._occupied_mask.any_set(
            new_num_slots, self.num_slots
        ):
            raise ResourceBusyError("resize cannot remove occupied slots")
        self._occupied_mask.resize(new_num_slots)
        self._capacity = new_num_slots
        self._num_active_slots = min(self._num_active_slots, new_num_slots)
        if new_num_slots < old_num_slots:
            new_recycled_slots = deque[Slot]()
            new_num_ready_recycled_slots = 0
            for idx_recycled, slot in enumerate(self._recycled_slots):
                assert type(slot) is Slot and slot.has_valid_slot
                if slot.slot_id >= new_num_slots:
                    slot.ready_event.synchronize()
                    slot._slot_id = None
                    slot.ready_event = CachedCudaEvent.NULL
                else:
                    new_recycled_slots.append(slot)
                    if idx_recycled < self._num_ready_recycled_slots:
                        new_num_ready_recycled_slots += 1
            self._recycled_slots = new_recycled_slots
            self._num_ready_recycled_slots = new_num_ready_recycled_slots
            self._scrub_events()
        self._target_capacity = self._capacity
        assert NDEBUG or self._check()

    def schedule_resize(self, new_num_slots: int) -> None:
        assert NDEBUG or self._check()
        if new_num_slots >= self.num_slots:
            self.cancel_scheduled_resize()
            self.resize(new_num_slots)
            return
        old_target_capacity = self._target_capacity
        if new_num_slots > old_target_capacity:
            self._recycled_slots.extend(
                remove_if(
                    self._overflow_slots,
                    lambda slot: old_target_capacity <= slot.slot_id < new_num_slots,
                )
            )
            self._num_ready_overflow_slots = 0
        if new_num_slots < old_target_capacity:
            self._overflow_slots.extend(
                remove_if(self._recycled_slots, lambda slot: slot.slot_id >= new_num_slots)
            )
            self._num_ready_recycled_slots = 0
        self._target_capacity = new_num_slots
        self._try_trigger_shrink()
        self._scrub_events()
        assert NDEBUG or self._check()

    def cancel_scheduled_resize(self) -> None:
        assert NDEBUG or self._check()
        self._target_capacity = self._capacity
        self._recycled_slots.extend(remove_if(self._overflow_slots, lambda slot: True))
        self._num_ready_overflow_slots = 0

    def shrink_in_progress(self) -> bool:
        "Indicates if a scheduled shrink is in progress."
        assert self._target_capacity <= self._capacity
        return self._target_capacity < self._capacity

    def get_slots_blocking_shrink(self) -> HomoTuple[SlotId]:
        return tuple(
            SlotId(id)
            for id in range(self._target_capacity, self._capacity)
            if self._occupied_mask.get(id)
        )

    def _try_trigger_shrink(self) -> bool:
        assert NDEBUG or self._check()
        if (
            self.shrink_in_progress()
            and self._target_capacity + len(self._overflow_slots) == self._capacity
        ):
            assert len(set(s.slot_id for s in self._overflow_slots)) == len(self._overflow_slots)
            for slot in self._overflow_slots:
                slot.ready_event.synchronize()
                slot.ready_event = CachedCudaEvent.NULL
            self._overflow_slots.clear()
            self._num_ready_overflow_slots = 0
            self._capacity = self._target_capacity
            self._num_active_slots = min(self._num_active_slots, self._capacity)
            self._scrub_events()
            assert NDEBUG or self._check()
            return True
        return False

    def _scrub_events(self) -> None:
        self._num_ready_recycled_slots = self._scrub_events_impl(
            self._recycled_slots, self._num_ready_recycled_slots
        )
        self._num_ready_overflow_slots = self._scrub_events_impl(
            self._overflow_slots, self._num_ready_overflow_slots
        )

    def _check(self) -> bool:
        return (
            self._num_active_slots <= self._capacity
            and self._target_capacity <= self._capacity
            and (self.shrink_in_progress() or len(self._overflow_slots) == 0)
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
        while self._num_ready_recycled_slots != len(
            self._recycled_slots
        ) or self._num_ready_overflow_slots != len(self._overflow_slots):
            self._scrub_events()


class PoolGroupBase:
    __slots__ = ("_slot_allocator", "_pools")

    _slot_allocator: SlotAllocator
    _pools: HomoTuple[SlotPoolBase]

    def __init__(self, num_slots: int) -> None:
        self._slot_allocator = SlotAllocator(num_slots)

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        if self._slot_allocator._capacity == 0:
            return
        self._slot_allocator._synchronize()
        for pool in self._pools:
            pool.destroy()
        self._slot_allocator.resize(0)

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

    def resize_slot_allocator(self, new_num_slots: int | None) -> None:
        """
        Resize the slot allocator, but not pools. If new_num_slots is None, make slot allocator match the pool sizes.
        """
        if new_num_slots is None:
            new_num_slots = self._get_num_slots_from_pools()
        self._slot_allocator.resize(new_num_slots)
        assert NDEBUG or self._check(True)

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
    def slot_size(self) -> HomoTuple[int]:
        return tuple(pool.slot_size for pool in self._pools)

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
        slot_size_list: Sequence[int],
        shared_phys_mem_pool: PooledPhysMemAllocator,
    ):
        super().__init__(num_slots)
        total_gpu_memory = query_total_gpu_memory()
        max_slot_size = max(slot_size_list)
        phys_mem_size = shared_phys_mem_pool.phys_mem_size
        self._pools = tuple(
            GpuSlotPool(
                slot_size,
                round_down(int(total_gpu_memory * slot_size / max_slot_size), phys_mem_size),
                shared_phys_mem_pool,
                num_slots,
            )
            for slot_size in slot_size_list
        )


class HostPoolGroup(PoolGroupBase):
    __slots__ = ()

    def __init__(self, num_slots: int, slot_size_list: Sequence[int]):
        super().__init__(num_slots)
        self._pools = tuple(HostSlotPool(slot_size, num_slots) for slot_size in slot_size_list)


class DiskPoolGroup(PoolGroupBase):
    __slots__ = ()

    def __init__(self, num_slots: int, slot_size_list: Sequence[int], filename_template: str):
        super().__init__(num_slots)
        self._pools = tuple(
            DiskSlotPool(filename_template.format(i), slot_size, num_slots)
            for i, slot_size in enumerate(slot_size_list)
        )


class CacheLevelStorage:
    TIER: ClassVar[CacheTier]
    __slots__ = ("_total_quota", "_ratio_list", "_pool_groups")
    _total_quota: int  # fixme: remove _total_quota and _ratio_list and compute from _pool_groups
    _ratio_list: HomoTuple[float]
    _pool_groups: HomoTuple[PoolGroupBase]

    def __init__(self, total_quota: int, ratio_list: Sequence[float]) -> None:
        if not hasattr(self.__class__, "TIER"):
            raise ValueError(f"{self.__class__.__name__} must define 'TIER' as a class variable")
        self._total_quota = total_quota
        self._ratio_list = tuple(ratio_list)

    def __del__(self) -> None:
        self.destroy()

    @property
    def cache_tier(self) -> CacheTier:
        return self.TIER

    def destroy(self) -> None:
        if self._total_quota == 0:
            return
        for pg in self._pool_groups:
            pg.destroy()
        self._total_quota = 0
        self._ratio_list = ()

    def allocate(self, pool_group_index: PoolGroupIndex) -> Slot:
        return self._pool_groups[pool_group_index].allocate()

    def allocate_multiple(self, pool_group_index: PoolGroupIndex, num_slots: int) -> list[Slot]:
        return self._pool_groups[pool_group_index].allocate_multiple(num_slots)

    def release(self, pool_group_index: PoolGroupIndex, slot: Slot) -> None:
        self._pool_groups[pool_group_index].release(slot)

    @property
    def total_quota(self) -> int:
        return self._total_quota

    @property
    def ratio_list(self) -> HomoTuple[float]:
        return self._ratio_list

    def num_slots(self, pool_group_index: PoolGroupIndex) -> int:
        return self._pool_groups[pool_group_index].num_slots

    def get_num_free_slots(self, pool_group_index: PoolGroupIndex) -> int:
        return self._pool_groups[pool_group_index].num_free_slots

    @property
    def slot_count_list(self) -> HomoTuple[int]:
        """
        The number of slots in each pool group.
        """
        return tuple(pg.num_slots for pg in self._pool_groups)

    def slot_size(self, pool_group_index: PoolGroupIndex) -> HomoTuple[int]:
        """
        The slot sizes of each pool in the pool group.
        """
        return self._pool_groups[pool_group_index].slot_size

    @property
    def slot_size_lists(self) -> HomoTuple[HomoTuple[int]]:
        """
        A tuple of tuples, each containing the slot sizes for a pool group.
        """
        return tuple(tuple(p.slot_size for p in pg._pools) for pg in self._pool_groups)

    @property
    def num_pool_groups(self) -> PoolGroupIndex:
        return PoolGroupIndex(len(self._pool_groups))

    def slot_address(
        self, pool_group_index: PoolGroupIndex, pool_index: PoolIndex, slot_id: SlotId
    ) -> Address:
        return self._pool(pool_group_index, pool_index).slot_address(slot_id)

    def resize(
        self, new_total_quota: int | None = None, new_ratio_list: Sequence[float] | None = None
    ) -> None:
        new_slot_count_list = self._compute_slot_count_list(new_total_quota, new_ratio_list)
        self._resize_impl(new_slot_count_list)
        if new_total_quota is not None:
            self._total_quota = new_total_quota
        if new_ratio_list is not None:
            self._ratio_list = tuple(new_ratio_list)

    def _resize_impl(self, new_slot_count_list: Sequence[int]) -> None:
        old_slot_count_list = self.slot_count_list
        assert old_slot_count_list == self._compute_slot_count_list(
            self.total_quota, self.ratio_list
        )
        try:
            # shrink first to avoid intermediate state with excessive memory usage
            for pg, new_slot_count, old_slot_count in zip(
                self._pool_groups, new_slot_count_list, old_slot_count_list
            ):
                if new_slot_count < old_slot_count:
                    pg.resize_slot_allocator(
                        new_slot_count
                    )  # shrink slot allocators first as it can fail for shrinking
                    pg.resize_pools(new_slot_count)
            for pg, new_slot_count, old_slot_count in zip(
                self._pool_groups, new_slot_count_list, old_slot_count_list
            ):
                if new_slot_count > old_slot_count:
                    pg.resize_pools(
                        new_slot_count
                    )  # expand pools first as it can fail for expanding
                    pg.resize_slot_allocator(new_slot_count)
        except Exception:
            self._resize_impl(old_slot_count_list)
            raise

    def _pool(self, pool_group_index: PoolGroupIndex, pool_index: PoolIndex) -> SlotPoolBase:
        return self._pool_groups[pool_group_index]._pools[pool_index]

    # Calculate how many slots will there be in each pool group with the given total_quota and
    # ratio_list. Use _ratio_to_slot_count_list for initialization.
    def _compute_slot_count_list(
        self, total_quota: int | None = None, ratio_list: Sequence[float] | None = None
    ) -> HomoTuple[int]:
        if total_quota is None:
            total_quota = self.total_quota
        if ratio_list is None:
            ratio_list = self.ratio_list
        assert len(ratio_list) == len(self._pool_groups), (
            f"Wrong ratio_list length. Expected {len(self._pool_groups)}, got {len(ratio_list)}"
        )
        return self._ratio_to_slot_count_list(
            total_quota, self.slot_size_lists, ratio_list, self.pool_size_granularity
        )

    @staticmethod
    def _ratio_to_slot_count_list(
        total_quota: int,
        slot_size_lists: Sequence[Sequence[int]],
        ratio_list: Sequence[float],
        pool_size_granularity: int,
    ) -> HomoTuple[int]:
        num_pool_groups = len(ratio_list)
        assert num_pool_groups == len(slot_size_lists)
        assert total_quota % pool_size_granularity == 0
        total_grains = total_quota // pool_size_granularity
        assert total_grains >= sum(len(sizes) for sizes in slot_size_lists)
        remaining_grains = total_grains
        granularity = pool_size_granularity
        slot_cnt_list = [0] * num_pool_groups
        # divide total_quota into pool groups based on init_ratio, then divide quote for each pool_group
        # into pools based on slot_size.
        pg_idx_lst = sorted(range(len(ratio_list)), key=lambda i: ratio_list[i])
        for i, pg in enumerate(pg_idx_lst):
            slot_size_list = slot_size_lists[pg]
            min_pool_grains = [div_up(s, granularity) for s in slot_size_list]
            pct: float = ratio_list[pg] / sum(ratio_list[j] for j in pg_idx_lst[i:])
            pg_grains = max(round(remaining_grains * pct), sum(min_pool_grains))
            num_slots: int = 1 << 63
            remaining_pg_grains = pg_grains
            pool_idx_lst = sorted(range(len(slot_size_list)), key=lambda i: slot_size_list[i])
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
            slot_cnt_list[pg] = num_slots
            remaining_grains -= pg_grains
        assert remaining_grains == 0
        return tuple(slot_cnt_list)

    @property
    def pool_size_granularity(self) -> int:
        return 2 << 20


class GpuCacheLevelStorage(CacheLevelStorage):
    TIER: ClassVar[CacheTier] = CacheTier.GPU_MEM
    __slots__ = ("shared_phys_mem_pool",)
    shared_phys_mem_pool: PooledPhysMemAllocator

    def __init__(
        self,
        total_quota: int,
        slot_size_lists: Sequence[Sequence[int]],
        init_ratio: Sequence[float],
        phys_mem_size: int,
    ):
        assert len(slot_size_lists) == len(init_ratio), (
            "slot_size_lists and init_ratio must have the same length"
        )
        super().__init__(total_quota, init_ratio)
        slot_count_list = self._ratio_to_slot_count_list(
            total_quota, slot_size_lists, init_ratio, phys_mem_size
        )
        self.shared_phys_mem_pool = PooledPhysMemAllocator(phys_mem_size)
        self._pool_groups = tuple(
            GpuPoolGroup(num_slots, slot_size_list, self.shared_phys_mem_pool)
            for slot_size_list, num_slots in zip(slot_size_lists, slot_count_list)
        )

    @override
    def resize(
        self, new_total_quota: int | None = None, new_ratio_list: Sequence[float] | None = None
    ):
        super().resize(new_total_quota, new_ratio_list)
        self.shared_phys_mem_pool.clear()  # clear cached unused phys mem

    @property
    def pool_size_granularity(self) -> int:
        return self.shared_phys_mem_pool.phys_mem_size


class HostCacheLevelStorage(CacheLevelStorage):
    TIER: ClassVar[CacheTier] = CacheTier.HOST_MEM
    POOL_SIZE_GRANULARITY: ClassVar[int] = HostMem.ALIGNMENT
    __slots__ = ()

    def __init__(
        self,
        total_quota: int,
        slot_size_lists: Sequence[Sequence[int]],
        init_ratio: Sequence[float],
    ):
        super().__init__(total_quota, init_ratio)
        slot_count_list = self._ratio_to_slot_count_list(
            total_quota, slot_size_lists, init_ratio, self.pool_size_granularity
        )
        self._pool_groups = tuple(
            HostPoolGroup(num_slots, slot_size_list)
            for slot_size_list, num_slots in zip(slot_size_lists, slot_count_list)
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
        total_quota: int,
        slot_size_lists: Sequence[Sequence[int]],
        init_ratio: Sequence[float],
        filename_template: str,
    ):
        super().__init__(total_quota, init_ratio)
        slot_count_list = self._ratio_to_slot_count_list(
            total_quota, slot_size_lists, init_ratio, self.pool_size_granularity
        )
        self._pool_groups = tuple(
            DiskPoolGroup(num_slots, slot_size_list, filename_template.format(pg_idx, "{}"))
            for pg_idx, (slot_size_list, num_slots) in enumerate(
                zip(slot_size_lists, slot_count_list)
            )
        )

    @property
    def pool_size_granularity(self) -> int:
        return self.POOL_SIZE_GRANULARITY
