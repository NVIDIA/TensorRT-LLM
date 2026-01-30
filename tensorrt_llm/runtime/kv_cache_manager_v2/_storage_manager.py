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

import math
import os
import warnings
from dataclasses import dataclass
from typing import Iterator, Sequence, cast

from . import rawref
from ._common import (
    GPU_LEVEL,
    NDEBUG,
    Address,
    CacheLevel,
    CacheTier,
    LayerId,
    MemAddress,
    PageIndex,
    PageStatus,
)
from ._config import CacheTierConfig, DataRole, DiskCacheTierConfig
from ._copy_engine import CopyTask, batched_copy
from ._eviction_controller import EvictablePage, PerLevelEvictionController
from ._exceptions import OutOfPagesError
from ._life_cycle_registry import LifeCycleId, LifeCycleRegistry
from ._page import Page
from ._storage import CacheLevelStorage
from ._storage._config import BufferAttr, BufferId, SlotDesc, StorageConfig
from ._storage._core import (
    DiskCacheLevelStorage,
    GpuCacheLevelStorage,
    HostCacheLevelStorage,
    PoolGroupBase,
    PoolGroupIndex,
    PoolIndex,
    Slot,
    SlotId,
)
from ._utils import (
    Array2D,
    CachedCudaEvent,
    HomoTuple,
    TemporaryCudaStream,
    TypedIndexList,
    filled_array2d,
    filled_list,
    get_uniform_attribute,
    make_typed,
    map_optional,
    partition,
    remove_if,
    round_up,
    typed_enumerate,
    typed_range,
)


class CacheLevelManager:
    __slots__ = ("cache_level", "storage", "controller")
    cache_level: CacheLevel
    storage: CacheLevelStorage
    controller: PerLevelEvictionController

    @property
    def cache_tier(self) -> CacheTier:
        return self.storage.cache_tier

    def __init__(
        self,
        life_cycle_grouping: TypedIndexList[LifeCycleId, PoolGroupIndex],
        cache_level: CacheLevel,
        config: CacheTierConfig,
        slot_size_lists: Sequence[Sequence[int]],
        init_ratio: Sequence[float],
    ):
        self.cache_level = cache_level
        self.storage = self._create_cache_level_storage(config, slot_size_lists, init_ratio)
        self.controller = PerLevelEvictionController(life_cycle_grouping, cache_level)

    @property
    def num_pool_groups(self) -> PoolGroupIndex:
        assert self.storage.num_pool_groups == self.controller.num_pool_groups
        return self.storage.num_pool_groups

    @staticmethod
    def _create_cache_level_storage(
        config: CacheTierConfig,
        slot_size_lists: Sequence[Sequence[int]],
        init_ratio: Sequence[float],
    ) -> CacheLevelStorage:
        quota = config.quota
        num_pools = sum(len(sizes) for sizes in slot_size_lists)

        def adjust_quota(quota: int, granularity: int) -> int:
            return max(granularity * num_pools, round_up(quota, granularity))

        if config.tier == CacheTier.GPU_MEM:
            page_size = 2 << 20
            phys_mem_size = page_size << min(4, max(0, int(math.log(quota / (page_size * 512), 2))))
            quota = adjust_quota(quota, phys_mem_size)
            return GpuCacheLevelStorage(quota, slot_size_lists, init_ratio, phys_mem_size)
        elif config.tier == CacheTier.HOST_MEM:
            quota = adjust_quota(quota, HostCacheLevelStorage.POOL_SIZE_GRANULARITY)
            return HostCacheLevelStorage(quota, slot_size_lists, init_ratio)
        elif config.tier == CacheTier.DISK:
            assert isinstance(config, DiskCacheTierConfig)
            assert os.path.isdir(config.path), (
                f"Disk path {config.path} does not exist or is not a directory"
            )
            quota = adjust_quota(quota, DiskCacheLevelStorage.POOL_SIZE_GRANULARITY)
            filename_template = os.path.join(config.path, "g{}p{}.bin")
            return DiskCacheLevelStorage(quota, slot_size_lists, init_ratio, filename_template)
        else:
            raise ValueError(f"Invalid cache tier: {config.tier}")


@dataclass(slots=True, frozen=True)
class StorageStatistics:
    "All in number of slots, for one pool group"

    slot_size: HomoTuple[int]
    total: int
    free: int
    evictable: int

    @property
    def available(self) -> int:
        return self.free + self.evictable

    @property
    def unavailable(self) -> int:
        return self.total - self.available


class StorageManager:
    __slots__ = (
        "_life_cycles",
        "_layer_to_life_cycle_ids",
        "_slot_to_page_indices",
        "_buffer_attr",
        "_life_cycle_grouping",
        "_levels",
        "_cached_num_pool_groups",
        "_slot_desc_list",
        "__rawref__",
    )
    _life_cycles: LifeCycleRegistry
    _layer_to_life_cycle_ids: dict[LayerId, LifeCycleId]
    _slot_to_page_indices: TypedIndexList[LifeCycleId, int]
    _buffer_attr: dict[BufferId, BufferAttr]
    _life_cycle_grouping: TypedIndexList[LifeCycleId, PoolGroupIndex]
    _levels: TypedIndexList[CacheLevel, CacheLevelManager]
    _cached_num_pool_groups: PoolGroupIndex
    _slot_desc_list: TypedIndexList[PoolGroupIndex, SlotDesc]
    __rawref__: rawref.ref["StorageManager"]

    def __init__(self, life_cycles: LifeCycleRegistry, config: StorageConfig) -> None:
        self.__rawref__ = rawref.NULL
        assert config.cache_tiers[GPU_LEVEL].tier == CacheTier.GPU_MEM, (
            "The first cache tier must be GPU memory"
        )
        self._life_cycles = life_cycles
        self._layer_to_life_cycle_ids = config.layer_to_life_cycle_ids()
        self._slot_to_page_indices = config.slot_to_page_indices()
        self._buffer_attr = config.buffer_attributes()
        self._life_cycle_grouping = config.life_cycle_grouping()
        slot_size_lists = [pg.slot_size_list for pg in config.slot_desc_list]
        # @TODO: accept an optional avg_seq_len param and consider sliding window.
        init_ratio = [
            float(sum(pg.slot_size_list) * len(pg.variants)) for pg in config.slot_desc_list
        ]
        total = sum(init_ratio)
        init_ratio = [x / total for x in init_ratio]
        num_levels = CacheLevel(len(config.cache_tiers))
        self._levels = cast(
            TypedIndexList,
            [
                CacheLevelManager(
                    self._life_cycle_grouping, i, config.cache_tiers[i], slot_size_lists, init_ratio
                )
                for i in typed_range(num_levels)
            ],
        )
        self._cached_num_pool_groups = get_uniform_attribute(
            self._levels, lambda level: level.storage.num_pool_groups
        )
        self._slot_desc_list = config.slot_desc_list

    def __del__(self) -> None:
        self.__rawref__.invalidate()

    def get_pool_group_index(self, life_cycle: LifeCycleId) -> PoolGroupIndex:
        return self._life_cycle_grouping[life_cycle]

    def new_gpu_slots(
        self, num_slots: TypedIndexList[LifeCycleId, int]
    ) -> TypedIndexList[LifeCycleId, list[Slot]]:
        return self.new_slots(GPU_LEVEL, num_slots)

    def new_slots(
        self, level: CacheLevel, num_slots: TypedIndexList[LifeCycleId, int]
    ) -> TypedIndexList[LifeCycleId, list[Slot]]:
        pg_num_slots = filled_list(0, self.num_pool_groups)
        for lc in typed_range(self.num_life_cycles):
            pg_num_slots[self.get_pool_group_index(lc)] += num_slots[lc]
        storage = self._levels[level].storage
        if any(
            pg_num_slots[pg] > storage.get_num_free_slots(pg)
            for pg in typed_range(self.num_pool_groups)
        ):
            self.prepare_free_slots(level, pg_num_slots)
        assert all(
            pg_num_slots[pg] <= storage.get_num_free_slots(pg)
            for pg in typed_range(self.num_pool_groups)
        )
        ret = filled_list(list[Slot](), self.num_life_cycles)
        try:
            for life_cycle in typed_range(self.num_life_cycles):
                pg_idx = self.get_pool_group_index(life_cycle)
                ret[life_cycle] = storage.allocate_multiple(pg_idx, num_slots[life_cycle])
        except Exception:
            warnings.warn("Exception not expected here. Please report a bug.")
            for lc, slots in typed_enumerate(ret):
                pg_idx = self.get_pool_group_index(lc)
                for s in slots:
                    storage.release(pg_idx, s)
            raise
        return ret

    @property
    def life_cycles(self) -> LifeCycleRegistry:
        return self._life_cycles

    @property
    def num_life_cycles(self) -> LifeCycleId:
        return LifeCycleId(len(self._life_cycle_grouping))

    @property
    def num_pool_groups(self) -> PoolGroupIndex:
        return self._cached_num_pool_groups

    @property
    def num_cache_levels(self) -> CacheLevel:
        return CacheLevel(len(self._levels))

    def is_last_level(self, level: CacheLevel) -> bool:
        return level == self.num_cache_levels - 1

    @property
    def cache_tiers(self) -> HomoTuple[CacheTier]:
        return tuple(cache_level.cache_tier for cache_level in self._levels)

    def is_evictable(self, page: EvictablePage, level: CacheLevel | None = None) -> bool:
        """
        Check if a page is evictable. If level is specified, check if the page will be evictable after
        migrating to the given level.
        """
        status = page.status
        level = page.cache_level if level is None else level
        # droppable pages that are not committed should be dropped immediately.
        # held pages in last level cache can't be evicted.
        return (status == PageStatus.DROPPABLE and page.is_committed()) or (
            status == PageStatus.HELD and level < self.num_cache_levels - 1
        )

    def prepare_free_slots(
        self, level: CacheLevel, requirements: TypedIndexList[PoolGroupIndex, int]
    ) -> None:
        goals = filled_array2d(self.num_cache_levels, self.num_pool_groups, 0)
        for pg in typed_range(self.num_pool_groups):
            goals[level, pg] = requirements[pg]
        fallen_pages = make_typed(lambda: list[Page](), self.num_pool_groups)
        self._prepare_free_slots(goals, level, fallen_pages)

    def _prepare_free_slots(
        self,
        goals: Array2D[CacheLevel, PoolGroupIndex, int],
        lvl_id: CacheLevel,
        fallen_pages: TypedIndexList[PoolGroupIndex, list[Page]],
    ) -> None:
        assert NDEBUG or goals.rows == self.num_cache_levels and goals.cols == self.num_pool_groups
        assert NDEBUG or all(
            all(p.cache_level < lvl_id for p in pages) for pages in fallen_pages
        ), "Fallen pages must come from upper cache levels"
        storage = self._levels[lvl_id].storage
        ctrl = self._levels[lvl_id].controller
        num_to_evict = filled_list(0, self.num_pool_groups)
        held_pages = make_typed(lambda: list[Page](), self.num_pool_groups)
        for pg_idx in typed_range(self.num_pool_groups):
            goal = goals[lvl_id, pg_idx]
            fallen = len(fallen_pages[pg_idx])
            old_free_cnt = storage.get_num_free_slots(pg_idx)
            evictable_cnt = ctrl.num_evictable_pages(pg_idx)
            num_to_evict[pg_idx] = max(0, min(goal + fallen - old_free_cnt, evictable_cnt))
            fallen_held_cnt = 0  # fallen held pages we must accept in the current level.
            if self.is_last_level(lvl_id):
                held_pages[pg_idx] = remove_if(
                    fallen_pages[pg_idx], lambda p: p.status == PageStatus.HELD
                )
                fallen_held_cnt = len(held_pages[pg_idx])
                if fallen_held_cnt > old_free_cnt + evictable_cnt:
                    # Do we need to revert the eviction we did before? Maybe not.
                    raise OutOfPagesError(
                        "Too many held pages are being evicted to the last-level cache for group {pg_idx}"
                    )
            if old_free_cnt + evictable_cnt - fallen_held_cnt < goal:
                raise OutOfPagesError(
                    "Impossible to meet the goal ({goal} free slots) for group {pg_idx}"
                )
        evicted = ctrl.evict(num_to_evict)
        accepted_pages = make_typed(lambda: list[Page](), self.num_pool_groups)
        is_last_level = self.is_last_level(lvl_id)
        if is_last_level:
            for pg_idx in typed_range(self.num_pool_groups):
                old_free_cnt = storage.get_num_free_slots(pg_idx)
                num_evicted = len(evicted[pg_idx])
                assert NDEBUG or all(p.status == PageStatus.DROPPABLE for p in evicted[pg_idx])
                if not NDEBUG:
                    dbg_rawrefs = [rawref.ref(p) for p in evicted[pg_idx]]
                evicted[pg_idx].clear()
                if not NDEBUG:
                    assert all(p() is None for p in dbg_rawrefs)  # pyright: ignore
                new_free_cnt = storage.get_num_free_slots(pg_idx)
                # GC of some pages may trigger removal of radix tree blocks and some other pages.
                assert new_free_cnt >= num_evicted + old_free_cnt
                assert len(held_pages[pg_idx]) <= new_free_cnt
                fallen_pages[pg_idx].extend(held_pages[pg_idx])
                held_pages[pg_idx].clear()
                goal = goals[lvl_id, pg_idx]
                num_accepted = min(new_free_cnt - goal, len(fallen_pages[pg_idx]))
                assert num_accepted >= 0
                accepted_pages[pg_idx] = (
                    fallen_pages[pg_idx][-num_accepted:] if num_accepted > 0 else []
                )
                fallen_pages[pg_idx].clear()
        else:
            assert all(len(g) == 0 for g in held_pages)
            for pg_idx in typed_range(self.num_pool_groups):
                old_free_cnt = storage.get_num_free_slots(pg_idx)
                e = evicted[pg_idx]
                num_evicted = len(e)
                fallen_pages[pg_idx][:0] = cast(list[Page], e)
                e.clear()
                num_accepted = min(
                    old_free_cnt + num_evicted - goals[lvl_id, pg_idx], len(fallen_pages[pg_idx])
                )
                assert num_accepted >= 0
                if num_accepted > 0:
                    accepted_pages[pg_idx] = fallen_pages[pg_idx][-num_accepted:]
                    del fallen_pages[pg_idx][-num_accepted:]
            self._prepare_free_slots(goals, CacheLevel(lvl_id + 1), fallen_pages)
        assert all(len(f) == 0 for f in fallen_pages)
        # migrate pages
        for pg_idx in typed_range(self.num_pool_groups):
            partitioned = partition(
                accepted_pages[pg_idx],
                lambda p: (p.cache_level, self.get_pool_group_index(p.life_cycle)),
            )
            accepted_pages[pg_idx].clear()
            for (src_lvl, pg_idx), pages in partitioned.items():
                dst_lvl = lvl_id
                self._batched_migrate(pg_idx, dst_lvl, src_lvl, pages, update_src=True)
                for p in pages:
                    if is_last_level and p.status == PageStatus.HELD:
                        continue
                    self._levels[dst_lvl].controller.schedule_for_eviction(p)
        return

    def _batched_migrate(
        self,
        pool_group_index: PoolGroupIndex,
        dst_level: CacheLevel,
        src_level: CacheLevel,
        src_pages: Sequence[Page],
        update_src: bool,
    ) -> Sequence[Slot] | None:
        "Free slots must be prepared before calling this function."
        assert dst_level != src_level, "dst_level and src_level must be different"
        num_slots = len(src_pages)
        num_pools = self.num_pools(pool_group_index)
        src_pool_group = self._pool_group(src_level, pool_group_index)
        dst_pool_group = self._pool_group(dst_level, pool_group_index)
        if dst_pool_group.num_free_slots < num_slots:
            raise OutOfPagesError("Not enough free slots")
        dst_slots = dst_pool_group.allocate_multiple(num_slots)
        try:
            assert len(dst_slots) == num_slots
            prior_events: set[CachedCudaEvent] = set()
            tasks_per_pool: list[list[CopyTask]] = [[]] * num_pools
            for src, dst in zip(src_pages, dst_slots):
                assert src.node_ref is None
                prior_events.update((dst.ready_event, src.ready_event))
                dst_addresses = dst_pool_group.slot_address(dst.slot_id)
                src_addresses = src_pool_group.slot_address(src.slot_id)
                for pool_idx in range(num_pools):
                    tasks_per_pool[pool_idx].append(
                        CopyTask(dst_addresses[pool_idx], src_addresses[pool_idx])
                    )
            dst_tier = self._levels[dst_level].cache_tier
            src_tier = self._levels[src_level].cache_tier
            with TemporaryCudaStream(prior_events) as stream:
                slot_sizes = self.slot_size(pool_group_index)
                for pool_idx, tasks in enumerate(tasks_per_pool):
                    batched_copy(dst_tier, src_tier, slot_sizes[pool_idx], tasks, stream.get())
            finish_event = stream.take_finish_event()
            for src, dst in zip(src_pages, dst_slots):
                dst.ready_event = finish_event
                src.ready_event = (
                    finish_event  # compulsory for the next owner getting this slot from the pool.
                )
                if update_src:
                    scheduled_for_eviction = src.scheduled_for_eviction
                    if scheduled_for_eviction:
                        self.exclude_from_eviction(src)
                    src_pool_group.release(src)
                    src.set_slot(dst)
                    src.cache_level = dst_level
                    if scheduled_for_eviction:
                        self.schedule_for_eviction(src)
            return None if update_src else dst_slots
        except Exception:
            for s in dst_slots:
                dst_pool_group.release(s)
            raise

    def _pool_group(
        self, cache_level: CacheLevel, pool_group_index: PoolGroupIndex
    ) -> PoolGroupBase:
        return self._levels[cache_level].storage._pool_groups[pool_group_index]

    def num_pools(self, pool_group_index: PoolGroupIndex) -> PoolIndex:
        return get_uniform_attribute(
            self._levels, lambda level: level.storage._pool_groups[pool_group_index].num_pools
        )

    def slot_size(self, pool_group_index: PoolGroupIndex) -> HomoTuple[int]:
        return get_uniform_attribute(
            self._levels, lambda level: level.storage.slot_size(pool_group_index)
        )

    def num_slots(
        self, pool_group_index: PoolGroupIndex, cache_level: CacheLevel = GPU_LEVEL
    ) -> int:
        return self._levels[cache_level].storage.num_slots(pool_group_index)

    def release_slot(self, life_cycle: LifeCycleId, cache_level: CacheLevel, slot: Slot) -> None:
        pg_idx = self.get_pool_group_index(life_cycle)
        self._levels[cache_level].storage.release(pg_idx, slot)

    def schedule_for_eviction(self, page: EvictablePage) -> None:
        if self.is_evictable(page):
            self._levels[page.cache_level].controller.schedule_for_eviction(page)

    def exclude_from_eviction(self, page: EvictablePage) -> None:
        assert page.node_ref is not None
        self._levels[page.cache_level].controller.remove(page.node_ref)

    def get_mem_pool_base_address(self, layer_id: LayerId, data_role: DataRole) -> MemAddress:
        storage = self._levels[GPU_LEVEL].storage
        attr = self.get_buffer_attr(layer_id, data_role)
        pg_idx = self.get_pool_group_index(attr.life_cycle_id)
        return MemAddress(
            cast(int, storage.slot_address(pg_idx, attr.pool_index, SlotId(0))) + attr.offset
        )

    def get_page_indices_ref(
        self, lc_id: LifeCycleId, pages: Iterator[Page | None]
    ) -> Iterator[int | None]:
        "Reference implementation. Not fast enough for production."
        scale = self._slot_to_page_indices[lc_id]
        return (map_optional(page, lambda p: scale * int(p.slot_id)) for page in pages)

    def get_buffer_attr(self, layer_id: LayerId, data_role: DataRole) -> BufferAttr:
        return self._buffer_attr[BufferId(layer_id, data_role)]

    def slot_address(
        self, level: CacheLevel, pg_idx: PoolGroupIndex, slot_id: SlotId, pool_idx: PoolIndex
    ) -> Address:
        return self._levels[level].storage.slot_address(pg_idx, pool_idx, slot_id)

    def get_page_indices_for_slot(self, life_cycle: LifeCycleId, slot_id: SlotId) -> PageIndex:
        scale = self._slot_to_page_indices[life_cycle]
        return PageIndex(scale * slot_id)

    def get_statistics(
        self, level: CacheLevel = GPU_LEVEL
    ) -> TypedIndexList[PoolGroupIndex, StorageStatistics]:
        ret = make_typed(lambda: StorageStatistics((), 0, 0, 0), self.num_pool_groups)
        for pg_idx in typed_range(self.num_pool_groups):
            pg = self._pool_group(level, pg_idx)
            evictable_cnt = self._levels[level].controller.num_evictable_pages(pg_idx)
            ret[pg_idx] = StorageStatistics(
                pg.slot_size, pg.num_slots, pg.num_free_slots, evictable_cnt
            )
        return ret

    def get_utilization(
        self, level: CacheLevel = GPU_LEVEL
    ) -> TypedIndexList[PoolGroupIndex, float]:
        ret = make_typed(lambda: 0.0, self.num_pool_groups)
        stats = self.get_statistics(level)
        for pg_idx in typed_range(self.num_pool_groups):
            ret[pg_idx] = stats[pg_idx].unavailable / stats[pg_idx].total
        return ret

    def get_overall_utilization(self, level: CacheLevel = GPU_LEVEL) -> float:
        stats = self.get_statistics(level)
        return sum(sum(s.slot_size) * s.unavailable for s in stats) / sum(
            sum(s.slot_size) * s.total for s in stats
        )
