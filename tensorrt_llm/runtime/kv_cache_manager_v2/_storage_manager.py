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
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Sequence, cast

from . import rawref
from ._common import (
    GPU_LEVEL,
    NDEBUG,
    Address,
    BlockOrdinal,
    CacheLevel,
    CacheTier,
    LayerId,
    MemAddress,
    PageStatus,
)
from ._config import BatchDesc, CacheTierConfig, DataRole, DiskCacheTierConfig, KVCacheDesc
from ._copy_engine import CopyTask, batched_copy
from ._eviction_controller import EvictablePage, PerLevelEvictionController
from ._exceptions import OutOfPagesError
from ._life_cycle_registry import LifeCycleId, LifeCycleRegistry, compute_scratch_range
from ._page import Page
from ._storage import CacheLevelStorage
from ._storage._config import BufferAttr, BufferId, LayerAttr, SlotDesc, StorageConfig
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
    HalfOpenRange,
    HomoTuple,
    TemporaryCudaStream,
    TypedIndexList,
    div_up,
    filled_array2d,
    filled_list,
    get_uniform_attribute,
    intersect,
    make_typed,
    partition,
    remove_if,
    round_up,
    typed_enumerate,
    typed_len,
    typed_map,
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
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        slot_count_list: TypedIndexList[PoolGroupIndex, int],
    ):
        self.cache_level = cache_level
        self.storage = self._create_cache_level_storage(config, slot_size_lists, slot_count_list)
        self.controller = PerLevelEvictionController(life_cycle_grouping, cache_level)

    @property
    def num_pool_groups(self) -> PoolGroupIndex:
        assert self.storage.num_pool_groups == self.controller.num_pool_groups
        return self.storage.num_pool_groups

    @staticmethod
    def cache_tier_granularity(tier: CacheTier, quota: int) -> int:
        """Compute pool size granularity for a given cache tier and quota."""
        match tier:
            case CacheTier.GPU_MEM:
                page_size = 2 << 20
                return page_size << min(4, max(0, int(math.log(quota / (page_size * 512), 2))))
            case CacheTier.HOST_MEM:
                return HostCacheLevelStorage.POOL_SIZE_GRANULARITY
            case CacheTier.DISK:
                return DiskCacheLevelStorage.POOL_SIZE_GRANULARITY
            case _:
                raise ValueError(f"Invalid cache tier: {tier}")

    @staticmethod
    def _create_cache_level_storage(
        config: CacheTierConfig,
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        slot_count_list: TypedIndexList[PoolGroupIndex, int],
    ) -> CacheLevelStorage:
        match config.tier:
            case CacheTier.GPU_MEM:
                granularity = CacheLevelManager.cache_tier_granularity(
                    CacheTier.GPU_MEM, config.quota
                )
                return GpuCacheLevelStorage(slot_size_lists, slot_count_list, granularity)
            case CacheTier.HOST_MEM:
                return HostCacheLevelStorage(slot_size_lists, slot_count_list)
            case CacheTier.DISK:
                assert isinstance(config, DiskCacheTierConfig)
                assert os.path.isdir(config.path), (
                    f"Disk path {config.path} does not exist or is not a directory"
                )
                filename_template = os.path.join(config.path, "g{}p{}.bin")
                return DiskCacheLevelStorage(slot_size_lists, slot_count_list, filename_template)
            case _:
                raise ValueError(f"Invalid cache tier: {config.tier}")


@dataclass(slots=True, frozen=True)
class StorageStatistics:
    "All in number of slots, for one pool group"

    slot_size: TypedIndexList[PoolIndex, int]
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
        "_layer_attributes",
        "_slot_util_frac_max",
        "_life_cycle_grouping",
        "_slot_desc_list",
        "_levels",
        "_min_slots",
        "__rawref__",
    )
    _life_cycles: LifeCycleRegistry
    _layer_to_life_cycle_ids: dict[LayerId, LifeCycleId]
    _slot_to_page_indices: TypedIndexList[LifeCycleId, TypedIndexList[PoolIndex, int]]
    _slot_util_frac_max: TypedIndexList[LifeCycleId, Fraction]
    _buffer_attr: dict[BufferId, BufferAttr]
    _layer_attributes: dict[LayerId, LayerAttr]
    _life_cycle_grouping: TypedIndexList[LifeCycleId, PoolGroupIndex]
    _slot_desc_list: TypedIndexList[PoolGroupIndex, SlotDesc]
    _levels: TypedIndexList[CacheLevel, CacheLevelManager]
    _min_slots: TypedIndexList[PoolGroupIndex, int]
    __rawref__: rawref.ref["StorageManager"]

    def __init__(
        self,
        life_cycles: LifeCycleRegistry,
        config: StorageConfig,
        tokens_per_block: int,
        enable_swa_scratch_reuse: bool,
        typical_batch: BatchDesc | None = None,
        constraints: list[BatchDesc] | None = None,
    ) -> None:
        self.__rawref__ = rawref.NULL
        assert config.cache_tiers[GPU_LEVEL].tier == CacheTier.GPU_MEM, (
            "The first cache tier must be GPU memory"
        )
        self._life_cycles = life_cycles
        self._layer_to_life_cycle_ids = config.layer_to_life_cycle_ids()
        self._slot_to_page_indices = config.slot_to_page_indices()
        self._layer_attributes = config.layer_attributes()
        self._slot_util_frac_max = filled_list(Fraction(0, 1), life_cycles.size)
        for attr in self._layer_attributes.values():
            if attr.slot_util_frac_max > self._slot_util_frac_max[attr.life_cycle_id]:
                self._slot_util_frac_max[attr.life_cycle_id] = attr.slot_util_frac_max
        self._buffer_attr = config.buffer_attributes()
        self._life_cycle_grouping = config.life_cycle_grouping()
        self._slot_desc_list = config.slot_desc_list
        assert all(pg < self.num_pool_groups for pg in self._life_cycle_grouping)
        assert self.num_pool_groups == PoolGroupIndex(len(set(self._life_cycle_grouping)))
        slot_size_lists = typed_map(self._slot_desc_list, lambda pg: pg.slot_size_list)

        gpu_quota = config.cache_tiers[GPU_LEVEL].quota
        gpu_granularity = CacheLevelManager.cache_tier_granularity(CacheTier.GPU_MEM, gpu_quota)

        self._min_slots = self._compute_min_slots_from_constraints(
            constraints or [], tokens_per_block, enable_swa_scratch_reuse
        )

        # Compute init_ratio from typical_batch, constraints, or fallback.
        if typical_batch is not None:
            init_ratio = self.ratio_from_batch(
                typical_batch, tokens_per_block, enable_swa_scratch_reuse, gpu_granularity
            )
        elif constraints:
            # Use the constraint slot counts as the ratio basis.
            min_bytes = self._slots_to_bytes(self._min_slots, gpu_granularity)
            total = sum(min_bytes)
            init_ratio = typed_map(min_bytes, lambda x: x / total)
        else:
            init_ratio = self.ratio_from_batch(
                BatchDesc([KVCacheDesc(capacity=2049, history_length=2048)]),
                tokens_per_block,
                enable_swa_scratch_reuse,
                gpu_granularity,
            )

        num_levels = CacheLevel(len(config.cache_tiers))
        self._levels = cast(
            TypedIndexList,
            [
                CacheLevelManager(
                    self._life_cycle_grouping,
                    i,
                    config.cache_tiers[i],
                    slot_size_lists,
                    self._compute_slot_count_for_level(
                        config.cache_tiers[i], slot_size_lists, init_ratio
                    ),
                )
                for i in typed_range(num_levels)
            ],
        )
        assert self.num_pool_groups == get_uniform_attribute(
            self._levels, lambda level: level.storage.num_pool_groups
        )

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        if self.__rawref__.is_valid:
            self.__rawref__.invalidate()
            for lvl in self._levels:
                lvl.storage.destroy()

    def get_pool_group_index(self, life_cycle: LifeCycleId) -> PoolGroupIndex:
        return self._life_cycle_grouping[life_cycle]

    def new_gpu_slots(
        self, num_slots: TypedIndexList[LifeCycleId, int]
    ) -> TypedIndexList[LifeCycleId, list[Slot]]:
        return self.new_slots(GPU_LEVEL, num_slots)

    def new_slots(
        self, level: CacheLevel, num_slots: TypedIndexList[LifeCycleId, int]
    ) -> TypedIndexList[LifeCycleId, list[Slot]]:
        lc2pg = self._life_cycle_grouping
        pg_num_slots = filled_list(0, self.num_pool_groups)
        for lc in typed_range(self.num_life_cycles):
            pg_num_slots[lc2pg[lc]] += num_slots[lc]
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
                pg_idx = lc2pg[life_cycle]
                ret[life_cycle] = storage.allocate_multiple(pg_idx, num_slots[life_cycle])
        except Exception:
            warnings.warn("Exception not expected here. Please report a bug.")
            for lc, slots in typed_enumerate(ret):
                pg_idx = lc2pg[lc]
                for s in slots:
                    storage.release(pg_idx, s)
            raise
        return ret

    def new_slots_for_pool_group(
        self, level: CacheLevel, pg_idx: PoolGroupIndex, num_slots: int
    ) -> list[Slot]:
        storage = self._levels[level].storage
        if num_slots > storage.get_num_free_slots(pg_idx):
            num_slots_list = filled_list(0, self.num_pool_groups)
            num_slots_list[pg_idx] = num_slots
            self.prepare_free_slots(level, num_slots_list)
        assert num_slots <= storage.get_num_free_slots(pg_idx)
        try:
            return storage.allocate_multiple(pg_idx, num_slots)
        except Exception:
            warnings.warn("Exception not expected here. Please report a bug.")
            raise

    @property
    def life_cycles(self) -> LifeCycleRegistry:
        return self._life_cycles

    @property
    def num_life_cycles(self) -> LifeCycleId:
        return typed_len(self._life_cycle_grouping)

    @property
    def num_pool_groups(self) -> PoolGroupIndex:
        return typed_len(self._slot_desc_list)

    @property
    def num_cache_levels(self) -> CacheLevel:
        return CacheLevel(len(self._levels))

    def is_last_level(self, level: CacheLevel) -> bool:
        return level == self.num_cache_levels - 1

    @property
    def cache_tiers(self) -> HomoTuple[CacheTier]:
        return tuple(cache_level.cache_tier for cache_level in self._levels)

    def get_ratio_list(self, level: CacheLevel) -> TypedIndexList[PoolGroupIndex, float]:
        return self._levels[level].storage.ratio_list

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
        fallen_pages = make_typed(lambda _: list[Page](), self.num_pool_groups)
        self._prepare_free_slots(goals, level, fallen_pages)

    def force_evict(
        self, level: CacheLevel, min_num_pages: TypedIndexList[PoolGroupIndex, int]
    ) -> None:
        # If we break inside this function with debugpy, pages in `evicted` won't be
        # released even after the function returns. This is a debugpy artifact.
        evicted = self._levels[level].controller.evict(min_num_pages)
        if int(level) == self.num_cache_levels - 1:
            assert all(p.status == PageStatus.DROPPABLE for pages in evicted for p in pages), (
                "Corrupted eviction controller"
            )
            return
        next_lvl = CacheLevel(level + 1)
        goals = filled_array2d(self.num_cache_levels, self.num_pool_groups, 0)
        self._prepare_free_slots(
            goals, next_lvl, cast(TypedIndexList[PoolGroupIndex, list[Page]], evicted)
        )

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
        held_pages = make_typed(lambda _: list[Page](), self.num_pool_groups)
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
        accepted_pages = make_typed(lambda _: list[Page](), self.num_pool_groups)
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
        defrag: bool = False,  # we are doing defragmentation
    ) -> Sequence[Slot] | None:
        "Free slots must be prepared before calling this function."
        assert defrag or dst_level != src_level, (
            "dst_level and src_level must be different unless performing defragmentation"
        )
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
            tasks_per_pool: TypedIndexList[PoolIndex, list[CopyTask]] = make_typed(
                lambda _: list[CopyTask](), num_pools
            )
            for src, dst in zip(src_pages, dst_slots):
                assert defrag or src.node_ref is None
                prior_events.update((dst.ready_event, src.ready_event))
                dst_addresses = dst_pool_group.slot_address(dst.slot_id)
                src_addresses = src_pool_group.slot_address(src.slot_id)
                for pool_idx in typed_range(num_pools):
                    tasks_per_pool[pool_idx].append(
                        CopyTask(dst_addresses[pool_idx], src_addresses[pool_idx])
                    )
            dst_tier = self._levels[dst_level].cache_tier
            src_tier = self._levels[src_level].cache_tier
            with TemporaryCudaStream(prior_events) as stream:
                slot_sizes = self.slot_size(pool_group_index)
                for pool_idx, tasks in typed_enumerate(tasks_per_pool):
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

    def slot_size(self, pool_group_index: PoolGroupIndex) -> TypedIndexList[PoolIndex, int]:
        return self._slot_desc_list[pool_group_index].slot_size_list

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

    def get_mem_pool_base_address(self, pg_idx: PoolGroupIndex, pool_idx: PoolIndex) -> MemAddress:
        storage = self._levels[GPU_LEVEL].storage
        return MemAddress(cast(int, storage.slot_address(pg_idx, pool_idx, SlotId(0))))

    def get_buffer_attr(self, layer_id: LayerId, data_role: DataRole) -> BufferAttr:
        return self._buffer_attr[BufferId(layer_id, data_role)]

    def get_layer_attr(self, layer_id: LayerId) -> LayerAttr:
        return self._layer_attributes[layer_id]

    def slot_address(
        self, level: CacheLevel, pg_idx: PoolGroupIndex, slot_id: SlotId, pool_idx: PoolIndex
    ) -> Address:
        return self._levels[level].storage.slot_address(pg_idx, pool_idx, slot_id)

    def get_statistics(
        self, level: CacheLevel = GPU_LEVEL
    ) -> TypedIndexList[PoolGroupIndex, StorageStatistics]:
        ret = make_typed(
            lambda pg_idx: StorageStatistics(filled_list(0, self.num_pools(pg_idx)), 0, 0, 0),
            self.num_pool_groups,
        )
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
        ret = filled_list(0.0, self.num_pool_groups)
        stats = self.get_statistics(level)
        for pg_idx in typed_range(self.num_pool_groups):
            ret[pg_idx] = stats[pg_idx].unavailable / stats[pg_idx].total
        return ret

    def get_overall_utilization(self, level: CacheLevel = GPU_LEVEL) -> float:
        stats = self.get_statistics(level)
        return sum(sum(s.slot_size) * s.unavailable for s in stats) / sum(
            sum(s.slot_size) * s.total for s in stats
        )

    def shrink_pool_group(
        self,
        level: CacheLevel,
        pg_idx: PoolGroupIndex,
        new_num_slots: int,
        persistent_pages: list[Page],
    ) -> None:
        """Move pages to eliminate overflow slots then shrink the pool group"""
        lc2pg = self._life_cycle_grouping
        assert len(persistent_pages) <= new_num_slots and all(
            p.cache_level == level and lc2pg[p.life_cycle] == pg_idx for p in persistent_pages
        ), "Not enough slots"
        pool_group = self._levels[level].storage._pool_groups[pg_idx]
        assert new_num_slots < pool_group.num_slots, "Not required for expansion of pools"
        ctrl = self._levels[level].controller
        # pages with overflow slots and their indices in the eviction queue.
        overflow_slots = deque[tuple[int, Page]]()
        for i, p in enumerate(cast(Iterator[Page], ctrl.page_iterator(pg_idx))):
            if p.slot_id >= new_num_slots:
                overflow_slots.append((i, p))
        overflow_persistent_pages = [p for p in persistent_pages if p.slot_id >= new_num_slots]
        num_overflow_persistent = len(overflow_persistent_pages)
        if num_overflow_persistent > new_num_slots:
            raise OutOfPagesError("Not enough slots to hold all persistent pages")
        allocator = pool_group._slot_allocator
        # prevent allocating slots with id >= new_num_slots
        allocator.prepare_for_shrink(new_num_slots)
        min_num_evicted = 0
        # Need this because evicted overflow pages won't become free, because only free
        # non-overflow slots can be used for defragmentation.
        num_evicted_overflow_slots = 0
        while overflow_slots and len(overflow_slots) + num_overflow_persistent > min(
            new_num_slots,
            overflow_slots[0][0] + allocator.num_free_slots - num_evicted_overflow_slots,
        ):
            min_num_evicted = overflow_slots.popleft()[0] + 1
            num_evicted_overflow_slots += 1
        self.force_evict(
            level, make_typed(lambda i: min_num_evicted if i == pg_idx else 0, self.num_pool_groups)
        )
        # These are the pages that will remain in the cache level and require defragmentation.
        overflow_pages = [s[1] for s in overflow_slots] + overflow_persistent_pages
        requirements = filled_list(0, self.num_pool_groups)
        requirements[pg_idx] = len(overflow_pages)
        self.prepare_free_slots(level, requirements)
        assert NDEBUG or all(p.cache_level == level for p in overflow_pages), (
            "Some pages are not overflowed"
        )
        self._batched_migrate(pg_idx, level, level, overflow_pages, update_src=True, defrag=True)
        assert (
            len(allocator._overflow_slots)
            == allocator._num_active_slots - allocator._target_capacity
        )
        allocator.finish_shrink()
        pool_group.resize_pools(new_num_slots)

    def expand_pool_group(
        self, level: CacheLevel, pg_idx: PoolGroupIndex, new_num_slots: int
    ) -> None:
        pool_group = self._levels[level].storage._pool_groups[pg_idx]
        assert new_num_slots > pool_group.num_slots
        pool_group.resize_pools(new_num_slots)
        pool_group._slot_allocator.expand(new_num_slots)

    def adjust_cache_level(
        self,
        level: CacheLevel,
        new_quota: int | None,
        new_ratio_list: TypedIndexList[PoolGroupIndex, float],
        persistent_pages: TypedIndexList[PoolGroupIndex, list[Page]] | None = None,
    ) -> None:
        """Adapt the cache level by adjusting the ratio list. Persistent pages are those held and not evictable."""
        num_cache_levels = self.num_cache_levels
        lvl_storage = self._levels[level].storage
        old_num_slots = lvl_storage.slot_count_list
        new_quota = (
            lvl_storage.total_quota
            if new_quota is None
            else round_up(new_quota, lvl_storage.pool_size_granularity)
        )
        min_quota = self._min_quota_for_level(
            lvl_storage.slot_size_lists, lvl_storage.pool_size_granularity
        )
        if new_quota < min_quota:
            raise ValueError(
                f"Quota {new_quota} is insufficient for min_slots constraints "
                f"(requires at least {min_quota})"
            )
        new_num_slots = lvl_storage.compute_slot_count_list(
            new_ratio_list, self._min_slots, new_quota
        )
        if level != num_cache_levels - 1:
            assert persistent_pages is None, (
                "Persistent pages should be None for non-last level cache"
            )
        # shrink first
        for pg_idx in typed_range(self.num_pool_groups):
            if new_num_slots[pg_idx] >= old_num_slots[pg_idx]:
                continue
            pages = persistent_pages[pg_idx] if persistent_pages is not None else []
            self.shrink_pool_group(level, pg_idx, new_num_slots[pg_idx], pages)
        # then expand
        for pg_idx in typed_range(self.num_pool_groups):
            if new_num_slots[pg_idx] <= old_num_slots[pg_idx]:
                continue
            self.expand_pool_group(level, pg_idx, new_num_slots[pg_idx])
        lvl_storage.post_resize()

    def ratio_from_length(
        self, tokens_per_block: int, history_length: int, capacity: int
    ) -> TypedIndexList[PoolGroupIndex, float]:
        if capacity < history_length:
            warnings.warn("Bad sampling for capacity and history_length")
            capacity = history_length
        num_blocks = div_up(capacity, tokens_per_block)
        num_bytes = filled_list(0.0, self.num_pool_groups)
        ssm_lc_idx = self._life_cycles.ssm_life_cycle_id
        for lc_idx, lc in typed_enumerate(self._life_cycles.get()):
            pg_idx = self.get_pool_group_index(lc_idx)
            slot_size = self.slot_size(pg_idx)
            num_required_blocks: int
            if lc_idx == ssm_lc_idx:
                num_required_blocks = 1
            else:
                stale = lc.get_stale_range(history_length, tokens_per_block)
                num_required_blocks = max(num_blocks - len(stale), 1)
            num_bytes[pg_idx] += num_required_blocks * sum(slot_size)
        total = sum(num_bytes)
        assert total > 0
        return typed_map(num_bytes, lambda x: x / total)

    def ratio_from_batch(
        self,
        batch: BatchDesc,
        tokens_per_block: int,
        enable_swa_scratch_reuse: bool,
        granularity: int,
    ) -> TypedIndexList[PoolGroupIndex, float]:
        """Compute the ratio of bytes needed per pool group for a batch described by a BatchDesc."""
        num_slots = self._compute_slots_for_batch(batch, tokens_per_block, enable_swa_scratch_reuse)
        num_bytes = self._slots_to_bytes(num_slots, granularity)
        total = sum(num_bytes)
        assert total > 0
        return typed_map(num_bytes, lambda x: x / total)

    def _compute_min_slots_from_constraints(
        self, constraints: list[BatchDesc], tokens_per_block: int, enable_swa_scratch_reuse: bool
    ) -> TypedIndexList[PoolGroupIndex, int]:
        """Compute the minimum slots per pool group across all constraints (element-wise max).

        Always returns at least 1 slot per life cycle in each pool group.
        """
        # Default floor: 1 slot per life cycle in each pool group.
        max_slots = filled_list(0, self.num_pool_groups)
        for pg_idx in self._life_cycle_grouping:
            max_slots[pg_idx] += 1
        for batch in constraints:
            slots = self._compute_slots_for_batch(batch, tokens_per_block, enable_swa_scratch_reuse)
            for pg_idx in typed_range(self.num_pool_groups):
                max_slots[pg_idx] = max(max_slots[pg_idx], slots[pg_idx])
        return max_slots

    def _compute_slots_for_batch(
        self, batch: BatchDesc, tokens_per_block: int, enable_swa_scratch_reuse: bool
    ) -> TypedIndexList[PoolGroupIndex, int]:
        """Compute the minimum number of slots per pool group to support a BatchDesc."""
        num_slots = filled_list(0, self.num_pool_groups)
        ssm_lc_idx = self._life_cycles.ssm_life_cycle_id
        sys_blocks = batch.system_prompt_length // tokens_per_block
        for lc_idx, lc in typed_enumerate(self._life_cycles.get()):
            pg_idx = self.get_pool_group_index(lc_idx)
            if lc_idx == ssm_lc_idx:
                # SSM: always 1 dedicated block per request, never shared.
                num_slots[pg_idx] += len(batch.kv_caches)
                continue
            # Shared sys blocks (counted once): union of non-stale sys blocks across all requests.
            # A sys block needs memory if it's non-stale for ANY request.
            # = sys_blocks - (blocks stale for ALL requests within [0, sys_blocks))
            sys_range = HalfOpenRange(BlockOrdinal(0), BlockOrdinal(sys_blocks))
            # Intersection of per-request stale ranges, clamped to sys_range.
            stale_intersection = sys_range
            for kv in batch.kv_caches:
                stale = lc.get_stale_range(kv.history_length, tokens_per_block)
                stale_intersection = intersect(stale_intersection, stale)
            num_slots[pg_idx] += sys_blocks - len(stale_intersection)
            # Per-request unique blocks (excluding shared sys blocks already counted above).
            for kv in batch.kv_caches:
                total_blocks = div_up(kv.capacity, tokens_per_block)
                stale = lc.get_stale_range(kv.history_length, tokens_per_block)
                non_stale = total_blocks - len(stale)
                # Non-stale sys blocks for this request.
                non_stale_sys = sys_blocks - len(intersect(stale, sys_range))
                unique_non_stale = max(0, non_stale - non_stale_sys)
                if enable_swa_scratch_reuse:
                    scratch = compute_scratch_range(
                        lc, kv.history_length, kv.capacity, tokens_per_block
                    )
                    # Scratch blocks are always input blocks, so they never
                    # overlap with shared sys blocks (which are history).
                    num_scratch = len(scratch)
                    frac_max = self._slot_util_frac_max[lc_idx]
                    num_slots[pg_idx] += (unique_non_stale - num_scratch) + math.ceil(
                        num_scratch * frac_max
                    )
                else:
                    num_slots[pg_idx] += unique_non_stale
        return num_slots

    def _slots_to_bytes(
        self, num_slots: TypedIndexList[PoolGroupIndex, int], granularity: int
    ) -> TypedIndexList[PoolGroupIndex, int]:
        """Convert slot counts to bytes, rounding up each pool to granularity."""
        num_bytes = filled_list(0, self.num_pool_groups)
        for pg_idx in typed_range(self.num_pool_groups):
            for pool_size in self.slot_size(pg_idx):
                num_bytes[pg_idx] += round_up(num_slots[pg_idx] * pool_size, granularity)
        return num_bytes

    def _min_quota_for_level(
        self,
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        granularity: int,
    ) -> int:
        """Minimum quota (in bytes) required to satisfy _min_slots constraints."""
        return sum(
            round_up(ms * s, granularity)
            for ms, sizes in zip(self._min_slots, slot_size_lists)
            for s in sizes
        )

    def _compute_slot_count_for_level(
        self,
        tier_config: CacheTierConfig,
        slot_size_lists: TypedIndexList[PoolGroupIndex, TypedIndexList[PoolIndex, int]],
        ratio: TypedIndexList[PoolGroupIndex, float],
    ) -> TypedIndexList[PoolGroupIndex, int]:
        """Compute slot counts for a cache level from its tier config and ratio.

        Applies min_slots constraints (always at least 1 per life cycle).
        """
        granularity = CacheLevelManager.cache_tier_granularity(tier_config.tier, tier_config.quota)
        quota = max(
            self._min_quota_for_level(slot_size_lists, granularity),
            round_up(tier_config.quota, granularity),
        )
        return CacheLevelStorage.ratio_to_slot_count_list(
            quota, slot_size_lists, ratio, granularity, self._min_slots
        )

    def constrain_ratio(
        self,
        ratio: TypedIndexList[PoolGroupIndex, float],
    ) -> TypedIndexList[PoolGroupIndex, float]:
        """Apply the stored min_slots constraint to a ratio list for GPU level.

        Converts ratio to slot counts (with min_slots floor),
        then converts back to a bytes-based ratio.
        """
        gpu_storage = self._levels[GPU_LEVEL].storage
        granularity = gpu_storage.pool_size_granularity
        slot_count_list = gpu_storage.compute_slot_count_list(ratio, self._min_slots)
        num_bytes = self._slots_to_bytes(slot_count_list, granularity)
        total = sum(num_bytes)
        assert total > 0
        return typed_map(num_bytes, lambda x: x / total)
