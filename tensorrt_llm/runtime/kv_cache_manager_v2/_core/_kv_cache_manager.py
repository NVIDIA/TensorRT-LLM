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

import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, cast

from .. import rawref
from .._block_radix_tree import BlockRadixTree
from .._common import (
    BAD_PAGE_INDEX,
    GPU_LEVEL,
    PRIORITY_DEFAULT,
    BlockOrdinal,
    CacheLevel,
    CacheTier,
    LayerId,
    MemAddress,
    PageStatus,
    Priority,
    TokenIdExt,
)
from .._config import DataRole, KVCacheManagerConfig
from .._life_cycle_registry import LayerGroupId, LifeCycle, LifeCycleId, LifeCycleRegistry
from .._page import Page, _PageHolder
from .._storage._config import BufferId, create_storage_config
from .._storage._core import PoolGroupIndex, PoolIndex, SlotId
from .._storage_manager import StorageManager
from .._utils import (
    HomoTuple,
    TypedIndexList,
    div_up,
    exact_div,
    filled_list,
    init_cuda_once,
    make_typed,
    typed_enumerate,
    typed_map,
    typed_range,
    unwrap_rawref,
)
from ._kv_cache import _KVCache
from ._moving_average import MovingAverage


@dataclass(slots=True, frozen=True)
class MemoryPoolDesc:
    base: MemAddress
    page_size: int


@dataclass(slots=True, frozen=True)
class MemoryPoolGroupDesc:
    num_pages: int
    pools: TypedIndexList[PoolIndex, MemoryPoolDesc]


@dataclass(slots=True, frozen=True)
class Range:
    start: int
    end: int

    def __add__(self, offset: int) -> "Range":
        return Range(self.start + offset, self.end + offset)

    def __radd__(self, offset: int) -> "Range":
        return self + offset


@dataclass(slots=True, frozen=True)
class ExpandedBuffer:
    id: BufferId
    expansion: int  # expansion factor of page due to heterogeneous tokens_per_block


@dataclass(slots=True, frozen=True)
class AggregatedPageDesc:
    """
    The data you need would be in the following byte ranges:
        (base + stride * i + Range(0, size) for i in aggregated_page_indices)
    """

    base: MemAddress
    size: int
    stride: int
    layer_group_id: LayerGroupId
    buffers: Sequence[ExpandedBuffer]


@dataclass(slots=True, frozen=True)
class PageIndexConverter:
    scale: int
    expansion: int

    def __call__(self, base_index: int) -> Iterator[int]:
        """
        Convert from base page indices to page indices expected by operators/kernels.
        This is just an reference implementation. Users are encouraged to do it with a CUDA kernel.
        """
        valid = base_index != BAD_PAGE_INDEX
        index = base_index * self.scale
        expansion = self.expansion
        return ((index * expansion + i if valid else BAD_PAGE_INDEX) for i in range(expansion))


class KVCacheManager:
    __slots__ = (
        "_init_config",
        "_life_cycles",
        "_radix_tree",
        "_storage",
        "_living_kv_caches",
        "_avg_reused_length",
        "_avg_sqr_capacity",
        "_avg_sqr_history_length",
        "_target_ratio_list_gpu",
        "_target_ratio_list_other",
        "_num_created_kv_caches",
        "_num_closed_kv_caches",
        "_last_adjustment_time",
        "_last_update_num_closed_requests",
    )
    _init_config: KVCacheManagerConfig
    _life_cycles: LifeCycleRegistry
    _radix_tree: BlockRadixTree
    _storage: StorageManager
    _living_kv_caches: set[rawref.ref[_KVCache]]
    # Eventually we should let the eviction controller evict associated pages together, i.e.
    # when a page eviction makes other pages in the same cache level useless, it should also
    # evict those pages. When we have that, we can simply decide capacity ratio based on
    # memory pool utilization. For now, we use a simpler approach based on sequence length.
    # But this ignores the fact that some pages are shared among multiple sequences.
    _avg_reused_length: MovingAverage
    # use squared because longer requests also lives for longer but we only update this on
    # request closing, use squared average to compensate that.
    _avg_sqr_capacity: MovingAverage
    _avg_sqr_history_length: MovingAverage
    _target_ratio_list_gpu: TypedIndexList[PoolGroupIndex, float]
    _target_ratio_list_other: TypedIndexList[PoolGroupIndex, float]
    _num_created_kv_caches: int
    _num_closed_kv_caches: int
    _last_adjustment_time: float
    _last_update_num_closed_requests: int

    def __init__(self, config: KVCacheManagerConfig) -> None:
        init_cuda_once()
        config = deepcopy(config)
        self._init_config = config
        self._life_cycles = LifeCycleRegistry(config)
        self._radix_tree = BlockRadixTree(self._life_cycles, config.tokens_per_block)
        storage_config = create_storage_config(config)
        self._storage = StorageManager(self._life_cycles, storage_config)
        self._living_kv_caches = set[rawref.ref[_KVCache]]()
        decay = 0.9999
        self._avg_reused_length = MovingAverage(decay)
        self._avg_sqr_capacity = MovingAverage(decay)
        self._avg_sqr_history_length = MovingAverage(decay)
        self._target_ratio_list_gpu = self._current_gpu_ratio
        self._target_ratio_list_other = self._current_other_ratios
        self._num_created_kv_caches = 0
        self._num_closed_kv_caches = 0
        self._last_adjustment_time = time.monotonic()
        self._last_update_num_closed_requests = 0

    def __del__(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        self.clear_reusable_blocks()
        self._storage.destroy()

    def clear_reusable_blocks(self) -> None:
        for ref in self._radix_tree.clear():
            assert unwrap_rawref(ref).status == PageStatus.DROPPABLE
            self._storage.exclude_from_eviction(unwrap_rawref(ref))
        for level in self._storage._levels:
            for pg_idx in typed_range(level.storage.num_pool_groups):
                assert level.controller.num_evictable_pages(pg_idx) == 0

    def get_mem_pool_base_address(self, layer_id: LayerId, data_role: DataRole) -> MemAddress:
        """
        Get the base address of the memory pool holding pages for the given layer and data role.
        It's guaranteed that for one layer, multiple buffers of the same size have the same base address.
        """
        return self._storage.get_mem_pool_base_address(layer_id, data_role)

    # Currently always equals to page size. In the future, that will change when kernels support page stride.
    def get_page_stride(self, layer_id: LayerId, data_role: DataRole) -> int:
        attr = self._storage.get_buffer_attr(layer_id, data_role)
        return exact_div(attr.size, attr.expansion)

    def get_page_index_upper_bound(self, layer_id: LayerId, data_role: DataRole) -> int:
        """
        The upper bound of page indices for the given layer and data role.
        Note that this is not the same as the max number of pages available for this layer and data role.
        Internally, multiple buffers may share one memory pool. The purpose of this API is just in case
        users want to wrap the memory pool as a tensor with known shape.
        """
        storage = self._storage
        lc_id = storage._layer_to_life_cycle_ids[layer_id]
        pg_idx = storage.get_pool_group_index(lc_id)
        pool_group = storage._levels[GPU_LEVEL].storage._pool_groups[pg_idx]
        num_slots = pool_group.num_slots
        attr = storage.get_buffer_attr(layer_id, data_role)
        pool_idx = attr.pool_index
        slot_size = pool_group.slot_size[pool_idx]
        return (
            exact_div(slot_size, attr.size) * num_slots - exact_div(attr.offset, attr.size)
        ) * attr.expansion

    def get_page_index_scale(self, layer_id: LayerId, data_role: DataRole) -> int:
        """
        Deprecated. Use get_page_index_converter instead.

        The multiplier to convert from base page indices to page indices expected by operators/kernels.

        For layers in the same layer group, users are encouraged to share the computed page indices
        between buffers of these layers, if the page index scale for these buffers are the same.
        """
        storage = self._storage
        attr = storage.get_buffer_attr(layer_id, data_role)
        return storage._slot_to_page_indices[attr.life_cycle_id][attr.pool_index]

    def get_page_index_converter(
        self, layer_id: LayerId, data_role: DataRole
    ) -> PageIndexConverter:
        """
        Get the converter to convert from base page indices to page indices expected by operators/kernels.

        For layers in the same layer group and with the same tokens_per_block, users are encouraged to
        share the computed page indices between buffers of these layers, if the page index scale for these
        buffers are the same.
        """
        storage = self._storage
        attr = storage.get_buffer_attr(layer_id, data_role)
        scale = storage._slot_to_page_indices[attr.life_cycle_id][attr.pool_index]
        return PageIndexConverter(scale, attr.expansion)

    def create_kv_cache(
        self,
        lora_task_id: int | None = None,
        input_tokens: Sequence[TokenIdExt] | None = None,
        id: Any = None,
        custom_priority_callback: Callable[[BlockOrdinal, LifeCycle], Priority] = lambda _,
        __: PRIORITY_DEFAULT,
    ) -> _KVCache:
        """
        lora_task_id: match lora_task_id before matching any tokens.
        custom_priority_callback: takes block index and layer sliding window size, returns priority.
        If priority returned is higher than existing priority for reused blocks, the block priority is updated.
        Newly created KV cache is suspended. You need to call resume() with a cuda stream to make it active
        & ready in that stream.
        Returns None if suspended=False and we don't have enough resource.
        This call will attempt to reuse KV cache blocks.
        It's user responsibility to remove the last token from prompts if we need to re-compute the token
        generated by prefill.
        """
        return _KVCache(self, lora_task_id, input_tokens, id, custom_priority_callback)

    def resize(self, cache_level: CacheLevel, quota: int, best_efforts: bool = False) -> bool:
        """
        When calling resize, all KV caches must be suspended.
        If best_efforts is True, we will try to resize the quota to the largest possible value that is
        still <= quota, and returns False only when we cannot resize the quota at all.
        If best_efforts is False, we will resize the quota to the exact value of quota, and give up
        if not possible.
        For now, best_efforts=True is not yet implemented.
        """
        if best_efforts:
            raise NotImplementedError("Not implemented")
        else:
            try:
                self._adjust_level(cache_level, quota)
                return True
            except Exception as e:
                print(f"Failed to resize cache level {cache_level} to {quota}: {e}")
                return False

    def get_quota(self, cache_level: CacheLevel) -> int:
        return self._storage._levels[cache_level].storage.total_quota

    # sorted by CacheLevel from warm to cold
    @property
    def cache_tier_list(self) -> HomoTuple[CacheTier]:
        return self._storage.cache_tiers

    @property
    def tokens_per_block(self) -> int:
        return self._radix_tree.tokens_per_block

    @property
    def allow_seq_rebasing(self) -> bool:
        """
        If True, when we commit a full block, we will try to find a existing reusable block with the
        same tokens and reuse that block instead to save some memory. Intra-batch reuse will be enabled
        if this is True.
        """
        return True

    @property
    def enable_partial_match(self) -> bool:
        return self._init_config.enable_partial_reuse

    @property
    def num_layers(self) -> int:
        return len(self._storage._layer_to_life_cycle_ids)

    @property
    def layer_ids(self) -> Iterator[LayerId]:
        return iter(self._storage._layer_to_life_cycle_ids.keys())

    def get_layer_group_id(self, layer_id: LayerId) -> LayerGroupId:
        return self._storage._layer_to_life_cycle_ids[layer_id]

    @property
    def layer_grouping(self) -> HomoTuple[HomoTuple[LayerId]]:
        """
        Layers are divided into multiple groups.
        Buffers in the same layer group for the same token block are always allocated/deallocated together.
        """
        layer_to_life_cycle_ids = self._storage._layer_to_life_cycle_ids
        num_life_cycles = self._life_cycles.size
        grouping = dict[LifeCycleId, list[LayerId]]({i: [] for i in typed_range(num_life_cycles)})
        for layer_id, life_cycle_id in layer_to_life_cycle_ids.items():
            grouping[life_cycle_id].append(layer_id)
        return tuple(tuple(grouping[i]) for i in typed_range(num_life_cycles))

    @property
    def all_buffer_ids(self) -> Iterator[BufferId]:
        return iter(self._storage._buffer_attr.keys())

    def get_aggregated_pages(self, buffers: Iterable[BufferId]) -> Iterator[AggregatedPageDesc]:
        """
        Internally, we concatenate buffers into larger buffers.
        This API takes a iterable of buffers (unordered), and try to find those that can form
        contiguous aggregated buffers.
        When we need data transfer, this helps us improve performance.
        Args:
            buffers: iterable of buffers to aggregate. Order does not matter.
        Returns:
            A iterator of aggregated buffers.
        """
        # Group by (life_cycle, pool_index)
        groups = defaultdict[tuple[LifeCycleId, PoolIndex], list[tuple[Range, ExpandedBuffer]]](
            list[tuple[Range, ExpandedBuffer]]
        )
        buffer_attr_map = self._storage._buffer_attr
        for b in buffers:
            attr = buffer_attr_map[b]
            size = attr.size
            start = attr.offset
            key = (attr.life_cycle_id, attr.pool_index)
            groups[key].append((Range(start, start + size), ExpandedBuffer(b, attr.expansion)))

        storage = self._storage._levels[GPU_LEVEL].storage
        lc2pg = self._storage._life_cycle_grouping
        for (lc, pool_idx), group in groups.items():
            pg_idx = lc2pg[lc]
            # Sort by start offset
            group.sort(key=lambda x: x[0].start)
            # Merge contiguous
            current_start, current_end, current_buffers = (
                group[0][0].start,
                group[0][0].end,
                [group[0][1]],
            )
            # cache stride and pool_base for this group
            stride = storage.slot_size(pg_idx)[pool_idx]
            pool_base = int(cast(int, storage.slot_address(pg_idx, pool_idx, SlotId(0))))
            for i in range(1, len(group)):
                next_range, next_buf = group[i]
                if next_range.start == current_end:
                    current_end = next_range.end
                    current_buffers.append(next_buf)
                else:
                    base = MemAddress(pool_base + current_start)
                    yield AggregatedPageDesc(
                        base, current_end - current_start, stride, lc, tuple(current_buffers)
                    )
                    current_start, current_end, current_buffers = (
                        next_range.start,
                        next_range.end,
                        [next_buf],
                    )
            # Flush last
            base = MemAddress(pool_base + current_start)
            yield AggregatedPageDesc(
                base, current_end - current_start, stride, lc, tuple(current_buffers)
            )

    @property
    def _current_gpu_ratio(self) -> TypedIndexList[PoolGroupIndex, float]:
        return self._storage.get_ratio_list(GPU_LEVEL)

    @property
    def _current_other_ratios(self) -> TypedIndexList[PoolGroupIndex, float]:
        storage = self._storage
        num_cache_levels = storage.num_cache_levels
        if num_cache_levels == 1:
            return self._current_gpu_ratio
        num_pool_groups = storage.num_pool_groups
        other_ratios = [
            storage.get_ratio_list(i) for i in typed_range(CacheLevel(1), num_cache_levels)
        ]
        other_ratio = filled_list(0.0, num_pool_groups)
        for j in typed_range(num_pool_groups):
            for i in range(1, num_cache_levels):
                other_ratio[j] += other_ratios[i - 1][j]
            other_ratio[j] /= num_cache_levels - 1
        return other_ratio

    def _get_target_ratio_list(self, level: CacheLevel) -> TypedIndexList[PoolGroupIndex, float]:
        return self._target_ratio_list_gpu if level == GPU_LEVEL else self._target_ratio_list_other

    def _need_adjustment(self, level: CacheLevel) -> bool:
        def check_mismatch(
            a: TypedIndexList[PoolGroupIndex, float],
            b: TypedIndexList[PoolGroupIndex, float],
            thres: float,
        ) -> bool:
            return any(not (1 / thres < x / y < thres) for x, y in zip(a, b))

        if level == GPU_LEVEL:
            return check_mismatch(self._target_ratio_list_gpu, self._current_gpu_ratio, 1.25)
        else:
            return check_mismatch(self._target_ratio_list_other, self._current_other_ratios, 1.25)

    def _adjust_level(self, level: CacheLevel, new_quota: int | None = None) -> None:
        new_ratio_list = self._get_target_ratio_list(level)
        storage = self._storage
        num_cache_levels = storage.num_cache_levels
        # held and not evictable as they are already in the last level cache.
        persistent_pages: TypedIndexList[PoolGroupIndex, list[Page]] | None = None
        if level == num_cache_levels - 1:
            persistent_pages = self._gather_persistent_pages()
        storage.adjust_cache_level(level, new_quota, new_ratio_list, persistent_pages)

    def _gather_persistent_pages(self) -> TypedIndexList[PoolGroupIndex, list[Page]]:
        last_level = self._storage.num_cache_levels - 1
        lc2pg = self._storage._life_cycle_grouping
        ret = make_typed(lambda _: list[Page](), self._storage.num_pool_groups)
        for r in self._living_kv_caches:
            kv_cache = unwrap_rawref(r)
            assert kv_cache.status == _KVCache.Status.SUSPENDED
            for block in kv_cache._blocks:
                for beam in block.pages:
                    for lc, holder in typed_enumerate(beam):
                        if holder is None:
                            continue
                        assert type(holder) is _PageHolder
                        page = holder.page
                        assert page.status == PageStatus.HELD
                        assert page.scheduled_for_eviction == (page.cache_level != last_level)
                        if not page.scheduled_for_eviction:
                            ret[lc2pg[lc]].append(holder.page)
        return ret

    @property
    def need_adjustment(self) -> bool:
        if self._num_closed_kv_caches < 2000:
            return False
        if time.monotonic() - self._last_adjustment_time < 120:
            return False
        return self._need_adjustment(GPU_LEVEL) or self._need_adjustment(
            CacheLevel(self._storage.num_cache_levels - 1)
        )

    def adjust(self) -> None:
        """
        Adjust the cache level and ratio list.
        This function should be called periodically to ensure the cache level and ratio list are
        adjusted to the optimal values. All KV caches must be suspended before calling this function.
        """
        assert all(
            unwrap_rawref(c).status == _KVCache.Status.SUSPENDED for c in self._living_kv_caches
        )
        storage = self._storage
        for level in typed_range(storage.num_cache_levels):
            if self._need_adjustment(level):
                self._adjust_level(level)
        self._last_adjustment_time = time.monotonic()

    def _try_update_target_ratios(self) -> None:
        if self._num_closed_kv_caches - self._last_update_num_closed_requests < 100:
            return
        self._last_update_num_closed_requests = self._num_closed_kv_caches
        tokens_per_blocks = self.tokens_per_block
        life_cycles = self._life_cycles.get()
        num_pool_groups = self._storage.num_pool_groups
        storage = self._storage
        lc2pg = storage._life_cycle_grouping

        def ratio_from_length(
            history_length: int, capacity: int
        ) -> TypedIndexList[PoolGroupIndex, float]:
            num_blocks = div_up(capacity, tokens_per_blocks)
            num_bytes = filled_list(0.0, num_pool_groups)
            for lc_idx, lc in typed_enumerate(life_cycles):
                stale_beg, stale_end = _KVCache._get_stale_range(
                    tokens_per_blocks, history_length, lc
                )
                pg_idx = lc2pg[lc_idx]
                slot_size = storage.slot_size(pg_idx)
                num_bytes[pg_idx] += (num_blocks - (stale_end - stale_beg)) * sum(slot_size)
            total = sum(num_bytes)
            assert total > 0
            return typed_map(num_bytes, lambda x: x / total)

        avg_reused_length: int = round(self._avg_reused_length.value)
        avg_capacity: int = round(self._avg_sqr_capacity.value**0.5)
        avg_history_length: int = round(self._avg_sqr_history_length.value**0.5)
        if avg_capacity > 0:
            self._target_ratio_list_gpu = ratio_from_length(avg_history_length, avg_capacity)
        if avg_reused_length > 0:
            self._target_ratio_list_other = ratio_from_length(avg_reused_length, avg_reused_length)

    # @TODO: need updating when dynamic resizing is supported.
    def clamp_max_seq_len_for_mem(self, batch_size: int, token_num_upper_bound: int) -> int:
        "Get the max possible sequence length limited by the GPU memory pools."
        assert batch_size > 0
        tokens_per_block = self.tokens_per_block
        life_cycles = self._life_cycles
        storage = self._storage
        num_pool_groups = storage.num_pool_groups
        remaining_slots = cast(
            TypedIndexList[PoolGroupIndex, int],
            [storage.num_slots(pg) for pg in typed_range(num_pool_groups)],
        )
        lc_to_pg_idx = storage._life_cycle_grouping

        def get_num_slots(seq_len: int) -> TypedIndexList[PoolGroupIndex, int]:
            ret = filled_list(0, num_pool_groups)
            for lc_id, lc in life_cycles.items():
                stale_range = _KVCache._get_stale_range(tokens_per_block, seq_len, lc)
                num_stale_blocks = stale_range[1] - stale_range[0]
                num_slots = div_up(seq_len, tokens_per_block) - num_stale_blocks
                pg_idx = lc_to_pg_idx[lc_id]
                ret[pg_idx] += num_slots
            return ret

        for pg in typed_range(num_pool_groups):
            remaining_slots[pg] -= get_num_slots(1)[pg] * (batch_size - 1)
            assert remaining_slots[pg] >= 0

        def is_enough(num_blocks: int) -> bool:
            return all(
                cnt <= rem
                for cnt, rem in zip(get_num_slots(num_blocks * tokens_per_block), remaining_slots)
            )

        assert is_enough(1)
        lb = 1
        ub = div_up(token_num_upper_bound, tokens_per_block)
        if is_enough(ub):
            return token_num_upper_bound
        while lb < ub - 1:
            mid = (lb + ub) // 2
            if is_enough(mid):
                lb = mid
            else:
                ub = mid
        return min(lb * tokens_per_block, token_num_upper_bound)
