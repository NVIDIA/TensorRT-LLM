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

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, cast

from .._block_radix_tree import BlockRadixTree
from .._common import (
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
    typed_range,
    unwrap_rawref,
)
from ._kv_cache import _KVCache


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
class BufferSlice:
    buffer_id: BufferId
    num_slices: int = 1
    slice_index: int = 1

    def __post_init__(self) -> None:
        assert 0 <= self.slice_index < self.num_slices


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
    buffers: Sequence[BufferSlice]


class KVCacheManager:
    __slots__ = ("_init_config", "_life_cycles", "_radix_tree", "_storage")
    _init_config: KVCacheManagerConfig
    _life_cycles: LifeCycleRegistry
    _radix_tree: BlockRadixTree
    _storage: StorageManager

    def __init__(self, config: KVCacheManagerConfig) -> None:
        init_cuda_once()
        self._init_config = config
        self._life_cycles = LifeCycleRegistry(config)
        self._radix_tree = BlockRadixTree(self._life_cycles, config.tokens_per_block)
        storage_config = create_storage_config(config)
        self._storage = StorageManager(self._life_cycles, storage_config)

    def __del__(self) -> None:
        self.clear_reusable_blocks()

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
        return attr.size

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
        return exact_div(slot_size, attr.size) * num_slots - exact_div(attr.offset, attr.size)

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
        If best_efforts is True, we will try to resize the quota to the largest possible value that is
        still <= quota, and returns False only when we cannot resize the quota at all.
        If best_efforts is False, we will resize the quota to the exact value of quota, and give up
        if not possible.
        """
        raise NotImplementedError("Not implemented")

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
        return True

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

    def get_aggregated_pages(self, buffers: Iterable[BufferSlice]) -> Iterator[AggregatedPageDesc]:
        """
        Internally, we concatenate buffers into larger buffers.
        This API takes a iterator of buffers (unordered), and try to find those that can form
        contiguous aggregated buffers.
        When we need data transfer, this helps us improve performance.
        Args:
            buffers: iterable of buffers to aggregate. Order does not matter.
        Returns:
            A iterator of aggregated buffers.
        """
        # Group by (life_cycle, pool_index)
        groups = defaultdict[tuple[LifeCycleId, PoolIndex], list[tuple[Range, BufferSlice]]](
            list[tuple[Range, BufferSlice]]
        )
        buffer_attr_map = self._storage._buffer_attr
        for b in buffers:
            attr = buffer_attr_map[b.buffer_id]
            slice_size = exact_div(attr.size, b.num_slices)
            start = attr.offset + slice_size * b.slice_index
            key = (attr.life_cycle_id, attr.pool_index)
            groups[key].append((Range(start, start + slice_size), b))

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

    # @TODO: need updating when dynamic resizing is supported.
    def clamp_max_seq_len_for_mem(self, batch_size: int) -> int:
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
        ub = lb
        while is_enough(ub):
            lb = ub
            ub *= 2
        while lb < ub - 1:
            mid = (lb + ub) // 2
            if is_enough(mid):
                lb = mid
            else:
                ub = mid
        return lb * tokens_per_block
