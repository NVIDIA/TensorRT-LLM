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
from dataclasses import dataclass
from typing import NamedTuple, cast

from .._common import LayerId
from .._config import CacheTierConfig, DataRole, KVCacheManagerConfig
from .._life_cycle_registry import LifeCycle, LifeCycleId, LifeCycleRegistry
from .._storage._core import PoolGroupIndex, PoolIndex
from .._utils import (
    HomoTuple,
    TypedIndexList,
    exact_div,
    filled_list,
    get_uniform_attribute,
    is_sorted,
    typed_range,
)


class BufferId(NamedTuple):
    layer_id: LayerId
    role: DataRole


@dataclass(slots=True)
class CoalescedBuffer:
    life_cycle_id: LifeCycleId
    single_buffer_size: int  # identical for all buffers in the same coalesced buffer
    buffer_ids: list[BufferId]

    @property
    def size(self) -> int:
        return self.single_buffer_size * len(self.buffer_ids)


@dataclass(slots=True)
class PageConfig:
    """
    A page is a group of coalesced buffers. Each coalesced buffer has multiple buffers with the same
    size. Multiple coalesced buffers can be in the same page if they share the same life cycle and
    coalesced size.
    """

    coalesced_buffers: list[CoalescedBuffer]

    @property
    def _coalesced_size(self) -> int:
        return get_uniform_attribute(self.coalesced_buffers, lambda b: b.size)

    @property
    def slot_size(self) -> int:
        return self._coalesced_size * len(self.coalesced_buffers)

    @property
    def life_cycle_id(self) -> LifeCycleId:
        return get_uniform_attribute(self.coalesced_buffers, lambda b: b.life_cycle_id)


@dataclass(slots=True, frozen=True)
class SlotConfig:
    "A group of pages for the same life cycle."

    pages: HomoTuple[PageConfig]

    def __post_init__(self) -> None:
        assert is_sorted(self.pages, key=lambda s: s.slot_size, reverse=True)
        assert all(
            len(p.coalesced_buffers) == len(self.pages[0].coalesced_buffers) for p in self.pages
        )

    @property
    def life_cycle_id(self) -> LifeCycleId:
        return get_uniform_attribute(self.pages, lambda s: s.life_cycle_id)

    @property
    def slot_size_list(self) -> HomoTuple[int]:
        return tuple(s.slot_size for s in self.pages)


@dataclass(slots=True, frozen=True)
class PoolGroupConfig:
    """
    A group of pools may contain slots (page groups) with different life cycles. They have identical
    slot size list, so we can put them in the same group of memory pools.
    """

    slots: HomoTuple[SlotConfig]

    @property
    def slot_size_list(self) -> HomoTuple[int]:
        return get_uniform_attribute(self.slots, lambda s: s.slot_size_list)


@dataclass(slots=True, frozen=True)
class BufferAttr:
    life_cycle_id: LifeCycleId
    pool_index: PoolIndex
    offset: int
    size: int


@dataclass(slots=True, frozen=True)
class StorageConfig:
    cache_tiers: HomoTuple[CacheTierConfig]
    pool_groups: HomoTuple[PoolGroupConfig]

    @property
    def num_life_cycles(self) -> LifeCycleId:
        return LifeCycleId(sum(len(pg.slots) for pg in self.pool_groups))

    def life_cycle_grouping(self) -> TypedIndexList[LifeCycleId, PoolGroupIndex]:
        ret = filled_list(PoolGroupIndex(-1), self.num_life_cycles)
        for pg_idx, pg in enumerate(self.pool_groups):
            pg_idx = PoolGroupIndex(pg_idx)
            for s in pg.slots:
                ret[s.life_cycle_id] = pg_idx
        return ret

    def buffer_attributes(self) -> dict[BufferId, BufferAttr]:
        ret = dict[BufferId, BufferAttr]()
        for pg in self.pool_groups:
            for slot in pg.slots:
                life_cycle_id = slot.life_cycle_id
                for pool, page in enumerate(slot.pages):
                    offset = 0
                    for cb in page.coalesced_buffers:
                        for b in cb.buffer_ids:
                            ret[b] = BufferAttr(
                                life_cycle_id, PoolIndex(pool), offset, cb.single_buffer_size
                            )
                            offset += cb.single_buffer_size
        return ret

    def slot_to_page_indices(self) -> TypedIndexList[LifeCycleId, int]:
        ret = filled_list(0, self.num_life_cycles)
        for pg in self.pool_groups:
            for slot in pg.slots:
                life_cycle = slot.life_cycle_id
                assert len(slot.pages) == 1
                page = slot.pages[0]
                assert len(page.coalesced_buffers) == 1
                scale = exact_div(page.slot_size, page.coalesced_buffers[0].single_buffer_size)
                ret[life_cycle] = scale
        return ret

    def layer_to_life_cycle_ids(self) -> TypedIndexList[LayerId, LifeCycleId]:
        map = dict[LayerId, LifeCycleId]()
        for (layer_id, _), attr in self.buffer_attributes().items():
            lc_id = map.setdefault(layer_id, attr.life_cycle_id)
            assert lc_id == attr.life_cycle_id
        assert len(map) == max(map.keys()) + 1
        return cast(
            TypedIndexList[LayerId, LifeCycleId],
            [map[LayerId(layer_id)] for layer_id in typed_range(len(map))],
        )

    def __post_init__(self) -> None:
        groups = [tuple(s.life_cycle_id for s in pg.slots) for pg in self.pool_groups]
        all_life_cycle_ids = sum((g for g in groups), ())
        assert len(all_life_cycle_ids) == len(set(all_life_cycle_ids))


def create_storage_config(config: KVCacheManagerConfig) -> StorageConfig:
    # group buffers first by life cycle, then by single buffer size.
    buffer_groups = defaultdict[LifeCycleId, defaultdict[int, list[BufferId]]](
        lambda: defaultdict[int, list[BufferId]](list[BufferId])
    )
    life_cycle_registry = LifeCycleRegistry(config)
    for layer in config.layers:
        life_cycle = LifeCycle.make(
            layer.window_size, layer.num_sink_tokens, config.tokens_per_block
        )
        life_cycle_id = life_cycle_registry.get_id(life_cycle)
        size_to_buffers = buffer_groups[life_cycle_id]
        for buffer in layer.buffers:
            size_to_buffers[buffer.size].append(BufferId(layer.layer_id, buffer.role))
    # Create one slot group for each life cycle.
    # It's possible that buffers with different sizes form coalesced buffers with the same coalesced size.
    # @TODO: add test for this case.
    slot_groups: list[SlotConfig] = []
    for life_cycle_id, size_to_buffers in buffer_groups.items():
        assert len(set(len(buffer_ids) for buffer_ids in size_to_buffers.values())) == 1, (
            "Not yet supported. While we can support this easily, we need to know whether the kernels "
            "need to share page indices or not. We haven't seen such models, yet. So we leave this as a "
            "future work."
        )
        size_to_coalesced_buffers = defaultdict[int, list[CoalescedBuffer]](list[CoalescedBuffer])
        for size, buffer_ids in size_to_buffers.items():
            coalesced_size = size * len(buffer_ids)
            coalesced_buffers = size_to_coalesced_buffers[coalesced_size]
            coalesced_buffers.append(
                CoalescedBuffer(
                    life_cycle_id=life_cycle_id, single_buffer_size=size, buffer_ids=buffer_ids
                )
            )
        slots = [
            PageConfig(coalesced_buffers)
            for coalesced_buffers in size_to_coalesced_buffers.values()
        ]
        slots.sort(key=lambda p: p.slot_size, reverse=True)
        slot_groups.append(SlotConfig(tuple(slots)))
    # Merge slot groups with the same slot_size_list
    pool_groups_by_slot_size_list = defaultdict[HomoTuple[int], list[SlotConfig]](list[SlotConfig])
    for slot_group in slot_groups:
        pool_groups_by_slot_size_list[slot_group.slot_size_list].append(slot_group)
    pool_groups = [
        PoolGroupConfig(tuple(slot_groups))
        for slot_groups in pool_groups_by_slot_size_list.values()
    ]
    return StorageConfig(cache_tiers=tuple(config.cache_tiers), pool_groups=tuple(pool_groups))
