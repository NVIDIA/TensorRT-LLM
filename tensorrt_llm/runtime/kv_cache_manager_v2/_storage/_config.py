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
from .._life_cycle_registry import LayerGroupId, LifeCycle, LifeCycleId, LifeCycleRegistry
from .._storage._core import PoolGroupIndex, PoolIndex
from .._utils import HomoTuple, TypedIndexList, filled_list, get_uniform_attribute, is_sorted


class BufferId(NamedTuple):
    layer_id: LayerId
    role: DataRole


@dataclass(slots=True, frozen=True)
class CoalescedBuffer:
    """Each coalesced buffer has multiple buffers with the same size and life cycle."""

    single_buffer_size: int  # identical for all buffers in the same coalesced buffer
    buffer_ids: HomoTuple[BufferId]

    @property
    def size(self) -> int:
        return self.single_buffer_size * len(self.buffer_ids)

    @property
    def num_buffers(self) -> int:
        return len(self.buffer_ids)


@dataclass(slots=True, frozen=True)
class SlotDescVariant:
    """
    A group of coalesced buffers for the same life cycle. Each coalesced buffer goes to one different memory pool.
    These pools forms a pool group. Allocation / deallocation of buffers in these pools are mirrored.
    """

    life_cycle_id: LifeCycleId
    coalesced_buffers: HomoTuple[CoalescedBuffer]

    def __post_init__(self) -> None:
        assert is_sorted(self.coalesced_buffers, key=lambda s: s.size, reverse=True)

    @property
    def layer_group_id(self) -> LayerGroupId:
        return self.life_cycle_id

    @property
    def slot_size_list(self) -> HomoTuple[int]:
        return tuple(s.size for s in self.coalesced_buffers)


@dataclass(slots=True, frozen=True)
class SlotDesc:
    """
    A slot in a memory pool group can have different composition, if they have the same slot_size_list.
    """

    variants: HomoTuple[SlotDescVariant]

    @property
    def slot_size_list(self) -> HomoTuple[int]:
        return get_uniform_attribute(self.variants, lambda s: s.slot_size_list)


@dataclass(slots=True, frozen=True)
class BufferAttr:
    life_cycle_id: LifeCycleId
    pool_index: PoolIndex
    offset: int
    size: int


@dataclass(slots=True, frozen=True)
class StorageConfig:
    cache_tiers: HomoTuple[CacheTierConfig]
    slot_desc_list: TypedIndexList[PoolGroupIndex, SlotDesc]

    @property
    def num_life_cycles(self) -> LifeCycleId:
        return LifeCycleId(sum(len(pg.variants) for pg in self.slot_desc_list))

    def life_cycle_grouping(self) -> TypedIndexList[LifeCycleId, PoolGroupIndex]:
        ret = filled_list(PoolGroupIndex(-1), self.num_life_cycles)
        for pg_idx, pg in enumerate(self.slot_desc_list):
            pg_idx = PoolGroupIndex(pg_idx)
            for s in pg.variants:
                ret[s.life_cycle_id] = pg_idx
        return ret

    def buffer_attributes(self) -> dict[BufferId, BufferAttr]:
        ret = dict[BufferId, BufferAttr]()
        for pg in self.slot_desc_list:
            for slot in pg.variants:
                life_cycle_id = slot.life_cycle_id
                for pool, cb in enumerate(slot.coalesced_buffers):
                    offset = 0
                    for b in cb.buffer_ids:
                        ret[b] = BufferAttr(
                            life_cycle_id, PoolIndex(pool), offset, cb.single_buffer_size
                        )
                        offset += cb.single_buffer_size
        return ret

    def slot_to_page_indices(self) -> TypedIndexList[LifeCycleId, int]:
        ret = filled_list(0, self.num_life_cycles)
        for pg in self.slot_desc_list:
            for slot in pg.variants:
                life_cycle = slot.life_cycle_id
                page = slot.coalesced_buffers[0]
                ret[life_cycle] = page.num_buffers
        return ret

    def layer_to_life_cycle_ids(self) -> dict[LayerId, LifeCycleId]:
        map = dict[LayerId, LifeCycleId]()
        for (layer_id, _), attr in self.buffer_attributes().items():
            lc_id = map.setdefault(layer_id, attr.life_cycle_id)
            assert lc_id == attr.life_cycle_id
        return map

    def __post_init__(self) -> None:
        groups = [tuple(s.life_cycle_id for s in pg.variants) for pg in self.slot_desc_list]
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
    slot_groups: list[SlotDescVariant] = []
    for life_cycle_id, size_to_buffers in buffer_groups.items():
        assert len(set(len(buffer_ids) for buffer_ids in size_to_buffers.values())) == 1, (
            "Not yet supported. While we can support this easily, we need to know whether the kernels "
            "need to share page indices or not. We haven't seen such models, yet. So we leave this as a "
            "future work."
        )
        slots = [
            CoalescedBuffer(size, tuple(buffer_ids)) for size, buffer_ids in size_to_buffers.items()
        ]
        slots.sort(key=lambda p: p.size, reverse=True)
        slot_groups.append(SlotDescVariant(life_cycle_id, tuple(slots)))
    # Merge slot groups with the same slot_size_list
    pool_groups_by_slot_size_list = defaultdict[HomoTuple[int], list[SlotDescVariant]](
        list[SlotDescVariant]
    )
    for slot_group in slot_groups:
        pool_groups_by_slot_size_list[slot_group.slot_size_list].append(slot_group)
    slot_desc_list = cast(
        TypedIndexList[PoolGroupIndex, SlotDesc],
        [SlotDesc(tuple(slot_groups)) for slot_groups in pool_groups_by_slot_size_list.values()],
    )
    return StorageConfig(cache_tiers=tuple(config.cache_tiers), slot_desc_list=slot_desc_list)
