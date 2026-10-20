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
"""What a ``KVCacheManagerV2`` holds, read once for every spec built on it.

``life_cycles`` walks the manager's pool groups: each slot-desc variant is one life cycle, whose
coalesced buffers place every ``(layer, role)`` buffer at an offset inside a slot of one physical
pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

from ..page_table import Pool


@dataclass(frozen=True)
class SlotBuffer:
    """One ``(layer, role)`` buffer of a life cycle: its pool and byte range inside a slot."""

    pool: int
    local_layer: int
    role: str
    offset: int
    size: int


@dataclass
class LifeCycle:
    index: int
    internal_layer_ids: List[int]
    buffers: List[SlotBuffer]


def life_cycles(manager) -> Tuple[List[Pool], List[LifeCycle]]:
    """The manager's pools (flattened over pool groups) and its life cycles by id.

    Life cycles sharing a pool group reuse its slot memory; each one's buffer offsets start at 0
    independently.
    """
    pools: List[Pool] = []
    cycles: List[Optional[LifeCycle]] = [None] * len(manager.impl.layer_grouping)
    for pg_desc in manager.impl.pool_group_descs:
        first_pool = len(pools)
        pools.extend(
            Pool(
                base_address=int(pool.base_address),
                slot_bytes=int(pool.slot_bytes),
                num_slots=int(pg_desc.num_slots),
            )
            for pool in pg_desc.pools
        )
        for variant in pg_desc.slot_desc.variants:
            index = int(variant.layer_group_id)
            buffers = []
            for pool_idx, coalesced in enumerate(variant.coalesced_buffers):
                size = int(coalesced.single_buffer_size)
                offset = 0
                for buffer_id in coalesced.buffer_ids:
                    buffers.append(
                        SlotBuffer(
                            pool=first_pool + pool_idx,
                            local_layer=int(buffer_id.layer_id),
                            role=str(buffer_id.role),
                            offset=offset,
                            size=size,
                        )
                    )
                    offset += size
            cycles[index] = LifeCycle(
                index=index,
                internal_layer_ids=[int(i) for i in manager.impl.layer_grouping[index]],
                buffers=buffers,
            )
    for index, cycle in enumerate(cycles):
        if cycle is None:
            raise ValueError(f"Missing V2 layer group descriptor for layer group {index}")
    return pools, cycles


def ignored_roles(manager) -> FrozenSet[str]:
    """Storage-only roles the manager declares: they keep slot offsets but carry no state."""
    getter = getattr(manager, "get_disagg_ignored_roles", None)
    return frozenset(str(role) for role in (getter() if getter is not None else ()))


def default_global_layer_ids(manager, cycle: LifeCycle) -> List[int]:
    """Standard managers number a life cycle's layers through ``pp_layers``."""
    return [int(manager.pp_layers[lid]) for lid in cycle.internal_layer_ids]


def window_size(manager, internal_layer_id: int) -> Optional[int]:
    config = manager.impl.init_config
    if internal_layer_id >= len(config.layers):
        raise ValueError(f"Cannot resolve layer config for internal layer {internal_layer_id}")
    return getattr(config.layers[internal_layer_id], "window_size", None)
