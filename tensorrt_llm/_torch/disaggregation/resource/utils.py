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

"""Where a rank's pages lie, read off its page table (``CacheLayoutAdapter.page_table``): layer
group ``i`` of the table is the manager's layer group ``i``, and pool ``p`` of a layer group is pool
``p`` of the group's device pool group."""

from __future__ import annotations

from typing import List

import numpy as np

from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout.page_table import (
    DevicePool,
    ManagerLayout,
)


def get_cache_kinds(page_table: ManagerLayout) -> List[CacheKind]:
    """Per layer group, how its slots are addressed: one per block (attention) or one per request
    (recurrent state)."""
    return [
        CacheKind.STATE if group.kind == "state" else CacheKind.PAGED
        for group in page_table.page_layout.layer_groups
    ]


def get_physical_pool(page_table: ManagerLayout, lg_idx: int, pool_idx: int) -> DevicePool:
    """The device pool backing pool ``pool_idx`` of layer group ``lg_idx``."""
    group = page_table.page_layout.layer_groups[int(lg_idx)]
    return page_table.device_pools[group.pool_group][int(pool_idx)]


def slot_pointers(
    page_table: ManagerLayout, lg_idx: int, pool_idx: int, slots: np.ndarray
) -> np.ndarray:
    """Where ``slots`` of pool ``pool_idx`` of layer group ``lg_idx`` start; slots below 0 (blocks
    not held) are left out."""
    pool = get_physical_pool(page_table, lg_idx, pool_idx)
    slots = np.asarray(slots, dtype=np.int64)
    return pool.base + pool.slot_bytes * slots[slots >= 0]


def get_transferred_pool_indices(page_table: ManagerLayout, lg_idx: int) -> List[int]:
    """Indices, within layer group *lg_idx*'s pool group, of the pools holding a buffer of the
    group's layers that transfers."""
    layout = page_table.page_layout
    group = layout.layer_groups[int(lg_idx)]
    layers = set(group.layers)
    return sorted(
        {
            b.pool_index
            for b in layout.buffers
            if b.transferred and b.pool_group == group.pool_group and b.layer in layers
        }
    )


def get_unique_pool_memory_descs(
    page_table: ManagerLayout, device_id: int
) -> list[tuple[int, int, int, str]]:
    """Return deduplicated (ptr, size, device_id, name) tuples for the device pools holding
    transferred buffers."""
    unique_pools: dict[tuple[int, int], int] = {}  # (ptr, size) -> index
    pool_counter = 0
    for lg_idx, kind in enumerate(get_cache_kinds(page_table)):
        pools = [
            get_physical_pool(page_table, lg_idx, pool_idx)
            for pool_idx in get_transferred_pool_indices(page_table, lg_idx)
        ]
        if kind == CacheKind.STATE:
            # MambaHybridCacheManagerV2: roles may share one allocation
            # (interleaved, equal sizes) or be separate (unequal sizes).
            # Process lowest-base first so the covering region registers
            # before higher-offset roles that fall inside it.
            for pool in sorted(pools, key=lambda p: p.base):
                if any(b <= pool.base < b + s for (b, s) in unique_pools):
                    continue
                pool_key = (pool.base, pool.num_slots * pool.slot_bytes)
                if pool_key not in unique_pools:
                    unique_pools[pool_key] = pool_counter
                    pool_counter += 1
        else:
            # PAGED (attention): each pool is an independent allocation
            for pool in pools:
                pool_key = (pool.base, pool.num_slots * pool.slot_bytes)
                if pool_key not in unique_pools:
                    unique_pools[pool_key] = pool_counter
                    pool_counter += 1
    return [
        (pool_ptr, pool_size, device_id, f"kv_cache_memory_pool{idx}")
        for (pool_ptr, pool_size), idx in unique_pools.items()
    ]
