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

from __future__ import annotations

from typing import Dict, List

from .page import AttentionLayerGroup, CacheKind, KVCachePageTable, PhysicalPool

# -------------------------------------------------------------------------
# PhysicalPool helpers
# -------------------------------------------------------------------------


def get_pool_bytes(pool: PhysicalPool) -> int:
    """Total transferable payload bytes across all slots in this pool."""
    return pool.slot_bytes * pool.num_slots


# -------------------------------------------------------------------------
# LayerGroup helpers
# -------------------------------------------------------------------------


def get_global_layer_ids(layer_group: AttentionLayerGroup) -> List[int]:
    """
    Ordered global layer IDs for *layer_group*
    """
    return [ll.global_layer_id for ll in layer_group.local_layers]


def get_layer_group_num_layers(layer_group: AttentionLayerGroup) -> int:
    """
    Number of layers in *layer_group*
    """
    return len(layer_group.local_layers)


# -------------------------------------------------------------------------
# Physical pool lookup helpers
# -------------------------------------------------------------------------


def get_physical_pool(page_table: KVCachePageTable, lg_idx: int, pool_idx: int) -> PhysicalPool:
    """
    Return the :class:`PhysicalPool` backing *pool_idx* within layer group *lg_idx*
    """
    lg = page_table.layer_groups[int(lg_idx)]
    return page_table.pool_groups[int(lg.pool_group_idx)].pools[int(pool_idx)]


def get_transferred_pool_indices(page_table: KVCachePageTable, lg_idx: int) -> List[int]:
    """Indices, within layer group *lg_idx*'s pool group, of the pools holding a buffer of the
    group's layers that transfers (``page_table.layout``)."""
    group = page_table.layout.layer_groups[int(lg_idx)]
    layers = set(group.layers)
    return sorted(
        {
            b.pool_index
            for b in page_table.layout.buffers
            if b.transferred and b.pool_group == group.pool_group and b.layer in layers
        }
    )


# -------------------------------------------------------------------------
# NIXL memory registration helpers
# -------------------------------------------------------------------------


def get_unique_pool_memory_descs(
    page_table: KVCachePageTable, device_id: int
) -> list[tuple[int, int, int, str]]:
    """Return deduplicated (ptr, size, device_id, name) tuples for all physical pools."""
    unique_pools: dict[tuple[int, int], int] = {}  # (ptr, size) -> index
    pool_counter = 0
    for lg_idx, lg in enumerate(page_table.layer_groups):
        pool_indices = get_transferred_pool_indices(page_table, lg_idx)
        if lg.kind == CacheKind.STATE:
            # MambaHybridCacheManagerV2: roles may share one allocation
            # (interleaved, equal sizes) or be separate (unequal sizes).
            # Process lowest-base first so the covering region registers
            # before higher-offset roles that fall inside it.
            pools = sorted(
                (get_physical_pool(page_table, lg_idx, pool_idx) for pool_idx in pool_indices),
                key=lambda p: p.base_address,
            )
            for pool in pools:
                pool_size = pool.num_slots * pool.slot_stride_bytes
                if any(b <= pool.base_address < b + s for (b, s) in unique_pools):
                    continue
                pool_key = (pool.base_address, pool_size)
                if pool_key not in unique_pools:
                    unique_pools[pool_key] = pool_counter
                    pool_counter += 1
        else:
            # PAGED (attention): each pool is an independent allocation
            for pool_idx in pool_indices:
                pool = get_physical_pool(page_table, lg_idx, pool_idx)
                pool_key = (pool.base_address, pool.num_slots * pool.slot_stride_bytes)
                if pool_key not in unique_pools:
                    unique_pools[pool_key] = pool_counter
                    pool_counter += 1
    return [
        (pool_ptr, pool_size, device_id, f"kv_cache_memory_pool{idx}")
        for (pool_ptr, pool_size), idx in unique_pools.items()
    ]


# -------------------------------------------------------------------------
# KVCachePageTable aggregate helpers
# -------------------------------------------------------------------------


def get_layer_to_layer_group(
    page_table: KVCachePageTable,
    kind: CacheKind | None = None,
) -> Dict[int, int]:
    """
    Build ``{global_layer_id: lg_idx}`` mapping.

    When *kind* is ``None`` (legacy default), only PAGED layer groups are
    indexed — this preserves backward compatibility with callers that
    assume attention layers partition without overlap. When a specific kind
    is given, only layer groups of that kind are indexed; within one kind,
    every global_layer_id must belong to exactly one group. Peer matching
    relies on this, so a duplicate raises instead of silently keeping the
    last group.
    """
    if kind is None:
        kind = CacheKind.PAGED
    out: Dict[int, int] = {}
    for lg_idx, lg in enumerate(page_table.layer_groups):
        if lg.kind == kind:
            for ll in lg.local_layers:
                gid = int(ll.global_layer_id)
                if gid in out:
                    raise ValueError(
                        f"global_layer_id {gid} appears in layer groups "
                        f"{out[gid]} and {lg_idx}; layer groups must partition "
                        f"a rank's layers (within kind {kind.name})"
                    )
                out[gid] = int(lg_idx)
    return out


def get_num_layers(page_table: KVCachePageTable) -> int:
    """
    Total number of attention layers across all layer groups
    """
    return sum(len(lg.local_layers) for lg in page_table.layer_groups if lg.kind == CacheKind.PAGED)


def get_num_layer_groups(page_table: KVCachePageTable) -> int:
    """Layer group count."""
    return len(page_table.layer_groups)
