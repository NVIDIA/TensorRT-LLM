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

from typing import Dict, List, Set

from .page import AttentionLayerGroup, KVCachePageTable, MambaLayerGroup, PhysicalPool, PoolView

# -------------------------------------------------------------------------
# PhysicalPool helpers
# -------------------------------------------------------------------------


def get_pool_bytes(pool: PhysicalPool) -> int:
    """Total bytes across all slots in this pool."""
    return pool.slot_bytes * pool.num_slots


def get_slot_address(pool: PhysicalPool, slot_id: int) -> int:
    """Base address of *slot_id*."""
    if slot_id >= pool.num_slots:
        raise ValueError(f"slot_id {slot_id} >= num_slots {pool.num_slots}")
    return pool.base_address + slot_id * pool.slot_bytes


# -------------------------------------------------------------------------
# PoolView helpers
# -------------------------------------------------------------------------


def compute_layer_byte_ranges(
    buffer_entries,
    *,
    declared_bytes_per_layer: "int | None" = None,
    context: str = "PoolView",
) -> tuple[Dict[int, int], int]:
    """Per-layer slot-relative byte offsets from raw buffer entries.

    Returns ``({local_layer_id: start_offset}, bytes_per_layer)``. A layer's
    region is the concatenation of its buffer entries, which must be
    contiguous within the slot; the region size must be uniform across
    layers (the slot may interleave other role classes between layers, so
    only the *size* is uniform — offsets are per layer). ``context`` labels
    error messages; ``declared_bytes_per_layer`` cross-checks a size that
    was recorded elsewhere.
    """
    starts: Dict[int, int] = {}
    totals: Dict[int, int] = {}
    entries_by_layer: Dict[int, list] = {}
    for entry in buffer_entries:
        entries_by_layer.setdefault(int(entry["local_layer_id"]), []).append(
            (int(entry["offset"]), int(entry["size"]))
        )
    if not entries_by_layer:
        raise ValueError(f"{context} has no buffer entries; per-layer byte ranges are undefined")
    for layer_id, spans in entries_by_layer.items():
        spans.sort()
        for (off, size), (next_off, _) in zip(spans, spans[1:]):
            if off + size != next_off:
                raise ValueError(
                    f"{context} layer {layer_id} buffers are "
                    f"not contiguous: [{off}, {off + size}) is followed by offset {next_off}"
                )
        starts[layer_id] = spans[0][0]
        totals[layer_id] = sum(size for _, size in spans)
    distinct_totals = set(totals.values())
    if len(distinct_totals) != 1:
        raise ValueError(
            f"{context} per-layer region sizes are not uniform: {sorted(totals.items())}"
        )
    bytes_per_layer = distinct_totals.pop()
    if declared_bytes_per_layer is not None and declared_bytes_per_layer != bytes_per_layer:
        raise ValueError(
            f"{context} declares bytes_per_layer={declared_bytes_per_layer} but buffer "
            f"entries sum to {bytes_per_layer} per layer"
        )
    return starts, bytes_per_layer


def get_layer_byte_ranges(pool_view: PoolView) -> tuple[Dict[int, int], int]:
    """Per-layer byte ranges of a view; see :func:`compute_layer_byte_ranges`."""
    return compute_layer_byte_ranges(
        pool_view.buffer_entries,
        declared_bytes_per_layer=pool_view.bytes_per_layer,
        context=f"PoolView(pool_idx={pool_view.pool_idx}, role={sorted(pool_view.pool_role)})",
    )


def get_unique_layers(pool_view: PoolView) -> Set[int]:
    """Unique local layer IDs in *pool_view*."""
    return {int(e["local_layer_id"]) for e in pool_view.buffer_entries}


def get_num_buffer_entries(pool_view: PoolView) -> int:
    """Number of buffer entries."""
    return len(pool_view.buffer_entries)


def get_pool_view_num_layers(pool_view: PoolView) -> int:
    """
    Number of unique layers represented in *pool_view*
    """
    return len(get_unique_layers(pool_view))


def get_pool_view_global_layer_ids(
    pool_view: PoolView, layer_group: AttentionLayerGroup
) -> List[int]:
    """
    Global layer IDs for the layers that appear in *pool_view*, ordered as in
    *layer_group.local_layers*.
    """
    local_ids_in_pool = get_unique_layers(pool_view)
    return [
        ll.global_layer_id
        for ll in layer_group.local_layers
        if ll.local_layer_id in local_ids_in_pool
    ]


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
        if isinstance(lg, MambaLayerGroup):
            num_mamba_layers = len(lg.mamba_layer_offsets)
            for pool in [lg.conv_states, lg.ssm_states]:
                pool_size = num_mamba_layers * pool.num_slots * pool.slot_bytes
                pool_key = (pool.base_address, pool_size)
                if pool_key not in unique_pools:
                    unique_pools[pool_key] = pool_counter
                    pool_counter += 1
        else:
            for pv in lg.pool_views:
                pool = get_physical_pool(page_table, lg_idx, pv.pool_idx)
                pool_key = (pool.base_address, get_pool_bytes(pool))
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


def get_layer_to_layer_group(page_table: KVCachePageTable) -> Dict[int, int]:
    """
    Build ``{global_layer_id: lg_idx}`` mapping
    """
    out: Dict[int, int] = {}
    for lg_idx, lg in enumerate(page_table.layer_groups):
        if isinstance(lg, AttentionLayerGroup):
            for ll in lg.local_layers:
                out[int(ll.global_layer_id)] = int(lg_idx)
    return out


def get_num_layers(page_table: KVCachePageTable) -> int:
    """
    Total number of attention layers across all layer groups
    """
    return sum(
        len(lg.local_layers)
        for lg in page_table.layer_groups
        if isinstance(lg, AttentionLayerGroup)
    )


def get_num_layer_groups(page_table: KVCachePageTable) -> int:
    """Layer group count."""
    return len(page_table.layer_groups)


def get_pool_views(page_table: KVCachePageTable) -> List[List[PoolView]]:
    """
    Pool views per attention layer group
    """
    return [lg.pool_views for lg in page_table.layer_groups if isinstance(lg, AttentionLayerGroup)]


def get_total_pools(page_table: KVCachePageTable) -> int:
    """Total pool-view count."""
    return sum(
        len(lg.pool_views) for lg in page_table.layer_groups if isinstance(lg, AttentionLayerGroup)
    )


def get_total_buffer_entries(page_table: KVCachePageTable) -> int:
    """Total buffer entries across all pools."""
    return sum(
        get_num_buffer_entries(pv)
        for lg in page_table.layer_groups
        if isinstance(lg, AttentionLayerGroup)
        for pv in lg.pool_views
    )


def get_total_pool_bytes(page_table: KVCachePageTable) -> int:
    """
    Total allocated bytes across all physical pools
    """
    return sum(
        get_pool_bytes(get_physical_pool(page_table, lg_idx, pv.pool_idx))
        for lg_idx, lg in enumerate(page_table.layer_groups)
        if isinstance(lg, AttentionLayerGroup)
        for pv in lg.pool_views
    )


def get_total_slots(page_table: KVCachePageTable) -> int:
    """
    Total slot count across all physical pools
    """
    return sum(
        get_physical_pool(page_table, lg_idx, pv.pool_idx).num_slots
        for lg_idx, lg in enumerate(page_table.layer_groups)
        if isinstance(lg, AttentionLayerGroup)
        for pv in lg.pool_views
    )
