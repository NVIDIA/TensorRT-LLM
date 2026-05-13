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
    Global layer IDs for the layers that appear in *pool_view*, ordered by
    each layer's first byte offset in the slot.

    Order matters: HeadMatchMapper computes ``self_layer_offset`` as the
    index of the first overlapping layer in this list and multiplies it by
    ``slot_size_per_layer`` to seek into the slot. That is correct only if
    this list mirrors the actual slot layout. ``local_layers`` order is set
    by the lifecycle's add order (V2's ``layer_grouping[lc_idx]``) which is
    *not* guaranteed to match storage's slot layout — e.g. when V2's pool
    grouping reorders layers by buffer-size class. Sorting by offset uses
    the authoritative source (buffer_entries) and works for both V1 and V2.
    """
    local_id_to_first_offset: Dict[int, int] = {}
    for entry in pool_view.buffer_entries:
        lid = int(entry["local_layer_id"])
        offset = int(entry["offset"])
        cur = local_id_to_first_offset.get(lid)
        if cur is None or offset < cur:
            local_id_to_first_offset[lid] = offset

    if not local_id_to_first_offset:
        # Empty buffer_entries (e.g. legacy V1 INDEXER pool); fall back to
        # local_layers order — consumers handle this case explicitly.
        return [ll.global_layer_id for ll in layer_group.local_layers]

    local_to_global = {ll.local_layer_id: ll.global_layer_id for ll in layer_group.local_layers}
    return [
        local_to_global[lid]
        for lid, _ in sorted(local_id_to_first_offset.items(), key=lambda x: x[1])
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
