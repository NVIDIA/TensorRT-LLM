from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Set

from tensorrt_llm._torch.disaggregation.base.region import DataRole as RegionDataRole

from .page import AttentionLayerGroup, KVCachePageTable, MambaLayerGroup, PhysicalPool, PoolView


class PoolRole(Enum):
    """Logical role of a memory pool within a layer group."""

    KV_CACHE = auto()
    KV_BLOCK_SCALE = auto()
    INDEXER = auto()


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


def get_unique_roles(pool_view: PoolView) -> Set[int]:
    """Unique role values in *pool_view*."""
    return {int(e["role"]) for e in pool_view.buffer_entries}


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


def get_pool_role(pool_view: PoolView, *, kv_factor: int) -> PoolRole:
    """
    Infer :class:`PoolRole` from the DataRole values in *pool_view*

    Raises ``ValueError`` if *pool_view* has no buffer entries — the caller
    must handle INDEXER pools (``len(pool_view.buffer_entries) == 0``) before
    invoking this function.
    """
    entries = pool_view.buffer_entries
    if entries is None or len(entries) == 0:
        raise ValueError(
            "get_pool_role called on a PoolView with empty buffer_entries. "
            "Check for INDEXER pools (len(pool_view.buffer_entries) == 0) "
            "before calling this function."
        )

    roles = {int(entry["role"]) for entry in entries}

    has_key = int(RegionDataRole.KEY) in roles
    has_value = int(RegionDataRole.VALUE) in roles
    has_key_bq = int(RegionDataRole.KEY | RegionDataRole.BLOCK_QUANT) in roles
    has_value_bq = int(RegionDataRole.VALUE | RegionDataRole.BLOCK_QUANT) in roles

    if has_key_bq or has_value_bq:
        return PoolRole.KV_BLOCK_SCALE
    if has_key and has_value:
        return PoolRole.KV_CACHE
    if has_key and not has_value:
        if int(kv_factor) == 1:
            return PoolRole.KV_CACHE
        raise ValueError("kv_factor != 1 but pool has only KEY without VALUE")
    if has_value and not has_key:
        raise ValueError("pool has only VALUE without KEY")
    raise ValueError(f"Unrecognized role combination in pool buffer_entries: {roles}")


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


def get_device_pointer(
    page_table: KVCachePageTable,
    *,
    lg_idx: int,
    pool_view: PoolView,
    slot_id: int,
    local_layer_id: int,
    role: int,
) -> int:
    """
    Compute the device pointer for a specific buffer entry
    """
    pool = get_physical_pool(page_table, lg_idx, int(pool_view.pool_idx))
    if slot_id >= pool.num_slots:
        raise ValueError(f"slot_id {slot_id} >= num_slots {pool.num_slots}")
    for e in pool_view.buffer_entries:
        if int(e["local_layer_id"]) == int(local_layer_id) and int(e["role"]) == int(role):
            return int(pool.base_address) + int(slot_id) * int(pool.slot_bytes) + int(e["offset"])
    raise ValueError(f"Buffer not found: local_layer_id={local_layer_id}, role={role}")


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
