from typing import Dict, List

import numpy as np

from tensorrt_llm._torch.disaggregation.base.region import (
    DataLayout,
    DataRole,
    MemRegionGroup,
    RegionExtractorBase,
    SpecRegion,
)
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    KVCachePageTable,
    LayerGroup,
    LocalLayer,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType


class KVRegionExtractorV1(RegionExtractorBase):
    """
    Descriptor and region extractor for KV cache pool managed by
    KVCacheManager, KVCacheManagerV2, or described by a KVCachePageTable.

    Provides region descriptors for adapting block-wise view.
    """

    def __init__(self, kv_arg):
        if isinstance(kv_arg, KVCachePageTable):
            self._page_table = kv_arg
        else:
            # Assume it is a manager (KVCacheManager / KVCacheManagerV2)
            self._page_table = build_page_table_from_manager(kv_arg)
        self._data_layout = DataLayout.HND

    @property
    def page_table(self) -> KVCachePageTable:
        return self._page_table

    def extract(
        self,
        region_ids: np.ndarray,
        layer_group_id: int = 0,
        pool_idx: int = 0,
    ) -> SpecRegion:
        """
        Given a list of region_ids (block IDs or slot IDs), returns a single
        SpecRegion whose memory is a MemRegionGroup containing all blocks
        described by region_ids.

        For KV cache: each ptr = base_address + slot_id * slot_bytes, pointing
        to the start of a full slot.  The slot contains buffer entries for all
        layers in this layer_group laid out contiguously from offset 0.

        Args:
            layer_group_id: The layer group index (= life cycle index).
            pool_idx: The pool index within the layer group.
        """
        lg = self._page_table.layer_groups[layer_group_id]
        pv = lg.pool_views[pool_idx]
        pool = get_physical_pool(self._page_table, layer_group_id, pv.pool_idx)

        base_ptr = pool.base_address
        block_size = pool.slot_bytes

        # KV cache: filter out invalid block_ids (BAD_PAGE_INDEX = -1)
        valid = region_ids >= 0
        ptrs = base_ptr + block_size * region_ids[valid]
        memory = MemRegionGroup(ptrs=ptrs, bytes_per_region=block_size)
        return SpecRegion(memory=memory)


# ---------------------------------------------------------------------------
# Page table builders
# ---------------------------------------------------------------------------


def build_page_table(kv_cache_manager: KVCacheManager) -> KVCachePageTable:
    """Build a KVCachePageTable from a KVCacheManager (V1)."""
    if kv_cache_manager.dtype == DataType.NVFP4:
        raise NotImplementedError("NVFP4 quantization not supported")

    tokens_per_block = kv_cache_manager.tokens_per_block

    # Group local layers by their window size (layer group)
    window_size_to_local_layer_ids = kv_cache_manager._get_window_size_to_layers()
    layer_offsets = kv_cache_manager.layer_offsets
    local_to_global = {local_id: global_id for global_id, local_id in layer_offsets.items()}

    if len(window_size_to_local_layer_ids) < 1:
        raise ValueError("KVRegionExtractorV1: window_size_to_local_layer_ids is empty")

    sorted_window_sizes = sorted(
        window_size_to_local_layer_ids.keys(), key=lambda x: (x is None, x)
    )

    pool_groups: List[PhysicalPoolGroup] = []
    layer_groups: List[LayerGroup] = []

    for group_id, window_size in enumerate(sorted_window_sizes):
        local_layer_ids = window_size_to_local_layer_ids[window_size]
        first_local_layer = local_layer_ids[0]

        # Get pool base address via pool_mapping -> pool_pointers
        pool_id = int(kv_cache_manager.kv_cache_pool_mapping[first_local_layer][0].item())
        base_addr = int(kv_cache_manager.kv_cache_pool_pointers[pool_id][0].item())

        # Get num_blocks from per-layer pool view: shape = (numBlocks, kvFactor, blockSize)
        pool_layer_view = kv_cache_manager.impl.get_primary_pool_data(first_local_layer)
        num_blocks = pool_layer_view.shape[0]

        num_kv_heads = kv_cache_manager.num_kv_heads_per_layer[first_local_layer]
        kv_factor = kv_cache_manager.kv_factor
        is_key_only = kv_factor == 1

        elements_per_buffer = tokens_per_block * num_kv_heads * kv_cache_manager.head_dim
        buffer_size = get_size_in_bytes(elements_per_buffer, kv_cache_manager.dtype)
        stride = buffer_size * kv_factor
        slot_bytes = stride * len(local_layer_ids)

        entries = []
        for i, lid in enumerate(local_layer_ids):
            base_offset = i * stride
            entries.append((lid, int(DataRole.KEY), base_offset, buffer_size))
            if not is_key_only:
                entries.append((lid, int(DataRole.VALUE), base_offset + buffer_size, buffer_size))

        kv_physical = PhysicalPool(
            base_address=base_addr, slot_bytes=slot_bytes, num_slots=num_blocks
        )
        kv_view = PoolView(pool_idx=0, buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE))
        physical_pools = [kv_physical]
        pool_views = [kv_view]

        # Indexer K cache support
        if getattr(kv_cache_manager, "enable_indexer_k_cache", False):
            indexer_pool = kv_cache_manager.impl.get_indexer_k_cache_pool()
            # indexer_pool shape: (numBlocks, numLayers, kvFactor, blockSize), dtype=UINT8
            # slot_bytes = numLayers * kvFactor * blockSize * element_size
            per_block_elems = 1
            for d in indexer_pool.shape[1:]:  # skip numBlocks dim
                per_block_elems *= d
            indexer_slot_bytes = per_block_elems * indexer_pool.element_size()
            indexer_physical = PhysicalPool(
                base_address=int(indexer_pool.data_ptr()),
                slot_bytes=indexer_slot_bytes,
                num_slots=num_blocks,
            )
            indexer_view = PoolView(
                pool_idx=1, buffer_entries=np.array([], dtype=BUFFER_ENTRY_DTYPE)
            )
            physical_pools.append(indexer_physical)
            pool_views.append(indexer_view)

        pool_groups.append(PhysicalPoolGroup(pools=physical_pools))
        local_layers = [
            LocalLayer(local_layer_id=int(lid), global_layer_id=int(local_to_global[lid]))
            for lid in local_layer_ids
        ]
        layer_groups.append(
            LayerGroup(
                pool_group_idx=group_id,
                kv_head_num_per_rank=num_kv_heads,
                sliding_window_size=window_size,
                local_layers=local_layers,
                pool_views=pool_views,
            )
        )

    return KVCachePageTable(
        tokens_per_block=tokens_per_block,
        layer_groups=layer_groups,
        pool_groups=pool_groups,
    )


def _compute_global_layer_ids(manager, lg_idx: int) -> List[int]:
    """Compute collision-free layer IDs for a pool group for disaggregated transfer.

    These IDs are NOT actual global model layer indices. They are synthetic IDs
    whose only guarantee is:
    - Collision-free: different internal layers always produce different IDs.
    - Consistent: the same internal layer produces the same ID regardless of
      PP configuration, so that peer matching in peer.py works correctly.

    For standard V2 managers: maps local IDs to global model layer IDs via
    pp_layers (which happen to also be collision-free).
    For managers with virtual layers: encodes
    (model_layer, attn_type) into synthetic IDs via the
    _layer_attn_to_layer_id inverse mapping.
    """
    local_layer_ids = manager.impl.layer_grouping[lg_idx]

    if not hasattr(manager, "_layer_attn_to_layer_id"):
        # Standard: local_layer_id is index into pp_layers
        return [manager.pp_layers[lid] for lid in local_layer_ids]

    # Virtual layers: build inverse mapping internal_layer_id -> (model_layer, attn_type)
    # and encode as model_layer * num_attn_types + attn_type_value
    inverse = {}
    for (model_layer, attn_type), layer_id in manager._layer_attn_to_layer_id.items():
        inverse[layer_id] = (model_layer, attn_type.value)

    # Use the full enum range for consistent encoding across all PP ranks.
    # Different PP ranks may have different subsets of attention types (e.g.,
    # a rank with only ratio=128 layers won't have INDEXER_* types), so using
    # max(local values) would produce different num_attn_types across ranks,
    # causing the same (model_layer, attn_type) to map to different global IDs.
    first_key = next(iter(manager._layer_attn_to_layer_id.keys()))
    attn_type_class = type(first_key[1])
    num_attn_types = max(e.value for e in attn_type_class) + 1

    return [inverse[lid][0] * num_attn_types + inverse[lid][1] for lid in local_layer_ids]


def _build_page_table_v2(manager) -> KVCachePageTable:
    """Build a KVCachePageTable from a KVCacheManagerV2.

    Uses the V2 storage layer APIs (pool.slot_address, pool.slot_size,
    pool.num_slots) for accurate pool metadata, and determines PoolRole
    from the DataRole of buffers in each pool.

    Important: iterates over life cycles (layer groups), not storage pool
    groups.  Multiple life cycles with different sliding-window sizes may
    share the same underlying storage pool group when their buffer sizes
    are identical.  The page table must reflect life cycles so that
    per-window transfer logic works correctly.
    """
    from collections import defaultdict

    from tensorrt_llm._torch.pyexecutor.resource_manager import Role
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

    _ROLE_STR_TO_ENUM: dict[str, DataRole] = {
        Role.KEY: DataRole.KEY,
        Role.VALUE: DataRole.VALUE,
        Role.KEY_BLOCK_SCALE: DataRole.KEY | DataRole.BLOCK_QUANT,
        Role.VALUE_BLOCK_SCALE: DataRole.VALUE | DataRole.BLOCK_QUANT,
    }

    def _role_str_to_enum(role: str) -> DataRole:
        if role not in _ROLE_STR_TO_ENUM:
            valid_roles = list(_ROLE_STR_TO_ENUM.keys())
            raise ValueError(f"Invalid role: '{role}'. Valid roles: {valid_roles}")
        return _ROLE_STR_TO_ENUM[role]

    storage = manager.impl._storage
    config = manager.impl._init_config

    # Find GPU level
    gpu_level = 0
    for level_idx, cache_tier_config in enumerate(config.cache_tiers):
        if cache_tier_config.tier == CacheTier.GPU_MEM:
            gpu_level = level_idx
            break

    # Collect buffer entries keyed by (life_cycle_id, pool_idx)
    buffer_by_lc_pool: Dict[tuple, list] = defaultdict(list)

    for buffer_id, attr in storage._buffer_attr.items():
        layer_id, role = buffer_id
        lc_id = attr.life_cycle_id
        pool_idx = attr.pool_index
        pool_key = (int(lc_id), pool_idx)
        buffer_by_lc_pool[pool_key].append(
            (layer_id, _role_str_to_enum(role), attr.offset, attr.size)
        )

    # Iterate over life cycles (layer groups), not storage pool groups.
    # Multiple layer_groups can share the same storage pool_group when their
    # slot_size_list (coalesced buffer sizes) are identical.  In that case,
    # different layer_groups draw slots from the same physical pool, but a
    # slot is exclusively allocated to one layer_group at a time (managed by
    # SlotAllocator).  Within a slot, each layer_group's buffer offsets start
    # from 0 independently — the memory is reused, not concatenated.
    # Therefore, slot_bytes / num_layers_for_this_layer_group correctly gives
    # the per-layer size, and buffer offsets within a slot are contiguous for
    # each layer_group.
    num_life_cycles = storage.num_life_cycles
    pool_group_storage = storage._levels[gpu_level].storage._pool_groups

    pool_groups: List[PhysicalPoolGroup] = []
    storage_pg_to_list_idx: Dict[int, int] = {}
    layer_groups: List[LayerGroup] = []

    for lc_idx in range(num_life_cycles):
        # Resolve the storage pool group for this life cycle.
        # storage_pg_idx may be the same for multiple lc_idx values.
        storage_pg_idx = storage.get_pool_group_index(lc_idx)
        pool_group = pool_group_storage[storage_pg_idx]
        num_pools = pool_group.num_pools

        # Build PhysicalPoolGroup once per unique storage pool group.
        if storage_pg_idx not in storage_pg_to_list_idx:
            storage_pg_to_list_idx[storage_pg_idx] = len(pool_groups)
            pool_groups.append(
                PhysicalPoolGroup(
                    pools=[
                        PhysicalPool(
                            base_address=int(pool_group._pools[pi].slot_address(0)),
                            slot_bytes=int(pool_group._pools[pi].slot_size),
                            num_slots=int(pool_group._pools[pi].num_slots),
                        )
                        for pi in range(num_pools)
                    ]
                )
            )

        # Compute group-level global layer IDs and internal layer IDs
        all_internal_layer_ids = list(manager.impl.layer_grouping[lc_idx])
        all_global_layer_ids = _compute_global_layer_ids(manager, lc_idx)

        local_layers = [
            LocalLayer(local_layer_id=int(iid), global_layer_id=int(gid))
            for iid, gid in zip(all_internal_layer_ids, all_global_layer_ids)
        ]

        pool_views = []
        for pool_idx in range(num_pools):
            pool_key = (lc_idx, pool_idx)
            buffers_info = buffer_by_lc_pool.get(pool_key, [])

            # Skip pools that have no buffers for this layer group.
            # Multiple life cycles may share the same storage pool group;
            # only include pools that actually belong to this life cycle.
            if not buffers_info:
                continue

            pool_views.append(
                PoolView(
                    pool_idx=pool_idx,
                    buffer_entries=np.array(buffers_info, dtype=BUFFER_ENTRY_DTYPE),
                )
            )

        # Determine layer group metadata.
        # For managers with virtual layers, internal layer_ids
        # may exceed the length of num_kv_heads_per_layer. Use index 0 as all
        # layers within a pool group share the same kv_heads count.
        first_local_layer = all_internal_layer_ids[0]
        if first_local_layer < len(manager.num_kv_heads_per_layer):
            num_kv_heads = manager.num_kv_heads_per_layer[first_local_layer]
        else:
            num_kv_heads = manager.num_kv_heads_per_layer[0]
        life_cycle = manager.impl._life_cycles[lc_idx]
        sliding_window_size = life_cycle.window_size

        layer_groups.append(
            LayerGroup(
                pool_group_idx=storage_pg_to_list_idx[storage_pg_idx],
                kv_head_num_per_rank=num_kv_heads,
                sliding_window_size=sliding_window_size,
                local_layers=local_layers,
                pool_views=pool_views,
            )
        )

    return KVCachePageTable(
        tokens_per_block=config.tokens_per_block,
        layer_groups=layer_groups,
        pool_groups=pool_groups,
    )


def _is_kv_cache_manager_v2(obj) -> bool:
    return hasattr(obj, "impl") and hasattr(obj.impl, "layer_grouping")


def build_page_table_from_manager(manager) -> KVCachePageTable:
    """Unified entry point: build a KVCachePageTable from any manager type.

    Supports KVCacheManager (V1) and KVCacheManagerV2.
    """
    if _is_kv_cache_manager_v2(manager):
        return _build_page_table_v2(manager)
    else:
        return build_page_table(manager)
