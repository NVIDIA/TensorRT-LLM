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
    AttentionLayerGroup,
    KVCachePageTable,
    LayerGroup,
    LocalLayer,
    MambaLayerGroup,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    MambaHybridCacheManager,
    PythonMambaCacheManager,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes, nvtx_range
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

    @nvtx_range("KVRegionExtractorV1.extract")
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


def _build_layer_group_for_mamba(
    manager: MambaHybridCacheManager, pool_group_idx: int
) -> MambaLayerGroup:
    assert isinstance(manager._impl, PythonMambaCacheManager), (
        "CppMambaCacheManager is not supported with Python transceiver, please set TRTLLM_USE_CPP_MAMBA=0"
    )

    mamba_layer_offsets = {
        int(global_layer_id): int(local_layer_id)
        for global_layer_id, local_layer_id in manager._impl.mamba_layer_offsets.items()
    }

    conv_state = manager._impl.mamba_cache.conv
    ssm_state = manager._impl.mamba_cache.temporal

    conv_pool = PhysicalPool(
        base_address=conv_state.data_ptr(),
        slot_bytes=conv_state.stride(1) * conv_state.element_size(),
        num_slots=conv_state.shape[1],
    )

    ssm_pool = PhysicalPool(
        base_address=ssm_state.data_ptr(),
        slot_bytes=ssm_state.stride(1) * ssm_state.element_size(),
        num_slots=ssm_state.shape[1],
    )

    # Per-section bytes for conv_state and per-head bytes for ssm_state.
    # conv_state layout: [x: d_inner/tp | B: ng*ds/tp | C: ng*ds/tp] x (d_conv-1)
    # ssm_state layout: (nheads/tp, head_dim, d_state)
    d_conv_m1 = conv_state.shape[3]
    conv_elem_size = conv_state.element_size()
    conv_section_dims = manager._impl.conv_section_dims
    conv_section_bytes = [dim * d_conv_m1 * conv_elem_size for dim in conv_section_dims]

    head_dim = ssm_state.shape[3]
    d_state = ssm_state.shape[4]
    ssm_elem_size = ssm_state.element_size()
    ssm_bytes_per_head = head_dim * d_state * ssm_elem_size

    return MambaLayerGroup(
        pool_group_idx=pool_group_idx,
        mamba_layer_offsets=mamba_layer_offsets,
        conv_states=conv_pool,
        ssm_states=ssm_pool,
        conv_section_bytes=conv_section_bytes,
        ssm_bytes_per_head=ssm_bytes_per_head,
    )


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
            AttentionLayerGroup(
                pool_group_idx=group_id,
                kv_head_num_per_rank=num_kv_heads,
                sliding_window_size=window_size,
                local_layers=local_layers,
                pool_views=pool_views,
            )
        )
    if isinstance(kv_cache_manager, MambaHybridCacheManager):
        mamba_layer_group_idx = len(pool_groups)
        mamba_layer_group = _build_layer_group_for_mamba(kv_cache_manager, mamba_layer_group_idx)
        layer_groups.append(mamba_layer_group)

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

    Uses KVCacheManagerV2's public pool_group_descs layout API. A physical
    pool group may be shared by several layer groups; layer_groups remains
    indexed by layer_group_id while pool_group_idx points at the shared
    physical pool group entry.
    """
    from tensorrt_llm._torch.pyexecutor.resource_manager import Role

    _ROLE_STR_TO_ENUM: dict[str, DataRole] = {
        Role.KEY: DataRole.KEY,
        Role.VALUE: DataRole.VALUE,
        Role.KEY_BLOCK_SCALE: DataRole.KEY | DataRole.BLOCK_QUANT,
        Role.VALUE_BLOCK_SCALE: DataRole.VALUE | DataRole.BLOCK_QUANT,
    }

    def _role_str_to_enum(role: str) -> DataRole:
        role = str(role)
        if role not in _ROLE_STR_TO_ENUM:
            valid_roles = list(_ROLE_STR_TO_ENUM.keys())
            raise ValueError(f"Invalid role: '{role}'. Valid roles: {valid_roles}")
        return _ROLE_STR_TO_ENUM[role]

    config = manager.impl.init_config
    pool_group_descs = manager.impl.pool_group_descs

    def _window_size_for_layer(internal_layer_id: int):
        if internal_layer_id < len(config.layers):
            return getattr(config.layers[internal_layer_id], "window_size", None)

        if hasattr(manager, "_layer_attn_to_layer_id"):
            for (model_layer, _attn_type), layer_id in manager._layer_attn_to_layer_id.items():
                if layer_id != internal_layer_id:
                    continue
                local_layer = manager.layer_offsets.get(model_layer)
                if local_layer is not None and local_layer < len(config.layers):
                    return getattr(config.layers[local_layer], "window_size", None)
                if model_layer < len(config.layers):
                    return getattr(config.layers[model_layer], "window_size", None)

        raise ValueError(f"Cannot resolve layer config for internal layer {internal_layer_id}")

    pool_groups: List[PhysicalPoolGroup] = []
    storage_pg_to_list_idx: Dict[int, int] = {}
    layer_groups_by_id: List[LayerGroup | None] = [None] * len(manager.impl.layer_grouping)

    for pg_desc in pool_group_descs:
        storage_pg_idx = int(pg_desc.pool_group_index)
        storage_pg_to_list_idx[storage_pg_idx] = len(pool_groups)
        pool_groups.append(
            PhysicalPoolGroup(
                pools=[
                    PhysicalPool(
                        base_address=int(pool.base_address),
                        slot_bytes=int(pool.slot_bytes),
                        num_slots=int(pg_desc.num_slots),
                    )
                    for pool in pg_desc.pools
                ]
            )
        )

        for variant in pg_desc.slot_desc.variants:
            layer_group_id = int(variant.layer_group_id)
            all_internal_layer_ids = list(manager.impl.layer_grouping[layer_group_id])
            all_global_layer_ids = _compute_global_layer_ids(manager, layer_group_id)

            local_layers = [
                LocalLayer(local_layer_id=int(iid), global_layer_id=int(gid))
                for iid, gid in zip(all_internal_layer_ids, all_global_layer_ids)
            ]

            pool_views = []
            for pool_idx, coalesced_buffer in enumerate(variant.coalesced_buffers):
                entries = []
                offset = 0
                single_buffer_size = int(coalesced_buffer.single_buffer_size)
                for buffer_id in coalesced_buffer.buffer_ids:
                    entries.append(
                        (
                            int(buffer_id.layer_id),
                            int(_role_str_to_enum(buffer_id.role)),
                            offset,
                            single_buffer_size,
                        )
                    )
                    offset += single_buffer_size

                if entries:
                    pool_views.append(
                        PoolView(
                            pool_idx=pool_idx,
                            buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE),
                        )
                    )

            first_local_layer = all_internal_layer_ids[0]
            if first_local_layer < len(manager.num_kv_heads_per_layer):
                num_kv_heads = manager.num_kv_heads_per_layer[first_local_layer]
            else:
                num_kv_heads = manager.num_kv_heads_per_layer[0]
            sliding_window_size = _window_size_for_layer(first_local_layer)

            layer_groups_by_id[layer_group_id] = AttentionLayerGroup(
                pool_group_idx=storage_pg_to_list_idx[storage_pg_idx],
                kv_head_num_per_rank=num_kv_heads,
                sliding_window_size=sliding_window_size,
                local_layers=local_layers,
                pool_views=pool_views,
            )

    layer_groups: List[LayerGroup] = []
    for layer_group_id, layer_group in enumerate(layer_groups_by_id):
        if layer_group is None:
            raise ValueError(f"Missing V2 layer group descriptor for layer group {layer_group_id}")
        layer_groups.append(layer_group)

    if isinstance(manager, MambaHybridCacheManager):
        mamba_layer_group_idx = len(pool_groups)
        mamba_layer_group = _build_layer_group_for_mamba(manager, mamba_layer_group_idx)
        layer_groups.append(mamba_layer_group)

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
