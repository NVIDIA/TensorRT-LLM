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

from typing import Dict, List

import numpy as np

from tensorrt_llm._torch.disaggregation.base.region import (
    DataLayout,
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
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    compute_layer_byte_ranges,
    get_physical_pool,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes, nvtx_range
from tensorrt_llm.bindings import DataType

# Mapper kinds a V2 manager may declare via get_disagg_role_mapper_kinds().
# A physical pool may mix kinds (V2 storage coalesces buffers purely by
# size within a life cycle); the page-table builder emits one PoolView per
# (pool, kind) so each view stays kind-homogeneous.
_V2_ROLE_MAPPER_KINDS = frozenset(
    {
        MapperKind.INDEXED,
        MapperKind.REPLICATED,
        MapperKind.NHD,
    }
)


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
        to the start of a full slot. Sub-slot selection (layers, role classes,
        heads) is the mappers' responsibility; logical views carry that
        geometry in their buffer entries.

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
        kv_role_names: set[str] = {"key"}
        if not is_key_only:
            kv_role_names.add("value")
        for i, lid in enumerate(local_layer_ids):
            base_offset = i * stride
            entries.append((lid, base_offset, buffer_size))
            if not is_key_only:
                entries.append((lid, base_offset + buffer_size, buffer_size))

        kv_physical = PhysicalPool(
            base_address=base_addr, slot_bytes=slot_bytes, num_slots=num_blocks
        )
        kv_view = PoolView(
            pool_idx=0,
            buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE),
            pool_role=frozenset(kv_role_names),
            mapper_kind=MapperKind.INDEXED,
            bytes_per_layer=stride,
        )
        physical_pools = [kv_physical]
        pool_views = [kv_view]

        # Indexer K cache support. The DSA indexer K cache is identical on
        # every TP rank (single index head), so its view is REPLICATED with
        # one synthesized buffer entry per local layer: the slot packs the
        # layers equal-sized in local-layer order.
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
            indexer_bytes_per_layer = indexer_slot_bytes // len(local_layer_ids)
            indexer_view = PoolView(
                pool_idx=1,
                buffer_entries=np.array(
                    [
                        (lid, i * indexer_bytes_per_layer, indexer_bytes_per_layer)
                        for i, lid in enumerate(local_layer_ids)
                    ],
                    dtype=BUFFER_ENTRY_DTYPE,
                ),
                pool_role=frozenset({"indexer_k"}),
                mapper_kind=MapperKind.REPLICATED,
                bytes_per_layer=indexer_bytes_per_layer,
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

    Uses the V2 storage layer APIs (pool.slot_address, pool.slot_size,
    pool.num_slots) for accurate pool metadata, and stamps each PoolView
    with the manager's native role-name strings (``pool_role``) plus the
    closed-set ``mapper_kind`` discriminator used by ``build_kv_mapper``.

    Important: iterates over life cycles (layer groups), not storage pool
    groups.  Multiple life cycles with different sliding-window sizes may
    share the same underlying storage pool group when their buffer sizes
    are identical.  The page table must reflect life cycles so that
    per-window transfer logic works correctly.
    """
    from collections import defaultdict

    from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

    storage = manager.impl._storage
    config = manager.impl._init_config

    # Find GPU level
    gpu_level = 0
    for level_idx, cache_tier_config in enumerate(config.cache_tiers):
        if cache_tier_config.tier == CacheTier.GPU_MEM:
            gpu_level = level_idx
            break

    # Every V2 manager declares how native roles map to the closed set of
    # disaggregation mapper kinds; Role.ALL is the required fallback.
    role_mapper_kinds = manager.get_disagg_role_mapper_kinds()
    if Role.ALL not in role_mapper_kinds:
        raise ValueError("Disaggregation role mapping must define Role.ALL")
    for role, mapper_kind in role_mapper_kinds.items():
        if not isinstance(mapper_kind, MapperKind):
            raise ValueError(
                f"Invalid disaggregation mapper kind {mapper_kind!r} for role {role!s}"
            )
        if mapper_kind not in _V2_ROLE_MAPPER_KINDS:
            supported = ", ".join(kind.name for kind in sorted(_V2_ROLE_MAPPER_KINDS))
            raise ValueError(
                f"Unsupported V2 disaggregation mapper kind {mapper_kind.name} "
                f"for role {role!s}; supported kinds: {supported}"
            )
        # INDEXED is the whole-manager legacy default, not a per-role
        # choice: it may only appear as the Role.ALL fallback. Side-cache
        # roles (e.g. INDEX_KEY) may declare their own non-INDEXED kind
        # alongside it.
        if mapper_kind is MapperKind.INDEXED and role != Role.ALL:
            raise ValueError(
                f"MapperKind.INDEXED is only valid as the Role.ALL mapping; "
                f"got it for role {role!s}"
            )
    default_mapper_kind = role_mapper_kinds[Role.ALL]

    # Bucket buffer entries by (life_cycle, pool, mapper kind). One PoolView
    # is emitted per bucket and spans every layer of that role class, so the
    # view count per layer group is bounded by the number of role classes —
    # never by the layer count. A physical pool may hold several classes
    # (V2 storage coalesces buffers purely by size within a life cycle, so
    # e.g. MiniMax M3's index-K shares the K/V pool when their per-block
    # sizes coincide); each class still gets its own view, which keeps peer
    # matching independent of that physical coalescing decision.
    # ``pool_role`` stays the manager-supplied equivalence label used for
    # peer matching without enumerating role names.
    bucket_entries: Dict[tuple, list] = defaultdict(list)
    bucket_roles: Dict[tuple, set] = defaultdict(set)
    for buffer_id, attr in storage._buffer_attr.items():
        layer_id, role = buffer_id
        kind = role_mapper_kinds.get(role, default_mapper_kind)
        bucket_key = (int(attr.life_cycle_id), int(attr.pool_index), kind)
        bucket_entries[bucket_key].append((int(layer_id), int(attr.offset), int(attr.size)))
        bucket_roles[bucket_key].add(str(role))

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

        # Emit this life cycle's views: one per bucket, i.e. per
        # (life_cycle, pool, mapper kind) — the life-cycle component is
        # fixed by the enclosing loop, so within this group it reads as
        # one view per (pool, mapper-kind class of roles). Roles sharing a
        # kind share a view (KEY+VALUE); roles with different kinds in the
        # same physical pool get separate views (M3 coalesced index-K).
        # The life-cycle filter also handles pool-group sharing: multiple
        # life cycles may draw slots from the same storage pool group, and
        # each group only picks up buckets holding its own buffers.
        # All ordering below is canonicalization — the page table is
        # serialized and matched against peers, so view order (pool, then
        # lowest slot offset), entry order (slot offset), and role text
        # must not depend on dict/set iteration order.
        pool_views = []
        lc_bucket_keys = sorted(
            (key for key in bucket_entries if key[0] == lc_idx),
            key=lambda key: (key[1], min(entry[1] for entry in bucket_entries[key])),
        )
        for bucket_key in lc_bucket_keys:
            _, pool_idx, mapper_kind = bucket_key
            roles = frozenset(bucket_roles[bucket_key])
            entries = np.array(
                sorted(bucket_entries[bucket_key], key=lambda entry: entry[1]),
                dtype=BUFFER_ENTRY_DTYPE,
            )
            # Fail fast on invalid geometry and record the uniform per-layer
            # region size on the wire. Every kind is entries-driven, so the
            # contiguous-layer-region / uniform-size invariants apply to all
            # views uniformly.
            _, bytes_per_layer = compute_layer_byte_ranges(
                entries,
                context=(
                    f"View(life_cycle={lc_idx}, pool={pool_idx}, "
                    f"kind={mapper_kind.name}, role={sorted(roles)})"
                ),
            )
            pool_views.append(
                PoolView(
                    pool_idx=pool_idx,
                    buffer_entries=entries,
                    pool_role=roles,
                    mapper_kind=mapper_kind,
                    bytes_per_layer=bytes_per_layer,
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
            AttentionLayerGroup(
                pool_group_idx=storage_pg_to_list_idx[storage_pg_idx],
                kv_head_num_per_rank=num_kv_heads,
                sliding_window_size=sliding_window_size,
                local_layers=local_layers,
                pool_views=pool_views,
            )
        )

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
