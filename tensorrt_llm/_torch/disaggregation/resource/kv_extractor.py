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
from dataclasses import replace
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tensorrt_llm._torch.disaggregation.base.region import (
    DataLayout,
    MemRegionGroup,
    RegionExtractorBase,
    SpecRegion,
)
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    MAMBA_CONV_ROLE,
    MAMBA_SSM_ROLE,
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
    get_layer_byte_ranges,
    get_physical_pool,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    MambaHybridCacheManager,
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes, nvtx_range
from tensorrt_llm.bindings import DataType

# Roles mapped this way hold identical bytes on every rank, so they need none
# of a life cycle's resharding metadata and are described straight from the
# V2 pool descriptors.
_REPLICATED_SIDE_KINDS = frozenset({MapperKind.REPLICATED})

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
        # extract_slot runs per request per view, but the page table is fixed
        # once built, so resolve each view's layer offsets only once.
        self._slot_layouts: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}

    def _slot_layout(self, layer_group_id: int, pool_idx: int) -> "Tuple[np.ndarray, int]":
        layout = self._slot_layouts.get((layer_group_id, pool_idx))
        if layout is None:
            pv = self._page_table.layer_groups[layer_group_id].pool_views[pool_idx]
            layer_offsets, bytes_per_layer = get_layer_byte_ranges(pv)
            layout = (np.array(sorted(layer_offsets.values()), dtype=np.int64), bytes_per_layer)
            self._slot_layouts[(layer_group_id, pool_idx)] = layout
        return layout

    @property
    def page_table(self) -> KVCachePageTable:
        return self._page_table

    @nvtx_range("KVRegionExtractorV1.extract_slot")
    def extract_slot(
        self,
        slot_id: int,
        layer_group_id: int = 0,
        pool_idx: int = 0,
    ) -> SpecRegion:
        """Extract per-layer pointers for a single slot (used for mamba state).

        Returns a SpecRegion with one pointer per layer:
            ptr[i] = base + view_offset[i] + slot_id * slot_stride

        View offsets are explicit because a recurrent role may exist on only a
        subset of layers or be interleaved with another role in a V2 pool.
        """
        lg = self._page_table.layer_groups[layer_group_id]
        pv = lg.pool_views[pool_idx]
        pool = get_physical_pool(self._page_table, layer_group_id, pv.pool_idx)

        base_ptr = pool.base_address
        slot_stride = pool.slot_stride_bytes
        assert slot_stride is not None

        ordered_offsets, bytes_per_layer = self._slot_layout(layer_group_id, pool_idx)
        ptrs = ordered_offsets + (base_ptr + slot_id * slot_stride)
        memory = MemRegionGroup(ptrs=ptrs, bytes_per_region=bytes_per_layer)
        return SpecRegion(memory=memory)

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
        block_stride = pool.slot_stride_bytes
        assert block_stride is not None

        # KV cache: filter out invalid block_ids (BAD_PAGE_INDEX = -1)
        valid = region_ids >= 0
        ptrs = base_ptr + block_stride * region_ids[valid]
        memory = MemRegionGroup(ptrs=ptrs, bytes_per_region=block_size)
        return SpecRegion(memory=memory)


# ---------------------------------------------------------------------------
# Page table builders
# ---------------------------------------------------------------------------


def _build_mamba_pool_views(conv_pool, ssm_pool, local_layers):
    """Build pool_views for mamba: conv at pool_idx=0, ssm at pool_idx=1.

    Conv uses mapper_kind=SECTIONED (section-level granularity for TP split),
    SSM uses mapper_kind=INDEXED (head-level granularity). MambaPolicy.build_mapper
    dispatches ConvStateMismatchMapper vs MambaHeadMismatchMapper accordingly.
    """
    sorted_lids = [
        ll.local_layer_id for ll in sorted(local_layers, key=lambda ll: ll.local_layer_id)
    ]
    conv_layer_stride = int(conv_pool.layer_stride_bytes)
    ssm_layer_stride = int(ssm_pool.layer_stride_bytes)
    return [
        PoolView(
            pool_idx=0,
            buffer_entries=np.array(
                [
                    (lid, offset * conv_layer_stride, conv_pool.slot_bytes)
                    for offset, lid in enumerate(sorted_lids)
                ],
                dtype=BUFFER_ENTRY_DTYPE,
            ),
            pool_role=MAMBA_CONV_ROLE,
            mapper_kind=MapperKind.SECTIONED,
            bytes_per_layer=conv_pool.slot_bytes,
        ),
        PoolView(
            pool_idx=1,
            buffer_entries=np.array(
                [
                    (lid, offset * ssm_layer_stride, ssm_pool.slot_bytes)
                    for offset, lid in enumerate(sorted_lids)
                ],
                dtype=BUFFER_ENTRY_DTYPE,
            ),
            pool_role=MAMBA_SSM_ROLE,
            mapper_kind=MapperKind.INDEXED,
            bytes_per_layer=ssm_pool.slot_bytes,
        ),
    ]


def _build_layer_group_for_mamba(
    manager: MambaHybridCacheManager, pool_group_idx: int
) -> "tuple[MambaLayerGroup, PhysicalPoolGroup]":
    local_layers = [
        LocalLayer(local_layer_id=int(lid), global_layer_id=int(gid))
        for gid, lid in sorted(manager._impl.mamba_layer_offsets.items(), key=lambda x: x[1])
    ]

    conv_state = manager._impl.mamba_cache.conv
    ssm_state = manager._impl.mamba_cache.temporal

    conv_pool = PhysicalPool(
        base_address=conv_state.data_ptr(),
        slot_bytes=conv_state.stride(1) * conv_state.element_size(),
        num_slots=conv_state.shape[1],
        layer_stride_bytes=conv_state.stride(0) * conv_state.element_size(),
    )

    ssm_pool = PhysicalPool(
        base_address=ssm_state.data_ptr(),
        slot_bytes=ssm_state.stride(1) * ssm_state.element_size(),
        num_slots=ssm_state.shape[1],
        layer_stride_bytes=ssm_state.stride(0) * ssm_state.element_size(),
    )

    # Per-section bytes for conv_state and per-head bytes for ssm_state.
    # The section ordering is supplied by the cache manager because Mamba2
    # uses [x | B | C], while GDN uses [Q | K | V].
    # ssm_state layout: (nheads/tp, head_dim, d_state)
    d_conv_m1 = conv_state.shape[3]
    conv_elem_size = conv_state.element_size()
    conv_section_dims = manager._impl.conv_section_dims
    conv_section_bytes = [dim * d_conv_m1 * conv_elem_size for dim in conv_section_dims]

    head_dim = ssm_state.shape[3]
    d_state = ssm_state.shape[4]
    ssm_elem_size = ssm_state.element_size()
    ssm_bytes_per_head = head_dim * d_state * ssm_elem_size

    pool_group = PhysicalPoolGroup(pools=[conv_pool, ssm_pool])
    layer_group = MambaLayerGroup(
        pool_group_idx=pool_group_idx,
        local_layers=local_layers,
        pool_views=_build_mamba_pool_views(conv_pool, ssm_pool, local_layers),
        conv_section_bytes=conv_section_bytes,
        ssm_bytes_per_head=ssm_bytes_per_head,
    )
    return layer_group, pool_group


def _slot_stride_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.stride(0) * tensor.element_size())


def _build_v2_mamba_state_pool(states: Sequence[torch.Tensor]) -> PhysicalPool:
    """Describe affine layer/slot addressing for one V2 Mamba state role."""
    if not states:
        raise ValueError("V2 Mamba state pool requires at least one layer")

    first_state = states[0]
    base_address = int(first_state.data_ptr())
    num_slots = int(first_state.shape[0])
    slot_bytes = int(first_state[0].numel() * first_state.element_size())
    slot_stride_bytes = _slot_stride_bytes(first_state)

    num_layers = len(states)
    # V2 coalesces buffers by size, so an unrelated role's buffer may sit
    # between two layers of this one. Take the stride from the layers themselves.
    layer_stride_bytes = (
        int(states[1].data_ptr()) - base_address if num_layers > 1 else slot_stride_bytes
    )
    if layer_stride_bytes < slot_bytes:
        raise ValueError("V2 Mamba state layers of one role must not overlap")
    if (num_layers - 1) * layer_stride_bytes + slot_bytes > slot_stride_bytes:
        raise ValueError("V2 Mamba state layers of one role must fit inside one slot")

    for layer_offset, state in enumerate(states):
        state_slot_bytes = int(state[0].numel() * state.element_size())
        if (
            int(state.shape[0]) != num_slots
            or state_slot_bytes != slot_bytes
            or _slot_stride_bytes(state) != slot_stride_bytes
        ):
            raise ValueError("V2 Mamba state tensors must share one slot layout per role")
        expected_address = base_address + layer_offset * layer_stride_bytes
        if int(state.data_ptr()) != expected_address:
            raise ValueError("V2 Mamba state tensors must have a uniform layer stride per role")

    return PhysicalPool(
        base_address=base_address,
        slot_bytes=slot_bytes,
        num_slots=num_slots,
        slot_stride_bytes=slot_stride_bytes,
        layer_stride_bytes=layer_stride_bytes,
    )


def _build_layer_group_for_v2_mamba(
    manager: MambaHybridCacheManagerV2, pool_group_idx: int, side_state=None
) -> "tuple[MambaLayerGroup, PhysicalPoolGroup]":
    local_layers = [
        LocalLayer(local_layer_id=int(lid), global_layer_id=int(gid))
        for gid, lid in sorted(manager.mamba_layer_offsets.items(), key=lambda x: x[1])
    ]

    num_layers = len(local_layers)
    expected_offsets = list(range(num_layers))
    if sorted(ll.local_layer_id for ll in local_layers) != expected_offsets:
        raise ValueError("V2 Mamba layer offsets must be dense")
    if len(manager.all_conv_states) != num_layers or len(manager.all_ssm_states) != num_layers:
        raise ValueError("V2 Mamba state tensors must match the layer-offset table")

    first_conv_state = manager.all_conv_states[0]
    first_ssm_state = manager.all_ssm_states[0]
    conv_pool = _build_v2_mamba_state_pool(manager.all_conv_states)
    ssm_pool = _build_v2_mamba_state_pool(manager.all_ssm_states)
    if conv_pool.num_slots != ssm_pool.num_slots:
        raise ValueError("V2 Mamba convolution and SSM states must have the same number of slots")

    d_conv_m1 = manager.conv_state_shape[1]
    conv_elem_size = first_conv_state.element_size()
    _, head_dim, d_state = manager.ssm_state_shape
    conv_section_bytes = [dim * d_conv_m1 * conv_elem_size for dim in manager.conv_section_dims]

    ssm_elem_size = first_ssm_state.element_size()
    ssm_bytes_per_head = head_dim * d_state * ssm_elem_size

    physical_pools = [conv_pool, ssm_pool]
    pool_views = _build_mamba_pool_views(conv_pool, ssm_pool, local_layers)
    if side_state is not None:
        side_views, side_pools = side_state
        remapped: Dict[int, int] = {}
        for view in side_views:
            if view.pool_idx not in remapped:
                remapped[view.pool_idx] = len(physical_pools)
                physical_pools.append(side_pools[view.pool_idx])
            pool_views.append(replace(view, pool_idx=remapped[view.pool_idx]))
    pool_group = PhysicalPoolGroup(pools=physical_pools)
    layer_group = MambaLayerGroup(
        pool_group_idx=pool_group_idx,
        local_layers=local_layers,
        pool_views=pool_views,
        conv_section_bytes=conv_section_bytes,
        ssm_bytes_per_head=ssm_bytes_per_head,
        slot_major_layout=True,
    )
    return layer_group, pool_group


def _build_non_kv_layers(
    manager,
    layer_groups: List[LayerGroup],
    pool_groups: List[PhysicalPoolGroup],
    *,
    has_v2_mamba: bool = False,
    v2_mamba_insert_idx: Optional[int] = None,
    v2_mamba_side_state=None,
) -> None:
    """Append (or insert) non-KV (recurrent/state) layer groups to the page table.

    Extension point for non-attention layer types. Currently handles Mamba;
    add elif branches here for future recurrent/state layer types.

    Args:
        has_v2_mamba: If True, the V2 manager owns mamba layers that need
            a dedicated pool group appended.
        v2_mamba_insert_idx: If set, insert the mamba layer group at this
            position (preserving original lifecycle ordering) instead of
            appending at the end.
    """
    if isinstance(manager, MambaHybridCacheManagerV2):
        if has_v2_mamba and manager.local_num_mamba_layers > 0:
            # Append a dedicated pool group (don't mutate the shared V2 entry).
            mamba_pg_idx = len(pool_groups)
            layer_group, local_pool_group = _build_layer_group_for_v2_mamba(
                manager, mamba_pg_idx, side_state=v2_mamba_side_state
            )
            pool_groups.append(local_pool_group)
            if v2_mamba_insert_idx is not None:
                layer_groups.insert(v2_mamba_insert_idx, layer_group)
            else:
                layer_groups.append(layer_group)
    elif isinstance(manager, MambaHybridCacheManager):
        pool_group_idx = len(pool_groups)
        layer_group, pool_group = _build_layer_group_for_mamba(manager, pool_group_idx)
        layer_groups.append(layer_group)
        pool_groups.append(pool_group)


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
        # every TP rank (single index head), so its view is REPLICATED. With a
        # per-layer indexer mask (cross-layer indexer sharing, e.g. GLM 5.2)
        # only the "full" indexer-owning layers get a pool row, so the view
        # covers that subset: one buffer entry per owning layer, each mapped to
        # its packed row in the (possibly masked) pool. When the mask is absent
        # every layer owns a row (dense/legacy layout) and this reduces to the
        # equal-sized packing in local-layer order.
        if kv_cache_manager.enable_indexer_k_cache:
            local_indexer_mask = kv_cache_manager.indexer_k_cache_local_layer_mask
            owning_layer_ids = [
                lid
                for lid in local_layer_ids
                if local_indexer_mask is None or local_indexer_mask[lid]
            ]
            # A layer group whose layers are all masked out owns no indexer pool
            # row on this rank (the pool getter would raise); skip it so the peer
            # simply transfers nothing for this rank's indexer.
            if owning_layer_ids:
                indexer_pool = kv_cache_manager.impl.get_indexer_k_cache_pool()
                if indexer_pool.shape[1] != len(owning_layer_ids):
                    raise RuntimeError(
                        "The DSA indexer K-cache pool row count does not match "
                        "the number of indexer-owning layers in its layer group: "
                        f"{indexer_pool.shape[1]} rows for {len(owning_layer_ids)} layers"
                    )
                # indexer_pool shape: (numBlocks, numIndexerLayers, kvFactor,
                # blockSize), dtype=UINT8. numIndexerLayers is the number of
                # owning layers on this rank (== the attention layer count when
                # unmasked). slot_bytes packs every owning-layer row.
                per_block_elems = 1
                for d in indexer_pool.shape[1:]:  # skip numBlocks dim
                    per_block_elems *= d
                indexer_slot_bytes = per_block_elems * indexer_pool.element_size()
                indexer_bytes_per_layer = indexer_slot_bytes // indexer_pool.shape[1]
                indexer_physical = PhysicalPool(
                    base_address=int(indexer_pool.data_ptr()),
                    slot_bytes=indexer_slot_bytes,
                    num_slots=num_blocks,
                )
                indexer_view = PoolView(
                    pool_idx=len(physical_pools),
                    buffer_entries=np.array(
                        [
                            (
                                lid,
                                kv_cache_manager.impl.get_indexer_k_cache_pool_layer_idx(lid)
                                * indexer_bytes_per_layer,
                                indexer_bytes_per_layer,
                            )
                            for lid in owning_layer_ids
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
    _build_non_kv_layers(kv_cache_manager, layer_groups, pool_groups)

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


def _build_pool_views_for_variant(
    variant,
    role_mapper_kinds: Dict,
    default_mapper_kind: MapperKind,
    layer_group_id: int,
    keep_mapper_kinds: Optional[FrozenSet[MapperKind]] = None,
    layer_id_map: Optional[Dict[int, int]] = None,
) -> List[PoolView]:
    """Bucket one slot-desc variant's coalesced buffers into pool views.

    ``keep_mapper_kinds`` selects which mapper kinds to emit. ``layer_id_map``
    renumbers buffer layer ids for a layer group that indexes its layers
    differently from the V2 descriptor; buffers it omits are skipped.
    """
    bucket_entries: Dict[tuple, list] = defaultdict(list)
    bucket_roles: Dict[tuple, set] = defaultdict(set)
    for pool_idx, coalesced_buffer in enumerate(variant.coalesced_buffers):
        single_buffer_size = int(coalesced_buffer.single_buffer_size)
        offset = 0
        for buffer_id in coalesced_buffer.buffer_ids:
            kind = role_mapper_kinds.get(buffer_id.role, default_mapper_kind)
            bucket_key = (pool_idx, kind)
            if keep_mapper_kinds is not None and kind not in keep_mapper_kinds:
                offset += single_buffer_size
                continue
            layer_id = int(buffer_id.layer_id)
            if layer_id_map is not None:
                if layer_id not in layer_id_map:
                    # Dropping it would leave the receiver with zeroed state.
                    raise ValueError(
                        f"Layer group {layer_group_id}: role {buffer_id.role!s} lives on "
                        f"layer {layer_id}, which this layer group does not index"
                    )
                layer_id = layer_id_map[layer_id]
            bucket_entries[bucket_key].append((layer_id, offset, single_buffer_size))
            bucket_roles[bucket_key].add(str(buffer_id.role))
            offset += single_buffer_size

    # Emit this layer group's views: one per (pool, mapper-kind
    # class of roles). Roles sharing a kind share a view
    # (KEY+VALUE); roles with different kinds in the same physical
    # pool get separate views (M3 coalesced index-K).
    # All ordering below is canonicalization — the page table is
    # serialized and matched against peers, so view order (pool,
    # then lowest slot offset), entry order (slot offset), and role
    # text must not depend on dict/set iteration order.
    pool_views = []
    lg_bucket_keys = sorted(
        bucket_entries,
        key=lambda key: (key[0], min(entry[1] for entry in bucket_entries[key])),
    )
    for bucket_key in lg_bucket_keys:
        pool_idx, mapper_kind = bucket_key
        roles = frozenset(bucket_roles[bucket_key])
        entries = np.array(
            sorted(bucket_entries[bucket_key], key=lambda entry: entry[1]),
            dtype=BUFFER_ENTRY_DTYPE,
        )
        # Fail fast on invalid geometry and record the uniform
        # per-layer region size on the wire. Every kind is
        # entries-driven, so the contiguous-layer-region /
        # uniform-size invariants apply to all views uniformly.
        _, bytes_per_layer = compute_layer_byte_ranges(
            entries,
            context=(
                f"View(layer_group={layer_group_id}, pool={pool_idx}, "
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

    return pool_views


def _build_page_table_v2(manager) -> KVCachePageTable:
    """Build a KVCachePageTable from a KVCacheManagerV2.

    Uses KVCacheManagerV2's public ``pool_group_descs`` layout API and
    stamps each PoolView with the manager's native role-name strings
    (``pool_role``) plus the closed-set ``mapper_kind`` discriminator used
    by the policy ``build_mapper`` methods.

    A physical pool group may be shared by several layer groups (life
    cycles whose coalesced-buffer sizes are identical); each layer group
    is exactly one ``SlotDescVariant`` of one pool group, so iterating
    variants visits every layer group once. ``layer_groups`` stays indexed
    by layer_group_id while ``pool_group_idx`` points at the shared
    physical pool group entry, so per-window transfer logic keeps working.
    """
    config = manager.impl.init_config
    pool_group_descs = manager.impl.pool_group_descs

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
    has_v2_mamba: bool = False
    v2_mamba_layer_group_ids: set = set()  # layer_group_ids handled by _build_non_kv_layers
    v2_mamba_side_state = None  # (replicated views, pools) split off the mamba life cycle

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

        # Each variant is one layer group (life cycle) drawing slots from
        # this pool group. Multiple layer groups share a pool group when
        # their coalesced-buffer sizes are identical; within a slot, each
        # layer group's buffer offsets start from 0 independently — the
        # memory is reused, not concatenated.
        for variant in pg_desc.slot_desc.variants:
            layer_group_id = int(variant.layer_group_id)
            all_internal_layer_ids = list(manager.impl.layer_grouping[layer_group_id])
            if isinstance(manager, MambaHybridCacheManagerV2) and any(
                manager._is_local_mamba_layer(int(layer_id)) for layer_id in all_internal_layer_ids
            ):
                # Conv and SSM state need resharding metadata the pool
                # descriptors do not carry, so _build_non_kv_layers describes
                # them. Replicated side state (e.g. PLE) needs none of it and
                # is taken straight from the descriptors here. The Mamba layer
                # group numbers layers by mamba offset, so renumber.
                has_v2_mamba = True
                v2_mamba_layer_group_ids.add(layer_group_id)
                internal_to_mamba_local = {
                    int(iid): manager.mamba_layer_offsets[int(gid)]
                    for iid, gid in zip(
                        all_internal_layer_ids, _compute_global_layer_ids(manager, layer_group_id)
                    )
                    if int(gid) in manager.mamba_layer_offsets
                }
                side_views = _build_pool_views_for_variant(
                    variant,
                    role_mapper_kinds,
                    default_mapper_kind,
                    layer_group_id,
                    keep_mapper_kinds=_REPLICATED_SIDE_KINDS,
                    layer_id_map=internal_to_mamba_local,
                )
                if side_views:
                    v2_mamba_side_state = (
                        side_views,
                        {
                            # slot_bytes is one layer's payload; the descriptor's
                            # slot_bytes is the whole coalesced slot, i.e. the stride.
                            view.pool_idx: PhysicalPool(
                                base_address=int(pg_desc.pools[view.pool_idx].base_address),
                                slot_bytes=int(view.bytes_per_layer),
                                num_slots=int(pg_desc.num_slots),
                                slot_stride_bytes=int(pg_desc.pools[view.pool_idx].slot_bytes),
                            )
                            for view in side_views
                        },
                    )
                continue

            all_global_layer_ids = _compute_global_layer_ids(manager, layer_group_id)

            local_layers = [
                LocalLayer(local_layer_id=int(iid), global_layer_id=int(gid))
                for iid, gid in zip(all_internal_layer_ids, all_global_layer_ids)
            ]

            # Bucket buffer entries by (pool, mapper kind). One PoolView is
            # emitted per bucket and spans every layer of that role class,
            # so the view count per layer group is bounded by the number of
            # role classes — never by the layer count. A physical pool may
            # hold several classes (V2 storage coalesces buffers purely by
            # size within a layer group, so e.g. MiniMax M3's index-K shares
            # the K/V pool when their per-block sizes coincide); each class
            # still gets its own view, which keeps peer matching independent
            # of that physical coalescing decision. ``pool_role`` stays the
            # manager-supplied equivalence label used for peer matching
            # without enumerating role names. Buffer offsets within a slot
            # follow ``buffer_ids`` order: the i-th buffer of a coalesced
            # buffer lives at ``i * single_buffer_size``.
            pool_views = _build_pool_views_for_variant(
                variant, role_mapper_kinds, default_mapper_kind, layer_group_id
            )

            # Determine layer group metadata.
            # For managers with virtual layers, internal layer_ids
            # may exceed the length of num_kv_heads_per_layer. Use index 0 as
            # all layers within a pool group share the same kv_heads count.
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

    # Preserve original lifecycle ordering: mamba groups stay at their
    # original layer_group_id positions. _build_non_kv_layers fills them in.
    layer_groups: List[LayerGroup] = []
    for layer_group_id, layer_group in enumerate(layer_groups_by_id):
        if layer_group is None and layer_group_id not in v2_mamba_layer_group_ids:
            raise ValueError(f"Missing V2 layer group descriptor for layer group {layer_group_id}")
        if layer_group is not None:
            layer_groups.append(layer_group)
        # For skipped mamba IDs, a placeholder is inserted by _build_non_kv_layers below

    _build_non_kv_layers(
        manager,
        layer_groups,
        pool_groups,
        has_v2_mamba=has_v2_mamba,
        v2_mamba_insert_idx=min(v2_mamba_layer_group_ids) if v2_mamba_layer_group_ids else None,
        v2_mamba_side_state=v2_mamba_side_state,
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
