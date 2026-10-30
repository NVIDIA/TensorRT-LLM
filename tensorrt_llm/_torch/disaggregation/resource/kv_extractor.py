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
from typing import Callable, Dict, List, Optional, Tuple

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
    CacheKind,
    KVCachePageTable,
    LayerGroup,
    LocalLayer,
    MambaLayerGroup,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
    RoleLayout,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    compute_layer_byte_ranges,
    get_layer_byte_ranges,
    get_physical_pool,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

# Mapper kinds a V2 manager may declare via get_disagg_role_mapper_kinds().
# A physical pool may mix kinds (V2 storage coalesces buffers purely by
# size within a life cycle); the page-table builder emits one PoolView per
# (pool, kind) so each view stays kind-homogeneous.
_V2_ROLE_MAPPER_KINDS = frozenset(
    {
        MapperKind.INDEXED,
        MapperKind.REPLICATED,
        MapperKind.NHD,
        MapperKind.SECTIONED,
    }
)

# Mapper kinds each life cycle's policy can serve. A SECTIONED attention
# view or an NHD recurrent view has no mapper and is rejected at build time.
_MAPPER_KINDS_BY_CACHE_KIND = {
    CacheKind.PAGED: frozenset({MapperKind.INDEXED, MapperKind.REPLICATED, MapperKind.NHD}),
    CacheKind.STATE: frozenset({MapperKind.INDEXED, MapperKind.REPLICATED, MapperKind.SECTIONED}),
}


class KVRegionExtractorV1(RegionExtractorBase):
    """
    Descriptor and region extractor for KV cache pool managed by
    KVCacheManagerV2, or described by a KVCachePageTable.

    Provides region descriptors for adapting block-wise view.
    """

    def __init__(self, kv_arg):
        if isinstance(kv_arg, KVCachePageTable):
            self._page_table = kv_arg
        else:
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


def _compute_global_layer_ids(manager, lg_idx: int) -> List[int]:
    """Compute collision-free layer IDs for a pool group for disaggregated transfer.

    These IDs are NOT actual global model layer indices. They are synthetic IDs
    whose only guarantee is:
    - Collision-free: different internal layers always produce different IDs.
    - Consistent: the same internal layer produces the same ID regardless of
      PP configuration, so that peer matching in peer.py works correctly.

    For managers that declare ``get_disagg_global_layer_ids``: returns the
    manager's declaration, which must satisfy the same two guarantees.
    For standard V2 managers: maps local IDs to global model layer IDs via
    pp_layers (which happen to also be collision-free).
    For managers with virtual layers: encodes
    (model_layer, attn_type) into synthetic IDs via the
    _layer_attn_to_layer_id inverse mapping.
    """
    local_layer_ids = manager.impl.layer_grouping[lg_idx]

    get_global_layer_ids = getattr(manager, "get_disagg_global_layer_ids", None)
    if get_global_layer_ids is not None:
        global_layer_ids = [int(layer_id) for layer_id in get_global_layer_ids(lg_idx)]
        if len(global_layer_ids) != len(local_layer_ids):
            raise ValueError(
                f"Layer group {lg_idx}: {len(global_layer_ids)} declared global layer ids "
                f"for {len(local_layer_ids)} layers"
            )
        return global_layer_ids

    if not hasattr(manager, "_layer_attn_to_layer_id"):
        # Standard: local_layer_id is index into pp_layers
        return [manager.pp_layers[lid] for lid in local_layer_ids]

    # Virtual layers: build inverse mapping internal_layer_id -> (model_layer, attn_type)
    # and encode as model_layer * num_attn_types + attn_type_value
    # `setdefault`, not assignment: the mapping is many-to-one under DeepSeek-V4.1,
    # where long-range layers that share a compressed cache all resolve to the
    # LayerId of the layer that allocated it. The owner is inserted first (it
    # precedes its dependents in layer order), so keeping the first entry names the
    # owner. Keeping the last would name whichever dependent happened to come last
    # on this rank, which differs between PP ranks and would break the peer
    # matching this ID is required to be consistent for.
    inverse = {}
    for (model_layer, attn_type), layer_id in manager._layer_attn_to_layer_id.items():
        inverse.setdefault(layer_id, (model_layer, attn_type.value))

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
    role_layouts: Optional[Dict] = None,
    ignored_roles: frozenset[DataRole] = frozenset(),
    view_partition: Optional[Callable] = None,
) -> List[PoolView]:
    """Bucket one slot-desc variant's coalesced buffers into pool views.

    One view is emitted per ``(physical pool, mapper kind)``; roles sharing a
    kind share a view (KEY+VALUE), roles with different kinds in the same
    physical pool get separate views (M3 coalesced index-K). ``role_layouts``
    supplies the per-role resharding geometry carried onto the view.
    """
    role_layouts = role_layouts or {}
    bucket_entries: Dict[tuple, list] = defaultdict(list)
    bucket_roles: Dict[tuple, set] = defaultdict(set)
    bucket_layouts: Dict[tuple, set] = defaultdict(set)
    for pool_idx, coalesced_buffer in enumerate(variant.coalesced_buffers):
        single_buffer_size = int(coalesced_buffer.single_buffer_size)
        offset = 0
        for buffer_id in coalesced_buffer.buffer_ids:
            # Local-only buffers still occupy physical slot offsets,
            # but do not contribute model state to a transfer view.
            if buffer_id.role not in ignored_roles:
                kind = role_mapper_kinds.get(buffer_id.role, default_mapper_kind)
                partition = (
                    view_partition(buffer_id.layer_id, buffer_id.role)
                    if view_partition is not None
                    else None
                )
                bucket_key = (pool_idx, kind, partition)
                bucket_entries[bucket_key].append(
                    (int(buffer_id.layer_id), offset, single_buffer_size)
                )
                bucket_roles[bucket_key].add(str(buffer_id.role))
                bucket_layouts[bucket_key].add(role_layouts.get(buffer_id.role))
            offset += single_buffer_size

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
        pool_idx, mapper_kind, _ = bucket_key
        roles = frozenset(bucket_roles[bucket_key])
        context = (
            f"View(layer_group={layer_group_id}, pool={pool_idx}, "
            f"kind={mapper_kind.name}, role={sorted(roles)})"
        )
        entries = np.array(
            sorted(bucket_entries[bucket_key], key=lambda entry: entry[1]),
            dtype=BUFFER_ENTRY_DTYPE,
        )
        # Fail fast on invalid geometry and record the uniform
        # per-layer region size on the wire. Every kind is
        # entries-driven, so the contiguous-layer-region /
        # uniform-size invariants apply to all views uniformly.
        _, bytes_per_layer = compute_layer_byte_ranges(entries, context=context)
        # Roles that share a view must agree on their resharding geometry;
        # roles without one contribute None.
        layouts = {layout for layout in bucket_layouts[bucket_key] if layout is not None}
        if len(layouts) > 1:
            raise ValueError(f"{context} mixes roles with different layouts: {sorted(roles)}")
        layout = layouts.pop() if layouts else RoleLayout()
        if layout.section_bytes is not None and sum(layout.section_bytes) != bytes_per_layer:
            raise ValueError(
                f"{context} section_bytes {list(layout.section_bytes)} do not sum to "
                f"bytes_per_layer={bytes_per_layer}"
            )
        if layout.bytes_per_head is not None and bytes_per_layer % layout.bytes_per_head != 0:
            raise ValueError(
                f"{context} bytes_per_layer={bytes_per_layer} is not a multiple of "
                f"bytes_per_head={layout.bytes_per_head}"
            )
        if layout.ring_slot_bytes is not None and (
            mapper_kind != MapperKind.REPLICATED
            or layout.ring_slot_bytes <= 0
            or bytes_per_layer % layout.ring_slot_bytes != 0
        ):
            raise ValueError(
                f"{context} ring_slot_bytes={layout.ring_slot_bytes} needs a REPLICATED view "
                f"whose bytes_per_layer={bytes_per_layer} is a positive multiple of it"
            )
        pool_views.append(
            PoolView(
                pool_idx=pool_idx,
                buffer_entries=entries,
                pool_role=roles,
                mapper_kind=mapper_kind,
                bytes_per_layer=bytes_per_layer,
                section_bytes=(
                    list(layout.section_bytes) if layout.section_bytes is not None else None
                ),
                bytes_per_head=layout.bytes_per_head,
                ring_slot_bytes=layout.ring_slot_bytes,
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
    ignored_roles = frozenset(getattr(manager, "get_disagg_ignored_roles", lambda: frozenset())())
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
    role_layouts = manager.get_disagg_role_layouts()
    view_partition = getattr(manager, "get_disagg_pool_view_partition", None)
    for role, layout in role_layouts.items():
        if not isinstance(layout, RoleLayout):
            raise ValueError(f"Invalid disaggregation role layout {layout!r} for role {role!s}")

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

        # Each variant is one layer group (life cycle) drawing slots from
        # this pool group. Multiple layer groups share a pool group when
        # their coalesced-buffer sizes are identical; within a slot, each
        # layer group's buffer offsets start from 0 independently — the
        # memory is reused, not concatenated.
        for variant in pg_desc.slot_desc.variants:
            layer_group_id = int(variant.layer_group_id)
            all_internal_layer_ids = list(manager.impl.layer_grouping[layer_group_id])
            all_global_layer_ids = _compute_global_layer_ids(manager, layer_group_id)
            local_layers = [
                LocalLayer(local_layer_id=int(iid), global_layer_id=int(gid))
                for iid, gid in zip(all_internal_layer_ids, all_global_layer_ids)
            ]

            # A life cycle is recurrent when its layers are the manager's
            # mamba layers. Recurrent and attention layers never share a life
            # cycle: their buffer configs differ, so V2 groups them apart.
            is_recurrent = isinstance(manager, MambaHybridCacheManagerV2) and any(
                manager._is_local_mamba_layer(int(layer_id)) for layer_id in all_internal_layer_ids
            )
            if is_recurrent and not all(
                manager._is_local_mamba_layer(int(layer_id)) for layer_id in all_internal_layer_ids
            ):
                raise ValueError(
                    f"Layer group {layer_group_id} mixes recurrent and attention layers"
                )
            cache_kind = CacheKind.STATE if is_recurrent else CacheKind.PAGED

            # Bucket buffer entries by (pool, mapper kind). One PoolView is
            # emitted per bucket and spans every layer of that role class,
            # so the view count per layer group is bounded by the number of
            # role classes — never by the layer count. ``pool_role`` stays
            # the manager-supplied equivalence label used for peer matching
            # without enumerating role names. Buffer offsets within a slot
            # follow ``buffer_ids`` order: the i-th buffer of a coalesced
            # buffer lives at ``i * single_buffer_size``.
            pool_views = _build_pool_views_for_variant(
                variant,
                role_mapper_kinds,
                default_mapper_kind,
                layer_group_id,
                role_layouts=role_layouts,
                ignored_roles=ignored_roles,
                view_partition=view_partition,
            )
            for pool_view in pool_views:
                if pool_view.mapper_kind not in _MAPPER_KINDS_BY_CACHE_KIND[cache_kind]:
                    raise ValueError(
                        f"Layer group {layer_group_id} ({cache_kind.name}) has no mapper for "
                        f"{pool_view.mapper_kind.name} view {sorted(pool_view.pool_role)}"
                    )

            if is_recurrent:
                # One slot per request holds every layer of the life cycle;
                # per-layer offsets live in the views' buffer entries.
                layer_groups_by_id[layer_group_id] = MambaLayerGroup(
                    pool_group_idx=storage_pg_to_list_idx[storage_pg_idx],
                    local_layers=local_layers,
                    pool_views=pool_views,
                )
                continue

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

    # Layer groups are indexed by layer_group_id (== life cycle id).
    layer_groups: List[LayerGroup] = []
    for layer_group_id, layer_group in enumerate(layer_groups_by_id):
        if layer_group is None:
            raise ValueError(f"Missing V2 layer group descriptor for layer group {layer_group_id}")
        layer_groups.append(layer_group)

    return KVCachePageTable(
        tokens_per_block=config.tokens_per_block,
        layer_groups=layer_groups,
        pool_groups=pool_groups,
    )


def build_page_table_from_manager(manager: KVCacheManagerV2) -> KVCachePageTable:
    """Build a KVCachePageTable from a V2 cache manager."""
    if not isinstance(manager, KVCacheManagerV2):
        raise ValueError("Python KV cache transfer requires KVCacheManagerV2; V1 is unsupported.")
    return _build_page_table_v2(manager)
