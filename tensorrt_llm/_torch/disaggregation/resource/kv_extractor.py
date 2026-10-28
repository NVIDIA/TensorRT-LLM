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

import math
from collections import defaultdict
from typing import Dict, List, Tuple

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
    RoleLayout,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    compute_layer_byte_ranges,
    get_physical_pool,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import BufferDesc
from tensorrt_llm._torch.pyexecutor.kv_cache.page_table import ManagerLayout
from tensorrt_llm._utils import nvtx_range


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


def native_kind(buffer: BufferDesc) -> MapperKind:
    """The mapper kind native's views label a buffer with, read off its tensors. Views bucket
    their buffers by it and peers match views on it; how bytes move comes from the tensors
    (``layout_mapping``), not from the label."""
    if not buffer.split:
        return MapperKind.REPLICATED
    (tensor, *rest) = buffer.tensors
    names = [axis.name for axis in tensor.axes]
    if "shard" in names:
        return MapperKind.INDEXED  # this rank's own bytes, its heads undeclared
    if rest or "head" not in names:
        return MapperKind.SECTIONED
    return MapperKind.NHD if names[0] == "token" else MapperKind.INDEXED


def _state_geometry(buffer: BufferDesc) -> RoleLayout:
    """The section sizes or bytes per head a recurrent-state view reports to peers' checks."""
    kind = native_kind(buffer)
    if kind is MapperKind.SECTIONED:
        sections = tuple(
            math.prod(a.held if a.split else a.extent for a in t.axes) for t in buffer.tensors
        )
        if sum(sections) != buffer.size:
            raise ValueError(
                f"buffer ({buffer.layer}, {buffer.role}): sections {list(sections)} do not cover "
                f"its {buffer.size} bytes"
            )
        return RoleLayout(section_bytes=sections)
    head = next((a for a in buffer.tensors[0].axes if a.name == "head"), None)
    if kind is MapperKind.REPLICATED or head is None:
        return RoleLayout()
    return RoleLayout(bytes_per_head=buffer.size // head.held)


def _pool_views(
    lg: int,
    state: bool,
    buffers: List[Tuple[int, BufferDesc]],
) -> List[PoolView]:
    """One layer group's views: its transferred buffers, by (pool, kind)."""
    bucket_entries: Dict[tuple, list] = defaultdict(list)
    bucket_roles: Dict[tuple, set] = defaultdict(set)
    bucket_layouts: Dict[tuple, set] = defaultdict(set)
    for internal, buffer in buffers:
        key = (buffer.pool_index, native_kind(buffer))
        bucket_entries[key].append((internal, buffer.slot_offset, buffer.size))
        bucket_roles[key].add(buffer.role)
        bucket_layouts[key].add(_state_geometry(buffer) if state else RoleLayout())

    # Views are matched against a peer's and serialized, so their order (pool, then lowest slot
    # offset), their entries' order (slot offset) and role text must not follow iteration order.
    views = []
    for key in sorted(bucket_entries, key=lambda k: (k[0], min(e[1] for e in bucket_entries[k]))):
        pool_idx, kind = key
        roles = frozenset(bucket_roles[key])
        context = f"View(layer_group={lg}, pool={pool_idx}, kind={kind.name}, role={sorted(roles)})"
        entries = np.array(
            sorted(bucket_entries[key], key=lambda entry: entry[1]), dtype=BUFFER_ENTRY_DTYPE
        )
        _, bytes_per_layer = compute_layer_byte_ranges(entries, context=context)
        if len(bucket_layouts[key]) > 1:
            raise ValueError(f"{context} mixes roles with different layouts")
        layout = next(iter(bucket_layouts[key]))
        views.append(
            PoolView(
                pool_idx=pool_idx,
                buffer_entries=entries,
                pool_role=roles,
                mapper_kind=kind,
                bytes_per_layer=bytes_per_layer,
                section_bytes=(
                    list(layout.section_bytes) if layout.section_bytes is not None else None
                ),
                bytes_per_head=layout.bytes_per_head,
            )
        )
    return views


def page_table_from_layout(table: ManagerLayout) -> KVCachePageTable:
    """Native's page table of a manager's page table: its device pools, and per layer group the
    views of each pool that bucket its transferred buffers, carrying the page layout itself.

    Args:
        table: The manager's page table (``KVCacheManagerV2.get_page_table``).
    """
    layout = table.page_layout
    pool_index_of = {g: i for i, g in enumerate(table.device_pools)}
    pool_groups = [
        PhysicalPoolGroup(
            pools=[
                PhysicalPool(base_address=p.base, slot_bytes=p.slot_bytes, num_slots=p.num_slots)
                for p in pools
            ]
        )
        for pools in table.device_pools.values()
    ]
    layer_groups: List[LayerGroup] = []
    for lg, desc in enumerate(layout.layer_groups):
        local = table.local_layers[lg]
        internal_of = {global_layer: internal for internal, global_layer in local}
        local_layers = [LocalLayer(local_layer_id=i, global_layer_id=g) for i, g in local]
        buffers = [
            (internal_of[b.layer], b)
            for b in layout.buffers
            if b.transferred and b.pool_group == desc.pool_group and b.layer in internal_of
        ]
        state = desc.kind == "state"
        views = _pool_views(lg, state, buffers)
        if state:
            layer_groups.append(
                MambaLayerGroup(
                    pool_group_idx=pool_index_of[desc.pool_group],
                    local_layers=local_layers,
                    pool_views=views,
                )
            )
        else:
            layer_groups.append(
                AttentionLayerGroup(
                    pool_group_idx=pool_index_of[desc.pool_group],
                    kv_head_num_per_rank=table.kv_heads[lg],
                    sliding_window_size=desc.kept_window,
                    local_layers=local_layers,
                    pool_views=views,
                )
            )
    return KVCachePageTable(
        tokens_per_block=layout.tokens_per_page,
        layer_groups=layer_groups,
        pool_groups=pool_groups,
        layout=layout,
    )


def build_page_table_from_manager(manager: KVCacheManagerV2) -> KVCachePageTable:
    """The native page table of a V2 cache manager, from the page table the manager builds."""
    if not isinstance(manager, KVCacheManagerV2):
        raise ValueError("Python KV cache transfer requires KVCacheManagerV2; V1 is unsupported.")
    return page_table_from_layout(manager.get_page_table())
