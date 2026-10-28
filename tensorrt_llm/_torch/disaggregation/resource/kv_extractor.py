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

from typing import List

import numpy as np

from tensorrt_llm._torch.disaggregation.base.region import (
    DataLayout,
    MemRegionGroup,
    RegionExtractorBase,
    SpecRegion,
)
from tensorrt_llm._torch.disaggregation.resource.page import (
    AttentionLayerGroup,
    KVCachePageTable,
    LayerGroup,
    LocalLayer,
    MambaLayerGroup,
    PhysicalPool,
    PhysicalPoolGroup,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
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
        to the start of a full slot. Sub-slot selection (layers, roles, heads)
        is the mappers' responsibility, from the page layout.

        Args:
            layer_group_id: The layer group index (= life cycle index).
            pool_idx: The pool's index within the layer group's pool group.
        """
        pool = get_physical_pool(self._page_table, layer_group_id, pool_idx)

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


def page_table_from_layout(table: ManagerLayout) -> KVCachePageTable:
    """Native's page table of a manager's page table: its device pools and layer groups, carrying
    the page layout itself.

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
        local_layers = [
            LocalLayer(local_layer_id=i, global_layer_id=g) for i, g in table.local_layers[lg]
        ]
        if desc.kind == "state":
            layer_groups.append(
                MambaLayerGroup(
                    pool_group_idx=pool_index_of[desc.pool_group],
                    local_layers=local_layers,
                )
            )
        else:
            layer_groups.append(
                AttentionLayerGroup(
                    pool_group_idx=pool_index_of[desc.pool_group],
                    kv_head_num_per_rank=table.kv_heads[lg],
                    sliding_window_size=desc.kept_window,
                    local_layers=local_layers,
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
