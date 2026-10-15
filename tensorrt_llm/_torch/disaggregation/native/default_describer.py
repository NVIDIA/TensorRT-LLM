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
"""The cache describer every model gets: one row per held slot, addressed by slot arithmetic."""

from __future__ import annotations

from typing import AbstractSet, List, Sequence, Tuple

import numpy as np

from tensorrt_llm._torch.disaggregation.base.planner import CacheDescriber, LayerGroupRows
from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool


class DefaultCacheDescriber(CacheDescriber):
    """Rows straight off the page table: every view of every layer group, one row per slot held.

    A PAGED row is a block's slot; a STATE row is the request's one state slot.
    Both are ``base_address + slot * slot_stride_bytes``, ``slot_bytes`` long.
    """

    describer_id = "trtllm.default"

    def rows(
        self,
        page_table: KVCachePageTable,
        slot_table_per_layer_group: Sequence[np.ndarray],
        excluded_views: AbstractSet[Tuple[int, int]],
    ) -> List[LayerGroupRows]:
        out: List[LayerGroupRows] = []
        for lg_idx, lg in enumerate(page_table.layer_groups):
            table = np.asarray(slot_table_per_layer_group[lg_idx], dtype=np.int64)
            held = np.flatnonzero(table >= 0)
            for view_idx, pool_view in enumerate(lg.pool_views):
                if (lg_idx, view_idx) in excluded_views:
                    continue
                pool = get_physical_pool(page_table, lg_idx, pool_view.pool_idx)
                assert pool.slot_stride_bytes is not None
                out.append(
                    LayerGroupRows(
                        layer_group_index=lg_idx,
                        view_index=view_idx,
                        kind=lg.kind,
                        block_ordinals=held,
                        addresses=pool.base_address + pool.slot_stride_bytes * table[held],
                        row_bytes=pool.slot_bytes,
                        span_blocks=int(table.size),
                    )
                )
        return out
