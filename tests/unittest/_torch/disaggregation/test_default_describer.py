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
"""The default describer turns a request's positional slot tables into rows off the page table."""

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.native.default_describer import DefaultCacheDescriber
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LocalLayer,
    MambaLayerGroup,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)

pytestmark = pytest.mark.cpu_only

KV_SLOT = 1024
STATE_SLOT = 4096
STATE_STRIDE = 8192  # the state pool pads every slot


def _page_table() -> KVCachePageTable:
    """A PAGED group with two views over two pools, and a STATE group with one padded pool."""
    entries = np.array([(0, 0, KV_SLOT)], dtype=BUFFER_ENTRY_DTYPE)
    paged = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=1,
        local_layers=[LocalLayer(local_layer_id=0, global_layer_id=0)],
        pool_views=[
            PoolView(pool_idx=0, buffer_entries=entries),
            PoolView(pool_idx=1, buffer_entries=entries, pool_role=frozenset({"index"})),
        ],
    )
    state = MambaLayerGroup(
        pool_group_idx=1,
        local_layers=[LocalLayer(local_layer_id=0, global_layer_id=1)],
        pool_views=[PoolView(pool_idx=0, buffer_entries=entries)],
    )
    return KVCachePageTable(
        tokens_per_block=16,
        layer_groups=[paged, state],
        pool_groups=[
            PhysicalPoolGroup(
                pools=[
                    PhysicalPool(base_address=0x1000, slot_bytes=KV_SLOT, num_slots=64),
                    PhysicalPool(base_address=0x9000, slot_bytes=KV_SLOT, num_slots=64),
                ]
            ),
            PhysicalPoolGroup(
                pools=[
                    PhysicalPool(
                        base_address=0x20000,
                        slot_bytes=STATE_SLOT,
                        num_slots=8,
                        slot_stride_bytes=STATE_STRIDE,
                    )
                ]
            ),
        ],
    )


def _tables(paged, state):
    return [np.array(paged, dtype=np.int64), np.array(state, dtype=np.int64)]


def test_rows_hold_only_the_ordinals_the_table_holds_and_keep_the_span():
    rows = DefaultCacheDescriber().rows(_page_table(), _tables([7, -1, 9, -1], [3]), frozenset())

    assert [(r.layer_group_index, r.view_index, r.kind) for r in rows] == [
        (0, 0, CacheKind.PAGED),
        (0, 1, CacheKind.PAGED),
        (1, 0, CacheKind.STATE),
    ]
    kv, index, state = rows
    assert kv.block_ordinals.tolist() == [0, 2] and kv.span_blocks == 4 and len(kv) == 2
    assert kv.addresses.tolist() == [0x1000 + 7 * KV_SLOT, 0x1000 + 9 * KV_SLOT]
    assert kv.row_bytes == KV_SLOT
    # The second view reads the same slot table against its own pool.
    assert index.block_ordinals.tolist() == [0, 2]
    assert index.addresses.tolist() == [0x9000 + 7 * KV_SLOT, 0x9000 + 9 * KV_SLOT]
    # A STATE group is the request's one slot, addressed by the pool's stride.
    assert state.block_ordinals.tolist() == [0] and state.span_blocks == 1
    assert state.addresses.tolist() == [0x20000 + 3 * STATE_STRIDE]
    assert state.row_bytes == STATE_SLOT


def test_excluded_views_get_no_rows_and_empty_tables_keep_their_span():
    rows = DefaultCacheDescriber().rows(_page_table(), _tables([-1, -1], []), {(0, 1)})

    assert [(r.layer_group_index, r.view_index) for r in rows] == [(0, 0), (1, 0)]
    kv, state = rows
    assert len(kv) == 0 and kv.span_blocks == 2
    assert len(state) == 0 and state.span_blocks == 0
