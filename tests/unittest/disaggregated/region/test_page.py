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

import pytest

from tensorrt_llm._torch.disaggregation.resource.page import (
    AttentionLayerGroup,
    KVCachePageTable,
    LayerGroup,
    LocalLayer,
    MambaLayerGroup,
    PhysicalPool,
    PhysicalPoolGroup,
)

pytestmark = pytest.mark.cpu_only


def test_physical_pool_construction():
    pool = PhysicalPool(base_address=0x10000, slot_bytes=256, num_slots=4)
    assert pool.base_address == 0x10000
    assert pool.slot_bytes == 256
    assert pool.num_slots == 4
    assert pool.slot_stride_bytes == 256


def test_physical_pool_roundtrip():
    pool = PhysicalPool(
        base_address=0x10000,
        slot_bytes=256,
        num_slots=4,
        slot_stride_bytes=2048,
    )
    d = pool.to_dict()
    restored = PhysicalPool.from_dict(d)
    assert restored.base_address == pool.base_address
    assert restored.slot_bytes == pool.slot_bytes
    assert restored.num_slots == pool.num_slots
    assert restored.slot_stride_bytes == pool.slot_stride_bytes


def test_local_layer_roundtrip():
    ll = LocalLayer(local_layer_id=0, global_layer_id=5)
    d = ll.to_dict()
    restored = LocalLayer.from_dict(d)
    assert restored.local_layer_id == 0
    assert restored.global_layer_id == 5


def test_layer_group_roundtrip():
    lg = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=8,
        sliding_window_size=None,
        local_layers=[LocalLayer(0, 5), LocalLayer(1, 6)],
    )
    d = lg.to_dict()
    restored = LayerGroup.from_dict(d)
    assert restored.pool_group_idx == 0
    assert restored.kv_head_num_per_rank == 8
    assert restored.sliding_window_size is None
    assert restored.local_layers == lg.local_layers


@pytest.mark.parametrize("include_layout", [False, True])
def test_mamba_v2_peer_layout_roundtrip(include_layout: bool) -> None:
    """V2 peers with an absent or true layout marker remain readable."""
    group = MambaLayerGroup(pool_group_idx=0, local_layers=[LocalLayer(0, 5)])
    data = group.to_dict()
    if not include_layout:
        data.pop("slot_major_layout")
    restored = LayerGroup.from_dict(data)
    assert isinstance(restored, MambaLayerGroup)
    assert restored.local_layers == group.local_layers


def test_page_table_rejects_legacy_mamba_peer_layout() -> None:
    """Reject a peer's explicit V1 layout through the page-table intake path."""
    group = MambaLayerGroup(pool_group_idx=0, local_layers=[]).to_dict()
    group["slot_major_layout"] = False
    with pytest.raises(ValueError, match="layer-major recurrent-state layout"):
        KVCachePageTable.from_dict(
            {"tokens_per_block": 64, "layer_groups": [group], "pool_groups": []}
        )


def test_kv_cache_page_table_roundtrip():
    page_table = KVCachePageTable(
        tokens_per_block=64,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                kv_head_num_per_rank=8,
                sliding_window_size=None,
                local_layers=[LocalLayer(0, 5)],
            )
        ],
        pool_groups=[
            PhysicalPoolGroup(
                pools=[PhysicalPool(base_address=0x10000, slot_bytes=256, num_slots=4)]
            )
        ],
    )
    d = page_table.to_dict()
    restored = KVCachePageTable.from_dict(d)
    assert restored.tokens_per_block == 64
    assert len(restored.layer_groups) == 1
    assert len(restored.pool_groups) == 1
    assert restored.pool_groups[0].pools[0].base_address == 0x10000
