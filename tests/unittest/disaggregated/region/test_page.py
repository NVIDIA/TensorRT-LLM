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

import numpy as np

from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LayerGroup,
    LocalLayer,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)


def _make_buffer_entries():
    return np.array(
        [
            (0, 0, 128),  # local_layer_id=0, offset=0, size=128
            (0, 128, 128),  # local_layer_id=0, offset=128, size=128
        ],
        dtype=BUFFER_ENTRY_DTYPE,
    )


def test_physical_pool_construction():
    pool = PhysicalPool(base_address=0x10000, slot_bytes=256, num_slots=4)
    assert pool.base_address == 0x10000
    assert pool.slot_bytes == 256
    assert pool.num_slots == 4


def test_physical_pool_roundtrip():
    pool = PhysicalPool(base_address=0x10000, slot_bytes=256, num_slots=4)
    d = pool.to_dict()
    restored = PhysicalPool.from_dict(d)
    assert restored.base_address == pool.base_address
    assert restored.slot_bytes == pool.slot_bytes
    assert restored.num_slots == pool.num_slots


def test_pool_view_roundtrip():
    entries = _make_buffer_entries()
    pv = PoolView(pool_idx=0, buffer_entries=entries)
    d = pv.to_dict()
    restored = PoolView.from_dict(d)
    assert restored.pool_idx == 0
    assert len(restored.buffer_entries) == 2
    assert restored.buffer_entries[0]["offset"] == 0
    assert restored.buffer_entries[1]["offset"] == 128


def test_pool_view_kind_roundtrip():
    """REPLICATED / NHD kinds survive serialization; default stays INDEXED."""
    entries = _make_buffer_entries()
    for kind in (MapperKind.REPLICATED, MapperKind.NHD):
        view = PoolView(
            pool_idx=2,
            buffer_entries=entries,
            pool_role=frozenset({"index_key"}),
            mapper_kind=kind,
        )
        restored = PoolView.from_dict(view.to_dict())
        assert restored.mapper_kind == kind
        assert restored.pool_role == frozenset({"index_key"})

    legacy = PoolView(pool_idx=0, buffer_entries=entries).to_dict()
    assert PoolView.from_dict(legacy).mapper_kind == MapperKind.INDEXED


def test_local_layer_roundtrip():
    ll = LocalLayer(local_layer_id=0, global_layer_id=5)
    d = ll.to_dict()
    restored = LocalLayer.from_dict(d)
    assert restored.local_layer_id == 0
    assert restored.global_layer_id == 5


def test_layer_group_roundtrip():
    entries = _make_buffer_entries()
    lg = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=8,
        sliding_window_size=None,
        local_layers=[LocalLayer(0, 5), LocalLayer(1, 6)],
        pool_views=[PoolView(pool_idx=0, buffer_entries=entries)],
    )
    d = lg.to_dict()
    restored = LayerGroup.from_dict(d)
    assert restored.pool_group_idx == 0
    assert restored.kv_head_num_per_rank == 8
    assert restored.sliding_window_size is None
    assert len(restored.local_layers) == 2
    assert len(restored.pool_views) == 1


def test_kv_cache_page_table_roundtrip():
    entries = _make_buffer_entries()
    page_table = KVCachePageTable(
        tokens_per_block=64,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                kv_head_num_per_rank=8,
                sliding_window_size=None,
                local_layers=[LocalLayer(0, 5)],
                pool_views=[PoolView(pool_idx=0, buffer_entries=entries)],
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


# ---------------------------------------------------------------------------
# bytes_per_layer serialization and get_layer_byte_ranges geometry
# ---------------------------------------------------------------------------


def test_pool_view_bytes_per_layer_roundtrip():
    entries = _make_buffer_entries()
    view = PoolView(
        pool_idx=1,
        buffer_entries=entries,
        pool_role=frozenset({"key", "value"}),
        mapper_kind=MapperKind.NHD,
        bytes_per_layer=256,
    )
    restored = PoolView.from_dict(view.to_dict())
    assert restored.bytes_per_layer == 256

    # None (INDEXED views) survives the roundtrip too.
    legacyless = PoolView(pool_idx=0, buffer_entries=entries)
    assert PoolView.from_dict(legacyless.to_dict()).bytes_per_layer is None


def _view(entries, bytes_per_layer=None):
    return PoolView(
        pool_idx=0,
        buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE),
        pool_role=frozenset({"key", "value"}),
        mapper_kind=MapperKind.NHD,
        bytes_per_layer=bytes_per_layer,
    )


def test_layer_byte_ranges_uniform_stride():
    from tensorrt_llm._torch.disaggregation.resource.utils import get_layer_byte_ranges

    # Dedicated K/V pool: dense layers, uniform stride.
    starts, bytes_per_layer = get_layer_byte_ranges(
        _view([(0, 0, 128), (0, 128, 128), (1, 256, 128), (1, 384, 128)])
    )
    assert starts == {0: 0, 1: 256}
    assert bytes_per_layer == 256


def test_layer_byte_ranges_non_uniform_stride():
    from tensorrt_llm._torch.disaggregation.resource.utils import get_layer_byte_ranges

    # Coalesced slot: another role class interleaves 64B after layer 0,
    # so the layer stride is non-uniform while sizes stay uniform.
    starts, bytes_per_layer = get_layer_byte_ranges(
        _view([(0, 0, 128), (0, 128, 128), (1, 320, 128), (1, 448, 128)])
    )
    assert starts == {0: 0, 1: 320}
    assert bytes_per_layer == 256


def test_layer_byte_ranges_noncontiguous_layer_raises():
    import pytest

    from tensorrt_llm._torch.disaggregation.resource.utils import get_layer_byte_ranges

    with pytest.raises(ValueError, match="not contiguous"):
        get_layer_byte_ranges(_view([(0, 0, 128), (0, 192, 128)]))


def test_layer_byte_ranges_nonuniform_size_raises():
    import pytest

    from tensorrt_llm._torch.disaggregation.resource.utils import get_layer_byte_ranges

    with pytest.raises(ValueError, match="not uniform"):
        get_layer_byte_ranges(_view([(0, 0, 128), (1, 128, 64)]))


def test_layer_byte_ranges_declared_size_mismatch_raises():
    import pytest

    from tensorrt_llm._torch.disaggregation.resource.utils import get_layer_byte_ranges

    with pytest.raises(ValueError, match="declares bytes_per_layer"):
        get_layer_byte_ranges(_view([(0, 0, 128)], bytes_per_layer=256))


def test_layer_byte_ranges_empty_view_raises():
    import pytest

    from tensorrt_llm._torch.disaggregation.resource.utils import get_layer_byte_ranges

    with pytest.raises(ValueError, match="no buffer entries"):
        get_layer_byte_ranges(_view([]))
