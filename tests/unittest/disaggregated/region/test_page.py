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

from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.cache_specs import (
    BufferDecl,
    LayerGroup,
    PageTable,
    Pool,
    Region,
)

pytestmark = pytest.mark.cpu_only


def _hnd(nbytes=128, heads=8):
    return BufferDecl(nbytes, (("head", heads),), "head")


def _kv_regions(layer, offset, local_layer=0, pool=0, nbytes=128):
    """K then V of one layer, back to back from ``offset``."""
    return [
        Region(pool, layer, "key", offset, (_hnd(nbytes),), local_layer=local_layer),
        Region(pool, layer, "value", offset + nbytes, (_hnd(nbytes),), local_layer=local_layer),
    ]


def _attention_layer_group(regions, window=None, tokens_per_block=64):
    return LayerGroup(
        spec="attention",
        kind=CacheKind.PAGED,
        regions=regions,
        params={"window": window},
    )


def test_physical_pool_construction():
    pool = Pool(base_address=0x10000, slot_bytes=256, num_slots=4)
    assert pool.base_address == 0x10000
    assert pool.slot_bytes == 256
    assert pool.num_slots == 4
    assert pool.slot_stride_bytes == 256


def test_physical_pool_roundtrip():
    pool = Pool(
        base_address=0x10000,
        slot_bytes=256,
        num_slots=4,
        slot_stride_bytes=2048,
    )
    d = pool.to_dict()
    restored = Pool.from_dict(d)
    assert restored.base_address == pool.base_address
    assert restored.slot_bytes == pool.slot_bytes
    assert restored.num_slots == pool.num_slots
    assert restored.slot_stride_bytes == pool.slot_stride_bytes


def test_physical_pool_rejects_stride_smaller_than_slot():
    with pytest.raises(ValueError, match="slot_stride_bytes"):
        Pool(base_address=0, slot_bytes=256, num_slots=4, slot_stride_bytes=128)


def test_region_roundtrip():
    """Each (layer, role) buffer is one region; its pool, offset and global layer id survive.

    The manager's own layer id stays on this rank.
    """
    layer_group = _attention_layer_group(_kv_regions(layer=5, offset=0, local_layer=0))
    restored = LayerGroup.from_dict(layer_group.to_dict())
    assert restored.regions == layer_group.regions
    key, value = restored.regions
    assert (key.pool, key.layer, key.role, key.offset, key.local_layer) == (0, 5, "key", 0, -1)
    assert (value.pool, value.layer, value.role, value.offset) == (0, 5, "value", 128)
    assert key.key == (5, "key")
    assert key.nbytes == value.nbytes == 128


@pytest.mark.parametrize(
    "buffers",
    [
        pytest.param((BufferDecl(128),), id="replicated"),
        pytest.param((BufferDecl(128, (("token", 4), ("head", 8)), "head"),), id="nhd"),
        pytest.param((_hnd(),), id="hnd"),
    ],
)
def test_region_layout_roundtrip(buffers):
    """Replicated / token-major / head-major declarations survive serialization."""
    layer_group = _attention_layer_group([Region(2, 0, "index_key", 64, buffers)])
    restored = LayerGroup.from_dict(layer_group.to_dict()).regions[0]
    assert restored.buffers == buffers
    assert restored.role == "index_key"
    assert restored.replicated == all(b.split_axis is None for b in buffers)


def test_layer_group_roundtrip():
    layer_group = _attention_layer_group(
        _kv_regions(layer=5, offset=0, local_layer=0) + _kv_regions(6, 256, local_layer=1),
        window=512,
    )
    restored = LayerGroup.from_dict(layer_group.to_dict())
    assert restored.spec == "attention"
    assert restored.kind == CacheKind.PAGED
    assert restored.params == {"window": 512}
    assert restored.layers == [5, 6]
    assert restored.pools == [0]
    assert len(restored.regions) == 4


def test_layer_group_sends_each_distinct_layout_once():
    regions = _kv_regions(layer=0, offset=0) + [Region(0, 0, "index_key", 256, (BufferDecl(64),))]
    d = _attention_layer_group(regions).to_dict()
    # K and V share one declaration; the replicated index key has its own.
    assert len(d["layouts"]) == 2
    assert [row[4] for row in d["regions"]] == [0, 0, 1]


def test_kv_cache_page_table_roundtrip():
    page_table = PageTable(
        pools=[Pool(base_address=0x10000, slot_bytes=256, num_slots=4)],
        layer_groups=[_attention_layer_group(_kv_regions(layer=5, offset=0))],
    )
    d = page_table.to_dict()
    restored = PageTable.from_dict(d)
    assert len(restored.layer_groups) == 1
    assert len(restored.pools) == 1
    assert restored.pools[0].base_address == 0x10000
    assert restored.layer_groups[0].params == {"window": None}
    assert restored.pool_of(restored.layer_groups[0].regions[0]) is restored.pools[0]


# ---------------------------------------------------------------------------
# Region and buffer declaration geometry
# ---------------------------------------------------------------------------


def test_region_nbytes_sums_buffers():
    sections = tuple(BufferDecl(s, (("byte", s),), "byte") for s in (512, 256, 256))
    region = Region(0, 3, "conv_state", 0, sections)
    assert region.nbytes == 1024
    assert not region.replicated


def test_buffer_decl_rows_and_units():
    # NHD: the head axis repeats once per token.
    nhd = BufferDecl(256, (("token", 4), ("head", 8)), "head")
    assert (nhd.rows, nhd.units, nhd.unit_bytes("nhd")) == (4, 8, 8)
    # HND: one contiguous head-major slice.
    hnd = _hnd(nbytes=256, heads=8)
    assert (hnd.rows, hnd.units, hnd.unit_bytes("hnd")) == (1, 8, 32)
    # Replicated: a single unit.
    replicated = BufferDecl(256)
    assert (replicated.rows, replicated.units) == (1, 1)


def test_buffer_decl_unit_not_byte_aligned_raises():
    decl = BufferDecl(100, (("head", 8),), "head")
    with pytest.raises(ValueError, match="not evenly divisible"):
        decl.unit_bytes("odd")


def test_buffer_decl_split_axis_must_be_an_axis():
    with pytest.raises(ValueError, match="split axis 'head' is not one of the axes"):
        BufferDecl(128, (("token", 4),), "head")


def test_buffer_decl_empty_raises():
    with pytest.raises(ValueError, match="at least one byte"):
        BufferDecl(0)


def test_region_negative_offset_raises():
    with pytest.raises(ValueError, match="negative offset"):
        Region(0, 0, "key", -1, (_hnd(),))


def test_region_without_buffers_raises():
    with pytest.raises(ValueError, match="declares no buffers"):
        Region(0, 0, "key", 0, ())
