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

__extra_import_path__ = [".."]
from disaggregated.pages_test_utils import Segment, buf, group, head_major, pages, pool, sectioned

from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.page_mapping import RankPages, replicated
from tensorrt_llm._torch.pyexecutor.kv_cache.layout_builder import DevicePool
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import BufferDesc

pytestmark = pytest.mark.cpu_only


def _kv_buffers(layer, offset, pool_index=0, nbytes=128):
    """K then V of one layer, back to back from ``offset``."""
    return [
        buf(pool_index, layer, "key", offset, head_major(nbytes, 8)),
        buf(pool_index, layer, "value", offset + nbytes, head_major(nbytes, 8)),
    ]


def _round_trip(rank_pages: RankPages) -> RankPages:
    return RankPages.from_dict(rank_pages.to_dict())


def test_device_pool_construction():
    device_pool = DevicePool(group=0, index=1, base=0x10000, slot_bytes=256, num_slots=4)
    assert (device_pool.group, device_pool.index) == (0, 1)
    assert device_pool.base == 0x10000
    assert device_pool.slot_bytes == 256
    assert device_pool.num_slots == 4


def test_device_pool_roundtrip():
    rank_pages = pages([pool(0x10000, 256, num_slots=4)], [group("attention", _kv_buffers(5, 0))])
    restored = _round_trip(rank_pages)
    assert restored.pools == rank_pages.pools
    assert restored.pool((0, 0)) == DevicePool(0, 0, 0x10000, 256, 4)


def test_buffer_roundtrip():
    """Each (layer, role) buffer is one buffer; its pool, offset and global layer id survive."""
    rank_pages = pages([pool(0x10000, 256)], [group("attention", _kv_buffers(layer=5, offset=0))])
    restored = _round_trip(rank_pages)
    assert restored.buffers(0) == rank_pages.buffers(0)
    key, value = restored.buffers(0)
    assert (key.pool_group, key.pool_index, key.layer, key.role, key.slot_offset) == (
        0,
        0,
        5,
        "key",
        0,
    )
    assert (value.layer, value.role, value.slot_offset) == (5, "value", 128)
    assert key.size == value.size == 128


@pytest.mark.parametrize(
    "segments",
    [
        pytest.param((Segment(128),), id="replicated"),
        pytest.param((Segment(128, (("token", 4), ("head", 8)), "head"),), id="nhd"),
        pytest.param(head_major(128, 8), id="hnd"),
    ],
)
def test_buffer_segments_roundtrip(segments):
    """Replicated / token-major / head-major segments survive serialization."""
    rank_pages = pages(
        [pool(0x10000, 256)], [group("attention", [buf(0, 0, "index_key", 64, segments)])]
    )
    (restored,) = _round_trip(rank_pages).buffers(0)
    assert restored.segments == segments
    assert restored.role == "index_key"
    assert replicated(restored) == all(s.split_axis is None for s in segments)


def test_layer_group_roundtrip():
    rank_pages = pages(
        [pool(0x10000, 512)],
        [group("attention", _kv_buffers(5, 0) + _kv_buffers(6, 256), window=512)],
    )
    restored = _round_trip(rank_pages)
    assert restored.kind(0) == CacheKind.PAGED
    assert restored.group(0).kept_window == 512
    assert restored.group(0).layers == (5, 6)
    assert [(p.group, p.index) for p in restored.group_pools(0)] == [(0, 0)]
    assert len(restored.buffers(0)) == 4
    assert restored.layout == rank_pages.layout


def test_rank_pages_send_each_distinct_layout_once():
    buffers = _kv_buffers(layer=0, offset=0) + [buf(0, 0, "index_key", 256, (Segment(64),))]
    d = pages([pool(0x10000, 320)], [group("attention", buffers)]).to_dict()
    # K and V share one layout; the replicated index key has its own.
    assert len(d["segments"]) == 2
    assert [row[6] for row in d["buffers"]] == [0, 0, 1]


def test_rank_pages_roundtrip():
    rank_pages = pages([pool(0x10000, 256, num_slots=4)], [group("attention", _kv_buffers(5, 0))])
    restored = _round_trip(rank_pages)
    assert restored.num_layer_groups == 1
    assert len(restored.pools) == 1
    assert restored.pools[0].base == 0x10000
    assert restored.group(0).kept_window is None
    key = restored.buffers(0)[0]
    assert restored.pool((key.pool_group, key.pool_index)) is restored.pools[0]


def test_untransferred_buffer_keeps_its_place_but_is_not_listed():
    buffers = _kv_buffers(layer=0, offset=0) + [
        buf(0, 0, "scratch", 256, (Segment(64),), transferred=False)
    ]
    rank_pages = _round_trip(pages([pool(0x10000, 320)], [group("attention", buffers)]))
    assert [b.role for b in rank_pages.buffers(0)] == ["key", "value"]
    assert [b.role for b in rank_pages.layout.buffers] == ["key", "value", "scratch"]


# ---------------------------------------------------------------------------
# Buffer and segment geometry
# ---------------------------------------------------------------------------


def test_buffer_size_sums_segments():
    sections = sectioned((512, 256, 256))
    buffer = BufferDesc(3, "conv_state", 0, 0, 0, 1024, sections)
    assert buffer.size == sum(s.nbytes for s in buffer.segments) == 1024
    assert not replicated(buffer)


def test_segment_rows_and_units():
    # NHD: the head axis repeats once per token.
    nhd = Segment(256, (("token", 4), ("head", 8)), "head")
    assert (nhd.rows, nhd.split_extent, nhd.unit_bytes) == (4, 8, 8)
    # HND: one contiguous head-major slice.
    (hnd,) = head_major(256, 8)
    assert (hnd.rows, hnd.split_extent, hnd.unit_bytes) == (1, 8, 32)
    # Replicated: a single unit, the whole segment.
    whole = Segment(256)
    assert (whole.rows, whole.split_extent, whole.unit_bytes) == (1, None, 256)


def test_segment_unit_not_byte_aligned_raises():
    with pytest.raises(ValueError, match="not a whole number"):
        Segment(100, (("head", 8),), "head")


def test_segment_split_axis_must_be_an_axis():
    with pytest.raises(ValueError, match="split axis 'head' is not one of the axes"):
        Segment(128, (("token", 4),), "head")


def test_segment_empty_raises():
    with pytest.raises(ValueError, match="segment size must be a positive integer"):
        Segment(0)


def test_buffer_negative_offset_raises():
    with pytest.raises(ValueError, match="slot_offset must be a non-negative integer"):
        BufferDesc(0, "key", 0, 0, -1, 128, head_major(128, 8))


def test_buffer_without_segments_is_one_replicated_segment():
    buffer = BufferDesc(0, "key", 0, 0, 0, 128)
    assert buffer.segments == (Segment(128),)
    assert replicated(buffer)


def test_buffer_segments_must_make_up_its_size():
    with pytest.raises(ValueError, match="do not make up a 256-byte buffer"):
        BufferDesc(0, "key", 0, 0, 0, 256, head_major(128, 8))
