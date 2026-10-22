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
"""Receiver-local fragment and writer-region validation."""

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base import CacheKind, Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import HNDHeadMismatchMapper
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import (
    ConvStateMismatchMapper,
    MambaHeadMismatchMapper,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.receive_bounds import ReceiveBounds
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    MAMBA_CONV_ROLE,
    MAMBA_SSM_ROLE,
    AttentionLayerGroup,
    KVCachePageTable,
    MambaLayerGroup,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_layer_byte_ranges

pytestmark = pytest.mark.cpu_only

_PAGED_BASE = 0x10000
_SECOND_POOL_BASE = 0x20000
_STATE_BASE = 0x30000
_WRITER_BASE = 0x50000


def _page_table() -> KVCachePageTable:
    empty_entries = np.array([], dtype=BUFFER_ENTRY_DTYPE)
    state_entries = np.array([(0, 0, 16), (1, 256, 16)], dtype=BUFFER_ENTRY_DTYPE)
    adjacent_role_entries = np.array([(0, 16, 8), (1, 272, 8)], dtype=BUFFER_ENTRY_DTYPE)
    return KVCachePageTable(
        tokens_per_block=32,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                pool_views=[
                    PoolView(pool_idx=0, buffer_entries=empty_entries),
                    PoolView(pool_idx=1, buffer_entries=empty_entries),
                ],
            ),
            MambaLayerGroup(
                pool_group_idx=1,
                pool_views=[
                    PoolView(pool_idx=0, buffer_entries=state_entries, bytes_per_layer=16),
                    PoolView(pool_idx=0, buffer_entries=adjacent_role_entries, bytes_per_layer=8),
                ],
            ),
        ],
        pool_groups=[
            PhysicalPoolGroup(
                pools=[
                    PhysicalPool(
                        base_address=_PAGED_BASE,
                        slot_bytes=128,
                        num_slots=8,
                        slot_stride_bytes=256,
                    ),
                    PhysicalPool(base_address=_SECOND_POOL_BASE, slot_bytes=64, num_slots=8),
                ]
            ),
            PhysicalPoolGroup(
                pools=[
                    PhysicalPool(
                        base_address=_STATE_BASE,
                        slot_bytes=32,
                        num_slots=4,
                        slot_stride_bytes=64,
                        layer_stride_bytes=256,
                    )
                ]
            ),
        ],
    )


def _chunk(paged_ids: tuple[int, ...] = (1, -1, 3), state_ids: tuple[int, ...] = (2,)) -> Chunk:
    return Chunk(
        block_ids_per_layer_groups=[
            np.array(paged_ids, dtype=np.int64),
            np.array(state_ids, dtype=np.int64),
        ],
        kind_per_layer_group=[CacheKind.PAGED, CacheKind.STATE],
        token_range=TokenRange(0, 96),
        is_last=True,
    )


def _grant() -> ReceiveBounds:
    return ReceiveBounds.from_chunk(
        _chunk(), _page_table(), {3: (_WRITER_BASE, 128), 7: (_WRITER_BASE + 128, 128)}
    )


def _mapper_page_table() -> KVCachePageTable:
    table = _page_table()
    table.pool_groups[0].pools = [
        PhysicalPool(base_address=_PAGED_BASE, slot_bytes=256, num_slots=8, slot_stride_bytes=320)
    ]
    attention_view = table.layer_groups[0].pool_views[0]
    attention_view.buffer_entries = np.array(
        [(0, 0, 64), (0, 64, 64), (1, 128, 64), (1, 192, 64)],
        dtype=BUFFER_ENTRY_DTYPE,
    )
    table.layer_groups[0].pool_views = [attention_view]
    conv_view, ssm_view = table.layer_groups[1].pool_views
    conv_view.pool_role = MAMBA_CONV_ROLE
    conv_view.mapper_kind = MapperKind.SECTIONED
    conv_view.section_bytes = [8, 4, 4]
    ssm_view.pool_role = MAMBA_SSM_ROLE
    ssm_view.mapper_kind = MapperKind.INDEXED
    ssm_view.bytes_per_head = 2
    return table


def _rank_info(tp_size: int, tp_rank: int, kv_heads: int) -> RankInfo:
    return RankInfo(
        instance_name="rank",
        instance_rank=tp_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dp_size=1,
        dp_rank=0,
        pp_size=1,
        pp_rank=0,
        cp_size=1,
        cp_rank=0,
        device_id=0,
        layer_num_per_pp=[2],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        attention=AttentionInfo(
            kv_heads_per_rank=kv_heads,
            tokens_per_block=32,
            dims_per_head=1,
            element_bytes=1,
            enable_attention_dp=False,
            is_mla=False,
        ),
        aux_meta=None,
    )


def _assert_mapper_destinations_accepted(table: KVCachePageTable, pairs) -> None:
    dst_ptrs = np.concatenate([pair.dst.memory.ptrs for pair in pairs])
    sizes = np.concatenate(
        [
            np.full(pair.dst.memory.ptrs.size, pair.dst.memory.bytes_per_region, dtype=np.int64)
            for pair in pairs
        ]
    )
    bounds = ReceiveBounds.from_chunk(_chunk(), table, {3: (_WRITER_BASE, 512)})
    checked_ptrs, checked_sizes, source = bounds.validate_result(
        3, True, dst_ptrs, sizes, _WRITER_BASE
    )
    np.testing.assert_array_equal(checked_ptrs, dst_ptrs)
    np.testing.assert_array_equal(checked_sizes, sizes)
    assert source == _WRITER_BASE


def test_accepts_hnd_mapper_destinations_for_tp_fan_in() -> None:
    table = _mapper_page_table()
    extractor = KVRegionExtractorV1(table)
    dst = extractor.extract(_chunk().block_ids_per_layer_groups[0], 0, 0)
    src = SpecRegion(
        memory=MemRegionGroup(
            ptrs=np.array([0x70000, 0x70400], dtype=np.int64), bytes_per_region=128
        )
    )
    dst_offsets, dst_bytes_per_layer = get_layer_byte_ranges(table.layer_groups[0].pool_views[0])
    mapper = HNDHeadMismatchMapper(
        src_layer_offsets=[0, 64],
        dst_layer_offsets=sorted(dst_offsets.values()),
        self_ri=_rank_info(2, 1, 1),
        peer_ri=_rank_info(1, 0, 2),
        self_bytes_per_layer=64,
        peer_bytes_per_layer=dst_bytes_per_layer,
        self_buffers_per_layer=2,
        peer_buffers_per_layer=2,
    )

    pair = mapper.map(src, dst)
    assert pair.dst.memory.ptrs.size == 8
    assert pair.dst.memory.bytes_per_region == 32
    _assert_mapper_destinations_accepted(table, [pair])


def test_accepts_mamba_sectioned_and_ssm_mapper_destinations() -> None:
    table = _mapper_page_table()
    extractor = KVRegionExtractorV1(table)
    dst_conv = extractor.extract_slot(2, 1, 0)
    dst_ssm = extractor.extract_slot(2, 1, 1)
    src_conv = SpecRegion(
        memory=MemRegionGroup(ptrs=np.array([0x80000, 0x80100], dtype=np.int64), bytes_per_region=8)
    )
    src_ssm = SpecRegion(
        memory=MemRegionGroup(ptrs=np.array([0x90000, 0x90100], dtype=np.int64), bytes_per_region=4)
    )
    conv_mapper = ConvStateMismatchMapper(
        transfer_layers=2,
        src_layer_off=0,
        dst_layer_off=0,
        self_section_bytes=[4, 2, 2],
        peer_section_bytes=[8, 4, 4],
        self_tp_per_dp=2,
        peer_tp_per_dp=1,
        self_tp_rank=1,
        peer_tp_rank=0,
    )
    ssm_mapper = MambaHeadMismatchMapper(
        transfer_layers=2,
        src_layer_off=0,
        dst_layer_off=0,
        bytes_per_head=2,
        self_nheads=2,
        peer_nheads=4,
        self_tp_per_dp=2,
        peer_tp_per_dp=1,
        self_tp_rank=1,
        peer_tp_rank=0,
    )

    conv_pairs = conv_mapper.map(src_conv, dst_conv)
    ssm_pair = ssm_mapper.map(src_ssm, dst_ssm)
    assert len(conv_pairs) == 3
    assert ssm_pair.dst.memory.ptrs.size == 2
    _assert_mapper_destinations_accepted(table, [*conv_pairs, ssm_pair])


def _validate(grant: ReceiveBounds, addresses: list[int], sizes: list[int], *, rank: int = 3):
    return grant.validate_result(
        rank,
        True,
        np.array(addresses, dtype=np.int64),
        np.array(sizes, dtype=np.int64),
        _WRITER_BASE if rank == 3 else _WRITER_BASE + 128,
    )


def test_accepts_paged_pools_and_per_layer_state_fragments() -> None:
    grant = _grant()
    addresses = [
        _PAGED_BASE + 256 + 120,
        _SECOND_POOL_BASE + 3 * 64,
        _STATE_BASE + 2 * 64,
        _STATE_BASE + 2 * 64 + 256 + 16,
    ]
    ptrs, sizes, source = _validate(grant, addresses, [8, 64, 16, 8])

    assert ptrs.tolist() == addresses
    assert sizes.tolist() == [8, 64, 16, 8]
    assert source == _WRITER_BASE
    assert not ptrs.flags.writeable and not sizes.flags.writeable


@pytest.mark.parametrize(
    "address,size",
    [
        (_PAGED_BASE + 2 * 256, 8),  # The positional block table has a hole here.
        (_SECOND_POOL_BASE + 2 * 64, 8),
        (_STATE_BASE + 1 * 64, 8),
        (_STATE_BASE + 2 * 64 + 24, 8),
        (_STATE_BASE + 2 * 64 + 256 + 24, 8),
    ],
)
def test_rejects_regions_outside_the_piece(address: int, size: int) -> None:
    with pytest.raises(ValueError, match="outside this receive piece"):
        _validate(_grant(), [address], [size])


def test_rejects_fragment_running_past_the_end_of_its_slot() -> None:
    with pytest.raises(ValueError, match="runs past its end"):
        _validate(_grant(), [_PAGED_BASE + 256 + 120], [9])


def test_one_fragment_cannot_span_adjacent_slots() -> None:
    """Touching slots are kept separate on purpose: a fragment belongs to exactly one slot."""
    grant = ReceiveBounds.from_chunk(
        _chunk(paged_ids=(1, 2), state_ids=()),
        _page_table(),
        {3: (_WRITER_BASE, 128)},
    )
    with pytest.raises(ValueError, match="may not span adjacent slots"):
        _validate(grant, [_SECOND_POOL_BASE + 64 + 60], [8])


def test_uses_the_writer_specific_source_and_capacity() -> None:
    grant = _grant()
    with pytest.raises(ValueError, match="different bounce source"):
        grant.validate_result(
            3,
            True,
            np.array([_PAGED_BASE + 256], dtype=np.int64),
            np.array([8], dtype=np.int64),
            _WRITER_BASE + 128,
        )
    with pytest.raises(ValueError, match="exceed"):
        _validate(grant, [_PAGED_BASE + 256, _PAGED_BASE + 3 * 256], [80, 80])
    with pytest.raises(ValueError, match="not assigned"):
        _validate(grant, [_PAGED_BASE + 256], [8], rank=8)


def test_alternative_writers_may_share_a_single_reserved_region() -> None:
    grant = ReceiveBounds.from_chunk(
        _chunk(),
        _page_table(),
        {3: (_WRITER_BASE, 128), 7: (_WRITER_BASE, 128)},
    )
    ptrs, sizes, source = grant.validate_result(
        7,
        True,
        np.array([_PAGED_BASE + 256], dtype=np.int64),
        np.array([8], dtype=np.int64),
        _WRITER_BASE,
    )
    assert ptrs.tolist() == [_PAGED_BASE + 256]
    assert sizes.tolist() == [8]
    assert source == _WRITER_BASE


def test_accepts_no_tail_for_bounced_fallback_and_non_bounced_writer() -> None:
    grant = _grant()
    assert grant.validate_result(3, True, None, None, None) == (None, None, None)
    assert grant.validate_result(3, False, None, None, None) == (None, None, None)

    no_bounce = ReceiveBounds.from_chunk(_chunk(), None, {5: None})
    assert no_bounce.validate_result(5, True, None, None, None) == (None, None, None)
    no_bounce_with_table = ReceiveBounds.from_chunk(
        _chunk(), KVCachePageTable(tokens_per_block=32, layer_groups=[], pool_groups=[]), {5: None}
    )
    assert no_bounce_with_table.validate_result(5, True, None, None, None) == (None, None, None)
    with pytest.raises(ValueError, match="without a local receive region"):
        no_bounce.validate_result(
            5,
            True,
            np.array([_PAGED_BASE + 256], dtype=np.int64),
            np.array([8], dtype=np.int64),
            _WRITER_BASE,
        )
    with pytest.raises(ValueError, match="local page table"):
        ReceiveBounds.from_chunk(_chunk(), None, {3: (_WRITER_BASE, 128)})


@pytest.mark.parametrize(
    "ptrs,sizes,source",
    [
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64), _WRITER_BASE),
        (np.array([_PAGED_BASE + 256], dtype=np.int64), np.array([], dtype=np.int64), _WRITER_BASE),
        (
            np.array([_PAGED_BASE + 256], dtype=np.float64),
            np.array([8], dtype=np.int64),
            _WRITER_BASE,
        ),
        (
            np.array([[_PAGED_BASE + 256]], dtype=np.int64),
            np.array([8], dtype=np.int64),
            _WRITER_BASE,
        ),
        (np.array([-1], dtype=np.int64), np.array([8], dtype=np.int64), _WRITER_BASE),
        (
            np.array([_PAGED_BASE + 256], dtype=np.int64),
            np.array([0], dtype=np.int64),
            _WRITER_BASE,
        ),
        (
            np.array([_PAGED_BASE + 256], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            _WRITER_BASE,
        ),
        (np.array([_PAGED_BASE + 256], dtype=np.int64), None, _WRITER_BASE),
    ],
)
def test_rejects_malformed_fragment_tables(ptrs, sizes, source) -> None:
    with pytest.raises(ValueError):
        _grant().validate_result(3, True, ptrs, sizes, source)


def test_rejects_tail_on_failure() -> None:
    with pytest.raises(ValueError, match="unexpected bounce tail"):
        _grant().validate_result(
            3,
            False,
            np.array([_PAGED_BASE + 256], dtype=np.int64),
            np.array([8], dtype=np.int64),
            _WRITER_BASE,
        )


def test_snapshots_arrays_before_async_scatter() -> None:
    grant = _grant()
    addresses = np.array([_PAGED_BASE + 256], dtype=np.int64)
    sizes = np.array([8], dtype=np.int64)
    checked_ptrs, checked_sizes, _ = grant.validate_result(3, True, addresses, sizes, _WRITER_BASE)
    addresses[0] = 0
    sizes[0] = 0
    assert checked_ptrs.tolist() == [_PAGED_BASE + 256]
    assert checked_sizes.tolist() == [8]


@pytest.mark.parametrize(
    "paged_ids,state_ids",
    [((8,), ()), ((-2,), ()), ((1,), (4,)), ((1,), (-2,)), ((1,), (1, 2))],
)
def test_rejects_invalid_local_slot_ids(paged_ids, state_ids) -> None:
    with pytest.raises(ValueError):
        ReceiveBounds.from_chunk(
            _chunk(paged_ids, state_ids), _page_table(), {3: (_WRITER_BASE, 128)}
        )


def test_checks_large_totals_without_signed_integer_wraparound() -> None:
    limit = (1 << 63) - 1
    table = KVCachePageTable(
        tokens_per_block=1,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                pool_views=[
                    PoolView(pool_idx=0, buffer_entries=np.array([], dtype=BUFFER_ENTRY_DTYPE))
                ],
            )
        ],
        pool_groups=[PhysicalPoolGroup(pools=[PhysicalPool(0, limit, 1)])],
    )
    chunk = Chunk([np.array([0], dtype=np.int64)], [CacheKind.PAGED], TokenRange(0, 1), True)
    grant = ReceiveBounds.from_chunk(chunk, table, {3: (0, limit)})
    with pytest.raises(ValueError, match="exceed"):
        grant.validate_result(
            3,
            True,
            np.array([0, 0], dtype=np.int64),
            np.array([1 << 62, 1 << 62], dtype=np.int64),
            0,
        )


def test_accepts_the_highest_signed_address_boundary() -> None:
    start = (1 << 63) - 8
    table = KVCachePageTable(
        tokens_per_block=1,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                pool_views=[
                    PoolView(pool_idx=0, buffer_entries=np.array([], dtype=BUFFER_ENTRY_DTYPE))
                ],
            )
        ],
        pool_groups=[PhysicalPoolGroup(pools=[PhysicalPool(start, 8, 1)])],
    )
    chunk = Chunk([np.array([0], dtype=np.int64)], [CacheKind.PAGED], TokenRange(0, 1), True)
    grant = ReceiveBounds.from_chunk(chunk, table, {3: (start, 8)})
    ptrs, sizes, source = grant.validate_result(
        3, True, np.array([start + 4], dtype=np.int64), np.array([4], dtype=np.int64), start
    )
    assert ptrs.tolist() == [start + 4]
    assert sizes.tolist() == [4]
    assert source == start
