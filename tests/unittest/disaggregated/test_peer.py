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

from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.native.peer as peer_module
from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    AuxBufferMeta,
    build_aux_transfer_layout,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import (
    AttentionPolicy,
    HNDHeadMismatchMapper,
    IntactMapper,
    NHDHeadMismatchMapper,
    ReplicatedMapper,
    RingMapper,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import (
    ConvStateMismatchMapper,
    MambaHeadMatchMapper,
    MambaPolicy,
)
from tensorrt_llm._torch.disaggregation.native.peer import PeerOverlap, PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    MAMBA_CONV_ROLE,
    MAMBA_SSM_ROLE,
    AttentionLayerGroup,
    CacheKind,
    KVCachePageTable,
    LocalLayer,
    MambaLayerGroup,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)

pytestmark = pytest.mark.cpu_only


def make_page_table(
    pool_ptrs=None, block_bytes=None, global_layer_ids=None, mamba_tp=2, kv_heads=2
):
    """Create a KVCachePageTable for testing.

    ``mamba_tp`` is the TP degree the synthetic recurrent-state group is
    sharded for: per-rank sizes are the ``mamba_tp`` shard of a fixed global
    state, so page tables built for different TP degrees stay
    peer-compatible under MambaPolicy's global (TP-aggregated) size check.
    Pass the rank's effective mamba TP (``make_rankinfo`` defaults to
    ``tp_size=2``).
    """
    if pool_ptrs is None:
        pool_ptrs = [1234]
    if block_bytes is None:
        block_bytes = [1024]
    if global_layer_ids is None:
        global_layer_ids = [0, 1]

    local_layers = [
        LocalLayer(local_layer_id=i, global_layer_id=gid) for i, gid in enumerate(global_layer_ids)
    ]
    # Build buffer entries: K + V per local layer, sized so the layers fill
    # the pool slot (keeps entry geometry consistent with slot_bytes).
    pool_views = []
    for pi, bs in enumerate(block_bytes):
        buffer_size = bs // (len(global_layer_ids) * 2)
        entries = []
        for i in range(len(global_layer_ids)):
            base_offset = i * buffer_size * 2
            entries.append((i, base_offset, buffer_size))
            entries.append((i, base_offset + buffer_size, buffer_size))
        pool_views.append(
            PoolView(
                pool_idx=pi,
                buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE),
                num_kv_heads=kv_heads,
            )
        )
    physical_pools = [
        PhysicalPool(base_address=ptr, slot_bytes=bs, num_slots=128)
        for ptr, bs in zip(pool_ptrs, block_bytes)
    ]

    attn_lg = AttentionLayerGroup(
        pool_group_idx=0,
        sliding_window_size=None,
        local_layers=local_layers,
        pool_views=pool_views,
    )
    conv_slot_bytes = 4096 // mamba_tp
    ssm_slot_bytes = 8192 // mamba_tp
    sorted_lids = [0, 1]
    mamba_local_layers = [
        LocalLayer(local_layer_id=0, global_layer_id=100),
        LocalLayer(local_layer_id=1, global_layer_id=101),
    ]
    mamba_pool_views = [
        PoolView(
            pool_idx=0,
            buffer_entries=np.array(
                [(lid, 0, conv_slot_bytes) for lid in sorted_lids], dtype=BUFFER_ENTRY_DTYPE
            ),
            pool_role=MAMBA_CONV_ROLE,
            mapper_kind=MapperKind.SECTIONED,
            bytes_per_layer=conv_slot_bytes,
            section_bytes=[1024 // mamba_tp, 512 // mamba_tp, 512 // mamba_tp],
        ),
        PoolView(
            pool_idx=1,
            buffer_entries=np.array(
                [(lid, 0, ssm_slot_bytes) for lid in sorted_lids], dtype=BUFFER_ENTRY_DTYPE
            ),
            pool_role=MAMBA_SSM_ROLE,
            mapper_kind=MapperKind.INDEXED,
            bytes_per_layer=ssm_slot_bytes,
            bytes_per_head=64,
        ),
    ]
    mamba_lg = MambaLayerGroup(
        pool_group_idx=1,
        local_layers=mamba_local_layers,
        pool_views=mamba_pool_views,
    )
    conv_pool = PhysicalPool(base_address=0xA000, slot_bytes=conv_slot_bytes, num_slots=128)
    ssm_pool = PhysicalPool(base_address=0xB000, slot_bytes=ssm_slot_bytes, num_slots=128)
    pool_groups = [
        PhysicalPoolGroup(pools=physical_pools),
        PhysicalPoolGroup(pools=[conv_pool, ssm_pool]),
    ]

    return KVCachePageTable(
        tokens_per_block=16,
        layer_groups=[attn_lg, mamba_lg],
        pool_groups=pool_groups,
    )


def make_rankinfo(
    instance_name="self",
    instance_rank=0,
    tp_size=2,
    tp_rank=0,
    pp_size=1,
    pp_rank=0,
    dp_size=1,
    dp_rank=0,
    cp_size=1,
    cp_rank=0,
    tokens_per_block=16,
    element_bytes=2,
    is_mla=False,
    enable_attention_dp=False,
    layer_num_per_pp=None,
    page_table=None,
):
    if layer_num_per_pp is None:
        layer_num_per_pp = [2] * pp_size
    return RankInfo(
        instance_name=instance_name,
        instance_rank=instance_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=pp_size,
        pp_rank=pp_rank,
        dp_size=dp_size,
        dp_rank=dp_rank,
        cp_size=cp_size,
        cp_rank=cp_rank,
        device_id=0,
        layer_num_per_pp=layer_num_per_pp,
        self_endpoint="",
        transfer_engine_info=b"",
        attention=AttentionInfo(
            tokens_per_block=tokens_per_block,
            element_bytes=element_bytes,
            enable_attention_dp=enable_attention_dp,
            is_mla=is_mla,
        ),
        aux_meta=None,
        page_table=page_table,
        sender_endpoints=[],
    )


def _make_peer_registrar_and_peer_ri(self_ri, peer_ri):
    pt = make_page_table()
    real_kv_extractor = KVRegionExtractorV1(pt)
    reg = PeerRegistrar(self_ri, real_kv_extractor)
    if peer_ri.page_table is None:
        peer_ri = make_rankinfo(
            instance_name=peer_ri.instance_name,
            instance_rank=peer_ri.instance_rank,
            tp_size=peer_ri.tp_size,
            tp_rank=peer_ri.tp_rank,
            pp_size=peer_ri.pp_size,
            pp_rank=peer_ri.pp_rank,
            dp_size=peer_ri.dp_size,
            dp_rank=peer_ri.dp_rank,
            cp_size=peer_ri.cp_size,
            cp_rank=peer_ri.cp_rank,
            tokens_per_block=peer_ri.attention.tokens_per_block,
            element_bytes=peer_ri.attention.element_bytes,
            is_mla=peer_ri.attention.is_mla,
            enable_attention_dp=peer_ri.attention.enable_attention_dp,
            layer_num_per_pp=peer_ri.layer_num_per_pp,
            page_table=make_page_table(),
        )
    return reg, peer_ri


def test_basic_overlap():
    self_ri = make_rankinfo(
        "self",
        pp_size=1,
        pp_rank=0,
        tp_size=1,
        tp_rank=0,
        cp_size=1,
        cp_rank=0,
        layer_num_per_pp=[2],
    )
    peer_ri = make_rankinfo(
        "peer",
        pp_size=2,
        pp_rank=0,
        tp_size=1,
        tp_rank=0,
        cp_size=1,
        cp_rank=0,
        layer_num_per_pp=[1, 1],
    )
    reg, peer_ri = _make_peer_registrar_and_peer_ri(self_ri, peer_ri)
    overlap = reg.get_peer_overlap(peer_ri, peer_dp_rank=0)
    assert overlap.overlap_pp_size == 2
    assert overlap.overlap_tp_size == 1
    assert overlap.ranks == [0, 1]


def test_no_overlap():
    self_ri = make_rankinfo(
        "self",
        pp_size=1,
        pp_rank=0,
        tp_size=1,
        tp_rank=0,
        cp_size=1,
        cp_rank=0,
        layer_num_per_pp=[0],
    )
    peer_ri = make_rankinfo(
        "peer",
        pp_size=1,
        pp_rank=0,
        tp_size=1,
        tp_rank=0,
        cp_size=1,
        cp_rank=0,
        layer_num_per_pp=[2],
    )
    reg, peer_ri = _make_peer_registrar_and_peer_ri(self_ri, peer_ri)
    overlap = reg.get_peer_overlap(peer_ri, 0)
    assert overlap.overlap_pp_size == 0
    assert overlap.ranks == []


def test_pp_ratio_peer_smaller():
    self_ri = make_rankinfo(
        "self",
        pp_size=2,
        pp_rank=1,
        tp_size=1,
        tp_rank=0,
        cp_size=1,
        cp_rank=0,
        layer_num_per_pp=[1, 2],
    )
    peer_ri = make_rankinfo(
        "peer",
        pp_size=1,
        pp_rank=0,
        tp_size=1,
        tp_rank=0,
        cp_size=1,
        cp_rank=0,
        layer_num_per_pp=[3],
    )
    reg, peer_ri = _make_peer_registrar_and_peer_ri(self_ri, peer_ri)
    overlap = reg.get_peer_overlap(peer_ri, 0)
    assert overlap.overlap_pp_size > 0
    assert all(r >= 0 for r in overlap.ranks)


def test_tp_overlap():
    self_ri = make_rankinfo(
        "self", tp_size=2, tp_rank=1, pp_size=1, pp_rank=0, cp_size=1, cp_rank=0
    )
    peer_ri = make_rankinfo(
        "peer", tp_size=4, tp_rank=0, pp_size=1, pp_rank=0, cp_size=1, cp_rank=0
    )
    reg, peer_ri = _make_peer_registrar_and_peer_ri(self_ri, peer_ri)
    overlap = reg.get_peer_overlap(peer_ri, 0)
    assert overlap.overlap_tp_size in [1, 2]
    assert all(isinstance(r, int) for r in overlap.ranks)


def test_cp_overlap():
    self_ri = make_rankinfo(
        "self", cp_size=2, cp_rank=1, pp_size=1, pp_rank=0, tp_size=1, tp_rank=0
    )
    peer_ri = make_rankinfo(
        "peer", cp_size=4, cp_rank=0, pp_size=1, pp_rank=0, tp_size=1, tp_rank=0
    )
    reg, peer_ri = _make_peer_registrar_and_peer_ri(self_ri, peer_ri)
    overlap = reg.get_peer_overlap(peer_ri, 0)
    assert overlap.overlap_cp_size in [1, 2]
    assert all(isinstance(r, int) for r in overlap.ranks)


def test_multiple_overlap():
    self_ri = make_rankinfo(
        "self",
        pp_size=2,
        pp_rank=1,
        tp_size=2,
        tp_rank=1,
        cp_size=2,
        cp_rank=1,
        layer_num_per_pp=[1, 2],
    )
    peer_ri = make_rankinfo(
        "peer",
        pp_size=4,
        pp_rank=2,
        tp_size=4,
        tp_rank=3,
        cp_size=4,
        cp_rank=0,
        layer_num_per_pp=[1, 1, 1, 1],
    )
    reg, peer_ri = _make_peer_registrar_and_peer_ri(self_ri, peer_ri)
    overlap = reg.get_peer_overlap(peer_ri, peer_dp_rank=0)
    expected_overlap = PeerOverlap(
        overlap_pp_size=2,
        overlap_tp_size=2,
        overlap_cp_size=2,
        ranks=[26, 27, 30, 31, 42, 43, 46, 47],
    )
    assert overlap == expected_overlap


def test_peer_overlap_cache_is_keyed_by_peer_dp_rank():
    """Each attention-DP rank of one peer instance gets the overlap of its own DP group."""
    reg = _make_peer_registrar(make_rankinfo(tp_size=1))
    peer_ri = make_rankinfo(instance_name="peer", tp_size=2, dp_size=2, enable_attention_dp=True)

    assert reg.get_peer_overlap(peer_ri, peer_dp_rank=0).ranks == [0]
    assert reg.get_peer_overlap(peer_ri, peer_dp_rank=1).ranks == [1]


def _make_peer_registrar(self_rankinfo):
    pt = self_rankinfo.page_table or make_page_table()
    real_kv_extractor = KVRegionExtractorV1(pt)
    reg = PeerRegistrar(self_rankinfo, real_kv_extractor)
    return reg


def _make_per_view_page_table(head_counts, mapper_kind):
    """Two-token K/V buffers with four bytes per token/head."""
    views = []
    pools = []
    for layer, heads in enumerate(head_counts):
        buffer_bytes = 8 * heads
        views.append(
            PoolView(
                pool_idx=layer,
                buffer_entries=np.array(
                    [(layer, 0, buffer_bytes), (layer, buffer_bytes, buffer_bytes)],
                    dtype=BUFFER_ENTRY_DTYPE,
                ),
                mapper_kind=mapper_kind,
                num_kv_heads=heads,
            )
        )
        pools.append(
            PhysicalPool(
                base_address=0x1000 * (layer + 1), slot_bytes=2 * buffer_bytes, num_slots=8
            )
        )
    return KVCachePageTable(
        tokens_per_block=2,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                sliding_window_size=None,
                local_layers=[LocalLayer(layer, layer) for layer in range(len(head_counts))],
                pool_views=views,
            )
        ],
        pool_groups=[PhysicalPoolGroup(pools=pools)],
    )


@pytest.mark.parametrize("mapper_kind", [MapperKind.INDEXED, MapperKind.NHD])
def test_per_view_heads_select_correct_transfer_bytes(mapper_kind):
    # TP4 -> TP1: the eight-head view shards all ranks, while the two-head
    # view duplicates each head on two ranks. Rank 2 sends its second head.
    self_ri = make_rankinfo(
        tp_size=4,
        tp_rank=2,
        tokens_per_block=2,
        page_table=_make_per_view_page_table([2, 1], mapper_kind),
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        tokens_per_block=2,
        page_table=_make_per_view_page_table([8, 2], mapper_kind),
    )
    reg = _make_peer_registrar(self_ri)
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri, (0, 1), (0, 1))
    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=16)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=32)),
    )

    if mapper_kind == MapperKind.NHD:
        assert isinstance(mapper, NHDHeadMismatchMapper)
        assert pair.src.memory.ptrs.tolist() == [1000, 1004, 1008, 1012]
        assert pair.dst.memory.ptrs.tolist() == [2004, 2012, 2020, 2028]
        assert pair.src.memory.bytes_per_region == 4
    else:
        assert isinstance(mapper, HNDHeadMismatchMapper)
        assert pair.src.memory.ptrs.tolist() == [1000, 1008]
        assert pair.dst.memory.ptrs.tolist() == [2008, 2024]
        assert pair.src.memory.bytes_per_region == 8
    assert pair.dst.memory.bytes_per_region == pair.src.memory.bytes_per_region


@pytest.mark.parametrize("mapper_kind", [MapperKind.INDEXED, MapperKind.NHD])
@pytest.mark.parametrize(
    "view_idx,peer_tp_rank,src_head_idx",
    [
        (0, 0, 0),
        (0, 1, 2),
        (0, 2, 4),
        (0, 3, 6),
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 1),
        (1, 3, 1),
    ],
)
def test_per_view_heads_select_correct_transfer_bytes_with_larger_gen_tp(
    mapper_kind: MapperKind, view_idx: int, peer_tp_rank: int, src_head_idx: int
) -> None:
    # TP1 -> TP4: the eight-head view sends two heads per receiver; the
    # two-head view sends head 0 to ranks 0/1 and head 1 to ranks 2/3.
    self_ri = make_rankinfo(
        tp_size=1,
        tokens_per_block=2,
        page_table=_make_per_view_page_table([8, 2], mapper_kind),
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=4,
        tp_rank=peer_tp_rank,
        tokens_per_block=2,
        page_table=_make_per_view_page_table([2, 1], mapper_kind),
    )
    reg = _make_peer_registrar(self_ri)
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri, (0, view_idx), (0, view_idx))
    src_heads = [8, 2][view_idx]
    dst_heads = [2, 1][view_idx]
    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=16 * src_heads)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=16 * dst_heads)),
    )

    if mapper_kind == MapperKind.NHD:
        assert isinstance(mapper, NHDHeadMismatchMapper)
        # Four bytes per token/head, with a fragment for each K/V token.
        expected_src_ptrs = [1000 + 4 * src_head_idx + 4 * src_heads * i for i in range(4)]
        expected_dst_ptrs = [2000 + 4 * dst_heads * i for i in range(4)]
        expected_fragment_bytes = 4 * dst_heads
    else:
        assert isinstance(mapper, HNDHeadMismatchMapper)
        # Eight bytes per head across both tokens, with one fragment per K/V.
        expected_src_ptrs = [1000 + 8 * src_head_idx, 1000 + 8 * src_heads + 8 * src_head_idx]
        expected_dst_ptrs = [2000, 2000 + 8 * dst_heads]
        expected_fragment_bytes = 8 * dst_heads
    assert pair.src.memory.ptrs.tolist() == expected_src_ptrs
    assert pair.dst.memory.ptrs.tolist() == expected_dst_ptrs
    assert pair.src.memory.bytes_per_region == expected_fragment_bytes
    assert pair.dst.memory.bytes_per_region == expected_fragment_bytes


def test_per_view_head_duplication_elects_senders_independently():
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        tokens_per_block=2,
        page_table=_make_per_view_page_table([8, 2], MapperKind.INDEXED),
    )
    ownership = [[], []]
    for rank in range(4):
        self_ri = make_rankinfo(
            tp_size=4,
            tp_rank=rank,
            tokens_per_block=2,
            page_table=_make_per_view_page_table([2, 1], MapperKind.INDEXED),
        )
        reg = _make_peer_registrar(self_ri)
        policy = AttentionPolicy(self_ri)
        for view, expected_dup in enumerate((1, 2)):
            self_pv = self_ri.page_table.layer_groups[0].pool_views[view]
            peer_pv = peer_ri.page_table.layer_groups[0].pool_views[view]
            assert policy.duplicate_head_factor(peer_ri, self_pv, peer_pv) == expected_dup
            ownership[view].append(reg.should_send_pool(peer_ri, (0, view), (0, view)))

    assert ownership == [[True, True, True, True], [True, False, True, False]]


def test_view_ownership_uses_the_peer_view_of_the_mapped_pair():
    """The peer lists its views in reverse, so each pair crosses view indices.

    Layer 0 (2 heads x TP4 vs 8 heads x TP1) is sent by every rank; layer 1
    (1 head x TP4 vs 2 heads x TP1) duplicates each head on two ranks. The
    peer view at the other index has the other layer's head count.
    """
    peer_page_table = _make_per_view_page_table([8, 2], MapperKind.INDEXED)
    peer_page_table.layer_groups[0].pool_views.reverse()
    peer_ri = make_rankinfo(
        instance_name="peer", tp_size=1, tokens_per_block=2, page_table=peer_page_table
    )
    ownership = {(0, 0): [], (0, 1): []}
    for rank in range(4):
        self_ri = make_rankinfo(
            tp_size=4,
            tp_rank=rank,
            tokens_per_block=2,
            page_table=_make_per_view_page_table([2, 1], MapperKind.INDEXED),
        )
        reg = _make_peer_registrar(self_ri)
        mapping = reg.get_pool_mapping(peer_ri)
        assert mapping == {(0, 0): (0, 1), (0, 1): (0, 0)}
        for self_key, peer_key in mapping.items():
            ownership[self_key].append(reg.should_send_pool(peer_ri, self_key, peer_key))

    assert ownership == {(0, 0): [True, True, True, True], (0, 1): [True, False, True, False]}


@pytest.mark.parametrize(("peer_dp_rank", "expected_senders"), [(0, [0, 2]), (1, [1, 3])])
def test_head_duplication_elects_sender_by_peer_dp_rank(
    peer_dp_rank: int, expected_senders: list[int]
) -> None:
    """TP4 ranks hold one of 2 KV heads; each attention-DP peer rank holds both.

    Ranks 0-1 hold head 0 and ranks 2-3 head 1. The peer has one TP rank per
    DP group, so the factor is 1 x 4 // (2 x 1) = 2, and of each pair the rank
    whose TP rank matches the peer's DP rank modulo 2 sends.
    """
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=2,
        tp_rank=peer_dp_rank,
        dp_size=2,
        dp_rank=peer_dp_rank,
        enable_attention_dp=True,
        tokens_per_block=2,
        page_table=_make_per_view_page_table([2], MapperKind.INDEXED),
    )
    senders = []
    for rank in range(4):
        self_ri = make_rankinfo(
            tp_size=4,
            tp_rank=rank,
            tokens_per_block=2,
            page_table=_make_per_view_page_table([1], MapperKind.INDEXED),
        )
        if _make_peer_registrar(self_ri).should_send_pool(peer_ri, (0, 0), (0, 0)):
            senders.append(rank)

    assert senders == expected_senders


def test_attention_dp_ranks_each_send_their_own_heads() -> None:
    """Each local TP4 attention-DP rank holds all 4 KV heads, as does the peer rank.

    A local DP group has one TP rank, so no two local ranks feed the peer the
    same heads and every rank sends. A factor above 1 would elect only ranks
    whose in-group TP rank (always 0) matches the peer's DP rank 1: none.
    """
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=2,
        tp_rank=1,
        dp_size=2,
        dp_rank=1,
        enable_attention_dp=True,
        tokens_per_block=2,
        page_table=_make_per_view_page_table([4], MapperKind.INDEXED),
    )
    senders = []
    for rank in range(4):
        self_ri = make_rankinfo(
            tp_size=4,
            tp_rank=rank,
            dp_size=4,
            dp_rank=rank,
            enable_attention_dp=True,
            tokens_per_block=2,
            page_table=_make_per_view_page_table([4], MapperKind.INDEXED),
        )
        if _make_peer_registrar(self_ri).should_send_pool(peer_ri, (0, 0), (0, 0)):
            senders.append(rank)

    assert senders == [0, 1, 2, 3]


@pytest.mark.parametrize("mapper_kind", [MapperKind.INDEXED, MapperKind.NHD])
@pytest.mark.parametrize("invalid_heads", [None, 0, -1])
@pytest.mark.parametrize("invalid_sides", [("local",), ("peer",), ("local", "peer")])
def test_per_view_rejects_missing_or_invalid_head_counts(
    mapper_kind: MapperKind, invalid_heads: int | None, invalid_sides: tuple[str, ...]
) -> None:
    self_ri = make_rankinfo(
        tp_size=1,
        tokens_per_block=2,
        page_table=_make_per_view_page_table([2, 1], mapper_kind),
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        tokens_per_block=2,
        page_table=_make_per_view_page_table([2, 1], mapper_kind),
    )
    for side, ri in (("local", self_ri), ("peer", peer_ri)):
        if side in invalid_sides:
            ri.page_table.layer_groups[0].pool_views[0].num_kv_heads = invalid_heads
    reg = _make_peer_registrar(self_ri)

    with pytest.raises(
        ValueError, match=f"the {invalid_sides[0]} view .* must publish a positive num_kv_heads"
    ):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_registration_accepts_replicated_view_without_head_count() -> None:
    """A K/V view beside a REPLICATED index-K view without num_kv_heads, as V2 builds them."""
    self_pt, peer_pt = (
        make_page_table(pool_ptrs=[0x1000, 0x2000], block_bytes=[1024, 512]) for _ in range(2)
    )
    for page_table in (self_pt, peer_pt):
        index_key = page_table.layer_groups[0].pool_views[1]
        index_key.pool_role = frozenset({"index_key"})
        index_key.mapper_kind = MapperKind.REPLICATED
        index_key.num_kv_heads = None
    reg = _make_peer_registrar(make_rankinfo(page_table=self_pt))
    peer_ri = make_rankinfo(instance_name="peer", page_table=peer_pt)

    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    assert reg.get_pool_mapping(peer_ri)[(0, 1)] == (0, 1)


@pytest.mark.parametrize("mapper_kind", [MapperKind.INDEXED, MapperKind.NHD])
def test_view_validation_rejects_unequal_bytes_per_head(mapper_kind: MapperKind) -> None:
    local_ri, peer_ri = [
        make_rankinfo(
            instance_name=name,
            tp_size=1,
            tokens_per_block=2,
            page_table=_make_per_view_page_table([2, 1], mapper_kind),
        )
        for name in ("local", "peer")
    ]
    # Same bytes split over twice the heads: half the head_dim.
    peer_ri.page_table.layer_groups[0].pool_views[1].num_kv_heads = 2
    registrar = _make_peer_registrar(local_ri)

    with pytest.raises(ValueError, match=r"bytes per KV head differ .* \(local=8, peer=4\)"):
        registrar.validate_view_head_geometry(peer_ri)


def test_registration_rejects_equal_heads_with_different_head_bytes() -> None:
    """Both sides hold 2 KV heads per rank; the peer's heads are twice as wide."""
    self_ri = make_rankinfo(page_table=make_page_table(block_bytes=[1024]))
    peer_ri = make_rankinfo(instance_name="peer", page_table=make_page_table(block_bytes=[2048]))
    reg = _make_peer_registrar(self_ri)

    with pytest.raises(ValueError, match=r"bytes per KV head differ .* \(local=128, peer=256\)"):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def _make_receiver_for_view_validation(
    monkeypatch: pytest.MonkeyPatch,
    registrar: PeerRegistrar,
    sender_payloads: dict[str, bytes],
    enforce_physical_ownership: bool,
) -> tuple:
    import tensorrt_llm._torch.disaggregation.native.transfer as transfer_module

    messengers = {}

    def make_messenger(*, mode: str, endpoint: str | None = None) -> Mock:
        messenger = Mock()
        messenger.endpoint = endpoint or "tcp://receiver"
        if endpoint in sender_payloads:
            messenger.receive.return_value = [sender_payloads[endpoint]]
        messengers[endpoint] = messenger
        return messenger

    messenger_factory = Mock(side_effect=make_messenger)
    monkeypatch.setattr(transfer_module, "ZMQMessenger", messenger_factory)
    receiver = transfer_module.Receiver(
        registrar,
        Mock(),
        enforce_physical_ownership=enforce_physical_ownership,
    )
    receiver._build_recv_req_info = Mock()
    receiver._request_sender_data = Mock()
    return receiver, messengers, messenger_factory


@pytest.mark.parametrize("enforce_physical_ownership", [False, True])
@pytest.mark.parametrize(
    ("mismatch", "diagnostic"),
    [
        ("head_bytes", "bytes per KV head differ"),
        ("pool_roles", "has no peer view with the same roles"),
        ("extra_field", "unexpected keyword argument 'kv_heads_per_rank'"),
        ("missing_field", "missing 1 required positional argument"),
        ("missing_key", "KeyError: 'pool_group_idx'"),
    ],
)
def test_receiver_rejects_view_mismatch_before_registration(
    monkeypatch: pytest.MonkeyPatch,
    enforce_physical_ownership: bool,
    mismatch: str,
    diagnostic: str,
) -> None:
    from tensorrt_llm._torch.disaggregation.base import TokenRange
    from tensorrt_llm._torch.disaggregation.native.extent import Chunk
    from tensorrt_llm._torch.disaggregation.native.transfer import (
        KVRecvTask,
        MessageType,
        PeerIncompatibleError,
        TaskStatus,
    )

    local_ri, bad_ri, good_ri = [
        make_rankinfo(
            instance_name=name,
            tp_size=1,
            tokens_per_block=2,
            page_table=_make_per_view_page_table([2, 1], MapperKind.INDEXED),
        )
        for name in ("receiver", "bad-sender", "good-sender")
    ]
    bad_view = bad_ri.page_table.layer_groups[0].pool_views[1]
    if mismatch == "head_bytes":
        # Same bytes split over twice the heads: half the head_dim.
        bad_view.num_kv_heads = 2
    elif mismatch == "pool_roles":
        bad_view.pool_role = frozenset({"other_kv"})
    bad_ri.sender_endpoints = ["tcp://bad-sender"]
    good_ri.sender_endpoints = ["tcp://good-sender"]
    bad_payload = bad_ri.to_bytes()
    if mismatch in ("extra_field", "missing_field", "missing_key"):
        # A peer on another build serializes RankInfo with other fields.
        data = msgpack.unpackb(bad_payload, strict_map_key=False)
        if mismatch == "extra_field":
            data["attention"]["kv_heads_per_rank"] = 2
        elif mismatch == "missing_field":
            del data["attention"]["element_bytes"]
        else:
            del data["page_table"]["layer_groups"][0]["pool_group_idx"]
        bad_payload = msgpack.packb(data)
    bad_endpoint = "tcp://bad-info"
    good_endpoint = "tcp://good-info"
    registrar = _make_peer_registrar(local_ri)
    receiver, messengers, messenger_factory = _make_receiver_for_view_validation(
        monkeypatch,
        registrar,
        {bad_endpoint: bad_payload, good_endpoint: good_ri.to_bytes()},
        enforce_physical_ownership,
    )
    chunk = Chunk(
        block_ids_per_layer_groups=[np.array([], dtype=np.int64)],
        kind_per_layer_group=[CacheKind.PAGED],
        token_range=TokenRange(start=0, end=0),
        is_last=True,
    )

    try:
        for rid in (1, 2):
            params = SimpleNamespace(ctx_info_endpoint=bad_endpoint, ctx_dp_rank=0)
            task = KVRecvTask(rid, chunk, slice_id=0, params=params, aux_slot=None)
            task._enforce_physical_ownership = enforce_physical_ownership
            if enforce_physical_ownership:
                task.begin_publication()

            receiver.dispatch_task(task)

            assert task.status == TaskStatus.ERROR
            assert task.is_done
            assert isinstance(task._exception, PeerIncompatibleError)
            assert diagnostic in str(task._exception)
            if enforce_physical_ownership:
                assert task.resources_drained

        assert bad_endpoint in receiver._incompatible_peers
        assert bad_endpoint not in receiver._sender_ep_instance_map
        assert receiver._dealers == {}
        assert messenger_factory.call_count == 2  # Receiver listener and one info lookup.
        receiver._request_sender_data.assert_not_called()
        messengers[bad_endpoint].send.assert_called_once_with([MessageType.REQUEST_INSTANCE_INFO])
        messengers[bad_endpoint].receive.assert_called_once_with()
        messengers[bad_endpoint].stop.assert_called_once_with()
        with pytest.raises(KeyError):
            registrar.get_peer_rank_info(bad_ri.instance_name, bad_ri.instance_rank)

        # The same receiver can still register a different, compatible sender.
        good_params = SimpleNamespace(ctx_info_endpoint=good_endpoint, ctx_dp_rank=0)
        sender_info = receiver._get_sender_info(good_params)
        assert sender_info.instance_name == good_ri.instance_name
        assert registrar.get_pool_mapping(sender_info) == {(0, 0): (0, 0), (0, 1): (0, 1)}
        messengers["tcp://good-sender"].send.assert_called_once_with(
            [MessageType.REGISTER_RANK_INFO, local_ri.to_bytes()]
        )
        assert receiver._get_sender_info(good_params) is sender_info
        assert messenger_factory.call_count == 4  # One more info lookup and one sender dealer.
        messengers[good_endpoint].receive.assert_called_once_with()
        with pytest.raises(KeyError):
            registrar.get_peer_rank_info(good_ri.instance_name, good_ri.instance_rank)
    finally:
        receiver.shutdown()


def test_view_validation_allows_nonoverlapping_pp_stage() -> None:
    local_page_table = _make_per_view_page_table([2, 1], MapperKind.INDEXED)
    local_page_table.layer_groups[0].local_layers = [LocalLayer(0, 2), LocalLayer(1, 3)]
    local_ri = make_rankinfo(
        tp_size=1,
        pp_size=2,
        pp_rank=1,
        instance_rank=1,
        page_table=local_page_table,
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        pp_size=2,
        page_table=_make_per_view_page_table([2, 1], MapperKind.INDEXED),
    )
    registrar = _make_peer_registrar(local_ri)

    registrar.validate_view_head_geometry(peer_ri)

    assert registrar.get_pool_mapping(peer_ri) == {}
    with pytest.raises(KeyError):
        registrar.get_peer_rank_info(peer_ri.instance_name, peer_ri.instance_rank)


def test_mla_hybrid_stage_without_attention_registers() -> None:
    # A PP stage of an MLA hybrid model may hold recurrent layers only: it has
    # no attention view pair to check and pairs its state views alone.
    self_pt = make_page_table()
    self_pt.layer_groups = self_pt.layer_groups[1:]
    self_ri = make_rankinfo(is_mla=True, page_table=self_pt)
    peer_ri = make_rankinfo(instance_name="peer", is_mla=True, page_table=make_page_table())
    reg = _make_peer_registrar(self_ri)

    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    assert reg.get_pool_mapping(peer_ri) == {(0, 0): (1, 0), (0, 1): (1, 1)}


def _make_view_split_page_table(view_layers: list[list[int]]) -> KVCachePageTable:
    """Layers 0-3 in one layer group; view ``i`` holds the K/V of ``view_layers[i]``."""
    views = []
    pools = []
    for pool_idx, layers in enumerate(view_layers):
        entries = [(layer, (2 * i + kv) * 8, 8) for i, layer in enumerate(layers) for kv in (0, 1)]
        views.append(
            PoolView(
                pool_idx=pool_idx,
                buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE),
                num_kv_heads=1,
            )
        )
        pools.append(
            PhysicalPool(
                base_address=0x1000 * (pool_idx + 1), slot_bytes=16 * len(layers), num_slots=8
            )
        )
    return KVCachePageTable(
        tokens_per_block=2,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                local_layers=[LocalLayer(layer, layer) for layer in range(4)],
                pool_views=views,
            )
        ],
        pool_groups=[PhysicalPoolGroup(pools=pools)],
    )


@pytest.mark.parametrize(
    ("self_views", "peer_views", "diagnostic"),
    [
        ([[0, 1], [2, 3]], [[0, 1, 2, 3]], "also holds layers"),
        ([[0, 1, 2, 3]], [[0, 1], [2, 3]], "overlaps several peer views"),
    ],
)
def test_registration_rejects_peers_that_split_views_differently(
    self_views: list[list[int]], peer_views: list[list[int]], diagnostic: str
) -> None:
    # Only one side keeps layers 2-3 in a view of their own, e.g. because it
    # partitions draft layers by owner and the other side does not.
    self_ri = make_rankinfo(
        tp_size=1,
        tokens_per_block=2,
        layer_num_per_pp=[4],
        page_table=_make_view_split_page_table(self_views),
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        tokens_per_block=2,
        layer_num_per_pp=[4],
        page_table=_make_view_split_page_table(peer_views),
    )
    reg = _make_peer_registrar(self_ri)

    with pytest.raises(ValueError, match=diagnostic):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def _make_gemma4_31b_page_table(tp_size: int) -> KVCachePageTable:
    """One sliding and one global layer of Gemma4-31B, BF16, two tokens per block.

    Sliding layers have 16 KV heads x head_dim 256, global layers 4 KV heads x
    head_dim 512. VSWA puts the two layer types in separate layer groups
    (layer group == layer id here). A rank holds ceil(heads / tp_size) heads.
    """
    layer_groups = []
    pool_groups = []
    for layer, total_heads, head_dim in ((0, 16, 256), (1, 4, 512)):
        heads = -(-total_heads // tp_size)
        buffer_bytes = heads * 2 * head_dim * 2  # heads x tokens x head_dim x BF16
        layer_groups.append(
            AttentionLayerGroup(
                pool_group_idx=layer,
                sliding_window_size=1024 if layer == 0 else None,
                local_layers=[LocalLayer(layer, layer)],
                pool_views=[
                    PoolView(
                        pool_idx=0,
                        buffer_entries=np.array(
                            [(layer, 0, buffer_bytes), (layer, buffer_bytes, buffer_bytes)],
                            dtype=BUFFER_ENTRY_DTYPE,
                        ),
                        num_kv_heads=heads,
                    )
                ],
            )
        )
        pool_groups.append(
            PhysicalPoolGroup(
                pools=[
                    PhysicalPool(
                        base_address=0x100000 * (layer + 1),
                        slot_bytes=2 * buffer_bytes,
                        num_slots=8,
                    )
                ]
            )
        )
    return KVCachePageTable(tokens_per_block=2, layer_groups=layer_groups, pool_groups=pool_groups)


def _register_gemma4_31b_ctx_ranks(ctx_tp: int, gen_ri: RankInfo) -> tuple[dict, dict, dict]:
    """Register ``gen_ri`` on every rank of a ctx instance at TP ``ctx_tp``.

    Returns the registrars by ctx TP rank, the ctx ranks that send each layer
    group to ``gen_ri`` (it is one of their overlap targets and they win the
    election), and each layer group's duplicate head factor.
    """
    registrars, senders, dup = {}, {0: [], 1: []}, {}
    for ctx_tp_rank in range(ctx_tp):
        ctx_ri = make_rankinfo(
            instance_name="ctx",
            instance_rank=ctx_tp_rank,
            tp_size=ctx_tp,
            tp_rank=ctx_tp_rank,
            tokens_per_block=2,
            page_table=_make_gemma4_31b_page_table(ctx_tp),
        )
        reg = _make_peer_registrar(ctx_ri)
        reg.register(gen_ri.instance_name, gen_ri.instance_rank, gen_ri)
        assert reg.get_pool_mapping(gen_ri) == {(0, 0): (0, 0), (1, 0): (1, 0)}
        registrars[ctx_tp_rank] = reg
        targets = reg.get_peer_overlap(gen_ri, gen_ri.dp_rank).ranks
        for lg in (0, 1):
            dup[lg] = AttentionPolicy(ctx_ri).duplicate_head_factor(
                gen_ri,
                ctx_ri.page_table.layer_groups[lg].pool_views[0],
                gen_ri.page_table.layer_groups[lg].pool_views[0],
            )
            if gen_ri.tp_rank in targets and reg.should_send_pool(gen_ri, (lg, 0), (lg, 0)):
                senders[lg].append(ctx_tp_rank)
    return registrars, senders, dup


def _region(ptr: int, nbytes: int) -> SpecRegion:
    return SpecRegion(memory=MemRegionGroup(ptrs=np.array([ptr]), bytes_per_region=nbytes))


@pytest.mark.parametrize("gen_tp_rank", [1, 2, 5, 6])
def test_gemma4_31b_ctx_tp4_to_gen_tp8_maps_each_layer_group_by_its_heads(gen_tp_rank: int):
    """Sliding views slice heads; global views copy one head that TP8 duplicates.

    Sliding: 16 heads give 4 per ctx rank and 2 per gen rank, so gen rank r
    holds heads [2r, 2r + 2), i.e. heads 2 * (r % 2) and up of ctx rank r // 2.
    Global: 4 heads give 1 per rank on both sides; gen rank r holds head
    r * 4 // 8 = r // 2, the one head of ctx rank r // 2, and shares it with
    the other gen rank of that pair. Both views therefore have exactly one
    sender, ctx rank r // 2, and no ctx rank shares a head with another.
    """
    ctx_tp, gen_tp = 4, 8
    sliding_ctx_heads, sliding_gen_heads = 16 // ctx_tp, 16 // gen_tp
    sliding_head_bytes = 2 * 256 * 2  # tokens x head_dim x BF16
    global_layer_bytes = 2 * (2 * 512 * 2)  # K and V of one head
    first_sliding_head = gen_tp_rank * sliding_gen_heads
    expected_sender = first_sliding_head // sliding_ctx_heads
    src_head_off = (first_sliding_head % sliding_ctx_heads) * sliding_head_bytes

    gen_ri = make_rankinfo(
        instance_name="gen",
        instance_rank=gen_tp_rank,
        tp_size=gen_tp,
        tp_rank=gen_tp_rank,
        tokens_per_block=2,
        page_table=_make_gemma4_31b_page_table(gen_tp),
    )
    registrars, senders, dup = _register_gemma4_31b_ctx_ranks(ctx_tp, gen_ri)
    assert dup == {0: 1, 1: 1}
    assert senders == {0: [expected_sender], 1: [expected_sender]}

    reg = registrars[expected_sender]
    sliding = reg.get_kv_map(gen_ri, (0, 0), (0, 0))
    assert isinstance(sliding, HNDHeadMismatchMapper)
    ctx_buffer = sliding_ctx_heads * sliding_head_bytes
    gen_buffer = sliding_gen_heads * sliding_head_bytes
    pair = sliding.map(_region(1000, 2 * ctx_buffer), _region(2000, 2 * gen_buffer))
    assert pair.src.memory.ptrs.tolist() == [1000 + src_head_off, 1000 + ctx_buffer + src_head_off]
    assert pair.dst.memory.ptrs.tolist() == [2000, 2000 + gen_buffer]
    assert pair.src.memory.bytes_per_region == gen_buffer

    global_mapper = reg.get_kv_map(gen_ri, (1, 0), (1, 0))
    assert isinstance(global_mapper, IntactMapper)
    pair = global_mapper.map(_region(1000, global_layer_bytes), _region(2000, global_layer_bytes))
    assert pair.src.memory.ptrs.tolist() == [1000]
    assert pair.dst.memory.ptrs.tolist() == [2000]
    assert pair.src.memory.bytes_per_region == global_layer_bytes


@pytest.mark.parametrize("gen_tp_rank", [1, 2])
def test_gemma4_31b_ctx_tp8_to_gen_tp4_elects_one_sender_per_duplicated_head(gen_tp_rank: int):
    """Global heads duplicated on the ctx side elect one sender; sliding heads do not.

    Sliding: 16 heads give 2 per ctx rank and 4 per gen rank, so gen rank g
    holds heads [4g, 4g + 4): all heads of ctx ranks 2g and 2g + 1, which both
    send (duplicate factor 16 // 16 = 1); ctx rank 2g + 1 writes its 2 heads
    at head offset 2 of the gen buffer. Global: 4 heads give 1 per rank on
    both sides; ctx rank c holds head c * 4 // 8 = c // 2, so ctx ranks 2g and
    2g + 1 both hold gen rank g's head (duplicate factor 8 // 4 = 2) and only
    the one whose TP rank matches the gen DP rank 0 modulo 2, ctx rank 2g,
    sends it.
    """
    ctx_tp, gen_tp = 8, 4
    g = gen_tp_rank
    sliding_ctx_heads, sliding_gen_heads = 16 // ctx_tp, 16 // gen_tp
    sliding_head_bytes = 2 * 256 * 2  # tokens x head_dim x BF16
    gen_ri = make_rankinfo(
        instance_name="gen",
        instance_rank=g,
        tp_size=gen_tp,
        tp_rank=g,
        tokens_per_block=2,
        page_table=_make_gemma4_31b_page_table(gen_tp),
    )
    registrars, senders, dup = _register_gemma4_31b_ctx_ranks(ctx_tp, gen_ri)
    assert dup == {0: 1, 1: ctx_tp // gen_tp}
    assert senders == {0: [2 * g, 2 * g + 1], 1: [2 * g]}

    sliding = registrars[2 * g + 1].get_kv_map(gen_ri, (0, 0), (0, 0))
    assert isinstance(sliding, HNDHeadMismatchMapper)
    ctx_buffer = sliding_ctx_heads * sliding_head_bytes
    gen_buffer = sliding_gen_heads * sliding_head_bytes
    dst_head_off = sliding_ctx_heads * sliding_head_bytes
    pair = sliding.map(_region(1000, 2 * ctx_buffer), _region(2000, 2 * gen_buffer))
    assert pair.src.memory.ptrs.tolist() == [1000, 1000 + ctx_buffer]
    assert pair.dst.memory.ptrs.tolist() == [2000 + dst_head_off, 2000 + gen_buffer + dst_head_off]
    assert pair.src.memory.bytes_per_region == ctx_buffer


@pytest.mark.parametrize(
    ("local_replay", "peer_replay"),
    [(False, False), (True, True), (False, True), (True, False)],
)
def test_peer_registrar_register_and_get(local_replay: bool, peer_replay: bool) -> None:
    self_rankinfo = make_rankinfo(instance_name="local")
    self_rankinfo.bounded_replay_on_generation = local_replay
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=1,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
    peer_ri.bounded_replay_on_generation = peer_replay
    if local_replay != peer_replay:
        with pytest.raises(
            ValueError,
            match=f"bounded_replay_on_generation.*local={local_replay}, peer={peer_replay}",
        ):
            reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
        with pytest.raises(KeyError):
            reg.get_peer_rank_info("peer", 1)
        return

    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    assert reg.get_peer_rank_info("peer", 1) == peer_ri


def test_peer_registrar_unregister():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=1,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    reg.unregister(peer_ri.instance_name, peer_ri.instance_rank)
    with pytest.raises(KeyError):
        reg.get_peer_rank_info("peer", 1)


def test_peer_registrar_caches_and_invalidates_aux_transfer_layout():
    self_rankinfo = make_rankinfo(instance_name="local")
    self_rankinfo.aux_meta = AuxBufferMeta(
        ptrs=np.array([0x1000, 0, 0x3000, 0x4000], dtype=np.int64),
        size=np.array([1024, 0, 128, 128], dtype=np.int64),
        item_sizes=np.array([4, 0, 8, 8], dtype=np.int64),
    )
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=1,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
    peer_ri.aux_meta = AuxBufferMeta(
        ptrs=np.array([0x5000, 0, 0x7000, 0x8000], dtype=np.int64),
        size=np.array([1024, 0, 128, 128], dtype=np.int64),
        item_sizes=np.array([4, 0, 8, 8], dtype=np.int64),
    )

    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    assert reg.get_aux_transfer_layout("peer", 1) is None
    layout = build_aux_transfer_layout(self_rankinfo.aux_meta, peer_ri.aux_meta)
    reg.cache_aux_transfer_layout("peer", 1, layout)
    np.testing.assert_array_equal(layout.src_base_ptrs, [0x1000, 0x3000, 0x4000])
    np.testing.assert_array_equal(layout.dst_base_ptrs, [0x5000, 0x7000, 0x8000])
    assert reg.get_aux_transfer_layout("peer", 1) is layout

    replacement_peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=1,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
    replacement_peer_ri.aux_meta = AuxBufferMeta(
        ptrs=np.array([0x9000, 0, 0xB000, 0xC000], dtype=np.int64),
        size=np.array([1024, 0, 128, 128], dtype=np.int64),
        item_sizes=np.array([4, 0, 8, 8], dtype=np.int64),
    )
    reg.register(
        replacement_peer_ri.instance_name,
        replacement_peer_ri.instance_rank,
        replacement_peer_ri,
    )

    assert reg.get_aux_transfer_layout("peer", 1) is None
    replacement_layout = build_aux_transfer_layout(
        self_rankinfo.aux_meta, replacement_peer_ri.aux_meta
    )
    reg.cache_aux_transfer_layout("peer", 1, replacement_layout)
    assert replacement_layout is not layout
    np.testing.assert_array_equal(replacement_layout.dst_base_ptrs, [0x9000, 0xB000, 0xC000])

    reg.unregister("peer", 1)
    assert reg.get_aux_transfer_layout("peer", 1) is None


def test_peer_registration_allows_asymmetric_aux_layout_for_context_first():
    # Peer registration is request-independent. Context-first requests do not
    # transfer aux data, so differing draft-token capacities must remain valid.
    self_rankinfo = make_rankinfo(instance_name="local")
    self_rankinfo.aux_meta = AuxBufferMeta(
        ptrs=np.array([0x1000, 0x2000], dtype=np.int64),
        size=np.array([1024, 1024], dtype=np.int64),
        item_sizes=np.array([8, 8], dtype=np.int64),
    )
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=1,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
    peer_ri.aux_meta = AuxBufferMeta(
        ptrs=np.array([0x3000, 0], dtype=np.int64),
        size=np.array([1024, 0], dtype=np.int64),
        item_sizes=np.array([8, 0], dtype=np.int64),
    )

    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    assert reg.get_peer_rank_info("peer", 1) is peer_ri
    assert reg.get_aux_transfer_layout("peer", 1) is None


def test_peer_registrar_incompatible_peer_raises():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer", instance_rank=3, is_mla=True, page_table=make_page_table()
    )
    with pytest.raises(ValueError):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_peer_registrar_self_rank_info_property():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    assert reg.self_rank_info == self_rankinfo


def test_peer_registrar_get_kv_map_identity():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=1,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    # Full contiguous overlap: one whole-region fragment per block.
    assert isinstance(mapper, IntactMapper)
    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=1024)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=1024)),
    )
    assert not isinstance(pair, list)
    assert pair.src.memory.ptrs.tolist() == [1000]
    assert pair.src.memory.bytes_per_region == 1024


def test_peer_registrar_get_kv_map_head_match():
    self_pt = make_page_table(global_layer_ids=[0, 1])
    peer_pt = make_page_table(global_layer_ids=[1], block_bytes=[512])

    self_rankinfo = make_rankinfo(instance_name="local", page_table=self_pt)
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=2,
        pp_size=2,
        pp_rank=1,
        layer_num_per_pp=[1, 1],
        tokens_per_block=16,
        page_table=peer_pt,
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    # Partial layer overlap with matching heads: a single shifted fragment.
    assert isinstance(mapper, IntactMapper)
    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=1024)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=512)),
    )
    # Overlap is global layer 1: self slot holds layers [0, 1] (512B each),
    # peer slot holds only layer 1.
    assert pair.src.memory.ptrs.tolist() == [1512]
    assert pair.dst.memory.ptrs.tolist() == [2000]
    assert pair.src.memory.bytes_per_region == 512


def test_peer_registrar_get_kv_map_head_mismatch():
    self_rankinfo = make_rankinfo(instance_name="local", page_table=make_page_table())
    reg = _make_peer_registrar(self_rankinfo)
    # Twice the KV heads per rank -> twice the slot bytes on the peer side.
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=3,
        pp_size=1,
        pp_rank=0,
        tp_size=1,
        tp_rank=0,
        tokens_per_block=16,
        layer_num_per_pp=[2],
        page_table=make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert isinstance(mapper, HNDHeadMismatchMapper)


def test_peer_registrar_get_kv_map_uses_physical_offset_order():
    """The layer byte offset must follow physical buffer order, not sorted global id.

    ``self`` lays out its two global layers in physical order ``[5, 3]`` (layer 3
    sits at physical slot 1); ``peer`` holds only layer 3. The transfer must copy
    ``self``'s slot at byte offset 512 -- a sort-by-global-id would wrongly pick
    offset 0 (layer 5's bytes). The entries-driven mapper resolves each layer's
    offset from the view's buffer entries, so no ordering convention is involved.
    """
    self_pt = make_page_table(global_layer_ids=[5, 3])
    peer_pt = make_page_table(global_layer_ids=[3], block_bytes=[512])

    self_rankinfo = make_rankinfo(instance_name="local", page_table=self_pt)
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=2,
        layer_num_per_pp=[1],
        page_table=peer_pt,
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert isinstance(mapper, IntactMapper)
    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=1024)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=512)),
    )
    # Layer 3 (512B) sits at physical slot 1 on self and slot 0 on peer.
    assert pair.src.memory.ptrs.tolist() == [1512]
    assert pair.dst.memory.ptrs.tolist() == [2000]
    assert pair.src.memory.bytes_per_region == 512


def test_peer_registrar_get_kv_map_non_contiguous_overlap_splits_fragments():
    """Shared layers that are not a contiguous slot run transfer as separate fragments.

    ``self`` holds layers ``[0, 1, 2]`` and ``peer`` holds ``[0, 2]``: the overlap
    ``{0, 2}`` is not contiguous within ``self``'s slot, so a single contiguous
    fragment transfer would corrupt layer 1's bytes. The entries-driven mapper
    selects layers by explicit per-layer byte offsets, so it must emit one
    correctly-shifted fragment per contiguous run instead of rejecting (or
    silently corrupting) the transfer.
    """
    self_pt = make_page_table(global_layer_ids=[0, 1, 2], block_bytes=[1536])
    peer_pt = make_page_table(global_layer_ids=[0, 2])

    self_rankinfo = make_rankinfo(instance_name="local", page_table=self_pt)
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=2,
        layer_num_per_pp=[2],
        page_table=peer_pt,
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert isinstance(mapper, IntactMapper)
    pairs = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=1536)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=1024)),
    )
    # Layer 0: self offset 0 -> peer offset 0; layer 2: self offset 1024 ->
    # peer offset 512. Non-adjacent on self, so two fragments.
    assert [p.src.memory.ptrs.tolist() for p in pairs] == [[1000], [2024]]
    assert [p.dst.memory.ptrs.tolist() for p in pairs] == [[2000], [2512]]
    assert [p.src.memory.bytes_per_region for p in pairs] == [512, 512]


def test_peer_registrar_get_kv_map_merges_non_monotonic_contiguous_layout():
    """Physically contiguous layers merge into one fragment despite global-id order.

    Both peers lay out global layers in physical order ``[5, 3]``: byte-contiguous
    on both sides, just not monotonic in global id. Iterating the overlap in
    self's physical order keeps the offset arrays adjacent, so the mapper merges
    the two layers into a single 1024B fragment; a sorted-by-global-id iteration
    would visit offsets ``[512, 0]`` and split the copy into two fragments.
    """
    self_pt = make_page_table(global_layer_ids=[5, 3])
    peer_pt = make_page_table(global_layer_ids=[5, 3])

    self_rankinfo = make_rankinfo(instance_name="local", page_table=self_pt)
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=2,
        layer_num_per_pp=[2],
        page_table=peer_pt,
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert isinstance(mapper, IntactMapper)
    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=1024)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=1024)),
    )
    assert not isinstance(pair, list)
    assert pair.src.memory.ptrs.tolist() == [1000]
    assert pair.dst.memory.ptrs.tolist() == [2000]
    assert pair.src.memory.bytes_per_region == 1024


def test_nhd_head_mismatch_mapper_slices_each_token():
    self_ri = make_rankinfo(
        instance_name="local",
        tp_size=2,
        tp_rank=0,
        dp_size=2,
        dp_rank=0,
        tokens_per_block=2,
        element_bytes=2,
        enable_attention_dp=True,
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=2,
        tp_rank=0,
        tokens_per_block=2,
        element_bytes=2,
    )
    mapper = NHDHeadMismatchMapper(
        src_layer_offsets=[0],
        dst_layer_offsets=[0],
        self_ri=self_ri,
        peer_ri=peer_ri,
        self_kv_heads=2,
        peer_kv_heads=1,
        self_bytes_per_layer=32,
        peer_bytes_per_layer=16,
        self_buffers_per_layer=2,
        peer_buffers_per_layer=2,
    )

    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=32)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=16)),
    )

    # Source NHD has two heads per token: select head 0 at offsets 0 and 8
    # for K, then 16 and 24 for V. Destination has one head per token.
    assert pair.src.memory.ptrs.tolist() == [1000, 1008, 1016, 1024]
    assert pair.dst.memory.ptrs.tolist() == [2000, 2004, 2008, 2012]
    assert pair.src.memory.bytes_per_region == 4
    assert pair.dst.memory.bytes_per_region == 4


def test_nhd_head_mismatch_mapper_uses_scale_pool_geometry():
    self_ri = make_rankinfo(
        tp_size=2,
        tokens_per_block=2,
        element_bytes=0.5,
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        tokens_per_block=2,
        element_bytes=0.5,
    )
    mapper = NHDHeadMismatchMapper(
        src_layer_offsets=[0],
        dst_layer_offsets=[0],
        self_ri=self_ri,
        peer_ri=peer_ri,
        self_kv_heads=2,
        peer_kv_heads=1,
        self_bytes_per_layer=8,
        peer_bytes_per_layer=4,
        self_buffers_per_layer=2,
        peer_buffers_per_layer=2,
    )

    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=8)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=4)),
    )

    assert pair.src.memory.ptrs.tolist() == [1000, 1002, 1004, 1006]
    assert pair.dst.memory.ptrs.tolist() == [2000, 2001, 2002, 2003]
    assert pair.src.memory.bytes_per_region == 1
    assert pair.dst.memory.bytes_per_region == 1


def test_replicated_mapper_ignores_kv_head_mismatch():
    self_pt = make_page_table(global_layer_ids=[0])
    peer_pt = make_page_table(global_layer_ids=[0])
    for page_table in (self_pt, peer_pt):
        view = page_table.layer_groups[0].pool_views[0]
        view.pool_role = frozenset({"index_key"})
        view.mapper_kind = MapperKind.REPLICATED
    peer_pt.layer_groups[0].pool_views[0].num_kv_heads = 8

    self_rankinfo = make_rankinfo(instance_name="local", page_table=self_pt)
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=3,
        tp_size=1,
        layer_num_per_pp=[1],
        page_table=peer_pt,
    )
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert isinstance(mapper, ReplicatedMapper)


@pytest.mark.parametrize("peer_dp_rank", [0, 1, 3, 7])
def test_replicated_pool_owner_rotates_by_destination_dp_rank(peer_dp_rank):
    """Exactly one owner per fan-in group, rotated by destination DP rank.

    Mirrors the C++ MLACacheFormatter::needSendCache pairing so the
    replicated traffic spreads across local ranks for multi-DP generation.
    """
    page_table = make_page_table(global_layer_ids=[0])
    view = page_table.layer_groups[0].pool_views[0]
    view.mapper_kind = MapperKind.REPLICATED
    view.num_kv_heads = None

    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=8,
        dp_size=8,
        dp_rank=peer_dp_rank,
        enable_attention_dp=True,
        page_table=page_table,
        layer_num_per_pp=[1],
    )
    ownership = []
    for tp_rank in range(8):
        self_ri = make_rankinfo(
            instance_name="local",
            tp_size=8,
            tp_rank=tp_rank,
            page_table=page_table,
            layer_num_per_pp=[1],
        )
        reg = _make_peer_registrar(self_ri)
        ownership.append(reg.should_send_pool(peer_ri, (0, 0), (0, 0)))

    # ratio = self_tp(8) / peer_tp_per_dp(1) = 8: exactly one owner, at the
    # slot selected by the destination dp rank.
    assert sum(ownership) == 1
    assert ownership.index(True) == peer_dp_rank % 8


@pytest.mark.parametrize(
    "peer_block_bytes,expected_mapper",
    [(1344, RingMapper), (1024, ReplicatedMapper)],
    ids=["ring_sizes_differ", "ring_sizes_match"],
)
def test_ring_views_move_positions_only_when_ring_sizes_differ(peer_block_bytes, expected_mapper):
    """A 16-slot ring feeding a 21-slot ring needs the position-wise mapper.

    Equal rings keep every position in the same slot, so the region copy is
    already exact and stays in use.
    """
    self_pt = make_page_table(global_layer_ids=[0], block_bytes=[1024])
    peer_pt = make_page_table(global_layer_ids=[0], block_bytes=[peer_block_bytes])
    for page_table in (self_pt, peer_pt):
        view = page_table.layer_groups[0].pool_views[0]
        view.pool_role = frozenset({"mla_hp_tail"})
        view.mapper_kind = MapperKind.REPLICATED
        view.ring_slot_bytes = 64

    reg = _make_peer_registrar(make_rankinfo(instance_name="local", page_table=self_pt))
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=4,
        tp_size=1,
        layer_num_per_pp=[1],
        page_table=peer_pt,
    )
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert type(mapper) is expected_mapper
    if expected_mapper is RingMapper:
        assert (mapper.src_num_slots, mapper.dst_num_slots) == (16, 21)
        assert mapper.retained_positions == 16


def test_ring_mapper_splits_positions_where_either_ring_wraps():
    """Positions 30..45 of a 16-slot ring land in their slots of a 21-slot ring."""
    mapper = RingMapper([0, 100], [0, 200], slot_bytes=2, src_num_slots=16, dst_num_slots=21)
    pairs = mapper.map_positions(np.array([1000]), np.array([5000]), [(30, 46)])

    moved = {}
    for pair in pairs:
        size = pair.src.memory.bytes_per_region
        assert pair.dst.memory.bytes_per_region == size
        for src, dst in zip(pair.src.memory.ptrs.tolist(), pair.dst.memory.ptrs.tolist()):
            for i in range(size // 2):
                moved[src + 2 * i] = dst + 2 * i
    expected = {
        1000 + layer_src + (p % 16) * 2: 5000 + layer_dst + (p % 21) * 2
        for layer_src, layer_dst in ((0, 0), (100, 200))
        for p in range(30, 46)
    }
    assert moved == expected
    # Runs break at src slot 16 (p=32) and dst slot 21 (p=42): 2 + 10 + 4.
    assert sorted(pair.src.memory.bytes_per_region // 2 for pair in pairs) == [2, 4, 10]

    with pytest.raises(ValueError, match="positions both rings retain"):
        mapper.map_positions(np.array([1000]), np.array([5000]), [(29, 46)])
    with pytest.raises(NotImplementedError):
        mapper.map(None, None)


def test_peer_registrar_tpb_divisible_warns_but_compatible():
    # local=16, peer=32: 32 % 16 == 0 → compatible with warning, register succeeds
    self_rankinfo = make_rankinfo(instance_name="local", tokens_per_block=16)
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=5,
        tokens_per_block=32,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    assert reg.get_peer_rank_info(peer_ri.instance_name, peer_ri.instance_rank) is not None


def test_peer_registrar_tpb_not_divisible_raises():
    # local=16, peer=24: 24 % 16 != 0 → incompatible, register raises ValueError
    self_rankinfo = make_rankinfo(instance_name="local", tokens_per_block=16)
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=6,
        tokens_per_block=24,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
    with pytest.raises(ValueError):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


@pytest.mark.parametrize("mapper_kind", [MapperKind.NHD, MapperKind.REPLICATED])
def test_peer_registrar_exact_tpb_mapper_rejects_divisible_mismatch(mapper_kind):
    self_pt = make_page_table()
    peer_pt = make_page_table()
    self_pt.layer_groups[0].pool_views[0].mapper_kind = mapper_kind
    peer_pt.layer_groups[0].pool_views[0].mapper_kind = mapper_kind
    self_ri = make_rankinfo(page_table=self_pt, tokens_per_block=16)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=7,
        page_table=peer_pt,
        tokens_per_block=32,
    )
    reg = _make_peer_registrar(self_ri)

    with pytest.raises(ValueError):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_intact_mapper_region_size_mismatch_raises():
    with pytest.raises(ValueError, match="cache region size mismatch"):
        IntactMapper([0], [0], 256, 128)


def test_replicated_mapper_per_layer_size_mismatch_raises():
    with pytest.raises(ValueError, match="Replicated cache region size mismatch"):
        ReplicatedMapper(
            src_layer_offsets=[0, 128],
            dst_layer_offsets=[0, 64],
            self_bytes_per_layer=128,
            peer_bytes_per_layer=64,
        )


def test_replicated_mapper_selects_partial_layer_range():
    """PP mismatch selects the overlap layers via explicit offsets.

    The peer slot holds a superset of layers; only the overlap moves, and
    contiguous layers on both sides merge into one fragment.
    """
    mapper = ReplicatedMapper(
        src_layer_offsets=[0, 128],  # self slot: 2 layers x 128B
        dst_layer_offsets=[128, 256],  # peer slot: 3 layers x 128B, overlap at 1..2
        self_bytes_per_layer=128,
        peer_bytes_per_layer=128,
    )

    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=256)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=384)),
    )

    assert pair.src.memory.ptrs.tolist() == [1000]
    assert pair.dst.memory.ptrs.tolist() == [2128]
    assert pair.src.memory.bytes_per_region == 256
    assert pair.dst.memory.bytes_per_region == 256


def test_intact_mapper_splits_non_contiguous_runs():
    """Interleaved slots (another role class between layers) split runs.

    Source layers sit at non-uniform strides (something interleaves after
    layer 0); destination is densely packed. Runs must break where either
    side is discontiguous, and each fragment must carry the run's bytes.
    """
    mapper = IntactMapper(
        src_layer_offsets=[0, 192, 320],  # gap after layer 0 (64B interleaved)
        dst_layer_offsets=[0, 128, 256],  # dense
        self_bytes_per_layer=128,
        peer_bytes_per_layer=128,
    )

    pairs = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=448)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=384)),
    )

    assert isinstance(pairs, list) and len(pairs) == 2
    # Run 1: layer 0 alone (src gap breaks the run).
    assert pairs[0].src.memory.ptrs.tolist() == [1000]
    assert pairs[0].dst.memory.ptrs.tolist() == [2000]
    assert pairs[0].src.memory.bytes_per_region == 128
    # Run 2: layers 1-2 contiguous on both sides -> merged.
    assert pairs[1].src.memory.ptrs.tolist() == [1192]
    assert pairs[1].dst.memory.ptrs.tolist() == [2128]
    assert pairs[1].src.memory.bytes_per_region == 256


def test_nhd_mapper_rejects_non_divisible_region_geometry():
    self_ri = make_rankinfo(tokens_per_block=2)
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        tokens_per_block=2,
    )

    with pytest.raises(ValueError, match="not evenly divisible"):
        NHDHeadMismatchMapper(
            src_layer_offsets=[0],
            dst_layer_offsets=[0],
            self_ri=self_ri,
            peer_ri=peer_ri,
            self_kv_heads=2,
            peer_kv_heads=4,
            self_bytes_per_layer=17,
            peer_bytes_per_layer=32,
            self_buffers_per_layer=2,
            peer_buffers_per_layer=2,
        )


def test_nhd_mapper_rejects_tokens_per_block_mismatch():
    self_ri = make_rankinfo(tokens_per_block=2)
    peer_ri = make_rankinfo(instance_name="peer", tokens_per_block=4)

    with pytest.raises(ValueError, match="requires equal tokens_per_block"):
        NHDHeadMismatchMapper(
            src_layer_offsets=[0],
            dst_layer_offsets=[0],
            self_ri=self_ri,
            peer_ri=peer_ri,
            self_kv_heads=2,
            peer_kv_heads=2,
            self_bytes_per_layer=32,
            peer_bytes_per_layer=64,
            self_buffers_per_layer=2,
            peer_buffers_per_layer=2,
        )


def test_get_buffers_per_layer_rejects_non_uniform_nhd_entries():
    pool_view = PoolView(
        pool_idx=0,
        buffer_entries=np.array(
            [(0, 0, 16), (0, 16, 16), (1, 32, 16)],
            dtype=BUFFER_ENTRY_DTYPE,
        ),
        mapper_kind=MapperKind.NHD,
    )

    with pytest.raises(ValueError, match="layer_group=3, pool=4"):
        PeerRegistrar._get_buffers_per_layer(
            pool_view,
            layer_group_id=3,
            pool_idx=4,
        )


def test_non_uniform_view_geometry_rejected_for_all_kinds():
    """Entries-driven views require uniform per-layer regions for every kind.

    Under the unified contract INDEXED views are no longer exempt: a view
    whose layers have different region sizes cannot be addressed per layer
    and must fail loudly instead of transferring garbage.
    """
    entries = np.array(
        [(0, 0, 16), (0, 16, 16), (1, 32, 16)],
        dtype=BUFFER_ENTRY_DTYPE,
    )
    self_pt = make_page_table()
    peer_pt = make_page_table()
    self_pt.layer_groups[0].pool_views[0].buffer_entries = entries
    peer_pt.layer_groups[0].pool_views[0].buffer_entries = entries.copy()
    self_ri = make_rankinfo(page_table=self_pt)
    peer_ri = make_rankinfo(instance_name="peer", page_table=peer_pt)
    reg = _make_peer_registrar(self_ri)
    with pytest.raises(ValueError, match="not evenly distributed"):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_peer_registrar_allows_byte_aligned_subbyte_head_mismatch():
    """Registration accepts a 0.5-byte-element head mismatch whose heads cover whole bytes.

    Each K/V buffer holds 2 heads in 256 bytes locally and 4 heads in 512
    bytes on the peer: 128 bytes per head (16 tokens x 16 dims x 0.5 bytes),
    so the HND path slices whole heads.
    """
    self_ri = make_rankinfo(
        element_bytes=0.5,
        tp_size=2,
        page_table=make_page_table(),
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        element_bytes=0.5,
        tp_size=1,
        page_table=make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4),
    )
    reg = _make_peer_registrar(self_ri)
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))

    assert isinstance(mapper, HNDHeadMismatchMapper)


def test_peer_registrar_rejects_misaligned_subbyte_head_mismatch():
    """Head slicing that lands mid-byte must fail at registration.

    tpb=1 x dims=1 x 0.5B = 0.5B per head is not byte-aligned: each K/V
    buffer holds 1 byte for 2 heads locally and 2 bytes for 4 heads on the peer.
    """
    self_ri = make_rankinfo(
        element_bytes=0.5,
        tokens_per_block=1,
        tp_size=2,
        page_table=make_page_table(block_bytes=[4]),
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        element_bytes=0.5,
        tokens_per_block=1,
        tp_size=1,
        page_table=make_page_table(block_bytes=[8], mamba_tp=1, kv_heads=4),
    )
    reg = _make_peer_registrar(self_ri)

    with pytest.raises(ValueError, match="not a multiple of 2 buffers x 2 KV heads"):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_registration_rejects_misaligned_heads_on_peer_side_only():
    """Only the peer's heads split mid-byte: 1.5 bytes each, versus 1 byte locally.

    Floor division would report 1 byte per head on both sides.
    """
    self_ri = make_rankinfo(
        element_bytes=0.5, tp_size=2, page_table=make_page_table(block_bytes=[8])
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        element_bytes=0.5,
        tp_size=1,
        page_table=make_page_table(block_bytes=[24], mamba_tp=1, kv_heads=4),
    )
    reg = _make_peer_registrar(self_ri)

    with pytest.raises(
        ValueError, match="the peer view .* not a multiple of 2 buffers x 4 KV heads"
    ):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_peer_registrar_dispatches_nhd_mapper():
    self_pt = make_page_table()
    peer_pt = make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4)
    self_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    peer_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    self_ri = make_rankinfo(
        tp_size=2,
        page_table=self_pt,
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        page_table=peer_pt,
    )
    reg = _make_peer_registrar(self_ri)
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))

    assert isinstance(mapper, NHDHeadMismatchMapper)


def test_peer_registrar_warns_for_nhd_head_mismatch(monkeypatch):
    self_pt = make_page_table()
    peer_pt = make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4)
    self_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    peer_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    self_ri = make_rankinfo(tp_size=2, page_table=self_pt)
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        page_table=peer_pt,
    )
    warnings = []
    monkeypatch.setattr(
        peer_module.logger,
        "warning_once",
        lambda *message, key: warnings.append((" ".join(map(str, message)), key)),
    )

    _make_peer_registrar(self_ri).register(
        peer_ri.instance_name,
        peer_ri.instance_rank,
        peer_ri,
    )

    assert len(warnings) == 1
    message, key = warnings[0]
    assert "4 NIXL descriptors per transferred token per peer" in message
    assert "local_kv_heads=2, peer_kv_heads=4" in message
    assert key == "native-nhd-head-mismatch-2-4-4"


def test_peer_registrar_nhd_head_match_uses_intact_mapper(monkeypatch):
    """NHD + equal heads takes the merged-run fast path, not per-token slicing.

    Registration emits no NHD fragment warning. With a dedicated
    (non-interleaved) K/V pool the run merge collapses the whole class region
    into a single fragment per block — the fragment-count budget for separate
    layouts.
    """
    self_pt = make_page_table()
    peer_pt = make_page_table()
    self_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    peer_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    self_ri = make_rankinfo(page_table=self_pt)
    peer_ri = make_rankinfo(instance_name="peer", instance_rank=9, page_table=peer_pt)
    warnings = []
    monkeypatch.setattr(
        peer_module.logger,
        "warning_once",
        lambda *message, key: warnings.append((" ".join(map(str, message)), key)),
    )
    reg = _make_peer_registrar(self_ri)
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    assert warnings == []

    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert isinstance(mapper, IntactMapper)
    assert not isinstance(mapper, NHDHeadMismatchMapper)

    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000, 5000]), bytes_per_region=1024)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000, 6000]), bytes_per_region=1024)),
    )
    # Fully contiguous on both sides -> exactly one fragment per block.
    assert not isinstance(pair, list)
    assert pair.src.memory.ptrs.tolist() == [1000, 5000]
    assert pair.dst.memory.ptrs.tolist() == [2000, 6000]
    assert pair.src.memory.bytes_per_region == 1024


def test_nhd_equal_heads_at_different_tp_does_not_warn(monkeypatch):
    """A 4-head layer at TP4 and TP8 leaves one KV head per rank on both sides: no slicing."""
    self_ri = make_rankinfo(
        tp_size=4,
        tp_rank=1,
        tokens_per_block=2,
        page_table=_make_per_view_page_table([1], MapperKind.NHD),
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=8,
        tp_rank=2,
        tokens_per_block=2,
        page_table=_make_per_view_page_table([1], MapperKind.NHD),
    )
    warnings = []
    monkeypatch.setattr(
        peer_module.logger,
        "warning_once",
        lambda *message, key: warnings.append((" ".join(map(str, message)), key)),
    )

    _make_peer_registrar(self_ri).register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    assert warnings == []


def test_indexed_head_mismatch_subbyte_geometry_is_entries_derived():
    """HND head-mismatch byte math comes from slot bytes, never element_bytes.

    Emulates an NVFP4-like cache: element_bytes is fractional (0.5), but the
    slot size registered by storage is whole bytes, so every derived offset
    and fragment size must be an exact integer.
    """
    # self: 2 kv heads/rank, per-head bytes = 16 tokens x 8 dims x 0.5B = 64.
    self_ri = make_rankinfo(
        tp_size=1,
        tp_rank=0,
        tokens_per_block=16,
        element_bytes=0.5,
    )
    # peer: twice the TP -> 1 kv head/rank, half the slot bytes.
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=2,
        tp_rank=1,
        tokens_per_block=16,
        element_bytes=0.5,
    )
    mapper = HNDHeadMismatchMapper(
        src_layer_offsets=[0, 256],  # 2 layers x 256B/layer (kv_factor 2 x 128B)
        dst_layer_offsets=[0, 128],  # 2 layers x 128B/layer (kv_factor 2 x 64B)
        self_ri=self_ri,
        peer_ri=peer_ri,
        self_kv_heads=2,
        peer_kv_heads=1,
        self_bytes_per_layer=256,
        peer_bytes_per_layer=128,
        self_buffers_per_layer=2,
        peer_buffers_per_layer=2,
    )

    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=512)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=256)),
    )
    # peer tp_rank=1 selects head 1 inside self's per-rank pair of heads:
    # src_head_off = 1 x 64B; fragments = (layer, k/v) x 2 layers.
    assert pair.src.memory.ptrs.tolist() == [1064, 1192, 1320, 1448]
    assert pair.dst.memory.ptrs.tolist() == [2000, 2064, 2128, 2192]
    assert pair.src.memory.bytes_per_region == 64
    assert pair.src.memory.ptrs.dtype == np.int64
    assert pair.dst.memory.ptrs.dtype == np.int64


def test_indexed_head_mismatch_inconsistent_slot_geometry_raises():
    self_ri = make_rankinfo(tp_size=1)
    peer_ri = make_rankinfo(instance_name="peer", tp_size=2)
    with pytest.raises(ValueError, match="HND bytes per head mismatch"):
        HNDHeadMismatchMapper(
            src_layer_offsets=[0, 256],
            dst_layer_offsets=[0, 256],
            self_ri=self_ri,
            peer_ri=peer_ri,
            self_kv_heads=2,
            peer_kv_heads=1,
            self_bytes_per_layer=256,
            peer_bytes_per_layer=256,  # should be 128 for half the heads
            self_buffers_per_layer=2,
            peer_buffers_per_layer=2,
        )


# ---------------------------------------------------------------------------
# Policy dispatch: the policy is chosen by CacheKind; each policy picks the
# mapper for a view from its MapperKind, including REPLICATED.
# ---------------------------------------------------------------------------


def _add_replicated_side_view(page_table, *, bytes_per_layer=8):
    """Graft a replicated recurrent side view (e.g. PLE) onto the mamba group."""
    mamba_lg = page_table.layer_groups[1]
    pool_group = page_table.pool_groups[mamba_lg.pool_group_idx]
    pool_group.pools.append(
        PhysicalPool(base_address=0xC000, slot_bytes=bytes_per_layer, num_slots=128)
    )
    mamba_lg.pool_views.append(
        PoolView(
            pool_idx=len(pool_group.pools) - 1,
            buffer_entries=np.array(
                [(0, 0, bytes_per_layer), (1, bytes_per_layer, bytes_per_layer)],
                dtype=BUFFER_ENTRY_DTYPE,
            ),
            pool_role=frozenset({"ple_ngram_context"}),
            mapper_kind=MapperKind.REPLICATED,
            bytes_per_layer=bytes_per_layer,
        )
    )
    return len(mamba_lg.pool_views) - 1


def test_policy_dispatch_keys_on_cache_kind_only():
    reg = _make_peer_registrar(make_rankinfo(instance_name="local"))
    assert isinstance(reg._get_policy(CacheKind.PAGED), AttentionPolicy)
    assert isinstance(reg._get_policy(CacheKind.STATE), MambaPolicy)


def test_mamba_replicated_side_view_elects_one_sender():
    """Replicated side state elects one sender per fan-in group.

    Sharded conv state on the same layer group is sent by every mamba TP rank.
    """
    peer_pt = make_page_table(mamba_tp=1)
    _add_replicated_side_view(peer_pt)
    peer_ri = make_rankinfo(instance_name="peer", tp_size=1, page_table=peer_pt)

    conv_ownership = []
    side_ownership = []
    for tp_rank in range(2):
        self_pt = make_page_table(mamba_tp=2)
        side_idx = _add_replicated_side_view(self_pt)
        self_ri = make_rankinfo(
            instance_name="local", tp_size=2, tp_rank=tp_rank, page_table=self_pt
        )
        reg = _make_peer_registrar(self_ri)
        conv_ownership.append(reg.should_send_pool(peer_ri, (1, 0), (1, 0)))
        side_ownership.append(reg.should_send_pool(peer_ri, (1, side_idx), (1, side_idx)))

    assert conv_ownership == [True, True]
    assert side_ownership == [True, False]


def test_mamba_replicated_side_view_copies_whole_layers_under_tp_mismatch():
    """REPLICATED state ignores mamba TP.

    The head-match mapper is used even when conv/SSM on the same group need a
    resharding mapper.
    """
    self_pt = make_page_table(mamba_tp=2)
    self_side = _add_replicated_side_view(self_pt)
    peer_pt = make_page_table(mamba_tp=1)
    peer_side = _add_replicated_side_view(peer_pt)
    self_ri = make_rankinfo(instance_name="local", tp_size=2, page_table=self_pt)
    peer_ri = make_rankinfo(instance_name="peer", tp_size=1, page_table=peer_pt)
    reg = _make_peer_registrar(self_ri)

    mapper = reg.get_kv_map(peer_ri, (1, self_side), (1, peer_side))
    assert isinstance(mapper, MambaHeadMatchMapper)
    conv_mapper = reg.get_kv_map(peer_ri, (1, 0), (1, 0))
    assert isinstance(conv_mapper, ConvStateMismatchMapper)


def test_attention_replicated_validation_ignores_state_groups():
    """A STATE-only replicated-role mismatch is MambaPolicy's to reject."""
    self_pt = make_page_table()
    _add_replicated_side_view(self_pt)
    peer_pt = make_page_table()
    AttentionPolicy.validate_peer_compatible(self_pt, peer_pt)
    ri = make_rankinfo(instance_name="local", page_table=self_pt)
    with pytest.raises(ValueError, match="MambaPolicy.*differ on overlapping layers"):
        MambaPolicy.validate_peer_compatible(ri, ri, self_pt, peer_pt)


def test_attention_replicated_validation_rejects_missing_index_key_view():
    self_pt = make_page_table(global_layer_ids=[0])
    view = self_pt.layer_groups[0].pool_views[0]
    view.pool_role = frozenset({"index_key"})
    view.mapper_kind = MapperKind.REPLICATED
    peer_pt = make_page_table(global_layer_ids=[0])
    with pytest.raises(ValueError, match="AttentionPolicy.*differ on overlapping layers"):
        AttentionPolicy.validate_peer_compatible(self_pt, peer_pt)


@pytest.mark.parametrize(("local_draft", "peer_draft"), [(0, 0), (1, 1), (1, 0), (0, 1)])
def test_peer_registrar_requires_matching_unified_draft_layers(
    local_draft: int, peer_draft: int
) -> None:
    self_rankinfo = make_rankinfo(instance_name="local")
    self_rankinfo.num_draft_layers = local_draft
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=1,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
    peer_ri.num_draft_layers = peer_draft
    if local_draft != peer_draft:
        with pytest.raises(
            ValueError,
            match=f"draft KV layer count differs.*local={local_draft}, peer={peer_draft}",
        ):
            reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
        return

    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    assert reg.get_peer_rank_info("peer", 1) == peer_ri
