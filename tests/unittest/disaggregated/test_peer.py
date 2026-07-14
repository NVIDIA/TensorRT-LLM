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
import pytest

import tensorrt_llm._torch.disaggregation.native.peer as peer_module
from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import (
    HNDHeadMismatchMapper,
    IntactMapper,
    NHDHeadMismatchMapper,
    ReplicatedMapper,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.peer import PeerOverlap, PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LocalLayer,
    MambaLayerGroup,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)


def make_page_table(pool_ptrs=None, block_bytes=None, global_layer_ids=None):
    """Create a KVCachePageTable for testing."""
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
            PoolView(pool_idx=pi, buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE))
        )
    physical_pools = [
        PhysicalPool(base_address=ptr, slot_bytes=bs, num_slots=128)
        for ptr, bs in zip(pool_ptrs, block_bytes)
    ]

    attn_lg = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=2,
        sliding_window_size=None,
        local_layers=local_layers,
        pool_views=pool_views,
    )
    mamba_lg = MambaLayerGroup(
        pool_group_idx=1,
        mamba_layer_offsets={100: 0, 101: 1},
        conv_states=PhysicalPool(base_address=0xA000, slot_bytes=2048, num_slots=128),
        ssm_states=PhysicalPool(base_address=0xB000, slot_bytes=4096, num_slots=128),
        conv_section_bytes=[512, 256, 256],
        ssm_bytes_per_head=64,
    )
    pool_groups = [PhysicalPoolGroup(pools=physical_pools)]

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
    kv_heads_per_rank=2,
    tokens_per_block=16,
    dims_per_head=8,
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
        server_endpoint="",
        self_endpoint="",
        transfer_engine_info=b"",
        attention=AttentionInfo(
            kv_heads_per_rank=kv_heads_per_rank,
            tokens_per_block=tokens_per_block,
            dims_per_head=dims_per_head,
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
            kv_heads_per_rank=peer_ri.attention.kv_heads_per_rank,
            tokens_per_block=peer_ri.attention.tokens_per_block,
            dims_per_head=peer_ri.attention.dims_per_head,
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
        duplicate_head_factor=1,
        peer_duplicate_head_factor=2,
        ranks=[26, 27, 30, 31, 42, 43, 46, 47],
    )
    assert overlap == expected_overlap


def _make_peer_registrar(self_rankinfo):
    pt = self_rankinfo.page_table if self_rankinfo.page_table else make_page_table()
    real_kv_extractor = KVRegionExtractorV1(pt)
    reg = PeerRegistrar(self_rankinfo, real_kv_extractor)
    return reg


def test_peer_registrar_register_and_get():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=1,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
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
        dims_per_head=8,
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
        kv_heads_per_rank=4,
        tokens_per_block=16,
        dims_per_head=8,
        layer_num_per_pp=[2],
        page_table=make_page_table(block_bytes=[2048]),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert isinstance(mapper, HNDHeadMismatchMapper)


def test_nhd_head_mismatch_mapper_slices_each_token():
    self_ri = make_rankinfo(
        instance_name="local",
        tp_size=2,
        tp_rank=0,
        dp_size=2,
        dp_rank=0,
        kv_heads_per_rank=2,
        tokens_per_block=2,
        dims_per_head=2,
        element_bytes=2,
        enable_attention_dp=True,
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=2,
        tp_rank=0,
        kv_heads_per_rank=1,
        tokens_per_block=2,
        dims_per_head=2,
        element_bytes=2,
    )
    mapper = NHDHeadMismatchMapper(
        src_layer_offsets=[0],
        dst_layer_offsets=[0],
        self_ri=self_ri,
        peer_ri=peer_ri,
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
        kv_heads_per_rank=2,
        tokens_per_block=2,
        dims_per_head=128,
        element_bytes=0.5,
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        kv_heads_per_rank=1,
        tokens_per_block=2,
        dims_per_head=128,
        element_bytes=0.5,
    )
    mapper = NHDHeadMismatchMapper(
        src_layer_offsets=[0],
        dst_layer_offsets=[0],
        self_ri=self_ri,
        peer_ri=peer_ri,
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

    self_rankinfo = make_rankinfo(instance_name="local", kv_heads_per_rank=1, page_table=self_pt)
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=3,
        tp_size=1,
        kv_heads_per_rank=8,
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

    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=8,
        dp_size=8,
        dp_rank=peer_dp_rank,
        enable_attention_dp=True,
        page_table=page_table,
        layer_num_per_pp=[1],
    )
    overlap = PeerOverlap()
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
        ownership.append(reg.should_send_pool(overlap, peer_ri, 0, 0))

    # ratio = self_tp(8) / peer_tp_per_dp(1) = 8: exactly one owner, at the
    # slot selected by the destination dp rank.
    assert sum(ownership) == 1
    assert ownership.index(True) == peer_dp_rank % 8


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
    self_ri = make_rankinfo(kv_heads_per_rank=2, tokens_per_block=2)
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        kv_heads_per_rank=4,
        tokens_per_block=2,
    )

    with pytest.raises(ValueError, match="not evenly divisible"):
        NHDHeadMismatchMapper(
            src_layer_offsets=[0],
            dst_layer_offsets=[0],
            self_ri=self_ri,
            peer_ri=peer_ri,
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
            2,
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
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    with pytest.raises(ValueError, match="not uniform"):
        reg.get_kv_map(peer_ri, (0, 0), (0, 0))


def test_peer_registrar_allows_byte_aligned_subbyte_head_mismatch():
    """Byte-aligned sub-byte head slicing is allowed on the HND path.

    Per-head bytes are integral here: tpb=16 x dims=8 x 0.5B = 64B.
    """
    self_ri = make_rankinfo(
        element_bytes=0.5,
        kv_heads_per_rank=2,
        tp_size=2,
        page_table=make_page_table(),
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        element_bytes=0.5,
        kv_heads_per_rank=4,
        tp_size=1,
        page_table=make_page_table(block_bytes=[2048]),
    )
    reg = _make_peer_registrar(self_ri)
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))

    assert isinstance(mapper, HNDHeadMismatchMapper)


def test_peer_registrar_rejects_misaligned_subbyte_head_mismatch():
    """Head slicing that lands mid-byte must fail at registration.

    tpb=1 x dims=1 x 0.5B = 0.5B per head is not byte-aligned.
    """
    self_ri = make_rankinfo(
        element_bytes=0.5,
        kv_heads_per_rank=2,
        tokens_per_block=1,
        dims_per_head=1,
        tp_size=2,
        page_table=make_page_table(),
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        element_bytes=0.5,
        kv_heads_per_rank=4,
        tokens_per_block=1,
        dims_per_head=1,
        tp_size=1,
        page_table=make_page_table(block_bytes=[2048]),
    )
    reg = _make_peer_registrar(self_ri)

    with pytest.raises(ValueError):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_peer_registrar_dispatches_nhd_mapper():
    self_pt = make_page_table()
    peer_pt = make_page_table(block_bytes=[2048])
    self_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    peer_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    self_ri = make_rankinfo(
        kv_heads_per_rank=2,
        tp_size=2,
        page_table=self_pt,
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        kv_heads_per_rank=4,
        tp_size=1,
        page_table=peer_pt,
    )
    reg = _make_peer_registrar(self_ri)
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))

    assert isinstance(mapper, NHDHeadMismatchMapper)


def test_peer_registrar_warns_for_nhd_head_mismatch(monkeypatch):
    self_pt = make_page_table()
    peer_pt = make_page_table(block_bytes=[2048])
    self_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    peer_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    self_ri = make_rankinfo(kv_heads_per_rank=2, tp_size=2, page_table=self_pt)
    peer_ri = make_rankinfo(
        instance_name="peer",
        kv_heads_per_rank=4,
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


def test_peer_registrar_nhd_head_match_uses_intact_mapper():
    """NHD + equal heads takes the merged-run fast path, not per-token slicing.

    With a dedicated (non-interleaved) K/V pool the run merge collapses the
    whole class region into a single fragment per block — the fragment-count
    budget for separate layouts.
    """
    self_pt = make_page_table()
    peer_pt = make_page_table()
    self_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    peer_pt.layer_groups[0].pool_views[0].mapper_kind = MapperKind.NHD
    self_ri = make_rankinfo(page_table=self_pt)
    peer_ri = make_rankinfo(instance_name="peer", instance_rank=9, page_table=peer_pt)
    reg = _make_peer_registrar(self_ri)
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

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
        kv_heads_per_rank=2,
        tokens_per_block=16,
        dims_per_head=8,
        element_bytes=0.5,
    )
    # peer: twice the TP -> 1 kv head/rank, half the slot bytes.
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=2,
        tp_rank=1,
        kv_heads_per_rank=1,
        tokens_per_block=16,
        dims_per_head=8,
        element_bytes=0.5,
    )
    mapper = HNDHeadMismatchMapper(
        src_layer_offsets=[0, 256],  # 2 layers x 256B/layer (kv_factor 2 x 128B)
        dst_layer_offsets=[0, 128],  # 2 layers x 128B/layer (kv_factor 2 x 64B)
        self_ri=self_ri,
        peer_ri=peer_ri,
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
    self_ri = make_rankinfo(tp_size=1, kv_heads_per_rank=2)
    peer_ri = make_rankinfo(instance_name="peer", tp_size=2, kv_heads_per_rank=1)
    with pytest.raises(ValueError, match="HND bytes per head mismatch"):
        HNDHeadMismatchMapper(
            src_layer_offsets=[0, 256],
            dst_layer_offsets=[0, 256],
            self_ri=self_ri,
            peer_ri=peer_ri,
            self_bytes_per_layer=256,
            peer_bytes_per_layer=256,  # should be 128 for half the heads
            self_buffers_per_layer=2,
            peer_buffers_per_layer=2,
        )
