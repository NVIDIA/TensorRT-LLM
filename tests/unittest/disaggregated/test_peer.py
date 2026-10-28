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
from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    AuxBufferMeta,
    build_aux_transfer_layout,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import AttentionPolicy
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import MambaPolicy
from tensorrt_llm._torch.disaggregation.native.peer import PeerOverlap, PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.runs import RunsMapper
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    page_table_from_layout,
)
from tensorrt_llm._torch.disaggregation.resource.page import CacheKind, MapperKind
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    BufferDesc,
    LayerGroupDesc,
    PageLayout,
    RankTopology,
    head_major,
    replicated,
    sectioned,
    token_major,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_table import DevicePool, ManagerLayout

pytestmark = pytest.mark.cpu_only

MAMBA_LAYERS = (100, 101)
SSM_HEAD_BYTES = 64


def manager_layout(layout, pool_bases, kv_heads=0):
    """The plain-value page table a manager would build around ``layout``.

    Each layer group has one internal layer per global layer, in order; device pools lie at
    ``pool_bases[group]``.
    """
    return ManagerLayout(
        page_layout=layout,
        tokens_per_block=layout.tokens_per_page,
        pool_group_of=tuple(g.pool_group for g in layout.layer_groups),
        windows=tuple(g.kept_window for g in layout.layer_groups),
        sink_blocks=tuple(g.num_sink_blocks for g in layout.layer_groups),
        recurrent=tuple(g.kind == "state" for g in layout.layer_groups),
        sparse=tuple(False for _ in layout.layer_groups),
        layers=tuple(g.layers for g in layout.layer_groups),
        shards=tuple((g.shard.count, g.shard.index) for g in layout.layer_groups),
        page_bytes={g: sum(widths) for g, widths in layout.pool_slot_bytes.items()},
        device_pools={
            g: tuple(
                DevicePool(group=g, index=i, base=base, slot_bytes=width, num_slots=128)
                for i, (base, width) in enumerate(zip(pool_bases[g], widths))
            )
            for g, widths in layout.pool_slot_bytes.items()
        },
        named_groups=tuple(True for _ in layout.layer_groups),
        local_layers=tuple(tuple(enumerate(g.layers)) for g in layout.layer_groups),
        kv_heads=tuple(kv_heads if g.kind == "attention" else 0 for g in layout.layer_groups),
    )


def make_page_table(
    pool_ptrs=None,
    block_bytes=None,
    global_layer_ids=None,
    mamba_tp=2,
    *,
    tp_rank=0,
    topology=None,
    kv_heads=2,
    attention="head_major",
    tokens_per_block=16,
    side_state_bytes=None,
):
    """A native page table built, as a manager's is, from its page layout.

    The attention group holds K and V of each of ``global_layer_ids`` in one pool of
    ``block_bytes[0]``-byte slots: ``kv_heads`` heads, this rank's share, laid out by ``attention``
    ("head_major", "token_major" over ``tokens_per_block`` tokens, or "replicated"). The recurrent
    group (layers 100 and 101) is the ``mamba_tp`` share of a fixed global state, a conv state of
    three sections and an SSM state of 64-byte heads, so page tables built for different TP degrees
    stay peer-compatible under MambaPolicy's global size check; with ``side_state_bytes`` it also
    holds that many bytes per layer of replicated side state (e.g. PLE) in a third pool.
    ``topology`` defaults to rank ``tp_rank`` of ``mamba_tp`` tensor-parallel ranks.
    """
    if pool_ptrs is None:
        pool_ptrs = [1234]
    if block_bytes is None:
        block_bytes = [1024]
    if global_layer_ids is None:
        global_layer_ids = [0, 1]
    if topology is None:
        topology = RankTopology(tp_size=mamba_tp, tp_rank=tp_rank)
    total_heads = kv_heads * (1 if topology.attention_dp else topology.tp_size)

    size = block_bytes[0] // (len(global_layer_ids) * 2)
    if attention == "replicated":
        kv_tensors = replicated(size)
    elif attention == "token_major":
        kv_tensors = token_major(size, tokens_per_block, kv_heads, total_heads)
    else:
        kv_tensors = head_major(size, kv_heads, total_heads)
    buffers = [
        BufferDesc(layer, role, 0, 0, (2 * i + j) * size, size, kv_tensors)
        for i, layer in enumerate(global_layer_ids)
        for j, role in enumerate(("key", "value"))
    ]

    sections = [1024 // mamba_tp, 512 // mamba_tp, 512 // mamba_tp]
    conv_bytes = sum(sections)
    ssm_bytes = 4096 // mamba_tp
    state_slots = [2 * conv_bytes, 2 * ssm_bytes]
    for i, layer in enumerate(MAMBA_LAYERS):
        buffers.append(
            BufferDesc(
                layer, "conv_state", 1, 0, i * conv_bytes, conv_bytes, sectioned(sections, mamba_tp)
            )
        )
        buffers.append(
            BufferDesc(
                layer,
                "ssm_state",
                1,
                1,
                i * ssm_bytes,
                ssm_bytes,
                head_major(ssm_bytes, ssm_bytes // SSM_HEAD_BYTES, 4096 // SSM_HEAD_BYTES),
            )
        )
    if side_state_bytes is not None:
        state_slots.append(2 * side_state_bytes)
        for i, layer in enumerate(MAMBA_LAYERS):
            buffers.append(
                BufferDesc(
                    layer,
                    "ple_ngram_context",
                    1,
                    2,
                    i * side_state_bytes,
                    side_state_bytes,
                    replicated(side_state_bytes),
                )
            )

    layout = PageLayout(
        tokens_per_page=tokens_per_block,
        pool_slot_bytes={0: (block_bytes[0],), 1: tuple(state_slots)},
        layer_groups=(
            LayerGroupDesc("attention", None, 0, 0, tuple(global_layer_ids)),
            LayerGroupDesc("state", None, 0, 1, MAMBA_LAYERS),
        ),
        buffers=tuple(buffers),
        topology=topology,
    )
    return page_table_from_layout(
        manager_layout(layout, {0: pool_ptrs, 1: [0xA000, 0xB000, 0xC000]}, kv_heads)
    )


def runs_of(mapper, src_base=1000, dst_base=2000):
    """The fragments the mapper copies of one slot, sorted.

    Each is (source offset, destination offset, bytes).
    """
    pairs = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([src_base]), bytes_per_region=1)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([dst_base]), bytes_per_region=1)),
    )
    if not isinstance(pairs, list):
        pairs = [pairs]
    return sorted(
        (int(s) - src_base, int(d) - dst_base, p.src.memory.bytes_per_region)
        for p in pairs
        for s, d in zip(p.src.memory.ptrs, p.dst.memory.ptrs)
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


@pytest.mark.parametrize(
    ("self_heads", "peer_heads"),
    [(0, 0), (0, 2), (2, 0)],
)
def test_attention_free_stage_has_no_head_duplication(
    self_heads: int,
    peer_heads: int,
) -> None:
    self_ri = make_rankinfo(
        "self",
        tp_size=2,
        tp_rank=0,
        kv_heads_per_rank=self_heads,
        layer_num_per_pp=[2],
    )
    peer_ri = make_rankinfo(
        "peer",
        tp_size=2,
        tp_rank=0,
        kv_heads_per_rank=peer_heads,
        layer_num_per_pp=[2],
    )
    reg, peer_ri = _make_peer_registrar_and_peer_ri(self_ri, peer_ri)

    overlap = reg.get_peer_overlap(peer_ri, peer_dp_rank=0)

    assert overlap.duplicate_head_factor == 1
    assert overlap.peer_duplicate_head_factor == 1
    assert overlap.ranks == [0]


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


def _registered(self_pt, peer_pt, **peer_kwargs):
    """A registrar with a peer registered, and that peer's rank info.

    The local rank holds ``self_pt`` and the peer ``peer_pt``.
    """
    reg = _make_peer_registrar(make_rankinfo(instance_name="local", page_table=self_pt))
    peer_ri = make_rankinfo(
        instance_name="peer", instance_rank=2, page_table=peer_pt, **peer_kwargs
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    return reg, peer_ri


def test_peer_registrar_get_kv_map_identity():
    reg, peer_ri = _registered(make_page_table(), make_page_table(), layer_num_per_pp=[2])
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    # Same layout on both sides: one whole-slot fragment per block.
    assert isinstance(mapper, RunsMapper)
    assert runs_of(mapper) == [(0, 0, 1024)]
    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000, 5000]), bytes_per_region=1024)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000, 6000]), bytes_per_region=1024)),
    )
    assert not isinstance(pair, list)
    assert pair.src.memory.ptrs.tolist() == [1000, 5000]
    assert pair.dst.memory.ptrs.tolist() == [2000, 6000]
    assert pair.src.memory.bytes_per_region == 1024


def test_peer_registrar_get_kv_map_partial_layer_overlap():
    reg, peer_ri = _registered(
        make_page_table(global_layer_ids=[0, 1]),
        make_page_table(global_layer_ids=[1], block_bytes=[512]),
        pp_size=2,
        pp_rank=1,
        layer_num_per_pp=[1, 1],
    )
    # Global layer 1: self holds layers [0, 1] (512B each), the peer only layer 1.
    assert runs_of(reg.get_kv_map(peer_ri, (0, 0), (0, 0))) == [(512, 0, 512)]


def test_peer_registrar_get_kv_map_head_mismatch():
    """TP2 with 2 heads per rank to TP1 with 4: rank 0 fills heads 0-1 of every K/V buffer."""
    reg, peer_ri = _registered(
        make_page_table(),
        make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4),
        tp_size=1,
        kv_heads_per_rank=4,
        layer_num_per_pp=[2],
    )
    # 256B buffers of 2 x 128B heads into 512B buffers of 4.
    assert runs_of(reg.get_kv_map(peer_ri, (0, 0), (0, 0))) == [
        (0, 0, 256),
        (256, 512, 256),
        (512, 1024, 256),
        (768, 1536, 256),
    ]


def test_peer_registrar_get_kv_map_head_mismatch_second_rank():
    """TP2 rank 1 fills heads 2-3 of the TP1 peer's buffers."""
    reg = _make_peer_registrar(
        make_rankinfo(instance_name="local", tp_rank=1, page_table=make_page_table(tp_rank=1))
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        kv_heads_per_rank=4,
        page_table=make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4),
    )
    assert runs_of(reg.get_kv_map(peer_ri, (0, 0), (0, 0))) == [
        (0, 256, 256),
        (256, 768, 256),
        (512, 1280, 256),
        (768, 1792, 256),
    ]


def test_peer_registrar_get_kv_map_uses_physical_offset_order():
    """Layer bytes are found by their slot offsets, not by sorted global id.

    ``self`` lays out its two global layers in physical order ``[5, 3]`` (layer 3 sits at byte
    512); ``peer`` holds only layer 3. A sort by global id would pick layer 5's bytes.
    """
    reg, peer_ri = _registered(
        make_page_table(global_layer_ids=[5, 3]),
        make_page_table(global_layer_ids=[3], block_bytes=[512]),
        layer_num_per_pp=[1],
    )
    assert runs_of(reg.get_kv_map(peer_ri, (0, 0), (0, 0))) == [(512, 0, 512)]


def test_peer_registrar_get_kv_map_non_contiguous_overlap_splits_fragments():
    """Shared layers that are not one contiguous run of the slot move as separate fragments.

    ``self`` holds layers ``[0, 1, 2]`` and ``peer`` holds ``[0, 2]``: one fragment that spanned
    both would carry layer 1's bytes over layer 2's.
    """
    reg, peer_ri = _registered(
        make_page_table(global_layer_ids=[0, 1, 2], block_bytes=[1536]),
        make_page_table(global_layer_ids=[0, 2]),
        layer_num_per_pp=[2],
    )
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert runs_of(mapper) == [(0, 0, 512), (1024, 512, 512)]
    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=1536)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=1024)),
    )
    # Fragments of one size share a region group.
    assert pair.src.memory.ptrs.tolist() == [1000, 2024]
    assert pair.dst.memory.ptrs.tolist() == [2000, 2512]


def test_peer_registrar_get_kv_map_merges_non_monotonic_contiguous_layout():
    """Layers contiguous on both sides merge into one fragment whatever their global ids' order."""
    reg, peer_ri = _registered(
        make_page_table(global_layer_ids=[5, 3]),
        make_page_table(global_layer_ids=[5, 3]),
        layer_num_per_pp=[2],
    )
    assert runs_of(reg.get_kv_map(peer_ri, (0, 0), (0, 0))) == [(0, 0, 1024)]


def test_rank_info_carries_the_page_layout_to_peers():
    """The page layout survives the bytes a rank info is sent as.

    A peer's runs come from the layout its rank info carries.
    """
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        kv_heads_per_rank=4,
        page_table=make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4, side_state_bytes=8),
    )
    received = RankInfo.from_bytes(peer_ri.to_bytes())
    assert received.page_table.layout == peer_ri.page_table.layout

    reg = _make_peer_registrar(make_rankinfo(instance_name="local", page_table=make_page_table()))
    assert runs_of(reg.get_kv_map(received, (0, 0), (0, 0))) == runs_of(
        reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    )


def test_peer_registrar_get_kv_map_is_cached_until_unregister():
    reg, peer_ri = _registered(make_page_table(), make_page_table(), layer_num_per_pp=[2])
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert reg.get_kv_map(peer_ri, (0, 0), (0, 0)) is mapper
    reg.unregister(peer_ri.instance_name, peer_ri.instance_rank)
    assert reg.get_kv_map(peer_ri, (0, 0), (0, 0)) is not mapper


def test_peer_registrar_get_kv_map_requires_a_page_layout():
    reg, peer_ri = _registered(make_page_table(), make_page_table(), layer_num_per_pp=[2])
    peer_ri.page_table.layout = None
    with pytest.raises(ValueError, match="carries no page layout"):
        reg.get_kv_map(peer_ri, (0, 0), (0, 0))


def test_replicated_pool_ignores_kv_head_mismatch():
    reg = _make_peer_registrar(
        make_rankinfo(
            instance_name="local",
            kv_heads_per_rank=1,
            page_table=make_page_table(attention="replicated"),
        )
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=3,
        tp_size=1,
        kv_heads_per_rank=8,
        layer_num_per_pp=[2],
        page_table=make_page_table(attention="replicated", mamba_tp=1),
    )
    assert runs_of(reg.get_kv_map(peer_ri, (0, 0), (0, 0))) == [(0, 0, 1024)]


@pytest.mark.parametrize("peer_dp_rank", [0, 1, 3, 7])
def test_replicated_pool_sender_rotates_by_destination_dp_rank(peer_dp_rank):
    """Exactly one sender per fan-in group, rotated by the destination's DP rank.

    Mirrors the C++ MLACacheFormatter::needSendCache pairing so the replicated traffic spreads
    across local ranks for multi-DP generation.
    """
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=8,
        tp_rank=peer_dp_rank,
        dp_size=8,
        dp_rank=peer_dp_rank,
        enable_attention_dp=True,
        layer_num_per_pp=[1],
        page_table=make_page_table(
            global_layer_ids=[0],
            attention="replicated",
            mamba_tp=1,
            topology=RankTopology(
                tp_size=8,
                tp_rank=peer_dp_rank,
                dp_size=8,
                dp_rank=peer_dp_rank,
                attention_dp=True,
            ),
        ),
    )
    senders = []
    for tp_rank in range(8):
        self_ri = make_rankinfo(
            instance_name="local",
            tp_size=8,
            tp_rank=tp_rank,
            layer_num_per_pp=[1],
            page_table=make_page_table(
                global_layer_ids=[0], attention="replicated", mamba_tp=8, tp_rank=tp_rank
            ),
        )
        senders.append(not _make_peer_registrar(self_ri).get_kv_map(peer_ri, (0, 0), (0, 0)).empty)

    # Eight local ranks fan into the peer's single attention rank: one sends, at the slot the
    # destination's DP rank selects.
    assert senders.count(True) == 1
    assert senders.index(True) == peer_dp_rank % 8


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


@pytest.mark.parametrize("attention", ["token_major", "replicated"])
def test_peer_registrar_exact_tpb_layout_rejects_divisible_mismatch(attention):
    self_ri = make_rankinfo(page_table=make_page_table(attention=attention), tokens_per_block=16)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=7,
        page_table=make_page_table(attention=attention, tokens_per_block=32),
        tokens_per_block=32,
    )
    reg = _make_peer_registrar(self_ri)

    with pytest.raises(ValueError):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_runs_mapper_groups_fragments_by_size():
    mapper = RunsMapper(
        src_offsets=np.array([0, 192, 320]),
        dst_offsets=np.array([0, 128, 256]),
        sizes=np.array([128, 64, 128]),
    )
    assert not mapper.empty
    assert mapper.bytes_per_slot == 320

    pairs = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000, 5000]), bytes_per_region=448)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000, 6000]), bytes_per_region=384)),
    )

    assert isinstance(pairs, list) and len(pairs) == 2
    assert pairs[0].src.memory.bytes_per_region == 64
    assert pairs[0].src.memory.ptrs.tolist() == [1192, 5192]
    assert pairs[0].dst.memory.ptrs.tolist() == [2128, 6128]
    # Slot-major: each slot's fragments of one size are adjacent.
    assert pairs[1].src.memory.bytes_per_region == 128
    assert pairs[1].src.memory.ptrs.tolist() == [1000, 1320, 5000, 5320]
    assert pairs[1].dst.memory.ptrs.tolist() == [2000, 2256, 6000, 6256]


def test_runs_mapper_without_runs_is_empty():
    mapper = RunsMapper(*(np.zeros(0, dtype=np.int64),) * 3)
    assert mapper.empty
    assert mapper.bytes_per_slot == 0


def test_runs_mapper_rejects_unequal_run_arrays():
    with pytest.raises(ValueError, match="as many offsets as sizes"):
        RunsMapper(np.array([0, 8]), np.array([0]), np.array([8, 8]))


def test_runs_mapper_rejects_unequal_region_counts():
    mapper = RunsMapper(np.array([0]), np.array([0]), np.array([8]))
    with pytest.raises(ValueError, match="must match"):
        mapper.map(
            SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000, 2000]), bytes_per_region=8)),
            SpecRegion(memory=MemRegionGroup(ptrs=np.array([3000]), bytes_per_region=8)),
        )


def test_non_uniform_view_geometry_is_rejected_when_building_the_page_table():
    """A view whose layers have different region sizes cannot be addressed per layer."""
    layout = PageLayout(
        tokens_per_page=16,
        pool_slot_bytes={0: (48,)},
        layer_groups=(LayerGroupDesc("attention", None, 0, 0, (0, 1)),),
        buffers=(
            BufferDesc(0, "key", 0, 0, 0, 16),
            BufferDesc(0, "value", 0, 0, 16, 16),
            BufferDesc(1, "key", 0, 0, 32, 16),
        ),
    )
    with pytest.raises(ValueError, match="not uniform"):
        page_table_from_layout(manager_layout(layout, {0: [0x1000]}))


def test_peer_registrar_allows_byte_aligned_subbyte_head_mismatch():
    """Byte-aligned sub-byte head slicing is allowed.

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
        page_table=make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4),
    )
    reg = _make_peer_registrar(self_ri)
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    assert not reg.get_kv_map(peer_ri, (0, 0), (0, 0)).empty


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
        page_table=make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4),
    )
    reg = _make_peer_registrar(self_ri)

    with pytest.raises(ValueError):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_peer_registrar_token_major_head_mismatch_slices_each_token():
    """NHD buffers: rank 0's two heads of each token land in the first two of the peer's four."""
    reg, peer_ri = _registered(
        make_page_table(attention="token_major"),
        make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4, attention="token_major"),
        tp_size=1,
        kv_heads_per_rank=4,
    )
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))

    # Self buffers: 16 tokens x 2 heads x 8B; peer buffers: 16 tokens x 4 heads x 8B.
    assert runs_of(mapper) == [
        (buffer * 256 + token * 16, buffer * 512 + token * 32, 16)
        for buffer in range(4)
        for token in range(16)
    ]


def test_peer_registrar_warns_for_nhd_head_mismatch(monkeypatch):
    self_ri = make_rankinfo(
        kv_heads_per_rank=2, tp_size=2, page_table=make_page_table(attention="token_major")
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        kv_heads_per_rank=4,
        tp_size=1,
        page_table=make_page_table(
            block_bytes=[2048], mamba_tp=1, kv_heads=4, attention="token_major"
        ),
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


def test_peer_registrar_token_major_head_match_moves_whole_slots():
    """NHD with equal heads: the runs merge into one fragment per block."""
    reg, peer_ri = _registered(
        make_page_table(attention="token_major"),
        make_page_table(attention="token_major"),
    )
    assert runs_of(reg.get_kv_map(peer_ri, (0, 0), (0, 0))) == [(0, 0, 1024)]


# ---------------------------------------------------------------------------
# Recurrent state: the policy is chosen by CacheKind; which bytes move follows from the
# layouts, replicated side state included.
# ---------------------------------------------------------------------------

SIDE_STATE_VIEW = 2  # conv state, SSM state, then the side state, by pool


def make_tp1_peer(**page_table_kwargs):
    """A TP1 peer holding all four KV heads and the whole recurrent state."""
    return make_rankinfo(
        instance_name="peer",
        tp_size=1,
        kv_heads_per_rank=4,
        page_table=make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4, **page_table_kwargs),
    )


def test_policy_dispatch_keys_on_cache_kind_only():
    reg = _make_peer_registrar(make_rankinfo(instance_name="local"))
    assert isinstance(reg._get_policy(CacheKind.PAGED), AttentionPolicy)
    assert isinstance(reg._get_policy(CacheKind.STATE), MambaPolicy)


def test_state_views_carry_the_geometry_peers_check():
    group = make_page_table(side_state_bytes=8).layer_groups[1]
    conv, ssm, side = group.pool_views
    assert conv.mapper_kind == MapperKind.SECTIONED
    assert conv.section_bytes == [512, 256, 256]
    assert ssm.mapper_kind == MapperKind.INDEXED
    assert ssm.bytes_per_head == SSM_HEAD_BYTES
    assert side.mapper_kind == MapperKind.REPLICATED
    assert side.bytes_per_layer == 8


def test_mamba_replicated_side_view_elects_one_sender():
    """Replicated side state has one sender per fan-in group.

    Every mamba TP rank sends its share of the sharded conv state of the same layer group.
    """
    peer_ri = make_tp1_peer(side_state_bytes=8)

    conv_senders = []
    side_senders = []
    for tp_rank in range(2):
        self_ri = make_rankinfo(
            instance_name="local",
            tp_size=2,
            tp_rank=tp_rank,
            page_table=make_page_table(mamba_tp=2, tp_rank=tp_rank, side_state_bytes=8),
        )
        reg = _make_peer_registrar(self_ri)
        conv_senders.append(not reg.get_kv_map(peer_ri, (1, 0), (1, 0)).empty)
        side = reg.get_kv_map(peer_ri, (1, SIDE_STATE_VIEW), (1, SIDE_STATE_VIEW))
        side_senders.append(not side.empty)

    assert conv_senders == [True, True]
    assert side_senders == [True, False]


def test_mamba_replicated_side_view_copies_whole_layers_under_tp_mismatch():
    """Replicated state ignores mamba TP, while the conv state of the same group reshards."""
    reg = _make_peer_registrar(
        make_rankinfo(
            instance_name="local",
            tp_size=2,
            page_table=make_page_table(mamba_tp=2, side_state_bytes=8),
        )
    )
    peer_ri = make_tp1_peer(side_state_bytes=8)

    side = reg.get_kv_map(peer_ri, (1, SIDE_STATE_VIEW), (1, SIDE_STATE_VIEW))
    assert runs_of(side) == [(0, 0, 16)]
    # Rank 0 holds the first half of each conv section: [512 | 256 | 256] of [1024 | 512 | 512].
    conv = reg.get_kv_map(peer_ri, (1, 0), (1, 0))
    assert runs_of(conv) == [
        (0, 0, 512),
        (512, 1024, 256),
        (768, 1536, 256),
        (1024, 2048, 512),
        (1536, 3072, 256),
        (1792, 3584, 256),
    ]


def test_mamba_ssm_heads_reshard_under_tp_mismatch():
    """TP2 rank 1 sends its 32 heads into the second half of the TP1 peer's 64."""
    reg = _make_peer_registrar(
        make_rankinfo(
            instance_name="local",
            tp_size=2,
            tp_rank=1,
            page_table=make_page_table(mamba_tp=2, tp_rank=1),
        )
    )
    peer_ri = make_tp1_peer()

    assert runs_of(reg.get_kv_map(peer_ri, (1, 1), (1, 1))) == [
        (0, 2048, 2048),
        (2048, 6144, 2048),
    ]


def test_attention_replicated_validation_ignores_state_groups():
    """A STATE-only replicated-role mismatch is MambaPolicy's to reject."""
    self_pt = make_page_table(side_state_bytes=8)
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
