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

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from layout_test_utils import manager_layout

import tensorrt_llm._torch.disaggregation.native.peer as peer_module
from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    AuxBufferMeta,
    build_aux_transfer_layout,
)
from tensorrt_llm._torch.disaggregation.native.peer import PeerOverlap, PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.runs import RunsMapper
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

pytestmark = pytest.mark.cpu_only

MAMBA_LAYERS = (100, 101)
SSM_HEAD_BYTES = 64


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
    index_key_bytes=None,
):
    """A native page table built, as a manager's is, from its page layout.

    The attention group holds K and V of each of ``global_layer_ids`` in one pool of
    ``block_bytes[0]``-byte slots: ``kv_heads`` heads, this rank's share, laid out by ``attention``
    ("head_major", "token_major" over ``tokens_per_block`` tokens, or "replicated"). The recurrent
    group (layers 100 and 101) is the ``mamba_tp`` share of a fixed global state, a conv state of
    three sections and an SSM state of 64-byte heads, so page tables built for different TP degrees
    describe the same global tensors; with ``side_state_bytes`` it also holds that many bytes per
    layer of replicated side state (e.g. PLE) in a third pool, and with ``index_key_bytes`` the
    attention group holds that many bytes per layer of replicated index-K in a second pool.
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
    attention_slots = [block_bytes[0]]
    if index_key_bytes is not None:
        attention_slots.append(len(global_layer_ids) * index_key_bytes)
        buffers.extend(
            BufferDesc(
                layer,
                "index_key",
                0,
                1,
                i * index_key_bytes,
                index_key_bytes,
                replicated(index_key_bytes),
            )
            for i, layer in enumerate(global_layer_ids)
        )

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
        pool_slot_bytes={0: tuple(attention_slots), 1: tuple(state_slots)},
        layer_groups=(
            LayerGroupDesc("attention", None, 0, 0, tuple(global_layer_ids)),
            LayerGroupDesc("state", None, 0, 1, MAMBA_LAYERS),
        ),
        buffers=tuple(buffers),
        topology=topology,
    )
    return manager_layout(
        layout, {0: [*pool_ptrs, 0xD000], 1: [0xA000, 0xB000, 0xC000]}, kv_heads=kv_heads
    )


def kv_map(reg, peer_ri, src, dst, excluded=frozenset()):
    """The mapper of one pool pair, or None when ``reg``'s rank sends nothing of it.

    The pair is this rank's pool ``src`` and the peer rank's pool ``dst``, each (layer group,
    pool).
    """
    return reg.get_kv_maps(peer_ri, excluded).get((src, dst))


def runs_of(mapper, src_base=1000, dst_base=2000):
    """The fragments the mapper copies of one slot, sorted; none without a mapper.

    Each is (source offset, destination offset, bytes).
    """
    if mapper is None:
        return []
    return sorted(
        (int(s) - src_base, int(d) - dst_base, size)
        for src, dst, size in mapper.map(np.array([src_base]), np.array([dst_base]))
        for s, d in zip(src, dst)
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
        enable_attention_dp=enable_attention_dp,
        aux_meta=None,
        page_table=page_table,
        sender_endpoints=[],
    )


def _make_peer_registrar_and_peer_ri(self_ri, peer_ri):
    if self_ri.page_table is None:
        self_ri.page_table = make_page_table()
    reg = PeerRegistrar(self_ri)
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
            enable_attention_dp=peer_ri.enable_attention_dp,
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


def _tp_rank_info(name, tp_size, tp_rank, total_heads=8):
    """Rank ``tp_rank`` of a TP ``tp_size`` instance holding its share of ``total_heads`` heads."""
    heads = total_heads // tp_size
    return make_rankinfo(
        instance_name=name,
        instance_rank=tp_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        page_table=make_page_table(
            block_bytes=[4 * 32 * heads], mamba_tp=tp_size, tp_rank=tp_rank, kv_heads=heads
        ),
    )


@pytest.mark.parametrize("src_tp,dst_tp", [(1, 2), (2, 1), (2, 4), (4, 2), (8, 2), (2, 8)])
def test_every_sender_whose_bytes_reach_a_receiver_is_one_it_pairs_with(src_tp, dst_tp):
    """No elected sender goes unheard.

    The sender ranks a receiver contacts (get_peer_overlap) include every rank whose runs reach it.
    """
    pools = [(0, 0), (1, 0), (1, 1)]  # K/V, conv state, SSM state
    for dst_rank in range(dst_tp):
        receiver = _tp_rank_info("gen", dst_tp, dst_rank)
        contacted = _make_peer_registrar(receiver).get_peer_overlap(
            _tp_rank_info("ctx", src_tp, 0), peer_dp_rank=0
        )
        senders = {
            src_rank
            for src_rank in range(src_tp)
            for pool in pools
            if kv_map(
                _make_peer_registrar(_tp_rank_info("ctx", src_tp, src_rank)), receiver, pool, pool
            )
            is not None
        }
        assert senders and senders <= set(contacted.ranks), (dst_rank, senders, contacted.ranks)


def _make_peer_registrar(self_rankinfo):
    if self_rankinfo.page_table is None:
        self_rankinfo.page_table = make_page_table()
    return PeerRegistrar(self_rankinfo)


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
    # Twice the heads per rank at the same TP: the K/V tensors are of 8 heads, not 4.
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=3,
        page_table=make_page_table(block_bytes=[2048], kv_heads=4),
    )
    with pytest.raises(ValueError, match="do not correspond"):
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


def test_peer_registrar_kv_map_identity():
    reg, peer_ri = _registered(make_page_table(), make_page_table(), layer_num_per_pp=[2])
    mapper = kv_map(reg, peer_ri, (0, 0), (0, 0))
    # Same layout on both sides: one whole-slot fragment per block.
    assert isinstance(mapper, RunsMapper)
    assert runs_of(mapper) == [(0, 0, 1024)]
    ((src, dst, size),) = mapper.map(np.array([1000, 5000]), np.array([2000, 6000]))
    assert src.tolist() == [1000, 5000]
    assert dst.tolist() == [2000, 6000]
    assert size == 1024


def test_peer_registrar_kv_map_partial_layer_overlap():
    reg, peer_ri = _registered(
        make_page_table(global_layer_ids=[0, 1]),
        make_page_table(global_layer_ids=[1], block_bytes=[512]),
        pp_size=2,
        pp_rank=1,
        layer_num_per_pp=[1, 1],
    )
    # Global layer 1: self holds layers [0, 1] (512B each), the peer only layer 1.
    assert runs_of(kv_map(reg, peer_ri, (0, 0), (0, 0))) == [(512, 0, 512)]


def test_peer_registrar_kv_map_head_mismatch():
    """TP2 with 2 heads per rank to TP1 with 4: rank 0 fills heads 0-1 of every K/V buffer."""
    reg, peer_ri = _registered(
        make_page_table(),
        make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4),
        tp_size=1,
        layer_num_per_pp=[2],
    )
    # 256B buffers of 2 x 128B heads into 512B buffers of 4.
    assert runs_of(kv_map(reg, peer_ri, (0, 0), (0, 0))) == [
        (0, 0, 256),
        (256, 512, 256),
        (512, 1024, 256),
        (768, 1536, 256),
    ]


def test_peer_registrar_kv_map_head_mismatch_second_rank():
    """TP2 rank 1 fills heads 2-3 of the TP1 peer's buffers."""
    reg = _make_peer_registrar(
        make_rankinfo(instance_name="local", tp_rank=1, page_table=make_page_table(tp_rank=1))
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        page_table=make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4),
    )
    assert runs_of(kv_map(reg, peer_ri, (0, 0), (0, 0))) == [
        (0, 256, 256),
        (256, 768, 256),
        (512, 1280, 256),
        (768, 1792, 256),
    ]


def test_peer_registrar_kv_map_uses_physical_offset_order():
    """Layer bytes are found by their slot offsets, not by sorted global id.

    ``self`` lays out its two global layers in physical order ``[5, 3]`` (layer 3 sits at byte
    512); ``peer`` holds only layer 3. A sort by global id would pick layer 5's bytes.
    """
    reg, peer_ri = _registered(
        make_page_table(global_layer_ids=[5, 3]),
        make_page_table(global_layer_ids=[3], block_bytes=[512]),
        layer_num_per_pp=[1],
    )
    assert runs_of(kv_map(reg, peer_ri, (0, 0), (0, 0))) == [(512, 0, 512)]


def test_peer_registrar_kv_map_non_contiguous_overlap_splits_fragments():
    """Shared layers that are not one contiguous run of the slot move as separate fragments.

    ``self`` holds layers ``[0, 1, 2]`` and ``peer`` holds ``[0, 2]``: one fragment that spanned
    both would carry layer 1's bytes over layer 2's.
    """
    reg, peer_ri = _registered(
        make_page_table(global_layer_ids=[0, 1, 2], block_bytes=[1536]),
        make_page_table(global_layer_ids=[0, 2]),
        layer_num_per_pp=[2],
    )
    mapper = kv_map(reg, peer_ri, (0, 0), (0, 0))
    assert runs_of(mapper) == [(0, 0, 512), (1024, 512, 512)]
    # Fragments of one size share a group.
    ((src, dst, _),) = mapper.map(np.array([1000]), np.array([2000]))
    assert src.tolist() == [1000, 2024]
    assert dst.tolist() == [2000, 2512]


def test_peer_registrar_kv_map_merges_non_monotonic_contiguous_layout():
    """Layers contiguous on both sides merge into one fragment whatever their global ids' order."""
    reg, peer_ri = _registered(
        make_page_table(global_layer_ids=[5, 3]),
        make_page_table(global_layer_ids=[5, 3]),
        layer_num_per_pp=[2],
    )
    assert runs_of(kv_map(reg, peer_ri, (0, 0), (0, 0))) == [(0, 0, 1024)]


def test_rank_info_carries_the_page_layout_to_peers():
    """The page layout survives the bytes a rank info is sent as.

    A peer's runs come from the layout its rank info carries.
    """
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        page_table=make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4, side_state_bytes=8),
    )
    received = RankInfo.from_bytes(peer_ri.to_bytes())
    assert received.page_table.page_layout == peer_ri.page_table.page_layout

    reg = _make_peer_registrar(make_rankinfo(instance_name="local", page_table=make_page_table()))
    assert runs_of(kv_map(reg, received, (0, 0), (0, 0))) == runs_of(
        kv_map(reg, peer_ri, (0, 0), (0, 0))
    )


def test_peer_registrar_kv_map_leaves_out_excluded_buffers():
    """Buffers a transfer leaves for the receiver to rebuild move no bytes."""
    reg, peer_ri = _registered(make_page_table(), make_page_table(), layer_num_per_pp=[2])
    assert runs_of(kv_map(reg, peer_ri, (0, 0), (0, 0))) == [(0, 0, 1024)]
    layer_1 = frozenset({(1, "key"), (1, "value")})
    assert runs_of(kv_map(reg, peer_ri, (0, 0), (0, 0), excluded=layer_1)) == [(0, 0, 512)]


def test_peer_registrar_kv_maps_are_cached_until_unregister():
    reg, peer_ri = _registered(make_page_table(), make_page_table(), layer_num_per_pp=[2])
    maps = reg.get_kv_maps(peer_ri)
    assert reg.get_kv_maps(peer_ri) is maps
    reg.unregister(peer_ri.instance_name, peer_ri.instance_rank)
    assert reg.get_kv_maps(peer_ri) is not maps


def test_peer_registrar_requires_a_page_table():
    reg = _make_peer_registrar(make_rankinfo(instance_name="local"))
    peer_ri = make_rankinfo(instance_name="peer", page_table=None)
    with pytest.raises(ValueError, match="carries no page table"):
        reg.get_kv_maps(peer_ri)
    with pytest.raises(ValueError, match="carries no page table"):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_replicated_pool_moves_whole_buffers_across_tensor_ranks():
    reg = _make_peer_registrar(
        make_rankinfo(
            instance_name="local",
            page_table=make_page_table(attention="replicated"),
        )
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=3,
        tp_size=1,
        layer_num_per_pp=[2],
        page_table=make_page_table(attention="replicated", mamba_tp=1),
    )
    assert runs_of(kv_map(reg, peer_ri, (0, 0), (0, 0))) == [(0, 0, 1024)]


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
        senders.append(kv_map(_make_peer_registrar(self_ri), peer_ri, (0, 0), (0, 0)) is not None)

    # Eight local ranks fan into the peer's single attention rank: one sends, at the slot the
    # destination's DP rank selects.
    assert senders.count(True) == 1
    assert senders.index(True) == peer_dp_rank % 8


@pytest.mark.parametrize("attention", ["head_major", "token_major", "replicated"])
@pytest.mark.parametrize("peer_tokens", [8, 32])
def test_peer_registrar_rejects_pages_of_other_token_counts(attention, peer_tokens):
    """Ranks whose pages hold different token counts cannot exchange them.

    Pages pair one to one by ordinal, whether or not one count divides the other.
    """
    reg = _make_peer_registrar(make_rankinfo(page_table=make_page_table(attention=attention)))
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=7,
        page_table=make_page_table(attention=attention, tokens_per_block=peer_tokens),
    )
    with pytest.raises(ValueError, match="pages of 16 tokens against"):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_runs_mapper_groups_fragments_by_size():
    mapper = RunsMapper(
        src_offsets=np.array([0, 192, 320]),
        dst_offsets=np.array([0, 128, 256]),
        sizes=np.array([128, 64, 128]),
    )
    assert mapper.bytes_per_slot == 320

    (small, large) = mapper.map(np.array([1000, 5000]), np.array([2000, 6000]))

    assert small[2] == 64
    assert small[0].tolist() == [1192, 5192]
    assert small[1].tolist() == [2128, 6128]
    # Slot-major: each slot's fragments of one size are adjacent.
    assert large[2] == 128
    assert large[0].tolist() == [1000, 1320, 5000, 5320]
    assert large[1].tolist() == [2000, 2256, 6000, 6256]


def test_runs_mapper_without_runs_moves_nothing():
    mapper = RunsMapper(*(np.zeros(0, dtype=np.int64),) * 3)
    assert mapper.bytes_per_slot == mapper.runs_per_slot == 0


def test_runs_mapper_rejects_unequal_run_arrays():
    with pytest.raises(ValueError, match="as many offsets as sizes"):
        RunsMapper(np.array([0, 8]), np.array([0]), np.array([8, 8]))


def test_runs_mapper_rejects_unequal_region_counts():
    mapper = RunsMapper(np.array([0]), np.array([0]), np.array([8]))
    with pytest.raises(ValueError, match="must match"):
        mapper.map(np.array([1000, 2000]), np.array([3000]))


def test_peer_registrar_token_major_head_mismatch_slices_each_token():
    """NHD buffers: rank 0's two heads of each token land in the first two of the peer's four."""
    reg, peer_ri = _registered(
        make_page_table(attention="token_major"),
        make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4, attention="token_major"),
        tp_size=1,
    )
    mapper = kv_map(reg, peer_ri, (0, 0), (0, 0))

    # Self buffers: 16 tokens x 2 heads x 8B; peer buffers: 16 tokens x 4 heads x 8B.
    assert runs_of(mapper) == [
        (buffer * 256 + token * 16, buffer * 512 + token * 32, 16)
        for buffer in range(4)
        for token in range(16)
    ]


def test_peer_registrar_warns_when_pages_move_in_fragments(monkeypatch):
    """Token-major K/V over different head counts moves a fragment per token and buffer."""
    self_ri = make_rankinfo(tp_size=2, page_table=make_page_table(attention="token_major"))
    peer_ri = make_rankinfo(
        instance_name="peer",
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
    # 4 buffers of 16 tokens, each token's two heads a fragment of its own.
    assert "64 fragments for 4 buffers" in message
    assert key == "native-fragmented-pages-64-4"


def test_peer_registrar_token_major_head_match_moves_whole_slots():
    """NHD with equal heads: the runs merge into one fragment per block."""
    reg, peer_ri = _registered(
        make_page_table(attention="token_major"),
        make_page_table(attention="token_major"),
    )
    assert runs_of(kv_map(reg, peer_ri, (0, 0), (0, 0))) == [(0, 0, 1024)]


# ---------------------------------------------------------------------------
# Recurrent state: the policy is chosen by CacheKind; which bytes move follows from the
# layouts, replicated side state included.
# ---------------------------------------------------------------------------

SIDE_STATE_POOL = 2  # conv state, SSM state, then the side state


def make_tp1_peer(**page_table_kwargs):
    """A TP1 peer holding all four KV heads and the whole recurrent state."""
    return make_rankinfo(
        instance_name="peer",
        tp_size=1,
        page_table=make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4, **page_table_kwargs),
    )


def test_mamba_replicated_side_state_elects_one_sender():
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
        conv_senders.append(kv_map(reg, peer_ri, (1, 0), (1, 0)) is not None)
        side = kv_map(reg, peer_ri, (1, SIDE_STATE_POOL), (1, SIDE_STATE_POOL))
        side_senders.append(side is not None)

    assert conv_senders == [True, True]
    assert side_senders == [True, False]


def test_mamba_replicated_side_state_copies_whole_layers_under_tp_mismatch():
    """Replicated state ignores mamba TP, while the conv state of the same group reshards."""
    reg = _make_peer_registrar(
        make_rankinfo(
            instance_name="local",
            tp_size=2,
            page_table=make_page_table(mamba_tp=2, side_state_bytes=8),
        )
    )
    peer_ri = make_tp1_peer(side_state_bytes=8)

    side = kv_map(reg, peer_ri, (1, SIDE_STATE_POOL), (1, SIDE_STATE_POOL))
    assert runs_of(side) == [(0, 0, 16)]
    # Rank 0 holds the first half of each conv section: [512 | 256 | 256] of [1024 | 512 | 512].
    conv = kv_map(reg, peer_ri, (1, 0), (1, 0))
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

    assert runs_of(kv_map(reg, peer_ri, (1, 1), (1, 1))) == [
        (0, 2048, 2048),
        (2048, 6144, 2048),
    ]


def test_a_buffer_one_rank_lacks_is_refused():
    """Side state transferred by one rank only would leave the other's zeroed."""
    reg = _make_peer_registrar(
        make_rankinfo(instance_name="local", page_table=make_page_table(side_state_bytes=8))
    )
    peer_ri = make_rankinfo(instance_name="peer", page_table=make_page_table())
    with pytest.raises(ValueError, match="ple_ngram_context.*transferred by one rank only"):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_an_index_cache_one_rank_lacks_is_refused():
    reg = _make_peer_registrar(
        make_rankinfo(instance_name="local", page_table=make_page_table(index_key_bytes=32))
    )
    peer_ri = make_rankinfo(instance_name="peer", page_table=make_page_table())
    with pytest.raises(ValueError, match="index_key.*transferred by one rank only"):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_side_state_of_another_size_is_refused():
    reg = _make_peer_registrar(
        make_rankinfo(instance_name="local", page_table=make_page_table(side_state_bytes=8))
    )
    peer_ri = make_rankinfo(instance_name="peer", page_table=make_page_table(side_state_bytes=16))
    with pytest.raises(ValueError, match="'byte': 8.*'byte': 16"):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_context_parallelism_on_both_sides_is_refused():
    reg = _make_peer_registrar(make_rankinfo(instance_name="local", cp_size=2))
    peer_ri = make_rankinfo(instance_name="peer", cp_size=2, page_table=make_page_table())
    with pytest.raises(ValueError, match="context parallelism on both sides"):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_a_receiver_asks_each_writer_what_it_sends(monkeypatch):
    """On first contact, the receiver records what each context rank writing it sends.

    It asks every context rank that may write its pages for that rank's own rank info, before it
    registers with the instance.
    """
    transfer = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    writers = [
        make_rankinfo(
            instance_name="ctx",
            instance_rank=r,
            tp_size=2,
            tp_rank=r,
            page_table=make_page_table(tp_rank=r, index_key_bytes=32),
        )
        for r in range(2)
    ]
    endpoints = ["tcp://ctx-0", "tcp://ctx-1"]
    replies = {
        "tcp://info": dataclasses.replace(writers[0], sender_endpoints=endpoints),
        **dict(zip(endpoints, writers)),
    }
    sent = []

    class Messenger:
        def __init__(self, mode, endpoint):
            self.endpoint = endpoint

        def send(self, messages):
            sent.append((self.endpoint, messages[0]))

        def receive(self):
            return [replies[self.endpoint].to_bytes()]

        def stop(self):
            pass

    monkeypatch.setattr(transfer, "ZMQMessenger", Messenger)
    receiver = object.__new__(transfer.Receiver)
    receiver._registrar = _make_peer_registrar(
        make_rankinfo(
            instance_name="gen",
            tp_size=1,
            page_table=make_page_table(
                block_bytes=[2048], mamba_tp=1, kv_heads=4, index_key_bytes=32
            ),
        )
    )
    receiver._incompatible_peers = {}
    receiver._sender_ep_instance_map = {}
    receiver._dealers = {}

    info = receiver._get_sender_info(SimpleNamespace(ctx_info_endpoint="tcp://info"))

    kind = transfer.MessageType
    assert sent == [
        ("tcp://info", kind.REQUEST_INSTANCE_INFO),
        ("tcp://ctx-0", kind.REQUEST_RANK_INFO),
        ("tcp://ctx-1", kind.REQUEST_RANK_INFO),
        ("tcp://ctx-0", kind.REGISTER_RANK_INFO),
        ("tcp://ctx-1", kind.REGISTER_RANK_INFO),
    ]
    assert [receiver._registrar.writer_share(info, r) for r in range(2)] == [
        {0: 1024 + 64, 1: 2 * (1024 + 2048)},
        {0: 1024, 1: 2 * (1024 + 2048)},
    ]


def test_each_writer_s_share_is_what_it_sends_into_a_page():
    """Two TP ranks writing a TP1 rank's pages have different shares.

    Each sends its heads of K/V and its share of the recurrent state, and one of them the
    replicated index cache.
    """
    receiver = make_rankinfo(
        instance_name="gen",
        tp_size=1,
        page_table=make_page_table(block_bytes=[2048], mamba_tp=1, kv_heads=4, index_key_bytes=32),
    )
    reg = _make_peer_registrar(receiver)
    writers = [
        make_rankinfo(
            instance_name="ctx",
            instance_rank=r,
            tp_size=2,
            tp_rank=r,
            page_table=make_page_table(tp_rank=r, index_key_bytes=32),
        )
        for r in range(2)
    ]
    shares = [reg.add_writer(writer) for writer in writers]
    assert [reg.writer_share(writers[0], r) for r in range(2)] == shares
    # Per attention page: 2 of 4 heads of two layers' K and V (256 B each), and the 32 B index-K
    # of both layers from one writer; per state slot: half of both layers' conv (1024 B) and SSM
    # (2048 B) state.
    assert shares == [{0: 1024 + 64, 1: 2 * (1024 + 2048)}, {0: 1024, 1: 2 * (1024 + 2048)}]

    stray = make_rankinfo(
        instance_name="ctx", instance_rank=2, page_table=make_page_table(side_state_bytes=8)
    )
    with pytest.raises(ValueError, match="transferred by one rank only"):
        reg.add_writer(stray)
