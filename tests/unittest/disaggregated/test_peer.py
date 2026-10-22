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
from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.disaggregation.cache_specs import (
    ATTENTION_TP,
    STATE_TP,
    BufferDecl,
    LayerGroup,
    PageTable,
    Pool,
    Region,
    plan_transfers,
    spec_for,
    spec_named,
    validate_peer_compatible,
)
from tensorrt_llm._torch.disaggregation.cache_specs.specs.variants import AttentionSpec, MambaSpec
from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    AuxBufferMeta,
    build_aux_transfer_layout,
)
from tensorrt_llm._torch.disaggregation.native.peer import PeerOverlap, PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import AttentionInfo, RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1

pytestmark = pytest.mark.cpu_only

_KV_ROLES = ("key", "value")
MAMBA_CONV_ROLE = "conv_state"
MAMBA_SSM_ROLE = "ssm_state"
_ATTN_SPACE = 0
_STATE_SPACE = 1


def _attention_decl(layout, nbytes, heads, tokens_per_block):
    """The declaration of one attention buffer laid out as ``layout`` (hnd / nhd / replicated)."""
    if layout == "replicated":
        return BufferDecl(nbytes, ATTENTION_TP)
    if layout == "nhd":
        axes = (("token", tokens_per_block), ("head", heads))
    else:
        axes = (("head", heads),)
    return BufferDecl(nbytes, ATTENTION_TP, axes, "head")


def make_page_table(
    pool_ptr=1234,
    block_bytes=1024,
    global_layer_ids=None,
    mamba_tp=2,
    kv_heads=2,
    tokens_per_block=16,
    attn_layout="hnd",
):
    """Create a hybrid page table: one paged attention layer group plus one recurrent-state layer group.

    The attention layer group holds a K and a V region per layer, sized so the layers fill the
    pool slot. ``mamba_tp`` is the TP degree the synthetic recurrent-state layer group is sharded
    for: per-rank sizes are the ``mamba_tp`` shard of a fixed global state, so page tables
    built for different TP degrees stay peer-compatible under the mamba spec's global
    (TP-aggregated) size check. Pass the rank's effective mamba TP (``make_rankinfo``
    defaults to ``tp_size=2``).
    """
    if global_layer_ids is None:
        global_layer_ids = [0, 1]

    buffer_size = block_bytes // (len(global_layer_ids) * 2)
    attn_regions = []
    for i, gid in enumerate(global_layer_ids):
        for j, role in enumerate(_KV_ROLES):
            attn_regions.append(
                Region(
                    pool=0,
                    layer=gid,
                    role=role,
                    offset=(2 * i + j) * buffer_size,
                    buffers=(
                        _attention_decl(attn_layout, buffer_size, kv_heads, tokens_per_block),
                    ),
                    local_layer=i,
                )
            )
    attn = LayerGroup(
        spec="mamba",
        kind=CacheKind.PAGED,
        regions=attn_regions,
        params={"tokens_per_block": tokens_per_block, "window": None},
    )

    conv_slot_bytes = 4096 // mamba_tp
    ssm_slot_bytes = 8192 // mamba_tp
    conv_sections = [2048 // mamba_tp, 1024 // mamba_tp, 1024 // mamba_tp]
    state_regions = []
    for lid, gid in enumerate([100, 101]):
        state_regions.append(
            Region(
                pool=1,
                layer=gid,
                role=MAMBA_CONV_ROLE,
                offset=lid * conv_slot_bytes,
                buffers=tuple(
                    BufferDecl(s, STATE_TP, (("byte", s),), "byte") for s in conv_sections
                ),
                local_layer=lid,
            )
        )
    for lid, gid in enumerate([100, 101]):
        state_regions.append(
            Region(
                pool=2,
                layer=gid,
                role=MAMBA_SSM_ROLE,
                offset=lid * ssm_slot_bytes,
                buffers=(
                    BufferDecl(ssm_slot_bytes, STATE_TP, (("head", ssm_slot_bytes // 64),), "head"),
                ),
                local_layer=lid,
            )
        )
    state = LayerGroup(spec="mamba", kind=CacheKind.STATE, regions=state_regions)

    return PageTable(
        pools=[
            Pool(base_address=pool_ptr, slot_bytes=block_bytes, num_slots=128),
            Pool(base_address=0xA000, slot_bytes=2 * conv_slot_bytes, num_slots=128),
            Pool(base_address=0xB000, slot_bytes=2 * ssm_slot_bytes, num_slots=128),
        ],
        layer_groups=[attn, state],
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


def _fragments(region_pairs):
    """Every ``(src, dst, bytes)`` fragment of a plan's ``map`` result, sorted by source."""
    return sorted(
        (int(s), int(d), int(p.src.memory.bytes_per_region))
        for p in region_pairs
        for s, d in zip(p.src.memory.ptrs.tolist(), p.dst.memory.ptrs.tolist())
    )


def _group(groups, layer_group=_ATTN_SPACE, pool=None):
    """The single transfer group of a local layer group (and pool, when given)."""
    found = [
        g
        for g in groups
        if g.self_layer_group == layer_group and (pool is None or g.self_pool == pool)
    ]
    assert len(found) == 1, found
    return found[0]


def _attention_plan(reg, peer_ri):
    """The run plan of the attention layer group's group to ``peer_ri``."""
    return _group(reg.transfer_groups(peer_ri)).plan


def _single_group_table(layer_offsets, bytes_per_layer, heads, tokens_per_block, buffers, layout):
    """One attention layer group, one pool: global layer ``i`` at ``layer_offsets[i]`` of the slot.

    Layer ``i`` holds ``buffers`` equal regions (roles key, value) back to back; ``None``
    offsets skip a global id, so two tables can overlap on any subset of layers.
    """
    regions = []
    roles = _KV_ROLES[:buffers]
    assert len(roles) == buffers
    for gid, offset in enumerate(layer_offsets):
        if offset is None:
            continue
        sizes = [bytes_per_layer // buffers] * buffers
        sizes[-1] += bytes_per_layer - sum(sizes)
        for role, size in zip(roles, sizes):
            regions.append(
                Region(
                    pool=0,
                    layer=gid,
                    role=role,
                    offset=offset,
                    buffers=(_attention_decl(layout, size, heads, tokens_per_block),),
                )
            )
            offset += size
    slot = max(o for o in layer_offsets if o is not None) + bytes_per_layer
    return PageTable(
        pools=[Pool(base_address=0, slot_bytes=slot, num_slots=4)],
        layer_groups=[
            LayerGroup(
                spec="attention",
                kind=CacheKind.PAGED,
                regions=regions,
                params={"tokens_per_block": tokens_per_block, "window": None},
            )
        ],
    )


def _plan(
    self_ri, peer_ri, self_offsets, peer_offsets, self_bpl, peer_bpl, buffers=2, layout="hnd"
):
    """The run plan between two single-layer-group tables built by ``_single_group_table``.

    Each side declares its own rank's KV heads and tokens per block.
    """
    tables = [
        _single_group_table(
            offsets,
            bpl,
            ri.attention.kv_heads_per_rank,
            ri.attention.tokens_per_block,
            buffers,
            layout,
        )
        for ri, offsets, bpl in (
            (self_ri, self_offsets, self_bpl),
            (peer_ri, peer_offsets, peer_bpl),
        )
    ]
    groups = plan_transfers(self_ri, peer_ri, *tables)
    assert len(groups) == 1, groups
    return groups[0].plan


def _map(plan, src_ptrs, dst_ptrs):
    return _fragments(
        plan.map(
            SpecRegion(memory=MemRegionGroup(ptrs=np.array(src_ptrs), bytes_per_region=0)),
            SpecRegion(memory=MemRegionGroup(ptrs=np.array(dst_ptrs), bytes_per_region=0)),
        )
    )


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


def test_peer_registrar_transfer_groups_identity():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=1,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    plan = _attention_plan(reg, peer_ri)
    # Full contiguous overlap: one whole-region fragment per block.
    pairs = plan.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=1024)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=1024)),
    )
    assert len(pairs) == 1
    assert pairs[0].src.memory.ptrs.tolist() == [1000]
    assert pairs[0].src.memory.bytes_per_region == 1024


def test_peer_registrar_transfer_groups_are_cached_per_peer_and_exclusion():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=1,
        layer_num_per_pp=[2],
        page_table=make_page_table(),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    groups = reg.transfer_groups(peer_ri)
    assert reg.transfer_groups(peer_ri) is groups
    # Excluding the first attention region (layer 0 key) drops it from the plan only.
    excluded = frozenset({(_ATTN_SPACE, 0)})
    partial = reg.transfer_groups(peer_ri, excluded)
    assert partial is not groups
    assert reg.transfer_groups(peer_ri, excluded) is partial
    assert _group(partial).regions == (1, 2, 3)
    assert _group(partial).plan.runs == [(256, 256, 768)]

    reg.unregister(peer_ri.instance_name, peer_ri.instance_rank)
    assert reg.transfer_groups(peer_ri) is not groups


def test_peer_registrar_transfer_groups_head_match():
    self_pt = make_page_table(global_layer_ids=[0, 1])
    peer_pt = make_page_table(global_layer_ids=[1], block_bytes=512)

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
    plan = _attention_plan(reg, peer_ri)
    # Partial layer overlap with matching heads: a single shifted fragment.
    (pair,) = plan.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=1024)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=512)),
    )
    # Overlap is global layer 1: self slot holds layers [0, 1] (512B each),
    # peer slot holds only layer 1.
    assert pair.src.memory.ptrs.tolist() == [1512]
    assert pair.dst.memory.ptrs.tolist() == [2000]
    assert pair.src.memory.bytes_per_region == 512


def test_peer_registrar_transfer_groups_head_mismatch():
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
        page_table=make_page_table(block_bytes=2048, mamba_tp=1, kv_heads=4),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    plan = _attention_plan(reg, peer_ri)
    # Local TP rank 0 holds heads [0, 2) of the peer's 4: each local K/V buffer (256B) lands
    # at the start of the peer's 512B buffer, one fragment per (layer, buffer).
    assert plan.runs == [(0, 0, 256), (256, 512, 256), (512, 1024, 256), (768, 1536, 256)]


def test_peer_registrar_transfer_groups_use_physical_offset_order():
    """The layer byte offset must follow physical buffer order, not sorted global id.

    ``self`` lays out its two global layers in physical order ``[5, 3]`` (layer 3
    sits at physical slot 1); ``peer`` holds only layer 3. The transfer must copy
    ``self``'s slot at byte offset 512 -- a sort-by-global-id would wrongly pick
    offset 0 (layer 5's bytes). Every region carries its own offset, so no
    ordering convention is involved.
    """
    self_pt = make_page_table(global_layer_ids=[5, 3])
    peer_pt = make_page_table(global_layer_ids=[3], block_bytes=512)

    self_rankinfo = make_rankinfo(instance_name="local", page_table=self_pt)
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=2,
        layer_num_per_pp=[1],
        page_table=peer_pt,
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    plan = _attention_plan(reg, peer_ri)
    (pair,) = plan.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=1024)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=512)),
    )
    # Layer 3 (512B) sits at physical slot 1 on self and slot 0 on peer.
    assert pair.src.memory.ptrs.tolist() == [1512]
    assert pair.dst.memory.ptrs.tolist() == [2000]
    assert pair.src.memory.bytes_per_region == 512


def test_peer_registrar_non_contiguous_overlap_splits_fragments():
    """Shared layers that are not a contiguous slot run transfer as separate fragments.

    ``self`` holds layers ``[0, 1, 2]`` and ``peer`` holds ``[0, 2]``: the overlap
    ``{0, 2}`` is not contiguous within ``self``'s slot, so a single contiguous
    fragment transfer would corrupt layer 1's bytes. Regions select layers by
    explicit byte offsets, so the plan must emit one correctly-shifted fragment
    per contiguous run instead of rejecting (or silently corrupting) the transfer.
    """
    self_pt = make_page_table(global_layer_ids=[0, 1, 2], block_bytes=1536)
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
    plan = _attention_plan(reg, peer_ri)
    # Layer 0: self offset 0 -> peer offset 0; layer 2: self offset 1024 ->
    # peer offset 512. Non-adjacent on self, so two fragments.
    assert _map(plan, [1000], [2000]) == [(1000, 2000, 512), (2024, 2512, 512)]


def test_peer_registrar_merges_non_monotonic_contiguous_layout():
    """Physically contiguous layers merge into one fragment despite global-id order.

    Both peers lay out global layers in physical order ``[5, 3]``: byte-contiguous
    on both sides, just not monotonic in global id. Iterating the overlap in
    self's physical order keeps the runs adjacent, so the plan merges the two
    layers into a single 1024B fragment; a sorted-by-global-id iteration would
    visit offsets ``[512, 0]`` and split the copy into two fragments.
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
    plan = _attention_plan(reg, peer_ri)
    pairs = plan.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=1024)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=1024)),
    )
    assert len(pairs) == 1
    assert pairs[0].src.memory.ptrs.tolist() == [1000]
    assert pairs[0].dst.memory.ptrs.tolist() == [2000]
    assert pairs[0].src.memory.bytes_per_region == 1024


def test_nhd_head_mismatch_slices_each_token():
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
    plan = _plan(self_ri, peer_ri, [0], [0], 32, 16, layout="nhd")

    # Source NHD has two heads per token: select head 0 at offsets 0 and 8
    # for K, then 16 and 24 for V. Destination has one head per token.
    assert _map(plan, [1000], [2000]) == [
        (1000, 2000, 4),
        (1008, 2004, 4),
        (1016, 2008, 4),
        (1024, 2012, 4),
    ]


def test_nhd_head_mismatch_uses_scale_pool_geometry():
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
    plan = _plan(self_ri, peer_ri, [0], [0], 8, 4, layout="nhd")

    assert _map(plan, [1000], [2000]) == [
        (1000, 2000, 1),
        (1002, 2001, 1),
        (1004, 2002, 1),
        (1006, 2003, 1),
    ]


def test_replicated_region_ignores_kv_head_mismatch():
    self_pt = _single_group_table([0], 1024, 1, 16, buffers=2, layout="replicated")
    peer_pt = _single_group_table([0], 1024, 8, 16, buffers=2, layout="replicated")

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
    plan = _attention_plan(reg, peer_ri)
    # Whole layer, whatever the head counts.
    assert plan.runs == [(0, 0, 1024)]


@pytest.mark.parametrize("peer_dp_rank", [0, 1, 3, 7])
def test_replicated_region_owner_rotates_by_destination_dp_rank(peer_dp_rank):
    """Exactly one owner per fan-in group, rotated by destination DP rank.

    Mirrors the C++ MLACacheFormatter::needSendCache pairing so the
    replicated traffic spreads across local ranks for multi-DP generation.
    """
    page_table = _single_group_table([0], 1024, 2, 16, buffers=2, layout="replicated")

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
        group = _group(reg.transfer_groups(peer_ri))
        assert group.owner_factor == 8
        ownership.append(reg.owns(group, peer_ri))

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


@pytest.mark.parametrize("attn_layout", ["nhd", "replicated"])
def test_peer_registrar_exact_tpb_layout_rejects_divisible_mismatch(attn_layout):
    self_pt = make_page_table(attn_layout=attn_layout)
    peer_pt = make_page_table(attn_layout=attn_layout, tokens_per_block=32)
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


def test_whole_copy_region_size_mismatch_raises():
    ri = make_rankinfo(kv_heads_per_rank=1)
    with pytest.raises(ValueError, match="cache region size mismatch"):
        _plan(ri, ri, [0], [0], 256, 128, buffers=1, layout="replicated")


def test_replicated_per_layer_size_mismatch_raises():
    ri = make_rankinfo()
    with pytest.raises(ValueError, match="cache region size mismatch"):
        _plan(ri, ri, [0, 128], [0, 64], 128, 64, layout="replicated")


def test_replicated_selects_partial_layer_range():
    """PP mismatch selects the overlap layers via explicit offsets.

    The peer slot holds a superset of layers; only the overlap moves, and
    contiguous layers on both sides merge into one fragment.
    """
    ri = make_rankinfo()
    # self slot: layers 1..2 x 128B; peer slot: layers 0..2 x 128B, overlap at 1..2.
    plan = _plan(ri, ri, [None, 0, 128], [0, 128, 256], 128, 128, layout="replicated")

    assert _map(plan, [1000], [2000]) == [(1000, 2128, 256)]


def test_whole_copy_splits_non_contiguous_runs():
    """Interleaved slots (another role class between layers) split runs.

    Source layers sit at non-uniform strides (something interleaves after
    layer 0); destination is densely packed. Runs must break where either
    side is discontiguous, and each fragment must carry the run's bytes.
    """
    ri = make_rankinfo()
    plan = _plan(ri, ri, [0, 192, 320], [0, 128, 256], 128, 128)

    # Run 1: layer 0 alone (src gap breaks the run); run 2: layers 1-2 merged.
    assert _map(plan, [1000], [2000]) == [(1000, 2000, 128), (1192, 2128, 256)]


def test_attention_region_rejects_non_divisible_layer_geometry():
    self_ri = make_rankinfo(kv_heads_per_rank=2, tokens_per_block=2)
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        kv_heads_per_rank=4,
        tokens_per_block=2,
    )

    with pytest.raises(ValueError, match="not evenly divisible"):
        _plan(self_ri, peer_ri, [0], [0], 17, 32, layout="nhd")


def test_nhd_head_mismatch_rejects_tokens_per_block_mismatch():
    self_ri = make_rankinfo(kv_heads_per_rank=2, tokens_per_block=2)
    peer_ri = make_rankinfo(
        instance_name="peer", tp_size=1, kv_heads_per_rank=4, tokens_per_block=4
    )

    with pytest.raises(ValueError, match="tokens_per_block"):
        _plan(self_ri, peer_ri, [0], [0], 32, 128, layout="nhd")


def test_buffer_declarations_reject_invalid_geometry():
    """A region's layout is validated where it is declared."""
    with pytest.raises(ValueError, match="split axis 'head' is not one of the axes"):
        BufferDecl(16, ATTENTION_TP, (("token", 2),), "head")
    with pytest.raises(ValueError, match="at least one byte"):
        BufferDecl(0, ATTENTION_TP)
    with pytest.raises(ValueError, match="declares no buffers"):
        Region(pool=0, layer=0, role="key", offset=0, buffers=())


def test_mismatched_region_layouts_rejected_at_registration():
    """Matched regions must declare the same layout shape on both sides.

    A region whose layout differs from its peer's cannot be addressed generically
    and must fail loudly at registration instead of transferring garbage.
    """
    self_pt = make_page_table()
    peer_pt = make_page_table(attn_layout="nhd")
    self_ri = make_rankinfo(page_table=self_pt)
    peer_ri = make_rankinfo(instance_name="peer", page_table=peer_pt)
    reg = _make_peer_registrar(self_ri)

    with pytest.raises(ValueError, match=r"mamba region \(layer=0, role='key'\) buffer 0: layout"):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


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
        page_table=make_page_table(block_bytes=2048, mamba_tp=1, kv_heads=4),
    )
    reg = _make_peer_registrar(self_ri)
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    plan = _attention_plan(reg, peer_ri)

    # Head-sliced: one fragment per (layer, buffer), each half of a peer buffer.
    assert [n for _, _, n in plan.runs] == [256] * 4


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
        page_table=make_page_table(block_bytes=2048, mamba_tp=1, kv_heads=4),
    )
    reg = _make_peer_registrar(self_ri)

    with pytest.raises(ValueError):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_peer_registrar_slices_nhd_heads_per_token():
    self_pt = make_page_table(attn_layout="nhd")
    peer_pt = make_page_table(block_bytes=2048, mamba_tp=1, kv_heads=4, attn_layout="nhd")
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

    plan = _attention_plan(reg, peer_ri)

    # One fragment per (layer, buffer, token): 2 layers x K/V x 16 tokens.
    assert len(plan.runs) == 2 * 2 * 16
    assert {n for _, _, n in plan.runs} == {16}


def test_peer_registrar_warns_for_nhd_head_mismatch(monkeypatch):
    self_pt = make_page_table(attn_layout="nhd")
    peer_pt = make_page_table(block_bytes=2048, mamba_tp=1, kv_heads=4, attn_layout="nhd")
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


def test_peer_registrar_nhd_head_match_moves_whole_region():
    """NHD + equal heads takes the merged-run fast path, not per-token slicing.

    With a dedicated (non-interleaved) K/V pool the run merge collapses the
    whole layer group's regions into a single fragment per block — the fragment-count
    budget for separate layouts.
    """
    self_pt = make_page_table(attn_layout="nhd")
    peer_pt = make_page_table(attn_layout="nhd")
    self_ri = make_rankinfo(page_table=self_pt)
    peer_ri = make_rankinfo(instance_name="peer", instance_rank=9, page_table=peer_pt)
    reg = _make_peer_registrar(self_ri)
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)

    plan = _attention_plan(reg, peer_ri)
    pairs = plan.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000, 5000]), bytes_per_region=1024)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000, 6000]), bytes_per_region=1024)),
    )
    # Fully contiguous on both sides -> exactly one fragment per block.
    assert len(pairs) == 1
    assert pairs[0].src.memory.ptrs.tolist() == [1000, 5000]
    assert pairs[0].dst.memory.ptrs.tolist() == [2000, 6000]
    assert pairs[0].src.memory.bytes_per_region == 1024


def test_hnd_head_mismatch_subbyte_geometry_is_declaration_derived():
    """HND head-mismatch byte math comes from region bytes, never element_bytes.

    Emulates an NVFP4-like cache: element_bytes is fractional (0.5), but the
    buffer size registered by storage is whole bytes, so every derived offset
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
    # 2 layers x 256B/layer (K/V x 128B) vs 2 layers x 128B/layer (K/V x 64B).
    plan = _plan(self_ri, peer_ri, [0, 256], [0, 128], 256, 128)

    (pair,) = plan.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=512)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=256)),
    )
    # peer tp_rank=1 selects head 1 inside self's per-rank pair of heads:
    # src head offset = 1 x 64B; fragments = (layer, k/v) x 2 layers.
    assert pair.src.memory.ptrs.tolist() == [1064, 1192, 1320, 1448]
    assert pair.dst.memory.ptrs.tolist() == [2000, 2064, 2128, 2192]
    assert pair.src.memory.bytes_per_region == 64
    assert pair.src.memory.ptrs.dtype == np.int64
    assert pair.dst.memory.ptrs.dtype == np.int64


def test_hnd_head_mismatch_inconsistent_slot_geometry_raises():
    self_ri = make_rankinfo(tp_size=1, kv_heads_per_rank=2)
    peer_ri = make_rankinfo(instance_name="peer", tp_size=2, kv_heads_per_rank=1)
    with pytest.raises(ValueError, match="bytes per 'head' unit mismatch"):
        # The peer should hold 128B per layer for half the heads.
        _plan(self_ri, peer_ri, [0, 256], [0, 256], 256, 256)


# ---------------------------------------------------------------------------
# Spec dispatch: a layer group's spec name selects its spec; each region's buffer
# declarations (replicated included) decide how its bytes map and who sends them.
# ---------------------------------------------------------------------------


def _add_replicated_side_regions(page_table, *, bytes_per_layer=8):
    """Graft a replicated recurrent side role (e.g. PLE) onto the state layer group, in its own pool."""
    page_table.pools.append(
        Pool(base_address=0xC000, slot_bytes=2 * bytes_per_layer, num_slots=128)
    )
    pool = len(page_table.pools) - 1
    page_table.layer_groups[_STATE_SPACE].regions.extend(
        Region(
            pool=pool,
            layer=gid,
            role="ple_ngram_context",
            offset=lid * bytes_per_layer,
            buffers=(BufferDecl(bytes_per_layer, STATE_TP),),
            local_layer=lid,
        )
        for lid, gid in enumerate([100, 101])
    )
    return pool


def test_spec_dispatch_keys_on_spec_name_only():
    page_table = make_page_table()
    assert [layer_group.kind for layer_group in page_table.layer_groups] == [
        CacheKind.PAGED,
        CacheKind.STATE,
    ]
    assert all(
        isinstance(spec_for(layer_group), MambaSpec) for layer_group in page_table.layer_groups
    )
    attention_only = _single_group_table([0], 64, 2, 16, buffers=2, layout="hnd")
    assert isinstance(spec_for(attention_only.layer_groups[0]), AttentionSpec)
    assert isinstance(spec_named("attention"), AttentionSpec)
    assert isinstance(spec_named("mamba"), MambaSpec)
    with pytest.raises(ValueError, match="unknown cache spec 'linear_state'"):
        spec_named("linear_state")


def test_mamba_replicated_side_region_elects_one_sender():
    """Replicated side state elects one sender per fan-in group.

    Sharded conv state in the same layer group is sent by every mamba TP rank.
    """
    peer_pt = make_page_table(mamba_tp=1)
    _add_replicated_side_regions(peer_pt)
    peer_ri = make_rankinfo(instance_name="peer", tp_size=1, page_table=peer_pt)

    conv_ownership = []
    side_ownership = []
    for tp_rank in range(2):
        self_pt = make_page_table(mamba_tp=2)
        side_pool = _add_replicated_side_regions(self_pt)
        self_ri = make_rankinfo(
            instance_name="local", tp_size=2, tp_rank=tp_rank, page_table=self_pt
        )
        reg = _make_peer_registrar(self_ri)
        groups = reg.transfer_groups(peer_ri)
        conv_ownership.append(reg.owns(_group(groups, _STATE_SPACE, pool=1), peer_ri))
        side_ownership.append(reg.owns(_group(groups, _STATE_SPACE, pool=side_pool), peer_ri))

    assert conv_ownership == [True, True]
    assert side_ownership == [True, False]


def test_mamba_replicated_side_region_copies_whole_layers_under_tp_mismatch():
    """Replicated state ignores the state grid; sharded conv state reshards per section."""
    self_pt = make_page_table(mamba_tp=2)
    self_side = _add_replicated_side_regions(self_pt)
    peer_pt = make_page_table(mamba_tp=1)
    _add_replicated_side_regions(peer_pt)
    self_ri = make_rankinfo(instance_name="local", tp_size=2, page_table=self_pt)
    peer_ri = make_rankinfo(instance_name="peer", tp_size=1, page_table=peer_pt)
    reg = _make_peer_registrar(self_ri)
    groups = reg.transfer_groups(peer_ri)

    side = _group(groups, _STATE_SPACE, pool=self_side).plan
    assert side.runs == [(0, 0, 16)]
    conv = _group(groups, _STATE_SPACE, pool=1).plan
    # TP rank 0 of 2 holds the first half of each of the three sections [x | B | C], per layer.
    assert conv.runs == [
        (0, 0, 1024),
        (1024, 2048, 512),
        (1536, 3072, 512),
        (2048, 4096, 1024),
        (3072, 6144, 512),
        (3584, 7168, 512),
    ]


def test_replicated_validation_covers_state_spaces():
    """A replicated role held on one side only is rejected for the spec owning the state."""
    self_pt = make_page_table()
    _add_replicated_side_regions(self_pt)
    peer_pt = make_page_table()
    ri = make_rankinfo(instance_name="local", page_table=self_pt)
    validate_peer_compatible(ri, ri, self_pt, self_pt)
    with pytest.raises(ValueError, match="mamba: replicated roles differ on overlapping layers"):
        validate_peer_compatible(ri, ri, self_pt, peer_pt)


def test_attention_replicated_validation_rejects_missing_index_key_region():
    self_pt = _single_group_table([0], 64, 2, 16, buffers=2, layout="hnd")
    self_pt.layer_groups[0].regions.append(
        Region(
            pool=0, layer=0, role="index_key", offset=64, buffers=(BufferDecl(16, ATTENTION_TP),)
        )
    )
    peer_pt = _single_group_table([0], 64, 2, 16, buffers=2, layout="hnd")
    ri = make_rankinfo(instance_name="local", page_table=self_pt)
    with pytest.raises(
        ValueError, match=r"attention: replicated roles differ on overlapping layers.*index_key"
    ):
        validate_peer_compatible(ri, ri, self_pt, peer_pt)
