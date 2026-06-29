import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import (
    HeadMatchMapper,
    HeadMismatchMapper,
    IdentityMapper,
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

    # Build buffer entries: K + V per local layer
    buffer_size = 256  # bytes per buffer entry (arbitrary for tests)
    entries = []
    for i in range(len(global_layer_ids)):
        base_offset = i * buffer_size * 2
        entries.append((i, base_offset, buffer_size))
        entries.append((i, base_offset + buffer_size, buffer_size))
    buffer_entries = np.array(entries, dtype=BUFFER_ENTRY_DTYPE)

    local_layers = [
        LocalLayer(local_layer_id=i, global_layer_id=gid) for i, gid in enumerate(global_layer_ids)
    ]
    pool_views = [
        PoolView(pool_idx=pi, buffer_entries=buffer_entries) for pi in range(len(pool_ptrs))
    ]
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
    assert isinstance(mapper, IdentityMapper)


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
    assert isinstance(mapper, HeadMatchMapper)


def test_peer_registrar_get_kv_map_head_mismatch():
    self_rankinfo = make_rankinfo(instance_name="local", page_table=make_page_table())
    reg = _make_peer_registrar(self_rankinfo)
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
        page_table=make_page_table(),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert isinstance(mapper, HeadMismatchMapper)
