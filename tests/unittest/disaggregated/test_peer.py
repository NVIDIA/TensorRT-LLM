import pytest

from tensorrt_llm._torch.disaggregation.native.peer import PeerOverlap, PeerRegistrar, RankInfo
from tensorrt_llm._torch.disaggregation.native.region.block import (
    HeadMatchMapper,
    HeadMismatchMapper,
    IdentityMapper,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVPoolAttrs,
    KVRegionExtractorV1,
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
        kv_heads_per_rank=kv_heads_per_rank,
        tokens_per_block=tokens_per_block,
        dims_per_head=dims_per_head,
        element_bytes=element_bytes,
        enable_attention_dp=enable_attention_dp,
        is_mla=is_mla,
        layer_num_per_pp=layer_num_per_pp,
        kv_ptrs=[],
        aux_ptrs=[],
        server_endpoint="",
        self_endpoint="",
        transfer_engine_info=b"",
        aux_meta=None,
    )


def _make_peer_registrar_and_peer_ri(self_ri, peer_ri):
    pool_attrs = KVPoolAttrs(pool_ptrs=[1234], block_bytes=[1024])
    real_kv_extractor = KVRegionExtractorV1(pool_attrs)
    reg = PeerRegistrar(self_ri, real_kv_extractor)
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
    assert overlap.target_peer_pp_layer_num == [1, 1]
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
    assert overlap.target_peer_pp_layer_num == []
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
    assert sum(overlap.target_peer_pp_layer_num) > 0
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
        target_peer_pp_layer_num=[1, 1],
        ranks=[26, 27, 30, 31, 42, 43, 46, 47],
    )
    assert overlap == expected_overlap


def _make_peer_registrar(self_rankinfo):
    pool_attrs = KVPoolAttrs(pool_ptrs=[1234], block_bytes=[1024])
    real_kv_extractor = KVRegionExtractorV1(pool_attrs)
    reg = PeerRegistrar(self_rankinfo, real_kv_extractor)
    return reg


def test_peer_registrar_register_and_get():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(instance_name="peer", instance_rank=1, layer_num_per_pp=[2])
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    assert reg.get_peer_rank_info("peer", 1) == peer_ri


def test_peer_registrar_unregister():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(instance_name="peer", instance_rank=1, layer_num_per_pp=[2])
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    reg.unregister(peer_ri.instance_name, peer_ri.instance_rank)
    with pytest.raises(KeyError):
        reg.get_peer_rank_info("peer", 1)


def test_peer_registrar_incompatible_peer_raises():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(instance_name="peer", instance_rank=3, is_mla=True)
    with pytest.raises(ValueError):
        reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)


def test_peer_registrar_self_rank_info_property():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    assert reg.self_rank_info == self_rankinfo


def test_peer_registrar_get_kv_map_identity():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(instance_name="peer", instance_rank=1, layer_num_per_pp=[2])
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri)
    assert isinstance(mapper, IdentityMapper)


def test_peer_registrar_get_kv_map_head_match():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=2,
        pp_size=2,
        pp_rank=1,
        layer_num_per_pp=[1, 1],
        tokens_per_block=16,
        dims_per_head=8,
    )
    mapper = reg.get_kv_map(peer_ri)
    assert isinstance(mapper, HeadMatchMapper)


def test_peer_registrar_get_kv_map_head_mismatch():
    self_rankinfo = make_rankinfo(instance_name="local")
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
    )
    mapper = reg.get_kv_map(peer_ri)
    assert isinstance(mapper, HeadMismatchMapper)
