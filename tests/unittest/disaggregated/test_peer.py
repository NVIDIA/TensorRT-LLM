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
    LayerGroupAttrs,
    PoolRole,
)


def make_kv_pool_attrs(pool_ptrs=None, block_bytes=None, global_layer_ids=None):
    """Create a KVPoolAttrs for testing."""
    if pool_ptrs is None:
        pool_ptrs = [1234]
    if block_bytes is None:
        block_bytes = [1024]
    if global_layer_ids is None:
        global_layer_ids = [0]
    layer_group_attrs = LayerGroupAttrs(
        group_id=0,
        pool_base_ptrs=pool_ptrs,
        pool_sizes=[0],
        roles_to_pool_idx={PoolRole.KV_CACHE: 0},
        block_bytes_per_pool=block_bytes,
        global_layer_ids=global_layer_ids,
        kv_head_num_per_rank=2,
    )
    layer_to_group_id = {lid: 0 for lid in global_layer_ids}
    return KVPoolAttrs(
        layer_to_group_id=layer_to_group_id,
        layer_group_attrs_list=[layer_group_attrs],
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
    kv_pool_attrs=None,
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
        server_endpoint="",
        self_endpoint="",
        transfer_engine_info=b"",
        aux_meta=None,
        kv_pool_attrs=kv_pool_attrs,
    )


def _make_peer_registrar_and_peer_ri(self_ri, peer_ri):
    pool_attrs = make_kv_pool_attrs()
    real_kv_extractor = KVRegionExtractorV1(pool_attrs)
    reg = PeerRegistrar(self_ri, real_kv_extractor)
    # Ensure peer_ri has kv_pool_attrs for registration
    if peer_ri.kv_pool_attrs is None:
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
            kv_heads_per_rank=peer_ri.kv_heads_per_rank,
            tokens_per_block=peer_ri.tokens_per_block,
            dims_per_head=peer_ri.dims_per_head,
            element_bytes=peer_ri.element_bytes,
            is_mla=peer_ri.is_mla,
            enable_attention_dp=peer_ri.enable_attention_dp,
            layer_num_per_pp=peer_ri.layer_num_per_pp,
            kv_pool_attrs=make_kv_pool_attrs(),
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
    pool_attrs = (
        self_rankinfo.kv_pool_attrs if self_rankinfo.kv_pool_attrs else make_kv_pool_attrs()
    )
    real_kv_extractor = KVRegionExtractorV1(pool_attrs)
    reg = PeerRegistrar(self_rankinfo, real_kv_extractor)
    return reg


def test_peer_registrar_register_and_get():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=1,
        layer_num_per_pp=[2],
        kv_pool_attrs=make_kv_pool_attrs(),
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
        kv_pool_attrs=make_kv_pool_attrs(),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    reg.unregister(peer_ri.instance_name, peer_ri.instance_rank)
    with pytest.raises(KeyError):
        reg.get_peer_rank_info("peer", 1)


def test_peer_registrar_incompatible_peer_raises():
    self_rankinfo = make_rankinfo(instance_name="local")
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer", instance_rank=3, is_mla=True, kv_pool_attrs=make_kv_pool_attrs()
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
        kv_pool_attrs=make_kv_pool_attrs(),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri)
    assert isinstance(mapper, IdentityMapper)


def test_peer_registrar_get_kv_map_head_match():
    # Self has 2 layers [0, 1], peer has only layer [1]
    # Overlap is [1], but self has more layers -> HeadMatchMapper needed for layer offset
    self_rankinfo = make_rankinfo(
        instance_name="local",
        kv_pool_attrs=make_kv_pool_attrs(global_layer_ids=[0, 1]),
    )
    reg = _make_peer_registrar(self_rankinfo)
    peer_ri = make_rankinfo(
        instance_name="peer",
        instance_rank=2,
        pp_size=2,
        pp_rank=1,
        layer_num_per_pp=[1, 1],
        tokens_per_block=16,
        dims_per_head=8,
        kv_pool_attrs=make_kv_pool_attrs(global_layer_ids=[1]),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    # self has layers [0,1], peer has layer [1], overlap is [1]
    # transfer_layers=1, self_group_layer_count=2, peer_group_layer_count=1
    # 1 != 2, so HeadMatchMapper is returned
    mapper = reg.get_kv_map(peer_ri, self_layer_group_id=0, peer_layer_group_id=0)
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
        kv_heads_per_rank=4,  # Different from self (default is 2)
        tokens_per_block=16,
        dims_per_head=8,
        layer_num_per_pp=[2],
        kv_pool_attrs=make_kv_pool_attrs(),
    )
    reg.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = reg.get_kv_map(peer_ri, self_layer_group_id=0, peer_layer_group_id=0)
    assert isinstance(mapper, HeadMismatchMapper)
