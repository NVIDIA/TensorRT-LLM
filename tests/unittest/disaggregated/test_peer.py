import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.native.peer as peer_module
from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import (
    HeadMatchMapper,
    HeadMismatchMapper,
    IdentityMapper,
    NHDHeadMismatchMapper,
    PoolBufferMapper,
    PoolBufferMapping,
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


def test_peer_registrar_get_kv_map_uses_physical_offset_order():
    """The layer slot offset must follow physical buffer order, not sorted global id.

    ``self`` lays out its two global layers in physical order ``[5, 3]`` (layer 3
    sits at physical slot 1); ``peer`` holds only layer 3. The transfer must copy
    ``self``'s slot at offset 1 -- a sort-by-global-id would wrongly pick offset 0
    (layer 5's bytes).
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
    assert isinstance(mapper, HeadMatchMapper)
    # slot_size_per_layer = 1024 // 2 = 512; layer 3 is at physical slot 1 on
    # self and slot 0 on peer.
    assert mapper._src_block_off == 512
    assert mapper._dst_block_off == 0


def test_peer_registrar_get_kv_map_rejects_non_contiguous_overlap():
    """Shared layers that are not an aligned contiguous slot run must be rejected.

    ``self`` holds layers ``[0, 1, 2]`` and ``peer`` holds ``[0, 2]``: the overlap
    ``{0, 2}`` is not contiguous within ``self``'s slot, so a single contiguous
    fragment transfer would corrupt layer 1's bytes. ``get_kv_map`` must raise
    rather than emit a wrong mapping.
    """
    self_pt = make_page_table(global_layer_ids=[0, 1, 2])
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
    with pytest.raises(ValueError, match="aligned contiguous run"):
        reg.get_kv_map(peer_ri, (0, 0), (0, 0))


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
        transfer_layers=1,
        src_layer_off=0,
        peer_layer_off=0,
        self_ri=self_ri,
        peer_ri=peer_ri,
        self_region_bytes=32,
        peer_region_bytes=16,
        self_pool_num_layers=1,
        peer_pool_num_layers=1,
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
        transfer_layers=1,
        src_layer_off=0,
        peer_layer_off=0,
        self_ri=self_ri,
        peer_ri=peer_ri,
        self_region_bytes=8,
        peer_region_bytes=4,
        self_pool_num_layers=1,
        peer_pool_num_layers=1,
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


def test_pool_buffer_mapper_uses_whole_pool_identity_for_equal_layout():
    self_ri = make_rankinfo(instance_name="local", kv_heads_per_rank=2)
    peer_ri = make_rankinfo(instance_name="peer", kv_heads_per_rank=2)
    mapper = PoolBufferMapper(
        mappings=[
            PoolBufferMapping(0, 0, 256, 256, MapperKind.NHD),
            PoolBufferMapping(256, 256, 128, 128, MapperKind.REPLICATED),
        ],
        self_ri=self_ri,
        peer_ri=peer_ri,
        self_region_bytes=384,
        peer_region_bytes=384,
        full_region_identity=True,
        include_sharded=True,
        include_replicated=True,
    )

    pair = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000, 2000]), bytes_per_region=384)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([3000, 4000]), bytes_per_region=384)),
    )

    assert isinstance(pair, SpecRegionPair)
    assert pair.src.memory.ptrs.size == 2  # One descriptor per input block.
    assert pair.src.memory.ptrs.tolist() == [1000, 2000]
    assert pair.dst.memory.ptrs.tolist() == [3000, 4000]
    assert pair.src.memory.bytes_per_region == 384


def test_pool_buffer_mapper_uses_entry_offsets_for_head_mismatch():
    self_ri = make_rankinfo(
        instance_name="local",
        tp_size=2,
        kv_heads_per_rank=2,
        tokens_per_block=2,
        dims_per_head=2,
        element_bytes=2,
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=1,
        kv_heads_per_rank=1,
        tokens_per_block=2,
        dims_per_head=2,
        element_bytes=2,
    )
    mapper = PoolBufferMapper(
        mappings=[
            PoolBufferMapping(0, 0, 16, 8, MapperKind.NHD),
            PoolBufferMapping(16, 8, 4, 4, MapperKind.REPLICATED),
        ],
        self_ri=self_ri,
        peer_ri=peer_ri,
        self_region_bytes=20,
        peer_region_bytes=12,
        full_region_identity=False,
        include_sharded=True,
        include_replicated=True,
    )

    pairs = mapper.map(
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([1000]), bytes_per_region=20)),
        SpecRegion(memory=MemRegionGroup(ptrs=np.array([2000]), bytes_per_region=12)),
    )

    assert len(pairs) == 1
    assert pairs[0].src.memory.ptrs.tolist() == [1000, 1008, 1016]
    assert pairs[0].dst.memory.ptrs.tolist() == [2000, 2004, 2008]
    assert pairs[0].src.memory.bytes_per_region == 4


def test_replicated_mapper_ignores_kv_head_mismatch():
    self_pt = make_page_table(global_layer_ids=[0])
    peer_pt = make_page_table(global_layer_ids=[0])
    for page_table in (self_pt, peer_pt):
        view = page_table.layer_groups[0].pool_views[0]
        view.pool_role = frozenset({"index_key"})
        view.mapper_kind = MapperKind.REPLICATED
        view.bytes_per_region = 256

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


def test_replicated_pool_has_single_owner_on_tp_fan_in():
    page_table = make_page_table(global_layer_ids=[0])
    view = page_table.layer_groups[0].pool_views[0]
    view.mapper_kind = MapperKind.REPLICATED

    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=8,
        dp_size=8,
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
        ownership.append(reg.should_send_pool(overlap, peer_ri, 0, 0, 0, 0))

    assert ownership == [True, False, False, False, False, False, False, False]


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


def test_identity_mapper_region_size_mismatch_raises():
    with pytest.raises(ValueError, match="Identity cache region size mismatch"):
        IdentityMapper(256, 128)


def test_replicated_mapper_region_size_mismatch_raises():
    with pytest.raises(ValueError, match="Replicated cache region size mismatch"):
        ReplicatedMapper(256, 128)


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
            transfer_layers=1,
            src_layer_off=0,
            peer_layer_off=0,
            self_ri=self_ri,
            peer_ri=peer_ri,
            self_region_bytes=17,
            peer_region_bytes=32,
            self_pool_num_layers=1,
            peer_pool_num_layers=1,
            self_buffers_per_layer=2,
            peer_buffers_per_layer=2,
        )


def test_nhd_mapper_rejects_tokens_per_block_mismatch():
    self_ri = make_rankinfo(tokens_per_block=2)
    peer_ri = make_rankinfo(instance_name="peer", tokens_per_block=4)

    with pytest.raises(ValueError, match="requires equal tokens_per_block"):
        NHDHeadMismatchMapper(
            transfer_layers=1,
            src_layer_off=0,
            peer_layer_off=0,
            self_ri=self_ri,
            peer_ri=peer_ri,
            self_region_bytes=32,
            peer_region_bytes=64,
            self_pool_num_layers=1,
            peer_pool_num_layers=1,
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


def test_indexed_mapper_ignores_non_uniform_buffer_entry_geometry():
    """Legacy/DSV4 indexed pools do not need NHD buffer geometry."""
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

    mapper = reg.get_kv_map(peer_ri, (0, 0), (0, 0))

    assert isinstance(mapper, IdentityMapper)


def test_peer_registrar_rejects_legacy_subbyte_head_mismatch():
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
