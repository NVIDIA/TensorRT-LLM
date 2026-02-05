from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.region.block import (
    HeadMatchMapper,
    HeadMismatchMapper,
    IdentityMapper,
)


def make_rankinfo(
    kv_heads_per_rank=2,
    tokens_per_block=4,
    dims_per_head=2,
    element_bytes=1,
    tp_size=2,
    tp_rank=0,
    dp_size=1,
    dp_rank=0,
    pp_size=1,
    pp_rank=0,
    cp_size=1,
    cp_rank=0,
    is_mla=False,
):
    return RankInfo(
        instance_name="rank",
        instance_rank=0,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dp_size=dp_size,
        dp_rank=dp_rank,
        pp_size=pp_size,
        pp_rank=pp_rank,
        cp_size=cp_size,
        cp_rank=cp_rank,
        device_id=0,
        kv_heads_per_rank=kv_heads_per_rank,
        tokens_per_block=tokens_per_block,
        dims_per_head=dims_per_head,
        element_bytes=element_bytes,
        enable_attention_dp=False,
        is_mla=is_mla,
        layer_num_per_pp=[1],
        server_endpoint="",
        self_endpoint="",
        transfer_engine_info=b"",
        aux_meta=None,
        kv_pool_attrs=None,
    )


def test_mem_region_group():
    ptrs = [11, 22, 33]
    bytes_per_region = 16
    region = MemRegionGroup(ptrs=ptrs, bytes_per_region=bytes_per_region)
    assert list(region.ptrs) == ptrs
    assert region.bytes_per_region == bytes_per_region


def test_spec_region_and_spec_region_pair():
    group_src = MemRegionGroup(ptrs=[101, 202], bytes_per_region=8)
    group_dst = MemRegionGroup(ptrs=[303, 404], bytes_per_region=8)
    spec_src = SpecRegion(memory=group_src, spec="spec_src")
    spec_dst = SpecRegion(memory=group_dst, spec="spec_dst")
    assert isinstance(spec_src, SpecRegion)
    assert isinstance(spec_dst, SpecRegion)
    pair = SpecRegionPair(src=spec_src, dst=spec_dst)
    assert isinstance(pair, SpecRegionPair)
    assert pair.src.memory.ptrs == [101, 202]
    assert pair.dst.memory.ptrs == [303, 404]
    assert pair.src.spec == "spec_src"
    assert pair.dst.spec == "spec_dst"


def test_identity_mapper():
    src_group = MemRegionGroup(ptrs=[100, 200], bytes_per_region=32)
    dst_group = MemRegionGroup(ptrs=[300, 400], bytes_per_region=32)
    src_spec = SpecRegion(memory=src_group, spec="a")
    dst_spec = SpecRegion(memory=dst_group, spec="b")
    mapper = IdentityMapper()
    result = mapper.map(src_spec, dst_spec)
    assert isinstance(result, SpecRegionPair)
    assert list(result.src.memory.ptrs) == [100, 200]
    assert list(result.dst.memory.ptrs) == [300, 400]
    assert result.src.memory.bytes_per_region == 32
    assert result.dst.memory.bytes_per_region == 32


def test_head_match_mapper():
    self_ri = make_rankinfo(kv_heads_per_rank=2)
    peer_ri = make_rankinfo(kv_heads_per_rank=2)
    transfer_layers = 2
    src_layer_off = 1
    dst_layer_off = 1
    src_group = MemRegionGroup(ptrs=[10, 20], bytes_per_region=1)
    dst_group = MemRegionGroup(ptrs=[30, 40], bytes_per_region=1)
    src_spec = SpecRegion(memory=src_group, spec="srcspec")
    dst_spec = SpecRegion(memory=dst_group, spec="dstspec")
    mapper = HeadMatchMapper(transfer_layers, src_layer_off, dst_layer_off, self_ri, peer_ri)
    result = mapper.map(src_spec, dst_spec)
    expected_off = (
        transfer_layers
        * mapper._kv_factor
        * self_ri.kv_heads_per_rank
        * self_ri.tokens_per_block
        * self_ri.dims_per_head
        * self_ri.element_bytes
    )
    assert list(result.src.memory.ptrs) == [10 + mapper._src_block_off, 20 + mapper._src_block_off]
    assert list(result.dst.memory.ptrs) == [30 + mapper._dst_block_off, 40 + mapper._dst_block_off]
    assert result.src.memory.bytes_per_region == expected_off
    assert result.dst.memory.bytes_per_region == expected_off


def test_head_mismatch_mapper():
    self_ri = make_rankinfo(kv_heads_per_rank=2, tp_size=2, tp_rank=1)
    peer_ri = make_rankinfo(kv_heads_per_rank=4, tp_size=4, tp_rank=2)
    transfer_layers = 1
    src_layer_off = 0
    peer_layer_off = 1
    src_group = MemRegionGroup(ptrs=[111], bytes_per_region=32)
    dst_group = MemRegionGroup(ptrs=[222], bytes_per_region=32)
    src_spec = SpecRegion(memory=src_group, spec="srcspec")
    dst_spec = SpecRegion(memory=dst_group, spec="dstspec")
    mapper = HeadMismatchMapper(transfer_layers, src_layer_off, peer_layer_off, self_ri, peer_ri)
    result = mapper.map(src_spec, dst_spec)
    expected_frag_count = self_ri.kv_factor * transfer_layers
    assert isinstance(result, SpecRegionPair)
    assert len(result.src.memory.ptrs) == expected_frag_count
    assert len(result.dst.memory.ptrs) == expected_frag_count
    assert all(isinstance(x, int) for x in result.src.memory.ptrs)
    assert all(isinstance(x, int) for x in result.dst.memory.ptrs)
    assert result.src.memory.bytes_per_region == mapper._bytes_cont_heads
    assert result.dst.memory.bytes_per_region == mapper._bytes_cont_heads


def test_rankinfo_kv_factor():
    ri1 = make_rankinfo(is_mla=False)
    ri2 = make_rankinfo(is_mla=True)
    assert ri1.kv_factor == 2
    assert ri2.kv_factor == 1
