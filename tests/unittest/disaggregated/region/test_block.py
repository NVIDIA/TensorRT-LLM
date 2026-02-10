from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.region.block import (
    ConvStateMismatchMapper,
    HeadMatchMapper,
    HeadMismatchMapper,
    IdentityMapper,
    MambaHeadMatchMapper,
    MambaHeadMismatchMapper,
    _compute_tp_offsets,
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
        page_table=None,
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


# ============== Mamba Mapper Tests ==============


def test_mamba_head_match_mapper_basic():
    """Test MambaHeadMatchMapper with basic layer selection."""
    # Simulate Mamba state regions with 4 layers
    # Input: 4 addresses (one per layer for a single slot)
    num_layers = 4
    block_bytes = 256  # bytes per slot per layer
    transfer_layers = 2
    src_layer_off = 1  # Start from layer 1 in source
    dst_layer_off = 0  # Start from layer 0 in destination

    # Source has 4 layers, destination has 4 layers
    src_base = 1000
    dst_base = 5000
    src_ptrs = [src_base + i * block_bytes for i in range(num_layers)]
    dst_ptrs = [dst_base + i * block_bytes for i in range(num_layers)]

    src_group = MemRegionGroup(ptrs=src_ptrs, bytes_per_region=block_bytes)
    dst_group = MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=block_bytes)
    src_spec = SpecRegion(memory=src_group, spec="src_mamba")
    dst_spec = SpecRegion(memory=dst_group, spec="dst_mamba")

    mapper = MambaHeadMatchMapper(
        transfer_layers=transfer_layers,
        src_layer_off=src_layer_off,
        dst_layer_off=dst_layer_off,
        block_bytes_per_layer=block_bytes,
    )

    result = mapper.map(src_spec, dst_spec)

    assert isinstance(result, SpecRegionPair)
    # Should only select transfer_layers addresses
    assert len(result.src.memory.ptrs) == transfer_layers
    assert len(result.dst.memory.ptrs) == transfer_layers

    # Source: layers 1, 2 (indices 1, 2)
    expected_src_ptrs = src_ptrs[src_layer_off : src_layer_off + transfer_layers]
    # Destination: layers 0, 1 (indices 0, 1)
    expected_dst_ptrs = dst_ptrs[dst_layer_off : dst_layer_off + transfer_layers]

    assert list(result.src.memory.ptrs) == expected_src_ptrs
    assert list(result.dst.memory.ptrs) == expected_dst_ptrs
    assert result.src.memory.bytes_per_region == block_bytes
    assert result.dst.memory.bytes_per_region == block_bytes


def test_mamba_head_match_mapper_all_layers():
    """Test MambaHeadMatchMapper when all layers overlap."""
    num_layers = 4
    block_bytes = 128

    src_ptrs = [100 + i * 100 for i in range(num_layers)]
    dst_ptrs = [500 + i * 100 for i in range(num_layers)]

    src_group = MemRegionGroup(ptrs=src_ptrs, bytes_per_region=block_bytes)
    dst_group = MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=block_bytes)
    src_spec = SpecRegion(memory=src_group)
    dst_spec = SpecRegion(memory=dst_group)

    # All layers overlap
    mapper = MambaHeadMatchMapper(
        transfer_layers=num_layers,
        src_layer_off=0,
        dst_layer_off=0,
        block_bytes_per_layer=block_bytes,
    )

    result = mapper.map(src_spec, dst_spec)

    assert len(result.src.memory.ptrs) == num_layers
    assert len(result.dst.memory.ptrs) == num_layers
    assert list(result.src.memory.ptrs) == src_ptrs
    assert list(result.dst.memory.ptrs) == dst_ptrs


def test_mamba_head_match_mapper_single_layer():
    """Test MambaHeadMatchMapper with single layer transfer."""
    block_bytes = 64

    src_ptrs = [1000, 2000, 3000, 4000]
    dst_ptrs = [5000, 6000, 7000, 8000]

    src_group = MemRegionGroup(ptrs=src_ptrs, bytes_per_region=block_bytes)
    dst_group = MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=block_bytes)
    src_spec = SpecRegion(memory=src_group)
    dst_spec = SpecRegion(memory=dst_group)

    # Transfer only 1 layer: src layer 2 -> dst layer 3
    mapper = MambaHeadMatchMapper(
        transfer_layers=1,
        src_layer_off=2,
        dst_layer_off=3,
        block_bytes_per_layer=block_bytes,
    )

    result = mapper.map(src_spec, dst_spec)

    assert len(result.src.memory.ptrs) == 1
    assert len(result.dst.memory.ptrs) == 1
    assert result.src.memory.ptrs[0] == 3000  # src layer 2
    assert result.dst.memory.ptrs[0] == 8000  # dst layer 3


def test_mamba_head_mismatch_mapper_basic():
    """Test MambaHeadMismatchMapper with different head counts."""
    num_layers = 4
    bytes_per_head = 64  # head_dim * d_state * element_bytes
    self_nheads = 8  # More heads (smaller TP)
    peer_nheads = 4  # Fewer heads (larger TP)

    # Self TP=2, Peer TP=4 -> ratio = 2
    self_tp_per_dp = 2
    peer_tp_per_dp = 4
    self_tp_rank = 0
    peer_tp_rank = 1

    transfer_layers = 2
    src_layer_off = 0
    dst_layer_off = 1

    # Source block bytes: nheads * bytes_per_head
    src_block_bytes = self_nheads * bytes_per_head
    dst_block_bytes = peer_nheads * bytes_per_head

    src_ptrs = [1000 + i * src_block_bytes for i in range(num_layers)]
    dst_ptrs = [5000 + i * dst_block_bytes for i in range(num_layers)]

    src_group = MemRegionGroup(ptrs=src_ptrs, bytes_per_region=src_block_bytes)
    dst_group = MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=dst_block_bytes)
    src_spec = SpecRegion(memory=src_group)
    dst_spec = SpecRegion(memory=dst_group)

    mapper = MambaHeadMismatchMapper(
        transfer_layers=transfer_layers,
        src_layer_off=src_layer_off,
        dst_layer_off=dst_layer_off,
        bytes_per_head=bytes_per_head,
        self_nheads=self_nheads,
        peer_nheads=peer_nheads,
        self_tp_per_dp=self_tp_per_dp,
        peer_tp_per_dp=peer_tp_per_dp,
        self_tp_rank=self_tp_rank,
        peer_tp_rank=peer_tp_rank,
    )

    result = mapper.map(src_spec, dst_spec)

    assert isinstance(result, SpecRegionPair)
    # Should have transfer_layers entries
    assert len(result.src.memory.ptrs) == transfer_layers
    assert len(result.dst.memory.ptrs) == transfer_layers

    # Bytes per region should be min(self_nheads, peer_nheads) * bytes_per_head
    expected_bytes_cont_heads = min(self_nheads, peer_nheads) * bytes_per_head
    assert result.src.memory.bytes_per_region == expected_bytes_cont_heads
    assert result.dst.memory.bytes_per_region == expected_bytes_cont_heads


def test_mamba_head_mismatch_mapper_same_tp():
    """Test MambaHeadMismatchMapper when TP sizes are the same (no head offset)."""
    num_layers = 3
    bytes_per_head = 32
    self_nheads = 4
    peer_nheads = 4

    # Same TP size -> no head offset
    self_tp_per_dp = 2
    peer_tp_per_dp = 2
    self_tp_rank = 0
    peer_tp_rank = 0

    transfer_layers = 2
    src_layer_off = 0
    dst_layer_off = 0

    block_bytes = self_nheads * bytes_per_head
    src_ptrs = [100 + i * 100 for i in range(num_layers)]
    dst_ptrs = [500 + i * 100 for i in range(num_layers)]

    src_group = MemRegionGroup(ptrs=src_ptrs, bytes_per_region=block_bytes)
    dst_group = MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=block_bytes)
    src_spec = SpecRegion(memory=src_group)
    dst_spec = SpecRegion(memory=dst_group)

    mapper = MambaHeadMismatchMapper(
        transfer_layers=transfer_layers,
        src_layer_off=src_layer_off,
        dst_layer_off=dst_layer_off,
        bytes_per_head=bytes_per_head,
        self_nheads=self_nheads,
        peer_nheads=peer_nheads,
        self_tp_per_dp=self_tp_per_dp,
        peer_tp_per_dp=peer_tp_per_dp,
        self_tp_rank=self_tp_rank,
        peer_tp_rank=peer_tp_rank,
    )

    result = mapper.map(src_spec, dst_spec)

    # When TP sizes are the same, head offsets should be 0
    # Addresses should be the selected layers directly
    assert len(result.src.memory.ptrs) == transfer_layers
    assert len(result.dst.memory.ptrs) == transfer_layers

    # With no offset, src layers 0,1 map directly
    assert list(result.src.memory.ptrs) == src_ptrs[:transfer_layers]
    assert list(result.dst.memory.ptrs) == dst_ptrs[:transfer_layers]


def test_mamba_head_mismatch_mapper_head_offsets():
    """Test TP offset computation (shared helper _compute_tp_offsets)."""
    bytes_cont_heads = 128

    # Case 1: self_tp < peer_tp -> src gets offset
    src_off, dst_off = _compute_tp_offsets(
        self_tp_per_dp=2,
        peer_tp_per_dp=4,
        self_tp_rank=0,
        peer_tp_rank=1,  # peer_tp_rank % ratio = 1 % 2 = 1
        transfer_bytes=bytes_cont_heads,
    )
    assert dst_off == 0
    assert src_off == 1 * bytes_cont_heads  # (peer_tp_rank % ratio) * bytes_cont_heads

    # Case 2: self_tp > peer_tp -> dst gets offset
    src_off, dst_off = _compute_tp_offsets(
        self_tp_per_dp=4,
        peer_tp_per_dp=2,
        self_tp_rank=3,  # self_tp_rank % ratio = 3 % 2 = 1
        peer_tp_rank=0,
        transfer_bytes=bytes_cont_heads,
    )
    assert src_off == 0
    assert dst_off == 1 * bytes_cont_heads  # (self_tp_rank % ratio) * bytes_cont_heads

    # Case 3: same TP -> no offset
    src_off, dst_off = _compute_tp_offsets(
        self_tp_per_dp=2,
        peer_tp_per_dp=2,
        self_tp_rank=1,
        peer_tp_rank=0,
        transfer_bytes=bytes_cont_heads,
    )
    assert src_off == 0
    assert dst_off == 0


def test_mamba_head_mismatch_mapper_with_layer_offset():
    """Test MambaHeadMismatchMapper with non-zero layer offsets and head mismatch."""
    num_layers = 6
    bytes_per_head = 16
    self_nheads = 8
    peer_nheads = 4

    # Self has more heads (smaller TP), peer has fewer (larger TP)
    self_tp_per_dp = 1
    peer_tp_per_dp = 2
    self_tp_rank = 0
    peer_tp_rank = 1  # Will add offset to src

    transfer_layers = 3
    src_layer_off = 2  # Start from layer 2
    dst_layer_off = 1  # Start from layer 1

    src_block_bytes = self_nheads * bytes_per_head
    dst_block_bytes = peer_nheads * bytes_per_head

    src_ptrs = [1000 * (i + 1) for i in range(num_layers)]
    dst_ptrs = [5000 * (i + 1) for i in range(num_layers)]

    src_group = MemRegionGroup(ptrs=src_ptrs, bytes_per_region=src_block_bytes)
    dst_group = MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=dst_block_bytes)
    src_spec = SpecRegion(memory=src_group)
    dst_spec = SpecRegion(memory=dst_group)

    mapper = MambaHeadMismatchMapper(
        transfer_layers=transfer_layers,
        src_layer_off=src_layer_off,
        dst_layer_off=dst_layer_off,
        bytes_per_head=bytes_per_head,
        self_nheads=self_nheads,
        peer_nheads=peer_nheads,
        self_tp_per_dp=self_tp_per_dp,
        peer_tp_per_dp=peer_tp_per_dp,
        self_tp_rank=self_tp_rank,
        peer_tp_rank=peer_tp_rank,
    )

    result = mapper.map(src_spec, dst_spec)

    # Should select layers based on offsets
    assert len(result.src.memory.ptrs) == transfer_layers
    assert len(result.dst.memory.ptrs) == transfer_layers

    # Verify bytes_per_region is min heads * bytes_per_head
    expected_bytes = min(self_nheads, peer_nheads) * bytes_per_head
    assert result.src.memory.bytes_per_region == expected_bytes
    assert result.dst.memory.bytes_per_region == expected_bytes

    # Verify layer selection: src layers 2,3,4 and dst layers 1,2,3
    # Plus head offset applied to src
    # src_head_off = (peer_tp_rank % ratio) * bytes_cont_heads = (1 % 2) * 64 = 64
    expected_src_head_off = 1 * expected_bytes
    expected_src_ptrs = [
        src_ptrs[src_layer_off + i] + expected_src_head_off for i in range(transfer_layers)
    ]
    expected_dst_ptrs = [dst_ptrs[dst_layer_off + i] for i in range(transfer_layers)]

    assert list(result.src.memory.ptrs) == expected_src_ptrs
    assert list(result.dst.memory.ptrs) == expected_dst_ptrs


# ============== ConvStateMismatchMapper Tests ==============


def test_conv_state_mismatch_mapper_same_tp():
    """ConvStateMismatchMapper with same TP -> no offset, transfer all bytes."""
    # Qwen3Next-like layout: [Q(64) | K(64) | V(128)] per rank (TP=2 both)
    self_sec = [64, 64, 128]
    peer_sec = [64, 64, 128]

    num_layers = 3
    transfer_layers = 2
    src_layer_off = 0
    dst_layer_off = 1

    src_ptrs = [1000 + i * 256 for i in range(num_layers)]
    dst_ptrs = [5000 + i * 256 for i in range(num_layers)]

    src_group = MemRegionGroup(ptrs=src_ptrs, bytes_per_region=256)
    dst_group = MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=256)
    src_spec = SpecRegion(memory=src_group)
    dst_spec = SpecRegion(memory=dst_group)

    mapper = ConvStateMismatchMapper(
        transfer_layers=transfer_layers,
        src_layer_off=src_layer_off,
        dst_layer_off=dst_layer_off,
        self_section_bytes=self_sec,
        peer_section_bytes=peer_sec,
        self_tp_per_dp=2,
        peer_tp_per_dp=2,
        self_tp_rank=0,
        peer_tp_rank=0,
    )

    result = mapper.map(src_spec, dst_spec)

    # Should return 3 SpecRegionPairs (one per section)
    assert isinstance(result, list)
    assert len(result) == 3

    # Section 0 (Q): offset=0, transfer=64
    rp0 = result[0]
    assert rp0.src.memory.bytes_per_region == 64
    assert len(rp0.src.memory.ptrs) == transfer_layers
    assert rp0.src.memory.ptrs[0] == src_ptrs[0] + 0  # offset 0
    assert rp0.dst.memory.ptrs[0] == dst_ptrs[1] + 0

    # Section 1 (K): offset=64, transfer=64
    rp1 = result[1]
    assert rp1.src.memory.bytes_per_region == 64
    assert rp1.src.memory.ptrs[0] == src_ptrs[0] + 64
    assert rp1.dst.memory.ptrs[0] == dst_ptrs[1] + 64

    # Section 2 (V): offset=128, transfer=128
    rp2 = result[2]
    assert rp2.src.memory.bytes_per_region == 128
    assert rp2.src.memory.ptrs[0] == src_ptrs[0] + 128
    assert rp2.dst.memory.ptrs[0] == dst_ptrs[1] + 128


def test_conv_state_mismatch_mapper_tp_mismatch():
    """ConvStateMismatchMapper with TP=2 (self) vs TP=4 (peer).

    Qwen3Next: global [Q=ng*ds | K=ng*ds | V=d_inner]
    self TP=2: per-rank sections = [ng*ds/2, ng*ds/2, d_inner/2]
    peer TP=4: per-rank sections = [ng*ds/4, ng*ds/4, d_inner/4]

    For concrete numbers: ng*ds=128, d_inner=256
    self: [64, 64, 128]
    peer: [32, 32, 64]
    """
    self_sec = [64, 64, 128]
    peer_sec = [32, 32, 64]

    num_layers = 4
    transfer_layers = 2
    src_layer_off = 0
    dst_layer_off = 0

    # Buffer layout: conv_state per slot per layer = sum(self_sec) = 256 bytes for self
    self_bytes_per_slot = sum(self_sec)  # 256
    peer_bytes_per_slot = sum(peer_sec)  # 128

    src_ptrs = [1000 + i * self_bytes_per_slot for i in range(num_layers)]
    dst_ptrs = [5000 + i * peer_bytes_per_slot for i in range(num_layers)]

    src_group = MemRegionGroup(ptrs=src_ptrs, bytes_per_region=self_bytes_per_slot)
    dst_group = MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=peer_bytes_per_slot)
    src_spec = SpecRegion(memory=src_group)
    dst_spec = SpecRegion(memory=dst_group)

    # self TP=2, peer TP=4, peer_tp_rank=1
    # ratio = 4/2 = 2, self_tp < peer_tp
    # -> src gets offset: (peer_tp_rank % ratio) * transfer_bytes
    #    for section 0: (1 % 2) * 32 = 32
    #    for section 1: (1 % 2) * 32 = 32
    #    for section 2: (1 % 2) * 64 = 64
    mapper = ConvStateMismatchMapper(
        transfer_layers=transfer_layers,
        src_layer_off=src_layer_off,
        dst_layer_off=dst_layer_off,
        self_section_bytes=self_sec,
        peer_section_bytes=peer_sec,
        self_tp_per_dp=2,
        peer_tp_per_dp=4,
        self_tp_rank=0,
        peer_tp_rank=1,
    )

    result = mapper.map(src_spec, dst_spec)

    assert len(result) == 3

    # Section 0 (Q): transfer_bytes=32, src_offset = 0 + 32 = 32, dst_offset = 0 + 0 = 0
    rp0 = result[0]
    assert rp0.src.memory.bytes_per_region == 32
    assert rp0.dst.memory.bytes_per_region == 32
    assert rp0.src.memory.ptrs[0] == src_ptrs[0] + 32  # section_start(0) + inner_off(32)
    assert rp0.dst.memory.ptrs[0] == dst_ptrs[0] + 0

    # Section 1 (K): transfer_bytes=32, src_offset = 64 + 32 = 96, dst_offset = 32 + 0 = 32
    rp1 = result[1]
    assert rp1.src.memory.bytes_per_region == 32
    assert rp1.src.memory.ptrs[0] == src_ptrs[0] + 96  # self_sec[0](64) + inner_off(32)
    assert rp1.dst.memory.ptrs[0] == dst_ptrs[0] + 32  # peer_sec[0](32) + inner_off(0)

    # Section 2 (V): transfer_bytes=64, src_offset = 128 + 64 = 192, dst_offset = 64 + 0 = 64
    rp2 = result[2]
    assert rp2.src.memory.bytes_per_region == 64
    assert rp2.src.memory.ptrs[0] == src_ptrs[0] + 192  # self_sec[0]+[1](128) + inner_off(64)
    assert rp2.dst.memory.ptrs[0] == dst_ptrs[0] + 64  # peer_sec[0]+[1](64) + inner_off(0)

    # Verify second layer pointers have same offsets
    assert rp0.src.memory.ptrs[1] == src_ptrs[1] + 32
    assert rp0.dst.memory.ptrs[1] == dst_ptrs[1] + 0


def test_conv_state_mismatch_mapper_reverse_tp():
    """ConvStateMismatchMapper with TP=4 (self) vs TP=2 (peer).

    Now self has finer granularity, peer has coarser.
    self_sec = [32, 32, 64]
    peer_sec = [64, 64, 128]
    self_tp_rank=3, ratio=2
    -> dst gets offset: (self_tp_rank % ratio) * transfer_bytes
       for section 0: (3 % 2) * 32 = 32
    """
    self_sec = [32, 32, 64]
    peer_sec = [64, 64, 128]

    num_layers = 2
    transfer_layers = 2
    src_layer_off = 0
    dst_layer_off = 0

    src_ptrs = [1000 + i * 128 for i in range(num_layers)]
    dst_ptrs = [5000 + i * 256 for i in range(num_layers)]

    src_group = MemRegionGroup(ptrs=src_ptrs, bytes_per_region=128)
    dst_group = MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=256)
    src_spec = SpecRegion(memory=src_group)
    dst_spec = SpecRegion(memory=dst_group)

    mapper = ConvStateMismatchMapper(
        transfer_layers=transfer_layers,
        src_layer_off=src_layer_off,
        dst_layer_off=dst_layer_off,
        self_section_bytes=self_sec,
        peer_section_bytes=peer_sec,
        self_tp_per_dp=4,
        peer_tp_per_dp=2,
        self_tp_rank=3,
        peer_tp_rank=0,
    )

    result = mapper.map(src_spec, dst_spec)
    assert len(result) == 3

    # Section 0: transfer=32, src_inner=0, dst_inner=(3%2)*32=32
    rp0 = result[0]
    assert rp0.src.memory.bytes_per_region == 32
    assert rp0.src.memory.ptrs[0] == src_ptrs[0] + 0  # section_start(0) + 0
    assert rp0.dst.memory.ptrs[0] == dst_ptrs[0] + 32  # section_start(0) + 32

    # Section 1: transfer=32, src_inner=0, dst_inner=32
    rp1 = result[1]
    assert rp1.src.memory.ptrs[0] == src_ptrs[0] + 32  # section_start(32) + 0
    assert rp1.dst.memory.ptrs[0] == dst_ptrs[0] + 64 + 32  # section_start(64) + 32

    # Section 2: transfer=64, src_inner=0, dst_inner=(3%2)*64=64
    rp2 = result[2]
    assert rp2.src.memory.bytes_per_region == 64
    assert rp2.src.memory.ptrs[0] == src_ptrs[0] + 64  # section_start(32+32) + 0
    assert rp2.dst.memory.ptrs[0] == dst_ptrs[0] + 128 + 64  # section_start(64+64) + 64


def test_conv_state_mismatch_mapper_mamba2_layout():
    """ConvStateMismatchMapper with Mamba2 layout: [x(d_inner) | B(ng*ds) | C(ng*ds)].

    d_inner=256, ng*ds=64, TP=2 vs TP=4.
    self (TP=2): [128, 32, 32]
    peer (TP=4): [64, 16, 16]
    """
    self_sec = [128, 32, 32]
    peer_sec = [64, 16, 16]

    transfer_layers = 2

    src_ptrs = [1000, 2000]
    dst_ptrs = [5000, 6000]

    src_group = MemRegionGroup(ptrs=src_ptrs, bytes_per_region=192)
    dst_group = MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=96)
    src_spec = SpecRegion(memory=src_group)
    dst_spec = SpecRegion(memory=dst_group)

    mapper = ConvStateMismatchMapper(
        transfer_layers=transfer_layers,
        src_layer_off=0,
        dst_layer_off=0,
        self_section_bytes=self_sec,
        peer_section_bytes=peer_sec,
        self_tp_per_dp=2,
        peer_tp_per_dp=4,
        self_tp_rank=0,
        peer_tp_rank=0,  # (0 % 2) * transfer_bytes = 0
    )

    result = mapper.map(src_spec, dst_spec)
    assert len(result) == 3

    # peer_tp_rank=0 -> src_inner_off = (0%2)*transfer_bytes = 0 for all sections
    rp0 = result[0]
    assert rp0.src.memory.bytes_per_region == 64  # min(128, 64)
    assert rp0.src.memory.ptrs[0] == 1000 + 0  # section_start(0) + 0

    rp1 = result[1]
    assert rp1.src.memory.bytes_per_region == 16  # min(32, 16)
    assert rp1.src.memory.ptrs[0] == 1000 + 128  # section_start(128) + 0

    rp2 = result[2]
    assert rp2.src.memory.bytes_per_region == 16  # min(32, 16)
    assert rp2.src.memory.ptrs[0] == 1000 + 160  # section_start(128+32) + 0


def test_compute_tp_offsets():
    """Test _compute_tp_offsets helper function."""
    # Same TP -> no offset
    assert _compute_tp_offsets(2, 2, 0, 0, 64) == (0, 0)

    # self < peer -> src gets offset
    assert _compute_tp_offsets(2, 4, 0, 1, 32) == (32, 0)  # (1%2)*32
    assert _compute_tp_offsets(2, 4, 0, 0, 32) == (0, 0)  # (0%2)*32
    assert _compute_tp_offsets(2, 4, 0, 3, 32) == (32, 0)  # (3%2)*32

    # self > peer -> dst gets offset
    assert _compute_tp_offsets(4, 2, 1, 0, 32) == (0, 32)  # (1%2)*32
    assert _compute_tp_offsets(4, 2, 0, 0, 32) == (0, 0)  # (0%2)*32
    assert _compute_tp_offsets(4, 2, 3, 0, 32) == (0, 32)  # (3%2)*32
