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

from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import (
    HNDHeadMismatchMapper,
    IntactMapper,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo


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
        layer_num_per_pp=[1],
        sender_endpoints=[],
        server_endpoint="",
        self_endpoint="",
        transfer_engine_info=b"",
        attention=AttentionInfo(
            kv_heads_per_rank=kv_heads_per_rank,
            tokens_per_block=tokens_per_block,
            dims_per_head=dims_per_head,
            element_bytes=element_bytes,
            enable_attention_dp=False,
            is_mla=is_mla,
        ),
        aux_meta=None,
    )


def test_mem_region_group():
    ptrs = np.array([11, 22, 33], dtype=np.int64)
    bytes_per_region = 16
    region = MemRegionGroup(ptrs=ptrs, bytes_per_region=bytes_per_region)
    np.testing.assert_array_equal(region.ptrs, ptrs)
    assert region.bytes_per_region == bytes_per_region


def test_spec_region_and_spec_region_pair():
    group_src = MemRegionGroup(ptrs=np.array([101, 202], dtype=np.int64), bytes_per_region=8)
    group_dst = MemRegionGroup(ptrs=np.array([303, 404], dtype=np.int64), bytes_per_region=8)
    spec_src = SpecRegion(memory=group_src, spec="spec_src")
    spec_dst = SpecRegion(memory=group_dst, spec="spec_dst")
    assert isinstance(spec_src, SpecRegion)
    assert isinstance(spec_dst, SpecRegion)
    pair = SpecRegionPair(src=spec_src, dst=spec_dst)
    assert isinstance(pair, SpecRegionPair)
    np.testing.assert_array_equal(pair.src.memory.ptrs, [101, 202])
    np.testing.assert_array_equal(pair.dst.memory.ptrs, [303, 404])
    assert pair.src.spec == "spec_src"
    assert pair.dst.spec == "spec_dst"


def test_intact_mapper_identity_degenerate():
    """Full contiguous overlap degrades to a whole-region pass-through copy."""
    src_group = MemRegionGroup(ptrs=np.array([100, 200], dtype=np.int64), bytes_per_region=32)
    dst_group = MemRegionGroup(ptrs=np.array([300, 400], dtype=np.int64), bytes_per_region=32)
    src_spec = SpecRegion(memory=src_group, spec="a")
    dst_spec = SpecRegion(memory=dst_group, spec="b")
    mapper = IntactMapper([0, 16], [0, 16], 16, 16)
    result = mapper.map(src_spec, dst_spec)
    assert isinstance(result, SpecRegionPair)
    np.testing.assert_array_equal(result.src.memory.ptrs, [100, 200])
    np.testing.assert_array_equal(result.dst.memory.ptrs, [300, 400])
    assert result.src.memory.bytes_per_region == 32
    assert result.dst.memory.bytes_per_region == 32


def test_intact_mapper_partial_layers():
    """Selecting a contiguous layer subset yields one shifted fragment."""
    self_ri = make_rankinfo(kv_heads_per_rank=2)
    # bytes_per_layer = kv_factor * kv_heads * tokens_per_block * dims_per_head * element_bytes
    bytes_per_layer = (
        self_ri.attention.kv_factor
        * self_ri.attention.kv_heads_per_rank
        * self_ri.attention.tokens_per_block
        * self_ri.attention.dims_per_head
        * self_ri.attention.element_bytes
    )
    src_group = MemRegionGroup(ptrs=np.array([10, 20], dtype=np.int64), bytes_per_region=1)
    dst_group = MemRegionGroup(ptrs=np.array([30, 40], dtype=np.int64), bytes_per_region=1)
    src_spec = SpecRegion(memory=src_group, spec="srcspec")
    dst_spec = SpecRegion(memory=dst_group, spec="dstspec")
    # Self selects layers 1..2 of its slot; the peer stores the same class at
    # layers 2..3. Distinct src/dst layer offsets make the test fail if the
    # mapper ever applies one side's offset to the other.
    src_offsets = [bytes_per_layer, 2 * bytes_per_layer]
    dst_offsets = [2 * bytes_per_layer, 3 * bytes_per_layer]
    mapper = IntactMapper(src_offsets, dst_offsets, bytes_per_layer, bytes_per_layer)
    result = mapper.map(src_spec, dst_spec)
    # Both sides are internally contiguous, so the two layers merge into a
    # single fragment shifted by each side's own first-layer offset.
    expected_bytes = 2 * bytes_per_layer
    np.testing.assert_array_equal(
        result.src.memory.ptrs, [10 + bytes_per_layer, 20 + bytes_per_layer]
    )
    np.testing.assert_array_equal(
        result.dst.memory.ptrs, [30 + 2 * bytes_per_layer, 40 + 2 * bytes_per_layer]
    )
    assert result.src.memory.bytes_per_region == expected_bytes
    assert result.dst.memory.bytes_per_region == expected_bytes


def test_head_mismatch_mapper():
    # Self holds 2 KV heads at TP=2; the peer holds 4 KV heads at TP=4. Using
    # peer tp_rank=1 (not 2) forces a *nonzero* source-side head offset, so an
    # incorrect head-offset computation cannot pass silently.
    self_ri = make_rankinfo(kv_heads_per_rank=2, tp_size=2, tp_rank=1)
    peer_ri = make_rankinfo(kv_heads_per_rank=4, tp_size=4, tp_rank=1)
    buffers_per_layer = 2  # K and V
    # buffer bytes = heads * tokens_per_block * dims_per_head * element_bytes
    self_buffer_bytes = 2 * 4 * 2 * 1
    peer_buffer_bytes = 4 * 4 * 2 * 1
    self_bytes_per_layer = buffers_per_layer * self_buffer_bytes  # kv_factor x K-buffer bytes
    peer_bytes_per_layer = buffers_per_layer * peer_buffer_bytes
    bytes_per_head = self_buffer_bytes // self_ri.attention.kv_heads_per_rank  # equals peer side
    src_base, dst_base = 111, 222
    src_group = MemRegionGroup(ptrs=np.array([src_base], dtype=np.int64), bytes_per_region=32)
    dst_group = MemRegionGroup(ptrs=np.array([dst_base], dtype=np.int64), bytes_per_region=32)
    src_spec = SpecRegion(memory=src_group, spec="srcspec")
    dst_spec = SpecRegion(memory=dst_group, spec="dstspec")
    peer_layer_off = peer_bytes_per_layer  # copy targets layer 1 on the peer side
    mapper = HNDHeadMismatchMapper(
        src_layer_offsets=[0],
        dst_layer_offsets=[peer_layer_off],
        self_ri=self_ri,
        peer_ri=peer_ri,
        self_bytes_per_layer=self_bytes_per_layer,
        peer_bytes_per_layer=peer_bytes_per_layer,
        self_buffers_per_layer=buffers_per_layer,
        peer_buffers_per_layer=buffers_per_layer,
    )
    result = mapper.map(src_spec, dst_spec)
    assert isinstance(result, SpecRegionPair)
    assert isinstance(result.src.memory.ptrs, np.ndarray)
    assert isinstance(result.dst.memory.ptrs, np.ndarray)
    # Source head offset for self rank 1 slicing into the peer's 4-head layout:
    #   (peer_tp_rank * self_kv_heads * self_tp) // peer_tp % self_kv_heads
    #   = (1 * 2 * 2) // 4 % 2 = 1 head -> 1 * bytes_per_head. The peer side is
    # the coarser layout, so its head offset is 0.
    src_head_off = 1 * bytes_per_head
    # One fragment per (layer, K/V buffer), buffers back-to-back within a layer.
    # Source layer offset is 0; the peer layer offset is nonzero.
    expected_src_ptrs = [
        src_base + src_head_off + buf * self_buffer_bytes for buf in range(buffers_per_layer)
    ]
    expected_dst_ptrs = [
        dst_base + peer_layer_off + buf * peer_buffer_bytes for buf in range(buffers_per_layer)
    ]
    np.testing.assert_array_equal(result.src.memory.ptrs, expected_src_ptrs)
    np.testing.assert_array_equal(result.dst.memory.ptrs, expected_dst_ptrs)
    # Each fragment copies min(self_heads, peer_heads) contiguous heads.
    cont_heads_bytes = (
        min(self_ri.attention.kv_heads_per_rank, peer_ri.attention.kv_heads_per_rank)
        * bytes_per_head
    )
    assert result.src.memory.bytes_per_region == cont_heads_bytes
    assert result.dst.memory.bytes_per_region == cont_heads_bytes


def test_rankinfo_kv_factor():
    ri1 = make_rankinfo(is_mla=False)
    ri2 = make_rankinfo(is_mla=True)
    assert ri1.attention.kv_factor == 2
    assert ri2.attention.kv_factor == 1
