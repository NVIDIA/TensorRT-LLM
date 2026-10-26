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

import numpy as np
import pytest

__extra_import_path__ = [".."]
from disaggregated.pages_test_utils import buf, group, head_major, pages, pool

from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.page_mapping import plan_transfers
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import RankTopology

pytestmark = pytest.mark.cpu_only


def make_rankinfo(
    tp_size=2,
    tp_rank=0,
    dp_size=1,
    dp_rank=0,
    pp_size=1,
    pp_rank=0,
    cp_size=1,
    cp_rank=0,
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
        self_endpoint="",
        transfer_engine_info=b"",
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


def _kv_pages(layer_offsets, bytes_per_layer, kv_heads, ri):
    """One attention layer group as rank ``ri`` publishes it (4 tokens per page).

    Global layer ``i`` sits at ``layer_offsets[i]`` (K then V).
    """
    half = bytes_per_layer // 2
    segments = head_major(half, kv_heads)
    buffers = []
    for gid, offset in enumerate(layer_offsets):
        if offset is None:
            continue
        buffers.append(buf(0, gid, "key", offset, segments))
        buffers.append(buf(0, gid, "value", offset + half, segments))
    slot = max(o for o in layer_offsets if o is not None) + bytes_per_layer
    topology = RankTopology(
        tp_size=ri.tp_size,
        tp_rank=ri.tp_rank,
        pp_size=ri.pp_size,
        pp_rank=ri.pp_rank,
        cp_size=ri.cp_size,
        cp_rank=ri.cp_rank,
        dp_size=ri.dp_size,
        dp_rank=ri.dp_rank,
    )
    return pages([pool(0, slot, num_slots=4)], [group("attention", buffers)], topology=topology)


def _plan(self_ri, peer_ri, self_offsets, peer_offsets, self_bpl, peer_bpl, heads=(2, 2)):
    """The run plan between the two ranks' ``_kv_pages``.

    ``heads`` are each side's KV heads per rank, as ``(local, peer)``.
    """
    self_ri, peer_ri = (
        dataclasses.replace(ri, pages=_kv_pages(offsets, bpl, h, ri))
        for ri, offsets, bpl, h in zip(
            (self_ri, peer_ri), (self_offsets, peer_offsets), (self_bpl, peer_bpl), heads
        )
    )
    groups = plan_transfers(self_ri, peer_ri, self_ri.pages, peer_ri.pages)
    assert len(groups) == 1
    return groups[0].plan


def test_whole_copy_identity_degenerate():
    """Full contiguous overlap degrades to a whole-region pass-through copy."""
    src_group = MemRegionGroup(ptrs=np.array([100, 200], dtype=np.int64), bytes_per_region=32)
    dst_group = MemRegionGroup(ptrs=np.array([300, 400], dtype=np.int64), bytes_per_region=32)
    src_spec = SpecRegion(memory=src_group, spec="a")
    dst_spec = SpecRegion(memory=dst_group, spec="b")
    ri = make_rankinfo()
    plan = _plan(ri, ri, [0, 16], [0, 16], 16, 16)
    # Runs of equal length come back as one fragment array.
    (result,) = plan.map(src_spec, dst_spec)
    assert isinstance(result, SpecRegionPair)
    np.testing.assert_array_equal(result.src.memory.ptrs, [100, 200])
    np.testing.assert_array_equal(result.dst.memory.ptrs, [300, 400])
    assert result.src.memory.bytes_per_region == 32
    assert result.dst.memory.bytes_per_region == 32
    assert result.src.spec == "a" and result.dst.spec == "b"


def test_whole_copy_partial_layers():
    """Selecting a contiguous layer subset yields one shifted fragment."""
    self_ri = make_rankinfo()
    # K/V x 2 kv heads x 4 tokens per block x 2 dims per head x 1 byte per element.
    bytes_per_layer = 2 * 2 * 4 * 2 * 1
    src_group = MemRegionGroup(ptrs=np.array([10, 20], dtype=np.int64), bytes_per_region=1)
    dst_group = MemRegionGroup(ptrs=np.array([30, 40], dtype=np.int64), bytes_per_region=1)
    src_spec = SpecRegion(memory=src_group, spec="srcspec")
    dst_spec = SpecRegion(memory=dst_group, spec="dstspec")
    # Self holds global layers 1..2 at slot positions 1..2; the peer stores them at
    # positions 2..3. Distinct src/dst layer offsets make the test fail if the plan
    # ever applies one side's offset to the other.
    plan = _plan(
        self_ri,
        self_ri,
        [None, bytes_per_layer, 2 * bytes_per_layer],
        [None, 2 * bytes_per_layer, 3 * bytes_per_layer],
        bytes_per_layer,
        bytes_per_layer,
    )
    # Runs of equal length come back as one fragment array.
    (result,) = plan.map(src_spec, dst_spec)
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


def test_head_mismatch_slices_heads():
    # Self holds 2 of 4 KV heads at TP=2; the peer holds 1 KV head at TP=4. Peer
    # tp_rank=1 needs head 1, inside self rank 0's pair, so the source-side head offset is
    # *nonzero* and an incorrect head-offset computation cannot pass silently.
    self_ri = make_rankinfo(tp_size=2, tp_rank=0)
    peer_ri = make_rankinfo(tp_size=4, tp_rank=1)
    buffers_per_layer = 2  # K and V
    # buffer bytes = heads * tokens_per_block * dims_per_head * element_bytes
    self_buffer_bytes = 2 * 4 * 2 * 1
    peer_buffer_bytes = 1 * 4 * 2 * 1
    self_bytes_per_layer = buffers_per_layer * self_buffer_bytes
    peer_bytes_per_layer = buffers_per_layer * peer_buffer_bytes
    bytes_per_head = self_buffer_bytes // 2  # equals peer side
    src_base, dst_base = 111, 222
    src_group = MemRegionGroup(ptrs=np.array([src_base], dtype=np.int64), bytes_per_region=32)
    dst_group = MemRegionGroup(ptrs=np.array([dst_base], dtype=np.int64), bytes_per_region=32)
    src_spec = SpecRegion(memory=src_group, spec="srcspec")
    dst_spec = SpecRegion(memory=dst_group, spec="dstspec")
    # Global layer 0 sits at the start of the local slot and at layer position 1 on the peer.
    peer_layer_off = peer_bytes_per_layer
    plan = _plan(
        self_ri,
        peer_ri,
        [0],
        [peer_layer_off],
        self_bytes_per_layer,
        peer_bytes_per_layer,
        heads=(2, 1),
    )
    # Runs of equal length come back as one fragment array.
    (result,) = plan.map(src_spec, dst_spec)
    assert isinstance(result, SpecRegionPair)
    assert isinstance(result.src.memory.ptrs, np.ndarray)
    assert isinstance(result.dst.memory.ptrs, np.ndarray)
    # Head 1 is one head into self's span [0, 2); the peer's span is just [1, 2).
    src_head_off = 1 * bytes_per_head
    # One fragment per (layer, K/V buffer), buffers back-to-back within a layer.
    expected_src_ptrs = [
        src_base + src_head_off + buf * self_buffer_bytes for buf in range(buffers_per_layer)
    ]
    expected_dst_ptrs = [
        dst_base + peer_layer_off + buf * peer_buffer_bytes for buf in range(buffers_per_layer)
    ]
    np.testing.assert_array_equal(result.src.memory.ptrs, expected_src_ptrs)
    np.testing.assert_array_equal(result.dst.memory.ptrs, expected_dst_ptrs)
    # Each fragment copies the heads both sides share: the peer's one head.
    assert result.src.memory.bytes_per_region == bytes_per_head
    assert result.dst.memory.bytes_per_region == bytes_per_head
