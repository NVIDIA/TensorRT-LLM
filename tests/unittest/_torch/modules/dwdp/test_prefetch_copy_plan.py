# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""CPU tests for the DWDP prefetch copy plan.

``_build_peer_descriptor_groups`` (one coalesced group per peer, consumed by the SM gather) and
``_build_batched_prefetch_copy_plan`` (the same bytes re-sliced and interleaved for
``cudaMemcpyBatchAsync``) must move identical bytes, so each plan is reduced to a canonical
interval set and compared across slice geometries, ranks and slice sizes.
"""

from typing import Dict, List, Tuple

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp import weight_manager as wm

pytestmark = pytest.mark.cpu_only

DWDP_SIZE = 16
EXPERTS_PER_PEER = 24
TOTAL_EXPERTS = DWDP_SIZE * EXPERTS_PER_PEER
LAYER = 0
WEIGHT_NAMES = ["w1_weight", "w2_weight", "w3_weight"]
EXPERT_ELEMS = 1024
DTYPE = torch.bfloat16
ELEM_BYTES = DTYPE.itemsize
LAYER_BYTES = len(WEIGHT_NAMES) * TOTAL_EXPERTS * EXPERT_ELEMS * ELEM_BYTES
PEER_WEIGHT_BYTES = EXPERTS_PER_PEER * EXPERT_ELEMS * ELEM_BYTES

PRODUCTION_BOUNDS = [0, TOTAL_EXPERTS]
FRAGMENTED_BOUNDS = [0, 10, 100, 101, 200, 373, TOTAL_EXPERTS]  # straddles peer boundaries
BOUNDS = pytest.mark.parametrize(
    "bounds", [PRODUCTION_BOUNDS, FRAGMENTED_BOUNDS], ids=["production", "fragmented"]
)
SLICE_SETTINGS = pytest.mark.parametrize(
    "slice_bytes", [wm.MEMCPY_BATCH_SLICE_BYTES, 4096], ids=["2MiB", "4KiB"]
)


class _FakeWeightBuffer:
    """Stand-in exposing only the method the plan builder calls."""

    def __init__(self, slices):
        self._slices = slices

    def get_remote_slices(self, layer_idx, name):
        return self._slices[(layer_idx, name)]


def _make_manager(slice_bounds: List[int], dwdp_rank: int = 0):
    """Return (manager, backings, src_ptr_to_peer); keep backings alive while a plan is in use."""
    assert slice_bounds[0] == 0 and slice_bounds[-1] == TOTAL_EXPERTS

    peer_views: Dict[Tuple[int, int, str], torch.Tensor] = {}
    src_ptr_to_peer: Dict[int, int] = {}
    for peer in range(DWDP_SIZE):
        for name in WEIGHT_NAMES:
            t = torch.empty((EXPERTS_PER_PEER, EXPERT_ELEMS), dtype=DTYPE)
            peer_views[(peer, LAYER, name)] = t
            src_ptr_to_peer[t.data_ptr()] = peer

    slices, backings = {}, []
    for name in WEIGHT_NAMES:
        backing = torch.empty((TOTAL_EXPERTS, EXPERT_ELEMS), dtype=DTYPE)
        backings.append(backing)
        slices[(LAYER, name)] = [
            (backing[a:b], a, b) for a, b in zip(slice_bounds, slice_bounds[1:])
        ]

    mgr = object.__new__(wm.DWDPWeightManager)
    mgr._weight_names = WEIGHT_NAMES
    mgr._weight_buffer = _FakeWeightBuffer(slices)
    mgr._peer_views = peer_views
    mgr._peer_ranges = [
        (p * EXPERTS_PER_PEER, (p + 1) * EXPERTS_PER_PEER) for p in range(DWDP_SIZE)
    ]
    mgr._dwdp_rank = dwdp_rank
    mgr._dwdp_size = DWDP_SIZE
    return mgr, backings, src_ptr_to_peer


def _groups(mgr):
    return wm.DWDPWeightManager._build_peer_descriptor_groups(mgr, LAYER)


def _plan(mgr, slice_bytes: int, monkeypatch) -> Tuple[List[int], List[int], List[int]]:
    monkeypatch.setattr(wm, "MEMCPY_BATCH_SLICE_BYTES", slice_bytes)
    return wm.DWDPWeightManager._build_batched_prefetch_copy_plan(mgr, LAYER)


def _flat_groups(groups) -> Tuple[List[int], List[int], List[int]]:
    triples = [t for g in groups for t in g]
    return [d for d, _, _ in triples], [s for _, s, _ in triples], [z for _, _, z in triples]


def _peer_of(src_ptr: int, src_ptr_to_peer: Dict[int, int]) -> int:
    return src_ptr_to_peer[max(b for b in src_ptr_to_peer if b <= src_ptr)]


def _canonical(flat) -> List[Tuple[int, int, int]]:
    """Minimal sorted (dst, src, size) set; independent re-derivation, keep separate from wm."""
    out: List[Tuple[int, int, int]] = []
    for dst, src, size in sorted(zip(*flat)):
        if out and out[-1][0] + out[-1][2] == dst and out[-1][1] + out[-1][2] == src:
            out[-1] = (out[-1][0], out[-1][1], out[-1][2] + size)
        else:
            out.append((dst, src, size))
    return out


def _overlap_count(flat) -> int:
    dst_ptrs, _src_ptrs, sizes = flat
    intervals = sorted(zip(dst_ptrs, sizes))
    return sum(1 for (d0, n0), (d1, _n1) in zip(intervals, intervals[1:]) if d0 + n0 > d1)


@BOUNDS
def test_groups_partition_layer_by_peer(bounds):
    """One ascending group per peer, one descriptor per weight, covering the layer exactly once."""
    mgr, _keep, ptrs = _make_manager(bounds)
    groups = _groups(mgr)
    peers = [{_peer_of(src, ptrs) for _d, src, _z in g} for g in groups]
    assert [next(iter(p)) for p in peers] == list(range(DWDP_SIZE))
    assert all(len(p) == 1 for p in peers)
    assert [len(g) for g in groups] == [len(WEIGHT_NAMES)] * DWDP_SIZE
    flat = _flat_groups(groups)
    assert set(flat[2]) == {PEER_WEIGHT_BYTES}
    assert sum(flat[2]) == LAYER_BYTES
    assert _overlap_count(flat) == 0


def test_groups_do_not_depend_on_the_rank():
    mgr, _keep, _ptrs = _make_manager(FRAGMENTED_BOUNDS)
    reference = _flat_groups(_groups(mgr))
    for rank in range(1, DWDP_SIZE):
        mgr._dwdp_rank = rank
        assert _flat_groups(_groups(mgr)) == reference, f"rank {rank} changed the plan"


@BOUNDS
@SLICE_SETTINGS
def test_batch_moves_the_same_bytes_without_overlap(bounds, slice_bytes, monkeypatch):
    mgr, _keep, _ptrs = _make_manager(bounds)
    plan = _plan(mgr, slice_bytes, monkeypatch)
    assert len(plan[0]) == len(plan[1]) == len(plan[2])
    assert max(plan[2]) <= slice_bytes
    assert sum(plan[2]) == LAYER_BYTES
    assert _overlap_count(plan) == 0
    assert _canonical(plan) == _canonical(_flat_groups(_groups(mgr)))


def test_batch_interleaves_peers_round_robin(monkeypatch):
    mgr, _keep, ptrs = _make_manager(PRODUCTION_BOUNDS)
    _dst, src, _sizes = _plan(mgr, 4096, monkeypatch)
    assert len(src) > DWDP_SIZE
    first_round = [_peer_of(s, ptrs) for s in src[:DWDP_SIZE]]
    assert sorted(first_round) == list(range(DWDP_SIZE)), first_round
