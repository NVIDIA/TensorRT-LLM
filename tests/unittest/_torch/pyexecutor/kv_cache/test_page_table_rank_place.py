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
"""A rank's place and the parallel dimensions recurrent state splits over, as a page table reads
them."""

import collections
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.disaggregation.base.role_layout import (
    ByHead,
    RankTopology,
    Replicated,
    RingSlots,
    Sectioned,
    check_pieces,
    grid_place,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_table import rank_topology, state_split

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("attention_dp", [False, True])
def test_rank_topology_reads_the_rank_s_place_from_its_mapping(attention_dp):
    mapping = SimpleNamespace(
        tp_size=4,
        tp_rank=3,
        pp_size=2,
        pp_rank=1,
        cp_size=2,
        cp_rank=1,
        dp_size=1,
        enable_attention_dp=attention_dp,
    )

    topology = rank_topology(mapping)

    assert (topology.tp_size, topology.tp_rank, topology.pp_size, topology.pp_rank) == (4, 3, 2, 1)
    assert (topology.cp_size, topology.cp_rank, topology.attention_dp) == (2, 1, attention_dp)
    # Under attention data parallelism each TP rank holds its buffers whole.
    assert grid_place(topology, ("tp",)) == ((1, 0) if attention_dp else (4, 3))
    assert rank_topology(None) == RankTopology()


@pytest.mark.parametrize("attention_dp", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_recurrent_state_rank_in_the_shard_grid(tp_size, cp_size, attention_dp):
    for tp_rank in range(tp_size):
        for cp_rank in range(cp_size):
            mapping = SimpleNamespace(
                tp_size=tp_size,
                tp_rank=tp_rank,
                cp_size=cp_size,
                cp_rank=cp_rank,
                enable_attention_dp=attention_dp,
            )
            manager = SimpleNamespace(mapping=mapping, _has_cp_helix=cp_size > 1)
            topology = rank_topology(mapping)
            split = state_split(manager)
            if attention_dp:
                expected = (cp_size, cp_rank) if tp_size == 1 else (1, 0)
            else:
                expected = (tp_size * cp_size, tp_rank * cp_size + cp_rank)
            assert grid_place(topology, split) == expected, topology


def _site(size, *, kv_heads=(0, None), state=False, split=("tp",), ranks=1, tokens=32):
    """The facts of one buffer that a layout in a manager's spec resolves against."""
    return SimpleNamespace(
        size=size, kv_heads=kv_heads, state=state, split=split, ranks=ranks, tokens=tokens
    )


_RANK_COUNTS = (1, 2, 4, 8)


def _kv_site(heads, head_bytes, ranks, **kwargs):
    """An attention buffer of ``heads`` KV heads of ``head_bytes`` bytes over ``ranks`` ranks."""
    local = max(1, heads // ranks)
    return _site(local * head_bytes, kv_heads=(local, heads), ranks=ranks, **kwargs)


# Each case: ranks -> (the layout a manager's spec states, one rank's buffer's facts).
_TILING_CASES = {
    "kv_by_head": lambda n: (ByHead(), _kv_site(8, 64, n)),
    "kv_token_major": lambda n: (ByHead(token_major=True), _kv_site(8, 64, n, tokens=4)),
    "kv_heads_held_by_several_ranks": lambda n: (ByHead(), _kv_site(2, 64, n)),
    "mla_latent": lambda n: (ByHead(total=1), _site(576, kv_heads=(1, None), ranks=n)),
    "recurrent_state": lambda n: (ByHead(head_bytes=32), _site(8 // n * 32, state=True, ranks=n)),
    "sections": lambda n: (Sectioned((64 // n, 32 // n, 32 // n)), _site(128 // n, ranks=n)),
    "replicated": lambda n: (Replicated(), _site(96, ranks=n)),
    "ring": lambda n: (RingSlots(4), _site(96, ranks=n)),
}


@pytest.mark.parametrize("case", sorted(_TILING_CASES))
def test_ranks_of_every_topology_together_hold_one_space_whole(case):
    spaces = set()
    for ranks in _RANK_COUNTS:
        layout, site = _TILING_CASES[case](ranks)
        resolved = layout.resolve(site)
        resolved.check(site.size)
        held = collections.Counter()
        for rank in range(ranks):
            check_pieces(resolved, site.size, ranks, rank)
            for piece in resolved.pieces(site.size, ranks, rank):
                held.update(range(piece.at, piece.at + piece.size))
        # Every byte of the space from 0 up is held, each by as many ranks as every other.
        assert sorted(held) == list(range(len(held))), (case, ranks)
        assert len(set(held.values())) == 1, (case, ranks)
        spaces.add(resolved.space(site.size, ranks))
    # Peers of any two of these rank counts lay out the same space, so they correspond.
    assert len(spaces) == 1, (case, spaces)
