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
"""Hardware-agnostic tests for Helix + Eagle3 KV ownership and positions.

Guards that each verify group is owned by exactly one CP rank, that global
positions and inactive flags are consistent, and that reserve/rewind FIFO
ownership matches under overlap scheduling.
"""

import pytest


def owns_decode_index(decode_index: int, tpb: int, cp_size: int, cp_rank: int) -> bool:
    return (decode_index // tpb) % cp_size == cp_rank


@pytest.mark.parametrize("tpb", [1, 4, 32])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("g", [0, 1, 5, 31, 32, 33, 64, 100, 257])
def test_verify_group_owned_by_exactly_one_rank(tpb, cp_size, g):
    # Exactly one rank owns decode-index g.
    owners = [cp_rank for cp_rank in range(cp_size) if owns_decode_index(g, tpb, cp_size, cp_rank)]
    assert owners == [(g // tpb) % cp_size]


def verify_token_params(
    total_input_len: int, g: int, num_draft: int, tpb: int, cp_size: int, cp_rank: int
) -> tuple[list[int], list[bool], int]:
    first_decode_index = g
    first_pos = total_input_len + first_decode_index
    global_positions = list(range(first_pos, first_pos + 1 + num_draft))
    owns_group = owns_decode_index(g, tpb, cp_size, cp_rank)
    inactive_flags = [not owns_group] * (1 + num_draft)
    num_active = (1 + num_draft) if owns_group else 0
    return global_positions, inactive_flags, num_active


@pytest.mark.parametrize("tpb", [4, 32])
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("num_draft", [0, 1, 3])
@pytest.mark.parametrize("g", [0, 1, 5, 32])
def test_verify_token_params_consistency(tpb, cp_size, num_draft, g):
    total_input_len = 100
    group_size = 1 + num_draft
    # Expected owner from anchor decode-index g.
    expected_owner = (g // tpb) % cp_size
    active_ranks = []
    for cp_rank in range(cp_size):
        positions, inactive, num_active = verify_token_params(
            total_input_len, g, num_draft, tpb, cp_size, cp_rank
        )
        # Global positions start at total_input_len + g.
        assert positions == list(range(positions[0], positions[0] + group_size))
        assert positions[0] == total_input_len + g
        # One inactive flag per request, not per token.
        assert len(set(inactive)) == 1
        # num_active counts owned query tokens.
        assert num_active == sum(1 for f in inactive if not f)
        if cp_rank == expected_owner:
            assert all(not f for f in inactive)
            assert num_active == group_size
            active_ranks.append(cp_rank)
        else:
            assert all(f for f in inactive)
            assert num_active == 0
    # No mixed ownership across ranks.
    assert active_ranks == [expected_owner]


@pytest.mark.parametrize(
    "tpb,cp_size,g,num_draft",
    [
        # Per-token indices would straddle a block boundary; group stays on one rank.
        (4, 2, 3, 3),
        (32, 2, 30, 3),
        (4, 4, 3, 3),
        # Groups within one block.
        (4, 2, 5, 3),
        (32, 2, 1, 3),
    ],
)
def test_verify_token_params_no_mixed_ownership(tpb, cp_size, g, num_draft):
    group_size = 1 + num_draft
    expected_owner = (g // tpb) % cp_size
    per_rank_active = []
    for cp_rank in range(cp_size):
        _, inactive, num_active = verify_token_params(100, g, num_draft, tpb, cp_size, cp_rank)
        per_rank_active.append(num_active)
        # Uniform inactive flag within the request.
        assert len(set(inactive)) == 1
        if cp_rank == expected_owner:
            assert num_active == group_size
        else:
            assert num_active == 0
    # Whole group on one rank, even across block boundaries.
    ranks_with_writes = sum(1 for a in per_rank_active if a > 0)
    assert ranks_with_writes == 1
    assert per_rank_active[expected_owner] == group_size


# ---------------------------------------------------------------------------
# Overlap scheduler: deterministic verify-group ownership + reserve/rewind FIFO
# ---------------------------------------------------------------------------


owns_decode_group = owns_decode_index


@pytest.mark.parametrize("tpb", [1, 4, 32])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("group_index", [0, 1, 5, 31, 32, 33, 64, 100, 257])
def test_decode_group_owned_by_exactly_one_rank(tpb, cp_size, group_index):
    owners = [
        cp_rank
        for cp_rank in range(cp_size)
        if owns_decode_group(group_index, tpb, cp_size, cp_rank)
    ]
    assert owners == [(group_index // tpb) % cp_size]


@pytest.mark.parametrize("tpb", [1, 4, 32])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_decode_group_ownership_is_balanced(tpb, cp_size):
    # Balanced ownership over cp_size * tpb groups.
    counts = [0] * cp_size
    for group_index in range(cp_size * tpb):
        counts[(group_index // tpb) % cp_size] += 1
    assert counts == [tpb] * cp_size


class _FifoRankState:
    """Per-rank reserve and rewind state."""

    def __init__(self, tpb: int, cp_size: int, cp_rank: int) -> None:
        self.tpb = tpb
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.group_index: int = 0
        self.pending: list[bool] = []
        self.reserve_log: list[bool] = []
        self.rewind_log: list[bool] = []

    def reserve(self) -> bool:
        owns = owns_decode_group(self.group_index, self.tpb, self.cp_size, self.cp_rank)
        self.group_index += 1
        self.pending.append(owns)
        self.reserve_log.append(owns)
        return owns

    def rewind(self) -> bool:
        owns = self.pending.pop(0)
        self.rewind_log.append(owns)
        return owns


@pytest.mark.parametrize("tpb", [1, 4])
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("pipeline_depth", [0, 1])
@pytest.mark.parametrize("num_iters", [1, 2, 5, 40])
def test_reserve_rewind_fifo_consistency(tpb, cp_size, pipeline_depth, num_iters):
    """Reserve and rewind must agree on ownership for every group."""
    for cp_rank in range(cp_size):
        state = _FifoRankState(tpb, cp_size, cp_rank)
        # Prime overlap pipeline before first rewind.
        for _ in range(pipeline_depth):
            state.reserve()
        for _ in range(num_iters):
            state.reserve()
            state.rewind()
        # Drain in-flight reserves.
        while state.pending:
            state.rewind()

        # Rewind log must match reserve log in FIFO order.
        assert state.rewind_log == state.reserve_log
        # Sequence matches deterministic group-ownership formula.
        expected = [
            owns_decode_group(i, tpb, cp_size, cp_rank) for i in range(len(state.reserve_log))
        ]
        assert state.reserve_log == expected


@pytest.mark.parametrize("tpb", [1, 4])
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("num_iters", [8, 40])
def test_each_group_owned_by_exactly_one_rank_across_ranks(tpb, cp_size, num_iters):
    # Each group index is owned by exactly one rank.
    for group_index in range(num_iters):
        owners = [r for r in range(cp_size) if owns_decode_group(group_index, tpb, cp_size, r)]
        assert len(owners) == 1
