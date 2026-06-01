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
"""Hardware-agnostic tests for the Helix + Eagle3 (speculative decode) KV
ownership and global-position bookkeeping.

These guard the core invariants used by the Helix speculative-decode integration:
- a verify group's anchor decode-index is owned by exactly one CP rank
  (block-cyclic by anchor decode-index).
- the global positions and ownership computed for a speculative verify forward are
  consistent. Ownership is per request (per verify group): the whole group of
  query tokens (re-fed golden token plus drafts) is assigned to the CP rank that
  owns the anchor decode-index `g`. Golden and draft tokens always land on the
  same rank, so a verify block crossing a tokens_per_block boundary is never split
  across ranks.

The helpers below mirror the production formulas in
`tensorrt_llm/_torch/pyexecutor/resource_manager.py`
(`_helix_owns_decode_group`, `_helix_prepare_generation_kv`,
`_helix_rewind_generation_kv`) and `model_engine._helix_verify_token_params`.

Under the overlap scheduler the accepted-token count is only known on-device, so
ownership is anchored on a deterministic per-request verify-group counter (which
advances by exactly one per reserved group, independent of the accepted count)
rather than on the accept-dependent global decode length. The reserve-time
decision is stashed in a FIFO so the (one-iteration-later) rewind consumes exactly
the decision reserve made. The overlap tests below guard that reserve and rewind
never disagree even when a rewind of an earlier group runs between them.
"""

import pytest


def owns_decode_index(decode_index, tpb, cp_size, cp_rank):
    return (decode_index // tpb) % cp_size == cp_rank


@pytest.mark.parametrize("tpb", [1, 4, 32])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("g", [0, 1, 5, 31, 32, 33, 64, 100, 257])
def test_verify_group_owned_by_exactly_one_rank(tpb, cp_size, g):
    # A verify group anchored at decode-index g is owned by exactly one rank.
    owners = [cp_rank for cp_rank in range(cp_size) if owns_decode_index(g, tpb, cp_size, cp_rank)]
    assert owners == [(g // tpb) % cp_size]


def verify_token_params(total_input_len, g, num_draft, tpb, cp_size, cp_rank):
    """Mirror of model_engine._helix_verify_token_params (per-request ownership).

    The whole verify group (the re-fed golden token at decode-index `g` plus the
    drafts at `[g + 1, g + num_draft]`) is owned by the single rank that owns the
    anchor decode-index `g`. Every query token therefore shares one inactive flag.
    """
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
    # The whole group is owned by exactly one rank: the anchor owner of g.
    expected_owner = (g // tpb) % cp_size
    active_ranks = []
    for cp_rank in range(cp_size):
        positions, inactive, num_active = verify_token_params(
            total_input_len, g, num_draft, tpb, cp_size, cp_rank
        )
        # Positions are global and contiguous, starting at total_input_len + g.
        assert positions == list(range(positions[0], positions[0] + group_size))
        assert positions[0] == total_input_len + g
        # Flags are uniform within the request (per-request, not per-token).
        assert len(set(inactive)) == 1
        # num_active equals the number of non-inactive (owned) tokens.
        assert num_active == sum(1 for f in inactive if not f)
        if cp_rank == expected_owner:
            assert all(not f for f in inactive)
            assert num_active == group_size
            active_ranks.append(cp_rank)
        else:
            assert all(f for f in inactive)
            assert num_active == 0
    # Exactly one rank is active for the whole group (no mixed ownership).
    assert active_ranks == [expected_owner]


@pytest.mark.parametrize(
    "tpb,cp_size,g,num_draft",
    [
        # g values where the *per-token* decode-indices [g, g+num_draft] would
        # straddle a tokens_per_block boundary. With per-request ownership the
        # whole group is still assigned to the single anchor owner of g, so there
        # is never a split across ranks.
        (4, 2, 3, 3),  # per-token drafts 3,4,5 would split 0,1,1
        (32, 2, 30, 3),  # per-token would split 0,0,1
        (4, 4, 3, 3),  # per-token would split 0,1,1
        # Groups that already stay within one block.
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
        # Flags uniform within the request.
        assert len(set(inactive)) == 1
        if cp_rank == expected_owner:
            assert num_active == group_size
        else:
            assert num_active == 0
    # The whole group (golden + drafts) is owned by exactly one rank, regardless
    # of any tokens_per_block boundary the per-token indices would have crossed.
    ranks_with_writes = sum(1 for a in per_rank_active if a > 0)
    assert ranks_with_writes == 1
    assert per_rank_active[expected_owner] == group_size


# ---------------------------------------------------------------------------
# Overlap scheduler: deterministic verify-group ownership + reserve/rewind FIFO
# ---------------------------------------------------------------------------


def owns_decode_group(group_index, tpb, cp_size, cp_rank):
    """Mirror of resource_manager._helix_owns_decode_group.

    Ownership is anchored on the deterministic per-request verify-group index
    (advances by exactly one per reserved group) rather than the accept-dependent
    global decode length, so it is host-computable under the overlap scheduler.
    """
    return (group_index // tpb) % cp_size == cp_rank


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
    # Over a full rotation (cp_size * tpb groups) each rank owns an equal share.
    counts = [0] * cp_size
    for group_index in range(cp_size * tpb):
        counts[(group_index // tpb) % cp_size] += 1
    assert counts == [tpb] * cp_size


class _FifoRankState:
    """Minimal mirror of the per-rank reserve/rewind bookkeeping.

    Reproduces resource_manager._helix_prepare_generation_kv (reserve) and
    _helix_rewind_generation_kv (rewind): reserve resolves ownership from the
    deterministic group index and pushes it to a FIFO; rewind pops the FIFO. This
    is the exact interleaving the overlap scheduler produces (reserve of the next
    forward runs before the rewind of the current one).
    """

    def __init__(self, tpb, cp_size, cp_rank):
        self.tpb = tpb
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.group_index = 0
        self.pending = []
        self.reserve_log = []
        self.rewind_log = []

    def reserve(self):
        owns = owns_decode_group(self.group_index, self.tpb, self.cp_size, self.cp_rank)
        self.group_index += 1
        self.pending.append(owns)
        self.reserve_log.append(owns)
        return owns

    def rewind(self):
        owns = self.pending.pop(0)
        self.rewind_log.append(owns)
        return owns


@pytest.mark.parametrize("tpb", [1, 4])
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("pipeline_depth", [0, 1])
@pytest.mark.parametrize("num_iters", [1, 2, 5, 40])
def test_reserve_rewind_fifo_consistency(tpb, cp_size, pipeline_depth, num_iters):
    """Reserve and rewind must agree on ownership for every group.

    pipeline_depth == 0 models the non-overlap scheduler (reserve then rewind in
    the same iteration). pipeline_depth == 1 models the overlap scheduler, where
    the rewind of a group lags its reserve by one iteration, so the reserve of the
    next group (which advances the group counter) runs in between. In both cases
    the FIFO must make reserve and rewind consume the same decision for a given
    group, on every rank.
    """
    for cp_rank in range(cp_size):
        state = _FifoRankState(tpb, cp_size, cp_rank)
        # Prime the pipeline: reserve `pipeline_depth` forwards before rewinding.
        for _ in range(pipeline_depth):
            state.reserve()
        for _ in range(num_iters):
            state.reserve()
            state.rewind()
        # Drain any in-flight forwards.
        while state.pending:
            state.rewind()

        # rewind_log is reserve_log consumed in FIFO order: identical sequences.
        assert state.rewind_log == state.reserve_log
        # And the sequence matches the deterministic group-ownership formula.
        expected = [
            owns_decode_group(i, tpb, cp_size, cp_rank) for i in range(len(state.reserve_log))
        ]
        assert state.reserve_log == expected


@pytest.mark.parametrize("tpb", [1, 4])
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("num_iters", [8, 40])
def test_each_group_owned_by_exactly_one_rank_across_ranks(tpb, cp_size, num_iters):
    # Across all ranks, every verify group index is owned by exactly one rank,
    # so no committed decode group is dropped or double-written under overlap.
    for group_index in range(num_iters):
        owners = [r for r in range(cp_size) if owns_decode_group(group_index, tpb, cp_size, r)]
        assert len(owners) == 1
