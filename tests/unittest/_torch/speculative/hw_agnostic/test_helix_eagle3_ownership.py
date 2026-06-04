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
- every decode token is owned by exactly one CP rank (round-robin by block);
- the closed-form owned-decode count matches a brute-force count;
- the per-query-token global positions and per-token ownership computed for a
  speculative verify forward are consistent (re-fed token never written; draft
  tokens owned by their block owner; kv_lens grows by the owned count).

The helpers below mirror the production formulas in
``tensorrt_llm/_torch/pyexecutor/resource_manager.py`` (``_helix_owns_decode_index``,
``_helix_owned_decode_count``) and ``model_engine._helix_verify_token_params``.
"""

import pytest


def owns_decode_index(decode_index, tpb, cp_size, cp_rank):
    return (decode_index // tpb) % cp_size == cp_rank


def owned_decode_count_closed_form(n, tpb, cp_size, cp_rank):
    if n <= 0:
        return 0
    full_blocks = n // tpb
    rem = n % tpb
    owned_full_blocks = full_blocks // cp_size + (1 if cp_rank < full_blocks % cp_size else 0)
    owned = owned_full_blocks * tpb
    if rem > 0 and (full_blocks % cp_size == cp_rank):
        owned += rem
    return owned


def owned_decode_count_bruteforce(n, tpb, cp_size, cp_rank):
    return sum(1 for j in range(n) if owns_decode_index(j, tpb, cp_size, cp_rank))


@pytest.mark.parametrize("tpb", [1, 4, 32])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("n", [0, 1, 5, 31, 32, 33, 64, 100, 257])
def test_owned_decode_count_closed_form_matches_bruteforce(tpb, cp_size, n):
    for cp_rank in range(cp_size):
        assert owned_decode_count_closed_form(
            n, tpb, cp_size, cp_rank
        ) == owned_decode_count_bruteforce(n, tpb, cp_size, cp_rank)


@pytest.mark.parametrize("tpb", [1, 4, 32])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("n", [1, 5, 32, 33, 100, 257])
def test_every_decode_token_owned_by_exactly_one_rank(tpb, cp_size, n):
    # Each decode-index is owned by exactly one rank.
    for j in range(n):
        owners = [
            cp_rank for cp_rank in range(cp_size) if owns_decode_index(j, tpb, cp_size, cp_rank)
        ]
        assert owners == [(j // tpb) % cp_size]
    # Owned counts across ranks partition the total.
    total = sum(
        owned_decode_count_closed_form(n, tpb, cp_size, cp_rank) for cp_rank in range(cp_size)
    )
    assert total == n


def verify_token_params(total_input_len, g, num_draft, tpb, cp_size, cp_rank):
    """Mirror of model_engine._helix_verify_token_params."""
    is_refed = g > 0
    first_decode_index = (g - 1) if is_refed else g
    first_pos = total_input_len + first_decode_index
    global_positions = list(range(first_pos, first_pos + 1 + num_draft))
    inactive_flags = []
    num_active = 0
    for j in range(1 + num_draft):
        if is_refed and j == 0:
            inactive_flags.append(True)  # re-fed token: never written
            continue
        decode_index = first_decode_index + j
        owned = owns_decode_index(decode_index, tpb, cp_size, cp_rank)
        inactive_flags.append(not owned)
        if owned:
            num_active += 1
    return global_positions, inactive_flags, num_active


@pytest.mark.parametrize("tpb", [4, 32])
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("num_draft", [0, 1, 3])
@pytest.mark.parametrize("g", [0, 1, 5, 32])
def test_verify_token_params_consistency(tpb, cp_size, num_draft, g):
    total_input_len = 100
    # Across all ranks, each newly written draft token (decode-index >= g) must be
    # owned by exactly one rank; the re-fed token (g-1) is never written.
    written_owner_counts = {}
    for cp_rank in range(cp_size):
        positions, inactive, num_active = verify_token_params(
            total_input_len, g, num_draft, tpb, cp_size, cp_rank
        )
        # Positions are global and contiguous.
        assert positions == list(range(positions[0], positions[0] + 1 + num_draft))
        is_refed = g > 0
        first_decode_index = (g - 1) if is_refed else g
        # First position corresponds to the first query token's global position.
        assert positions[0] == total_input_len + first_decode_index
        # num_active equals the number of non-inactive (owned) tokens.
        assert num_active == sum(1 for f in inactive if not f)
        if is_refed:
            assert inactive[0] is True  # re-fed token never written
        for j, flag in enumerate(inactive):
            if is_refed and j == 0:
                continue
            decode_index = first_decode_index + j
            if not flag:
                written_owner_counts.setdefault(decode_index, 0)
                written_owner_counts[decode_index] += 1
    # Every written token is owned by exactly one rank.
    for decode_index, count in written_owner_counts.items():
        assert count == 1, (decode_index, count)
    # The set of written tokens is exactly the draft tokens [g, g+num_draft)
    # plus the new token at g when there is no re-fed token (g == 0).
    expected_written = set(range(g, g + num_draft)) if g > 0 else set(range(g, g + 1 + num_draft))
    assert set(written_owner_counts.keys()) == expected_written


@pytest.mark.parametrize(
    "tpb,cp_size,g,num_draft,expect_split",
    [
        # Drafts [g, g+num_draft) straddle a tokens_per_block boundary, so they are
        # split across two ranks.
        (4, 2, 3, 3, True),  # drafts 3,4,5 -> ranks 0,1,1
        (32, 2, 30, 3, True),  # drafts 30,31,32 -> ranks 0,0,1
        (4, 4, 3, 3, True),  # drafts 3,4,5 -> ranks 0,1,1
        # Drafts stay within one block -> all on a single rank.
        (4, 2, 5, 3, False),  # drafts 5,6,7 -> rank 1
        (32, 2, 1, 3, False),  # drafts 1,2,3 -> rank 0
    ],
)
def test_verify_token_params_straddler(tpb, cp_size, g, num_draft, expect_split):
    # All g values here are > 0, so the re-fed token at g-1 is never written and
    # the written drafts are exactly [g, g+num_draft).
    assert g > 0
    per_rank_active = []
    written_owner_counts = {}
    for cp_rank in range(cp_size):
        _, inactive, num_active = verify_token_params(100, g, num_draft, tpb,
                                                      cp_size, cp_rank)
        per_rank_active.append(num_active)
        # Re-fed token (index 0) is inactive on every rank.
        assert inactive[0] is True
        for j, flag in enumerate(inactive):
            if j == 0:
                continue
            if not flag:
                decode_index = (g - 1) + j
                written_owner_counts.setdefault(decode_index, 0)
                written_owner_counts[decode_index] += 1
    # Every written draft owned by exactly one rank; total equals num_draft.
    assert set(written_owner_counts.keys()) == set(range(g, g + num_draft))
    assert all(c == 1 for c in written_owner_counts.values())
    assert sum(per_rank_active) == num_draft
    ranks_with_writes = sum(1 for a in per_rank_active if a > 0)
    if expect_split:
        # Drafts of a single verify step are owned by more than one rank.
        assert ranks_with_writes >= 2
    else:
        assert ranks_with_writes == 1
