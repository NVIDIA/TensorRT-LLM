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
"""Overlap-scheduler input layout under ragged (per-request) verification.

The overlap path builds this iteration's inputs from the *previous* iteration's
device tensors, and does it with a fixed stride: every request is assumed to
contribute ``runtime_tokens_per_gen_step`` tokens. Two places break silently
when that assumption is dropped:

  * the flat gathers out of ``new_tokens`` / ``next_draft_tokens``, and the
    ``position_ids`` / ``gather_ids`` / ``seq_lens`` the host appends -- a wrong
    count shifts every following request, so RoPE phases and attention windows
    go wrong without any shape error; and
  * ``kv_len_offsets``, which is subtracted from a host-side estimate that was
    built with the *same* count. If only one of the two is made per-request the
    KV length is off by exactly their difference.

These tests pin the arithmetic of both, and -- most importantly -- that a
uniform batch reproduces the strided layout exactly, since that is the path
every non-DSpark model keeps taking.
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import get_request_tokens_per_gen_step
from tensorrt_llm._torch.speculative.dspark_ragged import ragged_gather_index_lists


class _Req:
    """Minimal stand-in exposing only what the layout code reads."""

    def __init__(self, verify_len=None):
        if verify_len is not None:
            self.py_verify_len = verify_len


# --------------------------------------------------------------------------
# get_request_tokens_per_gen_step
# --------------------------------------------------------------------------


def test_missing_attribute_yields_the_batch_wide_count():
    # Every non-DSpark request lands here: the attribute does not exist at all.
    assert get_request_tokens_per_gen_step(_Req(), 4) == 4


def test_none_verify_len_yields_the_batch_wide_count():
    req = _Req()
    req.py_verify_len = None
    assert get_request_tokens_per_gen_step(req, 4) == 4


def test_verify_len_is_a_draft_length_not_a_token_count():
    # py_verify_len counts draft positions, matching runtime_draft_len; the
    # accepted token makes it one more.
    assert get_request_tokens_per_gen_step(_Req(2), 6) == 3


def test_zero_verify_len_still_carries_the_accepted_token():
    assert get_request_tokens_per_gen_step(_Req(0), 6) == 1


# --------------------------------------------------------------------------
# ragged_gather_index_lists
# --------------------------------------------------------------------------


def test_uniform_counts_reproduce_the_strided_gather():
    # This is the safety property the whole change rests on: with equal counts
    # the index list picks exactly the elements tensor[slots, :width] does.
    table = torch.arange(6 * 5).reshape(6, 5)
    slots = [3, 0, 4]
    width = 5

    rows, cols = ragged_gather_index_lists(slots, [width] * len(slots))
    ragged = table[torch.tensor(rows), torch.tensor(cols)]
    strided = table[torch.tensor(slots), :width].flatten()

    assert torch.equal(ragged, strided)


def test_ragged_counts_pack_each_request_window():
    table = torch.arange(4 * 5).reshape(4, 5)
    rows, cols = ragged_gather_index_lists([2, 0], [3, 1])
    gathered = table[torch.tensor(rows), torch.tensor(cols)]

    # slot 2 columns 0..2, then slot 0 column 0
    assert gathered.tolist() == [10, 11, 12, 0]


def test_zero_count_contributes_nothing():
    rows, cols = ragged_gather_index_lists([1, 2, 3], [2, 0, 1])
    assert rows == [1, 1, 3]
    assert cols == [0, 1, 0]


def test_empty_batch():
    assert ragged_gather_index_lists([], []) == ([], [])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        ragged_gather_index_lists([0, 1], [3])


def test_negative_count_raises():
    with pytest.raises(ValueError, match="negative gather count"):
        ragged_gather_index_lists([0], [-1])


def test_total_gathered_equals_the_sum_of_counts():
    counts = [4, 1, 3, 2]
    rows, cols = ragged_gather_index_lists([0, 1, 2, 3], counts)
    assert len(rows) == len(cols) == sum(counts)


# --------------------------------------------------------------------------
# Host-side layout arithmetic (mirrors the extend-request loop)
# --------------------------------------------------------------------------


def _build_overlap_layout(requests, past_seens, runtime_tokens_per_gen_step):
    """Reproduce the per-request bookkeeping the overlap branch emits.

    Mirrors _prepare_tp_inputs' previous-batch branch: one entry per request in
    seq_lens/draft_lens/cached, and ``tokens`` flat entries per request in
    position_ids/gather_ids/previous_pos_indices.
    """
    seq_lens, draft_lens, cached, tokens_each = [], [], [], []
    position_ids, gather_ids, pos_indices = [], [], []
    for slot, (req, past_seen) in enumerate(zip(requests, past_seens)):
        tokens = get_request_tokens_per_gen_step(req, runtime_tokens_per_gen_step)
        seq_lens.append(tokens)
        draft_lens.append(tokens - 1)
        tokens_each.append(tokens)
        gather_ids.extend(range(len(position_ids), len(position_ids) + tokens))
        position_ids.extend(range(past_seen, past_seen + tokens))
        pos_indices.extend([slot] * tokens)
        cached.append(past_seen + tokens)
    return {
        "seq_lens": seq_lens,
        "draft_lens": draft_lens,
        "cached": cached,
        "tokens_each": tokens_each,
        "position_ids": position_ids,
        "gather_ids": gather_ids,
        "pos_indices": pos_indices,
    }


def test_uniform_batch_reproduces_the_fixed_stride_layout():
    n = 4
    reqs = [_Req(), _Req(), _Req()]
    past = [100, 250, 7]
    got = _build_overlap_layout(reqs, past, n)

    assert got["seq_lens"] == [n] * 3
    assert got["draft_lens"] == [n - 1] * 3
    assert len(got["position_ids"]) == 3 * n
    assert got["pos_indices"] == [0] * n + [1] * n + [2] * n
    assert got["gather_ids"] == list(range(3 * n))
    for slot, base in enumerate(past):
        window = got["position_ids"][slot * n : (slot + 1) * n]
        assert window == list(range(base, base + n))


def test_ragged_batch_keeps_each_request_positions_contiguous():
    # The bug this pins: emitting a fixed count per request shifts every later
    # request's position_ids, which is a silent RoPE-phase error.
    reqs = [_Req(3), _Req(0), _Req(1)]
    past = [100, 250, 7]
    got = _build_overlap_layout(reqs, past, 6)

    assert got["tokens_each"] == [4, 1, 2]
    assert got["seq_lens"] == [4, 1, 2]
    assert got["draft_lens"] == [3, 0, 1]
    assert got["position_ids"] == [100, 101, 102, 103, 250, 7, 8]
    assert got["gather_ids"] == list(range(7))
    assert got["pos_indices"] == [0, 0, 0, 0, 1, 2, 2]


def test_flat_token_total_matches_the_sum_of_windows():
    reqs = [_Req(3), _Req(0), _Req(1)]
    got = _build_overlap_layout(reqs, [0, 0, 0], 6)
    assert len(got["position_ids"]) == sum(got["tokens_each"])
    assert len(got["pos_indices"]) == sum(got["tokens_each"])


# --------------------------------------------------------------------------
# kv_len_offsets: the host estimate and the correction must use one count
# --------------------------------------------------------------------------


def _corrected_kv_lens(
    requests, past_seens, accepted, runtime_tokens_per_gen_step, correction_tokens
):
    """Host estimate (past_seen + tokens) plus the device correction.

    ``correction_tokens`` is what kv_len_offsets subtracts; passing the
    batch-wide value while the estimate used a per-request one is exactly the
    A2 bug.
    """
    out = []
    for req, past_seen, acc, corr in zip(requests, past_seens, accepted, correction_tokens):
        tokens = get_request_tokens_per_gen_step(req, runtime_tokens_per_gen_step)
        estimate = past_seen + tokens
        out.append(estimate + (acc - corr))
    return out


def test_paired_counts_cancel_to_past_seen_plus_accepted():
    reqs = [_Req(3), _Req(0), _Req(1)]
    past = [100, 250, 7]
    accepted = [2, 1, 2]
    tokens_each = [get_request_tokens_per_gen_step(r, 6) for r in reqs]

    got = _corrected_kv_lens(reqs, past, accepted, 6, tokens_each)

    # The token count drops out entirely -- that is what makes the correction
    # correct regardless of which window the scheduler picked.
    assert got == [p + a for p, a in zip(past, accepted)]


def test_batch_wide_correction_under_ragged_is_off_by_the_window_difference():
    reqs = [_Req(3), _Req(0), _Req(1)]
    past = [100, 250, 7]
    accepted = [2, 1, 2]
    n = 6

    wrong = _corrected_kv_lens(reqs, past, accepted, n, [n] * 3)
    right = [p + a for p, a in zip(past, accepted)]

    # Off by (per-request tokens - batch-wide tokens) for every request.
    deltas = [w - r for w, r in zip(wrong, right)]
    assert deltas == [4 - n, 1 - n, 2 - n]
    assert all(d != 0 for d in deltas)


def test_uniform_batch_is_unaffected_by_the_pairing_rule():
    reqs = [_Req(), _Req(), _Req()]
    past = [100, 250, 7]
    accepted = [2, 1, 2]
    n = 4

    per_request = _corrected_kv_lens(reqs, past, accepted, n, [n] * 3)
    assert per_request == [p + a for p, a in zip(past, accepted)]


# --------------------------------------------------------------------------
# Derived totals the callers compute from the per-request counts
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verify_lens",
    [
        [None, None, None],
        [3, 3, 3],
        [3, 0, 1],
        [0, 0, 0],
    ],
)
def test_draft_token_total_is_tokens_minus_one_per_request(verify_lens):
    # previous_batch_draft_tokens = total_num_tokens - num_requests, which has
    # to agree with summing (tokens - 1) request by request.
    reqs = [_Req(v) for v in verify_lens]
    tokens = [get_request_tokens_per_gen_step(r, 4) for r in reqs]
    total_num_tokens = sum(tokens)
    assert total_num_tokens - len(reqs) == sum(t - 1 for t in tokens)


def test_uniform_totals_match_the_multiplicative_formulas():
    # The formulas the uniform path used before this change.
    n, num_requests = 4, 5
    reqs = [_Req() for _ in range(num_requests)]
    tokens = [get_request_tokens_per_gen_step(r, n) for r in reqs]
    assert sum(tokens) == num_requests * n
    assert sum(tokens) - num_requests == num_requests * (n - 1)
