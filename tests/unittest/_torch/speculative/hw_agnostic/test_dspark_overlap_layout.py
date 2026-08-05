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
contribute ``runtime_tokens_per_gen_step`` tokens. The flat gathers out of
``new_tokens`` / ``next_draft_tokens`` and the ``position_ids`` /
``gather_ids`` / ``seq_lens`` the host appends break silently when that
assumption is dropped: a wrong count shifts every following request, so RoPE
phases and attention windows go wrong without any shape error.

These tests pin that arithmetic, and -- most importantly -- that a
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


_MISSING = object()


@pytest.mark.parametrize(
    "verify_len, runtime_tokens, expected",
    [
        # Every non-DSpark request lands here: the attribute does not exist.
        (_MISSING, 4, 4),
        # The ragged path declined this request: the attribute is None.
        (None, 4, 4),
        # py_verify_len counts draft positions, matching runtime_draft_len;
        # the accepted token makes it one more.
        (2, 6, 3),
        # A zero-draft window still carries the accepted token.
        (0, 6, 1),
    ],
)
def test_tokens_per_gen_step_resolves_each_request_window(
        verify_len, runtime_tokens, expected):
    req = _Req()
    if verify_len is not _MISSING:
        req.py_verify_len = verify_len
    assert get_request_tokens_per_gen_step(req, runtime_tokens) == expected


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

    # A zero count contributes nothing while its neighbours keep their order,
    # and the empty batch degenerates cleanly.
    assert ragged_gather_index_lists([1, 2, 3], [2, 0, 1]) == ([1, 1, 3], [0, 1, 0])
    assert ragged_gather_index_lists([], []) == ([], [])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        ragged_gather_index_lists([0, 1], [3])
    with pytest.raises(ValueError, match="negative gather count"):
        ragged_gather_index_lists([0], [-1])
