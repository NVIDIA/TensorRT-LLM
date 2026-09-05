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
"""Ragged overlap layout and sparse-attention metadata tests."""

import types

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.metadata import (
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.pyexecutor.llm_request import get_request_tokens_per_gen_step
from tensorrt_llm._torch.speculative.dspark_ragged import ragged_gather_index_lists

# --------------------------------------------------------------------------
# overlap-scheduler input layout: the strided gather assumption and its
# ragged replacement, with the uniform batch reproducing the strided layout.
# --------------------------------------------------------------------------


class _Req:
    """Minimal stand-in exposing only what the layout code reads."""

    def __init__(self, verify_len=None):
        if verify_len is not None:
            self.py_verify_len = verify_len


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
def test_tokens_per_gen_step_resolves_each_request_window(verify_len, runtime_tokens, expected):
    req = _Req()
    if verify_len is not _MISSING:
        req.py_verify_len = verify_len
    assert get_request_tokens_per_gen_step(req, runtime_tokens) == expected


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


# --------------------------------------------------------------------------
# compressor gen counts: _sync_gen_tokens_per_seq must source its per-request
# vector from the DEVICE seq-lens buffer (D2D, not a captured H2D of the host
# split) and use the global max window as the next_n scalar bound.
# --------------------------------------------------------------------------


def _dummy(nc, staged_split, seq_lens_dev, max_draft_tokens=5, cap=16):
    d = types.SimpleNamespace()
    d.is_ragged_verify = True
    d.num_generations = len(staged_split)
    d.num_contexts = nc
    d.max_draft_tokens = max_draft_tokens
    # What the host shape split says (S); the fixed code must NOT read it.
    d.ragged_verify_lens = list(staged_split)
    d.gen_new_tokens_per_seq_cuda = torch.zeros(cap, dtype=torch.int)
    d._seq_lens_cuda = torch.tensor(seq_lens_dev, dtype=torch.int)
    return d


def _call(d, num_gen_tokens):
    return DeepseekV4TrtllmAttentionMetadata._sync_gen_tokens_per_seq(d, num_gen_tokens)


def test_vector_sources_from_seq_lens_not_host_split():
    # Simulate the prologue: host staged S, device rewrote seq lens to w.
    S = [3, 3, 3]
    w = [2, 4, 3]
    d = _dummy(nc=0, staged_split=S, seq_lens_dev=w)
    _call(d, sum(w))
    assert d.gen_new_tokens_per_seq.tolist() == w


def test_scalar_is_global_bound_not_batch_max():
    S = [2, 2]
    d = _dummy(nc=0, staged_split=S, seq_lens_dev=S)
    next_n = _call(d, sum(S))
    assert next_n == 1 + d.max_draft_tokens
    assert d.num_gen_tokens_per_seq == 1 + d.max_draft_tokens


def test_gen_rows_slice_skips_context_rows():
    ctx_lens = [17, 9]
    w = [4, 2]
    d = _dummy(nc=2, staged_split=w, seq_lens_dev=ctx_lens + w)
    _call(d, sum(w))
    assert d.gen_new_tokens_per_seq.tolist() == w


def test_uniform_branch_unchanged():
    d = _dummy(nc=0, staged_split=[3, 3], seq_lens_dev=[3, 3])
    d.is_ragged_verify = False
    next_n = _call(d, 6)
    assert next_n == 3
    assert d.gen_new_tokens_per_seq is None
