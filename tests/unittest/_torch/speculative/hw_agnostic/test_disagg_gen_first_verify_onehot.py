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
"""Disaggregated generation: a request received from the context server verifies its first draft tokens against a
draft-probability row that this server never wrote (the prefill and any drafting ran on the context server). Under
rejection sampling it must verify dummy tokens against the one-hot placeholder row, or the stale / uninitialized row
accepts the dummies or makes chain_speculative_sampling emit an out-of-range token id."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.interface import SpecMetadata
from tensorrt_llm._torch.speculative.utils import disagg_gen_first_draft_tokens

pytestmark = pytest.mark.cpu_only

K, SLOTS, VOCAB = 3, 4, 8
REJ = SimpleNamespace(use_rejection_sampling=True)
STRICT = SimpleNamespace(use_rejection_sampling=False)


@pytest.mark.parametrize("ctx_drafts", [None, [], [5, 6, 7]])
def test_rejection_always_verifies_dummies_against_the_one_hot_row(ctx_drafts):
    assert disagg_gen_first_draft_tokens(ctx_drafts, REJ, True, K) == ([0] * K, True)


def test_strict_keeps_context_server_drafts():
    assert disagg_gen_first_draft_tokens([5, 6, 7], STRICT, True, K) == ([5, 6, 7], False)


@pytest.mark.parametrize("ctx_drafts", [None, []])
def test_strict_without_drafts_gets_dummies(ctx_drafts):
    assert disagg_gen_first_draft_tokens(ctx_drafts, STRICT, True, K) == ([0] * K, False)


@pytest.mark.parametrize("ctx_drafts, expected", [(None, []), ([5, 6, 7], [5, 6, 7])])
def test_no_spec_decode_is_unchanged(ctx_drafts, expected):
    assert disagg_gen_first_draft_tokens(ctx_drafts, REJ, False, K) == (expected, False)


def test_flagged_slot_gets_a_one_hot_row_at_the_dummy():
    # The flag makes input preparation hand the slot to write_padding_onehot_draft_probs; its row then matches the
    # dummy draft (token 0) in full, and a slot that is not flagged keeps its stale row.
    md = SimpleNamespace(
        use_rejection_sampling=True,
        draft_probs=torch.full((SLOTS, K, VOCAB), 0.5),
        draft_probs_last_dim=VOCAB,
        draft_probs_vocab_size=VOCAB,
    )
    tokens, needs_onehot = disagg_gen_first_draft_tokens(None, REJ, True, K)
    assert needs_onehot
    SpecMetadata.write_padding_onehot_draft_probs(md, [2], K)
    expected = torch.zeros(K, VOCAB)
    expected[:, tokens[0]] = 1.0
    assert torch.equal(md.draft_probs[2], expected)
    assert torch.equal(md.draft_probs[1], torch.full((K, VOCAB), 0.5))
