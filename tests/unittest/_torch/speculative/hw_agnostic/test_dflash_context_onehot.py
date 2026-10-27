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
"""DFlash block worker: a mixed step's context one-hot draft-prob rows span the width acceptance reads.

With a vocab-sharded drafter the draft logits are a shard of the vocab, but the gen scatter gathers them to the full
vocab and publishes that width (``draft_probs_last_dim``). The new requests' one-hot rows must cover the same width,
or the columns past the shard keep the previous slot occupant's probabilities and the request's first rejection step
reads a corrupt draft distribution."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import _context_onehot_vocab
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

pytestmark = pytest.mark.cpu_only

SHARD, FULL, K, SLOTS = 4, 8, 3, 4


def _metadata(published):
    # Every slot row holds a stale distribution from an earlier occupant.
    return SimpleNamespace(
        use_rejection_sampling=True,
        draft_probs=torch.full((SLOTS, K, FULL), 0.5),
        # One context request (slot 2), then one generation request (slot 0).
        batch_slot_ids=torch.tensor([2, 0]),
        draft_probs_last_dim=published,
        draft_probs_vocab_size=FULL,
    )


def test_width_is_the_published_one():
    assert _context_onehot_vocab(_metadata(FULL), num_gens=1, draft_logits_vocab=SHARD) == FULL


def test_width_falls_back_to_the_logits_before_any_scatter():
    assert _context_onehot_vocab(_metadata(0), num_gens=1, draft_logits_vocab=SHARD) == SHARD


def test_no_generation_requests_leaves_the_width_to_the_writer():
    assert _context_onehot_vocab(_metadata(FULL), num_gens=0, draft_logits_vocab=SHARD) is None


@pytest.mark.parametrize("index_fill", [True, False])
def test_new_request_rows_are_a_full_width_one_hot(monkeypatch, index_fill):
    from tensorrt_llm._torch.speculative import interface

    monkeypatch.setattr(interface, "_ONEHOT_INDEX_FILL_ARMED", True)
    monkeypatch.setattr(interface, "_ONEHOT_INDEX_FILL", index_fill)
    md = _metadata(FULL)
    width = _context_onehot_vocab(md, num_gens=1, draft_logits_vocab=SHARD)
    SpecWorkerBase.write_context_onehot_draft_probs(None, md, 1, 1, K, width)
    row = md.draft_probs[2, :K, : md.draft_probs_last_dim]
    expected = torch.zeros(K, FULL)
    expected[:, 0] = 1.0
    # Every column acceptance reads, not just the shard; the generation request's row is untouched.
    assert torch.equal(row, expected)
    assert torch.equal(md.draft_probs[0], torch.full((K, FULL), 0.5))
