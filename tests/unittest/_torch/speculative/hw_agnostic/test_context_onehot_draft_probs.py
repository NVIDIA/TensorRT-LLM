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
"""Context requests' one-hot draft-prob rows for rejection sampling.

Each context request's slot row becomes a one-hot distribution at draft token
0 over the draft length, every other row and column is left alone, and the
write never synchronizes the host: it runs on every context iteration.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DRAFT_LEN = 3
VOCAB = 6


def _metadata(device):
    # One more step and two more vocab columns than the write covers, so the
    # test sees anything it touches outside [:DRAFT_LEN, :vocab].
    return SimpleNamespace(
        use_rejection_sampling=True,
        draft_probs=torch.full((5, DRAFT_LEN + 1, VOCAB + 2), 0.5, device=device),
        batch_slot_ids=torch.tensor([3, 1, 4, 0], device=device),
        draft_probs_vocab_size=VOCAB,
        draft_probs_last_dim=None,
    )


def _write(metadata, num_contexts, num_gens, gen_vocab=None):
    SpecWorkerBase.write_context_onehot_draft_probs(
        None, metadata, num_contexts, num_gens, DRAFT_LEN, gen_vocab
    )


def _expected(before, slots, vocab):
    expected = before.clone()
    expected[slots, :DRAFT_LEN, :vocab] = 0.0
    expected[slots, :DRAFT_LEN, 0] = 1.0
    return expected


@pytest.mark.parametrize("device", DEVICES)
def test_pure_context_iteration_writes_onehot_rows_over_the_full_vocab(device):
    metadata = _metadata(device)
    before = metadata.draft_probs.clone()

    _write(metadata, num_contexts=2, num_gens=0)

    assert torch.equal(metadata.draft_probs, _expected(before, [3, 1], VOCAB))
    assert metadata.draft_probs_last_dim == VOCAB


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_iteration_uses_the_gen_vocab_width(device):
    metadata = _metadata(device)
    before = metadata.draft_probs.clone()

    _write(metadata, num_contexts=1, num_gens=3, gen_vocab=4)

    assert torch.equal(metadata.draft_probs, _expected(before, [3], 4))
    assert metadata.draft_probs_last_dim is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_context_onehot_write_does_not_synchronize():
    metadata = _metadata("cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _write(metadata, num_contexts=2, num_gens=0)
    finally:
        torch.cuda.set_sync_debug_mode("default")
