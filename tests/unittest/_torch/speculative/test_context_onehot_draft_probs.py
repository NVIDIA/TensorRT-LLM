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
"""write_context_onehot_draft_probs: same rows as an advanced-index write, without a host sync."""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("num_gens", [0, 3])
def test_context_onehot_matches_indexed_write_without_sync(num_gens):
    torch.manual_seed(0)
    num_slots, draft_len, vocab, gen_vocab = 9, 4, 96, 64
    num_contexts = 2
    draft_probs = torch.rand((num_slots, draft_len + 1, vocab), device="cuda")
    slot_ids = torch.tensor([5, 1, 7, 2, 0], dtype=torch.long, device="cuda")
    metadata = types.SimpleNamespace(
        use_rejection_sampling=True,
        draft_probs=draft_probs.clone(),
        batch_slot_ids=slot_ids,
        draft_probs_vocab_size=vocab,
        draft_probs_last_dim=None,
    )
    onehot_vocab = gen_vocab if num_gens > 0 else vocab
    expected = draft_probs.clone()
    expected[slot_ids[:num_contexts], :draft_len, :onehot_vocab] = 0.0
    expected[slot_ids[:num_contexts], :draft_len, 0] = 1.0
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        SpecWorkerBase.write_context_onehot_draft_probs(
            None, metadata, num_contexts, num_gens, draft_len, gen_vocab
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert torch.equal(metadata.draft_probs, expected)
    assert metadata.draft_probs_last_dim == (None if num_gens > 0 else vocab)
