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
"""write_padding_onehot_draft_probs: same rows as an advanced-index write; with the metadata's
context_onehot_index_fill opt-in, without a host sync. Off, the advanced-index write itself."""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.interface import SpecMetadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

NUM_SLOTS, DRAFT_LEN, VOCAB = 9, 4, 96
# A slot may repeat: one row written twice with the same values.
SLOT_IDS = [5, 1, 7, 1]


def _case(last_dim, index_fill=True):
    torch.manual_seed(0)
    draft_probs = torch.rand((NUM_SLOTS, DRAFT_LEN + 1, VOCAB), device="cuda")
    metadata = types.SimpleNamespace(
        use_rejection_sampling=True,
        draft_probs=draft_probs.clone(),
        draft_probs_last_dim=last_dim,
        draft_probs_vocab_size=VOCAB,
        # The last row is the shared scratch row; every padding slot sits below it.
        dummy_slot_row=NUM_SLOTS - 1,
        num_seq_slots=NUM_SLOTS - 1,
        max_num_requests=NUM_SLOTS - 1,
        is_cuda_graph=False,
        context_onehot_index_fill=index_fill,
    )
    onehot_vocab = last_dim if last_dim > 0 else VOCAB
    slots = torch.tensor(SLOT_IDS, dtype=torch.long, device="cuda")
    expected = draft_probs.clone()
    expected[slots, :DRAFT_LEN, :onehot_vocab] = 0.0
    expected[slots, :DRAFT_LEN, 0] = 1.0
    torch.cuda.synchronize()
    return metadata, expected


@pytest.mark.parametrize("last_dim", [0, 64])
def test_padding_onehot_matches_indexed_write_without_sync(last_dim):
    metadata, expected = _case(last_dim)

    torch.cuda.set_sync_debug_mode("error")
    try:
        SpecMetadata.write_padding_onehot_draft_probs(metadata, SLOT_IDS, DRAFT_LEN)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert torch.equal(metadata.draft_probs, expected)


def _index_fill_calls(monkeypatch):
    calls = []
    index_fill = torch.Tensor.index_fill_

    def spy(self, *args):
        calls.append(args)
        return index_fill(self, *args)

    monkeypatch.setattr(torch.Tensor, "index_fill_", spy)
    return calls


@pytest.mark.parametrize("index_fill", [True, False])
def test_the_opt_in_alone_selects_index_fill(monkeypatch, index_fill):
    # Off (a model that opts into nothing, or the kill switch): main's advanced-index write.
    metadata, expected = _case(64, index_fill=index_fill)
    calls = _index_fill_calls(monkeypatch)

    SpecMetadata.write_padding_onehot_draft_probs(metadata, SLOT_IDS, DRAFT_LEN)

    assert torch.equal(metadata.draft_probs, expected)
    assert (len(calls) > 0) is index_fill


@pytest.mark.parametrize(
    "opted_in, killed, enabled",
    [
        (False, False, False),
        (True, False, True),
        (True, True, False),
    ],
)
def test_the_kill_switch_wins_over_the_opt_in(monkeypatch, opted_in, killed, enabled):
    if killed:
        monkeypatch.setenv("TRTLLM_M31_DISABLE_ONEHOT_INDEX_FILL", "1")
    else:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_ONEHOT_INDEX_FILL", raising=False)
    assert interface.onehot_index_fill_enabled(opted_in) is enabled
