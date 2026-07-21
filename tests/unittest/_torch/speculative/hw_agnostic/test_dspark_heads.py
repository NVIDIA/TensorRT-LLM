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
"""Unit tests for the DSpark draft-network heads (hardware-agnostic, CPU)."""

import pytest
import torch

from tensorrt_llm._torch.models.dspark.heads import (
    DSparkConfidenceHead,
    RNNHead,
    VanillaMarkov,
    build_markov_head,
    confident_prefix_length,
)

VOCAB, RANK, HID, B, BLK = 257, 16, 32, 3, 5


@pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
def test_markov_block_sampling_shapes_and_determinism(head_type):
    torch.manual_seed(0)
    head = build_markov_head(
        markov_head_type=head_type, vocab_size=VOCAB, markov_rank=RANK, hidden_size=HID
    ).eval()
    base = torch.randn(B, BLK, VOCAB)
    first = torch.randint(0, VOCAB, (B,))
    hid = torch.randn(B, BLK, HID)
    with torch.no_grad():
        tok, logits = head.sample_block_tokens(
            base, first_prev_token_ids=first, hidden_states=hid, temperature=0.0
        )
        tok2, _ = head.sample_block_tokens(
            base, first_prev_token_ids=first, hidden_states=hid, temperature=0.0
        )
    assert tok.shape == (B, BLK)
    assert logits.shape == (B, BLK, VOCAB)
    # Greedy is deterministic.
    assert torch.equal(tok, tok2)
    # Each sampled token is the argmax of its (bias-corrected) step logits.
    assert torch.equal(tok, logits.argmax(dim=-1))


def test_markov_bias_is_additive_low_rank():
    # bias = W2(W1[token]); the corrected first-step logits == base + bias.
    torch.manual_seed(1)
    head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK).eval()
    base = torch.randn(B, BLK, VOCAB)
    first = torch.randint(0, VOCAB, (B,))
    with torch.no_grad():
        _, corrected = head.sample_block_tokens(
            base, first_prev_token_ids=first, hidden_states=None, temperature=0.0
        )
        expected0 = base[:, 0] + head.markov_w2(head.markov_w1(first))
    assert torch.allclose(corrected[:, 0], expected0, atol=1e-5)


def test_rnn_state_carries_across_positions():
    torch.manual_seed(2)
    head = RNNHead(vocab_size=VOCAB, markov_rank=RANK, hidden_size=HID).eval()
    initial_state = torch.zeros(1, RANK)
    prev_embedding = head.get_prev_embeddings(torch.zeros(1, dtype=torch.long))
    prefix_hidden = torch.randn(1, HID)
    current_hidden = torch.randn(1, HID)

    with torch.no_grad():
        state_a, _ = head._rnn_step(initial_state, prev_embedding, prefix_hidden)
        state_b, _ = head._rnn_step(initial_state, prev_embedding, -prefix_hidden)
        _, bias_a = head._rnn_step(state_a, prev_embedding, current_hidden)
        _, bias_b = head._rnn_step(state_b, prev_embedding, current_hidden)

    assert not torch.allclose(state_a, state_b)
    assert not torch.allclose(bias_a, bias_b)


def test_build_markov_head_rank_zero_returns_none():
    assert (
        build_markov_head(
            markov_head_type="vanilla", vocab_size=VOCAB, markov_rank=0, hidden_size=HID
        )
        is None
    )


def test_confidence_head_and_prefix_truncation():
    head = DSparkConfidenceHead(hidden_size=HID)
    conf = head(torch.randn(B, BLK, HID))
    assert conf.shape == (B, BLK)
    # threshold 0 disables truncation.
    assert confident_prefix_length(conf, block_size=BLK, threshold=0.0) == BLK
    # First sub-threshold position truncates the prefix.
    logits = torch.tensor([[10.0, 10.0, -10.0, 10.0, 10.0]])
    assert confident_prefix_length(logits, block_size=BLK, threshold=0.5) == 2
    # All-confident -> full block.
    logits_hi = torch.full((1, BLK), 10.0)
    assert confident_prefix_length(logits_hi, block_size=BLK, threshold=0.5) == BLK


def test_confidence_head_with_markov_concat_dim():
    head = DSparkConfidenceHead(hidden_size=HID, markov_rank=RANK, with_markov=True)
    hid = torch.randn(B, BLK, HID)
    prev_emb = torch.randn(B, BLK, RANK)
    out = head(hid, prev_embeddings=prev_emb)
    assert out.shape == (B, BLK)
    with pytest.raises(AssertionError):
        head(hid)  # with_markov requires prev_embeddings
