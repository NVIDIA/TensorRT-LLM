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
"""Unit tests for the DSpark draft I/O proposal stage."""

import torch

from tensorrt_llm._torch.models.dspark.draft import build_draft_input_ids, dspark_propose
from tensorrt_llm._torch.models.dspark.heads import DSparkConfidenceHead, build_markov_head

VOCAB, HID, RANK, B, BLK = 257, 32, 16, 2, 5
NOISE_ID = 199


def test_build_draft_input_ids():
    bonus = torch.tensor([7, 9])
    ids = build_draft_input_ids(bonus, block_size=BLK, noise_token_id=NOISE_ID)
    assert ids.shape == (B, BLK)
    assert torch.equal(ids[:, 0], bonus)
    assert torch.all(ids[:, 1:] == NOISE_ID)


def test_dspark_propose_full_block_no_confidence():
    torch.manual_seed(0)
    markov = build_markov_head(
        markov_head_type="rnn", vocab_size=VOCAB, markov_rank=RANK, hidden_size=HID
    ).eval()
    base = torch.randn(B, BLK, VOCAB)
    bonus = torch.randint(0, VOCAB, (B,))
    hid = torch.randn(B, BLK, HID)
    with torch.no_grad():
        tokens, num = dspark_propose(
            base,
            bonus_token_ids=bonus,
            block_hidden=hid,
            markov_head=markov,
            confidence_head=None,
            block_size=BLK,
        )
    assert tokens.shape == (B, BLK)
    # No confidence head -> propose the full block.
    assert torch.all(num == BLK)
    # Tokens match the markov head's own greedy block sampling.
    ref_tokens, _ = markov.sample_block_tokens(
        base, first_prev_token_ids=bonus, hidden_states=hid, temperature=0.0
    )
    assert torch.equal(tokens, ref_tokens)


def test_dspark_propose_confidence_truncates():
    torch.manual_seed(1)
    markov = build_markov_head(
        markov_head_type="vanilla", vocab_size=VOCAB, markov_rank=RANK, hidden_size=HID
    ).eval()
    conf = DSparkConfidenceHead(hidden_size=HID).eval()
    # The confidence proj is bias-free, so drive the logit via a constant weight
    # against a constant hidden: logit = weight_val * HID per position.
    base = torch.randn(1, BLK, VOCAB)
    bonus = torch.randint(0, VOCAB, (1,))
    hid = torch.ones(1, BLK, HID)
    with torch.no_grad():
        conf.proj.weight.fill_(5.0 / HID)  # logit ~ 5 -> sigmoid ~ 0.993, all confident
    with torch.no_grad():
        _, num = dspark_propose(
            base,
            bonus_token_ids=bonus,
            block_hidden=hid,
            markov_head=markov,
            confidence_head=conf,
            block_size=BLK,
            confidence_threshold=0.5,
        )
    # All-confident -> full block proposed.
    assert int(num[0]) == BLK
    # Now make the head output low confidence everywhere -> truncate to 0... but
    # confident_prefix_length returns first sub-threshold index (0 here).
    with torch.no_grad():
        conf.proj.weight.fill_(-5.0 / HID)  # logit ~ -5 -> sigmoid ~ 0.0067 < 0.5
        _, num2 = dspark_propose(
            base,
            bonus_token_ids=bonus,
            block_hidden=hid,
            markov_head=markov,
            confidence_head=conf,
            block_size=BLK,
            confidence_threshold=0.5,
        )
    assert int(num2[0]) == 0
