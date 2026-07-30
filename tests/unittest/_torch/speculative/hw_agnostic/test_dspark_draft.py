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
        tokens, confidence = dspark_propose(
            base,
            bonus_token_ids=bonus,
            block_hidden=hid,
            markov_head=markov,
            confidence_head=None,
            block_size=BLK,
        )
    assert tokens.shape == (B, BLK)
    assert confidence is None
    # Tokens match the markov head's own greedy block sampling.
    ref_tokens, _ = markov.sample_block_tokens(
        base, first_prev_token_ids=bonus, hidden_states=hid, temperature=0.0
    )
    assert torch.equal(tokens, ref_tokens)


def _propose_with_confidence(conf, markov, *, return_confidence):
    base = torch.randn(1, BLK, VOCAB)
    bonus = torch.randint(0, VOCAB, (1,))
    hid = torch.ones(1, BLK, HID)
    with torch.no_grad():
        return dspark_propose(
            base,
            bonus_token_ids=bonus,
            block_hidden=hid,
            markov_head=markov,
            confidence_head=conf,
            block_size=BLK,
            return_confidence=return_confidence,
        )


def test_dspark_propose_scores_without_shortening_the_block():
    """The block is always proposed in full; confidence only scores it."""
    torch.manual_seed(1)
    markov = build_markov_head(
        markov_head_type="vanilla", vocab_size=VOCAB, markov_rank=RANK, hidden_size=HID
    ).eval()
    conf = DSparkConfidenceHead(hidden_size=HID, block_size=BLK).eval()
    # The confidence proj is bias-free, so drive the logit via a constant weight
    # against a constant hidden: logit = weight_val * HID per position.
    with torch.no_grad():
        conf.proj.weight.fill_(-5.0 / HID)  # sigmoid ~ 0.0067, i.e. hopeless

    tokens, confidence = _propose_with_confidence(conf, markov, return_confidence=True)
    assert tokens.shape == (1, BLK)
    assert confidence.shape == (1, BLK)
    # Low confidence must NOT shorten the proposal -- that decision belongs to
    # the verification scheduler, not the drafter.
    assert torch.all(confidence < 0.0)
    assert torch.all(conf.apply_sts(confidence) < 0.5)


def test_dspark_propose_confidence_is_opt_in():
    torch.manual_seed(2)
    markov = build_markov_head(
        markov_head_type="vanilla", vocab_size=VOCAB, markov_rank=RANK, hidden_size=HID
    ).eval()
    conf = DSparkConfidenceHead(hidden_size=HID, block_size=BLK).eval()
    _, confidence = _propose_with_confidence(conf, markov, return_confidence=False)
    assert confidence is None


def test_dspark_propose_is_free_of_host_syncs():
    """No ``.item()``/``nonzero`` on this path: it runs inside the target's graph."""
    import inspect
    import io
    import tokenize

    from tensorrt_llm._torch.models.dspark import draft as draft_mod

    src = inspect.getsource(draft_mod.dspark_propose)
    # Strip comments and string literals so the check reads the code, not the
    # prose describing it (the implementation comments name these very calls).
    code = "".join(
        tok.string if tok.type not in (tokenize.COMMENT, tokenize.STRING) else " "
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
    )
    for banned in (".item(", "nonzero", "range("):
        assert banned not in code, f"dspark_propose must stay capture-safe: found {banned!r}"
