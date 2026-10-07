# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU numerical regressions for confidence input routing, not model accuracy."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dspark import DSv4DSparkDraftModel, dspark_propose
from tensorrt_llm._torch.models.modeling_speculative import DSparkConfidenceHead, build_markov_head


@pytest.mark.parametrize("physical_k", range(1, 9))
@pytest.mark.parametrize(
    "head_type,with_markov",
    [
        ("none", False),
        ("vanilla", False),
        ("vanilla", True),
        ("gated", False),
        ("gated", True),
        ("rnn", False),
        ("rnn", True),
    ],
)
@torch.no_grad()
def test_confidence_uses_prenorm_without_changing_proposals(physical_k, head_type, with_markov):
    torch.manual_seed(18000 + physical_k)
    hidden_size, rank, vocab_size, batch = 16, 8, 31, 3
    markov = build_markov_head(
        markov_head_type=head_type,
        vocab_size=vocab_size,
        markov_rank=0 if head_type == "none" else rank,
        hidden_size=hidden_size,
    )
    confidence = DSparkConfidenceHead(
        hidden_size=hidden_size,
        markov_rank=rank,
        with_markov=with_markov,
        block_size=physical_k,
    )
    norm = torch.nn.RMSNorm(hidden_size, eps=1e-6)
    norm.weight.copy_(torch.linspace(0.25, 2.5, hidden_size))
    last = SimpleNamespace(
        hc_head=lambda h: h.mean(dim=-2),
        norm=norm,
        markov_head=markov,
        confidence_head=confidence,
    )
    model = SimpleNamespace(
        mtp_layers=[last],
        lm_head=torch.nn.Linear(hidden_size, vocab_size, bias=False),
        block_size=physical_k,
    )
    hidden = torch.randn(batch, physical_k, 4, hidden_size) * 3
    bonus = torch.arange(batch) + 2
    pre_norm = last.hc_head(hidden)
    post_norm = norm(pre_norm)

    for temperature in (0.0, 0.8):
        outputs, states = [], []
        for enabled in (False, True):
            torch.manual_seed(912)
            outputs.append(
                DSv4DSparkDraftModel.forward_head(
                    model,
                    hidden,
                    bonus,
                    return_confidence=enabled,
                    return_logits=True,
                    temperature=temperature,
                )
            )
            states.append(torch.get_rng_state())
        off, on = outputs
        assert off[1] is None
        assert torch.equal(off[0], on[0])
        assert torch.equal(off[2], on[2])
        assert torch.equal(states[0], states[1])
        # The legacy functional caller keeps its proposal-head contract.
        torch.manual_seed(912)
        legacy = dspark_propose(
            model.lm_head(post_norm),
            bonus_token_ids=bonus,
            block_hidden=post_norm,
            markov_head=markov,
            confidence_head=confidence,
            block_size=physical_k,
            temperature=temperature,
            return_confidence=True,
            return_logits=True,
        )
        assert torch.equal(on[0], legacy[0])
        assert torch.equal(on[2], legacy[2])
        prev = torch.cat([bonus.unsqueeze(1), on[0][:, :-1]], dim=1)
        embeddings = markov.get_prev_embeddings(prev) if with_markov else None
        expected = confidence(pre_norm, prev_embeddings=embeddings)
        torch.testing.assert_close(on[1], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            legacy[1], confidence(post_norm, prev_embeddings=embeddings), rtol=0, atol=0
        )
        # This fixture must expose the previous post-norm scoring defect.
        assert not torch.allclose(on[1], legacy[1], rtol=1e-6, atol=1e-6)
