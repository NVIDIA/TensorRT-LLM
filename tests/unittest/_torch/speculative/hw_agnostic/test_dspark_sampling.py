# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark ancestral drafting and block verification under rejection sampling."""

import math
import types
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias
from tensorrt_llm._torch.speculative.block_verification import block_verify
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

VOCAB, K = 3, 3


def _prefix_model(generator, sharpness):
    """A next-token distribution for every prefix of length 0..K."""
    num_prefixes = sum(VOCAB**m for m in range(K + 1))
    logits = sharpness * torch.randn(num_prefixes, VOCAB, generator=generator, dtype=torch.float64)
    return torch.softmax(logits, -1)


def _rows(model, prefixes):
    ids = torch.zeros(prefixes.shape[0], dtype=torch.int64)
    for m in range(prefixes.shape[1]):
        ids = ids * VOCAB + prefixes[:, m]
    return model[ids + sum(VOCAB**m for m in range(prefixes.shape[1]))]


def test_block_verification_is_lossless():
    """Blocks verified, then completed by sampling the target, are distributed as the target's own samples."""
    generator = torch.Generator().manual_seed(0)
    target, draft = _prefix_model(generator, 1.5), _prefix_model(generator, 1.0)
    num = 200_000
    drafts = torch.empty(num, 0, dtype=torch.int64)
    draft_rows = []
    for _ in range(K):
        draft_rows.append(_rows(draft, drafts))
        drafts = torch.cat([drafts, torch.multinomial(draft_rows[-1], 1, generator=generator)], 1)
    target_rows = [_rows(target, drafts[:, :j]) for j in range(K + 1)]
    tokens, num_accepted = block_verify(
        drafts,
        torch.stack(draft_rows, 1).float(),
        torch.stack(target_rows, 1).float(),
        torch.rand(num, K, generator=generator, dtype=torch.float64),
        lambda weights: torch.multinomial(weights, 1, generator=generator).squeeze(-1),
    )

    sequences = torch.zeros(num, K + 1, dtype=torch.int64)
    for m in range(K + 1):
        continued = torch.multinomial(_rows(target, sequences[:, :m]), 1, generator=generator)
        sequences[:, m] = torch.where(
            m < num_accepted.long(), tokens[:, m].long(), continued.squeeze(-1)
        )
    grid = torch.cartesian_prod(*[torch.arange(VOCAB)] * (K + 1))
    expected = torch.full((grid.shape[0],), float(num), dtype=torch.float64)
    for m in range(K + 1):
        expected *= _rows(target, grid[:, :m]).gather(-1, grid[:, m : m + 1]).squeeze(-1)
    ids = (sequences * VOCAB ** torch.arange(K, -1, -1)).sum(-1)
    observed = torch.bincount(ids, minlength=grid.shape[0]).double()
    big = expected >= 5  # pool the rare sequences
    observed = torch.cat([observed[big], observed[~big].sum().view(1)])
    expected = torch.cat([expected[big], expected[~big].sum().view(1)])
    stat = ((observed - expected) ** 2 / expected).sum().item()
    dof = observed.numel() - 1
    critical = dof * (1 - 2 / (9 * dof) + 3.719 * math.sqrt(2 / (9 * dof))) ** 3  # p = 1e-4
    assert stat < critical, f"chi-square {stat:.1f} >= {critical:.1f}"


@pytest.mark.parametrize(
    "configured, drawn",
    [
        (AdvancedSamplingMode.FULL, AdvancedSamplingMode.FULL),
        (AdvancedSamplingMode.NO_TOPK, AdvancedSamplingMode.NO_TOPK_NO_TOPP),
        (AdvancedSamplingMode.NO_TOPP, AdvancedSamplingMode.NO_TOPK_NO_TOPP),
        (AdvancedSamplingMode.NO_TOPK_NO_TOPP, AdvancedSamplingMode.NO_TOPK_NO_TOPP),
    ],
)
def test_ancestral_drafts_condition_each_position_on_the_drawn_token(configured, drawn):
    """Position k is drawn, with the temperature only, from base_k plus the Markov bias of the token drawn before it;
    the distribution it was drawn from is stored as its proposal. A non-fused mode draws without its filter kernels;
    the fused mode is kept."""
    generator = torch.Generator().manual_seed(1)
    num_gens, vocab, rank = 4, 32, 8
    drafter = types.SimpleNamespace(
        has_markov_head=True,
        markov_w1=torch.randn(vocab, rank, generator=generator),
        markov_w2=torch.randn(vocab, rank, generator=generator),
    )
    base = torch.randn(num_gens, K, vocab, generator=generator)
    anchors = torch.randint(vocab, (num_gens,), generator=generator)
    meta = types.SimpleNamespace(
        request_temperatures=torch.full((num_gens,), 0.7),
        request_seeds=torch.arange(num_gens, dtype=torch.int64),
        request_offsets=torch.zeros(num_gens, dtype=torch.int64),
        rng_draft_slot=lambda step: K + 1 + step,
        batch_slot_ids=torch.arange(num_gens).flip(0),
        draft_probs=torch.zeros(num_gens, K, vocab),
        draft_probs_last_dim=0,
        dummy_slot_row=0,
        advanced_sampling_mode=configured,
        is_all_greedy_sample=False,
        wants_advanced_draft_sampling=True,
        draft_vocab_size=vocab,
        vocab_size=vocab,
    )
    calls = []

    def sampler(mode, logits, temperatures, top_ks, top_ps, min_ps, *, seed, offset):
        probs = torch.softmax(logits / temperatures[:, None], -1)
        calls.append((mode, logits, probs, top_ks, top_ps, min_ps))
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1).int(), probs

    worker = DSparkWorker.__new__(DSparkWorker)
    torch.nn.Module.__init__(worker)
    worker.mapping = None
    assert not worker._samples_block_ancestrally(drafter, meta)  # off by default
    worker.enable_ancestral_draft_sampling()
    assert worker._samples_block_ancestrally(drafter, meta)
    with patch.object(SpecWorkerBase, "_sample_from_logits_with_probs", staticmethod(sampler)):
        tokens, stored_vocab = worker._sample_block_ancestrally(
            drafter, base, {"first_prev_tokens": anchors}, meta, 0, num_gens
        )

    assert tokens.dtype == torch.int32 and stored_vocab == vocab == meta.draft_probs_last_dim
    prev = anchors
    for step, (mode, logits, probs, top_ks, top_ps, min_ps) in enumerate(calls):
        assert mode is drawn
        bias = dspark_markov_step_bias(prev, drafter.markov_w1, drafter.markov_w2)
        torch.testing.assert_close(logits, base[:, step] + bias)
        assert (
            (top_ks == torch.iinfo(torch.int32).max).all()
            and (top_ps == 1).all()
            and (min_ps == 0).all()
        )
        torch.testing.assert_close(meta.draft_probs[meta.batch_slot_ids, step], probs)
        prev = tokens[:, step]
    assert len(calls) == K


def test_ancestral_drafting_is_an_opt_in(monkeypatch):
    """Off by default (the block is refined along the greedy chain); a model turns it on, or installs an ancestral
    override; never for a drafter without a Markov head or a step that does not sample its drafts."""
    from tensorrt_llm._torch.speculative import dspark

    monkeypatch.setattr(dspark, "_ANCESTRAL_BLOCK_OVERRIDE", None)
    monkeypatch.setattr(dspark, "_ANCESTRAL_GLUE_OVERRIDE", None)
    markov = types.SimpleNamespace(has_markov_head=True)
    sampled = types.SimpleNamespace(wants_advanced_draft_sampling=True)
    worker = DSparkWorker.__new__(DSparkWorker)
    torch.nn.Module.__init__(worker)
    assert not worker._samples_block_ancestrally(markov, sampled)
    dspark.set_ancestral_block_override(lambda *args: None)
    assert worker._samples_block_ancestrally(markov, sampled)
    dspark.set_ancestral_block_override(None)
    dspark.set_ancestral_glue_override(object())
    assert worker._samples_block_ancestrally(markov, sampled)
    dspark.set_ancestral_glue_override(None)
    worker.enable_ancestral_draft_sampling()
    assert worker._samples_block_ancestrally(markov, sampled)
    assert not worker._samples_block_ancestrally(
        types.SimpleNamespace(has_markov_head=False), sampled
    )
    assert not worker._samples_block_ancestrally(
        markov, types.SimpleNamespace(wants_advanced_draft_sampling=False)
    )
    worker.enable_ancestral_draft_sampling(False)
    assert not worker._samples_block_ancestrally(markov, sampled)
