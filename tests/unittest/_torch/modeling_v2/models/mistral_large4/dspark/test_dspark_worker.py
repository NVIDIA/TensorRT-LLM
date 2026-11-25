# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral Large 4 DSpark worker hands the drafter its block embedding, the step's
attention metadata and the block's draft hidden states, and never drafts through the
engine's vanilla Markov chains."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.mistral_large4.dspark.worker import (
    MistralLarge4DSparkWorker,
)

pytestmark = pytest.mark.cpu_only


def _worker():
    return object.__new__(MistralLarge4DSparkWorker)


def test_the_drafter_embeds_its_block():
    bonus = torch.tensor([2, 5])
    block = torch.randn(2, 3, 4)
    calls = []

    def embed_draft_block(anchors, block_size):
        calls.append((anchors, block_size))
        return block

    # No token embedding: the worker must not read one for this drafter.
    draft = SimpleNamespace(embed_draft_block=embed_draft_block)
    assert _worker()._draft_block_embedding(draft, bonus, 1, 3, torch.zeros(4)) is block
    assert len(calls) == 1 and calls[0][0] is bonus and calls[0][1] == 3


def test_the_block_forward_reads_the_step_attention_metadata():
    attn_metadata = object()
    out = torch.empty(0)
    seen = {}

    def dflash_forward_with_metadata(**kwargs):
        seen.update(kwargs)
        return out

    draft = SimpleNamespace(dflash_forward_with_metadata=dflash_forward_with_metadata)
    actual = _worker()._draft_block_forward(draft, attn_metadata, noise_embedding=1)
    assert actual is out
    assert seen == {"attn_metadata": attn_metadata, "noise_embedding": 1}


def test_the_fused_paged_latent_preparation_is_not_used():
    assert _worker()._uses_fused_paged_latent_prepare(SimpleNamespace(), 1) is False


@pytest.mark.parametrize("with_hidden_states", [False, True])
def test_the_recurrent_markov_head_refines_over_the_draft_hidden_states(
    monkeypatch, with_hidden_states
):
    draft_hidden_flat = torch.randn(6, 4) if with_hidden_states else None
    logits = torch.randn(2, 3, 8)
    anchors = torch.tensor([0, 1])
    seen = {}

    def head(base, first, *, hidden_states):
        assert base is logits and first is anchors
        seen["hidden_states"] = hidden_states
        return base + 1

    def sample(refined, spec_metadata, batch_size, *, num_contexts):
        seen["refined"] = refined
        seen["sample_args"] = (spec_metadata, batch_size, num_contexts)
        return torch.zeros(2, 3, dtype=torch.int32)

    worker = _worker()
    # The DSpark ancestral and fused chains compute the vanilla Markov head only.
    worker.spec_config = SimpleNamespace(use_ancestral_draft_sampling=True)
    monkeypatch.setattr(worker, "sample_draft_tokens", sample)
    draft = SimpleNamespace(has_markov_head=True, apply_markov_chain_logits=head)
    metadata = SimpleNamespace(advanced_sampling_mode=None, wants_advanced_draft_sampling=True)
    tokens, vocab = worker._sample_block_draft_tokens(
        draft, logits, draft_hidden_flat, {"first_prev_tokens": anchors}, metadata, 0, 2
    )
    assert tokens.shape == (2, 3) and vocab == 8
    torch.testing.assert_close(seen["refined"], logits + 1)
    assert seen["sample_args"] == (metadata, 2, 0)
    if with_hidden_states:
        torch.testing.assert_close(seen["hidden_states"], draft_hidden_flat.reshape(2, 3, 4))
    else:
        assert seen["hidden_states"] is None


def test_vocab_shard_drafting_keeps_the_gathered_block_logits():
    # The shard chain computes the vanilla Markov head only.
    assert _worker()._vocab_parallel_draft_tokens(SimpleNamespace(has_markov_head=True)) is None
