# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The optional draft-model methods the DFlash and DSpark workers call.

A drafter may build its own input block (``embed_draft_block``), take the
step's attention metadata in its block forward
(``dflash_forward_with_metadata``), and chain a recurrent Markov head over its
draft hidden states (``has_recurrent_markov_head``). These tests pin how the
workers dispatch to them, and that a drafter without them keeps the default
behavior.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import (
    dflash_block_forward,
    dflash_draft_block_embedding,
    dflash_noise_block_embedding,
)
from tensorrt_llm._torch.speculative.dspark import DSparkWorker

pytestmark = pytest.mark.cpu_only


def _embedding_drafter(vocab=6, hidden=4):
    embed_tokens = torch.nn.Embedding(vocab, hidden)
    return SimpleNamespace(
        draft_model_full=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed_tokens))
    )


def test_block_embedding_defaults_to_the_token_embedding():
    draft = _embedding_drafter()
    bonus = torch.tensor([2, 5])
    actual = dflash_draft_block_embedding(draft, bonus, 1, 3)
    expected = dflash_noise_block_embedding(draft.draft_model_full.model.embed_tokens, bonus, 1, 3)
    torch.testing.assert_close(actual, expected)


def test_drafter_builds_its_own_block_embedding():
    bonus = torch.tensor([2, 5])
    block = torch.randn(2, 3, 4)
    calls = []

    def embed_draft_block(anchors, block_size):
        calls.append((anchors, block_size))
        return block

    # No token embedding: the worker must not read it for such a drafter.
    draft = SimpleNamespace(embed_draft_block=embed_draft_block)
    actual = dflash_draft_block_embedding(draft, bonus, 1, 3, torch.zeros(4))
    assert actual is block
    assert len(calls) == 1 and calls[0][0] is bonus and calls[0][1] == 3


@pytest.mark.parametrize("with_metadata", [False, True])
def test_block_forward_passes_attention_metadata_only_on_request(with_metadata):
    attn_metadata = object()
    out = torch.empty(0)
    seen = {}

    def dflash_forward(**kwargs):
        seen["dflash_forward"] = kwargs
        return out

    def dflash_forward_with_metadata(**kwargs):
        seen["dflash_forward_with_metadata"] = kwargs
        return out

    draft = SimpleNamespace(dflash_forward=dflash_forward)
    if with_metadata:
        draft.dflash_forward_with_metadata = dflash_forward_with_metadata
    actual = dflash_block_forward(draft, attn_metadata, noise_embedding=1, query_positions=2)
    assert actual is out
    if with_metadata:
        assert seen == {
            "dflash_forward_with_metadata": {
                "attn_metadata": attn_metadata,
                "noise_embedding": 1,
                "query_positions": 2,
            }
        }
    else:
        assert seen == {"dflash_forward": {"noise_embedding": 1, "query_positions": 2}}


@pytest.mark.parametrize("recurrent", [False, True])
def test_only_a_recurrent_markov_head_receives_the_draft_hidden_states(recurrent):
    hidden = torch.randn(2, 3, 4)
    logits = torch.randn(2, 3, 8)
    anchors = torch.tensor([0, 1])
    worker = SimpleNamespace(_markov_tp_exchange=None)
    worker._apply_dspark_markov_bias = MethodType(DSparkWorker._apply_dspark_markov_bias, worker)

    def head(base, first, **kwargs):
        assert base is logits and first is anchors
        if recurrent:
            assert kwargs.keys() == {"hidden_states"} and kwargs["hidden_states"] is hidden
        else:
            assert not kwargs
        return base + 1

    draft = SimpleNamespace(
        has_markov_head=True,
        has_recurrent_markov_head=recurrent,
        markov_w2=torch.empty(8, 2),
        apply_markov_chain_logits=head,
    )
    actual = DSparkWorker._refine_block_logits(
        worker, draft, logits, {"first_prev_tokens": anchors}, object(), hidden
    )
    torch.testing.assert_close(actual, logits + 1)


@pytest.mark.parametrize("recurrent", [False, True])
def test_vocab_shard_drafting_declines_a_recurrent_markov_head(recurrent):
    # The shard chain computes the vanilla Markov head only, so a recurrent
    # head must keep the gathered block logits and its own chain.
    lm_head = object()
    worker = object.__new__(DSparkWorker)
    worker._vocab_parallel = SimpleNamespace(head=SimpleNamespace(lm_head=lm_head))
    worker._d2t = None
    draft = SimpleNamespace(
        lm_head=lm_head,
        has_candidate_selector=False,
        has_markov_head=True,
        has_recurrent_markov_head=recurrent,
    )
    draft_tokens = worker._vocab_parallel_draft_tokens(draft)
    if recurrent:
        assert draft_tokens is None
    else:
        assert draft_tokens == worker._markov_block_tokens
