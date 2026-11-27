# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-hot draft-probability rows for requests whose draft distribution this step did not produce.

* ``write_padding_onehot_draft_probs`` / ``write_context_onehot_draft_probs`` equal an advanced-index write; with the
  ``context_onehot_index_fill`` opt-in they issue no host sync.
* A DFlash mixed step's context rows span the published (full-vocab) width, not a vocab-sharded drafter's shard.
* A disaggregated generation request verifies dummy tokens against the one-hot row under rejection sampling.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.dflash import _context_onehot_vocab
from tensorrt_llm._torch.speculative.interface import SpecMetadata, SpecWorkerBase
from tensorrt_llm._torch.speculative.utils import disagg_gen_first_draft_tokens

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

NUM_SLOTS, DRAFT_LEN, VOCAB = 9, 4, 96
# A slot may repeat: one row written twice with the same values.
SLOT_IDS = [5, 1, 7, 1]


def _padding_case(last_dim, index_fill=True):
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


@needs_cuda
@pytest.mark.parametrize("last_dim", [0, 64])
def test_padding_onehot_matches_indexed_write_without_sync(last_dim):
    metadata, expected = _padding_case(last_dim)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        SpecMetadata.write_padding_onehot_draft_probs(metadata, SLOT_IDS, DRAFT_LEN)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert torch.equal(metadata.draft_probs, expected)


@needs_cuda
@pytest.mark.parametrize("index_fill", [True, False])
def test_the_opt_in_alone_selects_index_fill(monkeypatch, index_fill):
    metadata, expected = _padding_case(64, index_fill=index_fill)
    calls = []
    index_fill_ = torch.Tensor.index_fill_

    def spy(self, *args):
        calls.append(args)
        return index_fill_(self, *args)

    monkeypatch.setattr(torch.Tensor, "index_fill_", spy)
    SpecMetadata.write_padding_onehot_draft_probs(metadata, SLOT_IDS, DRAFT_LEN)
    assert torch.equal(metadata.draft_probs, expected)
    assert (len(calls) > 0) is index_fill


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "opted_in, killed, enabled", [(False, False, False), (True, False, True), (True, True, False)]
)
def test_the_kill_switch_wins_over_the_opt_in(monkeypatch, opted_in, killed, enabled):
    if killed:
        monkeypatch.setenv("TRTLLM_M31_DISABLE_ONEHOT_INDEX_FILL", "1")
    else:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_ONEHOT_INDEX_FILL", raising=False)
    assert interface.onehot_index_fill_enabled(opted_in) is enabled


@needs_cuda
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
        context_onehot_index_fill=True,
    )
    onehot_vocab = gen_vocab if num_gens > 0 else vocab
    expected = draft_probs.clone()
    expected[slot_ids[:num_contexts], :draft_len, :onehot_vocab] = 0.0
    expected[slot_ids[:num_contexts], :draft_len, 0] = 1.0
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        SpecWorkerBase.write_context_onehot_draft_probs(
            None, metadata, num_contexts, num_gens, draft_len, gen_vocab
        )
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert torch.equal(metadata.draft_probs, expected)
    assert metadata.draft_probs_last_dim == (None if num_gens > 0 else vocab)


SHARD, FULL, K, SLOTS = 4, 8, 3, 4


def _stale_metadata(published):
    # Every slot row holds a stale distribution; one context request (slot 2), then one generation request (slot 0).
    return types.SimpleNamespace(
        use_rejection_sampling=True,
        draft_probs=torch.full((SLOTS, K, FULL), 0.5),
        batch_slot_ids=torch.tensor([2, 0]),
        draft_probs_last_dim=published,
        draft_probs_vocab_size=FULL,
        dummy_slot_row=SLOTS - 1,
        num_seq_slots=SLOTS - 1,
        max_num_requests=SLOTS - 1,
        is_cuda_graph=False,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "published, num_gens, expected", [(FULL, 1, FULL), (0, 1, SHARD), (FULL, 0, None)]
)
def test_context_onehot_width(published, num_gens, expected):
    """The published width; the logits' before any scatter; no generation request leaves it to the writer."""
    md = _stale_metadata(published)
    assert _context_onehot_vocab(md, num_gens=num_gens, draft_logits_vocab=SHARD) == expected


REJ = types.SimpleNamespace(use_rejection_sampling=True)
STRICT = types.SimpleNamespace(use_rejection_sampling=False)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "ctx_drafts, policy, spec, expected",
    [
        (None, REJ, True, ([0] * K, True)),
        ([], REJ, True, ([0] * K, True)),
        ([5, 6, 7], REJ, True, ([0] * K, True)),
        ([5, 6, 7], STRICT, True, ([5, 6, 7], False)),
        ([5], STRICT, True, ([5, 0, 0], False)),
        (None, STRICT, True, ([0] * K, False)),
        (None, REJ, False, ([], False)),
        ([5, 6, 7], REJ, False, ([5, 6, 7], False)),
    ],
)
def test_disagg_gen_first_draft_tokens(ctx_drafts, policy, spec, expected):
    assert disagg_gen_first_draft_tokens(ctx_drafts, policy, spec, K) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "ctx_drafts, policy, expected",
    [
        ([5, 6], REJ, ([5, 6] + [0] * (K - 2), True)),  # greedy keeps the drafts, padded
        (None, REJ, ([0] * K, True)),  # no drafts: dummies
        ([5, 6], STRICT, ([5, 6] + [0] * (K - 2), False)),  # unchanged without rejection
    ],
)
def test_disagg_gen_first_draft_tokens_keeps_greedy_drafts(ctx_drafts, policy, expected):
    assert disagg_gen_first_draft_tokens(ctx_drafts, policy, True, K, is_greedy=True) == expected


@pytest.mark.cpu_only
def test_disagg_flagged_slot_gets_a_one_hot_row_at_the_dummy():
    md = _stale_metadata(FULL)
    tokens, needs_onehot = disagg_gen_first_draft_tokens(None, REJ, True, K)
    assert needs_onehot
    SpecMetadata.write_padding_onehot_draft_probs(md, [2], K)
    expected = torch.zeros(K, FULL)
    expected[:, tokens[0]] = 1.0
    assert torch.equal(md.draft_probs[2], expected)
    assert torch.equal(md.draft_probs[1], torch.full((K, FULL), 0.5))
