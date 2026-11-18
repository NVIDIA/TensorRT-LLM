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
"""``TRTLLM_RS_COMPACT_DRAFT_PROBS`` on CUDA: FlashInfer's chain speculative sampling on the
rebuilt rows, inside a CUDA graph, and the verify's peak memory, at the serving shapes
(97 slot rows, K = 7, top_k = 16, vocab 202,048, 96 generation requests).
"""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative import compact_draft_probs as cdp
from tensorrt_llm._torch.speculative.interface import (
    SpecMetadata,
    SpeculativeDecodingMode,
    rejection_sampling_one_model,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SLOTS, K, TOP_K, VOCAB, REQUESTS = 96, 7, 16, 202_048, 96


def _rows(gen, requests, steps=K):
    scores = torch.rand((requests, steps, VOCAB), generator=gen, device="cuda")
    ids = scores.topk(TOP_K, dim=-1).indices
    proposal = torch.softmax(
        torch.randn((requests, steps, TOP_K), generator=gen, device="cuda") * 3.0, dim=-1
    )
    return ids, proposal


def _stores(seed):
    """A dense and a compact store after the same sampled and one-hot writes."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dense = torch.empty((SLOTS + 1, K, VOCAB), dtype=torch.float32, device="cuda")
    compact = cdp.CompactDraftProbs(SLOTS + 1, K, VOCAB, device="cuda")
    slots = torch.randperm(SLOTS, device="cuda", generator=gen)
    onehot, sampled = slots[:8], slots[8:]
    dense[:, :K, :VOCAB].index_fill_(0, onehot, 0.0)
    dense[:, :K, 0].index_fill_(0, onehot, 1.0)
    compact.write_onehot_index_fill(onehot, K, VOCAB)
    ids, proposal = _rows(gen, sampled.shape[0])
    dense[sampled, :K, :VOCAB] = dense.new_zeros(())
    steps = torch.arange(K, device="cuda")
    dense[sampled[:, None, None], steps[:, None], ids] = proposal
    compact.write_rows(sampled, ids, proposal, VOCAB)
    return gen, dense, compact, slots[:REQUESTS]


def _target(gen, dense, order):
    base = torch.log(dense[order, :K].clamp_min(1e-6))
    bonus = torch.randn((order.shape[0], 1, VOCAB), generator=gen, device="cuda")
    logits = torch.cat([base, bonus], 1) + 0.7 * torch.randn(
        (order.shape[0], K + 1, VOCAB), generator=gen, device="cuda"
    )
    return torch.softmax(logits, dim=-1)


def _draft_tokens(gen, dense, order):
    flat = dense[order, :K].reshape(-1, VOCAB).clamp_min(0.0)
    return torch.multinomial(flat, 1, generator=gen).reshape(-1, K).to(torch.int32)


@pytest.mark.parametrize("seed", [0, 1])
def test_chain_sampling_on_rebuilt_rows_is_the_dense_result(seed):
    gen, dense, compact, order = _stores(seed)
    rebuilt = compact.densify(order, K, VOCAB)
    assert torch.equal(rebuilt.view(torch.int32), dense[order, :K].contiguous().view(torch.int32))
    target = _target(gen, dense, order)
    tokens = _draft_tokens(gen, dense, order)
    seeds = torch.arange(REQUESTS, device="cuda", dtype=torch.int64) * 7919 + 13
    offsets = torch.full_like(seeds, 7)
    a_d, n_d = rejection_sampling_one_model(
        dense[order, :K], tokens, target, deterministic=True, seed=seeds, offset=offsets
    )
    a_c, n_c = rejection_sampling_one_model(
        rebuilt, tokens, target, deterministic=True, seed=seeds, offset=offsets
    )
    assert torch.equal(a_d, a_c) and torch.equal(n_d, n_c)


def test_writes_and_rebuild_replay_in_a_cuda_graph():
    gen, dense, compact, order = _stores(3)
    ids, proposal = _rows(gen, REQUESTS)
    scratch = cdp.mark_private_verify_logits(
        torch.empty((REQUESTS * (K + 1), VOCAB), dtype=torch.float32, device="cuda")
    )
    deferred = cdp.DeferredDraftProbs(compact, order, K, VOCAB)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        compact.write_rows(order, ids, proposal, VOCAB)
        compact.write_onehot_index_fill(order[:4], K, VOCAB)
        rows = deferred.materialize(cdp.private_verify_scratch(scratch))
    graph.replay()
    torch.cuda.synchronize()
    eager = cdp.CompactDraftProbs(SLOTS + 1, K, VOCAB, device="cuda")
    eager.write_rows(order, ids, proposal, VOCAB)
    eager.write_onehot_index_fill(order[:4], K, VOCAB)
    assert rows.untyped_storage().data_ptr() == scratch.untyped_storage().data_ptr()
    assert torch.equal(rows, eager.densify(order, K, VOCAB))


def test_the_verify_rows_add_no_allocation_over_private_logits():
    gen, dense, compact, order = _stores(4)
    logits = cdp.mark_private_verify_logits(
        torch.randn((REQUESTS * (K + 1), VOCAB), device="cuda", generator=gen)
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    gathered = dense[order, :K]
    dense_peak = torch.cuda.max_memory_allocated() - before
    del gathered
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    rows = cdp.DeferredDraftProbs(compact, order, K, VOCAB).materialize(
        cdp.private_verify_scratch(logits)
    )
    compact_peak = torch.cuda.max_memory_allocated() - before
    q_bytes = REQUESTS * K * VOCAB * 4
    assert dense_peak >= q_bytes
    assert compact_peak < q_bytes // 16  # the gathered ids / values only
    assert rows.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr()


def _alloc_meta(**over):
    base = dict(
        use_rejection_sampling=True,
        draft_probs=None,
        compact_draft_probs=None,
        vocab_size=VOCAB,
        draft_vocab_size=VOCAB,
        max_num_requests=SLOTS,
        max_draft_len=K,
        num_seq_slots=0,
        batch_slot_ids=None,
        full_draft_probs=None,
        draft_probs_vocab_size=0,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_prepare_allocates_the_compact_store_only_under_the_knob(monkeypatch):
    monkeypatch.delenv(cdp.ENV, raising=False)
    meta = _alloc_meta()
    SpecMetadata.prepare_rejection_sampling_buffers(meta)
    assert meta.draft_probs is not None and meta.compact_draft_probs is None

    monkeypatch.setenv(cdp.ENV, "1")
    meta = _alloc_meta()
    SpecMetadata.prepare_rejection_sampling_buffers(meta)
    assert meta.draft_probs is None and meta.compact_draft_probs is not None
    assert meta.compact_draft_probs.shape == (SLOTS + 1, K, VOCAB)
    assert meta.dummy_slot_row == SLOTS and meta.draft_probs_vocab_size == VOCAB

    for refused in (
        _alloc_meta(spec_dec_mode=SpeculativeDecodingMode.MTP),
        _alloc_meta(draft_vocab_size=32_000),
    ):
        SpecMetadata.prepare_rejection_sampling_buffers(refused)
        assert refused.draft_probs is not None and refused.compact_draft_probs is None
