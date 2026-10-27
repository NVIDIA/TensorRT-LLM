# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU unit tests for the context one-hot draft-prob rows written by ``DFlashWorker._forward_impl``.

The worker's own ``_forward_impl`` and ``SpecWorkerBase.write_context_onehot_draft_probs`` run unchanged; every step
around them (KV / block-table bookkeeping, acceptance, the drafter forward and its sampling) is replaced on the
instance by a stand-in, and the metadata are ``types.SimpleNamespace`` objects carrying only the attributes those two
functions read. Covered: a step without generation requests (warmup, every disaggregated context step), a mixed step
with a vocab-sharded drafter (the context rows span the published width, not the shard), and a generation-only step.

Requires CUDA because the worker allocates its draft-token tensors on ``device='cuda'``.
"""

import contextlib
import types

import pytest
import torch

from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the worker allocates CUDA tensors"
)

K, SHARD, FULL, SLOTS, HIDDEN = 3, 4, 8, 4, 16
STALE = 0.5


def _worker():
    """A DFlashWorker whose surrounding steps are stand-ins; ``_forward_impl`` itself is the real one."""
    w = DFlashWorker.__new__(DFlashWorker)
    torch.nn.Module.__init__(w)
    w._accept_stats = None
    w._ctx_buf_inited = False
    w._lean_glue = False
    w._ctx_block_tables = None
    w._compute_block_size = K + 1
    w._d2t = None
    w._ctx_len = torch.zeros(SLOTS, dtype=torch.int32)
    w._ctx_len_host = [0] * SLOTS
    w._ctx_base_pages = torch.zeros(SLOTS, dtype=torch.int32)
    w._ctx_base_host = [0] * SLOTS
    w._req_ctx_pos = {}

    def noop(*args, **kwargs):
        return None

    for name in (
        "_lazy_init_ctx_buffers",
        "_refresh_ctx_block_tables",
        "_execute_guided_decoder_if_present",
        "_prepare_attn_metadata_for_dflash",
        "_prepare_kv_for_draft_forward",
        "_store_prefill_context",
        "_record_disagg_draft_ctx_ready",
        "_restore_attn_metadata_from_spec_dec",
        "_apply_kv_rewind_after_draft",
        "_rollback_guided_decoder_after_verify",
        "_restore_ctx_len_device",
        "_restore_ctx_len_host",
    ):
        setattr(w, name, noop)
    w.get_draft_kv_cache_manager = lambda resource_manager: None
    w.draft_kv_cache_context = lambda attn_metadata, manager: contextlib.nullcontext()

    def accept(logits, attn_metadata, spec_metadata):
        n = attn_metadata.num_seqs
        return (
            torch.zeros(n, K + 1, dtype=torch.int32, device="cuda"),
            torch.ones(n, dtype=torch.int32, device="cuda"),
        )

    w.sample_and_accept_draft_tokens = accept
    w.prepare_1st_drafter_inputs = lambda **kwargs: dict.fromkeys(
        (
            "noise_embedding",
            "query_positions",
            "num_ctx_per_req",
            "ctx_k_cache",
            "ctx_v_cache",
            "ctx_cache_batch_idx",
            "ctx_kv_cache",
            "ctx_page_table",
        )
    )
    w._draft_slot_ids = lambda draft_model, num_gens, block_size, k: torch.arange(
        num_gens * k, device="cuda"
    )
    w._refine_block_logits = lambda draft_model, logits, inputs, spec_metadata: logits
    w.sample_draft_tokens = lambda logits, spec_metadata, batch_size, num_contexts: logits.argmax(
        -1
    ).to(torch.int32)
    w._prepare_next_new_tokens = lambda accepted, drafts, batch_indices, batch_size, num_acc: drafts
    return w


def _drafter(num_gens):
    """A vocab-sharded drafter: its block outputs give SHARD-wide logits."""
    return types.SimpleNamespace(
        lm_head=None,
        dflash_forward=lambda **kwargs: torch.zeros(num_gens * (K + 1), HIDDEN, device="cuda"),
        logits_processor=lambda hidden, lm_head, attn_metadata, flag: torch.zeros(
            hidden.shape[0], SHARD, device="cuda"
        ),
    )


def _metadata(num_contexts, num_gens, published):
    attn = types.SimpleNamespace(
        num_seqs=num_contexts + num_gens, num_contexts=num_contexts, kv_cache_manager=None
    )
    # Context requests sit in slots 2, 3; generation requests in slots 0, 1. Every row starts stale.
    slots = [2, 3][:num_contexts] + [0, 1][:num_gens]
    spec = types.SimpleNamespace(
        runtime_draft_len=K,
        is_cuda_graph=False,
        request_ids=None,
        batch_indices_cuda=None,
        use_rejection_sampling=True,
        draft_probs=torch.full((SLOTS, K, FULL), STALE, device="cuda"),
        batch_slot_ids=torch.tensor(slots, device="cuda"),
        draft_probs_last_dim=published,
        draft_probs_vocab_size=FULL,
    )
    return attn, spec


def _run(num_contexts, num_gens, published=FULL):
    attn, spec = _metadata(num_contexts, num_gens, published)
    n = num_contexts + num_gens
    out = _worker()._forward_impl(
        input_ids=torch.zeros(n, dtype=torch.int32, device="cuda"),
        position_ids=torch.zeros(1, n, dtype=torch.int32, device="cuda"),
        hidden_states=torch.zeros(n, HIDDEN, device="cuda"),
        logits=torch.zeros(n, FULL, device="cuda"),
        attn_metadata=attn,
        spec_metadata=spec,
        draft_model=_drafter(num_gens),
    )
    return out, spec


def _onehot():
    row = torch.zeros(K, FULL, device="cuda")
    row[:, 0] = 1.0
    return row


@pytest.fixture(params=[True, False], ids=["index_fill", "put"])
def onehot_write(request, monkeypatch):
    monkeypatch.setattr(interface, "_ONEHOT_INDEX_FILL_ARMED", True)
    monkeypatch.setattr(interface, "_ONEHOT_INDEX_FILL", request.param)


def test_step_without_generation_requests(onehot_write):
    out, spec = _run(num_contexts=2, num_gens=0, published=0)
    assert torch.equal(
        out["next_draft_tokens"], torch.zeros(2, K, dtype=torch.int32, device="cuda")
    )
    # The rows span the buffer's width, which is published for next iteration's acceptance.
    assert spec.draft_probs_last_dim == FULL
    for slot in (2, 3):
        assert torch.equal(spec.draft_probs[slot], _onehot())
    for slot in (0, 1):
        assert torch.equal(spec.draft_probs[slot], torch.full((K, FULL), STALE, device="cuda"))


def test_mixed_step_rows_span_the_published_width(onehot_write):
    out, spec = _run(num_contexts=1, num_gens=1, published=FULL)
    assert out["next_draft_tokens"].shape == (2, K)
    # Past the drafter's SHARD-wide logits: every column acceptance reads is one-hot, none stale.
    assert torch.equal(spec.draft_probs[2], _onehot())
    assert spec.draft_probs_last_dim == FULL
    # Rows of slots without a context request are left to their own writers.
    for slot in (0, 1, 3):
        assert torch.equal(spec.draft_probs[slot], torch.full((K, FULL), STALE, device="cuda"))


def test_generation_only_step_writes_no_context_row(onehot_write):
    out, spec = _run(num_contexts=0, num_gens=2, published=FULL)
    assert out["next_draft_tokens"].shape == (2, K)
    assert torch.equal(spec.draft_probs, torch.full((SLOTS, K, FULL), STALE, device="cuda"))
