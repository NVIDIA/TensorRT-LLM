# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PARD block-capture rejection-sampling runtime validation.

PARD predicts all K draft tokens in one forward (parallel/block capture) using
mask tokens. The PARD worker subclasses ``SpecWorkerBase`` and captures drafts
through the shared public ``sample_draft_block`` (scatters the per-step proposal
rows into ``draft_probs[gen_slot_ids, :K]``) and accepts through the shared public
``compare_and_accept``. These tests prove the block-capture rejection path at
runtime, including the token/prob order/provenance:

  - non-greedy: ``sample_draft_block`` block capture + public ``compare_and_accept``
    + rejection dispatch + kernel all engage, and the scattered
    ``draft_probs[gen_slot_ids, :K, :vocab]`` equals the sampler's
    ``flat_probs.reshape(num_gens, K, vocab)`` (returned tokens equal
    ``flat_tokens.reshape`` for identity d2t);
  - all-greedy: ``compare_and_accept`` reached but the shared bypass keeps the
    kernel uninvoked;
  - rejection-disabled: ``compare_and_accept`` falls through to the strict base
    path, no rejection kernel.

Requires a GPU with enough memory plus the Llama-3.2-1B target and the
PARD-Llama-3.2-1B draft checkpoint under ``llm_models_root()``.
"""

import os
import sys

import pytest
import torch

import tensorrt_llm._torch.speculative.interface as iface
import tensorrt_llm._torch.speculative.one_model_sampler as oms
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.llmapi import KvCacheConfig, PARDDecodingConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root


def _paths():
    root = llm_models_root()
    if root is None:
        return None, None
    target = os.path.join(root, "llama-3.2-models", "Llama-3.2-1B-Instruct")
    draft = os.path.join(root, "PARD-Llama-3.2-1B")
    if not (os.path.isdir(target) and os.path.isdir(draft)):
        return None, None
    return target, draft


def _skip_or_paths():
    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough GPU memory for Llama-3.2-1B + PARD draft")
    target, draft = _paths()
    if target is None:
        pytest.skip("Llama-3.2-1B / PARD-Llama-3.2-1B draft not available")
    return target, draft


def _build(target, draft, use_rejection):
    spec = PARDDecodingConfig(
        max_draft_len=4,
        speculative_model=draft,
        use_rejection_sampling=use_rejection,
    )
    return LLM(
        model=target,
        speculative_config=spec,
        max_batch_size=1,
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        max_seq_len=8192,
        kv_cache_config=KvCacheConfig(max_tokens=8192),
    )


def _install(monkeypatch, state):
    orig_kernel = oms.rejection_sampling_one_model

    def _kernel(*a, **k):
        state["kernel"] += 1
        return orig_kernel(*a, **k)

    monkeypatch.setattr(oms, "rejection_sampling_one_model", _kernel)
    monkeypatch.setattr(iface, "rejection_sampling_one_model", _kernel)

    orig_cmp = SpecWorkerBase.compare_and_accept

    def _cmp(self, *a, **k):
        state["compare_and_accept"] += 1
        return orig_cmp(self, *a, **k)

    monkeypatch.setattr(SpecWorkerBase, "compare_and_accept", _cmp)

    holder = {"tokens": None, "probs": None}
    orig_sampler = iface.sampling_batch_spec_dec_one_model_for_rejection

    def _sampler(*a, **k):
        flat_tokens, flat_probs = orig_sampler(*a, **k)
        holder["tokens"] = flat_tokens
        holder["probs"] = flat_probs
        return flat_tokens, flat_probs

    monkeypatch.setattr(iface, "sampling_batch_spec_dec_one_model_for_rejection", _sampler)

    orig_blk = SpecWorkerBase.sample_draft_block

    def _blk(self, gen_logits, spec_metadata, num_contexts, batch_size, d2t=None):
        state["block"] += 1
        holder["tokens"] = None
        holder["probs"] = None
        out = orig_blk(self, gen_logits, spec_metadata, num_contexts, batch_size, d2t)
        if holder["probs"] is not None and not spec_metadata.is_all_greedy_sample:
            num_gens, K, vocab = gen_logits.shape
            gen_slot_ids = spec_metadata.batch_slot_ids[num_contexts:batch_size]
            stored = spec_metadata.draft_probs[gen_slot_ids, :K, :vocab]
            expected = holder["probs"].reshape(num_gens, K, vocab)
            state["order_checked"] += 1
            if not torch.equal(stored, expected):
                state["order_mismatch"] += 1
            if d2t is None:
                exp_tok = holder["tokens"].reshape(num_gens, K).type(out.dtype)
                if not torch.equal(out, exp_tok):
                    state["token_mismatch"] += 1
        return out

    monkeypatch.setattr(SpecWorkerBase, "sample_draft_block", _blk)

    orig_disp = SpecWorkerBase._sample_and_accept_draft_tokens_rejection

    def _disp(self, *a, **k):
        state["dispatch"] += 1
        return orig_disp(self, *a, **k)

    monkeypatch.setattr(SpecWorkerBase, "_sample_and_accept_draft_tokens_rejection", _disp)

    orig_base = SpecWorkerBase._sample_and_accept_draft_tokens_base

    def _base(self, *a, **k):
        state["base"] += 1
        return orig_base(self, *a, **k)

    monkeypatch.setattr(SpecWorkerBase, "_sample_and_accept_draft_tokens_base", _base)


def _zero():
    return {
        "kernel": 0,
        "compare_and_accept": 0,
        "block": 0,
        "dispatch": 0,
        "base": 0,
        "order_checked": 0,
        "order_mismatch": 0,
        "token_mismatch": 0,
    }


_PROMPT = "The capital of France is"


@pytest.mark.high_cuda_memory
def test_pard_block_capture_rejection_and_greedy(monkeypatch):
    target, draft = _skip_or_paths()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    state = _zero()
    _install(monkeypatch, state)

    llm = _build(target, draft, use_rejection=True)
    try:
        sp = SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
        t1 = llm.generate([_PROMPT], sp)[0].outputs[0].text
        ng = dict(state)
        t2 = llm.generate([_PROMPT], SamplingParams(max_tokens=32, top_k=1))[0].outputs[0].text
    finally:
        llm.shutdown()

    assert t1 and t2, "generation produced empty output"
    assert ng["block"] > 0, "PARD did not capture drafts via sample_draft_block"
    assert ng["compare_and_accept"] > 0, (
        "PARD did not route acceptance through public compare_and_accept"
    )
    assert ng["dispatch"] > 0 and ng["kernel"] > 0, (
        "PARD non-greedy rejection dispatch/kernel did not engage"
    )
    assert ng["order_checked"] > 0, "block-capture order/provenance was never checked"
    assert ng["order_mismatch"] == 0, (
        "scattered draft_probs did not match the sampler's flat_probs (order bug)"
    )
    assert ng["token_mismatch"] == 0, (
        "returned block draft tokens did not match the sampler's flat_tokens"
    )
    assert state["compare_and_accept"] > ng["compare_and_accept"], (
        "all-greedy phase did not reach public compare_and_accept"
    )
    assert state["kernel"] == ng["kernel"], (
        "all-greedy phase invoked the rejection kernel (must bypass)"
    )


@pytest.mark.high_cuda_memory
def test_pard_rejection_disabled_uses_base(monkeypatch):
    target, draft = _skip_or_paths()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    state = _zero()
    _install(monkeypatch, state)

    llm = _build(target, draft, use_rejection=False)
    try:
        sp = SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
        text = llm.generate([_PROMPT], sp)[0].outputs[0].text
    finally:
        llm.shutdown()

    assert text, "generation produced empty output"
    assert state["compare_and_accept"] > 0, (
        "rejection-disabled PARD bypassed the public compare_and_accept"
    )
    assert state["base"] > 0, "rejection-disabled PARD did not use the strict base path"
    assert state["kernel"] == 0, "rejection kernel invoked with rejection disabled"


@pytest.mark.high_cuda_memory
def test_pard_fail_closed_recovery_and_acceptance_shape(monkeypatch):
    """One-shot forced-invalid fail-closed + recovery, plus PARD acceptance shape.

    Forces _rejection_buffers_valid False once on the first gen-bearing
    acceptance: that iteration must run no rejection kernel and clear
    draft_probs_valid; a later normal acceptance on the same worker must recover
    to the rejection dispatch + kernel. Also asserts the gen-bearing
    compare_and_accept sees draft_tokens.shape == (num_gens, K) and
    logits.shape[0] == num_contexts + num_gens*(K+1) (the PARD 2K -> K+1 reshape).
    """
    target, draft = _skip_or_paths()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    K = 4
    state = _zero()
    _install(monkeypatch, state)

    shape = {"checked": 0, "bad_tokens": 0, "bad_logits": 0}
    orig_cmp = SpecWorkerBase.compare_and_accept

    def _cmp_shape(self, logits, draft_tokens, num_contexts, batch_size, spec_metadata):
        num_gens = batch_size - num_contexts
        if num_gens > 0:
            shape["checked"] += 1
            if tuple(draft_tokens.shape) != (num_gens, K):
                shape["bad_tokens"] += 1
            if logits.shape[0] != num_contexts + num_gens * (K + 1):
                shape["bad_logits"] += 1
        return orig_cmp(self, logits, draft_tokens, num_contexts, batch_size, spec_metadata)

    monkeypatch.setattr(SpecWorkerBase, "compare_and_accept", _cmp_shape)

    fc = {"done": False, "kernel_delta": None, "validity_after": None}
    orig_valid = SpecWorkerBase._rejection_buffers_valid

    def _valid(
        self, draft_tokens, draft_len, stored_vocab, num_contexts, batch_size, logits, spec_metadata
    ):
        if not fc["done"] and (batch_size - num_contexts) > 0:
            return False
        return orig_valid(
            self,
            draft_tokens,
            draft_len,
            stored_vocab,
            num_contexts,
            batch_size,
            logits,
            spec_metadata,
        )

    monkeypatch.setattr(SpecWorkerBase, "_rejection_buffers_valid", _valid)

    orig_accept = SpecWorkerBase._accept_draft_tokens

    def _accept(self, logits, draft_tokens, num_contexts, batch_size, spec_metadata):
        is_forced = (not fc["done"]) and (batch_size - num_contexts) > 0
        k_before = state["kernel"]
        out = orig_accept(self, logits, draft_tokens, num_contexts, batch_size, spec_metadata)
        if is_forced:
            fc["done"] = True
            fc["kernel_delta"] = state["kernel"] - k_before
            fc["validity_after"] = bool(spec_metadata.draft_probs_valid)
        return out

    monkeypatch.setattr(SpecWorkerBase, "_accept_draft_tokens", _accept)

    llm = _build(target, draft, use_rejection=True)
    try:
        sp = SamplingParams(max_tokens=48, temperature=0.8, top_p=0.95, seed=0)
        text = llm.generate([_PROMPT], sp)[0].outputs[0].text
    finally:
        llm.shutdown()

    assert text, "generation produced empty output"
    # PARD acceptance shape (2K -> K+1).
    assert shape["checked"] > 0, "no gen-bearing PARD acceptance observed"
    assert shape["bad_tokens"] == 0, "PARD acceptance draft_tokens shape != (num_gens, K)"
    assert shape["bad_logits"] == 0, "PARD acceptance logits rows != num_contexts + num_gens*(K+1)"
    # One-shot fail-closed clearing + recovery.
    assert fc["done"], "no gen-bearing acceptance was force-invalidated"
    assert fc["kernel_delta"] == 0, "rejection kernel ran during the forced-invalid iteration"
    assert fc["validity_after"] is False, "draft_probs_valid not cleared after fail-closed fallback"
    assert state["dispatch"] > 0 and state["kernel"] > 0, (
        "rejection did not recover on a later acceptance after one-shot failure"
    )


@pytest.mark.high_cuda_memory
def test_pard_context_only_forward_clears_validity(monkeypatch):
    """A context-only PARD forward must clear draft_probs_valid (no stale reuse)."""
    from tensorrt_llm._torch.speculative.pard import PARDWorker

    target, draft = _skip_or_paths()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    state = _zero()
    _install(monkeypatch, state)

    obs = {"ctx_only": 0, "ctx_only_valid": 0, "gen_set_valid": 0}
    orig_fwd = PARDWorker.forward

    def _fwd(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        out = orig_fwd(
            self,
            input_ids,
            position_ids,
            hidden_states,
            logits,
            attn_metadata,
            spec_metadata,
            draft_model,
            resource_manager,
        )
        num_gens = attn_metadata.num_seqs - attn_metadata.num_contexts
        valid = bool(getattr(spec_metadata, "draft_probs_valid", False))
        if num_gens == 0:
            obs["ctx_only"] += 1
            if valid:
                obs["ctx_only_valid"] += 1
        elif valid:
            obs["gen_set_valid"] += 1
        return out

    monkeypatch.setattr(PARDWorker, "forward", _fwd)

    llm = _build(target, draft, use_rejection=True)
    try:
        sp = SamplingParams(max_tokens=24, temperature=0.8, top_p=0.95, seed=0)
        t1 = llm.generate([_PROMPT], sp)[0].outputs[0].text
        t2 = llm.generate(["The president of the United States is"], sp)[0].outputs[0].text
    finally:
        llm.shutdown()

    assert t1 and t2, "generation produced empty output"
    assert obs["ctx_only"] > 0, "no context-only PARD forward observed"
    assert obs["gen_set_valid"] > 0, "no gen forward set draft_probs_valid (vacuous test)"
    assert obs["ctx_only_valid"] == 0, (
        "a context-only forward left draft_probs_valid True (stale-reuse risk)"
    )
    assert state["kernel"] > 0, "rejection kernel never ran"
