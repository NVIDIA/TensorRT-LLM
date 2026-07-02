# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_TARGET_ONE_MODEL rejection-sampling runtime validation.

DRAFT_TARGET uses a chained per-step draft loop (like MTP): each step calls the
public ``sample_draft(..., draft_step=i)`` which samples and scatters the exact
proposal row into ``draft_probs[slot, step]``, and acceptance routes through the
public ``compare_and_accept``. These tests prove the per-step proposal-distribution
identity at runtime plus the standard controls:

  - non-greedy: public ``sample_draft`` per step, per-step provenance (stored row
    == sampler's exact probs), public ``compare_and_accept`` + rejection dispatch
    + kernel; all-greedy ``top_k=1`` bypasses the kernel;
  - rejection-disabled: strict base path, no kernel;
  - one-shot forced-invalid: fail-closed (kernel delta 0 + validity cleared) then
    same-worker recovery.

Requires the Llama-3.1-8B target and Llama-3.2-1B draft (shared Llama-3 vocab,
identity d2t) under ``llm_models_root()``.
"""

import os
import sys

import pytest
import torch

import tensorrt_llm._torch.speculative.interface as iface
import tensorrt_llm._torch.speculative.one_model_sampler as oms
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.llmapi import DraftTargetDecodingConfig, KvCacheConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root


def _paths():
    root = llm_models_root()
    if root is None:
        return None, None
    target = os.path.join(root, "llama-3.1-model", "Llama-3.1-8B-Instruct")
    draft = os.path.join(root, "llama-3.2-models", "Llama-3.2-1B-Instruct")
    if not (os.path.isdir(target) and os.path.isdir(draft)):
        return None, None
    return target, draft


def _skip_or_paths():
    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 45:
        pytest.skip("Not enough GPU memory for Llama-3.1-8B + Llama-3.2-1B draft")
    target, draft = _paths()
    if target is None:
        pytest.skip("Llama-3.1-8B / Llama-3.2-1B draft not available")
    return target, draft


def _build(target, draft, use_rejection):
    spec = DraftTargetDecodingConfig(
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


def _zero():
    return {
        "kernel": 0,
        "compare_and_accept": 0,
        "sample_draft_step": 0,
        "dispatch": 0,
        "base": 0,
        "prov_checked": 0,
        "prov_mismatch": 0,
    }


def _install(monkeypatch, state, provenance=False):
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

    holder = {"probs": None}
    if provenance:
        orig_sampler = iface.sampling_batch_spec_dec_one_model_for_rejection

        def _sampler(*a, **k):
            flat_tokens, flat_probs = orig_sampler(*a, **k)
            holder["probs"] = flat_probs
            return flat_tokens, flat_probs

        monkeypatch.setattr(iface, "sampling_batch_spec_dec_one_model_for_rejection", _sampler)

    orig_sd = SpecWorkerBase.sample_draft

    def _sd(self, logits, spec_metadata, batch_size, d2t=None, draft_step=None):
        if provenance:
            holder["probs"] = None
        out = orig_sd(self, logits, spec_metadata, batch_size, d2t, draft_step)
        if draft_step is not None:
            state["sample_draft_step"] += 1
            # Per-step provenance: the stored row must equal the sampler's exact
            # probs for the same (slot, step).
            if (
                provenance
                and holder["probs"] is not None
                and not spec_metadata.is_all_greedy_sample
            ):
                vocab = holder["probs"].shape[-1]
                slots = spec_metadata.batch_slot_ids[:batch_size]
                stored = spec_metadata.draft_probs[slots, draft_step, :vocab]
                state["prov_checked"] += 1
                if not torch.equal(stored, holder["probs"]):
                    state["prov_mismatch"] += 1
        return out

    monkeypatch.setattr(SpecWorkerBase, "sample_draft", _sd)


_PROMPT = "The capital of France is"


@pytest.mark.high_cuda_memory
def test_draft_target_per_step_provenance_and_greedy(monkeypatch):
    target, draft = _skip_or_paths()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    state = _zero()
    _install(monkeypatch, state, provenance=True)

    llm = _build(target, draft, use_rejection=True)
    try:
        sp = SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
        t1 = llm.generate([_PROMPT], sp)[0].outputs[0].text
        ng = dict(state)
        t2 = llm.generate([_PROMPT], SamplingParams(max_tokens=32, top_k=1))[0].outputs[0].text
    finally:
        llm.shutdown()

    assert t1 and t2, "generation produced empty output"
    assert ng["sample_draft_step"] > 0, (
        "DRAFT_TARGET did not capture per-step drafts via public sample_draft"
    )
    assert ng["prov_checked"] > 0, "per-step provenance was never checked"
    assert ng["prov_mismatch"] == 0, (
        "stored draft_probs[slot, step] != sampler's exact probs (provenance)"
    )
    assert ng["compare_and_accept"] > 0, (
        "DRAFT_TARGET did not route acceptance through compare_and_accept"
    )
    assert ng["dispatch"] > 0 and ng["kernel"] > 0, (
        "DRAFT_TARGET non-greedy rejection dispatch/kernel did not engage"
    )
    assert state["compare_and_accept"] > ng["compare_and_accept"], (
        "all-greedy phase did not reach compare_and_accept"
    )
    assert state["kernel"] == ng["kernel"], (
        "all-greedy phase invoked the rejection kernel (must bypass)"
    )


@pytest.mark.high_cuda_memory
def test_draft_target_rejection_disabled_uses_base(monkeypatch):
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
        "rejection-disabled DRAFT_TARGET bypassed compare_and_accept"
    )
    assert state["base"] > 0, "rejection-disabled DRAFT_TARGET did not use the strict base path"
    assert state["kernel"] == 0, "rejection kernel invoked with rejection disabled"


@pytest.mark.high_cuda_memory
def test_draft_target_fail_closed_recovery(monkeypatch):
    target, draft = _skip_or_paths()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    state = _zero()
    _install(monkeypatch, state)

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
    assert fc["done"], "no gen-bearing acceptance was force-invalidated"
    assert fc["kernel_delta"] == 0, "rejection kernel ran during the forced-invalid iteration"
    assert fc["validity_after"] is False, "draft_probs_valid not cleared after fail-closed fallback"
    assert state["dispatch"] > 0 and state["kernel"] > 0, (
        "rejection did not recover on a later acceptance after one-shot failure"
    )
