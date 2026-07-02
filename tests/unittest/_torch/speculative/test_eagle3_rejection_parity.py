# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eagle3 one-model rejection-sampling parity with vanilla MTP.

Eagle3's ``Eagle3OneModelWorker`` and the vanilla-MTP worker both subclass
``SpecWorkerBase`` and route draft capture through the public ``sample_draft``
and acceptance through the public ``compare_and_accept`` (a thin wrapper over
``_accept_draft_tokens`` with internal rejection routing and a fail-closed
fallback). These tests prove Eagle3 exercises those SAME public interfaces at
runtime with parity to the vanilla-MTP suite:

  - non-greedy: ``sample_draft`` captures per draft step, ``compare_and_accept``
    is the acceptance entry point, the rejection dispatch + kernel engage;
  - rejection-disabled: ``compare_and_accept`` falls through to the strict base
    path, no rejection kernel;
  - fail-closed: a forced-invalid ``_rejection_buffers_valid`` makes
    ``compare_and_accept`` fall back to strict (no kernel);
  - all-greedy: ``compare_and_accept`` is reached but the shared bypass keeps the
    kernel uninvoked.

Requires a GPU with enough memory plus the Llama-3.1-8B target and the
EAGLE3-LLaMA3.1-Instruct-8B draft checkpoint under ``llm_models_root()``.
"""

import os
import sys

import pytest
import torch

import tensorrt_llm._torch.speculative.interface as iface
import tensorrt_llm._torch.speculative.one_model_sampler as oms
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.llmapi import Eagle3DecodingConfig, KvCacheConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root


def _paths():
    root = llm_models_root()
    if root is None:
        return None, None
    target = os.path.join(root, "llama-3.1-model", "Llama-3.1-8B-Instruct")
    eagle = os.path.join(root, "EAGLE3-LLaMA3.1-Instruct-8B")
    if not (os.path.isdir(target) and os.path.isdir(eagle)):
        return None, None
    return target, eagle


def _skip_or_paths():
    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 40:
        pytest.skip("Not enough GPU memory for Llama-3.1-8B + Eagle3 draft")
    target, eagle = _paths()
    if target is None:
        pytest.skip("Llama-3.1-8B / EAGLE3-LLaMA3.1-Instruct-8B not available")
    return target, eagle


def _build(target, eagle, use_rejection):
    spec = Eagle3DecodingConfig(
        max_draft_len=4,
        speculative_model=eagle,
        eagle3_one_model=True,
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


def _install_counters(monkeypatch, state):
    """Wrap the shared public/dispatch seams; returns nothing (mutates state)."""
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

    orig_sd = SpecWorkerBase.sample_draft

    def _sd(self, logits, spec_metadata, batch_size, d2t=None, draft_step=None):
        if draft_step is not None:
            state["sample_draft_step"] += 1
        return orig_sd(self, logits, spec_metadata, batch_size, d2t, draft_step)

    monkeypatch.setattr(SpecWorkerBase, "sample_draft", _sd)


def _zero():
    return {
        "kernel": 0,
        "compare_and_accept": 0,
        "dispatch": 0,
        "base": 0,
        "sample_draft_step": 0,
    }


_PROMPT = "The capital of France is"


@pytest.mark.high_cuda_memory
def test_eagle3_public_interface_rejection_and_greedy(monkeypatch):
    """Non-greedy reaches public compare_and_accept + sample_draft + kernel;
    all-greedy reaches compare_and_accept but bypasses the kernel."""
    target, eagle = _skip_or_paths()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    state = _zero()
    _install_counters(monkeypatch, state)

    llm = _build(target, eagle, use_rejection=True)
    try:
        sp = SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
        t1 = llm.generate([_PROMPT], sp)[0].outputs[0].text
        ng = dict(state)
        sp_g = SamplingParams(max_tokens=32, top_k=1)
        t2 = llm.generate([_PROMPT], sp_g)[0].outputs[0].text
    finally:
        llm.shutdown()

    assert t1 and t2, "generation produced empty output"
    # Non-greedy: public acceptance + draft capture + rejection engaged.
    assert ng["compare_and_accept"] > 0, (
        "Eagle3 did not route acceptance through public compare_and_accept"
    )
    assert ng["sample_draft_step"] > 0, "Eagle3 did not capture drafts through public sample_draft"
    assert ng["dispatch"] > 0 and ng["kernel"] > 0, (
        "Eagle3 non-greedy rejection dispatch/kernel did not engage"
    )
    # All-greedy phase: compare_and_accept still reached, kernel not re-invoked.
    assert state["compare_and_accept"] > ng["compare_and_accept"], (
        "all-greedy phase did not reach the public compare_and_accept wrapper"
    )
    assert state["kernel"] == ng["kernel"], (
        "all-greedy phase invoked the rejection kernel (must bypass)"
    )


@pytest.mark.high_cuda_memory
def test_eagle3_rejection_disabled_uses_base_via_public_interface(monkeypatch):
    """Rejection-disabled Eagle3 reaches compare_and_accept and falls through to
    the strict base path with no rejection kernel."""
    target, eagle = _skip_or_paths()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    state = _zero()
    _install_counters(monkeypatch, state)

    llm = _build(target, eagle, use_rejection=False)
    try:
        sp = SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
        text = llm.generate([_PROMPT], sp)[0].outputs[0].text
    finally:
        llm.shutdown()

    assert text, "generation produced empty output"
    assert state["compare_and_accept"] > 0, (
        "rejection-disabled Eagle3 bypassed the public compare_and_accept"
    )
    assert state["base"] > 0, "rejection-disabled Eagle3 did not use the strict base path"
    assert state["kernel"] == 0, "rejection kernel invoked with rejection disabled"


@pytest.mark.high_cuda_memory
def test_eagle3_fail_closed_via_public_interface(monkeypatch):
    """One-shot forced-invalid: the forced gen-bearing acceptance falls back to
    strict (no kernel) and clears draft_probs_valid; a LATER normal acceptance on
    the same worker recovers to the rejection dispatch + kernel."""
    target, eagle = _skip_or_paths()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    state = _zero()
    _install_counters(monkeypatch, state)

    fc = {
        "forced_done": False,
        "forced_kernel_delta": None,
        "forced_validity_after": None,
    }

    orig_valid = SpecWorkerBase._rejection_buffers_valid

    def _valid(
        self, draft_tokens, draft_len, stored_vocab, num_contexts, batch_size, logits, spec_metadata
    ):
        # Force invalid exactly once, on the first gen-bearing acceptance.
        if not fc["forced_done"] and (batch_size - num_contexts) > 0:
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
        # Detect the forced iteration: first gen-bearing acceptance while the
        # one-shot force is still pending.
        is_forced = (not fc["forced_done"]) and (batch_size - num_contexts) > 0
        k_before = state["kernel"]
        out = orig_accept(self, logits, draft_tokens, num_contexts, batch_size, spec_metadata)
        if is_forced:
            fc["forced_done"] = True
            fc["forced_kernel_delta"] = state["kernel"] - k_before
            fc["forced_validity_after"] = bool(spec_metadata.draft_probs_valid)
        return out

    monkeypatch.setattr(SpecWorkerBase, "_accept_draft_tokens", _accept)

    llm = _build(target, eagle, use_rejection=True)
    try:
        # Enough tokens that a later forward recovers after the one-shot force.
        sp = SamplingParams(max_tokens=48, temperature=0.8, top_p=0.95, seed=0)
        text = llm.generate([_PROMPT], sp)[0].outputs[0].text
    finally:
        llm.shutdown()

    assert text, "generation produced empty output"
    assert state["compare_and_accept"] > 0, (
        "Eagle3 did not reach the public compare_and_accept wrapper"
    )
    # A gen-bearing acceptance was actually forced invalid.
    assert fc["forced_done"], "no gen-bearing acceptance was force-invalidated"
    # The forced iteration ran no rejection kernel and cleared validity.
    assert fc["forced_kernel_delta"] == 0, (
        "rejection kernel ran during the forced-invalid iteration"
    )
    assert fc["forced_validity_after"] is False, (
        "draft_probs_valid was not cleared after the fail-closed fallback"
    )
    assert state["base"] > 0, "fail-closed path did not fall through to strict base acceptance"
    # Recovery: a later normal acceptance reached the rejection path + kernel.
    assert state["dispatch"] > 0 and state["kernel"] > 0, (
        "rejection did not recover on a later acceptance after one-shot failure"
    )
