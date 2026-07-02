# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end runtime tests for vanilla-MTP rejection sampling.

Runs a real DeepSeek-V3-Lite (vanilla MTP) generation and verifies the rejection
runtime contract directly (not just non-empty output):

  - non-greedy: each captured ``draft_probs[batch_slot_ids, draft_step]`` row is a
    valid proposal distribution (finite, non-negative, sums to ~1);
    ``draft_probs_valid`` is False during capture and observed True at the
    consuming acceptance; and the rejection kernel
    (``rejection_sampling_one_model``) is actually invoked;
  - all-greedy (top_k=1): the batch is classified all-greedy and the rejection
    kernel is NOT invoked (all-greedy bypass) while generation still produces
    output (no stale reuse);
  - exact provenance + gather identity: the stored ``draft_probs`` row equals the
    exact sampler ``probs`` and the rows received by acceptance on the next
    forward equal those scattered for the same slots;
  - all-greedy-after-non-greedy on the SAME worker: the greedy phase neither
    scatters nor runs the rejection kernel (no stale-state reuse).

These tests run eagerly (``cuda_graph_config=None``) so the per-iteration
monkeypatches observe and host-sync safely; CUDA-graph-specific validation is a
separate concern. Requires a GPU with enough memory plus the DeepSeek-V3-Lite
checkpoint under ``llm_models_root()``; skipped otherwise.
"""

import os
import sys

import pytest
import torch

import tensorrt_llm._torch.speculative.interface as iface
import tensorrt_llm._torch.speculative.one_model_sampler as oms
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root


def _model_dir():
    root = llm_models_root()
    if root is None:
        return None
    path = os.path.join(root, "DeepSeek-V3-Lite", "bf16")
    return path if os.path.isdir(path) else None


def _skip_if_unavailable():
    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 60:
        pytest.skip("Not enough GPU memory for DeepSeek-V3-Lite")
    model_dir = _model_dir()
    if model_dir is None:
        pytest.skip("DeepSeek-V3-Lite checkpoint not available")
    return model_dir


def _build_llm(model_dir, max_batch_size=1):
    spec = MTPDecodingConfig(max_draft_len=1, use_mtp_vanilla=True, use_rejection_sampling=True)
    # Eager (no CUDA graph) so the per-iteration monkeypatches run every step
    # and can host-sync without violating stream capture.
    return LLM(
        model=model_dir,
        speculative_config=spec,
        max_batch_size=max_batch_size,
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        kv_cache_config=KvCacheConfig(max_tokens=8192),
    )


@pytest.mark.high_cuda_memory
def test_mtp_vanilla_rejection_runtime(monkeypatch):
    """Non-greedy: per-step provenance + validity lifecycle + path hit."""
    model_dir = _skip_if_unavailable()
    # Single-process worker so the in-process monkeypatches are visible to the
    # generation path (MPI-spawned workers would not see them).
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    state = {
        "kernel_calls": 0,
        "valid_at_capture": [],  # draft_probs_valid observed during capture
        "valid_at_accept": [],  # draft_probs_valid observed at acceptance
        "bad_prob_rows": 0,  # rows that are not valid distributions
        "captured_steps": 0,
    }

    orig_kernel = oms.rejection_sampling_one_model

    def _counting_kernel(*args, **kwargs):
        state["kernel_calls"] += 1
        return orig_kernel(*args, **kwargs)

    monkeypatch.setattr(oms, "rejection_sampling_one_model", _counting_kernel)
    monkeypatch.setattr(iface, "rejection_sampling_one_model", _counting_kernel)

    orig_sample_draft = SpecWorkerBase.sample_draft

    def _wrapped_sample_draft(self, logits, spec_metadata, batch_size, d2t=None, draft_step=None):
        # Producer side: validity must be False while the draft is being
        # captured (it is only set True after the full draft loop).
        if draft_step is not None and getattr(spec_metadata, "use_rejection_sampling", False):
            state["valid_at_capture"].append(bool(spec_metadata.draft_probs_valid))
        token = orig_sample_draft(self, logits, spec_metadata, batch_size, d2t, draft_step)
        # After the step, the scattered proposal row must be a valid prob dist.
        if (
            draft_step is not None
            and getattr(spec_metadata, "use_rejection_sampling", False)
            and not spec_metadata.is_all_greedy_sample
            and spec_metadata.draft_probs is not None
        ):
            slots = spec_metadata.batch_slot_ids[:batch_size]
            rows = spec_metadata.draft_probs[slots, draft_step].float()
            sums = rows.sum(dim=-1)
            ok = (
                bool(torch.isfinite(rows).all())
                and bool((rows >= 0).all())
                and bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-2))
            )
            state["captured_steps"] += 1
            if not ok:
                state["bad_prob_rows"] += 1
        return token

    monkeypatch.setattr(SpecWorkerBase, "sample_draft", _wrapped_sample_draft)

    orig_accept = SpecWorkerBase.compare_and_accept

    def _wrapped_accept(self, logits, draft_tokens, num_contexts, batch_size, spec_metadata):
        # Consumer side: by the time acceptance runs on a captured draft, the
        # producer should have flipped validity to True.
        if getattr(spec_metadata, "use_rejection_sampling", False):
            state["valid_at_accept"].append(bool(spec_metadata.draft_probs_valid))
        return orig_accept(self, logits, draft_tokens, num_contexts, batch_size, spec_metadata)

    monkeypatch.setattr(SpecWorkerBase, "compare_and_accept", _wrapped_accept)

    llm = _build_llm(model_dir)
    try:
        sp = SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
        out = llm.generate(["The capital of France is"], sp)
        text = out[0].outputs[0].text
    finally:
        llm.shutdown()

    assert text, "generation produced empty output"
    # Path hit: rejection kernel actually ran.
    assert state["kernel_calls"] > 0, "rejection kernel never invoked"
    # Per-step provenance: at least one step captured, none malformed.
    assert state["captured_steps"] > 0, "no draft steps captured"
    assert state["bad_prob_rows"] == 0, "a captured draft_probs row was not a valid distribution"
    # Validity lifecycle: never True during capture.
    assert state["valid_at_capture"], "no capture observations"
    assert not any(state["valid_at_capture"]), (
        "draft_probs_valid was True during draft capture (must be False)"
    )
    # And observed True at least once at acceptance (consumer side).
    assert any(state["valid_at_accept"]), "draft_probs_valid never True at acceptance"


@pytest.mark.high_cuda_memory
def test_mtp_all_greedy_bypasses_rejection(monkeypatch):
    """All-greedy (top_k=1) must bypass the rejection kernel entirely.

    For an all-greedy batch the accepted result equals argmax, so the worker
    keeps the cheaper (THOP) strict path and never invokes
    ``rejection_sampling_one_model``.
    """
    model_dir = _skip_if_unavailable()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    calls = {"n": 0}
    orig_kernel = oms.rejection_sampling_one_model

    def _counting_kernel(*args, **kwargs):
        calls["n"] += 1
        return orig_kernel(*args, **kwargs)

    monkeypatch.setattr(oms, "rejection_sampling_one_model", _counting_kernel)
    monkeypatch.setattr(iface, "rejection_sampling_one_model", _counting_kernel)

    llm = _build_llm(model_dir)
    try:
        # top_k=1 unambiguously implies greedy decoding.
        sp = SamplingParams(max_tokens=32, top_k=1)
        out = llm.generate(["The capital of France is"], sp)
        text = out[0].outputs[0].text
    finally:
        llm.shutdown()

    assert text, "generation produced empty output"
    assert calls["n"] == 0, "rejection kernel was invoked for an all-greedy batch (must bypass)"


@pytest.mark.high_cuda_memory
def test_mtp_rejection_exact_provenance_and_gather(monkeypatch):
    """Exact proposal provenance + next-forward gather identity.

    (1) The row stored at ``draft_probs[batch_slot_ids, draft_step]`` equals the
        exact ``probs`` tensor returned by the draft sampler (not merely a
        normalized distribution).
    (2) The ``draft_probs`` argument received by the rejection acceptance on the
        NEXT forward equals the rows that were scattered for those same request
        slots on the previous forward (the cross-forward slot contract).
    """
    model_dir = _skip_if_unavailable()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    state = {
        "last_probs": None,  # most recent sampler output
        "scattered": {},  # (slot, step) -> exact stored row
        "provenance_checked": 0,
        "provenance_mismatch": 0,
        "gather_checked": 0,
        "gather_mismatch": 0,
    }

    orig_sampler = iface.sampling_batch_spec_dec_one_model_for_rejection

    def _capturing_sampler(*args, **kwargs):
        draft_tokens, probs = orig_sampler(*args, **kwargs)
        state["last_probs"] = probs.detach().clone()
        return draft_tokens, probs

    monkeypatch.setattr(
        iface, "sampling_batch_spec_dec_one_model_for_rejection", _capturing_sampler
    )

    orig_draft = SpecWorkerBase._draft_sampler_advanced_for_rejection

    def _wrapped_draft(self, logits, spec_metadata, batch_size, d2t=None, draft_step=0):
        token = orig_draft(self, logits, spec_metadata, batch_size, d2t, draft_step)
        if (
            not spec_metadata.is_all_greedy_sample
            and state["last_probs"] is not None
            and spec_metadata.draft_probs is not None
        ):
            probs = state["last_probs"]
            vocab = probs.shape[-1]
            slots = spec_metadata.batch_slot_ids[:batch_size]
            stored = spec_metadata.draft_probs[slots, draft_step, :vocab]
            state["provenance_checked"] += 1
            if not torch.equal(stored, probs):
                state["provenance_mismatch"] += 1
            # Record exact rows by stable slot id for the gather check.
            for j in range(batch_size):
                state["scattered"][(int(slots[j]), int(draft_step))] = stored[j].detach().clone()
            state["last_probs"] = None
        return token

    monkeypatch.setattr(SpecWorkerBase, "_draft_sampler_advanced_for_rejection", _wrapped_draft)

    orig_rej = SpecWorkerBase._sample_and_accept_draft_tokens_rejection

    def _wrapped_rej(
        self, logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata
    ):
        gen_slots = spec_metadata.batch_slot_ids[num_contexts:batch_size]
        draft_len = draft_probs.shape[1]
        for i in range(draft_probs.shape[0]):
            slot = int(gen_slots[i])
            for s in range(draft_len):
                key = (slot, s)
                if key in state["scattered"]:
                    state["gather_checked"] += 1
                    expected = state["scattered"][key]
                    got = draft_probs[i, s, : expected.shape[-1]]
                    if not torch.equal(got, expected):
                        state["gather_mismatch"] += 1
        return orig_rej(
            self, logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata
        )

    monkeypatch.setattr(SpecWorkerBase, "_sample_and_accept_draft_tokens_rejection", _wrapped_rej)

    llm = _build_llm(model_dir)
    try:
        sp = SamplingParams(max_tokens=48, temperature=0.8, top_p=0.95, seed=0)
        out = llm.generate(["The capital of France is"], sp)
        text = out[0].outputs[0].text
    finally:
        llm.shutdown()

    assert text, "generation produced empty output"
    # Exact proposal provenance: stored row == sampler output, every step.
    assert state["provenance_checked"] > 0, "no draft steps observed"
    assert state["provenance_mismatch"] == 0, (
        "stored draft_probs row differed from the sampler's exact probs"
    )
    # Next-forward gather identity: acceptance received exactly what was stored.
    assert state["gather_checked"] > 0, "no gather observations at acceptance"
    assert state["gather_mismatch"] == 0, (
        "acceptance draft_probs differed from the previously-scattered slot rows"
    )


@pytest.mark.high_cuda_memory
def test_mtp_greedy_after_nongreedy_no_stale_reuse(monkeypatch):
    """All-greedy after a non-greedy capture must not reuse stale state.

    Reuses the SAME LLM/worker: first a non-greedy generation (rejection
    captures + runs), then a ``top_k=1`` greedy generation. The greedy phase must
    neither scatter draft probs nor invoke the rejection kernel.
    """
    model_dir = _skip_if_unavailable()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    state = {
        "phase": "nongreedy",
        "kernel": {"nongreedy": 0, "greedy": 0},
        "scatter": {"nongreedy": 0, "greedy": 0},
    }

    orig_kernel = oms.rejection_sampling_one_model

    def _counting_kernel(*args, **kwargs):
        state["kernel"][state["phase"]] += 1
        return orig_kernel(*args, **kwargs)

    monkeypatch.setattr(oms, "rejection_sampling_one_model", _counting_kernel)
    monkeypatch.setattr(iface, "rejection_sampling_one_model", _counting_kernel)

    orig_draft = SpecWorkerBase._draft_sampler_advanced_for_rejection

    def _counting_draft(self, logits, spec_metadata, batch_size, d2t=None, draft_step=0):
        # A non-all-greedy call is the only path that scatters probs.
        if not spec_metadata.is_all_greedy_sample:
            state["scatter"][state["phase"]] += 1
        return orig_draft(self, logits, spec_metadata, batch_size, d2t, draft_step)

    monkeypatch.setattr(SpecWorkerBase, "_draft_sampler_advanced_for_rejection", _counting_draft)

    llm = _build_llm(model_dir)
    try:
        state["phase"] = "nongreedy"
        out1 = llm.generate(
            ["The capital of France is"],
            SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, seed=0),
        )
        state["phase"] = "greedy"
        out2 = llm.generate(["The capital of France is"], SamplingParams(max_tokens=32, top_k=1))
        t1, t2 = out1[0].outputs[0].text, out2[0].outputs[0].text
    finally:
        llm.shutdown()

    assert t1 and t2, "generation produced empty output"
    # Non-greedy phase exercised rejection (sanity that the worker is shared).
    assert state["kernel"]["nongreedy"] > 0, "non-greedy phase never rejected"
    assert state["scatter"]["nongreedy"] > 0, "non-greedy phase never scattered"
    # Greedy phase on the SAME worker must not reuse or run rejection.
    assert state["scatter"]["greedy"] == 0, "greedy phase scattered draft probs (stale-state risk)"
    assert state["kernel"]["greedy"] == 0, (
        "greedy phase invoked the rejection kernel (stale-state reuse)"
    )


@pytest.mark.high_cuda_memory
def test_mtp_rejection_fail_closed_fallback(monkeypatch):
    """Real-worker fail-closed: invalid buffers => strict fallback, no kernel.

    Forces ``_rejection_buffers_valid`` to report invalid on the first
    gen-bearing acceptance (modelling a skipped/partial/stale scatter). The
    worker must then take the strict (base) path: no ``rejection_sampling_one_model``
    call that iteration, and ``draft_probs_valid`` cleared to False. A later
    unforced iteration must still run rejection, proving the test is not
    trivially passing.
    """
    model_dir = _skip_if_unavailable()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    state = {
        "kernel": 0,
        "forced_once": False,
        "forced_kernel_delta": None,
        "forced_validity_after": None,
        "normal_kernel_seen": 0,
    }

    orig_kernel = oms.rejection_sampling_one_model

    def _counting_kernel(*args, **kwargs):
        state["kernel"] += 1
        return orig_kernel(*args, **kwargs)

    monkeypatch.setattr(oms, "rejection_sampling_one_model", _counting_kernel)
    monkeypatch.setattr(iface, "rejection_sampling_one_model", _counting_kernel)

    orig_valid = SpecWorkerBase._rejection_buffers_valid
    force = {"on": False}

    def _maybe_invalid(self, *args, **kwargs):
        if force["on"]:
            return False  # simulate detected stale/partial scatter
        return orig_valid(self, *args, **kwargs)

    monkeypatch.setattr(SpecWorkerBase, "_rejection_buffers_valid", _maybe_invalid)

    orig_accept = SpecWorkerBase._accept_draft_tokens

    def _wrapped_accept(self, logits, draft_tokens, num_contexts, batch_size, spec_metadata):
        num_gens = batch_size - num_contexts
        # Force the fail-closed path exactly once, on the first gen acceptance
        # where rejection would otherwise engage.
        do_force = (
            not state["forced_once"]
            and num_gens > 0
            and spec_metadata.use_rejection_sampling
            and spec_metadata.draft_probs_valid
            and not spec_metadata.is_all_greedy_sample
        )
        if do_force:
            force["on"] = True
            before = state["kernel"]
            result = orig_accept(
                self, logits, draft_tokens, num_contexts, batch_size, spec_metadata
            )
            force["on"] = False
            state["forced_once"] = True
            state["forced_kernel_delta"] = state["kernel"] - before
            state["forced_validity_after"] = bool(spec_metadata.draft_probs_valid)
            return result
        before = state["kernel"]
        result = orig_accept(self, logits, draft_tokens, num_contexts, batch_size, spec_metadata)
        if state["kernel"] > before:
            state["normal_kernel_seen"] += 1
        return result

    monkeypatch.setattr(SpecWorkerBase, "_accept_draft_tokens", _wrapped_accept)

    llm = _build_llm(model_dir)
    try:
        sp = SamplingParams(max_tokens=48, temperature=0.8, top_p=0.95, seed=0)
        out = llm.generate(["The capital of France is"], sp)
        text = out[0].outputs[0].text
    finally:
        llm.shutdown()

    assert text, "generation produced empty output"
    assert state["forced_once"], "fail-closed path was never exercised"
    # The forced iteration must NOT have run the rejection kernel...
    assert state["forced_kernel_delta"] == 0, (
        "rejection kernel ran despite invalid buffers (must fail closed)"
    )
    # ...and must have cleared validity so stale state cannot be reused.
    assert state["forced_validity_after"] is False, (
        "draft_probs_valid not cleared after fail-closed fallback"
    )
    # A later normal iteration still ran rejection (test is not vacuous).
    assert state["normal_kernel_seen"] > 0, (
        "rejection never ran on a normal iteration (test would be vacuous)"
    )


@pytest.mark.high_cuda_memory
def test_mtp_rejection_multi_request_slot_identity(monkeypatch):
    """Multi-request (max_batch_size>1): each acceptance gathers its OWN slots.

    Runs several concurrent prompts of differing lengths so the batch contains
    multiple request slots (and possibly mixed context+generation forwards). The
    rejection acceptance for each gen request must receive exactly the draft-prob
    rows scattered for its own ``batch_slot_ids[num_contexts:batch_size]`` slots —
    never another request's rows.
    """
    model_dir = _skip_if_unavailable()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    state = {
        "last_probs": None,
        "scattered": {},  # (slot, step) -> exact stored row
        "gather_checked": 0,
        "gather_mismatch": 0,
        "mixed_forward_seen": 0,
        "mixed_gather_checked": 0,
        "mixed_gather_mismatch": 0,
    }

    orig_sampler = iface.sampling_batch_spec_dec_one_model_for_rejection

    def _capturing_sampler(*args, **kwargs):
        draft_tokens, probs = orig_sampler(*args, **kwargs)
        state["last_probs"] = probs.detach().clone()
        return draft_tokens, probs

    monkeypatch.setattr(
        iface, "sampling_batch_spec_dec_one_model_for_rejection", _capturing_sampler
    )

    orig_draft = SpecWorkerBase._draft_sampler_advanced_for_rejection

    def _wrapped_draft(self, logits, spec_metadata, batch_size, d2t=None, draft_step=0):
        token = orig_draft(self, logits, spec_metadata, batch_size, d2t, draft_step)
        if (
            not spec_metadata.is_all_greedy_sample
            and state["last_probs"] is not None
            and spec_metadata.draft_probs is not None
        ):
            vocab = state["last_probs"].shape[-1]
            slots = spec_metadata.batch_slot_ids[:batch_size]
            stored = spec_metadata.draft_probs[slots, draft_step, :vocab]
            for j in range(batch_size):
                state["scattered"][(int(slots[j]), int(draft_step))] = stored[j].detach().clone()
            state["last_probs"] = None
        return token

    monkeypatch.setattr(SpecWorkerBase, "_draft_sampler_advanced_for_rejection", _wrapped_draft)

    orig_rej = SpecWorkerBase._sample_and_accept_draft_tokens_rejection

    def _wrapped_rej(
        self, logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata
    ):
        is_mixed = num_contexts > 0 and (batch_size - num_contexts) > 0
        if is_mixed:
            state["mixed_forward_seen"] += 1
        gen_slots = spec_metadata.batch_slot_ids[num_contexts:batch_size]
        draft_len = draft_probs.shape[1]
        for i in range(draft_probs.shape[0]):
            slot = int(gen_slots[i])
            for s in range(draft_len):
                key = (slot, s)
                if key in state["scattered"]:
                    state["gather_checked"] += 1
                    expected = state["scattered"][key]
                    got = draft_probs[i, s, : expected.shape[-1]]
                    match = torch.equal(got, expected)
                    if not match:
                        state["gather_mismatch"] += 1
                    if is_mixed:
                        state["mixed_gather_checked"] += 1
                        if not match:
                            state["mixed_gather_mismatch"] += 1
        return orig_rej(
            self, logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata
        )

    monkeypatch.setattr(SpecWorkerBase, "_sample_and_accept_draft_tokens_rejection", _wrapped_rej)

    llm = _build_llm(model_dir, max_batch_size=4)
    try:
        prompts = [
            "The capital of France is",
            "In a galaxy far away, there once lived a",
            "Explain in detail why the sky appears blue during the day and",
            "List three reasons that regular exercise improves",
        ]
        sp = SamplingParams(max_tokens=48, temperature=0.8, top_p=0.95, seed=0)
        outs = llm.generate(prompts, sp)
        texts = [o.outputs[0].text for o in outs]
    finally:
        llm.shutdown()

    assert all(texts), "a request produced empty output"
    # At least two distinct request slots were exercised.
    assert len({k[0] for k in state["scattered"]}) >= 2, (
        "fewer than two request slots were observed"
    )
    # Each acceptance gathered exactly its own scattered rows.
    assert state["gather_checked"] > 0, "no gather observations at acceptance"
    assert state["gather_mismatch"] == 0, (
        "a gen request read another request's stored draft probs (slot churn bug)"
    )
    # A real mixed context+generation forward must have occurred AND its gen
    # subset gathered exactly its own slot rows (enforced, not just printed).
    assert state["mixed_forward_seen"] > 0, (
        "no mixed context+generation rejection forward was observed"
    )
    assert state["mixed_gather_checked"] > 0, (
        "mixed forward observed but no gather rows checked in it"
    )
    assert state["mixed_gather_mismatch"] == 0, (
        "mixed-forward gen subset read the wrong slot's draft probs"
    )


@pytest.mark.high_cuda_memory
def test_mtp_rejection_slot_reuse_across_generations(monkeypatch):
    """True slot churn: a freed slot reused by a later request reads fresh rows.

    Runs two sequential NON-greedy generations on a small-batch LLM so the same
    request slot is reused across generations. Each scattered ``(slot, step)`` row
    is tagged with its generation epoch. Asserts at least one slot id is reused
    across both epochs, and every acceptance gathers a row whose recorded epoch
    equals the current epoch (no stale prior-epoch row survives), with no value
    mismatch.
    """
    model_dir = _skip_if_unavailable()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    state = {
        "epoch": 0,
        "last_probs": None,
        "scattered": {},  # (slot, step) -> (epoch, row)
        "epoch_slots": {0: set(), 1: set()},
        "gather_checked": 0,
        "value_mismatch": 0,
        "epoch_mismatch": 0,
    }

    orig_sampler = iface.sampling_batch_spec_dec_one_model_for_rejection

    def _capturing_sampler(*args, **kwargs):
        draft_tokens, probs = orig_sampler(*args, **kwargs)
        state["last_probs"] = probs.detach().clone()
        return draft_tokens, probs

    monkeypatch.setattr(
        iface, "sampling_batch_spec_dec_one_model_for_rejection", _capturing_sampler
    )

    orig_draft = SpecWorkerBase._draft_sampler_advanced_for_rejection

    def _wrapped_draft(self, logits, spec_metadata, batch_size, d2t=None, draft_step=0):
        token = orig_draft(self, logits, spec_metadata, batch_size, d2t, draft_step)
        if (
            not spec_metadata.is_all_greedy_sample
            and state["last_probs"] is not None
            and spec_metadata.draft_probs is not None
        ):
            vocab = state["last_probs"].shape[-1]
            slots = spec_metadata.batch_slot_ids[:batch_size]
            stored = spec_metadata.draft_probs[slots, draft_step, :vocab]
            ep = state["epoch"]
            for j in range(batch_size):
                slot = int(slots[j])
                state["scattered"][(slot, int(draft_step))] = (ep, stored[j].detach().clone())
                state["epoch_slots"][ep].add(slot)
            state["last_probs"] = None
        return token

    monkeypatch.setattr(SpecWorkerBase, "_draft_sampler_advanced_for_rejection", _wrapped_draft)

    orig_rej = SpecWorkerBase._sample_and_accept_draft_tokens_rejection

    def _wrapped_rej(
        self, logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata
    ):
        gen_slots = spec_metadata.batch_slot_ids[num_contexts:batch_size]
        draft_len = draft_probs.shape[1]
        for i in range(draft_probs.shape[0]):
            slot = int(gen_slots[i])
            for s in range(draft_len):
                key = (slot, s)
                if key in state["scattered"]:
                    rec_epoch, expected = state["scattered"][key]
                    state["gather_checked"] += 1
                    got = draft_probs[i, s, : expected.shape[-1]]
                    if not torch.equal(got, expected):
                        state["value_mismatch"] += 1
                    if rec_epoch != state["epoch"]:
                        state["epoch_mismatch"] += 1
        return orig_rej(
            self, logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata
        )

    monkeypatch.setattr(SpecWorkerBase, "_sample_and_accept_draft_tokens_rejection", _wrapped_rej)

    # max_batch_size=1 forces the single request slot to be reused each epoch.
    llm = _build_llm(model_dir, max_batch_size=1)
    try:
        sp = SamplingParams(max_tokens=40, temperature=0.8, top_p=0.95, seed=0)
        state["epoch"] = 0
        o0 = llm.generate(["The capital of France is"], sp)
        state["epoch"] = 1
        o1 = llm.generate(["A short history of the Roman empire begins"], sp)
        t0, t1 = o0[0].outputs[0].text, o1[0].outputs[0].text
    finally:
        llm.shutdown()

    assert t0 and t1, "generation produced empty output"
    # A slot was reused across the two generations.
    reused = state["epoch_slots"][0] & state["epoch_slots"][1]
    assert reused, "no slot was reused across the two generations"
    assert state["gather_checked"] > 0, "no gather observations at acceptance"
    # Every gathered row was the current epoch's row (no stale survivor)...
    assert state["epoch_mismatch"] == 0, (
        "acceptance gathered a stale prior-epoch draft-prob row (slot reuse bug)"
    )
    # ...and matched the exact stored values.
    assert state["value_mismatch"] == 0, "acceptance gathered a value-mismatched draft-prob row"


@pytest.mark.high_cuda_memory
def test_mtp_rejection_cuda_graph_key_separation(monkeypatch):
    """CUDA-graph greedy/non-greedy graph-key separation.

    With CUDA graphs ENABLED, runs greedy -> non-greedy -> greedy on one LLM plus
    a separate eager greedy reference, and proves the greedy and non-greedy
    requests dispatch to DISTINCT captured graph variants at replay.

    The MTP rejection acceptance is captured INSIDE the CUDA graph, so a Python
    counter on the rejection kernel only fires during capture (warmup) and reads
    0 for every replayed generate phase. The decisive evidence is therefore the
    replayed graph KEY: `CUDAGraphRunner.get_graph_key()` puts
    `is_all_greedy_sample` as the last key element, so the greedy requests must
    replay only `key[-1] is True` graphs and the non-greedy request must replay
    at least one `key[-1] is False` graph. Output invariants
    (greedy-graph == eager-greedy, greedy stable across the intervening
    non-greedy run) confirm no stale rejection state is replayed for greedy.
    """
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner

    model_dir = _skip_if_unavailable()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    phase = {"name": "warmup"}
    # Rejection-kernel calls observed at capture time, keyed by phase.
    counts = {"warmup": 0, "greedy1": 0, "nongreedy": 0, "greedy2": 0}
    # Replayed graph keys observed per phase (host-side, outside capture).
    replay_keys = {"warmup": [], "greedy1": [], "nongreedy": [], "greedy2": []}

    orig_kernel = oms.rejection_sampling_one_model

    def _counting_kernel(*args, **kwargs):
        # Pure host-side Python increment: runs during graph CAPTURE (tracing),
        # never reads a CUDA tensor, so it is stream-capture safe.
        counts[phase["name"]] = counts.get(phase["name"], 0) + 1
        return orig_kernel(*args, **kwargs)

    monkeypatch.setattr(oms, "rejection_sampling_one_model", _counting_kernel)
    monkeypatch.setattr(iface, "rejection_sampling_one_model", _counting_kernel)

    orig_replay = CUDAGraphRunner.replay

    def _recording_replay(self, key, current_inputs):
        # replay() runs OUTSIDE stream capture (it launches an already-captured
        # graph), so recording the dispatched key here is capture-safe.
        replay_keys[phase["name"]].append(key)
        return orig_replay(self, key, current_inputs)

    monkeypatch.setattr(CUDAGraphRunner, "replay", _recording_replay)

    spec = MTPDecodingConfig(max_draft_len=1, use_mtp_vanilla=True, use_rejection_sampling=True)
    graph_llm = LLM(
        model=model_dir,
        speculative_config=spec,
        max_batch_size=1,
        disable_overlap_scheduler=True,
        cuda_graph_config=CudaGraphConfig(),
        kv_cache_config=KvCacheConfig(max_tokens=4096),
    )
    greedy_sp = SamplingParams(max_tokens=32, top_k=1)
    nongreedy_sp = SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
    prompt = "The capital of France is"
    try:
        phase["name"] = "greedy1"
        g1 = graph_llm.generate([prompt], greedy_sp)[0].outputs[0].text
        phase["name"] = "nongreedy"
        n1 = graph_llm.generate([prompt], nongreedy_sp)[0].outputs[0].text
        phase["name"] = "greedy2"
        g2 = graph_llm.generate([prompt], greedy_sp)[0].outputs[0].text
    finally:
        graph_llm.shutdown()

    # Separate eager greedy reference (no CUDA graph).
    eager_llm = _build_llm(model_dir, max_batch_size=1)
    try:
        g_eager = eager_llm.generate([prompt], greedy_sp)[0].outputs[0].text
    finally:
        eager_llm.shutdown()

    print(f"CUDA_GRAPH_REJECTION_COUNTS={counts}")
    print("CUDA_GRAPH_REPLAY_KEYS=" + str({k: v for k, v in replay_keys.items() if v}))
    assert g1 and n1 and g2 and g_eager, "generation produced empty output"

    # --- Direct replay-key separation -------------------------------------
    # Each real generate phase must have replayed at least one captured graph.
    for ph in ("greedy1", "nongreedy", "greedy2"):
        assert replay_keys[ph], f"no CUDA graph replayed during {ph}"
    # is_all_greedy_sample is the last graph-key element.
    for ph in ("greedy1", "greedy2"):
        assert all(k[-1] is True for k in replay_keys[ph]), (
            f"{ph} replayed a non-greedy (rejection) graph key: {replay_keys[ph]}"
        )
    assert any(k[-1] is False for k in replay_keys["nongreedy"]), (
        f"non-greedy phase did not replay a non-greedy graph key: {replay_keys['nongreedy']}"
    )

    # --- Capture-time kernel counts (corroborating) -----------------------
    # The rejection kernel is captured during warmup and not re-counted at
    # replay; if the non-greedy phase had fallen back to eager rejection, its
    # phase counter would be > 0.
    assert counts["warmup"] > 0, "rejection path was never captured at warmup"
    assert counts["greedy1"] == counts["nongreedy"] == counts["greedy2"] == 0, (
        f"unexpected eager rejection execution during replay: {counts}"
    )

    # --- Output invariants (no stale rejection state for greedy) ----------
    assert g1 == g2, "greedy output changed after a non-greedy run (stale graph state)"
    assert g1 == g_eager, "greedy CUDA-graph output diverged from the eager greedy reference"


@pytest.mark.high_cuda_memory
def test_mtp_rejection_disabled_keeps_strict_path(monkeypatch):
    """Rejection-disabled vanilla MTP must keep the strict acceptance path.

    With use_rejection_sampling=False and non-greedy sampling, the worker must
    never invoke the rejection kernel and must accept through the strict base
    path (``_sample_and_accept_draft_tokens_base``). ``is_thop`` defaults False
    in this build, so the reachable strict path is the base implementation.
    """
    model_dir = _skip_if_unavailable()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    calls = {"reject": 0, "strict": 0}
    orig_kernel = oms.rejection_sampling_one_model

    def _count_reject(*args, **kwargs):
        calls["reject"] += 1
        return orig_kernel(*args, **kwargs)

    monkeypatch.setattr(oms, "rejection_sampling_one_model", _count_reject)
    monkeypatch.setattr(iface, "rejection_sampling_one_model", _count_reject)

    orig_strict = SpecWorkerBase._sample_and_accept_draft_tokens_base

    def _count_strict(self, *args, **kwargs):
        calls["strict"] += 1
        return orig_strict(self, *args, **kwargs)

    monkeypatch.setattr(SpecWorkerBase, "_sample_and_accept_draft_tokens_base", _count_strict)

    spec = MTPDecodingConfig(max_draft_len=1, use_mtp_vanilla=True, use_rejection_sampling=False)
    llm = LLM(
        model=model_dir,
        speculative_config=spec,
        max_batch_size=1,
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        kv_cache_config=KvCacheConfig(max_tokens=4096),
    )
    try:
        sp = SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
        text = llm.generate(["The capital of France is"], sp)[0].outputs[0].text
    finally:
        llm.shutdown()

    assert text, "generation produced empty output"
    assert calls["reject"] == 0, "rejection kernel was invoked with rejection sampling disabled"
    assert calls["strict"] > 0, "strict base acceptance path was never used (regression)"


@pytest.mark.high_cuda_memory
def test_mtp_nongreedy_rejection_uses_base_rejection_dispatch(monkeypatch):
    """Non-greedy rejection routes through the base rejection dispatch.

    Proves the gen-bearing forwards reach
    ``_sample_and_accept_draft_tokens_rejection`` (the non-THOP rejection path),
    not the THOP fused strict op (``is_thop`` is False here) or the relaxed op.
    """
    model_dir = _skip_if_unavailable()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    calls = {"reject_dispatch": 0, "kernel": 0}
    orig_kernel = oms.rejection_sampling_one_model

    def _count_kernel(*args, **kwargs):
        calls["kernel"] += 1
        return orig_kernel(*args, **kwargs)

    monkeypatch.setattr(oms, "rejection_sampling_one_model", _count_kernel)
    monkeypatch.setattr(iface, "rejection_sampling_one_model", _count_kernel)

    orig_dispatch = SpecWorkerBase._sample_and_accept_draft_tokens_rejection

    def _count_dispatch(self, *args, **kwargs):
        calls["reject_dispatch"] += 1
        return orig_dispatch(self, *args, **kwargs)

    monkeypatch.setattr(
        SpecWorkerBase, "_sample_and_accept_draft_tokens_rejection", _count_dispatch
    )

    spec = MTPDecodingConfig(max_draft_len=1, use_mtp_vanilla=True, use_rejection_sampling=True)
    llm = LLM(
        model=model_dir,
        speculative_config=spec,
        max_batch_size=1,
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        kv_cache_config=KvCacheConfig(max_tokens=4096),
    )
    try:
        sp = SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
        text = llm.generate(["The capital of France is"], sp)[0].outputs[0].text
    finally:
        llm.shutdown()

    assert text, "generation produced empty output"
    assert calls["reject_dispatch"] > 0, (
        "non-greedy rejection did not use the base rejection dispatch"
    )
    assert calls["kernel"] > 0, "rejection kernel was never invoked"


@pytest.mark.high_cuda_memory
def test_mtp_relaxed_acceptance_runs_without_rejection(monkeypatch):
    """Relaxed-thinking acceptance still executes when rejection is OFF.

    With use_relaxed_acceptance_for_thinking=True and use_rejection_sampling=False
    the worker must take the relaxed acceptance path (observable via
    MTPWorker.topk_kernel, which is only called inside the relaxed branch) and
    must never invoke the rejection kernel.
    """
    from tensorrt_llm._torch.speculative.mtp import MTPWorker

    model_dir = _skip_if_unavailable()
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    counts = {"relaxed": 0, "kernel": 0}
    orig_kernel = oms.rejection_sampling_one_model

    def _kernel(*a, **k):
        counts["kernel"] += 1
        return orig_kernel(*a, **k)

    monkeypatch.setattr(oms, "rejection_sampling_one_model", _kernel)
    monkeypatch.setattr(iface, "rejection_sampling_one_model", _kernel)

    orig_topk = MTPWorker.topk_kernel

    def _topk(self, *a, **k):
        counts["relaxed"] += 1
        return orig_topk(self, *a, **k)

    monkeypatch.setattr(MTPWorker, "topk_kernel", _topk)

    spec = MTPDecodingConfig(
        max_draft_len=1,
        use_mtp_vanilla=True,
        use_relaxed_acceptance_for_thinking=True,
        use_rejection_sampling=False,
    )
    llm = LLM(
        model=model_dir,
        speculative_config=spec,
        max_batch_size=1,
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        kv_cache_config=KvCacheConfig(max_tokens=4096),
    )
    try:
        sp = SamplingParams(max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
        text = llm.generate(["The capital of France is"], sp)[0].outputs[0].text
    finally:
        llm.shutdown()

    assert text, "generation produced empty output"
    assert counts["relaxed"] > 0, "relaxed acceptance path (topk_kernel) was never executed"
    assert counts["kernel"] == 0, "rejection kernel ran with rejection disabled + relaxed enabled"
