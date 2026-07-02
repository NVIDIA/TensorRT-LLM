# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strengthened multi-method rejection-sampling hardware gate.

Runs the same validation dimensions for every enabled one-model rejection method
(vanilla MTP, PARD, DFLASH, DRAFT_TARGET) on the local B200:

  - greedy bypass (HARD, all): a greedy (top_k=1) rejection-ENABLED run produces
    non-empty output and never invokes the rejection kernel. With the kernel never
    invoked, the greedy output IS the strict/argmax path by construction -- this
    is the substantive "rejection does not alter greedy" baseline property;
  - strict baseline (HARD, all): a greedy rejection-DISABLED build also produces
    non-empty output and never invokes the rejection kernel;
  - rejection path hit (HARD, all): the non-greedy run's rejection dispatch +
    kernel ran (not strict fallback);
  - non-degenerate acceptance (HARD, all): over a small prompt set the accepted
    counts are not all 1 and not all (runtime_draft_len + 1) -- real rejection;
  - CUDA graph (HARD, all): a non-greedy rejection run with CUDA graphs enabled
    replays >=1 captured graph whose key is non-greedy (is_all_greedy_sample is
    False), proving the rejection variant is dispatched at replay, and captures
    the rejection path.

There is no per-method xfail: every method passes all of the above. A bitwise
output-token replay-equality dimension is intentionally NOT asserted: the
TRT-LLM forward kernels use non-associative atomic reductions, so repeated runs
are not guaranteed bitwise-identical for greedy or non-greedy on these
speculative configs (empirically MTP greedy is reproducible same-process while
PARD greedy is not, and non-greedy is reproducible for none). That is a
whole-model kernel property, NOT a rejection-sampling defect; the rejection
MECHANISM's exact determinism is proven by the committed per-step/block
exact-provenance tests (stored draft_probs == the sampler's exact output via
torch.equal; Rounds 14/24/25/29). See hardware-gate-amendment.md.

Per the round-32 plan amendment (hardware-gate-amendment.md), the local B200
release-overlay is the accepted equivalent of the plan's viking-prod-260 gate
(viking ssh unavailable). Requires the relevant checkpoints under
``llm_models_root()``; each method skips if its models or GPU memory are absent.
"""

import os
import sys

import pytest
import torch

import tensorrt_llm._torch.speculative.interface as iface
import tensorrt_llm._torch.speculative.one_model_sampler as oms
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DFlashDecodingConfig,
    DraftTargetDecodingConfig,
    KvCacheConfig,
    MTPDecodingConfig,
    PARDDecodingConfig,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root

K = 4  # max_draft_len for the block/draft-target methods; MTP uses 1
_PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a distant land,",
    "The three primary colors are",
]


def _p(*parts):
    root = llm_models_root()
    return None if root is None else os.path.join(root, *parts)


def _mtp_spec(use_rejection):
    return MTPDecodingConfig(
        max_draft_len=1, use_mtp_vanilla=True, use_rejection_sampling=use_rejection
    ), 1


def _pard_spec(use_rejection):
    return PARDDecodingConfig(
        max_draft_len=K,
        speculative_model=_p("PARD-Llama-3.2-1B"),
        use_rejection_sampling=use_rejection,
    ), K


def _dflash_spec(use_rejection):
    return DFlashDecodingConfig(
        max_draft_len=K,
        speculative_model=_p("LLaMA3.1-8B-Instruct-DFlash-UltraChat"),
        use_rejection_sampling=use_rejection,
    ), K


def _draft_target_spec(use_rejection):
    return DraftTargetDecodingConfig(
        max_draft_len=K,
        speculative_model=_p("llama-3.2-models", "Llama-3.2-1B-Instruct"),
        use_rejection_sampling=use_rejection,
    ), K


# name -> (target_path_parts, draft_path_parts_or_None, spec_builder, min_mem_gb,
#         _reserved)
# The 5th field is reserved/unused: the gate no longer asserts any bitwise
# output-token replay/baseline equality (the forward kernels are not guaranteed
# bitwise-reproducible across runs; see the module docstring and
# hardware-gate-amendment.md). All asserted dimensions (greedy bypass, strict
# kernel-gating, non-greedy path-hit, non-degenerate acceptance, CUDA-graph
# replay-key) are hard for every method, and there is no xfail.
_METHODS = {
    "mtp": (("DeepSeek-V3-Lite", "bf16"), None, _mtp_spec, 60, False),
    "pard": (
        ("llama-3.2-models", "Llama-3.2-1B-Instruct"),
        ("PARD-Llama-3.2-1B",),
        _pard_spec,
        20,
        True,
    ),
    "dflash": (
        ("llama-3.1-model", "Llama-3.1-8B-Instruct"),
        ("LLaMA3.1-8B-Instruct-DFlash-UltraChat",),
        _dflash_spec,
        40,
        True,
    ),
    "draft_target": (
        ("llama-3.1-model", "Llama-3.1-8B-Instruct"),
        ("llama-3.2-models", "Llama-3.2-1B-Instruct"),
        _draft_target_spec,
        45,
        True,
    ),
}


def _skip_check(method):
    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU")
    target_parts, draft_parts, _, min_mem, _det = _METHODS[method]
    if torch.cuda.get_device_properties(0).total_memory / 1e9 < min_mem:
        pytest.skip(f"Not enough GPU memory for {method}")
    target = _p(*target_parts)
    if target is None or not os.path.isdir(target):
        pytest.skip(f"{method}: target model not available")
    if draft_parts is not None and not os.path.isdir(_p(*draft_parts)):
        pytest.skip(f"{method}: draft model not available")
    return target


def _build(method, target, use_rejection, cuda_graph=False):
    spec, _ = _METHODS[method][2](use_rejection)
    return LLM(
        model=target,
        speculative_config=spec,
        max_batch_size=len(_PROMPTS),
        disable_overlap_scheduler=True,
        cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
        max_seq_len=8192,
        kv_cache_config=KvCacheConfig(max_tokens=8192),
    )


def _install(monkeypatch, state):
    orig_kernel = oms.rejection_sampling_one_model

    def _kernel(*a, **k):
        out = orig_kernel(*a, **k)
        # Host-side int increment is capture-safe; the .cpu() collection is NOT,
        # so skip it while a CUDA graph is being captured (would invalidate it).
        if not torch.cuda.is_current_stream_capturing():
            try:
                state["accepts"].extend(out[1].detach().cpu().tolist())
            except Exception:
                pass
        state["kernel"] += 1
        return out

    monkeypatch.setattr(oms, "rejection_sampling_one_model", _kernel)
    monkeypatch.setattr(iface, "rejection_sampling_one_model", _kernel)

    orig_disp = SpecWorkerBase._sample_and_accept_draft_tokens_rejection

    def _disp(self, *a, **k):
        state["dispatch"] += 1
        return orig_disp(self, *a, **k)

    monkeypatch.setattr(SpecWorkerBase, "_sample_and_accept_draft_tokens_rejection", _disp)

    # Record the graph key dispatched at each CUDA-graph replay (host-side,
    # outside capture). is_all_greedy_sample is the final key element.
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner

    orig_replay = CUDAGraphRunner.replay

    def _replay(self, key, *a, **k):
        state["replay_keys"].append(key)
        return orig_replay(self, key, *a, **k)

    monkeypatch.setattr(CUDAGraphRunner, "replay", _replay)


def _new_state():
    return {"kernel": 0, "dispatch": 0, "accepts": [], "replay_keys": []}


def _nongreedy():
    return SamplingParams(max_tokens=16, temperature=0.8, top_p=0.95, seed=0)


def _greedy():
    return SamplingParams(max_tokens=16, top_k=1)


def _run(monkeypatch, method, target, use_rejection, greedy, cuda_graph=False, n_gen=1):
    """Fresh worker; run `n_gen` generations over the prompt set in the SAME
    process. Returns (list_of_ids_per_gen, state)."""
    state = _new_state()
    with monkeypatch.context() as m:
        m.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        _install(m, state)
        llm = _build(method, target, use_rejection, cuda_graph=cuda_graph)
        try:
            sp = _greedy() if greedy else _nongreedy()
            runs = []
            for _ in range(n_gen):
                outs = llm.generate(_PROMPTS, sp)
                runs.append([list(o.outputs[0].token_ids) for o in outs])
        finally:
            llm.shutdown()
    return runs, state


@pytest.mark.high_cuda_memory
@pytest.mark.parametrize("method", list(_METHODS.keys()))
def test_rejection_hardware_gate(method, monkeypatch):
    target = _skip_check(method)
    runtime_draft_len = _METHODS[method][2](True)[1]
    max_accept = runtime_draft_len + 1

    # --- greedy bypass: enabled greedy run never invokes the rejection kernel ---
    # (Bitwise output-token replay is intentionally NOT asserted -- see the module
    # docstring / hardware-gate-amendment.md: the TRT-LLM forward kernels use
    # non-associative atomic reductions, so repeated runs are not guaranteed
    # bitwise-identical for either greedy or non-greedy on these speculative
    # configs -- empirically MTP greedy is reproducible same-process while PARD
    # greedy is not. That is a whole-model kernel property, NOT a rejection
    # defect; the rejection MECHANISM's exact determinism is proven by the
    # committed per-step/block exact-provenance tests, Rounds 14/24/25/29.)
    greedy_runs, st_g = _run(monkeypatch, method, target, True, greedy=True)
    assert all(greedy_runs[0]), f"{method}: empty greedy output"
    # Greedy must bypass rejection entirely; with the kernel never invoked, the
    # greedy output IS the strict/argmax path by construction (this is the
    # substantive "rejection does not alter greedy" baseline property).
    assert st_g["kernel"] == 0, f"{method}: rejection kernel ran for an all-greedy batch"

    # --- strict baseline: greedy rejection-DISABLED run also never rejects ---
    strict_runs, st_strict = _run(monkeypatch, method, target, False, greedy=True)
    assert all(strict_runs[0]), f"{method}: empty strict-baseline output"
    assert st_strict["kernel"] == 0, f"{method}: rejection kernel ran with rejection disabled"

    # --- non-greedy: path hit + non-degenerate acceptance (asserted, all) ---
    ng_runs, st_ng = _run(monkeypatch, method, target, True, greedy=False)
    ids_ng = ng_runs[0]
    assert all(ids_ng), f"{method}: empty non-greedy output"
    assert st_ng["kernel"] > 0 and st_ng["dispatch"] > 0, (
        f"{method}: rejection dispatch/kernel did not run"
    )
    accepts = st_ng["accepts"]
    assert accepts, f"{method}: no accepted-count samples captured"
    assert not all(a == 1 for a in accepts), f"{method}: degenerate accepted counts (all == 1)"
    assert not all(a == max_accept for a in accepts), (
        f"{method}: degenerate accepted counts (all == runtime_draft_len+1)"
    )
    # NOTE: a two-run output-token replay-equality is intentionally NOT asserted
    # (see module docstring): the forward kernels are not guaranteed
    # bitwise-reproducible across runs. The rejection MECHANISM's exact
    # determinism is proven by the committed per-step/block exact-provenance
    # tests (Rounds 14/24/25/29).

    # --- CUDA graph (asserted, all): replays a non-greedy rejection variant ---
    graph_runs, st_graph = _run(monkeypatch, method, target, True, greedy=False, cuda_graph=True)
    assert all(graph_runs[0]), f"{method}: empty CUDA-graph output"
    # Acceptance is captured inside the graph, so the Python kernel counter fires
    # during warmup capture; >0 confirms the rejection path was captured.
    assert st_graph["kernel"] > 0, f"{method}: rejection path not captured under CUDA graph"
    # Replay-key evidence: >=1 replayed key is non-greedy (is_all_greedy_sample
    # is False) -> the rejection variant was dispatched at replay, not eager.
    assert st_graph["replay_keys"], f"{method}: no CUDA-graph replay observed"
    assert any(k[-1] is False for k in st_graph["replay_keys"]), (
        f"{method}: no non-greedy (rejection) CUDA-graph key replayed"
    )
