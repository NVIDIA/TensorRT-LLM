# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the one-model sampling-scan per-request normalization cache.

``_scan_one_model_sampling`` normalizes each request's (temperature, top_k,
top_p) into a ``(temperature, top_k, top_p, is_greedy)`` tuple. Those params are
fixed for a request's lifetime, so the tuple is cached on
``request.py_one_model_norm_sampling`` and reused by both per-step scans
(``update_is_all_greedy_sample`` and ``populate_sampling_params_for_one_model``)
instead of being recomputed every step. The per-step, state-dependent quantities
(``num_tokens`` and the batch-level flags) are always recomputed fresh -- never
cached.

Exercised unbound: the scan reads only ``self.runtime_draft_len``,
``self.dummy_slot_row`` and ``self.group_all_greedy_sample`` and writes
``self.is_all_greedy_sample``, so no real ``SpecMetadata`` or GPU buffers are
constructed.
"""

from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import SpecMetadata


def _make_self(runtime_draft_len: int = 2):
    return SimpleNamespace(
        runtime_draft_len=runtime_draft_len,
        dummy_slot_row=0,
        group_all_greedy_sample=None,
    )


def _make_request(
    *,
    temperature=None,
    top_k=None,
    top_p=None,
    seq_slot: int = 0,
    state=LlmRequestState.GENERATION_IN_PROGRESS,
):
    return SimpleNamespace(
        sampling_config=SimpleNamespace(temperature=temperature, top_k=top_k, top_p=top_p),
        state=state,
        py_seq_slot=seq_slot,
    )


def test_scan_caches_normalization_on_request():
    req = _make_request()  # greedy (all params None)
    assert not hasattr(req, "py_one_model_norm_sampling")

    SpecMetadata._scan_one_model_sampling(_make_self(), [req])

    norm = req.py_one_model_norm_sampling
    assert isinstance(norm, tuple) and len(norm) == 4
    assert norm[3] is True  # is_greedy


def test_scan_reuses_cached_normalization():
    # A greedy request, but pre-seed the cache with a NON-greedy tuple. If the
    # scan recomputed from sampling_config it would classify the row greedy;
    # reuse means the cached non-greedy tuple wins, and both the batch flag and
    # the emitted per-request row come from the cache.
    req = _make_request()
    req.py_one_model_norm_sampling = (0.7, 50, 0.9, False)
    self_ = _make_self()

    per_request_normalized, _ = SpecMetadata._scan_one_model_sampling(self_, [req])

    assert self_.is_all_greedy_sample is False
    assert per_request_normalized[0][:3] == (0.7, 50, 0.9)


def test_scan_computes_when_no_cache_nongreedy():
    req = _make_request(temperature=0.7, top_k=50, top_p=0.9)
    self_ = _make_self()

    SpecMetadata._scan_one_model_sampling(self_, [req])

    assert req.py_one_model_norm_sampling[3] is False  # non-greedy
    assert self_.is_all_greedy_sample is False


def test_num_tokens_recomputed_not_cached():
    # The same request scanned twice with a different state: num_tokens tracks
    # state each call (generation -> 1 + runtime_draft_len; context -> 1),
    # proving it is not frozen into the per-request normalization cache.
    req = _make_request(state=LlmRequestState.GENERATION_IN_PROGRESS)
    self_ = _make_self(runtime_draft_len=2)

    gen_norm, _ = SpecMetadata._scan_one_model_sampling(self_, [req])
    assert gen_norm[0][3] == 3  # 1 + runtime_draft_len
    cached = req.py_one_model_norm_sampling

    req.state = LlmRequestState.CONTEXT_INIT
    ctx_norm, _ = SpecMetadata._scan_one_model_sampling(self_, [req])
    assert ctx_norm[0][3] == 1  # context request: single token
    # The cached normalization tuple itself is unchanged across the two calls.
    assert req.py_one_model_norm_sampling is cached


def test_dummy_request_routes_to_scratch_slot_row():
    # Caching must not disturb the slot-id pass: a dummy/padding request
    # (py_seq_slot None) still routes to the scratch row.
    self_ = _make_self()
    self_.dummy_slot_row = 7
    requests = [_make_request(seq_slot=3), _make_request(seq_slot=None)]

    _, per_request_slot_ids = SpecMetadata._scan_one_model_sampling(self_, requests)

    assert per_request_slot_ids == [3, 7]
