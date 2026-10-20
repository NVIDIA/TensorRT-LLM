# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_SAMPLING_PARAMS_CACHE: one-model sampling-parameter refills are skipped only when the
buffers already hold this step's values (upstream 089a2573ac), and never with the flag off.
TRTLLM_Q35_SAMPLING_SCAN_CACHE: the one-model sampling scan is reused for the same requests in the same
states / slots / draft length / overrides with identical results and flags, and redone otherwise."""

import types

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import (SAMPLING_PARAMS_CACHE_ENV,
                                                       SAMPLING_SCAN_CACHE_ENV, SpecMetadata)

pytestmark = pytest.mark.cpu_only


def _stub():
    stub = types.SimpleNamespace(_sampling_params_signature=[None, None])
    stub.invalidate_sampling_params_cache = types.MethodType(
        SpecMetadata.invalidate_sampling_params_cache, stub)
    return stub


def _need(stub, normalized):
    return SpecMetadata._sampling_params_buffers_need_update(stub, normalized)


def test_flag_off_always_refills(monkeypatch):
    monkeypatch.delenv(SAMPLING_PARAMS_CACHE_ENV, raising=False)
    stub = _stub()
    step = [(1.0, 0, 1.0, 7)]
    assert _need(stub, step) == (True, True)
    assert _need(stub, step) == (True, True)
    assert stub._sampling_params_signature == [None, None]


def test_flag_on_skips_only_unchanged_groups(monkeypatch):
    monkeypatch.setenv(SAMPLING_PARAMS_CACHE_ENV, "1")
    stub = _stub()
    a, b = (1.0, 0, 1.0, 7), (0.7, 20, 0.9, 7)
    assert _need(stub, [a, b]) == (True, True)
    assert _need(stub, [a, b]) == (False, False)
    # A context row becoming a generation row changes only the expanded layout.
    assert _need(stub, [a, (0.7, 20, 0.9, 1)]) == (False, True)
    assert _need(stub, [a, b]) == (False, True)
    # Negative controls: reordered or changed values refill both groups.
    assert _need(stub, [b, a]) == (True, True)
    assert _need(stub, [b, (0.5, 0, 1.0, 7)]) == (True, True)
    # Reallocated buffers hold nothing recorded.
    stub.invalidate_sampling_params_cache()
    assert _need(stub, [b, (0.5, 0, 1.0, 7)]) == (True, True)


class _Req:
    def __init__(self, rid, slot, temperature):
        self.py_request_id, self.py_seq_slot, self.temperature = rid, slot, temperature
        self.state = LlmRequestState.GENERATION_IN_PROGRESS
        self.reads = 0

    @property
    def sampling_config(self):
        self.reads += 1
        return types.SimpleNamespace(temperature=[self.temperature], top_k=[20], top_p=[0.9])


def _scan(stub, requests):
    result = SpecMetadata._scan_one_model_sampling(stub, requests)
    return result, (stub.skip_temperature, stub.skip_top_k, stub.skip_top_p, stub.is_all_greedy_sample)


def _scan_stub():
    return types.SimpleNamespace(runtime_draft_len=6, group_all_greedy_sample=None, dummy_slot_row=8)


@pytest.mark.parametrize("cache", [False, True])
def test_scan_cache_same_results(monkeypatch, cache):
    if cache:
        monkeypatch.setenv(SAMPLING_SCAN_CACHE_ENV, "1")
    else:
        monkeypatch.delenv(SAMPLING_SCAN_CACHE_ENV, raising=False)
    stub, ref = _scan_stub(), _scan_stub()
    requests = [_Req(5, 2, 0.7), _Req(6, 0, 0.0)]
    monkeypatch.delenv(SAMPLING_SCAN_CACHE_ENV, raising=False)
    expected = _scan(ref, requests)
    if cache:
        monkeypatch.setenv(SAMPLING_SCAN_CACHE_ENV, "1")
    reads = [r.reads for r in requests]
    steps = []
    for _ in range(3):
        stub.is_all_greedy_sample = stub.skip_top_k = None  # clobbered between scans: restored by the cache
        steps.append(_scan(stub, requests))
    assert all(step == expected for step in steps)
    new_reads = [r.reads - n for r, n in zip(requests, reads)]
    assert new_reads == ([1, 1] if cache else [3, 3])


def test_scan_cache_rescans_on_changes(monkeypatch):
    monkeypatch.setenv(SAMPLING_SCAN_CACHE_ENV, "1")
    stub = _scan_stub()
    requests = [_Req(5, 2, 0.7), _Req(6, 0, 0.0)]
    _scan(stub, requests)

    def rescanned(change):
        before = requests[0].reads
        change()
        result = _scan(stub, requests)
        monkeypatch.delenv(SAMPLING_SCAN_CACHE_ENV)
        fresh = _scan(_scan_stub_like(stub), requests)
        monkeypatch.setenv(SAMPLING_SCAN_CACHE_ENV, "1")
        assert result == fresh
        return requests[0].reads == before + 2  # one cached-arm rescan + the fresh scan

    def _scan_stub_like(src):
        return types.SimpleNamespace(runtime_draft_len=src.runtime_draft_len,
                                     group_all_greedy_sample=src.group_all_greedy_sample,
                                     dummy_slot_row=src.dummy_slot_row)

    def to_context():
        requests[1].state = LlmRequestState.CONTEXT_INIT

    def move_slot():
        requests[0].py_seq_slot = 4

    def shorter_drafts():
        stub.runtime_draft_len = 3

    def group_override():
        stub.group_all_greedy_sample = True

    for change in (to_context, move_slot, shorter_drafts, group_override):
        assert rescanned(change), change.__name__
    before = requests[0].reads
    _scan(stub, requests)  # unchanged since the last scan: cached
    assert requests[0].reads == before
