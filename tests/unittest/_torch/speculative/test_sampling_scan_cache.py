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
"""TRTLLM_Q35_SAMPLING_SCAN_CACHE: the one-model sampling scan is reused for the same requests in the same
states / slots / draft length / group override with identical results, and redone otherwise."""

import types

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import SAMPLING_SCAN_CACHE_ENV, SpecMetadata

pytestmark = pytest.mark.cpu_only


class _Req:
    def __init__(self, rid, slot, temperature):
        self.py_request_id, self.py_seq_slot, self.temperature = rid, slot, temperature
        self.state = LlmRequestState.GENERATION_IN_PROGRESS
        self.reads = 0

    @property
    def sampling_config(self):
        self.reads += 1
        return types.SimpleNamespace(temperature=self.temperature, top_k=20, top_p=0.9, min_p=None)


def _scan(stub, requests):
    result = SpecMetadata._scan_one_model_sampling(stub, requests)
    return result, stub.is_all_greedy_sample


def _scan_stub(**overrides):
    fields = dict(
        runtime_draft_len=6,
        is_ragged_verify=False,
        group_all_greedy_sample=None,
        dummy_slot_row=8,
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _like(src):
    return _scan_stub(
        runtime_draft_len=src.runtime_draft_len,
        is_ragged_verify=src.is_ragged_verify,
        group_all_greedy_sample=src.group_all_greedy_sample,
        dummy_slot_row=src.dummy_slot_row,
    )


@pytest.mark.parametrize("cache", [False, True])
def test_scan_cache_same_results(monkeypatch, cache):
    monkeypatch.delenv(SAMPLING_SCAN_CACHE_ENV, raising=False)
    requests = [_Req(5, 2, 0.7), _Req(6, 0, 0.0)]
    expected = _scan(_scan_stub(), requests)
    if cache:
        monkeypatch.setenv(SAMPLING_SCAN_CACHE_ENV, "1")
    stub = _scan_stub()
    reads = [r.reads for r in requests]
    steps = []
    for _ in range(3):
        stub.is_all_greedy_sample = None  # clobbered between scans: restored by the cache
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
        fresh = _scan(_like(stub), requests)
        monkeypatch.setenv(SAMPLING_SCAN_CACHE_ENV, "1")
        assert result == fresh
        return requests[0].reads == before + 2  # one cached-arm rescan + the fresh scan

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


def test_scan_cache_skips_warmup_dummies_and_ragged(monkeypatch):
    """Warmup dummies reuse request ids across capture passes with different sampling params, and ragged
    verification changes per-request row counts: neither is cached."""
    monkeypatch.setenv(SAMPLING_SCAN_CACHE_ENV, "1")
    stub = _scan_stub()
    greedy = [_Req(0, 0, 0.0)]
    greedy[0].is_dummy_request = True
    assert _scan(stub, greedy)[1] is True
    sampled = [_Req(0, 0, 0.7)]
    sampled[0].is_dummy_request = True
    assert _scan(stub, sampled)[1] is False

    padding = [_Req(5, 2, 0.7), _Req(7, None, 0.0)]
    padding[1].is_dummy_request = padding[1].is_cuda_graph_dummy = True
    _scan(stub, padding)
    before = padding[0].reads
    _scan(stub, padding)  # the CUDA graph padding request does not prevent caching
    assert padding[0].reads == before

    ragged = _scan_stub(is_ragged_verify=True)
    requests = [_Req(5, 2, 0.7)]
    requests[0].py_verify_len = 3
    _scan(ragged, requests)
    before = requests[0].reads
    _scan(ragged, requests)
    assert requests[0].reads == before + 1
