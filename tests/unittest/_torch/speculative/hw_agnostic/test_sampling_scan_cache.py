# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpecMetadata._scan_one_model_sampling reuses its last result for an unchanged batch and rescans otherwise."""

import types

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import SpecMetadata

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


def _metadata(**overrides):
    fields = dict(
        runtime_draft_len=6,
        is_ragged_verify=False,
        group_all_greedy_sample=None,
        dummy_slot_row=8,
        _sampling_scan_cache=None,
        temperature_only_graphs=False,
        use_rejection_sampling=False,
        enable_penalty=False,
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _scan(metadata, requests):
    result = SpecMetadata._scan_one_model_sampling(metadata, requests)
    return result, metadata.is_all_greedy_sample


def _rescanned(metadata, requests):
    before = requests[0].reads
    result = _scan(metadata, requests)
    fresh = _metadata(
        runtime_draft_len=metadata.runtime_draft_len,
        is_ragged_verify=metadata.is_ragged_verify,
        group_all_greedy_sample=metadata.group_all_greedy_sample,
    )
    assert result == _scan(fresh, requests)
    return requests[0].reads > before + 1


def test_scan_cache_reuses_and_rescans_on_changes():
    metadata = _metadata()
    requests = [_Req(5, 2, 0.7), _Req(6, 0, 0.0)]
    expected = _scan(metadata, requests)
    metadata.is_all_greedy_sample = None  # clobbered between scans: restored from the cache
    assert _scan(metadata, requests) == expected
    assert not _rescanned(metadata, requests)

    def to_context():
        requests[1].state = LlmRequestState.CONTEXT_INIT

    def move_slot():
        requests[0].py_seq_slot = 4

    def shorter_drafts():
        metadata.runtime_draft_len = 3

    def group_override():
        metadata.group_all_greedy_sample = True

    for change in (to_context, move_slot, shorter_drafts, group_override):
        change()
        assert _rescanned(metadata, requests), change.__name__


def test_scan_cache_skips_warmup_dummies_and_ragged():
    # Warmup dummies reuse request ids across capture passes with different sampling params.
    metadata = _metadata()
    for temperature, greedy in ((0.0, True), (0.7, False)):
        dummy = _Req(0, 0, temperature)
        dummy.is_dummy_request = True
        assert _scan(metadata, [dummy])[1] is greedy
    # The CUDA graph padding request does not prevent caching.
    padding = [_Req(5, 2, 0.7), _Req(7, None, 0.0)]
    padding[1].is_dummy_request = padding[1].is_cuda_graph_dummy = True
    _scan(metadata, padding)
    assert not _rescanned(metadata, padding)
    # Ragged verification changes per-request row counts between steps.
    ragged = _metadata(is_ragged_verify=True)
    requests = [_Req(5, 2, 0.7)]
    requests[0].py_verify_len = 3
    _scan(ragged, requests)
    assert _rescanned(ragged, requests)
