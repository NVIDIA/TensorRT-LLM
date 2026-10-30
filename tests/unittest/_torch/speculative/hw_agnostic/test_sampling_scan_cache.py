# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpecMetadata._scan_one_model_sampling reuses its last result for an unchanged batch and rescans otherwise,
and sets the temperature-only graph key bit only for batches the sharded Gumbel-max sampler can draw."""

import types

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import SpecMetadata

pytestmark = pytest.mark.cpu_only


class _Req:
    def __init__(self, rid, slot, temperature, **params):
        self.py_request_id, self.py_seq_slot, self.temperature = rid, slot, temperature
        self.params = dict(top_k=20, top_p=0.9, min_p=None, seed=None) | params
        self.state = LlmRequestState.GENERATION_IN_PROGRESS
        self.reads = 0

    @property
    def sampling_config(self):
        self.reads += 1
        return types.SimpleNamespace(temperature=self.temperature, **self.params)


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


_TEMPERATURE_ONLY = dict(top_k=None, top_p=None)


def _temperature_only(requests):
    metadata = _metadata(temperature_only_graphs=True)
    _scan(metadata, requests)
    first = metadata.temperature_only
    metadata.temperature_only = None  # clobbered between scans: restored from the cache
    _scan(metadata, requests)
    assert metadata.temperature_only is first
    return first


@pytest.mark.parametrize(
    "params,expected",
    [
        (_TEMPERATURE_ONLY, True),
        (dict(top_k=0, top_p=1.0, min_p=0.0), True),  # filters given but disabled
        (dict(top_k=20, top_p=None), False),
        (dict(top_k=None, top_p=0.9), False),
        (dict(_TEMPERATURE_ONLY, min_p=0.05), False),
        (dict(_TEMPERATURE_ONLY, seed=7), False),
        (dict(_TEMPERATURE_ONLY, seed=2**64 - 1), False),
    ],
)
def test_temperature_only_takes_unfiltered_unseeded_sampled_rows(params, expected):
    # Greedy rows stay greedy on the sharded path whatever their filters or seed.
    greedy = [_Req(6, 0, 0.0, seed=3), _Req(7, 1, 0.7, top_k=1, top_p=None, seed=3)]
    assert _temperature_only([_Req(5, 2, 0.7, **params), *greedy]) is expected
    # One filtered or seeded sampled row keeps the whole batch on the stock sampler.
    other = _Req(8, 3, 1.3, **_TEMPERATURE_ONLY)
    assert _temperature_only([other, _Req(5, 2, 0.7, **params)]) is expected


def test_temperature_only_min_p_alone_samples_off_the_sharded_path():
    # No temperature, top-k or top-p, but an active min-p: a sampled row (temperature 1) with a filter.
    assert _temperature_only([_Req(5, 2, None, top_k=None, top_p=None, min_p=0.1)]) is False


@pytest.mark.parametrize(
    "overrides",
    [
        dict(temperature_only_graphs=False),
        dict(use_rejection_sampling=True),
        dict(enable_penalty=True),
        dict(group_all_greedy_sample=True),
    ],
)
def test_temperature_only_batch_gates(overrides):
    requests = [_Req(5, 2, 0.7, **_TEMPERATURE_ONLY)]
    assert _temperature_only(requests) is True
    metadata = _metadata(**(dict(temperature_only_graphs=True) | overrides))
    _scan(metadata, requests)
    assert metadata.temperature_only is False


def test_temperature_only_needs_a_sampled_row():
    assert _temperature_only([_Req(5, 2, 0.0), _Req(6, 0, 0.0, seed=3)]) is False
