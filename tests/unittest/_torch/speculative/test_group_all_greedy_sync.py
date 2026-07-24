# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the group-synchronized ``is_all_greedy_sample`` override.

Under ADP + LM-head TP with rejection sampling, the greedy-vs-advanced path
choice gates group collectives, so the model engine all-gathers the per-rank
flags and stores the group AND in ``SpecMetadata.group_all_greedy_sample``;
``_scan_one_model_sampling`` must then re-apply it on every rescan (populate
runs after the CUDA graph key is built and would otherwise resurrect the
rank-local value).

These tests call ``_scan_one_model_sampling`` unbound on a SimpleNamespace
stand-in, mirroring test_rejection_buffers_guard.py, so no GPU or full
SpecMetadata construction is needed.
"""

import types

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import SpecMetadata


def _fake_request(temperature=None, top_k=None, top_p=None, min_p=None, slot=0):
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=[temperature] if temperature is not None else None,
            top_k=[top_k] if top_k is not None else None,
            top_p=[top_p] if top_p is not None else None,
            min_p=[min_p] if min_p is not None else None,
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=slot,
    )


def _fake_meta(group_all_greedy_sample=None, force_capture=False):
    return types.SimpleNamespace(
        runtime_draft_len=2,
        dummy_slot_row=0,
        group_all_greedy_sample=group_all_greedy_sample,
        _force_non_greedy_for_capture=force_capture,
    )


def _scan(meta, requests):
    return SpecMetadata._scan_one_model_sampling(meta, requests)


def test_local_value_used_when_no_group_sync():
    meta = _fake_meta(group_all_greedy_sample=None)
    _scan(meta, [_fake_request(), _fake_request()])
    assert meta.is_all_greedy_sample is True

    _scan(meta, [_fake_request(), _fake_request(temperature=0.8)])
    assert meta.is_all_greedy_sample is False


def test_group_override_pulls_greedy_rank_onto_advanced_path():
    # This rank's batch is all-greedy, but another rank in the LM-head-TP
    # group has a sampling request: the group AND (False) must win so the
    # whole group takes the advanced path together.
    meta = _fake_meta(group_all_greedy_sample=False)
    _scan(meta, [_fake_request(), _fake_request()])
    assert meta.is_all_greedy_sample is False


def test_group_override_survives_rescan():
    # populate_sampling_params_for_one_model rescans after the CUDA graph key
    # is built; the override must keep applying so the key, the buffers, and
    # the worker branches all agree.
    meta = _fake_meta(group_all_greedy_sample=False)
    for _ in range(3):
        _scan(meta, [_fake_request()])
        assert meta.is_all_greedy_sample is False


def test_group_override_true_keeps_greedy():
    meta = _fake_meta(group_all_greedy_sample=True)
    _scan(meta, [_fake_request()])
    assert meta.is_all_greedy_sample is True


def test_capture_override_composes_with_group_sync():
    # Warmup forces the advanced variant to capture its CUDA graph; the group
    # value is derived from capture-forced locals (all False), so the final
    # flag stays False regardless of composition order.
    meta = _fake_meta(group_all_greedy_sample=False, force_capture=True)
    _scan(meta, [_fake_request()])
    assert meta.is_all_greedy_sample is False


def test_min_p_only_request_is_not_greedy():
    # A min_p-only request (no temperature/top_k/top_p) must take the advanced
    # sampling path, not the argmax greedy fast-path, and enable the min_p filter
    # while the other filters stay disabled.
    meta = _fake_meta(group_all_greedy_sample=None)
    _scan(meta, [_fake_request(min_p=0.1)])
    assert meta.is_all_greedy_sample is False
    assert meta.skip_min_p is False
    assert meta.skip_temperature is True
    assert meta.skip_top_k is True
    assert meta.skip_top_p is True


def test_min_p_zero_stays_greedy():
    # min_p == 0 disables min_p; an otherwise-unset request stays greedy.
    meta = _fake_meta(group_all_greedy_sample=None)
    _scan(meta, [_fake_request(min_p=0.0)])
    assert meta.is_all_greedy_sample is True
    assert meta.skip_min_p is True
