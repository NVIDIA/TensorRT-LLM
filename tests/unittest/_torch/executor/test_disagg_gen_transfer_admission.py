# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated generation admission factor (TRTLLM_DISAGG_GEN_ADMISSION_FACTOR).

A generation server counts requests whose KV cache is still being received
against max_num_active_requests. With a factor f > 1 a non-attention-DP server
that has seen a generation-only request admits floor(f * max_num_active_requests)
requests, so the next requests' KV transfers overlap the current decode, and the
overlap sequence-slot headroom is enabled outside attention-DP.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor._util import should_enable_disagg_adp_overlap_headroom
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import (
    DISAGG_GEN_ADMISSION_FACTOR_ENV,
    disagg_gen_admission_factor,
)
from tensorrt_llm.bindings.executor import RequestType
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping


class _FakeExecutor:
    def __init__(self, factor: float, enable_attention_dp: bool = False, tp_size: int = 1):
        self.max_num_active_requests = 4
        self.enable_attention_dp = enable_attention_dp
        self.dist = Mock()
        self.dist.tp_size = tp_size
        self._disagg_gen_admission_factor = factor
        self._is_disagg_gen_server = False
        self.reverted = None

    def _revert_deferred_disagg_gen_init_alloc(self, candidates, admitted_requests):
        self.reverted = [r for r in candidates if r not in admitted_requests]

    _get_request_admission_capacity = PyExecutor._get_request_admission_capacity
    _note_disagg_gen_server = PyExecutor._note_disagg_gen_server
    _cap_disagg_gen_init_batch = PyExecutor._cap_disagg_gen_init_batch


class _FakeQueue:
    def __init__(self, request_types):
        self._items = [
            SimpleNamespace(request=SimpleNamespace(request_type=t)) for t in request_types
        ]

    def __bool__(self):
        return bool(self._items)

    def peek_request(self):
        return self._items[0]


@pytest.mark.parametrize(
    "value,expected",
    [(None, 1.0), ("", 1.0), ("1", 1.0), ("1.5", 1.5), ("2", 2.0), ("3", 2.0), ("0.5", 1.0)],
)
def test_factor_parsing_and_clamp(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(DISAGG_GEN_ADMISSION_FACTOR_ENV, raising=False)
    else:
        monkeypatch.setenv(DISAGG_GEN_ADMISSION_FACTOR_ENV, value)
    assert disagg_gen_admission_factor() == expected


@pytest.mark.parametrize("value", ["abc", "nan", "inf"])
def test_invalid_factor_raises(monkeypatch, value):
    monkeypatch.setenv(DISAGG_GEN_ADMISSION_FACTOR_ENV, value)
    with pytest.raises(ValueError):
        disagg_gen_admission_factor()


@pytest.mark.parametrize(
    "factor,head_type,expected",
    [
        (2.0, RequestType.REQUEST_TYPE_GENERATION_ONLY, 8),
        (1.5, RequestType.REQUEST_TYPE_GENERATION_ONLY, 6),
        (1.3, RequestType.REQUEST_TYPE_GENERATION_ONLY, 5),
        (1.0, RequestType.REQUEST_TYPE_GENERATION_ONLY, 4),
        (2.0, RequestType.REQUEST_TYPE_CONTEXT_ONLY, 4),
        (2.0, RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION, 4),
    ],
)
def test_capacity_scales_only_on_a_generation_server(factor, head_type, expected):
    executor = _FakeExecutor(factor)
    assert executor._get_request_admission_capacity() == 4
    executor._note_disagg_gen_server(_FakeQueue([head_type]))
    assert executor._get_request_admission_capacity() == expected


def test_generation_server_detection_is_sticky():
    executor = _FakeExecutor(2.0)
    executor._note_disagg_gen_server(_FakeQueue([]))
    assert executor._get_request_admission_capacity() == 4
    executor._note_disagg_gen_server(_FakeQueue([RequestType.REQUEST_TYPE_GENERATION_ONLY]))
    executor._note_disagg_gen_server(_FakeQueue([RequestType.REQUEST_TYPE_CONTEXT_ONLY]))
    assert executor._get_request_admission_capacity() == 8


def test_attention_dp_capacity_unchanged():
    executor = _FakeExecutor(2.0, enable_attention_dp=True, tp_size=2)
    executor._is_disagg_gen_server = True
    assert executor._get_request_admission_capacity() == 8


def test_executor_without_the_attributes_keeps_the_base_capacity():
    executor = _FakeExecutor(2.0)
    del executor._disagg_gen_admission_factor
    del executor._is_disagg_gen_server
    executor._note_disagg_gen_server(_FakeQueue([RequestType.REQUEST_TYPE_GENERATION_ONLY]))
    assert executor._get_request_admission_capacity() == 4


@pytest.mark.parametrize(
    "factor,num_admitted,expected_kept", [(2.0, 6, 4), (2.0, 3, 3), (1.5, 6, 4), (1.0, 6, 6)]
)
def test_transfers_started_per_iteration_are_capped(factor, num_admitted, expected_kept):
    executor = _FakeExecutor(factor)
    admitted = [object() for _ in range(num_admitted)]
    kept = executor._cap_disagg_gen_init_batch(admitted)
    assert kept == admitted[:expected_kept]
    if num_admitted > expected_kept:
        assert executor.reverted == admitted[expected_kept:]
    else:
        assert executor.reverted is None


@pytest.mark.parametrize(
    "value,is_disagg,pp_size,disable_overlap,expected",
    [
        ("2", True, 1, False, True),
        ("1.5", True, 1, False, True),
        ("1", True, 1, False, False),
        (None, True, 1, False, False),
        ("2", False, 1, False, False),
        ("2", True, 2, False, False),
        ("2", True, 1, True, False),
    ],
)
def test_overlap_slot_headroom_outside_attention_dp(
    monkeypatch, value, is_disagg, pp_size, disable_overlap, expected
):
    if value is None:
        monkeypatch.delenv(DISAGG_GEN_ADMISSION_FACTOR_ENV, raising=False)
    else:
        monkeypatch.setenv(DISAGG_GEN_ADMISSION_FACTOR_ENV, value)
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size, enable_attention_dp=False)
    cache_config = CacheTransceiverConfig(backend="NIXL") if is_disagg else None
    assert (
        should_enable_disagg_adp_overlap_headroom(mapping, cache_config, disable_overlap)
        is expected
    )
