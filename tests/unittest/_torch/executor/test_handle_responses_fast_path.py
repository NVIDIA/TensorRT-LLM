# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-step shortcuts of ``PyExecutor._handle_responses``."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def _request(request_id, *, streaming, finished=False, perf_metrics=False, decoding_iter=5):
    request = MagicMock()
    request.py_request_id = request_id
    request.is_attention_dp_dummy = False
    request.py_kv_transfer_timed_out = False
    request.is_generation_only_request = False
    request.py_draft_tokens = []
    request.py_decoding_iter = decoding_iter
    request.streaming = streaming
    request.is_finished = finished
    request.return_perf_metrics = perf_metrics
    request.create_response.return_value = None
    return request


def _executor(requests, *, stream_interval=1, pending_preemption=False):
    executor = object.__new__(PyExecutor)
    executor.active_requests = requests
    executor.perf_manager = MagicMock()
    executor.perf_manager.enabled = False
    executor.perf_manager.get_timestamp.return_value = None
    executor.iter_counter = 1
    executor.stream_interval = stream_interval
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.kv_cache_transceiver = None
    executor.enable_attention_dp = False
    executor._enqueue_responses = MagicMock()
    executor._is_kv_manager_v2 = True
    executor.kv_cache_manager = MagicMock()
    executor.kv_cache_manager.has_pending_preemption.return_value = pending_preemption
    executor.kv_cache_manager.is_preemption_pending.return_value = False
    return executor


def test_non_streaming_request_in_progress_creates_no_response():
    request = _request(1, streaming=False)
    PyExecutor._handle_responses(_executor([request]))
    request.create_response.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(streaming=False, finished=True),  # a finished request always reports
        dict(streaming=True),  # streaming reports each stream_interval step
        dict(streaming=False, perf_metrics=True),  # perf metrics finalize ctx totals
    ],
)
def test_response_still_created(kwargs):
    request = _request(1, **kwargs)
    PyExecutor._handle_responses(_executor([request]))
    request.create_response.assert_called_once()


def test_preemption_lookup_skipped_when_nothing_is_parked():
    requests = [_request(i, streaming=False) for i in range(3)]
    executor = _executor(requests, pending_preemption=False)
    PyExecutor._handle_responses(executor)
    executor.kv_cache_manager.is_preemption_pending.assert_not_called()


def test_parked_preemption_victim_is_kept_active():
    victim = _request(1, streaming=True, finished=True)
    executor = _executor([victim], pending_preemption=True)
    executor.kv_cache_manager.is_preemption_pending.return_value = True
    PyExecutor._handle_responses(executor)
    victim.create_response.assert_not_called()
    assert executor.active_requests == [victim]


def test_step_metrics_skipped_while_disabled_and_kept_when_enabled():
    executor = _executor([_request(1, streaming=False)])
    PyExecutor._handle_responses(executor)
    executor.perf_manager.append_step_metrics.assert_not_called()

    executor = _executor([_request(1, streaming=False)])
    executor.perf_manager.enabled = True
    PyExecutor._handle_responses(executor)
    executor.perf_manager.append_step_metrics.assert_called_once()
