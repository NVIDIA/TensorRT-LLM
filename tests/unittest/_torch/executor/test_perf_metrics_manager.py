# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from tensorrt_llm._torch.pyexecutor.perf_metrics_manager import PerfMetricsManager
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


class _FakeEvent:
    def __init__(self, elapsed_time_ms=0.0):
        self.elapsed_time_ms = elapsed_time_ms
        self.query_calls = 0
        self.synchronize_calls = 0
        self.elapsed_time_calls = 0

    def query(self):
        self.query_calls += 1
        return True

    def synchronize(self):
        self.synchronize_calls += 1

    def elapsed_time(self, _end_event):
        self.elapsed_time_calls += 1
        return self.elapsed_time_ms


def _make_request(events, forward_start_time):
    forward_start, forward_end, sample_end = events
    metric = {"gpu_forward_time": 0, "gpu_sample_time": 0}
    timing = SimpleNamespace(
        step_metrics=[metric],
        ctx_chunk_metrics=[],
        forward_start_time=forward_start_time,
        gpu_forward_start_event=forward_start,
        gpu_forward_end_event=forward_end,
        gpu_sample_end_event=sample_end,
        ctx_gpu_forward_time=None,
        ctx_gpu_sample_time=None,
    )
    return SimpleNamespace(py_perf_timing=timing), metric


def _make_emitting_request(request_id):
    request = MagicMock()
    request.py_request_id = request_id
    request.is_attention_dp_dummy = False
    request.py_kv_transfer_timed_out = False
    request.is_generation_only_request.return_value = False
    request.py_draft_tokens = []
    request.py_decoding_iter = 1
    request.return_perf_metrics = True
    request.is_finished = False
    request.create_response.return_value = None
    return request


def test_gpu_times_cache_reuses_events_across_singleton_calls():
    forward_start = _FakeEvent(elapsed_time_ms=1.25)
    forward_end = _FakeEvent(elapsed_time_ms=0.5)
    sample_end = _FakeEvent()
    events = (forward_start, forward_end, sample_end)
    first_request, first_metric = _make_request(events, forward_start_time=10.0)
    second_request, second_metric = _make_request(events, forward_start_time=10.0)
    manager = PerfMetricsManager(enabled=True)
    gpu_times_cache = {}

    manager.compute_batch_gpu_times([first_request], gpu_times_cache=gpu_times_cache)
    manager.compute_batch_gpu_times([second_request], gpu_times_cache=gpu_times_cache)

    assert first_metric == {"gpu_forward_time": 1.25, "gpu_sample_time": 0.5}
    assert second_metric == first_metric
    assert forward_end.query_calls == 1
    assert sample_end.query_calls == 1
    assert forward_start.elapsed_time_calls == 1
    assert forward_end.elapsed_time_calls == 1
    assert len(gpu_times_cache) == 1


def test_gpu_times_cache_distinguishes_reused_events_across_iterations():
    forward_start = _FakeEvent(elapsed_time_ms=1.25)
    forward_end = _FakeEvent(elapsed_time_ms=0.5)
    sample_end = _FakeEvent()
    events = (forward_start, forward_end, sample_end)
    first_request, _ = _make_request(events, forward_start_time=10.0)
    second_request, _ = _make_request(events, forward_start_time=12.0)
    manager = PerfMetricsManager(enabled=True)
    gpu_times_cache = {}

    manager.compute_batch_gpu_times([first_request], gpu_times_cache=gpu_times_cache)
    manager.compute_batch_gpu_times([second_request], gpu_times_cache=gpu_times_cache)

    assert forward_end.query_calls == 2
    assert sample_end.query_calls == 2
    assert forward_start.elapsed_time_calls == 2
    assert forward_end.elapsed_time_calls == 2
    assert len(gpu_times_cache) == 2


def test_handle_responses_shares_gpu_times_cache():
    requests = [_make_emitting_request(1), _make_emitting_request(2)]
    executor = object.__new__(PyExecutor)
    executor.active_requests = requests
    executor.perf_manager = MagicMock()
    executor.perf_manager.get_timestamp.return_value = 10.0
    executor.iter_counter = 1
    executor.stream_interval = 20
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.kv_cache_transceiver = None
    executor.enable_attention_dp = False
    executor._enqueue_responses = MagicMock()

    PyExecutor._handle_responses(executor)

    first_call, second_call = executor.perf_manager.compute_batch_gpu_times.call_args_list
    assert first_call == call([requests[0]], gpu_times_cache=first_call.kwargs["gpu_times_cache"])
    assert second_call == call([requests[1]], gpu_times_cache=second_call.kwargs["gpu_times_cache"])
    assert first_call.kwargs["gpu_times_cache"] is second_call.kwargs["gpu_times_cache"]
