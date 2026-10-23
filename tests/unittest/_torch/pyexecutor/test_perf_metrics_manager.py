# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from threading import Lock
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.pyexecutor.perf_metrics_manager import PerfMetricsManager
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "enabled,rank,compiled,multimodal",
    [
        (True, 0, False, True),
        (False, 0, False, True),
        (True, 1, False, True),
        (True, 0, True, True),
        (True, 0, False, False),
    ],
)
def test_encoder_timing_callback_installation(enabled, rank, compiled, multimodal):
    model = MultimodalModelMixin() if multimodal else SimpleNamespace(_encoder_forward_context=None)
    model._encoder_forward_context = None
    executor = SimpleNamespace(
        model_engine=SimpleNamespace(model=SimpleNamespace(_orig_mod=model) if compiled else model),
        enable_iter_perf_stats=enabled,
        dist=SimpleNamespace(rank=rank),
        perf_manager=MagicMock(),
    )

    PyExecutor._setup_encoder_perf_metrics(executor)

    expected = (
        executor.perf_manager.record_encoder_forward
        if enabled and rank == 0 and multimodal
        else None
    )
    assert model._encoder_forward_context == expected


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


def test_encoder_timing_retains_unready_events_and_reports_each_call_once():
    manager = PerfMetricsManager(enabled=False)
    start, end = MagicMock(), MagicMock()
    start.elapsed_time.return_value = 2.5
    end.query.side_effect = [False, True]
    manager.borrow_forward_timing_events = MagicMock(return_value=(start, end))

    with manager.record_encoder_forward():
        start.record.assert_called_once_with()
        end.record.assert_not_called()
    end.record.assert_called_once_with()

    assert manager.take_encoder_gpu_times_ms() == []
    assert manager._forward_event_pool == []
    start.elapsed_time.assert_not_called()

    assert manager.take_encoder_gpu_times_ms() == [2.5]
    assert manager._forward_event_pool == [(start, end)]
    assert manager.take_encoder_gpu_times_ms() == []
    start.synchronize.assert_not_called()
    end.synchronize.assert_not_called()


def test_encoder_timing_discards_failed_events_without_reusing_them():
    manager = PerfMetricsManager(enabled=False)
    start, end = MagicMock(), MagicMock()
    end.query.return_value = True
    start.elapsed_time.side_effect = RuntimeError("invalid event")
    manager._encoder_events = [(start, end)]

    assert manager.take_encoder_gpu_times_ms() == []
    assert manager._encoder_events == []
    assert manager._forward_event_pool == []


def test_encoder_events_are_reused_between_downsampled_stats_records():
    manager = PerfMetricsManager(enabled=False)
    start, end = MagicMock(), MagicMock()
    start.elapsed_time.return_value = 2.5
    end.query.return_value = True
    manager._forward_event_pool = [(start, end)]

    for _ in range(3):
        with manager.record_encoder_forward():
            pass

    assert manager.take_encoder_gpu_times_ms() == [2.5, 2.5, 2.5]
    assert manager._forward_event_pool == [(start, end)]


@pytest.mark.parametrize("adp_rank", [None, 0, 1])
def test_encoder_samples_are_not_duplicated_by_adp_fanout(adp_rank, monkeypatch):
    monkeypatch.delenv("TLLM_METRICS_ALL_RANKS", raising=False)
    manager = MagicMock()
    manager.take_encoder_gpu_times_ms.return_value = [2.5]
    executor = SimpleNamespace(
        enable_attention_dp=True,
        disable_overlap_scheduler=False,
        dist=SimpleNamespace(tp_size=2, pp_size=1),
        perf_manager=manager,
        stats_lock=Lock(),
        stats=[],
        max_stats_len=100,
    )

    PyExecutor._append_iter_stats(executor, MagicMock(), attention_dp_rank=adp_rank)

    expected = None if adp_rank == 1 else {"gpuTimesMS": [2.5]}
    assert executor.stats[0][8] == expected
    assert manager.take_encoder_gpu_times_ms.call_count == (adp_rank != 1)


def _make_emitting_request(request_id):
    request = MagicMock()
    request.py_request_id = request_id
    request.is_attention_dp_dummy = False
    request.py_kv_transfer_timed_out = False
    request.is_generation_only_request = False
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
