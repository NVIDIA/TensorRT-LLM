# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import PerfTimingInfo
from tensorrt_llm._torch.pyexecutor.perf_metrics_manager import PerfMetricsManager

pytestmark = pytest.mark.cpu_only


class _FakeEvent:
    def __init__(self, elapsed_time_ms=0.0, ready=True):
        self.elapsed_time_ms = elapsed_time_ms
        self.ready = ready
        self.query_calls = 0
        self.synchronize_calls = 0
        self.elapsed_time_calls = 0

    def query(self):
        self.query_calls += 1
        return self.ready

    def synchronize(self):
        # Blocking on a CUDA event costs the executor its host run-ahead under
        # the overlap scheduler, so the instrumentation must never do it.
        self.synchronize_calls += 1
        raise AssertionError("perf instrumentation must not synchronize on a CUDA event")

    def elapsed_time(self, _end_event):
        self.elapsed_time_calls += 1
        return self.elapsed_time_ms


def _make_request(generation_only=True, context_remaining_length=0):
    return SimpleNamespace(
        py_perf_timing=PerfTimingInfo(),
        py_decoding_iter=1,
        is_generation_only_request=generation_only,
        context_remaining_length=context_remaining_length,
    )


def _run_iteration(manager, requests, events, forward_start_time):
    """Drive one executor iteration's worth of instrumentation calls."""
    forward_end_time = forward_start_time + 0.4
    manager.save_timing_to_requests(
        requests,
        *events,
        forward_start_time,
        forward_end_time,
        forward_end_time,
        forward_end_time + 0.01,
    )
    for request in requests:
        manager.append_step_metrics(
            request, 0, batch_token_time=forward_end_time + 0.02
        )


def test_batch_shares_one_step_entry():
    """The whole batch records one entry per iteration, not one per request.

    Every field of a step entry is batch-level, so allocating a dict per request
    is pure overhead on the executor's critical path.
    """
    events = (_FakeEvent(elapsed_time_ms=1.25), _FakeEvent(elapsed_time_ms=0.5), _FakeEvent())
    manager = PerfMetricsManager(enabled=True)
    requests = [_make_request() for _ in range(4)]

    _run_iteration(manager, requests, events, forward_start_time=10.0)
    manager.compute_batch_gpu_times(requests)

    entries = [request.py_perf_timing.step_metrics for request in requests]
    assert all(len(entry) == 1 for entry in entries)
    assert all(entry[0] is entries[0][0] for entry in entries)
    assert entries[0][0]["gpu_forward_time"] == 1.25
    assert entries[0][0]["gpu_sample_time"] == 0.5
    assert entries[0][0]["forward_start_time"] == 10.0


def test_gpu_times_read_once_per_batch_not_per_request():
    """elapsed_time() is read once per batch even when called per request.

    _handle_responses calls compute_batch_gpu_times([request]) inside its
    response loop, so per-request calls are the common case.
    """
    forward_start, forward_end, sample_end = events = (
        _FakeEvent(elapsed_time_ms=1.25),
        _FakeEvent(elapsed_time_ms=0.5),
        _FakeEvent(),
    )
    manager = PerfMetricsManager(enabled=True)
    requests = [_make_request() for _ in range(3)]

    _run_iteration(manager, requests, events, forward_start_time=10.0)
    for request in requests:
        manager.compute_batch_gpu_times([request])

    assert forward_start.elapsed_time_calls == 1
    assert forward_end.elapsed_time_calls == 1
    assert forward_end.query_calls == 1
    assert all(
        request.py_perf_timing.step_metrics[0]["gpu_forward_time"] == 1.25
        for request in requests
    )


def test_reused_events_with_new_timings_are_a_new_batch():
    """Ping-pong events are reused, so a new iteration must still be read afresh."""
    forward_start, forward_end, sample_end = events = (
        _FakeEvent(elapsed_time_ms=1.25),
        _FakeEvent(elapsed_time_ms=0.5),
        _FakeEvent(),
    )
    manager = PerfMetricsManager(enabled=True)
    request = _make_request()

    for forward_start_time in (10.0, 12.0):
        _run_iteration(manager, [request], events, forward_start_time)
        manager.compute_batch_gpu_times([request])
        request.py_decoding_iter += 1

    step_metrics = request.py_perf_timing.step_metrics
    assert len(step_metrics) == 2
    assert step_metrics[0] is not step_metrics[1]
    assert forward_start.elapsed_time_calls == 2
    assert forward_end.query_calls == 2


def test_unscheduled_and_duplicate_appends_are_ignored():
    manager = PerfMetricsManager(enabled=True)
    scheduled, unscheduled = _make_request(), _make_request()
    events = (_FakeEvent(elapsed_time_ms=3.0), _FakeEvent(elapsed_time_ms=0.5), _FakeEvent())

    manager.save_timing_to_requests([scheduled, unscheduled], *events, 20.0, 20.4, 20.4, 20.5)
    manager.append_step_metrics(scheduled, 0, batch_token_time=20.6)
    manager.append_step_metrics(scheduled, 0, batch_token_time=20.6)
    manager.compute_batch_gpu_times([scheduled, unscheduled])

    assert len(scheduled.py_perf_timing.step_metrics) == 1
    assert unscheduled.py_perf_timing.step_metrics == []


def test_step_iter_base_recovers_absolute_iteration():
    """step_iter_base + index replaces a per-entry "iter" field."""
    manager = PerfMetricsManager(enabled=True)
    request = _make_request()
    request.py_decoding_iter = 7
    events = (_FakeEvent(elapsed_time_ms=1.0), _FakeEvent(elapsed_time_ms=0.1), _FakeEvent())

    for step, forward_start_time in enumerate((30.0, 31.0, 32.0)):
        _run_iteration(manager, [request], events, forward_start_time)
        manager.compute_batch_gpu_times([request])
        request.py_decoding_iter += 1

    perf = request.py_perf_timing
    assert perf.step_iter_base == 7
    assert [perf.step_iter_base + i for i in range(len(perf.step_metrics))] == [7, 8, 9]


def test_ctx_gpu_totals_accumulate_once_per_chunk_per_request():
    """Each request's context totals cover every chunk exactly once.

    The chunk entry is shared by the batch, so "has this chunk been counted"
    is tracked per request rather than by probing the entry's gpu_forward_time.
    """
    manager = PerfMetricsManager(enabled=True)
    requests = [_make_request(generation_only=False) for _ in range(2)]
    chunks = ((1.0, 0.1), (2.0, 0.2), (4.0, 0.4))

    for index, (gpu_forward_time, gpu_sample_time) in enumerate(chunks):
        is_last = index == len(chunks) - 1
        events = (
            _FakeEvent(elapsed_time_ms=gpu_forward_time),
            _FakeEvent(elapsed_time_ms=gpu_sample_time),
            _FakeEvent(),
        )
        for request in requests:
            request.py_decoding_iter = 1 if is_last else 0
            request.context_remaining_length = 0 if is_last else 10
        _run_iteration(manager, requests, events, forward_start_time=40.0 + index)
        # Per-request calls, as _handle_responses does.
        for request in requests:
            manager.compute_batch_gpu_times([request])

    for request in requests:
        perf = request.py_perf_timing
        assert len(perf.ctx_chunk_metrics) == len(chunks)
        assert perf.step_metrics == []
        assert perf.ctx_gpu_forward_time == pytest.approx(sum(f for f, _ in chunks))
        assert perf.ctx_gpu_sample_time == pytest.approx(sum(s for _, s in chunks))
        assert perf.ctx_chunks_complete


def test_unready_event_is_retried_instead_of_blocking():
    """A CUDA event still in flight is read on a later call, never awaited."""
    manager = PerfMetricsManager(enabled=True)
    request = _make_request(generation_only=False)
    forward_start = _FakeEvent(elapsed_time_ms=5.0)
    forward_end = _FakeEvent(elapsed_time_ms=0.5, ready=False)
    events = (forward_start, forward_end, _FakeEvent())

    _run_iteration(manager, [request], events, forward_start_time=50.0)
    manager.compute_batch_gpu_times([request])

    perf = request.py_perf_timing
    entry = perf.ctx_chunk_metrics[0]
    assert entry["gpu_forward_time"] == 0
    assert perf.ctx_gpu_forward_time is None
    assert forward_end.synchronize_calls == 0

    forward_end.ready = True
    manager.compute_batch_gpu_times([])

    assert entry["gpu_forward_time"] == 5.0
    # The chunk's contribution to the context totals survives the deferral.
    assert perf.ctx_gpu_forward_time == pytest.approx(5.0)


def test_deferred_reads_are_bounded():
    """An event that never completes is dropped rather than retained forever."""
    manager = PerfMetricsManager(enabled=True)
    request = _make_request()
    events = (_FakeEvent(elapsed_time_ms=1.0), _FakeEvent(ready=False), _FakeEvent())

    _run_iteration(manager, [request], events, forward_start_time=60.0)
    manager.compute_batch_gpu_times([request])
    for _ in range(PerfMetricsManager._MAX_DEFERRED_GPU_READS + 1):
        manager.compute_batch_gpu_times([])

    assert manager._deferred_gpu_reads == []


def test_disabled_manager_records_nothing():
    manager = PerfMetricsManager(enabled=False)
    request = _make_request()

    manager.append_step_metrics(request, 0, batch_token_time=70.0)
    manager.compute_batch_gpu_times([request])

    assert request.py_perf_timing.step_metrics == []
