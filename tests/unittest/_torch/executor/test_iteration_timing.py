# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import profiling

pytestmark = pytest.mark.cpu_only


class _Event:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.ready = True
        self.recorded_at = None
        self.records = 0
        self.queries = 0
        self.synchronizations = 0
        self.reads = 0

    def record(self) -> None:
        assert self.ready, "An incomplete event was reused"
        self.ready = False
        self.recorded_at = self.factory.loop_id
        self.records += 1

    def query(self) -> bool:
        self.queries += 1
        return self.ready

    def synchronize(self) -> None:
        assert self.factory.allow_synchronize, "Statistics waited for the GPU"
        self.synchronizations += 1
        self.factory.complete()

    def elapsed_time(self, end_event) -> float:
        assert self.ready and end_event.ready
        assert self.recorded_at <= end_event.recorded_at
        self.reads += 1
        return 10.0 + self.recorded_at


class _EventFactory:
    def __init__(self) -> None:
        self.events = []
        self.loop_id = 0
        self.allow_synchronize = False

    def __call__(self, *, enable_timing: bool):
        assert enable_timing
        event = _Event(self)
        self.events.append(event)
        return event

    def complete(self) -> None:
        for event in self.events:
            event.ready = True


@pytest.fixture
def events(monkeypatch):
    factory = _EventFactory()
    monkeypatch.setattr(profiling.torch.cuda, "Event", factory)
    return factory


def test_loop_timing_reports_only_the_previous_loop(events):
    timing = profiling._LoopTimingEvents()
    for loop_id in range(8):
        events.loop_id = loop_id
        timing.start(loop_id)
        events.complete()
        result = timing.finish(loop_id, synchronize=False)
        assert result == (None if loop_id == 0 else 9.0 + loop_id)
        events.complete()

    assert len(events.events) == 4
    assert sum(event.synchronizations for event in events.events) == 0
    assert sum(event.reads for event in events.events) == 7


def test_unready_timing_keeps_owned_events_and_bounds_the_backlog(events):
    timing = profiling._LoopTimingEvents()
    for loop_id in range(1000):
        events.loop_id = loop_id
        timing.start(loop_id)
        assert timing.finish(loop_id, synchronize=False) is None

    assert len(events.events) == 4
    assert [event.records for event in events.events] == [1, 1, 1, 1]
    assert sum(event.synchronizations for event in events.events) == 0
    assert sum(event.reads for event in events.events) == 0


def test_completed_stale_timing_is_reclaimed_without_mislabeling(events):
    timing = profiling._LoopTimingEvents()
    for loop_id in range(4):
        events.loop_id = loop_id
        timing.start(loop_id)
        assert timing.finish(loop_id, synchronize=False) is None

    events.complete()
    for loop_id in range(4, 6):
        events.loop_id = loop_id
        timing.start(loop_id)
        assert timing.finish(loop_id, synchronize=False) is None
        events.complete()

    events.loop_id = 6
    timing.start(6)
    events.complete()
    assert timing.finish(6, synchronize=False) == 15.0
    assert sum(event.reads for event in events.events) == 1
    assert len(events.events) == 4


class _Executor:
    def _should_capture_iter_gpu_timing(self) -> bool:
        return self.enable_iter_perf_stats


def _executor(*, stats: bool, print_log: bool):
    executor = _Executor()
    executor.iter_counter = 0
    executor.profile_start_iters = set()
    executor.profile_stop_iters = set()
    executor.is_warmup = False
    executor.enable_iter_perf_stats = stats
    executor.print_log = print_log
    executor._profile_state_lock = threading.Lock()
    executor.dist = SimpleNamespace(rank=1)
    executor.global_rank = 1
    executor._latest_host_step_time_ms = None
    executor._latest_prev_device_step_time_ms = None
    return executor


@pytest.mark.parametrize(
    "stats,print_log", [(False, False), (True, False), (False, True), (True, True)]
)
def test_profile_step_keeps_timing_modes_and_progress(events, monkeypatch, stats, print_log):
    executor = _executor(stats=stats, print_log=print_log)
    manager = profiling.PyExecutorProfileManager(executor)
    calibrator = Mock()
    monkeypatch.setattr(profiling, "get_calibrator", lambda: calibrator)
    monkeypatch.setattr(profiling, "get_global_profiler", lambda: None)
    monkeypatch.delenv(profiling.PROFILE_TRACE_ENV_VAR_NAME, raising=False)
    monkeypatch.setenv(profiling.PROFILE_LOG_RANKS_ENV_VAR_NAME, "0")
    events.allow_synchronize = print_log
    with manager.profile_step() as step:
        for loop_id in range(8):
            # A checkpoint closes the body begun by the previous call.
            events.loop_id = max(0, loop_id - 1)
            step()
            executor.iter_counter += 1

    assert executor.iter_counter == 8
    assert calibrator.pre_step.call_count == 8
    assert calibrator.post_step.call_count == 8
    if not (stats or print_log):
        assert events.events == []
        assert executor._latest_host_step_time_ms is None
    else:
        assert executor._latest_host_step_time_ms is not None
    if print_log:
        assert sum(event.synchronizations for event in events.events) == 6
        assert executor._latest_prev_device_step_time_ms is not None
    else:
        assert sum(event.synchronizations for event in events.events) == 0
        assert executor._latest_prev_device_step_time_ms is None


def test_profile_step_clears_last_duration_when_next_measurement_is_unready(events, monkeypatch):
    executor = _executor(stats=True, print_log=False)
    manager = profiling.PyExecutorProfileManager(executor)
    monkeypatch.setattr(profiling, "get_calibrator", Mock())
    monkeypatch.setattr(profiling, "get_global_profiler", lambda: None)
    monkeypatch.delenv(profiling.PROFILE_TRACE_ENV_VAR_NAME, raising=False)
    with manager.profile_step() as step:
        step()
        step()
        events.complete()
        events.loop_id = 1
        step()
        assert executor._latest_prev_device_step_time_ms == 10.0
        events.loop_id = 2
        step()
        assert executor._latest_prev_device_step_time_ms is None

    assert sum(event.synchronizations for event in events.events) == 0
