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
"""Where the executor hands errors and its shutdown to the GPU memory monitor.

* A failed rematerialization is reported once, by ``materialize_with_tag``,
  whichever caller made it.
* An error reaches the clients, and a crashed rank arms its kill watchdog,
  before the out-of-memory report is built.
* Each monitor teardown step runs once per executor, and only the executor
  that recorded startup stops stack recording.
"""

import contextlib
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch import virtual_memory
from tensorrt_llm._torch.memory.monitor import tagging, triggers
from tensorrt_llm._torch.pyexecutor import py_executor as pe

pytestmark = pytest.mark.cpu_only

# What TLLM_CU_CHECK raises when cuMemCreate runs out of device memory.
_CU_MEM_CREATE_OOM = (
    "[TensorRT-LLM][ERROR] CUDA driver error in "
    "cuMemCreate(&handle, mSize, &mProp, 0): CUDA_ERROR_OUT_OF_MEMORY: "
    "out of memory."
)


class _FailingManager:
    """A virtual-memory manager whose materialize fails like cuMemCreate."""

    def materialize_with_tag(self, tag):
        raise RuntimeError(_CU_MEM_CREATE_OOM)


@pytest.fixture
def reported(monkeypatch):
    """Messages ``materialize_with_tag`` hands to the monitor."""
    seen = []
    monkeypatch.setattr(virtual_memory, "get_virtual_memory_manager", _FailingManager)
    monkeypatch.setattr(virtual_memory, "notify_error", seen.append)
    # A second report from the executor would land here.
    monkeypatch.setattr(pe.gpu_memory_triggers, "notify_error", seen.append)
    return seen


def test_failed_materialize_is_reported_and_reraised(reported):
    assert triggers.is_device_oom_message(_CU_MEM_CREATE_OOM)
    with pytest.raises(RuntimeError, match="CUDA_ERROR_OUT_OF_MEMORY"):
        virtual_memory.materialize_with_tag("kv_cache")
    assert reported == [_CU_MEM_CREATE_OOM]


def test_single_rank_wakeup_reports_a_failed_materialize(reported, monkeypatch):
    from tensorrt_llm.executor.base_worker import BaseWorker
    from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    engine = mock.MagicMock()
    engine._sleep_wakeup_lock = threading.Lock()
    engine.control_action = contextlib.nullcontext
    del engine.get_memory_status  # every requested tag is parked
    worker = object.__new__(BaseWorker)
    worker._backend = "pytorch"
    worker.llm_args = SimpleNamespace(
        sleep_config=object(), parallel_config=SimpleNamespace(world_size=1)
    )
    worker.engine = engine

    with pytest.raises(RuntimeError, match="CUDA_ERROR_OUT_OF_MEMORY"):
        worker.wakeup([ExecutorMemoryType.KV_CACHE.value])

    assert reported == [_CU_MEM_CREATE_OOM]
    engine.fail_sleep_wakeup_transition.assert_called_once()
    engine.abort_wakeup_transition.assert_called_once()


def test_sleep_listener_reports_a_failed_wakeup_once(reported, monkeypatch):
    """The listener thread of a rank > 0 catches the failure itself."""
    monkeypatch.setattr(pe, "set_thread_local_mpi_comm", lambda *_: None)
    monkeypatch.setattr(pe, "CUASSERT", lambda *_: None)
    monkeypatch.setattr(pe, "cudart", SimpleNamespace(cudaSetDevice=lambda *_: None))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)

    messages = [
        {"action": pe._SleepWakeupAction.WAKEUP, "tags": ["kv_cache"]},
        {"action": pe._SleepWakeupAction.SHUTDOWN},
    ]
    acks = []
    comm = SimpleNamespace(recv=lambda **_: messages.pop(0), send=lambda ack, **_: acks.append(ack))
    executor = object.__new__(pe.PyExecutor)
    executor.device_id = 0
    executor.dist = SimpleNamespace(rank=1)
    executor._sleep_wakeup_comm = comm
    executor.control_request_barrier = threading.Event()
    executor.control_request_barrier.set()
    executor.control_action_done = threading.Event()
    executor._active_control_id = None
    executor._run_mnnvl_checkpoint_resources = mock.Mock()

    executor._sleep_wakeup_listener_loop()

    assert acks[0]["status"] == "error"
    assert reported == [_CU_MEM_CREATE_OOM]


def _error_handling_executor(events, *, fatal, attention_dp=False, world_size=1):
    executor = mock.MagicMock()
    executor.enable_attention_dp = attention_dp
    executor.kv_connector_manager = None
    executor.dist = SimpleNamespace(rank=0, world_size=world_size)
    executor._fatal_error = RuntimeError("Fatal error: earlier") if fatal else None
    executor.waiting_queue = []
    executor.executor_request_queue.get_request_queue.return_value.empty.return_value = True
    executor.executor_request_queue.enqueue_shutdown_request.side_effect = lambda: events.append(
        "shutdown"
    )
    executor.active_requests = []
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._enqueue_responses.side_effect = lambda *_: events.append("responses")
    return executor


@pytest.mark.parametrize("fatal", [False, True])
def test_error_handler_reports_after_the_responses(monkeypatch, fatal):
    events = []
    monkeypatch.setattr(pe.gpu_memory_triggers, "notify_error", lambda m: events.append(m))
    executor = _error_handling_executor(events, fatal=fatal)

    pe.PyExecutor._handle_errors(executor, "CUDA error: out of memory", charge_budget=False)

    assert events[-1] == "CUDA error: out of memory", events
    assert "responses" in events[:-1], events
    assert events.count("CUDA error: out of memory") == 1


def test_rank_local_fatal_leaves_the_report_to_the_event_loop(monkeypatch):
    """The raise reaches _event_loop_wrapper, which arms the watchdog first."""
    events = []
    monkeypatch.setattr(pe.gpu_memory_triggers, "notify_error", lambda m: events.append(m))
    executor = _error_handling_executor(events, fatal=True, attention_dp=True, world_size=2)

    with pytest.raises(RuntimeError, match="Fatal error"):
        pe.PyExecutor._handle_errors(executor, "CUDA error: out of memory", charge_budget=False)

    assert "CUDA error: out of memory" not in events


def test_crashed_rank_arms_its_watchdog_before_the_report(monkeypatch):
    """The ERROR log of a crash is where the out-of-memory report is built."""
    events = []
    monkeypatch.setattr(pe, "host_profiler_context", lambda enable: contextlib.nullcontext())
    monkeypatch.setattr(pe, "customized_gc_thresholds", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(
        pe,
        "start_rank_crash_kill_watchdog",
        lambda world_size, *, error_delivered: events.append("watchdog"),
    )
    monkeypatch.setattr(pe, "hard_kill_on_rank_crash", lambda *_, **__: events.append("kill"))
    monkeypatch.setattr(pe.logger, "error", lambda message: events.append("report"))
    executor = object.__new__(pe.PyExecutor)
    executor.dist = SimpleNamespace(world_size=2)
    executor.garbage_collection_gen0_threshold = None
    executor._event_loop_error_delivered = threading.Event()
    executor._executor_loop_cleanup = lambda: events.append("cleanup")

    def crash():
        raise torch.OutOfMemoryError("CUDA out of memory")

    executor.event_loop = crash

    with pytest.raises(torch.OutOfMemoryError):
        executor._event_loop_wrapper()

    assert events.index("watchdog") < events.index("report"), events
    assert events.count("watchdog") == 1, events
    assert events[-2:] == ["cleanup", "kill"], events


@pytest.fixture
def teardown(monkeypatch):
    calls = mock.Mock()
    monkeypatch.setattr(pe.gpu_memory_triggers, "release_oom_reporter", calls.release_oom_reporter)
    monkeypatch.setattr(pe.gpu_memory_providers, "end_executor", calls.end_executor)
    monkeypatch.setattr(tagging, "stop_stack_recording", calls.stop_stack_recording)
    monkeypatch.setattr(tagging.startup, "clear", lambda: None)
    monkeypatch.setattr(tagging, "set_frozen_map", lambda _: None)
    return calls


def _monitored_executor(token):
    executor = object.__new__(pe.PyExecutor)
    executor._rank_reporter = mock.Mock()
    executor._oom_reporter_armed = True
    executor._memory_monitor_token = token
    return executor


def test_each_teardown_step_runs_once(teardown, monkeypatch):
    token = object()
    monkeypatch.setattr(tagging, "_startup_executor", token)
    executor = _monitored_executor(token)
    rank_reporter = executor._rank_reporter

    for _ in range(2):  # a retried shutdown repeats nothing
        executor._release_memory_monitor()

    rank_reporter.stop.assert_called_once_with()
    teardown.release_oom_reporter.assert_called_once_with(owner=token)
    teardown.end_executor.assert_called_once_with(token)
    teardown.stop_stack_recording.assert_called_once_with()
    assert executor._rank_reporter is None
    assert executor._oom_reporter_armed is False
    assert executor._memory_monitor_token is None


def test_only_the_startup_executor_stops_stack_recording(teardown, monkeypatch):
    first, second = object(), object()
    monkeypatch.setattr(tagging, "_startup_executor", first)

    _monitored_executor(second)._release_memory_monitor()
    teardown.stop_stack_recording.assert_not_called()

    _monitored_executor(first)._release_memory_monitor()
    teardown.stop_stack_recording.assert_called_once_with()


def test_unmonitored_executor_tears_down_nothing(teardown):
    executor = object.__new__(pe.PyExecutor)
    executor._rank_reporter = None
    executor._oom_reporter_armed = False
    executor._memory_monitor_token = None

    executor._release_memory_monitor()

    assert teardown.mock_calls == []
