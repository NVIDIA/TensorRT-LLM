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
* Each monitor teardown step runs once per executor, on every way an
  executor ends, and only the monitored executor stops stack recording.
* Only the serving executor gets a rank reporter, and only when MPI allows
  calls from several threads; a KV pool leaves the report before it is freed.
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
    monkeypatch.setattr(
        virtual_memory.monitor_triggers, "notify_error", lambda m, **_: seen.append(m)
    )
    # A second report from the executor would land here.
    monkeypatch.setattr(pe.monitor_triggers, "notify_error", lambda m, **_: seen.append(m))
    return seen


def test_failed_materialize_is_reported_and_reraised(reported):
    assert triggers.is_device_oom(_CU_MEM_CREATE_OOM)
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
    monkeypatch.setattr(pe.monitor_triggers, "notify_error", lambda m, **_: events.append(m))
    executor = _error_handling_executor(events, fatal=fatal)

    pe.PyExecutor._handle_errors(executor, "CUDA error: out of memory", charge_budget=False)

    assert events[-1] == "CUDA error: out of memory", events
    assert "responses" in events[:-1], events
    assert events.count("CUDA error: out of memory") == 1


def test_rank_local_fatal_leaves_the_report_to_the_event_loop(monkeypatch):
    """The raise reaches _event_loop_wrapper, which arms the watchdog first."""
    events = []
    monkeypatch.setattr(pe.monitor_triggers, "notify_error", lambda m, **_: events.append(m))
    executor = _error_handling_executor(events, fatal=True, attention_dp=True, world_size=2)

    with pytest.raises(RuntimeError, match="Fatal error"):
        pe.PyExecutor._handle_errors(executor, "CUDA error: out of memory", charge_budget=False)

    assert "CUDA error: out of memory" not in events


def test_crashed_rank_arms_its_watchdog_before_the_report(monkeypatch):
    """A crashed loop reports its out-of-memory error itself, in every mode but off."""
    events = []
    monkeypatch.setattr(pe, "host_profiler_context", lambda enable: contextlib.nullcontext())
    monkeypatch.setattr(pe, "customized_gc_thresholds", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(
        pe,
        "start_rank_crash_kill_watchdog",
        lambda world_size, *, error_delivered: events.append("watchdog"),
    )
    monkeypatch.setattr(pe, "hard_kill_on_rank_crash", lambda *_, **__: events.append("kill"))
    monkeypatch.setattr(
        pe.monitor_triggers, "notify_error", lambda *_, **kw: events.append(("report", kw))
    )
    monkeypatch.setattr(pe.logger, "error", lambda message: events.append("log"))
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

    (report,) = [e for e in events if isinstance(e, tuple)]
    assert isinstance(report[1]["error"], torch.OutOfMemoryError)
    assert report[1]["background"], "the error still reaches the waiters first"
    assert events.index("watchdog") < events.index(report) < events.index("log"), events
    assert events.count("watchdog") == 1, events
    assert events[-2:] == ["cleanup", "kill"], events


@pytest.fixture
def teardown(monkeypatch):
    calls = mock.Mock()
    monkeypatch.setattr(pe.monitor_providers, "end_executor", calls.end_executor)
    return calls


def _monitored_executor(token):
    executor = object.__new__(pe.PyExecutor)
    executor._rank_reporter = mock.Mock()
    executor._memory_monitor_token = token
    return executor


def test_each_teardown_step_runs_once(teardown):
    token = object()
    executor = _monitored_executor(token)
    rank_reporter = executor._rank_reporter

    for _ in range(2):  # a retried shutdown repeats nothing
        executor._release_memory_monitor()

    rank_reporter.stop.assert_called_once_with()
    teardown.end_executor.assert_called_once_with(token)
    assert executor._rank_reporter is None
    assert executor._memory_monitor_token is None


def test_only_the_monitored_executor_stops_stack_recording(monitor_state, monkeypatch):
    providers, _ = monitor_state
    stop = mock.Mock()
    monkeypatch.setattr(tagging, "stop_stack_recording", stop)
    monkeypatch.setattr(tagging.startup, "clear", lambda: None)
    monkeypatch.setattr(tagging, "set_frozen_map", lambda _: None)
    first = providers.begin_executor()
    second = providers.begin_executor()
    assert first.monitored and not second.monitored
    monkeypatch.setattr(tagging, "_startup_executor", first)

    _monitored_executor(second)._release_memory_monitor()
    stop.assert_not_called()

    _monitored_executor(first)._release_memory_monitor()
    stop.assert_called_once_with()


def test_unmonitored_executor_tears_down_nothing(teardown):
    executor = object.__new__(pe.PyExecutor)
    executor._rank_reporter = None
    executor._memory_monitor_token = None

    executor._release_memory_monitor()

    assert teardown.mock_calls == []


@pytest.fixture
def monitor_state():
    """The process-wide monitor state these tests change, reset afterwards."""
    from tensorrt_llm._torch.memory.monitor import providers, settings

    providers.clear_registered_providers()
    settings.reset()
    yield providers, settings
    providers.clear_registered_providers()
    settings.reset()


def test_a_loop_that_ends_without_shutdown_leaves_the_monitor(monitor_state, monkeypatch):
    """A follower under the RPC orchestrator's rank-local shutdown never runs shutdown()."""
    from tensorrt_llm.llmapi.llm_args import MemoryMonitorConfig

    providers, settings = monitor_state
    monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
    monkeypatch.setattr(pe, "host_profiler_context", lambda enable: contextlib.nullcontext())
    monkeypatch.setattr(pe, "customized_gc_thresholds", lambda _: contextlib.nullcontext())
    settings.configure(MemoryMonitorConfig(mode="oom"))
    token = providers.begin_executor()
    executor = object.__new__(pe.PyExecutor)
    executor.garbage_collection_gen0_threshold = None
    executor._memory_monitor_token = token
    executor._rank_reporter = None

    def loop():
        executor._event_loop_completed = True

    executor.event_loop = loop
    executor._executor_loop_cleanup = lambda: None

    executor._event_loop_wrapper()

    assert executor._memory_monitor_token is None
    assert not providers.monitor_held()
    settings.configure(MemoryMonitorConfig(mode="full"))
    assert settings.full_mode(rank=0), "the next executor's configuration applies"


def test_a_failed_shutdown_step_still_leaves_the_monitor(monkeypatch):
    released = []
    executor = object.__new__(pe.PyExecutor)
    executor._openengine_node_server = None
    executor.worker_started = True

    def fail():
        raise RuntimeError("KV connector shutdown failed")

    executor.executor_request_queue = SimpleNamespace(enqueue_shutdown_request=fail)
    executor._release_memory_monitor = lambda: released.append(True)

    with pytest.raises(RuntimeError, match="KV connector shutdown failed"):
        executor._release_execution_resources()
    assert released == [True]


def test_a_kv_pool_leaves_the_report_before_it_is_freed():
    """The reading that closes the step releasing a pool must not count it.

    A KV cache manager V1 keeps reporting the bytes it allocated after its
    shutdown, so the KV-sizing pass's pool would otherwise still be counted
    when the step that frees it closes.
    """
    from tensorrt_llm._torch.memory.monitor import providers
    from tensorrt_llm._torch.memory.monitor.schema import Basis, Category, Entry, Placement, Source

    seen_at_release = []

    class Manager:
        def shutdown(self):
            seen_at_release.append([e.owner for e in providers.collect_all(registered())])

    class Provider:
        name = category = "KV cache (main)"

        def __init__(self, manager):
            self.manager = manager

        def describes(self, component):
            return component is self.manager

        def available(self):
            return True

        def collect(self):
            return [
                Entry(
                    category=Category.KV_CACHE_POOL,
                    size_bytes=1 << 30,
                    source=Source.COMPONENT,
                    placement=Placement.NON_TORCH,
                    basis=Basis.PHYSICAL_BACKING,
                    owner=self.name,
                )
            ]

    def registered():
        return [p for _, p in providers._registered]

    manager = Manager()
    try:
        providers.register_provider(Provider(manager))
        assert [e.owner for e in providers.collect_all(registered())] == ["KV cache (main)"]
        pe.PyExecutor._shutdown_resource_managers([manager, None])
    finally:
        providers.clear_registered_providers()
    assert seen_at_release == [[]], "the pool is out of the reading taken as it is freed"


def test_a_late_report_request_says_the_ranks_are_shutting_down(teardown):
    executor = _monitored_executor(object())
    executor.dist = SimpleNamespace(rank=0, world_size=3)
    executor._disable_mpi = False
    executor._memory_monitor_released = False

    executor._release_memory_monitor()

    assert executor.gather_memory_reports() == [
        "rank 1: not gathered; shutting down",
        "rank 2: not gathered; shutting down",
    ]


def test_an_unmonitored_executor_says_why_it_has_no_reports(monkeypatch):
    """Another executor in the process holds the monitor; a report would describe it."""
    from tensorrt_llm._torch.memory.monitor import providers
    from tensorrt_llm.executor.base_worker import BaseWorker

    unmonitored = providers.ExecutorToken(monitored=False)
    executor = object.__new__(pe.PyExecutor)
    executor._rank_reporter = None
    executor._memory_monitor_token = unmonitored
    executor._memory_monitor_released = False
    executor._disable_mpi = False
    executor.dist = SimpleNamespace(rank=0, world_size=2)
    expected = f"rank 1: {providers.NOT_MONITORED}"
    assert executor.gather_memory_reports() == [expected]

    monkeypatch.setattr(triggers, "report_now", mock.Mock(side_effect=AssertionError("built")))
    worker = SimpleNamespace(engine=executor)
    assert BaseWorker.get_memory_report(worker) == [f"rank 0: {providers.NOT_MONITORED}", expected]


class _Comm:
    def __init__(self):
        self.calls = []

    def bcast(self, value, root):
        self.calls.append("bcast")
        return value

    def Dup(self):
        self.calls.append("Dup")
        return self


@pytest.mark.parametrize("warmup", [True, False])
def test_only_the_serving_executor_gets_a_rank_reporter(monkeypatch, warmup):
    """No collective decides it: every rank reads the same arguments."""
    comm = _Comm()
    monkeypatch.setattr(pe, "mpi_comm", lambda: comm)
    monkeypatch.setattr(pe, "_rank_reports_wanted", lambda config: config == "full")
    monkeypatch.setattr(pe, "RankReporter", lambda *args: ("reporter",) + args)
    executor = object.__new__(pe.PyExecutor)
    executor.dist = SimpleNamespace(rank=1, world_size=2)
    executor._disable_mpi = False
    executor.device_id = 0
    executor._rank_reporter = None
    executor.llm_args = SimpleNamespace(memory_monitor_config="full")
    executor.is_warmup = warmup

    executor._start_rank_reporter()

    if warmup:  # the KV-sizing pass
        assert comm.calls == [] and executor._rank_reporter is None
    else:
        assert comm.calls == ["Dup"]
        assert executor._rank_reporter == ("reporter", comm, 1, 2, 0)


def test_ranks_answer_reports_only_when_mpi_allows_threads(monitor_state, monkeypatch):
    from mpi4py import MPI

    from tensorrt_llm.llmapi.llm_args import MemoryMonitorConfig

    providers, _ = monitor_state
    full, oom = MemoryMonitorConfig(mode="full"), MemoryMonitorConfig(mode="oom")
    warnings = []
    monkeypatch.setattr(pe.logger, "warning_once", lambda message, **_: warnings.append(message))
    first = providers.begin_executor()
    monkeypatch.setattr(MPI, "Query_thread", lambda: MPI.THREAD_SERIALIZED)
    assert not pe._rank_reports_wanted(full)
    assert any("MPI_THREAD_MULTIPLE" in w for w in warnings)

    monkeypatch.setattr(MPI, "Query_thread", lambda: MPI.THREAD_MULTIPLE)
    assert pe._rank_reports_wanted(full)
    assert not pe._rank_reports_wanted(oom)
    providers.begin_executor()
    assert not pe._rank_reports_wanted(full), "only the monitored executor"
    providers.end_executor(first)


@pytest.fixture
def creator(monitor_state, monkeypatch):
    """create_py_executor around a fake build that records the monitor's state."""
    from tensorrt_llm._torch.memory.monitor import startup
    from tensorrt_llm._torch.memory.monitor import triggers as trig
    from tensorrt_llm._torch.pyexecutor import py_executor_creator as creator_module

    providers, settings = monitor_state
    seen = SimpleNamespace(windows=[], builds=[], armed=[], full_mode_after=[])

    @contextlib.contextmanager
    def startup_window():
        seen.windows.append(settings.full_mode(rank=0))
        yield

    def build(**_):
        seen.full_mode_after.append(settings.full_mode(rank=0))
        seen.builds.append(
            SimpleNamespace(
                recording_paused=startup._paused > 0, reporting_suspended=trig._suspended > 0
            )
        )
        return SimpleNamespace(device_id=0)

    monkeypatch.setattr(creator_module.monitor_tagging, "startup_window", startup_window)
    monkeypatch.setattr(
        creator_module.monitor_triggers,
        "install_oom_reporter",
        lambda **kw: seen.armed.append(kw["owner"]),
    )
    monkeypatch.setattr(creator_module.monitor_pools, "prime", lambda: None)
    monkeypatch.setattr(creator_module, "monitor_executor_initialization", contextlib.nullcontext)
    monkeypatch.setattr(creator_module, "_create_py_executor_impl", build)

    def create(config, world_size=1):
        llm_args = SimpleNamespace(
            memory_monitor_config=config, parallel_config=SimpleNamespace(world_size=world_size)
        )
        return creator_module.create_py_executor(llm_args=llm_args)

    seen.create = create
    return seen


def test_a_rank_filter_outside_the_job_still_records_startup(creator, monkeypatch):
    """``TLLM_MEM_MONITOR_RANKS=8`` at world size 8 selects no rank of the job.

    The arguments fold the environment in with the job's world size, so the
    executor receives a configuration without the filter and every rank
    records its startup.
    """
    from tensorrt_llm._torch.memory.monitor import settings
    from tensorrt_llm.llmapi.llm_args import MemoryMonitorConfig

    monkeypatch.setenv(settings.RANK_ENV_VAR, "8")
    resolved = settings.resolve_environment(MemoryMonitorConfig(mode="full"), world_size=8)
    monkeypatch.delenv(settings.RANK_ENV_VAR)

    creator.create(resolved, world_size=8)

    assert creator.windows == [True], "the filter is ignored before startup is recorded"


def test_only_the_first_executor_in_a_process_is_monitored(creator):
    from tensorrt_llm.llmapi.llm_args import MemoryMonitorConfig

    first = creator.create(MemoryMonitorConfig(mode="full"))
    second = creator.create(MemoryMonitorConfig(mode="off"))

    assert first._memory_monitor_token.monitored
    assert not second._memory_monitor_token.monitored
    assert creator.armed == [first._memory_monitor_token], "only the first is armed"
    assert creator.windows == [True], "only the first records its startup"
    assert [b.recording_paused for b in creator.builds] == [False, True]
    assert [b.reporting_suspended for b in creator.builds] == [True, False], (
        "the first executor's OOMs are still reported while the second is built"
    )
    assert creator.full_mode_after == [True, True], "the second's config is not applied"


@pytest.mark.parametrize(
    "error, explained",
    [
        (torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"), "GPU memory"),
        (RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"), "GPU memory"),
        (
            RuntimeError(
                "Failed, NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)"
            ),
            "NCCL error",
        ),
        (RuntimeError("CUDA runtime error in cudaHostRegister(ptr, size, 0): out of memory"), None),
        (MemoryError("Cannot allocate memory"), None),
        (KeyError("model.layers.0.weight"), None),
    ],
)
def test_startup_explains_the_oom_errors_the_monitor_reports(error, explained):
    """Startup's explanation and the monitor classify errors the same way."""
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import _ExecutorMemoryMonitor
    from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

    monitor = object.__new__(_ExecutorMemoryMonitor)
    monitor._total_gpu_memory_bytes = 80 << 30
    monitor._samples = []

    message = monitor._maybe_explain_if_oom(
        error, current_stage=ExecutorMemoryType.KV_CACHE, free_gpu_memory_bytes_pre=1 << 30
    )

    if explained is None:
        assert message is None
    else:
        assert explained in message
