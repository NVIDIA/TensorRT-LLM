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
"""Unit tests for fatal error detection and health check functionality.

Tests cover:
- classify_error() and the PyExecutor token-bucket error budget
- GenerationExecutor.check_health(), _set_fatal_error(), is_shutdown()
- WorkerProcessMonitor process-liveness detection
- GenerationExecutorProxy.check_health(), pre_shutdown() and
  _error_monitor_loop(), run on a proxy built without workers
- GrpcRequestManager.health_check(), the OpenAIServer /health endpoint and
  BaseLLM._check_health()
"""

import asyncio
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from utils.executor_stubs import bare_proxy

from tensorrt_llm._torch.pyexecutor.error_classification import ErrorBudget, classify_error
from tensorrt_llm.executor import worker_process_monitor as _monitor_mod
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.proxy import GenerationExecutorFrontendProxy
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.executor.worker_process_monitor import (
    WorkerProcessMonitor,
    _read_process_state,
    capture_worker_process_identity,
)
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.serve import openai_server
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# PyExecutor mock — uses the real classify_error() for classification and
# mirrors the token-bucket / _handle_errors logic from PyExecutor.
# ---------------------------------------------------------------------------
class MockPyExecutorForFatalError:
    """Mirrors PyExecutor's error budget and _handle_errors logic.

    Uses the real ``ErrorBudget`` dataclass and ``classify_error()``
    function so that tests exercise the actual classification and
    budget logic.
    """

    def __init__(self):
        self._fatal_error = None
        self.is_shutdown: bool = False
        self._error_budget = ErrorBudget()
        self.active_requests = []
        self.waiting_queue = []
        self.waiting_drained: list = []
        self.request_queue_items: list = []
        self.request_queue_drained: list = []
        self.executor_request_queue = Mock()

    def _handle_errors(self, error_msg=None, *, requests=None, charge_budget=True):
        """Fail requests and optionally initiate shutdown on fatal errors."""
        error_msg = error_msg or "error"
        is_fatal = self._error_budget.consume(error_msg) if charge_budget else False
        if is_fatal:
            self._fatal_error = RuntimeError(f"Fatal error: {error_msg}")
            self.is_shutdown = True
            requests = None
            # Drain waiting queue on fatal (mirrors real PyExecutor)
            self.waiting_drained = list(self.waiting_queue)
            self.waiting_queue.clear()
            # Drain executor_request_queue on fatal
            self.request_queue_drained = list(self.request_queue_items)
            self.request_queue_items.clear()
        failed_requests = list(self.active_requests) if requests is None else requests
        for request in failed_requests:
            request.state = "GENERATION_COMPLETE"
        if requests is None:
            self.active_requests.clear()
        else:
            self.active_requests = [r for r in self.active_requests if r not in requests]
        if self._fatal_error is not None:
            self.executor_request_queue.enqueue_shutdown_request()


# ===========================================================================
# Tests
# ===========================================================================


def _make_request(req_id: int) -> Mock:
    """Create a mock LlmRequest with the given ID."""
    req = Mock()
    req.py_request_id = req_id
    req.py_client_id = 1
    req.state = "GENERATION_IN_PROGRESS"
    return req


# ---------------------------------------------------------------------------
# classify_error() — tests the real module-level function directly
# ---------------------------------------------------------------------------
class TestClassifyError:
    """Tests for the classify_error() module-level function."""

    @pytest.mark.parametrize(
        "error_msg,expected",
        [
            ("cudaErrorIllegalAddress: illegal memory access", "immediate_fatal"),
            ("cudaErrorLaunchFailure: unspecified failure", "immediate_fatal"),
            ("RuntimeError: device-side assert triggered", "immediate_fatal"),
            ("Unrecoverable error in the engine", "immediate_fatal"),
            ("CUDA error: an illegal memory access was encountered", "immediate_fatal"),
            ("CUDA out of memory. Tried to allocate 2 GiB", "severe"),
            ("RuntimeError: CUDA error: out of memory", "severe"),
            ("NCCL error: unhandled system error", "severe"),
            ("Input length exceeds maximum", "transient"),
            ("Request timed out", "transient"),
            ("Invalid sampling parameter: top_k must be > 0", "transient"),
            ("KV cache capacity exceeded", "transient"),
            ("", "transient"),
            ("DEVICE-SIDE ASSERT triggered", "immediate_fatal"),
            ("Nccl Error: timeout", "severe"),
            ("cuda OUT OF MEMORY", "severe"),
        ],
    )
    def test_classification(self, error_msg, expected):
        """Verify error messages are classified into the correct severity tier."""
        assert classify_error(error_msg) == expected


# ---------------------------------------------------------------------------
# Token-bucket error budget + _handle_errors
# ---------------------------------------------------------------------------
class TestErrorBudget:
    """Tests for the token-bucket error budget and _handle_errors interaction."""

    @pytest.fixture
    def executor(self):
        ex = MockPyExecutorForFatalError()
        ex._error_budget.recovery_rate = 0.0
        return ex

    def test_immediate_fatal_bypasses_budget(self, executor):
        """Immediate-fatal errors ignore the budget and trigger shutdown."""
        assert executor._error_budget.budget == 1.0
        executor._handle_errors("device-side assert triggered")
        assert executor._fatal_error is not None
        assert executor.is_shutdown is True

    def test_severe_errors_exhaust_budget_in_two(self, executor):
        """budget=1.0, severe cost=0.5 -> 2 severe errors exhaust it."""
        for i in range(2):
            executor.active_requests = [_make_request(i)]
            executor._handle_errors("CUDA out of memory", requests=executor.active_requests[:])
        assert executor._fatal_error is not None
        executor.executor_request_queue.enqueue_shutdown_request.assert_called()

    def test_transient_errors_exhaust_budget_in_four(self, executor):
        """cost=0.25 -> 4 transient errors exhaust budget=1.0."""
        executor._error_budget.cost = 0.25
        for i in range(3):
            executor.active_requests = [_make_request(i)]
            executor._handle_errors("some transient error", requests=executor.active_requests[:])
            assert executor._fatal_error is None
        executor._handle_errors("some transient error", requests=[_make_request(99)])
        assert executor._fatal_error is not None

    def test_budget_recovers_over_time(self):
        """Error budget replenishes based on elapsed error-free wall time."""
        executor = MockPyExecutorForFatalError()
        executor._handle_errors("transient error", requests=[_make_request(1)])
        budget_after = executor._error_budget.budget

        executor._error_budget.last_error_time = time.monotonic() - 5.0

        executor._handle_errors("transient error", requests=[_make_request(2)])
        assert executor._fatal_error is None
        assert executor._error_budget.budget > budget_after - 0.1

    def test_non_fatal_only_fails_specified_requests(self, executor):
        """Non-fatal errors only fail the specified requests, not all."""
        req1, req2 = _make_request(1), _make_request(2)
        executor.active_requests = [req1, req2]
        executor._handle_errors("Input too long", requests=[req1])
        assert executor._fatal_error is None
        assert req1 not in executor.active_requests
        assert req2 in executor.active_requests
        executor.executor_request_queue.enqueue_shutdown_request.assert_not_called()

    def test_fatal_fails_all_and_enqueues_shutdown(self, executor):
        """Fatal error fails all active requests and enqueues shutdown."""
        reqs = [_make_request(i) for i in range(3)]
        executor.active_requests = list(reqs)
        executor._handle_errors("cudaErrorIllegalAddress", requests=[reqs[0]])
        assert executor._fatal_error is not None
        assert len(executor.active_requests) == 0
        for r in reqs:
            assert r.state == "GENERATION_COMPLETE"
        executor.executor_request_queue.enqueue_shutdown_request.assert_called_once()

    def test_fatal_error_with_no_active_requests(self, executor):
        """Fatal error works even with an empty active_requests list."""
        executor.active_requests = []
        executor._handle_errors("cudaErrorLaunchFailure")
        assert executor._fatal_error is not None
        executor.executor_request_queue.enqueue_shutdown_request.assert_called_once()

    def test_default_error_msg_is_transient(self, executor):
        """Default error message (None -> 'error') is classified as transient."""
        executor.active_requests = []
        executor._handle_errors()
        assert executor._fatal_error is None

    def test_fatal_drains_waiting_queue(self, executor):
        """Fatal error drains waiting_queue so queued requests are not activated."""
        executor.waiting_queue = ["req_a", "req_b", "req_c"]
        executor._handle_errors("cudaErrorIllegalAddress")
        assert executor._fatal_error is not None
        assert len(executor.waiting_queue) == 0
        assert executor.waiting_drained == ["req_a", "req_b", "req_c"]

    def test_non_fatal_does_not_drain_waiting_queue(self, executor):
        """Non-fatal errors leave the waiting_queue untouched."""
        executor.waiting_queue = ["req_a", "req_b"]
        executor._handle_errors("Input too long", requests=[_make_request(1)])
        assert len(executor.waiting_queue) == 2
        assert executor.waiting_drained == []

    def test_fatal_drains_executor_request_queue(self, executor):
        """Fatal error drains executor_request_queue items."""
        executor.request_queue_items = ["queued_1", "queued_2"]
        executor._handle_errors("cudaErrorIllegalAddress")
        assert executor._fatal_error is not None
        assert len(executor.request_queue_items) == 0
        assert executor.request_queue_drained == ["queued_1", "queued_2"]

    def test_non_fatal_does_not_drain_executor_request_queue(self, executor):
        """Non-fatal errors leave executor_request_queue untouched."""
        executor.request_queue_items = ["queued_1"]
        executor._handle_errors("Input too long", requests=[_make_request(1)])
        assert len(executor.request_queue_items) == 1
        assert executor.request_queue_drained == []

    def test_charge_budget_false_skips_budget(self, executor):
        """Request-scoped errors with charge_budget=False don't consume budget."""
        initial_budget = executor._error_budget.budget
        req = _make_request(1)
        executor.active_requests = [req]
        executor._handle_errors("validation error", requests=[req], charge_budget=False)
        assert executor._error_budget.budget == initial_budget
        assert executor._fatal_error is None
        assert req not in executor.active_requests

    def test_charge_budget_false_never_triggers_fatal(self, executor):
        """Even immediate-fatal patterns don't trigger shutdown when uncharged."""
        req = _make_request(1)
        executor.active_requests = [req]
        executor._handle_errors("cudaErrorIllegalAddress", requests=[req], charge_budget=False)
        assert executor._fatal_error is None
        assert executor.is_shutdown is False
        assert req not in executor.active_requests


# ---------------------------------------------------------------------------
# GenerationExecutor: _set_fatal_error, is_shutdown, check_health
# ---------------------------------------------------------------------------
class _Executor(GenerationExecutor):
    """A GenerationExecutor with no engine; shutdown() only records the call."""

    def __init__(self):
        super().__init__()
        self.shutdowns = 0

    def submit(self, request):
        raise NotImplementedError

    def abort_request(self, request_id):
        raise NotImplementedError

    def shutdown(self):
        self.shutdowns += 1
        self.doing_shutdown = True


class TestGenerationExecutor:
    """Tests for GenerationExecutor's _set_fatal_error, is_shutdown, check_health."""

    def test_fatal_error_none_initially(self):
        """Fatal error starts as None."""
        assert _Executor()._fatal_error is None

    def test_set_fatal_error_first_wins(self):
        """Only the first fatal error is recorded; subsequent calls are no-ops."""
        executor = _Executor()
        first, second = RuntimeError("a"), RuntimeError("b")
        executor._set_fatal_error(first)
        executor._set_fatal_error(second)
        assert executor._fatal_error is first

    @pytest.mark.parametrize(
        "doing_shutdown,fatal_error,expected",
        [
            (False, None, False),
            (True, None, True),
            (False, RuntimeError("x"), True),
            (True, RuntimeError("x"), True),
        ],
    )
    def test_is_shutdown(self, doing_shutdown, fatal_error, expected):
        """is_shutdown reflects both doing_shutdown flag and fatal error state."""
        executor = _Executor()
        executor.doing_shutdown = doing_shutdown
        executor._fatal_error = fatal_error
        assert executor.is_shutdown() is expected

    @pytest.mark.parametrize(
        "doing_shutdown,fatal_error,queued,expected",
        [
            (False, None, [], True),
            (True, None, [], False),
            (False, RuntimeError("x"), [], False),
            (False, None, [RuntimeError("bg")], False),
            (False, None, ["bad request", RequestError("bad request")], True),
            (False, None, ["bad request", RuntimeError("bg")], False),
        ],
    )
    def test_check_health(self, doing_shutdown, fatal_error, queued, expected):
        """check_health is False for shutdown, a fatal error, or a queued system error."""
        executor = _Executor()
        executor.doing_shutdown = doing_shutdown
        executor._fatal_error = fatal_error
        for item in queued:
            executor._error_queue.put(item)
        assert executor.check_health() is expected

    def test_queued_system_error_is_recorded_and_shuts_down(self):
        """A system error behind a per-request error is recorded and shuts down."""
        executor = _Executor()
        failure = RuntimeError("bg")
        executor._error_queue.put("bad request")
        executor._error_queue.put(failure)
        assert executor.check_health() is False
        assert executor._fatal_error is failure
        assert executor.shutdowns == 1


# ---------------------------------------------------------------------------
# WorkerProcessMonitor: pidfd and procfs liveness detection
# ---------------------------------------------------------------------------
class TestWorkerProcessMonitor:
    """Tests for the production OS process-liveness monitor."""

    @pytest.fixture
    def identity(self):
        return capture_worker_process_identity(rank=3)

    def test_pidfd_exit_is_detected(self, identity):
        monitor = WorkerProcessMonitor()
        poller = Mock()
        poller.poll.return_value = [(42, 1)]
        monitor._poller = poller

        with (
            patch.object(_monitor_mod.os, "pidfd_open", return_value=42, create=True),
            patch.object(_monitor_mod.os, "close"),
        ):
            monitor.register([identity])
            assert monitor.find_dead_worker() == identity
            monitor.close()

        poller.register.assert_called_once()
        poller.unregister.assert_called_once_with(42)

    def test_exit_before_pidfd_registration_is_detected(self, identity):
        monitor = WorkerProcessMonitor()
        with patch.object(
            _monitor_mod.os, "pidfd_open", side_effect=ProcessLookupError, create=True
        ):
            monitor.register([identity])
            assert monitor.find_dead_worker() == identity

    def test_pid_reuse_before_pidfd_registration_is_detected(self, identity):
        if identity.start_time is None:
            identity = identity._replace(start_time=100)
        monitor = WorkerProcessMonitor()
        with (
            patch.object(_monitor_mod.os, "pidfd_open", return_value=42, create=True),
            patch.object(
                _monitor_mod,
                "_read_process_state",
                return_value=("S", identity.start_time + 1),
            ),
            patch.object(_monitor_mod.os, "close") as close,
        ):
            monitor.register([identity])
            assert monitor.find_dead_worker() == identity

        close.assert_called_once_with(42)

    def test_procfs_start_time_change_is_detected(self, identity):
        if identity.start_time is None:
            identity = identity._replace(start_time=100)
        monitor = WorkerProcessMonitor()
        with (
            patch.object(_monitor_mod.os, "pidfd_open", None, create=True),
            patch.object(
                _monitor_mod, "_read_process_state", return_value=("S", identity.start_time + 1)
            ),
        ):
            monitor.register([identity])
            assert monitor.find_dead_worker() == identity

    def test_procfs_read_error_does_not_report_death(self, identity):
        if identity.start_time is None:
            identity = identity._replace(start_time=100)
        monitor = WorkerProcessMonitor()
        with (
            patch.object(_monitor_mod.os, "pidfd_open", None, create=True),
            patch.object(
                _monitor_mod, "_read_process_state", side_effect=PermissionError("denied")
            ),
        ):
            monitor.register([identity])
            assert monitor.find_dead_worker() is None

    def test_remote_worker_is_not_registered(self, identity):
        monitor = WorkerProcessMonitor()
        remote_identity = identity._replace(hostname="remote-host")
        with patch.object(
            _monitor_mod.os, "pidfd_open", return_value=42, create=True
        ) as pidfd_open:
            monitor.register([remote_identity])

        pidfd_open.assert_not_called()
        assert monitor.find_dead_worker() is None

    def test_different_pid_namespace_is_not_registered(self, identity):
        monitor = WorkerProcessMonitor()
        namespace = monitor._local_pid_namespace
        other_namespace = 1 if namespace is None else namespace + 1
        other_identity = identity._replace(pid_namespace=other_namespace)
        with patch.object(
            _monitor_mod.os, "pidfd_open", return_value=42, create=True
        ) as pidfd_open:
            monitor.register([other_identity])

        pidfd_open.assert_not_called()
        assert monitor.find_dead_worker() is None


# ---------------------------------------------------------------------------
# GenerationExecutorProxy: check_health, pre_shutdown, _error_monitor_loop
# ---------------------------------------------------------------------------
class _RecordingSender:
    """Command sender stand-in that records how the proxy fails and closes it."""

    def __init__(self):
        self.available = True
        self.failures = []
        self.closes = []

    def fail(self, error):
        self.failures.append(error)
        self.available = False

    def close(self, **kwargs):
        self.closes.append(kwargs)


@pytest.fixture
def live_proxy():
    """Return a factory for proxies whose workers count as started."""
    proxies = []

    def make(*futures, monitor=None):
        sender = _RecordingSender()
        proxy = bare_proxy(workers_started=True, mpi_futures=list(futures), _request_sender=sender)
        if monitor is not None:
            proxy._worker_process_monitor = monitor
        proxies.append(proxy)
        return proxy, sender

    yield make
    for proxy in proxies:
        proxy._shutdown_event.set()
        proxy._worker_process_monitor.close()
        # No workers exist, so __del__ -> shutdown() must not wait for them.
        proxy.workers_started = False


def _future(state):
    if state == "alive":
        return Future()
    if state == "cancelled":
        return _cancelled_future()
    return _done_future(RuntimeError("crash") if state == "crashed" else None)


def _lost_worker_monitor(rank):
    """A process monitor that reports ``rank`` as lost."""
    monitor = Mock(spec=WorkerProcessMonitor)
    monitor.find_dead_worker.return_value = capture_worker_process_identity(rank=rank)
    return monitor


def _start_monitor(proxy):
    thread = threading.Thread(target=proxy._error_monitor_loop, daemon=True)
    thread.start()
    return thread


def _wait_until(condition, message, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, message
        time.sleep(0.01)


class TestProxyCheckHealth:
    """Tests for GenerationExecutorProxy.check_health and its check order."""

    @pytest.mark.parametrize(
        "state,expected_healthy,error_substr",
        [
            ("alive", True, None),
            ("crashed", False, "crash"),
            ("done", False, "unexpectedly"),
            ("cancelled", False, "unexpectedly"),
        ],
    )
    def test_worker_states(self, live_proxy, state, expected_healthy, error_substr):
        """A finished MPI worker future is fatal and starts teardown."""
        proxy, sender = live_proxy(_future(state))
        assert proxy.check_health() is expected_healthy
        if error_substr:
            assert error_substr in str(proxy._fatal_error)
            assert proxy.doing_shutdown and len(sender.closes) == 1
        else:
            assert not sender.closes

    def test_healthy_with_no_mpi_futures(self, live_proxy):
        """No MPI futures means healthy (remote session workers)."""
        proxy, _ = live_proxy()
        assert proxy.check_health() is True

    def test_recorded_fatal_error_short_circuits(self, live_proxy):
        """A recorded fatal error is unhealthy without checking the workers."""
        future = Mock(spec=Future)
        proxy, _ = live_proxy(future)
        proxy._fatal_error = RuntimeError("parent")
        assert proxy.check_health() is False
        future.done.assert_not_called()

    def test_unavailable_command_sender_is_unhealthy(self, live_proxy):
        proxy, sender = live_proxy(Future())
        sender.available = False
        assert proxy.check_health() is False

    @pytest.mark.parametrize("error", ["bad request", RequestError("bad request")])
    def test_per_request_errors_stay_healthy(self, live_proxy, error):
        proxy, sender = live_proxy(Future())
        proxy._error_queue.put(error)
        assert proxy.check_health() is True
        assert proxy._error_queue.empty()
        assert proxy._fatal_error is None and not sender.closes

    def test_queued_error_is_checked_before_worker_liveness(self, live_proxy):
        proxy, sender = live_proxy(_future("crashed"), monitor=_lost_worker_monitor(0))
        queued = RuntimeError("queued failure")
        proxy._error_queue.put(queued)
        assert proxy.check_health() is False
        assert proxy._fatal_error is queued
        assert len(sender.closes) == 1

    def test_worker_future_error_precedes_process_liveness_loss(self, live_proxy):
        """A future carries the worker's actual failure, so it wins over liveness loss."""
        monitor = _lost_worker_monitor(0)
        proxy, _ = live_proxy(_future("crashed"), monitor=monitor)
        assert proxy.check_health() is False
        assert "crash" in str(proxy._fatal_error)
        monitor.find_dead_worker.assert_not_called()

    def test_process_liveness_loss_with_a_pending_future(self, live_proxy):
        """Health checks read local liveness only; remote supervision is the monitor's."""
        monitor = _lost_worker_monitor(2)
        proxy, sender = live_proxy(Future(), monitor=monitor)
        assert proxy.check_health() is False
        assert "rank 2" in str(proxy._fatal_error)
        monitor.find_dead_worker.assert_called_once_with(include_remote=False)
        # A worker may still be alive, so teardown asks it to stop.
        assert [close["send_sentinel"] for close in sender.closes] == [True]

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="pidfd/procfs worker monitoring is Linux-only"
    )
    def test_killed_process_detected_while_future_remains_pending(self, live_proxy):
        """A SIGKILLed worker is fatal even when its MPI future stays pending."""
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        future = Future()
        proxy, sender = live_proxy(future)
        try:
            process_state = _read_process_state(process.pid)
            assert process_state is not None
            identity = capture_worker_process_identity(rank=0, pid=process.pid)
            assert identity.start_time == process_state[1]
            proxy._worker_process_monitor.register([identity])
            assert proxy.check_health() is True

            process.kill()
            process.wait(timeout=5)

            _wait_until(lambda: not proxy.check_health(), "the killed worker was not detected")
            assert future.done() is False
            assert "rank 0" in str(proxy._fatal_error)
            assert proxy.doing_shutdown and len(sender.closes) == 1
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


class TestPreShutdownSentinel:
    """pre_shutdown has the command sender send the stop sentinel while a worker may live.

    ``RemoteMpiCommSessionClient`` (``trtllm-llmapi-launch``) returns no
    futures because its workers run in a separate process. Without the
    sentinel, those workers never stop and never send the ``None`` that ends
    the proxy's ``dispatch_result_thread``.
    """

    @pytest.mark.parametrize(
        "states,sentinel",
        [
            ((), True),
            (("alive", "alive"), True),
            (("done", "done"), False),
            (("crashed", "alive"), True),
        ],
        ids=["no-futures", "all-alive", "all-done", "partial-crash"],
    )
    def test_sentinel_rule(self, live_proxy, states, sentinel):
        proxy, sender = live_proxy(*(_future(state) for state in states))
        proxy.pre_shutdown()
        assert [close["send_sentinel"] for close in sender.closes] == [sentinel]
        assert proxy.doing_shutdown and proxy._shutdown_event.is_set()

    def test_aborts_pending_requests_and_stops_process_monitoring(self, live_proxy):
        monitor = Mock(spec=WorkerProcessMonitor)
        proxy, _ = live_proxy(Future(), monitor=monitor)
        pending = Mock()
        proxy._results = {7: pending}
        proxy.pre_shutdown()
        pending.abort.assert_called_once_with()
        monitor.close.assert_called()

    def test_double_call_is_idempotent(self, live_proxy):
        """Calling pre_shutdown twice must not close the sender twice."""
        proxy, sender = live_proxy()
        proxy.pre_shutdown()
        proxy.pre_shutdown()
        assert len(sender.closes) == 1

    def test_workers_not_started_is_noop(self, live_proxy):
        """If workers were never started, there is no one to stop."""
        proxy, sender = live_proxy()
        proxy.workers_started = False
        proxy.pre_shutdown()
        assert not sender.closes
        assert proxy.doing_shutdown is False

    def test_attached_frontend_cancels_its_requests_without_the_sentinel(self):
        """Only the launcher stops the engine; a frontend cancels its own requests."""
        sender = _RecordingSender()
        frontend = bare_proxy(
            GenerationExecutorFrontendProxy,
            _request_sender=sender,
            _results={5: Mock(), 6: Mock()},
            rpc_client=None,
        )
        frontend.pre_shutdown()
        assert sender.closes == [
            dict(timeout=7.0, send_sentinel=False, cancel_ids=(5, 6), wait=False)
        ]


class TestErrorMonitorLoop:
    """Tests for the _error_monitor_loop background thread."""

    def test_detects_worker_crash(self, live_proxy):
        """The monitor reaps a finished worker future without a health check."""
        proxy, sender = live_proxy(_future("crashed"))
        thread = _start_monitor(proxy)
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert "crash" in str(proxy._fatal_error)
        assert proxy.doing_shutdown and len(sender.closes) == 1

    def test_detects_error_queue_item(self, live_proxy):
        """The monitor records a queued system error and starts teardown."""
        proxy, sender = live_proxy(Future())
        failure = RuntimeError("bg failure")
        proxy._error_queue.put(failure)
        thread = _start_monitor(proxy)
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert proxy._fatal_error is failure
        assert proxy.doing_shutdown and len(sender.closes) == 1

    def test_skips_per_request_errors(self, live_proxy):
        """Per-request errors do not trigger fatal shutdown."""
        proxy, sender = live_proxy(Future())
        proxy._error_queue.put("per-request error string")
        proxy._error_queue.put(RequestError("bad request"))
        thread = _start_monitor(proxy)
        _wait_until(
            lambda: proxy._error_queue.unfinished_tasks == 0, "the monitor did not drain the queue"
        )
        assert thread.is_alive()
        assert proxy._fatal_error is None and not sender.closes
        proxy.doing_shutdown = True
        proxy._shutdown_event.set()
        thread.join(timeout=10)
        assert not thread.is_alive()

    def test_stops_on_shutdown_flag(self, live_proxy):
        """The monitor exits promptly when doing_shutdown is set and it is woken."""
        proxy, _ = live_proxy(Future())
        thread = _start_monitor(proxy)
        proxy.doing_shutdown = True
        proxy._shutdown_event.set()
        thread.join(timeout=10)
        assert not thread.is_alive()

    def test_forwarded_worker_error_precedes_process_liveness_loss(self, live_proxy):
        """A remote session's forwarded worker error wins over a liveness loss."""
        monitor = _lost_worker_monitor(1)
        proxy, _ = live_proxy(monitor=monitor)
        forwarded = ValueError("rank 1 failed to load")
        proxy.mpi_session = SimpleNamespace(check_worker_error=lambda: forwarded)
        thread = _start_monitor(proxy)
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert proxy._fatal_error is forwarded
        monitor.find_dead_worker.assert_not_called()

    def test_monitor_consumes_remote_supervision(self, live_proxy):
        monitor = _lost_worker_monitor(3)
        proxy, _ = live_proxy(Future(), monitor=monitor)
        thread = _start_monitor(proxy)
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert "rank 3" in str(proxy._fatal_error)
        monitor.find_dead_worker.assert_called_once_with(include_remote=True)


# ---------------------------------------------------------------------------
# gRPC health check
# ---------------------------------------------------------------------------
class TestGrpcHealthCheck:
    """Tests for GrpcRequestManager.health_check against executor health state."""

    @pytest.mark.parametrize(
        "setup,expected_healthy,msg_substr",
        [
            ("healthy", True, "OK"),
            ("fatal", False, "RuntimeError: CUDA OOM"),
            ("no_executor", False, "not available"),
            ("no_llm", False, "not initialized"),
            ("shutdown", False, "unhealthy"),
        ],
    )
    def test_health_check(self, setup, expected_healthy, msg_substr):
        """Verify gRPC health check returns correct status and message."""
        pytest.importorskip(
            "smg_grpc_proto",
            reason='SMG gRPC adapter dependency not installed (pip install "tensorrt_llm[grpc-smg]")',
        )
        from tensorrt_llm.grpc.smg.request_manager import GrpcRequestManager

        llm = None
        if setup != "no_llm":
            executor = None if setup == "no_executor" else _Executor()
            if setup == "fatal":
                executor._set_fatal_error(RuntimeError("CUDA OOM\nsecond line"))
            elif setup == "shutdown":
                executor.doing_shutdown = True
            llm = SimpleNamespace(_executor=executor)

        healthy, msg = asyncio.run(GrpcRequestManager(llm).health_check())
        assert healthy is expected_healthy
        assert msg_substr in msg
        assert "second line" not in msg


# ---------------------------------------------------------------------------
# OpenAI /health endpoint
# ---------------------------------------------------------------------------
class TestOpenAIHealthEndpoint:
    """Tests for the OpenAIServer /health endpoint with fatal error shutdown."""

    @pytest.mark.parametrize(
        "state,expect_code,sigint",
        [
            ("healthy", 200, False),
            ("shutting_down", 503, False),
            ("fatal", 503, True),
            # Executor cleanup can start before HTTP shutdown is requested.
            ("fatal_during_cleanup", 503, True),
        ],
    )
    def test_health(self, monkeypatch, state, expect_code, sigint):
        """A fatal engine answers 503 and requests HTTP shutdown once."""
        executor = _Executor()
        if state == "shutting_down":
            executor.doing_shutdown = True
        elif state.startswith("fatal"):
            executor._set_fatal_error(RuntimeError("CUDA OOM"))
            executor.doing_shutdown = state == "fatal_during_cleanup"
        server = object.__new__(OpenAIServer)
        server.generator = SimpleNamespace(_executor=executor, _check_health=executor.check_health)
        server.embedding_batcher = None
        server._uvicorn_server = None
        server._http_shutdown_requested = False
        signals = []
        monkeypatch.setattr(openai_server.signal, "raise_signal", signals.append)
        monkeypatch.setattr(openai_server, "_record_generator_termination", lambda _: None)

        async def probe_twice():
            return [(await server.health()).status_code for _ in range(2)]

        assert asyncio.run(probe_twice()) == [expect_code, expect_code]
        assert signals == ([signal.SIGINT] if sigint else [])


# ---------------------------------------------------------------------------
# BaseLLM._check_health delegation
# ---------------------------------------------------------------------------
def _llm(**attributes):
    llm = object.__new__(BaseLLM)
    llm._encode_only = False
    # No engine was started, so __del__ -> shutdown() has nothing to stop.
    llm.shutdown = lambda: None
    for name, value in attributes.items():
        setattr(llm, name, value)
    return llm


class TestBaseLLMCheckHealth:
    """Tests for BaseLLM._check_health delegation to the executor."""

    @pytest.mark.parametrize(
        "has_executor,fatal,expected",
        [
            (True, False, True),
            (True, True, False),
            (False, False, False),
        ],
    )
    def test_delegation(self, has_executor, fatal, expected):
        """_check_health delegates to executor.check_health."""
        executor = None
        if has_executor:
            executor = _Executor()
            if fatal:
                executor._set_fatal_error(RuntimeError("x"))
        assert _llm(_executor=executor)._check_health() is expected

    def test_no_executor_attr(self):
        """Returns False when the LLM has no _executor attribute."""
        assert _llm()._check_health() is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _done_future(exc=None):
    """Create a Future that is already done, optionally with an exception."""
    f = Future()
    if exc is not None:
        f.set_exception(exc)
    else:
        f.set_result(None)
    return f


def _cancelled_future():
    """Create a Future that has been cancelled."""
    f = Future()
    f.cancel()
    return f
