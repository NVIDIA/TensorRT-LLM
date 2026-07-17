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
- classify_error() module-level function (tested directly, not via mock)
- PyExecutor token-bucket error budget and _handle_errors()
- GenerationExecutor.check_health(), _set_fatal_error(), is_shutdown()
- GenerationExecutorProxy.check_health() with MPI worker liveness
- GenerationExecutorProxy._error_monitor_loop() background detection
- GrpcRequestManager.health_check() integration
- OpenAIServer /health endpoint with fatal error shutdown
- BaseLLM._check_health() delegation
"""

# Import classify_error directly from the source file to avoid triggering
# the heavy tensorrt_llm.__init__ (which loads C++ extensions).
import importlib.util
import logging
import pathlib
import signal
import threading
import time
from concurrent.futures import Future
from queue import Empty, Queue
from unittest.mock import MagicMock, Mock, patch

import pytest

logger = logging.getLogger(__name__)

_mod_path = (
    pathlib.Path(__file__).resolve().parents[3]
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "error_classification.py"
)
_spec = importlib.util.spec_from_file_location("error_classification", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
classify_error = _mod.classify_error
ErrorBudget = _mod.ErrorBudget


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


# ---------------------------------------------------------------------------
# Minimal executor mocks for GenerationExecutor / Proxy tests
# ---------------------------------------------------------------------------
class ConcreteExecutor:
    """Mirrors GenerationExecutor's fatal-error and health-check logic."""

    def __init__(self):
        self._error_queue: Queue = Queue()
        self.doing_shutdown: bool = False
        self._fatal_error = None

    def _set_fatal_error(self, error):
        """Record the first fatal error; subsequent calls are no-ops."""
        if self._fatal_error is None:
            self._fatal_error = error

    def is_shutdown(self) -> bool:
        """Return True if shutdown is in progress or a fatal error occurred."""
        return self.doing_shutdown or self._fatal_error is not None

    def check_health(self) -> bool:
        """Check if executor is healthy, draining all queued errors."""
        if self.doing_shutdown or self._fatal_error is not None:
            return False
        drained = False
        while True:
            try:
                e = self._error_queue.get_nowait()
                self._error_queue.task_done()
                drained = True
                if not isinstance(e, str):
                    self._set_fatal_error(e)
                    self.doing_shutdown = True
                    break
            except Empty:
                break
        if drained:
            return self._fatal_error is None and not self.doing_shutdown
        return True

    def shutdown(self):
        """Mark executor as shutting down."""
        self.doing_shutdown = True


class ConcreteProxyExecutor(ConcreteExecutor):
    """Mirrors GenerationExecutorProxy's MPI-aware health check + monitor.

    Uses shared ``_check_mpi_futures()`` and ``_drain_error_queue()``
    helpers to match the production proxy's unified pattern.
    """

    def __init__(self):
        super().__init__()
        self.mpi_futures: list = []
        self._pre_shutdown_called: bool = False
        self._shutdown_event = threading.Event()
        # Track sentinel sends so the pre_shutdown sentinel test can
        # observe whether ``request_queue.put_noblock(None, ...)`` would
        # have been called by the real proxy.
        self.request_queue = Mock()
        self.workers_started: bool = True
        self._abort_all_requests_called: bool = False

    def _check_mpi_futures(self) -> bool:
        """Return True if any MPI worker future has completed."""
        for f in self.mpi_futures:
            if f.done():
                exc = f.exception() if not f.cancelled() else None
                error = exc or RuntimeError("MPI worker exited unexpectedly")
                self._set_fatal_error(error)
                if not self.doing_shutdown:
                    self.pre_shutdown()
                return True
        return False

    def _drain_error_queue(self) -> bool:
        """Drain all queued errors, skipping per-request str errors."""
        drained = False
        while True:
            try:
                e = self._error_queue.get_nowait()
                self._error_queue.task_done()
                drained = True
                if isinstance(e, str):
                    continue
                self._set_fatal_error(e)
                if not self.doing_shutdown:
                    self.pre_shutdown()
                break
            except Empty:
                break
        return drained

    def check_health(self) -> bool:
        """Check executor health including MPI worker liveness."""
        if self.doing_shutdown or self._fatal_error is not None:
            return False
        if self._drain_error_queue():
            return self._fatal_error is None and not self.doing_shutdown
        if self._check_mpi_futures():
            return False
        return True

    def _error_monitor_loop(self):
        """Background loop using shared helpers."""
        while not self.doing_shutdown and self._fatal_error is None:
            try:
                if self._check_mpi_futures():
                    return
                self._drain_error_queue()
                if self._fatal_error is not None:
                    return
            except Exception as exc:
                logger.debug(f"Mock monitor: unexpected exception (ignored): {exc}")
            self._shutdown_event.wait(timeout=0.5)

    def pre_shutdown(self):
        """Mirror production GenerationExecutorProxy.pre_shutdown.

        Sentinel-send logic must match production exactly so the
        ``TestPreShutdownSentinel`` regression suite catches a future
        revert.  See PR #12718 investigation -- the bug was a one-character
        change from ``all`` to ``any`` here, which silently dropped the
        sentinel for the empty-``mpi_futures`` case used by
        ``RemoteMpiCommSessionClient`` / ``trtllm-llmapi-launch``.
        """
        if not self.workers_started:
            return
        if self.doing_shutdown:
            return
        self.doing_shutdown = True
        self._pre_shutdown_called = True
        self._shutdown_event.set()
        self._abort_all_requests_called = True
        if not self.mpi_futures or any(not f.done() for f in self.mpi_futures):
            self.request_queue.put_noblock(None, retry=4)


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


# NOTE: An earlier version of this file had a `test_charge_budget_true_is_default`
# test here that was functionally identical to `test_immediate_fatal_bypasses_budget`
# (both call `_handle_errors("device-side assert ...")` and assert `_fatal_error is
# not None`).  The intent of "true is default" is now captured by
# `test_immediate_fatal_bypasses_budget` which does not pass `charge_budget` and
# therefore relies on the default of True.  Removed to reduce noise.


# ---------------------------------------------------------------------------
# GenerationExecutor: _set_fatal_error, is_shutdown, check_health
# ---------------------------------------------------------------------------
class TestGenerationExecutor:
    """Tests for GenerationExecutor's _set_fatal_error, is_shutdown, check_health."""

    @pytest.fixture
    def executor(self):
        """Return a fresh ConcreteExecutor for each test."""
        return ConcreteExecutor()

    def test_fatal_error_none_initially(self, executor):
        """Fatal error starts as None."""
        assert executor._fatal_error is None

    def test_set_fatal_error_first_wins(self, executor):
        """Only the first fatal error is recorded; subsequent calls are no-ops."""
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
        ex = ConcreteExecutor()
        ex.doing_shutdown = doing_shutdown
        ex._fatal_error = fatal_error
        assert ex.is_shutdown() is expected

    @pytest.mark.parametrize(
        "doing_shutdown,fatal_error,queue_error,expected",
        [
            (False, None, None, True),
            (True, None, None, False),
            (False, RuntimeError("x"), None, False),
            (False, None, RuntimeError("bg"), False),
        ],
    )
    def test_check_health(self, doing_shutdown, fatal_error, queue_error, expected):
        """check_health returns False for shutdown, fatal error, or queued errors."""
        ex = ConcreteExecutor()
        ex.doing_shutdown = doing_shutdown
        ex._fatal_error = fatal_error
        if queue_error is not None:
            ex._error_queue.put(queue_error)
        assert ex.check_health() is expected


# ---------------------------------------------------------------------------
# GenerationExecutorProxy: check_health with MPI futures
# ---------------------------------------------------------------------------
class TestProxyCheckHealth:
    """Tests for GenerationExecutorProxy's check_health with MPI futures."""

    @pytest.fixture
    def executor(self):
        """Return a fresh ConcreteProxyExecutor for each test."""
        return ConcreteProxyExecutor()

    @pytest.mark.parametrize(
        "future_factory,expected_healthy,error_substr",
        [
            (lambda: Future(), True, None),
            (lambda: _done_future(RuntimeError("segfault")), False, "segfault"),
            (lambda: _done_future(None), False, "unexpectedly"),
            (lambda: _cancelled_future(), False, "unexpectedly"),
        ],
    )
    def test_worker_states(self, executor, future_factory, expected_healthy, error_substr):
        """Verify health status for different MPI worker future states."""
        executor.mpi_futures = [future_factory()]
        assert executor.check_health() is expected_healthy
        if error_substr:
            assert error_substr in str(executor._fatal_error)

    def test_healthy_with_no_mpi_futures(self, executor):
        """No MPI futures means healthy (single-process mode)."""
        assert executor.check_health() is True

    def test_parent_unhealthy_short_circuits(self, executor):
        """Parent executor being unhealthy short-circuits MPI future checks."""
        executor._set_fatal_error(RuntimeError("parent"))
        executor.mpi_futures = [Future()]
        assert executor.check_health() is False


# ---------------------------------------------------------------------------
# pre_shutdown sentinel send (regression: PR #12718)
# ---------------------------------------------------------------------------
class TestPreShutdownSentinel:
    """Verify pre_shutdown puts a None sentinel on request_queue when needed.

    Regression: switching the gate from ``all(not f.done() ...)`` to bare
    ``any(...)`` silently dropped the sentinel when ``mpi_futures`` is
    empty (the ``RemoteMpiCommSessionClient`` / ``trtllm-llmapi-launch``
    case).  Without the sentinel, workers never receive the quit signal,
    never put None on ``result_queue``, and the proxy's
    ``dispatch_result_thread`` blocks indefinitely on
    ``result_queue.get()`` -- causing a 2400 s test timeout.
    """

    @pytest.fixture
    def executor(self):
        return ConcreteProxyExecutor()

    def test_empty_mpi_futures_sends_sentinel(self, executor):
        """Empty ``mpi_futures`` must still trigger the sentinel send.

        This is the case used by ``RemoteMpiCommSessionClient.submit()``
        which returns ``[]`` because workers run in a separate
        ``mgmn_leader_node`` process.
        """
        executor.mpi_futures = []
        executor.pre_shutdown()
        executor.request_queue.put_noblock.assert_called_once_with(None, retry=4)

    def test_all_workers_alive_sends_sentinel(self, executor):
        """At least one alive future means workers need to be told to quit."""
        executor.mpi_futures = [Future(), Future()]
        executor.pre_shutdown()
        executor.request_queue.put_noblock.assert_called_once_with(None, retry=4)

    def test_all_workers_done_skips_sentinel(self, executor):
        """All futures already done means there is no one to notify."""
        executor.mpi_futures = [_done_future(None), _done_future(None)]
        executor.pre_shutdown()
        executor.request_queue.put_noblock.assert_not_called()

    def test_partial_crash_sends_sentinel(self, executor):
        """One worker dead, one alive: surviving worker still needs the sentinel."""
        executor.mpi_futures = [_done_future(RuntimeError("crash")), Future()]
        executor.pre_shutdown()
        executor.request_queue.put_noblock.assert_called_once_with(None, retry=4)

    def test_double_call_is_idempotent(self, executor):
        """Calling pre_shutdown twice must not double-send the sentinel."""
        executor.mpi_futures = []
        executor.pre_shutdown()
        executor.pre_shutdown()
        assert executor.request_queue.put_noblock.call_count == 1

    def test_workers_not_started_is_noop(self, executor):
        """If workers were never started, do not send the sentinel."""
        executor.workers_started = False
        executor.mpi_futures = []
        executor.pre_shutdown()
        executor.request_queue.put_noblock.assert_not_called()
        assert executor.doing_shutdown is False


# ---------------------------------------------------------------------------
# _error_monitor_loop background thread
# ---------------------------------------------------------------------------
class TestErrorMonitorLoop:
    """Tests for the _error_monitor_loop background thread."""

    def test_detects_worker_crash(self):
        """Monitor thread detects MPI worker crash and triggers pre_shutdown."""
        executor = ConcreteProxyExecutor()
        f = Future()
        executor.mpi_futures = [f]

        t = threading.Thread(target=executor._error_monitor_loop, daemon=True)
        t.start()
        time.sleep(0.05)
        f.set_exception(RuntimeError("OOM"))
        t.join(timeout=2.0)

        assert not t.is_alive()
        assert executor._fatal_error is not None
        assert executor._pre_shutdown_called

    def test_detects_error_queue_item(self):
        """Monitor thread detects system errors in the error queue."""
        executor = ConcreteProxyExecutor()

        t = threading.Thread(target=executor._error_monitor_loop, daemon=True)
        t.start()
        time.sleep(0.05)
        executor._error_queue.put(RuntimeError("bg failure"))
        t.join(timeout=2.0)

        assert not t.is_alive()
        assert executor._fatal_error is not None

    def test_skips_per_request_string_errors(self):
        """Per-request string errors should not trigger fatal shutdown."""
        executor = ConcreteProxyExecutor()

        t = threading.Thread(target=executor._error_monitor_loop, daemon=True)
        t.start()
        time.sleep(0.05)
        executor._error_queue.put("per-request error string")
        # Wait past one full poll cycle (0.5s) so the monitor processes
        # the string error and loops back, proving it didn't go fatal.
        time.sleep(0.7)

        assert executor._fatal_error is None
        executor.doing_shutdown = True
        executor._shutdown_event.set()
        t.join(timeout=2.0)

    def test_stops_on_shutdown_flag(self):
        """Monitor thread exits promptly when doing_shutdown is set."""
        executor = ConcreteProxyExecutor()

        t = threading.Thread(target=executor._error_monitor_loop, daemon=True)
        t.start()
        time.sleep(0.05)
        executor.doing_shutdown = True
        executor._shutdown_event.set()  # Wake the monitor immediately
        t.join(timeout=2.0)

        assert not t.is_alive()


# ---------------------------------------------------------------------------
# gRPC health check
# ---------------------------------------------------------------------------
class TestGrpcHealthCheck:
    """Tests for gRPC health check integration with fatal error state."""

    @staticmethod
    async def _health_check(self):
        """Mirrors GrpcRequestManager.health_check logic."""
        try:
            if self.llm is None:
                return False, "LLM not initialized"
            if hasattr(self.llm, "_executor"):
                if self.llm._executor is None:
                    return False, "Executor is not available"
                if not self.llm._executor.check_health():
                    error_msg = "Executor is unhealthy"
                    if self.llm._executor._fatal_error is not None:
                        exc = self.llm._executor._fatal_error
                        short = str(exc).splitlines()[0][:200]
                        error_msg = f"{type(exc).__name__}: {short}"
                    return False, error_msg
            return True, "OK"
        except Exception as e:
            return False, f"Error: {e}"

    def _make_manager(self, executor=None):
        """Create a mock gRPC manager with the given executor."""
        m = MagicMock()
        m.llm = MagicMock()
        m.llm._executor = executor
        m.health_check = self._health_check.__get__(m)
        return m

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "setup,expected_healthy,msg_substr",
        [
            ("healthy", True, "OK"),
            ("fatal", False, "CUDA OOM"),
            ("no_executor", False, "not available"),
            ("no_llm", False, "not initialized"),
            ("shutdown", False, "unhealthy"),
        ],
    )
    async def test_health_check(self, setup, expected_healthy, msg_substr):
        """Verify gRPC health check returns correct status and message."""
        if setup == "no_llm":
            m = MagicMock()
            m.llm = None
            m.health_check = self._health_check.__get__(m)
        else:
            ex = ConcreteExecutor()
            if setup == "fatal":
                ex._set_fatal_error(RuntimeError("CUDA OOM"))
            elif setup == "no_executor":
                ex = None
            elif setup == "shutdown":
                ex.doing_shutdown = True
            m = self._make_manager(ex)

        healthy, msg = await m.health_check()
        assert healthy is expected_healthy
        assert msg_substr in msg


# ---------------------------------------------------------------------------
# OpenAI /health endpoint
# ---------------------------------------------------------------------------
class TestOpenAIHealthEndpoint:
    """Tests for OpenAI /health endpoint with fatal error shutdown."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "check_healthy,fatal_error,expect_code,sigint",
        [
            (True, None, 200, False),
            (False, None, 503, False),
            (False, RuntimeError("CUDA OOM"), 503, True),
        ],
    )
    async def test_health(self, check_healthy, fatal_error, expect_code, sigint):
        """Verify status code and SIGINT behavior for different health states."""
        from starlette.responses import Response

        server = MagicMock()
        server._check_health = Mock(return_value=check_healthy)
        executor = Mock()
        executor._fatal_error = fatal_error
        executor.doing_shutdown = False
        server.generator = Mock()
        server.generator._executor = executor

        with patch.object(signal, "raise_signal") as mock_sig:
            if server._check_health():
                response = Response(status_code=200)
            else:
                ex = getattr(server.generator, "_executor", None)
                if (
                    ex is not None
                    and getattr(ex, "_fatal_error", None) is not None
                    and not getattr(ex, "doing_shutdown", True)
                ):
                    signal.raise_signal(signal.SIGINT)
                response = Response(status_code=503)

            assert response.status_code == expect_code
            if sigint:
                mock_sig.assert_called_once_with(signal.SIGINT)
            else:
                mock_sig.assert_not_called()


# ---------------------------------------------------------------------------
# BaseLLM._check_health delegation
# ---------------------------------------------------------------------------
class TestBaseLLMCheckHealth:
    """Tests for BaseLLM._check_health delegation to executor."""

    @staticmethod
    def _check_health(llm) -> bool:
        """Mirrors BaseLLM._check_health logic."""
        if hasattr(llm, "_executor") and llm._executor is not None:
            return llm._executor.check_health()
        return False

    @pytest.mark.parametrize(
        "has_executor,fatal,expected",
        [
            (True, False, True),
            (True, True, False),
            (False, False, False),
        ],
    )
    def test_delegation(self, has_executor, fatal, expected):
        """_check_health delegates to executor.check_health correctly."""
        llm = Mock()
        if has_executor:
            ex = ConcreteExecutor()
            if fatal:
                ex._set_fatal_error(RuntimeError("x"))
            llm._executor = ex
        else:
            llm._executor = None
        assert self._check_health(llm) is expected

    def test_no_executor_attr(self):
        """Returns False when the object has no _executor attribute."""
        assert self._check_health(object()) is False


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
