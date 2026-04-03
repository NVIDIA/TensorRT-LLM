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
import pathlib
import signal
import threading
import time
from concurrent.futures import Future
from queue import Queue
from unittest.mock import MagicMock, Mock, patch

import pytest

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


# ---------------------------------------------------------------------------
# PyExecutor mock — uses the real classify_error() for classification and
# mirrors the token-bucket / _handle_errors logic from PyExecutor.
# ---------------------------------------------------------------------------
class MockPyExecutorForFatalError:
    def __init__(self):
        self._fatal_error = None
        self.is_shutdown: bool = False
        self._error_budget: float = 1.0
        self._last_error_time = None
        self._error_budget_recovery_rate: float = 0.1
        self._error_budget_cost: float = 0.1
        self.active_requests = []
        self.executor_request_queue = Mock()

    def _consume_error_budget(self, error_msg: str) -> bool:
        now = time.monotonic()
        classification = classify_error(error_msg)
        if classification == "immediate_fatal":
            return True
        if self._last_error_time is not None:
            elapsed = now - self._last_error_time
            self._error_budget = min(
                1.0, self._error_budget + elapsed * self._error_budget_recovery_rate
            )
        self._last_error_time = now
        cost = self._error_budget_cost
        if classification == "severe":
            cost *= 5
        self._error_budget -= cost
        return self._error_budget < 1e-9

    def _handle_errors(self, error_msg=None, *, requests=None):
        error_msg = error_msg or "error"
        is_fatal = self._consume_error_budget(error_msg)
        if is_fatal:
            self._fatal_error = RuntimeError(f"Fatal error: {error_msg}")
            self.is_shutdown = True
            requests = None
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
        if self._fatal_error is None:
            self._fatal_error = error

    def is_shutdown(self) -> bool:
        return self.doing_shutdown or self._fatal_error is not None

    def check_health(self) -> bool:
        if self.doing_shutdown or self._fatal_error is not None:
            return False
        if not self._error_queue.empty():
            try:
                e = self._error_queue.get_nowait()
                self._error_queue.task_done()
                if not isinstance(e, str):
                    self._set_fatal_error(e)
                    self.doing_shutdown = True
            except Exception:
                pass
            return self._fatal_error is None and not self.doing_shutdown
        return True

    def shutdown(self):
        self.doing_shutdown = True


class ConcreteProxyExecutor(ConcreteExecutor):
    """Mirrors GenerationExecutorProxy's MPI-aware health check + monitor."""

    def __init__(self):
        super().__init__()
        self.mpi_futures: list = []
        self._shutdown_called: bool = False

    def check_health(self) -> bool:
        if not super().check_health():
            return False
        if self.mpi_futures:
            for f in self.mpi_futures:
                if f.done():
                    exc = f.exception() if not f.cancelled() else None
                    error = exc or RuntimeError("MPI worker exited unexpectedly")
                    self._set_fatal_error(error)
                    if not self.doing_shutdown:
                        self.shutdown()
                    return False
        return True

    def _error_monitor_loop(self):
        while not self.doing_shutdown and self._fatal_error is None:
            try:
                if self.mpi_futures:
                    for f in self.mpi_futures:
                        if f.done():
                            exc = f.exception() if not f.cancelled() else None
                            error = exc or RuntimeError("MPI worker exited unexpectedly")
                            self._set_fatal_error(error)
                            if not self.doing_shutdown:
                                self.shutdown()
                            return
                if not self._error_queue.empty():
                    try:
                        e = self._error_queue.get_nowait()
                        self._error_queue.task_done()
                        if isinstance(e, str):
                            continue
                        self._set_fatal_error(e)
                        if not self.doing_shutdown:
                            self.shutdown()
                    except Exception:
                        pass
                    if self._fatal_error is not None:
                        return
            except Exception:
                pass
            for _ in range(50):
                if self.doing_shutdown or self._fatal_error is not None:
                    return
                time.sleep(0.01)

    def shutdown(self):
        self.doing_shutdown = True
        self._shutdown_called = True


# ===========================================================================
# Tests
# ===========================================================================


def _make_request(req_id: int) -> Mock:
    req = Mock()
    req.py_request_id = req_id
    req.py_client_id = 1
    req.state = "GENERATION_IN_PROGRESS"
    return req


# ---------------------------------------------------------------------------
# classify_error() — tests the real module-level function directly
# ---------------------------------------------------------------------------
class TestClassifyError:
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
        assert classify_error(error_msg) == expected


# ---------------------------------------------------------------------------
# Token-bucket error budget + _handle_errors
# ---------------------------------------------------------------------------
class TestErrorBudget:
    @pytest.fixture
    def executor(self):
        ex = MockPyExecutorForFatalError()
        ex._error_budget_recovery_rate = 0.0
        return ex

    def test_immediate_fatal_bypasses_budget(self, executor):
        assert executor._error_budget == 1.0
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
        executor._error_budget_cost = 0.25
        for i in range(3):
            executor.active_requests = [_make_request(i)]
            executor._handle_errors("some transient error", requests=executor.active_requests[:])
            assert executor._fatal_error is None
        executor._handle_errors("some transient error", requests=[_make_request(99)])
        assert executor._fatal_error is not None

    def test_budget_recovers_over_time(self):
        executor = MockPyExecutorForFatalError()
        executor._handle_errors("transient error", requests=[_make_request(1)])
        budget_after = executor._error_budget

        executor._last_error_time = time.monotonic() - 5.0

        executor._handle_errors("transient error", requests=[_make_request(2)])
        assert executor._fatal_error is None
        assert executor._error_budget > budget_after - 0.1

    def test_non_fatal_only_fails_specified_requests(self, executor):
        req1, req2 = _make_request(1), _make_request(2)
        executor.active_requests = [req1, req2]
        executor._handle_errors("Input too long", requests=[req1])
        assert executor._fatal_error is None
        assert req1 not in executor.active_requests
        assert req2 in executor.active_requests
        executor.executor_request_queue.enqueue_shutdown_request.assert_not_called()

    def test_fatal_fails_all_and_enqueues_shutdown(self, executor):
        reqs = [_make_request(i) for i in range(3)]
        executor.active_requests = list(reqs)
        executor._handle_errors("cudaErrorIllegalAddress", requests=[reqs[0]])
        assert executor._fatal_error is not None
        assert len(executor.active_requests) == 0
        for r in reqs:
            assert r.state == "GENERATION_COMPLETE"
        executor.executor_request_queue.enqueue_shutdown_request.assert_called_once()

    def test_fatal_error_with_no_active_requests(self, executor):
        executor.active_requests = []
        executor._handle_errors("cudaErrorLaunchFailure")
        assert executor._fatal_error is not None
        executor.executor_request_queue.enqueue_shutdown_request.assert_called_once()

    def test_default_error_msg_is_transient(self, executor):
        executor.active_requests = []
        executor._handle_errors()
        assert executor._fatal_error is None


# ---------------------------------------------------------------------------
# GenerationExecutor: _set_fatal_error, is_shutdown, check_health
# ---------------------------------------------------------------------------
class TestGenerationExecutor:
    @pytest.fixture
    def executor(self):
        return ConcreteExecutor()

    def test_fatal_error_none_initially(self, executor):
        assert executor._fatal_error is None

    def test_set_fatal_error_first_wins(self, executor):
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
    @pytest.fixture
    def executor(self):
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
        executor.mpi_futures = [future_factory()]
        assert executor.check_health() is expected_healthy
        if error_substr:
            assert error_substr in str(executor._fatal_error)

    def test_healthy_with_no_mpi_futures(self, executor):
        assert executor.check_health() is True

    def test_parent_unhealthy_short_circuits(self, executor):
        executor._set_fatal_error(RuntimeError("parent"))
        executor.mpi_futures = [Future()]
        assert executor.check_health() is False


# ---------------------------------------------------------------------------
# _error_monitor_loop background thread
# ---------------------------------------------------------------------------
class TestErrorMonitorLoop:
    def test_detects_worker_crash(self):
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
        assert executor._shutdown_called

    def test_detects_error_queue_item(self):
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
        time.sleep(0.2)

        assert executor._fatal_error is None
        executor.doing_shutdown = True
        t.join(timeout=2.0)

    def test_stops_on_shutdown_flag(self):
        executor = ConcreteProxyExecutor()

        t = threading.Thread(target=executor._error_monitor_loop, daemon=True)
        t.start()
        time.sleep(0.05)
        executor.doing_shutdown = True
        t.join(timeout=2.0)

        assert not t.is_alive()


# ---------------------------------------------------------------------------
# gRPC health check
# ---------------------------------------------------------------------------
class TestGrpcHealthCheck:
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
        from starlette.responses import Response

        server = MagicMock()
        server._check_health = Mock(return_value=check_healthy)
        executor = Mock()
        executor._fatal_error = fatal_error
        server.generator = Mock()
        server.generator._executor = executor

        with patch.object(signal, "raise_signal") as mock_sig:
            if server._check_health():
                response = Response(status_code=200)
            else:
                ex = getattr(server.generator, "_executor", None)
                if ex is not None and getattr(ex, "_fatal_error", None) is not None:
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
    @staticmethod
    def _check_health(llm) -> bool:
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
        assert self._check_health(object()) is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _done_future(exc=None):
    f = Future()
    if exc is not None:
        f.set_exception(exc)
    else:
        f.set_result(None)
    return f


def _cancelled_future():
    f = Future()
    f.cancel()
    return f
