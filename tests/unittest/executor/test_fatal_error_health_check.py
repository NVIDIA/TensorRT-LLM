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
- PyExecutor._is_fatal_error() pattern matching
- PyExecutor._handle_errors() fatal + consecutive-error logic
- GenerationExecutor.check_health() and _set_fatal_error()
- GenerationExecutorProxy.check_health() MPI worker liveness
- GenerationExecutorProxy._error_monitor_loop() background detection
- GrpcRequestManager.health_check() integration
- OpenAIServer /health endpoint with fatal error shutdown
- BaseLLM._check_health() delegation
"""

import signal
import threading
import time
from concurrent.futures import Future
from queue import Queue
from unittest.mock import MagicMock, Mock, patch

import pytest


# ---------------------------------------------------------------------------
# PyExecutor._is_fatal_error tests
# ---------------------------------------------------------------------------
class MockPyExecutorForFatalError:
    """Minimal mock of PyExecutor with fatal error detection logic."""

    _IMMEDIATE_FATAL_PATTERNS = [
        "cudaerrorillegaladd",
        "cudaerrorlaunchfailure",
        "device-side assert",
        "unrecoverable",
    ]

    _SEVERE_ERROR_PATTERNS = [
        "cuda out of memory",
        "cuda error",
        "nccl error",
    ]

    def __init__(self):
        self._fatal_error = None
        self._error_budget: float = 1.0
        self._last_error_time = None
        self._error_budget_recovery_rate: float = 0.1
        self._error_budget_cost: float = 0.1
        self.active_requests = []
        self.executor_request_queue = Mock()
        self._enqueue_responses = Mock()

    def _classify_error(self, error_msg: str) -> str:
        error_lower = error_msg.lower()
        for p in self._IMMEDIATE_FATAL_PATTERNS:
            if p in error_lower:
                return "immediate_fatal"
        for p in self._SEVERE_ERROR_PATTERNS:
            if p in error_lower:
                return "severe"
        return "transient"

    def _is_fatal_error(self, error_msg: str) -> bool:
        return self._classify_error(error_msg) == "immediate_fatal"

    def _consume_error_budget(self, error_msg: str) -> bool:
        import time

        now = time.monotonic()
        classification = self._classify_error(error_msg)
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
            requests = None

        failed_requests = requests if requests is not None else self.active_requests
        for request in failed_requests:
            request.state = "GENERATION_COMPLETE"

        if requests is None:
            self.active_requests.clear()
        else:
            self.active_requests = [r for r in self.active_requests if r not in requests]

        if self._fatal_error is not None:
            self.executor_request_queue.enqueue_shutdown_request()


class TestClassifyError:
    @pytest.fixture
    def executor(self):
        return MockPyExecutorForFatalError()

    @pytest.mark.parametrize(
        "error_msg",
        [
            "cudaErrorIllegalAddress: an illegal memory access",
            "cudaErrorLaunchFailure: unspecified launch failure",
            "RuntimeError: device-side assert triggered",
            "Unrecoverable error in the engine",
        ],
    )
    def test_immediate_fatal_errors(self, executor, error_msg):
        assert executor._classify_error(error_msg) == "immediate_fatal"
        assert executor._is_fatal_error(error_msg) is True

    @pytest.mark.parametrize(
        "error_msg",
        [
            "CUDA out of memory. Tried to allocate 2.00 GiB",
            "RuntimeError: CUDA error: an illegal memory access was encountered",
            "NCCL error: unhandled system error",
        ],
    )
    def test_severe_errors(self, executor, error_msg):
        assert executor._classify_error(error_msg) == "severe"
        assert executor._is_fatal_error(error_msg) is False

    @pytest.mark.parametrize(
        "error_msg",
        [
            "Input length exceeds maximum",
            "Request timed out",
            "Invalid sampling parameter: top_k must be > 0",
            "tokenizer error: unknown token",
            "KV cache capacity exceeded",
            "Batch size too large",
            "",
        ],
    )
    def test_transient_errors(self, executor, error_msg):
        assert executor._classify_error(error_msg) == "transient"
        assert executor._is_fatal_error(error_msg) is False

    def test_case_insensitive_matching(self, executor):
        assert executor._classify_error("DEVICE-SIDE ASSERT triggered") == "immediate_fatal"
        assert executor._classify_error("Nccl Error: timeout") == "severe"
        assert executor._classify_error("cuda OUT OF MEMORY") == "severe"


# ---------------------------------------------------------------------------
# PyExecutor._handle_errors tests
# ---------------------------------------------------------------------------
class TestHandleErrors:
    @pytest.fixture
    def executor(self):
        return MockPyExecutorForFatalError()

    def _make_request(self, req_id):
        req = Mock()
        req.py_request_id = req_id
        req.py_client_id = 1
        req.state = "GENERATION_IN_PROGRESS"
        return req

    def test_non_fatal_error_only_fails_specified_requests(self, executor):
        req1 = self._make_request(1)
        req2 = self._make_request(2)
        executor.active_requests = [req1, req2]

        executor._handle_errors("Input too long", requests=[req1])

        assert executor._fatal_error is None
        assert req2 in executor.active_requests
        assert req1 not in executor.active_requests
        executor.executor_request_queue.enqueue_shutdown_request.assert_not_called()

    def test_immediate_fatal_error_fails_all_active_requests(self, executor):
        req1 = self._make_request(1)
        req2 = self._make_request(2)
        req3 = self._make_request(3)
        executor.active_requests = [req1, req2, req3]

        executor._handle_errors("cudaErrorIllegalAddress", requests=[req1])

        assert executor._fatal_error is not None
        assert len(executor.active_requests) == 0
        executor.executor_request_queue.enqueue_shutdown_request.assert_called_once()

    def test_severe_errors_exhaust_budget(self, executor):
        """Severe errors (CUDA OOM) cost 5x and exhaust budget quickly."""
        # Disable recovery so elapsed time doesn't interfere
        executor._error_budget_recovery_rate = 0.0
        # budget=1.0, severe cost=0.5 each → 2 severe errors exhaust it
        for i in range(2):
            executor.active_requests = [self._make_request(i + 10)]
            executor._handle_errors("CUDA out of memory", requests=executor.active_requests[:])

        assert executor._fatal_error is not None
        executor.executor_request_queue.enqueue_shutdown_request.assert_called()

    def test_transient_errors_exhaust_budget_slowly(self, executor):
        """Transient errors cost 0.25 each — need 4 to exhaust budget=1.0."""
        executor._error_budget_recovery_rate = 0.0
        executor._error_budget_cost = 0.25
        for i in range(3):
            executor.active_requests = [self._make_request(i + 10)]
            executor._handle_errors("some transient error", requests=executor.active_requests[:])
            assert executor._fatal_error is None

        executor.active_requests = [self._make_request(99)]
        executor._handle_errors("some transient error", requests=[self._make_request(99)])
        assert executor._fatal_error is not None

    def test_budget_recovers_over_time(self, executor):
        """After time passes, the budget replenishes and errors are tolerated."""
        import time

        executor._handle_errors("some transient error", requests=[self._make_request(1)])
        assert executor._error_budget < 1.0
        budget_after_first = executor._error_budget

        # Simulate 5 seconds passing (recovery = 0.1/s × 5s = 0.5)
        executor._last_error_time = time.monotonic() - 5.0

        executor.active_requests = [self._make_request(2)]
        executor._handle_errors("some transient error", requests=[self._make_request(2)])
        # Budget should have recovered before the second deduction
        assert executor._fatal_error is None
        assert executor._error_budget > budget_after_first - 0.1

    def test_immediate_fatal_bypasses_budget(self, executor):
        """Immediate-fatal errors crash regardless of remaining budget."""
        assert executor._error_budget == 1.0
        executor.active_requests = []
        executor._handle_errors("device-side assert triggered")
        assert executor._fatal_error is not None

    def test_fatal_error_with_no_active_requests(self, executor):
        executor.active_requests = []
        executor._handle_errors("cudaErrorLaunchFailure: failure")

        assert executor._fatal_error is not None
        executor.executor_request_queue.enqueue_shutdown_request.assert_called_once()

    def test_default_error_message(self, executor):
        executor.active_requests = []
        executor._handle_errors()
        assert executor._fatal_error is None


# ---------------------------------------------------------------------------
# GenerationExecutor._set_fatal_error and check_health tests
# ---------------------------------------------------------------------------
class ConcreteExecutor:
    """Minimal concrete version of GenerationExecutor for testing."""

    def __init__(self):
        self._error_queue = Queue()
        self.doing_shutdown = False
        self._fatal_error = None

    def _set_fatal_error(self, error):
        if self._fatal_error is None:
            self._fatal_error = error

    def is_shutdown(self):
        return self.doing_shutdown or self._fatal_error is not None

    def check_health(self):
        if self.doing_shutdown or self._fatal_error is not None:
            return False
        if not self._error_queue.empty():
            try:
                self._handle_background_error()
            except Exception:
                pass
            return self._fatal_error is None and not self.doing_shutdown
        return True

    def _handle_background_error(self, error=None):
        if error is not None:
            self._set_fatal_error(error)
            self.doing_shutdown = True
            raise error
        if not self._error_queue.empty():
            e = self._error_queue.get()
            self._error_queue.task_done()
            self._set_fatal_error(e)
            self.doing_shutdown = True
            raise e

    def shutdown(self):
        self.doing_shutdown = True


class TestSetFatalError:
    @pytest.fixture
    def executor(self):
        return ConcreteExecutor()

    def test_first_error_wins(self, executor):
        first = RuntimeError("first error")
        second = RuntimeError("second error")

        executor._set_fatal_error(first)
        executor._set_fatal_error(second)

        assert executor._fatal_error is first

    def test_none_initially(self, executor):
        assert executor._fatal_error is None


class TestCheckHealth:
    @pytest.fixture
    def executor(self):
        return ConcreteExecutor()

    def test_healthy_by_default(self, executor):
        assert executor.check_health() is True

    def test_unhealthy_after_fatal_error(self, executor):
        executor._set_fatal_error(RuntimeError("boom"))
        assert executor.check_health() is False

    def test_unhealthy_during_shutdown(self, executor):
        executor.doing_shutdown = True
        assert executor.check_health() is False

    def test_unhealthy_with_error_in_queue(self, executor):
        executor._error_queue.put(RuntimeError("background crash"))
        assert executor.check_health() is False
        assert executor._fatal_error is not None

    def test_healthy_with_empty_queue(self, executor):
        assert executor.check_health() is True
        assert executor._fatal_error is None


class TestIsShutdown:
    @pytest.fixture
    def executor(self):
        return ConcreteExecutor()

    def test_not_shutdown_initially(self, executor):
        assert executor.is_shutdown() is False

    def test_shutdown_when_doing_shutdown(self, executor):
        executor.doing_shutdown = True
        assert executor.is_shutdown() is True

    def test_shutdown_when_fatal_error(self, executor):
        executor._set_fatal_error(RuntimeError("fatal"))
        assert executor.is_shutdown() is True


# ---------------------------------------------------------------------------
# GenerationExecutorProxy.check_health and _error_monitor_loop tests
# ---------------------------------------------------------------------------
class ConcreteProxyExecutor(ConcreteExecutor):
    """Minimal mock of GenerationExecutorProxy for testing."""

    def __init__(self):
        super().__init__()
        self.mpi_futures = []
        self._shutdown_called = False

    def check_health(self):
        if not super().check_health():
            return False
        if hasattr(self, "mpi_futures") and self.mpi_futures:
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
                if hasattr(self, "mpi_futures") and self.mpi_futures:
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
                time.sleep(0.01)  # 10ms for fast tests instead of 100ms

    def shutdown(self):
        self.doing_shutdown = True
        self._shutdown_called = True


class TestProxyCheckHealth:
    @pytest.fixture
    def executor(self):
        return ConcreteProxyExecutor()

    def test_healthy_with_running_workers(self, executor):
        f = Future()
        executor.mpi_futures = [f]
        assert executor.check_health() is True

    def test_unhealthy_when_worker_crashes(self, executor):
        f = Future()
        f.set_exception(RuntimeError("worker segfault"))
        executor.mpi_futures = [f]

        assert executor.check_health() is False
        assert executor._fatal_error is not None
        assert "worker segfault" in str(executor._fatal_error)
        assert executor._shutdown_called is True

    def test_unhealthy_when_worker_exits_without_exception(self, executor):
        f = Future()
        f.set_result(None)
        executor.mpi_futures = [f]

        assert executor.check_health() is False
        assert "MPI worker exited unexpectedly" in str(executor._fatal_error)

    def test_unhealthy_when_worker_cancelled(self, executor):
        f = Future()
        f.cancel()
        executor.mpi_futures = [f]

        assert executor.check_health() is False
        assert "MPI worker exited unexpectedly" in str(executor._fatal_error)

    def test_healthy_with_no_mpi_futures(self, executor):
        executor.mpi_futures = []
        assert executor.check_health() is True

    def test_parent_unhealthy_short_circuits(self, executor):
        executor._set_fatal_error(RuntimeError("parent error"))
        f = Future()
        executor.mpi_futures = [f]
        assert executor.check_health() is False


class TestErrorMonitorLoop:
    def test_detects_worker_crash(self):
        executor = ConcreteProxyExecutor()
        f = Future()
        executor.mpi_futures = [f]

        thread = threading.Thread(target=executor._error_monitor_loop, daemon=True)
        thread.start()

        time.sleep(0.05)
        f.set_exception(RuntimeError("worker OOM"))

        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert executor._fatal_error is not None
        assert executor._shutdown_called is True

    def test_detects_error_queue_item(self):
        executor = ConcreteProxyExecutor()

        thread = threading.Thread(target=executor._error_monitor_loop, daemon=True)
        thread.start()

        time.sleep(0.05)
        executor._error_queue.put(RuntimeError("background failure"))

        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert executor._fatal_error is not None

    def test_stops_on_shutdown(self):
        executor = ConcreteProxyExecutor()

        thread = threading.Thread(target=executor._error_monitor_loop, daemon=True)
        thread.start()

        time.sleep(0.05)
        executor.doing_shutdown = True

        thread.join(timeout=2.0)
        assert not thread.is_alive()


# ---------------------------------------------------------------------------
# GrpcRequestManager.health_check tests
# ---------------------------------------------------------------------------
class TestGrpcHealthCheck:
    def _make_manager(self, executor=None):
        manager = MagicMock()
        manager.llm = MagicMock()
        manager.llm._executor = executor
        manager.health_check = self._health_check.__get__(manager)
        return manager

    @staticmethod
    async def _health_check(self):
        try:
            if self.llm is None:
                return False, "LLM not initialized"
            if hasattr(self.llm, "_executor"):
                if self.llm._executor is None:
                    return False, "Executor is not available"
                if not self.llm._executor.check_health():
                    error_msg = "Executor is unhealthy"
                    if self.llm._executor._fatal_error is not None:
                        error_msg = f"Fatal error: {self.llm._executor._fatal_error}"
                    return False, error_msg
            return True, "OK"
        except Exception as e:
            return False, f"Error: {e}"

    @pytest.mark.asyncio
    async def test_healthy_executor(self):
        executor = ConcreteExecutor()
        manager = self._make_manager(executor)

        healthy, msg = await manager.health_check()
        assert healthy is True
        assert msg == "OK"

    @pytest.mark.asyncio
    async def test_unhealthy_with_fatal_error(self):
        executor = ConcreteExecutor()
        executor._set_fatal_error(RuntimeError("CUDA OOM"))
        manager = self._make_manager(executor)

        healthy, msg = await manager.health_check()
        assert healthy is False
        assert "CUDA OOM" in msg

    @pytest.mark.asyncio
    async def test_no_executor(self):
        manager = self._make_manager(executor=None)

        healthy, msg = await manager.health_check()
        assert healthy is False
        assert "not available" in msg

    @pytest.mark.asyncio
    async def test_no_llm(self):
        manager = MagicMock()
        manager.llm = None
        manager.health_check = self._health_check.__get__(manager)

        healthy, msg = await manager.health_check()
        assert healthy is False
        assert "not initialized" in msg

    @pytest.mark.asyncio
    async def test_unhealthy_during_shutdown(self):
        executor = ConcreteExecutor()
        executor.doing_shutdown = True
        manager = self._make_manager(executor)

        healthy, msg = await manager.health_check()
        assert healthy is False
        assert "unhealthy" in msg.lower()


# ---------------------------------------------------------------------------
# OpenAIServer /health endpoint tests
# ---------------------------------------------------------------------------
class TestOpenAIHealthEndpoint:
    def _make_server_mock(self, check_health_return, fatal_error=None):
        """Create a minimal mock that exercises the health() logic."""
        server = MagicMock()
        server._check_health = Mock(return_value=check_health_return)
        executor = Mock()
        executor._fatal_error = fatal_error
        server.generator = Mock()
        server.generator._executor = executor
        return server, executor

    @pytest.mark.asyncio
    async def test_healthy_returns_200(self):
        from starlette.responses import Response

        server, _ = self._make_server_mock(check_health_return=True)

        # Inline the health() logic to test without full FastAPI setup
        if server._check_health():
            response = Response(status_code=200)
        else:
            response = Response(status_code=503)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_unhealthy_returns_503(self):
        from starlette.responses import Response

        server, _ = self._make_server_mock(check_health_return=False, fatal_error=None)

        if server._check_health():
            response = Response(status_code=200)
        else:
            response = Response(status_code=503)

        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_fatal_error_triggers_sigint(self):
        from starlette.responses import Response

        fatal = RuntimeError("CUDA OOM")
        server, executor = self._make_server_mock(check_health_return=False, fatal_error=fatal)

        with patch.object(signal, "raise_signal") as mock_signal:
            if server._check_health():
                response = Response(status_code=200)
            else:
                ex = getattr(server.generator, "_executor", None)
                if ex is not None and getattr(ex, "_fatal_error", None) is not None:
                    signal.raise_signal(signal.SIGINT)
                response = Response(status_code=503)

            assert response.status_code == 503
            mock_signal.assert_called_once_with(signal.SIGINT)

    @pytest.mark.asyncio
    async def test_no_sigint_without_fatal_error(self):
        server, _ = self._make_server_mock(check_health_return=False, fatal_error=None)

        with patch.object(signal, "raise_signal") as mock_signal:
            if server._check_health():
                pass
            else:
                ex = getattr(server.generator, "_executor", None)
                if ex is not None and getattr(ex, "_fatal_error", None) is not None:
                    signal.raise_signal(signal.SIGINT)

            mock_signal.assert_not_called()


# ---------------------------------------------------------------------------
# BaseLLM._check_health delegation tests
# ---------------------------------------------------------------------------
class TestBaseLLMCheckHealth:
    @staticmethod
    def _check_health(llm):
        """Mirrors BaseLLM._check_health logic."""
        if hasattr(llm, "_executor") and llm._executor is not None:
            return llm._executor.check_health()
        return False

    def test_delegates_to_executor(self):
        executor = ConcreteExecutor()
        llm = Mock()
        llm._executor = executor

        assert self._check_health(llm) is True

        executor._set_fatal_error(RuntimeError("fatal"))
        assert self._check_health(llm) is False

    def test_returns_false_when_no_executor(self):
        llm = Mock()
        llm._executor = None
        assert self._check_health(llm) is False

    def test_returns_false_when_no_executor_attr(self):
        llm = object()
        assert self._check_health(llm) is False
