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
"""Unit tests for AwaitResponseHelper.__call__ event-loop crash handling.

When PyExecutor's event-loop thread dies (e.g. KV cache OOM), every
pending ``GenerationResult`` parked in ``queue.get()`` / ``aqueue.get()``
must wake up with a meaningful ``ErrorResponse`` rather than hang
forever. See nvbug 6038228 and PR #12735.

These tests bind the real ``AwaitResponseHelper.__call__`` /
``_broadcast_event_loop_error`` to lightweight stubs, so they need
neither GPUs nor models.
"""

import asyncio
import datetime
import queue as _stdlib_queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm.executor.base_worker import AwaitResponseHelper
from tensorrt_llm.executor.ipc import IpcQueue
from tensorrt_llm.executor.postproc_worker import PostprocWorker
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.utils import ErrorResponse, WorkerFatalError
from tensorrt_llm.executor.worker import GenerationExecutorWorker
from tensorrt_llm.llmapi.mpi_session import (
    MpiCommSession,
    MpiPoolSession,
    RemoteMpiCommSessionClient,
)


class _EngineStub:
    """Stub for self.worker.engine: returns whatever the test plugged in."""

    def __init__(
        self,
        await_responses_result=None,
        await_responses_raises=None,
        event_loop_error=None,
        is_shutdown=False,
    ):
        self._await_responses_result = await_responses_result or []
        self._await_responses_raises = await_responses_raises
        self._event_loop_error = event_loop_error
        self.is_shutdown = is_shutdown
        self.calls = 0

    def await_responses(self, timeout: datetime.timedelta):
        self.calls += 1
        if self._await_responses_raises is not None:
            raise self._await_responses_raises
        return list(self._await_responses_result)


class _ResultStub:
    """Minimal GenerationResult: just exposes a queue."""

    def __init__(self):
        self.queue = _stdlib_queue.Queue()


class _WorkerStub:
    """Stub for BaseWorker exposing only the attributes the helper touches."""

    def __init__(self, engine, num_pending: int = 1):
        self.engine = engine
        self._results = {cid: _ResultStub() for cid in range(1, num_pending + 1)}
        self.popped = []
        self.result_queue = None
        self.postproc_queues = None
        self.system_result_queue = None
        self.system_fatal_event = threading.Event()

        # Echoed straight back so __call__'s filter is a no-op.
        def _engine_response_callback(r):
            return r

        self._engine_response_callback = _engine_response_callback

    def return_queue(self, client_id: int):
        return self._results[client_id].queue

    def _pop_result(self, client_id: int):
        self.popped.append(client_id)
        self._results.pop(client_id, None)


def _make_helper(engine, num_pending: int = 1):
    helper = AwaitResponseHelper.__new__(AwaitResponseHelper)
    helper.worker = _WorkerStub(engine, num_pending=num_pending)
    helper.handler_kind = AwaitResponseHelper.HandlerKind.unknown
    helper.enable_postprocprocess_parallel = False
    helper.temp_error_responses = _stdlib_queue.Queue()
    return helper


class TestAwaitResponseHelperEventLoopError:
    def test_normal_path_returns_true(self):
        """No engine error and no responses: ManagedThread should keep going."""
        engine = _EngineStub(await_responses_result=[])
        helper = _make_helper(engine, num_pending=1)

        assert helper(timeout=0.01) is True
        # No ErrorResponse should have been pushed.
        for rs in helper.worker._results.values():
            assert rs.queue.empty()
        assert helper.worker.popped == []

    def test_broadcasts_when_await_responses_raises(self):
        """Defensive: any exception out of engine.await_responses triggers broadcast."""
        original = RuntimeError("Event loop terminated with error: KV OOM")
        engine = _EngineStub(await_responses_raises=original)
        helper = _make_helper(engine, num_pending=2)

        assert helper(timeout=0.01) is False  # ManagedThread should stop

        # Each pending GenerationResult got an ErrorResponse.
        for cid in (1, 2):
            err = helper.worker._results.get(cid)
            assert err is None, "result should have been popped"
        # popped order is iteration order over dict keys (insertion order in py3.7+)
        assert sorted(helper.worker.popped) == [1, 2]

    def test_broadcasts_when_event_loop_error_set_after_empty_response(self):
        """Broadcast must fire even when await_responses returns [] silently.

        ``_await_any_response`` returns ``[]`` on shutdown without raising,
        but ``_event_loop_error`` is still stashed on the engine.
        """
        original = RuntimeError("KV cache OOM")
        engine = _EngineStub(await_responses_result=[], event_loop_error=original, is_shutdown=True)
        helper = _make_helper(engine, num_pending=3)

        assert helper(timeout=0.01) is False
        assert sorted(helper.worker.popped) == [1, 2, 3]

    def test_pushed_response_is_error_response_with_message(self):
        """Pushed item is an ErrorResponse carrying the original error text."""
        original = RuntimeError("KV cache OOM at iteration 42")
        engine = _EngineStub(event_loop_error=original, is_shutdown=True)
        # Capture queue refs before they get popped from _results.
        helper = _make_helper(engine, num_pending=1)
        result_queue = helper.worker.return_queue(client_id=1)

        helper(timeout=0.01)

        item = result_queue.get_nowait()
        assert isinstance(item, ErrorResponse)
        assert item.client_id == 1
        assert "KV cache OOM" in item.error_msg
        assert "Event loop terminated" in item.error_msg

    def test_no_pending_results_returns_false_quietly(self):
        """Crash with no pending requests still stops the thread cleanly."""
        original = RuntimeError("crash")
        engine = _EngineStub(event_loop_error=original, is_shutdown=True)
        helper = _make_helper(engine, num_pending=0)

        assert helper(timeout=0.01) is False
        assert helper.worker.popped == []

    def test_broadcast_helper_idempotent_via_pop(self):
        """Calling _broadcast_event_loop_error twice is safe (second is a no-op)."""
        original = RuntimeError("crash")
        engine = _EngineStub(event_loop_error=original, is_shutdown=True)
        helper = _make_helper(engine, num_pending=2)

        assert helper._broadcast_event_loop_error(original) is False
        assert sorted(helper.worker.popped) == [1, 2]
        # second time around: nothing left to wake.
        assert helper._broadcast_event_loop_error(original) is False

    def test_ipc_sends_system_fatal_marker_without_request_local_cleanup(self):
        """IPC mode delegates atomic fan-out and shutdown to the proxy."""
        original = RuntimeError("topology-poisoned transfer buffer")
        engine = _EngineStub(event_loop_error=original, is_shutdown=True)
        helper = _make_helper(engine, num_pending=2)
        system_result_queue = _stdlib_queue.Queue()
        helper.worker.result_queue = system_result_queue
        helper.worker.system_result_queue = system_result_queue

        assert helper(timeout=0.01) is False

        marker = system_result_queue.get_nowait()
        assert isinstance(marker, WorkerFatalError)
        assert "topology-poisoned transfer buffer" in marker.error_msg
        assert helper.worker.system_fatal_event.is_set()
        assert sorted(helper.worker._results) == [1, 2]
        assert helper.worker.popped == []

    def test_ipc_fatal_bypasses_normal_response_handler(self):
        """A post-processing stall cannot delay the direct fatal marker."""
        original = RuntimeError("topology-poisoned transfer buffer")
        engine = _EngineStub(
            await_responses_result=[object()],
            event_loop_error=original,
            is_shutdown=True,
        )
        helper = _make_helper(engine)
        system_result_queue = _stdlib_queue.Queue()
        postproc_queue = Mock()
        helper.worker.postproc_queues = [postproc_queue]
        helper.worker.system_result_queue = system_result_queue
        helper.responses_handler = Mock(
            side_effect=AssertionError("normal response path must be bypassed")
        )

        assert helper(timeout=0.01) is False

        assert isinstance(system_result_queue.get_nowait(), WorkerFatalError)
        postproc_queue.put_noblock.assert_called_once_with(PostprocWorker.FatalStop(), retry=0)
        helper.responses_handler.assert_not_called()

    def test_ipc_fatal_retirement_survives_postproc_stop_failure(self):
        original = RuntimeError("topology-poisoned transfer buffer")
        helper = _make_helper(_EngineStub(event_loop_error=original, is_shutdown=True))
        system_result_queue = _stdlib_queue.Queue()
        postproc_queue = Mock()
        postproc_queue.put_noblock.side_effect = RuntimeError("postproc stalled")
        helper.worker.postproc_queues = [postproc_queue]
        helper.worker.system_result_queue = system_result_queue

        assert helper(timeout=0.01) is False
        assert helper.worker.system_fatal_event.is_set()
        assert isinstance(system_result_queue.get_nowait(), WorkerFatalError)

    def test_ipc_missing_system_queue_exits_worker_instead_of_failing_open(self):
        original = RuntimeError("topology-poisoned transfer buffer")
        helper = _make_helper(_EngineStub())
        helper.worker.postproc_queues = [Mock()]

        with pytest.raises(RuntimeError, match="system result queue"):
            helper._broadcast_event_loop_error(original)

        assert helper.worker.system_fatal_event.is_set()


class TestProxyWorkerFatalError:
    @staticmethod
    def _make_proxy(num_pending: int = 2):
        proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
        proxy.result_queue = _stdlib_queue.Queue()
        proxy._results = {client_id: _ResultStub() for client_id in range(1, num_pending + 1)}
        proxy._worker_fatal_error = None
        proxy._fatal_error = None
        proxy.doing_shutdown = False
        proxy._error_queue = _stdlib_queue.Queue()
        proxy._shutdown_event = threading.Event()
        return proxy

    def test_marker_fails_all_pending_results_and_wakes_error_monitor(self):
        proxy = self._make_proxy()
        result_queues = {client_id: result.queue for client_id, result in proxy._results.items()}
        proxy.result_queue.put(WorkerFatalError("topology poison requires process restart"))

        assert proxy.dispatch_result_task() is False
        assert proxy._results == {}
        assert proxy._shutdown_event.is_set()
        assert isinstance(proxy._worker_fatal_error, RuntimeError)
        assert isinstance(proxy._error_queue.get_nowait(), RuntimeError)
        for client_id, result_queue in result_queues.items():
            response = result_queue.get_nowait()
            assert isinstance(response, ErrorResponse)
            assert response.client_id == client_id
            assert "process restart" in response.error_msg

    def test_marker_rejects_later_work(self):
        proxy = self._make_proxy(num_pending=0)
        proxy._worker_fatal_error = RuntimeError("worker quarantined")

        with pytest.raises(RuntimeError, match="worker quarantined"):
            proxy._raise_if_worker_unavailable()

    def test_inflight_cancel_requires_disposable_worker_processes(self):
        proxy_supports = GenerationExecutorProxy._supports_inflight_cancel_process_restart
        pool_session = object.__new__(MpiPoolSession)
        pool_session.mpi_pool = None
        comm_session = object.__new__(MpiCommSession)
        comm_session.mpi_pool = None
        comm_session.thread_pool = None
        comm_session.owns_mpi_pool = False

        assert proxy_supports(pool_session)
        assert not proxy_supports(comm_session)
        assert not proxy_supports(object.__new__(RemoteMpiCommSessionClient))

    def test_submit_race_removes_result_and_raises(self):
        proxy = self._make_proxy(num_pending=0)
        proxy._start_dispatch_threads = Mock()
        proxy._get_next_client_id = Mock(return_value=41)
        proxy._get_logprob_params = Mock(return_value=None)
        proxy._handle_background_error = Mock()
        proxy.request_queue = Mock()
        proxy.request_queue.put.side_effect = lambda _: setattr(
            proxy, "_worker_fatal_error", RuntimeError("worker quarantined")
        )
        request = Mock(id=None, disaggregated_params=None)
        request.set_id.side_effect = lambda request_id: setattr(request, "id", request_id)

        with (
            patch(
                "tensorrt_llm.executor.proxy.GenerationResult",
                side_effect=lambda *_args, **_kwargs: _ResultStub(),
            ),
            pytest.raises(RuntimeError, match="worker quarantined"),
        ):
            proxy.submit(request)

        assert 41 not in proxy._results
        proxy._handle_background_error.assert_not_called()

    def test_worker_fatal_pre_shutdown_does_not_cross_request_socket_threads(self):
        request_queue = IpcQueue(is_server=True, name="fatal_pre_shutdown_test")
        request_queue._zmq_debug_enabled = True
        request_queue._check_thread_safety()
        owner_thread_id = request_queue._zmq_thread_id
        proxy = self._make_proxy(num_pending=0)
        proxy.workers_started = True
        proxy._worker_fatal_error = RuntimeError("worker quarantined")
        proxy.request_queue = request_queue
        proxy.mpi_futures = []
        errors = []

        def pre_shutdown():
            try:
                proxy.pre_shutdown()
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=pre_shutdown)
        try:
            thread.start()
            thread.join(timeout=1.0)

            assert not thread.is_alive()
            assert errors == []
            assert request_queue._zmq_thread_id == owner_thread_id
        finally:
            request_queue.close()

    def test_worker_fatal_shutdown_uses_bounded_session_abort(self):
        proxy = self._make_proxy(num_pending=0)
        proxy.workers_started = True
        proxy.doing_shutdown = True
        proxy._worker_fatal_error = RuntimeError("worker quarantined")
        worker_future = Mock()
        proxy.mpi_futures = [worker_future]
        proxy.mpi_session = Mock()
        proxy.dispatch_result_thread = None
        proxy.rpc_client = None
        proxy.request_queue = Mock()
        proxy.worker_init_status_queue = Mock()
        proxy.result_queue = Mock()
        proxy._resource_governor_queue = None
        proxy._handle_background_error = Mock()

        proxy.shutdown()

        worker_future.result.assert_not_called()
        proxy.mpi_session.shutdown_abort.assert_called_once_with(
            grace=proxy.WORKER_FATAL_SHUTDOWN_GRACE_SECONDS,
            reason=proxy._worker_fatal_error,
        )
        proxy.mpi_session.shutdown.assert_not_called()


def test_unsafe_engine_quarantines_complete_executor_worker():
    worker = GenerationExecutorWorker.__new__(GenerationExecutorWorker)
    worker.doing_shutdown = False
    worker.await_response_thread = Mock()
    engine = Mock()
    engine.can_enqueue_requests.return_value = False
    engine._unsafe_transfer_shutdown = True
    worker.engine = engine
    worker.llm_args = SimpleNamespace(backend="pytorch")
    worker._executor_config = None
    worker.checkpoint_loader = Mock()

    try:
        with patch("torch.distributed.destroy_process_group") as destroy_group:
            worker.shutdown()

        engine.shutdown.assert_called_once_with()
        assert worker.engine is engine
        worker.checkpoint_loader.cleanup.assert_not_called()
        destroy_group.assert_not_called()
        assert any(
            quarantined is worker
            for quarantined in GenerationExecutorWorker._UNSAFE_ENGINE_SHUTDOWN_QUARANTINE
        )
    finally:
        with GenerationExecutorWorker._UNSAFE_ENGINE_SHUTDOWN_QUARANTINE_LOCK:
            GenerationExecutorWorker._UNSAFE_ENGINE_SHUTDOWN_QUARANTINE[:] = [
                quarantined
                for quarantined in GenerationExecutorWorker._UNSAFE_ENGINE_SHUTDOWN_QUARANTINE
                if quarantined is not worker
            ]


def test_postproc_fatal_stop_does_not_forward_proxy_sentinel():
    class _PullPipe:
        async def get_async(self):
            return [PostprocWorker.FatalStop()]

    worker = PostprocWorker.__new__(PostprocWorker)
    worker._pull_pipe = _PullPipe()
    worker._to_stop = asyncio.Event()

    async def collect_batches():
        return [batch async for batch in worker._mainloop()]

    assert asyncio.run(collect_batches()) == []
    assert worker._to_stop.is_set()
