# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control-loop and owner-teardown regressions; real MPI coverage is separate."""

import threading
from collections import deque
from concurrent.futures import CancelledError, Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import zmq

from tensorrt_llm.llmapi import mpi_session as mpi

pytestmark = pytest.mark.cpu_only


def _future(result=None, error=None):
    future = Future()
    if error is None:
        future.set_result(result)
    else:
        future.set_exception(error)
    return future


def _task():
    return 42


class _Queue:
    def __init__(self, messages, on_poll=None):
        self.messages = deque(messages)
        self.sent = []
        self.socket = Mock()
        self.closed = False
        self.on_poll = on_poll
        self.thread_ids = set()
        self.polls = 0

    def poll(self, timeout):
        self.thread_ids.add(threading.get_ident())
        self.polls += 1
        assert self.polls < 100, "server failed to make progress"
        if self.on_poll:
            self.on_poll(self)
        return bool(self.messages)

    def get(self):
        self.thread_ids.add(threading.get_ident())
        return self.messages.popleft()

    def put(self, message):
        self.thread_ids.add(threading.get_ident())
        self.sent.append(message)

    def close(self):
        self.closed = True


def _server(queue, batches):
    server = object.__new__(mpi.RemoteMpiCommSessionServer)
    server.queue = queue
    server.session = SimpleNamespace(
        n_workers=2, submit=Mock(side_effect=batches), shutdown=Mock(), abort=Mock()
    )
    server._shutdown_session = Mock()
    return server


def test_failed_task_publishes_without_a_final_collective(monkeypatch):
    barrier = Mock()
    monkeypatch.setattr(mpi, "mpi_barrier", barrier)
    monkeypatch.setattr(mpi, "mpi_rank", lambda: 0)
    monkeypatch.setattr(mpi, "mpi_world_size", lambda: 2)

    def fail():
        raise ValueError("injected failure")

    with pytest.raises(ValueError, match="injected failure"):
        mpi.RemoteMpiCommSessionServer.task_wrapper(fail)
    barrier.assert_called_once()


def test_stop_is_processed_while_all_futures_are_pending(monkeypatch):
    monkeypatch.setattr(mpi, "_mgmn_shutdown_grace_seconds", lambda: 0.01)
    queue = _Queue([mpi.RemoteTask(_task, (), {}), None])
    server = _server(queue, [[Future(), Future()]])
    with pytest.raises(RuntimeError, match="shutdown deadline expired"):
        server.serve()
    server.session.submit.assert_called_once()
    server._shutdown_session.assert_called_once_with(0)
    assert queue.closed


@pytest.mark.parametrize("peer_fails", [False, True], ids=["stuck-peer", "all-failed"])
@pytest.mark.parametrize("send_stop", [False, True], ids=["deadline", "stop"])
@pytest.mark.parametrize("error_type", [ValueError, SystemExit, KeyboardInterrupt])
def test_async_failure_waits_for_client_without_starting_more_work(
    monkeypatch: pytest.MonkeyPatch,
    peer_fails: bool,
    send_stop: bool,
    error_type: type[BaseException],
) -> None:
    now = [100.0]
    monkeypatch.setattr(mpi.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(mpi, "_mgmn_shutdown_grace_seconds", lambda: 10)
    failed, peer = Future(), Future()
    error = error_type("rank failed")
    queue = _Queue([mpi.RemoteTask(_task, (0,), {}), mpi.RemoteTask(_task, (1,), {})])
    server = _server(queue, [[failed, peer]])

    def advance(queue):
        now[0] += 1
        if queue.polls == 3:
            # The next batch is already queued when the first batch fails.
            failed.set_exception(error)
            if peer_fails:
                peer.set_exception(ValueError("peer failed"))
        if queue.sent:
            server._shutdown_session.assert_not_called()
            server.session.submit.assert_called_once()
        if queue.polls == 4:
            # Requests arriving after the failure must not start either.
            queue.messages.append(mpi.RemoteTask(_task, (2,), {}))
        if send_stop and queue.polls == 8:
            queue.messages.append(None)

    queue.on_poll = advance
    with pytest.raises(RuntimeError, match="asynchronous task failed") as raised:
        server.serve()
    assert raised.value.__cause__ is error
    assert queue.sent == [mpi.RemoteWorkerDeath(error_type.__name__, "rank failed")]
    assert queue.thread_ids == {threading.get_ident()}
    assert queue.polls == (8 if send_stop else 13)
    server.session.submit.assert_called_once()
    # A later stop request cannot extend the original failure deadline.
    server._shutdown_session.assert_called_once_with(5 if send_stop else 0)
    assert queue.closed


def test_failed_error_delivery_still_shuts_down():
    queue = _Queue([mpi.RemoteTask(_task, (), {})])
    queue.put = Mock(side_effect=zmq.Again())
    server = _server(queue, [[Future(), _future(error=ValueError("failed"))]])
    with pytest.raises(zmq.Again):
        server.serve()
    server._shutdown_session.assert_called_once()
    assert queue.closed
    queue.socket.setsockopt.assert_any_call(zmq.SNDTIMEO, 1000)
    queue.socket.setsockopt.assert_any_call(zmq.LINGER, 1000)


@pytest.mark.parametrize("error_type", [ValueError, SystemExit, KeyboardInterrupt])
def test_sync_error_does_not_leak_responses_into_next_batch(
    error_type: type[BaseException],
) -> None:
    def stop_after_responses(queue):
        if len(queue.sent) == 2:
            queue.messages.append(None)

    queue = _Queue([mpi.RemoteTask(_task, (), {}, True)] * 2, on_poll=stop_after_responses)
    error = error_type("first batch failed")
    server = _server(queue, [[_future(7), _future(error=error)], [_future(8), _future(9)]])
    server.serve()
    assert queue.sent == [error, [8, 9]]
    assert queue.thread_ids == {threading.get_ident()}
    assert server.session.submit.call_count == 2


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
def test_cancelled_future_sends_one_error_response(sync: bool) -> None:
    def stop_after_response(queue: _Queue) -> None:
        if queue.sent:
            queue.messages.append(None)

    cancelled = Future()
    assert cancelled.cancel()
    queue = _Queue([mpi.RemoteTask(_task, (), {}, sync)], on_poll=stop_after_response)
    server = _server(queue, [[cancelled, _future(7)]])
    if sync:
        server.serve()
        assert len(queue.sent) == 1
        assert isinstance(queue.sent[0], CancelledError)
    else:
        with pytest.raises(RuntimeError, match="asynchronous task failed") as raised:
            server.serve()
        assert isinstance(raised.value.__cause__, CancelledError)
        assert queue.sent == [mpi.RemoteWorkerDeath("CancelledError", "")]
    server._shutdown_session.assert_called_once()
    assert queue.closed


@pytest.mark.parametrize("operation", ["exception", "result"])
def test_server_local_interrupt_is_not_reported_as_worker_death(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    future = _future(7)
    interrupt = KeyboardInterrupt("server interrupted")
    monkeypatch.setattr(future, operation, Mock(side_effect=interrupt))
    queue = _Queue([mpi.RemoteTask(_task, (), {})])
    server = _server(queue, [[future, _future(8)]])
    with pytest.raises(KeyboardInterrupt) as raised:
        server.serve()
    assert raised.value is interrupt
    assert queue.sent == []
    server._shutdown_session.assert_called_once()
    assert queue.closed


def test_sync_multiple_failures_send_one_response():
    def stop_after_response(queue):
        if queue.sent:
            queue.messages.append(None)

    queue = _Queue([mpi.RemoteTask(_task, (), {}, True)], on_poll=stop_after_response)
    first_error = ValueError("first")
    server = _server(queue, [[_future(error=first_error), _future(error=ValueError("second"))]])
    server.serve()
    assert queue.sent == [first_error]


def test_next_batch_waits_for_every_previous_future():
    pending = Future()
    queue = _Queue([mpi.RemoteTask(_task, (0,), {}), mpi.RemoteTask(_task, (1,), {}, True)])
    server = _server(queue, [[_future(0), pending], [_future(1), _future(1)]])

    def advance(queue):
        if queue.polls == 3:
            assert server.session.submit.call_count == 1
            pending.set_result(0)
        if queue.sent:
            queue.messages.append(None)

    queue.on_poll = advance
    server.serve()
    assert [call.args[2] for call in server.session.submit.call_args_list] == [0, 1]
    assert queue.sent == [[1, 1]]


def test_sync_error_bounds_peer_drain(monkeypatch):
    clock = iter([0, 0, 61, 61])
    monkeypatch.setattr(mpi.time, "monotonic", lambda: next(clock))
    queue = _Queue([mpi.RemoteTask(_task, (), {}, True)])
    error = ValueError("rank failed")
    server = _server(queue, [[_future(error=error), Future()]])
    with pytest.raises(RuntimeError, match="did not drain"):
        server.serve()
    assert queue.sent == [error]
    server._shutdown_session.assert_called_once()


def test_stop_drains_preceding_async_batches():
    pending = Future()

    def release_first_batch_at_stop(queue):
        if queue.polls == 4:
            pending.set_result(0)

    queue = _Queue(
        [mpi.RemoteTask(_task, (i,), {}) for i in range(3)] + [None],
        on_poll=release_first_batch_at_stop,
    )
    server = _server(queue, [[_future(0), pending], [_future(1)] * 2, [_future(2)] * 2])
    server.serve()
    assert [call.args[2] for call in server.session.submit.call_args_list] == [0, 1, 2]
    server._shutdown_session.assert_called_once()
    assert queue.closed


def test_stop_passes_remaining_deadline_to_final_shutdown(monkeypatch):
    monkeypatch.setattr(mpi, "_mgmn_shutdown_grace_seconds", lambda: 10)
    clock = iter([100, 103])
    monkeypatch.setattr(mpi.time, "monotonic", lambda: next(clock))
    queue = _Queue([mpi.RemoteTask(_task, (), {}), None])
    server = _server(queue, [[_future(1), _future(2)]])
    server.serve()
    server._shutdown_session.assert_called_once_with(7)


@pytest.mark.parametrize("failure", ["session", "executor", None])
def test_final_owner_shutdown_closes_shared_executor(monkeypatch, failure):
    class TestSession(mpi.MpiCommSession):
        def __del__(self):
            pass

    session = object.__new__(TestSession)
    session.mpi_pool = object()
    session.shutdown = Mock(side_effect=RuntimeError("shutdown") if failure == "session" else None)
    session.abort = Mock()
    executor = Mock()
    executor.__exit__ = Mock(
        side_effect=RuntimeError("executor") if failure == "executor" else None
    )
    monkeypatch.setattr(mpi.MPINodeState, "_global_mpi_pool", session.mpi_pool)
    monkeypatch.setattr(mpi.MPINodeState, "_global_comm_executor", executor)
    server = object.__new__(mpi.RemoteMpiCommSessionServer)
    server.session = session
    server._shutdown_session(1)
    session.shutdown.assert_called_once_with()
    if failure:
        session.abort.assert_called_once_with()
    else:
        executor.__exit__.assert_called_once_with(None, None, None)
        session.abort.assert_not_called()
        assert mpi.MPINodeState._global_comm_executor is None
        assert mpi.MPINodeState._global_mpi_pool is None


@pytest.mark.parametrize("comm_session", [False, True])
def test_unrelated_global_executor_is_not_closed(monkeypatch, comm_session):
    executor = Mock()
    executor.__exit__ = Mock()
    monkeypatch.setattr(mpi.MPINodeState, "_global_comm_executor", executor)
    monkeypatch.setattr(mpi.MPINodeState, "_global_mpi_pool", object())
    server = object.__new__(mpi.RemoteMpiCommSessionServer)
    if comm_session:

        class TestSession(mpi.MpiCommSession):
            def __del__(self):
                pass

        server.session = object.__new__(TestSession)
        server.session.mpi_pool = object()
        server.session.shutdown = Mock()
        server.session.abort = Mock()
    else:
        server.session = SimpleNamespace(shutdown=Mock(), abort=Mock())
    server._shutdown_session(1)
    executor.__exit__.assert_not_called()
    server.session.abort.assert_not_called()


@pytest.mark.parametrize("shutdown_seconds", [7, 10])
def test_global_executor_close_uses_remaining_session_deadline(monkeypatch, shutdown_seconds):
    class TestSession(mpi.MpiCommSession):
        def __del__(self):
            pass

    session = object.__new__(TestSession)
    session.mpi_pool = object()
    session.abort = Mock()
    monkeypatch.setattr(mpi.MPINodeState, "_global_mpi_pool", session.mpi_pool)
    now = [100.0]
    monkeypatch.setattr(mpi.time, "monotonic", lambda: now[0])

    def shutdown():
        now[0] += shutdown_seconds
        session.mpi_pool = None

    session.shutdown = shutdown
    server = object.__new__(mpi.RemoteMpiCommSessionServer)
    server.session = session
    server._close_global_comm_executor = Mock()
    server._shutdown_session(10)
    if shutdown_seconds == 7:
        server._close_global_comm_executor.assert_called_once_with(grace=3, abort=session.abort)
        session.abort.assert_not_called()
    else:
        server._close_global_comm_executor.assert_not_called()
        session.abort.assert_called_once_with()


def test_shutdown_timeout_aborts_the_world():
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def shutdown():
        entered.set()
        release.wait(5)
        finished.set()

    server = object.__new__(mpi.RemoteMpiCommSessionServer)
    server.session = SimpleNamespace(shutdown=shutdown, abort=Mock())
    try:
        server._shutdown_session(0.02)
        assert entered.is_set()
        server.session.abort.assert_called_once_with()
    finally:
        release.set()
        assert finished.wait(1)


@pytest.mark.parametrize("raw", ["", "0", "-1", "nan", "inf", "bad"])
def test_invalid_shutdown_grace_uses_default(monkeypatch, raw):
    monkeypatch.setenv("TLLM_MGMN_SHUTDOWN_GRACE_SECONDS", raw)
    assert mpi._mgmn_shutdown_grace_seconds() == 60


def test_shutdown_grace_accepts_positive_finite_value(monkeypatch):
    monkeypatch.setenv("TLLM_MGMN_SHUTDOWN_GRACE_SECONDS", "0.5")
    assert mpi._mgmn_shutdown_grace_seconds() == 0.5
