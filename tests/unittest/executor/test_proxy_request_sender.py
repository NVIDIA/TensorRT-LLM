# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for request batching between the proxy and the rank-0 worker.

``TRTLLM_PROXY_BATCH_REQUESTS=1`` makes the proxy pickle each request on the
submitting thread and send bursts of payloads as one list; the worker unpickles
each payload.
"""

import pickle
import queue
import threading
import time
from types import SimpleNamespace

import pytest
import zmq

from tensorrt_llm.executor.ipc import IpcQueue, RequestSendCancelled
from tensorrt_llm.executor.proxy import (
    GenerationExecutorProxy,
    _create_request_sender,
    _RequestSender,
)
from tensorrt_llm.executor.worker import _incoming_requests

pytestmark = pytest.mark.cpu_only


class _RecordingQueue:
    """Request-queue stand-in that records each message; the first put can block.

    Like ``ZeroMqQueue`` with ``TLLM_LLMAPI_ZMQ_DEBUG=1``, it rejects a write from
    any thread other than the first one that wrote.
    """

    def __init__(self, block_first_put: bool = False):
        self.messages = []
        self.writer = None
        self.first_put_started = threading.Event()
        self.release_first_put = threading.Event()
        if not block_first_put:
            self.release_first_put.set()

    def _check_writer(self):
        if self.writer is None:
            self.writer = threading.get_ident()
        elif self.writer != threading.get_ident():
            raise RuntimeError("request queue written from a second thread")

    def put(self, msg):
        self._check_writer()
        if not self.messages:
            self.first_put_started.set()
            self.release_first_put.wait(timeout=10)
        self.messages.append(msg)

    def put_interruptible(self, msg, should_stop):
        if should_stop():
            raise RequestSendCancelled("request send cancelled")
        self.put(msg)

    def put_nowait(self, msg):
        self._check_writer()
        self.messages.append(msg)


def _payloads(*items):
    return [pickle.dumps(item) for item in items]


def _bare_proxy():
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy._init_lifecycle()
    proxy._engine_dead = False
    # These tests close their transports explicitly, without running a full
    # executor lifecycle on the request-only stand-ins.
    proxy._shutdown_complete = True
    proxy.workers_started = False
    return proxy


def _pre_shutdown(proxy):
    """Run ``pre_shutdown`` on a bare proxy with live workers, on a new thread."""
    proxy.workers_started = True
    proxy.doing_shutdown = False
    proxy._worker_process_monitor = SimpleNamespace(close=lambda: None)
    proxy._results = {}
    proxy.mpi_futures = []
    shutdown = threading.Thread(target=proxy.pre_shutdown)
    shutdown.start()
    return shutdown


def test_batching_is_opt_in(monkeypatch):
    monkeypatch.delenv("TRTLLM_PROXY_BATCH_REQUESTS", raising=False)
    assert _create_request_sender(_RecordingQueue(), queue.Queue()) is None
    monkeypatch.setenv("TRTLLM_PROXY_BATCH_REQUESTS", "1")
    assert isinstance(_create_request_sender(_RecordingQueue(), queue.Queue()), _RequestSender)


def test_lone_request_is_sent_as_one_payload_list():
    request_queue = _RecordingQueue()
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send(pickle.dumps("a"))
    sender.close(timeout=10)
    assert request_queue.messages == [_payloads("a")]


def test_burst_is_sent_as_one_list_in_order():
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send(pickle.dumps(0))
    assert request_queue.first_put_started.wait(timeout=10)
    # Everything queued while the first send is in flight goes out as one list.
    for item in range(1, 6):
        sender.send(pickle.dumps(item))
    request_queue.release_first_put.set()
    sender.close(timeout=10)
    assert request_queue.messages == [_payloads(0), _payloads(1, 2, 3, 4, 5)]


def test_close_flushes_queued_requests_and_rejects_later_ones():
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send(pickle.dumps("first"))
    assert request_queue.first_put_started.wait(timeout=10)
    sender.send(pickle.dumps("queued"))
    closer = threading.Thread(target=sender.close, kwargs={"timeout": 10})
    closer.start()
    request_queue.release_first_put.set()
    closer.join(timeout=10)
    assert not closer.is_alive()
    assert request_queue.messages == [_payloads("first"), _payloads("queued")]
    with pytest.raises(RuntimeError, match="closed"):
        sender.send(pickle.dumps("late"))
    assert request_queue.messages == [_payloads("first"), _payloads("queued")]


def test_close_is_idempotent():
    request_queue = _RecordingQueue()
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send(pickle.dumps("a"))
    sender.close(timeout=10)
    assert sender._thread is None
    sender.close(timeout=10)
    assert request_queue.messages == [_payloads("a")]


def test_sender_thread_sends_the_sentinel_after_the_queued_requests():
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send(pickle.dumps("first"))
    assert request_queue.first_put_started.wait(timeout=10)
    sender.send(pickle.dumps("queued"))
    result = []
    closer = threading.Thread(
        target=lambda: result.append(sender.close(timeout=10, send_sentinel=True))
    )
    closer.start()
    request_queue.release_first_put.set()
    closer.join(timeout=10)
    assert result == [False]
    # A bare None after the payload lists, written by the sender thread.
    assert request_queue.messages == [_payloads("first"), _payloads("queued"), None]


def test_caller_sends_the_sentinel_when_the_sender_thread_did_not():
    # The thread never started, so nothing has written the queue yet.
    assert _RequestSender(_RecordingQueue(), queue.Queue()).close(send_sentinel=True)
    # The thread stopped before the sentinel was asked for.
    sender = _RequestSender(_RecordingQueue(), queue.Queue())
    sender.send(pickle.dumps("a"))
    sender.close(timeout=10)
    assert sender.close(timeout=10, send_sentinel=True)


def test_close_during_interpreter_finalization_does_not_join():
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send(pickle.dumps("first"))
    assert request_queue.first_put_started.wait(timeout=10)
    sender.send(pickle.dumps("queued"))
    # Joining would wait for the blocked send; at teardown close() must return at
    # once, and must not ask anyone to send the sentinel.
    sender._is_finalizing = lambda: True
    assert sender.close(send_sentinel=True) is False
    assert sender._thread is not None
    # Outside finalization the same sender still flushes and stops.
    sender._is_finalizing = lambda: False
    request_queue.release_first_put.set()
    sender.close(timeout=10)
    assert sender._thread is None
    assert request_queue.messages == [_payloads("first"), _payloads("queued")]


def test_proxy_finalizer_does_not_start_shutdown_during_late_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = _bare_proxy()
    calls = []
    monkeypatch.setattr(proxy, "shutdown", lambda: calls.append("shutdown"))
    with monkeypatch.context() as patch:
        patch.setattr(proxy, "_is_finalizing", lambda: True, raising=False)
        proxy.__del__()
    assert not calls
    proxy.__del__()
    assert calls == ["shutdown"]


def test_pre_shutdown_flushes_batched_requests_before_the_sentinel():
    request_queue = _RecordingQueue(block_first_put=True)
    proxy = _bare_proxy()
    proxy.request_queue = request_queue
    proxy._request_sender = _RequestSender(request_queue, queue.Queue())
    proxy._send_request("first")
    assert request_queue.first_put_started.wait(timeout=10)
    proxy._send_request("queued")

    shutdown = _pre_shutdown(proxy)
    request_queue.release_first_put.set()
    shutdown.join(timeout=10)
    assert not shutdown.is_alive()
    proxy.workers_started = False
    # The sender thread wrote everything, the sentinel included.
    assert request_queue.messages == [_payloads("first"), _payloads("queued"), None]
    assert request_queue.writer != shutdown.ident


@pytest.mark.parametrize("batching", [True, False])
def test_pre_shutdown_sends_the_sentinel_itself_if_nothing_else_wrote(batching):
    request_queue = _RecordingQueue()
    proxy = _bare_proxy()
    proxy.request_queue = request_queue
    if batching:
        proxy._request_sender = _RequestSender(request_queue, queue.Queue())

    shutdown = _pre_shutdown(proxy)
    shutdown.join(timeout=10)
    assert not shutdown.is_alive()
    proxy.workers_started = False
    assert request_queue.messages == [None]
    assert request_queue.writer == shutdown.ident


@pytest.mark.parametrize("error_type", [RuntimeError, zmq.ZMQError, AssertionError])
def test_transport_error_reaches_error_queue(error_type) -> None:
    stopped = threading.Event()
    failure = error_type("socket or debug failure")

    class _FailingQueue:
        def put_interruptible(self, msg, should_stop) -> None:
            stopped.set()
            raise failure

    proxy = _bare_proxy()
    error_queue = queue.Queue()
    sender = _RequestSender(_FailingQueue(), error_queue, stopped.is_set)
    proxy._request_sender = sender
    try:
        sender.send(pickle.dumps("a"))
        sender._thread.join(timeout=2.0)
        assert not sender._thread.is_alive()
        assert error_queue.get_nowait() is failure
        # A sender that stopped itself is not an orderly frontend shutdown.
        assert not proxy.doing_shutdown
        with pytest.raises(RuntimeError, match="sender is closed"):
            proxy.abort_request(1)
        assert error_queue.empty()
    finally:
        sender.close(timeout=2.0)


@pytest.mark.parametrize("stop_during_quit", [False, True])
def test_batched_quit_failure_is_best_effort(
    monkeypatch: pytest.MonkeyPatch, stop_during_quit: bool
) -> None:
    from tensorrt_llm.executor import proxy as proxy_module

    warnings = []
    stopped = threading.Event()
    monkeypatch.setattr(proxy_module.logger, "warning", warnings.append)

    class GoneAfterBatch(_RecordingQueue):
        def put(self, msg: object) -> None:
            if msg is None:
                if stop_during_quit:
                    stopped.set()
                raise RuntimeError("peer closed")
            super().put(msg)

    request_queue = GoneAfterBatch()
    errors = queue.Queue()
    sender = _RequestSender(request_queue, errors, stopped.is_set)
    sender.send(pickle.dumps("request"))
    assert not sender.close(timeout=2, send_sentinel=True)
    assert request_queue.messages == [_payloads("request")]
    assert errors.empty()
    assert len(warnings) == 1 and "peer closed" in warnings[0]


def test_batched_request_is_captured_when_sent():
    proxy = _bare_proxy()
    sent = []
    proxy._request_sender = SimpleNamespace(send=sent.append)

    request = {"max_tokens": 1}
    proxy._send_request(request)
    # A change after submission does not reach the worker.
    request["max_tokens"] = 2
    assert pickle.loads(sent[0]) == {"max_tokens": 1}

    # A request that cannot be pickled fails in its caller, and nothing is queued.
    with pytest.raises(TypeError):
        proxy._send_request({"lock": threading.Lock()})
    assert len(sent) == 1


def test_unbatched_request_is_put_directly():
    proxy = _bare_proxy()
    request_queue = _RecordingQueue()
    proxy.request_queue = request_queue
    proxy._send_request("a")
    assert request_queue.messages == ["a"]


def test_worker_unpacks_batched_requests_in_order():
    class _ListQueue:
        def __init__(self, messages):
            self._messages = list(messages)

        def get(self):
            return self._messages.pop(0)

    request_queue = _ListQueue(["a", _payloads("b", "c"), "d", None, "after sentinel"])
    assert list(_incoming_requests(request_queue)) == ["a", "b", "c", "d"]


def test_sender_preserves_fifo_and_bare_quit_with_private_peer() -> None:
    request_queue = IpcQueue(is_server=True)
    peer = IpcQueue(request_queue.address, is_server=False)
    sender = _RequestSender(request_queue, queue.Queue())
    try:
        peer.setup_lazily()
        sender.send(pickle.dumps("generate"))
        sender.send(pickle.dumps("cancel"))
        thread = sender._thread
        sender.close(timeout=2.0, send_sentinel=True)
        assert not thread.is_alive()
        messages = []
        while (message := peer.get(timeout=2.0)) is not None:
            messages.extend(pickle.loads(payload) for payload in message)
        assert messages == ["generate", "cancel"]
    finally:
        sender.close(timeout=2.0)
        peer.close()
        request_queue.close()


@pytest.mark.parametrize("batching", [False, True])
@pytest.mark.parametrize("engine_dead", [False, True])
def test_absent_peer_send_is_interrupted_before_transport_close(
    monkeypatch: pytest.MonkeyPatch, batching: bool, engine_dead: bool
) -> None:
    proxy = _bare_proxy()
    proxy.request_queue = IpcQueue(is_server=True)
    proxy._REQUEST_SHUTDOWN_TIMEOUT_S = 0.05
    errors = queue.Queue()
    entered = threading.Event()
    original_send = proxy.request_queue._send_data

    def record_send(*args, **kwargs):
        entered.set()
        return original_send(*args, **kwargs)

    monkeypatch.setattr(proxy.request_queue, "_send_data", record_send)
    proxy._request_sender = (
        _RequestSender(proxy.request_queue, errors, proxy._request_send_stopped)
        if batching
        else None
    )
    outcome = []

    def send():
        try:
            proxy._send_request("generate")
        except RuntimeError as exc:
            outcome.append(exc)

    caller = threading.Thread(target=send)
    try:
        caller.start()
        assert entered.wait(timeout=2.0)
        sender_thread = proxy._request_sender._thread if batching else None
        proxy._engine_dead = engine_dead
        started = time.monotonic()
        proxy._close_request_sender(send_sentinel=True)
        assert time.monotonic() - started < 1.0
        if sender_thread is not None:
            assert not sender_thread.is_alive()
        # Acquiring the send lock in close proves a direct writer has stopped
        # touching the socket; the caller can still be returning its exception.
        assert proxy._request_transport_closed
        caller.join(timeout=2.0)
        assert not caller.is_alive()
        assert errors.empty() if batching else outcome
    finally:
        proxy._engine_dead = True
        proxy._close_request_sender()
        if caller.ident is not None:
            caller.join(timeout=2.0)
        assert not caller.is_alive()
        proxy.request_queue.close()


def test_close_reports_a_noncooperative_sender_instead_of_success() -> None:
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send(pickle.dumps("generate"))
    assert request_queue.first_put_started.wait(timeout=2.0)
    try:
        with pytest.raises(TimeoutError, match="still closing"):
            sender.close(timeout=0.0)
        assert sender._thread.is_alive()
    finally:
        request_queue.release_first_put.set()
        sender.close(timeout=2.0)


@pytest.mark.parametrize("batching", [False, True])
def test_generation_send_waiting_at_admission_is_rejected_after_close(
    monkeypatch: pytest.MonkeyPatch, batching: bool
) -> None:
    proxy = _bare_proxy()
    proxy.request_queue = _RecordingQueue()
    proxy._request_sender = _RequestSender(proxy.request_queue, queue.Queue()) if batching else None
    entered = threading.Event()
    release_pickle = threading.Event()
    errors = []
    original_pickle = pickle.dumps

    def blocked_pickle(request):
        entered.set()
        assert release_pickle.wait(timeout=2.0)
        return original_pickle(request)

    if batching:
        monkeypatch.setattr(pickle, "dumps", blocked_pickle)

    def submit():
        if not batching:
            entered.set()
        try:
            proxy._send_generation_request("generate")
        except RuntimeError as exc:
            errors.append(exc)

    with proxy._request_send_lock:
        caller = threading.Thread(target=submit)
        caller.start()
        assert entered.wait(timeout=2.0)
        with proxy._admission_lock:
            proxy.doing_shutdown = True
        release_pickle.set()
    caller.join(timeout=2.0)
    assert not caller.is_alive()
    assert len(errors) == 1
    proxy._close_request_sender()
    assert proxy.request_queue.messages == []


def test_abort_racing_orderly_sender_close_is_harmless(monkeypatch) -> None:
    proxy = _bare_proxy()
    proxy.doing_shutdown = True
    proxy.request_queue = _RecordingQueue()
    errors = queue.Queue()
    proxy._request_sender = _RequestSender(proxy.request_queue, errors)
    sender_closed = threading.Event()
    original_close = proxy._request_sender.close

    def close(*args, **kwargs) -> bool:
        result = original_close(*args, **kwargs)
        sender_closed.set()
        return result

    monkeypatch.setattr(proxy._request_sender, "close", close)
    closer = threading.Thread(target=proxy._close_request_sender)
    try:
        # Hold the transport open after the sender stops accepting requests.
        with proxy._request_send_lock:
            closer.start()
            assert sender_closed.wait(timeout=2.0)
            assert not proxy._request_transport_closed
            proxy.abort_request(1)
        closer.join(timeout=2.0)
        assert not closer.is_alive()
        assert proxy._request_transport_closed
        assert not proxy.request_queue.messages
        assert errors.empty()
    finally:
        closer.join(timeout=2.0)
        proxy._close_request_sender()
        assert not closer.is_alive()


@pytest.mark.parametrize("failure", [None, RuntimeError("debug failure")])
def test_abort_racing_send_deadline_only_ignores_cancellation(monkeypatch, failure) -> None:
    proxy = _bare_proxy()
    proxy.request_queue = IpcQueue(is_server=True)
    original_send = proxy._send_request

    def send(request) -> None:
        proxy._request_shutdown_deadline = time.monotonic() - 1.0
        if failure is not None:
            raise failure
        original_send(request)

    monkeypatch.setattr(proxy, "_send_request", send)
    proxy._results = {1: SimpleNamespace(abort=lambda: proxy.abort_request(1))}
    try:
        if failure is None:
            proxy._abort_all_requests()
        else:
            with pytest.raises(RuntimeError, match="debug failure"):
                proxy._abort_all_requests()
        assert not proxy.request_queue._setup_done
    finally:
        proxy.request_queue.close()


@pytest.mark.parametrize("batching", [False, True])
@pytest.mark.parametrize("failure_kind", ["transport", "thread_safety"])
def test_send_failure_is_not_hidden_when_budget_expires(
    monkeypatch, batching, failure_kind
) -> None:
    proxy = _bare_proxy()
    proxy.request_queue = IpcQueue(is_server=True)
    errors = queue.Queue()
    if failure_kind == "thread_safety":
        proxy.request_queue._zmq_debug_enabled = True
        proxy.request_queue._check_thread_safety()
        original_check = proxy.request_queue._check_thread_safety

        def check() -> None:
            proxy._request_shutdown_deadline = time.monotonic() - 1.0
            original_check()

        monkeypatch.setattr(proxy.request_queue, "_check_thread_safety", check)
    else:
        original_send = proxy.request_queue._send_data

        def send(*args, **kwargs) -> None:
            # The socket's writer provokes an actual ZeroMQ ENOTSOCK error.
            proxy.request_queue.socket.close(linger=0)
            proxy._request_shutdown_deadline = time.monotonic() - 1.0
            try:
                original_send(*args, **kwargs)
            finally:
                proxy.request_queue.socket = None

        monkeypatch.setattr(proxy.request_queue, "_send_data", send)
    if batching:
        proxy._request_sender = _RequestSender(
            proxy.request_queue, errors, proxy._request_send_stopped
        )

    def abort() -> None:
        try:
            proxy.abort_request(1)
        except (RuntimeError, zmq.ZMQError) as exc:
            errors.put_nowait(exc)

    caller = threading.Thread(target=abort)
    try:
        caller.start()
        caller.join(timeout=2.0)
        assert not caller.is_alive()
        proxy._close_request_sender()
        error = errors.get_nowait()
        if failure_kind == "thread_safety":
            assert isinstance(error, RuntimeError)
            assert "thread safety violation" in str(error)
        else:
            assert isinstance(error, zmq.ZMQError)
            assert error.errno == zmq.ENOTSOCK
        assert errors.empty()
    finally:
        caller.join(timeout=2.0)
        proxy._close_request_sender()
        proxy.request_queue.close()
        assert not caller.is_alive()


def test_dead_engine_abort_does_not_attempt_transport() -> None:
    proxy = _bare_proxy()
    proxy._engine_dead = True
    proxy.request_queue = _RecordingQueue(block_first_put=True)
    proxy.abort_request(1)
    assert not proxy.request_queue.first_put_started.is_set()
