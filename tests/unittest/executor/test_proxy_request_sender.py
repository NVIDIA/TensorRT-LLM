# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for request batching between the proxy and the rank-0 worker.

``TRTLLM_PROXY_BATCH_REQUESTS=1`` makes the proxy pickle each request on the
submitting thread and send bursts of payloads as one list; the worker unpickles
each payload.
"""

import asyncio
import pickle
import queue
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import zmq

from tensorrt_llm.executor.ipc import IpcQueue
from tensorrt_llm.executor.proxy import (
    GenerationExecutorProxy,
    _create_request_sender,
    _RequestSender,
)
from tensorrt_llm.executor.request import CancellingRequest
from tensorrt_llm.executor.utils import EngineDeadError, EngineUnavailableError
from tensorrt_llm.executor.worker import _incoming_requests

pytestmark = pytest.mark.cpu_only


class _RecordingQueue:
    """Request-queue stand-in that records each message; the first put can block.

    Like ``ZeroMqQueue`` with ``TLLM_LLMAPI_ZMQ_DEBUG=1``, it rejects a write from
    any thread other than the first one that wrote, so the production factory
    keeps every write on the sender thread.
    """

    _zmq_debug_enabled = True

    def __init__(self, block_first_put: bool = False):
        self.messages = []
        self.socket = SimpleNamespace(
            setsockopt=lambda *args: self._check_writer(),
            get_monitor_socket=lambda **kwargs: SimpleNamespace(
                poll=lambda timeout: False, close=lambda **kwargs: self._check_writer()
            ),
            disable_monitor=lambda: self._check_writer(),
        )
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

    def setup_lazily(self):
        self._check_writer()

    def put(self, msg, *, timeout=None):
        self._check_writer()
        if not self.messages:
            self.first_put_started.set()
            self.release_first_put.wait(timeout=10)
        self.messages.append(msg)

    def put_noblock(self, msg, *, retry=1, wait_time=0.001):
        self._check_writer()
        self.messages.append(msg)

    def close(self):
        self._check_writer()


def _payloads(*items):
    return [pickle.dumps(item) for item in items]


def _bare_proxy():
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    # Keeps the __del__ -> shutdown() path a no-op at GC time.
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


@pytest.mark.parametrize("setting", [None, "0", "1"])
def test_batching_is_opt_in(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("TRTLLM_PROXY_BATCH_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_PROXY_BATCH_REQUESTS", setting)
    batching = setting == "1"
    q = _RecordingQueue(block_first_put=True)
    sender = _create_request_sender(q, queue.Queue())
    sender.send(pickle.dumps("a"))
    assert not q.messages, "submission waited for its command to be written"
    assert q.first_put_started.wait(10)
    sender.send(pickle.dumps("b"))
    sender.send(pickle.dumps("c"))
    q.release_first_put.set()
    sender.close(timeout=10)
    expected = (
        [_payloads("a"), _payloads("b", "c")]
        if batching
        else [_payloads("a"), _payloads("b"), _payloads("c")]
    )
    assert q.messages == expected


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


def test_close_flushes_queued_requests_and_drops_later_ones():
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
    with pytest.raises(EngineUnavailableError):
        sender.send(pickle.dumps("late"))
    assert request_queue.messages == [_payloads("first"), _payloads("queued")]


def test_close_is_idempotent():
    request_queue = _RecordingQueue()
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send(pickle.dumps("a"))
    sender.close(timeout=10)
    assert not sender._thread.is_alive()
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
    assert result == [None]
    # A bare None after the payload lists, written by the sender thread.
    assert request_queue.messages == [_payloads("first"), _payloads("queued"), None]


def test_shutdown_without_requests_still_uses_the_socket_owner():
    q = _RecordingQueue()
    sender = _RequestSender(q, queue.Queue())
    assert sender.close(timeout=10, send_sentinel=True) is None
    assert q.messages == [None]
    assert q.writer != threading.get_ident()


def test_close_during_interpreter_finalization_does_not_join():
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send(pickle.dumps("first"))
    assert request_queue.first_put_started.wait(timeout=10)
    sender.send(pickle.dumps("queued"))
    # Joining would wait for the blocked send; at teardown close() must return at
    # once, and must not ask anyone to send the sentinel.
    sender._is_finalizing = lambda: True
    assert sender.close(send_sentinel=True) is None
    assert sender._thread is not None
    # Outside finalization the same sender still flushes and stops.
    sender._is_finalizing = lambda: False
    request_queue.release_first_put.set()
    sender.close(timeout=10)
    assert not sender._thread.is_alive()
    assert request_queue.messages == [_payloads("first"), _payloads("queued")]


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
    proxy._request_sender = _RequestSender(request_queue, queue.Queue(), batching=batching)

    shutdown = _pre_shutdown(proxy)
    shutdown.join(timeout=10)
    assert not shutdown.is_alive()
    proxy.workers_started = False
    assert request_queue.messages == [None]
    assert request_queue.writer != shutdown.ident


@pytest.mark.timeout(20, method="thread")
@pytest.mark.parametrize("late_close", [False, True])
def test_transport_error_reaches_error_queue(monkeypatch, late_close):
    monkeypatch.setenv("TRTLLM_PROXY_BATCH_REQUESTS", "1")
    published = threading.Event()
    finish_failure = threading.Event()

    class _ErrorQueue(queue.Queue):
        def put_nowait(self, error):
            super().put_nowait(error)
            published.set()
            if not late_close:
                assert finish_failure.wait(timeout=10)

    class _FailingQueue(_RecordingQueue):
        attempts = 0

        def put(self, msg, *, timeout=None):
            if msg is None:
                return super().put(msg, timeout=timeout)
            self._check_writer()
            self.attempts += 1
            self.first_put_started.set()
            assert self.release_first_put.wait(timeout=10)
            raise RuntimeError("socket closed")

    request_queue = _FailingQueue(block_first_put=True)
    error_queue = _ErrorQueue()
    sender = _create_request_sender(request_queue, error_queue)
    closer = None
    try:
        sender.send(pickle.dumps("a"))
        assert request_queue.first_put_started.wait(timeout=10)
        sender.send(pickle.dumps("queued"))
        request_queue.release_first_put.set()
        assert published.wait(timeout=10)
        if late_close:
            # The failure owner must already know how to stop its workers;
            # cleanup can arrive after that owner has closed its socket.
            sender._thread.join(timeout=3)
            assert not sender._thread.is_alive()
        responsibility = []
        closer = threading.Thread(
            target=lambda: responsibility.append(sender.close(timeout=10, send_sentinel=True))
        )
        closer.start()
        if not late_close:
            with sender._cond:
                assert sender._cond.wait_for(lambda: sender._send_sentinel, timeout=10)
        finish_failure.set()
        closer.join(timeout=10)
        assert not closer.is_alive()
        assert responsibility == [None]
        assert request_queue.attempts == 1
        assert request_queue.messages == [None]
        error = error_queue.get_nowait()
        assert isinstance(error, RuntimeError)
        assert str(error) == "socket closed"
    finally:
        request_queue.release_first_put.set()
        finish_failure.set()
        if closer is not None:
            closer.join(timeout=10)
        sender.close(timeout=10)


def test_batched_request_is_captured_when_sent():
    proxy = _bare_proxy()
    sent = []
    proxy._request_sender = SimpleNamespace(
        send=lambda payload, **kwargs: sent.append(payload), close=lambda **kwargs: None
    )

    request = {"max_tokens": 1}
    proxy._send_request(request)
    # A change after submission does not reach the worker.
    request["max_tokens"] = 2
    assert pickle.loads(sent[0]) == {"max_tokens": 1}

    # A request that cannot be pickled fails in its caller, and nothing is queued.
    with pytest.raises(TypeError):
        proxy._send_request({"lock": threading.Lock()})
    assert len(sent) == 1


def test_worker_unpacks_batched_requests_in_order():
    class _ListQueue:
        def __init__(self, messages):
            self._messages = list(messages)

        def get(self):
            return self._messages.pop(0)

    request_queue = _ListQueue(["a", _payloads("b", "c"), "d", None, "after sentinel"])
    assert list(_incoming_requests(request_queue)) == ["a", "b", "c", "d"]


@pytest.mark.timeout(10, method="thread")
@pytest.mark.parametrize("push", [False, True])
def test_initial_connection_timeout_fails_accepted_work(tmp_path, push):
    if push:
        commands = IpcQueue(
            (f"ipc://{tmp_path}/absent", b"k" * 32),
            is_server=False,
            socket_type=zmq.PUSH,
        )
    else:
        commands = IpcQueue(is_server=True)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, connect_timeout=0.15)
    try:
        sender.send(pickle.dumps("accepted"))
        assert isinstance(errors.get(timeout=2), TimeoutError)
        with pytest.raises(EngineDeadError):
            sender.send(pickle.dumps("late"))
    finally:
        sender.close(timeout=2)
    assert not sender._thread.is_alive()


@pytest.mark.timeout(10, method="thread")
@pytest.mark.parametrize("transport", ["tcp", "ipc"])
def test_established_disconnect_fails_while_idle_without_replay(tmp_path, transport):
    address = (f"ipc://{tmp_path}/commands", b"k" * 32) if transport == "ipc" else None
    peer = IpcQueue(address, is_server=True, socket_type=zmq.PULL)
    address = peer.address
    commands = IpcQueue(address, is_server=False, socket_type=zmq.PUSH)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False)
    replacement = None
    try:
        sender.send(pickle.dumps("accepted"))
        assert peer.get(timeout=2) == _payloads("accepted")
        # No subsequent send or Again may be needed to discover missing tail data.
        peer.close()
        assert isinstance(errors.get(timeout=2), ConnectionError)
        sender._thread.join(timeout=2)
        assert not sender._thread.is_alive()
        replacement = IpcQueue(address, is_server=True, socket_type=zmq.PULL)
        with pytest.raises(EngineDeadError):
            sender.send(pickle.dumps("after-disconnect"))
        assert not replacement.poll(0.1)
    finally:
        sender.close(timeout=2)
        peer.close()
        if replacement is not None:
            replacement.close()


@pytest.mark.timeout(12, method="thread")
def test_control_capacity_is_reserved_when_admission_is_full():
    q = _RecordingQueue(block_first_put=True)
    sender = _RequestSender(q, queue.Queue())
    sender.MAX_BATCH_SIZE = 1
    sender.MAX_PENDING_REQUESTS = 1
    sender.MAX_PENDING_CONTROLS = 1
    try:
        sender.send(pickle.dumps("in-flight"))
        assert q.first_put_started.wait(3)
        sender.send(pickle.dumps("queued"))
        with pytest.raises(EngineUnavailableError):
            sender.send(pickle.dumps("overload"))
        sender.send(pickle.dumps("cancel"), control=True)
        with pytest.raises(EngineUnavailableError):
            sender.send(pickle.dumps("extra-cancel"), control=True)
    finally:
        q.release_first_put.set()
        sender.close(timeout=3, send_sentinel=True)
    assert q.messages == [_payloads("in-flight"), _payloads("queued"), _payloads("cancel"), None]


@pytest.mark.timeout(12, method="thread")
@pytest.mark.parametrize("local_final", [False, True])
def test_proxy_retains_cancellations_when_control_queue_is_full(local_final):
    q = _RecordingQueue(block_first_put=True)
    proxy = _bare_proxy()
    proxy._engine_dead = False
    proxy._fatal_error = None
    proxy.doing_shutdown = False
    proxy._error_queue = queue.Queue()
    unrelated = SimpleNamespace(queue=queue.Queue())
    proxy._results = {
        42: SimpleNamespace(queue=queue.Queue()),
        43: SimpleNamespace(queue=queue.Queue()),
        44: unrelated,
    }
    sender = _RequestSender(q, proxy._error_queue)
    sender.MAX_PENDING_CONTROLS = 1
    sender.MAX_BATCH_SIZE = 1
    proxy._request_sender = sender
    try:
        sender.send(pickle.dumps("in-flight"))
        assert q.first_put_started.wait(3)
        sender.send(pickle.dumps("queued-control"), control=True)
        if local_final:
            from tensorrt_llm.executor.postproc_worker import PostprocWorker

            response = PostprocWorker.Output(42, None, True, should_abort=True)
            proxy.result_queue = SimpleNamespace(get=lambda: response)
            assert proxy.dispatch_result_task()
            assert 42 not in proxy._results
        else:
            proxy.abort_request(42)
        proxy.abort_request(42)
        proxy.abort_request(43)
        assert not proxy._engine_dead
        assert unrelated.queue.empty()
        assert proxy._error_queue.empty()
    finally:
        q.release_first_put.set()
        sender.close(timeout=3)
    delivered = [pickle.loads(message[0]) for message in q.messages[2:]]
    assert [request.id for request in delivered] == [42, 43]
    assert all(isinstance(request, CancellingRequest) for request in delivered)


@pytest.mark.timeout(12, method="thread")
def test_shutdown_cancels_survive_full_control_queue():
    q = _RecordingQueue(block_first_put=True)
    sender = _RequestSender(q, queue.Queue())
    sender.MAX_PENDING_CONTROLS = 1
    sender.MAX_BATCH_SIZE = 1
    sender.send(pickle.dumps("in-flight"))
    assert q.first_put_started.wait(3)
    sender.send(pickle.dumps("queued-control"), control=True)
    closed = threading.Thread(target=lambda: sender.close(timeout=3, cancel_ids=(7, 8)))
    closed.start()
    q.release_first_put.set()
    closed.join(timeout=4)
    assert not closed.is_alive()
    assert q.messages[:2] == [_payloads("in-flight"), _payloads("queued-control")]
    cancels = [pickle.loads(message[0]) for message in q.messages[2:]]
    assert all(isinstance(request, CancellingRequest) for request in cancels)
    assert [request.id for request in cancels] == [7, 8]
    assert None not in q.messages


@pytest.mark.timeout(20, method="thread")
@pytest.mark.parametrize("direct_send", [False, True])
def test_connected_backpressure_preserves_queued_requests_beyond_old_timeout(direct_send):
    blocked = threading.Event()

    class ObservedQueue(IpcQueue):
        def put(self, message, **kwargs):
            try:
                return super().put(message, **kwargs)
            except zmq.Again:
                blocked.set()
                raise

    peer = IpcQueue(is_server=True, socket_type=zmq.PULL, receive_hwm=1)
    peer.socket.setsockopt(zmq.RCVBUF, 256 * 1024)
    commands = ObservedQueue(peer.address, is_server=False, socket_type=zmq.PUSH, send_hwm=1)
    commands.socket.setsockopt(zmq.SNDBUF, 256 * 1024)
    errors = queue.Queue()
    sender = _RequestSender(
        commands, errors, batching=False, connect_timeout=0.15, direct_send=direct_send
    )
    count = 16
    data = b"x" * (1024 * 1024)
    try:
        for i in range(count):
            sender.send(pickle.dumps((i, data)))
        assert blocked.wait(3), "real transport did not reach backpressure"
        # A connected worker can legitimately stop consuming for longer than
        # the old five-second recovery deadline. It must retain accepted work.
        time.sleep(5.2)
        assert sender.available
        assert errors.empty()
        sender.send(pickle.dumps((count, data)))
        received = []
        for _ in range(count + 1):
            message = peer.get(timeout=3)
            assert len(message) == 1
            request_id, contents = pickle.loads(message[0])
            assert contents == data
            received.append(request_id)
        assert received == list(range(count + 1))
        assert errors.empty()
    finally:
        sender.close(timeout=2)
        peer.socket.setsockopt(zmq.LINGER, 0)
        peer.close()
    assert not sender._thread.is_alive()


@pytest.mark.timeout(10, method="thread")
@pytest.mark.parametrize("stop", ["capacity", "close", "failure"])
def test_capacity_wait_unblocks_on_progress_or_shutdown(stop):
    commands = _RecordingQueue(block_first_put=True)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False)
    sender.MAX_PENDING_REQUESTS = 1
    outcomes = queue.Queue()
    entered = threading.Event()

    def wait_for_capacity():
        entered.set()
        try:
            sender.send(pickle.dumps("waiting"), wait_for_capacity=True)
            outcomes.put(None)
        except (EngineDeadError, EngineUnavailableError) as error:
            outcomes.put(error)

    waiter = threading.Thread(target=wait_for_capacity)
    closer = None
    try:
        sender.send(pickle.dumps("in-flight"))
        assert commands.first_put_started.wait(2)
        sender.send(pickle.dumps("queued"))
        waiter.start()
        assert entered.wait(2)
        assert outcomes.empty()
        if stop == "failure":
            sender.fail(RuntimeError("worker died"))
        elif stop == "close":
            closer = threading.Thread(target=sender.close, kwargs={"timeout": 2})
            closer.start()
        else:
            commands.release_first_put.set()
        outcome = outcomes.get(timeout=2)
        if stop == "capacity":
            assert outcome is None
        elif stop == "failure":
            assert isinstance(outcome, EngineDeadError)
        else:
            assert isinstance(outcome, EngineUnavailableError)
    finally:
        commands.release_first_put.set()
        waiter.join(timeout=2)
        if closer is not None:
            closer.join(timeout=2)
        sender.close(timeout=2)
    assert not waiter.is_alive()
    if stop == "capacity":
        assert commands.messages == [
            _payloads("in-flight"),
            _payloads("queued"),
            _payloads("waiting"),
        ]


@pytest.mark.parametrize(
    ("batching", "debug", "direct"),
    [("0", False, True), (None, False, True), ("1", False, False), ("0", True, False)],
)
def test_unbatched_production_sender_writes_on_the_submitting_thread(
    monkeypatch, batching, debug, direct
):
    if batching is None:
        monkeypatch.delenv("TRTLLM_PROXY_BATCH_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_PROXY_BATCH_REQUESTS", batching)
    commands = IpcQueue(is_server=True)
    commands._zmq_debug_enabled = debug
    sender = _create_request_sender(commands, queue.Queue())
    try:
        assert sender._direct_send is direct
    finally:
        sender.close(timeout=2)
        commands.close()


class _OwnerStalledQueue(IpcQueue):
    """Real command socket whose sender thread cannot write until released.

    This stands in for a sender thread that CPython does not schedule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release = threading.Event()
        self.writers = set()

    def put(self, message, **kwargs):
        if threading.current_thread().name == "proxy_request_sender":
            assert self.release.wait(10)
        self.writers.add(threading.current_thread().name)
        return super().put(message, **kwargs)


def _command_peer(commands):
    """Connect a worker-side PAIR peer and wait until commands can reach it."""
    peer = IpcQueue(commands.address, is_server=False)
    peer.setup_lazily()
    deadline = time.monotonic() + 5
    while not commands.socket.poll(0, zmq.POLLOUT):
        assert time.monotonic() < deadline, "command peer did not connect"
        time.sleep(0.01)
    return peer


def _close_quietly(*queues):
    for ipc_queue in queues:
        if ipc_queue.socket is not None:
            ipc_queue.socket.setsockopt(zmq.LINGER, 0)
        ipc_queue.close()


@pytest.mark.timeout(20, method="thread")
def test_unbatched_admission_does_not_wait_for_the_sender_thread():
    # Regression: unbatched commands could only be written by the sender
    # thread. A busy event loop starved it of the GIL, so a connected and idle
    # worker received nothing while admission filled and rejected requests.
    commands = _OwnerStalledQueue(is_server=True, send_hwm=_RequestSender.MAX_INFLIGHT_BATCHES)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=True)
    sender.MAX_PENDING_REQUESTS = 4
    peer = _command_peer(commands)
    count = 4 * sender.MAX_PENDING_REQUESTS
    try:
        for item in range(count):
            sender.send(pickle.dumps(item))
        assert [pickle.loads(peer.get(timeout=2)[0]) for _ in range(count)] == list(range(count))
        assert commands.writers == {threading.current_thread().name}
        assert sender._pending_requests == 0
        assert errors.empty()
    finally:
        commands.release.set()
        sender.close(timeout=2)
        _close_quietly(peer)
    assert not sender._thread.is_alive()


@pytest.mark.timeout(30, method="thread")
def test_busy_event_loop_does_not_fill_unbatched_admission():
    # Regression: the exact serving shape. A coroutine that keeps the loop busy
    # and yields between submissions starved the sender thread of the GIL.
    commands = IpcQueue(is_server=True, send_hwm=_RequestSender.MAX_INFLIGHT_BATCHES)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=True)
    sender.MAX_PENDING_REQUESTS = 64
    peer = _command_peer(commands)
    count = 1000

    async def submit():
        rejected = 0
        for item in range(count):
            deadline = time.perf_counter() + 300e-6
            while time.perf_counter() < deadline:
                pass
            try:
                sender.send(pickle.dumps(item))
            except EngineUnavailableError:
                rejected += 1
            await asyncio.sleep(0)
        return rejected

    try:
        assert asyncio.run(submit()) == 0
        assert [pickle.loads(peer.get(timeout=2)[0]) for _ in range(count)] == list(range(count))
        assert errors.empty()
    finally:
        sender.close(timeout=2)
        _close_quietly(peer)


def _probe(port, kind):
    if kind in ("tcp", "garbage"):
        with socket.create_connection(("127.0.0.1", port)) as probe:
            if kind == "garbage":
                probe.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
                time.sleep(0.05)
        return
    socket_type = zmq.PAIR if kind == "zmtp" else zmq.DEALER
    with zmq.Context() as context, context.socket(socket_type) as probe:
        probe.setsockopt(zmq.LINGER, 0)
        probe.connect(f"tcp://127.0.0.1:{port}")
        time.sleep(0.2)


@pytest.mark.timeout(15, method="thread")
@pytest.mark.parametrize("direct_send", [False, True])
@pytest.mark.parametrize("kind", ["tcp", "garbage", "zmtp", "incompatible"])
def test_unrelated_connection_to_command_port_does_not_fail_the_engine(kind, direct_send):
    # Regression: any local connect/close on the loopback command port
    # produced a socket-monitor disconnect, which declared a healthy engine dead.
    commands = IpcQueue(is_server=True, send_hwm=_RequestSender.MAX_INFLIGHT_BATCHES)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=direct_send)
    peer = _command_peer(commands)
    try:
        sender.send(pickle.dumps("before"))
        assert peer.get(timeout=2) == _payloads("before")
        _probe(int(commands.address[0].rsplit(":", 1)[1]), kind)
        # The sender thread checks connection events at least every 50 ms.
        time.sleep(0.5)
        assert errors.empty()
        assert sender.available
        sender.send(pickle.dumps("after"))
        assert peer.get(timeout=2) == _payloads("after")
        # Losing the command peer itself remains terminal.
        _close_quietly(peer)
        assert isinstance(errors.get(timeout=2), ConnectionError)
        with pytest.raises(EngineDeadError):
            sender.send(pickle.dumps("after-disconnect"))
    finally:
        sender.close(timeout=2)
        _close_quietly(peer)


def test_command_connection_identifies_the_command_peer():
    from tensorrt_llm.executor.request_sender import _CommandConnection

    accepted, connected = zmq.EVENT_ACCEPTED, zmq.EVENT_CONNECTED
    succeeded, disconnected = zmq.EVENT_HANDSHAKE_SUCCEEDED, zmq.EVENT_DISCONNECTED
    failed = zmq.EVENT_HANDSHAKE_FAILED_NO_DETAIL

    def lost(*events):
        connection = _CommandConnection()
        return [connection.observe(event, value) for event, value in events]

    # A rejected connection, then the established peer's own loss.
    assert lost((accepted, 7), (succeeded, 0), (accepted, 8), (disconnected, 8), (disconnected, 7)) == [
        False, False, False, False, True,
    ]
    # A second completed handshake is not the command peer.
    assert lost((accepted, 7), (succeeded, 0), (accepted, 8), (succeeded, 0), (disconnected, 8))[-1] is False
    # Concurrent handshakes: the failed one is removed and the other remains the peer.
    assert lost(
        (accepted, 7), (accepted, 8), (succeeded, 0), (failed, 104), (disconnected, 8), (disconnected, 7)
    ) == [False, False, False, False, False, True]
    # Without a failure event, either concurrent candidate may be the peer.
    assert lost((accepted, 7), (accepted, 8), (succeeded, 0), (disconnected, 8))[-1] is True
    # A connection that disconnects before any handshake carried no commands.
    assert lost((connected, 9), (disconnected, 9)) == [False, False]
    # A completed handshake for an unobserved connection keeps every loss terminal.
    assert lost((succeeded, 0), (disconnected, 5)) == [False, True]
