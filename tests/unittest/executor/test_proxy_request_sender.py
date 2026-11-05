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
from types import SimpleNamespace

import pytest
import zmq

from tensorrt_llm.executor.ipc import IpcQueue
from tensorrt_llm.executor.proxy import (
    GenerationExecutorProxy,
    _create_request_sender,
    _RequestSender,
)
from tensorrt_llm.executor.request import CancellingRequest, _CommandBarrier
from tensorrt_llm.executor.utils import EngineUnavailableError
from tensorrt_llm.executor.worker import _incoming_requests

pytestmark = pytest.mark.cpu_only


class _RecordingQueue:
    """Request-queue stand-in that records each message; the first put can block.

    Like ``ZeroMqQueue`` with ``TLLM_LLMAPI_ZMQ_DEBUG=1``, it rejects a write from
    any thread other than the first one that wrote.
    """

    def __init__(self, block_first_put: bool = False):
        self.messages = []
        self.socket = SimpleNamespace(setsockopt=lambda *args: None)
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
    q = _RecordingQueue(block_first_put=batching)
    sender = _create_request_sender(q, queue.Queue())
    sender.send(pickle.dumps("a"))
    delivered_on_return = bool(q.messages)
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
    if not batching:
        assert delivered_on_return, "unbatched submission returned before its command was sent"


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
    assert result == [False]
    # A bare None after the payload lists, written by the sender thread.
    assert request_queue.messages == [_payloads("first"), _payloads("queued"), None]


def test_shutdown_without_requests_still_uses_the_socket_owner():
    q = _RecordingQueue()
    sender = _RequestSender(q, queue.Queue())
    assert sender.close(timeout=10, send_sentinel=True) is False
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
    assert sender.close(send_sentinel=True) is False
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
        assert responsibility == [False]
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


@pytest.mark.timeout(12, method="thread")
@pytest.mark.parametrize("push,acknowledge", [(False, True), (True, True), (False, False)])
def test_connection_recovery_requires_worker_receipt(tmp_path, push, acknowledge):
    """Connection recovery must prove delivery before accepting new work."""
    if push:
        address = (f"ipc://{tmp_path}/commands", b"k" * 32)
        commands = IpcQueue(address, is_server=False, socket_type=zmq.PUSH)
    else:
        commands = IpcQueue(is_server=True)
        address = commands.address
    errors = queue.Queue()
    sender = _RequestSender(
        commands, errors, batching=False, recovery_timeout=2, stop_workers_on_failure=True
    )
    received = []
    worker_errors = []

    def worker():
        peer = IpcQueue(address, is_server=push, socket_type=zmq.PULL if push else zmq.PAIR)
        try:
            while peer.poll(timeout=3):
                message = peer.get()
                if message is None:
                    return
                for payload in message:
                    request = pickle.loads(payload)
                    if isinstance(request, _CommandBarrier):
                        if acknowledge:
                            ack = IpcQueue(request.ack_addr, is_server=False, socket_type=zmq.PUSH)
                            try:
                                ack.put(request.token, timeout=1)
                            finally:
                                ack.close()
                    else:
                        received.append(request)
            raise AssertionError("Worker did not receive shutdown")
        except BaseException as error:
            worker_errors.append(error)
        finally:
            peer.close()

    thread = None
    try:
        sender.send(pickle.dumps("first"))
        with sender._cond:
            assert sender._cond.wait_for(lambda: sender._recovering, timeout=2)
        with pytest.raises(EngineUnavailableError):
            sender.send(pickle.dumps("rejected"))
        thread = threading.Thread(target=worker)
        thread.start()
        with sender._cond:
            assert sender._cond.wait_for(
                lambda: not sender._recovering or sender._terminal_error is not None, timeout=3
            )
        if acknowledge:
            assert errors.empty()
            assert not sender.recovering
            sender.send(pickle.dumps("after-recovery"))
        else:
            assert isinstance(errors.get(timeout=1), TimeoutError)
        sender.close(timeout=3, send_sentinel=True)
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not worker_errors
        assert received == (["first", "after-recovery"] if acknowledge else ["first"])
    finally:
        sender.close(timeout=3, send_sentinel=True)
        if thread is not None:
            thread.join(timeout=3)


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


@pytest.mark.timeout(12, method="thread")
def test_connected_worker_backpressure_pauses_admission():
    # A connected peer that stops reading must eventually block delivery too.
    # Large packets fill the bounded TCP buffers as well as the ZMQ queues.
    peer = IpcQueue(is_server=True, receive_hwm=1)
    commands = IpcQueue(peer.address, is_server=False, send_hwm=1)
    sender = _RequestSender(commands, queue.Queue(), batching=False, recovery_timeout=2)
    payload = pickle.dumps(b"x" * (1024 * 1024))
    try:
        for _ in range(64):
            try:
                sender.send(payload)
            except EngineUnavailableError:
                break
        with sender._cond:
            assert sender._cond.wait_for(lambda: sender._recovering, timeout=2)
        with pytest.raises(EngineUnavailableError):
            sender.send(payload)
    finally:
        sender.fail(RuntimeError("test complete"))
        sender.close(timeout=3)
        peer.socket.setsockopt(zmq.LINGER, 0)
        peer.close()


@pytest.mark.timeout(12, method="thread")
@pytest.mark.parametrize("drop_first_ack", [False, True])
def test_recovery_acknowledgements_survive_reconnecting_worker(drop_first_ack):
    delivered = []

    class Commands(_RecordingQueue):
        retry = False
        acknowledged = 0
        dropped_ack = False

        def put(self, message, **kwargs):
            request = pickle.loads(message[0])
            if isinstance(request, _CommandBarrier):
                if drop_first_ack and not self.dropped_ack:
                    # The worker can time out opening its ACK connection while
                    # the frontend is paused, then continue processing commands.
                    self.dropped_ack = True
                    return
                # Each barrier gets a fresh worker-side connection, as in
                # worker_main. The receiver persists across recoveries.
                ack = IpcQueue(request.ack_addr, is_server=False, socket_type=zmq.PUSH)
                try:
                    ack.socket.setsockopt(zmq.IMMEDIATE, 1)
                    ack.put(request.token, timeout=0.2)
                    self.acknowledged += 1
                finally:
                    ack.socket.setsockopt(zmq.LINGER, 200)
                    ack.close()
            elif not self.retry:
                self.retry = True
                raise zmq.Again()
            else:
                self.retry = False
                delivered.append(request)

    commands = Commands()
    sender = _RequestSender(commands, queue.Queue(), batching=False, recovery_timeout=1)
    try:
        for cycle in range(3):
            sender.send(pickle.dumps(cycle))
            with sender._cond:
                assert sender._cond.wait_for(
                    lambda: (commands.acknowledged == cycle + 1 and not sender._recovering)
                    or sender._terminal_error is not None,
                    timeout=2,
                )
            assert sender.available
        assert delivered == [0, 1, 2]
    finally:
        sender.close(timeout=3)
