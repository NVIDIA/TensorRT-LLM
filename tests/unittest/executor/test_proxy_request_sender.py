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

    def put_noblock(self, msg, *, retry=1, wait_time=0.001):
        self._check_writer()
        self.messages.append(msg)


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


def test_transport_error_reaches_error_queue():
    class _FailingQueue:
        def put(self, msg):
            raise RuntimeError("socket closed")

    error_queue = queue.Queue()
    sender = _RequestSender(_FailingQueue(), error_queue)
    sender.send(pickle.dumps("a"))
    sender.close(timeout=10)
    error = error_queue.get_nowait()
    assert isinstance(error, RuntimeError)
    assert str(error) == "socket closed"


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
