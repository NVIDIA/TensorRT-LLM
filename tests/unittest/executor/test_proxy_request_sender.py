# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the proxy's request sender, the sole writer of its request queue."""

import queue
import threading

from tensorrt_llm.executor.proxy import _RequestSender


class _RecordingQueue:
    """Request-queue stand-in that records each message; the first put can block."""

    def __init__(self, block_first_put: bool = False):
        self.messages = []
        self.first_put_started = threading.Event()
        self.release_first_put = threading.Event()
        if not block_first_put:
            self.release_first_put.set()

    def put(self, msg):
        if not self.messages:
            self.first_put_started.set()
            self.release_first_put.wait(timeout=10)
        self.messages.append(msg)


def test_lone_request_is_sent_unbatched():
    request_queue = _RecordingQueue()
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send("a")
    sender.close(timeout=10)
    assert request_queue.messages == ["a"]


def test_burst_is_sent_as_one_list_in_order():
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send(0)
    assert request_queue.first_put_started.wait(timeout=10)
    # Everything queued while the first send is in flight goes out as one list.
    for item in range(1, 6):
        sender.send(item)
    request_queue.release_first_put.set()
    sender.close(timeout=10)
    assert request_queue.messages == [0, [1, 2, 3, 4, 5]]


def test_close_flushes_queued_requests_and_drops_later_ones():
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _RequestSender(request_queue, queue.Queue())
    sender.send("first")
    assert request_queue.first_put_started.wait(timeout=10)
    sender.send("queued")
    closer = threading.Thread(target=sender.close, kwargs={"timeout": 10})
    closer.start()
    request_queue.release_first_put.set()
    closer.join(timeout=10)
    assert not closer.is_alive()
    # A shutdown sentinel written after close() follows every queued request.
    assert request_queue.messages == ["first", "queued"]
    sender.send("late")
    assert request_queue.messages == ["first", "queued"]


def test_send_error_reaches_error_queue():
    class _FailingQueue:
        def put(self, msg):
            raise RuntimeError("socket closed")

    error_queue = queue.Queue()
    sender = _RequestSender(_FailingQueue(), error_queue)
    sender.send("a")
    sender.close(timeout=10)
    error = error_queue.get_nowait()
    assert isinstance(error, RuntimeError)
    assert str(error) == "socket closed"
