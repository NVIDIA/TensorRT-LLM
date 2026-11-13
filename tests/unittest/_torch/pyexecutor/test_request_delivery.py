# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm._torch.pyexecutor.request_delivery import ResponseChannel

pytestmark = pytest.mark.cpu_only


def _response(request_id: int, client_id: int | None = None) -> LlmResponse:
    return LlmResponse(request_id=request_id, error_msg="done", client_id=client_id)


class _WaitQueue:
    def __init__(self):
        self.received = []
        self.put_response = SimpleNamespace(
            remote=lambda client_id, response: self.received.append((client_id, response))
        )


def _await_single(channel, request_id, timeout, *, shutdown=False, error=None, delivered=None):
    return channel.await_single(
        request_id,
        timeout,
        is_shutdown=lambda: shutdown,
        read_error=lambda: error,
        mark_error_delivered=(delivered or threading.Event()).set,
    )


def test_await_any_takes_every_response_once():
    channel = ResponseChannel()
    first, second, third = _response(1), _response(1), _response(2)
    channel.publish([(1, first), (2, third)])
    channel.publish([(1, second)])

    assert channel.await_any(0.01, is_shutdown=lambda: False) == [first, second, third]
    assert channel.await_any(0.01, is_shutdown=lambda: False) == []


def test_await_any_returns_empty_on_shutdown_without_waiting_for_timeout():
    channel = ResponseChannel()
    assert channel.await_any(None, is_shutdown=lambda: True) == []


def test_await_single_takes_only_its_own_responses():
    channel = ResponseChannel()
    mine, other = _response(1), _response(2)
    channel.publish([(1, mine), (2, other)])

    assert _await_single(channel, 1, 0.01) == [mine]
    assert _await_single(channel, 1, 0.01) == []
    assert channel.await_any(0.01, is_shutdown=lambda: False) == [other]


def test_await_single_wakes_when_a_response_is_published():
    channel = ResponseChannel()
    response = _response(5)
    publisher = threading.Timer(0.05, channel.publish, args=([(5, response)],))
    publisher.start()
    try:
        assert _await_single(channel, 5, 5.0) == [response]
    finally:
        publisher.join()


def test_await_single_returns_pending_response_after_shutdown():
    channel = ResponseChannel()
    response = _response(3)
    channel.publish([(3, response)])
    delivered = threading.Event()

    assert _await_single(
        channel, 3, 0.01, shutdown=True, error=ValueError("x"), delivered=delivered
    ) == [response]
    assert not delivered.is_set()


def test_await_single_raises_original_error_on_shutdown_and_marks_it_delivered():
    channel = ResponseChannel()
    error = ValueError("loop crashed")
    delivered = threading.Event()

    with pytest.raises(RuntimeError, match="Event loop terminated") as raised:
        _await_single(channel, 3, None, shutdown=True, error=error, delivered=delivered)

    assert raised.value.__cause__ is error
    assert delivered.is_set()


def test_await_single_raises_on_clean_shutdown_without_marking_delivery():
    channel = ResponseChannel()
    delivered = threading.Event()

    with pytest.raises(RuntimeError, match="shut down before a response"):
        _await_single(channel, 3, None, shutdown=True, delivered=delivered)

    assert not delivered.is_set()


def test_registered_wait_queue_receives_responses_until_released():
    channel = ResponseChannel()
    wait_queue = _WaitQueue()
    channel.register([7, 8], wait_queue)

    first = _response(7, client_id=70)
    channel.publish([(7, first)])
    channel.release(7)
    channel.publish([(7, _response(7, client_id=70)), (9, _response(9, client_id=90))])

    assert wait_queue.received == [(70, first)]
