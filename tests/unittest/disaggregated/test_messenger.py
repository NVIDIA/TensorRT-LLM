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

import socket
import time
import unittest
from threading import Event, Lock, Thread

import pytest
import zmq
from parameterized import parameterized

from tensorrt_llm._torch.disaggregation.native.messenger import ZMQMessenger, decode_message
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip

TEST_CASES = [
    {
        "name": "valid_message",
        "message": [b"hello", b"world"],
        "encoding": "utf-8",
        "err_mode": "strict",
        "expected": ("hello", "world"),
        "raises": None,
    },
    {
        "name": "invalid_input",
        "message": ["hello", b"world"],
        "encoding": "utf-8",
        "err_mode": "strict",
        "expected": None,
        "raises": ValueError,
    },
    {
        "name": "decoding_error",
        "message": [b"\xff"],
        "encoding": "utf-8",
        "err_mode": "strict",
        "expected": None,
        "raises": UnicodeDecodeError,
    },
    {
        "name": "decoding_with_ignore",
        "message": [b"\xff"],
        "encoding": "utf-8",
        "err_mode": "ignore",
        "expected": ("",),
        "raises": None,
    },
]


class TestDecodeMessage(unittest.TestCase):
    @parameterized.expand([(case["name"], case) for case in TEST_CASES])
    def test_decode_message(self, name, case):
        message = case["message"]
        encoding = case["encoding"]
        err_mode = case["err_mode"]
        expected = case["expected"]
        raises = case["raises"]

        if raises:
            with self.assertRaises(raises):
                decode_message(message, encoding=encoding, err_mode=err_mode)
        else:
            decoded = decode_message(message, encoding=encoding, err_mode=err_mode)
            self.assertEqual(decoded, expected)


@pytest.fixture
def dynamic_endpoint():
    """Fixture to dynamically generate an available endpoint with a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to an available port provided by the OS
        port = s.getsockname()[1]
        return f"tcp://{get_local_ip()}:{port}"


@pytest.fixture
def create_messenger_pair(dynamic_endpoint):
    def _create_messenger_pair(mode1, mode2):
        messenger1 = ZMQMessenger(
            mode1, endpoint=dynamic_endpoint if mode1 in ["ROUTER", "REP"] else None
        )
        messenger2 = ZMQMessenger(
            mode2, endpoint=dynamic_endpoint if mode2 in ["DEALER", "REQ"] else None
        )
        return messenger1, messenger2

    yield _create_messenger_pair


def test_router_dealer(create_messenger_pair):
    """Test ROUTER and DEALER communication."""
    router, dealer = create_messenger_pair("ROUTER", "DEALER")

    received_messages = []

    def on_message(messages):
        received_messages.extend(messages)

    router.start_listener(on_message)

    dealer.send([b"Hello, ROUTER!"])

    time.sleep(0.1)

    assert len(received_messages) > 0
    assert b"Hello, ROUTER!" in received_messages

    router.stop()
    dealer.stop()


def test_req_rep(create_messenger_pair):
    """Test REQ and REP communication."""
    rep, req = create_messenger_pair("REP", "REQ")

    def on_message(messages):
        rep.send(messages)

    rep.start_listener(on_message)

    req.send([b"Hello, REP!"])
    response = req.receive()
    assert response == [b"Hello, REP!"]

    req.stop()
    rep.stop()


def test_zmq_messenger_context_manager(dynamic_endpoint):
    with ZMQMessenger("ROUTER", endpoint=dynamic_endpoint) as messenger:
        assert messenger.endpoint == dynamic_endpoint
    assert messenger._closed is True


def test_zmq_messenger_invalid_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        ZMQMessenger("INVALID_MODE")


def test_zmq_messenger_double_start_listener(dynamic_endpoint):
    messenger = ZMQMessenger("ROUTER", endpoint=dynamic_endpoint)
    messenger.start_listener(lambda msgs: None)
    with pytest.raises(RuntimeError, match="Listener already running"):
        messenger.start_listener(lambda msgs: None)
    messenger.stop()


class _CloseTrackingSocket:
    def __init__(self):
        self.closed = False
        self.close_calls = 0
        self.setsockopt_calls = 0
        self.send_calls = 0
        self.send_flags = []
        self.send_multipart_calls = 0

    def send(self, _message, flags=0):
        self.send_calls += 1
        self.send_flags.append(flags)

    def send_multipart(self, _message):
        self.send_multipart_calls += 1

    def setsockopt(self, _option, _value):
        self.setsockopt_calls += 1

    def close(self):
        self.close_calls += 1
        self.closed = True


class _FailOnceSocket(_CloseTrackingSocket):
    def __init__(self, operation):
        super().__init__()
        self._operation = operation
        self._failed = False

    def setsockopt(self, option, value):
        super().setsockopt(option, value)
        if self._operation == "setsockopt" and not self._failed:
            self._failed = True
            raise RuntimeError("injected setsockopt failure")

    def close(self):
        self.close_calls += 1
        if self._operation == "close" and not self._failed:
            self._failed = True
            raise RuntimeError("injected close failure")
        self.closed = True


class _CloseTrackingContext:
    def __init__(self):
        self.closed = False
        self.term_calls = 0

    def term(self):
        self.term_calls += 1
        self.closed = True


class _FailOnceContext(_CloseTrackingContext):
    def term(self):
        self.term_calls += 1
        if self.term_calls == 1:
            raise RuntimeError("injected context termination failure")
        self.closed = True


class _ControllableThread:
    def __init__(self):
        self.alive = True
        self.join_calls = []

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.join_calls.append(timeout)


def _make_test_messenger(main_socket=None, listener_thread=None):
    messenger = object.__new__(ZMQMessenger)
    messenger._lock = Lock()
    messenger._stop_lock = Lock()
    messenger._socket_io_lock = Lock()
    messenger._control_send_lock = Lock()
    messenger._io_waiters_lock = Lock()
    messenger._io_waiters = 0
    messenger._io_waiters_drained = Event()
    messenger._io_waiters_drained.set()
    messenger._closed = False
    messenger._closing = False
    messenger._stop_event = Event()
    messenger._socket = main_socket or _CloseTrackingSocket()
    messenger._internal_socket = _CloseTrackingSocket()
    messenger._control_socket = _CloseTrackingSocket()
    messenger._context = _CloseTrackingContext()
    messenger._listener_thread = listener_thread
    return messenger


def _wait_for_closing(messenger, timeout=1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with messenger._lock:
            if messenger._closing:
                return
        time.sleep(0.001)
    pytest.fail("messenger did not enter the closing state")


def test_stop_retries_without_closing_sockets_under_live_listener():
    messenger = _make_test_messenger(listener_thread=_ControllableThread())

    with pytest.raises(RuntimeError, match="listener thread did not terminate"):
        messenger.stop(timeout=0)

    assert messenger._stop_event.is_set()
    assert messenger._closed is False
    assert messenger._listener_thread.join_calls == [0]
    assert messenger._internal_socket.send_calls == 1
    assert messenger._internal_socket.send_flags == [zmq.DONTWAIT]
    assert messenger._socket.close_calls == 0
    assert messenger._internal_socket.close_calls == 0
    assert messenger._control_socket.close_calls == 0
    assert messenger._context.term_calls == 0

    messenger._listener_thread.alive = False
    messenger.stop(timeout=1)
    messenger.stop(timeout=1)

    assert messenger._closed is True
    assert messenger._socket.close_calls == 1
    assert messenger._internal_socket.close_calls == 1
    assert messenger._control_socket.close_calls == 1
    assert messenger._context.term_calls == 1


@pytest.mark.parametrize(
    ("operation", "error_match", "expected_close_calls"),
    [
        ("setsockopt", "main socket linger configuration failed", 1),
        ("close", "main socket close failed", 2),
    ],
)
def test_stop_retries_failed_socket_teardown(operation, error_match, expected_close_calls):
    main_socket = _FailOnceSocket(operation)
    messenger = _make_test_messenger(main_socket=main_socket)

    with pytest.raises(RuntimeError, match=error_match):
        messenger.stop()

    assert messenger._closing is True
    assert messenger._closed is False
    assert main_socket.closed is False
    assert messenger._internal_socket.closed is True
    assert messenger._control_socket.closed is True
    assert messenger._context.term_calls == 0

    messenger.stop()

    assert messenger._closed is True
    assert main_socket.setsockopt_calls == 2
    assert main_socket.close_calls == expected_close_calls
    # Sockets that closed on the first attempt are not touched by the retry.
    assert messenger._internal_socket.setsockopt_calls == 1
    assert messenger._internal_socket.close_calls == 1
    assert messenger._control_socket.setsockopt_calls == 1
    assert messenger._control_socket.close_calls == 1
    assert messenger._context.term_calls == 1


def test_stop_retries_failed_context_termination_without_reclosing_sockets():
    messenger = _make_test_messenger()
    messenger._context = _FailOnceContext()

    with pytest.raises(RuntimeError, match="context termination failed"):
        messenger.stop()

    assert messenger._closing is True
    assert messenger._closed is False
    assert messenger._socket.closed is True
    assert messenger._internal_socket.closed is True
    assert messenger._control_socket.closed is True
    assert messenger._context.term_calls == 1

    messenger.stop()

    assert messenger._closed is True
    assert messenger._socket.close_calls == 1
    assert messenger._internal_socket.close_calls == 1
    assert messenger._control_socket.close_calls == 1
    assert messenger._context.term_calls == 2


class _BlockingSendSocket(_CloseTrackingSocket):
    def __init__(self):
        super().__init__()
        self.first_send_started = Event()
        self.second_send_started = Event()
        self.release_first_send = Event()
        self.first_send_finished = Event()
        self.close_overlapped_send = False
        self._call_lock = Lock()

    def send_multipart(self, _message):
        with self._call_lock:
            self.send_multipart_calls += 1
            call_index = self.send_multipart_calls
        if call_index == 1:
            self.first_send_started.set()
            assert self.release_first_send.wait(timeout=2)
            self.first_send_finished.set()
        else:
            self.second_send_started.set()

    def close(self):
        if self.first_send_started.is_set() and not self.first_send_finished.is_set():
            self.close_overlapped_send = True
        super().close()


def test_main_socket_serializes_concurrent_sends():
    main_socket = _BlockingSendSocket()
    messenger = _make_test_messenger(main_socket=main_socket)
    errors = []

    def send(message):
        try:
            messenger.send([message])
        except BaseException as error:
            errors.append(error)

    first = Thread(target=send, args=(b"first",))
    second = Thread(target=send, args=(b"second",))
    first.start()
    assert main_socket.first_send_started.wait(timeout=1)
    second.start()
    try:
        assert not main_socket.second_send_started.wait(timeout=0.1)
    finally:
        main_socket.release_first_send.set()
        first.join(timeout=2)
        second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert main_socket.second_send_started.is_set()
    assert main_socket.send_multipart_calls == 2


class _BlockingReceiveSocket(_CloseTrackingSocket):
    def __init__(self):
        super().__init__()
        self.receive_started = Event()
        self.release_receive = Event()
        self.send_started = Event()

    def recv_multipart(self):
        self.receive_started.set()
        assert self.release_receive.wait(timeout=2)
        return [b"response"]

    def send_multipart(self, _message):
        super().send_multipart(_message)
        self.send_started.set()


def test_main_socket_serializes_receive_with_send():
    main_socket = _BlockingReceiveSocket()
    messenger = _make_test_messenger(main_socket=main_socket)
    responses = []
    errors = []

    def receive():
        try:
            responses.append(messenger.receive())
        except BaseException as error:
            errors.append(error)

    def send():
        try:
            messenger.send([b"request"])
        except BaseException as error:
            errors.append(error)

    receive_thread = Thread(target=receive)
    send_thread = Thread(target=send)
    receive_thread.start()
    assert main_socket.receive_started.wait(timeout=1)
    send_thread.start()
    try:
        assert not main_socket.send_started.wait(timeout=0.1)
    finally:
        main_socket.release_receive.set()
        receive_thread.join(timeout=2)
        send_thread.join(timeout=2)

    assert not receive_thread.is_alive()
    assert not send_thread.is_alive()
    assert errors == []
    assert responses == [[b"response"]]
    assert main_socket.send_started.is_set()


def test_stop_waits_for_admitted_send_and_rejects_late_send():
    main_socket = _BlockingSendSocket()
    messenger = _make_test_messenger(main_socket=main_socket)
    errors = []

    def run_send(message):
        try:
            messenger.send([message])
        except BaseException as error:
            errors.append(error)

    first = Thread(target=run_send, args=(b"first",))
    stop_thread = Thread(target=messenger.stop)
    late = Thread(target=run_send, args=(b"late",))
    first.start()
    assert main_socket.first_send_started.wait(timeout=1)
    stop_thread.start()
    _wait_for_closing(messenger)
    late.start()
    try:
        assert main_socket.close_calls == 0
    finally:
        main_socket.release_first_send.set()
        first.join(timeout=2)
        stop_thread.join(timeout=2)
        late.join(timeout=2)

    assert not first.is_alive()
    assert not stop_thread.is_alive()
    assert not late.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "ZMQMessenger is stopping or closed"
    assert main_socket.send_multipart_calls == 1
    assert main_socket.close_calls == 1
    assert not main_socket.close_overlapped_send
    assert messenger._closed


def test_stop_times_out_without_closing_under_blocked_main_socket_io():
    main_socket = _BlockingSendSocket()
    messenger = _make_test_messenger(main_socket=main_socket)
    send_errors = []

    def run_send():
        try:
            messenger.send([b"blocked"])
        except BaseException as error:
            send_errors.append(error)

    send_thread = Thread(target=run_send)
    send_thread.start()
    assert main_socket.first_send_started.wait(timeout=1)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="I/O did not quiesce"):
        messenger.stop(timeout=0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert main_socket.close_calls == 0
    assert not main_socket.close_overlapped_send

    main_socket.release_first_send.set()
    send_thread.join(timeout=2)
    assert not send_thread.is_alive()
    assert send_errors == []

    messenger.stop(timeout=1)
    assert messenger._closed
    assert main_socket.close_calls == 1


def test_stop_does_not_deadlock_listener_callback_entering_send():
    main_socket = _CloseTrackingSocket()
    callback_started = Event()
    enter_send = Event()
    send_rejected = Event()
    messenger = _make_test_messenger(main_socket=main_socket)

    def listener_callback():
        callback_started.set()
        assert enter_send.wait(timeout=2)
        with pytest.raises(RuntimeError, match="stopping or closed"):
            messenger.send([b"callback"])
        send_rejected.set()

    listener_thread = Thread(target=listener_callback)
    messenger._listener_thread = listener_thread
    listener_thread.start()
    assert callback_started.wait(timeout=1)

    stop_thread = Thread(target=messenger.stop)
    stop_thread.start()
    _wait_for_closing(messenger)
    enter_send.set()
    stop_thread.join(timeout=2)
    listener_thread.join(timeout=2)

    assert not stop_thread.is_alive()
    assert not listener_thread.is_alive()
    assert send_rejected.is_set()
    assert main_socket.send_multipart_calls == 0
    assert main_socket.close_calls == 1
    assert messenger._closed


def test_stop_can_run_from_listener_thread():
    messenger = _make_test_messenger()
    start_stop = Event()
    errors = []

    def stop_from_listener():
        assert start_stop.wait(timeout=1)
        try:
            messenger.stop(timeout=1)
        except BaseException as error:
            errors.append(error)

    listener_thread = Thread(target=stop_from_listener)
    messenger._listener_thread = listener_thread
    listener_thread.start()
    start_stop.set()
    listener_thread.join(timeout=2)

    assert not listener_thread.is_alive()
    assert errors == []
    assert messenger._closed
    assert messenger._socket.close_calls == 1


if __name__ == "__main__":
    unittest.main()
