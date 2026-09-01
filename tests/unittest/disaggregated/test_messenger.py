# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import socket
import time
import unittest
from threading import Lock, Thread, get_ident

import pytest
from parameterized import parameterized

from tensorrt_llm._torch.disaggregation.native import messenger as messenger_module
from tensorrt_llm._torch.disaggregation.native.messenger import (
    ZMQDealerPool,
    ZMQMessenger,
    decode_message,
)
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


def test_zmq_messengers_share_process_context(dynamic_endpoint):
    first = ZMQMessenger("DEALER", endpoint=dynamic_endpoint)
    second = ZMQMessenger("DEALER", endpoint=dynamic_endpoint)

    try:
        assert first._context is second._context
        first.stop()
        assert not second._context.closed
    finally:
        first.stop()
        second.stop()


def test_zmq_messenger_parallel_creation_uses_one_context(dynamic_endpoint):
    contexts = []
    errors = []
    lock = Lock()

    def create_and_stop() -> None:
        try:
            messenger = ZMQMessenger("DEALER", endpoint=dynamic_endpoint)
            with lock:
                contexts.append(messenger._context)
            messenger.stop()
        except BaseException as error:
            with lock:
                errors.append(error)

    threads = [Thread(target=create_and_stop) for _ in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert len(contexts) == len(threads)
    assert len({id(context) for context in contexts}) == 1


def test_zmq_dealer_pool_confines_sockets_to_owner_thread(monkeypatch):
    socket_threads = set()
    lock = Lock()

    class FakeMessenger:
        def __init__(self, mode, endpoint):
            assert mode == "DEALER"
            assert endpoint in {"tcp://peer-a:1", "tcp://peer-b:2"}
            with lock:
                socket_threads.add(get_ident())

        def send(self, messages):
            assert messages
            with lock:
                socket_threads.add(get_ident())

        def stop(self):
            with lock:
                socket_threads.add(get_ident())

    monkeypatch.setattr(messenger_module, "ZMQMessenger", FakeMessenger)
    pool = ZMQDealerPool()
    threads = [
        Thread(
            target=pool.send,
            args=(
                f"tcp://peer-{'a' if index % 2 == 0 else 'b'}:{1 if index % 2 == 0 else 2}",
                [b"x"],
            ),
        )
        for index in range(32)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    pool.stop()

    assert not any(thread.is_alive() for thread in threads)
    assert len(socket_threads) == 1


def test_zmq_dealer_pool_rejects_send_after_stop():
    pool = ZMQDealerPool()
    pool.stop()
    with pytest.raises(RuntimeError, match="closed"):
        pool.send("tcp://peer:1", [b"late"])


if __name__ == "__main__":
    unittest.main()
