import socket
import time
import unittest

import pytest
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


if __name__ == "__main__":
    unittest.main()
