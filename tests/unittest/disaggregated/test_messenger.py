import socket
import time
import unittest

import pytest
from parameterized import parameterized

from tensorrt_llm._torch.disaggregation.native.messenger import (
    FrameAuthenticationError,
    FrameSigner,
    ZMQMessenger,
    decode_message,
)
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip

pytestmark = pytest.mark.cpu_only


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
            mode1, endpoint=dynamic_endpoint if mode1 in ["ROUTER", "REP"] else None, signer=None
        )
        messenger2 = ZMQMessenger(
            mode2, endpoint=dynamic_endpoint if mode2 in ["DEALER", "REQ"] else None, signer=None
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
    with ZMQMessenger("ROUTER", endpoint=dynamic_endpoint, signer=None) as messenger:
        assert messenger.endpoint == dynamic_endpoint
    assert messenger._closed is True


def test_zmq_messenger_invalid_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        ZMQMessenger("INVALID_MODE", signer=None)


def test_zmq_messenger_double_start_listener(dynamic_endpoint):
    messenger = ZMQMessenger("ROUTER", endpoint=dynamic_endpoint, signer=None)
    messenger.start_listener(lambda msgs: None)
    with pytest.raises(RuntimeError, match="Listener already running"):
        messenger.start_listener(lambda msgs: None)
    messenger.stop()


_KEY = b"0123456789abcdef0123456789abcdef"
_OTHER_KEY = b"fedcba9876543210fedcba9876543210"
_CONTEXT = b"trtllm-test-ctl-v1\0"


def _signer(keys=None, active_key_id=1, context=_CONTEXT):
    return FrameSigner(keys or {1: _KEY}, active_key_id, context)


def _signed(frames, signer=None, **kwargs):
    params = {
        "sender_id": b"sender",
        "recipient_id": b"recipient",
        "flow": FrameSigner.FLOW_TO_ROUTER,
        **kwargs,
    }
    return list(frames) + (signer or _signer()).sign(list(frames), **params)


def _verify(message, signer=None, recipient_id=b"recipient", flow=FrameSigner.FLOW_TO_ROUTER):
    return (signer or _signer()).verify(message, recipient_id=recipient_id, flow=flow)


def test_frame_signer_round_trip():
    payload, header = _verify(_signed([b"KV_AGENT_RESULT", b"prefix", b""]))
    assert payload == [b"KV_AGENT_RESULT", b"prefix", b""]
    assert header.sender_id == b"sender"
    assert header.key_id == 1
    assert header.flow == FrameSigner.FLOW_TO_ROUTER


@pytest.mark.parametrize("index", [0, 1, 2])
def test_frame_signer_rejects_tampered_payload(index):
    message = _signed([b"AUX_AGENT_RESULT", b"1", b"23"])
    message[index] += b"x"
    with pytest.raises(FrameAuthenticationError, match="MAC mismatch"):
        _verify(message)


def test_frame_signer_rejects_shifted_frame_boundary():
    # Plain concatenation of both payloads is b"AUX123OK"; the length prefixes
    # are what keep rank 1 / rid 23 from being re-cut into rank 12 / rid 3.
    message = _signed([b"AUX", b"1", b"23", b"OK"])
    shifted = [b"AUX", b"12", b"3", b"OK"] + message[-2:]
    with pytest.raises(FrameAuthenticationError, match="MAC mismatch"):
        _verify(shifted)


def test_frame_signer_rejects_dropped_or_appended_frames():
    message = _signed([b"KV_AGENT_RESULT", b"prefix", b"dst", b"sizes"])
    with pytest.raises(FrameAuthenticationError, match="MAC mismatch"):
        _verify(message[:2] + message[-2:])
    with pytest.raises(FrameAuthenticationError, match="MAC mismatch"):
        _verify(message[:-2] + [b"extra"] + message[-2:])


@pytest.mark.parametrize(
    "verifier_kwargs, match",
    [
        ({"signer": FrameSigner({1: _OTHER_KEY}, 1, _CONTEXT)}, "MAC mismatch"),
        ({"signer": FrameSigner({1: _KEY}, 1, b"another-protocol-v1\0")}, "MAC mismatch"),
        ({"recipient_id": b"someone-else"}, "MAC mismatch"),
        ({"flow": FrameSigner.FLOW_TO_DEALER}, "other flow"),
    ],
    ids=["wrong_key", "wrong_context", "wrong_recipient", "reflected"],
)
def test_frame_signer_rejects_mismatched_verifier(verifier_kwargs, match):
    with pytest.raises(FrameAuthenticationError, match=match):
        _verify(_signed([b"REQUEST_DATA", b"info", b"token"]), **verifier_kwargs)


def test_frame_signer_rejects_rewritten_header_fields():
    message = _signed([b"CANCEL_SESSION", b"1234"])
    message[-2] = message[-2][: -len(b"sender")] + b"spoofd"
    with pytest.raises(FrameAuthenticationError, match="MAC mismatch"):
        _verify(message)


def test_frame_signer_rejects_malformed_auth_frames():
    header, mac = _signed([b"TAG"])[-2:]
    for message in (
        [b"TAG"],
        [b"TAG", header, mac[:-1]],
        [b"TAG", header[:5], mac],
        [b"TAG", header + b"extra", mac],
    ):
        with pytest.raises(FrameAuthenticationError):
            _verify(message)


def test_frame_signer_rotation_accepts_every_configured_key():
    old = _signer({1: _KEY}, 1)
    rolling = _signer({1: _KEY, 2: _OTHER_KEY}, 2)
    new = _signer({2: _OTHER_KEY}, 2)
    from_old = _signed([b"TAG"], signer=old)
    assert _verify(from_old, signer=rolling)[1].key_id == 1
    assert _verify(_signed([b"TAG"], signer=rolling), signer=rolling)[1].key_id == 2
    with pytest.raises(FrameAuthenticationError, match="Unknown key id"):
        _verify(from_old, signer=new)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"keys": {}, "active_key_id": 1}, "at least one key"),
        ({"keys": {1: b"short"}, "active_key_id": 1}, "at least 16 bytes"),
        ({"keys": {256: _KEY}, "active_key_id": 256}, "outside 0-255"),
        ({"keys": {1: _KEY}, "active_key_id": 2}, "not among"),
        ({"keys": {1: _KEY}, "active_key_id": 1, "context": b""}, "non-empty context"),
    ],
)
def test_frame_signer_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FrameSigner(**{"context": _CONTEXT, **kwargs})


def test_frame_signer_repr_omits_key_material():
    assert "0123456789abcdef" not in repr(_signer())


def test_messenger_requires_explicit_signer_choice():
    with pytest.raises(TypeError):
        ZMQMessenger("ROUTER", "tcp://127.0.0.1:*")


def test_signed_messenger_validates_mode_and_dealer_identity():
    with pytest.raises(ValueError, match="only ROUTER and DEALER"):
        ZMQMessenger("REP", "tcp://127.0.0.1:*", signer=_signer())
    with pytest.raises(ValueError, match="identity"):
        ZMQMessenger("DEALER", "tcp://127.0.0.1:5555", signer=_signer())


@pytest.fixture
def signed_pair():
    signer = _signer()
    router = ZMQMessenger("ROUTER", "tcp://127.0.0.1:*", signer=signer)
    dealer = ZMQMessenger(
        "DEALER", router.endpoint, signer=signer, identity="tcp://127.0.0.1:40001"
    )
    yield router, dealer
    dealer.stop()
    router.stop()


def test_signed_router_dealer_round_trip(signed_pair):
    router, dealer = signed_pair
    dealer.send([b"REQUEST_INSTANCE_INFO"])
    routing_id, request = router.receive(timeout_ms=1000)
    assert request == b"REQUEST_INSTANCE_INFO"
    router.send([routing_id, b"rank-info"])
    assert dealer.receive(timeout_ms=1000) == [b"rank-info"]
    router.send([b"second"], recipient=routing_id)
    assert dealer.receive(timeout_ms=1000) == [b"second"]


def test_signed_router_refuses_reply_to_unauthenticated_peer(signed_pair):
    router, _ = signed_pair
    with pytest.raises(FrameAuthenticationError, match="No authenticated peer"):
        router.send([b"unknown-routing-id", b"reply"])


def test_signed_listener_drops_forged_messages_and_keeps_running(signed_pair):
    router, dealer = signed_pair
    received = []
    router.start_listener(received.append)
    unsigned = ZMQMessenger("DEALER", router.endpoint, signer=None)
    wrong_key = ZMQMessenger(
        "DEALER",
        router.endpoint,
        signer=_signer({1: _OTHER_KEY}),
        identity="tcp://127.0.0.1:40002",
    )
    try:
        unsigned.send([b"KV_AGENT_RESULT", b"forged"])
        wrong_key.send([b"KV_AGENT_RESULT", b"wrong key"])
        dealer.send([b"KV_AGENT_RESULT", b"genuine"])
        deadline = time.monotonic() + 5
        while (not received or router.auth_failures < 2) and time.monotonic() < deadline:
            time.sleep(0.01)
        listener_alive = router._listener_thread.is_alive()
    finally:
        unsigned.stop()
        wrong_key.stop()
        # Stop the listener inside the test body; thread-leak checks run
        # before fixture teardown.
        router.stop()
    assert [message[1:] for message in received] == [[b"KV_AGENT_RESULT", b"genuine"]]
    assert router.auth_failures == 2
    assert listener_alive


def test_signed_dealer_rejects_reply_for_other_recipient_or_from_other_peer(signed_pair):
    router, dealer = signed_pair
    signer = _signer()
    router_id = router.endpoint.encode()
    for_other = [b"reply"] + signer.sign(
        [b"reply"],
        sender_id=router_id,
        recipient_id=b"tcp://127.0.0.1:40003",
        flow=FrameSigner.FLOW_TO_DEALER,
    )
    with pytest.raises(FrameAuthenticationError, match="MAC mismatch"):
        dealer._verify(for_other)
    from_other = [b"reply"] + signer.sign(
        [b"reply"],
        sender_id=b"tcp://10.0.0.9:1",
        recipient_id=b"tcp://127.0.0.1:40001",
        flow=FrameSigner.FLOW_TO_DEALER,
    )
    with pytest.raises(FrameAuthenticationError, match="connected peer"):
        dealer._verify(from_other)


if __name__ == "__main__":
    unittest.main()
