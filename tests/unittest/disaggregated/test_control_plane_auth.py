# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-plane frame authentication wired through the native transfer layer."""

import threading
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.disaggregation.native.messenger import FrameSigner, ZMQMessenger
from tensorrt_llm._torch.disaggregation.native.rank_info import NATIVE_TRANSFER_PROTOCOL_VERSION
from tensorrt_llm._torch.disaggregation.native.transfer import (
    MessageType,
    RankInfoAuthenticationError,
    RankInfoServer,
    Receiver,
    _derive_native_auth_key,
    _make_native_frame_signer,
)

pytestmark = pytest.mark.cpu_only

_MASTER_KEY = b"m" * 32
_OTHER_MASTER_KEY = b"o" * 32
_TRANSFER_MODULE = "tensorrt_llm._torch.disaggregation.native.transfer"
_RECEIVER_ENDPOINT = "tcp://127.0.0.1:40100"


def test_derived_keys_are_distinct_per_use():
    rank_info = _derive_native_auth_key(_MASTER_KEY, b"rank-info")
    control = _derive_native_auth_key(_MASTER_KEY, b"control-plane")
    assert len({_MASTER_KEY, rank_info, control}) == 3
    assert rank_info != _derive_native_auth_key(_OTHER_MASTER_KEY, b"rank-info")


def test_frame_signer_only_when_key_configured():
    assert _make_native_frame_signer(None) is None
    signer = _make_native_frame_signer(_MASTER_KEY)
    message = [b"KV_AGENT_RESULT"] + signer.sign(
        [b"KV_AGENT_RESULT"], sender_id=b"a", recipient_id=b"b", flow=FrameSigner.FLOW_TO_ROUTER
    )
    signer.verify(message, recipient_id=b"b", flow=FrameSigner.FLOW_TO_ROUTER)
    with pytest.raises(PermissionError):
        _make_native_frame_signer(_OTHER_MASTER_KEY).verify(
            message, recipient_id=b"b", flow=FrameSigner.FLOW_TO_ROUTER
        )


def _make_receiver(master_key, timeout_s=5.0) -> Receiver:
    receiver = Receiver.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._shutdown = True
    receiver._rank_info_timeout_s = timeout_s
    receiver._rank_info_auth_key = (
        _derive_native_auth_key(master_key, b"rank-info") if master_key else None
    )
    receiver._frame_signer = _make_native_frame_signer(master_key)
    receiver._messenger = Mock(endpoint=_RECEIVER_ENDPOINT)
    receiver._incompatible_peers = {}
    receiver._sender_ep_instance_map = {}
    receiver._should_register_peer = Mock(return_value=True)
    receiver._registrar = Mock()
    return receiver


@contextmanager
def _rank_info_server(master_key):
    # Shut down inside the test body: thread-leak checks run before fixture teardown.
    rank_info = Mock(to_bytes=Mock(return_value=b"rank-info"))
    with patch(f"{_TRANSFER_MODULE}.get_local_ip", return_value="127.0.0.1"):
        server = RankInfoServer(
            rank_info,
            rank_info_auth_key=(
                _derive_native_auth_key(master_key, b"rank-info") if master_key else None
            ),
            frame_signer=_make_native_frame_signer(master_key),
        )
    try:
        yield server
    finally:
        server.shutdown()


def _fetch(receiver, server):
    receiver._extract_info_endpoint = Mock(return_value=server.endpoint)
    sender_info = Mock(sender_endpoints=[], protocol_version=NATIVE_TRANSFER_PROTOCOL_VERSION)
    with (
        patch(f"{_TRANSFER_MODULE}.RankInfo.from_bytes", return_value=sender_info) as decode,
        patch(f"{_TRANSFER_MODULE}.MambaPolicy.validate_peer_compatible"),
        patch(f"{_TRANSFER_MODULE}.AttentionPolicy.validate_peer_compatible"),
    ):
        assert receiver._get_sender_info(Mock()) is sender_info
    decode.assert_called_once_with(b"rank-info")


@pytest.mark.parametrize("master_key", [None, _MASTER_KEY])
def test_rank_info_exchange_over_real_sockets(master_key):
    with _rank_info_server(master_key) as server:
        _fetch(_make_receiver(master_key), server)


@pytest.mark.parametrize(
    "receiver_key, server_key, match",
    [
        # The signed server drops the frames; the receiver times out.
        (_OTHER_MASTER_KEY, _MASTER_KEY, "internal_request_auth_key"),
        (None, _MASTER_KEY, "internal_request_auth_key"),
        # The unsigned reply fails verification on the signed receiver.
        (_MASTER_KEY, None, "failed control-plane authentication"),
    ],
    ids=["wrong_key", "receiver_unsigned", "server_unsigned"],
)
def test_rank_info_exchange_rejects_key_mismatch(receiver_key, server_key, match):
    receiver = _make_receiver(receiver_key, timeout_s=0.5)
    with _rank_info_server(server_key) as server:
        receiver._extract_info_endpoint = Mock(return_value=server.endpoint)
        with pytest.raises(RankInfoAuthenticationError, match=match):
            receiver._get_sender_info(Mock())
    assert not receiver._incompatible_peers


def test_signed_server_drops_forged_rank_info_request():
    with _rank_info_server(_MASTER_KEY) as server:
        with ZMQMessenger("DEALER", server.endpoint, signer=None) as forger:
            forger.send([MessageType.REQUEST_INSTANCE_INFO])
            with pytest.raises(TimeoutError):
                forger.receive(timeout_ms=300)
        auth_failures = server._messenger.auth_failures
    assert auth_failures == 1
    server._rank_info.to_bytes.assert_not_called()
