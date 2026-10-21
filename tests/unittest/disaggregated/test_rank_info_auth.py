# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from unittest.mock import Mock, call, patch

import pytest

from tensorrt_llm._torch.disaggregation.native.transfer import (
    MessageType,
    RankInfoServer,
    Receiver,
    _rank_info_auth_tag,
)

pytestmark = pytest.mark.cpu_only

_AUTH_KEY = b"k" * 32
_ENDPOINT = "tcp://127.0.0.1:1234"
_TRANSFER_MODULE = "tensorrt_llm._torch.disaggregation.native.transfer"


def _make_server() -> RankInfoServer:
    server = RankInfoServer.__new__(RankInfoServer)
    server._rank_info = Mock(to_bytes=Mock(return_value=b"rank-info"))
    server._rank_info_auth_key = _AUTH_KEY
    server._rank_info_auth_challenges = OrderedDict()
    server._messenger = Mock(endpoint=_ENDPOINT)
    return server


def test_rank_info_server_challenges_unauthenticated_request():
    server = _make_server()

    with patch(f"{_TRANSFER_MODULE}.secrets.token_bytes", return_value=b"n" * 32):
        server._handle_rank_info_request(b"client", [MessageType.REQUEST_INSTANCE_INFO])

    server._rank_info.to_bytes.assert_not_called()
    server._messenger.send.assert_called_once_with(
        [b"client", MessageType.AUTHENTICATION_CHALLENGE, b"n" * 32]
    )


def test_rank_info_server_accepts_once_with_valid_tag():
    server = _make_server()
    nonce = b"n" * 32
    server._rank_info_auth_challenges[b"client"] = nonce
    request = [
        MessageType.REQUEST_INSTANCE_INFO,
        nonce,
        _rank_info_auth_tag(_AUTH_KEY, _ENDPOINT, nonce),
    ]

    server._handle_rank_info_request(b"client", request)
    server._messenger.send.assert_called_once_with([b"client", b"rank-info"])

    server._messenger.reset_mock()
    server._handle_rank_info_request(b"client", request)
    server._messenger.send.assert_called_once_with([b"client", MessageType.AUTHENTICATION_ERROR])


def test_receiver_authenticates_rank_info_request():
    receiver = Receiver.__new__(Receiver)
    receiver._rank_info_auth_key = _AUTH_KEY
    receiver._incompatible_peers = {}
    receiver._sender_ep_instance_map = {}
    receiver._extract_info_endpoint = Mock(return_value=_ENDPOINT)
    receiver._should_register_peer = Mock(return_value=True)
    receiver._registrar = Mock()
    sender_info = Mock(sender_endpoints=[])
    messenger = Mock()
    messenger.receive.side_effect = [
        [MessageType.AUTHENTICATION_CHALLENGE, b"n" * 32],
        [b"rank-info"],
    ]

    with (
        patch(f"{_TRANSFER_MODULE}.ZMQMessenger", return_value=messenger),
        patch(f"{_TRANSFER_MODULE}.RankInfo.from_bytes", return_value=sender_info),
        patch(f"{_TRANSFER_MODULE}.MambaPolicy.validate_peer_compatible"),
        patch(f"{_TRANSFER_MODULE}.AttentionPolicy.validate_peer_compatible"),
    ):
        assert receiver._get_sender_info(Mock()) is sender_info

    authenticated_request = [
        MessageType.REQUEST_INSTANCE_INFO,
        b"n" * 32,
        _rank_info_auth_tag(_AUTH_KEY, _ENDPOINT, b"n" * 32),
    ]
    assert messenger.send.call_args_list == [
        call([MessageType.REQUEST_INSTANCE_INFO]),
        call(authenticated_request),
    ]
    messenger.stop.assert_called_once_with()
