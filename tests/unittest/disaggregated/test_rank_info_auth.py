# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from tensorrt_llm._torch.disaggregation.native.messenger import ZMQMessenger
from tensorrt_llm._torch.disaggregation.native.transfer import (
    _RANK_INFO_CHALLENGE_TTL_S,
    MessageType,
    RankInfoAuthenticationError,
    RankInfoServer,
    Receiver,
    TransferWorker,
    TransferWorkerConfig,
    _rank_info_auth_tag,
    _validate_rank_info_auth_key,
)
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only

_AUTH_KEY = b"k" * 32
_ENDPOINT = "tcp://127.0.0.1:1234"
_TRANSFER_MODULE = "tensorrt_llm._torch.disaggregation.native.transfer"


def _make_server(key=_AUTH_KEY) -> RankInfoServer:
    server = RankInfoServer.__new__(RankInfoServer)
    server._shutdown = True
    server._rank_info = Mock(to_bytes=Mock(return_value=b"rank-info"))
    server._rank_info_auth_key = key
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
    server._rank_info_auth_challenges[b"client"] = (
        nonce,
        time.monotonic() + _RANK_INFO_CHALLENGE_TTL_S,
    )
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


def test_rank_info_server_rejects_wrong_key_and_consumes_challenge():
    server = _make_server()
    nonce = b"n" * 32
    server._rank_info_auth_challenges[b"client"] = (
        nonce,
        time.monotonic() + _RANK_INFO_CHALLENGE_TTL_S,
    )
    request = [
        MessageType.REQUEST_INSTANCE_INFO,
        nonce,
        _rank_info_auth_tag(b"wrong-key" * 4, _ENDPOINT, nonce),
    ]

    server._handle_rank_info_request(b"client", request)

    server._messenger.send.assert_called_once_with([b"client", MessageType.AUTHENTICATION_ERROR])
    assert b"client" not in server._rank_info_auth_challenges
    server._rank_info.to_bytes.assert_not_called()


@pytest.mark.parametrize("key", [None, _AUTH_KEY])
def test_receiver_rank_info_exchange(key):
    receiver = Receiver.__new__(Receiver)
    receiver._rank_info_timeout_s = 60.0
    receiver._rank_info_auth_key = key
    receiver._incompatible_peers = {}
    receiver._sender_ep_instance_map = {}
    receiver._extract_info_endpoint = Mock(return_value=_ENDPOINT)
    receiver._should_register_peer = Mock(return_value=True)
    receiver._registrar = Mock()
    sender_info = Mock(sender_endpoints=[])
    messenger = Mock()
    messenger.receive.side_effect = (
        [[MessageType.AUTHENTICATION_CHALLENGE, b"n" * 32], [b"rank-info"]]
        if key is not None
        else [[b"rank-info"]]
    )

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
    expected_calls = [call([MessageType.REQUEST_INSTANCE_INFO])]
    if key is not None:
        expected_calls.append(call(authenticated_request))
    assert messenger.send.call_args_list == expected_calls
    messenger.stop.assert_called_once_with()


@pytest.mark.parametrize("key", [b"", "not-bytes"])
def test_transfer_worker_rejects_invalid_configured_key_before_native_setup(key):
    with pytest.raises(ValueError, match="internal_request_auth_key"):
        TransferWorker(
            TransferWorkerConfig(
                kv_cache_manager=Mock(), device_id=0, instance_name="test", rank_info_auth_key=key
            )
        )


@pytest.mark.parametrize("ownership", [False, True])
def test_authentication_failure_fails_only_affected_task(ownership):
    receiver = Receiver.__new__(Receiver)
    receiver._rank_info_timeout_s = 60.0
    receiver._build_recv_req_info = Mock()
    receiver._get_sender_info = Mock(side_effect=RankInfoAuthenticationError("wrong key"))
    receiver._enforce_physical_ownership = ownership
    session = Mock()
    receiver._get_session = Mock(return_value=session)
    task = Mock()
    receiver.dispatch_task(task)
    task.fail.assert_called_once()
    assert isinstance(task.fail.call_args.args[0], RankInfoAuthenticationError)
    if ownership:
        session.cancel_unpublished_task.assert_called_once_with(task)


def test_full_cache_preserves_live_challenges():
    server = _make_server()
    with patch(f"{_TRANSFER_MODULE}._RANK_INFO_CHALLENGE_CACHE_SIZE", 1):
        server._handle_rank_info_request(b"client", [MessageType.REQUEST_INSTANCE_INFO])
        pending = server._rank_info_auth_challenges[b"client"]
        server._messenger.reset_mock()
        server._handle_rank_info_request(b"other", [MessageType.REQUEST_INSTANCE_INFO])
        server._messenger.send.assert_called_once_with([b"other", MessageType.AUTHENTICATION_ERROR])
        # A retry by the original client reuses the nonce without refreshing its expiry.
        server._handle_rank_info_request(b"client", [MessageType.REQUEST_INSTANCE_INFO])
        assert list(server._rank_info_auth_challenges.items()) == [(b"client", pending)]
        nonce = pending[0]
        server._handle_rank_info_request(
            b"client",
            [
                MessageType.REQUEST_INSTANCE_INFO,
                nonce,
                _rank_info_auth_tag(_AUTH_KEY, _ENDPOINT, nonce),
            ],
        )
    server._messenger.send.assert_called_with([b"client", b"rank-info"])


@pytest.mark.parametrize("request_new_first", [False, True])
def test_expired_challenge_cannot_authenticate_and_frees_capacity(request_new_first):
    server = _make_server()
    with (
        patch(f"{_TRANSFER_MODULE}._RANK_INFO_CHALLENGE_CACHE_SIZE", 1),
        patch(f"{_TRANSFER_MODULE}.time.monotonic", return_value=0) as clock,
    ):
        server._handle_rank_info_request(b"old", [MessageType.REQUEST_INSTANCE_INFO])
        nonce = server._rank_info_auth_challenges[b"old"][0]
        clock.return_value = _RANK_INFO_CHALLENGE_TTL_S
        if request_new_first:
            server._handle_rank_info_request(b"new", [MessageType.REQUEST_INSTANCE_INFO])
            assert list(server._rank_info_auth_challenges) == [b"new"]
        server._handle_rank_info_request(
            b"old",
            [
                MessageType.REQUEST_INSTANCE_INFO,
                nonce,
                _rank_info_auth_tag(_AUTH_KEY, _ENDPOINT, nonce),
            ],
        )
    server._rank_info.to_bytes.assert_not_called()
    server._messenger.send.assert_called_with([b"old", MessageType.AUTHENTICATION_ERROR])
    if not request_new_first:
        server._handle_rank_info_request(b"new", [MessageType.REQUEST_INSTANCE_INFO])
        assert list(server._rank_info_auth_challenges) == [b"new"]


@pytest.mark.parametrize("extra_frames", [[b"nonce"], [b"nonce", b"tag", b"extra"]])
def test_malformed_followup_consumes_challenge(extra_frames):
    server = _make_server()
    server._handle_rank_info_request(b"client", [MessageType.REQUEST_INSTANCE_INFO])
    nonce = server._rank_info_auth_challenges[b"client"][0]
    server._handle_rank_info_request(b"client", [MessageType.REQUEST_INSTANCE_INFO, *extra_frames])
    assert b"client" not in server._rank_info_auth_challenges
    server._handle_rank_info_request(
        b"client",
        [
            MessageType.REQUEST_INSTANCE_INFO,
            nonce,
            _rank_info_auth_tag(_AUTH_KEY, _ENDPOINT, nonce),
        ],
    )
    server._rank_info.to_bytes.assert_not_called()
    server._messenger.send.assert_called_with([b"client", MessageType.AUTHENTICATION_ERROR])


def test_receiver_reports_full_cache_as_busy_not_version_mismatch():
    receiver = Receiver.__new__(Receiver)
    receiver._rank_info_timeout_s = 60.0
    receiver._incompatible_peers = {}
    receiver._extract_info_endpoint = Mock(return_value=_ENDPOINT)
    receiver._should_register_peer = Mock(return_value=True)
    messenger = Mock()
    messenger.receive.return_value = [MessageType.AUTHENTICATION_ERROR]
    with patch(f"{_TRANSFER_MODULE}.ZMQMessenger", return_value=messenger):
        with pytest.raises(RankInfoAuthenticationError, match="server is busy; retry"):
            receiver._get_sender_info(Mock())
    messenger.send.assert_called_once_with([MessageType.REQUEST_INSTANCE_INFO])
    messenger.stop.assert_called_once()
    assert not receiver._incompatible_peers


@pytest.mark.parametrize("key", [None, _AUTH_KEY])
def test_server_warns_only_when_authentication_is_disabled(key):
    with (
        patch(f"{_TRANSFER_MODULE}.ZMQMessenger"),
        patch(f"{_TRANSFER_MODULE}.logger.warning") as warning,
    ):
        server = RankInfoServer(Mock(), addr="127.0.0.1", port=1234, rank_info_auth_key=key)
        if key is None:
            warning.assert_called_once()
            assert "without authentication" in warning.call_args.args[0]
        else:
            warning.assert_not_called()
        assert _validate_rank_info_auth_key(key) == key
        server.shutdown()


def test_server_without_key_returns_rank_info_without_challenge():
    server = _make_server(key=None)
    server._handle_rank_info_request(b"client", [MessageType.REQUEST_INSTANCE_INFO])
    server._messenger.send.assert_called_once_with([b"client", b"rank-info"])
    assert not server._rank_info_auth_challenges


@pytest.mark.parametrize(
    "key,reply",
    [
        (None, [MessageType.AUTHENTICATION_CHALLENGE, b"n" * 32]),
        (_AUTH_KEY, [b"unauthenticated-rank-info"]),
    ],
)
def test_receiver_rejects_mismatched_authentication_modes(key, reply):
    receiver = Receiver.__new__(Receiver)
    receiver._rank_info_timeout_s = 60.0
    receiver._rank_info_auth_key = key
    receiver._incompatible_peers = {}
    receiver._extract_info_endpoint = Mock(return_value=_ENDPOINT)
    receiver._should_register_peer = Mock(return_value=True)
    messenger = Mock()
    messenger.receive.return_value = reply
    with (
        patch(f"{_TRANSFER_MODULE}.ZMQMessenger", return_value=messenger),
        patch(f"{_TRANSFER_MODULE}.RankInfo.from_bytes") as decode,
    ):
        with pytest.raises(RankInfoAuthenticationError, match="internal_request_auth_key"):
            receiver._get_sender_info(Mock())
        decode.assert_not_called()
    messenger.send.assert_called_once_with([MessageType.REQUEST_INSTANCE_INFO])
    messenger.stop.assert_called_once()


@pytest.mark.parametrize("rejected_rank", [None, 0, 1])
@pytest.mark.parametrize("error_type", [RankInfoAuthenticationError, ValueError])
def test_rank_info_admission_waits_for_asymmetric_receive_fences(rejected_rank, error_type):
    barrier = threading.Barrier(2, timeout=5)
    payloads = [None, None]
    histories = [[], []]
    coordinators, transceivers, requests, fences = [], [], [], []

    def gather(rank, payload):
        histories[rank].append(type(payload))
        payloads[rank] = payload
        barrier.wait()
        result = list(payloads)
        barrier.wait()
        return result

    for rank in range(2):
        tc = object.__new__(KvCacheTransceiverV2)
        tc._gen_need_sync = True
        tc._gen_allgather = lambda payload, rank=rank: gather(rank, payload)
        tc._mapping = SimpleNamespace(world_size=2, pp_size=1, enable_attention_dp=False)
        tc._pending_rank_info_admissions = {}
        tc._recv_sessions, tc._recv_reqs = {}, {}
        tc._ever_had_recv_session = False
        tc._validate_bridge_req = Mock(return_value=True)
        tc._start_async_receive = Mock()
        tc._transfer_worker = Mock()
        if rank == rejected_rank:
            tc._transfer_worker.prepare_rank_info.side_effect = error_type("peer rejected")
        req = SimpleNamespace(
            py_request_id=7,
            request_id=7,
            py_disaggregated_params=None,
            state=LlmRequestState.DISAGG_GENERATION_INIT,
            set_kv_cache_transfer_start=Mock(),
        )
        fence = Mock(query=Mock(return_value=rank == 0))
        coordinator = object.__new__(DisaggTransferCoordinator)
        coordinator._transceiver = tc
        coordinator._defer_unready_gen_receives = True
        coordinator._unpublished_gen_receives = {7: (req, fence)}
        coordinator._stamp_gen_transfer_start = Mock()
        coordinator.inflight_cancel_active = Mock(return_value=False)
        coordinator._check_transfer_errors = Mock()
        coordinators.append(coordinator)
        transceivers.append(tc)
        requests.append(req)
        fences.append(fence)

    def poll_both():
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(c.poll_gen_transfers) for c in coordinators]
            for future in futures:
                future.result(timeout=10)

    with patch(
        "tensorrt_llm._torch.disaggregation.orchestration.coordinator.uses_async_gen_transfer",
        return_value=True,
    ):
        poll_both()
        for tc in transceivers:
            tc._start_async_receive.assert_not_called()
        transceivers[0]._transfer_worker.prepare_rank_info.assert_called_once()
        transceivers[1]._transfer_worker.prepare_rank_info.assert_not_called()
        # A duplicate local dispatch also must not enter a collective or redo admission.
        transceivers[0].request_and_receive_async(requests[0])
        transceivers[0]._transfer_worker.prepare_rank_info.assert_called_once()
        assert histories == [[tuple, list], [tuple, list]]

        fences[1].query.return_value = True
        poll_both()

    assert histories == [[tuple, list] * 2, [tuple, list] * 2]
    for tc, req in zip(transceivers, requests):
        assert not tc._pending_rank_info_admissions
        if rejected_rank is None:
            tc._start_async_receive.assert_called_once_with(req)
        else:
            tc._start_async_receive.assert_not_called()
            assert req.state == LlmRequestState.DISAGG_TRANS_ERROR


def test_cancel_pending_rank_info_admission_does_not_publish():
    tc = object.__new__(KvCacheTransceiverV2)
    req = SimpleNamespace(py_request_id=7, request_id=7, py_disaggregated_params=None)
    tc._pending_rank_info_admissions = {7: (req, None)}
    tc._wait_reqs, tc._send_sessions, tc._recv_sessions = {}, {}, {}
    tc._gen_allgather = Mock(return_value=[([], {}), ([], {7: None})])
    tc._start_async_receive = Mock()
    assert tc.cancel_request(req)
    assert tc._poll_rank_info_admissions([]) == ([], [])
    tc._start_async_receive.assert_not_called()


@pytest.mark.parametrize("after_challenge", [False, True])
def test_rank_info_timeout_uses_one_deadline_and_closes_socket(after_challenge):
    receiver = Receiver.__new__(Receiver)
    receiver._rank_info_timeout_s = 1.0
    receiver._rank_info_auth_key = _AUTH_KEY
    receiver._incompatible_peers = {}
    receiver._extract_info_endpoint = Mock(return_value=_ENDPOINT)
    receiver._should_register_peer = Mock(return_value=True)
    messenger = Mock()
    messenger.receive.side_effect = (
        [[MessageType.AUTHENTICATION_CHALLENGE, b"n" * 32], TimeoutError()]
        if after_challenge
        else [TimeoutError()]
    )
    with (
        patch(f"{_TRANSFER_MODULE}.ZMQMessenger", return_value=messenger),
        patch(f"{_TRANSFER_MODULE}.time.monotonic", side_effect=[100.0, 100.0, 100.25]),
        patch(f"{_TRANSFER_MODULE}.RankInfo.from_bytes") as decode,
    ):
        with pytest.raises(RankInfoAuthenticationError, match="timed out"):
            receiver._get_sender_info(Mock())
        decode.assert_not_called()
    expected = [call(timeout_ms=1000)]
    if after_challenge:
        expected.append(call(timeout_ms=750))
    assert messenger.receive.call_args_list == expected
    messenger.stop.assert_called_once_with()
    assert not receiver._incompatible_peers


def test_messenger_receive_timeout_allows_subsequent_reply():
    with ZMQMessenger("ROUTER", "tcp://127.0.0.1:*") as server:
        with ZMQMessenger("DEALER", server.endpoint) as client:
            with pytest.raises(TimeoutError):
                client.receive(timeout_ms=10)
            client.send([b"request"])
            identity, request = server.receive(timeout_ms=1000)
            assert request == b"request"
            server.send([identity, b"reply"])
            assert client.receive(timeout_ms=1000) == [b"reply"]


@pytest.mark.parametrize("rejected_rank", [None, 0, 1])
@pytest.mark.parametrize("reason", ["server busy", "handshake timed out"])
def test_sync_rank_info_admission_rejects_all_ranks_before_publication(rejected_rank, reason):
    from tensorrt_llm._torch.disaggregation.base.transfer import WaitResult

    barrier = threading.Barrier(2, timeout=5)
    votes = [None, None]
    admitted = [False, False]
    transceivers, requests = [], []

    def gather(rank, error):
        votes[rank] = error
        admitted[rank] = True
        barrier.wait()
        return list(votes)

    for rank in range(2):
        tc = object.__new__(KvCacheTransceiverV2)
        tc._pending_rank_info_admissions = {}
        tc._gen_allgather = Mock(side_effect=lambda error, rank=rank: gather(rank, error))
        tc._transfer_worker = Mock()
        if rank == rejected_rank:
            tc._transfer_worker.prepare_rank_info.side_effect = RankInfoAuthenticationError(reason)
        tc._validate_bridge_req = Mock(return_value=True)
        tc._recv_sessions, tc._recv_reqs = {}, {}
        tc._create_cache_extent = Mock()
        fetches = Mock()
        fetches.session.wait_complete.return_value = WaitResult.COMPLETED
        tc._open_peer_source = Mock(return_value=fetches)
        tc._chunk_num_bytes = Mock(return_value=8)
        tc._kv_size_rank_factor = 1
        tc._need_aux_transfer = Mock(return_value=False)
        tc._assert_disagg_history_declared = Mock()

        def publish(_extent):
            assert all(admitted)
            assert rejected_rank is None

        fetches.fetch.side_effect = publish
        req = SimpleNamespace(
            py_request_id=7,
            request_id=7,
            py_disaggregated_params=None,
            state=LlmRequestState.DISAGG_GENERATION_INIT,
            set_kv_cache_size=Mock(),
        )
        transceivers.append(tc)
        requests.append(req)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(tc.request_and_receive_sync, req) for tc, req in zip(transceivers, requests)
        ]
        for future in futures:
            future.result(timeout=10)

    for tc, req in zip(transceivers, requests):
        tc._gen_allgather.assert_called_once()
        if rejected_rank is None:
            tc._open_peer_source.return_value.fetch.assert_called_once()
            tc._open_peer_source.return_value.session.close.assert_called_once()
            assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        else:
            tc._create_cache_extent.assert_not_called()
            tc._open_peer_source.assert_not_called()
            assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
        assert not tc._recv_sessions
        assert not tc._recv_reqs
