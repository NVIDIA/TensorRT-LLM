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

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest

from tensorrt_llm._torch.disaggregation.native.replay import (
    BoundedExactReplayMap,
    BoundedExactReplaySet,
    ReplayCapacityError,
)
from tensorrt_llm._torch.disaggregation.native.transfer import (
    _KV_RESULT_PREFIX,
    _NON_DRAINED_TRANSFER_WORKERS,
    AgentResult,
    MessageType,
    Receiver,
    Sender,
    TerminalReceipt,
    TransferWorker,
    WriteMetaType,
    _make_aux_result_msg,
    _make_kv_result_msg,
)
from tensorrt_llm._torch.disaggregation.protocol import (
    AllocationWireIdentity,
    AttemptIdentity,
    EndpointIdentity,
    OperationIdentity,
    PublicationIdentity,
    TransferProtocolIdentity,
    decode_wire_identity,
    encode_wire_identity,
)


def _uuid(value: int) -> UUID:
    return UUID(int=value)


def _operation() -> OperationIdentity:
    return OperationIdentity(
        publication=PublicationIdentity(
            session=TransferProtocolIdentity(
                attempt=AttemptIdentity(
                    logical_request_id=17,
                    prefill_artifact_id=_uuid(2),
                    artifact_version=3,
                    handoff_attempt_uuid=_uuid(4),
                ),
                transfer_session_id=_uuid(5),
            ),
            source_endpoint=EndpointIdentity("ctx", 0, _uuid(6)),
            destination_endpoint=EndpointIdentity("gen", 1, _uuid(7)),
            destination_allocation=AllocationWireIdentity("gen-domain", 17, 12),
            operation_id=_uuid(9),
            slice_id=0,
            writer_rank=0,
        ),
        source_allocation=AllocationWireIdentity("ctx-domain", 17, 11),
    )


def _sender(dealer: SimpleNamespace) -> Sender:
    sender = object.__new__(Sender)
    sender._shutdown_complete = True  # Keep focused fixtures out of __del__ cleanup.
    peer_info = SimpleNamespace(
        instance_name="gen",
        instance_rank=1,
        endpoint_incarnation=_uuid(7),
        self_endpoint="tcp://gen",
    )
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(
            instance_name="ctx",
            instance_rank=0,
            endpoint_incarnation=_uuid(6),
        ),
        get_peer_rank_info=Mock(return_value=peer_info),
    )
    sender._instance_rank = 0
    sender._terminal_replay_outbox = BoundedExactReplayMap(8, _uuid(6))
    sender._terminal_replay_lock = threading.RLock()
    sender._terminal_replay_retry_lock = threading.Lock()
    sender._next_terminal_replay_at = 0.0
    sender._terminal_reconfirm_pending = False
    sender._get_or_connect_dealer = Mock(return_value=dealer)
    return sender


def _receiver(dealer: SimpleNamespace) -> Receiver:
    receiver = object.__new__(Receiver)
    receiver._shutdown = True  # Keep focused fixtures out of __del__ cleanup.
    receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(
            instance_name="gen",
            instance_rank=1,
            endpoint_incarnation=_uuid(7),
        )
    )
    receiver._terminal_receipts = BoundedExactReplayMap(8, _uuid(7))
    receiver._terminal_receipts_lock = threading.RLock()
    receiver._terminal_receipt_retry_lock = threading.Lock()
    receiver._next_terminal_receipt_retry_at = 0.0
    receiver._get_or_connect_dealer = Mock(return_value=dealer)
    receiver._sender_ep_instance_map = {}
    return receiver


def _terminal_receipt(operation: OperationIdentity) -> TerminalReceipt:
    terminal = (
        MessageType.KV_AGENT_RESULT,
        _KV_RESULT_PREFIX.pack(0, 17, 0, True, 0, 0),
    )
    return TerminalReceipt(
        result_identity=operation,
        channel=WriteMetaType.KV,
        unique_rid=17,
        writer_rank=0,
        slice_id=0,
        peer_endpoint="tcp://ctx",
        terminal_message=terminal,
    )


@pytest.mark.parametrize("channel", [WriteMetaType.KV, WriteMetaType.AUX])
def test_terminal_outbox_replays_in_order_until_exact_ack(channel) -> None:
    operation = _operation()
    dealer = SimpleNamespace(send=Mock())
    sender = _sender(dealer)
    submission = (
        MessageType.TRANSFER_SUBMITTED_V1,
        encode_wire_identity(operation),
    )
    terminal = (
        _make_kv_result_msg(
            0,
            17,
            0,
            True,
            AgentResult.SUCCESS,
            result_identity=operation,
        )
        if channel is WriteMetaType.KV
        else _make_aux_result_msg(
            0,
            17,
            AgentResult.SUCCESS,
            result_identity=operation,
        )
    )

    sender._retain_submission_replay("tcp://gen", submission)
    sender._retain_terminal_replay("tcp://gen", terminal)
    # Replay has no dependency on a TxSession or allocation owner.
    sender._sessions = {}
    assert sender.retry_terminal_replay(
        force=True,
        publication=operation.publication,
    )

    assert [call.args[0][0] for call in dealer.send.call_args_list] == [
        MessageType.TRANSFER_SUBMITTED_V1,
        terminal[0],
    ]
    assert len(sender._terminal_replay_outbox) == 1

    stale = replace(
        operation,
        source_allocation=replace(
            operation.source_allocation,
            allocation_generation=12,
        ),
    )
    sender._handle_terminal_result_ack(
        [
            MessageType.TERMINAL_RESULT_ACK_V1,
            encode_wire_identity(stale),
        ]
    )
    assert len(sender._terminal_replay_outbox) == 1

    exact_ack = [
        MessageType.TERMINAL_RESULT_ACK_V1,
        encode_wire_identity(operation),
    ]
    sender._handle_terminal_result_ack(exact_ack)
    assert len(sender._terminal_replay_outbox) == 0
    confirmation = dealer.send.call_args.args[0]
    assert confirmation[0] == MessageType.TERMINAL_RESULT_ACK_CONFIRM_V1
    assert decode_wire_identity(confirmation[1]) == operation
    # If the first confirmation was lost, an exact duplicate ACK replays it
    # statelessly from current endpoint-incarnation metadata.
    dealer.send.reset_mock()
    sender._handle_terminal_result_ack(exact_ack)
    duplicate_confirmation = dealer.send.call_args.args[0]
    assert duplicate_confirmation[0] == MessageType.TERMINAL_RESULT_ACK_CONFIRM_V1
    assert decode_wire_identity(duplicate_confirmation[1]) == operation


def test_receiver_reacks_exact_terminal_after_session_close() -> None:
    operation = _operation()
    dealer = SimpleNamespace(send=Mock())
    receiver = _receiver(dealer)
    receipt = _terminal_receipt(operation)
    terminal = receipt.terminal_message
    receiver._terminal_receipts.put_exact(operation, receipt)

    assert receiver._ack_recorded_terminal_replay(
        operation,
        channel=WriteMetaType.KV,
        unique_rid=17,
        writer_rank=0,
        slice_id=0,
        terminal_message=terminal,
    )
    ack = dealer.send.call_args.args[0]
    assert ack[0] == MessageType.TERMINAL_RESULT_ACK_V1
    assert decode_wire_identity(ack[1]) == operation

    dealer.send.reset_mock()
    assert receiver._ack_recorded_terminal_replay(
        operation,
        channel=WriteMetaType.KV,
        unique_rid=17,
        writer_rank=0,
        slice_id=0,
        terminal_message=(terminal[0], terminal[1] + b"conflict"),
    )
    dealer.send.assert_not_called()


def test_receiver_retires_only_an_exact_ack_confirmation() -> None:
    operation = _operation()
    dealer = SimpleNamespace(send=Mock())
    receiver = _receiver(dealer)
    receipt = _terminal_receipt(operation)
    receiver._terminal_receipts.put_exact(operation, receipt)

    stale_operation = replace(
        operation,
        source_allocation=replace(
            operation.source_allocation,
            allocation_generation=12,
        ),
    )
    receiver._handle_terminal_ack_confirmation(
        [
            MessageType.TERMINAL_RESULT_ACK_CONFIRM_V1,
            encode_wire_identity(stale_operation),
        ]
    )
    assert receiver._terminal_receipts.get(operation) is receipt

    stale_destination = replace(
        operation,
        publication=replace(
            operation.publication,
            destination_endpoint=replace(
                operation.publication.destination_endpoint,
                incarnation=_uuid(70),
            ),
        ),
    )
    receiver._handle_terminal_ack_confirmation(
        [
            MessageType.TERMINAL_RESULT_ACK_CONFIRM_V1,
            encode_wire_identity(stale_destination),
        ]
    )
    assert receiver._terminal_receipts.get(operation) is receipt

    receiver._handle_terminal_ack_confirmation(
        [
            MessageType.TERMINAL_RESULT_ACK_CONFIRM_V1,
            encode_wire_identity(operation),
        ]
    )
    assert len(receiver._terminal_receipts) == 0
    # Exact duplicates remain inert after retirement.
    receiver._handle_terminal_ack_confirmation(
        [
            MessageType.TERMINAL_RESULT_ACK_CONFIRM_V1,
            encode_wire_identity(operation),
        ]
    )


def _shutdown_receiver(dealer: SimpleNamespace) -> Receiver:
    receiver = _receiver(dealer)
    receiver._shutdown = False
    receiver._shutdown_attempt_lock = threading.Lock()
    receiver._shutdown_started = False
    receiver._listener_stopped = False
    receiver._control_only = False
    receiver._terminal_receipts_fenced = False
    receiver._dealer_admission_open = True
    receiver._dealers_lock = threading.Lock()
    receiver._dealers = {}
    receiver._messenger = Mock()
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = BoundedExactReplaySet(8, _uuid(7))
    receiver._recv_registry = Mock()
    receiver._recv_registry.is_drained.return_value = True
    receiver._bounce = Mock()
    receiver._bounce.retry_settlements.return_value = True
    receiver.begin_shutdown = Mock()
    return receiver


def _shutdown_sender(dealer: SimpleNamespace) -> Sender:
    sender = _sender(dealer)
    sender._shutdown = False
    sender._shutdown_complete = False
    sender._shutdown_attempt_lock = threading.Lock()
    sender._shutdown_sentinels_sent = False
    sender._listener_stopped = False
    sender._operation_admission_lock = threading.RLock()
    sender._dealer_admission_closed = False
    sender._control_only = False
    sender._data_plane_drained = False
    sender._terminal_replay_fenced = False
    sender._send_task_queues = []
    sender._worker_threads = []
    sender._failed_thread_dealers = []
    sender._failed_thread_dealers_lock = threading.Lock()
    sender._in_doubt_transfers = []
    sender._in_doubt_transfers_lock = threading.Lock()
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._pre_cancelled_rids = BoundedExactReplaySet(8, _uuid(6))
    sender._pre_session_terminal_results = BoundedExactReplayMap(8, _uuid(6))
    sender._pre_session_terminal_results_lock = threading.Lock()
    sender._pre_session_terminal_retry_lock = threading.Lock()
    sender._next_pre_session_terminal_retry_at = 0.0
    sender._loaded_remote_agents = {"gen-agent"}
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._agent = Mock()
    sender._dealers = {"tcp://gen": dealer}
    sender._dealers_lock = threading.Lock()
    sender._messenger = Mock()
    return sender


def test_receiver_shutdown_waits_for_terminal_ack_confirmation() -> None:
    operation = _operation()
    dealer = SimpleNamespace(send=Mock())
    receiver = _shutdown_receiver(dealer)
    receiver._terminal_receipts.put_exact(operation, _terminal_receipt(operation))

    assert receiver.transfers_drained is True
    assert receiver.control_drained is False
    assert receiver.shutdown() is False
    receiver._messenger.stop.assert_not_called()
    ack = dealer.send.call_args.args[0]
    assert ack[0] == MessageType.TERMINAL_RESULT_ACK_V1
    assert decode_wire_identity(ack[1]) == operation

    receiver._handle_terminal_ack_confirmation(
        [
            MessageType.TERMINAL_RESULT_ACK_CONFIRM_V1,
            encode_wire_identity(operation),
        ]
    )
    assert receiver.shutdown() is True
    receiver._messenger.stop.assert_called_once_with()


def test_global_fence_retires_unconfirmed_terminal_receipts() -> None:
    operation = _operation()
    receiver = _shutdown_receiver(SimpleNamespace(send=Mock()))
    receiver._terminal_receipts.put_exact(operation, _terminal_receipt(operation))
    receiver._recv_registry.mark_backend_quiesced.return_value = ()

    receiver.mark_backend_quiesced()

    assert receiver.control_drained is True
    assert len(receiver._terminal_receipts) == 0


def test_sender_global_fence_retires_unacknowledged_terminal_replay() -> None:
    operation = _operation()
    dealer = SimpleNamespace(send=Mock(), stop=Mock())
    sender = _shutdown_sender(dealer)
    terminal = _make_kv_result_msg(
        0,
        17,
        0,
        True,
        AgentResult.SUCCESS,
        result_identity=operation,
    )
    sender._retain_terminal_replay("tcp://gen", terminal)

    assert sender.shutdown() is False
    assert sender.data_plane_drained is True
    sender._agent.invalidate_remote_agent.assert_called_once_with("gen-agent")
    sender._messenger.stop.assert_not_called()
    assert len(sender._terminal_replay_outbox) == 1

    sender.mark_backend_quiesced()

    assert sender._control_only is True
    assert len(sender._terminal_replay_outbox) == 0
    assert sender.shutdown() is True
    sender._messenger.stop.assert_called_once_with()
    dealer.stop.assert_called_once_with()


def test_lost_confirmation_keeps_stateless_reconfirm_service_until_fence() -> None:
    operation = _operation()
    sender_dealer = SimpleNamespace(
        send=Mock(side_effect=RuntimeError("confirmation lost")),
        stop=Mock(),
    )
    sender = _shutdown_sender(sender_dealer)
    receiver_dealer = SimpleNamespace(send=Mock())
    receiver = _shutdown_receiver(receiver_dealer)
    receipt = _terminal_receipt(operation)
    receiver._terminal_receipts.put_exact(operation, receipt)
    sender._retain_terminal_replay(
        "tcp://gen",
        _make_kv_result_msg(
            0,
            17,
            0,
            True,
            AgentResult.SUCCESS,
            result_identity=operation,
        ),
    )

    assert receiver._send_terminal_result_ack(receipt)
    sender._handle_terminal_result_ack(receiver_dealer.send.call_args.args[0])
    assert len(sender._terminal_replay_outbox) == 0
    assert sender._terminal_reconfirm_pending is True

    worker = object.__new__(TransferWorker)
    worker._shutdown = False
    worker._agent_shutdown_failed = False
    worker._shutdown_started = False
    worker._session_admission_lock = threading.Lock()
    worker._rank_info_server = Mock()
    worker._rank_info_server.shutdown.return_value = True
    worker._sender = sender
    worker._receiver = receiver
    worker._bounce = Mock()
    worker._agent = sender._agent
    worker._registered_mem = [object()]

    try:
        # Receiver shutdown retries its exact ACK, while sender shutdown keeps
        # only stateless control ingress alive and releases physical owners.
        assert worker.shutdown() is False
        assert sender.data_plane_drained is True
        sender._messenger.stop.assert_not_called()
        sender_dealer.stop.assert_not_called()
        assert worker._agent is None
        assert worker._registered_mem == []

        retry_ack = receiver_dealer.send.call_args.args[0]
        sender_dealer.send.reset_mock(side_effect=True)
        sender._handle_terminal_result_ack(retry_ack)
        reconfirm = sender_dealer.send.call_args.args[0]
        assert reconfirm[0] == MessageType.TERMINAL_RESULT_ACK_CONFIRM_V1
        assert decode_wire_identity(reconfirm[1]) == operation
        receiver._handle_terminal_ack_confirmation(reconfirm)
        assert receiver.control_drained is True

        # Consuming a reconfirmation drains the receiver, but cannot prove the
        # last in-band message survived at the sender. The epoch-level sender
        # service therefore remains until the external fence.
        assert worker.shutdown() is False
        assert sender._terminal_reconfirm_pending is True
        worker.mark_backend_quiesced()
        assert sender._terminal_reconfirm_pending is False
        assert worker.shutdown() is True
        sender._messenger.stop.assert_called_once_with()
        sender_dealer.stop.assert_called_once_with()
    finally:
        _NON_DRAINED_TRANSFER_WORKERS.discard(worker)


def test_worker_releases_data_plane_while_terminal_control_is_pending() -> None:
    worker = object.__new__(TransferWorker)
    worker._shutdown = False
    worker._agent_shutdown_failed = False
    worker._shutdown_started = False
    worker._session_admission_lock = threading.Lock()
    worker._rank_info_server = Mock()
    worker._rank_info_server.shutdown.return_value = True
    worker._sender = Mock()
    worker._sender.shutdown.return_value = True
    worker._receiver = Mock()
    worker._receiver.transfers_drained = True
    worker._receiver.shutdown.side_effect = [False, True]
    worker._bounce = Mock()
    agent = Mock()
    worker._agent = agent
    descriptor = object()
    worker._registered_mem = [descriptor]

    try:
        assert worker.shutdown() is False
        worker._receiver.enter_control_only_shutdown.assert_called_once_with()
        worker._bounce.close.assert_called_once_with()
        agent.deregister_memory.assert_called_once_with(descriptor)
        agent.shutdown.assert_called_once_with()
        assert worker._agent is None
        assert worker._registered_mem == []

        assert worker.shutdown() is True
        worker._bounce.close.assert_called_once_with()
        assert worker._receiver.shutdown.call_count == 2
    finally:
        _NON_DRAINED_TRANSFER_WORKERS.discard(worker)


def test_exact_replay_capacity_is_fail_stop_until_epoch_rotation() -> None:
    replay_map = BoundedExactReplayMap[str, int](1, _uuid(1))
    replay_set = BoundedExactReplaySet[str](1, _uuid(1))
    replay_map.put_exact("first", 1)
    replay_set.add("first")

    with pytest.raises(ReplayCapacityError):
        replay_map.put_exact("second", 2)
    with pytest.raises(ReplayCapacityError):
        replay_set.add("second")

    assert not replay_map.rotate(_uuid(1))
    assert not replay_set.rotate(_uuid(1))
    replay_map.pop_exact("first", 1)
    with pytest.raises(ReplayCapacityError):
        replay_map.put_exact("second", 2)

    assert replay_map.rotate(_uuid(2))
    assert replay_set.rotate(_uuid(2))
    replay_map.put_exact("second", 2)
    replay_set.add("second")
    assert dict(replay_map.items_snapshot()) == {"second": 2}
    assert set(replay_set) == {"second"}
