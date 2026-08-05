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
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import msgpack
import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxAllocationIdentity
from tensorrt_llm._torch.disaggregation.native.receive_lifecycle import WriterMode, WriterResult
from tensorrt_llm._torch.disaggregation.native.transfer import (
    _KV_RESULT_PREFIX,
    AgentResult,
    AuxSendTask,
    KVSendTask,
    MessageType,
    Receiver,
    RecvReqInfo,
    RxSession,
    Sender,
    TaskStatus,
    TxSession,
    WriteMeta,
    WriteMetaType,
    _make_kv_result_msg,
)
from tensorrt_llm._torch.disaggregation.protocol import (
    AllocationWireIdentity,
    AttemptIdentity,
    EndpointIdentity,
    OperationIdentity,
    ProtocolIdentityError,
    PublicationIdentity,
    QualifiedLegacyIdentity,
    StaleProtocolMessageError,
    TransferSessionIdentity,
    decode_wire_identity,
    encode_wire_identity,
    require_exact_identity,
    transfer_protocol_identity_from_params,
)
from tensorrt_llm.disaggregated_params import DisaggregatedParams


def _uuid(value: int) -> UUID:
    return UUID(int=value)


def _operation(*, attempt: int = 4, session: int = 5, generation: int = 11):
    return OperationIdentity(
        publication=PublicationIdentity(
            session=TransferSessionIdentity(
                attempt=AttemptIdentity(
                    logical_request_id=17,
                    prefill_artifact_id=_uuid(2),
                    artifact_version=3,
                    handoff_attempt_uuid=_uuid(attempt),
                ),
                transfer_session_id=_uuid(session),
            ),
            source_endpoint=EndpointIdentity("ctx", 0, _uuid(6)),
            destination_endpoint=EndpointIdentity("gen", 1, _uuid(7)),
            destination_allocation=AllocationWireIdentity("42", 17, generation + 1),
            operation_id=_uuid(9),
            slice_id=0,
            writer_rank=0,
        ),
        source_allocation=AllocationWireIdentity("ctx-domain", 17, generation),
    )


def test_generation_safe_identity_roundtrip_is_exact_and_immutable() -> None:
    identity = _operation()

    restored = decode_wire_identity(encode_wire_identity(identity))

    assert restored == identity
    with pytest.raises(FrozenInstanceError):
        restored.publication.slice_id = 1


def test_recv_req_info_roundtrip_carries_exact_publication_identity() -> None:
    publication = _operation().publication
    info = RecvReqInfo(
        sender_req_id=17,
        instance_name="gen",
        instance_rank=1,
        block_ids_per_layer_groups=[np.asarray([3, 4], dtype=np.int64)],
        unique_rid=17,
        slice_id=0,
        publication_identity=publication,
    )

    restored = RecvReqInfo.from_bytes(info.to_bytes())

    assert restored.publication_identity == publication
    np.testing.assert_array_equal(
        restored.block_ids_per_layer_groups[0],
        np.asarray([3, 4], dtype=np.int64),
    )


def test_protocol_v1_kv_result_keeps_identity_separate_from_binary_prefix() -> None:
    identity = _operation()

    message = _make_kv_result_msg(
        0,
        17,
        0,
        True,
        AgentResult.SUCCESS,
        transfer_size=4096,
        result_identity=identity,
    )

    assert message[0] == MessageType.KV_AGENT_RESULT_V1
    assert decode_wire_identity(message[1]) == identity
    assert _KV_RESULT_PREFIX.unpack(message[2]) == (0, 17, 0, True, 0, 4096)


def test_destination_rejects_changed_source_generation_before_ledger_mutation() -> None:
    expected = _operation(generation=11)
    stale = _operation(generation=12)
    session = SimpleNamespace(
        lock=threading.Lock(),
        protocol_identity=expected.publication.session,
        _publication_identities={(0, 0): expected.publication},
        _result_identities={(0, 0): expected},
    )

    with pytest.raises(StaleProtocolMessageError, match="source allocation"):
        RxSession.validate_result_identity(session, 0, 0, stale)

    assert session._result_identities == {(0, 0): expected}


def test_destination_requires_authenticated_submission_before_operation_result() -> None:
    operation = _operation()
    session = SimpleNamespace(
        lock=threading.Lock(),
        protocol_identity=operation.publication.session,
        _publication_identities={(0, 0): operation.publication},
        _submitted_identities={},
        _active_submission_times={},
        _result_identities={},
        transfer_start_time=None,
        _transfer_submitted_at_s=None,
    )

    with pytest.raises(StaleProtocolMessageError, match="authenticated backend submission"):
        RxSession.validate_result_identity(session, 0, 0, operation)

    assert RxSession.record_transfer_submitted(
        session,
        0,
        0,
        operation,
        transfer_start_time=100,
        monotonic_start_s=10.0,
    )
    RxSession.validate_result_identity(session, 0, 0, operation)
    RxSession.record_result_identity(session, 0, 0, operation)

    assert session._active_submission_times == {}


def test_destination_submission_is_exact_idempotent_and_stamps_once() -> None:
    operation = _operation()
    destination_owner = SimpleNamespace(
        py_kv_transfer_start_time=None,
        set_kv_cache_transfer_start=Mock(),
    )
    changed_source = replace(
        operation,
        source_allocation=replace(
            operation.source_allocation,
            allocation_generation=operation.source_allocation.allocation_generation + 1,
        ),
    )
    session = SimpleNamespace(
        lock=threading.Lock(),
        protocol_identity=operation.publication.session,
        _publication_identities={(0, 0): operation.publication},
        _submitted_identities={},
        _active_submission_times={},
        _result_identities={},
        transfer_start_time=None,
        _transfer_submitted_at_s=None,
        _destination_owner=destination_owner,
    )

    assert RxSession.record_transfer_submitted(
        session,
        0,
        0,
        operation,
        transfer_start_time=100,
        monotonic_start_s=10.0,
    )
    assert not RxSession.record_transfer_submitted(
        session,
        0,
        0,
        operation,
        transfer_start_time=200,
        monotonic_start_s=20.0,
    )
    with pytest.raises(StaleProtocolMessageError, match="source allocation generation"):
        RxSession.record_transfer_submitted(
            session,
            0,
            0,
            changed_source,
            transfer_start_time=300,
            monotonic_start_s=30.0,
        )

    assert session._submitted_identities == {(0, 0): operation}
    assert session._active_submission_times == {(0, 0): 10.0}
    assert session.transfer_start_time == 100
    assert session._transfer_submitted_at_s == 10.0
    assert destination_owner.py_kv_transfer_start_time == 10.0
    destination_owner.set_kv_cache_transfer_start.assert_called_once_with(100)


def test_source_submission_is_exact_idempotent_and_stamps_once() -> None:
    operation = _operation()
    source_owner = SimpleNamespace(
        py_kv_transfer_start_time=None,
        set_kv_cache_transfer_start=Mock(),
    )
    changed_source = replace(
        operation,
        source_allocation=replace(
            operation.source_allocation,
            allocation_generation=operation.source_allocation.allocation_generation + 1,
        ),
    )
    session = SimpleNamespace(
        lock=threading.Lock(),
        protocol_identity=operation.publication.session,
        _submitted_identities={},
        _active_submission_times={},
        transfer_start_time=None,
        _transfer_submitted_at_s=None,
        _source_owner=source_owner,
    )

    assert TxSession.record_transfer_submitted(
        session,
        operation,
        transfer_start_time=100,
        monotonic_start_s=10.0,
    )
    assert not TxSession.record_transfer_submitted(
        session,
        operation,
        transfer_start_time=200,
        monotonic_start_s=20.0,
    )
    with pytest.raises(StaleProtocolMessageError, match="source allocation generation"):
        TxSession.record_transfer_submitted(
            session,
            changed_source,
            transfer_start_time=300,
            monotonic_start_s=30.0,
        )

    assert session._submitted_identities == {operation.publication: operation}
    assert session._active_submission_times == {operation.publication: 10.0}
    assert session.transfer_start_time == 100
    assert session._transfer_submitted_at_s == 10.0
    assert source_owner.py_kv_transfer_start_time == 10.0
    source_owner.set_kv_cache_transfer_start.assert_called_once_with(100)
    TxSession.record_transfer_terminal(session, operation)
    assert session._active_submission_times == {}


def test_generation_safe_source_timeout_waits_for_submission_boundary(
    monkeypatch,
) -> None:
    session = SimpleNamespace(
        protocol_identity=_operation().publication.session,
        _timeout_s=0.5,
        lock=threading.Lock(),
        _transfer_submitted_at_s=None,
        _active_submission_times={},
    )
    wait_count = 0

    def wait(*, timeout):
        nonlocal wait_count
        assert timeout == 0.01
        wait_count += 1
        if wait_count == 2:
            session._active_submission_times[_operation().publication] = 1.0
        return False

    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        lambda: 2.0,
    )

    assert not TxSession._wait_task_with_transfer_timeout(
        session,
        SimpleNamespace(wait=wait, slice_id=0),
    )
    assert wait_count == 2


def test_sender_announces_exact_operation_after_submission_acceptance() -> None:
    operation = _operation()
    dealer = SimpleNamespace(send=Mock())
    task = SimpleNamespace(
        cache_submission_message=Mock(),
        mark_operation_submission_delivered=Mock(),
    )
    session = SimpleNamespace(
        protocol_identity=operation.publication.session,
        record_transfer_submitted=Mock(),
    )
    sender = SimpleNamespace(
        _registrar=SimpleNamespace(
            self_rank_info=SimpleNamespace(
                instance_name="ctx",
                instance_rank=0,
                endpoint_incarnation=_uuid(6),
            ),
            get_peer_rank_info=Mock(
                return_value=SimpleNamespace(
                    instance_name="gen",
                    instance_rank=1,
                    endpoint_incarnation=_uuid(7),
                )
            ),
        ),
        _get_or_connect_thread_dealer=Mock(return_value=dealer),
    )
    write_meta = SimpleNamespace(
        result_identity=operation,
        session=session,
        peer_endpoint="tcp://gen",
        peer_rank=1,
        unique_rid=17,
        operation_key=("gen", 1),
        task=task,
    )

    assert Sender._announce_transfer_submitted(sender, write_meta)

    source_stamp = session.record_transfer_submitted.call_args
    assert source_stamp.args == (operation,)
    assert source_stamp.kwargs["monotonic_start_s"] > 0
    task.cache_submission_message.assert_called_once()
    task.mark_operation_submission_delivered.assert_called_once_with(("gen", 1))
    dealer.send.assert_called_once()
    frame = dealer.send.call_args.args[0]
    assert frame[0] == MessageType.TRANSFER_SUBMITTED_V1
    assert decode_wire_identity(frame[1]) == operation


@pytest.mark.parametrize("channel", ["kv", "aux"])
@pytest.mark.parametrize("backend_succeeded", [True, False])
def test_submission_send_failure_waits_for_backend_terminal_and_retries_in_order(
    monkeypatch,
    channel,
    backend_succeeded,
) -> None:
    operation = _operation()
    params = DisaggregatedParams(
        disagg_request_id=17,
        ctx_request_id=17,
        ctx_dp_rank=0,
    )
    session = SimpleNamespace(
        protocol_identity=operation.publication.session,
        lock=threading.Lock(),
        status=SessionStatus.READY,
        source_owner=object(),
        kv_tasks=[],
        aux_task=None,
        _aux_buffer=None,
        disagg_request_id=17,
        _unbound_terminal_results={},
        _terminal_retry_lock=threading.Lock(),
        _next_terminal_retry_at=0.0,
        record_transfer_submitted=Mock(),
        record_transfer_terminal=Mock(),
        mark_transfer_end_if_complete=Mock(),
        set_exception=Mock(),
    )
    if channel == "kv":
        task = KVSendTask(KVSlice(), params, 0, session=session)
        session.kv_tasks.append(task)
        meta_type = WriteMetaType.KV
        slice_id = 0
    else:
        aux_identity = AuxAllocationIdentity(_uuid(21), 17, 11)
        operation = replace(
            operation,
            source_allocation=AllocationWireIdentity.from_local(aux_identity),
        )
        session._aux_buffer = SimpleNamespace(
            allocation_identity=Mock(return_value=aux_identity),
        )
        task = AuxSendTask(
            params,
            7,
            session=session,
            allocation_identity=aux_identity,
        )
        session.aux_task = task
        meta_type = WriteMetaType.AUX
        slice_id = None

    operation_key = ("gen", 1)
    admitted, state, _message, _delivered = task.admit_operation(
        operation_key,
        allow_source_access=True,
        no_access_message=(b"unused",),
    )
    assert admitted
    assert state.value == "PENDING"

    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen1",
        peer_rank=1,
        peer_endpoint="tcp://gen",
        unique_rid=17,
        src_ptrs=np.asarray([100], dtype=np.int64),
        dst_ptrs=np.asarray([200], dtype=np.int64),
        sizes=np.asarray([8], dtype=np.int64),
        dst_device_id=0,
        slice_id=slice_id,
        is_last_slice=True,
        meta_type=meta_type,
        bounce_dst_base=300 if channel == "kv" else None,
        session=session,
        source_access_enrolled=True,
        operation_key=operation_key,
        result_identity=operation,
    )
    transfer_status = SimpleNamespace(
        wait=Mock(return_value=backend_succeeded),
    )
    worker_dealer = SimpleNamespace(
        send=Mock(
            side_effect=[
                RuntimeError("initial submission notification failed"),
                RuntimeError("worker retry failed"),
            ]
        )
    )
    req_info = RecvReqInfo(
        sender_req_id=17,
        instance_name="gen",
        instance_rank=1,
        block_ids_per_layer_groups=[],
        unique_rid=17,
        slice_id=0,
        publication_identity=operation.publication,
        aux_publication_identity=operation.publication,
    )
    sender = object.__new__(Sender)
    sender._operation_admission_lock = threading.RLock()
    sender._dealer_admission_closed = False
    sender._device_id = 0
    sender._instance_rank = 0
    sender._bounce = Mock()
    if channel == "kv":
        sender._bounce.build_request.return_value = (object(), 11)
    sender._agent = SimpleNamespace(
        name="test-agent",
        submit_transfer_requests=Mock(return_value=transfer_status),
    )
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(
            instance_name="ctx",
            instance_rank=0,
            endpoint_incarnation=_uuid(6),
        ),
        get_peer_rank_info=Mock(
            return_value=SimpleNamespace(
                instance_name="gen",
                instance_rank=1,
                endpoint_incarnation=_uuid(7),
            )
        ),
    )
    sender._get_or_connect_thread_dealer = Mock(return_value=worker_dealer)
    sender._get_req_info = Mock(return_value={operation_key: req_info})
    sender._send_operation_message = Mock(side_effect=[True, True])
    sender._in_doubt_transfers = []
    sender._in_doubt_transfers_lock = threading.Lock()
    monkeypatch.setattr(Sender, "_make_agent_request", Mock(return_value=object()))

    deliver = sender._deliver_kv_to_agent if channel == "kv" else sender._deliver_aux_to_agent
    deliver(write_meta)

    transfer_status.wait.assert_called_once_with()
    session.record_transfer_terminal.assert_called_once_with(operation)
    assert not task.source_access_active
    assert not write_meta.source_access_enrolled
    assert sender._in_doubt_transfers == []
    if channel == "kv":
        sender._bounce.release_send.assert_called_once_with(11)
        sender._bounce.quarantine_send.assert_not_called()
    expected_status = TaskStatus.TRANSFERRED if backend_succeeded else TaskStatus.ERROR
    assert task.status is expected_status
    assert task.has_pending_result_delivery
    assert worker_dealer.send.call_count == 2
    assert all(
        call.args[0][0] == MessageType.TRANSFER_SUBMITTED_V1
        for call in worker_dealer.send.call_args_list
    )

    sender.retry_terminal_results(session, force=True)

    replayed = [call.args[1][0] for call in sender._send_operation_message.call_args_list]
    terminal_type = (
        MessageType.KV_AGENT_RESULT_V1 if channel == "kv" else MessageType.AUX_AGENT_RESULT_V1
    )
    assert replayed == [
        MessageType.TRANSFER_SUBMITTED_V1,
        terminal_type,
    ]
    assert not task.has_pending_result_delivery


def test_receiver_rejects_stale_submission_endpoint_before_session_lookup() -> None:
    operation = _operation()
    stale = replace(
        operation,
        publication=replace(
            operation.publication,
            destination_endpoint=EndpointIdentity("gen", 1, _uuid(70)),
        ),
    )
    receiver = SimpleNamespace(
        _registrar=SimpleNamespace(
            self_rank_info=SimpleNamespace(
                instance_name="gen",
                instance_rank=1,
                endpoint_incarnation=_uuid(7),
            )
        ),
        _get_session=Mock(),
    )

    Receiver._process_transfer_submitted(
        receiver,
        b"",
        [
            MessageType.TRANSFER_SUBMITTED_V1,
            encode_wire_identity(stale),
        ],
    )

    receiver._get_session.assert_not_called()


def test_receiver_stamps_exact_authenticated_submission() -> None:
    operation = _operation()
    session = SimpleNamespace(record_transfer_submitted=Mock())
    receiver = SimpleNamespace(
        _registrar=SimpleNamespace(
            self_rank_info=SimpleNamespace(
                instance_name="gen",
                instance_rank=1,
                endpoint_incarnation=_uuid(7),
            )
        ),
        _get_session=Mock(return_value=session),
    )

    Receiver._process_transfer_submitted(
        receiver,
        b"",
        [
            MessageType.TRANSFER_SUBMITTED_V1,
            encode_wire_identity(operation),
        ],
    )

    destination_stamp = session.record_transfer_submitted.call_args
    assert destination_stamp.args == (0, 0, operation)
    assert destination_stamp.kwargs["monotonic_start_s"] > 0


def test_exact_publication_only_no_access_result_is_accepted() -> None:
    publication = _operation().publication
    session = SimpleNamespace(
        lock=threading.Lock(),
        protocol_identity=publication.session,
        _publication_identities={(0, 0): publication},
        _result_identities={},
    )

    RxSession.validate_result_identity(session, 0, 0, publication)

    assert session._result_identities == {}


def test_publication_only_no_access_cannot_contradict_authenticated_submission() -> None:
    operation = _operation()
    session = SimpleNamespace(
        lock=threading.Lock(),
        protocol_identity=operation.publication.session,
        _publication_identities={(0, 0): operation.publication},
        _submitted_identities={(0, 0): operation},
        _result_identities={},
    )

    with pytest.raises(StaleProtocolMessageError, match="contradicts"):
        RxSession.validate_result_identity(
            session,
            0,
            0,
            operation.publication,
        )

    assert session._result_identities == {}


def test_publication_only_success_is_rejected_before_session_lookup() -> None:
    receiver = SimpleNamespace(_get_session=Mock())
    message = _make_kv_result_msg(
        0,
        17,
        0,
        True,
        AgentResult.SUCCESS,
        result_identity=_operation().publication,
    )

    Receiver._process_kv_agent_result(receiver, b"", message)

    receiver._get_session.assert_not_called()


def test_publication_only_failure_records_authenticated_bounce_no_access() -> None:
    publication = _operation().publication
    update = SimpleNamespace(accepted=True, conflict=False)
    session = SimpleNamespace(
        validate_result_identity=Mock(),
        record_result_identity=Mock(),
        record_transfer_size=Mock(),
    )
    receiver = SimpleNamespace(
        _get_session=Mock(return_value=session),
        _recv_registry=SimpleNamespace(
            target_mode=Mock(return_value=WriterMode.BOUNCE),
            record_result=Mock(return_value=update),
        ),
        _bounce=SimpleNamespace(record_no_access=Mock()),
        _handle_lifecycle_update=Mock(),
        _finish_bounce=Mock(),
    )
    message = _make_kv_result_msg(
        0,
        17,
        0,
        True,
        AgentResult.FAILED,
        result_identity=publication,
    )

    Receiver._process_kv_agent_result(receiver, b"", message)

    receiver._recv_registry.record_result.assert_called_once_with(
        (17, 0),
        0,
        WriterResult.FAILED,
        WriterMode.NO_REMOTE_ACCESS,
    )
    no_access_call = receiver._bounce.record_no_access.call_args
    assert no_access_call.args[:2] == ((17, 0), 0)
    assert not no_access_call.kwargs["succeeded"]
    session.record_result_identity.assert_called_once_with(0, 0, publication)
    receiver._handle_lifecycle_update.assert_called_once_with(update, peer_rank=0)


def test_pre_submission_failure_uses_publication_only_no_access_identity() -> None:
    operation = _operation()
    sender = SimpleNamespace(_instance_rank=0)
    write_meta = SimpleNamespace(
        meta_type=WriteMetaType.KV,
        unique_rid=17,
        slice_id=0,
        is_last_slice=True,
        result_identity=operation,
    )

    message = Sender._failed_write_meta_message(
        sender,
        write_meta,
        no_remote_access=True,
    )

    assert message[0] == MessageType.KV_AGENT_RESULT_V1
    assert decode_wire_identity(message[1]) == operation.publication
    assert _KV_RESULT_PREFIX.unpack(message[2])[4] == 1


@pytest.mark.parametrize(
    "transfer_size, tail",
    [
        (1, None),
        (0, [b"unexpected-tail"]),
    ],
)
def test_publication_only_no_access_rejects_contradictory_payload_before_lookup(
    transfer_size,
    tail,
) -> None:
    receiver = SimpleNamespace(_get_session=Mock())
    message = _make_kv_result_msg(
        0,
        17,
        0,
        True,
        AgentResult.FAILED,
        transfer_size=transfer_size,
        tail=tail,
        result_identity=_operation().publication,
    )

    Receiver._process_kv_agent_result(receiver, b"", message)

    receiver._get_session.assert_not_called()


def test_no_access_result_still_rejects_a_stale_transfer_session() -> None:
    current = _operation().publication
    stale = replace(
        current,
        session=replace(current.session, transfer_session_id=_uuid(120)),
        operation_id=_uuid(121),
    )
    task = SimpleNamespace(
        _session=SimpleNamespace(protocol_identity=current.session),
        slice_id=0,
    )
    info = RecvReqInfo(
        sender_req_id=17,
        instance_name="gen",
        instance_rank=1,
        block_ids_per_layer_groups=[],
        unique_rid=17,
        slice_id=0,
        publication_identity=stale,
    )

    with pytest.raises(StaleProtocolMessageError, match="source transfer session"):
        Sender._result_identity(
            SimpleNamespace(),
            task,
            info,
            include_source_allocation=False,
        )


@pytest.mark.parametrize(
    "replacement",
    [
        _operation(attempt=40),
        _operation(session=50),
        _operation(generation=110),
    ],
)
def test_attempt_session_and_allocation_replay_is_rejected(replacement) -> None:
    with pytest.raises(StaleProtocolMessageError, match="active transfer operation"):
        require_exact_identity(replacement, _operation())


def test_endpoint_incarnation_replay_is_rejected() -> None:
    current = _operation()
    stale = OperationIdentity(
        publication=PublicationIdentity(
            session=TransferSessionIdentity(
                attempt=current.publication.session.attempt,
                transfer_session_id=current.publication.session.transfer_session_id,
            ),
            source_endpoint=EndpointIdentity("ctx", 0, _uuid(60)),
            destination_endpoint=current.publication.destination_endpoint,
            destination_allocation=current.publication.destination_allocation,
            operation_id=current.publication.operation_id,
            slice_id=current.publication.slice_id,
            writer_rank=current.publication.writer_rank,
        ),
        source_allocation=current.source_allocation,
    )

    with pytest.raises(StaleProtocolMessageError):
        require_exact_identity(stale, current)


def test_one_session_can_publish_to_distinct_writer_endpoint_incarnations() -> None:
    session = _operation().publication.session
    destination_incarnation = _uuid(70)
    receiver = SimpleNamespace(
        _registrar=SimpleNamespace(
            self_rank_info=SimpleNamespace(
                instance_name="gen",
                instance_rank=3,
                endpoint_incarnation=destination_incarnation,
                lifecycle_protocol_version=1,
            )
        )
    )
    peer_info = SimpleNamespace(
        instance_name="ctx",
        lifecycle_protocol_version=1,
        sender_endpoint_incarnations=[_uuid(71), _uuid(72)],
    )
    task = SimpleNamespace(
        _unique_rid=17,
        slice_id=0,
        _kv_slice=SimpleNamespace(
            allocation_lease=SimpleNamespace(
                identity=SimpleNamespace(
                    allocator_domain_id="gen-domain",
                    request_id=17,
                    allocation_generation=4,
                )
            )
        ),
    )
    rx_session = SimpleNamespace(_need_aux=False, protocol_identity=session)

    first = Receiver._publication_identity_for_writer(
        receiver, rx_session, task, peer_info, writer_rank=0
    )
    second = Receiver._publication_identity_for_writer(
        receiver, rx_session, task, peer_info, writer_rank=1
    )

    assert first.session == second.session == session
    assert first.source_endpoint == EndpointIdentity("ctx", 0, _uuid(71))
    assert second.source_endpoint == EndpointIdentity("ctx", 1, _uuid(72))
    assert (
        first.destination_endpoint
        == second.destination_endpoint
        == EndpointIdentity("gen", 3, destination_incarnation)
    )
    assert decode_wire_identity(encode_wire_identity(first)) == first
    assert decode_wire_identity(encode_wire_identity(second)) == second


def test_generation_safe_session_cannot_silently_downgrade_to_protocol_v0() -> None:
    receiver = SimpleNamespace(
        _registrar=SimpleNamespace(
            self_rank_info=SimpleNamespace(
                lifecycle_protocol_version=0,
                qualified_legacy_mode=True,
            )
        )
    )
    peer_info = SimpleNamespace(
        lifecycle_protocol_version=0,
        qualified_legacy_mode=True,
    )
    task = SimpleNamespace(_unique_rid=17)
    rx_session = SimpleNamespace(
        _need_aux=False,
        protocol_identity=_operation().publication.session,
    )

    with pytest.raises(ProtocolIdentityError, match="cannot downgrade"):
        Receiver._publication_identity_for_writer(
            receiver,
            rx_session,
            task,
            peer_info,
            writer_rank=0,
        )


def test_conflicting_publication_replay_cannot_replace_admitted_addresses() -> None:
    admitted_publication = _operation().publication
    stale_publication = replace(
        admitted_publication,
        destination_allocation=AllocationWireIdentity("42", 17, 99),
        operation_id=_uuid(98),
    )
    admitted = RecvReqInfo(
        sender_req_id=17,
        instance_name="gen",
        instance_rank=1,
        block_ids_per_layer_groups=[np.asarray([3], dtype=np.int64)],
        unique_rid=17,
        slice_id=0,
        publication_identity=admitted_publication,
    )
    stale = replace(
        admitted,
        block_ids_per_layer_groups=[np.asarray([999], dtype=np.int64)],
        publication_identity=stale_publication,
    )
    sender = object.__new__(Sender)
    sender._peer_requests = {}
    sender._peer_requests_timestamps = {}
    sender._peer_requests_lock = threading.Lock()

    assert sender._add_req_info(17, admitted) is admitted
    replay = replace(
        admitted,
        block_ids_per_layer_groups=[np.asarray([777], dtype=np.int64)],
    )
    assert sender._add_req_info(17, replay) is admitted
    with pytest.raises(StaleProtocolMessageError, match="conflicting publication replay"):
        sender._add_req_info(17, stale)

    assert next(iter(sender._peer_requests[17].values())) is admitted


def test_cancel_only_settles_the_matching_transfer_session() -> None:
    current_publication = _operation().publication
    next_session = replace(
        current_publication.session,
        transfer_session_id=_uuid(100),
    )
    next_publication = replace(
        current_publication,
        session=next_session,
        operation_id=_uuid(101),
    )
    current_info = RecvReqInfo(
        sender_req_id=17,
        instance_name="gen",
        instance_rank=1,
        block_ids_per_layer_groups=[],
        unique_rid=17,
        slice_id=0,
        publication_identity=current_publication,
    )
    next_info = replace(current_info, publication_identity=next_publication)
    tx_session = SimpleNamespace(
        disagg_request_id=17,
        protocol_identity=current_publication.session,
    )
    sender = object.__new__(Sender)
    sender._shutdown_complete = False
    sender._sessions_lock = threading.Lock()
    sender._peer_requests_lock = threading.Lock()
    sender._peer_requests = {17: {"current": current_info, "next": next_info}}
    sender._pre_cancelled_rids = set()
    sender._settle_cancelled_info = Mock()
    sender.send_cancel_to_receivers = Mock()

    sender.cancel_session(tx_session)

    sender._settle_cancelled_info.assert_called_once_with(
        tx_session,
        current_info,
        settle_aux=True,
    )
    sender.send_cancel_to_receivers.assert_called_once_with(
        17,
        current_publication.session,
    )
    assert sender._pre_cancelled_rids == {current_publication.session}


def test_pre_session_cancel_settles_all_publications_for_exact_session() -> None:
    first_publication = _operation().publication
    second_publication = replace(
        first_publication,
        operation_id=_uuid(110),
        slice_id=1,
    )
    stale_session = replace(
        first_publication.session,
        transfer_session_id=_uuid(111),
    )
    stale_publication = replace(
        first_publication,
        session=stale_session,
        operation_id=_uuid(112),
    )

    def info(publication: PublicationIdentity) -> RecvReqInfo:
        return RecvReqInfo(
            sender_req_id=17,
            instance_name="gen",
            instance_rank=1,
            block_ids_per_layer_groups=[],
            unique_rid=17,
            slice_id=publication.slice_id,
            publication_identity=publication,
        )

    first_info = info(first_publication)
    second_info = info(second_publication)
    stale_info = info(stale_publication)
    sender = object.__new__(Sender)
    sender._operation_admission_lock = threading.RLock()
    sender._dealer_admission_closed = False
    sender._sessions_lock = threading.Lock()
    sender._sessions = {}
    sender._pre_cancelled_rids = set()
    sender._peer_requests_lock = threading.Lock()
    sender._peer_requests = {
        17: {
            "first": first_info,
            "second": second_info,
            "stale": stale_info,
        }
    }
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(
            instance_name="ctx",
            instance_rank=0,
            endpoint_incarnation=_uuid(6),
        ),
        get_peer_rank_info=Mock(
            return_value=SimpleNamespace(
                instance_name="gen",
                instance_rank=1,
                endpoint_incarnation=_uuid(7),
            )
        ),
    )
    sender._send_failed_result_to_receiver = Mock()
    sender._send_aux_failed_result_to_receiver = Mock()

    sender._handle_cancel_session(
        [
            MessageType.CANCEL_SESSION_V1,
            encode_wire_identity(first_publication),
        ]
    )

    assert sender._pre_cancelled_rids == {first_publication.session}
    assert [call.args[0] for call in sender._send_failed_result_to_receiver.call_args_list] == [
        first_info,
        second_info,
    ]
    sender._send_aux_failed_result_to_receiver.assert_not_called()


def test_legacy_cancel_cannot_cancel_generation_safe_session() -> None:
    session = SimpleNamespace(
        protocol_identity=_operation().publication.session,
        cancel=Mock(),
    )
    sender = object.__new__(Sender)
    sender._sessions_lock = threading.Lock()
    sender._sessions = {17: session}
    sender._pre_cancelled_rids = set()

    sender._handle_cancel_session([MessageType.CANCEL_SESSION, b"17"])

    session.cancel.assert_not_called()
    assert sender._pre_cancelled_rids == set()


def test_legacy_clear_preserves_prearrived_generation_safe_request() -> None:
    publication = _operation().publication
    legacy = RecvReqInfo(
        sender_req_id=17,
        instance_name="gen",
        instance_rank=0,
        block_ids_per_layer_groups=[],
        unique_rid=17,
        slice_id=0,
    )
    generation_safe = replace(
        legacy,
        instance_rank=1,
        publication_identity=publication,
    )
    sender = object.__new__(Sender)
    sender._sessions_lock = threading.Lock()
    sender._sessions = {}
    sender._peer_requests_lock = threading.Lock()
    sender._peer_requests = {17: {"legacy": legacy, "v1": generation_safe}}
    sender._peer_requests_timestamps = {17: 1.0}

    sender.clear_session(17)

    assert sender._peer_requests == {17: {"v1": generation_safe}}
    assert sender._peer_requests_timestamps == {17: 1.0}


def test_v1_cancel_refuses_missing_endpoint_publication_instead_of_downgrading() -> None:
    receiver = SimpleNamespace(_get_or_connect_dealer=Mock())

    Receiver.send_cancel_to_senders(
        receiver,
        17,
        {"tcp://ctx0", "tcp://ctx1"},
        {"tcp://ctx0": _operation().publication},
    )

    receiver._get_or_connect_dealer.assert_called_once_with("tcp://ctx0")


def test_protocol_v0_requires_explicit_legacy_qualification() -> None:
    payload = encode_wire_identity(QualifiedLegacyIdentity(logical_request_id=17))

    with pytest.raises(ProtocolIdentityError, match="was not negotiated"):
        decode_wire_identity(payload)

    assert decode_wire_identity(payload, allow_qualified_legacy=True) == QualifiedLegacyIdentity(
        logical_request_id=17
    )


def test_protocol_v0_rejects_unqualified_payload() -> None:
    payload = msgpack.packb(
        {
            "protocol_version": 0,
            "qualified_legacy_mode": False,
            "logical_request_id": 17,
        },
        use_bin_type=True,
    )

    with pytest.raises(ProtocolIdentityError, match="not qualified"):
        decode_wire_identity(payload, allow_qualified_legacy=True)


def test_protocol_v1_rejects_missing_identity_field() -> None:
    payload = msgpack.unpackb(encode_wire_identity(_operation()), raw=False)
    del payload["identity"]["publication"]["destination_allocation"]

    with pytest.raises(ProtocolIdentityError, match="destination_allocation"):
        decode_wire_identity(msgpack.packb(payload, use_bin_type=True))


def test_wire_identity_encoder_rejects_oversized_payload() -> None:
    publication = replace(
        _operation().publication,
        source_endpoint=EndpointIdentity("x" * 4096, 0, _uuid(6)),
    )

    with pytest.raises(ProtocolIdentityError, match="too large"):
        encode_wire_identity(publication)


def test_local_allocation_identity_conversion_is_wire_neutral() -> None:
    class LocalIdentity:
        allocator_domain_id = 42
        request_id = None
        allocation_generation = 9

    assert AllocationWireIdentity.from_local(LocalIdentity()) == AllocationWireIdentity(
        allocator_domain_id="42",
        request_id=None,
        allocation_generation=9,
    )


def test_local_allocation_identity_rejects_missing_allocator_domain() -> None:
    identity = SimpleNamespace(
        allocator_domain_id=None,
        request_id=17,
        allocation_generation=1,
    )

    with pytest.raises(ProtocolIdentityError, match="allocator domain"):
        AllocationWireIdentity.from_local(identity)


def test_allocator_local_request_ids_are_not_coupled_to_logical_request_id() -> None:
    identity = _operation()
    identity = replace(
        identity,
        publication=replace(
            identity.publication,
            destination_allocation=AllocationWireIdentity("gen", 1801, 2),
        ),
        source_allocation=AllocationWireIdentity("ctx", 901, 1),
    )

    assert decode_wire_identity(encode_wire_identity(identity)) == identity
    assert identity.logical_request_id == 17
    assert identity.publication.destination_allocation.request_id == 1801
    assert identity.source_allocation.request_id == 901


def test_transfer_protocol_identity_from_complete_params() -> None:
    params = SimpleNamespace(
        logical_request_id=17,
        prefill_artifact_id=str(_uuid(2)),
        artifact_version=3,
        handoff_attempt_uuid=str(_uuid(4)),
        consumer_grant_id=str(_uuid(5)),
        transfer_session_id=str(_uuid(6)),
    )

    identity = transfer_protocol_identity_from_params(params)

    assert identity == TransferSessionIdentity(
        attempt=AttemptIdentity(
            logical_request_id=17,
            prefill_artifact_id=_uuid(2),
            artifact_version=3,
            handoff_attempt_uuid=_uuid(4),
        ),
        transfer_session_id=_uuid(6),
    )


@pytest.mark.parametrize("params", [None, SimpleNamespace()])
def test_transfer_protocol_identity_from_absent_params_is_none(params) -> None:
    assert transfer_protocol_identity_from_params(params) is None


@pytest.mark.parametrize(
    "params, match",
    [
        (
            SimpleNamespace(logical_request_id=17),
            "must be provided together",
        ),
        (
            SimpleNamespace(
                logical_request_id=17,
                prefill_artifact_id=str(_uuid(2)),
                artifact_version=3,
                handoff_attempt_uuid=str(_uuid(4)),
                consumer_grant_id=str(_uuid(5)),
                transfer_session_id="not-a-uuid",
            ),
            "canonical non-nil UUID",
        ),
        (
            SimpleNamespace(
                logical_request_id=True,
                prefill_artifact_id=str(_uuid(2)),
                artifact_version=3,
                handoff_attempt_uuid=str(_uuid(4)),
                consumer_grant_id=str(_uuid(5)),
                transfer_session_id=str(_uuid(6)),
            ),
            "logical_request_id",
        ),
    ],
)
def test_transfer_protocol_identity_from_params_rejects_partial_or_invalid(
    params,
    match,
) -> None:
    with pytest.raises(ProtocolIdentityError, match=match):
        transfer_protocol_identity_from_params(params)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EndpointIdentity("", 0, _uuid(1)),
        lambda: EndpointIdentity("ctx", -1, _uuid(1)),
        lambda: EndpointIdentity("ctx", 0, UUID(int=0)),
        lambda: AllocationWireIdentity("domain", 1, 0),
        lambda: AttemptIdentity(1, _uuid(1), -1, _uuid(2)),
        lambda: replace(_operation().publication, writer_rank=1),
    ],
)
def test_identity_construction_rejects_invalid_fields(factory) -> None:
    with pytest.raises(ProtocolIdentityError):
        factory()
