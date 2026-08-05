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

import msgpack
import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBuffer
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    AuxSendTask,
    MessageType,
    Receiver,
    RecvReqInfo,
    RxSession,
    Sender,
    WriteMeta,
    WriteMetaType,
    _make_aux_result_msg,
)
from tensorrt_llm._torch.disaggregation.protocol import (
    AllocationWireIdentity,
    AttemptIdentity,
    EndpointIdentity,
    OperationIdentity,
    ProtocolIdentityError,
    PublicationIdentity,
    StaleProtocolMessageError,
    TransferProtocolIdentity,
    decode_wire_identity,
)


def _uuid(value: int) -> UUID:
    return UUID(int=value)


def _session_identity() -> TransferProtocolIdentity:
    return TransferProtocolIdentity(
        attempt=AttemptIdentity(
            logical_request_id=17,
            prefill_artifact_id=_uuid(1),
            artifact_version=0,
            handoff_attempt_uuid=_uuid(2),
        ),
        transfer_session_id=_uuid(3),
    )


def _aux_publication(
    destination_allocation: AllocationWireIdentity,
    *,
    operation_id: int = 6,
) -> PublicationIdentity:
    return PublicationIdentity(
        session=_session_identity(),
        source_endpoint=EndpointIdentity("ctx", 0, _uuid(4)),
        destination_endpoint=EndpointIdentity("gen", 1, _uuid(5)),
        destination_allocation=destination_allocation,
        operation_id=_uuid(operation_id),
        slice_id=0,
        writer_rank=0,
    )


def _operation(publication: PublicationIdentity, generation: int = 11) -> OperationIdentity:
    return OperationIdentity(
        publication=publication,
        source_allocation=AllocationWireIdentity(
            "ctx-aux-domain",
            17,
            generation,
        ),
    )


def _rx_state(
    buffer: AuxBuffer,
    publication: PublicationIdentity,
    slot,
) -> SimpleNamespace:
    return SimpleNamespace(
        lock=threading.Lock(),
        protocol_identity=publication.session,
        _aux_buffer=buffer,
        aux_slot=slot.id,
        _aux_allocation_identity=slot.identity,
        _aux_publication_identities={0: publication},
        _aux_submitted_identities={},
        _aux_active_submission_times={},
        _aux_result_identities={},
        _publication_identities={},
        _submitted_identities={},
        _active_submission_times={},
        _result_identities={},
        transfer_start_time=None,
        _transfer_submitted_at_s=None,
        _destination_owner=None,
    )


def test_recv_req_info_roundtrip_keeps_aux_publication_separate() -> None:
    kv_publication = _aux_publication(
        AllocationWireIdentity("gen-kv-domain", 17, 20),
        operation_id=7,
    )
    aux_publication = _aux_publication(
        AllocationWireIdentity("gen-aux-domain", 17, 21),
    )
    info = RecvReqInfo(
        sender_req_id=17,
        instance_name="gen",
        instance_rank=1,
        block_ids_per_layer_groups=[np.asarray([3], dtype=np.int64)],
        unique_rid=17,
        aux_slot=2,
        slice_id=0,
        publication_identity=kv_publication,
        aux_publication_identity=aux_publication,
    )

    restored = RecvReqInfo.from_bytes(info.to_bytes())

    assert restored.publication_identity == kv_publication
    assert restored.aux_publication_identity == aux_publication


def test_protocol_v0_recv_req_info_wire_shape_is_unchanged() -> None:
    info = RecvReqInfo(
        sender_req_id=17,
        instance_name="gen",
        instance_rank=1,
        block_ids_per_layer_groups=[np.asarray([3], dtype=np.int64)],
        unique_rid=17,
        aux_slot=2,
        slice_id=0,
    )

    payload = msgpack.unpackb(info.to_bytes(), raw=False)

    assert set(payload) == {
        "sender_req_id",
        "instance_name",
        "instance_rank",
        "block_ids_per_layer_groups",
        "unique_rid",
        "dst_start_token",
        "aux_slot",
        "mamba_state_index",
        "slice_id",
        "bounce_dst_base",
    }


def test_protocol_v0_aux_result_wire_shape_is_unchanged() -> None:
    message = _make_aux_result_msg(0, 17, AgentResult.SUCCESS)

    assert message == [
        MessageType.AUX_AGENT_RESULT,
        b"0",
        b"17",
        b"SUCCESS",
    ]


def test_protocol_v1_aux_result_carries_exact_operation_identity() -> None:
    publication = _aux_publication(AllocationWireIdentity("gen-aux-domain", 17, 21))
    operation = _operation(publication)

    message = _make_aux_result_msg(
        0,
        17,
        AgentResult.SUCCESS,
        result_identity=operation,
    )

    assert message[0] == MessageType.AUX_AGENT_RESULT_V1
    assert decode_wire_identity(message[1]) == operation
    assert message[2:] == [b"0", b"17", b"SUCCESS"]


def test_aux_submission_and_result_require_both_allocation_generations() -> None:
    buffer = AuxBuffer(1, 1, 1)
    slot = buffer.alloc_slot(request_id=17)
    publication = _aux_publication(AllocationWireIdentity.from_local(slot.identity))
    operation = _operation(publication)
    state = _rx_state(buffer, publication, slot)

    assert RxSession.record_transfer_submitted(
        state,
        0,
        0,
        operation,
        transfer_start_time=100,
        monotonic_start_s=10.0,
    )
    RxSession.validate_aux_result_identity(state, 0, operation)
    RxSession.record_aux_result_identity(state, 0, operation)

    assert state._aux_submitted_identities == {0: operation}
    assert state._aux_result_identities == {0: operation}
    assert state._aux_active_submission_times == {}


def test_aux_publication_only_result_is_an_exact_no_access_proof() -> None:
    buffer = AuxBuffer(1, 1, 1)
    slot = buffer.alloc_slot(request_id=17)
    publication = _aux_publication(AllocationWireIdentity.from_local(slot.identity))
    state = _rx_state(buffer, publication, slot)

    RxSession.validate_aux_result_identity(state, 0, publication)
    RxSession.record_aux_result_identity(state, 0, publication)

    assert state._aux_submitted_identities == {}
    assert state._aux_result_identities == {0: publication}


def test_aux_result_with_changed_source_generation_is_stale() -> None:
    buffer = AuxBuffer(1, 1, 1)
    slot = buffer.alloc_slot(request_id=17)
    publication = _aux_publication(AllocationWireIdentity.from_local(slot.identity))
    operation = _operation(publication)
    changed = replace(
        operation,
        source_allocation=replace(
            operation.source_allocation,
            allocation_generation=12,
        ),
    )
    state = _rx_state(buffer, publication, slot)

    RxSession.record_transfer_submitted(
        state,
        0,
        0,
        operation,
        transfer_start_time=100,
        monotonic_start_s=10.0,
    )
    with pytest.raises(
        StaleProtocolMessageError,
        match="source allocation",
    ):
        RxSession.validate_aux_result_identity(state, 0, changed)

    assert state._aux_result_identities == {}


def test_aux_result_cannot_settle_a_reused_destination_slot() -> None:
    buffer = AuxBuffer(1, 1, 1)
    first = buffer.alloc_slot(request_id=17)
    publication = _aux_publication(AllocationWireIdentity.from_local(first.identity))
    state = _rx_state(buffer, publication, first)
    buffer.free_slot(first.id, first.identity)
    second = buffer.alloc_slot(request_id=18)

    with pytest.raises(
        StaleProtocolMessageError,
        match="destination allocation",
    ):
        RxSession.record_transfer_submitted(
            state,
            0,
            0,
            _operation(publication),
            transfer_start_time=100,
            monotonic_start_s=10.0,
        )
    with pytest.raises(
        StaleProtocolMessageError,
        match="destination allocation",
    ):
        RxSession.validate_aux_result_identity(state, 0, publication)

    assert buffer.allocation_identity(second.id) == second.identity
    assert state._aux_submitted_identities == {}
    assert state._aux_result_identities == {}


def test_aux_data_access_cannot_cross_a_reused_slot_generation() -> None:
    buffer = AuxBuffer(1, 1, 1)
    first = buffer.alloc_slot(request_id=17)
    buffer.free_slot(first.id, first.identity)
    second = buffer.alloc_slot(request_id=18)

    with pytest.raises(ValueError, match="different allocation generation"):
        buffer.fill_slot(
            second.id,
            SimpleNamespace(),
            first.identity,
        )
    with pytest.raises(ValueError, match="different allocation generation"):
        buffer.get_slot_data(second.id, first.identity)

    assert buffer.allocation_identity(second.id) == second.identity


def test_aux_publication_is_reused_across_kv_slices() -> None:
    buffer = AuxBuffer(1, 1, 1)
    slot = buffer.alloc_slot(request_id=17)
    self_info = SimpleNamespace(
        instance_name="gen",
        instance_rank=1,
        endpoint_incarnation=_uuid(5),
    )
    peer_info = SimpleNamespace(
        instance_name="ctx",
        sender_endpoint_incarnations=(_uuid(4),),
    )
    receiver = object.__new__(Receiver)
    receiver._registrar = SimpleNamespace(self_rank_info=self_info)
    session = SimpleNamespace(
        protocol_identity=_session_identity(),
        _aux_publication_identities={},
        _aux_buffer=buffer,
        aux_slot=slot.id,
        _aux_allocation_identity=slot.identity,
    )

    first = Receiver._aux_publication_identity_for_writer(
        receiver,
        session,
        peer_info,
        0,
    )
    second = Receiver._aux_publication_identity_for_writer(
        receiver,
        session,
        peer_info,
        0,
    )

    assert second is first
    assert first.destination_allocation == AllocationWireIdentity.from_local(slot.identity)


def test_sender_rejects_conflicting_aux_publication_across_kv_slices() -> None:
    destination_allocation = AllocationWireIdentity("gen-aux-domain", 17, 21)
    aux_publication = _aux_publication(destination_allocation)
    kv_publication = _aux_publication(
        AllocationWireIdentity("gen-kv-domain", 17, 31),
        operation_id=7,
    )
    sender = object.__new__(Sender)
    sender._peer_requests_lock = threading.Lock()
    sender._peer_requests = {}
    sender._peer_requests_timestamps = {}
    first = RecvReqInfo(
        sender_req_id=17,
        instance_name="gen",
        instance_rank=1,
        block_ids_per_layer_groups=[],
        unique_rid=17,
        aux_slot=0,
        slice_id=0,
        publication_identity=kv_publication,
        aux_publication_identity=aux_publication,
    )
    second = replace(
        first,
        slice_id=1,
        publication_identity=replace(
            kv_publication,
            operation_id=_uuid(8),
            slice_id=1,
        ),
        aux_publication_identity=replace(
            aux_publication,
            operation_id=_uuid(9),
        ),
    )

    Sender._add_req_info(sender, 17, first)
    with pytest.raises(
        StaleProtocolMessageError,
        match="conflicting session-level AUX publications",
    ):
        Sender._add_req_info(sender, 17, second)


def test_aux_source_operation_is_bound_to_the_live_slot_generation() -> None:
    source_buffer = AuxBuffer(1, 1, 1)
    source_slot = source_buffer.alloc_slot(request_id=17)
    destination_allocation = AllocationWireIdentity("gen-aux-domain", 17, 5)
    publication = _aux_publication(destination_allocation)
    peer_info = SimpleNamespace(
        instance_name="gen",
        instance_rank=1,
        endpoint_incarnation=_uuid(5),
    )
    sender = object.__new__(Sender)
    sender._instance_rank = 0
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(
            instance_name="ctx",
            instance_rank=0,
            endpoint_incarnation=_uuid(4),
        ),
        get_peer_rank_info=Mock(return_value=peer_info),
    )
    session = SimpleNamespace(
        protocol_identity=publication.session,
        _aux_buffer=source_buffer,
    )
    task = AuxSendTask(
        SimpleNamespace(disagg_request_id=17),
        source_slot.id,
        session=session,
        allocation_identity=source_slot.identity,
    )
    info = RecvReqInfo(
        sender_req_id=17,
        instance_name="gen",
        instance_rank=1,
        block_ids_per_layer_groups=[],
        unique_rid=17,
        aux_slot=0,
        slice_id=0,
        aux_publication_identity=publication,
    )

    operation = Sender._result_identity(sender, task, info)

    assert isinstance(operation, OperationIdentity)
    assert operation.source_allocation == AllocationWireIdentity.from_local(source_slot.identity)
    source_buffer.free_slot(source_slot.id, source_slot.identity)
    source_buffer.alloc_slot(request_id=18)
    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen1",
        peer_rank=1,
        peer_endpoint="tcp://gen:1234",
        unique_rid=17,
        src_ptrs=np.asarray([1], dtype=np.int64),
        dst_ptrs=np.asarray([2], dtype=np.int64),
        sizes=np.asarray([4], dtype=np.int64),
        meta_type=WriteMetaType.AUX,
        session=session,
        result_identity=operation,
    )
    with pytest.raises(
        StaleProtocolMessageError,
        match="source allocation",
    ):
        Sender._validate_transfer_submission(sender, write_meta)
    with pytest.raises(
        StaleProtocolMessageError,
        match="source allocation generation",
    ):
        Sender._result_identity(sender, task, info)


def test_generation_first_v1_rejects_unknown_adp_writer_before_publication() -> None:
    session = SimpleNamespace(
        protocol_identity=_session_identity(),
        _need_aux=True,
    )

    with pytest.raises(ProtocolIdentityError, match="exact context DP writer"):
        Receiver._validate_aux_writer_cohort(session, None)

    Receiver._validate_aux_writer_cohort(session, 0)


def test_stale_aux_result_is_dropped_before_session_state_mutates() -> None:
    publication = _aux_publication(AllocationWireIdentity("gen-aux-domain", 17, 21))
    operation = _operation(publication)
    session = Mock()
    session.validate_aux_result_identity.side_effect = StaleProtocolMessageError(
        "stale AUX generation"
    )
    receiver = object.__new__(Receiver)
    receiver._get_session = Mock(return_value=session)

    Receiver._process_aux_agent_result(
        receiver,
        b"",
        _make_aux_result_msg(
            0,
            17,
            AgentResult.SUCCESS,
            result_identity=operation,
        ),
    )

    session.validate_aux_result_identity.assert_called_once_with(0, operation)
    session.record_aux_result_identity.assert_not_called()
    session.process_aux_agent_result.assert_not_called()
