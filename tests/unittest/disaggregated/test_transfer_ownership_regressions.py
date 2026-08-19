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
"""Regressions for receive-side KV transfer ownership boundaries."""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVRecvTask,
    KVSendTask,
    MessageType,
    Receiver,
    RecvReqInfo,
    RxSession,
    Sender,
    TransferWorker,
    TransferWorkerConfig,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle


class _BounceProbe:
    """No-bounce probe that records physical-owner cleanup decisions."""

    def __init__(self) -> None:
        self.failed_writers: list[tuple[tuple[int, int], int]] = []
        self.orphaned: list[tuple[int, int]] = []

    def record_failure(self, rid_slice: tuple[int, int], peer_rank: int) -> None:
        self.failed_writers.append((rid_slice, peer_rank))

    def reserve(self, _receiver_req, _num_writers: int, *, extra_bytes: int = 0) -> bool:
        del extra_bytes
        return False

    def release_idle_reservation(self, _rid_slice: tuple[int, int]) -> None:
        return

    def orphan_reservation(self, rid_slice: tuple[int, int]) -> None:
        self.orphaned.append(rid_slice)

    def is_bounced(self, _rid_slice: tuple[int, int]) -> bool:
        return False


class _LateReservationBounce(_BounceProbe):
    """Track a reservation created after cancellation already ran cleanup."""

    def __init__(self) -> None:
        super().__init__()
        self.active_reservations: set[tuple[int, int]] = set()

    def release_idle_reservation(self, rid_slice: tuple[int, int]) -> None:
        self.active_reservations.discard(rid_slice)


class _TrackingLock:
    """Record when one selected thread attempts to enter a critical section."""

    def __init__(self, race_outcomes: queue.Queue[str]) -> None:
        self._lock = threading.Lock()
        self._race_outcomes = race_outcomes
        self.tracked_thread_id: int | None = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if threading.get_ident() == self.tracked_thread_id:
            if self._lock.acquire(blocking=False):
                return True
            self._race_outcomes.put("blocked")
            if not blocking:
                return False
        if timeout == -1:
            return self._lock.acquire(blocking)
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "_TrackingLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.release()


class _ReceiverProbe:
    """Minimal receiver that publishes one destination to two writers."""

    def __init__(self) -> None:
        self._bounce = _BounceProbe()
        self._enforce_physical_ownership = True
        self._physical_ownership_fault: BaseException | None = None
        self._physical_ownership_fault_lock = threading.Lock()
        self._session: RxSession | None = None
        self._next_owner_generation = 1
        self.clear_count = 0
        self.cancel_count = 0

    def setup_session(self, session: RxSession) -> None:
        self._session = session

    def allocate_owner_generation(self) -> int:
        generation = self._next_owner_generation
        self._next_owner_generation += 1
        return generation

    def dispatch_task(self, task: KVRecvTask) -> None:
        assert self._session is not None
        task.expected_transfers = 2
        assert self._session.try_begin_transfer(task.slice_id, set(), {0, 1})

    def send_cancel_to_senders(self, _unique_rid: int, _sender_endpoints: set[str]) -> None:
        self.cancel_count += 1

    def clear_session(self, _unique_rid: int) -> None:
        self.clear_count += 1

    def _record_physical_ownership_fault(self, error: BaseException) -> None:
        with self._physical_ownership_fault_lock:
            if self._physical_ownership_fault is None:
                self._physical_ownership_fault = error

    @property
    def physical_ownership_fault(self) -> BaseException | None:
        with self._physical_ownership_fault_lock:
            return self._physical_ownership_fault


class _OneSlotAllocator:
    """Model the caller that releases a request allocation on a True result."""

    def __init__(self, owner: int) -> None:
        self.owner: int | None = owner
        self.release_count = 0

    @property
    def is_reusable(self) -> bool:
        return self.owner is None

    def apply_reuse_decision(self, safe_to_reuse: bool) -> None:
        if safe_to_reuse and self.owner is not None:
            self.owner = None
            self.release_count += 1


def _make_rank_info(
    physical_ownership_protocol: int,
    *,
    sender_endpoints: list[str] | None = None,
) -> RankInfo:
    return RankInfo(
        instance_name="peer",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[1],
        sender_endpoints=sender_endpoints or [],
        self_endpoint="tcp://peer",
        transfer_engine_info=b"agent",
        physical_ownership_protocol=physical_ownership_protocol,
    )


def _make_rx_session(receiver: object, rid: int) -> RxSession:
    return RxSession(
        request_id=rid,
        params=DisaggregatedParams(disagg_request_id=rid),
        receiver=receiver,
    )


@pytest.mark.cpu_only
def test_failed_writer_cannot_authorize_reuse_while_sibling_is_active() -> None:
    sibling_started = threading.Event()
    release_sibling = threading.Event()
    sibling_errors: list[Exception] = []
    receiver = _ReceiverProbe()
    session = _make_rx_session(receiver, rid=41)
    session.receive(KVSlice(is_last_slice=True))
    request = SimpleNamespace(
        request_id=41,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=41),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {41: session}
    transceiver._recv_reqs = {41: request}
    allocator = _OneSlotAllocator(owner=41)

    def finish_sibling_writer() -> None:
        try:
            sibling_started.set()
            release_sibling.wait()
            session.process_kv_agent_result(
                peer_rank=1,
                sender_slice_id=0,
                is_last_slice=True,
                status=AgentResult.SUCCESS,
            )
        except Exception as error:
            sibling_errors.append(error)

    sibling_thread = threading.Thread(target=finish_sibling_writer, daemon=True)
    sibling_thread.start()

    try:
        assert sibling_started.wait(timeout=10)

        # Writer 0 is terminal, but writer 1 has not reported a terminal result
        # and may still write to the same receive-side KV allocation.
        session.process_kv_agent_result(
            peer_rank=0,
            sender_slice_id=0,
            is_last_slice=True,
            status=AgentResult.FAILED,
        )

        safe_to_reuse = transceiver.cancel_request(request)
        allocator.apply_reuse_decision(safe_to_reuse)

        assert safe_to_reuse is False, (
            "cancel_request() authorized receive-side KV reuse before every "
            "published writer reached a terminal physical state"
        )
        assert not allocator.is_reusable
        assert 41 in transceiver._recv_sessions
        assert receiver.clear_count == 0
    finally:
        release_sibling.set()
        sibling_thread.join(timeout=10)

    assert not sibling_thread.is_alive()
    assert sibling_errors == []

    # The allocation becomes reusable only after the remaining writer reports
    # a terminal physical result. Repeated cancellation must not clean it up
    # more than once.
    safe_to_reuse = transceiver.cancel_request(request)
    allocator.apply_reuse_decision(safe_to_reuse)
    assert safe_to_reuse is True
    assert allocator.is_reusable
    assert allocator.release_count == 1
    assert 41 not in transceiver._recv_sessions
    assert receiver.clear_count == 1
    repeated_decision = transceiver.cancel_request(request)
    assert repeated_decision is True
    allocator.apply_reuse_decision(repeated_decision)
    assert allocator.release_count == 1
    assert receiver.clear_count == 1


@pytest.mark.cpu_only
def test_pre_cancelled_rx_session_never_publishes_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 73
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = {rid}
    receiver._bounce = _BounceProbe()
    receiver._enforce_physical_ownership = True
    receiver._physical_ownership_fault = None
    receiver._physical_ownership_fault_lock = threading.Lock()
    receiver._next_owner_generation = 1
    receiver._shutdown = True
    receiver.dispatch_task = Mock()
    receiver.send_cancel_to_senders = Mock()

    monkeypatch.setattr(
        transfer_mod.tensorrt_llm.bindings,
        "global_steady_clock_now",
        lambda: 0,
    )

    session = _make_rx_session(receiver, rid)
    assert session.status == SessionStatus.CANCELLED

    session.receive(KVSlice(is_last_slice=True))

    assert (len(session._kv_tasks), receiver.dispatch_task.call_count) == (0, 0), (
        "a pre-cancelled receive session created and published a destination task"
    )


@pytest.mark.cpu_only
def test_remote_cancel_resolves_strong_owned_session() -> None:
    rid = 77
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = set()
    receiver._bounce = _BounceProbe()
    receiver._enforce_physical_ownership = True
    receiver.send_cancel_to_senders = Mock()
    session = _make_rx_session(receiver, rid)

    assert receiver._sessions[rid] is session
    receiver._handle_cancel_session([MessageType.CANCEL_SESSION, str(rid).encode("ascii")])

    assert session.status == SessionStatus.CANCELLED
    receiver.send_cancel_to_senders.assert_not_called()


@pytest.mark.cpu_only
def test_remote_cancelled_session_is_retained_until_writers_drain() -> None:
    rid = 78
    request = SimpleNamespace(request_id=rid)
    session = SimpleNamespace(
        _enforce_physical_ownership=True,
        status=SessionStatus.CANCELLED,
        is_completed=Mock(return_value=False),
        has_failed=Mock(return_value=True),
        wait_complete=Mock(return_value=None),
        resources_drained=Mock(return_value=False),
        close=Mock(return_value=False),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._mapping = SimpleNamespace(
        pp_size=1,
        enable_attention_dp=False,
        world_size=1,
    )
    transceiver._gen_allgather = Mock()
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: request}

    completed, failed, cancelled = transceiver.check_gen_transfer_status(0)

    assert (completed, failed, cancelled) == ([], [], [])
    assert transceiver._recv_sessions[rid] is session
    assert transceiver._recv_reqs[rid] is request
    session.close.assert_not_called()


@pytest.mark.cpu_only
def test_non_terminal_writer_result_does_not_authorize_reuse() -> None:
    receiver = _ReceiverProbe()
    session = _make_rx_session(receiver, rid=80)
    session.receive(KVSlice(is_last_slice=True))

    session.process_kv_agent_result(
        peer_rank=0,
        sender_slice_id=0,
        is_last_slice=False,
        status=AgentResult.SUCCESS,
    )

    assert session.status == SessionStatus.TRANSFERRING
    assert not session.resources_drained()


@pytest.mark.cpu_only
def test_cancel_after_publication_cannot_overtake_request_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 79
    request_data_started = threading.Event()
    finish_request_data = threading.Event()
    race_outcomes: queue.Queue[str] = queue.Queue()
    protocol_order: list[str] = []
    receive_errors: list[Exception] = []
    cancel_errors: list[Exception] = []
    initial_cancel_outcome: str | None = None

    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = set()
    receiver._bounce = _BounceProbe()
    receiver._enforce_physical_ownership = True
    receiver._shutdown = True
    receiver._dealers = {}
    receiver._build_recv_req_info = Mock(
        return_value=SimpleNamespace(
            unique_rid=rid,
            slice_id=0,
            mamba_state_index=None,
            bounce_dst_base=None,
            to_bytes=Mock(return_value=b"receiver-request"),
        )
    )
    overlap = SimpleNamespace(ranks=[0])
    receiver._registrar = SimpleNamespace(
        get_peer_overlap=Mock(return_value=overlap),
        self_extractor=SimpleNamespace(page_table=None),
    )
    receiver._get_sender_info = Mock(
        return_value=SimpleNamespace(
            sender_endpoints={0: "tcp://sender-0"},
            page_table=None,
            tp_size=1,
            pp_size=1,
            cp_size=1,
            dp_size=1,
            attention=None,
            physical_ownership_protocol=1,
        )
    )

    def request_sender_data(_endpoint: str, _receiver_info_bytes: bytes) -> None:
        protocol_order.append("request_data_started")
        request_data_started.set()
        assert finish_request_data.wait(timeout=10)
        protocol_order.append("request_data_sent")

    def send_cancel_to_senders(_unique_rid: int, _sender_endpoints: set[str]) -> None:
        protocol_order.append("cancel_sent")
        race_outcomes.put("cancel_sent")

    receiver._request_sender_data = request_sender_data
    receiver.send_cancel_to_senders = send_cancel_to_senders

    monkeypatch.setattr(
        transfer_mod.tensorrt_llm.bindings,
        "global_steady_clock_now",
        lambda: 0,
    )

    session = RxSession(
        request_id=rid,
        params=DisaggregatedParams(disagg_request_id=rid, ctx_dp_rank=0),
        receiver=receiver,
    )
    session_lock = _TrackingLock(race_outcomes)
    session.lock = session_lock

    def receive() -> None:
        try:
            session.receive(KVSlice(is_last_slice=True))
        except Exception as error:
            receive_errors.append(error)

    def cancel() -> None:
        session_lock.tracked_thread_id = threading.get_ident()
        try:
            session.cancel()
        except Exception as error:
            cancel_errors.append(error)

    receive_thread = threading.Thread(target=receive, daemon=True)
    cancel_thread = threading.Thread(target=cancel, daemon=True)
    receive_thread.start()
    try:
        assert request_data_started.wait(timeout=10)
        cancel_thread.start()
        initial_cancel_outcome = race_outcomes.get(timeout=10)
    finally:
        finish_request_data.set()
        receive_thread.join(timeout=10)
        cancel_thread.join(timeout=10)

    assert not receive_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert receive_errors == []
    assert cancel_errors == []
    assert initial_cancel_outcome == "blocked"
    assert protocol_order == ["request_data_started", "request_data_sent", "cancel_sent"], (
        "cancellation overtook an already-authorized REQUEST_DATA publication"
    )


@pytest.mark.cpu_only
def test_cancel_before_dispatch_releases_late_idle_reservation() -> None:
    rid = 81
    receiver = _ReceiverProbe()
    bounce = _LateReservationBounce()
    receiver._bounce = bounce
    session = _make_rx_session(receiver, rid)
    task = session.prepare_receive(KVSlice(is_last_slice=True))
    assert task is not None

    assert session.cancel_local()

    bounce.active_reservations.add((rid, task.slice_id))
    with pytest.raises(RuntimeError, match="became terminal before publication"):
        session.dispatch_prepared_receive(task)

    assert bounce.active_reservations == set(), (
        "cancel-before-dispatch left a reservation allocated after cancellation cleanup"
    )
    assert session.status == SessionStatus.CANCELLED
    assert task.resources_drained


@pytest.mark.cpu_only
def test_recv_req_info_generation_wire_extension_is_component_gated() -> None:
    kwargs = {
        "sender_req_id": 1,
        "instance_name": "gen",
        "instance_rank": 0,
        "block_ids_per_layer_groups": [np.array([1, 2], dtype=np.int64)],
        "unique_rid": 81,
    }
    legacy = RecvReqInfo(**kwargs)
    owned = RecvReqInfo(**kwargs, owner_generation=7)

    legacy_payload = msgpack.unpackb(legacy.to_bytes(), raw=False)
    owned_payload = msgpack.unpackb(owned.to_bytes(), raw=False)

    assert "owner_generation" not in legacy_payload
    assert owned_payload["owner_generation"] == 7
    assert RecvReqInfo.from_bytes(legacy.to_bytes()).owner_generation is None
    assert RecvReqInfo.from_bytes(owned.to_bytes()).owner_generation == 7


@pytest.mark.cpu_only
def test_owner_generation_result_wire_extension_is_component_gated() -> None:
    legacy = transfer_mod._make_kv_result_msg(0, 81, 0, True, AgentResult.FAILED)
    owned = transfer_mod._make_kv_result_msg(
        0,
        81,
        0,
        True,
        AgentResult.FAILED,
        owner_generation=7,
    )

    assert len(legacy[1]) == transfer_mod._KV_RESULT_PREFIX.size
    assert len(owned[1]) == transfer_mod._KV_RESULT_PREFIX_V1.size
    assert transfer_mod._KV_RESULT_PREFIX_V1.unpack(owned[1])[3] == 7


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("enforce_ownership", "owner_generation"),
    [(False, None), (True, 7)],
    ids=["legacy", "owned"],
)
def test_result_parser_preserves_component_wire_contract(
    enforce_ownership: bool,
    owner_generation: int | None,
) -> None:
    session = SimpleNamespace(process_kv_agent_result=Mock())
    receiver = object.__new__(Receiver)
    receiver._enforce_physical_ownership = enforce_ownership
    receiver._get_session = Mock(return_value=session)
    message = transfer_mod._make_kv_result_msg(
        0,
        82,
        0,
        True,
        AgentResult.SUCCESS,
        owner_generation=owner_generation,
    )

    receiver._process_kv_agent_result(b"peer", message)

    assert session.process_kv_agent_result.call_args.kwargs["owner_generation"] == owner_generation


@pytest.mark.cpu_only
def test_stale_terminal_generation_cannot_settle_current_receive_owner() -> None:
    receiver = _ReceiverProbe()
    receiver._next_owner_generation = 2
    session = _make_rx_session(receiver, rid=83)
    session.receive(KVSlice(is_last_slice=True))
    task = session._kv_tasks[0]
    assert task.owner_generation == 2

    session.process_kv_agent_result(
        peer_rank=0,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.FAILED,
        owner_generation=1,
    )

    assert task.status == transfer_mod.TaskStatus.TRANSFERRING
    assert not task.resources_drained
    assert receiver.physical_ownership_fault is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("owner_generation", [0, 3], ids=["non-positive", "future"])
def test_unmatchable_generation_poison_retains_destination(owner_generation: int) -> None:
    receiver = _ReceiverProbe()
    receiver._next_owner_generation = 2
    session = _make_rx_session(receiver, rid=84)
    session.receive(KVSlice(is_last_slice=True))

    with pytest.raises((ValueError, RuntimeError), match="owner generation|owner_generation"):
        session.process_kv_agent_result(
            peer_rank=0,
            sender_slice_id=0,
            is_last_slice=True,
            status=AgentResult.SUCCESS,
            owner_generation=owner_generation,
        )

    assert receiver.physical_ownership_fault is not None
    assert not session.resources_drained()
    assert not session.close()


@pytest.mark.cpu_only
def test_duplicate_writer_result_cannot_substitute_for_missing_sibling() -> None:
    receiver = _ReceiverProbe()
    session = _make_rx_session(receiver, rid=85)
    session.receive(KVSlice(is_last_slice=True))
    generation = session._kv_tasks[0].owner_generation

    for _ in range(2):
        session.process_kv_agent_result(
            peer_rank=0,
            sender_slice_id=0,
            is_last_slice=True,
            status=AgentResult.FAILED,
            owner_generation=generation,
        )

    assert not session.resources_drained()
    assert receiver.physical_ownership_fault is None

    session.process_kv_agent_result(
        peer_rank=1,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.SUCCESS,
        owner_generation=generation,
    )
    assert session.resources_drained()


@pytest.mark.cpu_only
def test_contradictory_writer_result_poison_retains_destination() -> None:
    receiver = _ReceiverProbe()
    session = _make_rx_session(receiver, rid=86)
    session.receive(KVSlice(is_last_slice=True))
    generation = session._kv_tasks[0].owner_generation

    session.process_kv_agent_result(
        peer_rank=0,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.FAILED,
        owner_generation=generation,
    )
    with pytest.raises(RuntimeError, match="contradictory terminal evidence"):
        session.process_kv_agent_result(
            peer_rank=0,
            sender_slice_id=0,
            is_last_slice=True,
            status=AgentResult.SUCCESS,
            owner_generation=generation,
        )

    assert receiver.physical_ownership_fault is not None
    assert not session.resources_drained()
    assert not session.close()


@pytest.mark.cpu_only
def test_out_of_cohort_writer_result_poison_retains_destination() -> None:
    receiver = _ReceiverProbe()
    session = _make_rx_session(receiver, rid=87)
    session.receive(KVSlice(is_last_slice=True))

    with pytest.raises(RuntimeError, match="outside the sealed cohort"):
        session.process_kv_agent_result(
            peer_rank=99,
            sender_slice_id=0,
            is_last_slice=True,
            status=AgentResult.SUCCESS,
            owner_generation=session._kv_tasks[0].owner_generation,
        )

    assert receiver.physical_ownership_fault is not None
    assert not session.resources_drained()
    assert not session.close()


@pytest.mark.cpu_only
def test_sticky_receive_fault_rejects_later_publication() -> None:
    receiver = _ReceiverProbe()
    first = _make_rx_session(receiver, rid=88)
    first.receive(KVSlice(is_last_slice=True))
    with pytest.raises(RuntimeError, match="owner generation mismatch"):
        first.process_kv_agent_result(
            peer_rank=0,
            sender_slice_id=0,
            is_last_slice=True,
            status=AgentResult.SUCCESS,
            owner_generation=first._kv_tasks[0].owner_generation + 1,
        )

    second = _make_rx_session(receiver, rid=89)
    with pytest.raises(RuntimeError, match="poisoned"):
        second.receive(KVSlice(is_last_slice=True))

    assert second.has_failed()
    assert second.resources_drained()


@pytest.mark.cpu_only
def test_sender_dispatch_rejects_missing_generation_before_pointer_build() -> None:
    rid = 90
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    task._unique_rid = rid
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._build_kv_write_meta = Mock()
    sender._enqueue = Mock()
    info = SimpleNamespace(unique_rid=rid, instance_rank=0, owner_generation=None)

    with pytest.raises(ValueError, match="positive integer owner_generation"):
        sender.dispatch_task(task, {0: info})

    assert task.status == transfer_mod.TaskStatus.ERROR
    sender._build_kv_write_meta.assert_not_called()
    sender._enqueue.assert_not_called()


@pytest.mark.cpu_only
def test_request_data_rejects_invalid_generation_before_saving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._save_peer_req_info = Mock()
    info = SimpleNamespace(unique_rid=91, instance_rank=0, owner_generation=0)
    monkeypatch.setattr(RecvReqInfo, "from_bytes", Mock(return_value=info))

    with pytest.raises(ValueError, match="positive integer owner_generation"):
        sender._respond_with_kv(b"peer", [MessageType.REQUEST_DATA, b"malformed"])

    sender._save_peer_req_info.assert_not_called()


@pytest.mark.cpu_only
def test_receiver_protocol_mismatch_stops_before_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer_info = _make_rank_info(0, sender_endpoints=["tcp://sender"])
    info_messenger = SimpleNamespace(
        send=Mock(),
        receive=Mock(return_value=[peer_info.to_bytes()]),
        stop=Mock(),
    )
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", Mock(return_value=info_messenger))
    receiver = object.__new__(Receiver)
    receiver._enforce_physical_ownership = True
    receiver._sender_ep_instance_map = {}
    receiver._incompatible_peers = {}
    receiver._registrar = SimpleNamespace(
        self_rank_info=_make_rank_info(1),
        self_extractor=SimpleNamespace(page_table=None),
    )
    receiver._get_or_connect_dealer = Mock()

    with pytest.raises(ValueError, match="protocol mismatch"):
        receiver._get_sender_info(SimpleNamespace(ctx_info_endpoint="tcp://info"))

    receiver._get_or_connect_dealer.assert_not_called()
    assert receiver._sender_ep_instance_map == {}


@pytest.mark.cpu_only
def test_receiver_matching_protocol_registers_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer_info = _make_rank_info(1, sender_endpoints=["tcp://sender"])
    info_messenger = SimpleNamespace(
        send=Mock(),
        receive=Mock(return_value=[peer_info.to_bytes()]),
        stop=Mock(),
    )
    registration_dealer = SimpleNamespace(send=Mock())
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", Mock(return_value=info_messenger))
    monkeypatch.setattr(transfer_mod.MambaPolicy, "validate_peer_compatible", Mock())
    receiver = object.__new__(Receiver)
    receiver._enforce_physical_ownership = True
    receiver._sender_ep_instance_map = {}
    receiver._incompatible_peers = {}
    receiver._registrar = SimpleNamespace(
        self_rank_info=_make_rank_info(1),
        self_extractor=SimpleNamespace(page_table=None),
    )
    receiver._get_or_connect_dealer = Mock(return_value=registration_dealer)

    resolved_peer = receiver._get_sender_info(SimpleNamespace(ctx_info_endpoint="tcp://info"))

    registration_dealer.send.assert_called_once()
    assert resolved_peer.physical_ownership_protocol == 1
    assert receiver._sender_ep_instance_map["tcp://info"] is resolved_peer


@pytest.mark.cpu_only
@pytest.mark.parametrize("peer_protocol", [0, True], ids=["legacy", "boolean"])
def test_sender_protocol_mismatch_stops_before_agent_registration(
    monkeypatch: pytest.MonkeyPatch,
    peer_protocol: int | bool,
) -> None:
    sender = object.__new__(Sender)
    sender._shutdown = False
    sender._device_id = 0
    sender._enforce_physical_ownership = True
    sender._registrar = SimpleNamespace(register=Mock())
    sender._agent = SimpleNamespace(load_remote_agent=Mock())
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._loaded_remote_agents = set()
    monkeypatch.setattr(transfer_mod.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transfer_mod.cudart, "cudaSetDevice", Mock(return_value=0))
    monkeypatch.setattr(transfer_mod, "CUASSERT", Mock())

    with pytest.raises(ValueError, match="protocol mismatch"):
        sender._register_peer_rank(
            b"peer",
            [MessageType.REGISTER_RANK_INFO, _make_rank_info(peer_protocol).to_bytes()],
        )

    sender._registrar.register.assert_not_called()
    sender._agent.load_remote_agent.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("schedule_style", "request_id", "error_match"),
    [
        (DisaggScheduleStyle.GENERATION_FIRST, 1 << 40, "context-first"),
        (None, 1 << 40, "context-first"),
        (DisaggScheduleStyle.CONTEXT_FIRST, 17, "global-shaped"),
        (DisaggScheduleStyle.CONTEXT_FIRST, True, "global-shaped"),
        (DisaggScheduleStyle.CONTEXT_FIRST, 1 << 63, "global-shaped"),
    ],
)
def test_phase1_rejects_unqualified_request_before_session_creation(
    schedule_style: DisaggScheduleStyle | None,
    request_id: int | bool,
    error_match: str,
) -> None:
    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(
            disagg_request_id=request_id,
            schedule_style=schedule_style,
        )
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock())

    with pytest.raises(ValueError, match=error_match):
        transceiver.request_and_receive_async(request)

    transceiver._transfer_worker.create_rx_session.assert_not_called()


@pytest.mark.cpu_only
def test_phase1_accepts_context_first_global_shaped_request() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(
            disagg_request_id=1 << 40,
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        )
    )

    transceiver._validate_phase1_request(request)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("peer_overrides", "ctx_dp_rank", "error_match"),
    [
        ({"tp_size": 2}, 0, "tp_size=2"),
        ({"pp_size": 2}, 0, "pp_size=2"),
        ({"cp_size": 2}, 0, "cp_size=2"),
        ({"dp_size": 2}, 0, "dp_size=2"),
        ({"attention": SimpleNamespace(enable_attention_dp=True)}, 0, "attention_dp"),
        ({}, None, "ctx_dp_rank=None"),
        ({}, True, "ctx_dp_rank=True"),
    ],
)
def test_phase1_rejects_remote_topology_before_destination_publication(
    peer_overrides: dict[str, object],
    ctx_dp_rank: object,
    error_match: str,
) -> None:
    peer_info = _make_rank_info(1)
    for name, value in peer_overrides.items():
        setattr(peer_info, name, value)

    with pytest.raises(ValueError, match=error_match):
        transfer_mod._validate_phase1_remote_topology(True, peer_info, ctx_dp_rank)


@pytest.mark.cpu_only
def test_legacy_remote_topology_path_remains_ungated() -> None:
    peer_info = _make_rank_info(0)
    peer_info.tp_size = 8

    transfer_mod._validate_phase1_remote_topology(False, peer_info, None)


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True], ids=["legacy", "phase1"])
def test_transfer_worker_advertises_only_enabled_protocol(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    rank_info = _make_rank_info(0)
    monkeypatch.setattr(
        transfer_mod.RankInfo,
        "from_kv_cache_manager",
        Mock(return_value=rank_info),
    )
    monkeypatch.setattr(TransferWorker, "_setup_peer_infrastructure", lambda _self, _kvm: None)
    monkeypatch.setattr(TransferWorker, "_setup_transfer_engine", lambda _self: None)
    worker = TransferWorker(
        TransferWorkerConfig(
            kv_cache_manager=SimpleNamespace(),
            device_id=0,
            instance_name="worker",
            enforce_physical_ownership=enabled,
        )
    )

    assert worker._rank_info.physical_ownership_protocol == int(enabled)
