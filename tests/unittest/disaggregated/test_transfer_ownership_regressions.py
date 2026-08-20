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

import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
import tensorrt_llm._torch.disaggregation.transceiver as transceiver_mod
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVRecvTask,
    MessageType,
    Receiver,
    RxSession,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only


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
        self._session: RxSession | None = None
        self.clear_count = 0
        self.cancel_count = 0

    def setup_session(self, session: RxSession) -> None:
        self._session = session

    def dispatch_task(self, task: KVRecvTask) -> None:
        assert self._session is not None
        task.expected_transfers = 2
        self._session.mark_transferring(task.slice_id)

    def send_cancel_to_senders(self, _unique_rid: int, _sender_endpoints: set[str]) -> None:
        self.cancel_count += 1

    def clear_session(self, _unique_rid: int) -> None:
        self.clear_count += 1


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


def _make_owned_sender() -> transfer_mod.Sender:
    sender = object.__new__(transfer_mod.Sender)
    sender._enforce_physical_ownership = True
    sender._sessions_lock, sender._sessions = threading.Lock(), {}
    sender._shutdown = sender._shutdown_requested = False
    sender._agent_operation_gate = transfer_mod._AgentOperationGate()
    sender._ownership_poisoned, sender._ownership_poison_lock = None, threading.Lock()
    sender._loaded_remote_agents_lock, sender._loaded_remote_agents = threading.Lock(), set()
    return sender


def test_receiver_bridge_ownership_boundaries() -> None:
    params = DisaggregatedParams(
        disagg_request_id=82, schedule_style=DisaggScheduleStyle.GENERATION_FIRST
    )
    receiver, cohorts = _ReceiverProbe(), [{0, 1}, {2, 3}]

    def prepare(rid: int) -> tuple[RxSession, KVRecvTask]:
        params.disagg_request_id = rid
        session = RxSession(rid, params, receiver)
        task = session.prepare_receive(KVSlice(is_last_slice=True))
        assert task is not None
        task.expected_transfers = 2
        return session, task

    session, task = prepare(83)
    assert session.try_begin_transfer(0, set(), candidate_cohorts=cohorts)
    assert session._accept_candidate_writer(0, True)
    aux = session._aux_physical_owner
    assert aux is not None
    assert task._get_physical_owner()._writer_cohort == aux._writer_cohort == {0, 1}
    with pytest.raises(RuntimeError, match="ambiguous ADP writer-cohort"):
        session._accept_candidate_writer(2, True)

    partial_session, _ = prepare(84)
    with pytest.raises(RuntimeError, match="partial publication"):
        partial_session.try_begin_transfer(
            0,
            set(),
            candidate_cohorts=cohorts,
            publish=Mock(side_effect=RuntimeError("partial publication")),
        )
    assert not partial_session.resources_drained()
    partial_session._timeout_s = 0
    assert partial_session.wait_complete(blocking=True) is None

    incompatible_session, incompatible_task = prepare(85)
    incompatible_receiver = object.__new__(Receiver)
    incompatible_receiver._enforce_physical_ownership = True
    incompatible_receiver._build_recv_req_info = Mock()
    incompatible_receiver._get_session = Mock(return_value=incompatible_session)
    incompatible_receiver._get_sender_info = Mock(
        side_effect=transfer_mod.PeerIncompatibleError("incompatible")
    )
    incompatible_receiver.dispatch_task(incompatible_task)
    assert incompatible_task.is_done and incompatible_session.resources_drained()


def test_sender_operation_ownership_covers_ambiguous_and_success_paths() -> None:
    sender = _make_owned_sender()
    sender._agent = SimpleNamespace()
    destructor_gate_counts: list[int] = []

    class _Request:
        def __del__(self) -> None:
            destructor_gate_counts.append(sender._agent_operation_gate._active_transfers)

    sender._agent.submit_transfer_requests = lambda _request: SimpleNamespace(wait=lambda: True)
    succeeded = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=92))
    assert succeeded.begin_physical_operation(7)
    request = _Request()
    assert sender._submit_transfer(succeeded, 7, request) == (True, None)
    del request
    succeeded.finish_physical_operation(7)
    assert destructor_gate_counts == [1] and succeeded.resources_drained

    ambiguous = SimpleNamespace(wait=lambda: False, last_status_str=lambda: "in progress")
    sender._agent.submit_transfer_requests = lambda _request: ambiguous
    retained = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=93))
    assert retained.begin_physical_operation(7)
    assert sender._submit_transfer(retained, 7, Mock()) == (False, "in progress")
    assert not retained.resources_drained and sender._ownership_poisoned is not None

    metadata_entered = threading.Event()

    def mutate_metadata() -> None:
        with sender._agent_operation_gate.metadata():
            metadata_entered.set()

    metadata_thread = threading.Thread(target=mutate_metadata, daemon=True)
    metadata_thread.start()
    with sender._agent_operation_gate._condition:
        assert sender._agent_operation_gate._condition.wait_for(
            lambda: sender._agent_operation_gate._metadata_pending, timeout=10
        )
    assert not metadata_entered.is_set()
    retained.finish_physical_operation(7)
    metadata_thread.join(timeout=10)
    assert metadata_entered.is_set()


def test_sender_duplicate_admission_is_idempotent(monkeypatch) -> None:
    sender = _make_owned_sender()
    rid, peer = 94, 7
    sender._registrar = Mock()
    sender._registrar.get_peer_rank_info.return_value = SimpleNamespace(
        self_endpoint="tcp://receiver"
    )
    sender._num_threads, sender._send_task_queues = 1, [queue.Queue()]
    sender._sessions[rid] = SimpleNamespace(
        lock=threading.Lock(), _closed=False, has_failed=Mock(return_value=False)
    )
    sender._build_aux_write_meta = Mock(return_value=object())
    sender._enqueue = Mock()
    task = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=rid))
    info = SimpleNamespace(instance_name="ctx", instance_rank=peer, unique_rid=rid)

    sender._dispatch_task_to_peer(task, info)
    sender._dispatch_task_to_peer(task, info)

    sender._enqueue.assert_called_once()
    assert sender._send_task_queues[0].empty()

    rejected = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=rid + 1))
    sender._sessions[rid + 1] = SimpleNamespace(
        lock=threading.Lock(), _closed=True, has_failed=Mock(return_value=False)
    )
    sender._dispatch_task_to_peer(
        rejected, SimpleNamespace(instance_name="ctx", instance_rank=peer, unique_rid=rid + 1)
    )
    assert rejected.is_done and rejected.resources_drained
    sender._send_task_queues[0].put(None)
    sender._device_id, sender._thread_local = 0, threading.local()
    dealer = Mock()
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    monkeypatch.setattr(transfer_mod.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transfer_mod.cudart, "cudaSetDevice", Mock())
    monkeypatch.setattr(transfer_mod, "CUASSERT", Mock())
    sender._process_task_queue(0)
    dealer.send.assert_called_once()


def test_stale_request_data_does_not_republish_closed_session(monkeypatch) -> None:
    sender, rid = _make_owned_sender(), 95
    session = SimpleNamespace(lock=threading.Lock(), _closed=False, kv_tasks=[])
    sender._sessions[rid] = session
    sender._save_peer_req_info = Mock()
    sender._send_failed_result_to_receiver = Mock()
    info = SimpleNamespace(unique_rid=rid)
    monkeypatch.setattr(transfer_mod.RecvReqInfo, "from_bytes", Mock(return_value=info))
    captured = threading.Event()
    get_session = sender._get_session

    def get_and_capture(unique_rid):
        result = get_session(unique_rid)
        captured.set()
        return result

    sender._get_session = get_and_capture
    with session.lock:
        worker = threading.Thread(
            target=sender._respond_with_kv, args=(b"", [b"", b"info"]), daemon=True
        )
        worker.start()
        assert captured.wait(timeout=10)
        with sender._sessions_lock:
            sender._sessions.pop(rid)
        session._closed = True
    worker.join(timeout=10)
    assert not sender._save_peer_req_info.called
    sender._send_failed_result_to_receiver.assert_called_once_with(info)


def test_sender_shutdown_waits_for_remote_agent_registration(monkeypatch) -> None:
    sender = _make_owned_sender()
    sender._device_id, sender._registrar, sender._agent = 0, Mock(), Mock()
    sender._messenger, sender._send_task_queues = Mock(), []
    sender._worker_threads, sender._dealers = [], {}
    ri = SimpleNamespace(instance_name="ctx", instance_rank=0, transfer_engine_info=b"descriptor")
    monkeypatch.setattr(transfer_mod.RankInfo, "from_bytes", staticmethod(lambda _data: ri))
    monkeypatch.setattr(transfer_mod.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transfer_mod.cudart, "cudaSetDevice", Mock())
    monkeypatch.setattr(transfer_mod, "CUASSERT", Mock())
    load_started, release_load = threading.Event(), threading.Event()

    def load_remote_agent(*_args) -> None:
        load_started.set()
        assert release_load.wait(timeout=10)

    sender._agent.load_remote_agent.side_effect = load_remote_agent
    release_transfer = sender._agent_operation_gate.acquire_transfer()
    registration = threading.Thread(
        target=sender._register_peer_rank, args=(b"", [b"", b"rank"]), daemon=True
    )
    registration.start()
    with sender._agent_operation_gate._condition:
        assert sender._agent_operation_gate._condition.wait_for(
            lambda: sender._agent_operation_gate._metadata_pending, timeout=10
        )
    assert not load_started.is_set()
    release_transfer()
    assert load_started.wait(timeout=10)
    shutdown = threading.Thread(target=sender.shutdown, daemon=True)
    shutdown.start()
    with sender._agent_operation_gate._condition:
        assert sender._agent_operation_gate._condition.wait_for(
            lambda: sender._agent_operation_gate._closing, timeout=10
        )
    sender._agent.invalidate_remote_agent.assert_not_called()
    release_load.set()
    registration.join(timeout=10)
    shutdown.join(timeout=10)
    sender._agent.invalidate_remote_agent.assert_called_once_with("ctx0")
    assert sender._shutdown and sender._loaded_remote_agents == set()
    with pytest.raises(RuntimeError, match="closing"):
        sender._register_peer_rank(b"", [b"", b"rank"])
    assert sender._agent.load_remote_agent.call_count == 1


def test_transceiver_pairs_requests_before_transfer_admission() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._validate_bridge_req = Mock()
    transceiver._send_sessions, transceiver._send_reqs = {}, {}
    transceiver._recv_sessions, transceiver._recv_reqs = {}, {}
    transceiver._create_kv_slice = Mock(return_value=KVSlice(is_last_slice=True))
    transceiver._slice_num_bytes = Mock(return_value=1)
    transceiver._kv_size_rank_factor = 1
    transceiver._dp_rank, transceiver._context_info_endpoint = 0, "ctx"
    transceiver._transfer_worker = Mock()

    def request(rid: int):
        return Mock(
            py_disaggregated_params=DisaggregatedParams(
                disagg_request_id=rid,
                schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
            ),
            state=None,
        )

    send_req, send_session = request(96), Mock()
    transceiver._transfer_worker.create_tx_session.return_value = send_session

    def observe_send(_slice) -> None:
        assert transceiver._send_sessions[96] is send_session
        assert transceiver._send_reqs[96] is send_req

    send_session.send.side_effect = observe_send
    send_session.send_aux.side_effect = RuntimeError("aux admission")
    with pytest.raises(RuntimeError, match="aux admission"):
        transceiver.respond_and_send_async(send_req)
    assert transceiver._send_reqs[96] is send_req
    send_session.set_exception.assert_called_once()

    recv_req, recv_session = request(97), Mock()
    transceiver._transfer_worker.create_rx_session.return_value = recv_session

    def fail_receive(_slice) -> None:
        assert transceiver._recv_sessions[97] is recv_session
        assert transceiver._recv_reqs[97] is recv_req
        raise RuntimeError("partial publication")

    recv_session.receive.side_effect = fail_receive
    with pytest.raises(RuntimeError, match="partial publication"):
        transceiver.request_and_receive_async(recv_req)
    assert transceiver._recv_reqs[97] is recv_req
    recv_session.fail_admission.assert_called_once()


def test_fp4_mla_bridge_accepts_only_exact_no_retry_profile(monkeypatch) -> None:
    mapping = SimpleNamespace(enable_attention_dp=True, pp_size=1, cp_size=1)
    manager = SimpleNamespace(is_disagg=True, get_fp4_mla_page_table_spec=lambda: None)
    config = SimpleNamespace(kv_cache_bounce_size_mb=0)
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    assert not transceiver_mod._validate_fp4_mla_bridge_profile(mapping, SimpleNamespace(), config)
    assert transceiver_mod._validate_fp4_mla_bridge_profile(mapping, manager, config)
    profile = (mapping, manager, config)
    with monkeypatch.context() as patch:
        patch.setattr(manager, "is_disagg", False)
        assert not transceiver_mod._validate_fp4_mla_bridge_profile(*profile)
    for owner, attribute, value in (
        (mapping, "enable_attention_dp", False),
        (mapping, "pp_size", 2),
        (mapping, "cp_size", 2),
        (config, "kv_cache_bounce_size_mb", 1),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(owner, attribute, value)
            with pytest.raises(ValueError):
                transceiver_mod._validate_fp4_mla_bridge_profile(*profile)
    for name in ("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "TRTLLM_DISAGG_LAYERWISE"):
        with monkeypatch.context() as patch:
            patch.setenv(name, "1")
            with pytest.raises(ValueError):
                transceiver_mod._validate_fp4_mla_bridge_profile(*profile)
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "0")
    with pytest.raises(ValueError):
        transceiver_mod._validate_fp4_mla_bridge_profile(*profile)
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._fp4_mla_bridge_enabled = True
    transceiver._wait_reqs, transceiver._send_sessions = {}, {}
    params = DisaggregatedParams(disagg_request_id=1)
    request = SimpleNamespace(py_disaggregated_params=params)
    for style, rid, synchronous in (
        (DisaggScheduleStyle.CONTEXT_FIRST, 1, False),
        (DisaggScheduleStyle.GENERATION_FIRST, -1, False),
        (DisaggScheduleStyle.GENERATION_FIRST, True, False),
        (DisaggScheduleStyle.GENERATION_FIRST, 1, True),
    ):
        params.schedule_style, params.disagg_request_id = style, rid
        with pytest.raises(ValueError):
            transceiver._validate_bridge_req(request, synchronous=synchronous)
    params.schedule_style, params.disagg_request_id = DisaggScheduleStyle.CONTEXT_FIRST, 1
    with pytest.raises(ValueError):
        transceiver.prepare_context_requests([request])
    assert transceiver._wait_reqs == {}


def test_transceiver_shutdown_refusal_is_retryable() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    session, worker = Mock(), Mock()
    session.resources_drained.side_effect = [False, True]
    session.close.return_value = True
    transceiver._shutdown = False
    transceiver._fp4_mla_bridge_enabled = True
    transceiver._send_sessions, transceiver._recv_sessions = {1: session}, {}
    transceiver._send_reqs, transceiver._recv_reqs = {}, {}
    transceiver._transfer_worker = worker
    with pytest.raises(RuntimeError, match="ownership is active"):
        transceiver.shutdown()
    assert not transceiver._shutdown and not worker.shutdown.called
    transceiver.shutdown()
    assert transceiver._shutdown and worker.shutdown.call_count == 1
