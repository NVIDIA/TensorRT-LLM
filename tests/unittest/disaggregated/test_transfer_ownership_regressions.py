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
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVRecvTask,
    MessageType,
    Receiver,
    RxSession,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2


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
        assert self._session.try_begin_transfer(
            task.slice_id,
            set(),
            writer_cohort={0, 1},
        )

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
    receiver._shutdown = True
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

    completed, failed, cancelled = transceiver.check_gen_transfer_status(None)

    assert (completed, failed, cancelled) == ([], [], [])
    assert transceiver._recv_sessions[rid] is session
    assert transceiver._recv_reqs[rid] is request
    session.close.assert_not_called()


@pytest.mark.cpu_only
def test_failed_receive_session_is_retained_until_writers_drain() -> None:
    rid = 87
    request = SimpleNamespace(request_id=rid)
    session = SimpleNamespace(
        _enforce_physical_ownership=True,
        status=SessionStatus.ERROR,
        is_completed=Mock(return_value=False),
        has_failed=Mock(return_value=True),
        wait_complete=Mock(return_value=WaitResult.FAILED),
        resources_drained=Mock(return_value=False),
        close=Mock(return_value=True),
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

    completed, failed, cancelled = transceiver.check_gen_transfer_status(None)

    assert (completed, failed, cancelled) == ([], [], [])
    assert transceiver._recv_sessions[rid] is session
    assert transceiver._recv_reqs[rid] is request
    session.close.assert_not_called()

    session.resources_drained.return_value = True
    completed, failed, cancelled = transceiver.check_gen_transfer_status(None)

    assert (completed, failed, cancelled) == ([], [rid], [])
    assert rid not in transceiver._recv_sessions
    assert rid not in transceiver._recv_reqs
    session.close.assert_called_once_with()


@pytest.mark.cpu_only
def test_failed_receive_consensus_waits_for_every_rank_to_drain() -> None:
    rid = 88
    request = SimpleNamespace(request_id=rid)
    session = SimpleNamespace(
        _enforce_physical_ownership=True,
        status=SessionStatus.ERROR,
        is_completed=Mock(return_value=False),
        has_failed=Mock(return_value=True),
        wait_complete=Mock(return_value=WaitResult.FAILED),
        resources_drained=Mock(return_value=False),
        close=Mock(return_value=True),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = True
    transceiver._mapping = SimpleNamespace(
        pp_size=1,
        enable_attention_dp=False,
        world_size=2,
    )
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: request}
    transceiver._gen_allgather = Mock(
        side_effect=[
            [[], []],
            [
                [[], [rid], [], [rid]],
                [[], [], [], []],
            ],
        ]
    )

    completed, failed, cancelled = transceiver.check_gen_transfer_status(None)

    assert (completed, failed, cancelled) == ([], [], [])
    assert transceiver._recv_sessions[rid] is session
    session.close.assert_not_called()

    session.resources_drained.return_value = True
    transceiver._gen_allgather = Mock(
        side_effect=[
            [[rid], [rid]],
            [
                [[], [rid], [], [rid]],
                [[], [rid], [], [rid]],
            ],
        ]
    )

    completed, failed, cancelled = transceiver.check_gen_transfer_status(None)

    assert (completed, failed, cancelled) == ([], [rid], [])
    assert rid not in transceiver._recv_sessions
    assert rid not in transceiver._recv_reqs
    session.close.assert_called_once_with()


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
def test_partial_publication_failure_quarantines_destination() -> None:
    published_writers: list[int] = []

    class _PartialPublicationReceiver(_ReceiverProbe):
        def dispatch_task(self, task: KVRecvTask) -> None:
            assert self._session is not None
            task.expected_transfers = 2

            def publish_prefix_then_fail() -> None:
                published_writers.append(0)
                raise RuntimeError("writer 1 publication failed after writer 0")

            self._session.try_begin_transfer(
                task.slice_id,
                {"tcp://sender-0", "tcp://sender-1"},
                writer_cohort={0, 1},
                publish=publish_prefix_then_fail,
            )

    receiver = _PartialPublicationReceiver()
    session = _make_rx_session(receiver, rid=86)

    with pytest.raises(RuntimeError, match="writer 1 publication failed"):
        session.receive(KVSlice(is_last_slice=True))

    assert published_writers == [0]
    # The send exception cannot prove which REQUEST_DATA messages reached a
    # writer. Even terminal evidence from one possible writer must not make
    # the destination reusable while the other writer remains unproven.
    session.process_kv_agent_result(
        peer_rank=0,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.FAILED,
    )

    assert session.status == SessionStatus.ERROR
    assert not session.resources_drained()
    assert session.close() is False
    assert receiver.clear_count == 0

    # Avoid a noisy best-effort destructor close for this deliberately
    # quarantined, resource-free unit-test fixture.
    session._closed = True
    receiver._session = None


@pytest.mark.cpu_only
def test_out_of_cohort_writer_cannot_authorize_reuse() -> None:
    receiver = _ReceiverProbe()
    session = _make_rx_session(receiver, rid=82)
    session.receive(KVSlice(is_last_slice=True))

    with pytest.raises(RuntimeError, match="outside the sealed cohort"):
        session.process_kv_agent_result(
            peer_rank=2,
            sender_slice_id=0,
            is_last_slice=True,
            status=AgentResult.SUCCESS,
        )

    assert not session.resources_drained()
    # Invalid evidence intentionally fails closed. Avoid a noisy best-effort
    # destructor close for this hand-built, resource-free unit-test fixture.
    session._closed = True
    receiver._session = None


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
    result_errors: list[Exception] = []
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
    publication_lock = _TrackingLock(race_outcomes)
    session._publication_lock = publication_lock

    def receive() -> None:
        try:
            session.receive(KVSlice(is_last_slice=True))
        except Exception as error:
            receive_errors.append(error)

    def cancel() -> None:
        publication_lock.tracked_thread_id = threading.get_ident()
        try:
            session.cancel()
        except Exception as error:
            cancel_errors.append(error)

    def finish_writer() -> None:
        try:
            session.process_kv_agent_result(
                peer_rank=0,
                sender_slice_id=0,
                is_last_slice=True,
                status=AgentResult.FAILED,
            )
        except Exception as error:
            result_errors.append(error)

    receive_thread = threading.Thread(target=receive, daemon=True)
    cancel_thread = threading.Thread(target=cancel, daemon=True)
    result_thread = threading.Thread(target=finish_writer, daemon=True)
    receive_thread.start()
    try:
        assert request_data_started.wait(timeout=10)
        cancel_thread.start()
        initial_cancel_outcome = race_outcomes.get(timeout=10)
        result_thread.start()
        result_thread.join(timeout=10)
        assert not result_thread.is_alive(), (
            "a blocked REQUEST_DATA send held the session state lock and stalled result handling"
        )
        assert not session.resources_drained(), (
            "terminal writer evidence authorized reuse before REQUEST_DATA publication finished"
        )
    finally:
        finish_request_data.set()
        receive_thread.join(timeout=10)
        cancel_thread.join(timeout=10)
        result_thread.join(timeout=10)

    assert not receive_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert receive_errors == []
    assert cancel_errors == []
    assert result_errors == []
    assert initial_cancel_outcome == "blocked"
    assert session.resources_drained()
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
def test_cancel_request_retains_session_when_close_refuses() -> None:
    rid = 83
    request = SimpleNamespace(
        request_id=rid,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid),
    )
    session = SimpleNamespace(
        cancel=Mock(),
        has_transferring_tasks=Mock(return_value=False),
        close=Mock(return_value=False),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: request}

    assert transceiver.cancel_request(request) is False
    assert transceiver._recv_sessions[rid] is session
    assert transceiver._recv_reqs[rid] is request
    session.close.assert_called_once_with()


@pytest.mark.cpu_only
def test_collect_done_waits_for_physical_drain() -> None:
    rid = 84
    drained = False
    session = SimpleNamespace(
        _enforce_physical_ownership=True,
        is_completed=Mock(return_value=False),
        has_failed=Mock(return_value=True),
        resources_drained=lambda: drained,
    )
    transceiver = object.__new__(KvCacheTransceiverV2)

    assert transceiver._collect_done({rid: session}, {rid: object()}) == ([], [])

    drained = True
    assert transceiver._collect_done({rid: session}, {rid: object()}) == ([], [rid])


@pytest.mark.cpu_only
def test_shutdown_refuses_to_drop_active_receive_owner() -> None:
    rid = 85
    session = SimpleNamespace(
        _enforce_physical_ownership=True,
        resources_drained=Mock(return_value=False),
        close=Mock(return_value=True),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._shutdown = False
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: object()}
    transceiver._transfer_worker = SimpleNamespace(shutdown=Mock())

    with pytest.raises(RuntimeError, match="receive resources remain active"):
        transceiver.shutdown()

    assert not transceiver._shutdown
    assert transceiver._recv_sessions[rid] is session
    session.close.assert_not_called()
    transceiver._transfer_worker.shutdown.assert_not_called()

    session.resources_drained.return_value = True
    transceiver.shutdown()

    assert transceiver._shutdown
    assert transceiver._recv_sessions == {}
    session.close.assert_called_once_with()
    transceiver._transfer_worker.shutdown.assert_called_once_with()
