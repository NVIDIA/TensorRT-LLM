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
"""Regressions for Python KV transfer ownership boundaries."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
import tensorrt_llm._torch.disaggregation.transceiver as transceiver_mod
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.bounce.core import TransferContext
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVRecvTask,
    MessageType,
    Receiver,
    RxSession,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
from tensorrt_llm.bindings import DataType
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

    def abort_publication(
        self,
        _rid_slice: tuple[int, int],
        _published_writers: set[int],
    ) -> None:
        return

    def is_bounced(self, _rid_slice: tuple[int, int]) -> bool:
        return False


class _LateReservationBounce(_BounceProbe):
    """Track a reservation created after cancellation already ran cleanup."""

    def __init__(self) -> None:
        super().__init__()
        self.active_reservations: set[tuple[int, int]] = set()

    def release_idle_reservation(self, rid_slice: tuple[int, int]) -> None:
        self.active_reservations.discard(rid_slice)


class _PartialFanInBounce(_BounceProbe):
    """CPU model of one bounced fan-in reservation."""

    def __init__(self) -> None:
        super().__init__()
        self.context: TransferContext | None = None
        self.release_count = 0
        self.scatter_count = 0

    def reserve(self, receiver_req, num_writers: int, *, extra_bytes: int = 0) -> bool:
        del extra_bytes
        self.context = TransferContext(
            rid_slice=(receiver_req.unique_rid, receiver_req.slice_id),
            slot_id=0,
            base_addr=0x1000,
            per_writer_bytes=0x100,
            num_writers=num_writers,
        )
        return True

    def writer_base(self, rid_slice: tuple[int, int], writer_index: int) -> int | None:
        if self.context is None or self.context.rid_slice != rid_slice:
            return None
        return self.context.writer_base(writer_index)

    def is_bounced(self, rid_slice: tuple[int, int]) -> bool:
        return self.context is not None and self.context.rid_slice == rid_slice

    def _advance(self) -> None:
        if self.context is None:
            return
        if self.context.ready_to_scatter():
            self.scatter_count += 1
            self.context.begin_scatter()
            self.context.finish_scatter(True)
        if not self.context.ready_to_settle():
            return
        settlement = self.context.settle()
        self.context = None
        assert settlement is not None
        self.release_count += 1
        if settlement.on_done is not None:
            settlement.on_done(settlement.success)

    def abort_publication(
        self,
        rid_slice: tuple[int, int],
        published_writers: set[int],
    ) -> None:
        assert self.context is not None and self.context.rid_slice == rid_slice
        self.context.abort_publication(published_writers)
        self._advance()

    def record_result(
        self,
        rid_slice: tuple[int, int],
        peer_rank: int,
        dst_ptrs=None,
        sizes=None,
        src_base=None,
        on_done=None,
    ) -> None:
        assert self.context is not None and self.context.rid_slice == rid_slice
        if on_done is not None:
            self.context.on_done = on_done
        self.context.record_writer_result(
            peer_rank,
            succeeded=True,
            src_base=src_base,
            dst_ptrs=dst_ptrs,
            sizes=sizes,
        )
        self._advance()


def _start_checked_thread(
    target: Callable[[], None],
    results: queue.Queue[Exception | None],
) -> threading.Thread:
    """Start a daemon worker and surface expected assertion/runtime failures."""

    def run() -> None:
        try:
            target()
        except (AssertionError, RuntimeError, ValueError) as error:
            results.put(error)
        else:
            results.put(None)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _raise_thread_errors(results: queue.Queue[Exception | None], expected: int) -> None:
    completed = []
    for _ in range(expected):
        try:
            completed.append(results.get_nowait())
        except queue.Empty as error:
            raise AssertionError("worker terminated with an unexpected exception") from error
    if not results.empty():
        raise AssertionError("worker reported more than one result")
    for error in completed:
        if error is not None:
            raise error


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
        self._session.mark_transferring(task.slice_id, writer_cohort={0, 1})

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
    thread_results: queue.Queue[Exception | None] = queue.Queue()
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
        sibling_started.set()
        release_sibling.wait()
        session.process_kv_agent_result(
            peer_rank=1,
            receiver_slice_id=0,
            is_last_slice=True,
            status=AgentResult.SUCCESS,
        )

    sibling_thread = _start_checked_thread(finish_sibling_writer, thread_results)
    try:
        assert sibling_started.wait(timeout=10)

        # Writer 0 is terminal, but writer 1 has not reported a terminal result
        # and may still write to the same receive-side KV allocation.
        session.process_kv_agent_result(
            peer_rank=0,
            receiver_slice_id=0,
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
    _raise_thread_errors(thread_results, expected=1)

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
    transceiver._dist = SimpleNamespace(rank=0)
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
    transceiver._dist = SimpleNamespace(rank=0)
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
    transceiver._dist = SimpleNamespace(rank=0)
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
        receiver_slice_id=0,
        is_last_slice=False,
        status=AgentResult.SUCCESS,
    )

    assert session.status == SessionStatus.TRANSFERRING
    assert not session.resources_drained()


@pytest.mark.cpu_only
def test_gen_first_no_retry_adp_count_seal_waits_for_one_writer_group() -> None:
    rid = 94
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = set()
    receiver._bounce = _BounceProbe()
    receiver._enforce_physical_ownership = True
    receiver._shutdown = True
    receiver_req = SimpleNamespace(
        unique_rid=rid,
        slice_id=0,
        bounce_dst_base=None,
        to_bytes=Mock(return_value=b"receiver-request"),
    )
    receiver._build_recv_req_info = Mock(return_value=receiver_req)
    overlaps = {
        0: SimpleNamespace(ranks=[0, 1]),
        1: SimpleNamespace(ranks=[2, 3]),
    }
    receiver._registrar = SimpleNamespace(
        get_peer_overlap=Mock(side_effect=lambda _peer, dp_rank: overlaps[dp_rank]),
        self_extractor=SimpleNamespace(page_table=None),
        self_rank_info=SimpleNamespace(
            cp_size=1,
            instance_name="gen",
            instance_rank=0,
        ),
    )
    receiver._get_sender_info = Mock(
        return_value=SimpleNamespace(
            sender_endpoints={rank: f"tcp://sender-{rank}" for rank in range(4)},
            page_table=None,
            dp_size=2,
            cp_size=1,
        )
    )
    receiver._request_sender_data = Mock()
    session = RxSession(
        request_id=rid,
        params=DisaggregatedParams(
            disagg_request_id=rid,
            ctx_dp_rank=None,
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        ),
        receiver=receiver,
    )
    task = session.prepare_receive(KVSlice(is_last_slice=True))
    assert task is not None

    session.dispatch_prepared_receive(task)

    assert task.expected_transfers == 2
    assert {call.args[0] for call in receiver._request_sender_data.call_args_list} == {
        f"tcp://sender-{rank}" for rank in range(4)
    }
    assert receiver._request_sender_data.call_count == 4
    session.process_kv_agent_result(
        peer_rank=2,
        receiver_slice_id=0,
        is_last_slice=True,
        status=AgentResult.FAILED,
    )
    assert task.status == transfer_mod.TaskStatus.ERROR
    assert not task.resources_drained

    session.process_kv_agent_result(
        peer_rank=3,
        receiver_slice_id=0,
        is_last_slice=True,
        status=AgentResult.SUCCESS,
    )
    assert task.resources_drained
    assert task.status == transfer_mod.TaskStatus.ERROR
    assert session.close() is True


@pytest.mark.cpu_only
@pytest.mark.parametrize("writer_settles_before_failure", [False, True])
def test_partial_bounced_publication_waits_for_queued_writer_success(
    monkeypatch: pytest.MonkeyPatch,
    writer_settles_before_failure: bool,
) -> None:
    rid = 86
    queued_endpoints: list[str] = []
    bounce = _PartialFanInBounce()
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = set()
    receiver._bounce = bounce
    receiver._enforce_physical_ownership = True
    receiver._shutdown = True
    receiver._dealers = {}
    receiver_req = SimpleNamespace(
        unique_rid=rid,
        slice_id=0,
        mamba_state_index=None,
        bounce_dst_base=None,
        to_bytes=Mock(side_effect=[b"writer-0", b"writer-1"]),
    )
    receiver._build_recv_req_info = Mock(return_value=receiver_req)
    overlap = SimpleNamespace(
        ranks=[0, 1],
        duplicate_head_factor=1,
        overlap_pp_size=1,
    )
    receiver._registrar = SimpleNamespace(
        get_peer_overlap=Mock(return_value=overlap),
        self_extractor=SimpleNamespace(page_table=None),
        self_rank_info=SimpleNamespace(instance_name="gen", instance_rank=0, cp_size=1),
    )
    receiver._get_sender_info = Mock(
        return_value=SimpleNamespace(
            sender_endpoints={0: "tcp://sender-0", 1: "tcp://sender-1"},
            page_table=None,
            cp_size=1,
            dp_size=1,
        )
    )

    def request_sender_data(endpoint: str, _payload: bytes) -> None:
        if endpoint == "tcp://sender-1":
            raise RuntimeError("writer 1 publication failed after writer 0")
        queued_endpoints.append(endpoint)
        if writer_settles_before_failure:
            session.process_kv_agent_result(
                peer_rank=0,
                receiver_slice_id=0,
                is_last_slice=True,
                status=AgentResult.SUCCESS,
                dst_ptrs=np.array([0x2000], dtype=np.int64),
                sizes=np.array([0x100], dtype=np.int64),
                src_base=0x1000,
            )

    receiver._request_sender_data = request_sender_data
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

    with pytest.raises(RuntimeError, match="writer 1 publication failed"):
        session.receive(KVSlice(is_last_slice=True))

    assert queued_endpoints == ["tcp://sender-0"]
    if not writer_settles_before_failure:
        assert bounce.context is not None
        assert bounce.release_count == 0
        assert not session.resources_drained()

        # Only writer 0's REQUEST_DATA was successfully queued. The destination
        # remains owned until that writer reports terminal physical evidence.
        session.process_kv_agent_result(
            peer_rank=0,
            receiver_slice_id=0,
            is_last_slice=True,
            status=AgentResult.SUCCESS,
            dst_ptrs=np.array([0x2000], dtype=np.int64),
            sizes=np.array([0x100], dtype=np.int64),
            src_base=0x1000,
        )

    assert session.status == SessionStatus.ERROR
    assert session.resources_drained()
    assert bounce.context is None
    assert bounce.scatter_count == 0
    assert bounce.release_count == 1
    assert session.close() is True
    assert rid not in receiver._sessions


@pytest.mark.cpu_only
def test_aborted_publication_cannot_complete_during_failure_unwind() -> None:
    published_writers = {0}

    class _FailureWindowReceiver(_ReceiverProbe):
        def dispatch_task(self, task: KVRecvTask) -> None:
            assert self._session is not None
            task.expected_transfers = 1
            original_abort = task.abort_publication

            def abort_then_deliver_success(writers: set[int]) -> None:
                original_abort(writers)
                self._session.process_kv_agent_result(
                    peer_rank=0,
                    receiver_slice_id=0,
                    is_last_slice=True,
                    status=AgentResult.SUCCESS,
                )
                assert task.status == transfer_mod.TaskStatus.TRANSFERRING

            task.abort_publication = abort_then_deliver_success

            def fail_after_publication() -> None:
                raise RuntimeError("publication failed after writer 0 was queued")

            self._session.try_begin_transfer(
                task.slice_id,
                {"tcp://sender-0"},
                writer_cohort={0},
                publish=fail_after_publication,
                published_writers=published_writers,
            )

    receiver = _FailureWindowReceiver()
    session = _make_rx_session(receiver, rid=91)

    with pytest.raises(RuntimeError, match="publication failed"):
        session.receive(KVSlice(is_last_slice=True))

    assert session.status == SessionStatus.ERROR
    assert session.resources_drained()
    assert session.close() is True


@pytest.mark.cpu_only
def test_out_of_cohort_writer_cannot_authorize_reuse() -> None:
    receiver = _ReceiverProbe()
    session = _make_rx_session(receiver, rid=82)
    session.receive(KVSlice(is_last_slice=True))

    with pytest.raises(RuntimeError, match="outside the sealed cohort"):
        session.process_kv_agent_result(
            peer_rank=2,
            receiver_slice_id=0,
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
    initial_cancel_outcome: str | None = None
    thread_results: queue.Queue[Exception | None] = queue.Queue()

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
        self_rank_info=SimpleNamespace(cp_size=1),
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
        session.receive(KVSlice(is_last_slice=True))

    def cancel() -> None:
        publication_lock.tracked_thread_id = threading.get_ident()
        session.cancel()

    def finish_writer() -> None:
        session.process_kv_agent_result(
            peer_rank=0,
            receiver_slice_id=0,
            is_last_slice=True,
            status=AgentResult.FAILED,
        )

    receive_thread = _start_checked_thread(receive, thread_results)
    cancel_thread = None
    result_thread = None
    try:
        assert request_data_started.wait(timeout=10)
        cancel_thread = _start_checked_thread(cancel, thread_results)
        initial_cancel_outcome = race_outcomes.get(timeout=10)
        result_thread = _start_checked_thread(finish_writer, thread_results)
        result_thread.join(timeout=10)
        assert not result_thread.is_alive()
        assert not session.resources_drained(), (
            "terminal writer evidence authorized reuse before REQUEST_DATA publication finished"
        )
    finally:
        finish_request_data.set()
        receive_thread.join(timeout=10)
        if cancel_thread is not None:
            cancel_thread.join(timeout=10)
        if result_thread is not None:
            result_thread.join(timeout=10)

    assert not receive_thread.is_alive()
    assert cancel_thread is not None and not cancel_thread.is_alive()
    _raise_thread_errors(thread_results, expected=3)
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
def test_failed_session_is_not_reported_retired_when_close_refuses() -> None:
    rid = 89
    initial_state = object()
    request = SimpleNamespace(state=initial_state)
    session = SimpleNamespace(
        _enforce_physical_ownership=True,
        resources_drained=Mock(return_value=True),
        close=Mock(return_value=False),
    )
    sessions = {rid: session}
    requests = {rid: request}
    failed = [rid]
    transceiver = object.__new__(KvCacheTransceiverV2)

    with pytest.raises(RuntimeError, match="session close refused"):
        transceiver._close_failed_sessions(sessions, requests, failed)

    assert failed == [rid]
    assert request.state is initial_state
    assert sessions == {rid: session}
    assert requests == {rid: request}

    session.close.return_value = True
    failed = [rid]
    transceiver._close_failed_sessions(sessions, requests, failed)

    assert failed == [rid]
    assert request.state is not initial_state
    assert sessions == {}
    assert requests == {}


@pytest.mark.cpu_only
def test_completed_session_is_not_reported_retired_when_close_refuses() -> None:
    rid = 90
    initial_state = object()
    request = SimpleNamespace(
        state=initial_state,
        py_kv_cache_xfer_bytes=0,
        set_kv_cache_size=Mock(),
    )
    session = SimpleNamespace(
        status=SessionStatus.KV_TRANSFERRED,
        transfer_end_time=None,
        kv_cache_size_bytes=0,
        is_completed=Mock(return_value=True),
        has_failed=Mock(return_value=False),
        wait_complete=Mock(return_value=WaitResult.COMPLETED),
        close=Mock(return_value=False),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._mapping = SimpleNamespace(pp_size=1, enable_attention_dp=False, world_size=1)
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: request}
    transceiver._gen_allgather = Mock()
    transceiver._gen_consensus = Mock(side_effect=lambda rids: rids)
    transceiver._need_aux_transfer = Mock(return_value=False)
    transceiver._assert_disagg_history_declared = Mock()

    with pytest.raises(RuntimeError, match="session close refused"):
        transceiver.check_gen_transfer_status(None)

    assert request.state is initial_state
    assert transceiver._recv_sessions == {rid: session}
    assert transceiver._recv_reqs == {rid: request}

    session.close.return_value = True
    completed, failed, cancelled = transceiver.check_gen_transfer_status(None)

    assert (completed, failed, cancelled) == ([rid], [], [])
    assert request.state is not initial_state
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}


@pytest.mark.cpu_only
def test_cancelled_session_close_refusal_fails_stop_after_consensus() -> None:
    rid = 92
    request = SimpleNamespace(state=object())
    session = SimpleNamespace(
        status=SessionStatus.CANCELLED,
        is_completed=Mock(return_value=False),
        has_failed=Mock(return_value=True),
        wait_complete=Mock(return_value=WaitResult.FAILED),
        close=Mock(return_value=False),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: request}
    transceiver._gen_allgather = Mock()
    transceiver._gen_consensus = Mock(side_effect=lambda rids: rids)

    with pytest.raises(RuntimeError, match="session close refused"):
        transceiver.check_gen_transfer_status(None)

    assert transceiver._recv_sessions == {rid: session}
    assert transceiver._recv_reqs == {rid: request}

    session.close.return_value = True
    completed, failed, cancelled = transceiver.check_gen_transfer_status(None)

    assert completed == []
    assert failed == []
    assert cancelled == [request]
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}


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
    transceiver._shutdown_complete = False
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: object()}
    transceiver._transfer_worker = SimpleNamespace(shutdown=Mock())

    with pytest.raises(RuntimeError, match="physical resources remain active"):
        transceiver.shutdown()

    assert not transceiver._shutdown_complete
    assert transceiver._recv_sessions[rid] is session
    session.close.assert_not_called()
    transceiver._transfer_worker.shutdown.assert_not_called()

    session.resources_drained.return_value = True
    transceiver.shutdown()

    assert transceiver._shutdown_complete
    assert transceiver._recv_sessions == {}
    session.close.assert_called_once_with()
    transceiver._transfer_worker.shutdown.assert_called_once_with()


@pytest.mark.cpu_only
def test_shutdown_fails_stop_when_receive_close_refuses_after_preflight() -> None:
    rid = 93
    send_session = SimpleNamespace(close=Mock(return_value=None))
    recv_session = SimpleNamespace(
        _enforce_physical_ownership=True,
        resources_drained=Mock(return_value=True),
        close=Mock(return_value=False),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._shutdown_complete = False
    transceiver._send_sessions = {rid + 1: send_session}
    transceiver._send_reqs = {rid + 1: object()}
    transceiver._recv_sessions = {rid: recv_session}
    transceiver._recv_reqs = {rid: object()}
    transceiver._transfer_worker = SimpleNamespace(shutdown=Mock())

    with pytest.raises(RuntimeError, match="session close refused"):
        transceiver.shutdown()

    assert not transceiver._shutdown_complete
    assert transceiver._recv_sessions == {rid: recv_session}
    assert transceiver._send_sessions == {rid + 1: send_session}
    send_session.close.assert_not_called()
    transceiver._transfer_worker.shutdown.assert_not_called()

    recv_session.close.return_value = True
    transceiver.shutdown()

    assert transceiver._shutdown_complete
    assert transceiver._recv_sessions == {}
    assert transceiver._send_sessions == {}
    send_session.close.assert_called_once_with()
    transceiver._transfer_worker.shutdown.assert_called_once_with()


def _make_owned_sender(
    operation_drain_timeout_s: float | None = None,
) -> transfer_mod.Sender:
    sender = object.__new__(transfer_mod.Sender)
    sender._enforce_physical_ownership = True
    sender._sessions_lock, sender._sessions = threading.Lock(), {}
    sender._shutdown = sender._shutdown_requested = False
    sender._agent_operation_gate = transfer_mod._AgentOperationGate(operation_drain_timeout_s)
    sender._ownership_poisoned, sender._ownership_poison_lock = None, threading.Lock()
    sender._loaded_remote_agents_lock, sender._loaded_remote_agents = threading.Lock(), set()
    sender._instance_rank = 0
    return sender


@pytest.mark.cpu_only
def test_sender_ignores_remote_agent_registration_after_shutdown_request(
    monkeypatch,
) -> None:
    sender = _make_owned_sender()
    sender._shutdown_requested = True
    sender._device_id, sender._registrar, sender._agent = 0, Mock(), Mock()
    set_device, cuda_set_device, cuassert, decode = Mock(), Mock(), Mock(), Mock()
    monkeypatch.setattr(transfer_mod.torch.cuda, "set_device", set_device)
    monkeypatch.setattr(transfer_mod.cudart, "cudaSetDevice", cuda_set_device)
    monkeypatch.setattr(transfer_mod, "CUASSERT", cuassert)
    monkeypatch.setattr(transfer_mod.RankInfo, "from_bytes", decode)

    sender._register_peer_rank(b"", [b"", b"rank"])

    set_device.assert_not_called()
    cuda_set_device.assert_not_called()
    cuassert.assert_not_called()
    decode.assert_not_called()
    sender._registrar.register.assert_not_called()
    sender._agent.load_remote_agent.assert_not_called()


@pytest.mark.cpu_only
def test_sender_retires_only_after_done_and_retains_ambiguous_operation() -> None:
    sender = _make_owned_sender()
    done_status = SimpleNamespace(wait=lambda: True)
    sender._agent = SimpleNamespace(submit_transfer_requests=lambda _request: done_status)
    done = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=92))
    done_request = Mock()

    assert done.begin_physical_operation(7)
    assert sender._submit_transfer(done, 7, done_request) == (True, None)
    assert not done.resources_drained
    assert done._physical_ops[7][:2] == [done_request, done_status]
    done.finish_physical_operation(7)
    assert done.resources_drained

    ambiguous_status = SimpleNamespace(
        wait=lambda: False,
        last_status_str=lambda: "still active",
    )
    sender._agent = SimpleNamespace(submit_transfer_requests=lambda _request: ambiguous_status)
    ambiguous = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=93))
    ambiguous_request = Mock()

    assert ambiguous.begin_physical_operation(7)
    assert sender._submit_transfer(ambiguous, 7, ambiguous_request) == (
        False,
        "still active",
    )
    assert sender._ownership_poisoned is not None
    assert not ambiguous.resources_drained
    assert ambiguous._physical_ops[7][:2] == [ambiguous_request, ambiguous_status]
    assert sender._agent_operation_gate._active_transfers == 1
    with pytest.raises(RuntimeError, match="quarantined"):
        with sender._agent_operation_gate.metadata():
            pass
    with pytest.raises(RuntimeError, match="refuses teardown"):
        sender._agent_operation_gate.close()
    assert sender._agent_operation_gate._active_transfers == 1


@pytest.mark.cpu_only
def test_sender_gate_timeout_retires_only_unsubmitted_operation() -> None:
    sender = _make_owned_sender(operation_drain_timeout_s=0.0)
    sender._agent = Mock()
    task = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=95))
    request = Mock()

    assert task.begin_physical_operation(7)
    with sender._agent_operation_gate.metadata():
        with pytest.raises(
            transfer_mod._TransferNotSubmittedError,
            match="before backend submission",
        ):
            sender._submit_transfer(task, 7, request)

    sender._agent.submit_transfer_requests.assert_not_called()
    assert sender._ownership_poisoned is not None
    assert task.resources_drained


@pytest.mark.cpu_only
def test_agent_gate_timeouts_retain_active_transfer() -> None:
    metadata_gate = transfer_mod._AgentOperationGate(drain_timeout_s=0.0)
    release_metadata = metadata_gate.acquire_transfer()

    with pytest.raises(RuntimeError, match="quarantined"):
        with metadata_gate.metadata():
            pass

    assert not metadata_gate._metadata_pending
    assert metadata_gate._active_transfers == 1
    release_metadata()
    assert metadata_gate._active_transfers == 0
    with pytest.raises(RuntimeError, match="quarantined"):
        metadata_gate.acquire_transfer()

    close_gate = transfer_mod._AgentOperationGate(drain_timeout_s=0.0)
    release_close = close_gate.acquire_transfer()

    with pytest.raises(RuntimeError, match="refuses teardown"):
        close_gate.close()

    assert close_gate._active_transfers == 1
    release_close()
    with pytest.raises(RuntimeError, match="quarantined"):
        close_gate.acquire_transfer()


@pytest.mark.cpu_only
def test_gen_first_aux_owner_uses_anonymous_no_retry_writer_count() -> None:
    receiver = _ReceiverProbe()
    session = RxSession(
        request_id=94,
        params=DisaggregatedParams(
            disagg_request_id=94,
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        ),
        receiver=receiver,
    )
    task = session.prepare_receive(KVSlice(is_last_slice=True))
    assert task is not None
    task.expected_transfers = 2
    published_writers: set[int] = set()

    def publish() -> None:
        published_writers.update({0, 1, 2, 3})

    assert session.try_begin_transfer(
        task.slice_id,
        set(),
        writer_cohort=None,
        publish=publish,
        published_writers=published_writers,
    )
    aux_owner = session._aux_physical_owner
    assert aux_owner is not None
    assert aux_owner._writer_cohort is None
    assert not aux_owner.resources_drained

    session.process_aux_agent_result(2, AgentResult.FAILED)
    assert not aux_owner.resources_drained
    session.process_aux_agent_result(3, AgentResult.SUCCESS)
    assert aux_owner.resources_drained


@pytest.mark.cpu_only
def test_fp4_mla_bridge_uses_production_cache_layout(monkeypatch) -> None:
    mapping = SimpleNamespace(enable_attention_dp=True, pp_size=1, cp_size=1)
    manager = object.__new__(KVCacheManagerV2)
    manager.is_disagg = True
    manager.dtype = DataType.NVFP4
    manager.kv_cache_type = CacheTypeCpp.SELFKONLY
    config = SimpleNamespace(
        backend="NIXL",
        transceiver_runtime="PYTHON",
        kv_transfer_timeout_ms=60_000,
        kv_cache_bounce_size_mb=0,
    )
    profile = (mapping, manager, config)

    monkeypatch.delenv("TRTLLM_DISAGG_NO_RETRY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    monkeypatch.delenv("TRTLLM_DISAGG_LAYERWISE", raising=False)
    assert not transceiver_mod._validate_fp4_mla_bridge_profile(*profile)
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    assert transceiver_mod._validate_fp4_mla_bridge_profile(*profile)

    for attribute, value in (
        ("is_disagg", False),
        ("dtype", DataType.BF16),
        ("kv_cache_type", CacheTypeCpp.SELF),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(manager, attribute, value)
            assert not transceiver_mod._validate_fp4_mla_bridge_profile(*profile)

    assert not transceiver_mod._validate_fp4_mla_bridge_profile(
        mapping,
        SimpleNamespace(
            is_disagg=True,
            dtype=DataType.NVFP4,
            kv_cache_type=CacheTypeCpp.SELFKONLY,
        ),
        config,
    )
    with monkeypatch.context() as patch:
        patch.setattr(mapping, "enable_attention_dp", False)
        with pytest.raises(ValueError, match="no-retry ADP"):
            transceiver_mod._validate_fp4_mla_bridge_profile(*profile)
