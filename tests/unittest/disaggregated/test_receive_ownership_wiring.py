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
"""CPU-only wiring tests for native receive ownership."""

from __future__ import annotations

import gc
import threading
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import (
    KVSlice,
    SessionArgsBase,
    SessionStatus,
    WaitResult,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import IdentityMapper
from tensorrt_llm._torch.disaggregation.native.peer import PeerOverlap
from tensorrt_llm._torch.disaggregation.native.receive_lifecycle import (
    LogicalState,
    PhysicalState,
    RecvTransferRegistry,
)
from tensorrt_llm._torch.disaggregation.native.transfer import (
    _KV_RESULT_PREFIX,
    _NON_DRAINED_TRANSFER_WORKERS,
    AgentResult,
    AuxSendTask,
    KVRecvTask,
    KVSendTask,
    MessageType,
    RankInfoServer,
    Receiver,
    RecvReqInfo,
    RxSession,
    Sender,
    SendOperationState,
    TaskStatus,
    TransferSourceInDoubtError,
    TransferWorker,
    TxSession,
    WriteMeta,
    WriteMetaType,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle

REQUEST_ID = 101
WRITER_RANK = 3


def _recv_info(*, writer_rank: int = WRITER_RANK, slice_id: int = 0) -> RecvReqInfo:
    return RecvReqInfo(
        sender_req_id=REQUEST_ID,
        instance_name="receiver",
        instance_rank=writer_rank,
        block_ids_per_layer_groups=[],
        unique_rid=REQUEST_ID,
        aux_slot=7,
        slice_id=slice_id,
    )


def _make_send_task_and_session(*, channel: str = "kv"):
    params = _params()
    session = SimpleNamespace(
        lock=threading.Lock(),
        kv_tasks=[],
        aux_task=None,
        status=SessionStatus.READY,
        disagg_request_id=REQUEST_ID,
        source_owner=object(),
        _accepting_operations=True,
        _closed=False,
    )
    if channel == "kv":
        task = KVSendTask(KVSlice(), params, 0, session=session)
        session.kv_tasks.append(task)
    else:
        task = AuxSendTask(params, 7, session=session)
        session.aux_task = task
    task._unique_rid = REQUEST_ID
    return task, session


def _make_owned_send_task_and_session(*, channel: str, expected_transfers: int):
    sender = _make_sender_for_shutdown()
    sender._shutdown_complete = True
    sender.setup_session = Mock()
    sender.cancel_session = Mock()
    sender.retry_terminal_results = Mock()
    sender._send_operation_message = Mock(return_value=True)
    sender._enqueue_owned = Mock()
    sender._instance_rank = 0
    sender._bounce = Mock()

    params = _params()
    session = TxSession(REQUEST_ID, params, sender, source_owner=object())
    session.receiver_ready = True
    if channel == "kv":
        task = KVSendTask(KVSlice(), params, 0, session=session)
        session.kv_tasks.append(task)
    else:
        completed_kv_task = KVSendTask(KVSlice(), params, 0, session=session)
        assert completed_kv_task.complete()
        session.kv_tasks.append(completed_kv_task)
        task = AuxSendTask(params, 7, session=session)
        session.aux_task = task
        session._need_aux = True
    task._unique_rid = REQUEST_ID

    write_meta = WriteMeta(
        task=task,
        expected_transfers=expected_transfers,
        peer_name="receiver1",
        peer_rank=1,
        peer_endpoint="tcp://receiver",
        unique_rid=REQUEST_ID,
        src_ptrs=np.array([], dtype=np.int64),
        dst_ptrs=np.array([], dtype=np.int64),
        sizes=np.array([], dtype=np.int64),
        slice_id=0 if channel == "kv" else None,
        meta_type=WriteMetaType.KV if channel == "kv" else WriteMetaType.AUX,
        session=session,
    )
    if channel == "kv":
        sender._build_kv_write_meta = Mock(return_value=write_meta)
    else:
        sender._build_aux_write_meta = Mock(return_value=write_meta)
    return sender, session, task, write_meta


def _make_sender_for_shutdown() -> Sender:
    sender = object.__new__(Sender)
    sender._shutdown_attempt_lock = threading.Lock()
    sender._operation_admission_lock = threading.RLock()
    sender._dealer_admission_closed = False
    sender._shutdown = False
    sender._shutdown_complete = False
    sender._listener_stopped = True
    sender._shutdown_sentinels_sent = False
    sender._messenger = Mock()
    sender._send_task_queues = []
    sender._worker_threads = []
    sender._failed_thread_dealers = []
    sender._failed_thread_dealers_lock = threading.Lock()
    sender._in_doubt_transfers = []
    sender._in_doubt_transfers_lock = threading.Lock()
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._pre_cancelled_rids = set()
    sender._pre_session_terminal_results = {}
    sender._pre_session_terminal_results_lock = threading.Lock()
    sender._pre_session_terminal_retry_lock = threading.Lock()
    sender._next_pre_session_terminal_retry_at = 0.0
    sender._peer_requests = {}
    sender._peer_requests_timestamps = {}
    sender._peer_requests_lock = threading.Lock()
    sender._loaded_remote_agents = set()
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._agent = Mock()
    sender._dealers = {}
    sender._dealers_lock = threading.Lock()
    sender._instance_rank = 0
    return sender


class _FakeBounce:
    enabled = False

    def __init__(self) -> None:
        self.logical_failure_callbacks: list = []
        self.protocol_conflict_callbacks: list = []
        self.backend_quiesced_callbacks: list = []
        self.release_calls: list[tuple[int, int]] = []

    def is_bounced(self, _key) -> bool:
        return False

    def reserve(self, _receiver_req, _writer_ranks, **_kwargs) -> bool:
        return False

    def release_idle_reservation(self, key) -> None:
        self.release_calls.append(key)

    def mark_logical_failure(self, key, on_done) -> None:
        self.logical_failure_callbacks.append((key, on_done))

    def mark_protocol_conflict(self, key, on_done) -> None:
        self.protocol_conflict_callbacks.append((key, on_done))

    def mark_backend_quiesced(self, key, on_done) -> None:
        self.backend_quiesced_callbacks.append((key, on_done))

    def retry_settlements(self, _scope=None) -> bool:
        return True


class _BlockingBounce(_FakeBounce):
    enabled = True

    def __init__(self) -> None:
        super().__init__()
        self.reserve_started = threading.Event()
        self.allow_reserve = threading.Event()
        self._state_lock = threading.Lock()
        self._reservation_active = False
        self.released_active_reservations: list[tuple[int, int]] = []

    def reserve(self, _receiver_req, _writer_ranks, **_kwargs) -> bool:
        self.reserve_started.set()
        assert self.allow_reserve.wait(timeout=2)
        with self._state_lock:
            self._reservation_active = True
        return True

    def release_idle_reservation(self, key) -> None:
        super().release_idle_reservation(key)
        with self._state_lock:
            if self._reservation_active:
                self._reservation_active = False
                self.released_active_reservations.append(key)


class _FailOnceLogicalFailureBounce(_FakeBounce):
    def __init__(self) -> None:
        super().__init__()
        self.logical_failure_attempts: list[tuple[int, int]] = []

    def mark_logical_failure(self, key, on_done) -> None:
        self.logical_failure_attempts.append(key)
        if len(self.logical_failure_attempts) == 1:
            raise RuntimeError("injected cancellation failure")
        super().mark_logical_failure(key, on_done)


def _params(
    *,
    request_id: int = REQUEST_ID,
    ctx_dp_rank: int | None = 0,
    schedule_style: DisaggScheduleStyle = DisaggScheduleStyle.CONTEXT_FIRST,
) -> DisaggregatedParams:
    return DisaggregatedParams(
        disagg_request_id=request_id,
        ctx_request_id=request_id,
        ctx_dp_rank=ctx_dp_rank,
        schedule_style=schedule_style,
    )


def _make_receiver(*, bounce=None, writer_ranks=(WRITER_RANK,)) -> Receiver:
    receiver = object.__new__(Receiver)
    receiver._shutdown = True  # Keep object.__new__ fixtures out of __del__ cleanup.
    receiver._shutdown_started = False
    receiver._shutdown_attempt_lock = threading.Lock()
    receiver._dealer_admission_open = True
    receiver._dealers = {}
    receiver._dealers_lock = threading.Lock()
    receiver._sessions = {}
    receiver._sessions_lock = threading.Lock()
    receiver._pre_cancelled_rids = set()
    receiver._recv_registry = RecvTransferRegistry()
    receiver._bounce_lifecycle_delivery_lock = threading.Lock()
    receiver._pending_bounce_lifecycle_deliveries = {}
    receiver._pending_bounce_logical_failure_deliveries = {}
    receiver._bounce = bounce or _FakeBounce()
    receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="receiver", instance_rank=0),
        get_peer_overlap=lambda _peer_info, _dp_rank: PeerOverlap(
            overlap_pp_size=1,
            duplicate_head_factor=1,
            ranks=list(writer_ranks),
        ),
    )
    peer_info = SimpleNamespace(
        dp_size=1,
        sender_endpoints=[f"tcp://sender-{rank}" for rank in range(max(writer_ranks) + 1)],
    )
    receiver._get_sender_info = lambda _params: peer_info
    receiver._build_recv_req_info = lambda task: RecvReqInfo(
        sender_req_id=REQUEST_ID,
        instance_name="receiver",
        instance_rank=0,
        block_ids_per_layer_groups=[],
        unique_rid=task._unique_rid,
        aux_slot=task._aux_slot,
        slice_id=task.slice_id,
    )
    receiver._request_sender_data = Mock()
    receiver.send_cancel_to_senders = Mock()
    # Most object.__new__ fixtures intentionally omit the real registrar's
    # page-table/mapping metadata. Tests exercising bounce admission opt in to
    # a proven single-writer map unless they explicitly test proof failure.
    receiver._single_writer_bounce_exact = Mock(return_value=True)
    return receiver


def test_destination_intervals_use_view_index_for_sparse_physical_pool() -> None:
    receiver = object.__new__(Receiver)
    page_table = KVCachePageTable(
        tokens_per_block=32,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                pool_views=[
                    PoolView(
                        pool_idx=1,
                        buffer_entries=np.empty(0, dtype=BUFFER_ENTRY_DTYPE),
                    )
                ],
            )
        ],
        pool_groups=[
            PhysicalPoolGroup(
                pools=[
                    PhysicalPool(base_address=0x1000, slot_bytes=8, num_slots=16),
                    PhysicalPool(base_address=0x4000, slot_bytes=16, num_slots=16),
                ]
            )
        ],
    )
    extractor = KVRegionExtractorV1(page_table)
    receiver._registrar = SimpleNamespace(self_extractor=extractor)
    task = SimpleNamespace(
        _kv_slice=SimpleNamespace(
            block_ids_per_layer_groups=[[7]],
            mamba_state_index=None,
        )
    )

    intervals = receiver._destination_intervals(task)

    assert intervals == {(0x4070, 16)}


def _make_legacy_session(cohorts) -> tuple[RxSession, KVRecvTask]:
    params = _params(schedule_style=DisaggScheduleStyle.GENERATION_FIRST)
    task = KVRecvTask(REQUEST_ID, KVSlice(), 0, params, aux_slot=None)
    task.set_valid_writer_cohorts(cohorts)
    for rank in {rank for cohort in cohorts for rank in cohort}:
        task.mark_writer_exposed(rank)
    task.close_publication()
    receiver = _make_receiver()
    session = object.__new__(RxSession)
    session._closed = True  # Keep object.__new__ fixtures out of __del__ cleanup.
    session._base_args = SessionArgsBase(params)
    session._receiver = receiver
    session.request_id = REQUEST_ID
    session.lock = threading.Lock()
    session.aux_slot = 7
    session._kv_tasks = [task]
    session._need_aux = True
    session._aux_results = {}
    session._aux_exposed_writer_ranks = set(task._exposed_writer_ranks)
    session._aux_publication_closed = True
    session._aux_result_conflict = False
    session._aux_drained = False
    session._aux_status = TaskStatus.INIT
    session._terminal_status = None
    session._exception = None
    session._sender_endpoints = set()
    session._selected_writer_cohort = None
    return session, task


@pytest.mark.parametrize("channel", ["kv", "aux"])
@pytest.mark.parametrize(
    "cohorts,results,expected_history,expected_conflict,expected_status",
    [
        (
            ((1, 2),),
            ((1, AgentResult.SUCCESS), (2, AgentResult.SUCCESS)),
            (False, True),
            False,
            TaskStatus.TRANSFERRED,
        ),
        (
            ((1, 2), (3, 4)),
            (
                (3, AgentResult.SUCCESS),
                (4, AgentResult.SUCCESS),
            ),
            (False, True),
            False,
            TaskStatus.TRANSFERRED,
        ),
        (
            ((1, 2),),
            ((9, AgentResult.SUCCESS),),
            (False,),
            True,
            TaskStatus.ERROR,
        ),
        (
            ((1, 2), (3, 4)),
            ((1, AgentResult.SUCCESS), (3, AgentResult.SUCCESS)),
            (False, False),
            True,
            TaskStatus.ERROR,
        ),
        (
            ((1, 2), (3, 4)),
            (
                (3, AgentResult.SUCCESS),
                (1, AgentResult.SUCCESS),
                (2, AgentResult.SUCCESS),
            ),
            (False, False, False),
            True,
            TaskStatus.ERROR,
        ),
        (
            ((1, 2),),
            (
                (1, AgentResult.SUCCESS),
                (1, AgentResult.SUCCESS),
                (2, AgentResult.SUCCESS),
            ),
            (False, False, True),
            False,
            TaskStatus.TRANSFERRED,
        ),
        (
            ((1, 2),),
            ((1, AgentResult.SUCCESS), (1, AgentResult.FAILED)),
            (False, False),
            True,
            TaskStatus.ERROR,
        ),
        (
            ((1, 2),),
            ((1, AgentResult.FAILED), (2, AgentResult.SUCCESS)),
            (False, True),
            False,
            TaskStatus.ERROR,
        ),
    ],
    ids=[
        "deterministic-valid",
        "legacy-adp-selected-cohort",
        "unexpected-writer",
        "partial-results-across-adp-cohorts",
        "complete-cohort-plus-partial-sibling",
        "idempotent-duplicate",
        "conflicting-duplicate",
        "failure-then-late-sibling",
    ],
)
def test_legacy_kv_and_aux_follow_selected_candidate_cohort(
    channel,
    cohorts,
    results,
    expected_history,
    expected_conflict,
    expected_status,
) -> None:
    session, task = _make_legacy_session(cohorts)
    drained_history = []

    for rank, status in results:
        if channel == "kv":
            session.process_kv_agent_result(rank, 0, True, status)
            drained_history.append(task.legacy_resources_drained)
        else:
            session.process_aux_agent_result(rank, status)
            drained_history.append(session._aux_drained)

    assert tuple(drained_history) == expected_history
    if channel == "kv":
        assert task._legacy_result_conflict is expected_conflict
        assert task.status is expected_status
    else:
        assert session._aux_result_conflict is expected_conflict
        assert session._aux_status is expected_status


def test_aux_protocol_conflict_requires_full_exposed_ledger_or_backend_fence() -> None:
    session, _task = _make_legacy_session(((1, 2),))

    assert not session._mark_aux_writer_exposed_locked(9)
    assert 9 not in session._aux_exposed_writer_ranks

    session.process_aux_protocol_conflict(9, "unknown result status")
    session.process_aux_agent_result(1, AgentResult.SUCCESS)
    session.process_aux_agent_result(2, AgentResult.SUCCESS)

    assert session._selected_writer_cohort == frozenset({1, 2})
    assert session._aux_results.keys() == {1, 2}
    assert not session._aux_drained

    session.mark_backend_quiesced()

    assert session._aux_drained
    assert session._aux_status is TaskStatus.ERROR


def test_aux_protocol_conflict_reopens_initially_drained_exposure_ledger() -> None:
    session, _task = _make_legacy_session(((1, 2),))
    session._aux_exposed_writer_ranks.clear()
    session._aux_drained = True

    session.process_aux_protocol_conflict(9, "unknown result status")

    assert session._aux_exposed_writer_ranks == {9}
    assert not session._aux_drained

    session.mark_backend_quiesced()
    assert session._aux_drained


def test_aux_protocol_conflict_before_first_receive_requires_backend_fence() -> None:
    receiver = _make_receiver()
    aux_buffer = SimpleNamespace(
        alloc_slot=Mock(return_value=SimpleNamespace(id=7)),
        free_slot=Mock(),
    )
    session = RxSession(
        REQUEST_ID,
        _params(schedule_style=DisaggScheduleStyle.GENERATION_FIRST),
        receiver,
        aux_buffer=aux_buffer,
    )

    assert session._aux_drained

    session.process_aux_protocol_conflict(9, "unknown result status")

    assert session._aux_exposed_writer_ranks == {9}
    assert not session._aux_drained
    assert session.has_untracked_receive_activity()

    session.mark_backend_quiesced()
    assert session._aux_drained
    session.close()
    aux_buffer.free_slot.assert_called_once_with(7)


def test_adp_cross_channel_cohort_conflict_reopens_aux_and_retains_kv() -> None:
    session, task = _make_legacy_session(((1, 2), (3, 4)))

    session.process_aux_agent_result(1, AgentResult.SUCCESS)
    session.process_aux_agent_result(2, AgentResult.SUCCESS)
    assert session._selected_writer_cohort == frozenset({1, 2})
    assert session._aux_drained

    session.process_kv_agent_result(3, 0, True, AgentResult.SUCCESS)

    assert task._legacy_result_conflict
    assert not task.legacy_resources_drained
    assert session._aux_result_conflict
    assert not session._aux_drained


def test_adp_selected_cohort_is_consistent_across_slices() -> None:
    session, first = _make_legacy_session(((1, 2), (3, 4)))
    session._need_aux = False
    session._aux_drained = True
    second = KVRecvTask(REQUEST_ID, KVSlice(), 1, first._params, aux_slot=None)
    second.set_valid_writer_cohorts(((1, 2), (3, 4)))
    for rank in (1, 2, 3, 4):
        second.mark_writer_exposed(rank)
    second.close_publication()
    session._kv_tasks.append(second)

    session.process_kv_agent_result(1, 0, True, AgentResult.SUCCESS)
    session.process_kv_agent_result(2, 0, True, AgentResult.SUCCESS)
    assert first.status is TaskStatus.TRANSFERRED

    session.process_kv_agent_result(3, 1, True, AgentResult.SUCCESS)

    assert second._legacy_result_conflict
    assert not second.legacy_resources_drained
    assert second.status is TaskStatus.ERROR


@pytest.mark.parametrize("status_code", [0, 255], ids=["known", "unknown"])
def test_negative_wire_slice_id_does_not_alias_last_task(status_code) -> None:
    session, task = _make_legacy_session(((WRITER_RANK,),))
    receiver = session._receiver
    receiver._sessions[REQUEST_ID] = session

    receiver._process_kv_agent_result(
        b"sender",
        [
            MessageType.KV_AGENT_RESULT,
            _KV_RESULT_PREFIX.pack(WRITER_RANK, REQUEST_ID, -1, True, status_code),
        ],
    )

    assert task._legacy_results == {}
    assert not task._legacy_result_conflict
    assert task.status is TaskStatus.INIT
    assert session._aux_results == {}


def test_dispatch_configures_one_deterministic_writer_cohort() -> None:
    receiver = _make_receiver(writer_ranks=(1, 2))
    session = RxSession(REQUEST_ID, _params(ctx_dp_rank=0), receiver)
    session._closed = True

    session.receive(KVSlice())

    task = session._kv_tasks[0]
    assert task._valid_writer_cohorts == (frozenset({1, 2}),)
    assert task.expected_transfers == 2
    assert task.lifecycle_managed
    assert receiver._request_sender_data.call_count == 2


def test_cancel_during_partial_publication_drains_only_exposed_aux_writer() -> None:
    receiver = _make_receiver(writer_ranks=(1, 2))
    aux_buffer = SimpleNamespace(
        alloc_slot=Mock(return_value=SimpleNamespace(id=7)),
        free_slot=Mock(),
    )
    session = RxSession(
        REQUEST_ID,
        _params(schedule_style=DisaggScheduleStyle.GENERATION_FIRST),
        receiver,
        aux_buffer=aux_buffer,
    )
    session._closed = True

    receiver._request_sender_data.side_effect = lambda *_args: session.cancel()

    session.receive(KVSlice(is_last_slice=True))

    assert receiver._request_sender_data.call_count == 1
    assert session._aux_exposed_writer_ranks == {1}
    assert session._aux_results == {}
    assert session._aux_publication_closed
    assert not session._aux_drained

    session.process_aux_agent_result(1, AgentResult.FAILED)

    assert session._aux_results == {1: AgentResult.FAILED}
    assert session._aux_drained
    assert session._aux_status is TaskStatus.ERROR


def test_later_suppressed_slice_cannot_revoke_earlier_aux_exposure() -> None:
    receiver = _make_receiver(writer_ranks=(1, 2))
    aux_buffer = SimpleNamespace(
        alloc_slot=Mock(return_value=SimpleNamespace(id=7)),
        free_slot=Mock(),
    )
    session = RxSession(
        REQUEST_ID,
        _params(schedule_style=DisaggScheduleStyle.GENERATION_FIRST),
        receiver,
        aux_buffer=aux_buffer,
    )
    session._closed = True

    session.receive(KVSlice(is_last_slice=False))
    assert session._aux_exposed_writer_ranks == {1, 2}
    assert not session._aux_publication_closed

    receiver._request_sender_data.side_effect = lambda *_args: session.cancel()
    session.receive(KVSlice(is_last_slice=True))

    assert receiver._request_sender_data.call_count == 3
    assert session._aux_exposed_writer_ranks == {1, 2}
    assert session._aux_results == {}
    assert session._aux_publication_closed

    session.process_aux_agent_result(1, AgentResult.FAILED)
    assert not session._aux_drained

    session.process_aux_agent_result(2, AgentResult.FAILED)
    assert session._aux_drained
    assert session._aux_status is TaskStatus.ERROR


def test_dispatch_configures_each_adp_group_as_a_candidate_cohort() -> None:
    receiver = _make_receiver(writer_ranks=(1, 2, 3, 4))
    receiver._registrar.get_peer_overlap = lambda _peer_info, dp_rank: PeerOverlap(
        overlap_pp_size=1,
        duplicate_head_factor=1,
        ranks=[1, 2] if dp_rank == 0 else [3, 4],
    )
    peer_info = SimpleNamespace(
        dp_size=2,
        sender_endpoints=[f"tcp://sender-{rank}" for rank in range(5)],
    )
    receiver._get_sender_info = lambda _params: peer_info
    session = RxSession(REQUEST_ID, _params(ctx_dp_rank=None), receiver)
    session._closed = True

    session.receive(KVSlice())

    task = session._kv_tasks[0]
    assert task._valid_writer_cohorts == (
        frozenset({1, 2}),
        frozenset({3, 4}),
    )
    assert task.expected_transfers == 2
    assert not task.lifecycle_managed
    assert receiver._request_sender_data.call_count == 4


def test_pre_cancelled_session_closes_receive_admission() -> None:
    receiver = _make_receiver()
    receiver._pre_cancelled_rids.add(REQUEST_ID)
    session = RxSession(REQUEST_ID, _params(), receiver)
    session._closed = True

    assert session.status is SessionStatus.CANCELLED

    with pytest.raises(RuntimeError, match="new receive slices are not accepted"):
        session.receive(KVSlice())

    assert session._kv_tasks == []
    receiver._request_sender_data.assert_not_called()
    assert REQUEST_ID in receiver._pre_cancelled_rids


def test_receiver_pre_cancel_setup_rolls_back_and_retains_tombstone_on_failure() -> None:
    receiver = _make_receiver()
    receiver._pre_cancelled_rids.add(REQUEST_ID)
    session = SimpleNamespace(
        disagg_request_id=REQUEST_ID,
        cancel=Mock(side_effect=RuntimeError("injected pre-cancel failure")),
    )

    with pytest.raises(RuntimeError, match="pre-cancel failure"):
        receiver.setup_session(session)

    assert REQUEST_ID not in receiver._sessions
    assert REQUEST_ID in receiver._pre_cancelled_rids


def test_receiver_cancel_for_live_session_creates_durable_tombstone() -> None:
    receiver = _make_receiver()
    session = SimpleNamespace(disagg_request_id=REQUEST_ID, cancel=Mock())
    receiver._sessions[REQUEST_ID] = session

    receiver._handle_cancel_session([MessageType.CANCEL_SESSION, str(REQUEST_ID).encode("ascii")])

    session.cancel.assert_called_once_with()
    assert REQUEST_ID in receiver._pre_cancelled_rids


def test_local_rx_session_cancel_creates_durable_tombstone_and_seals_admission() -> None:
    receiver = _make_receiver()
    session = RxSession(REQUEST_ID, _params(), receiver)
    session._closed = True

    session.cancel()

    assert REQUEST_ID in receiver._pre_cancelled_rids
    with pytest.raises(RuntimeError, match="new receive slices are not accepted"):
        session.receive(KVSlice())


def test_local_rx_session_timeout_creates_durable_tombstone_and_seals_admission() -> None:
    receiver = _make_receiver()
    session = RxSession(REQUEST_ID, _params(), receiver, timeout_s=0)
    session._closed = True
    session.resources_drained = Mock(side_effect=(False, False, True))

    session.wait_complete(blocking=True)

    assert REQUEST_ID in receiver._pre_cancelled_rids
    with pytest.raises(RuntimeError, match="new receive slices are not accepted"):
        session.receive(KVSlice())
    receiver.send_cancel_to_senders.assert_called_once_with(REQUEST_ID, set())


def test_receiver_shutdown_closes_admission_and_duplicate_ids_are_rejected() -> None:
    receiver = _make_receiver()
    receiver.begin_shutdown()

    with pytest.raises(RuntimeError, match="shutting down"):
        receiver.setup_session(SimpleNamespace(disagg_request_id=REQUEST_ID))

    receiver = _make_receiver()
    first = SimpleNamespace(disagg_request_id=REQUEST_ID)
    receiver.setup_session(first)
    with pytest.raises(RuntimeError, match="already registered"):
        receiver.setup_session(SimpleNamespace(disagg_request_id=REQUEST_ID))


def test_receiver_shutdown_retries_and_continues_after_session_cancel_failure() -> None:
    receiver = _make_receiver()
    first = SimpleNamespace(
        disagg_request_id=REQUEST_ID,
        cancel=Mock(side_effect=[RuntimeError("injected failure"), None]),
    )
    second = SimpleNamespace(disagg_request_id=REQUEST_ID + 1, cancel=Mock())
    receiver._sessions = {REQUEST_ID: first, REQUEST_ID + 1: second}

    with pytest.raises(RuntimeError, match="encountered 1 error"):
        receiver.begin_shutdown()

    assert receiver._shutdown_started
    first.cancel.assert_called_once_with()
    second.cancel.assert_called_once_with()

    receiver.begin_shutdown()

    assert first.cancel.call_count == 2
    assert second.cancel.call_count == 2


def test_receiver_shutdown_is_serialized_and_closes_dealer_admission() -> None:
    receiver = _make_receiver()
    receiver._shutdown = False
    receiver._listener_stopped = False
    receiver._messenger = Mock()
    dealer = Mock()
    receiver._dealers = {"tcp://sender": dealer}
    listener_stop_started = threading.Event()
    allow_listener_stop = threading.Event()

    def stop_listener() -> None:
        listener_stop_started.set()
        assert allow_listener_stop.wait(timeout=2)

    receiver._messenger.stop.side_effect = stop_listener
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(receiver.shutdown())) for _ in range(2)
    ]
    threads[0].start()
    assert listener_stop_started.wait(timeout=1)
    threads[1].start()

    with pytest.raises(RuntimeError, match="dealer admission is closed"):
        receiver._get_or_connect_dealer("tcp://late-sender")

    allow_listener_stop.set()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert results == [True, True]
    receiver._messenger.stop.assert_called_once_with()
    dealer.stop.assert_called_once_with()


def test_receiver_shutdown_does_not_remove_replacement_dealer() -> None:
    receiver = _make_receiver()
    receiver._shutdown = False
    receiver._listener_stopped = True
    old_dealer = Mock()
    replacement = Mock()
    endpoint = "tcp://sender"
    receiver._dealers = {endpoint: old_dealer}
    old_dealer.stop.side_effect = lambda: receiver._dealers.__setitem__(endpoint, replacement)

    assert receiver.shutdown() is False
    assert receiver._dealers[endpoint] is replacement

    assert receiver.shutdown() is True
    replacement.stop.assert_called_once_with()


def test_rx_session_cancel_retries_failed_contexts_and_continues_siblings() -> None:
    bounce = _FailOnceLogicalFailureBounce()
    receiver = _make_receiver(bounce=bounce)
    session = RxSession(REQUEST_ID, _params(), receiver)
    session._closed = True
    session.receive(KVSlice(is_last_slice=False))
    session.receive(KVSlice(is_last_slice=True))

    with pytest.raises(RuntimeError, match="cancellation encountered 1 error"):
        session.cancel()

    assert bounce.logical_failure_attempts == [(REQUEST_ID, 0), (REQUEST_ID, 1)]
    assert session.status is SessionStatus.CANCELLED
    assert all(task.status is TaskStatus.ERROR for task in session._kv_tasks)

    session.cancel()

    assert bounce.logical_failure_attempts == [
        (REQUEST_ID, 0),
        (REQUEST_ID, 1),
        (REQUEST_ID, 0),
        (REQUEST_ID, 1),
    ]


def test_rx_session_cancel_retries_legacy_idle_release_after_task_failure() -> None:
    bounce = _FakeBounce()
    bounce.release_idle_reservation = Mock(
        side_effect=[RuntimeError("injected idle-release failure"), None]
    )
    receiver = _make_receiver(bounce=bounce)
    session = RxSession(
        REQUEST_ID,
        _params(
            ctx_dp_rank=None,
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        ),
        receiver,
    )
    session._closed = True
    session.receive(KVSlice())

    with pytest.raises(RuntimeError, match="injected idle-release failure"):
        session.cancel()

    assert session._kv_tasks[0].status is TaskStatus.ERROR

    session.cancel()

    assert bounce.release_idle_reservation.call_count == 2


def test_rx_session_serializes_slice_dispatch_before_closing_aux_publication() -> None:
    receiver = _make_receiver()
    aux_buffer = SimpleNamespace(
        alloc_slot=Mock(return_value=SimpleNamespace(id=7)),
        free_slot=Mock(),
    )
    session = RxSession(
        REQUEST_ID,
        _params(schedule_style=DisaggScheduleStyle.GENERATION_FIRST),
        receiver,
        aux_buffer=aux_buffer,
    )
    session._closed = True
    aux_buffer.alloc_slot.assert_called_once_with()
    assert session.aux_slot == 7
    first_send_started = threading.Event()
    allow_first_send = threading.Event()
    second_receive_started = threading.Event()
    second_receive_finished = threading.Event()
    call_lock = threading.Lock()
    send_count = 0
    thread_errors = []

    def request_sender_data(*_args) -> None:
        nonlocal send_count
        with call_lock:
            send_count += 1
            current_send = send_count
        if current_send == 1:
            first_send_started.set()
            assert allow_first_send.wait(timeout=2)

    receiver._request_sender_data.side_effect = request_sender_data

    def receive_first_slice() -> None:
        try:
            session.receive(KVSlice(is_last_slice=False))
        except Exception as e:
            thread_errors.append(e)

    first_thread = threading.Thread(target=receive_first_slice)

    def receive_last_slice() -> None:
        second_receive_started.set()
        try:
            session.receive(KVSlice(is_last_slice=True))
        except Exception as e:
            thread_errors.append(e)
        finally:
            second_receive_finished.set()

    second_thread = threading.Thread(target=receive_last_slice)
    first_thread.start()
    assert first_send_started.wait(timeout=1)
    second_thread.start()
    assert second_receive_started.wait(timeout=1)
    try:
        assert not second_receive_finished.wait(timeout=0.05)
        assert len(session._kv_tasks) == 1
        assert not session._aux_publication_closed
    finally:
        allow_first_send.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not thread_errors
    assert len(session._kv_tasks) == 2
    assert session._aux_publication_closed
    assert session._aux_exposed_writer_ranks == {WRITER_RANK}

    with pytest.raises(RuntimeError, match="already admitted its last slice"):
        session.receive(KVSlice())


def test_rx_session_close_is_exactly_once_and_closes_receive_admission() -> None:
    receiver = _make_receiver()
    aux_buffer = SimpleNamespace(
        alloc_slot=Mock(return_value=SimpleNamespace(id=7)),
        free_slot=Mock(),
    )
    session = RxSession(
        REQUEST_ID,
        _params(schedule_style=DisaggScheduleStyle.GENERATION_FIRST),
        receiver,
        aux_buffer=aux_buffer,
    )
    session.mark_backend_quiesced()
    start = threading.Barrier(3)
    errors = []

    def close() -> None:
        start.wait()
        try:
            session.close()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=close) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    aux_buffer.free_slot.assert_called_once_with(7)
    assert REQUEST_ID not in receiver._sessions

    with pytest.raises(RuntimeError, match="new receive slices are not accepted"):
        session.receive(KVSlice())


def test_bounce_reservation_wait_is_outside_session_cancel_gate() -> None:
    bounce = _BlockingBounce()
    receiver = _make_receiver(bounce=bounce)
    receiver._destination_intervals = Mock(return_value=())
    session = RxSession(REQUEST_ID, _params(), receiver)
    session._closed = True
    receive_finished = threading.Event()
    cancel_finished = threading.Event()

    def receive() -> None:
        session.receive(KVSlice())
        receive_finished.set()

    def cancel() -> None:
        session.cancel()
        cancel_finished.set()

    receive_thread = threading.Thread(target=receive)
    receive_thread.start()
    assert bounce.reserve_started.wait(timeout=1)

    cancel_thread = threading.Thread(target=cancel)
    cancel_thread.start()
    try:
        assert cancel_finished.wait(timeout=1), "cancel blocked behind bounce.reserve()"
    finally:
        bounce.allow_reserve.set()
        receive_thread.join(timeout=2)
        cancel_thread.join(timeout=2)

    assert receive_finished.is_set()
    assert bounce.released_active_reservations == [(REQUEST_ID, 0)]
    assert session._kv_tasks[0].status is TaskStatus.ERROR
    receiver._request_sender_data.assert_not_called()


def test_receiver_shutdown_waits_for_admitted_receive_dispatch() -> None:
    receiver = _make_receiver()
    receiver._shutdown = False
    receiver._listener_stopped = False
    receiver._messenger = Mock()
    discovery_started = threading.Event()
    allow_discovery = threading.Event()
    peer_info = SimpleNamespace(
        dp_size=1,
        sender_endpoints=[f"tcp://sender-{rank}" for rank in range(WRITER_RANK + 1)],
    )

    def get_sender_info(_params):
        discovery_started.set()
        allow_discovery.wait()
        return peer_info

    receiver._get_sender_info = get_sender_info
    session = RxSession(REQUEST_ID, _params(), receiver)
    receive_errors = []

    def receive() -> None:
        try:
            session.receive(KVSlice())
        except Exception as e:
            receive_errors.append(e)

    receive_thread = threading.Thread(
        target=receive,
        name="blocked-receive",
        daemon=True,
    )
    shutdown_thread = None
    receive_thread.start()
    try:
        assert discovery_started.wait(timeout=1)

        shutdown_results: list[bool] = []
        shutdown_errors: list[Exception] = []
        shutdown_finished = threading.Event()

        def shutdown() -> None:
            try:
                shutdown_results.append(receiver.shutdown())
            except Exception as e:
                shutdown_errors.append(e)
            finally:
                shutdown_finished.set()

        shutdown_thread = threading.Thread(
            target=shutdown,
            name="receiver-shutdown",
            daemon=True,
        )
        shutdown_thread.start()
        assert shutdown_finished.wait(timeout=5), (
            "receiver.shutdown() blocked behind admitted sender discovery"
        )
        assert not shutdown_errors
        assert shutdown_results == [False]
        receiver._messenger.stop.assert_not_called()
        assert not receiver.transfers_drained
        with pytest.raises(RuntimeError, match="new receive slices are not accepted"):
            session.receive(KVSlice())
    finally:
        # Always unblock discovery so a failed deadlock assertion cannot leak
        # either worker into the rest of the test process.
        allow_discovery.set()
        if shutdown_thread is not None:
            shutdown_thread.join(timeout=2)
        receive_thread.join(timeout=2)

    assert shutdown_thread is not None
    assert not shutdown_thread.is_alive()
    assert not receive_thread.is_alive()
    assert not receive_errors
    assert receiver.transfers_drained
    assert receiver.shutdown() is True
    receiver._messenger.stop.assert_called_once_with()
    session.close()


def test_rx_session_close_waits_for_dispatch_and_prevents_later_receive() -> None:
    receiver = _make_receiver()
    send_started = threading.Event()
    allow_send = threading.Event()

    def request_sender_data(*_args) -> None:
        send_started.set()
        assert allow_send.wait(timeout=2)

    receiver._request_sender_data.side_effect = request_sender_data
    session = RxSession(REQUEST_ID, _params(), receiver)
    receive_finished = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    close_errors = []

    def receive() -> None:
        session.receive(KVSlice())
        receive_finished.set()

    def close() -> None:
        close_started.set()
        try:
            session.close()
        except Exception as e:
            close_errors.append(e)
        finally:
            close_finished.set()

    receive_thread = threading.Thread(target=receive)
    receive_thread.start()
    assert send_started.wait(timeout=1)

    close_thread = threading.Thread(target=close)
    close_thread.start()
    assert close_started.wait(timeout=1)
    try:
        assert not close_finished.wait(timeout=0.05)
        assert REQUEST_ID in receiver._sessions
    finally:
        allow_send.set()
        receive_thread.join(timeout=2)
        close_thread.join(timeout=2)

    assert receive_finished.is_set()
    assert close_finished.is_set()
    assert len(close_errors) == 1
    assert "receive resources are not drained" in str(close_errors[0])
    assert REQUEST_ID in receiver._sessions
    with pytest.raises(RuntimeError, match="new receive slices are not accepted"):
        session.receive(KVSlice())
    session._closed = True


def test_mamba_receive_uses_direct_path_until_bound_layout_includes_state_bytes() -> None:
    bounce = _FakeBounce()
    bounce.enabled = True
    bounce.reserve = Mock(return_value=True)
    receiver = _make_receiver(bounce=bounce)
    receiver._destination_intervals = Mock(return_value=())
    session = RxSession(REQUEST_ID, _params(), receiver)
    session._closed = True

    session.receive(KVSlice(mamba_state_index=0))

    bounce.reserve.assert_not_called()
    receiver._destination_intervals.assert_not_called()
    assert receiver._recv_registry.target_mode((REQUEST_ID, 0)) is not None


def test_standalone_receive_dispatch_error_closes_publication_transactionally() -> None:
    receiver = _make_receiver()
    receiver.dispatch_task = Mock(side_effect=RuntimeError("prepare failed"))
    destination_owner = object()
    session = RxSession(
        REQUEST_ID,
        _params(),
        receiver,
        destination_owner=destination_owner,
    )

    with pytest.raises(RuntimeError, match="prepare failed"):
        session.receive(KVSlice())

    task = session._kv_tasks[0]
    assert task.status is TaskStatus.ERROR
    assert task._publication_closed
    assert session._active_receive_dispatches == 0
    assert session.status is SessionStatus.ERROR
    assert session.resources_drained()
    session.close()
    assert REQUEST_ID not in receiver._sessions
    assert session._destination_owner is None


def test_standalone_receive_dispatch_error_retains_ambiguous_target_owner() -> None:
    receiver = _make_receiver()
    destination_owner = object()
    session = RxSession(
        REQUEST_ID,
        _params(),
        receiver,
        destination_owner=destination_owner,
    )
    key = (REQUEST_ID, 0)

    def fail_after_publication_started(task) -> None:
        task.set_valid_writer_cohorts(((WRITER_RANK,),))
        task.lifecycle_managed = True
        task.status = TaskStatus.TRANSFERRING
        assert receiver._recv_registry.prepare(key, (WRITER_RANK,), has_bounce_slot=False).accepted
        assert receiver._recv_registry.begin_publication(key, WRITER_RANK).publication_allowed
        task.mark_writer_exposed(WRITER_RANK)
        raise RuntimeError("publication outcome is ambiguous")

    receiver.dispatch_task = fail_after_publication_started

    with pytest.raises(RuntimeError, match="publication outcome is ambiguous"):
        session.receive(KVSlice())

    snapshot = receiver._recv_registry.context_snapshot(key)
    assert snapshot.logical_state is LogicalState.CANCELLED
    assert snapshot.physical_state is PhysicalState.IN_DOUBT
    assert session._kv_tasks[0]._publication_closed
    assert not session.resources_drained()
    with pytest.raises(RuntimeError, match="not drained"):
        session.close()
    assert receiver._sessions[REQUEST_ID] is session
    assert session._destination_owner is destination_owner
    session._closed = True


def test_pp_fanin_with_uniform_peer_stages_uses_direct_path() -> None:
    bounce = _FakeBounce()
    bounce.enabled = True
    bounce.reserve = Mock(return_value=True)
    receiver = _make_receiver(bounce=bounce, writer_ranks=(0, 1))
    receiver._registrar.get_peer_overlap = lambda _peer_info, _dp_rank: PeerOverlap(
        overlap_pp_size=2,
        duplicate_head_factor=1,
        peer_duplicate_head_factor=1,
        ranks=[0, 1],
    )
    receiver._get_sender_info = lambda _params: SimpleNamespace(
        dp_size=1,
        layer_num_per_pp=[20, 20],
        sender_endpoints=["tcp://sender-0", "tcp://sender-1"],
    )
    receiver._destination_intervals = Mock(return_value=())
    session = RxSession(REQUEST_ID, _params(), receiver)
    session._closed = True

    # A local [0, 30) stage intersects the uniform peer stages [0, 20) and
    # [20, 40) by 20 and 10 layers. Peer uniformity therefore cannot authorize
    # an equal two-writer bounce subdivision.
    session.receive(KVSlice())

    bounce.reserve.assert_not_called()
    receiver._destination_intervals.assert_not_called()
    assert receiver._request_sender_data.call_count == 2


def test_tp_fanin_uses_direct_path_without_byte_accurate_writer_extents() -> None:
    bounce = _FakeBounce()
    bounce.enabled = True
    bounce.reserve = Mock(return_value=True)
    receiver = _make_receiver(bounce=bounce, writer_ranks=(0, 1))
    receiver._destination_intervals = Mock(return_value=())
    session = RxSession(REQUEST_ID, _params(), receiver)
    session._closed = True

    session.receive(KVSlice())

    bounce.reserve.assert_not_called()
    receiver._destination_intervals.assert_not_called()
    assert receiver._request_sender_data.call_count == 2


def test_partial_layer_mapping_uses_direct_path() -> None:
    bounce = _FakeBounce()
    bounce.enabled = True
    bounce.reserve = Mock(return_value=True)
    receiver = _make_receiver(bounce=bounce)
    receiver._single_writer_bounce_exact = Receiver._single_writer_bounce_exact.__get__(
        receiver, Receiver
    )
    receiver._registrar.self_rank_info.layer_num_per_pp = [12]
    receiver._get_sender_info = lambda _params: SimpleNamespace(
        dp_size=1,
        layer_num_per_pp=[10],
        sender_endpoints=["tcp://sender-0"] * (WRITER_RANK + 1),
    )
    receiver._destination_intervals = Mock(return_value=())
    session = RxSession(REQUEST_ID, _params(), receiver)
    session._closed = True

    session.receive(KVSlice())

    bounce.reserve.assert_not_called()
    receiver._destination_intervals.assert_not_called()
    receiver._request_sender_data.assert_called_once()


def _bounce_proof_page_table(*slot_bytes: int, tokens_per_block: int = 32):
    pool_views = [
        PoolView(
            pool_idx=pool_idx,
            buffer_entries=np.empty(0, dtype=BUFFER_ENTRY_DTYPE),
        )
        for pool_idx in range(len(slot_bytes))
    ]
    return KVCachePageTable(
        tokens_per_block=tokens_per_block,
        layer_groups=[AttentionLayerGroup(pool_group_idx=0, pool_views=pool_views)],
        pool_groups=[
            PhysicalPoolGroup(
                pools=[
                    PhysicalPool(
                        base_address=0x1000 + pool_idx * 0x1000,
                        slot_bytes=size,
                        num_slots=16,
                    )
                    for pool_idx, size in enumerate(slot_bytes)
                ]
            )
        ],
    )


def _single_writer_bounce_proof(local_page_table, peer_page_table, mapping) -> bool:
    receiver = object.__new__(Receiver)
    receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(layer_num_per_pp=[1], page_table=local_page_table),
        get_pool_mapping=Mock(return_value=mapping),
        get_kv_map=Mock(return_value=IdentityMapper()),
    )
    peer_info = SimpleNamespace(layer_num_per_pp=[1], page_table=peer_page_table)
    return receiver._single_writer_bounce_exact(peer_info)


def test_single_writer_bounce_requires_byte_exact_pool_bijection() -> None:
    local_page_table = _bounce_proof_page_table(64, 128)
    peer_page_table = _bounce_proof_page_table(64, 128)

    assert _single_writer_bounce_proof(
        local_page_table,
        peer_page_table,
        {(0, 0): (0, 0), (0, 1): (0, 1)},
    )
    assert not _single_writer_bounce_proof(
        local_page_table,
        peer_page_table,
        {(0, 0): (0, 0), (0, 1): (0, 0)},
    )


def test_single_writer_bounce_rejects_asymmetric_slot_bytes() -> None:
    assert not _single_writer_bounce_proof(
        _bounce_proof_page_table(64),
        _bounce_proof_page_table(128),
        {(0, 0): (0, 0)},
    )


def test_single_writer_bounce_rejects_mismatched_block_geometry() -> None:
    assert not _single_writer_bounce_proof(
        _bounce_proof_page_table(64, tokens_per_block=32),
        _bounce_proof_page_table(64, tokens_per_block=64),
        {(0, 0): (0, 0)},
    )


def test_receiver_clear_session_does_not_hold_map_lock_during_bounce_retry() -> None:
    receiver = _make_receiver()

    class _RetryingSession:
        def resources_drained(self) -> bool:
            assert receiver._get_session(REQUEST_ID) is self
            return True

    session = _RetryingSession()
    receiver._sessions[REQUEST_ID] = session
    errors = []

    def clear() -> None:
        try:
            receiver.clear_session(REQUEST_ID)
        except Exception as e:
            errors.append(e)

    thread = threading.Thread(target=clear)
    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive(), "clear_session deadlocked through resources_drained()"
    assert not errors
    assert REQUEST_ID not in receiver._sessions


@pytest.mark.parametrize("has_bounce", [False, True], ids=["direct", "bounce"])
def test_unknown_managed_wire_status_remains_undrained(has_bounce) -> None:
    bounce = _FakeBounce()
    receiver = _make_receiver(bounce=bounce)
    key = (REQUEST_ID, 0)
    assert receiver._recv_registry.prepare(
        key,
        {WRITER_RANK},
        has_bounce_slot=has_bounce,
    ).accepted
    assert receiver._recv_registry.begin_publication(key, WRITER_RANK).publication_allowed
    assert receiver._recv_registry.mark_published(key, WRITER_RANK).accepted
    message = [
        MessageType.KV_AGENT_RESULT,
        _KV_RESULT_PREFIX.pack(WRITER_RANK, REQUEST_ID, 0, True, 255),
    ]

    receiver._process_kv_agent_result(b"sender", message)

    snapshot = receiver._recv_registry.context_snapshot(key)
    assert snapshot is not None
    assert snapshot.logical_state is LogicalState.FAILED
    assert snapshot.physical_state is PhysicalState.IN_DOUBT
    assert snapshot.writers[0].result is None
    assert not receiver._recv_registry.is_request_drained(REQUEST_ID)
    assert len(bounce.protocol_conflict_callbacks) == int(has_bounce)


def test_post_success_contradiction_corrects_receiver_session_outcome() -> None:
    receiver = _make_receiver()
    session = RxSession(REQUEST_ID, _params(), receiver)
    session._closed = True
    session.receive(KVSlice())
    message_prefix = (WRITER_RANK, REQUEST_ID, 0, True)

    receiver._process_kv_agent_result(
        b"sender",
        [MessageType.KV_AGENT_RESULT, _KV_RESULT_PREFIX.pack(*message_prefix, 0)],
    )
    assert session._kv_tasks[0].status is TaskStatus.TRANSFERRED

    receiver._process_kv_agent_result(
        b"sender",
        [MessageType.KV_AGENT_RESULT, _KV_RESULT_PREFIX.pack(*message_prefix, 1)],
    )

    assert session._kv_tasks[0].status is TaskStatus.ERROR
    assert session.status is SessionStatus.ERROR
    assert not receiver._recv_registry.is_request_drained(REQUEST_ID)


@pytest.mark.parametrize("peer_rank", [WRITER_RANK, WRITER_RANK + 1])
@pytest.mark.parametrize("has_bounce", [False, True], ids=["direct", "bounce"])
def test_unknown_managed_wire_status_before_publication_requires_backend_fence(
    has_bounce, peer_rank
) -> None:
    bounce = _FakeBounce()
    receiver = _make_receiver(bounce=bounce)
    key = (REQUEST_ID, 0)
    assert receiver._recv_registry.prepare(
        key,
        {WRITER_RANK},
        has_bounce_slot=has_bounce,
    ).accepted

    receiver._process_kv_agent_result(
        b"sender",
        [
            MessageType.KV_AGENT_RESULT,
            _KV_RESULT_PREFIX.pack(peer_rank, REQUEST_ID, 0, True, 255),
        ],
    )

    snapshot = receiver._recv_registry.context_snapshot(key)
    assert snapshot.logical_state is LogicalState.FAILED
    assert snapshot.physical_state is PhysicalState.IN_DOUBT
    assert not receiver._recv_registry.is_request_drained(REQUEST_ID)
    assert len(bounce.protocol_conflict_callbacks) == int(has_bounce)


def test_unknown_aux_status_fails_logically_without_retiring_writer() -> None:
    session, task = _make_legacy_session(((WRITER_RANK,),))
    receiver = session._receiver
    receiver._sessions[REQUEST_ID] = session

    receiver._process_aux_agent_result(
        b"sender",
        [
            MessageType.AUX_AGENT_RESULT,
            str(WRITER_RANK).encode("ascii"),
            str(REQUEST_ID).encode("ascii"),
            b"NEWER_STATUS",
        ],
    )

    assert session._aux_status is TaskStatus.ERROR
    assert session._aux_result_conflict
    assert not session._aux_drained
    assert session._aux_results == {}
    assert not task.legacy_resources_drained

    session.process_aux_agent_result(WRITER_RANK, AgentResult.FAILED)

    assert session._aux_drained
    assert session._aux_status is TaskStatus.ERROR


def test_sender_cancel_before_session_replays_failure_for_stored_and_late_requests() -> None:
    info = _recv_info(writer_rank=0)
    sender = object.__new__(Sender)
    sender._instance_rank = WRITER_RANK
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._peer_requests = {REQUEST_ID: {(0, 0): info}}
    sender._peer_requests_lock = threading.Lock()
    sender._pre_cancelled_rids = set()
    sender._send_failed_result_to_receiver = Mock()
    sender._send_aux_failed_result_to_receiver = Mock()

    sender._handle_cancel_session([MessageType.CANCEL_SESSION, str(REQUEST_ID).encode("ascii")])

    assert REQUEST_ID in sender._pre_cancelled_rids
    sender._send_failed_result_to_receiver.assert_called_once_with(info)
    sender._send_aux_failed_result_to_receiver.assert_called_once_with(info)

    sender._send_failed_result_to_receiver.reset_mock()
    sender._send_aux_failed_result_to_receiver.reset_mock()
    sender._save_peer_req_info = Mock()
    sender._respond_with_kv(b"receiver", [MessageType.REQUEST_DATA, info.to_bytes()])
    sender._save_peer_req_info.assert_called_once()
    sender._send_failed_result_to_receiver.assert_called_once()
    sender._send_aux_failed_result_to_receiver.assert_called_once()

    sender._shutdown = False
    late_session = Mock(disagg_request_id=REQUEST_ID)
    sender.setup_session(late_session)
    late_session.cancel.assert_called_once_with()
    assert REQUEST_ID in sender._pre_cancelled_rids


def test_sender_shutdown_late_request_retains_kv_and_aux_no_access_results() -> None:
    sender = _make_sender_for_shutdown()
    sender._shutdown = True
    sender._instance_rank = WRITER_RANK
    sender._save_peer_req_info = Mock()
    sender._send_operation_message = Mock(return_value=False)
    info = _recv_info(writer_rank=0)

    sender._respond_with_kv(b"receiver", [MessageType.REQUEST_DATA, info.to_bytes()])

    assert REQUEST_ID in sender._pre_cancelled_rids
    assert len(sender._pre_session_terminal_results) == 2
    assert {key[1] for key in sender._pre_session_terminal_results} == {
        WriteMetaType.KV,
        WriteMetaType.AUX,
    }
    assert sender._has_pending_pre_session_terminal_results()
    sender._shutdown_complete = True


def test_sender_shutdown_late_request_cannot_admit_session_source_access() -> None:
    sender = _make_sender_for_shutdown()
    sender._save_peer_req_info = Mock()
    sender._send_operation_message = Mock(return_value=False)
    session = TxSession(
        request_id=REQUEST_ID,
        params=_params(),
        sender=sender,
        source_owner=object(),
    )
    kv_task = KVSendTask(KVSlice(), _params(), 0, session=session)
    aux_task = AuxSendTask(_params(), 7, session=session)
    session.kv_tasks.append(kv_task)
    session.aux_task = aux_task
    sender._shutdown = True
    info = _recv_info()
    key = (info.instance_name, info.instance_rank)

    sender._respond_with_kv(b"receiver", [MessageType.REQUEST_DATA, info.to_bytes()])

    for task in (kv_task, aux_task):
        snapshot = task.operation_snapshot(key)
        assert snapshot is not None
        assert snapshot[0] is SendOperationState.TERMINAL
        assert not snapshot[2]
        assert not task.source_access_active

    assert REQUEST_ID in sender._pre_cancelled_rids
    kv_task.mark_operation_result_delivered(key)
    aux_task.mark_operation_result_delivered(key)
    kv_task.fail(RuntimeError("shutdown"))
    aux_task.fail(RuntimeError("shutdown"))
    session.close()
    sender._shutdown_complete = True


def test_sender_strongly_retains_registered_session_until_explicit_clear() -> None:
    class Session:
        disagg_request_id = REQUEST_ID

    sender = _make_sender_for_shutdown()
    session = Session()
    session_ref = weakref.ref(session)

    sender.setup_session(session)
    del session
    gc.collect()

    assert session_ref() is sender._sessions[REQUEST_ID]

    sender.clear_session(REQUEST_ID)
    gc.collect()
    assert session_ref() is None
    sender._shutdown_complete = True


def test_tx_session_setup_failure_rolls_back_sender_and_source_owner() -> None:
    class SourceOwner:
        pass

    sender = _make_sender_for_shutdown()
    info = _recv_info()
    sender._peer_requests = {REQUEST_ID: {(WRITER_RANK, 0): info}}
    sender._registrar = Mock()
    sender._registrar.get_peer_rank_info.side_effect = RuntimeError("peer lookup failed")
    source_owner = SourceOwner()
    source_owner_ref = weakref.ref(source_owner)

    with pytest.raises(RuntimeError, match="peer lookup failed") as exc_info:
        TxSession(
            request_id=REQUEST_ID,
            params=_params(),
            sender=sender,
            source_owner=source_owner,
        )

    del source_owner
    gc.collect()
    assert sender._sessions == {}
    assert source_owner_ref() is None
    assert str(exc_info.value) == "peer lookup failed"
    sender._shutdown_complete = True


def test_rx_session_setup_failure_rolls_back_destination_owner() -> None:
    class DestinationOwner:
        pass

    receiver = _make_receiver()
    receiver.begin_shutdown()
    destination_owner = DestinationOwner()
    destination_owner_ref = weakref.ref(destination_owner)

    with pytest.raises(RuntimeError, match="shutting down") as exc_info:
        RxSession(
            request_id=REQUEST_ID,
            params=_params(),
            receiver=receiver,
            destination_owner=destination_owner,
        )

    del destination_owner
    gc.collect()
    assert destination_owner_ref() is None
    assert "shutting down" in str(exc_info.value)


@pytest.mark.parametrize(
    "session_type",
    [
        pytest.param(TxSession, id="tx"),
        pytest.param(RxSession, id="rx"),
    ],
)
@pytest.mark.parametrize("failure_stage", ["base-init", "aux-allocation"])
def test_early_session_constructor_failure_releases_allocator_owner(
    session_type, failure_stage, monkeypatch
) -> None:
    class AllocatorOwner:
        pass

    if session_type is TxSession:
        endpoint_name = "sender"
        endpoint = _make_sender_for_shutdown()
        owner_name = "source_owner"
    else:
        endpoint_name = "receiver"
        endpoint = _make_receiver()
        owner_name = "destination_owner"

    aux_buffer = SimpleNamespace(
        alloc_slot=Mock(),
        free_slot=Mock(),
    )
    error_message = f"{failure_stage} failed"
    base_type = session_type.__mro__[1]
    original_base_init = base_type.__init__
    failed_instances = []
    if failure_stage == "base-init":

        def capture_and_fail_base_init(session, *_args, **_kwargs) -> None:
            failed_instances.append(session)
            raise RuntimeError(error_message)

        monkeypatch.setattr(base_type, "__init__", capture_and_fail_base_init)
    else:

        def capture_base_init(session, *args, **kwargs) -> None:
            failed_instances.append(session)
            original_base_init(session, *args, **kwargs)

        monkeypatch.setattr(base_type, "__init__", capture_base_init)
        aux_buffer.alloc_slot.side_effect = RuntimeError(error_message)

    owner = AllocatorOwner()
    owner_ref = weakref.ref(owner)
    with pytest.raises(RuntimeError, match=error_message) as exc_info:
        session_type(
            request_id=REQUEST_ID,
            params=_params(schedule_style=DisaggScheduleStyle.GENERATION_FIRST),
            aux_buffer=aux_buffer,
            **{endpoint_name: endpoint, owner_name: owner},
        )

    assert len(failed_instances) == 1
    failed_session = failed_instances.pop()
    # Constructor rollback already retired every resource it could have
    # acquired. Explicit close, and therefore the later __del__, must stop at
    # that marker without touching terminal fields that may not exist yet.
    failed_session.close()
    failed_session_ref = weakref.ref(failed_session)
    observed_error = str(exc_info.value)
    del exc_info
    del failed_session
    del owner
    gc.collect()
    assert owner_ref() is None
    assert failed_session_ref() is None
    assert observed_error == error_message
    if failure_stage == "base-init":
        aux_buffer.alloc_slot.assert_not_called()
    else:
        aux_buffer.alloc_slot.assert_called_once_with()
    aux_buffer.free_slot.assert_not_called()
    if session_type is TxSession:
        endpoint._shutdown_complete = True


def test_pre_session_terminal_result_is_retried_by_sender_shutdown() -> None:
    sender = _make_sender_for_shutdown()
    info = _recv_info()
    info.aux_slot = None
    sender._peer_requests = {REQUEST_ID: {(WRITER_RANK, 0): info}}
    sender._send_operation_message = Mock(side_effect=[False, True])

    sender._handle_cancel_session([MessageType.CANCEL_SESSION, str(REQUEST_ID).encode("ascii")])

    assert sender._has_pending_pre_session_terminal_results()
    assert sender.shutdown()
    assert not sender._has_pending_pre_session_terminal_results()
    assert sender._send_operation_message.call_count == 2


def test_pre_session_terminal_result_is_retried_by_runtime_progress() -> None:
    sender = _make_sender_for_shutdown()
    info = _recv_info()
    info.aux_slot = None
    sender._peer_requests = {REQUEST_ID: {(WRITER_RANK, 0): info}}
    sender._send_operation_message = Mock(side_effect=[False, True])

    sender._handle_cancel_session([MessageType.CANCEL_SESSION, str(REQUEST_ID).encode("ascii")])

    assert sender._has_pending_pre_session_terminal_results()
    sender.sweep_stale_req_infos()
    assert not sender._has_pending_pre_session_terminal_results()
    assert sender._send_operation_message.call_count == 2
    sender._shutdown_complete = True


def test_sender_claims_one_exact_operation_during_request_send_race() -> None:
    task, session = _make_send_task_and_session()
    info = _recv_info()
    sender = object.__new__(Sender)
    sender._shutdown_complete = True
    sender._instance_rank = 0
    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="receiver3",
        peer_rank=WRITER_RANK,
        peer_endpoint="tcp://receiver",
        unique_rid=REQUEST_ID,
        src_ptrs=np.array([], dtype=np.int64),
        dst_ptrs=np.array([], dtype=np.int64),
        sizes=np.array([], dtype=np.int64),
        dst_device_id=0,
        slice_id=0,
        is_last_slice=True,
        session=session,
    )
    sender._build_kv_write_meta = Mock(return_value=write_meta)
    sender._enqueue_owned = Mock()
    sender._send_operation_message = Mock(return_value=True)

    start = threading.Barrier(3)
    results: list = []

    def dispatch() -> None:
        start.wait()
        results.append(sender._dispatch_operation(task, info))

    threads = [threading.Thread(target=dispatch) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert results == [None, None]
    sender._build_kv_write_meta.assert_called_once_with(task, info)
    sender._enqueue_owned.assert_called_once_with(write_meta)
    assert task.source_access_active

    key = (info.instance_name, info.instance_rank)
    terminal_message = (b"terminal",)
    task.cache_terminal_message(key, terminal_message)
    task.mark_operation_result_delivered(key)
    task.finish_source_access()

    sender._dispatch_operation(task, info)

    sender._send_operation_message.assert_called_once_with(info, terminal_message)
    sender._enqueue_owned.assert_called_once_with(write_meta)


@pytest.mark.parametrize("channel", ["kv", "aux"])
def test_sender_settles_every_writer_after_descriptor_build_failures(channel) -> None:
    task, _session = _make_send_task_and_session(channel=channel)
    infos = {rank: _recv_info(writer_rank=rank) for rank in (1, 2)}
    sender = object.__new__(Sender)
    sender._shutdown_complete = True
    sender._instance_rank = 0
    sender._send_operation_message = Mock(return_value=True)
    sender._enqueue_owned = Mock()
    build = Mock(
        side_effect=[RuntimeError("first descriptor failure"), RuntimeError("second failure")]
    )
    if channel == "kv":
        sender._build_kv_write_meta = build
    else:
        sender._build_aux_write_meta = build

    with pytest.raises(RuntimeError, match="first descriptor failure"):
        sender.dispatch_task(task, infos)

    assert build.call_count == 2
    assert sender._send_operation_message.call_count == 2
    assert not task.source_access_active
    for info in infos.values():
        snapshot = task.operation_snapshot((info.instance_name, info.instance_rank))
        assert snapshot is not None
        assert snapshot[0] is SendOperationState.TERMINAL
        assert snapshot[2]


@pytest.mark.parametrize("channel", ["kv", "aux"])
def test_sender_failure_is_monotonic_while_sibling_source_access_drains(channel) -> None:
    task, session = _make_send_task_and_session(channel=channel)
    first_info = _recv_info(writer_rank=1)
    second_info = _recv_info(writer_rank=2)
    first_meta = WriteMeta(
        task=task,
        expected_transfers=2,
        peer_name="receiver1",
        peer_rank=1,
        peer_endpoint="tcp://receiver",
        unique_rid=REQUEST_ID,
        src_ptrs=np.array([], dtype=np.int64),
        dst_ptrs=np.array([], dtype=np.int64),
        sizes=np.array([], dtype=np.int64),
        slice_id=0 if channel == "kv" else None,
        session=session,
    )
    sender = object.__new__(Sender)
    sender._shutdown_complete = True
    sender._instance_rank = 0
    sender._send_operation_message = Mock(return_value=True)
    sender._enqueue_owned = Mock()
    build = Mock(side_effect=[first_meta, RuntimeError("second descriptor failed")])
    if channel == "kv":
        sender._build_kv_write_meta = build
    else:
        sender._build_aux_write_meta = build

    assert sender._dispatch_operation(task, first_info) is None
    assert task.mark_transferring()
    assert isinstance(sender._dispatch_operation(task, second_info), RuntimeError)

    assert task.status is TaskStatus.ERROR
    assert task.source_access_active
    assert not task.complete()
    assert task.status is TaskStatus.ERROR

    task.finish_source_access()
    first_meta.source_access_enrolled = False
    assert not task.source_access_active


@pytest.mark.parametrize("channel", ["kv", "aux"])
def test_cancelled_partial_fanout_drains_and_late_writer_gets_no_access(channel) -> None:
    sender, session, task, first_meta = _make_owned_send_task_and_session(
        channel=channel, expected_transfers=2
    )
    first_info = _recv_info(writer_rank=1)
    late_info = _recv_info(writer_rank=2)

    assert sender._dispatch_operation(task, first_info) is None
    assert task.mark_transferring()
    session.cancel()

    assert session.status is SessionStatus.CANCELLED
    assert task.status is TaskStatus.ERROR
    assert task.source_access_active
    assert session.has_transferring_tasks()

    first_key = (first_info.instance_name, first_info.instance_rank)
    task.cache_terminal_message(first_key, (b"first-writer-terminal",))
    task.mark_operation_result_delivered(first_key)
    task.finish_source_access()
    first_meta.source_access_enrolled = False
    if channel == "kv":
        task.transferred_count = 1
    else:
        task._transfer_count = 1

    assert not session.has_transferring_tasks()
    assert sender._dispatch_operation(task, late_info) is None
    late_key = (late_info.instance_name, late_info.instance_rank)
    late_snapshot = task.operation_snapshot(late_key)
    assert late_snapshot is not None
    assert late_snapshot[0] is SendOperationState.TERMINAL
    assert late_snapshot[2]
    assert not task.source_access_active
    assert task.status is TaskStatus.ERROR
    assert session.status is SessionStatus.CANCELLED
    assert session.wait_complete(blocking=False) is WaitResult.FAILED
    assert not session.has_transferring_tasks()


@pytest.mark.parametrize("channel", ["kv", "aux"])
@pytest.mark.parametrize("transfer_succeeded", [True, False])
def test_active_writer_completion_after_cancel_preserves_cancelled(
    channel, transfer_succeeded, monkeypatch
) -> None:
    sender, session, task, write_meta = _make_owned_send_task_and_session(
        channel=channel, expected_transfers=1
    )
    info = _recv_info(writer_rank=1)
    write_meta.src_ptrs = np.array([1], dtype=np.int64)
    write_meta.dst_ptrs = np.array([2], dtype=np.int64)
    write_meta.sizes = np.array([4], dtype=np.int64)
    assert sender._dispatch_operation(task, info) is None

    transfer_started = threading.Event()
    release_transfer = threading.Event()
    transfer_status = Mock()

    def wait_for_transfer() -> bool:
        transfer_started.set()
        assert release_transfer.wait(timeout=1)
        return transfer_succeeded

    transfer_status.wait.side_effect = wait_for_transfer
    sender._agent.submit_transfer_requests.return_value = transfer_status
    sender._device_id = 0
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="sender", instance_rank=0)
    )
    dealer = Mock()
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    if channel == "kv":
        monkeypatch.setattr(
            "tensorrt_llm._torch.disaggregation.native.bounce.build_send_request",
            lambda _bounce, _write_meta, _factory: (object(), None),
        )
    else:
        monkeypatch.setattr(
            Sender, "_make_agent_request", staticmethod(lambda *_args, **_kwargs: object())
        )

    errors = []

    def deliver() -> None:
        try:
            if channel == "kv":
                sender._deliver_kv_to_agent(write_meta)
            else:
                sender._deliver_aux_to_agent(write_meta)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=deliver)
    thread.start()
    assert transfer_started.wait(timeout=1)
    assert task.status is TaskStatus.TRANSFERRING

    session.cancel()
    assert session.status is SessionStatus.CANCELLED
    assert task.status is TaskStatus.ERROR
    assert session.has_transferring_tasks()

    release_transfer.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert errors == []
    assert session.status is SessionStatus.CANCELLED
    assert task.status is TaskStatus.ERROR
    assert not task.source_access_active
    assert not task.has_pending_result_delivery
    assert not session.has_transferring_tasks()


@pytest.mark.parametrize("first_terminal", ["cancel", "error"])
def test_tx_session_first_terminal_failure_wins_and_settlement_retries(
    first_terminal,
) -> None:
    sender, session, task, _write_meta = _make_owned_send_task_and_session(
        channel="kv", expected_transfers=1
    )

    if first_terminal == "cancel":
        session.cancel()
        first_exception = session.exception
        session.set_exception("late worker failure")
        session.cancel()

        assert session.status is SessionStatus.CANCELLED
        assert session.exception is first_exception is None
    else:
        session.set_exception("first worker failure")
        first_exception = session.exception
        session.cancel()
        session.set_exception("late worker failure")

        assert session.status is SessionStatus.ERROR
        assert session.exception is first_exception
        assert str(session.exception).endswith(": first worker failure")

    assert task.status is TaskStatus.ERROR
    assert sender.cancel_session.call_count == 3


def test_local_cancel_installs_tombstone_and_retries_settlement() -> None:
    info = _recv_info()
    session = object.__new__(TxSession)
    session._base_args = SessionArgsBase(_params())
    session.request_id = REQUEST_ID
    session.lock = threading.Lock()
    session.kv_tasks = []
    session.aux_task = None
    session._unbound_terminal_results = {}
    session._terminal_retry_lock = threading.Lock()
    session._next_terminal_retry_at = 0.0
    session._terminal_status = SessionStatus.CANCELLED
    session._closed = True
    session._sender = None
    sender = object.__new__(Sender)
    sender._shutdown_complete = False
    sender._instance_rank = 0
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._peer_requests = {REQUEST_ID: {(WRITER_RANK, 0): info}}
    sender._peer_requests_lock = threading.Lock()
    sender._pre_cancelled_rids = set()
    sender._send_operation_message = Mock(return_value=False)
    sender.send_cancel_to_receivers = Mock()

    sender.cancel_session(session)
    sender.cancel_session(session)

    assert REQUEST_ID in sender._pre_cancelled_rids
    assert sender._send_operation_message.call_count == 4
    assert sender.send_cancel_to_receivers.call_count == 2
    assert len(session._unbound_terminal_results) == 2
    assert session.has_transferring_tasks()

    sender._send_operation_message.return_value = True
    session._sender = sender

    assert session.has_failed()
    assert not session.has_transferring_tasks()
    sender._shutdown_complete = True


def test_tx_session_repeated_cancel_retries_sender_settlement() -> None:
    sender = Mock()
    sender._operation_admission_lock = threading.RLock()
    session = object.__new__(TxSession)
    session._base_args = SessionArgsBase(_params())
    session.request_id = REQUEST_ID
    session.lock = threading.Lock()
    session.kv_tasks = []
    session.aux_task = None
    session._terminal_status = None
    session._accepting_operations = True
    session._closed = True
    session._sender = sender

    session.cancel()
    session.cancel()

    assert session.status is SessionStatus.CANCELLED
    assert not session._accepting_operations
    assert sender.cancel_session.call_count == 2
    sender.cancel_session.assert_called_with(session)


def test_tx_session_error_installs_sender_tombstone() -> None:
    sender = Mock()
    sender._operation_admission_lock = threading.RLock()
    session = object.__new__(TxSession)
    session._base_args = SessionArgsBase(_params())
    session.request_id = REQUEST_ID
    session.lock = threading.Lock()
    session.kv_tasks = []
    session.aux_task = None
    session._terminal_status = None
    session._accepting_operations = True
    session._closed = True
    session._sender = sender

    session.set_exception("transfer failed")

    assert session.status is SessionStatus.ERROR
    assert not session._accepting_operations
    sender.cancel_session.assert_called_once_with(session)


def test_sender_req_info_returns_a_stable_snapshot() -> None:
    info = _recv_info()
    sender = object.__new__(Sender)
    sender._shutdown_complete = True
    sender._instance_rank = 0
    sender._peer_requests_lock = threading.Lock()
    sender._peer_requests = {REQUEST_ID: {(WRITER_RANK, 0): info}}

    snapshot = sender._get_req_info(REQUEST_ID)
    assert snapshot is not None
    snapshot.clear()

    assert sender._peer_requests[REQUEST_ID] == {(WRITER_RANK, 0): info}


def test_external_backend_fence_coordinates_registry_and_bounce_settlement() -> None:
    bounce = _FakeBounce()
    receiver = _make_receiver(bounce=bounce)
    key = (REQUEST_ID, 0)
    assert receiver._recv_registry.prepare(
        key,
        {WRITER_RANK},
        has_bounce_slot=True,
    ).accepted
    assert receiver._recv_registry.begin_publication(key, WRITER_RANK).publication_allowed
    assert receiver._recv_registry.mark_published(key, WRITER_RANK).accepted

    receiver.mark_backend_quiesced()

    snapshot = receiver._recv_registry.context_snapshot(key)
    assert snapshot is not None
    assert snapshot.physical_state is PhysicalState.DRAINING
    assert not receiver._recv_registry.is_request_drained(REQUEST_ID)
    assert len(bounce.backend_quiesced_callbacks) == 1

    callback_key, callback = bounce.backend_quiesced_callbacks[0]
    assert callback_key == key
    callback(True)

    assert receiver._recv_registry.is_request_drained(REQUEST_ID)
    assert receiver._recv_registry.context_snapshot(key).physical_state is PhysicalState.DRAINED


def test_sender_retries_listener_stop_after_first_failure() -> None:
    sender = object.__new__(Sender)
    sender._instance_rank = 0
    sender._shutdown = False
    sender._shutdown_complete = False
    sender._listener_stopped = False
    sender._shutdown_sentinels_sent = False
    sender._messenger = Mock()
    sender._messenger.stop.side_effect = [RuntimeError("busy"), None]
    sender._send_task_queues = [Mock()]
    sender._worker_threads = []
    sender._loaded_remote_agents = set()
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._agent = Mock()
    sender._dealers = {}

    assert sender.shutdown() is False
    sender._send_task_queues[0].put.assert_not_called()

    assert sender.shutdown() is True
    assert sender._messenger.stop.call_count == 2
    sender._send_task_queues[0].put.assert_called_once_with(None)


def test_sender_shutdown_continues_siblings_after_cancel_failure() -> None:
    sender = _make_sender_for_shutdown()
    sender._listener_stopped = False
    sender.retry_terminal_results = Mock()
    first = Mock(disagg_request_id=REQUEST_ID)
    first.cancel.side_effect = [RuntimeError("injected cancel failure"), None]
    first.has_transferring_tasks.return_value = False
    second = Mock(disagg_request_id=REQUEST_ID + 1)
    second.has_transferring_tasks.return_value = False
    sender._sessions = {REQUEST_ID: first, REQUEST_ID + 1: second}

    assert sender.shutdown() is False

    first.cancel.assert_called_once_with()
    first.close.assert_not_called()
    second.cancel.assert_called_once_with()
    second.close.assert_called_once_with()
    sender._messenger.stop.assert_called_once_with()
    assert sender._listener_stopped
    assert sender._sessions == {REQUEST_ID: first}

    assert sender.shutdown() is True
    assert first.cancel.call_count == 2
    first.close.assert_called_once_with()
    assert sender._sessions == {}


def test_sender_shutdown_continues_siblings_after_close_failure() -> None:
    sender = _make_sender_for_shutdown()
    sender.retry_terminal_results = Mock()
    first = Mock(disagg_request_id=REQUEST_ID)
    first.has_transferring_tasks.return_value = False
    first.close.side_effect = [RuntimeError("injected close failure"), None]
    second = Mock(disagg_request_id=REQUEST_ID + 1)
    second.has_transferring_tasks.return_value = False
    sender._sessions = {REQUEST_ID: first, REQUEST_ID + 1: second}

    assert sender.shutdown() is False

    first.close.assert_called_once_with()
    second.close.assert_called_once_with()
    assert sender._sessions == {REQUEST_ID: first}

    assert sender.shutdown() is True
    assert first.close.call_count == 2
    assert sender._sessions == {}


def test_sender_shutdown_forces_cached_terminal_result_progress() -> None:
    sender = _make_sender_for_shutdown()
    info = _recv_info()
    sender._peer_requests = {REQUEST_ID: {(WRITER_RANK, 0): info}}
    sender._send_operation_message = Mock(return_value=True)

    params = _params()
    session = object.__new__(TxSession)
    session._base_args = SessionArgsBase(params)
    session.request_id = REQUEST_ID
    session.lock = threading.Lock()
    session.kv_tasks = []
    session.aux_task = None
    session._unbound_terminal_results = {}
    session._terminal_retry_lock = threading.Lock()
    session._next_terminal_retry_at = float("inf")
    session._sender = sender
    session._need_aux = False
    session._terminal_status = SessionStatus.CANCELLED
    session._closed = False
    session._accepting_operations = False
    session._close_lock = threading.Lock()
    session._aux_buffer = None
    session.aux_slot = None
    source_owner = object()
    session._source_owner = source_owner
    session.cancel = Mock()

    task = KVSendTask(KVSlice(), params, 0, session=session)
    session.kv_tasks.append(task)
    key = (info.instance_name, info.instance_rank)
    admitted, state, _, _ = task.admit_operation(
        key,
        allow_source_access=True,
        no_access_message=(b"unused",),
    )
    assert admitted and state is SendOperationState.PENDING
    terminal_message = (b"terminal",)
    task.cache_terminal_message(key, terminal_message)
    task.finish_source_access()
    task.fail(RuntimeError("logical transfer failure"))
    sender._sessions[REQUEST_ID] = session

    assert sender.shutdown()

    session.cancel.assert_called_once_with()
    sender._send_operation_message.assert_called_once_with(info, terminal_message)
    assert not task.has_pending_result_delivery
    assert sender._sessions == {}
    assert session._source_owner is None
    assert session._closed


def test_sender_shutdown_is_serialized_and_closes_late_control_admission() -> None:
    sender = _make_sender_for_shutdown()
    task_queue = Mock()
    sender._send_task_queues = [task_queue]
    sender._loaded_remote_agents = {"peer0"}
    dealer = Mock()
    sender._dealers = {"tcp://receiver": dealer}
    invalidation_started = threading.Event()
    release_invalidation = threading.Event()

    def invalidate(_agent_name: str) -> None:
        invalidation_started.set()
        assert release_invalidation.wait(timeout=1)

    sender._agent.invalidate_remote_agent.side_effect = invalidate
    results: list[bool] = []
    threads = [threading.Thread(target=lambda: results.append(sender.shutdown())) for _ in range(2)]

    threads[0].start()
    assert invalidation_started.wait(timeout=1)
    threads[1].start()
    sender._handle_cancel_session([MessageType.CANCEL_SESSION, b"999"])
    assert 999 not in sender._pre_cancelled_rids
    assert sender._pre_session_terminal_results == {}

    release_invalidation.set()
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert results == [True, True]
    task_queue.put.assert_called_once_with(None)
    sender._agent.invalidate_remote_agent.assert_called_once_with("peer0")
    dealer.stop.assert_called_once_with()


def test_tx_send_admission_finishes_before_sender_shutdown_sentinel() -> None:
    sender = object.__new__(Sender)
    sender._instance_rank = 0
    sender._operation_admission_lock = threading.RLock()
    sender._shutdown = False
    sender._shutdown_complete = False
    sender._listener_stopped = True
    sender._shutdown_sentinels_sent = False
    task_queue = Mock()
    sender._send_task_queues = [task_queue]
    sender._num_threads = 1
    sender._worker_threads = []
    sender._failed_thread_dealers = []
    sender._failed_thread_dealers_lock = threading.Lock()
    sender._in_doubt_transfers = []
    sender._in_doubt_transfers_lock = threading.Lock()
    sender._loaded_remote_agents = set()
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._agent = Mock()
    sender._dealers = {}
    sender._peer_requests = {REQUEST_ID: {(WRITER_RANK, 0): _recv_info()}}
    sender._peer_requests_lock = threading.Lock()
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()

    session = object.__new__(TxSession)
    session._base_args = SessionArgsBase(_params())
    session.request_id = REQUEST_ID
    session.lock = threading.Lock()
    session.kv_tasks = []
    session.aux_task = None
    session._need_aux = False
    session._terminal_status = None
    session._accepting_operations = True
    session._closed = False
    session._unbound_terminal_results = {}
    session._source_owner = object()
    session._sender = sender
    sender._sessions[REQUEST_ID] = session

    build_started = threading.Event()
    release_build = threading.Event()

    def build(task, _info):
        build_started.set()
        assert release_build.wait(timeout=1)
        return WriteMeta(
            task=task,
            expected_transfers=1,
            peer_name="receiver3",
            peer_rank=WRITER_RANK,
            peer_endpoint="tcp://receiver",
            unique_rid=REQUEST_ID,
            src_ptrs=np.array([], dtype=np.int64),
            dst_ptrs=np.array([], dtype=np.int64),
            sizes=np.array([], dtype=np.int64),
            slice_id=0,
            session=session,
        )

    sender._build_kv_write_meta = build
    send_errors: list[Exception] = []
    shutdown_results: list[bool] = []

    def send() -> None:
        try:
            session.send(KVSlice())
        except Exception as e:
            send_errors.append(e)

    send_thread = threading.Thread(target=send)
    send_thread.start()
    assert build_started.wait(timeout=1)

    shutdown_thread = threading.Thread(target=lambda: shutdown_results.append(sender.shutdown()))
    shutdown_thread.start()
    task_queue.put.assert_not_called()

    release_build.set()
    send_thread.join(timeout=1)
    shutdown_thread.join(timeout=1)

    assert not send_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert not send_errors
    assert shutdown_results == [False]
    assert len(task_queue.put.call_args_list) == 2
    assert isinstance(task_queue.put.call_args_list[0].args[0], WriteMeta)
    assert task_queue.put.call_args_list[1].args[0] is None

    session.kv_tasks[0].finish_source_access()
    session._closed = True
    sender._shutdown_complete = True


def test_sender_retries_failed_worker_local_dealer_close() -> None:
    sender = object.__new__(Sender)
    sender._instance_rank = 0
    sender._failed_thread_dealers = []
    sender._failed_thread_dealers_lock = threading.Lock()
    dealer = Mock()
    dealer.stop.side_effect = [
        RuntimeError("worker close failed"),
        RuntimeError("first retry failed"),
        None,
    ]

    dealers = {"tcp://receiver": dealer}
    sender._close_thread_dealers(0, dealers)

    assert dealers == {}
    assert sender._failed_thread_dealers == [dealer]

    sender._operation_admission_lock = threading.RLock()
    sender._shutdown = False
    sender._shutdown_complete = False
    sender._listener_stopped = True
    sender._shutdown_sentinels_sent = False
    sender._send_task_queues = []
    sender._worker_threads = []
    sender._in_doubt_transfers = []
    sender._in_doubt_transfers_lock = threading.Lock()
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._loaded_remote_agents = set()
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._agent = Mock()
    sender._dealers = {}

    assert sender.shutdown() is False
    assert sender._failed_thread_dealers == [dealer]

    assert sender.shutdown() is True
    assert sender._failed_thread_dealers == []
    assert dealer.stop.call_count == 3


def test_sender_shutdown_retains_agent_for_in_doubt_transfer_context() -> None:
    sender = object.__new__(Sender)
    sender._instance_rank = 0
    sender._shutdown = False
    sender._shutdown_complete = False
    sender._listener_stopped = True
    sender._shutdown_sentinels_sent = False
    sender._send_task_queues = []
    sender._worker_threads = []
    sender._in_doubt_transfers = [object()]
    sender._in_doubt_transfers_lock = threading.Lock()
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._loaded_remote_agents = {"peer0"}
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._agent = Mock()
    sender._dealers = {}

    assert sender.shutdown() is False
    sender._agent.invalidate_remote_agent.assert_not_called()


def test_rank_info_server_retries_listener_stop_after_first_failure() -> None:
    server = object.__new__(RankInfoServer)
    server._shutdown = False
    server._shutdown_started = False
    server._messenger = Mock()
    server._messenger.stop.side_effect = [RuntimeError("busy"), None]

    assert server.shutdown() is False
    assert not server._shutdown

    assert server.shutdown() is True
    assert server._messenger.stop.call_count == 2


@pytest.mark.parametrize("factory", ["create_tx_session", "create_rx_session"])
def test_transfer_worker_rejects_new_sessions_after_shutdown_starts(factory) -> None:
    worker = object.__new__(TransferWorker)
    worker._shutdown = True  # Keep object.__new__ fixtures out of __del__ cleanup.
    worker._shutdown_started = True
    worker._session_admission_lock = threading.Lock()

    with pytest.raises(RuntimeError, match="shutting down"):
        getattr(worker, factory)(SimpleNamespace())


def test_transfer_worker_rx_session_retains_destination_owner_until_close() -> None:
    class _DestinationOwner:
        pass

    owner = _DestinationOwner()
    owner.py_disaggregated_params = _params()
    owner.py_request_id = REQUEST_ID
    owner.prompt_len = 8
    owner.py_beam_width = 1
    owner_ref = weakref.ref(owner)
    receiver = _make_receiver()
    worker = object.__new__(TransferWorker)
    worker._shutdown = True  # Keep object.__new__ fixture out of __del__ cleanup.
    worker._shutdown_started = False
    worker._session_admission_lock = threading.Lock()
    worker._receiver = receiver
    worker._aux_buffer = None
    worker._config = SimpleNamespace(rx_timeout_s=None)

    session = worker.create_rx_session(owner)
    del owner
    gc.collect()

    assert owner_ref() is not None
    assert session._destination_owner is owner_ref()

    session.close()
    gc.collect()

    assert owner_ref() is None


def test_transfer_worker_retains_owner_before_nested_shutdown_failure() -> None:
    worker = object.__new__(TransferWorker)
    worker._shutdown = False
    worker._agent_shutdown_failed = False
    worker._shutdown_started = False
    worker._session_admission_lock = threading.Lock()
    worker._rank_info_server = Mock()
    worker._rank_info_server.shutdown.side_effect = [RuntimeError("listener failed"), True]
    worker._receiver = None
    worker._sender = None
    worker._bounce = None
    worker._agent = None

    try:
        assert worker.shutdown() is False
        assert worker in _NON_DRAINED_TRANSFER_WORKERS

        assert worker.shutdown() is True
        assert worker not in _NON_DRAINED_TRANSFER_WORKERS
    finally:
        _NON_DRAINED_TRANSFER_WORKERS.discard(worker)


def test_transfer_worker_shutdown_attempts_are_serialized() -> None:
    worker = object.__new__(TransferWorker)
    worker._shutdown_attempt_lock = threading.Lock()
    worker._shutdown = False
    worker._agent_shutdown_failed = False
    worker._shutdown_started = False
    worker._session_admission_lock = threading.Lock()
    worker._rank_info_server = Mock()
    worker._receiver = None
    worker._sender = None
    worker._bounce = None
    agent = Mock()
    worker._agent = agent
    descriptor = object()
    worker._registered_mem = [descriptor]
    shutdown_started = threading.Event()
    release_shutdown = threading.Event()

    def stop_rank_info_server() -> bool:
        shutdown_started.set()
        assert release_shutdown.wait(timeout=1)
        return True

    worker._rank_info_server.shutdown.side_effect = stop_rank_info_server
    results: list[bool] = []
    threads = [threading.Thread(target=lambda: results.append(worker.shutdown())) for _ in range(2)]

    try:
        threads[0].start()
        assert shutdown_started.wait(timeout=1)
        threads[1].start()
        release_shutdown.set()
        for thread in threads:
            thread.join(timeout=1)

        assert all(not thread.is_alive() for thread in threads)
        assert results == [True, True]
        worker._rank_info_server.shutdown.assert_called_once_with()
        agent.deregister_memory.assert_called_once_with(descriptor)
        agent.shutdown.assert_called_once_with()
    finally:
        _NON_DRAINED_TRANSFER_WORKERS.discard(worker)


def test_tx_session_retains_source_owner_while_local_access_is_ambiguous() -> None:
    task = SimpleNamespace(status=TaskStatus.TRANSFERRING)
    session = object.__new__(TxSession)
    session._base_args = SessionArgsBase(_params())
    session.request_id = REQUEST_ID
    session.lock = threading.Lock()
    session._terminal_status = SessionStatus.CANCELLED
    session.kv_tasks = [task]
    session._need_aux = False
    session.aux_task = None
    session._closed = False
    session._aux_buffer = None
    session.aux_slot = None
    session._sender = None

    assert not session.has_failed()
    assert session.wait_complete(blocking=False) is None
    with pytest.raises(RuntimeError, match="source work is pending or active"):
        session.close()

    task.status = TaskStatus.TRANSFERRED
    assert session.has_failed()
    assert session.wait_complete(blocking=False) is WaitResult.FAILED
    session.close()


def test_tx_session_includes_aux_source_access_in_drain_gate() -> None:
    kv_task = SimpleNamespace(status=TaskStatus.TRANSFERRED)
    aux_task = SimpleNamespace(status=TaskStatus.TRANSFERRING)
    session = object.__new__(TxSession)
    session._base_args = SessionArgsBase(_params())
    session.request_id = REQUEST_ID
    session.lock = threading.Lock()
    session._terminal_status = SessionStatus.CANCELLED
    session.kv_tasks = [kv_task]
    session._need_aux = True
    session.aux_task = aux_task
    session._closed = False
    session._aux_buffer = None
    session.aux_slot = None
    session._sender = None

    assert session.has_transferring_tasks()
    assert not session.has_failed()
    assert session.wait_complete(blocking=False) is None
    with pytest.raises(RuntimeError, match="source work is pending or active"):
        session.close()

    aux_task.status = TaskStatus.TRANSFERRED
    assert not session.has_transferring_tasks()
    assert session.wait_complete(blocking=False) is WaitResult.FAILED
    session.close()


def test_tx_session_close_is_serialized_across_concurrent_callers() -> None:
    sender = SimpleNamespace(
        _operation_admission_lock=threading.RLock(),
        clear_session=Mock(),
    )
    aux_buffer = Mock()
    free_started = threading.Event()
    release_free = threading.Event()

    def free_slot(_slot: int) -> None:
        free_started.set()
        assert release_free.wait(timeout=1)

    aux_buffer.free_slot.side_effect = free_slot
    session = object.__new__(TxSession)
    session._base_args = SessionArgsBase(_params())
    session.request_id = REQUEST_ID
    session.lock = threading.Lock()
    session._close_lock = threading.Lock()
    session._terminal_status = SessionStatus.CANCELLED
    session.kv_tasks = []
    session.aux_task = None
    session._unbound_terminal_results = {}
    session._closed = False
    session._accepting_operations = False
    session._aux_buffer = aux_buffer
    session.aux_slot = 7
    session._source_owner = object()
    session._sender = sender
    errors: list[Exception] = []

    def close() -> None:
        try:
            session.close()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=close) for _ in range(2)]
    threads[0].start()
    assert free_started.wait(timeout=1)
    threads[1].start()
    release_free.set()
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    aux_buffer.free_slot.assert_called_once_with(7)
    sender.clear_session.assert_called_once_with(REQUEST_ID)
    assert session.aux_slot is None
    assert session._source_owner is None
    assert session._closed


def test_aux_pack_holds_sender_admission_until_fill_finishes() -> None:
    class ObservableRLock:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self.attempted = threading.Event()

        def __enter__(self):
            self.attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
            self._lock.release()

    sender = _make_sender_for_shutdown()
    admission_lock = ObservableRLock()
    sender._operation_admission_lock = admission_lock
    aux_buffer = Mock()
    aux_buffer.alloc_slot.return_value = SimpleNamespace(id=7)
    fill_started = threading.Event()
    release_fill = threading.Event()

    def fill_slot(_slot: int, _request) -> None:
        fill_started.set()
        assert release_fill.wait(timeout=1)

    aux_buffer.fill_slot.side_effect = fill_slot
    session = TxSession(
        request_id=REQUEST_ID,
        params=_params(),
        sender=sender,
        aux_buffer=aux_buffer,
        source_owner=object(),
    )
    admission_lock.attempted.clear()
    pack_errors: list[Exception] = []
    shutdown_results: list[bool] = []

    def pack() -> None:
        try:
            session.pack_aux(SimpleNamespace())
        except Exception as e:
            pack_errors.append(e)

    pack_thread = threading.Thread(target=pack)
    shutdown_thread = threading.Thread(target=lambda: shutdown_results.append(sender.shutdown()))
    pack_thread.start()
    assert fill_started.wait(timeout=1)
    admission_lock.attempted.clear()
    shutdown_thread.start()
    assert admission_lock.attempted.wait(timeout=1)

    assert not sender._shutdown
    aux_buffer.free_slot.assert_not_called()

    release_fill.set()
    pack_thread.join(timeout=1)
    shutdown_thread.join(timeout=1)

    assert not pack_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert pack_errors == []
    assert shutdown_results == [True]
    aux_buffer.fill_slot.assert_called_once()
    aux_buffer.free_slot.assert_called_once_with(7)
    assert session._closed


def test_aux_send_requires_packed_data_and_freezes_slot_contents() -> None:
    sender = _make_sender_for_shutdown()
    aux_buffer = Mock()
    aux_buffer.alloc_slot.return_value = SimpleNamespace(id=7)
    session = TxSession(
        request_id=REQUEST_ID,
        params=_params(),
        sender=sender,
        aux_buffer=aux_buffer,
        source_owner=object(),
    )

    with pytest.raises(RuntimeError, match="must be packed before send"):
        session.send_aux()

    request = SimpleNamespace()
    session.pack_aux(request)
    task = session.send_aux()

    with pytest.raises(RuntimeError, match="frozen after send"):
        session.pack_aux(request)

    assert session.aux_task is task
    aux_buffer.fill_slot.assert_called_once_with(7, request)
    assert sender.shutdown()


def test_task_failure_does_not_hide_a_sibling_source_operation() -> None:
    task = KVSendTask(KVSlice(), _params(), 0)
    task.begin_source_access()
    task.begin_source_access()
    task.fail(RuntimeError("one writer failed"))
    session = object.__new__(TxSession)
    session._base_args = SessionArgsBase(_params())
    session.request_id = REQUEST_ID
    session._terminal_status = SessionStatus.ERROR
    session.kv_tasks = [task]
    session._need_aux = False
    session.aux_task = None

    task.finish_source_access()

    assert task.status is TaskStatus.ERROR
    assert session.has_transferring_tasks()
    assert not session.has_failed()
    assert session.wait_complete(blocking=False) is None

    task.finish_source_access()
    assert not session.has_transferring_tasks()
    assert session.has_failed()


def test_ambiguous_agent_exception_retains_context_when_quarantine_fails() -> None:
    params = _params()
    task = KVSendTask(KVSlice(), params, 0)
    source_owner = object()
    session = SimpleNamespace(
        kv_tasks=[task],
        lock=threading.Lock(),
        status=SessionStatus.READY,
        source_owner=source_owner,
    )
    bounce = Mock()
    bounce.build_request.return_value = (object(), 11)
    bounce.quarantine_send.side_effect = RuntimeError("quarantine failed")
    sender = object.__new__(Sender)
    sender._sessions_lock = threading.Lock()
    sender._get_session = lambda _rid: session
    sender._bounce = bounce
    sender._agent = Mock()
    sender._agent.submit_transfer_requests.side_effect = RuntimeError("submit crossed backend")
    sender._device_id = 0
    sender._instance_rank = 0
    sender._in_doubt_transfers = []
    sender._in_doubt_transfers_lock = threading.Lock()
    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="receiver0",
        peer_rank=0,
        peer_endpoint="tcp://receiver",
        unique_rid=REQUEST_ID,
        src_ptrs=np.array([100], dtype=np.int64),
        dst_ptrs=np.array([200], dtype=np.int64),
        sizes=np.array([8], dtype=np.int64),
        dst_device_id=0,
        slice_id=0,
        is_last_slice=True,
        bounce_dst_base=300,
        session=session,
    )

    with pytest.raises(TransferSourceInDoubtError):
        sender._deliver_kv_to_agent(write_meta)

    assert task.status is TaskStatus.TRANSFERRING
    assert task.source_access_active
    bounce.quarantine_send.assert_called_once_with(11)
    bounce.release_send.assert_not_called()
    assert len(sender._in_doubt_transfers) == 1
    context = sender._in_doubt_transfers[0]
    assert context.session is session
    assert context.source_owner is source_owner
    assert context.transfer_request is bounce.build_request.return_value[0]
    assert context.bounce_slot_id == 11


def test_ambiguous_gather_rollback_retains_sender_source_context() -> None:
    from tensorrt_llm._torch.disaggregation.native.bounce import GatherSourceInDoubtError

    params = _params()
    task = KVSendTask(KVSlice(), params, 0)
    source_owner = object()
    session = SimpleNamespace(
        kv_tasks=[task],
        lock=threading.Lock(),
        status=SessionStatus.READY,
        source_owner=source_owner,
    )
    sender = object.__new__(Sender)
    sender._sessions_lock = threading.Lock()
    sender._get_session = lambda _rid: session
    sender._bounce = Mock()
    sender._bounce.build_request.side_effect = GatherSourceInDoubtError(
        "gather rollback has no positive fence"
    )
    sender._agent = Mock()
    sender._device_id = 0
    sender._instance_rank = 0
    sender._in_doubt_transfers = []
    sender._in_doubt_transfers_lock = threading.Lock()
    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="receiver0",
        peer_rank=0,
        peer_endpoint="tcp://receiver",
        unique_rid=REQUEST_ID,
        src_ptrs=np.array([100], dtype=np.int64),
        dst_ptrs=np.array([200], dtype=np.int64),
        sizes=np.array([8], dtype=np.int64),
        dst_device_id=0,
        slice_id=0,
        is_last_slice=True,
        bounce_dst_base=300,
        session=session,
    )

    with pytest.raises(GatherSourceInDoubtError):
        sender._deliver_kv_to_agent(write_meta)

    assert task.source_access_active
    sender._agent.submit_transfer_requests.assert_not_called()
    assert len(sender._in_doubt_transfers) == 1
    context = sender._in_doubt_transfers[0]
    assert context.session is session
    assert context.source_owner is source_owner
    assert context.transfer_request is None


def test_fenced_pre_submit_failure_reports_terminal_writer_result() -> None:
    params = _params()
    task = KVSendTask(KVSlice(), params, 0)
    session = SimpleNamespace(
        kv_tasks=[task],
        lock=threading.Lock(),
        status=SessionStatus.READY,
        source_owner=object(),
    )
    sender = object.__new__(Sender)
    sender._sessions_lock = threading.Lock()
    sender._get_session = lambda _rid: session
    sender._bounce = Mock()
    sender._bounce.build_request.side_effect = RuntimeError("descriptor build failed")
    sender._agent = Mock()
    sender._device_id = 0
    sender._instance_rank = 0
    dealer = Mock()
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="receiver0",
        peer_rank=0,
        peer_endpoint="tcp://receiver",
        unique_rid=REQUEST_ID,
        src_ptrs=np.array([100], dtype=np.int64),
        dst_ptrs=np.array([200], dtype=np.int64),
        sizes=np.array([8], dtype=np.int64),
        dst_device_id=0,
        slice_id=0,
        is_last_slice=True,
        bounce_dst_base=300,
        session=session,
    )

    sender._deliver_kv_to_agent(write_meta)

    assert task.status is TaskStatus.ERROR
    assert not task.source_access_active
    sender._agent.submit_transfer_requests.assert_not_called()
    dealer.send.assert_called_once()
    result = dealer.send.call_args.args[0]
    assert result[0] == MessageType.KV_AGENT_RESULT
    assert _KV_RESULT_PREFIX.unpack(result[1])[-1] == 1


def test_tx_poll_retries_cached_terminal_result_after_worker_send_failure() -> None:
    params = _params()
    sender = object.__new__(Sender)
    sender._shutdown_complete = True
    sender._instance_rank = 0
    sender._peer_requests = {REQUEST_ID: {(WRITER_RANK, 0): _recv_info()}}
    sender._peer_requests_lock = threading.Lock()
    sender._send_operation_message = Mock(return_value=True)
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="sender", instance_rank=0)
    )
    sender._device_id = 0
    dealer = Mock()
    dealer.send.side_effect = RuntimeError("transient send failure")
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)

    session = object.__new__(TxSession)
    session._base_args = SessionArgsBase(params)
    session.request_id = REQUEST_ID
    session.lock = threading.Lock()
    session.kv_tasks = []
    session.aux_task = None
    session._need_aux = False
    session._terminal_status = None
    session.receiver_ready = True
    session._accepting_operations = True
    session._unbound_terminal_results = {}
    session._terminal_retry_lock = threading.Lock()
    session._next_terminal_retry_at = 0.0
    session._source_owner = object()
    session._sender = sender
    session._closed = False

    task = KVSendTask(KVSlice(), params, 0, session=session)
    session.kv_tasks.append(task)
    key = ("receiver", WRITER_RANK)
    admitted, state, _message, _delivered = task.admit_operation(
        key,
        allow_source_access=True,
        no_access_message=(b"unused",),
    )
    assert admitted
    assert state is SendOperationState.PENDING
    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="receiver3",
        peer_rank=WRITER_RANK,
        peer_endpoint="tcp://receiver",
        unique_rid=REQUEST_ID,
        src_ptrs=np.array([], dtype=np.int64),
        dst_ptrs=np.array([], dtype=np.int64),
        sizes=np.array([], dtype=np.int64),
        dst_device_id=0,
        slice_id=0,
        is_last_slice=True,
        session=session,
        source_access_enrolled=True,
        operation_key=key,
    )

    sender._deliver_kv_to_agent(write_meta)

    assert not task.source_access_active
    assert task.has_pending_result_delivery
    assert session.has_transferring_tasks()
    assert task.status is TaskStatus.TRANSFERRED

    assert not session.has_failed()
    sender._send_operation_message.assert_called_once()
    assert not task.has_pending_result_delivery


def test_aux_poll_retries_success_after_worker_result_send_failure() -> None:
    params = _params()
    sender = object.__new__(Sender)
    sender._shutdown_complete = True
    sender._instance_rank = 0
    sender._peer_requests = {REQUEST_ID: {(WRITER_RANK, 0): _recv_info()}}
    sender._peer_requests_lock = threading.Lock()
    sender._send_operation_message = Mock(return_value=True)
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="sender", instance_rank=0)
    )
    sender._device_id = 0
    dealer = Mock()
    dealer.send.side_effect = RuntimeError("transient send failure")
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)

    session = object.__new__(TxSession)
    session._base_args = SessionArgsBase(params)
    session.request_id = REQUEST_ID
    session.lock = threading.Lock()
    session.kv_tasks = []
    session._need_aux = True
    session._terminal_status = None
    session.receiver_ready = True
    session._accepting_operations = True
    session._unbound_terminal_results = {}
    session._terminal_retry_lock = threading.Lock()
    session._next_terminal_retry_at = 0.0
    session._source_owner = object()
    session._sender = sender
    session._closed = False

    task = AuxSendTask(params, 7, session=session)
    session.aux_task = task
    key = ("receiver", WRITER_RANK)
    admitted, state, _message, _delivered = task.admit_operation(
        key,
        allow_source_access=True,
        no_access_message=(b"unused",),
    )
    assert admitted
    assert state is SendOperationState.PENDING
    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="receiver3",
        peer_rank=WRITER_RANK,
        peer_endpoint="tcp://receiver",
        unique_rid=REQUEST_ID,
        src_ptrs=np.array([], dtype=np.int64),
        dst_ptrs=np.array([], dtype=np.int64),
        sizes=np.array([], dtype=np.int64),
        dst_device_id=0,
        meta_type=WriteMetaType.AUX,
        session=session,
        source_access_enrolled=True,
        operation_key=key,
    )

    sender._deliver_aux_to_agent(write_meta)

    assert not task.source_access_active
    assert task.has_pending_result_delivery
    assert session.has_transferring_tasks()
    assert task.status is TaskStatus.TRANSFERRED

    assert not session.has_failed()
    sender._send_operation_message.assert_called_once()
    assert not task.has_pending_result_delivery


def test_async_send_enrolls_source_owner_before_launch_and_retains_on_error() -> None:
    request = SimpleNamespace(
        request_id=REQUEST_ID,
        py_disaggregated_params=None,
        state=None,
    )
    session = Mock()
    session.has_transferring_tasks.return_value = True
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = False
    transceiver._send_reqs = {}
    transceiver._send_sessions = {REQUEST_ID: session}

    def get_session(owner):
        assert transceiver._send_reqs == {REQUEST_ID: owner}
        return session

    def fail_send(_slice):
        assert transceiver._send_reqs == {REQUEST_ID: request}
        raise RuntimeError("launch failed")

    transceiver._get_or_create_send_session = get_session
    transceiver._create_kv_slice = Mock(return_value=object())
    session.send.side_effect = fail_send

    with pytest.raises(RuntimeError, match="launch failed"):
        transceiver.respond_and_send_async(request)

    assert transceiver._send_reqs == {REQUEST_ID: request}
    assert transceiver._send_sessions == {REQUEST_ID: session}
    session.cancel.assert_called_once_with()
