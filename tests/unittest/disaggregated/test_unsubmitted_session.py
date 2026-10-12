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
"""Session acknowledgments preserve sender admission and physical-access ordering."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base import Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native import transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.native.retirement import RetirementWatchdog
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only


def _params() -> DisaggregatedParams:
    return DisaggregatedParams(
        disagg_request_id=401, schedule_style=DisaggScheduleStyle.GENERATION_FIRST
    )


def _sender(*, ownership: bool = True) -> transfer_mod.Sender:
    """Use real session/admission logic without network or backend endpoints."""
    sender = object.__new__(transfer_mod.Sender)
    sender._enforce_physical_ownership = ownership
    sender._sessions_lock, sender._sessions = threading.Lock(), {}
    sender._pre_cancelled_rids = {}
    sender._peer_requests_lock = threading.Lock()
    sender._peer_requests, sender._peer_requests_timestamps = {}, {}
    sender._shutdown, sender._shutdown_requested = True, False
    sender._retirement_watchdog = None
    sender._instance_rank = 7
    sender._num_threads = 1
    sender._send_task_queues = [queue.Queue()]
    sender._pending_settlements = [{}]
    sender._pending_session_quiescence = {}
    sender._get_or_connect_thread_dealer = Mock(return_value=Mock())
    sender._ownership_poisoned = None
    sender._ownership_poison_lock = threading.Lock()
    sender._agent = Mock()
    sender._registrar = Mock()
    sender._registrar.get_peer_rank_info.return_value = SimpleNamespace(
        instance_name="gen", instance_rank=0, dp_rank=0, self_endpoint="tcp://gen:1234"
    )
    sender._registrar.get_peer_overlap.return_value = SimpleNamespace(ranks=[0])
    return sender


def _cancel(*, routing: bool = False) -> list[bytes]:
    message = [transfer_mod.MessageType.CANCEL_SESSION, b"401"]
    return message + [b"gen", b"0"] if routing else message


def _request_data() -> list[bytes]:
    info = transfer_mod.RecvReqInfo(
        sender_req_id=401,
        instance_name="gen",
        instance_rank=0,
        block_ids_per_layer_groups=[],
        unique_rid=401,
        aux_slot=3,
        slice_id=0,
    )
    return [transfer_mod.MessageType.REQUEST_DATA, info.to_bytes()]


def _assert_session_ack(sender: transfer_mod.Sender) -> None:
    marker = sender._send_task_queues[0].get_nowait()
    assert isinstance(marker, transfer_mod._SessionQuiescence)
    assert sender._send_session_quiesced(marker)
    sender._get_or_connect_thread_dealer.assert_called_with("tcp://gen:1234")
    dealer = sender._get_or_connect_thread_dealer.return_value
    dealer.send.assert_called_once_with([transfer_mod.MessageType.SESSION_QUIESCED, b"7", b"401"])
    dealer.send.reset_mock()
    sender._agent.submit_transfer_requests.assert_not_called()


def _assert_no_session_ack(sender: transfer_mod.Sender) -> None:
    marker = sender._send_task_queues[0].get_nowait()
    assert isinstance(marker, transfer_mod._SessionQuiescence)
    assert not sender._send_session_quiesced(marker)
    sender._send_task_queues[0].put(marker)
    sender._get_or_connect_thread_dealer.return_value.send.assert_not_called()


@pytest.mark.parametrize("cancel_first", [False, True])
def test_pre_session_cancel_acknowledges_without_waiting_for_scheduler(cancel_first: bool) -> None:
    sender = _sender()
    if cancel_first:
        sender._handle_cancel_session(_cancel(routing=True))
        _assert_session_ack(sender)
    sender._respond_with_kv(b"gen", _request_data())
    if not cancel_first:
        sender._handle_cancel_session(_cancel(routing=True))
    _assert_session_ack(sender)
    assert not sender._sessions
    late = transfer_mod.TxSession(401, _params(), sender)
    try:
        assert late.status is SessionStatus.CANCELLED
        if not cancel_first:
            # Saved gen-first metadata must not synthesize KV/AUX writer results.
            _assert_session_ack(sender)
        assert sender._send_task_queues[0].empty()
    finally:
        late.close()


def test_metadata_expiry_does_not_prevent_cancel_acknowledgment() -> None:
    sender = _sender()
    sender._respond_with_kv(b"gen", _request_data())
    sender._peer_requests_timestamps[401] = float("-inf")
    sender.sweep_stale_req_infos()
    assert sender._get_req_info(401) is None

    sender._handle_cancel_session(_cancel(routing=True))
    _assert_session_ack(sender)
    sender._respond_with_kv(b"gen", _request_data())
    _assert_session_ack(sender)
    assert not sender._sessions


@pytest.mark.parametrize("pending_settlement", [False, True])
@pytest.mark.parametrize("ack_fails_once", [False, True])
def test_worker_orders_session_ack_after_piece_reports(
    monkeypatch: pytest.MonkeyPatch, pending_settlement: bool, ack_fails_once: bool
) -> None:
    sender = _sender()
    sender._device_id, sender._thread_local = 0, threading.local()
    work_queue = sender._send_task_queues[0]
    result = transfer_mod._make_kv_result_msg(7, 401, 0, True, transfer_mod.AgentResult.SUCCESS)
    acknowledgement = [transfer_mod.MessageType.SESSION_QUIESCED, b"7", b"401"]
    sent, errors = [], queue.Queue()
    ack_attempts = 0
    acknowledged = threading.Event()

    def send(message: list[bytes]) -> None:
        nonlocal ack_attempts
        if message == acknowledgement:
            ack_attempts += 1
            if ack_fails_once and ack_attempts == 1:
                assert (401, 0) in sender._pending_session_quiescence
                raise RuntimeError("ACK delivery failed")
        sent.append(message)
        if message == acknowledgement:
            acknowledged.set()
            work_queue.put(None)

    def run_worker() -> None:
        try:
            sender._process_task_queue(0)
        except Exception as error:
            errors.put(error)

    sender._get_or_connect_thread_dealer.return_value.send.side_effect = send
    monkeypatch.setattr(transfer_mod.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transfer_mod.cudart, "cudaSetDevice", Mock(return_value=0))
    monkeypatch.setattr(transfer_mod, "CUASSERT", Mock())
    session = None
    if pending_settlement:
        session = transfer_mod.TxSession(401, _params(), sender)
        task = transfer_mod.KVSendTask(Chunk([], [], TokenRange(0, 16), True), _params(), 0)
        task.bind_logical_outcomes(session._logical_outcomes)
        session.kv_tasks.append(task)
        assert task.begin_physical_operation(0)
        task.begin_backend_submission(0, object())
        task.record_backend_submission(0, Mock(is_completed=Mock(return_value=True)))
        task.mark_physical_operation_in_doubt(0)
        empty = np.empty(0, dtype=np.int64)
        meta = transfer_mod.WriteMeta(task, 1, "gen", 0, "tcp://gen:1234", 401, empty, empty, empty)
        initial = transfer_mod._make_kv_result_msg(
            7, 401, 0, False, transfer_mod.AgentResult.IN_DOUBT
        )
        settled = transfer_mod._make_kv_result_msg(
            7, 401, 0, True, transfer_mod.AgentResult.FAILED_QUIESCED
        )
        sender._pending_settlements[0][(task, 0)] = transfer_mod._PendingSettlement(meta, initial)
        expected = [initial, settled, acknowledgement]
    else:
        work_queue.put(("tcp://gen:1234", result))
        expected = [result, acknowledgement]
    sender._handle_cancel_session(_cancel(routing=True))
    assert not sent, "the listener must not bypass earlier worker results"
    worker = threading.Thread(target=run_worker, daemon=True)
    worker.start()
    try:
        assert acknowledged.wait(5), "the worker did not deliver the session acknowledgment"
    finally:
        work_queue.put(None)
        worker.join(timeout=5)
    assert not worker.is_alive()
    if not errors.empty():
        raise errors.get_nowait()
    assert sent == expected
    assert ack_attempts == (2 if ack_fails_once else 1)
    assert not sender._pending_session_quiescence
    assert not sender._pending_settlements[0]
    if session is not None:
        assert session.status is SessionStatus.CANCELLED
        assert session.resources_drained()
        assert session.close()
    sender._agent.submit_transfer_requests.assert_not_called()


def test_pre_session_cancel_remains_closed_after_late_session_cleanup() -> None:
    sender = _sender()
    sender._handle_cancel_session(_cancel())
    first = transfer_mod.TxSession(401, _params(), sender)
    assert first.status is SessionStatus.CANCELLED
    assert first.close()

    second = transfer_mod.TxSession(401, _params(), sender)
    try:
        assert second.status is SessionStatus.CANCELLED, (
            "session cleanup reopened cancelled admission"
        )
        assert second.resources_drained()
        sender._agent.submit_transfer_requests.assert_not_called()
    finally:
        second.close()


def test_existing_session_cancel_remains_closed_after_cleanup() -> None:
    sender = _sender()
    first = transfer_mod.TxSession(401, _params(), sender)
    sender._handle_cancel_session(_cancel())
    assert first.status is SessionStatus.CANCELLED
    assert first.close()

    second = transfer_mod.TxSession(401, _params(), sender)
    try:
        assert second.status is SessionStatus.CANCELLED
        assert second.cancelled_by_peer
        sender._agent.submit_transfer_requests.assert_not_called()
    finally:
        second.close()


@pytest.mark.parametrize("cancelled_by_peer", [False, True])
def test_repeated_cancel_preserves_first_origin(cancelled_by_peer: bool) -> None:
    sender = _sender()
    sender._pre_cancelled_rids[401] = cancelled_by_peer
    for _ in range(2):
        sender._handle_cancel_session(_cancel())
        session = transfer_mod.TxSession(401, _params(), sender)
        try:
            assert session.status is SessionStatus.CANCELLED
            assert session.cancelled_by_peer is cancelled_by_peer
        finally:
            session.close()


def test_legacy_sender_still_consumes_pre_cancel_once() -> None:
    sender = _sender(ownership=False)
    sender._handle_cancel_session(_cancel())
    first = transfer_mod.TxSession(401, _params(), sender)
    assert first.status is SessionStatus.CANCELLED
    assert first.cancelled_by_peer
    assert 401 not in sender._pre_cancelled_rids
    assert first.close()

    second = transfer_mod.TxSession(401, _params(), sender)
    try:
        assert second.status is SessionStatus.INIT
    finally:
        second.close()


def test_metadata_expiry_does_not_remove_cancelled_admission_fence() -> None:
    sender = _sender()
    sender._handle_cancel_session(_cancel())
    sender._peer_requests[401] = {}
    sender._peer_requests_timestamps[401] = float("-inf")
    sender.sweep_stale_req_infos()
    assert sender._get_req_info(401) is None

    session = transfer_mod.TxSession(401, _params(), sender)
    try:
        assert session.status is SessionStatus.CANCELLED
        assert sender._pre_cancelled_rids.get(401) is True
    finally:
        session.close()


def test_cancel_does_not_acknowledge_while_a_physical_write_is_unproven() -> None:
    sender = _sender()
    session = transfer_mod.TxSession(401, _params(), sender)
    task = transfer_mod.SendTaskBase(_params())
    task.bind_logical_outcomes(session._logical_outcomes)
    session.kv_tasks.append(task)
    assert task.begin_physical_operation(0)
    request = object()
    task.begin_backend_submission(0, request)
    status = Mock(is_completed=Mock(return_value=False))
    task.record_backend_submission(0, status)
    task.mark_physical_operation_in_doubt(0)

    sender._handle_cancel_session(_cancel(routing=True))
    assert session.status is SessionStatus.CANCELLED
    assert not session.resources_drained()
    _assert_no_session_ack(sender)
    assert task._physical_operations[0].request is request
    assert task._physical_operations[0].status is status
    assert not session.close()
    with pytest.raises(RuntimeError):
        transfer_mod.TxSession(401, _params(), sender)
    assert sender._get_session(401) is session
    assert task._physical_operations[0].request is request

    status.is_completed.return_value = True
    assert task.poll_in_doubt_physical_operation(0)
    _assert_session_ack(sender)
    assert session.close()


@pytest.mark.parametrize("piece", ["kv", "aux"])
@pytest.mark.parametrize("state", ["admitted", "submitting"])
def test_session_ack_waits_for_admitted_and_submitting_pieces(piece: str, state: str) -> None:
    sender = _sender()
    session = transfer_mod.TxSession(401, _params(), sender)
    task = transfer_mod.SendTaskBase(_params())
    task.bind_logical_outcomes(session._logical_outcomes)
    if piece == "kv":
        session.kv_tasks.append(task)
    else:
        session.aux_task = task
    assert task.begin_physical_operation(0)
    if state == "submitting":
        task.begin_backend_submission(0, object())
    sender._handle_cancel_session(_cancel(routing=True))
    assert not session.resources_drained()
    _assert_no_session_ack(sender)
    if state == "admitted":
        task.retire_unsubmitted_physical_operation(0)
    else:
        task.record_backend_submission(0, Mock(is_completed=Mock(return_value=True)))
        assert task.retire_backend_done_physical_operation(0)
    _assert_session_ack(sender)
    assert session.close()


def test_cancel_wins_while_late_session_creation_waits_for_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _sender()
    locked, creating = threading.Event(), threading.Event()
    created = queue.Queue()
    lookup = sender._get_session

    def get_session(rid: int) -> transfer_mod.TxSession | None:
        locked.set()
        assert creating.wait(5)
        return lookup(rid)

    def create_session() -> None:
        assert locked.wait(5)
        creating.set()
        created.put(transfer_mod.TxSession(401, _params(), sender))

    monkeypatch.setattr(sender, "_get_session", get_session)
    thread = threading.Thread(target=create_session)
    thread.start()
    try:
        sender._handle_cancel_session(_cancel(routing=True))
        thread.join(timeout=5)
        assert not thread.is_alive()
        session = created.get_nowait()
        assert session.status is SessionStatus.CANCELLED
        assert session.close()
        _assert_session_ack(sender)
    finally:
        locked.set()
        creating.set()
        thread.join(timeout=5)


def _receive_session(
    watchdog: RetirementWatchdog | None = None, *, cancel: bool = True
) -> transfer_mod.RxSession:
    receiver = object.__new__(transfer_mod.Receiver)
    receiver._enforce_physical_ownership = True
    receiver._sessions_lock, receiver._sessions = threading.Lock(), {}
    receiver._pre_cancelled_rids = {}
    receiver._retirement_watchdog = watchdog
    receiver._shutdown = False
    receiver._ownership_admission_lock = threading.Lock()
    receiver._ownership_poisoned = None
    receiver._bounce = Mock()
    receiver._dealers = {}
    receiver._messenger = Mock()
    receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="gen", instance_rank=0)
    )
    receiver._get_or_connect_dealer = Mock(return_value=Mock())
    session = transfer_mod.RxSession(
        401, _params(), receiver, timeout_s=1.0, retirement_watchdog=watchdog
    )
    task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), True))
    assert task is not None
    task.expected_transfers = 2
    candidates, published = {0, 1, 2, 3}, set()
    assert session.try_begin_transfer(
        0,
        {"tcp://ctx:1234"},
        writer_candidates=candidates,
        publish=lambda: published.update(candidates),
        published_writers=published,
    )
    if cancel:
        session.cancel_local()
        session.notify_cancel()
    return session


@pytest.mark.parametrize("actual_writer_results", [False, True])
def test_adp_candidate_acks_do_not_replace_selected_writer_or_aux_evidence(
    actual_writer_results: bool,
) -> None:
    session = _receive_session()
    task = session._kv_tasks[0]
    for rank in (0, 1):
        session.process_session_quiesced(rank)
        session.process_session_quiesced(rank)
    assert not task._get_physical_owner().all_writers_reported
    assert not session.resources_drained()
    if actual_writer_results:
        for rank in (2, 3):
            session.process_kv_agent_result(rank, 0, True, transfer_mod.AgentResult.FAILED)
        assert task._get_physical_owner().all_writers_reported
        assert not session.resources_drained(), "KV completion cannot retire outstanding AUX"
    session.process_session_quiesced(2)
    assert not session.resources_drained()
    session.process_session_quiesced(3)
    session.process_session_quiesced(3)
    assert session.resources_drained()
    assert session.status is SessionStatus.CANCELLED
    assert not session.is_completed()
    assert session.close()


@pytest.mark.parametrize("unsettled", ["backend", "local_cuda"])
def test_session_ack_cannot_bypass_existing_unsettled_access(unsettled: str) -> None:
    owner = transfer_mod._ReceiveOperationOwner()
    owner.begin_publication()
    owner.seal_writer_cohort(1, published_writers={7, 8})
    owner.finish_publication()
    if unsettled == "backend":
        owner.record_writer_in_doubt(7)
    else:
        owner.record_writer_result(7, True, wait_for_local_completion=True)
    owner.record_session_quiesced(7)
    owner.record_session_quiesced(8)
    assert not owner.resources_drained
    if unsettled == "backend":
        owner.record_writer_settlement(7)
    else:
        owner.finish_local_completion()
    assert owner.resources_drained


def test_partial_publication_ignores_ack_from_known_unpublished_candidate() -> None:
    owner = transfer_mod._ReceiveOperationOwner()
    owner.begin_publication()
    owner.seal_writer_cohort(1, published_writers={7, 8})
    owner.abort_publication({7})
    owner.record_session_quiesced(8)
    assert not owner.resources_drained
    owner.record_session_quiesced(7)
    assert owner.resources_drained


def test_stale_or_unsolicited_ack_cannot_settle_an_active_receive() -> None:
    session = _receive_session(cancel=False)
    for rid in (400, 401):
        for rank in (0, 1, 2, 3):
            session._receiver._process_session_quiesced(
                [
                    transfer_mod.MessageType.SESSION_QUIESCED,
                    str(rank).encode("ascii"),
                    str(rid).encode("ascii"),
                ]
            )
    assert session.status is SessionStatus.TRANSFERRING
    assert not session.resources_drained()
    assert not session.close()


def test_foreign_session_ack_is_not_retirement_evidence() -> None:
    session = _receive_session()
    with pytest.raises(RuntimeError):
        session.process_session_quiesced(4)
    for rank in (0, 1, 2, 3):
        session.process_session_quiesced(rank)
    assert not session.resources_drained()
    assert session._receiver._ownership_poisoned is not None


def test_missing_candidate_still_expires_and_late_ack_cannot_reverse_fatal() -> None:
    now, contain = [10.0], Mock()
    watchdog = RetirementWatchdog(contain, clock=lambda: now[0])
    session = _receive_session(watchdog)
    for rank in (0, 1, 2):
        session.process_session_quiesced(rank)
    now[0] = 10.5
    session.process_session_quiesced(2)
    watchdog.progress()
    contain.assert_not_called()
    now[0] = 11.0
    watchdog.progress()
    fatal = watchdog.fatal
    assert fatal is not None and fatal.deadline == 11.0
    session.process_session_quiesced(3)
    watchdog.progress()
    contain.assert_called_once_with(fatal)
    assert watchdog.fatal is fatal
    assert not session.resources_drained()
    assert not session.close()


@pytest.mark.parametrize("terminal", ["cancellation", "timeout"])
def test_nonblocking_gen_progress_notifies_unsettled_senders_once(terminal: str) -> None:
    now = [10.0]
    watchdog = RetirementWatchdog(Mock(), clock=lambda: now[0])
    session = _receive_session(watchdog, cancel=False)
    if terminal == "cancellation":
        session.cancel_local()
    else:
        now[0] = 11.0
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._mapping = SimpleNamespace(enable_attention_dp=False, world_size=1, pp_size=1)
    transceiver._recv_sessions, transceiver._recv_reqs = {401: session}, {401: object()}
    transceiver._gen_allgather = Mock()
    for _ in range(2):
        assert transceiver.check_gen_transfer_status(0) == ([], [], [])
    session._receiver._get_or_connect_dealer.return_value.send.assert_called_once_with(
        _cancel(routing=True)
    )
    assert transceiver._recv_sessions == {401: session}
    assert not session.resources_drained()
    for rank in (0, 1, 2, 3):
        session._receiver._process_session_quiesced(
            [transfer_mod.MessageType.SESSION_QUIESCED, str(rank).encode("ascii"), b"401"]
        )
    assert session.resources_drained()
    assert session.close()
    watchdog.stop()
