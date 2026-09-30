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
"""Late physical evidence must not turn an unsuccessful transfer into delivery."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.base import Cancelled, Chunk, Failed, TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native.handle import TaskHandle
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle


def _params() -> DisaggregatedParams:
    return DisaggregatedParams(
        disagg_request_id=401, schedule_style=DisaggScheduleStyle.GENERATION_FIRST
    )


def _chunk() -> Chunk:
    return Chunk([], [], TokenRange(0, 16), True)


def _in_doubt_task(status: object | None) -> transfer_mod.SendTaskBase:
    task = transfer_mod.SendTaskBase(_params())
    assert task.begin_physical_operation(7)
    task.begin_backend_submission(7, object())
    if status is not None:
        task.record_backend_submission(7, status)
    task.mark_physical_operation_in_doubt(7)
    return task


def _owner(cohort: set[int]) -> transfer_mod._ReceiveOperationOwner:
    owner = transfer_mod._ReceiveOperationOwner()
    owner.begin_publication()
    owner.seal_writer_cohort(len(cohort), cohort)
    owner.finish_publication()
    return owner


def _sender() -> transfer_mod.Sender:
    sender = object.__new__(transfer_mod.Sender)
    sender._enforce_physical_ownership = True
    sender._sessions_lock, sender._sessions = threading.Lock(), {}
    sender._shutdown = sender._shutdown_requested = False
    sender._ownership_poisoned, sender._ownership_poison_lock = None, threading.Lock()
    sender._instance_rank = 7
    sender._device_id = 0
    sender._num_threads = 1
    sender._pending_settlements = [{}]
    sender._bounce = Mock()
    return sender


@pytest.mark.cpu_only
@pytest.mark.parametrize("query", [None, False, "DONE", RuntimeError("query failed")])
def test_late_settlement_requires_positive_retained_status(query: object) -> None:
    status = None if query is None else Mock()
    if status is not None:
        if isinstance(query, Exception):
            status.is_completed.side_effect = query
        else:
            status.is_completed.return_value = query
    task = _in_doubt_task(status)
    operation = task._physical_operations[7]
    request = operation.request

    assert not task.poll_in_doubt_physical_operation(7)
    assert operation.state is transfer_mod._PhysicalOperationState.IN_DOUBT
    assert operation.request is request
    assert operation.status is status
    assert not task.resources_drained


@pytest.mark.cpu_only
def test_late_settlement_rejects_replaced_status() -> None:
    status = Mock()
    task = _in_doubt_task(status)
    replacement = Mock()

    def replace() -> bool:
        task._physical_operations[7].status = replacement
        return True

    status.is_completed.side_effect = replace
    assert not task.poll_in_doubt_physical_operation(7)
    assert not task.resources_drained
    assert task._physical_operations[7].status is replacement


@pytest.mark.cpu_only
def test_concurrent_late_done_retires_once_without_changing_failure() -> None:
    barrier = threading.Barrier(2, timeout=10)
    status = Mock()

    def done() -> bool:
        barrier.wait()
        return True

    status.is_completed.side_effect = done
    task = _in_doubt_task(status)
    original_error = RuntimeError("original failure")
    task.fail(original_error)
    outcomes = [False, False]

    def poll(index: int) -> None:
        outcomes[index] = task.poll_in_doubt_physical_operation(7)

    threads = [threading.Thread(target=poll, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert sum(outcomes) == 1
    assert task.resources_drained
    assert task.status is transfer_mod.TaskStatus.ERROR
    assert task._exception is original_error
    assert task._physical_operations[7].request is None
    assert task._physical_operations[7].status is None
    assert not task.poll_in_doubt_physical_operation(7)


@pytest.mark.cpu_only
def test_late_done_does_not_retire_active_sibling() -> None:
    task = _in_doubt_task(Mock(is_completed=Mock(return_value=True)))
    assert task.begin_physical_operation(8)
    assert task.poll_in_doubt_physical_operation(7)
    assert not task.resources_drained
    task.retire_unsubmitted_physical_operation(8)
    assert task.resources_drained


@pytest.mark.cpu_only
@pytest.mark.parametrize("count_only", [False, True])
def test_only_explicit_settlement_closes_ambiguous_writer(count_only: bool) -> None:
    owner = transfer_mod._ReceiveOperationOwner()
    owner.begin_publication()
    # Gen-first ADP counts the immutable, no-retry selected group's writers.
    owner.seal_writer_cohort(2, None if count_only else {7, 8})
    owner.finish_publication()
    assert owner.record_writer_in_doubt(7)
    assert owner.record_writer_result(7, False, wait_for_local_completion=False) == (False, False)
    assert not owner.all_writers_reported
    assert not owner.resources_drained
    assert owner.record_writer_settlement(7)
    assert not owner.resources_drained
    assert not owner.record_writer_settlement(7)
    assert not owner.record_writer_in_doubt(7)
    owner.record_writer_result(8, False, wait_for_local_completion=False)
    assert owner.resources_drained


@pytest.mark.cpu_only
@pytest.mark.parametrize("foreign", [False, True])
def test_reordered_or_foreign_settlement_stays_invalid(foreign: bool) -> None:
    owner = _owner({7})
    with pytest.raises(RuntimeError):
        owner.record_writer_settlement(8 if foreign else 7)
    owner.record_writer_in_doubt(7)
    assert owner.record_writer_settlement(7)
    assert not owner.resources_drained


@pytest.mark.cpu_only
def test_settlement_does_not_clear_contradictory_evidence() -> None:
    owner = _owner({7})
    owner.record_writer_in_doubt(7)
    with pytest.raises(RuntimeError, match="success while in doubt"):
        owner.record_writer_result(7, True, wait_for_local_completion=True)
    assert owner.record_writer_settlement(7)
    assert not owner.resources_drained


@pytest.mark.cpu_only
def test_abort_publication_cannot_exclude_ambiguous_writer() -> None:
    owner = _owner({7, 8})
    owner.record_writer_in_doubt(7)
    with pytest.raises(RuntimeError, match="unpublished writer"):
        owner.abort_publication({8})
    assert owner.record_writer_settlement(7)
    owner.record_writer_result(8, False, wait_for_local_completion=False)
    assert not owner.resources_drained


@pytest.mark.cpu_only
def test_settlement_does_not_clear_local_completion_or_publication() -> None:
    owner = _owner({7})
    owner._local_completion_pending = True
    owner._publication_pending = True
    owner.record_writer_in_doubt(7)
    owner.record_writer_settlement(7)
    assert not owner.resources_drained
    owner.finish_local_completion()
    assert not owner.resources_drained
    owner.finish_publication()
    assert owner.resources_drained


@pytest.mark.cpu_only
@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("poll_early", [False, True])
def test_sender_late_done_reports_physical_failure_only(
    monkeypatch: pytest.MonkeyPatch, auxiliary: bool, poll_early: bool
) -> None:
    sender = _sender()
    task = (
        transfer_mod.AuxSendTask(_params(), slot=0)
        if auxiliary
        else transfer_mod.KVSendTask(_chunk(), _params(), slice_id=0)
    )
    session = SimpleNamespace(
        kv_tasks=[task],
        aux_task=task,
        status=SessionStatus.READY,
        exception=None,
        lock=threading.Lock(),
        _claim_aux_terminal_result=Mock(return_value=True),
        set_exception=lambda reason: task.fail(RuntimeError(reason)),
    )
    sender._sessions[401] = session
    if not auxiliary:
        task.expected_transfers = 1
    status = Mock(wait=Mock(return_value=False), is_completed=Mock(return_value=False))
    status.last_status_str.return_value = "ERROR"
    sender._agent = SimpleNamespace(name="nixl", submit_transfer_requests=lambda request: status)
    request = SimpleNamespace(op="WRITE", remote_name="gen")
    monkeypatch.setattr(transfer_mod.Sender, "_make_agent_request", Mock(return_value=request))
    dealer = Mock()
    sender._get_result_dealer = Mock(return_value=dealer)
    meta = transfer_mod.WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen",
        peer_rank=7,
        peer_endpoint="receiver",
        unique_rid=401,
        src_ptrs=np.array([0x1000]),
        dst_ptrs=np.array([0x2000]),
        sizes=np.array([0x100]),
        slice_id=0,
        receiver_slice_id=3,
        is_last_slice=False,
        meta_type=transfer_mod.WriteMetaType.AUX if auxiliary else transfer_mod.WriteMetaType.KV,
    )
    assert task.begin_physical_operation(7)
    if auxiliary:
        sender._deliver_aux_to_agent(meta)
    else:
        sender._deliver_kv_to_agent(meta)
    original_error = task._exception
    handle = None if auxiliary else TaskHandle(session, task, token_end=16)
    if poll_early and handle is not None:
        assert isinstance(handle.poll(), Failed)
    assert not task.resources_drained
    sender._poll_in_doubt_transfers(0)
    assert dealer.send.call_count == 1
    status.is_completed.return_value = True
    sender._poll_in_doubt_transfers(0)
    sender._poll_in_doubt_transfers(0)

    assert task.resources_drained
    assert task.status is transfer_mod.TaskStatus.ERROR
    assert task._exception is original_error
    if handle is not None:
        assert isinstance(handle.poll(), Failed)
    assert sender._ownership_poisoned is not None
    assert not sender._pending_settlements[0]
    assert dealer.send.call_count == 2
    messages = [call.args[0] for call in dealer.send.call_args_list]
    if auxiliary:
        assert [message[-1].decode() for message in messages] == ["IN_DOUBT", "FAILED_QUIESCED"]
        assert task._transfer_count == 1
        session._claim_aux_terminal_result.assert_called_once_with(7)
    else:
        decoded = [transfer_mod._KV_RESULT_PREFIX.unpack(message[1]) for message in messages]
        assert [transfer_mod._AGENT_RESULT_BY_CODE[item[4]] for item in decoded] == [
            transfer_mod.AgentResult.IN_DOUBT,
            transfer_mod.AgentResult.FAILED_QUIESCED,
        ]
        assert decoded[1][:3] == (7, 401, 3)
        assert task.transferred_count == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize("auxiliary", [False, True])
def test_receiver_late_settlement_retains_logical_failure_and_quarantine(auxiliary: bool) -> None:
    receiver = object.__new__(transfer_mod.Receiver)
    receiver._enforce_physical_ownership = True
    receiver._sessions_lock, receiver._sessions = threading.Lock(), {}
    receiver._pre_cancelled_rids = {}
    receiver._shutdown = False
    receiver._ownership_admission_lock = threading.Lock()
    receiver._ownership_poisoned = None
    receiver._bounce = Mock()
    receiver._bounce.is_bounced.return_value = False
    session = transfer_mod.RxSession(request_id=401, params=_params(), receiver=receiver)
    task = session.prepare_receive(_chunk())
    assert task is not None
    task.expected_transfers = 1
    assert session.try_begin_transfer(task.slice_id, set(), writer_cohort={7})
    result = transfer_mod.AgentResult
    if auxiliary:
        session.process_kv_agent_result(7, 0, True, result.FAILED)
    else:
        session.process_aux_agent_result(7, result.FAILED)

    def report(status: transfer_mod.AgentResult) -> None:
        if auxiliary:
            session.process_aux_agent_result(7, status)
        else:
            session.process_kv_agent_result(7, 0, True, status)

    report(result.IN_DOUBT)
    error = session.exception
    assert not session.resources_drained()
    report(result.FAILED)
    assert not session.resources_drained()
    report(result.FAILED_QUIESCED)
    report(result.FAILED_QUIESCED)
    assert session.resources_drained()
    assert session.status is SessionStatus.ERROR
    assert session.exception is error
    assert receiver._ownership_poisoned is not None
    receiver._bounce.record_failure.assert_called_once_with((401, 0), 7)
    assert session.close()


@pytest.mark.cpu_only
def test_settlement_retries_in_order_without_releasing_twice() -> None:
    sender = _sender()
    task = transfer_mod.KVSendTask(_chunk(), _params(), slice_id=0)
    status = Mock(is_completed=Mock(return_value=True))
    assert task.begin_physical_operation(7)
    task.begin_backend_submission(7, object())
    task.record_backend_submission(7, status)
    task.mark_physical_operation_in_doubt(7)
    task.fail(RuntimeError("original failure"))
    meta = transfer_mod.WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen",
        peer_rank=7,
        peer_endpoint="receiver",
        unique_rid=401,
        src_ptrs=np.array([]),
        dst_ptrs=np.array([]),
        sizes=np.array([]),
        receiver_slice_id=3,
    )
    initial = transfer_mod._make_kv_result_msg(7, 401, 3, False, transfer_mod.AgentResult.IN_DOUBT)
    dealer = Mock()
    dealer.send.side_effect = [RuntimeError("initial send"), None, RuntimeError("late send"), None]
    sender._get_result_dealer = Mock(return_value=dealer)
    sender._retain_in_doubt_transfer(meta, initial, send_slot_id=12)
    status.is_completed.assert_not_called()
    sender._poll_in_doubt_transfers(0)
    assert task.resources_drained
    assert sender._pending_settlements[0]
    sender._bounce.release_send.assert_called_once_with(12)
    with pytest.raises(RuntimeError, match="settlement reports are pending"):
        sender.shutdown()
    sender._poll_in_doubt_transfers(0)
    assert not sender._pending_settlements[0]
    sender._bounce.release_send.assert_called_once_with(12)
    status.is_completed.assert_called_once()
    assert task.transferred_count == 1
    codes = [
        transfer_mod._KV_RESULT_PREFIX.unpack(call.args[0][1])[4]
        for call in dealer.send.call_args_list
    ]
    assert codes == [2, 2, 3, 3]
    sender._shutdown = True


@pytest.mark.cpu_only
@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("terminal_kind", ["failed", "local_cancel", "peer_cancel"])
@pytest.mark.parametrize("poll_early", [False, True])
def test_committed_session_outcome_survives_late_settlement(
    direction: str, terminal_kind: str, poll_early: bool
) -> None:
    """Integration seam with the event-time logical-outcome commit change."""
    status = Mock(is_completed=Mock(return_value=True))
    if direction == "send":
        sender = Mock()
        sender._enforce_physical_ownership = True
        sender._get_req_info.return_value = {}
        session = transfer_mod.TxSession(request_id=401, params=_params(), sender=sender)
        session.send(_chunk())
        task = session.kv_tasks[0]
        task.expected_transfers = 1
        task.status = transfer_mod.TaskStatus.TRANSFERRING
        assert task.begin_physical_operation(7)
        task.begin_backend_submission(7, object())
        task.record_backend_submission(7, status)
    else:
        receiver = Mock()
        receiver._enforce_physical_ownership = True
        receiver._get_ownership_admission_lock.return_value = threading.Lock()
        receiver._bounce.is_bounced.return_value = False
        session = transfer_mod.RxSession(request_id=401, params=_params(), receiver=receiver)
        task = session.prepare_receive(_chunk())
        assert task is not None
        task.expected_transfers = 1
        assert session.try_begin_transfer(task.slice_id, set(), writer_cohort={7})
    cancelled = terminal_kind != "failed"
    if cancelled:
        assert session.cancel_local(by_peer=terminal_kind == "peer_cancel")
    if direction == "send":
        task.mark_physical_operation_in_doubt(7)
        task.fail(RuntimeError("backend outcome ambiguous"))
    else:
        session.process_kv_agent_result(7, 0, True, transfer_mod.AgentResult.IN_DOUBT)
    handle = TaskHandle(session, task, token_end=16)
    expected_type = Cancelled if cancelled else Failed
    if poll_early:
        early_outcome = handle.poll()
        assert isinstance(early_outcome, expected_type)
        if cancelled:
            assert early_outcome.by_peer is (terminal_kind == "peer_cancel")
    assert not task.resources_drained
    if direction == "send":
        assert task.poll_in_doubt_physical_operation(7)
    else:
        session.process_kv_agent_result(7, 0, True, transfer_mod.AgentResult.FAILED_QUIESCED)
        session.process_aux_agent_result(7, transfer_mod.AgentResult.FAILED)
    assert task.resources_drained
    for outcome in (handle.poll(), TaskHandle(session, task, token_end=16).poll()):
        assert isinstance(outcome, expected_type)
        if cancelled:
            assert outcome.by_peer is (terminal_kind == "peer_cancel")
    assert session.close()


@pytest.mark.cpu_only
def test_listener_failure_uses_worker_stream_in_ownership_bridge() -> None:
    sender = _sender()
    sender._send_task_queues = [queue.Queue()]
    sender._get_or_connect_dealer = Mock()
    info = SimpleNamespace(unique_rid=401, instance_rank=7)
    message = [transfer_mod.MessageType.KV_AGENT_RESULT, b"failed"]
    sender._route_result_messages_to_receiver(info, "receiver", [message], defer_to_worker=False)
    sender._get_or_connect_dealer.assert_not_called()
    assert sender._send_task_queues[0].get_nowait() == ("receiver", message)
    sender._shutdown = True


@pytest.mark.cpu_only
@pytest.mark.parametrize("initial_send_fails", [False, True])
def test_worker_polls_retained_status_when_queue_is_idle(
    monkeypatch: pytest.MonkeyPatch, initial_send_fails: bool
) -> None:
    sender = _sender()
    sender._thread_local = threading.local()
    task = transfer_mod.KVSendTask(_chunk(), _params(), slice_id=0)
    status = Mock(is_completed=Mock(side_effect=[False, True]))
    assert task.begin_physical_operation(7)
    task.begin_backend_submission(7, object())
    task.record_backend_submission(7, status)
    task.mark_physical_operation_in_doubt(7)
    task.fail(RuntimeError("original failure"))
    meta = transfer_mod.WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen",
        peer_rank=7,
        peer_endpoint="receiver",
        unique_rid=401,
        src_ptrs=np.array([]),
        dst_ptrs=np.array([]),
        sizes=np.array([]),
    )
    work_queue = Mock(get=Mock(side_effect=[queue.Empty, None]))
    dealer = Mock()
    if initial_send_fails:
        attempts = 0

        def send(_message: list[bytes]) -> None:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                # Even later safely rejected work must not overtake IN_DOUBT.
                work_queue.get.assert_not_called()
                raise RuntimeError("initial send failed")

        dealer.send.side_effect = send
    sender._get_result_dealer = Mock(return_value=dealer)
    sender._retain_in_doubt_transfer(
        meta,
        transfer_mod._make_kv_result_msg(7, 401, 0, False, transfer_mod.AgentResult.IN_DOUBT),
    )
    sender._send_task_queues = [work_queue]
    monkeypatch.setattr(transfer_mod.time, "sleep", Mock())
    monkeypatch.setattr(transfer_mod.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transfer_mod.cudart, "cudaSetDevice", Mock(return_value=0))
    monkeypatch.setattr(transfer_mod, "CUASSERT", Mock())
    sender._process_task_queue(0)
    assert task.resources_drained
    assert not sender._pending_settlements[0]
    assert dealer.send.call_count == (4 if initial_send_fails else 2)
    assert status.is_completed.call_count == 2
    assert task.status is transfer_mod.TaskStatus.ERROR
    sender._shutdown = True
