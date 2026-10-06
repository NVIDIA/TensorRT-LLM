# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deadline arbitration tests; no backend or GPU is needed to advance the clock."""

import threading
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base import Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native import transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.native.bounce import NoBounceTransport
from tensorrt_llm._torch.disaggregation.native.retirement import RetirementWatchdog
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("reason", ["cancelled", "failed", "shutdown", "peer lost"])
def test_first_drain_trigger_cannot_be_extended(reason: str) -> None:
    """Repeated notifications do not reset the physical grace period."""
    clock, notify = Mock(return_value=10.0), Mock()
    watchdog = RetirementWatchdog(notify, clock=clock)
    owner = watchdog.create_owner(4, "receive", 5.0)
    claim = object()
    assert owner.expose(claim)
    owner.request_drain(reason)
    clock.return_value = 14.0
    owner.request_drain("another failure")
    watchdog.progress()
    assert watchdog.fatal is None
    clock.return_value = 15.0
    watchdog.progress()
    event = watchdog.fatal
    assert event is not None
    assert (event.started_at, event.deadline, event.reason) == (10.0, 15.0, reason)
    assert not owner.settle(claim)
    assert not owner.can_retire()
    assert not owner.close()
    watchdog.progress()
    notify.assert_called_once_with(event)


@pytest.mark.parametrize("settle_at,retirable", [(14.999, True), (15.0, False), (15.001, False)])
def test_settlement_and_deadline_share_one_boundary(settle_at: float, retirable: bool) -> None:
    """Evidence arriving exactly at expiry cannot authorize memory reuse."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    owner = watchdog.create_owner(4, "send", 5.0)
    claim = object()
    assert owner.expose(claim)
    owner.request_drain("failed")
    clock.return_value = settle_at
    assert owner.settle(claim) is retirable
    assert owner.close() is retirable
    clock.return_value = 100.0
    watchdog.progress()
    assert (watchdog.fatal is None) is retirable


def test_request_timeout_starts_grace_without_executor_polling() -> None:
    """A stalled backend gets a request timeout followed by exactly one grace window."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    owner = watchdog.create_owner(4, "send", 5.0)
    claim = object()
    assert owner.expose(claim)
    clock.return_value = 19.0
    watchdog.progress()
    assert watchdog.fatal is None
    assert not owner.expose(object())
    clock.return_value = 20.0
    watchdog.progress()
    assert watchdog.fatal is not None
    assert watchdog.fatal.started_at == 15.0
    assert watchdog.fatal.reason == "transfer timeout"


@pytest.mark.parametrize("reason", ["cancellation", "shutdown"])
def test_overdue_request_cannot_restart_grace_before_next_watchdog_poll(reason: str) -> None:
    """A new notification observes the earlier timeout rather than extending its deadline."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    owner = watchdog.create_owner(4, "send", 5.0)
    outcomes = transfer_mod._LogicalOutcomes(owner)
    index = outcomes.add_task()
    assert owner.expose(object())
    clock.return_value = 16.0
    owner.request_drain(reason)
    outcomes.cancel(by_peer=False)
    assert outcomes.get(index).status is SessionStatus.ERROR
    clock.return_value = 20.0
    watchdog.progress()
    assert watchdog.fatal is not None
    assert watchdog.fatal.started_at == 15.0
    assert watchdog.fatal.deadline == 20.0
    assert watchdog.fatal.reason == "transfer timeout"


def test_unpublished_cancellation_is_retirable_without_fatal() -> None:
    """Closing an unexposed owner requires neither backend evidence nor grace."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    owner = watchdog.create_owner(4, "receive", 5.0)
    owner.request_drain("cancelled")
    assert not owner.expose(object())
    assert owner.close()
    assert owner.close()
    clock.return_value = 100.0
    watchdog.progress()
    assert watchdog.fatal is None
    watchdog.stop()


def test_fatal_closes_all_admission_and_keeps_sibling_roots() -> None:
    """One fatal owner closes the worker, not merely its own next operation."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    sender = watchdog.create_owner(4, "send", 5.0)
    receiver = watchdog.create_owner(5, "receive", 5.0)
    source, destination = object(), object()
    assert sender.expose(source)
    assert receiver.expose(destination)
    sender.request_drain("failed")
    clock.return_value = 15.0
    watchdog.progress()
    assert not receiver.settle(destination)
    assert not receiver.expose(object())
    with pytest.raises(RuntimeError, match="admission is closed"):
        watchdog.create_owner(6, "send", 5.0)
    with pytest.raises(RuntimeError, match="retained owners"):
        watchdog.stop()


def test_callback_failure_is_once_only_and_cannot_erase_fatal() -> None:
    """A containment failure never permits ordinary cleanup to resume."""
    clock = Mock(return_value=10.0)
    notify = Mock(side_effect=RuntimeError("supervisor unavailable"))
    watchdog = RetirementWatchdog(notify, clock=clock)
    owner = watchdog.create_owner(4, "send", 5.0)
    claim = object()
    assert owner.expose(claim)
    owner.request_drain("cancelled")
    clock.return_value = 15.0
    watchdog.progress()
    watchdog.progress()
    notify.assert_called_once()
    assert watchdog.fatal is not None
    assert not owner.settle(claim)


def test_watchdog_notifies_outside_arbiter_lock() -> None:
    """Another thread can inspect the fatal state while the callback executes."""
    clock = Mock(return_value=10.0)
    acquired = threading.Event()
    observed_while_callback_active = []
    threads = []

    def inspect() -> None:
        """Acquire the arbiter from a different thread, avoiding RLock reentry."""
        with watchdog.lock:
            acquired.set()

    def notify(_event: object) -> None:
        """Exercise an external callback that waits for independent inspection."""
        thread = threading.Thread(target=inspect)
        threads.append(thread)
        thread.start()
        observed_while_callback_active.append(acquired.wait(1))

    watchdog = RetirementWatchdog(notify, clock=clock)
    owner = watchdog.create_owner(4, "receive", 5.0)
    assert owner.expose(object())
    owner.request_drain("failed")
    clock.return_value = 15.0
    watchdog.progress()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()
    assert observed_while_callback_active == [True]


@pytest.mark.parametrize("blocked_call", ["submit", "wait", "query"])
def test_blocked_backend_cannot_delay_fatal_deadline(blocked_call: str) -> None:
    """Production sender calls cannot hold the metadata watchdog hostage."""
    clock = Mock(return_value=10.0)
    entered, release, notified = threading.Event(), threading.Event(), threading.Event()
    watchdog = RetirementWatchdog(lambda _event: notified.set(), clock=clock)
    retirement = watchdog.create_owner(4, "send", 5.0)
    task = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=4))
    task.bind_logical_outcomes(transfer_mod._LogicalOutcomes(retirement))
    assert task.begin_physical_operation(7)
    request = object()
    status = Mock(wait=Mock(return_value=True), is_completed=Mock(return_value=True))

    def block() -> object:
        """Block only the external backend boundary, never production state logic."""
        entered.set()
        assert release.wait(5)
        return status if blocked_call == "submit" else True

    sender = object.__new__(transfer_mod.Sender)
    sender._shutdown = True
    sender._sessions_lock = threading.Lock()
    sender._enforce_physical_ownership = True
    sender._ownership_poisoned = None
    sender._ownership_poison_lock = watchdog.lock
    sender._agent = Mock(submit_transfer_requests=Mock(return_value=status))
    if blocked_call == "query":
        task.begin_backend_submission(7, request)
        task.record_backend_submission(7, status)
        task.mark_physical_operation_in_doubt(7)
        status.is_completed.side_effect = block
        work = partial(task.poll_in_doubt_physical_operation, 7)
    else:
        if blocked_call == "submit":
            sender._agent.submit_transfer_requests.side_effect = lambda _request: block()
        else:
            status.wait.side_effect = block
        work = partial(sender._submit_transfer, task, 7, request)
    thread = threading.Thread(target=work)
    watchdog.start()
    thread.start()
    try:
        assert entered.wait(1)
        task.fail(RuntimeError("request cancelled"))
        clock.return_value = 15.0
        watchdog.wake.set()
        assert notified.wait(1)
        operation = task._physical_operations[7]
        assert operation.request is request
        assert not task.resources_drained
    finally:
        release.set()
        thread.join(timeout=1)
        # Test-only cleanup after releasing the fake backend. Production stop()
        # refuses fatal quarantine and cannot signal this event.
        watchdog._stopped.set()
        watchdog._thread.join(timeout=1)
    assert not thread.is_alive()
    assert task._physical_operations[7].request is request
    assert task._physical_operations[7].status is status
    assert not task.resources_drained


@pytest.mark.parametrize("terminal", ["timeout", "cancelled", "failed"])
def test_done_during_grace_preserves_logical_outcome(terminal: str) -> None:
    """Late positive evidence retires memory without changing the delivery decision."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    retirement = watchdog.create_owner(4, "send", 5.0)
    outcomes = transfer_mod._LogicalOutcomes(retirement)
    task = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=4))
    task.bind_logical_outcomes(outcomes)
    assert task.begin_physical_operation(7)
    task.begin_backend_submission(7, object())
    task.record_backend_submission(7, Mock())
    if terminal == "timeout":
        clock.return_value = 15.0
        watchdog.progress()
    elif terminal == "cancelled":
        outcomes.cancel(by_peer=False)
    else:
        task.fail(RuntimeError("original failure"))
    before = task.logical_outcome
    assert before is not None
    assert before.status is (
        SessionStatus.CANCELLED if terminal == "cancelled" else SessionStatus.ERROR
    )
    assert task.retire_backend_done_physical_operation(7)
    task.complete()
    assert task.logical_outcome is before
    assert task.resources_drained
    assert retirement.close()
    clock.return_value = 100.0
    watchdog.progress()
    assert watchdog.fatal is None


def test_repeated_done_is_idempotent_after_another_owner_becomes_fatal() -> None:
    """A completed operation never changes back to IN_DOUBT during global quarantine."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    retired = watchdog.create_owner(4, "send", 5.0)
    task = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=4))
    task.bind_logical_outcomes(transfer_mod._LogicalOutcomes(retired))
    assert task.begin_physical_operation(7)
    task.begin_backend_submission(7, object())
    task.record_backend_submission(7, Mock())
    assert task.retire_backend_done_physical_operation(7)
    active = watchdog.create_owner(5, "send", 5.0)
    assert active.expose(object())
    active.request_drain("failed")
    clock.return_value = 15.0
    watchdog.progress()
    assert task.retire_backend_done_physical_operation(7)
    assert task._physical_operations[7].state is transfer_mod._PhysicalOperationState.BACKEND_DONE
    assert not task.resources_drained


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_deadline_is_rejected_before_admission(timeout: float) -> None:
    """Unbounded and non-positive settings cannot activate deadline enforcement."""
    watchdog = RetirementWatchdog(Mock())
    with pytest.raises(ValueError, match="finite positive"):
        watchdog.create_owner(4, "send", timeout)
    assert not watchdog._owners


@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("enabled", [False, True])
def test_worker_explicitly_selects_session_deadline(direction: str, enabled: bool) -> None:
    """Only the worker's configured controller can activate a session deadline."""
    watchdog = RetirementWatchdog(Mock()) if enabled else None
    worker = object.__new__(transfer_mod.TransferWorker)
    worker._retirement_watchdog = watchdog
    worker._sender = Mock(_enforce_physical_ownership=True)
    worker._receiver = Mock(_enforce_physical_ownership=True)
    worker._aux_buffer = None
    worker._config = SimpleNamespace(tx_timeout_s=1.0, tx_overall_timeout_s=5.0, rx_timeout_s=5.0)
    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=4),
        py_request_id=4,
        prompt_len=16,
    )
    create_session = worker.create_tx_session if direction == "send" else worker.create_rx_session
    session = create_session(request)
    if watchdog is None:
        assert session._retirement is None
    else:
        assert session._retirement is not None
        assert session._retirement.controller is watchdog
    assert session.close()
    if watchdog is not None:
        watchdog.stop()


def _receiver(watchdog: RetirementWatchdog) -> transfer_mod.Receiver:
    """Construct the real receive/session logic without network or CUDA endpoints."""
    receiver = object.__new__(transfer_mod.Receiver)
    receiver._retirement_watchdog = watchdog
    receiver._enforce_physical_ownership = True
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = {}
    receiver._shutdown = False
    receiver._ownership_admission_lock = threading.Lock()
    receiver._ownership_poisoned = None
    receiver._bounce = Mock()
    return receiver


def _rx_session(receiver: transfer_mod.Receiver) -> transfer_mod.RxSession:
    """Create a generation-first session, including its auxiliary receive owner."""
    return transfer_mod.RxSession(
        request_id=4,
        params=DisaggregatedParams(
            disagg_request_id=4, schedule_style=DisaggScheduleStyle.GENERATION_FIRST
        ),
        receiver=receiver,
        timeout_s=5.0,
        retirement_watchdog=receiver._retirement_watchdog,
    )


def _tx_session(watchdog: RetirementWatchdog, *, need_aux: bool = True) -> transfer_mod.TxSession:
    """Use real session/admission logic with only queue and backend boundary doubles."""
    sender = object.__new__(transfer_mod.Sender)
    sender._retirement_watchdog = watchdog
    sender._enforce_physical_ownership = True
    sender._ownership_poisoned = None
    sender._ownership_poison_lock = watchdog.lock
    sender._sessions_lock = threading.Lock()
    sender._sessions = {}
    sender._pre_cancelled_rids = {}
    sender._shutdown = True
    sender._shutdown_requested = False
    sender._peer_requests_lock = threading.Lock()
    sender._peer_requests = {}
    sender._peer_requests_timestamps = {}
    sender.dispatch_task = Mock()
    sender._agent = Mock()
    sender._agent.submit_transfer_requests.return_value.wait.return_value = True
    params = DisaggregatedParams(disagg_request_id=4)
    if need_aux:
        params.schedule_style = DisaggScheduleStyle.GENERATION_FIRST
    return transfer_mod.TxSession(
        4, params, sender, overall_timeout_s=5.0, retirement_watchdog=watchdog
    )


def _submit_piece(session: transfer_mod.TxSession, task: transfer_mod.SendTaskBase) -> None:
    """Drive the real sender's submission, backend-DONE and delivery transitions."""
    assert session._sender._begin_task_operation(task, 7)
    assert session._sender._submit_transfer(task, 7, object()) == (True, None)
    task.complete()


@pytest.mark.parametrize("piece", ["kv", "aux"])
@pytest.mark.parametrize("submit_at", [14.999, 15.0, 15.001])
def test_idle_session_keeps_original_deadline_for_later_piece(piece: str, submit_at: float) -> None:
    """An empty physical-claim gap never grants KV or AUX a fresh admission window."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    session = _tx_session(watchdog, need_aux=piece == "aux")
    session.send(Chunk([], [], TokenRange(0, 16), piece == "aux"))
    first = session.kv_tasks[0]
    _submit_piece(session, first)
    delivered = first.logical_outcome
    assert not session._retirement._claims
    # Queue the next piece before expiry; only backend admission decides whether
    # it may cross the exposure boundary when its worker eventually runs.
    clock.return_value = 14.0
    if piece == "aux":
        later = session.send_aux()
    else:
        session.send(Chunk([], [], TokenRange(16, 32), True))
        later = session.kv_tasks[-1]
    assert session._sender._begin_task_operation(later, 7)
    clock.return_value = submit_at
    if submit_at < 15.0:
        assert session._sender._submit_transfer(later, 7, object()) == (True, None)
        later.complete()
        assert session.is_completed()
    else:
        with pytest.raises(transfer_mod._TransferNotSubmittedError):
            session._sender._submit_transfer(later, 7, object())
        assert (
            later._physical_operations[7].state
            is transfer_mod._PhysicalOperationState.NOT_SUBMITTED
        )
        assert later.logical_outcome.status is SessionStatus.ERROR
        assert session._sender._agent.submit_transfer_requests.call_count == 1
        assert not session._retirement._claims
        assert session._retirement._drain_started == 15.0
    assert first.logical_outcome is delivered
    clock.return_value = 100.0
    watchdog.progress()
    assert watchdog.fatal is None
    assert session.close()
    watchdog.stop()


def test_missing_aux_times_out_without_another_session_call() -> None:
    """Metadata progress expires idle, incomplete sessions without inventing physical risk."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    session = _tx_session(watchdog)
    session.send(Chunk([], [], TokenRange(0, 16), True))
    task = session.kv_tasks[0]
    _submit_piece(session, task)
    delivered = task.logical_outcome
    assert session.aux_task is None
    assert not session._retirement._claims
    clock.return_value = 100.0
    watchdog.progress()
    assert session._logical_outcomes._terminal.status is SessionStatus.ERROR
    assert session._retirement._drain_started == 15.0
    assert watchdog.fatal is None
    assert task.logical_outcome is delivered
    assert session.close()
    watchdog.stop()


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_complete_session_stops_deadline_before_delayed_consumer_poll(direction: str) -> None:
    """Final runtime events close success; consumer polling and cleanup may happen much later."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    if direction == "send":
        session = _tx_session(watchdog)
        session.send(Chunk([], [], TokenRange(0, 16), True))
        _submit_piece(session, session.kv_tasks[0])
        auxiliary = session.send_aux()
        _submit_piece(session, auxiliary)
        assert session.send_aux() is auxiliary
    else:
        session, _, _ = _settled_rx_session(watchdog, succeeded=True)
        session.process_aux_agent_result(7, transfer_mod.AgentResult.SUCCESS)
    assert session._retirement._pieces_complete
    assert not session._retirement._claims
    clock.return_value = 100.0
    watchdog.progress()
    assert session.status is SessionStatus.TRANSFERRED
    assert session.is_completed()
    assert session._retirement._drain_started is None
    assert watchdog.fatal is None
    assert not session._retirement.expose(object())
    assert session.close()
    watchdog.stop()


@pytest.mark.parametrize("final_piece", ["send_kv", "send_aux", "receive_kv"])
def test_final_completion_precedes_delayed_perf_logging(final_piece: str, monkeypatch) -> None:
    """Best-effort diagnostics cannot time out or fail a fully settled session."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    if final_piece == "receive_kv":
        receiver = _receiver(watchdog)
        receiver._bounce = NoBounceTransport()
        receiver._registrar = SimpleNamespace(
            self_rank_info=SimpleNamespace(instance_name="receiver", instance_rank=0)
        )
        session = _rx_session(receiver)
        task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), True))
        task.expected_transfers = 1
        published = set()
        assert session.try_begin_transfer(
            0, {"sender"}, {7}, publish=lambda: published.add(7), published_writers=published
        )
        session.process_aux_agent_result(7, transfer_mod.AgentResult.SUCCESS)
        finish = partial(
            session.process_kv_agent_result, 7, 0, True, transfer_mod.AgentResult.SUCCESS
        )
    else:
        need_aux = final_piece == "send_aux"
        session = _tx_session(watchdog, need_aux=need_aux)
        session.send(Chunk([], [], TokenRange(0, 16), True))
        task = session.kv_tasks[0]
        if need_aux:
            _submit_piece(session, task)
            task = session.send_aux()
        sender = session._sender
        sender._instance_rank = sender._device_id = 0
        sender._bounce = NoBounceTransport()
        sender._registrar = SimpleNamespace(
            self_rank_info=SimpleNamespace(instance_name="sender", instance_rank=0)
        )
        sender._get_result_dealer = Mock()
        assert sender._begin_task_operation(task, 7)
        meta = transfer_mod.WriteMeta(
            task,
            1,
            "receiver",
            7,
            "receiver",
            4,
            np.array([16], dtype=np.int64),
            np.array([32], dtype=np.int64),
            np.array([16], dtype=np.int64),
            dst_device_id=0,
            slice_id=0,
            is_last_slice=True,
            meta_type=transfer_mod.WriteMetaType.AUX if need_aux else transfer_mod.WriteMetaType.KV,
        )
        deliver = sender._deliver_aux_to_agent if need_aux else sender._deliver_kv_to_agent
        finish = partial(deliver, meta)

    def delayed_diagnostic(*args):
        if final_piece.startswith("send"):
            session._sender._get_result_dealer.return_value.send.assert_called_once()
        clock.return_value = 100.0
        watchdog.progress()
        raise RuntimeError("diagnostic failure after transfer completion")

    monkeypatch.setattr(task, "print_perf_info", delayed_diagnostic)
    finish()
    assert clock.return_value == 100.0
    assert task.logical_outcome.status is SessionStatus.TRANSFERRED
    assert session.is_completed()
    assert session._retirement._drain_started is None
    assert watchdog.fatal is None
    assert session.close()
    watchdog.stop()


def test_logical_completion_cannot_stop_unresolved_physical_deadline() -> None:
    """Delivered results remain stable while their outstanding access still requires containment."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    session = _tx_session(watchdog, need_aux=False)
    session.send(Chunk([], [], TokenRange(0, 16), True))
    task = session.kv_tasks[0]
    assert session._sender._begin_task_operation(task, 7)
    request, status = object(), Mock()
    task.begin_backend_submission(7, request)
    task.record_backend_submission(7, status)
    task.complete()
    delivered = task.logical_outcome
    assert session._retirement._pieces_complete
    assert session._retirement._claims
    assert not session.is_completed()
    assert session.wait_complete(blocking=False) is None
    assert session.wait_complete(blocking=True) is None
    clock.return_value = 15.0
    watchdog.progress()
    assert session.status is SessionStatus.ERROR
    assert task.logical_outcome is delivered
    clock.return_value = 20.0
    watchdog.progress()
    assert watchdog.fatal is not None
    assert not task.retire_backend_done_physical_operation(7)
    assert task._physical_operations[7].request is request
    assert task._physical_operations[7].status is status
    assert not session.close()


@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("blocking", [False, True])
def test_finished_nonfinal_piece_does_not_complete_session(direction: str, blocking: bool) -> None:
    """Consumer polling cannot mistake an idle gap for the producer's end-of-pieces signal."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    chunk = Chunk([], [], TokenRange(0, 16), False)
    if direction == "send":
        session = _tx_session(watchdog, need_aux=False)
        session.send(chunk)
        _submit_piece(session, session.kv_tasks[0])
    else:
        receiver = _receiver(watchdog)
        receiver._bounce = NoBounceTransport()
        receiver._registrar = SimpleNamespace(
            self_rank_info=SimpleNamespace(instance_name="receiver", instance_rank=0)
        )
        session = transfer_mod.RxSession(
            4,
            DisaggregatedParams(disagg_request_id=4),
            receiver,
            timeout_s=5.0,
            retirement_watchdog=watchdog,
        )
        task = session.prepare_receive(chunk)
        task.expected_transfers = 1
        assert session.try_begin_transfer(0, {"sender"}, {7})
        session.process_kv_agent_result(7, 0, True, transfer_mod.AgentResult.SUCCESS)
    assert not session._retirement._claims
    assert not session._retirement.is_complete
    assert not session.is_completed()
    assert session.wait_complete(blocking=blocking) is None
    clock.return_value = 15.0
    watchdog.progress()
    assert session.status is SessionStatus.ERROR
    assert watchdog.fatal is None
    assert session.close()
    watchdog.stop()


def test_completion_at_transfer_deadline_cannot_replace_timeout() -> None:
    """Physically safe completion at the request boundary still preserves logical timeout."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    session = _tx_session(watchdog)
    session.send(Chunk([], [], TokenRange(0, 16), True))
    _submit_piece(session, session.kv_tasks[0])
    task = session.send_aux()
    assert session._sender._begin_task_operation(task, 7)
    assert session._sender._submit_transfer(task, 7, object()) == (True, None)
    clock.return_value = 15.0
    task.complete()
    assert task.logical_outcome.status is SessionStatus.ERROR
    assert session.status is SessionStatus.ERROR
    assert not session._retirement._pieces_complete
    assert session.close()
    watchdog.stop()


def test_expired_fanout_rejects_queued_peer_but_retains_submitted_sibling() -> None:
    """A rejected peer never clears another peer's existing backend roots."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    session = _tx_session(watchdog, need_aux=False)
    session.send(Chunk([], [], TokenRange(0, 16), True))
    task = session.kv_tasks[0]
    assert session._sender._begin_task_operation(task, 7)
    task.begin_backend_submission(7, object())
    task.record_backend_submission(7, Mock())
    assert session._sender._begin_task_operation(task, 8)
    clock.return_value = 15.0
    with pytest.raises(transfer_mod._TransferNotSubmittedError):
        session._sender._submit_transfer(task, 8, object())
    session._sender._agent.submit_transfer_requests.assert_not_called()
    assert task._physical_operations[8].state is transfer_mod._PhysicalOperationState.NOT_SUBMITTED
    assert list(session._retirement._claims.values()) == [task._physical_operations[7]]
    clock.return_value = 20.0
    watchdog.progress()
    assert watchdog.fatal is not None
    assert not session.close()


@pytest.mark.parametrize("need_aux", [False, True])
def test_expired_second_receive_preserves_existing_claims(need_aux: bool) -> None:
    """Reject an expired receive even when idle; retain any previously published AUX."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    receiver = _receiver(watchdog)
    receiver._bounce = NoBounceTransport()
    receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="receiver", instance_rank=0)
    )
    params = DisaggregatedParams(disagg_request_id=4)
    if need_aux:
        params.schedule_style = DisaggScheduleStyle.GENERATION_FIRST
    session = transfer_mod.RxSession(
        4, params, receiver, timeout_s=5.0, retirement_watchdog=watchdog
    )
    task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), False))
    task.expected_transfers = 1
    assert session.try_begin_transfer(0, {"sender"}, {7})
    session.process_kv_agent_result(7, 0, True, transfer_mod.AgentResult.SUCCESS)
    second = session.prepare_receive(Chunk([], [], TokenRange(16, 32), True))
    second.expected_transfers = 1
    publish = Mock()
    clock.return_value = 15.0
    with pytest.raises(RuntimeError, match="retirement admission is closed"):
        session.try_begin_transfer(1, {"sender"}, {7}, publish=publish, published_writers=set())
    publish.assert_not_called()
    assert second.cancel_unpublished()
    assert second.resources_drained is not need_aux
    expected_claims = [session._aux_physical_owner] if need_aux else []
    assert list(session._retirement._claims.values()) == expected_claims
    if need_aux:
        assert not session._aux_physical_owner.resources_drained
    clock.return_value = 20.0
    watchdog.progress()
    assert (watchdog.fatal is not None) is need_aux
    assert session.close() is not need_aux
    if not need_aux:
        watchdog.stop()


def _settled_rx_session(
    watchdog: RetirementWatchdog, *, succeeded: bool = False
) -> tuple[transfer_mod.RxSession, transfer_mod.KVRecvTask, Mock]:
    """Settle real KV and AUX result handlers before the executor closes the session."""
    receiver = _receiver(watchdog)
    receiver._bounce = NoBounceTransport()
    receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="receiver", instance_rank=0)
    )
    aux = Mock()
    aux.alloc_slot.return_value = SimpleNamespace(id=1)
    session = transfer_mod.RxSession(
        4,
        DisaggregatedParams(
            disagg_request_id=4, schedule_style=DisaggScheduleStyle.GENERATION_FIRST
        ),
        receiver,
        aux_buffer=aux,
        timeout_s=5.0,
        retirement_watchdog=watchdog,
    )
    task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), True))
    assert task is not None
    task.expected_transfers = 1
    published = set()
    assert session.try_begin_transfer(
        0, {"sender"}, {7}, publish=lambda: published.add(7), published_writers=published
    )
    result = transfer_mod.AgentResult.SUCCESS if succeeded else transfer_mod.AgentResult.FAILED
    session.process_kv_agent_result(7, 0, True, result)
    session.process_aux_agent_result(7, result)
    assert session.resources_drained()
    assert not session._retirement._claims
    return session, task, aux


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("payload", ["kv", "aux"])
@pytest.mark.parametrize("evidence", ["contradictory", "in_doubt"])
def test_late_receive_evidence_cannot_escape_retirement_deadline(
    closed: bool, payload: str, evidence: str
) -> None:
    """Unclosed owners regain deadline tracking; callbacks cannot revive closed sessions."""
    clock, notify = Mock(return_value=10.0), Mock()
    watchdog = RetirementWatchdog(notify, clock=clock)
    session, task, aux = _settled_rx_session(watchdog)
    original = task.logical_outcome
    assert original.status is SessionStatus.ERROR
    if closed:
        assert session.close()
    clock.return_value = 11.0
    report = (
        partial(session.process_kv_agent_result, 7, 0, True)
        if payload == "kv"
        else partial(session.process_aux_agent_result, 7)
    )
    result = (
        transfer_mod.AgentResult.SUCCESS
        if evidence == "contradictory"
        else transfer_mod.AgentResult.IN_DOUBT
    )
    if not closed and evidence == "contradictory":
        with pytest.raises(RuntimeError, match="contradictory terminal evidence"):
            report(result)
    else:
        report(result)
    assert task.logical_outcome is original
    if closed:
        assert session.close()
        assert session.resources_drained()
        assert session._receiver._ownership_poisoned is None
        assert not watchdog._owners
        assert not session._receiver._sessions
        aux.free_slot.assert_called_once_with(1)
        clock.return_value = 100.0
        watchdog.progress()
        notify.assert_not_called()
        watchdog.stop()
        return
    assert session._retirement._claims
    assert not session.resources_drained()
    assert not session.close()
    clock.return_value = 14.0
    session.cancel_local()
    watchdog.progress()
    assert watchdog.fatal is None
    clock.return_value = 15.0
    watchdog.progress()
    assert watchdog.fatal is not None
    assert (watchdog.fatal.started_at, watchdog.fatal.deadline) == (10.0, 15.0)
    notify.assert_called_once_with(watchdog.fatal)
    assert task.logical_outcome is original
    assert not session.close()
    assert session._receiver._sessions[4] is session
    aux.free_slot.assert_not_called()


def test_restored_receive_claim_can_settle_safely_before_original_deadline() -> None:
    """Late positive physical evidence clears the restored claim without changing failure."""
    clock, notify = Mock(return_value=10.0), Mock()
    watchdog = RetirementWatchdog(notify, clock=clock)
    session, task, aux = _settled_rx_session(watchdog)
    original = task.logical_outcome
    clock.return_value = 11.0
    session.process_kv_agent_result(7, 0, True, transfer_mod.AgentResult.IN_DOUBT)
    assert not session.close()
    clock.return_value = 14.0
    session.process_kv_agent_result(7, 0, True, transfer_mod.AgentResult.FAILED_QUIESCED)
    assert task.logical_outcome is original
    assert session.close()
    assert session.close()
    aux.free_slot.assert_called_once_with(1)
    clock.return_value = 100.0
    watchdog.progress()
    notify.assert_not_called()
    watchdog.stop()


def test_new_ambiguity_after_success_does_not_inherit_expired_request_clock() -> None:
    """Previously settled success anchors its first drain at the new evidence event."""
    clock, notify = Mock(return_value=10.0), Mock()
    watchdog = RetirementWatchdog(notify, clock=clock)
    session, task, aux = _settled_rx_session(watchdog, succeeded=True)
    original = task.logical_outcome
    assert original.status is SessionStatus.TRANSFERRED
    clock.return_value = 30.0
    session.process_kv_agent_result(7, 0, True, transfer_mod.AgentResult.IN_DOUBT)
    assert task.logical_outcome is original
    assert not session.close()
    watchdog.progress()
    assert watchdog.fatal is None
    clock.return_value = 35.0
    watchdog.progress()
    assert watchdog.fatal is not None
    assert (watchdog.fatal.started_at, watchdog.fatal.deadline) == (30.0, 35.0)
    notify.assert_called_once_with(watchdog.fatal)
    aux.free_slot.assert_not_called()


@pytest.mark.parametrize("reservation_boundary", ["kv", "aux"])
def test_receive_reserves_kv_and_aux_atomically_against_cancellation(
    monkeypatch: pytest.MonkeyPatch, reservation_boundary: str
) -> None:
    """Cancellation cannot strand half of a never-published receive reservation."""
    watchdog = RetirementWatchdog(Mock())
    receiver = _receiver(watchdog)
    session = _rx_session(receiver)
    task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), True))
    assert task is not None
    task.expected_transfers = 1
    owner = (
        task._get_physical_owner() if reservation_boundary == "kv" else session._aux_physical_owner
    )
    original_seal = owner.seal_writer_cohort
    attempted = threading.Event()
    interleaved, errors = [], []

    def cancel() -> None:
        """Probe the arbiter without sleeping, then exercise real cancellation."""
        acquired = watchdog.lock.acquire(blocking=False)
        interleaved.append(acquired)
        if acquired:
            try:
                session._retirement.request_drain("cancellation")
            finally:
                watchdog.lock.release()
        attempted.set()
        try:
            session.cancel_local()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=cancel)

    def seal(expected: int, cohort: set[int], *, published_writers: set[int] | None = None) -> None:
        """Pause at a real metadata boundary while cancellation competes."""
        original_seal(expected, cohort, published_writers=published_writers)
        thread.start()
        assert attempted.wait(1)
        assert interleaved == [False]

    monkeypatch.setattr(owner, "seal_writer_cohort", seal)
    published = set()
    try:
        assert session.try_begin_transfer(
            0, {"sender"}, {7}, publish=lambda: published.add(7), published_writers=published
        )
    finally:
        if thread.ident is not None:
            thread.join(timeout=1)
    assert not thread.is_alive()
    assert not errors
    assert published == {7}
    assert session.status is SessionStatus.CANCELLED
    assert not session.resources_drained()
    task.record_writer_result(7, False, wait_for_local_completion=False)
    assert not session.resources_drained()
    session._aux_physical_owner.record_writer_result(7, False, wait_for_local_completion=False)
    assert session.close()
    watchdog.stop()


@pytest.mark.parametrize("rejection", ["invalid cohort", "draining"])
def test_receive_reservation_rejection_retires_all_unpublished_owners(rejection: str) -> None:
    """Validation or closed admission cannot leave an exposed partial reservation."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    receiver = _receiver(watchdog)
    session = _rx_session(receiver)
    task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), True))
    assert task is not None
    task.expected_transfers = 2 if rejection == "invalid cohort" else 1
    if rejection == "draining":
        session._retirement.request_drain("cancellation")
    publish = Mock()
    with pytest.raises((ValueError, RuntimeError)) as caught:
        session.try_begin_transfer(0, {"sender"}, {7}, publish=publish, published_writers=set())
    publish.assert_not_called()
    assert session.cancel_unpublished_task(task)
    session.fail_admission(caught.value)
    assert session.resources_drained()
    assert session.close()
    clock.return_value = 100.0
    watchdog.progress()
    assert watchdog.fatal is None
    watchdog.stop()


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("already_failed", [False, True])
def test_shutdown_retires_unpublished_generation_first_session(
    prepared: bool, already_failed: bool
) -> None:
    """An unsealed auxiliary owner cannot hang shutdown or cause double release."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    receiver = _receiver(watchdog)
    aux = Mock()
    aux.alloc_slot.return_value = SimpleNamespace(id=1)
    session = transfer_mod.RxSession(
        4,
        DisaggregatedParams(
            disagg_request_id=4, schedule_style=DisaggScheduleStyle.GENERATION_FIRST
        ),
        receiver,
        aux_buffer=aux,
        timeout_s=5.0,
        retirement_watchdog=watchdog,
    )
    task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), True)) if prepared else None
    if already_failed:
        session._logical_outcomes.fail(RuntimeError("earlier failure"))
    worker = object.__new__(transfer_mod.TransferWorker)
    worker._retirement_watchdog = watchdog
    assert not session.close()
    worker.request_shutdown()
    assert session.close()
    assert session.close()
    expected = SessionStatus.ERROR if already_failed else SessionStatus.CANCELLED
    assert session.status is expected
    if task is not None:
        assert task.logical_outcome.status is expected
        assert task.is_done
    aux.free_slot.assert_called_once_with(1)
    assert not receiver._sessions
    assert not watchdog._owners
    clock.return_value = 100.0
    watchdog.progress()
    assert watchdog.fatal is None
    watchdog.stop()


@pytest.mark.parametrize("trigger", ["cancel", "failure", "shutdown", "timeout"])
def test_receiver_deadline_retains_kv_aux_and_registration(trigger: str) -> None:
    """Late receiver evidence cannot bypass fatal session or worker teardown guards."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    receiver = _receiver(watchdog)
    session = _rx_session(receiver)
    task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), True))
    assert task is not None
    task.expected_transfers = 1
    published = set()
    assert session.try_begin_transfer(
        0, {"sender"}, {7}, publish=lambda: published.add(7), published_writers=published
    )
    worker = object.__new__(transfer_mod.TransferWorker)
    worker._retirement_watchdog = watchdog
    worker._receiver = receiver
    worker._agent = Mock()
    worker._registered_mem = [object()]
    worker._bounce = Mock()
    if trigger == "cancel":
        session.cancel_local()
    elif trigger == "failure":
        session.fail_admission(RuntimeError("lost peer"))
    elif trigger == "shutdown":
        worker.request_shutdown()
    else:
        clock.return_value = 15.0
        watchdog.progress()
        assert session.status is SessionStatus.ERROR
        assert session.has_failed()
        assert session.wait_complete(blocking=False) is None
    clock.return_value = 20.0 if trigger == "timeout" else 15.0
    watchdog.progress()
    assert watchdog.fatal is not None
    task.record_writer_result(7, False, wait_for_local_completion=False)
    session._aux_physical_owner.record_writer_result(7, False, wait_for_local_completion=False)
    assert not session.resources_drained()
    assert not session.close()
    assert receiver._sessions[4] is session
    with pytest.raises(RuntimeError, match="teardown refused"):
        worker.shutdown()
    worker._agent.deregister_memory.assert_not_called()
    worker._agent.shutdown.assert_not_called()
    worker._bounce.close.assert_not_called()
    assert len(worker._registered_mem) == 1
    with pytest.raises(RuntimeError, match="admission is closed"):
        _rx_session(receiver)


def test_blocked_publication_does_not_delay_receiver_deadline() -> None:
    """The real publication path never holds the watchdog lock during a network send."""
    clock = Mock(return_value=10.0)
    entered, release, notified = threading.Event(), threading.Event(), threading.Event()
    watchdog = RetirementWatchdog(lambda _event: notified.set(), clock=clock)
    receiver = _receiver(watchdog)
    session = _rx_session(receiver)
    task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), True))
    assert task is not None
    task.expected_transfers = 1
    published, errors = set(), []

    def publish() -> None:
        """Represent a network send that has not yet returned."""
        entered.set()
        assert release.wait(5)
        published.add(7)

    def dispatch() -> None:
        """Capture any production-path exception for the main test thread."""
        try:
            session.try_begin_transfer(
                0, {"sender"}, {7}, publish=publish, published_writers=published
            )
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=dispatch)
    watchdog.start()
    thread.start()
    try:
        assert entered.wait(1)
        session.cancel_local()
        clock.return_value = 15.0
        watchdog.wake.set()
        assert notified.wait(1)
        assert not session.resources_drained()
        assert not session.close()
    finally:
        release.set()
        thread.join(timeout=1)
        watchdog._stopped.set()
        watchdog._thread.join(timeout=1)
    assert not thread.is_alive()
    assert not errors
    assert not session.resources_drained()


@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("failure", ["aux allocation", "session registration"])
def test_unpublished_constructor_failure_removes_watchdog_owner(
    direction: str, failure: str
) -> None:
    """Failed construction must not leave an unexposed owner blocking worker shutdown."""
    watchdog = RetirementWatchdog(Mock())
    aux = Mock()
    aux.alloc_slot.return_value = SimpleNamespace(id=1)
    if failure == "aux allocation":
        aux.alloc_slot.side_effect = RuntimeError("aux allocation")
    params = DisaggregatedParams(disagg_request_id=4)
    existing = object()
    if direction == "send":
        sender = SimpleNamespace(
            _retirement_watchdog=watchdog,
            _enforce_physical_ownership=True,
            _sessions_lock=threading.Lock(),
            _sessions={4: existing},
            setup_session=Mock(side_effect=RuntimeError("session registration")),
            clear_session=Mock(),
        )
        construct = partial(
            transfer_mod.TxSession,
            4,
            params,
            sender,
            aux_buffer=aux,
            overall_timeout_s=5.0,
            retirement_watchdog=watchdog,
        )
    else:
        receiver = _receiver(watchdog)
        receiver._shutdown = True
        receiver._sessions[4] = existing
        construct = partial(
            transfer_mod.RxSession,
            4,
            params,
            receiver,
            aux_buffer=aux,
            timeout_s=5.0,
            retirement_watchdog=watchdog,
        )
    with pytest.raises(RuntimeError):
        construct()
    assert (sender if direction == "send" else receiver)._sessions[4] is existing
    assert not watchdog._owners
    if failure == "session registration":
        aux.free_slot.assert_called_once_with(1)
    else:
        aux.free_slot.assert_not_called()
    watchdog.stop()


def test_receiver_timeout_during_blocking_wait_cannot_report_delivery() -> None:
    """A task waking a waiter during grace does not undo the already committed timeout."""
    clock = Mock(return_value=10.0)
    watchdog = RetirementWatchdog(Mock(), clock=clock)
    receiver = _receiver(watchdog)
    # No auxiliary payload is needed to isolate the blocking KV wait branch.
    session = transfer_mod.RxSession(
        4,
        DisaggregatedParams(disagg_request_id=4),
        receiver,
        timeout_s=5.0,
        retirement_watchdog=watchdog,
    )
    task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), True))
    assert task is not None
    task.expected_transfers = 1
    assert session.try_begin_transfer(0, {"sender"}, {7})

    def complete_during_grace(*, timeout: float) -> bool:
        """Inject late backend evidence at the external wait boundary."""
        assert timeout == 5.0
        clock.return_value = 15.0
        task.record_writer_result(7, True, wait_for_local_completion=False)
        task.complete()
        return True

    task.wait = Mock(side_effect=complete_during_grace)
    assert session.wait_complete(blocking=True) is transfer_mod.WaitResult.FAILED
    assert session.status is SessionStatus.ERROR
    assert session.has_failed()
    assert session.close()
    assert not watchdog._owners


@pytest.mark.parametrize("ownership,timeout", [(False, 5.0), (True, None), (True, float("inf"))])
def test_deadline_activation_rejects_unqualified_config(
    ownership: bool, timeout: float | None
) -> None:
    """The internal callback requires ownership and bounded clocks before runtime setup."""
    config = transfer_mod.TransferWorkerConfig(
        kv_cache_manager=Mock(),
        device_id=0,
        instance_name="test",
        enforce_physical_ownership=ownership,
        quiescence_fatal_callback=Mock(),
        tx_overall_timeout_s=timeout,
        rx_timeout_s=timeout,
    )
    with pytest.raises(ValueError, match="deadline retirement requires"):
        transfer_mod.TransferWorker(config)
