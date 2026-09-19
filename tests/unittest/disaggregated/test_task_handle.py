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
"""What ``reports_pending`` answers for a piece several peers write into.

A report is still owed until every expected writer has said something terminal about the piece.
The outcome settles earlier than that -- one writer's failure ends the piece while its siblings
write on -- so the two questions are asked of different state here. Whether the piece can be
handed back is a third question, and nothing in the contract answers it.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base import (
    CacheExtent,
    CacheKind,
    Cancelled,
    Chunk,
    Delivered,
    Failed,
    TokenRange,
)
from tensorrt_llm._torch.disaggregation.native.fetch import PeerFetch
from tensorrt_llm._torch.disaggregation.native.handle import TaskHandle
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVRecvTask,
    KVSendTask,
    MessageType,
    Receiver,
    RxSession,
    Sender,
    SessionStatus,
    TaskStatus,
    TxSession,
)

pytestmark = pytest.mark.cpu_only


def _owes_a_report(outcome) -> bool:
    """No conclusion yet, or one that is still owed a report.

    Local to the tests on purpose. It used to be exported from the contract, where the name read
    like permission to hand memory back -- which is the one thing it never meant.
    """
    return outcome is None or outcome.reports_pending


TOKENS = 8


def _sole_piece() -> Chunk:
    return Chunk(
        block_ids_per_layer_groups=[[0]],
        kind_per_layer_group=[CacheKind.PAGED],
        token_range=TokenRange(start=0, end=TOKENS),
        is_last=True,
    )


def _stub_receiver():
    """Only what a receive session calls on its receiver; dispatch is what the peers would do."""
    receiver = MagicMock()
    receiver._enforce_physical_ownership = False
    receiver._bounce.is_bounced.return_value = False
    return receiver


def _receiving_from(
    writers: int, rid: int = 7, *, owns_transfers: bool = False
) -> tuple[RxSession, TaskHandle]:
    """One piece published to ``writers`` peers, and the handle the caller polls it through."""
    receiver = _stub_receiver()
    receiver._enforce_physical_ownership = owns_transfers
    session = RxSession(
        request_id=rid,
        params=DisaggregatedParams(disagg_request_id=rid),
        receiver=receiver,
        prompt_len=TOKENS,
    )
    session.receive(_sole_piece())
    task = session._kv_tasks[0]
    task.expected_transfers = writers
    cohort = set(range(writers)) if owns_transfers else None
    session.mark_transferring(task.slice_id, cohort)
    return session, TaskHandle(session, task, TOKENS)


def _report(
    session: RxSession,
    peer_rank: int,
    status: AgentResult,
    is_last_slice: bool = True,
    transfer_size: int = 0,
):
    session.process_kv_agent_result(
        peer_rank=peer_rank,
        receiver_slice_id=0,
        is_last_slice=is_last_slice,
        status=status,
        transfer_size=transfer_size,
    )


# ---------------------------------------------------------------------------
# Fan-in: the outcome settles before the writers do
# ---------------------------------------------------------------------------


def test_one_writer_failing_leaves_the_other_still_owed():
    """The piece has failed, but the peer that has not reported may still be writing."""
    session, handle = _receiving_from(2)

    _report(session, 0, AgentResult.FAILED)

    outcome = handle.poll()
    assert isinstance(outcome, Failed)
    assert outcome.reports_pending is True
    assert _owes_a_report(outcome) is True


def test_the_second_writer_settles_what_the_failure_did_not():
    session, handle = _receiving_from(2)

    _report(session, 0, AgentResult.FAILED)
    _report(session, 1, AgentResult.SUCCESS)

    outcome = handle.poll()
    assert isinstance(outcome, Failed)
    assert outcome.reports_pending is False
    assert _owes_a_report(outcome) is False


def test_every_writer_succeeding_delivers_with_nothing_owed():
    session, handle = _receiving_from(2)

    _report(session, 0, AgentResult.SUCCESS)
    assert handle.poll() is None

    _report(session, 1, AgentResult.SUCCESS)

    outcome = handle.poll()
    assert isinstance(outcome, Delivered)
    assert outcome.token_end == TOKENS
    assert _owes_a_report(outcome) is False


# ---------------------------------------------------------------------------
# Which writer reported, not how many reports arrived
# ---------------------------------------------------------------------------


def test_a_repeat_does_not_stand_in_for_the_writer_that_never_reported():
    session, handle = _receiving_from(2)

    _report(session, 0, AgentResult.SUCCESS)
    _report(session, 0, AgentResult.SUCCESS)

    assert session._kv_tasks[0].status is TaskStatus.TRANSFERRING
    assert handle.poll() is None
    assert _owes_a_report(handle.poll()) is True


def test_a_repeat_is_refused_rather_than_answered_again():
    """The first answer is an edge, not a level.

    A rank that already spoke gets no second hearing, or the caller settles the piece once per
    frame instead of once.
    """
    session, _ = _receiving_from(2)
    task = session._kv_tasks[0]

    assert task.note_writer_report(0, True) == (True, False)
    assert task.note_writer_report(0, True) == (False, False)
    assert task.note_writer_report(1, True) == (True, True)
    assert task.note_writer_report(1, True) == (False, False)


def test_a_repeated_terminal_frame_is_counted_once():
    """A terminal word is acted on once.

    Byte accounting, scatter and completion run on the report that settled the piece, not on
    every copy of it.
    """
    session, _ = _receiving_from(1)

    _report(session, 0, AgentResult.SUCCESS, transfer_size=512)
    _report(session, 0, AgentResult.SUCCESS, transfer_size=512)

    assert session.kv_cache_size_bytes == 512


def test_a_writer_contradicting_itself_keeps_its_first_word():
    """Nothing un-fails a piece: the later success stands in for no missing report either."""
    session, handle = _receiving_from(2)

    _report(session, 0, AgentResult.FAILED)
    _report(session, 0, AgentResult.SUCCESS)

    outcome = handle.poll()
    assert isinstance(outcome, Failed)
    assert outcome.reports_pending is True

    _report(session, 1, AgentResult.SUCCESS)
    assert _owes_a_report(handle.poll()) is False


def test_a_success_short_of_the_last_slice_is_not_a_writers_last_word():
    session, handle = _receiving_from(1)

    _report(session, 0, AgentResult.SUCCESS, is_last_slice=False)

    assert handle.poll() is None
    assert _owes_a_report(handle.poll()) is True


# ---------------------------------------------------------------------------
# Nothing to hear from
# ---------------------------------------------------------------------------


def test_a_piece_no_writer_was_told_about_is_owed_nothing():
    """Publication sets the expected writer count, so zero means no peer holds the destination."""
    session, handle = _receiving_from(0)
    session.fail_admission(RuntimeError("no peer took it"))

    outcome = handle.poll()
    assert isinstance(outcome, Failed)
    assert _owes_a_report(outcome) is False


def test_closing_leaves_the_tasks_where_they_were():
    """Closing deregisters the session.

    Ending its tasks is what a release gate would need, and nothing here reads that answer.
    """
    session, handle = _receiving_from(2)
    task = session._kv_tasks[0]

    assert session.close() is True

    assert task.status is TaskStatus.TRANSFERRING
    assert handle.poll() is None


def test_report_progress_remains_separate_from_a_committed_failure():
    session, handle = _receiving_from(1)
    task = session._kv_tasks[0]
    task.fail(RuntimeError("peer died"))

    assert handle.poll().reports_pending is True

    _report(session, 0, AgentResult.SUCCESS)
    assert handle.poll().reports_pending is False
    assert isinstance(handle.poll(), Failed)


def test_a_send_task_owes_until_every_peer_write_is_done():
    """One peer failing ends the task while the writes to its siblings run on.

    The task's own status says ERROR at that moment, so reading it would answer that nothing is
    outstanding while this side is still writing into pages the peers hold.
    """
    task = KVSendTask(_sole_piece(), DisaggregatedParams(disagg_request_id=20), slice_id=0)
    task.expected_transfers = 2
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        status=SessionStatus.TRANSFERRING,
        exception=None,
        cancelled_by_peer=False,
        _closed=False,
    )
    handle = TaskHandle(session, task, TOKENS)

    task.transferred_count = 1
    task.fail(RuntimeError("first peer died"))
    outcome = handle.poll()
    assert isinstance(outcome, Failed)
    assert outcome.reports_pending is True
    assert _owes_a_report(outcome) is True

    # The surviving peer's write finishes; only now is nothing of ours still out.
    task.transferred_count = 2
    assert _owes_a_report(handle.poll()) is False


def test_a_send_task_that_published_to_nobody_owes_nothing():
    """Zero expected writes means the submission never reached a peer."""
    task = KVSendTask(_sole_piece(), DisaggregatedParams(disagg_request_id=21), slice_id=0)
    session = SimpleNamespace(
        status=SessionStatus.ERROR,
        exception=RuntimeError("submission failed"),
        cancelled_by_peer=False,
        _closed=False,
    )
    task.fail(RuntimeError("submission failed"))

    assert _owes_a_report(TaskHandle(session, task, TOKENS).poll()) is False


# ---------------------------------------------------------------------------
# Ownership mode keeps its own cohort
# ---------------------------------------------------------------------------


def test_an_owned_task_answers_from_the_cohort_it_sealed():
    """The ownership owner already tracks writers by rank; the handle asks it rather than count."""
    task = KVRecvTask(7, _sole_piece(), 0, DisaggregatedParams(disagg_request_id=7), aux_slot=None)
    task.expected_transfers = 2
    task.begin_publication()
    task.seal_writer_cohort({0, 1})
    task.finish_publication()
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        status=SessionStatus.TRANSFERRING,
        exception=None,
        cancelled_by_peer=False,
    )
    handle = TaskHandle(session, task, TOKENS)

    task.record_writer_result(0, False, wait_for_local_completion=False)
    task.fail(RuntimeError("first writer failed"))
    assert handle.poll().reports_pending is True

    task.record_writer_result(1, True, wait_for_local_completion=False)
    assert _owes_a_report(handle.poll()) is False


# ---------------------------------------------------------------------------
# An ending stands
# ---------------------------------------------------------------------------


def test_a_siblings_failure_does_not_reopen_a_delivered_piece():
    """Pieces share a session, so its verdict moves after one of them has already ended."""
    task = KVRecvTask(9, _sole_piece(), 0, DisaggregatedParams(disagg_request_id=9), aux_slot=None)
    task.expected_transfers = 1
    task.complete()
    task.note_writer_report(0, True)
    session = SimpleNamespace(
        status=SessionStatus.TRANSFERRED,
        exception=None,
        cancelled_by_peer=False,
        _closed=False,
    )
    handle = TaskHandle(session, task, TOKENS)
    assert isinstance(handle.poll(), Delivered)

    # A sibling of this piece fails, which is what the session as a whole now reports.
    session.status = SessionStatus.ERROR
    session.exception = RuntimeError("another piece failed")
    assert isinstance(handle.poll(), Delivered)


def test_a_latched_ending_still_tracks_whether_a_report_is_owed():
    """Only the ending is fixed; whether a writer still owes word keeps moving."""
    task = KVRecvTask(
        10, _sole_piece(), 0, DisaggregatedParams(disagg_request_id=10), aux_slot=None
    )
    task.expected_transfers = 2
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        status=SessionStatus.TRANSFERRING,
        exception=None,
        cancelled_by_peer=False,
        _closed=False,
    )
    handle = TaskHandle(session, task, TOKENS)

    task.note_writer_report(0, False)
    task.fail(RuntimeError("first writer failed"))
    first = handle.poll()
    assert isinstance(first, Failed)
    assert first.reports_pending is True

    task.note_writer_report(1, True)
    later = handle.poll()
    assert isinstance(later, Failed)
    assert later.reason == first.reason
    assert _owes_a_report(later) is False


def test_a_delivered_piece_survives_its_session_being_cancelled():
    """Cancelling is not refused just because the session ended.

    Peers may still be touching memory, and this is what tells them to stop. What a piece already
    reported is the handle's to keep.
    """
    session, handle = _receiving_from(1)
    _report(session, 0, AgentResult.SUCCESS)
    assert session.status is SessionStatus.TRANSFERRED
    assert isinstance(handle.poll(), Delivered)

    assert session.cancel_local() is True

    assert isinstance(handle.poll(), Delivered)


def test_a_failed_session_still_tells_its_peers_to_stop():
    """One writer failing does not stop the others; cancelling is how they are told."""
    session, _ = _receiving_from(2)
    _report(session, 0, AgentResult.FAILED)
    assert session.status is SessionStatus.ERROR

    assert session.cancel_local() is True


def _unstarted_piece(rid: int) -> tuple[RxSession, TaskHandle]:
    """A piece admitted but not yet published, which is what a cancel arrives in time to stop."""
    session = RxSession(
        request_id=rid,
        params=DisaggregatedParams(disagg_request_id=rid),
        receiver=_stub_receiver(),
        prompt_len=TOKENS,
    )
    session.receive(_sole_piece())
    task = session._kv_tasks[0]
    assert task.status is TaskStatus.INIT
    return session, TaskHandle(session, task, TOKENS)


def test_a_piece_the_cancel_stopped_reports_as_cancelled():
    """A cancel ends what had not started by failing it.

    That is also how a broken transfer ends, so the task's own state cannot tell the two apart and
    the caller would be told the request failed when it was stopped.
    """
    session, handle = _unstarted_piece(13)

    assert session.cancel_local() is True

    assert session._kv_tasks[0].status is TaskStatus.ERROR
    outcome = handle.poll()
    assert isinstance(outcome, Cancelled)
    assert outcome.by_peer is False


def test_who_asked_survives_the_mapping():
    """A peer's cancel is a transfer error and ours is an ordinary end; only the session knows."""
    session, handle = _unstarted_piece(14)

    session.cancel_local(by_peer=True)

    assert handle.poll().by_peer is True


def test_a_piece_that_broke_on_its_own_is_not_relabelled_by_a_later_cancel():
    """Cancelling a session that already has a broken piece does not rewrite why that piece ended."""
    session, handle = _unstarted_piece(15)
    session._kv_tasks[0].fail(RuntimeError("peer died"))

    session.cancel_local()

    outcome = handle.poll()
    assert isinstance(outcome, Failed)
    assert "peer died" in outcome.reason


def test_a_piece_that_landed_is_delivered_even_if_the_request_was_cancelled():
    """Cancellation stops what has not happened; it does not un-write bytes already in place."""
    task = KVRecvTask(
        12, _sole_piece(), 0, DisaggregatedParams(disagg_request_id=12), aux_slot=None
    )
    task.expected_transfers = 1
    task.note_writer_report(0, True)
    task.complete()
    session = SimpleNamespace(
        status=SessionStatus.CANCELLED,
        exception=None,
        cancelled_by_peer=False,
        _closed=False,
    )

    outcome = TaskHandle(session, task, TOKENS).poll()
    assert isinstance(outcome, Delivered)
    assert outcome.token_end == TOKENS


# ---------------------------------------------------------------------------
# A cancel off the wire stays the peer's, whenever it lands
# ---------------------------------------------------------------------------


def _wired_receiver(owns_transfers: bool = False) -> Receiver:
    """The receive-side tables a cancel and a session setup touch, plus what teardown reaches."""
    receiver = object.__new__(Receiver)
    receiver._enforce_physical_ownership = owns_transfers
    receiver._sessions = {}
    receiver._sessions_lock = threading.Lock()
    receiver._pre_cancelled_rids = {}
    receiver._shutdown = False
    receiver._ownership_admission_lock = threading.Lock()
    receiver._ownership_poisoned = None
    receiver._bounce = MagicMock()
    receiver._dealers = {}
    receiver._messenger = MagicMock()
    receiver.dispatch_task = MagicMock()
    receiver.send_cancel_to_senders = MagicMock()
    return receiver


def _wired_sender() -> Sender:
    """The send-side tables the same two paths touch.

    Already shut down, so the destructor's own shutdown returns before reaching the sockets and
    worker threads a real sender owns.
    """
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = False
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._pre_cancelled_rids = {}
    sender._shutdown = True
    sender._shutdown_requested = False
    sender._peer_requests = {}
    sender._peer_requests_timestamps = {}
    sender._peer_requests_lock = threading.Lock()
    sender.send_cancel_to_receivers = MagicMock()
    return sender


def _cancel_off_the_wire(side, rid: int) -> None:
    """The peer's CANCEL_SESSION, handed over the way the listener thread hands it over."""
    side._handle_cancel_session([MessageType.CANCEL_SESSION, str(rid).encode("ascii")])


def _receiving_over(receiver: Receiver, rid: int) -> RxSession:
    """A session on a real receiver, which is what a cancel off the wire arrives at."""
    return RxSession(
        request_id=rid,
        params=DisaggregatedParams(disagg_request_id=rid),
        receiver=receiver,
        prompt_len=TOKENS,
    )


def test_a_cancel_parked_before_the_session_existed_is_still_the_peers():
    """A cancel for a session not yet built is parked by id, and the session applies it later.

    Parking the id alone loses who asked, so the peer's cancel surfaces as our own.
    """
    receiver = _wired_receiver()
    _cancel_off_the_wire(receiver, 16)

    session = _receiving_over(receiver, 16)

    assert session.status is SessionStatus.CANCELLED
    session.receive(_sole_piece())
    outcome = TaskHandle(session, session._kv_tasks[0], TOKENS).poll()
    assert isinstance(outcome, Cancelled)
    assert outcome.by_peer is True


def test_a_queued_piece_the_peer_cancelled_says_who_asked():
    """The cancel arrives with the piece queued, so the session is there to take it."""
    receiver = _wired_receiver()
    session = _receiving_over(receiver, 17)
    session.receive(_sole_piece())
    task = session._kv_tasks[0]
    assert task.status is TaskStatus.INIT

    _cancel_off_the_wire(receiver, 17)

    assert task.status is TaskStatus.ERROR
    outcome = TaskHandle(session, task, TOKENS).poll()
    assert isinstance(outcome, Cancelled)
    assert outcome.by_peer is True


def test_a_transferring_piece_the_peer_cancelled_says_who_asked():
    """A piece mid-write keeps running, so its ending comes from the session's verdict."""
    receiver = _wired_receiver()
    session = _receiving_over(receiver, 18)
    session.receive(_sole_piece())
    task = session._kv_tasks[0]
    task.expected_transfers = 1
    session.mark_transferring(task.slice_id)

    _cancel_off_the_wire(receiver, 18)

    assert task.status is TaskStatus.TRANSFERRING
    outcome = TaskHandle(session, task, TOKENS).poll()
    assert isinstance(outcome, Cancelled)
    assert outcome.by_peer is True
    assert _owes_a_report(outcome) is True


def test_a_fetch_the_cancel_beat_reports_the_peers_cancel_rather_than_a_failure():
    """A session already cancelled takes no task, so the piece reaches the adapter with none.

    Reporting that as a failure makes the same remote cancel two different endings, decided by
    whether it arrived before or after the piece was admitted.
    """
    receiver = _wired_receiver(owns_transfers=True)
    _cancel_off_the_wire(receiver, 19)
    session = _receiving_over(receiver, 19)
    assert session.status is SessionStatus.CANCELLED
    worker = SimpleNamespace(create_rx_session=lambda request: session)
    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=19), request_id=19
    )

    attempt = PeerFetch(worker, request).fetch(CacheExtent(name=19, local=_sole_piece()))

    assert session._kv_tasks == []
    outcome = attempt.poll()
    assert isinstance(outcome, Cancelled)
    assert outcome.by_peer is True
    # No peer was told where to write, so there is no report to wait for.
    assert _owes_a_report(outcome) is False


def test_a_send_session_reads_a_parked_cancel_as_the_peers_too():
    """The other direction parks cancels the same way, and loses the same thing."""
    sender = _wired_sender()
    _cancel_off_the_wire(sender, 20)

    session = TxSession(
        request_id=20, params=DisaggregatedParams(disagg_request_id=20), sender=sender
    )

    assert session.status is SessionStatus.CANCELLED
    sender.dispatch_task = MagicMock()
    session.send(_sole_piece())
    task = session.kv_tasks[0]
    outcome = TaskHandle(session, task, TOKENS).poll()
    assert isinstance(outcome, Cancelled)
    assert outcome.by_peer is True


# ---------------------------------------------------------------------------
# Logical decisions belong to the transition, not the first observer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("poll_before_completion", [False, True])
@pytest.mark.parametrize("ending", ["failure", "local_cancel", "peer_cancel"])
def test_receive_outcome_does_not_depend_on_polling_before_late_completion(
    ending: str, poll_before_completion: bool
) -> None:
    session, handle = _receiving_from(1)
    if ending == "failure":
        session.fail_admission(RuntimeError("another publication failed"))
    else:
        session.cancel_local(by_peer=ending == "peer_cancel")
    if poll_before_completion:
        assert handle.poll() is not None

    # The writer finishes after the logical decision. Its report still drains the transfer.
    _report(session, 0, AgentResult.SUCCESS)

    for observer in (handle, TaskHandle(session, session._kv_tasks[0], TOKENS)):
        outcome = observer.poll()
        if ending == "failure":
            assert isinstance(outcome, Failed)
            assert "another publication failed" in outcome.reason
        else:
            assert isinstance(outcome, Cancelled)
            assert outcome.by_peer is (ending == "peer_cancel")
        assert outcome.reports_pending is False


@pytest.mark.parametrize("by_peer", [False, True])
def test_receive_cancel_keeps_its_outcome_when_a_writer_later_fails(by_peer: bool) -> None:
    session, _ = _receiving_from(1)
    session.cancel_local(by_peer=by_peer)

    _report(session, 0, AgentResult.FAILED)

    outcome = TaskHandle(session, session._kv_tasks[0], TOKENS).poll()
    assert isinstance(outcome, Cancelled)
    assert outcome.by_peer is by_peer
    assert outcome.reports_pending is False


def test_receive_session_failure_precedes_later_cancel_without_an_observer() -> None:
    session, handle = _receiving_from(1)
    session.fail_admission(RuntimeError("first publication failure"))

    assert session.cancel_local() is True

    outcome = handle.poll()
    assert isinstance(outcome, Failed)
    assert "first publication failure" in outcome.reason
    assert outcome.reports_pending is True


def _sending_pieces(count: int = 1) -> tuple[Sender, TxSession]:
    sender = _wired_sender()
    sender.dispatch_task = MagicMock()
    sender._get_result_dealer = MagicMock()
    sender._instance_rank = 0
    session = TxSession(
        request_id=30, params=DisaggregatedParams(disagg_request_id=30), sender=sender
    )
    for _ in range(count):
        session.send(_sole_piece())
    return sender, session


@pytest.mark.parametrize("by_peer", [False, True])
@pytest.mark.parametrize("queued_status", [TaskStatus.INIT, TaskStatus.TRANSFERRING])
def test_queued_sender_abort_preserves_the_committed_cancellation(
    by_peer: bool, queued_status: TaskStatus
) -> None:
    sender, session = _sending_pieces()
    task = session.kv_tasks[0]
    task.status = queued_status
    session.cancel_local(by_peer=by_peer)
    # Execute the real worker's pre-submission abort branch after cancellation won.
    empty = SimpleNamespace(size=0)
    write_meta = SimpleNamespace(
        src_ptrs=empty,
        dst_ptrs=empty,
        sizes=empty,
        unique_rid=30,
        slice_id=0,
        receiver_slice_id=0,
        peer_rank=0,
        peer_endpoint="tcp://receiver:1234",
        task=task,
    )

    sender._deliver_kv_to_agent(write_meta)

    outcome = TaskHandle(session, task, TOKENS).poll()
    assert isinstance(outcome, Cancelled)
    assert outcome.by_peer is by_peer
    sender._get_result_dealer.return_value.send.assert_called_once()


@pytest.mark.parametrize("poll_before_completion", [False, True])
def test_sender_sibling_failure_is_stable_across_late_completion(
    poll_before_completion: bool,
) -> None:
    _, session = _sending_pieces(2)
    failed, pending = session.kv_tasks
    pending.status = TaskStatus.TRANSFERRING
    handle = TaskHandle(session, pending, TOKENS)

    failed.fail(RuntimeError("first sibling failed"))
    if poll_before_completion:
        assert isinstance(handle.poll(), Failed)
    pending.complete()

    for observer in (handle, TaskHandle(session, pending, TOKENS)):
        outcome = observer.poll()
        assert isinstance(outcome, Failed)
        assert "first sibling failed" in outcome.reason


def test_terminal_failure_cause_is_stable_without_polling() -> None:
    session, handle = _receiving_from(1)
    task = session._kv_tasks[0]
    task.fail(RuntimeError("original failure"))
    task.fail(RuntimeError("cleanup failure"))

    outcome = handle.poll()
    assert isinstance(outcome, Failed)
    assert outcome.reason == "original failure"


@pytest.mark.parametrize("ending", ["failure", "local_cancel", "peer_cancel"])
def test_delivered_outcome_precedes_later_session_terminal_events(ending: str) -> None:
    session, handle = _receiving_from(1)
    _report(session, 0, AgentResult.SUCCESS)
    if ending == "failure":
        session.fail_admission(RuntimeError("later failure"))
    else:
        session.cancel_local(by_peer=ending == "peer_cancel")

    assert isinstance(handle.poll(), Delivered)
    assert handle.poll().token_end == TOKENS


def test_delivered_sender_piece_survives_a_direct_sibling_failure_without_polling() -> None:
    _, session = _sending_pieces(2)
    delivered, failed = session.kv_tasks
    delivered.complete()

    failed.fail(RuntimeError("sibling failed later"))

    assert isinstance(TaskHandle(session, delivered, TOKENS).poll(), Delivered)
    assert isinstance(TaskHandle(session, failed, TOKENS).poll(), Failed)


@pytest.mark.parametrize("owns_transfers", [False, True])
@pytest.mark.parametrize("scatter_succeeded", [False, True])
@pytest.mark.parametrize("cancel_first", [False, True])
def test_scatter_callback_and_cancel_commit_in_event_order(
    monkeypatch: pytest.MonkeyPatch,
    owns_transfers: bool,
    scatter_succeeded: bool,
    cancel_first: bool,
) -> None:
    from tensorrt_llm._torch.disaggregation.native import bounce

    session, handle = _receiving_from(1, owns_transfers=owns_transfers)
    deferred = []
    monkeypatch.setattr(bounce, "scatter_write_result", lambda *args: deferred.append(args[-1]))
    _report(session, 0, AgentResult.SUCCESS)
    assert len(deferred) == 1
    assert handle.poll() is None
    if owns_transfers:
        assert session.resources_drained() is False

    ready, resume, finished = threading.Event(), threading.Event(), threading.Event()

    def finish_scatter() -> None:
        ready.set()
        if resume.wait(timeout=5):
            deferred[0](scatter_succeeded)
            finished.set()

    worker = threading.Thread(target=finish_scatter)
    worker.start()
    try:
        assert ready.wait(timeout=5)
        if cancel_first:
            session.cancel_local(by_peer=True)
            assert isinstance(handle.poll(), Cancelled)
            if owns_transfers:
                assert session.resources_drained() is False
        resume.set()
        assert finished.wait(timeout=5)
        if not cancel_first:
            session.cancel_local(by_peer=True)
    finally:
        resume.set()
        worker.join(timeout=5)
    assert not worker.is_alive()

    for observer in (handle, TaskHandle(session, session._kv_tasks[0], TOKENS)):
        outcome = observer.poll()
        if cancel_first:
            assert isinstance(outcome, Cancelled)
            assert outcome.by_peer is True
        elif scatter_succeeded:
            assert isinstance(outcome, Delivered)
        else:
            assert isinstance(outcome, Failed)
            assert "bounce scatter failed" in outcome.reason
    if owns_transfers:
        assert session.resources_drained() is True
