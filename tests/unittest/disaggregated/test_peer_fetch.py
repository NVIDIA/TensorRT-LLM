# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the native backend answers the contract's two questions.

The mapping is the whole point of the adapter: which member of ``Outcome`` a native session state
becomes, and whether a writer is still owed word about this piece. Both are exercised against stub
sessions, because the interesting states are the ones a real transfer reaches only under a race.

The last section covers what the transceiver holds once admission returns, since an adapter that
starts nothing still decides whether a request stays paired with a session the sweep can retire.
"""

import inspect
import re
from types import SimpleNamespace

import numpy as np
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
from tensorrt_llm._torch.disaggregation.base.transfer import get_unique_rid
from tensorrt_llm._torch.disaggregation.native.fetch import PeerFetch
from tensorrt_llm._torch.disaggregation.native.transfer import SessionStatus, TaskStatus
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only


def _owes_a_report(outcome) -> bool:
    """No conclusion yet, or one that is still owed a report.

    Local to the tests on purpose. It used to be exported from the contract, where the name read
    like permission to hand memory back -- which is the one thing it never meant.
    """
    return outcome is None or outcome.reports_pending


class _StubTask:
    """A receive task answers for its writers.

    Without that the handle falls back to the send-side probe, and the fan-in assertions would be
    testing the wrong branch. Transferring means no writer has reported yet; a test about a settled
    piece says so.
    """

    def __init__(self, status, reports_outstanding=True):
        self.status = status
        self._exception = None
        self.reports_outstanding = reports_outstanding


class _StubSession:
    """Only what the adapter reads, in states a real session reaches only under a race."""

    def __init__(self):
        self._kv_tasks = []
        self.status = SessionStatus.INIT
        self.cancelled_by_peer = False
        self.exception = None
        self.raise_on_receive = None
        self.append_before_raising = True
        self.admits = True
        self.cancel_committed = 0
        self.notified = 0
        self.closed = 0

    def receive(self, chunk, expected_write_bytes=None):
        if self.raise_on_receive is not None:
            if self.append_before_raising:
                self._kv_tasks.append(_StubTask(TaskStatus.TRANSFERRING))
            raise self.raise_on_receive
        if not self.admits:
            # A closed or already terminal session returns without taking a task.
            return
        self._kv_tasks.append(_StubTask(TaskStatus.TRANSFERRING))

    def fail_admission(self, error):
        self.status = SessionStatus.ERROR
        self.exception = error

    def cancel_local(self, by_peer=False):
        self.cancel_committed += 1
        return self.cancel_committed == 1

    def notify_cancel(self):
        self.notified += 1

    def cancel(self, by_peer=False):
        if not self.cancel_local(by_peer=by_peer):
            return False
        self.notify_cancel()
        return True

    def close(self):
        self.closed += 1
        return True


class _StubWorker:
    def __init__(self):
        self.session = _StubSession()
        self.sessions_created = 0

    def create_rx_session(self, request):
        self.sessions_created += 1
        return self.session


def _request(rid: int = 42):
    """Only what ``get_unique_rid`` reads off a request."""
    return SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid), request_id=rid
    )


def _extent(end=8, groups=None, name=42):
    blocks = groups if groups is not None else [np.array([1, 2], dtype=np.int64)]
    return CacheExtent(
        name=name,
        local=Chunk(
            block_ids_per_layer_groups=blocks,
            kind_per_layer_group=[CacheKind.PAGED] * len(blocks),
            token_range=TokenRange(start=0, end=end),
            is_last=True,
        ),
    )


def _fetch_one(end=8):
    worker = _StubWorker()
    peer = PeerFetch(worker, _request())
    attempt = peer.fetch(_extent(end))
    return worker, peer, attempt


# ---------------------------------------------------------------------------
# Starting a pull
# ---------------------------------------------------------------------------


def test_pieces_of_one_request_share_a_session():
    """Several pieces, one session: the session belongs to the request, not to the piece."""
    worker, peer, first = _fetch_one()
    second = peer.fetch(_extent())
    assert worker.sessions_created == 1
    assert first is not second
    assert len(worker.session._kv_tasks) == 2


def test_each_attempt_holds_its_own_piece():
    worker, peer, first = _fetch_one()
    second = peer.fetch(_extent())
    worker.session._kv_tasks[0].status = TaskStatus.TRANSFERRED
    assert isinstance(first.poll(), Delivered)
    assert second.poll() is None


def test_naming_a_peer_is_refused_rather_than_ignored():
    """Accepting ``src`` and ignoring it would read from a peer the caller did not ask for."""
    worker = _StubWorker()
    peer = PeerFetch(worker, _request())
    with pytest.raises(NotImplementedError):
        peer.fetch(_extent(), src="tcp://elsewhere:1")
    assert worker.sessions_created == 0


# ---------------------------------------------------------------------------
# Reading the outcome
# ---------------------------------------------------------------------------


def test_no_conclusion_while_the_piece_is_in_flight():
    _, _, attempt = _fetch_one()
    assert attempt.poll() is None
    assert _owes_a_report(attempt.poll()) is True


def test_delivery_reports_how_far_the_piece_reaches():
    worker, _, attempt = _fetch_one(end=64)
    worker.session._kv_tasks[0].status = TaskStatus.TRANSFERRED
    worker.session._kv_tasks[0].reports_outstanding = False
    outcome = attempt.poll()
    assert isinstance(outcome, Delivered)
    assert outcome.token_end == 64
    assert _owes_a_report(outcome) is False


def test_failure_is_reported_before_every_writer_has_gone_quiet():
    """The gate has to stay shut on a failure whose writers have not reported."""
    worker, _, attempt = _fetch_one()
    worker.session.status = SessionStatus.ERROR
    worker.session.exception = RuntimeError("peer died")
    outcome = attempt.poll()
    assert isinstance(outcome, Failed)
    assert "peer died" in outcome.reason
    assert outcome.reports_pending is True
    assert _owes_a_report(outcome) is True


def test_a_settled_failure_opens_the_gate():
    worker, _, attempt = _fetch_one()
    worker.session.status = SessionStatus.ERROR
    worker.session._kv_tasks[0].status = TaskStatus.ERROR
    worker.session._kv_tasks[0].reports_outstanding = False
    assert _owes_a_report(attempt.poll()) is False


def test_a_cancellation_says_who_asked():
    worker, _, attempt = _fetch_one()
    worker.session.status = SessionStatus.CANCELLED
    worker.session.cancelled_by_peer = True
    outcome = attempt.poll()
    assert isinstance(outcome, Cancelled)
    assert outcome.by_peer is True
    assert outcome.reports_pending is True


def test_failure_without_a_recorded_cause_still_says_something():
    worker, _, attempt = _fetch_one()
    worker.session.status = SessionStatus.ERROR
    assert isinstance(attempt.poll(), Failed)
    assert attempt.poll().reason


# ---------------------------------------------------------------------------
# Submission that fails
# ---------------------------------------------------------------------------


def test_a_failing_submission_is_passed_on_and_the_session_goes_terminal():
    """The error is passed on, not turned into a handle.

    Nothing polls handles yet, so the caller above is what stops the request; the session is still
    pushed terminal so the sweep can retire it.
    """
    worker = _StubWorker()
    worker.session.raise_on_receive = RuntimeError("second rank refused")
    peer = PeerFetch(worker, _request())

    with pytest.raises(RuntimeError, match="second rank refused"):
        peer.fetch(_extent())

    assert worker.session.status is SessionStatus.ERROR


def test_a_refused_piece_is_a_settled_failure():
    """A session already terminal takes no task, and a piece with no task reaches no peer."""
    worker = _StubWorker()
    worker.session.admits = False
    peer = PeerFetch(worker, _request())
    outcome = peer.fetch(_extent()).poll()
    assert isinstance(outcome, Failed)
    assert "terminal" in outcome.reason
    assert _owes_a_report(outcome) is False
    assert worker.session._kv_tasks == []


# ---------------------------------------------------------------------------
# What the transceiver is left holding
# ---------------------------------------------------------------------------


class _ExplodingWorker:
    """A worker whose sessions cannot be built, so the adapter never gets one."""

    def __init__(self, error):
        self.error = error

    def create_rx_session(self, request):
        raise self.error


def _gen_request(rid=7):
    return SimpleNamespace(
        request_id=rid,
        py_disaggregated_params=None,
        state=None,
        set_kv_cache_transfer_start=lambda now: None,
    )


def _transceiver_over(worker):
    """The tables and the few helpers the admission path and teardown reach.

    The receive-side tables and the few helpers the admission path calls.
    """
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._transfer_worker = worker
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._kv_size_rank_factor = 1
    transceiver._validate_bridge_req = lambda req, synchronous=False: True
    # Named after the request, the way the real builder does it: the adapter refuses an extent
    # built for anyone else.
    transceiver._create_cache_extent = lambda req: _extent(name=get_unique_rid(req))
    transceiver._chunk_num_bytes = lambda chunk: 0
    return transceiver


def test_a_session_that_cannot_be_built_leaves_nothing_behind():
    """With no session there is nothing to pair the request with, so nothing would retire it."""
    transceiver = _transceiver_over(_ExplodingWorker(RuntimeError("no rx session")))
    request = _gen_request()
    with pytest.raises(RuntimeError, match="no rx session"):
        KvCacheTransceiverV2.request_and_receive_async(transceiver, request)
    assert transceiver._recv_reqs == {}
    assert transceiver._recv_sessions == {}
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR


def test_the_transceiver_keeps_no_method_of_its_own():
    """No attribute may be a method of the instance itself.

    That is a cycle, so the object and everything it owns are freed by the collector rather than
    by the last reference going away. This one owns the transfer engine, whose teardown would then
    run on whichever thread happened to allocate next.
    """
    source = inspect.getsource(KvCacheTransceiverV2.__init__)
    self_bound = re.findall(r"^\s*self\.\w+\s*(?::[^=\n]*)?=\s*self\.\w+\s*$", source, re.M)

    assert self_bound == []


def test_a_failed_publication_stays_paired_with_its_request():
    """Peers may already hold the destination, so the sweep has to find both session and request."""
    worker = _StubWorker()
    worker.session.raise_on_receive = RuntimeError("second rank refused")
    transceiver = _transceiver_over(worker)
    request = _gen_request()

    with pytest.raises(RuntimeError, match="second rank refused"):
        KvCacheTransceiverV2.request_and_receive_async(transceiver, request)

    assert transceiver._recv_reqs[7] is request
    assert transceiver._recv_sessions[7] is worker.session


def test_a_failed_synchronous_submission_still_closes_its_session():
    """The blocking caller owns the teardown, and the adapter opens the session before it submits.

    Leaving the close to the destructor puts the transfer engine's teardown on whichever thread
    happens to drop the last reference.
    """
    worker = _StubWorker()
    worker.session.raise_on_receive = RuntimeError("second rank refused")
    transceiver = _transceiver_over(worker)
    request = _gen_request()

    with pytest.raises(RuntimeError, match="second rank refused"):
        KvCacheTransceiverV2.request_and_receive_sync(transceiver, request)

    assert worker.session.closed == 1
    assert transceiver._recv_reqs == {}
    assert transceiver._recv_sessions == {}
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR


def test_an_extent_for_another_request_is_refused():
    """An adapter moves ``extent.local`` for the request it holds.

    So the wrong extent would transfer the bound request's blocks under another name, and nothing
    else in the path would say so.
    """
    worker = _StubWorker()
    extent = _extent()
    extent.name = 99

    with pytest.raises(ValueError, match="bound to 42"):
        PeerFetch(worker, _request(42)).fetch(extent)
    assert worker.sessions_created == 0
