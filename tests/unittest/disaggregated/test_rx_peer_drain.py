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
"""RxSession.has_transferring_tasks(): the predicate that pins receiver KV pages.

It answers "may a remote peer still be writing into my destination buffers?".
cancel_request() must return False while it is True, so being wrong in one
direction corrupts KV and in the other direction hangs the request forever.

The three properties covered here are exactly the three ways it can go wrong:
  * ADP broadcast reaches more peers than will ever answer,
  * a peer can disappear without answering at all,
  * a duplicated answer must not count twice.
All three are load-bearing; there is no bespoke unwind path for a partial
dispatch, so the deadline has to carry that case too.
"""

from __future__ import annotations

import threading
import time

import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native import transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVRecvTask,
    RxSession,
    TaskStatus,
)


class _FakeBounce:
    """``bounced`` mimics the real transport: the scatter is queued, not inline."""

    def __init__(self, bounced: bool = False) -> None:
        self._bounced = bounced

    def record_failure(self, *_args, **_kwargs) -> None:
        pass

    def is_bounced(self, *_args, **_kwargs) -> bool:
        return self._bounced

    def record_result(self, *_args, **_kwargs) -> None:
        pass  # real impl defers on_done to the scatter worker

    def release_idle_reservation(self, *_args, **_kwargs) -> None:
        pass

    def orphan_reservation(self, *_args, **_kwargs) -> None:
        pass


class _FakeRegistrar:
    class _RankInfo:
        instance_name = "fake"
        instance_rank = 0

    self_rank_info = _RankInfo()


class _FakeReceiver:
    def __init__(self, bounced: bool = False) -> None:
        self._bounce = _FakeBounce(bounced)
        self._registrar = _FakeRegistrar()


class _FakeParams:
    disagg_request_id = 7
    ctx_request_id = None


class _FakeSessionArgs:
    params = _FakeParams()


def _make_session(
    num_peers_expected: int, *, need_aux: bool = False, bounced: bool = False
) -> RxSession:
    """Build an RxSession without touching ZMQ, NIXL or the peer registrar."""
    session = object.__new__(RxSession)
    session.lock = threading.Lock()
    session.request_id = 7
    session._base_args = _FakeSessionArgs()
    session._receiver = _FakeReceiver(bounced)
    session._need_aux = need_aux
    session._terminal_status = None
    session._exception = None
    session._kv_tasks = []
    session._aux_count = 0
    session._aux_status = TaskStatus.INIT
    session._aux_responded_peer_ranks = set()
    session._sender_endpoints = set()
    session._cancel_notified = False
    session._last_peer_progress = None
    session._drain_timeout_logged = False
    session.transfer_end_time = None
    session.kv_cache_size_bytes = 0

    task = object.__new__(KVRecvTask)
    task._event = threading.Event()
    task.slice_id = 0
    task.status = TaskStatus.INIT
    task.expected_transfers = num_peers_expected
    task.last_slice_count = 0
    task.dispatched = False
    task.responded_peer_ranks = set()
    task._exception = None
    task._perf_timer = None
    session._kv_tasks.append(task)
    return session


def _dispatch(session: RxSession, peer_ranks) -> None:
    for rank in peer_ranks:
        assert session.mark_peer_dispatched(0, rank, f"tcp://peer{rank}")


def _kv_result(session: RxSession, peer_rank: int, status=AgentResult.SUCCESS) -> None:
    session.process_kv_agent_result(peer_rank, 0, True, status)


# RxSession has a module-level attribute for the drain budget; patch it per test.
@pytest.fixture
def drain_timeout(monkeypatch):
    def _set(seconds: float) -> None:
        monkeypatch.setattr(transfer_mod, "_PEER_DRAIN_TIMEOUT_S", seconds)

    return _set


def test_untouched_session_is_not_transferring() -> None:
    session = _make_session(2)
    assert session.has_transferring_tasks() is False


def test_dispatched_peers_pin_the_session_until_they_answer() -> None:
    session = _make_session(2)
    _dispatch(session, [0, 1])
    assert session.has_transferring_tasks() is True

    _kv_result(session, 0)
    assert session.has_transferring_tasks() is True

    _kv_result(session, 1)
    assert session.has_transferring_tasks() is False


def test_adp_broadcast_ignores_peers_that_never_owned_the_request() -> None:
    # Gen-first ADP broadcasts REQUEST_DATA to every DP group, but only one
    # group holds the context request and replies. Waiting on dispatches rather
    # than on expected responders would pin these pages for the four silent
    # peers indefinitely.
    session = _make_session(2)
    _dispatch(session, [0, 1, 2, 3, 4, 5])

    _kv_result(session, 0)
    _kv_result(session, 1)

    assert session.has_transferring_tasks() is False


def test_failed_result_also_resolves_its_peer() -> None:
    session = _make_session(2)
    _dispatch(session, [0, 1])
    _kv_result(session, 0, AgentResult.FAILED)
    _kv_result(session, 1, AgentResult.FAILED)
    assert session.has_transferring_tasks() is False


def test_duplicate_result_does_not_resolve_a_second_peer() -> None:
    session = _make_session(2)
    _dispatch(session, [0, 1])
    _kv_result(session, 0)
    _kv_result(session, 0)  # duplicate; must be ignored
    assert session.has_transferring_tasks() is True


def test_silent_peer_releases_the_session_after_the_drain_deadline(drain_timeout) -> None:
    # A CTX worker killed mid-transfer never sends its terminal result. Without
    # a deadline the request would stay uncancellable forever and, because the
    # transceiver feeds this predicate into a cross-rank consensus, would stall
    # every other request in the same polling batch.
    drain_timeout(0.05)
    session = _make_session(2)
    _dispatch(session, [0, 1])
    _kv_result(session, 0)
    assert session.has_transferring_tasks() is True

    time.sleep(0.1)
    assert session.has_transferring_tasks() is False


def test_progress_rearms_the_drain_deadline(drain_timeout) -> None:
    # The budget is an inactivity budget, not a total-duration budget: a healthy
    # transfer that keeps producing results must never trip it.
    drain_timeout(2.0)
    session = _make_session(3)
    _dispatch(session, [0, 1, 2])

    for rank in (0, 1):
        time.sleep(0.3)  # well under the budget alone, over it cumulatively
        assert session.has_transferring_tasks() is True
        _kv_result(session, rank)

    _kv_result(session, 2)
    assert session.has_transferring_tasks() is False


def test_partial_dispatch_failure_falls_back_to_the_drain_deadline(drain_timeout) -> None:
    # Only 2 of 4 peers were reached before dispatch raised, so the responder
    # count can never be satisfied.  There is no bespoke unwind path: the
    # inactivity deadline is what releases the buffers.
    drain_timeout(0.05)
    session = _make_session(4)
    _dispatch(session, [0, 1])
    _kv_result(session, 0)
    _kv_result(session, 1)
    assert session.has_transferring_tasks() is True

    time.sleep(0.1)
    assert session.has_transferring_tasks() is False


def test_aux_transfer_keeps_the_session_pinned_until_aux_answers() -> None:
    session = _make_session(1, need_aux=True)
    _dispatch(session, [0])
    _kv_result(session, 0)
    assert session.has_transferring_tasks() is True

    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    assert session.has_transferring_tasks() is False


def test_bounced_transfer_stays_pinned_until_the_scatter_lands() -> None:
    # On the bounce path a SUCCESS result only means the data reached the bounce
    # arena; scatter_write_result queues the copy into the KV pages and
    # task.complete() runs later, in the scatter worker's on_done. Releasing on
    # responder count alone would free those pages under the pending scatter.
    session = _make_session(1, bounced=True)
    _dispatch(session, [0])
    _kv_result(session, 0)

    assert session._kv_tasks[0].responded_peer_ranks == {0}
    assert session._kv_tasks[0].status == TaskStatus.TRANSFERRING
    assert session.has_transferring_tasks() is True

    # What the scatter worker's on_done does once the copy has landed.
    session._kv_tasks[0].complete()
    assert session.has_transferring_tasks() is False


def test_cancel_is_idempotent_and_notifies_senders_once() -> None:
    session = _make_session(2)
    sent: list = []
    session._receiver.send_cancel_to_senders = lambda rid, eps: sent.append((rid, set(eps)))
    _dispatch(session, [0, 1])

    session.cancel()
    session.cancel()
    session.cancel()

    assert len(sent) == 1
    assert sent[0][1] == {"tcp://peer0", "tcp://peer1"}
    assert session._terminal_status == SessionStatus.CANCELLED


def test_dispatch_is_refused_once_the_session_is_cancelled() -> None:
    # dispatch_task must not contact a sender after cancellation won the race,
    # or that peer writes into buffers we are about to release.
    session = _make_session(2)
    session._receiver.send_cancel_to_senders = lambda *_a: None
    assert session.mark_peer_dispatched(0, 0, "tcp://peer0") is True

    session.cancel()

    assert session.mark_peer_dispatched(0, 1, "tcp://peer1") is False
    assert session._kv_tasks[0].responded_peer_ranks == set()
