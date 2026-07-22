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
"""Bounded polling and shutdown ownership tests for the native transceiver."""

from __future__ import annotations

import gc
import threading
import weakref
from dataclasses import dataclass
from typing import Optional
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.transfer import (
    Receiver,
    RxSession,
    Sender,
    TaskStatus,
    TransferWorker,
    TxSession,
)
from tensorrt_llm._torch.disaggregation.transceiver import (
    _NON_DRAINED_TRANSCEIVERS,
    KvCacheTransceiverV2,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.bindings import LlmRequestState


@dataclass
class _FakeRequest:
    state: Optional[LlmRequestState] = None
    request_id: int = 0
    py_disaggregated_params: Optional[object] = None


class _RetirementRequest:
    """Request with an optionally fallible terminal-state assignment."""

    def __init__(
        self,
        rid: int,
        *,
        fail_state: Optional[LlmRequestState] = None,
    ) -> None:
        self.request_id = rid
        self.py_request_id = rid
        self.py_disaggregated_params = None
        self.py_kv_cache_xfer_bytes = 64
        self.set_kv_cache_size = Mock()
        self._state: Optional[LlmRequestState] = None
        self._fail_state = fail_state
        self._failure_pending = fail_state is not None
        self.terminal_state_set_calls = 0

    @property
    def state(self) -> Optional[LlmRequestState]:
        return self._state

    @state.setter
    def state(self, value: LlmRequestState) -> None:
        if value in (
            LlmRequestState.DISAGG_CONTEXT_COMPLETE,
            LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
            LlmRequestState.DISAGG_TRANS_ERROR,
        ):
            self.terminal_state_set_calls += 1
        if value == self._fail_state:
            if self._failure_pending:
                self._failure_pending = False
                raise RuntimeError("terminal state update failed")
        self._state = value


class _FakeTransferWorker:
    def __init__(self, shutdown_results: Optional[list[bool]] = None) -> None:
        self.sweep_count = 0
        self.shutdown_results = shutdown_results or []
        self.shutdown_count = 0

    def sweep_stale_req_infos(self) -> None:
        self.sweep_count += 1

    def shutdown(self) -> bool:
        self.shutdown_count += 1
        if self.shutdown_results:
            return self.shutdown_results.pop(0)
        return True


class _FakeSession:
    def __init__(
        self,
        rid: int,
        wait_result: Optional[WaitResult],
        *,
        status: SessionStatus = SessionStatus.READY,
        is_completed: bool = False,
        has_failed: bool = False,
        has_transferring_tasks: bool = False,
    ) -> None:
        self._rid = rid
        self._wait_result = wait_result
        self._status = status
        self._is_completed = is_completed
        self._has_failed = has_failed
        self._has_transferring_tasks = has_transferring_tasks
        self.blocking_calls: list[bool] = []
        self.cancel_calls = 0
        self.closed = False

    @property
    def disagg_request_id(self) -> int:
        return self._rid

    @property
    def status(self) -> SessionStatus:
        return self._status

    def wait_complete(self, blocking: bool = True) -> Optional[WaitResult]:
        self.blocking_calls.append(blocking)
        return self._wait_result

    def is_completed(self) -> bool:
        return self._is_completed

    def has_failed(self) -> bool:
        return self._has_failed

    def cancel(self) -> None:
        self.cancel_calls += 1

    def has_transferring_tasks(self) -> bool:
        return self._has_transferring_tasks

    def close(self) -> None:
        self.closed = True


class _FakeWorkerThread:
    def __init__(self, *, alive: bool) -> None:
        self.alive = alive
        self.join_calls: list[Optional[float]] = []

    def join(self, timeout: Optional[float] = None) -> None:
        self.join_calls.append(timeout)

    def is_alive(self) -> bool:
        return self.alive


@dataclass
class _FakeReceiveTaskOwnership:
    lifecycle_managed: bool
    status: TaskStatus
    publication_started: bool = True

    @property
    def legacy_resources_drained(self) -> bool:
        return self.lifecycle_managed or not self.publication_started


class _FakeTask:
    def __init__(self, status: TaskStatus, wait_result: bool = True) -> None:
        self.status = status
        self._wait_result = wait_result
        self.wait_calls: list[Optional[float]] = []

    def wait(self, timeout: Optional[float] = None) -> bool:
        self.wait_calls.append(timeout)
        return self._wait_result


def _make_transceiver(
    sessions: dict[int, _FakeSession],
    reqs: Optional[dict[int, _FakeRequest]] = None,
) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._send_sessions = sessions
    transceiver._send_reqs = reqs or {rid: _FakeRequest() for rid in sessions}
    transceiver._sender_future_timeout_ms = 123
    # Attributes read by check_context_transfer_status before it processes sessions.
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._ctx_consensus = lambda local_ids: list(local_ids)
    transceiver._ctx_consensus_outcome = (
        lambda _to_process, cancelled, failed, completed, timed_out, cleanup_ready: (
            [rid for rid in cancelled if rid in cleanup_ready],
            [rid for rid in failed if rid in cleanup_ready],
            [rid for rid in completed if rid in cleanup_ready],
            timed_out,
        )
    )
    return transceiver


def _make_tx_session(
    kv_tasks: list[_FakeTask],
    *,
    need_aux: bool = False,
    aux_task: Optional[_FakeTask] = None,
) -> TxSession:
    session = object.__new__(TxSession)
    session._timeout_s = 0.25
    session._need_aux = need_aux
    session._terminal_status = None
    session.receiver_ready = True
    session.kv_tasks = kv_tasks
    session.aux_task = aux_task
    session._closed = False
    session._aux_buffer = None
    session.aux_slot = None
    session._sender = None
    return session


def test_context_transfer_status_bounded_poll_keeps_not_ready_session_queued() -> None:
    session = _FakeSession(rid=11, wait_result=None)
    transceiver = _make_transceiver({11: session})

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=1)

    assert completed == []
    assert failed == []
    assert session.blocking_calls == [False]
    assert not session.closed
    assert 11 in transceiver._send_sessions
    assert 11 in transceiver._send_reqs
    assert transceiver._transfer_worker.sweep_count == 1


def test_context_transfer_status_retains_cancelled_source_until_physically_terminal() -> None:
    session = _FakeSession(
        rid=19,
        wait_result=None,
        status=SessionStatus.CANCELLED,
        has_failed=True,
        has_transferring_tasks=True,
    )
    transceiver = _make_transceiver({19: session})

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=1)

    assert completed == []
    assert failed == []
    assert session.blocking_calls == [False]
    assert not session.closed
    assert 19 in transceiver._send_sessions
    assert 19 in transceiver._send_reqs


def test_context_transfer_status_reports_drained_remote_cancellation_as_failed() -> None:
    rid = 24
    session = _FakeSession(
        rid=rid,
        wait_result=WaitResult.FAILED,
        status=SessionStatus.CANCELLED,
        has_failed=True,
    )
    req = _FakeRequest(request_id=rid)
    transceiver = _make_transceiver({rid: session}, {rid: req})

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=1)

    assert completed == []
    assert failed == [rid]
    assert session.closed
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert rid not in transceiver._send_sessions
    assert rid not in transceiver._send_reqs


def test_context_transfer_status_block_all_uses_blocking_wait() -> None:
    session = _FakeSession(rid=12, wait_result=WaitResult.COMPLETED)
    req = _FakeRequest()
    transceiver = _make_transceiver({12: session}, {12: req})

    completed, failed = transceiver.check_context_transfer_status(
        at_least_request_num=None,
        mark_complete=True,
    )

    assert completed == [12]
    assert failed == []
    assert session.blocking_calls == [True]
    assert session.closed
    assert req.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    assert 12 not in transceiver._send_sessions
    assert 12 not in transceiver._send_reqs


def test_context_transfer_status_zero_budget_processes_task_level_failure() -> None:
    session = _FakeSession(
        rid=13,
        wait_result=WaitResult.FAILED,
        has_failed=True,
    )
    req = _FakeRequest()
    transceiver = _make_transceiver({13: session}, {13: req})

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == [13]
    assert session.blocking_calls == [False]
    assert session.closed
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert 13 not in transceiver._send_sessions
    assert 13 not in transceiver._send_reqs


def test_context_provisional_cleanup_retries_and_readvertises_without_extra_consensus() -> None:
    rid = 70
    request = _RetirementRequest(rid)
    session = _FakeSession(
        rid=rid,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )
    session.close = Mock(side_effect=[RuntimeError("close failed"), None])
    transceiver = _make_transceiver({rid: session}, {rid: request})
    transceiver._ctx_consensus = Mock(side_effect=lambda local_ids: list(local_ids))

    assert transceiver.check_context_transfer_status(0, mark_complete=True) == ([], [])
    assert request.state is None
    assert transceiver._send_sessions == {rid: session}
    assert transceiver._send_reqs == {rid: request}
    assert set(transceiver._get_context_retirements()) == {rid}

    # The closed/provisional candidate remains in the normal ready
    # advertisement. The existing outcome consensus is the only prepare
    # barrier; no post-decision consensus round is added.
    assert transceiver.check_context_transfer_status(0, mark_complete=True) == ([rid], [])
    consensus_inputs = [call.args[0] for call in transceiver._ctx_consensus.call_args_list]
    assert consensus_inputs == [[rid], [rid]]
    assert transceiver._get_context_retirements() == {}
    assert transceiver._send_sessions == {}
    assert transceiver._send_reqs == {}
    assert request.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    assert session.close.call_count == 2


def test_context_commit_invariant_failure_latches_fail_stop() -> None:
    rid = 76
    request = _RetirementRequest(rid, fail_state=LlmRequestState.DISAGG_CONTEXT_COMPLETE)
    session = _FakeSession(
        rid=rid,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )
    transceiver = _make_transceiver({rid: session}, {rid: request})

    with pytest.raises(RuntimeError, match="terminal state update failed"):
        transceiver.check_context_transfer_status(0, mark_complete=True)

    assert transceiver._shutdown_started
    assert transceiver._retirement_fault is not None
    assert set(transceiver._get_context_retirements()) == {rid}
    with pytest.raises(RuntimeError, match="retirement invariant failed"):
        transceiver.check_context_transfer_status(0, mark_complete=True)
    _NON_DRAINED_TRANSCEIVERS.discard(transceiver)


def test_pending_context_retirement_does_not_block_positive_wait_on_unrelated_request() -> None:
    rid = 71
    session = _FakeSession(
        rid=rid,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )
    session.close = Mock(side_effect=[RuntimeError("close failed"), None])
    request = _RetirementRequest(rid)
    transceiver = _make_transceiver({rid: session}, {rid: request})
    transceiver._ctx_consensus = Mock(side_effect=lambda local_ids: list(local_ids))

    assert transceiver.check_context_transfer_status(0, mark_complete=True) == ([], [])

    unrelated_rid = 72
    unrelated_session = _FakeSession(rid=unrelated_rid, wait_result=None)
    transceiver._send_sessions[unrelated_rid] = unrelated_session
    transceiver._send_reqs[unrelated_rid] = _FakeRequest(request_id=unrelated_rid)

    assert transceiver.check_context_transfer_status(1, mark_complete=True) == ([rid], [])
    assert unrelated_session.blocking_calls == []
    assert unrelated_rid in transceiver._send_sessions


def test_provisional_context_owner_blocks_cancellation_and_rid_replacement() -> None:
    rid = 77
    request = _RetirementRequest(rid)
    replacement = _RetirementRequest(rid)
    session = _FakeSession(
        rid=rid,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )
    session.close = Mock(side_effect=RuntimeError("close pending"))
    transceiver = _make_transceiver({rid: session}, {rid: request})
    transceiver._wait_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}

    assert transceiver.check_context_transfer_status(0) == ([], [])
    assert set(transceiver._get_context_retirements()) == {rid}
    assert not transceiver.cancel_request(request)
    with pytest.raises(RuntimeError, match="pending terminal retirement"):
        transceiver.respond_and_send_async(replacement)

    assert transceiver._send_sessions == {rid: session}
    assert transceiver._send_reqs == {rid: request}


def test_active_status_snapshot_blocks_cancellation_retirement_until_enrollment() -> None:
    rid = 76
    request = _RetirementRequest(rid)
    session = _FakeSession(
        rid=rid,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )
    transceiver = _make_transceiver({rid: session}, {rid: request})
    transceiver._wait_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}

    snapshot_taken = threading.Event()
    release_consensus = threading.Event()

    def pause_after_snapshot(local_ids: list[int]) -> list[int]:
        snapshot_taken.set()
        assert release_consensus.wait(timeout=2)
        return list(local_ids)

    transceiver._ctx_consensus = pause_after_snapshot
    status_results: list[tuple[list[int], list[int]]] = []
    errors: list[BaseException] = []

    def poll_status() -> None:
        try:
            status_results.append(transceiver.check_context_transfer_status(0))
        except BaseException as error:
            errors.append(error)

    poll_thread = threading.Thread(target=poll_status)
    poll_thread.start()
    assert snapshot_taken.wait(timeout=1)
    try:
        assert transceiver.cancel_request(request) is False
        assert session.cancel_calls == 1
        assert not session.closed
        assert transceiver._send_sessions == {rid: session}
        assert transceiver._send_reqs == {rid: request}
    finally:
        release_consensus.set()
        poll_thread.join(timeout=2)

    assert not poll_thread.is_alive()
    assert errors == []
    assert status_results == [([rid], [])]
    assert transceiver._send_sessions == {}
    assert transceiver._send_reqs == {}


def test_prepare_context_cancellation_does_not_reverse_consensus_promotion() -> None:
    rid = 75
    request = _FakeRequest(request_id=rid)
    transceiver = _make_transceiver({})
    transceiver._wait_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker.has_all_peer_req_infos_for_send = Mock(return_value=True)

    consensus_started = threading.Event()
    release_consensus = threading.Event()

    def pause_consensus(local_ids: list[int]) -> list[int]:
        consensus_started.set()
        assert release_consensus.wait(timeout=2)
        return list(local_ids)

    transceiver._ctx_consensus = pause_consensus
    errors: list[BaseException] = []

    def prepare() -> None:
        try:
            transceiver.prepare_context_requests([request])
        except BaseException as error:
            errors.append(error)

    prepare_thread = threading.Thread(target=prepare)
    prepare_thread.start()
    assert consensus_started.wait(timeout=1)
    try:
        assert transceiver.cancel_request(request) is False
        assert transceiver._wait_reqs == {rid: request}
        assert transceiver._get_cancelled_wait_reqs() == {rid: request}
    finally:
        release_consensus.set()
        prepare_thread.join(timeout=2)

    assert not prepare_thread.is_alive()
    assert errors == []
    assert request.state == LlmRequestState.CONTEXT_INIT
    assert transceiver._wait_reqs == {}
    assert transceiver._get_cancelled_wait_reqs() == {rid: request}

    assert transceiver.cancel_request(request) is True
    assert transceiver._wait_reqs == {}
    assert transceiver._get_cancelled_wait_reqs() == {}


def test_prepare_context_cancellation_before_wait_enrollment_blocks_admission() -> None:
    rid = 73
    request = _FakeRequest(request_id=rid)
    transceiver = _make_transceiver({})
    transceiver._wait_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker.has_all_peer_req_infos_for_send = Mock(return_value=True)
    transceiver._ctx_consensus = Mock(side_effect=lambda local_ids: list(local_ids))

    status_admitted = threading.Event()
    release_enrollment = threading.Event()
    prepare_impl = transceiver._prepare_context_requests_impl

    def pause_before_enrollment(requests: list[_FakeRequest]) -> None:
        status_admitted.set()
        assert release_enrollment.wait(timeout=2)
        prepare_impl(requests)

    transceiver._prepare_context_requests_impl = pause_before_enrollment
    errors: list[BaseException] = []

    def prepare() -> None:
        try:
            transceiver.prepare_context_requests([request])
        except BaseException as error:
            errors.append(error)

    prepare_thread = threading.Thread(target=prepare)
    prepare_thread.start()
    assert status_admitted.wait(timeout=1)
    try:
        assert transceiver.cancel_request(request) is False
        assert transceiver._wait_reqs == {}
        assert transceiver._get_cancelled_wait_reqs() == {rid: request}
    finally:
        release_enrollment.set()
        prepare_thread.join(timeout=2)

    assert not prepare_thread.is_alive()
    assert errors == []
    assert request.state == LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
    assert transceiver._wait_reqs == {rid: request}
    assert transceiver._get_cancelled_wait_reqs() == {rid: request}
    transceiver._ctx_consensus.assert_called_once_with([])

    assert transceiver.cancel_request(request) is True
    assert transceiver._wait_reqs == {}
    assert transceiver._get_cancelled_wait_reqs() == {}


def test_respond_and_send_rejects_wait_ledger_owner_collision() -> None:
    rid = 74
    original = _FakeRequest(request_id=rid)
    replacement = _FakeRequest(request_id=rid)
    transceiver = _make_transceiver({})
    transceiver._shutdown_started = False
    transceiver._wait_reqs = {rid: original}

    with pytest.raises(RuntimeError, match="different live source owner"):
        transceiver.respond_and_send_async(replacement)
    with pytest.raises(RuntimeError, match="still waiting for scheduler promotion"):
        transceiver.respond_and_send_async(original)

    assert transceiver._send_sessions == {}
    assert transceiver._send_reqs == {}
    assert transceiver._wait_reqs == {rid: original}


def test_context_transfer_status_skips_consensus_when_never_sent() -> None:
    # A worker that never sends skips the ctx consensus even when TP sync would need it, but still
    # sweeps so nothing leaks.
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = False
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_need_pp_sync = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._ctx_consensus = Mock(side_effect=AssertionError("consensus must be skipped"))

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    transceiver._ctx_consensus.assert_not_called()
    assert transceiver._transfer_worker.sweep_count == 1


def test_context_transfer_idle_fast_path_still_runs_sender_progress() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = False
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._transfer_worker = _FakeTransferWorker()

    assert transceiver.check_context_transfer_status(at_least_request_num=0) == ([], [])
    assert transceiver._transfer_worker.sweep_count == 1


def test_gen_transfer_status_enters_consensus_when_sync_required() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = False
    transceiver._gen_need_sync = True
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._gen_consensus = Mock(return_value=[])
    transceiver._build_to_process = Mock(return_value=[])
    transceiver._gen_consensus_outcome = Mock(return_value=([], [], []))
    transceiver._dist = Mock(rank=0)

    completed, failed, cancelled = transceiver.check_gen_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    assert cancelled == []
    transceiver._gen_consensus.assert_called_once_with([])
    transceiver._gen_consensus_outcome.assert_called_once_with([], [], [], [], [])


def test_gen_transfer_status_retains_logical_failure_until_physical_drain() -> None:
    rid = 14
    session = _FakeSession(
        rid=rid,
        wait_result=None,
        has_failed=True,
    )
    req = _FakeRequest(request_id=rid)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: req}
    transceiver._gen_consensus = lambda local_ids: list(local_ids)
    transceiver._gen_consensus_outcome = (
        lambda _to_process, cancelled, failed, completed, _cleanup_ready: (
            cancelled,
            failed,
            completed,
        )
    )

    completed, failed, cancelled = transceiver.check_gen_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    assert cancelled == []
    assert session.blocking_calls == [False]
    assert not session.closed
    assert req.state is None
    assert transceiver._recv_sessions == {rid: session}
    assert transceiver._recv_reqs == {rid: req}

    session._wait_result = WaitResult.FAILED
    transceiver._dist = Mock(rank=0)
    completed, failed, cancelled = transceiver.check_gen_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == [rid]
    assert cancelled == []
    assert session.closed
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert rid not in transceiver._recv_sessions
    assert rid not in transceiver._recv_reqs


def test_gen_transfer_status_retains_cancelled_session_until_wait_is_terminal() -> None:
    rid = 16
    session = _FakeSession(
        rid=rid,
        wait_result=None,
        status=SessionStatus.CANCELLED,
        has_failed=True,
    )
    req = _FakeRequest(request_id=rid)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: req}
    transceiver._gen_consensus = lambda local_ids: list(local_ids)
    transceiver._gen_consensus_outcome = (
        lambda _to_process, cancelled, failed, completed, _cleanup_ready: (
            cancelled,
            failed,
            completed,
        )
    )

    completed, failed, cancelled = transceiver.check_gen_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    assert cancelled == []
    assert not session.closed
    assert transceiver._recv_sessions == {rid: session}
    assert transceiver._recv_reqs == {rid: req}

    session._wait_result = WaitResult.FAILED
    completed, failed, cancelled = transceiver.check_gen_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    assert cancelled == [req]
    assert session.closed
    assert rid not in transceiver._recv_sessions
    assert rid not in transceiver._recv_reqs


@pytest.mark.parametrize(
    "failure_stage",
    ["kv_size", "aux", "history", "close"],
)
def test_generation_provisional_preparation_retries_each_fallible_phase(failure_stage: str) -> None:
    rid = 73
    request = _RetirementRequest(rid)
    session = _FakeSession(
        rid=rid,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )
    session.close = Mock()
    if failure_stage == "close":
        session.close.side_effect = [RuntimeError("close failed"), None]

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver.kv_transfer_poll_interval_ms = 0
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: request}
    transceiver._gen_consensus = Mock(side_effect=lambda local_ids: list(local_ids))
    transceiver._gen_consensus_outcome = (
        lambda _to_process, cancelled, failed, completed, cleanup_ready: (
            [rid for rid in cancelled if rid in cleanup_ready],
            [rid for rid in failed if rid in cleanup_ready],
            [rid for rid in completed if rid in cleanup_ready],
        )
    )
    transceiver._need_aux_transfer = Mock(return_value=True)
    transceiver._apply_aux = Mock()
    transceiver._assert_disagg_history_declared = Mock()
    transceiver._dist = Mock(rank=0)
    if failure_stage == "kv_size":
        request.set_kv_cache_size.side_effect = [
            RuntimeError("kv size failed"),
            None,
        ]
    elif failure_stage == "aux":
        transceiver._apply_aux.side_effect = [RuntimeError("aux failed"), None]
    elif failure_stage == "history":
        transceiver._assert_disagg_history_declared.side_effect = [
            RuntimeError("history failed"),
            None,
        ]

    assert transceiver.check_gen_transfer_status(0) == ([], [], [])
    progress = transceiver._get_generation_retirements()[rid]
    failed_flag = {
        "kv_size": "kv_size_set",
        "aux": "aux_applied",
        "history": "history_checked",
        "close": "session_closed",
    }[failure_stage]
    assert not getattr(progress, failed_flag)
    assert request.state is None
    assert transceiver._recv_sessions == {rid: session}
    assert transceiver._recv_reqs == {rid: request}

    assert transceiver.check_gen_transfer_status(0) == ([rid], [], [])
    consensus_inputs = [call.args[0] for call in transceiver._gen_consensus.call_args_list]
    assert consensus_inputs == [[rid], [rid]]
    assert transceiver._get_generation_retirements() == {}
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}
    assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE

    expected_kv_size_calls = 2 if failure_stage == "kv_size" else 1
    assert request.set_kv_cache_size.call_count == expected_kv_size_calls
    request.set_kv_cache_size.assert_called_with(64)
    expected_aux_calls = 2 if failure_stage == "aux" else 1
    assert transceiver._apply_aux.call_count == expected_aux_calls
    expected_history_calls = 2 if failure_stage == "history" else 1
    assert transceiver._assert_disagg_history_declared.call_count == expected_history_calls
    expected_close_calls = 2 if failure_stage == "close" else 1
    assert session.close.call_count == expected_close_calls
    assert request.terminal_state_set_calls == 1


def test_generation_provisional_candidate_waits_for_peer_and_readvertises() -> None:
    rid = 75
    request = _RetirementRequest(rid)
    session = _FakeSession(
        rid=rid,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = True
    transceiver.kv_transfer_poll_interval_ms = 0
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: request}
    transceiver._gen_consensus = Mock(side_effect=lambda local_ids: list(local_ids))
    # The local candidate is fully prepared on the first poll, but the peer is
    # not cleanup-ready. On the next poll the peer catches up and the same
    # existing outcome consensus admits the decision.
    transceiver._gen_consensus_outcome = Mock(side_effect=[([], [], []), ([], [], [rid])])
    transceiver._need_aux_transfer = Mock(return_value=False)
    transceiver._assert_disagg_history_declared = Mock()
    transceiver._dist = Mock(rank=0)

    assert transceiver.check_gen_transfer_status(0) == ([], [], [])
    assert request.state is None
    assert transceiver._recv_sessions == {rid: session}
    assert transceiver._recv_reqs == {rid: request}
    assert set(transceiver._get_generation_retirements()) == {rid}
    assert session.closed
    assert session.blocking_calls == [False]

    assert transceiver.check_gen_transfer_status(0) == ([rid], [], [])
    assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}
    assert session.blocking_calls == [False]
    assert [call.args[0] for call in transceiver._gen_consensus.call_args_list] == [[rid], [rid]]


def test_shutdown_retains_decided_retirement_until_local_commit() -> None:
    rid = 74
    close_allowed = False
    session = _FakeSession(
        rid=rid,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )

    def close_session() -> None:
        if not close_allowed:
            raise RuntimeError("close still pending")
        session.closed = True

    session.close = Mock(side_effect=close_session)
    request = _RetirementRequest(rid)
    transceiver = _make_transceiver({rid: session}, {rid: request})
    transceiver._shutdown = False
    transceiver._shutdown_started = False
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._wait_reqs = {}

    assert transceiver.check_context_transfer_status(0, mark_complete=True) == ([], [])
    try:
        assert transceiver.shutdown() is False
        assert transceiver in _NON_DRAINED_TRANSCEIVERS
        assert set(transceiver._get_context_retirements()) == {rid}
        assert transceiver._send_sessions == {rid: session}
        assert session.cancel_calls == 0

        close_allowed = True
        assert transceiver.shutdown() is True
        assert transceiver not in _NON_DRAINED_TRANSCEIVERS
        assert transceiver._get_context_retirements() == {}
        assert transceiver._send_sessions == {}
        assert request.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    finally:
        _NON_DRAINED_TRANSCEIVERS.discard(transceiver)


def test_gen_transfer_timeout_is_not_cleanup_ready() -> None:
    rid = 22
    session = _FakeSession(rid=rid, wait_result=WaitResult.TIMEOUT)
    req = _FakeRequest(request_id=rid)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: req}
    transceiver._gen_consensus = lambda local_ids: list(local_ids)
    transceiver._gen_consensus_outcome = Mock(return_value=([], [], []))

    completed, failed, cancelled = transceiver.check_gen_transfer_status(at_least_request_num=None)

    assert completed == []
    assert failed == []
    assert cancelled == []
    transceiver._gen_consensus_outcome.assert_called_once_with([rid], [], [], [], [])
    assert not session.closed
    assert transceiver._recv_sessions == {rid: session}
    assert transceiver._recv_reqs == {rid: req}


def test_async_receive_enrolls_owner_before_allocation_and_retains_on_publication_error() -> None:
    rid = 17
    req = _FakeRequest(request_id=rid)
    session = Mock()
    session.receive.side_effect = RuntimeError("publication failed")
    session.has_transferring_tasks.return_value = True
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = False
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._kv_size_rank_factor = 1
    transceiver._slice_num_bytes = Mock(return_value=8)

    def create_rx_session(owner):
        assert transceiver._recv_reqs == {rid: owner}
        return session

    def create_kv_slice(owner):
        assert transceiver._recv_reqs == {rid: owner}
        assert transceiver._recv_sessions == {rid: session}
        return object()

    transceiver._transfer_worker = Mock()
    transceiver._transfer_worker.create_rx_session.side_effect = create_rx_session
    transceiver._create_kv_slice = Mock(side_effect=create_kv_slice)

    with pytest.raises(RuntimeError, match="publication failed"):
        transceiver.request_and_receive_async(req)

    assert transceiver._recv_reqs == {rid: req}
    assert transceiver._recv_sessions == {rid: session}


def test_async_send_rejects_a_distinct_owner_for_a_live_request_id() -> None:
    rid = 20
    original = _FakeRequest(request_id=rid)
    replacement = _FakeRequest(request_id=rid)
    session = _FakeSession(
        rid=rid,
        wait_result=None,
        has_transferring_tasks=True,
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._shutdown_started = False
    transceiver._send_sessions = {rid: session}
    transceiver._send_reqs = {rid: original}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._ever_had_send_session = True

    with pytest.raises(RuntimeError, match="different live source owner"):
        transceiver.respond_and_send_async(replacement)

    assert transceiver._send_reqs == {rid: original}
    assert transceiver._send_sessions == {rid: session}


def test_context_status_poll_retains_owner_and_blocks_replacement() -> None:
    rid = 21
    original = _FakeRequest(request_id=rid)
    replacement = _FakeRequest(request_id=rid)
    wait_started = threading.Event()
    release_wait = threading.Event()
    errors: list[BaseException] = []
    results: list[tuple[list[int], list[int]]] = []

    original_session = Mock()
    original_session.status = SessionStatus.READY
    original_session.is_completed.return_value = False
    original_session.has_failed.side_effect = release_wait.is_set
    original_session.has_transferring_tasks.return_value = False

    def wait_complete(*, blocking: bool) -> WaitResult:
        assert not blocking
        wait_started.set()
        assert release_wait.wait(timeout=2)
        return WaitResult.FAILED

    original_session.wait_complete.side_effect = wait_complete
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown_started = False
    transceiver._send_sessions = {rid: original_session}
    transceiver._send_reqs = {rid: original}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._wait_reqs = {}
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._ctx_consensus = lambda local_ids: list(local_ids)
    transceiver._ctx_consensus_outcome = (
        lambda _to_process, cancelled, failed, completed, timed_out, _cleanup_ready: (
            cancelled,
            failed,
            completed,
            timed_out,
        )
    )

    def poll_status() -> None:
        try:
            results.append(transceiver.check_context_transfer_status(at_least_request_num=1))
        except BaseException as error:
            errors.append(error)

    poll_thread = threading.Thread(target=poll_status)
    poll_thread.start()
    assert wait_started.wait(timeout=1)
    try:
        assert transceiver.cancel_request(original) is False
        assert transceiver._send_sessions == {rid: original_session}
        assert transceiver._send_reqs == {rid: original}
        with pytest.raises(RuntimeError, match="different live source owner"):
            transceiver.respond_and_send_async(replacement)
    finally:
        release_wait.set()
        poll_thread.join(timeout=2)

    assert not poll_thread.is_alive()
    assert errors == []
    assert results == [([], [rid])]
    assert transceiver._send_sessions == {}
    assert transceiver._send_reqs == {}
    original_session.cancel.assert_called_once_with()
    original_session.close.assert_called_once_with()


def test_gen_status_poll_retains_owner_and_blocks_replacement() -> None:
    rid = 23
    original = _FakeRequest(request_id=rid)
    replacement = _FakeRequest(request_id=rid)
    wait_started = threading.Event()
    release_wait = threading.Event()
    errors: list[BaseException] = []
    results: list[tuple[list[int], list[int], list[_FakeRequest]]] = []

    original_session = Mock()
    original_session.status = SessionStatus.READY
    original_session.is_completed.return_value = False
    original_session.has_failed.side_effect = release_wait.is_set
    original_session.has_transferring_tasks.return_value = False

    def wait_complete(*, blocking: bool) -> WaitResult:
        assert not blocking
        wait_started.set()
        assert release_wait.wait(timeout=2)
        return WaitResult.FAILED

    original_session.wait_complete.side_effect = wait_complete
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown_started = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {rid: original_session}
    transceiver._recv_reqs = {rid: original}
    transceiver._wait_reqs = {}
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver.kv_transfer_poll_interval_ms = 1000
    transceiver._gen_consensus = lambda local_ids: list(local_ids)
    transceiver._gen_consensus_outcome = (
        lambda _to_process, cancelled, failed, completed, _cleanup_ready: (
            cancelled,
            failed,
            completed,
        )
    )

    def poll_status() -> None:
        try:
            results.append(transceiver.check_gen_transfer_status(at_least_request_num=1))
        except BaseException as error:
            errors.append(error)

    poll_thread = threading.Thread(target=poll_status)
    poll_thread.start()
    assert wait_started.wait(timeout=1)
    try:
        assert transceiver.cancel_request(original) is False
        assert transceiver._recv_sessions == {rid: original_session}
        assert transceiver._recv_reqs == {rid: original}
        with pytest.raises(RuntimeError, match="different live destination owner"):
            transceiver.request_and_receive_async(replacement)
    finally:
        release_wait.set()
        poll_thread.join(timeout=2)

    assert not poll_thread.is_alive()
    assert errors == []
    assert results == [([], [rid], [])]
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}
    original_session.cancel.assert_called_once_with()
    original_session.close.assert_called_once_with()


def test_cancel_request_retains_receive_session_until_writers_drain() -> None:
    rid = 15
    session = _FakeSession(
        rid=rid,
        wait_result=None,
        has_transferring_tasks=True,
    )
    req = _FakeRequest(request_id=rid)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: req}

    assert not transceiver.cancel_request(req)
    assert session.cancel_calls == 1
    assert not session.closed
    assert transceiver._recv_sessions == {rid: session}
    assert transceiver._recv_reqs == {rid: req}

    session._has_transferring_tasks = False

    assert transceiver.cancel_request(req)
    assert session.cancel_calls == 2
    assert session.closed
    assert rid not in transceiver._recv_sessions
    assert rid not in transceiver._recv_reqs


def test_cancel_request_continues_with_receive_after_send_cleanup_error() -> None:
    rid = 25
    req = _FakeRequest(request_id=rid)
    send_session = Mock()
    send_session.cancel.side_effect = [RuntimeError("send cancel failed"), None]
    send_session.has_transferring_tasks.return_value = False
    recv_session = Mock()
    recv_session.has_transferring_tasks.return_value = False
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {rid: send_session}
    transceiver._send_reqs = {rid: req}
    transceiver._recv_sessions = {rid: recv_session}
    transceiver._recv_reqs = {rid: req}

    assert not transceiver.cancel_request(req)

    assert transceiver._send_sessions == {rid: send_session}
    assert transceiver._send_reqs == {rid: req}
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}
    recv_session.cancel.assert_called_once_with()
    recv_session.close.assert_called_once_with()

    assert transceiver.cancel_request(req)
    assert transceiver._send_sessions == {}
    assert transceiver._send_reqs == {}
    assert send_session.cancel.call_count == 2
    send_session.close.assert_called_once_with()


def test_prepare_context_requests_rejects_admission_after_shutdown() -> None:
    req = _FakeRequest(request_id=26)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown_started = True
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._wait_reqs = {}
    transceiver._transfer_worker = Mock()

    with pytest.raises(RuntimeError, match="shutting down"):
        transceiver.prepare_context_requests([req])

    assert transceiver._wait_reqs == {}
    transceiver._transfer_worker.has_all_peer_req_infos_for_send.assert_not_called()


def test_prepare_context_requests_rejects_replacement_wait_owner() -> None:
    rid = 27
    original = _FakeRequest(request_id=rid)
    replacement = _FakeRequest(request_id=rid)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown_started = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._wait_reqs = {rid: original}
    transceiver._transfer_worker = Mock()

    with pytest.raises(RuntimeError, match="different live source owner"):
        transceiver.prepare_context_requests([replacement])

    assert transceiver._wait_reqs == {rid: original}


def test_prepare_context_requests_does_not_partially_enroll_on_owner_collision() -> None:
    existing_rid = 28
    new_req = _FakeRequest(request_id=29)
    original = _FakeRequest(request_id=existing_rid)
    replacement = _FakeRequest(request_id=existing_rid)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown_started = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._wait_reqs = {existing_rid: original}
    transceiver._transfer_worker = Mock()

    with pytest.raises(RuntimeError, match="different live source owner"):
        transceiver.prepare_context_requests([new_req, replacement])

    assert transceiver._wait_reqs == {existing_rid: original}
    assert new_req.state is None


def test_generation_timeout_keeps_request_active_until_cancel_drains() -> None:
    request = Mock()
    request.py_request_id = 18
    request.is_attention_dp_dummy = False
    request.py_kv_transfer_timed_out = True
    executor = object.__new__(PyExecutor)
    executor.active_requests = [request]
    executor.perf_manager = Mock()
    executor.perf_manager.get_timestamp.return_value = 0.0
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.cancel_request.return_value = False
    executor._enqueue_responses = Mock()
    executor.enable_attention_dp = False
    executor.dist = Mock(world_size=1)

    finished = executor._handle_responses()

    assert finished == []
    assert executor.active_requests == [request]
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)


def test_consensus_outcome_uses_single_batched_allgather() -> None:
    # Outcomes and physical cleanup readiness are exchanged with ONE allgather
    # (packed as a list-of-lists) instead of three; verify a single call and that
    # union (cancelled/failed) + intersection (completed/readiness) semantics are preserved.
    transceiver = object.__new__(KvCacheTransceiverV2)
    calls: list = []

    def fake_allgather(payload):
        calls.append(payload)
        # rank0 = this rank's [cancelled, failed, completed, cleanup_ready];
        # rank1 = a peer rank.
        return [payload, [[], [99], [7, 8], [1, 2, 7, 8, 99]]]

    to_process = [1, 2, 7, 8, 99]
    new_cancelled, new_failed, new_completed = transceiver._consensus_outcome(
        to_process,
        [1],
        [2],
        [7],
        [1, 2, 7, 8, 99],
        fake_allgather,
        True,
    )

    assert len(calls) == 1  # batched: a single allgather, not three
    assert calls[0] == [[1], [2], [7], [1, 2, 7, 8, 99]]
    assert new_cancelled == [1]  # union of cancelled across ranks
    assert new_failed == [2, 99]  # union of failed across ranks
    assert new_completed == [7]  # intersection only (8 is completed on the peer only)


def test_consensus_retains_remote_failure_until_every_rank_is_cleanup_ready() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    rid = 21

    def peer_failed(payload):
        return [payload, [[], [rid], [], [rid]]]

    cancelled, failed, completed = transceiver._consensus_outcome(
        [rid], [], [], [], [], peer_failed, True
    )
    assert (cancelled, failed, completed) == ([], [], [])

    cancelled, failed, completed = transceiver._consensus_outcome(
        [rid], [], [], [], [rid], peer_failed, True
    )
    assert (cancelled, failed, completed) == ([], [rid], [])


def test_tx_session_wait_complete_defaults_to_blocking() -> None:
    task = _FakeTask(TaskStatus.INIT, wait_result=False)
    session = _make_tx_session([task])

    assert session.wait_complete() == WaitResult.TIMEOUT
    assert task.wait_calls == [0.25]


def test_tx_session_wait_complete_nonblocking_returns_none_without_waiting() -> None:
    task = _FakeTask(TaskStatus.TRANSFERRING)
    session = _make_tx_session([task])

    assert session.wait_complete(blocking=False) is None
    assert task.wait_calls == []


def test_tx_session_wait_complete_nonblocking_reports_later_task_error() -> None:
    pending_task = _FakeTask(TaskStatus.TRANSFERRING)
    failed_task = _FakeTask(TaskStatus.ERROR)
    session = _make_tx_session([pending_task, failed_task])

    assert session.wait_complete(blocking=False) is None
    pending_task.status = TaskStatus.TRANSFERRED
    assert session.wait_complete(blocking=False) == WaitResult.FAILED
    assert pending_task.wait_calls == []
    assert failed_task.wait_calls == []


def test_tx_session_has_failed_reports_task_error() -> None:
    task = _FakeTask(TaskStatus.ERROR)
    session = _make_tx_session([task])

    assert session.has_failed()


def test_check_context_runs_consensus_after_a_send() -> None:
    # Once the worker has sent, the ctx consensus runs as usual.
    transceiver = _make_transceiver({})
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_consensus = Mock(return_value=[])
    transceiver._ctx_consensus_outcome = Mock(return_value=([], [], [], []))

    transceiver.check_context_transfer_status(0)
    transceiver._ctx_consensus.assert_called_once()
    transceiver._ctx_consensus_outcome.assert_called_once_with([], [], [], [], [], [])


def test_prepare_context_requests_skips_consensus_when_nothing_waiting() -> None:
    # With nothing waiting on any rank, prepare_context_requests returns before the consensus; the
    # waiting set is the same on every rank.
    transceiver = _make_transceiver({})
    transceiver._wait_reqs = {}
    transceiver._ctx_consensus = Mock(side_effect=AssertionError("consensus must be skipped"))

    transceiver.prepare_context_requests([])
    transceiver._ctx_consensus.assert_not_called()


def test_sender_shutdown_retains_remote_agents_while_worker_is_alive() -> None:
    sender = object.__new__(Sender)
    worker = _FakeWorkerThread(alive=True)
    dealer = Mock()
    sender._shutdown = False
    sender._shutdown_started = False
    sender._shutdown_complete = False
    sender._shutdown_sentinels_sent = False
    sender._messenger = Mock()
    sender._send_task_queues = [Mock()]
    sender._worker_threads = [worker]
    sender._loaded_remote_agents = {"peer-agent"}
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._agent = Mock()
    sender._dealers = {"peer": dealer}

    assert sender.shutdown() is False

    assert sender._shutdown_complete is False
    sender._agent.invalidate_remote_agent.assert_not_called()
    dealer.stop.assert_not_called()
    assert sender._loaded_remote_agents == {"peer-agent"}

    worker.alive = False

    assert sender.shutdown() is True
    assert sender._shutdown is True
    assert sender._shutdown_complete is True
    sender._agent.invalidate_remote_agent.assert_called_once_with("peer-agent")
    dealer.stop.assert_called_once_with()


def test_transfer_worker_does_not_treat_local_agent_shutdown_as_remote_quiescence() -> None:
    worker = object.__new__(TransferWorker)
    worker._shutdown = False
    worker._shutdown_started = False
    worker._shutdown_complete = False
    worker._rank_info_server = Mock()
    worker._sender = Mock()
    worker._sender.shutdown.return_value = True
    worker._receiver = Mock()
    worker._receiver.transfers_drained = False
    worker._agent = Mock()
    worker._bounce = Mock()
    worker._registered_mem = [Mock()]

    assert worker.shutdown() is False

    assert worker._shutdown is False
    worker._receiver.begin_shutdown.assert_called_once_with()
    worker._agent.shutdown.assert_not_called()
    worker._agent.deregister_memory.assert_not_called()
    worker._bounce.close.assert_not_called()
    worker._receiver.shutdown.assert_not_called()


def _make_receiver_with_session(session: RxSession) -> Receiver:
    receiver = object.__new__(Receiver)
    receiver._shutdown = True  # Keep object.__new__ fixtures out of __del__ cleanup.
    receiver._recv_registry = Mock()
    receiver._recv_registry.is_drained.return_value = True
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {41: weakref.ref(session)}
    return receiver


def test_receiver_keeps_session_owner_alive_until_explicit_retirement() -> None:
    receiver = object.__new__(Receiver)
    receiver._shutdown = True  # Keep object.__new__ fixtures out of __del__ cleanup.
    receiver._sessions = {}
    receiver._sessions_lock = threading.Lock()
    receiver._pre_cancelled_rids = set()
    session = _FakeSession(rid=40, wait_result=None)

    receiver.setup_session(session)
    session_ref = weakref.ref(session)
    del session
    gc.collect()

    assert session_ref() is not None
    assert receiver._sessions[40] is session_ref()


def _make_receive_ownership_session(
    task: _FakeReceiveTaskOwnership,
    *,
    need_aux: bool,
    aux_drained: bool,
) -> RxSession:
    session = object.__new__(RxSession)
    session._closed = True  # Keep object.__new__ test fixtures out of __del__ cleanup.
    session.lock = threading.Lock()
    session._kv_tasks = [task]
    session._need_aux = need_aux
    session._aux_status = TaskStatus.TRANSFERRED if aux_drained else TaskStatus.TRANSFERRING
    session._aux_drained = aux_drained
    return session


def test_receiver_not_drained_while_legacy_receive_is_active() -> None:
    session = _make_receive_ownership_session(
        _FakeReceiveTaskOwnership(
            lifecycle_managed=False,
            status=TaskStatus.TRANSFERRING,
        ),
        need_aux=False,
        aux_drained=True,
    )
    receiver = _make_receiver_with_session(session)

    assert receiver.transfers_drained is False


def test_receiver_not_drained_after_published_legacy_failure() -> None:
    session = _make_receive_ownership_session(
        _FakeReceiveTaskOwnership(
            lifecycle_managed=False,
            status=TaskStatus.ERROR,
        ),
        need_aux=False,
        aux_drained=True,
    )
    receiver = _make_receiver_with_session(session)

    assert receiver.transfers_drained is False


def test_receiver_not_drained_while_aux_receive_is_active() -> None:
    session = _make_receive_ownership_session(
        _FakeReceiveTaskOwnership(
            lifecycle_managed=True,
            status=TaskStatus.TRANSFERRED,
        ),
        need_aux=True,
        aux_drained=False,
    )
    receiver = _make_receiver_with_session(session)

    assert receiver.transfers_drained is False


def test_receiver_not_drained_after_nonterminal_aux_failure() -> None:
    session = _make_receive_ownership_session(
        _FakeReceiveTaskOwnership(
            lifecycle_managed=True,
            status=TaskStatus.TRANSFERRED,
        ),
        need_aux=True,
        aux_drained=False,
    )
    session._aux_status = TaskStatus.ERROR
    receiver = _make_receiver_with_session(session)

    assert receiver.transfers_drained is False


def test_transceiver_shutdown_retains_sessions_until_worker_drains() -> None:
    send_session = _FakeSession(rid=51, wait_result=None)
    recv_session = _FakeSession(rid=52, wait_result=None)
    send_req = _FakeRequest(request_id=51)
    recv_req = _FakeRequest(request_id=52)
    wait_req = _FakeRequest(request_id=53)
    worker = _FakeTransferWorker(shutdown_results=[False, True])
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._shutdown = False
    transceiver._shutdown_started = False
    transceiver._transfer_worker = worker
    transceiver._send_sessions = {51: send_session}
    transceiver._send_reqs = {51: send_req}
    transceiver._recv_sessions = {52: recv_session}
    transceiver._recv_reqs = {52: recv_req}
    transceiver._wait_reqs = {53: wait_req}

    assert transceiver.shutdown() is False
    assert transceiver._shutdown is False
    assert worker.shutdown_count == 1
    assert not send_session.closed
    assert not recv_session.closed
    assert transceiver._send_sessions == {51: send_session}
    assert transceiver._send_reqs == {51: send_req}
    assert transceiver._recv_sessions == {52: recv_session}
    assert transceiver._recv_reqs == {52: recv_req}
    assert transceiver._wait_reqs == {53: wait_req}

    assert transceiver.shutdown() is True
    assert transceiver._shutdown is True
    assert worker.shutdown_count == 2
    assert send_session.closed
    assert recv_session.closed
    assert transceiver._send_sessions == {}
    assert transceiver._send_reqs == {}
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}
    assert transceiver._wait_reqs == {}


@pytest.mark.parametrize("failure_stage", ["cancel", "worker", "close"])
def test_transceiver_shutdown_retains_owner_when_cleanup_raises(failure_stage: str) -> None:
    send_session = Mock()
    recv_session = Mock()
    worker = Mock()
    worker.shutdown.return_value = True
    if failure_stage == "cancel":
        send_session.cancel.side_effect = [RuntimeError("cancel failed"), None]
    elif failure_stage == "worker":
        worker.shutdown.side_effect = [RuntimeError("worker failed"), True]
    else:
        send_session.close.side_effect = [RuntimeError("close failed"), None]

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._shutdown = False
    transceiver._shutdown_started = False
    transceiver._transfer_worker = worker
    transceiver._send_sessions = {51: send_session}
    transceiver._send_reqs = {51: _FakeRequest(request_id=51)}
    transceiver._recv_sessions = {52: recv_session}
    transceiver._recv_reqs = {52: _FakeRequest(request_id=52)}
    wait_req = _FakeRequest(request_id=53)
    transceiver._wait_reqs = {53: wait_req}

    try:
        assert transceiver.shutdown() is False
        assert transceiver in _NON_DRAINED_TRANSCEIVERS
        assert transceiver._send_sessions
        if failure_stage == "worker":
            assert transceiver._recv_sessions
            recv_session.close.assert_not_called()
        else:
            assert transceiver._recv_sessions == {}
            recv_session.cancel.assert_called_once_with()
            recv_session.close.assert_called_once_with()
        assert transceiver._wait_reqs == {53: wait_req}

        assert transceiver.shutdown() is True
        assert transceiver not in _NON_DRAINED_TRANSCEIVERS
        assert transceiver._send_sessions == {}
        assert transceiver._recv_sessions == {}
        assert transceiver._wait_reqs == {}
    finally:
        _NON_DRAINED_TRANSCEIVERS.discard(transceiver)


def test_shutdown_does_not_drain_provisional_owner_during_active_status_call() -> None:
    rid = 78
    request = _RetirementRequest(rid)
    session = _FakeSession(
        rid=rid,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )
    transceiver = _make_transceiver({rid: session}, {rid: request})
    transceiver._shutdown = False
    transceiver._shutdown_started = False
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._wait_reqs = {}

    enrolled = threading.Event()
    release_prepare = threading.Event()
    original_prepare = transceiver._prepare_context_retirements
    prepare_calls = 0

    def pause_after_enrollment() -> None:
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls == 2:
            enrolled.set()
            assert release_prepare.wait(timeout=2)
        original_prepare()

    transceiver._prepare_context_retirements = pause_after_enrollment
    status_results: list[tuple[list[int], list[int]]] = []
    errors: list[BaseException] = []

    def poll_status() -> None:
        try:
            status_results.append(transceiver.check_context_transfer_status(0, mark_complete=True))
        except BaseException as error:
            errors.append(error)

    poll_thread = threading.Thread(target=poll_status)
    poll_thread.start()
    assert enrolled.wait(timeout=1)
    try:
        assert transceiver.shutdown() is False
        assert set(transceiver._get_context_retirements()) == {rid}
        assert transceiver._send_sessions == {rid: session}
        assert transceiver._transfer_worker.shutdown_count == 0
    finally:
        release_prepare.set()
        poll_thread.join(timeout=2)

    assert not poll_thread.is_alive()
    assert errors == []
    assert status_results == [([rid], [])]
    assert transceiver.shutdown() is True
    assert transceiver._transfer_worker.shutdown_count == 1
    _NON_DRAINED_TRANSCEIVERS.discard(transceiver)


def test_blocking_generation_poll_shutdown_cancels_without_deadlock() -> None:
    rid = 79
    request = _RetirementRequest(rid)
    wait_started = threading.Event()
    cancel_seen = threading.Event()
    session = _FakeSession(rid=rid, wait_result=None)

    def wait_complete(*, blocking: bool) -> WaitResult:
        assert blocking
        wait_started.set()
        assert cancel_seen.wait(timeout=2)
        return WaitResult.FAILED

    def cancel_session() -> None:
        session.cancel_calls += 1
        session._status = SessionStatus.CANCELLED
        cancel_seen.set()

    session.wait_complete = wait_complete
    session.cancel = cancel_session
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown = False
    transceiver._shutdown_started = False
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver.kv_transfer_poll_interval_ms = 0
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: request}
    transceiver._wait_reqs = {}
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._gen_consensus = lambda local_ids: list(local_ids)
    transceiver._gen_consensus_outcome = (
        lambda _to_process, cancelled, failed, completed, cleanup_ready: (
            [candidate for candidate in cancelled if candidate in cleanup_ready],
            [candidate for candidate in failed if candidate in cleanup_ready],
            [candidate for candidate in completed if candidate in cleanup_ready],
        )
    )

    status_results: list[tuple[list[int], list[int], list[_RetirementRequest]]] = []
    errors: list[BaseException] = []

    def poll_status() -> None:
        try:
            status_results.append(transceiver.check_gen_transfer_status(None))
        except BaseException as error:
            errors.append(error)

    poll_thread = threading.Thread(target=poll_status)
    poll_thread.start()
    assert wait_started.wait(timeout=1)
    assert transceiver.shutdown() is False
    poll_thread.join(timeout=2)

    assert not poll_thread.is_alive()
    assert errors == []
    assert status_results == [([], [], [request])]
    assert transceiver._transfer_worker.shutdown_count == 0
    assert transceiver.shutdown() is True
    assert transceiver._transfer_worker.shutdown_count == 1
    _NON_DRAINED_TRANSCEIVERS.discard(transceiver)


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_transceiver_shutdown_waits_for_async_enrollment_and_launch(direction) -> None:
    rid = 61 if direction == "send" else 62
    req = _FakeRequest(request_id=rid)
    launch_started = threading.Event()
    release_launch = threading.Event()
    shutdown_attempted = threading.Event()
    shutdown_called = threading.Event()
    errors: list[BaseException] = []
    session = Mock()
    session.has_transferring_tasks.return_value = False

    def block_launch(_slice) -> None:
        launch_started.set()
        assert release_launch.wait(timeout=2)

    if direction == "send":
        session.send.side_effect = block_launch
    else:
        session.receive.side_effect = block_launch

    worker = Mock()

    def shutdown_worker() -> bool:
        shutdown_called.set()
        return True

    worker.shutdown.side_effect = shutdown_worker
    worker.create_rx_session.return_value = session
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown_started = False
    transceiver._shutdown = False
    transceiver._transfer_worker = worker
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._ever_had_send_session = False
    transceiver._ever_had_recv_session = False
    transceiver._create_kv_slice = Mock(return_value=object())
    transceiver._slice_num_bytes = Mock(return_value=8)
    transceiver._kv_size_rank_factor = 1
    transceiver._finalize_send = Mock()

    if direction == "send":

        def get_send_session(_req):
            transceiver._send_sessions[rid] = session
            return session

        transceiver._get_or_create_send_session = get_send_session
        launch = transceiver.respond_and_send_async
    else:
        launch = transceiver.request_and_receive_async

    def run_launch() -> None:
        try:
            launch(req)
        except BaseException as error:
            errors.append(error)

    def run_shutdown() -> None:
        try:
            shutdown_attempted.set()
            assert transceiver.shutdown()
        except BaseException as error:
            errors.append(error)

    launch_thread = threading.Thread(target=run_launch)
    shutdown_thread = threading.Thread(target=run_shutdown)
    launch_thread.start()
    assert launch_started.wait(timeout=1)
    shutdown_thread.start()
    try:
        assert shutdown_attempted.wait(timeout=1)
        assert not shutdown_called.wait(timeout=0.1)
    finally:
        release_launch.set()
        launch_thread.join(timeout=2)
        shutdown_thread.join(timeout=2)

    assert not launch_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert errors == []
    assert shutdown_called.is_set()
    assert transceiver._shutdown


def test_sync_receive_releases_admission_gate_while_waiting_for_shutdown() -> None:
    rid = 63
    req = _FakeRequest(request_id=rid)
    wait_started = threading.Event()
    cancel_seen = threading.Event()
    shutdown_finished = threading.Event()
    errors: list[BaseException] = []
    shutdown_results: list[bool] = []

    session = Mock()
    session.status = SessionStatus.CANCELLED
    session.has_transferring_tasks.return_value = False

    def wait_complete(*, blocking: bool) -> WaitResult:
        assert blocking
        wait_started.set()
        assert cancel_seen.wait(timeout=2)
        return WaitResult.FAILED

    def cancel_session() -> None:
        cancel_seen.set()

    session.wait_complete.side_effect = wait_complete
    session.cancel.side_effect = cancel_session

    worker = Mock()
    worker.create_rx_session.return_value = session
    worker.shutdown.return_value = True

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown_started = False
    transceiver._shutdown = False
    transceiver._transfer_worker = worker
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._create_kv_slice = Mock(return_value=object())

    def run_receive() -> None:
        try:
            transceiver.request_and_receive_sync(req)
        except BaseException as error:
            errors.append(error)

    def run_shutdown() -> None:
        try:
            shutdown_results.append(transceiver.shutdown())
        except BaseException as error:
            errors.append(error)
        finally:
            shutdown_finished.set()

    receive_thread = threading.Thread(target=run_receive)
    shutdown_thread = threading.Thread(target=run_shutdown)
    receive_thread.start()
    assert wait_started.wait(timeout=1)
    shutdown_thread.start()
    try:
        # shutdown() is the only path that signals cancel_seen. If the sync
        # receive still held the admission gate while waiting, neither thread
        # could make progress here.
        assert shutdown_finished.wait(timeout=1)
    finally:
        cancel_seen.set()
        receive_thread.join(timeout=2)
        shutdown_thread.join(timeout=2)

    assert not receive_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert errors == []
    assert shutdown_results == [True]
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}
    session.close.assert_called_once_with()


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_cancel_request_waits_for_async_enrollment_and_launch(direction) -> None:
    rid = 64 if direction == "send" else 65
    req = _FakeRequest(request_id=rid)
    launch_started = threading.Event()
    release_launch = threading.Event()
    cancel_attempted = threading.Event()
    cancel_called = threading.Event()
    errors: list[BaseException] = []
    cancel_results: list[bool] = []

    session = Mock()
    session.has_transferring_tasks.return_value = False
    session.cancel.side_effect = cancel_called.set

    def block_launch(_slice) -> None:
        launch_started.set()
        assert release_launch.wait(timeout=2)

    if direction == "send":
        session.send.side_effect = block_launch
    else:
        session.receive.side_effect = block_launch

    worker = Mock()
    worker.create_rx_session.return_value = session
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown_started = False
    transceiver._shutdown = False
    transceiver._transfer_worker = worker
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._wait_reqs = {}
    transceiver._ever_had_send_session = False
    transceiver._ever_had_recv_session = False
    transceiver._create_kv_slice = Mock(return_value=object())
    transceiver._slice_num_bytes = Mock(return_value=8)
    transceiver._kv_size_rank_factor = 1
    transceiver._finalize_send = Mock()

    if direction == "send":

        def get_send_session(_req):
            transceiver._send_sessions[rid] = session
            return session

        transceiver._get_or_create_send_session = get_send_session
        launch = transceiver.respond_and_send_async
    else:
        launch = transceiver.request_and_receive_async

    def run_launch() -> None:
        try:
            launch(req)
        except BaseException as error:
            errors.append(error)

    def run_cancel() -> None:
        try:
            cancel_attempted.set()
            cancel_results.append(transceiver.cancel_request(req))
        except BaseException as error:
            errors.append(error)

    launch_thread = threading.Thread(target=run_launch)
    cancel_thread = threading.Thread(target=run_cancel)
    launch_thread.start()
    assert launch_started.wait(timeout=1)
    cancel_thread.start()
    try:
        assert cancel_attempted.wait(timeout=1)
        assert not cancel_called.wait(timeout=0.1)
    finally:
        release_launch.set()
        launch_thread.join(timeout=2)
        cancel_thread.join(timeout=2)

    assert not launch_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert errors == []
    assert cancel_results == [True]
    assert cancel_called.is_set()
    assert transceiver._send_sessions == {}
    assert transceiver._send_reqs == {}
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}


def test_transceiver_context_manager_surfaces_non_drained_shutdown() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver.shutdown = Mock(return_value=False)

    with pytest.raises(RuntimeError, match="shutdown did not drain"):
        transceiver.__exit__(None, None, None)


def test_transceiver_context_manager_does_not_mask_body_exception_on_shutdown_failure() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver.shutdown = Mock(return_value=False)

    assert transceiver.__exit__(ValueError, ValueError("body failed"), None) is False
