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
"""Bounded polling tests for KvCacheTransceiverV2 Tx sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import Mock

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.transfer import TaskStatus, TxSession
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState


@dataclass
class _FakeRequest:
    state: Optional[LlmRequestState] = None


class _FakeTransferWorker:
    def __init__(self) -> None:
        self.sweep_count = 0

    def sweep_stale_req_infos(self) -> None:
        self.sweep_count += 1


class _FakeSession:
    def __init__(
        self,
        rid: int,
        wait_result: Optional[WaitResult],
        *,
        status: SessionStatus = SessionStatus.READY,
        is_completed: bool = False,
        has_failed: bool = False,
    ) -> None:
        self._rid = rid
        self._wait_result = wait_result
        self._status = status
        self._is_completed = is_completed
        self._has_failed = has_failed
        self.blocking_calls: list[bool] = []
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

    def close(self) -> None:
        self.closed = True


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
        lambda _to_process, cancelled, failed, completed, timed_out: (
            cancelled,
            failed,
            completed,
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


def test_context_transfer_status_enters_consensus_when_tp_sync_required() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = False
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_need_pp_sync = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._ctx_consensus = Mock(return_value=[])
    transceiver._build_to_process = Mock(return_value=[])
    transceiver._ctx_consensus_outcome = Mock(return_value=([], [], [], []))
    transceiver._close_failed_sessions = Mock()

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    transceiver._ctx_consensus.assert_called_once_with([])
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
    transceiver._close_failed_sessions = Mock()

    completed, failed, cancelled = transceiver.check_gen_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    assert cancelled == []
    transceiver._gen_consensus.assert_called_once_with([])


def test_consensus_outcome_uses_single_batched_allgather() -> None:
    # The cancelled/failed/completed id lists are exchanged with ONE allgather
    # (packed as a list-of-lists) instead of three; verify a single call and that
    # union (cancelled/failed) + intersection (completed) semantics are preserved.
    transceiver = object.__new__(KvCacheTransceiverV2)
    calls: list = []

    def fake_allgather(payload):
        calls.append(payload)
        # rank0 = this rank's [cancelled, failed, completed]; rank1 = a peer rank.
        return [payload, [[], [99], [7, 8]]]

    to_process = [1, 2, 7, 8, 99]
    new_cancelled, new_failed, new_completed = transceiver._consensus_outcome(
        to_process, [1], [2], [7], fake_allgather, True
    )

    assert len(calls) == 1  # batched: a single allgather, not three
    assert calls[0] == [[1], [2], [7]]
    assert new_cancelled == [1]  # union of cancelled across ranks
    assert new_failed == [2, 99]  # union of failed across ranks
    assert new_completed == [7]  # intersection only (8 is completed on the peer only)


def test_ctx_consensus_fastpath_skips_when_idle(monkeypatch) -> None:
    # With the fast-path enabled, an all-zero terminal count (one fixed-size
    # allreduce) makes every rank skip the variable-length consensus; a non-zero
    # count falls through to the normal consensus path.
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver._CTX_CONSENSUS_FASTPATH", True
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_need_pp_sync = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._dist = Mock()
    transceiver._dist.allreduce = Mock(return_value=0)
    transceiver._ctx_consensus = Mock(return_value=[])
    transceiver._build_to_process = Mock(return_value=[])
    transceiver._ctx_consensus_outcome = Mock(return_value=([], [], [], []))
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._close_failed_sessions = Mock()

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=0)

    assert completed == [] and failed == []
    transceiver._dist.allreduce.assert_called_once()
    transceiver._ctx_consensus.assert_not_called()  # idle fast-path skipped the consensus

    # Non-zero global terminal count => fast-path does not skip; consensus runs.
    transceiver._dist.allreduce = Mock(return_value=2)
    transceiver.check_context_transfer_status(at_least_request_num=0)
    transceiver._ctx_consensus.assert_called_once()


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

    assert session.wait_complete(blocking=False) == WaitResult.FAILED
    assert pending_task.wait_calls == []
    assert failed_task.wait_calls == []


def test_tx_session_has_failed_reports_task_error() -> None:
    task = _FakeTask(TaskStatus.ERROR)
    session = _make_tx_session([task])

    assert session.has_failed()
