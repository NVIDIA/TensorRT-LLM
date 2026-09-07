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

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import (
    CtxTransferStatus,
    GenTransferStatus,
)
from tensorrt_llm._torch.disaggregation.native.transfer import (
    TaskStatus,
    TransferWorker,
    TransferWorkerConfig,
    TxSession,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState


@dataclass
class _FakeRequest:
    state: Optional[LlmRequestState] = None
    py_kv_send_session_retired: bool = False


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
        has_transferring_tasks: bool = False,
    ) -> None:
        self._rid = rid
        self._wait_result = wait_result
        self._status = status
        self._is_completed = is_completed
        self._has_failed = has_failed
        self._has_transferring_tasks = has_transferring_tasks
        self.blocking_calls: list[bool] = []
        self.closed = False
        self.aux_slot: Optional[int] = 0

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

    def has_transferring_tasks(self) -> bool:
        return self._has_transferring_tasks

    def close(self) -> None:
        self.closed = True
        self.aux_slot = None


class _FakeTask:
    def __init__(
        self,
        status: TaskStatus,
        wait_result: bool | list[bool] = True,
        on_wait: Optional[Callable[[Optional[float]], None]] = None,
    ) -> None:
        self.status = status
        self._wait_results = list(wait_result) if isinstance(wait_result, list) else [wait_result]
        self._on_wait = on_wait
        self.wait_calls: list[Optional[float]] = []

    def wait(self, timeout: Optional[float] = None) -> bool:
        self.wait_calls.append(timeout)
        if self._on_wait is not None:
            self._on_wait(timeout)
        result = self._wait_results.pop(0) if len(self._wait_results) > 1 else self._wait_results[0]
        if result and self.status != TaskStatus.ERROR:
            self.status = TaskStatus.TRANSFERRED
        return result


class _FakeClock:
    def __init__(self, now_s: float = 0.0) -> None:
        self.now_s = now_s

    def monotonic(self) -> float:
        return self.now_s

    def advance(self, elapsed_s: Optional[float]) -> None:
        assert elapsed_s is not None
        self.now_s += elapsed_s


def _make_transceiver(
    sessions: dict[int, _FakeSession],
    reqs: Optional[dict[int, _FakeRequest]] = None,
) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._send_sessions = sessions
    transceiver._send_reqs = reqs or {rid: _FakeRequest() for rid in sessions}
    transceiver._sender_future_timeout_ms = 123
    transceiver.kv_transfer_timeout_ms = 60_000
    # Attributes read by check_context_transfer_status before it processes sessions.
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._ctx_consensus = lambda local_ids: list(local_ids)
    transceiver._ctx_consensus_outcome = (
        lambda _to_process, cancelled, failed, completed, quiesced: (
            cancelled,
            failed,
            completed,
            quiesced,
        )
    )
    return transceiver


def _make_tx_session(
    kv_tasks: list[_FakeTask],
    *,
    need_aux: bool = False,
    aux_task: Optional[_FakeTask] = None,
    timeout_s: Optional[float] = 0.25,
    deadline_monotonic_s: Optional[float] = None,
) -> TxSession:
    session = object.__new__(TxSession)
    session._timeout_s = timeout_s
    session._overall_timeout_s = None
    session._deadline_monotonic_s = deadline_monotonic_s
    session._need_aux = need_aux
    session._terminal_status = None
    session._exception = None
    session.receiver_ready = True
    session.kv_tasks = kv_tasks
    session.aux_task = aux_task
    session._has_last_slice = True
    session.lock = threading.Lock()
    session._closed = False
    session._aux_buffer = None
    session.aux_slot = None
    session._sender = None
    return session


def test_context_transfer_status_bounded_poll_keeps_not_ready_session_queued(
    monkeypatch,
) -> None:
    session = _FakeSession(rid=11, wait_result=None)
    transceiver = _make_transceiver({11: session})
    monotonic = Mock(side_effect=[0.0, 0.0, 0.123])
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.time.monotonic",
        monotonic,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.time.sleep",
        Mock(),
    )

    status = transceiver.check_context_transfer_status(at_least_request_num=1)
    assert isinstance(status, CtxTransferStatus)
    completed, failed = status

    assert completed == []
    assert failed == []
    assert session.blocking_calls == [False]
    assert not session.closed
    assert 11 in transceiver._send_sessions
    assert 11 in transceiver._send_reqs
    assert transceiver._transfer_worker.sweep_count == 1


def test_context_transfer_status_bounded_poll_reaps_completion(monkeypatch) -> None:
    session = _FakeSession(rid=14, wait_result=WaitResult.COMPLETED)
    req = _FakeRequest()
    transceiver = _make_transceiver({14: session}, {14: req})

    def complete_on_poll(blocking: bool = True) -> WaitResult:
        session.blocking_calls.append(blocking)
        session._is_completed = True
        return WaitResult.COMPLETED

    session.wait_complete = complete_on_poll
    sleep = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.time.monotonic",
        Mock(return_value=0.0),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.time.sleep",
        sleep,
    )

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=1)

    assert completed == [14]
    assert failed == []
    assert session.blocking_calls == [False, False]
    sleep.assert_called_once_with(0.001)
    assert session.closed
    assert 14 not in transceiver._send_sessions
    assert 14 not in transceiver._send_reqs


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


def test_context_transfer_status_timeout_retains_session_and_request(monkeypatch) -> None:
    session = _FakeSession(rid=16, wait_result=WaitResult.TIMEOUT)
    req = _FakeRequest()
    transceiver = _make_transceiver({16: session}, {16: req})
    transceiver.kv_transfer_timeout_ms = 60_000
    warning = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.logger.warning",
        warning,
    )

    completed, failed = transceiver.check_context_transfer_status(None)

    completed_again, failed_again = transceiver.check_context_transfer_status(None)

    assert completed == []
    assert failed == []
    assert completed_again == []
    assert failed_again == []
    assert session.blocking_calls == [True, True]
    assert not session.closed
    assert session.aux_slot == 0
    assert transceiver._send_sessions == {16: session}
    assert transceiver._send_reqs == {16: req}
    assert warning.call_count == 2
    messages = [args[0] for args, _kwargs in warning.call_args_list]
    assert all("rid=16" in message for message in messages)
    assert all("kv_transfer_timeout_ms=60000ms" in message for message in messages)
    assert all("keeping it in progress" in message for message in messages)


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
    assert req.state is None
    assert req.py_kv_send_session_retired
    assert 13 not in transceiver._send_sessions
    assert 13 not in transceiver._send_reqs


@pytest.mark.parametrize(
    ("status", "has_failed"),
    [
        (SessionStatus.CANCELLED, True),
        (SessionStatus.ERROR, True),
    ],
)
def test_context_transfer_status_retains_terminal_session_during_write(
    status: SessionStatus,
    has_failed: bool,
) -> None:
    session = _FakeSession(
        rid=17,
        wait_result=WaitResult.FAILED,
        status=status,
        has_failed=has_failed,
        has_transferring_tasks=True,
    )
    req = _FakeRequest()
    transceiver = _make_transceiver({17: session}, {17: req})

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    assert not session.closed
    assert transceiver._send_sessions == {17: session}
    assert transceiver._send_reqs == {17: req}
    assert req.state is None


def test_context_transfer_status_retires_quiesced_cancelled_session() -> None:
    session = _FakeSession(
        rid=18,
        wait_result=WaitResult.FAILED,
        status=SessionStatus.CANCELLED,
        has_failed=True,
    )
    req = _FakeRequest()
    transceiver = _make_transceiver({18: session}, {18: req})

    completed, failed = transceiver.check_context_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    assert session.closed
    assert 18 not in transceiver._send_sessions
    assert 18 not in transceiver._send_reqs


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


def test_context_transfer_status_never_sent_no_sync_is_a_noop() -> None:
    # With no tp/pp sync (e.g. attention_dp), a never-sent worker skips the consensus and the sweep,
    # unchanged from before -- a true no-op, so the fix can't slow attention_dp workers.
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_send_session = False
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._ctx_consensus = Mock(side_effect=AssertionError("consensus must be skipped"))

    assert transceiver.check_context_transfer_status(at_least_request_num=0) == ([], [])
    transceiver._ctx_consensus.assert_not_called()
    assert transceiver._transfer_worker.sweep_count == 0  # matches the original early-out exactly


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

    status = transceiver.check_gen_transfer_status(at_least_request_num=0)
    assert isinstance(status, GenTransferStatus)
    completed, failed, cancelled = status

    assert completed == []
    assert failed == []
    assert cancelled == []
    transceiver._gen_consensus.assert_called_once_with([])


def test_consensus_outcome_uses_single_batched_allgather() -> None:
    # The outcome and quiescence id lists are exchanged with ONE allgather.
    transceiver = object.__new__(KvCacheTransceiverV2)
    calls: list = []

    def fake_allgather(payload):
        calls.append(payload)
        # rank0 = this rank; rank1 = a peer rank.
        return [payload, [[], [99], [7, 8], [7, 99]]]

    to_process = [1, 2, 7, 8, 99]
    new_cancelled, new_failed, new_completed, new_quiesced = transceiver._consensus_outcome(
        to_process, [1], [2], [7], fake_allgather, True, [1, 7]
    )

    assert len(calls) == 1  # batched: a single allgather, not three
    assert calls[0] == [[1], [2], [7], [1, 7]]
    assert new_cancelled == [1]  # union of cancelled across ranks
    assert new_failed == [2, 99]  # union of failed across ranks
    assert new_completed == [7]  # intersection only (8 is completed on the peer only)
    assert new_quiesced == [7]


def test_ctx_tp_consensus_does_not_complete_when_peer_times_out() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_need_pp_sync = False
    transceiver._dist = SimpleNamespace(
        tp_allgather=lambda payload: [payload, [[], [], [], []]],
    )

    cancelled, failed, completed, quiesced = transceiver._ctx_consensus_outcome(
        [21], [], [], [21], [21]
    )

    assert cancelled == []
    assert failed == []
    assert completed == []
    assert quiesced == []


def test_ctx_pp_consensus_does_not_complete_when_peer_times_out() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = True
    transceiver._dist = SimpleNamespace(
        tp_allgather=Mock(side_effect=AssertionError("TP allgather must be skipped")),
        pp_allgather=lambda payload: [payload, [[], [], [], []]],
    )

    cancelled, failed, completed, quiesced = transceiver._ctx_consensus_outcome(
        [22], [], [], [22], [22]
    )

    assert cancelled == []
    assert failed == []
    assert completed == []
    assert quiesced == []
    transceiver._dist.tp_allgather.assert_not_called()


@pytest.mark.skip(
    reason="ctx idle fast-path was dropped from this branch. TODO: when the "
    "fast-path is reintroduced, its terminal-count reduction must mirror "
    "_ctx_consensus()'s communicator scope (TP group, then PP group; TP "
    "skipped under attention DP) — a WORLD-scoped allreduce hangs under "
    "ADP+PP because independent attention-DP lanes poll on their own "
    "schedules. Re-enable this test and add scoped mock coverage for the "
    "TP+PP and ADP+PP configurations plus real-collective MP tests."
)
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


def test_tx_session_blocking_wait_retries_wait_slices_until_complete() -> None:
    task = _FakeTask(TaskStatus.INIT, wait_result=[False, True])
    session = _make_tx_session([task])

    assert session.wait_complete() == WaitResult.COMPLETED
    assert task.wait_calls == [0.25, 0.25]


def test_tx_session_blocking_wait_times_out_stalled_task_and_does_not_reset_deadline(
    monkeypatch,
) -> None:
    clock = _FakeClock()
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False, on_wait=clock.advance)
    session = _make_tx_session(
        [task],
        timeout_s=0.25,
        deadline_monotonic_s=0.6,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.TIMEOUT
    assert task.wait_calls == pytest.approx([0.25, 0.25, 0.1])
    assert not session._closed
    assert not session.has_failed()

    wait_call_count = len(task.wait_calls)
    assert session.wait_complete(blocking=True) == WaitResult.TIMEOUT
    assert len(task.wait_calls) == wait_call_count


def test_tx_session_blocking_wait_uses_finite_overall_fallback_when_unset(
    monkeypatch,
) -> None:
    clock = _FakeClock()
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False, on_wait=clock.advance)
    session = _make_tx_session([task], timeout_s=0.25)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer._FALLBACK_TX_OVERALL_TIMEOUT_S",
        0.6,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.TIMEOUT
    assert task.wait_calls == pytest.approx([0.25, 0.25, 0.1])
    assert session._deadline_monotonic_s == pytest.approx(0.6)

    wait_call_count = len(task.wait_calls)
    assert session.wait_complete(blocking=True) == WaitResult.TIMEOUT
    assert len(task.wait_calls) == wait_call_count


def test_tx_session_completion_observed_at_deadline_wins_over_timeout(monkeypatch) -> None:
    clock = _FakeClock()
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)

    def complete_at_deadline(timeout_s: Optional[float]) -> None:
        clock.advance(timeout_s)
        task.status = TaskStatus.TRANSFERRED

    task._on_wait = complete_at_deadline
    session = _make_tx_session(
        [task],
        timeout_s=0.25,
        deadline_monotonic_s=0.25,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.COMPLETED
    assert task.wait_calls == [0.25]


@pytest.mark.parametrize(
    ("terminal", "expected"),
    [
        ("completed", WaitResult.COMPLETED),
        ("failed", WaitResult.FAILED),
        ("cancelled", WaitResult.FAILED),
    ],
)
def test_tx_session_terminal_transition_during_deadline_read_wins_over_timeout(
    monkeypatch,
    terminal: str,
    expected: WaitResult,
) -> None:
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session(
        [task],
        timeout_s=0.25,
        deadline_monotonic_s=0.25,
    )

    def expire_after_terminal_transition() -> float:
        if terminal == "completed":
            task.status = TaskStatus.TRANSFERRED
        elif terminal == "failed":
            task.status = TaskStatus.ERROR
        else:
            session._terminal_status = SessionStatus.CANCELLED
        return 0.25

    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        expire_after_terminal_transition,
    )

    assert session.wait_complete(blocking=True) == expected
    assert task.wait_calls == []


def test_tx_session_failure_observed_at_deadline_wins_over_timeout(monkeypatch) -> None:
    clock = _FakeClock()
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)

    def fail_at_deadline(timeout_s: Optional[float]) -> None:
        clock.advance(timeout_s)
        task.status = TaskStatus.ERROR

    task._on_wait = fail_at_deadline
    session = _make_tx_session(
        [task],
        timeout_s=0.25,
        deadline_monotonic_s=0.25,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert task.wait_calls == [0.25]


def test_tx_session_cancellation_observed_at_deadline_wins_over_timeout(monkeypatch) -> None:
    clock = _FakeClock()
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session(
        [task],
        timeout_s=0.25,
        deadline_monotonic_s=0.25,
    )

    def cancel_at_deadline(timeout_s: Optional[float]) -> None:
        clock.advance(timeout_s)
        session._terminal_status = SessionStatus.CANCELLED

    task._on_wait = cancel_at_deadline
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert task.wait_calls == [0.25]


def test_tx_session_tasks_and_aux_share_one_deadline(monkeypatch) -> None:
    clock = _FakeClock()
    first_wait_durations = iter([0.5, 0.1])

    def advance_first_task(_timeout_s: Optional[float]) -> None:
        clock.advance(next(first_wait_durations))

    first_task = _FakeTask(
        TaskStatus.TRANSFERRING,
        wait_result=[False, True],
        on_wait=advance_first_task,
    )
    second_task = _FakeTask(
        TaskStatus.TRANSFERRING,
        wait_result=True,
        on_wait=lambda _timeout_s: clock.advance(0.2),
    )
    aux_task = _FakeTask(
        TaskStatus.TRANSFERRING,
        wait_result=False,
        on_wait=clock.advance,
    )
    session = _make_tx_session(
        [first_task, second_task],
        need_aux=True,
        aux_task=aux_task,
        timeout_s=0.5,
        deadline_monotonic_s=1.0,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )

    assert session.wait_complete(blocking=True) == WaitResult.TIMEOUT
    assert first_task.wait_calls == pytest.approx([0.5, 0.5])
    assert second_task.wait_calls == pytest.approx([0.4])
    assert aux_task.wait_calls == pytest.approx([0.2])


def test_tx_session_first_send_anchors_deadline_once(monkeypatch) -> None:
    clock = _FakeClock(now_s=10.0)
    sender = Mock()
    sender._get_req_info.return_value = {}
    params = SimpleNamespace(
        schedule_style="CONTEXT_FIRST",
        disagg_request_id=31,
        ctx_request_id=None,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.time.monotonic",
        clock.monotonic,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.tensorrt_llm.bindings.global_steady_clock_now",
        Mock(return_value=123),
    )
    session = TxSession(
        request_id=31,
        params=params,
        sender=sender,
        prompt_len=8,
        timeout_s=0.25,
        overall_timeout_s=2.0,
    )

    assert session._deadline_monotonic_s is None
    session.send(Mock(is_last_slice=False))
    assert session._deadline_monotonic_s == 12.0

    clock.advance(0.5)
    session.send(Mock(is_last_slice=False))
    assert session._deadline_monotonic_s == 12.0
    assert sender.dispatch_task.call_count == 2
    session.close()


@pytest.mark.parametrize(
    ("transfer_timeout_ms", "sender_wait_ms", "expected_timeout_s", "expected_slice_s"),
    [
        (60_000, 1_000, 60.0, 1.0),
        (60_000, None, 60.0, None),
    ],
)
def test_transceiver_wires_separate_sender_slice_and_overall_timeout(
    monkeypatch,
    transfer_timeout_ms: Optional[int],
    sender_wait_ms: Optional[int],
    expected_timeout_s: Optional[float],
    expected_slice_s: Optional[float],
) -> None:
    worker = SimpleNamespace(page_table=None)
    worker_constructor = Mock(return_value=worker)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.TransferWorker",
        worker_constructor,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.create_cache_reuse_adapter",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.bounce_config_from_size",
        Mock(return_value=None),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.torch.cuda.current_device",
        Mock(return_value=0),
    )
    monkeypatch.setattr(
        KvCacheTransceiverV2,
        "_broadcast_instance_name",
        lambda _self: "ctx",
    )
    monkeypatch.setattr(
        KvCacheTransceiverV2,
        "_broadcast_context_endpoint",
        lambda _self: "endpoint",
    )
    monkeypatch.setattr(KvCacheTransceiverV2, "_init_sync_policy", lambda _self: None)
    monkeypatch.setattr(KvCacheTransceiverV2, "_exchange_rank_info", lambda _self: None)
    mapping = SimpleNamespace(
        cp_size=1,
        tp_rank=0,
        tp_size=1,
        enable_attention_dp=False,
    )
    cache_config = SimpleNamespace(
        kv_transfer_timeout_ms=transfer_timeout_ms,
        kv_transfer_poll_interval_ms=5_000,
        kv_transfer_sender_future_timeout_ms=sender_wait_ms,
        kv_cache_bounce_size_mb=0,
        enable_pipelined_transfer=False,
    )

    KvCacheTransceiverV2(
        mapping=mapping,
        dist=Mock(),
        kv_cache_manager=SimpleNamespace(max_batch_size=4),
        cache_transceiver_config=cache_config,
    )

    worker_config = worker_constructor.call_args.args[0]
    assert isinstance(worker_config, TransferWorkerConfig)
    assert worker_config.tx_timeout_s == expected_slice_s
    assert worker_config.tx_overall_timeout_s == expected_timeout_s
    assert worker_config.rx_timeout_s == expected_timeout_s


def test_transceiver_rejects_unset_transfer_timeout() -> None:
    cache_config = SimpleNamespace(
        kv_transfer_timeout_ms=None,
        kv_transfer_poll_interval_ms=5_000,
        kv_transfer_sender_future_timeout_ms=1_000,
    )

    with pytest.raises(
        ValueError,
        match="KvCacheTransceiverV2 requires a finite kv_transfer_timeout_ms",
    ):
        KvCacheTransceiverV2(
            mapping=Mock(),
            dist=Mock(),
            kv_cache_manager=Mock(),
            cache_transceiver_config=cache_config,
        )


def test_transfer_worker_passes_overall_timeout_to_tx_session(monkeypatch) -> None:
    session_constructor = Mock(return_value=Mock())
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.TxSession",
        session_constructor,
    )
    worker = object.__new__(TransferWorker)
    worker._config = TransferWorkerConfig(
        kv_cache_manager=Mock(),
        device_id=0,
        instance_name="ctx",
        tx_timeout_s=0.25,
        tx_overall_timeout_s=60.0,
    )
    worker._sender = Mock()
    worker._aux_buffer = Mock()
    request = SimpleNamespace(
        py_disaggregated_params=Mock(),
        py_request_id=41,
        prompt_len=128,
        py_beam_width=1,
    )

    worker.create_tx_session(request)

    session_constructor.assert_called_once_with(
        request_id=41,
        params=request.py_disaggregated_params,
        sender=worker._sender,
        aux_buffer=worker._aux_buffer,
        timeout_s=0.25,
        prompt_len=128,
        beam_width=1,
        overall_timeout_s=60.0,
    )


def test_context_transfer_status_block_all_drains_wait_slices_before_close() -> None:
    task = _FakeTask(TaskStatus.INIT, wait_result=[False, True])
    session = _make_tx_session([task])
    transceiver = _make_transceiver({15: session}, {15: _FakeRequest()})

    completed, failed = transceiver.check_context_transfer_status(None)

    assert completed == [15]
    assert failed == []
    assert task.wait_calls == [0.25, 0.25]
    assert session._closed
    assert 15 not in transceiver._send_sessions


def test_tx_session_blocking_wait_treats_cancelled_session_as_terminal() -> None:
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session([task])
    session._terminal_status = SessionStatus.CANCELLED

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert task.wait_calls == []


def test_tx_session_blocking_wait_observes_cancellation_between_slices() -> None:
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session([task])
    wait = task.wait

    def cancel_during_wait(timeout: Optional[float] = None) -> bool:
        result = wait(timeout)
        session._terminal_status = SessionStatus.CANCELLED
        return result

    task.wait = cancel_during_wait

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert task.wait_calls == [0.25]


@pytest.mark.parametrize("timeout_s", [None, 0.0, -1.0])
def test_tx_session_blocking_wait_uses_fallback_without_positive_timeout(
    timeout_s: Optional[float],
) -> None:
    task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session([task], timeout_s=timeout_s)
    wait = task.wait

    def cancel_during_wait(timeout: Optional[float] = None) -> bool:
        result = wait(timeout)
        session._terminal_status = SessionStatus.CANCELLED
        return result

    task.wait = cancel_during_wait

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert task.wait_calls == [1.0]


def test_tx_session_blocking_wait_treats_task_failure_as_terminal() -> None:
    failed_task = _FakeTask(TaskStatus.ERROR)
    pending_task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=[False, True])
    session = _make_tx_session([failed_task, pending_task])

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert failed_task.wait_calls == []
    # A failed task event does not prove sibling physical writers quiesced, so
    # precheck callers retain the wave instead of treating failure as drained.
    assert pending_task.wait_calls == []


def test_tx_session_blocking_wait_detects_failed_sibling_behind_pending_task() -> None:
    pending_task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    failed_task = _FakeTask(TaskStatus.TRANSFERRING)
    session = _make_tx_session([pending_task, failed_task])

    def fail_sibling(_timeout: Optional[float]) -> None:
        failed_task.status = TaskStatus.ERROR

    pending_task._on_wait = fail_sibling

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert pending_task.wait_calls == [0.25]
    assert failed_task.wait_calls == []


def test_tx_session_blocking_wait_retries_aux_wait_slices() -> None:
    kv_task = _FakeTask(TaskStatus.TRANSFERRED)
    aux_task = _FakeTask(TaskStatus.INIT, wait_result=[False, True])
    session = _make_tx_session([kv_task], need_aux=True, aux_task=aux_task)

    assert session.wait_complete(blocking=True) == WaitResult.COMPLETED
    assert kv_task.wait_calls == []
    assert aux_task.wait_calls == [0.25, 0.25]


def test_tx_session_blocking_aux_wait_observes_cancellation_between_slices() -> None:
    kv_task = _FakeTask(TaskStatus.TRANSFERRED)
    aux_task = _FakeTask(TaskStatus.TRANSFERRING, wait_result=False)
    session = _make_tx_session([kv_task], need_aux=True, aux_task=aux_task)
    wait = aux_task.wait

    def cancel_during_wait(timeout: Optional[float] = None) -> bool:
        result = wait(timeout)
        session._terminal_status = SessionStatus.CANCELLED
        return result

    aux_task.wait = cancel_during_wait

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert kv_task.wait_calls == []
    assert aux_task.wait_calls == [0.25]


def test_tx_session_blocking_wait_fails_missing_required_aux() -> None:
    task = _FakeTask(TaskStatus.TRANSFERRED)
    session = _make_tx_session([task], need_aux=True)

    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert session.status == SessionStatus.ERROR
    assert isinstance(session.exception, RuntimeError)
    assert task.wait_calls == []


def test_generation_first_tx_session_nonblocking_missing_aux_stays_pending() -> None:
    task = _FakeTask(TaskStatus.TRANSFERRED)
    session = _make_tx_session([task], need_aux=True)

    assert session.wait_complete(blocking=False) is None
    assert session.status == SessionStatus.KV_TRANSFERRED
    assert session.exception is None
    assert task.wait_calls == []


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


def test_check_context_runs_consensus_after_a_send() -> None:
    # Once the worker has sent, the ctx consensus runs as usual.
    transceiver = _make_transceiver({})
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_consensus = Mock(return_value=[])
    transceiver._ctx_consensus_outcome = Mock(return_value=([], [], [], []))

    transceiver.check_context_transfer_status(0)
    transceiver._ctx_consensus.assert_called_once()


def test_prepare_context_requests_skips_consensus_when_nothing_waiting() -> None:
    # With nothing waiting on any rank, prepare_context_requests returns before the consensus; the
    # waiting set is the same on every rank.
    transceiver = _make_transceiver({})
    transceiver._wait_reqs = {}
    transceiver._ctx_consensus = Mock(side_effect=AssertionError("consensus must be skipped"))

    transceiver.prepare_context_requests([])
    transceiver._ctx_consensus.assert_not_called()


# ---------------------------------------------------------------------------
# _poll_sessions_for_interval clamp (nvbugs 6647405)
#
# The idle executor loop calls check_context_transfer_status(1) on every
# iteration where no batch is scheduled. The poll's exit condition
# (completed + failed >= wait_num) can only ever count in-flight sessions, so
# with _send_sessions empty the target used to be unsatisfiable and the helper
# slept out the full kv_transfer_sender_future_timeout_ms (default 1000 ms)
# per idle iteration, delaying scheduling of newly arrived requests.
# ---------------------------------------------------------------------------


def _make_bare_transceiver() -> KvCacheTransceiverV2:
    """Bare instance; _poll_sessions_for_interval only needs _collect_done."""
    return object.__new__(KvCacheTransceiverV2)


class _PollFakeSession:
    """Session stub that flips to completed after an optional wall-clock delay."""

    def __init__(self, complete_after_s: Optional[float] = None, failed: bool = False) -> None:
        self._failed = failed
        self._complete_at = (
            time.monotonic() + complete_after_s if complete_after_s is not None else None
        )

    def is_completed(self) -> bool:
        return self._complete_at is not None and time.monotonic() >= self._complete_at

    def has_failed(self) -> bool:
        return self._failed

    def wait_complete(self, blocking: bool = False) -> None:
        pass


class _PumpDrivenSession(_PollFakeSession):
    """Completes only after wait_complete has been pumped, never by wall clock."""

    def __init__(self, pumps_to_complete: int) -> None:
        super().__init__()
        self._pumps_left = pumps_to_complete
        self._done = False

    def is_completed(self) -> bool:
        return self._done

    def wait_complete(self, blocking: bool = False) -> None:
        self._pumps_left -= 1
        if self._pumps_left <= 0:
            self._done = True


_POLL_INTERVAL_MS = 1000


def test_poll_interval_empty_sessions_returns_immediately() -> None:
    """No in-flight session: the unsatisfiable target must not sleep out the interval."""
    tc = _make_bare_transceiver()
    start = time.monotonic()
    tc._poll_sessions_for_interval({}, {}, 1, _POLL_INTERVAL_MS)
    assert time.monotonic() - start < 0.1


def test_poll_interval_wait_num_clamped_to_session_count() -> None:
    """A target above len(sessions) waits only for what can actually complete."""
    tc = _make_bare_transceiver()
    sessions = {1: _PollFakeSession(complete_after_s=0.05)}
    start = time.monotonic()
    tc._poll_sessions_for_interval(sessions, {1: object()}, 2, _POLL_INTERVAL_MS)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5
    assert sessions[1].is_completed()


def test_poll_interval_waits_for_inflight_completion() -> None:
    """An in-flight session is still awaited (the PR #17535 semantics are kept)."""
    tc = _make_bare_transceiver()
    sessions = {1: _PollFakeSession(complete_after_s=0.05)}
    start = time.monotonic()
    tc._poll_sessions_for_interval(sessions, {1: object()}, 1, _POLL_INTERVAL_MS)
    elapsed = time.monotonic() - start
    assert 0.04 <= elapsed < 0.5
    assert sessions[1].is_completed()


def test_poll_interval_deadline_bounds_never_completing_session() -> None:
    """A session that never completes releases the caller at the deadline."""
    tc = _make_bare_transceiver()
    sessions = {1: _PollFakeSession(complete_after_s=None)}
    start = time.monotonic()
    tc._poll_sessions_for_interval(sessions, {1: object()}, 1, 100)
    elapsed = time.monotonic() - start
    assert 0.09 <= elapsed < 1.0


def test_poll_interval_failed_session_counts_toward_target() -> None:
    """A failed session satisfies the exit condition without waiting."""
    tc = _make_bare_transceiver()
    sessions = {1: _PollFakeSession(failed=True)}
    start = time.monotonic()
    tc._poll_sessions_for_interval(sessions, {1: object()}, 1, _POLL_INTERVAL_MS)
    assert time.monotonic() - start < 0.1


def test_poll_interval_completed_session_releases_despite_inflight_peer() -> None:
    """One already-completed session satisfies wait_num=1 even with an in-flight peer."""
    tc = _make_bare_transceiver()
    sessions = {
        1: _PollFakeSession(complete_after_s=0.0),
        2: _PollFakeSession(complete_after_s=None),
    }
    reqs = {1: object(), 2: object()}
    start = time.monotonic()
    tc._poll_sessions_for_interval(sessions, reqs, 1, _POLL_INTERVAL_MS)
    assert time.monotonic() - start < 0.1


def test_poll_interval_pump_drives_completion() -> None:
    """Completion observed only through the wait_complete(blocking=False) pump exits the poll."""
    tc = _make_bare_transceiver()
    session = _PumpDrivenSession(pumps_to_complete=3)
    start = time.monotonic()
    tc._poll_sessions_for_interval({1: session}, {1: object()}, 1, _POLL_INTERVAL_MS)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5
    assert session.is_completed()
