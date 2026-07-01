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

"""Unit tests for the WideEP MPI failure-broadcast control plane.

The tests use in-memory MPI, communicator, and request doubles. They exercise
the real progress thread without requiring mpi4py, GPUs, or a multi-rank
launcher.
"""

from __future__ import annotations

import gc
import sys
import threading
import time
import types
import weakref
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from tensorrt_llm._torch.modules.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.pyexecutor import ep_failure_broadcast
from tensorrt_llm._torch.pyexecutor.ep_failure_broadcast import (
    DetectedRankState,
    MpiFtSubcomm,
    MpiFtSubcommConfig,
)

_TEST_CONFIG = MpiFtSubcommConfig(
    poll_interval_sec=0.001,
    startup_timeout_sec=0.5,
    stop_timeout_sec=0.5,
    reconcile_timeout_sec=2.0,
    unattributed_error_timeout_sec=2.0,
    abort_timeout_sec=2.0,
)

_FAILURE_MESSAGE = int(ep_failure_broadcast._MessageKind.FAILURE)
_ABORT_MESSAGE = int(ep_failure_broadcast._MessageKind.ABORT)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_config_requires_positive_finite_timeouts(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite and > 0"):
        MpiFtSubcommConfig(poll_interval_sec=value)
    with pytest.raises(ValueError, match="must be finite and > 0"):
        MpiFtSubcommConfig(unattributed_error_timeout_sec=value)
    with pytest.raises(ValueError, match="must be finite and > 0"):
        MpiFtSubcommConfig(abort_timeout_sec=value)


def test_default_poll_interval_leaves_margin_below_agreement_budget() -> None:
    assert MpiFtSubcommConfig().poll_interval_sec == 0.01


@dataclass(frozen=True)
class _FakeMapping:
    world_size: int
    rank: int
    moe_ep_size: int
    moe_ep_rank: int
    moe_ep_group: tuple[int, ...]

    @classmethod
    def ep_world(cls, size: int, rank: int = 0) -> "_FakeMapping":
        return cls(
            world_size=size,
            rank=rank,
            moe_ep_size=size,
            moe_ep_rank=rank,
            moe_ep_group=tuple(range(size)),
        )


class _FakeMpiException(RuntimeError):
    def __init__(self, error_class: int, message: str = "fake MPI failure") -> None:
        super().__init__(message)
        self._error_class = error_class

    def Get_error_class(self) -> int:
        return self._error_class


class _FakeMPI:
    ERRORS_RETURN = object()
    ERR_PROC_FAILED = 101
    ERR_REVOKED = 102
    ERR_UNSUPPORTED_OPERATION = 103
    ERR_UNKNOWN = -1
    THREAD_MULTIPLE = 3
    Exception = _FakeMpiException

    def __init__(self, thread_level: int = THREAD_MULTIPLE) -> None:
        self._thread_level = thread_level

    def Query_thread(self) -> int:
        return self._thread_level


class _FakeRequest:
    """Controllable nonblocking request that never owns its MPI buffer."""

    def __init__(self, *, completed: bool = False) -> None:
        self._completed = threading.Event()
        if completed:
            self._completed.set()
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self.cancel_calls = 0
        self.test_calls = 0

    def Cancel(self) -> None:
        with self._lock:
            self.cancel_calls += 1
        self._completed.set()

    def Test(self) -> bool:
        with self._lock:
            self.test_calls += 1
            error = self._error
        if error is not None:
            raise error
        return self._completed.is_set()

    def complete(self) -> None:
        self._completed.set()

    def fail(self, error: BaseException) -> None:
        with self._lock:
            self._error = error


class _BlockingRequest(_FakeRequest):
    """Request whose Test call can hold the progress thread for stop tests."""

    def __init__(self) -> None:
        super().__init__()
        self.entered_test = threading.Event()
        self.release_test = threading.Event()

    def Test(self) -> bool:
        self.entered_test.set()
        self.release_test.wait()
        return super().Test()


class _NonCompletingCancelRequest(_FakeRequest):
    """Cancellation request that remains incomplete until explicitly released."""

    def Cancel(self) -> None:
        with self._lock:
            self.cancel_calls += 1


@dataclass(frozen=True)
class _ReceiveRecord:
    request: _FakeRequest
    buffer: np.ndarray
    thread_id: int


@dataclass(frozen=True)
class _SendRecord:
    request: _FakeRequest
    buffer_ref: weakref.ReferenceType[np.ndarray]
    destination: int
    message_kind: int
    payload: int
    tag: int
    thread_id: int


class _FakeComm:
    def __init__(
        self,
        *,
        size: int,
        rank: int = 0,
        receive_factory: Callable[[int], _FakeRequest] | None = None,
        send_factory: Callable[[int], _FakeRequest] | None = None,
        ulfm_probe_error: BaseException | None = None,
        ulfm_revoked: bool = False,
        revoke_error: BaseException | None = None,
        abort_error: BaseException | None = None,
    ) -> None:
        self._size = size
        self._rank = rank
        self._receive_factory = receive_factory or (lambda _source: _FakeRequest())
        self._send_factory = send_factory or (lambda _destination: _FakeRequest())
        self._ulfm_probe_error = ulfm_probe_error
        self._ulfm_revoked = ulfm_revoked
        self._revoke_error = revoke_error
        self._abort_error = abort_error
        self._lock = threading.Lock()
        self._receives: dict[int, list[_ReceiveRecord]] = defaultdict(list)
        self._sends: list[_SendRecord] = []
        self.errhandler: object | None = None
        self.errhandler_calls = 0
        self.is_revoked_calls = 0
        self.revoke_calls = 0
        self.abort_calls = 0

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def Set_errhandler(self, errhandler: object) -> None:
        self.errhandler_calls += 1
        self.errhandler = errhandler

    def Is_revoked(self) -> bool:
        self.is_revoked_calls += 1
        if self._ulfm_probe_error is not None:
            raise self._ulfm_probe_error
        return self._ulfm_revoked

    def Revoke(self) -> None:
        self.revoke_calls += 1
        if self._revoke_error is not None:
            raise self._revoke_error

    def Abort(self, _errorcode: int) -> None:
        self.abort_calls += 1
        if self._abort_error is not None:
            raise self._abort_error

    def Irecv(self, buffer: np.ndarray, source: int, tag: int) -> _FakeRequest:
        del tag
        request = self._receive_factory(source)
        record = _ReceiveRecord(
            request=request,
            buffer=buffer,
            thread_id=threading.get_ident(),
        )
        with self._lock:
            self._receives[source].append(record)
        return request

    def Isend(self, buffer: np.ndarray, dest: int, tag: int) -> _FakeRequest:
        request = self._send_factory(dest)
        record = _SendRecord(
            request=request,
            buffer_ref=weakref.ref(buffer),
            destination=dest,
            message_kind=int(buffer[0]),
            payload=int(buffer[1]),
            tag=tag,
            thread_id=threading.get_ident(),
        )
        with self._lock:
            self._sends.append(record)
        return request

    def receive_count(self, source: int) -> int:
        with self._lock:
            return len(self._receives[source])

    def latest_receive(self, source: int) -> _ReceiveRecord:
        with self._lock:
            return self._receives[source][-1]

    def deliver(
        self,
        source: int,
        failed_rank: int,
        *,
        message_kind: int = _FAILURE_MESSAGE,
    ) -> None:
        record = self.latest_receive(source)
        record.buffer[:] = (message_kind, failed_rank)
        record.request.complete()

    @property
    def sends(self) -> tuple[_SendRecord, ...]:
        with self._lock:
            return tuple(self._sends)


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 1.0,
    description: str = "condition",
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError(f"timed out waiting for {description}")


def _make_broadcaster(
    *,
    size: int = 4,
    rank: int = 0,
    detected_state: DetectedRankState | None = None,
    comm: _FakeComm | None = None,
    mpi: _FakeMPI | None = None,
    callback: Callable[[int, int, float], None] | None = None,
) -> tuple[MpiFtSubcomm, DetectedRankState, _FakeComm, _FakeMPI]:
    detected_state = detected_state or DetectedRankState(size)
    comm = comm or _FakeComm(size=size, rank=rank)
    mpi = mpi or _FakeMPI()
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(size, rank),
        detected_state,
        _TEST_CONFIG,
        callback,
        comm=comm,
        mpi_module=mpi,
    )
    return broadcaster, detected_state, comm, mpi


def test_committed_ep_group_health_is_rejected_before_mpi_setup() -> None:
    comm = _FakeComm(size=2)

    with pytest.raises(TypeError, match="committed EPGroupHealth cannot be used"):
        MpiFtSubcomm(
            _FakeMapping.ep_world(2),
            EPGroupHealth(2),
            _TEST_CONFIG,
            comm=comm,
            mpi_module=_FakeMPI(),
        )

    assert comm.errhandler_calls == 0
    assert comm.is_revoked_calls == 0


def test_watchdog_callback_records_and_fans_out_detected_failure() -> None:
    detected_state = DetectedRankState(4)
    broadcaster, _, comm, _ = _make_broadcaster(detected_state=detected_state)
    broadcaster.start()
    caller_thread = threading.get_ident()
    try:
        watchdog_callback = broadcaster.report_detected_failure
        assert watchdog_callback(2) is True
        assert detected_state.get_failed_ranks() == frozenset({2})
        _wait_until(lambda: len(comm.sends) == 2, description="failure fanout")
        _wait_until(
            lambda: all(record.request.test_calls > 0 for record in comm.sends),
            description="send request polling",
        )

        assert {record.destination for record in comm.sends} == {1, 3}
        assert {record.payload for record in comm.sends} == {2}
        assert all(record.thread_id != caller_thread for record in comm.sends)
        assert all(record.request.test_calls > 0 for record in comm.sends)
        assert all(record.buffer_ref() is not None for record in comm.sends)

        # A repeated detector report is locally coalesced.
        assert watchdog_callback(2) is False
        time.sleep(2 * _TEST_CONFIG.poll_interval_sec)
        assert len(comm.sends) == 2

        for send in comm.sends:
            send.request.complete()
        comm.deliver(source=1, failed_rank=2)
        comm.deliver(source=3, failed_rank=2)
        _wait_until(
            broadcaster.detected_state_is_reconciled,
            description="detection reconciliation",
        )
    finally:
        broadcaster.stop()


def test_externally_premarked_detected_state_fails_closed() -> None:
    detected_state = DetectedRankState(2)
    assert detected_state.record_failure(1) is True
    broadcaster, _, comm, _ = _make_broadcaster(
        size=2,
        detected_state=detected_state,
    )

    try:
        try:
            broadcaster.start()
        except RuntimeError:
            # The progress thread can detect the invalid initial state before
            # start() observes RUNNING, or immediately after start() returns.
            pass
        _wait_until(
            lambda: isinstance(broadcaster.last_error, RuntimeError),
            description="pre-recorded detected-state rejection",
        )
        assert "without pre-recording evidence" in str(broadcaster.last_error)
        _wait_until(lambda: comm.revoke_calls == 1, description="pre-mark fail-closed revoke")
    finally:
        broadcaster.stop()


def test_detection_reconciliation_does_not_mutate_committed_health() -> None:
    callbacks: list[tuple[int, int, float]] = []
    detected_state = DetectedRankState(2)
    committed_health = EPGroupHealth(2)
    initial_committed_snapshot = committed_health.snapshot()
    broadcaster, _, _, _ = _make_broadcaster(
        size=2,
        detected_state=detected_state,
        callback=lambda failed, source, when: callbacks.append((failed, source, when)),
    )
    broadcaster.start()
    try:
        assert broadcaster.report_detected_failure(1) is True
        _wait_until(
            lambda: len(callbacks) == 1,
            description="local detection coordinator handoff",
        )
        _wait_until(
            broadcaster.detected_state_is_reconciled,
            description="detection reconciliation",
        )

        assert detected_state.get_failed_ranks() == frozenset({1})
        assert detected_state.generation == 1
        assert committed_health.snapshot() == initial_committed_snapshot
        assert committed_health.all_active() is True
        assert callbacks[0][:2] == (1, 0)
    finally:
        broadcaster.stop()


def test_survivor_echoes_reconcile_failure_detection() -> None:
    broadcaster, _, comm, _ = _make_broadcaster(size=4)
    broadcaster.start()
    try:
        assert broadcaster.report_detected_failure(2) is True
        assert broadcaster.failure_detection_is_reconciled(2) is False
        _wait_until(lambda: len(comm.sends) == 2, description="initial failure fanout")

        comm.deliver(source=1, failed_rank=2)
        _wait_until(
            lambda: comm.receive_count(1) == 2,
            description="source 1 receive repost",
        )
        assert broadcaster.failure_detection_is_reconciled(2) is False

        comm.deliver(source=3, failed_rank=2)
        for send in comm.sends:
            send.request.complete()
        _wait_until(
            lambda: broadcaster.failure_detection_is_reconciled(2),
            description="all survivor echoes",
        )
        time.sleep(2 * _TEST_CONFIG.poll_interval_sec)
        assert len(comm.sends) == 2
        assert {send.destination for send in comm.sends} == {1, 3}
        assert all(send.message_kind == _FAILURE_MESSAGE for send in comm.sends)
        assert broadcaster.detected_health_is_reconciled() is True
        assert comm.revoke_calls == 0
    finally:
        broadcaster.stop()


def test_detected_state_is_monotonic_for_one_communicator_epoch() -> None:
    state = DetectedRankState(3)

    assert state.snapshot() == ep_failure_broadcast.DetectedRankStateSnapshot(
        mask=0b111,
        failed_ranks=frozenset(),
        generation=0,
    )
    assert state.record_failure(2) is True
    assert state.record_failure(2) is False
    assert state.get_mask() == 0b011
    assert state.get_failed_ranks() == frozenset({2})
    assert state.generation == 1
    assert not hasattr(state, "mark_active")


def test_detected_state_coalesces_concurrent_evidence() -> None:
    state = DetectedRankState(4)
    barrier = threading.Barrier(8)
    results: list[bool] = []
    results_lock = threading.Lock()

    def record_same_failure() -> None:
        barrier.wait()
        changed = state.record_failure(2)
        with results_lock:
            results.append(changed)

    threads = [threading.Thread(target=record_same_failure) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 1
    assert results.count(False) == 7
    assert state.snapshot() == ep_failure_broadcast.DetectedRankStateSnapshot(
        mask=0b1011,
        failed_ranks=frozenset({2}),
        generation=1,
    )


def test_second_failure_between_claim_and_observe_cannot_open_single_failure_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health = DetectedRankState(3)
    broadcaster, _, _, _ = _make_broadcaster(size=3, detected_state=health)
    original_claim = broadcaster._claim_failure
    mutated = False

    def claim_then_add_second_failure(failed_rank: int) -> None:
        nonlocal mutated
        original_claim(failed_rank)
        if not mutated:
            mutated = True
            assert health.record_failure(1) is True

    monkeypatch.setattr(broadcaster, "_claim_failure", claim_then_add_second_failure)
    broadcaster.start()
    try:
        assert broadcaster.broadcast_failure(2) is True
        _wait_until(
            lambda: 2 in broadcaster._failure_fanout_complete,
            description="single-survivor fanout completion",
        )
        assert health.get_failed_ranks() == frozenset({1, 2})
        _wait_until(
            lambda: isinstance(broadcaster.last_error, RuntimeError),
            description="second-detected-failure terminal handling",
        )
        assert broadcaster.failure_is_reconciled(2) is False
        assert broadcaster.health_is_reconciled() is False
    finally:
        broadcaster.stop()


def test_reconciliation_timeout_fails_closed_while_mpi_test_is_wedged() -> None:
    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.5,
        stop_timeout_sec=0.5,
        reconcile_timeout_sec=0.02,
    )
    blocking_request = _BlockingRequest()
    comm = _FakeComm(
        size=3,
        receive_factory=lambda source: blocking_request if source == 1 else _FakeRequest(),
    )
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(3),
        DetectedRankState(3),
        config,
        comm=comm,
        mpi_module=_FakeMPI(),
    )
    broadcaster.start()
    _wait_until(blocking_request.entered_test.is_set, description="blocked MPI progress")
    assert broadcaster.pre_failover(2) is True
    assert blocking_request.release_test.is_set() is False
    assert broadcaster._outbound_reports.empty() is False

    _wait_until(
        lambda: isinstance(broadcaster.last_error, TimeoutError),
        description="terminal reconciliation timeout",
    )
    _wait_until(lambda: comm.revoke_calls == 1, description="deadline-monitor revoke")
    assert blocking_request.release_test.is_set() is False
    assert broadcaster.health_is_reconciled() is False
    assert broadcaster.world_is_poisoned() is True
    with pytest.raises(RuntimeError, match="not running"):
        broadcaster.pre_failover(1)
    blocking_request.release_test.set()
    broadcaster.stop()


def test_reconciliation_timeout_world_aborts_when_mpi_test_is_wedged_without_ulfm() -> None:
    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.5,
        stop_timeout_sec=0.5,
        reconcile_timeout_sec=0.01,
        abort_timeout_sec=0.01,
    )
    blocking_request = _BlockingRequest()
    comm = _FakeComm(
        size=3,
        receive_factory=lambda source: blocking_request if source == 1 else _FakeRequest(),
        ulfm_probe_error=NotImplementedError(),
    )
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(3),
        DetectedRankState(3),
        config,
        comm=comm,
        mpi_module=_FakeMPI(),
    )
    broadcaster.start()
    _wait_until(blocking_request.entered_test.is_set, description="blocked MPI progress")
    assert broadcaster.pre_failover(2) is True

    _wait_until(lambda: comm.abort_calls == 1, description="deadline-monitor MPI_Abort")
    assert blocking_request.release_test.is_set() is False
    assert isinstance(broadcaster.last_error, TimeoutError)
    assert broadcaster.world_is_poisoned() is True

    blocking_request.release_test.set()
    broadcaster.stop()


def test_second_distinct_local_failure_fails_closed_before_health_mutation() -> None:
    broadcaster, health, comm, _ = _make_broadcaster(size=4)
    broadcaster.start()
    assert broadcaster.pre_failover(3) is True

    with pytest.raises(RuntimeError, match="exactly one distinct failed rank") as raised:
        broadcaster.pre_failover(2)

    assert health.get_failed_ranks() == frozenset({3})
    _wait_until(lambda: broadcaster.last_error is raised.value, description="terminal failure")
    _wait_until(lambda: comm.revoke_calls == 1, description="progress-thread abort revoke")
    assert broadcaster.health_is_reconciled() is False
    with pytest.raises(RuntimeError, match="not running"):
        broadcaster.pre_failover(1)
    broadcaster.stop()


def test_no_ulfm_terminal_abort_relays_and_converges_without_world_abort() -> None:
    comm = _FakeComm(size=3, ulfm_probe_error=NotImplementedError())
    broadcaster, health, _, _ = _make_broadcaster(size=3, comm=comm)
    broadcaster.start()
    assert broadcaster.pre_failover(2) is True
    with pytest.raises(RuntimeError, match="exactly one distinct failed rank"):
        broadcaster.pre_failover(1)

    _wait_until(
        lambda: any(send.message_kind == _ABORT_MESSAGE for send in comm.sends),
        description="terminal ABORT relay",
    )
    for send in comm.sends:
        if send.message_kind == _ABORT_MESSAGE:
            send.request.complete()
    comm.deliver(source=1, failed_rank=1, message_kind=_ABORT_MESSAGE)
    _wait_until(broadcaster._progress_failed.is_set, description="ABORT convergence")

    assert health.get_failed_ranks() == frozenset({2})
    assert comm.abort_calls == 0
    assert comm.revoke_calls == 0
    broadcaster.stop()


def test_remote_conflicting_failure_relays_abort_before_world_abort() -> None:
    comm = _FakeComm(size=3, ulfm_probe_error=NotImplementedError())
    broadcaster, health, _, _ = _make_broadcaster(size=3, comm=comm)
    broadcaster.start()
    assert broadcaster.pre_failover(2) is True
    _wait_until(
        lambda: any(send.message_kind == _FAILURE_MESSAGE for send in comm.sends),
        description="first failure fanout",
    )
    _wait_until(lambda: comm.receive_count(1) == 1, description="conflicting report receive")
    comm.deliver(source=1, failed_rank=1)

    _wait_until(
        lambda: any(send.message_kind == _ABORT_MESSAGE for send in comm.sends),
        description="remote-conflict ABORT relay",
    )
    assert comm.abort_calls == 0
    for send in comm.sends:
        if send.message_kind == _ABORT_MESSAGE:
            send.request.complete()
    _wait_until(lambda: comm.receive_count(1) == 2, description="survivor ABORT receive")
    comm.deliver(source=1, failed_rank=1, message_kind=_ABORT_MESSAGE)
    _wait_until(broadcaster._progress_failed.is_set, description="remote-conflict convergence")

    assert health.get_failed_ranks() == frozenset({2})
    assert isinstance(broadcaster.last_error, RuntimeError)
    assert "exactly one distinct failed rank" in str(broadcaster.last_error)
    assert comm.abort_calls == 0
    broadcaster.stop()


def test_reconciled_terminal_abort_at_deadline_does_not_world_abort() -> None:
    comm = _FakeComm(size=1, ulfm_probe_error=NotImplementedError())
    broadcaster, _, _, _ = _make_broadcaster(size=1, comm=comm)
    broadcaster._request_terminal_abort(RuntimeError("terminal"), -1, 0)
    broadcaster._drain_outbound_reports([])
    with broadcaster._protocol_lock:
        broadcaster._abort_deadline = time.monotonic() - 1.0

    assert broadcaster._progress_terminal_abort() is True
    assert comm.abort_calls == 0
    broadcaster.stop()


def test_monitor_does_not_world_abort_reconciled_relay_when_mpi_test_resumes() -> None:
    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.5,
        stop_timeout_sec=0.5,
        reconcile_timeout_sec=0.2,
        abort_timeout_sec=0.2,
    )
    blocking_request = _BlockingRequest()
    comm = _FakeComm(
        size=2,
        receive_factory=lambda _source: blocking_request,
        ulfm_probe_error=NotImplementedError(),
    )
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(2),
        DetectedRankState(2),
        config,
        comm=comm,
        mpi_module=_FakeMPI(),
    )
    broadcaster.start()
    _wait_until(blocking_request.entered_test.is_set, description="blocked MPI progress")
    broadcaster._request_terminal_abort(RuntimeError("terminal"), -1, 0)
    with broadcaster._protocol_lock:
        broadcaster._abort_reporters.add(1)
        broadcaster._abort_fanout_posted = True
        broadcaster._abort_fanout_complete = True
    broadcaster._deadline_monitor_wake_event.set()

    _wait_until(
        lambda: (
            broadcaster._deadline_monitor_thread is not None
            and not broadcaster._deadline_monitor_thread.is_alive()
        ),
        description="monitor-observed terminal reconciliation",
    )
    assert broadcaster._progress_failed.is_set() is False
    assert comm.abort_calls == 0

    blocking_request.release_test.set()
    _wait_until(broadcaster._progress_failed.is_set, description="clean terminal progress stop")
    broadcaster.stop()
    assert comm.abort_calls == 0


def test_terminal_abort_waits_until_its_relay_completes() -> None:
    comm = _FakeComm(size=3, ulfm_probe_error=NotImplementedError())
    broadcaster, _, _, _ = _make_broadcaster(size=3, comm=comm)
    broadcaster._handle_received_abort(-1, 1)
    broadcaster._handle_received_abort(-1, 2)

    assert broadcaster._progress_terminal_abort() is False
    pending_sends: list[ep_failure_broadcast._PendingSend] = []
    broadcaster._drain_outbound_reports(pending_sends)
    assert {send.destination for send in comm.sends} == {1, 2}
    broadcaster._progress_sends(pending_sends)
    assert broadcaster._progress_terminal_abort() is False

    for send in comm.sends:
        send.request.complete()
    broadcaster._progress_sends(pending_sends)
    assert pending_sends == []
    assert broadcaster._progress_terminal_abort() is True
    assert comm.abort_calls == 0
    broadcaster.stop()


def test_received_terminal_abort_is_relayed_without_mutating_health() -> None:
    comm = _FakeComm(size=3, ulfm_probe_error=NotImplementedError())
    broadcaster, health, _, _ = _make_broadcaster(size=3, comm=comm)
    broadcaster.start()
    _wait_until(lambda: comm.receive_count(1) == 1, description="source 1 receive")
    comm.deliver(source=1, failed_rank=2, message_kind=_ABORT_MESSAGE)

    _wait_until(
        lambda: len([send for send in comm.sends if send.message_kind == _ABORT_MESSAGE]) == 2,
        description="relayed terminal ABORT",
    )
    for send in comm.sends:
        if send.message_kind == _ABORT_MESSAGE:
            send.request.complete()
    _wait_until(lambda: comm.receive_count(2) == 1, description="source 2 receive")
    comm.deliver(source=2, failed_rank=2, message_kind=_ABORT_MESSAGE)
    _wait_until(broadcaster._progress_failed.is_set, description="peer ABORT convergence")

    assert health.all_active() is True
    assert isinstance(broadcaster.last_error, RuntimeError)
    assert broadcaster.world_is_poisoned() is True
    assert comm.abort_calls == 0
    broadcaster.stop()
    assert broadcaster.world_is_poisoned() is True


def test_no_ulfm_abort_timeout_uses_world_abort_before_stop_completes() -> None:
    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.5,
        stop_timeout_sec=0.5,
        reconcile_timeout_sec=0.2,
        abort_timeout_sec=0.02,
    )
    comm = _FakeComm(size=3, ulfm_probe_error=NotImplementedError())
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(3),
        DetectedRankState(3),
        config,
        comm=comm,
        mpi_module=_FakeMPI(),
    )
    broadcaster.start()
    assert broadcaster.pre_failover(2) is True
    with pytest.raises(RuntimeError, match="exactly one distinct failed rank"):
        broadcaster.pre_failover(1)

    # stop_event must not let the progress loop skip the pending ABORT fallback.
    broadcaster.stop(timeout=0.5)
    assert comm.abort_calls == 1
    assert comm.revoke_calls == 0


def test_stop_cannot_bypass_abort_requested_during_blocked_drain() -> None:
    send_entered = threading.Event()
    release_send = threading.Event()

    def blocking_send(_destination: int) -> _FakeRequest:
        send_entered.set()
        release_send.wait()
        return _FakeRequest()

    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.5,
        stop_timeout_sec=0.5,
        reconcile_timeout_sec=0.2,
        abort_timeout_sec=0.02,
    )
    comm = _FakeComm(
        size=3,
        send_factory=blocking_send,
        ulfm_probe_error=NotImplementedError(),
    )
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(3),
        DetectedRankState(3),
        config,
        comm=comm,
        mpi_module=_FakeMPI(),
    )
    broadcaster.start()
    _wait_until(lambda: comm.receive_count(1) == 1, description="terminal receive")
    comm.deliver(source=1, failed_rank=-1, message_kind=_ABORT_MESSAGE)
    _wait_until(send_entered.is_set, description="blocked ABORT drain")

    stop_errors: list[BaseException] = []

    def stop_broadcaster() -> None:
        try:
            broadcaster.stop(timeout=0.5)
        except BaseException as error:
            stop_errors.append(error)

    stop_thread = threading.Thread(target=stop_broadcaster)
    stop_thread.start()
    _wait_until(broadcaster._stop_event.is_set, description="stop during ABORT drain")
    release_send.set()
    stop_thread.join(timeout=1.0)

    assert stop_thread.is_alive() is False
    assert stop_errors == []
    assert comm.abort_calls == 1
    assert broadcaster._progress_failed.is_set()


def test_concurrent_local_and_remote_distinct_failures_accept_only_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcaster, health, comm, _ = _make_broadcaster(size=4)
    broadcaster.start()
    _wait_until(lambda: comm.receive_count(1) == 1, description="remote receive")

    original_claim = broadcaster._claim_failure
    local_claimed = threading.Event()
    release_local_claim = threading.Event()

    def racing_claim(failed_rank: int) -> None:
        original_claim(failed_rank)
        if failed_rank == 2:
            local_claimed.set()
            release_local_claim.wait()

    monkeypatch.setattr(broadcaster, "_claim_failure", racing_claim)
    local_errors: list[BaseException] = []

    def report_local_failure() -> None:
        try:
            broadcaster.pre_failover(2)
        except BaseException as error:
            local_errors.append(error)

    local_thread = threading.Thread(target=report_local_failure)
    local_thread.start()
    _wait_until(local_claimed.is_set, description="local failure claim")
    comm.deliver(source=1, failed_rank=3)
    # The remote handler must wait for the local lifecycle transaction to commit
    # its accepted rank and health generation atomically.
    time.sleep(2 * _TEST_CONFIG.poll_interval_sec)
    assert health.all_active() is True
    release_local_claim.set()
    local_thread.join(timeout=1.0)
    assert local_thread.is_alive() is False

    _wait_until(lambda: broadcaster.last_error is not None, description="second-failure rejection")
    _wait_until(lambda: comm.revoke_calls == 1, description="terminal abort revoke")
    assert len(health.get_failed_ranks()) == 1
    assert health.get_failed_ranks() == frozenset({2})
    assert "exactly one distinct failed rank" in str(broadcaster.last_error)
    assert local_errors == []
    broadcaster.stop()


def test_isend_post_error_to_survivor_is_terminal_without_ulfm() -> None:
    error = _FakeMpiException(999, "send post failed")

    def fail_send(_destination: int) -> _FakeRequest:
        raise error

    comm = _FakeComm(
        size=3,
        send_factory=fail_send,
        ulfm_probe_error=NotImplementedError(),
    )
    broadcaster, _, _, _ = _make_broadcaster(size=3, comm=comm)
    broadcaster.start()
    assert broadcaster.pre_failover(2) is True

    _wait_until(lambda: broadcaster.last_error is error, description="terminal Isend error")
    assert broadcaster.health_is_reconciled() is False
    assert comm.revoke_calls == 0
    assert comm.sends == ()
    broadcaster.stop()


def test_send_test_error_is_terminal_and_revokes_with_ulfm() -> None:
    error = _FakeMpiException(999, "send Test failed")
    failed_request = _FakeRequest()
    failed_request.fail(error)
    comm = _FakeComm(size=3, send_factory=lambda _destination: failed_request)
    broadcaster, _, _, _ = _make_broadcaster(size=3, comm=comm)
    broadcaster.start()
    assert broadcaster.pre_failover(2) is True

    _wait_until(lambda: broadcaster.last_error is error, description="terminal send Test error")
    _wait_until(lambda: comm.revoke_calls == 1, description="ULFM emergency revoke")
    assert broadcaster.health_is_reconciled() is False
    assert failed_request.cancel_calls == 0
    assert broadcaster._retained_requests
    broadcaster.stop()


def test_received_report_is_idempotent_and_callback_runs_once() -> None:
    callbacks: list[tuple[int, int, float]] = []
    broadcaster, health, comm, _ = _make_broadcaster(
        size=3,
        callback=lambda failed, source, when: callbacks.append((failed, source, when)),
    )
    broadcaster.start()
    try:
        _wait_until(lambda: comm.receive_count(1) == 1, description="initial receive")
        comm.deliver(source=1, failed_rank=2)
        _wait_until(lambda: health.is_active(2) is False, description="received failure")
        _wait_until(lambda: comm.receive_count(1) == 2, description="reposted receive")

        comm.deliver(source=1, failed_rank=2)
        _wait_until(lambda: comm.receive_count(1) == 3, description="duplicate receive progress")
        _wait_until(lambda: len(callbacks) == 1, description="failure callback")

        assert health.generation == 1
        assert len(callbacks) == 1
        failed_rank, source, event_time = callbacks[0]
        assert (failed_rank, source) == (2, 1)
        assert isinstance(event_time, float)
        _wait_until(lambda: len(comm.sends) == 1, description="failure relay")
        comm.sends[0].request.complete()
        _wait_until(broadcaster.health_is_reconciled, description="failure reconciliation")
    finally:
        broadcaster.stop()


def test_monitor_cannot_observe_remote_claim_before_detected_state_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcaster, health, comm, _ = _make_broadcaster(size=3)
    original_claim = broadcaster._claim_failure
    claim_entered = threading.Event()
    release_claim = threading.Event()

    def claim_then_pause(failed_rank: int) -> None:
        original_claim(failed_rank)
        claim_entered.set()
        release_claim.wait()

    monkeypatch.setattr(broadcaster, "_claim_failure", claim_then_pause)
    broadcaster.start()
    try:
        comm.deliver(source=1, failed_rank=2)
        _wait_until(claim_entered.is_set, description="paused remote failure claim")
        time.sleep(3 * _TEST_CONFIG.poll_interval_sec)

        assert health.is_active(2) is True
        assert broadcaster.last_error is None
        assert comm.revoke_calls == 0

        release_claim.set()
        _wait_until(lambda: not health.is_active(2), description="recorded remote detection")
        _wait_until(lambda: len(comm.sends) == 1, description="remote failure echo")
        comm.sends[0].request.complete()
        _wait_until(broadcaster.health_is_reconciled, description="failure reconciliation")
    finally:
        release_claim.set()
        broadcaster.stop()


def test_blocking_callback_does_not_block_mpi_progress_and_stop_is_bounded() -> None:
    callback_entered = threading.Event()
    release_callback = threading.Event()

    def blocking_callback(_failed: int, _source: int, _when: float) -> None:
        callback_entered.set()
        release_callback.wait()

    broadcaster, health, comm, _ = _make_broadcaster(
        size=3,
        callback=blocking_callback,
    )
    broadcaster.start()
    _wait_until(lambda: comm.receive_count(1) == 1, description="initial receive")
    comm.deliver(source=1, failed_rank=2)
    _wait_until(callback_entered.is_set, description="blocking callback")
    _wait_until(lambda: len(comm.sends) == 1, description="failure relay")
    comm.sends[0].request.complete()

    # The MPI thread must repost and continue polling while user telemetry is blocked.
    _wait_until(lambda: comm.receive_count(1) == 2, description="receive repost")
    comm.deliver(source=1, failed_rank=2)
    _wait_until(lambda: comm.receive_count(1) == 3, description="duplicate receive progress")
    assert health.get_failed_ranks() == frozenset({2})

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="callback thread did not stop"):
        broadcaster.stop(timeout=0.02)
    assert time.monotonic() - start < 0.2
    assert broadcaster.last_error is None
    assert broadcaster.world_is_poisoned() is True

    release_callback.set()
    broadcaster.stop(timeout=0.5)


def test_invalid_received_rank_is_ignored_and_receive_is_reposted() -> None:
    callbacks: list[tuple[int, int, float]] = []
    broadcaster, health, comm, _ = _make_broadcaster(
        size=3,
        callback=lambda failed, source, when: callbacks.append((failed, source, when)),
    )
    broadcaster.start()
    try:
        _wait_until(lambda: comm.receive_count(1) == 1, description="initial receive")
        comm.deliver(source=1, failed_rank=99)
        _wait_until(lambda: comm.receive_count(1) == 2, description="receive after invalid payload")

        assert health.all_active() is True
        assert callbacks == []
        assert broadcaster.last_error is None
    finally:
        broadcaster.stop()


def test_single_rank_group_has_no_peer_requests_or_sends() -> None:
    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.5,
        stop_timeout_sec=0.5,
        reconcile_timeout_sec=0.02,
        abort_timeout_sec=0.02,
    )
    comm = _FakeComm(size=1, ulfm_probe_error=NotImplementedError())
    health = DetectedRankState(1)
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(1),
        health,
        config,
        comm=comm,
        mpi_module=_FakeMPI(),
    )
    broadcaster.start()
    try:
        assert broadcaster.broadcast_failure(0) is True
        _wait_until(broadcaster._outbound_reports.empty, description="empty single-rank report")
        assert health.get_failed_ranks() == frozenset({0})
        assert broadcaster.world_is_poisoned() is True
        assert broadcaster.failure_is_reconciled(0) is False
        assert broadcaster.health_is_reconciled() is False
        assert comm.sends == ()
        assert comm.receive_count(0) == 0
        _wait_until(
            lambda: isinstance(broadcaster.last_error, TimeoutError),
            description="empty-survivor terminal handling",
        )
    finally:
        broadcaster.stop()


def test_lifecycle_is_not_restartable_and_stop_before_start_is_terminal() -> None:
    stopped, _, _, _ = _make_broadcaster(size=1)
    stopped.stop()
    stopped.stop()
    with pytest.raises(RuntimeError, match="cannot start from stopped"):
        stopped.start()

    running, _, _, _ = _make_broadcaster(size=1)
    running.start()
    try:
        with pytest.raises(RuntimeError, match="cannot start from running"):
            running.start()
    finally:
        running.stop()
    with pytest.raises(RuntimeError, match="cannot start from stopped"):
        running.start()


@pytest.mark.parametrize(
    "failing_thread_name",
    [
        "wide-ep-ft-callback",
        "wide-ep-ft-deadline-monitor",
        "wide-ep-ft-broadcast",
    ],
    ids=["callback", "deadline-monitor", "progress"],
)
def test_thread_start_failure_is_terminal_and_stoppable(
    monkeypatch: pytest.MonkeyPatch,
    failing_thread_name: str,
) -> None:
    registry = ep_failure_broadcast._PROCESS_LIFETIME_REFS
    initial_registry_size = len(registry)
    original_start = threading.Thread.start
    start_error = OSError(f"cannot start {failing_thread_name}")

    def failing_start(thread: threading.Thread) -> None:
        if thread.name == failing_thread_name:
            raise start_error
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", failing_start)
    comm = _FakeComm(size=2)
    broadcaster, _, _, _ = _make_broadcaster(
        size=2,
        comm=comm,
        callback=lambda _failed, _source, _when: None,
    )

    try:
        with pytest.raises(RuntimeError, match="thread failed to start") as raised:
            broadcaster.start()

        assert raised.value.__cause__ is start_error
        assert broadcaster.last_error is start_error
        assert broadcaster._progress_failed.is_set()
        assert broadcaster._thread is None
        assert comm.revoke_calls == 1
        if failing_thread_name == "wide-ep-ft-callback":
            assert broadcaster._callback_thread is None
            assert broadcaster._deadline_monitor_thread is None
        elif failing_thread_name == "wide-ep-ft-deadline-monitor":
            assert broadcaster._callback_thread is not None
            assert broadcaster._callback_thread.is_alive() is False
            assert broadcaster._deadline_monitor_thread is None
        else:
            assert broadcaster._callback_thread is not None
            assert broadcaster._callback_thread.is_alive() is False
            assert broadcaster._deadline_monitor_thread is not None
            assert broadcaster._deadline_monitor_thread.is_alive() is False

        # A failed start must not leave an unstarted Thread object that stop()
        # later attempts to join.
        broadcaster.stop()
    finally:
        del registry[initial_registry_size:]


def test_concurrent_stop_prevents_start_from_publishing_running() -> None:
    receive_entered = threading.Event()
    release_receive = threading.Event()

    def blocking_receive(_source: int) -> _FakeRequest:
        receive_entered.set()
        release_receive.wait()
        return _FakeRequest()

    comm = _FakeComm(size=2, receive_factory=blocking_receive)
    broadcaster, _, _, _ = _make_broadcaster(size=2, comm=comm)
    start_errors: list[BaseException] = []
    stop_errors: list[BaseException] = []

    def start_broadcaster() -> None:
        try:
            broadcaster.start()
        except BaseException as error:
            start_errors.append(error)

    def stop_broadcaster() -> None:
        try:
            broadcaster.stop(timeout=0.5)
        except BaseException as error:
            stop_errors.append(error)

    start_thread = threading.Thread(target=start_broadcaster)
    start_thread.start()
    _wait_until(receive_entered.is_set, description="blocked startup receive")
    stop_thread = threading.Thread(target=stop_broadcaster)
    stop_thread.start()
    _wait_until(broadcaster._stop_event.is_set, description="concurrent stop")
    release_receive.set()

    start_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)
    assert start_thread.is_alive() is False
    assert stop_thread.is_alive() is False
    assert len(start_errors) == 1
    assert "did not reach running state" in str(start_errors[0])
    assert stop_errors == []
    assert broadcaster.health_is_reconciled() is False


def test_startup_failure_publishes_error_before_failed_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comm = _FakeComm(size=1)
    broadcaster, _, _, _ = _make_broadcaster(size=1, comm=comm)
    error = _FakeMpiException(999, "forced startup failure")
    record_entered = threading.Event()
    release_record = threading.Event()
    original_record_error = broadcaster._record_error

    def blocked_record_error(recorded_error: BaseException) -> None:
        record_entered.set()
        release_record.wait()
        original_record_error(recorded_error)

    def failing_progress_loop() -> None:
        broadcaster._fail_closed_immediately(error)
        broadcaster._ready_event.set()

    monkeypatch.setattr(broadcaster, "_record_error", blocked_record_error)
    monkeypatch.setattr(broadcaster, "_progress_loop", failing_progress_loop)
    start_errors: list[BaseException] = []

    def start_broadcaster() -> None:
        try:
            broadcaster.start()
        except BaseException as start_error:
            start_errors.append(start_error)

    start_thread = threading.Thread(target=start_broadcaster)
    start_thread.start()
    _wait_until(record_entered.is_set, description="blocked terminal publication")
    assert broadcaster._progress_failed.is_set() is False
    assert broadcaster.last_error is None
    release_record.set()
    start_thread.join(timeout=1.0)

    assert start_thread.is_alive() is False
    assert len(start_errors) == 1
    assert "failed during startup" in str(start_errors[0])
    assert broadcaster.last_error is error
    assert broadcaster._progress_failed.is_set()
    assert comm.revoke_calls == 1


def test_startup_timeout_fails_closed_before_progress_resumes() -> None:
    receive_entered = threading.Event()
    release_receive = threading.Event()

    def blocking_receive(_source: int) -> _FakeRequest:
        receive_entered.set()
        release_receive.wait()
        return _FakeRequest()

    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.02,
        stop_timeout_sec=0.5,
        reconcile_timeout_sec=0.2,
        abort_timeout_sec=0.02,
    )
    comm = _FakeComm(
        size=2,
        receive_factory=blocking_receive,
        ulfm_probe_error=NotImplementedError(),
    )
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(2),
        DetectedRankState(2),
        config,
        comm=comm,
        mpi_module=_FakeMPI(),
    )
    start_errors: list[BaseException] = []

    def start_broadcaster() -> None:
        try:
            broadcaster.start()
        except BaseException as error:
            start_errors.append(error)

    start_thread = threading.Thread(target=start_broadcaster)
    start_thread.start()
    _wait_until(receive_entered.is_set, description="blocked startup")
    start_thread.join(timeout=1.0)

    assert start_thread.is_alive() is False
    assert len(start_errors) == 1
    assert isinstance(start_errors[0], TimeoutError)
    assert "did not start" in str(start_errors[0])
    assert broadcaster._abort_requested.is_set() is False
    assert isinstance(broadcaster.last_error, TimeoutError)
    # The progress thread is still blocked in Irecv, so the caller thread must
    # execute the no-ULFM fail-stop fallback before start() returns.
    assert release_receive.is_set() is False
    assert comm.abort_calls == 1
    release_receive.set()
    broadcaster.stop(timeout=0.5)
    assert comm.abort_calls == 1


def test_startup_timeout_stops_callback_before_blocked_receive_is_released() -> None:
    receive_entered = threading.Event()
    release_receive = threading.Event()

    def blocking_receive(_source: int) -> _FakeRequest:
        receive_entered.set()
        release_receive.wait()
        return _FakeRequest()

    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.02,
        stop_timeout_sec=0.5,
        reconcile_timeout_sec=0.2,
        unattributed_error_timeout_sec=0.2,
        abort_timeout_sec=0.02,
    )
    comm = _FakeComm(
        size=2,
        receive_factory=blocking_receive,
        ulfm_probe_error=NotImplementedError(),
    )
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(2),
        DetectedRankState(2),
        config,
        lambda _failed, _source, _when: None,
        comm=comm,
        mpi_module=_FakeMPI(),
    )

    try:
        with pytest.raises(TimeoutError, match="did not start"):
            broadcaster.start()
        assert receive_entered.is_set()
        assert release_receive.is_set() is False
        assert broadcaster._callback_thread is not None
        assert broadcaster._callback_thread.is_alive() is False
        assert comm.abort_calls == 1
    finally:
        release_receive.set()
        broadcaster.stop(timeout=0.5)
    assert comm.abort_calls == 1


def test_reconciliation_gates_fail_closed_outside_running_lifecycle() -> None:
    broadcaster, _, _, _ = _make_broadcaster(size=2)
    assert broadcaster.health_is_reconciled() is False
    assert broadcaster.failure_is_reconciled(1) is False

    broadcaster.start()
    assert broadcaster.pre_failover(1) is True
    _wait_until(
        lambda: broadcaster.failure_is_reconciled(1),
        description="single-survivor reconciliation",
    )
    assert broadcaster.health_is_reconciled() is True

    broadcaster.stop()
    assert broadcaster.failure_is_reconciled(1) is False
    assert broadcaster.health_is_reconciled() is False


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_stop_requires_positive_finite_timeout(timeout: float) -> None:
    broadcaster, _, _, _ = _make_broadcaster(size=1)
    with pytest.raises(ValueError, match="finite and > 0"):
        broadcaster.stop(timeout)
    broadcaster.start()
    broadcaster.stop()


def test_stop_is_bounded_when_mpi_test_is_stuck() -> None:
    blocking_request = _BlockingRequest()
    comm = _FakeComm(size=2, receive_factory=lambda _source: blocking_request)
    broadcaster, _, _, _ = _make_broadcaster(size=2, comm=comm)
    broadcaster.start()
    _wait_until(blocking_request.entered_test.is_set, description="blocked MPI Test")

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="did not stop"):
        broadcaster.stop(timeout=0.01)
    assert time.monotonic() - start < 0.2
    assert blocking_request.release_test.is_set() is False
    assert comm.revoke_calls == 1
    assert comm.abort_calls == 0

    blocking_request.release_test.set()
    broadcaster.stop(timeout=0.5)
    assert comm.revoke_calls == 1
    assert comm.abort_calls == 0


def test_broadcast_rejects_report_after_stop_begins() -> None:
    blocking_request = _BlockingRequest()
    comm = _FakeComm(size=2, receive_factory=lambda _source: blocking_request)
    broadcaster, health, _, _ = _make_broadcaster(size=2, comm=comm)
    broadcaster.start()
    _wait_until(blocking_request.entered_test.is_set, description="blocked MPI Test")

    stop_errors: list[BaseException] = []

    def stop_broadcaster() -> None:
        try:
            broadcaster.stop(timeout=0.5)
        except BaseException as error:
            stop_errors.append(error)

    stop_thread = threading.Thread(target=stop_broadcaster)
    stop_thread.start()
    _wait_until(broadcaster._stop_event.is_set, description="stop transition")

    with pytest.raises(RuntimeError, match="state=stopping"):
        broadcaster.broadcast_failure(1)
    assert health.all_active() is True

    blocking_request.release_test.set()
    stop_thread.join(timeout=1.0)
    assert stop_thread.is_alive() is False
    assert stop_errors == []


def test_healthy_stop_cancels_preposted_receives() -> None:
    broadcaster, health, comm, _ = _make_broadcaster(size=3)
    broadcaster.start()
    _wait_until(
        lambda: comm.receive_count(1) == 1 and comm.receive_count(2) == 1,
        description="preposted receives",
    )
    receives = [comm.latest_receive(source).request for source in (1, 2)]

    assert health.all_active() is True
    broadcaster.stop()

    assert [request.cancel_calls for request in receives] == [1, 1]
    assert all(request.Test() is True for request in receives)
    assert broadcaster._retained_requests == []


def test_healthy_stop_cleanup_timeout_aborts_world_and_retains_request() -> None:
    request = _NonCompletingCancelRequest()
    comm = _FakeComm(size=2, receive_factory=lambda _source: request)
    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.5,
        stop_timeout_sec=0.05,
        reconcile_timeout_sec=0.2,
    )
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(2),
        DetectedRankState(2),
        config,
        comm=comm,
        mpi_module=_FakeMPI(),
    )
    broadcaster.start()
    _wait_until(lambda: comm.receive_count(1) == 1, description="preposted receive")

    start = time.monotonic()
    broadcaster.stop()
    assert time.monotonic() - start < 0.2
    assert request.cancel_calls == 1
    assert isinstance(broadcaster.last_error, TimeoutError)
    assert broadcaster._retained_requests
    assert comm.abort_calls == 1
    assert comm.revoke_calls == 0


def test_poisoned_stop_keeps_progressing_until_failure_is_reconciled() -> None:
    broadcaster, health, comm, _ = _make_broadcaster(size=3)
    broadcaster.start()
    _wait_until(
        lambda: comm.receive_count(1) == 1 and comm.receive_count(2) == 1,
        description="preposted receives",
    )
    receives = [comm.latest_receive(source).request for source in (1, 2)]
    assert broadcaster.broadcast_failure(2) is True
    _wait_until(lambda: len(comm.sends) == 1, description="pending failure send")
    send = comm.sends[0]

    assert health.get_failed_ranks() == frozenset({2})
    initial_test_calls = send.request.test_calls
    stop_errors: list[BaseException] = []

    def stop_broadcaster() -> None:
        try:
            broadcaster.stop()
        except BaseException as error:
            stop_errors.append(error)

    stop_thread = threading.Thread(target=stop_broadcaster)
    stop_thread.start()
    _wait_until(broadcaster._stop_event.is_set, description="stop request")
    _wait_until(
        lambda: send.request.test_calls > initial_test_calls,
        description="continued send progress during stop",
    )
    assert stop_thread.is_alive() is True

    send.request.complete()
    comm.deliver(source=1, failed_rank=2)
    stop_thread.join(timeout=1.0)

    assert stop_thread.is_alive() is False
    assert stop_errors == []
    assert [request.cancel_calls for request in receives] == [0, 0]
    assert send.request.cancel_calls == 0
    assert broadcaster.last_error is None
    assert broadcaster._retained_requests


def test_poisoned_resources_outlive_broadcaster_object() -> None:
    registry = ep_failure_broadcast._PROCESS_LIFETIME_REFS
    initial_registry_size = len(registry)

    def retain_then_drop_owner() -> tuple[
        weakref.ReferenceType[_FakeRequest],
        weakref.ReferenceType[np.ndarray],
        weakref.ReferenceType[_FakeComm],
    ]:
        broadcaster, _, comm, _ = _make_broadcaster(size=3)
        broadcaster.start()
        _wait_until(
            lambda: comm.receive_count(1) == 1 and comm.receive_count(2) == 1,
            description="preposted receives",
        )
        assert broadcaster.pre_failover(2) is True
        _wait_until(lambda: len(comm.sends) == 1, description="pending send")
        send = comm.sends[0]
        send.request.complete()
        comm.deliver(source=1, failed_rank=2)
        _wait_until(broadcaster.health_is_reconciled, description="failure reconciliation")
        retained_receive = comm.latest_receive(2)
        references = (
            weakref.ref(retained_receive.request),
            weakref.ref(retained_receive.buffer),
            weakref.ref(comm),
        )
        broadcaster.stop()
        # The fake communicator owns test bookkeeping references that a real
        # MPI communicator does not. Drop those so only the process-lifetime
        # registry can keep the poisoned request and buffer reachable.
        comm._receives.clear()
        return references

    request_ref, buffer_ref, comm_ref = retain_then_drop_owner()
    gc.collect()
    try:
        assert request_ref() is not None
        assert buffer_ref() is not None
        assert comm_ref() is not None
        assert len(registry) > initial_registry_size
    finally:
        del registry[initial_registry_size:]


def test_startup_failure_retains_poisoned_comm_without_pending_requests() -> None:
    registry = ep_failure_broadcast._PROCESS_LIFETIME_REFS
    initial_registry_size = len(registry)

    def fail_first_receive(_source: int) -> _FakeRequest:
        raise _FakeMpiException(999, "first Irecv failed")

    def start_then_drop_owner() -> weakref.ReferenceType[_FakeComm]:
        comm = _FakeComm(size=2, receive_factory=fail_first_receive)
        comm_ref = weakref.ref(comm)
        broadcaster, _, _, _ = _make_broadcaster(size=2, comm=comm)
        with pytest.raises(RuntimeError, match="failed during startup"):
            broadcaster.start()
        assert broadcaster._retained_requests == []
        assert isinstance(broadcaster.last_error, _FakeMpiException)
        return comm_ref

    comm_ref = start_then_drop_owner()
    gc.collect()
    try:
        assert comm_ref() is not None
        assert len(registry) > initial_registry_size
        assert any(reference is comm_ref() for reference in registry[initial_registry_size:])
    finally:
        del registry[initial_registry_size:]


def test_constructor_requires_mpi_thread_multiple() -> None:
    mpi = _FakeMPI(thread_level=_FakeMPI.THREAD_MULTIPLE - 1)
    comm = _FakeComm(size=2)
    with pytest.raises(RuntimeError, match="MPI.THREAD_MULTIPLE"):
        MpiFtSubcomm(
            _FakeMapping.ep_world(2),
            DetectedRankState(2),
            _TEST_CONFIG,
            comm=comm,
            mpi_module=mpi,
        )
    assert comm.errhandler is None


def test_collectively_created_comm_uses_frozen_setup_and_lives_for_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ep_failure_broadcast._PROCESS_LIFETIME_REFS
    initial_registry_size = len(registry)
    comm = _FakeComm(size=2)
    mpi = _FakeMPI()

    class SingleReadMapping:
        def __init__(self) -> None:
            self.rank_reads = 0

        @property
        def moe_ep_rank(self) -> int:
            self.rank_reads += 1
            if self.rank_reads > 1:
                raise RuntimeError("mapping rank was read after collective validation")
            return 0

    mapping = SingleReadMapping()

    def create_mpi_ft_subcomm(
        setup_mapping: SingleReadMapping,
        *,
        health_size: int,
    ) -> types.SimpleNamespace:
        assert health_size == 2
        local_rank = setup_mapping.moe_ep_rank
        comm.Set_errhandler(mpi.ERRORS_RETURN)
        return types.SimpleNamespace(
            comm=comm,
            local_rank=local_rank,
            ep_size=2,
            ulfm_available=False,
        )

    distributed = types.ModuleType("tensorrt_llm._torch.distributed")
    distributed.__path__ = []
    communicator = types.ModuleType("tensorrt_llm._torch.distributed.communicator")
    communicator.create_mpi_ft_subcomm = create_mpi_ft_subcomm
    distributed.communicator = communicator
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.distributed", distributed)
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.distributed.communicator",
        communicator,
    )

    try:
        broadcaster = MpiFtSubcomm(
            mapping,
            DetectedRankState(2),
            _TEST_CONFIG,
            mpi_module=mpi,
        )

        assert broadcaster._comm is comm
        assert broadcaster._local_rank == 0
        assert broadcaster._ep_size == 2
        assert mapping.rank_reads == 1
        assert comm.errhandler_calls == 1
        assert comm.is_revoked_calls == 0
        assert any(reference is comm for reference in registry[initial_registry_size:])
        broadcaster.stop()
        del broadcaster
        gc.collect()
        assert any(reference is comm for reference in registry[initial_registry_size:])
    finally:
        del registry[initial_registry_size:]


def test_constructor_rejects_non_world_spanning_ep_group_for_mvp() -> None:
    mapping = _FakeMapping(
        world_size=4,
        rank=0,
        moe_ep_size=2,
        moe_ep_rank=0,
        moe_ep_group=(0, 1),
    )
    comm = _FakeComm(size=2)
    with pytest.raises(ValueError, match="spanning the full MPI world"):
        MpiFtSubcomm(
            mapping,
            DetectedRankState(2),
            _TEST_CONFIG,
            comm=comm,
            mpi_module=_FakeMPI(),
        )
    assert comm.errhandler is None


@pytest.mark.parametrize(
    "probe_error",
    [
        NotImplementedError("ULFM not compiled"),
        _FakeMpiException(_FakeMPI.ERR_UNSUPPORTED_OPERATION),
    ],
    ids=["not-implemented", "unsupported-operation"],
)
def test_ulfm_probe_falls_back_when_runtime_does_not_support_it(
    probe_error: BaseException,
) -> None:
    comm = _FakeComm(size=2, ulfm_probe_error=probe_error)
    broadcaster, _, _, _ = _make_broadcaster(size=2, comm=comm)
    assert broadcaster.ulfm_available is False
    assert comm.is_revoked_calls == 1
    broadcaster.stop()


def test_ulfm_probe_propagates_unrelated_mpi_error() -> None:
    registry = ep_failure_broadcast._PROCESS_LIFETIME_REFS
    initial_registry_size = len(registry)
    error = _FakeMpiException(_FakeMPI.ERR_UNKNOWN, "corrupt communicator")
    comm = _FakeComm(size=2, ulfm_probe_error=error)
    try:
        with pytest.raises(_FakeMpiException, match="corrupt communicator") as raised:
            _make_broadcaster(size=2, comm=comm)
        assert raised.value is error
        assert any(reference is comm for reference in registry[initial_registry_size:])
    finally:
        del registry[initial_registry_size:]


def test_ulfm_probe_retains_original_error_when_error_class_inspection_raises() -> None:
    registry = ep_failure_broadcast._PROCESS_LIFETIME_REFS
    initial_registry_size = len(registry)

    class UnclassifiableMpiError(_FakeMpiException):
        def Get_error_class(self) -> int:
            raise RuntimeError("error-class lookup failed")

    probe_error = UnclassifiableMpiError(
        _FakeMPI.ERR_UNSUPPORTED_OPERATION,
        "unclassifiable ULFM probe failure",
    )
    comm = _FakeComm(size=2, ulfm_probe_error=probe_error)
    try:
        with pytest.raises(UnclassifiableMpiError, match="unclassifiable") as raised:
            _make_broadcaster(size=2, comm=comm)
        assert raised.value is probe_error
        assert any(reference is comm for reference in registry[initial_registry_size:])
    finally:
        del registry[initial_registry_size:]


def test_constructor_rejects_and_retains_already_revoked_comm() -> None:
    registry = ep_failure_broadcast._PROCESS_LIFETIME_REFS
    initial_registry_size = len(registry)

    def construct_then_drop_owner() -> weakref.ReferenceType[_FakeComm]:
        comm = _FakeComm(size=2, ulfm_revoked=True)
        comm_ref = weakref.ref(comm)
        with pytest.raises(RuntimeError, match="already revoked"):
            _make_broadcaster(size=2, comm=comm)
        return comm_ref

    comm_ref = construct_then_drop_owner()
    gc.collect()
    try:
        assert comm_ref() is not None
        assert len(registry) > initial_registry_size
    finally:
        del registry[initial_registry_size:]


def test_peer_failure_reconciles_without_revoking_control_plane() -> None:
    callbacks: list[tuple[int, int, float]] = []
    broadcaster, health, comm, _ = _make_broadcaster(
        size=3,
        callback=lambda failed, source, when: callbacks.append((failed, source, when)),
    )
    assert broadcaster.ulfm_available is True
    broadcaster.start()
    try:
        _wait_until(lambda: comm.receive_count(1) == 1, description="peer receive")
        comm.latest_receive(1).request.fail(_FakeMpiException(_FakeMPI.ERR_PROC_FAILED))

        _wait_until(lambda: health.is_active(1) is False, description="peer failure classification")
        _wait_until(lambda: len(comm.sends) == 1, description="peer failure relay")

        assert health.get_failed_ranks() == frozenset({1})
        _wait_until(lambda: len(callbacks) == 1, description="failed-peer callback")
        assert [(failed, source) for failed, source, _ in callbacks] == [(1, 1)]
        assert comm.sends[0].destination == 2
        assert comm.sends[0].message_kind == _FAILURE_MESSAGE
        assert comm.sends[0].payload == 1
        assert comm.revoke_calls == 0
        assert broadcaster.last_error is None

        comm.sends[0].request.complete()
        comm.deliver(source=2, failed_rank=1)
        _wait_until(broadcaster.health_is_reconciled, description="failure reconciliation")
        _wait_until(lambda: comm.receive_count(2) == 2, description="reposted receive")
        assert broadcaster.health_is_reconciled() is True
        assert broadcaster.last_error is None
        assert comm.revoke_calls == 0
    finally:
        broadcaster.stop()


def test_remote_revoke_after_local_reconciliation_still_fails_closed() -> None:
    error = _FakeMpiException(_FakeMPI.ERR_REVOKED, "remote revoke")
    comm = _FakeComm(size=3)
    broadcaster, _, _, _ = _make_broadcaster(size=3, comm=comm)
    broadcaster.start()
    assert broadcaster.pre_failover(2) is True
    _wait_until(lambda: len(comm.sends) == 1, description="failure relay")
    comm.sends[0].request.complete()
    _wait_until(lambda: comm.receive_count(1) == 1, description="initial peer receive")
    comm.deliver(source=1, failed_rank=2)
    _wait_until(broadcaster.health_is_reconciled, description="local reconciliation")
    _wait_until(
        lambda: comm.receive_count(1) == 2,
        description="receive after failure report",
    )

    comm.latest_receive(1).request.fail(error)
    _wait_until(lambda: broadcaster.last_error is error, description="terminal remote revoke")
    assert broadcaster.health_is_reconciled() is False
    assert comm.revoke_calls == 1
    with pytest.raises(RuntimeError, match="not running"):
        broadcaster.pre_failover(1)
    assert broadcaster._detected_state.get_failed_ranks() == frozenset({2})
    broadcaster.stop()


def test_premature_remote_revoke_fails_closed() -> None:
    error = _FakeMpiException(_FakeMPI.ERR_REVOKED, "premature revoke")
    comm = _FakeComm(size=3)
    broadcaster, _, _, _ = _make_broadcaster(size=3, comm=comm)
    broadcaster.start()
    assert broadcaster.pre_failover(2) is True
    _wait_until(lambda: comm.receive_count(1) == 1, description="initial peer receive")

    comm.latest_receive(1).request.fail(error)
    _wait_until(lambda: broadcaster.last_error is error, description="terminal revoke")
    assert broadcaster.health_is_reconciled() is False
    assert comm.revoke_calls == 1
    broadcaster.stop()


def test_generic_mpi_error_is_terminal_and_revokes_once() -> None:
    comm = _FakeComm(size=2)
    broadcaster, health, _, _ = _make_broadcaster(size=2, comm=comm)
    broadcaster.start()
    try:
        _wait_until(lambda: comm.receive_count(1) == 1, description="peer receive")
        error = _FakeMpiException(999, "generic transport failure")
        comm.latest_receive(1).request.fail(error)

        _wait_until(lambda: broadcaster.last_error is error, description="terminal MPI error")
        _wait_until(lambda: comm.revoke_calls == 1, description="communicator revoke")
        assert health.all_active() is True
        with pytest.raises(RuntimeError, match="broadcaster is not running"):
            broadcaster.broadcast_failure(1)
    finally:
        broadcaster.stop()
    assert comm.revoke_calls == 1


def test_proc_failed_is_classified_without_full_ulfm_support() -> None:
    comm = _FakeComm(size=3, ulfm_probe_error=NotImplementedError())
    broadcaster, health, _, _ = _make_broadcaster(size=3, comm=comm)
    assert broadcaster.ulfm_available is False
    broadcaster.start()
    try:
        _wait_until(lambda: comm.receive_count(1) == 1, description="peer receive")
        comm.latest_receive(1).request.fail(_FakeMpiException(_FakeMPI.ERR_PROC_FAILED))
        _wait_until(lambda: health.is_active(1) is False, description="peer failure classification")
        _wait_until(lambda: len(comm.sends) == 1, description="peer failure relay")

        assert health.get_failed_ranks() == frozenset({1})
        assert comm.sends[0].destination == 2
        assert comm.sends[0].payload == 1
        assert broadcaster.last_error is None
        assert comm.revoke_calls == 0
        comm.sends[0].request.complete()
        comm.deliver(source=2, failed_rank=1)
        _wait_until(broadcaster.health_is_reconciled, description="failure reconciliation")
    finally:
        broadcaster.stop()


def test_non_ulfm_generic_receive_error_keeps_other_sources_live() -> None:
    callbacks: list[tuple[int, int, float]] = []
    comm = _FakeComm(size=3, ulfm_probe_error=NotImplementedError())
    broadcaster, health, _, _ = _make_broadcaster(
        size=3,
        comm=comm,
        callback=lambda failed, source, when: callbacks.append((failed, source, when)),
    )
    broadcaster.start()
    try:
        _wait_until(
            lambda: comm.receive_count(1) == 1 and comm.receive_count(2) == 1,
            description="fixed-source receives",
        )
        failed_receive = comm.latest_receive(1).request
        failed_receive.fail(_FakeMpiException(999, "implementation-specific peer error"))
        _wait_until(
            lambda: len(broadcaster._retained_requests) == 1,
            description="retained failed receive",
        )

        assert health.all_active() is True
        assert broadcaster.last_error is None
        assert failed_receive.cancel_calls == 0

        # Another survivor's rank-attributed detection remains usable even
        # when the dead source's fixed receive failed with a generic error.
        comm.deliver(source=2, failed_rank=1)
        _wait_until(lambda: health.is_active(1) is False, description="detection report")
        _wait_until(lambda: len(comm.sends) == 1, description="failure relay")
        comm.sends[0].request.complete()
        _wait_until(lambda: comm.receive_count(2) == 2, description="live-source receive repost")
        _wait_until(lambda: len(callbacks) == 1, description="detection callback")

        assert [(failed, source) for failed, source, _ in callbacks] == [(1, 2)]
        assert broadcaster.last_error is None
        assert comm.revoke_calls == 0
        _wait_until(broadcaster.health_is_reconciled, description="failure reconciliation")
    finally:
        broadcaster.stop()


def test_non_ulfm_generic_error_after_watchdog_report_does_not_arm_stale_deadline() -> None:
    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.5,
        stop_timeout_sec=0.5,
        reconcile_timeout_sec=0.2,
        unattributed_error_timeout_sec=0.01,
        abort_timeout_sec=0.02,
    )
    comm = _FakeComm(size=3, ulfm_probe_error=NotImplementedError())
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(3),
        DetectedRankState(3),
        config,
        comm=comm,
        mpi_module=_FakeMPI(),
    )
    broadcaster.start()
    try:
        _wait_until(
            lambda: comm.receive_count(1) == 1 and comm.receive_count(2) == 1,
            description="fixed-source receives",
        )
        comm.deliver(source=2, failed_rank=1)
        _wait_until(lambda: len(comm.sends) == 1, description="failure relay")

        # The transport notification can arrive after the watchdog report. It
        # is already attributed to rank 1 and must not create a deadline that
        # aborts an otherwise reconciled survivor set later.
        failed_receive = comm.latest_receive(1).request
        failed_receive.fail(_FakeMpiException(999, "late generic peer error"))
        _wait_until(
            lambda: (
                failed_receive in [pending.request for pending in broadcaster._retained_requests]
            ),
            description="retained explained receive",
        )
        comm.sends[0].request.complete()
        _wait_until(broadcaster.health_is_reconciled, description="failure reconciliation")
        time.sleep(3 * config.unattributed_error_timeout_sec)

        assert broadcaster.last_error is None
        assert broadcaster.health_is_reconciled() is True
        assert comm.abort_calls == 0
    finally:
        broadcaster.stop()


def test_unattributed_error_from_other_source_keeps_reconciliation_gate_closed() -> None:
    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.5,
        stop_timeout_sec=0.5,
        reconcile_timeout_sec=0.5,
        unattributed_error_timeout_sec=0.05,
        abort_timeout_sec=0.02,
    )
    comm = _FakeComm(size=4, ulfm_probe_error=NotImplementedError())
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(4),
        DetectedRankState(4),
        config,
        comm=comm,
        mpi_module=_FakeMPI(),
    )
    broadcaster.start()
    try:
        _wait_until(
            lambda: all(comm.receive_count(source) == 1 for source in (1, 2, 3)),
            description="fixed-source receives",
        )
        comm.deliver(source=1, failed_rank=3)
        _wait_until(lambda: comm.receive_count(1) == 2, description="source 1 receive repost")
        _wait_until(lambda: len(comm.sends) == 2, description="failure fanout")

        # Source 1 already reported rank 3, but a later generic error from
        # source 1 is not explained by that different failed rank.
        comm.latest_receive(1).request.fail(_FakeMpiException(999, "second transport error"))
        _wait_until(
            lambda: bool(broadcaster._unattributed_error_deadlines),
            description="unattributed deadline",
        )
        comm.deliver(source=2, failed_rank=3)
        for send in comm.sends:
            send.request.complete()
        _wait_until(lambda: comm.receive_count(2) == 2, description="source 2 report")
        _wait_until(
            lambda: 3 in broadcaster._failure_fanout_complete,
            description="completed failure fanout",
        )

        assert broadcaster.failure_is_reconciled(3) is False
        assert broadcaster.health_is_reconciled() is False
        _wait_until(
            lambda: isinstance(broadcaster.last_error, TimeoutError),
            description="unattributed transport deadline",
        )
        _wait_until(lambda: comm.abort_calls == 1, description="terminal fallback")
    finally:
        broadcaster.stop()


def test_non_ulfm_unattributed_receive_error_aborts_after_bounded_deadlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MpiFtSubcommConfig(
        poll_interval_sec=0.001,
        startup_timeout_sec=0.5,
        stop_timeout_sec=0.5,
        reconcile_timeout_sec=0.2,
        unattributed_error_timeout_sec=0.01,
        abort_timeout_sec=0.05,
    )
    comm = _FakeComm(size=2, ulfm_probe_error=NotImplementedError())
    health = DetectedRankState(2)
    broadcaster = MpiFtSubcomm(
        _FakeMapping.ep_world(2),
        health,
        config,
        comm=comm,
        mpi_module=_FakeMPI(),
    )
    terminal_request_entered = threading.Event()
    release_terminal_request = threading.Event()
    original_request_terminal_abort = broadcaster._request_terminal_abort

    def blocking_terminal_request(
        error: BaseException,
        payload: int,
        reporter: int,
    ) -> None:
        terminal_request_entered.set()
        assert release_terminal_request.wait(timeout=1.0)
        original_request_terminal_abort(error, payload, reporter)

    monkeypatch.setattr(broadcaster, "_request_terminal_abort", blocking_terminal_request)
    broadcaster.start()
    _wait_until(lambda: comm.receive_count(1) == 1, description="fixed-source receive")
    failed_receive = comm.latest_receive(1).request
    failed_receive.fail(_FakeMpiException(999, "unattributed peer error"))

    _wait_until(terminal_request_entered.is_set, description="blocked terminal publication")
    # The request is deliberately blocked before last_error is recorded. The
    # lifecycle and protocol poison must already make the public gate fail
    # closed, with no lock held that prevents the iteration thread from polling.
    try:
        assert broadcaster.last_error is None
        assert broadcaster._lifecycle is ep_failure_broadcast._Lifecycle.FAILED
        assert broadcaster._unattributed_error_deadlines[1] == float("inf")
        assert broadcaster.health_is_reconciled() is False
    finally:
        release_terminal_request.set()

    _wait_until(
        lambda: isinstance(broadcaster.last_error, TimeoutError),
        description="unattributed transport deadline",
    )
    assert "could not attribute" in str(broadcaster.last_error)
    # Expiry remains protocol-visible for the rest of this communicator epoch.
    assert 1 in broadcaster._unattributed_error_deadlines
    assert broadcaster.health_is_reconciled() is False
    _wait_until(
        lambda: any(send.message_kind == _ABORT_MESSAGE for send in comm.sends),
        description="terminal ABORT relay",
    )
    _wait_until(lambda: comm.abort_calls == 1, description="bounded MPI_Abort fallback")

    assert health.all_active() is True
    assert failed_receive.cancel_calls == 0
    assert comm.revoke_calls == 0
    broadcaster.stop()


def test_send_buffer_lives_until_request_test_completes() -> None:
    comm = _FakeComm(size=3)
    broadcaster, _, _, _ = _make_broadcaster(size=3, comm=comm)
    broadcaster.start()
    try:
        assert broadcaster.broadcast_failure(2) is True
        _wait_until(lambda: len(comm.sends) == 1, description="pending send")
        record = comm.sends[0]
        _wait_until(lambda: record.request.test_calls > 0, description="send Test polling")

        gc.collect()
        assert record.buffer_ref() is not None

        record.request.complete()

        def buffer_released() -> bool:
            gc.collect()
            return record.buffer_ref() is None

        _wait_until(buffer_released, description="completed send buffer release")
        comm.deliver(source=1, failed_rank=2)
        _wait_until(broadcaster.health_is_reconciled, description="failure reconciliation")
    finally:
        broadcaster.stop()
