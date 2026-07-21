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
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation import transceiver as transceiver_module
from tensorrt_llm._torch.disaggregation.async_consensus import (
    ConsensusEvent,
    ConsensusEventKind,
    ConsensusOutcome,
)
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.transfer import TaskStatus, TxSession
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState


@pytest.fixture(autouse=True)
def _clear_async_consensus_env(monkeypatch) -> None:
    monkeypatch.delenv(transceiver_module._ASYNC_TERMINAL_ENV, raising=False)
    monkeypatch.delenv(transceiver_module._ASYNC_PEER_READY_ENV, raising=False)
    monkeypatch.delenv(transceiver_module._DIAG_ENV, raising=False)


@dataclass
class _FakeRequest:
    state: Optional[LlmRequestState] = None
    request_id: int = 0
    py_disaggregated_params: Optional[object] = None


class _FakeTransferWorker:
    def __init__(self) -> None:
        self.sweep_count = 0
        self.ready_request_ids: set[int] = set()
        self.tx_session = None
        self.rx_session = None

    def sweep_stale_req_infos(self) -> None:
        self.sweep_count += 1

    def has_all_peer_req_infos_for_send(self, rid: int) -> bool:
        return rid in self.ready_request_ids

    def create_tx_session(self, _req):
        assert self.tx_session is not None
        return self.tx_session

    def create_rx_session(self, _req):
        assert self.rx_session is not None
        return self.rx_session


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
        seal_quiescent: Optional[bool] = None,
    ) -> None:
        self._rid = rid
        self._wait_result = wait_result
        self._status = status
        self._is_completed = is_completed
        self._has_failed = has_failed
        self._has_transferring_tasks = has_transferring_tasks
        self._seal_quiescent = seal_quiescent
        self.blocking_calls: list[bool] = []
        self.closed = False
        self.sealed = False
        self.cancel_calls = 0

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

    def seal_and_check_quiescent(self) -> bool:
        self.sealed = True
        if self._seal_quiescent is not None:
            return self._seal_quiescent
        return not self._has_transferring_tasks

    def cancel(self) -> None:
        self.cancel_calls += 1
        self._status = SessionStatus.CANCELLED

    def close(self) -> None:
        self.closed = True


class _FakeAsyncCoordinator:
    def __init__(self, *, rank: int = 1, scheduling_rank: int = 0) -> None:
        self.rank = rank
        self.scheduling_rank = scheduling_rank
        self.events: list[ConsensusEvent] = []
        self.terminal_votes: list[tuple[int, ConsensusOutcome, int]] = []
        self.ready_votes: list[tuple[int, int]] = []
        self.ready_acks: list[tuple[int, int]] = []
        self.ready_activation_acks: list[tuple[int, int]] = []
        self.ready_abort_acks: list[tuple[int, int]] = []
        self.ready_withdrawals: list[tuple[int, int]] = []
        self.withdraw_result = True
        self.poll_count = 0
        self.poll_hook: Optional[Callable[["_FakeAsyncCoordinator"], None]] = None

    def poll(self) -> list[ConsensusEvent]:
        self.poll_count += 1
        if self.poll_hook is not None:
            self.poll_hook(self)
        events, self.events = self.events, []
        return events

    def publish_terminal(self, rid: int, outcome: ConsensusOutcome, epoch: int) -> None:
        self.terminal_votes.append((rid, outcome, epoch))

    def publish_ready(self, rid: int, epoch: int) -> None:
        self.ready_votes.append((rid, epoch))

    def acknowledge_ready(self, rid: int, epoch: int) -> None:
        self.ready_acks.append((rid, epoch))

    def acknowledge_ready_activation(self, rid: int, epoch: int) -> None:
        self.ready_activation_acks.append((rid, epoch))

    def acknowledge_ready_abort(self, rid: int, epoch: int) -> None:
        self.ready_abort_acks.append((rid, epoch))

    def withdraw_ready(self, rid: int, epoch: int) -> bool:
        self.ready_withdrawals.append((rid, epoch))
        return self.withdraw_result


class _FakeTask:
    def __init__(self, status: TaskStatus, wait_result: bool = True) -> None:
        self.status = status
        self._wait_result = wait_result
        self.wait_calls: list[Optional[float]] = []

    def wait(self, timeout: Optional[float] = None) -> bool:
        self.wait_calls.append(timeout)
        return self._wait_result


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_transceiver(
    sessions: dict[int, _FakeSession],
    reqs: Optional[dict[int, _FakeRequest]] = None,
) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._send_sessions = sessions
    transceiver._send_reqs = reqs or {rid: _FakeRequest() for rid in sessions}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._wait_reqs = {}
    transceiver._legacy_failed_sessions = set()
    transceiver._context_cancelled_request_ids = []
    transceiver._async_ready_epoch = OrderedDict()
    transceiver._async_ready_published = {}
    transceiver._shutdown = False
    transceiver._shutdown_complete = False
    transceiver._shutdown_sessions_complete = False
    transceiver._shutdown_consensus_complete = False
    transceiver._shutdown_worker_complete = False
    transceiver._shutdown_worker_event = None
    transceiver._shutdown_deferred_errors = []
    transceiver._sender_future_timeout_ms = 123
    transceiver.kv_transfer_poll_interval_ms = 10
    # Attributes read by check_context_transfer_status before it processes sessions.
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._transfer_worker = _FakeTransferWorker()
    transceiver._ctx_consensus = lambda local_ids: list(local_ids)
    transceiver._ctx_consensus_outcome = (
        lambda _to_process,
        _known_ids,
        cancelled,
        cancel_quiescent,
        failed,
        failed_quiescent,
        completed,
        timed_out: (
            cancelled,
            cancel_quiescent,
            failed,
            failed_quiescent,
            completed,
            timed_out,
        )
    )
    return transceiver


def _enable_fake_async_consensus(
    transceiver: KvCacheTransceiverV2,
    *,
    terminal: bool = False,
    peer_ready: bool = False,
) -> _FakeAsyncCoordinator:
    coordinator = _FakeAsyncCoordinator()
    transceiver._async_terminal_consensus_enabled = terminal
    transceiver._async_peer_ready_consensus_enabled = peer_ready
    transceiver._async_consensus = coordinator
    transceiver._async_terminal_epoch = OrderedDict()
    transceiver._async_terminal_published = {}
    transceiver._async_terminal_commits = {}
    transceiver._async_terminal_cancelled = {}
    transceiver._async_ready_epoch = OrderedDict()
    transceiver._async_ready_published = {}
    transceiver._async_ready_prepared = {}
    transceiver._async_ready_released = set()
    transceiver._async_ready_activated = {}
    transceiver._async_ready_acknowledged = set()
    transceiver._async_ready_withdrawn = set()
    transceiver._async_ready_aborted = {}
    transceiver._async_ready_finalized_without_request = OrderedDict()
    transceiver._async_consensus_counters = defaultdict(int)
    transceiver._wait_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._dist = Mock(rank=0)
    return coordinator


class _FakeStartupMpiDist:
    def __init__(self, gather_result=None) -> None:
        self.rank = 0
        self.gather_result = gather_result
        self.descriptors: list = []
        self.pp_values: list = []

    def allgather(self, descriptor):
        self.descriptors.append(descriptor)
        if self.gather_result is None:
            return [descriptor, descriptor]
        return self.gather_result(descriptor)

    def pp_allgather(self, value):
        self.pp_values.append(value)
        return [value, value]


def _make_startup_transceiver(dist: _FakeStartupMpiDist) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._dist = dist
    transceiver._mapping = SimpleNamespace(
        world_size=2,
        tp_size=1,
        pp_size=2,
        cp_size=1,
        enable_attention_dp=False,
        pp_group=(0, 1),
    )
    transceiver._transfer_worker = SimpleNamespace(
        sender_endpoint="local-endpoint",
        populate_instance_and_rank_info=Mock(),
    )
    transceiver._kv_cache_manager = SimpleNamespace(pp_layers=[])
    transceiver._context_info_endpoint = "context-endpoint"
    return transceiver


def test_async_startup_flag_off_preserves_legacy_endpoint_exchange(monkeypatch) -> None:
    monkeypatch.setattr(transceiver_module, "MPIDist", _FakeStartupMpiDist)
    dist = _FakeStartupMpiDist()
    transceiver = _make_startup_transceiver(dist)
    config = SimpleNamespace(backend="NIXL", transceiver_runtime="PYTHON")

    transceiver._init_async_consensus(config)
    assert dist.descriptors == []

    transceiver._exchange_rank_info()

    assert dist.descriptors == ["local-endpoint"]
    transceiver._transfer_worker.populate_instance_and_rank_info.assert_called_once_with(
        endpoints=["local-endpoint", "local-endpoint"], layer_num_per_pp=[0, 0]
    )
    assert transceiver._async_consensus is None
    assert transceiver._diagnostics is None


def test_diagnostics_opt_in_does_not_enable_async_consensus(monkeypatch) -> None:
    monkeypatch.setattr(transceiver_module, "MPIDist", _FakeStartupMpiDist)
    monkeypatch.setenv(transceiver_module._DIAG_ENV, "1")
    dist = _FakeStartupMpiDist()
    transceiver = _make_startup_transceiver(dist)
    config = SimpleNamespace(backend="NIXL", transceiver_runtime="PYTHON")

    transceiver._init_async_consensus(config)

    assert transceiver._diagnostics is not None
    assert not transceiver._async_terminal_consensus_enabled
    assert not transceiver._async_peer_ready_consensus_enabled
    assert transceiver._async_consensus is None
    assert dist.descriptors == []


def test_malformed_diagnostic_env_does_not_interrupt_startup(monkeypatch) -> None:
    monkeypatch.setattr(transceiver_module, "MPIDist", _FakeStartupMpiDist)
    monkeypatch.setenv(transceiver_module._DIAG_ENV, "malformed")
    dist = _FakeStartupMpiDist()
    transceiver = _make_startup_transceiver(dist)
    config = SimpleNamespace(backend="NIXL", transceiver_runtime="PYTHON")

    transceiver._init_async_consensus(config)
    transceiver._exchange_rank_info()

    assert transceiver._diagnostics is None
    assert dist.descriptors == ["local-endpoint"]
    transceiver._transfer_worker.populate_instance_and_rank_info.assert_called_once_with(
        endpoints=["local-endpoint", "local-endpoint"], layer_num_per_pp=[0, 0]
    )


def test_diagnostic_allgather_is_transparent_when_disabled() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._diagnostics = None
    allgather = Mock(return_value=[[1], [1]])

    result = transceiver._diagnostic_allgather("test", allgather, [1])

    assert result == [[1], [1]]
    allgather.assert_called_once_with([1])


def test_diagnostic_wait_is_transparent_when_disabled() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._diagnostics = None
    session = _FakeSession(11, WaitResult.TIMEOUT)

    result = transceiver._diagnostic_wait_complete("gen", 11, session, blocking=False)

    assert result == WaitResult.TIMEOUT
    assert session.blocking_calls == [False]


def test_diagnostic_wait_reports_individual_call_duration(monkeypatch) -> None:
    fake_clock = _FakeClock()
    info = Mock()
    monkeypatch.setattr(transceiver_module.logger, "info", info)
    monkeypatch.setattr(transceiver_module.time, "monotonic", fake_clock)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._diagnostics = transceiver_module._PythonTransceiverDiagnostics(
        0,
        clock=fake_clock,
    )
    session = _FakeSession(12, WaitResult.TIMEOUT)
    session.wait_complete = Mock(
        side_effect=lambda *, blocking: (
            fake_clock.advance(2.0),
            WaitResult.TIMEOUT,
        )[1]
    )

    result = transceiver._diagnostic_wait_complete("gen", 12, session, blocking=True)

    assert result == WaitResult.TIMEOUT
    messages = [call.args[0] for call in info.call_args_list]
    assert any(
        "transition=gen_future_wait_call_exit" in message and "duration_s=2.000000" in message
        for message in messages
    )


def test_diagnostics_bound_samples_and_report_slow_collective(monkeypatch) -> None:
    fake_clock = _FakeClock()
    info = Mock()
    monkeypatch.setattr(transceiver_module.logger, "info", info)
    diagnostics = transceiver_module._PythonTransceiverDiagnostics(
        3,
        clock=fake_clock,
        summary_interval_s=1.0,
    )

    for request_id in range(transceiver_module._DIAG_MAX_SAMPLED_REQUESTS + 4):
        diagnostics.record_transition("created", request_id)
    sequence = diagnostics.collective_enter("gen_ready_ids")
    fake_clock.advance(0.2)
    diagnostics.collective_exit("gen_ready_ids", sequence)
    diagnostics.enter_state("gen_recv", 7)
    fake_clock.advance(2.0)
    diagnostics.maybe_log_snapshot({"recv_sessions": 1}, None, force=True)
    status_summary = KvCacheTransceiverV2._session_status_summary(
        {
            request_id: SimpleNamespace(status=SessionStatus.READY)
            for request_id in range(transceiver_module._DIAG_MAX_SAMPLED_REQUESTS + 4)
        }
    )

    assert len(diagnostics._sampled_request_ids) == transceiver_module._DIAG_MAX_SAMPLED_REQUESTS
    assert diagnostics._transition_counts["created"] == (
        transceiver_module._DIAG_MAX_SAMPLED_REQUESTS + 4
    )
    assert status_summary["READY"]["count"] == transceiver_module._DIAG_MAX_SAMPLED_REQUESTS + 4
    assert len(status_summary["READY"]["sample_ids"]) == (
        transceiver_module._DIAG_MAX_SAMPLED_REQUESTS
    )
    messages = [call.args[0] for call in info.call_args_list]
    assert any(
        "event=collective_exit label=gen_ready_ids" in message and "duration_s=0.200000" in message
        for message in messages
    )
    assert any(
        "event=snapshot" in message and "'gen_recv': {'count': 1, 'oldest_s': 2.0" in message
        for message in messages
    )


def test_diagnostic_snapshot_gate_avoids_hot_path_state_scans(monkeypatch) -> None:
    class _Session:
        @property
        def status(self):
            return status_getter()

    fake_clock = _FakeClock()
    diagnostics = transceiver_module._PythonTransceiverDiagnostics(
        0,
        clock=fake_clock,
        summary_interval_s=10.0,
    )
    monkeypatch.setattr(transceiver_module.logger, "info", Mock())
    diagnostics.maybe_log_snapshot({}, None, force=True)
    status_getter = Mock(return_value=SessionStatus.READY)
    coordinator = SimpleNamespace(diagnostic_snapshot=Mock(return_value={"rank": 0}))
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._diagnostics = diagnostics
    transceiver._async_consensus = coordinator
    transceiver._send_sessions = {1: _Session()}
    transceiver._recv_sessions = {2: _Session()}

    transceiver._maybe_log_diagnostics()

    status_getter.assert_not_called()
    coordinator.diagnostic_snapshot.assert_not_called()

    fake_clock.advance(10.0)
    transceiver._maybe_log_diagnostics()

    assert status_getter.call_count == 2
    coordinator.diagnostic_snapshot.assert_called_once_with()


def test_direct_cancel_retires_diagnostic_ownership(monkeypatch) -> None:
    info = Mock()
    monkeypatch.setattr(transceiver_module.logger, "info", info)
    send_session = _FakeSession(71, None)
    recv_session = _FakeSession(72, None)
    send_req = _FakeRequest(request_id=71)
    recv_req = _FakeRequest(request_id=72)
    transceiver = _make_transceiver({71: send_session}, {71: send_req})
    _enable_fake_async_consensus(transceiver)
    transceiver._recv_sessions[72] = recv_session
    transceiver._recv_reqs[72] = recv_req
    transceiver._diagnostics = transceiver_module._PythonTransceiverDiagnostics(0)
    transceiver._enter_diag_state("ctx_send", 71)
    transceiver._enter_diag_state("gen_recv", 72)

    assert transceiver.cancel_request(send_req)
    assert transceiver.cancel_request(recv_req)
    transceiver._maybe_log_diagnostics(force=True)

    assert not transceiver._diagnostics._active_states["ctx_send"]
    assert not transceiver._diagnostics._active_states["gen_recv"]
    snapshot = info.call_args_list[-1].args[0]
    assert "active_states={}" in snapshot


def test_async_startup_rejects_same_version_flag_mismatch(monkeypatch) -> None:
    def mismatch_flag(contribution):
        tag, endpoint, descriptor = contribution
        peer_descriptor = list(descriptor)
        peer_descriptor[-2] = "0"
        return [contribution, (tag, endpoint, tuple(peer_descriptor))]

    monkeypatch.setattr(transceiver_module, "MPIDist", _FakeStartupMpiDist)
    monkeypatch.setenv(transceiver_module._ASYNC_TERMINAL_ENV, "1")
    dist = _FakeStartupMpiDist(mismatch_flag)
    transceiver = _make_startup_transceiver(dist)
    config = SimpleNamespace(backend="NIXL", transceiver_runtime="PYTHON")
    transceiver._init_async_consensus(config)

    with pytest.raises(RuntimeError, match="startup descriptor mismatch"):
        transceiver._exchange_rank_info()

    assert len(dist.descriptors) == 1
    assert dist.pp_values == [0]


def test_async_startup_rejects_uniform_malformed_flag_after_exchange(monkeypatch) -> None:
    monkeypatch.setattr(transceiver_module, "MPIDist", _FakeStartupMpiDist)
    monkeypatch.setenv(transceiver_module._ASYNC_TERMINAL_ENV, "malformed")
    dist = _FakeStartupMpiDist()
    transceiver = _make_startup_transceiver(dist)
    config = SimpleNamespace(backend="NIXL", transceiver_runtime="PYTHON")
    transceiver._init_async_consensus(config)

    with pytest.raises(ValueError, match="must be 0 or 1"):
        transceiver._exchange_rank_info()

    assert len(dist.descriptors) == 1
    assert dist.pp_values == [0]


def test_async_startup_rejects_mixed_legacy_worker_group(monkeypatch) -> None:
    monkeypatch.setattr(transceiver_module, "MPIDist", _FakeStartupMpiDist)
    monkeypatch.setenv(transceiver_module._ASYNC_TERMINAL_ENV, "1")
    dist = _FakeStartupMpiDist(lambda contribution: [contribution, "legacy-peer-endpoint"])
    transceiver = _make_startup_transceiver(dist)
    config = SimpleNamespace(backend="NIXL", transceiver_runtime="PYTHON")
    transceiver._init_async_consensus(config)

    with pytest.raises(RuntimeError, match="same-version worker group"):
        transceiver._exchange_rank_info()

    assert len(dist.descriptors) == 1
    assert dist.pp_values == [0]


def test_async_startup_rejects_unsupported_opt_in_after_exchange(monkeypatch) -> None:
    monkeypatch.setattr(transceiver_module, "MPIDist", _FakeStartupMpiDist)
    monkeypatch.setenv(transceiver_module._ASYNC_TERMINAL_ENV, "1")
    dist = _FakeStartupMpiDist()
    transceiver = _make_startup_transceiver(dist)
    transceiver._mapping.tp_size = 2
    transceiver._mapping.pp_size = 1
    config = SimpleNamespace(backend="NIXL", transceiver_runtime="PYTHON")
    transceiver._init_async_consensus(config)

    with pytest.raises(RuntimeError, match="currently requires"):
        transceiver._exchange_rank_info()

    assert len(dist.descriptors) == 1
    assert dist.pp_values == [0]


@pytest.mark.parametrize(
    "terminal_value,ready_value",
    [("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")],
)
def test_async_startup_supported_mode_matrix(
    monkeypatch, terminal_value: str, ready_value: str
) -> None:
    monkeypatch.setattr(transceiver_module, "MPIDist", _FakeStartupMpiDist)
    monkeypatch.setenv(transceiver_module._ASYNC_TERMINAL_ENV, terminal_value)
    monkeypatch.setenv(transceiver_module._ASYNC_PEER_READY_ENV, ready_value)
    transport = object()
    transport_constructor = Mock(return_value=transport)
    coordinator = object()
    coordinator_constructor = Mock(return_value=coordinator)
    monkeypatch.setattr(transceiver_module, "MpiConsensusTransport", transport_constructor)
    monkeypatch.setattr(transceiver_module, "AsyncConsensusCoordinator", coordinator_constructor)
    dist = _FakeStartupMpiDist()
    transceiver = _make_startup_transceiver(dist)
    config = SimpleNamespace(backend="NIXL", transceiver_runtime="PYTHON")

    transceiver._init_async_consensus(config)
    assert dist.descriptors == []
    transceiver._exchange_rank_info()

    assert transceiver._async_terminal_consensus_enabled is (terminal_value == "1")
    assert transceiver._async_peer_ready_consensus_enabled is (ready_value == "1")
    if terminal_value == "1" or ready_value == "1":
        transport_constructor.assert_called_once_with((0, 1))
        coordinator_constructor.assert_called_once_with(transport, scheduling_rank=0)
        assert transceiver._async_consensus is coordinator
    else:
        transport_constructor.assert_not_called()
        coordinator_constructor.assert_not_called()
        assert transceiver._async_consensus is None


def test_async_startup_closes_transport_when_coordinator_construction_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(transceiver_module, "MPIDist", _FakeStartupMpiDist)
    monkeypatch.setenv(transceiver_module._ASYNC_TERMINAL_ENV, "1")
    transport = Mock()
    monkeypatch.setattr(
        transceiver_module,
        "MpiConsensusTransport",
        Mock(return_value=transport),
    )
    monkeypatch.setattr(
        transceiver_module,
        "AsyncConsensusCoordinator",
        Mock(side_effect=RuntimeError("coordinator construction failed")),
    )
    dist = _FakeStartupMpiDist()
    transceiver = _make_startup_transceiver(dist)
    config = SimpleNamespace(backend="NIXL", transceiver_runtime="PYTHON")

    transceiver._init_async_consensus(config)
    with pytest.raises(RuntimeError, match="coordinator construction failed"):
        transceiver._exchange_rank_info()

    transport.close.assert_called_once_with(transceiver_module._CONSENSUS_STARTUP_CLOSE_TIMEOUT_S)
    assert transceiver._async_consensus is None


def test_constructor_rolls_back_transfer_worker_when_consensus_startup_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(transceiver_module, "MPIDist", _FakeStartupMpiDist)
    monkeypatch.setenv(transceiver_module._ASYNC_TERMINAL_ENV, "1")
    monkeypatch.setattr(
        transceiver_module.torch.cuda,
        "current_device",
        Mock(return_value=0),
    )
    monkeypatch.setattr(
        KvCacheTransceiverV2,
        "_check_compatible",
        Mock(),
    )
    monkeypatch.setattr(
        KvCacheTransceiverV2,
        "_init_sync_policy",
        Mock(),
    )
    monkeypatch.setattr(
        KvCacheTransceiverV2,
        "_broadcast_instance_name",
        Mock(return_value="instance"),
    )
    monkeypatch.setattr(
        KvCacheTransceiverV2,
        "_broadcast_context_endpoint",
        Mock(return_value="context-endpoint"),
    )
    monkeypatch.setattr(
        transceiver_module,
        "create_cache_reuse_adapter",
        Mock(return_value=object()),
    )
    worker = SimpleNamespace(
        sender_endpoint="local-endpoint",
        shutdown=Mock(return_value=None),
    )
    monkeypatch.setattr(transceiver_module, "TransferWorker", Mock(return_value=worker))
    monkeypatch.setattr(
        transceiver_module,
        "MpiConsensusTransport",
        Mock(side_effect=RuntimeError("transport construction failed")),
    )
    mapping = SimpleNamespace(
        world_size=2,
        tp_size=1,
        pp_size=2,
        cp_size=1,
        tp_rank=0,
        enable_attention_dp=False,
        pp_group=(0, 1),
    )
    dist = _FakeStartupMpiDist()
    kv_cache_manager = SimpleNamespace(max_batch_size=1, pp_layers=[])
    config = SimpleNamespace(
        backend="NIXL",
        transceiver_runtime="PYTHON",
        kv_transfer_timeout_ms=1000,
        kv_transfer_poll_interval_ms=10,
        kv_transfer_sender_future_timeout_ms=1000,
        kv_cache_bounce_size_mb=0,
    )

    with pytest.raises(RuntimeError, match="transport construction failed"):
        KvCacheTransceiverV2(mapping, dist, kv_cache_manager, config)

    worker.shutdown.assert_called_once_with()


def test_startup_rollback_bounds_deferred_worker_wait(monkeypatch) -> None:
    transceiver = _make_transceiver({})
    worker_event = threading.Event()
    worker_event.wait = Mock(return_value=False)
    transceiver.shutdown = Mock(return_value=worker_event)

    transceiver._rollback_failed_startup()

    worker_event.wait.assert_called_once_with(transceiver_module._STARTUP_ROLLBACK_TIMEOUT_S)
    transceiver.shutdown.assert_called_once_with()


def test_shutdown_continues_after_session_close_failure_and_is_idempotent() -> None:
    failed_session = _FakeSession(rid=41, wait_result=None)
    failed_session.close = Mock(side_effect=RuntimeError("close failed"))
    healthy_session = _FakeSession(rid=42, wait_result=None)
    healthy_session.close = Mock()
    transceiver = _make_transceiver(
        {41: failed_session, 42: healthy_session},
        {41: _FakeRequest(request_id=41), 42: _FakeRequest(request_id=42)},
    )
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._async_consensus = Mock()
    transceiver._async_consensus_counters = defaultdict(int)
    transceiver._dist = SimpleNamespace(rank=0)
    transceiver._transfer_worker.shutdown = Mock(return_value=None)

    with pytest.raises(RuntimeError, match="close failed"):
        transceiver.shutdown()

    failed_session.close.assert_called_once_with()
    healthy_session.close.assert_called_once_with()
    transceiver._async_consensus.shutdown.assert_called_once_with()
    transceiver._transfer_worker.shutdown.assert_called_once_with()
    assert not transceiver._send_sessions
    assert not transceiver._send_reqs

    transceiver.shutdown()
    failed_session.close.assert_called_once_with()
    transceiver._async_consensus.shutdown.assert_called_once_with()
    transceiver._transfer_worker.shutdown.assert_called_once_with()


def test_shutdown_retries_only_incomplete_consensus_teardown() -> None:
    transceiver = _make_transceiver({})
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._async_consensus = Mock()
    transceiver._async_consensus.shutdown = Mock(
        side_effect=[RuntimeError("peer shutdown timeout"), None]
    )
    transceiver._async_consensus_counters = defaultdict(int)
    transceiver._dist = SimpleNamespace(rank=0)
    transceiver._transfer_worker.shutdown = Mock(return_value=None)

    with pytest.raises(RuntimeError, match="peer shutdown timeout"):
        transceiver.shutdown()

    assert transceiver._shutdown
    assert not transceiver._shutdown_complete
    transceiver._async_consensus.shutdown.assert_called_once_with()
    transceiver._transfer_worker.shutdown.assert_called_once_with()

    transceiver.shutdown()

    assert transceiver._shutdown_complete
    assert transceiver._async_consensus.shutdown.call_count == 2
    # The worker completed on the first attempt and must not be torn down a
    # second time while the delayed peer's consensus close is retried.
    transceiver._transfer_worker.shutdown.assert_called_once_with()

    transceiver.shutdown()
    assert transceiver._async_consensus.shutdown.call_count == 2


def test_shutdown_propagates_deferred_worker_completion_before_marking_complete() -> None:
    transceiver = _make_transceiver({})
    transceiver._async_consensus = None
    completion = threading.Event()
    transceiver._transfer_worker.shutdown = Mock(side_effect=[completion, None])

    assert transceiver.shutdown() is completion
    assert not transceiver._shutdown_worker_complete
    assert not transceiver._shutdown_complete

    # Repeated calls while native teardown is live must propagate the same
    # ownership barrier without invoking shutdown again.
    assert transceiver.shutdown() is completion
    transceiver._transfer_worker.shutdown.assert_called_once_with()

    completion.set()
    assert transceiver.shutdown() is None
    assert transceiver._shutdown_worker_complete
    assert transceiver._shutdown_complete
    assert transceiver._transfer_worker.shutdown.call_count == 2


def test_shutdown_defers_primary_error_until_worker_completion() -> None:
    session = _FakeSession(rid=43, wait_result=None)
    session.close = Mock(side_effect=RuntimeError("session close failed"))
    transceiver = _make_transceiver({43: session})
    transceiver._async_consensus = None
    completion = threading.Event()
    transceiver._transfer_worker.shutdown = Mock(side_effect=[completion, None])

    # Native ownership is still live, so propagate the barrier instead of an
    # error that could let the caller release registered memory too early.
    assert transceiver.shutdown() is completion

    completion.set()
    with pytest.raises(RuntimeError, match="session close failed"):
        transceiver.shutdown()

    assert transceiver._shutdown_complete
    session.close.assert_called_once_with()
    assert transceiver._transfer_worker.shutdown.call_count == 2


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
    session._terminal_snapshot = None
    session._exception = None
    session._outstanding_operations = 0
    session._sealed = False
    session.lock = threading.Lock()
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


def test_legacy_context_cancel_retains_request_until_native_transfer_is_quiescent() -> None:
    session = _FakeSession(
        rid=14,
        wait_result=None,
        status=SessionStatus.CANCELLED,
        has_transferring_tasks=True,
    )
    req = _FakeRequest(request_id=14)
    transceiver = _make_transceiver({14: session}, {14: req})

    assert transceiver.check_context_transfer_status(0) == ([], [])

    assert transceiver._send_sessions[14] is session
    assert transceiver._send_reqs[14] is req
    assert not session.closed

    session._has_transferring_tasks = False
    assert transceiver.check_context_transfer_status(0) == ([], [])

    assert 14 not in transceiver._send_sessions
    assert 14 not in transceiver._send_reqs
    assert session.closed


def test_legacy_peer_context_cancel_surfaces_quiescent_request_id() -> None:
    session = _FakeSession(rid=16, wait_result=None)
    req = _FakeRequest(request_id=16)
    transceiver = _make_transceiver({16: session}, {16: req})
    transceiver._ctx_consensus_outcome = Mock(return_value=([16], [16], [], [], [], []))

    # No local cancel_request() call: cancellation is learned from a peer's
    # packed legacy outcome vote.
    assert transceiver.check_context_transfer_status(0) == ([], [])

    assert session.cancel_calls == 1
    assert session.closed
    assert 16 not in transceiver._send_sessions
    assert 16 not in transceiver._send_reqs
    assert transceiver.take_context_cancelled_request_ids() == [16]
    assert transceiver.take_context_cancelled_request_ids() == []


def test_legacy_context_failure_retains_request_until_sibling_operations_are_quiescent() -> None:
    session = _FakeSession(
        rid=15,
        wait_result=WaitResult.FAILED,
        status=SessionStatus.ERROR,
        has_failed=True,
        has_transferring_tasks=True,
    )
    req = _FakeRequest(request_id=15)
    transceiver = _make_transceiver({15: session}, {15: req})

    assert transceiver.check_context_transfer_status(0) == ([], [])

    assert transceiver._send_sessions[15] is session
    assert transceiver._send_reqs[15] is req
    assert session.sealed
    assert not session.closed

    session._has_transferring_tasks = False
    assert transceiver.check_context_transfer_status(0) == ([], [15])

    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert 15 not in transceiver._send_sessions
    assert 15 not in transceiver._send_reqs
    assert session.closed


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


def test_async_context_zero_budget_publishes_all_quiescent_terminal_sessions() -> None:
    completed_session = _FakeSession(
        rid=21,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )
    transferring_session = _FakeSession(
        rid=22,
        wait_result=None,
        has_transferring_tasks=True,
    )
    transceiver = _make_transceiver({21: completed_session, 22: transferring_session})
    coordinator = _enable_fake_async_consensus(transceiver, terminal=True)
    transceiver._ctx_consensus = Mock(
        side_effect=AssertionError("legacy readiness collective must not run")
    )
    transceiver._ctx_consensus_outcome = Mock(
        side_effect=AssertionError("legacy outcome collective must not run")
    )

    completed, failed = transceiver.check_context_transfer_status(0)

    assert completed == []
    assert failed == []
    assert coordinator.terminal_votes == [(21, ConsensusOutcome.COMPLETED, 0)]
    assert completed_session.sealed
    assert completed_session.blocking_calls == [False]
    assert not completed_session.closed
    assert not transferring_session.blocking_calls
    assert coordinator.poll_count == 2
    transceiver._ctx_consensus.assert_not_called()
    transceiver._ctx_consensus_outcome.assert_not_called()

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.TERMINAL_COMMIT,
            21,
            0,
            ConsensusOutcome.COMPLETED,
        )
    )
    completed, failed = transceiver.check_context_transfer_status(0)

    assert completed == [21]
    assert failed == []
    assert completed_session.closed
    assert 21 not in transceiver._send_sessions
    assert 22 in transceiver._send_sessions


def test_async_context_does_not_publish_if_work_starts_at_seal_boundary() -> None:
    session = _FakeSession(
        rid=24,
        wait_result=WaitResult.FAILED,
        has_failed=True,
        seal_quiescent=False,
    )
    transceiver = _make_transceiver({24: session})
    coordinator = _enable_fake_async_consensus(transceiver, terminal=True)

    assert transceiver.check_context_transfer_status(0) == ([], [])
    assert session.sealed
    assert coordinator.terminal_votes == []
    assert 24 in transceiver._send_sessions


def test_async_context_cancel_retains_session_until_authoritative_commit() -> None:
    session = _FakeSession(rid=23, wait_result=WaitResult.FAILED)
    req = _FakeRequest(request_id=23)
    transceiver = _make_transceiver({23: session}, {23: req})
    coordinator = _enable_fake_async_consensus(transceiver, terminal=True)

    assert not transceiver.cancel_request(req)
    assert coordinator.terminal_votes == [(23, ConsensusOutcome.CANCELLED, 0)]
    assert 23 in transceiver._send_sessions
    assert not session.closed

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.TERMINAL_COMMIT,
            23,
            0,
            ConsensusOutcome.CANCELLED,
        )
    )
    assert transceiver.cancel_request(req)
    assert session.closed
    assert 23 not in transceiver._send_sessions
    # The same local retry consumed the tombstone, so a later status poll must
    # not misclassify this already-terminated request as a peer cancellation.
    assert transceiver.take_context_cancelled_request_ids() == []


def test_async_cancelled_commit_status_poll_retains_owner_until_cancel_retry() -> None:
    session = _FakeSession(rid=35, wait_result=WaitResult.FAILED)
    req = _FakeRequest(request_id=35)
    transceiver = _make_transceiver({35: session}, {35: req})
    coordinator = _enable_fake_async_consensus(transceiver, terminal=True)

    assert not transceiver.cancel_request(req)
    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.TERMINAL_COMMIT,
            35,
            0,
            ConsensusOutcome.CANCELLED,
        )
    )

    # The ordinary status poll wins the race with the executor's cancellation
    # retry. Native resources close, but request-scoped ownership remains.
    assert transceiver.check_context_transfer_status(0) == ([], [])
    assert session.closed
    assert 35 not in transceiver._send_sessions
    assert transceiver.owns_request(req)
    with pytest.raises(RuntimeError, match="before its asynchronous terminal cancellation"):
        transceiver._get_or_create_send_session(_FakeRequest(request_id=35))

    # The retry acknowledges exactly the cancelled transceiver leg and makes
    # the next request-ID epoch reusable.
    assert transceiver.cancel_request(req)
    assert not transceiver.owns_request(req)
    assert transceiver._async_terminal_epoch[35] == 1


def test_async_peer_context_cancel_surfaces_id_without_local_retry() -> None:
    session = _FakeSession(rid=38, wait_result=None)
    req = _FakeRequest(request_id=38)
    transceiver = _make_transceiver({38: session}, {38: req})
    coordinator = _enable_fake_async_consensus(transceiver, terminal=True)
    transceiver._async_terminal_published[38] = 0
    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.TERMINAL_COMMIT,
            38,
            0,
            ConsensusOutcome.CANCELLED,
        )
    )

    # No local cancel_request() call: the status poll must hand the peer's
    # authoritative cancellation to PyExecutor while retaining its tombstone.
    assert transceiver.check_context_transfer_status(0) == ([], [])

    assert session.closed
    assert transceiver.owns_request(req)
    assert transceiver.take_context_cancelled_request_ids() == [38]
    assert transceiver.take_context_cancelled_request_ids() == []
    assert transceiver.cancel_request(req)
    assert not transceiver.owns_request(req)


def test_async_context_cancel_does_not_replace_published_terminal_vote() -> None:
    session = _FakeSession(
        rid=25,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )
    req = _FakeRequest(request_id=25)
    transceiver = _make_transceiver({25: session}, {25: req})
    coordinator = _enable_fake_async_consensus(transceiver, terminal=True)

    assert transceiver.check_context_transfer_status(0) == ([], [])
    assert coordinator.terminal_votes == [(25, ConsensusOutcome.COMPLETED, 0)]

    assert not transceiver.cancel_request(req)
    assert session.cancel_calls == 0
    assert coordinator.terminal_votes == [(25, ConsensusOutcome.COMPLETED, 0)]

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.TERMINAL_COMMIT,
            25,
            0,
            ConsensusOutcome.COMPLETED,
        )
    )
    assert transceiver.cancel_request(req)
    assert session.cancel_calls == 0
    assert session.closed


def test_async_context_positive_budget_waits_for_authoritative_commit() -> None:
    session = _FakeSession(
        rid=26,
        wait_result=WaitResult.COMPLETED,
        is_completed=True,
    )
    transceiver = _make_transceiver({26: session})
    coordinator = _enable_fake_async_consensus(transceiver, terminal=True)

    def publish_commit_after_first_sweep(fake: _FakeAsyncCoordinator) -> None:
        if fake.poll_count == 3 and fake.terminal_votes:
            fake.events.append(
                ConsensusEvent(
                    ConsensusEventKind.TERMINAL_COMMIT,
                    26,
                    0,
                    ConsensusOutcome.COMPLETED,
                )
            )

    coordinator.poll_hook = publish_commit_after_first_sweep

    completed, failed = transceiver.check_context_transfer_status(1)

    assert completed == [26]
    assert failed == []
    assert coordinator.poll_count >= 3
    assert session.blocking_calls == [False]


def test_async_context_block_all_drains_entry_snapshot_through_commits() -> None:
    sessions = {
        rid: _FakeSession(
            rid=rid,
            wait_result=WaitResult.COMPLETED,
            is_completed=True,
        )
        for rid in (27, 28)
    }
    session_objects = list(sessions.values())
    transceiver = _make_transceiver(sessions)
    coordinator = _enable_fake_async_consensus(transceiver, terminal=True)

    def publish_commits_after_first_sweep(fake: _FakeAsyncCoordinator) -> None:
        if fake.poll_count != 3:
            return
        for rid, _outcome, epoch in fake.terminal_votes:
            fake.events.append(
                ConsensusEvent(
                    ConsensusEventKind.TERMINAL_COMMIT,
                    rid,
                    epoch,
                    ConsensusOutcome.COMPLETED,
                )
            )

    coordinator.poll_hook = publish_commits_after_first_sweep

    completed, failed = transceiver.check_context_transfer_status(None)

    assert completed == [27, 28]
    assert failed == []
    assert not transceiver._send_sessions
    assert all(session.blocking_calls == [False] for session in session_objects)


def test_async_context_block_all_stops_at_existing_sender_timeout() -> None:
    session = _FakeSession(rid=30, wait_result=None)
    transceiver = _make_transceiver({30: session})
    _enable_fake_async_consensus(transceiver, terminal=True)
    transceiver._sender_future_timeout_ms = 1

    assert transceiver.check_context_transfer_status(None) == ([], [])

    assert 30 in transceiver._send_sessions
    assert session.blocking_calls
    assert all(not blocking for blocking in session.blocking_calls)


def test_async_context_uses_native_atomic_terminal_snapshot_when_available() -> None:
    session = _FakeSession(rid=29, wait_result=None)
    session.seal_and_snapshot_terminal = Mock(  # type: ignore[attr-defined]
        return_value=SessionStatus.KV_TRANSFERRED
    )
    session.seal_and_check_quiescent = Mock(  # type: ignore[method-assign]
        side_effect=AssertionError("compatibility seal must not run")
    )
    transceiver = _make_transceiver({29: session})
    coordinator = _enable_fake_async_consensus(transceiver, terminal=True)

    assert transceiver.check_context_transfer_status(0) == ([], [])

    assert coordinator.terminal_votes == [(29, ConsensusOutcome.COMPLETED, 0)]
    session.seal_and_snapshot_terminal.assert_called_once_with()  # type: ignore[attr-defined]


def test_respond_pre_cancel_keeps_paired_ownership_without_dispatch() -> None:
    transceiver = _make_transceiver({})
    session = _FakeSession(
        rid=41,
        wait_result=WaitResult.FAILED,
        status=SessionStatus.CANCELLED,
    )
    session.send = Mock()  # type: ignore[attr-defined]
    transceiver._transfer_worker.tx_session = session
    transceiver._create_kv_slice = Mock(side_effect=AssertionError("must not build a slice"))
    req = _FakeRequest(request_id=41)

    transceiver.respond_and_send_async(req)

    assert transceiver._send_sessions[41] is session
    assert transceiver._send_reqs[41] is req
    session.send.assert_not_called()  # type: ignore[attr-defined]


def test_respond_cancel_between_kv_and_aux_keeps_paired_ownership() -> None:
    transceiver = _make_transceiver({})
    session = _FakeSession(rid=42, wait_result=None)
    session.send = Mock()  # type: ignore[attr-defined]
    session.pack_aux = Mock(  # type: ignore[attr-defined]
        side_effect=lambda _req: session.cancel()
    )
    session.send_aux = Mock(  # type: ignore[attr-defined]
        side_effect=RuntimeError("session sealed by cancellation")
    )
    transceiver._transfer_worker.tx_session = session
    transceiver._create_kv_slice = Mock(return_value=Mock())
    transceiver._need_aux_transfer = lambda _req: True
    transceiver._dp_rank = 0
    transceiver._context_info_endpoint = "ctx"
    req = _FakeRequest(request_id=42)

    transceiver.respond_and_send_async(req)

    assert transceiver._send_sessions[42] is session
    assert transceiver._send_reqs[42] is req
    session.send.assert_called_once()  # type: ignore[attr-defined]
    session.send_aux.assert_called_once()  # type: ignore[attr-defined]


def test_respond_dispatch_error_preserves_paired_ownership_before_reraising() -> None:
    transceiver = _make_transceiver({})
    session = _FakeSession(rid=43, wait_result=None)
    session.send = Mock(side_effect=RuntimeError("dispatch failed"))  # type: ignore[attr-defined]
    transceiver._transfer_worker.tx_session = session
    transceiver._create_kv_slice = Mock(return_value=Mock())
    req = _FakeRequest(request_id=43)

    with pytest.raises(RuntimeError, match="dispatch failed"):
        transceiver.respond_and_send_async(req)

    assert transceiver._send_sessions[43] is session
    assert transceiver._send_reqs[43] is req


def test_receive_pre_cancel_keeps_paired_ownership_without_dispatch() -> None:
    transceiver = _make_transceiver({})
    session = _FakeSession(
        rid=44,
        wait_result=WaitResult.FAILED,
        status=SessionStatus.CANCELLED,
    )
    session.receive = Mock()  # type: ignore[attr-defined]
    transceiver._transfer_worker.rx_session = session
    transceiver._create_kv_slice = Mock(side_effect=AssertionError("must not build a slice"))
    req = _FakeRequest(request_id=44)

    transceiver.request_and_receive_async(req)

    assert transceiver._recv_sessions[44] is session
    assert transceiver._recv_reqs[44] is req
    session.receive.assert_not_called()  # type: ignore[attr-defined]


def test_receive_cancel_during_dispatch_keeps_paired_ownership() -> None:
    transceiver = _make_transceiver({})
    session = _FakeSession(rid=45, wait_result=None)

    def cancel_and_raise(_slice) -> None:
        session.cancel()
        raise RuntimeError("session sealed by cancellation")

    session.receive = Mock(side_effect=cancel_and_raise)  # type: ignore[attr-defined]
    transceiver._transfer_worker.rx_session = session
    transceiver._create_kv_slice = Mock(return_value=Mock())
    transceiver._slice_num_bytes = Mock(return_value=0)
    transceiver._kv_size_rank_factor = 1
    req = _FakeRequest(request_id=45)

    transceiver.request_and_receive_async(req)

    assert transceiver._recv_sessions[45] is session
    assert transceiver._recv_reqs[45] is req
    session.receive.assert_called_once()  # type: ignore[attr-defined]


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
    transceiver._gen_consensus_outcome = Mock(return_value=([], [], [], [], []))
    transceiver._close_failed_sessions = Mock()

    completed, failed, cancelled = transceiver.check_gen_transfer_status(at_least_request_num=0)

    assert completed == []
    assert failed == []
    assert cancelled == []
    transceiver._gen_consensus.assert_called_once_with([])


def test_legacy_gen_cancel_retains_request_until_native_cancel_ack_is_quiescent() -> None:
    session = _FakeSession(
        rid=51,
        wait_result=None,
        status=SessionStatus.CANCELLED,
        has_transferring_tasks=True,
    )
    req = _FakeRequest(request_id=51)
    transceiver = _make_transceiver({})
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._gen_allgather = Mock()
    transceiver._gen_consensus = lambda local_ids: list(local_ids)
    transceiver._recv_sessions = {51: session}
    transceiver._recv_reqs = {51: req}

    assert transceiver.check_gen_transfer_status(0) == ([], [], [])

    assert transceiver._recv_sessions[51] is session
    assert transceiver._recv_reqs[51] is req
    assert not session.closed

    # RxSession.has_transferring_tasks() includes the sender cancellation ACK,
    # so this transition models both the active write and its ACK draining.
    session._has_transferring_tasks = False
    completed, failed, cancelled = transceiver.check_gen_transfer_status(0)

    assert completed == []
    assert failed == []
    assert cancelled == [req]
    assert 51 not in transceiver._recv_sessions
    assert 51 not in transceiver._recv_reqs
    assert session.closed


def test_legacy_gen_failure_retains_request_until_sibling_operations_are_quiescent() -> None:
    session = _FakeSession(
        rid=52,
        wait_result=WaitResult.FAILED,
        status=SessionStatus.ERROR,
        has_failed=True,
        has_transferring_tasks=True,
    )
    req = _FakeRequest(request_id=52)
    transceiver = _make_transceiver({})
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._gen_allgather = Mock()
    transceiver._gen_consensus = lambda local_ids: list(local_ids)
    transceiver._recv_sessions = {52: session}
    transceiver._recv_reqs = {52: req}
    transceiver._dist = SimpleNamespace(rank=0)

    assert transceiver.check_gen_transfer_status(0) == ([], [], [])

    assert transceiver._recv_sessions[52] is session
    assert transceiver._recv_reqs[52] is req
    assert session.sealed
    assert not session.closed

    session._has_transferring_tasks = False
    assert transceiver.check_gen_transfer_status(0) == ([], [52], [])

    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert 52 not in transceiver._recv_sessions
    assert 52 not in transceiver._recv_reqs
    assert session.closed


def test_consensus_outcome_uses_single_batched_allgather() -> None:
    # Decision, quiescence, failure, and completion are exchanged with ONE
    # allgather. Verify union (cancel/fail) and intersection
    # (cancel-quiescence/complete) semantics without an extra rendezvous.
    transceiver = object.__new__(KvCacheTransceiverV2)
    calls: list = []

    def fake_allgather(payload):
        calls.append(payload)
        # rank0 = this rank's payload; rank1 = a peer rank.
        return [payload, [[], [1], [99], [2, 99], [7, 8]]]

    to_process = [1, 2, 7, 8, 99]
    new_cancelled, reclaimable_cancelled, new_failed, reclaimable_failed, new_completed = (
        transceiver._consensus_outcome(
            to_process,
            to_process,
            [1],
            [1],
            [2],
            [2],
            [7],
            fake_allgather,
            True,
        )
    )

    assert len(calls) == 1  # batched: a single allgather, not three
    assert calls[0] == [[1], [1], [2], [2], [7]]
    assert new_cancelled == [1]  # union of cancelled across ranks
    assert reclaimable_cancelled == [1]  # quiescent only because both ranks ACKed
    assert new_failed == [2, 99]  # union of failed across ranks
    assert reclaimable_failed == [2]  # 99 lacks this rank's quiescence ACK
    assert new_completed == [7]  # intersection only (8 is completed on the peer only)


def test_consensus_outcome_defers_cancel_reclamation_until_every_rank_is_quiescent() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)

    def fake_allgather(payload):
        # The peer has observed the same cancellation decision but still has
        # an active native transfer, so it omits the quiescence ACK.
        return [payload, [[61], [], [], [], []]]

    cancelled, reclaimable, failed, reclaimable_failed, completed = transceiver._consensus_outcome(
        [61],
        [61],
        [61],
        [61],
        [],
        [],
        [],
        fake_allgather,
        True,
    )

    assert cancelled == [61]
    assert reclaimable == []
    assert failed == []
    assert reclaimable_failed == []
    assert completed == []


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


def test_check_context_runs_consensus_after_a_send() -> None:
    # Once the worker has sent, the ctx consensus runs as usual.
    transceiver = _make_transceiver({})
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_consensus = Mock(return_value=[])
    transceiver._ctx_consensus_outcome = Mock(return_value=([], [], [], [], [], []))

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


def test_async_peer_ready_activates_only_from_authoritative_schedule() -> None:
    transceiver = _make_transceiver({})
    coordinator = _enable_fake_async_consensus(transceiver, peer_ready=True)
    req = _FakeRequest(request_id=31)
    transceiver._wait_reqs[31] = req
    transceiver._transfer_worker.ready_request_ids.add(31)

    transceiver._prepare_context_requests_async()
    assert coordinator.ready_votes == [(31, 0)]
    assert req.state is None

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_PREPARE,
            31,
            0,
            ConsensusOutcome.READY,
        )
    )
    transceiver._progress_async_consensus()
    assert req.state is None
    assert 31 not in transceiver._wait_reqs
    assert coordinator.ready_acks == [(31, 0)]

    # Omitting the request from rank zero's schedule leaves the lease hidden.
    transceiver.activate_context_requests_for_schedule([])
    assert req.state is None
    assert coordinator.ready_activation_acks == []

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_RELEASE,
            31,
            0,
            ConsensusOutcome.READY,
        )
    )
    transceiver._progress_async_consensus()
    assert req.state == LlmRequestState.CONTEXT_INIT
    assert (31, 0) in transceiver._async_ready_prepared

    transceiver.activate_context_requests_for_schedule([req])
    assert coordinator.ready_activation_acks == [(31, 0)]
    assert not transceiver._async_ready_prepared
    assert (31, 0) in transceiver._async_ready_activated
    assert 31 in transceiver._async_ready_published

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_COMPLETE,
            31,
            0,
            ConsensusOutcome.READY,
        )
    )
    transceiver._progress_async_consensus()
    assert not transceiver._async_ready_activated
    assert 31 not in transceiver._async_ready_published


def test_async_peer_ready_follower_stays_hidden_until_schedule_arrives() -> None:
    transceiver = _make_transceiver({})
    coordinator = _enable_fake_async_consensus(transceiver, peer_ready=True)
    transceiver._dist.rank = 1
    req = _FakeRequest(request_id=35)
    transceiver._wait_reqs[35] = req
    transceiver._async_ready_published[35] = 0
    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_PREPARE,
            35,
            0,
            ConsensusOutcome.READY,
        )
    )
    transceiver._progress_async_consensus()

    assert req.state is None
    transceiver.activate_context_requests_for_schedule([])
    assert req.state is None

    transceiver.activate_context_requests_for_schedule([req])
    assert req.state == LlmRequestState.CONTEXT_INIT
    assert coordinator.ready_activation_acks == [(35, 0)]


def test_async_peer_ready_cancel_withdraws_and_tombstones_request() -> None:
    transceiver = _make_transceiver({})
    coordinator = _enable_fake_async_consensus(transceiver, peer_ready=True)
    req = _FakeRequest(request_id=32)
    transceiver._wait_reqs[32] = req
    transceiver._transfer_worker.ready_request_ids.add(32)
    transceiver._prepare_context_requests_async()

    assert not transceiver.cancel_request(req)
    assert coordinator.ready_withdrawals == [(32, 0)]
    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_ABORT,
            32,
            0,
            ConsensusOutcome.WITHDRAWN,
        )
    )
    transceiver._progress_async_consensus()
    assert coordinator.ready_abort_acks == [(32, 0)]
    assert not transceiver.cancel_request(req)

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_ABORT_FINALIZE,
            32,
            0,
            ConsensusOutcome.WITHDRAWN,
        )
    )
    transceiver._progress_async_consensus()
    assert transceiver.cancel_request(req)

    transceiver.prepare_context_requests([req])
    assert coordinator.ready_votes == [(32, 0)]
    assert 32 not in transceiver._wait_reqs


def test_async_peer_ready_cancel_before_local_vote_waits_for_abort_finalize() -> None:
    transceiver = _make_transceiver({})
    coordinator = _enable_fake_async_consensus(transceiver, peer_ready=True)
    req = _FakeRequest(request_id=36)
    transceiver._wait_reqs[36] = req

    # Local peer metadata is not ready, but another rank may already have
    # voted. Join the current epoch with a withdrawal instead of freeing early.
    assert not transceiver.cancel_request(req)
    assert coordinator.ready_votes == []
    assert coordinator.ready_withdrawals == [(36, 0)]
    assert transceiver.owns_request(req)
    assert 36 in transceiver._wait_reqs

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_ABORT,
            36,
            0,
            ConsensusOutcome.WITHDRAWN,
        )
    )
    transceiver._progress_async_consensus()
    assert coordinator.ready_abort_acks == [(36, 0)]
    assert not transceiver.cancel_request(req)

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_ABORT_FINALIZE,
            36,
            0,
            ConsensusOutcome.WITHDRAWN,
        )
    )
    transceiver._progress_async_consensus()
    assert transceiver.cancel_request(req)
    assert not transceiver.owns_request(req)
    assert transceiver._async_ready_epoch[36] == 1


def test_async_peer_ready_abort_before_local_request_binds_without_fail_stop() -> None:
    transceiver = _make_transceiver({})
    coordinator = _enable_fake_async_consensus(transceiver, peer_ready=True)
    req = _FakeRequest(request_id=40)

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_ABORT,
            40,
            0,
            ConsensusOutcome.WITHDRAWN,
        )
    )
    transceiver._progress_async_consensus()

    assert coordinator.ready_abort_acks == [(40, 0)]
    assert transceiver._async_ready_aborted[(40, 0)] is None

    transceiver.prepare_context_requests([req])

    assert transceiver._async_ready_aborted[(40, 0)] is req
    assert req.state == LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
    assert req._trtllm_async_ready_cancelled_epoch == 0
    assert 40 not in transceiver._wait_reqs
    assert coordinator.ready_votes == []

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_ABORT_FINALIZE,
            40,
            0,
            ConsensusOutcome.WITHDRAWN,
        )
    )
    transceiver._progress_async_consensus()
    assert transceiver.cancel_request(req)


def test_async_peer_ready_finalized_abort_excludes_late_request() -> None:
    transceiver = _make_transceiver({})
    coordinator = _enable_fake_async_consensus(transceiver, peer_ready=True)
    req = _FakeRequest(request_id=41)

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_ABORT,
            41,
            0,
            ConsensusOutcome.WITHDRAWN,
        )
    )
    transceiver._progress_async_consensus()
    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_ABORT_FINALIZE,
            41,
            0,
            ConsensusOutcome.WITHDRAWN,
        )
    )
    transceiver._progress_async_consensus()

    assert transceiver._async_ready_finalized_without_request[41] == 0
    assert transceiver._async_ready_epoch[41] == 1

    transceiver.prepare_context_requests([req])

    assert 41 not in transceiver._async_ready_finalized_without_request
    assert req.state == LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
    assert req._trtllm_async_ready_cancelled_epoch == 0
    assert 41 not in transceiver._wait_reqs
    assert coordinator.ready_votes == []


def test_known_cancel_withdraws_before_readiness_poll_or_vote() -> None:
    transceiver = _make_transceiver({})
    coordinator = _enable_fake_async_consensus(transceiver, peer_ready=True)
    req = _FakeRequest(request_id=37)
    transceiver._wait_reqs[37] = req
    transceiver._transfer_worker.ready_request_ids.add(37)

    transceiver.exclude_context_requests_from_readiness([req])

    assert coordinator.poll_count == 0
    assert coordinator.ready_votes == []
    assert coordinator.ready_withdrawals == [(37, 0)]
    assert req.state == LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
    assert transceiver.owns_request(req)

    # The immediately following readiness progression observes the
    # cancellation marker and cannot publish a contradictory READY vote.
    transceiver.prepare_context_requests([req])
    assert coordinator.ready_votes == []


def test_default_off_cancel_removes_existing_legacy_readiness_waiter() -> None:
    transceiver = _make_transceiver({})
    transceiver._async_peer_ready_consensus_enabled = False
    req = _FakeRequest(request_id=39)
    transceiver._wait_reqs[39] = req
    transceiver._transfer_worker.ready_request_ids.add(39)

    transceiver.exclude_context_requests_from_readiness([req])

    assert 39 not in transceiver._wait_reqs
    assert req.state == LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
    transceiver.prepare_context_requests([])
    assert 39 not in transceiver._wait_reqs
    assert req.state == LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER


def test_async_peer_ready_abort_does_not_requeue_prepared_request() -> None:
    transceiver = _make_transceiver({})
    coordinator = _enable_fake_async_consensus(transceiver, peer_ready=True)
    req = _FakeRequest(request_id=33)
    transceiver._wait_reqs[33] = req
    transceiver._async_ready_published[33] = 0
    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_PREPARE,
            33,
            0,
            ConsensusOutcome.READY,
        )
    )
    transceiver._progress_async_consensus()

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_ABORT,
            33,
            0,
            ConsensusOutcome.WITHDRAWN,
        )
    )
    transceiver._progress_async_consensus()

    assert req.state == LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
    assert 33 not in transceiver._wait_reqs
    assert not transceiver._async_ready_prepared
    assert req._trtllm_async_ready_cancelled_epoch == 0
    assert coordinator.ready_abort_acks == [(33, 0)]


def test_async_peer_ready_cancel_after_ack_waits_for_completion() -> None:
    transceiver = _make_transceiver({})
    coordinator = _enable_fake_async_consensus(transceiver, peer_ready=True)
    req = _FakeRequest(request_id=34)
    transceiver._wait_reqs[34] = req
    transceiver._async_ready_published[34] = 0
    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_PREPARE,
            34,
            0,
            ConsensusOutcome.READY,
        )
    )
    transceiver._progress_async_consensus()

    assert not transceiver.cancel_request(req)
    assert coordinator.ready_withdrawals == []

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_RELEASE,
            34,
            0,
            ConsensusOutcome.READY,
        )
    )
    transceiver._progress_async_consensus()
    assert not transceiver.cancel_request(req)

    transceiver.activate_context_requests_for_schedule([req])
    assert not transceiver.cancel_request(req)

    coordinator.events.append(
        ConsensusEvent(
            ConsensusEventKind.READY_COMPLETE,
            34,
            0,
            ConsensusOutcome.READY,
        )
    )
    transceiver._progress_async_consensus()
    assert transceiver.cancel_request(req)
