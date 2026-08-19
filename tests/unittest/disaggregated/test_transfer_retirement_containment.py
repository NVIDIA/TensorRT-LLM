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
"""Regressions for fail-closed KV transfer retirement containment."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.transfer import (
    KVRecvTask,
    KVSendTask,
    MessageType,
    Receiver,
    RxSession,
    Sender,
    TaskStatus,
    TransferWorker,
    TxSession,
    WriteMeta,
)
from tensorrt_llm._torch.disaggregation.transceiver import (
    _NON_DRAINED_TRANSCEIVERS,
    KvCacheTransceiverV2,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


def _make_send_task(rid: int) -> KVSendTask:
    return KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )


def _make_owned_tx_session(rid: int, task: KVSendTask) -> TxSession:
    session = object.__new__(TxSession)
    session._base_args = SimpleNamespace(params=DisaggregatedParams(disagg_request_id=rid))
    session._timeout_s = 0.001
    session._overall_timeout_s = 0.001
    session._deadline_monotonic_s = 0.0
    session._need_aux = False
    session._enforce_physical_ownership = True
    session._sender = SimpleNamespace(
        capture_receiver_endpoints=Mock(return_value=set()),
        send_cancel_to_receivers=Mock(),
        clear_session=Mock(),
    )
    session.request_id = rid
    session._aux_buffer = None
    session.aux_slot = None
    session.receiver_ready = True
    session.kv_tasks = [task]
    session.aux_task = None
    session.lock = threading.Lock()
    session._exception = None
    session._closed = False
    session._terminal_status = None
    session.transfer_start_time = None
    session.transfer_end_time = None
    return session


def _make_fault_endpoint(cls, state):
    endpoint = object.__new__(cls)
    endpoint._enforce_physical_ownership = True
    endpoint._physical_ownership_fault_state = state
    endpoint._physical_ownership_fault = None
    endpoint._physical_ownership_fault_lock = state.lock
    return endpoint


class _ObservedLock:
    """Non-reentrant lock that exposes when another thread waits for it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner = None
        self.waiter = threading.Event()

    def __enter__(self):
        owner = threading.get_ident()
        if self._owner == owner:
            raise RuntimeError("lock re-entry")
        if self._lock.locked():
            self.waiter.set()
        self._lock.acquire()
        self._owner = owner
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self._owner = None
        self._lock.release()


@pytest.mark.cpu_only
def test_in_doubt_source_retires_only_after_backend_completion() -> None:
    completed = False
    status = SimpleNamespace(is_completed=lambda: completed)
    request = object()
    owner = transfer_mod._SendOperationOwner()
    owner.begin_operations({3})
    owner.retain_request(3, request)
    owner.attach_status(3, status)
    owner.mark_operation_in_doubt(3)

    assert owner.has_in_doubt_operations
    assert not owner.resources_drained

    completed = True

    assert owner.resources_drained
    assert not owner.has_in_doubt_operations


@pytest.mark.cpu_only
def test_shared_fault_blocks_opposite_direction_source_admission() -> None:
    rid = 401
    state = transfer_mod._PhysicalOwnershipFaultState()
    sender = _make_fault_endpoint(Sender, state)
    receiver = _make_fault_endpoint(Receiver, state)
    task = _make_send_task(rid)
    session = SimpleNamespace(
        lock=threading.Lock(),
        _closed=False,
        _has_logical_failure=Mock(return_value=False),
    )
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._shutdown_started = False

    receiver._record_physical_ownership_fault(RuntimeError("receive evidence lost"))
    admission = sender._begin_task_operation(task, 7)

    assert sender.physical_ownership_fault is receiver.physical_ownership_fault
    assert admission.newly_started == frozenset()
    assert admission.rejected_unsubmitted == frozenset({7})
    assert task.status == TaskStatus.ERROR
    assert task.resources_drained


@pytest.mark.cpu_only
def test_sender_listener_failure_poisoned_before_new_admission() -> None:
    callbacks = []
    state = transfer_mod._PhysicalOwnershipFaultState()
    sender = _make_fault_endpoint(Sender, state)
    sender._messenger = SimpleNamespace(start_listener=callbacks.append)
    sender._respond_with_kv = Mock(side_effect=RuntimeError("malformed request data"))

    sender._start_listener()
    callbacks[0]([b"peer", MessageType.REQUEST_DATA, b"payload"])

    assert isinstance(sender.physical_ownership_fault, RuntimeError)


@pytest.mark.cpu_only
def test_failed_terminal_abort_publication_poisoned() -> None:
    state = transfer_mod._PhysicalOwnershipFaultState()
    sender = _make_fault_endpoint(Sender, state)
    sender._registrar = SimpleNamespace(
        get_peer_rank_info=Mock(side_effect=RuntimeError("peer unavailable"))
    )
    info = SimpleNamespace(
        instance_name="gen",
        instance_rank=0,
        unique_rid=407,
        slice_id=0,
        owner_generation=1,
    )

    sender._send_failed_result_to_receiver(info)

    assert isinstance(sender.physical_ownership_fault, RuntimeError)


@pytest.mark.cpu_only
def test_receiver_owner_scan_does_not_reenter_session_lock() -> None:
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = _ObservedLock()
    receiver._sessions = {
        408: SimpleNamespace(
            _enforce_physical_ownership=True,
            resources_drained=Mock(return_value=True),
        )
    }

    assert not receiver.has_active_owners()


@pytest.mark.cpu_only
def test_partial_receive_publication_poisoned_and_retained() -> None:
    rid = 409
    state = transfer_mod._PhysicalOwnershipFaultState()
    receiver = _make_fault_endpoint(Receiver, state)
    receiver._bounce = SimpleNamespace(release_idle_reservation=Mock())
    receiver.send_cancel_to_senders = Mock()
    params = DisaggregatedParams(disagg_request_id=rid)
    task = KVRecvTask(
        rid,
        KVSlice(is_last_slice=True),
        slice_id=0,
        params=params,
        aux_slot=None,
        owner_generation=1,
    )
    task.begin_publication()

    def fail_after_seal(_task) -> None:
        _task.expected_transfers = 2
        _task.seal_writer_cohort({0, 1})
        _task.status = TaskStatus.TRANSFERRING
        raise RuntimeError("second writer publication failed")

    receiver.dispatch_task = Mock(side_effect=fail_after_seal)
    session = object.__new__(RxSession)
    session._receiver = receiver
    session._base_args = SimpleNamespace(params=params)
    session.request_id = rid
    session._kv_tasks = [task]
    session._sender_endpoints = {"tcp://ctx-0", "tcp://ctx-1"}
    session.lock = threading.Lock()
    session._terminal_status = None

    with pytest.raises(RuntimeError, match="second writer publication failed"):
        session.dispatch_prepared_receive(task)

    assert isinstance(receiver.physical_ownership_fault, RuntimeError)
    assert not task.resources_drained
    assert task.status == TaskStatus.ERROR
    receiver.send_cancel_to_senders.assert_called_once_with(rid, {"tcp://ctx-0", "tcp://ctx-1"})


@pytest.mark.cpu_only
def test_phase1_timeout_retains_source_until_backend_evidence() -> None:
    rid = 402
    task = _make_send_task(rid)
    task.begin_physical_operations({5})
    task.status = TaskStatus.TRANSFERRING
    session = _make_owned_tx_session(rid, task)

    assert session.wait_complete(blocking=True) == WaitResult.TIMEOUT
    assert not session.resources_drained()
    assert not session.close()

    task.finish_physical_operation(5)

    assert session.resources_drained()


def _make_delivery_sender(rid: int, session, dealer, state) -> Sender:
    sender = _make_fault_endpoint(Sender, state)
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._device_id = 0
    sender._instance_rank = 0
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="ctx", instance_rank=0)
    )
    status = SimpleNamespace(wait=Mock(return_value=True))
    sender._agent = SimpleNamespace(submit_transfer_requests=Mock(return_value=status))
    sender._bounce = SimpleNamespace(release_send=Mock())
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    sender._send_failed_write_meta_result = Mock()
    sender.send_cancel_to_receivers = Mock()
    return sender


def _make_write_meta(rid: int, task: KVSendTask) -> WriteMeta:
    one = np.array([1], dtype=np.int64)
    return WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen0",
        peer_rank=0,
        peer_endpoint="tcp://gen-0",
        unique_rid=rid,
        src_ptrs=one,
        dst_ptrs=one,
        sizes=one,
        dst_device_id=0,
        slice_id=0,
        is_last_slice=True,
        owner_generation=1,
    )


@pytest.mark.cpu_only
def test_ambiguous_result_publication_poisoned_without_contradictory_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 405
    task = _make_send_task(rid)
    task.begin_physical_operations({0})
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        lock=threading.Lock(),
        status=SessionStatus.TRANSFERRING,
        kv_tasks=[task],
        _enforce_physical_ownership=True,
        set_exception=Mock(),
        transfer_end_time=None,
    )
    state = transfer_mod._PhysicalOwnershipFaultState()
    dealer = SimpleNamespace(send=Mock(side_effect=RuntimeError("send failed")))
    sender = _make_delivery_sender(rid, session, dealer, state)
    sender._get_or_connect_dealer = Mock()
    monkeypatch.setattr(Sender, "_make_agent_request", Mock(return_value=object()))
    write_meta = _make_write_meta(rid, task)

    with pytest.raises(RuntimeError, match="publication is ambiguous"):
        sender._deliver_kv_to_agent(write_meta)

    assert write_meta.terminal_result_published is False
    assert sender.physical_ownership_fault is not None
    assert task.status == TaskStatus.ERROR
    assert task.resources_drained
    sender._send_failed_write_meta_result.assert_not_called()
    sender.send_cancel_to_receivers.assert_called()


@pytest.mark.cpu_only
def test_post_publication_diagnostics_cannot_poison_physical_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 406
    task = _make_send_task(rid)
    task.begin_physical_operations({0})
    task.status = TaskStatus.TRANSFERRING
    task.print_perf_info = Mock(side_effect=RuntimeError("diagnostics failed"))
    session = SimpleNamespace(
        lock=threading.Lock(),
        status=SessionStatus.TRANSFERRING,
        kv_tasks=[task],
        _enforce_physical_ownership=True,
        set_exception=Mock(),
        transfer_end_time=None,
    )
    state = transfer_mod._PhysicalOwnershipFaultState()
    dealer = SimpleNamespace(send=Mock())
    sender = _make_delivery_sender(rid, session, dealer, state)
    monkeypatch.setattr(Sender, "_make_agent_request", Mock(return_value=object()))
    write_meta = _make_write_meta(rid, task)

    sender._deliver_kv_to_agent(write_meta)

    assert write_meta.terminal_result_published is True
    assert sender.physical_ownership_fault is None
    assert task.status == TaskStatus.TRANSFERRED


def _make_owned_transceiver(session, request, worker) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    worker._config = SimpleNamespace(enforce_physical_ownership=True)
    transceiver._lifecycle_lock = threading.Lock()
    transceiver._shutdown_lock = threading.Lock()
    transceiver._shutdown_started = False
    transceiver._shutdown = False
    transceiver._send_sessions = {request.request_id: session}
    transceiver._send_reqs = {request.request_id: request}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._wait_reqs = {}
    transceiver._transfer_worker = worker
    return transceiver


@pytest.mark.cpu_only
def test_failed_shutdown_quarantines_roots_until_retry_proves_drain() -> None:
    rid = 403
    drained = False
    session = SimpleNamespace(
        disagg_request_id=rid,
        cancel_local=Mock(return_value=True),
        resources_drained=lambda: drained,
        close=Mock(return_value=True),
    )
    request = SimpleNamespace(request_id=rid)
    worker = SimpleNamespace(
        physical_ownership_fault=None,
        shutdown=Mock(return_value=True),
    )
    transceiver = _make_owned_transceiver(session, request, worker)
    _NON_DRAINED_TRANSCEIVERS.discard(transceiver)

    assert not transceiver.shutdown()
    assert transceiver in _NON_DRAINED_TRANSCEIVERS
    assert transceiver._send_sessions[rid] is session
    assert transceiver._send_reqs[rid] is request
    worker.shutdown.assert_not_called()

    drained = True

    assert transceiver.shutdown()
    assert transceiver not in _NON_DRAINED_TRANSCEIVERS
    assert transceiver._send_sessions == {}
    assert transceiver._send_reqs == {}
    worker.shutdown.assert_called_once_with()


@pytest.mark.cpu_only
def test_worker_shutdown_refuses_deregistration_while_owner_is_active() -> None:
    active = True
    sender = SimpleNamespace(
        _shutdown_started=False,
        has_active_owners=lambda: active,
        shutdown=Mock(return_value=True),
    )
    receiver = SimpleNamespace(
        _shutdown_started=False,
        has_active_owners=Mock(return_value=False),
        shutdown=Mock(return_value=True),
    )
    rank_server = SimpleNamespace(shutdown=Mock(return_value=True))
    bounce = SimpleNamespace(close=Mock())
    agent = SimpleNamespace(deregister_memory=Mock(), shutdown=Mock())
    worker = object.__new__(TransferWorker)
    worker._shutdown_lock = threading.Lock()
    worker._shutdown = False
    worker._shutdown_started = False
    worker._config = SimpleNamespace(enforce_physical_ownership=True)
    worker._rank_info_server = rank_server
    worker._sender = sender
    worker._receiver = receiver
    worker._bounce = bounce
    worker._agent = agent
    worker._registered_mem = ["kv"]

    assert not worker.shutdown()
    rank_server.shutdown.assert_not_called()
    agent.deregister_memory.assert_not_called()
    assert worker._registered_mem == ["kv"]

    active = False

    assert worker.shutdown()
    rank_server.shutdown.assert_called_once_with(strict=True)
    sender.shutdown.assert_called_once_with(strict=True)
    receiver.shutdown.assert_called_once_with(strict=True)
    agent.deregister_memory.assert_called_once_with("kv")
    agent.shutdown.assert_called_once_with()


@pytest.mark.cpu_only
def test_shutdown_linearizes_with_writer_admission() -> None:
    rid = 410
    state = transfer_mod._PhysicalOwnershipFaultState()
    state.lock = _ObservedLock()
    sender = _make_fault_endpoint(Sender, state)
    task = _make_send_task(rid)
    entered, release = threading.Event(), threading.Event()
    begin = task.begin_physical_operations

    def pause_begin(peer_ranks):
        entered.set()
        assert release.wait(timeout=1)
        return begin(peer_ranks)

    task.begin_physical_operations = pause_begin
    session = SimpleNamespace(
        lock=threading.Lock(),
        _closed=False,
        _has_logical_failure=Mock(return_value=False),
        _enforce_physical_ownership=True,
        resources_drained=lambda: task.resources_drained,
    )
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._shutdown_started = False
    worker = object.__new__(TransferWorker)
    worker._shutdown_lock = threading.Lock()
    worker._shutdown = False
    worker._config = SimpleNamespace(enforce_physical_ownership=True)
    worker._physical_ownership_fault_state = state
    worker._rank_info_server = SimpleNamespace(shutdown=Mock())
    worker._sender, worker._receiver = sender, None
    worker._agent = SimpleNamespace(deregister_memory=Mock())
    worker._registered_mem = ["kv"]

    admission = threading.Thread(target=sender._begin_task_operation, args=(task, 0))
    shutdown_result = []
    shutdown = threading.Thread(target=lambda: shutdown_result.append(worker.shutdown()))
    admission.start()
    assert entered.wait(timeout=1)
    shutdown.start()
    assert state.lock.waiter.wait(timeout=1)
    release.set()
    admission.join(timeout=1)
    shutdown.join(timeout=1)

    assert shutdown_result == [False]
    worker._agent.deregister_memory.assert_not_called()
    assert worker._registered_mem == ["kv"]


@pytest.mark.cpu_only
def test_legacy_worker_listener_failure_prevents_memory_release() -> None:
    sender = SimpleNamespace(shutdown=Mock(side_effect=RuntimeError("listener still active")))
    receiver = SimpleNamespace(shutdown=Mock())
    agent = SimpleNamespace(deregister_memory=Mock(), shutdown=Mock())
    worker = object.__new__(TransferWorker)
    worker._shutdown_lock = threading.Lock()
    worker._shutdown = False
    worker._config = SimpleNamespace(enforce_physical_ownership=False)
    worker._rank_info_server = SimpleNamespace(shutdown=Mock())
    worker._sender = sender
    worker._receiver = receiver
    worker._bounce = SimpleNamespace(close=Mock())
    worker._agent = agent
    worker._registered_mem = ["kv"]

    with pytest.raises(RuntimeError, match="listener still active"):
        worker.shutdown()

    receiver.shutdown.assert_not_called()
    agent.deregister_memory.assert_not_called()
    agent.shutdown.assert_not_called()
    assert worker._registered_mem == ["kv"]


@pytest.mark.cpu_only
def test_partial_worker_init_without_endpoints_can_shutdown() -> None:
    rank_server = SimpleNamespace(shutdown=Mock(return_value=True))
    worker = object.__new__(TransferWorker)
    worker._shutdown_lock = threading.Lock()
    worker._shutdown = False
    worker._config = SimpleNamespace(enforce_physical_ownership=True)
    worker._rank_info_server = rank_server

    assert worker.shutdown()
    rank_server.shutdown.assert_called_once_with(strict=True)


@pytest.mark.cpu_only
def test_executor_shutdown_proves_transfer_drain_before_freeing_memory() -> None:
    order = []
    transceiver = SimpleNamespace(
        requires_physical_drain_before_request_release=True,
        shutdown=Mock(side_effect=lambda: order.append("transfer") or True),
    )
    manager = SimpleNamespace(shutdown=Mock(side_effect=lambda: order.append("manager")))
    executor = SimpleNamespace(
        kv_cache_transceiver=transceiver,
        resource_manager=SimpleNamespace(resource_managers={"kv": manager}),
    )

    PyExecutor._shutdown_resource_managers(executor)

    assert order == ["transfer", "manager"]
