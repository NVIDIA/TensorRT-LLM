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

import threading
import weakref
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native import transfer as transfer_module
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVSendTask,
    MessageType,
    Receiver,
    RecvReqInfo,
    RxSession,
    Sender,
    TaskStatus,
    TransferWorker,
    TxSession,
    _ControlPlane,
    _decode_protocol_capabilities,
    _encode_protocol_capabilities,
)


class _Task:
    def __init__(self, status: TaskStatus, *, expected_transfers: int = 1, slice_id: int = 0):
        self.status = status
        self.expected_transfers = expected_transfers
        self.slice_id = slice_id
        self.last_slice_count = 0
        self._perf_timer = None
        self._event = threading.Event()
        if status in (TaskStatus.TRANSFERRED, TaskStatus.ERROR):
            self._event.set()

    @property
    def is_done(self) -> bool:
        return self._event.is_set()

    def fail(self, _error: Exception) -> None:
        self.status = TaskStatus.ERROR
        self._event.set()

    def complete(self) -> None:
        self.status = TaskStatus.TRANSFERRED
        self._event.set()

    def print_perf_info(self, *_args) -> None:
        pass


class _Sender:
    def __init__(self):
        self.cancelled: list[int] = []
        self.acks: list[tuple[str, int, bool]] = []
        self.ack_epochs: list[int | None] = []
        self.cleared: list[int] = []

    def send_cancel_to_receivers(self, unique_rid: int) -> None:
        self.cancelled.append(unique_rid)

    def send_cancel_ack(
        self,
        endpoint: str,
        unique_rid: int,
        *,
        request_epoch=None,
        from_worker: bool,
    ) -> bool:
        self.acks.append((endpoint, unique_rid, from_worker))
        self.ack_epochs.append(request_epoch)
        return True

    def clear_session(self, unique_rid: int, *_args) -> None:
        self.cleared.append(unique_rid)

    def _get_req_info(self, _unique_rid: int) -> dict:
        return {}

    def dispatch_task(self, _task, _snapshot, *, operation_owner: TxSession) -> None:
        operation_owner.finish_dispatch([])


class _Bounce:
    def __init__(self):
        self.released: list[tuple[int, int]] = []
        self.orphaned: list[tuple[int, int]] = []
        self.drained: list[tuple[int, int]] = []
        self.settlement_callbacks = {}

    def release_idle_reservation(self, rid_slice: tuple[int, int]) -> None:
        self.released.append(rid_slice)

    def orphan_reservation(self, rid_slice: tuple[int, int]) -> None:
        if rid_slice not in self.orphaned:
            self.orphaned.append(rid_slice)

    def confirm_drained(self, rid_slice: tuple[int, int]) -> None:
        self.drained.append(rid_slice)
        callback = self.settlement_callbacks.pop(rid_slice, None)
        if callback is not None:
            callback(False)

    def set_completion_callback(self, rid_slice, on_settled) -> None:
        assert rid_slice not in self.settlement_callbacks
        self.settlement_callbacks[rid_slice] = on_settled

    def is_bounced(self, _rid_slice: tuple[int, int]) -> bool:
        return False

    def record_result(self, *_args, **_kwargs) -> None:
        callback = _kwargs.get("on_done")
        if callback is None and len(_args) >= 6:
            callback = _args[5]
        if callback is not None:
            callback(True)

    def record_failure(self, *_args, **_kwargs) -> None:
        callback = _kwargs.get("on_done")
        if callback is not None:
            callback(False)


class _InlineBounce(_Bounce):
    def is_bounced(self, _rid_slice: tuple[int, int]) -> bool:
        return True


class _RaisingBounce(_InlineBounce):
    def record_result(self, *_args, **_kwargs) -> None:
        raise RuntimeError("injected scatter setup failure")


class _Receiver:
    endpoint = "tcp://receiver"

    def __init__(self):
        self._bounce = _Bounce()
        self._registrar = SimpleNamespace(
            self_rank_info=SimpleNamespace(instance_name="gen", instance_rank=0)
        )
        self.cancelled: list[tuple[int, set[str], set[str]]] = []
        self.retained: list[RxSession] = []
        self.cleared: list[int] = []

    def send_cancel_to_senders(
        self,
        unique_rid: int,
        endpoints: set[str],
        ack_capable_endpoints: set[str],
    ) -> None:
        self.cancelled.append((unique_rid, endpoints, ack_capable_endpoints))

    def retain_draining_session(self, session: RxSession) -> None:
        self.retained.append(session)

    def clear_session(self, unique_rid: int, *_args) -> None:
        self.cleared.append(unique_rid)
        self.retained.clear()

    def _fail_protocol(self, _error: RuntimeError) -> None:
        pass


class _AuxBuffer:
    def __init__(self):
        self.freed: list[int] = []

    def free_slot(self, slot: int) -> None:
        self.freed.append(slot)


class _ShutdownMessenger:
    endpoint = "tcp://receiver"

    def __init__(self, order: list[str]):
        self._listener_thread = None
        self._order = order
        self.stopped = threading.Event()

    def stop(self) -> None:
        self._order.append("listener-stop")
        self.stopped.set()


class _ShutdownBounce(_Bounce):
    def __init__(self, order: list[str]):
        super().__init__()
        self._order = order
        self._scatter_thread = None
        self.closed = threading.Event()

    def close(self) -> None:
        self._order.append("bounce-close")
        self.closed.set()


class _ShutdownAgent:
    def __init__(self, order: list[str]):
        self._order = order

    def deregister_memory(self, _desc) -> None:
        self._order.append("deregister-memory")

    def shutdown(self) -> None:
        self._order.append("agent-shutdown")


def _make_tx_session(task_status: TaskStatus = TaskStatus.INIT) -> tuple[TxSession, object]:
    task = _Task(task_status)
    session = object.__new__(TxSession)
    session.lock = threading.Lock()
    session._sealed = False
    session._closed = False
    session._close_requested = False
    session._terminal_status = None
    session._terminal_snapshot = None
    session._exception = None
    session._outstanding_operations = 0
    session._cancel_ack_endpoints = set()
    session._cancel_acked_endpoints = set()
    session._request_operations = set()
    session._cancel_ack_operations = set()
    session._cancel_acked_operations = set()
    session._cancel_notified = False
    session._need_aux = False
    session.kv_tasks = [task]
    session.aux_task = None
    session._sender = _Sender()
    session._aux_buffer = None
    session.aux_slot = None
    session._base_args = SimpleNamespace(
        params=SimpleNamespace(disagg_request_id=11, ctx_request_id=None),
        prompt_len=None,
        beam_width=1,
    )
    session.request_id = 11
    return session, task


def _make_rx_session(task_status: TaskStatus = TaskStatus.INIT) -> tuple[RxSession, object]:
    task = _Task(task_status)
    session = object.__new__(RxSession)
    session.lock = threading.Lock()
    session._sealed = False
    session._closed = False
    session._close_requested = False
    session._terminal_status = None
    session._exception = None
    session._kv_tasks = [task]
    session._need_aux = False
    session._outstanding_operations = 0
    session._expected_operations = {}
    session._retired_operation_keys = set()
    session._pending_scatter_callbacks = 0
    session._pending_bounce_settlements = set()
    session._aux_obligations_reserved = False
    session._aux_obligation_slice_id = None
    session._aux_count = 0
    session._aux_status = TaskStatus.INIT
    session._sender_endpoints = set()
    session._cancel_pending_endpoints = set()
    session._cancel_pending_operations = set()
    session._ack_capable_endpoints = set()
    session._v2_enabled = None
    session._legacy_rank_cohorts = {}
    session._legacy_bound_cohorts = {}
    session.request_epoch = 123
    session._receiver = _Receiver()
    session._aux_buffer = None
    session.aux_slot = None
    session._base_args = SimpleNamespace(
        params=SimpleNamespace(disagg_request_id=17, ctx_request_id=None)
    )
    session.request_id = 17
    return session, task


def test_tx_seal_rejects_an_already_credited_queued_operation() -> None:
    session, task = _make_tx_session()
    session._outstanding_operations = 1

    assert not session.seal_and_check_quiescent()
    assert not session.try_mark_transferring(task)
    assert task.status == TaskStatus.INIT

    session.retire_operation()
    assert not session.has_transferring_tasks()


def test_tx_quiescence_includes_aux_transfer() -> None:
    session, _ = _make_tx_session(TaskStatus.TRANSFERRED)
    session.aux_task = SimpleNamespace(status=TaskStatus.TRANSFERRING)
    session._outstanding_operations = 1

    assert not session.seal_and_check_quiescent()
    assert session.has_transferring_tasks()


def test_tx_exception_does_not_hide_an_active_transfer() -> None:
    session, task = _make_tx_session(TaskStatus.TRANSFERRING)
    session._outstanding_operations = 2

    session.set_exception("peer transfer failed")

    assert task.status == TaskStatus.TRANSFERRING
    assert not session.seal_and_check_quiescent()

    session.retire_operation()
    assert session.has_transferring_tasks()
    session.retire_operation()
    assert not session.has_transferring_tasks()


def test_tx_cancel_ack_waits_for_every_peer_obligation() -> None:
    session, _ = _make_tx_session(TaskStatus.TRANSFERRING)
    session._outstanding_operations = 2

    session.cancel(ack_endpoint="tcp://receiver")
    session.retire_operation()
    assert session._sender.acks == []

    session.retire_operation()
    assert session._sender.acks == [("tcp://receiver", 11, True)]

    session.cancel(ack_endpoint="tcp://receiver")
    assert session._sender.acks == [
        ("tcp://receiver", 11, True),
        ("tcp://receiver", 11, False),
    ]
    assert session._sender.cancelled == [11]


def test_tx_v2_cancel_after_terminal_snapshot_still_acknowledges_drain() -> None:
    session, _ = _make_tx_session(TaskStatus.TRANSFERRED)
    session.register_request_operation("tcp://receiver", 123)
    session._terminal_snapshot = SessionStatus.KV_TRANSFERRED

    session.cancel(ack_endpoint="tcp://receiver", request_epoch=123)

    assert session._sender.acks == [("tcp://receiver", 11, False)]
    assert session._sender.ack_epochs == [123]

    # The receiver retransmits CANCEL until it observes an ACK. A duplicate
    # request after drain must resend the same idempotent epoch-bound ACK.
    session.cancel(ack_endpoint="tcp://receiver", request_epoch=123)
    assert session._sender.acks == [
        ("tcp://receiver", 11, False),
        ("tcp://receiver", 11, False),
    ]
    assert session._sender.ack_epochs == [123, 123]


def test_tx_v2_cancel_tracks_every_receiver_operation() -> None:
    session, _ = _make_tx_session(TaskStatus.TRANSFERRED)
    session.register_request_operation("tcp://receiver-0", 123)
    session.register_request_operation("tcp://receiver-1", 123)

    session.cancel(ack_endpoint="tcp://receiver-0", request_epoch=123)
    session.cancel(ack_endpoint="tcp://receiver-1", request_epoch=123)

    assert set(session._sender.acks) == {
        ("tcp://receiver-0", 11, False),
        ("tcp://receiver-1", 11, False),
    }
    assert session._sender.ack_epochs == [123, 123]


def test_tx_close_defers_aux_reclamation_until_peer_response() -> None:
    session, _ = _make_tx_session(TaskStatus.TRANSFERRING)
    aux_buffer = _AuxBuffer()
    session._aux_buffer = aux_buffer
    session.aux_slot = 7
    session._outstanding_operations = 1

    session.close()
    assert aux_buffer.freed == []

    session.retire_operation()
    assert aux_buffer.freed == [7]
    assert session._sender.cleared == [11]


def test_tx_terminal_snapshot_does_not_freeze_pending_dispatch() -> None:
    session, task = _make_tx_session(TaskStatus.TRANSFERRED)
    session._outstanding_operations = 1

    assert session.seal_and_snapshot_terminal() is None
    assert not session._sealed

    session.retire_operation()
    assert session.seal_and_snapshot_terminal() == SessionStatus.KV_TRANSFERRED
    assert task.status == TaskStatus.TRANSFERRED


def test_tx_terminal_snapshot_is_immutable_across_cancel_race() -> None:
    for _ in range(100):
        session, _ = _make_tx_session(TaskStatus.TRANSFERRED)
        barrier = threading.Barrier(3)

        def snapshot() -> None:
            barrier.wait()
            session.seal_and_snapshot_terminal()

        def cancel() -> None:
            barrier.wait()
            session.cancel()

        snapshot_thread = threading.Thread(target=snapshot)
        cancel_thread = threading.Thread(target=cancel)
        snapshot_thread.start()
        cancel_thread.start()
        barrier.wait()
        snapshot_thread.join()
        cancel_thread.join()

        snapshot_status = session.seal_and_snapshot_terminal()
        assert snapshot_status in (SessionStatus.KV_TRANSFERRED, SessionStatus.CANCELLED)
        assert session.status == snapshot_status


def test_tx_pre_cancel_rejects_kv_and_aux_without_throwing() -> None:
    session, _ = _make_tx_session()
    session.kv_tasks.clear()

    session.cancel()
    session.send(object())
    aux_task = session.send_aux()

    assert session.kv_tasks[-1].status == TaskStatus.ERROR
    assert aux_task.status == TaskStatus.ERROR


def test_tx_cancel_between_kv_and_aux_keeps_both_terminal() -> None:
    session, _ = _make_tx_session()
    session.kv_tasks.clear()

    session.send(object())
    session.cancel()
    aux_task = session.send_aux()

    assert session.kv_tasks[0].status == TaskStatus.ERROR
    assert aux_task.status == TaskStatus.ERROR


def test_rx_seal_blocks_transfer_start_and_endpoint_publication() -> None:
    session, task = _make_rx_session()

    assert session.seal_and_check_quiescent()
    assert not session.mark_transferring(0, {"tcp://sender"})
    assert task.status == TaskStatus.INIT
    assert session._sender_endpoints == set()


def test_rx_seal_waits_for_started_transfer() -> None:
    session, task = _make_rx_session()
    task.expected_transfers = 1

    assert session.mark_transferring(
        0,
        {0: "tcp://sender"},
        {"tcp://sender"},
        request_epoch=session.request_epoch,
    )
    assert not session.seal_and_check_quiescent()
    assert task.status == TaskStatus.TRANSFERRING
    assert session._sender_endpoints == {"tcp://sender"}


@pytest.mark.parametrize(
    "peer_rank,sender_endpoint,request_epoch,error",
    [
        (1, "tcp://sender", 123, "unexpected native-transfer result operation"),
        (0, "tcp://other", 123, "result source mismatch"),
        (0, "tcp://sender", 124, "result epoch mismatch"),
    ],
)
def test_rx_wrong_v2_result_identity_fails_closed_without_retiring_credit(
    peer_rank, sender_endpoint, request_epoch, error
) -> None:
    session, task = _make_rx_session()
    task.expected_transfers = 1
    assert session.mark_transferring(
        0,
        {0: "tcp://sender"},
        {"tcp://sender"},
        request_epoch=session.request_epoch,
    )

    session.process_kv_agent_result(
        peer_rank,
        0,
        True,
        AgentResult.SUCCESS,
        sender_endpoint=sender_endpoint,
        request_epoch=request_epoch,
    )

    assert session._outstanding_operations == 1
    assert task.status == TaskStatus.TRANSFERRING
    assert session.status == SessionStatus.ERROR
    assert error in str(session.exception)
    assert session._receiver.cancelled == [(17, {"tcp://sender"}, {("tcp://sender", 123)})]


def test_rx_exact_v2_result_retires_once_and_duplicate_is_idempotent() -> None:
    session, task = _make_rx_session()
    task.expected_transfers = 1
    assert session.mark_transferring(
        0,
        {0: "tcp://sender"},
        {"tcp://sender"},
        request_epoch=session.request_epoch,
    )

    for _ in range(2):
        session.process_kv_agent_result(
            0,
            0,
            True,
            AgentResult.SUCCESS,
            sender_endpoint="tcp://sender",
            request_epoch=session.request_epoch,
        )

    assert session._outstanding_operations == 0
    assert task.status == TaskStatus.TRANSFERRED
    assert session._retired_operation_keys == {("kv", 0, 0)}


def test_rx_generation_first_aux_obligations_block_quiescence() -> None:
    session, task = _make_rx_session()
    session._need_aux = True
    task.expected_transfers = 2

    assert session.mark_transferring(0, {"tcp://sender-0", "tcp://sender-1"})
    assert session._outstanding_operations == 4
    with session.lock:
        session._retire_receive_operation_unlocked(("kv", 0, 0))
        session._retire_receive_operation_unlocked(("kv", 0, 1))

    assert session._outstanding_operations == 2
    assert session.has_transferring_tasks()


def test_rx_cancel_ack_defers_aux_and_bounce_reclamation_until_drain() -> None:
    session, task = _make_rx_session()
    session._need_aux = True
    task.expected_transfers = 1
    aux_buffer = _AuxBuffer()
    session._aux_buffer = aux_buffer
    session.aux_slot = 3

    assert session.mark_transferring(
        0,
        {0: "tcp://sender"},
        {"tcp://sender"},
        request_epoch=session.request_epoch,
    )
    session.cancel()
    session.close()

    assert aux_buffer.freed == []
    assert session._receiver.cleared == []
    assert session._receiver._bounce.orphaned == [(17, 0)]

    session.process_cancel_ack("tcp://sender", session.request_epoch)

    assert aux_buffer.freed == [3]
    assert session._receiver.cleared == [17]
    assert session._receiver._bounce.drained == []


def test_rx_bounced_cancel_before_result_retires_unconditional_settlement_credit() -> None:
    session, task = _make_rx_session()
    task.expected_transfers = 1
    assert session.mark_transferring(
        0,
        {0: "tcp://sender"},
        {"tcp://sender"},
        request_epoch=session.request_epoch,
        bounced=True,
    )
    rid_slice = (session.disagg_request_id, task.slice_id)
    session._receiver._bounce.set_completion_callback(
        rid_slice, session._make_bounce_settlement_callback(task)
    )

    session.cancel()
    session.close()
    assert session._pending_bounce_settlements == {0}
    assert session._receiver.cleared == []

    session.process_cancel_ack("tcp://sender", session.request_epoch)

    assert session._receiver._bounce.drained == [rid_slice]
    assert session._pending_bounce_settlements == set()
    assert session._receiver.cleared == [17]


def test_rx_malformed_bounce_result_keeps_credit_until_exact_ack_settles() -> None:
    session, task = _make_rx_session()
    task.expected_transfers = 1
    session._receiver._bounce = _RaisingBounce()
    session._receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="gen", instance_rank=0)
    )
    assert session.mark_transferring(
        0,
        {0: "tcp://sender"},
        {"tcp://sender"},
        request_epoch=session.request_epoch,
        bounced=True,
    )
    rid_slice = (session.disagg_request_id, task.slice_id)
    session._receiver._bounce.set_completion_callback(
        rid_slice, session._make_bounce_settlement_callback(task)
    )

    session.process_kv_agent_result(
        0,
        0,
        True,
        AgentResult.SUCCESS,
        sender_endpoint="tcp://sender",
        request_epoch=session.request_epoch,
    )

    assert task.status == TaskStatus.ERROR
    assert session._outstanding_operations == 1
    assert session._pending_bounce_settlements == {0}
    session.close()

    session.process_cancel_ack("tcp://sender", session.request_epoch)

    assert session._receiver._bounce.drained == [rid_slice]
    assert session._pending_bounce_settlements == set()
    assert session._receiver.cleared == [17]


def test_rx_legacy_cancel_finalizes_after_kv_and_aux_terminal_results() -> None:
    session, task = _make_rx_session()
    session._need_aux = True
    task.expected_transfers = 1
    aux_buffer = _AuxBuffer()
    session._aux_buffer = aux_buffer
    session.aux_slot = 4

    assert session.mark_transferring(0, {"tcp://legacy-sender"}, set())
    session.cancel()
    session.close()

    session.process_kv_agent_result(0, 0, True, AgentResult.FAILED)
    assert aux_buffer.freed == []
    session.process_aux_agent_result(0, AgentResult.FAILED)

    assert aux_buffer.freed == [4]
    assert session._receiver.cleared == [17]


def test_rx_bounce_fallback_inline_callback_does_not_reenter_session_lock() -> None:
    session, task = _make_rx_session(TaskStatus.TRANSFERRING)
    task.expected_transfers = 1
    session._outstanding_operations = 1
    session._receiver._bounce = _InlineBounce()
    session._receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="gen", instance_rank=0)
    )

    finished = threading.Event()

    def deliver_result() -> None:
        session.process_kv_agent_result(
            0,
            0,
            True,
            AgentResult.SUCCESS,
        )
        finished.set()

    thread = threading.Thread(target=deliver_result)
    thread.start()
    thread.join(timeout=1)

    assert finished.is_set(), "inline bounce completion deadlocked on RxSession.lock"
    assert task.status == TaskStatus.TRANSFERRED
    assert session._outstanding_operations == 0
    assert session._pending_scatter_callbacks == 0


def test_rx_scatter_setup_failure_releases_only_after_terminal_result_proof() -> None:
    session, task = _make_rx_session(TaskStatus.TRANSFERRING)
    task.expected_transfers = 1
    session._outstanding_operations = 1
    session._receiver._bounce = _RaisingBounce()
    session._pending_bounce_settlements = {0}
    session._pending_scatter_callbacks = 1
    session._receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="gen", instance_rank=0)
    )

    session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS)

    assert task.status == TaskStatus.ERROR
    assert session._outstanding_operations == 1
    assert session._pending_scatter_callbacks == 1
    assert session._receiver._bounce.orphaned == [(17, 0)]
    assert session._receiver._bounce.drained == []


def test_rx_scatter_callback_credit_defers_close() -> None:
    session, task = _make_rx_session(TaskStatus.TRANSFERRED)
    task.expected_transfers = 1
    session._pending_scatter_callbacks = 1
    aux_buffer = _AuxBuffer()
    session._aux_buffer = aux_buffer
    session.aux_slot = 5

    session.close()
    assert aux_buffer.freed == []

    session._retire_scatter_callback()
    assert aux_buffer.freed == [5]


def test_worker_shutdown_keeps_rx_progress_and_memory_alive_until_cancel_ack() -> None:
    order: list[str] = []
    bounce = _ShutdownBounce(order)
    messenger = _ShutdownMessenger(order)

    receiver = object.__new__(Receiver)
    receiver._shutdown = False
    receiver._shutdown_lock = threading.Lock()
    receiver._shutdown_complete = threading.Event()
    receiver._shutdown_thread = None
    receiver._shutdown_error = None
    receiver._messenger = messenger
    receiver._bounce = bounce
    receiver._control = SimpleNamespace(flush=lambda: None, shutdown=lambda: None)
    receiver._sessions = {}
    receiver._sessions_lock = threading.Lock()
    receiver._sessions_drained = threading.Condition(receiver._sessions_lock)
    receiver._draining_sessions = {}
    receiver._pre_cancelled_rids = {}
    receiver._closed_rids = OrderedDict()

    session, task = _make_rx_session(TaskStatus.TRANSFERRING)
    task.expected_transfers = 1
    session._receiver = receiver
    session._terminal_status = SessionStatus.CANCELLED
    session._outstanding_operations = 1
    session._sender_endpoints = {"tcp://sender"}
    session._ack_capable_endpoints = {"tcp://sender"}
    session._cancel_pending_endpoints = {"tcp://sender"}
    aux_buffer = _AuxBuffer()
    session._aux_buffer = aux_buffer
    session.aux_slot = 9
    receiver._sessions[session.disagg_request_id] = weakref.ref(session)

    drain_started = threading.Event()
    retain_draining_session = receiver.retain_draining_session

    def retain_and_signal(rx_session: RxSession) -> None:
        retain_draining_session(rx_session)
        drain_started.set()

    receiver.retain_draining_session = retain_and_signal

    worker = object.__new__(TransferWorker)
    worker._shutdown = False
    worker._shutdown_lock = threading.Lock()
    worker._shutdown_complete = threading.Event()
    worker._shutdown_thread = None
    worker._shutdown_error = None
    worker._rank_info_server = None
    worker._sender = SimpleNamespace(
        shutdown=lambda: order.append("sender-stop"),
        _worker_threads=[],
        _messenger=None,
    )
    worker._receiver = receiver
    worker._bounce = bounce
    worker._registered_mem = [object()]
    worker._agent = _ShutdownAgent(order)

    # Model shutdown requested from a bounce completion callback. It must
    # return and let that callback process the final ACK rather than joining
    # itself or blocking the callback's operation-credit retirement.
    bounce._scatter_thread = threading.current_thread()
    shutdown_complete = worker.shutdown()

    assert drain_started.wait(timeout=1)
    assert shutdown_complete is worker._shutdown_complete
    assert not messenger.stopped.is_set()
    assert not bounce.closed.is_set()
    assert aux_buffer.freed == []
    assert "deregister-memory" not in order

    session.process_cancel_ack("tcp://sender")

    assert worker._shutdown_complete.wait(timeout=1)
    assert session._closed
    assert aux_buffer.freed == [9]
    assert order == [
        "sender-stop",
        "listener-stop",
        "bounce-close",
        "deregister-memory",
        "agent-shutdown",
    ]


def test_rx_pre_cancel_receive_is_nonthrowing() -> None:
    session, _ = _make_rx_session()
    session._kv_tasks.clear()

    session.cancel()
    session.receive(object())

    assert session._kv_tasks[-1].status == TaskStatus.ERROR


def test_sender_closed_tombstone_rejects_late_request() -> None:
    sender = object.__new__(Sender)
    sender._shutdown = True
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._closed_rids = OrderedDict([(29, None)])
    sender._pre_cancelled_rids = {}
    sender._peer_capabilities = {}
    sender._registrar = SimpleNamespace(
        get_peer_rank_info=lambda _name, _rank: SimpleNamespace(self_endpoint="tcp://receiver")
    )
    failed: list[int] = []
    saved: list[int] = []
    sender._send_failed_result_to_receiver = lambda info, **_kwargs: failed.append(info.unique_rid)
    sender._save_peer_req_info = lambda info: saved.append(info.unique_rid)
    info = RecvReqInfo(
        sender_req_id=1,
        instance_name="gen",
        instance_rank=0,
        block_ids_per_layer_groups=[np.array([], dtype=np.int64)],
        unique_rid=29,
    )

    sender._respond_with_kv(b"", [MessageType.REQUEST_DATA, info.to_bytes()])

    assert failed == [29]
    assert saved == []


def test_closed_tombstones_are_bounded() -> None:
    tombstones: OrderedDict[int, None] = OrderedDict()
    original_limit = Receiver._TOMBSTONE_LIMIT
    Receiver._TOMBSTONE_LIMIT = 3
    try:
        for request_id in range(5):
            Receiver._remember_tombstone(tombstones, request_id)
    finally:
        Receiver._TOMBSTONE_LIMIT = original_limit

    assert list(tombstones) == [2, 3, 4]


def test_receiver_epoch_tombstone_drops_only_matching_delayed_result() -> None:
    receiver = object.__new__(Receiver)
    receiver._sessions = {}
    receiver._sessions_lock = threading.Lock()
    receiver._closed_operations = OrderedDict([((17, 123), None)])
    receiver._protocol_error = None
    matching = transfer_module._make_kv_result_msg(
        0,
        17,
        0,
        True,
        AgentResult.SUCCESS,
        request_epoch=123,
        sender_endpoint="tcp://sender",
    )
    receiver._process_kv_agent_result(b"", matching)
    assert receiver._protocol_error is None

    stale = transfer_module._make_kv_result_msg(
        0,
        17,
        0,
        True,
        AgentResult.SUCCESS,
        request_epoch=124,
        sender_endpoint="tcp://sender",
    )
    receiver._process_kv_agent_result(b"", stale)
    assert receiver._protocol_error is not None
    assert "unknown request incarnation" in str(receiver._protocol_error)


def test_sender_acknowledged_pre_cancel_is_not_count_evicted() -> None:
    sender = object.__new__(Sender)
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._ingress_lock = threading.Lock()
    sender._shutdown = False
    sender._pre_cancelled_rids = {}
    sender._pre_cancelled_operations = {}
    sender._cancelled_operation_tombstones = OrderedDict()
    sender._closed_rids = OrderedDict()
    sender._protocol_error = None
    sender._peer_requests = {}
    sender._peer_requests_lock = threading.Lock()

    original_limit = Sender._TOMBSTONE_LIMIT
    Sender._TOMBSTONE_LIMIT = 3
    try:
        for request_id in range(5):
            sender._remember_pre_cancelled_unlocked(request_id)
    finally:
        Sender._TOMBSTONE_LIMIT = original_limit

    class _DelayedSession:
        disagg_request_id = 0

        def __init__(self):
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    session = _DelayedSession()
    sender.setup_session(session)

    assert session.cancelled
    assert 0 not in sender._pre_cancelled_rids


def test_sender_shutdown_gate_rejects_enqueue_after_worker_sentinels() -> None:
    sender = object.__new__(Sender)
    sender._shutdown = True
    sender._ingress_lock = threading.Lock()
    sender._num_threads = 1
    sender._send_task_queues = [SimpleNamespace(put=lambda _item: None)]
    write_meta = SimpleNamespace(unique_rid=31, peer_rank=0)

    with pytest.raises(RuntimeError, match="shutting down"):
        sender._enqueue(write_meta)


def test_sender_metadata_failure_retains_owner_if_notification_cannot_queue() -> None:
    sender = object.__new__(Sender)
    sender._stalled_operations_lock = threading.Lock()
    sender._stalled_session_owners = []

    def fail_metadata(_task, _info):
        raise RuntimeError("injected metadata failure")

    sender._build_kv_write_meta = fail_metadata
    sender._send_failed_result_to_receiver = lambda _info, **_kwargs: False

    task = object.__new__(KVSendTask)
    task.status = TaskStatus.INIT
    task._event = threading.Event()
    task._exception = None
    task._unique_rid = 11
    task._perf_timer = None
    owner, _ = _make_tx_session()
    owner._outstanding_operations = 1
    infos = {
        0: SimpleNamespace(instance_rank=0),
        1: SimpleNamespace(instance_rank=1),
    }

    sender.dispatch_task(task, infos, operation_owner=owner)

    assert task.status == TaskStatus.ERROR
    assert owner._outstanding_operations == 2
    assert sender._stalled_session_owners == [owner]


def test_native_protocol_capabilities_are_optional_and_backward_compatible() -> None:
    legacy = _decode_protocol_capabilities(None)
    negotiated = _decode_protocol_capabilities(_encode_protocol_capabilities())

    assert legacy.version == 1
    assert not legacy.drain_ack
    assert negotiated.version >= 2
    assert negotiated.drain_ack

    with pytest.raises(RuntimeError, match="invalid native-transfer capability"):
        _decode_protocol_capabilities(b"not-a-capability-frame")


def test_explicit_async_mode_rejects_legacy_peer_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LegacyInfoMessenger:
        def __init__(self, *, mode: str, endpoint: str):
            assert mode == "DEALER"
            assert endpoint == "tcp://legacy-info"

        def send(self, message: list[bytes]) -> None:
            assert message == [MessageType.REQUEST_INSTANCE_INFO]

        def receive(self) -> list[bytes]:
            # No optional capability frame means protocol v1.
            return [b"legacy-rank-info"]

        def stop(self) -> None:
            pass

    receiver = object.__new__(Receiver)
    receiver._sender_ep_instance_map = {}
    receiver._sender_info_capabilities = {}
    receiver._sender_endpoint_capabilities = {}
    receiver._registrar = SimpleNamespace(self_rank_info=SimpleNamespace())
    sender_info = SimpleNamespace(sender_endpoints=["tcp://legacy-sender"])

    monkeypatch.setattr(transfer_module, "ZMQMessenger", _LegacyInfoMessenger)
    monkeypatch.setattr(
        transfer_module,
        "RankInfo",
        SimpleNamespace(from_bytes=lambda _data: sender_info),
        raising=False,
    )
    monkeypatch.setenv("TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_TERMINAL_CONSENSUS", "1")

    with pytest.raises(RuntimeError, match="requires native-transfer protocol"):
        receiver._get_sender_info(SimpleNamespace(ctx_info_endpoint="tcp://legacy-info"))


def test_control_plane_owns_dealers_on_one_thread_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_on: list[int] = []
    sent_on: list[int] = []
    delivered: list[bytes] = []
    attempts: dict[bytes, int] = {}

    class _FakeMessenger:
        def __init__(self, *, mode: str, endpoint: str):
            assert mode == "DEALER"
            assert endpoint == "tcp://peer"
            created_on.append(threading.get_ident())

        def send(self, message: list[bytes]) -> None:
            sent_on.append(threading.get_ident())
            marker = message[0]
            attempts[marker] = attempts.get(marker, 0) + 1
            if marker == b"retry" and attempts[marker] == 1:
                raise RuntimeError("injected transient send failure")
            delivered.append(marker)

        def stop(self) -> None:
            assert threading.get_ident() in created_on

    monkeypatch.setattr(transfer_module, "ZMQMessenger", _FakeMessenger)
    control = _ControlPlane("test")
    callback_count = 0
    callback_lock = threading.Lock()

    def on_sent() -> None:
        nonlocal callback_count
        with callback_lock:
            callback_count += 1

    callers = [
        threading.Thread(
            target=control.send,
            args=("tcp://peer", [marker]),
            kwargs={"retry": marker == b"retry", "on_sent": on_sent},
        )
        for marker in (b"retry", b"a", b"b", b"c")
    ]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=1)

    assert all(not caller.is_alive() for caller in callers)
    control.flush()
    owner_ident = control.owner_ident
    control.shutdown()

    assert attempts[b"retry"] == 2
    assert sorted(delivered) == [b"a", b"b", b"c", b"retry"]
    assert callback_count == 4
    assert owner_ident is not None
    assert set(created_on) == {owner_ident}
    assert set(sent_on) == {owner_ident}


def test_control_plane_retransmits_cancel_until_ack_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    acknowledged = threading.Event()

    class _FakeMessenger:
        def __init__(self, *, mode: str, endpoint: str):
            assert mode == "DEALER"
            assert endpoint == "tcp://peer"

        def send(self, _message: list[bytes]) -> None:
            nonlocal attempts
            attempts += 1

        def stop(self) -> None:
            pass

    monkeypatch.setattr(transfer_module, "ZMQMessenger", _FakeMessenger)
    control = _ControlPlane("cancel-retransmit")
    control.send(
        "tcp://peer",
        [MessageType.CANCEL_SESSION],
        retry=True,
        wait=False,
        repeat_until=acknowledged.is_set,
    )
    deadline = threading.Event()
    for _ in range(100):
        if attempts >= 2:
            break
        deadline.wait(0.005)
    assert attempts >= 2

    acknowledged.set()
    control.flush()
    settled_attempts = attempts
    deadline.wait(0.02)
    control.shutdown()

    assert attempts == settled_attempts


def test_rx_partial_fanout_waits_for_all_negotiated_drain_acks() -> None:
    session, task = _make_rx_session()
    task.expected_transfers = 2
    endpoints = {"tcp://sender-0", "tcp://sender-1"}

    assert session.mark_transferring(
        0,
        {0: "tcp://sender-0", 1: "tcp://sender-1"},
        endpoints,
        request_epoch=session.request_epoch,
    )
    cancel_endpoints, cancel_operations, orphaned = session.fail_partial_dispatch(
        0,
        {0},
        RuntimeError("injected fan-out failure"),
    )

    assert cancel_endpoints == {"tcp://sender-0"}
    assert cancel_operations == {("tcp://sender-0", session.request_epoch)}
    assert orphaned == [(17, 0)]
    assert session._outstanding_operations == 1
    assert task.status == TaskStatus.ERROR

    session.process_cancel_ack("tcp://sender-0", session.request_epoch)
    assert session._outstanding_operations == 0
    assert session._receiver._bounce.drained == []


def test_rx_partial_later_slice_does_not_retire_first_slice_aux_credits() -> None:
    session, first_task = _make_rx_session()
    session._need_aux = True
    first_task.expected_transfers = 2
    endpoints = {"tcp://sender-0", "tcp://sender-1"}
    assert session.mark_transferring(0, endpoints, endpoints)

    second_task = _Task(TaskStatus.INIT, expected_transfers=2, slice_id=1)
    session._kv_tasks.append(second_task)
    assert session.mark_transferring(1, endpoints, endpoints)
    assert session._outstanding_operations == 6

    session.fail_partial_dispatch(
        1,
        {"tcp://sender-0"},
        RuntimeError("injected second-slice fan-out failure"),
    )

    # Only one unsent KV writer from slice 1 is retired. Slice 0 owns both
    # auxiliary credits and they remain live until result/ACK proof.
    assert session._outstanding_operations == 5


def test_rx_mixed_version_fanout_keeps_legacy_result_authoritative() -> None:
    session, first_task = _make_rx_session()
    first_task.expected_transfers = 1
    assert session.mark_transferring(0, {"tcp://new-sender"}, {"tcp://new-sender"})

    second_task = _Task(TaskStatus.INIT, expected_transfers=1, slice_id=1)
    session._kv_tasks.append(second_task)
    assert session.mark_transferring(1, {"tcp://legacy-sender"}, set())

    session.cancel()

    assert session._ack_capable_endpoints == set()
    assert session._cancel_pending_endpoints == set()
    assert session._receiver.cancelled == [
        (
            17,
            {"tcp://new-sender", "tcp://legacy-sender"},
            set(),
        )
    ]
    assert session._outstanding_operations == 2


def test_rx_legacy_adp_cohort_is_bound_once_for_the_whole_request() -> None:
    session, first_task = _make_rx_session()
    cohorts = (frozenset({0, 1}), frozenset({2, 3}))
    first_task.expected_transfers = 2
    assert session.mark_transferring(
        0,
        {0: "tcp://sender-0", 1: "tcp://sender-1", 2: "tcp://sender-2", 3: "tcp://sender-3"},
        set(),
        allowed_rank_cohorts=cohorts,
    )
    with session.lock:
        session._bind_legacy_cohort_unlocked(0, 0)

    second_task = _Task(TaskStatus.INIT, expected_transfers=2, slice_id=1)
    session._kv_tasks.append(second_task)
    assert session.mark_transferring(
        1,
        {0: "tcp://sender-0", 1: "tcp://sender-1", 2: "tcp://sender-2", 3: "tcp://sender-3"},
        set(),
        allowed_rank_cohorts=cohorts,
    )
    with session.lock, pytest.raises(RuntimeError, match="different ADP writer cohort"):
        session._bind_legacy_cohort_unlocked(1, 2)


def test_rx_legacy_adp_aux_first_binds_the_request_cohort() -> None:
    session, task = _make_rx_session()
    session._need_aux = True
    cohorts = (frozenset({0, 1}), frozenset({2, 3}))
    task.expected_transfers = 2
    assert session.mark_transferring(
        0,
        {0: "tcp://sender-0", 1: "tcp://sender-1", 2: "tcp://sender-2", 3: "tcp://sender-3"},
        set(),
        allowed_rank_cohorts=cohorts,
    )

    with session.lock:
        assert (
            session._validate_receive_operation_unlocked(
                ("aux", 2), sender_endpoint=None, request_epoch=None
            )
            == "accept"
        )
        with pytest.raises(RuntimeError, match="outside the bound ADP writer cohort"):
            session._validate_receive_operation_unlocked(
                ("kv", 0, 0), sender_endpoint=None, request_epoch=None
            )
