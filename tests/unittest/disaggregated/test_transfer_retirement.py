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
"""Deadline-bounded retirement regressions for Python KV transfers."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.native.bounce as bounce_mod
import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
import tensorrt_llm._torch.disaggregation.transceiver as transceiver_mod
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    MessageType,
    PeerIncompatibleError,
    Receiver,
    RecvReqInfo,
    RxSession,
    Sender,
    SendTaskBase,
    TransferWorker,
    TxSession,
    WriteMeta,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle


class _Status:
    def __init__(self, complete_after_checks: int | None = None) -> None:
        self._done = False
        self._complete_after_checks = complete_after_checks
        self.check_count = 0

    def is_completed(self) -> bool:
        self.check_count += 1
        if (
            self._complete_after_checks is not None
            and self.check_count >= self._complete_after_checks
        ):
            self._done = True
        return self._done

    def wait(self, timeout_ms=None) -> bool:
        raise AssertionError("retirement must use the quiet status probe")


class _Session:
    def __init__(self, deadline_s: float, grace_s: float = 1.0) -> None:
        self._deadline_monotonic_s = deadline_s
        self._overall_timeout_s = grace_s
        self.status = SessionStatus.TRANSFERRING
        self.lock = threading.Lock()
        self._failed = False

    def has_failed(self) -> bool:
        return self._failed

    def set_exception(self, _reason: str) -> None:
        self._failed = True
        self.status = SessionStatus.ERROR


def _make_sender() -> Sender:
    sender = object.__new__(Sender)
    sender._shutdown = False
    sender._physical_tasks = set()
    sender._physical_tasks_lock = threading.Lock()
    sender._admission_gate = transfer_mod._TransferAdmissionGate()
    return sender


def _make_task(rid: int = 7) -> SendTaskBase:
    return SendTaskBase(SimpleNamespace(disagg_request_id=rid))


def _make_meta(task: SendTaskBase, op_id: int) -> WriteMeta:
    empty = np.array([], dtype=np.int64)
    return WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="peer",
        peer_rank=3,
        peer_endpoint="tcp://peer",
        unique_rid=7,
        src_ptrs=empty,
        dst_ptrs=empty,
        sizes=empty,
        slice_id=0,
        physical_op_id=op_id,
    )


@pytest.mark.cpu_only
def test_local_batch_is_fully_published_before_first_enqueue() -> None:
    sender = _make_sender()
    first_task, second_task = _make_task(7), _make_task(8)
    first_meta = _make_meta(first_task, 0)
    second_meta = _make_meta(second_task, 0)
    observed = []

    def enqueue(_meta) -> None:
        observed.append((first_task.resources_drained, second_task.resources_drained))

    sender._enqueue = enqueue
    sender._publish_and_enqueue([first_meta, second_meta])

    assert observed[0] == (False, False)


@pytest.mark.cpu_only
def test_source_retires_only_after_every_local_operation_finishes() -> None:
    task = _make_task()
    first = task.begin_physical_access()
    second = task.begin_physical_access()

    task.finish_physical_access(first)
    assert not task.resources_drained

    task.finish_physical_access(second)
    assert task.resources_drained


@pytest.mark.cpu_only
def test_request_scoped_retirement_blocks_only_the_same_request() -> None:
    gate = transfer_mod._TransferAdmissionGate()
    token = ("rx", 7)
    gate.suspend(7, token)

    with pytest.raises(transfer_mod._TransferAdmissionClosed):
        gate.publish_if_open(lambda: None, scope=7)
    assert gate.run_if_open(lambda: "admitted", scope=8) == "admitted"

    gate.resume(7, token)
    assert gate.publish_if_open(lambda: "reopened", scope=7) == "reopened"


@pytest.mark.cpu_only
def test_sender_membership_cannot_lose_concurrent_late_publication() -> None:
    checked = threading.Event()
    proceed = threading.Event()

    class _Task(SendTaskBase):
        finish_thread_id = None
        publish_thread_id = None

        @property
        def resources_drained(self) -> bool:
            drained = super().resources_drained
            if threading.get_ident() == self.finish_thread_id and not checked.is_set():
                checked.set()
                assert proceed.wait(1)
            return drained

        def begin_physical_access(self) -> int:
            op_id = super().begin_physical_access()
            if threading.get_ident() == self.publish_thread_id:
                proceed.set()
            return op_id

    class _Lock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.publish_thread_id = None

        def __enter__(self):
            if threading.get_ident() == self.publish_thread_id and self._lock.locked():
                proceed.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_args) -> None:
            self._lock.release()

    sender = _make_sender()
    sender._physical_tasks_lock = _Lock()
    task = _Task(SimpleNamespace(disagg_request_id=7))
    first_op = sender._publish_physical_access(task)

    def finish() -> None:
        task.finish_thread_id = threading.get_ident()
        sender._finish_physical_access(task, first_op)

    def publish() -> None:
        task.publish_thread_id = threading.get_ident()
        sender._physical_tasks_lock.publish_thread_id = threading.get_ident()
        sender._publish_physical_access(task)

    finish_thread = threading.Thread(target=finish)
    finish_thread.start()
    assert checked.wait(1)
    publish_thread = threading.Thread(target=publish)
    publish_thread.start()
    finish_thread.join(1)
    publish_thread.join(1)

    assert not finish_thread.is_alive()
    assert not publish_thread.is_alive()
    assert task in sender._physical_tasks
    assert not task.resources_drained


@pytest.mark.cpu_only
def test_unproven_backend_access_fails_closed_without_reuse(monkeypatch) -> None:
    sender = _make_sender()
    task = _make_task()
    op_id = sender._publish_physical_access(task)
    monkeypatch.setattr(transfer_mod.time, "monotonic", lambda: 2.0)

    result = sender._wait_for_backend_done(
        _Session(deadline_s=0.0), task, _Status(), _make_meta(task, op_id)
    )

    assert result is None
    assert not task.resources_drained
    assert sender.has_unresolved_accessors
    error = sender.retirement_error
    assert error is not None
    assert "rid=7" in error
    assert "peer_rank=3" in error


@pytest.mark.cpu_only
def test_done_during_grace_retires_memory_but_preserves_logical_failure(
    monkeypatch,
) -> None:
    clock = SimpleNamespace(now=0.0)
    sender = _make_sender()
    task = _make_task()
    op_id = sender._publish_physical_access(task)
    status = _Status(complete_after_checks=3)
    monkeypatch.setattr(transfer_mod.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(
        transfer_mod.time,
        "sleep",
        lambda delay: setattr(clock, "now", clock.now + max(delay, 0.5)),
    )

    result = sender._wait_for_backend_done(
        _Session(deadline_s=0.5, grace_s=2.0),
        task,
        status,
        _make_meta(task, op_id),
    )

    assert result is AgentResult.FAILED
    assert task.resources_drained
    assert sender.retirement_error is None


@pytest.mark.cpu_only
def test_backend_error_starts_grace_before_request_deadline(monkeypatch) -> None:
    clock = SimpleNamespace(now=0.0)
    sender = _make_sender()
    task = _make_task()
    op_id = sender._publish_physical_access(task)
    status = SimpleNamespace(
        is_completed=Mock(return_value=False),
        is_failed=Mock(return_value=True),
        last_status_str=Mock(return_value="NIXL_ERR_REMOTE_DISCONNECTED"),
    )
    monkeypatch.setattr(transfer_mod.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(
        transfer_mod.time,
        "sleep",
        lambda delay: setattr(clock, "now", clock.now + max(delay, 0.6)),
    )

    result = sender._wait_for_backend_done(
        _Session(deadline_s=100.0, grace_s=1.0),
        task,
        status,
        _make_meta(task, op_id),
    )

    assert result is None
    assert clock.now < 100.0
    assert not task.resources_drained
    retirement_error = sender.retirement_error
    assert retirement_error is not None
    assert "backend reported transfer error" in retirement_error


@pytest.mark.cpu_only
def test_successful_backend_poll_keeps_sub_10ms_probe_interval(monkeypatch) -> None:
    clock = SimpleNamespace(now=0.0)
    sleeps = []
    sender = _make_sender()
    task = _make_task()
    op_id = sender._publish_physical_access(task)
    monkeypatch.setattr(transfer_mod.time, "monotonic", lambda: clock.now)

    def advance(delay: float) -> None:
        sleeps.append(delay)
        clock.now += delay

    monkeypatch.setattr(transfer_mod.time, "sleep", advance)
    result = sender._wait_for_backend_done(
        _Session(deadline_s=1.0),
        task,
        _Status(complete_after_checks=3),
        _make_meta(task, op_id),
    )

    assert result is AgentResult.SUCCESS
    assert sleeps
    assert max(sleeps) <= 0.01


@pytest.mark.cpu_only
def test_ambiguous_submission_retains_source_and_poison_gate(monkeypatch) -> None:
    clock = SimpleNamespace(now=0.0)
    sender = _make_sender()
    sender._agent = SimpleNamespace(
        submit_transfer_requests=Mock(side_effect=RuntimeError("submit lost"))
    )
    task = _make_task()
    op_id = sender._publish_physical_access(task)
    meta = _make_meta(task, op_id)
    monkeypatch.setattr(transfer_mod.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(
        transfer_mod.time,
        "sleep",
        lambda delay: setattr(clock, "now", clock.now + max(delay, 0.5)),
    )

    result = sender._submit_and_wait(_Session(deadline_s=1.0, grace_s=1.0), task, object(), meta)

    assert result is None
    assert not task.resources_drained
    assert sender.retirement_error is not None


@pytest.mark.cpu_only
def test_known_admission_rejection_returns_safe_failed_evidence() -> None:
    sender = _make_sender()
    sender._agent = SimpleNamespace(submit_transfer_requests=Mock())
    task = _make_task()
    op_id = sender._publish_physical_access(task)
    sender._admission_gate.suspend(7, "older-attempt")

    result = sender._submit_and_wait(
        _Session(deadline_s=1.0), task, object(), _make_meta(task, op_id)
    )

    assert result is AgentResult.FAILED
    assert task.resources_drained
    sender._agent.submit_transfer_requests.assert_not_called()


@pytest.mark.cpu_only
def test_poison_does_not_wait_for_blocked_backend_submission() -> None:
    entered = threading.Event()
    release = threading.Event()
    poison_done = threading.Event()
    status = _Status()
    status._done = True

    def submit(_request):
        entered.set()
        assert release.wait(1)
        return status

    sender = _make_sender()
    sender._agent = SimpleNamespace(submit_transfer_requests=submit)
    task = _make_task()
    op_id = sender._publish_physical_access(task)
    result = []
    transfer_thread = threading.Thread(
        target=lambda: result.append(
            sender._submit_and_wait(
                _Session(deadline_s=transfer_mod.time.monotonic() + 10),
                task,
                object(),
                _make_meta(task, op_id),
            )
        )
    )
    transfer_thread.start()
    assert entered.wait(1)

    poison_thread = threading.Thread(
        target=lambda: (sender._poison("retirement expired"), poison_done.set())
    )
    poison_thread.start()
    try:
        assert poison_done.wait(1)
    finally:
        release.set()
    poison_thread.join(1)
    transfer_thread.join(1)

    assert result == [AgentResult.SUCCESS]


@pytest.mark.cpu_only
def test_transceiver_retirement_clock_is_sticky(monkeypatch) -> None:
    clock = SimpleNamespace(now=10.0)
    session = SimpleNamespace(drained=False)
    session.resources_drained = lambda: session.drained
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver.kv_transfer_timeout_ms = 1_000
    transceiver._retirement_deadlines = {}
    transceiver._retirement_reasons = {}
    transceiver._retirement_error = None
    transceiver._send_sessions = {7: session}
    transceiver._recv_sessions = {}
    transceiver._transfer_worker = SimpleNamespace(retirement_error=None)
    monkeypatch.setattr(transceiver_mod.time, "monotonic", lambda: clock.now)

    transceiver._start_retirement("tx", 7, "first observation")
    clock.now = 10.5
    transceiver._start_retirement("tx", 7, "retry")
    assert transceiver._retirement_deadlines[("tx", 7)] == 11.0
    assert transceiver.get_transfer_retirement_error() is None

    clock.now = 11.0
    error = transceiver.get_transfer_retirement_error()
    assert error is not None
    assert "role=tx rid=7" in error


@pytest.mark.cpu_only
def test_transceiver_adopts_listener_side_retirement_anchor(monkeypatch) -> None:
    session = SimpleNamespace(
        _retirement_deadline_s=10.25,
        _retirement_reason="remote cancellation",
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver.kv_transfer_timeout_ms = 1_000
    transceiver._retirement_deadlines = {}
    transceiver._retirement_reasons = {}
    transceiver._retirement_waits_for_peer = set()
    transceiver._recv_sessions = {7: session}
    transceiver._send_sessions = {}
    transceiver._transfer_worker = SimpleNamespace(suspend_admission=Mock())
    monkeypatch.setattr(transceiver_mod.time, "monotonic", lambda: 20.0)

    transceiver._start_retirement("rx", 7, "late poll")

    assert transceiver._retirement_deadlines[("rx", 7)] == 10.25
    assert transceiver._retirement_reasons[("rx", 7)] == "remote cancellation"


@pytest.mark.cpu_only
def test_rx_cancel_anchors_grace_at_event_and_suspends_same_rid(monkeypatch) -> None:
    clock = SimpleNamespace(now=3.0)
    receiver = SimpleNamespace(
        setup_session=Mock(),
        suspend_session_admission=Mock(),
        resume_session_admission=Mock(),
        _enforce_physical_ownership=True,
        _bounce=SimpleNamespace(retain_orphaned_reservation=Mock()),
    )
    params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        disagg_request_id=7,
        ctx_request_id=None,
    )
    session = RxSession(7, params, receiver, timeout_s=2.0)
    session._kv_tasks = [
        SimpleNamespace(
            slice_id=0,
            status=transfer_mod.TaskStatus.TRANSFERRING,
            resources_drained=False,
        )
    ]
    monkeypatch.setattr(transfer_mod.time, "monotonic", lambda: clock.now)

    assert session.cancel_local()

    assert session._retirement_deadline_s == 5.0
    receiver.suspend_session_admission.assert_called_once_with(7, session._retirement_token)


@pytest.mark.cpu_only
def test_mixed_rank_retirement_flags_fail_before_worker_setup() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver._mapping = SimpleNamespace(world_size=2)
    transceiver._dist = SimpleNamespace(allgather=Mock(return_value=[1, 0]))

    with pytest.raises(ValueError, match="same value"):
        transceiver._validate_retirement_flag_consensus()


@pytest.mark.cpu_only
def test_owned_cancel_waits_for_rank_aligned_status_cleanup(monkeypatch) -> None:
    session = SimpleNamespace(
        _enforce_physical_ownership=True,
        cancel=Mock(),
        has_transferring_tasks=Mock(return_value=False),
        close=Mock(),
    )
    request = object()
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {7: session}
    transceiver._send_reqs = {7: request}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._clear_retirement = Mock()
    monkeypatch.setattr(transceiver_mod, "get_unique_rid", lambda _req: 7)

    assert not transceiver.cancel_request(request)

    session.close.assert_not_called()
    assert transceiver._send_sessions[7] is session
    assert transceiver._send_reqs[7] is request


@pytest.mark.cpu_only
def test_refused_failed_session_close_is_not_reported_terminal() -> None:
    request = SimpleNamespace(state=None)
    session = SimpleNamespace(
        _enforce_physical_ownership=True,
        resources_drained=Mock(return_value=True),
        close=Mock(return_value=False),
    )
    sessions = {7: session}
    requests = {7: request}
    failed = [7]
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver._start_retirement = Mock()
    transceiver._clear_retirement = Mock()

    transceiver._close_failed_sessions(sessions, requests, failed, role="rx")

    assert failed == []
    assert sessions == {7: session}
    assert requests == {7: request}
    assert request.state is None
    transceiver._start_retirement.assert_called_once_with(
        "rx", 7, "session close refused before drain", wait_for_peer=True
    )
    transceiver._clear_retirement.assert_not_called()


@pytest.mark.cpu_only
def test_synchronous_receive_retains_unquiesced_destination_and_fails_closed(
    monkeypatch,
) -> None:
    clock = SimpleNamespace(now=10.0)
    req = SimpleNamespace(
        state=None,
        py_disaggregated_params=SimpleNamespace(schedule_style=DisaggScheduleStyle.CONTEXT_FIRST),
    )
    session = SimpleNamespace(
        receive=Mock(),
        wait_complete=Mock(return_value=transceiver_mod.WaitResult.FAILED),
        close=Mock(return_value=False),
        resources_drained=Mock(return_value=False),
    )
    worker = SimpleNamespace(
        create_rx_session=Mock(return_value=session),
        retirement_error=None,
        poison=Mock(),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver.kv_transfer_timeout_ms = 1_000
    transceiver._retirement_deadlines = {}
    transceiver._retirement_reasons = {}
    transceiver._retirement_error = None
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = worker
    transceiver._create_kv_slice = Mock(return_value=object())
    monkeypatch.setattr(transceiver_mod, "get_unique_rid", lambda _req: 7)
    monkeypatch.setattr(transceiver_mod.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(
        transceiver_mod.time,
        "sleep",
        lambda delay: setattr(clock, "now", clock.now + max(delay, 0.6)),
    )

    with pytest.raises(RuntimeError, match="role=rx rid=7"):
        transceiver.request_and_receive_sync(req)

    assert transceiver._recv_sessions[7] is session
    assert transceiver._recv_reqs[7] is req
    worker.poison.assert_called_once()


@pytest.mark.cpu_only
def test_internal_generation_first_receive_does_not_partially_enable_ownership() -> None:
    receiver = SimpleNamespace(
        _enforce_physical_ownership=True,
        setup_session=Mock(),
    )
    params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        disagg_request_id=7,
        ctx_request_id=None,
    )

    session = RxSession(7, params, receiver)

    assert not session._enforce_physical_ownership


@pytest.mark.cpu_only
def test_internal_generation_first_send_does_not_partially_enable_ownership() -> None:
    sender = SimpleNamespace(
        _enforce_physical_ownership=True,
        setup_session=Mock(),
    )
    params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        disagg_request_id=7,
        ctx_request_id=None,
    )

    session = TxSession(7, params, sender)

    assert not session._enforce_physical_ownership


@pytest.mark.cpu_only
def test_enabled_profile_rejects_generation_first_requests() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    req = SimpleNamespace(
        py_disaggregated_params=SimpleNamespace(schedule_style=DisaggScheduleStyle.GENERATION_FIRST)
    )

    with pytest.raises(ValueError, match="supports only context-first"):
        transceiver._require_supported_retirement_profile(req)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("enable_attention_dp", "pp_size", "cp_size"),
    [(True, 1, 1), (False, 2, 1), (False, 1, 2)],
)
def test_enabled_profile_rejects_unqualified_topologies(
    enable_attention_dp: bool, pp_size: int, cp_size: int
) -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver._mapping = SimpleNamespace(
        cp_size=cp_size,
        has_cp_helix=Mock(return_value=cp_size > 1),
        enable_attention_dp=enable_attention_dp,
        pp_size=pp_size,
    )

    with pytest.raises(ValueError, match="initially qualified"):
        transceiver._check_compatible()


@pytest.mark.cpu_only
def test_late_generation_first_request_uses_legacy_send_path(monkeypatch) -> None:
    sender = _make_sender()
    sender._shutdown = False
    task = _make_task()
    session = SimpleNamespace(
        lock=threading.Lock(),
        kv_tasks=[task],
        status=SessionStatus.INIT,
        has_failed=Mock(return_value=False),
        _closed=False,
        _enforce_physical_ownership=False,
    )
    sender._sessions_lock = threading.Lock()
    sender._get_session = Mock(return_value=session)
    sender._save_peer_req_info = Mock()
    sender._build_kv_write_meta = Mock(return_value=_make_meta(task, 0))
    sender._publish_and_enqueue = Mock()
    sender._enqueue = Mock()
    info = SimpleNamespace(unique_rid=7, instance_rank=0)
    monkeypatch.setattr(RecvReqInfo, "from_bytes", lambda _payload: info)

    sender._respond_with_kv(b"", [b"REQUEST_DATA", b"payload"])

    sender._publish_and_enqueue.assert_not_called()
    sender._enqueue.assert_called_once()


@pytest.mark.cpu_only
def test_receiver_rejects_mixed_retirement_protocol_before_publication(
    monkeypatch,
) -> None:
    peer = RankInfo(
        instance_name="ctx",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[1],
        sender_endpoints=["tcp://ctx"],
        self_endpoint="tcp://ctx",
        transfer_engine_info=b"",
        transfer_retirement_protocol=0,
    )

    class _Messenger:
        def __init__(self, **_kwargs) -> None:
            pass

        def send(self, _message) -> None:
            pass

        def receive(self):
            return [peer.to_bytes()]

        def stop(self) -> None:
            pass

    receiver = object.__new__(Receiver)
    receiver._incompatible_peers = {}
    receiver._sender_ep_instance_map = {}
    receiver._dealers = {}
    receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(transfer_retirement_protocol=1),
        self_extractor=SimpleNamespace(page_table=None),
    )
    params = SimpleNamespace(ctx_info_endpoint="tcp://info")
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", _Messenger)

    with pytest.raises(PeerIncompatibleError, match="protocol mismatch"):
        receiver._get_sender_info(params)

    assert "tcp://info" in receiver._incompatible_peers
    assert receiver._sender_ep_instance_map == {}


@pytest.mark.cpu_only
def test_failed_rx_with_missing_writer_starts_retirement_deadline(
    monkeypatch,
) -> None:
    session = SimpleNamespace(
        _enforce_physical_ownership=True,
        is_completed=Mock(return_value=False),
        has_failed=Mock(return_value=True),
        resources_drained=Mock(return_value=False),
        wait_complete=Mock(),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver.kv_transfer_timeout_ms = 1_000
    transceiver._retirement_deadlines = {}
    transceiver._retirement_reasons = {}
    transceiver._retirement_error = None
    transceiver._recv_sessions = {7: session}
    transceiver._recv_reqs = {7: object()}
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._mapping = SimpleNamespace(
        pp_size=1,
        enable_attention_dp=False,
        world_size=1,
    )
    transceiver._gen_allgather = Mock()
    monkeypatch.setattr(transceiver_mod.time, "monotonic", lambda: 10.0)

    assert transceiver.check_gen_transfer_status(0) == ([], [], [])

    assert transceiver._retirement_deadlines[("rx", 7)] == 11.0
    session.wait_complete.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("completed", [False, True], ids=["in-progress", "completed"])
def test_generation_timeout_wins_before_progress_or_completion(
    completed: bool,
) -> None:
    request = SimpleNamespace(py_kv_transfer_timed_out=True)
    session = SimpleNamespace(
        _enforce_physical_ownership=True,
        cancel=Mock(),
        is_completed=Mock(return_value=completed),
        has_failed=Mock(return_value=False),
        resources_drained=Mock(return_value=True),
        wait_complete=Mock(),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver._recv_sessions = {7: session}
    transceiver._recv_reqs = {7: request}
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = False
    transceiver._mapping = SimpleNamespace(
        pp_size=1,
        enable_attention_dp=False,
        world_size=1,
    )
    transceiver._gen_allgather = Mock()
    transceiver._gen_consensus_outcome = Mock(return_value=([7], [], []))
    transceiver._withhold_unquiesced_terminal_outcomes = Mock(return_value=([], []))

    assert transceiver.check_gen_transfer_status(0) == ([], [], [])

    session.cancel.assert_called_once_with()
    session.wait_complete.assert_not_called()


@pytest.mark.cpu_only
def test_invalid_writer_evidence_anchors_listener_retirement(monkeypatch) -> None:
    clock = SimpleNamespace(now=4.0)
    task = SimpleNamespace(resources_drained=False, fail=Mock())
    receiver = SimpleNamespace(
        setup_session=Mock(),
        suspend_session_admission=Mock(),
        resume_session_admission=Mock(),
        _enforce_physical_ownership=True,
    )
    params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        disagg_request_id=7,
        ctx_request_id=None,
    )
    session = RxSession(7, params, receiver, timeout_s=2.0)
    session._kv_tasks = [task]
    monkeypatch.setattr(transfer_mod.time, "monotonic", lambda: clock.now)
    error = RuntimeError("writer 9 is outside the sealed cohort")

    session.fail_result_processing(error)

    task.fail.assert_called_once_with(error)
    assert session.status == SessionStatus.ERROR
    assert session._retirement_deadline_s == 6.0
    receiver.suspend_session_admission.assert_called_once_with(7, session._retirement_token)


@pytest.mark.cpu_only
def test_malformed_result_tail_routes_to_session_fail_close(monkeypatch) -> None:
    session = SimpleNamespace(
        process_kv_agent_result=Mock(),
        fail_result_processing=Mock(),
    )
    receiver = object.__new__(Receiver)
    receiver._get_session = Mock(return_value=session)
    prefix = transfer_mod._KV_RESULT_PREFIX.pack(
        3,
        7,
        0,
        True,
        transfer_mod._AGENT_RESULT_CODE[AgentResult.SUCCESS],
        0,
    )
    error = ValueError("malformed writer result tail")
    monkeypatch.setattr(bounce_mod, "decode_result_tail", Mock(side_effect=error))

    with pytest.raises(ValueError, match="malformed writer result tail"):
        receiver._process_kv_agent_result(
            b"sender",
            [MessageType.KV_AGENT_RESULT, prefix],
        )

    session.process_kv_agent_result.assert_not_called()
    session.fail_result_processing.assert_called_once_with(error)


@pytest.mark.cpu_only
def test_context_timeout_starts_sticky_retirement() -> None:
    session = SimpleNamespace(
        disagg_request_id=7,
        status=SessionStatus.TRANSFERRING,
        _enforce_physical_ownership=True,
        resources_drained=Mock(return_value=False),
        wait_complete=Mock(return_value=WaitResult.TIMEOUT),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver.kv_transfer_timeout_ms = 1_000
    transceiver._send_sessions = {7: session}
    transceiver._send_reqs = {7: object()}
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._sender_future_timeout_ms = 1_000
    transceiver._collect_done = Mock(return_value=([7], []))
    transceiver._build_to_process = Mock(return_value=[7])
    transceiver._ctx_consensus_outcome = Mock(return_value=([], [], []))
    transceiver._withhold_unquiesced_terminal_outcomes = Mock(return_value=([], []))
    transceiver._start_retirement = Mock()
    transceiver._transfer_worker = SimpleNamespace(sweep_stale_req_infos=Mock())

    assert transceiver.check_context_transfer_status(0) == ([], [])

    transceiver._start_retirement.assert_called_once_with(
        "tx", 7, "request deadline expired before quiescence"
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("completed", [False, True], ids=["in-progress", "completed"])
def test_context_timeout_wins_before_progress_or_completion(
    completed: bool,
) -> None:
    request = SimpleNamespace(py_kv_transfer_timed_out=True)
    session = SimpleNamespace(
        status=SessionStatus.TRANSFERRING,
        _enforce_physical_ownership=True,
        cancel=Mock(),
        is_completed=Mock(return_value=completed),
        has_failed=Mock(return_value=False),
        resources_drained=Mock(return_value=True),
        wait_complete=Mock(),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver._send_sessions = {7: session}
    transceiver._send_reqs = {7: request}
    transceiver._ever_had_send_session = True
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._sender_future_timeout_ms = 1_000
    transceiver._ctx_consensus_outcome = Mock(return_value=([7], [], []))
    transceiver._withhold_unquiesced_terminal_outcomes = Mock(return_value=([], []))
    transceiver._transfer_worker = SimpleNamespace(sweep_stale_req_infos=Mock())

    assert transceiver.check_context_transfer_status(0) == ([], [])

    session.cancel.assert_called_once_with()
    session.wait_complete.assert_not_called()


@pytest.mark.cpu_only
def test_global_failure_is_withheld_until_every_rank_is_drained() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver._start_retirement = Mock()
    session = SimpleNamespace(resources_drained=Mock(return_value=True))

    cancelled, failed = transceiver._withhold_unquiesced_terminal_outcomes(
        "tx",
        {7: session},
        [],
        [7],
        consensus=lambda _local: [],
    )

    assert cancelled == []
    assert failed == []
    transceiver._start_retirement.assert_called_once_with(
        "tx",
        7,
        "terminal outcome is waiting for peer-rank quiescence",
        wait_for_peer=True,
    )


@pytest.mark.cpu_only
def test_poisoned_worker_shutdown_retains_backend_resources() -> None:
    worker = object.__new__(TransferWorker)
    worker._config = SimpleNamespace(enforce_physical_ownership=True)
    sender = SimpleNamespace(has_unresolved_accessors=False, _poison=Mock())
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {
        7: SimpleNamespace(
            _enforce_physical_ownership=True,
            resources_drained=lambda: False,
        )
    }
    receiver.shutdown = Mock()
    worker._sender = sender
    worker._receiver = receiver
    worker._bounce = SimpleNamespace(close=Mock())
    worker._agent = SimpleNamespace(deregister_memory=Mock(), shutdown=Mock())
    worker._registered_mem = [object()]
    retained_before = len(transfer_mod._POISONED_TRANSFER_WORKERS)

    worker.shutdown()

    sender._poison.assert_called_once()
    receiver.shutdown.assert_not_called()
    worker._bounce.close.assert_not_called()
    worker._agent.deregister_memory.assert_not_called()
    worker._agent.shutdown.assert_not_called()
    assert transfer_mod._POISONED_TRANSFER_WORKERS[retained_before] is worker
    transfer_mod._POISONED_TRANSFER_WORKERS.pop()


@pytest.mark.cpu_only
def test_worker_shutdown_retains_orphaned_bounce_reservation() -> None:
    worker = object.__new__(TransferWorker)
    worker._config = SimpleNamespace(enforce_physical_ownership=True)
    sender = SimpleNamespace(has_unresolved_accessors=False, _poison=Mock())
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver.shutdown = Mock()
    bounce = SimpleNamespace(has_unresolved_accessors=True, close=Mock())
    worker._sender = sender
    worker._receiver = receiver
    worker._bounce = bounce
    worker._agent = SimpleNamespace(deregister_memory=Mock(), shutdown=Mock())
    worker._registered_mem = []
    retained_before = len(transfer_mod._POISONED_TRANSFER_WORKERS)

    worker.shutdown()

    sender._poison.assert_called_once()
    bounce.close.assert_not_called()
    worker._agent.shutdown.assert_not_called()
    assert transfer_mod._POISONED_TRANSFER_WORKERS[retained_before] is worker
    transfer_mod._POISONED_TRANSFER_WORKERS.pop()


@pytest.mark.cpu_only
def test_default_off_worker_shutdown_preserves_legacy_cleanup() -> None:
    sender = SimpleNamespace(has_unresolved_accessors=True, shutdown=Mock())
    receiver = SimpleNamespace(has_unresolved_accessors=True, shutdown=Mock())
    bounce = SimpleNamespace(has_unresolved_accessors=True, close=Mock())
    agent = SimpleNamespace(shutdown=Mock())
    worker = object.__new__(TransferWorker)
    worker._shutdown = False
    worker._config = SimpleNamespace(enforce_physical_ownership=False)
    worker._admission_gate = Mock()
    worker._rank_info_server = SimpleNamespace(shutdown=Mock())
    worker._sender = sender
    worker._receiver = receiver
    worker._bounce = bounce
    worker._agent = agent
    worker._registered_mem = []

    worker.shutdown()

    sender.shutdown.assert_called_once_with()
    receiver.shutdown.assert_called_once_with()
    bounce.close.assert_called_once_with()
    agent.shutdown.assert_called_once_with()


@pytest.mark.cpu_only
def test_transceiver_shutdown_retains_unquiesced_session_and_worker() -> None:
    session = SimpleNamespace(
        resources_drained=Mock(return_value=False),
        close=Mock(),
    )
    worker = SimpleNamespace(retirement_error=None, poison=Mock(), shutdown=Mock())
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._retirement_enabled = True
    transceiver.kv_transfer_timeout_ms = 1_000
    transceiver._retirement_deadlines = {}
    transceiver._retirement_reasons = {}
    transceiver._retirement_error = None
    transceiver._send_sessions = {7: session}
    transceiver._send_reqs = {7: object()}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = worker
    retained_before = len(transceiver_mod._POISONED_TRANSCEIVERS)

    transceiver.shutdown()

    session.close.assert_not_called()
    worker.shutdown.assert_not_called()
    assert transceiver_mod._POISONED_TRANSCEIVERS[retained_before] is transceiver
    transceiver_mod._POISONED_TRANSCEIVERS.pop()
