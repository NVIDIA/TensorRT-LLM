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
"""Regressions for source-side KV transfer ownership."""

from __future__ import annotations

import gc
import threading
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.transfer import (
    KVSendTask,
    MessageType,
    Sender,
    TaskStatus,
    TxSession,
    WriteMeta,
)


def _make_send_task(rid: int) -> KVSendTask:
    return KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )


def _make_owned_session(rid: int, task: KVSendTask, sender=None) -> TxSession:
    params = DisaggregatedParams(disagg_request_id=rid)
    if sender is None:
        sender = SimpleNamespace(
            capture_receiver_endpoints=Mock(return_value=set()),
            send_cancel_to_receivers=Mock(),
            clear_session=Mock(),
        )
    session = object.__new__(TxSession)
    session._base_args = SimpleNamespace(params=params)
    session._timeout_s = 0.01
    session._overall_timeout_s = 1.0
    session._deadline_monotonic_s = None
    session._need_aux = False
    session._enforce_physical_ownership = True
    session._sender = sender
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


@pytest.mark.cpu_only
def test_send_failure_does_not_hide_sibling_source_operation() -> None:
    task = _make_send_task(301)
    task.begin_physical_operations({0, 1})
    task.status = TaskStatus.TRANSFERRING
    session = _make_owned_session(301, task)

    task.fail(RuntimeError("writer 0 failed"))
    task.finish_physical_operation(0)

    assert session.has_failed()
    assert not session.resources_drained()
    assert session.wait_complete(blocking=False) is None
    assert not session.close()

    task.finish_physical_operation(1)

    assert session.resources_drained()
    assert session.wait_complete(blocking=False) == WaitResult.FAILED
    assert session.close()


@pytest.mark.cpu_only
def test_send_task_completion_cannot_overwrite_error() -> None:
    task = _make_send_task(307)

    task.fail(RuntimeError("transfer failed"))
    task.complete()

    assert task.status == TaskStatus.ERROR
    assert task.is_done


@pytest.mark.cpu_only
def test_sender_build_failure_settles_every_unsubmitted_writer() -> None:
    rid = 302
    task = _make_send_task(rid)
    session = SimpleNamespace(
        lock=threading.Lock(),
        _closed=False,
        _has_logical_failure=Mock(return_value=False),
    )
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._build_kv_write_meta = Mock(side_effect=RuntimeError("metadata failed"))
    sender._enqueue = Mock()
    sender._send_failed_task_result_to_receiver = Mock()
    recv_infos = {
        0: SimpleNamespace(unique_rid=rid, instance_rank=10, owner_generation=1),
        1: SimpleNamespace(unique_rid=rid, instance_rank=11, owner_generation=1),
    }

    with pytest.raises(RuntimeError, match="metadata failed"):
        sender.dispatch_task(task, recv_infos)

    assert task.status == TaskStatus.ERROR
    assert task.resources_drained
    sender._enqueue.assert_not_called()
    assert {
        call.args[1].instance_rank
        for call in sender._send_failed_task_result_to_receiver.call_args_list
    } == {10, 11}


@pytest.mark.cpu_only
def test_duplicate_request_data_does_not_resubmit_owned_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 303
    peer_rank = 7
    task = _make_send_task(rid)
    session = SimpleNamespace(
        lock=threading.Lock(),
        kv_tasks=[task],
        status=SessionStatus.TRANSFERRING,
        _closed=False,
        _has_logical_failure=Mock(return_value=False),
    )
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._save_peer_req_info = Mock()
    sender._build_kv_write_meta = Mock(return_value=SimpleNamespace(peer_rank=peer_rank))
    sender._enqueue = Mock()
    info = SimpleNamespace(
        unique_rid=rid,
        instance_rank=peer_rank,
        owner_generation=1,
    )
    monkeypatch.setattr(
        transfer_mod.RecvReqInfo,
        "from_bytes",
        Mock(return_value=info),
    )

    message = [MessageType.REQUEST_DATA, b"request"]
    sender._respond_with_kv(b"peer", message)
    sender._respond_with_kv(b"peer", message)

    sender._build_kv_write_meta.assert_called_once_with(task, info)
    sender._enqueue.assert_called_once()
    assert not task.resources_drained


@pytest.mark.cpu_only
def test_cancel_and_source_admission_share_one_session_gate() -> None:
    rid = 309
    peer_rank = 9
    sender = object.__new__(Sender)
    sender._sessions_lock = threading.Lock()

    cancelled_task = _make_send_task(rid)
    cancelled_session = _make_owned_session(rid, cancelled_task, sender)
    sender._sessions = {rid: cancelled_session}

    assert cancelled_session.cancel_local()
    cancel_first = sender._begin_task_operation(cancelled_task, peer_rank)

    assert cancel_first.newly_started == frozenset()
    assert cancel_first.rejected_unsubmitted == frozenset({peer_rank})
    assert cancelled_task.resources_drained

    admitted_rid = rid + 1
    admitted_task = _make_send_task(admitted_rid)
    admitted_session = _make_owned_session(admitted_rid, admitted_task, sender)
    sender._sessions = {admitted_rid: admitted_session}

    admission_first = sender._begin_task_operation(admitted_task, peer_rank)
    assert admission_first.newly_started == frozenset({peer_rank})
    assert admitted_session.cancel_local()
    assert not admitted_task.resources_drained


def _make_delivery_sender(rid: int, session, agent, dealer) -> Sender:
    sender = object.__new__(Sender)
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._device_id = 0
    sender._instance_rank = 0
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="ctx", instance_rank=0)
    )
    sender._agent = agent
    sender._bounce = SimpleNamespace(release_send=Mock())
    sender._enforce_physical_ownership = True
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    sender._send_failed_write_meta_result = Mock()
    return sender


def _make_write_meta(rid: int, peer_rank: int, task: KVSendTask) -> WriteMeta:
    one = np.array([1], dtype=np.int64)
    return WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name=f"gen{peer_rank}",
        peer_rank=peer_rank,
        peer_endpoint=f"tcp://gen-{peer_rank}",
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
def test_backend_success_retires_source_before_result_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 304
    peer_rank = 3
    task = _make_send_task(rid)
    task.begin_physical_operations({peer_rank})
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        lock=threading.Lock(),
        status=SessionStatus.TRANSFERRING,
        kv_tasks=[task],
        _enforce_physical_ownership=True,
        set_exception=Mock(),
        transfer_end_time=None,
    )
    status = SimpleNamespace(wait=Mock(return_value=True))

    def publish_result(_message) -> None:
        assert task.resources_drained
        assert task.status == TaskStatus.TRANSFERRING

    dealer = SimpleNamespace(send=Mock(side_effect=publish_result))
    sender = _make_delivery_sender(
        rid,
        session,
        SimpleNamespace(submit_transfer_requests=Mock(return_value=status)),
        dealer,
    )
    monkeypatch.setattr(Sender, "_make_agent_request", Mock(return_value=object()))

    sender._deliver_kv_to_agent(_make_write_meta(rid, peer_rank, task))

    assert task.status == TaskStatus.TRANSFERRED
    assert task.resources_drained
    dealer.send.assert_called_once()
    session.set_exception.assert_not_called()


@pytest.mark.cpu_only
def test_submit_exception_retains_request_and_source_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BackendRequest:
        pass

    rid = 305
    peer_rank = 4
    task = _make_send_task(rid)
    task.begin_physical_operations({peer_rank})
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        lock=threading.Lock(),
        status=SessionStatus.TRANSFERRING,
        kv_tasks=[task],
        _enforce_physical_ownership=True,
        set_exception=Mock(),
    )
    sender = _make_delivery_sender(
        rid,
        session,
        SimpleNamespace(submit_transfer_requests=Mock(side_effect=RuntimeError("submit failed"))),
        SimpleNamespace(send=Mock()),
    )
    request_refs: list[weakref.ReferenceType[BackendRequest]] = []

    def make_request(_write_meta, device_id):
        del device_id
        request = BackendRequest()
        request_refs.append(weakref.ref(request))
        return request

    monkeypatch.setattr(Sender, "_make_agent_request", make_request)

    with pytest.raises(RuntimeError, match="submit failed"):
        sender._deliver_kv_to_agent(_make_write_meta(rid, peer_rank, task))

    task.fail(RuntimeError("worker observed submit failure"))
    gc.collect()
    assert request_refs[0]() is not None
    assert task.status == TaskStatus.ERROR
    assert not task.resources_drained


@pytest.mark.cpu_only
def test_backend_failure_without_quiescence_retains_source_and_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BackendRequest:
        pass

    rid = 311
    peer_rank = 6
    task = _make_send_task(rid)
    task.begin_physical_operations({peer_rank})
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        lock=threading.Lock(),
        status=SessionStatus.TRANSFERRING,
        kv_tasks=[task],
        _enforce_physical_ownership=True,
        set_exception=Mock(),
    )
    status = SimpleNamespace(
        wait=Mock(return_value=False),
        last_status_str=Mock(return_value="backend failure"),
    )
    dealer = SimpleNamespace(send=Mock())
    sender = _make_delivery_sender(
        rid,
        session,
        SimpleNamespace(name="test-agent", submit_transfer_requests=Mock(return_value=status)),
        dealer,
    )
    request_refs: list[weakref.ReferenceType[BackendRequest]] = []

    def build_bounce_request(_write_meta):
        request = BackendRequest()
        request_refs.append(weakref.ref(request))
        return request, 12

    sender._bounce = SimpleNamespace(
        build_request=Mock(side_effect=build_bounce_request),
        release_send=Mock(),
    )
    fallback = Mock(side_effect=AssertionError("bounce request should be used"))
    monkeypatch.setattr(Sender, "_make_agent_request", fallback)
    write_meta = _make_write_meta(rid, peer_rank, task)
    write_meta.bounce_dst_base = 0x1234

    with pytest.raises(RuntimeError, match="transfer failed"):
        sender._deliver_kv_to_agent(write_meta)

    gc.collect()
    assert task.status == TaskStatus.ERROR
    assert not task.resources_drained
    assert request_refs[0]() is not None
    dealer.send.assert_not_called()
    sender._send_failed_write_meta_result.assert_not_called()
    sender._bounce.build_request.assert_called_once_with(write_meta)
    sender._bounce.release_send.assert_not_called()


@pytest.mark.cpu_only
def test_cancelled_session_cannot_close_while_source_is_active() -> None:
    rid = 306
    peer_rank = 5
    task = _make_send_task(rid)
    task.begin_physical_operations({peer_rank})
    task.status = TaskStatus.TRANSFERRING
    sender = SimpleNamespace(
        capture_receiver_endpoints=Mock(return_value={"tcp://gen-5"}),
        send_cancel_to_receivers=Mock(),
        clear_session=Mock(),
    )
    session = _make_owned_session(rid, task, sender)

    session.cancel()

    assert session.status == SessionStatus.CANCELLED
    assert not session.close()
    sender.send_cancel_to_receivers.assert_called_once_with(
        rid,
        {"tcp://gen-5"},
    )
    sender.clear_session.assert_not_called()

    task.finish_physical_operation(peer_rank)

    assert session.close()
    sender.clear_session.assert_called_once_with(rid)


@pytest.mark.cpu_only
def test_legacy_cancel_does_not_require_owned_endpoint_snapshot() -> None:
    rid = 308
    task = _make_send_task(rid)
    sender = SimpleNamespace(
        capture_receiver_endpoints=Mock(side_effect=RuntimeError("ownership-only path")),
        send_cancel_to_receivers=Mock(),
        clear_session=Mock(),
    )
    session = _make_owned_session(rid, task, sender)
    session._enforce_physical_ownership = False

    session.cancel()

    assert session.status == SessionStatus.CANCELLED
    sender.capture_receiver_endpoints.assert_not_called()
    sender.send_cancel_to_receivers.assert_called_once_with(rid)
