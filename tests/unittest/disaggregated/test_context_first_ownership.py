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
"""Context-first activation and native retirement with controlled backend evidence."""

import threading
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_context_first_profile import _rank_info
from test_transfer_ownership_regressions import (
    _BounceProbe,
    _make_owned_receiver,
    _make_owned_sender,
    _OneSlotAllocator,
    _sole_piece,
)

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.lifecycle.retirement import RetirementWatchdog
from tensorrt_llm._torch.disaggregation.native.extent import (
    CacheExtent,
    Cancelled,
    Delivered,
    Failed,
)
from tensorrt_llm._torch.disaggregation.native.handle import TaskHandle
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only


def _worker(
    *, legacy_ownership: bool = False, context_first_ownership: bool = True
) -> transfer_mod.TransferWorker:
    """Use native session factories without creating CUDA agents or network threads."""
    sender = _make_owned_sender()
    receiver = _make_owned_receiver()
    receiver._context_first_incompatible_peers = {}
    for endpoint in (sender, receiver):
        endpoint._enforce_physical_ownership = legacy_ownership
        endpoint._context_first_ownership = context_first_ownership
        endpoint._registrar = SimpleNamespace(
            self_rank_info=SimpleNamespace(instance_name="worker", instance_rank=0),
            get_peer_rank_info=Mock(return_value=SimpleNamespace(self_endpoint="receiver")),
        )
    sender._peer_requests_lock = threading.Lock()
    sender._peer_requests = {}
    sender._peer_requests_timestamps = {}
    sender._device_id = 0
    sender._bounce = _BounceProbe()
    worker = object.__new__(transfer_mod.TransferWorker)
    worker._config = SimpleNamespace(
        tx_timeout_s=1,
        tx_overall_timeout_s=10,
        rx_timeout_s=10,
        enforce_physical_ownership=legacy_ownership,
        context_first_ownership=context_first_ownership,
    )
    worker._sender, worker._receiver = sender, receiver
    worker._aux_buffer = None
    worker._retirement_watchdog = RetirementWatchdog(
        lambda event: pytest.fail(f"unexpected fatal event: {event}"), clock=lambda: 1.0
    )
    return worker


def _request(
    style: DisaggScheduleStyle = DisaggScheduleStyle.CONTEXT_FIRST,
) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=401,
        prompt_len=16,
        py_disaggregated_params=DisaggregatedParams(
            disagg_request_id=401, schedule_style=style, ctx_info_endpoint="ctx-info"
        ),
    )


@pytest.mark.parametrize("active", [False, True])
def test_context_first_ownership_is_set_before_rank_publication(
    monkeypatch: pytest.MonkeyPatch, active: bool
) -> None:
    """Publish the final worker selection, not the metadata factory's default."""
    rank_info = _rank_info()
    rank_info.context_first_ownership = not active
    monkeypatch.setattr(transfer_mod, "_make_aux_buffer", Mock(return_value=None))
    monkeypatch.setattr(
        transfer_mod.RankInfo, "from_kv_cache_manager", Mock(return_value=rank_info)
    )

    def check_publication(worker: transfer_mod.TransferWorker, manager: object) -> None:
        advertised = transfer_mod.RankInfo.from_bytes(worker._rank_info.to_bytes())
        assert advertised.context_first_ownership is active

    monkeypatch.setattr(
        transfer_mod.TransferWorker, "_setup_peer_infrastructure", check_publication
    )
    monkeypatch.setattr(transfer_mod.TransferWorker, "_setup_transfer_engine", Mock())
    transfer_mod.TransferWorker(
        transfer_mod.TransferWorkerConfig(
            kv_cache_manager=Mock(),
            device_id=0,
            instance_name="worker",
            context_first_ownership=active,
        )
    )


@pytest.mark.parametrize("style", list(DisaggScheduleStyle))
@pytest.mark.parametrize(
    ("legacy_ownership", "context_first_ownership"), [(False, False), (False, True), (True, False)]
)
def test_native_ownership_selection_preserves_legacy_generation_first(
    style: DisaggScheduleStyle, legacy_ownership: bool, context_first_ownership: bool
) -> None:
    worker = _worker(
        legacy_ownership=legacy_ownership, context_first_ownership=context_first_ownership
    )
    expected = legacy_ownership or (
        context_first_ownership and style == DisaggScheduleStyle.CONTEXT_FIRST
    )
    tx = worker.create_tx_session(_request(style))
    rx = worker.create_rx_session(_request(style))
    for endpoint, session in ((worker._sender, tx), (worker._receiver, rx)):
        assert session._enforce_physical_ownership is expected
        assert (session._retirement is not None) is expected
        entry = endpoint._sessions[401]
        if expected:
            assert entry is session
        else:
            assert isinstance(entry, weakref.ReferenceType)
        if session is rx and expected and style == DisaggScheduleStyle.GENERATION_FIRST:
            # AUX is pending, but this fixture has not published any destination.
            assert not session.close()
            assert session.cancel_local()
            assert session.resources_drained()
        assert session.close()
        assert session.close()
    assert not worker._retirement_watchdog._owners


def _native_pair(
    monkeypatch: pytest.MonkeyPatch, *, completes: bool
) -> tuple[
    transfer_mod.TransferWorker,
    transfer_mod.TxSession,
    transfer_mod.RxSession,
    transfer_mod.WriteMeta,
    SimpleNamespace,
    Mock,
    Mock,
]:
    worker = _worker()
    tx = worker.create_tx_session(_request())
    rx = worker.create_rx_session(_request())
    tx.send(_sole_piece(tokens=16))
    send_task = tx.kv_tasks[0]
    recv_task = rx.prepare_receive(_sole_piece(tokens=16), expected_write_bytes=256)
    assert recv_task is not None
    recv_task.expected_transfers = 1
    assert rx.try_begin_transfer(recv_task.slice_id, {"ctx"}, writer_cohort={0})
    assert worker._sender._begin_task_operation(send_task, 0)
    status = Mock(wait=Mock(return_value=completes), is_completed=Mock(return_value=completes))
    status.last_status_str.return_value = "ERROR"
    worker._sender._agent = SimpleNamespace(
        name="nixl", submit_transfer_requests=Mock(return_value=status)
    )
    request = SimpleNamespace(op="WRITE", remote_name="gen")
    monkeypatch.setattr(transfer_mod.Sender, "_make_agent_request", Mock(return_value=request))
    dealer = Mock()
    dealer.send.side_effect = lambda message: worker._receiver._process_kv_agent_result(
        b"ctx", message
    )
    worker._sender._get_result_dealer = Mock(return_value=dealer)
    meta = transfer_mod.WriteMeta(
        task=send_task,
        expected_transfers=1,
        peer_name="gen",
        peer_rank=0,
        peer_endpoint="receiver",
        unique_rid=401,
        src_ptrs=np.array([0x1000], dtype=np.int64),
        dst_ptrs=np.array([0x2000], dtype=np.int64),
        sizes=np.array([256], dtype=np.int64),
        slice_id=0,
        receiver_slice_id=0,
        is_last_slice=True,
        meta_type=transfer_mod.WriteMetaType.KV,
    )
    return worker, tx, rx, meta, request, status, dealer


def test_context_first_completes_after_kv_without_auxiliary_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, tx, rx, meta, _, _, dealer = _native_pair(monkeypatch, completes=True)
    worker._sender._deliver_kv_to_agent(meta)

    assert not tx._need_aux and tx.aux_task is None
    assert not rx._need_aux and rx._aux_physical_owner is None
    for session, task in ((tx, tx.kv_tasks[0]), (rx, rx._kv_tasks[0])):
        assert session.is_completed()
        assert session.resources_drained()
        assert session.wait_complete(blocking=False) is WaitResult.COMPLETED
        assert isinstance(TaskHandle(session, task, token_end=16).poll(), Delivered)
        assert session.close()
        assert session.close()
    assert rx.kv_write_verified()
    dealer.send.assert_called_once()
    assert not worker._retirement_watchdog._owners


def test_context_first_cancel_before_gen_rejects_late_sender_admission() -> None:
    worker = _worker()
    sender = worker._sender
    tx = worker.create_tx_session(_request())
    tx.send(_sole_piece(tokens=16))
    task = tx.kv_tasks[0]
    handle = TaskHandle(tx, task, token_end=16)
    assert not sender._peer_requests
    assert not task._physical_operations
    assert tx.cancel_local()
    assert tx.close()
    source = _OneSlotAllocator(401)
    source.apply_reuse_decision(tx.resources_drained())
    assert source.is_reusable

    sender._build_kv_write_meta = Mock(side_effect=AssertionError("late source access"))
    sender._dispatch_task_to_peer(
        task,
        transfer_mod.RecvReqInfo(
            sender_req_id=401,
            instance_name="gen",
            instance_rank=0,
            block_ids_per_layer_groups=[],
            unique_rid=401,
            slice_id=0,
            context_first=True,
        ),
    )

    sender._build_kv_write_meta.assert_not_called()
    assert task.resources_drained
    assert isinstance(handle.poll(), Cancelled)
    assert source.release_count == 1
    _, message = sender._send_task_queues[0].get_nowait()
    result_code = transfer_mod._KV_RESULT_PREFIX.unpack(message[1])[4]
    assert transfer_mod._AGENT_RESULT_BY_CODE[result_code] is transfer_mod.AgentResult.FAILED


def test_context_first_roots_request_before_publishing_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker()
    request = _request()
    request.set_kv_cache_transfer_start = Mock()
    tx = worker.create_tx_session(request)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._fp4_mla_bridge_enabled = False
    transceiver._context_first_ownership = True
    transceiver._context_first_ownership_logged = True
    transceiver._enable_pipelined_transfer = False
    transceiver._send_reqs = {}
    transceiver._get_or_create_send_session = Mock(return_value=tx)
    transceiver._create_cache_extent = Mock(
        return_value=CacheExtent(name=401, local=_sole_piece(tokens=16))
    )
    transceiver._finalize_send = Mock()
    send = tx.send

    def send_after_rooting(chunk: transfer_mod.Chunk) -> None:
        assert transceiver._send_reqs[401] is request
        assert request.state is LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        send(chunk)

    monkeypatch.setattr(tx, "send", send_after_rooting)
    transceiver.respond_and_send_async(request)

    assert transceiver._send_reqs[401] is request
    assert tx._enforce_physical_ownership
    assert tx.kv_tasks[0]._enforce_physical_ownership
    assert tx.cancel_local()
    assert tx.close()


def test_context_first_late_done_retires_both_owners_without_rewriting_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, tx, rx, meta, request, status, dealer = _native_pair(monkeypatch, completes=False)
    sender = worker._sender
    sender._deliver_kv_to_agent(meta)
    send_task, recv_task = tx.kv_tasks[0], rx._kv_tasks[0]
    handles = [TaskHandle(tx, send_task, 16), TaskHandle(rx, recv_task, 16)]
    source, destination = _OneSlotAllocator(401), _OneSlotAllocator(401)

    for session, allocation in ((tx, source), (rx, destination)):
        assert session.status is SessionStatus.ERROR
        allocation.apply_reuse_decision(session.close())
        assert not allocation.is_reusable
    assert all(isinstance(handle.poll(), Failed) for handle in handles)
    operation = send_task._physical_operations[0]
    assert operation.request is request and operation.status is status
    sender._poll_in_doubt_transfers(0)
    assert not tx.resources_drained() and not rx.resources_drained()
    assert dealer.send.call_count == 1

    status.is_completed.return_value = True
    sender._poll_in_doubt_transfers(0)
    sender._poll_in_doubt_transfers(0)

    assert operation.request is None and operation.status is None
    assert all(isinstance(handle.poll(), Failed) for handle in handles)
    assert dealer.send.call_count == 2
    for session, allocation in ((tx, source), (rx, destination)):
        assert session.status is SessionStatus.ERROR
        assert session.resources_drained()
        allocation.apply_reuse_decision(session.close())
        allocation.apply_reuse_decision(session.close())
        assert allocation.is_reusable and allocation.release_count == 1
    assert not sender._pending_settlements[0]
    assert not worker._retirement_watchdog._owners


def test_context_first_cancel_settles_published_destination_without_sender_session() -> None:
    worker = _worker()
    sender, receiver = worker._sender, worker._receiver
    rx = worker.create_rx_session(_request())
    task = rx.prepare_receive(_sole_piece(tokens=16))
    assert task is not None
    task.expected_transfers = 1
    assert rx.try_begin_transfer(task.slice_id, {"ctx"}, writer_cohort={0})
    request_dealer = Mock()
    request_dealer.send.side_effect = sender._handle_cancel_session
    receiver._get_or_connect_dealer = Mock(return_value=request_dealer)
    result_dealer = Mock()
    result_dealer.send.side_effect = receiver._process_session_quiesced
    sender._get_or_connect_thread_dealer = Mock(return_value=result_dealer)

    assert rx.cancel()
    request_dealer.send.assert_called_once_with(
        [transfer_mod.MessageType.CANCEL_SESSION, b"401", b"worker", b"0"]
    )
    assert not sender._sessions
    assert not rx.resources_drained()
    marker = sender._send_task_queues[0].get_nowait()
    assert sender._send_session_quiesced(marker)
    assert rx.resources_drained()
    assert isinstance(TaskHandle(rx, task, 16).poll(), Cancelled)
    assert rx.close()

    late_sender = worker.create_tx_session(_request())
    assert late_sender.status is SessionStatus.CANCELLED
    assert not late_sender.kv_tasks
    assert late_sender.close()
    assert not worker._retirement_watchdog._owners


def test_context_first_unproven_deadline_keeps_both_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry notifies containment without authorizing source or destination reuse."""
    worker, tx, rx, meta, request, status, _ = _native_pair(monkeypatch, completes=False)
    watchdog = worker._retirement_watchdog
    notify = Mock()
    monkeypatch.setattr(watchdog, "_callback", notify)
    worker._sender._deliver_kv_to_agent(meta)
    monkeypatch.setattr(watchdog, "clock", lambda: 11.0)
    watchdog.progress()

    assert watchdog.fatal is not None
    notify.assert_called_once_with(watchdog.fatal)
    operation = tx.kv_tasks[0]._physical_operations[0]
    assert operation.request is request and operation.status is status
    for endpoint, session in ((worker._sender, tx), (worker._receiver, rx)):
        assert not session.resources_drained()
        assert not session.close()
        assert endpoint._sessions[401] is session
    with pytest.raises(RuntimeError, match="admission is closed"):
        worker.create_tx_session(_request())


@pytest.mark.parametrize("mismatch", ["topology", "ownership"])
def test_context_first_incompatible_peer_never_publishes_destination(
    monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    """The profile validator runs before any receive descriptor is sent."""
    worker = _worker()
    rx = worker.create_rx_session(_request())
    task = rx.prepare_receive(_sole_piece(tokens=16))
    assert task is not None
    local, peer = _rank_info(), _rank_info()
    if mismatch == "ownership":
        peer.context_first_ownership = False
    else:
        peer.tp_size = 1
    worker._receiver._registrar.self_rank_info = local
    worker._receiver._get_sender_info = Mock(
        return_value=transfer_mod.RankInfo.from_bytes(peer.to_bytes())
    )
    worker._receiver._get_or_connect_dealer = Mock()
    validator = Mock(wraps=transfer_mod.validate_context_first_peer)
    monkeypatch.setattr(transfer_mod, "validate_context_first_peer", validator)

    rx.dispatch_prepared_receive(task)

    validator.assert_called_once()
    worker._receiver._get_or_connect_dealer.assert_not_called()
    assert rx.has_failed()
    assert rx.resources_drained()
    assert rx.close()

    second_request = _request()
    second_request.py_request_id = 402
    second_request.py_disaggregated_params.disagg_request_id = 402
    second_rx = worker.create_rx_session(second_request)
    second_task = second_rx.prepare_receive(_sole_piece(tokens=16))
    assert second_task is not None
    second_rx.dispatch_prepared_receive(second_task)
    validator.assert_called_once()
    worker._receiver._get_sender_info.assert_called_once()
    worker._receiver._get_or_connect_dealer.assert_not_called()
    assert second_rx.has_failed() and second_rx.resources_drained()
    assert second_rx.close()


@pytest.mark.parametrize("style", [DisaggScheduleStyle.GENERATION_FIRST, None])
def test_context_first_rejection_cache_does_not_block_legacy_requests(
    monkeypatch: pytest.MonkeyPatch, style: DisaggScheduleStyle | None
) -> None:
    """A context-first-only rejection must not disable other requests to the same peer."""
    worker = _worker()
    receiver = worker._receiver
    receiver._context_first_incompatible_peers["ctx-info"] = "peer ownership is inactive"
    local, peer = _rank_info(tp_size=1), _rank_info(tp_size=1)
    peer.context_first_ownership = False
    peer.sender_endpoints = ["ctx-sender"]
    receiver._registrar.self_rank_info = local
    receiver._registrar.self_extractor = SimpleNamespace(page_table=local.page_table)
    receiver._registrar.get_peer_overlap = Mock(return_value=SimpleNamespace(ranks=[0]))
    receiver._get_sender_info = Mock(return_value=peer)
    receiver._request_sender_data = Mock()
    validator = Mock(side_effect=AssertionError("legacy request used context-first validation"))
    monkeypatch.setattr(transfer_mod, "validate_context_first_peer", validator)
    request = _request()
    request.py_disaggregated_params.schedule_style = style
    rx = worker.create_rx_session(request)

    try:
        rx.receive(_sole_piece(tokens=16))

        assert not rx._enforce_physical_ownership and rx._retirement is None
        assert not rx.has_failed()
        assert rx._kv_tasks[0].status is transfer_mod.TaskStatus.TRANSFERRING
        receiver._get_sender_info.assert_called_once_with(request.py_disaggregated_params)
        receiver._request_sender_data.assert_called_once()
        assert receiver._request_sender_data.call_args.args[0] == "ctx-sender"
        validator.assert_not_called()
    finally:
        assert rx.close()


@pytest.mark.parametrize("peer_context_first", [False, True])
def test_context_first_inactive_peer_never_submits_source(peer_context_first: bool) -> None:
    """Peer scheduling metadata cannot bypass the local protected task's admission check."""
    worker = _worker()
    sender = worker._sender
    tx = worker.create_tx_session(_request())
    tx.send(_sole_piece(tokens=16))
    task = tx.kv_tasks[0]
    local, peer = _rank_info(), _rank_info()
    peer.context_first_ownership = False
    sender._registrar.self_rank_info = local
    sender._registrar.get_peer_rank_info.return_value = transfer_mod.RankInfo.from_bytes(
        peer.to_bytes()
    )
    sender._build_kv_write_meta = Mock(side_effect=AssertionError("unqualified source access"))
    sender._enqueue = Mock()
    info = transfer_mod.RecvReqInfo(
        sender_req_id=401,
        instance_name="gen",
        instance_rank=0,
        block_ids_per_layer_groups=[],
        unique_rid=401,
        slice_id=0,
        context_first=peer_context_first,
    )

    with pytest.raises(ValueError, match="ownership-enabled peers"):
        sender._dispatch_task_to_peer(task, info)

    sender._build_kv_write_meta.assert_not_called()
    sender._enqueue.assert_not_called()
    assert task.resources_drained
    assert tx.cancel_local()
    assert tx.close()
    assert tx.close()


@pytest.mark.parametrize(
    "style,rid,synchronous,accepted",
    [
        (DisaggScheduleStyle.CONTEXT_FIRST, 401, False, True),
        (DisaggScheduleStyle.CONTEXT_FIRST, None, False, False),
        (DisaggScheduleStyle.CONTEXT_FIRST, -1, False, False),
        (DisaggScheduleStyle.CONTEXT_FIRST, 401, True, False),
        (DisaggScheduleStyle.GENERATION_FIRST, 401, True, True),
    ],
)
def test_context_first_request_gate_preserves_legacy_generation_first(
    style: DisaggScheduleStyle, rid: int | None, synchronous: bool, accepted: bool
) -> None:
    """Only the selected context-first path acquires the new admission contract."""
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._fp4_mla_bridge_enabled = False
    transceiver._context_first_ownership = True
    transceiver._context_first_ownership_logged = True
    request = SimpleNamespace(
        py_disaggregated_params=SimpleNamespace(schedule_style=style, disagg_request_id=rid)
    )
    assert transceiver._validate_bridge_req(request, synchronous=synchronous) is accepted
    if not accepted:
        assert request.state is LlmRequestState.DISAGG_TRANS_ERROR
