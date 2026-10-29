# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context cancellation leaves native transfer retirement independently owned."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_context_first_ownership import _native_pair, _request, _worker
from test_transfer_ownership_regressions import _sole_piece

from tensorrt_llm._torch.disaggregation.native import transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.native.extent import Cancelled, Failed
from tensorrt_llm._torch.disaggregation.native.handle import TaskHandle
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only


def _context_request() -> SimpleNamespace:
    """Build the request fields consumed by native context send retirement."""
    request = _request()
    request.is_context_only_request = True
    request.is_child = False
    request.py_kv_send_session_retired = False
    request.py_kv_transfer_timed_out = False
    request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    return request


def _harness(
    worker: transfer_mod.TransferWorker,
    request: SimpleNamespace,
    session: transfer_mod.TxSession | None,
) -> SimpleNamespace:
    """Run real coordinator/transfer-manager/reaper code with controlled engine effects."""
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._context_first_ownership = True
    transceiver._send_sessions = {} if session is None else {401: session}
    transceiver._send_reqs = {} if session is None else {401: request}
    transceiver._ever_had_send_session = session is not None
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._dist = Mock(tp_size=1)
    transceiver._transfer_worker = worker
    # Expiring unrelated peer-discovery metadata is outside this cancellation test.
    worker.sweep_stale_req_infos = Mock()
    manager = Mock(spec=["store_blocks_for_reuse", "unpin_blocks_by_id"])
    manager.store_blocks_for_reuse.return_value = 91
    transfers = AsyncTransferManager(
        SimpleNamespace(resource_managers={ResourceManagerType.KV_CACHE_MANAGER: manager})
    )
    active = []
    registry = SimpleNamespace(
        contains=lambda req: req in active,
        active_requests=lambda: active,
        canceled_request_ids=lambda: [],
    )
    effects = Mock()
    coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=transfers,
        kv_cache_manager=manager,
        dist=SimpleNamespace(world_size=1),
        effects=effects,
        registry=registry,
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
    )
    return SimpleNamespace(
        transceiver=transceiver,
        coordinator=coordinator,
        transfers=transfers,
        manager=manager,
        active=active,
        effects=effects,
    )


@pytest.mark.parametrize(
    "enabled,context_only,style,rid,expected",
    [
        (True, True, DisaggScheduleStyle.CONTEXT_FIRST, 401, True),
        (False, True, DisaggScheduleStyle.CONTEXT_FIRST, 401, False),
        (True, False, DisaggScheduleStyle.CONTEXT_FIRST, 401, False),
        (True, True, DisaggScheduleStyle.GENERATION_FIRST, 401, False),
        (True, True, DisaggScheduleStyle.CONTEXT_FIRST, -1, False),
        (True, True, DisaggScheduleStyle.CONTEXT_FIRST, True, False),
    ],
)
def test_logical_context_cancel_keeps_the_qualified_profile_gate(
    enabled: bool,
    context_only: bool,
    style: DisaggScheduleStyle,
    rid: int,
    expected: bool,
) -> None:
    """Cancellation does not silently activate another layout, direction, or identity."""
    request = _context_request()
    request.is_context_only_request = context_only
    request.py_disaggregated_params.schedule_style = style
    request.py_disaggregated_params.disagg_request_id = rid
    harness = _harness(_worker(), request, None)
    harness.transceiver._context_first_ownership = enabled

    assert harness.coordinator.supports_logical_context_cancellation() is enabled
    assert harness.coordinator.supports_logical_context_cancellation(request) is expected
    assert harness.coordinator.request_logical_context_cancellation(request) is expected
    assert request.py_kv_send_session_retired is expected
    assert not harness.transfers.requests_in_transfer()
    harness.effects.terminate_request.assert_not_called()


def test_context_cancel_before_publication_prevents_a_new_send() -> None:
    """Cancellation without a send session creates no peer or transfer dependency."""
    request = _context_request()
    harness = _harness(_worker(), request, None)
    harness.transceiver._transfer_worker.create_tx_session = Mock()

    assert harness.coordinator.request_logical_context_cancellation(request)
    assert harness.coordinator.request_logical_context_cancellation(request)
    assert harness.transceiver.has_retired_send_session(request)
    assert harness.transceiver._get_or_create_send_session(request) is None
    harness.transceiver._transfer_worker.create_tx_session.assert_not_called()


@pytest.mark.parametrize("result_already_sent", [False, True])
def test_context_cancel_preserves_an_earlier_request_failure(
    result_already_sent: bool,
) -> None:
    """Stopping a send preserves a committed request error without replaying its result."""
    worker = _worker()
    request = _context_request()
    tx = worker.create_tx_session(request)
    tx.send(_sole_piece(tokens=16))
    harness = _harness(worker, request, tx)
    harness.transfers.start_transfer(request)
    request.state = LlmRequestState.DISAGG_TRANS_ERROR
    tx.set_exception("original failure")
    if not result_already_sent:
        harness.active.append(request)

        def fail_requests(*_args: object, **_kwargs: object) -> None:
            harness.active.remove(request)

        harness.effects.fail_requests.side_effect = fail_requests

    assert harness.coordinator.request_logical_context_cancellation(request)
    assert request.state is LlmRequestState.DISAGG_TRANS_ERROR
    harness.coordinator.reap_context_sends()
    harness.coordinator.reap_context_sends()

    assert request.state is LlmRequestState.DISAGG_TRANS_ERROR
    harness.manager.unpin_blocks_by_id.assert_called_once_with(91)
    harness.effects.stage_transfer_response.assert_not_called()
    if result_already_sent:
        harness.effects.fail_requests.assert_not_called()
        harness.effects.terminate_request.assert_called_once_with(request)
    else:
        harness.effects.fail_requests.assert_called_once()
        harness.effects.terminate_request.assert_not_called()
    assert not harness.coordinator._context_cancellation_drains


def test_native_failure_cannot_independently_rewrite_request_cancellation() -> None:
    """A rank-local task error is not a synchronized request outcome."""
    worker = _worker()
    request = _context_request()
    tx = worker.create_tx_session(request)
    tx.send(_sole_piece(tokens=16))
    harness = _harness(worker, request, tx)
    harness.transfers.start_transfer(request)
    handle = TaskHandle(tx, tx.kv_tasks[0], token_end=16)
    tx.set_exception("rank-local transfer failure")

    assert harness.coordinator.request_logical_context_cancellation(request)
    assert request.state is LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    assert isinstance(handle.poll(), Failed)
    # The executor commits the synchronized request cancellation independently
    # of this native task's earlier outcome.
    request.state = LlmRequestState.GENERATION_COMPLETE
    harness.coordinator.reap_context_sends()

    assert request.state is LlmRequestState.GENERATION_COMPLETE
    assert isinstance(handle.poll(), Failed)
    harness.manager.unpin_blocks_by_id.assert_called_once_with(91)
    harness.effects.terminate_request.assert_called_once_with(request)
    harness.effects.fail_requests.assert_not_called()
    harness.effects.stage_transfer_response.assert_not_called()


def test_context_retirement_leaves_a_pending_logical_response_to_the_executor() -> None:
    """A fast physical drain cannot publish or terminate ahead of the cancel response."""
    worker = _worker()
    request = _context_request()
    tx = worker.create_tx_session(request)
    tx.send(_sole_piece(tokens=16))
    harness = _harness(worker, request, tx)
    harness.transfers.start_transfer(request)
    harness.active.append(request)

    assert harness.coordinator.request_logical_context_cancellation(request)
    request.state = LlmRequestState.GENERATION_COMPLETE
    harness.coordinator.reap_context_sends()

    assert request.state is LlmRequestState.GENERATION_COMPLETE
    assert harness.active == [request]
    harness.manager.unpin_blocks_by_id.assert_called_once_with(91)
    harness.effects.terminate_request.assert_not_called()
    harness.effects.stage_transfer_response.assert_not_called()


@pytest.mark.parametrize("result_already_sent", [False, True])
def test_context_cancel_without_gen_retires_once_and_fences_late_receiver(
    result_already_sent: bool,
) -> None:
    """A source with no receiver can retire without fabricating a GEN request."""
    worker = _worker()
    request = _context_request()
    tx = worker.create_tx_session(request)
    tx.send(_sole_piece(tokens=16))
    harness = _harness(worker, request, tx)
    harness.transfers.start_transfer(request)
    if not result_already_sent:
        request.state = LlmRequestState.GENERATION_COMPLETE
    original_state = request.state
    assert not worker._sender._peer_requests

    assert harness.coordinator.request_logical_context_cancellation(request)
    assert harness.coordinator.request_logical_context_cancellation(request)
    assert harness.transceiver._send_sessions[401] is tx
    assert harness.transfers.requests_in_transfer()[401] is request
    harness.manager.unpin_blocks_by_id.assert_not_called()

    harness.coordinator.reap_context_sends()
    harness.coordinator.reap_context_sends()

    assert not harness.transceiver._send_sessions
    assert not harness.transfers.requests_in_transfer()
    harness.manager.unpin_blocks_by_id.assert_called_once_with(91)
    harness.effects.terminate_request.assert_called_once_with(request)
    harness.effects.stage_transfer_response.assert_not_called()
    harness.effects.fail_requests.assert_not_called()
    if not result_already_sent:
        assert request.state is original_state
    assert not harness.coordinator._context_cancellation_drains
    assert not worker._retirement_watchdog._owners

    late = transfer_mod.RecvReqInfo(
        sender_req_id=401,
        instance_name="gen",
        instance_rank=0,
        block_ids_per_layer_groups=[],
        unique_rid=401,
        slice_id=0,
        context_first=True,
    )
    worker._sender._queue_session_quiescence = Mock()
    worker._sender._save_peer_req_info = Mock()
    worker._sender._respond_with_kv(b"peer", [b"REQUEST_DATA", late.to_bytes()])
    worker._sender._queue_session_quiescence.assert_called_once_with(401, "gen", 0)
    worker._sender._save_peer_req_info.assert_not_called()
    assert not worker._sender._sessions


@pytest.mark.parametrize("settles", [False, True])
def test_context_logical_cancel_does_not_wait_for_native_done(
    monkeypatch: pytest.MonkeyPatch, settles: bool
) -> None:
    """Cancellation accepts after submission, retaining roots until DONE or fatal expiry."""
    worker, tx, rx, meta, backend_request, status, _ = _native_pair(monkeypatch, completes=False)
    request = _context_request()
    harness = _harness(worker, request, tx)
    harness.transfers.start_transfer(request)
    handle = TaskHandle(tx, tx.kv_tasks[0], token_end=16)

    def cancel_during_wait() -> bool:
        assert harness.coordinator.request_logical_context_cancellation(request)
        request.state = LlmRequestState.GENERATION_COMPLETE
        assert harness.transfers.requests_in_transfer()[401] is request
        assert not tx.resources_drained()
        return False

    status.wait.side_effect = cancel_during_wait
    worker._sender._deliver_kv_to_agent(meta)
    assert isinstance(handle.poll(), Cancelled)
    harness.coordinator.reap_context_sends()
    harness.effects.terminate_request.assert_not_called()
    harness.manager.unpin_blocks_by_id.assert_not_called()
    operation = tx.kv_tasks[0]._physical_operations[0]
    assert operation.request is backend_request and operation.status is status

    if settles:
        status.is_completed.return_value = True
        worker._sender._poll_in_doubt_transfers(0)
        harness.coordinator.reap_context_sends()
        harness.coordinator.reap_context_sends()
        assert isinstance(handle.poll(), Cancelled)
        assert request.state is LlmRequestState.GENERATION_COMPLETE
        assert operation.request is None and operation.status is None
        harness.manager.unpin_blocks_by_id.assert_called_once_with(91)
        harness.effects.terminate_request.assert_called_once_with(request)
        harness.effects.stage_transfer_response.assert_not_called()
        harness.effects.fail_requests.assert_not_called()
        assert rx.close()
        assert not worker._retirement_watchdog._owners
    else:
        notify = Mock()
        monkeypatch.setattr(worker._retirement_watchdog, "_callback", notify)
        monkeypatch.setattr(worker._retirement_watchdog, "clock", lambda: 11.0)
        worker._retirement_watchdog.progress()
        notify.assert_called_once()
        assert not tx.close() and not rx.close()
        assert harness.transfers.requests_in_transfer()[401] is request
        assert operation.request is backend_request and operation.status is status
        harness.manager.unpin_blocks_by_id.assert_not_called()
        harness.effects.terminate_request.assert_not_called()
