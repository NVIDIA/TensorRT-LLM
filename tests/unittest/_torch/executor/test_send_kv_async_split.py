# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structure tests for the _send_kv_async split.

``_send_kv_async`` composes two independent legs plus a reap, and the order
is load-bearing (see the comment in the wrapper): the disagg send must
register its transfer before the connector does, and the ctx reap must run
last so a quickly-completed send cannot terminate a request whose connector
transfer is not registered yet. These tests pin that structure, and pin the
property this split exists for: the connector leg keeps running when the
transceiver is disabled. After the coordinator extraction, that property comes
from the lazily built no-op coordinator, so the transceiver-off test goes
through the real builder rather than an injected coordinator.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import CtxTransferStatus
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import NoopDisaggCoordinator
from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType

pytestmark = pytest.mark.cpu_only


def _stub_executor() -> PyExecutor:
    return object.__new__(PyExecutor)


def _wrapper_calls(executor: PyExecutor) -> list:
    calls = []
    executor._disagg_coordinator = SimpleNamespace(
        send_completed_context=lambda reqs: calls.append("disagg_send"),
        reap_context_sends=lambda n: calls.append(f"ctx_reap:{n}"),
    )
    executor._save_kv_to_connector_async = lambda reqs: calls.append("connector_save")
    return calls


def test_wrapper_order_disagg_then_connector_then_reap() -> None:
    executor = _stub_executor()
    calls = _wrapper_calls(executor)

    PyExecutor._send_kv_async(executor, [])

    assert calls == ["disagg_send", "connector_save", "ctx_reap:0"]


def test_wrapper_keeps_connector_leg_without_transceiver() -> None:
    """Connector-only configs: with no transceiver the executor lazily builds
    the no-op coordinator, which turns both disagg legs into no-ops around the
    connector save. No coordinator is injected so the builder is exercised."""
    executor = _stub_executor()
    executor.kv_cache_transceiver = None
    calls = []
    executor._save_kv_to_connector_async = lambda reqs: calls.append("connector_save")

    PyExecutor._send_kv_async(executor, [])

    assert isinstance(executor.disagg, NoopDisaggCoordinator)
    assert calls == ["connector_save"]


def test_connector_save_leg_is_noop_without_connector() -> None:
    executor = _stub_executor()
    executor.kv_connector_manager = None
    PyExecutor._save_kv_to_connector_async(executor, [Mock()])


def _finished_ctx_only_request(request_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        is_context_only_request=True,
        is_context_finished=True,
        is_finished_due_to_length=False,
        is_finished_due_to_cancellation=False,
        is_child=False,
        parent_request_id=None,
        py_request_id=request_id,
        py_kv_transfer_start_time=None,
    )


def _connector_executor() -> PyExecutor:
    executor = _stub_executor()
    executor.kv_connector_manager = Mock()
    executor.kv_connector_manager.request_finished.return_value = True
    executor.kv_cache_manager = Mock()
    executor.kv_cache_manager.get_cache_indices.return_value = [7]
    executor.async_transfer_manager = Mock()
    return executor


def test_connector_save_uses_previous_batch_with_overlap_scheduler() -> None:
    executor = _connector_executor()
    executor.disable_overlap_scheduler = False
    prev_req = SimpleNamespace(is_finished=True, py_request_id=2)
    executor.previous_batch = SimpleNamespace(
        scheduled_requests=SimpleNamespace(all_requests=lambda: [prev_req])
    )
    current_req = SimpleNamespace(is_finished=True, py_request_id=3)

    PyExecutor._save_kv_to_connector_async(executor, [current_req])

    executor.kv_connector_manager.request_finished.assert_called_once_with(prev_req, [7])
    executor.async_transfer_manager.start_transfer.assert_called_once_with(prev_req)


def test_connector_save_uses_scheduled_batch_without_overlap_scheduler() -> None:
    executor = _connector_executor()
    executor.disable_overlap_scheduler = True
    finished = SimpleNamespace(is_finished=True, py_request_id=4)
    running = SimpleNamespace(is_finished=False, py_request_id=5)

    PyExecutor._save_kv_to_connector_async(executor, [finished, running])

    executor.kv_connector_manager.request_finished.assert_called_once_with(finished, [7])
    executor.async_transfer_manager.start_transfer.assert_called_once_with(finished)


def test_connector_save_skips_transfer_when_connector_declines() -> None:
    executor = _connector_executor()
    executor.disable_overlap_scheduler = True
    executor.kv_connector_manager.request_finished.return_value = False
    finished = SimpleNamespace(is_finished=True, py_request_id=6)

    PyExecutor._save_kv_to_connector_async(executor, [finished])

    executor.async_transfer_manager.start_transfer.assert_not_called()


def _dual_claim_executor() -> PyExecutor:
    """Executor running the real wrapper, a real coordinator, the real
    connector leg and a real AsyncTransferManager; only the transceiver,
    connector, and KV cache manager boundaries are mocked."""
    executor = _stub_executor()
    kv_cache_manager = Mock()
    kv_cache_manager.get_cache_indices.return_value = [7]
    executor.kv_cache_manager = kv_cache_manager
    resource_manager = SimpleNamespace(
        resource_managers={ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager}
    )
    executor.async_transfer_manager = AsyncTransferManager(resource_manager)
    transceiver = Mock()
    transceiver.kv_transfer_timeout_ms = None
    transceiver.has_retired_send_session.return_value = False
    executor.kv_cache_transceiver = transceiver
    executor.kv_connector_manager = Mock()
    executor.disable_overlap_scheduler = True
    executor.active_requests = []
    executor.canceled_req_ids = []
    executor.force_terminate_ctx_for_partial_reuse = False
    executor.dist = SimpleNamespace(rank=0, world_size=2)
    executor._terminate_request = Mock()
    # Make the reap's trailing _check_cache_transfer_errors a no-op.
    executor.enable_attention_dp = True
    return executor


def _dual_claim_request(request_id: int) -> SimpleNamespace:
    req = _finished_ctx_only_request(request_id)
    req.is_finished = True  # the connector leg selects finished requests
    req.py_kv_transfer_timed_out = False
    req.state = None  # start_transfer overwrites
    return req


def test_reap_keeps_request_still_claimed_by_connector() -> None:
    """The hazard the wrapper order exists for: a send that completes within
    the same iteration must not release a request the connector also claimed.
    Reaping before the connector leg would drop the transfer refcount to zero
    and terminate the request; this test fails under that reordering."""
    executor = _dual_claim_executor()
    executor.kv_connector_manager.request_finished.return_value = True
    req = _dual_claim_request(9)
    # The send completes instantly, so the reap sees it in the same call.
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = CtxTransferStatus(
        [9], []
    )

    PyExecutor._send_kv_async(executor, [req])

    # The reap released only the send's claim; the connector's claim keeps
    # the request pinned.
    assert 9 in executor.async_transfer_manager.requests_in_transfer()
    executor.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    executor._terminate_request.assert_not_called()


def test_reap_releases_request_once_connector_declines() -> None:
    """Counterpart: with only the send's claim outstanding, the same fast
    completion does release and terminate the request."""
    executor = _dual_claim_executor()
    executor.kv_connector_manager.request_finished.return_value = False
    req = _dual_claim_request(9)
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = CtxTransferStatus(
        [9], []
    )

    PyExecutor._send_kv_async(executor, [req])

    assert 9 not in executor.async_transfer_manager.requests_in_transfer()
    executor.kv_cache_manager.unpin_blocks_by_id.assert_called_once()
    executor._terminate_request.assert_called_once_with(req)


def test_connector_release_without_transceiver_terminates_on_last_claim() -> None:
    """Connector-only configs release through the executor, not the
    coordinator: the request terminates as soon as its save completes."""
    executor = _stub_executor()
    executor.kv_cache_transceiver = None
    executor.force_terminate_ctx_for_partial_reuse = False
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.end_transfer.return_value = True
    executor._terminate_request = Mock()
    req = Mock()

    PyExecutor._release_transfer(executor, req)

    executor._terminate_request.assert_called_once_with(req)
