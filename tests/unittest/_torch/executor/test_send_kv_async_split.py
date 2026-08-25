# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structure tests for the _send_kv_async split.

``_send_kv_async`` composes two independent legs plus a reap, and the order
is load-bearing (see the comment in the wrapper): the disagg send must
register its transfer before the connector does, and the ctx reap must run
last so a quickly-completed send cannot terminate a request whose connector
transfer is not registered yet. These tests pin that structure, and pin the
property this split exists for: the connector leg keeps running when the
transceiver is disabled.
"""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def _stub_executor() -> PyExecutor:
    return object.__new__(PyExecutor)


def _wrapper_calls(executor: PyExecutor) -> list:
    calls = []
    executor._send_disagg_ctx_kv_async = lambda reqs: calls.append("disagg_send")
    executor._save_kv_to_connector_async = lambda reqs: calls.append("connector_save")
    executor._check_disagg_ctx_cache_transfer_status = lambda n: calls.append(f"ctx_reap:{n}")
    return calls


def test_wrapper_order_disagg_then_connector_then_reap() -> None:
    executor = _stub_executor()
    calls = _wrapper_calls(executor)
    executor.kv_cache_transceiver = object()

    PyExecutor._send_kv_async(executor, [])

    assert calls == ["disagg_send", "connector_save", "ctx_reap:0"]


def test_wrapper_keeps_connector_leg_without_transceiver() -> None:
    """Connector-only configs must keep working when the disagg path is off."""
    executor = _stub_executor()
    calls = _wrapper_calls(executor)
    executor.kv_cache_transceiver = None

    PyExecutor._send_kv_async(executor, [])

    assert calls == ["disagg_send", "connector_save"]


def test_disagg_send_leg_is_noop_without_transceiver() -> None:
    executor = _stub_executor()
    executor.kv_cache_transceiver = None
    # Anything past the guard would hit unset executor attributes and raise.
    PyExecutor._send_disagg_ctx_kv_async(executor, [Mock()])


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
        py_request_id=request_id,
        py_kv_transfer_start_time=None,
    )


def test_disagg_send_leg_stores_blocks_before_sending() -> None:
    executor = _stub_executor()
    transceiver = Mock()
    transceiver.kv_transfer_timeout_ms = 1000
    executor.kv_cache_transceiver = transceiver
    executor.kv_cache_manager = Mock(spec=[])  # no release_index_slot
    executor.async_transfer_manager = Mock()
    order = Mock()
    order.attach_mock(executor.async_transfer_manager.start_transfer, "start")
    order.attach_mock(transceiver.respond_and_send_async, "send")
    req = _finished_ctx_only_request()

    PyExecutor._send_disagg_ctx_kv_async(executor, [req])

    assert order.mock_calls == [call.start(req), call.send(req)]
    assert req.py_kv_transfer_start_time is not None


def test_disagg_send_leg_skips_timeout_stamp_when_disabled() -> None:
    executor = _stub_executor()
    transceiver = Mock()
    transceiver.kv_transfer_timeout_ms = None
    executor.kv_cache_transceiver = transceiver
    executor.kv_cache_manager = Mock(spec=[])
    executor.async_transfer_manager = Mock()
    req = _finished_ctx_only_request()

    PyExecutor._send_disagg_ctx_kv_async(executor, [req])

    transceiver.respond_and_send_async.assert_called_once_with(req)
    assert req.py_kv_transfer_start_time is None


def test_disagg_send_leg_skips_cancelled_and_unfinished_requests() -> None:
    executor = _stub_executor()
    transceiver = Mock()
    executor.kv_cache_transceiver = transceiver
    executor.kv_cache_manager = Mock(spec=[])
    executor.async_transfer_manager = Mock()
    cancelled = _finished_ctx_only_request(2)
    cancelled.is_finished_due_to_cancellation = True
    unfinished = _finished_ctx_only_request(3)
    unfinished.is_context_finished = False

    PyExecutor._send_disagg_ctx_kv_async(executor, [cancelled, unfinished])

    executor.async_transfer_manager.start_transfer.assert_not_called()
    transceiver.respond_and_send_async.assert_not_called()


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
