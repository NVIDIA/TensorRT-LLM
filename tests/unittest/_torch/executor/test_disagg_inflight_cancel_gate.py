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

import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tensorrt_llm._torch.pyexecutor import kv_cache_transceiver as transceiver_module
from tensorrt_llm._torch.pyexecutor import py_executor as executor_module
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import BindKvCacheTransceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import (
    AsyncTransferManager,
    AsyncTransferProvider,
    PyExecutor,
)
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig


@pytest.fixture(autouse=True)
def _reset_inflight_cancel_env_cache(monkeypatch):
    monkeypatch.delenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, raising=False)
    monkeypatch.delenv(transceiver_module._NIXL_KVCACHE_BACKEND_ENV, raising=False)
    monkeypatch.delenv(transceiver_module._DISABLE_KV_CACHE_TRANSFER_OVERLAP_ENV, raising=False)
    monkeypatch.delenv(transceiver_module._DISAGG_LAYERWISE_ENV, raising=False)
    monkeypatch.delenv(transceiver_module._TRY_ZCOPY_FOR_KV_CACHE_TRANSFER_ENV, raising=False)
    for env_name, _ in transceiver_module._CACHE_TRANSCEIVER_BACKEND_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(transceiver_module, "_disagg_inflight_cancel_enabled_cache", None)


def _make_timeout_request(request_id=7, in_progress=False):
    return SimpleNamespace(
        is_attention_dp_dummy=False,
        py_kv_transfer_timed_out=True,
        py_request_id=request_id,
        is_disagg_generation_transmission_in_progress=in_progress,
        state=object(),
    )


def _make_response_handler_stub(active_requests, tp_allgather_result):
    executor = object.__new__(PyExecutor)
    executor.active_requests = list(active_requests)
    executor.perf_manager = Mock()
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = True
    executor._disagg_inflight_cancel_unsupported_logged = False
    executor._pending_timed_out_requests = []
    executor.enable_attention_dp = True
    executor.dist = SimpleNamespace(
        rank=0,
        world_size=2,
        tp_allgather=Mock(return_value=tp_allgather_result),
    )
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()
    executor._handle_errors = Mock()
    executor._timeout_cleanup_order = Mock()
    executor._timeout_cleanup_order.attach_mock(executor.dist.tp_allgather, "vote")
    executor._timeout_cleanup_order.attach_mock(executor._handle_errors, "handle")
    return executor


def test_flag_unset_short_circuits_before_capability_query(monkeypatch):
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor._disagg_inflight_cancel_unsupported_logged = False
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled", lambda: False)

    assert not PyExecutor._is_disagg_inflight_cancel_active(executor)
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.assert_not_called()


def test_unsupported_transceiver_warns_once(monkeypatch):
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = False
    executor._disagg_inflight_cancel_unsupported_logged = False
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled", lambda: True)
    warning = Mock()
    monkeypatch.setattr(executor_module.logger, "warning", warning)

    assert not PyExecutor._is_disagg_inflight_cancel_active(executor)
    assert not PyExecutor._is_disagg_inflight_cancel_active(executor)

    assert executor._disagg_inflight_cancel_unsupported_logged
    warning.assert_called_once()


def test_flag_unset_generation_timeout_uses_rank_uniform_cleanup():
    request = _make_timeout_request()
    executor = _make_response_handler_stub([request], [True, False])

    PyExecutor._handle_responses(executor)
    PyExecutor._handle_kv_transfer_timeouts_synced(executor)

    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    assert executor.active_requests == []
    executor.dist.tp_allgather.assert_called_once_with(True)
    executor._handle_errors.assert_called_once_with(
        error_msg="Request timed out (KV transfer)",
        requests=[request],
        charge_budget=False,
    )
    assert executor._timeout_cleanup_order.mock_calls == [
        call.vote(True),
        call.handle(
            error_msg="Request timed out (KV transfer)",
            requests=[request],
            charge_budget=False,
        ),
    ]


def test_flag_unset_generation_timeout_peer_enters_cleanup():
    executor = _make_response_handler_stub([], [False, True])

    PyExecutor._handle_responses(executor)
    PyExecutor._handle_kv_transfer_timeouts_synced(executor)

    executor.kv_cache_transceiver.cancel_request.assert_not_called()
    executor.dist.tp_allgather.assert_called_once_with(False)
    executor._handle_errors.assert_called_once_with(
        error_msg="Request timed out (KV transfer)",
        requests=[],
        charge_budget=False,
    )
    assert executor._timeout_cleanup_order.mock_calls == [
        call.vote(False),
        call.handle(
            error_msg="Request timed out (KV transfer)",
            requests=[],
            charge_budget=False,
        ),
    ]


def test_flag_unset_generation_timeout_keeps_uncancellable_request_active():
    request = _make_timeout_request(in_progress=True)
    executor = _make_response_handler_stub([request], [False, False])
    executor.kv_cache_transceiver.cancel_request.return_value = False

    PyExecutor._handle_responses(executor)

    assert executor.active_requests == [request]
    assert executor._pending_timed_out_requests == []
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)


def test_enabled_generation_timeout_waits_for_inflight_terminal_state(monkeypatch):
    request = _make_timeout_request(in_progress=True)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    executor = _make_response_handler_stub([request], [False, False])
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled", lambda: True)

    PyExecutor._handle_responses(executor)

    assert executor.active_requests == [request]
    assert executor._pending_timed_out_requests == []
    executor.kv_cache_transceiver.cancel_request.assert_not_called()


def test_enabled_generation_timeout_fails_transfer_that_completed_late(monkeypatch):
    request = _make_timeout_request(in_progress=False)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    executor = _make_response_handler_stub([request], [True, False])
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled", lambda: True)

    PyExecutor._handle_responses(executor)

    assert executor.active_requests == []
    assert executor._pending_timed_out_requests == [request]
    executor.kv_cache_transceiver.cancel_request.assert_not_called()


def test_flag_unset_context_timeout_preserves_legacy_cleanup():
    request = _make_timeout_request()
    request.py_kv_transfer_start_time = 1.0
    request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    executor = object.__new__(PyExecutor)
    executor.canceled_req_ids = []
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = ([], [])
    executor.kv_cache_transceiver.take_context_cancelled_request_ids.return_value = []
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }
    executor._disagg_timed_out_ctx_cancelled_ids = set()
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = True
    executor._disagg_inflight_cancel_unsupported_logged = False
    executor._end_transfer_and_maybe_terminate = Mock()
    executor._check_cache_transfer_errors = Mock()

    PyExecutor._check_disagg_ctx_cache_transfer_status(executor, 0)

    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    assert request.py_kv_transfer_start_time is None
    assert request.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    executor._end_transfer_and_maybe_terminate.assert_called_once_with(
        request, AsyncTransferProvider.TRANSCEIVER
    )
    assert request.py_request_id not in executor._disagg_timed_out_ctx_cancelled_ids


def test_enabled_context_timeout_defers_cleanup_until_cpp_terminal_state(monkeypatch):
    request = _make_timeout_request()
    request.py_kv_transfer_start_time = 1.0
    request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    executor = object.__new__(PyExecutor)
    executor.canceled_req_ids = []
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = ([], [])
    executor.kv_cache_transceiver.take_context_cancelled_request_ids.return_value = []
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = True
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }
    executor._disagg_timed_out_ctx_cancelled_ids = set()
    executor._disagg_inflight_cancel_unsupported_logged = False
    executor._end_transfer_and_maybe_terminate = Mock()
    executor._check_cache_transfer_errors = Mock()
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled", lambda: True)

    PyExecutor._check_disagg_ctx_cache_transfer_status(executor, 0)

    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    assert request.py_request_id in executor._disagg_timed_out_ctx_cancelled_ids
    executor._end_transfer_and_maybe_terminate.assert_not_called()


def test_context_transfer_error_keeps_request_active_until_all_owners_release():
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_TRANS_ERROR,
        py_request_id=7,
    )
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.active_requests = [request]
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.end_transfer.return_value = True
    executor._terminate_request = Mock()

    PyExecutor._end_transfer_and_maybe_terminate(executor, request)

    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, AsyncTransferProvider.GENERIC
    )
    assert executor.active_requests == [request]
    executor._terminate_request.assert_not_called()


def test_context_transfer_error_cleanup_waits_for_async_owners():
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_TRANS_ERROR,
        py_request_id=7,
        is_child=False,
        is_context_only_request=True,
    )
    executor = object.__new__(PyExecutor)
    executor.active_requests = [request]
    executor.canceled_req_ids = []
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }

    assert PyExecutor._get_disagg_reqs_in_error_state(executor) == []

    executor.async_transfer_manager.requests_in_transfer.return_value = {}
    assert PyExecutor._get_disagg_reqs_in_error_state(executor) == [request]


def test_user_cancel_waits_for_context_transfer_owners(monkeypatch):
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_TRANS_ERROR,
        py_request_id=7,
        is_child=False,
        is_context_only_request=True,
        py_kv_transfer_timed_out=True,
        py_decoding_iter=3,
        finish_by_reason=Mock(),
    )
    executor = object.__new__(PyExecutor)
    executor.active_requests = [request]
    executor.canceled_req_ids = [request.py_request_id]
    executor.waiting_queue = Mock()
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = True
    executor._disagg_inflight_cancel_unsupported_logged = False
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }
    executor.async_transfer_manager.has_transfer.return_value = True
    executor.async_transfer_manager.end_transfer.return_value = False
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled", lambda: True)

    PyExecutor._handle_canceled_requests(executor)

    assert executor.canceled_req_ids == [request.py_request_id]
    request.finish_by_reason.assert_not_called()
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)

    executor.async_transfer_manager.requests_in_transfer.return_value = {}
    PyExecutor._handle_canceled_requests(executor)

    assert executor.canceled_req_ids == []
    assert request.py_kv_transfer_timed_out is False
    request.finish_by_reason.assert_called_once()


def test_user_cancel_routes_waiting_request_through_transceiver_owner():
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_GENERATION_INIT,
        py_request_id=7,
        is_context_only_request=False,
    )
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.owns_request.return_value = True
    executor.kv_cache_transceiver.cancel_request.return_value = False

    assert not PyExecutor._try_cancel_request(executor, request)
    executor.kv_cache_transceiver.owns_request.assert_called_once_with(request)
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)


def test_user_cancel_frees_unowned_request_without_transceiver_call():
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_GENERATION_INIT,
        py_request_id=7,
        is_context_only_request=False,
    )
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.owns_request.return_value = False

    assert PyExecutor._try_cancel_request(executor, request)
    executor.kv_cache_transceiver.cancel_request.assert_not_called()


def test_user_cancel_notifies_transceiver_owner_before_async_manager_release():
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER,
        py_request_id=7,
        is_context_only_request=True,
    )
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.owns_request.return_value = True
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }
    executor.async_transfer_manager.has_transfer.return_value = True
    # A connector leg remains after the transceiver leg is acknowledged.
    executor.async_transfer_manager.end_transfer.return_value = False

    assert not PyExecutor._try_cancel_request(executor, request)
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, AsyncTransferProvider.TRANSCEIVER
    )
    assert request.py_request_id in executor._disagg_acknowledged_ctx_cancel_legs

    # Retrying must not decrement the connector leg a second time.
    assert not PyExecutor._try_cancel_request(executor, request)
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, AsyncTransferProvider.TRANSCEIVER
    )

    executor.async_transfer_manager.requests_in_transfer.return_value = {}
    executor.kv_cache_transceiver.owns_request.return_value = False
    assert PyExecutor._try_cancel_request(executor, request)
    assert request.py_request_id not in executor._disagg_acknowledged_ctx_cancel_legs


def test_user_cancel_does_not_retire_connector_after_transceiver_completion(
    monkeypatch,
):
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        py_request_id=17,
        is_context_only_request=True,
        py_kv_transfer_start_time=1.0,
        py_kv_transfer_timed_out=False,
    )
    resource_manager = Mock()
    resource_manager.resource_managers = {}
    transfer_manager = AsyncTransferManager(resource_manager, should_store_blocks=False)
    transfer_manager.start_transfer(request, AsyncTransferProvider.TRANSCEIVER)
    transfer_manager.start_transfer(request, AsyncTransferProvider.CONNECTOR)

    executor = object.__new__(PyExecutor)
    executor.async_transfer_manager = transfer_manager
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.owns_request.return_value = False
    # A no-session cancellation historically reported success and retired the
    # anonymous counter's remaining connector leg.
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.kv_cache_transceiver.kv_transfer_timeout_ms = 1
    executor.active_requests = []
    executor.force_terminate_ctx_for_partial_reuse = False
    executor._terminate_request = Mock()
    executor._is_disagg_inflight_cancel_active = Mock(return_value=True)

    PyExecutor._end_transfer_and_maybe_terminate(
        executor, request, AsyncTransferProvider.TRANSCEIVER
    )

    assert not transfer_manager.has_transfer(request, AsyncTransferProvider.TRANSCEIVER)
    assert transfer_manager.has_transfer(request, AsyncTransferProvider.CONNECTOR)
    assert request.py_request_id in transfer_manager.requests_in_transfer()

    monkeypatch.setattr(executor_module.time, "monotonic", lambda: 10.0)
    PyExecutor._check_kv_transfer_timeout(executor)
    assert not request.py_kv_transfer_timed_out

    assert not PyExecutor._try_cancel_request(executor, request)

    executor.kv_cache_transceiver.cancel_request.assert_not_called()
    executor._is_disagg_inflight_cancel_active.assert_not_called()
    assert transfer_manager.has_transfer(request, AsyncTransferProvider.CONNECTOR)
    assert request.py_request_id in transfer_manager.requests_in_transfer()
    resource_manager.free_resources.assert_not_called()
    executor._terminate_request.assert_not_called()


def test_duplicate_transceiver_completion_does_not_touch_connector_leg():
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        py_request_id=18,
        is_child=False,
        is_context_only_request=True,
        py_kv_transfer_timed_out=False,
    )
    resource_manager = Mock()
    resource_manager.resource_managers = {}
    transfer_manager = AsyncTransferManager(resource_manager, should_store_blocks=False)
    transfer_manager.start_transfer(request, AsyncTransferProvider.CONNECTOR)

    executor = object.__new__(PyExecutor)
    executor.async_transfer_manager = transfer_manager
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = (
        [request.py_request_id],
        [],
    )
    executor.kv_cache_transceiver.take_context_cancelled_request_ids.return_value = []
    executor.canceled_req_ids = []
    executor._disagg_timed_out_ctx_cancelled_ids = set()
    executor._end_transfer_and_maybe_terminate = Mock()
    executor._check_cache_transfer_errors = Mock()

    PyExecutor._check_disagg_ctx_cache_transfer_status(executor, 0)

    executor._end_transfer_and_maybe_terminate.assert_not_called()
    assert transfer_manager.has_transfer(request, AsyncTransferProvider.CONNECTOR)
    assert request.py_request_id in transfer_manager.requests_in_transfer()


def test_peer_cancel_preserves_connector_only_leg_until_its_callback():
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        py_request_id=19,
        is_child=False,
        is_context_only_request=True,
        py_kv_transfer_timed_out=False,
    )
    resource_manager = Mock()
    resource_manager.resource_managers = {}
    transfer_manager = AsyncTransferManager(resource_manager, should_store_blocks=False)
    transfer_manager.start_transfer(request, AsyncTransferProvider.CONNECTOR)

    executor = object.__new__(PyExecutor)
    executor.async_transfer_manager = transfer_manager
    executor.active_requests = [request]
    executor.canceled_req_ids = []
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = ([], [])
    executor.kv_cache_transceiver.take_context_cancelled_request_ids.return_value = [
        request.py_request_id
    ]
    executor._disagg_peer_cancelled_ctx_ids = set()
    executor._disagg_timed_out_ctx_cancelled_ids = set()
    executor._check_cache_transfer_errors = Mock()

    PyExecutor._check_disagg_ctx_cache_transfer_status(executor, 0)

    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    executor.kv_cache_transceiver.cancel_request.assert_not_called()
    assert transfer_manager.has_transfer(request, AsyncTransferProvider.CONNECTOR)
    assert request.py_request_id in transfer_manager.requests_in_transfer()
    assert executor._disagg_peer_cancelled_ctx_ids == set()


@pytest.mark.parametrize(
    "connector_final_state,is_child",
    [
        (LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS, False),
        (LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS, True),
        (LlmRequestState.DISAGG_TRANS_ERROR, False),
        (LlmRequestState.DISAGG_TRANS_ERROR, True),
    ],
    ids=[
        "connector-complete-request",
        "connector-complete-child",
        "connector-failed-request",
        "connector-failed-child",
    ],
)
def test_connector_callback_preserves_pending_user_cancel(connector_final_state, is_child):
    request_id = 17
    parent_request_id = 117
    cancel_id = parent_request_id if is_child else request_id
    response = SimpleNamespace(result=SimpleNamespace())
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        py_request_id=request_id,
        parent_request_id=parent_request_id,
        is_child=is_child,
        is_context_only_request=True,
        is_dummy_request=False,
        is_attention_dp_dummy=False,
        py_kv_transfer_timed_out=False,
        py_decoding_iter=2,
        py_draft_tokens=[],
        py_num_accepted_draft_tokens=0,
        py_per_pos_drafted=[],
        py_per_pos_accepted=[],
        return_perf_metrics=False,
        cached_tokens=0,
        is_finished=False,
        is_disagg_context_complete_state=False,
        is_disagg_context_transmission_state=True,
        is_generation_only_request=Mock(return_value=False),
        create_response=Mock(return_value=response),
    )

    def finish_by_reason(reason):
        assert reason == FinishReason.CANCELLED
        request.state = LlmRequestState.GENERATION_COMPLETE
        request.is_finished = True
        request.is_disagg_context_complete_state = False
        request.is_disagg_context_transmission_state = False

    request.finish_by_reason = Mock(side_effect=finish_by_reason)

    resource_manager = Mock()
    resource_manager.resource_managers = {}
    executor = object.__new__(PyExecutor)
    executor.resource_manager = resource_manager
    executor.async_transfer_manager = AsyncTransferManager(
        resource_manager, should_store_blocks=False
    )
    # One transceiver leg and one connector leg own the request.
    executor.async_transfer_manager.start_transfer(request, AsyncTransferProvider.TRANSCEIVER)
    executor.async_transfer_manager.start_transfer(request, AsyncTransferProvider.CONNECTOR)
    executor.active_requests = [request]
    executor.canceled_req_ids = [cancel_id]
    executor.waiting_queue = Mock()
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.owns_request.return_value = True
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.kv_connector_manager = Mock()
    executor.kv_connector_manager.get_finished.return_value = [request]
    executor._disagg_acknowledged_ctx_cancel_legs = set()
    executor._disagg_pp_termination_handler = None
    executor._prefetched_request_ids = set()
    executor._disagg_timed_out_ctx_cancelled_ids = set()
    executor._disagg_timed_out_gen_cancelled_ids = set()
    executor.result_wait_queues = {request_id: Mock()}
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    executor.perf_manager = Mock()
    executor.perf_manager.get_timestamp.return_value = 0
    executor.iter_counter = 1
    executor.stream_interval = 1
    executor.force_terminate_ctx_for_partial_reuse = False
    executor.enable_attention_dp = False
    executor._pending_timed_out_requests = []
    executor._pending_transfer_responses = []
    executor._enqueue_responses = Mock()
    executor._maybe_attach_ctx_usage = Mock()
    executor._is_disagg_inflight_cancel_active = Mock(return_value=True)

    # User cancellation acknowledges only the transceiver leg. The connector
    # still owns the request, so cancellation remains pending.
    PyExecutor._handle_canceled_requests(executor)

    assert executor.canceled_req_ids == [cancel_id]
    assert request_id in executor._disagg_acknowledged_ctx_cancel_legs
    assert request_id in executor.async_transfer_manager.requests_in_transfer()
    request.finish_by_reason.assert_not_called()

    # The real connector completion callback retires only its own leg. It must
    # not create or enqueue a successful context response, even if that leg
    # reports a failure while cancellation is pending.
    request.state = connector_final_state
    PyExecutor._kv_connector_terminate_requests(executor)

    assert request in executor.active_requests
    assert request_id not in executor.async_transfer_manager.requests_in_transfer()
    request.create_response.assert_not_called()
    executor._enqueue_responses.assert_not_called()
    assert executor._pending_transfer_responses == []
    resource_manager.free_resources.assert_not_called()

    if connector_final_state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS:
        # AsyncTransferManager normally publishes CONTEXT_COMPLETE on its last
        # leg. Reproduce the supported-mode race where transceiver state
        # publication leaves the request's visible state stale.
        request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    # The acknowledged transceiver leg plus an empty transfer manager is
    # sufficient proof of quiescence even when inflight mode is active and the
    # request state is stale. The ordinary response path then emits CANCELLED
    # and frees the request exactly once.
    PyExecutor._handle_canceled_requests(executor)

    assert executor.canceled_req_ids == []
    request.finish_by_reason.assert_called_once_with(FinishReason.CANCELLED)
    executor.kv_cache_transceiver.owns_request.assert_called_once_with(request)
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    executor._is_disagg_inflight_cancel_active.assert_not_called()
    assert request_id not in executor._disagg_acknowledged_ctx_cancel_legs
    PyExecutor._handle_responses(executor)

    request.create_response.assert_called_once_with(False, 0)
    executor._enqueue_responses.assert_called_once_with([(request_id, response)])
    assert executor.active_requests == []
    resource_manager.free_resources.assert_called_once_with(request)
    assert request_id not in executor._disagg_acknowledged_ctx_cancel_legs
    assert request_id not in executor.result_wait_queues


@pytest.mark.parametrize("status_mode", ["async", "legacy-default-off"])
def test_peer_context_cancel_retires_transceiver_leg_as_error(status_mode):
    request_id = 27
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
        py_request_id=request_id,
        parent_request_id=127,
        is_child=False,
        is_context_only_request=True,
        is_dummy_request=False,
        py_kv_transfer_timed_out=False,
        create_response=Mock(),
    )
    resource_manager = Mock()
    resource_manager.resource_managers = {}
    executor = object.__new__(PyExecutor)
    executor.resource_manager = resource_manager
    executor.async_transfer_manager = AsyncTransferManager(
        resource_manager, should_store_blocks=False
    )
    executor.async_transfer_manager.start_transfer(request, AsyncTransferProvider.TRANSCEIVER)
    executor.active_requests = [request]
    # This is a peer/global decision, not a locally queued user cancellation.
    executor.canceled_req_ids = []
    executor.kv_cache_transceiver = Mock(name=f"{status_mode}_transceiver")
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = ([], [])
    executor.kv_cache_transceiver.take_context_cancelled_request_ids.return_value = [request_id]
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor._disagg_peer_cancelled_ctx_ids = set()
    executor._disagg_acknowledged_ctx_cancel_legs = set()
    executor._disagg_timed_out_ctx_cancelled_ids = set()
    executor._disagg_timed_out_gen_cancelled_ids = set()
    executor._disagg_pp_termination_handler = None
    executor._prefetched_request_ids = set()
    executor.result_wait_queues = {request_id: Mock()}
    executor.gather_all_responses = False
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(rank=0, world_size=1)
    observed_error_states = []

    def handle_transfer_error(error_msg, requests, charge_budget):
        assert error_msg == "Error in kv cache transfer for context requests"
        assert charge_budget is False
        assert requests == [request]
        observed_error_states.append(request.state)
        executor.active_requests.remove(request)
        PyExecutor._terminate_request(executor, request)

    executor._handle_errors = Mock(side_effect=handle_transfer_error)

    PyExecutor._check_disagg_ctx_cache_transfer_status(executor, 0)

    assert observed_error_states == [LlmRequestState.DISAGG_TRANS_ERROR]
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    request.create_response.assert_not_called()
    assert request_id not in executor.async_transfer_manager.requests_in_transfer()
    assert executor.active_requests == []
    resource_manager.free_resources.assert_called_once_with(request)
    assert executor._disagg_peer_cancelled_ctx_ids == set()
    assert request_id not in executor.result_wait_queues


def test_user_cancel_acknowledges_default_off_transceiver_leg(monkeypatch):
    request = SimpleNamespace(
        state=LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER,
        py_request_id=8,
        is_context_only_request=True,
    )
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.owns_request.return_value = True
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }
    executor.async_transfer_manager.has_transfer.return_value = True
    executor.async_transfer_manager.end_transfer.return_value = True
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled", lambda: False)

    assert PyExecutor._try_cancel_request(executor, request)

    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    executor.async_transfer_manager.end_transfer.assert_called_once_with(
        request, AsyncTransferProvider.TRANSCEIVER
    )
    assert request.py_request_id not in executor._disagg_acknowledged_ctx_cancel_legs


def test_known_cancel_withdraws_before_readiness_progression():
    request = SimpleNamespace(
        py_request_id=9,
        is_child=False,
        is_context_only_request=True,
        py_disaggregated_params=SimpleNamespace(
            schedule_style=executor_module.DisaggScheduleStyle.GENERATION_FIRST
        ),
    )
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.active_requests = [request]
    executor.canceled_req_ids = [request.py_request_id]
    call_order = Mock()
    call_order.attach_mock(
        executor.kv_cache_transceiver.exclude_context_requests_from_readiness,
        "exclude",
    )
    call_order.attach_mock(
        executor.kv_cache_transceiver.prepare_context_requests,
        "prepare",
    )

    PyExecutor._check_disagg_ctx_schedulable_status(executor, [request])

    assert call_order.mock_calls == [call.exclude([request]), call.prepare([])]


def test_shutdown_bounds_transceiver_retries_before_registered_memory_release(
    monkeypatch,
):
    executor = object.__new__(PyExecutor)
    executor.executor_request_queue = Mock()
    executor.shutdown_event = Mock()
    executor.hang_detector = Mock()
    executor.hang_detector.detected.return_value = False
    executor.worker_thread = Mock()
    executor.dist = SimpleNamespace(pp_size=1)
    executor._shutdown_sleep_wakeup_listeners = Mock()
    executor.worker_started = True
    model_engine = Mock()
    executor.model_engine = model_engine
    executor.draft_model_engine = None
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.shutdown.side_effect = RuntimeError("consensus shutdown failed")
    resource_manager = Mock()
    executor.resource_manager = SimpleNamespace(resource_managers={"kv": resource_manager})
    executor.virtual_memory_pools = {0: object()}
    sampler = object.__new__(executor_module.AsyncWorkerMixin)
    sampler._enable_async_worker = True
    sampler.async_worker_stop = Mock()
    executor.sampler = sampler
    dwdp_exit = Mock()
    executor.dwdp_manager = SimpleNamespace(__exit__=dwdp_exit)
    monkeypatch.setattr(executor_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="consensus shutdown failed"):
        PyExecutor.shutdown(executor)

    assert executor.kv_cache_transceiver.shutdown.call_count == 3
    resource_manager.shutdown.assert_not_called()
    model_engine._release_cuda_graphs.assert_called_once_with()
    sampler.async_worker_stop.assert_called_once_with()
    dwdp_exit.assert_called_once_with(None, None, None)
    assert executor.dwdp_manager is None
    assert executor.virtual_memory_pools == {}
    assert not hasattr(executor, "model_engine")


def test_shutdown_retries_transient_transceiver_failure_before_resource_release(
    monkeypatch,
):
    executor = object.__new__(PyExecutor)
    executor.executor_request_queue = Mock()
    executor.shutdown_event = Mock()
    executor.hang_detector = Mock()
    executor.hang_detector.detected.return_value = False
    executor.worker_thread = Mock()
    executor.dist = SimpleNamespace(pp_size=1)
    executor._shutdown_sleep_wakeup_listeners = Mock()
    executor.worker_started = True
    executor.model_engine = Mock()
    executor.draft_model_engine = None
    executor.kv_cache_transceiver = Mock()
    call_order = []

    def first_shutdown_attempt():
        call_order.append("transceiver-1")
        raise RuntimeError("peer consensus shutdown timed out")

    def second_shutdown_attempt():
        call_order.append("transceiver-2")

    shutdown_attempts = iter([first_shutdown_attempt, second_shutdown_attempt])
    executor.kv_cache_transceiver.shutdown.side_effect = lambda: next(shutdown_attempts)()
    resource_manager = Mock()
    resource_manager.shutdown.side_effect = lambda: call_order.append("resource")
    executor.resource_manager = SimpleNamespace(resource_managers={"kv": resource_manager})
    executor.virtual_memory_pools = None
    executor.sampler = object()
    executor.dwdp_manager = None
    monkeypatch.setattr(executor_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="peer consensus shutdown timed out"):
        PyExecutor.shutdown(executor)

    assert call_order == ["transceiver-1", "transceiver-2", "resource"]
    assert executor.kv_cache_transceiver.shutdown.call_count == 2
    resource_manager.shutdown.assert_called_once_with()


def test_shutdown_waits_for_deferred_transfer_teardown_before_resource_release(
    monkeypatch,
):
    executor = object.__new__(PyExecutor)
    executor.executor_request_queue = Mock()
    executor.shutdown_event = Mock()
    executor.hang_detector = Mock()
    executor.hang_detector.detected.return_value = False
    executor.worker_thread = Mock()
    executor.dist = SimpleNamespace(pp_size=1)
    executor._shutdown_sleep_wakeup_listeners = Mock()
    executor.worker_started = True
    executor.model_engine = Mock()
    executor.draft_model_engine = None
    executor.kv_cache_transceiver = Mock()
    completion = threading.Event()
    executor.kv_cache_transceiver.shutdown.side_effect = [completion, None]
    resource_manager = Mock()

    def release_registered_memory() -> None:
        assert completion.is_set()
        assert executor.kv_cache_transceiver.shutdown.call_count == 2

    resource_manager.shutdown.side_effect = release_registered_memory
    executor.resource_manager = SimpleNamespace(resource_managers={"kv": resource_manager})
    executor.virtual_memory_pools = None
    executor.sampler = object()
    executor.dwdp_manager = None
    monkeypatch.setattr(executor_module.torch.cuda, "is_available", lambda: False)

    timer = threading.Timer(0.01, completion.set)
    timer.start()
    try:
        PyExecutor.shutdown(executor)
    finally:
        timer.join()

    assert completion.is_set()
    assert executor.kv_cache_transceiver.shutdown.call_count == 2
    resource_manager.shutdown.assert_called_once_with()


def test_flag_unset_generation_driver_skips_cancel_pipeline():
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = True
    executor._disagg_inflight_cancel_unsupported_logged = False
    executor._check_disagg_gen_cache_transfer_status = Mock()
    executor._cancel_timed_out_gen_transfers = Mock()
    executor._check_gen_cache_transfer_errors_consensus = Mock()

    PyExecutor._check_disagg_gen_transfer_status(executor)

    executor._check_disagg_gen_cache_transfer_status.assert_called_once_with(0)
    executor._cancel_timed_out_gen_transfers.assert_not_called()
    executor._check_gen_cache_transfer_errors_consensus.assert_not_called()


def test_peer_buffer_poison_triggers_world_consistent_fatal_cleanup(monkeypatch):
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = True
    executor.kv_cache_transceiver.has_poisoned_transfer_buffer.return_value = False
    executor._disagg_inflight_cancel_unsupported_logged = False
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(
        world_size=2,
        allreduce=Mock(return_value=1),
    )
    executor._fatal_error = None
    executor.is_shutdown = False
    executor._handle_errors = Mock()
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled", lambda: True)

    PyExecutor._handle_disagg_cache_errors_synced(executor)

    executor.dist.allreduce.assert_called_once_with(0, op=executor_module.ReduceOp.MAX)
    assert isinstance(executor._fatal_error, RuntimeError)
    assert executor.is_shutdown
    executor._handle_errors.assert_called_once_with(
        "Disagg KV cache transfer buffer is poisoned; process restart is required",
        requests=None,
        charge_budget=False,
    )


def test_preclassified_fatal_error_keeps_adp_response_collectives_aligned():
    executor = object.__new__(PyExecutor)
    executor._fatal_error = RuntimeError("already fatal")
    executor._error_budget = Mock()
    executor.is_shutdown = False
    executor.waiting_queue = []
    raw_queue = Mock()
    raw_queue.empty.return_value = True
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_request_queue.return_value = raw_queue
    executor.active_requests = []
    executor.gather_all_responses = False
    executor.enable_attention_dp = True
    executor.dist = SimpleNamespace(rank=1, world_size=2)
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()

    PyExecutor._handle_errors(
        executor, "poisoned transfer buffer", requests=None, charge_budget=False
    )

    executor._error_budget.consume.assert_not_called()
    assert executor.is_shutdown
    assert executor._enqueue_responses.call_args_list == [call([]), call([])]
    executor.executor_request_queue.enqueue_shutdown_request.assert_called_once_with()


@pytest.mark.parametrize(
    "backend,runtime,nixl_backend",
    [
        ("UCX", None, "UCX"),
        ("MPI", None, "UCX"),
        ("MOONCAKE", None, "UCX"),
        ("NIXL", "PYTHON", "UCX"),
        ("NIXL", "CPP", "LIBFABRIC"),
    ],
)
def test_feature_opt_in_rejects_unqualified_config(monkeypatch, backend, runtime, nixl_backend):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    monkeypatch.setenv(transceiver_module._NIXL_KVCACHE_BACKEND_ENV, nixl_backend)
    config = CacheTransceiverConfig(backend=backend, transceiver_runtime=runtime)

    with pytest.raises(ValueError, match="currently supported only"):
        transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)


@pytest.mark.parametrize("nixl_backend", [None, "UCX"])
def test_feature_opt_in_accepts_cpp_nixl_ucx(monkeypatch, nixl_backend):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    if nixl_backend is not None:
        monkeypatch.setenv(transceiver_module._NIXL_KVCACHE_BACKEND_ENV, nixl_backend)
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="CPP")
    expected = object()
    constructor = Mock(return_value=expected)
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver", constructor)

    result = transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert result is expected
    assert config.backend == "NIXL"
    constructor.assert_called_once()


@pytest.mark.parametrize(
    "unsupported_env",
    [
        transceiver_module._DISABLE_KV_CACHE_TRANSFER_OVERLAP_ENV,
        transceiver_module._DISAGG_LAYERWISE_ENV,
        transceiver_module._TRY_ZCOPY_FOR_KV_CACHE_TRANSFER_ENV,
    ],
)
def test_feature_opt_in_rejects_unsupported_transfer_mode(monkeypatch, unsupported_env):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    monkeypatch.setenv(unsupported_env, "1")
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="CPP")

    with pytest.raises(ValueError, match="currently supported only"):
        transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)


def test_feature_opt_in_requires_finite_transfer_timeout(monkeypatch):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    config = CacheTransceiverConfig(
        backend="NIXL", transceiver_runtime="CPP", kv_transfer_timeout_ms=None
    )

    with pytest.raises(ValueError, match="finite kv_transfer_timeout_ms"):
        transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)


def test_feature_opt_in_rejects_default_backend(monkeypatch):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    monkeypatch.setenv(transceiver_module._NIXL_KVCACHE_BACKEND_ENV, "UCX")
    config = CacheTransceiverConfig(backend="DEFAULT")

    with pytest.raises(ValueError, match="backend='DEFAULT'"):
        transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)


def test_feature_opt_in_rejects_ambiguous_legacy_backend_env(monkeypatch):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    monkeypatch.setenv("TRTLLM_USE_NIXL_KVCACHE", "1")
    monkeypatch.setenv("TRTLLM_USE_UCX_KVCACHE", "1")
    config = CacheTransceiverConfig(backend="DEFAULT")

    with pytest.raises(ValueError, match="multiple legacy backend selectors"):
        transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)


def test_feature_opt_in_explicit_backend_ignores_legacy_selectors(monkeypatch):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    monkeypatch.setenv("TRTLLM_USE_NIXL_KVCACHE", "1")
    monkeypatch.setenv("TRTLLM_USE_UCX_KVCACHE", "1")
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="CPP")
    expected = object()
    constructor = Mock(return_value=expected)
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver", constructor)

    result = transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert result is expected
    constructor.assert_called_once()


def test_direct_cpp_wrapper_rejects_python_runtime_opt_in(monkeypatch):
    monkeypatch.setenv(transceiver_module._DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "1")
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")

    with pytest.raises(ValueError, match="currently supported only"):
        BindKvCacheTransceiver(Mock(), Mock(), Mock(), Mock(), config)


def test_flag_unset_preserves_existing_backend_selection(monkeypatch):
    config = CacheTransceiverConfig(backend="UCX")
    expected = object()
    constructor = Mock(return_value=expected)
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver", constructor)

    result = transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert result is expected
    assert config.backend == "UCX"
    constructor.assert_called_once()


def test_flag_unset_preserves_python_transceiver(monkeypatch):
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")
    expected = object()
    constructor = Mock(return_value=expected)
    fake_module = SimpleNamespace(KvCacheTransceiverV2=constructor)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.disaggregation.transceiver", fake_module)

    result = transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert result is expected
    constructor.assert_called_once()


def test_flag_unset_preserves_libfabric_selection(monkeypatch):
    monkeypatch.setenv(transceiver_module._NIXL_KVCACHE_BACKEND_ENV, "LIBFABRIC")
    config = CacheTransceiverConfig(backend="NIXL")
    expected = object()
    constructor = Mock(return_value=expected)
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver", constructor)

    result = transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert result is expected
    constructor.assert_called_once()


@pytest.mark.parametrize(
    "selector,expected_backend",
    [
        ("TRTLLM_USE_NIXL_KVCACHE", "NIXL"),
        ("TRTLLM_USE_UCX_KVCACHE", "UCX"),
        ("TRTLLM_USE_MOONCAKE_KVCACHE", "MOONCAKE"),
        ("TRTLLM_USE_MPI_KVCACHE", "MPI"),
    ],
)
def test_flag_unset_preserves_legacy_backend_env(monkeypatch, selector, expected_backend):
    monkeypatch.setenv(selector, "1")
    config = CacheTransceiverConfig(backend="DEFAULT")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver", constructor)

    transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert config.backend == expected_backend
    constructor.assert_called_once()


def test_flag_unset_preserves_legacy_backend_env_precedence(monkeypatch):
    for selector in (
        "TRTLLM_USE_MPI_KVCACHE",
        "TRTLLM_USE_MOONCAKE_KVCACHE",
        "TRTLLM_USE_UCX_KVCACHE",
        "TRTLLM_USE_NIXL_KVCACHE",
    ):
        monkeypatch.setenv(selector, "1")
    config = CacheTransceiverConfig(backend="DEFAULT")
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver", Mock())

    transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)

    assert config.backend == "NIXL"


@pytest.mark.parametrize(
    "backend,runtime,nixl_backend,expected",
    [
        ("NIXL", "CPP", None, True),
        ("NIXL", None, "UCX", True),
        ("NIXL", "CPP", "UCX", True),
        ("NIXL", "PYTHON", "UCX", False),
        ("NIXL", "CPP", "LIBFABRIC", False),
        ("UCX", "CPP", "UCX", False),
    ],
)
def test_cpp_capability_is_config_scoped(monkeypatch, backend, runtime, nixl_backend, expected):
    if nixl_backend is not None:
        monkeypatch.setenv(transceiver_module._NIXL_KVCACHE_BACKEND_ENV, nixl_backend)
    config = CacheTransceiverConfig(backend=backend, transceiver_runtime=runtime)

    assert transceiver_module._is_disagg_inflight_cancel_config_supported(config) is expected

    monkeypatch.setattr(transceiver_module, "mapping_to_world_config", lambda mapping: object())
    constructor = Mock(return_value=Mock())
    monkeypatch.setattr(transceiver_module, "CacheTransceiverCpp", constructor)
    monkeypatch.setattr(CacheTransceiverConfig, "_to_pybind", lambda config: object())
    dist = Mock()
    dist.pp_allgather.return_value = [1]
    kv_cache_manager = SimpleNamespace(
        total_num_kv_heads_per_layer=[1],
        head_dim=64,
        tokens_per_block=32,
        dtype=object(),
        num_kv_heads_per_layer=[1],
        impl=object(),
    )

    transceiver = BindKvCacheTransceiver(Mock(), dist, kv_cache_manager, Mock(), config)

    assert transceiver.supports_inflight_request_cancellation() is expected
    constructor.assert_called_once()


def test_python_transceiver_capability_defaults_to_unsupported():
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = object.__new__(KvCacheTransceiverV2)

    assert not transceiver.supports_inflight_request_cancellation()
    assert not transceiver.has_poisoned_transfer_buffer()


def test_flag_unset_skips_cpp_poison_query():
    transceiver = SimpleNamespace(impl=Mock())

    assert not BindKvCacheTransceiver.has_poisoned_transfer_buffer(transceiver)
    transceiver.impl.has_poisoned_transfer_buffer.assert_not_called()
