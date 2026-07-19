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
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from tensorrt_llm._torch.pyexecutor import kv_cache_transceiver as transceiver_module
from tensorrt_llm._torch.pyexecutor import py_executor as executor_module
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import BindKvCacheTransceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
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
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = ([], [])
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
    executor._end_transfer_and_maybe_terminate.assert_called_once_with(request)
    assert request.py_request_id not in executor._disagg_timed_out_ctx_cancelled_ids


def test_enabled_context_timeout_defers_cleanup_until_cpp_terminal_state(monkeypatch):
    request = _make_timeout_request()
    request.py_kv_transfer_start_time = 1.0
    request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = ([], [])
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

    executor.async_transfer_manager.end_transfer.assert_called_once_with(request)
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
    executor.waiting_queue.remove_by_ids.return_value = []
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.cancel_request.return_value = True
    executor.kv_cache_transceiver.supports_inflight_request_cancellation.return_value = True
    executor._disagg_inflight_cancel_unsupported_logged = False
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }
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


def test_waiting_cancellation_emits_terminal_response_before_cleanup(monkeypatch):
    request_item = SimpleNamespace(id=7)
    response = object()
    request = SimpleNamespace(
        py_request_id=7,
        is_child=False,
        py_decoding_iter=0,
        finish_by_reason=Mock(),
        create_response=Mock(return_value=response),
    )
    merge = Mock(return_value=[request])
    monkeypatch.setattr(executor_module, "merge_requests", merge)

    executor = object.__new__(PyExecutor)
    executor.enable_iter_perf_stats = True
    executor.num_fetch_requests = 0
    executor.enable_attention_dp = False
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0, world_size=1, cp_config={}, cp_rank=0, cp_size=1)
    executor.executor_request_queue = Mock()
    executor._should_exclude_last_generation_logits = Mock(return_value=False)
    executor._enqueue_responses = Mock()
    executor.result_wait_queues = Mock()
    order = Mock()
    order.attach_mock(executor._enqueue_responses, "respond")
    order.attach_mock(executor.result_wait_queues.pop, "cleanup")

    PyExecutor._terminalize_canceled_waiting_requests(executor, [request_item])

    request.finish_by_reason.assert_called_once_with(executor_module.FinishReason.CANCELLED)
    request.create_response.assert_called_once_with(False, 0)
    executor.executor_request_queue.calculate_queue_latency.assert_called_once()
    executor._enqueue_responses.assert_called_once_with([(7, response)])
    assert executor.num_fetch_requests == 1
    assert order.mock_calls == [call.respond([(7, response)]), call.cleanup(7, None)]


@pytest.mark.parametrize("gather_all_responses", [False, True])
def test_waiting_cancellation_enters_adp_response_collective_on_nonleader(
    monkeypatch, gather_all_responses
):
    merge = Mock()
    monkeypatch.setattr(executor_module, "merge_requests", merge)

    executor = object.__new__(PyExecutor)
    executor.enable_iter_perf_stats = False
    executor.num_fetch_requests = 0
    executor.enable_attention_dp = True
    executor.gather_all_responses = gather_all_responses
    executor.dist = SimpleNamespace(rank=1, world_size=2, cp_config={}, cp_rank=0, cp_size=1)
    executor._enqueue_responses = Mock()
    waiter = object()
    executor.result_wait_queues = {7: waiter}

    PyExecutor._terminalize_canceled_waiting_requests(executor, [SimpleNamespace(id=7)])

    merge.assert_not_called()
    executor._enqueue_responses.assert_called_once_with([])
    if gather_all_responses:
        assert 7 not in executor.result_wait_queues
    else:
        assert executor.result_wait_queues[7] is waiter


def test_handle_canceled_requests_terminalizes_removed_waiting_items():
    request_item = SimpleNamespace(id=7)
    executor = object.__new__(PyExecutor)
    executor.canceled_req_ids = [7]
    executor.active_requests = []
    executor.waiting_queue = Mock()
    executor.waiting_queue.remove_by_ids.return_value = [request_item]
    executor._terminalize_canceled_waiting_requests = Mock()

    PyExecutor._handle_canceled_requests(executor)

    executor._terminalize_canceled_waiting_requests.assert_called_once_with([request_item])
    assert executor.canceled_req_ids == []


def test_handle_canceled_requests_retains_marker_for_control_deferred_item():
    executor = object.__new__(PyExecutor)
    executor.canceled_req_ids = [7]
    executor.control_requests = [object()]
    executor.active_requests = []
    executor.waiting_queue = Mock()
    executor.waiting_queue.remove_by_ids.return_value = []
    executor._terminalize_canceled_waiting_requests = Mock()

    PyExecutor._handle_canceled_requests(executor)

    assert executor.canceled_req_ids == [7]


def test_fetch_terminalizes_waiting_cancellation_before_admission():
    cancel_marker = object()
    canceled_item = SimpleNamespace(id=7)
    executor = object.__new__(PyExecutor)
    executor.control_requests = []
    executor.canceled_req_ids = []
    executor.num_fetch_requests = 0
    executor._disable_mpi = False
    executor.dist = SimpleNamespace(rank=0, tp_size=1, has_pp=False, cp_size=1)
    executor.request_accumulated = []
    executor.hang_detector = SimpleNamespace(pause=lambda: nullcontext())
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_from_request_queue.return_value = [cancel_marker]
    executor.request_broadcaster = Mock()
    executor.request_broadcaster.broadcast.return_value = ([cancel_marker], [])

    def handle_special_items(_items):
        executor.canceled_req_ids.append(7)
        return []

    executor._handle_special_queue_items = Mock(side_effect=handle_special_items)
    executor._terminalize_canceled_waiting_requests = Mock()
    waiting_queue = MagicMock()
    waiting_queue.__len__.return_value = 1
    waiting_queue.remove_by_ids.return_value = [canceled_item]

    order = Mock()
    order.attach_mock(waiting_queue.add_requests, "enqueue")
    order.attach_mock(waiting_queue.remove_by_ids, "remove")
    order.attach_mock(executor._terminalize_canceled_waiting_requests, "terminalize")

    PyExecutor._fetch_and_enqueue_requests(executor, waiting_queue, 0)

    assert order.mock_calls == [
        call.enqueue([]),
        call.remove({7}),
        call.terminalize([canceled_item]),
    ]
    assert executor.canceled_req_ids == []


def test_fetch_processes_cancellation_while_control_waits_for_drain():
    cancel_marker = SimpleNamespace(
        id=7,
        is_shutdown_request=False,
        is_canceled_request=True,
    )
    deferred_request = SimpleNamespace(
        id=8,
        is_shutdown_request=False,
        is_canceled_request=False,
    )
    canceled_item = SimpleNamespace(id=7)
    executor = object.__new__(PyExecutor)
    executor.control_requests = [object()]
    executor.canceled_req_ids = []
    executor._disable_mpi = False
    executor.dist = SimpleNamespace(rank=0)
    executor.request_accumulated = []
    executor.hang_detector = SimpleNamespace(pause=lambda: nullcontext())
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_from_request_queue.return_value = [
        deferred_request,
        cancel_marker,
    ]
    executor.request_broadcaster = Mock()
    executor.request_broadcaster.broadcast.return_value = (
        [deferred_request, cancel_marker],
        [],
    )
    executor._terminalize_canceled_waiting_requests = Mock()
    executor.is_shutdown = False
    waiting_queue = MagicMock()
    waiting_queue.remove_by_ids.return_value = [canceled_item]

    PyExecutor._fetch_and_enqueue_requests(executor, waiting_queue, 1)

    assert executor.request_accumulated == [deferred_request]
    assert executor.canceled_req_ids == []
    waiting_queue.add_requests.assert_not_called()
    waiting_queue.remove_by_ids.assert_called_once_with({7})
    executor._terminalize_canceled_waiting_requests.assert_called_once_with([canceled_item])


def test_fetch_processes_same_batch_cancellation_behind_control():
    target_item = SimpleNamespace(
        id=7,
        is_shutdown_request=False,
        is_canceled_request=False,
        is_control_request=False,
    )
    control_marker = SimpleNamespace(
        id=8,
        is_shutdown_request=False,
        is_canceled_request=False,
        is_control_request=True,
    )
    cancel_marker = SimpleNamespace(
        id=7,
        is_shutdown_request=False,
        is_canceled_request=True,
        is_control_request=False,
    )
    executor = object.__new__(PyExecutor)
    executor.control_requests = []
    executor.canceled_req_ids = []
    executor._disable_mpi = False
    executor.dist = SimpleNamespace(rank=0, tp_size=1, has_pp=False, cp_size=1)
    executor.request_accumulated = []
    executor.hang_detector = SimpleNamespace(pause=lambda: nullcontext())
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_from_request_queue.return_value = [
        target_item,
        control_marker,
        cancel_marker,
    ]
    executor.request_broadcaster = Mock()
    executor.request_broadcaster.broadcast.return_value = (
        [target_item, control_marker, cancel_marker],
        [],
    )
    executor._terminalize_canceled_waiting_requests = Mock()
    executor.is_shutdown = False
    waiting_queue = MagicMock()
    waiting_queue.__len__.return_value = 0
    waiting_queue.remove_by_ids.return_value = [target_item]

    PyExecutor._fetch_and_enqueue_requests(executor, waiting_queue, 0)

    assert executor.control_requests == [control_marker]
    assert executor.request_accumulated == []
    assert executor.canceled_req_ids == []
    waiting_queue.add_requests.assert_called_once_with([target_item])
    waiting_queue.remove_by_ids.assert_called_once_with({7})
    executor._terminalize_canceled_waiting_requests.assert_called_once_with([target_item])


def test_fetch_defers_shutdown_until_pending_control_completes():
    shutdown_marker = SimpleNamespace(
        id=9,
        is_shutdown_request=True,
        is_canceled_request=False,
        is_control_request=False,
    )
    executor = object.__new__(PyExecutor)
    executor.control_requests = [object()]
    executor.canceled_req_ids = []
    executor._disable_mpi = False
    executor.dist = SimpleNamespace(rank=0, tp_size=1, has_pp=False, cp_size=1)
    executor.request_accumulated = []
    executor.hang_detector = SimpleNamespace(pause=lambda: nullcontext())
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_from_request_queue.side_effect = [
        [shutdown_marker],
        [],
    ]
    executor.request_broadcaster = Mock()
    executor.request_broadcaster.broadcast.side_effect = lambda requests: (requests, [])
    executor._terminalize_canceled_waiting_requests = Mock()
    executor.is_shutdown = False
    waiting_queue = MagicMock()
    waiting_queue.__len__.return_value = 0

    PyExecutor._fetch_and_enqueue_requests(executor, waiting_queue, 0)

    assert not executor.is_shutdown
    assert executor.request_accumulated == [shutdown_marker]

    executor.control_requests.clear()
    PyExecutor._fetch_and_enqueue_requests(executor, waiting_queue, 0)

    assert executor.is_shutdown
    assert executor.request_accumulated == []


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
