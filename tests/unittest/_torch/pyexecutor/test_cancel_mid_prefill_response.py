# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancelling a context-only request mid-prefill under a KV-moving connector.

The cancellation hands the partly filled pages to the connector, whose save
moves the request into a context transmission state. ``LlmRequest::createResult``
reads ``ContextPhaseParams`` in that state, and a request cancelled before its
KV transfer was set up has none. The cancel-path guards must turn that into a
CANCELLED response (or, failing that, an error response for that request
alone) while the pages stay pinned until the save ends.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorManager
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.perf_metrics_manager import PerfMetricsManager
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings.executor import ContextPhaseParams, FinishReason
from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

pytestmark = pytest.mark.cpu_only

KILL_SWITCHES = ("TRTLLM_DISABLE_CANCEL_CTX_PARAMS", "TRTLLM_DISABLE_RESPONSE_GUARD")
PROMPT_LEN = 64
CHUNK = 16
BLOCK_ID = 7


@pytest.fixture(autouse=True)
def _clear_knobs(monkeypatch):
    for name in KILL_SWITCHES:
        monkeypatch.delenv(name, raising=False)


def _ctx_only_request(request_id=11, prompt_len=PROMPT_LEN, done=CHUNK):
    """A real context-only request that has prefilled one of several chunks."""
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    request.context_current_position = done
    assert request.state == LlmRequestState.CONTEXT_INIT
    assert request.context_phase_params is None
    return request


def _executor(*, ctx_params_guard=True, response_guard=True):
    """A minimal context executor whose connector saves every finished request."""
    executor = object.__new__(PyExecutor)
    executor._cancel_ctx_params_guard = ctx_params_guard
    executor._response_guard = response_guard
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(
        rank=0,
        world_size=1,
        tp_size=1,
        pp_size=1,
        mapping=SimpleNamespace(tp_rank=0, enable_attention_dp=False),
    )
    connector = Mock(spec=KvCacheConnectorManager)
    connector.capacity_only = False
    connector.defer_load_termination.return_value = False
    connector.has_outstanding_save.return_value = False
    connector.request_finished.return_value = True
    executor.kv_connector_manager = connector
    kv_cache_manager = Mock()
    kv_cache_manager.get_cache_indices.return_value = [1, 2, 3]
    kv_cache_manager.store_blocks_for_reuse.return_value = BLOCK_ID
    kv_cache_manager.is_preemption_pending.return_value = False
    executor.kv_cache_manager = kv_cache_manager
    executor._is_kv_manager_v2 = True
    executor.kv_cache_transceiver = None
    resource_manager = SimpleNamespace(
        resource_managers={
            ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager,
            ResourceManagerType.SEQ_SLOT_MANAGER: Mock(),
        }
    )
    executor.async_transfer_manager = AsyncTransferManager(resource_manager)
    executor.waiting_queue = Mock()
    executor.canceled_req_ids = []
    executor.active_requests = []
    executor._is_preemption_pending = lambda request: False
    # _handle_responses
    executor.perf_manager = PerfMetricsManager(enabled=False)
    executor.iter_counter = 0
    executor.stream_interval = 1
    executor.first_content_emit_iters = 0
    executor.disable_overlap_scheduler = False
    executor.force_terminate_ctx_for_partial_reuse = False
    executor._disagg_coordinator = SimpleNamespace(
        fail_timed_out=lambda requests: None,
        inflight_cancel_active=lambda: False,
        request_cancellation=lambda request: False,
    )
    executor._maybe_attach_ctx_usage = lambda request, response: None
    executor.enqueued = []
    executor._enqueue_responses = lambda responses: executor.enqueued.extend(responses)
    executor._disagg_pp_termination_handler = None
    executor._do_terminate_request = Mock()
    return executor


def _cancel(executor, request):
    executor.active_requests = [request]
    executor.canceled_req_ids = [request.py_request_id]
    executor._handle_canceled_requests()
    assert executor.canceled_req_ids == []
    assert request.is_finished
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    assert request.py_request_id in executor.async_transfer_manager.requests_in_transfer()


def _assert_pages_held_until_save_ends(executor, request, returned):
    # Dropped from the active set without a free: the save still reads the pages.
    assert executor.active_requests == []
    assert request not in returned
    executor._do_terminate_request.assert_not_called()
    executor.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    # The connector's save ends; only now are the pages unpinned.
    assert executor.async_transfer_manager.end_transfer(request) is True
    executor.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(BLOCK_ID)
    assert request.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE


def test_cancel_mid_prefill_emits_a_cancelled_response():
    executor = _executor()
    request = _ctx_only_request()

    _cancel(executor, request)
    assert request.context_phase_params is not None
    assert executor._cancel_ctx_params_attached == 1

    returned = executor._handle_responses()

    assert [req_id for req_id, _ in executor.enqueued] == [request.py_request_id]
    response = executor.enqueued[0][1]
    assert not response.has_error()
    result = response.result.get_result()
    assert result.is_final
    assert list(result.finish_reasons) == [FinishReason.CANCELLED]
    assert executor._response_guard_failures == 0
    _assert_pages_held_until_save_ends(executor, request, returned)


def test_response_guard_alone_fails_only_the_request():
    """Without placeholder params the old library raises; only the request fails."""
    executor = _executor(ctx_params_guard=False)
    request = _ctx_only_request()

    _cancel(executor, request)
    assert request.context_phase_params is None

    returned = executor._handle_responses()

    assert [req_id for req_id, _ in executor.enqueued] == [request.py_request_id]
    response = executor.enqueued[0][1]
    if response.has_error():
        # Library without the createResult guard: bad optional access, caught.
        assert "optional" in response.error_msg
        assert executor._response_guard_failures == 1
    else:
        # Library with the createResult guard: a plain CANCELLED result.
        assert list(response.result.get_result().finish_reasons) == [FinishReason.CANCELLED]
    _assert_pages_held_until_save_ends(executor, request, returned)


def test_a_deterministic_response_failure_fails_only_that_request():
    """The Python guard turns one request's response error into an error response for it alone."""
    executor = _executor()
    request = _ctx_only_request()
    _cancel(executor, request)
    other = _ctx_only_request(request_id=12)
    executor.active_requests = [request, other]
    create = executor._create_request_response

    def _failing(req, *args, **kwargs):
        if req is request:
            raise RuntimeError("bad optional access")
        return create(req, *args, **kwargs)

    executor._create_request_response = _failing

    returned = executor._handle_responses()

    assert executor._response_guard_failures == 1
    assert [req_id for req_id, _ in executor.enqueued] == [request.py_request_id]
    response = executor.enqueued[0][1]
    assert response.has_error() and "bad optional access" in response.error_msg
    assert other in executor.active_requests
    assert request not in executor.active_requests and request not in returned
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    executor._do_terminate_request.assert_not_called()
    executor.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    assert executor.async_transfer_manager.end_transfer(request) is True
    executor.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(BLOCK_ID)


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("CUDA out of memory"),
        RuntimeError("CUDA error: an illegal memory access was encountered"),
    ],
    ids=["oom", "cuda_error"],
)
def test_device_errors_are_not_guarded(error):
    """A device fault or OOM is not one request's problem: it still ends the loop."""
    executor = _executor()
    request = _ctx_only_request()
    _cancel(executor, request)

    def _raise(*args, **kwargs):
        raise error

    executor._create_request_response = _raise
    with pytest.raises(type(error)):
        executor._handle_responses()
    assert executor.enqueued == []
    assert executor._response_guard_failures == 0


@pytest.mark.parametrize(
    "tp_size, pp_size, attention_dp, guarded",
    [(2, 1, False, False), (2, 1, True, True), (1, 2, False, False)],
    ids=["replicated_tp", "attention_dp", "pipeline_parallel"],
)
def test_response_guard_needs_a_rank_owned_request(tp_size, pp_size, attention_dp, guarded):
    """A request every rank holds is not retired on one rank: the error still ends the loop."""
    executor = _executor()
    executor.dist.tp_size, executor.dist.pp_size = tp_size, pp_size
    executor.enable_attention_dp = attention_dp
    request = _ctx_only_request()
    _cancel(executor, request)

    def _raise(*args, **kwargs):
        raise RuntimeError("response creation failed")

    executor._create_request_response = _raise
    if guarded:
        executor._handle_responses()
        assert executor._response_guard_failures == 1
    else:
        with pytest.raises(RuntimeError, match="response creation failed"):
            executor._handle_responses()
        assert executor._response_guard_failures == 0


def test_a_failed_response_keeps_a_transmission_state():
    executor = _executor()
    request = _ctx_only_request()
    request.state = LlmRequestState.DISAGG_CONTEXT_INIT_AND_TRANS

    response = executor._guarded_response_error(request, RuntimeError("bad optional access"))

    assert response.has_error()
    assert request.state == LlmRequestState.DISAGG_CONTEXT_INIT_AND_TRANS


def test_a_finished_request_keeps_its_own_params():
    executor = _executor()
    request = _ctx_only_request()
    request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    executor._attach_cancel_ctx_phase_params(request)
    assert executor._cancel_ctx_params_attached == 1
    # The binding returns a copy, so mark the request's own params with values the placeholder
    # never uses and check that they survive a second attach.
    own_req_id = request.py_request_id + 1000
    request.context_phase_params = ContextPhaseParams(
        first_gen_tokens=[3],
        req_id=own_req_id,
        opaque_state=None,
        draft_tokens=None,
        ctx_dp_rank=0,
        disagg_info_endpoint="own-endpoint",
    )
    executor._attach_cancel_ctx_phase_params(request)
    assert executor._cancel_ctx_params_attached == 1
    kept = request.context_phase_params
    assert kept.req_id == own_req_id
    assert kept.disagg_info_endpoint == "own-endpoint"
    assert list(kept.first_gen_tokens) == [3]


@pytest.mark.parametrize("switch", (None,) + KILL_SWITCHES)
def test_each_kill_switch_disables_only_its_guard(monkeypatch, switch):
    if switch is not None:
        monkeypatch.setenv(switch, "1")
    executor = object.__new__(PyExecutor)
    executor.dist = SimpleNamespace(rank=0)
    executor._init_cancel_path_guards()
    assert executor._cancel_ctx_params_guard is (switch != KILL_SWITCHES[0])
    assert executor._response_guard is (switch != KILL_SWITCHES[1])


def test_a_follow_up_request_is_unaffected_once_the_cancelled_save_ends():
    executor = _executor()
    request = _ctx_only_request()
    _cancel(executor, request)
    returned = executor._handle_responses()
    _assert_pages_held_until_save_ends(executor, request, returned)
    assert executor.async_transfer_manager.requests_in_transfer() == {}

    follow_up = _ctx_only_request(request_id=12)
    executor.active_requests = [follow_up]
    executor.enqueued.clear()
    executor._handle_canceled_requests()  # nothing pending: the follow-up is left alone
    assert not follow_up.is_finished
    assert follow_up.state == LlmRequestState.CONTEXT_INIT
    assert follow_up.context_phase_params is None
    assert executor.active_requests == [follow_up]
    assert executor._cancel_ctx_params_attached == 1
    executor.kv_connector_manager.request_finished.assert_called_once()
    assert executor.enqueued == []
