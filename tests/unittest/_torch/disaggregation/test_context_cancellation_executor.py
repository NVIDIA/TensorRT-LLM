# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CTX cancellation routing and logical completion, independent of I/O retirement.

The executor, queue, native request, and transfer claim manager are real. The
transceiver boundary controls acceptance; native I/O settlement is exercised in
test_context_cancellation_retirement.py rather than simulated here.
"""

import pickle
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import FinishReason, RequestType
from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor.base_worker import BaseWorker

pytestmark = pytest.mark.cpu_only


def _request(request_id: int = 11) -> LlmRequest:
    """Create a model-idle native context request with a distinct client identity."""
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(8)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        draft_tokens=None,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    request.py_client_id = 101
    request.py_disaggregated_params = DisaggregatedParams(
        disagg_request_id=request_id,
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
    )
    request.state = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
    return request


@pytest.fixture
def executor() -> PyExecutor:
    """Keep lifecycle methods real while replacing model, transport, and output effects."""
    engine = object.__new__(PyExecutor)
    engine.active_requests = []
    engine.canceled_req_ids = []
    engine.waiting_queue = FCFSWaitingQueue()
    engine._pending_transfer_responses = []
    engine._pending_response_terminations = []
    engine.enable_attention_dp = False
    engine.kv_connector_manager = None
    engine.previous_batch = None
    engine.inflight_req_ids = set()
    engine._disagg_pp_termination_handler = None
    engine.dist = SimpleNamespace(rank=0, tp_size=1)
    engine.executor_request_queue = ExecutorRequestQueue(
        dist=engine.dist,
        max_batch_size=8,
        enable_iter_perf_stats=False,
        batch_wait_timeout_ms=0.0,
    )
    cache = Mock(spec=["store_blocks_for_reuse", "unpin_blocks_by_id"])
    cache.store_blocks_for_reuse.return_value = 91
    engine.async_transfer_manager = AsyncTransferManager(
        SimpleNamespace(resource_managers={ResourceManagerType.KV_CACHE_MANAGER: cache})
    )
    engine.kv_cache_transceiver = Mock()
    engine._disagg_coordinator = Mock(
        spec=["supports_logical_context_cancellation", "request_logical_context_cancellation"]
    )
    engine.disagg.supports_logical_context_cancellation.return_value = True
    engine.disagg.request_logical_context_cancellation.return_value = True
    engine._enqueue_responses = Mock()
    engine._do_terminate_request = Mock()
    return engine


def _consume_abort(engine: PyExecutor, client_id: int, request_id: int | None) -> None:
    """Deliver the real worker abort through the executor control queue."""
    mapping = {} if request_id is None else {client_id: request_id}
    worker = SimpleNamespace(engine=engine, _client_id_to_request_id=mapping)
    BaseWorker.abort_request(worker, client_id)
    item = engine.executor_request_queue.get_request_queue().get_nowait()
    # The distributed request envelope carries the whole dataclass, including
    # its explicit namespace; a remote rank must not reinterpret the client ID.
    assert engine._handle_special_queue_items([pickle.loads(pickle.dumps(item))]) == []


@pytest.mark.parametrize("has_transfer", [False, True])
def test_context_cancellation_result_precedes_physical_retirement(
    executor: PyExecutor,
    has_transfer: bool,
) -> None:
    """The native cancelled result is available even while the real claim manager pins KV."""
    request = _request()
    executor.active_requests.append(request)
    if has_transfer:
        executor.async_transfer_manager.start_transfer(request)
    _consume_abort(executor, request.py_client_id, request.py_request_id)

    assert executor._progress_context_cancellations([request]) == []

    assert request.state == LlmRequestState.GENERATION_COMPLETE
    assert not executor.active_requests and not executor.canceled_req_ids
    executor._enqueue_responses.assert_called_once()
    responses = executor._enqueue_responses.call_args.args[0]
    assert len(responses) == 1 and responses[0][0] == request.py_request_id
    assert responses[0][1].result.finish_reasons == [FinishReason.CANCELLED]
    if has_transfer:
        assert executor.async_transfer_manager.requests_in_transfer() == {11: request}
        executor.async_transfer_manager.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
        executor._do_terminate_request.assert_not_called()
    else:
        executor._do_terminate_request.assert_called_once_with(request)


def test_late_abort_resolves_retained_client_not_colliding_engine_id(executor: PyExecutor) -> None:
    """Frontend mapping removal does not lose CTX cleanup or cancel an unrelated request."""
    retired_result = _request()
    unrelated = _request(101)
    unrelated.py_client_id = 909
    executor.async_transfer_manager.start_transfer(retired_result)
    executor.active_requests.append(unrelated)

    _consume_abort(executor, 101, None)
    assert executor.canceled_req_ids == [11]
    assert executor._progress_context_cancellations([unrelated]) == [unrelated]

    executor.disagg.request_logical_context_cancellation.assert_called_once_with(retired_result)
    assert executor.active_requests == [unrelated]
    assert not executor.canceled_req_ids
    executor._enqueue_responses.assert_not_called()
    executor._do_terminate_request.assert_not_called()
    assert executor.async_transfer_manager.requests_in_transfer() == {11: retired_result}


def test_unknown_late_abort_is_noop(executor: PyExecutor) -> None:
    """An expired client identity cannot be treated as an executor request ID."""
    _consume_abort(executor, 101, None)
    assert not executor.canceled_req_ids
    executor.disagg.request_logical_context_cancellation.assert_not_called()


def test_late_abort_namespace_is_not_exported_as_pending_engine_id(executor: PyExecutor) -> None:
    """Connector admission probes only see actual executor cancellation identities."""
    executor._cancel_context_transfer(101)
    executor.cancel_request(11)
    assert executor.executor_request_queue.pending_cancellation_ids() == {11}


def test_previous_model_batch_defers_cancellation_until_post_forward(executor: PyExecutor) -> None:
    """The logical-result path cannot free resources owned by an outstanding model batch."""
    request = _request()
    executor.active_requests.append(request)
    executor.previous_batch = SimpleNamespace(
        scheduled_requests=SimpleNamespace(all_requests=lambda: [request])
    )
    executor.canceled_req_ids = [11]

    assert executor._progress_context_cancellations([request]) == [request]
    executor.disagg.request_logical_context_cancellation.assert_not_called()
    executor._do_terminate_request.assert_not_called()
    assert executor.canceled_req_ids == [11]

    executor._handle_canceled_requests()
    assert request.state == LlmRequestState.GENERATION_COMPLETE
    executor.disagg.request_logical_context_cancellation.assert_called_once_with(request)


@pytest.mark.parametrize("at_post_forward", [False, True])
def test_committed_failure_is_not_rewritten_as_cancellation(
    executor: PyExecutor,
    at_post_forward: bool,
) -> None:
    """A previously committed request failure remains the logical outcome."""
    request = _request()
    executor.active_requests.append(request)
    executor.canceled_req_ids = [11]

    request.state = LlmRequestState.DISAGG_TRANS_ERROR
    if at_post_forward:
        executor._handle_canceled_requests()
    else:
        assert executor._progress_context_cancellations([request]) == [request]
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert not executor.canceled_req_ids
    executor._enqueue_responses.assert_not_called()
    executor._do_terminate_request.assert_not_called()


@pytest.mark.parametrize("disabled", ["profile", "connector"])
def test_unqualified_paths_keep_existing_cancellation_behavior(
    executor: PyExecutor,
    disabled: str,
) -> None:
    """This increment does not activate other profiles or connector ownership paths."""
    request = _request()
    executor.active_requests.append(request)
    executor.canceled_req_ids = [11]
    if disabled == "profile":
        executor.disagg.supports_logical_context_cancellation.return_value = False
    else:
        executor.kv_connector_manager = object()
    assert executor._progress_context_cancellations([request]) == [request]
    executor.disagg.request_logical_context_cancellation.assert_not_called()
    executor._enqueue_responses.assert_not_called()


def test_queued_context_cancel_bypasses_capacity_without_selecting_other_work(
    executor: PyExecutor,
) -> None:
    """Only valid context-first cancelled items bypass execution-slot admission."""
    items = []
    for rid, kind, style in (
        (11, RequestType.REQUEST_TYPE_CONTEXT_ONLY, DisaggScheduleStyle.CONTEXT_FIRST),
        (12, RequestType.REQUEST_TYPE_CONTEXT_ONLY, DisaggScheduleStyle.GENERATION_FIRST),
        (13, RequestType.REQUEST_TYPE_GENERATION_ONLY, DisaggScheduleStyle.CONTEXT_FIRST),
        (14, RequestType.REQUEST_TYPE_CONTEXT_ONLY, DisaggScheduleStyle.CONTEXT_FIRST),
    ):
        raw = SimpleNamespace(
            request_type=kind,
            priority=0,
            py_disaggregated_params=DisaggregatedParams(
                disagg_request_id=rid, schedule_style=style
            ),
        )
        items.append(RequestQueueItem(rid, raw))
    executor.waiting_queue.add_requests(items)
    executor.canceled_req_ids = [11, 12, 13]
    selected = executor._take_cancelled_context_requests(executor.waiting_queue)
    assert selected == [items[0]]
    assert list(executor.waiting_queue) == items[1:]
    # The selected item is converted normally, but cancelled before the model
    # scheduler is entered. No capacity reservation is involved in extraction.
    request = _request(selected[0].id)
    executor.active_requests.append(request)
    assert executor._progress_context_cancellations([request]) == []
    executor._enqueue_responses.assert_called_once()


def test_duplicate_cancellation_emits_one_terminal_result(executor: PyExecutor) -> None:
    """Repeated notifications join the same cancellation instead of replaying a result."""
    request = _request()
    executor.active_requests.append(request)
    executor.async_transfer_manager.start_transfer(request)
    executor.canceled_req_ids = [11, 11]
    executor._progress_context_cancellations([request])
    executor.canceled_req_ids = [11]
    executor._progress_context_cancellations([])
    executor._enqueue_responses.assert_called_once()
    executor._do_terminate_request.assert_not_called()


@pytest.mark.parametrize("available_slots", [0, 1])
def test_fetch_preserves_admission_result_when_extracting_cancelled_context(
    executor: PyExecutor, monkeypatch: pytest.MonkeyPatch, available_slots: int
) -> None:
    """Cancelled CTX items bypass capacity without altering normal admission."""
    cancelled, pending = [
        RequestQueueItem(
            rid,
            SimpleNamespace(
                request_type=RequestType.REQUEST_TYPE_CONTEXT_ONLY,
                priority=0,
                py_disaggregated_params=DisaggregatedParams(
                    disagg_request_id=rid, schedule_style=DisaggScheduleStyle.CONTEXT_FIRST
                ),
            ),
        )
        for rid in (11, 12)
    ]
    executor.waiting_queue.add_requests([cancelled, pending])
    executor.canceled_req_ids = [11]
    executor.max_num_active_requests = 1
    executor.is_benchmark_disagg = False
    executor.enable_iter_perf_stats = False
    executor.num_fetch_requests = 0
    executor._kv_load_tracker = SimpleNamespace(begin_step=lambda *_args: None, enabled=False)
    executor._kv_cache_capacity = 0
    executor.adp_router = SimpleNamespace(exclude_retiring_requests=False)
    executor.dist = SimpleNamespace(rank=0, tp_size=1, cp_config=None, cp_rank=0, cp_size=1)
    executor._fetch_and_enqueue_requests = Mock()
    executor._should_exclude_last_generation_logits = Mock(return_value=False)
    # Keep the real waiting-queue extraction and capacity admission. Conversion
    # is unrelated to the admission API's (requests, placement) return value.
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.merge_requests",
        lambda requests, **_kwargs: requests,
    )

    fetched = executor._fetch_new_requests(
        executor.waiting_queue, [_request(99)] * (1 - available_slots)
    )

    assert fetched == [cancelled] + ([pending] if available_slots else [])
    assert list(executor.waiting_queue) == ([] if available_slots else [pending])
    assert executor.num_fetch_requests == 1 + available_slots


def test_legacy_engine_without_late_cancel_hook_remains_unchanged() -> None:
    """C++/other backends retain their old already-completed-request behavior."""
    engine = Mock(spec=["can_enqueue_requests", "cancel_request"])
    engine.can_enqueue_requests.return_value = True
    worker = SimpleNamespace(engine=engine, _client_id_to_request_id={})
    BaseWorker.abort_request(worker, 101)
    engine.cancel_request.assert_not_called()


def test_outstanding_encoder_work_defers_model_idle_cancellation(executor: PyExecutor) -> None:
    """Model ownership is broader than previous_batch, including pending encoder work."""
    request = _request()
    executor.active_requests.append(request)
    executor.canceled_req_ids = [11]
    executor.inflight_req_ids.add(11)
    assert executor._progress_context_cancellations([request]) == [request]
    assert not executor._try_cancel_request(request)
    executor.disagg.request_logical_context_cancellation.assert_not_called()
    executor.inflight_req_ids.clear()
    assert executor._progress_context_cancellations([request]) == []


def test_retained_source_prevents_idle_wait_and_premature_shutdown(executor: PyExecutor) -> None:
    """A transfer-only iteration must progress without a GEN or another user request."""
    request = _request()
    executor.async_transfer_manager.start_transfer(request)
    executor.control_requests = []
    executor.request_accumulated = []
    executor.is_shutdown = False
    executor._disable_mpi = False
    executor.hang_detector = SimpleNamespace(pause=nullcontext)
    executor.request_broadcaster = SimpleNamespace(broadcast=lambda items, **_kwargs: (items, None))
    executor.executor_request_queue.get_from_request_queue = Mock(return_value=[])

    executor._fetch_and_enqueue_requests(executor.waiting_queue, 0)

    executor.executor_request_queue.get_from_request_queue.assert_called_once_with(timedelta(0))
    executor.is_shutdown = True
    assert not executor.should_stop_processing
    # Normal reaping releases the final claim; only then may shutdown finish.
    assert executor.async_transfer_manager.end_transfer(request)
    assert executor.should_stop_processing
