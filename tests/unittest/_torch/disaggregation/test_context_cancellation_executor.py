# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CTX cancellation routing and logical completion, independent of I/O retirement.

The executor, queue, native request, and transfer claim manager are real. The
transceiver boundary controls acceptance; native I/O settlement is exercised in
test_context_cancellation_retirement.py rather than simulated here.
"""

import pickle
import queue
import threading
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from fake_dist import FakeDistGroup, FakeDistRank

from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    CONTROL_REQUEST_ID,
    PROFILE_STOP_REQUEST_ID,
    SHUTDOWN_REQUEST_ID,
    ExecutorRequestQueue,
    RequestAdmissionState,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import RequestBroadcaster
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import FinishReason, RequestType
from tensorrt_llm.bindings.executor import Request as ExecutorRequest
from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor.base_worker import BaseWorker

pytestmark = pytest.mark.cpu_only


def _request(
    request_id: int = 11, *, context: bool = True, child_ids: tuple[int, ...] = ()
) -> LlmRequest:
    """Create a model-idle native request with a distinct client identity."""
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(8)),
        sampling_config=SamplingConfig(1, num_return_sequences=1 + len(child_ids)),
        is_streaming=False,
        draft_tokens=None,
        llm_request_type=(
            LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
            if context
            else LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
        ),
    )
    request.py_client_id = 101
    request.py_disaggregated_params = DisaggregatedParams(
        disagg_request_id=request_id,
        schedule_style=(
            DisaggScheduleStyle.CONTEXT_FIRST if context else DisaggScheduleStyle.GENERATION_FIRST
        ),
    )
    for child_id in child_ids:
        request.create_child_request(child_id)
    request.state = (
        LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
        if context
        else LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    )
    return request


def _executor(dist: FakeDistRank | None = None) -> PyExecutor:
    """Keep lifecycle methods real while replacing model, transport, and output effects."""
    engine = object.__new__(PyExecutor)
    engine.active_requests = []
    engine.canceled_req_ids = []
    engine.waiting_queue = FCFSWaitingQueue()
    engine._pending_transfer_responses = []
    engine._pending_response_terminations = []
    engine.response_cv = threading.Condition()
    engine.result_wait_queues = {}
    engine.gather_all_responses = False
    engine.enable_attention_dp = False
    engine.kv_connector_manager = None
    engine.previous_batch = None
    engine.inflight_req_ids = set()
    engine._disagg_pp_termination_handler = None
    engine.dist = dist or SimpleNamespace(rank=0, tp_size=1, world_size=1, has_pp=False)
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
        spec=[
            "supports_logical_context_cancellation",
            "request_logical_context_cancellation",
            "has_retiring_gen_receives",
            "cancel_gen_logically",
            "defer_gen_termination",
            "request_cancellation",
            "inflight_cancel_active",
        ]
    )
    engine.disagg.supports_logical_context_cancellation.return_value = True
    engine.disagg.request_logical_context_cancellation.return_value = True
    engine.disagg.has_retiring_gen_receives = False
    engine.disagg.cancel_gen_logically.return_value = False
    engine.disagg.defer_gen_termination.return_value = False
    engine.disagg.request_cancellation.return_value = False
    engine.disagg.inflight_cancel_active.return_value = False
    engine._enqueue_responses = Mock()
    engine._do_terminate_request = Mock()
    return engine


@pytest.fixture
def executor() -> PyExecutor:
    return _executor()


def _prepare_fetch(engine: PyExecutor) -> None:
    engine.control_requests = []
    engine.request_accumulated = []
    engine.is_shutdown = False
    engine._disable_mpi = False
    engine.hang_detector = SimpleNamespace(pause=nullcontext)
    engine.request_broadcaster = RequestBroadcaster(engine.dist, engine.hang_detector)


def _broadcast_rank(group: FakeDistGroup, rank: int) -> FakeDistRank:
    """Keep request-envelope broadcasts on the same checked group rendezvous."""
    dist = group.rank(rank)
    dist.has_pp = False

    def broadcast(value: object, root: int) -> object:
        return dist._gather(group.world_rendezvous(), "broadcast", value)[root]

    def broadcast_int64(value: list[int], root: int, *, prefer_cpu: bool = False) -> list[int]:
        return dist._gather(
            group.world_rendezvous(),
            "broadcast_int64",
            value,
            label=f"broadcast_int64(prefer_cpu={prefer_cpu})",
        )[root]

    dist.broadcast = broadcast
    dist.broadcast_int64 = broadcast_int64
    return dist


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
    assert responses[0][1].result.get_result().finish_reasons == [FinishReason.CANCELLED]
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
@pytest.mark.parametrize(
    "state",
    [LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER, LlmRequestState.GENERATION_COMPLETE],
)
def test_rejected_context_cancellation_remains_pending(
    executor: PyExecutor, at_post_forward: bool, state: LlmRequestState
) -> None:
    """A rejected cancellation keeps the request and intent for a later boundary."""
    request = _request()
    request.state = state
    executor.active_requests.append(request)
    executor.canceled_req_ids = [11]
    executor.disagg.request_logical_context_cancellation.return_value = False

    if at_post_forward:
        executor._handle_canceled_requests()
    else:
        assert executor._progress_context_cancellations([request]) == [request]

    executor.disagg.request_logical_context_cancellation.assert_called_once_with(request)
    assert executor.active_requests == [request]
    assert executor.canceled_req_ids == [11]
    assert request.state == state
    assert not executor._pending_transfer_responses
    assert not executor._pending_response_terminations
    executor._enqueue_responses.assert_not_called()
    executor._do_terminate_request.assert_not_called()


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
def test_fetch_preserves_admission_result_when_completing_queued_context_cancel(
    executor: PyExecutor, monkeypatch: pytest.MonkeyPatch, available_slots: int
) -> None:
    """Cancelled CTX items bypass capacity without altering normal admission."""
    cancelled, pending = [
        RequestQueueItem(
            rid,
            ExecutorRequest(
                input_token_ids=[1, 2, 3],
                max_tokens=4,
                client_id=rid + 100,
                type=RequestType.REQUEST_TYPE_CONTEXT_ONLY,
                priority=0,
                sampling_config=SamplingConfig(1),
            ),
        )
        for rid in (11, 12)
    ]
    executor.waiting_queue.add_requests([cancelled, pending])
    executor.canceled_req_ids = [11]
    executor.result_wait_queues = {11: object(), 12: object()}
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
    # Keep the real queued cancellation and capacity admission. Conversion
    # is unrelated to the admission API's (requests, placement) return value.
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.merge_requests",
        lambda requests, **_kwargs: requests,
    )

    fetched = executor._fetch_new_requests(
        executor.waiting_queue, [_request(99)] * (1 - available_slots)
    )

    assert fetched == ([pending] if available_slots else [])
    assert list(executor.waiting_queue) == ([] if available_slots else [pending])
    assert executor.num_fetch_requests == available_slots
    assert not executor.canceled_req_ids
    assert set(executor.result_wait_queues) == {12}
    executor._enqueue_responses.assert_called_once()
    responses = executor._enqueue_responses.call_args.args[0]
    assert len(responses) == 1 and responses[0][0] == 11
    assert responses[0][1].result.get_result().finish_reasons == [FinishReason.CANCELLED]
    executor.disagg.request_logical_context_cancellation.assert_not_called()
    executor._do_terminate_request.assert_not_called()


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


@pytest.mark.parametrize("is_shutdown", [False, True])
@pytest.mark.parametrize("cancel_queued", [False, True])
@pytest.mark.parametrize("pending_kind", ["context", "generation"])
def test_retained_transfer_prevents_idle_wait_and_premature_shutdown(
    executor: PyExecutor,
    monkeypatch: pytest.MonkeyPatch,
    is_shutdown: bool,
    cancel_queued: bool,
    pending_kind: str,
) -> None:
    """Either source claims or the destination snapshot keep owner progress reachable."""
    request = _request(context=pending_kind == "context")
    if pending_kind == "context":
        executor.async_transfer_manager.start_transfer(request)
    else:
        executor.disagg.has_retiring_gen_receives = True
    executor.control_requests = []
    executor.request_accumulated = []
    executor.is_shutdown = is_shutdown
    executor._disable_mpi = False
    executor.hang_detector = SimpleNamespace(pause=nullcontext)
    executor.request_broadcaster = RequestBroadcaster(executor.dist, executor.hang_detector)
    request_queue = executor.executor_request_queue
    request_queue.batch_wait_timeout_ms = 1000
    if cancel_queued:
        request_queue.request_queue.put(
            RequestQueueItem(request.py_request_id, is_canceled_request=True)
        )
    original_get = request_queue.request_queue.get

    def nonblocking_get(block: bool = True, timeout: float | None = None) -> RequestQueueItem:
        assert not block, "Retained transfers must progress without waiting for a request batch"
        return original_get(block=block, timeout=timeout)

    monkeypatch.setattr(request_queue.request_queue, "get", nonblocking_get)

    executor._fetch_and_enqueue_requests(executor.waiting_queue, 0)

    assert executor.canceled_req_ids == ([request.py_request_id] if cancel_queued else [])
    assert request_queue.request_queue.empty()
    executor.is_shutdown = True
    assert not executor.should_stop_processing
    # Observe the owner's completed retirement; only then may shutdown finish.
    if pending_kind == "context":
        assert executor.async_transfer_manager.end_transfer(request)
    else:
        executor.disagg.has_retiring_gen_receives = False
    assert executor.should_stop_processing


@pytest.mark.parametrize("is_shutdown", [False, True])
def test_batch_wait_without_retained_source_respects_shutdown(
    executor: PyExecutor, monkeypatch: pytest.MonkeyPatch, is_shutdown: bool
) -> None:
    """Normal batching remains enabled; shutdown does not wait for more requests."""
    executor.control_requests = []
    executor.request_accumulated = []
    executor.is_shutdown = is_shutdown
    executor._disable_mpi = False
    executor.hang_detector = SimpleNamespace(pause=nullcontext)
    executor.request_broadcaster = RequestBroadcaster(executor.dist, executor.hang_detector)
    request_queue = executor.executor_request_queue
    request_queue.batch_wait_timeout_ms = 1000
    original_get = request_queue.request_queue.get
    batch_waits = []

    def observe_get(block: bool = True, timeout: float | None = None) -> RequestQueueItem:
        if block:
            batch_waits.append(timeout)
            # Exercise the real batching loop without sleeping for its timeout.
            raise queue.Empty
        return original_get(block=block, timeout=timeout)

    monkeypatch.setattr(request_queue.request_queue, "get", observe_get)
    # A live request keeps the initial queue fetch nonblocking in the normal case.
    executor._fetch_and_enqueue_requests(executor.waiting_queue, int(not is_shutdown))

    assert len(batch_waits) == (0 if is_shutdown else 1)
    assert all(timeout is not None and timeout > 0 for timeout in batch_waits)


@pytest.mark.parametrize("pending_kind", ["context", "generation"])
def test_draining_control_waits_for_retained_transfer(
    executor: PyExecutor, monkeypatch: pytest.MonkeyPatch, pending_kind: str
) -> None:
    """An empty logical queue cannot authorize a control action while either owner remains."""
    request = _request(context=pending_kind == "context")
    if pending_kind == "context":
        executor.async_transfer_manager.start_transfer(request)
    else:
        executor.disagg.has_retiring_gen_receives = True
    control = RequestQueueItem(CONTROL_REQUEST_ID, control_requires_drain=True)
    executor.control_requests = [control]
    executor.control_request_barrier = Mock()
    executor.control_action_done = Mock()
    executor.hang_detector = SimpleNamespace(pause=nullcontext)
    executor.dwdp_manager = None
    executor._retire_inflight_batch_for_control = Mock()
    executor._pop_sleep_wakeup_abort = Mock(return_value=None)
    synchronize = Mock()
    monkeypatch.setattr("torch.cuda.synchronize", synchronize)
    # Even an already-expired legacy timer cannot release protected ownership.
    executor._control_drain_transfer_deadline = 30.0
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.time.monotonic", lambda: 60.0)

    executor._handle_control_request()

    assert executor.control_requests == [control]
    executor.control_request_barrier.set.assert_not_called()
    executor.control_action_done.wait.assert_not_called()
    synchronize.assert_not_called()
    assert executor._control_drain_transfer_deadline is None
    executor.async_transfer_manager.kv_cache_manager.unpin_blocks_by_id.assert_not_called()

    if pending_kind == "context":
        assert executor.async_transfer_manager.end_transfer(request)
    else:
        executor.disagg.has_retiring_gen_receives = False
    executor._handle_control_request()

    assert not executor.control_requests
    executor.control_request_barrier.set.assert_called_once()
    executor.control_action_done.wait.assert_called_once()
    executor._retire_inflight_batch_for_control.assert_called_once()
    synchronize.assert_called_once()


@pytest.mark.parametrize("late_abort", [False, True])
@pytest.mark.parametrize("new_requests_hint", [None, 0, 2])
def test_control_drain_delivers_accumulated_and_live_aborts_on_every_rank(
    monkeypatch: pytest.MonkeyPatch, late_abort: bool, new_requests_hint: int | None
) -> None:
    """Only existing-work aborts cross a control; future submissions keep their aborts."""
    group = FakeDistGroup(2, 2)
    engines = []
    retained = []
    for rank in range(2):
        engine = _executor(_broadcast_rank(group, rank))
        _prepare_fetch(engine)
        requests = [_request(11), _request(22)]
        requests[1].py_client_id = 202
        for request in requests:
            engine.async_transfer_manager.start_transfer(request)
        engines.append(engine)
        retained.append(requests)

    def abort(request: LlmRequest) -> RequestQueueItem:
        return RequestQueueItem(
            request.py_client_id if late_abort else request.py_request_id,
            is_canceled_request=True,
            cancel_context_client_id=request.py_client_id if late_abort else None,
        )

    future, later = [
        RequestQueueItem(rid, SimpleNamespace(priority=0, py_conversation_params=None))
        for rid in (111, 112)
    ]
    future_abort = RequestQueueItem(future.id, is_canceled_request=True)
    later_abort = RequestQueueItem(later.id, is_canceled_request=True)
    control = RequestQueueItem(CONTROL_REQUEST_ID, control_id="drain")
    next_control = RequestQueueItem(CONTROL_REQUEST_ID, control_id="next")
    profile = RequestQueueItem(PROFILE_STOP_REQUEST_ID)
    shutdown = RequestQueueItem(SHUTDOWN_REQUEST_ID)
    root_queue = engines[0].executor_request_queue
    for item in (control, future, future_abort, abort(retained[0][0])):
        root_queue.request_queue.put(item)

    def fetch(rank: int) -> None:
        engine = engines[rank]
        engine._fetch_and_enqueue_requests(
            engine.waiting_queue,
            len(engine.active_requests),
            new_requests_hint=new_requests_hint if engine.control_requests else None,
        )

    group.run(fetch)
    assert engines[0].request_accumulated[:2] == [future, future_abort]
    for item in (abort(retained[0][1]), next_control, later, later_abort, profile, shutdown):
        root_queue.request_queue.put(item)
    root_queue.batch_wait_timeout_ms = 1000
    original_get = root_queue.request_queue.get

    def nonblocking_get(block: bool = True, timeout: float | None = None) -> RequestQueueItem:
        assert block is False, "A draining control must not wait to fill a request batch"
        return original_get(block=block, timeout=timeout)

    monkeypatch.setattr(root_queue.request_queue, "get", nonblocking_get)
    group.run(fetch)

    deferred = [future, future_abort, next_control, later, later_abort, profile, shutdown]
    assert engines[0].request_accumulated == deferred
    assert root_queue.request_queue.empty()
    assert engines[1].request_accumulated == []
    for engine, requests in zip(engines, retained):
        assert engine.canceled_req_ids == [11, 22]
        assert not engine.waiting_queue
        assert engine._progress_context_cancellations([]) == []
        assert engine.disagg.request_logical_context_cancellation.call_args_list == [
            call(request) for request in requests
        ]
        assert engine.control_requests[0].control_id == "drain"
        assert engine.async_transfer_manager.requests_in_transfer() == dict(zip((11, 22), requests))
        engine.async_transfer_manager.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
        engine._enqueue_responses.assert_not_called()
    group.run(lambda rank: engines[rank]._handle_control_request())
    assert all(engine.control_requests[0].control_id == "drain" for engine in engines)

    # No new intent still requires the same request-count collective on both ranks.
    group.run(fetch)
    assert [name for name, _ in group.rank(0).calls] == [
        "broadcast_int64",
        "broadcast",
        "broadcast_int64",
        "broadcast",
        "tp_allgather",
        "broadcast_int64",
    ]
    assert [name for name, _ in group.rank(0).calls] == [name for name, _ in group.rank(1).calls]
    monkeypatch.setattr("torch.cuda.synchronize", Mock())
    for engine in engines:
        engine.control_request_barrier = Mock()
        engine.control_action_done = Mock()
        engine._retire_inflight_batch_for_control = Mock()
        engine._pop_sleep_wakeup_abort = Mock(return_value=None)
        engine.dwdp_manager = None
    # Exercise the group control gate with asymmetric local readiness. This is
    # not a substitute for the native retirement agreement tests.
    for request in retained[0]:
        assert engines[0].async_transfer_manager.end_transfer(request)
    group.run(lambda rank: engines[rank]._handle_control_request())
    for engine in engines:
        engine.control_request_barrier.set.assert_not_called()
        assert engine.control_requests[0].control_id == "drain"
    for request in retained[1]:
        assert engines[1].async_transfer_manager.end_transfer(request)
    group.run(lambda rank: engines[rank]._handle_control_request())
    for engine in engines:
        engine.control_request_barrier.set.assert_called_once()

    root_queue.batch_wait_timeout_ms = 0
    group.run(fetch)
    for engine in engines:
        assert [item.id for item in engine.waiting_queue] == [future.id]
        assert engine.canceled_req_ids == [future.id]
        assert engine.control_requests[0].control_id == "next"
        engine._handle_canceled_requests()
        assert not engine.waiting_queue and not engine.canceled_req_ids
    assert engines[0].request_accumulated == [later, later_abort, profile, shutdown]


@pytest.mark.parametrize("owner_rank", [0, 1])
def test_adp_drain_delivers_parking_abort_to_the_owning_rank(
    monkeypatch: pytest.MonkeyPatch, owner_rank: int
) -> None:
    """An empty leader must deliver GEN cancellation without admitting future work."""
    group = FakeDistGroup(2, 2)
    engines = [_executor(_broadcast_rank(group, rank)) for rank in range(2)]
    request = _request(context=False)
    engines[owner_rank].active_requests.append(request)
    for engine in engines:
        _prepare_fetch(engine)
        engine.enable_attention_dp = True
        engine.kv_cache_transceiver.supports_idle_request_cancellation.return_value = True
        engine.disagg.supports_logical_context_cancellation.return_value = False
        engine.control_requests = [RequestQueueItem(CONTROL_REQUEST_ID, control_id="drain")]
        engine.control_request_barrier = Mock()
        engine.control_action_done = Mock()
        engine._retire_inflight_batch_for_control = Mock()
        engine._pop_sleep_wakeup_abort = Mock(return_value=None)
        engine.dwdp_manager = None
    monkeypatch.setattr("torch.cuda.synchronize", Mock())

    root = engines[0]
    future = RequestQueueItem(111, SimpleNamespace(priority=0, py_conversation_params=None))
    future_abort = RequestQueueItem(future.id, is_canceled_request=True)
    wrong_namespace = RequestQueueItem(
        request.py_request_id,
        is_canceled_request=True,
        cancel_context_client_id=request.py_request_id,
    )
    unmatched = RequestQueueItem(777, is_canceled_request=True)
    deferred = [future, future_abort, wrong_namespace, unmatched]
    root.request_accumulated = deferred.copy()
    root.executor_request_queue.begin_sleep_transition(["model_weights"])
    root.executor_request_queue.batch_wait_timeout_ms = 1000
    worker = SimpleNamespace(engine=root, _client_id_to_request_id={101: 11})
    BaseWorker.abort_request(worker, request.py_client_id)
    original_get = root.executor_request_queue.request_queue.get

    def nonblocking_get(block: bool = True, timeout: float | None = None) -> RequestQueueItem:
        assert not block, "ADP drain must not wait for batch accumulation"
        return original_get(block=block, timeout=timeout)

    monkeypatch.setattr(root.executor_request_queue.request_queue, "get", nonblocking_get)

    def fetch(rank: int) -> None:
        engines[rank]._fetch_and_enqueue_requests(engines[rank].waiting_queue, 1)

    group.run(fetch)
    for engine in engines:
        assert engine.canceled_req_ids == [request.py_request_id]
        assert not engine.waiting_queue
        engine._cancel_waiting_disagg_requests([])
    owner = engines[owner_rank]
    owner.disagg.request_cancellation.assert_called_once_with(request)
    engines[1 - owner_rank].disagg.request_cancellation.assert_not_called()
    assert owner.active_requests == [request]
    assert not request.is_finished
    assert root.request_accumulated == deferred
    group.run(lambda rank: engines[rank]._handle_control_request())
    for engine in engines:
        engine.control_request_barrier.set.assert_not_called()
        engine._do_terminate_request.assert_not_called()

    # An empty drain iteration still enters the same group snapshot and probe.
    group.run(fetch)
    for rank in range(2):
        assert [name for name, _ in group.rank(rank).calls] == [
            "tp_allgather",
            "broadcast_int64",
            "broadcast",
            "tp_allgather",
            "tp_allgather",
            "broadcast_int64",
        ]
    owner.disagg.request_cancellation.return_value = True
    for engine in engines:
        engine._cancel_waiting_disagg_requests([])
        engine._flush_pending_transfer_responses()
    assert request.is_finished_due_to_cancellation
    owner._do_terminate_request.assert_called_once_with(request)
    group.run(lambda rank: engines[rank]._handle_control_request())
    for engine in engines:
        assert not engine.control_requests
        engine.control_request_barrier.set.assert_called_once()
    assert root.request_accumulated == deferred


@pytest.mark.parametrize("excluded", ["legacy", "connector", "non-draining"])
def test_unqualified_adp_control_keeps_original_fetch_gate(
    executor: PyExecutor, excluded: str
) -> None:
    _prepare_fetch(executor)
    executor.enable_attention_dp = True
    executor.kv_cache_transceiver.supports_idle_request_cancellation.return_value = (
        excluded != "legacy"
    )
    if excluded == "connector":
        executor.kv_connector_manager = Mock()
    executor.control_requests = [
        RequestQueueItem(CONTROL_REQUEST_ID, control_requires_drain=excluded != "non-draining")
    ]
    executor.cancel_request(11)

    executor._fetch_and_enqueue_requests(executor.waiting_queue, 1)

    assert executor.executor_request_queue.get_request_queue_size() == 1
    assert not executor.canceled_req_ids and not executor.waiting_queue


@pytest.mark.parametrize("target_location", ["active", "waiting"])
def test_control_drain_cancels_admitted_work_without_admitting_future_work(
    executor: PyExecutor, target_location: str
) -> None:
    _prepare_fetch(executor)
    request = _request()
    queued = RequestQueueItem(
        request.py_request_id,
        ExecutorRequest(
            input_token_ids=[1, 2, 3],
            max_tokens=4,
            client_id=request.py_client_id,
            type=RequestType.REQUEST_TYPE_CONTEXT_ONLY,
            priority=0,
            sampling_config=SamplingConfig(1),
        ),
    )
    if target_location == "active":
        executor.active_requests.append(request)
    else:
        executor.waiting_queue.add_requests([queued])
    future = RequestQueueItem(
        12,
        SimpleNamespace(
            priority=0,
            request_type=RequestType.REQUEST_TYPE_CONTEXT_ONLY,
            py_disaggregated_params=DisaggregatedParams(
                disagg_request_id=12, schedule_style=DisaggScheduleStyle.CONTEXT_FIRST
            ),
        ),
    )
    future_abort = RequestQueueItem(future.id, is_canceled_request=True)
    executor.control_requests = [RequestQueueItem(CONTROL_REQUEST_ID)]
    executor.request_accumulated = [future, future_abort]
    executor.cancel_request(request.py_request_id)

    executor._fetch_and_enqueue_requests(executor.waiting_queue, len(executor.active_requests))

    assert executor.canceled_req_ids == [request.py_request_id]
    assert executor.request_accumulated == [future, future_abort]
    assert len(executor.control_requests) == 1
    if target_location == "active":
        assert executor._progress_context_cancellations([request]) == []
        executor.disagg.request_logical_context_cancellation.assert_called_once_with(request)
    else:
        executor._cancel_queued_context_requests(executor.waiting_queue)
        executor.disagg.request_logical_context_cancellation.assert_not_called()
        executor._do_terminate_request.assert_not_called()
    executor._enqueue_responses.assert_called_once()
    assert not executor.canceled_req_ids
    assert not executor.waiting_queue


def test_control_drain_uses_parent_ids_and_retained_client_namespace(executor: PyExecutor) -> None:
    """A late client ID is neither an engine ID nor an active-only client mapping."""
    _prepare_fetch(executor)
    parent = _request(child_ids=(12,))
    child = parent.child_requests[0]
    assert child.is_child
    assert child.parent_request_id == parent.py_request_id
    assert child.py_client_id == parent.py_client_id
    executor.async_transfer_manager.start_transfer(child)
    unrelated = _request(101)
    unrelated.py_client_id = 909
    executor.active_requests.append(unrelated)
    executor.control_requests = [RequestQueueItem(CONTROL_REQUEST_ID)]
    deferred = [
        RequestQueueItem(12, is_canceled_request=True),
        RequestQueueItem(909, is_canceled_request=True, cancel_context_client_id=909),
    ]
    executor.request_accumulated = deferred.copy()
    executor.cancel_request(parent.py_request_id)
    executor._cancel_context_transfer(parent.py_client_id)

    executor._fetch_and_enqueue_requests(executor.waiting_queue, 1)

    assert executor.canceled_req_ids == [11, 11]
    assert executor.request_accumulated == deferred
    assert executor._progress_context_cancellations([unrelated]) == [unrelated]
    executor.disagg.request_logical_context_cancellation.assert_called_once_with(child)
    assert executor.active_requests == [unrelated]


@pytest.mark.parametrize("late_abort", [False, True])
@pytest.mark.parametrize("state", list(RequestAdmissionState))
def test_worker_cancellation_admission_does_not_reopen_requests(
    executor: PyExecutor, state: RequestAdmissionState, late_abort: bool
) -> None:
    """PARKING accepts both abort namespaces; other closed states reject even direct ingress."""
    request_queue = executor.executor_request_queue
    if state is not RequestAdmissionState.RUNNING:
        request_queue.begin_sleep_transition(["model"])
        if state in (RequestAdmissionState.PARKED, RequestAdmissionState.WAKING):
            request_queue.complete_sleep_transition()
        if state is RequestAdmissionState.WAKING:
            request_queue.begin_wakeup_transition(["model"])
        if state is RequestAdmissionState.FAILED:
            request_queue.fail_sleep_wakeup_transition()
        assert not executor.can_enqueue_requests()
        with pytest.raises(RuntimeError, match="admission is"):
            request_queue.enqueue_request(Mock())
    worker = SimpleNamespace(
        engine=executor, _client_id_to_request_id={} if late_abort else {101: 11}
    )
    BaseWorker.abort_request(worker, 101)
    items = request_queue.get_from_request_queue(timedelta(0), wait_for_batch=False)
    allowed = state in (RequestAdmissionState.RUNNING, RequestAdmissionState.PARKING)
    assert len(items) == int(allowed)
    if allowed:
        assert items[0].id == (101 if late_abort else 11)
        assert items[0].cancel_context_client_id == (101 if late_abort else None)
    executor.cancel_request(11)
    executor._cancel_context_transfer(101)
    assert request_queue.get_request_queue_size() == (2 if allowed else 0)


@pytest.mark.parametrize("late_abort", [False, True])
def test_cancellation_enqueue_rechecks_state_after_worker_admission(
    executor: PyExecutor, monkeypatch: pytest.MonkeyPatch, late_abort: bool
) -> None:
    """An abort racing the end of PARKING cannot bypass the queue's locked state check."""
    request_queue = executor.executor_request_queue
    request_queue.begin_sleep_transition(["model"])

    def stale_admission() -> bool:
        request_queue.complete_sleep_transition()
        return True

    monkeypatch.setattr(executor, "_can_enqueue_cancellation", stale_admission)
    worker = SimpleNamespace(
        engine=executor, _client_id_to_request_id={} if late_abort else {101: 11}
    )
    BaseWorker.abort_request(worker, 101)
    assert request_queue.request_queue.empty()


@pytest.mark.parametrize("rank,active", [(1, True), (0, False)])
def test_cancellation_ingress_rejects_non_root_or_inactive_executor(
    executor: PyExecutor, rank: int, active: bool
) -> None:
    executor.dist.rank = rank
    executor.executor_request_queue.active = active
    executor.executor_request_queue.begin_sleep_transition(["model"])
    for mapping in ({}, {101: 11}):
        BaseWorker.abort_request(
            SimpleNamespace(engine=executor, _client_id_to_request_id=mapping), 101
        )
    executor.cancel_request(11)
    executor._cancel_context_transfer(101)
    assert executor.executor_request_queue.request_queue.empty()


@pytest.mark.parametrize("can_enqueue", [False, True])
def test_legacy_engine_cancellation_keeps_original_admission_gate(can_enqueue: bool) -> None:
    engine = Mock(spec=["can_enqueue_requests", "cancel_request"])
    engine.can_enqueue_requests.return_value = can_enqueue
    BaseWorker.abort_request(
        SimpleNamespace(engine=engine, _client_id_to_request_id={101: 11}), 101
    )
    if can_enqueue:
        engine.cancel_request.assert_called_once_with(11)
    else:
        engine.cancel_request.assert_not_called()


def test_non_draining_control_keeps_its_suffix_parked(executor: PyExecutor) -> None:
    _prepare_fetch(executor)
    request = _request()
    executor.async_transfer_manager.start_transfer(request)
    executor.control_requests = [RequestQueueItem(CONTROL_REQUEST_ID, control_requires_drain=False)]
    cancellation = RequestQueueItem(request.py_request_id, is_canceled_request=True)
    executor.request_accumulated = [cancellation]
    executor._cancel_context_transfer(request.py_client_id)

    executor._fetch_and_enqueue_requests(executor.waiting_queue, 0)

    assert not executor.canceled_req_ids
    assert executor.request_accumulated == [cancellation]
    assert executor.executor_request_queue.get_request_queue_size() == 1
