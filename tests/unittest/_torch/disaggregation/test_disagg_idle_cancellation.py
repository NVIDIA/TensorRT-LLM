# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Progress session-free CTX and ownership-bridge cancellation without a model batch.

The worker, request queue, executor cancellation/response methods, coordinator,
and receive-session ownership are real. Network endpoints, GPU readiness and
resource allocation are controlled boundaries; these CPU tests require the
native TRT-LLM runtime but do not qualify GPU/NIXL deployment or DMA fencing.
"""

import threading
from contextlib import nullcontext
from datetime import timedelta
from types import MethodType, SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock, Mock

import pytest
import torch
import zmq
from _torch.pyexecutor.llm_request_factory import make_llm_request
from coordinator_harness import CoordinatorHarness
from fake_dist import FakeDistGroup

from tensorrt_llm._torch.disaggregation import diagnostics as disagg_diagnostics
from tensorrt_llm._torch.disaggregation.base import TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import BindKvCacheTransceiver
from tensorrt_llm._torch.disaggregation.native.extent import Cancelled, Chunk, Failed
from tensorrt_llm._torch.disaggregation.native.handle import TaskHandle
from tensorrt_llm._torch.disaggregation.native.messenger import ZMQMessenger
from tensorrt_llm._torch.disaggregation.native.retirement import RetirementWatchdog
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVRecvTask,
    Receiver,
    RxSession,
    TransferWorker,
)
from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_mod
from tensorrt_llm._torch.disaggregation.orchestration.cancellation import ReceiveCancellation
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.disagg_adapter import (
    PyExecutorEffects,
    PyExecutorRequestRegistry,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import (
    DecoderInputPreparer,
    ExtraInputsCollector,
    HostInputs,
)
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmResponse, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import BatchState, PyExecutor
from tensorrt_llm._torch.pyexecutor.request_delivery import ResponseChannel
from tensorrt_llm._torch.pyexecutor.sampler import (
    SampleStateTensorsHostTorch,
    SampleStateTorch,
    TorchSampler,
)
from tensorrt_llm._torch.pyexecutor.scheduler import (
    FCFSWaitingQueue,
    ScheduledRequests,
    SchedulerOutput,
)
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import FinishReason, RequestType
from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

# The receiver mints one report token per writer; a KV result is trusted
# only if it echoes that token. See RxSession.process_kv_agent_result.
_KV_RESULT_REPORT_TOKEN = b"report-token-123"


def _request(
    request_id: int,
    *,
    context: bool = False,
    num_return_sequences: int = 1,
    lora_task_id: int | None = None,
) -> LlmRequest:
    """Create a native request waiting for its generation-first peer.

    Args:
        request_id: Executor and transfer identity.
        context: Whether to create the context-side request.
        num_return_sequences: Number of sequences, including any native children.
        lora_task_id: Adapter identity, if the request holds PEFT resources.

    Returns:
        A request that has not run a model batch.
    """
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(8)),
        sampling_config=SamplingConfig(1, num_return_sequences=num_return_sequences),
        is_streaming=False,
        draft_tokens=None,
        lora_task_id=lora_task_id,
        llm_request_type=(
            LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
            if context
            else LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
        ),
    )
    request.py_disaggregated_params = DisaggregatedParams(
        disagg_request_id=request_id,
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
    )
    request.state = (
        LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
        if context
        else LlmRequestState.DISAGG_GENERATION_INIT
    )
    return request


@pytest.fixture
def executor(monkeypatch: pytest.MonkeyPatch) -> PyExecutor:
    """Build the real cancellation path with controlled external boundaries.

    Args:
        monkeypatch: Fixture isolating environment and legacy capability flags.

    Returns:
        An unstarted executor with native queue and receive-session handling.
    """
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    monkeypatch.setattr(coordinator_mod, "is_disagg_inflight_cancel_enabled", lambda: False)
    engine = object.__new__(PyExecutor)
    engine.active_requests = []
    engine.canceled_req_ids = []
    engine.waiting_queue = FCFSWaitingQueue()
    engine._pending_transfer_responses = []
    engine._pending_response_terminations = []
    engine.is_shutdown = False
    engine.enable_attention_dp = False
    engine.gather_all_responses = False
    engine.kv_connector_manager = None
    engine.previous_batch = None
    engine.capturable_guided_decoder = None
    engine.is_encoder_decoder = False
    engine._mm_encoder_item_scheduling_enabled = False
    engine.dist = SimpleNamespace(
        rank=0,
        tp_size=1,
        pp_size=1,
        world_size=1,
        mapping=SimpleNamespace(tp_group=[0]),
        tp_allreduce=lambda value, op=None: value,
    )
    engine.executor_request_queue = ExecutorRequestQueue(
        dist=engine.dist, max_batch_size=8, enable_iter_perf_stats=False, batch_wait_timeout_ms=0.0
    )
    engine._response_channel = ResponseChannel()
    engine._disagg_pp_termination_handler = None
    engine._prefetched_request_ids = set()
    engine.resource_manager = Mock(spec=["free_resources"])
    engine.sampler = Mock(spec=["release_request"])
    engine.model_engine = SimpleNamespace(wait_for_input_copy=Mock(), is_spec_decode=False)
    engine.draft_model_engine = None
    engine._maybe_record_hang_diagnostic_phase = Mock()
    engine.async_transfer_manager = SimpleNamespace(
        requests_in_transfer=lambda: {}, has_any_inflight_requests=lambda: False
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._fp4_mla_bridge_enabled = True
    transceiver._ever_had_recv_session = False
    transceiver._gen_need_sync = False
    transceiver._gen_allgather = Mock(side_effect=AssertionError("unexpected collective"))
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._mapping = SimpleNamespace(enable_attention_dp=True, pp_size=1)
    transceiver._dist = engine.dist
    transceiver.kv_transfer_timeout_ms = 1000
    worker = object.__new__(TransferWorker)
    worker._retirement_watchdog = RetirementWatchdog(Mock(), clock=lambda: 0.0)
    worker._config = SimpleNamespace(rx_timeout_s=1.0)
    transceiver._transfer_worker = worker
    engine.kv_cache_transceiver = transceiver
    engine._disagg_coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=engine.async_transfer_manager,
        kv_cache_manager=None,
        dist=engine.dist,
        effects=PyExecutorEffects(engine),
        registry=PyExecutorRequestRegistry(engine),
        enable_attention_dp=False,
    )
    return engine


def _enable_context_first(executor: PyExecutor, request: LlmRequest) -> None:
    """Select the existing protected BF16 context-first capability in the harness.

    Args:
        executor: Executor whose external runtime boundaries are controlled.
        request: Generation request to give a context-first immutable identity.
    """
    request.py_disaggregated_params = DisaggregatedParams(
        disagg_request_id=request.py_request_id,
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
    )
    transceiver = executor.kv_cache_transceiver
    transceiver._fp4_mla_bridge_enabled = False
    transceiver._context_first_ownership = True
    transceiver._cancelled_gen_receives = {}


def _abort(executor: PyExecutor, request: LlmRequest, *, repeats: int = 1) -> None:
    """Deliver worker cancellation through the real executor request queue.

    Args:
        executor: Executor receiving the cancellation.
        request: Request mapped from the worker's client ID.
        repeats: Number of duplicate abort messages to deliver.
    """
    worker = object.__new__(BaseWorker)
    worker.doing_shutdown = True  # The test does not start or own an executor process.
    worker.engine = executor
    worker._client_id_to_request_id = {101: request.py_request_id}
    for _ in range(repeats):
        worker.abort_request(101)
    raw = executor.executor_request_queue.get_request_queue()
    items = [raw.get_nowait() for _ in range(repeats)]
    assert executor._handle_special_queue_items(items) == []


def _assert_released_once(executor: PyExecutor, request: LlmRequest) -> None:
    """Verify one terminal response and one ordinary resource release.

    Args:
        executor: Executor with a staged cancellation response.
        request: Cancelled request whose physical claims have settled.
    """
    assert request.is_finished_due_to_cancellation
    assert request not in executor.active_requests
    executor._flush_pending_transfer_responses()
    (response,) = executor._response_channel.responses[request.py_request_id]
    assert response.error_msg is None
    executor.sampler.release_request.assert_called_once_with(request)
    executor.resource_manager.free_resources.assert_called_once_with(request)
    executor._flush_pending_transfer_responses()
    assert len(executor._response_channel.responses[request.py_request_id]) == 1
    executor.sampler.release_request.assert_called_once_with(request)
    executor.resource_manager.free_resources.assert_called_once_with(request)


@pytest.mark.parametrize("bridge_enabled", [False, True])
def test_queued_cancellation_remains_gated_before_activation(
    executor: PyExecutor, bridge_enabled: bool
) -> None:
    """Only qualified runtimes remove queued aborts before a model batch."""
    request = _request(1)
    executor.kv_cache_transceiver._fp4_mla_bridge_enabled = bridge_enabled
    executor.waiting_queue.add_requests([RequestQueueItem(1), RequestQueueItem(2)])
    _abort(executor, request)

    assert executor._cancel_waiting_disagg_requests([]) == []

    assert [item.id for item in executor.waiting_queue] == ([2] if bridge_enabled else [1, 2])
    assert executor.canceled_req_ids == ([] if bridge_enabled else [1])
    assert executor.active_requests == []
    assert executor._pending_transfer_responses == []
    executor.resource_manager.free_resources.assert_not_called()


@pytest.mark.parametrize("context, bridge_enabled", [(False, True), (True, True), (True, False)])
def test_worker_abort_settles_waiting_request_without_model_batch(
    executor: PyExecutor, context: bool, bridge_enabled: bool
) -> None:
    """Duplicate aborts settle idle work once and preserve a healthy sibling."""
    cancelled, healthy = _request(1, context=context), _request(2, context=context)
    executor.kv_cache_transceiver._fp4_mla_bridge_enabled = bridge_enabled
    executor.active_requests = [cancelled, healthy]
    executor.waiting_queue.add_request(RequestQueueItem(1))
    if context:
        executor.kv_cache_transceiver._wait_reqs = {1: cancelled, 2: healthy}
    _abort(executor, cancelled, repeats=2)

    remaining = executor._cancel_waiting_disagg_requests([cancelled, healthy])

    assert remaining == [healthy]
    assert executor.active_requests == [healthy]
    assert not executor.waiting_queue
    assert executor.canceled_req_ids == []
    assert not healthy.is_finished
    if context:
        assert executor.kv_cache_transceiver._wait_reqs == {2: healthy}
    _assert_released_once(executor, cancelled)
    _abort(executor, cancelled)
    executor._cancel_waiting_disagg_requests([])
    executor._flush_pending_transfer_responses()
    executor.resource_manager.free_resources.assert_called_once_with(cancelled)


@pytest.mark.parametrize("bridge_enabled", [False, True])
def test_worker_abort_settles_context_waiting_from_previous_pass(
    executor: PyExecutor, bridge_enabled: bool
) -> None:
    """Cancel an earlier peer wait once without reviving it on late readiness."""
    cancelled, healthy = _request(1, context=True), _request(2, context=True)
    for request in (cancelled, healthy):
        request.state = LlmRequestState.CONTEXT_INIT
    executor.active_requests = [cancelled, healthy]
    transceiver = executor.kv_cache_transceiver
    transceiver._fp4_mla_bridge_enabled = bridge_enabled
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    peer_ready = Mock(return_value=False)
    transceiver._transfer_worker = SimpleNamespace(has_all_peer_req_infos_for_send=peer_ready)
    executor.disagg.prepare_context_schedulable([cancelled, healthy])
    assert transceiver._wait_reqs == {1: cancelled, 2: healthy}
    assert cancelled.state == healthy.state == LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER

    _abort(executor, cancelled, repeats=2)
    assert executor._cancel_waiting_disagg_requests([]) == []

    assert executor.active_requests == [healthy]
    assert transceiver._wait_reqs == {2: healthy}
    assert executor.canceled_req_ids == []
    assert not healthy.is_finished
    executor.resource_manager.free_resources.assert_not_called()
    _assert_released_once(executor, cancelled)

    peer_ready.reset_mock()
    peer_ready.return_value = True
    executor.disagg.prepare_context_schedulable([])

    peer_ready.assert_called_once_with(2)
    assert transceiver._wait_reqs == {}
    assert healthy.state == LlmRequestState.CONTEXT_INIT
    assert executor.active_requests == [healthy]
    _assert_released_once(executor, cancelled)


def test_cancellation_beats_late_transfer_readiness(executor: PyExecutor) -> None:
    """Late receive completion cannot schedule an already cancelled request."""
    request = _request(1)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    executor.active_requests = [request]
    _abort(executor, request)

    executor._cancel_waiting_disagg_requests([])

    assert not executor.disagg.try_finish_gen_receive(request)
    _assert_released_once(executor, request)


def test_existing_transfer_error_is_not_rewritten_as_cancellation(executor: PyExecutor) -> None:
    """Cancellation unblocks error cleanup without replacing its outcome."""
    request = _request(1)
    request.state = LlmRequestState.DISAGG_TRANS_ERROR
    executor.active_requests = [request]
    _abort(executor, request)

    executor._cancel_waiting_disagg_requests([])

    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert not request.is_finished_due_to_cancellation
    assert executor.active_requests == [request]
    assert executor.canceled_req_ids == []
    assert executor.disagg._requests_in_error_state() == [request]
    assert executor._pending_transfer_responses == []
    executor.resource_manager.free_resources.assert_not_called()


@pytest.mark.parametrize(
    "runtime", ["bridge", "python", "cpp", "context_bridge", "context_python", "connector"]
)
def test_post_batch_error_preservation_stays_in_the_qualified_path(
    executor: PyExecutor, runtime: str
) -> None:
    """Keep legacy cancellation when idle transfer-error cleanup is unsupported."""
    transceiver = executor.kv_cache_transceiver
    transceiver._fp4_mla_bridge_enabled = runtime in ("bridge", "context_bridge", "connector")
    if runtime == "cpp":
        transceiver = object.__new__(BindKvCacheTransceiver)
    executor.kv_cache_transceiver = transceiver
    executor.disagg._transceiver = transceiver
    if runtime == "connector":
        executor.kv_connector_manager = Mock(capacity_only=False)
        executor.kv_connector_manager.defer_load_termination.return_value = False
    request = _request(1, context=runtime.startswith("context"))
    request.state = LlmRequestState.DISAGG_TRANS_ERROR
    executor.active_requests = [request]
    _abort(executor, request)

    executor._handle_canceled_requests()

    if runtime == "bridge":
        assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
        assert not request.is_finished
        assert executor.disagg._requests_in_error_state() == [request]
    else:
        assert request.is_finished_due_to_cancellation
    assert executor.canceled_req_ids == []
    assert executor.active_requests == [request]
    assert executor._pending_transfer_responses == []
    executor.resource_manager.free_resources.assert_not_called()


class _ReachedIdleFetch(Exception):
    """Stop at the blocking queue boundary without waiting for more traffic."""


@pytest.mark.parametrize("overlap", [False, True], ids=["normal", "overlap"])
@pytest.mark.parametrize("failure_rank", [0, 1, None], ids=["local", "peer", "healthy"])
@pytest.mark.parametrize("response_scope", ["tp", "matching_world", "wider_world"])
def test_preserved_failure_is_flushed_before_idle_fetch(
    executor: PyExecutor,
    monkeypatch: pytest.MonkeyPatch,
    overlap: bool,
    failure_rank: int | None,
    response_scope: str,
) -> None:
    """Flush before fetching only when error consensus covers the response group."""
    request = _request(1)
    request.state = LlmRequestState.DISAGG_TRANS_ERROR
    if failure_rank == 0:
        executor.active_requests = [request]
        _abort(executor, request)
        executor._handle_canceled_requests()
        assert executor.canceled_req_ids == []
        assert request.state == LlmRequestState.DISAGG_TRANS_ERROR

    events = []
    executor.enable_attention_dp = True
    executor.dist.tp_size = 2
    executor.dist.world_size = 4 if response_scope == "wider_world" else 2
    executor.gather_all_responses = response_scope != "tp"
    should_flush = failure_rank is not None and response_scope != "wider_world"
    executor.disagg._enable_attention_dp = True
    executor.dist.tp_allgather = Mock(
        side_effect=lambda vote: [
            vote,
            {"error_ids": [1] if failure_rank == 1 else [], "blocked_ids": []},
        ]
    )

    def gather(responses: list[tuple[int, LlmResponse]]) -> list[list[tuple[int, LlmResponse]]]:
        """Model a response gather, including a failure owned by the other rank."""
        events.append("flush")
        remote = (
            [(1, LlmResponse(request_id=1, error_msg="Disagg KV cache transfer error"))]
            if failure_rank == 1
            else []
        )
        return [responses, remote]

    executor.dist.tp_gather = Mock(side_effect=gather)
    executor.dist.allgather = Mock(side_effect=gather)
    response_gather = (
        executor.dist.allgather if executor.gather_all_responses else executor.dist.tp_gather
    )
    executor.resource_manager.free_resources.side_effect = lambda _: events.append("terminate")
    executor._wait_for_model_engine_input_copy = Mock(
        side_effect=lambda: events.append("copy_wait")
    )
    executor._fatal_error = None
    executor.device_id = 0
    executor._profiler = lambda: nullcontext(lambda: None)
    executor.hang_detector = MagicMock()
    executor.enable_iter_perf_stats = False
    executor._resource_governor_enabled = False
    executor._is_kv_manager_v2 = False
    executor.is_benchmark_disagg = False
    executor.control_requests = []
    executor.request_accumulated = []
    executor._disable_mpi = False
    executor._release_unused_connector_reservations = Mock()
    executor._poll_encoder_steps = Mock()
    executor._fetch_and_activate_new_requests = lambda: executor._fetch_and_enqueue_requests(
        executor.waiting_queue, len(executor.active_requests)
    )
    for target in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"tensorrt_llm._torch.pyexecutor.py_executor.{target}", Mock())

    def stop_before_blocking(timeout: timedelta | None, *, wait_for_batch: bool) -> None:
        """Check publication and cleanup before the real fetch would wait forever."""
        events.append("fetch")
        assert timeout is None
        assert wait_for_batch
        assert executor.active_requests == []
        if failure_rank == 0 and not should_flush:
            # TP consensus cannot authorize a world-wide collective. Keep the
            # buffered response and termination for an existing aligned flush.
            (pending,) = executor._pending_transfer_responses
            assert pending[0] == 1
            assert pending[1].error_msg == "Disagg KV cache transfer error"
            assert executor._pending_response_terminations == [request]
        else:
            assert executor._pending_transfer_responses == []
            assert executor._pending_response_terminations == []
        if not should_flush:
            assert executor._response_channel.responses == {}
            executor.dist.tp_gather.assert_not_called()
            executor.dist.allgather.assert_not_called()
        else:
            (response,) = executor._response_channel.responses[1]
            assert response.error_msg == "Disagg KV cache transfer error"
            response_gather.assert_called_once()
        if failure_rank == 0 and should_flush:
            assert not request.is_finished_due_to_cancellation
            executor.resource_manager.free_resources.assert_called_once_with(request)
        else:
            executor.resource_manager.free_resources.assert_not_called()
        raise _ReachedIdleFetch

    monkeypatch.setattr(
        executor.executor_request_queue, "get_from_request_queue", stop_before_blocking
    )
    loop = executor._executor_loop_overlap if overlap else executor._executor_loop
    with pytest.raises(_ReachedIdleFetch):
        loop()

    expected = ["copy_wait"] if failure_rank is not None else []
    if overlap:
        expected.append("copy_wait")
    if should_flush:
        expected.append("flush")
    if failure_rank == 0 and should_flush:
        expected.append("terminate")
    assert events == expected + ["fetch"]


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("bridge_enabled", [False, True])
def test_existing_compute_request_keeps_post_batch_cancellation(
    executor: PyExecutor, context: bool, bridge_enabled: bool
) -> None:
    """Model-active work retains the existing post-sampler cancel boundary."""
    request = _request(1, context=context)
    executor.kv_cache_transceiver._fp4_mla_bridge_enabled = bridge_enabled
    request.state = (
        LlmRequestState.CONTEXT_INIT if context else LlmRequestState.GENERATION_IN_PROGRESS
    )
    batch = ScheduledRequests()
    if context:
        batch.context_requests_chunking = [request]
    else:
        batch.generation_requests = [request]
    executor.previous_batch = BatchState(
        scheduled_requests=batch, sample_state=SampleStateTorch(requests=[])
    )
    executor.active_requests = [request]
    _abort(executor, request)

    assert executor._cancel_waiting_disagg_requests([]) == []

    assert executor.active_requests == [request]
    assert executor.canceled_req_ids == [1]
    assert not request.is_finished
    assert executor._pending_transfer_responses == []
    executor.resource_manager.free_resources.assert_not_called()


@pytest.mark.parametrize("context, bridge_enabled", [(False, True), (True, True), (True, False)])
def test_model_request_cancels_between_batches_without_another_forward(
    executor: PyExecutor, context: bool, bridge_enabled: bool
) -> None:
    """A consumed abort needs no further model work to publish its result."""
    request, healthy = _request(1, context=context, lora_task_id=0), _request(2)
    request.state = (
        LlmRequestState.CONTEXT_INIT if context else LlmRequestState.GENERATION_IN_PROGRESS
    )
    if context:
        request.context_chunk_size = 4
        request.move_to_next_context_chunk()
        assert request.context_remaining_length == 4
        executor.kv_cache_transceiver._wait_reqs[1] = request
    executor.kv_cache_transceiver._fp4_mla_bridge_enabled = bridge_enabled
    executor.active_requests = [request, healthy]
    _abort(executor, request, repeats=2)

    assert executor._cancel_waiting_disagg_requests([]) == []

    assert executor.active_requests == [healthy]
    assert not healthy.is_finished
    assert executor.canceled_req_ids == []
    assert executor.kv_cache_transceiver._wait_reqs == {}
    assert executor._pending_response_terminations == [request]
    executor.resource_manager.free_resources.assert_not_called()
    _assert_released_once(executor, request)
    _abort(executor, request)
    executor._cancel_waiting_disagg_requests([])
    _assert_released_once(executor, request)


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("child", [False, True])
def test_scheduler_excludes_cancelled_model_work_without_releasing_its_roots(
    executor: PyExecutor, context: bool, child: bool
) -> None:
    """Stop future batches by executor ID, including cancellation of a parent."""
    parent = _request(1, context=context, num_return_sequences=2 if child else 1)
    if context:
        parent.state = LlmRequestState.CONTEXT_INIT
    if child:
        parent.create_child_request(3)
        (request,) = parent.child_requests
    else:
        request = parent
    request.state = (
        LlmRequestState.CONTEXT_INIT if context else LlmRequestState.GENERATION_IN_PROGRESS
    )
    healthy = _request(2)
    healthy.state = LlmRequestState.GENERATION_IN_PROGRESS
    executor.active_requests = [request, healthy]
    executor.kv_cache_manager = SimpleNamespace()
    executor.inflight_req_ids = set()
    executor.scheduler = Mock()
    executor.scheduler.schedule_request.side_effect = _ReachedScheduler
    _abort(executor, parent)

    with pytest.raises(_ReachedScheduler):
        executor._schedule()

    executor.scheduler.schedule_request.assert_called_once_with([healthy], set())
    assert executor.active_requests == [request, healthy]
    assert executor.canceled_req_ids == [parent.py_request_id]
    assert not request.is_finished
    assert executor._pending_transfer_responses == []
    executor.resource_manager.free_resources.assert_not_called()
    executor.sampler.release_request.assert_not_called()


@pytest.mark.parametrize("completed", [False, True], ids=["cancelled", "already-complete"])
@pytest.mark.parametrize("context", [False, True])
def test_cancelled_overlap_request_retires_at_the_existing_post_sampler_boundary(
    executor: PyExecutor, completed: bool, context: bool
) -> None:
    """Update physical model resources before finishing the retained request."""
    request = _request(1, context=context, lora_task_id=0)
    request.state = (
        LlmRequestState.CONTEXT_INIT if context else LlmRequestState.GENERATION_IN_PROGRESS
    )
    executor.active_requests = [request]
    batch = ScheduledRequests()
    if context:
        request.context_chunk_size = 4
        request.move_to_next_context_chunk()
        batch.context_requests_chunking = [request]
    else:
        batch.generation_requests = [request]
    executor.previous_batch = BatchState(
        scheduled_requests=batch,
        sample_state=SampleStateTorch(requests=[]),
        guided_failures=[],
    )
    executor._is_kv_manager_v2 = True
    executor.enable_joint_kv_cache_reuse = False
    executor.kv_cache_manager = Mock(
        spec=["update_resources", "is_preemption_pending", "has_pending_preemption"]
    )
    executor.kv_cache_manager.is_preemption_pending.return_value = False
    executor.kv_cache_manager.has_pending_preemption.return_value = False
    executor.resource_manager = Mock(spec=["free_resources", "update_resources"])
    executor.model_engine.attn_metadata = object()
    executor.model_engine.kv_cache_dtype_byte_size = 2
    executor.enable_early_first_token_response = False
    executor.enable_iter_perf_stats = False
    executor.disable_overlap_scheduler = False
    executor.force_terminate_ctx_for_partial_reuse = False
    executor.perf_manager = Mock()
    executor.iter_counter = 0
    executor.stream_interval = 1
    events = []
    executor.kv_cache_manager.update_resources.side_effect = lambda *args: events.append("update")
    executor.resource_manager.free_resources.side_effect = lambda req: events.append("free")
    _abort(executor, request)

    executor._cancel_waiting_disagg_requests([])
    assert not request.is_finished
    executor.resource_manager.free_resources.assert_not_called()

    # The previous sampler has now settled. Its terminal result, if any,
    # precedes the cancellation boundary and must remain the result.
    if completed:
        request.finish_by_reason(FinishReason.LENGTH)
    executor._process_previous_batch()

    assert events == ["update", "free"]
    assert request.is_finished_due_to_length is completed
    assert request.is_finished_due_to_cancellation is not completed
    assert executor.active_requests == []
    assert executor.canceled_req_ids == []
    assert len(executor._response_channel.responses[1]) == 1
    executor.resource_manager.free_resources.assert_called_once_with(request)
    executor.sampler.release_request.assert_called_once_with(request)
    executor._cancel_waiting_disagg_requests([])
    executor._flush_pending_transfer_responses()
    assert len(executor._response_channel.responses[1]) == 1
    executor.resource_manager.free_resources.assert_called_once_with(request)


def _prepare_generation_harvest(executor: PyExecutor, requests: list[LlmRequest]) -> None:
    """Use native sampling with controlled resource and response boundaries."""
    for slot, request in enumerate(requests):
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        request.py_seq_slot = slot
        request.py_batch_idx = slot
    executor.active_requests = requests.copy()
    batch = ScheduledRequests()
    batch.generation_requests = requests
    state = SampleStateTorch(
        requests=requests,
        device=None,
        host=SampleStateTensorsHostTorch(
            new_tokens=torch.tensor(
                [40 + slot for slot in range(len(requests))], dtype=torch.int32
            ),
            finish_reasons=None,
            first_finish_reasons=None,
            single_step_greedy=True,
        ),
    )
    executor.previous_batch = BatchState(
        scheduled_requests=batch, sample_state=state, guided_failures=[]
    )
    sampler = object.__new__(TorchSampler)
    sampler.max_seq_len = 1024
    sampler._track_pending_steps = False
    executor.sampler.update_requests = sampler.update_requests
    executor._is_kv_manager_v2 = True
    executor.enable_joint_kv_cache_reuse = False
    executor.kv_cache_manager = Mock(
        spec=["update_resources", "is_preemption_pending", "has_pending_preemption"]
    )
    executor.kv_cache_manager.is_preemption_pending.return_value = False
    executor.kv_cache_manager.has_pending_preemption.return_value = False
    executor.resource_manager = Mock(spec=["free_resources", "update_resources"])
    executor.model_engine.attn_metadata = object()
    executor.model_engine.kv_cache_dtype_byte_size = 2
    executor.enable_early_first_token_response = False
    executor.enable_iter_perf_stats = False
    executor.disable_overlap_scheduler = False
    executor.force_terminate_ctx_for_partial_reuse = False
    executor.perf_manager = Mock()
    executor.iter_counter = 0
    executor.stream_interval = 1
    executor.max_draft_len = 0


def test_deferred_lora_generation_resumes_from_harvested_host_token(executor: PyExecutor) -> None:
    """Keep adapter retention without treating a resumed request as overlapping."""
    held = _request(1, lora_task_id=0)
    deferred = _request(2, lora_task_id=1)
    continuing = _request(3)
    _prepare_generation_harvest(executor, [held, deferred, continuing])
    executor.inflight_req_ids = set()
    executor.enable_batch_waiting = False
    executor._get_ctx_mla_kv_len_cap = lambda: None
    executor.scheduler = Mock()
    executor.scheduler.schedule_request.side_effect = lambda requests, _: SchedulerOutput(
        [], [], list(requests), [], [], len(requests)
    )
    # As the base scheduler: no request lost its KV pages to a failed remote cache level.
    executor.scheduler.take_lost_kv_requests.return_value = []
    _abort(executor, held)

    current_batch, _, _ = executor._schedule()
    assert current_batch.generation_requests == [continuing]
    assert deferred.py_batch_idx == 1  # Its previous sample has not been harvested.
    executor._update_requests(executor.previous_batch.sample_state)
    executor._process_previous_batch(current_batch)

    assert deferred.get_last_tokens(0) == 41
    assert deferred.max_beam_num_tokens == 9
    assert deferred.py_batch_idx is None
    assert deferred.py_seq_slot == 1
    assert not deferred.is_finished
    assert continuing.py_batch_idx == 2
    executor.resource_manager.free_resources.assert_called_once_with(held)
    executor.sampler.release_request.assert_called_once_with(held)

    resumed_batch, _, _ = executor._schedule()
    assert resumed_batch.generation_requests == [deferred, continuing]
    inputs = DecoderInputPreparer(
        buffers=None,
        config=None,
        mapping=Mapping(),
        dist=None,
        model=None,
        get_runtime_tokens_per_gen_step=None,
        iter_states={},
        lora=None,
        use_beam_search=False,
        steady_gen_cache_supported=False,
    )
    host = HostInputs(
        use_mrope=False,
        mrope_dummy_seq_slot=0,
        extra_inputs=ExtraInputsCollector(),
        context_prompt_lookahead=None,
    )
    host.lists.generation_requests.extend(resumed_batch.generation_requests)
    # Only the continuously scheduled request has a pending device token.
    inputs._pack_generation_requests(
        host, frozenset(), torch.tensor([[[999], [999], [43]]], dtype=torch.int32)
    )
    assert host.lists.input_ids == [41]
    assert host.lists.previous_batch_indices == [2]
    assert host.lists.position_ids == [8, 9]
    assert host.lists.num_cached_tokens_per_seq == [8, 9]
    assert deferred.py_seq_slot == 1


@pytest.mark.parametrize("local_batch_ready", [False, True], ids=["empty", "adp-deferred"])
def test_overlap_marker_changes_only_after_previous_sample_is_harvested(
    executor: PyExecutor, monkeypatch: pytest.MonkeyPatch, local_batch_ready: bool
) -> None:
    """An idle pass retires markers; an ADP-delayed harvest must preserve them."""
    request = _request(1, lora_task_id=0)
    _prepare_generation_harvest(executor, [request])
    previous_batch = executor.previous_batch
    batch = ScheduledRequests()
    if local_batch_ready:
        executor.enable_attention_dp = True
        executor.dist.tp_size = executor.dist.world_size = 2
        executor.dist.mapping.tp_group = [0, 1]
        executor.dist.tp_gather = Mock(side_effect=lambda responses: [responses, []])
        batch.generation_requests = [request]
    executor._prepare_and_schedule_batch = Mock(side_effect=[(batch, None), (None, None)])
    executor._can_queue = Mock(return_value=(False, local_batch_ready))
    executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
    executor._can_pause_for_rebalance = Mock(return_value=False)
    executor._release_unused_connector_reservations = Mock()
    executor._terminate_recompute_paused_requests = Mock()
    executor._pause_recompute_paused_requests = Mock()
    executor._revert_unexecuted_alloc = Mock()
    executor._finalize_adp_dummy_allocation = Mock()
    executor._send_kv_async = Mock()
    executor._commit_kv_cache_stats = Mock()
    executor._flush_iter_stats_synced = Mock()
    executor._kv_connector_terminate_requests = Mock()
    executor._disagg_coordinator = Mock()
    executor.disagg.handle_errors_synced.return_value = False
    executor.device_id = 0
    executor._profiler = lambda: nullcontext(lambda: None)
    executor.hang_detector = MagicMock()
    executor._intake_window = threading.Event()
    executor._execution_stream_loop_armed = False
    executor._resource_governor_enabled = False
    executor.is_benchmark_disagg = False
    executor.speculation_gate = None
    executor.enable_kv_cache_events = False
    for target in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"tensorrt_llm._torch.pyexecutor.py_executor.{target}", Mock())

    executor._executor_loop_overlap()

    if local_batch_ready:
        assert request.py_batch_idx == 0
        assert request.max_beam_num_tokens == 8
        assert executor.previous_batch is previous_batch
        executor.resource_manager.update_resources.assert_not_called()
    else:
        assert request.py_batch_idx is None
        assert request.get_last_tokens(0) == 40
        assert request.max_beam_num_tokens == 9
        assert executor.previous_batch is None
        executor.resource_manager.update_resources.assert_called_once()
    assert request.py_seq_slot == 0
    executor.resource_manager.free_resources.assert_not_called()
    executor.sampler.release_request.assert_not_called()


@pytest.mark.parametrize(
    "unsupported",
    ["bridge_disabled", "cpp", "connector", "pp", "encoder", "mm_items", "mm", "mm_empty"],
)
def test_model_cancellation_keeps_unqualified_scheduler_paths_unchanged(
    executor: PyExecutor, unsupported: str
) -> None:
    """No new profile or connector path inherits this admission change."""
    request = _request(1)
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    executor.active_requests = [request]
    executor.kv_cache_manager = SimpleNamespace()
    executor.inflight_req_ids = set()
    executor.scheduler = Mock()
    executor.scheduler.schedule_request.side_effect = _ReachedScheduler
    if unsupported == "bridge_disabled":
        executor.kv_cache_transceiver._fp4_mla_bridge_enabled = False
    elif unsupported == "cpp":
        executor.kv_cache_transceiver = object.__new__(BindKvCacheTransceiver)
    elif unsupported == "connector":
        executor.kv_connector_manager = Mock(capacity_only=False)
    elif unsupported == "pp":
        executor.dist.pp_size = 2
    elif unsupported == "encoder":
        executor.is_encoder_decoder = True
    elif unsupported == "mm_items":
        executor._mm_encoder_item_scheduling_enabled = True
    elif unsupported == "mm":
        request.py_multimodal_data = {"image": object()}
    else:
        request.py_multimodal_data = {}
    _abort(executor, request)

    assert executor._cancel_waiting_disagg_requests([]) == []
    with pytest.raises(_ReachedScheduler):
        executor._schedule()

    executor.scheduler.schedule_request.assert_called_once_with([request], set())
    assert executor.active_requests == [request]
    assert executor.canceled_req_ids == [1]
    assert not request.is_finished
    assert executor._pending_transfer_responses == []
    executor.resource_manager.free_resources.assert_not_called()


def test_model_cancel_failure_keeps_ownership_and_retries_without_new_work(
    executor: PyExecutor,
) -> None:
    """A failed cancellation call is not permission to release the request."""
    request = _request(1, context=True)
    request.state = LlmRequestState.CONTEXT_INIT
    executor.active_requests = [request]
    transceiver = executor.kv_cache_transceiver
    cancel = transceiver.cancel_request
    transceiver.cancel_request = Mock(side_effect=RuntimeError("control endpoint unavailable"))
    _abort(executor, request)

    executor._cancel_waiting_disagg_requests([])

    assert executor.active_requests == [request]
    assert executor.canceled_req_ids == [1]
    assert executor._cancelled_disagg_model_request_ids() == {1}
    assert not request.is_finished
    assert executor._pending_transfer_responses == []
    executor.resource_manager.free_resources.assert_not_called()
    transceiver.cancel_request = cancel
    executor._cancel_waiting_disagg_requests([])
    _assert_released_once(executor, request)


@pytest.mark.parametrize(
    "case",
    ["generation", "published_generation", "context_send", "context_receive", "context_first"],
)
def test_disabled_bridge_preserves_unqualified_cancellation_paths(
    executor: PyExecutor, case: str
) -> None:
    """Session-free CTX support does not enable other unqualified paths."""
    request = _request(1, context=case.startswith("context"))
    transceiver = executor.kv_cache_transceiver
    transceiver._fp4_mla_bridge_enabled = False
    session = Mock()
    if case == "published_generation":
        _published_receive(executor, request)
    elif case == "context_send":
        transceiver._send_sessions[1] = session
    elif case == "context_receive":
        transceiver._recv_sessions[1] = session
    elif case == "context_first":
        request.py_disaggregated_params.schedule_style = DisaggScheduleStyle.CONTEXT_FIRST
    if request.is_context_only_request:
        transceiver._wait_reqs[1] = request
    transceiver.cancel_request = Mock(wraps=transceiver.cancel_request)
    executor.active_requests = [request]
    _abort(executor, request)

    assert executor._cancel_waiting_disagg_requests([request]) == [request]

    assert executor.active_requests == [request]
    assert executor.canceled_req_ids == [1]
    assert not request.is_finished
    assert executor._pending_transfer_responses == []
    transceiver.cancel_request.assert_not_called()
    if request.is_context_only_request:
        assert transceiver._wait_reqs == {1: request}
    executor.resource_manager.free_resources.assert_not_called()


@pytest.mark.parametrize("runtime", ["cpp", "python"])
def test_idle_cancellation_preserves_unhandled_queued_abort(
    executor: PyExecutor, runtime: str
) -> None:
    """Keep queued cancellation pending even when an eligible CTX wait retires."""
    transceiver = executor.kv_cache_transceiver
    transceiver._fp4_mla_bridge_enabled = False
    if runtime == "cpp":
        transceiver = object.__new__(BindKvCacheTransceiver)
        transceiver.cancel_request = Mock()
    executor.kv_cache_transceiver = transceiver
    executor.disagg._transceiver = transceiver
    request = _request(1, context=True)
    executor.active_requests = [request]
    executor.waiting_queue.add_request(RequestQueueItem(2))
    _abort(executor, request)
    _abort(executor, _request(2))

    remaining = executor._cancel_waiting_disagg_requests([request])

    assert [item.id for item in executor.waiting_queue] == [2]
    if runtime == "cpp":
        assert remaining == executor.active_requests == [request]
        assert executor.canceled_req_ids == [1, 2]
        assert not request.is_finished
        transceiver.cancel_request.assert_not_called()
        assert executor._pending_transfer_responses == []
        executor.resource_manager.free_resources.assert_not_called()
    else:
        assert remaining == executor.active_requests == []
        assert executor.canceled_req_ids == [2]
        _assert_released_once(executor, request)


def test_connector_owned_request_stays_on_existing_cancellation_path(executor: PyExecutor) -> None:
    """Connector cancellation remains governed by its existing path."""
    request = _request(1)
    executor.kv_connector_manager = Mock(capacity_only=False)
    executor.active_requests = [request]
    _abort(executor, request)

    assert executor._cancel_waiting_disagg_requests([request]) == [request]

    assert executor.active_requests == [request]
    assert executor.canceled_req_ids == [1]
    assert not request.is_finished
    executor.kv_connector_manager.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()


def test_cancel_does_not_rewrite_an_already_finished_request(executor: PyExecutor) -> None:
    """Finished CTX work stays on its existing response and cleanup path."""
    request = _request(1, context=True)
    request.finish_by_reason(FinishReason.LENGTH)
    executor.active_requests = [request]
    _abort(executor, request)

    assert executor._cancel_waiting_disagg_requests([request]) == [request]

    assert executor.active_requests == [request]
    assert request.is_finished_due_to_length
    assert not request.is_finished_due_to_cancellation
    assert executor._pending_transfer_responses == []
    assert executor._pending_response_terminations == []
    executor.resource_manager.free_resources.assert_not_called()


def _published_receive(
    executor: PyExecutor,
    request: LlmRequest,
    *,
    watchdog: RetirementWatchdog | None = None,
) -> tuple[RxSession, Mock]:
    """Publish real receive claims against controlled network and AUX storage.

    Args:
        executor: Executor holding the receive session and request.
        request: Request whose destination is exposed to one writer.
        watchdog: Optional real deadline and logical-outcome arbiter.

    Returns:
        The receive session and its controlled AUX allocation boundary.
    """
    aux = Mock()
    aux.alloc_slot.return_value = SimpleNamespace(id=0)
    if (
        watchdog is None
        and getattr(executor.kv_cache_transceiver, "_context_first_ownership", False) is True
    ):
        watchdog = executor.kv_cache_transceiver._transfer_worker._retirement_watchdog
    receiver = SimpleNamespace(
        _enforce_physical_ownership=True,
        _registrar=SimpleNamespace(
            self_rank_info=SimpleNamespace(instance_name="gen", instance_rank=0)
        ),
        setup_session=Mock(),
        clear_session=Mock(),
        send_cancel_to_senders=Mock(
            side_effect=lambda _rid, endpoints, *, nonblocking=False: set(endpoints)
        ),
    )
    session = RxSession(
        request.py_request_id,
        request.py_disaggregated_params,
        receiver,
        aux,
        timeout_s=1.0 if watchdog is not None else None,
        retirement_watchdog=watchdog,
    )

    def publish(task: KVRecvTask) -> None:
        """Seal one writer and record publication of the receive task."""
        task.expected_transfers = 1
        task._kv_result_report_tokens[0] = _KV_RESULT_REPORT_TOKEN
        published: set[int] = set()
        assert session.try_begin_transfer(
            task.slice_id,
            {"ctx"},
            writer_cohort={0},
            publish=lambda: published.add(0),
            published_writers=published,
        )

    receiver.dispatch_task = publish
    session.receive(
        Chunk(
            block_ids_per_layer_groups=[],
            kind_per_layer_group=[],
            token_range=TokenRange(0, 8),
            is_last=True,
        )
    )
    transceiver = executor.kv_cache_transceiver
    transceiver._recv_sessions[request.py_request_id] = session
    transceiver._recv_reqs[request.py_request_id] = request
    transceiver._ever_had_recv_session = True
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    request.py_kv_transfer_start_time = 12.0
    return session, aux


@pytest.mark.parametrize("boundary", ["early", "post_batch"])
@pytest.mark.parametrize("settled_before_cancel", [False, True])
def test_parked_executor_failure_precedes_generation_cancellation(
    executor: PyExecutor, boundary: str, settled_before_cancel: bool
) -> None:
    """Preserve an executor error while cancellation progresses physical retirement.

    Args:
        executor: Real executor cancellation and error paths with controlled resources.
        boundary: Executor cancellation boundary to exercise.
        settled_before_cancel: Whether physical evidence arrives before cancellation.
    """
    request = _request(1)
    executor.active_requests = [request]
    executor._fatal_error = None
    session, aux = _published_receive(executor, request)
    message = "executor failure before cancellation"
    executor._handle_errors(message, requests=[request], charge_budget=False)
    assert executor._held_transfer_failures() == {id(request): (request, message)}
    assert not request.is_finished
    assert executor._response_channel.responses == {}
    executor.resource_manager.free_resources.assert_not_called()

    def settle() -> None:
        """Supply safe completion evidence for both retained receive resources."""
        session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
        session.process_aux_agent_result(0, AgentResult.SUCCESS)

    def cancel() -> None:
        """Use the selected production cancellation boundary."""
        if boundary == "early":
            executor._cancel_waiting_disagg_requests([])
        else:
            executor._handle_canceled_requests()

    if settled_before_cancel:
        settle()
    _abort(executor, request, repeats=2)
    cancel()
    assert executor.active_requests == [request]
    assert set(executor.canceled_req_ids) == {1}
    assert not request.is_finished_due_to_cancellation
    assert executor._response_channel.responses == {}
    assert executor.disagg._retiring_gen_receives == {}
    executor.resource_manager.free_resources.assert_not_called()
    if not settled_before_cancel:
        executor._sweep_transfer_held_failures()
        assert executor._held_transfer_failures() == {id(request): (request, message)}
        assert executor._response_channel.responses == {}
        aux.free_slot.assert_not_called()
        executor.resource_manager.free_resources.assert_not_called()
        settle()
    executor.disagg.reap_gen_receives(0)
    cancel()
    executor._sweep_transfer_held_failures()
    executor._flush_pending_transfer_responses()

    assert executor.active_requests == []
    assert executor._held_transfer_failures() == {}
    (response,) = executor._response_channel.responses[1]
    assert response.error_msg == message
    assert not request.is_finished_due_to_cancellation
    executor.resource_manager.free_resources.assert_called_once_with(request)
    aux.free_slot.assert_called_once_with(0)
    cancel()
    executor.disagg.reap_gen_receives(0)
    executor._sweep_transfer_held_failures()
    executor._flush_pending_transfer_responses()
    assert len(executor._response_channel.responses[1]) == 1
    executor.resource_manager.free_resources.assert_called_once_with(request)
    aux.free_slot.assert_called_once_with(0)


def test_parked_native_failure_waits_for_physical_cancel_completion(
    executor: PyExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A C++ cancellation acknowledgment is not proof that its receive has retired.

    Args:
        executor: Real executor cancellation and parked-failure handling.
        monkeypatch: Enables the native in-flight cancellation policy.
    """
    transceiver = object.__new__(BindKvCacheTransceiver)
    transceiver.impl = SimpleNamespace(cancel_request=Mock(return_value=True))
    transceiver._supports_inflight_request_cancellation = True
    executor.kv_cache_transceiver = transceiver
    executor.disagg._transceiver = transceiver
    monkeypatch.setattr(coordinator_mod, "is_disagg_inflight_cancel_enabled", lambda: True)
    request = _request(1)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    executor.active_requests = [request]
    executor._fatal_error = None
    message = "executor failure before native receive cancellation"
    executor._handle_errors(message, requests=[request], charge_budget=False)
    _abort(executor, request)

    executor._handle_canceled_requests()
    executor._sweep_transfer_held_failures()

    transceiver.impl.cancel_request.assert_called_once_with(request)
    assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert executor.canceled_req_ids == [1]
    assert executor.active_requests == [request]
    assert executor._held_transfer_failures() == {id(request): (request, message)}
    assert executor._response_channel.responses == {}
    executor.resource_manager.free_resources.assert_not_called()


@pytest.mark.parametrize("context_first", [False, True], ids=["fp4-bridge", "bf16-context-first"])
@pytest.mark.parametrize("boundary", ["early", "post_batch"])
@pytest.mark.parametrize("expires", [False, True], ids=["settles", "grace-expires"])
@pytest.mark.parametrize("transfer_error", [False, True], ids=["in-progress", "transfer-error"])
def test_parked_failure_closes_unpublished_receive_admission(
    executor: PyExecutor,
    boundary: str,
    expires: bool,
    transfer_error: bool,
    context_first: bool,
) -> None:
    """Keep the original failure while an unpublished CUDA claim drains within its grace.

    Args:
        executor: Real executor error and cancellation paths with controlled resources.
        boundary: Executor cancellation boundary to exercise.
        expires: Whether the local access remains unproven until its grace expires.
        transfer_error: Whether the request has also entered the transfer-error state.
        context_first: Whether to include the earlier executor error in the BF16 vote.
    """
    request = _request(1)
    if context_first:
        _enable_context_first(executor, request)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    executor.active_requests = [request]
    executor._fatal_error = None
    clock = [0.0]
    containment = Mock()
    watchdog = RetirementWatchdog(containment, clock=lambda: clock[0])
    transceiver = executor.kv_cache_transceiver
    transceiver._transfer_worker._retirement_watchdog = watchdog
    transceiver.request_and_receive_async = Mock()
    ready = Mock(spec=["query"])
    ready.query.return_value = False
    executor.disagg._defer_unready_gen_receives = True
    executor.disagg._unpublished_gen_receives[1] = (request, ready)
    message = "executor failure before unpublished receive cancellation"
    executor._handle_errors(message, requests=[request], charge_budget=False)
    assert executor._held_transfer_failures() == {id(request): (request, message)}

    if transfer_error:
        request.state = LlmRequestState.DISAGG_TRANS_ERROR

    def cancel() -> None:
        """Use the selected production cancellation boundary."""
        if boundary == "early":
            executor._cancel_waiting_disagg_requests([])
        else:
            executor._handle_canceled_requests()

    _abort(executor, request, repeats=2)
    cancel()
    assert executor.disagg._unpublished_gen_receives == {}
    assert executor.disagg._retiring_gen_receives == {}
    assert transceiver._recv_reqs == {1: request}
    assert transceiver.has_inflight_transfer(request)
    assert executor.active_requests == [request]
    assert not request.is_finished_due_to_cancellation
    assert executor._response_channel.responses == {}
    executor.resource_manager.free_resources.assert_not_called()

    clock[0] = 0.75
    _abort(executor, request, repeats=2)
    cancel()
    executor.disagg.poll_gen_transfers()
    executor._sweep_transfer_held_failures()
    assert len(watchdog._owners) == 1
    assert watchdog.fatal is None
    assert executor._held_transfer_failures() == {id(request): (request, message)}
    assert executor._response_channel.responses == {}
    transceiver.request_and_receive_async.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()

    clock[0] = 1.01 if expires else 0.9
    watchdog.progress()
    ready.query.return_value = True
    cancel()
    executor.disagg.poll_gen_transfers()
    if executor.disagg.supports_group_receive_cancellation:
        # Apply group outcomes after polling, before the parked-failure sweep
        # and any subsequent model admission, as the executor does.
        executor._cancel_waiting_disagg_requests([])
    executor._sweep_transfer_held_failures()
    executor._flush_pending_transfer_responses()
    if expires:
        assert watchdog.fatal is not None
        assert watchdog.fatal.started_at == 0.0
        assert watchdog.fatal.deadline == 1.0
        containment.assert_called_once_with(watchdog.fatal)
        assert executor.active_requests == [request]
        assert executor._held_transfer_failures() == {id(request): (request, message)}
        assert transceiver._recv_reqs == {1: request}
        assert executor._response_channel.responses == {}
        executor.resource_manager.free_resources.assert_not_called()
    else:
        containment.assert_not_called()
        assert executor.active_requests == []
        assert executor._held_transfer_failures() == {}
        assert transceiver._recv_reqs == {}
        assert not transceiver._cancelled_receive_fences
        (response,) = executor._response_channel.responses[1]
        assert response.error_msg == message
        executor.resource_manager.free_resources.assert_called_once_with(request)
        watchdog.require_retired()
        watchdog.stop()
        cancel()
        executor.disagg.poll_gen_transfers()
        executor._sweep_transfer_held_failures()
        executor._flush_pending_transfer_responses()
        assert len(executor._response_channel.responses[1]) == 1
        executor.resource_manager.free_resources.assert_called_once_with(request)
    assert not request.is_finished_due_to_cancellation
    assert executor.disagg._retiring_gen_receives == {}
    transceiver.request_and_receive_async.assert_not_called()


@pytest.mark.parametrize("failure_first", [False, True], ids=["cancel-first", "failure-first"])
@pytest.mark.parametrize("poll_first", [False, True], ids=["cancel-retires", "poll-retires"])
@pytest.mark.parametrize("boundary", ["early", "post_batch"])
def test_receive_outcome_does_not_depend_on_cancellation_cleanup_order(
    executor: PyExecutor, failure_first: bool, poll_first: bool, boundary: str
) -> None:
    """Preserve the committed result through either cancellation and retirement path."""
    request = _request(1)
    executor.active_requests = [request]
    executor._fatal_error = None
    executor.perf_manager = Mock()
    executor.iter_counter = 0
    executor.stream_interval = 1
    executor.force_terminate_ctx_for_partial_reuse = False
    watchdog = RetirementWatchdog(Mock(), clock=lambda: 0.0)
    session, aux = _published_receive(executor, request, watchdog=watchdog)
    handle = TaskHandle(session, session._kv_tasks[0], token_end=8)
    transceiver = executor.kv_cache_transceiver
    transceiver._ever_had_recv_session = True
    transceiver._mapping = SimpleNamespace(enable_attention_dp=True, pp_size=1)
    transceiver._dist = executor.dist

    def cancel() -> None:
        """Run the selected real executor cancellation boundary."""
        if boundary == "early":
            executor._cancel_waiting_disagg_requests([])
        else:
            executor._handle_canceled_requests()

    if failure_first:
        session.process_kv_agent_result(0, 0, True, AgentResult.FAILED, _KV_RESULT_REPORT_TOKEN)
    _abort(executor, request)
    cancel()
    assert not session.resources_drained()
    assert executor.canceled_req_ids == ([1] if failure_first else [])
    executor.resource_manager.free_resources.assert_not_called()
    aux.free_slot.assert_not_called()
    if not failure_first:
        session.process_kv_agent_result(0, 0, True, AgentResult.FAILED, _KV_RESULT_REPORT_TOKEN)

    expected_status = SessionStatus.ERROR if failure_first else SessionStatus.CANCELLED
    expected_outcome = Failed if failure_first else Cancelled
    assert session.status is expected_status
    assert isinstance(handle.poll(), expected_outcome)
    cancel()
    if failure_first:
        assert executor.active_requests == [request]
        assert not request.is_finished
        assert executor._response_channel.responses == {}
    else:
        if boundary == "post_batch":
            executor._handle_responses()
        executor._flush_pending_transfer_responses()
        assert executor.active_requests == []
        assert request.is_finished_due_to_cancellation
        (response,) = executor._response_channel.responses[request.py_request_id]
        assert response.error_msg is None
        assert executor.disagg._retiring_gen_receives == {1: request}
    aux.free_slot.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()

    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    assert session.resources_drained()
    if poll_first:
        executor.disagg.reap_gen_receives(0)
    elif not failure_first:
        assert executor.disagg.request_cancellation(request)
    cancel()
    assert executor.canceled_req_ids == []
    assert session.status is expected_status
    assert isinstance(handle.poll(), expected_outcome)
    if failure_first:
        assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
        assert not request.is_finished_due_to_cancellation
        executor.disagg.reap_gen_receives(0)
    else:
        executor.disagg.reap_gen_receives(0)
    executor._flush_pending_transfer_responses()

    assert executor.active_requests == []
    (response,) = executor._response_channel.responses[request.py_request_id]
    assert (response.error_msg is not None) is failure_first
    assert request.is_finished_due_to_cancellation is not failure_first
    executor.resource_manager.free_resources.assert_called_once_with(request)
    aux.free_slot.assert_called_once_with(0)
    assert transceiver._recv_sessions == transceiver._recv_reqs == {}
    cancel()
    executor.disagg.reap_gen_receives(0)
    executor._flush_pending_transfer_responses()
    assert len(executor._response_channel.responses[request.py_request_id]) == 1
    executor.resource_manager.free_resources.assert_called_once_with(request)
    aux.free_slot.assert_called_once_with(0)
    watchdog.require_retired()
    watchdog.stop()


@pytest.mark.parametrize("last_resource", ["kv", "aux"])
def test_worker_cancel_retains_published_kv_and_aux_until_both_settle(
    executor: PyExecutor, last_resource: str
) -> None:
    """Respond immediately, but neither KV nor AUX alone permits destination reuse."""
    request = _request(1)
    executor.active_requests = [request]
    session, aux = _published_receive(executor, request)
    handle = TaskHandle(session, session._kv_tasks[0], token_end=8)
    _abort(executor, request, repeats=2)

    def complete(resource: str) -> None:
        """Deliver terminal evidence for one of the session's two resources."""
        if resource == "kv":
            session.process_kv_agent_result(
                0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN
            )
        else:
            session.process_aux_agent_result(0, AgentResult.SUCCESS)

    for resource in (None, "aux" if last_resource == "kv" else "kv"):
        if resource is not None:
            complete(resource)
        executor._cancel_waiting_disagg_requests([])
        executor._flush_pending_transfer_responses()
        executor.disagg.poll_gen_transfers()
        assert executor.active_requests == []
        assert executor.canceled_req_ids == []
        assert executor.disagg._retiring_gen_receives == {1: request}
        assert request.py_kv_transfer_start_time == 12.0
        assert session.status is SessionStatus.CANCELLED
        assert isinstance(handle.poll(), Cancelled)
        (response,) = executor._response_channel.responses[request.py_request_id]
        assert response.error_msg is None
        assert request.is_finished_due_to_cancellation
        executor.sampler.release_request.assert_not_called()
        executor.resource_manager.free_resources.assert_not_called()
        aux.free_slot.assert_not_called()

    complete(last_resource)
    executor.disagg.poll_gen_transfers()
    _assert_released_once(executor, request)
    assert executor.disagg._retiring_gen_receives == {}
    assert isinstance(handle.poll(), Cancelled)
    assert not executor.kv_cache_transceiver._recv_sessions
    aux.free_slot.assert_called_once_with(0)
    complete(last_resource)
    _abort(executor, request)
    executor._cancel_waiting_disagg_requests([])
    executor.disagg.poll_gen_transfers()
    executor._flush_pending_transfer_responses()
    executor.resource_manager.free_resources.assert_called_once_with(request)


def test_settled_receive_waits_for_terminal_response_before_request_release(
    executor: PyExecutor,
) -> None:
    """Receive polling cannot remove response routing before cancellation is delivered."""
    request = _request(1)
    executor.active_requests = [request]
    session, _ = _published_receive(executor, request)
    _abort(executor, request)
    executor._cancel_waiting_disagg_requests([])
    session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
    session.process_aux_agent_result(0, AgentResult.SUCCESS)

    executor.disagg.poll_gen_transfers()

    assert executor._response_channel.responses == {}
    assert executor.disagg._retiring_gen_receives == {1: request}
    executor.resource_manager.free_resources.assert_not_called()
    executor._flush_pending_transfer_responses()
    (response,) = executor._response_channel.responses[1]
    assert response.error_msg is None
    executor.resource_manager.free_resources.assert_not_called()
    executor.disagg.poll_gen_transfers()
    _assert_released_once(executor, request)


@pytest.mark.parametrize("capacity_only", [False, True])
def test_connector_save_waits_for_receive_retirement(
    executor: PyExecutor, capacity_only: bool
) -> None:
    """A KV-moving connector cannot read a cancelled destination before it settles.

    Args:
        executor: Real cancellation path with controlled external resources.
        capacity_only: Whether the connector only manages capacity, without KV I/O.
    """
    request = _request(1)
    executor.active_requests = [request]
    session, aux = _published_receive(executor, request)
    connector = SimpleNamespace(
        capacity_only=capacity_only,
        release_unstarted_prefix_loads=Mock(),
        defer_load_termination=Mock(return_value=False),
        has_outstanding_save=Mock(return_value=False),
        request_finished=Mock(return_value=False),
    )
    executor.kv_connector_manager = connector
    executor.kv_cache_manager = SimpleNamespace(get_cache_indices=Mock(return_value=[3, 4]))
    _abort(executor, request)

    executor._handle_canceled_requests()

    assert request.is_finished_due_to_cancellation == capacity_only
    assert executor.canceled_req_ids == ([] if capacity_only else [1])
    connector.request_finished.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()
    aux.free_slot.assert_not_called()
    session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
    executor._handle_canceled_requests()
    connector.request_finished.assert_not_called()
    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    executor._handle_canceled_requests()

    assert request.is_finished_due_to_cancellation
    assert executor.canceled_req_ids == []
    if capacity_only:
        connector.request_finished.assert_not_called()
        assert executor.disagg.has_local_retiring_gen_receives
    else:
        connector.request_finished.assert_called_once_with(request, [3, 4], None)
        assert not executor.kv_cache_transceiver.has_inflight_transfer(request)
        assert not executor.disagg.has_local_retiring_gen_receives
        aux.free_slot.assert_called_once_with(0)


@pytest.mark.parametrize("handoff", [False, True], ids=["logical-cancel", "fence-handoff"])
@pytest.mark.parametrize("error_type", [RuntimeError, TypeError])
def test_new_cancellation_boundary_preserves_unexpected_errors(
    executor: PyExecutor, handoff: bool, error_type: type[Exception]
) -> None:
    """Retry operational errors without hiding bugs or dropping an unpublished fence.

    Args:
        executor: Real coordinator with controlled transceiver failure injection.
        handoff: Whether to enter through the physical cancellation handoff.
        error_type: Operational or unexpected error injected at the new boundary.
    """
    request = _request(1)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    ready = Mock(spec=["query"])
    executor.disagg._unpublished_gen_receives[1] = (request, ready)
    executor.kv_cache_transceiver.cancel_gen_logically = Mock(
        side_effect=error_type("cancellation error")
    )
    executor.kv_cache_transceiver.cancel_request = Mock()
    cancel = (
        executor.disagg.request_cancellation if handoff else executor.disagg.cancel_gen_logically
    )

    if error_type is RuntimeError:
        assert cancel(request) is (False if handoff else ReceiveCancellation.PENDING)
    else:
        with pytest.raises(error_type, match="cancellation error"):
            cancel(request)

    assert executor.disagg._unpublished_gen_receives == {1: (request, ready)}
    assert executor.disagg._retiring_gen_receives == {}
    executor.kv_cache_transceiver.cancel_request.assert_not_called()
    ready.query.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()
    if error_type is RuntimeError:
        executor.disagg._defer_unready_gen_receives = True
        ready.query.return_value = True
        executor.kv_cache_transceiver.request_and_receive_async = Mock()
        executor.disagg.poll_gen_transfers()
        assert executor.disagg._unpublished_gen_receives == {1: (request, ready)}
        executor.kv_cache_transceiver.request_and_receive_async.assert_not_called()
        executor.kv_cache_transceiver.cancel_request.assert_not_called()
        ready.query.assert_not_called()
        executor.resource_manager.free_resources.assert_not_called()


@pytest.mark.parametrize("handoff", [False, True], ids=["logical-cancel", "fence-handoff"])
def test_transient_receive_cancellation_handoff_recovers_without_publication(
    executor: PyExecutor, handoff: bool
) -> None:
    """Normal polls retry a failed ownership handoff without publishing the destination.

    Args:
        executor: Real cancellation and retirement path with controlled CUDA readiness.
        handoff: Whether the first cancellation enters through the physical handoff API.
    """
    request = _request(1)
    _enable_context_first(executor, request)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    executor.active_requests = [request]
    ready = Mock(spec=["query"])
    ready.query.return_value = True
    executor.disagg._defer_unready_gen_receives = True
    executor.disagg._unpublished_gen_receives[1] = (request, ready)
    transceiver = executor.kv_cache_transceiver
    real_cancel = transceiver.cancel_gen_logically
    transceiver.cancel_gen_logically = Mock(side_effect=RuntimeError("transient handoff"))
    transceiver.request_and_receive_async = Mock()
    _abort(executor, request)
    cancel = (
        executor.disagg.request_cancellation if handoff else executor.disagg.cancel_gen_logically
    )
    assert cancel(request) is (False if handoff else ReceiveCancellation.PENDING)
    assert executor.disagg._unpublished_gen_receives == {1: (request, ready)}
    ready.query.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()

    transceiver.cancel_gen_logically = real_cancel
    executor.disagg.poll_gen_transfers()
    assert request.py_kv_receive_cancel_outcome is ReceiveCancellation.CANCELLED
    assert executor.disagg._pending_gen_cancellation_intents == {}
    assert executor.disagg._unpublished_gen_receives == {}
    assert transceiver._cancelled_receive_fences == {}
    ready.query.assert_called_once()
    transceiver.request_and_receive_async.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()

    executor._cancel_waiting_disagg_requests([])
    _assert_released_once(executor, request)
    executor.disagg.poll_gen_transfers()
    executor._cancel_waiting_disagg_requests([])
    _assert_released_once(executor, request)
    transceiver.request_and_receive_async.assert_not_called()


def test_receive_progress_does_not_hide_notification_programming_error(
    executor: PyExecutor,
) -> None:
    """Unexpected errors escape the new polling guard without releasing ownership.

    Args:
        executor: Real receive session with a controlled notification boundary.
    """
    request = _request(1)
    session, aux = _published_receive(executor, request)
    session.cancel_local()
    session.notify_cancel = Mock(side_effect=TypeError("invalid notification"))
    transceiver = executor.kv_cache_transceiver

    with pytest.raises(TypeError, match="invalid notification"):
        transceiver._collect_done(transceiver._recv_sessions, transceiver._recv_reqs)

    assert transceiver.has_inflight_transfer(request)
    assert transceiver._recv_reqs == {1: request}
    aux.free_slot.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()


def test_peer_notification_error_does_not_delay_logical_cancellation(executor: PyExecutor) -> None:
    """Retry peer notification while the accepted cancellation retains its physical claims."""
    request = _request(1)
    executor.active_requests = [request]
    session, aux = _published_receive(executor, request)
    notify = session._receiver.send_cancel_to_senders
    notify.side_effect = RuntimeError("peer notification unavailable")
    _abort(executor, request)

    executor._cancel_waiting_disagg_requests([])
    executor._flush_pending_transfer_responses()

    assert request.is_finished_due_to_cancellation
    assert executor.active_requests == []
    assert executor.disagg._retiring_gen_receives == {1: request}
    (response,) = executor._response_channel.responses[1]
    assert response.error_msg is None
    notify.assert_called_once_with(1, {"ctx"})
    executor.resource_manager.free_resources.assert_not_called()
    aux.free_slot.assert_not_called()

    executor.disagg.poll_gen_transfers()
    assert len(executor._response_channel.responses[1]) == 1
    executor.resource_manager.free_resources.assert_not_called()
    notify.reset_mock()
    notify.side_effect = lambda _rid, endpoints: set(endpoints)
    executor.disagg.poll_gen_transfers()
    executor.disagg.poll_gen_transfers()
    assert [call.args for call in notify.call_args_list if call.args[1]] == [(1, {"ctx"})]
    assert len(executor._response_channel.responses[1]) == 1
    executor.resource_manager.free_resources.assert_not_called()
    session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    executor.disagg.poll_gen_transfers()
    _assert_released_once(executor, request)
    aux.free_slot.assert_called_once_with(0)


@pytest.mark.parametrize("context_first", [False, True], ids=["fp4-bridge", "bf16-context-first"])
@pytest.mark.parametrize("expires", [False, True], ids=["settles", "grace-expires"])
def test_cancelled_unpublished_receive_retains_local_fence_with_fixed_grace(
    executor: PyExecutor, expires: bool, context_first: bool
) -> None:
    """A logical cancel closes publication without freeing an unfinished local CUDA access.

    Args:
        executor: Real cancellation and retirement path with controlled resources.
        expires: Whether the admission fence remains unproven past its grace.
        context_first: Whether to exercise BF16 logical-result consensus.
    """
    request = _request(1)
    if context_first:
        _enable_context_first(executor, request)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    executor.active_requests = [request]
    clock = [0.0]
    containment = Mock()
    watchdog = RetirementWatchdog(containment, clock=lambda: clock[0])
    transceiver = executor.kv_cache_transceiver
    transceiver._transfer_worker._retirement_watchdog = watchdog
    transceiver.request_and_receive_async = Mock()
    ready = Mock(spec=["query"])
    ready.query.return_value = False
    executor.disagg._defer_unready_gen_receives = True
    executor.disagg._unpublished_gen_receives[1] = (request, ready)
    _abort(executor, request)

    executor._cancel_waiting_disagg_requests([])
    if context_first:
        executor.disagg.poll_gen_transfers()
        executor._cancel_waiting_disagg_requests([])
    executor._flush_pending_transfer_responses()

    assert request.is_finished_due_to_cancellation
    assert executor.active_requests == []
    assert executor.disagg._unpublished_gen_receives == {}
    assert executor.disagg._retiring_gen_receives == {1: request}
    assert transceiver.has_inflight_transfer(request)
    assert not transceiver.check_gen_transfer_complete()
    assert transceiver._recv_reqs == {1: request}
    (response,) = executor._response_channel.responses[1]
    assert response.error_msg is None
    executor.resource_manager.free_resources.assert_not_called()

    clock[0] = 0.75
    _abort(executor, request, repeats=2)
    executor._cancel_waiting_disagg_requests([])
    executor.disagg.poll_gen_transfers()
    assert len(watchdog._owners) == 1
    assert watchdog.fatal is None
    assert len(executor._response_channel.responses[1]) == 1
    transceiver.request_and_receive_async.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()

    clock[0] = 1.01 if expires else 0.9
    watchdog.progress()
    ready.query.return_value = True
    executor.disagg.poll_gen_transfers()
    if expires:
        assert watchdog.fatal is not None
        assert watchdog.fatal.started_at == 0.0
        assert watchdog.fatal.deadline == 1.0
        containment.assert_called_once_with(watchdog.fatal)
        assert transceiver._recv_reqs == {1: request}
        assert executor.disagg._retiring_gen_receives == {1: request}
        executor.resource_manager.free_resources.assert_not_called()
    else:
        containment.assert_not_called()
        assert transceiver._recv_reqs == {}
        assert transceiver.check_gen_transfer_complete()
        _assert_released_once(executor, request)
        watchdog.require_retired()
        watchdog.stop()
    transceiver.request_and_receive_async.assert_not_called()


@pytest.mark.parametrize("query_raises", [False, True], ids=["not-ready", "query-error"])
def test_cancelled_receive_shutdown_preserves_roots_until_fence_settles(
    executor: PyExecutor, query_raises: bool
) -> None:
    """Shutdown must not discard an unready or unobservable local CUDA claim.

    Args:
        executor: Real cancellation path with controlled teardown boundaries.
        query_raises: Whether CUDA completion cannot currently be queried.
    """
    request = _request(1)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    executor.active_requests = [request]
    transceiver = executor.kv_cache_transceiver
    worker = transceiver._transfer_worker
    worker.request_shutdown = Mock()
    worker.shutdown = Mock()
    unrelated_root = object()
    transceiver._send_reqs[2] = unrelated_root
    ready = Mock(spec=["query"])
    ready.query.return_value = False
    if query_raises:
        ready.query.side_effect = RuntimeError("CUDA completion query failed")
    executor.disagg._defer_unready_gen_receives = True
    executor.disagg._unpublished_gen_receives[1] = (request, ready)
    _abort(executor, request)

    executor._cancel_waiting_disagg_requests([])
    executor._flush_pending_transfer_responses()

    assert request.is_finished_due_to_cancellation
    (response,) = executor._response_channel.responses[1]
    assert response.error_msg is None
    expected = "CUDA completion query failed" if query_raises else "still owns destination memory"
    with pytest.raises(RuntimeError, match=expected):
        transceiver.shutdown()
    assert not getattr(transceiver, "_shutdown_complete", False)
    assert transceiver._recv_reqs == {1: request}
    assert transceiver._send_reqs == {2: unrelated_root}
    assert transceiver._cancelled_receive_fences[1].ready is ready
    assert executor.disagg._retiring_gen_receives == {1: request}
    executor.resource_manager.free_resources.assert_not_called()
    worker.shutdown.assert_not_called()

    ready.query.side_effect = None
    ready.query.return_value = True
    transceiver.shutdown()
    transceiver.shutdown()

    assert transceiver._shutdown_complete
    assert not transceiver._cancelled_receive_fences
    assert not transceiver._recv_reqs
    assert not transceiver._send_reqs
    worker.shutdown.assert_called_once_with()
    worker._retirement_watchdog.require_retired()
    worker._retirement_watchdog.stop()


@pytest.mark.parametrize("context_first", [False, True], ids=["fp4-bridge", "bf16-context-first"])
@pytest.mark.parametrize("shutdown", [False, True])
@pytest.mark.parametrize("published", [False, True], ids=["unpublished-fence", "published-session"])
def test_pending_receive_retirement_keeps_idle_fetch_and_shutdown_live(
    executor: PyExecutor,
    monkeypatch: pytest.MonkeyPatch,
    shutdown: bool,
    published: bool,
    context_first: bool,
) -> None:
    """The last logical result must not stop background receive retirement.

    Args:
        executor: Executor retaining a receive session or an unpublished CUDA fence.
        monkeypatch: Guards blocking queue reads and the CUDA control boundary.
        shutdown: Whether the executor has consumed its shutdown sentinel.
        published: Whether retirement owns a published session or a local admission fence.
        context_first: Whether to exercise BF16 logical-result consensus.
    """
    request = _request(1)
    if context_first:
        _enable_context_first(executor, request)
    executor.active_requests = [request]
    session = ready = None
    if published:
        session, _ = _published_receive(executor, request)
    else:
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        ready = Mock(spec=["query"])
        ready.query.return_value = False
        executor.disagg._defer_unready_gen_receives = True
        executor.disagg._unpublished_gen_receives[1] = (request, ready)
    _abort(executor, request)
    executor._cancel_waiting_disagg_requests([])
    if context_first:
        executor.disagg.poll_gen_transfers()
        executor._cancel_waiting_disagg_requests([])
    executor._flush_pending_transfer_responses()
    executor.is_shutdown = True
    assert not executor.should_stop_processing
    executor.control_requests = [SimpleNamespace(control_requires_drain=True)]
    synchronize = Mock(side_effect=AssertionError("control ran before receive retirement"))
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.synchronize", synchronize
    )
    executor._handle_control_request()
    synchronize.assert_not_called()
    executor.control_requests.clear()
    executor.is_shutdown = shutdown
    executor._disable_mpi = False
    executor.request_accumulated = []
    executor.hang_detector = MagicMock()
    executor.request_broadcaster = SimpleNamespace(broadcast=Mock(return_value=[]))
    executor.executor_request_queue.batch_wait_timeout_ms = 1000.0
    queue = executor.executor_request_queue.request_queue
    original_get = queue.get

    def nonblocking_get(block: bool = True, timeout: float | None = None) -> RequestQueueItem:
        assert not block, "batch accumulation blocked receive retirement"
        return original_get(block=block, timeout=timeout)

    monkeypatch.setattr(queue, "get", nonblocking_get)

    executor._fetch_and_enqueue_requests(executor.waiting_queue, 0)

    assert queue.empty()
    executor.request_broadcaster.broadcast.assert_called_once_with([], prefer_cpu=False)
    executor.disagg.poll_gen_transfers()
    executor.resource_manager.free_resources.assert_not_called()
    assert executor.disagg._retiring_gen_receives == {1: request}
    if published:
        session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
        session.process_aux_agent_result(0, AgentResult.SUCCESS)
    else:
        ready.query.return_value = True
    executor.disagg.poll_gen_transfers()
    executor.is_shutdown = True
    assert executor.should_stop_processing
    _assert_released_once(executor, request)


def test_shutdown_without_retirement_skips_batch_wait(
    executor: PyExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consumed shutdown sentinel cannot wait for another batch to arrive.

    Args:
        executor: Idle executor with its shutdown sentinel already consumed.
        monkeypatch: Guards blocking queue reads.
    """
    executor.is_shutdown = True
    executor.control_requests = []
    executor._disable_mpi = False
    executor.request_accumulated = []
    executor.hang_detector = MagicMock()
    executor.request_broadcaster = SimpleNamespace(broadcast=Mock(return_value=[]))
    executor.executor_request_queue.batch_wait_timeout_ms = 1000.0
    queue = executor.executor_request_queue.request_queue
    original_get = queue.get

    def nonblocking_get(block: bool = True, timeout: float | None = None) -> RequestQueueItem:
        """Reject blocking reads while preserving the real empty-queue behavior.

        Args:
            block: Whether the queue read may wait.
            timeout: Queue read timeout.

        Returns:
            An available queue item, if any.

        Raises:
            AssertionError: The executor attempts a blocking read.
            queue.Empty: No item is available.
        """
        assert not block, "shutdown waited for a batch that will never arrive"
        return original_get(block=block, timeout=timeout)

    monkeypatch.setattr(queue, "get", nonblocking_get)
    assert executor.should_stop_processing
    executor._fetch_and_enqueue_requests(executor.waiting_queue, 0)
    executor.request_broadcaster.broadcast.assert_called_once_with([], prefer_cpu=False)


def test_adp_receive_retirement_snapshot_keeps_empty_peer_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse the existing ADP vote so an empty rank cannot stop a peer's retirement.

    Args:
        monkeypatch: Guards CUDA control execution while retirement remains pending.
    """
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = [
        CoordinatorHarness(dist=group.rank(rank), enable_attention_dp=True) for rank in range(2)
    ]
    request = _request(1)
    request.finish_by_reason(FinishReason.CANCELLED)
    ranks[1].coordinator._retiring_gen_receives[1] = request

    assert group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced()) == [False, False]

    assert all(rank.coordinator.has_retiring_gen_receives for rank in ranks)
    assert [len(rank.dist.calls) for rank in ranks] == [1, 1]
    assert ranks[0].dist.calls[0] == ("tp_allgather", {"error_ids": [], "blocked_ids": []})
    assert ranks[1].dist.calls[0] == (
        "tp_allgather",
        {"error_ids": [], "blocked_ids": [], "retiring_gen_receives": True},
    )
    engines = []
    for rank in ranks:
        engine = object.__new__(PyExecutor)
        engine._disagg_coordinator = rank.coordinator
        engine.dist = rank.dist
        engine.enable_attention_dp = True
        engine.kv_connector_manager = None
        engine.active_requests = []
        engine.waiting_queue = FCFSWaitingQueue()
        engine.is_shutdown = True
        engine.control_requests = [SimpleNamespace(control_requires_drain=True)]
        engines.append(engine)
    synchronize = Mock(side_effect=AssertionError("control crossed pending retirement"))
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.synchronize", synchronize
    )
    assert [engine.should_stop_processing for engine in engines] == [False, False]
    group.run(lambda rank: engines[rank]._handle_control_request())
    synchronize.assert_not_called()
    assert [rank.dist.calls[-1] for rank in ranks] == [("tp_allgather", 0)] * 2

    ranks[1].coordinator._retiring_gen_receives.clear()
    group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())
    assert not any(rank.coordinator.has_retiring_gen_receives for rank in ranks)
    assert [engine.should_stop_processing for engine in engines] == [True, True]
    assert [len(rank.dist.calls) for rank in ranks] == [3, 3]


class _ReachedScheduler(Exception):
    """Stop the real preparation path before model/capacity scheduling."""


@pytest.mark.parametrize("context, bridge_enabled", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("model_active", [False, True], ids=["waiting", "model-active"])
def test_preparation_applies_worker_cancel_before_publication_or_scheduling(
    executor: PyExecutor, context: bool, bridge_enabled: bool, model_active: bool
) -> None:
    """The real preparation path applies abort before admitting more work."""
    cancelled, healthy = _request(1, context=context), _request(2)
    if context:
        cancelled.state = LlmRequestState.CONTEXT_INIT
    executor.active_requests = [cancelled, healthy]
    _abort(executor, cancelled)
    transceiver = executor.kv_cache_transceiver
    transceiver._fp4_mla_bridge_enabled = bridge_enabled
    transceiver.prepare_context_requests = Mock()
    transceiver.request_and_receive_async = Mock()
    ready = Mock()
    ready.query.return_value = True
    executor.disagg._defer_unready_gen_receives = True
    if model_active:
        cancelled.state = (
            LlmRequestState.CONTEXT_INIT if context else LlmRequestState.GENERATION_IN_PROGRESS
        )
    elif not context:
        cancelled.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        executor.disagg._unpublished_gen_receives[1] = (cancelled, ready)
    executor._release_unused_connector_reservations = Mock()
    executor._poll_encoder_steps = Mock()
    executor._fetch_and_activate_new_requests = Mock(
        return_value=[] if model_active else [cancelled, healthy]
    )
    executor._handle_control_request = Mock()
    executor._activate_completed_remote_tails = Mock()
    executor._init_iter_stats_if_sampled = Mock()
    executor._pad_attention_dp_dummy_request = Mock()
    executor._prefetch_for_context_requests = Mock()
    executor.drafter = None
    executor._schedule = Mock(side_effect=_ReachedScheduler)

    with pytest.raises(_ReachedScheduler):
        executor._prepare_and_schedule_batch()

    transceiver.prepare_context_requests.assert_called_once_with([])
    transceiver.request_and_receive_async.assert_not_called()
    if context or model_active:
        ready.query.assert_not_called()
    else:
        ready.query.assert_called()
    assert executor.active_requests == [healthy]
    assert not healthy.is_finished
    _assert_released_once(executor, cancelled)


@pytest.mark.parametrize(
    ("multimodal_params", "mm_item_scheduling", "encoder_decoder"),
    [
        (object(), False, False),
        (None, True, True),
    ],
)
def test_legacy_tp_cancel_reaches_multimodal_model_generation_work(
    executor: PyExecutor,
    multimodal_params: object | None,
    mm_item_scheduling: bool,
    encoder_decoder: bool,
) -> None:
    """Consensus cancellation reaches multimodal generation work between batches.

    The scheduler-exclusion gate used to drop any request with multimodal
    parameters or encoder-decoder models outright. With the replicated-GEN
    consensus active, such a request is admitted to model-work cancellation, the
    previous batch protects it for one iteration, and the settled vote retires
    it together with exactly one resource release per rank.
    """
    group = FakeDistGroup(world_size=2, tp_size=2)
    engines = []
    requests = []
    for rank in range(2):
        engine = object.__new__(PyExecutor)
        engine.__dict__.update(executor.__dict__)
        engine.dist = group.rank(rank)
        engine.is_encoder_decoder = encoder_decoder
        engine._mm_encoder_item_scheduling_enabled = mm_item_scheduling
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver.__dict__.update(executor.kv_cache_transceiver.__dict__)
        transceiver._fp4_mla_bridge_enabled = False
        transceiver._dist = engine.dist
        transceiver._recv_reqs = {}
        transceiver._recv_sessions = {}
        transceiver._send_reqs = {}
        transceiver._send_sessions = {}
        transceiver._gen_need_sync = False
        transceiver._gen_allgather = Mock(side_effect=AssertionError("unexpected collective"))
        engine.kv_cache_transceiver = transceiver
        engine._disagg_coordinator = DisaggTransferCoordinator(
            transceiver=transceiver,
            transfer_manager=engine.async_transfer_manager,
            kv_cache_manager=None,
            dist=engine.dist,
            effects=PyExecutorEffects(engine),
            registry=PyExecutorRequestRegistry(engine),
            enable_attention_dp=False,
        )
        request = _request(1)
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        request.py_multimodal_params = multimodal_params
        engine.active_requests = [request]
        engine.canceled_req_ids = [1]
        engine.waiting_queue = FCFSWaitingQueue()
        engine.previous_batch = None
        engine._pending_transfer_responses = []
        engine._pending_response_terminations = []
        engine.disagg.request_cancellation = Mock(return_value=True)
        engines.append(engine)
        requests.append(request)

    # Pre-fix, encoder-decoder models and multimodal parameters excluded the
    # request from model-work cancellation entirely. The consensus branch now
    # admits it on every rank.
    for engine in engines:
        assert engine._cancelled_disagg_model_request_ids() == {1}

    # The previous model batch still owns the request: the vote must hold it.
    for rank, engine in enumerate(engines):
        scheduled = ScheduledRequests()
        scheduled.generation_requests = [requests[rank]]
        engine.previous_batch = SimpleNamespace(scheduled_requests=scheduled)
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    assert all(not request.is_finished for request in requests)
    assert all(engine.canceled_req_ids == [1] for engine in engines)

    for engine in engines:
        engine.previous_batch = None
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    assert all(request.is_finished_due_to_cancellation for request in requests)
    assert all(engine.canceled_req_ids == [] for engine in engines)
    assert all(engine.active_requests == [] for engine in engines)


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize(
    ("state", "encoder_decoder"),
    [
        (LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS, False),
        (LlmRequestState.GENERATION_IN_PROGRESS, False),
        (LlmRequestState.GENERATION_IN_PROGRESS, True),
    ],
)
def test_legacy_tp_cancel_waits_for_peer_retirement_and_retries_while_idle(
    executor: PyExecutor,
    tp_size: int,
    state: LlmRequestState,
    encoder_decoder: bool,
) -> None:
    """One drained receive cannot remove a replicated request ahead of its peers."""
    group = FakeDistGroup(world_size=tp_size, tp_size=tp_size)
    ready = [True] * tp_size
    ready[-1] = False
    engines = []
    requests = []
    for rank in range(tp_size):
        engine = object.__new__(PyExecutor)
        engine.__dict__.update(executor.__dict__)
        engine.dist = group.rank(rank)
        request = _request(1)
        request.state = state
        engine.is_encoder_decoder = encoder_decoder
        engine.active_requests = [request]
        engine.canceled_req_ids = [1]
        engine.waiting_queue = FCFSWaitingQueue()
        engine._pending_transfer_responses = []
        engine._pending_response_terminations = []
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver.__dict__.update(executor.kv_cache_transceiver.__dict__)
        transceiver._fp4_mla_bridge_enabled = False
        transceiver._dist = engine.dist
        engine.kv_cache_transceiver = transceiver
        engine._disagg_coordinator = DisaggTransferCoordinator(
            transceiver=transceiver,
            transfer_manager=engine.async_transfer_manager,
            kv_cache_manager=None,
            dist=engine.dist,
            effects=PyExecutorEffects(engine),
            registry=PyExecutorRequestRegistry(engine),
            enable_attention_dp=False,
        )
        engine.disagg.request_cancellation = Mock(side_effect=lambda req, r=rank: ready[r])
        engine._try_cancel_request = Mock(side_effect=lambda req, r=rank: ready[r])
        engines.append(engine)
        requests.append(request)

    # A receive still referenced by a prior model batch cannot be closed by
    # the loop-head retry, even when local transport retirement reports ready.
    for rank, engine in enumerate(engines):
        scheduled = ScheduledRequests()
        scheduled.generation_requests = [requests[rank]]
        engine.previous_batch = SimpleNamespace(scheduled_requests=scheduled)
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    assert all(not request.is_finished for request in requests)
    for engine in engines:
        engine.disagg.request_cancellation.assert_not_called()
        engine.previous_batch = None

    group.run(lambda rank: engines[rank]._handle_canceled_requests())

    assert all(not request.is_finished for request in requests)
    assert all(engine.canceled_req_ids == [1] for engine in engines)
    assert all(engine.active_requests == [requests[rank]] for rank, engine in enumerate(engines))

    # No model batch follows. The loop-head path must keep retrying without
    # releasing the rank whose physical receive already drained.
    remaining = group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    assert remaining == [[] for _ in engines]
    assert all(not request.is_finished for request in requests)
    assert all(engine.canceled_req_ids == [1] for engine in engines)
    assert all(engine._pending_response_terminations == [] for engine in engines)

    ready[-1] = True
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))

    assert all(request.is_finished_due_to_cancellation for request in requests)
    assert all(engine.active_requests == [] for engine in engines)
    assert all(engine.canceled_req_ids == [] for engine in engines)
    assert all(
        engine._pending_response_terminations == [requests[rank]]
        for rank, engine in enumerate(engines)
    )
    assert all(
        [name for name, _ in engine.dist.calls] == ["tp_allgather"] * 3 for engine in engines
    )


@pytest.mark.parametrize("peer_owns_receive", [False, True])
@pytest.mark.parametrize("cancel_rank", [0, 1])
def test_legacy_tp_cancel_with_request_missing_on_peer(
    executor: PyExecutor, peer_owns_receive: bool, cancel_rank: int
) -> None:
    """A peer with no active copy must still participate in cancellation."""
    group = FakeDistGroup(world_size=2, tp_size=2, timeout_s=1)
    engines = []
    request = _request(1)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    for rank in range(2):
        engine = object.__new__(PyExecutor)
        engine.__dict__.update(executor.__dict__)
        engine.dist = group.rank(rank)
        engine.active_requests = [request] if rank == 0 else []
        engine.canceled_req_ids = [1] if rank == cancel_rank else []
        engine.waiting_queue = FCFSWaitingQueue()
        engine._pending_transfer_responses = []
        engine._pending_response_terminations = []
        engine.previous_batch = None
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver.__dict__.update(executor.kv_cache_transceiver.__dict__)
        transceiver._fp4_mla_bridge_enabled = False
        transceiver._dist = engine.dist
        if rank == 1 and peer_owns_receive:
            transceiver._recv_reqs = {1: _request(1)}
        engine.kv_cache_transceiver = transceiver
        engine._disagg_coordinator = DisaggTransferCoordinator(
            transceiver=transceiver,
            transfer_manager=engine.async_transfer_manager,
            kv_cache_manager=None,
            dist=engine.dist,
            effects=PyExecutorEffects(engine),
            registry=PyExecutorRequestRegistry(engine),
            enable_attention_dp=False,
        )
        engine.disagg.request_cancellation = Mock(return_value=True)
        engine._try_cancel_request = Mock(return_value=True)
        engines.append(engine)

    group.run(lambda rank: engines[rank]._handle_canceled_requests())
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    if peer_owns_receive:
        assert not request.is_finished
        engines[1].kv_cache_transceiver._recv_reqs.clear()
    if not request.is_finished:
        group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    assert request.is_finished_due_to_cancellation
    assert all(not engine.canceled_req_ids for engine in engines)


def test_legacy_tp_cancel_propagates_to_peer_with_active_request(executor: PyExecutor) -> None:
    """A peer missing the cancel ID must drain its own transfer before retirement."""
    group = FakeDistGroup(world_size=2, tp_size=2, timeout_s=1)
    engines = []
    requests = []
    for rank in range(2):
        engine = object.__new__(PyExecutor)
        engine.__dict__.update(executor.__dict__)
        engine.dist = group.rank(rank)
        request = _request(1)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        engine.active_requests = [request]
        engine.canceled_req_ids = [1] if rank == 0 else []
        engine.waiting_queue = FCFSWaitingQueue()
        engine._pending_transfer_responses = []
        engine._pending_response_terminations = []
        engine.previous_batch = None
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver.__dict__.update(executor.kv_cache_transceiver.__dict__)
        transceiver._fp4_mla_bridge_enabled = False
        transceiver._dist = engine.dist
        engine.kv_cache_transceiver = transceiver
        engine._disagg_coordinator = DisaggTransferCoordinator(
            transceiver=transceiver,
            transfer_manager=engine.async_transfer_manager,
            kv_cache_manager=None,
            dist=engine.dist,
            effects=PyExecutorEffects(engine),
            registry=PyExecutorRequestRegistry(engine),
            enable_attention_dp=False,
        )
        engine.disagg.request_cancellation = Mock(return_value=True)
        engines.append(engine)
        requests.append(request)

    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    assert all(not request.is_finished for request in requests)
    assert all(engine.canceled_req_ids == [1] for engine in engines)
    engines[1].disagg.request_cancellation.assert_not_called()
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    assert all(request.is_finished_due_to_cancellation for request in requests)
    assert all(not engine.canceled_req_ids for engine in engines)


@pytest.mark.parametrize("missing_replica", [None, 0, 3])
def test_legacy_tp_cp_cancel_vote_covers_cp_ranks_with_missing_replica(
    executor: PyExecutor, missing_replica: int | None
) -> None:
    """The cancellation vote rides the TP+CP collective with asymmetric readiness.

    With tp_size=2 and cp_size=2 the vote must gather over ``tp_cp_allgather``,
    not the TP-only collective. A cancellation originating on one rank propagates its
    ID to peers (including one with no active replica), retirement is held while a
    peer's receive is not drained, and once every rank settles the batch retires
    together with the ID cleared everywhere.
    """
    ranks = 4
    group = FakeDistGroup(world_size=ranks, tp_size=2, cp_size=2)
    engines = []
    requests = []
    ready = [True] * ranks
    ready[1] = False  # peer whose receive is not drained
    for rank in range(ranks):
        engine = object.__new__(PyExecutor)
        engine.__dict__.update(executor.__dict__)
        engine.dist = group.rank(rank)
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver.__dict__.update(executor.kv_cache_transceiver.__dict__)
        transceiver._fp4_mla_bridge_enabled = False
        transceiver._dist = engine.dist
        transceiver._recv_reqs = {}
        transceiver._recv_sessions = {}
        transceiver._send_reqs = {}
        transceiver._send_sessions = {}
        transceiver._gen_need_sync = False
        transceiver._gen_allgather = Mock(side_effect=AssertionError("unexpected collective"))
        engine.kv_cache_transceiver = transceiver
        engine._disagg_coordinator = DisaggTransferCoordinator(
            transceiver=transceiver,
            transfer_manager=engine.async_transfer_manager,
            kv_cache_manager=None,
            dist=engine.dist,
            effects=PyExecutorEffects(engine),
            registry=PyExecutorRequestRegistry(engine),
            enable_attention_dp=False,
        )
        request = _request(1)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        if rank == missing_replica:
            engine.active_requests = []
        else:
            engine.active_requests = [request]
        # The cancel ID starts only on rank 0; propagation must fan it out.
        engine.canceled_req_ids = [1] if rank == 0 else []
        engine.waiting_queue = FCFSWaitingQueue()
        engine.previous_batch = None
        engine._pending_transfer_responses = []
        engine._pending_response_terminations = []
        engine.disagg.request_cancellation = Mock(side_effect=lambda req, r=rank: ready[r])
        engines.append(engine)
        requests.append(request)

    # Vote one: the not-ready rank holds the batch everywhere and the cancel ID
    # propagates to every rank, including the one with no active replica.
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    assert all(engine.canceled_req_ids == [1] for engine in engines)
    assert all(
        not request.is_finished for rank, request in enumerate(requests) if rank != missing_replica
    )
    for engine in engines:
        assert "tp_cp_allgather" in [name for name, _ in engine.dist.calls]

    # Vote two, still not ready: the propagated owner attempts cancellation but stays
    # unretired, so retirement cannot slip past an unready CP owner.
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    if missing_replica != 1:
        assert engines[1].disagg.request_cancellation.call_count >= 1
    assert all(
        not request.is_finished for rank, request in enumerate(requests) if rank != missing_replica
    )
    assert all(engine.canceled_req_ids == [1] for engine in engines)

    # Every rank's drain settles: the batch retires together and the ID clears.
    ready[1] = True
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    for rank, (engine, request) in enumerate(zip(engines, requests)):
        if rank != missing_replica:
            assert request.is_finished_due_to_cancellation
        assert engine.canceled_req_ids == []
    for engine in engines:
        assert "tp_cp_allgather" in [name for name, _ in engine.dist.calls]


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("context_first_runtime", [False, True])
def test_legacy_tp_cancel_with_capacity_connector_completes_consensus(
    executor: PyExecutor, tp_size: int, context_first_runtime: bool
) -> None:
    """A capacity-only connector must not drop a consensus cancellation.

    A generation server can attach a capacity-only KV connector (external storage
    that moves no KV, e.g. for shared capacity) alongside the legacy
    asynchronous transceiver that requires cross-rank retirement consensus.
    Such a connector only gates cancellation through its per-request release
    hooks; the loop-head vote must still run on every rank, a not-ready
    rank must hold its peers' retirement, and settled drains must retire the
    batch together.
    """
    group = FakeDistGroup(world_size=tp_size, tp_size=tp_size)
    engines = []
    requests = []
    ready = [True] * tp_size
    ready[-1] = False
    for rank in range(tp_size):
        engine = object.__new__(PyExecutor)
        engine.__dict__.update(executor.__dict__)
        engine.dist = group.rank(rank)
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver.__dict__.update(executor.kv_cache_transceiver.__dict__)
        transceiver._fp4_mla_bridge_enabled = False
        transceiver._context_first_ownership = context_first_runtime
        transceiver._dist = engine.dist
        transceiver._recv_reqs = {}
        transceiver._recv_sessions = {}
        transceiver._send_reqs = {}
        transceiver._send_sessions = {}
        transceiver._gen_need_sync = False
        transceiver._gen_allgather = Mock(side_effect=AssertionError("unexpected collective"))
        engine.kv_cache_transceiver = transceiver
        engine._disagg_coordinator = DisaggTransferCoordinator(
            transceiver=transceiver,
            transfer_manager=engine.async_transfer_manager,
            kv_cache_manager=None,
            dist=engine.dist,
            effects=PyExecutorEffects(engine),
            registry=PyExecutorRequestRegistry(engine),
            enable_attention_dp=False,
        )
        engine.kv_cache_manager = Mock(spec=["is_preemption_pending"])
        engine.kv_cache_manager.is_preemption_pending.return_value = False
        engine.kv_connector_manager = Mock(
            spec=[
                "capacity_only",
                "release_unstarted_prefix_loads",
                "defer_load_termination",
                "has_pending_load",
            ]
        )
        engine.kv_connector_manager.capacity_only = True
        engine.kv_connector_manager.defer_load_termination.return_value = False
        request = _request(1)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        engine.active_requests = [request]
        engine.canceled_req_ids = [1]
        engine.waiting_queue = FCFSWaitingQueue()
        engine.previous_batch = None
        engine._pending_transfer_responses = []
        engine._pending_response_terminations = []
        engine.disagg.request_cancellation = Mock(side_effect=lambda req, r=rank: ready[r])
        engines.append(engine)
        requests.append(request)

    # The vote runs with the connector attached. The not-ready rank's physical
    # drain holds every rank's cancellation, so no request retires early.
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    assert all(not request.is_finished for request in requests)
    assert all(engine.canceled_req_ids == [1] for engine in engines)
    assert all(engine.active_requests == [requests[rank]] for rank, engine in enumerate(engines))
    for engine in engines:
        engine.kv_connector_manager.release_unstarted_prefix_loads.assert_called_once()
        # The gate consults physical retirement through the connector-aware
        # _try_cancel_request tail, then the logical-cancel fallback consults it
        # again; both polls run before the vote.
        engine.disagg.request_cancellation.assert_called()

    # Every drain settles. The batch retires together and each cancel ID leaves
    # canceled_req_ids on every rank.
    ready[-1] = True
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    assert all(request.is_finished_due_to_cancellation for request in requests)
    assert all(engine.active_requests == [] for engine in engines)
    assert all(engine.canceled_req_ids == [] for engine in engines)


@pytest.mark.parametrize("overlap_disabled", [False, True])
def test_pending_cancel_idle_hold_needs_the_consensus_vote(
    executor: PyExecutor, monkeypatch: pytest.MonkeyPatch, overlap_disabled: bool
) -> None:
    """A pending cancel ID holds idle only when the loop-head vote can clear it.

    With the consensus disabled (e.g. TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1),
    no loop-head path clears a late cancel ID without a batch, so a non-idle
    hold would busy-poll the queue until the next request arrives; the fetch
    keeps its normal idle behavior instead. With the consensus enabled, the
    pending ID keeps the rank awake for the loop-head vote.
    """
    if overlap_disabled:
        monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
    else:
        monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)

    executor.canceled_req_ids = [9]
    executor.control_requests = []
    executor.enable_iter_perf_stats = False
    executor._disable_mpi = False
    executor.request_accumulated = []
    executor.hang_detector = MagicMock()
    broadcast = Mock(return_value=[])
    executor.request_broadcaster = SimpleNamespace(broadcast=broadcast)

    def fetch_probe(timeout, *, wait_for_batch=True):
        # A busy-poll regression shows up as an endless timeout=0 wait; the idle
        # path parks with no timeout at all. Record the first queue read the
        # fetch makes instead of parking this test on the real wait.
        if fetch_probe.last_timeout == "unset":
            fetch_probe.last_timeout = timeout
        return []

    fetch_probe.last_timeout = "unset"
    monkeypatch.setattr(executor.executor_request_queue, "get_from_request_queue", fetch_probe)
    executor.disagg.request_cancellation = Mock(return_value=True)

    executor._fetch_and_enqueue_requests(executor.waiting_queue, 0)

    consensus = (
        executor.kv_cache_transceiver.consumes_transfer_buffer is False
        and executor.disagg.requires_gen_cancellation_consensus
    )
    if not consensus:
        # The vote cannot clear this ID without a batch; the fetch stays idle so
        # the worker parks rather than burning a zero-timeout poll loop.
        broadcast.assert_called_once_with([], prefer_cpu=True)
        assert fetch_probe.last_timeout is None
    else:
        broadcast.assert_called_once_with([], prefer_cpu=False)
        assert fetch_probe.last_timeout == timedelta(0)


@pytest.mark.parametrize("timeout_entry", ["idle_progress", "response"])
def test_pending_rank_info_timeout_precedes_later_user_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    timeout_entry: str,
) -> None:
    """An executor timeout wins before native publication or a later cancel vote."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    engines = [executor.__wrapped__(monkeypatch) for _ in range(2)]
    requests = [_request(1), _request(1)]
    now = [0.0]
    monkeypatch.setattr(coordinator_mod.time, "monotonic", lambda: now[0])
    for rank, (engine, request) in enumerate(zip(engines, requests)):
        _enable_context_first(engine, request)
        dist = group.rank(rank)
        dist.mapping = Mapping(world_size=2, rank=rank, tp_size=2)
        engine.dist = dist
        engine.disagg._dist = dist
        engine._fatal_error = None
        engine.perf_manager = Mock()
        engine.active_requests = [request]
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        transceiver = engine.kv_cache_transceiver
        transceiver._dist = dist
        transceiver._mapping = dist.mapping
        transceiver._gen_need_sync = True
        transceiver._gen_allgather = dist.tp_allgather
        transceiver._init_gen_idle_poll(True)
        transceiver._pending_rank_info_admissions = {1: (request, None)}
        transceiver._start_async_receive = Mock()
        engine.disagg._stamp_gen_transfer_start([request])

    # GEN's stamped transfer budget is 1s; CTX's separate peer-wait multiplier
    # is not involved in a destination already admitted to RankInfo progress.
    now[0] = 2.0
    for engine, request in zip(engines, requests):
        if timeout_entry == "idle_progress":
            engine.disagg._retire_legacy_timed_out_gen_transfers()
        else:
            engine.disagg.check_transfer_timeouts()
            engine._handle_responses()
        assert request.py_kv_transfer_timed_out
        assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        assert not engine.kv_cache_transceiver._pending_rank_info_admissions
        _abort(engine, request)
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    for engine, request in zip(engines, requests):
        assert not request.is_finished
        engine.resource_manager.free_resources.assert_not_called()

    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    for engine, request in zip(engines, requests):
        assert request.py_kv_receive_cancel_outcome is ReceiveCancellation.FAILED
        assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
        assert not request.is_finished_due_to_cancellation
        engine.kv_cache_transceiver._start_async_receive.assert_not_called()
    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    for engine, request in zip(engines, requests):
        engine._flush_pending_transfer_responses()
        engine.resource_manager.free_resources.assert_called_once_with(request)
    (response,) = engines[0].await_responses(timeout=timedelta(0))
    assert response.error_msg is not None


@pytest.mark.parametrize("timeout", [False, True])
def test_direct_physical_cancel_preserves_existing_request_failure(
    executor: PyExecutor, timeout: bool
) -> None:
    """The physical API passes earlier state or timeout failure into logical agreement."""
    request = _request(1)
    _enable_context_first(executor, request)
    request.state = (
        LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        if timeout
        else LlmRequestState.DISAGG_TRANS_ERROR
    )
    request.py_kv_transfer_timed_out = timeout
    initial_state = request.state
    transceiver = executor.kv_cache_transceiver
    assert not transceiver.cancel_request(request)
    transceiver.check_gen_transfer_status(0)
    assert request.py_kv_receive_cancel_outcome is ReceiveCancellation.FAILED
    assert request.state == initial_state
    assert transceiver.cancel_request(request)
    assert not request.is_finished_due_to_cancellation


@pytest.mark.parametrize("commit_first", [False, True], ids=["pending", "committed"])
def test_context_first_repeat_cancel_preserves_the_outcome_boundary(
    executor: PyExecutor, commit_first: bool
) -> None:
    """An executor failure joins a pending vote, but cannot rewrite a committed cancel."""
    request = _request(1)
    _enable_context_first(executor, request)
    session, aux = _published_receive(executor, request)
    transceiver = executor.kv_cache_transceiver
    initial_state = request.state
    assert executor.disagg.cancel_gen_logically(request) is ReceiveCancellation.PENDING
    if commit_first:
        transceiver.check_gen_transfer_status(0)
        assert transceiver._cancelled_gen_receives[1].outcome is ReceiveCancellation.CANCELLED

    # The executor can report an already-held failure only on a repeat call.
    result = executor.disagg.cancel_gen_logically(request, prior_failure=True)
    assert result is (
        ReceiveCancellation.CANCELLED if commit_first else ReceiveCancellation.PENDING
    )
    transceiver.check_gen_transfer_status(0)
    expected = ReceiveCancellation.CANCELLED if commit_first else ReceiveCancellation.FAILED
    assert transceiver._cancelled_gen_receives[1].outcome is expected
    assert request.state == initial_state
    assert request.py_kv_receive_cancel_outcome is None
    assert not transceiver.cancel_request(request)
    aux.free_slot.assert_not_called()

    session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    transceiver.check_gen_transfer_status(0)
    assert request.py_kv_receive_cancel_outcome is expected
    assert request.state == initial_state
    assert transceiver.cancel_request(request)
    aux.free_slot.assert_called_once_with(0)
    assert executor.disagg.cancel_gen_logically(request, prior_failure=True) is expected
    executor.resource_manager.free_resources.assert_not_called()


@pytest.mark.parametrize("mixed_publication", [False, True], ids=["published", "mixed-publication"])
@pytest.mark.parametrize(
    "prior_failure",
    [None, "backend", "peer"],
    ids=["cancel-first", "peer-failed-first", "peer-cancel-first"],
)
def test_context_first_cancel_agrees_outcome_before_group_retirement(
    monkeypatch: pytest.MonkeyPatch, mixed_publication: bool, prior_failure: str | None
) -> None:
    """Agree TP cancellation without releasing a locally safe peer's destination.

    Args:
        monkeypatch: Isolate the controlled native-runtime test boundaries.
        mixed_publication: Leave one destination behind its local CUDA fence.
        prior_failure: Record an earlier backend error or peer cancellation on one rank.
    """
    group = FakeDistGroup(world_size=2, tp_size=2)
    engines = [executor.__wrapped__(monkeypatch) for _ in range(2)]
    requests = [_request(1), _request(1)]
    resources = []
    for rank, (engine, request) in enumerate(zip(engines, requests)):
        _enable_context_first(engine, request)
        dist = group.rank(rank)
        dist.mapping = Mapping(world_size=2, rank=rank, tp_size=2)
        engine.dist = dist
        engine.disagg._dist = dist
        engine._fatal_error = None
        transceiver = engine.kv_cache_transceiver
        transceiver._dist = dist
        transceiver._mapping = dist.mapping
        transceiver._gen_need_sync = True
        transceiver._gen_allgather = dist.tp_allgather
        transceiver._init_gen_idle_poll(True)
        healthy = _request(2)
        engine.active_requests = [request, healthy]
        if mixed_publication and rank == 0:
            ready = Mock(spec=["query"])
            ready.query.return_value = False
            engine.disagg._defer_unready_gen_receives = True
            engine.disagg._unpublished_gen_receives[1] = (request, ready)
            resources.append((None, ready))
        else:
            session, aux = _published_receive(engine, request)
            resources.append((session, aux))
            if prior_failure == "peer" and rank == 1:
                session.cancel_local(by_peer=True)
                assert session.cancelled_by_peer
            elif prior_failure == "backend" and rank == 1:
                original_cancel = session.cancel_local

                def fail_before_cancel(
                    session: RxSession = session,
                    cancel: Callable[[], bool] = original_cancel,
                ) -> bool:
                    # The report wins after the transceiver's first status
                    # snapshot, but before native cancellation commits.
                    session.process_kv_agent_result(
                        0, 0, True, AgentResult.IN_DOUBT, _KV_RESULT_REPORT_TOKEN
                    )
                    return cancel()

                session.cancel_local = fail_before_cancel
        _abort(engine, request, repeats=2)
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    for engine, request in zip(engines, requests):
        assert not request.is_finished
        assert not engine.disagg._unpublished_gen_receives
        engine.kv_cache_transceiver.request_and_receive_async(request)
    before_progress = [len(group.rank(rank).calls) for rank in range(2)]
    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    assert [len(group.rank(rank).calls) - before_progress[rank] for rank in range(2)] == [2, 2]
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    for engine, request in zip(engines, requests):
        record = engine.kv_cache_transceiver._cancelled_gen_receives[1]
        assert record.outcome is (
            ReceiveCancellation.FAILED if prior_failure else ReceiveCancellation.CANCELLED
        )
        engine._flush_pending_transfer_responses()
        assert request.is_finished_due_to_cancellation is (prior_failure is None)
        assert request.state == (
            LlmRequestState.DISAGG_TRANS_ERROR
            if prior_failure
            else LlmRequestState.GENERATION_COMPLETE
        )
        engine.resource_manager.free_resources.assert_not_called()
        assert not engine.active_requests[-1].is_finished
        assert engine.active_requests[-1].py_request_id == 2

    def settle(rank: int) -> None:
        session, resource = resources[rank]
        if session is None:
            resource.query.return_value = True
        else:
            session.process_kv_agent_result(
                0,
                0,
                True,
                (
                    AgentResult.FAILED_QUIESCED
                    if rank == 1 and prior_failure == "backend"
                    else AgentResult.FAILED
                    if rank == 1
                    else AgentResult.SUCCESS
                ),
                _KV_RESULT_REPORT_TOKEN,
            )

    settle(0)
    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    for engine in engines:
        assert engine.kv_cache_transceiver._recv_reqs[1].py_request_id == 1
        engine.resource_manager.free_resources.assert_not_called()
    if resources[0][0] is not None:
        resources[0][1].free_slot.assert_not_called()

    settle(1)
    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    group.run(lambda rank: engines[rank]._cancel_waiting_disagg_requests([]))
    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    for engine, request in zip(engines, requests):
        engine._flush_pending_transfer_responses()
        assert request.py_kv_receive_cancel_outcome is (
            ReceiveCancellation.FAILED if prior_failure else ReceiveCancellation.CANCELLED
        )
        assert engine.kv_cache_transceiver._recv_reqs == {}
        assert engine.kv_cache_transceiver._recv_sessions == {}
        assert engine.kv_cache_transceiver._cancelled_gen_receives == {}
        engine.resource_manager.free_resources.assert_called_once_with(request)
        assert not engine.active_requests[0].is_finished
        assert engine.active_requests[0].py_request_id == 2
        for _ in range(2):
            assert engine.disagg.request_cancellation(request)
        assert not engine.kv_cache_transceiver._cancelled_gen_receives
        engine.kv_cache_transceiver.request_and_receive_async(request)
    (response,) = engines[0].await_responses(timeout=timedelta(0))
    assert (response.error_msg is not None) is (prior_failure is not None)
    calls = [len(group.rank(rank).calls) for rank in range(2)]
    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    assert [len(group.rank(rank).calls) for rank in range(2)] == calls


@pytest.mark.parametrize(
    "enabled,context,style,request_id,expected",
    [
        (True, False, DisaggScheduleStyle.CONTEXT_FIRST, 1, True),
        (False, False, DisaggScheduleStyle.CONTEXT_FIRST, 1, False),
        (True, True, DisaggScheduleStyle.CONTEXT_FIRST, 1, False),
        (True, False, DisaggScheduleStyle.GENERATION_FIRST, 1, False),
        (True, False, DisaggScheduleStyle.CONTEXT_FIRST, None, False),
        (True, False, DisaggScheduleStyle.CONTEXT_FIRST, -1, False),
    ],
)
def test_context_first_logical_gen_cancel_keeps_the_existing_profile_gate(
    executor: PyExecutor,
    enabled: bool,
    context: bool,
    style: DisaggScheduleStyle,
    request_id: int | None,
    expected: bool,
) -> None:
    """Only an enabled context-first GEN with an immutable identity uses the new vote."""
    request = _request(1, context=context)
    _enable_context_first(executor, request)
    request.py_disaggregated_params = DisaggregatedParams(
        disagg_request_id=request_id, schedule_style=style
    )
    transceiver = executor.kv_cache_transceiver
    transceiver._context_first_ownership = enabled
    assert transceiver.supports_group_receive_cancellation() is enabled
    assert transceiver._supports_group_receive_cancellation(request) is expected


def test_context_first_pending_vote_cannot_reach_next_scheduler_batch(
    executor: PyExecutor,
) -> None:
    """The real preparation path commits a pre-publication cancel before scheduling."""
    cancelled, healthy = _request(1), _request(2)
    _enable_context_first(executor, cancelled)
    executor.active_requests = [cancelled, healthy]
    _abort(executor, cancelled)
    transceiver = executor.kv_cache_transceiver
    transceiver.prepare_context_requests = Mock()
    transceiver.request_and_receive_async = Mock()
    executor._release_unused_connector_reservations = Mock()
    executor._poll_encoder_steps = Mock()
    executor._fetch_and_activate_new_requests = Mock(return_value=[cancelled, healthy])
    executor._handle_control_request = Mock()
    executor._activate_completed_remote_tails = Mock()
    executor._init_iter_stats_if_sampled = Mock()
    executor._pad_attention_dp_dummy_request = Mock()
    executor._prefetch_for_context_requests = Mock()
    executor.drafter = None
    executor._schedule = Mock(side_effect=_ReachedScheduler)

    with pytest.raises(_ReachedScheduler):
        executor._prepare_and_schedule_batch()

    assert executor.active_requests == [healthy]
    assert not healthy.is_finished
    assert cancelled.py_kv_receive_cancel_outcome is ReceiveCancellation.CANCELLED
    transceiver.request_and_receive_async.assert_not_called()
    _assert_released_once(executor, cancelled)


def test_context_first_post_batch_cancel_releases_a_prior_failure_vote(
    executor: PyExecutor,
) -> None:
    """After group retirement, post-batch cancellation must not block failure cleanup."""
    request = _request(1)
    _enable_context_first(executor, request)
    executor._fatal_error = None
    executor.active_requests = [request]
    session, aux = _published_receive(executor, request)
    session.process_kv_agent_result(0, 0, True, AgentResult.IN_DOUBT, _KV_RESULT_REPORT_TOKEN)
    _abort(executor, request)
    executor._handle_canceled_requests()
    executor.disagg.reap_gen_receives(0)
    executor._handle_canceled_requests()
    assert executor.canceled_req_ids == [1]
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    executor.resource_manager.free_resources.assert_not_called()

    session.process_kv_agent_result(
        0, 0, True, AgentResult.FAILED_QUIESCED, _KV_RESULT_REPORT_TOKEN
    )
    executor.disagg.reap_gen_receives(0)
    assert request.py_kv_receive_cancel_outcome is ReceiveCancellation.FAILED
    executor._handle_canceled_requests()
    assert executor.canceled_req_ids == []
    executor.disagg.reap_gen_receives(0)
    executor._flush_pending_transfer_responses()

    assert not request.is_finished_due_to_cancellation
    (response,) = executor.await_responses(timeout=timedelta(0))
    assert response.error_msg is not None
    executor.resource_manager.free_resources.assert_called_once_with(request)
    aux.free_slot.assert_called_once_with(0)
    executor.disagg.reap_gen_receives(0)
    assert executor.await_responses(timeout=timedelta(0)) == []
    executor.resource_manager.free_resources.assert_called_once_with(request)


@pytest.mark.parametrize("stopped", [False, True], ids=["shutting-down", "stopped"])
def test_context_first_failed_fence_handoff_contains_without_release(
    executor: PyExecutor, stopped: bool
) -> None:
    """Untrackable CUDA access triggers containment instead of unbounded retention."""
    destination = _request(1)
    _enable_context_first(executor, destination)
    destination.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    executor.active_requests = [destination]
    executor._fatal_error = None
    ready = Mock(spec=["query"])
    ready.query.return_value = False
    executor.disagg._defer_unready_gen_receives = True
    executor.disagg._unpublished_gen_receives[1] = (destination, ready)
    transceiver = executor.kv_cache_transceiver
    worker = transceiver._transfer_worker
    containment = Mock()
    watchdog = RetirementWatchdog(containment, clock=lambda: 0.0)
    worker._retirement_watchdog = watchdog

    def clean_up_test_thread() -> None:
        """Release only the controlled test's intentionally parked fatal thread."""
        watchdog._stopped.set()
        watchdog.wake.set()
        if watchdog._thread is not None:
            watchdog._thread.join(timeout=1)
            assert not watchdog._thread.is_alive()

    try:
        if stopped:
            watchdog.stop()
        else:
            watchdog.request_shutdown()
        executor._handle_errors(
            "earlier executor failure", requests=[destination], charge_budget=False
        )
        _abort(executor, destination)
        executor._cancel_waiting_disagg_requests([])
        executor.disagg.poll_gen_transfers()

        assert executor.disagg._unpublished_gen_receives == {1: (destination, ready)}
        assert executor.disagg._pending_gen_cancellation_intents == {1: True}
        assert transceiver._cancelled_gen_receives[1].unhanded_ready is ready
        assert transceiver._recv_reqs == {1: destination}
        assert destination.py_kv_receive_cancel_outcome is None
        assert watchdog.fatal is not None
        assert watchdog.fatal.failure_kind == "ownership_handoff"
        containment.assert_called_once_with(watchdog.fatal)
        executor.resource_manager.free_resources.assert_not_called()
        ready.query.return_value = True
        executor._cancel_waiting_disagg_requests([])
        executor.disagg.poll_gen_transfers()
        executor._sweep_transfer_held_failures()
        assert transceiver._cancelled_gen_receives[1].unhanded_ready is ready
        assert executor.await_responses(timeout=timedelta(0)) == []
        executor.resource_manager.free_resources.assert_not_called()
        with pytest.raises(RuntimeError, match="await group retirement"):
            transceiver.shutdown()
        assert transceiver._recv_reqs == {1: destination}
        containment.assert_called_once_with(watchdog.fatal)
    finally:
        clean_up_test_thread()


@pytest.mark.parametrize("published", [False, True], ids=["fence-only", "published"])
@pytest.mark.parametrize("diagnostics_raise", [False, True])
def test_context_first_cancel_emits_settlement_only_after_session_close(
    executor: PyExecutor,
    monkeypatch: pytest.MonkeyPatch,
    published: bool,
    diagnostics_raise: bool,
) -> None:
    """Settlement diagnostics observe physical retirement and cannot obstruct it."""
    request = _request(1)
    _enable_context_first(executor, request)
    executor.active_requests = [request]
    transceiver = executor.kv_cache_transceiver
    transceiver._mapping = Mapping()
    transceiver._instance_name = "gen"
    transceiver._dp_rank = 0
    if published:
        session, _ = _published_receive(executor, request)
    else:
        ready = Mock(spec=["query"])
        ready.query.return_value = False
        executor.disagg._defer_unready_gen_receives = True
        executor.disagg._unpublished_gen_receives[1] = (request, ready)
    observed_closed = []

    def observe_event(event: str, **kwargs) -> None:
        if event == "gen_transfer_settled":
            # Observe closure at emission, not only after the poll returns.
            observed_closed.append(session._closed)
        if diagnostics_raise:
            raise RuntimeError("diagnostic sink failed")

    emitted = Mock(side_effect=observe_event)
    monkeypatch.setattr(disagg_diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(disagg_diagnostics, "emit_event", emitted)
    _abort(executor, request)
    executor._cancel_waiting_disagg_requests([])
    executor.disagg.reap_gen_receives(0)
    executor._cancel_waiting_disagg_requests([])
    executor._flush_pending_transfer_responses()
    assert not any(call.args[0] == "gen_transfer_settled" for call in emitted.call_args_list)
    if published:
        session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
        session.process_aux_agent_result(0, AgentResult.SUCCESS)
    else:
        ready.query.return_value = True
    executor.disagg.reap_gen_receives(0)
    _assert_released_once(executor, request)
    settled = [call for call in emitted.call_args_list if call.args[0] == "gen_transfer_settled"]
    assert len(settled) == int(published)
    assert observed_closed == ([True] if published else [])
    if settled:
        assert settled[0].kwargs["session_retired"] is True
        assert settled[0].kwargs["outcome"] == "cancelled"
    executor.disagg.reap_gen_receives(0)
    assert len(
        [call for call in emitted.call_args_list if call.args[0] == "gen_transfer_settled"]
    ) == int(published)


@pytest.mark.parametrize(
    "state",
    [
        LlmRequestState.DISAGG_GENERATION_INIT,
        LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
        LlmRequestState.GENERATION_IN_PROGRESS,
    ],
)
@pytest.mark.parametrize("capacity_only", [False, True])
def test_pending_group_cancel_is_excluded_by_real_schedule(
    executor: PyExecutor, state: LlmRequestState, capacity_only: bool
) -> None:
    """Uncommitted cancellation blocks new scheduling without releasing its owner."""
    cancelled, healthy = _request(1), _request(2)
    _enable_context_first(executor, cancelled)
    cancelled.state = state
    healthy.state = LlmRequestState.GENERATION_IN_PROGRESS
    executor.active_requests = [cancelled, healthy]
    executor.kv_cache_manager = SimpleNamespace()
    executor.inflight_req_ids = set()
    executor.ctx_decoder_batcher = None
    executor.scheduler = Mock()
    executor.scheduler.schedule_request.side_effect = _ReachedScheduler
    if capacity_only:
        executor.kv_connector_manager = SimpleNamespace(capacity_only=True)
    _abort(executor, cancelled)
    if state != LlmRequestState.GENERATION_IN_PROGRESS:
        assert executor.disagg.cancel_gen_logically(cancelled) is ReceiveCancellation.PENDING
    with pytest.raises(_ReachedScheduler):
        executor._schedule()
    executor.scheduler.schedule_request.assert_called_once_with([healthy], set())
    assert executor.active_requests == [cancelled, healthy]
    assert not cancelled.is_finished
    executor.resource_manager.free_resources.assert_not_called()


@pytest.mark.parametrize("entry", ["waiting", "post_batch"])
@pytest.mark.parametrize("capacity_only", [False, True])
def test_latched_timeout_wins_before_first_user_cancel_vote(
    executor: PyExecutor, entry: str, capacity_only: bool
) -> None:
    """A recorded executor timeout cannot be cleared by a later user cancellation."""
    request = _request(1)
    _enable_context_first(executor, request)
    executor._fatal_error = None
    executor.active_requests = [request]
    session, aux = _published_receive(executor, request)
    if capacity_only:
        executor.kv_connector_manager = SimpleNamespace(
            capacity_only=True,
            release_unstarted_prefix_loads=Mock(),
            defer_load_termination=Mock(return_value=False),
        )
    request.py_kv_transfer_timed_out = True
    _abort(executor, request)
    cancel = (
        (lambda: executor._cancel_waiting_disagg_requests([]))
        if entry == "waiting"
        else executor._handle_canceled_requests
    )
    cancel()
    executor.disagg.reap_gen_receives(0)
    assert (
        executor.kv_cache_transceiver._cancelled_gen_receives[1].outcome
        is ReceiveCancellation.FAILED
    )
    cancel()
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert not request.is_finished_due_to_cancellation
    executor.resource_manager.free_resources.assert_not_called()
    session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    executor.disagg.reap_gen_receives(0)
    cancel()
    executor.disagg.reap_gen_receives(0)
    executor._flush_pending_transfer_responses()
    (response,) = executor.await_responses(timeout=timedelta(0))
    assert response.error_msg is not None
    assert request.py_kv_receive_cancel_outcome is ReceiveCancellation.FAILED
    executor.resource_manager.free_resources.assert_called_once_with(request)
    aux.free_slot.assert_called_once_with(0)


def test_protected_tp_cancel_with_capacity_connector_never_enters_receive_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admit the healthy TP2 receive while the protected cancellation stays unpublished."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    engines = [executor.__wrapped__(monkeypatch) for _ in range(2)]
    requests = [_request(1), _request(1)]

    def start_receive(request: LlmRequest) -> None:
        """Model the network boundary after real scheduler and admission processing."""
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS

    for rank, (engine, request) in enumerate(zip(engines, requests)):
        _enable_context_first(engine, request)
        healthy = _request(2)
        _enable_context_first(engine, healthy)
        dist = group.rank(rank)
        dist.mapping = Mapping(world_size=2, rank=rank, tp_size=2)
        engine.dist = dist
        engine.disagg._dist = dist
        tc = engine.kv_cache_transceiver
        tc._dist, tc._mapping = dist, dist.mapping
        tc._gen_need_sync = True
        tc._gen_allgather = dist.tp_allgather
        tc._init_gen_idle_poll(True)
        tc.prepare_context_requests = Mock()
        tc.request_and_receive_async = Mock(side_effect=start_receive)
        tc._ever_had_send_session = False
        tc._ctx_need_tp_sync = True
        tc._ctx_need_pp_sync = False
        tc._transfer_worker.sweep_stale_req_infos = Mock()
        engine.kv_connector_manager = SimpleNamespace(
            capacity_only=True,
            release_unstarted_prefix_loads=Mock(),
            defer_load_termination=Mock(return_value=False),
            has_pending_load=Mock(return_value=False),
        )
        engine.kv_cache_manager = SimpleNamespace(is_preemption_pending=Mock(return_value=False))
        engine.active_requests = [request, healthy]
        engine.inflight_req_ids = set()
        engine.ctx_decoder_batcher = None
        engine.enable_batch_waiting = False
        engine.benchmark_req_queues_size = 0
        engine._get_ctx_mla_kv_len_cap = lambda: None
        engine.resource_manager.resource_managers = {}
        engine.scheduler = Mock()
        engine.scheduler.schedule_request.side_effect = lambda candidates, _: SchedulerOutput(
            [], [], [], [], list(candidates), len(candidates)
        )
        engine.scheduler.take_lost_kv_requests.return_value = []
        engine._release_unused_connector_reservations = Mock()
        engine._poll_encoder_steps = Mock()
        engine._fetch_and_activate_new_requests = Mock(return_value=[request, healthy])
        engine._handle_control_request = Mock()
        engine._activate_completed_remote_tails = Mock()
        engine._init_iter_stats_if_sampled = Mock()
        engine._pad_attention_dp_dummy_request = Mock()
        engine._prefetch_for_context_requests = Mock()
        engine.drafter = None
        _abort(engine, request)

    def prepare(rank: int) -> None:
        batch, _ = engines[rank]._prepare_and_schedule_batch()
        assert batch is not None and batch.batch_size == 0

    group.run(prepare)
    for engine, request in zip(engines, requests):
        assert request.py_kv_receive_cancel_outcome is ReceiveCancellation.CANCELLED
        assert request not in engine.active_requests
        engine.scheduler.schedule_request.assert_called_once_with(engine.active_requests, set())
        assert [req.py_request_id for req in engine.active_requests] == [2]
        (healthy,) = engine.active_requests
        engine.kv_cache_transceiver.request_and_receive_async.assert_called_once_with(healthy)
        assert healthy.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        assert engine.disagg._started_gen_receives == {2: healthy}
        assert not healthy.is_finished
        engine.kv_connector_manager.release_unstarted_prefix_loads.assert_called()
        assert engine.dist.calls
    assert [name for name, _ in engines[0].dist.calls] == [
        name for name, _ in engines[1].dist.calls
    ]


@pytest.mark.parametrize("timed_out", [False, True], ids=["on-time", "latched-timeout"])
def test_keyed_receive_cannot_turn_recorded_timeout_into_first_token_success(
    monkeypatch: pytest.MonkeyPatch, timed_out: bool
) -> None:
    """A recorded timeout enters the receive vote before first-token close-out.

    Args:
        monkeypatch: Isolate native-runtime boundaries and the executor clock.
        timed_out: Record the timeout on one rank before the receive completes.
    """
    monkeypatch.setattr(coordinator_mod.time, "monotonic", lambda: 0.5)
    group = FakeDistGroup(world_size=2, tp_size=2)
    engines = [executor.__wrapped__(monkeypatch) for _ in range(2)]
    requests = [
        make_llm_request(
            1,
            RequestType.REQUEST_TYPE_GENERATION_ONLY,
            input_token_ids=list(range(8)),
            max_tokens=1,
        )
        for _ in engines
    ]
    sessions = []
    for rank, (engine, request) in enumerate(zip(engines, requests)):
        _enable_context_first(engine, request)
        dist = group.rank(rank)
        dist.mapping = Mapping(world_size=2, rank=rank, tp_size=2)
        engine.dist = engine.disagg._dist = dist
        engine.active_requests = [request]
        engine._fatal_error = None
        engine.enable_iter_perf_stats = False
        engine._is_kv_manager_v2 = False
        engine._benchmark_fill_phase_active = False
        engine.iter_counter = 0
        engine.perf_manager = Mock()
        engine.perf_manager.get_timestamp.return_value = 0.5
        engine.should_exclude_last_generation_logits = False
        engine._first_token_timer = None
        engine.sampler.max_seq_len = 64
        engine.resource_manager.resource_managers = {}
        tc = engine.kv_cache_transceiver
        tc._dist, tc._mapping = dist, dist.mapping
        tc._gen_need_sync = True
        tc._gen_allgather = dist.tp_allgather
        tc._init_gen_idle_poll(True)
        tc._pending_rank_info_admissions = {1: (request, None)}
        tc._kv_cache_manager = SimpleNamespace(get_history_length=lambda req: req.prompt_len)
        tc._need_aux_transfer = Mock(return_value=False)
        tc.commit_blocks_for_reuse = Mock()
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        request.py_kv_transfer_start_time = 0.0

        def publish(req: LlmRequest, engine: PyExecutor = engine) -> None:
            # The existing native fixture installs a representative timestamp;
            # keep this request's earlier keyed-admission start instead.
            original_start = req.py_kv_transfer_start_time
            session, _ = _published_receive(engine, req)
            req.py_kv_transfer_start_time = original_start
            sessions.append(session)

        tc._start_async_receive = Mock(side_effect=publish)
        assert engine._first_gen_token_finish_reason(request) is FinishReason.LENGTH

    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    assert len(sessions) == 2
    for rank, (engine, request) in enumerate(zip(engines, requests)):
        assert not engine.kv_cache_transceiver._pending_rank_info_admissions
        engine.disagg._flag_transfer_timeout(
            request, "generation", 1000, current_time=2.0 if timed_out and rank == 0 else 0.5
        )
        assert request.py_kv_transfer_timed_out is (timed_out and rank == 0)
    for session in sessions:
        session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
        session.process_aux_agent_result(0, AgentResult.SUCCESS)

    # First poll propagates a one-rank timeout. The second supplies the peer's
    # real vote. Neither interval may expose late completion as decode-ready.
    for _ in range(2 if timed_out else 1):
        group.run(lambda rank: engines[rank].disagg.poll_gen_transfers())
        group.run(lambda rank: engines[rank]._finish_gen_requests_ended_by_first_token())
        if timed_out:
            assert all(not request.is_finished_due_to_length for request in requests)
            for engine in engines:
                engine.kv_cache_transceiver.commit_blocks_for_reuse.assert_not_called()

    group.run(lambda rank: engines[rank]._flush_pending_transfer_responses())
    (response,) = engines[0].await_responses(timeout=timedelta(0))
    assert (response.error_msg is not None) is timed_out
    for engine, request in zip(engines, requests):
        assert request.is_finished_due_to_length is not timed_out
        assert request.py_kv_receive_cancel_outcome is (
            ReceiveCancellation.FAILED if timed_out else None
        )
        assert not engine.active_requests
        assert not engine.kv_cache_transceiver._recv_reqs
        engine.resource_manager.free_resources.assert_called_once_with(request)
        if not timed_out:
            engine.kv_cache_transceiver.commit_blocks_for_reuse.assert_called_once_with(request)
    assert [name for name, _ in engines[0].dist.calls] == [
        name for name, _ in engines[1].dist.calls
    ]


@pytest.mark.parametrize("published", [False, True], ids=["unpublished-fence", "published-session"])
@pytest.mark.parametrize("prior_failure", [False, True], ids=["cancel-first", "failure-first"])
def test_fp4_receive_cancellation_disposition_survives_retry_and_retirement(
    executor: PyExecutor, published: bool, prior_failure: bool
) -> None:
    """Legacy logical disposition stays fixed independently of physical release.

    Args:
        executor: Real executor and native cancellation with controlled resources.
        published: Whether a receive session exists or only local CUDA work remains.
        prior_failure: Whether failure precedes the first logical cancellation.
    """
    request = _request(1)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    tc = executor.kv_cache_transceiver
    ready = None
    if published:
        session, aux = _published_receive(executor, request)
    else:
        ready = Mock(query=Mock(return_value=False))
    expected = ReceiveCancellation.FAILED if prior_failure else ReceiveCancellation.CANCELLED
    initial_state = request.state
    assert tc.cancel_gen_logically(request, ready, prior_failure=prior_failure) is expected
    assert tc.cancel_gen_logically(request) is expected
    assert tc.cancel_gen_logically(request, prior_failure=not prior_failure) is expected
    assert request.py_kv_receive_cancel_outcome is None
    assert request.state == initial_state
    assert not tc.cancel_request(request)
    assert tc.has_inflight_transfer(request)
    executor.resource_manager.free_resources.assert_not_called()

    if published:
        aux.free_slot.assert_not_called()
        session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
        session.process_aux_agent_result(0, AgentResult.SUCCESS)
    else:
        ready.query.return_value = True
    assert tc.cancel_request(request)
    assert not tc.has_inflight_transfer(request)
    assert tc.cancel_gen_logically(request) is expected
    assert tc.cancel_gen_logically(request, prior_failure=not prior_failure) is expected
    assert request.py_kv_receive_cancel_outcome is None
    assert request.state == initial_state
    assert not tc._recv_reqs
    assert not tc._recv_sessions
    executor.resource_manager.free_resources.assert_not_called()
    if published:
        aux.free_slot.assert_called_once_with(0)


@pytest.mark.parametrize("pressure", ["dealer-full", "publication-lock"])
def test_group_cancellation_backpressure_does_not_block_healthy_receive(
    monkeypatch: pytest.MonkeyPatch, pressure: str
) -> None:
    """Keep TP progress and ownership while an unsent notification must retry."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    engines = [executor.__wrapped__(monkeypatch) for _ in range(2)]
    cancelled_sessions, sockets, auxiliary = [], [], []
    requests, healthy_requests = [], []
    for rank, engine in enumerate(engines):
        request, healthy = _request(1), _request(2)
        _enable_context_first(engine, request)
        _enable_context_first(engine, healthy)
        dist = group.rank(rank)
        dist.mapping = Mapping(world_size=2, rank=rank, tp_size=2)
        engine.dist = engine.disagg._dist = dist
        tc = engine.kv_cache_transceiver
        tc._dist, tc._mapping = dist, dist.mapping
        tc._gen_need_sync = True
        tc._gen_allgather = dist.tp_allgather
        tc._init_gen_idle_poll(True)
        tc._kv_cache_manager = SimpleNamespace(get_history_length=lambda req: req.prompt_len)
        engine.active_requests = [request, healthy]
        session, aux = _published_receive(engine, request)
        healthy_session, _ = _published_receive(engine, healthy)
        healthy_session.process_kv_agent_result(
            0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN
        )
        healthy_session.process_aux_agent_result(0, AgentResult.SUCCESS)
        dealer = object.__new__(ZMQMessenger)
        dealer._mode = "DEALER"
        dealer._socket = Mock()
        dealer._socket.send_multipart.side_effect = (
            zmq.Again() if rank == 0 and pressure == "dealer-full" else None
        )
        receiver = session._receiver
        receiver._get_session = lambda _rid, session=session: session
        receiver._get_or_connect_dealer = Mock(return_value=dealer)
        receiver.send_cancel_to_senders = MethodType(Receiver.send_cancel_to_senders, receiver)
        assert engine.disagg.cancel_gen_logically(request) is ReceiveCancellation.PENDING
        cancelled_sessions.append(session)
        sockets.append(dealer._socket)
        auxiliary.append(aux)
        requests.append(request)
        healthy_requests.append(healthy)
    first_drain = [session._retirement._drain_started for session in cancelled_sessions]
    held_lock = cancelled_sessions[0]._publication_lock
    lock_release_guard = None
    if pressure == "publication-lock":
        held_lock.acquire()
        lock_release_guard = threading.Timer(30.0, held_lock.release)
        lock_release_guard.start()
    try:
        group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
        if lock_release_guard is not None:
            assert held_lock.locked(), (
                "notification waited for the hang guard to release publication"
            )
    finally:
        if lock_release_guard is not None:
            lock_release_guard.cancel()
            lock_release_guard.join()
            if held_lock.locked():
                held_lock.release()
    assert cancelled_sessions[0]._cancelled_sender_endpoints == set()
    assert sockets[0].send_multipart.call_count == int(pressure == "dealer-full")
    for engine, request, healthy, aux in zip(engines, requests, healthy_requests, auxiliary):
        assert (
            engine.kv_cache_transceiver._cancelled_gen_receives[1].outcome
            is ReceiveCancellation.CANCELLED
        )
        assert engine.kv_cache_transceiver._recv_reqs == {1: request}
        assert healthy.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        assert request.py_kv_receive_cancel_outcome is None
        aux.free_slot.assert_not_called()
        engine.resource_manager.free_resources.assert_not_called()
    sockets[0].send_multipart.side_effect = None
    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    assert cancelled_sessions[0]._cancelled_sender_endpoints == {"ctx"}
    assert sockets[0].send_multipart.call_args.kwargs == {"flags": zmq.DONTWAIT}
    assert [session._retirement._drain_started for session in cancelled_sessions] == first_drain
    notification_counts = [socket.send_multipart.call_count for socket in sockets]
    for session, socket in zip(cancelled_sessions, sockets):
        session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
        session.process_aux_agent_result(0, AgentResult.SUCCESS)
        socket.send_multipart.side_effect = AssertionError("drained session must not notify")
    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    assert [socket.send_multipart.call_count for socket in sockets] == notification_counts
    for engine, request, aux in zip(engines, requests, auxiliary):
        assert request.py_kv_receive_cancel_outcome is ReceiveCancellation.CANCELLED
        assert not engine.kv_cache_transceiver._recv_reqs
        aux.free_slot.assert_called_once_with(0)
    assert [name for name, _ in engines[0].dist.calls] == [
        name for name, _ in engines[1].dist.calls
    ]


@pytest.mark.parametrize("mode", ["ROUTER", "REQ", "REP"])
def test_nonblocking_notification_rejects_non_dealer_socket(mode: str) -> None:
    """A routed or request/reply socket cannot claim the DEALER queueing contract."""
    messenger = object.__new__(ZMQMessenger)
    messenger._mode = mode
    messenger._socket = Mock()
    with pytest.raises(ValueError, match="requires a DEALER"):
        messenger.try_send([b"cancel"])
    messenger._socket.send_multipart.assert_not_called()


@pytest.mark.parametrize("context_first", [False, True], ids=["fp4-bridge", "bf16-context-first"])
@pytest.mark.parametrize("peer_first", [False, True], ids=["local-first", "peer-first"])
def test_receive_peer_cancellation_preserves_the_winning_disposition(
    executor: PyExecutor, context_first: bool, peer_first: bool
) -> None:
    """A prior peer cancel remains failure; a late peer cancel cannot rewrite local intent.

    Args:
        executor: Real cancellation and receive ownership with controlled resources.
        context_first: Whether the receive commits a group or legacy local outcome.
        peer_first: Whether the peer wins before the worker accepts user cancellation.
    """
    request = _request(1)
    if context_first:
        _enable_context_first(executor, request)
    executor._fatal_error = None
    executor.active_requests = [request]
    session, aux = _published_receive(executor, request)
    if peer_first:
        assert session.cancel_local(by_peer=True)
    _abort(executor, request)
    executor._cancel_waiting_disagg_requests([])
    if context_first:
        executor.disagg.reap_gen_receives(0)
        executor._cancel_waiting_disagg_requests([])
    expected = ReceiveCancellation.FAILED if peer_first else ReceiveCancellation.CANCELLED
    assert executor.kv_cache_transceiver.cancel_gen_logically(request) is expected
    assert session.cancel_local(by_peer=True) is False
    assert session.cancelled_by_peer is peer_first
    assert executor.kv_cache_transceiver.cancel_gen_logically(request) is expected
    assert request.is_finished_due_to_cancellation is not peer_first
    assert request.py_kv_receive_cancel_outcome is None
    aux.free_slot.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()

    executor._flush_pending_transfer_responses()
    executor.resource_manager.free_resources.assert_not_called()
    session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS, _KV_RESULT_REPORT_TOKEN)
    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    executor.disagg.reap_gen_receives(0)
    executor._cancel_waiting_disagg_requests([])
    executor.disagg.reap_gen_receives(0)
    executor._flush_pending_transfer_responses()
    (response,) = executor.await_responses(timeout=timedelta(0))
    assert (response.error_msg is not None) is peer_first
    assert request.is_finished_due_to_cancellation is not peer_first
    assert executor.kv_cache_transceiver.cancel_gen_logically(request) is expected
    assert (
        executor.kv_cache_transceiver.cancel_gen_logically(request, prior_failure=True) is expected
    )
    assert not executor.kv_cache_transceiver._recv_reqs
    aux.free_slot.assert_called_once_with(0)
    executor.resource_manager.free_resources.assert_called_once_with(request)


def test_post_vote_owner_close_refusal_enters_existing_rank_crash_protocol(
    executor: PyExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused owner close retains roots and reaches the existing world-stop path."""
    from tensorrt_llm._torch.pyexecutor import py_executor as executor_module

    request = _request(1)
    _enable_context_first(executor, request)
    executor.active_requests = [request]
    executor.dist.world_size = executor.dist.tp_size = 2
    tc = executor.kv_cache_transceiver
    tc._gen_need_sync = True
    tc._gen_allgather = lambda payload: [payload, payload]
    tc._init_gen_idle_poll(True)
    ready = Mock(query=Mock(side_effect=[True, AssertionError("accepted fence was queried again")]))
    executor.disagg._unpublished_gen_receives[1] = (request, ready)
    assert executor.disagg.cancel_gen_logically(request) is ReceiveCancellation.PENDING
    owner = tc._cancelled_receive_fences[1].owner
    monkeypatch.setattr(owner, "close", Mock(return_value=False))
    events = []
    watchdog = SimpleNamespace(deadline=17.0, cancel=lambda: events.append("cancel-watchdog"))

    def start_watchdog(world_size: int, *, error_delivered: threading.Event) -> SimpleNamespace:
        assert world_size == 2
        assert error_delivered is executor._event_loop_error_delivered
        events.append("arm-watchdog")
        return watchdog

    def kill_world(world_size: int, *, deadline: float, error_delivered: threading.Event) -> None:
        assert world_size == 2 and deadline == watchdog.deadline
        assert error_delivered is executor._event_loop_error_delivered
        events.append("kill-world")

    monkeypatch.setattr(executor_module, "start_rank_crash_kill_watchdog", start_watchdog)
    monkeypatch.setattr(executor_module, "hard_kill_on_rank_crash", kill_world)
    monkeypatch.setattr(executor_module, "host_profiler_context", lambda **_: nullcontext())
    monkeypatch.setattr(executor_module, "customized_gc_thresholds", lambda _: nullcontext())
    executor._apply_dynamo_recompile_limit = Mock()
    executor.garbage_collection_gen0_threshold = None
    executor._event_loop_error_delivered = threading.Event()
    executor._executor_loop_cleanup = lambda: events.append("cleanup")
    executor.event_loop = lambda: executor.disagg.reap_gen_receives(0)
    with pytest.raises(RuntimeError, match="lost safe retirement evidence"):
        executor._event_loop_wrapper()
    assert events == ["arm-watchdog", "cleanup", "cancel-watchdog", "kill-world"]
    ready.query.assert_called_once_with()
    owner.close.assert_called_once_with()
    assert tc._cancelled_gen_receives[1].outcome is ReceiveCancellation.CANCELLED
    assert tc._cancelled_receive_fences[1].ready is ready
    assert tc._recv_reqs[1] is request
    assert request.py_kv_receive_cancel_outcome is None
    executor.resource_manager.free_resources.assert_not_called()


@pytest.mark.parametrize("fatal_after_settlement", [False, True], ids=["settles", "peer-fatal"])
def test_context_first_receive_keeps_accepted_fence_evidence(
    monkeypatch: pytest.MonkeyPatch, fatal_after_settlement: bool
) -> None:
    """A safe local fence stays safe while its TP peer drains, unless fatal wins.

    Args:
        monkeypatch: Isolate the native-runtime harness boundaries.
        fatal_after_settlement: Expire a different owner after accepting local evidence.
    """
    group = FakeDistGroup(world_size=2, tp_size=2)
    engines = [executor.__wrapped__(monkeypatch) for _ in range(2)]
    requests = [_request(1), _request(1)]
    ready = [
        Mock(query=Mock(side_effect=[True, RuntimeError("accepted fence was queried again")])),
        Mock(query=Mock(return_value=False)),
    ]
    clock = [0.0]
    containment = Mock()
    watchdog = RetirementWatchdog(containment, clock=lambda: clock[0])
    for rank, (engine, request) in enumerate(zip(engines, requests)):
        _enable_context_first(engine, request)
        dist = group.rank(rank)
        dist.mapping = Mapping(world_size=2, rank=rank, tp_size=2)
        engine.dist = engine.disagg._dist = dist
        engine.active_requests = [request]
        tc = engine.kv_cache_transceiver
        tc._dist, tc._mapping = dist, dist.mapping
        tc._gen_need_sync = True
        tc._gen_allgather = dist.tp_allgather
        tc._init_gen_idle_poll(True)
        if rank == 0:
            tc._transfer_worker._retirement_watchdog = watchdog
        engine.disagg._unpublished_gen_receives[1] = (request, ready[rank])
        assert engine.disagg.cancel_gen_logically(request) is ReceiveCancellation.PENDING

    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    for engine, request in zip(engines, requests):
        tc = engine.kv_cache_transceiver
        assert tc._cancelled_gen_receives[1].outcome is ReceiveCancellation.CANCELLED
        assert tc._recv_reqs == {1: request}
        assert request.py_kv_receive_cancel_outcome is None
        engine.resource_manager.free_resources.assert_not_called()
    ready[0].query.assert_called_once_with()
    first_owner = engines[0].kv_cache_transceiver._cancelled_receive_fences[1].owner
    assert first_owner.can_retire()

    if fatal_after_settlement:
        other_owner = watchdog.create_owner(99, "receive", timeout_s=1.0)
        other_owner.retain_unproven(object(), "unsettled unrelated access")
        clock[0] = 1.01
        watchdog.progress()
        containment.assert_called_once_with(watchdog.fatal)
        assert watchdog.fatal is not None
    ready[1].query.return_value = True
    group.run(lambda rank: engines[rank].disagg.reap_gen_receives(0))
    ready[0].query.assert_called_once_with()
    assert ready[1].query.call_count == 2
    for engine, request in zip(engines, requests):
        tc = engine.kv_cache_transceiver
        if fatal_after_settlement:
            assert tc._recv_reqs == {1: request}
            assert tc._cancelled_gen_receives[1].outcome is ReceiveCancellation.CANCELLED
            assert request.py_kv_receive_cancel_outcome is None
        else:
            assert not tc._recv_reqs
            assert not tc._cancelled_gen_receives
            assert not tc._cancelled_receive_fences
            assert request.py_kv_receive_cancel_outcome is ReceiveCancellation.CANCELLED
            tc._transfer_worker._retirement_watchdog.require_retired()
        engine.resource_manager.free_resources.assert_not_called()
    assert [name for name, _ in engines[0].dist.calls] == [
        name for name, _ in engines[1].dist.calls
    ]
