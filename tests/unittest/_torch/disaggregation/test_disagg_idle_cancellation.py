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
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.disaggregation.base import TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import BindKvCacheTransceiver
from tensorrt_llm._torch.disaggregation.native.bounce.impl import NoBounceTransport
from tensorrt_llm._torch.disaggregation.native.extent import Cancelled, Chunk, Failed
from tensorrt_llm._torch.disaggregation.native.handle import TaskHandle
from tensorrt_llm._torch.disaggregation.native.retirement import RetirementWatchdog
from tensorrt_llm._torch.disaggregation.native.transfer import AgentResult, KVRecvTask, RxSession
from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_mod
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.disagg_adapter import (
    PyExecutorEffects,
    PyExecutorRequestRegistry,
)
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmResponse, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor.base_worker import BaseWorker

pytestmark = pytest.mark.cpu_only


def _request(request_id: int, *, context: bool = False) -> LlmRequest:
    """Create a native request waiting for its generation-first peer.

    Args:
        request_id: Executor and transfer identity.
        context: Whether to create the context-side request.

    Returns:
        A request that has not run a model batch.
    """
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(8)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        draft_tokens=None,
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
    engine.dist = SimpleNamespace(
        rank=0, tp_size=1, pp_size=1, world_size=1, mapping=SimpleNamespace(tp_group=[0])
    )
    engine.executor_request_queue = ExecutorRequestQueue(
        dist=engine.dist, max_batch_size=8, enable_iter_perf_stats=False, batch_wait_timeout_ms=0.0
    )
    engine.response_cv = threading.Condition()
    engine.responses = {}
    engine.result_wait_queues = {}
    engine._disagg_pp_termination_handler = None
    engine._prefetched_request_ids = set()
    engine.resource_manager = Mock(spec=["free_resources"])
    engine.sampler = Mock(spec=["release_request"])
    engine.model_engine = SimpleNamespace(wait_for_input_copy=Mock(), is_spec_decode=False)
    engine.draft_model_engine = None
    engine._maybe_record_hang_diagnostic_phase = Mock()
    engine.async_transfer_manager = SimpleNamespace(requests_in_transfer=lambda: {})
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
    transceiver.kv_transfer_timeout_ms = 1000
    engine.kv_cache_transceiver = transceiver
    engine._disagg_coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=engine.async_transfer_manager,
        kv_cache_manager=None,
        dist=engine.dist,
        effects=PyExecutorEffects(engine),
        registry=PyExecutorRequestRegistry(engine),
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
    )
    return engine


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
    (response,) = executor.responses[request.py_request_id]
    assert response.error_msg is None
    executor.sampler.release_request.assert_called_once_with(request)
    executor.resource_manager.free_resources.assert_called_once_with(request)
    executor._flush_pending_transfer_responses()
    assert len(executor.responses[request.py_request_id]) == 1
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
        executor.kv_connector_manager = Mock(capacity_only=True)
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

    def stop_before_blocking(timeout: timedelta | None) -> None:
        """Check publication and cleanup before the real fetch would wait forever."""
        events.append("fetch")
        assert timeout is None
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
            assert executor.responses == {}
            executor.dist.tp_gather.assert_not_called()
            executor.dist.allgather.assert_not_called()
        else:
            (response,) = executor.responses[1]
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
    executor.active_requests = [request]
    _abort(executor, request)

    assert executor._cancel_waiting_disagg_requests([]) == []

    assert executor.active_requests == [request]
    assert executor.canceled_req_ids == [1]
    assert not request.is_finished
    assert executor._pending_transfer_responses == []
    executor.resource_manager.free_resources.assert_not_called()


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
    executor.kv_connector_manager = Mock()
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
    receiver = SimpleNamespace(
        _enforce_physical_ownership=True,
        _bounce=NoBounceTransport(),
        _registrar=SimpleNamespace(
            self_rank_info=SimpleNamespace(instance_name="gen", instance_rank=0)
        ),
        setup_session=Mock(),
        clear_session=Mock(),
        send_cancel_to_senders=Mock(side_effect=lambda _rid, endpoints: set(endpoints)),
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
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    request.py_kv_transfer_start_time = 12.0
    return session, aux


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
        session.process_kv_agent_result(0, 0, True, AgentResult.FAILED)
    _abort(executor, request)
    cancel()
    assert not session.resources_drained()
    assert executor.canceled_req_ids == [1]
    executor.resource_manager.free_resources.assert_not_called()
    aux.free_slot.assert_not_called()
    if not failure_first:
        session.process_kv_agent_result(0, 0, True, AgentResult.FAILED)

    expected_status = SessionStatus.ERROR if failure_first else SessionStatus.CANCELLED
    expected_outcome = Failed if failure_first else Cancelled
    assert session.status is expected_status
    assert isinstance(handle.poll(), expected_outcome)
    cancel()
    assert executor.active_requests == [request]
    assert not request.is_finished
    assert executor.responses == {}
    aux.free_slot.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()

    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    assert session.resources_drained()
    if poll_first:
        executor.disagg.reap_gen_receives(0)
    cancel()
    assert executor.canceled_req_ids == []
    assert session.status is expected_status
    assert isinstance(handle.poll(), expected_outcome)
    if failure_first:
        assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
        assert not request.is_finished_due_to_cancellation
        executor.disagg.reap_gen_receives(0)
    elif boundary == "post_batch":
        executor._handle_responses()
    executor._flush_pending_transfer_responses()

    assert executor.active_requests == []
    (response,) = executor.responses[request.py_request_id]
    assert (response.error_msg is not None) is failure_first
    assert request.is_finished_due_to_cancellation is not failure_first
    executor.resource_manager.free_resources.assert_called_once_with(request)
    aux.free_slot.assert_called_once_with(0)
    assert transceiver._recv_sessions == transceiver._recv_reqs == {}
    cancel()
    executor.disagg.reap_gen_receives(0)
    executor._flush_pending_transfer_responses()
    assert len(executor.responses[request.py_request_id]) == 1
    executor.resource_manager.free_resources.assert_called_once_with(request)
    aux.free_slot.assert_called_once_with(0)
    watchdog.require_retired()
    watchdog.stop()


@pytest.mark.parametrize("last_resource", ["kv", "aux"])
def test_worker_cancel_retains_published_kv_and_aux_until_both_settle(
    executor: PyExecutor, last_resource: str
) -> None:
    """Neither KV nor AUX completion alone permits cancelled receive reuse."""
    request = _request(1)
    executor.active_requests = [request]
    session, aux = _published_receive(executor, request)
    handle = TaskHandle(session, session._kv_tasks[0], token_end=8)
    _abort(executor, request)

    def complete(resource: str) -> None:
        """Deliver terminal evidence for one of the session's two resources."""
        if resource == "kv":
            session.process_kv_agent_result(0, 0, True, AgentResult.SUCCESS)
        else:
            session.process_aux_agent_result(0, AgentResult.SUCCESS)

    for resource in (None, "aux" if last_resource == "kv" else "kv"):
        if resource is not None:
            complete(resource)
        executor._cancel_waiting_disagg_requests([])
        executor._flush_pending_transfer_responses()
        assert executor.active_requests == [request]
        assert executor.canceled_req_ids == [1]
        assert request.py_kv_transfer_start_time == 12.0
        assert session.status is SessionStatus.CANCELLED
        assert isinstance(handle.poll(), Cancelled)
        assert executor.responses == {}
        executor.sampler.release_request.assert_not_called()
        executor.resource_manager.free_resources.assert_not_called()
        aux.free_slot.assert_not_called()

    complete(last_resource)
    executor._cancel_waiting_disagg_requests([])
    _assert_released_once(executor, request)
    assert isinstance(handle.poll(), Cancelled)
    assert not executor.kv_cache_transceiver._recv_sessions
    aux.free_slot.assert_called_once_with(0)
    complete(last_resource)
    executor._cancel_waiting_disagg_requests([])
    executor._flush_pending_transfer_responses()
    executor.resource_manager.free_resources.assert_called_once_with(request)


class _ReachedScheduler(Exception):
    """Stop the real preparation path before model/capacity scheduling."""


@pytest.mark.parametrize("context, bridge_enabled", [(False, True), (True, True), (True, False)])
def test_preparation_applies_worker_cancel_before_publication_or_scheduling(
    executor: PyExecutor, context: bool, bridge_enabled: bool
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
    if not context:
        cancelled.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        executor.disagg._unpublished_gen_receives[1] = (cancelled, ready)
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

    transceiver.prepare_context_requests.assert_called_once_with([])
    transceiver.request_and_receive_async.assert_not_called()
    ready.query.assert_not_called()
    assert executor.active_requests == [healthy]
    assert not healthy.is_finished
    _assert_released_once(executor, cancelled)
