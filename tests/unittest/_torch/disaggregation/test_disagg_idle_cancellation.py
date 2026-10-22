# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker cancellation must progress without a model batch in the ownership bridge.

The worker, request queue, executor cancellation/response methods, coordinator,
and receive-session ownership are real. Network endpoints, GPU readiness and
resource allocation are controlled boundaries; these CPU tests require the
native TRT-LLM runtime but do not qualify GPU/NIXL deployment or DMA fencing.
"""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.base import Cancelled, Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native.bounce.impl import NoBounceTransport
from tensorrt_llm._torch.disaggregation.native.handle import TaskHandle
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
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
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
    engine.model_engine = None
    engine.draft_model_engine = None
    engine._maybe_record_hang_diagnostic_phase = Mock()
    engine.async_transfer_manager = SimpleNamespace(requests_in_transfer=lambda: {})
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._fp4_mla_bridge_enabled = True
    transceiver._ever_had_recv_session = False
    transceiver._gen_need_sync = False
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
    executor.resource_manager.free_resources.assert_called_once_with(request)
    executor._flush_pending_transfer_responses()
    assert len(executor.responses[request.py_request_id]) == 1
    executor.resource_manager.free_resources.assert_called_once_with(request)


def test_queued_cancellation_removes_only_the_target_before_activation(
    executor: PyExecutor,
) -> None:
    """A queued abort removes only its target without allocating resources."""
    request = _request(1)
    executor.waiting_queue.add_requests([RequestQueueItem(1), RequestQueueItem(2)])
    _abort(executor, request)

    assert executor._cancel_waiting_disagg_requests([]) == []

    assert [item.id for item in executor.waiting_queue] == [2]
    assert executor.canceled_req_ids == []
    assert executor.active_requests == []
    assert executor._pending_transfer_responses == []
    executor.resource_manager.free_resources.assert_not_called()


@pytest.mark.parametrize("context", [False, True])
def test_worker_abort_settles_waiting_request_without_model_batch(
    executor: PyExecutor, context: bool
) -> None:
    """Duplicate aborts settle idle work once and preserve a healthy sibling."""
    cancelled, healthy = _request(1, context=context), _request(2, context=context)
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


def test_worker_abort_settles_context_waiting_from_previous_pass(executor: PyExecutor) -> None:
    """Cancel an earlier peer wait once without reviving it on late readiness."""
    cancelled, healthy = _request(1, context=True), _request(2, context=True)
    for request in (cancelled, healthy):
        request.state = LlmRequestState.CONTEXT_INIT
    executor.active_requests = [cancelled, healthy]
    transceiver = executor.kv_cache_transceiver
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


@pytest.mark.parametrize("context", [False, True])
def test_existing_compute_request_keeps_post_batch_cancellation(
    executor: PyExecutor, context: bool
) -> None:
    """Model-active work retains the existing post-sampler cancel boundary."""
    request = _request(1, context=context)
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


def test_disabled_bridge_preserves_existing_cancellation_path(executor: PyExecutor) -> None:
    """The new idle cancellation path does not activate outside the bridge."""
    request = _request(1)
    executor.kv_cache_transceiver._fp4_mla_bridge_enabled = False
    executor.active_requests = [request]
    _abort(executor, request)

    assert executor._cancel_waiting_disagg_requests([request]) == [request]

    assert executor.active_requests == [request]
    assert executor.canceled_req_ids == [1]
    assert not request.is_finished
    assert executor._pending_transfer_responses == []


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
    """A late abort preserves the request's already committed finish reason."""
    request = _request(1)
    request.finish_by_reason(FinishReason.LENGTH)
    executor.active_requests = [request]
    _abort(executor, request)

    executor._cancel_waiting_disagg_requests([])

    assert request.is_finished_due_to_length
    assert not request.is_finished_due_to_cancellation


def _published_receive(executor: PyExecutor, request: LlmRequest) -> tuple[RxSession, Mock]:
    """Publish real receive claims against controlled network and AUX storage.

    Args:
        executor: Executor holding the receive session and request.
        request: Request whose destination is exposed to one writer.

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
    session = RxSession(request.py_request_id, request.py_disaggregated_params, receiver, aux)

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


@pytest.mark.parametrize("context", [False, True])
def test_preparation_applies_worker_cancel_before_publication_or_scheduling(
    executor: PyExecutor, context: bool
) -> None:
    """The real preparation path applies abort before admitting more work."""
    cancelled, healthy = _request(1, context=context), _request(2)
    if context:
        cancelled.state = LlmRequestState.CONTEXT_INIT
    executor.active_requests = [cancelled, healthy]
    _abort(executor, cancelled)
    transceiver = executor.kv_cache_transceiver
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
