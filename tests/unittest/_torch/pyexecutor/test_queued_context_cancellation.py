# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete never-activated CTX cancellation through native request/result objects.

The queue, fetch path and response handling are real. Admission, rank transport
and model/resource work are controlled boundaries; these CPU tests do not
qualify in-flight transfer cancellation or GPU/NIXL retirement.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse, executor_request_to_llm_request
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_delivery import ResponseChannel
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue, WaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import PriorityWaitingQueue
from tensorrt_llm.bindings.executor import (
    FinishReason,
    OutputConfig,
    Request,
    RequestType,
    SamplingConfig,
)

pytestmark = pytest.mark.cpu_only


def _request(
    *,
    client_id: int | None = 101,
    request_type: RequestType = RequestType.REQUEST_TYPE_CONTEXT_ONLY,
    priority: float = 0.0,
    sequences: int = 1,
    beams: int = 1,
    streaming: bool = False,
) -> Request:
    """Create accepted CTX input with native beam and sequence configuration."""
    return Request(
        input_token_ids=[1, 2, 3],
        max_tokens=4,
        client_id=client_id,
        type=request_type,
        priority=priority,
        sampling_config=SamplingConfig(
            beam_width=beams, num_return_sequences=beams if beams > 1 else sequences
        ),
        output_config=OutputConfig(exclude_input_from_output=True),
        streaming=streaming,
    )


@pytest.fixture
def executor() -> PyExecutor:
    """Keep fetch and response methods real while isolating model and transport work."""
    engine = object.__new__(PyExecutor)
    engine.dist = SimpleNamespace(
        rank=0,
        tp_rank=0,
        tp_size=1,
        world_size=1,
        has_pp=False,
        cp_rank=0,
        cp_size=1,
        cp_config={},
        mapping=SimpleNamespace(tp_group=[0]),
    )
    engine.active_requests = []
    engine.canceled_req_ids = []
    engine.waiting_queue = FCFSWaitingQueue()
    engine._pending_transfer_responses = []
    engine._pending_response_terminations = []
    engine._response_channel = ResponseChannel()
    engine.enable_attention_dp = False
    engine.gather_all_responses = False
    engine.kv_connector_manager = None
    engine.kv_cache_transceiver = SimpleNamespace(_fp4_mla_bridge_enabled=False)
    engine.executor_request_queue = ExecutorRequestQueue(
        dist=engine.dist,
        max_batch_size=8,
        enable_iter_perf_stats=True,
        batch_wait_timeout_ms=0,
    )
    engine._kv_load_tracker = SimpleNamespace(enabled=False, begin_step=Mock(return_value=None))
    engine._kv_cache_capacity = 0
    engine.adp_router = SimpleNamespace(exclude_retiring_requests=False)
    engine.control_requests = []
    engine.request_accumulated = []
    engine.is_shutdown = False
    engine._disable_mpi = False
    engine._has_pending_connector_transfers = Mock(return_value=False)
    engine.hang_detector = SimpleNamespace(pause=nullcontext)
    engine.request_broadcaster = SimpleNamespace(broadcast=lambda requests, **kwargs: requests)
    engine.enable_iter_perf_stats = False
    engine.num_fetch_requests = 0
    engine.should_exclude_last_generation_logits = False
    engine._pop_from_waiting_queue = Mock(return_value=([], {}))
    engine._maybe_record_hang_diagnostic_phase = Mock()
    engine._terminate_request = Mock(side_effect=AssertionError("unactivated request teardown"))
    engine.resource_manager = Mock(spec=["free_resources"])
    engine.sampler = Mock(spec=["release_request"])
    return engine


def _assert_no_resource_cleanup(executor: PyExecutor) -> None:
    """Unactivated requests must not enter active-request resource teardown."""
    assert executor.active_requests == []
    assert executor._pending_response_terminations == []
    executor._terminate_request.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()
    executor.sampler.release_request.assert_not_called()


def _results(executor: PyExecutor, request_id: int) -> list[LlmResponse]:
    """Read responses already delivered by cancellation, without driving another batch."""
    responses = executor._response_channel.responses[request_id]
    for response in responses:
        assert not response.has_error()
        if isinstance(response.result._result, bytes):
            response.result.deserialize()
    return responses


@pytest.mark.parametrize("queue_type", [FCFSWaitingQueue, PriorityWaitingQueue])
@pytest.mark.parametrize("already_waiting", [False, True])
def test_fetch_completes_queued_context_cancel_without_admission(
    executor: PyExecutor, queue_type: type[WaitingQueue], already_waiting: bool
) -> None:
    """Real fetch must respond even when capacity never admits another request."""
    executor.waiting_queue = queue_type()
    request_id = executor.executor_request_queue.enqueue_request(_request())
    if already_waiting:
        executor._fetch_and_enqueue_requests(executor.waiting_queue, 0)
    executor.executor_request_queue.enqueue_cancel_request(request_id)

    def no_capacity(waiting_queue: WaitingQueue, *_: object) -> tuple[list, dict]:
        """Reject admission after verifying cancellation already left the queue."""
        assert request_id not in {item.id for item in waiting_queue}
        return [], {}

    executor._pop_from_waiting_queue.side_effect = no_capacity
    assert executor._fetch_new_requests(executor.waiting_queue, []) == []

    executor._pop_from_waiting_queue.assert_called_once()
    assert not executor.waiting_queue
    assert executor.canceled_req_ids == []
    assert request_id not in executor.executor_request_queue.start_times
    (response,) = _results(executor, request_id)
    assert response.client_id == 101
    assert response.result.is_final
    assert response.result.finish_reasons == [FinishReason.CANCELLED]
    assert response.result.output_token_ids == [[]]
    assert response.result.context_phase_params is None
    assert executor.num_fetch_requests == 0
    _assert_no_resource_cleanup(executor)


@pytest.mark.parametrize("queue_type", [FCFSWaitingQueue, PriorityWaitingQueue])
def test_duplicate_cancel_preserves_other_waiters_and_cancel_ids(
    executor: PyExecutor, queue_type: type[WaitingQueue]
) -> None:
    executor.waiting_queue = queue_type()
    entries = [
        RequestQueueItem(1, _request(priority=0.1)),
        RequestQueueItem(2, _request(priority=0.9)),
        RequestQueueItem(3, _request(priority=0.5)),
    ]
    executor.waiting_queue.add_requests(entries)
    surviving_order = [item.id for item in executor.waiting_queue if item.id != 2]
    executor.canceled_req_ids = [2, 2, 999]

    executor._cancel_queued_context_requests(executor.waiting_queue)
    executor.canceled_req_ids.append(2)
    executor._cancel_queued_context_requests(executor.waiting_queue)

    assert [item.id for item in executor.waiting_queue] == surviving_order
    assert executor.canceled_req_ids == [999, 2]
    assert len(_results(executor, 2)) == 1
    assert len(executor._response_channel.responses[2]) == 1
    _assert_no_resource_cleanup(executor)


@pytest.mark.parametrize(
    "sequences,beams,streaming", [(1, 1, False), (1, 1, True), (3, 1, False), (1, 2, False)]
)
def test_cancel_preserves_native_sequence_and_client_identity(
    executor: PyExecutor, sequences: int, beams: int, streaming: bool
) -> None:
    request_id = executor.executor_request_queue.enqueue_request(
        _request(sequences=sequences, beams=beams, streaming=streaming)
    )
    executor.executor_request_queue.enqueue_cancel_request(request_id)
    assert executor._fetch_new_requests(executor.waiting_queue, []) == []

    responses = _results(executor, request_id)
    assert len(responses) == sequences
    assert [response.result.sequence_index for response in responses] == list(range(sequences))
    expected_final = [False] * (sequences - 1) + [True]
    assert [response.result.is_final for response in responses] == expected_final
    for response in responses:
        assert response.request_id == request_id
        assert response.client_id == 101
        assert response.result.finish_reasons == [FinishReason.CANCELLED] * beams
        assert response.result.output_token_ids == [[] for _ in range(beams)]
        assert response.result.context_phase_params is None
    assert executor.executor_request_queue.start_times == {}
    _assert_no_resource_cleanup(executor)


def test_cancel_delivers_all_sequences_before_removing_result_route(executor: PyExecutor) -> None:
    """Drop the raw request's Ray route only after all final responses are sent."""
    route = Mock(spec=["put_response"])
    request_id = executor.enqueue_request(_request(sequences=3), result_wait_queue=route)
    delivered = []

    def record_delivery(client_id: int, response: LlmResponse) -> None:
        """Observe the registered route while each sequence response is enqueued."""
        assert executor._response_channel.result_wait_queues[request_id] is route
        assert client_id == 101
        delivered.append(response)

    route.put_response.remote.side_effect = record_delivery
    executor.executor_request_queue.enqueue_cancel_request(request_id)
    assert executor._fetch_new_requests(executor.waiting_queue, []) == []

    assert len(delivered) == 3
    assert [response.result.is_final for response in delivered] == [False, False, True]
    assert request_id not in executor._response_channel.result_wait_queues
    _assert_no_resource_cleanup(executor)


def test_cancel_uses_request_id_when_client_id_is_absent(executor: PyExecutor) -> None:
    executor.waiting_queue.add_request(RequestQueueItem(7, _request(client_id=None)))
    executor.canceled_req_ids = [7]
    executor._cancel_queued_context_requests(executor.waiting_queue)
    (response,) = _results(executor, 7)
    assert response.client_id == response.request_id == 7


def test_activation_winner_stays_on_existing_cancel_path(executor: PyExecutor) -> None:
    entry = RequestQueueItem(1, _request())
    executor.waiting_queue.add_request(entry)
    activated = executor.waiting_queue.pop_request()
    active = executor_request_to_llm_request(
        activated.id, activated.request, child_req_ids=[], exclude_last_generation_logits=False
    )
    executor.active_requests = [active]
    executor.canceled_req_ids = [1]
    executor._cancel_queued_context_requests(executor.waiting_queue)
    assert executor.active_requests == [active]
    assert not active.is_finished
    assert executor.canceled_req_ids == [1]
    assert executor._pending_transfer_responses == []
    executor._terminate_request.assert_not_called()


def test_completed_result_is_not_replaced_when_queue_entry_is_absent(executor: PyExecutor) -> None:
    original = LlmResponse(request_id=1, client_id=101, error_msg="prior failure")
    executor._pending_transfer_responses = [(1, original)]
    executor.canceled_req_ids = [1]
    executor._cancel_queued_context_requests(executor.waiting_queue)
    assert executor._pending_transfer_responses == [(1, original)]
    assert executor.canceled_req_ids == [1]
    _assert_no_resource_cleanup(executor)


def test_queued_result_does_not_flush_unrelated_retirement(executor: PyExecutor) -> None:
    original = LlmResponse(request_id=2, client_id=202, error_msg="earlier transfer failure")
    retained = object()
    executor._pending_transfer_responses = [(2, original)]
    executor._pending_response_terminations = [retained]
    executor.waiting_queue.add_request(RequestQueueItem(1, _request()))
    executor.canceled_req_ids = [1]

    executor._cancel_queued_context_requests(executor.waiting_queue)

    assert len(_results(executor, 1)) == 1
    assert 2 not in executor._response_channel.responses
    assert executor._pending_transfer_responses == [(2, original)]
    assert executor._pending_response_terminations == [retained]
    executor._terminate_request.assert_not_called()


def test_non_context_requests_and_connector_path_are_unchanged(executor: PyExecutor) -> None:
    entries = [
        RequestQueueItem(1, _request()),
        RequestQueueItem(2, _request(request_type=RequestType.REQUEST_TYPE_GENERATION_ONLY)),
        RequestQueueItem(3, _request(request_type=RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION)),
    ]
    executor.waiting_queue.add_requests(entries)
    executor.canceled_req_ids = [1, 2, 3]
    executor.kv_connector_manager = object()
    executor._cancel_queued_context_requests(executor.waiting_queue)
    assert list(executor.waiting_queue) == entries
    assert executor.canceled_req_ids == [1, 2, 3]
    assert executor._pending_transfer_responses == []

    executor.kv_connector_manager = None
    executor._cancel_queued_context_requests(executor.waiting_queue)
    assert list(executor.waiting_queue) == entries[1:]
    assert executor.canceled_req_ids == [2, 3]
    assert len(_results(executor, 1)) == 1
    _assert_no_resource_cleanup(executor)


@pytest.mark.parametrize(
    "adp,gather_all,rank,produces_response",
    [
        (False, False, 0, True),
        (False, False, 1, False),
        (False, True, 1, True),
        (True, False, 0, True),
        (True, False, 1, False),
        (True, True, 0, True),
        (True, True, 1, False),
    ],
)
def test_replicated_queue_cancellation_preserves_response_collective(
    executor: PyExecutor, adp: bool, gather_all: bool, rank: int, produces_response: bool
) -> None:
    executor.enable_attention_dp = adp
    executor.gather_all_responses = gather_all
    executor.dist.rank = executor.dist.tp_rank = rank
    executor.dist.tp_size = executor.dist.world_size = 2
    executor.dist.mapping.tp_group = [0, 1]
    executor.dist.tp_gather = Mock(side_effect=lambda responses: [responses, []])
    executor.dist.allgather = Mock(side_effect=lambda responses: [responses, []])
    executor._enqueue_responses = Mock(wraps=executor._enqueue_responses)
    executor.waiting_queue.add_request(RequestQueueItem(1, _request()))
    executor.canceled_req_ids = [1]

    executor._cancel_queued_context_requests(executor.waiting_queue)

    assert not executor.waiting_queue
    assert executor.canceled_req_ids == []
    assert executor._pending_transfer_responses == []
    executor._enqueue_responses.assert_called_once()
    assert len(executor._enqueue_responses.call_args.args[0]) == int(produces_response)
    if adp:
        collective = executor.dist.allgather if gather_all else executor.dist.tp_gather
        collective.assert_called_once()
        assert len(collective.call_args.args[0]) == int(produces_response)
    else:
        executor.dist.tp_gather.assert_not_called()
        executor.dist.allgather.assert_not_called()
    _assert_no_resource_cleanup(executor)
