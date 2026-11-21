# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Response publication and allocation lifetime across independent transfer owners."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest

from tensorrt_llm._torch.disaggregation.orchestration.coordinator import NoopDisaggCoordinator
from tensorrt_llm._torch.disaggregation.orchestration.pp_termination import (
    DisaggPPTerminationHandler,
)
from tensorrt_llm._torch.pyexecutor.disagg_adapter import (
    PyExecutorEffects,
    PyExecutorRequestRegistry,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, executor_request_to_llm_request
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_delivery import ResponseChannel
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings import executor as trtllm
from tensorrt_llm.bindings.internal.batch_manager import ReqIdsSet

pytestmark = pytest.mark.cpu_only


def _executor() -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.enable_attention_dp = False
    executor._is_kv_manager_v2 = True
    executor.kv_cache_manager = Mock()
    executor.kv_cache_manager.is_preemption_pending.return_value = False
    executor.kv_cache_transceiver = None
    executor.kv_connector_manager = None
    executor.async_transfer_manager = None
    executor._disagg_coordinator = NoopDisaggCoordinator()
    executor._disagg_pp_termination_handler = None
    executor._pending_recompute_pause_ids = set()
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._prefetched_request_ids = set()
    executor.inflight_req_ids = ReqIdsSet()
    executor.sampler = Mock()
    executor.resource_manager = Mock()
    executor.model_engine = None
    executor.draft_model_engine = None
    executor._release_multimodal_resources = Mock()
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0)
    executor._response_channel = ResponseChannel()
    executor.active_requests = []
    executor.canceled_req_ids = []
    executor._enqueue_responses = Mock()
    executor.scheduler = Mock()
    executor.scheduler.is_request_in_schedulable_state.return_value = True
    return executor


def _request() -> LlmRequest:
    return executor_request_to_llm_request(
        1, trtllm.Request(input_token_ids=[1, 2], max_tokens=8), [], False
    )


@pytest.mark.parametrize("attention_dp", [False, True])
@pytest.mark.parametrize("first_owner", ["connector", "transceiver"])
def test_both_transfer_owners_finish_before_publication(
    first_owner: str, attention_dp: bool
) -> None:
    """The second completion must not remove routing for the first staged response."""
    executor = _executor()
    executor.enable_attention_dp = attention_dp
    harness = CoordinatorHarness()
    executor.active_requests = harness.active
    executor.async_transfer_manager = harness.transfers
    executor.kv_cache_transceiver = harness.transceiver
    executor._disagg_coordinator = harness.coordinator
    harness.coordinator._effects = PyExecutorEffects(executor)
    harness.coordinator._registry = PyExecutorRequestRegistry(executor)
    response = SimpleNamespace(result=SimpleNamespace())
    request = TransferRequest(1, is_dummy_request=False)
    request.create_response = Mock(side_effect=[response, None])
    executor.active_requests.append(request)
    executor._response_channel.result_wait_queues[1] = object()
    harness.send(request)
    harness.transfers.start_transfer(request)  # independent connector claim
    delivered = []

    def publish(responses) -> None:
        assert 1 in executor._response_channel.result_wait_queues
        executor.resource_manager.free_resources.assert_not_called()
        delivered.extend(responses)

    executor._enqueue_responses.side_effect = publish

    def finish(owner) -> None:
        if owner == "transceiver":
            harness.transceiver.finish_send(request)
            harness.coordinator.reap_context_sends(0)
        else:
            executor._release_transfer(request)

    finish(first_owner)
    assert harness.in_transfer(request)
    executor.resource_manager.free_resources.assert_not_called()
    finish("transceiver" if first_owner == "connector" else "connector")
    assert not harness.in_transfer(request)
    assert 1 in executor._response_channel.result_wait_queues
    executor.resource_manager.free_resources.assert_not_called()

    executor._flush_pending_transfer_responses()
    executor._terminate_request(request)  # repeated terminal callback

    assert delivered == [(1, response)]
    assert 1 not in executor._response_channel.result_wait_queues
    executor.sampler.release_request.assert_called_once_with(request)
    executor.resource_manager.free_resources.assert_called_once_with(request)
    executor._release_multimodal_resources.assert_called_once_with(request)
    harness.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(1)


@pytest.mark.parametrize("pp_consensus", [False, True])
def test_terminal_release_is_idempotent(pp_consensus: bool) -> None:
    executor = _executor()
    request = _request()
    executor._response_channel.result_wait_queues[request.py_request_id] = object()
    if pp_consensus:
        executor._disagg_pp_termination_handler = Mock()
    executor._terminate_request(request)
    executor._terminate_request(request)
    if pp_consensus:
        executor.resource_manager.free_resources.assert_not_called()
        assert request.py_request_id in executor._response_channel.result_wait_queues
        executor._on_disagg_pp_termination(request)
        executor._on_disagg_pp_termination(request)
    executor._terminate_request(request)
    executor.resource_manager.free_resources.assert_called_once_with(request)
    executor.sampler.release_request.assert_called_once_with(request)
    executor._release_multimodal_resources.assert_called_once_with(request)
    assert request.py_request_terminated
    assert not executor._response_channel.result_wait_queues


@pytest.mark.parametrize("cancel_before_reset", [False, True])
def test_recompute_frees_each_allocation_once(cancel_before_reset: bool) -> None:
    executor = _executor()
    request = _request()
    executor.active_requests.append(request)
    executor._response_channel.result_wait_queues[request.py_request_id] = object()
    batch = ScheduledRequests()
    batch.recompute_paused_requests = [request]
    executor._terminate_recompute_paused_requests(batch)
    if cancel_before_reset:
        request.state = LlmRequestState.GENERATION_COMPLETE
    executor._pause_recompute_paused_requests(batch)
    assert not request.py_request_terminated
    assert request.py_request_id in executor._response_channel.result_wait_queues
    executor._terminate_request(request)
    expected = 1 if cancel_before_reset else 2
    assert executor.resource_manager.free_resources.call_count == expected
    assert executor.sampler.release_request.call_count == expected
    executor._terminate_request(request)
    assert executor.resource_manager.free_resources.call_count == expected


def test_legacy_pause_starts_a_fresh_allocation_lifetime() -> None:
    executor = _executor()
    executor.max_input_len = 128
    request = _request()
    executor._terminate_request(request)
    executor._pause_requests([request])
    assert not request.py_request_terminated
    assert not request.py_request_resources_released
    executor._terminate_request(request)
    assert executor.resource_manager.free_resources.call_count == 2


@pytest.mark.parametrize("parked", [False, True])
@pytest.mark.parametrize("in_transfer", [False, True])
@pytest.mark.parametrize("loading", [False, True])
def test_final_response_and_safe_release_have_independent_readiness(
    parked: bool, in_transfer: bool, loading: bool
) -> None:
    executor = _executor()
    request = _request()
    request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    executor.kv_cache_manager.is_preemption_pending.return_value = parked
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = (
        {request.py_request_id: request} if in_transfer else {}
    )
    executor.kv_connector_manager = Mock()
    executor.kv_connector_manager.has_pending_load.return_value = loading
    assert executor._is_final_response_ready(request) is (not parked)
    assert executor._can_release_request(request) is (not (parked or in_transfer or loading))


@pytest.mark.parametrize("parked", [False, True])
@pytest.mark.parametrize("recomputing", [False, True])
def test_parked_and_recomputing_requests_are_not_schedulable(
    parked: bool, recomputing: bool
) -> None:
    executor = _executor()
    request = _request()
    executor.kv_cache_manager.is_preemption_pending.return_value = parked
    if recomputing:
        executor._pending_recompute_pause_ids.add(request.py_request_id)
    assert executor._is_request_schedulable(request) is (not (parked or recomputing))


@pytest.mark.parametrize("owner", ["response", "load", "save"])
def test_pp_consensus_does_not_override_a_remaining_owner(owner: str) -> None:
    executor = _executor()
    request = _request()
    executor._disagg_pp_termination_handler = Mock()
    executor._response_channel.result_wait_queues[request.py_request_id] = object()
    response = SimpleNamespace(result=SimpleNamespace())
    if owner == "response":
        executor._pending_transfer_responses = [(request.py_request_id, response)]
    elif owner == "load":
        executor.kv_connector_manager = Mock()
        executor.kv_connector_manager.has_pending_load.return_value = True
        executor.kv_connector_manager.defer_load_termination.return_value = True
    else:
        executor.async_transfer_manager = Mock()
        executor.async_transfer_manager.requests_in_transfer.return_value = {
            request.py_request_id: request
        }

    executor._on_disagg_pp_termination(request)
    executor.resource_manager.free_resources.assert_not_called()
    assert request.py_request_id in executor._response_channel.result_wait_queues
    if owner == "response":
        executor._flush_pending_transfer_responses()
        executor._enqueue_responses.assert_called_once_with([(request.py_request_id, response)])
    else:
        if owner == "load":
            executor.kv_connector_manager.has_pending_load.return_value = False
            executor.kv_connector_manager.defer_load_termination.return_value = False
        else:
            executor.async_transfer_manager.requests_in_transfer.return_value = {}
        executor._terminate_request(request)
    executor._disagg_pp_termination_handler.terminate.assert_not_called()
    executor.resource_manager.free_resources.assert_called_once_with(request)
    assert request.py_request_pp_termination_agreed


@pytest.mark.parametrize("early_release", [False, True])
def test_last_owner_retries_teardown_deferred_by_the_early_path(early_release: bool) -> None:
    executor = _executor()
    harness = CoordinatorHarness()
    executor._is_kv_manager_v2 = not early_release
    executor.force_terminate_ctx_for_partial_reuse = early_release
    harness.transfers.should_store_blocks = early_release
    executor.async_transfer_manager = harness.transfers
    executor.kv_cache_transceiver = harness.transceiver
    executor._disagg_coordinator = harness.coordinator
    harness.coordinator._effects = PyExecutorEffects(executor)
    request = TransferRequest(1, is_dummy_request=False)
    harness.send(request)
    executor._terminate_request(request)
    executor.resource_manager.free_resources.assert_not_called()
    harness.transceiver.finish_send(request)
    harness.coordinator.reap_context_sends(0)
    executor.resource_manager.free_resources.assert_called_once_with(request)


def test_pp_ring_releases_a_delayed_owner_without_another_consensus() -> None:
    mailboxes = [[], []]
    sends = []

    def dist(rank: int) -> SimpleNamespace:
        def send(payload: dict, dest: int, tag: object) -> Mock:
            sends.append((rank, dest, tag))
            mailboxes[dest].append(deepcopy(payload))
            return Mock()

        return SimpleNamespace(
            rank=rank,
            pp_rank=rank,
            pp_size=2,
            is_first_pp_rank=rank == 0,
            is_last_pp_rank=rank == 1,
            prev_pp_rank=1 - rank,
            next_pp_rank=1 - rank,
            isend_object=send,
            recv_object=lambda src, tag: mailboxes[rank].pop(0),
        )

    executors = [_executor(), _executor()]
    requests = [_request(), _request()]
    for rank, (executor, request) in enumerate(zip(executors, requests)):
        executor.dist = dist(rank)
        executor._disagg_pp_termination_handler = DisaggPPTerminationHandler(
            executor.dist, executor._on_disagg_pp_termination
        )
        executor._response_channel.result_wait_queues[request.py_request_id] = object()
        executor._terminate_request(request)
        executor._disagg_pp_termination_handler.terminate_pending_requests()

    # Both ranks voted, but rank 0 must still publish its response before freeing.
    response = SimpleNamespace(result=SimpleNamespace())
    executors[0]._pending_transfer_responses = [(requests[0].py_request_id, response)]
    for executor in executors:
        executor._disagg_pp_termination_handler.terminate_pending_requests()
    executors[0].resource_manager.free_resources.assert_not_called()
    executors[1].resource_manager.free_resources.assert_called_once_with(requests[1])
    executors[0]._flush_pending_transfer_responses()

    assert len(sends) == 4
    for executor, request in zip(executors, requests):
        executor.resource_manager.free_resources.assert_called_once_with(request)
        assert executor._disagg_pp_termination_handler._pending_termination == {}
        assert request.py_request_pp_termination_agreed
