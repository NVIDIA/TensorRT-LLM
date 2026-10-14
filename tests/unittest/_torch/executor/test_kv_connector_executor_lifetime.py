# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor ownership of connector loads through cancellation and completion."""

from contextlib import nullcontext
from copy import deepcopy
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorManager
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    CONTROL_REQUEST_ID,
    PREFIX_LOAD_COMPLETION_REQUEST_ID,
    SHUTDOWN_REQUEST_ID,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import RequestBroadcaster
from tensorrt_llm.bindings.executor import FinishReason

pytestmark = pytest.mark.cpu_only


def _executor() -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = Mock(spec=KvCacheConnectorManager)
    # Set explicitly: `capacity_only` is not on the spec, so a double that
    # leaves it unset fails loudly rather than silently lifting the executor
    # off the connector's per-request path.
    executor.kv_connector_manager.capacity_only = False
    executor.kv_connector_manager.prefix_reservations_enabled = True
    executor.kv_connector_manager.defer_load_termination.return_value = False
    executor.kv_connector_manager.has_outstanding_save.return_value = False
    executor.kv_connector_manager.has_pending_loads.return_value = False
    executor.kv_connector_manager.get_finished.return_value = []
    executor.kv_connector_manager.take_finished_load_terminations.return_value = []
    executor.kv_cache_manager = Mock()
    # A bare Mock answers both of these truthily, which would route every
    # finished request through the deferred-preemption path.
    executor.kv_cache_manager.try_complete_preemption.return_value = False
    executor.kv_cache_manager.is_preemption_pending.return_value = False
    executor._is_kv_manager_v2 = True
    executor.kv_cache_transceiver = None
    executor.active_requests = []
    executor.canceled_req_ids = []
    executor.waiting_queue = Mock()
    executor.dist = SimpleNamespace(rank=0)
    executor._disagg_pp_termination_handler = None
    executor._do_terminate_request = Mock()
    executor._enqueue_responses = Mock()
    executor._release_transfer = Mock()
    return executor


def _request(request_id: int = 1) -> SimpleNamespace:
    response = SimpleNamespace(result=SimpleNamespace(cached_tokens=0))
    request = SimpleNamespace(
        request_id=request_id,
        py_request_id=request_id,
        is_child=False,
        is_dummy_request=False,
        is_finished=False,
        py_decoding_iter=0,
        cached_tokens=32,
        create_response=Mock(return_value=response),
    )

    def finish(reason: FinishReason) -> None:
        assert reason == FinishReason.CANCELLED
        request.is_finished = True

    request.finish_by_reason = Mock(side_effect=finish)
    return request


def test_cancel_waits_for_connector_without_a_transceiver() -> None:
    executor = _executor()
    request = _request()
    executor.kv_connector_manager.defer_load_termination.return_value = True

    assert not executor._try_cancel_request(request)
    executor.kv_connector_manager.defer_load_termination.assert_called_once_with(request)
    assert not request.is_finished
    executor._do_terminate_request.assert_not_called()


def test_teardown_cannot_free_an_outstanding_load() -> None:
    executor = _executor()
    request = _request()
    executor.kv_connector_manager.defer_load_termination.return_value = True

    executor._terminate_request(request)

    executor.kv_connector_manager.release_unstarted_prefix_loads.assert_called_once_with(request)
    executor.kv_connector_manager.defer_load_termination.assert_called_once_with(request)
    executor._do_terminate_request.assert_not_called()


def test_unstarted_load_can_be_released_before_teardown() -> None:
    executor = _executor()
    request = _request()
    calls = []
    executor.kv_connector_manager.release_unstarted_prefix_loads.side_effect = (
        lambda req: calls.append("release")
    )
    executor._do_terminate_request.side_effect = lambda req: calls.append("free")

    executor._terminate_request(request)

    assert calls == ["release", "free"]


def test_lone_loading_request_consumes_cancellation_before_completion() -> None:
    executor = _executor()
    request = _request()
    executor.active_requests = [request]
    executor.canceled_req_ids = [request.py_request_id]
    connector = executor.kv_connector_manager
    calls = []
    connector.defer_load_termination.side_effect = lambda req: calls.append("defer") or False

    def complete() -> list:
        assert calls == ["defer"]
        calls.append("complete")
        return []

    connector.get_finished.side_effect = complete
    connector.take_finished_load_terminations.return_value = [request]

    executor._kv_connector_terminate_requests()

    request.finish_by_reason.assert_called_once_with(FinishReason.CANCELLED)
    assert executor.active_requests == []
    assert executor.canceled_req_ids == []
    response = request.create_response.return_value
    assert response.result.cached_tokens == 32
    executor._enqueue_responses.assert_called_once_with([(request.py_request_id, response)])
    executor._do_terminate_request.assert_called_once_with(request)
    executor._release_transfer.assert_not_called()


def test_load_error_already_reported_does_not_emit_another_response() -> None:
    executor = _executor()
    request = _request()
    request.is_finished = True
    executor.kv_connector_manager.take_finished_load_terminations.return_value = [request]

    executor._kv_connector_terminate_requests()

    request.create_response.assert_not_called()
    executor._enqueue_responses.assert_not_called()
    executor._do_terminate_request.assert_called_once_with(request)


def test_allocation_survives_until_the_completion_poll() -> None:
    executor = _executor()
    request = _request()
    executor.active_requests = [request]
    executor.canceled_req_ids = [request.py_request_id]
    executor.kv_connector_manager.defer_load_termination.return_value = True

    executor._kv_connector_terminate_requests()

    assert executor.active_requests == [request]
    assert executor.canceled_req_ids == [request.py_request_id]
    request.finish_by_reason.assert_not_called()
    executor._do_terminate_request.assert_not_called()


def test_load_dispatch_marks_ownership_before_worker_launch(monkeypatch) -> None:
    executor = _executor()
    connector = executor.kv_connector_manager
    calls = []
    connector.take_scheduled_requests_pending_load.side_effect = lambda batch: calls.append("park")
    connector.handle_metadata.side_effect = lambda: calls.append("metadata")
    connector.mark_prefix_loads_dispatched.side_effect = lambda: calls.append("own")
    connector.worker = Mock()
    connector.worker.start_load_kv.side_effect = lambda stream: calls.append("start")
    monkeypatch.setattr("torch.cuda.current_stream", lambda: None)

    executor._kv_connector_start_batch(SimpleNamespace())

    assert calls == ["park", "metadata", "own", "start"]


@pytest.mark.parametrize("selected", [True, False])
def test_final_batch_releases_every_unselected_reservation(selected: bool) -> None:
    executor = _executor()
    batch = SimpleNamespace(context_requests=[_request(7)]) if selected else None

    executor._release_unused_connector_reservations(batch)

    executor.kv_cache_manager.release_unused_connector_reservations.assert_called_once_with(
        {7} if selected else set()
    )


def test_shutdown_waits_for_an_outstanding_load_after_request_error() -> None:
    executor = _executor()
    executor.is_shutdown = True
    executor.waiting_queue = []
    executor.kv_connector_manager.has_pending_loads.return_value = True

    assert not executor.should_stop_processing

    executor.kv_connector_manager.has_pending_loads.return_value = False
    assert executor.should_stop_processing


def test_completing_one_child_preserves_cancellation_for_its_sibling() -> None:
    executor = _executor()
    loading = _request(1)
    waiting = _request(2)
    for request in (loading, waiting):
        request.is_child = True
        request.parent_request_id = 9
    executor.active_requests = [loading, waiting]
    executor.canceled_req_ids = [9]

    executor._finish_connector_load_termination(loading)

    assert executor.active_requests == [waiting]
    assert executor.canceled_req_ids == [9]
    executor._do_terminate_request.assert_called_once_with(loading)


def test_completed_load_does_not_release_a_pending_save() -> None:
    executor = _executor()
    request = _request()
    request.is_finished = True
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }

    executor._finish_connector_load_termination(request)

    executor._do_terminate_request.assert_not_called()
    executor._enqueue_responses.assert_not_called()

    executor.async_transfer_manager.requests_in_transfer.return_value = {}
    executor._terminate_request(request)
    executor._do_terminate_request.assert_called_once_with(request)


@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True])
def test_polling_continues_for_save_after_load_completion(use_kv_cache_manager_v2: bool) -> None:
    executor = _executor()
    executor._is_kv_manager_v2 = use_kv_cache_manager_v2
    connector = executor.kv_connector_manager
    connector.prefix_reservations_enabled = use_kv_cache_manager_v2
    connector.take_finished_prefix_loads.return_value = []
    executor.is_shutdown = False
    executor.control_requests = []
    executor._disable_mpi = False
    executor.request_accumulated = []
    executor.hang_detector = SimpleNamespace(pause=nullcontext)
    executor.dist.world_size = 1
    executor.request_broadcaster = RequestBroadcaster(executor.dist, executor.hang_detector)
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_from_request_queue.return_value = []
    executor.waiting_queue = Mock()
    executor.waiting_queue.__len__ = Mock(return_value=0)
    executor.async_transfer_manager = AsyncTransferManager(
        SimpleNamespace(resource_managers={}), should_store_blocks=False
    )
    executor.force_terminate_ctx_for_partial_reuse = False
    executor._release_transfer = MethodType(PyExecutor._release_transfer, executor)
    request = _request()
    request.is_finished = True
    executor.async_transfer_manager.start_transfer(request)
    connector.get_finished.side_effect = [[], [request]]

    for _ in range(2):
        executor._fetch_and_enqueue_requests(executor.waiting_queue, total_num_live_requests=0)
        timeout = executor.executor_request_queue.get_from_request_queue.call_args.args[0]
        assert timeout.total_seconds() == 0
        executor._do_terminate_request.assert_not_called()
        executor.is_shutdown = True
        assert not executor.should_stop_processing
        executor.is_shutdown = False
        executor._kv_connector_terminate_requests()

    connector.get_finished.assert_called_with()
    assert connector.get_finished.call_count == 2
    executor._do_terminate_request.assert_called_once_with(request)
    assert not executor.async_transfer_manager.has_any_inflight_requests()

    executor._fetch_and_enqueue_requests(executor.waiting_queue, total_num_live_requests=0)
    executor.executor_request_queue.get_from_request_queue.assert_called_with(None)
    executor.is_shutdown = True
    assert executor.should_stop_processing


def test_legacy_pinned_save_keeps_its_existing_early_teardown() -> None:
    executor = _executor()
    executor._is_kv_manager_v2 = False
    request = _request()
    request.is_finished = True
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer.return_value = {
        request.py_request_id: request
    }

    executor._terminate_request(request)

    executor._do_terminate_request.assert_called_once_with(request)


def test_control_drain_waits_for_detached_connector_load() -> None:
    executor = _executor()
    executor.waiting_queue = []
    pending = SimpleNamespace(control_requires_drain=True)
    executor.control_requests = [pending]
    executor.kv_connector_manager.has_pending_loads.return_value = True

    executor._handle_control_request()

    assert executor.control_requests == [pending]


@pytest.mark.parametrize("control_pending", [False, True])
def test_completion_is_broadcast_without_new_requests(control_pending: bool) -> None:
    executor = _executor()
    connector = executor.kv_connector_manager
    connector.has_pending_loads.return_value = True
    connector.take_finished_prefix_loads.return_value = [17]
    executor.control_requests = [RequestQueueItem(CONTROL_REQUEST_ID)] if control_pending else []
    executor.is_shutdown = False
    executor._disable_mpi = False
    executor.request_accumulated = []
    executor.hang_detector = SimpleNamespace(pause=nullcontext)
    executor.dist.world_size = 1
    executor.request_broadcaster = RequestBroadcaster(executor.dist, executor.hang_detector)
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_from_request_queue.return_value = []
    waiting_queue = Mock()
    waiting_queue.__len__ = Mock(return_value=0)

    executor._fetch_and_enqueue_requests(waiting_queue, total_num_live_requests=0)

    connector.finish_prefix_loads.assert_called_once_with([17])
    waiting_queue.add_requests.assert_called_once_with([])
    if control_pending:
        executor.executor_request_queue.get_from_request_queue.assert_not_called()
    else:
        timeout = executor.executor_request_queue.get_from_request_queue.call_args.args[0]
        assert timeout.total_seconds() == 0


@pytest.mark.parametrize("stop_id", [SHUTDOWN_REQUEST_ID, CONTROL_REQUEST_ID])
def test_completion_is_applied_before_a_control_or_shutdown_boundary(stop_id: int) -> None:
    executor = _executor()
    executor.control_requests = []
    executor.request_accumulated = []
    executor.is_shutdown = False
    completion = RequestQueueItem(PREFIX_LOAD_COMPLETION_REQUEST_ID, finished_prefix_load_ids=[17])

    accepted = executor._handle_special_queue_items([completion, RequestQueueItem(stop_id)])

    assert accepted == []
    assert not completion.is_normal_request
    executor.kv_connector_manager.finish_prefix_loads.assert_called_once_with([17])


def test_cancellation_in_completion_envelope_is_seen_before_resumption() -> None:
    executor = _executor()
    req = _request()
    executor.active_requests = [req]
    connector = executor.kv_connector_manager
    order = []
    connector.defer_load_termination.side_effect = lambda request: order.append("cancel")
    connector.finish_prefix_loads.side_effect = lambda ids: order.append("finish")

    executor._handle_special_queue_items(
        [
            RequestQueueItem(PREFIX_LOAD_COMPLETION_REQUEST_ID, finished_prefix_load_ids=[17]),
            RequestQueueItem(req.py_request_id, is_canceled_request=True),
        ]
    )

    assert order == ["cancel", "finish"]


def test_empty_iteration_keeps_request_broadcast_fast_path() -> None:
    dist = SimpleNamespace(rank=0, world_size=1)
    broadcaster = RequestBroadcaster(dist, SimpleNamespace(pause=nullcontext))
    broadcaster._broadcast_requests = Mock(side_effect=AssertionError("Unexpected payload"))

    assert broadcaster.broadcast([]) == ([], None)


def test_completion_envelope_reaches_every_rank_without_a_compute_batch() -> None:
    wire = {}
    executors = []
    for rank in range(2):
        executor = _executor()
        executor.control_requests = []
        executor.is_shutdown = False
        executor._disable_mpi = False
        executor.request_accumulated = []
        executor.hang_detector = SimpleNamespace(pause=nullcontext)

        def broadcast(value, root, *, rank=rank):
            assert root == 0
            if rank == 0:
                wire["payload"] = deepcopy(value)
            return deepcopy(wire["payload"])

        def broadcast_count(value, root, *, rank=rank):
            assert root == 0
            if rank == 0:
                wire["count"] = value
            return wire["count"]

        executor.dist = SimpleNamespace(
            rank=rank,
            world_size=2,
            tp_size=2,
            cp_size=1,
            has_pp=False,
            broadcast=broadcast,
            broadcast_int64=broadcast_count,
        )
        executor.request_broadcaster = RequestBroadcaster(executor.dist, executor.hang_detector)
        executor.executor_request_queue = Mock()
        executor.executor_request_queue.get_from_request_queue.return_value = []
        executor.kv_connector_manager.has_pending_loads.return_value = True
        executor.kv_connector_manager.take_finished_prefix_loads.return_value = [17]
        executors.append(executor)

    for executor in executors:
        waiting_queue = Mock()
        waiting_queue.__len__ = Mock(return_value=0)
        executor._fetch_and_enqueue_requests(waiting_queue, total_num_live_requests=0)
        executor.kv_connector_manager.finish_prefix_loads.assert_called_once_with([17])
        waiting_queue.add_requests.assert_called_once_with([])
    executors[1].kv_connector_manager.take_finished_prefix_loads.assert_not_called()


# ---- deferred preemption ----


class _PreemptionBookkeeping:
    """The victim ledger `KVCacheManagerV2.preempt_request` keeps.

    Borrows the manager's own methods so the deferral contract under test is
    the shipped one, and supplies only what they reach for. Constructing a real
    manager needs a GPU; `test_kvcm2_integration.py` covers it there.
    """

    preempt_request = KVCacheManagerV2.preempt_request
    try_complete_preemption = KVCacheManagerV2.try_complete_preemption
    is_preemption_pending = KVCacheManagerV2.is_preemption_pending
    has_pending_preemption = KVCacheManagerV2.has_pending_preemption
    _release_preempted = KVCacheManagerV2._release_preempted

    def __init__(self, connector_manager) -> None:
        self.kv_connector_manager = connector_manager
        self.is_draft = False
        self._pending_preemption = {}
        self.released = []

    def get_page_indices_by_layer_group(self, req):
        return [[1, 2, 3]]

    def get_cache_indices(self, req):
        # This double is not a `KVCacheManagerV2` instance, so the connector
        # handshake reads the flat list rather than the grouped one above.
        return [1, 2, 3]

    def free_resources(self, req):
        self.released.append(req.py_request_id)


def _preemption_executor() -> PyExecutor:
    """An executor with the pieces `_handle_responses` reads."""
    executor = _executor()
    executor.kv_cache_manager = _PreemptionBookkeeping(executor.kv_connector_manager)
    executor.perf_manager = Mock()
    executor._disagg_coordinator = Mock()
    executor.model_engine = None
    executor.iter_counter = 1
    executor._free_request_resources = Mock()
    executor._terminate_request = Mock()
    executor.force_terminate_ctx_for_partial_reuse = False
    # Wide enough that a mid-stream request emits nothing of its own, so any
    # response seen below came from the finish path rather than from streaming.
    executor.stream_interval = 100
    return executor


def _preemption_victim(request_id: int = 1) -> SimpleNamespace:
    """A generation request the scheduler is about to preempt."""
    request = _request(request_id)
    request.py_decoding_iter = 5
    request.py_draft_tokens = []
    request.py_per_pos_drafted = None
    request.py_per_pos_accepted = None
    request.py_total_draft_tokens = 0
    request.py_kv_transfer_timed_out = False
    request.is_attention_dp_dummy = False
    request.is_generation_only_request = False
    request.is_disagg_context_complete_state = False
    request.is_disagg_context_transmission_state = True
    request.return_perf_metrics = False
    request.py_num_connector_matched_tokens = 8
    request.py_connector_prefix_start = 0
    request.py_connector_prefix_end = 8
    request.py_connector_load_async = False
    request.py_connector_delivered = True

    def reset_for_recompute(_max_input_len):
        # The real teardown puts the request back in context state, which is
        # what makes it schedulable again.
        request.is_finished = False
        request.is_disagg_context_transmission_state = False

    request.reset_for_recompute = Mock(side_effect=reset_for_recompute)
    return request


def test_deferred_preemption_is_not_reported_to_the_client_as_finished() -> None:
    """A parked victim must survive the response pass that runs before its saves retire.

    `preempt_request` hands the victim to the connector the way a finished
    request is handed over, which leaves it in the state `LlmRequest.isFinished`
    reports as finished. `_handle_responses` runs before `get_finished` in the
    same iteration, so reading that state as "done generating" would send the
    client a final response for a request that is about to re-prefill and drop
    it from `active_requests`, where the resume path would find nothing to put
    back.
    """
    executor = _preemption_executor()
    victim = _preemption_victim()
    executor.active_requests = [victim]
    connector = executor.kv_connector_manager
    connector.request_finished.return_value = True

    assert executor.kv_cache_manager.preempt_request(victim) is False
    assert executor.kv_cache_manager.has_pending_preemption()
    # The state the connector handshake leaves behind is exactly the one that
    # makes this bug reachable, so assert the test is standing in it.
    victim.is_finished = True
    assert executor._is_preemption_pending(victim)

    executor._handle_responses()

    assert executor.active_requests == [victim]
    victim.create_response.assert_not_called()
    executor._enqueue_responses.assert_called_once_with([])
    executor._terminate_request.assert_not_called()


def test_a_capacity_only_connector_defers_no_preemption() -> None:
    """It has no save to wait for, so the victim's pages go back now.

    The deferral costs a page index gather per victim and parks the request in
    the state the finish path uses, which every response pass then has to ask
    about.
    """
    executor = _preemption_executor()
    victim = _preemption_victim()
    connector = executor.kv_connector_manager
    connector.capacity_only = True
    connector.request_finished.return_value = True

    assert executor.kv_cache_manager.preempt_request(victim) is True

    assert executor.kv_cache_manager.released == [victim.py_request_id]
    assert not executor.kv_cache_manager.has_pending_preemption()
    connector.request_finished.assert_not_called()


def test_a_capacity_only_connector_is_polled_and_waited_on_by_nothing() -> None:
    """A role that moves no KV has nothing per-iteration to drive.

    `get_finished` agrees across ranks through an allgather, and the
    cancellation sweep that precedes it walks every active request.
    """
    executor = _preemption_executor()
    connector = executor.kv_connector_manager
    connector.capacity_only = True
    # Not on the spec either, for the same reason `capacity_only` is not.
    connector.worker = Mock()
    executor.active_requests = [_request(3)]

    executor._kv_connector_terminate_requests()
    PyExecutor._kv_connector_wait_for_save(executor)

    connector.get_finished.assert_not_called()
    connector.defer_load_termination.assert_not_called()
    connector.worker.wait_for_save.assert_not_called()


def test_a_victim_whose_saves_retired_goes_back_to_being_schedulable() -> None:
    """The round trip: deferral, a response pass, then completion."""
    executor = _preemption_executor()
    victim = _preemption_victim()
    executor.active_requests = [victim]
    connector = executor.kv_connector_manager
    connector.request_finished.return_value = True

    assert executor.kv_cache_manager.preempt_request(victim) is False
    victim.is_finished = True
    executor._handle_responses()

    connector.get_finished.return_value = [victim]
    executor._kv_connector_terminate_requests()

    # Preempted, not finished: the transfer release belongs to the other kind.
    executor._release_transfer.assert_not_called()
    assert executor.kv_cache_manager.released == [victim.py_request_id]
    assert not executor.kv_cache_manager.has_pending_preemption()
    executor._free_request_resources.assert_called_once_with(victim)
    victim.reset_for_recompute.assert_called_once()
    assert victim.py_num_connector_matched_tokens == 0

    # And the next response pass leaves it alone, so the scheduler sees it.
    connector.get_finished.return_value = []
    executor._handle_responses()
    assert executor.active_requests == [victim]
    victim.create_response.assert_not_called()


def test_a_finished_request_still_takes_the_transfer_release() -> None:
    """Only a preemption victim is resumed; everything else retires as before."""
    executor = _preemption_executor()
    request = _preemption_victim()
    request.is_finished = True
    executor.kv_connector_manager.get_finished.return_value = [request]

    executor._kv_connector_terminate_requests()

    executor._release_transfer.assert_called_once_with(request)
    executor._free_request_resources.assert_not_called()
    request.reset_for_recompute.assert_not_called()


def test_a_cancellation_waits_for_a_deferred_preemption() -> None:
    """Cancelling a parked victim now would be undone by its own completion.

    The completion that ends a deferred preemption releases the victim's pages
    and puts it back in context state to be re-prefilled, which is the one
    thing a cancelled request must not do. Held, the cancellation outlives the
    completion and is applied to the request it hands back.
    """
    executor = _preemption_executor()
    victim = _preemption_victim()
    executor.active_requests = [victim]
    executor.canceled_req_ids = [victim.py_request_id]
    connector = executor.kv_connector_manager
    connector.request_finished.return_value = True

    assert executor.kv_cache_manager.preempt_request(victim) is False
    victim.is_finished = True

    executor._handle_canceled_requests()

    # Still queued, so the next pass tries again.
    assert executor.canceled_req_ids == [victim.py_request_id]
    victim.finish_by_reason.assert_not_called()
    assert executor.kv_cache_manager.has_pending_preemption()

    # Its saves retire, and the preemption completes as it would have.
    connector.get_finished.return_value = [victim]
    executor._kv_connector_terminate_requests()
    victim.reset_for_recompute.assert_called_once()
    assert not executor.kv_cache_manager.has_pending_preemption()

    # Only now does the cancellation land, on a request that is schedulable
    # again rather than on one halfway out of the KV cache. Its pages are
    # already in the pool, so the connector has nothing left to hold.
    connector.get_finished.return_value = []
    connector.request_finished.return_value = False
    executor._handle_canceled_requests()
    victim.finish_by_reason.assert_called_once_with(FinishReason.CANCELLED)
    assert executor.canceled_req_ids == []


# ---- cancellation with a save in flight ----


def _cancelling_executor() -> PyExecutor:
    """An executor with the pieces `_handle_canceled_requests` reads."""
    executor = _executor()
    executor.async_transfer_manager = Mock()
    executor.kv_cache_manager.get_cache_indices.return_value = [1, 2, 3]
    return executor


def test_a_cancelled_request_keeps_its_pages_until_its_saves_land() -> None:
    """A request cancelled mid-prefill can have a save reading its blocks.

    The response pass frees a request's pages as soon as it reads as finished,
    so cancellation has to offer the connector the same handshake a finished
    request gets. Without it the next request allocated those pages overwrites
    bytes a transfer is still reading.
    """
    executor = _cancelling_executor()
    request = _request()
    executor.active_requests = [request]
    executor.canceled_req_ids = [request.py_request_id]
    connector = executor.kv_connector_manager
    connector.request_finished.return_value = True

    executor._handle_canceled_requests()

    request.finish_by_reason.assert_called_once_with(FinishReason.CANCELLED)
    connector.request_finished.assert_called_once_with(request, [1, 2, 3], None)
    executor.async_transfer_manager.start_transfer.assert_called_once_with(request)
    assert executor.canceled_req_ids == []


def test_a_cancelled_request_the_connector_declines_is_freed_at_once() -> None:
    """Nothing is reading its pages, so holding them would only cost capacity."""
    executor = _cancelling_executor()
    request = _request()
    executor.active_requests = [request]
    executor.canceled_req_ids = [request.py_request_id]
    executor.kv_connector_manager.request_finished.return_value = False

    executor._handle_canceled_requests()

    request.finish_by_reason.assert_called_once_with(FinishReason.CANCELLED)
    executor.async_transfer_manager.start_transfer.assert_not_called()


# ---- shutdown ----


def test_shutdown_drains_the_connector_before_the_pools_it_reads_are_freed() -> None:
    """A connector save reads the KV pools a resource manager owns.

    Freeing those pools first takes the memory out from under a transfer in
    flight, so the connector has to be given its chance to drain while they
    are still allocated.
    """
    executor = _executor()
    order = []
    executor.kv_connector_manager.shutdown.side_effect = lambda: order.append("connector")
    pools = Mock()
    pools.shutdown.side_effect = lambda: order.append("kv pools")

    executor.executor_request_queue = Mock()
    executor.shutdown_event = Mock()
    executor._profile_manager = Mock()
    executor.hang_detector = Mock()
    executor.hang_detector.detected.return_value = False
    executor.worker_thread = Mock()
    executor.dist = SimpleNamespace(rank=0, pp_size=1)
    executor._shutdown_sleep_wakeup_listeners = Mock()
    executor.worker_started = True
    executor.model_engine = None
    executor.draft_model_engine = None
    executor.resource_manager = SimpleNamespace(resource_managers={"kv_cache": pools})
    executor.virtual_memory_pools = None
    executor.sampler = Mock()
    executor.dwdp_manager = None

    executor.shutdown()

    # The worker thread first: it is the one still handing the connector work.
    executor.worker_thread.join.assert_called_once_with()
    assert order == ["connector", "kv pools"]
