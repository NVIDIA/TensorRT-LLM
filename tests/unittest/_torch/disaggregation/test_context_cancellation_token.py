# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route authenticated CTX cancellation through real executor queue boundaries.

Worker authentication/admission serialization is tested separately. These tests
keep queue dispatch and request identity real; the cancellation harness controls
transceiver acceptance and transport, not physical GPU/NIXL retirement.
"""

import datetime
import pickle
from unittest.mock import call, patch

import pytest

__extra_import_path__ = ["../pyexecutor"]

from fake_dist import FakeDistGroup, FakeDistRank
from test_context_cancellation_executor import _broadcast_rank, _executor, _prepare_fetch
from test_context_cancellation_executor import _request as _active_request
from test_queued_context_cancellation import _request as _raw_request

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    CONTROL_REQUEST_ID,
    RequestAdmissionState,
)
from tensorrt_llm._torch.pyexecutor.llm_request import ExecutorRequest
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only


def _raw_with_token(token: str) -> ExecutorRequest:
    """Create real CTX input with its independently authenticated identity.

    Args:
        token: Unique context-request token assigned by the worker.

    Returns:
        Native input carrying Python disaggregation metadata.
    """
    request = _raw_request()
    request.py_conversation_params = None
    request.py_disaggregated_params = DisaggregatedParams(
        request_type="context_only",
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        context_cancellation_token=token,
    )
    return request


def _serialized_rank(group: FakeDistGroup, rank: int) -> FakeDistRank:
    """Preserve native request serialization across the checked rank transport.

    Args:
        group: The group that checks identical collective ordering.
        rank: Rank using this transport.

    Returns:
        A checked transport that also exercises native envelope serialization.
    """
    dist = _broadcast_rank(group, rank)
    broadcast = dist.broadcast

    def transmit(value: object, root: int) -> object:
        """Broadcast independent serialized objects to each receiving rank."""
        return pickle.loads(pickle.dumps(broadcast(value, root)))

    dist.broadcast = transmit
    return dist


def _engines() -> tuple[FakeDistGroup, list[PyExecutor]]:
    """Build two executor ranks with real queue and checked transport order.

    Returns:
        The checked collective group and its two executor instances.
    """
    group = FakeDistGroup(2, 2)
    engines = [_executor(_serialized_rank(group, rank)) for rank in range(2)]
    for engine in engines:
        _prepare_fetch(engine)
    return group, engines


@pytest.mark.parametrize("already_waiting", [False, True])
def test_token_cancellation_completes_only_matching_queued_ctx(already_waiting: bool) -> None:
    """A token survives rank transport and cancels before admission, not its healthy peer."""
    group, engines = _engines()
    queue = engines[0].executor_request_queue
    cancelled_id = queue.enqueue_request(_raw_with_token("cancelled"))
    healthy_id = queue.enqueue_request(_raw_with_token("healthy"))

    def fetch(rank: int) -> None:
        """Fetch a replicated envelope without waiting for model work.

        Args:
            rank: Executor rank participating in the queue broadcast.
        """
        engine = engines[rank]
        engine._fetch_and_enqueue_requests(engine.waiting_queue, 1)

    if already_waiting:
        group.run(fetch)
    engines[0].cancel_context_request("cancelled")
    assert queue.pending_cancellation_ids() == set()
    group.run(fetch)

    for engine in engines:
        assert engine.canceled_req_ids == [cancelled_id]
        engine._cancel_queued_context_requests(engine.waiting_queue)
        assert [item.id for item in engine.waiting_queue] == [healthy_id]
        assert not engine.canceled_req_ids
        engine._do_terminate_request.assert_not_called()
        engine.async_transfer_manager.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    engines[0]._enqueue_responses.assert_called_once()
    engines[1]._enqueue_responses.assert_called_once()
    responses = engines[0]._enqueue_responses.call_args.args[0]
    assert len(responses) == 1 and responses[0][0] == cancelled_id
    assert responses[0][1].result.get_result().finish_reasons == [FinishReason.CANCELLED]
    assert engines[1]._enqueue_responses.call_args.args[0] == []


@pytest.mark.parametrize("retained_only", [False, True])
def test_token_abort_preserves_retained_owner_and_earlier_result(retained_only: bool) -> None:
    """Token routing reaches retained CTX without reinterpreting IDs or releasing resources."""
    engine = _executor()
    request, healthy = _active_request(11), _active_request(101)
    request.py_disaggregated_params.context_cancellation_token = "target"
    healthy.py_disaggregated_params.context_cancellation_token = "healthy"
    engine.async_transfer_manager.start_transfer(request)
    engine.active_requests = [healthy] if retained_only else [request, healthy]
    if retained_only:
        request.state = LlmRequestState.GENERATION_COMPLETE
    previous_state = request.state

    engine.cancel_context_request("target")
    engine.cancel_context_request("target")
    engine.disagg.request_logical_context_cancellation.assert_not_called()
    queued = engine.executor_request_queue.get_from_request_queue(
        datetime.timedelta(0), wait_for_batch=False
    )
    assert len(queued) == 2
    engine._handle_special_queue_items(queued)
    assert engine._progress_context_cancellations([healthy]) == [healthy]

    engine.disagg.request_logical_context_cancellation.assert_called_once_with(request)
    assert engine.active_requests == [healthy]
    assert engine.async_transfer_manager.requests_in_transfer() == {11: request}
    engine.async_transfer_manager.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    engine._do_terminate_request.assert_not_called()
    assert not engine.canceled_req_ids
    if retained_only:
        assert request.state == previous_state
        engine._enqueue_responses.assert_not_called()


def test_batched_token_cancellations_scan_existing_work_once_in_queue_order() -> None:
    """A burst sees only preceding submits, without rescanning existing work per token."""
    engine = _executor()
    queue = engine.executor_request_queue
    transfers = engine.async_transfer_manager
    active = _active_request(101)
    active.py_disaggregated_params.context_cancellation_token = "active"
    engine.active_requests.append(active)
    transfers.start_transfer(active)

    with patch.object(
        transfers, "requests_in_transfer", wraps=transfers.requests_in_transfer
    ) as scan:
        waiting_id = queue.enqueue_request(_raw_with_token("waiting"))
        items = queue.get_from_request_queue(datetime.timedelta(0), wait_for_batch=False)
        engine.waiting_queue.add_requests(engine._handle_special_queue_items(items))
        scan.assert_not_called()

        engine.cancel_context_request("later")
        later_id = queue.enqueue_request(_raw_with_token("later"))
        engine.cancel_context_request("active")
        engine.cancel_context_request("active")
        engine.cancel_context_request("later")
        engine.cancel_context_request("waiting")
        items = queue.get_from_request_queue(datetime.timedelta(0), wait_for_batch=False)
        accepted = engine._handle_special_queue_items(items)

        assert [item.id for item in accepted] == [later_id]
        assert engine.canceled_req_ids == [101, 101, later_id, waiting_id]
        scan.assert_called_once_with()


def test_token_cancel_drain_preserves_future_submit_and_cancel_order() -> None:
    """Only old retained work crosses a drain barrier on either participating rank."""
    group, engines = _engines()
    retained = []
    for engine in engines:
        request = _active_request()
        request.py_disaggregated_params.context_cancellation_token = "old"
        engine.async_transfer_manager.start_transfer(request)
        retained.append(request)
    queue = engines[0].executor_request_queue
    queue.enqueue_control_request(control_id="drain")
    future_id = queue.enqueue_request(_raw_with_token("future"))
    engines[0].cancel_context_request("future")
    engines[0].cancel_context_request("old")

    def fetch(rank: int) -> None:
        """Run the actual control-aware request fetch on one rank.

        Args:
            rank: Participating executor rank.
        """
        engine = engines[rank]
        engine._fetch_and_enqueue_requests(engine.waiting_queue, 0)

    group.run(fetch)
    group.run(fetch)
    for engine, request in zip(engines, retained):
        assert engine.control_requests[0].id == CONTROL_REQUEST_ID
        assert not engine.waiting_queue
        assert engine.canceled_req_ids == [request.py_request_id]
        assert engine._progress_context_cancellations([]) == []
        assert engine.disagg.request_logical_context_cancellation.call_args_list == [call(request)]
        engine.async_transfer_manager.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    deferred = engines[0].request_accumulated
    assert len(deferred) == 2
    assert deferred[0].id == future_id
    assert deferred[0].context_cancellation_token == "future"
    assert deferred[1].cancel_context_token == "future"
    assert engines[1].request_accumulated == []
    assert [name for name, _ in engines[0].dist.calls] == [
        name for name, _ in engines[1].dist.calls
    ]


@pytest.mark.parametrize("retained_only", [False, True])
def test_token_cancel_drain_finds_work_on_another_attention_dp_rank(
    retained_only: bool,
) -> None:
    """The fixed drain collective includes remote tokens without admitting future work."""
    group, engines = _engines()
    for engine in engines:
        engine.enable_attention_dp = True
    request = _active_request(11)
    request.py_disaggregated_params.context_cancellation_token = "remote"
    if retained_only:
        engines[1].async_transfer_manager.start_transfer(request)
    else:
        engines[1].active_requests.append(request)
    queue = engines[0].executor_request_queue
    future_id = queue.enqueue_request(_raw_with_token("future"))
    engines[0].cancel_context_request("future")
    engines[0].cancel_context_request("remote")
    fetched = [None, None]

    def fetch(rank: int) -> None:
        """Run the real symmetric drain fetch with rank-local request ownership."""
        engine = engines[rank]
        fetched[rank] = engine._fetch_drain_cancellations(engine.waiting_queue)

    group.run(fetch)
    assert [item.cancel_context_token for item in fetched[0]] == ["remote"]
    assert fetched[1] == []
    for engine in engines:
        engine._handle_special_queue_items(fetched[0])
    assert engines[0].canceled_req_ids == []
    assert engines[1].canceled_req_ids == [11]
    deferred = engines[0].request_accumulated
    assert len(deferred) == 2
    assert deferred[0].id == future_id
    assert deferred[1].cancel_context_token == "future"
    assert not engines[1].request_accumulated
    for engine in engines:
        engine._do_terminate_request.assert_not_called()
        engine.async_transfer_manager.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    assert [name for name, _ in engines[0].dist.calls] == [
        name for name, _ in engines[1].dist.calls
    ]


def test_nonmatching_token_does_not_cancel_or_create_pending_state() -> None:
    """A retired token cannot attach cancellation to another request instance."""
    engine = _executor()
    request = _active_request()
    request.py_disaggregated_params.context_cancellation_token = "successor"
    engine.active_requests.append(request)
    engine.cancel_context_request("old")
    queued = engine.executor_request_queue.get_from_request_queue(
        datetime.timedelta(0), wait_for_batch=False
    )
    assert len(queued) == 1
    assert engine._handle_special_queue_items(queued) == []
    assert not engine.canceled_req_ids
    assert engine._progress_context_cancellations([request]) == [request]
    engine.disagg.request_logical_context_cancellation.assert_not_called()


@pytest.mark.parametrize(
    "state,accepted",
    [
        (RequestAdmissionState.RUNNING, True),
        (RequestAdmissionState.PARKING, True),
        (RequestAdmissionState.PARKED, False),
        (RequestAdmissionState.FAILED, False),
    ],
)
def test_token_cancel_receipt_means_queue_only(
    state: RequestAdmissionState, accepted: bool
) -> None:
    """Queue acceptance does not claim that executor handling or retirement finished."""
    engine = _executor()
    engine.executor_request_queue.admission_state = state
    if accepted:
        engine.cancel_context_request("target")
    else:
        with pytest.raises(RuntimeError, match="cancellation queue is closed"):
            engine.cancel_context_request("target")
    assert engine.executor_request_queue.request_queue.qsize() == int(accepted)
    assert not engine.canceled_req_ids
    engine.disagg.request_logical_context_cancellation.assert_not_called()


def test_queue_distinguishes_numeric_cancel_token_cancel_and_benchmark_vote() -> None:
    """A benchmark vote is not a cancellation, and only numeric cancels expose pending IDs."""
    engine = _executor()
    queue = engine.executor_request_queue
    engine.cancel_request(11)
    engine.cancel_context_request("11")
    queue.enqueue_benchmark_stage_vote_request()
    assert queue.pending_cancellation_ids() == {11}
    numeric, token, vote = queue.get_from_request_queue(datetime.timedelta(0), wait_for_batch=False)
    assert numeric.is_canceled_request and numeric.id == 11
    assert token.is_canceled_request and token.cancel_context_token == "11"
    assert not token.is_benchmark_stage_vote_request
    assert vote.is_benchmark_stage_vote_request and not vote.is_canceled_request


@pytest.mark.parametrize(
    ("runtime_support", "connector", "expected"),
    [(True, None, True), (False, None, False), (1, None, False), (True, object(), False)],
    ids=["supported", "unsupported", "truthy_not_true", "connector"],
)
def test_executor_context_capability_uses_coordinator(
    runtime_support: object, connector: object, expected: bool
) -> None:
    """The facade uses the coordinator, not a concrete transceiver attribute."""
    engine = _executor()
    engine.kv_cache_transceiver = None
    engine.kv_connector_manager = connector
    engine.disagg.supports_logical_context_cancellation.return_value = runtime_support
    assert engine.supports_logical_context_cancellation() is expected
    if connector is None:
        engine.disagg.supports_logical_context_cancellation.assert_called_once_with(None)
    else:
        engine.disagg.supports_logical_context_cancellation.assert_not_called()
