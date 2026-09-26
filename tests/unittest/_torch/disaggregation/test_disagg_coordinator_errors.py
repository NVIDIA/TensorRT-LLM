# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transfer error handling of the coordinator: the rank-local check, the ADP
vote and the poison consensus.

Single-rank cases run one ``CoordinatorHarness``; multi-rank cases run one per
rank over a ``FakeDistGroup`` and pin what each rank exchanges and what it
then asks its executor to do.
"""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest
from fake_dist import FakeDistGroup
from fake_kv_cache_transceiver import FakeKvCacheTransceiver

from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.disagg_adapter import (
    PyExecutorEffects,
    PyExecutorRequestRegistry,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

pytestmark = pytest.mark.cpu_only

_POISON_MSG = "Disagg KV cache transfer buffer is poisoned; process restart is required"
_VOTE_MSG = "Disagg KV cache transfer error"


def _failed_gen(rid: int, **overrides) -> TransferRequest:
    return TransferRequest(
        rid, is_context_only_request=False, state=LlmRequestState.DISAGG_TRANS_ERROR, **overrides
    )


def _running_gen(rid: int, **overrides) -> TransferRequest:
    return TransferRequest(
        rid,
        is_context_only_request=False,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        **overrides,
    )


def _single_rank(world_size: int = 1, **kwargs) -> CoordinatorHarness:
    """One harness on rank 0 of a world; its dist records but never blocks."""
    return CoordinatorHarness(dist=FakeDistGroup(world_size, world_size).rank(0), **kwargs)


def _ranks(group: FakeDistGroup, **kwargs) -> list:
    return [CoordinatorHarness(dist=group.rank(rank), **kwargs) for rank in range(group.world_size)]


def _vote(error_ids=(), blocked_ids=()) -> tuple:
    return "tp_allgather", {"error_ids": list(error_ids), "blocked_ids": list(blocked_ids)}


# -- rank-local check --------------------------------------------------------


def test_rank_local_check_fails_failed_requests_and_names_the_kind() -> None:
    h = _single_rank()
    failed, running = _failed_gen(1), _running_gen(2)
    h.active.extend([failed, running])

    h.coordinator._check_transfer_errors("generation requests")

    assert h.effects.failed == [
        ("Error in kv cache transfer for generation requests", [failed], False)
    ]
    assert h.dist.calls == []


def test_rank_local_check_defers_to_the_vote_under_multi_rank_adp() -> None:
    """Failing here would enter the executor's response collective from one
    rank; the loop-top vote fails replicas on every rank together instead."""
    h = _single_rank(world_size=2, enable_attention_dp=True)
    h.active.append(_failed_gen(1))

    h.coordinator._check_transfer_errors("context requests")

    assert h.effects.failed == []
    assert h.dist.calls == []


def test_rank_local_check_handles_errors_on_a_single_adp_rank() -> None:
    h = _single_rank(enable_attention_dp=True)
    failed = _failed_gen(1)
    h.active.append(failed)

    h.coordinator._check_transfer_errors("generation requests")

    assert [requests for _, requests, _ in h.effects.failed] == [[failed]]


def test_user_cancelled_failed_requests_are_left_to_the_cancel_path() -> None:
    h = _single_rank()
    h.active.append(_failed_gen(1))
    h.registry.canceled = [1]

    h.coordinator._check_transfer_errors("generation requests")

    assert h.effects.failed == []


def test_failed_context_send_waits_until_every_transfer_owner_released_it() -> None:
    """The KV connector may still hold the request's blocks when its send
    fails. The reap releases only the transceiver's claim, and its own error
    check must leave the request alone while the connector's claim stands;
    the error is applied once the last owner lets go, and the blocks and the
    error are each applied exactly once after that."""
    h = _single_rank()
    failed = TransferRequest(7)
    h.active.append(failed)
    h.send(failed)
    h.transfers.start_transfer(failed)  # the connector's claim
    h.transceiver.finish_send(failed, outcome="error")

    h.coordinator.reap_context_sends(0)

    assert failed.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert h.in_transfer(failed)
    assert h.active == [failed]
    assert h.effects.failed == []
    h.kv_cache_manager.unpin_blocks_by_id.assert_not_called()

    h.coordinator.release_transfer(failed)  # the connector lets go
    h.coordinator._check_transfer_errors("context requests")

    assert not h.in_transfer(failed)
    h.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(7)
    assert h.effects.failed == [
        ("Error in kv cache transfer for context requests", [failed], False)
    ]

    # A stray extra release finds no claim to drop, and once the executor's
    # error path has removed the request a further check has nothing to fail.
    h.coordinator.release_transfer(failed)
    h.active.remove(failed)
    h.coordinator._check_transfer_errors("context requests")

    h.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(7)
    assert len(h.effects.failed) == 1


# -- synced handler outside the ADP vote -------------------------------------


def test_synced_handler_leaves_rank_local_errors_to_the_reaps_outside_adp() -> None:
    """Without multi-rank ADP the reaps already failed what they could; the
    loop-top pass only applies context failures reported after release."""
    h = _single_rank(world_size=2)
    h.active.append(_failed_gen(1))

    h.coordinator.handle_errors_synced()

    assert h.effects.failed == []
    assert h.dist.calls == []


def test_context_failure_reported_after_release_is_applied_at_the_synced_pass() -> None:
    """An error id the transfer manager no longer knows is parked by the reap;
    the next loop-top pass flips the request to the error state and fails it
    as a context request."""
    h = _single_rank()
    late = TransferRequest(9)
    h.active.append(late)
    status = SimpleNamespace(completed_request_ids=[], error_request_ids=[9])
    h.transceiver.check_context_transfer_status = lambda at_least, mark_complete=False: status

    h.coordinator.reap_context_sends(0)
    assert h.effects.failed == []

    h.coordinator.handle_errors_synced()

    assert late.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert h.effects.failed == [("Error in kv cache transfer for context requests", [late], False)]


# -- ADP vote ----------------------------------------------------------------


def test_peer_error_fails_the_local_replica_and_spares_unrelated_requests() -> None:
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group, enable_attention_dp=True)
    replica, unrelated, failed = _running_gen(7), _running_gen(8), _failed_gen(7)
    ranks[0].active.extend([replica, unrelated])
    ranks[1].active.append(failed)

    group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())

    assert ranks[0].dist.calls == [_vote()]
    assert ranks[1].dist.calls == [_vote(error_ids=[7])]
    assert ranks[0].effects.failed == [(_VOTE_MSG, [replica], False)]
    assert ranks[1].effects.failed == [(_VOTE_MSG, [failed], False)]


def test_every_rank_votes_even_when_no_rank_has_errors() -> None:
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group, enable_attention_dp=True)
    for h in ranks:
        h.active.append(_running_gen(1))

    group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())

    assert [h.dist.calls for h in ranks] == [[_vote()], [_vote()]]
    assert [h.effects.failed for h in ranks] == [[], []]


def test_peer_error_without_a_local_replica_still_enters_the_error_path() -> None:
    """The executor's error path runs a response collective; a rank with no
    matching request enters it with an empty list rather than skipping it."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group, enable_attention_dp=True)
    ranks[0].active.append(_running_gen(8))
    ranks[1].active.append(_failed_gen(7))

    group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())

    assert ranks[0].effects.failed == [(_VOTE_MSG, [], False)]
    assert [requests for _, requests, _ in ranks[1].effects.failed] == [[ranks[1].active[0]]]


def test_child_requests_vote_and_fail_by_parent_id() -> None:
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group, enable_attention_dp=True)
    child = _running_gen(101, is_child=True, parent_request_id=9)
    ranks[0].active.append(child)
    ranks[1].active.append(_failed_gen(9))

    group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())

    assert ranks[1].dist.calls == [_vote(error_ids=[9])]
    assert ranks[0].effects.failed == [(_VOTE_MSG, [child], False)]


@pytest.mark.parametrize("blocker", ["user_cancelled", "context_send_still_owned"])
def test_a_locally_blocked_request_vetoes_the_vote_on_every_rank(blocker: str) -> None:
    """Rank 0 cannot clean request 7 yet: the cancel path owns it, or a
    transfer owner still holds its blocks. Its blocked vote keeps every rank,
    including the one that reported the error, from failing 7 this round."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group, enable_attention_dp=True)
    blocked = TransferRequest(7)
    ranks[0].active.append(blocked)
    if blocker == "user_cancelled":
        ranks[0].registry.canceled = [7]
    else:
        ranks[0].transfers.start_transfer(blocked)
    # The send fails after the transfer started: start_transfer itself moves
    # the request to TRANS_IN_PROGRESS, so the error state must come last.
    blocked.state = LlmRequestState.DISAGG_TRANS_ERROR
    ranks[1].active.append(_failed_gen(7))

    group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())

    assert ranks[0].dist.calls == [_vote(error_ids=[7], blocked_ids=[7])]
    assert ranks[1].dist.calls == [_vote(error_ids=[7])]
    assert [h.effects.failed for h in ranks] == [[], []]


@pytest.mark.parametrize("blocker", ["user_cancelled", "context_send_still_owned"])
def test_a_vetoed_error_is_cleaned_up_once_the_blocker_clears(blocker: str) -> None:
    """Continuation of the veto: the round after the blocker clears reaches
    consensus and fails the error once on every rank that still holds it. A
    connector release puts rank 0's request back into the vote; a completed
    user cancel has already taken it out of active, so rank 0 enters the error
    path with nothing local while rank 1 fails its copy. Once the executors
    have removed the failed requests, the next round has nothing to do."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group, enable_attention_dp=True)
    blocked = TransferRequest(7)
    ranks[0].active.append(blocked)
    if blocker == "user_cancelled":
        ranks[0].registry.canceled = [7]
    else:
        ranks[0].transfers.start_transfer(blocked)
    blocked.state = LlmRequestState.DISAGG_TRANS_ERROR
    failed = _failed_gen(7)
    ranks[1].active.append(failed)

    def vote_round():
        group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())

    vote_round()  # vetoed
    assert [h.effects.failed for h in ranks] == [[], []]

    if blocker == "user_cancelled":
        # The cancel path finished the request: it is gone from rank 0.
        ranks[0].active.remove(blocked)
        ranks[0].registry.canceled = []
        rank0_round2_vote, rank0_failed = _vote(), [(_VOTE_MSG, [], False)]
    else:
        ranks[0].coordinator.release_transfer(blocked)  # the connector lets go
        rank0_round2_vote, rank0_failed = _vote(error_ids=[7]), [(_VOTE_MSG, [blocked], False)]

    vote_round()  # consensus
    assert ranks[0].dist.calls[-1] == rank0_round2_vote
    assert ranks[1].dist.calls[-1] == _vote(error_ids=[7])
    assert ranks[0].effects.failed == rank0_failed
    assert ranks[1].effects.failed == [(_VOTE_MSG, [failed], False)]

    for h in ranks:  # the executors' error paths removed what they failed
        h.active.clear()
    vote_round()
    assert [h.dist.calls[-1] for h in ranks] == [_vote(), _vote()]
    assert [len(h.effects.failed) for h in ranks] == [len(rank0_failed), 1]


# -- poison consensus --------------------------------------------------------


def test_one_poisoned_rank_takes_the_whole_world_down(inflight_cancel) -> None:
    """Poison is reduced over the world, not the TP group: with TP groups of
    one, rank 0 learns of rank 1's poison only through the world allreduce.
    Both ranks then fail fatally exactly once and skip the vote that would
    otherwise have failed their local error requests."""
    group = FakeDistGroup(world_size=2, tp_size=1)
    ranks = _ranks(group, enable_attention_dp=True, supports_inflight_cancellation=True)
    ranks[1].transceiver.has_poisoned_transfer_buffer = lambda: True
    for h in ranks:
        h.active.append(_failed_gen(1))

    group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())

    assert [h.dist.calls for h in ranks] == [[("allreduce", 0)], [("allreduce", 1)]]
    assert [h.effects.fatal for h in ranks] == [[_POISON_MSG], [_POISON_MSG]]
    assert [h.effects.failed for h in ranks] == [[], []]


def test_poison_is_only_checked_when_in_flight_cancellation_is_active() -> None:
    group = FakeDistGroup(world_size=2, tp_size=1)
    ranks = _ranks(group, supports_inflight_cancellation=True)
    ranks[1].transceiver.has_poisoned_transfer_buffer = lambda: True

    group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())

    assert [h.dist.calls for h in ranks] == [[], []]
    assert [h.effects.fatal for h in ranks] == [[], []]


def test_single_rank_poison_needs_no_collective(inflight_cancel) -> None:
    h = _single_rank(supports_inflight_cancellation=True)
    h.transceiver.has_poisoned_transfer_buffer = lambda: True

    h.coordinator.handle_errors_synced()

    assert h.dist.calls == []
    assert h.effects.fatal == [_POISON_MSG]


# -- through the real executor error path ------------------------------------


def _executor_with_real_error_path(monkeypatch) -> tuple:
    """Bare single-rank executor whose ``_handle_errors`` and
    ``_terminate_request`` run for real behind the production adapters, over
    a coordinator on the contract fake transceiver.

    Returns ``(executor, coordinator, transceiver)``.
    """
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    executor = object.__new__(PyExecutor)
    executor.active_requests = []
    executor.canceled_req_ids = []
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._fatal_error = None
    executor.is_shutdown = False
    executor.enable_attention_dp = False
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(
        rank=0, world_size=1, tp_size=1, pp_size=1, mapping=SimpleNamespace(tp_group=[0])
    )
    executor.response_cv = threading.Condition()
    executor.responses = {}
    executor.result_wait_queues = {}
    executor._disagg_pp_termination_handler = None
    executor._prefetched_request_ids = set()
    executor.resource_manager = Mock(spec=["free_resources"])
    # Request-scoped failures never consult the error budget; make any consult loud.
    executor._error_budget = Mock(spec=[])
    # What the cancel and response passes read besides the above.
    executor.waiting_queue = Mock(spec=["remove_by_ids"])
    executor.perf_manager = Mock()
    executor.model_engine = SimpleNamespace(route_capture=None)
    executor.iter_counter = 0
    executor.disable_overlap_scheduler = True
    executor.stream_interval = 1
    executor.force_terminate_ctx_for_partial_reuse = False
    transceiver = FakeKvCacheTransceiver()
    executor.kv_cache_transceiver = transceiver
    kv_cache_manager = Mock(spec=["store_blocks_for_reuse", "unpin_blocks_by_id"])
    transfers = AsyncTransferManager(
        SimpleNamespace(resource_managers={ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
    )
    coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=transfers,
        kv_cache_manager=kv_cache_manager,
        dist=executor.dist,
        effects=PyExecutorEffects(executor),
        registry=PyExecutorRequestRegistry(executor),
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
    )
    executor._disagg_coordinator = coordinator
    return executor, coordinator, transceiver


def _receiving_gen(rid: int) -> TransferRequest:
    return TransferRequest(
        rid,
        is_context_only_request=False,
        is_disagg_generation_transmission_in_progress=True,
        py_client_id=100 + rid,
        is_dummy_request=False,
    )


def test_generation_receive_failure_reaches_the_response_queue_through_the_real_error_path(
    monkeypatch,
) -> None:
    """Coordinator, production adapter, real ``_handle_errors``: the failed
    request's error response is enqueued for the consumer, the request leaves
    active_requests with its resources freed, the healthy request is
    untouched, and a later poll changes none of that."""
    executor, coordinator, transceiver = _executor_with_real_error_path(monkeypatch)
    failing, healthy = _receiving_gen(1), _receiving_gen(2)
    executor.active_requests.extend([failing, healthy])
    transceiver.request_and_receive_async(failing)
    transceiver.request_and_receive_async(healthy)
    transceiver.finish_recv(failing, outcome="error")

    coordinator.poll_gen_transfers()

    (response,) = executor.responses[1]
    assert (response.request_id, response.client_id) == (1, 101)
    assert response.error_msg == "Error in kv cache transfer for generation requests"
    assert failing.state == LlmRequestState.GENERATION_COMPLETE
    assert executor.active_requests == [healthy]
    executor.resource_manager.free_resources.assert_called_once_with(failing)
    assert healthy.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert 2 not in executor.responses

    coordinator.poll_gen_transfers()
    assert len(executor.responses[1]) == 1
    executor.resource_manager.free_resources.assert_called_once_with(failing)

    transceiver.finish_recv(healthy)
    coordinator.poll_gen_transfers()
    assert healthy.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    assert executor.active_requests == [healthy]


def _gen_request(rid: int) -> LlmRequest:
    """Real generation-only request: cancellation drives the C++ finish state,
    and the response pass serializes a real result."""
    return LlmRequest(
        request_id=rid,
        max_new_tokens=8,
        input_tokens=list(range(8)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        draft_tokens=None,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
    )


def _iteration(executor: PyExecutor, coordinator: DisaggTransferCoordinator) -> None:
    """The slice of an executor iteration that settles user cancellations: the
    cancel pass, the transfer poll and the response pass."""
    executor._handle_canceled_requests()
    coordinator.poll_gen_transfers()
    executor._handle_responses()


def test_refused_python_cancel_is_retried_and_settled_through_the_real_response_path(
    monkeypatch,
) -> None:
    """Python transceiver semantics: ``cancel_request`` returning False means a
    task is mid-write and the caller retries next iteration; True means the KV
    may be freed. The refused request stays pending, unfinished and unreleased.
    The accepted retry finishes it; the real response pass answers it once and
    terminates it once, freeing its resources; the other request keeps
    receiving and completes afterwards."""
    executor, coordinator, transceiver = _executor_with_real_error_path(monkeypatch)
    cancelled, kept = _gen_request(1), _gen_request(2)
    for req in (cancelled, kept):
        transceiver.request_and_receive_async(req)
    executor.active_requests.extend([cancelled, kept])
    attempts, immediate_cancel = [], transceiver.cancel_request

    def cancel(request):  # mid-write on the first attempt, then the real cancel
        attempts.append(request.py_request_id)
        return immediate_cancel(request) if len(attempts) > 1 else False

    transceiver.cancel_request = cancel
    executor.canceled_req_ids = [1]

    _iteration(executor, coordinator)  # refused
    assert attempts == [1]
    assert executor.canceled_req_ids == [1]
    assert not cancelled.is_finished
    assert cancelled.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert executor.active_requests == [cancelled, kept]
    assert executor.responses == {}
    executor.resource_manager.free_resources.assert_not_called()

    _iteration(executor, coordinator)  # accepted
    assert attempts == [1, 1]
    assert executor.canceled_req_ids == []
    assert cancelled.is_finished
    assert cancelled.state == LlmRequestState.GENERATION_COMPLETE
    (response,) = executor.responses[1]
    assert response.error_msg is None
    assert executor.active_requests == [kept]
    executor.resource_manager.free_resources.assert_called_once_with(cancelled)
    assert kept.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS

    transceiver.finish_recv(kept)
    coordinator.poll_gen_transfers()
    assert kept.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    executor.resource_manager.free_resources.assert_called_once_with(cancelled)


def test_cancelling_part_of_a_saturated_receive_batch_settles_only_those_requests(
    monkeypatch,
) -> None:
    """Sixteen generation requests are receiving and the user cancels eight.
    Through the real cancel and response passes each cancelled request is
    cancelled at the transceiver once, finished as CANCELLED, answered once and
    terminated once with its resources freed; the other eight are untouched,
    keep receiving and complete afterwards."""
    executor, coordinator, transceiver = _executor_with_real_error_path(monkeypatch)
    requests = [_gen_request(rid) for rid in range(1, 17)]
    for req in requests:
        transceiver.request_and_receive_async(req)
    executor.active_requests.extend(requests)
    cancelled, kept = requests[:8], requests[8:]
    executor.canceled_req_ids = [req.py_request_id for req in cancelled]

    _iteration(executor, coordinator)

    assert executor.canceled_req_ids == []
    executor.waiting_queue.remove_by_ids.assert_called_once_with(set(range(1, 9)))
    for req in cancelled:
        assert req.is_finished
        assert req.state == LlmRequestState.GENERATION_COMPLETE
        assert transceiver.call_log.count(f"cancel_request:{req.py_request_id}") == 1
        (response,) = executor.responses[req.py_request_id]
        assert response.error_msg is None
    assert executor.active_requests == kept
    freed = [call.args[0] for call in executor.resource_manager.free_resources.call_args_list]
    assert freed == cancelled
    for req in kept:
        assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        assert f"cancel_request:{req.py_request_id}" not in transceiver.call_log
        assert req.py_request_id not in executor.responses

    for req in kept:
        transceiver.finish_recv(req)
    coordinator.poll_gen_transfers()

    assert all(req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE for req in kept)
    assert transceiver.check_gen_transfer_complete()
    assert len(executor.resource_manager.free_resources.call_args_list) == 8
