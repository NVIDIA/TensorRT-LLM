# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transfer error handling of the coordinator: the rank-local check, the ADP
vote and the poison consensus.

Single-rank cases run one ``CoordinatorHarness``; multi-rank cases run one per
rank over a ``FakeDistGroup`` and pin what each rank exchanges and what it
then asks its executor to do.
"""

from types import SimpleNamespace

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest
from fake_dist import FakeDistGroup

from tensorrt_llm.bindings import LlmRequestState

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

    h.coordinator.check_transfer_errors("generation requests")

    assert h.effects.failed == [
        ("Error in kv cache transfer for generation requests", [failed], False)
    ]
    assert h.dist.calls == []


def test_rank_local_check_defers_to_the_vote_under_multi_rank_adp() -> None:
    """Failing here would enter the executor's response collective from one
    rank; the loop-top vote fails replicas on every rank together instead."""
    h = _single_rank(world_size=2, enable_attention_dp=True)
    h.active.append(_failed_gen(1))

    h.coordinator.check_transfer_errors("context requests")

    assert h.effects.failed == []
    assert h.dist.calls == []


def test_rank_local_check_handles_errors_on_a_single_adp_rank() -> None:
    h = _single_rank(enable_attention_dp=True)
    failed = _failed_gen(1)
    h.active.append(failed)

    h.coordinator.check_transfer_errors("generation requests")

    assert [requests for _, requests, _ in h.effects.failed] == [[failed]]


def test_user_cancelled_failed_requests_are_left_to_the_cancel_path() -> None:
    h = _single_rank()
    h.active.append(_failed_gen(1))
    h.registry.canceled = [1]

    h.coordinator.check_transfer_errors("generation requests")

    assert h.effects.failed == []


def test_failed_context_send_waits_until_every_transfer_owner_released_it() -> None:
    """The KV connector may still hold the request's blocks when its send
    fails. The reap releases only the transceiver's claim, and its own error
    check must leave the request alone while the connector's claim stands;
    the error is applied once the last owner lets go."""
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

    h.coordinator.release_transfer(failed)  # the connector lets go
    h.coordinator.check_transfer_errors("context requests")

    assert not h.in_transfer(failed)
    assert h.effects.failed == [
        ("Error in kv cache transfer for context requests", [failed], False)
    ]


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
