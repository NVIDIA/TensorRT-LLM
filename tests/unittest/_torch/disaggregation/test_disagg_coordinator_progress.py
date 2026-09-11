# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-synchronized progress of the coordinator under FakeDist.

Every test runs one coordinator per rank, each on its own thread, over a
shared ``FakeDistGroup``. The ranks hold different local state; what is
pinned is that each rank still enters the same collectives in the same
order, with the payload the protocol really exchanges, and what each rank
then asks its executor to do. FakeDist checks the protocol only; blocking
semantics of real collectives need multi-process coverage.
"""

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest
from fake_dist import FakeDistGroup

from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only


def _ranks(group: FakeDistGroup, **kwargs) -> list:
    return [CoordinatorHarness(dist=group.rank(rank), **kwargs) for rank in range(group.world_size)]


def _collectives(h: CoordinatorHarness) -> list:
    return [name for name, _ in h.dist.calls]


def _receiving(h: CoordinatorHarness, rid: int, started_at: float) -> TransferRequest:
    req = TransferRequest(
        rid,
        is_context_only_request=False,
        is_disagg_generation_transmission_in_progress=True,
        py_kv_transfer_start_time=started_at,
    )
    h.active.append(req)
    h.transceiver.request_and_receive_async(req)
    return req


# -- CS-1 collectives under rank skew ----------------------------------------


def test_timeout_drain_exchanges_a_flag_and_each_rank_fails_its_own_requests() -> None:
    """Under multi-rank ADP the drain gathers one bool per rank; the rank whose
    peer timed out enters the error path with an empty list. Ids are not
    exchanged here, so nothing is failed on behalf of a peer."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group, enable_attention_dp=True)
    req = TransferRequest(1, is_context_only_request=False)
    ranks[0].coordinator.fail_timed_out([req])

    group.run(lambda rank: ranks[rank].coordinator.handle_timeouts_synced())

    assert ranks[0].dist.calls == [("tp_allgather_int64", [True])]
    assert ranks[1].dist.calls == [("tp_allgather_int64", [False])]
    assert ranks[0].effects.failed == [("Request timed out (KV transfer)", [req], False)]
    assert ranks[1].effects.failed == [("Request timed out (KV transfer)", [], False)]


def test_peer_timeout_is_mirrored_through_the_tp_wide_id_union(inflight_cancel, clock) -> None:
    """Only rank 1's copy of the receive has expired. Both ranks enter the same
    two collectives -- the flag reduce, then the id gather -- and rank 0
    cancels its copy on the peer's decision. Neither rank cancels twice."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group, kv_transfer_timeout_ms=1000, supports_inflight_cancellation=True)
    fresh = _receiving(ranks[0], 1, started_at=clock["t"])
    expired = _receiving(ranks[1], 1, started_at=clock["t"] - 2.0)

    group.run(lambda rank: ranks[rank].coordinator.poll_gen_transfers())

    assert ranks[0].dist.calls[:2] == [("tp_allreduce", 0), ("tp_allgather", [])]
    assert ranks[1].dist.calls[:2] == [("tp_allreduce", 1), ("tp_allgather", [1])]
    assert _collectives(ranks[0]) == _collectives(ranks[1])
    assert fresh.py_kv_transfer_timed_out and expired.py_kv_transfer_timed_out
    assert [h.transceiver.call_log.count("cancel_request:1") for h in ranks] == [1, 1]

    group.run(lambda rank: ranks[rank].coordinator.poll_gen_transfers())

    assert [h.transceiver.call_log.count("cancel_request:1") for h in ranks] == [1, 1]


def test_generation_error_flush_is_entered_by_the_rank_without_errors_too(
    inflight_cancel,
) -> None:
    """The flush reduces a flag over TP; the rank without local errors still
    enters the executor error path so the response collective stays aligned."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group, supports_inflight_cancellation=True)
    error_req = TransferRequest(
        1, is_context_only_request=False, state=LlmRequestState.DISAGG_TRANS_ERROR
    )
    ranks[0].active.append(error_req)
    ranks[0].delegates.requests_in_error_state.return_value = [error_req]

    group.run(lambda rank: ranks[rank].coordinator.poll_gen_transfers())

    assert ranks[0].dist.calls == [("tp_allreduce", 1)]
    assert ranks[1].dist.calls == [("tp_allreduce", 0)]
    assert ranks[0].effects.failed == [
        ("Error in kv cache transfer for generation requests", [error_req], False)
    ]
    assert ranks[1].effects.failed == [
        ("Error in kv cache transfer for generation requests", [], False)
    ]


def test_timeout_check_is_rank_local(clock) -> None:
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group, kv_transfer_timeout_ms=1000)
    requests = [_receiving(h, 1, started_at=clock["t"]) for h in ranks]
    clock["t"] += 2.0

    group.run(lambda rank: ranks[rank].coordinator.check_transfer_timeouts())

    assert all(req.py_kv_transfer_timed_out for req in requests)
    assert [h.dist.calls for h in ranks] == [[], []]
