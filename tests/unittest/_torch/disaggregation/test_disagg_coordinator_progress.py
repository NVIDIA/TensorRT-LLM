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

from types import SimpleNamespace

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest
from fake_dist import FakeDistGroup

from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _async_transfer_mode(monkeypatch) -> None:
    """Asynchronous generation transfers unless a test sets a mode knob itself."""
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)


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


# -- idle progress poll ------------------------------------------------------


_CONTEXT_POLL = "check_context_transfer_status:0"


def test_idle_poll_reaps_context_sends_and_leaves_gen_status_to_the_loop_head() -> None:
    """The loop head already polls generation status every iteration. The
    context poll is a consensus inside the transceiver and rank-symmetric, so
    no dist collective gates it, whatever the model-parallel layout."""
    h = CoordinatorHarness(dist=FakeDistGroup(world_size=16, tp_size=4).rank(0))

    h.coordinator.poll_progress_when_idle()

    assert h.transceiver.call_log == [_CONTEXT_POLL]
    assert h.dist.calls == []


def test_gen_only_benchmark_still_reaps_context_sends_when_idle(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", "1")
    h = CoordinatorHarness(dist=FakeDistGroup(world_size=4, tp_size=4).rank(0))

    h.coordinator.poll_progress_when_idle()

    assert h.transceiver.call_log == [_CONTEXT_POLL]
    assert h.dist.calls == []


def test_sync_transfers_skip_the_idle_poll_on_a_multi_rank_worker(monkeypatch) -> None:
    """A synchronous GEN receive blocks rank-locally, so a multi-rank worker
    must not enter the context status collective from the idle path, even
    with a send in flight."""
    monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
    h = CoordinatorHarness(dist=FakeDistGroup(world_size=4, tp_size=4).rank(0))
    h.send(TransferRequest(1))
    sent = list(h.transceiver.call_log)

    h.coordinator.poll_progress_when_idle()

    assert h.transceiver.call_log == sent
    assert h.dist.calls == []


@pytest.mark.parametrize("send_in_flight", [False, True])
def test_sync_transfers_on_a_single_rank_poll_only_while_a_send_is_in_flight(
    monkeypatch, send_in_flight: bool
) -> None:
    monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
    h = CoordinatorHarness()
    if send_in_flight:
        h.send(TransferRequest(1))
    before = list(h.transceiver.call_log)

    h.coordinator.poll_progress_when_idle()

    assert h.transceiver.call_log == before + ([_CONTEXT_POLL] if send_in_flight else [])


def test_async_idle_poll_is_entered_by_every_rank_regardless_of_local_sends() -> None:
    """A rank with nothing in flight enters the context status poll too; the
    consensus is inside the transceiver and needs no dist collective."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group)
    ranks[1].send(TransferRequest(1))

    group.run(lambda rank: ranks[rank].coordinator.poll_progress_when_idle())

    assert [h.transceiver.call_log.count(_CONTEXT_POLL) for h in ranks] == [1, 1]
    assert [h.dist.calls for h in ranks] == [[], []]


def test_sync_idle_poll_is_skipped_by_every_rank_of_a_multi_rank_worker(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group)
    ranks[1].send(TransferRequest(1))
    logs = [list(h.transceiver.call_log) for h in ranks]

    group.run(lambda rank: ranks[rank].coordinator.poll_progress_when_idle())

    assert [h.transceiver.call_log for h in ranks] == logs


# -- generation-first context gate -------------------------------------------


def _ctx_request(rid: int, schedule_style: DisaggScheduleStyle) -> TransferRequest:
    return TransferRequest(
        rid, py_disaggregated_params=SimpleNamespace(schedule_style=schedule_style)
    )


def test_context_gate_is_entered_even_without_new_requests() -> None:
    """The transceiver runs a consensus inside prepare_context_requests, so it
    is entered every iteration; skipping it on an empty list would leave a
    waiting request unpromoted on the ranks whose peer info arrived late."""
    h = CoordinatorHarness()

    h.coordinator.prepare_context_schedulable([])

    assert h.transceiver.call_log == ["prepare_context_requests:[]"]


def test_context_gate_receives_only_generation_first_context_requests() -> None:
    """Disaggregated params are read for context-only requests only."""
    h = CoordinatorHarness()
    gen_first = _ctx_request(1, DisaggScheduleStyle.GENERATION_FIRST)
    ctx_first = _ctx_request(2, DisaggScheduleStyle.CONTEXT_FIRST)
    gen_only = TransferRequest(3, is_context_only_request=False)

    h.coordinator.prepare_context_schedulable([ctx_first, gen_first, gen_only])

    assert h.transceiver.call_log == ["prepare_context_requests:[1]"]


def test_context_gate_is_entered_once_per_rank_whatever_arrived_locally() -> None:
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group)
    arrivals = [[], [_ctx_request(1, DisaggScheduleStyle.GENERATION_FIRST)]]

    group.run(lambda rank: ranks[rank].coordinator.prepare_context_schedulable(arrivals[rank]))

    assert [h.transceiver.call_log for h in ranks] == [
        ["prepare_context_requests:[]"],
        ["prepare_context_requests:[1]"],
    ]
    assert [h.dist.calls for h in ranks] == [[], []]
