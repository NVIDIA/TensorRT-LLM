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

from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_module
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


# -- Collectives under rank skew ---------------------------------------------


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


# -- generation transfer timeout on an idle server ---------------------------

_GEN_ERROR = "Error in kv cache transfer for generation requests"


def _cancel_only_marks_inflight_receives(transceiver) -> None:
    """Model the Python transceiver with a receive in flight: cancelling it only
    marks the session cancelled and returns False; the next status poll
    reports the cancelled session."""
    cancel_finished_transfer = transceiver.cancel_request

    def cancel_request(req) -> bool:
        if req.py_request_id not in transceiver._pending_recvs:
            return cancel_finished_transfer(req)
        transceiver.call_log.append(f"cancel_request:{req.py_request_id}")
        transceiver.cancel_recv_remotely(req)
        return False

    transceiver.cancel_request = cancel_request


def _executor_drops_failed_requests(h: CoordinatorHarness) -> None:
    """The executor's error path removes the requests it fails from active_requests."""
    record_failure = h.effects.fail_requests

    def fail_requests(error_msg, requests, *, charge_budget) -> None:
        record_failure(error_msg, requests, charge_budget=charge_budget)
        for req in requests:
            if req in h.active:
                h.active.remove(req)

    h.effects.fail_requests = fail_requests


def _idle_loop_heads(h: CoordinatorHarness, iterations: int) -> None:
    """Loop heads of an executor with nothing scheduled: the transfer poll,
    then the timeout check, and no response pass."""
    for _ in range(iterations):
        h.coordinator.poll_gen_transfers()
        h.coordinator.check_transfer_timeouts()


@pytest.mark.parametrize("inflight_requested", [False, True])
@pytest.mark.parametrize(
    "cancel_finishes_at_once, error_msg",
    [(True, "Request 1 timed out"), (False, _GEN_ERROR)],
)
def test_idle_generation_server_fails_a_receive_whose_kv_never_arrives(
    monkeypatch, clock, cancel_finishes_at_once: bool, error_msg: str, inflight_requested: bool
) -> None:
    """Nothing is scheduled, so the response pass never runs. The loop head
    alone must cancel the receive once kv_transfer_timeout_ms passes and fail
    the request, whether the cancel finishes at once or is reported by the
    next status poll. Requesting in-flight cancellation from a transceiver
    that does not support it takes the same path."""
    monkeypatch.setattr(
        coordinator_module, "is_disagg_inflight_cancel_enabled", lambda: inflight_requested
    )
    h = CoordinatorHarness(kv_transfer_timeout_ms=1000)
    if not cancel_finishes_at_once:
        _cancel_only_marks_inflight_receives(h.transceiver)
    _executor_drops_failed_requests(h)
    req = _receiving(h, 1, started_at=clock["t"])
    clock["t"] += 2.0

    _idle_loop_heads(h, 3)

    assert h.transceiver.call_log.count("cancel_request:1") == 1
    assert h.effects.failed == [(error_msg, [req], False)]
    assert h.active == []
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert h.registry.removed == ([req] if cancel_finishes_at_once else [])


def test_idle_generation_timeout_asks_a_refused_cancel_only_once(no_inflight_cancel, clock) -> None:
    """C++ semantics without in-flight cancellation: a receive already in
    flight cannot be cancelled, and nothing is reported for the refusal.
    Asking again every loop head would only repeat the transceiver's warning,
    so the cancel is asked once and the request waits until the transceiver
    settles the receive."""
    h = CoordinatorHarness(kv_transfer_timeout_ms=1000)
    _executor_drops_failed_requests(h)
    req = _receiving(h, 1, started_at=clock["t"])
    clock["t"] += 2.0

    def refuse(request) -> bool:
        h.transceiver.call_log.append(f"cancel_request:{request.py_request_id}")
        return False

    h.transceiver.cancel_request = refuse

    _idle_loop_heads(h, 5)

    assert h.transceiver.call_log.count("cancel_request:1") == 1
    assert h.effects.failed == []
    assert h.active == [req]
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS

    h.transceiver.finish_recv(req, outcome="error")
    h.coordinator.poll_gen_transfers()

    assert h.effects.failed == [(_GEN_ERROR, [req], False)]
    assert h.active == []
    assert h.transceiver.call_log.count("cancel_request:1") == 1


def test_idle_generation_timeout_under_adp_waits_for_the_synced_drain(
    no_inflight_cancel, clock
) -> None:
    """Under multi-rank ADP the loop head cancels the receive and ends its
    receive state at once, but the error response waits for the synced
    drain. The loop head itself enters no collective."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = _ranks(group, kv_transfer_timeout_ms=1000, enable_attention_dp=True)
    req = _receiving(ranks[0], 1, started_at=clock["t"])
    clock["t"] += 2.0
    ranks[0].coordinator.check_transfer_timeouts()

    group.run(lambda rank: ranks[rank].coordinator.poll_gen_transfers())

    assert ranks[0].transceiver.call_log.count("cancel_request:1") == 1
    assert ranks[0].active == []
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert [h.effects.failed for h in ranks] == [[], []]
    assert [h.dist.calls for h in ranks] == [[], []]

    group.run(lambda rank: ranks[rank].coordinator.handle_timeouts_synced())

    assert ranks[0].dist.calls == [("tp_allgather_int64", [True])]
    assert ranks[1].dist.calls == [("tp_allgather_int64", [False])]
    assert ranks[0].effects.failed == [("Request timed out (KV transfer)", [req], False)]
    assert ranks[1].effects.failed == [("Request timed out (KV transfer)", [], False)]


def test_idle_generation_timeout_leaves_fresh_and_user_canceled_receives(
    no_inflight_cancel, clock
) -> None:
    """A receive inside its timeout is not touched, and a timed-out receive
    the user cancelled is left to the executor's cancel path."""
    h = CoordinatorHarness(kv_transfer_timeout_ms=1000)
    fresh = _receiving(h, 1, started_at=clock["t"])
    user_canceled = _receiving(h, 2, started_at=clock["t"] - 2.0)
    h.registry.canceled.append(2)
    h.coordinator.check_transfer_timeouts()

    h.coordinator.poll_gen_transfers()

    assert not fresh.py_kv_transfer_timed_out and user_canceled.py_kv_transfer_timed_out
    assert "cancel_request:1" not in h.transceiver.call_log
    assert "cancel_request:2" not in h.transceiver.call_log
    assert h.active == [fresh, user_canceled]
    assert h.effects.failed == []
