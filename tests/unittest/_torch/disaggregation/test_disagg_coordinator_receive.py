# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Receive start: ``receive_gen_init`` prepares executor resources for the
admitted gen-init requests and starts their KV receive in the transfer mode
the environment selects.

Single-rank cases run one ``CoordinatorHarness``; the multi-rank case runs one
per rank over a ``FakeDistGroup``.
"""

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest
from fake_dist import FakeDistGroup

from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only

_GEN_ERROR_MSG = "Error in kv cache transfer for generation requests"
_VOTE_MSG = "Disagg KV cache transfer error"


@pytest.fixture(autouse=True)
def _async_transfer_mode(monkeypatch) -> None:
    """Asynchronous generation transfers unless a test sets a mode knob itself."""
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)


@pytest.fixture
def sync_mode(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")


@pytest.fixture
def gen_only_benchmark(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", "1")


def _gen_init(h: CoordinatorHarness, rid: int) -> TransferRequest:
    """A generation request admitted for receive, registered as active."""
    req = TransferRequest(
        rid, is_context_only_request=False, state=LlmRequestState.DISAGG_GENERATION_INIT
    )
    h.active.append(req)
    return req


def test_nothing_admitted_touches_neither_executor_nor_transceiver() -> None:
    """With no admitted request there is nothing to prepare and no receive to
    start, so no status poll is entered either."""
    h = CoordinatorHarness()

    h.coordinator.receive_gen_init([])

    assert h.effects.history == []
    assert h.transceiver.call_log == []


@pytest.mark.parametrize(
    "mode_env",
    [
        pytest.param({}, id="async"),
        pytest.param({"TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP": "1"}, id="sync"),
        pytest.param({"TRTLLM_DISAGG_BENCHMARK_GEN_ONLY": "1"}, id="gen_only_benchmark"),
    ],
)
def test_resources_are_prepared_once_before_any_receive_changes_state(
    monkeypatch, mode_env
) -> None:
    """The executor prepares resources for exactly the admitted batch, while
    every request is still in DISAGG_GENERATION_INIT; afterwards no request is
    left in that state, whatever the transfer mode."""
    for name, value in mode_env.items():
        monkeypatch.setenv(name, value)
    h = CoordinatorHarness()
    admitted = [_gen_init(h, 1), _gen_init(h, 2)]
    states_at_prepare = []
    prepare = h.effects.prepare_gen_resources

    def prepare_and_snapshot(requests):
        states_at_prepare.append([req.state for req in requests])
        prepare(requests)

    h.effects.prepare_gen_resources = prepare_and_snapshot

    h.coordinator.receive_gen_init(admitted)

    assert h.effects.prepared == [admitted]
    assert states_at_prepare == [[LlmRequestState.DISAGG_GENERATION_INIT] * 2]
    assert all(req.state != LlmRequestState.DISAGG_GENERATION_INIT for req in admitted)


@pytest.mark.parametrize("timeout_ms", [None, 1000])
def test_async_receive_starts_every_request_then_polls_status_once(clock, timeout_ms) -> None:
    """Every admitted request starts an asynchronous receive; the transfer
    timer is stamped only when a timeout is configured; one status poll
    follows so the receive-side consensus is entered this iteration."""
    h = CoordinatorHarness(kv_transfer_timeout_ms=timeout_ms)
    first, second = _gen_init(h, 1), _gen_init(h, 2)

    h.coordinator.receive_gen_init([first, second])

    assert h.transceiver.call_log == [
        "request_and_receive_async:1",
        "request_and_receive_async:2",
        "check_gen_transfer_status:0",
    ]
    assert [req.state for req in (first, second)] == [
        LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    ] * 2
    expected_start = clock["t"] if timeout_ms is not None else None
    assert [req.py_kv_transfer_start_time for req in (first, second)] == [expected_start] * 2
    assert h.effects.failed == []


@pytest.mark.parametrize("outcome", ["complete", "error"])
def test_async_receive_that_settles_immediately_is_reaped_in_the_same_call(outcome) -> None:
    """The trailing status poll picks up a receive that settled as soon as it
    started, so its completion or failure is applied without waiting for the
    next iteration."""
    h = CoordinatorHarness()
    req = _gen_init(h, 1)
    start = h.transceiver.request_and_receive_async

    def start_and_settle(request):
        start(request)
        h.transceiver.finish_recv(request, outcome)

    h.transceiver.request_and_receive_async = start_and_settle

    h.coordinator.receive_gen_init([req])

    if outcome == "complete":
        assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        assert h.effects.failed == []
    else:
        assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
        assert h.effects.failed == [(_GEN_ERROR_MSG, [req], False)]


def test_sync_receive_drains_every_request_before_the_rank_local_error_check(sync_mode) -> None:
    """A blocking receive that fails must not stop the later ones: every
    prepared request leaves DISAGG_GENERATION_INIT, and the failure is reported
    only once all receives have settled. No asynchronous status poll runs."""
    h = CoordinatorHarness()
    failing, following = _gen_init(h, 1), _gen_init(h, 2)
    h.transceiver.script_sync_recv(failing, "error")
    receives_before_fail = []
    fail = h.effects.fail_requests

    def fail_and_snapshot(*args, **kwargs):
        receives_before_fail.append(list(h.transceiver.call_log))
        fail(*args, **kwargs)

    h.effects.fail_requests = fail_and_snapshot

    h.coordinator.receive_gen_init([failing, following])

    assert h.transceiver.call_log == ["request_and_receive_sync:1", "request_and_receive_sync:2"]
    assert receives_before_fail == [h.transceiver.call_log]
    assert failing.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert following.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    assert h.effects.failed == [(_GEN_ERROR_MSG, [failing], False)]


def test_sync_receive_failure_under_multi_rank_adp_waits_for_the_synced_vote(sync_mode) -> None:
    """A synchronous receive enters no collective and, under multi-rank ADP,
    reports nothing rank-locally; the failed receive is applied by the next
    synced pass, where every rank votes."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = [
        CoordinatorHarness(dist=group.rank(rank), enable_attention_dp=True) for rank in range(2)
    ]
    failing, peer = _gen_init(ranks[0], 1), _gen_init(ranks[1], 1)
    ranks[0].transceiver.script_sync_recv(failing, "error")
    admitted = {0: [failing], 1: [peer]}

    group.run(lambda rank: ranks[rank].coordinator.receive_gen_init(admitted[rank]))

    assert [h.dist.calls for h in ranks] == [[], []]
    assert [h.effects.failed for h in ranks] == [[], []]
    assert failing.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert peer.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE

    group.run(lambda rank: ranks[rank].coordinator.handle_errors_synced())

    assert ranks[0].effects.failed == [(_VOTE_MSG, [failing], False)]
    assert ranks[1].effects.failed == [(_VOTE_MSG, [peer], False)]


def test_gen_only_benchmark_marks_requests_complete_without_a_transceiver_call(
    gen_only_benchmark,
) -> None:
    """Without a context worker there is nothing to receive: the requests are
    marked transmission-complete right after their resources are prepared."""
    h = CoordinatorHarness()
    admitted = [_gen_init(h, 1), _gen_init(h, 2)]

    h.coordinator.receive_gen_init(admitted)

    assert [req.state for req in admitted] == [LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE] * 2
    assert h.transceiver.call_log == []
    assert h.effects.prepared == [admitted]
    assert h.effects.failed == []
