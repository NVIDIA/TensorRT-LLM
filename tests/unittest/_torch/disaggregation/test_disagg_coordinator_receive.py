# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generation side of a KV transfer.

Receive start: ``receive_gen_init`` prepares executor resources for the
admitted gen-init requests and starts their KV receive in the transfer mode
the environment selects. Receive completion: the executor's
``_prepare_disagg_gen_transmission_complete`` prepares the batch, then asks the
coordinator to finish each completed receive before initialising the request
for generation; a received request that the context server's first token
already ends (by the sampler's stop criteria) is closed out before scheduling
by ``_close_out_gen_receives_at_first_token``.

Single-rank cases run one ``CoordinatorHarness``; the multi-rank case runs one
per rank over a ``FakeDistGroup``.
"""

import copy
import datetime
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from _torch.pyexecutor.llm_request_factory import make_llm_request
from coordinator_harness import CoordinatorHarness, TransferRequest
from fake_dist import FakeDistGroup

from tensorrt_llm._torch.disaggregation.orchestration.coordinator import (
    DisaggTransferCoordinator,
    NoopDisaggCoordinator,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.py_executor import BatchState, PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import partition_context_for_helix
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings import executor as trtllm
from tensorrt_llm.bindings.executor import FinishReason, IterationStats, RequestStage
from tensorrt_llm.disaggregated_params import DisaggregatedParams

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


def _cp_dist(
    peer_failed_ids, *, enable_attention_dp=False, tp_size=1, pp_size=1
) -> SimpleNamespace:
    """Dist stub for CP rank 0 of N = len(peer_failed_ids) + 1 CP ranks.

    ``peer_failed_ids`` holds one id list per CP peer. ``cp_allgather`` records
    each payload and returns it followed by those lists. Under attention DP,
    ``world_size`` is tp x N x pp so errors wait for the synced vote, and
    ``tp_allgather`` gathers this rank alone; otherwise ``world_size`` is 1 and
    errors fail at once."""

    def cp_allgather(ids, *, small_payload=False):
        dist.gathered.append(list(ids))
        return [ids, *peer_failed_ids]

    cp_size = len(peer_failed_ids) + 1
    world_size = tp_size * cp_size * pp_size if enable_attention_dp else 1
    dist = SimpleNamespace(
        rank=0,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        world_size=world_size,
        gathered=[],
    )
    dist.cp_allgather = cp_allgather
    dist.tp_allgather = lambda obj: [obj]
    return dist


def test_sync_receive_that_failed_on_a_cp_peer_fails_here_too(sync_mode) -> None:
    """A blocking receive settles on its own rank. A request this CP rank
    received but a CP peer failed is failed here as well, so the helix CP ranks
    keep decoding the same batch; the rest keep their outcome."""
    dist = _cp_dist([[2]])
    h = CoordinatorHarness(dist=dist)
    first, second = _gen_init(h, 1), _gen_init(h, 2)

    h.coordinator.receive_gen_init([first, second])

    assert dist.gathered == [[]]
    assert first.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    assert second.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert h.effects.failed == [(_GEN_ERROR_MSG, [second], False)]


def test_sync_receive_failure_here_is_shared_with_cp_peers(sync_mode) -> None:
    dist = _cp_dist([[]])
    h = CoordinatorHarness(dist=dist)
    failing, received = _gen_init(h, 1), _gen_init(h, 2)
    h.transceiver.script_sync_recv(failing, "error")

    h.coordinator.receive_gen_init([failing, received])

    assert dist.gathered == [[1]]
    assert received.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    assert h.effects.failed == [(_GEN_ERROR_MSG, [failing], False)]


def test_sync_receive_failed_on_a_cp_peer_under_adp_reaches_the_synced_vote(sync_mode) -> None:
    """Under attention DP the error waits for the synced pass, whose vote spans
    only the TP group; the CP gather already put it in error on this CP rank."""
    dist = _cp_dist([[1]], enable_attention_dp=True)
    h = CoordinatorHarness(dist=dist, enable_attention_dp=True)
    req = _gen_init(h, 1)

    h.coordinator.receive_gen_init([req])

    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert h.effects.failed == []

    h.coordinator.handle_errors_synced()

    assert h.effects.failed == [(_VOTE_MSG, [req], False)]


@pytest.mark.parametrize(
    "tp_size,pp_size,gathers",
    [
        pytest.param(2, 2, False, id="tp2_pp2_skips"),
        pytest.param(1, 2, True, id="tp1_pp2_gathers"),
        pytest.param(2, 1, True, id="tp2_pp1_gathers"),
    ],
)
def test_sync_receive_failure_gather_under_attention_dp_follows_pp_scheduling(
    sync_mode, tp_size, pp_size, gathers
) -> None:
    """Under attention DP with tp_size > 1 the PP loop schedules on each first-stage
    rank, so the CP ranks may hold different requests and cannot share one gather;
    it is skipped there. Every other layout gives the CP ranks the same requests."""
    dist = _cp_dist([[1]], enable_attention_dp=True, tp_size=tp_size, pp_size=pp_size)
    h = CoordinatorHarness(dist=dist, enable_attention_dp=True)
    req = _gen_init(h, 1)

    h.coordinator.receive_gen_init([req])

    assert dist.gathered == ([[]] if gathers else [])
    assert req.state == (
        LlmRequestState.DISAGG_TRANS_ERROR
        if gathers
        else LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    )


def test_sync_receive_without_cp_does_not_gather(sync_mode) -> None:
    dist = _cp_dist([])
    dist.cp_allgather = Mock(side_effect=AssertionError("no CP peer to gather with"))
    h = CoordinatorHarness(dist=dist)
    req = _gen_init(h, 1)

    h.coordinator.receive_gen_init([req])

    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE


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


# -- deferred receive publication ----------------------------------------------


class _ScriptedEvent:
    """Readiness event stub whose ``query`` answers come from a script."""

    def __init__(self, *answers: bool) -> None:
        self._answers = list(answers)
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]

    def synchronize(self) -> None:
        raise AssertionError("a deferred receive fence must be polled, never waited on")


def _deferring_harness(fences, **kwargs) -> CoordinatorHarness:
    """Harness whose coordinator defers receives behind per-request fences.

    ``fences`` maps request id to the event the cache manager hands over;
    requests absent from it get ``None``, i.e. no fence to wait on.
    """
    h = CoordinatorHarness(defer_unready_gen_receives=True, **kwargs)
    taken = []

    def take(req):
        taken.append(req.py_request_id)
        return fences.get(req.py_request_id)

    h.kv_cache_manager.take_disagg_receive_ready = take
    h.taken_fences = taken
    return h


def test_deferred_receive_starts_only_after_its_fence_is_ready(clock) -> None:
    """An unready receive is held, not started; the request still leaves
    DISAGG_GENERATION_INIT so it is not admitted again. The receive starts on
    the first poll that observes the fence ready, and the transfer timer is
    stamped then rather than at admission."""
    fence = _ScriptedEvent(False, True)
    h = _deferring_harness({1: fence}, kv_transfer_timeout_ms=1000)
    req = _gen_init(h, 1)

    h.coordinator.receive_gen_init([req])

    assert h.taken_fences == [1]
    assert h.transceiver.call_log == ["check_gen_transfer_status:0"]
    assert req.state == _RECEIVING
    assert req.py_kv_transfer_start_time is None

    clock["t"] += 5.0
    h.coordinator.poll_gen_transfers()

    assert h.transceiver.call_log == [
        "check_gen_transfer_status:0",
        "request_and_receive_async:1",
        "check_gen_transfer_status:0",
    ]
    assert fence.queries == 2
    assert req.state == _RECEIVING
    assert req.py_kv_transfer_start_time == clock["t"]

    h.coordinator.poll_gen_transfers()

    assert h.transceiver.call_log.count("request_and_receive_async:1") == 1
    assert fence.queries == 2
    assert h.effects.failed == []


def test_deferred_receives_publish_independently_and_in_admission_order() -> None:
    """A ready or fence-less request starts at admission while an unready one
    waits; each starts once, in admission order, as its own fence completes."""
    late = _ScriptedEvent(False, True)
    h = _deferring_harness({1: _ScriptedEvent(True), 2: late})
    first, second, unfenced = _gen_init(h, 1), _gen_init(h, 2), _gen_init(h, 3)

    h.coordinator.receive_gen_init([first, second, unfenced])

    assert h.transceiver.call_log == [
        "request_and_receive_async:1",
        "request_and_receive_async:3",
        "check_gen_transfer_status:0",
    ]

    h.coordinator.poll_gen_transfers()

    assert h.transceiver.call_log[3:] == [
        "request_and_receive_async:2",
        "check_gen_transfer_status:0",
    ]
    assert [req.state for req in (first, second, unfenced)] == [_RECEIVING] * 3


def test_deferred_receives_are_reported_from_admission_until_published() -> None:
    """The transceiver learns the waiting count at admission, before any
    receive starts, and again after each publication pass: its idle-poll skip
    counts a waiting receive as a session on every rank."""
    late = _ScriptedEvent(False, False, True)
    h = _deferring_harness({1: _ScriptedEvent(True), 2: late})
    first, second = _gen_init(h, 1), _gen_init(h, 2)

    h.coordinator.receive_gen_init([first, second])

    # admission (both waiting), then the publication pass that started 1
    assert h.transceiver.deferred_receive_reports == [2, 1]
    assert h.transceiver.call_log[0] == "request_and_receive_async:1"

    h.coordinator.poll_gen_transfers()
    assert h.transceiver.deferred_receive_reports == [2, 1, 1]
    h.coordinator.poll_gen_transfers()
    assert h.transceiver.deferred_receive_reports == [2, 1, 1, 0]
    assert h.transceiver.call_log.count("request_and_receive_async:2") == 1


def test_forgotten_deferred_receive_is_never_published() -> None:
    """A request freed while its receive waits is dropped from the queue, so
    a later ready fence does not start a receive into freed blocks."""
    fence = _ScriptedEvent(False)
    h = _deferring_harness({1: fence})
    req = _gen_init(h, 1)
    h.coordinator.receive_gen_init([req])

    h.coordinator.forget_request(1)
    fence._answers = [True]
    h.coordinator.poll_gen_transfers()

    assert "request_and_receive_async:1" not in h.transceiver.call_log
    assert fence.queries == 1


@pytest.mark.parametrize(
    "terminal_state",
    [LlmRequestState.DISAGG_TRANS_ERROR, LlmRequestState.GENERATION_COMPLETE],
)
def test_deferred_receive_of_a_terminated_request_is_dropped_unpublished(
    terminal_state,
) -> None:
    """A request that failed or finished while its receive waited is removed
    on the next poll without querying its fence or starting the receive. It
    stays active, as it does until the executor's error or completion pass
    retires it, so the drop is decided by the state alone."""
    fence = _ScriptedEvent(False)
    h = _deferring_harness({1: fence})
    req = _gen_init(h, 1)
    h.coordinator.receive_gen_init([req])
    req.state = terminal_state
    fence._answers = [True]

    h.coordinator.poll_gen_transfers()
    h.coordinator.poll_gen_transfers()

    assert "request_and_receive_async:1" not in h.transceiver.call_log
    assert fence.queries == 1
    assert req.state == terminal_state


@pytest.mark.parametrize(
    "mode_env",
    [
        pytest.param({"TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP": "1"}, id="sync"),
        pytest.param({"TRTLLM_DISAGG_BENCHMARK_GEN_ONLY": "1"}, id="gen_only_benchmark"),
    ],
)
def test_deferral_takes_no_fence_outside_async_transfer_mode(monkeypatch, mode_env) -> None:
    """Deferral applies only to asynchronous receives: in the other modes the
    fence stays with the cache manager, which waits on it itself."""
    for name, value in mode_env.items():
        monkeypatch.setenv(name, value)
    h = _deferring_harness({1: _ScriptedEvent(False)})

    h.coordinator.receive_gen_init([_gen_init(h, 1)])

    assert h.taken_fences == []


def test_without_deferral_no_fence_is_taken_and_receives_start_at_admission() -> None:
    """The default coordinator leaves the fence to the cache manager and
    starts every receive at admission."""
    h = CoordinatorHarness()
    h.kv_cache_manager.take_disagg_receive_ready = Mock()
    req = _gen_init(h, 1)

    h.coordinator.receive_gen_init([req])

    h.kv_cache_manager.take_disagg_receive_ready.assert_not_called()
    assert h.transceiver.call_log[0] == "request_and_receive_async:1"


# -- receive completion -------------------------------------------------------

_COMPLETE = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
_RECEIVING = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS


class _GenRequest(TransferRequest):
    """Generation request stub with the fields the receive tail reads and
    writes; ``events`` (if given) records the first-token writes in order."""

    def __init__(self, rid: int, events=None, **overrides) -> None:
        defaults = dict(
            is_context_only_request=False,
            state=_COMPLETE,
            prompt_len=8,
            context_current_position=0,
            decoding_iter=0,
            py_decoding_iter=0,
            py_draft_tokens=None,
            py_beam_width=1,
            py_seq_slot=None,
            context_phase_params=SimpleNamespace(first_gen_tokens=[40 + rid], draft_tokens=None),
        )
        defaults.update(overrides)
        super().__init__(rid, **defaults)
        self.events = events
        self.new_tokens = []

    @property
    def is_disagg_generation_transmission_complete(self) -> bool:
        # Derived from ``state`` so a request terminated mid-way (e.g. by the
        # executor error path) stops reporting itself as complete.
        return self.state == _COMPLETE

    def add_new_token(self, token: int, beam: int) -> None:
        self.new_tokens.append((token, beam))
        if self.events is not None:
            self.events.append(("token", self.py_request_id))


def _batch(*generation_requests) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.generation_requests = list(generation_requests)
    return batch


def test_completed_gen_receives_reports_only_completed_requests_in_batch_order() -> None:
    """The query picks the transmission-complete generation requests, keeps
    the batch order and changes nothing."""
    h = CoordinatorHarness()
    done_a, done_b = _GenRequest(1), _GenRequest(3)
    receiving = _GenRequest(2, state=_RECEIVING)
    running = _GenRequest(4, state=LlmRequestState.GENERATION_IN_PROGRESS)

    completed = h.coordinator.completed_gen_receives(_batch(done_a, receiving, done_b, running))

    assert [req.py_request_id for req in completed] == [1, 3]
    assert [req.state for req in (done_a, receiving, done_b, running)] == [
        _COMPLETE,
        _RECEIVING,
        _COMPLETE,
        LlmRequestState.GENERATION_IN_PROGRESS,
    ]
    assert h.transceiver.call_log == []


def test_try_finish_gen_receive_hands_a_completed_request_to_generation() -> None:
    """The request leaves the transfer state, its blocks are committed for
    reuse with the position already at the prompt end (the fake transceiver
    asserts that precondition), and the transfer timer fields are cleared."""
    h = CoordinatorHarness()
    req = _GenRequest(1, py_kv_transfer_start_time=5.0, py_kv_transfer_timed_out=True)

    assert h.coordinator.try_finish_gen_receive(req) is True

    assert req.state == LlmRequestState.GENERATION_IN_PROGRESS
    assert req.context_current_position == req.prompt_len
    assert h.transceiver.call_log == ["commit_blocks_for_reuse:1"]
    assert req.py_kv_transfer_start_time is None
    assert req.py_kv_transfer_timed_out is False


@pytest.mark.parametrize(
    "state",
    [
        LlmRequestState.DISAGG_GENERATION_INIT,
        _RECEIVING,
        LlmRequestState.DISAGG_TRANS_ERROR,
        LlmRequestState.GENERATION_COMPLETE,
    ],
)
def test_try_finish_gen_receive_declines_requests_that_did_not_complete(state) -> None:
    """A request that is not transmission-complete is left exactly as it is:
    no state or position change, no reuse commit, timers untouched."""
    h = CoordinatorHarness()
    req = _GenRequest(1, state=state, py_kv_transfer_start_time=5.0, py_kv_transfer_timed_out=True)

    assert h.coordinator.try_finish_gen_receive(req) is False

    assert req.state == state
    assert req.context_current_position == 0
    assert h.transceiver.call_log == []
    assert (req.py_kv_transfer_start_time, req.py_kv_transfer_timed_out) == (5.0, True)


def _executor_over(coordinator: DisaggTransferCoordinator, events: list) -> PyExecutor:
    """Bare executor whose receive tail runs for real over ``coordinator``.

    Seq-slot preparation, sampler setup, each finish attempt and the
    first-token logprobs step append to ``events`` in call order; the two
    batch-level steps record the request states they observed.
    """
    executor = object.__new__(PyExecutor)
    executor._disagg_coordinator = coordinator

    def record_batch(name):
        def _record(batch):
            events.append((name, [req.state for req in batch.context_requests_last_chunk]))

        return _record

    executor.resource_manager = SimpleNamespace(
        resource_managers={
            ResourceManagerType.SEQ_SLOT_MANAGER: SimpleNamespace(
                prepare_resources=record_batch("seq_slot")
            )
        }
    )
    executor.sampler = SimpleNamespace(setup_sampler_step=record_batch("sampler"))
    executor.model_engine = SimpleNamespace()
    executor.draft_model_engine = None
    executor.enable_spec_decode = False
    executor._handle_errors = Mock()

    finish = coordinator.try_finish_gen_receive

    def finish_and_record(req):
        finished = finish(req)
        events.append(("finish", req.py_request_id, finished))
        return finished

    coordinator.try_finish_gen_receive = finish_and_record

    prepend = executor._maybe_prepend_logprobs_and_logits

    def prepend_and_record(req, beam_width):
        events.append(("logprobs", req.py_request_id))
        prepend(req, beam_width)

    executor._maybe_prepend_logprobs_and_logits = prepend_and_record
    return executor


def test_receive_tail_prepares_the_batch_before_finishing_each_request() -> None:
    """Order contract: seq slots and the sampler are prepared for the whole
    batch while its requests are still transmission-complete; only then is
    each completed request finished and initialised for generation, in batch
    order, while a request still receiving is declined and left alone."""
    h = CoordinatorHarness()
    events = []
    executor = _executor_over(h.coordinator, events)
    first, second = _GenRequest(1, events=events), _GenRequest(2, events=events)
    receiving = _GenRequest(3, events=events, state=_RECEIVING)

    executor._prepare_disagg_gen_transmission_complete(_batch(first, receiving, second))

    assert events == [
        ("seq_slot", [_COMPLETE, _COMPLETE]),
        ("sampler", [_COMPLETE, _COMPLETE]),
        ("finish", 1, True),
        ("token", 1),
        ("logprobs", 1),
        ("finish", 3, False),
        ("finish", 2, True),
        ("token", 2),
        ("logprobs", 2),
    ]
    for req, token in ((first, 41), (second, 42)):
        assert req.state == LlmRequestState.GENERATION_IN_PROGRESS
        assert (req.decoding_iter, req.py_decoding_iter) == (1, 1)
        assert req.py_draft_tokens == []
        assert req.new_tokens == [(token, 0)]
    assert receiving.state == _RECEIVING
    assert receiving.new_tokens == []
    assert h.transceiver.call_log == ["commit_blocks_for_reuse:1", "commit_blocks_for_reuse:2"]
    executor._handle_errors.assert_not_called()


@pytest.mark.parametrize("coordinator_kind", ["real", "noop"])
def test_receive_tail_does_nothing_without_a_completed_receive(coordinator_kind) -> None:
    """With nothing to finish the batch-level preparation is skipped and no
    request is asked; the no-op coordinator reports nothing to finish even
    for a request that claims to be complete."""
    events = []
    if coordinator_kind == "real":
        coordinator = CoordinatorHarness().coordinator
        request = _GenRequest(1, events=events, state=_RECEIVING)
    else:
        coordinator = NoopDisaggCoordinator()
        request = _GenRequest(1, events=events)
    executor = _executor_over(coordinator, events)

    executor._prepare_disagg_gen_transmission_complete(_batch(request))

    assert events == []
    assert request.new_tokens == []
    assert (request.decoding_iter, request.context_current_position) == (0, 0)


def test_receive_tail_does_not_finish_requests_that_sampler_setup_failed() -> None:
    """When sampler setup raises, the executor error path fails and terminates
    the batch's requests; they must not be finished afterwards: no reuse
    commit, no first token, and their terminal state stays. The real
    ``_setup_sampler_step`` runs; only ``_handle_errors`` is replaced, with
    the state change the real one makes."""
    h = CoordinatorHarness()
    events = []
    executor = _executor_over(h.coordinator, events)
    first, second = _GenRequest(1, events=events), _GenRequest(2, events=events)
    executor.active_requests = [first, second]

    def raise_in_sampler(_requests):
        raise RuntimeError("sampler setup failed")

    executor.sampler = SimpleNamespace(setup_sampler_step=raise_in_sampler)

    def terminate_all(error_msg, **_kwargs):
        events.append(("handle_errors", error_msg))
        for req in executor.active_requests:
            req.state = LlmRequestState.GENERATION_COMPLETE
        executor.active_requests = []

    executor._handle_errors = Mock(side_effect=terminate_all)

    executor._prepare_disagg_gen_transmission_complete(_batch(first, second))

    assert events == [
        ("seq_slot", [_COMPLETE, _COMPLETE]),
        ("handle_errors", "sampler setup failed"),
        ("finish", 1, False),
        ("finish", 2, False),
    ]
    assert [req.state for req in (first, second)] == [LlmRequestState.GENERATION_COMPLETE] * 2
    assert h.transceiver.call_log == []
    assert [req.new_tokens for req in (first, second)] == [[], []]
    assert [req.decoding_iter for req in (first, second)] == [0, 0]


# -- first-token close-out ----------------------------------------------------

_PROMPT = [3, 5, 7, 9, 11, 13, 15, 17]  # prompt_len of _GenRequest


class _ClosingRequest(_GenRequest):
    """A received generation request with the fields the first-token close-out
    reads; the first token of request ``rid`` is ``40 + rid``. ``prompt`` is
    the part of the prompt this rank holds (a Helix CP rank holds a slice)."""

    def __init__(self, rid: int, max_new_tokens: int = 1, prompt=_PROMPT, **overrides) -> None:
        defaults = dict(
            prompt_len=len(prompt),
            total_input_len_cp=len(prompt),
            py_max_new_tokens=max_new_tokens,
            py_orig_prompt_len=len(prompt),
            py_end_id=-1,
            py_stop_words_list=None,
            return_perf_metrics=False,
            response=SimpleNamespace(result=SimpleNamespace()),
            cached_tokens=5,
            py_per_pos_drafted=None,
            py_per_pos_accepted=None,
            py_total_draft_tokens=0,
            py_total_accepted_draft_tokens=0,
        )
        defaults.update(overrides)
        super().__init__(rid, **defaults)
        self.prompt = list(prompt)
        self.finished_reason = None

    def get_num_tokens(self, beam: int) -> int:
        return len(self.get_tokens(beam))

    def get_tokens(self, beam: int) -> list:
        return self.prompt + [token for token, _ in self.new_tokens]

    def finish_by(self, reason, beam: int) -> None:
        self.state = LlmRequestState.GENERATION_COMPLETE
        self.finished_reason = (reason, beam)


def _closing_executor(
    coordinator: DisaggTransferCoordinator, active, attention_dp=False, max_seq_len=64, cp_size=1
) -> PyExecutor:
    """Bare executor over ``coordinator`` whose response and termination sinks are mocks."""
    executor = object.__new__(PyExecutor)
    executor._disagg_coordinator = coordinator
    executor.active_requests = list(active)
    executor.enable_attention_dp = attention_dp
    executor.dist = SimpleNamespace(rank=0, cp_size=cp_size)
    executor.canceled_req_ids = []
    executor.enable_iter_perf_stats = False
    executor._is_kv_manager_v2 = False
    executor.is_benchmark_disagg = False
    executor.iter_counter = 7
    executor.perf_manager = Mock()
    executor.sampler = SimpleNamespace(max_seq_len=max_seq_len)
    executor.max_seq_len = max_seq_len
    executor.should_exclude_last_generation_logits = False
    executor.resource_manager = SimpleNamespace(resource_managers={})
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    return executor


def test_close_out_ends_a_single_token_request_at_its_first_token() -> None:
    """A received request with max_new_tokens 1 gets the context server's first
    token and finishes by LENGTH before scheduling: its blocks are committed for
    reuse, one terminal response carrying the context-side cached tokens is
    buffered with its termination for the synchronized flush, and it leaves
    active_requests, so no decode step appends a second token. A longer request,
    one still receiving, a beam-search request and a remote-tail destination
    keep the regular path untouched."""
    h = CoordinatorHarness()
    single = _ClosingRequest(1)
    longer = _ClosingRequest(2, max_new_tokens=8)
    receiving = _ClosingRequest(3, state=_RECEIVING)
    beams = _ClosingRequest(4, py_beam_width=2)
    remote_tail = _ClosingRequest(5, py_csa2_remote_tail_mode="destination")
    untouched = (longer, receiving, beams, remote_tail)
    executor = _closing_executor(h.coordinator, [single, *untouched])

    executor._close_out_gen_receives_at_first_token()

    assert single.new_tokens == [(41, 0)]
    assert single.finished_reason == (FinishReason.LENGTH, 0)
    assert single.state_at_response_creation == LlmRequestState.GENERATION_COMPLETE
    assert (single.decoding_iter, single.py_decoding_iter) == (1, 1)
    assert single.py_draft_tokens == []
    assert single.response.result.cached_tokens == 5
    assert executor._pending_transfer_responses == [(1, single.response)]
    assert executor._pending_response_terminations == [single]
    executor._enqueue_responses.assert_not_called()
    executor._terminate_request.assert_not_called()
    assert h.transceiver.call_log == ["commit_blocks_for_reuse:1"]
    assert executor.active_requests == list(untouched)
    assert [req.state for req in untouched] == [_COMPLETE, _RECEIVING, _COMPLETE, _COMPLETE]
    assert all(req.new_tokens == [] and req.finished_reason is None for req in untouched)


@pytest.mark.parametrize(
    ("overrides", "max_seq_len", "reason"),
    [
        pytest.param({"max_new_tokens": 8}, 9, FinishReason.LENGTH, id="fills_max_seq_len"),
        pytest.param(
            {"max_new_tokens": 8, "py_stop_words_list": [[99], [41]]},
            64,
            FinishReason.STOP_WORDS,
            id="single_token_stop_word",
        ),
        pytest.param(
            {"max_new_tokens": 8, "py_stop_words_list": [[99], [17, 41]]},
            64,
            FinishReason.STOP_WORDS,
            id="stop_word_ending_the_prompt",
        ),
        pytest.param({"max_new_tokens": 8, "py_end_id": 41}, 64, FinishReason.END_ID, id="end_id"),
        pytest.param(
            {"max_new_tokens": 1, "py_stop_words_list": [[41]]},
            64,
            FinishReason.LENGTH,
            id="length_before_stop_words",
        ),
    ],
)
def test_close_out_applies_the_sampler_stop_criteria_in_order(
    overrides, max_seq_len, reason
) -> None:
    """The first token ends the request under the checks and order of
    handle_stop_criteria (END_ID, LENGTH including max_seq_len, STOP_WORDS,
    with stop words matched across the prompt tail), so the result equals
    aggregated serving's."""
    h = CoordinatorHarness()
    req = _ClosingRequest(1, **overrides)
    executor = _closing_executor(h.coordinator, [req], max_seq_len=max_seq_len)

    executor._close_out_gen_receives_at_first_token()

    assert req.finished_reason == (reason, 0)
    assert req.new_tokens == [(41, 0)]
    assert executor.active_requests == []
    assert executor._pending_response_terminations == [req]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"max_new_tokens": 2}, id="limit_not_reached"),
        pytest.param(
            {"max_new_tokens": 8, "py_stop_words_list": [[42], [15, 41]]},
            id="no_stop_word_ends_at_the_token",
        ),
        pytest.param({"max_new_tokens": 8, "py_end_id": 2}, id="other_end_id"),
    ],
)
def test_close_out_leaves_requests_the_first_token_does_not_end(overrides) -> None:
    """Without a stop the request stays on the regular path: no reuse commit,
    no first token, nothing buffered and no timestamp taken."""
    h = CoordinatorHarness()
    req = _ClosingRequest(1, **overrides)
    executor = _closing_executor(h.coordinator, [req])

    executor._close_out_gen_receives_at_first_token()

    assert (req.state, req.new_tokens, req.finished_reason) == (_COMPLETE, [], None)
    assert executor.active_requests == [req]
    assert executor._pending_transfer_responses == []
    assert h.transceiver.call_log == []
    executor.perf_manager.get_timestamp.assert_not_called()


def test_close_out_leaves_parked_failures_and_cancellations_to_their_paths() -> None:
    """A request with a parked transfer failure is left for the parked-failure
    sweep and a cancelled one for the cancellation path, so neither reaches the
    client as a success."""
    h = CoordinatorHarness()
    parked = _ClosingRequest(1)
    cancelled = _ClosingRequest(2)
    executor = _closing_executor(h.coordinator, [parked, cancelled])
    executor._transfer_held_failures = {id(parked): (parked, _GEN_ERROR_MSG)}
    executor.canceled_req_ids = [2]

    executor._close_out_gen_receives_at_first_token()

    assert [req.new_tokens for req in (parked, cancelled)] == [[], []]
    assert executor.active_requests == [parked, cancelled]
    assert executor._pending_transfer_responses == []
    assert h.transceiver.call_log == []


def test_close_out_does_not_depend_on_the_rank_local_timeout_flag() -> None:
    """Each rank checks the transfer deadline on its own clock, so a completed
    receive can be flagged timed out on one rank only. Both ranks close it out
    alike, as the regular receive path schedules it on both, and finishing the
    receive clears the flag."""
    requests = []
    for timed_out in (True, False):
        h = CoordinatorHarness()
        req = _ClosingRequest(1, py_kv_transfer_timed_out=timed_out)
        executor = _closing_executor(h.coordinator, [req])

        executor._close_out_gen_receives_at_first_token()

        assert executor.active_requests == []
        requests.append(req)

    assert [req.finished_reason for req in requests] == [(FinishReason.LENGTH, 0)] * 2
    assert [req.py_kv_transfer_timed_out for req in requests] == [False, False]


@pytest.mark.parametrize("capacity_only", [False, True])
def test_close_out_leaves_requests_to_a_connector_that_moves_kv(capacity_only) -> None:
    """A connector that moves KV is handed a finished request's blocks on the
    batch path only, so its requests stay on that path; a capacity-only
    connector moves nothing and does not stop the close-out."""
    h = CoordinatorHarness()
    req = _ClosingRequest(1)
    executor = _closing_executor(h.coordinator, [req])
    executor.kv_connector_manager = SimpleNamespace(capacity_only=capacity_only)

    executor._close_out_gen_receives_at_first_token()

    if capacity_only:
        assert req.finished_reason == (FinishReason.LENGTH, 0)
        assert executor.active_requests == []
    else:
        assert (req.state, req.new_tokens, req.finished_reason) == (_COMPLETE, [], None)
        assert executor.active_requests == [req]
        assert h.transceiver.call_log == []


_HELIX_CP_SIZE = 2
_HELIX_TOKENS_PER_BLOCK = 4


@pytest.mark.parametrize(
    "prompt_len",
    [
        # Blocks 0 and 2 (6 tokens, including the prompt's tail) on rank 0, block 1 on rank 1.
        pytest.param(10, id="uneven_slices"),
        # One block, all on rank 0; rank 1 holds no token.
        pytest.param(3, id="empty_rank"),
    ],
)
@pytest.mark.parametrize(
    ("max_new_tokens", "max_seq_len_margin", "stop_words", "reason"),
    [
        pytest.param(1, 8, None, FinishReason.LENGTH, id="max_new_tokens"),
        pytest.param(8, 1, None, FinishReason.LENGTH, id="full_sequence_reaches_max_seq_len"),
        pytest.param(8, 2, None, None, id="full_sequence_below_max_seq_len"),
        pytest.param(8, 8, "multi_token", None, id="multi_token_stop_word"),
        pytest.param(8, 8, "mixed", FinishReason.STOP_WORDS, id="single_token_word_in_mixed_list"),
    ],
)
def test_close_out_decides_alike_on_every_helix_rank(
    prompt_len, max_new_tokens, max_seq_len_margin, stop_words, reason
) -> None:
    """Under Helix each CP rank holds its own slice of the prompt, possibly an
    empty one, and every rank must decide the same way or one would schedule
    a request another closed out. The length limit is checked on the full
    sequence; multi-token stop words, which need the prompt's tail, are left to
    the decode steps, single-token ones are still checked, and the request's
    stop words are not modified."""
    prompt = list(range(100, 100 + prompt_len))
    words = {
        None: None,
        "multi_token": [[prompt[-1], 41]],
        "mixed": [[prompt[-1], 41], [41]],
    }[stop_words]
    slice_lens = []
    outcomes = []
    for cp_rank in range(_HELIX_CP_SIZE):
        local_prompt, _, full_len, _ = partition_context_for_helix(
            np.array(prompt), cp_rank, _HELIX_CP_SIZE, _HELIX_TOKENS_PER_BLOCK
        )
        slice_lens.append(len(local_prompt))
        req = _ClosingRequest(
            1,
            max_new_tokens,
            prompt=local_prompt,
            total_input_len_cp=full_len,
            py_stop_words_list=copy.deepcopy(words),
        )
        executor = _closing_executor(
            CoordinatorHarness().coordinator,
            [req],
            max_seq_len=prompt_len + max_seq_len_margin,
            cp_size=_HELIX_CP_SIZE,
        )

        executor._close_out_gen_receives_at_first_token()

        outcomes.append((req.finished_reason, req.py_stop_words_list))

    assert len(set(slice_lens)) == _HELIX_CP_SIZE, "slices must differ for this case"
    expected_reason = None if reason is None else (reason, 0)
    assert outcomes == [(expected_reason, words)] * _HELIX_CP_SIZE


def _native_received_request(rid: int, **request_options) -> LlmRequest:
    """C++-backed generation-only request whose KV receive has completed; the
    context server's first token is 100."""
    request = make_llm_request(
        rid, trtllm.RequestType.REQUEST_TYPE_GENERATION_ONLY, **request_options
    )
    request.state = _COMPLETE
    return request


def _flushing_executor(coordinator: DisaggTransferCoordinator, active) -> PyExecutor:
    """``_closing_executor`` whose synchronized flush enqueues responses and
    terminates requests for real; only the sampler and resource release are mocks."""
    executor = _closing_executor(coordinator, active)
    del executor._enqueue_responses
    del executor._terminate_request
    executor.dist = SimpleNamespace(rank=0, cp_size=1, mapping=SimpleNamespace(tp_group=[0]))
    executor.gather_all_responses = False
    executor.response_cv = threading.Condition()
    executor.responses = {}
    executor.result_wait_queues = {}
    executor._maybe_record_hang_diagnostic_phase = Mock()
    executor._disagg_pp_termination_handler = None
    executor._prefetched_request_ids = set()
    executor.sampler = SimpleNamespace(max_seq_len=64, release_request=Mock())
    executor.resource_manager = SimpleNamespace(resource_managers={}, free_resources=Mock())
    executor.model_engine = SimpleNamespace()
    executor.draft_model_engine = None
    return executor


@pytest.mark.parametrize(
    ("max_tokens", "stop_words", "reason"),
    [
        pytest.param(1, None, FinishReason.LENGTH, id="max_tokens_1"),
        pytest.param(8, [[100]], FinishReason.STOP_WORDS, id="stop_word"),
    ],
)
def test_close_out_of_a_native_request_responds_and_releases_once(
    max_tokens, stop_words, reason
) -> None:
    """On the C++-backed request the client gets exactly one final response
    whose only generated token is the context server's first token (100),
    with the finish reason aggregated serving reports, and the request's
    resources are released exactly once."""
    request = _native_received_request(
        1,
        max_tokens=max_tokens,
        stop_words=stop_words,
        output_config=trtllm.OutputConfig(exclude_input_from_output=True),
    )
    h = CoordinatorHarness()
    executor = _flushing_executor(h.coordinator, [request])

    executor._close_out_gen_receives_at_first_token()
    executor._flush_pending_transfer_responses()
    executor._flush_pending_transfer_responses()

    (response,) = executor.responses[1]
    assert response.result.is_final
    result = response.result.get_result()
    assert result.output_token_ids == [[100]]
    assert result.finish_reasons == [reason]
    assert executor.active_requests == []
    assert h.transceiver.call_log == ["commit_blocks_for_reuse:1"]
    executor.sampler.release_request.assert_called_once_with(request)
    executor.resource_manager.free_resources.assert_called_once_with(request)


def _closed_out_response(request: LlmRequest):
    """The single response a native ``request`` gets when its first token ends it."""
    executor = _flushing_executor(CoordinatorHarness().coordinator, [request])
    executor._close_out_gen_receives_at_first_token()
    executor._flush_pending_transfer_responses()
    (response,) = executor.responses[request.py_request_id]
    return response


def test_close_out_reports_when_the_request_was_scheduled() -> None:
    """A scheduled request gets its first scheduled time at seq slot
    assignment; a request finished at its first token gets one too, so serving
    can compute its queue and inference times."""
    request = _native_received_request(
        1, max_tokens=1, output_config=trtllm.OutputConfig(return_perf_metrics=True)
    )

    response = _closed_out_response(request)

    timing = response.result.get_result().request_perf_metrics.timing_metrics
    assert timing.first_scheduled_time > datetime.timedelta(0)


def test_close_out_returns_the_first_token_logits_without_a_forward() -> None:
    """Non-streaming generation logits are moved to host storage after a
    forward; with none following, the first token's logits prepended from the
    context server still reach the client."""
    first_gen_logits = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    request = _native_received_request(
        1, max_tokens=1, output_config=trtllm.OutputConfig(return_generation_logits=True)
    )
    request.py_disaggregated_params = DisaggregatedParams(first_gen_logits=[first_gen_logits])

    response = _closed_out_response(request)

    assert torch.equal(response.result.generation_logits.reshape(-1), first_gen_logits.reshape(-1))


def test_close_out_commits_kv_cache_stats_before_the_response() -> None:
    """KV cache manager V2 commits a request's allocation stats from its
    scheduled batch; a request finished at its first token is never scheduled,
    so they are committed before its response, which reports them."""
    h = CoordinatorHarness()
    events = []
    req = _ClosingRequest(1)
    executor = _closing_executor(h.coordinator, [req])
    executor._is_kv_manager_v2 = True
    executor.kv_cache_manager = SimpleNamespace(
        commit_scheduled_kv_cache_stats=lambda batch: events.append(
            ("commit_stats", [request.py_request_id for request in batch.all_requests()])
        )
    )
    create_response = executor._create_first_token_terminal_response

    def create_and_record(request, batch_token_time):
        events.append(("response", request.py_request_id))
        return create_response(request, batch_token_time)

    executor._create_first_token_terminal_response = create_and_record

    executor._close_out_gen_receives_at_first_token()

    assert events == [("commit_stats", [1]), ("response", 1)]


def test_first_token_finishes_count_as_benchmark_fill_progress() -> None:
    """In benchmark-disagg fill, a completed receive whose first token ends
    the request leaves active_requests before the fill gate looks for
    completed receives; it still counts as progress, once."""
    h = CoordinatorHarness()
    executor = _closing_executor(h.coordinator, [_ClosingRequest(1)])
    executor.is_benchmark_disagg = True
    executor._benchmark_first_token_finished_ids = set()
    executor._benchmark_completed_gen_transfer_ids = set()

    executor._close_out_gen_receives_at_first_token()

    assert executor.active_requests == []
    assert executor._benchmark_gen_transfer_made_progress() is True
    assert executor._benchmark_gen_transfer_made_progress() is False


def _stats_executor(coordinator: DisaggTransferCoordinator, active) -> PyExecutor:
    """``_closing_executor`` with iteration and request stats on, sampling every
    iteration; each record reaches ``_append_iter_stats``, a mock."""
    executor = _closing_executor(coordinator, active)
    executor.enable_iter_perf_stats = True
    executor.enable_iter_req_stats = True
    executor.iter_perf_stats_interval = 1
    executor._pending_num_new_active_requests = 0
    executor._pending_num_completed_requests = 0
    executor._close_outs_pending_iter_stats = False
    executor._pending_close_out_requests = []
    executor._latest_host_step_time_ms = None
    executor._latest_prev_device_step_time_ms = None
    executor.executor_request_queue = SimpleNamespace(
        get_request_queue=lambda: SimpleNamespace(queue=[])
    )

    def update_iter_stats(stats, _iter_latency_ms, num_completed_requests, *_args):
        stats.num_completed_requests = num_completed_requests
        return stats

    executor._update_iter_stats = update_iter_stats
    executor._append_iter_stats = Mock()
    return executor


def _recorded(executor: PyExecutor) -> list:
    """(completed count, final requestStats entries) of every record, the
    entries as (request id, generated tokens)."""
    return [
        (
            call.args[0].num_completed_requests,
            sorted(
                (stat.id, stat.num_generated_tokens)
                for stat in call.args[1]
                if stat.stage == RequestStage.GENERATION_COMPLETE
            ),
        )
        for call in executor._append_iter_stats.call_args_list
    ]


def test_close_outs_are_reported_once_when_no_batch_follows() -> None:
    """Requests closed out at their first token run no batch, so the iteration
    that closed them out records them: the completed count and each request's
    final requestStats entry, exactly once even with no further request."""
    h = CoordinatorHarness()
    requests = [_native_received_request(rid, max_tokens=1) for rid in (1, 2)]
    executor = _stats_executor(h.coordinator, requests)

    executor._close_out_gen_receives_at_first_token()
    # The closing iteration, then an idle one.
    for _ in range(2):
        executor._process_close_out_iter_stats(IterationStats(), 0.0)

    assert _recorded(executor) == [(2, [(1, 1), (2, 1)])]


def test_close_outs_ride_on_a_batch_record_without_double_counting() -> None:
    """A batch record of the same iteration carries the close-out next to the
    batch's own finished request, each counted once and each with one final
    requestStats entry; no separate record follows."""
    h = CoordinatorHarness()
    executor = _stats_executor(h.coordinator, [_native_received_request(1, max_tokens=1)])
    batch_finished = make_llm_request(3, trtllm.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION)

    executor._close_out_gen_receives_at_first_token()
    executor._process_iter_stats(
        [batch_finished],
        executor.active_requests,
        BatchState(
            scheduled_requests=ScheduledRequests(),
            sample_state=Mock(),
            iter_start_time=0.0,
            iter_stats=IterationStats(),
            iter_id=7,
        ),
    )
    executor._process_close_out_iter_stats(IterationStats(), 0.0)

    assert _recorded(executor) == [(2, [(1, 1), (3, 0)])]


def test_close_out_rebuilds_derived_cache_state_before_committing_blocks() -> None:
    """As in the regular receive tail, the cache state the generation side
    derives from the imported KV is rebuilt before the blocks enter the reuse
    tree, where another request could read them."""
    h = CoordinatorHarness()
    events = []
    commit = h.transceiver.commit_blocks_for_reuse

    def commit_and_record(req):
        events.append(("commit", req.py_request_id))
        commit(req)

    h.transceiver.commit_blocks_for_reuse = commit_and_record
    req = _ClosingRequest(1)
    executor = _closing_executor(h.coordinator, [req])
    kv_cache_manager = SimpleNamespace(
        rebuild_disagg_imported_cache=lambda requests: events.append(
            ("rebuild", [request.py_request_id for request in requests])
        )
    )
    executor.resource_manager = SimpleNamespace(
        resource_managers={ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager}
    )

    executor._close_out_gen_receives_at_first_token()

    assert events == [("rebuild", [1]), ("commit", 1)]


def test_close_out_response_carries_the_prepended_first_token_logits() -> None:
    """The terminal response is also the first-token response, so the logits
    prepended from the context server are snapshotted past
    exclude_last_generation_logits, as the first-token response path does."""
    h = CoordinatorHarness()
    logits = object()
    req = _ClosingRequest(1, py_result=SimpleNamespace(get_latest_logits_unexcluded=lambda: logits))
    executor = _closing_executor(h.coordinator, [req])
    executor._has_prepended_logits = lambda _req: True

    executor._close_out_gen_receives_at_first_token()

    assert req.response.result.generation_logits is logits


def test_close_out_under_attention_dp_waits_for_the_synchronized_flush() -> None:
    """Under attention DP the response and the termination are buffered for
    _flush_pending_transfer_responses instead of entering a rank-local gather."""
    h = CoordinatorHarness()
    req = _ClosingRequest(1)
    executor = _closing_executor(h.coordinator, [req], attention_dp=True)

    executor._close_out_gen_receives_at_first_token()

    executor._enqueue_responses.assert_not_called()
    executor._terminate_request.assert_not_called()
    assert executor._pending_transfer_responses == [(1, req.response)]
    assert executor._pending_response_terminations == [req]
    assert executor.active_requests == []
