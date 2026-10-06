# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generation side of a KV transfer.

Receive start: ``receive_gen_init`` prepares executor resources for the
admitted gen-init requests and starts their KV receive in the transfer mode
the environment selects. Receive completion: the executor's
``_prepare_disagg_gen_transmission_complete`` prepares the batch, then asks the
coordinator to finish each completed receive before initialising the request
for generation.

Single-rank cases run one ``CoordinatorHarness``; the multi-rank case runs one
per rank over a ``FakeDistGroup``.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest
from fake_dist import FakeDistGroup

from tensorrt_llm._torch.disaggregation.orchestration.coordinator import (
    DisaggTransferCoordinator,
    NoopDisaggCoordinator,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests
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
    executor.model_engine = SimpleNamespace(enable_spec_decode=False)
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
