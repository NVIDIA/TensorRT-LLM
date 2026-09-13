# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the coordinator's send / reap / timeout / cancel paths.

The coordinator runs against the contract fake transceiver, a real
``AsyncTransferManager`` and a stateful fake of the executor effects, so every
assertion is about request state, transfer ownership and what the executor
was asked to do -- not about which internal method was called.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fake_executor_effects import FakeExecutorEffects, FakeRequestRegistry
from fake_kv_cache_transceiver import FakeKvCacheTransceiver

from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_module
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only


class _Request(SimpleNamespace):
    """Request stub with the attributes the transfer paths read."""

    def __init__(self, rid: int, **overrides) -> None:
        defaults = dict(
            py_request_id=rid,
            request_id=rid,
            parent_request_id=None,
            is_child=False,
            state=LlmRequestState.CONTEXT_INIT,
            is_context_only_request=True,
            is_context_finished=True,
            is_finished_due_to_length=False,
            is_finished_due_to_cancellation=False,
            is_disagg_generation_init_state=False,
            is_disagg_generation_transmission_in_progress=False,
            py_kv_transfer_start_time=None,
            py_kv_transfer_timed_out=False,
            py_disaggregated_params=None,
            cached_tokens=0,
            response=None,
        )
        defaults.update(overrides)
        super().__init__(**defaults)
        self.state_at_response_creation = None

    def create_response(self, _use_fast_logits, _rank):
        self.state_at_response_creation = self.state
        return self.response

    @property
    def is_generation_only_request(self) -> bool:
        return not self.is_context_only_request


class _Harness:
    def __init__(
        self,
        *,
        kv_transfer_timeout_ms=None,
        supports_inflight_cancellation=False,
        enable_attention_dp=False,
        world_size=1,
        tp_size=1,
        force_terminate_ctx_for_partial_reuse=False,
        draft_kv_cache_manager=None,
    ) -> None:
        self.transceiver = FakeKvCacheTransceiver(
            kv_transfer_timeout_ms=kv_transfer_timeout_ms,
            supports_inflight_cancellation=supports_inflight_cancellation,
        )
        self.transceiver.has_retired_send_session = lambda req: False
        self.kv_cache_manager = Mock(spec=["store_blocks_for_reuse", "unpin_blocks_by_id"])
        self.kv_cache_manager.store_blocks_for_reuse.side_effect = lambda req, _: req.py_request_id
        resource_manager = SimpleNamespace(
            resource_managers={ResourceManagerType.KV_CACHE_MANAGER: self.kv_cache_manager}
        )
        self.transfers = AsyncTransferManager(resource_manager)
        self.active = []
        self.registry = FakeRequestRegistry(self.active)
        self.effects = FakeExecutorEffects()
        self.dist = Mock(rank=0, tp_size=tp_size, world_size=world_size)
        self.delegates = Mock()
        self.delegates.requests_in_error_state.return_value = []
        self.coordinator = DisaggTransferCoordinator(
            transceiver=self.transceiver,
            transfer_manager=self.transfers,
            kv_cache_manager=self.kv_cache_manager,
            dist=self.dist,
            effects=self.effects,
            registry=self.registry,
            enable_attention_dp=enable_attention_dp,
            force_terminate_ctx_for_partial_reuse=force_terminate_ctx_for_partial_reuse,
            delegates=self.delegates,
            draft_kv_cache_manager=draft_kv_cache_manager,
        )

    def send(self, *requests: _Request) -> None:
        self.coordinator.send_completed_context(list(requests))

    def in_transfer(self, req: _Request) -> bool:
        return req.py_request_id in self.transfers.requests_in_transfer()


@pytest.fixture
def inflight_cancel(monkeypatch):
    monkeypatch.setattr(coordinator_module, "is_disagg_inflight_cancel_enabled", lambda: True)


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr(coordinator_module.time, "monotonic", lambda: now["t"])
    return now


# -- sending -----------------------------------------------------------------


def test_send_pins_blocks_before_the_final_slice_leaves() -> None:
    """The slice must not be sent before the request's blocks are committed to
    the reuse tree and pinned; otherwise the peer could read blocks that get
    recycled under it."""
    h = _Harness()
    pinned_at_send = []
    original_send = h.transceiver.respond_and_send_async

    def send_and_record(req):
        pinned_at_send.append(h.in_transfer(req))
        original_send(req)

    h.transceiver.respond_and_send_async = send_and_record
    req = _Request(1)

    h.send(req)

    assert pinned_at_send == [True]
    assert req.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS


def test_send_stamps_the_timeout_clock_only_when_a_timeout_is_configured(clock) -> None:
    timed = _Harness(kv_transfer_timeout_ms=1000)
    untimed = _Harness(kv_transfer_timeout_ms=None)
    req_timed, req_untimed = _Request(1), _Request(2)

    timed.send(req_timed)
    untimed.send(req_untimed)

    assert req_timed.py_kv_transfer_start_time == clock["t"]
    assert req_untimed.py_kv_transfer_start_time is None


def test_send_releases_the_index_slot_on_the_target_and_draft_kv_managers() -> None:
    """Forward is done once the final slice is sent, so the IndexMapper slot is
    released on every KV manager that has one; the draft manager is optional
    and a manager without index slots is skipped."""
    draft = Mock(spec=["release_index_slot"])
    h = _Harness(draft_kv_cache_manager=draft)
    h.kv_cache_manager.release_index_slot = Mock()
    req = _Request(1)

    h.send(req)

    h.kv_cache_manager.release_index_slot.assert_called_once_with(1)
    draft.release_index_slot.assert_called_once_with(1)
    assert h.in_transfer(req)


@pytest.mark.parametrize(
    ("bridge_enabled", "has_inflight", "releases_claim"),
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_bridge_rejection_releases_the_claim_only_without_a_physical_owner(
    bridge_enabled: bool, has_inflight: bool, releases_claim: bool
) -> None:
    """The FP4 MLA bridge may reject a send before any transfer session exists.
    Only then is there no physical accessor for the reap to poll, so the claim
    is released on the spot; a rejection with a live session, or a failure
    without the bridge, stays claimed until the reap and the error path."""
    h = _Harness(kv_transfer_timeout_ms=1000)
    h.transceiver._fp4_mla_bridge_enabled = bridge_enabled
    h.transceiver.has_inflight_transfer = lambda _req: has_inflight
    h.transceiver.respond_and_send_async = lambda req: setattr(
        req, "state", LlmRequestState.DISAGG_TRANS_ERROR
    )
    req = _Request(1)
    h.active.append(req)

    h.send(req)

    assert h.in_transfer(req) is not releases_claim
    assert (req.py_kv_transfer_start_time is None) is releases_claim
    assert h.active == [req]
    assert h.effects.terminated == []


def test_send_skips_requests_that_must_not_send() -> None:
    """Cancelled, unfinished, user-cancel-pending and retired-session requests
    all stay out of the transfer manager and never reach the transceiver."""
    h = _Harness()
    h.registry.canceled = [4]
    cancelled = _Request(2, is_finished_due_to_cancellation=True)
    unfinished = _Request(3, is_context_finished=False)
    cancel_pending = _Request(4)
    retired = _Request(5)
    h.transceiver.has_retired_send_session = lambda req: req is retired

    h.send(cancelled, unfinished, cancel_pending, retired)

    assert h.transfers.requests_in_transfer() == {}
    assert h.transceiver.call_log == []


# -- reaping context sends ---------------------------------------------------


def test_completed_send_releases_blocks_and_terminates_a_departed_request() -> None:
    """A request already out of active_requests (PP>1 early path) terminates
    exactly once when its send completes."""
    h = _Harness()
    req = _Request(1)
    h.send(req)
    h.transceiver.finish_send(req)

    h.coordinator.reap_context_sends(0)

    assert not h.in_transfer(req)
    h.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(1)
    assert h.effects.terminated == [req]
    assert h.effects.staged_responses == []


def test_fast_completion_of_an_active_request_stages_its_response_first() -> None:
    """A send that completes before the response pass ran: the response is
    created while the request is still TRANS_IN_PROGRESS (C++ createResult
    requires it), staged for the synchronized flush, and termination waits
    for that flush."""
    h = _Harness()
    response = SimpleNamespace(result=SimpleNamespace(cached_tokens=None, ctx_usage=None))
    req = _Request(1, response=response, cached_tokens=5)
    h.active.append(req)
    h.send(req)
    h.transceiver.finish_send(req)

    h.coordinator.reap_context_sends(0)

    assert req.state_at_response_creation == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    assert response.result.cached_tokens == 5
    assert h.effects.staged_responses == [(1, response, req)]
    assert h.effects.terminated == []
    assert h.active == []
    assert not h.in_transfer(req)


def test_fast_completion_without_a_response_terminates_immediately() -> None:
    h = _Harness()
    req = _Request(1, response=None)
    h.active.append(req)
    h.send(req)
    h.transceiver.finish_send(req)

    h.coordinator.reap_context_sends(0)

    assert h.effects.terminated == [req]
    assert h.effects.staged_responses == []
    assert h.active == []


def test_failed_send_releases_its_claim_but_leaves_the_request_to_the_error_path() -> None:
    """The transfer ends (blocks unpinned) but the request stays active so the
    rank-synchronized error pass can respond; nothing is terminated here."""
    h = _Harness()
    req = _Request(1)
    h.active.append(req)
    h.send(req)
    h.transceiver.finish_send(req, outcome="error")

    h.coordinator.reap_context_sends(0)

    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert h.active == [req]
    assert not h.in_transfer(req)
    assert h.effects.terminated == []
    assert h.effects.staged_responses == []
    h.delegates.check_transfer_errors.assert_called_once_with("context requests")


def test_failure_reported_after_release_is_kept_for_the_synced_error_pass() -> None:
    """An error id the manager no longer knows cannot be applied locally; it is
    handed to the next synchronized error pass exactly once."""
    h = _Harness()
    status = SimpleNamespace(completed_request_ids=[], error_request_ids=[9])
    h.transceiver.check_context_transfer_status = lambda at_least, mark_complete=False: status

    h.coordinator.reap_context_sends(0)

    assert h.coordinator.take_pending_context_failures() == {9}
    assert h.coordinator.take_pending_context_failures() == set()


def test_send_completion_does_not_release_a_claim_the_connector_still_holds() -> None:
    """Transceiver and KV connector share one refcount per request; the
    request is released only when the last owner lets go."""
    h = _Harness()
    req = _Request(1)
    h.send(req)
    h.transfers.start_transfer(req)  # the connector's claim
    h.transceiver.finish_send(req)

    h.coordinator.reap_context_sends(0)

    assert h.in_transfer(req)
    h.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    assert h.effects.terminated == []

    h.coordinator.release_transfer(req)  # the connector finishes

    assert not h.in_transfer(req)
    assert h.effects.terminated == [req]


def test_legacy_timeout_cancels_and_releases_a_queued_send(clock) -> None:
    """Without in-flight cancellation a timed-out send that the transceiver can
    cancel is released from the manager at once."""
    h = _Harness(kv_transfer_timeout_ms=1000)
    req = _Request(1)
    h.send(req)
    clock["t"] += 2.0
    h.coordinator.check_transfer_timeouts()
    assert req.py_kv_transfer_timed_out

    h.coordinator.reap_context_sends(0)

    assert "cancel_request:1" in h.transceiver.call_log
    assert req.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    assert req.py_kv_transfer_start_time is None
    assert not h.in_transfer(req)
    assert h.effects.terminated == [req]


def test_inflight_cancel_keeps_ownership_until_the_transceiver_reports(
    inflight_cancel, clock
) -> None:
    """With in-flight cancellation the send is cancelled once and stays owned
    by the transceiver until it reports the terminal state."""
    h = _Harness(kv_transfer_timeout_ms=1000, supports_inflight_cancellation=True)
    req = _Request(1)
    h.send(req)
    clock["t"] += 2.0
    h.coordinator.check_transfer_timeouts()

    h.coordinator.reap_context_sends(0)
    h.coordinator.reap_context_sends(0)

    assert h.transceiver.call_log.count("cancel_request:1") == 1
    assert h.in_transfer(req)
    assert req.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    assert h.effects.terminated == []


# -- reaping generation receives ---------------------------------------------


def test_remote_cancellation_of_a_receive_fails_the_request_unless_the_user_cancelled() -> None:
    h = _Harness()
    remote, user = (
        _Request(1, is_context_only_request=False),
        _Request(2, is_context_only_request=False),
    )
    h.registry.canceled = [2]
    h.transceiver.request_and_receive_async(remote)
    h.transceiver.request_and_receive_async(user)
    h.transceiver.cancel_recv_remotely(remote)
    h.transceiver.cancel_recv_remotely(user)

    h.coordinator.reap_gen_receives(0)

    assert remote.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert user.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    h.delegates.check_transfer_errors.assert_called_once_with("generation requests")


def test_gen_reap_leaves_error_handling_to_the_consensus_path_under_inflight_cancel(
    inflight_cancel,
) -> None:
    h = _Harness(supports_inflight_cancellation=True)

    h.coordinator.reap_gen_receives(0)

    h.delegates.check_transfer_errors.assert_not_called()


# -- timeouts ----------------------------------------------------------------


def test_timeout_check_flags_both_directions(clock) -> None:
    h = _Harness(kv_transfer_timeout_ms=1000)
    ctx = _Request(1)
    h.send(ctx)
    gen = _Request(
        2,
        is_context_only_request=False,
        is_disagg_generation_transmission_in_progress=True,
        py_kv_transfer_start_time=clock["t"],
    )
    fresh = _Request(
        3,
        is_context_only_request=False,
        is_disagg_generation_transmission_in_progress=True,
        py_kv_transfer_start_time=clock["t"] + 1.5,
    )
    h.active.extend([gen, fresh])
    clock["t"] += 2.0

    h.coordinator.check_transfer_timeouts()

    assert ctx.py_kv_transfer_timed_out
    assert gen.py_kv_transfer_timed_out
    assert not fresh.py_kv_transfer_timed_out


def test_post_batch_timeout_check_is_gated_on_an_inflight_context_send(clock) -> None:
    """The post-batch call sites keep their historical gate: with no context
    send in flight they do not flag anything, generation included."""
    h = _Harness(kv_transfer_timeout_ms=1000)
    gen = _Request(
        2,
        is_context_only_request=False,
        is_disagg_generation_transmission_in_progress=True,
        py_kv_transfer_start_time=clock["t"],
    )
    h.active.append(gen)
    clock["t"] += 2.0

    h.coordinator.check_transfer_timeouts(only_with_context_sends=True)
    assert not gen.py_kv_transfer_timed_out

    h.send(_Request(1))
    h.coordinator.check_transfer_timeouts(only_with_context_sends=True)
    assert gen.py_kv_transfer_timed_out


def test_timeout_check_is_inert_without_a_configured_timeout(clock) -> None:
    h = _Harness(kv_transfer_timeout_ms=None)
    req = _Request(1)
    h.send(req)
    req.py_kv_transfer_start_time = clock["t"] - 1e6

    h.coordinator.check_transfer_timeouts()

    assert not req.py_kv_transfer_timed_out


def test_timed_out_requests_fail_inline_without_multi_rank_adp() -> None:
    h = _Harness(enable_attention_dp=False, world_size=2)
    a, b = _Request(1), _Request(2)

    h.coordinator.fail_timed_out([a, b])

    assert h.effects.failed == [
        ("Request 1 timed out", [a], False),
        ("Request 2 timed out", [b], False),
    ]


def test_timed_out_requests_under_adp_fail_only_at_the_synced_drain() -> None:
    """Under multi-rank ADP the error response enters a collective, so it is
    deferred until every rank drains together; a rank with nothing pending
    still votes and follows the peer's decision."""
    h = _Harness(enable_attention_dp=True, world_size=2)
    req = _Request(1)
    h.dist.tp_allgather_int64.return_value = Mock(any=lambda: True)

    h.coordinator.fail_timed_out([req])
    assert h.effects.failed == []

    h.coordinator.handle_timeouts_synced()

    h.dist.tp_allgather_int64.assert_called_once_with([True])
    assert h.effects.failed == [("Request timed out (KV transfer)", [req], False)]

    # The buffer was drained: the next vote carries nothing.
    h.dist.tp_allgather_int64.return_value = Mock(any=lambda: False)
    h.coordinator.handle_timeouts_synced()
    assert h.dist.tp_allgather_int64.call_args_list[-1].args == ([False],)
    assert len(h.effects.failed) == 1


def test_fatal_error_during_the_synced_drain_leaves_no_stale_pending_state() -> None:
    """If the executor's error path raises (fatal), the drained requests must
    not be re-failed by the next drain."""
    h = _Harness(enable_attention_dp=True, world_size=2)
    h.dist.tp_allgather_int64.return_value = Mock(any=lambda: True)
    h.coordinator.fail_timed_out([_Request(1)])
    h.effects.fail_raises = RuntimeError("fatal")

    with pytest.raises(RuntimeError, match="fatal"):
        h.coordinator.handle_timeouts_synced()

    h.effects.fail_raises = None
    h.dist.tp_allgather_int64.return_value = Mock(any=lambda: False)
    h.coordinator.handle_timeouts_synced()
    assert len(h.effects.failed) == 1


# -- in-flight cancellation of generation receives ---------------------------


def _receiving(h: _Harness, rid: int, started_at: float) -> _Request:
    req = _Request(
        rid,
        is_context_only_request=False,
        is_disagg_generation_transmission_in_progress=True,
        py_kv_transfer_start_time=started_at,
    )
    h.active.append(req)
    h.transceiver.request_and_receive_async(req)
    return req


def test_timed_out_receive_is_cancelled_once_until_the_request_is_forgotten(
    inflight_cancel, clock
) -> None:
    """A timed-out receive is cancelled once; later polls do not re-cancel
    while the transceiver still owns it. Once the executor frees the request
    the bookkeeping is dropped so the id can be cancelled again if reused."""
    h = _Harness(kv_transfer_timeout_ms=1000, supports_inflight_cancellation=True)
    req = _receiving(h, 1, started_at=clock["t"])
    clock["t"] += 2.0

    h.coordinator.poll_gen_transfers()
    h.coordinator.poll_gen_transfers()

    assert req.py_kv_transfer_timed_out
    assert h.transceiver.call_log.count("cancel_request:1") == 1

    h.coordinator.forget_request(1)
    h.transceiver.request_and_receive_async(req)
    h.coordinator.poll_gen_transfers()
    assert h.transceiver.call_log.count("cancel_request:1") == 2


def test_refused_cancellation_is_retried_on_the_next_poll(inflight_cancel, clock) -> None:
    h = _Harness(kv_transfer_timeout_ms=1000, supports_inflight_cancellation=True)
    req = _receiving(h, 1, started_at=clock["t"])
    clock["t"] += 2.0
    attempts = []
    real_cancel = h.transceiver.cancel_request

    def refuse_first(request):
        attempts.append(request.py_request_id)
        return real_cancel(request) if len(attempts) > 1 else False

    h.transceiver.cancel_request = refuse_first

    h.coordinator.poll_gen_transfers()
    h.coordinator.poll_gen_transfers()
    h.coordinator.poll_gen_transfers()

    assert attempts == [1, 1]
    assert req.py_kv_transfer_timed_out


def test_peer_rank_timeout_decision_is_mirrored_locally(inflight_cancel, clock) -> None:
    """Under TP the timeout decision is the union over ranks: a request that
    has not expired on this rank's clock is still cancelled when a peer says
    it timed out, so later error responses enter the TP collective together."""
    h = _Harness(kv_transfer_timeout_ms=1000, supports_inflight_cancellation=True, tp_size=2)
    req = _receiving(h, 1, started_at=clock["t"])  # fresh on this rank
    h.dist.tp_allreduce.return_value = 1
    h.dist.tp_allgather.return_value = [[], [1]]

    h.coordinator.poll_gen_transfers()

    assert req.py_kv_transfer_timed_out
    assert "cancel_request:1" in h.transceiver.call_log


def test_generation_error_consensus_fails_only_when_some_rank_needs_it(inflight_cancel) -> None:
    h = _Harness(supports_inflight_cancellation=True, tp_size=2)
    error_req = _Request(1, is_context_only_request=False, state=LlmRequestState.DISAGG_TRANS_ERROR)
    h.delegates.requests_in_error_state.return_value = [error_req]
    h.dist.tp_allgather.return_value = [[], []]  # no timed-out receives on any rank

    h.dist.tp_allreduce.return_value = 0
    h.coordinator.poll_gen_transfers()
    assert h.effects.failed == []

    h.dist.tp_allreduce.return_value = 1
    h.coordinator.poll_gen_transfers()
    assert h.effects.failed == [
        ("Error in kv cache transfer for generation requests", [error_req], False)
    ]


# -- pacing ------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctx_inflight, request_kwargs, expect_sleep",
    [
        pytest.param(False, {}, False, id="nothing_pending"),
        pytest.param(True, {}, True, id="context_send_inflight"),
        pytest.param(False, {"is_disagg_generation_init_state": True}, True, id="gen_awaiting"),
        pytest.param(
            False,
            {"is_disagg_generation_transmission_in_progress": True},
            True,
            id="gen_receive_inflight",
        ),
    ],
)
def test_pace_idle_sleeps_only_when_a_transfer_can_unblock_the_loop(
    monkeypatch, ctx_inflight, request_kwargs, expect_sleep
) -> None:
    sleep = Mock()
    monkeypatch.setattr(coordinator_module.time, "sleep", sleep)
    h = _Harness()
    if ctx_inflight:
        h.send(_Request(1))
    h.active.append(_Request(2, is_context_only_request=False, **request_kwargs))

    h.coordinator.pace_idle()

    assert sleep.called is expect_sleep
