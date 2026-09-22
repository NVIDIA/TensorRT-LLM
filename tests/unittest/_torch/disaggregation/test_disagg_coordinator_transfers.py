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
from coordinator_harness import CoordinatorHarness as _Harness
from coordinator_harness import TransferRequest as _Request
from fake_dist import FakeDistGroup

from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_module
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only


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
    """The transfer ends (blocks unpinned) and the still-active request is
    handed to the executor's error path as a context failure; the reap itself
    terminates nothing."""
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
    assert h.effects.failed == [("Error in kv cache transfer for context requests", [req], False)]


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
    h.active.extend([remote, user])
    h.registry.canceled = [2]
    h.transceiver.request_and_receive_async(remote)
    h.transceiver.request_and_receive_async(user)
    h.transceiver.cancel_recv_remotely(remote)
    h.transceiver.cancel_recv_remotely(user)

    h.coordinator.reap_gen_receives(0)

    assert remote.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert user.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert h.effects.failed == [
        ("Error in kv cache transfer for generation requests", [remote], False)
    ]


def test_gen_reap_leaves_error_handling_to_the_consensus_path_under_inflight_cancel(
    inflight_cancel,
) -> None:
    h = _Harness(supports_inflight_cancellation=True)
    h.active.append(
        _Request(1, is_context_only_request=False, state=LlmRequestState.DISAGG_TRANS_ERROR)
    )

    h.coordinator.reap_gen_receives(0)

    assert h.effects.failed == []


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
    h.active.append(error_req)
    h.dist.tp_allgather.return_value = [[], []]  # no timed-out receives on any rank

    h.dist.tp_allreduce.return_value = 0
    h.coordinator.poll_gen_transfers()
    assert h.effects.failed == []

    h.dist.tp_allreduce.return_value = 1
    h.coordinator.poll_gen_transfers()
    assert h.effects.failed == [
        ("Error in kv cache transfer for generation requests", [error_req], False)
    ]


# -- settling cancellations and hangs ----------------------------------------
#
# The cases above stop where the coordinator asks for something (a cancel, a
# retry) or keeps waiting. These follow each scenario through to its end: the
# transceiver finally reports, the executor cleans up, and nothing is
# released, failed or cancelled a second time.

_GEN_ERROR = "Error in kv cache transfer for generation requests"
_CTX_ERROR = "Error in kv cache transfer for context requests"


def _keep_session_on_cancel(h: _Harness, *, refuse_first: int = 0) -> list:
    """Model in-flight cancellation: ``cancel_request`` is accepted after
    ``refuse_first`` refusals, but the session stays owned by the transceiver
    until a later status poll reports it. Tests script that report with
    ``cancel_recv_remotely`` / ``finish_send``. Returns the attempt log."""
    attempts = []

    def cancel(request):
        attempts.append(request.py_request_id)
        return len(attempts) > refuse_first

    h.transceiver.cancel_request = cancel
    return attempts


def _executor_cleans_up(h: _Harness, req: _Request) -> None:
    """What the executor's error path does after ``fail_requests``: the request
    leaves active_requests and the coordinator drops its bookkeeping."""
    h.active.remove(req)
    h.coordinator.forget_request(req.py_request_id)


def test_refused_cancellation_is_settled_once_after_the_retry_is_accepted(
    inflight_cancel, clock
) -> None:
    """A refusal leaves the receive owned and untouched; the accepted retry is
    not repeated; the request fails exactly once when the transceiver reports
    the cancelled session, and nothing happens to it after that."""
    h = _Harness(kv_transfer_timeout_ms=1000, supports_inflight_cancellation=True)
    req = _receiving(h, 1, started_at=clock["t"])
    clock["t"] += 2.0
    attempts = _keep_session_on_cancel(h, refuse_first=1)

    h.coordinator.poll_gen_transfers()  # refused
    assert attempts == [1]
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert not h.transceiver.check_gen_transfer_complete()
    assert h.effects.history == []

    h.coordinator.poll_gen_transfers()  # accepted; the session is still owned
    h.coordinator.poll_gen_transfers()  # nothing left to ask for
    assert attempts == [1, 1]
    assert not h.transceiver.check_gen_transfer_complete()
    assert h.effects.history == []

    h.transceiver.cancel_recv_remotely(req)
    h.coordinator.poll_gen_transfers()
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert h.transceiver.check_gen_transfer_complete()
    assert h.effects.failed == [(_GEN_ERROR, [req], False)]

    _executor_cleans_up(h, req)
    h.coordinator.poll_gen_transfers()
    assert attempts == [1, 1]
    assert h.effects.history == [("fail", _GEN_ERROR)]


@pytest.mark.parametrize("outcome", ["error", "complete"])
def test_timed_out_send_is_settled_once_when_the_transceiver_finally_reports(
    inflight_cancel, clock, outcome
) -> None:
    """After the in-flight cancel the send stays owned until the transceiver
    reports. A late failure releases the blocks once and fails the request
    once; a completion that beat the cancel releases once and terminates
    normally, with no error."""
    h = _Harness(kv_transfer_timeout_ms=1000, supports_inflight_cancellation=True)
    req = _Request(1)
    h.active.append(req)
    h.send(req)
    clock["t"] += 2.0
    h.coordinator.check_transfer_timeouts()
    attempts = _keep_session_on_cancel(h)
    h.coordinator.reap_context_sends(0)  # cancels once, ownership kept
    assert attempts == [1]
    assert h.in_transfer(req)
    h.kv_cache_manager.unpin_blocks_by_id.assert_not_called()

    h.transceiver.finish_send(req, outcome=outcome)
    h.coordinator.reap_context_sends(0)

    assert not h.in_transfer(req)
    h.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(1)
    assert attempts == [1]
    if outcome == "error":
        assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
        assert h.effects.history == [("fail", _CTX_ERROR)]
        assert h.effects.failed == [(_CTX_ERROR, [req], False)]
        _executor_cleans_up(h, req)
    else:
        assert req.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
        assert h.effects.history == [("terminate", req)]
        assert h.active == []

    h.coordinator.reap_context_sends(0)
    assert attempts == [1]
    assert len(h.effects.history) == 1


def test_mirrored_timeout_cancels_and_fails_each_replica_once_across_ranks(
    inflight_cancel, clock
) -> None:
    """Under TP the peer's timeout is mirrored: both ranks cancel their replica
    once, keep polling in lockstep while the transceiver still owns it, fail
    it once when the cancelled session is reported, and stay quiet once the
    executor cleaned up. Both ranks enter the same collectives throughout."""
    group = FakeDistGroup(world_size=2, tp_size=2)
    ranks = [
        _Harness(
            kv_transfer_timeout_ms=1000, supports_inflight_cancellation=True, dist=group.rank(rank)
        )
        for rank in range(2)
    ]
    expired = _receiving(ranks[0], 1, started_at=clock["t"])
    fresh = _receiving(ranks[1], 1, started_at=clock["t"] + 1.5)
    attempts = [_keep_session_on_cancel(h) for h in ranks]
    clock["t"] += 2.0

    def poll(rank):
        ranks[rank].coordinator.poll_gen_transfers()

    group.run(poll)  # rank 0 expired; rank 1 mirrors the decision
    group.run(poll)  # nothing to repeat
    assert attempts == [[1], [1]]
    assert expired.py_kv_transfer_timed_out and fresh.py_kv_transfer_timed_out
    assert [h.effects.history for h in ranks] == [[], []]

    for h, req in zip(ranks, (expired, fresh)):
        h.transceiver.cancel_recv_remotely(req)
    group.run(poll)
    assert [h.effects.failed for h in ranks] == [
        [(_GEN_ERROR, [expired], False)],
        [(_GEN_ERROR, [fresh], False)],
    ]

    for h, req in zip(ranks, (expired, fresh)):
        _executor_cleans_up(h, req)
    group.run(poll)
    assert attempts == [[1], [1]]
    assert [len(h.effects.history) for h in ranks] == [1, 1]
    collectives = [[name for name, _ in h.dist.calls] for h in ranks]
    assert collectives[0] == collectives[1]
    assert collectives[0].count("tp_allgather") == 1  # the id union, once


@pytest.mark.parametrize("direction", ["context_send", "generation_receive"])
def test_a_hanging_transfer_keeps_its_resources_until_the_transceiver_settles(
    direction,
) -> None:
    """A transfer that reports nothing for many polls is left exactly as it
    is: still owned, blocks still pinned, no error, no termination. Holding on
    is the correct outcome of a hang, not releasing. Once the transceiver
    settles it, the request is released exactly once."""
    h = _Harness()
    if direction == "context_send":
        req = _Request(1)
        h.active.append(req)
        h.send(req)
        pending_state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

        def poll():
            h.coordinator.reap_context_sends(0)

        def settle():
            h.transceiver.finish_send(req)
    else:
        req = _receiving(h, 1, started_at=None)
        pending_state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        poll = h.coordinator.poll_gen_transfers

        def settle():
            h.transceiver.finish_recv(req)

    for _ in range(3):
        poll()

    assert req.state == pending_state
    assert h.active == [req]
    assert h.effects.history == []
    h.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
    assert "cancel_request:1" not in h.transceiver.call_log
    if direction == "context_send":
        assert h.in_transfer(req)
    else:
        assert not h.transceiver.check_gen_transfer_complete()

    settle()
    poll()

    if direction == "context_send":
        assert not h.in_transfer(req)
        h.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(1)
        assert h.effects.history == [("terminate", req)]
        assert h.active == []
    else:
        assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        assert h.transceiver.check_gen_transfer_complete()
        assert h.effects.history == []


def test_consecutive_sends_release_every_pinned_block_exactly_once() -> None:
    """Context requests hold their blocks only while a send is in flight. With
    several sends settling out of order across polls, one of them failing,
    every block is unpinned exactly once and nothing stays in transfer, so the
    capacity a context worker lent to transfers comes back in full."""
    h = _Harness()
    first, second, third, fourth = requests = [_Request(rid) for rid in (1, 2, 3, 4)]
    h.active.extend(requests)
    h.send(*requests)
    assert sorted(h.transfers.requests_in_transfer()) == [1, 2, 3, 4]

    h.transceiver.finish_send(third)
    h.coordinator.reap_context_sends(0)
    h.transceiver.finish_send(first, outcome="error")
    h.transceiver.finish_send(fourth)
    h.coordinator.reap_context_sends(0)
    _executor_cleans_up(h, first)
    h.transceiver.finish_send(second)
    h.coordinator.reap_context_sends(0)

    assert h.transfers.requests_in_transfer() == {}
    unpinned = sorted(call.args[0] for call in h.kv_cache_manager.unpin_blocks_by_id.call_args_list)
    assert unpinned == [1, 2, 3, 4]
    assert h.effects.terminated == [third, fourth, second]
    assert h.effects.failed == [(_CTX_ERROR, [first], False)]
    assert h.active == []


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
