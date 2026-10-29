# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retirement of generation-side KV receives across a gen sync group.

When the gen sync group has more than one rank (any multi-rank worker without
attention DP; the PP or helix CP ranks of one DP rank under attention DP), its
ranks must retire a receive in the same iteration. A client cancel reaches every
rank in the same iteration, so each rank retires its receive at once. A transfer
timeout passes on each rank at its own time, so it only cancels that rank's
session, and ``check_gen_transfer_status`` retires the request on every rank in
the same call once each rank's session has ended.

Each rank runs a real ``KvCacheTransceiverV2`` (and, where the executor side
matters, a real ``DisaggTransferCoordinator``) on its own thread over a shared
``FakeDistGroup``. The sessions are real ``RxSession`` objects whose single
writer the test drives, so cancellation, in-flight writes and completion
behave natively.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest
from fake_dist import FakeDistGroup

from tensorrt_llm._torch.disaggregation.base import TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import GenTransferStatus
from tensorrt_llm._torch.disaggregation.native.bounce.impl import NoBounceTransport
from tensorrt_llm._torch.disaggregation.native.extent import Chunk
from tensorrt_llm._torch.disaggregation.native.transfer import AgentResult, RxSession
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.disaggregated_params import DisaggregatedParams

pytestmark = pytest.mark.cpu_only

_RID = 1
_TIMEOUT_MS = 1000
_NOTHING_SETTLED = GenTransferStatus([], [], [])
_GEN_ERROR_MSG = "Error in kv cache transfer for generation requests"
_VOTED_ERROR_MSG = "Disagg KV cache transfer error"

# Two-rank gen sync groups: TP ranks without attention DP gather over the world;
# under attention DP, the two helix CP ranks or the two PP ranks of one DP rank
# gather over CP or PP.
_TOPOLOGIES = ["tp2", "adp_helix_cp2", "adp_pp2"]


@pytest.fixture(autouse=True)
def _async_transfer_mode(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)


class _OneWriterReceiver:
    """Receiver without a network: each published destination has one writer,
    whose final result the test reports with ``_writer_reports``."""

    _enforce_physical_ownership = False

    def __init__(self) -> None:
        self._bounce = NoBounceTransport()
        self._registrar = SimpleNamespace(
            self_rank_info=SimpleNamespace(instance_name="gen", instance_rank=0)
        )
        self._session = None

    def setup_session(self, session: RxSession) -> None:
        self._session = session

    def dispatch_task(self, task) -> None:
        task.expected_transfers = 1
        self._session.mark_transferring(task.slice_id)

    def send_cancel_to_senders(self, _rid, _endpoints) -> set:
        return set()

    def clear_session(self, _rid) -> None:
        return None


def _writing_session() -> RxSession:
    """A receive session whose destination is published and being written."""
    session = RxSession(
        request_id=_RID,
        params=DisaggregatedParams(disagg_request_id=_RID),
        receiver=_OneWriterReceiver(),
    )
    session.receive(
        Chunk(
            block_ids_per_layer_groups=[],
            kind_per_layer_group=[],
            token_range=TokenRange(start=0, end=1),
            is_last=True,
        )
    )
    assert session.has_transferring_tasks()
    return session


def _writer_reports(session: RxSession, result: AgentResult) -> None:
    """The writer's final word: its write landed (SUCCESS) or stopped (FAILED)."""
    session.process_kv_agent_result(
        peer_rank=0, receiver_slice_id=0, is_last_slice=True, status=result
    )


def _transceiver(mapping, dist) -> KvCacheTransceiverV2:
    """A V2 transceiver reduced to its receive-side bookkeeping."""
    tc = object.__new__(KvCacheTransceiverV2)
    tc._mapping = mapping
    tc._dist = dist
    tc._init_sync_policy()
    tc.kv_transfer_timeout_ms = _TIMEOUT_MS
    tc._ever_had_recv_session = True
    tc._wait_reqs, tc._send_sessions, tc._send_reqs = {}, {}, {}
    tc._recv_sessions, tc._recv_reqs = {}, {}
    return tc


def _group_of_two(topology: str):
    """Return the FakeDistGroup and one transceiver per rank of the gen sync group."""
    if topology == "tp2":
        group = FakeDistGroup(world_size=2, tp_size=2)
        mapping = SimpleNamespace(
            tp_size=2, pp_size=1, cp_size=1, world_size=2, enable_attention_dp=False
        )
        dists = [group.rank(rank) for rank in range(2)]
    elif topology == "adp_helix_cp2":
        group = FakeDistGroup(world_size=2, tp_size=1)
        mapping = SimpleNamespace(
            tp_size=1, pp_size=1, cp_size=2, world_size=2, enable_attention_dp=True
        )
        # With tp_size 1 the world is the DP rank's CP group.
        dists = [
            SimpleNamespace(rank=rank, cp_allgather=group.rank(rank).allgather) for rank in range(2)
        ]
    elif topology == "adp_pp2":
        group = FakeDistGroup(world_size=2, tp_size=1)
        mapping = SimpleNamespace(
            tp_size=1, pp_size=2, cp_size=1, world_size=2, enable_attention_dp=True
        )
        # With tp_size 1 the world is the DP rank's PP group.
        dists = [
            SimpleNamespace(rank=rank, pp_allgather=group.rank(rank).allgather) for rank in range(2)
        ]
    else:
        raise ValueError(f"unknown topology {topology}")
    return group, [_transceiver(mapping, dist) for dist in dists]


def _receiving_request(started_at=None) -> TransferRequest:
    """A request whose receive is in flight, with no session registered yet."""
    return TransferRequest(
        _RID,
        is_context_only_request=False,
        state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
        is_disagg_generation_transmission_in_progress=True,
        py_kv_transfer_start_time=started_at,
        set_kv_cache_transfer_end=lambda _end: None,
        set_kv_cache_size=lambda _size: None,
    )


def _start_receive(tc: KvCacheTransceiverV2, started_at=None) -> TransferRequest:
    """Register an in-flight receive the way request_and_receive_async leaves it."""
    req = _receiving_request(started_at)
    tc._recv_reqs[_RID] = req
    tc._recv_sessions[_RID] = _writing_session()
    return req


def _check_status_on_all_ranks(group: FakeDistGroup, tcs) -> list:
    return group.run(lambda rank: tcs[rank].check_gen_transfer_status(0))


def _harnesses(group: FakeDistGroup, tcs) -> list:
    """One coordinator harness per rank, over that rank's real transceiver."""
    attention_dp = tcs[0]._mapping.enable_attention_dp
    return [
        CoordinatorHarness(dist=group.rank(rank), enable_attention_dp=attention_dp, transceiver=tc)
        for rank, tc in enumerate(tcs)
    ]


def _run_iteration(group: FakeDistGroup, harnesses) -> None:
    """One executor iteration's receive poll, then the attention-DP error vote
    that fails a request there (a no-op without attention DP)."""
    group.run(lambda rank: harnesses[rank].coordinator.poll_gen_transfers())
    group.run(lambda rank: harnesses[rank].coordinator.handle_errors_synced())


def _expected_failure_msg(tcs) -> str:
    """Without attention DP the poll fails the request; with it, the error vote does."""
    return _VOTED_ERROR_MSG if tcs[0]._mapping.enable_attention_dp else _GEN_ERROR_MSG


def _start_timed_receives(clock, group, tcs, elapsed_s) -> tuple:
    """Start one receive per rank, rank r's started ``elapsed_s[r]`` seconds ago."""
    harnesses = _harnesses(group, tcs)
    reqs = [
        _start_receive(tc, started_at=clock["t"] - elapsed) for tc, elapsed in zip(tcs, elapsed_s)
    ]
    for h, req in zip(harnesses, reqs):
        h.active.append(req)
    return harnesses, reqs


@pytest.mark.parametrize("peer_state", ["landed", "unpublished"])
@pytest.mark.parametrize("topology", _TOPOLOGIES)
def test_a_client_cancel_retires_the_receive_on_every_rank_at_once(topology, peer_state) -> None:
    """A client cancel reaches every rank of the group in the same iteration, so
    each rank retires its receive at once, whatever its session's state. Rank 0
    is still being written; rank 1's write has landed, or rank 1 has not
    published its receive yet. Waiting for rank 0's write to drain, as without
    gen sync, would leave rank 0 holding the request after rank 1 freed it."""
    group, tcs = _group_of_two(topology)
    req0 = _start_receive(tcs[0])
    session0 = tcs[0]._recv_sessions[_RID]
    if peer_state == "landed":
        req1 = _start_receive(tcs[1])
        _writer_reports(tcs[1]._recv_sessions[_RID], AgentResult.SUCCESS)
    else:
        req1 = _receiving_request()

    assert [tcs[0].cancel_request(req0), tcs[1].cancel_request(req1)] == [True, True]

    assert session0.status == SessionStatus.CANCELLED
    assert all(tc.check_gen_transfer_complete() for tc in tcs)
    assert _check_status_on_all_ranks(group, tcs) == [_NOTHING_SETTLED] * 2


@pytest.mark.parametrize("topology", _TOPOLOGIES)
def test_a_timeout_on_one_rank_fails_the_request_on_every_rank_in_one_round(
    clock, topology
) -> None:
    """Rank 0's receive started earlier and times out; rank 1's does not, and its
    write then lands. The poll cancels rank 0's session without a rank-local
    timeout flag, and the next poll fails the request as a transfer error on
    both ranks together instead of on rank 0 alone."""
    group, tcs = _group_of_two(topology)
    harnesses, reqs = _start_timed_receives(clock, group, tcs, (2.0, 0.0))
    sessions = [tc._recv_sessions[_RID] for tc in tcs]

    _run_iteration(group, harnesses)
    group.run(lambda rank: harnesses[rank].coordinator.check_transfer_timeouts())

    assert [session.status for session in sessions] == [
        SessionStatus.CANCELLED,
        SessionStatus.TRANSFERRING,
    ]
    assert [h.effects.failed for h in harnesses] == [[], []]
    assert not any(req.py_kv_transfer_timed_out for req in reqs)

    _writer_reports(sessions[1], AgentResult.SUCCESS)
    _run_iteration(group, harnesses)

    for h, req in zip(harnesses, reqs):
        assert h.effects.failed == [(_expected_failure_msg(tcs), [req], False)]
        assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
        assert not req.py_kv_transfer_timed_out
    assert all(tc.check_gen_transfer_complete() for tc in tcs)


@pytest.mark.parametrize("topology", _TOPOLOGIES)
def test_a_receive_that_landed_is_not_failed_by_its_timeout(clock, topology) -> None:
    """Both writes land before the poll, but rank 0's timeout has passed by
    then. The landed session is not cancelled, so the request completes."""
    group, tcs = _group_of_two(topology)
    harnesses, reqs = _start_timed_receives(clock, group, tcs, (2.0, 0.0))
    for tc in tcs:
        _writer_reports(tc._recv_sessions[_RID], AgentResult.SUCCESS)

    _run_iteration(group, harnesses)

    for h, req in zip(harnesses, reqs):
        assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        assert h.effects.failed == []
    assert all(tc.check_gen_transfer_complete() for tc in tcs)


@pytest.mark.parametrize("topology", _TOPOLOGIES)
def test_timed_out_receives_are_retired_even_if_their_writers_never_report(clock, topology) -> None:
    """The context side dropped the transfer, so no writer will ever report.
    Once every rank's receive has timed out, all of them fail the request in
    the same poll instead of waiting for a write that never lands. Waiting for
    in-flight writes is physical ownership's job, which no gen sync group
    enables."""
    group, tcs = _group_of_two(topology)
    harnesses, reqs = _start_timed_receives(clock, group, tcs, (2.0, 2.0))

    _run_iteration(group, harnesses)

    for h, req in zip(harnesses, reqs):
        assert h.effects.failed == [(_expected_failure_msg(tcs), [req], False)]
    assert all(tc.check_gen_transfer_complete() for tc in tcs)


@pytest.mark.parametrize("topology", _TOPOLOGIES)
def test_a_timeout_cancel_that_raises_still_lets_the_rank_join_the_consensus(
    clock, topology
) -> None:
    """Both receives have timed out, and rank 0's cancel raises before marking
    its session. Rank 0 still enters the status gather, so neither rank hangs;
    the next poll cancels rank 0's session and both ranks fail the request."""
    group, tcs = _group_of_two(topology)
    harnesses, reqs = _start_timed_receives(clock, group, tcs, (2.0, 2.0))
    sessions = [tc._recv_sessions[_RID] for tc in tcs]
    cancel = sessions[0].cancel
    attempts = []

    def cancel_raising_once():
        attempts.append(_RID)
        if len(attempts) == 1:
            raise RuntimeError("cancel failed")
        return cancel()

    sessions[0].cancel = cancel_raising_once

    _run_iteration(group, harnesses)

    assert [session.status for session in sessions] == [
        SessionStatus.TRANSFERRING,
        SessionStatus.CANCELLED,
    ]
    assert [h.effects.failed for h in harnesses] == [[], []]

    _run_iteration(group, harnesses)

    assert attempts == [_RID, _RID]
    for h, req in zip(harnesses, reqs):
        assert h.effects.failed == [(_expected_failure_msg(tcs), [req], False)]


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(dict(tp_size=1, world_size=1, enable_attention_dp=False), id="single_rank"),
        pytest.param(dict(tp_size=2, world_size=2, enable_attention_dp=True), id="adp_tp2"),
    ],
)
def test_without_a_gen_sync_group_a_rank_retires_its_own_receive(clock, layout) -> None:
    """With no gen sync group the rank-local rules stand: the transceiver does
    not cancel a timed-out receive, the timeout check flags the request for the
    executor's timeout path, and cancel_request waits for a write to drain
    before it retires the receive. No collective is entered."""
    no_peer = Mock(side_effect=AssertionError("no gen sync group to gather over"))
    dist = SimpleNamespace(rank=0, allgather=no_peer, cp_allgather=no_peer, pp_allgather=no_peer)
    tc = _transceiver(SimpleNamespace(pp_size=1, cp_size=1, **layout), dist)
    h = CoordinatorHarness(transceiver=tc)
    req = _start_receive(tc, started_at=clock["t"] - 2.0)
    h.active.append(req)
    session = tc._recv_sessions[_RID]

    assert tc.check_gen_transfer_status(0) == _NOTHING_SETTLED
    h.coordinator.check_transfer_timeouts()

    assert req.py_kv_transfer_timed_out
    assert session.status == SessionStatus.TRANSFERRING

    assert tc.cancel_request(req) is False
    _writer_reports(session, AgentResult.FAILED)
    assert tc.cancel_request(req) is True
    assert tc.check_gen_transfer_complete()
