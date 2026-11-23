# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision logic of the lagged PP disagg send agreement (``pp_lagged_consensus.py``).

The rule is checked against the stock collective passes it replaces
(``_find_consensus_request_ids`` + ``_consensus_outcome``), the agreement object
is run on one thread per PP rank with a point-to-point stand-in, and the
transceiver's lagged mode is checked with stand-in sessions and a ``dist`` whose
collectives raise.
"""

import pickle
import queue
import random
import threading
from types import SimpleNamespace
from typing import Dict, List

import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.orchestration.pp_lagged_consensus import (
    PP_DISAGG_DECISION_TAG,
    PP_DISAGG_REPORT_TAG,
    SEND_CANCELLED,
    SEND_COMPLETED,
    SEND_FAILED,
    SEND_PENDING,
    PPDisaggLaggedConsensus,
    create_pp_disagg_lagged_consensus,
    decide_common,
    decide_outcomes,
    pp_disagg_lagged_consensus_applies,
)
from tensorrt_llm._torch.disaggregation.orchestration.pp_termination import (
    LaggedDisaggPPTerminationHandler,
)
from tensorrt_llm._torch.disaggregation.transceiver import (
    KvCacheTransceiverV2,
    _find_consensus_request_ids,
)
from tensorrt_llm._torch.pyexecutor.pp_utils import PPCommTag
from tensorrt_llm.bindings import LlmRequestState

TIMEOUT_S = 30.0


def test_message_tags_are_outside_the_pp_tags():
    used = {int(tag) for tag in PPCommTag}
    assert PP_DISAGG_REPORT_TAG not in used
    assert PP_DISAGG_DECISION_TAG not in used
    assert PP_DISAGG_REPORT_TAG != PP_DISAGG_DECISION_TAG


# --------------------------------------------------------------------------- the rule


def _stock_settle(per_rank: List[Dict[int, tuple]]):
    """What the stock passes settle when every rank reports its state now.

    ``per_rank[r]`` maps each send finished on rank r to ``(outcome, quiesced)``.
    Runs the real ``_find_consensus_request_ids`` and ``_consensus_outcome`` with
    an allgather that returns every rank's contribution.
    """
    pp = len(per_rank)
    to_process = _find_consensus_request_ids([list(r) for r in per_rank], pp)
    order = [rid for rid in per_rank[0] if rid in set(to_process)]
    transceiver = object.__new__(KvCacheTransceiverV2)
    packed = []
    for report in per_rank:
        packed.append(
            [
                [rid for rid in order if report[rid][0] == SEND_CANCELLED],
                [rid for rid in order if report[rid][0] == SEND_FAILED],
                [rid for rid in order if report[rid][0] == SEND_COMPLETED],
                [rid for rid in order if report[rid][1]],
            ]
        )
    cancelled, failed, completed, quiesced = transceiver._consensus_outcome(
        order,
        packed[0][0],
        packed[0][1],
        packed[0][2],
        lambda _local: packed,
        True,
        packed[0][3],
    )
    quiesced = set(quiesced)
    return (
        [rid for rid in cancelled if rid in quiesced],
        [rid for rid in failed if rid in quiesced],
        completed,
    )


@pytest.mark.parametrize("seed", range(40))
def test_rule_equals_the_stock_collective_passes(seed):
    rng = random.Random(seed)
    pp = rng.choice([2, 3, 4])
    rids = list(range(rng.randint(1, 12)))
    per_rank = []
    for _ in range(pp):
        report = {}
        for rid in rids:
            if rng.random() < 0.8:
                outcome = rng.choice(
                    [SEND_COMPLETED] * 4 + [SEND_FAILED, SEND_CANCELLED, SEND_PENDING]
                )
                report[rid] = (outcome, rng.random() < 0.8)
        per_rank.append(report)
    assert decide_outcomes(per_rank) == _stock_settle(per_rank)


def test_rule_cases():
    done = (SEND_COMPLETED, True)
    assert decide_outcomes([{1: done}, {1: done}]) == ([], [], [1])
    # Not finished on one stage: nothing is settled.
    assert decide_outcomes([{1: done}, {}]) == ([], [], [])
    # A failure anywhere fails it, once every stage has quiesced.
    assert decide_outcomes([{1: done}, {1: (SEND_FAILED, True)}]) == ([], [1], [])
    assert decide_outcomes([{1: done}, {1: (SEND_FAILED, False)}]) == ([], [], [])
    # Cancelled wins over failed.
    assert decide_outcomes([{1: (SEND_FAILED, True)}, {1: (SEND_CANCELLED, True)}]) == (
        [1],
        [],
        [],
    )
    # A stage whose final wait is pending keeps it from completing.
    assert decide_outcomes([{1: done}, {1: (SEND_PENDING, True)}]) == ([], [], [])
    # The first rank fixes the order.
    assert decide_outcomes([{3: done, 1: done}, {1: done, 3: done}]) == ([], [], [3, 1])
    assert decide_common([5, 2, 9], [[9, 2], [2, 9, 7]]) == [2, 9]
    assert decide_common([5], []) == [5]


# --------------------------------------------------------------------------- agreement on threads


class _Handle:
    def __init__(self, consumed: threading.Event):
        self._consumed = consumed

    def wait(self):
        if not self._consumed.wait(TIMEOUT_S):
            raise TimeoutError("isend never matched")


class _World:
    """Point-to-point objects between rank threads; an isend completes on receipt."""

    def __init__(self):
        self._lock = threading.Lock()
        self._queues = {}
        self.blocked_receives: Dict[int, int] = {}

    def _queue(self, src, dst, tag):
        with self._lock:
            return self._queues.setdefault((src, dst, int(tag)), queue.Queue())

    def isend(self, obj, src, dst, tag):
        consumed = threading.Event()
        self._queue(src, dst, tag).put((pickle.dumps(obj), consumed))
        return _Handle(consumed)

    def recv(self, src, dst, tag):
        q = self._queue(src, dst, tag)
        if q.empty():
            with self._lock:
                self.blocked_receives[dst] = self.blocked_receives.get(dst, 0) + 1
        payload, consumed = q.get(timeout=TIMEOUT_S)
        consumed.set()
        return pickle.loads(payload)


class _P2PDist:
    def __init__(self, world, rank, pp_size):
        self._world = world
        self.rank = rank
        self.pp_rank = rank
        self.pp_size = pp_size
        self.tp_size = 1
        self.cp_size = 1

    def isend_object(self, obj, dest, tag):
        return self._world.isend(obj, self.rank, dest, tag)

    def recv_object(self, src, tag):
        return self._world.recv(src, self.rank, tag)


class _StateTransceiver:
    """Per-rank monotone state: ready ids and finished sends appear over iterations."""

    def __init__(self, rank, schedule):
        self.rank = rank
        # schedule[(kind, rid)] = (iteration it appears on this rank, payload)
        self._schedule = schedule
        self.iteration = 0
        self.applied: List[dict] = []
        self.retired = set()
        self.promoted = set()

    def pp_lagged_local_report(self):
        ready = [
            rid
            for (kind, rid), (at, _) in self._schedule.items()
            if kind == "ready" and at <= self.iteration and rid not in self.promoted
        ]
        sends = {
            rid: outcome
            for (kind, rid), (at, outcome) in self._schedule.items()
            if kind == "send" and at <= self.iteration and rid not in self.retired
        }
        return ready, sends

    def pp_lagged_apply(self, *, ready, cancelled, failed, completed):
        for rid in cancelled + failed + completed:
            assert rid not in self.retired, f"rank {self.rank} retired {rid} twice"
            self.retired.add(rid)
        self.promoted.update(ready)
        self.applied.append(
            {"ready": ready, "cancelled": cancelled, "failed": failed, "completed": completed}
        )
        self.iteration += 1


def _run_ranks(pp_size, fn):
    errors = []

    def worker(rank):
        try:
            fn(rank)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(r,), daemon=True) for r in range(pp_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT_S * 2)
    assert not any(t.is_alive() for t in threads), "deadlock"
    assert not errors, errors


@pytest.mark.parametrize("pp_size,lag", [(2, 1), (4, 3), (4, 7), (3, 2)])
@pytest.mark.parametrize("seed", range(3))
def test_every_rank_applies_the_same_valid_decisions(pp_size, lag, seed):
    rng = random.Random(seed * 100 + pp_size * 10 + lag)
    iterations = 40
    schedules = [dict() for _ in range(pp_size)]
    for rid in range(20):
        for rank in range(pp_size):
            schedules[rank][("ready", rid)] = (rng.randint(0, 25), None)
            outcome = rng.choice([SEND_COMPLETED] * 6 + [SEND_FAILED, SEND_CANCELLED])
            schedules[rank][("send", rid)] = (rng.randint(0, 25), (outcome, rng.random() < 0.9))
    world = _World()
    transceivers = [_StateTransceiver(r, schedules[r]) for r in range(pp_size)]
    terminations = [LaggedDisaggPPTerminationHandler(lambda req: None) for _ in range(pp_size)]
    agreements = [None] * pp_size

    def rank_main(rank):
        released = []
        agreement = PPDisaggLaggedConsensus(
            _P2PDist(world, rank, pp_size),
            transceivers[rank],
            lag,
            release_outcomes=lambda: released.append(transceivers[rank].iteration),
            termination_handler=terminations[rank],
        )
        agreements[rank] = agreement
        for i in range(iterations):
            # Requests queued for termination at rank-dependent iterations.
            for rid in range(5):
                if i == rid * 3 + rank:
                    terminations[rank].terminate(SimpleNamespace(py_request_id=100 + rid))
            agreement.step()
        agreement.drain()
        assert released == list(range(1, iterations + 1))

    _run_ranks(pp_size, rank_main)
    # Identical decisions on every rank, iteration by iteration.
    for t in transceivers[1:]:
        assert t.applied == transceivers[0].applied
    for i, decision in enumerate(transceivers[0].applied):
        # Nothing is decided before every follower's first report arrived.
        if i < lag:
            assert not any(decision.values())
        # Each settled send was finished on every rank no later than the
        # report the first rank used (i - lag for followers, i for itself).
        for kind in ("cancelled", "failed", "completed"):
            for rid in decision[kind]:
                for rank in range(pp_size):
                    seen = i if rank == 0 else i - lag
                    assert schedules[rank][("send", rid)][0] <= seen
        for rid in decision["ready"]:
            for rank in range(pp_size):
                seen = i if rank == 0 else i - lag
                assert schedules[rank][("ready", rid)][0] <= seen
    # Terminations: every rank freed the same requests.
    for handler in terminations:
        assert handler.pending_ids() == terminations[0].pending_ids()
    assert agreements[0].num_terminated == 5 - len(terminations[0].pending_ids())
    # In steady state the first rank never waited for a report (all sent long before).
    # (It may block in the first ``lag`` iterations' drains; not asserted.)


def test_out_of_step_report_is_refused():
    agreement = PPDisaggLaggedConsensus(
        SimpleNamespace(pp_size=2, pp_rank=0, rank=0),
        transceiver=None,
        lag=1,
        release_outcomes=lambda: None,
    )
    with pytest.raises(RuntimeError, match="out of step"):
        agreement._accept(1, {"iteration": 3}, 2)


def test_lag_and_pp_size_are_checked():
    with pytest.raises(ValueError):
        PPDisaggLaggedConsensus(SimpleNamespace(pp_size=1, pp_rank=0), None, 1, lambda: None)
    with pytest.raises(ValueError):
        PPDisaggLaggedConsensus(SimpleNamespace(pp_size=4, pp_rank=0), None, 0, lambda: None)


# --------------------------------------------------------------------------- factory


class _EnablingTransceiver:
    def __init__(self, refusal=()):
        self.enabled = False
        self.refusal = list(refusal)

    def pp_lagged_consensus_refusal(self):
        return list(self.refusal)

    def enable_pp_lagged_consensus(self):
        self.enabled = True


def _dist(pp=4, tp=1, cp=1, rank=0):
    return SimpleNamespace(pp_size=pp, tp_size=tp, cp_size=cp, rank=rank, pp_rank=rank)


def _create(dist=None, transceiver="default", adp=False, handler=None):
    transceiver = _EnablingTransceiver() if transceiver == "default" else transceiver
    return create_pp_disagg_lagged_consensus(
        dist=dist or _dist(),
        transceiver=transceiver,
        lag=3,
        enable_attention_dp=adp,
        release_outcomes=lambda: None,
        termination_handler=handler,
    )


def test_factory_builds_only_on_a_pp_disagg_worker():
    assert _create(dist=_dist(pp=1)) is None
    assert _create(transceiver=None) is None
    transceiver = _EnablingTransceiver()
    agreement = _create(transceiver=transceiver, handler=LaggedDisaggPPTerminationHandler(id))
    assert isinstance(agreement, PPDisaggLaggedConsensus) and transceiver.enabled


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dist": _dist(tp=2)},
        {"dist": _dist(cp=2)},
        {"adp": True},
        {"transceiver": object()},
        {"transceiver": _EnablingTransceiver(refusal=["enable_pipelined_transfer off"])},
    ],
)
def test_factory_skips_uncovered_configurations(kwargs):
    """A configuration the agreement does not cover keeps the stock collectives
    (no agreement, the transceiver left as it is) instead of being refused."""
    assert _create(**kwargs) is None
    transceiver = kwargs.get("transceiver")
    if isinstance(transceiver, _EnablingTransceiver):
        assert not transceiver.enabled


def test_applies_matches_the_factory():
    assert pp_disagg_lagged_consensus_applies(
        dist=_dist(), transceiver=_EnablingTransceiver(), enable_attention_dp=False
    )
    assert not pp_disagg_lagged_consensus_applies(
        dist=_dist(pp=1), transceiver=_EnablingTransceiver(), enable_attention_dp=False
    )
    assert not pp_disagg_lagged_consensus_applies(
        dist=_dist(), transceiver=None, enable_attention_dp=False
    )


def test_factory_refuses_a_stock_termination_handler():
    with pytest.raises(TypeError, match="lagged PP termination handler"):
        _create(handler=object())


# --------------------------------------------------------------------------- transceiver lagged mode


class _RaisingDist:
    """A ``dist`` whose collectives fail the test: lagged mode must not enter any."""

    def __init__(self):
        self.collectives = []

    def _collective(self, name):
        def call(*args, **kwargs):
            self.collectives.append(name)
            raise AssertionError(f"lagged mode entered {name}")

        return call

    def __getattr__(self, name):
        if name in ("pp_allgather", "allgather", "tp_allgather", "allreduce", "broadcast"):
            return self._collective(name)
        raise AttributeError(name)


class _Session:
    def __init__(self, rid, *, status=SessionStatus.TRANSFERRING, result=None, transferring=False):
        self.disagg_request_id = rid
        self.status = status
        self._result = result
        self._transferring = transferring
        self.closed = False
        self.cancelled = False

    def is_completed(self):
        return self.status == SessionStatus.TRANSFERRED

    def has_failed(self):
        return self.status in (SessionStatus.ERROR, SessionStatus.CANCELLED)

    def wait_complete(self, blocking=False):
        assert not blocking
        return self._result

    def has_transferring_tasks(self):
        return self._transferring

    def close(self):
        self.closed = True
        return True


class _Worker:
    def __init__(self, ready=()):
        self.ready = set(ready)
        self.sweeps = 0

    def has_all_peer_req_infos_for_send(self, rid):
        return rid in self.ready

    def sweep_stale_req_infos(self):
        self.sweeps += 1


class _Req:
    def __init__(self, rid):
        self.py_request_id = rid
        self.request_id = rid
        self.py_disaggregated_params = None
        self.state = None
        self.py_kv_send_session_retired = False


def _lagged_transceiver(sessions=None, wait=None, ready=(), pipelined=False, pp=4):
    t = object.__new__(KvCacheTransceiverV2)
    t._dist = _RaisingDist()
    t._mapping = SimpleNamespace(
        pp_size=pp, tp_size=1, cp_size=1, world_size=pp, enable_attention_dp=False
    )
    t._send_sessions = dict(sessions or {})
    t._send_reqs = {rid: _Req(rid) for rid in t._send_sessions}
    t._recv_sessions = {}
    t._recv_reqs = {}
    t._wait_reqs = dict(wait or {})
    t._transfer_worker = _Worker(ready)
    t._ever_had_send_session = True
    t._ever_had_recv_session = False
    t._ctx_need_tp_sync = False
    t._ctx_need_pp_sync = pp > 1
    t._gen_need_sync = pp > 1
    t._sender_future_timeout_ms = 0
    t._enable_pipelined_transfer = pipelined
    t.kv_transfer_timeout_ms = 1000
    return t


def test_transceiver_refuses_pipelined_transfer():
    assert _lagged_transceiver().pp_lagged_consensus_refusal() == []
    assert _lagged_transceiver(pipelined=True).pp_lagged_consensus_refusal() == [
        "enable_pipelined_transfer off"
    ]
    with pytest.raises(ValueError, match="enable_pipelined_transfer"):
        _lagged_transceiver(pipelined=True).enable_pp_lagged_consensus()
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        _lagged_transceiver(pp=1).enable_pp_lagged_consensus()


def test_local_report_classifies_as_the_stock_pass():
    sessions = {
        1: _Session(1, status=SessionStatus.TRANSFERRED, result=WaitResult.COMPLETED),
        2: _Session(2, status=SessionStatus.ERROR, result=WaitResult.FAILED, transferring=True),
        3: _Session(3, status=SessionStatus.CANCELLED, result=WaitResult.FAILED),
        4: _Session(4, status=SessionStatus.TRANSFERRED, result=None),
        5: _Session(5),  # still transferring: not finished, not reported
    }
    t = _lagged_transceiver(sessions, wait={7: _Req(7), 8: _Req(8)}, ready=[8])
    t.enable_pp_lagged_consensus()
    ready, sends = t.pp_lagged_local_report()
    assert ready == [8]
    assert sends == {
        1: (SEND_COMPLETED, True),
        2: (SEND_FAILED, False),
        3: (SEND_CANCELLED, True),
        4: (SEND_PENDING, True),
    }
    assert t._dist.collectives == []


def test_apply_retires_once_and_reports_through_the_status_call():
    sessions = {
        1: _Session(1, status=SessionStatus.TRANSFERRED, result=WaitResult.COMPLETED),
        2: _Session(2, status=SessionStatus.ERROR, result=WaitResult.FAILED),
        3: _Session(3, status=SessionStatus.CANCELLED, result=WaitResult.FAILED),
    }
    waiting = {7: _Req(7)}
    t = _lagged_transceiver(sessions, wait=waiting)
    reqs = dict(t._send_reqs)
    t.enable_pp_lagged_consensus()
    # 9 is named but this rank no longer holds it (retired by a local timeout cancel).
    t.pp_lagged_apply(ready=[7], cancelled=[3], failed=[2, 9], completed=[1, 9])
    assert waiting[7].state == LlmRequestState.CONTEXT_INIT and not t._wait_reqs
    assert t._send_sessions == {} and t._send_reqs == {}
    assert all(s.closed for s in sessions.values())
    assert all(reqs[rid].py_kv_send_session_retired for rid in (1, 2, 3))
    status = t.check_context_transfer_status(0, mark_complete=True)
    assert status.completed_request_ids == [1] and status.error_request_ids == [2]
    assert reqs[1].state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    # Reported once.
    status = t.check_context_transfer_status(1)
    assert status.completed_request_ids == [] and status.error_request_ids == []
    assert t._transfer_worker.sweeps == 2
    assert t._dist.collectives == []


def test_lagged_mode_enters_no_collective_on_a_ctx():
    t = _lagged_transceiver(
        {1: _Session(1, status=SessionStatus.TRANSFERRED, result=WaitResult.COMPLETED)},
        wait={7: _Req(7)},
        ready=[7],
    )
    t.enable_pp_lagged_consensus()
    t.prepare_context_requests([])
    t.check_context_transfer_status(0)
    t.check_context_transfer_status(1)
    status = t.check_gen_transfer_status(0)
    assert list(status) == [[], [], []]
    assert t._dist.collectives == []
    # Nothing was settled without the decision.
    assert 1 in t._send_sessions and 7 in t._wait_reqs


def test_stock_mode_still_runs_the_pp_collectives():
    """Without the lagged agreement: the same calls reach the stock collectives (unchanged path)."""
    t = _lagged_transceiver(
        {1: _Session(1, status=SessionStatus.TRANSFERRED, result=WaitResult.COMPLETED)}
    )
    calls = []
    t._dist = SimpleNamespace(
        pp_allgather=lambda obj: calls.append("pp_allgather") or [obj] * 4,
        allgather=lambda obj: calls.append("allgather") or [obj] * 4,
        tp_allgather=lambda obj: calls.append("tp_allgather") or [obj],
    )
    t._gen_allgather = t._dist.allgather
    status = t.check_context_transfer_status(0)
    assert status.completed_request_ids == [1]
    assert calls == ["pp_allgather", "pp_allgather"]
    calls.clear()
    t.check_gen_transfer_status(0)
    assert calls == ["allgather", "allgather"]


def test_lagged_mode_keeps_receive_consensus_once_it_received():
    t = _lagged_transceiver()
    t.enable_pp_lagged_consensus()
    t._ever_had_recv_session = True
    t._gen_allgather = t._dist.allgather
    with pytest.raises(AssertionError, match="allgather"):
        t.check_gen_transfer_status(0)


# --------------------------------------------------------------------------- termination handler


def test_lagged_termination_handler_frees_only_decided_ids_once():
    freed = []
    handler = LaggedDisaggPPTerminationHandler(freed.append)
    a, b = SimpleNamespace(py_request_id=1), SimpleNamespace(py_request_id=2)
    handler.terminate(a)
    handler.terminate(b)
    handler.terminate(a)
    handler.terminate_pending_requests()
    assert freed == [] and handler.pending_ids() == [1, 2]
    handler.apply([2, 5])
    handler.apply([2])
    assert freed == [b] and handler.pending_ids() == [1]
