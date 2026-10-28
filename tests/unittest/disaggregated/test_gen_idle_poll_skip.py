# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Idle receive-status skip of KvCacheTransceiverV2 on generation workers.

Several simulated ranks run ``check_gen_transfer_status`` in lockstep threads
whose allgather is a barrier: a rank that skips a collective another rank
enters breaks the barrier, so every test also checks the ranks stay in step.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Callable

import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only


class _Collective:
    """Allgather over threads; a missing participant breaks the barrier."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size, timeout=5.0)
        self.slots: list = [None] * world_size
        self.calls = [0] * world_size
        self.payloads: list = [[] for _ in range(world_size)]

    def allgather_for(self, rank: int) -> Callable:
        def allgather(payload):
            self.calls[rank] += 1
            self.payloads[rank].append(payload)
            self.slots[rank] = payload
            self.barrier.wait()
            gathered = list(self.slots)
            self.barrier.wait()
            return gathered

        return allgather


class _Session:
    def __init__(self, completed: bool = False) -> None:
        self.completed = completed
        self.status = SessionStatus.READY
        self.transfer_end_time = None
        self.kv_cache_size_bytes = 0
        self.closed = False

    def is_completed(self) -> bool:
        return self.completed

    def has_failed(self) -> bool:
        return False

    def wait_complete(self, blocking: bool = False):
        return WaitResult.COMPLETED if self.completed else None

    def kv_write_verified(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def _request(rid: int) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=rid,
        request_id=rid,
        state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
        py_disaggregated_params=None,
        set_kv_cache_transfer_start=lambda _t: None,
        set_kv_cache_transfer_end=lambda _t: None,
        set_kv_cache_size=lambda _s: None,
    )


def _make_rank(collective: _Collective, rank: int, *, skip: bool) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._ever_had_recv_session = True
    transceiver._gen_need_sync = True
    transceiver._gen_allgather = collective.allgather_for(rank)
    transceiver._mapping = SimpleNamespace(
        world_size=collective.world_size, enable_attention_dp=False, pp_size=1, cp_size=1
    )
    transceiver._dist = SimpleNamespace(rank=rank)
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._gen_idle_poll_skip = skip
    transceiver._gen_idle_agreed = False
    transceiver._assert_disagg_history_declared = lambda _req: None
    return transceiver


def _lockstep(ranks, fn: Callable) -> list:
    """Run fn(rank_index, transceiver) on every rank concurrently; re-raise any failure."""
    results: list = [None] * len(ranks)
    errors: list = []

    def run(i: int) -> None:
        try:
            results[i] = fn(i, ranks[i])
        except BaseException as error:  # noqa: BLE001 - surfaced below
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(ranks))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def _poll(ranks) -> list:
    return _lockstep(ranks, lambda _i, t: t.check_gen_transfer_status(0))


@pytest.fixture
def two_ranks():
    collective = _Collective(2)
    return collective, [_make_rank(collective, r, skip=True) for r in range(2)]


def test_skip_disabled_polls_every_round_with_unchanged_payload() -> None:
    collective = _Collective(2)
    ranks = [_make_rank(collective, r, skip=False) for r in range(2)]
    for _ in range(3):
        _poll(ranks)
    assert collective.calls == [6, 6]
    assert not any(t._gen_idle_agreed for t in ranks)
    # Outcome payload: cancelled, failed, completed, retirable, verified -- no session count.
    outcome_payloads = [p for p in collective.payloads[0] if p and isinstance(p[0], list)]
    assert outcome_payloads and all(len(p) == 5 for p in outcome_payloads)


def test_session_on_one_rank_keeps_every_rank_polling(two_ranks) -> None:
    collective, ranks = two_ranks
    # Rank 0 still holds a receive session (e.g. a cancel retired rank 1's first).
    ranks[0]._recv_sessions[7] = _Session(completed=False)
    ranks[0]._recv_reqs[7] = _request(7)
    for _ in range(3):
        _poll(ranks)
    assert collective.calls == [6, 6]
    assert not any(t._gen_idle_agreed for t in ranks)
    assert 7 in ranks[0]._recv_sessions


class _FailingFetch:
    session = None

    def fetch(self, _extent, expected_write_bytes: int) -> None:
        raise RuntimeError("no peer")


class _Fetch:
    def __init__(self) -> None:
        self.session = None

    def fetch(self, _extent, expected_write_bytes: int) -> None:
        self.session = _Session()


def _make_receiving_rank(collective: _Collective, rank: int, fetch_cls=_Fetch):
    transceiver = _make_rank(collective, rank, skip=True)
    transceiver._open_peer_source = lambda _req: fetch_cls()
    transceiver._create_cache_extent = lambda _req: SimpleNamespace(local=None)
    transceiver._chunk_num_bytes = lambda _local: 0
    transceiver._kv_size_rank_factor = 1
    return transceiver


def test_receive_request_resumes_polling_on_every_rank() -> None:
    collective = _Collective(2)
    ranks = [_make_receiving_rank(collective, r, fetch_cls=_FailingFetch) for r in range(2)]
    _poll(ranks)
    assert all(t._gen_idle_agreed for t in ranks)

    for t in ranks:
        # A receive that fails before creating its session still clears the flag.
        with pytest.raises(RuntimeError):
            t.request_and_receive_async(_request(9))
    assert not any(t._gen_idle_agreed for t in ranks)
    assert all(not t._recv_sessions for t in ranks)
    calls = list(collective.calls)
    _poll(ranks)
    assert collective.calls == [c + 2 for c in calls]
    assert all(t._gen_idle_agreed for t in ranks)


def test_divergent_skip_would_break_lockstep() -> None:
    # Negative control: a rank that skips while its peer polls breaks the barrier.
    collective = _Collective(2)
    ranks = [_make_rank(collective, r, skip=True) for r in range(2)]
    ranks[0]._gen_idle_agreed = True
    collective.barrier = threading.Barrier(2, timeout=0.5)
    with pytest.raises(threading.BrokenBarrierError):
        _poll(ranks)


def test_consensus_outcome_reports_all_idle() -> None:
    stub = SimpleNamespace(
        _union=KvCacheTransceiverV2._union,
        _intersection=KvCacheTransceiverV2._intersection,
    )

    def outcome(gathered, **kwargs):
        return KvCacheTransceiverV2._consensus_outcome(
            stub, [], [], [], [], lambda _local: gathered, True, **kwargs
        )

    idle = [[[], [], [], [0]], [[], [], [], [0]]]
    busy = [[[], [], [], [0]], [[], [], [], [2]]]
    assert outcome(idle, local_session_count=0) == ([], [], [], True)
    assert outcome(busy, local_session_count=0) == ([], [], [], False)
    # The count follows the optional quiescence / verification lists.
    full = [[[], [], [], [], [], [0]], [[], [], [], [], [], [0]]]
    assert outcome(full, locally_quiesced=[], locally_verified=[], local_session_count=0) == (
        [],
        [],
        [],
        [],
        set(),
        True,
    )
    # Without a count the arity is unchanged.
    assert outcome([[[], [], []], [[], [], []]]) == ([], [], [])


@pytest.mark.parametrize(
    "adp,pp,cp,need_sync,expected",
    [
        (False, 1, 1, True, True),
        (True, 1, 1, True, False),
        (False, 2, 1, True, False),
        (False, 1, 2, True, False),
        (False, 1, 1, False, False),
    ],
)
def test_skip_applies_only_to_replicated_tp(adp, pp, cp, need_sync, expected) -> None:
    mapping = SimpleNamespace(enable_attention_dp=adp, pp_size=pp, cp_size=cp)
    assert KvCacheTransceiverV2._gen_idle_poll_skip_applies(mapping, need_sync) is expected


def test_staggered_transfers_keep_lockstep_and_agreed_flag() -> None:
    """4 ranks with staggered transfers stay in lockstep; the idle flag follows an all-idle round.

    Shards finish in different rounds and receives arrive mid-flight and in the round the last session
    retires; completions and flags stay identical, and the flag is set only after no rank held a session.
    """
    world = 4
    collective = _Collective(world)
    ranks = [_make_receiving_rank(collective, r) for r in range(world)]

    def round_(admit=(), finish=None):
        # finish: {rid: [ranks whose shard completes in this round]}
        for rid, done_ranks in (finish or {}).items():
            for r in done_ranks:
                ranks[r]._recv_sessions[rid].completed = True

        def step(_i, t):
            for rid in admit:
                t.request_and_receive_async(_request(rid))
            return t.check_gen_transfer_status(0)

        statuses = _lockstep(ranks, step)
        completed = [sorted(s[0]) for s in statuses]
        assert all(c == completed[0] for c in completed), completed
        flags = {t._gen_idle_agreed for t in ranks}
        assert len(flags) == 1, flags
        if any(t._recv_sessions for t in ranks):
            assert flags == {False}
        assert len(set(collective.calls)) == 1, collective.calls
        return completed[0]

    assert round_() == [] and ranks[0]._gen_idle_agreed  # idle: one agreed round
    calls = list(collective.calls)
    assert round_() == [] and collective.calls == calls  # skipped on every rank
    assert round_(admit=(1, 2, 3)) == []  # the admission clears the flag on every rank
    assert round_(finish={1: [0, 1], 2: [2]}) == []  # partial shards: no consensus yet
    assert round_(finish={1: [2, 3]}) == [1]  # last shard of 1: retired on every rank together
    assert round_(admit=(4,), finish={2: [0, 1, 3], 3: [3]}) == [2]
    assert round_(finish={3: [0, 1, 2], 4: [0, 1, 2, 3]}) == [3, 4]
    assert not any(t._recv_sessions for t in ranks)
    assert not ranks[0]._gen_idle_agreed  # this round still gathered sessions
    assert round_() == [] and ranks[0]._gen_idle_agreed
    # A receive in the round in which the last session retires: the flag stays clear.
    assert round_(admit=(5,)) == []
    assert round_(admit=(6,), finish={5: [0, 1, 2, 3]}) == [5]
    assert not ranks[0]._gen_idle_agreed
    assert round_(finish={6: [0, 1, 2, 3]}) == [6]
    assert round_() == [] and ranks[0]._gen_idle_agreed
    calls = list(collective.calls)
    for _ in range(3):
        assert round_() == []
    assert collective.calls == calls


def test_deferred_publication_keeps_lockstep() -> None:
    """Deferred receive starts keep the ranks in lockstep.

    A receive admitted on every rank in one round but started in staggered later rounds (each rank
    waits for its own admission readiness event) keeps the ranks in lockstep: the admission report
    clears the idle flag on every rank, and a deferred receive counts as a session until it starts.

    Without the report the admission round would agree idle (no rank holds a session yet), and the
    first rank to start the receive would enter the status collective that its peers skip.
    """
    world = 4
    collective = _Collective(world)
    ranks = [_make_receiving_rank(collective, r) for r in range(world)]
    _poll(ranks)
    assert all(t._gen_idle_agreed for t in ranks)

    def round_(start=(), admit=0):
        def step(i, t):
            if admit:
                t.set_gen_deferred_receives(admit)
            if i in start:
                t.request_and_receive_async(_request(11))
                t.set_gen_deferred_receives(0)
            return t.check_gen_transfer_status(0)

        statuses = _lockstep(ranks, step)
        assert len(set(collective.calls)) == 1, collective.calls
        assert len({t._gen_idle_agreed for t in ranks}) == 1
        return [sorted(s[0]) for s in statuses]

    calls = list(collective.calls)
    round_(admit=1)  # admitted everywhere, started nowhere: no rank may stay idle
    assert collective.calls == [c + 2 for c in calls]
    assert not any(t._gen_idle_agreed for t in ranks)
    round_(start=(0, 2))
    assert not any(t._gen_idle_agreed for t in ranks)
    round_(start=(1, 3))
    for t in ranks:
        t._recv_sessions[11].completed = True
    assert round_() == [[11]] * world
    assert not any(t._recv_sessions for t in ranks)
    round_()
    assert all(t._gen_idle_agreed for t in ranks)
