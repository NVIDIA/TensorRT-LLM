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
"""The first rank's admission ledger of a pipeline-parallel DFlash context server.

``pp_admission.py`` on its own: the slot bounds per constraint, the report
lag, the commitments of started prompts, head-of-line holding, and a lagged
pipeline simulation with per-stage page accounting (full-attention and
sliding-window pools, a last stage with about half the sliding-window pages,
a slow generation side that pulls finished contexts) in which no follower
ever fails to allocate what the first rank scheduled, every prompt finishes,
and the same workload without the ledger fails on the last stage.

The module imports nothing from the package, so it is loaded from its file
when ``tensorrt_llm`` cannot be imported.
"""

import importlib.util
import math
import random
import sys
from collections import deque
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


def _load():
    try:
        from tensorrt_llm._torch.pyexecutor import pp_admission

        return pp_admission
    except ImportError:
        path = (
            Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/pyexecutor/pp_admission.py"
        )
        spec = importlib.util.spec_from_file_location("pp_admission_under_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


A = _load()
TPB = 32
WINDOW_BLOCKS = 2048 // TPB + 1  # 65
CHUNK = 8192


def _snapshot(*constraints):
    return A.StageSnapshot(TPB, tuple(constraints)).to_wire()


def _full(available, capacity=None, resume_room=None):
    return A.Constraint(available, resume_room, capacity or 10**9, (None,))


def _swa(available, capacity=None, resume_room=None, lcs=1):
    return A.Constraint(available, resume_room, capacity or 10**9, (WINDOW_BLOCKS,) * lcs)


# --------------------------------------------------------------------------- bounds


def test_blocks_for_is_an_upper_bound_of_any_alignment():
    assert A.blocks_for(0, TPB) == 0
    assert A.blocks_for(-3, TPB) == 0
    for num_tokens in range(1, 200):
        worst = max(
            math.ceil((start + num_tokens) / TPB) - math.ceil(start / TPB) for start in range(TPB)
        )
        assert A.blocks_for(num_tokens, TPB) >= worst


def test_constraint_demands():
    full, swa, mixed = _full(0), _swa(0), A.Constraint(0, None, 0, (None, WINDOW_BLOCKS))
    # A reserved prefix pins every block in full attention, the window in a sliding window.
    assert full.prefix_demand(85504, TPB) == 2673
    assert swa.prefix_demand(85504, TPB) == WINDOW_BLOCKS
    assert swa.prefix_demand(100, TPB) == 5
    assert mixed.prefix_demand(85504, TPB) == 2673 + WINDOW_BLOCKS
    # A started prompt: all remaining blocks, or two chunks plus the window.
    assert full.remaining_demand(85504, CHUNK, TPB) == 2673
    assert mixed.remaining_demand(CHUNK, CHUNK, TPB) == 2 * 257
    assert swa.remaining_demand(85504, CHUNK, TPB) == 2 * 257 + WINDOW_BLOCKS
    assert swa.remaining_demand(1000, CHUNK, TPB) == 33
    assert swa.remaining_demand(85504, None, TPB) == 2673
    assert swa.remaining_demand(0, CHUNK, TPB) == 0


def test_room_is_the_resume_gate_on_the_gpu_level():
    assert _full(100).room == 100
    assert _full(100, resume_room=40).room == 40
    assert _full(100, resume_room=-5).room == -5


def test_snapshot_wire_round_trip_and_merge():
    snapshot = A.StageSnapshot(TPB, (_full(5, 9, 3), _swa(7, 11, lcs=2)))
    assert A.StageSnapshot.from_wire(snapshot.to_wire()) == snapshot
    assert all(len(c) == 5 for c in snapshot.to_wire()["c"])
    merged = A.StageSnapshot.merge([snapshot, A.StageSnapshot(TPB, (_full(1),))])
    assert merged.constraints == snapshot.constraints + (_full(1),)
    with pytest.raises(ValueError, match="block sizes"):
        A.StageSnapshot.merge([snapshot, A.StageSnapshot(64, ())])


# --------------------------------------------------------------------------- ledger


def _ledger(pp_size=2, max_chunk=CHUNK):
    return A.PPAdmissionLedger(pp_size, max_chunk)


def _cand(request_id, total, depth=0):
    return A.Candidate(request_id, depth, total)


def test_construction_is_checked():
    with pytest.raises(ValueError):
        A.PPAdmissionLedger(1, CHUNK)
    with pytest.raises(ValueError):
        A.PPAdmissionLedger(2, 0)


def test_every_first_chunk_waits_for_the_first_reports():
    ledger = _ledger(pp_size=3)
    held, line = ledger.hold(0, [_cand(1, 100), _cand(2, 100)], [1, 2])
    assert held == {1, 2} and line is None and not ledger.blocked
    ledger.record([], [])
    ledger.accept(1, 0, _snapshot(_swa(10**6)))
    held, _ = ledger.hold(1, [_cand(1, 100)], [1])
    assert held == {1}  # stage 2 has not reported yet
    ledger.record([], [])
    ledger.accept(2, 0, _snapshot(_swa(10**6)))
    held, _ = ledger.hold(2, [_cand(1, 100)], [1])
    assert held == set()
    assert ledger.num_startup_holds == 2


def test_head_of_line_holds_everything_behind_the_first_misfit():
    ledger = _ledger()
    # Room for 300 slots: a 8192-token prompt needs 257, a 1000-token one 33.
    ledger.accept(1, 0, _snapshot(_swa(300)))
    candidates = [_cand(1, 1000), _cand(2, CHUNK), _cand(3, 1000), _cand(4, 100)]
    held, line = ledger.hold(1, candidates, [1, 2, 3, 4])
    # 33 + 257 = 290 fits; 3 would need 33 more, so 3 and 4 wait though 4 fits.
    assert held == {3, 4}
    assert ledger.blocked and "holding 2 first chunk(s)" in line and "PP stage 1" in line


def test_every_follower_and_every_constraint_must_fit():
    ledger = _ledger(pp_size=3)
    ledger.accept(1, 0, _snapshot(_full(10**6), _swa(10**6)))
    # Stage 2's window pool group is at its resume gate: 40 slots left; a
    # 1000-token prompt needs 33.
    ledger.accept(2, 0, _snapshot(_full(10**6), _swa(10**6, resume_room=40)))
    held, line = ledger.hold(1, [_cand(1, 1000), _cand(2, 1000)], [1, 2])
    assert held == {2} and "PP stage 2 pool constraint 1 has room for 40" in line


def test_charges_after_the_report_and_only_after_it():
    ledger = _ledger()
    ledger.accept(1, 0, _snapshot(_swa(10**6)))
    chunks = {
        1: ([A.ScheduledChunk(9, 0, CHUNK, 4 * CHUNK, True)], []),
        2: ([A.ScheduledChunk(8, 0, CHUNK, CHUNK, True)], []),
        3: ([A.ScheduledChunk(7, 3008, 992, 4000, True)], [3000]),
        4: ([], [0]),
    }
    for iteration, (batch, depths) in chunks.items():
        ledger.hold(iteration, [], [9, 8, 7])
        ledger.record(batch, depths)
    # The report of iteration 2 has seen prompts 9 (first chunk) and 8 (all of it).
    ledger.accept(1, 2, _snapshot(_swa(600)))
    # Prompt 9: 24576 tokens to go, at most two chunks and the window (579).
    # Iteration 3: prompt 7's first chunk from a 3008-token prefix (window 65,
    # then 32 blocks) and a reservation at depth 3000 (65). Iteration 4 proposes nothing.
    assert ledger._needed(1) == [579 + 65 + 32 + 65]
    ledger.accept(1, 3, _snapshot(_swa(2000)))
    # Prompt 7 is finished and reported; only prompt 9 is still charged.
    assert ledger._needed(1) == [579]
    held, _ = ledger.hold(5, [_cand(1, 16000), _cand(2, 32)], [9, 1, 2])
    # 2000 - 579 = 1421 slots: 501 for the 16000-token prompt, then 2 more fit.
    assert held == set()
    ledger.record([], [])
    ledger.accept(1, 4, _snapshot(_swa(579 + 501)))
    held, _ = ledger.hold(6, [_cand(1, 16000), _cand(2, 32)], [9, 1, 2])
    assert held == {2}


def test_started_prompts_are_charged_once_and_never_held():
    ledger = _ledger()
    ledger.accept(1, 0, _snapshot(_full(1000)))
    held, _ = ledger.hold(1, [_cand(1, 20000)], [1])
    assert held == set()
    ledger.record([A.ScheduledChunk(1, 0, CHUNK, 20000, True)], [])
    # Before a report sees it, prompt 1 is charged all of its 626 blocks.
    assert ledger._needed(1) == [626]
    # Report 1 sees the first chunk's pages (257 of 1000 used).
    ledger.accept(1, 1, _snapshot(_full(1000 - 257)))
    # Prompt 1 still has 11808 tokens to go: 370 blocks.
    assert ledger._needed(1) == [370]
    # A second 20000-token prompt needs 626: 743 - 370 < 626, so it waits.
    held, _ = ledger.hold(2, [_cand(2, 20000)], [1, 2])
    assert held == {2}
    ledger.record([A.ScheduledChunk(1, CHUNK, CHUNK, 20000, False)], [])
    assert ledger.num_started == 1
    # Prompt 1's last chunk: it is no longer started, but stays charged until reported.
    ledger.hold(3, [_cand(2, 20000)], [1, 2])
    ledger.record([A.ScheduledChunk(1, 2 * CHUNK, 20000 - 2 * CHUNK, 20000, False)], [])
    assert ledger.num_started == 0
    assert ledger._needed(1) == [370]
    ledger.accept(1, 3, _snapshot(_full(1000)))
    held, _ = ledger.hold(4, [_cand(2, 20000)], [2])
    assert held == set()


def test_a_restarted_or_gone_prompt_drops_its_commitment():
    ledger = _ledger()
    ledger.accept(1, 0, _snapshot(_full(10**6)))
    ledger.hold(1, [_cand(1, 20000), _cand(2, 20000)], [1, 2])
    ledger.record(
        [A.ScheduledChunk(1, 0, CHUNK, 20000, True), A.ScheduledChunk(2, 0, 100, 20000, True)],
        [],
    )
    assert ledger.num_started == 2
    ledger.hold(2, [], live=[2], restarted=[])
    assert ledger.num_started == 1  # 1 is no longer active
    ledger.record([], [])
    ledger.hold(3, [_cand(2, 20000)], live=[2], restarted=[2])
    assert ledger.num_started == 0  # 2 starts over from its first chunk
    ledger.record([A.ScheduledChunk(2, 0, 100, 20000, True)], [])
    # Until a report sees them, prompt 1's chunk and both starts of prompt 2
    # are charged.
    assert ledger._needed(1) == [3 * 626]


def test_a_prompt_above_a_pools_capacity_bound_waits_for_the_whole_pool():
    """The demand is a loose bound: a prompt above a pool group's capacity is
    warned about once and let through only once that pool group is wholly free
    and nothing else is charged there."""
    ledger = _ledger()
    ledger.accept(1, 0, _snapshot(_full(60, capacity=100, resume_room=60)))
    big = _cand(1, 100 * TPB)
    held, _ = ledger.hold(1, [big, _cand(2, 32)], [1, 2])
    assert held == {1, 2} and ledger.blocked
    (warning,) = ledger.take_warnings()
    assert "may need up to 101" in warning and "offers at most 100" in warning
    ledger.record([], [])
    ledger.accept(1, 1, _snapshot(_full(100, capacity=100, resume_room=100)))
    held, _ = ledger.hold(2, [big, _cand(2, 32)], [1, 2])
    # The big prompt takes the whole pool group; the one behind it waits.
    assert held == {2} and ledger.take_warnings() == []
    assert ledger.num_oversized_admitted == 1
    ledger.record([A.ScheduledChunk(1, 0, CHUNK, 100 * TPB, True)], [])
    assert "1 prompt(s) let through above" in ledger.summary()


def test_the_managers_extra_tokens_are_charged_on_the_remaining_tokens():
    with_extra = A.Constraint(10**6, None, 10**9, (None,), extra_tokens=8)
    ledger = _ledger()
    # 64 tokens: 3 blocks; 64 + 8 tokens: 4 blocks.
    assert ledger.charge(_full(0), 0, 64, True, TPB) == 3
    assert ledger.charge(with_extra, 0, 64, True, TPB) == 4
    assert ledger.charge(with_extra, 32, 64, True, TPB) == 2 + 3
    snapshot = A.StageSnapshot(TPB, (with_extra,))
    assert A.StageSnapshot.from_wire(snapshot.to_wire()) == snapshot


def test_forgotten_requests_leave_no_epoch_behind():
    ledger = _ledger()
    ledger.accept(1, 0, _snapshot(_full(10**6)))
    for iteration, rid in enumerate(range(5)):
        ledger.hold(iteration, [_cand(rid, 20000)], [rid], restarted=[])
        ledger.record([A.ScheduledChunk(rid, 0, 100, 20000, True)], [])
        ledger.hold(iteration, [_cand(rid, 20000)], [rid], restarted=[rid])
        ledger.record([], [])
    # Every request restarted once; only the live one keeps its epoch.
    assert list(ledger._epochs) == [4]
    ledger.hold(9, [], [], restarted=[])
    assert ledger._epochs == {} and ledger.num_iterations == 11
    assert "ms of host time over 11 iteration(s)" in ledger.summary()


def test_reports_and_records_are_checked():
    ledger = _ledger()
    with pytest.raises(RuntimeError, match="no KV pool snapshot"):
        ledger.accept(1, 0, None)
    ledger.accept(1, 4, _snapshot(_full(1)))
    with pytest.raises(RuntimeError, match="after the one of iteration 4"):
        ledger.accept(1, 4, _snapshot(_full(1)))
    with pytest.raises(RuntimeError, match="without hold"):
        ledger.record([], [])


def test_events_every_follower_reported_are_dropped():
    ledger = _ledger(pp_size=3)
    for iteration in range(4):
        ledger.hold(iteration, [], [])
        ledger.record([], [7])
    assert len(ledger._events) == 4
    ledger.accept(1, 2, _snapshot(_full(1)))
    ledger.hold(4, [], [])
    ledger.record([], [])
    assert len(ledger._events) == 5  # stage 2 has not reported
    ledger.accept(2, 1, _snapshot(_full(1)))
    ledger.hold(5, [], [])
    ledger.record([], [])
    assert [e[0] for e in ledger._events] == [2, 3, 4, 5]
    # Stage 2 reported iteration 1: the reservations of iterations 2 and 3 (2 slots each).
    assert ledger._needed(2) == [4]


# --------------------------------------------------------------------------- lagged pipeline


class _Stage:
    """One follower's pools in slots, as the KV core allocates them.

    A full-attention pool group keeps every block of a prompt; a sliding-window
    pool group keeps the chunk's blocks while it runs and only the window's
    blocks once the chunk is committed. A first chunk's resume is refused above
    ``max_util`` on any GPU pool group. Finished contexts keep their pages until
    the generation side pulls them.
    """

    def __init__(self, full_slots, swa_slots, swa_lcs=1, max_util=1.0):
        self.capacity = [full_slots, swa_slots]
        self.lcs = [1, swa_lcs]
        self.max_util = max_util
        self.full = {}
        self.swa = {}

    def used(self):
        return [sum(self.full.values()), sum(self.swa.values())]

    def snapshot(self):
        constraints = []
        for i, (capacity, used) in enumerate(zip(self.capacity, self.used())):
            margin = math.ceil(capacity * (1.0 - self.max_util)) + 1
            window = None if i == 0 else WINDOW_BLOCKS
            constraints.append(
                A.Constraint(
                    capacity - used,
                    capacity - used - margin,
                    capacity - margin,
                    (window,) * self.lcs[i],
                )
            )
        return A.StageSnapshot(TPB, tuple(constraints)).to_wire()

    def _window_blocks(self, end):
        return math.ceil(end / TPB) - max(0, end - 2048) // TPB

    def adopt(self, rid, position, num_tokens, first):
        used = self.used()
        if first and any(u / c > self.max_util for u, c in zip(used, self.capacity)):
            return False
        new = math.ceil((position + num_tokens) / TPB) - math.ceil(position / TPB)
        if first and position:
            # The reserved prefix comes back from a lower tier.
            full_new = new + math.ceil(position / TPB)
            swa_new = new + self._window_blocks(position)
        else:
            full_new, swa_new = new, new
        need = [full_new * self.lcs[0], swa_new * self.lcs[1]]
        if any(u + n > c for u, n, c in zip(used, need, self.capacity)):
            return False
        self.full[rid] = self.full.get(rid, 0) + need[0]
        self.swa[rid] = self.swa.get(rid, 0) + need[1]
        return True

    def commit(self, rid, end):
        if rid in self.swa:
            self.swa[rid] = min(self.swa[rid], self._window_blocks(end) * self.lcs[1])

    def free(self, rid):
        self.full.pop(rid, None)
        self.swa.pop(rid, None)


def _simulate(stages, prompts, *, lag, inflight, pull_every, ledger_on, depths=None, seed=0):
    """Run ``prompts`` (arrival order) through a lagged pipeline.

    Returns ``(finished, failure, ledger, passes)``; ``failure`` names the first
    follower allocation that failed (None when none did).
    """
    rng = random.Random(seed)
    pp_size = len(stages) + 1
    ledger = A.PPAdmissionLedger(pp_size, CHUNK) if ledger_on else None
    depths = depths or {}
    position = {rid: depths.get(rid, 0) for rid in range(len(prompts))}
    started, busy_until, done_at = set(), {}, {}
    arrived = deque(range(len(prompts)))
    active = []
    reports = deque()  # (iteration, [snapshot per follower])
    pending_commits = deque()  # (iteration due, rid, end)
    finished_contexts = deque()
    finished = []
    for iteration in range(200000):
        for _ in range(rng.randint(0, 2)):
            if arrived:
                active.append(arrived.popleft())
        if ledger_on:
            while reports and reports[0][0] <= iteration - lag:
                reported, snapshots = reports.popleft()
                for stage, wire in enumerate(snapshots, start=1):
                    ledger.accept(stage, reported, wire)
        waiting = [rid for rid in active if rid not in started]
        held = set()
        if ledger_on:
            candidates = [A.Candidate(rid, depths.get(rid, 0), prompts[rid]) for rid in waiting]
            held, _ = ledger.hold(iteration, candidates, active, waiting)
        # The first rank's pass: started prompts first, then new ones, FCFS.
        budget, batch = CHUNK, []
        order = [r for r in active if r in started] + [r for r in waiting if r not in held]
        for rid in order:
            if budget == 0:
                break
            if busy_until.get(rid, -1) > iteration or position[rid] >= prompts[rid]:
                continue
            size = min(budget, prompts[rid] - position[rid])
            budget -= size
            batch.append((rid, position[rid], size, rid not in started))
        assert not held & {rid for rid, *_ in batch}
        if ledger_on:
            ledger.record(
                [
                    A.ScheduledChunk(rid, pos, size, prompts[rid], first)
                    for rid, pos, size, first in batch
                ],
                [],
            )
        # Every follower adopts the batch of this iteration.
        for stage_index, stage in enumerate(stages, start=1):
            for rid, pos, size, first in batch:
                if not stage.adopt(rid, pos, size, first):
                    return finished, (stage_index, iteration, rid), ledger, iteration
        for rid, pos, size, first in batch:
            started.add(rid)
            position[rid] = pos + size
            busy_until[rid] = iteration + inflight
            pending_commits.append((iteration + inflight, rid, pos + size))
        # Chunks complete ``inflight`` iterations after they were scheduled.
        while pending_commits and pending_commits[0][0] <= iteration:
            _, rid, end = pending_commits.popleft()
            for stage in stages:
                stage.commit(rid, end)
            if end >= prompts[rid]:
                active.remove(rid)
                finished_contexts.append(rid)
                done_at[rid] = iteration
        # The generation side pulls one finished context every ``pull_every`` iterations.
        if finished_contexts and iteration % pull_every == 0:
            rid = finished_contexts.popleft()
            for stage in stages:
                stage.free(rid)
            finished.append(rid)
        reports.append((iteration, [stage.snapshot() for stage in stages]))
        if len(finished) == len(prompts):
            return finished, None, ledger, iteration
    raise AssertionError("the simulated pipeline never finished")


def _workload(seed, num_prompts=120, longest=40000):
    rng = random.Random(seed)
    return [rng.randint(500, longest) for _ in range(num_prompts)]


def _tight_last_stage(max_util=1.0, swa_lcs=1):
    # Stages 1 and 2 like the first rank; the last stage has about half the
    # sliding-window slots (its window page also carries the drafter's layers).
    return [
        _Stage(400000, 4000, max_util=max_util),
        _Stage(400000, 4000, max_util=max_util),
        _Stage(400000, 2140, swa_lcs=swa_lcs, max_util=max_util),
    ]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lag, inflight", [(1, 2), (3, 4), (7, 8)])
def test_lagged_pipeline_never_fails_a_follower_and_finishes(seed, lag, inflight):
    prompts = _workload(seed)
    stock_finished, stock_failure, _, _ = _simulate(
        _tight_last_stage(),
        prompts,
        lag=lag,
        inflight=inflight,
        pull_every=6,
        ledger_on=False,
        seed=seed,
    )
    assert stock_failure is not None and stock_failure[0] == 3, (
        "the workload must overrun the last stage"
    )
    finished, failure, ledger, passes = _simulate(
        _tight_last_stage(),
        prompts,
        lag=lag,
        inflight=inflight,
        pull_every=6,
        ledger_on=True,
        seed=seed,
    )
    assert failure is None, failure
    assert sorted(finished) == list(range(len(prompts)))
    assert ledger.num_held_iterations > 0
    assert ledger.num_started == 0


@pytest.mark.parametrize("seed", range(3))
def test_resume_gate_and_shared_pool_groups(seed):
    """The last stage resumes nothing above 0.95 utilization, and its window pool
    group serves two life cycles (a chunk takes two slots per block)."""
    prompts = _workload(seed, num_prompts=80)
    _, stock_failure, _, _ = _simulate(
        _tight_last_stage(0.95, swa_lcs=2),
        prompts,
        lag=3,
        inflight=4,
        pull_every=6,
        ledger_on=False,
        seed=seed,
    )
    assert stock_failure is not None
    finished, failure, ledger, _ = _simulate(
        _tight_last_stage(0.95, swa_lcs=2),
        prompts,
        lag=3,
        inflight=4,
        pull_every=6,
        ledger_on=True,
        seed=seed,
    )
    assert failure is None, failure
    assert len(finished) == len(prompts) and ledger.num_held_iterations > 0


def test_reused_prefixes_are_charged_when_their_first_chunk_runs():
    """Admitted at a reused depth, a first chunk brings its prefix back on every
    stage (the reservation's pages may sit on a lower tier)."""
    prompts = _workload(11, num_prompts=60)
    depths = {rid: (p // 2) // TPB * TPB for rid, p in enumerate(prompts) if rid % 3 == 0}
    finished, failure, ledger, _ = _simulate(
        _tight_last_stage(), prompts, lag=3, inflight=4, pull_every=6, ledger_on=True, depths=depths
    )
    assert failure is None, failure
    assert len(finished) == len(prompts)


def test_a_last_stage_that_fits_one_prompt_still_finishes():
    """No deadlock at the smallest pool the head of the line fits in alone."""
    prompts = [CHUNK * 2] * 12
    window_need = 2 * 257 + WINDOW_BLOCKS
    stages = [_Stage(10**6, 10**5), _Stage(10**6, 10**5), _Stage(10**6, window_need + 1)]
    finished, failure, ledger, _ = _simulate(
        stages, prompts, lag=3, inflight=4, pull_every=2, ledger_on=True
    )
    assert failure is None and len(finished) == len(prompts)
    assert ledger.max_held >= 1


# --------------------------------------------------------------------------- tight margins


def _tight_window_stages():
    # The last stage's window pool group is barely larger than the ledger's
    # exact charge for the report lag.
    return [_Stage(400000, 4000), _Stage(400000, 4000), _Stage(400000, 700)]


def _tight_reused_stages():
    # Both of the last stage's pool groups bind: deep reused prefixes fill the
    # full-attention pool, the window pool group is small.
    return [_Stage(400000, 4000), _Stage(400000, 4000), _Stage(2000, 1000)]


def _reused_workload(seed):
    prompts = _workload(11 + seed, num_prompts=60)
    depths = {rid: (p * 3 // 4) // TPB * TPB for rid, p in enumerate(prompts) if rid % 2 == 0}
    return prompts, depths


def _run_tight(stages, seed, lag, inflight, depths_too):
    if depths_too:
        prompts, depths = _reused_workload(seed)
    else:
        prompts, depths = _workload(seed, num_prompts=60), None
    finished, failure, _, _ = _simulate(
        stages(),
        prompts,
        lag=lag,
        inflight=inflight,
        pull_every=6,
        ledger_on=True,
        depths=depths,
        seed=seed,
    )
    return failure, len(finished) == len(prompts)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("lag, inflight", [(3, 4), (7, 8)])
def test_a_tight_window_pool_never_fails(seed, lag, inflight):
    failure, all_done = _run_tight(_tight_window_stages, seed, lag, inflight, False)
    assert failure is None and all_done, failure


@pytest.mark.parametrize("seed", range(4))
def test_tight_reused_full_and_window_pools_never_fail(seed):
    failure, all_done = _run_tight(_tight_reused_stages, seed, 3, 4, True)
    assert failure is None and all_done, failure


def _skip_one_unreported_iteration(original):
    def needed(self, stage):
        report = self._reports.get(stage)
        if report is None:
            return None
        self._reports[stage] = (report[0] + 1, report[1])
        try:
            return original(self, stage)
        finally:
            self._reports[stage] = report

    return needed


def _no_first_chunk_prefix(self, c, position, total_tokens, first, tokens_per_block):
    return c.remaining_demand(
        total_tokens - position + c.extra_tokens, self.max_chunk_tokens, tokens_per_block
    )


@pytest.mark.parametrize("mutation", ["report_lag_off_by_one", "no_first_chunk_prefix"])
def test_the_tight_cases_catch_an_unsafe_ledger(mutation, monkeypatch):
    """The tight cases above fail when the ledger skips one unreported
    iteration or forgets a first chunk's prefix."""
    if mutation == "report_lag_off_by_one":
        monkeypatch.setattr(
            A.PPAdmissionLedger,
            "_needed",
            _skip_one_unreported_iteration(A.PPAdmissionLedger._needed),
        )
    else:
        monkeypatch.setattr(A.PPAdmissionLedger, "charge", _no_first_chunk_prefix)
    failures = [_run_tight(_tight_reused_stages, seed, 3, 4, True)[0] for seed in range(4)]
    assert all(f is not None and f[0] == 3 for f in failures), failures
    if mutation == "report_lag_off_by_one":
        failures = [_run_tight(_tight_window_stages, seed, 3, 4, False)[0] for seed in range(4)]
        assert any(f is not None and f[0] == 3 for f in failures), failures
