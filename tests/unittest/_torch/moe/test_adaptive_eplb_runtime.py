# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the adaptive EPLB runtime: statistics windows, the decision process and installs."""

import contextlib
import dataclasses
import os
import resource
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import runtime as rt
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.balancer import AdaptiveMoeLoadBalancer
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
    ExpertGeometry,
    ExpertStatistics,
    TimeModel,
    statistics_buffers,
)
from tensorrt_llm._torch.pyexecutor.llm_request import generation_progress
from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

pytestmark = pytest.mark.cpu_only

E, R, LOCAL, L = 8, 2, 6, 2
GEOMETRY = ExpertGeometry(E, R * LOCAL, R)
MODEL = TimeModel(1.0, 1.0, 1.0, 0.5, 1.0, 128)
HOME = GEOMETRY.home_layout()
TABLE = 2 * (E + R * LOCAL)
FORWARDS = 8  # forwards of the windows after the first: the scripted decisions' and the spy's


def fake_event(enable_timing: bool = False, blocking: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        record=lambda stream=None: None,
        query=lambda: True,
        synchronize=lambda: None,
        elapsed_time=lambda end: 0.0,
    )


STREAM = SimpleNamespace(wait_event=lambda event: None)


@pytest.fixture
def host_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", fake_event)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: STREAM)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


class FakeLayer:
    expert_count, num_slots = E, R * LOCAL

    def __init__(self, clock: list[int], buffers: dict, index: int) -> None:
        on = torch.ones(1, dtype=torch.int32)
        self.clock, self.statistics = clock, ExpertStatistics.of(buffers, index, on)

    def home_layout(self) -> list[int]:
        return HOME.tolist()


class FakeTransport:
    """A 0.25 ms batch completing ``delay`` calls after the runtime's call that issued it;
    ``waiting()`` jobs still wait for the clock."""

    def __init__(self, clock: list[int], delay: int = 1) -> None:
        self.clock, self.delay, self.batches, self.due = clock, delay, [], []
        self.waiting = lambda: sum(due > self.clock[0] for due in self.due)
        self.runtime = None  # the runtime it serves (make_ranks)

    def copy(self, experts, stream, start, end) -> None:
        due = self.runtime.call + self.delay
        self.batches.append(list(experts))

        def synchronize() -> None:  # the test's settle guards the wait with its own deadline
            self.due.append(due)
            while self.clock[0] < due:
                time.sleep(0.0005)
            self.due.remove(due)

        end.synchronize, start.elapsed_time = synchronize, lambda end: 0.25


class FakeGroup:
    """Blocking all-gathers of in-process worker threads; ``waiting[rank]`` until all posted."""

    def __init__(self) -> None:
        self.posts, self.waiting = {}, {}
        self.ready = threading.Condition()

    def gather(self, rank: int):
        counts = [0]

        def start(send, rows, leave=None) -> None:
            key = counts[0]
            counts[0] += 1
            with self.ready:
                posts = self.posts.setdefault(key, {})
                posts[rank] = send.clone()
                self.ready.notify_all()
                self.waiting[rank] = (posts, len(rows))
                self.ready.wait_for(lambda: len(posts) == len(rows), timeout=30)
                del self.waiting[rank]
                assert len(posts) == len(rows), f"gather {key}: {sorted(posts)} posted"
                for r, row in enumerate(rows):
                    row.copy_(posts[r])

        return start


class FakeDecision:
    """The decision process in this process, or the updates ``script(window)`` prepares."""

    def __init__(self, rank: int, script=None, copy_ms=None) -> None:
        self.stats = torch.zeros(R, rt.window_stats_size(L, E))
        homes = np.tile(HOME, (L, 1))
        config = rt.DecisionConfig(GEOMETRY, MODEL, homes, rank, 1, copy_ms=copy_ms)
        self.decider = rt._Decider(config, self.stats.numpy())
        self.script = script
        self.open_after = 4  # calls from a scripted apply call to the next window's open
        self.queue = []
        self.sent = []

    def submit(self, message) -> None:
        self.sent.append(message)
        if self.script is None:
            reply = self.decider.handle(message)
        elif message[0] == "window":
            reply = ("update", self.script(message[1]))
        else:
            apply_call = int(message[2][:, rt.ROW.index("copied_call")].max()) + 1
            open_call = apply_call + self.open_after
            reply = ("schedule", message[1] + 1, open_call, apply_call, None, FORWARDS)
        self.queue.append(reply)

    def recv(self):
        return self.queue.pop(0)

    def close(self) -> None:
        pass


class Events:
    """The runtime's logged events of one rank (in place of its ``_event``): the decision records,
    completed by the later events."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.records = []
        self.by_window = {}
        self.kinds = []
        self.stats = []
        self.open_calls = {}

    def __call__(self, event: str, **data) -> None:
        self.kinds.append(event)
        if event == "window":
            self.stats.append(data["stats"].copy())
            self.open_calls[data["window"]] = data["open_call"]
        elif event == "decision":
            record = dict(
                window=data["window"],
                open_call=self.open_calls[data["window"]],
                solved_call=data["call"],
                tables=sorted(data["tables"]),
                moved=data["moved"],
                effect={},
            )
            self.records.append(record)
            self.by_window[data["window"]] = record
        elif event == "copied":
            row = data["rows"][self.rank]
            self.by_window[data["window"]]["copied_call"] = int(row[rt.ROW.index("copied_call")])
        elif event == "scheduled":
            keys = ("apply_call", "plan", "next_open", "forwards")
            self.records[-1].update({key: data[key] for key in keys})
        elif event == "applied":
            self.by_window[data["window"]]["effect"].update(
                dict.fromkeys(data["layers"], data["call"])
            )


def records(runtime) -> list[dict]:
    return runtime._event.records


def executor_progress(requests: list) -> tuple:
    """The arguments of ``add_request_progress`` for ``requests``, as the executor passes them."""
    return len(requests), lambda start, stop: generation_progress(requests, start, stop)


def make_ranks(decisions, clock, copy_delays=(1, 1)):
    group = FakeGroup()
    ranks = []
    for rank, (decision, delay) in enumerate(zip(decisions, copy_delays)):
        buffers = statistics_buffers(L, E, "cpu")
        layers = [FakeLayer(clock, buffers, index) for index in range(L)]
        transport = FakeTransport(clock, delay)
        gather = group.gather(rank)
        tables = torch.zeros(L, TABLE, dtype=torch.int32)
        runtime = rt.AdaptiveRuntime(
            layers, transport, STREAM, decision, gather, "cpu", tables, buffers
        )
        runtime._event = Events(rank)
        transport.runtime = runtime

        def waiting(t=transport, r=rank) -> int:
            waiting = group.waiting.get(r)
            return t.waiting() + (1 if waiting and len(waiting[0]) < waiting[1] else 0)

        runtime.waiting = waiting
        ranks.append(runtime)
    return ranks


def settle(ranks) -> None:
    """Let every worker thread run until its jobs are done or waiting (for the clock or a peer)."""
    deadline = time.monotonic() + 30
    while any(r.worker.jobs.unfinished_tasks > r.waiting() for r in ranks):
        assert time.monotonic() < deadline, [
            (r.worker.jobs.unfinished_tasks, r.waiting()) for r in ranks
        ]
        time.sleep(0.0002)
    for runtime in ranks:
        assert runtime.worker.failure is None, runtime.worker.failure


def run(ranks, clock, calls: int) -> None:
    """Calls as the executor makes them, each seeing its batch first (no requests here). The
    clock reaches a call once every boundary ran it: a copy completes at its due call."""
    for _ in range(calls):
        for runtime in ranks:
            if runtime.wants_request_progress:
                runtime.add_request_progress(*executor_progress([]))
            runtime.boundary()
        clock[0] = ranks[0].call
        settle(ranks)


def set_statistics(runtime, rows: np.ndarray, forwards: int) -> None:
    for layer, counts in zip(runtime.layers, rows):
        layer.statistics.rows.copy_(torch.from_numpy(counts))
        layer.statistics.forwards.fill_(forwards)


def even_split(rows, layout, geometry) -> np.ndarray:
    """Shares that split every expert's rows evenly over its copies."""
    copies = np.bincount(layout[layout >= 0], minlength=geometry.num_experts)
    return np.where(layout >= 0, 1.0 / copies[np.maximum(layout, 0)], 0.0)


# The runtime's own functions: a test that spies twice must not spy on its first spy.
REAL_PLAN, REAL_CHOOSE = rt.plan_layout, rt.choose_layers_to_install


def spy(monkeypatch, seen: dict) -> None:
    """Record what the planner, the share solver, the layer choice and the window length are
    given; every window after the first lasts ``FORWARDS`` forwards."""

    def plan(load, *args):
        seen.setdefault("load", []).append(np.array(load))
        return REAL_PLAN(load, *args)

    def solve(rows, layout, geometry):
        seen.setdefault("rows", []).append(rows)
        return even_split(rows, layout, geometry)

    def choose(gains, steps_left, copy_cost):
        seen.setdefault("gains", []).append(dict(gains))
        seen.setdefault("steps_left", []).append(steps_left)
        return REAL_CHOOSE(gains, steps_left, copy_cost)

    def window_forwards(rows_per_forward, experts):
        seen.setdefault("per_forward", []).append((rows_per_forward, experts))
        return FORWARDS

    monkeypatch.setattr(rt, "plan_layout", plan)
    monkeypatch.setattr(rt, "solve_shares", solve)
    monkeypatch.setattr(rt, "choose_layers_to_install", choose)
    monkeypatch.setattr(rt, "window_forwards", window_forwards)


def spied_ranks(monkeypatch, timed: bool = False, first=None):
    """Both ranks with in-process deciders once their first window (counting ``first``: every
    rank's rows and forwards) has passed, and ``seen``; ``timed``: a copy group was timed."""
    seen = {}
    clock = [0]
    spy(monkeypatch, seen)
    decisions = [FakeDecision(rank, copy_ms=0.5 if timed else None) for rank in range(R)]
    ranks = make_ranks(decisions, clock)
    run(ranks, clock, 1)
    rows, forwards = first or (np.ones((R, L, E), dtype=np.float32), (1,) * R)
    for runtime, counts, count in zip(ranks, rows, forwards):
        set_statistics(runtime, counts, count)
    while not all(runtime.window == 1 and runtime.close_at is not None for runtime in ranks):
        assert clock[0] < 32, "the second window did not open"
        run(ranks, clock, 1)
    return seen, clock, decisions, ranks


def manager() -> AdaptiveMoeLoadBalancer:
    config = MoeAdaptiveLoadBalancerConfig(num_slots=R * LOCAL)
    config.setup(ep_rank=0, ep_size=R)
    return AdaptiveMoeLoadBalancer(ep_rank=0, ep_size=R, config=config, mapping=None)


def request(max_new_tokens, decoding_iter, accepted_draft_tokens, dummy=False):
    return SimpleNamespace(
        is_attention_dp_dummy=dummy,
        is_cuda_graph_dummy=False,
        py_max_new_tokens=max_new_tokens,
        py_decoding_iter=decoding_iter,
        py_num_accepted_draft_tokens=accepted_draft_tokens,
    )


def window_inputs(monkeypatch, forwards=(FORWARDS,) * R):
    """The second window's decision on every rank's random rows over its ``forwards``."""
    seen, clock, decisions, ranks = spied_ranks(monkeypatch)
    rows = np.random.default_rng(7).integers(0, 64, (R, L, E)).astype(np.float32)
    progress = ([(120, 10, 1), (70, 10, 1), (1, 1, 0, True)], [(75, 15, 2)])
    for runtime, counts, count in zip(ranks, rows, forwards):
        set_statistics(runtime, counts, count)
    run(ranks, clock, FORWARDS)
    # The window closed: the gather waits for the requests' progress, read in slices.
    assert all(runtime.wants_request_progress for runtime in ranks)
    for runtime, items in zip(ranks, progress):
        runtime.add_request_progress(*executor_progress([request(*item) for item in items]))
    run(ranks, clock, 2)
    return seen, rows, decisions, ranks


def test_a_window_counts_every_ranks_rows_over_its_own_forwards(host_cuda, monkeypatch):
    """The planner and the share solver get every rank's rows over its own forwards, summed. A
    rank whose device counted one forward more (a step without a boundary) moves only that count:
    not the window's calls, the steps left or the decision."""
    base, _, _, base_ranks = window_inputs(monkeypatch)
    seen, rows, _, ranks = window_inputs(monkeypatch, (FORWARDS, FORWARDS + 1))
    expected = rows[0].astype(np.float64) / FORWARDS + rows[1].astype(np.float64) / (FORWARDS + 1)
    assert len(seen["load"]) == R * L  # every layer planned on both ranks: not skipped
    for i, (load, solved) in enumerate(zip(seen["load"], seen["rows"])):
        assert np.array_equal(load, expected[i % L]) and np.array_equal(solved, expected[i % L])
    last = [records(runtime)[-1] for runtime in (*ranks, base_ranks[0])]
    for key in ("window", "open_call", "solved_call"):
        assert last[0][key] == last[1][key] == last[2][key]
    # 100 + 50 + 30 tokens remain over 3 requests (the dummy does not count) at 7 / 3 per step.
    assert seen["steps_left"] == base["steps_left"] == [180 / (7 / 3) / 3] * R


def test_the_first_window_lasts_one_call_and_sizes_the_next(host_cuda, monkeypatch):
    """The first window lasts one call and sets the length of the next ones from its rows per
    forward, which the schedule hands to every rank's runtime; it is decided only when it holds
    the rows of a window, ``ROWS_PER_EXPERT`` per expert."""
    for per_rank, planned in ((rt.ROWS_PER_EXPERT // R - 1, 0), (rt.ROWS_PER_EXPERT // R, R * L)):
        rows = np.full((R, L, E), per_rank, dtype=np.float32)
        seen, _, _, ranks = spied_ranks(monkeypatch, first=(rows, (1,) * R))
        assert len(seen.get("load", [])) == planned
        assert seen["per_forward"] == [(R * per_rank * E, E)] * R
        for runtime in ranks:
            (first,) = records(runtime)
            # It closed at call 1, was gathered at call 2 and decided at call 3.
            assert (first["open_call"], first["solved_call"], first["tables"]) == (0, 3, [])
            assert first["forwards"] == FORWARDS
            assert runtime.close_at - runtime.window_opened_call == FORWARDS


def test_a_rank_without_a_forward_skips_the_window(host_cuda, monkeypatch):
    seen, clock, _, ranks = spied_ranks(monkeypatch)
    for runtime, forwards in zip(ranks, (FORWARDS, 0)):
        set_statistics(runtime, np.ones((L, E), dtype=np.float32), forwards)
    run(ranks, clock, FORWARDS + 4)
    assert "load" not in seen
    last = [records(runtime)[-1] for runtime in ranks]
    assert all(record["window"] == 1 and record["tables"] == [] for record in last)
    # Nothing was learnt: the next window opens the call after the skipped one's apply call.
    assert [record["next_open"] - record["apply_call"] for record in last] == [1, 1]


def test_layers_not_live_on_every_rank_are_not_planned(monkeypatch):
    seen = {}
    spy(monkeypatch, seen)
    decision = FakeDecision(0)
    decision.decider.next_forwards = FORWARDS  # an earlier window measured the rows per forward
    stats = decision.stats.numpy()
    stats[:, : L * E] = 1.0
    stats[:, L * E : L * (E + 1)] = 8
    stats[:, L * (E + 1) : L * (E + 2)] = [[0, -1], [-1, -1]]
    decision.decider.committed[:] = [0, -1]
    decision.decider.decide(0)
    assert len(seen["load"]) == 1 and np.array_equal(seen["load"][0], np.full(E, 2 / 8))


def test_copy_items_take_free_slots_from_the_node_first():
    geometry = ExpertGeometry(8, 12, 4)
    old = geometry.home_layout()
    old[11] = 0
    new = old.copy()
    new[[5, 8]] = 0
    assert rt.copy_items(old, new, geometry, 1, 2) == [(2, 0, 0)]
    assert rt.copy_items(old, new, geometry, 2, 2) == [(2, 3, 2)]
    assert rt.copy_items(old, new, geometry, 0, 2) == []
    new[0] = 3
    with pytest.raises(ValueError, match="not free"):
        rt.copy_items(old, new, geometry, 0, 2)


def script(rank, table_of_layer_1=2):
    def update(window):
        tables = {0: np.full(TABLE, 1, np.int32), 1: np.full(TABLE, table_of_layer_1, np.int32)}
        copies = {0: [(4, 0, 0)] if rank else []}
        result = rt.LayoutUpdate(window, copy_items=copies, tables=tables, moved=[0])
        result.digest = result.fingerprint()
        return result

    return update


@pytest.mark.parametrize("delays,open_after", [((1, 5), 4), ((1, 1), 1)])
def test_tables_apply_after_every_rank_copied_and_the_window_opens_as_scheduled(
    host_cuda, delays, open_after
):
    """Rank 1's copies finish ``delays[1]`` calls after the decision: both ranks apply the new
    tables at the same call, the one after the latest completion, before the next window is
    scheduled. The window event hands over the gathered statistics. The schedule is read before
    windows open: the next window opens at its call, also when that is the call its schedule
    arrives (``open_after`` 1), not a call later."""
    clock = [0]
    decisions = [FakeDecision(rank, script=script(rank)) for rank in range(R)]
    for decision in decisions:
        decision.open_after = open_after
    ranks = make_ranks(decisions, clock, copy_delays=delays)
    while not all(runtime.window == 1 and runtime.close_at for runtime in ranks):
        assert clock[0] < 32, "the second window did not open"
        run(ranks, clock, 1)
    first = [records(runtime)[0] for runtime in ranks]
    solved = first[0]["solved_call"]
    copied = [r["copied_call"] for r in first]
    assert first[1]["solved_call"] == solved and copied == [solved + d for d in delays]
    apply = copied[1] + 1
    assert [r["apply_call"] for r in first] == [apply, apply]
    assert [r["effect"] for r in first] == [{0: apply, 1: apply}] * 2
    for runtime, decision in zip(ranks, decisions):
        assert runtime.tables.tolist() == [[1] * TABLE, [2] * TABLE]
        assert runtime._event.kinds == "window decision copied applied scheduled".split()
        (stats,) = runtime._event.stats
        assert np.array_equal(stats, decision.stats.numpy())
        assert runtime.window_opened_call == records(runtime)[0]["next_open"] == apply + open_after
        assert runtime.applied_window.tolist() == [0, 0]
    assert ranks[0].transport.batches == [[]] and ranks[1].transport.batches == [[(0, 4, 0, 0)]]


def test_a_publish_copies_only_the_layers_it_changes(host_cuda):
    """A device row the host mirror does not hold survives the publish of another layer."""
    runtime = make_ranks([FakeDecision(rank) for rank in range(R)], [0])[0]
    runtime._event = lambda event, **data: None  # a bare publish: no decision recorded
    runtime.tables[0] = 7
    runtime._apply_tables(0, rt.LayoutUpdate(0, tables={1: np.full(TABLE, 3, np.int32)}))
    assert runtime.tables.tolist() == [[7] * TABLE, [3] * TABLE]


def test_an_update_applies_before_a_window_that_opens_at_the_same_call(host_cuda):
    """The copies' completion and the next window's schedule are read at one call, after both the
    apply call and the open call: the tables apply first, so the window counts under them."""
    runtime = make_ranks([FakeDecision(rank) for rank in range(R)], [0])[0]
    kinds, opened, open_window = [], [], runtime._open_window
    runtime._event = lambda event, **data: kinds.append(event)
    runtime._open_window = lambda: opened.append(runtime.tables.tolist()) or open_window()
    runtime.worker.post = lambda job: None  # the decision process's answers are scripted here
    update = script(0)(0)
    rows = np.array([[8, 1, 0.25, update.digest]] * R)  # every rank copied at call 8: apply at 9
    runtime.worker.inbox.extend([("copied", update, rows), ("schedule", 1, 9, 9, None, FORWARDS)])
    runtime.call, runtime.open_at = 9, None
    runtime.boundary()  # call 10
    assert kinds == ["copied", "scheduled", "applied"] and runtime.window_opened_call == 10
    assert opened == [[[1] * TABLE, [2] * TABLE]] and not runtime.pending_updates


def test_ranks_that_decided_differently_stop_before_applying(host_cuda):
    """The digest covers the tables and the moves, not a rank's own copies. Ranks that decided a
    window differently stop adaptive EPLB, not the server: no table applies, and later calls only
    drop the worker's messages."""
    update = script(1)(0)
    same_decision_other_rank = script(0)(0)
    other_table = script(1, 3)(0)
    other_moves = script(1)(0)
    other_moves.moved = []
    assert update.fingerprint() == same_decision_other_rank.fingerprint()
    assert update.fingerprint() not in (other_table.fingerprint(), other_moves.fingerprint())
    assert float(update.fingerprint()) == update.fingerprint()  # exact in the float64 row
    clock = [0]
    decisions = [FakeDecision(0, script=script(0)), FakeDecision(1, script=script(1, 3))]
    ranks = make_ranks(decisions, clock)
    run(ranks, clock, 22)
    for runtime in ranks:
        assert "decided window 0 differently" in str(runtime.failure)
        assert not runtime.tables.any() and not runtime.pending_updates
    calls = [runtime.call for runtime in ranks]
    run(ranks, clock, 5)
    assert [runtime.call for runtime in ranks] == calls
    assert all(not runtime.tables.any() for runtime in ranks)


def test_a_failed_worker_stops_adaptive_eplb_and_keeps_the_tables(host_cuda):
    """An error on the worker thread (a decision process that exited, a failed copy) stops adaptive
    EPLB at the next boundary instead of raising in the main thread; the tables in use stay."""
    clock = [0]
    ranks = make_ranks([FakeDecision(rank) for rank in range(R)], clock)
    runtime = ranks[0]
    runtime.tables[0] = 7
    before = runtime.tables.clone()
    runtime.worker.failure = RuntimeError("adaptive EPLB decision process exited (1)")
    runtime.boundary()
    assert isinstance(runtime.failure, RuntimeError)
    assert torch.equal(runtime.tables, before) and not runtime.wants_request_progress
    runtime.worker.inbox.append(("copied", script(0)(0), np.zeros((R, 4))))
    runtime.boundary()
    assert not runtime.worker.inbox and torch.equal(runtime.tables, before)


def test_windows_keep_their_numbers_across_installs(host_cuda, monkeypatch):
    """A window gathered while an earlier update's tables are still on record is sent to the
    decision process under its own number."""
    _, clock, decisions, ranks = spied_ranks(monkeypatch, timed=True)
    hot = np.tile([400.0, 400.0] + [1.0] * (E - 2), (L, 1)).astype(np.float32)
    for _ in range(3):  # three more windows: the later ones gather after an install
        for runtime in ranks:
            set_statistics(runtime, hot, FORWARDS)
        run(ranks, clock, 22)
    sent = [m[1] for m in decisions[0].sent if m[0] == "window"]
    assert sent == list(range(len(sent))) and len(sent) >= 3
    assert [r["window"] for r in records(ranks[0])] == sent


def test_two_ranks_move_an_expert_and_apply(host_cuda, monkeypatch):
    """Real deciders on both ranks: a window that replicates the two hot experts gives every rank
    its own copy items, the same digest, and new tables on both."""
    _, clock, _, ranks = spied_ranks(monkeypatch, timed=True)
    hot = np.tile([400.0, 400.0] + [1.0] * (E - 2), (L, 1)).astype(np.float32)
    for runtime in ranks:
        set_statistics(runtime, hot, FORWARDS)
    run(ranks, clock, FORWARDS)
    for runtime in ranks:
        runtime.add_request_progress(*executor_progress([request(1000, 10, 0) for _ in range(4)]))
    run(ranks, clock, 12)
    decided = [runtime._event.by_window[1] for runtime in ranks]
    assert all(record["moved"] == [0, 1] for record in decided)
    assert decided[0]["apply_call"] == decided[1]["apply_call"]
    # Experts 0 and 1 are rank 0's homes: rank 1 copies them in one batch, rank 0 copies nothing.
    assert ranks[0].transport.batches == [[]]
    (batch,) = ranks[1].transport.batches
    assert sorted(batch) == [
        (layer, slot, 0, expert) for layer in range(L) for slot, expert in ((4, 0), (5, 1))
    ]
    for record, runtime in zip(decided, ranks):
        assert sorted(record["effect"]) == [0, 1] and runtime.tables.any()
        assert runtime.applied_window.tolist() == [1, 1]
        counters = runtime.counters()  # what the executor reports in its iteration statistics
        assert counters["installs"] == 1 and counters["installedLayers"] == 2
        assert counters["windows"] >= 1


def test_no_boundary_runs_while_a_graph_is_captured(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    balancer = manager()
    calls = []
    balancer.runtime = SimpleNamespace(boundary=lambda: calls.append(1))
    balancer.enable_update_weights = True
    balancer.start_iter()
    balancer.end_iter()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    balancer.start_iter()
    assert calls == [1]


def window_stats(stats: np.ndarray, hot=(0, 1)) -> np.ndarray:
    """Every rank's window: its ``hot`` experts take 400 rows per layer, the others 1, over 8
    forwards; no layer is live yet; 1e5 tokens remain over 4 requests at 2 accepted per step."""
    stats[:, : L * E] = np.tile(np.where(np.isin(np.arange(E), hot), 400.0, 1.0), L)
    stats[:, L * E : L * (E + 1)] = 8
    stats[:, L * (E + 1) : L * (E + 2)] = -1
    totals = L * (E + 2) + rt.FIELDS.index("remaining")
    stats[:, totals : totals + 3] = [1e5, 4, 8]  # remaining, requests, accepted
    return stats


def test_updates_copy_once_and_a_cost_jump_does_not_hold_back_the_correction(monkeypatch):
    monkeypatch.setattr(rt, "solve_shares", even_split)
    decider = FakeDecision(0, copy_ms=5.0).decider  # copies of 5 ms per layer, all timed so
    decider.next_forwards = FORWARDS  # an earlier window measured the rows per forward
    # Window 0 moves both layers in one copy; window 1 plans no layer whose install is not live.
    # Window 1: GPU 1's experts turn hot before the install is live and the cost jumps; read as a
    # steady drift, the correction's gain would be gone within a dozen steps and repay no copy.
    for w, (h, live, moved) in enumerate(((0, -1, [0, 1]), (4, -1, []), (4, 0, [0, 1]))):
        window_stats(decider.stats, (h, h + 1))[:, L * (E + 1) : L * (E + 2)] = live
        update = decider.decide(w)
        assert update.moved == moved and sorted(update.copy_items) == moved
        decider.schedule(w, np.array([[10 + 20 * w, len(moved), 5.0 * len(moved), 0]] * R))


def test_first_update_moves_all_paying_layers_and_corrections_are_judged_out_of_sample(monkeypatch):
    monkeypatch.setattr(rt, "solve_shares", even_split)
    decision = FakeDecision(0, copy_ms=0.5)  # timed at start-up: the first update moves both layers
    window_stats(decision.stats.numpy())  # GPU 0's experts 0 and 1 are hot
    decision.decider.next_forwards = FORWARDS  # an earlier window measured the rows per forward
    assert decision.decider.decide(0).moved == [0, 1]
    window_stats(decision.stats.numpy(), (4, 5))[:, L * (E + 1) : L * (E + 2)] = 0  # installs live
    # GPU 1's experts hot: on window 0, the installed layouts' own, the correction would be held.
    assert decision.decider.decide(1).moved == [0, 1]


def test_a_candidate_must_gain_on_the_previous_window_too(monkeypatch):
    seen = {}
    spy(monkeypatch, seen)
    candidate = np.where(np.arange(R * LOCAL) == LOCAL + 4, 2, HOME)  # expert 2 copied to GPU 1
    monkeypatch.setattr(rt, "plan_layout", lambda *args: SimpleNamespace(layout=candidate))
    decision = FakeDecision(0, copy_ms=0.5)
    decision.decider.next_forwards = FORWARDS  # an earlier window measured the rows per forward

    def window(number, hot):
        window_stats(decision.stats.numpy(), (hot,))
        return decision.decider.decide(number)

    assert window(0, 6).tables == {}  # GPU 1 is the busiest: the copy would load it further
    assert window(1, 2).tables == {} and max(seen["gains"][-1].values()) < 0  # loses on window 0
    assert sorted(window(2, 2).tables) == [0, 1]  # gains on both windows: installed


def test_decision_process_answers_and_reports_its_failure():
    process = rt.DecisionProcess(rt.DecisionConfig(GEOMETRY, MODEL, np.tile(HOME, (L, 1)), 1, 1))
    try:
        window_stats(process.stats.numpy())
        # The first window holds fewer rows than a window collects: it only sizes the next ones,
        # to collect them at its 2 x (2 x 400 + 6) / 8 rows per forward.
        process.submit(("window", 0))
        assert process.recv() == ("update", rt.LayoutUpdate(0))
        rows = np.array([[10, 0, 0.0, 0], [12, 0, 0.0, 0]])  # ROW of both ranks
        process.submit(("copied", 0, rows))
        kind, window, open_call, apply_call, plan, forwards = process.recv()
        assert (kind, window, apply_call, open_call) == ("schedule", 1, 13, 13 + plan.open_after)
        assert forwards == np.ceil(rt.ROWS_PER_EXPERT * E / (2 * (2 * 400 + 6) / 8))
        process.submit(("window", 1))
        kind, update = process.recv()
        assert kind == "update" and update.window == 1
        # No copy was timed yet: the best layer alone is installed, to time one.
        assert update.moved == [0] and set(update.tables) == {0}
        assert all(t.dtype == np.int32 and t.size == TABLE for t in update.tables.values())
        process.submit(("copied", 5, rows))
        with pytest.raises(RuntimeError, match="KeyError"):
            process.recv()
    finally:
        process.close()


def test_shutdown_agrees_over_the_ep_group_then_stops_the_process_and_the_copies():
    calls = []
    for peer_safe, done in ((False, False), (True, True)):
        balancer = manager()
        balancer.runtime = SimpleNamespace(gathers_posted=3, close=calls.append)
        balancer.copy_stream = SimpleNamespace(synchronize=lambda: calls.append("stream"))
        balancer.transport = SimpleNamespace(allgather=lambda vote, p=peer_safe: [vote, (p, 4)])
        assert balancer.shutdown() is done and balancer.is_shutdown is done
    assert calls == [4, "stream"]  # the runtime closes at the most all-gathers any rank posted


@pytest.mark.parametrize("rank_0_failed", [False, True], ids=["rank_0_stopped", "rank_0_failed"])
def test_shutdown_completes_an_all_gather_that_only_some_ranks_posted(host_cuda, rank_0_failed):
    """Rank 1's worker waits in the all-gather of window 0's install, which rank 0 never posts:
    rank 0 stopped first, or its worker failed. Rank 0's shutdown runs it, and both workers end."""
    clock = [0]
    decisions = [FakeDecision(rank) for rank in range(R)]
    if rank_0_failed:
        decisions[0].script = {}.__getitem__  # the decision of window 0 fails
    ranks = make_ranks(decisions, clock)
    run(ranks, clock, 2)  # window 0 opens at call 0 and closes at call 1
    for runtime in ranks:  # call 2 posts the window's all-gather
        runtime.add_request_progress(*executor_progress([]))
        runtime.boundary()
    deadline = time.monotonic() + 30
    while not (ranks[1].worker.inbox and (ranks[0].worker.inbox or ranks[0].worker.failure)):
        assert time.monotonic() < deadline, "window 0 was not decided"
    for runtime in ranks[0 if rank_0_failed else 1 :]:  # call 3: rank 1 posts the install's
        runtime.boundary()
    votes = [(True, 1), (True, 2)]  # the all-gathers each rank posted
    for rank, runtime in enumerate(ranks):
        balancer = manager()
        balancer.runtime, balancer.copy_stream = runtime, SimpleNamespace(synchronize=lambda: None)

        def allgather(vote, r=rank):  # this rank's own vote among the others'
            return votes[:r] + [vote] + votes[r + 1 :]

        balancer.transport = SimpleNamespace(allgather=allgather)
        assert balancer.shutdown() is True and not runtime.worker.is_alive()
    assert ranks[1].worker.failure is None and ranks[1].worker.inbox[-1][0] == "copied"


_EXIT_SCRIPT = """
import os, sys, time
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import runtime as rt
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.balancer import ep_gather
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
    ExpertGeometry, TimeModel, statistics_buffers)

rank, store, exiting, peer_joins = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1"
dist.init_process_group("gloo", store=dist.FileStore(store, 2), rank=rank, world_size=2)
send, rows = torch.full((4,), float(rank)), [torch.empty(4) for _ in range(2)]
if rank == 1:  # a peer that reaches the same all-gather while rank 0 exits, or never does
    while not os.path.exists(exiting):
        time.sleep(0.01)
    time.sleep(0.5)
    if peer_joins:
        dist.all_gather(rows, send)
    while not os.path.exists(exiting + ".done"):  # rank 0 exited
        time.sleep(0.01)
    sys.exit(0)
rt.CLOSE_WAIT_S = 3.0  # longer than a slice of the all-gather's wait
torch.cuda.Event = lambda **kwargs: SimpleNamespace()
geometry = ExpertGeometry(8, 12, 2)
homes = np.tile(geometry.home_layout(), (2, 1))
config = rt.DecisionConfig(geometry, TimeModel(1, 1, 1, 0.5, 1, 128), homes, 0, 1)
decision = rt.DecisionProcess(config)
runtime = rt.AdaptiveRuntime(
    [None, None], None, None, decision, ep_gather(dist.group.WORLD),
    torch.device("cpu"), torch.zeros(2, 40, dtype=torch.int32), statistics_buffers(2, 8, "cpu"))
runtime.worker.post(lambda: runtime.gather(send, rows, runtime.leave))  # it waits in the all-gather
print("decision", decision.process.pid, flush=True)


class Teardown:
    def __del__(self):  # the seconds a serving process spends finalizing its model
        time.sleep(3)


teardown = Teardown()
open(exiting, "w").close()
"""


@pytest.mark.parametrize("peer_joins", ["1", "0"], ids=["peer_joins", "peer_never_joins"])
def test_a_runtime_left_open_ends_its_worker_before_the_interpreter_finalizes(tmp_path, peer_joins):
    """A rank that exits without a balancer shutdown, its worker inside an all-gather that a peer
    completes while the rank exits, or never: the worker must end before the interpreter
    finalizes, where a thread returning from the collective would abort the process, and the
    decision process must end with it."""
    script = tmp_path / "exit.py"
    script.write_text(_EXIT_SCRIPT)
    args = [str(tmp_path / "store"), str(tmp_path / "exiting"), peer_joins]
    # Two plain processes: no MPI or PMIx of a launcher, and no core file if one aborts.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMIX", "PMI_", "OMPI_"))}
    env["TLLM_DISABLE_MPI"] = "1"
    run = dict(env=env, preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
    peer = subprocess.Popen([sys.executable, str(script), "1", *args], **run)
    done = subprocess.run(
        [sys.executable, str(script), "0", *args],
        capture_output=True,
        text=True,
        timeout=120,
        **run,
    )
    (tmp_path / "exiting.done").touch()
    peer.wait(timeout=60)
    assert done.returncode == 0, done.stderr[-4000:]
    assert "did not end" not in done.stdout + done.stderr
    pid = int(done.stdout.split("decision", 1)[1].split()[0])
    for _ in range(100):
        if not os.path.exists(f"/proc/{pid}"):
            break
        time.sleep(0.1)
    assert not os.path.exists(f"/proc/{pid}"), "the decision process outlived its rank"


def test_a_window_prices_at_the_scale_of_the_bucket_its_busiest_rank_pads_to(monkeypatch):
    gains = []
    for scales in ((), ((100.0, 2.0), (200.0, 3.0))):
        seen = {}
        spy(monkeypatch, seen)
        decider = FakeDecision(0).decider
        decider.config = dataclasses.replace(
            decider.config, model=dataclasses.replace(MODEL, scales=scales)
        )
        decider.next_forwards = FORWARDS  # an earlier window measured the rows per forward
        window_stats(decider.stats)  # every rank: 2 x 400 + 6 rows per layer over 8 forwards
        decider.decide(0)
        gains.append(seen["gains"][0])
    # 100.75 rows per forward on the busiest rank pad to the 200-row bucket: its scale 3.
    assert gains[1] == pytest.approx({layer: 3.0 * gain for layer, gain in gains[0].items()})
