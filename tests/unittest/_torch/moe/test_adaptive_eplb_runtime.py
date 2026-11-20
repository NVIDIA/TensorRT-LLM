# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the adaptive EPLB runtime: statistics windows, the decision process and installs."""

import contextlib
import dataclasses
import functools
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
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, generation_progress
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

pytestmark = pytest.mark.cpu_only

E, R, LOCAL, L = 8, 2, 6, 2
GEOMETRY = ExpertGeometry(E, R * LOCAL, R)
MODEL = TimeModel(1.0, 1.0, 1.0, 0.5, 1.0, 128)
HOME = GEOMETRY.home_layout()
TABLE = 2 * (E + R * LOCAL)
ROWS = 512  # rows a window collects per expert: rows_per_expert's default
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
    """A 0.25 ms copy batch waiting until its issuing call plus ``delay``."""

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

    def __init__(self, rank: int, script=None, copy_ms=None, model=MODEL, open_after=4) -> None:
        self.stats = torch.zeros(R, rt.window_stats_size(L, E))
        homes = np.tile(HOME, (L, 1))
        config = rt.DecisionConfig(GEOMETRY, model, homes, rank, 1, ROWS, copy_ms=copy_ms)
        self.decider = rt._Decider(config, self.stats.numpy())
        self.script = script
        self.open_after = open_after  # calls from a scripted apply call to the next window's open
        self.sent = []

    def ask(self, message):
        self.sent.append(message)
        if self.script is None:
            return self.decider.handle(message)
        if message[0] == "window":
            return ("update", self.script(message[1]))
        apply_call = int(message[2][:, rt.ROW.index("copied_call")].max()) + 1
        open_call = apply_call + self.open_after
        return ("schedule", message[1] + 1, open_call, apply_call, None, FORWARDS)

    def close(self) -> None:
        pass


class Events:
    """Collect one rank's decisions and their later completion events."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.records, self.stats = [], []
        self.by_window, self.open_calls = {}, {}

    def __call__(self, event: str, **data) -> None:
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
    _OPEN_RANKS.append(ranks)
    return ranks


# The ranks make_ranks returned that the running test has not closed yet.
_OPEN_RANKS: list[list] = []


def close_ranks(ranks) -> None:
    """Close every rank's runtime as the balancer's shutdown does: concurrently, each owing the
    all-gathers the most advanced rank posted, so that every worker thread ends."""
    gathers = max(runtime.gathers_posted for runtime in ranks)
    closers = [threading.Thread(target=runtime.close, args=(gathers,)) for runtime in ranks]
    for closer in closers:
        closer.start()
    for closer in closers:
        closer.join()
    assert not any(runtime.worker.is_alive() for runtime in ranks), "a worker outlived its close"


def closes_runtimes(test):
    """End the worker threads of the runtimes ``test`` makes before it returns: pytest-threadleak
    judges the test call itself, before any fixture teardown."""

    @functools.wraps(test)
    def closing(*args, **kwargs):
        opened = len(_OPEN_RANKS)
        try:
            return test(*args, **kwargs)
        finally:
            while len(_OPEN_RANKS) > opened:
                close_ranks(_OPEN_RANKS.pop())

    return closing


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


def run(ranks, clock, calls: int, requests=((), ())) -> None:
    """Execute per-rank batches and advance the clock after every boundary."""
    for _ in range(calls):
        for runtime, batch in zip(ranks, requests):
            if runtime.wants_request_progress:
                runtime.add_request_progress(*executor_progress([request(*i) for i in batch]))
            runtime.boundary()
        clock[0] = ranks[0].call
        settle(ranks)


def set_statistics(runtime, rows: np.ndarray, forwards: int) -> None:
    for layer, counts in zip(runtime.layers, rows):
        layer.statistics.rows.copy_(torch.from_numpy(counts))
        layer.statistics.forwards.fill_(forwards)


def even_split(rows, layout, geometry, model=None) -> np.ndarray:
    """Shares that split every expert's rows evenly over its copies."""
    copies = np.bincount(layout[layout >= 0], minlength=geometry.num_experts)
    return np.where(layout >= 0, 1.0 / copies[np.maximum(layout, 0)], 0.0)


# The runtime's own functions: a test that spies twice must not spy on its first spy.
REAL_PLAN, REAL_CHOOSE = rt.plan_layout, rt.choose_layers_to_install


def spy(monkeypatch, seen: dict) -> None:
    """Capture decision inputs and use ``FORWARDS`` after the first window."""

    def plan(load, *args):
        seen.setdefault("load", []).append(np.array(load))
        return REAL_PLAN(load, *args)

    def solve(rows, layout, geometry, model=None):
        seen.setdefault("rows", []).append(rows)
        return even_split(rows, layout, geometry)

    def choose(gains, steps_left, copy_cost):
        seen.setdefault("gains", []).append(dict(gains))
        seen.setdefault("steps_left", []).append(steps_left)
        return REAL_CHOOSE(gains, steps_left, copy_cost)

    def window_forwards(rows_per_forward, experts, rows_per_expert):
        seen.setdefault("per_forward", []).append((rows_per_forward, experts))
        return FORWARDS

    monkeypatch.setattr(rt, "plan_layout", plan)
    monkeypatch.setattr(rt, "solve_shares", solve)
    monkeypatch.setattr(rt, "choose_layers_to_install", choose)
    monkeypatch.setattr(rt, "window_forwards", window_forwards)


def spied_ranks(monkeypatch, timed: bool = False, first=None):
    """Observe two in-process deciders after their first statistics window."""
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


def request(max_new_tokens, decoding_iter, accepted_draft_tokens, dummy=False, beams=1):
    req = LlmRequest(
        request_id=1,
        max_new_tokens=max_new_tokens,
        input_tokens=[7] * 5,
        sampling_config=SamplingConfig(beam_width=beams),
        is_streaming=False,
    )
    for _ in range(decoding_iter * (1 + accepted_draft_tokens)):
        req.add_new_tokens([9] * beams)
    req.py_decoding_iter, req.py_num_accepted_draft_tokens = decoding_iter, accepted_draft_tokens
    req.is_attention_dp_dummy = dummy
    return req


@pytest.mark.parametrize("history", [(1, 4), (4, 1), (1,), (2, 2)])
def test_progress_uses_confirmed_tokens_across_beams_and_recompute(history):
    req = request(20, 0, 0)
    for count in history:
        for _ in range(count):
            req.add_new_token(9, 0)
        req.py_decoding_iter += 1
        req.py_num_accepted_draft_tokens = count - 1
    left = 20 - sum(history)
    assert generation_progress([req], 0, 1) == (left, 1, history[-1])
    req, left = request(20, 3, 0, beams=2), 17
    req.add_new_token(11, 1)
    assert generation_progress([req], 0, 1) == (left - 1, 1, 1)
    for _ in range(left - 1):
        req.add_new_token(11, 1)
    assert generation_progress([req], 0, 1) == (0, 1, 1)
    req.add_new_token(11, 1)
    assert generation_progress([req], 0, 1) == (0, 1, 1)
    req.reset_for_recompute(64)
    assert (req.py_orig_prompt_len, req.py_max_new_tokens) == (5, 20)
    assert generation_progress([req], 0, 1) == (20, 1, 1)
    req.is_cuda_graph_dummy = True
    assert generation_progress([req], 0, 1) == (0, 0, 0)
    req.is_cuda_graph_dummy = False
    req.is_attention_dp_dummy = True
    assert generation_progress([req], 0, 1) == (0, 0, 0)
    req.is_attention_dp_dummy = False
    assert generation_progress([req], 1, 1) == (0, 0, 0)
    req = request(20, 0, 0)
    for _ in range(5):
        req.add_new_token(9, 0)
    req.reset_for_recompute(64)
    assert (req.py_orig_prompt_len, req.py_max_new_tokens) == (10, 15)
    req.add_new_token(9, 0)  # The first token imported by a PD handoff is already confirmed.
    req.py_decoding_iter = 1
    assert generation_progress([req], 0, 1) == (14, 1, 1)


def window_inputs(monkeypatch, forwards=(FORWARDS,) * R):
    seen, clock, decisions, ranks = spied_ranks(monkeypatch)
    rows = np.random.default_rng(7).integers(0, 64, (R, L, E)).astype(np.float32)
    progress = ([(120, 10, 1), (70, 10, 1), (1, 1, 0, True)], [(75, 15, 2)])
    for runtime, counts, count in zip(ranks, rows, forwards):
        set_statistics(runtime, counts, count)
    run(ranks, clock, FORWARDS - 1)
    assert all(runtime.wants_request_progress for runtime in ranks)
    run(ranks, clock, 2, progress)
    return seen, rows, decisions, ranks


@closes_runtimes
def test_a_window_counts_every_ranks_rows_over_its_own_forwards(host_cuda, monkeypatch):
    """Normalize each rank by its own forwards without changing the request budget."""
    base, _, _, base_ranks = window_inputs(monkeypatch)
    seen, rows, _, ranks = window_inputs(monkeypatch, (FORWARDS, FORWARDS + 1))
    expected = rows[0].astype(np.float64) / FORWARDS + rows[1].astype(np.float64) / (FORWARDS + 1)
    # Both ranks plan and solve each layer; their worker calls can interleave.
    for name, per_layer in (("load", R), ("rows", 2 * R)):
        assert sorted(map(list, seen[name])) == sorted(map(list, [*expected] * per_layer))
    last = [runtime._event.records[-1] for runtime in (*ranks, base_ranks[0])]
    for key in ("window", "open_call", "solved_call"):
        assert last[0][key] == last[1][key] == last[2][key]
    # 100 + 50 + 30 tokens remain over 3 requests (the dummy does not count) at 7 / 3 per step.
    assert seen["steps_left"] == base["steps_left"] == [180 / (7 / 3) / 3] * R


@closes_runtimes
def test_the_first_window_lasts_one_call_and_sizes_the_next(host_cuda, monkeypatch):
    """Size future windows from rows per forward; decide only after enough rows."""
    for per_rank, planned in ((ROWS // R - 1, 0), (ROWS // R, R * L)):
        rows = np.full((R, L, E), per_rank, dtype=np.float32)
        seen, _, _, ranks = spied_ranks(monkeypatch, first=(rows, (1,) * R))
        assert len(seen.get("load", [])) == planned
        assert seen["per_forward"] == [(R * per_rank * E, E)] * R
        for runtime in ranks:
            (first,) = runtime._event.records
            # It closed and was gathered at call 1 and decided at call 2.
            assert (first["open_call"], first["solved_call"], first["tables"]) == (0, 2, [])
            assert first["forwards"] == FORWARDS
            assert runtime.close_at - runtime.window_opened_call == FORWARDS


@closes_runtimes
def test_a_rank_without_a_forward_skips_the_window(host_cuda, monkeypatch):
    seen, clock, _, ranks = spied_ranks(monkeypatch)
    for runtime, forwards in zip(ranks, (FORWARDS, 0)):
        set_statistics(runtime, np.ones((L, E), dtype=np.float32), forwards)
    run(ranks, clock, FORWARDS + 4)
    assert "load" not in seen
    last = [runtime._event.records[-1] for runtime in ranks]
    assert all(record["window"] == 1 and record["tables"] == [] for record in last)
    # Nothing was learnt: the next window opens the call after the skipped one's apply call.
    assert [record["next_open"] - record["apply_call"] for record in last] == [1, 1]


def test_layers_not_live_on_every_rank_are_not_planned(monkeypatch):
    seen = {}
    spy(monkeypatch, seen)
    decider = measured_decider()
    window_stats(decider.stats, (), live=[[0, -1], [-1, -1]])  # rank 1 runs no install of layer 0
    decider.committed[:] = [0, -1]
    decider.decide(0)
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
@closes_runtimes
def test_tables_apply_after_every_rank_copied_and_the_window_opens_as_scheduled(
    host_cuda, delays, open_after
):
    """Apply after the last copy; read schedules before opening their windows."""
    clock = [0]
    decisions = [FakeDecision(rank, script(rank), open_after=open_after) for rank in range(R)]
    ranks = make_ranks(decisions, clock, copy_delays=delays)
    while not all(runtime.window == 1 and runtime.close_at for runtime in ranks):
        assert clock[0] < 32, "the second window did not open"
        run(ranks, clock, 1)
    first = [runtime._event.records[0] for runtime in ranks]
    solved = first[0]["solved_call"]
    copied = [r["copied_call"] for r in first]
    assert first[1]["solved_call"] == solved and copied == [solved + d for d in delays]
    apply = copied[1] + 1
    assert [r["apply_call"] for r in first] == [apply, apply]
    assert [r["effect"] for r in first] == [{0: apply, 1: apply}] * 2
    for runtime, decision, record in zip(ranks, decisions, first):
        assert runtime.tables.tolist() == [[1] * TABLE, [2] * TABLE]
        (stats,) = runtime._event.stats
        assert np.array_equal(stats, decision.stats.numpy())
        assert runtime.window_opened_call == record["next_open"] == apply + open_after
        assert runtime.applied_window.tolist() == [0, 0]
    assert ranks[0].transport.batches == [[]] and ranks[1].transport.batches == [[(0, 4, 0, 0)]]


@closes_runtimes
def test_ranks_that_decided_differently_stop_before_applying(host_cuda):
    """Regression: ranks deciding differently stop before applying. A digest disagreement stops
    adaptive updates without applying any tables."""
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


@closes_runtimes
def test_a_failed_worker_stops_adaptive_eplb_and_keeps_the_tables(host_cuda):
    """Regression: a failed worker keeps the tables. A worker failure stops adaptive updates at
    the next boundary, preserving tables."""
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


@closes_runtimes
def test_two_ranks_move_an_expert_and_apply(host_cuda, monkeypatch):
    """Two real deciders agree on tables and digests, with rank-specific copy items."""
    _, clock, decisions, ranks = spied_ranks(monkeypatch, timed=True)
    hot = np.tile([400.0, 400.0] + [1.0] * (E - 2), (L, 1)).astype(np.float32)
    for runtime in ranks:
        set_statistics(runtime, hot, FORWARDS)
    run(ranks, clock, FORWARDS + 12, [[(1000, 10, 0)] * 4] * R)
    decided = [runtime._event.by_window[1] for runtime in ranks]
    assert all(record["moved"] == [0, 1] for record in decided)
    assert decided[0]["apply_call"] == decided[1]["apply_call"]
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
    for _ in range(2):  # two more windows, gathered after the install
        for runtime in ranks:
            set_statistics(runtime, hot, FORWARDS)
        run(ranks, clock, 22)
    sent = [m[1] for m in decisions[0].sent if m[0] == "window"]
    assert sent == list(range(len(sent))) and len(sent) >= 3
    pending = [m[1].window for m in ranks[0].worker.inbox if m[0] == "update"]  # read next call
    assert [r["window"] for r in ranks[0]._event.records] + pending == sent


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


def window_stats(stats: np.ndarray, hot=(0, 1), live=-1, remaining=1e5, requests=4) -> np.ndarray:
    """Count 400 hot/1 cold rows per rank over 8 forwards, with 8 accepted tokens per step."""
    stats[:, : L * E] = np.tile(np.where(np.isin(np.arange(E), hot), 400.0, 1.0), L)
    stats[:, L * E : L * (E + 1)] = 8
    stats[:, L * (E + 1) : L * (E + 2)] = live
    totals = L * (E + 2) + rt.FIELDS.index("remaining")
    stats[:, totals : totals + 3] = [remaining, requests, 8]  # remaining, requests, accepted
    return stats


def measured_decider(copy_ms=None, model=MODEL) -> rt._Decider:
    """Rank 0's decider once an earlier window measured the rows per forward."""
    decider = FakeDecision(0, copy_ms=copy_ms, model=model).decider
    decider.next_forwards = FORWARDS
    return decider


def after_window_0(ms=1.0, **window) -> rt._Decider:
    """Replicate hot experts 0/1 at a 0.5 ms prior, then record both copies done at call 10."""
    decider = measured_decider(0.5)
    window_stats(decider.stats, **window)
    assert decider.decide(0).moved == [0, 1]
    decider.schedule(0, np.array([[10, 2, ms, 0]] * R))
    return decider


def test_updates_copy_once_and_a_cost_jump_does_not_hold_back_the_correction(monkeypatch):
    monkeypatch.setattr(rt, "solve_shares", even_split)
    decider = measured_decider(5.0)  # copies of 5 ms per layer, all timed so
    # Window 0 moves both layers in one copy; window 1 plans no layer whose install is not live.
    # A cost jump before the install is live must not be mistaken for steady gain decay.
    for w, (h, live, moved) in enumerate(((0, -1, [0, 1]), (4, -1, []), (4, 0, [0, 1]))):
        window_stats(decider.stats, (h, h + 1), live)
        update = decider.decide(w)
        assert update.moved == moved and sorted(update.copy_items) == moved
        decider.schedule(w, np.array([[10 + 20 * w, len(moved), 5.0 * len(moved), 0]] * R))


def test_a_candidate_must_gain_on_the_previous_window_too(monkeypatch):
    seen = {}
    spy(monkeypatch, seen)
    candidate = np.where(np.arange(R * LOCAL) == LOCAL + 4, 2, HOME)  # expert 2 copied to GPU 1
    monkeypatch.setattr(rt, "plan_layout", lambda *args: SimpleNamespace(layout=candidate))
    decider = measured_decider(0.5)

    def window(number, *hot):
        window_stats(decider.stats, hot)
        return decider.decide(number)

    assert window(0, 6).tables == {}  # GPU 1 is the busiest: the copy would load it further
    assert window(1, 2).tables == {} and max(seen["gains"][-1].values()) < 0  # loses on window 0
    assert sorted(window(2, 2).tables) == [0, 1]  # gains on both windows: installed
    # Expert 6 favors a copy now, but home beats it on the previous window: choose home.
    candidate[:] = np.where(np.arange(R * LOCAL) == 4, 6, HOME)
    assert window(3, *range(E)).tables == {}
    window_stats(decider.stats, range(E), live=2)[:, 6 : L * E : E] = 1600
    update = decider.decide(4)
    assert sorted(update.tables) == [0, 1] and update.moved == []
    assert (decider.layout == HOME).all()


def test_layers_go_home_without_a_copy_once_their_rows_even_out(monkeypatch):
    monkeypatch.setattr(rt, "solve_shares", even_split)
    decider = after_window_0()
    # Even rows make installed replicas cost extra: retire them without a copy.
    for w in (1, 2):
        window_stats(decider.stats, range(E), live=w - 1)
        update = decider.decide(w)
        assert sorted(update.tables) == ([0, 1] if w == 1 else []) and update.moved == []
        assert not update.copy_items and (decider.layout == HOME).all()
        decider.schedule(w, np.array([[10 + 20 * w, 0, 0.0, 0]] * R))


@pytest.mark.parametrize("before,after,ahead", [(40, 2, 0.2), (48, 121.5, 243.0)])
def test_a_draining_batch_prices_the_rows_one_window_ahead(monkeypatch, before, after, ahead):
    monkeypatch.setattr(rt, "solve_shares", even_split)
    decider = after_window_0(requests=before)
    # A rise from 96 to 243 requests is priced at the 243 requests at the close, not carried past them.
    window_stats(decider.stats, live=0, requests=after)
    update = decider.decide(1)
    assert decider.kept[-1][2] == pytest.approx(ahead)
    if after > before:
        return
    # A fall from 80 to 4 requests forecasts too few rows to repay replica weight streams.
    assert sorted(update.tables) == [0, 1] and update.moved == [] and (decider.layout == HOME).all()


def test_shares_solved_again_on_the_installed_layouts_are_published_without_a_copy(monkeypatch):
    decider = after_window_0()  # the share op itself, by rows
    old, installed = decider.shares.copy(), decider.layout.copy()
    # One step left cannot repay copying warm expert 4; refresh the installed shares.
    candidate = np.where(np.arange(R * LOCAL) == 4, 4, HOME)
    monkeypatch.setattr(rt, "plan_layout", lambda *args: SimpleNamespace(layout=candidate))
    window_stats(decider.stats, live=0, remaining=8)[:, 4 : L * E : E] = 200
    update = decider.decide(1)
    assert sorted(update.tables) == [0, 1] and update.moved == [] and not update.copy_items
    assert (decider.layout == installed).all() and not np.array_equal(decider.shares, old)


def test_one_slow_copy_does_not_price_the_next_installs_out(monkeypatch):
    """Regression: one slow copy priced every later install out (copy price taken as a mean).
    Copies timed in service price the installs after them."""
    monkeypatch.setattr(rt, "solve_shares", even_split)
    # Priced at start-up only (no copy timed yet), the correction is judged out of sample.
    decider = measured_decider(0.5)
    window_stats(decider.stats)  # GPU 0's experts 0 and 1 are hot
    assert decider.decide(0).moved == [0, 1]
    window_stats(decider.stats, (4, 5), live=0)  # GPU 1's experts hot
    assert decider.decide(1).moved == [0, 1]
    costs = []
    monkeypatch.setattr(
        rt, "choose_layers_to_install", lambda *args: costs.append(args[2]) or REAL_CHOOSE(*args)
    )
    # 100 steps left; a replica took 0.5 ms at start-up, then window 0's copies took 100 ms (one
    # outlier) or 4 ms. GPU 1's experts hot: the correction still repays.
    for ms in (100.0, 4.0):
        costs.clear()
        decider = after_window_0(ms, remaining=800)
        window_stats(decider.stats, (4, 5), live=0, remaining=800)
        assert decider.decide(1).moved == [0, 1], ms
    decider.schedule(1, np.array([[30, 2, 4.0, 0]] * R))  # 1 ms per replica again
    window_stats(decider.stats, live=1, remaining=8)  # one step left: both copies are past it
    decider.decide(2)
    assert costs[-1] == {layer: 2 * cost for layer, cost in costs[0].items()}


def test_decision_process_answers_and_reports_its_failure():
    process = rt.DecisionProcess(
        rt.DecisionConfig(GEOMETRY, MODEL, np.tile(HOME, (L, 1)), 1, 1, ROWS)
    )
    try:
        window_stats(process.stats.numpy())
        # The first window holds fewer rows than a window collects: it only sizes the next ones,
        # to collect them at its 2 x (2 x 400 + 6) / 8 rows per forward.
        assert process.ask(("window", 0)) == ("update", rt.LayoutUpdate(0))
        rows = np.array([[10, 0, 0.0, 0], [12, 0, 0.0, 0]])  # ROW of both ranks
        kind, window, open_call, apply_call, plan, forwards = process.ask(("copied", 0, rows))
        assert (kind, window, apply_call, open_call) == ("schedule", 1, 13, 13 + plan.open_after)
        assert forwards == np.ceil(ROWS * E / (2 * (2 * 400 + 6) / 8))
        kind, update = process.ask(("window", 1))
        assert kind == "update" and update.window == 1
        # No copy was timed yet: the best layer alone is installed, to time one.
        assert update.moved == [0] and set(update.tables) == {0}
        assert all(t.dtype == np.int32 and t.size == TABLE for t in update.tables.values())
        with pytest.raises(RuntimeError, match="KeyError"):
            process.ask(("copied", 5, rows))
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
@closes_runtimes
def test_shutdown_completes_an_all_gather_that_only_some_ranks_posted(host_cuda, rank_0_failed):
    """Shutdown completes the all-gather where rank 1 waits for stopped rank 0."""
    clock = [0]
    decisions = [FakeDecision(rank) for rank in range(R)]
    if rank_0_failed:
        decisions[0].script = {}.__getitem__  # the decision of window 0 fails
    ranks = make_ranks(decisions, clock)
    run(ranks, clock, 1)  # window 0 opens at call 0
    for runtime in ranks:  # call 1 closes it and posts its all-gather
        runtime.boundary()
    deadline = time.monotonic() + 30
    while not (ranks[1].worker.inbox and (ranks[0].worker.inbox or ranks[0].worker.failure)):
        assert time.monotonic() < deadline, "window 0 was not decided"
    for runtime in ranks[0 if rank_0_failed else 1 :]:  # call 2: rank 1 posts the install's
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
import atexit, os, sys, time
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
    while not os.path.exists(exiting + ".atexit"):  # rank 0 is in its exit handlers
        time.sleep(0.01)
    if peer_joins:
        dist.all_gather(rows, send)
    while not os.path.exists(exiting + ".done"):  # rank 0 exited
        time.sleep(0.01)
    sys.exit(0)
rt.CLOSE_WAIT_S = 3.0  # longer than a slice of the all-gather's wait
torch.cuda.Event = lambda **kwargs: SimpleNamespace()
geometry = ExpertGeometry(8, 12, 2)
homes = np.tile(geometry.home_layout(), (2, 1))
config = rt.DecisionConfig(geometry, TimeModel(1, 1, 1, 0.5, 1, 128), homes, 0, 1, 512)
decision = rt.DecisionProcess(config)
runtime = rt.AdaptiveRuntime(
    [None, None], None, None, decision, ep_gather(dist.group.WORLD),
    torch.device("cpu"), torch.zeros(2, 40, dtype=torch.int32), statistics_buffers(2, 8, "cpu"))
runtime.worker.post(lambda: runtime.gather(send, rows, runtime.leave))  # it waits in the all-gather
print("decision", decision.process.pid, flush=True)
atexit.register(lambda: open(exiting + ".atexit", "w").close())  # after the runtime's: runs first


class Teardown:
    def __del__(self):  # a load, not an order: the seconds a server spends finalizing its model
        time.sleep(3)


teardown = Teardown()
open(exiting, "w").close()
"""


@pytest.mark.parametrize("peer_joins", ["1", "0"], ids=["peer_joins", "peer_never_joins"])
def test_a_runtime_left_open_ends_its_worker_before_the_interpreter_finalizes(tmp_path, peer_joins):
    """Regression: worker-thread leak/close. End workers and decision processes before exit,
    including unfinished collectives."""
    script = tmp_path / "exit.py"
    script.write_text(_EXIT_SCRIPT)
    args = [str(tmp_path / "store"), str(tmp_path / "exiting"), peer_joins]
    # Two plain processes: no MPI or PMIx of a launcher, and no core file if one aborts.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMIX", "PMI_", "OMPI_"))}
    env["TLLM_DISABLE_MPI"] = "1"
    run = dict(env=env, preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
    cmd = [sys.executable, str(script)]
    peer = subprocess.Popen([*cmd, "1", *args], **run)
    done = subprocess.run([*cmd, "0", *args], capture_output=True, text=True, timeout=120, **run)
    (tmp_path / "exiting.done").touch()
    peer.wait(timeout=60)
    assert done.returncode == 0, done.stderr[-4000:]
    assert "did not end" not in done.stdout + done.stderr
    pid = int(done.stdout.split("decision", 1)[1].split()[0])
    deadline = time.monotonic() + 10
    while os.path.exists(f"/proc/{pid}") and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not os.path.exists(f"/proc/{pid}"), "the decision process outlived its rank"


def test_a_window_prices_at_the_scale_of_the_bucket_its_busiest_rank_pads_to(monkeypatch):
    gains = []
    read = MODEL.per_active_slot_us  # the buckets differ in their scales alone
    for scales in ((), ((100.0, 2.0, read, 0.0, False), (200.0, 3.0, read, 0.0, False))):
        seen = {}
        spy(monkeypatch, seen)
        decider = measured_decider(model=dataclasses.replace(MODEL, scales=scales))
        window_stats(decider.stats)  # every rank: 2 x 400 + 6 rows per layer over 8 forwards
        decider.decide(0)
        gains.append(seen["gains"][0])
    # 100.75 rows per forward on the busiest rank pad to the 200-row bucket: its scale 3.
    assert gains[1] == pytest.approx({layer: 3.0 * gain for layer, gain in gains[0].items()})
    # That bucket failed its calibration's hold-out check: no trusted model, no decision there.
    scales = ((100.0, 2.0, read, 0.0, False), (200.0, 3.0, read, 0.0, False, 8, False))
    decider = measured_decider(model=dataclasses.replace(MODEL, scales=scales))
    window_stats(decider.stats)
    update = decider.decide(0)
    assert update.moved == [] and not update.copy_items


@closes_runtimes
def test_statistics_count_only_inside_the_windows_the_runtime_schedules(host_cuda, monkeypatch):
    """The recording flag is set from a window's open to its close and clear between windows, on the
    runtime's own schedule; the counts a window closes with equal the counts of always-on recording
    since its open, and the calls between windows count nothing."""
    balancer = manager()
    flag = balancer.recording_flag("cpu")
    balancer.observe_scheduled_batch(4, True, False, lambda start, stop: (0, 0, 0))
    assert balancer.recording and flag.item() == 1, (
        "without a runtime every recorded forward counts"
    )
    seen, clock, _, ranks = spied_ranks(monkeypatch)
    runtime = ranks[0]
    runtime.on_window = balancer._set_window_open
    per_call = torch.arange(E, dtype=torch.float32)
    gated = torch.zeros(E)
    always = torch.zeros(E)
    opened_calls, closed = [], []
    for _ in range(4 * FORWARDS + 8):
        was_open = runtime.close_at is not None
        run(ranks, clock, 1)
        if runtime.close_at is not None and not was_open:
            opened_calls.append(runtime.call)
            gated.zero_()
            always.zero_()
        if was_open and runtime.close_at is None and opened_calls:
            # The window closed at this call: its forwards were the calls since its open.
            closed.append((gated.clone(), always.clone()))
        # The forward of this call: the statistics ops add the routes while the flag is set.
        always += per_call
        if flag.item():
            gated += per_call
        if runtime.close_at is None:
            assert flag.item() == 0, f"call {runtime.call} counts outside a window"
        elif opened_calls:
            assert flag.item() == 1, f"call {runtime.call} of an open window does not count"
    assert opened_calls and closed
    for counted, reference in closed:
        assert torch.equal(counted, reference)
    balancer.observe_scheduled_batch(0, True, False, lambda start, stop: (0, 0, 0))
    assert flag.item() == 0
