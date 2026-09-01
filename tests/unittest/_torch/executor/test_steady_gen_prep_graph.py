# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replaying a generation-step prepare's device work from a captured graph."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.steady_gen_prep_graph import (
    RECHECK_STEPS,
    WARMUP_STEPS,
    SteadyGenPrepGraph,
)
from tensorrt_llm._torch.utils import run_device_work

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for graph capture"
)

KEY = ("batch", 1)


class _FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


def _step(graph, calls, key=KEY, extra=0):
    """One prepare: collect two operations (plus `extra`), then issue them."""
    work = graph.begin(key)
    try:
        run_device_work(calls.append, "a")
        run_device_work(calls.append, "b")
        for i in range(extra):
            run_device_work(calls.append, f"x{i}")
    finally:
        graph.end()
    return graph.issue(work)


@pytest.fixture
def captures(monkeypatch):
    """Capture into fake graphs; returns the list of captured work lists."""
    taken = []

    def capture(self, work):
        taken.append(list(work))
        return _FakeGraph()

    monkeypatch.setattr(SteadyGenPrepGraph, "_capture", capture)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return taken


def test_empty_work_is_not_captured(captures):
    graph = SteadyGenPrepGraph()
    for _ in range(WARMUP_STEPS + 2):
        work = graph.begin(KEY)
        graph.end()
        assert graph.issue(work) is False
    assert captures == []
    assert graph.enabled


def test_warmup_steps_are_issued_not_replayed(captures):
    graph = SteadyGenPrepGraph()
    calls = []
    for step in range(WARMUP_STEPS):
        assert _step(graph, calls) is False
        assert calls == ["a", "b"] * (step + 1)
    assert captures == []


def test_captures_once_and_replays_afterwards(captures):
    graph = SteadyGenPrepGraph()
    calls = []
    for _ in range(WARMUP_STEPS):
        _step(graph, calls)
    issued_during_warmup = list(calls)

    for _ in range(3):
        assert _step(graph, calls) is True
    # A replay does not re-issue the operations host-side.
    assert calls == issued_during_warmup
    assert len(captures) == 1
    assert captures[0] == [(calls.append, ("a",), {}), (calls.append, ("b",), {})]


def test_a_replaying_step_does_not_collect(captures):
    graph = SteadyGenPrepGraph()
    calls = []
    for _ in range(WARMUP_STEPS + 1):
        _step(graph, calls)
    # Past warmup and capture, the region no longer records anything: that is
    # the host cost the replay is meant to remove.
    work = graph.begin(KEY)
    graph.end()
    assert work is None
    assert graph.issue(work) is True


def test_a_recheck_step_records_and_still_replays(captures):
    graph = SteadyGenPrepGraph()
    calls = []
    for _ in range(WARMUP_STEPS + 1):
        _step(graph, calls)
    issued = list(calls)

    recorded_at = []
    for step in range(WARMUP_STEPS + 2, RECHECK_STEPS + 2):
        work = graph.begin(KEY)
        try:
            run_device_work(calls.append, "a")
            run_device_work(calls.append, "b")
        finally:
            graph.end()
        assert graph.issue(work) is True
        if work is not None:
            recorded_at.append(step)
    assert recorded_at == [RECHECK_STEPS]
    # A recheck step replays what it recorded rather than issuing it.
    assert calls == issued
    assert len(captures) == 1


def test_a_changed_sequence_is_caught_and_re_captured(captures):
    graph = SteadyGenPrepGraph()
    calls = []
    for _ in range(WARMUP_STEPS + 1):
        _step(graph, calls)
    assert len(captures) == 1
    issued = list(calls)

    # Drive the next recheck step with one operation more than was captured.
    for _ in range(RECHECK_STEPS - (WARMUP_STEPS + 1) - 1):
        _step(graph, calls)
    replayed = _step(graph, calls, extra=1)

    # The mismatch is caught: the step's work is issued rather than replayed,
    # and the stale graph is gone.
    assert replayed is False
    assert calls == issued + ["a", "b", "x0"]

    # ... and the new sequence is captured after another warmup.
    for _ in range(WARMUP_STEPS):
        _step(graph, calls, extra=1)
    assert _step(graph, calls, extra=1) is True
    assert len(captures) == 2
    assert len(captures[1]) == 3


def test_a_new_key_re_warms_and_re_captures(captures):
    graph = SteadyGenPrepGraph()
    calls = []
    for _ in range(WARMUP_STEPS + 1):
        _step(graph, calls)
    assert len(captures) == 1

    other = ("batch", 2)
    for _ in range(WARMUP_STEPS):
        assert _step(graph, calls, key=other) is False
    assert len(captures) == 1
    assert _step(graph, calls, key=other) is True
    assert len(captures) == 2


def test_an_alternating_key_never_captures(captures):
    graph = SteadyGenPrepGraph()
    calls = []
    for step in range(8):
        assert _step(graph, calls, key=("batch", step % 2)) is False
    assert captures == []
    assert calls == ["a", "b"] * 8


def test_a_failed_capture_issues_the_work_and_stays_eager(monkeypatch):
    def fail(self, work):
        raise RuntimeError("no capture here")

    monkeypatch.setattr(SteadyGenPrepGraph, "_capture", fail)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    graph = SteadyGenPrepGraph()
    calls = []
    for _ in range(WARMUP_STEPS + 3):
        assert _step(graph, calls) is False
    # Nothing is skipped: a capture records without running, so the step whose
    # capture failed still has to issue its work.
    assert calls == ["a", "b"] * (WARMUP_STEPS + 3)
    assert not graph.enabled


def test_work_is_issued_while_another_capture_is_open(monkeypatch):
    monkeypatch.setattr(SteadyGenPrepGraph, "_capture", lambda self, work: pytest.fail("captured"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    graph = SteadyGenPrepGraph()
    calls = []
    for _ in range(WARMUP_STEPS + 2):
        assert _step(graph, calls) is False
    assert calls == ["a", "b"] * (WARMUP_STEPS + 2)
    # Still capturable once the outer capture ends.
    assert graph.enabled


@skip_no_cuda
def test_replay_reproduces_the_eager_result_bitwise():
    device = torch.device("cuda")
    counter = torch.zeros(4, dtype=torch.int32, device=device)
    source = torch.arange(4, dtype=torch.int32, device=device)
    gathered = torch.zeros(4, dtype=torch.int32, device=device)
    indices = torch.tensor([3, 2, 1, 0], device=device)
    staged_host = torch.zeros(4, dtype=torch.int32).pin_memory()
    staged = torch.zeros(4, dtype=torch.int32, device=device)

    graph = SteadyGenPrepGraph()

    def step():
        work = graph.begin(("k",))
        try:
            run_device_work(counter.add_, 1)
            run_device_work(torch.index_select, source, 0, indices, out=gathered)
            run_device_work(staged.copy_, staged_host, non_blocking=True)
        finally:
            graph.end()
        return graph.issue(work)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        eager = []
        for value in range(WARMUP_STEPS):
            staged_host.fill_(value)
            assert step() is False
            torch.cuda.current_stream().synchronize()
            eager.append((counter.clone(), gathered.clone(), staged.clone()))

        replayed = []
        for value in range(WARMUP_STEPS, WARMUP_STEPS + 3):
            staged_host.fill_(value)
            assert step() is True
            torch.cuda.current_stream().synchronize()
            replayed.append((counter.clone(), gathered.clone(), staged.clone()))

    # The counter advances by exactly one per step across the capture, so the
    # capture neither skipped nor doubled the step it was taken on.
    assert [int(c[0]) for c, _, _ in eager + replayed] == list(range(1, WARMUP_STEPS + 4))
    for step_index, (_, gather, stage) in enumerate(replayed):
        assert torch.equal(gather, eager[0][1])
        # A replay re-reads the host buffer, so a value written after the
        # capture still lands on the device.
        assert torch.equal(stage, torch.full_like(stage, WARMUP_STEPS + step_index))
