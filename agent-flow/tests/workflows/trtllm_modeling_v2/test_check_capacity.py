# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many more checks would start at once, how many run, and what a running task holds.

The sessions are the ones ``test_spool_placement`` builds: a real scheduler
that ticked once, its free devices and queue then edited. A queued job is a
script in ``pending/``, as a submitter leaves it. :class:`CheckAdmission`
runs on a fake reading of the sessions and a clock the test moves.
"""

from __future__ import annotations

import threading
import time

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import (
    DEFAULT_CHECK_FALLBACK,
    CheckAdmission,
    SpoolCommandExecutor,
    check_capacity,
    more_checks,
    startable_checks,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.execution.capacity import (
    TUNE_KIND,
    held_gpus,
    offsite_kinds,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import placement
from agent_flow.workflows.trtllm_modeling_v2.run.graph import Task

from .test_spool_placement import HOUR, _environment, _queue, session

_WHOLE = "#spool: gpus=all wall=1800\ntrue\n"


def _job(gpus: int | None = 1) -> placement.JobNeeds:
    return placement.JobNeeds(name="check", gpus=gpus, wall_s=1800, holds_gpus=True)


# ---------------------------------------------------- what would start


def test_every_spool_offers_the_fits_of_its_free_devices(tmp_path):
    g1, g2 = session(tmp_path, "gates-1", free=2), session(tmp_path, "gates-2", free=0)
    w1, dev = session(tmp_path, "work-1", free=3), session(tmp_path, "dev")
    executor = SpoolCommandExecutor(g1, pool=[g1, g2], overflow=[w1], last_resort=[dev])
    offers = {p.name: executor._offer(p, _job(), time.time(), tier) for p, tier in [
        (g1, "pool"), (g2, "pool"), (w1, "overflow"), (dev, "last_resort")]}  # fmt: skip
    # The dev session lends every free device while nothing waits there.
    assert offers == {"gates-1": 2, "gates-2": 0, "work-1": 3, "dev": 4}
    assert executor.startable(_job()) == 9
    assert executor.startable(_job(None)) == 1, "only the idle dev node is whole"
    assert executor.startable(_job(2)) == 1 + 0 + 1 + 2


def test_the_jobs_queued_on_a_gate_spool_take_their_devices_first(tmp_path):
    g1 = session(tmp_path, "gates-1")
    executor = SpoolCommandExecutor(g1)
    _queue(g1, "a")
    assert executor.startable(_job()) == 3
    _queue(g1, "b", _WHOLE)
    assert executor.startable(_job()) == 0
    assert executor.startable(_job(None)) == 0


def test_an_overflow_spool_with_anything_queued_offers_nothing(tmp_path):
    g1, w1 = session(tmp_path, "gates-1", free=0), session(tmp_path, "work-1")
    executor = SpoolCommandExecutor(g1, overflow=[w1])
    assert executor.startable(_job()) == 4
    _queue(w1, "worker-job")
    assert executor.startable(_job()) == 0


def test_free_devices_on_two_nodes_of_one_spool_are_not_pooled(tmp_path):
    """A multi-node session runs one watcher per node; a 2-GPU check needs two on one."""
    session(tmp_path, "gates-1", watcher="node-a", free=1)
    g1 = session(tmp_path, "gates-1", watcher="node-b", free=1)
    executor = SpoolCommandExecutor(g1)
    assert executor.startable(_job(1)) == 2
    assert executor.startable(_job(2)) == 0


def test_a_session_that_cannot_take_the_check_offers_nothing(tmp_path):
    ending = session(tmp_path, "gates-1", left_s=0.5 * HOUR, lead_s=1.5 * HOUR)
    stale = session(tmp_path, "gates-2", age_s=600)
    executor = SpoolCommandExecutor(ending, pool=[ending, stale])
    assert executor.startable(_job()) == 0
    executor = SpoolCommandExecutor(stale)
    assert executor.startable(_job()) is None, "no live watcher: nothing to count"


def test_the_check_capacity_reads_the_runs_sessions(tmp_path):
    g1, w1 = session(tmp_path, "gates-1", free=1), session(tmp_path, "work-1")
    dev = session(tmp_path, "dev")
    env = _environment([g1], [w1], [dev])
    assert startable_checks(env) == 1 + 4 + 4
    assert startable_checks(env, "all") == 2
    assert startable_checks(_environment([g1], [w1], [dev], world_size=4)) == 2
    assert check_capacity(env) == 9
    assert check_capacity(env, in_flight=10) == 9
    assert check_capacity(env, in_flight=10, starting=4) == 5
    assert check_capacity(env, in_flight=30) == 2, "never more than the cap"
    # A spool that ends before an accuracy check's wall limit is no place for it.
    short = session(tmp_path, "gates-2", left_s=2.5 * HOUR, lead_s=HOUR)
    assert startable_checks(_environment([short]), wall_s=1800) == 4
    assert startable_checks(_environment([short]), wall_s=4 * HOUR) == 0


def test_without_a_session_to_read_the_fixed_pool_stands(tmp_path):
    local = RunEnvironment(kind="local", python="python3")
    assert startable_checks(local) is None
    assert check_capacity(local) == DEFAULT_CHECK_FALLBACK
    assert check_capacity(local, in_flight=3) == DEFAULT_CHECK_FALLBACK - 3
    gone = session(tmp_path, "gates-1", age_s=600)
    assert check_capacity(_environment([gone])) == DEFAULT_CHECK_FALLBACK


@pytest.mark.parametrize(
    ("startable", "in_flight", "starting", "more"),
    [
        (0, 0, 0, 2),  # the floor
        (0, 2, 0, 0),
        (5, 3, 0, 5),
        (5, 3, 2, 3),  # two started too recently to show in a spool
        (5, 3, 9, 0),
        (100, 10, 0, 22),  # the cap
        (None, 3, 0, 5),  # nothing to read: the fallback
    ],
)
def test_more_checks(startable, in_flight, starting, more):
    assert more_checks(startable, in_flight=in_flight, starting=starting, cap=32) == more


# ------------------------------------------------------------ admission


class _Sessions:
    """A reading of the sessions the test sets, counting how often it is read."""

    def __init__(self, free: int | None) -> None:
        self.free = free
        self.reads = 0

    def __call__(self, environment, gpus, wall_s):
        self.reads += 1
        if self.free is None or gpus != "all":
            return self.free
        return self.free // 4


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


_ENV = RunEnvironment(kind="session_spool", python="python3", spool="/s/gates-1")


def _admission(sessions, clock=None, **kwargs) -> CheckAdmission:
    kwargs.setdefault("poll_s", 0.01)
    return CheckAdmission(sessions, clock=clock or _Clock(), **kwargs)


class _Check:
    """A check thread that starts once admitted and runs until :meth:`finish`."""

    def __init__(self, admission: CheckAdmission, order: list | None = None, **slot) -> None:
        self.started, self._done = threading.Event(), threading.Event()
        self._order = order if order is not None else []
        self._thread = threading.Thread(target=self._run, args=(admission, slot), daemon=True)
        self._thread.start()

    def _run(self, admission: CheckAdmission, slot: dict) -> None:
        with admission.slot(_ENV, **slot):
            self._order.append(self)
            self.started.set()
            self._done.wait(10)

    def finish(self) -> None:
        self._done.set()
        self._thread.join(5)


def _settles(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_checks_start_while_devices_are_free_and_wait_once_they_are_not():
    sessions, clock = _Sessions(free=3), _Clock()
    admission = _admission(sessions, clock, floor=1, settle_s=0)
    held = [admission.slot(_ENV) for _ in range(3)]
    for slot in held:
        slot.__enter__()
    assert admission.in_flight == 3
    sessions.free = 0
    clock.now += 5
    check = _Check(admission)
    assert _settles(lambda: admission.waiting == 1)
    assert not check.started.wait(0.1)
    # A check ends; its device is free, and the sessions are read again at once.
    sessions.free = 1
    held.pop().__exit__(None, None, None)
    assert check.started.wait(5)
    check.finish()
    for slot in held:
        slot.__exit__(None, None, None)
    assert admission.in_flight == 0


def test_the_floor_starts_checks_however_busy_the_sessions_read():
    admission = _admission(_Sessions(free=0), floor=2)
    with admission.slot(_ENV), admission.slot(_ENV):
        assert admission.in_flight == 2
        check = _Check(admission)
        assert not check.started.wait(0.1)
    assert check.started.wait(5)
    check.finish()


def test_a_check_just_started_still_counts_against_the_free_devices():
    """Its job is not in a spool yet, so the next reading still shows its device free."""
    sessions, clock = _Sessions(free=1), _Clock()
    admission = _admission(sessions, clock, floor=1, settle_s=60, refresh_s=0)
    with admission.slot(_ENV):
        assert not admission._may_start(99, _ENV, None, None)
        clock.now += 61
        assert admission._may_start(99, _ENV, None, None)


def test_a_check_that_waited_long_enough_is_not_passed_by_narrower_ones():
    sessions, clock, order = _Sessions(free=0), _Clock(), []
    admission = _admission(sessions, clock, floor=1, settle_s=0, patience_s=600, refresh_s=0)
    with admission.slot(_ENV):
        wide = _Check(admission, order, gpus="all")
        assert _settles(lambda: admission.waiting == 1)
        clock.now += 601
        narrow = _Check(admission, order)
        assert _settles(lambda: admission.waiting == 2)
        sessions.free = 3  # devices for the narrow check, no whole node for the wide one
        assert not narrow.started.wait(0.2)
        sessions.free = 4
        assert wide.started.wait(5) and narrow.started.wait(5)
        assert order == [wide, narrow]
    wide.finish()
    narrow.finish()


def test_never_more_than_the_cap_at_once():
    admission = _admission(_Sessions(free=100), floor=1, cap=2, settle_s=0)
    with admission.slot(_ENV), admission.slot(_ENV):
        check = _Check(admission)
        assert not check.started.wait(0.1)
    assert check.started.wait(5)
    check.finish()


def test_the_sessions_are_read_once_per_width_until_a_check_ends():
    sessions, clock = _Sessions(free=10), _Clock()
    admission = _admission(sessions, clock, floor=1, settle_s=0, refresh_s=2)
    with admission.slot(_ENV):
        for _ in range(3):
            assert admission._may_start(99, _ENV, None, None)
        assert admission._may_start(99, _ENV, "all", None)
        assert sessions.reads == 2
        clock.now += 2
        assert admission._may_start(99, _ENV, None, None)
        assert sessions.reads == 3
    assert admission._may_start(99, _ENV, None, None)
    assert sessions.reads == 3, "below the floor nothing is read"
    with admission.slot(_ENV):
        assert admission._may_start(99, _ENV, None, None)
    assert sessions.reads == 4, "a check that ended dropped the reading"


def test_with_nothing_to_read_the_fixed_pool_size_runs_at_once():
    admission = _admission(_Sessions(free=None), floor=1, fallback=3, settle_s=0)
    with admission.slot(_ENV), admission.slot(_ENV), admission.slot(_ENV):
        assert not admission._may_start(99, _ENV, None, None)
    assert admission._may_start(99, _ENV, None, None)


# ---------------------------------------------------- what a running task holds


def _env(*spools: str, spool: str | None = None) -> RunEnvironment:
    return RunEnvironment(kind="session_spool", python="python3", spool=spool,
                          sessions=tuple((f"s{i}", sp, "gates") for i, sp in enumerate(spools)))  # fmt: skip


def test_a_tune_task_holds_no_session_device_once_its_jobs_run_on_its_own_sessions():
    env = _env("/r/spool/gates-1", "/r/spool/work-1", spool="/r/spool/gates-1")
    tune = Task(id="tune", kind=TUNE_KIND, goal="tune", gpus=4)
    smith = Task(id="k1", kind="kernel", goal="k", gpus=1)
    perf = {"spools": ["/r/spool/perf", "/r/spool/perf-2"], "dev_spools": ["/r/spool/dev"]}
    offsite = offsite_kinds(perf, env)
    assert offsite == {TUNE_KIND}
    assert held_gpus(tune, offsite) == 0 and held_gpus(smith, offsite) == 1
    assert held_gpus(tune) == 4  # without the rule, what it declares
    # Its measurements or its keep gate on a run session, or no perf sessions at all: it counts.
    assert offsite_kinds({**perf, "spools": ["/r/spool/work-1"]}, env) == frozenset()
    assert offsite_kinds({**perf, "dev_spools": ["/r/spool/gates-1/"]}, env) == frozenset()
    assert offsite_kinds({"dev_spools": ["/r/spool/dev"]}, env) == frozenset()
    assert offsite_kinds(None, env) == frozenset()
    # Without dev spools, the keep gate runs on the measuring sessions: still its own.
    assert offsite_kinds({"spools": ["/r/spool/perf"]}, env) == {TUNE_KIND}
