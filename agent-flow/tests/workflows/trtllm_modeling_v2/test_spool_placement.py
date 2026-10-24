# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a job goes: the submit-time warning, ``--pick``, and the gate pool's choice.

Each session is a real :class:`Scheduler` that ticks once, so its status
record, heartbeat and hand-over request are the files a live watcher writes;
only its free devices and queue are then edited, which needs no GPU and no
job. Times are relative to the wall clock because ``spoolctl`` reads it.
"""

from __future__ import annotations

import json
import shlex
import time
from pathlib import Path

import pytest

import agent_flow.workflows.trtllm_modeling_v2 as pkg
from agent_flow.workflows.trtllm_modeling_v2.infra.driver import core
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import (
    RunEnvironment,
    resolve_environment,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import (
    CommandSpec,
    SpoolCommandExecutor,
    select_command_executor,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import placement, sessions, spoolctl
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
    parse_duration,
)

DEVICES = ("0", "1", "2", "3")
HOUR = 3600.0
#: Added to every time left, so the spans read whole minutes however long a test takes.
SLACK = 30.0


def session(
    parent: Path,
    name: str,
    *,
    left_s: float | None = 20 * HOUR,
    lead_s: float = HOUR,
    free: int | None = None,
    waiting: int = 0,
    age_s: float = 0.0,
    default_wall: str = "2h",
    watcher: str | None = None,
    job: str = "100",
) -> Path:
    """A spool watched by one scheduler that has just ticked ``age_s`` ago."""
    root = parent / name
    now = time.time()
    beat = now - age_s
    config = SchedulerConfig(
        devices=DEVICES,
        session=name,
        ends_at=None if left_s is None else now + left_s + SLACK,
        handover_lead_s=lead_s,
        default_wall_s=parse_duration(default_wall),
    )
    sched = Scheduler(
        root,
        config,
        clock=lambda: beat,
        environ={"SLURM_JOB_ID": job},
        watcher_id=watcher or f"{name}-w",
    )
    sched.tick()
    paths = SpoolPaths(root)
    own = paths.watchers / f"{sched.watcher_id}.json"
    record = json.loads(own.read_text())
    if free is not None:
        record["free"] = list(DEVICES[:free])
    record["pending"] = [
        {"name": f"other-{i}", "gpus": "1", "reason": "waiting for 1 GPU(s); 0 free"}
        for i in range(waiting)
    ]
    for path in (own, paths.snapshot):
        path.write_text(json.dumps(record))
    flags = sessions.scheduler_arguments(
        sessions.SessionSpec(name=name, spool=str(root), scheduler={"default_wall": default_wall})
    )
    (root / placement.ENTRY_SCRIPT).write_text(
        f'exec python3 scheduler.py watch "$SPOOL" {" ".join(shlex.quote(f) for f in flags)}\n'
    )
    return root


def pending(root: Path) -> list[str]:
    return sorted(p.name[: -len(".sh")] for p in (root / "pending").glob("*.sh"))


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = spoolctl.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture(autouse=True)
def _no_ambient_spool(monkeypatch):
    for key in ("SPOOL", "MV2FLOW_SPOOL", "MV2FLOW_JOB_PREFIX"):
        monkeypatch.delenv(key, raising=False)


# ------------------------------------------------------------------ kinds


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("gates-1", "gates"),
        ("gates-3", "gates"),
        ("work-2", "work"),
        ("work_2", "work"),
        ("perf", "perf"),
        ("perf-3", "perf"),
        ("dev-2", "dev"),
        ("spool-a", "spool-a"),
        ("b200", "b200"),
        ("gates-1-n1", "gates-1-n1"),
    ],
)
def test_a_spools_kind_is_its_name_without_the_replica_number(name, kind):
    assert placement.spool_kind(Path("/runs/x") / name) == kind


def test_siblings_are_the_spools_of_one_kind_in_one_directory(tmp_path):
    for name in ("gates-1", "gates-2", "work-1", "perf", "perf-2", "gates-1-n1"):
        SpoolPaths(tmp_path / name).ensure()
    (tmp_path / "gates-9").mkdir()  # not a spool: nothing was ever made in it

    def names(spool: str) -> list[str]:
        return [p.name for p in placement.siblings(tmp_path / spool)]

    assert names("gates-1") == ["gates-1", "gates-2"]
    assert names("perf-2") == ["perf", "perf-2"]
    assert names("work-1") == ["work-1"]
    assert names("gates-1-n1") == ["gates-1-n1"]


# ------------------------------------------------------- submit-time warning


def test_a_spool_in_hand_over_warns_and_names_the_siblings_that_can_start_it(tmp_path, capsys):
    """Minutes before its allocation ends, a session holds a 2h job for a successor."""
    work1 = session(tmp_path, "work-1", left_s=420, lead_s=1.5 * HOUR, job="1000033")
    sessions.acknowledge(work1, job_id="1000248", by="driver")
    session(tmp_path, "work-2", free=3)
    session(tmp_path, "work-3", left_s=600, lead_s=1.5 * HOUR)  # in hand-over too
    session(tmp_path, "gates-1")  # idle, but of another kind
    code, out, err = run(
        capsys, "--spool", str(work1), "launch", "--gpus", "1", "--wall", "2h", "r1-x", "true"
    )
    assert code == 0 and "queued r1-x" in out
    assert pending(work1) == ["r1-x"]  # the explicit choice still stands
    lines = err.strip().splitlines()
    assert len(lines) == 1, err
    line = lines[0]
    assert line.startswith("spoolctl: warning: work-1 will not start r1-x soon: ")
    assert "in hand-over and its allocation ends in 7m, too soon for a job allowed 2h" in line
    assert "successor job 1000248 submitted, not up yet" in line
    assert "work-2 (3 of 4 GPUs free, 20h" in line
    assert "work-3" not in line and "gates-1" not in line
    assert f"spoolctl.py cancel r1-x, then resubmit with --spool {tmp_path / 'work-2'}" in line


def test_a_session_ending_before_the_jobs_wall_limit_warns(tmp_path, capsys):
    """Outside hand-over the scheduler admits the job, and the allocation's end kills it."""
    work1 = session(tmp_path, "work-1", left_s=1.5 * HOUR)
    session(tmp_path, "work-2")
    code, _, err = run(capsys, "--spool", str(work1), "launch", "--wall", "2h", "r1-x", "true")
    assert code == 0 and pending(work1) == ["r1-x"]
    assert "the allocation ends in 1h30m, before the job's 2h wall limit" in err
    assert "work-2 (4 of 4 GPUs free" in err
    code, _, err = run(capsys, "--spool", str(work1), "launch", "--wall", "1h", "r1-y", "true")
    assert code == 0 and not err


def test_without_a_wall_the_sessions_own_default_is_the_limit(tmp_path, capsys):
    gates = session(tmp_path, "gates-1", left_s=2.5 * HOUR, default_wall="3h")
    code, _, err = run(capsys, "--spool", str(gates), "launch", "--gpus", "all", "g-x", "true")
    assert code == 0 and "before the job's 3h wall limit" in err
    assert "No other spool of kind 'gates'" in err
    work = session(tmp_path, "work-1", left_s=2.5 * HOUR, default_wall="2h")
    code, _, err = run(capsys, "--spool", str(work), "launch", "--gpus", "all", "w-x", "true")
    assert code == 0 and not err
    # A device-free job has no default limit, so it fits any session.
    code, _, err = run(capsys, "--spool", str(gates), "launch", "free-x", "true")
    assert code == 0 and not err


def test_a_dead_watcher_refuses_the_job_and_names_where_it_can_go(tmp_path, capsys):
    work1 = session(tmp_path, "work-1", age_s=600)
    session(tmp_path, "work-2")
    code, _, err = run(capsys, "--spool", str(work1), "launch", "--gpus", "1", "r1-x", "true")
    assert code == spoolctl.EXIT_NO_SESSION
    assert pending(work1) == []
    warning, refusal = err.strip().splitlines()
    assert "its watcher is not alive (last heartbeat 10m ago)" in warning
    assert f"To run it there: resubmit with --spool {tmp_path / 'work-2'}" in warning
    assert "cancel" not in warning
    assert "stale: the session is gone" in refusal


def test_a_healthy_spool_takes_the_job_without_a_word(tmp_path, capsys):
    work1 = session(tmp_path, "work-1", free=0, waiting=3)  # busy, but it will run it
    session(tmp_path, "work-2", left_s=300, lead_s=HOUR)
    code, _, err = run(capsys, "--spool", str(work1), "launch", "--gpus", "1", "r1-x", "true")
    assert code == 0 and err == "" and pending(work1) == ["r1-x"]


def test_a_job_that_fits_before_the_end_is_admitted_in_hand_over_without_a_warning(
    tmp_path, capsys
):
    work1 = session(tmp_path, "work-1", left_s=40 * 60, lead_s=HOUR)
    code, _, err = run(capsys, "--spool", str(work1), "launch", "--wall", "10m", "r1-x", "true")
    assert code == 0 and err == ""


def test_a_spool_whose_successor_is_up_is_judged_by_the_successor(tmp_path, capsys):
    root = tmp_path / "work-1"
    session(tmp_path, "work-1", watcher="new", job="200")
    session(tmp_path, "work-1", left_s=600, lead_s=HOUR, watcher="old", job="100", free=4)
    old = json.loads((root / "watchers" / "old.json").read_text())
    assert old["handover"]["successor"] == "new"
    job = placement.JobNeeds.from_script("r1-x", "#spool: gpus=1\ntrue\n")
    view = placement.assess(root, job, stale_s=spoolctl.WATCHER_STALE_S, now=time.time())
    assert view.can_take and not view.in_handover and view.remaining_s > 19 * HOUR
    code, _, err = run(capsys, "--spool", str(root), "launch", "--gpus", "1", "r1-x", "true")
    assert code == 0 and err == ""


def test_a_stopping_session_warns(tmp_path, capsys):
    work1 = session(tmp_path, "work-1")
    SpoolPaths(work1).stop.write_text("")
    code, _, err = run(capsys, "--spool", str(work1), "launch", "free-x", "true")
    assert code == 0 and "the session is stopping (STOP)" in err


# ------------------------------------------------------------------- --pick


def test_pick_chooses_the_sibling_that_can_start_the_job_now(tmp_path, capsys):
    work1 = session(tmp_path, "work-1", free=1)
    session(tmp_path, "work-2", free=4, left_s=0.5 * HOUR, lead_s=HOUR)  # in hand-over
    session(tmp_path, "work-3", free=3)
    session(tmp_path, "gates-1", free=4)  # more room, another kind
    code, out, err = run(
        capsys, "--spool", str(work1), "launch", "--pick", "--gpus", "2", "r1-x", "true"
    )
    assert code == 0
    assert err.strip() == (
        f"spoolctl: --pick chose {tmp_path / 'work-3'} (3 of 4 GPUs free, 20h left)"
    )
    assert pending(tmp_path / "work-3") == ["r1-x"]
    assert f"log at {tmp_path / 'work-3' / 'logs' / 'r1-x.log'}" in out
    for other in ("work-1", "work-2", "gates-1"):
        assert pending(tmp_path / other) == []


def test_pick_respects_the_jobs_devices_and_wall_limit(tmp_path, monkeypatch, capsys):
    session(tmp_path, "perf", free=4, left_s=2 * HOUR)
    session(tmp_path, "perf-2", free=2)
    session(tmp_path, "perf-3", free=0, waiting=1)
    monkeypatch.setenv("SPOOL", str(tmp_path / "perf"))
    # 4h does not fit perf's 2h; perf-2 has the 2 devices free now, perf-3 none.
    code, _, err = run(capsys, "launch", "--pick", "--gpus", "2", "--wall", "4h", "p-a", "true")
    assert code == 0 and f"chose {tmp_path / 'perf-2'} " in err
    # The whole node is free nowhere it fits: queue where the most devices are free.
    code, _, err = run(capsys, "launch", "--pick", "--gpus", "all", "--wall", "4h", "p-b", "true")
    assert code == 0 and f"chose {tmp_path / 'perf-2'} " in err
    # A short job fits everywhere, and the whole node is free on perf.
    code, _, err = run(capsys, "launch", "--pick", "--gpus", "all", "--wall", "1h", "p-c", "true")
    assert code == 0 and f"chose {tmp_path / 'perf'} " in err


def test_pick_prefers_the_given_spool_when_siblings_are_equal(tmp_path, monkeypatch, capsys):
    session(tmp_path, "dev")
    session(tmp_path, "dev-2")
    monkeypatch.setenv("MV2FLOW_SPOOL", str(tmp_path / "dev-2"))
    code, _, err = run(capsys, "launch", "--pick", "--gpus", "1", "d-a", "true")
    assert code == 0 and pending(tmp_path / "dev-2") == ["d-a"]


def test_pick_fails_clearly_when_no_sibling_can_take_the_job(tmp_path, capsys):
    work1 = session(tmp_path, "work-1", left_s=600, lead_s=HOUR)
    session(tmp_path, "work-2", age_s=900)
    session(tmp_path, "work-3", left_s=40 * 60, lead_s=HOUR)  # would fit, but in hand-over
    session(tmp_path, "gates-1")
    code, out, err = run(
        capsys, "--spool", str(work1), "launch", "--pick", "--wall", "30m", "r1-x", "true"
    )
    assert code == spoolctl.EXIT_NO_SESSION and out == ""
    assert f"--pick: no spool of kind 'work' in {tmp_path} can take r1-x now (" in err
    assert "work-1: it is in hand-over and its allocation ends in 10m" in err
    assert "work-2: its watcher is not alive (last heartbeat 15m ago)" in err
    assert "work-3: it is in hand-over, its allocation ending in 40m" in err
    assert "gates-1" not in err
    assert all(pending(p) == [] for p in tmp_path.iterdir())


def test_spool_may_follow_the_action(tmp_path, capsys):
    one, two = session(tmp_path, "a-1"), session(tmp_path, "b-1")
    assert run(capsys, "launch", "--spool", str(one), "free-x", "true")[0] == 0
    assert run(capsys, "--spool", str(two), "launch", "free-y", "true")[0] == 0
    assert pending(one) == ["free-x"] and pending(two) == ["free-y"]
    assert run(capsys, "status", "--spool", str(one), "--json")[0] == 0


# --------------------------------------------------------------- gate pool


def test_the_gate_pool_skips_a_session_in_hand_over_for_one_that_can_take_the_job(tmp_path):
    """The least loaded gate spool was the one about to end: the gate parked there."""
    ending = session(tmp_path, "gates-2", left_s=0.5 * HOUR, lead_s=1.5 * HOUR, default_wall="3h")
    busy = session(tmp_path, "gates-1", default_wall="3h")
    (busy / "pending" / "other.sh").write_text("true\n")
    executor = SpoolCommandExecutor(ending, poll_s=0.01, queue_timeout_s=0.05, pool=[ending, busy])
    executor.execute(CommandSpec(argv="echo hi", timeout_s=0))
    assert pending(ending) == []
    assert len(pending(busy)) == 2


def test_the_gate_pool_keeps_the_least_loaded_live_spool_when_none_can_take_the_job(tmp_path):
    first = session(tmp_path, "gates-1", left_s=1.5 * HOUR)
    second = session(tmp_path, "gates-2", left_s=1.5 * HOUR)
    (first / "pending" / "other.sh").write_text("true\n")
    executor = SpoolCommandExecutor(first, pool=[first, second])
    long = placement.JobNeeds.from_script("g", "#spool: gpus=all wall=14400\ntrue\n")
    assert executor._pick(long) == second
    short = placement.JobNeeds.from_script("g", "#spool: gpus=all wall=1800\ntrue\n")
    (second / "pending" / "x.sh").write_text("true\n")
    (second / "pending" / "y.sh").write_text("true\n")
    assert executor._pick(short) == first
    assert executor._pick() == first


_GATE = "#spool: gpus=all wall=1800\ntrue\n"


def test_a_gate_no_gate_session_can_start_goes_to_an_idle_work_session(tmp_path):
    """Gates queued behind each other while a work session's devices stood idle."""
    g1 = session(tmp_path, "gates-1", free=2)
    g2 = session(tmp_path, "gates-2", free=0, waiting=1)
    work = session(tmp_path, "work-1")
    executor = SpoolCommandExecutor(g1, pool=[g1, g2], overflow=[work])
    assert executor._pick(placement.JobNeeds.from_script("g", _GATE)) == work


def test_a_gate_session_that_starts_the_gate_now_keeps_it(tmp_path):
    g1 = session(tmp_path, "gates-1", free=0)
    g2 = session(tmp_path, "gates-2")
    work = session(tmp_path, "work-1")
    executor = SpoolCommandExecutor(g1, pool=[g1, g2], overflow=[work])
    assert executor._pick(placement.JobNeeds.from_script("g", _GATE)) == g2


def test_a_busy_or_ending_work_session_takes_no_gate(tmp_path):
    g1 = session(tmp_path, "gates-1", free=0)
    busy = session(tmp_path, "work-1", free=1)
    ending = session(tmp_path, "work-2", left_s=0.5 * HOUR, lead_s=1.5 * HOUR)
    executor = SpoolCommandExecutor(g1, pool=[g1], overflow=[busy, ending])
    assert executor._pick(placement.JobNeeds.from_script("g", _GATE)) == g1
    # Without a job to judge, nothing overflows.
    assert executor._pick() == g1


# ------------------------------------------- the tune campaign's dev sessions


_ONE = "#spool: gpus=1 wall=1800\ntrue\n"


def _queue(root: Path, name: str, script: str = _ONE) -> None:
    (root / "pending" / f"{name}.sh").write_text(script)


def _run(root: Path, name: str) -> None:
    (root / "running" / f"{name}.sh").write_text(_ONE)


def _environment(
    gates: list[Path], work: list[Path] = (), dev: list[Path] = (), world_size: int = 1
) -> RunEnvironment:
    sessions = [(p.name, str(p), "gates") for p in gates] + [(p.name, str(p), "work") for p in work]
    return RunEnvironment(
        kind="session_spool",
        python="python3",
        spool=str(gates[0]),
        world_size=world_size,
        sessions=tuple(sessions),
        dev_spools=tuple(str(p) for p in dev),
    )


def test_the_environment_carries_the_declared_dev_spools(tmp_path):
    task = {
        "environment": {"kind": "session_spool", "spool": str(tmp_path / "gates-1")},
        "perf": {"dev_spools": [str(tmp_path / "dev"), str(tmp_path / "dev-2")]},
    }
    env = resolve_environment(task)
    assert env.dev_spools == (str(tmp_path / "dev"), str(tmp_path / "dev-2"))
    assert "dev-2" not in env.describe(), "no prompt names the campaign's sessions"


@pytest.mark.parametrize(
    "perf",
    [
        None,
        {"spools": ["/s/perf"]},  # the campaign falls back to these: they measure
        {"spools": ["/s/perf"], "dev_spools": ["/s/perf"]},
    ],
)
def test_only_a_declared_dev_spool_that_measures_nothing_is_lent(perf):
    task = {"environment": {"kind": "session_spool", "spool": "/s/gates-1"}}
    if perf is not None:
        task["perf"] = perf
    assert resolve_environment(task).dev_spools == ()


def test_the_executor_takes_the_dev_spools_as_its_last_resort(tmp_path):
    g1, w1, dev = tmp_path / "gates-1", tmp_path / "work-1", tmp_path / "dev"
    executor = select_command_executor(_environment([g1], [w1], [dev, w1]))
    assert isinstance(executor, SpoolCommandExecutor)
    assert executor.overflow == [w1]
    assert executor.last_resort == [dev]


def test_a_gate_nothing_else_starts_goes_to_an_idle_dev_session(tmp_path):
    g1 = session(tmp_path, "gates-1", free=0)
    w1 = session(tmp_path, "work-1", free=1)
    dev = session(tmp_path, "dev")
    executor = SpoolCommandExecutor(g1, pool=[g1], overflow=[w1], last_resort=[dev])
    assert executor._pick(placement.JobNeeds.from_script("g", _GATE)) == dev


def test_the_last_resort_alone_is_still_offered_the_gate(tmp_path):
    g1, dev = session(tmp_path, "gates-1", free=0), session(tmp_path, "dev")
    executor = SpoolCommandExecutor(g1, last_resort=[dev])
    assert executor._pick(placement.JobNeeds.from_script("g", _GATE)) == dev
    assert executor._pick() == g1


def test_a_work_session_that_starts_the_gate_comes_before_a_dev_session(tmp_path):
    g1 = session(tmp_path, "gates-1", free=0)
    w1, dev = session(tmp_path, "work-1"), session(tmp_path, "dev")
    executor = SpoolCommandExecutor(g1, overflow=[w1], last_resort=[dev])
    assert executor._pick(placement.JobNeeds.from_script("g", _GATE)) == w1


@pytest.mark.parametrize("busy", [_run, _queue])
def test_a_dev_session_with_anything_running_or_waiting_takes_no_gate(tmp_path, busy):
    """Its devices are free for a 1-GPU gate, but the campaign is using the session."""
    g1 = session(tmp_path, "gates-1", free=0)
    dev = session(tmp_path, "dev")
    busy(dev, "tuneg-nsys")
    executor = SpoolCommandExecutor(g1, last_resort=[dev])
    assert executor._pick(placement.JobNeeds.from_script("g", _ONE)) == g1


# ------------------------------------------------------------------- docs


def test_the_worker_facing_instructions_teach_the_warning_and_pick():
    package = Path(pkg.__file__).parent
    skill = "agents/skills/trtllm-modeling-v2-perf-tuning/references/modeling-iterations.md"
    for path in (core.TURN_INSTRUCTIONS_TEMPLATE, package / skill, package / "README.md"):
        text = path.read_text(encoding="utf-8")
        assert "--pick" in text and "warning:" in text, path


def test_the_default_wall_is_read_from_the_watchers_own_record(tmp_path):
    """A watcher publishes its default wall; the entry script is only the fallback."""
    gates = session(tmp_path, "gates-1", default_wall="3h")
    (gates / placement.ENTRY_SCRIPT).unlink()
    record = json.loads(SpoolPaths(gates).snapshot.read_text())
    assert record["default_wall_s"] == 3 * HOUR
    assert placement.default_wall_s(gates, record) == 3 * HOUR


def test_a_burst_of_gates_spreads_over_the_sessions_before_their_watchers_tick(tmp_path):
    """Each watcher still reports every device free; the jobs already queued tell the truth."""
    g1, g2 = session(tmp_path, "gates-1"), session(tmp_path, "gates-2")
    w1, w2 = session(tmp_path, "work-1"), session(tmp_path, "work-2")
    executor = SpoolCommandExecutor(g1, pool=[g1, g2], overflow=[w1, w2])
    job = placement.JobNeeds.from_script("g", _GATE)
    chosen = []
    for n in range(5):
        root = executor._pick(job)
        (root / "pending" / f"g{n}.sh").write_text(_GATE)
        chosen.append(root.name)
    assert sorted(chosen[:4]) == ["gates-1", "gates-2", "work-1", "work-2"]
    assert chosen[4].startswith("gates-")


def _finished(root: Path, n: int, *, owner="gate", wall=14400.0, gpus="all", seconds=290.0):
    for i in range(n):
        (root / "done" / f"{owner}-{i}.json").write_text(
            json.dumps(
                {"owner": owner, "wall_s": wall, "gpus": gpus, "duration_s": seconds,
                 "reason": "exit", "timed_out": False}
            )
        )  # fmt: skip


def test_a_job_is_judged_by_how_long_such_jobs_ran_not_by_its_worst_case(tmp_path):
    """A 4 h wall limit on gates that finish in 5 minutes kept every short session from them."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import scheduler

    ending = session(tmp_path, "gates-1", left_s=1.5 * HOUR)
    job = placement.JobNeeds.from_script("g", "#spool: gpus=all wall=14400 owner=gate\ntrue\n")
    assert job.owner == "gate"
    assert job.expected_on(ending, 0) == 14400  # no history: the wall limit
    _finished(ending, 4)
    assert job.expected_on(ending, 0) == 14400  # too little of it
    _finished(ending, 8)
    assert job.expected_on(ending, 0) == 2 * 290.0
    assert scheduler.expected_s(ending / "done", "gate", 14400, "4") is None  # another ask
    assert not placement.assess(ending, job, stale_s=180, now=time.time()).reasons


def test_a_session_near_its_end_admits_a_gate_that_such_gates_finish_in_time_for(tmp_path):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import scheduler

    root = tmp_path / "gates-1"
    (root / "done").mkdir(parents=True)
    watcher = SimpleNamespace(
        config=SimpleNamespace(ends_at=time.time() + 1800, handover_lead_s=5400),
        handover={},
        paths=scheduler.SpoolPaths(root),
    )
    spec = scheduler.job_spec_from_script("g", "#spool: gpus=all wall=14400 owner=gate\ntrue\n")
    hold = scheduler.Scheduler._handover_hold(watcher, spec, time.time())
    assert hold is not None and "too soon" in hold.reason
    _finished(root, 6)
    assert scheduler.Scheduler._handover_hold(watcher, spec, time.time()) is None


# ------------------------------------------------------------- moving to where it starts


def test_a_queued_gate_moves_to_a_spool_that_would_start_it_now(tmp_path):
    """Placed on a full session, it waited while another's devices stood idle."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import spoolctl

    full = session(tmp_path, "gates-1", free=0)
    idle = session(tmp_path, "gates-2", free=0)
    executor = SpoolCommandExecutor(full, poll_s=0.01, pool=[full, idle])
    job = placement.JobNeeds.from_script("gate-1", _ONE)
    _queue(full, "gate-1")
    assert executor._relocate(job, full) is None, "nowhere starts it now: it stays"
    idle = session(tmp_path, "gates-2", free=4)
    assert executor._relocate(job, full) == idle
    assert pending(full) == [] and pending(idle) == ["gate-1"]

    # The waiter follows it and reads its receipt where it ran.
    (idle / "pending" / "gate-1.sh").unlink()  # its watcher took it
    _queue(full, "gate-2")

    def ran_there(current):
        moved = executor._relocate(placement.JobNeeds.from_script("gate-2", _ONE), current)
        if moved is not None:
            (moved / "done" / "gate-2.json").write_text(json.dumps({"rc": 0}))
            (moved / "done" / "gate-2.rc").write_text("0\n")
        return moved

    record = spoolctl.wait_for_job(
        full, "gate-2", poll_s=0.01, relocate=ran_there, relocate_every_s=0, stale_s=5
    )
    assert record["rc"] == 0


def test_a_job_a_watcher_claimed_first_is_never_moved(tmp_path):
    full = session(tmp_path, "gates-1", free=0)
    idle = session(tmp_path, "gates-2", free=4)
    executor = SpoolCommandExecutor(full, pool=[full, idle])
    job = placement.JobNeeds.from_script("gate-1", _ONE)
    _run(full, "gate-1")  # renamed into running/ by its watcher
    assert executor._relocate(job, full) is None
    assert pending(idle) == []


def test_a_dev_session_lends_its_free_devices_beside_the_gates_it_lent_already(tmp_path):
    g1 = session(tmp_path, "gates-1", free=0)
    dev = session(tmp_path, "dev", free=3)
    (dev / "running" / "gate-9.sh").write_text("#spool: gpus=1 wall=1800 owner=gate\ntrue\n")
    executor = SpoolCommandExecutor(g1, last_resort=[dev])
    assert executor._pick(placement.JobNeeds.from_script("g", _ONE)) == dev
    _run(dev, "tuneg-nsys")  # the campaign's own work: the node is not lent beside it
    assert executor._pick(placement.JobNeeds.from_script("g", _ONE)) == g1
