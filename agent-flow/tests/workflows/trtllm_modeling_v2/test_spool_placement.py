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

from agent_flow.workflows.trtllm_modeling_v2.infra.execution import (
    CommandSpec,
    SpoolCommandExecutor,
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
        environ={"SLURM_JOB_ID": "100"},
        watcher_id=f"{name}-w",
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


_GATE = "#spool: gpus=all wall=1800\ntrue\n"


def test_a_gate_no_gate_session_can_start_goes_to_an_idle_work_session(tmp_path):
    """Gates queued behind each other while a work session's devices stood idle."""
    g1 = session(tmp_path, "gates-1", free=2)
    g2 = session(tmp_path, "gates-2", free=0, waiting=1)
    work = session(tmp_path, "work-1")
    executor = SpoolCommandExecutor(g1, pool=[g1, g2], overflow=[work])
    assert executor._pick(placement.JobNeeds.from_script("g", _GATE)) == work


_ONE = "#spool: gpus=1 wall=1800\ntrue\n"


def _run(root: Path, name: str) -> None:
    (root / "running" / f"{name}.sh").write_text(_ONE)


def test_a_dev_session_with_anything_running_takes_no_gate(tmp_path):
    """Its devices are free for a 1-GPU gate, but the campaign is using the session."""
    g1 = session(tmp_path, "gates-1", free=0)
    dev = session(tmp_path, "dev")
    _run(dev, "tuneg-nsys")
    executor = SpoolCommandExecutor(g1, last_resort=[dev])
    assert executor._pick(placement.JobNeeds.from_script("g", _ONE)) == g1


def test_a_job_a_watcher_claimed_first_is_never_moved(tmp_path):
    full = session(tmp_path, "gates-1", free=0)
    idle = session(tmp_path, "gates-2", free=4)
    executor = SpoolCommandExecutor(full, pool=[full, idle])
    job = placement.JobNeeds.from_script("gate-1", _ONE)
    _run(full, "gate-1")  # renamed into running/ by its watcher
    assert executor._relocate(job, full) is None
    assert pending(idle) == []
