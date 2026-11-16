# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 spool scheduler behavior and regression checks."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import executor as priority
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import (
    scheduler as _spool_scheduler_scheduler,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import spoolctl
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import (
    TIMEOUT_RETURNCODE,
    CommandSpec,
    LocalCommandExecutor,
    ended_early,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
    MULTI_TASK_SLURM_SCRUB,
    RunEnvironment,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import (
    CANCELLED,
    SpoolCommandExecutor,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
    RC_CANCELLED,
    RC_MIXED_CHECKOUT,
    RC_NO_ORIGIN,
    RC_REFUSED,
    RC_WALL_LIMIT,
    RERUN_TAG,
    SIZE_MISSING,
    WALL_MISSING,
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
    write_json_atomic as _spool_requeue_write_json_atomic,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.files import (
    write_json_atomic as _spool_sessions_write_json_atomic,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor import status
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor.heartbeat import (
    ORCHESTRATOR_EXIT,
    ORCHESTRATOR_HEARTBEAT,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor.status import load_config
from agent_flow.workflows.trtllm_modeling_v2.operators.inventory import Inventory
from agent_flow.workflows.trtllm_modeling_v2.run.records.state import STAGE_PREFLIGHT
from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap as preflight_stage
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from ._helpers import Script, _install, _state, healthy_smi

_spool_scheduler_DEVICES = ("0", "1", "2", "3")


@pytest.fixture(autouse=True)
def _devices_read_clean(monkeypatch):
    """This host has no device to read: every test device reads as free unless a test says more.

    A reading that fails is "not free" (Clean), so without one no GPU job would ever start.
    """
    monkeypatch.setattr(
        _spool_scheduler_scheduler,
        "read_device_memory_mib",
        lambda: {str(index): 0 for index in range(8)},
    )
    monkeypatch.setattr(_spool_scheduler_scheduler, "read_device_holders", lambda: {})


class _spool_scheduler_FakeClock:
    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def _spool_scheduler_clock():
    return _spool_scheduler_FakeClock()


@pytest.fixture
def _spool_scheduler_spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    return root


def _spool_scheduler_make(spool, clock, **overrides) -> Scheduler:
    environ = overrides.pop("environ", None)
    watcher_id = overrides.pop("watcher_id", None)
    config = SchedulerConfig(
        devices=overrides.pop("devices", _spool_scheduler_DEVICES),
        leftover_grace_s=overrides.pop("leftover_grace_s", 0.0),
        **overrides,
    )
    if environ is None:
        environ = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    return Scheduler(spool, config, clock=clock, environ=environ, watcher_id=watcher_id)


def _spool_scheduler_submit(spool, name, body="true", *, header="", submitted=None) -> Path:
    paths = SpoolPaths(spool)
    text = "#!/bin/bash\nset -u\n" + (f"#spool: {header}\n" if header else "") + body + "\n"
    target = paths.pending / f"{name}.sh"
    staging = paths.pending / f".{name}.tmp"
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, target)
    if submitted is not None:
        os.utime(target, (submitted, submitted))
    return target


def _spool_scheduler_blocking(spool, name) -> str:
    """A body that runs until `release(spool, name)`."""
    flag = Path(spool) / f"release-{name}"
    return f'while [ ! -f "{flag}" ]; do sleep 0.02; done'


def _spool_scheduler_release(spool, name) -> None:
    (Path(spool) / f"release-{name}").write_text("", encoding="utf-8")


def _spool_scheduler_pump(sched: Scheduler, until, timeout: float = 15.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        sched.tick()
        if until():
            return
        time.sleep(0.02)
    raise AssertionError(
        f"condition not reached; status: {json.dumps(sched.snapshot(0), indent=1)}"
    )


def _spool_scheduler_rc(spool, name) -> int | None:
    path = SpoolPaths(spool).receipt(name)
    return int(path.read_text().strip()) if path.exists() else None


def _spool_scheduler_record(spool, name) -> dict:
    return json.loads(SpoolPaths(spool).record(name).read_text())


def running(sched) -> set[str]:
    return {name for name, job in sched.jobs.items() if job.verdict is None}


def drain(sched, spool) -> None:
    """Release every blocking job, queued or running, and let the scheduler close them out."""

    def released_and_idle() -> bool:
        names = set(sched.jobs) | {p.stem for p in SpoolPaths(spool).pending.glob("*.sh")}
        for name in names:
            _spool_scheduler_release(spool, name)
        return not names

    _spool_scheduler_pump(sched, released_and_idle)


@pytest.fixture
def served(_spool_scheduler_spool):
    """A scheduler serving the spool in a background thread, on the real clock."""
    sched = _spool_scheduler_make(_spool_scheduler_spool, time.time)
    sched.announce()
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            sched.tick()
            time.sleep(0.05)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    yield sched
    stop.set()
    thread.join(timeout=5)


def test_an_mpi_attached_process_is_never_signalled(_spool_scheduler_spool, _spool_scheduler_clock):
    sched = _spool_scheduler_make(
        _spool_scheduler_spool, _spool_scheduler_clock, devices=("0", "1")
    )
    _spool_scheduler_submit(
        _spool_scheduler_spool,
        "attached",
        "export SLURM_JOB_ID=1; exec sleep 300",
        header="gpus=1 wall=30",
    )
    _spool_scheduler_pump(sched, lambda: "attached" in running(sched))
    pid = sched.jobs["attached"].proc.pid
    _spool_scheduler_pump(
        sched, lambda: "SLURM_JOB_ID" in (_spool_scheduler_scheduler.read_environ(pid) or {})
    )
    _spool_scheduler_clock.advance(31)
    _spool_scheduler_pump(
        sched, lambda: _spool_scheduler_rc(_spool_scheduler_spool, "attached") is not None
    )
    try:
        assert _spool_scheduler_rc(_spool_scheduler_spool, "attached") == RC_WALL_LIMIT
        assert _spool_scheduler_record(_spool_scheduler_spool, "attached")["unsafe_pids"] == [pid]
        assert os.path.exists(f"/proc/{pid}"), "the attached process must not have been signalled"
        snapshot = sched.snapshot(_spool_scheduler_clock.now)
        assert snapshot["quarantined"] == [sched.jobs["attached"].devices[0]]
        _spool_scheduler_submit(
            _spool_scheduler_spool,
            "next",
            _spool_scheduler_blocking(_spool_scheduler_spool, "next"),
            header="gpus=2 wall=60",
        )
        for _ in range(3):
            sched.tick()
        assert "next" not in running(sched), "quarantined devices are not handed out"
    finally:
        os.kill(pid, signal.SIGKILL)
    _spool_scheduler_pump(sched, lambda: "attached" not in sched.jobs)
    _spool_scheduler_pump(sched, lambda: "next" in running(sched))
    drain(sched, _spool_scheduler_spool)


def test_two_watchers_on_one_spool_run_a_job_once(_spool_scheduler_spool, _spool_scheduler_clock):
    a = _spool_scheduler_make(_spool_scheduler_spool, _spool_scheduler_clock, watcher_id="a")
    b = _spool_scheduler_make(_spool_scheduler_spool, _spool_scheduler_clock, watcher_id="b")
    _spool_scheduler_submit(
        _spool_scheduler_spool,
        "free-once",
        "echo ran >> " + str(Path(_spool_scheduler_spool) / "count"),
        header="gpus=0 wall=60",
    )
    for _ in range(50):
        a.tick()
        b.tick()
        if _spool_scheduler_rc(_spool_scheduler_spool, "free-once") is not None:
            break
        time.sleep(0.02)
    _spool_scheduler_pump(a, lambda: not a.jobs)
    _spool_scheduler_pump(b, lambda: not b.jobs)
    assert (Path(_spool_scheduler_spool) / "count").read_text() == "ran\n"


def reason(sched, clock, name: str) -> str:
    return next(p["reason"] for p in sched.snapshot(clock.now)["pending"] if p["name"] == name)


def test_a_job_states_its_size_and_its_wall_or_is_refused_with_one_line(
    _spool_scheduler_spool, _spool_scheduler_clock
):
    """Nothing is sized by a default: no whole node for a bare name, no two hours for no wall."""
    sched = _spool_scheduler_make(_spool_scheduler_spool, _spool_scheduler_clock)
    _spool_scheduler_submit(_spool_scheduler_spool, "bare", "echo ran")
    _spool_scheduler_submit(_spool_scheduler_spool, "free-bare", "echo ran", header="wall=60")
    _spool_scheduler_submit(_spool_scheduler_spool, "sized", "echo ran", header="gpus=1")
    _spool_scheduler_submit(_spool_scheduler_spool, "whole", "echo ran", header="nodes=1 wall=60")
    _spool_scheduler_pump(
        sched, lambda: _spool_scheduler_rc(_spool_scheduler_spool, "whole") is not None
    )
    bare = _spool_scheduler_record(_spool_scheduler_spool, "bare")
    assert bare["rc"] == RC_REFUSED and bare["detail"] == f"{SIZE_MISSING}; {WALL_MISSING}"
    assert "\n" not in bare["detail"]
    # A `free-` name states nothing either.
    assert _spool_scheduler_record(_spool_scheduler_spool, "free-bare")["detail"] == SIZE_MISSING
    assert _spool_scheduler_record(_spool_scheduler_spool, "sized")["detail"] == WALL_MISSING
    # `nodes=1` is a size: the whole node.
    whole = _spool_scheduler_record(_spool_scheduler_spool, "whole")
    assert whole["rc"] == 0 and whole["devices"] == list(_spool_scheduler_DEVICES)
    _spool_scheduler_pump(sched, lambda: not sched.jobs)


def test_a_device_whose_memory_is_not_back_is_not_free_and_nothing_is_killed(
    _spool_scheduler_spool, _spool_scheduler_clock
):
    """Clean: read each pass, no timer. A reading that fails leaves the device not free."""
    sched = _spool_scheduler_make(
        _spool_scheduler_spool, _spool_scheduler_clock, devices=("0", "1")
    )
    sched.device_memory_mib = lambda: {"0": 40211, "1": 0}
    sched.device_holders = lambda: {"0": [777]}
    _spool_scheduler_submit(
        _spool_scheduler_spool,
        "holder",
        _spool_scheduler_blocking(_spool_scheduler_spool, "holder"),
        header="gpus=1 wall=8h",
    )
    _spool_scheduler_pump(sched, lambda: "holder" in running(sched))
    assert sched.jobs["holder"].devices == ("1",)
    _spool_scheduler_submit(_spool_scheduler_spool, "boot", "echo started", header="gpus=1 wall=60")
    sched.tick()
    waits = "waiting for 1 GPU(s); 0 free; memory not back on: 0 (40211 MiB)"
    assert reason(sched, _spool_scheduler_clock, "boot") == waits
    # However long it lasts: the device is reported with who holds it, and stays out.
    _spool_scheduler_clock.advance(3600)
    sched.tick()
    snapshot = sched.snapshot(_spool_scheduler_clock.now)
    assert reason(sched, _spool_scheduler_clock, "boot") == waits
    assert snapshot["free"] == [] and snapshot["quarantined"] == []
    held = snapshot["held"]["0"]
    assert held["used_mib"] == 40211 and held["pids"] == [777]
    assert held["since"] == pytest.approx(_spool_scheduler_clock.now - 3600)
    assert os.path.exists(f"/proc/{os.getpid()}"), "nothing was signalled"
    # A reading that fails is not a free device.
    sched.device_memory_mib = lambda: None
    sched.tick()
    assert reason(sched, _spool_scheduler_clock, "boot").startswith("no device reading this pass")
    # Nor is a device the reading does not name.
    sched.device_memory_mib = lambda: {"1": 0}
    sched.tick()
    assert reason(sched, _spool_scheduler_clock, "boot").endswith(
        "memory not back on: 0 (no reading)"
    )
    # The memory is back: the device is handed out again, with no step in between.
    sched.device_memory_mib = lambda: {"0": 3, "1": 0}
    _spool_scheduler_pump(
        sched, lambda: _spool_scheduler_rc(_spool_scheduler_spool, "boot") is not None
    )
    assert _spool_scheduler_record(_spool_scheduler_spool, "boot")["devices"] == ["0"]
    assert sched.snapshot(_spool_scheduler_clock.now)["held"] == {}
    drain(sched, _spool_scheduler_spool)


def test_a_cancelled_job_measured_nothing_and_says_so(_spool_scheduler_spool, served):
    """Cancelled through the spool (by the operator, say): no verdict, and no lost session."""
    results = []
    executor = SpoolCommandExecutor(_spool_scheduler_spool, gpus=1, poll_s=0.05)
    thread = threading.Thread(
        target=lambda: results.append(executor.execute(CommandSpec(argv="sleep 30", timeout_s=60)))
    )
    thread.start()
    running = SpoolPaths(_spool_scheduler_spool).running
    deadline = time.time() + 10
    while not list(running.glob("*.sh")) and time.time() < deadline:
        time.sleep(0.05)
    (job,) = running.glob("*.sh")
    spoolctl.cancel_job(_spool_scheduler_spool, job.stem)
    thread.join(timeout=20)
    (result,) = results
    assert result.returncode == RC_CANCELLED and not result.timed_out
    assert result.infrastructure.startswith(CANCELLED)


def _jobs_of(spool: Path) -> set[str]:
    """Every job the spool holds or held, by name."""
    paths = SpoolPaths(spool)
    return {p.stem for d in (paths.pending, paths.running, paths.done) for p in d.glob("*.sh")}


def test_a_job_left_in_the_spool_is_adopted_by_the_next_process_not_queued_twice(
    _spool_scheduler_spool, served, monkeypatch, tmp_path
):
    """A controller that stops while a check's job is queued or running starts again: one job."""
    listed = tmp_path / "selection.txt"
    listed.write_text("tests/test_a.py\n")
    spec = CommandSpec(argv=f"sleep 1; cat {listed}; echo measured", timeout_s=60)
    waited = spoolctl.wait_for_job

    def dies(*args, **kwargs):
        raise RuntimeError("the process waiting for the job is gone")

    # The first process queues the job and dies before it reads the result.
    monkeypatch.setattr(spoolctl, "wait_for_job", dies)
    first = SpoolCommandExecutor(_spool_scheduler_spool, gpus=1, poll_s=0.05, name_prefix="gate")
    with pytest.raises(RuntimeError):
        first.execute(spec)
    (queued,) = _jobs_of(_spool_scheduler_spool)
    # A start says what the last process left there.
    assert priority.left_in_spool(_spool_scheduler_spool) == [queued]
    # The next process makes the same request: it takes that job and queues none beside it.
    monkeypatch.setattr(spoolctl, "wait_for_job", waited)
    notes = []
    second = SpoolCommandExecutor(
        _spool_scheduler_spool, gpus=1, poll_s=0.05, name_prefix="gate", note=notes.append
    )
    result = second.execute(spec)
    assert result.returncode == 0 and "measured" in result.stdout
    assert _jobs_of(_spool_scheduler_spool) == {queued}
    assert notes and f"{queued} on" in notes[0] and "adopted, not queued a second time" in notes[0]
    assert priority.left_in_spool(_spool_scheduler_spool) == []
    # Its result was read: the same request made again is a new job, deliberately.
    assert second.execute(spec).returncode == 0
    assert len(_jobs_of(_spool_scheduler_spool)) == 2


def test_a_finished_job_nobody_read_is_adopted_and_other_inputs_are_another_request(
    _spool_scheduler_spool, served, monkeypatch, tmp_path
):
    listed = tmp_path / "selection.txt"
    listed.write_text("tests/test_a.py\n")
    spec = CommandSpec(argv=f"cat {listed}; echo measured", timeout_s=60)
    waited = spoolctl.wait_for_job
    monkeypatch.setattr(spoolctl, "wait_for_job", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    executor = SpoolCommandExecutor(_spool_scheduler_spool, gpus=1, poll_s=0.05)
    with pytest.raises(OSError):
        executor.execute(spec)
    (queued,) = _jobs_of(_spool_scheduler_spool)
    deadline = time.time() + 20
    while _spool_scheduler_rc(_spool_scheduler_spool, queued) is None and time.time() < deadline:
        time.sleep(0.05)
    assert _spool_scheduler_rc(_spool_scheduler_spool, queued) == 0
    monkeypatch.setattr(spoolctl, "wait_for_job", waited)
    # A job that would read other inputs is another request: the file its command names
    # says something else now, and what the finished job measured is not its result.
    listed.write_text("tests/test_b.py\n")
    other = executor.execute(spec)
    assert "tests/test_b.py" in other.stdout and len(_jobs_of(_spool_scheduler_spool)) == 2
    # The same inputs again: the finished job's receipt is read, and nothing is queued.
    listed.write_text("tests/test_a.py\n")
    result = executor.execute(spec)
    assert result.returncode == 0 and "tests/test_a.py" in result.stdout
    assert len(_jobs_of(_spool_scheduler_spool)) == 2
    # A job of another size is another request too.
    wider = SpoolCommandExecutor(_spool_scheduler_spool, gpus=0, poll_s=0.05)
    assert wider.execute(spec).returncode == 0 and len(_jobs_of(_spool_scheduler_spool)) == 3


def _hold(root: Path, name: str, reason: str) -> None:
    """``name`` queued on ``root``, held by its scheduler for ``reason`` (its ``status.json``)."""
    (root / "pending" / f"{name}.sh").write_text("true\n")
    snapshot = json.loads(SpoolPaths(root).snapshot.read_text())
    snapshot["pending"] = [{"name": name, "reason": reason}]
    _spool_sessions_write_json_atomic(SpoolPaths(root).snapshot, snapshot)


@pytest.mark.usefixtures("_no_ambient_spool")
def test_a_long_wait_on_a_gate_job_is_said_once_per_period(tmp_path):
    """A gate queued past the period says so once, with the session's hold reason."""
    gates = session(tmp_path, "gates-1")
    hold = "the allocation ends in 4974s, too soon for a job allowed 7200s"
    _hold(gates, "gate-core", hold)
    lines: list[str] = []
    waiting = threading.Thread(
        target=spoolctl.wait_for_job,
        args=(gates, "gate-core"),
        kwargs={"poll_s": 0.01, "submitted_at": time.time() - 16 * 60, "note": lines.append},
        daemon=True,
    )
    waiting.start()
    deadline = time.time() + 10
    while not lines and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)  # further polls inside the period say nothing more
    (gates / "done" / "gate-core.rc").write_text("0\n")
    waiting.join(timeout=10)
    assert len(lines) == 1
    assert lines[0].startswith(f"gate-core on {gates} queued since ")
    assert lines[0].endswith(f"; held: {hold}")
    # A wait shorter than the period says nothing; a status record that does not read, no line.
    _hold(gates, "gate-quick", hold)
    (gates / "done" / "gate-quick.rc").write_text("0\n")
    spoolctl.wait_for_job(gates, "gate-quick", submitted_at=time.time() - 14 * 60, note=pytest.fail)
    SpoolPaths(gates).snapshot.write_text('{"pending": [{"reason": "no name"}]}')
    assert spoolctl.waiting_line(gates, "gate-core", time.time()) == ""


@pytest.mark.usefixtures("_no_ambient_spool")
def test_a_queued_job_names_what_runs_ahead_of_it_on_its_spool(tmp_path):
    """A job sat 20 min behind a profiling job an agent had launched, and nothing said so."""
    work = session(tmp_path, "work-1")
    hold = "waiting for the whole node to drain"
    _hold(work, "gate-core", hold)
    running = SpoolPaths(work).running
    (running / "sol-profile.sh").write_text("true\n")
    line = spoolctl.waiting_line(work, "gate-core", time.time())
    assert line.endswith(f"; held: {hold}; running there: sol-profile")
    for name in ("a", "b", "c"):
        (running / f"{name}.sh").write_text("true\n")
    line = spoolctl.waiting_line(work, "gate-core", time.time())
    assert line.endswith("; running there: a, b, c and 1 more")


def _sched(spool, *, job, ends_in=20 * 3600.0, nodes=1) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["SLURM_JOB_ID"] = job
    config = SchedulerConfig(
        devices=("0",),
        session="g3",
        ends_at=time.time() + ends_in,
        leftover_grace_s=0,
        holder=f"h{job}",
        nodes=nodes,
    )
    return Scheduler(spool, config, environ=env, watcher_id=f"w{job}")


def test_a_job_waits_for_a_holder_it_fits_and_an_idle_holder_retires_for_a_later_one(tmp_path):
    """Fit and Retire, read from the files each pass: no hand-over is stored anywhere."""
    spool = tmp_path / "gates-3"
    old = _sched(spool, job="1000027", ends_in=600)
    old.tick()
    assert old._retire_reason(time.time()) == ""

    # A GPU job queued now cannot finish before the allocation ends: it waits where it is.
    (SpoolPaths(spool).pending / "late.sh").write_text(
        "#!/bin/bash\n#spool: gpus=1 wall=2h\ntrue\n"
    )
    old.tick()
    assert "waits for a holder that ends later" in old.pending_reasons["late"]

    later = _sched(spool, job="1000247")
    _spool_scheduler_pump(later, lambda: _spool_scheduler_rc(spool, "late") is not None)
    assert _spool_scheduler_rc(spool, "late") == 0
    # The idle one retires: a live holder of its shape ends later.
    assert old._exit_reason(time.time()) == "holder h1000247 ends later and has as many nodes"
    assert later._exit_reason(time.time()) == "", "the later one has nobody to retire for"
    wide = _sched(spool, job="1000300", ends_in=900, nodes=2)
    wide.tick()
    # A stop marker ends them all once idle, and admits nothing meanwhile.
    SpoolPaths(spool).stop.write_text("now\n")
    (SpoolPaths(spool).pending / "more.sh").write_text(
        "#!/bin/bash\n#spool: gpus=0 wall=60\ntrue\n"
    )
    later.tick()
    assert later.pending_reasons["more"] == "the pool is ending: this watcher takes nothing more"
    assert later._exit_reason(time.time()) == "the pool is ending"
    SpoolPaths(spool).stop.unlink()
    (SpoolPaths(spool).holders).mkdir(exist_ok=True)
    (SpoolPaths(spool).holders / "h1000300.stop").write_text("now\n")
    assert wide._exit_reason(time.time()) == "holder h1000300 was asked to end"
    assert later._exit_reason(time.time()) == ""


def test_monitor_cli_tells_a_busy_orchestrator_from_one_that_has_exited(
    tmp_path, monkeypatch, capsys
):
    from agent_flow.workflows.trtllm_modeling_v2 import cli

    monitor = cli.SUBCOMMANDS["monitor"]
    monkeypatch.delenv("MV2FLOW_WORKSPACE", raising=False)
    assert monitor(["status"]) == 2
    assert "pass --workspace or set MV2FLOW_WORKSPACE" in capsys.readouterr().err
    assert monitor(["--workspace", str(tmp_path / "missing"), "status"]) == 1
    assert "not a trtllm_modeling_v2 workspace" in capsys.readouterr().err

    (tmp_path / "task.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("MV2FLOW_WORKSPACE", str(tmp_path))
    assert monitor(["status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data == {
        "workspace": str(tmp_path.resolve()),
        "orchestrator": {"state": "unknown", "detail": "no orchestrator heartbeat"},
        "onboarded": None,
        "blocked": [],
        "pool": None,
    }
    dispatch = tmp_path / "dispatch"
    dispatch.mkdir()
    _spool_sessions_write_json_atomic(
        dispatch / ORCHESTRATOR_HEARTBEAT, {"pid": 1, "host": "h", "at": time.time()}
    )
    assert status.orchestrator_state(load_config(tmp_path))[0] == "running"
    assert monitor(["wait", "--timeout", "0", "--no-sessions"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "TIMEOUT  nothing new"
    # A closed run changes nothing else, so the waiter must notice the exit.
    _spool_sessions_write_json_atomic(
        dispatch / ORCHESTRATOR_EXIT, {"pid": 1, "host": "h", "at": time.time() + 1}
    )
    assert monitor(["--workspace", str(tmp_path), "wait", "--timeout", "0"]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("ORCHESTRATOR_EXITED  pid 1")


_ALLOCATION = "4242"


@pytest.fixture
def _spool_gangs_spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    return root


def _node(spool, clock, host, *, allocation=_ALLOCATION) -> Scheduler:
    environ = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    environ["SLURM_JOB_ID"] = allocation
    config = SchedulerConfig(
        devices=("0", "1", "2", "3"), leftover_grace_s=0.0, holder="h1", nodes=2
    )
    node = Scheduler(spool, config, clock=clock, environ=environ, watcher_id=f"{host}-w")
    node.host = host
    node.heartbeat()
    return node


def _lead(spool, clock, *, allocation=_ALLOCATION) -> Scheduler:
    """The holder's lead: the batch step's environment, no devices."""
    environ = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "PMI"))}
    environ.update(
        SLURM_JOB_ID=allocation,
        SLURM_JOB_NODELIST="nodeA,nodeB",
        SLURM_PROCID="0",
        SLURM_NTASKS="1",
        SLURM_STEP_ID="4294967291",
        PMIX_RANK="0",
    )
    config = SchedulerConfig(leftover_grace_s=0.0, holder="h1", nodes=2, lead=True)
    lead = Scheduler(spool, config, clock=clock, environ=environ, watcher_id="batch-lead")
    lead.heartbeat()
    return lead


def _pump(nodes, until, timeout: float = 15.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for node in nodes:
            node.tick()
        if until():
            return
        time.sleep(0.02)
    raise AssertionError(f"not reached; pending: {[n.snapshot(0)['pending'] for n in nodes]}")


def test_the_lead_runs_a_multi_node_job_once_and_the_members_hold_their_devices(
    _spool_gangs_spool,
):
    """The job's identity reaches its script, the lead's own task does not; a rank is a step's."""
    a, b = (
        _node(_spool_gangs_spool, time.time, "nodeA"),
        _node(_spool_gangs_spool, time.time, "nodeB"),
    )
    lead = _lead(_spool_gangs_spool, time.time)
    said = _spool_gangs_spool / "said"
    body = (
        f'echo "$MV2FLOW_JOB_NODES $MV2FLOW_JOB_HOSTS job=$SLURM_JOB_ID '
        f"task=${{SLURM_PROCID:-none}} step=${{SLURM_STEP_ID:-none}} rank=${{PMIX_RANK:-none}} "
        f'devices=${{CUDA_VISIBLE_DEVICES:-none}}" >> {said}\n'
        + _spool_scheduler_blocking(_spool_gangs_spool, "pair")
        + "\nexit 3"
    )
    _spool_scheduler_submit(_spool_gangs_spool, "pair", body, header="nodes=2 wall=60")
    nodes = [a, b, lead]
    _pump(nodes, lambda: "pair" in lead.jobs)
    # A node's watcher never starts it, and gives its devices to nothing else meanwhile.
    assert "pair" not in a.jobs and "pair" not in b.jobs
    _pump(nodes, lambda: a._reserved_for == "pair" == b._reserved_for)
    _spool_scheduler_submit(_spool_gangs_spool, "single", "true", header="gpus=1 wall=60")
    for _ in range(3):
        _pump(nodes, lambda: True)
    assert _spool_scheduler_rc(_spool_gangs_spool, "single") is None
    assert "reserved for the multi-node job pair" in a.pending_reasons["single"]
    _spool_scheduler_release(_spool_gangs_spool, "pair")
    _pump(nodes, lambda: _spool_scheduler_rc(_spool_gangs_spool, "pair") is not None)
    receipt = _spool_scheduler_record(_spool_gangs_spool, "pair")
    assert receipt["rc"] == 3 and receipt["watcher"] == "batch-lead"
    assert receipt["nodes"] == 2 and receipt["hosts"] == ["nodeA", "nodeB"]
    assert said.read_text() == (
        "2 nodeA,nodeB job=4242 task=none step=none rank=none devices=none\n"
    ), "run once, by the lead, with the job's identity and nothing of its task"
    # The receipt releases the members: the waiting job starts.
    _pump(nodes, lambda: _spool_scheduler_rc(_spool_gangs_spool, "single") == 0)
    assert not (SpoolPaths(_spool_gangs_spool).gangs / "pair").exists()
    assert a._reserved_for == "" == b._reserved_for


def test_a_lead_may_stop_its_own_launchers_and_never_a_rank(_spool_gangs_spool):
    """The safe-kill rule under a lead: the job's identity is not an MPI context."""
    identity = {"SLURM_JOB_ID": "4242", "SLURM_JOB_NODELIST": "nodeA,nodeB", "SLURM_CONF": "/c"}
    rank = {**identity, "SLURM_PROCID": "3", "PMIX_NAMESPACE": "slurm.pmix.4242.7"}
    attached = _spool_scheduler_scheduler.is_mpi_attached
    assert attached(identity) and attached(rank)
    assert not attached(identity, job_identity=True)
    assert attached(rank, job_identity=True)
    kept = _spool_scheduler_scheduler.lead_environment(
        {**rank, "PATH": "/bin", "OMPI_COMM_WORLD_RANK": "3", "SLURM_NTASKS": "8"}
    )
    assert kept == {**identity, "PATH": "/bin"}


def test_canceling_an_unstarted_gang_releases_the_drained_node(_spool_gangs_spool):
    a, b = (
        _node(_spool_gangs_spool, time.time, "nodeA"),
        _node(_spool_gangs_spool, time.time, "nodeB"),
    )
    lead = _lead(_spool_gangs_spool, time.time)
    _spool_scheduler_submit(
        _spool_gangs_spool,
        "busy",
        _spool_scheduler_blocking(_spool_gangs_spool, "busy"),
        header="gpus=all wall=60s",
    )
    _pump([a], lambda: "busy" in a.jobs)
    _spool_scheduler_submit(_spool_gangs_spool, "pair", "true", header="nodes=2 wall=60s")
    _pump([a, b, lead], lambda: b._reserved_for == "pair")
    paths = SpoolPaths(_spool_gangs_spool)
    assert not (paths.gangs / "pair/go.json").exists()
    assert "waiting for nodeA to drain and read clean for it" in lead.pending_reasons["pair"]
    (paths.cancel / "pair").touch()
    _pump([a, b, lead], lambda: _spool_scheduler_rc(_spool_gangs_spool, "pair") is not None)
    assert _spool_scheduler_record(_spool_gangs_spool, "pair")["reason"] == "cancelled"
    assert "busy" in a.jobs
    _spool_scheduler_submit(_spool_gangs_spool, "after", "true", header="gpus=all wall=30s")
    _pump([b, a, lead], lambda: _spool_scheduler_rc(_spool_gangs_spool, "after") is not None)
    assert (
        _spool_scheduler_rc(_spool_gangs_spool, "after") == 0
        and _spool_scheduler_record(_spool_gangs_spool, "after")["host"] == "nodeB"
    )
    _spool_scheduler_release(_spool_gangs_spool, "busy")
    _pump([a, b, lead], lambda: _spool_scheduler_rc(_spool_gangs_spool, "busy") is not None)
    assert (
        _spool_scheduler_rc(_spool_gangs_spool, "busy") == 0 and not (paths.gangs / "pair").exists()
    )


def test_a_gang_closed_while_a_member_reads_its_devices_does_not_end_that_watcher(
    _spool_gangs_spool,
):
    """A member reading its devices when the lead closes the job serves on.

    The lead removes a gang's record when the job is over or cancelled. The member then has
    nothing left to be ready for; a watcher that died there would take its node's jobs along.
    """
    a, b = (
        _node(_spool_gangs_spool, time.time, "nodeA"),
        _node(_spool_gangs_spool, time.time, "nodeB"),
    )
    lead = _lead(_spool_gangs_spool, time.time)
    _spool_scheduler_submit(
        _spool_gangs_spool,
        "busy",
        _spool_scheduler_blocking(_spool_gangs_spool, "busy"),
        header="gpus=all wall=60s",
    )
    _pump([a], lambda: "busy" in a.jobs)
    _spool_scheduler_submit(_spool_gangs_spool, "pair", "true", header="nodes=2 wall=60s")
    _pump([a, b, lead], lambda: b._reserved_for == "pair")
    gang = SpoolPaths(_spool_gangs_spool).gangs / "pair"
    read_clean = b._drained_and_clean

    def closed_meanwhile(now):
        shutil.rmtree(gang)
        return read_clean(now)

    b._drained_and_clean = closed_meanwhile
    b.tick()
    assert b._reserved_for == "" and not gang.exists()
    b._drained_and_clean = read_clean
    _spool_scheduler_release(_spool_gangs_spool, "busy")
    _pump([a, b, lead], lambda: _spool_scheduler_rc(_spool_gangs_spool, "busy") is not None)


def test_a_lead_ends_its_allocation_only_once_its_watchers_are_gone(_spool_gangs_spool):
    """Retire for a holder of several nodes: the watchers leave first, then the lead."""
    a = _node(_spool_gangs_spool, time.time, "nodeA")
    lead = _lead(_spool_gangs_spool, time.time)
    SpoolPaths(_spool_gangs_spool).stop.write_text("now\n")
    assert a._exit_reason(time.time()) == "the pool is ending"
    assert lead._exit_reason(time.time()) == "", "a watcher of its holder still heart-beats"
    (SpoolPaths(_spool_gangs_spool).watchers / "nodeA-w.json").unlink()
    assert lead._exit_reason(time.time()) == "the pool is ending"


DEAD = "deadhost-1"


@pytest.fixture
def _spool_requeue_clock():
    return _spool_scheduler_FakeClock()


@pytest.fixture
def _spool_requeue_spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    return root


def lost_run(spool, clock, name: str, body: str = "echo again", reruns: int = 0) -> None:
    """``name`` as the next scheduler finds it: started by ``DEAD``, silent for an hour."""
    paths = SpoolPaths(spool)
    script = paths.running / f"{name}.sh"
    stamp = f"{RERUN_TAG} {reruns}\n" if reruns else ""
    script.write_text(
        f"#!/bin/bash\n{stamp}set -u\n#spool: gpus=0 wall=60 token=tok-1\n{body}\n",
        encoding="utf-8",
    )
    paths.log(name).write_text("=== start on deadhost ===\nthe first run, cut off\n")
    _spool_requeue_write_json_atomic(
        paths.running / f"{name}.json",
        {
            "name": name,
            "watcher": DEAD,
            "host": "deadhost",
            "pid": 4242,
            "devices": ["0", "3"],
            "started": "2026-10-18T21:33:35Z",
            "started_epoch": clock.now - 900,
            "token": "tok-1",
            "owner": "",
            "slurm_job": "100",
            "session": "w2",
        },
    )
    _spool_requeue_write_json_atomic(
        paths.watchers / f"{DEAD}.json",
        {"watcher": DEAD, "slurm_job": "100", "alive_epoch": clock.now - 3600},
    )


def test_a_job_lost_with_its_allocation_runs_again_once(_spool_requeue_spool, _spool_requeue_clock):
    lost_run(_spool_requeue_spool, _spool_requeue_clock, "free-r063-u1b", "echo second-run")
    submitted = (SpoolPaths(_spool_requeue_spool).running / "free-r063-u1b.sh").stat().st_mtime
    sched = _spool_scheduler_make(_spool_requeue_spool, _spool_requeue_clock, session="w2")
    sched._recover_orphans(_spool_requeue_clock.now, {})
    queued = SpoolPaths(_spool_requeue_spool).pending / "free-r063-u1b.sh"
    assert f"{RERUN_TAG} 1\n" in queued.read_text()
    assert queued.stat().st_mtime == submitted, "a re-run keeps its place in the queue"
    _spool_scheduler_pump(
        sched, lambda: _spool_scheduler_rc(_spool_requeue_spool, "free-r063-u1b") is not None
    )
    paths = SpoolPaths(_spool_requeue_spool)
    assert _spool_scheduler_rc(_spool_requeue_spool, "free-r063-u1b") == 0
    assert _spool_scheduler_record(_spool_requeue_spool, "free-r063-u1b")["reruns"] == 1
    # The cut-off run's log is kept in history/, ending with what became of it.
    (lost_log,) = paths.history.glob("free-r063-u1b.*.log")
    assert "the first run, cut off" in lost_log.read_text()
    assert "queued again (re-run 1 of at most 1)" in lost_log.read_text()
    assert "second-run" in paths.log("free-r063-u1b").read_text()
    assert "free-r063-u1b lost" in paths.status.read_text()
    _spool_scheduler_pump(sched, lambda: not sched.jobs)


IMPORT = "python3 -c 'import tensorrt_llm'"


@pytest.fixture
def checkouts(tmp_path):
    trees = {}
    for name in ("main", "wt"):
        package = tmp_path / name / "tensorrt_llm"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"WHERE = {name!r}\n", encoding="utf-8")
        trees[name] = tmp_path / name
    return trees


@pytest.fixture
def _spool_tree_spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    return root


def _spool_tree_scheduler(spool) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "PYTHON"))}
    return Scheduler(spool, SchedulerConfig(devices=("0",), leftover_grace_s=0), environ=env)


def run_job(spool, name, body, header) -> dict:
    paths = SpoolPaths(spool)
    (paths.pending / f"{name}.sh").write_text(
        f"#!/bin/bash\n#spool: {header}\n{body}\n", encoding="utf-8"
    )
    sched = _spool_tree_scheduler(spool)
    end = time.monotonic() + 30
    while time.monotonic() < end:
        sched.tick()
        if paths.receipt(name).exists() and not sched.jobs:
            return json.loads(paths.record(name).read_text())
        time.sleep(0.02)
    raise AssertionError(paths.log(name).read_text() if paths.log(name).exists() else "no log")


def test_ranks_that_resolved_different_checkouts_are_refused(_spool_tree_spool, checkouts):
    """The mixed shape: one rank imports the tree, another the main checkout via its cwd."""
    body = f"{IMPORT}; (cd {checkouts['main']} && {IMPORT})"
    record = run_job(_spool_tree_spool, "mixed", body, f"gpus=1 wall=60 tree={checkouts['wt']}")
    assert record["rc"] == RC_MIXED_CHECKOUT
    assert record["reason"] == "mixed_checkout"
    assert "2 checkouts" in record["detail"]
    assert str(checkouts["main"]) in record["detail"] and str(checkouts["wt"]) in record["detail"]


def test_an_engine_job_hears_from_every_rank_it_states_or_has_no_result(
    _spool_tree_spool, checkouts
):
    """Origin: a missing record is no result, never a pass; one source and one build."""
    tree = checkouts["wt"]
    (tree / "tensorrt_llm" / "libs").mkdir()
    (tree / "tensorrt_llm" / "libs" / "libth_common.so").write_bytes(b"build-one")
    as_rank = "MV2FLOW_ROLE={role} PMIX_RANK={rank} " + IMPORT
    header = f"gpus=1 wall=60 tree={tree} ranks=ctx0:2,gen0:1"
    # One rank of ctx0 never imported the package.
    body = "; ".join(as_rank.format(role=r, rank=k) for r, k in (("ctx0", 0), ("gen0", 0)))
    record = run_job(_spool_tree_spool, "short", body, header)
    assert record["rc"] == RC_NO_ORIGIN and record["reason"] == "no_origin"
    assert record["detail"].startswith(
        "no origin record from ctx0[1]: the job states ctx0:2,gen0:1"
    )
    # Each (role, instance, world rank) reported: the result stands.
    full = "; ".join(
        as_rank.format(role=r, rank=k) for r, k in (("ctx0", 0), ("ctx0", 1), ("gen0", 0))
    )
    assert run_job(_spool_tree_spool, "full", full, header)["rc"] == 0
    # A rank that loaded another build is refused, as one that imported another source is.
    rebuilt = (
        full
        + f"; echo other > {tree}/tensorrt_llm/libs/libth_common.so; "
        + as_rank.format(role="gen0", rank=0)
    )
    record = run_job(_spool_tree_spool, "rebuilt", rebuilt, header)
    assert record["reason"] == "mixed_checkout" and "2 native builds" in record["detail"]

    # Complete evidence or none: a record of another module, one with no build identity and
    # one whose rank is no number satisfy no rank; a lone process is rank 0 of a world of one.
    def row(rank, **changes):
        base = {"module": "tensorrt_llm", "origin": "/t/tensorrt_llm/__init__.py", "build": "b1"}
        return {**base, "role": "ctx0", "rank": rank, **changes}

    missing = _spool_scheduler_scheduler._provenance.missing_ranks
    stated, watched = [("ctx0", 2)], ("tensorrt_llm",)
    assert missing([row("0"), row(1)], stated, watched) == []
    assert missing([row("0"), row("1", module="numpy")], stated, watched) == ["ctx0[1]"]
    assert missing([row("0"), row("1", build=None)], stated, watched) == ["ctx0[1]"]
    assert missing([row("0"), row("one")], stated, watched) == ["ctx0[1]"]
    assert missing([row("-")], stated, watched) == ["ctx0[0]", "ctx0[1]"]
    assert missing([row("-", role=None)], [("engine", 1)], watched) == []
    # A job that states no ranks is judged as before: by what did report.
    assert run_job(_spool_tree_spool, "plain", "true", f"gpus=1 wall=60 tree={tree}")["rc"] == 0


def test_a_spool_run_puts_its_commands_in_the_spool(tmp_path, task_file, monkeypatch):
    """`--dry-run` stops once preflight has resolved the executor; the survey's probe is stubbed.

    It stops before the first agent turn and leaves the checkpoint at preflight, so the real
    run re-runs it.
    """
    spool = tmp_path / "spool"
    spool.mkdir()
    spec = yaml.safe_load(task_file.read_text())
    spec["environment"].update(kind="local", spool=str(spool))
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    monkeypatch.setattr(preflight_stage, "run_inventory_probe", lambda *a, **k: Inventory())
    workflow = TrtllmModelingV2Workflow(
        workspace=tmp_path / "ws",
        command_runner=healthy_smi(),
        dry_run=True,
    )
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)

    assert calls == [] and _state(tmp_path).stage == STAGE_PREFLIGHT
    assert isinstance(workflow._command_executor, SpoolCommandExecutor)
    assert workflow._command_executor.root == spool

    # And the gate runner really goes there: nothing is watching, so it
    # times out, but the work landed as a file.
    result = workflow._shell_runner("echo gate-command", None, 1)
    assert result.timed_out
    scripts = sorted((spool / "pending").glob("*.sh"))
    assert scripts and "echo gate-command" in scripts[-1].read_text(encoding="utf-8")


def test_cpu_introspection_does_not_reserve_a_gpu(tmp_path):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment

    flow = SimpleNamespace(
        _shell_runner_override=None, task_data={"trtllm_repo_path": str(tmp_path)}
    )
    result = SimpleNamespace(environment=RunEnvironment("slurm", "/p", spool=str(tmp_path / "s")))
    runner = preflight_stage._cpu_probe_runner(flow, result)
    runner("true", None, 0)
    script = next((tmp_path / "s" / "pending").glob("*.sh")).read_text()
    assert "gpus=0 wall=0" in script

    def supplied(*args):
        return CommandResult(0)

    flow._shell_runner_override = supplied
    assert preflight_stage._cpu_probe_runner(flow, result) is supplied


def test_every_command_scrubs_the_allocations_launcher_environment():
    """Otherwise every rank is rank 0 of a world of 1, and the run hangs.

    `trtllm-llmapi-launch` reads `SLURM_PROCID` before
    `OMPI_COMM_WORLD_RANK` and `SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE`.
    Inside a one-task allocation those are `0` and `1` for every rank, so all
    four become `mgmn_leader_node` and wait forever for workers.

    mpi4py auto-initializes whenever PMI variables are present. A spool script
    runs inside the session's own allocation, so they are, and a lone probe
    would join a world it never asked for.
    """
    env = RunEnvironment(kind="slurm", python="/p", world_size=4, spool="/s")
    command = env.command("/p x.py")
    for name in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID", "SLURM_NODEID"):
        assert f"-u {name}" in command, name
    assert command.index(MULTI_TASK_SLURM_SCRUB) < command.index("mpirun")
    single = env.command("/p probe.py", single_process=True)
    assert "${!PMIX@}" in single
    assert "exec /p probe.py" in single


def test_spool_exports_preserve_shell_quotes():
    import re

    env = RunEnvironment(kind="slurm", python="/p", spool="/s", env={"EXAMPLE": "a b'$c"})
    text = env.describe()
    block = next(code for code in re.findall(r"```bash\n(.*?)```", text, re.S) if "export " in code)
    found = subprocess.run(
        ["bash", "-c", block + '\nprintf "%s" "$EXAMPLE"'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert found.stdout == "a b'$c"


def test_hung_gpu_inspection_returns_unknown_instead_of_blocking_teardown(monkeypatch):
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import server_runtime

    def hung(argv, **kwargs):
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(argv, 30)

    monkeypatch.setattr(server_runtime.subprocess, "run", hung)
    assert server_runtime.device_holders() == set()


def test_a_command_that_outlives_its_deadline_is_timed_out_not_merely_failed():
    started = time.monotonic()
    record = LocalCommandExecutor().execute(CommandSpec(argv="sleep 30", timeout_s=1))
    assert record.timed_out
    assert record.returncode == TIMEOUT_RETURNCODE
    assert not record.ok
    assert time.monotonic() - started < 20
    # A verdict names a stall by its cause, not by a time limit the job never reached.
    assert ended_early(record, 1) == "timed out after 1s"
    stalled = dataclasses.replace(record, stalled="no progress for 601s after a failure line")
    assert ended_early(stalled, 14400) == "stalled: no progress for 601s after a failure line"


_execution_DEVICES = ("0", "1", "2", "3")


_execution_HOUR = 3600.0


def session(
    parent: Path,
    name: str,
    *,
    left_s: float = 20 * _execution_HOUR,
    age_s: float = 0.0,
) -> Path:
    """A spool watched by one scheduler that has just ticked ``age_s`` ago."""
    root = parent / name
    now = time.time()
    config = SchedulerConfig(devices=_execution_DEVICES, session=name, ends_at=now + left_s)
    Scheduler(
        root,
        config,
        clock=lambda: now - age_s,
        environ={"SLURM_JOB_ID": "100"},
        watcher_id=f"{name}-w",
    ).tick()
    return root


def pending(root: Path) -> list[str]:
    return sorted(p.name[: -len(".sh")] for p in (root / "pending").glob("*.sh"))


@pytest.fixture
def _no_ambient_spool(monkeypatch):
    for key in ("SPOOL", "MV2FLOW_SPOOL", "MV2FLOW_JOB_PREFIX"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.usefixtures("_no_ambient_spool")
def test_a_dead_watcher_refuses_the_job(tmp_path, capsys):
    work1 = session(tmp_path, "work-1", age_s=600)
    launch = ["--spool", str(work1), "launch", "--gpus", "1", "--wall", "60", "r1-x", "true"]
    code = spoolctl.main(launch)
    assert code == spoolctl.EXIT_NO_SESSION
    assert pending(work1) == []
    assert "stale: the session is gone" in capsys.readouterr().err
    # Not while the run's controller is getting the spool a holder: the job queues for it.
    _spool_sessions_write_json_atomic(
        SpoolPaths(work1).pool, {"at": time.time(), "stopping": False}
    )
    assert spoolctl.main(launch) == 0 and pending(work1) == ["r1-x"]
    executor = SpoolCommandExecutor(work1, gpus=1)
    _spool_sessions_write_json_atomic(SpoolPaths(work1).pool, {"at": time.time(), "stopping": True})
    result = executor.execute(CommandSpec(argv="true", timeout_s=5))
    assert result.timed_out and "nothing was queued" in result.infrastructure
    assert pending(work1) == ["r1-x"]


@pytest.mark.usefixtures("_no_ambient_spool")
def test_a_launched_job_queues_at_the_priority_asked(tmp_path):
    work1 = session(tmp_path, "work-1")
    code = spoolctl.main(
        ["--spool", str(work1), "launch", "--gpus", "1", "--wall", "10m", "--priority", "2"]
        + ["--ranks", "engine:4", "r1-x", "true"]
    )
    assert code == 0 and pending(work1) == ["r1-x"]
    header = (work1 / "pending" / "r1-x.sh").read_text().split("#spool:", 1)[1].splitlines()[0]
    assert {"gpus=1", "wall=600", "priority=2", "ranks=engine:4"} <= set(header.split())


@pytest.mark.usefixtures("_no_ambient_spool")
def test_the_client_refuses_a_job_that_states_no_size_or_no_wall(tmp_path, capsys):
    """The same one line the scheduler would refuse it with, before anything is queued."""
    work1 = session(tmp_path, "work-1")
    for options, line in (
        (["--wall", "60"], SIZE_MISSING),
        (["--gpus", "1"], WALL_MISSING),
        (["--nodes", "2"], WALL_MISSING),
    ):
        code = spoolctl.main(["--spool", str(work1), "launch", *options, "r1-x", "true"])
        assert code == spoolctl.EXIT_REFUSED and pending(work1) == []
        assert capsys.readouterr().err.strip() == f"spoolctl: job 'r1-x' {line}"


def test_every_caller_states_its_size(tmp_path):
    """No whole-node default: a stated size."""
    env = RunEnvironment("slurm", "/p", spool=str(tmp_path / "main"), world_size=4)
    with pytest.raises(ValueError, match="a spool job states its size"):
        priority.select_command_executor(env)
    assert priority.select_command_executor(env, gpus=4).gpus == "4"
    assert priority.select_command_executor(env, gpus="all").gpus == "all"
    # One node with a device count is a device job; whole nodes carry no count.
    sized = priority.select_command_executor(env, gpus=2, nodes=1)
    assert (sized.gpus, sized.nodes) == ("2", 0)
    with pytest.raises(Exception, match="needs environment.slurm: the flow starts engines"):
        priority.select_command_executor(env, nodes=2).execute(CommandSpec("true", timeout_s=5))
    # A GPU command with no deadline states no wall.
    with pytest.raises(ValueError, match="states a wall"):
        sized.execute(CommandSpec("true", timeout_s=0))
    # Outside a run, `local` with no spool runs here; any other kind needs its spool.
    assert isinstance(
        priority.select_command_executor(RunEnvironment("local", "/p"), gpus=1),
        LocalCommandExecutor,
    )
    with pytest.raises(ValueError, match="names no spool"):
        priority.select_command_executor(RunEnvironment("slurm", "/p"), gpus=1)


def test_a_command_on_several_nodes_is_one_step_per_node_run_by_the_lead(tmp_path):
    """Main's one multi-node check keeps its contract: its command once on each node."""
    slurm = {"account": "ACCOUNT", "partition": "PARTITION", "time": "01:00:00", "gpus_per_node": 4}
    env = RunEnvironment("slurm", "/p", spool=str(tmp_path / "spool"), slurm=slurm)
    SpoolPaths(env.spool).ensure()
    _spool_sessions_write_json_atomic(
        SpoolPaths(env.spool).pool, {"at": time.time(), "stopping": False}
    )
    executor = priority.select_command_executor(env, nodes=2, name_prefix="disagg")
    threading.Thread(
        target=executor.execute, args=(CommandSpec("pytest -q", timeout_s=30),), daemon=True
    ).start()
    deadline = time.time() + 10
    queued = list((tmp_path / "spool" / "pending").glob("*.sh"))
    while not queued and time.time() < deadline:
        time.sleep(0.05)
        queued = list((tmp_path / "spool" / "pending").glob("*.sh"))
    script = queued[0].read_text()
    assert "nodes=2 wall=30" in script and "gpus=" not in script
    steps = [line for line in script.splitlines() if line.startswith("srun ")]
    assert len(steps) == 2
    for rank, step in enumerate(steps):
        assert step.startswith(
            "srun --overlap --mpi=none --kill-on-bad-exit=1 --nodes=1 --ntasks=1"
        )
        assert f'--nodelist="${{MV2FLOW_HOSTS[{rank}]}}"' in step
        assert f"env MV2FLOW_JOB_NODE_RANK={rank} bash " in step and step.endswith("node.sh &")
    inner = next((tmp_path / "spool" / "jobs").glob("*/node.sh")).read_text()
    assert inner.rstrip().endswith("pytest -q")


class _Stage(CheckRunner):
    def __init__(self):
        self.task_data = {
            "deployments": [{"name": "core", "options": {}}, {"name": "mtp", "options": {}}]
        }
        self.seen = []

    def _run_contract_check(self, check, log):
        self.seen.append((check.key, priority.current_job_priority()))
        return SimpleNamespace(passed=True)


def test_the_first_units_inner_checks_go_ahead_of_every_other():
    stage = _Stage()
    for check in (
        c.Check("core", "deployment", "tp1"),
        c.Check("core", "tests", "tp1"),
        c.Check("mtp", "deployment", "tp1"),
        c.Check("mtp", "tests", "tp1"),
    ):
        stage._prioritized_check(check, None)
    # The first deployment's session is on the way to `onboarded`.
    assert stage.seen == [
        ("core:deployment@tp1", priority.ACCURACY),
        ("core:tests@tp1", priority.FEATURE),
        ("mtp:deployment@tp1", priority.FEATURE),
        ("mtp:tests@tp1", priority.FEATURE),
    ]
    # Outside a check: the scheduler's default, a worker's own jobs.
    assert priority.current_job_priority() == 0
