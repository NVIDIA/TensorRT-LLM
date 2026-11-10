# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 spool scheduler behavior and regression checks."""

from __future__ import annotations

import dataclasses
import json
import os
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
    placement,
    rebuild,
    sessions,
    spoolctl,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import (
    scheduler as _spool_scheduler_scheduler,
)
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
    DEVICE_RELEASE_WAIT_S,
    RC_CANCELLED,
    RC_GANG_STOPPED,
    RC_MIXED_CHECKOUT,
    RC_WALL_LIMIT,
    RERUN_TAG,
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
    parse_duration,
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
from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap as preflight_stage
from agent_flow.workflows.trtllm_modeling_v2.run.state import STAGE_PREFLIGHT
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from ._helpers import Script, _install, _state, healthy_smi

_spool_scheduler_DEVICES = ("0", "1", "2", "3")


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
            header="gpus=2",
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


def test_a_device_that_never_releases_its_memory_is_quarantined(
    _spool_scheduler_spool, _spool_scheduler_clock
):
    sched = _spool_scheduler_make(
        _spool_scheduler_spool, _spool_scheduler_clock, devices=("0", "1")
    )
    sched.device_memory_mib = lambda: {"0": 40211, "1": 0}
    _spool_scheduler_submit(
        _spool_scheduler_spool,
        "holder",
        _spool_scheduler_blocking(_spool_scheduler_spool, "holder"),
        header="gpus=1",
    )
    _spool_scheduler_pump(sched, lambda: "holder" in running(sched))
    assert sched.jobs["holder"].devices == ("1",)
    _spool_scheduler_submit(_spool_scheduler_spool, "boot", "echo started", header="gpus=1")
    sched.tick()
    assert reason(sched, _spool_scheduler_clock, "boot").startswith(
        "device memory in use: 0 (40211 MiB)"
    )
    _spool_scheduler_clock.advance(DEVICE_RELEASE_WAIT_S + 1)
    sched.tick()
    snapshot = sched.snapshot(_spool_scheduler_clock.now)
    assert snapshot["quarantined"] == ["0"] and snapshot["free"] == []
    assert snapshot["quarantined_memory"]["0"]["used_mib"] == 40211
    assert reason(sched, _spool_scheduler_clock, "boot") == (
        "waiting for 1 GPU(s); 0 free; quarantined, memory not released: 0 (40211 MiB)"
    )
    assert "watcher: device 0 quarantined" in SpoolPaths(_spool_scheduler_spool).status.read_text()
    # Placed elsewhere: the device the holder gives back, not the quarantined one.
    _spool_scheduler_release(_spool_scheduler_spool, "holder")
    _spool_scheduler_pump(
        sched, lambda: _spool_scheduler_rc(_spool_scheduler_spool, "boot") is not None
    )
    assert _spool_scheduler_record(_spool_scheduler_spool, "boot")["devices"] == ["1"]
    assert sched.snapshot(_spool_scheduler_clock.now)["quarantined"] == ["0"]


def test_a_cancelled_job_measured_nothing_and_says_so(_spool_scheduler_spool, served):
    """Cancelled through the spool (by the operator, say): no verdict, and no lost session."""
    results = []
    executor = SpoolCommandExecutor(_spool_scheduler_spool, poll_s=0.05)
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


def _task(tmp_path):
    return {
        "trtllm_repo_path": str(tmp_path / "repo"),
        "environment": {
            "kind": "session_spool",
            "python": "/opt/venv/bin/python",
            "spool": str(tmp_path / "spool-a"),
            "env": {"PATH": "/opt/venv/bin"},
            "sessions": [{"name": "a", "spool": str(tmp_path / "spool-a"), "role": "gates"}],
        },
    }


def test_session_cli_prepares_a_missing_spool_without_submitting(tmp_path, monkeypatch, capsys):
    from agent_flow.workflows.trtllm_modeling_v2 import cli

    def no_submit(*args, **kwargs):
        raise AssertionError("session commands must not submit jobs")

    monkeypatch.setattr(subprocess, "run", no_submit)
    monkeypatch.setattr(subprocess, "Popen", no_submit)
    task = _task(tmp_path)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task), encoding="utf-8")
    spool = tmp_path / "spool-a"
    session = cli.SUBCOMMANDS["session"]
    options = ["--task", str(path), "--name", "a"]
    assert session(["render", *options, "--stdout"]) == 0
    assert "exec" in capsys.readouterr().out
    assert not spool.exists()
    assert session(["render", *options]) == 0
    assert (spool / "session_entry.sh").is_file()
    for removed in ("submission", "allocate", "ack"):
        with pytest.raises(SystemExit):
            session([removed, *options])


def test_session_status_follows_the_latest_watcher(tmp_path):
    task = _task(tmp_path)
    spool = tmp_path / "spool-a"
    (spool / "watchers").mkdir(parents=True)
    (spool / "status.json").write_text(
        json.dumps({"watcher": "old", "host": "old-node", "alive_epoch": 100, "slurm_job": "1"})
    )
    (spool / "watchers" / "successor.json").write_text(
        json.dumps({"watcher": "new", "host": "new-node", "alive_epoch": 190, "slurm_job": "2"})
    )
    status = sessions.session_status(task, now=200)[0]
    assert status["slurm_job"] == "2" and status["heartbeat_age_s"] == 10


_spool_sessions_HOUR = 3600.0


def _handover_task(tmp_path):
    spool = str(tmp_path / "gates-3")
    return {
        "environment": {
            "kind": "session_spool",
            "spool": spool,
            "sessions": [{"name": "g3", "spool": spool, "role": "gates", "time_limit": "24:00:00"}],
        },
    }


def _sched(spool, *, job, ends_in=20 * _spool_sessions_HOUR) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["SLURM_JOB_ID"] = job
    config = SchedulerConfig(
        devices=("0",),
        session="g3",
        ends_at=time.time() + ends_in,
        handover_lead_s=_spool_sessions_HOUR,
        leftover_grace_s=0,
    )
    return Scheduler(spool, config, environ=env, watcher_id=f"w{job}")


def test_an_ending_session_is_handed_over_to_the_next_session_on_its_spool(tmp_path):
    task, spool = _handover_task(tmp_path), tmp_path / "gates-3"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    cfg = status.load_config(workspace)

    old = _sched(spool, job="1000027", ends_in=600)
    old.tick()
    alert = status.poll_once(cfg)
    assert alert is not None and alert.kind == "SESSION_ENDING"
    assert "no newer session watches" in alert.detail
    assert str(spool / "session_entry.sh") in alert.detail

    # A GPU job queued now cannot finish before the allocation ends: it waits for the successor.
    (SpoolPaths(spool).pending / "late.sh").write_text(
        "#!/bin/bash\n#spool: gpus=1 wall=2h\ntrue\n"
    )
    old.tick()
    assert "waits for the successor" in old.pending_reasons["late"]

    successor = _sched(spool, job="1000247")
    _spool_scheduler_pump(successor, lambda: _spool_scheduler_rc(spool, "late") is not None)
    assert _spool_scheduler_rc(spool, "late") == 0
    # Once the successor is up, the ending session admits nothing more.
    (SpoolPaths(spool).pending / "free-short.sh").write_text("#!/bin/bash\ntrue\n")
    old.tick()
    assert old.handover["successor"] == successor.watcher_id
    assert status.poll_once(cfg) is None, "a newer session on the spool answers the alert"
    assert (
        old.pending_reasons["free-short"]
        == f"this session is handing over to {successor.watcher_id}"
    )


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
        "run": None,
        "sessions": [],
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
    config = SchedulerConfig(devices=("0", "1", "2", "3"), leftover_grace_s=0.0)
    node = Scheduler(spool, config, clock=clock, environ=environ, watcher_id=f"{host}-w")
    node.host = host
    node.heartbeat()
    return node


def _pump(nodes, until, timeout: float = 15.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for node in nodes:
            node.tick()
        if until():
            return
        time.sleep(0.02)
    raise AssertionError(f"not reached; pending: {[n.snapshot(0)['pending'] for n in nodes]}")


def test_a_failed_part_stops_the_others_and_names_the_node(_spool_gangs_spool):
    a, b = (
        _node(_spool_gangs_spool, time.time, "nodeA"),
        _node(_spool_gangs_spool, time.time, "nodeB"),
    )
    body = 'if [ "$MV2FLOW_JOB_NODE_RANK" = 1 ]; then exit 3; fi; sleep 30'
    _spool_scheduler_submit(_spool_gangs_spool, "pair", body, header="nodes=2")
    _pump([a, b], lambda: _spool_scheduler_rc(_spool_gangs_spool, "pair") is not None, timeout=40.0)
    receipt = _spool_scheduler_record(_spool_gangs_spool, "pair")
    assert receipt["rc"] == 3 and receipt["detail"].startswith("node 1 (nodeB)")
    assert {p["rank"]: p["rc"] for p in receipt["parts"]} == {0: RC_GANG_STOPPED, 1: 3}


def test_canceling_an_unstarted_gang_releases_the_drained_node(_spool_gangs_spool):
    a, b = (
        _node(_spool_gangs_spool, time.time, "nodeA"),
        _node(_spool_gangs_spool, time.time, "nodeB"),
    )
    _spool_scheduler_submit(
        _spool_gangs_spool,
        "busy",
        _spool_scheduler_blocking(_spool_gangs_spool, "busy"),
        header="gpus=all wall=60s",
    )
    _pump([a], lambda: "busy" in a.jobs)
    _spool_scheduler_submit(_spool_gangs_spool, "pair", "true", header="nodes=2 wall=60s")
    _pump([a, b], lambda: b._reserved_for == "pair")
    paths = SpoolPaths(_spool_gangs_spool)
    assert not (paths.gangs / "pair/go.json").exists()
    (paths.cancel / "pair").touch()
    _pump([a, b], lambda: _spool_scheduler_rc(_spool_gangs_spool, "pair") is not None)
    assert _spool_scheduler_record(_spool_gangs_spool, "pair")["reason"] == "cancelled"
    assert "busy" in a.jobs
    _spool_scheduler_submit(_spool_gangs_spool, "after", "true", header="gpus=all wall=30s")
    _pump([b, a], lambda: _spool_scheduler_rc(_spool_gangs_spool, "after") is not None)
    assert (
        _spool_scheduler_rc(_spool_gangs_spool, "after") == 0
        and _spool_scheduler_record(_spool_gangs_spool, "after")["host"] == "nodeB"
    )
    _spool_scheduler_release(_spool_gangs_spool, "busy")
    _pump([a, b], lambda: _spool_scheduler_rc(_spool_gangs_spool, "busy") is not None)
    assert (
        _spool_scheduler_rc(_spool_gangs_spool, "busy") == 0 and not (paths.gangs / "pair").exists()
    )


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
        f"#!/bin/bash\n{stamp}set -u\n#spool: token=tok-1\n{body}\n", encoding="utf-8"
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
            "restart": 0,
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
    record = run_job(_spool_tree_spool, "mixed", body, f"gpus=1 tree={checkouts['wt']}")
    assert record["rc"] == RC_MIXED_CHECKOUT
    assert record["reason"] == "mixed_checkout"
    assert "2 checkouts" in record["detail"]
    assert str(checkouts["main"]) in record["detail"] and str(checkouts["wt"]) in record["detail"]


def test_a_spool_run_puts_its_commands_in_the_spool(tmp_path, task_file, monkeypatch):
    """`--dry-run` stops once preflight has resolved the executor; the survey's probe is stubbed.

    It stops before the first agent turn and leaves the checkpoint at preflight, so the real
    run re-runs it.
    """
    spool = tmp_path / "spool"
    spool.mkdir()
    spec = yaml.safe_load(task_file.read_text())
    spec["environment"].update(kind="session_spool", spool=str(spool))
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
    result = workflow._shell_runner("echo gate-command", None, 0)
    assert result.timed_out
    scripts = sorted((spool / "pending").glob("*.sh"))
    assert scripts and "echo gate-command" in scripts[-1].read_text(encoding="utf-8")


def test_cpu_introspection_does_not_reserve_a_gpu(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment

    flow = SimpleNamespace(
        _shell_runner_override=None, task_data={"trtllm_repo_path": str(tmp_path)}
    )
    result = SimpleNamespace(
        environment=RunEnvironment("session_spool", "/p", spool=str(tmp_path / "s"))
    )
    runner = preflight_stage._cpu_probe_runner(flow, result)
    runner("true", None, 0)
    script = next((tmp_path / "s" / "pending").glob("*.sh")).read_text()
    assert "gpus=0" in script

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
    env = RunEnvironment(kind="session_spool", python="/p", world_size=4, spool="/s")
    command = env.command("/p x.py")
    for name in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID", "SLURM_NODEID"):
        assert f"-u {name}" in command, name
    assert command.index(MULTI_TASK_SLURM_SCRUB) < command.index("mpirun")
    single = env.command("/p probe.py", single_process=True)
    assert "${!PMIX@}" in single
    assert "exec /p probe.py" in single


def test_spool_exports_preserve_shell_quotes():
    import re
    import subprocess

    env = RunEnvironment(kind="session_spool", python="/p", spool="/s", env={"EXAMPLE": "a b'$c"})
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
    import subprocess

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import server_runtime

    def hung(argv, **kwargs):
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(argv, 30)

    monkeypatch.setattr(server_runtime.subprocess, "run", hung)
    assert server_runtime.device_memory_used_mib() is None
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
    lead_s: float = _execution_HOUR,
    age_s: float = 0.0,
    default_wall: str = "2h",
) -> Path:
    """A spool watched by one scheduler that has just ticked ``age_s`` ago."""
    root = parent / name
    now = time.time()
    config = SchedulerConfig(
        devices=_execution_DEVICES,
        session=name,
        ends_at=now + left_s,
        handover_lead_s=lead_s,
        default_wall_s=parse_duration(default_wall),
    )
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
    code = spoolctl.main(["--spool", str(work1), "launch", "--gpus", "1", "r1-x", "true"])
    assert code == spoolctl.EXIT_NO_SESSION
    assert pending(work1) == []
    assert "stale: the session is gone" in capsys.readouterr().err


@pytest.mark.usefixtures("_no_ambient_spool")
def test_a_launched_job_queues_at_the_priority_asked(tmp_path):
    work1 = session(tmp_path, "work-1")
    code = spoolctl.main(
        ["--spool", str(work1), "launch", "--gpus", "1", "--priority", "-2", "r1-x", "true"]
    )
    assert code == 0 and pending(work1) == ["r1-x"]
    header = (work1 / "pending" / "r1-x.sh").read_text().split("#spool:", 1)[1].splitlines()[0]
    assert "priority=-2" in header.split()


@pytest.mark.usefixtures("_no_ambient_spool")
def test_the_gate_pool_skips_a_session_in_hand_over_for_one_that_can_take_the_job(tmp_path):
    """The least loaded gate spool was the one about to end: the gate would park there."""
    ending = session(
        tmp_path,
        "gates-2",
        left_s=0.5 * _execution_HOUR,
        lead_s=1.5 * _execution_HOUR,
        default_wall="3h",
    )
    busy = session(tmp_path, "gates-1", default_wall="3h")
    (busy / "pending" / "other.sh").write_text("true\n")
    executor = SpoolCommandExecutor(ending, pool=[ending, busy])
    gate = placement.JobNeeds(gpus=None, wall_s=None, holds_gpus=True)
    assert executor._pick(gate) == busy


def test_gate_jobs_reach_any_sessions_but_not_work_ones():
    """An `any` session on another spool shares the gates; a `work` one keeps to its workers."""
    env = RunEnvironment(
        "session_spool",
        "/p",
        spool="/s/main",
        sessions=(
            sessions.SessionSpec("shared", "/s/shared"),
            sessions.SessionSpec("gates", "/s/gates", role="gates"),
            sessions.SessionSpec("work", "/s/work", role="work"),
        ),
    )
    executor = priority.select_command_executor(env)
    assert executor.pool == [Path("/s/main"), Path("/s/shared"), Path("/s/gates")]


def test_a_fresh_inode_leaves_a_mapped_library_untouched(tmp_path):
    lib = tmp_path / "libs" / "libth_common.so"
    lib.parent.mkdir()
    lib.write_bytes(b"old-library-bytes")
    os.chmod(lib, 0o755)
    held = lib.open("rb")  # a running process's mapping
    before = os.stat(lib).st_ino
    try:
        refreshed = rebuild.refresh_inodes(tmp_path)
        assert refreshed == [lib]
        assert os.stat(lib).st_ino != before
        assert lib.read_bytes() == b"old-library-bytes"
        assert os.stat(lib).st_mode & 0o777 == 0o755
        # The build then rewrites the file in place -- the new inode, not the held one.
        with lib.open("r+b") as handle:
            handle.write(b"NEW")
        held.seek(0)
        assert held.read() == b"old-library-bytes"
    finally:
        held.close()
    assert not list(lib.parent.glob(".*.tmp"))
