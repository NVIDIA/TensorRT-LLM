# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No hang that needs a human: deadlines from start, no-progress detection, memory, watchdog.

The cases are the ones that would otherwise need a human: a rank-0 init
failure that leaves the other ranks spinning until somebody sends SIGTERM; a
gate timeout that counts the time a job spent queued; an engine started while
the previous one's host memory is still being released; a session that dies
and is noticed hours later.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.execution import (
    CommandSpec,
    SpoolCommandExecutor,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import (
    SUBSTRATE_DIR,
    scheduler,
    watchdog,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    RC_HOST_MEMORY,
    RC_ORPHANED,
    RC_STALLED,
    RC_WALL_LIMIT,
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
    write_json_atomic,
)

from .test_spool_scheduler import FakeClock, blocking, pump, rc, record, release, running, submit


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    return root


def make(spool, clock, **config) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(config.pop("environ", {}))
    config.setdefault("devices", ("0", "1"))
    config.setdefault("kill_grace_s", 0.5)
    config.setdefault("leftover_grace_s", 0.0)
    return Scheduler(spool, SchedulerConfig(**config), clock=clock, environ=env)


RANK0_FAILS = "echo 'Traceback (most recent call last):'; echo 'RuntimeError: rank 0 init'; "


# ------------------------------------------------------------- no progress


def test_a_rank_failure_followed_by_silence_is_marked_failed(spool, clock):
    """Rank 0 died at init; its peers wait in a collective forever and print nothing."""
    sched = make(spool, clock)
    submit(spool, "hung", RANK0_FAILS + "sleep 300", header="gpus=2 failure_stall=60")
    pump(sched, lambda: "hung" in sched.jobs and sched.jobs["hung"].failure_line)
    sched.tick()
    assert rc(spool, "hung") is None, "a failure line alone is not a hang"
    clock.advance(61)
    pump(sched, lambda: rc(spool, "hung") is not None)
    data = record(spool, "hung")
    assert data["rc"] == RC_STALLED and data["reason"] == "stalled" and data["timed_out"]
    assert "Traceback" in data["detail"] and "no progress for" in data["detail"]
    pump(sched, lambda: not sched.jobs)


def test_output_that_keeps_coming_after_a_failure_line_is_progress(spool, clock):
    """Pytest prints a failed test's traceback and carries on with the next one."""
    sched = make(spool, clock)
    flag = Path(spool) / "release-busy"
    body = RANK0_FAILS + f'while [ ! -f "{flag}" ]; do echo still-testing; sleep 0.05; done'
    submit(spool, "busy", body, header="gpus=1 failure_stall=60")
    pump(sched, lambda: "busy" in sched.jobs and sched.jobs["busy"].failure_line)
    for _ in range(5):
        clock.advance(30)
        time.sleep(0.15)
        sched.tick()
    assert rc(spool, "busy") is None
    release(spool, "busy")
    pump(sched, lambda: rc(spool, "busy") is not None)
    assert record(spool, "busy")["reason"] == "exit"


def test_plain_silence_is_a_hang_only_when_the_job_says_so(spool, clock):
    sched = make(spool, clock)
    submit(spool, "quiet", "sleep 300", header="gpus=1")
    submit(spool, "quiet-bounded", "sleep 300", header="gpus=1 stall=60")
    pump(sched, lambda: {"quiet", "quiet-bounded"} <= running(sched))
    clock.advance(61)
    pump(sched, lambda: rc(spool, "quiet-bounded") is not None)
    assert record(spool, "quiet-bounded")["reason"] == "stalled"
    assert "quiet" in running(sched)
    (SpoolPaths(spool).cancel / "quiet").write_text("")
    pump(sched, lambda: not sched.jobs)


def test_a_hung_mpi_attached_process_is_marked_failed_but_never_killed(spool, clock):
    """The safe-kill rule holds for no-progress verdicts too."""
    sched = make(spool, clock)
    submit(
        spool,
        "attached",
        RANK0_FAILS + "export SLURM_JOB_ID=1; exec sleep 300",
        header="gpus=1 failure_stall=60",
    )
    pump(sched, lambda: "attached" in sched.jobs and sched.jobs["attached"].failure_line)
    pid = sched.jobs["attached"].proc.pid
    pump(sched, lambda: "SLURM_JOB_ID" in (scheduler.read_environ(pid) or {}))
    clock.advance(61)
    pump(sched, lambda: rc(spool, "attached") is not None)
    try:
        assert record(spool, "attached")["reason"] == "stalled"
        assert os.path.exists(f"/proc/{pid}")
        assert sched.snapshot(clock.now)["quarantined"]
    finally:
        os.kill(pid, signal.SIGKILL)
    pump(sched, lambda: not sched.jobs)


def test_gpu_jobs_get_a_failure_stall_by_default_and_free_jobs_do_not(spool, clock):
    sched = make(spool, clock, failure_stall_s=600.0)
    gpu = sched._classify(scheduler.job_spec_from_script("g", "#spool: gpus=1\nx"))
    free = sched._classify(scheduler.job_spec_from_script("free-orch", "x"))
    assert gpu.failure_stall_s == 600 and free.failure_stall_s == 0


# ------------------------------------------------------------- host memory


def test_an_engine_waits_for_host_memory_before_it_starts(spool, clock, monkeypatch):
    """VMM-backed host tables come back 1-6 minutes after the previous engine exits."""
    sched = make(spool, clock, engine_host_mem_gb=100.0)
    available = {"gb": 40.0}
    monkeypatch.setattr(sched, "available_memory_gb", lambda: available["gb"])
    submit(spool, "engine", "echo started", header="gpus=all")
    sched.tick()
    assert "engine" not in sched.jobs
    (pending,) = sched.snapshot(clock.now)["pending"]
    assert "host memory 40 GiB available, 100 needed" in pending["reason"]
    available["gb"] = 300.0
    pump(sched, lambda: rc(spool, "engine") is not None)
    assert rc(spool, "engine") == 0


def test_memory_that_never_comes_back_refuses_the_job_with_the_reason(spool, clock, monkeypatch):
    sched = make(spool, clock, host_mem_wait_s=300.0)
    monkeypatch.setattr(sched, "available_memory_gb", lambda: 10.0)
    submit(spool, "engine", "echo never", header="gpus=all host_mem_gb=200")
    sched.tick()
    clock.advance(301)
    sched.tick()
    assert rc(spool, "engine") == RC_HOST_MEMORY
    assert "host memory 10 GiB available, 200 needed" in record(spool, "engine")["detail"]


def test_time_queued_for_devices_is_not_charged_to_the_memory_wait(spool, clock, monkeypatch):
    sched = make(spool, clock, host_mem_wait_s=300.0)
    memory = {"gb": 10.0}
    monkeypatch.setattr(sched, "available_memory_gb", lambda: memory["gb"])
    submit(spool, "first", blocking(spool, "first"), header="gpus=all")
    pump(sched, lambda: "first" in running(sched))
    submit(spool, "second", "echo second", header="gpus=all host_mem_gb=50")
    clock.advance(3600)  # an hour queued behind the first engine
    sched.tick()
    release(spool, "first")
    pump(sched, lambda: "first" not in sched.jobs)
    sched.tick()
    assert rc(spool, "second") is None, "its memory wait starts when its devices are free"
    memory["gb"] = 100.0
    pump(sched, lambda: rc(spool, "second") is not None)
    assert rc(spool, "second") == 0


def test_one_pass_does_not_hand_the_same_memory_to_two_jobs(spool, clock, monkeypatch):
    sched = make(spool, clock)
    monkeypatch.setattr(sched, "available_memory_gb", lambda: 100.0)
    submit(spool, "a", blocking(spool, "a"), header="gpus=1 host_mem_gb=60")
    submit(spool, "b", blocking(spool, "b"), header="gpus=1 host_mem_gb=60")
    sched.tick()
    assert len(running(sched)) == 1
    for name in ("a", "b"):
        release(spool, name)
    pump(sched, lambda: not sched.jobs and not list(SpoolPaths(spool).pending.glob("*.sh")))


def test_a_job_can_wait_for_dev_shm(spool, clock, monkeypatch):
    sched = make(spool, clock)
    monkeypatch.setattr(scheduler, "shm_available_gb", lambda path="/dev/shm": 5.0)
    submit(spool, "engram", "true", header="gpus=1 shm_gb=203")
    sched.tick()
    (pending,) = sched.snapshot(clock.now)["pending"]
    assert "/dev/shm 5 GiB free, 203 needed" in pending["reason"]


# ---------------------------------------------- the orchestrator's deadline


@pytest.fixture
def served(spool):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    sched = Scheduler(
        spool, SchedulerConfig(devices=("0",), kill_grace_s=0.3, leftover_grace_s=0), environ=env
    )
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


def test_a_gates_deadline_counts_from_its_start_not_its_queueing(spool, served):
    """Queued time is not the gate's: charging it would fail the gate before it ran."""
    submit(spool, "earlier", blocking(spool, "earlier"), header="gpus=all")
    pump_until = time.monotonic() + 10
    while "earlier" not in served.jobs and time.monotonic() < pump_until:
        time.sleep(0.05)
    threading.Timer(2.0, release, args=(spool, "earlier")).start()
    executor = SpoolCommandExecutor(spool, poll_s=0.05)
    started = time.monotonic()
    result = executor.execute(CommandSpec(argv="echo ran", timeout_s=1))
    assert result.ok, result.stdout
    assert time.monotonic() - started > 1.5, "it waited longer than its own timeout in the queue"


def test_a_gate_past_its_deadline_is_stopped_by_the_session_and_reads_as_timed_out(spool, served):
    executor = SpoolCommandExecutor(spool, poll_s=0.05)
    result = executor.execute(CommandSpec(argv="sleep 30", timeout_s=1))
    assert result.timed_out
    assert result.returncode == RC_WALL_LIMIT
    assert "wall limit of 1s" in result.stdout


def test_a_dead_session_fails_a_queued_command_instead_of_hanging(spool):
    paths = SpoolPaths(spool)
    write_json_atomic(
        paths.watchers / "dead.json", {"watcher": "dead", "alive_epoch": time.time() - 10}
    )
    executor = SpoolCommandExecutor(spool, poll_s=0.05, stale_s=1.0)
    started = time.monotonic()
    result = executor.execute(CommandSpec(argv="true", timeout_s=3600))
    assert result.timed_out and time.monotonic() - started < 30
    assert "stopped heart-beating" in result.stdout


def _heartbeat(spool, age: float) -> None:
    """A watcher on ``spool`` that last heart-beat ``age`` seconds ago."""
    write_json_atomic(
        SpoolPaths(spool).watchers / "w.json", {"watcher": "w", "alive_epoch": time.time() - age}
    )


def test_a_dead_session_gets_nothing_queued(spool):
    """A script queued on a dead session runs whenever a successor drains the spool."""
    _heartbeat(spool, age=600)
    started = time.monotonic()
    result = SpoolCommandExecutor(spool, poll_s=0.05).execute(
        CommandSpec(argv="true", timeout_s=3600)
    )
    assert result.timed_out and time.monotonic() - started < 5
    assert "stopped heart-beating 600s ago" in result.infrastructure
    assert "nothing was queued" in result.infrastructure
    assert not list(SpoolPaths(spool).pending.glob("*.sh"))


def test_a_command_its_session_died_under_is_withdrawn_before_a_successor_runs_it(spool, clock):
    """Otherwise a requeued session that comes back runs a gate nobody is waiting for."""
    paths = SpoolPaths(spool)
    _heartbeat(spool, age=0)
    ran = Path(spool) / "ran"
    result = SpoolCommandExecutor(spool, poll_s=0.05, stale_s=0.5).execute(
        CommandSpec(argv=f"touch {ran}", timeout_s=3600)
    )
    assert result.timed_out and "stopped heart-beating" in result.infrastructure
    (script,) = paths.pending.glob("*.sh")
    assert (paths.cancel / script.stem).exists() and "withdrawn" in result.infrastructure

    # The allocation comes back on the same spool.
    successor = make(spool, clock)
    pump(successor, lambda: rc(spool, script.stem) is not None)
    assert record(spool, script.stem)["reason"] == "cancelled"
    assert not ran.exists()


def test_a_running_command_given_up_on_is_stopped_once_its_session_answers(spool):
    """Withdrawing a job that already started goes through the safe-kill rule."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    sched = Scheduler(
        spool, SchedulerConfig(devices=("0",), kill_grace_s=0.3, leftover_grace_s=0), environ=env
    )
    answering, stop = threading.Event(), threading.Event()
    answering.set()

    def loop():
        while not stop.is_set():
            if answering.is_set():
                sched.tick()
            time.sleep(0.05)

    def stall_once_running():
        while not sched.jobs:
            time.sleep(0.02)
        answering.clear()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    threading.Thread(target=stall_once_running, daemon=True).start()
    try:
        result = SpoolCommandExecutor(spool, poll_s=0.05, stale_s=1.0).execute(
            CommandSpec(argv="sleep 30", timeout_s=3600)
        )
        assert "stopped heart-beating" in result.infrastructure
        (name,) = list(sched.jobs)
        answering.set()
        end = time.monotonic() + 15
        while rc(spool, name) is None and time.monotonic() < end:
            time.sleep(0.05)
        assert record(spool, name)["reason"] == "cancelled"
    finally:
        stop.set()
        thread.join(timeout=5)


def _claim_for_a_dead_watcher(spool, clock, then) -> None:
    """Once the executor queues its job, hand it to watcher ``dead-1``, gone an hour, then ``then(name)``."""
    paths = SpoolPaths(spool)

    def claimed_then_lost():
        end = time.monotonic() + 10
        while time.monotonic() < end:
            for script in paths.pending.glob("*.sh"):
                token = re.search(r"token=(\w+)", script.read_text()).group(1)
                write_json_atomic(
                    paths.watchers / "dead-1.json",
                    {"watcher": "dead-1", "alive_epoch": clock.now - 3600},
                )
                write_json_atomic(
                    paths.running / f"{script.stem}.json",
                    {"name": script.stem, "watcher": "dead-1", "token": token, "started": "x"},
                )
                os.replace(script, paths.running / script.name)
                then(script.stem)
                return
            time.sleep(0.02)

    threading.Thread(target=claimed_then_lost, daemon=True).start()


def test_a_job_whose_processes_outlived_their_session_reads_as_a_lost_session(spool, clock):
    """Its watcher died but the job still runs here: a successor closes it out as orphaned."""
    _heartbeat(spool, age=0)
    successor = make(spool, clock)
    strays = []

    def outlived(name):
        marker = {"MV2FLOW_SPOOL": str(spool), "MV2FLOW_SPOOL_JOB": name}
        env = {**os.environ, **marker, "MV2FLOW_SPOOL_WATCHER": "dead-1"}
        strays.append(subprocess.Popen(["sleep", "60"], env=env))
        successor.tick()

    _claim_for_a_dead_watcher(spool, clock, outlived)
    try:
        result = SpoolCommandExecutor(spool, poll_s=0.05).execute(
            CommandSpec(argv="true", timeout_s=60)
        )
    finally:
        for stray in strays:
            stray.kill()
            stray.wait()
    assert result.returncode == RC_ORPHANED and result.timed_out
    assert "orphaned" in result.infrastructure and "dead-1" in result.infrastructure


def test_a_job_its_session_died_under_runs_again_and_the_gate_gets_that_run(spool, clock):
    """The allocation took the job with it: a successor runs it again, and the gate waits for it."""
    _heartbeat(spool, age=0)
    successor = make(spool, clock)
    names = []

    def run_again(name):
        names.append(name)
        pump(successor, lambda: rc(spool, name) is not None)

    _claim_for_a_dead_watcher(spool, clock, run_again)
    result = SpoolCommandExecutor(spool, poll_s=0.05).execute(
        CommandSpec(argv="echo ran-again", timeout_s=60)
    )
    assert result.ok and not result.infrastructure, result.stdout
    assert "ran-again" in result.stdout
    lost, rerun = record(spool, names[0])["attempts"]
    assert lost["watcher"] == "dead-1" and rerun["watcher"] == successor.watcher_id


def test_a_spool_nobody_ever_watched_is_a_lost_session_and_is_withdrawn(spool):
    result = SpoolCommandExecutor(spool, poll_s=0.05).execute(CommandSpec(argv="true", timeout_s=0))
    assert "no session watches" in result.infrastructure
    (script,) = SpoolPaths(spool).pending.glob("*.sh")
    assert (SpoolPaths(spool).cancel / script.stem).exists()


def test_a_wall_limit_is_a_timeout_not_a_lost_session(spool, served):
    result = SpoolCommandExecutor(spool, poll_s=0.05).execute(
        CommandSpec(argv="sleep 30", timeout_s=1)
    )
    assert result.timed_out and result.returncode == RC_WALL_LIMIT
    assert result.infrastructure == ""


def test_waiting_for_a_session_returns_when_one_heart_beats_again(spool):
    _heartbeat(spool, age=600)
    executor = SpoolCommandExecutor(spool, poll_s=0.02)
    assert not executor.session_alive()
    threading.Timer(0.2, _heartbeat, args=(spool, 0)).start()
    started = time.monotonic()
    assert executor.wait_for_session(30)
    assert executor.session_alive() and time.monotonic() - started < 10


def test_waiting_for_a_session_that_does_not_come_back_is_bounded(spool):
    _heartbeat(spool, age=600)
    started = time.monotonic()
    assert not SpoolCommandExecutor(spool, poll_s=0.02).wait_for_session(0.3)
    assert time.monotonic() - started < 5


# ------------------------------------------------------------------ watchdog


def _watcher(spool, **fields):
    paths = SpoolPaths(spool)
    paths.ensure()
    write_json_atomic(paths.watchers / "w.json", {"watcher": "w", "slurm_job": "9", **fields})


def test_the_watchdog_names_a_dead_session(tmp_path):
    _watcher(tmp_path / "a", alive_epoch=time.time() - 600)
    (alert,) = watchdog.check([tmp_path / "a"])
    assert alert.kind == "STALE" and "600s" in alert.detail or "60" in alert.detail


def test_the_watchdog_warns_before_an_unplanned_end(tmp_path):
    _watcher(tmp_path / "a", alive_epoch=time.time(), ends_at_epoch=time.time() + 600)
    assert [a.kind for a in watchdog.check([tmp_path / "a"])] == ["ENDING"]
    (tmp_path / "a" / "handover").mkdir()
    write_json_atomic(tmp_path / "a" / "handover" / "request.json", {"request_id": "r1"})
    write_json_atomic(tmp_path / "a" / "handover" / "ack.json", {"request_id": "r1", "job_id": "5"})
    assert watchdog.check([tmp_path / "a"]) == []


def test_the_watchdog_reports_quarantined_devices(tmp_path):
    _watcher(
        tmp_path / "a",
        alive_epoch=time.time(),
        quarantined=["2"],
        running=[{"name": "stuck", "unsafe_pids": [123]}],
    )
    (alert,) = watchdog.check([tmp_path / "a"])
    assert alert.kind == "QUARANTINED" and "stuck" in alert.detail


def test_the_watchdog_skips_retired_and_never_started_spools(tmp_path):
    _watcher(tmp_path / "old", alive_epoch=0)
    (tmp_path / "old" / ".retired").write_text("")
    (tmp_path / "new").mkdir()
    assert watchdog.check([tmp_path / "old", tmp_path / "new"]) == []


def test_the_watchdog_script_returns_ok_on_timeout(tmp_path):
    _watcher(tmp_path / "a", alive_epoch=time.time() + 3600)
    done = subprocess.run(
        [
            sys.executable,
            str(SUBSTRATE_DIR / "watchdog.py"),
            "--spool",
            str(tmp_path / "a"),
            "--timeout",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0 and done.stdout.startswith("OK")


def test_the_watchdog_returns_on_the_first_alert(tmp_path):
    _watcher(tmp_path / "a", alive_epoch=time.time())
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)
        _watcher(tmp_path / "a", alive_epoch=time.time() - 1000)

    alerts, lines = watchdog.watch([tmp_path / "a"], timeout_s=3600, interval_s=1, sleep=fake_sleep)
    assert [a.kind for a in alerts] == ["STALE"] and len(calls) == 1
    assert lines and str(tmp_path / "a") in lines[0]


# ------------------------------------------------- processes that setsid away


def test_a_process_that_escaped_the_jobs_session_is_reaped(spool, clock, tmp_path):
    """Nsys profile setsid()s itself and the server it launches.

    Once the job's leader exits they are re-parented to init; left alone they hold the GPU
    for every later job.
    """
    pidfile = tmp_path / "escaped.pid"
    sched = make(spool, clock)
    submit(
        spool,
        "escapes",
        f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 300' & sleep 0.5; exit 0",
        header="gpus=1",
    )
    pump(sched, lambda: pidfile.exists() and pidfile.read_text().strip())
    escaped = int(pidfile.read_text().strip())
    pump(sched, lambda: rc(spool, "escapes") is not None)
    clock.advance(1)
    pump(sched, lambda: not sched.jobs)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(escaped, 0)
        except ProcessLookupError:
            break
        with open(f"/proc/{escaped}/stat") as fh:
            if fh.read().split()[2] == "Z":
                break
        time.sleep(0.1)
    else:
        os.kill(escaped, 9)
        raise AssertionError("the escaped process outlived its job")


def test_the_watchdog_says_a_device_was_quarantined_for_its_memory(tmp_path):
    _watcher(
        tmp_path / "a",
        alive_epoch=time.time(),
        quarantined=["1", "2"],
        quarantined_memory={"2": {"used_mib": 40211, "since": "2026-10-17T22:12:00Z"}},
        running=[{"name": "stuck", "unsafe_pids": [123]}],
    )
    held, memory = watchdog.check([tmp_path / "a"])
    assert held.kind == memory.kind == "QUARANTINED"
    assert held.detail.startswith("devices 1 held") and "stuck" in held.detail
    assert "devices 2 quarantined: memory not released" in memory.detail
    assert "40211 MiB in use since 2026-10-17T22:12:00Z" in memory.detail
