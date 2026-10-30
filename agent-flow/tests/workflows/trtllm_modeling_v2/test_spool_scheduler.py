# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The spool scheduler, driven with real ``bash`` jobs in a temporary spool and a fake clock."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import scheduler, spoolctl
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandSpec
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import (
    CANCELLED,
    SpoolCommandExecutor,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
    DEVICE_RELEASE_WAIT_S,
    RC_CANCELLED,
    RC_WALL_LIMIT,
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
)

DEVICES = ("0", "1", "2", "3")


class FakeClock:
    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    return root


def make(spool, clock, **overrides) -> Scheduler:
    environ = overrides.pop("environ", None)
    watcher_id = overrides.pop("watcher_id", None)
    config = SchedulerConfig(
        devices=overrides.pop("devices", DEVICES),
        leftover_grace_s=overrides.pop("leftover_grace_s", 0.0),
        **overrides,
    )
    if environ is None:
        environ = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    return Scheduler(spool, config, clock=clock, environ=environ, watcher_id=watcher_id)


def submit(spool, name, body="true", *, header="", submitted=None) -> Path:
    paths = SpoolPaths(spool)
    text = "#!/bin/bash\nset -u\n" + (f"#spool: {header}\n" if header else "") + body + "\n"
    target = paths.pending / f"{name}.sh"
    staging = paths.pending / f".{name}.tmp"
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, target)
    if submitted is not None:
        os.utime(target, (submitted, submitted))
    return target


def blocking(spool, name) -> str:
    """A body that runs until `release(spool, name)`."""
    flag = Path(spool) / f"release-{name}"
    return f'while [ ! -f "{flag}" ]; do sleep 0.02; done'


def release(spool, name) -> None:
    (Path(spool) / f"release-{name}").write_text("", encoding="utf-8")


def pump(sched: Scheduler, until, timeout: float = 15.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        sched.tick()
        if until():
            return
        time.sleep(0.02)
    raise AssertionError(
        f"condition not reached; status: {json.dumps(sched.snapshot(0), indent=1)}"
    )


def rc(spool, name) -> int | None:
    path = SpoolPaths(spool).receipt(name)
    return int(path.read_text().strip()) if path.exists() else None


def record(spool, name) -> dict:
    return json.loads(SpoolPaths(spool).record(name).read_text())


def running(sched) -> set[str]:
    return {name for name, job in sched.jobs.items() if job.verdict is None}


def drain(sched, spool) -> None:
    """Release every blocking job, queued or running, and let the scheduler close them out."""

    def released_and_idle() -> bool:
        names = set(sched.jobs) | {p.stem for p in SpoolPaths(spool).pending.glob("*.sh")}
        for name in names:
            release(spool, name)
        return not names

    pump(sched, released_and_idle)


@pytest.fixture
def served(spool):
    """A scheduler serving the spool in a background thread, on the real clock."""
    sched = make(spool, time.time)
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


def test_an_mpi_attached_process_is_never_signalled(spool, clock):
    sched = make(spool, clock, devices=("0", "1"))
    submit(spool, "attached", "export SLURM_JOB_ID=1; exec sleep 300", header="gpus=1 wall=30")
    pump(sched, lambda: "attached" in running(sched))
    pid = sched.jobs["attached"].proc.pid
    pump(sched, lambda: "SLURM_JOB_ID" in (scheduler.read_environ(pid) or {}))
    clock.advance(31)
    pump(sched, lambda: rc(spool, "attached") is not None)
    try:
        assert rc(spool, "attached") == RC_WALL_LIMIT
        assert record(spool, "attached")["unsafe_pids"] == [pid]
        assert os.path.exists(f"/proc/{pid}"), "the attached process must not have been signalled"
        snapshot = sched.snapshot(clock.now)
        assert snapshot["quarantined"] == [sched.jobs["attached"].devices[0]]
        submit(spool, "next", blocking(spool, "next"), header="gpus=2")
        for _ in range(3):
            sched.tick()
        assert "next" not in running(sched), "quarantined devices are not handed out"
    finally:
        os.kill(pid, signal.SIGKILL)
    pump(sched, lambda: "attached" not in sched.jobs)
    pump(sched, lambda: "next" in running(sched))
    drain(sched, spool)


def test_two_watchers_on_one_spool_run_a_job_once(spool, clock):
    a = make(spool, clock, watcher_id="a")
    b = make(spool, clock, watcher_id="b")
    submit(spool, "free-once", "echo ran >> " + str(Path(spool) / "count"))
    for _ in range(50):
        a.tick()
        b.tick()
        if rc(spool, "free-once") is not None:
            break
        time.sleep(0.02)
    pump(a, lambda: not a.jobs)
    pump(b, lambda: not b.jobs)
    assert (Path(spool) / "count").read_text() == "ran\n"


def reason(sched, clock, name: str) -> str:
    return next(p["reason"] for p in sched.snapshot(clock.now)["pending"] if p["name"] == name)


def test_a_device_that_never_releases_its_memory_is_quarantined(spool, clock):
    sched = make(spool, clock, devices=("0", "1"))
    sched.device_memory_mib = lambda: {"0": 40211, "1": 0}
    submit(spool, "holder", blocking(spool, "holder"), header="gpus=1")
    pump(sched, lambda: "holder" in running(sched))
    assert sched.jobs["holder"].devices == ("1",)
    submit(spool, "boot", "echo started", header="gpus=1")
    sched.tick()
    assert reason(sched, clock, "boot").startswith("device memory in use: 0 (40211 MiB)")
    clock.advance(DEVICE_RELEASE_WAIT_S + 1)
    sched.tick()
    snapshot = sched.snapshot(clock.now)
    assert snapshot["quarantined"] == ["0"] and snapshot["free"] == []
    assert snapshot["quarantined_memory"]["0"]["used_mib"] == 40211
    assert reason(sched, clock, "boot") == (
        "waiting for 1 GPU(s); 0 free; quarantined, memory not released: 0 (40211 MiB)"
    )
    assert "watcher: device 0 quarantined" in SpoolPaths(spool).status.read_text()
    # Placed elsewhere: the device the holder gives back, not the quarantined one.
    release(spool, "holder")
    pump(sched, lambda: rc(spool, "boot") is not None)
    assert record(spool, "boot")["devices"] == ["1"]
    assert sched.snapshot(clock.now)["quarantined"] == ["0"]


def test_a_campaigns_job_never_ages_ahead_of_the_onboardings_nor_holds_one_back(spool, clock):
    """Background work (``priority=-1``) waits behind every other job, however long it waited."""
    sched = make(spool, clock)
    submit(spool, "holder", blocking(spool, "holder"), header="gpus=1 wall=8h")
    pump(sched, lambda: "holder" in running(sched))
    # The campaign's whole-node measurement cannot place beside the holder, and waits.
    submit(spool, "tune-measure", "echo measured", header="gpus=4 priority=-1")
    sched.tick()
    clock.advance(3 * scheduler.AGING_S)
    # A check queued hours later still ranks first, and starts on the free devices.
    submit(spool, "boot", "echo started", header="gpus=1 priority=1", submitted=clock.now)
    pump(sched, lambda: rc(spool, "boot") is not None)
    assert rc(spool, "boot") == 0 and rc(spool, "tune-measure") is None
    pending = {p["name"]: p for p in sched.snapshot(clock.now)["pending"]}
    assert pending["tune-measure"]["effective_priority"] == -1
    release(spool, "holder")
    pump(sched, lambda: rc(spool, "tune-measure") is not None)


def test_a_cancelled_job_measured_nothing_and_says_so(spool, served):
    """Cancelled through the spool (by the operator, say): no verdict, and no lost session."""
    results = []
    executor = SpoolCommandExecutor(spool, poll_s=0.05)
    thread = threading.Thread(
        target=lambda: results.append(executor.execute(CommandSpec(argv="sleep 30", timeout_s=60)))
    )
    thread.start()
    running = SpoolPaths(spool).running
    deadline = time.time() + 10
    while not list(running.glob("*.sh")) and time.time() < deadline:
        time.sleep(0.05)
    (job,) = running.glob("*.sh")
    spoolctl.cancel_job(spool, job.stem)
    thread.join(timeout=20)
    (result,) = results
    assert result.returncode == RC_CANCELLED and not result.timed_out
    assert result.infrastructure.startswith(CANCELLED)
