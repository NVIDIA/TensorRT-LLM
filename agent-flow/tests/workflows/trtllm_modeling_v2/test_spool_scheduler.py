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

from agent_flow.workflows.trtllm_modeling_v2.infra.execution import (
    CommandSpec,
    SpoolCommandExecutor,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import scheduler, spoolctl
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    RC_CANCELLED,
    RC_STALLED,
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
        kill_grace_s=overrides.pop("kill_grace_s", 0.5),
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
    sched = make(spool, time.time, poll_s=0.05)
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


def test_the_receipt_carries_the_exit_code(spool, clock):
    sched = make(spool, clock)
    submit(spool, "free-fail", "exit 7")
    pump(sched, lambda: rc(spool, "free-fail") is not None)
    assert rc(spool, "free-fail") == 7
    assert record(spool, "free-fail")["reason"] == "exit"


def test_the_watcher_removes_the_allocations_mpi_context(spool, clock):
    environ = {
        "PATH": os.environ["PATH"],
        "SLURM_JOB_ID": "500326",
        "SLURM_STEP_ID": "0",
        "PMIX_NAMESPACE": "slurm.pmix.500326.0",
        "PMI_FD": "7",
        "OMPI_COMM_WORLD_RANK": "0",
        "PRTE_X": "1",
        "MPI_LOCALRANKID": "0",
        "KEEP_ME": "yes",
    }
    sched = make(spool, clock, environ=environ)
    submit(spool, "slot", "env | sort", header="gpus=2")
    pump(sched, lambda: rc(spool, "slot") is not None)
    log = SpoolPaths(spool).log("slot").read_text()
    for gone in (
        "SLURM_STEP_ID",
        "PMIX_NAMESPACE",
        "PMI_FD",
        "OMPI_COMM",
        "PRTE_X",
        "MPI_LOCALRANKID",
    ):
        assert f"\n{gone}" not in log
    assert "\nKEEP_ME=yes" in log
    assert "\nCUDA_VISIBLE_DEVICES=0,1" in log
    assert "\nMV2FLOW_SPOOL_SESSION_JOBID=500326" in log
    assert "\nMV2FLOW_SPOOL_JOB=slot" in log


def test_slot_jobs_get_disjoint_devices(spool, clock):
    sched = make(spool, clock)
    for name, gpus in (("a", 1), ("b", 2), ("c", 1)):
        submit(
            spool,
            name,
            f"echo DEV=$CUDA_VISIBLE_DEVICES; {blocking(spool, name)}",
            header=f"gpus={gpus}",
        )
    pump(sched, lambda: running(sched) == {"a", "b", "c"})
    held = [set(job.devices) for job in sched.jobs.values()]
    assert sorted(len(h) for h in held) == [1, 1, 2]
    assert set.union(*held) == set(DEVICES)
    assert sum(len(h) for h in held) == 4
    drain(sched, spool)


def test_a_waiting_whole_node_job_preempts_the_admission_of_slot_jobs(spool, clock):
    sched = make(spool, clock)
    t0 = clock.now - 100
    submit(spool, "s1", blocking(spool, "s1"), header="gpus=1", submitted=t0)
    pump(sched, lambda: "s1" in running(sched))
    # A slot job submitted BEFORE the whole-node job, and one after.
    submit(spool, "s2", blocking(spool, "s2"), header="gpus=1", submitted=t0 + 1)
    submit(spool, "gate", blocking(spool, "gate"), header="gpus=all", submitted=t0 + 2)
    submit(spool, "s3", blocking(spool, "s3"), header="gpus=1", submitted=t0 + 3)
    for _ in range(5):
        sched.tick()
    assert running(sched) == {"s1"}, "three devices are free, but the whole-node job is waiting"
    reasons = {p["name"]: p["reason"] for p in sched.snapshot(clock.now)["pending"]}
    assert "drain" in reasons["gate"]
    assert "gate" in reasons["s2"] and "gate" in reasons["s3"]

    release(spool, "s1")
    pump(sched, lambda: "gate" in running(sched))
    assert running(sched) == {"gate"}
    assert sched.jobs["gate"].devices == DEVICES

    release(spool, "gate")
    pump(sched, lambda: running(sched) == {"s2", "s3"})
    drain(sched, spool)


def test_a_multi_device_job_is_not_starved_by_single_device_jobs(spool, clock):
    sched = make(spool, clock)
    t0 = clock.now - 100
    for i in range(3):
        submit(spool, f"one{i}", blocking(spool, f"one{i}"), header="gpus=1", submitted=t0 + i)
    pump(sched, lambda: len(running(sched)) == 3)
    submit(spool, "four", blocking(spool, "four"), header="gpus=4", submitted=t0 + 10)
    submit(spool, "late", blocking(spool, "late"), header="gpus=1", submitted=t0 + 11)
    for _ in range(3):
        sched.tick()
    assert "late" not in running(sched), "the free device is held for the job ahead of it"
    for i in range(3):
        release(spool, f"one{i}")
    pump(sched, lambda: "four" in running(sched))
    assert "late" not in running(sched)
    drain(sched, spool)


def test_device_free_jobs_start_even_while_gpu_jobs_wait(spool, clock):
    sched = make(spool, clock)
    submit(spool, "g", blocking(spool, "g"), header="gpus=all")
    pump(sched, lambda: "g" in running(sched))
    submit(spool, "w", blocking(spool, "w"), header="gpus=all")
    submit(spool, "free-orch", blocking(spool, "free-orch"))
    pump(sched, lambda: "free-orch" in running(sched))
    assert "w" not in running(sched)
    drain(sched, spool)


def test_the_wall_limit_counts_from_start_not_from_submission(spool, clock):
    sched = make(spool, clock, devices=("0",))
    submit(spool, "first", blocking(spool, "first"), header="gpus=1")
    pump(sched, lambda: "first" in running(sched))
    submit(spool, "second", blocking(spool, "second"), header="gpus=1 wall=60")
    clock.advance(600)  # queued for ten minutes, ten times its wall
    sched.tick()
    release(spool, "first")
    pump(sched, lambda: "second" in running(sched))
    sched.tick()
    assert rc(spool, "second") is None, "a job must not be timed out by its queueing time"

    clock.advance(61)
    pump(sched, lambda: rc(spool, "second") is not None)
    assert rc(spool, "second") == RC_WALL_LIMIT
    data = record(spool, "second")
    assert data["reason"] == "wall_limit" and data["timed_out"] is True
    assert "from start" in data["detail"]
    pump(sched, lambda: not sched.jobs)


def test_cancel_is_the_sanctioned_kill(spool, clock):
    sched = make(spool, clock)
    submit(spool, "long", "sleep 300", header="gpus=1")
    pump(sched, lambda: "long" in running(sched))
    assert spoolctl.main(["--spool", str(spool), "cancel", "long"]) == 0
    pump(sched, lambda: rc(spool, "long") is not None)
    assert rc(spool, "long") == RC_CANCELLED
    pump(sched, lambda: not sched.jobs)
    assert not (SpoolPaths(spool).cancel / "long").exists()


def test_processes_a_finished_job_leaves_behind_are_reaped(spool, clock):
    sched = make(spool, clock)
    submit(spool, "leaky", "sleep 300 & echo started", header="gpus=2")
    pump(sched, lambda: rc(spool, "leaky") is not None)
    assert rc(spool, "leaky") == 0
    pump(sched, lambda: not sched.jobs)
    assert "left behind" in SpoolPaths(spool).log("leaky").read_text()


def test_a_namesake_job_of_another_spool_is_not_reaped(tmp_path, clock):
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    mine, other = make(ours, clock, watcher_id="w"), make(theirs, clock, watcher_id="w")
    submit(theirs, "free-same", blocking(theirs, "free-same"))
    pump(other, lambda: "free-same" in running(other))
    submit(ours, "free-same", "true")
    pump(mine, lambda: not mine.jobs)
    assert other.jobs["free-same"].proc.poll() is None, "another spool's job was signalled"
    drain(other, theirs)


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


def test_a_live_peer_watchers_job_is_left_alone(spool, clock):
    paths = SpoolPaths(spool)
    (paths.running / "theirs.sh").write_text("sleep 1\n")
    scheduler.write_json_atomic(
        paths.running / "theirs.json", {"name": "theirs", "watcher": "peer-2"}
    )
    scheduler.write_json_atomic(
        paths.watchers / "peer-2.json", {"watcher": "peer-2", "alive_epoch": clock.now}
    )
    make(spool, clock).tick()
    assert rc(spool, "theirs") is None


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


RANK0_FAILS = "echo 'Traceback (most recent call last):'; echo 'RuntimeError: rank 0 init'; "


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
    clock.advance(301)
    sched.tick()
    snapshot = sched.snapshot(clock.now)
    assert snapshot["quarantined"] == ["0"] and snapshot["free"] == []
    assert snapshot["quarantined_memory"]["0"]["used_mib"] == 40211
    assert reason(sched, clock, "boot") == (
        "waiting for 1 GPU(s); 0 free; quarantined, memory not released: 0 (40211 MiB)"
    )
    assert "watcher: device 0 quarantined" in SpoolPaths(spool).status.read_text()
    assert sched.counters["device_quarantined"] == 1
    # Placed elsewhere: the device the holder gives back, not the quarantined one.
    release(spool, "holder")
    pump(sched, lambda: rc(spool, "boot") is not None)
    assert record(spool, "boot")["devices"] == ["1"]
    assert sched.snapshot(clock.now)["quarantined"] == ["0"]


def test_a_wall_limit_is_a_timeout_not_a_lost_session(spool, served):
    result = SpoolCommandExecutor(spool, poll_s=0.05).execute(
        CommandSpec(argv="sleep 30", timeout_s=1)
    )
    assert result.timed_out and result.returncode == RC_WALL_LIMIT
    assert result.infrastructure == ""
