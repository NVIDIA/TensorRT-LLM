# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The spool scheduler, driven with fake jobs in a temporary spool.

Jobs are real ``bash`` processes that block on a release file, so admission
order, device allocation, limits and the safe-kill rule are observed on real
processes; the scheduler's clock is fake, so wall limits and aging are tested
without waiting for them.
"""

from __future__ import annotations

import json
import os
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
    SCRIPTS_DIR,
    SPOOLCTL_PATH,
    scheduler,
    spoolctl,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    RC_CANCELLED,
    RC_MEMORY_LIMIT,
    RC_REFUSED,
    RC_WALL_LIMIT,
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
    job_spec_from_script,
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


def submit(spool, name, body="true", *, header="", submitted=None, queued=None) -> Path:
    paths = SpoolPaths(spool)
    text = "#!/bin/bash\nset -u\n" + (f"#spool: {header}\n" if header else "") + body + "\n"
    if queued is not None:
        text = scheduler.stamp_queued(text, now=queued)
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


def resident(mib: int) -> str:
    """A body that touches ``mib`` MiB, so it is resident rather than merely reserved."""
    code = f"import time; b = bytearray({mib} << 20); b[::4096] = bytes(len(b[::4096])); time.sleep(60)"
    return f"python3 -c '{code}'"


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


# ------------------------------------------------------------- classification


def test_without_a_header_the_flock_era_naming_still_decides():
    assert job_spec_from_script("free-orch", "echo").gpus == 0
    assert job_spec_from_script("gate-1", "echo").whole
    assert job_spec_from_script("free-t1", "bash /x/gpu_slot.sh 2 pytest").gpus == 2


def test_a_header_overrides_the_name():
    spec = job_spec_from_script("free-x", "#spool: gpus=3 wall=90m priority=2 owner=r014\nrun")
    assert (spec.gpus, spec.wall_s, spec.priority, spec.owner) == (3, 5400, 2, "r014")
    assert not spec.errors


@pytest.mark.parametrize(
    "header, fragment",
    [
        ("gpus=two", "gpus"),
        ("walll=1h", "unknown header key"),
        ("wall=-5", "wall"),
        ("requeue=-1", "requeue"),
        ("requeue=once", "requeue"),
        ("bogus", "key=value"),
    ],
)
def test_a_malformed_header_is_an_error_not_a_default(header, fragment):
    spec = job_spec_from_script("x", f"#spool: {header}\nrun")
    assert any(fragment in error for error in spec.errors)


def test_a_commented_gpu_slot_mention_does_not_classify():
    assert job_spec_from_script("free-x", "# uses gpu_slot.sh 4 later\necho").gpus == 0


# -------------------------------------------------------------- the protocol


def test_a_job_runs_and_its_receipt_is_written_last(spool, clock):
    sched = make(spool, clock)
    submit(spool, "free-hello", "echo hello-from-job")
    pump(sched, lambda: rc(spool, "free-hello") is not None)
    paths = SpoolPaths(spool)
    log = paths.log("free-hello").read_text()
    assert log.startswith("=== start ")
    assert "hello-from-job" in log
    assert "=== end rc=0 reason=exit" in log
    assert record(spool, "free-hello")["rc"] == 0
    assert (paths.done / "free-hello.sh").is_file()
    assert not list(paths.running.iterdir())


def test_the_receipt_carries_the_exit_code(spool, clock):
    sched = make(spool, clock)
    submit(spool, "free-fail", "exit 7")
    pump(sched, lambda: rc(spool, "free-fail") is not None)
    assert rc(spool, "free-fail") == 7
    assert record(spool, "free-fail")["reason"] == "exit"


def test_the_watcher_removes_the_allocations_mpi_context(spool, clock):
    """A job that inherits it takes the whole allocation down when it dies."""
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


def test_core_dumps_are_disabled_for_every_job(spool, clock):
    sched = make(spool, clock)
    submit(spool, "free-ulimit", "ulimit -c")
    pump(sched, lambda: rc(spool, "free-ulimit") is not None)
    assert "\n0\n" in SpoolPaths(spool).log("free-ulimit").read_text()


# --------------------------------------------------------- device allocation


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


def test_an_impossible_request_is_refused_with_its_reason(spool, clock):
    sched = make(spool, clock)
    submit(spool, "greedy", "true", header="gpus=8")
    pump(sched, lambda: rc(spool, "greedy") is not None)
    assert rc(spool, "greedy") == RC_REFUSED
    assert "asks for 8 GPU(s)" in record(spool, "greedy")["detail"]
    assert "asks for 8 GPU(s)" in SpoolPaths(spool).log("greedy").read_text()


def test_a_misspelt_limit_refuses_the_job_instead_of_running_it_unlimited(spool, clock):
    sched = make(spool, clock)
    submit(spool, "typo", "true", header="gpus=1 walll=1h")
    pump(sched, lambda: rc(spool, "typo") is not None)
    assert rc(spool, "typo") == RC_REFUSED
    assert "walll" in record(spool, "typo")["detail"]


# ------------------------------------------------------- queue and priority


def test_a_waiting_whole_node_job_preempts_the_admission_of_slot_jobs(spool, clock):
    """Otherwise a stream of slot jobs starves whole-node engine runs indefinitely."""
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
    """An all-or-nothing retry would starve a 4-rank test behind single-device jobs."""
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


def test_higher_priority_starts_first(spool, clock):
    sched = make(spool, clock, devices=("0",))
    t0 = clock.now
    submit(spool, "hold", blocking(spool, "hold"), header="gpus=1", submitted=t0 - 10)
    pump(sched, lambda: "hold" in running(sched))
    submit(spool, "low", blocking(spool, "low"), header="gpus=1", submitted=t0 - 5)
    submit(spool, "high", blocking(spool, "high"), header="gpus=1 priority=5", submitted=t0 - 1)
    release(spool, "hold")
    pump(sched, lambda: running(sched) == {"high"})
    drain(sched, spool)


def test_the_queued_stamp_follows_the_shebang_and_is_no_header_key():
    script = "#!/bin/bash\nset -u\n#spool: gpus=1\ntrue\n"
    stamped = scheduler.stamp_queued(script, now=1760000000.25)
    assert stamped.splitlines()[:2] == ["#!/bin/bash", "#spool-queued: 1760000000.250000"]
    again = scheduler.stamp_queued(stamped, now=1760000001.5)
    assert again.count(scheduler.QUEUED_TAG) == 1 and "1760000001.500000" in again
    spec = job_spec_from_script("r1-a", again)
    assert spec.queued == 1760000001.5 and spec.gpus == 1 and not spec.errors
    assert scheduler.queued_at("true\n") == 0.0
    assert scheduler.queued_at("#spool-queued: soon\ntrue\n") == 0.0
    assert scheduler.stamp_queued("true\n", now=2.0) == "#spool-queued: 2.000000\ntrue\n"


def test_jobs_queued_in_one_second_run_in_the_order_they_were_queued(spool, clock):
    """Some shared filesystems keep a file's modification time to the second only.

    Jobs queued within that second ran in name order; they run in the order
    their submitter queued them.
    """
    sched = make(spool, clock, devices=("0",))
    second = float(int(clock.now) - 30)
    submit(spool, "hold", blocking(spool, "hold"), header="gpus=1", submitted=second - 10)
    pump(sched, lambda: "hold" in running(sched))
    for offset, name in ((0.1, "r9-zeta"), (0.2, "r1-alpha"), (0.3, "r5-mid")):
        submit(
            spool,
            name,
            blocking(spool, name),
            header="gpus=1",
            submitted=second,
            queued=second + offset,
        )
    # A job queued in the same second by a submitter that stamps nothing ranks
    # by the second alone, so ahead of those stamped within it.
    submit(
        spool, "r0-unstamped", blocking(spool, "r0-unstamped"), header="gpus=1", submitted=second
    )
    sched.tick()
    order = [p["name"] for p in sched.snapshot(clock.now)["pending"]]
    assert order == ["r0-unstamped", "r9-zeta", "r1-alpha", "r5-mid"]
    release(spool, "hold")
    release(spool, "r0-unstamped")
    pump(sched, lambda: running(sched) == {"r9-zeta"})
    release(spool, "r9-zeta")
    pump(sched, lambda: running(sched) == {"r1-alpha"})
    drain(sched, spool)


def test_an_earlier_second_still_runs_first_whatever_its_stamp_says(spool, clock):
    sched = make(spool, clock, devices=("0",))
    second = float(int(clock.now) - 30)
    submit(spool, "hold", blocking(spool, "hold"), header="gpus=1", submitted=second - 10)
    pump(sched, lambda: "hold" in running(sched))
    submit(
        spool, "late", blocking(spool, "late"), header="gpus=1", submitted=second, queued=second - 5
    )
    submit(spool, "early", blocking(spool, "early"), header="gpus=1", submitted=second - 1)
    sched.tick()
    assert [p["name"] for p in sched.snapshot(clock.now)["pending"]] == ["early", "late"]
    drain(sched, spool)


def test_waiting_ages_a_job_past_a_newer_higher_priority_one(spool, clock):
    sched = make(spool, clock, devices=("0",), aging_s=60)
    submit(spool, "hold", blocking(spool, "hold"), header="gpus=1", submitted=clock.now - 1)
    pump(sched, lambda: "hold" in running(sched))
    submit(spool, "old", blocking(spool, "old"), header="gpus=1", submitted=clock.now - 600)
    submit(spool, "new", blocking(spool, "new"), header="gpus=1 priority=3", submitted=clock.now)
    release(spool, "hold")
    pump(sched, lambda: running(sched) == {"old"})
    drain(sched, spool)


# ------------------------------------------------------------------- limits


def test_the_wall_limit_counts_from_start_not_from_submission(spool, clock):
    """Time spent queued behind another job is not charged to the job."""
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


def test_a_job_over_its_memory_limit_is_stopped_alone(spool, clock):
    """One job's memory must cost that job, not the session."""
    sched = make(spool, clock)
    submit(spool, "hog", resident(300), header="gpus=1 mem_gb=0.1")
    submit(spool, "bystander", blocking(spool, "bystander"), header="gpus=1")
    pump(sched, lambda: rc(spool, "hog") is not None)
    assert rc(spool, "hog") == RC_MEMORY_LIMIT
    assert "bystander" in running(sched)
    drain(sched, spool)


def test_memory_pressure_stops_the_largest_job(spool, clock, monkeypatch):
    sched = make(spool, clock, pressure_floor_gb=10.0)
    submit(spool, "big", resident(200), header="gpus=1")
    submit(spool, "small", blocking(spool, "small"), header="gpus=1")
    pump(sched, lambda: {"big", "small"} <= running(sched) and sched.jobs["big"].rss_gb > 0.15)
    monkeypatch.setattr(sched, "available_memory_gb", lambda: 1.0)
    pump(sched, lambda: rc(spool, "big") is not None)
    assert record(spool, "big")["reason"] == "memory_pressure"
    assert "small" in running(sched), "one victim at a time, while its memory comes back"
    monkeypatch.setattr(sched, "available_memory_gb", lambda: 100.0)
    pump(sched, lambda: "big" not in sched.jobs)
    assert "small" in running(sched)
    drain(sched, spool)


def test_cancel_is_the_sanctioned_kill(spool, clock):
    sched = make(spool, clock)
    submit(spool, "long", "sleep 300", header="gpus=1")
    pump(sched, lambda: "long" in running(sched))
    assert spoolctl.main(["--spool", str(spool), "cancel", "long"]) == 0
    pump(sched, lambda: rc(spool, "long") is not None)
    assert rc(spool, "long") == RC_CANCELLED
    pump(sched, lambda: not sched.jobs)
    assert not (SpoolPaths(spool).cancel / "long").exists()


def test_a_job_cancelled_before_it_starts_never_runs(spool, clock):
    sched = make(spool, clock)
    (SpoolPaths(spool).cancel / "never").write_text("", encoding="utf-8")
    submit(spool, "never", "echo RAN", header="gpus=1")
    pump(sched, lambda: rc(spool, "never") is not None)
    assert rc(spool, "never") == RC_CANCELLED
    assert "RAN" not in SpoolPaths(spool).log("never").read_text()


def test_processes_a_finished_job_leaves_behind_are_reaped(spool, clock):
    """A gate that leaks a server must not hand its devices over with the server still on them."""
    sched = make(spool, clock)
    submit(spool, "leaky", "sleep 300 & echo started", header="gpus=2")
    pump(sched, lambda: rc(spool, "leaky") is not None)
    assert rc(spool, "leaky") == 0
    pump(sched, lambda: not sched.jobs)
    assert "left behind" in SpoolPaths(spool).log("leaky").read_text()


def test_a_namesake_job_of_another_spool_is_not_reaped(tmp_path, clock):
    """Job names and watcher ids are unique only within a spool."""
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    mine, other = make(ours, clock, watcher_id="w"), make(theirs, clock, watcher_id="w")
    submit(theirs, "free-same", blocking(theirs, "free-same"))
    pump(other, lambda: "free-same" in running(other))
    submit(ours, "free-same", "true")
    pump(mine, lambda: not mine.jobs)
    assert other.jobs["free-same"].proc.poll() is None, "another spool's job was signalled"
    drain(other, theirs)


@pytest.mark.skipif(scheduler.adopts_orphans(), reason="this process collects orphans")
def test_a_child_the_watcher_did_not_start_keeps_its_exit_code(spool, clock):
    """Only orphans re-parented to the watcher are its to collect.

    Collecting another exited child of its process takes that child's exit
    status, and ``subprocess`` then reports 0 for it.
    """
    sched = make(spool, clock)
    child = subprocess.Popen(["bash", "-c", "exit 3"])
    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)  # exited, not yet collected
    sched.tick()
    assert child.wait(timeout=10) == 3


def test_an_mpi_attached_process_is_never_signalled(spool, clock):
    """The safe-kill rule: a process carrying the allocation's context is left alone.

    The job is still marked failed, so its waiter unblocks, and its devices
    stay quarantined until the process exits by itself.
    """
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


def test_the_safe_kill_rule_itself():
    assert scheduler.is_mpi_attached({"SLURM_JOB_ID": "1"})
    assert scheduler.is_mpi_attached({"PMI_FD": "4"})
    assert scheduler.is_mpi_attached({"PMIX_NAMESPACE": "slurm.pmix.1.0"})
    # An mpirun started inside a scrubbed job has its own namespace.
    assert not scheduler.is_mpi_attached({"PMIX_NAMESPACE": "prterun-node-123@1", "OMPI_X": "1"})


# ------------------------------------------------------------ resilience


def test_a_job_whose_watcher_died_runs_again(spool, clock):
    """Its allocation is gone, and the job with it: the next scheduler runs it (see test_spool_requeue)."""
    paths = SpoolPaths(spool)
    (paths.running / "lost.sh").write_text("echo again\n")
    scheduler.write_json_atomic(
        paths.running / "lost.json",
        {"name": "lost", "watcher": "deadhost-1", "host": "deadhost", "started": "x"},
    )
    scheduler.write_json_atomic(
        paths.watchers / "deadhost-1.json",
        {"watcher": "deadhost-1", "alive_epoch": clock.now - 3600},
    )
    sched = make(spool, clock)
    pump(sched, lambda: rc(spool, "lost") is not None)
    assert rc(spool, "lost") == 0
    lost, rerun = record(spool, "lost")["attempts"]
    assert lost["host"] == "deadhost" and "deadhost-1" in lost["detail"]
    assert rerun["host"] == sched.host
    pump(sched, lambda: not sched.jobs)


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


def test_reusing_a_name_archives_the_previous_result(spool, clock):
    sched = make(spool, clock)
    submit(spool, "free-again", "echo first")
    pump(sched, lambda: rc(spool, "free-again") is not None)
    pump(sched, lambda: not sched.jobs)
    submit(spool, "free-again", "echo second")
    pump(sched, lambda: "second" in SpoolPaths(spool).log("free-again").read_text())
    history = list(SpoolPaths(spool).history.glob("free-again.*.log"))
    assert history and "first" in history[0].read_text()


def test_the_heartbeat_and_status_files(spool, clock):
    sched = make(spool, clock, environ={**os.environ, "SLURM_JOB_ID": "777"}, session="a")
    submit(spool, "w1", blocking(spool, "w1"), header="gpus=all")
    submit(spool, "w2", blocking(spool, "w2"), header="gpus=all")
    pump(sched, lambda: "w1" in running(sched))
    paths = SpoolPaths(spool)
    beat = paths.heartbeat.read_text()
    assert beat.startswith("alive=") and beat.strip().endswith("Z")
    time.strptime(beat.strip()[len("alive=") :], "%Y-%m-%dT%H:%M:%SZ")
    assert paths.jobid.read_text().strip() == "777"
    snap = json.loads(paths.snapshot.read_text())
    assert snap["session"] == "a"
    assert [j["name"] for j in snap["running"]] == ["w1"]
    assert snap["pending"][0]["name"] == "w2" and snap["pending"][0]["reason"]
    assert spoolctl.watcher_age(spool) is not None
    assert "w2" in spoolctl.format_status(spoolctl.status(spool))
    drain(sched, spool)


def test_stop_drains_running_work_then_exits(spool, clock):
    sched = make(spool, clock, poll_s=0.02)
    submit(spool, "free-last", blocking(spool, "free-last"))
    pump(sched, lambda: "free-last" in running(sched))
    SpoolPaths(spool).stop.write_text("")
    submit(spool, "free-never", "echo never")
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault("rc", sched.serve()), daemon=True)
    thread.start()
    time.sleep(0.2)
    assert thread.is_alive()
    release(spool, "free-last")
    thread.join(timeout=10)
    assert result.get("rc") == 0
    assert rc(spool, "free-last") == 0
    assert rc(spool, "free-never") is None


def test_a_full_filesystem_does_not_end_the_session(spool, clock):
    """A shared project quota hits its limit: the heartbeat write fails with EDQUOT.

    The watcher keeps serving (its running job finishes and is reaped) instead of
    exiting and orphaning every job in the session.
    """
    import errno

    sched = make(spool, clock, poll_s=0.02)
    submit(spool, "survivor", blocking(spool, "survivor"))
    pump(sched, lambda: "survivor" in running(sched))
    real, failures = sched.heartbeat, {"n": 0, "written": 0}

    def full():
        if failures["n"] < 3:
            failures["n"] += 1
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        real()
        failures["written"] += 1

    sched.heartbeat = full
    SpoolPaths(spool).stop.write_text("")
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault("rc", sched.serve()), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not failures["written"] and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert thread.is_alive() and failures["n"] == 3 and failures["written"]
    finally:
        # Even on failure: a watcher left serving would outlive the test, and so would its job.
        release(spool, "survivor")
        thread.join(timeout=10)
    assert result.get("rc") == 0 and rc(spool, "survivor") == 0


# ---------------------------------------------------------------- the client


@pytest.fixture
def served(spool, clock):
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


def test_queueing_refuses_a_spool_nobody_watches(spool):
    with pytest.raises(spoolctl.SpoolError, match="not running"):
        spoolctl.queue_job(spool, "x", "true\n")


def test_names_are_issued_per_requester(spool, served):
    """Colliding names would let one request's job delete another request's receipt."""
    with pytest.raises(spoolctl.SpoolError, match="prefix 'r014'"):
        spoolctl.queue_job(spool, "free-r010-test1", "true\n", owner="r014")
    spoolctl.queue_job(spool, "free-r010-test1", "#spool: owner=r010\ntrue\n", owner="r010")
    spoolctl.wait_for_job(spool, "free-r010-test1", poll_s=0.05)
    with pytest.raises(spoolctl.SpoolError, match="owned by 'r010'"):
        spoolctl.queue_job(spool, "free-r010-test1", "true\n", owner=None)


def test_submit_waits_and_returns_the_jobs_exit_code(spool, served, capsys):
    code = spoolctl.main(
        ["--spool", str(spool), "submit", "--gpus", "1", "free-t1", "echo", "hi;", "exit", "3"]
    )
    out = capsys.readouterr().out
    assert code == 3
    assert "hi" in out and "--- free-t1 rc=3" in out


def test_a_submitted_job_carries_its_queue_stamp(spool, served):
    """Jobs queued within one mtime second still run in the order they were queued."""
    assert spoolctl.main(["--spool", str(spool), "submit", "--gpus", "1", "free-stamp", "true"]) == 0
    [script] = (spool / "done").glob("free-stamp*.sh")
    assert scheduler.QUEUED_TAG in script.read_text()


def test_the_shell_wrappers_keep_their_old_command_line(spool, served, tmp_path):
    """`spoolctl.py submit <name> '<cmd>'` and `gpu_slot.sh N` work with no flags."""
    env = {**os.environ, "SPOOL": str(spool)}
    env.pop("MV2FLOW_JOB_PREFIX", None)
    done = subprocess.run(
        [
            sys.executable,
            str(SPOOLCTL_PATH),
            "submit",
            "free-compat",
            f"bash {SCRIPTS_DIR / 'gpu_slot.sh'} 2 env",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert "CUDA_VISIBLE_DEVICES=0,1" in done.stdout
    assert "--- free-compat rc=0" in done.stdout
    assert record(spool, "free-compat")["gpus"] == "2"

    launched = subprocess.run(
        [sys.executable, str(SPOOLCTL_PATH), "launch", "free-later", "echo later"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert launched.returncode == 0 and "queued free-later" in launched.stdout
    spoolctl.wait_for_job(spool, "free-later", poll_s=0.05)


def test_gpu_slot_refuses_more_devices_than_the_job_holds(spool, served, tmp_path):
    env = {**os.environ, "SPOOL": str(spool)}
    done = subprocess.run(
        [
            sys.executable,
            str(SPOOLCTL_PATH),
            "submit",
            "--gpus",
            "1",
            "free-short",
            f"bash {SCRIPTS_DIR / 'gpu_slot.sh'} 2 true",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert done.returncode == 64
    assert "holds 1 device(s) but asks for 2" in done.stdout


def test_the_orchestrators_executor_speaks_the_scheduler_protocol(spool, served):
    executor = SpoolCommandExecutor(spool, poll_s=0.05, name_prefix="gate")
    result = executor.execute(CommandSpec(argv="echo DEV=$CUDA_VISIBLE_DEVICES", timeout_s=60))
    assert result.ok
    assert "DEV=0,1,2,3" in result.stdout, "a gate holds the whole node"
    name = sorted(SpoolPaths(spool).logs.glob("gate-*.log"))[-1].stem
    data = record(spool, name)
    assert data["gpus"] == "all" and data["owner"] == "gate"
    script = (SpoolPaths(spool).done / f"{name}.sh").read_text()
    assert "#spool: gpus=all wall=60 owner=gate token=" in script


def test_two_gates_in_one_millisecond_get_distinct_names(spool):
    executor = SpoolCommandExecutor(spool, poll_s=0.01)
    names = {executor._name() for _ in range(100)}
    assert len(names) == 100
