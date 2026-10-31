# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A job whose allocation is lost under it gets an outcome: it runs again, or says why not."""

from __future__ import annotations

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
    RERUN_TAG,
    SpoolPaths,
    write_json_atomic,
)

from .test_spool_scheduler import FakeClock, make, pump, rc, record

DEAD = "deadhost-1"


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def spool(tmp_path):
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
    write_json_atomic(
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
    write_json_atomic(
        paths.watchers / f"{DEAD}.json",
        {"watcher": DEAD, "slurm_job": "100", "alive_epoch": clock.now - 3600},
    )


def test_a_job_lost_with_its_allocation_runs_again_once(spool, clock):
    lost_run(spool, clock, "free-r063-u1b", "echo second-run")
    submitted = (SpoolPaths(spool).running / "free-r063-u1b.sh").stat().st_mtime
    sched = make(spool, clock, session="w2")
    sched._recover_orphans(clock.now, {})
    queued = SpoolPaths(spool).pending / "free-r063-u1b.sh"
    assert f"{RERUN_TAG} 1\n" in queued.read_text()
    assert queued.stat().st_mtime == submitted, "a re-run keeps its place in the queue"
    pump(sched, lambda: rc(spool, "free-r063-u1b") is not None)
    paths = SpoolPaths(spool)
    assert rc(spool, "free-r063-u1b") == 0
    assert record(spool, "free-r063-u1b")["reruns"] == 1
    # The cut-off run's log is kept in history/, ending with what became of it.
    (lost_log,) = paths.history.glob("free-r063-u1b.*.log")
    assert "the first run, cut off" in lost_log.read_text()
    assert "queued again (re-run 1 of at most 1)" in lost_log.read_text()
    assert "second-run" in paths.log("free-r063-u1b").read_text()
    assert "free-r063-u1b lost" in paths.status.read_text()
    pump(sched, lambda: not sched.jobs)
