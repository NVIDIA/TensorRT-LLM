# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A job whose allocation is lost under it gets an outcome: it runs again, or says why not."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    RC_NODE_LOST,
    RC_ORPHANED,
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


def lost_run(spool, clock, name: str, body: str = "echo again") -> None:
    """``name`` as the next scheduler finds it: started by ``DEAD``, silent for an hour."""
    paths = SpoolPaths(spool)
    script = paths.running / f"{name}.sh"
    script.write_text(f"#!/bin/bash\nset -u\n#spool: token=tok-1\n{body}\n", encoding="utf-8")
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


def test_a_job_lost_with_its_allocation_runs_again_and_its_receipt_lists_both_runs(spool, clock):
    lost_run(spool, clock, "free-r063-u1b", "echo second-run")
    sched = make(spool, clock, session="w2")
    pump(sched, lambda: rc(spool, "free-r063-u1b") is not None)
    paths = SpoolPaths(spool)
    assert rc(spool, "free-r063-u1b") == 0
    got = record(spool, "free-r063-u1b")
    assert got["attempt"] == 2
    first, second = got["attempts"]
    assert (first["host"], first["slurm_job"], first["outcome"]) == ("deadhost", "100", "requeued")
    assert first["devices"] == ["0", "3"] and DEAD in first["detail"]
    assert (second["host"], second["outcome"], second["rc"]) == (sched.host, "exit", 0)
    # The cut-off run's log is kept in history/, ending with what became of it.
    lost_log = Path(first["log"]).read_text()
    assert Path(first["log"]).parent == paths.history
    assert "the first run, cut off" in lost_log and "queued again for run 2" in lost_log
    assert "second-run" in paths.log("free-r063-u1b").read_text()
    assert not paths.requeue_record("free-r063-u1b").exists()
    assert sched.counters["requeued"] == 1
    assert "free-r063-u1b lost" in paths.status.read_text()
    pump(sched, lambda: not sched.jobs)


def test_a_job_lost_again_after_its_last_rerun_is_closed_out_node_lost(spool, clock):
    paths = SpoolPaths(spool)
    write_json_atomic(
        paths.requeue_record("free-twice"),
        {
            "name": "free-twice",
            "token": "tok-1",
            "requeue": 1,
            "attempts": [{"attempt": 1, "host": "firsthost", "slurm_job": "99"}],
        },
    )
    lost_run(spool, clock, "free-twice")
    make(spool, clock).tick()
    assert rc(spool, "free-twice") == RC_NODE_LOST
    got = record(spool, "free-twice")
    assert [run["host"] for run in got["attempts"]] == ["firsthost", "deadhost"]
    assert "already ran again 1 time(s)" in got["detail"]
    assert not paths.requeue_record("free-twice").exists()


def test_a_job_whose_processes_outlived_their_watcher_is_orphaned_not_run_beside_itself(
    spool, clock
):
    lost_run(spool, clock, "free-alive")
    marker = {"MV2FLOW_SPOOL": str(spool), "MV2FLOW_SPOOL_JOB": "free-alive"}
    stray = subprocess.Popen(
        ["sleep", "60"], env={**os.environ, **marker, "MV2FLOW_SPOOL_WATCHER": DEAD}
    )
    try:
        make(spool, clock).tick()
    finally:
        stray.kill()
        stray.wait()
    assert rc(spool, "free-alive") == RC_ORPHANED
    detail = record(spool, "free-alive")["detail"]
    assert "still runs here" in detail and str(stray.pid) in detail
    assert not (SpoolPaths(spool).pending / "free-alive.sh").exists()
