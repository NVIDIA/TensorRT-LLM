# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A job whose allocation is lost under it gets an outcome: it runs again, or says why not.

Slurm requeues an allocation onto another node, or ends it, and the jobs that
ran on it go with it. The scheduler that next drains the spool finds them in
``running/``; these are the cases it meets, and what ``spoolctl`` says about
them to whoever is waiting.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import scheduler, spoolctl
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    RC_CANCELLED,
    RC_NODE_LOST,
    RC_ORPHANED,
    SpoolPaths,
    write_json_atomic,
)

from .test_spool_scheduler import (
    FakeClock,
    blocking,
    drain,
    make,
    pump,
    rc,
    record,
    release,
    running,
    submit,
)

#: The watcher of the allocation that was lost.
DEAD = "deadhost-1"


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    return root


def lost_run(
    spool,
    clock,
    name: str,
    body: str = "echo again",
    *,
    header: str = "",
    token: str = "tok-1",
    watcher: str = DEAD,
    silent_s: float = 3600.0,
    slurm_job: str = "100",
    restart: int = 0,
    submitted: float | None = None,
) -> None:
    """``name`` as the next scheduler finds it: started by ``watcher``, silent for ``silent_s``."""
    paths = SpoolPaths(spool)
    script = paths.running / f"{name}.sh"
    script.write_text(
        f"#!/bin/bash\nset -u\n#spool: token={token} {header}\n{body}\n", encoding="utf-8"
    )
    if submitted is not None:
        os.utime(script, (submitted, submitted))
    paths.log(name).write_text("=== start on deadhost ===\nthe first run, cut off\n")
    write_json_atomic(
        paths.running / f"{name}.json",
        {
            "name": name,
            "watcher": watcher,
            "host": "deadhost",
            "pid": 4242,
            "devices": ["0", "3"],
            "started": "2026-10-18T21:33:35Z",
            "started_epoch": clock.now - 900,
            "token": token,
            "owner": "",
            "slurm_job": slurm_job,
            "restart": restart,
            "session": "w2",
        },
    )
    write_json_atomic(
        paths.watchers / f"{watcher}.json",
        {"watcher": watcher, "slurm_job": slurm_job, "alive_epoch": clock.now - silent_s},
    )


# ------------------------------------------------------------- run it again


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


def test_a_job_run_again_keeps_its_place_in_the_queue(spool, clock):
    sched = make(spool, clock)
    submit(spool, "holder", blocking(spool, "holder"), header="gpus=all")
    pump(sched, lambda: "holder" in running(sched))
    submit(spool, "newer", blocking(spool, "newer"), header="gpus=all")
    lost_run(
        spool,
        clock,
        "older",
        blocking(spool, "older"),
        header="gpus=all",
        submitted=clock.now - 600,
    )
    sched.tick()
    assert (SpoolPaths(spool).pending / "older.sh").exists()
    release(spool, "holder")
    pump(sched, lambda: running(sched) - {"holder"})
    assert running(sched) == {"older"}
    drain(sched, spool)


def test_a_requeue_record_of_another_submission_is_not_this_jobs(spool, clock):
    paths = SpoolPaths(spool)
    write_json_atomic(
        paths.requeue_record("free-reused"),
        {"name": "free-reused", "token": "an-earlier-job", "attempts": [{"attempt": 1}]},
    )
    lost_run(spool, clock, "free-reused")
    sched = make(spool, clock)
    pump(sched, lambda: rc(spool, "free-reused") is not None)
    assert rc(spool, "free-reused") == 0
    assert [run["attempt"] for run in record(spool, "free-reused")["attempts"]] == [1, 2]
    pump(sched, lambda: not sched.jobs)


def test_an_earlier_start_of_a_requeued_allocation_is_lost_at_once(spool, clock):
    """Its watcher heart-beat a minute ago, but Slurm ended that start before this one."""
    lost_run(spool, clock, "free-early", silent_s=60, slurm_job="555", restart=0)
    environ = {**os.environ, "SLURM_JOB_ID": "555", "SLURM_RESTART_COUNT": "1"}
    sched = make(spool, clock, environ=environ)
    pump(sched, lambda: rc(spool, "free-early") is not None)
    assert rc(spool, "free-early") == 0
    lost, rerun = record(spool, "free-early")["attempts"]
    assert "allocation 555 was requeued" in lost["detail"] and lost["restart"] == 0
    assert (rerun["slurm_job"], rerun["restart"]) == ("555", 1)
    pump(sched, lambda: not sched.jobs)


def test_a_run_under_this_watchers_name_from_before_it_started_is_lost(spool, clock):
    """A node rebooted and the requeued start got the same pid: the same watcher id."""
    lost_run(spool, clock, "free-namesake", watcher="w", silent_s=0)
    sched = make(spool, clock, watcher_id="w")
    pump(sched, lambda: rc(spool, "free-namesake") is not None)
    assert rc(spool, "free-namesake") == 0
    lost, _ = record(spool, "free-namesake")["attempts"]
    assert "before the watcher of that name" in lost["detail"]
    pump(sched, lambda: not sched.jobs)


def test_a_run_of_a_watcher_still_heart_beating_is_left_alone(spool, clock):
    lost_run(spool, clock, "free-theirs", silent_s=60, slurm_job="555", restart=1)
    environ = {**os.environ, "SLURM_JOB_ID": "555", "SLURM_RESTART_COUNT": "1"}
    make(spool, clock, environ=environ).tick()
    assert rc(spool, "free-theirs") is None
    assert (SpoolPaths(spool).running / "free-theirs.sh").exists()


# ------------------------------------------------------------ or say why not


def test_a_job_that_opts_out_is_closed_out_node_lost_with_where_it_ran(spool, clock):
    lost_run(spool, clock, "once", header="gpus=2 requeue=0 label=probe", submitted=clock.now - 60)
    sched = make(spool, clock)
    sched.tick()
    assert rc(spool, "once") == RC_NODE_LOST
    got = record(spool, "once")
    assert got["reason"] == "node_lost" and "requeue=0" in got["detail"]
    assert (got["gpus"], got["label"], got["devices"]) == ("2", "probe", ["0", "3"])
    assert got["submitted"] and got["token"] == "tok-1"
    (only,) = got["attempts"]
    assert (only["host"], only["outcome"]) == ("deadhost", "node_lost")
    assert not (SpoolPaths(spool).pending / "once.sh").exists()
    assert "not run again" in SpoolPaths(spool).log("once").read_text()
    assert sched.counters["node_lost"] == 1


def test_the_session_decides_for_a_job_that_does_not(spool, clock):
    lost_run(spool, clock, "free-default")
    lost_run(spool, clock, "free-asked", header="requeue=1")
    sched = make(spool, clock, requeue=0)
    pump(sched, lambda: rc(spool, "free-asked") is not None)
    assert rc(spool, "free-default") == RC_NODE_LOST
    assert rc(spool, "free-asked") == 0
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


def test_a_job_whose_cancel_was_requested_is_not_run_again(spool, clock):
    lost_run(spool, clock, "free-withdrawn")
    spoolctl.cancel_job(spool, "free-withdrawn")
    make(spool, clock).tick()
    assert rc(spool, "free-withdrawn") == RC_NODE_LOST
    assert "cancel was requested" in record(spool, "free-withdrawn")["detail"]
    assert not (SpoolPaths(spool).cancel / "free-withdrawn").exists()


def test_a_job_cancelled_while_queued_again_keeps_the_run_it_lost(spool, clock):
    sched = make(spool, clock)
    submit(spool, "holder", blocking(spool, "holder"), header="gpus=all")
    pump(sched, lambda: "holder" in running(sched))
    lost_run(spool, clock, "again", "true", header="gpus=all")
    sched.tick()
    spoolctl.cancel_job(spool, "again")
    pump(sched, lambda: rc(spool, "again") is not None)
    assert rc(spool, "again") == RC_CANCELLED
    (lost,) = record(spool, "again")["attempts"]
    assert lost["host"] == "deadhost"
    assert not SpoolPaths(spool).requeue_record("again").exists()
    drain(sched, spool)


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


def test_a_watcher_that_falls_silent_does_not_record_over_the_run_that_replaced_it(spool, clock):
    paths = SpoolPaths(spool)
    first = make(spool, clock, watcher_id="first")
    submit(spool, "free-slow", blocking(spool, "free-slow"))
    pump(first, lambda: "free-slow" in running(first))
    # While "first" was silent, "second" found the run lost and started it again.
    meta = json.loads((paths.running / "free-slow.json").read_text())
    write_json_atomic(paths.running / "free-slow.json", {**meta, "watcher": "second"})
    write_json_atomic(
        paths.watchers / "second.json", {"watcher": "second", "alive_epoch": clock.now}
    )
    release(spool, "free-slow")
    pump(first, lambda: not first.jobs)
    assert rc(spool, "free-slow") is None
    assert (paths.running / "free-slow.sh").exists()
    assert "is not recorded" in paths.log("free-slow").read_text()


# ------------------------------------------------------------- what is said


def test_status_shows_a_job_queued_again_and_the_run_it_lost(spool, clock):
    sched = make(spool, clock, session="w2")
    submit(spool, "holder", blocking(spool, "holder"), header="gpus=all")
    pump(sched, lambda: "holder" in running(sched))
    lost_run(spool, clock, "r063-u1b-base", "true", header="gpus=2")
    sched.tick()
    data = spoolctl.status(spool)
    (entry,) = data["requeued"]
    assert (entry["name"], entry["state"]) == ("r063-u1b-base", "pending")
    (mine,) = [w for w in data["watchers"] if w["watcher"] == sched.watcher_id]
    (waiting,) = mine["pending"]
    assert waiting["attempt"] == 2
    text = spoolctl.format_status(data)
    assert "queued again after a lost allocation" in text
    assert "r063-u1b-base: run 2 of at most 2, pending" in text
    assert "run 1: allocation 100, start 0 on deadhost" in text
    assert "(run 2)" in text
    drain(sched, spool)


def test_wait_reports_the_requeue_and_returns_the_reruns_receipt(spool, clock, capsys):
    name = "free-r063-u1b"
    lost_run(spool, clock, name, blocking(spool, name))
    sched = make(spool, time.time)
    sched.tick()
    threading.Timer(0.5, release, args=(spool, name)).start()
    thread = threading.Thread(
        target=pump, args=(sched, lambda: rc(spool, name) is not None and not sched.jobs, 30)
    )
    thread.start()
    try:
        code = spoolctl.main(["--spool", str(spool), "wait", name])
    finally:
        thread.join(timeout=30)
    captured = capsys.readouterr()
    assert code == 0
    assert "deadhost" in captured.err and "queued again for run 2 of at most 2" in captured.err
    got = json.loads(captured.out)
    assert [run["host"] for run in got["attempts"]] == ["deadhost", sched.host]


def test_submit_lists_every_allocation_the_job_ran_on(spool, clock, capsys, monkeypatch):
    write_json_atomic(
        SpoolPaths(spool).watchers / "live.json", {"watcher": "live", "alive_epoch": time.time()}
    )
    runs = [
        {"attempt": 1, "slurm_job": "100", "restart": 0, "host": "deadhost", "outcome": "requeued"},
        {"attempt": 2, "slurm_job": "100", "restart": 1, "host": "newhost", "outcome": "exit"},
    ]
    monkeypatch.setattr(
        spoolctl,
        "wait_for_job",
        lambda *a, **k: {"rc": 0, "reason": "exit", "attempt": 2, "attempts": runs},
    )
    assert spoolctl.main(["--spool", str(spool), "submit", "free-s", "true"]) == 0
    out = capsys.readouterr().out
    assert "--- free-s ran on 2 allocation(s):" in out
    assert "run 1: allocation 100, start 0 on deadhost" in out
    assert "run 2: allocation 100, start 1 on newhost" in out


def test_a_wait_its_session_ends_says_the_job_is_still_in_the_spool(spool, clock, capsys):
    lost_run(spool, clock, "free-left")
    code = spoolctl.main(["--spool", str(spool), "wait", "free-left"])
    err = capsys.readouterr().err
    assert code == spoolctl.EXIT_NO_SESSION
    assert "stopped heart-beating" in err and "free-left is still in running/" in err
    assert "spoolctl.py wait free-left" in err


def test_session_wait_rides_out_a_session_that_comes_back(spool, clock):
    lost_run(spool, clock, "free-back", silent_s=200)
    argv = ["--spool", str(spool), "wait", "--session-wait", "10m", "free-back"]
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault("code", spoolctl.main(argv)))
    thread.start()
    try:
        time.sleep(0.5)
        assert thread.is_alive(), "the wait gave up on a session that was given time to come back"
        sched = make(spool, time.time)
        pump(sched, lambda: rc(spool, "free-back") is not None and not sched.jobs)
    finally:
        thread.join(timeout=15)
    assert result.get("code") == 0


def test_the_queue_timeout_counts_from_when_the_job_was_queued_again(spool):
    paths = SpoolPaths(spool)
    write_json_atomic(paths.watchers / "live.json", {"watcher": "live", "alive_epoch": time.time()})
    submit(spool, "free-q", "true")
    write_json_atomic(
        paths.requeue_record("free-q"),
        {"name": "free-q", "attempts": [{"attempt": 1}], "requeued_epoch": time.time()},
    )

    def finish():
        write_json_atomic(paths.record("free-q"), {"name": "free-q", "rc": 0})
        paths.receipt("free-q").write_text("0\n")

    threading.Timer(0.3, finish).start()
    got = spoolctl.wait_for_job(
        spool, "free-q", poll_s=0.05, queue_timeout_s=60, submitted_at=time.time() - 3600
    )
    assert got["rc"] == 0


def test_resubmitting_a_name_drops_what_its_earlier_job_left(spool, clock):
    """A cancel asked after the earlier job ended would refuse the new one before it ran."""
    paths = SpoolPaths(spool)
    write_json_atomic(paths.watchers / "live.json", {"watcher": "live", "alive_epoch": time.time()})
    spoolctl.cancel_job(spool, "free-again")
    write_json_atomic(
        paths.requeue_record("free-again"),
        {"name": "free-again", "token": "old", "attempts": [{"attempt": 1}]},
    )
    spoolctl.queue_job(spool, "free-again", "true\n")
    assert not (paths.cancel / "free-again").exists()
    assert not paths.requeue_record("free-again").exists()
    sched = make(spool, clock)
    pump(sched, lambda: rc(spool, "free-again") is not None)
    assert rc(spool, "free-again") == 0


def test_a_submitter_says_how_often_its_job_may_run_again(spool):
    parser = spoolctl.build_parser()
    args = parser.parse_args(["--spool", str(spool), "launch", "--requeue", "0", "free-x", "true"])
    header = spoolctl.job_header(args, None, "tok")
    spec = scheduler.job_spec_from_script("free-x", spoolctl.render_job("true", header=header))
    assert spec.requeue == 0 and not spec.errors
    args = parser.parse_args(["--spool", str(spool), "launch", "--requeue", "-1", "free-x", "true"])
    with pytest.raises(spoolctl.SpoolError, match="--requeue"):
        spoolctl.job_header(args, None, "tok")


def test_the_restart_count_is_slurms():
    assert scheduler.restart_count({"SLURM_JOB_ID": "1", "SLURM_RESTART_COUNT": "2"}) == 2
    assert scheduler.restart_count({"SLURM_JOB_ID": "1"}) == 0
    assert scheduler.restart_count({}) is None


def test_what_an_operator_reads_is_documented_in_those_words():
    import agent_flow.workflows.trtllm_modeling_v2 as pkg

    doc = (Path(pkg.__file__).parent / "TROUBLESHOOTING.md").read_text(encoding="utf-8")
    assert "`reason=node_lost` (rc 218)" in doc
    assert scheduler.RC_NODE_LOST == 218 and "node_lost" in scheduler.SESSION_LOSS_REASONS
    assert "queued again for run 2 of at most 2" in doc and "is still in running/" in doc


def test_a_job_lost_with_its_allocation_is_infrastructure_everywhere():
    """A ``node_lost`` close-out is the session's loss, not a verdict, for every reader."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.execution import spool
    from agent_flow.workflows.trtllm_modeling_v2.tune import measure

    assert spool.SESSION_LOSS_REASONS == scheduler.SESSION_LOSS_REASONS
    assert {"orphaned", "node_lost"} <= spool.SESSION_LOSS_REASONS
    assert scheduler.RC_NODE_LOST in measure.INFRA_RCS
