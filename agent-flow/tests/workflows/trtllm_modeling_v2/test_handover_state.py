# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One hand-over, end to end: the alert, allocate and ack, and the successor taking the spool."""

from __future__ import annotations

import json
import os
import time

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import sessions
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor import status

from .test_spool_scheduler import FakeClock, blocking, make, pump, rc, release, submit

HOUR = 3600.0


def _task(tmp_path):
    spool = str(tmp_path / "gates-3")
    return {
        "environment": {
            "kind": "session_spool",
            "spool": spool,
            "sessions": [{"name": "g3", "spool": spool, "role": "gates", "time_limit": "24:00:00"}],
        },
    }


def _sched(spool, *, job, ends_in=20 * HOUR) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["SLURM_JOB_ID"] = job
    config = SchedulerConfig(
        devices=("0",),
        session="g3",
        ends_at=time.time() + ends_in,
        handover_lead_s=HOUR,
        leftover_grace_s=0,
    )
    return Scheduler(spool, config, environ=env, watcher_id=f"w{job}")


def test_an_ending_session_is_handed_over_to_the_successor_the_operator_acknowledged(tmp_path):
    task, spool = _task(tmp_path), tmp_path / "gates-3"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    cfg = status.load_config(workspace)

    old = _sched(spool, job="1000027", ends_in=600)
    old.tick()
    alert = status.poll_once(cfg)
    assert alert is not None and alert.kind == "SESSION_ENDING"
    assert "session allocate" in alert.detail and "--name g3" in alert.detail

    plan = sessions.plan_successor(task, "g3")
    assert plan.after == "1000027" and "afterany:1000027" in plan.call
    sessions.acknowledge(spool, job_id="1000247", by="driver", session="g3", after=plan.after)
    assert status.poll_once(cfg) is None, "the ack answers the alert"

    # A GPU job queued now cannot finish before the allocation ends: it waits for the successor.
    (SpoolPaths(spool).pending / "late.sh").write_text(
        "#!/bin/bash\n#spool: gpus=1 wall=2h\ntrue\n"
    )
    old.tick()
    assert "waits for the successor" in old.pending_reasons["late"]

    successor = _sched(spool, job="1000247")
    pump(successor, lambda: rc(spool, "late") is not None)
    assert rc(spool, "late") == 0
    # Once the successor is up, the ending session admits nothing more.
    (SpoolPaths(spool).pending / "free-short.sh").write_text("#!/bin/bash\ntrue\n")
    old.tick()
    assert old.handover["successor"] == successor.watcher_id
    assert (
        old.pending_reasons["free-short"]
        == f"this session is handing over to {successor.watcher_id}"
    )


@pytest.mark.parametrize(
    "wall, fits", [(0, False), (7200, False), (3479, True), (3480, True), (3481, False)]
)
def test_admission_respects_allocation_end_before_the_handover_window(tmp_path, wall, fits):
    spool, clock = tmp_path / "spool", FakeClock()
    SpoolPaths(spool).ensure()
    ending = make(
        spool,
        clock,
        watcher_id="ending",
        ends_at=clock() + 3600,
        handover_lead_s=1200,
    )
    submit(spool, "bounded", header=f"gpus=1 wall={wall}s")
    ending.tick()
    assert not ending.handover, "this allocation is outside its handover window"
    if fits:
        pump(ending, lambda: rc(spool, "bounded") is not None)
        assert rc(spool, "bounded") == 0
        return
    assert not ending.jobs
    assert "waits for the successor" in ending.pending_reasons["bounded"]
    assert rc(spool, "bounded") is None
    successor = make(spool, clock, watcher_id="successor", ends_at=clock() + 14400)
    if wall == 0:
        # An unlimited GPU job cannot be bounded by either allocation.
        successor.tick()
        assert not successor.jobs
        assert "waits for the successor" in successor.pending_reasons["bounded"]
    else:
        pump(successor, lambda: rc(spool, "bounded") is not None)
        assert rc(spool, "bounded") == 0
        receipt = json.loads((spool / "done/bounded.json").read_text())
        assert receipt["watcher"] == "successor"


def test_a_ready_successor_drains_the_old_session_before_its_alert_window(tmp_path):
    spool = tmp_path / "spool"
    old = _sched(spool, job="old", ends_in=2 * HOUR)
    old.tick()
    successor = _sched(spool, job="new", ends_in=3 * HOUR)
    successor.tick()
    old.tick()
    assert old.handover["successor"] == successor.watcher_id
    submit(spool, "later", header="gpus=1 wall=30s")
    old.tick()
    assert not old.jobs and "handing over" in old.pending_reasons["later"]
    pump(successor, lambda: rc(spool, "later") is not None)
    assert rc(spool, "later") == 0


def test_early_handover_preserves_the_running_job_until_it_finishes(tmp_path):
    spool = tmp_path / "spool"
    old = _sched(spool, job="old", ends_in=2 * HOUR)
    old.tick()
    submit(spool, "busy", blocking(spool, "busy"), header="gpus=1 wall=60s")
    pump(old, lambda: "busy" in old.jobs)
    successor = _sched(spool, job="new", ends_in=3 * HOUR)
    successor.tick()
    old.tick()
    assert "busy" in old.jobs and old.handover["successor"] == successor.watcher_id
    submit(spool, "later", header="gpus=1 wall=30s")
    old.tick()
    assert "later" not in old.jobs
    pump(successor, lambda: rc(spool, "later") is not None)
    release(spool, "busy")
    pump(old, lambda: rc(spool, "busy") is not None)
    assert rc(spool, "busy") == 0 and rc(spool, "later") == 0


def test_nodes_with_the_same_allocation_end_do_not_handover_to_each_other(tmp_path):
    spool, clock = tmp_path / "spool", FakeClock()
    a = make(spool, clock, watcher_id="nodeA", ends_at=clock() + 2 * HOUR)
    b = make(spool, clock, watcher_id="nodeB", ends_at=clock() + 2 * HOUR)
    a.tick()
    b.tick()
    a.tick()
    assert not a.handover and not b.handover
