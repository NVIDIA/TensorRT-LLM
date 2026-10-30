# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One hand-over, end to end: the alert, allocate and ack, and the successor taking the spool."""

from __future__ import annotations

import os
import time

import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import sessions
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor import status

from .test_spool_scheduler import pump, rc

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
