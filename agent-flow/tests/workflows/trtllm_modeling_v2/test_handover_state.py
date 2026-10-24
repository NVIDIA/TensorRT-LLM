# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-over state keyed by the allocation it is about, so none of it goes stale."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import sessions
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
    write_json_atomic,
)

HOUR = 3600.0


def _task(tmp_path):
    return {
        "environment": {
            "kind": "session_spool",
            "spool": str(tmp_path / "gates-3"),
            "sessions": [
                {
                    "name": "g3",
                    "spool": str(tmp_path / "gates-3"),
                    "role": "gates",
                    "time_limit": "24:00:00",
                },
            ],
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
        poll_s=0.02,
        leftover_grace_s=0,
    )
    return Scheduler(spool, config, environ=env, watcher_id=f"w{job}")


def _requesting(spool, *, job) -> Scheduler:
    """A watcher of ``job`` that has reached its hand-over lead and written its request."""
    sched = _sched(spool, job=job, ends_in=600)
    sched.tick()
    assert SpoolPaths(spool).handover_request.is_file()
    return sched


def _ended(sched: Scheduler) -> None:
    """The allocation is gone: its watcher stops heart-beating (SIGTERM leaves its record)."""
    path = sched.paths.watchers / f"{sched.watcher_id}.json"
    record = json.loads(path.read_text())
    write_json_atomic(path, {**record, "alive_epoch": time.time() - 2 * HOUR})


def _archived(spool) -> dict[str, dict]:
    """``handover/archive/``, by the name each file had before its time stamp."""
    return {
        p.name.split(".", 1)[0]: json.loads(p.read_text())
        for p in sorted((Path(spool) / "handover" / "archive").glob("*.json"))
    }


def _acks(spool) -> dict[str, dict]:
    return {
        p.stem: json.loads(p.read_text())
        for p in sorted((Path(spool) / "handover" / "acks").glob("*.json"))
    }


def test_the_successor_sets_aside_its_predecessors_request_and_acks_when_it_starts(tmp_path):
    task, spool = _task(tmp_path), tmp_path / "gates-3"
    old = _requesting(spool, job="1000027")
    request_id = json.loads(SpoolPaths(spool).handover_request.read_text())["request_id"]
    sessions.acknowledge(spool, job_id="1000247", by="driver", session="g3")
    _ended(old)
    successor = _sched(spool, job="1000247")
    # An ack for the successor's own successor, written before it came up, is its own.
    sessions.acknowledge(spool, job_id="1000421", by="driver", session="g3", after="1000247")
    successor.tick()

    assert not SpoolPaths(spool).handover_request.exists()
    assert list(_acks(spool)) == ["1000247"]
    archived = _archived(spool)
    assert archived["request-1000027"]["request_id"] == request_id
    assert (
        "successor w1000247 (allocation 1000247) started"
        in (archived["request-1000027"]["archived"]["why"])
    )
    assert archived["ack-1000027"]["job_id"] == "1000247"
    assert "set aside the hand-over it succeeds" in SpoolPaths(spool).status.read_text()
    assert sessions.pending_requests(task) == []
