# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a job goes: a dead session refuses it, and the gate pool picks a session that can finish it.

Each session is a real :class:`Scheduler` that ticks once, so its status
record is the file a live watcher writes, which needs no GPU and no job.
Times are relative to the wall clock because ``spoolctl`` reads it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import placement, spoolctl
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import SpoolCommandExecutor
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
    Scheduler,
    SchedulerConfig,
    parse_duration,
)

DEVICES = ("0", "1", "2", "3")
HOUR = 3600.0


def session(
    parent: Path,
    name: str,
    *,
    left_s: float = 20 * HOUR,
    lead_s: float = HOUR,
    age_s: float = 0.0,
    default_wall: str = "2h",
) -> Path:
    """A spool watched by one scheduler that has just ticked ``age_s`` ago."""
    root = parent / name
    now = time.time()
    config = SchedulerConfig(
        devices=DEVICES,
        session=name,
        ends_at=now + left_s,
        handover_lead_s=lead_s,
        default_wall_s=parse_duration(default_wall),
    )
    Scheduler(
        root,
        config,
        clock=lambda: now - age_s,
        environ={"SLURM_JOB_ID": "100"},
        watcher_id=f"{name}-w",
    ).tick()
    return root


def pending(root: Path) -> list[str]:
    return sorted(p.name[: -len(".sh")] for p in (root / "pending").glob("*.sh"))


@pytest.fixture(autouse=True)
def _no_ambient_spool(monkeypatch):
    for key in ("SPOOL", "MV2FLOW_SPOOL", "MV2FLOW_JOB_PREFIX"):
        monkeypatch.delenv(key, raising=False)


def test_a_dead_watcher_refuses_the_job(tmp_path, capsys):
    work1 = session(tmp_path, "work-1", age_s=600)
    code = spoolctl.main(["--spool", str(work1), "launch", "--gpus", "1", "r1-x", "true"])
    assert code == spoolctl.EXIT_NO_SESSION
    assert pending(work1) == []
    assert "stale: the session is gone" in capsys.readouterr().err


def test_the_gate_pool_skips_a_session_in_hand_over_for_one_that_can_take_the_job(tmp_path):
    """The least loaded gate spool was the one about to end: the gate would park there."""
    ending = session(tmp_path, "gates-2", left_s=0.5 * HOUR, lead_s=1.5 * HOUR, default_wall="3h")
    busy = session(tmp_path, "gates-1", default_wall="3h")
    (busy / "pending" / "other.sh").write_text("true\n")
    executor = SpoolCommandExecutor(ending, pool=[ending, busy])
    gate = placement.JobNeeds(gpus=None, wall_s=None, holds_gpus=True)
    assert executor._pick(gate) == busy
