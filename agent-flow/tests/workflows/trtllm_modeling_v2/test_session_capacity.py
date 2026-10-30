# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole multi-node sessions count once per physical node through handover."""

import time
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run.execution import tasks


@pytest.mark.parametrize(
    "scenario,expected", [("two_nodes", 8), ("successor", 4), ("duplicate", 8), ("quarantine", 7)]
)
def test_session_capacity_counts_live_nodes_without_double_counting(
    scenario, expected, monkeypatch
):
    now = time.time()
    records = [
        dict(
            alive_epoch=now,
            ends_at_epoch=now + 1000,
            slurm_job="old",
            host=f"h{i}",
            devices=list("0123"),
        )
        for i in range(2)
    ]
    if scenario == "successor":
        records.append(
            dict(
                alive_epoch=now - 1,
                ends_at_epoch=now + 7200,
                slurm_job="new",
                host="h2",
                devices=list("0123"),
            )
        )
    elif scenario == "duplicate":
        records.append(dict(records[0], alive_epoch=now - 2))
    elif scenario == "quarantine":
        records[1]["quarantined"] = ["3"]
    records.append(dict(records[0], host="stale", alive_epoch=now - tasks.WATCHER_STALE_S - 1))
    env = SimpleNamespace(spool="gates", sessions=[])
    stage = tasks.ExecutionStage.__new__(tasks.ExecutionStage)
    stage.workflow = SimpleNamespace(_gate_env=lambda: env)
    monkeypatch.setattr(tasks, "watcher_records", lambda spool: records)
    assert stage._spool_capacity() == expected
