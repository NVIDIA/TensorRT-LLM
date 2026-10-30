# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-node jobs: one watcher per node of an allocation, all on one spool."""

from __future__ import annotations

import os
import time

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
    RC_GANG_STOPPED,
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
)

from .test_spool_scheduler import blocking, rc, record, release, submit

_ALLOCATION = "4242"


@pytest.fixture
def spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    return root


def _node(spool, clock, host, *, allocation=_ALLOCATION) -> Scheduler:
    environ = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    environ["SLURM_JOB_ID"] = allocation
    config = SchedulerConfig(devices=("0", "1", "2", "3"), leftover_grace_s=0.0)
    node = Scheduler(spool, config, clock=clock, environ=environ, watcher_id=f"{host}-w")
    node.host = host
    node.heartbeat()
    return node


def _pump(nodes, until, timeout: float = 15.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for node in nodes:
            node.tick()
        if until():
            return
        time.sleep(0.02)
    raise AssertionError(f"not reached; pending: {[n.snapshot(0)['pending'] for n in nodes]}")


def test_a_failed_part_stops_the_others_and_names_the_node(spool):
    a, b = _node(spool, time.time, "nodeA"), _node(spool, time.time, "nodeB")
    body = 'if [ "$MV2FLOW_JOB_NODE_RANK" = 1 ]; then exit 3; fi; sleep 30'
    submit(spool, "pair", body, header="nodes=2")
    _pump([a, b], lambda: rc(spool, "pair") is not None, timeout=40.0)
    receipt = record(spool, "pair")
    assert receipt["rc"] == 3 and receipt["detail"].startswith("node 1 (nodeB)")
    assert {p["rank"]: p["rc"] for p in receipt["parts"]} == {0: RC_GANG_STOPPED, 1: 3}


def test_canceling_an_unstarted_gang_releases_the_drained_node(spool):
    a, b = _node(spool, time.time, "nodeA"), _node(spool, time.time, "nodeB")
    submit(spool, "busy", blocking(spool, "busy"), header="gpus=all wall=60s")
    _pump([a], lambda: "busy" in a.jobs)
    submit(spool, "pair", "true", header="nodes=2 wall=60s")
    _pump([a, b], lambda: b._reserved_for == "pair")
    paths = SpoolPaths(spool)
    assert not (paths.gangs / "pair/go.json").exists()
    (paths.cancel / "pair").touch()
    _pump([a, b], lambda: rc(spool, "pair") is not None)
    assert record(spool, "pair")["reason"] == "cancelled"
    assert "busy" in a.jobs
    submit(spool, "after", "true", header="gpus=all wall=30s")
    _pump([b, a], lambda: rc(spool, "after") is not None)
    assert rc(spool, "after") == 0 and record(spool, "after")["host"] == "nodeB"
    release(spool, "busy")
    _pump([a, b], lambda: rc(spool, "busy") is not None)
    assert rc(spool, "busy") == 0 and not (paths.gangs / "pair").exists()
