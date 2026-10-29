# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-node jobs: one watcher per node of an allocation, all on one spool."""

from __future__ import annotations

import os
import time
from pathlib import Path

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


_REPORT = (
    'echo "$MV2FLOW_JOB_NODE_RANK $MV2FLOW_JOB_NODES $MV2FLOW_JOB_HOSTS $MV2FLOW_JOB_MASTER" > '
)


def test_a_multi_node_job_runs_once_on_every_node(spool):
    a, b = _node(spool, time.time, "nodeA"), _node(spool, time.time, "nodeB")
    out = Path(spool) / "out"
    submit(spool, "pair", _REPORT + f'"{out}.$MV2FLOW_JOB_NODE_RANK"', header="nodes=2")
    _pump([a, b], lambda: rc(spool, "pair") is not None)
    assert rc(spool, "pair") == 0
    assert (Path(f"{out}.0")).read_text() == "0 2 nodeA,nodeB nodeA\n"
    assert (Path(f"{out}.1")).read_text() == "1 2 nodeA,nodeB nodeA\n"
    receipt = record(spool, "pair")
    assert receipt["hosts"] == ["nodeA", "nodeB"]
    assert [p["rc"] for p in receipt["parts"]] == [0, 0]
    assert SpoolPaths(spool).log("pair.node1").is_file()
    assert not (SpoolPaths(spool).gangs / "pair").exists()
    _pump([a, b], lambda: not a.jobs and not b.jobs)


def test_a_member_drains_for_it_and_takes_no_new_work_meanwhile(spool):
    a, b = _node(spool, time.time, "nodeA"), _node(spool, time.time, "nodeB")
    submit(spool, "busy", blocking(spool, "busy"), header="gpus=1")
    _pump([b], lambda: "busy" in b.jobs)
    submit(spool, "pair", "true", header="nodes=2 priority=5")
    _pump([a, b], lambda: b._reserved_for == "pair")
    submit(spool, "late", "true", header="gpus=1")
    for _ in range(3):
        b.tick()
    assert "late" not in b.jobs
    reasons = {p["name"]: p["reason"] for p in b.snapshot(time.time())["pending"]}
    assert reasons["late"] == "this node is reserved for the multi-node job pair"
    assert rc(spool, "pair") is None
    release(spool, "busy")
    _pump([a, b], lambda: rc(spool, "pair") == 0)


def test_a_failed_part_stops_the_others_and_names_the_node(spool):
    a, b = _node(spool, time.time, "nodeA"), _node(spool, time.time, "nodeB")
    body = 'if [ "$MV2FLOW_JOB_NODE_RANK" = 1 ]; then exit 3; fi; sleep 30'
    submit(spool, "pair", body, header="nodes=2")
    _pump([a, b], lambda: rc(spool, "pair") is not None, timeout=40.0)
    receipt = record(spool, "pair")
    assert receipt["rc"] == 3 and receipt["detail"].startswith("node 1 (nodeB)")
    assert {p["rank"]: p["rc"] for p in receipt["parts"]} == {0: RC_GANG_STOPPED, 1: 3}


def test_a_job_wider_than_the_allocation_waits_without_holding_the_node(spool):
    a = _node(spool, time.time, "nodeA")
    # Another allocation's watcher on the spool is not one of this job's nodes.
    _node(spool, time.time, "nodeZ", allocation="other")
    submit(spool, "pair", "true", header="nodes=2")
    submit(spool, "single", "true", header="gpus=1")
    # A multi-node job that asks for part of a node is refused outright.
    submit(spool, "partial", "true", header="nodes=2 gpus=2")
    _pump([a], lambda: rc(spool, "single") == 0 and rc(spool, "partial") is not None)
    reasons = {p["name"]: p["reason"] for p in a.snapshot(time.time())["pending"]}
    assert reasons["pair"] == f"asks for 2 nodes; allocation {_ALLOCATION} has 1 live on this spool"
    assert "a multi-node job holds whole nodes" in record(spool, "partial")["detail"]


def test_a_session_of_several_nodes_runs_a_watcher_on_each(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import sessions, spoolctl

    gates = str(tmp_path / "gates")
    task = {
        "environment": {
            "kind": "session_spool",
            "spool": gates,
            "sessions": [{"name": "g", "spool": gates, "role": "gates", "nodes": 2}],
        }
    }
    (session,) = sessions.sessions_from_task(task)
    assert session.nodes == 2
    call = sessions.broker_submission(task, "g")
    assert "nodes=2" in call and "ntasks_per_node=1" in call
    entry = sessions.render_session_entry(task, "g")
    assert "session_entry.$(hostname).log" in entry and "runs once on each node" in entry
    errors: list[str] = []
    sessions.validate_sessions({"sessions": [{"name": "g", "spool": gates, "nodes": 0}]}, errors)
    assert any("nodes' must be an integer >= 1" in e for e in errors)
    # A job submitted with --nodes holds that many nodes.
    args = spoolctl.build_parser().parse_args(["submit", "--nodes", "2", "pair", "true"])
    assert spoolctl.job_header(args, None, "t")["nodes"] == 2
