# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--tree``: a job runs against one checkout, and a result that mixed checkouts is refused.

An MPI worker resolves the engine package from its working directory ahead
of ``PYTHONPATH``, so a worktree run with only ``PYTHONPATH`` set runs the
worktree on rank 0 and the main checkout everywhere else. These tests build a
fake package in two checkouts and reproduce that with plain ``python3``
processes, one standing in for each rank.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import (
    CommandSpec,
    select_command_executor,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import provenance
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    RC_MIXED_CHECKOUT,
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
)

IMPORT = "python3 -c 'import fakepkg'"


@pytest.fixture
def checkouts(tmp_path):
    trees = {}
    for name in ("main", "wt"):
        package = tmp_path / name / "fakepkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"WHERE = {name!r}\n", encoding="utf-8")
        trees[name] = tmp_path / name
    return trees


@pytest.fixture
def spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    return root


def scheduler(spool) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "PYTHON"))}
    return Scheduler(spool, SchedulerConfig(devices=("0",), leftover_grace_s=0), environ=env)


def run_job(spool, name, body, header) -> dict:
    paths = SpoolPaths(spool)
    (paths.pending / f"{name}.sh").write_text(
        f"#!/bin/bash\n#spool: {header}\n{body}\n", encoding="utf-8"
    )
    sched = scheduler(spool)
    end = time.monotonic() + 30
    while time.monotonic() < end:
        sched.tick()
        if paths.receipt(name).exists() and not sched.jobs:
            return json.loads(paths.record(name).read_text())
        time.sleep(0.02)
    raise AssertionError(paths.log(name).read_text() if paths.log(name).exists() else "no log")


def test_a_tree_job_runs_in_the_tree_and_imports_from_it(spool, checkouts):
    body = f"pwd; echo PYTHONPATH=$PYTHONPATH; {IMPORT}"
    record = run_job(spool, "t", body, f"gpus=1 tree={checkouts['wt']} resolve=fakepkg")
    assert record["rc"] == 0 and record["reason"] == "exit"
    log = SpoolPaths(spool).log("t").read_text()
    assert f"\n{checkouts['wt']}\n" in log
    assert "=== tree " in log
    assert f"module=fakepkg origin={checkouts['wt']}/fakepkg/__init__.py" in log
    entries = provenance.read_records(SpoolPaths(spool).resolve_log("t"))
    assert [e["origin"] for e in entries] == [f"{checkouts['wt']}/fakepkg/__init__.py"]


def test_ranks_that_resolved_different_checkouts_are_refused(spool, checkouts):
    """The mixed shape: one rank imports the tree, another the main checkout via its cwd."""
    body = f"{IMPORT}; (cd {checkouts['main']} && {IMPORT})"
    record = run_job(spool, "mixed", body, f"gpus=1 tree={checkouts['wt']} resolve=fakepkg")
    assert record["rc"] == RC_MIXED_CHECKOUT
    assert record["reason"] == "mixed_checkout"
    assert "2 checkouts" in record["detail"]
    assert str(checkouts["main"]) in record["detail"] and str(checkouts["wt"]) in record["detail"]


def test_the_orchestrators_gates_declare_the_checkout_under_test(tmp_path, checkouts):
    env = RunEnvironment(kind="session_spool", python="/p", spool=str(tmp_path))
    executor = select_command_executor(env, tree=checkouts["wt"])
    assert executor.tree == str(checkouts["wt"].resolve())

    executor.poll_s = 0.01
    executor.execute(CommandSpec(argv="true", timeout_s=0))
    script = next((tmp_path / "pending").glob("*.sh")).read_text()
    assert f"tree={checkouts['wt'].resolve()}" in script
