# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A holder of two stand-in nodes and its lead, and a session run on it (the ``two_nodes`` fixture)."""

from __future__ import annotations

import json
import sys

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import session
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import (
    select_command_executor,
    shell_runner_for,
)
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments

BOOT = {"prompts": [{"prompt": "The capital of France is", "keywords_any": ["Paris"]}]}


@pytest.fixture
def two_nodes(tmp_path, site, monkeypatch):
    """A holder of two nodes of four devices and its lead, on one spool; the task that uses it."""
    # Imported here: the stand-ins' module imports this one, and that module's tests are not
    # this one's to collect.
    from .conftest import engine_definitions
    from .test_pool import _serving

    spool = tmp_path / "spool"
    steps = tmp_path / "steps.log"
    monkeypatch.setenv("FAKE_STEP_LOG", str(steps))
    tree = tmp_path / "tree"
    (tree / "tensorrt_llm" / "libs").mkdir(parents=True)
    (tree / "tensorrt_llm" / "__init__.py").write_text("")
    (tree / "tensorrt_llm" / "libs" / "libth_common.so").write_bytes(b"build-one")
    engine_definitions(tree)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    # The steps are Slurm's own lines; `srun` on PATH is the stand-in (the `site` fixture).
    del site
    slurm = {"gpus_per_node": 4}
    block = {
        "python": sys.executable,
        "kind": "slurm",
        "spool": str(spool),
        "gpus_per_node": 4,
        "slurm": slurm,
    }
    task = {
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": str(tree),
        "environment": block,
        "accuracy": {"boot": BOOT},
    }
    environment = RunEnvironment(
        "slurm", sys.executable, spool=str(spool), slurm=slurm, gpus_per_node=4
    )

    four = ("0", "1", "2", "3")
    stops = [
        _serving(spool, host="nodeA", devices=four, holder="h1", nodes=2),
        _serving(spool, host="nodeB", devices=four, holder="h1", nodes=2),
        _serving(spool, host="batch", devices=(), holder="h1", nodes=2, lead=True),
    ]
    yield SimpleRun(task, environment, spool, steps, tmp_path)
    for stop in stops:
        stop.set()


class SimpleRun:
    def __init__(self, task, environment, spool, steps, tmp_path):
        self.task, self.environment, self.spool = task, environment, spool
        self.steps, self.tmp_path = steps, tmp_path
        #: The pin every job of this run names, when the measured tree is a pin's.
        self.pin = None

    def session(self, block):
        data = {**self.task, "deployments": [block]}
        deployment = deployments.named(data)
        placement = session.place(data, deployment)
        executor = select_command_executor(
            self.environment,
            gpus=placement.gpus,
            nodes=placement.nodes,
            once=placement.nodes > 1,
            ranks=session.stated_ranks(deployment),
            tree=data["trtllm_repo_path"],
            pin=self.pin,
            name_prefix="session",
        )
        result = session.run_session(
            deployment,
            self.environment,
            task_data=data,
            workdir=self.tmp_path / "sessions",
            runner=shell_runner_for(executor),
        )
        return placement, result

    def record(self):
        (path,) = sorted((self.spool / "done").glob("session-*.json"))[-1:]
        return json.loads(path.read_text()), path.with_suffix(".sh").read_text()
