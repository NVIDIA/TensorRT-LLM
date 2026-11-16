# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the design cases stand on: a spec for sessions on this host, and one way to run one.

The cases in ``test_design_*.py`` were written from the design, each from an obligation of
it and not from this code; they borrow the suite's stand-in servers and nothing else.
"""

from __future__ import annotations

import copy
import sys

import pytest


@pytest.fixture
def task(tmp_path, site):
    from .conftest import engine_definitions

    tree = tmp_path / "tree"
    (tree / "tensorrt_llm").mkdir(parents=True)
    engine_definitions(tree)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    return {
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": str(tree),
        "environment": {"kind": "local", "python": sys.executable, "gpus_per_node": 8},
        "accuracy": {
            "boot": {"prompts": [{"prompt": "The capital of France is", "keywords_any": ["Paris"]}]}
        },
    }


@pytest.fixture
def run_session(task, tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.checks import session
    from agent_flow.workflows.trtllm_modeling_v2.checks.check_runtime import (
        default_shell_runner,
        in_tree,
    )
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
    from agent_flow.workflows.trtllm_modeling_v2.spec import deployments

    def run(block, *, data=None, runner=None, **kwargs):
        data = copy.deepcopy(data or task)
        data["deployments"] = [copy.deepcopy(block)]
        deployment = deployments.named(data)
        environment = RunEnvironment(kind="local", python=sys.executable)
        result = session.run_session(
            deployment,
            environment,
            task_data=data,
            workdir=tmp_path / "sessions",
            runner=runner or in_tree(default_shell_runner, data["trtllm_repo_path"]),
            **kwargs,
        )
        return result, data

    return run
