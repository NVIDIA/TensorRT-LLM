# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The disagg gate: the tree's own disaggregated-serving tests for this target.

Disaggregation is proven the way the engine's CI proves it: a test in the tree that
serves the model through ``trtllm-serve disaggregated`` (context and generation workers,
the KV hand-off between them) and scores a benchmark across that hop. The disaggregated
deployment names those tests (``tests``); each runs as one pytest job
(``checks/pytest_runner.py``) under the deployment's environment. The flow deploys no
serving stack here.

The gate passes when every test the node ids select passed. A test that skipped (too few
devices, a missing model root) proves nothing and fails it, as does a node id that
selects nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..infra.compute.environment import RunEnvironment
from ..spec.deployments import Deployment
from . import pytest_runner
from .check_runtime import ShellRunner, default_shell_runner
from .verdict import GateResult

GATE_NAME = "tests"
#: Seconds one test may take: two engines to start and a benchmark to score.
DEFAULT_TIMEOUT_S = 7200


def run_disagg_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    deployment: Deployment,
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    timeout: int | None = None,
) -> GateResult:
    """Run ``deployment``'s disaggregated-serving tests and judge their reports."""
    return pytest_runner.run_tests(
        environment,
        task_data=task_data,
        deployment=deployment,
        workdir=workdir,
        runner=runner,
        timeout=timeout or deployment.timeout or DEFAULT_TIMEOUT_S,
        name=GATE_NAME,
        missing="the deployment names no test: write the target's disaggregated accuracy test "
        "beside the tree's own (the tests that serve through `trtllm-serve disaggregated`) "
        "and name its node id in the deployment's `tests`",
    )
