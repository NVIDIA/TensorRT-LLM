# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression jobs respect the task's session budget and retain their complete commands."""

from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.sessions import SessionSpec
from agent_flow.workflows.trtllm_modeling_v2.run.validation import regression


@pytest.mark.parametrize(
    "kind,wall,expected",
    [
        ("session_spool", "90m", 5400),
        ("session_spool", "3h", 7200),
        ("local", "90m", 7200),
    ],
)
def test_regression_timeout_uses_session_budget(kind, wall, expected):
    env = RunEnvironment(
        kind, "python", sessions=(SessionSpec("gates", "/spool", scheduler={"default_wall": wall}),)
    )
    assert regression._job_timeout(env) == expected


def test_regression_runner_preserves_test_command_with_admissible_budget(tmp_path, monkeypatch):
    sessions = (
        SessionSpec("gates", "/g", time_limit_s=7200, scheduler={"default_wall": "90m"}),
        SessionSpec("work", "/w", role="work", scheduler={"default_wall": "5m"}),
    )
    env = RunEnvironment("session_spool", "python", sessions=sessions)
    observed = []

    class Executor:
        def execute(self, request):
            observed.append(request)
            return SimpleNamespace(returncode=0, stdout="complete", stderr="")

    monkeypatch.setattr(regression, "select_command_executor", lambda *args, **kwargs: Executor())
    wf = SimpleNamespace(task_data={"trtllm_repo_path": str(tmp_path)}, _gate_env=lambda: env)
    _, run, _ = regression.job_runner(wf, "regression")
    command = "python test_driver.py --selection full-package.txt"
    assert run(command, tmp_path) == (0, "complete")
    assert observed[0].argv == command and observed[0].timeout_s == 5400
    assert regression.PACKAGE_TESTS and regression.MAX_JOBS == 3
