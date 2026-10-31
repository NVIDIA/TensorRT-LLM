# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gates — the commands whose exit codes are the verdicts."""

from __future__ import annotations

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment

_TASK = {
    "checkpoint_path": "/ckpt/Qwen3-30B-A3B",
    "parallel": "tp1",
    "trtllm_repo_path": "/repo",
    "accuracy": {
        "benchmark": "gsm8k",
        "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
        "tolerance": 5.0,
    },
}


def _env(world_size: int = 1, devices=(0,)):
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/opt/venv/bin/python", "env": {}}},
        devices=devices,
        world_size=world_size,
    )


def recorder(result: CommandResult | list[CommandResult]):
    """A runner that returns canned results and records the commands it saw."""
    results = result if isinstance(result, list) else [result]
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return results[min(len(seen) - 1, len(results) - 1)]

    run.seen = seen  # type: ignore[attr-defined]
    return run


# ------------------------------------------------------- a lost session


def test_a_cancelled_gate_has_no_verdict_and_is_not_queued_again_at_once(tmp_path):
    """A gate job cancelled through the spool raises: missing, not FAIL, and no re-run here."""
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import CANCELLED
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import check_runner
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.environment = RunEnvironment(kind="local", python="python3", world_size=1)
    workflow.task_data = dict(_TASK, trtllm_repo_path=str(tmp_path))
    runner = check_runner.CheckRunner(workflow)
    runner._await_session = lambda log: pytest.fail("a cancelled gate waits for no session")
    cancelled = f"{CANCELLED}: cancelled before it started"
    runs = []

    def run():
        runs.append(1)
        return SimpleNamespace(
            name="accuracy[gsm8k]", member="dep4", passed=False, infrastructure=cancelled
        )

    with pytest.raises(check_runner._SessionLost, match="measured nothing"):
        runner._gate_until_verdict(run, None)
    assert runs == [1]
