# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the orchestrator's commands go: a run configured for a spool queues them there."""

from __future__ import annotations

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import SpoolCommandExecutor
from agent_flow.workflows.trtllm_modeling_v2.operators.inventory import Inventory
from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap as preflight_stage
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from .test_workflow import Script, _install, healthy_smi


@pytest.fixture
def spool_task_file(tmp_path, repo, checkpoint):
    spool = tmp_path / "spool"
    spool.mkdir()
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(repo),
        "target": {"family": "gpt_oss", "checkpoint_segment": "gpt_oss_120b"},
        "environment": {
            "kind": "session_spool",
            "python": "/opt/venv/bin/python",
            "spool": str(spool),
            "env": {
                "TMPDIR": "/scratch/mv2flow-run-tmp",
                "CLAUDE_CODE_TMPDIR": "/scratch/mv2flow-run-tmp",
            },
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
            "model_key": "openai/gpt-oss-120b",
            "tolerance": 5.0,
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def test_a_spool_run_puts_its_commands_in_the_spool(tmp_path, spool_task_file, monkeypatch):
    """`--dry-run` stops once preflight has resolved the executor; the survey's probe is stubbed."""
    monkeypatch.setattr(preflight_stage, "run_inventory_probe", lambda *a, **k: Inventory())
    workflow = TrtllmModelingV2Workflow(
        workspace=tmp_path / "ws",
        command_runner=healthy_smi(),
        dry_run=True,
    )
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(spool_task_file)

    assert isinstance(workflow._command_executor, SpoolCommandExecutor)
    assert workflow._command_executor.root == tmp_path / "spool"

    # And the gate runner really goes there: nothing is watching, so it
    # times out, but the work landed as a file.
    result = workflow._shell_runner("echo gate-command", None, 0)
    assert result.timed_out
    scripts = sorted((tmp_path / "spool" / "pending").glob("*.sh"))
    assert scripts and "echo gate-command" in scripts[-1].read_text(encoding="utf-8")
