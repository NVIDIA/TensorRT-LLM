# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the orchestrator's commands go: a run configured for a spool queues them there."""

from __future__ import annotations

import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import SpoolCommandExecutor
from agent_flow.workflows.trtllm_modeling_v2.operators.inventory import Inventory
from agent_flow.workflows.trtllm_modeling_v2.run.records.state import STAGE_PREFLIGHT
from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap as preflight_stage
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from .test_workflow import Script, _install, _state, healthy_smi


def test_a_spool_run_puts_its_commands_in_the_spool(tmp_path, task_file, monkeypatch):
    """`--dry-run` stops once preflight has resolved the executor; the survey's probe is stubbed.

    It stops before the first agent turn and leaves the checkpoint at preflight, so the real
    run re-runs it.
    """
    spool = tmp_path / "spool"
    spool.mkdir()
    spec = yaml.safe_load(task_file.read_text())
    spec["environment"].update(kind="session_spool", spool=str(spool))
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    monkeypatch.setattr(preflight_stage, "run_inventory_probe", lambda *a, **k: Inventory())
    workflow = TrtllmModelingV2Workflow(
        workspace=tmp_path / "ws",
        command_runner=healthy_smi(),
        dry_run=True,
    )
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)

    assert calls == [] and _state(tmp_path).stage == STAGE_PREFLIGHT
    assert isinstance(workflow._command_executor, SpoolCommandExecutor)
    assert workflow._command_executor.root == spool

    # And the gate runner really goes there: nothing is watching, so it
    # times out, but the work landed as a file.
    result = workflow._shell_runner("echo gate-command", None, 0)
    assert result.timed_out
    scripts = sorted((spool / "pending").glob("*.sh"))
    assert scripts and "echo gate-command" in scripts[-1].read_text(encoding="utf-8")
