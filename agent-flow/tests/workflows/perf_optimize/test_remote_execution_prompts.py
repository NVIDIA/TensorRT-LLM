# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote execution policy and per-turn context contracts."""

from __future__ import annotations

from pathlib import Path

import fsspec
import pytest

from agent_flow.workflows.perf_optimize.execution import ExecutionLayout, RunFileSystems
from agent_flow.workflows.perf_optimize.prompts import (
    DEFAULT_PROMPTS,
    REMOTE_EXECUTION_POLICY,
    RemoteExecutionContext,
    append_remote_execution_context,
    with_remote_execution_policy,
)
from agent_flow.workflows.perf_optimize.state import STAGE_OPTIMIZER, WorkflowState
from agent_flow.workflows.perf_optimize.workflow import PerfOptimizeWorkflow

_ROLES = (
    "benchmarker",
    "projector",
    "analyzer",
    "optimizer",
    "evaluator",
    "integrator",
    "qa",
    "reporter",
)


def _context(**locations: str) -> RemoteExecutionContext:
    return RemoteExecutionContext(
        remote_host="aga-300",
        control_workspace="/local/run",
        control_cwd="/local/run/rounds/round_1/item_1_x",
        control_task_path="/local/run/task.yaml",
        execution_workspace="/remote/run/workspace",
        execution_task_path="/remote/run/workspace/task.execution.yaml",
        execution_campaign_repo="/remote/run/repo",
        execution_command_cwd="/remote/run/workspace/worktrees/round_1/item_1_x",
        locations=locations,
    )


def _bind_remote_filesystems(
    workflow: PerfOptimizeWorkflow,
    workspace: Path,
    *,
    run_root: str = "/remote/run",
) -> ExecutionLayout:
    layout = ExecutionLayout.remote(
        control_workspace=workspace,
        remote_host="aga-300",
        run_root=run_root,
        campaign_repo="/remote/repo",
    )
    local_fs = fsspec.filesystem("file")
    workflow.execution_layout = layout
    workflow.run_fs = RunFileSystems(
        layout=layout,
        control_fs=local_fs,
        execution_fs=local_fs,
    )
    return layout


def test_remote_policy_is_appended_last_to_every_role() -> None:
    prompts = with_remote_execution_policy(DEFAULT_PROMPTS)
    for role in _ROLES:
        prompt = getattr(prompts, role)
        assert prompt.endswith(REMOTE_EXECUTION_POLICY)
        assert 'Bash("ssh <remote_host>' in prompt
        assert "Never run `sbatch`, `srun`, `squeue`, `sacct`, or `scancel` directly" in prompt


def test_remote_context_is_flat_authoritative_and_appended_last() -> None:
    context = _context(
        control_output_dir="/local/run/rounds/round_1/item_1_x/attempt_1",
        execution_worktree="/remote/run/workspace/worktrees/round_1/item_1_x",
        execution_artifact_dir="/remote/run/workspace/rounds/round_1/item_1_x/attempt_1",
    )
    prompt = append_remote_execution_context("Implement the item.\n", context)

    assert prompt.startswith("Implement the item.\n\n")
    assert "REMOTE_EXECUTION_CONTEXT:" in prompt
    assert "remote_host: aga-300" in prompt
    assert "execution_task_path: /remote/run/workspace/task.execution.yaml" in prompt
    assert "execution_artifact_dir:" in prompt
    assert prompt.rstrip().endswith(
        "execution_artifact_dir: /remote/run/workspace/rounds/round_1/item_1_x/attempt_1\n```"
    )


def test_remote_context_rejects_unqualified_role_locations() -> None:
    with pytest.raises(ValueError, match="must start with"):
        _context(output_dir="/ambiguous/path")


def test_remote_workflow_injects_policy_and_optimizer_turn_context(tmp_path: Path) -> None:
    workspace = tmp_path / "control"
    workflow = PerfOptimizeWorkflow(
        workspace=workspace,
        execution_run_root="/remote/run",
    )
    _bind_remote_filesystems(workflow, workspace)
    workflow.task_path.write_text(
        "checkpoint_path: /remote/model\ntrtllm_repo_path: /remote/repo\nsol:\n  enabled: false\n",
        encoding="utf-8",
    )
    state = WorkflowState(
        task_path=str(workflow.task_path),
        current_item_id="item-x",
        item_worktree_path="/remote/run/workspace/worktrees/round_1/item_1_item-x",
        item_branch="perf-optimize/item-x",
        campaign_git_branch="perf-optimize/campaign",
        campaign_git_base_commit="a" * 40,
        stage=STAGE_OPTIMIZER,
    )
    captured: list[str] = []
    original = workflow.optimizer
    workflow.optimizer = captured.append
    try:
        workflow._run_optimizer(state)
    finally:
        workflow.optimizer = original
        workflow.close()

    assert original.config.system_prompt.endswith(REMOTE_EXECUTION_POLICY)
    turn = captured[0]
    assert "Control attempt directory" in turn
    assert "Execution attempt directory" in turn
    assert "REMOTE_EXECUTION_CONTEXT:" in turn
    assert "control_task_path:" in turn
    assert "/remote/run/workspace/task.execution.yaml" in turn
    assert "execution_worktree: /remote/run/workspace/worktrees/round_1/item_1_item-x" in turn


def test_remote_raw_artifact_housekeeping_uses_execution_side(tmp_path: Path) -> None:
    workspace = tmp_path / "control"
    workflow = PerfOptimizeWorkflow(
        workspace=workspace,
        execution_run_root="/remote/run",
    )
    layout = _bind_remote_filesystems(
        workflow,
        workspace,
        run_root=str(tmp_path / "execution"),
    )
    assert workflow.run_fs is not None
    result_dir = "rounds/round_1/item_1_x/attempt_1"
    workflow.run_fs.write_text(
        f"{result_dir}/concurrency_8/openai-result.json",
        "{}\n",
        on="execution",
    )
    workflow.run_fs.write_text(
        f"{result_dir}/openai-result.json",
        "{}\n",
        on="execution",
    )
    workflow.run_fs.write_text(
        f"{result_dir}/serve.log",
        "keep\n",
        on="execution",
    )
    profile_dir = f"{result_dir}/profile"
    workflow.run_fs.write_text(
        f"{profile_dir}/capture.nsys-rep",
        "trace\n",
        on="execution",
    )
    state = WorkflowState(task_path=str(workflow.task_path))
    try:
        workflow._record_nsys_capture(state, profile_dir, on="execution")
        workflow._clear_stale_benchmark_results(result_dir)
    finally:
        workflow.close()

    execution_workspace = Path(layout.execution_workspace)
    assert state.last_nsys_dir == str(execution_workspace / profile_dir)
    assert not (execution_workspace / result_dir / "concurrency_8").exists()
    assert not (execution_workspace / result_dir / "openai-result.json").exists()
    assert (execution_workspace / result_dir / "serve.log").is_file()


def test_local_workflow_does_not_inject_remote_policy_or_context(tmp_path: Path) -> None:
    workflow = PerfOptimizeWorkflow(workspace=tmp_path / "control")
    try:
        assert not workflow.optimizer.config.system_prompt.endswith(REMOTE_EXECUTION_POLICY)
        assert workflow._with_remote_execution_context("local turn") == "local turn"
    finally:
        workflow.close()
