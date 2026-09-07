# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import fsspec
import pytest

from agent_flow.workflows.perf_optimize.execution import (
    EXECUTION_LAYOUT_SCHEMA_VERSION,
    ExecutionLayout,
    PerfOptimizeLayout,
    RunFileSystems,
    initialize_remote_execution,
    sync_benchmarker_results_to_control,
    sync_qa_results_to_control,
    sync_run_inputs_to_execution,
)


def test_layout_uses_one_relative_contract_for_both_sides(tmp_path: Path) -> None:
    layout = PerfOptimizeLayout()
    relative = layout.attempt_dir(2, 3, "cuda graph / capture", 4)
    assert relative == "rounds/round_2/item_3_cuda-graph-capture/attempt_4"

    execution = ExecutionLayout.remote(
        control_workspace=tmp_path / "control",
        remote_host="aga-300",
        run_root="/scratch/runs/perf-1",
        campaign_repo="/scratch/repos/trtllm",
    )
    assert execution.execution_workspace == "/scratch/runs/perf-1/workspace"
    assert ExecutionLayout.from_dict(execution.to_dict()) == execution


def test_run_filesystems_routes_both_sides_without_leaking_backends(tmp_path: Path) -> None:
    control = tmp_path / "control"
    execution = tmp_path / "execution"
    layout = ExecutionLayout(
        schema_version=EXECUTION_LAYOUT_SCHEMA_VERSION,
        mode="local",
        run_id="run-1",
        control_workspace=str(control),
        run_root=str(execution),
        execution_workspace=str(execution),
        campaign_repo=str(tmp_path / "repo"),
    )
    local_fs = fsspec.filesystem("file")
    run_fs = RunFileSystems(
        layout=layout,
        control_fs=local_fs,
        execution_fs=local_fs,
    )

    relative = "rounds/round_1/item_1_x/attempt_1"
    run_fs.makedirs(relative, on="both")
    assert (control / relative).is_dir()
    assert (execution / relative).is_dir()

    run_fs.write_text(f"{relative}/remote.txt", "result\n", on="execution")
    run_fs.copy_file(
        f"{relative}/remote.txt",
        f"{relative}/result.txt",
        source_side="execution",
        destination_side="control",
    )
    assert run_fs.read_text(f"{relative}/result.txt", on="control") == "result\n"

    with pytest.raises(ValueError, match="must be relative"):
        run_fs.exists("/absolute/path", on="control")
    with pytest.raises(ValueError, match="may not contain"):
        run_fs.makedirs("rounds/../escape", on="both")


def test_remote_initializer_uploads_only_run_inputs(tmp_path: Path) -> None:
    control = tmp_path / "control"
    execution = tmp_path / "execution"
    layout = ExecutionLayout(
        schema_version=EXECUTION_LAYOUT_SCHEMA_VERSION,
        mode="remote",
        run_id="run-1",
        control_workspace=str(control),
        run_root=str(execution.parent),
        execution_workspace=str(execution),
        campaign_repo="/remote/repo",
        remote_host="aga-300",
    )
    local_fs = fsspec.filesystem("file")
    run_fs = RunFileSystems(
        layout=layout,
        control_fs=local_fs,
        execution_fs=local_fs,
    )
    perf_layout = PerfOptimizeLayout()
    run_fs.write_text(perf_layout.task, "checkpoint_path: /model\n", on="control")
    run_fs.write_text(perf_layout.tuning_live, "x: 1\n", on="control")
    run_fs.write_text(perf_layout.tuning_accepted, "x: 1\n", on="control")

    initialize_remote_execution(layout, run_fs, perf_layout)
    sync_run_inputs_to_execution(layout, run_fs, perf_layout)

    assert (
        run_fs.read_text(perf_layout.execution_task, on="execution") == "checkpoint_path: /model\n"
    )
    assert not run_fs.exists(perf_layout.task, on="execution")
    assert run_fs.read_text(perf_layout.tuning_live, on="execution") == "x: 1\n"
    assert run_fs.exists(perf_layout.worktrees, on="execution")
    assert not run_fs.exists("progress.yaml", on="execution")

    run_fs.write_text("baseline/c1/result.json", "{}\n", on="execution")
    run_fs.write_text("baseline/serve.log", "raw\n", on="execution")
    run_fs.write_text("final_verification/result.json", "{}\n", on="execution")
    sync_benchmarker_results_to_control(layout, run_fs, perf_layout)
    sync_qa_results_to_control(layout, run_fs, perf_layout)
    assert run_fs.exists("baseline/c1/result.json", on="control")
    assert run_fs.exists("final_verification/result.json", on="control")
    assert not run_fs.exists("baseline/serve.log", on="control")
