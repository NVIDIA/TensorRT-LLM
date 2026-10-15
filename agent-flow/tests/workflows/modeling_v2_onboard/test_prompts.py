# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for prompt composition."""

from __future__ import annotations

from pathlib import Path

from agent_flow.workflows.modeling_v2_onboard.plan import TAGS
from agent_flow.workflows.modeling_v2_onboard.prompts import (
    TAG_PROMPTS,
    TAG_SUMMARIES,
    WorkerTask,
    build_planner_system_prompt,
    build_worker_system_prompt,
    worker_turn_prompt,
)
from agent_flow.workflows.modeling_v2_onboard.prompts._common import SLURM_ENVIRONMENT


def test_every_tag_has_a_prompt_and_a_summary():
    assert set(TAG_PROMPTS) == set(TAGS) == set(TAG_SUMMARIES)


def test_worker_prompt_carries_exactly_its_tags_in_order():
    prompt = build_worker_system_prompt(("merge", "target"), slurm=False)
    merge = prompt.index("## Tag `merge`")
    target = prompt.index("## Tag `target`")
    assert merge < target
    for other in ("new_op", "module_accuracy", "e2e_accuracy", "perf"):
        assert f"## Tag `{other}`" not in prompt


def test_slurm_block_only_when_the_task_uses_slurm():
    heading = SLURM_ENVIRONMENT.splitlines()[0]
    assert heading not in build_worker_system_prompt(("new_op",), slurm=False)
    assert heading in build_worker_system_prompt(("new_op",), slurm=True)


def test_planner_prompt_lists_every_tag():
    prompt = build_planner_system_prompt(max_parallel=8)
    for tag in TAGS:
        assert f"| `{tag}` |" in prompt


def test_planner_prompt_states_the_parallelism():
    prompt = build_planner_system_prompt(max_parallel=6)
    assert "Up to **6** tasks run at once" in prompt
    assert "keep every worker busy" in prompt.lower()


def test_planner_example_plan_validates():
    from agent_flow.workflows.modeling_v2_onboard.plan import parse_plan
    from agent_flow.workflows.modeling_v2_onboard.prompts import PLAN_EXAMPLE

    spec = parse_plan(PLAN_EXAMPLE)
    assert spec.sink_id() == "land"
    assert spec.tags["perf-moe-gemm"] == ("perf",)


def test_worker_turn_prompt_names_every_path(tmp_path: Path):
    task = WorkerTask(
        task_id="op-a",
        tags=("new_op",),
        worktree=tmp_path / "wt",
        is_final=False,
        task_yaml=tmp_path / "task.yaml",
        plan=tmp_path / "plan.md",
        result=tmp_path / "task_result_op-a.md",
        upstream_results=(("dep", tmp_path / "task_result_dep.md"),),
        notes_dir=tmp_path / "nodes" / "op-a",
        jobs=tmp_path / "nodes" / "op-a" / "jobs.json",
        previous_attempts=(tmp_path / "task_result_op-a.attempt1.md",),
    )
    prompt = worker_turn_prompt(task)
    for path in (
        task.worktree,
        task.plan,
        task.result,
        task.jobs,
        task.upstream_results[0][1],
        task.previous_attempts[0],
    ):
        assert str(path) in prompt
    assert "### op-a" in prompt
    assert "final task" not in prompt
    assert "final task" in worker_turn_prompt(WorkerTask(**{**task.__dict__, "is_final": True}))
