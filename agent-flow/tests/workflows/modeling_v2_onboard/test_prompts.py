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
    build_primer_system_prompt,
    build_worker_system_prompt,
    initial_plan_prompt,
    planner_turn_prompt,
    primer_prompt,
    primer_version,
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
    assert "Up to **6** workers run at once, and the goal is that\n6 are always working" in prompt
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


def test_planner_turn_prompt_reports_what_changed(tmp_path: Path):
    def prompt(**overrides):
        args = dict(
            turn_file=tmp_path / "turn_3.yaml",
            plan=tmp_path / "plan.md",
            task_yaml=tmp_path / "task.yaml",
            results_dir=tmp_path / "task_results",
            finished=[("c", "FAILED", "no entry", tmp_path / "task_result_c.md")],
            states=[("a", "done"), ("c", "failed"), ("p", "running")],
            running=5,
            ready=1,
            max_parallel=8,
            finish=False,
            stalled=False,
            resumed=False,
            rejected=[],
        )
        return planner_turn_prompt(**{**args, **overrides})

    text = prompt()
    assert "- `c` FAILED: no entry" in text
    assert "5 running, 1 ready and waiting for a slot, 2 of 8 slots free" in text
    assert "held until you decide FINISH" in text
    assert f"End the turn by writing `{tmp_path / 'turn_3.yaml'}`" in text
    assert "restarted" not in text and "stalled" not in text and "discarded" not in text
    assert "held" not in prompt(finish=True)
    assert "The run is stalled" in prompt(stalled=True)
    assert "orchestrator restarted" in prompt(resumed=True)
    assert "- task 'a' is DONE" in prompt(rejected=["task 'a' is DONE"])


def test_human_messages_lead_the_turn_prompt(tmp_path: Path):
    text = planner_turn_prompt(
        turn_file=tmp_path / "turn_1.yaml",
        plan=tmp_path / "plan.md",
        task_yaml=tmp_path / "task.yaml",
        results_dir=tmp_path,
        finished=[("a", "DONE", "ok", tmp_path / "r.md")],
        states=[("a", "done")],
        running=0,
        ready=0,
        max_parallel=8,
        finish=False,
        stalled=False,
        resumed=False,
        rejected=[],
        feedback=["Remove fla-fix-test.", "Prioritize gsm8k."],
    )
    assert text.startswith("## Messages from the human — act on these first")
    assert (
        text.index("Remove fla-fix-test.")
        < text.index("Prioritize gsm8k.")
        < text.index("- `a` DONE")
    )


def test_planner_prompt_limits_filler_work_and_puts_humans_first():
    prompt = build_planner_system_prompt(max_parallel=8)
    assert "Do not add\nwork outside this list" in prompt
    assert "Human messages come first." in prompt


def test_planner_prompt_records_off_goal_bugs_instead_of_chasing_them():
    prompt = build_planner_system_prompt(max_parallel=8)
    assert "recorded, never chased" in prompt
    assert "append it to\n`issues.md` next to `plan.md`" in prompt


def test_planner_prompt_parks_unmerged_perf_work_on_side_once_the_goal_is_met():
    prompt = build_planner_system_prompt(max_parallel=8)
    assert "## Side tasks: performance work the run does not merge" in prompt
    assert "**When every goal in `task.yaml` is met**" in prompt
    assert (
        "**A human asking to keep optimizing, or to merge a side result, undoes\n  this.**"
        in prompt
    )
    assert "UNMERGED_OPTIMIZATIONS.md" in prompt


def test_planner_turn_prompt_says_when_the_run_has_delivered(tmp_path: Path):
    args = dict(
        turn_file=tmp_path / "turn_9.yaml",
        plan=tmp_path / "plan.md",
        task_yaml=tmp_path / "task.yaml",
        results_dir=tmp_path,
        finished=[],
        states=[("land", "done"), ("s", "running")],
        running=1,
        ready=0,
        max_parallel=8,
        finish=True,
        stalled=False,
        resumed=False,
        rejected=[],
    )
    assert "The final task is DONE" not in planner_turn_prompt(**args)
    text = planner_turn_prompt(**args, delivered=True, side_wait_minutes=30)
    assert "The final task is DONE" in text and "up to 30 minutes" in text


def test_the_planner_writes_its_plan_in_batches(tmp_path: Path):
    system = build_planner_system_prompt(max_parallel=8)
    assert "## Plan in batches" in system and "planning: false" in system
    first = initial_plan_prompt(
        task_yaml=tmp_path / "task.yaml",
        plan=tmp_path / "plan.md",
        repo=tmp_path,
        turn_file=tmp_path / "turn_0.yaml",
    )
    assert f"end the first batch by writing `{tmp_path / 'turn_0.yaml'}`" in first
    args = dict(
        turn_file=tmp_path / "turn_1.yaml",
        plan=tmp_path / "plan.md",
        task_yaml=tmp_path / "task.yaml",
        results_dir=tmp_path,
        finished=[],
        states=[("land", "pending")],
        running=0,
        ready=0,
        max_parallel=8,
        finish=False,
        stalled=False,
        resumed=False,
        rejected=[],
    )
    assert "You are still planning" not in planner_turn_prompt(**args)
    assert "You are still planning" in planner_turn_prompt(**args, planning=True)


def test_the_primer_only_reads_and_leaves_a_summary(tmp_path: Path):
    system = build_primer_system_prompt()
    assert "you only read" in system and "## ModelingV2: where things live" in system
    text = primer_prompt(
        repo=tmp_path,
        commit="abc123",
        task_yaml=tmp_path / "task.yaml",
        test_commands=tmp_path / "test_commands.md",
        summary=tmp_path / "knowledge" / "op.md",
    )
    assert "(commit `abc123`)" in text and "catalog/_op.py" in text
    assert f"Then write `{tmp_path / 'knowledge' / 'op.md'}`" in text
    assert primer_version() == primer_version() and len(primer_version()) == 12
