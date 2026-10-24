# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The planner is told the scheduling rules the orchestrator applies, and how to plan for them."""

from __future__ import annotations

import yaml

from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import DEFAULT_PROMPTS
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g

PLANNER = DEFAULT_PROMPTS.planner
#: The prompt with its line breaks folded, so a test does not depend on where a sentence wraps.
TEXT = " ".join(PLANNER.split())


def test_the_planner_learns_what_runs_at_the_same_time():
    assert "whatever directories they share" in TEXT
    assert "name the very same file in `touches`" in TEXT
    assert "an `exclusive: true` task runs alone" in TEXT
    assert "native (C++) rebuild" in TEXT
    assert "A rebase that conflicts goes back to the same worker" in TEXT


def test_the_planner_writes_touches_at_file_granularity():
    assert "write it at file granularity" in TEXT
    assert "one op file, one catalog entry, its test" in TEXT
    assert "not `catalog/ssm/`" in TEXT


def test_the_planner_splits_width_from_feature():
    assert "two axes, width and feature" in TEXT
    assert "forks from `core` alone" in TEXT
    assert "not on `core-tp` or `core-dp`" in TEXT
    assert "An edge that only keeps two tasks apart" in TEXT


def test_the_planner_reserves_devices_rather_than_waiting_for_idle_ones():
    assert "`resources.gpus` is its reservation" in TEXT


def test_the_example_task_is_one_the_graph_reads():
    """The example in the prompt names files, a tp1 core, and `exclusive`."""
    block = PLANNER.split("```yaml\n", 1)[1].split("```", 1)[0]
    raw = yaml.safe_load(block)["tasks"][0]
    assert raw["exclusive"] is False
    task = g.Task.from_dict(raw)
    assert task.id == "core" and task.gpus == 1
    assert all(t.endswith(".py") for t in task.touches)
    assert task.satisfies == ["core:boot@tp1", "core:accuracy[gsm8k]@tp1"]


def test_the_planner_keeps_correctness_off_the_tune_campaigns_path():
    assert "Correctness does not wait for the campaign." in TEXT
    flat = " ".join(TEXT.split())
    assert "A `tune` task is background work" in flat
    assert "no task depends on the `tune` task, and none integrates its branch" in flat
    assert "tells the campaign to wrap up and delivers the run" in flat
    assert "it ranks after the feature work whatever its estimate" in flat
    assert "Integrating tune's best branch" not in flat
    assert "start the `tune` task as soon as the core is correct" not in flat


def test_the_planner_learns_what_starts_without_it():
    assert "A new run starts with the survey (task `survey`), beside your first turn" in TEXT
    assert "one kernel task per vocabulary queue row, `k<row>-<entry>`" in TEXT
    assert "they start at once" in TEXT
    assert "nothing waits for it, and its words reach your next turn" in TEXT
    assert "no kernel or engine task depends on it" in TEXT


def test_the_first_graph_starts_the_work_that_needs_no_queue_row():
    assert (
        "Your first graph starts, beside the survey, all the work that needs no queue row" in TEXT
    )
    assert "Only what calls a catalog entry waits for the kernel tasks" in TEXT


def test_a_tune_task_holds_no_gate_or_work_devices_when_perf_sessions_run_its_jobs():
    assert "(`perf.spools`)" in TEXT and "give it `resources.gpus: 0`" in TEXT
    assert "Without perf sessions, give it the devices one measurement needs" in TEXT


def test_the_planner_learns_the_check_task():
    assert "- `check`: contract checks and nothing else" in TEXT
    assert "the orchestrator sends its checks to the check pool itself, with no agent turn" in TEXT
    assert "only a failure reaches you" in TEXT
    assert "a `check` task names contract checks in `satisfies`, an empty `touches`" in TEXT


def test_the_planner_learns_soft_dependencies():
    block = PLANNER.split("```yaml\n", 1)[1].split("```", 1)[0]
    assert yaml.safe_load(block)["tasks"][0]["depends_on_soft"] == []
    assert "The dependent starts once the other has started" in TEXT
    assert "its integration waits for the other to land or end" in TEXT
    assert "Use `depends_on` where the dependent writes against what the other produces" in TEXT


def test_the_planner_keeps_one_regression_review_open_instead_of_one_per_change():
    assert "Cross-model regression reviews are batched." in TEXT
    assert "up to 30 minutes after the batch's first landing" in TEXT
    assert "keep one regression review open (not started)" in TEXT
    assert "instead of writing a review per change" in TEXT
    assert "The run is not delivered while a landed change to shared code" in TEXT


def test_reviews_verify_rows_and_check_tasks_reserve_no_devices():
    assert "A review, a `verify` kernel row and a `check` task hold no reservation" in TEXT
    critic = " ".join(DEFAULT_PROMPTS.plan_reviewer.split())
    assert "## When the request names a task graph" in critic
    assert "`depends_on_soft` starts a task beside the one it names" in critic


def test_core_reaches_kernel_rows_through_soft_edges():
    assert "`core` reaches the rows it calls through `depends_on_soft`" in TEXT
    assert "Keep `depends_on` for what it is written on" in TEXT
    assert "`core` depends on the rows it calls" not in TEXT


def test_the_first_turn_releases_what_can_start_and_leaves_the_rest():
    assert "Keep that first turn short" in TEXT
    assert "leave the rest of the graph to the turn after the survey lands" in TEXT


def test_a_width_member_with_no_known_change_is_a_check_first():
    assert "A width or parallel member that no known code change separates" in TEXT
    assert "is a `check` task first" in TEXT
    assert "depends only on the width it runs at" in TEXT


def test_failed_and_blocked_events_are_answered_first_in_a_small_revision():
    assert "Events that report a task FAILED or BLOCKED come first" in TEXT
    assert "in a small revision of their own" in TEXT


def test_a_shared_code_task_leaves_the_cross_model_regression_to_the_review():
    assert "its goal does not ask it to show no regression on other models" in TEXT
