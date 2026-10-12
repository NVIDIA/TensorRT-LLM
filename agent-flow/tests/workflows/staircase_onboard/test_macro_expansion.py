# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `MACRO` — the outcome that says what the next work is.

`SMITH_DECISIONS` has always included `MACRO`, and the orchestrator has
always scored it as a failure. On the first real onboarding the largest
dispatch returned MACRO with an eleven-item decomposition, and the run
closed `smith-blocked` while holding exactly the queue extension it needed.

The fix is two-sided: the decomposition has to be machine-readable, and the
orchestrator has to act on it. These tests hold both sides.
"""

from __future__ import annotations

import yaml

from agent_flow.workflows.staircase_onboard.progress import (
    ProgressContext,
    collect_decision_file,
    decision_path,
    init_progress_file,
    latest_entry,
    parse_constituents,
)
from agent_flow.workflows.staircase_onboard.state import WorkflowState
from agent_flow.workflows.staircase_onboard.vocabulary import QueueItem
from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

# ------------------------------------------------------- parsing the list


def test_a_constituent_needs_a_target_and_gets_a_default_tier():
    rows = parse_constituents([{"target": "causal_conv1d_fwd"}])
    assert rows == [{"target": "causal_conv1d_fwd", "tier": 1}]


def test_a_constituent_without_a_target_is_dropped():
    """It cannot be dispatched, and guessing a name is worse than dropping it."""
    assert parse_constituents([{"tier": 2, "note": "something"}]) == []


def test_a_non_list_decomposition_is_not_a_crash():
    """A smith writing YAML by hand should not take the run down with it."""
    assert parse_constituents("conv, then the delta rule") == []
    assert parse_constituents(None) == []


def test_optional_fields_survive_and_unknown_ones_do_not():
    rows = parse_constituents(
        [{"target": "x", "tier": 3, "unit": "mtp", "surface": "P>1", "colour": "red"}]
    )
    assert rows[0] == {"target": "x", "tier": 3, "unit": "mtp", "surface": "P>1"}


def test_a_bad_tier_falls_back_rather_than_raising():
    assert parse_constituents([{"target": "x", "tier": "two"}])[0]["tier"] == 1


def test_constituents_round_trip_through_a_decision_file(tmp_path):
    progress = tmp_path / "progress.yaml"
    init_progress_file(progress)
    ctx = ProgressContext(path=progress)
    path = decision_path(tmp_path, "smith", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "summary": "registry lookup, not an op",
                "decision": "MACRO",
                "constituents": [
                    {"target": "causal_conv1d_fwd", "tier": 1},
                    {"target": "chunk_gated_delta_rule", "tier": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert collect_decision_file(ctx, tmp_path, "smith", 1) == "MACRO"
    stored = latest_entry(progress, "smith")["constituents"]
    assert [c["target"] for c in stored] == ["causal_conv1d_fwd", "chunk_gated_delta_rule"]


# ------------------------------------------------ the orchestrator's action


def _workflow(tmp_path, *, features):
    workflow = StaircaseOnboardWorkflow(tmp_path / "ws")
    workflow.task_data = {"features": features}
    init_progress_file(workflow.progress_path)
    return workflow


def _record_macro(workflow, children):
    ctx = ProgressContext(path=workflow.progress_path)
    ctx.append(
        "smith",
        {"iteration": 1, "agent": "smith", "decision": "MACRO", "constituents": children},
    )


def test_constituents_are_appended_to_the_queue(tmp_path):
    workflow = _workflow(tmp_path, features=[])
    state = WorkflowState(task_path="t", queue=[{"n": 1, "target": "gdn", "status": "running"}])
    _record_macro(workflow, [{"target": "conv_fwd", "tier": 1}, {"target": "delta", "tier": 2}])

    added = workflow._enqueue_constituents(
        state, QueueItem(n=1, mode="onboard", target="gdn"), None
    )

    assert added == 2
    assert [row["target"] for row in state.queue[1:]] == ["conv_fwd", "delta"]
    assert [row["n"] for row in state.queue[1:]] == [2, 3]
    assert state.queue[2]["tier"] == 2
    # Provenance, so a later reader knows why this dispatch exists.
    assert "constituent of dispatch 1" in state.queue[1]["note"]


def test_constituents_for_a_later_unit_are_deferred(tmp_path):
    """Core must be green before the first feature increment.

    Measured: of one macro's eleven constituents, four were MTP-only.
    Pulling them into the core batch would gate core against vocabulary it
    does not use.
    """
    workflow = _workflow(tmp_path, features=["mtp"])
    state = WorkflowState(task_path="t", unit_index=0, queue=[{"n": 1, "target": "gdn"}])
    _record_macro(
        workflow,
        [
            {"target": "conv_fwd", "tier": 1, "unit": "core"},
            {"target": "recurrent_update", "tier": 2, "unit": "mtp"},
        ],
    )

    added = workflow._enqueue_constituents(
        state, QueueItem(n=1, mode="onboard", target="gdn"), None
    )

    assert added == 1
    assert [row["target"] for row in state.queue[1:]] == ["conv_fwd"]


def test_a_later_unit_accepts_its_own_constituents(tmp_path):
    workflow = _workflow(tmp_path, features=["mtp"])
    state = WorkflowState(task_path="t", unit_index=1, queue=[{"n": 1, "target": "gdn"}])
    _record_macro(workflow, [{"target": "recurrent_update", "tier": 2, "unit": "mtp"}])

    assert (
        workflow._enqueue_constituents(state, QueueItem(n=1, mode="onboard", target="gdn"), None)
        == 1
    )


def test_constituents_of_a_feature_only_row_carry_its_unit(tmp_path):
    """A feature-only row is dispatched in the smith phase, so its pieces are too.

    Under its tag, though: a piece is needed no earlier than the row it
    decomposes, and inheriting the tag is what makes a failure among the
    pieces defer to that unit rather than block core.
    """
    workflow = _workflow(tmp_path, features=["mtp"])
    state = WorkflowState(
        task_path="t",
        unit_index=0,
        queue=[{"n": 1, "mode": "onboard", "target": "gdn", "unit": "mtp"}],
    )
    _record_macro(
        workflow,
        [{"target": "conv_fwd", "tier": 1}, {"target": "delta", "tier": 2, "unit": "core"}],
    )

    added = workflow._enqueue_constituents(
        state, QueueItem(n=1, mode="onboard", target="gdn", unit="mtp"), None
    )

    assert added == 2
    assert [row["unit"] for row in state.queue[1:]] == ["mtp", "mtp"]


def test_a_target_already_queued_is_not_dispatched_twice(tmp_path):
    """Two macros naming a shared piece, or a resume, must not duplicate it."""
    workflow = _workflow(tmp_path, features=[])
    state = WorkflowState(task_path="t", queue=[{"n": 1, "target": "conv_fwd"}])
    _record_macro(workflow, [{"target": "conv_fwd", "tier": 1}])

    assert (
        workflow._enqueue_constituents(state, QueueItem(n=1, mode="onboard", target="gdn"), None)
        == 0
    )


def test_a_macro_with_no_usable_constituents_adds_nothing(tmp_path):
    """Then the caller marks it failed — which is the old behaviour, correctly."""
    workflow = _workflow(tmp_path, features=[])
    state = WorkflowState(task_path="t", queue=[{"n": 1, "target": "gdn"}])
    _record_macro(workflow, [{"tier": 1, "note": "no target"}])

    assert (
        workflow._enqueue_constituents(state, QueueItem(n=1, mode="onboard", target="gdn"), None)
        == 0
    )


# ------------------------------------------------- the tier reaches the agent


def test_a_writing_dispatch_is_told_it_is_writing(tmp_path):
    """The tier must reach the agent.

    Silently dropping this was a real bug: the variable was built and never
    interpolated, so every dispatch read as though it were wrapping.
    """
    from agent_flow.workflows.staircase_onboard.vocabulary import TIER_TRITON

    workflow = _workflow(tmp_path, features=[])
    workflow.task_data.update({"trtllm_repo_path": "/repo", "arch": "sm_103"})
    prompt = workflow._smith_prompt(
        QueueItem(n=1, mode="onboard", target="exotic_norm", tier=TIER_TRITON)
    )
    assert "tier 4" in prompt
    assert "you are writing it, not wrapping one" in prompt


def test_a_wrapping_dispatch_is_not_told_to_write(tmp_path):
    workflow = _workflow(tmp_path, features=[])
    workflow.task_data.update({"trtllm_repo_path": "/repo", "arch": "sm_103"})
    prompt = workflow._smith_prompt(QueueItem(n=1, mode="onboard", target="x", tier=1))
    assert "tier 1" in prompt
    assert "you are writing it" not in prompt
