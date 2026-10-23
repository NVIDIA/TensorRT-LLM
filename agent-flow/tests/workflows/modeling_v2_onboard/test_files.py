# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the task-result and planner-decision handoff files."""

from __future__ import annotations

import pytest

from agent_flow.workflows.modeling_v2_onboard.files import (
    HandoffError,
    Layout,
    parse_planner_decision,
    parse_task_result,
    read_handoff,
)


def test_task_result_front_matter_is_parsed():
    result = parse_task_result("---\nstatus: done\nsummary: entry landed\n---\n\n## Evidence\n")
    assert (result.status, result.summary) == ("DONE", "entry landed")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("status: DONE\n", "must start with"),
        ("---\nstatus: DONE\n", "no closing"),
        ("---\nstatus: MAYBE\n---\n", "`status` must be one of DONE, FAILED"),
        ("---\n- a list\n---\n", "expected a YAML mapping"),
        ("---\nstatus: [\n---\n", "invalid YAML"),
    ],
)
def test_malformed_task_results_are_rejected(text, message):
    with pytest.raises(HandoffError, match=message):
        parse_task_result(text)


def test_planner_decision_is_validated():
    decision = parse_planner_decision("decision: finish\nsummary: x\nretry: [c, c]\ncancel: [p]\n")
    assert (decision.decision, decision.retry, decision.cancel) == ("FINISH", ("c",), ("p",))
    assert parse_planner_decision("decision: CONTINUE\n").retry == ()
    with pytest.raises(HandoffError, match="CONTINUE, FINISH, ABORT"):
        parse_planner_decision("decision: REVISED\n")
    with pytest.raises(HandoffError, match="`cancel` must be a list"):
        parse_planner_decision("decision: CONTINUE\ncancel: p\n")
    assert not parse_planner_decision("decision: CONTINUE\n").planning
    assert parse_planner_decision("decision: CONTINUE\nplanning: true\n").planning
    with pytest.raises(HandoffError, match="`planning` must be true or false"):
        parse_planner_decision("decision: CONTINUE\nplanning: later\n")
    assert parse_planner_decision("decision: CONTINUE\n").gpu_nodes is None
    assert parse_planner_decision("decision: CONTINUE\ngpu_nodes: 6\n").gpu_nodes == 6
    with pytest.raises(HandoffError, match="`gpu_nodes` must be a positive integer"):
        parse_planner_decision("decision: CONTINUE\ngpu_nodes: 0\n")


def test_missing_handoff_file_is_an_error(tmp_path):
    with pytest.raises(HandoffError, match="was not written"):
        read_handoff(tmp_path / "absent.md", parse_task_result)


def test_previous_attempts_are_ordered_numerically(tmp_path):
    layout = Layout(tmp_path)
    layout.results_dir.mkdir()
    for attempt in (10, 2, 1):
        layout.attempt_path("t.1", attempt).write_text("x")
    assert [p.name for p in layout.previous_attempts("t.1")] == [
        "task_result_t.1.attempt1.md",
        "task_result_t.1.attempt2.md",
        "task_result_t.1.attempt10.md",
    ]
    assert layout.result_path("t.1").name == "task_result_t.1.md"


def test_inbox_messages_are_read_oldest_first(tmp_path):
    from agent_flow.workflows.modeling_v2_onboard.files import post_feedback, read_inbox

    layout = Layout(tmp_path)
    assert read_inbox(layout) == []
    first = post_feedback(tmp_path, "  first  ")
    second = layout.inbox_dir / "99999999T999999-1.md"
    second.write_text("second\n")
    assert read_inbox(layout) == [(first, "first\n"), (second, "second\n")]
    with pytest.raises(ValueError, match="empty"):
        post_feedback(tmp_path, "\n")
