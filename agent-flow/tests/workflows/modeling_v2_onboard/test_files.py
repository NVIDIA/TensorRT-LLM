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
    assert parse_planner_decision("decision: revised\nsummary: x\n").decision == "REVISED"
    with pytest.raises(HandoffError, match="REVISED, DONE, ABORT"):
        parse_planner_decision("decision: CONTINUE\n")


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
