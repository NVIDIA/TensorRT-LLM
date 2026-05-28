"""Tests for the agent-team progress.yaml tool flow."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import yaml

from agent_flow.workflows.agent_team import progress as progress_module
from agent_flow.workflows.agent_team import workflow as _workflow_module
from agent_flow.workflows.agent_team.state import (STAGE_CODER, WorkflowState,
                                                   save_state)


def _load_progress_module():
    return progress_module


def _load_workflow_module():
    return _workflow_module


# --------------------------------------------------------------------- module


def test_read_progress_handles_missing_empty_and_seeded(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"

    empty = {"plan_stage": [], "build_stage": [], "human_feedback": []}
    assert progress.read_progress(path) == empty  # missing

    path.write_text("", encoding="utf-8")
    assert progress.read_progress(path) == empty  # empty

    path.write_text("plan_stage: []\nbuild_stage: []\n", encoding="utf-8")
    assert progress.read_progress(path) == empty  # explicit empty mapping

    path.write_text(
        "plan_stage:\n"
        "  - iteration: 1\n"
        "    agent: plan_drafter\n"
        "    summary: hi\n"
        "build_stage: []\n",
        encoding="utf-8",
    )
    assert progress.read_progress(path) == {
        "plan_stage": [
            {
                "iteration": 1,
                "agent": "plan_drafter",
                "summary": "hi"
            },
        ],
        "build_stage": [],
        "human_feedback": [],
    }


def test_read_progress_rejects_legacy_list(tmp_path):
    """The legacy flat-list format must surface as an explicit error rather
    than silently mis-routing entries."""
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    path.write_text("- iteration: 1\n  agent: plan_drafter\n", encoding="utf-8")
    with pytest.raises(ValueError, match="plan_stage"):
        progress.read_progress(path)


def test_read_progress_rejects_non_list_stage(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    path.write_text("plan_stage: oops\nbuild_stage: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="plan_stage"):
        progress.read_progress(path)


def test_init_progress_file_writes_all_top_level_keys(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.init_progress_file(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data == {
        "plan_stage": [],
        "build_stage": [],
        "human_feedback": [],
    }
    # Top-level key order is fixed so diffs stay stable across runs.
    assert list(data.keys()) == ["plan_stage", "build_stage", "human_feedback"]


def _call(handler, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(handler(args))


def test_tool_handlers_route_entries_to_their_stage(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.init_progress_file(path)

    ctx = progress.ProgressContext(path=path, current_iteration=1)
    tools = progress.build_progress_tools(ctx)

    # PlanDrafter draft turn — DRAFT_READY.
    _call(tools["plan_drafter"][0].handler, {
        "summary": "initial plan draft",
        "decision": "DRAFT_READY",
    })
    # PlanReviewer APPROVE.
    _call(tools["plan_reviewer"][0].handler, {
        "summary": "plan covers task.yaml",
        "decision": "APPROVE",
    })
    # PlanDrafter human turn — HUMAN_APPROVED.
    _call(tools["plan_drafter"][0].handler, {
        "summary": "human approved",
        "decision": "HUMAN_APPROVED",
    })

    _call(tools["coder"][0].handler, {"summary": "implemented v1"})
    _call(tools["reviewer"][0].handler, {
        "summary": "looks right",
        "decision": "APPROVE",
    })
    _call(
        tools["qa"][0].handler, {
            "summary": "per-criterion scores...",
            "decision": "APPROVE",
            "weighted_score": 9.2,
        })

    data = progress.read_progress(path)

    # Plan and build entries land in their own stage.
    assert [e["agent"] for e in data["plan_stage"]] == [
        "plan_drafter",
        "plan_reviewer",
        "plan_drafter",
    ]
    assert [e["agent"] for e in data["build_stage"]] == [
        "coder",
        "reviewer",
        "qa",
    ]
    assert all(e["iteration"] == 1
               for e in data["plan_stage"] + data["build_stage"])

    # Role-specific fields are present only where they belong.
    plan = data["plan_stage"]
    build = data["build_stage"]
    assert plan[0]["decision"] == "DRAFT_READY"
    assert "weighted_score" not in plan[0]
    assert plan[1]["decision"] == "APPROVE"  # plan_reviewer
    assert "weighted_score" not in plan[1]
    assert plan[2]["decision"] == "HUMAN_APPROVED"
    assert "decision" not in build[0]  # coder has no decision
    assert "weighted_score" not in build[0]
    assert build[1]["decision"] == "APPROVE"
    assert "weighted_score" not in build[1]
    assert build[2]["decision"] == "APPROVE"
    assert build[2]["weighted_score"] == 9.2

    # Timestamp is set by the tool, ISO 8601-ish without microseconds.
    for e in plan + build:
        assert "T" in e["timestamp"]
        assert len(e["timestamp"]) == len("2026-04-23T14:32:11")


def test_plan_drafter_tool_handler_validates_decision(tmp_path):
    """append_plan_drafter_progress requires a decision from the enum."""
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.init_progress_file(path)
    ctx = progress.ProgressContext(path=path, current_iteration=2)
    tools = progress.build_progress_tools(ctx)

    # Each enum value writes the matching decision.
    for decision in ("DRAFT_READY", "POLISHING", "HUMAN_APPROVED"):
        progress.init_progress_file(path)
        _call(tools["plan_drafter"][0].handler, {
            "summary": f"r {decision}",
            "decision": decision,
        })
        plan = progress.read_progress(path)["plan_stage"]
        assert plan[0]["agent"] == "plan_drafter"
        assert plan[0]["decision"] == decision


def test_plan_reviewer_tool_handler_records_reject_decision(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.init_progress_file(path)
    ctx = progress.ProgressContext(path=path, current_iteration=1)
    tools = progress.build_progress_tools(ctx)

    _call(tools["plan_reviewer"][0].handler, {
        "summary": "plan misses streaming requirement",
        "decision": "REJECT",
    })

    plan = progress.read_progress(path)["plan_stage"]
    assert plan[0]["agent"] == "plan_reviewer"
    assert plan[0]["decision"] == "REJECT"


def test_reviewer_tool_handler_records_reject_decision(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.init_progress_file(path)
    ctx = progress.ProgressContext(path=path, current_iteration=1)
    tools = progress.build_progress_tools(ctx)

    _call(tools["reviewer"][0].handler, {
        "summary": "missing ValueError for negative n",
        "decision": "REJECT",
    })

    build = progress.read_progress(path)["build_stage"]
    assert build[0]["agent"] == "reviewer"
    assert build[0]["decision"] == "REJECT"


def test_qa_tool_handler_records_reject_decision(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.init_progress_file(path)
    ctx = progress.ProgressContext(path=path, current_iteration=2)
    tools = progress.build_progress_tools(ctx)

    _call(
        tools["qa"][0].handler, {
            "summary": "2/4 tests fail; functionality incomplete",
            "decision": "REJECT",
            "weighted_score": 4.5,
        })

    build = progress.read_progress(path)["build_stage"]
    assert build[0]["agent"] == "qa"
    assert build[0]["decision"] == "REJECT"
    assert build[0]["weighted_score"] == 4.5


def test_find_entries(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.write_progress(
        path, {
            "plan_stage": [
                {
                    "iteration": 1,
                    "agent": "plan_drafter",
                    "summary": "draft 1",
                    "decision": "DRAFT_READY",
                },
                {
                    "iteration": 1,
                    "agent": "plan_reviewer",
                    "summary": "reject",
                    "decision": "REJECT",
                },
                {
                    "iteration": 2,
                    "agent": "plan_drafter",
                    "summary": "draft 2",
                    "decision": "DRAFT_READY",
                },
            ],
            "build_stage": [
                {
                    "iteration": 1,
                    "agent": "coder",
                    "summary": "c1"
                },
                {
                    "iteration": 1,
                    "agent": "reviewer",
                    "summary": "r1",
                    "decision": "APPROVE"
                },
                {
                    "iteration": 1,
                    "agent": "qa",
                    "summary": "q1",
                    "decision": "REJECT",
                    "weighted_score": 6.0
                },
                {
                    "iteration": 2,
                    "agent": "coder",
                    "summary": "c2"
                },
                {
                    "iteration": 2,
                    "agent": "reviewer",
                    "summary": "r2",
                    "decision": "APPROVE"
                },
                {
                    "iteration": 2,
                    "agent": "qa",
                    "summary": "q2",
                    "decision": "APPROVE",
                    "weighted_score": 9.0
                },
            ],
        })

    # No filters → all build entries (within the requested stage).
    assert len(progress.find_entries(path, stage="build_stage")) == 6
    assert len(progress.find_entries(path, stage="plan_stage")) == 3

    # last_iterations=1 in the build stage → just iteration 2.
    assert [
        e["agent"] for e in progress.find_entries(
            path, stage="build_stage", last_iterations=1)
    ] == ["coder", "reviewer", "qa"]

    # last_iterations=2 in the build stage → iterations 1 and 2.
    assert [(e["iteration"], e["agent"]) for e in progress.find_entries(
        path, stage="build_stage", last_iterations=2)] == [
            (1, "coder"),
            (1, "reviewer"),
            (1, "qa"),
            (2, "coder"),
            (2, "reviewer"),
            (2, "qa"),
        ]

    # Agent filter only (all iterations).
    coders = progress.find_entries(path, stage="build_stage", agent="coder")
    assert [e["iteration"] for e in coders] == [1, 2]

    # Combined filters.
    qa_entries = progress.find_entries(path,
                                       stage="build_stage",
                                       agent="qa",
                                       last_iterations=2)
    assert [e["weighted_score"] for e in qa_entries] == [6.0, 9.0]
    assert [e["decision"] for e in qa_entries] == ["REJECT", "APPROVE"]

    # Plan-stage agent filter on plan stage.
    drafts = progress.find_entries(path,
                                   stage="plan_stage",
                                   agent="plan_drafter")
    assert [e["iteration"] for e in drafts] == [1, 2]

    # Cross-stage agent filter returns empty (e.g. reviewer is build_stage).
    assert progress.find_entries(path, stage="plan_stage",
                                 agent="reviewer") == []

    # Empty file → empty list regardless of filters.
    empty = tmp_path / "empty.yaml"
    progress.init_progress_file(empty)
    assert progress.find_entries(empty, stage="plan_stage") == []
    assert progress.find_entries(empty, stage="build_stage",
                                 last_iterations=5) == []

    # Invalid last_iterations.
    with pytest.raises(ValueError):
        progress.find_entries(path, stage="build_stage", last_iterations=0)

    # Unknown stage.
    with pytest.raises(ValueError, match="unknown stage"):
        progress.find_entries(path, stage="not_a_stage")


def test_find_entries_uses_stage_local_iteration_cutoff(tmp_path):
    """Plan and build phases keep independent iteration counters. A
    build-stage coder entry at iteration 1 must not be hidden by plan-stage
    entries that have already advanced to iteration 2 — the cutoff is
    local to the stage being queried.
    """
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.write_progress(
        path, {
            "plan_stage": [
                {
                    "iteration": 1,
                    "agent": "plan_drafter",
                    "summary": "draft",
                    "decision": "DRAFT_READY",
                },
                {
                    "iteration": 1,
                    "agent": "plan_reviewer",
                    "summary": "reject",
                    "decision": "REJECT",
                },
                {
                    "iteration": 2,
                    "agent": "plan_drafter",
                    "summary": "redraft",
                    "decision": "DRAFT_READY",
                },
                {
                    "iteration": 2,
                    "agent": "plan_reviewer",
                    "summary": "approve",
                    "decision": "APPROVE",
                },
                {
                    "iteration": 2,
                    "agent": "plan_drafter",
                    "summary": "human approved",
                    "decision": "HUMAN_APPROVED",
                },
            ],
            "build_stage": [
                {
                    "iteration": 1,
                    "agent": "coder",
                    "summary": "implemented",
                },
            ],
        })

    coder_latest = progress.find_entries(path,
                                         stage="build_stage",
                                         agent="coder",
                                         last_iterations=1)
    assert len(coder_latest) == 1
    assert coder_latest[0]["agent"] == "coder"
    assert coder_latest[0]["iteration"] == 1


def test_read_latest_progress_tool_handler_is_stage_scoped(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.write_progress(
        path, {
            "plan_stage": [
                {
                    "iteration": 1,
                    "agent": "plan_drafter",
                    "timestamp": "p1",
                    "summary": "p1",
                    "decision": "DRAFT_READY",
                },
                {
                    "iteration": 1,
                    "agent": "plan_reviewer",
                    "timestamp": "p2",
                    "summary": "p2",
                    "decision": "APPROVE",
                },
            ],
            "build_stage": [
                {
                    "iteration": 1,
                    "agent": "coder",
                    "timestamp": "t1",
                    "summary": "c1",
                },
                {
                    "iteration": 1,
                    "agent": "reviewer",
                    "timestamp": "t2",
                    "summary": "r1",
                    "decision": "APPROVE",
                },
                {
                    "iteration": 2,
                    "agent": "coder",
                    "timestamp": "t3",
                    "summary": "c2",
                },
            ],
        })
    ctx = progress.ProgressContext(path=path, current_iteration=2)
    tools = progress.build_progress_tools(ctx)

    # Plan-phase agents have their append tool + read_latest_progress.
    # Build-phase agents additionally get read_human_feedback so they
    # can pick up user-injected guidance.
    for role in ("plan_drafter", "plan_reviewer"):
        assert len(tools[role]) == 2
        names = [t.name for t in tools[role]]
        assert "read_latest_progress" in names
        assert "read_human_feedback" not in names
    for role in ("coder", "reviewer"):
        assert len(tools[role]) == 3
        names = [t.name for t in tools[role]]
        assert "read_latest_progress" in names
        assert "read_human_feedback" in names
    # QA: append_qa_progress + read_human_feedback only. The rest of
    # progress.yaml is intentionally off-limits.
    assert len(tools["qa"]) == 2
    qa_names = sorted(t.name for t in tools["qa"])
    assert qa_names == ["append_qa_progress", "read_human_feedback"]

    # The read tool's `agent` enum is restricted to the caller's stage.
    plan_read = next(t for t in tools["plan_drafter"]
                     if t.name == "read_latest_progress")
    build_read = next(t for t in tools["coder"]
                      if t.name == "read_latest_progress")
    plan_enum = plan_read.input_schema["properties"]["agent"]["enum"]
    build_enum = build_read.input_schema["properties"]["agent"]["enum"]
    assert sorted(plan_enum) == ["plan_drafter", "plan_reviewer"]
    assert sorted(build_enum) == ["coder", "qa", "reviewer"]

    # Build-side default: latest build_stage iteration only — the plan
    # entries are intentionally invisible.
    out = _call(build_read.handler, {})
    rendered = yaml.safe_load(out["content"][0]["text"])
    assert [e["agent"] for e in rendered] == ["coder"]
    assert rendered[0]["iteration"] == 2

    # Build-side: iterations=2, agent=reviewer → only iter-1 reviewer entry.
    out = _call(build_read.handler, {"iterations": 2, "agent": "reviewer"})
    rendered = yaml.safe_load(out["content"][0]["text"])
    assert rendered == [{
        "iteration": 1,
        "agent": "reviewer",
        "timestamp": "t2",
        "summary": "r1",
        "decision": "APPROVE",
    }]

    # Plan-side default: latest plan_stage iteration only.
    out = _call(plan_read.handler, {})
    rendered = yaml.safe_load(out["content"][0]["text"])
    # iter 1 is the latest plan iteration in this fixture.
    assert [e["agent"] for e in rendered] == ["plan_drafter", "plan_reviewer"]

    # Empty file gives a stage-aware human-readable stub.
    empty = tmp_path / "empty.yaml"
    progress.init_progress_file(empty)
    ctx2 = progress.ProgressContext(path=empty)
    plan_read2 = next(
        t for t in progress.build_progress_tools(ctx2)["plan_drafter"]
        if t.name == "read_latest_progress")
    out = _call(plan_read2.handler, {})
    assert "No plan_stage progress entries yet" in out["content"][0]["text"]


def test_append_human_feedback_stamps_metadata_and_appends(tmp_path):
    """``append_human_feedback`` adds one entry per call, never overwrites,
    and preserves prior plan/build entries."""
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.write_progress(
        path, {
            "plan_stage": [
                {
                    "iteration": 1,
                    "agent": "plan_drafter",
                    "summary": "x",
                    "decision": "DRAFT_READY",
                },
            ],
            "build_stage": [
                {
                    "iteration": 2,
                    "agent": "coder",
                    "summary": "c1",
                },
            ],
            "human_feedback": [],
        })

    entry = progress.append_human_feedback(
        path,
        summary="please add streaming",
        iteration=3,
        stage="build_stage",
    )
    assert entry["iteration"] == 3
    assert entry["stage"] == "build_stage"
    assert entry["summary"] == "please add streaming"
    assert "T" in entry["timestamp"]

    # Second call appends rather than overwrites.
    progress.append_human_feedback(
        path,
        summary="also handle errors",
        iteration=4,
        stage="build_stage",
    )

    data = progress.read_progress(path)
    # Plan and build entries are preserved untouched.
    assert len(data["plan_stage"]) == 1
    assert data["plan_stage"][0]["agent"] == "plan_drafter"
    assert len(data["build_stage"]) == 1
    assert data["build_stage"][0]["agent"] == "coder"
    # Both human entries are present, in order.
    assert [e["summary"] for e in data["human_feedback"]] == [
        "please add streaming",
        "also handle errors",
    ]
    assert [e["iteration"] for e in data["human_feedback"]] == [3, 4]
    assert all(e["stage"] == "build_stage" for e in data["human_feedback"])


def test_append_human_feedback_rejects_unknown_stage(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.init_progress_file(path)
    with pytest.raises(ValueError, match="unknown stage"):
        progress.append_human_feedback(
            path,
            summary="x",
            iteration=1,
            stage="human_feedback",
        )


def test_read_human_feedback_tool_returns_user_entries(tmp_path):
    """The build-phase ``read_human_feedback`` tool returns the user's
    feedback list as YAML text. The plan-phase agents do not get the tool."""
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.write_progress(
        path, {
            "plan_stage": [],
            "build_stage": [],
            "human_feedback": [
                {
                    "timestamp": "2026-05-13T10:00:00",
                    "iteration": 2,
                    "stage": "build_stage",
                    "summary": "add streaming output",
                },
                {
                    "timestamp": "2026-05-13T10:10:00",
                    "iteration": 3,
                    "stage": "build_stage",
                    "summary": "support negative integers",
                },
            ],
        })

    ctx = progress.ProgressContext(path=path, current_iteration=3)
    tools = progress.build_progress_tools(ctx)

    # Plan-phase agents are not granted the tool — feedback is a build-phase
    # signal in this design.
    for role in ("plan_drafter", "plan_reviewer"):
        assert all(t.name != "read_human_feedback" for t in tools[role])

    # Each build-phase agent has a read_human_feedback tool. Returned YAML
    # is structurally identical to the on-disk list.
    for role in ("coder", "reviewer", "qa"):
        feedback_tool = next(t for t in tools[role]
                             if t.name == "read_human_feedback")
        out = _call(feedback_tool.handler, {})
        rendered = yaml.safe_load(out["content"][0]["text"])
        assert rendered == progress.read_progress(path)["human_feedback"]


def test_read_human_feedback_tool_handles_empty_list(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.init_progress_file(path)
    ctx = progress.ProgressContext(path=path, current_iteration=1)
    tools = progress.build_progress_tools(ctx)
    feedback_tool = next(t for t in tools["coder"]
                         if t.name == "read_human_feedback")
    out = _call(feedback_tool.handler, {})
    assert "No human_feedback entries yet" in out["content"][0]["text"]


def test_read_human_feedback_logs_caller_and_count(tmp_path, monkeypatch):
    """The read-feedback tool emits a styled panel attributed to the
    calling agent so the user sees who read what."""
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.write_progress(
        path, {
            "plan_stage": [],
            "build_stage": [],
            "human_feedback": [
                {
                    "timestamp": "t1",
                    "iteration": 1,
                    "stage": "build_stage",
                    "summary": "fix the bug",
                },
            ],
        })

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(progress,
                        "print_layer_panel",
                        lambda layer, suffix, body, _console=None: calls.append(
                            {
                                "layer": layer,
                                "suffix": suffix,
                            }))

    ctx = progress.ProgressContext(path=path)
    tools = progress.build_progress_tools(ctx)
    reviewer_feedback = next(t for t in tools["reviewer"]
                             if t.name == "read_human_feedback")
    _call(reviewer_feedback.handler, {})

    assert calls
    assert calls[-1]["layer"] == "reviewer"
    assert "human_feedback" in calls[-1]["suffix"]
    assert "1 entry" in calls[-1]["suffix"]


def test_append_human_feedback_logs_panel(tmp_path, monkeypatch):
    """``append_human_feedback`` emits a styled orchestrator panel so the
    user can see what landed in progress.yaml."""
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.init_progress_file(path)

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(progress,
                        "print_layer_panel",
                        lambda layer, suffix, body, _console=None: calls.append(
                            {
                                "layer": layer,
                                "suffix": suffix,
                            }))

    progress.append_human_feedback(
        path,
        summary="please address X",
        iteration=2,
        stage="build_stage",
    )

    assert calls
    last = calls[-1]
    assert last["layer"] == "orchestrator"
    assert "human_feedback" in last["suffix"]
    assert "stage=build_stage" in last["suffix"]
    assert "iter 2" in last["suffix"]


def test_tool_handlers_log_write_and_read(tmp_path, monkeypatch):
    """Each tool handler should emit a styled panel with the relevant YAML."""
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.init_progress_file(path)

    calls: list[dict[str, Any]] = []

    def _capture(layer_name, title_suffix, body, extra=None):
        calls.append({
            "layer": layer_name,
            "suffix": title_suffix,
            "body": body,
        })

    # The progress module captured ``print_layer_panel`` at import time, so
    # patch the name in that module's namespace.
    monkeypatch.setattr(progress, "print_layer_panel", _capture)

    ctx = progress.ProgressContext(path=path, current_iteration=1)
    tools = progress.build_progress_tools(ctx)

    _call(tools["plan_drafter"][0].handler, {
        "summary": "d",
        "decision": "DRAFT_READY",
    })
    _call(tools["plan_reviewer"][0].handler, {
        "summary": "p",
        "decision": "APPROVE",
    })
    _call(tools["coder"][0].handler, {"summary": "c"})
    _call(tools["reviewer"][0].handler, {
        "summary": "r",
        "decision": "APPROVE",
    })
    _call(tools["qa"][0].handler, {
        "summary": "q",
        "decision": "APPROVE",
        "weighted_score": 8.0,
    })

    plan_drafter_read = next(t for t in tools["plan_drafter"]
                             if t.name == "read_latest_progress")
    coder_read = next(t for t in tools["coder"]
                      if t.name == "read_latest_progress")

    # PlanDrafter reads the PlanReviewer's entries → should be styled
    # PLAN_DRAFTER (the caller), not PLAN_REVIEWER (the filter).
    _call(plan_drafter_read.handler, {
        "iterations": 1,
        "agent": "plan_reviewer",
    })
    # Coder reads with no filter → CODER.
    _call(coder_read.handler, {})

    layers = [c["layer"] for c in calls]
    suffixes = [c["suffix"] for c in calls]

    # Five writes in agent-specific styles, then two reads.
    assert layers[:5] == [
        "plan_drafter",
        "plan_reviewer",
        "coder",
        "reviewer",
        "qa",
    ]
    assert all(s.startswith("system · wrote iter 1") for s in suffixes[:5])

    # Reads are attributed to the caller, and the suffix still records the
    # filter applied.
    assert layers[5] == "plan_drafter"
    assert "agent=plan_reviewer" in suffixes[5]
    assert layers[6] == "coder"
    assert "all agents" in suffixes[6]


def test_latest_entry_returns_most_recent_for_agent(tmp_path):
    progress = _load_progress_module()
    path = tmp_path / "progress.yaml"
    progress.init_progress_file(path)

    ctx = progress.ProgressContext(path=path, current_iteration=1)
    tools = progress.build_progress_tools(ctx)

    _call(tools["qa"][0].handler, {
        "summary": "s1",
        "decision": "REJECT",
        "weighted_score": 5,
    })
    ctx.current_iteration = 2
    _call(tools["qa"][0].handler, {
        "summary": "s2",
        "decision": "APPROVE",
        "weighted_score": 9.2,
    })

    latest = progress.latest_entry(path, "qa")
    assert latest is not None
    assert latest["iteration"] == 2
    assert latest["decision"] == "APPROVE"
    assert latest["weighted_score"] == 9.2
    assert progress.latest_entry(path, "plan_drafter") is None


# ------------------------------------------------------------------- workflow


class ScriptedAgent:
    """Fake AgentLayer that runs a user-supplied action and returns text."""

    def __init__(self, response_text: str, action):
        self.response_text = response_text
        self.action = action
        self.prompts: list[str] = []

    def __call__(self, content: str) -> str:
        self.prompts.append(content)
        self.action()
        return self.response_text

    def __exit__(self, *_exc) -> None:  # for AgentTeamWorkflow.close()
        pass


def _install_scripted_agents(workflow, reviewer_decisions, qa_decisions,
                             qa_scores):
    """Replace workflow's real AgentLayers with scripted fakes that call the
    real tool handlers — exercising the YAML path through the tool flow.

    ``reviewer_decisions`` feeds the reviewer's per-iteration decision in
    order; ``qa_decisions`` and ``qa_scores`` do the same for QA. The
    PlanDrafter is scripted to draft once (DRAFT_READY) then approve via
    the human stage (HUMAN_APPROVED); the PlanReviewer always APPROVEs.
    """
    tools = workflow._progress_tools

    def run_tool(agent_key: str, args: dict[str, Any]) -> None:
        asyncio.run(tools[agent_key][0].handler(args))

    reviewer_iter = iter(reviewer_decisions)
    qa_decision_iter = iter(qa_decisions)
    qa_score_iter = iter(qa_scores)

    plan_drafter_calls = {"count": 0}

    def plan_drafter_action():
        plan_drafter_calls["count"] += 1
        workflow.plan_path.write_text("# Plan\nDo X.\n", encoding="utf-8")
        # First call is the draft phase; subsequent calls are the human
        # phase, which approves immediately.
        decision = ("DRAFT_READY"
                    if plan_drafter_calls["count"] == 1 else "HUMAN_APPROVED")
        run_tool("plan_drafter", {
            "summary": f"plan {decision}",
            "decision": decision,
        })

    def plan_reviewer_action():
        run_tool("plan_reviewer", {
            "summary": "plan looks fine",
            "decision": "APPROVE",
        })

    def coder_action():
        run_tool("coder", {"summary": "implemented the thing"})

    def reviewer_action():
        decision = next(reviewer_iter)
        run_tool("reviewer", {
            "summary": f"review {decision}",
            "decision": decision,
        })

    def qa_action():
        decision = next(qa_decision_iter)
        score = next(qa_score_iter)
        run_tool(
            "qa", {
                "summary": f"qa {decision}",
                "decision": decision,
                "weighted_score": float(score),
            })

    workflow.plan_drafter = ScriptedAgent("plan_drafter reply",
                                          plan_drafter_action)
    workflow.plan_reviewer = ScriptedAgent("plan_reviewer reply",
                                           plan_reviewer_action)
    workflow.coder = ScriptedAgent("coder reply", coder_action)
    workflow.reviewer = ScriptedAgent("reviewer reply", reviewer_action)
    workflow.qa = ScriptedAgent("qa reply", qa_action)


def test_workflow_init_creates_empty_yaml(tmp_path):
    workflow_mod = _load_workflow_module()
    workflow = workflow_mod.AgentTeamWorkflow(tmp_path)
    assert workflow.progress_path == tmp_path / "progress.yaml"
    data = yaml.safe_load(workflow.progress_path.read_text(encoding="utf-8"))
    assert data == {
        "plan_stage": [],
        "build_stage": [],
        "human_feedback": [],
    }


def test_workflow_uses_tool_entries_for_reviewer_and_qa_decisions(tmp_path):
    progress = _load_progress_module()
    workflow_mod = _load_workflow_module()

    workflow = workflow_mod.AgentTeamWorkflow(
        tmp_path,
        num_iterations=3,
        coder_context_reset_interval=0,
        reviewer_context_reset_interval=0,
        min_score=8.0,
        plan_human_review_enabled=True,
    )
    # plan iter 1: drafter DRAFT_READY, reviewer APPROVE, drafter HUMAN_APPROVED
    # build iter 1: reviewer REJECT → skip QA → iter 2
    # build iter 2: reviewer APPROVE + QA APPROVE (score 9.0 ≥ min_score) → done
    _install_scripted_agents(
        workflow,
        reviewer_decisions=["REJECT", "APPROVE"],
        qa_decisions=["APPROVE"],
        qa_scores=[9.0],
    )

    task_yaml = workflow.workspace / "task.yaml"
    task_yaml.write_text("description: hello\n", encoding="utf-8")
    workflow.run(task_yaml)

    data = progress.read_progress(workflow.progress_path)

    # Plan-stage entries: drafter + reviewer + drafter human approve.
    assert [e["agent"] for e in data["plan_stage"]
            ] == ["plan_drafter", "plan_reviewer", "plan_drafter"]
    # Build-stage entries: iter 1 coder+reviewer (REJECT skips QA),
    # iter 2 coder+reviewer+qa (APPROVE ends).
    assert [e["agent"] for e in data["build_stage"]] == [
        "coder",
        "reviewer",
        "coder",
        "reviewer",
        "qa",
    ]

    # Orchestrator read decision values from YAML, not from text.
    reviewer_decisions = [
        e["decision"] for e in data["build_stage"] if e["agent"] == "reviewer"
    ]
    assert reviewer_decisions == ["REJECT", "APPROVE"]
    qa_entry = data["build_stage"][-1]
    assert qa_entry["decision"] == "APPROVE"
    assert qa_entry["weighted_score"] == 9.0

    plan_drafter_decisions = [
        e["decision"] for e in data["plan_stage"]
        if e["agent"] == "plan_drafter"
    ]
    assert plan_drafter_decisions == ["DRAFT_READY", "HUMAN_APPROVED"]

    # Final state: done.
    state_path = tmp_path / ".agent_team_state.json"
    assert state_path.is_file()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["done"] is True
    assert state["version"] == 4


def test_latest_entry_drives_reviewer_and_qa_decisions(tmp_path):
    progress = _load_progress_module()
    workflow_mod = _load_workflow_module()

    workflow = workflow_mod.AgentTeamWorkflow(tmp_path)

    # Seed progress.yaml with a reviewer APPROVE + qa APPROVE pair, then
    # verify the decision readers return the right values.
    progress.write_progress(
        workflow.progress_path, {
            "plan_stage": [],
            "build_stage": [
                {
                    "iteration": 1,
                    "agent": "reviewer",
                    "timestamp": "2026-01-01T00:00:00",
                    "summary": "ok",
                    "decision": "APPROVE",
                },
                {
                    "iteration": 1,
                    "agent": "qa",
                    "timestamp": "2026-01-01T00:00:00",
                    "summary": "done",
                    "decision": "APPROVE",
                    "weighted_score": 9.1,
                },
            ],
        })
    assert workflow._latest_reviewer_decision() == "APPROVE"
    assert workflow._latest_qa_decision() == "APPROVE"

    # Flip both to the negative side.
    progress.write_progress(
        workflow.progress_path, {
            "plan_stage": [],
            "build_stage": [
                {
                    "iteration": 1,
                    "agent": "reviewer",
                    "timestamp": "2026-01-01T00:00:00",
                    "summary": "defect",
                    "decision": "REJECT",
                },
                {
                    "iteration": 1,
                    "agent": "qa",
                    "timestamp": "2026-01-01T00:00:00",
                    "summary": "fails",
                    "decision": "REJECT",
                    "weighted_score": 3.1,
                },
            ],
        })
    assert workflow._latest_reviewer_decision() == "REJECT"
    assert workflow._latest_qa_decision() == "REJECT"

    # Empty progress → None for both.
    progress.init_progress_file(workflow.progress_path)
    assert workflow._latest_reviewer_decision() is None
    assert workflow._latest_qa_decision() is None


def test_workflow_resume_preserves_progress_yaml(tmp_path):
    progress = _load_progress_module()
    workflow_mod = _load_workflow_module()

    # Initial (fresh) construction writes an empty mapping.
    workflow = workflow_mod.AgentTeamWorkflow(tmp_path)
    progress.write_progress(
        workflow.progress_path, {
            "plan_stage": [
                {
                    "iteration": 1,
                    "agent": "plan_drafter",
                    "timestamp": "t",
                    "summary": "x",
                    "decision": "DRAFT_READY",
                },
            ],
            "build_stage": [],
        })

    # Persist a minimal state so resume is legal.
    save_state(
        tmp_path / ".agent_team_state.json",
        WorkflowState(task_path=str(tmp_path / "task.yaml"),
                      num_iterations=1,
                      stage=STAGE_CODER))

    # Resume must NOT clobber progress.yaml. With the checkpoint on disk
    # the constructor auto-detects resume mode — no explicit flag.
    resumed = workflow_mod.AgentTeamWorkflow(tmp_path)
    assert resumed.resume is True
    data = progress.read_progress(resumed.progress_path)
    assert data["plan_stage"][0]["agent"] == "plan_drafter"
    assert data["build_stage"] == []
