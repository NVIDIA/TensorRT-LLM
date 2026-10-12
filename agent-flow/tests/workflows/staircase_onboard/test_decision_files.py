# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the file-based decision protocol.

This mode exists because an enterprise MCP configuration
(``/etc/claude-code/managed-mcp.json``) makes the Agent SDK refuse every
dynamically-registered server — including the in-process one the six typed
decision tools live in. On such a host those tools cannot be created at all,
and the run died on its first dispatch with an opaque
``ProcessError: exit code 1``.

The guarantee the tools provided was a **closed value**: the orchestrator
branches on ``decision`` and never on prose. These tests are about where
that guarantee went — it moved from the tool schema to validation on read,
and it has to be just as strict there or the mode is a downgrade pretending
not to be one.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import yaml

from agent_flow.workflows.staircase_onboard.progress import (
    DECISIONS_BY_AGENT,
    EXTRA_KEYS_BY_AGENT,
    ProgressContext,
    collect_decision_file,
    decision_is_stale,
    decision_path,
    init_progress_file,
    latest_decision,
    latest_entry,
    mark_decision_dispatch,
)
from agent_flow.workflows.staircase_onboard.prompts import build_staircase_onboard_prompts
from agent_flow.workflows.staircase_onboard.prompts._common import decision_file_block


def _context(tmp_path: Path) -> ProgressContext:
    progress = tmp_path / "progress.yaml"
    init_progress_file(progress)
    return ProgressContext(path=progress)


def _write(tmp_path: Path, agent: str, iteration: int, payload: dict) -> None:
    path = decision_path(tmp_path, agent, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_a_valid_decision_is_recorded_and_readable_downstream(tmp_path):
    """The whole point: existing `latest_decision` callers keep working."""
    ctx = _context(tmp_path)
    _write(tmp_path, "plan_drafter", 1, {"summary": "wrote both files", "decision": "DRAFT_READY"})

    assert collect_decision_file(ctx, tmp_path, "plan_drafter", 1) == "DRAFT_READY"
    assert (
        latest_decision(ctx.path, "plan_drafter", DECISIONS_BY_AGENT["plan_drafter"])
        == "DRAFT_READY"
    )
    assert latest_entry(ctx.path, "plan_drafter")["summary"] == "wrote both files"


def test_a_decision_outside_the_enum_is_refused(tmp_path):
    """The tool schema would have refused it; so must the reader.

    Accepting it would turn the orchestrator's one closed input into
    whatever a role felt like writing — which is the failure the typed
    tools existed to prevent.
    """
    ctx = _context(tmp_path)
    _write(tmp_path, "plan_drafter", 1, {"summary": "s", "decision": "LOOKS_GOOD_TO_ME"})
    assert collect_decision_file(ctx, tmp_path, "plan_drafter", 1) is None
    assert latest_entry(ctx.path, "plan_drafter") is None


def test_another_roles_decision_is_not_accepted(tmp_path):
    """`APPROVE` is real — for the reviewer. It is not a drafter decision."""
    ctx = _context(tmp_path)
    _write(tmp_path, "plan_drafter", 1, {"summary": "s", "decision": "APPROVE"})
    assert collect_decision_file(ctx, tmp_path, "plan_drafter", 1) is None


def test_a_missing_file_is_none_rather_than_an_exception(tmp_path):
    """A role that ended without deciding is a case the caller handles."""
    ctx = _context(tmp_path)
    assert collect_decision_file(ctx, tmp_path, "smith", 3) is None


def test_malformed_yaml_is_refused_without_raising(tmp_path):
    ctx = _context(tmp_path)
    path = decision_path(tmp_path, "smith", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("decision: [unclosed\n", encoding="utf-8")
    assert collect_decision_file(ctx, tmp_path, "smith", 1) is None


def test_role_specific_extra_keys_survive(tmp_path):
    """The assembler's `blocked_reason` is control flow, not commentary.

    `_handle_blocked_assembler` branches on it — a BLOCKED entry that lost
    it is a run that cannot triage itself.
    """
    ctx = _context(tmp_path)
    _write(
        tmp_path,
        "assembler",
        1,
        {
            "summary": "no entry for the gated delta rule",
            "decision": "BLOCKED",
            "blocked_reason": "vocabulary_gap",
            "missing_entry": "gdn_custom_op_inplace",
        },
    )
    assert collect_decision_file(ctx, tmp_path, "assembler", 1) == "BLOCKED"
    entry = latest_entry(ctx.path, "assembler")
    assert entry["blocked_reason"] == "vocabulary_gap"
    assert entry["missing_entry"] == "gdn_custom_op_inplace"


def test_empty_extra_keys_are_dropped_rather_than_stored_blank(tmp_path):
    ctx = _context(tmp_path)
    _write(tmp_path, "smith", 1, {"summary": "s", "decision": "LANDED", "entry": "   "})
    collect_decision_file(ctx, tmp_path, "smith", 1)
    assert "entry" not in latest_entry(ctx.path, "smith")


def test_the_iteration_is_the_orchestrators_not_the_agents(tmp_path):
    """A role could write any iteration into the file; it is not asked to.

    Stamping it here keeps a mistyped or hallucinated number from
    reordering the record.
    """
    ctx = _context(tmp_path)
    _write(tmp_path, "smith", 7, {"summary": "s", "decision": "LANDED", "iteration": 99})
    collect_decision_file(ctx, tmp_path, "smith", 7)
    assert latest_entry(ctx.path, "smith")["iteration"] == 7


# ---------------------------------------------- a stale decision is not a decision


def _write_decision(workspace: Path, agent: str, iteration: int, decision: str) -> Path:
    target = decision_path(workspace, agent, iteration)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"summary: s\ndecision: {decision}\n", encoding="utf-8")
    return target


def test_a_decision_written_after_the_marker_is_collected(tmp_path):
    """The ordinary case: a turn answered its own dispatch."""
    ctx = _context(tmp_path)
    mark_decision_dispatch(tmp_path, "auditor", 1)
    _write_decision(tmp_path, "auditor", 1, "CLEAN")
    assert collect_decision_file(ctx, tmp_path, "auditor", 1) == "CLEAN"


def test_a_decision_older_than_the_marker_is_refused(tmp_path):
    """The failure this exists for, and it happened five times in one run.

    `(agent, iteration)` is not unique across a run — `iteration` is 1 for
    every deliver-stage turn and restarts per unit in assembly. A turn that
    ends without writing leaves the PREVIOUS turn's verdict at the same path,
    looking fresh. Three audit verdicts were read back that way, including a
    `CLEAN` that could therefore never end the run.
    """
    ctx = _context(tmp_path)
    stale = _write_decision(tmp_path, "auditor", 1, "UNCOVERED")
    os.utime(stale, (1_000_000, 1_000_000))  # written long before this dispatch
    mark_decision_dispatch(tmp_path, "auditor", 1)

    assert decision_is_stale(tmp_path, "auditor", 1)
    assert collect_decision_file(ctx, tmp_path, "auditor", 1) is None
    # and nothing was appended, so `latest_decision` cannot read it either
    assert latest_decision(ctx.path, "auditor", DECISIONS_BY_AGENT["auditor"]) is None


def test_without_a_marker_the_old_behaviour_stands(tmp_path):
    """A run that never stamped one must not have its decisions refused.

    Absence means "this run does not use the marker", not "stale" — the same
    three-state rule the rest of the workflow applies to receipts.
    """
    ctx = _context(tmp_path)
    _write_decision(tmp_path, "reviewer", 2, "APPROVE")
    assert not decision_is_stale(tmp_path, "reviewer", 2)
    assert collect_decision_file(ctx, tmp_path, "reviewer", 2) == "APPROVE"


def test_a_missing_decision_is_not_stale(tmp_path):
    """A stamped turn that was never answered is the caller's "no decision"."""
    mark_decision_dispatch(tmp_path, "assembler", 3)
    assert not decision_is_stale(tmp_path, "assembler", 3)


def test_the_marker_does_not_collide_with_the_decision_it_guards(tmp_path):
    """A dotfile beside it, so a marker is never mistaken for an answer."""
    mark_decision_dispatch(tmp_path, "smith", 4)
    written = sorted(p.name for p in decision_path(tmp_path, "smith", 4).parent.iterdir())
    assert written == [".smith-4.dispatched"]


def test_staleness_is_judged_by_mtime_not_by_the_clock():
    """Both files live under the workspace, so one filesystem orders them.

    An agent may answer from a node whose clock is skewed by minutes; a
    wall-clock comparison would reject its perfectly good decision.
    """
    src = inspect.getsource(decision_is_stale)
    assert "st_mtime" in src
    assert "time.time" not in src and "datetime.now" not in src


# ------------------------------------------------------------------ prompts


def test_each_role_is_shown_only_its_own_options():
    """A role shown another role's enum will eventually pick from it."""
    block = decision_file_block("assembler", DECISIONS_BY_AGENT["assembler"])
    assert "UNIT_READY" in block and "BLOCKED" in block
    for foreign in ("DRAFT_READY", "CEILING", "LANDED", "FAILED_RECEIPT"):
        assert foreign not in block, foreign


def test_every_role_gets_a_decision_block_when_the_mode_is_on():
    bundle = build_staircase_onboard_prompts(decision_files=True)
    for role, decisions in DECISIONS_BY_AGENT.items():
        prompt = getattr(bundle, role)
        assert "Ending your turn" in prompt, role
        assert f"decisions/{role}-<iteration>.yaml" in prompt, role
        for value in decisions:
            assert f"`{value}`" in prompt, (role, value)


def test_roles_with_extra_keys_are_told_about_them():
    bundle = build_staircase_onboard_prompts(decision_files=True)
    for role, keys in EXTRA_KEYS_BY_AGENT.items():
        for key in keys:
            assert key in getattr(bundle, role), (role, key)


def test_no_decision_block_when_the_tools_are_available():
    """The tools are the preferred mechanism; this mode is the fallback."""
    bundle = build_staircase_onboard_prompts(decision_files=False)
    assert "decisions/plan_drafter-<iteration>.yaml" not in bundle.plan_drafter


# ------------------------------------------------- the workspace spec copy


def test_a_resumed_run_refreshes_the_spec_the_agents_read(tmp_path):
    """Two copies of task.yaml must not drift apart.

    `_init_state` re-reads and resolves the operator's file every run, but
    the workspace copy — which every role is told to read as "the run's
    spec" — was written once and never again. A resumed run therefore gave
    the gates one spec and the assembler another: both self-consistent,
    neither correct, and nothing would say so.
    """
    import yaml as _yaml

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    workspace = tmp_path / "ws"
    workspace.mkdir()
    stale = {"features": [], "environment": {"env": {}}}
    (workspace / "task.yaml").write_text(_yaml.safe_dump(stale), encoding="utf-8")

    source = tmp_path / "task.yaml"
    source.write_text("", encoding="utf-8")

    workflow = StaircaseOnboardWorkflow(workspace)
    workflow.task_data = {"features": ["mtp"], "environment": {"env": {"PATH": "/venv/bin"}}}
    workflow._refresh_workspace_spec(source, None)

    written = _yaml.safe_load((workspace / "task.yaml").read_text(encoding="utf-8"))
    assert written["features"] == ["mtp"]
    assert written["environment"]["env"]["PATH"] == "/venv/bin"


def test_refreshing_is_a_no_op_when_the_spec_already_matches(tmp_path):
    """Rewriting an identical file would churn its mtime for no reason."""
    import yaml as _yaml

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    workspace = tmp_path / "ws"
    workspace.mkdir()
    data = {"features": ["mtp"]}
    (workspace / "task.yaml").write_text(
        _yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    before = (workspace / "task.yaml").stat().st_mtime_ns

    source = tmp_path / "task.yaml"
    source.write_text("", encoding="utf-8")

    workflow = StaircaseOnboardWorkflow(workspace)
    workflow.task_data = data
    workflow._refresh_workspace_spec(source, None)

    assert (workspace / "task.yaml").stat().st_mtime_ns == before
