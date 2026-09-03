"""Tests for the no-in-process-MCP mode helpers (``--no-mcp-tools``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.agent_team import mcpless
from agent_flow.workflows.agent_team import progress as P

# --------------------------------------------------------------------------
# Handoff parsing
# --------------------------------------------------------------------------


def test_parse_handoff_qa_happy():
    text = "summary: ok\ndecision: APPROVE\nweighted_score: 8.5\n"
    assert mcpless.parse_handoff("qa", text) == {
        "summary": "ok",
        "decision": "APPROVE",
        "weighted_score": 8.5,
    }


def test_parse_handoff_coder_has_no_decision():
    assert mcpless.parse_handoff("coder", "summary: built X\n") == {"summary": "built X"}


def test_parse_handoff_score_zero_is_valid():
    out = mcpless.parse_handoff("qa", "summary: s\ndecision: REJECT\nweighted_score: 0\n")
    assert out["weighted_score"] == 0.0


def test_parse_handoff_missing_field_raises():
    with pytest.raises(mcpless.HandoffError, match="decision"):
        mcpless.parse_handoff("reviewer", "summary: ok\n")


def test_parse_handoff_bad_enum_raises():
    with pytest.raises(mcpless.HandoffError, match="decision"):
        mcpless.parse_handoff("reviewer", "summary: ok\ndecision: MAYBE\n")


def test_parse_handoff_plan_drafter_rejects_human_approved():
    """``HUMAN_APPROVED`` needs ``ask_human``, which no-MCP mode disables."""
    with pytest.raises(mcpless.HandoffError, match="decision"):
        mcpless.parse_handoff("plan_drafter", "summary: ok\ndecision: HUMAN_APPROVED\n")


def test_parse_handoff_score_out_of_range_raises():
    with pytest.raises(mcpless.HandoffError, match="weighted_score"):
        mcpless.parse_handoff("qa", "summary: ok\ndecision: APPROVE\nweighted_score: 42\n")


def test_parse_handoff_score_not_number_raises():
    with pytest.raises(mcpless.HandoffError, match="weighted_score"):
        mcpless.parse_handoff("qa", "summary: ok\ndecision: APPROVE\nweighted_score: high\n")


def test_parse_handoff_not_mapping_raises():
    with pytest.raises(mcpless.HandoffError, match="mapping"):
        mcpless.parse_handoff("coder", "- just\n- a list\n")


def test_parse_handoff_empty_summary_raises():
    with pytest.raises(mcpless.HandoffError, match="summary"):
        mcpless.parse_handoff("coder", "summary: '   '\n")


def test_parse_handoff_invalid_yaml_raises():
    with pytest.raises(mcpless.HandoffError, match="YAML"):
        mcpless.parse_handoff("coder", "summary: : :\n  - broken\n:")


# --------------------------------------------------------------------------
# Context gathering + preamble
# --------------------------------------------------------------------------


def _seed(tmp_path) -> tuple[Path, Path]:
    prog = tmp_path / "progress.yaml"
    P.write_progress(
        prog,
        {
            "plan_stage": [
                {
                    "iteration": 1,
                    "agent": "plan_drafter",
                    "summary": "drafted the plan",
                    "decision": "DRAFT_READY",
                }
            ],
            "build_stage": [{"iteration": 3, "agent": "coder", "summary": "impl v3"}],
            "human_feedback": [
                {"iteration": 2, "stage": "build_stage", "summary": "watch multi-gpu"}
            ],
        },
    )
    status = tmp_path / "status.md"
    status.write_text("# current\nstate here\n", encoding="utf-8")
    return prog, status


def test_gather_context_coder_includes_build_feedback_and_status(tmp_path):
    prog, status = _seed(tmp_path)
    ctx = mcpless.gather_context("coder", progress_path=prog, status_path=status)
    assert "impl v3" in ctx
    assert "watch multi-gpu" in ctx
    assert "state here" in ctx


def test_gather_context_reviewer_uses_coder_entry(tmp_path):
    prog, status = _seed(tmp_path)
    ctx = mcpless.gather_context("reviewer", progress_path=prog, status_path=status)
    assert "impl v3" in ctx
    assert "state here" in ctx


def test_gather_context_qa_only_feedback(tmp_path):
    prog, status = _seed(tmp_path)
    ctx = mcpless.gather_context("qa", progress_path=prog, status_path=status)
    assert "watch multi-gpu" in ctx
    assert "impl v3" not in ctx
    assert "state here" not in ctx


def test_gather_context_plan_reviewer_sees_drafter(tmp_path):
    prog, status = _seed(tmp_path)
    ctx = mcpless.gather_context("plan_reviewer", progress_path=prog, status_path=status)
    assert "drafted the plan" in ctx
    # No human feedback unless feedback-triggered.
    assert "watch multi-gpu" not in ctx
    ctx2 = mcpless.gather_context(
        "plan_reviewer", progress_path=prog, status_path=status, feedback_triggered=True
    )
    assert "watch multi-gpu" in ctx2


def test_gather_context_plan_drafter_replan_sees_build(tmp_path):
    prog, status = _seed(tmp_path)
    draft_ctx = mcpless.gather_context("plan_drafter", progress_path=prog, status_path=status)
    assert "impl v3" not in draft_ctx  # draft mode reads plan_stage only
    replan_ctx = mcpless.gather_context(
        "plan_drafter", progress_path=prog, status_path=status, replan=True
    )
    assert "impl v3" in replan_ctx
    assert "watch multi-gpu" in replan_ctx


def test_gather_context_unknown_role_raises(tmp_path):
    prog, status = _seed(tmp_path)
    with pytest.raises(ValueError, match="unknown role"):
        mcpless.gather_context("nope", progress_path=prog, status_path=status)


def test_preamble_names_handoff_path_and_status_for_coder(tmp_path):
    hp = mcpless.handoff_path(tmp_path / ".turn", "coder")
    pre = mcpless.build_recording_preamble("coder", hp, tmp_path / "status.md", "CTXBODY")
    assert str(hp) in pre
    assert str(tmp_path / "status.md") in pre
    assert "CTXBODY" in pre
    assert "append_coder_progress" in pre  # explicitly reinterprets the old tool
    assert "update_status" in pre


def test_preamble_qa_has_score_and_no_status_duty(tmp_path):
    hp = mcpless.handoff_path(tmp_path / ".turn", "qa")
    pre = mcpless.build_recording_preamble("qa", hp, tmp_path / "status.md", "CTX")
    assert "weighted_score" in pre
    # QA never writes status.md, so the status overwrite clause must be absent.
    assert "OVERWRITE" not in pre


def test_preamble_write_instruction_is_backend_neutral(tmp_path):
    """The write step must be satisfiable on claude-code *and* codex.

    ``--no-mcp-tools`` disables tools for every role at once, and this
    workflow runs plan_drafter/reviewer on codex (no ``Write`` tool) while the
    rest run on claude-code. Naming only ``Write`` would send the codex roles
    after a tool they do not have.
    """
    for role in ("plan_drafter", "reviewer", "coder", "qa"):
        hp = mcpless.handoff_path(tmp_path / ".turn", role)
        pre = mcpless.build_recording_preamble(role, hp, tmp_path / "status.md", "CTX")
        assert "Write" in pre
        assert "apply_patch" in pre


def test_preamble_plan_drafter_omits_human_approved(tmp_path):
    hp = mcpless.handoff_path(tmp_path / ".turn", "plan_drafter")
    pre = mcpless.build_recording_preamble("plan_drafter", hp, tmp_path / "status.md", "CTX")
    assert "DRAFT_READY" in pre
    assert "HUMAN_APPROVED" not in pre


def test_handoff_path():
    assert mcpless.handoff_path(Path("/w/.turn"), "reviewer") == Path("/w/.turn/reviewer.yaml")
