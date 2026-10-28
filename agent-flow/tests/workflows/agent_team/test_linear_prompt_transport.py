"""Tests that the LINEAR path's per-turn prompts respect the transport split.

``PromptBundle`` declares the invariant: role prompts say WHAT a turn reads and
records, and every MCP tool name lives in a block that is appended only when
the run actually registers those tools. The role *system* prompts have always
honored it via ``MCP_TOOLS_EXTENSIONS``; the per-turn prompts built in
``workflow.py`` did not, so a ``--no-mcp-tools`` run ordered each role to call
tools it never registered.

These tests are the contract for both directions: the turn body alone must name
no tool, and MCP mode must still spell out exactly the tools that role uses.

``human`` / ``replan_human`` are deliberately absent. ``ask_human`` needs an
in-process MCP server, so those modes are rejected at construction under
``--no-mcp-tools`` (covered by ``test_no_mcp_rejects_human_review``) and only
ever run with tools present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.agent_team import progress as progress_module
from agent_flow.workflows.agent_team import workflow as _workflow_module

# Every MCP tool the workflow can register. No per-turn body may name one.
_MCP_TOOL_NAMES = (
    "append_plan_drafter_progress",
    "append_plan_reviewer_progress",
    "append_coder_progress",
    "append_reviewer_progress",
    "append_qa_progress",
    "read_latest_progress",
    "read_latest_build_progress",
    "read_human_feedback",
    "read_status",
    "update_status",
    "ask_human",
)


def _make_workflow(tmp_path: Path, *, use_in_process_tools: bool):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    workflow = _workflow_module.AgentTeamWorkflow(
        workspace=workspace,
        replan_on_qa=True,
        use_in_process_tools=use_in_process_tools,
    )
    # Seeded AFTER construction: a pre-populated progress.yaml with no
    # checkpoint is exactly what the workspace guard refuses. The replan turn
    # reads the latest QA verdict out of this file.
    progress_module.init_progress_file(workspace / "progress.yaml")
    return workflow


def _capture_prompts(workflow) -> list[tuple[str, str]]:
    """Redirect ``_invoke_agent`` so each turn's prompt is recorded, not run."""
    captured: list[tuple[str, str]] = []

    def _fake_invoke(role, _agent, prompt, _iteration, **_kwargs):
        captured.append((role, prompt))

    workflow._invoke_agent = _fake_invoke
    return captured


def _all_turn_prompts(tmp_path: Path, *, use_in_process_tools: bool) -> dict[str, str]:
    """Build one prompt per reachable turn kind and return them keyed by kind."""
    workflow = _make_workflow(tmp_path, use_in_process_tools=use_in_process_tools)
    captured = _capture_prompts(workflow)
    try:
        workflow._run_plan_drafter(1, mode="draft")
        workflow._run_plan_drafter(1, mode="replan")
        workflow._run_plan_drafter(1, mode="replan", feedback_triggered=True)
        workflow._run_plan_reviewer(1, phase="initial")
        workflow._run_plan_reviewer(1, phase="replan", feedback_triggered=True)
        workflow._run_coder(1)
        workflow._run_reviewer(1)
        workflow._run_qa(1)
    finally:
        workflow.close()

    kinds = [
        "plan_drafter_draft",
        "plan_drafter_replan",
        "plan_drafter_replan_feedback",
        "plan_reviewer_initial",
        "plan_reviewer_replan_feedback",
        "coder",
        "reviewer",
        "qa",
    ]
    assert len(captured) == len(kinds)
    return {kind: prompt for kind, (_role, prompt) in zip(kinds, captured)}


# ------------------------------------------------------------------ no-MCP mode


def test_no_turn_body_names_an_mcp_tool(tmp_path):
    """Under ``--no-mcp-tools`` no turn prompt may name a tool.

    This is the defect a real ``--no-mcp-tools`` run surfaced: the plan-phase
    turns told the PlanDrafter to call ``read_latest_progress``, which that run
    never registered.
    """
    prompts = _all_turn_prompts(tmp_path, use_in_process_tools=False)
    for kind, prompt in prompts.items():
        for tool in _MCP_TOOL_NAMES:
            assert tool not in prompt, f"{kind} body names the MCP tool {tool!r}"


def test_no_turn_body_carries_a_protocol_block(tmp_path):
    """The MCP protocol header is dropped wholesale, not merely emptied."""
    prompts = _all_turn_prompts(tmp_path, use_in_process_tools=False)
    for kind, prompt in prompts.items():
        assert "RECORDING PROTOCOL" not in prompt, f"{kind} kept the MCP block"


def test_no_mcp_bodies_still_state_what_to_record(tmp_path):
    """Dropping the tool names must not drop the duty they carried."""
    prompts = _all_turn_prompts(tmp_path, use_in_process_tools=False)
    assert "record a Coder progress entry" in prompts["coder"]
    assert "refresh status.md" in prompts["coder"]
    assert "record a Reviewer progress entry" in prompts["reviewer"]
    assert "record a QA progress entry" in prompts["qa"]
    assert "record a PlanReviewer progress entry" in prompts["plan_reviewer_initial"]
    assert "record a PlanDrafter progress entry" in prompts["plan_drafter_draft"]
    assert "record exactly one PlanDrafter progress entry" in prompts["plan_drafter_replan"]


# -------------------------------------------------------------------- MCP mode


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("plan_drafter_draft", ("read_latest_progress", "append_plan_drafter_progress")),
        (
            "plan_drafter_replan",
            ("read_latest_build_progress", "read_human_feedback", "append_plan_drafter_progress"),
        ),
        ("plan_reviewer_initial", ("read_latest_progress", "append_plan_reviewer_progress")),
        (
            "coder",
            ("read_status", "read_latest_progress", "read_human_feedback", "append_coder_progress"),
        ),
        ("reviewer", ("read_status", "read_latest_progress", "append_reviewer_progress")),
        ("qa", ("read_human_feedback", "append_qa_progress")),
    ],
)
def test_mcp_mode_still_names_each_role_tools(tmp_path, kind, expected):
    """Regression guard: the default path keeps the full tool protocol."""
    prompts = _all_turn_prompts(tmp_path, use_in_process_tools=True)
    prompt = prompts[kind]
    assert "RECORDING PROTOCOL" in prompt
    for tool in expected:
        assert tool in prompt, f"{kind} lost the MCP tool {tool!r}"


def test_mcp_mode_keeps_required_stop_hook_tools_in_the_prompt(tmp_path):
    """The two roles with AND-semantics Stop hooks must still be told both tools."""
    prompts = _all_turn_prompts(tmp_path, use_in_process_tools=True)
    for kind, append_tool in (
        ("coder", "append_coder_progress"),
        ("reviewer", "append_reviewer_progress"),
    ):
        assert "**both** required tools" in prompts[kind]
        assert append_tool in prompts[kind]
        assert "update_status" in prompts[kind]


def test_protocol_table_covers_every_reachable_kind():
    """A kind with no table entry would raise KeyError at prompt-build time."""
    assert set(_workflow_module._LINEAR_MCP_PROTOCOL) == {
        "plan_drafter_draft",
        "plan_drafter_replan",
        "plan_reviewer",
        "coder",
        "reviewer",
        "qa",
    }


def test_with_protocol_is_a_passthrough_without_tools():
    body = "turn body"
    assert _workflow_module._with_protocol(body, "coder", False) == body
    assert _workflow_module._with_protocol(body, "coder", True).startswith(body)
    assert "RECORDING PROTOCOL" in _workflow_module._with_protocol(body, "coder", True)
