# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``source_embargo`` as a tool-call hook: refusals, redactions, the log, the wiring."""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_flow.workflows.staircase_onboard.embargo import (
    SourceEmbargo,
    embargo_hooks,
    embargo_transport_note,
    merge_hook_maps,
    source_embargo_hooks,
)

REL_FILE = "tensorrt_llm/_torch/models/modeling_famv4.py"


@pytest.fixture
def tree(tmp_path):
    repo = tmp_path / "repo"
    models = repo / "tensorrt_llm" / "_torch" / "models"
    models.mkdir(parents=True)
    (models / "modeling_famv4.py").write_text("SECRET = 'the existing implementation'\n")
    (models / "modeling_famv41.py").write_text("ALSO_FINE = 1\n")
    (models / "modeling_other.py").write_text("FINE = 1\n")
    sibling = tmp_path / "vllm"
    (sibling / "vllm" / "model_executor").mkdir(parents=True)
    (sibling / "vllm" / "model_executor" / "fam.py").write_text("SECRET = 2\n")
    workspace = repo / "onboard" / "ws"
    workspace.mkdir(parents=True)
    embargo = SourceEmbargo(
        [REL_FILE, str(sibling)],
        repo_root=repo,
        log_path=workspace / "embargo" / "attempts.jsonl",
    )
    return repo, sibling, workspace, embargo


# ------------------------------------------------------------------ refusals


def test_file_tools_are_refused_on_an_embargoed_file_or_tree(tree, tmp_path):
    repo, sibling, _, embargo = tree
    for tool, key in (("Read", "file_path"), ("Edit", "file_path"), ("Write", "file_path")):
        assert not embargo.check_tool(tool, {key: str(repo / REL_FILE)}).allowed
    assert not embargo.check_tool("NotebookEdit", {"notebook_path": str(repo / REL_FILE)}).allowed
    inside = sibling / "vllm" / "model_executor" / "fam.py"
    verdict = embargo.check_tool("Read", {"file_path": str(inside)})
    assert not verdict.allowed and verdict.entry == str(sibling)
    assert "logged" in verdict.reason
    # Relative to the session's cwd, and through a symlink.
    assert not embargo.check_tool("Read", {"file_path": REL_FILE}, cwd=repo).allowed
    link = tmp_path / "innocent.py"
    link.symlink_to(repo / REL_FILE)
    assert not embargo.check_tool("Read", {"file_path": str(link)}).allowed


def test_neighbours_are_not_embargoed(tree):
    repo, _, _, embargo = tree
    models = repo / "tensorrt_llm" / "_torch" / "models"
    for name in ("modeling_famv41.py", "modeling_other.py"):
        assert embargo.check_tool("Read", {"file_path": str(models / name)}).allowed
    assert embargo.check_tool("Glob", {"pattern": "**/modeling_*.py"}).allowed


def test_grep_inside_an_embargoed_path_is_refused_and_a_broad_grep_is_not(tree):
    repo, sibling, _, embargo = tree
    assert not embargo.check_tool("Grep", {"pattern": "x", "path": str(sibling / "vllm")}).allowed
    assert not embargo.check_tool("Grep", {"pattern": "x", "path": str(repo / REL_FILE)}).allowed
    assert embargo.check_tool("Grep", {"pattern": "x", "path": str(repo)}).allowed
    assert embargo.check_tool("Grep", {"pattern": "x"}).allowed


@pytest.mark.parametrize(
    "command",
    [
        "cat {repo}/" + REL_FILE,
        "sed -n 1,40p " + REL_FILE,
        "grep -n SECRET modeling_famv4.py",
        "head {repo}/tensorrt_llm/_torch/models/modeling_fam*.py",
        "python3 -c \"print(open('{repo}/" + REL_FILE + "').read())\"",
        "grep -rn SECRET {sibling}",
        "cd {sibling} && cat vllm/model_executor/fam.py",
        "git -C {repo} show HEAD:" + REL_FILE,
        "python -c 'import pathlib; print(pathlib.Path(\"{sibling}/vllm\").read_text())'",
        "less < {repo}/" + REL_FILE,
    ],
)
def test_shell_reads_of_an_embargoed_path_are_refused(tree, command):
    repo, sibling, _, embargo = tree
    verdict = embargo.check_tool(
        "Bash", {"command": command.format(repo=repo, sibling=sibling)}, cwd=repo
    )
    assert not verdict.allowed, command


@pytest.mark.parametrize(
    "command",
    [
        "TRTLLM_MODELING_V2=off python examples/llm-api/quickstart_advanced.py",
        "cd {sibling} && python -m vllm.entrypoints.openai.api_server --port 8000",
        "PYTHONPATH={sibling} python3 -m vllm.entrypoints.openai.api_server",
        "cat {repo}/tensorrt_llm/_torch/models/modeling_famv41.py",
        "pytest tests/unittest/test_modeling_famv4.py",
        "grep -rn forward {repo}/tensorrt_llm",
    ],
)
def test_running_the_implementation_and_unrelated_reads_stay_allowed(tree, command):
    repo, sibling, _, embargo = tree
    verdict = embargo.check_tool(
        "Bash", {"command": command.format(repo=repo, sibling=sibling)}, cwd=repo
    )
    assert verdict.allowed, command


# ------------------------------------------------------------------ redaction


def test_a_broad_search_has_the_embargoed_lines_removed_and_keeps_its_shape(tree):
    repo, sibling, _, embargo = tree
    grep_out = {
        "mode": "content",
        "numFiles": 2,
        "filenames": [str(repo / REL_FILE), str(repo / "tensorrt_llm/x.py")],
        "content": (
            f"{repo}/{REL_FILE}:1:SECRET = 'the existing implementation'\n"
            f"{repo}/tensorrt_llm/x.py:3:SECRET_NOT = 0\n"
            f"{sibling}/vllm/model_executor/fam.py:1:SECRET = 2\n"
        ),
    }
    redacted, removed = embargo.redact(grep_out)
    assert removed == 3
    assert set(redacted) == set(grep_out)
    assert redacted["filenames"] == [str(repo / "tensorrt_llm/x.py")]
    assert redacted["content"] == f"{repo}/tensorrt_llm/x.py:3:SECRET_NOT = 0\n"
    assert redacted["numFiles"] == 2  # numbers are left alone; only lines and names go

    bash_out = {
        "stdout": f"{REL_FILE}:1:SECRET\nother.py:2:x\n",
        "stderr": "",
        "interrupted": False,
    }
    redacted, removed = embargo.redact(bash_out)
    assert (
        removed == 1 and redacted["stdout"] == "other.py:2:x\n" and redacted["interrupted"] is False
    )
    assert embargo.redact({"stdout": "nothing here\n"}) == ({"stdout": "nothing here\n"}, 0)


# ------------------------------------------------------------------ the hooks


def test_the_hooks_refuse_redact_and_log_every_attempt(tree):
    repo, _, workspace, embargo = tree
    hooks = embargo_hooks(embargo, role="smith")
    pre = hooks["PreToolUse"][0]
    post = hooks["PostToolUse"][0]
    assert "Read" in pre.matcher and "Bash" in pre.matcher and "Grep" in pre.matcher

    denied = asyncio.run(
        pre.hooks[0](
            {
                "tool_name": "Read",
                "tool_input": {"file_path": str(repo / REL_FILE)},
                "cwd": str(workspace),
            },
            "t1",
            None,
        )
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    allowed = asyncio.run(
        pre.hooks[0](
            {"tool_name": "Read", "tool_input": {"file_path": str(repo / "x.py")}}, "t2", None
        )
    )
    assert allowed == {}

    response = {"stdout": f"{repo}/{REL_FILE}:1:SECRET\nkept\n", "stderr": "", "interrupted": False}
    redacted = asyncio.run(
        post.hooks[0](
            {
                "tool_name": "Bash",
                "tool_input": {"command": "grep -rn SECRET ."},
                "tool_response": response,
            },
            "t3",
            None,
        )
    )
    out = redacted["hookSpecificOutput"]
    assert out["updatedToolOutput"]["stdout"] == "kept\n"
    assert "withheld 1 line" in out["additionalContext"]

    attempts = [
        json.loads(line)
        for line in (workspace / "embargo" / "attempts.jsonl").read_text().splitlines()
    ]
    assert [a["action"] for a in attempts] == ["deny", "redact"]
    assert attempts[0]["role"] == "smith" and attempts[0]["tool"] == "Read"
    assert attempts[0]["entry"] == REL_FILE
    assert attempts[1]["removed"] == 1


def test_no_embargo_means_no_hooks_and_maps_merge_per_event(tmp_path):
    assert source_embargo_hooks({"source_embargo": []}, tmp_path, role="smith") is None
    assert merge_hook_maps(None, None) is None
    merged = merge_hook_maps({"Stop": [1]}, {"Stop": [2], "PreToolUse": [3]}, None)
    assert merged == {"Stop": [1, 2], "PreToolUse": [3]}


# ------------------------------------------------------------------ wiring


def test_every_in_process_role_gets_the_embargo_its_backend_can_enforce(tree, tmp_path):
    """A hook on claude-code; on codex, which refuses SDK callbacks, the instruction."""
    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    repo, _, _, _ = tree
    workflow = StaircaseOnboardWorkflow(workspace=tmp_path / "ws")
    workflow.task_data = {"trtllm_repo_path": str(repo), "source_embargo": [REL_FILE]}
    smith = workflow._make_agent("smith", "system", required_tools=["append_smith_progress"])
    assert smith.config.backend.kind == "claude-code"
    assert set(smith.config.backend.hooks) == {"Stop", "PreToolUse", "PostToolUse"}
    assert "not enforced by a hook" not in smith.config.system_prompt

    reviewer = workflow._make_agent("plan_reviewer", "system", required_tools=[])
    assert reviewer.config.backend.kind == "codex"
    assert not reviewer.config.backend.hooks
    assert "not enforced by a hook here" in reviewer.config.system_prompt
    assert "`codex` backend" in reviewer.config.system_prompt
    assert f"`{repo / REL_FILE}`" in reviewer.config.system_prompt
    attempts = SourceEmbargo.from_task(workflow.task_data, tmp_path / "ws").attempts()
    assert [(a["action"], a["role"]) for a in attempts] == [("instruct", "plan_reviewer")]

    workflow.task_data = {"trtllm_repo_path": str(repo), "source_embargo": []}
    plain = workflow._make_agent("smith", "system", required_tools=["append_smith_progress"])
    assert set(plain.config.backend.hooks) == {"Stop"}
    assert (
        workflow._make_agent("plan_reviewer", "system", required_tools=[]).config.system_prompt
        == "system"
    )


def test_a_codex_role_routed_by_the_task_is_instructed_too(tree, tmp_path):
    from agent_flow.workflows.staircase_onboard.embargo import source_embargo_enforcement

    repo, _, workspace, _ = tree
    task = {"trtllm_repo_path": str(repo), "source_embargo": [REL_FILE]}
    hooks, note = source_embargo_enforcement(task, workspace, role="smith", backend="codex")
    assert hooks is None and "`codex` backend" in note
    hooks, note = source_embargo_enforcement(task, workspace, role="smith", backend="claude-code")
    assert set(hooks) == {"PreToolUse", "PostToolUse"} and note == ""


def test_package_relative_entries_name_the_workflows_own_files(tree, tmp_path):
    """`package:<path>` is relative to the workflow package, wherever it is installed."""
    from agent_flow.workflows.staircase_onboard.embargo import (
        PACKAGE_ROOT,
        display_entry,
        resolve_entry,
    )

    repo, _, workspace, _ = tree
    entry = "package:task.example.yaml"
    assert resolve_entry(entry) == PACKAGE_ROOT / "task.example.yaml"
    embargo = SourceEmbargo([entry], repo_root=repo, log_path=workspace / "e.jsonl")
    target = PACKAGE_ROOT / "task.example.yaml"
    assert not embargo.check_tool("Read", {"file_path": str(target)}).allowed
    assert not embargo.check_tool("Bash", {"command": f"cat {target}"}).allowed
    assert embargo.check_tool("Read", {"file_path": str(PACKAGE_ROOT / "README.md")}).allowed
    # Outside the repository it is shown by its absolute path; inside, repo-relative.
    # The prefix is kept for the prompt's own wording; outside the repository
    # the path is absolute, inside it repo-relative.
    assert display_entry(entry, repo) == f"package:{target}"
    assert (
        display_entry(entry, PACKAGE_ROOT.parent) == "package:staircase_onboard/task.example.yaml"
    )
    assert display_entry(REL_FILE, repo) == REL_FILE


def test_the_schema_accepts_package_entries_that_exist_and_only_those(tree):
    from agent_flow.workflows.staircase_onboard.task_schema import _validate_source_embargo

    repo, _, _, _ = tree
    errors: list[str] = []
    _validate_source_embargo(["package:task.example.yaml", REL_FILE], errors, str(repo))
    assert errors == []
    _validate_source_embargo(["package:examples/no-such-model.md"], errors, str(repo))
    assert len(errors) == 1 and "workflow package" in errors[0]


def test_the_external_transport_restates_the_embargo_as_an_instruction(tree, tmp_path):
    from agent_flow.workflows.staircase_onboard.transport import ExternalAgentTransport

    repo, sibling, _, _ = tree
    transport = ExternalAgentTransport(
        tmp_path / "ws", poll_s=0.01, source_embargo=[REL_FILE, str(sibling)], repo_root=repo
    )
    note = transport._transport_note("assembler")
    assert "append_assembler_progress" in note  # the existing note is kept
    assert "not enforced by a hook here" in note and "external transport" in note
    assert f"`{repo}/{REL_FILE}`" in note and f"`{sibling}`" in note
    assert "Embargo exposures" in note
    bare = ExternalAgentTransport(tmp_path / "ws2", poll_s=0.01)
    assert "embargo" not in bare._transport_note("assembler").lower()
    assert embargo_transport_note([]) == ""


def test_the_embargo_is_built_from_the_task(tree):
    repo, sibling, workspace, _ = tree
    embargo = SourceEmbargo.from_task(
        {"trtllm_repo_path": str(repo), "source_embargo": [REL_FILE, str(sibling)]}, workspace
    )
    assert embargo is not None
    assert embargo.log_path == workspace / "embargo" / "attempts.jsonl"
    assert [e.is_dir for e in embargo.entries] == [False, True]
    assert SourceEmbargo.from_task({"source_embargo": []}, workspace) is None


@pytest.mark.parametrize(
    "entry,why",
    [
        ("package:../perf_analyze", "outside the workflow package"),
        ("package:examples/../../perf_analyze", "outside the workflow package"),
        ("package:../../workflows", "outside the workflow package"),
        ("package:", "whole workflow package"),
        ("package:.", "whole workflow package"),
    ],
)
def test_a_package_entry_may_not_escape_the_package(tree, entry, why):
    """`package:../perf_analyze` resolved to the sibling workflow and passed validation."""
    from agent_flow.workflows.staircase_onboard.embargo import EmbargoEntryError, resolve_entry
    from agent_flow.workflows.staircase_onboard.task_schema import _validate_source_embargo

    repo, _, _, _ = tree
    errors: list[str] = []
    _validate_source_embargo([entry], errors, str(repo))
    assert len(errors) == 1 and why in errors[0], errors
    with pytest.raises(EmbargoEntryError, match=why):
        resolve_entry(entry)
    with pytest.raises(EmbargoEntryError):
        SourceEmbargo([entry], repo_root=repo)


def test_the_role_prompt_calls_a_package_entry_reference_material(tree):
    """Not "this checkpoint already has an implementation in this repository"."""
    from agent_flow.workflows.staircase_onboard.embargo import PACKAGE_ROOT, display_entries
    from agent_flow.workflows.staircase_onboard.prompts import source_embargo_block

    repo, _, _, _ = tree
    inside = {
        "trtllm_repo_path": str(PACKAGE_ROOT.parent),
        "source_embargo": ["package:task.example.yaml"],
    }
    block = source_embargo_block(display_entries(inside))
    assert "- `staircase_onboard/task.example.yaml` (workflow reference material)" in block
    assert "reference material about this model" in block
    assert "already has an implementation in this repository" not in block
    assert "Running the in-repository implementation" not in block

    mixed = {
        "trtllm_repo_path": str(repo),
        "source_embargo": [REL_FILE, "package:task.example.yaml"],
    }
    block = source_embargo_block(display_entries(mixed))
    assert "already has an implementation in this repository" in block
    assert "reference material about this model" in block
    assert f"- `{REL_FILE}`\n" in block
    assert f"- `{PACKAGE_ROOT / 'task.example.yaml'}` (workflow reference material)" in block
