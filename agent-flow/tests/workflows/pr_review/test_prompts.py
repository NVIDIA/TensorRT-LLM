from __future__ import annotations

import pytest

from agent_flow.workflows.pr_review import workflow as workflow_module
from agent_flow.workflows.pr_review.prompts import (
    CODER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    SOURCING_SYSTEM_PROMPT,
)
from agent_flow.workflows.pr_review.state import STAGE1


@pytest.mark.parametrize("prompt", [REVIEWER_SYSTEM_PROMPT, CODER_SYSTEM_PROMPT])
def test_no_posting_rule_present_in_both_roles(prompt):
    # Hard rule 1: never post to the PR/MR — the forbidden commands are named.
    assert "gh pr comment" in prompt
    assert "gh pr review" in prompt
    assert "glab mr note" in prompt
    assert "glab mr approve" in prompt
    assert "local" in prompt.lower()


@pytest.mark.parametrize("prompt", [REVIEWER_SYSTEM_PROMPT, CODER_SYSTEM_PROMPT])
def test_leave_in_working_tree_rule_present_in_both_roles(prompt):
    # Hard rule 2: never commit or push.
    assert "git commit" in prompt
    assert "git push" in prompt


def test_reviewer_prompt_decision_contract():
    assert "APPROVE" in REVIEWER_SYSTEM_PROMPT
    assert "REQUEST_CHANGES" in REVIEWER_SYSTEM_PROMPT
    # The reviewer must engage with the coder's right to push back.
    assert "push back" in REVIEWER_SYSTEM_PROMPT.lower()


def test_coder_prompt_grants_pushback_and_decision_contract():
    assert "push back" in CODER_SYSTEM_PROMPT.lower()
    assert "STAND_FIRM" in CODER_SYSTEM_PROMPT
    assert "AGREE" in CODER_SYSTEM_PROMPT
    assert "REVISE" in CODER_SYSTEM_PROMPT
    # Explicitly grants the right to decline a change it disagrees with.
    assert "disagree" in CODER_SYSTEM_PROMPT.lower()


def test_sourcing_prompt_checks_out_reports_and_forbids_writes():
    p = SOURCING_SYSTEM_PROMPT
    # It checks the PR/MR out with gh/glab itself...
    assert "gh pr checkout" in p
    assert "glab mr checkout" in p
    # ...reports back via the tool...
    assert "report_pr_context" in p
    # ...and never posts, commits, or pushes.
    assert "gh pr comment" in p
    assert "glab mr note" in p
    assert "git commit" in p
    assert "git push" in p


class _PromptStub:
    """Captures the runtime prompt; a no-op context manager for ``close``."""

    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)

    def __exit__(self, *_args):
        return None


class _SourcingStub(_PromptStub):
    """Prompt stub that also simulates the agent calling ``report_pr_context``."""

    def __init__(self, ctx):
        super().__init__()
        self._ctx = ctx

    def __call__(self, prompt):
        super().__call__(prompt)
        self._ctx.base = "main"
        self._ctx.reported = True


def _fresh_workflow(tmp_path):
    return workflow_module.PrReviewWorkflow(repo=tmp_path, target="1", workspace=tmp_path)


def test_run_sourcing_prompt_content(tmp_path):
    workflow = _fresh_workflow(tmp_path)
    workflow.sourcing.__exit__(None, None, None)
    stub = _SourcingStub(workflow._sourcing_ctx)
    workflow.sourcing = stub
    try:
        meta = workflow._run_sourcing()
    finally:
        workflow.close()

    assert meta["base"] == "main"
    assert len(stub.prompts) == 1
    p = stub.prompts[0]
    # Platform-agnostic: the agent is told to detect GitHub PR vs GitLab MR
    # itself, not handed one fixed CLI.
    assert "GitHub PR" in p and "GitLab MR" in p
    assert "1" in p
    assert "report_pr_context" in p
    assert "do NOT commit" in p


def test_run_reviewer_prompt_content(tmp_path):
    workflow = _fresh_workflow(tmp_path)
    workflow.s1_reviewer.__exit__(None, None, None)
    stub = _PromptStub()
    workflow.s1_reviewer = stub
    try:
        workflow._run_reviewer(STAGE1, 1, "s1_reviewer")
    finally:
        workflow.close()

    assert len(stub.prompts) == 1
    p = stub.prompts[0]
    assert "pr_context.md" in p
    assert "append_reviewer_progress" in p
    assert "read_latest_progress" in p
    assert "do NOT post" in p
    assert "do NOT commit" in p


def test_run_coder_prompt_content(tmp_path):
    workflow = _fresh_workflow(tmp_path)
    workflow.s1_coder.__exit__(None, None, None)
    stub = _PromptStub()
    workflow.s1_coder = stub
    try:
        workflow._run_coder(STAGE1, 1, "s1_coder")
    finally:
        workflow.close()

    assert len(stub.prompts) == 1
    p = stub.prompts[0]
    assert "pr_context.md" in p
    assert "append_coder_progress" in p
    assert "push back" in p.lower()
    assert "STAND_FIRM" in p
    assert "do NOT commit" in p
