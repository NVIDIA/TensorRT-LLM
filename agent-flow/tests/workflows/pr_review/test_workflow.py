from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow import CLAUDE_CODE_DEFAULT_MODEL, CODEX_DEFAULT_MODEL
from agent_flow.workflows.pr_review import progress as progress_module
from agent_flow.workflows.pr_review import state as state_module
from agent_flow.workflows.pr_review import workflow as workflow_module
from agent_flow.workflows.pr_review.state import STAGE1, STAGE2

_S1R = state_module.STAGE_S1_REVIEWER
_S1C = state_module.STAGE_S1_CODER
_S2R = state_module.STAGE_S2_REVIEWER


def _seeded_workflow(
    tmp_path,
    *,
    stage=None,
    next_round_index=0,
    num_rounds=20,
    done=False,
    progress_yaml="stage1: []\nstage2: []\n",
):
    """Seed a resume checkpoint so ``run`` skips the sourcing-agent bootstrap."""
    stage = stage or _S1R
    (tmp_path / "pr_context.md").write_text("# PR under review\n", encoding="utf-8")
    (tmp_path / "progress.yaml").write_text(progress_yaml, encoding="utf-8")
    (tmp_path / "discussion.md").write_text("", encoding="utf-8")
    state_module.save_state(
        tmp_path / state_module.STATE_FILENAME,
        state_module.WorkflowState(
            pr_context_path=str(tmp_path / "pr_context.md"),
            num_rounds=num_rounds,
            next_round_index=next_round_index,
            stage=stage,
            done=done,
        ),
    )
    return workflow_module.PrReviewWorkflow(
        repo=tmp_path, target="1", workspace=tmp_path, num_rounds=num_rounds
    )


def _stub_agents(workflow, decisions=None):
    """Replace the agent entry points with recorders; never invoke a backend.

    ``decisions`` is ``{stage_key: {"reviewer": [...], "coder": [...]}}``,
    consumed per round. Defaults: reviewer APPROVE, coder AGREE.
    """
    decisions = decisions or {}
    trace: list[tuple[str, int, str]] = []
    iters = {}
    for stage in (STAGE1, STAGE2):
        s = decisions.get(stage, {})
        iters[(stage, "reviewer")] = iter(s.get("reviewer", []))
        iters[(stage, "coder")] = iter(s.get("coder", []))

    def _append(stage_key, agent, decision, round_no):
        data = progress_module.read_progress(workflow.progress_path)
        data[stage_key].append(
            {"round": round_no, "stage": stage_key, "agent": agent, "decision": decision}
        )
        progress_module.write_progress(workflow.progress_path, data)

    def reviewer(stage_key, round_no, agent_attr):
        trace.append((stage_key, round_no, "reviewer"))
        _append(stage_key, "reviewer", next(iters[(stage_key, "reviewer")], "APPROVE"), round_no)

    def coder(stage_key, round_no, agent_attr):
        trace.append((stage_key, round_no, "coder"))
        _append(stage_key, "coder", next(iters[(stage_key, "coder")], "AGREE"), round_no)

    workflow._run_reviewer = reviewer
    workflow._run_coder = coder
    workflow._reset_agent = lambda attr: None
    return trace


def test_clean_convergence_runs_each_stage_once(tmp_path):
    workflow = _seeded_workflow(tmp_path)
    trace = _stub_agents(workflow)
    try:
        workflow.run()
    finally:
        workflow.close()

    assert trace == [
        (STAGE1, 1, "reviewer"),
        (STAGE1, 1, "coder"),
        (STAGE2, 1, "reviewer"),
        (STAGE2, 1, "coder"),
    ]
    state = state_module.load_state(tmp_path / state_module.STATE_FILENAME)
    assert state.done is True


def test_stage1_loops_then_converges(tmp_path):
    workflow = _seeded_workflow(tmp_path)
    trace = _stub_agents(
        workflow,
        decisions={
            STAGE1: {
                "reviewer": ["REQUEST_CHANGES", "APPROVE"],
                "coder": ["REVISE", "AGREE"],
            }
        },
    )
    try:
        workflow.run()
    finally:
        workflow.close()

    assert trace == [
        (STAGE1, 1, "reviewer"),
        (STAGE1, 1, "coder"),
        (STAGE1, 2, "reviewer"),
        (STAGE1, 2, "coder"),
        (STAGE2, 1, "reviewer"),
        (STAGE2, 1, "coder"),
    ]
    assert state_module.load_state(tmp_path / state_module.STATE_FILENAME).done is True


def test_coder_stand_firm_ends_stage_and_advances(tmp_path):
    workflow = _seeded_workflow(tmp_path)
    # Reviewer keeps requesting changes, but the coder stands firm on round 1 →
    # push-back wins, stage 1 ends immediately and stage 2 runs.
    trace = _stub_agents(
        workflow,
        decisions={
            STAGE1: {"reviewer": ["REQUEST_CHANGES"], "coder": ["STAND_FIRM"]},
        },
    )
    try:
        workflow.run()
    finally:
        workflow.close()

    assert trace == [
        (STAGE1, 1, "reviewer"),
        (STAGE1, 1, "coder"),
        (STAGE2, 1, "reviewer"),
        (STAGE2, 1, "coder"),
    ]
    assert state_module.load_state(tmp_path / state_module.STATE_FILENAME).done is True


def test_round_budget_exhausted_advances(tmp_path):
    workflow = _seeded_workflow(tmp_path, num_rounds=2)
    # Stage 1 never settles (always REQUEST_CHANGES / REVISE) → after 2 rounds
    # the workflow advances anyway; stage 2 then converges.
    trace = _stub_agents(
        workflow,
        decisions={
            STAGE1: {
                "reviewer": ["REQUEST_CHANGES", "REQUEST_CHANGES"],
                "coder": ["REVISE", "REVISE"],
            }
        },
    )
    try:
        workflow.run()
    finally:
        workflow.close()

    assert trace == [
        (STAGE1, 1, "reviewer"),
        (STAGE1, 1, "coder"),
        (STAGE1, 2, "reviewer"),
        (STAGE1, 2, "coder"),
        (STAGE2, 1, "reviewer"),
        (STAGE2, 1, "coder"),
    ]
    assert state_module.load_state(tmp_path / state_module.STATE_FILENAME).done is True


def test_coder_agree_while_reviewer_requests_changes_loops(tmp_path):
    workflow = _seeded_workflow(tmp_path)
    # Coder AGREE alone must NOT end the stage while the reviewer still wants
    # changes — only reviewer APPROVE + coder AGREE converges.
    trace = _stub_agents(
        workflow,
        decisions={
            STAGE1: {
                "reviewer": ["REQUEST_CHANGES", "APPROVE"],
                "coder": ["AGREE", "AGREE"],
            }
        },
    )
    try:
        workflow.run()
    finally:
        workflow.close()

    assert trace[:4] == [
        (STAGE1, 1, "reviewer"),
        (STAGE1, 1, "coder"),
        (STAGE1, 2, "reviewer"),
        (STAGE1, 2, "coder"),
    ]


def test_resume_mid_round_skips_completed_reviewer(tmp_path):
    # Checkpoint mid-round: stage-1 reviewer already ran (APPROVE on disk) and
    # the workflow stopped at the coder sub-stage. Resume must run only the
    # coder for round 1, not re-run the reviewer.
    workflow = _seeded_workflow(
        tmp_path,
        stage=_S1C,
        progress_yaml=(
            "stage1:\n"
            "  - round: 1\n"
            "    stage: stage1\n"
            "    agent: reviewer\n"
            "    decision: APPROVE\n"
            "stage2: []\n"
        ),
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run()
    finally:
        workflow.close()

    assert trace == [
        (STAGE1, 1, "coder"),
        (STAGE2, 1, "reviewer"),
        (STAGE2, 1, "coder"),
    ]
    assert state_module.load_state(tmp_path / state_module.STATE_FILENAME).done is True


def test_resume_done_is_noop(tmp_path):
    workflow = _seeded_workflow(tmp_path, done=True)
    trace = _stub_agents(workflow)
    try:
        workflow.run()
    finally:
        workflow.close()
    assert trace == []


def test_model_and_role_wiring(tmp_path):
    # Fresh construction (no checkpoint) does no network — that happens in run().
    workflow = workflow_module.PrReviewWorkflow(repo=tmp_path, target="1", workspace=tmp_path)
    try:
        assert workflow.s1_reviewer.config.backend.kind == "claude-code"
        assert workflow.s1_coder.config.backend.kind == "codex"
        assert workflow.s2_reviewer.config.backend.kind == "codex"
        assert workflow.s2_coder.config.backend.kind == "claude-code"

        assert workflow.s1_reviewer.config.backend.model == CLAUDE_CODE_DEFAULT_MODEL
        assert workflow.s1_coder.config.backend.model == CODEX_DEFAULT_MODEL
        assert workflow.s2_reviewer.config.backend.model == CODEX_DEFAULT_MODEL
        assert workflow.s2_coder.config.backend.model == CLAUDE_CODE_DEFAULT_MODEL
    finally:
        workflow.close()


def test_tools_wired_per_agent(tmp_path):
    workflow = workflow_module.PrReviewWorkflow(repo=tmp_path, target="1", workspace=tmp_path)
    try:

        def _names(layer):
            return sorted(t.name for t in (layer.config.backend.tools or []))

        for attr in ("s1_reviewer", "s2_reviewer"):
            names = _names(getattr(workflow, attr))
            assert names == [
                "append_reviewer_progress",
                "read_discussion",
                "read_latest_progress",
                "update_discussion",
            ]
        for attr in ("s1_coder", "s2_coder"):
            names = _names(getattr(workflow, attr))
            assert names == [
                "append_coder_progress",
                "read_discussion",
                "read_latest_progress",
                "update_discussion",
            ]
    finally:
        workflow.close()


def test_sourcing_agent_wired(tmp_path):
    workflow = workflow_module.PrReviewWorkflow(repo=tmp_path, target="1", workspace=tmp_path)
    try:
        s = workflow.sourcing
        # Sourcing runs gh/glab itself, so it's a Claude Code agent with the
        # single report_pr_context tool, one-shot (stateless) session.
        assert s.config.backend.kind == "claude-code"
        assert s.config.session.mode == "stateless"
        assert sorted(t.name for t in (s.config.backend.tools or [])) == ["report_pr_context"]
    finally:
        workflow.close()


def test_fresh_run_sources_via_agent_and_writes_context(tmp_path, monkeypatch):
    # The orchestrator no longer runs gh/glab. On a fresh run it drives the
    # sourcing agent (which reports metadata) and derives the diff with git.
    monkeypatch.setattr(
        workflow_module.vcs, "resolve_diff_base_ref", lambda repo, base: f"origin/{base}"
    )
    monkeypatch.setattr(workflow_module.vcs, "diff_stat", lambda repo, ref: "")

    workflow = workflow_module.PrReviewWorkflow(repo=tmp_path, target="7", workspace=tmp_path)
    captured = {}

    def fake_sourcing(prompt):
        captured["prompt"] = prompt
        ctx = workflow._sourcing_ctx
        ctx.base, ctx.head, ctx.title = "main", "feature/x", "Add x"
        ctx.author, ctx.url, ctx.body = "alice", "https://example/pr/7", "does x"
        ctx.reported = True

    workflow.sourcing.__exit__(None, None, None)
    workflow.sourcing = fake_sourcing  # a plain callable; close() skips non-AgentLayer

    try:
        state = workflow._init_state(None)
    finally:
        workflow.close()

    assert state is not None
    assert state.stage == _S1R
    # The agent, not the orchestrator, was asked to detect the platform, check
    # out, and report — the orchestrator never names gh/glab as a fixed choice.
    assert "report_pr_context" in captured["prompt"]
    assert "7" in captured["prompt"]
    assert "GitHub PR" in captured["prompt"] and "GitLab MR" in captured["prompt"]
    md = (tmp_path / "pr_context.md").read_text(encoding="utf-8")
    assert "Add x" in md
    assert "merge-base origin/main HEAD" in md


def test_run_sourcing_raises_when_agent_does_not_report(tmp_path):
    workflow = workflow_module.PrReviewWorkflow(repo=tmp_path, target="7", workspace=tmp_path)
    workflow.sourcing.__exit__(None, None, None)
    workflow.sourcing = lambda prompt: None  # never calls report_pr_context
    try:
        with pytest.raises(Exception, match="report_pr_context"):
            workflow._run_sourcing()
    finally:
        workflow.close()


def test_fresh_start_raises_on_nonempty_progress(tmp_path):
    (tmp_path / "progress.yaml").write_text(
        "stage1:\n  - round: 1\n    agent: reviewer\n    decision: APPROVE\nstage2: []\n",
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError, match="progress.yaml"):
        workflow_module.PrReviewWorkflow(repo=tmp_path, target="1", workspace=tmp_path)


def test_fresh_start_raises_on_nonempty_discussion(tmp_path):
    (tmp_path / "discussion.md").write_text("# stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="discussion.md"):
        workflow_module.PrReviewWorkflow(repo=tmp_path, target="1", workspace=tmp_path)


def test_fresh_start_allows_empty_shell_progress(tmp_path):
    # init_progress_file's empty shell must not block a retry.
    (tmp_path / "progress.yaml").write_text("stage1: []\nstage2: []\n", encoding="utf-8")
    workflow = workflow_module.PrReviewWorkflow(repo=tmp_path, target="1", workspace=tmp_path)
    try:
        assert workflow.resume is False
    finally:
        workflow.close()


def test_clean_wipes_and_restarts(tmp_path):
    (tmp_path / "progress.yaml").write_text(
        "stage1:\n  - round: 1\n    agent: reviewer\n    decision: APPROVE\nstage2: []\n",
        encoding="utf-8",
    )
    (tmp_path / "discussion.md").write_text("# stale\n", encoding="utf-8")
    state_module.save_state(
        tmp_path / state_module.STATE_FILENAME,
        state_module.WorkflowState(
            pr_context_path=str(tmp_path / "pr_context.md"), num_rounds=1, stage=_S1R
        ),
    )
    workflow = workflow_module.PrReviewWorkflow(
        repo=tmp_path, target="1", workspace=tmp_path, clean=True
    )
    try:
        assert workflow.resume is False
        assert not (tmp_path / state_module.STATE_FILENAME).is_file()
        assert (tmp_path / "discussion.md").read_text(encoding="utf-8") == ""
        assert progress_module.read_progress(tmp_path / "progress.yaml") == {
            "stage1": [],
            "stage2": [],
        }
    finally:
        workflow.close()


def test_default_workspace_is_namespaced_by_id(tmp_path, monkeypatch):
    # Default workspace mirrors the other workflows' relative-path convention
    # (e.g. agent_team's ``workspace/agent-team``), namespaced per PR/MR id —
    # no platform prefix, since we don't distinguish PR from MR.
    monkeypatch.chdir(tmp_path)
    workflow = workflow_module.PrReviewWorkflow(repo=tmp_path, target="42")
    try:
        assert workflow.workspace == Path("workspace") / "pr-review" / "42"
    finally:
        workflow.close()


def test_default_workspace_slugifies_url_target(tmp_path, monkeypatch):
    # A URL target is reduced to a filesystem-safe slug so it still gets its
    # own isolated workspace directory.
    monkeypatch.chdir(tmp_path)
    workflow = workflow_module.PrReviewWorkflow(
        repo=tmp_path, target="https://github.com/o/r/pull/7"
    )
    try:
        ws = workflow.workspace
        assert ws.parent == Path("workspace") / "pr-review"
        assert "/" not in ws.name and ":" not in ws.name
        assert "7" in ws.name
    finally:
        workflow.close()


def test_constructor_requires_a_target(tmp_path):
    # A single target is required; empty / missing raises (no pr/mr split).
    with pytest.raises(ValueError, match="empty target"):
        workflow_module.PrReviewWorkflow(repo=tmp_path, workspace=tmp_path)
    with pytest.raises(ValueError, match="empty target"):
        workflow_module.PrReviewWorkflow(repo=tmp_path, target="   ", workspace=tmp_path)
