from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.agent_team import cli as _cli_module
from agent_flow.workflows.agent_team import progress as progress_module
from agent_flow.workflows.agent_team import workflow as _workflow_module


def _load_module():
    return _workflow_module


def _load_cli_module():
    return _cli_module


def _make_workflow(module, workspace, *, clean=False):
    return module.AgentTeamWorkflow(workspace=workspace, clean=clean)


def _write_task_yaml(workspace, content: str = "description: demo\n") -> Path:
    """Write a minimal ``task.yaml`` into ``workspace`` and return its path.

    ``--task`` is path-only since the YAML-input switch; tests that
    historically passed a literal task string call this helper to
    materialize a YAML file alongside the workspace and hand its path to
    ``workflow.run``. agent_team imposes no schema on the YAML, so any
    valid YAML body is acceptable.
    """
    path = Path(workspace) / "task.yaml"
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return path


def test_fresh_start_raises_when_plan_is_non_empty(tmp_path):
    module = _load_module()
    (tmp_path / "plan.md").write_text("# stale plan\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="plan.md"):
        _make_workflow(module, tmp_path)


def test_fresh_start_raises_when_acceptance_criteria_is_non_empty(tmp_path):
    module = _load_module()
    (tmp_path / "acceptance-criteria.md").write_text("- [ ] stale\n",
                                                     encoding="utf-8")

    with pytest.raises(FileExistsError, match="acceptance-criteria.md"):
        _make_workflow(module, tmp_path)


def test_fresh_start_raises_when_progress_is_non_empty(tmp_path):
    module = _load_module()
    (tmp_path / "progress.yaml").write_text("- iteration: 1\n",
                                            encoding="utf-8")

    with pytest.raises(FileExistsError, match="progress.yaml"):
        _make_workflow(module, tmp_path)


def test_fresh_start_raises_when_status_md_is_non_empty(tmp_path):
    module = _load_module()
    (tmp_path / "status.md").write_text("# stale snapshot\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="status.md"):
        _make_workflow(module, tmp_path)


def _empty_progress_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_fresh_start_allows_empty_plan_and_progress(tmp_path):
    module = _load_module()
    (tmp_path / "plan.md").write_text("   \n", encoding="utf-8")
    (tmp_path / "acceptance-criteria.md").write_text("\n", encoding="utf-8")
    (tmp_path / "progress.yaml").write_text("", encoding="utf-8")
    (tmp_path / "status.md").write_text("\n  \n", encoding="utf-8")

    # Should not raise; whitespace-only content is treated as empty.
    workflow = _make_workflow(module, tmp_path)
    try:
        assert (tmp_path / "plan.md").read_text(encoding="utf-8") == ""
        assert (tmp_path /
                "acceptance-criteria.md").read_text(encoding="utf-8") == ""
        # Fresh start seeds progress.yaml with all top-level keys empty.
        assert _empty_progress_yaml(tmp_path / "progress.yaml") == {
            "plan_stage": [],
            "build_stage": [],
            "human_feedback": [],
        }
        # status.md is reset to empty on fresh start.
        assert (tmp_path / "status.md").read_text(encoding="utf-8") == ""
    finally:
        workflow.close()


def test_fresh_start_allows_missing_files(tmp_path):
    module = _load_module()

    workflow = _make_workflow(module, tmp_path)
    try:
        assert (tmp_path / "plan.md").read_text(encoding="utf-8") == ""
        # Fresh start also initializes an empty acceptance-criteria.md.
        assert (tmp_path /
                "acceptance-criteria.md").read_text(encoding="utf-8") == ""
        assert _empty_progress_yaml(tmp_path / "progress.yaml") == {
            "plan_stage": [],
            "build_stage": [],
            "human_feedback": [],
        }
        # Fresh start initializes an empty status.md scratchpad.
        assert (tmp_path / "status.md").read_text(encoding="utf-8") == ""
    finally:
        workflow.close()


class _PromptStub:
    """Stand-in for an ``AgentLayer`` that captures the prompt and is a
    no-op context manager — drop-in safe for ``workflow.close()``."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> None:
        self.prompts.append(prompt)

    def __exit__(self, *_args, **_kwargs) -> None:
        return None


def test_run_coder_prompt_does_not_mention_test_command_md(tmp_path):
    """Base agent_team should not know about ``test_command.md``; that
    file is a modeling-bringup-only mechanism layered in via prompt
    extensions, not by the base workflow."""
    module = _load_module()
    workflow = _make_workflow(module, tmp_path)

    workflow.coder.__exit__(None, None, None)
    stub = _PromptStub()
    workflow.coder = stub
    try:
        workflow._run_coder(1)
    finally:
        workflow.close()

    assert len(stub.prompts) == 1
    assert "test_command.md" not in stub.prompts[0]


def test_run_reviewer_prompt_does_not_mention_test_command_md(tmp_path):
    module = _load_module()
    workflow = _make_workflow(module, tmp_path)

    workflow.reviewer.__exit__(None, None, None)
    stub = _PromptStub()
    workflow.reviewer = stub
    try:
        workflow._run_reviewer(1)
    finally:
        workflow.close()

    assert len(stub.prompts) == 1
    assert "test_command.md" not in stub.prompts[0]


def test_run_qa_prompt_does_not_mention_test_command_md(tmp_path):
    module = _load_module()
    workflow = _make_workflow(module, tmp_path)

    workflow.qa.__exit__(None, None, None)
    stub = _PromptStub()
    workflow.qa = stub
    try:
        workflow._run_qa(1)
    finally:
        workflow.close()

    assert len(stub.prompts) == 1
    assert "test_command.md" not in stub.prompts[0]


def test_fresh_start_does_not_create_test_command_md(tmp_path):
    """Base agent_team must not create ``test_command.md`` on a fresh
    run; the file is bring-up-only and the build-phase agents create it
    on first write only when their domain extension tells them to."""
    module = _load_module()

    workflow = _make_workflow(module, tmp_path)
    try:
        assert not (tmp_path / "test_command.md").exists()
    finally:
        workflow.close()


def test_should_reset_coder_always_resets_after_first_iteration(tmp_path):
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        coder_context_reset_interval=2,
    )
    try:
        # i == 0 never resets (coder just started).
        assert workflow._should_reset_coder(0) is False
        # i == 1: interval=2 would skip, but iteration 1 always triggers a
        # reset so refinement starts from a clean context.
        assert workflow._should_reset_coder(1) is True
        # Regular interval cadence still applies at later iterations.
        assert workflow._should_reset_coder(2) is True
        assert workflow._should_reset_coder(3) is False
        assert workflow._should_reset_coder(4) is True
    finally:
        workflow.close()


def test_should_reset_coder_interval_zero_still_resets_after_first(tmp_path):
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        coder_context_reset_interval=0,
    )
    try:
        assert workflow._should_reset_coder(0) is False
        # Iteration-1 reset is unconditional, independent of the interval.
        assert workflow._should_reset_coder(1) is True
        # Interval disabled: no further resets.
        assert workflow._should_reset_coder(2) is False
        assert workflow._should_reset_coder(5) is False
    finally:
        workflow.close()


def test_auto_resume_skips_non_empty_guard(tmp_path):
    module = _load_module()
    (tmp_path / "plan.md").write_text("# kept plan\n", encoding="utf-8")
    (tmp_path / "acceptance-criteria.md").write_text("- [ ] kept\n",
                                                     encoding="utf-8")
    seeded = ("plan_stage:\n"
              "  - iteration: 1\n"
              "    agent: plan_drafter\n"
              "    decision: DRAFT_READY\n"
              "build_stage: []\n")
    (tmp_path / "progress.yaml").write_text(seeded, encoding="utf-8")

    # Presence of the checkpoint flips the workflow into resume mode; the
    # non-empty-files guard must not fire even though plan.md /
    # progress.yaml are non-empty.
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(task_path=str(tmp_path / "task.yaml"),
                             num_iterations=1))

    workflow = _make_workflow(module, tmp_path)
    try:
        # Non-empty files must be preserved on resume.
        assert workflow.resume is True
        assert (tmp_path /
                "plan.md").read_text(encoding="utf-8") == "# kept plan\n"
        assert (tmp_path / "acceptance-criteria.md").read_text(
            encoding="utf-8") == "- [ ] kept\n"
        assert (tmp_path /
                "progress.yaml").read_text(encoding="utf-8") == seeded
    finally:
        workflow.close()


def test_clean_wipes_checkpoint_and_managed_files(tmp_path):
    """``--clean`` deletes the checkpoint plus the workflow-managed files
    so the constructor proceeds as a fresh run."""
    module = _load_module()
    (tmp_path / "plan.md").write_text("# stale plan\n", encoding="utf-8")
    (tmp_path / "acceptance-criteria.md").write_text("- [ ] stale\n",
                                                     encoding="utf-8")
    (tmp_path / "progress.yaml").write_text("- iteration: 1\n",
                                            encoding="utf-8")
    (tmp_path / "status.md").write_text("# stale snapshot\n", encoding="utf-8")
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(task_path=str(tmp_path / "task.yaml"),
                             num_iterations=1))

    # Construction with clean=True must succeed even though every managed
    # file is non-empty.
    workflow = _make_workflow(module, tmp_path, clean=True)
    try:
        assert workflow.resume is False
        # Checkpoint was deleted.
        assert not (tmp_path / module.STATE_FILENAME).is_file()
        # Managed files were wiped and re-initialized to empty by the
        # fresh-start path.
        assert (tmp_path / "plan.md").read_text(encoding="utf-8") == ""
        assert (tmp_path /
                "acceptance-criteria.md").read_text(encoding="utf-8") == ""
        assert (tmp_path / "status.md").read_text(encoding="utf-8") == ""
    finally:
        workflow.close()


def test_clean_on_empty_workspace_is_a_noop(tmp_path):
    """``--clean`` against an empty workspace just starts fresh, no error."""
    module = _load_module()

    workflow = _make_workflow(module, tmp_path, clean=True)
    try:
        assert workflow.resume is False
    finally:
        workflow.close()


def test_fresh_start_error_message_points_to_clean(tmp_path):
    """The non-empty-files guard hint must point users at --clean."""
    module = _load_module()
    (tmp_path / "plan.md").write_text("# stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--clean"):
        _make_workflow(module, tmp_path)


def test_stage_roundtrips_through_save_and_load(tmp_path):
    module = _load_module()
    path = tmp_path / module.STATE_FILENAME
    for stage in (
            module.STAGE_PLAN_DRAFTER,
            module.STAGE_PLAN_REVIEWER,
            module.STAGE_PLAN_HUMAN,
            module.STAGE_CODER,
            module.STAGE_REVIEWER,
            module.STAGE_QA,
    ):
        state = module.WorkflowState(task_path=str(tmp_path / "task.yaml"),
                                     num_iterations=3,
                                     next_iteration_index=1,
                                     plan_next_iteration_index=2,
                                     stage=stage)
        module.save_state(path, state)
        loaded = module.load_state(path)
        assert loaded.stage == stage
        assert loaded.next_iteration_index == 1
        assert loaded.plan_next_iteration_index == 2


def test_load_state_rejects_schema_v1(tmp_path):
    module = _load_module()
    path = tmp_path / module.STATE_FILENAME
    # v1 checkpoints are rejected outright; users delete and restart.
    path.write_text(
        '{"version": 1, "task_path": "t", "num_iterations": 2, '
        '"next_iteration_index": 0, "stage": "generator", "done": false}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        module.load_state(path)


def test_load_state_rejects_schema_v2(tmp_path):
    """v2 checkpoints (single-stage planner) are no longer compatible."""
    module = _load_module()
    path = tmp_path / module.STATE_FILENAME
    path.write_text(
        '{"version": 2, "task_path": "t", "num_iterations": 1, '
        '"stage": "coder"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        module.load_state(path)


def test_load_state_rejects_schema_v3(tmp_path):
    """v3 checkpoints (carrying ``plan_iterations``) are rejected so the
    user restarts cleanly under the current schema."""
    module = _load_module()
    path = tmp_path / module.STATE_FILENAME
    path.write_text(
        '{"version": 3, "task_path": "t", "num_iterations": 1, '
        '"plan_iterations": 20, "stage": "plan_drafter"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        module.load_state(path)


def test_save_state_round_trips_task_path(tmp_path):
    module = _load_module()
    path = tmp_path / module.STATE_FILENAME
    module.save_state(
        path,
        module.WorkflowState(task_path=str(tmp_path / "task.yaml"),
                             num_iterations=5,
                             next_iteration_index=2,
                             stage=module.STAGE_REVIEWER),
    )
    # The persisted JSON must carry the path, not a duplicated task string.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["task_path"] == str(tmp_path / "task.yaml")
    assert "task" not in data
    loaded = module.load_state(path)
    assert loaded.task_path == str(tmp_path / "task.yaml")


def test_load_state_rejects_unknown_stage(tmp_path):
    module = _load_module()
    path = tmp_path / module.STATE_FILENAME
    path.write_text(
        '{"version": 4, "task_path": "t", "num_iterations": 1, "stage": "bogus"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported stage"):
        module.load_state(path)


def _stub_agents(
    workflow,
    plan_drafter_decisions: dict | None = None,
    plan_reviewer_decisions: list[str] | None = None,
    reviewer_decisions: list[str] | None = None,
    qa_decisions: list[str] | None = None,
    qa_scores: list[float] | None = None,
):
    """Replace the workflow's agent entry points with recorders.

    Returns ``trace`` — a list of ``(iteration, stage)`` tuples capturing the
    order in which stages executed. Real backends are never invoked. The
    PlanReviewer / Reviewer / QA stubs append minimal entries to
    ``progress.yaml`` so the workflow's decision readers pick up the
    stubbed decisions. The PlanDrafter stub appends a ``plan_drafter``
    entry whose ``decision`` is taken from
    ``plan_drafter_decisions[mode]`` (cycled).

    ``plan_drafter_decisions`` defaults to ``{"draft": ["DRAFT_READY"],
    "human": ["HUMAN_APPROVED"]}`` (one entry per call, cycled).
    ``plan_reviewer_decisions``, ``reviewer_decisions``, ``qa_decisions``
    are consumed per stub call in order. Defaults: all ``APPROVE`` for
    both reviewers and QA.
    """
    trace: list[tuple[int, str]] = []

    plan_drafter_decisions = dict(plan_drafter_decisions or {})
    plan_drafter_decisions.setdefault("draft", ["DRAFT_READY"])
    plan_drafter_decisions.setdefault("human", ["HUMAN_APPROVED"])
    drafter_iters = {
        mode: iter(seq * 1)  # eager copy via list multiplication
        for mode, seq in plan_drafter_decisions.items()
    }

    plan_reviewer_iter = iter(plan_reviewer_decisions or [])
    reviewer_iter = iter(reviewer_decisions or [])
    qa_iter = iter(qa_decisions or [])
    score_iter = iter(qa_scores or [])

    def _append(agent: str, entry: dict) -> None:
        data = progress_module.read_progress(workflow.progress_path)
        data[progress_module._STAGE_BY_AGENT[agent]].append(entry)
        progress_module.write_progress(workflow.progress_path, data)

    def plan_drafter(iteration: int, mode: str):
        trace.append((iteration, f"plan_drafter:{mode}"))
        decision = next(drafter_iters[mode], plan_drafter_decisions[mode][-1])
        _append("plan_drafter", {
            "iteration": iteration,
            "agent": "plan_drafter",
            "decision": decision,
        })

    def plan_reviewer(iteration: int):
        trace.append((iteration, "plan_reviewer"))
        decision = next(plan_reviewer_iter, "APPROVE")
        _append(
            "plan_reviewer", {
                "iteration": iteration,
                "agent": "plan_reviewer",
                "decision": decision,
            })

    def coder(iteration: int):
        trace.append((iteration, "coder"))

    def reviewer(iteration: int):
        trace.append((iteration, "reviewer"))
        decision = next(reviewer_iter, "APPROVE")
        _append("reviewer", {
            "iteration": iteration,
            "agent": "reviewer",
            "decision": decision,
        })

    def qa(iteration: int):
        trace.append((iteration, "qa"))
        decision = next(qa_iter, "APPROVE")
        score = next(score_iter, 9.5)
        _append(
            "qa", {
                "iteration": iteration,
                "agent": "qa",
                "decision": decision,
                "weighted_score": float(score),
            })

    workflow._run_plan_drafter = plan_drafter
    workflow._run_plan_reviewer = plan_reviewer
    workflow._run_coder = coder
    workflow._run_reviewer = reviewer
    workflow._run_qa = qa
    # Suppress agent recycling (the real reset tears down a backend
    # client). QA is stateless and the plan agents are not recycled, so
    # only the coder/reviewer resets need suppression.
    workflow._reset_coder = lambda: None
    workflow._reset_reviewer = lambda: None
    return trace


def test_fresh_run_executes_stages_in_order(tmp_path):
    """Happy path with plan-stage human review enabled: plan APPROVE +
    human approve + reviewer APPROVE + qa APPROVE ends the workflow
    immediately."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=3,
        plan_human_review_enabled=True,
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Plan iter 1 produces DRAFT_READY, plan_reviewer APPROVES, human
    # approves, then build iter 1 runs coder + reviewer + qa.
    assert trace == [
        (1, "plan_drafter:draft"),
        (1, "plan_reviewer"),
        (1, "plan_drafter:human"),
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
    ]

    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True
    assert state.next_iteration_index == 1


def test_no_plan_human_review_skips_plan_human_stage(tmp_path):
    """With ``plan_human_review_enabled=False``, PlanReviewer APPROVE
    jumps straight to the build phase without invoking the PlanDrafter
    in human mode."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        plan_human_review_enabled=False,
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # No ``plan_drafter:human`` entry in the trace; coder runs straight
    # after plan_reviewer APPROVE.
    assert trace == [
        (1, "plan_drafter:draft"),
        (1, "plan_reviewer"),
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True


def test_no_plan_human_review_loops_back_on_plan_reviewer_reject(tmp_path):
    """Plan-reviewer REJECT still loops back to PlanDrafter even when
    plan-stage human review is disabled."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        plan_human_review_enabled=False,
    )
    trace = _stub_agents(
        workflow,
        plan_reviewer_decisions=["REJECT", "APPROVE"],
    )
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Plan iter 1: drafter then reviewer REJECT → loop.
    # Plan iter 2: drafter then reviewer APPROVE → straight to build phase.
    assert trace == [
        (1, "plan_drafter:draft"),
        (1, "plan_reviewer"),
        (2, "plan_drafter:draft"),
        (2, "plan_reviewer"),
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True


def test_no_plan_human_review_resume_from_plan_human_skips_to_coder(tmp_path):
    """A checkpoint at STAGE_PLAN_HUMAN resumed with plan-stage human
    review now disabled must skip straight to the build phase without
    running the PlanDrafter in human mode."""
    module = _load_module()
    (tmp_path / "plan.md").write_text("# plan\n", encoding="utf-8")
    (tmp_path / "progress.yaml").write_text(
        "plan_stage:\n"
        "  - iteration: 1\n"
        "    agent: plan_drafter\n"
        "    decision: DRAFT_READY\n"
        "  - iteration: 1\n"
        "    agent: plan_reviewer\n"
        "    decision: APPROVE\n"
        "build_stage: []\n",
        encoding="utf-8",
    )
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(
            task_path=str(tmp_path / "task.yaml"),
            num_iterations=1,
            plan_next_iteration_index=0,
            stage=module.STAGE_PLAN_HUMAN,
        ),
    )

    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        plan_human_review_enabled=False,
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # No plan_drafter:human entry — resume jumped straight to the build
    # phase.
    assert trace == [
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True


_TASK_CLI = ["--task", "task.yaml"]


def test_plan_and_build_human_review_cli_flags(tmp_path):
    """The dual human-review flags parse independently with the
    expected defaults: both OFF."""
    module = _load_cli_module()
    args = module._parse_args(_TASK_CLI)
    assert args.plan_human_review_enabled is False
    assert args.build_human_review_enabled is False

    args = module._parse_args(_TASK_CLI + ["--plan-human-review"])
    assert args.plan_human_review_enabled is True
    assert args.build_human_review_enabled is False

    args = module._parse_args(_TASK_CLI + ["--build-human-review"])
    assert args.plan_human_review_enabled is False
    assert args.build_human_review_enabled is True

    args = module._parse_args(
        _TASK_CLI + ["--plan-human-review", "--build-human-review"], )
    assert args.plan_human_review_enabled is True
    assert args.build_human_review_enabled is True


def test_clean_cli_flag_default_and_value(tmp_path):
    """``--clean`` parses to False by default and to True when set."""
    module = _load_cli_module()
    args = module._parse_args(_TASK_CLI)
    assert args.clean is False
    args = module._parse_args(_TASK_CLI + ["--clean"])
    assert args.clean is True


def test_feedback_cli_flag_default_and_value():
    """``--feedback`` parses to None by default and to the supplied string."""
    module = _load_cli_module()
    args = module._parse_args(_TASK_CLI)
    assert args.feedback is None
    args = module._parse_args(_TASK_CLI +
                              ["--feedback", "please add streaming"])
    assert args.feedback == "please add streaming"


def test_feedback_on_fresh_run_raises(tmp_path):
    """``--feedback`` is rejected when there is no checkpoint to resume from.

    On a fresh run there are no prior iterations to course-correct and the
    plan phase has not produced a plan yet, so the entry would land before
    any build-phase agent can consume it. The constructor must surface this
    to the user instead of silently appending an orphan entry.
    """
    module = _load_module()
    with pytest.raises(ValueError, match="--feedback is only valid") as exc:
        module.AgentTeamWorkflow(workspace=tmp_path,
                                 feedback="please add streaming")
    # The error redirects the user to put initial guidance into the task
    # description, since --feedback on a fresh run would otherwise look
    # like the right knob for "tell the workflow what to do up front".
    assert "--task" in str(exc.value)
    # The check fires before any workspace files are materialized.
    assert not (tmp_path / "progress.yaml").exists()
    assert not (tmp_path / module.STATE_FILENAME).exists()


def test_feedback_with_clean_raises_even_when_checkpoint_existed(tmp_path):
    """``--clean`` wipes the checkpoint *before* the resume check, so passing
    ``--feedback`` alongside ``--clean`` is still a fresh run and must error.
    Otherwise the user would think they were course-correcting an existing
    workflow while actually starting over.
    """
    module = _load_module()
    _seed_build_phase_resume(module, tmp_path)
    with pytest.raises(ValueError, match="--feedback is only valid"):
        module.AgentTeamWorkflow(workspace=tmp_path,
                                 clean=True,
                                 feedback="please add streaming")


def _seed_build_phase_resume(module, tmp_path, *, next_iteration_index=0):
    """Seed a workspace so the workflow auto-resumes in the build phase.

    Returns the populated state path. ``num_iterations`` on the saved
    state is 0 so the build loop is empty and ``run()`` returns
    immediately after recording the pending feedback.
    """
    (tmp_path / "task.yaml").write_text("Do it.\n", encoding="utf-8")
    (tmp_path / "plan.md").write_text("# Plan\n", encoding="utf-8")
    (tmp_path / "acceptance-criteria.md").write_text("- [ ] x\n",
                                                     encoding="utf-8")
    (tmp_path / "progress.yaml").write_text(
        "plan_stage: []\nbuild_stage: []\nhuman_feedback: []\n",
        encoding="utf-8",
    )
    (tmp_path / "status.md").write_text("", encoding="utf-8")
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(task_path=str(tmp_path / "task.yaml"),
                             num_iterations=0,
                             next_iteration_index=next_iteration_index,
                             stage=module.STAGE_CODER))


def _stub_workflow_agents(workflow):
    """Replace the workflow's AgentLayers with no-op stubs.

    Safe because the seeded checkpoint puts the workflow in the build
    phase with ``num_iterations=0`` — no agent should actually be invoked.
    """

    class _Stub:

        def __call__(self, _prompt):
            return None

        def __exit__(self, *_):
            return None

    stub = _Stub()
    workflow.coder = stub
    workflow.reviewer = stub
    workflow.qa = stub
    workflow.plan_drafter = stub
    workflow.plan_reviewer = stub


def _run_workflow_with_feedback(module, tmp_path, feedback):
    """Construct, run, and close a workflow with the given ``--feedback``.

    Caller must have already seeded a STAGE_CODER + num_iterations=0
    checkpoint via ``_seed_build_phase_resume`` (unless the test is
    explicitly exercising the plan-stage path).
    """
    workflow = module.AgentTeamWorkflow(workspace=tmp_path, feedback=feedback)
    _stub_workflow_agents(workflow)
    # Defense in depth: skip the plan loop unconditionally so a future
    # state change does not turn this test into an infinite loop.
    workflow._run_plan_phase = lambda *_a, **_kw: None
    workflow.num_iterations = 0
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()


def _load_progress_yaml(path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_feedback_text_appends_to_progress_yaml_on_resume(tmp_path):
    """Resuming with ``--feedback "..."`` appends a ``human_feedback`` entry
    stamped with the upcoming build iteration, without disturbing prior
    plan/build entries.
    """
    module = _load_module()
    # Seed prior plan + build entries so we can verify they survive.
    seeded = ("plan_stage:\n"
              "  - iteration: 1\n"
              "    agent: plan_drafter\n"
              "    timestamp: \"t1\"\n"
              "    summary: drafted\n"
              "    decision: HUMAN_APPROVED\n"
              "build_stage:\n"
              "  - iteration: 1\n"
              "    agent: coder\n"
              "    timestamp: \"t2\"\n"
              "    summary: implemented v1\n"
              "human_feedback: []\n")
    (tmp_path / "task.yaml").write_text("Do the thing.\n", encoding="utf-8")
    (tmp_path / "plan.md").write_text("# Plan\n", encoding="utf-8")
    (tmp_path / "acceptance-criteria.md").write_text("- [ ] x\n",
                                                     encoding="utf-8")
    (tmp_path / "progress.yaml").write_text(seeded, encoding="utf-8")
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(task_path=str(tmp_path / "task.yaml"),
                             num_iterations=0,
                             next_iteration_index=1,
                             stage=module.STAGE_CODER))

    _run_workflow_with_feedback(module,
                                tmp_path,
                                feedback="please add streaming output")

    data = _load_progress_yaml(tmp_path / "progress.yaml")
    # Prior plan/build entries are preserved.
    assert len(data["plan_stage"]) == 1
    assert data["plan_stage"][0]["agent"] == "plan_drafter"
    assert len(data["build_stage"]) == 1
    assert data["build_stage"][0]["agent"] == "coder"
    # The new human-feedback entry lands with the upcoming iteration and
    # the active stage.
    assert len(data["human_feedback"]) == 1
    fb = data["human_feedback"][0]
    assert fb["summary"].strip() == "please add streaming output"
    assert fb["iteration"] == 2  # next_iteration_index (1) + 1
    assert fb["stage"] == "build_stage"
    assert "T" in fb["timestamp"]


def test_feedback_path_reads_file_contents(tmp_path):
    """When ``--feedback`` points to a real file, its contents are used."""
    module = _load_module()
    feedback_file = tmp_path / "feedback.txt"
    feedback_file.write_text("multi\nline feedback\n", encoding="utf-8")
    _seed_build_phase_resume(module, tmp_path)

    _run_workflow_with_feedback(module, tmp_path, feedback=str(feedback_file))

    data = _load_progress_yaml(tmp_path / "progress.yaml")
    assert len(data["human_feedback"]) == 1
    # Trailing whitespace is trimmed but interior newlines are preserved.
    assert data["human_feedback"][0]["summary"].strip() == (
        "multi\nline feedback")


def test_feedback_resolves_to_text_when_path_does_not_exist(tmp_path):
    """A ``--feedback`` value that is not a path is treated as literal text."""
    module = _load_module()
    _seed_build_phase_resume(module, tmp_path)

    _run_workflow_with_feedback(module,
                                tmp_path,
                                feedback="not/a/real/file.txt")

    data = _load_progress_yaml(tmp_path / "progress.yaml")
    assert len(data["human_feedback"]) == 1
    assert data["human_feedback"][0]["summary"].strip() == "not/a/real/file.txt"


def test_feedback_empty_input_records_nothing(tmp_path):
    """Whitespace-only ``--feedback`` is a no-op; no entry is appended."""
    module = _load_module()
    _seed_build_phase_resume(module, tmp_path)

    _run_workflow_with_feedback(module, tmp_path, feedback="   \n")

    data = _load_progress_yaml(tmp_path / "progress.yaml")
    assert data["human_feedback"] == []


def test_feedback_accumulates_across_invocations(tmp_path):
    """Re-running with another ``--feedback`` appends a new entry rather
    than overwriting the previous one — supporting iterative course
    correction.

    Real usage: the user Ctrl-Cs mid-iteration, leaving ``state.done=False``;
    re-running with another ``--feedback`` appends. Between the two
    invocations here we reset ``state.done`` to mimic that "stopped but
    not finished" state.
    """
    module = _load_module()
    _seed_build_phase_resume(module, tmp_path)

    _run_workflow_with_feedback(module, tmp_path, feedback="first feedback")

    # Mimic Ctrl-C: the previous run finished its budget and flipped
    # done=True, but a real interruption mid-iteration leaves it False.
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    state.done = False
    module.save_state(tmp_path / module.STATE_FILENAME, state)

    _run_workflow_with_feedback(module, tmp_path, feedback="second feedback")

    data = _load_progress_yaml(tmp_path / "progress.yaml")
    assert [e["summary"].strip() for e in data["human_feedback"]
            ] == ["first feedback", "second feedback"]


def test_resume_done_without_feedback_is_noop(tmp_path):
    """Re-running a completed workflow with no ``--feedback`` short-circuits:
    no agent runs, no feedback is recorded, the checkpoint stays done."""
    module = _load_module()
    _seed_build_phase_resume(module, tmp_path, next_iteration_index=2)
    # Mark the checkpoint as already-completed.
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    state.done = True
    module.save_state(tmp_path / module.STATE_FILENAME, state)

    workflow = module.AgentTeamWorkflow(workspace=tmp_path, num_iterations=5)
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # No agent ran; the checkpoint is left intact.
    assert trace == []
    after = module.load_state(tmp_path / module.STATE_FILENAME)
    assert after.done is True
    assert after.next_iteration_index == 2
    assert after.stage == module.STAGE_CODER
    data = _load_progress_yaml(tmp_path / "progress.yaml")
    assert data["human_feedback"] == []


def test_resume_done_with_feedback_re_engages_build_phase(tmp_path):
    """Re-running a completed workflow with ``--feedback`` flips done back
    to False, records the feedback, and lets the build phase pick up at
    the next iteration to address it."""
    module = _load_module()
    (tmp_path / "task.yaml").write_text("Do it.\n", encoding="utf-8")
    (tmp_path / "plan.md").write_text("# Plan\n", encoding="utf-8")
    (tmp_path / "acceptance-criteria.md").write_text("- [ ] x\n",
                                                     encoding="utf-8")
    (tmp_path / "progress.yaml").write_text(
        "plan_stage: []\nbuild_stage: []\nhuman_feedback: []\n",
        encoding="utf-8",
    )
    (tmp_path / "status.md").write_text("", encoding="utf-8")
    # QA accepted at iter 2 of 5: state.done=True, next_iteration_index=2,
    # stage=STAGE_CODER (the post-APPROVE bookkeeping in run()).
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(task_path=str(tmp_path / "task.yaml"),
                             num_iterations=5,
                             next_iteration_index=2,
                             stage=module.STAGE_CODER,
                             done=True))

    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=5,
        feedback="please address X",
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Build phase re-engaged at iter 3; default stub decisions APPROVE +
    # APPROVE terminate the run on the first new iteration.
    build_trace = [t for t in trace if not t[1].startswith("plan_")]
    assert build_trace == [
        (3, "coder"),
        (3, "reviewer"),
        (3, "qa"),
    ]
    # The new feedback landed in human_feedback, stamped with the upcoming
    # iteration (next_iteration_index + 1 == 3).
    data = _load_progress_yaml(tmp_path / "progress.yaml")
    assert len(data["human_feedback"]) == 1
    fb = data["human_feedback"][0]
    assert fb["summary"].strip() == "please address X"
    assert fb["iteration"] == 3
    assert fb["stage"] == "build_stage"


def test_resume_done_with_feedback_checkpoints_done_false_before_running(
        tmp_path):
    """The done→not-done flip must hit disk before any agent runs so a
    crash mid-iteration doesn't strand the workflow in the done branch."""
    module = _load_module()
    (tmp_path / "task.yaml").write_text("Do it.\n", encoding="utf-8")
    (tmp_path / "plan.md").write_text("# Plan\n", encoding="utf-8")
    (tmp_path / "acceptance-criteria.md").write_text("- [ ] x\n",
                                                     encoding="utf-8")
    (tmp_path / "progress.yaml").write_text(
        "plan_stage: []\nbuild_stage: []\nhuman_feedback: []\n",
        encoding="utf-8",
    )
    (tmp_path / "status.md").write_text("", encoding="utf-8")
    # num_iterations=2 with next_iteration_index=1 → range(1, 2) runs
    # exactly one build iteration regardless of reviewer/qa decisions.
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(task_path=str(tmp_path / "task.yaml"),
                             num_iterations=2,
                             next_iteration_index=1,
                             stage=module.STAGE_CODER,
                             done=True))

    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=2,
        feedback="address Y",
    )
    observed: list[bool] = []

    def coder(_iteration: int):
        observed.append(
            module.load_state(tmp_path / module.STATE_FILENAME).done)

    workflow._run_coder = coder
    workflow._run_reviewer = lambda _it: None
    workflow._run_qa = lambda _it: None
    workflow._reset_coder = lambda: None
    workflow._reset_reviewer = lambda: None
    workflow._run_plan_drafter = lambda *a, **kw: pytest.fail(
        "plan phase must not run when resuming a completed build")
    workflow._run_plan_reviewer = lambda *a, **kw: pytest.fail(
        "plan phase must not run when resuming a completed build")

    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # When the coder ran, the on-disk checkpoint already reflected
    # done=False — set before any agent was invoked.
    assert observed == [False]


def test_resume_done_with_feedback_extends_budget_when_exhausted(tmp_path):
    """Re-engaging a budget-exhausted completed run with ``--feedback``
    must grant a fresh budget so the build loop actually runs.

    Without the extension, ``range(next_iteration_index, num_iterations)``
    is empty when the prior run finished at the iteration cap, so the
    feedback would be appended and the workflow immediately marked done
    again — contradicting the README promise that ``--feedback``
    re-engages completed workflows.
    """
    module = _load_module()
    (tmp_path / "task.yaml").write_text("Do it.\n", encoding="utf-8")
    (tmp_path / "plan.md").write_text("# Plan\n", encoding="utf-8")
    (tmp_path / "acceptance-criteria.md").write_text("- [ ] x\n",
                                                     encoding="utf-8")
    (tmp_path / "progress.yaml").write_text(
        "plan_stage: []\nbuild_stage: []\nhuman_feedback: []\n",
        encoding="utf-8",
    )
    (tmp_path / "status.md").write_text("", encoding="utf-8")
    # Budget exhaustion: the prior 3-iteration run reached iter 3 of 3
    # without QA accepting, so ``run`` marked the workflow done with
    # next_iteration_index == num_iterations.
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(task_path=str(tmp_path / "task.yaml"),
                             num_iterations=3,
                             next_iteration_index=3,
                             stage=module.STAGE_CODER,
                             done=True))

    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=3,
        feedback="please address X",
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Build phase re-engaged at iter 4; default stub decisions APPROVE +
    # APPROVE terminate the run on the first new iteration.
    build_trace = [t for t in trace if not t[1].startswith("plan_")]
    assert build_trace == [
        (4, "coder"),
        (4, "reviewer"),
        (4, "qa"),
    ]
    # The checkpoint reflects the extended budget so a subsequent resume
    # has a coherent view of what was actually run.
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.num_iterations == 6  # 3 prior + 3 fresh
    assert state.next_iteration_index == 4
    # The feedback landed at the upcoming iteration (3 + 1 == 4).
    data = _load_progress_yaml(tmp_path / "progress.yaml")
    assert len(data["human_feedback"]) == 1
    assert data["human_feedback"][0]["iteration"] == 4


def test_feedback_long_literal_falls_back_when_path_probe_errors(tmp_path):
    """A long literal ``--feedback`` value that exceeds the OS path-length
    limit must be treated as text, not raise ``OSError`` from the
    path-existence probe.

    ``--feedback`` advertises that the value is either the literal text
    or a path to a file. ``Path.is_file`` raises ``OSError: [Errno 36]
    File name too long`` on values longer than ``NAME_MAX`` on most
    Linux filesystems; the resolver must catch that and fall back to
    treating the value as literal text.
    """
    module = _load_module()
    # 5000 chars is well above NAME_MAX (255) on every common filesystem
    # but short enough that a literal feedback note is still plausible.
    long_text = "x" * 5000
    _seed_build_phase_resume(module, tmp_path)

    _run_workflow_with_feedback(module, tmp_path, feedback=long_text)

    data = _load_progress_yaml(tmp_path / "progress.yaml")
    assert len(data["human_feedback"]) == 1
    assert data["human_feedback"][0]["summary"].strip() == long_text


def test_resolve_feedback_input_handles_overlong_string():
    """Direct unit test for the path-probe ``OSError`` guard.

    ``_resolve_feedback_input`` is called with raw CLI values; if the
    user pastes a long free-form note, the ``Path(...).is_file`` probe
    can raise ``OSError`` before the fallback-to-literal branch is
    reached. The resolver must return the literal text in that case.
    """
    module = _load_module()
    long_text = "y" * 5000
    assert module.AgentTeamWorkflow._resolve_feedback_input(
        long_text) == long_text


def test_feedback_iteration_uses_plan_next_when_in_plan_stage(tmp_path):
    """If the workflow is mid-plan when --feedback lands, the entry's
    ``stage`` reflects the plan phase and the iteration is the upcoming
    plan iteration. (Build agents won't read it until the build phase,
    but the metadata stays honest.)"""
    module = _load_module()
    (tmp_path / "task.yaml").write_text("Do it.\n", encoding="utf-8")
    (tmp_path / "plan.md").write_text("# Plan\n", encoding="utf-8")
    (tmp_path / "acceptance-criteria.md").write_text("- [ ] x\n",
                                                     encoding="utf-8")
    (tmp_path / "progress.yaml").write_text(
        "plan_stage: []\nbuild_stage: []\nhuman_feedback: []\n",
        encoding="utf-8",
    )
    (tmp_path / "status.md").write_text("", encoding="utf-8")
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(task_path=str(tmp_path / "task.yaml"),
                             num_iterations=0,
                             plan_next_iteration_index=2,
                             stage=module.STAGE_PLAN_DRAFTER))

    _run_workflow_with_feedback(module, tmp_path, feedback="reconsider X")

    data = _load_progress_yaml(tmp_path / "progress.yaml")
    assert len(data["human_feedback"]) == 1
    assert data["human_feedback"][0]["stage"] == "plan_stage"
    assert data["human_feedback"][0]["iteration"] == 3  # 2 + 1


def test_plan_reviewer_reject_loops_back_to_drafter(tmp_path):
    """PlanReviewer REJECT must loop back to PlanDrafter without entering
    the human-review stage."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        plan_human_review_enabled=True,
    )
    trace = _stub_agents(
        workflow,
        plan_reviewer_decisions=["REJECT", "APPROVE"],
    )
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Plan iter 1: drafter then reviewer REJECT → loop.
    # Plan iter 2: drafter then reviewer APPROVE → human → coder/reviewer/qa.
    assert trace == [
        (1, "plan_drafter:draft"),
        (1, "plan_reviewer"),
        (2, "plan_drafter:draft"),
        (2, "plan_reviewer"),
        (2, "plan_drafter:human"),
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True


def test_plan_human_polish_loop_does_not_rerun_plan_reviewer(tmp_path):
    """When the human asks for changes (plan_drafter decision != HUMAN_APPROVED),
    the orchestrator re-invokes the PlanDrafter in human mode without
    rerunning the AI PlanReviewer."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        plan_human_review_enabled=True,
    )
    # Drafter draft phase: DRAFT_READY (default).
    # Drafter human phase: POLISHING twice, then HUMAN_APPROVED.
    trace = _stub_agents(
        workflow,
        plan_drafter_decisions={
            "draft": ["DRAFT_READY"],
            "human": ["POLISHING", "POLISHING", "HUMAN_APPROVED"],
        },
    )
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Exactly one plan_reviewer call across the whole plan phase.
    plan_reviewer_calls = [s for _, s in trace if s == "plan_reviewer"]
    assert len(plan_reviewer_calls) == 1
    # Three human-mode invocations until HUMAN_APPROVED.
    human_calls = [s for _, s in trace if s == "plan_drafter:human"]
    assert len(human_calls) == 3
    # Build phase still runs once, ending with QA APPROVE.
    assert trace[-3:] == [(1, "coder"), (1, "reviewer"), (1, "qa")]


def test_reviewer_reject_loops_back_without_running_qa(tmp_path):
    """Reviewer REJECT must skip QA and loop to the next coder pass."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(workspace=tmp_path, num_iterations=2)
    # iter 1: REJECT → skip QA → next iter
    # iter 2: APPROVE + APPROVE → done
    trace = _stub_agents(
        workflow,
        reviewer_decisions=["REJECT", "APPROVE"],
        qa_decisions=["APPROVE"],
    )
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Drop the plan-phase prefix; assertions below focus on the build
    # phase loop semantics.
    build_trace = [t for t in trace if not t[1].startswith("plan_")]
    assert build_trace == [
        (1, "coder"),
        (1, "reviewer"),
        (2, "coder"),
        (2, "reviewer"),
        (2, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True


def test_qa_reject_loops_back_to_coder(tmp_path):
    """QA REJECT must loop back to the coder (reviewer re-checks)."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(workspace=tmp_path, num_iterations=2)
    trace = _stub_agents(
        workflow,
        reviewer_decisions=["APPROVE", "APPROVE"],
        qa_decisions=["REJECT", "APPROVE"],
    )
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    build_trace = [t for t in trace if not t[1].startswith("plan_")]
    assert build_trace == [
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
        (2, "coder"),
        (2, "reviewer"),
        (2, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True


def test_qa_approve_ends_workflow_even_with_budget_remaining(tmp_path):
    """APPROVE at iter 1 stops the loop even if num_iterations=5."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(workspace=tmp_path, num_iterations=5)
    trace = _stub_agents(workflow)  # defaults: APPROVE + APPROVE + score 9.5
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    build_trace = [t for t in trace if not t[1].startswith("plan_")]
    assert build_trace == [
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True
    assert state.next_iteration_index == 1


def test_qa_approve_below_min_score_loops_back(tmp_path):
    """An APPROVE with score below min_score is downgraded to a loop-back."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=3,
        min_score=8.0,
    )
    # iter 1: QA APPROVE but score 7 < 8 → must loop, not terminate.
    # iter 2: QA APPROVE with score 9 ≥ 8 → done.
    trace = _stub_agents(
        workflow,
        qa_decisions=["APPROVE", "APPROVE"],
        qa_scores=[7.0, 9.0],
    )
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    build_trace = [t for t in trace if not t[1].startswith("plan_")]
    assert build_trace == [
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
        (2, "coder"),
        (2, "reviewer"),
        (2, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True
    assert state.next_iteration_index == 2


def test_min_score_zero_disables_score_gate(tmp_path):
    """min_score=0 means the bool decision alone terminates the run."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=3,
        min_score=0.0,
    )
    # QA APPROVE with a very low score — with the gate off, APPROVE wins.
    trace = _stub_agents(workflow, qa_decisions=["APPROVE"], qa_scores=[2.0])
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    build_trace = [t for t in trace if not t[1].startswith("plan_")]
    assert build_trace == [
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True


def test_status_tools_wired_only_to_coder_and_reviewer(tmp_path):
    """Coder and Reviewer must have ``update_status`` / ``read_status``;
    PlanDrafter, PlanReviewer, and QA must not.
    """
    module = _load_module()
    workflow = _make_workflow(module, tmp_path)
    try:

        def _tool_names(layer):
            return [t.name for t in (layer.config.backend.tools or [])]

        coder_names = _tool_names(workflow.coder)
        reviewer_names = _tool_names(workflow.reviewer)
        plan_drafter_names = _tool_names(workflow.plan_drafter)
        plan_reviewer_names = _tool_names(workflow.plan_reviewer)
        qa_names = _tool_names(workflow.qa)

        assert "update_status" in coder_names
        assert "read_status" in coder_names
        assert "update_status" in reviewer_names
        assert "read_status" in reviewer_names

        assert "update_status" not in plan_drafter_names
        assert "read_status" not in plan_drafter_names
        assert "update_status" not in plan_reviewer_names
        assert "read_status" not in plan_reviewer_names
        assert "update_status" not in qa_names
        assert "read_status" not in qa_names
    finally:
        workflow.close()


def test_coder_human_input_disabled_by_default(tmp_path):
    """Build-stage human review is OFF by default — the Coder must NOT
    have ``human_input_enabled`` unless ``--build-human-review`` is
    passed."""
    module = _load_module()
    workflow = _make_workflow(module, tmp_path)
    try:
        assert workflow.coder.config.human_input_enabled is False
    finally:
        workflow.close()


def test_coder_human_input_enabled_with_build_human_review(tmp_path):
    """``--build-human-review`` (constructor arg
    ``build_human_review_enabled=True``) gives the Coder
    ``human_input_enabled`` so it can call ``ask_human``."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        build_human_review_enabled=True,
    )
    try:
        assert workflow.coder.config.human_input_enabled is True
    finally:
        workflow.close()


def test_coder_human_input_independent_of_plan_human_review(tmp_path):
    """The two flags are independent: turning off plan-stage human
    review does NOT disable the Coder's ``ask_human`` if the build-stage
    opt-in is set; and turning on plan-stage human review does NOT
    enable the Coder's ``ask_human`` on its own."""
    module = _load_module()

    plan_off_build_on = module.AgentTeamWorkflow(
        workspace=tmp_path / "plan_off_build_on",
        plan_human_review_enabled=False,
        build_human_review_enabled=True,
    )
    try:
        assert plan_off_build_on.coder.config.human_input_enabled is True
    finally:
        plan_off_build_on.close()

    plan_on_build_off = module.AgentTeamWorkflow(
        workspace=tmp_path / "plan_on_build_off",
        plan_human_review_enabled=True,
        build_human_review_enabled=False,
    )
    try:
        assert plan_on_build_off.coder.config.human_input_enabled is False
    finally:
        plan_on_build_off.close()


def test_plan_drafter_always_has_human_input_enabled(tmp_path):
    """PlanDrafter's flag is unconditional regardless of either review
    flag; PlanReviewer/Reviewer/QA stay off in all combinations."""
    module = _load_module()
    combos = [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ]
    for i, (plan_flag, build_flag) in enumerate(combos):
        workflow = module.AgentTeamWorkflow(
            workspace=tmp_path / f"combo_{i}",
            plan_human_review_enabled=plan_flag,
            build_human_review_enabled=build_flag,
        )
        try:
            assert workflow.plan_drafter.config.human_input_enabled is True
            for layer in (workflow.plan_reviewer, workflow.reviewer,
                          workflow.qa):
                assert layer.config.human_input_enabled is False
        finally:
            workflow.close()


def test_reset_coder_preserves_human_input_flag(tmp_path):
    """After ``_reset_coder``, the rebuilt Coder must keep
    ``human_input_enabled`` aligned with ``build_human_review_enabled``;
    otherwise mid-run resets would silently drop the ``ask_human``
    tool."""
    module = _load_module()

    enabled = module.AgentTeamWorkflow(
        workspace=tmp_path / "on",
        build_human_review_enabled=True,
    )
    try:
        assert enabled.coder.config.human_input_enabled is True
        enabled._reset_coder()
        assert enabled.coder.config.human_input_enabled is True
    finally:
        enabled.close()

    disabled = _make_workflow(module, tmp_path / "off")
    try:
        assert disabled.coder.config.human_input_enabled is False
        disabled._reset_coder()
        assert disabled.coder.config.human_input_enabled is False
    finally:
        disabled.close()


def test_coder_prompt_documents_ask_human():
    """Coder system prompt must mention ``ask_human``, the strict
    last-resort framing, the build-stage opt-in flag, and the no-reply
    contract."""
    module = _load_module()
    text = module.DEFAULT_PROMPTS.coder
    assert "ask_human" in text
    assert "Asking the human as a last resort" in text
    assert "(no response from human)" in text
    assert "--build-human-review" in text
    # The strict framing must survive prompt edits — the default
    # behavior is to NOT ask. If a refactor softens this, the test
    # should flag it.
    assert "default: do not call it" in text.lower()


def test_coder_and_reviewer_required_tools_include_status_update(tmp_path):
    """The composed Stop hook must enforce both progress and status calls."""
    module = _load_module()
    workflow = _make_workflow(module, tmp_path)
    try:
        # Each per-tool hook is a separate matcher; with two required tools
        # we expect two matchers stacked together (composition = AND).
        coder_hooks = workflow.coder.config.backend.hooks
        reviewer_hooks = workflow.reviewer.config.backend.hooks
        plan_drafter_hooks = workflow.plan_drafter.config.backend.hooks
        plan_reviewer_hooks = workflow.plan_reviewer.config.backend.hooks
        qa_hooks = workflow.qa.config.backend.hooks

        assert coder_hooks is not None
        assert len(coder_hooks["Stop"]) == 2
        assert reviewer_hooks is not None
        assert len(reviewer_hooks["Stop"]) == 2

        # PlanDrafter, PlanReviewer, and QA each have a single required
        # tool — one matcher.
        assert plan_drafter_hooks is not None
        assert len(plan_drafter_hooks["Stop"]) == 1
        assert plan_reviewer_hooks is not None
        assert len(plan_reviewer_hooks["Stop"]) == 1
        assert qa_hooks is not None
        assert len(qa_hooks["Stop"]) == 1
    finally:
        workflow.close()


def test_compose_required_tools_hooks_handles_empty_and_single(tmp_path):
    """Helper returns ``None`` for empty and one matcher for a single tool."""
    module = _load_module()

    assert module._compose_required_tools_hooks([]) is None

    single = module._compose_required_tools_hooks(
        ["append_plan_drafter_progress"])
    assert single is not None
    assert len(single["Stop"]) == 1

    pair = module._compose_required_tools_hooks(
        ["append_coder_progress", "update_status"])
    assert pair is not None
    assert len(pair["Stop"]) == 2


def test_qa_uses_stateless_session(tmp_path):
    """QA's session must be stateless so each iteration starts fresh."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(workspace=tmp_path)
    try:
        assert workflow.qa.config.session.mode == "stateless"
        # Coder, reviewer, plan_drafter, and plan_reviewer remain
        # persistent (their context is recycled by the reset interval,
        # not rebuilt every turn).
        assert workflow.coder.config.session.mode == "persistent"
        assert workflow.reviewer.config.session.mode == "persistent"
        assert workflow.plan_drafter.config.session.mode == "persistent"
        assert workflow.plan_reviewer.config.session.mode == "persistent"
    finally:
        workflow.close()


def test_budget_exhaustion_marks_done_without_approve(tmp_path):
    """When QA never APPROVEs within the budget, state.done still flips to True."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(workspace=tmp_path, num_iterations=2)
    trace = _stub_agents(
        workflow,
        reviewer_decisions=["APPROVE", "APPROVE"],
        qa_decisions=["REJECT", "REJECT"],
    )
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    build_trace = [t for t in trace if not t[1].startswith("plan_")]
    assert build_trace == [
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
        (2, "coder"),
        (2, "reviewer"),
        (2, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True
    assert state.next_iteration_index == 2


def test_fresh_run_checkpoints_stage_between_agents(tmp_path):
    """After the coder finishes, state.stage advances to 'reviewer'."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(workspace=tmp_path, num_iterations=3)

    captured: list[str] = []
    trace: list[tuple[int, str]] = []

    def plan_drafter(iteration: int, mode: str):
        trace.append((iteration, f"plan_drafter:{mode}"))
        # Stub minimal plan_drafter entry so the orchestrator's decision
        # readers see HUMAN_APPROVED in the human phase.
        decision = "HUMAN_APPROVED" if mode == "human" else "DRAFT_READY"
        data = progress_module.read_progress(workflow.progress_path)
        data["plan_stage"].append({
            "iteration": iteration,
            "agent": "plan_drafter",
            "decision": decision,
        })
        progress_module.write_progress(workflow.progress_path, data)

    def plan_reviewer(iteration: int):
        trace.append((iteration, "plan_reviewer"))
        data = progress_module.read_progress(workflow.progress_path)
        data["plan_stage"].append({
            "iteration": iteration,
            "agent": "plan_reviewer",
            "decision": "APPROVE",
        })
        progress_module.write_progress(workflow.progress_path, data)

    def coder(iteration: int):
        trace.append((iteration, "coder"))
        captured.append(
            module.load_state(tmp_path /
                              module.STATE_FILENAME).stage)  # pre-coder

    def reviewer(iteration: int):
        trace.append((iteration, "reviewer"))
        # After coder completes, the checkpoint must now point at the
        # reviewer so a crash here resumes at the reviewer, not the coder.
        captured.append(
            module.load_state(tmp_path / module.STATE_FILENAME).stage)
        raise KeyboardInterrupt

    workflow._run_plan_drafter = plan_drafter
    workflow._run_plan_reviewer = plan_reviewer
    workflow._run_coder = coder
    workflow._run_reviewer = reviewer
    workflow._run_qa = lambda _it: None
    workflow._reset_coder = lambda: None
    workflow._reset_reviewer = lambda: None

    try:
        with pytest.raises(KeyboardInterrupt):
            workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # captured[0]: stage recorded *before* the coder for iteration 1 ran —
    # the plan phase has already completed, so stage is STAGE_CODER.
    # captured[1]: stage recorded *after* the coder finished, so the
    # reviewer is next.
    assert captured == [module.STAGE_CODER, module.STAGE_REVIEWER]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.stage == module.STAGE_REVIEWER
    assert state.next_iteration_index == 0  # iteration 1 is still incomplete


def test_fresh_run_checkpoints_plan_drafter_stage_before_running(tmp_path):
    """Stage is ``plan_drafter`` on disk before the initial PlanDrafter runs."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(workspace=tmp_path, num_iterations=1)

    observed: list[str] = []

    def plan_drafter(iteration: int, mode: str):
        # Checkpoint written in _init_state *before* the PlanDrafter call
        # must reflect the plan_drafter stage so a crash here resumes
        # there.
        observed.append(
            module.load_state(tmp_path / module.STATE_FILENAME).stage)
        raise RuntimeError("plan_drafter blew up")

    workflow._run_plan_drafter = plan_drafter
    try:
        with pytest.raises(RuntimeError, match="plan_drafter blew up"):
            workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    assert observed == [module.STAGE_PLAN_DRAFTER]
    # Even after the crash, the on-disk state should still be
    # STAGE_PLAN_DRAFTER (no advance happened).
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.stage == module.STAGE_PLAN_DRAFTER


def test_resume_from_plan_human_skips_drafter_and_reviewer(tmp_path):
    """A crash in the human stage resumes at the human stage; drafter/reviewer
    don't re-run."""
    module = _load_module()
    # Pre-populate as if drafter+reviewer already ran for plan iteration 1.
    (tmp_path / "plan.md").write_text("# plan\n", encoding="utf-8")
    (tmp_path / "progress.yaml").write_text(
        "plan_stage:\n"
        "  - iteration: 1\n"
        "    agent: plan_drafter\n"
        "    decision: DRAFT_READY\n"
        "  - iteration: 1\n"
        "    agent: plan_reviewer\n"
        "    decision: APPROVE\n"
        "build_stage: []\n",
        encoding="utf-8",
    )
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(
            task_path=str(tmp_path / "task.yaml"),
            num_iterations=1,
            plan_next_iteration_index=0,
            stage=module.STAGE_PLAN_HUMAN,
        ),
    )

    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        plan_human_review_enabled=True,
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Plan phase resumes directly at the human stage; build phase runs
    # iteration 1 to completion.
    assert trace == [
        (1, "plan_drafter:human"),
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True


def test_resume_from_reviewer_skips_coder(tmp_path):
    module = _load_module()
    # Pre-populate the workspace as if iteration 1's coder already ran.
    (tmp_path / "plan.md").write_text("plan\n", encoding="utf-8")
    (tmp_path / "progress.yaml").write_text(
        "plan_stage: []\n"
        "build_stage:\n"
        "  - iteration: 1\n"
        "    agent: coder\n",
        encoding="utf-8")
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(
            task_path=str(tmp_path / "task.yaml"),
            num_iterations=2,
            next_iteration_index=0,
            stage=module.STAGE_REVIEWER,
        ),
    )

    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=2,
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Coder must not run for iteration 1 (already done); reviewer + qa run
    # for iteration 1, and QA APPROVE ends the loop. Plan phase is also
    # skipped because the resumed stage is already in the build phase.
    assert trace == [
        (1, "reviewer"),
        (1, "qa"),
    ]

    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True


def test_both_presets_skip_plan_phase_and_start_at_coder(tmp_path):
    """Supplying both ``plan`` and ``acceptance_criteria`` on a fresh run
    must populate both files, set the initial stage to coder, and run
    only the build phase."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        plan="# pre-supplied plan\nstep one.",
        acceptance_criteria="- [ ] hello world prints",
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Plan-phase agents must not run; the build phase runs to completion.
    assert trace == [
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
    ]
    assert (tmp_path / "plan.md").read_text(
        encoding="utf-8") == ("# pre-supplied plan\nstep one.\n")
    assert (tmp_path / "acceptance-criteria.md").read_text(
        encoding="utf-8") == ("- [ ] hello world prints\n")
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True


def test_both_presets_initial_checkpoint_starts_at_coder(tmp_path):
    """Even before ``run`` is called, the first checkpoint must reflect
    that the plan phase is being skipped when both presets are supplied."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        plan="plan text",
        acceptance_criteria="- [ ] criteria text",
    )
    captured: list[str] = []

    def coder(_iteration: int):
        captured.append(
            module.load_state(tmp_path / module.STATE_FILENAME).stage)

    workflow._run_coder = coder
    workflow._run_reviewer = lambda _it: None
    workflow._run_qa = lambda _it: None
    workflow._reset_coder = lambda: None
    workflow._reset_reviewer = lambda: None
    # Plan-phase agents must never be invoked when both presets are set.
    workflow._run_plan_drafter = lambda *a, **kw: pytest.fail(
        "plan_drafter must not run when both presets are set")
    workflow._run_plan_reviewer = lambda *a, **kw: pytest.fail(
        "plan_reviewer must not run when both presets are set")

    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Pre-coder checkpoint shows STAGE_CODER, not STAGE_PLAN_DRAFTER.
    assert captured == [module.STAGE_CODER]


def test_only_plan_preset_runs_plan_phase(tmp_path):
    """Supplying only ``plan`` (no acceptance_criteria) must still run
    the plan phase so the PlanDrafter can generate
    `acceptance-criteria.md`. plan.md is materialized from the preset
    before the plan phase runs."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        plan="# pre-supplied plan\nstep one.",
        plan_human_review_enabled=True,
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Plan phase must run (drafter + reviewer + human + build phase).
    assert trace == [
        (1, "plan_drafter:draft"),
        (1, "plan_reviewer"),
        (1, "plan_drafter:human"),
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
    ]
    # plan.md was materialized from the preset before the plan phase ran.
    assert (tmp_path / "plan.md").read_text(
        encoding="utf-8") == ("# pre-supplied plan\nstep one.\n")
    # acceptance-criteria.md exists but is empty (the stub PlanDrafter
    # doesn't actually generate content; the real one would).
    assert (tmp_path /
            "acceptance-criteria.md").read_text(encoding="utf-8") == ""


def test_only_acceptance_criteria_preset_runs_plan_phase(tmp_path):
    """Symmetric: supplying only ``acceptance_criteria`` must still run
    the plan phase so the PlanDrafter can generate `plan.md`."""
    module = _load_module()
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        acceptance_criteria="- [ ] pre-supplied criterion",
        plan_human_review_enabled=True,
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    assert trace == [
        (1, "plan_drafter:draft"),
        (1, "plan_reviewer"),
        (1, "plan_drafter:human"),
        (1, "coder"),
        (1, "reviewer"),
        (1, "qa"),
    ]
    assert (tmp_path / "acceptance-criteria.md").read_text(
        encoding="utf-8") == "- [ ] pre-supplied criterion\n"
    assert (tmp_path / "plan.md").read_text(encoding="utf-8") == ""


def test_plan_arg_accepts_path_to_file(tmp_path):
    """Passing a path to an existing file copies that file's content into
    plan.md verbatim (mirrors how ``--task`` handles a path)."""
    module = _load_module()
    src = tmp_path / "external_plan.md"
    src.write_text("# from a file\n- bullet\n", encoding="utf-8")
    crit_src = tmp_path / "external_criteria.md"
    crit_src.write_text("- [ ] from-file criterion\n", encoding="utf-8")

    work = tmp_path / "ws"
    workflow = module.AgentTeamWorkflow(
        workspace=work,
        num_iterations=1,
        plan=str(src),
        acceptance_criteria=str(crit_src),
    )
    _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Both plan-phase outputs must match their source files byte-for-byte
    # after ``run`` materializes them.
    assert (work / "plan.md").read_text(
        encoding="utf-8") == ("# from a file\n- bullet\n")
    assert (work / "acceptance-criteria.md").read_text(
        encoding="utf-8") == ("- [ ] from-file criterion\n")


def test_acceptance_criteria_arg_accepts_path_to_file(tmp_path):
    """``--acceptance-criteria`` accepts a path to an existing file just
    like ``--plan`` does."""
    module = _load_module()
    src = tmp_path / "external_criteria.md"
    src.write_text("- [ ] one\n- [ ] two\n", encoding="utf-8")

    work = tmp_path / "ws"
    # Only the criteria preset is supplied; the plan phase still runs to
    # generate plan.md. We don't care about the trace here — just that
    # construction + materialization succeed.
    workflow = module.AgentTeamWorkflow(
        workspace=work,
        num_iterations=1,
        acceptance_criteria=str(src),
    )
    _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    assert (work / "acceptance-criteria.md").read_text(
        encoding="utf-8") == ("- [ ] one\n- [ ] two\n")


def test_presets_allow_existing_files_and_overwrite_them(tmp_path):
    """Stale plan.md and acceptance-criteria.md are allowed when their
    matching presets are set; ``run`` overwrites them."""
    module = _load_module()
    (tmp_path / "plan.md").write_text("# stale plan\n", encoding="utf-8")
    (tmp_path / "acceptance-criteria.md").write_text("- [ ] stale\n",
                                                     encoding="utf-8")

    # Construction must not raise — both guards relaxed because the
    # matching presets are provided.
    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        plan="# fresh plan",
        acceptance_criteria="- [ ] fresh",
    )
    _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    assert (tmp_path /
            "plan.md").read_text(encoding="utf-8") == "# fresh plan\n"
    assert (tmp_path / "acceptance-criteria.md").read_text(
        encoding="utf-8") == "- [ ] fresh\n"


def test_plan_arg_still_guards_progress_and_status(tmp_path):
    """``--plan`` only relaxes the plan.md guard; non-empty progress.yaml
    or status.md must still raise."""
    module = _load_module()
    (tmp_path / "progress.yaml").write_text("- iteration: 1\n",
                                            encoding="utf-8")
    with pytest.raises(FileExistsError, match="progress.yaml"):
        module.AgentTeamWorkflow(
            workspace=tmp_path,
            plan="# any plan",
        )


def test_plan_arg_alone_still_guards_acceptance_criteria(tmp_path):
    """``--plan`` does NOT relax the acceptance-criteria.md guard; a
    non-empty `acceptance-criteria.md` must still raise unless
    ``--acceptance-criteria`` is also set."""
    module = _load_module()
    (tmp_path / "acceptance-criteria.md").write_text("- [ ] stale\n",
                                                     encoding="utf-8")
    with pytest.raises(FileExistsError, match="acceptance-criteria.md"):
        module.AgentTeamWorkflow(
            workspace=tmp_path,
            plan="# any plan",
        )


def test_acceptance_criteria_arg_alone_still_guards_plan(tmp_path):
    """``--acceptance-criteria`` does NOT relax the plan.md guard
    symmetrically."""
    module = _load_module()
    (tmp_path / "plan.md").write_text("# stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="plan.md"):
        module.AgentTeamWorkflow(
            workspace=tmp_path,
            acceptance_criteria="- [ ] any",
        )


def test_plan_cli_flag_default_and_value(tmp_path):
    """``--plan`` parses to None by default and to the supplied string."""
    module = _load_cli_module()
    args = module._parse_args(_TASK_CLI)
    assert args.plan is None
    args = module._parse_args(_TASK_CLI + ["--plan", "/path/to/plan.md"])
    assert args.plan == "/path/to/plan.md"


def test_acceptance_criteria_cli_flag_default_and_value(tmp_path):
    """``--acceptance-criteria`` parses to None by default and to the
    supplied string."""
    module = _load_cli_module()
    args = module._parse_args(_TASK_CLI)
    assert args.acceptance_criteria is None
    args = module._parse_args(_TASK_CLI +
                              ["--acceptance-criteria", "/path/to/criteria.md"])
    assert args.acceptance_criteria == "/path/to/criteria.md"


def test_resume_from_qa_skips_coder_and_reviewer(tmp_path):
    module = _load_module()
    (tmp_path / "plan.md").write_text("plan\n", encoding="utf-8")
    (tmp_path / "progress.yaml").write_text(
        "plan_stage: []\n"
        "build_stage:\n"
        "  - iteration: 1\n"
        "    agent: coder\n"
        "  - iteration: 1\n"
        "    agent: reviewer\n"
        "    decision: APPROVE\n",
        encoding="utf-8")
    module.save_state(
        tmp_path / module.STATE_FILENAME,
        module.WorkflowState(
            task_path=str(tmp_path / "task.yaml"),
            num_iterations=2,
            next_iteration_index=0,
            stage=module.STAGE_QA,
        ),
    )

    workflow = module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=2,
    )
    trace = _stub_agents(workflow)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # Iteration 1 resumes directly at QA; QA ACCEPTs and ends the loop.
    assert trace == [
        (1, "qa"),
    ]
    state = module.load_state(tmp_path / module.STATE_FILENAME)
    assert state.done is True
