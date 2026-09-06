"""Integration tests for the ``--concurrent`` DAG execution path.

The concurrent branch of :meth:`AgentTeamWorkflow.run` is driven end-to-end
here with the *same* fake-backend harness Task 11 used for the per-node loop
(:class:`tests.workflows.agent_team.test_node_runner._FakeInvoker`) plus the
built-in :class:`~agent_flow.orchestration.NoOpIsolation`, so no real backend
or git is touched. A pre-supplied ``--plan`` carries a two-goal
``## Execution Graph`` block and ``--acceptance-criteria`` skips the plan
phase, so ``run`` drops straight into ``_run_concurrent_build``.

``workflow.run`` is synchronous and calls ``anyio.run(scheduler.run)``
internally, so these tests are plain ``def`` functions (no running event
loop) — an ``async def`` test would already own a loop and ``anyio.run``
would refuse to nest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_flow.workflows.agent_team import node_runner as node_runner_module
from agent_flow.workflows.agent_team import progress as progress_module
from agent_flow.workflows.agent_team import workflow as _workflow_module

from .test_node_runner import _FakeInvoker, _install_fake
from .test_workflow import _stub_agents, _write_task_yaml

# A plan.md whose ``## Execution Graph`` block declares two independent goal
# nodes. Goal nodes close on Reviewer APPROVE under the workflow-agnostic
# default policy (runs_qa=False), so the fake reviewer's single APPROVE drives
# each straight to DONE.
_TWO_GOAL_PLAN = """\
# Build Plan

Prose the planner writes above the machine-readable graph.

## Execution Graph

```yaml
nodes:
  - id: g1
    type: goal
  - id: g2
    type: goal
```
"""

# A plan.md with prose but NO ``## Execution Graph`` section — the concurrent
# path must reject it loudly.
_NO_GRAPH_PLAN = """\
# Build Plan

Just prose. No execution graph block here.
"""


def _make_concurrent_workflow(tmp_path: Path, *, plan: str, num_iterations: int = 5):
    return _workflow_module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=num_iterations,
        concurrent=True,
        plan=plan,
        acceptance_criteria="- [ ] the goals are built",
    )


def _node_progress_path(workspace: Path, node_id: str) -> Path:
    return workspace / "nodes" / node_runner_module.node_dir_slug(node_id) / "progress.yaml"


def test_concurrent_run_drives_both_goals_to_done(tmp_path, monkeypatch):
    """A concurrent run drives BOTH goals to DONE and finishes with state.done."""
    _install_fake(monkeypatch, _FakeInvoker(reviewer_decisions=["APPROVE"]))

    workflow = _make_concurrent_workflow(tmp_path, plan=_TWO_GOAL_PLAN)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    state = _workflow_module.load_state(tmp_path / _workflow_module.STATE_FILENAME)
    assert state.done is True

    # Both nodes recorded their own coder+reviewer turns in their PRIVATE files.
    for node_id in ("g1", "g2"):
        data = progress_module.read_progress(_node_progress_path(tmp_path, node_id))
        assert [e["agent"] for e in data["build_stage"]] == ["coder", "reviewer"]

    # The concurrent path never touches the shared-workspace progress.yaml as a
    # per-node coordination file — nodes write only under nodes/<id>/.
    shared = progress_module.read_progress(tmp_path / "progress.yaml")
    assert shared["build_stage"] == []


def test_concurrent_graph_state_checkpoint_shows_both_done(tmp_path, monkeypatch):
    """The scheduler's ``.graph_state.json`` checkpoint shows both nodes DONE."""
    _install_fake(monkeypatch, _FakeInvoker(reviewer_decisions=["APPROVE"]))

    workflow = _make_concurrent_workflow(tmp_path, plan=_TWO_GOAL_PLAN)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    checkpoint = tmp_path / ".graph_state.json"
    assert checkpoint.is_file()
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert data["states"] == {"g1": "done", "g2": "done"}


def test_concurrent_each_node_gets_its_own_sub_workspace(tmp_path, monkeypatch):
    """Each node wrote to its own ``nodes/<id>/`` sub-workspace, never shared."""
    _install_fake(monkeypatch, _FakeInvoker(reviewer_decisions=["APPROVE"]))

    workflow = _make_concurrent_workflow(tmp_path, plan=_TWO_GOAL_PLAN)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    ws_a = tmp_path / "nodes" / "g1"
    ws_b = tmp_path / "nodes" / "g2"
    assert (ws_a / "progress.yaml").is_file()
    assert (ws_a / "status.md").is_file()
    assert (ws_b / "progress.yaml").is_file()
    assert (ws_b / "status.md").is_file()

    # The fake coder stamps each node's private status.md with its own id.
    assert "g1" in (ws_a / "status.md").read_text(encoding="utf-8")
    assert "g1" not in (ws_b / "status.md").read_text(encoding="utf-8")
    assert "g2" in (ws_b / "status.md").read_text(encoding="utf-8")


def test_concurrent_missing_graph_raises_clear_error(tmp_path, monkeypatch):
    """``--concurrent`` with a plan.md lacking an Execution Graph raises clearly."""
    _install_fake(monkeypatch, _FakeInvoker(reviewer_decisions=["APPROVE"]))

    workflow = _make_concurrent_workflow(tmp_path, plan=_NO_GRAPH_PLAN)
    try:
        with pytest.raises(ValueError, match="Execution Graph"):
            workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # A rejected run never wrote a graph checkpoint.
    assert not (tmp_path / ".graph_state.json").exists()


def test_default_policy_runs_no_qa_for_any_type(tmp_path, monkeypatch):
    """The workflow-agnostic default policy runs no QA even for a ``stage`` type.

    agent_team is workflow-agnostic; its default ``policy_for_type`` returns
    ``runs_qa=False`` for every type, so a node typed ``stage`` still closes on
    Reviewer APPROVE (no QA turn). modeling_bringup injects the real
    stage/goal policy separately.
    """
    invoker = _FakeInvoker(reviewer_decisions=["APPROVE"])
    _install_fake(monkeypatch, invoker)

    stage_plan = (
        "# Plan\n\n## Execution Graph\n\n```yaml\nnodes:\n  - id: s1\n    type: stage\n```\n"
    )
    workflow = _make_concurrent_workflow(tmp_path, plan=stage_plan)
    try:
        workflow.run(_write_task_yaml(workflow.workspace))
    finally:
        workflow.close()

    # No QA turn ran for the stage node — the default policy runs no QA.
    roles = [role for (_nid, role, _it) in invoker.calls]
    assert roles == ["coder", "reviewer"]
    data = progress_module.read_progress(_node_progress_path(tmp_path, "s1"))
    assert [e["agent"] for e in data["build_stage"]] == ["coder", "reviewer"]


_ONE_GOAL_PLAN = """\
# Build Plan

## Execution Graph

```yaml
nodes:
  - id: g1
    type: goal
```
"""


def _make_replan_workflow(tmp_path: Path, *, num_iterations: int = 1, max_replan_rounds: int = 3):
    return _workflow_module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=num_iterations,
        concurrent=True,
        replan_on_qa=True,
        max_replan_rounds=max_replan_rounds,
        plan=_ONE_GOAL_PLAN,
        acceptance_criteria="- [ ] g1 built",
    )


def _graph_states(tmp_path: Path) -> dict:
    return json.loads((tmp_path / ".graph_state.json").read_text())["states"]


def test_concurrent_replan_retries_failed_node_to_done(tmp_path, monkeypatch):
    """A failed node + --replan-on-qa enters replan, resets the node, re-runs it to DONE."""
    # num_iterations=1: reviewer REJECT exhausts g1 to FAILED on pass 1; APPROVE → DONE on pass 2.
    _install_fake(monkeypatch, _FakeInvoker(reviewer_decisions=["REJECT", "APPROVE"]))
    wf = _make_replan_workflow(tmp_path)
    calls: list = []
    wf._run_plan_drafter = lambda *a, **kw: calls.append(kw.get("mode"))
    wf._latest_plan_drafter_decision = lambda: "POLISHING"
    try:
        wf.run(_write_task_yaml(wf.workspace))
    finally:
        wf.close()

    assert calls == ["replan"]  # exactly one replan round
    assert _graph_states(tmp_path)["g1"] == "done"  # retried to DONE
    assert _workflow_module.load_state(tmp_path / _workflow_module.STATE_FILENAME).done is True


def test_concurrent_replan_planner_done_terminates(tmp_path, monkeypatch):
    """PlanDrafter DONE ends the loop — no reconcile / re-run; the node stays FAILED."""
    _install_fake(monkeypatch, _FakeInvoker(reviewer_decisions=["REJECT"]))
    wf = _make_replan_workflow(tmp_path)
    calls: list = []
    wf._run_plan_drafter = lambda *a, **kw: calls.append(kw.get("mode"))
    wf._latest_plan_drafter_decision = lambda: "DONE"
    try:
        wf.run(_write_task_yaml(wf.workspace))
    finally:
        wf.close()

    assert calls == ["replan"]  # invoked once, then DONE broke the loop
    assert _graph_states(tmp_path)["g1"] == "failed"  # not re-run
    assert _workflow_module.load_state(tmp_path / _workflow_module.STATE_FILENAME).done is True


def test_concurrent_replan_stops_after_max_rounds(tmp_path, monkeypatch):
    """A node that keeps failing stops after ``max_replan_rounds`` replans (not infinite)."""
    _install_fake(monkeypatch, _FakeInvoker(reviewer_decisions=["REJECT"]))
    wf = _make_replan_workflow(tmp_path, max_replan_rounds=2)
    calls: list = []
    wf._run_plan_drafter = lambda *a, **kw: calls.append(kw.get("mode"))
    wf._latest_plan_drafter_decision = lambda: "POLISHING"
    try:
        wf.run(_write_task_yaml(wf.workspace))
    finally:
        wf.close()

    assert calls == ["replan", "replan"]  # capped at max_replan_rounds
    assert _workflow_module.load_state(tmp_path / _workflow_module.STATE_FILENAME).done is True


def test_concurrent_no_replan_when_flag_off(tmp_path, monkeypatch):
    """Without --replan-on-qa, a failed node ends the run in one pass (unchanged behavior)."""
    _install_fake(monkeypatch, _FakeInvoker(reviewer_decisions=["REJECT"]))
    wf = _workflow_module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        concurrent=True,
        replan_on_qa=False,
        plan=_ONE_GOAL_PLAN,
        acceptance_criteria="- [ ] g1 built",
    )
    wf._run_plan_drafter = lambda *a, **kw: pytest.fail(
        "replan must not run without --replan-on-qa"
    )
    try:
        wf.run(_write_task_yaml(wf.workspace))
    finally:
        wf.close()

    assert _graph_states(tmp_path)["g1"] == "failed"
    assert _workflow_module.load_state(tmp_path / _workflow_module.STATE_FILENAME).done is True


def test_concurrent_feedback_resume_enters_replan(tmp_path, monkeypatch):
    """A ``--feedback`` resume on the concurrent path enters replan.

    It folds the feedback via a replan turn (``feedback_triggered=True``) and
    re-runs the failed subtree.
    """
    from agent_flow.orchestration import GraphState, NodeState

    _install_fake(monkeypatch, _FakeInvoker(reviewer_decisions=["APPROVE"]))
    # A previously-finished concurrent run whose only node ended FAILED.
    (tmp_path / "plan.md").write_text(_ONE_GOAL_PLAN)
    (tmp_path / "acceptance-criteria.md").write_text("## g1\n- [ ] g1 built\n")
    _write_task_yaml(tmp_path)
    GraphState(states={"g1": NodeState.FAILED}, worktrees={}).save(tmp_path / ".graph_state.json")
    _workflow_module.save_state(
        tmp_path / _workflow_module.STATE_FILENAME,
        _workflow_module.WorkflowState(
            task_path=str(tmp_path / "task.yaml"),
            done=True,
            stage=_workflow_module.STAGE_CODER,
            num_iterations=1,
        ),
    )

    wf = _workflow_module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        concurrent=True,
        replan_on_qa=True,
        feedback="fix g1",
    )
    seen: list = []
    wf._run_plan_drafter = lambda *a, **kw: seen.append(kw.get("feedback_triggered"))
    wf._latest_plan_drafter_decision = lambda: "POLISHING"
    try:
        wf.run(tmp_path / "task.yaml")
    finally:
        wf.close()

    assert True in seen  # the replan folded the pending feedback in
    assert _graph_states(tmp_path)["g1"] == "done"  # failed subtree re-run to DONE


def test_concurrent_feedback_replan_runs_before_scheduler_pass(tmp_path, monkeypatch):
    """A ``--feedback`` concurrent resume replans BEFORE the first scheduler pass.

    Mirrors a real interrupted run: a node was in-flight (``RUNNING``) when the
    user stopped the run, then resumes with new ``--feedback`` and no failure.
    ``GraphState.load`` resets the ``RUNNING`` node to ``PENDING``, so round 0
    of the scheduler would drive it straight to DONE and ``succeeded`` would
    break the loop — under the old post-scheduler replan the feedback would be
    silently dropped (the replan was only reachable via a FAILED node). The
    feedback replan must instead run *before* the scheduler pass so a newly
    added independent stage can dispatch in the SAME pass, in parallel with the
    resumed node. This test proves the replan fires (``feedback_triggered=True``)
    even though nothing failed.
    """
    from agent_flow.orchestration import GraphState, NodeState

    _install_fake(monkeypatch, _FakeInvoker(reviewer_decisions=["APPROVE"]))
    (tmp_path / "plan.md").write_text(_ONE_GOAL_PLAN)
    (tmp_path / "acceptance-criteria.md").write_text("## g1\n- [ ] g1 built\n")
    _write_task_yaml(tmp_path)
    # In-flight when stopped: RUNNING is reset to PENDING on resume (only
    # DONE/FAILED survive), so g1 re-runs and would succeed with no failure.
    GraphState(states={"g1": NodeState.RUNNING}, worktrees={}).save(tmp_path / ".graph_state.json")
    _workflow_module.save_state(
        tmp_path / _workflow_module.STATE_FILENAME,
        _workflow_module.WorkflowState(
            task_path=str(tmp_path / "task.yaml"),
            done=False,
            stage=_workflow_module.STAGE_CODER,
            num_iterations=1,
        ),
    )

    wf = _workflow_module.AgentTeamWorkflow(
        workspace=tmp_path,
        num_iterations=1,
        concurrent=True,
        replan_on_qa=True,
        feedback="add a prep stage for downstream work",
    )
    seen: list = []
    wf._run_plan_drafter = lambda *a, **kw: seen.append(kw.get("feedback_triggered"))
    wf._latest_plan_drafter_decision = lambda: "POLISHING"
    try:
        wf.run(tmp_path / "task.yaml")
    finally:
        wf.close()

    # Exactly one replan, feedback-triggered, and it happened even though g1
    # never failed — only a pre-scheduler replan can be reached in that case.
    assert seen == [True]
    assert _graph_states(tmp_path)["g1"] == "done"  # the resumed node still ran


def test_linear_path_is_unchanged_when_not_concurrent(tmp_path):
    """Regression: with ``concurrent=False`` the linear build loop runs unchanged.

    Uses the existing linear-path stub harness. The concurrent machinery must
    stay dormant: no per-node sub-workspaces and no ``.graph_state.json``.
    """
    workflow = _workflow_module.AgentTeamWorkflow(
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

    assert trace == [(1, "coder"), (1, "reviewer"), (1, "qa")]
    state = _workflow_module.load_state(tmp_path / _workflow_module.STATE_FILENAME)
    assert state.done is True
    # The concurrent DAG machinery stayed dormant.
    assert not (tmp_path / ".graph_state.json").exists()
    assert not (tmp_path / "nodes").exists()
