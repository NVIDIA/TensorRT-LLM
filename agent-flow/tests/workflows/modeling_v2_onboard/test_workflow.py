# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests of the planner/worker loop on a real git repo.

Agents are scripted (``_fakes``); the scheduler, the git worktree provider,
the checkpoints and the replan reset are real.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.orchestration import GraphState, NodeState
from agent_flow.workflows.modeling_v2_onboard.files import Layout
from agent_flow.workflows.modeling_v2_onboard.plan import PlanError
from agent_flow.workflows.modeling_v2_onboard.workflow import (
    ModelingV2OnboardWorkflow,
    RunOutcome,
    WorkflowError,
)

from ._fakes import FakeFactory, result_path_in, write_result

# a and b are independent; c builds on a's code; land joins b and c.
_GRAPH = """\
  - id: a
    type: new_op
  - id: b
    type: new_op
  - id: c
    type: target
    depends_on: [a]
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [b, c]
"""


def _plan_text(nodes: str = _GRAPH, note: str = "") -> str:
    ids = re.findall(r"- id: (\S+)", nodes)
    sections = "\n".join(f"### {i}\nDo {i}. {note}\n" for i in ids)
    return f"# Plan\n\n{sections}\n## Execution Graph\n\n```yaml\nnodes:\n{nodes}```\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _workflow(
    tmp_path: Path, repo: Path, factory: FakeFactory, **task
) -> ModelingV2OnboardWorkflow:
    task_path = tmp_path / "task.yaml"
    task_data = {"trtllm_repo_path": str(repo), **task}
    task_path.write_text(yaml.safe_dump(task_data))
    return ModelingV2OnboardWorkflow(
        workspace=tmp_path / "ws",
        task_path=task_path,
        task_data=task_data,
        agent_factory=factory,
        max_parallel=4,
    )


def _plan_tag(prompt: str) -> Path:
    return Path(re.search(r"Write the plan to `([^`]+)`", prompt).group(1))


def _task_id(agent) -> str:
    return agent.name[len("worker[") : -1]


def _worker_writing_a_file(fail_first: set[str] = frozenset()):
    """Each task writes ``<id>.txt`` in its worktree; ids in ``fail_first`` fail once."""
    attempts: dict[str, int] = {}

    async def worker(agent, prompt):
        task_id = _task_id(agent)
        attempts[task_id] = attempts.get(task_id, 0) + 1
        if task_id in fail_first and attempts[task_id] == 1:
            write_result(result_path_in(prompt), "FAILED", f"{task_id} broke")
            return
        if task_id != "land":
            (agent.cwd / f"{task_id}.txt").write_text(f"{task_id}\n")
        write_result(result_path_in(prompt), "DONE", f"{task_id} works")

    worker.attempts = attempts
    return worker


def test_all_tasks_done_lands_every_branch_in_the_checkout(tmp_path, tmp_git_repo):
    async def planner(agent, prompt):
        _plan_tag(prompt).write_text(_plan_text())

    worker = _worker_writing_a_file()
    factory = FakeFactory(planner=planner, worker=worker)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)

    assert workflow.run() == RunOutcome.SUCCEEDED

    for task_id in ("a", "b", "c"):
        assert (tmp_git_repo / f"{task_id}.txt").read_text() == f"{task_id}\n"
    assert _git(tmp_git_repo, "status", "--porcelain") == ""
    assert _git(tmp_git_repo, "branch", "--list", "node/*") == ""
    # c started from a's code.
    (c_agent,) = factory.workers("c")
    assert "branch `node/a`" in c_agent.prompts[0]
    # land ran in the checkout itself.
    (land_agent,) = factory.workers("land")
    assert land_agent.cwd == tmp_git_repo.absolute()

    layout = Layout(tmp_path / "ws")
    assert json.loads(layout.state_path.read_text())["outcome"] == "succeeded"
    assert (layout.workspace / "task.yaml").is_file()
    # A finished run does not run again, whatever the checkout holds now.
    (tmp_git_repo / "later.txt").write_text("x")
    assert workflow.run() == RunOutcome.SUCCEEDED
    assert len(factory.agents) == 5


def test_a_failed_task_is_replanned_and_rerun_on_its_dependency_code(tmp_path, tmp_git_repo):
    decisions = []

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _plan_tag(prompt).write_text(_plan_text())
            return
        decision = Path(re.search(r"Then write\s+`([^`]+)`", prompt).group(1))
        assert "`c`" in prompt.split("reported FAILED", 1)[1].split("- DONE", 1)[0]
        plan = Path(re.search(r"- The plan: `([^`]+)`", prompt).group(1))
        plan.write_text(_plan_text(note="Second try: narrower scope."))
        decision.write_text("decision: REVISED\nsummary: narrowed c\n")
        decisions.append(decision)

    worker = _worker_writing_a_file(fail_first={"c"})
    factory = FakeFactory(planner=planner, worker=worker)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)

    assert workflow.run() == RunOutcome.SUCCEEDED

    assert worker.attempts == {"a": 1, "b": 1, "c": 2, "land": 1}
    assert [d.name for d in decisions] == ["round_1.yaml"]
    retry = factory.workers("c")[1]
    layout = Layout(tmp_path / "ws")
    # The retry sees the failed attempt, and still starts from a's code even
    # though a's only dependent had failed.
    assert str(layout.attempt_path("c", 1)) in retry.prompts[0]
    assert "FAILED" in layout.attempt_path("c", 1).read_text()
    assert (tmp_git_repo / "a.txt").is_file() and (tmp_git_repo / "c.txt").is_file()


def test_a_revision_resets_changed_tasks_and_keeps_done_ones(tmp_path, tmp_git_repo):
    revised = """\
  - id: a
    type: new_op
  - id: b
    type: new_op
  - id: fix
    type: new_op
  - id: c
    type: target
    depends_on: [a]
  - id: d
    type: merge
    kind: merge
    depends_on: [c, fix]
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [b, d]
"""

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _plan_tag(prompt).write_text(_plan_text())
            return
        plan = Path(re.search(r"- The plan: `([^`]+)`", prompt).group(1))
        plan.write_text(_plan_text(revised))
        decision = Path(re.search(r"Then write\s+`([^`]+)`", prompt).group(1))
        decision.write_text("decision: REVISED\nsummary: add fix\n")

    worker = _worker_writing_a_file(fail_first={"c"})
    workflow = _workflow(tmp_path, tmp_git_repo, FakeFactory(planner=planner, worker=worker))

    assert workflow.run() == RunOutcome.SUCCEEDED
    assert worker.attempts == {"a": 1, "b": 1, "c": 2, "fix": 1, "d": 1, "land": 1}
    for task_id in ("a", "b", "c", "fix", "d"):
        assert (tmp_git_repo / f"{task_id}.txt").is_file()


@pytest.mark.parametrize(
    ("decision", "outcome"),
    [("ABORT", RunOutcome.ABORTED), ("DONE", RunOutcome.PLANNER_DONE)],
)
def test_the_planner_can_stop_the_run(tmp_path, tmp_git_repo, decision, outcome):
    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _plan_tag(prompt).write_text(_plan_text())
            return
        path = Path(re.search(r"Then write\s+`([^`]+)`", prompt).group(1))
        path.write_text(f"decision: {decision}\nsummary: stop\n")

    worker = _worker_writing_a_file(fail_first={"c"})
    workflow = _workflow(tmp_path, tmp_git_repo, FakeFactory(planner=planner, worker=worker))

    assert workflow.run() == outcome
    assert worker.attempts.get("land") is None
    # Nothing was landed, so the task branches are kept.
    assert "node/a" in _git(tmp_git_repo, "branch", "--list", "node/*")
    states = GraphState.load(Layout(tmp_path / "ws").graph_state_path).states
    assert states["c"] == NodeState.FAILED


def test_an_invalid_plan_is_sent_back_until_it_validates(tmp_path, tmp_git_repo):
    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            plan = _plan_tag(prompt)
            plan.write_text("# Plan\n\nno graph yet\n")
            agent.plan_path = plan
            return
        assert "cannot be run as written" in prompt and "Execution Graph" in prompt
        agent.plan_path.write_text(_plan_text())

    factory = FakeFactory(planner=planner, worker=_worker_writing_a_file())
    workflow = _workflow(tmp_path, tmp_git_repo, factory)

    assert workflow.run() == RunOutcome.SUCCEEDED
    (planner_agent,) = [a for a in factory.agents if a.name == "planner"]
    assert len(planner_agent.prompts) == 2 and planner_agent.closed


def test_a_plan_that_never_validates_stops_the_run(tmp_path, tmp_git_repo):
    async def planner(agent, prompt):
        (tmp_path / "ws" / "plan.md").write_text("# Plan\n")

    workflow = _workflow(tmp_path, tmp_git_repo, FakeFactory(planner=planner))
    with pytest.raises(PlanError):
        workflow.run()
    # The run resumes in the plan phase.
    assert json.loads((tmp_path / "ws" / ".state.json").read_text())["phase"] == "plan"


def test_a_dirty_checkout_is_refused(tmp_path, tmp_git_repo):
    (tmp_git_repo / "stray.txt").write_text("x")
    with pytest.raises(WorkflowError, match="untracked"):
        _workflow(tmp_path, tmp_git_repo, FakeFactory()).run()


def test_a_workspace_inside_the_checkout_is_refused(tmp_path, tmp_git_repo):
    task_path = tmp_path / "task.yaml"
    task_path.write_text("x: 1\n")
    workflow = ModelingV2OnboardWorkflow(
        workspace=tmp_git_repo / "ws",
        task_path=task_path,
        task_data={"trtllm_repo_path": str(tmp_git_repo)},
        agent_factory=FakeFactory(),
    )
    with pytest.raises(WorkflowError, match="inside trtllm_repo_path"):
        workflow.run()


def test_clean_removes_state_and_task_branches(tmp_path, tmp_git_repo):
    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _plan_tag(prompt).write_text(_plan_text())
            return
        path = Path(re.search(r"Then write\s+`([^`]+)`", prompt).group(1))
        path.write_text("decision: ABORT\nsummary: stop\n")

    worker = _worker_writing_a_file(fail_first={"c"})
    workflow = _workflow(
        tmp_path,
        tmp_git_repo,
        FakeFactory(planner=planner, worker=worker),
        **{"slurm-environment": {"slurm_partition": "p", "docker_image": "i"}},
    )
    assert workflow.run() == RunOutcome.ABORTED
    assert Layout(tmp_path / "ws").test_commands_path.is_file()

    workflow.clean()
    layout = Layout(tmp_path / "ws")
    assert not layout.state_path.exists() and not layout.results_dir.exists()
    assert _git(tmp_git_repo, "branch", "--list", "node/*") == ""
