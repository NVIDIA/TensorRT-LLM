# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests of the planner/worker loop on a real git repo.

Agents are scripted (``_fakes``); the event loop, the git worktree provider,
the checkpoints and the resets are real.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.modeling_v2_onboard.files import Layout
from agent_flow.workflows.modeling_v2_onboard.plan import PlanError
from agent_flow.workflows.modeling_v2_onboard.workflow import (
    MAX_STALLED_TURNS,
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


def _plan_text(nodes: str = _GRAPH, notes: dict[str, str] | None = None) -> str:
    ids = re.findall(r"- id: (\S+)", nodes)
    notes = notes or {}
    sections = "\n".join(f"### {i}\nDo {i}. {notes.get(i, '')}\n" for i in ids)
    return f"# Plan\n\n{sections}\n## Execution Graph\n\n```yaml\nnodes:\n{nodes}```\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _workflow(
    tmp_path: Path, repo: Path, factory: FakeFactory, max_parallel: int = 4, **task
) -> ModelingV2OnboardWorkflow:
    task_path = tmp_path / "task.yaml"
    task_data = {"trtllm_repo_path": str(repo), **task}
    task_path.write_text(yaml.safe_dump(task_data))
    return ModelingV2OnboardWorkflow(
        workspace=tmp_path / "ws",
        task_path=task_path,
        task_data=task_data,
        agent_factory=factory,
        max_parallel=max_parallel,
    )


def _plan_path(prompt: str) -> Path:
    return Path(re.search(r"Write the plan to `([^`]+)`", prompt).group(1))


def _turn_file(prompt: str) -> Path:
    return Path(re.search(r"End the turn by writing `([^`]+)`", prompt).group(1))


def _edit_plan(prompt: str) -> Path:
    return Path(re.search(r"The plan \(edit in place\): `([^`]+)`", prompt).group(1))


def _states(prompt: str) -> dict[str, str]:
    return dict(re.findall(r"^- `([^`]+)`: (\w+)", prompt, flags=re.MULTILINE))


def _decide(prompt: str, decision: str = "CONTINUE", **lists: list[str]) -> None:
    _turn_file(prompt).write_text(yaml.safe_dump({"decision": decision, "summary": "s", **lists}))


def _is_turn(prompt: str) -> bool:
    return "End the turn by writing" in prompt


def _finish_when_ready(prompt: str) -> None:
    """FINISH once every task but the final one is done."""
    states = _states(prompt)
    pending = [t for t, s in states.items() if t != "land" and s != "done"]
    _decide(prompt, "CONTINUE" if pending else "FINISH")


def _task_id(agent) -> str:
    return agent.name[len("worker[") : -1]


class _Workers:
    """Each task writes ``<id>.txt`` in its worktree and reports DONE.

    ``behaviour[id]`` overrides a task: an async callable ``(agent, prompt,
    attempt) -> status | None`` (``None`` means the default: write the file,
    DONE).
    """

    def __init__(self, **behaviour) -> None:
        self.behaviour = behaviour
        self.attempts: dict[str, int] = {}
        self.order: list[str] = []
        self.running = 0
        self.max_running = 0

    async def __call__(self, agent, prompt) -> None:
        task_id = _task_id(agent)
        self.attempts[task_id] = self.attempts.get(task_id, 0) + 1
        self.order.append(task_id)
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            status = None
            special = self.behaviour.get(task_id)
            if special is not None:
                status = await special(agent, prompt, self.attempts[task_id])
            await asyncio.sleep(0.01)
            if status is None:
                if task_id != "land":
                    (agent.cwd / f"{task_id}.txt").write_text(f"{task_id}\n")
                status = "DONE"
            write_result(result_path_in(prompt), status, f"{task_id} {status.lower()}")
        finally:
            self.running -= 1


def _planner(on_turn):
    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _plan_path(prompt).write_text(_plan_text())
        elif _is_turn(prompt):
            await on_turn(agent, prompt)

    return planner


def test_tasks_run_in_parallel_and_the_final_task_waits_for_finish(tmp_path, tmp_git_repo):
    turns: list[str] = []

    async def on_turn(agent, prompt):
        turns.append(prompt)
        _finish_when_ready(prompt)

    started = {"a": asyncio.Event(), "b": asyncio.Event()}

    def meet(me: str, other: str):
        # Each finishes only once the other has started: a and b run side by side.
        async def behaviour(agent, prompt, attempt):
            started[me].set()
            await started[other].wait()

        return behaviour

    workers = _Workers(a=meet("a", "b"), b=meet("b", "a"))
    factory = FakeFactory(planner=_planner(on_turn), worker=workers)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)

    assert workflow.run() == RunOutcome.SUCCEEDED

    assert workers.order[-1] == "land"
    assert "held until you decide FINISH" in turns[0]
    for task_id in ("a", "b", "c"):
        assert (tmp_git_repo / f"{task_id}.txt").read_text() == f"{task_id}\n"
    assert _git(tmp_git_repo, "status", "--porcelain") == ""
    assert _git(tmp_git_repo, "branch", "--list", "node/*") == ""
    (c_agent,) = factory.workers("c")
    assert "branch `node/a`" in c_agent.prompts[0]
    (land_agent,) = factory.workers("land")
    assert land_agent.cwd == tmp_git_repo.absolute()
    # One planner session for the whole run.
    (planner_agent,) = [a for a in factory.agents if a.name == "planner"]
    assert planner_agent.closed

    layout = Layout(tmp_path / "ws")
    assert json.loads(layout.state_path.read_text())["outcome"] == "succeeded"
    # A finished run does not run again, whatever the checkout holds now.
    (tmp_git_repo / "later.txt").write_text("x")
    assert workflow.run() == RunOutcome.SUCCEEDED
    assert len(factory.agents) == 5


def test_the_planner_adds_work_while_other_workers_run(tmp_path, tmp_git_repo):
    graph = """\
  - id: quick
    type: new_op
  - id: slow
    type: new_op
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [quick, slow]
"""
    grown = """\
  - id: quick
    type: new_op
  - id: slow
    type: new_op
  - id: perf
    type: perf
    depends_on: [quick]
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [quick, slow, perf]
"""
    perf_started = asyncio.Event()

    async def slow(agent, prompt, attempt):
        # Finishes only once the task the planner added has started.
        await perf_started.wait()

    async def perf(agent, prompt, attempt):
        perf_started.set()

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _plan_path(prompt).write_text(_plan_text(graph))
            return
        states = _states(prompt)
        if "perf" not in states:
            assert states["slow"] == "running"
            _edit_plan(prompt).write_text(_plan_text(grown))
            _decide(prompt)
        else:
            _finish_when_ready(prompt)

    workers = _Workers(slow=slow, perf=perf)
    workflow = _workflow(tmp_path, tmp_git_repo, FakeFactory(planner=planner, worker=workers))

    assert workflow.run() == RunOutcome.SUCCEEDED
    assert workers.attempts == {"quick": 1, "slow": 1, "perf": 1, "land": 1}
    assert (tmp_git_repo / "perf.txt").is_file()
    (perf_agent,) = [a for a in workflow.agent_factory.agents if a.name == "worker[perf]"]
    assert "## Tag `perf`" in perf_agent.system_prompt


@pytest.mark.parametrize("how", ["retry", "rewrite"])
def test_a_failed_task_holds_only_its_dependents_and_reruns(tmp_path, tmp_git_repo, how):
    failure_seen = asyncio.Event()

    async def b(agent, prompt, attempt):
        # b keeps running while the planner handles c's failure: nothing pauses.
        await failure_seen.wait()

    async def c(agent, prompt, attempt):
        return "FAILED" if attempt == 1 else None

    async def on_turn(agent, prompt):
        states = _states(prompt)
        if states.get("c") == "failed":
            assert states["b"] == "running"
            assert "`c` FAILED: c failed" in prompt
            if how == "retry":
                _decide(prompt, retry=["c"])
            else:
                _edit_plan(prompt).write_text(_plan_text(notes={"c": "Narrower."}))
                _decide(prompt)
            failure_seen.set()
            return
        _finish_when_ready(prompt)

    workers = _Workers(b=b, c=c)
    factory = FakeFactory(planner=_planner(on_turn), worker=workers)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)

    assert workflow.run() == RunOutcome.SUCCEEDED
    assert workers.attempts == {"a": 1, "b": 1, "c": 2, "land": 1}
    layout = Layout(tmp_path / "ws")
    retry = factory.workers("c")[1]
    # The rerun sees the failed attempt and starts from a's code again.
    assert str(layout.attempt_path("c", 1)) in retry.prompts[0]
    assert "FAILED" in layout.attempt_path("c", 1).read_text()
    assert (tmp_git_repo / "a.txt").is_file() and (tmp_git_repo / "c.txt").is_file()


def test_a_running_task_can_be_cancelled_and_removed(tmp_path, tmp_git_repo):
    graph = """\
  - id: a
    type: new_op
  - id: stuck
    type: new_op
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a, stuck]
"""
    without_stuck = """\
  - id: a
    type: new_op
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a]
"""

    async def stuck(agent, prompt, attempt):
        (agent.cwd / "partial.txt").write_text("x")
        await asyncio.Event().wait()

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _plan_path(prompt).write_text(_plan_text(graph))
            return
        if "stuck" in _states(prompt):
            _edit_plan(prompt).write_text(_plan_text(without_stuck))
            _decide(prompt, "FINISH", cancel=["stuck"])

    workers = _Workers(stuck=stuck)
    factory = FakeFactory(planner=planner, worker=workers)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)

    assert workflow.run() == RunOutcome.SUCCEEDED
    (stuck_agent,) = factory.workers("stuck")
    assert stuck_agent.closed
    assert not (tmp_git_repo / "partial.txt").exists()
    assert (tmp_git_repo / "a.txt").is_file()
    assert _git(tmp_git_repo, "branch", "--list", "node/*") == ""


def test_a_running_task_cannot_be_rewritten_without_cancelling_it(tmp_path, tmp_git_repo):
    release = asyncio.Event()

    async def b(agent, prompt, attempt):
        await release.wait()

    turns: list[str] = []

    async def on_turn(agent, prompt):
        turns.append(prompt)
        if len(turns) == 1:
            assert _states(prompt)["b"] == "running"
            # Rewriting running b's section is refused every time.
            _edit_plan(prompt).write_text(_plan_text(notes={"b": "Changed."}))
            _decide(prompt)
            return
        release.set()
        _finish_when_ready(prompt)

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _plan_path(prompt).write_text(_plan_text())
        elif _is_turn(prompt):
            await on_turn(agent, prompt)
        else:
            assert "task 'b' is RUNNING" in prompt
            _edit_plan(turns[0]).write_text(_plan_text(notes={"b": "Changed."}))

    workers = _Workers(b=b)
    factory = FakeFactory(planner=planner, worker=workers)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)

    assert workflow.run() == RunOutcome.SUCCEEDED
    # The discarded turn is reported on the next one, and b was never restarted.
    assert "were discarded" in turns[1]
    assert workers.attempts["b"] == 1
    assert "Changed." not in Layout(tmp_path / "ws").running_plan_path.read_text()


def test_abort_stops_running_workers_and_keeps_branches(tmp_path, tmp_git_repo):
    async def b(agent, prompt, attempt):
        await asyncio.Event().wait()

    async def on_turn(agent, prompt):
        _decide(prompt, "ABORT")

    workers = _Workers(b=b)
    factory = FakeFactory(planner=_planner(on_turn), worker=workers)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)

    assert workflow.run() == RunOutcome.ABORTED
    assert factory.workers("b")[0].closed
    assert "land" not in workers.attempts
    assert "node/a" in _git(tmp_git_repo, "branch", "--list", "node/*")
    state = json.loads(Layout(tmp_path / "ws").state_path.read_text())
    assert (state["phase"], state["outcome"]) == ("done", "aborted")


def test_a_stalled_run_stops_after_the_stalled_turns(tmp_path, tmp_git_repo):
    turns: list[str] = []

    async def on_turn(agent, prompt):
        turns.append(prompt)
        _decide(prompt)  # never FINISH

    workers = _Workers()
    workflow = _workflow(
        tmp_path, tmp_git_repo, FakeFactory(planner=_planner(on_turn), worker=workers)
    )

    with pytest.raises(WorkflowError, match="nothing has run"):
        workflow.run()
    assert "land" not in workers.attempts
    assert sum("The run is stalled" in t for t in turns) == MAX_STALLED_TURNS


def test_the_run_resumes_after_an_interruption(tmp_path, tmp_git_repo):
    async def slow_forever(agent, prompt, attempt):
        await asyncio.Event().wait()

    async def crashing_turn(agent, prompt):
        raise RuntimeError("planner backend died")

    first = _Workers(b=slow_forever)
    workflow = _workflow(
        tmp_path, tmp_git_repo, FakeFactory(planner=_planner(crashing_turn), worker=first)
    )
    with pytest.raises(RuntimeError, match="planner backend died"):
        workflow.run()
    state = json.loads(Layout(tmp_path / "ws").state_path.read_text())
    assert state["tasks"]["b"] == "running"

    turns: list[str] = []

    async def on_turn(agent, prompt):
        turns.append(prompt)
        _finish_when_ready(prompt)

    second = _Workers()
    factory = FakeFactory(planner=_planner(on_turn), worker=second)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)
    assert workflow.run() == RunOutcome.SUCCEEDED
    assert "The orchestrator restarted" in turns[0]
    assert "a" not in second.attempts  # a was already DONE
    assert second.attempts["b"] == 1
    assert (tmp_git_repo / "b.txt").is_file()


def test_no_more_than_max_parallel_workers_run(tmp_path, tmp_git_repo):
    graph = "".join(f"  - id: t{i}\n    type: new_op\n" for i in range(5)) + (
        "  - id: land\n    type: merge\n    kind: merge\n    isolation: shared\n"
        "    depends_on: [t0, t1, t2, t3, t4]\n"
    )

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _plan_path(prompt).write_text(_plan_text(graph))
        else:
            _finish_when_ready(prompt)

    workers = _Workers()
    workflow = _workflow(
        tmp_path, tmp_git_repo, FakeFactory(planner=planner, worker=workers), max_parallel=2
    )
    assert workflow.run() == RunOutcome.SUCCEEDED
    assert workers.max_running == 2


def test_an_invalid_plan_is_sent_back_until_it_validates(tmp_path, tmp_git_repo):
    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            plan = _plan_path(prompt)
            plan.write_text("# Plan\n\nno graph yet\n")
            agent.plan_path = plan
        elif "cannot be run as written" in prompt:
            assert "Execution Graph" in prompt
            agent.plan_path.write_text(_plan_text())
        else:
            _finish_when_ready(prompt)

    factory = FakeFactory(planner=planner, worker=_Workers())
    assert _workflow(tmp_path, tmp_git_repo, factory).run() == RunOutcome.SUCCEEDED
    (planner_agent,) = [a for a in factory.agents if a.name == "planner"]
    assert planner_agent.prompts[1].startswith("`")


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
    async def on_turn(agent, prompt):
        _decide(prompt, "ABORT")

    workflow = _workflow(
        tmp_path,
        tmp_git_repo,
        FakeFactory(planner=_planner(on_turn), worker=_Workers()),
        **{"slurm-environment": {"slurm_partition": "p", "docker_image": "i"}},
    )
    assert workflow.run() == RunOutcome.ABORTED
    assert Layout(tmp_path / "ws").test_commands_path.is_file()

    workflow.clean()
    layout = Layout(tmp_path / "ws")
    assert not layout.state_path.exists() and not layout.results_dir.exists()
    assert _git(tmp_git_repo, "branch", "--list", "node/*") == ""
