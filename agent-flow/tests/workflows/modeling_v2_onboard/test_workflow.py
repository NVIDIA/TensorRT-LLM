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

from agent_flow.workflows.modeling_v2_onboard import gpu_pool
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
    tmp_path: Path,
    repo: Path,
    factory: FakeFactory,
    max_parallel: int = 4,
    side_wait_minutes: float = 60,
    primer: bool = False,
    **task,
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
        side_wait_minutes=side_wait_minutes,
        primer=primer,
    )


def _first_batch(prompt: str, planning: bool = False) -> Path:
    """Answer the first planning turn; return the plan to write."""
    turn = Path(re.search(r"end the first batch by writing `([^`]+)`", prompt).group(1))
    turn.write_text(yaml.safe_dump({"decision": "CONTINUE", "summary": "s", "planning": planning}))
    return Path(re.search(r"Write the plan to `([^`]+)`", prompt).group(1))


def _turn_file(prompt: str) -> Path:
    return Path(re.search(r"End the turn by writing `([^`]+)`", prompt).group(1))


def _edit_plan(prompt: str) -> Path:
    return Path(re.search(r"The plan \(edit in place\): `([^`]+)`", prompt).group(1))


def _states(prompt: str) -> dict[str, str]:
    return dict(re.findall(r"^- `([^`]+)`: (\w+)", prompt, flags=re.MULTILINE))


def _decide(prompt: str, decision: str = "CONTINUE", **fields) -> None:
    _turn_file(prompt).write_text(yaml.safe_dump({"decision": decision, "summary": "s", **fields}))


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
            _first_batch(prompt).write_text(_plan_text())
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
    # No e2e_accuracy and no perf task in this graph: every task defaults to feature.
    assert set(json.loads(layout.priorities_path.read_text()).values()) == {"feature"}
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
            _first_batch(prompt).write_text(_plan_text(graph))
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
        if attempt == 1:
            (agent.cwd / "first-try.txt").write_text("x\n")
            return "FAILED"
        # A rerun carries on from its own branch.
        assert (agent.cwd / "first-try.txt").is_file()
        return None

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


_WITH_STUCK = """\
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
_WITHOUT_STUCK = """\
  - id: a
    type: new_op
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a]
"""


def test_a_cancelled_task_is_abandoned_and_left_as_it_was(tmp_path, tmp_git_repo):
    async def stuck(agent, prompt, attempt):
        (agent.cwd / "partial.txt").write_text("x")
        await asyncio.Event().wait()

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(_plan_text(_WITH_STUCK))
            return
        if "stuck" in _states(prompt):
            _edit_plan(prompt).write_text(_plan_text(_WITHOUT_STUCK))
            _decide(prompt, "FINISH", cancel=["stuck"])

    workers = _Workers(stuck=stuck)
    factory = FakeFactory(planner=planner, worker=workers)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)

    assert workflow.run() == RunOutcome.SUCCEEDED
    (stuck_agent,) = factory.workers("stuck")
    assert stuck_agent.closed
    assert (tmp_path / "ws" / "worktrees" / "stuck" / "partial.txt").is_file()
    assert "node/stuck" in _git(tmp_git_repo, "branch", "--list", "node/*")
    assert "## `stuck`" in workflow.layout.abandoned_path.read_text()
    assert (tmp_git_repo / "a.txt").is_file()
    assert not (tmp_git_repo / "partial.txt").exists()


def test_a_cancelled_task_leaves_the_plan_and_its_id_is_not_reused(tmp_path, tmp_git_repo):
    with_stuck_2 = _WITHOUT_STUCK.replace(
        "  - id: land", "  - id: stuck-2\n    type: new_op\n  - id: land"
    ).replace("depends_on: [a]", "depends_on: [a, stuck-2]")
    fixes: list[str] = []
    last: dict[str, Path] = {}

    async def stuck(agent, prompt, attempt):
        await asyncio.Event().wait()

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(_plan_text(_WITH_STUCK))
            return
        if "cannot be run as written" in prompt:
            fixes.append(prompt)
            if "is still in the plan" in prompt:
                last["plan"].write_text(_plan_text(_WITHOUT_STUCK))
            else:
                last["plan"].write_text(_plan_text(with_stuck_2))
            return
        last["plan"] = _edit_plan(prompt)
        states = _states(prompt)
        if states.get("stuck") == "running":
            _decide(prompt, cancel=["stuck"])
        elif "stuck" not in states and "stuck-2" not in states:
            _edit_plan(prompt).write_text(_plan_text(_WITH_STUCK))
            _decide(prompt)
        else:
            _finish_when_ready(prompt)

    workers = _Workers(stuck=stuck)
    workflow = _workflow(tmp_path, tmp_git_repo, FakeFactory(planner=planner, worker=workers))

    assert workflow.run() == RunOutcome.SUCCEEDED
    assert "is still in the plan" in fixes[0]
    assert "cannot be used again" in fixes[1]
    assert workers.attempts == {"a": 1, "stuck": 1, "stuck-2": 1, "land": 1}


def test_a_jobs_json_the_worker_wrote_itself_does_not_stop_a_cancel(tmp_path, tmp_git_repo):
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
    layout = []

    async def stuck(agent, prompt, attempt):
        jobs = layout[0].jobs_path("stuck")
        jobs.parent.mkdir(parents=True, exist_ok=True)
        jobs.write_text('{"run1": {"what": "tests", "log": "run1.log"}}')
        await asyncio.Event().wait()

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(_plan_text(graph))
            return
        if "stuck" in _states(prompt):
            _edit_plan(prompt).write_text(_plan_text(without_stuck))
            _decide(prompt, "FINISH", cancel=["stuck"])

    factory = FakeFactory(planner=planner, worker=_Workers(stuck=stuck))
    workflow = _workflow(tmp_path, tmp_git_repo, factory)
    layout.append(workflow.layout)

    assert workflow.run() == RunOutcome.SUCCEEDED
    (stuck_agent,) = factory.workers("stuck")
    assert stuck_agent.closed


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
            _first_batch(prompt).write_text(_plan_text())
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
    # a finished before the crash; the planner never got to act on it, so it still hears it.
    assert "- `a` DONE" in turns[0]
    assert "a" not in second.attempts  # a was already DONE
    assert second.attempts["b"] == 1
    assert (tmp_git_repo / "b.txt").is_file()


def test_a_stop_keeps_what_a_running_worker_had_not_committed(tmp_path, tmp_git_repo):
    async def write_then_hang(agent, prompt, attempt):
        (agent.cwd / "draft.py").write_text("x = 1\n")
        await asyncio.Event().wait()

    async def crashing_turn(agent, prompt):
        raise RuntimeError("planner backend died")

    workflow = _workflow(
        tmp_path,
        tmp_git_repo,
        FakeFactory(planner=_planner(crashing_turn), worker=_Workers(b=write_then_hang)),
    )
    with pytest.raises(RuntimeError, match="planner backend died"):
        workflow.run()

    assert _git(tmp_git_repo, "show", "node/b:draft.py") == "x = 1\n"
    assert "Signed-off-by: " in _git(tmp_git_repo, "log", "-1", "--format=%B", "node/b")


def test_no_more_than_max_parallel_workers_run(tmp_path, tmp_git_repo):
    graph = "".join(f"  - id: t{i}\n    type: new_op\n" for i in range(5)) + (
        "  - id: land\n    type: merge\n    kind: merge\n    isolation: shared\n"
        "    depends_on: [t0, t1, t2, t3, t4]\n"
    )

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(_plan_text(graph))
        else:
            _finish_when_ready(prompt)

    async def overlap(agent, prompt, attempt):
        # Hold on until another worker runs too (the fifth eventually runs alone).
        for _ in range(500):
            if workers.running >= 2:
                break
            await asyncio.sleep(0.01)

    workers = _Workers(**{f"t{i}": overlap for i in range(5)})
    workflow = _workflow(
        tmp_path, tmp_git_repo, FakeFactory(planner=planner, worker=workers), max_parallel=2
    )
    assert workflow.run() == RunOutcome.SUCCEEDED
    assert workers.max_running == 2


def test_free_slots_go_to_the_most_urgent_ready_task_first(tmp_path, tmp_git_repo):
    # Listed perf first, then feature, then accuracy; one slot.
    graph = (
        "  - id: p\n    type: perf\n"
        "  - id: f\n    type: new_op\n"
        "  - id: a\n    type: new_op\n    priority: accuracy\n"
        "  - id: land\n    type: merge\n    kind: merge\n    isolation: shared\n"
        "    depends_on: [p, f, a]\n"
    )

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(_plan_text(graph))
        else:
            _finish_when_ready(prompt)

    workers = _Workers()
    workflow = _workflow(
        tmp_path, tmp_git_repo, FakeFactory(planner=planner, worker=workers), max_parallel=1
    )
    assert workflow.run() == RunOutcome.SUCCEEDED
    assert workers.order == ["a", "f", "p", "land"]


def test_workers_start_while_the_planner_is_still_planning(tmp_path, tmp_git_repo):
    first = """\
  - id: env
    type: target
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [env]
"""
    whole = """\
  - id: env
    type: target
  - id: op
    type: new_op
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [env, op]
"""
    release = asyncio.Event()

    async def env(agent, prompt, attempt):
        await release.wait()

    turns: list[str] = []

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt, planning=True).write_text(_plan_text(first))
            return
        if "cannot be run as written" in prompt:
            agent.fixes = [*getattr(agent, "fixes", []), prompt]
            _edit_plan(turns[-1]).write_text(_plan_text(whole))
            return
        turns.append(prompt)
        if len(turns) == 1:
            # Woken at once, with the first batch already running.
            assert "You are still planning" in prompt
            assert _states(prompt)["env"] == "running"
            release.set()
            await _until(lambda: _task_state(tmp_path, "env") == "done")
            # A planning turn that changes nothing is sent back.
            _decide(prompt, planning=True)
        elif len(turns) == 2:
            assert "Finished since your last turn:\n- `env` DONE" in prompt
            _decide(prompt, planning=False)
        else:
            assert "You are still planning" not in prompt
            _finish_when_ready(prompt)

    factory = FakeFactory(planner=planner, worker=_Workers(env=env))
    assert _workflow(tmp_path, tmp_git_repo, factory).run() == RunOutcome.SUCCEEDED
    (planner_agent,) = [a for a in factory.agents if a.name == "planner"]
    assert "changes nothing" in planner_agent.fixes[0]
    assert (tmp_git_repo / "op.txt").is_file()


def test_planning_cannot_finish(tmp_path, tmp_git_repo):
    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(_plan_text())
            turn = tmp_path / "ws" / "planner_turns" / "turn_0.yaml"
            turn.write_text("decision: FINISH\nplanning: true\n")
        elif "cannot be run as written" in prompt:
            assert "`planning: true` with `FINISH`" in prompt
            turn = tmp_path / "ws" / "planner_turns" / "turn_0.yaml"
            turn.write_text("decision: FINISH\nplanning: false\n")
        else:
            _finish_when_ready(prompt)

    workers = _Workers()
    factory = FakeFactory(planner=planner, worker=workers)
    assert _workflow(tmp_path, tmp_git_repo, factory).run() == RunOutcome.SUCCEEDED
    # FINISH in the first batch lets the final task start as soon as it can.
    assert workers.order[-1] == "land"


def test_the_gpu_pool_is_requested_while_the_planner_plans(tmp_path, tmp_git_repo, monkeypatch):
    calls: list[list[str]] = []

    def slurm(args):
        calls.append(args)
        jobs = sum(c[0] == "sbatch" for c in calls)
        out = "".join(f"{i}\n" for i in range(jobs)) if args[0] == "squeue" else "Submitted\n"
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(gpu_pool, "_run", slurm)

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            # The pool's first holder job is submitted before any plan exists.
            await _until(lambda: any(c[0] == "sbatch" for c in calls))
            _first_batch(prompt).write_text(_plan_text())
        else:
            _finish_when_ready(prompt)

    factory = FakeFactory(planner=planner, worker=_Workers())
    pool = {"nodes": 1, "account": "acct"}
    workflow = _workflow(
        tmp_path,
        tmp_git_repo,
        factory,
        **{
            "slurm-environment": {
                "slurm_partition": "p",
                "docker_image": "/i.sqsh",
                "gpu_pool": pool,
            }
        },
    )
    assert workflow.run() == RunOutcome.SUCCEEDED
    assert calls[-1][0] == "scancel" and workflow.gpu_pool.job_name in calls[-1]
    (a_agent,) = factory.workers("a")
    assert "## Running GPU work: the run's GPU pool" in a_agent.system_prompt
    assert f"gpu_run --pool {tmp_path / 'ws' / 'gpu_pool'} --task a --gpus" in a_agent.prompts[0]


def test_the_planner_grows_the_gpu_pool_up_to_its_limit(tmp_path, tmp_git_repo, monkeypatch):
    calls: list[list[str]] = []

    def slurm(args):
        calls.append(args)
        jobs = sum(c[0] == "sbatch" for c in calls)
        out = "".join(f"{i} R\n" for i in range(jobs)) if args[0] == "squeue" else "Submitted\n"
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(gpu_pool, "_run", slurm)
    turns: list[str] = []

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(_plan_text())
        elif "cannot be run as written" in prompt:
            assert "`gpu_nodes`: 5 is more than the run's limit of 3" in prompt
            _decide(turns[-1], gpu_nodes=2)
        else:
            turns.append(prompt)
            if len(turns) == 1:
                assert "GPU pool: 1 node(s) of 4 GPUs wanted, 1 running" in prompt
                assert "The pool may grow to 3 nodes" in prompt
                _decide(prompt, gpu_nodes=5)
            else:
                assert "GPU pool: 2 node(s) of 4 GPUs wanted, 2 running" in prompt
                _finish_when_ready(prompt)

    factory = FakeFactory(planner=planner, worker=_Workers())
    slurm_env = {
        "slurm_partition": "p",
        "docker_image": "/i.sqsh",
        "gpu_pool": {"nodes": 1, "max_nodes": 3, "account": "acct"},
    }
    workflow = _workflow(tmp_path, tmp_git_repo, factory, **{"slurm-environment": slurm_env})
    assert workflow.run() == RunOutcome.SUCCEEDED
    assert [c[0] for c in calls].count("sbatch") == 2
    state = json.loads(Layout(tmp_path / "ws").state_path.read_text())
    assert state["gpu_nodes"] == 2


def test_a_bad_gpu_pool_is_refused(tmp_path, tmp_git_repo):
    slurm = {"slurm_partition": "p", "docker_image": "/i.sqsh", "gpu_pool": {"nodes": 1}}
    with pytest.raises(WorkflowError, match="gpu_pool.account"):
        _workflow(tmp_path, tmp_git_repo, FakeFactory(), **{"slurm-environment": slurm})


def _summary_path(prompt: str) -> Path:
    return Path(re.search(r"Then write `([^`]+)`", prompt).group(1))


def test_op_workers_start_as_forks_of_the_primer(tmp_path, tmp_git_repo):
    async def primer(agent, prompt):
        _summary_path(prompt).write_text("what entry workers need\n")
        agent.session_id = "primer-1"

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            # The primer reads while the planner plans.
            await _until(lambda: any(a.name == "primer[op]" for a in factory.agents))
            _first_batch(prompt).write_text(_plan_text())
        else:
            _finish_when_ready(prompt)

    factory = FakeFactory(planner=planner, worker=_Workers(), primer=primer)
    workflow = _workflow(tmp_path, tmp_git_repo, factory, primer=True)
    assert workflow.run() == RunOutcome.SUCCEEDED

    (primer_agent,) = [a for a in factory.agents if a.name == "primer[op]"]
    assert primer_agent.cwd == tmp_git_repo.absolute() and primer_agent.closed
    head = _git(tmp_git_repo, "rev-list", "--max-parents=0", "HEAD").strip()
    assert f"(commit `{head}`)" in primer_agent.prompts[0]
    for op in ("a", "b"):
        (worker,) = factory.workers(op)
        assert worker.fork_from == "primer-1"
        assert worker.prompts[0].startswith("**You are now a worker")
    for other in ("c", "land"):
        (worker,) = factory.workers(other)
        assert worker.fork_from is None
    recorded = json.loads(Layout(tmp_path / "ws").primers_path.read_text())["op"]
    assert (recorded["session_id"], recorded["commit"]) == ("primer-1", head)


def test_op_workers_start_afresh_from_the_summary_when_the_primer_fails(tmp_path, tmp_git_repo):
    async def primer(agent, prompt):
        _summary_path(prompt).write_text("partial notes\n")
        raise RuntimeError("primer session died")

    factory = FakeFactory(
        planner=_planner(_finish_when_ready_async), worker=_Workers(), primer=primer
    )
    assert _workflow(tmp_path, tmp_git_repo, factory, primer=True).run() == RunOutcome.SUCCEEDED
    (worker,) = factory.workers("a")
    assert worker.fork_from is None
    knowledge = Layout(tmp_path / "ws").knowledge_path("op")
    assert f"read it first: `{knowledge}`" in worker.prompts[0]
    assert not Layout(tmp_path / "ws").primers_path.exists()


def test_a_resumed_run_reuses_its_primer(tmp_path, tmp_git_repo):
    from agent_flow.workflows.modeling_v2_onboard.prompts import primer_version

    layout = Layout(tmp_path / "ws")
    layout.workspace.mkdir(parents=True)
    record = {"session_id": "earlier", "commit": "c0ffee", "version": primer_version()}
    layout.primers_path.write_text(json.dumps({"op": record}))

    async def primer(agent, prompt):
        raise AssertionError("the recorded primer is reused")

    factory = FakeFactory(
        planner=_planner(_finish_when_ready_async), worker=_Workers(), primer=primer
    )
    workflow = _workflow(tmp_path, tmp_git_repo, factory, primer=True)
    # A fresh run: the workspace holds only the primer record.
    assert workflow.run() == RunOutcome.SUCCEEDED
    (worker,) = factory.workers("a")
    assert worker.fork_from == "earlier"
    assert "at commit `c0ffee`" in worker.prompts[0]


def test_op_workers_wait_for_the_primer_only_so_long(tmp_path, tmp_git_repo, monkeypatch):
    from agent_flow.workflows.modeling_v2_onboard import workflow as workflow_module

    monkeypatch.setattr(workflow_module, "PRIMER_WAIT_MINUTES", 0.001)
    stuck = asyncio.Event()

    async def primer(agent, prompt):
        await stuck.wait()

    factory = FakeFactory(
        planner=_planner(_finish_when_ready_async), worker=_Workers(), primer=primer
    )
    assert _workflow(tmp_path, tmp_git_repo, factory, primer=True).run() == RunOutcome.SUCCEEDED
    (worker,) = factory.workers("a")
    assert worker.fork_from is None
    (primer_agent,) = [a for a in factory.agents if a.name == "primer[op]"]
    # The run does not leave the primer behind.
    assert primer_agent.closed


def test_clean_deletes_the_primer_and_worker_sessions(tmp_path, tmp_git_repo, monkeypatch):
    from agent_flow.workflows.modeling_v2_onboard.workflow import _session_dir

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    workflow = _workflow(tmp_path, tmp_git_repo, FakeFactory())
    layout = workflow.layout
    layout.workspace.mkdir(parents=True)
    layout.primers_path.write_text(json.dumps({"op": {"session_id": "p1"}}))
    primer_file = _session_dir(workflow.repo_path) / "p1.jsonl"
    other_file = _session_dir(workflow.repo_path) / "planner.jsonl"
    worker_dir = _session_dir(layout.worktrees_dir / "op-a")
    for path in (primer_file, other_file, worker_dir / "w.jsonl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    workflow.clean()
    assert not primer_file.exists() and not worker_dir.exists()
    assert other_file.exists()
    assert not layout.primers_path.exists()


async def _finish_when_ready_async(agent, prompt):
    _finish_when_ready(prompt)


def test_an_invalid_plan_is_sent_back_until_it_validates(tmp_path, tmp_git_repo):
    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            plan = _first_batch(prompt)
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


def test_a_human_message_wakes_the_planner_and_is_acted_on(tmp_path, tmp_git_repo, monkeypatch):
    from agent_flow.workflows.modeling_v2_onboard import workflow as workflow_module
    from agent_flow.workflows.modeling_v2_onboard.files import post_feedback

    monkeypatch.setattr(workflow_module, "INBOX_POLL_SECONDS", 0.05)
    graph = """\
  - id: a
    type: new_op
  - id: busywork
    type: module_accuracy
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a, busywork]
"""
    without_busywork = """\
  - id: a
    type: new_op
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a]
"""
    release_a = asyncio.Event()

    async def a(agent, prompt, attempt):
        # Nothing finishes until the human's message has been handled.
        post_feedback(tmp_path / "ws", "busywork is useless: remove it.")
        await release_a.wait()

    async def busywork(agent, prompt, attempt):
        await asyncio.Event().wait()

    turns: list[str] = []

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(_plan_text(graph))
            return
        turns.append(prompt)
        if "busywork is useless" in prompt:
            _edit_plan(prompt).write_text(_plan_text(without_busywork))
            _decide(prompt, "FINISH", cancel=["busywork"])
            release_a.set()
        else:
            _decide(prompt)

    workers = _Workers(a=a, busywork=busywork)
    factory = FakeFactory(planner=planner, worker=workers)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)

    assert workflow.run() == RunOutcome.SUCCEEDED
    assert turns[0].startswith("## Messages from the human")
    assert "busywork is useless" not in "".join(turns[1:])
    layout = Layout(tmp_path / "ws")
    assert list(layout.inbox_dir.glob("*.md")) == []
    assert len(list(layout.consumed_dir.glob("*.md"))) == 1
    # busywork may be cancelled before its worker even starts.
    assert all(agent.closed for agent in factory.workers("busywork"))


def _task_state(tmp_path: Path, task_id: str) -> str | None:
    return json.loads(Layout(tmp_path / "ws").state_path.read_text())["tasks"].get(task_id)


async def _until(condition) -> None:
    for _ in range(2000):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


def _finish_once_done(*task_ids: str):
    """FINISH as soon as ``task_ids`` are done, whatever else is running."""

    async def on_turn(agent, prompt):
        states = _states(prompt)
        _decide(prompt, "FINISH" if all(states[t] == "done" for t in task_ids) else "CONTINUE")

    return on_turn


# s is side work on a's code; t is side work on s's code. land only needs a.
_SIDE_GRAPH = """\
  - id: a
    type: new_op
  - id: s
    type: perf
    depends_on: [a]
    side: true
  - id: t
    type: perf
    depends_on: [s]
    side: true
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a]
"""


def test_side_work_lands_beside_the_run_and_is_listed(tmp_path, tmp_git_repo):
    async def s(agent, prompt, attempt):
        # Still running when the final task lands.
        await _until(lambda: _task_state(tmp_path, "land") == "done")
        assert "**side task**" in prompt

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(_plan_text(_SIDE_GRAPH, notes={"s": "Fuse the MoE."}))
        else:
            await _finish_once_done("a")(agent, prompt)

    workers = _Workers(s=s)
    factory = FakeFactory(planner=planner, worker=workers)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)
    landed = _git(tmp_git_repo, "symbolic-ref", "--short", "HEAD").strip()

    assert workflow.run() == RunOutcome.SUCCEEDED
    # s only finished once land was done; t never started once the run had landed.
    assert _task_state(tmp_path, "s") == "done"
    assert "t" not in workers.attempts
    assert not (tmp_git_repo / "s.txt").exists()
    assert _git(tmp_git_repo, "branch", "--list", "node/*") == ""
    branch = f"{landed}-unmerged/s"
    assert _git(tmp_git_repo, "show", f"{branch}:s.txt") == "s\n"
    assert _git(tmp_git_repo, "branch", "--list", f"{landed}-unmerged/t") == ""

    listing = (tmp_git_repo / "UNMERGED_OPTIMIZATIONS.md").read_text()
    assert f"- Branch: `{branch}`" in listing
    assert "- Result: DONE: s done" in listing
    assert "Fuse the MoE." in listing
    assert "## t\n\n- Branch: none\n- Result: not started\n- Base: s" in listing
    assert _git(tmp_git_repo, "status", "--porcelain") == ""
    assert "Signed-off-by:" in _git(tmp_git_repo, "log", "-1", "--format=%B")


def test_side_work_still_running_is_stopped_after_the_wait(tmp_path, tmp_git_repo):
    async def s(agent, prompt, attempt):
        (agent.cwd / "partial.txt").write_text("half\n")
        await asyncio.Event().wait()

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(_plan_text(_SIDE_GRAPH))
        else:
            await _finish_once_done("a")(agent, prompt)

    workers = _Workers(s=s)
    factory = FakeFactory(planner=planner, worker=workers)
    workflow = _workflow(tmp_path, tmp_git_repo, factory, side_wait_minutes=0.002)
    landed = _git(tmp_git_repo, "symbolic-ref", "--short", "HEAD").strip()

    assert workflow.run() == RunOutcome.SUCCEEDED
    assert _task_state(tmp_path, "s") == "failed"
    assert factory.workers("s")[0].closed
    # What s had done is kept on its branch.
    assert _git(tmp_git_repo, "show", f"{landed}-unmerged/s:partial.txt") == "half\n"
    listing = (tmp_git_repo / "UNMERGED_OPTIMIZATIONS.md").read_text()
    assert "- Result: FAILED: stopped 0.002 minutes after the final task was DONE" in listing


def test_side_work_is_merged_after_landing_when_asked(tmp_path, tmp_git_repo):
    merged = """\
  - id: a
    type: new_op
  - id: s
    type: perf
    depends_on: [a]
  - id: land
    type: merge
    kind: merge
    isolation: shared
    depends_on: [a]
  - id: merge-s
    type: merge
    kind: merge
    depends_on: [land, s]
  - id: land-2
    type: merge
    kind: merge
    isolation: shared
    depends_on: [merge-s]
"""
    side_only = _SIDE_GRAPH.replace(
        "  - id: t\n    type: perf\n    depends_on: [s]\n    side: true\n", ""
    )
    turns: list[str] = []

    async def s(agent, prompt, attempt):
        await _until(lambda: "land-2" in Layout(tmp_path / "ws").running_plan_path.read_text())

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(_plan_text(side_only))
            return
        turns.append(prompt)
        states = _states(prompt)
        if states.get("land") == "done" and "land-2" not in states:
            assert "The final task is DONE" in prompt
            _edit_plan(prompt).write_text(_plan_text(merged))
            _decide(prompt)
        else:
            await _finish_once_done("a")(agent, prompt)

    workers = _Workers(s=s)
    factory = FakeFactory(planner=planner, worker=workers)
    workflow = _workflow(tmp_path, tmp_git_repo, factory)

    assert workflow.run() == RunOutcome.SUCCEEDED
    assert workers.order[-1] == "land-2"
    assert (tmp_git_repo / "s.txt").read_text() == "s\n"
    (land_2,) = factory.workers("land-2")
    assert land_2.cwd == tmp_git_repo.absolute() and "final task" in land_2.prompts[0]
    assert not (tmp_git_repo / "UNMERGED_OPTIMIZATIONS.md").exists()
    assert _git(tmp_git_repo, "branch", "--list", "*unmerged*") == ""


def test_the_final_task_cannot_be_cancelled(tmp_path, tmp_git_repo):
    land_started, refused = asyncio.Event(), asyncio.Event()
    fixes: list[str] = []

    async def land(agent, prompt, attempt):
        land_started.set()
        await refused.wait()

    async def s(agent, prompt, attempt):
        await land_started.wait()

    async def planner(agent, prompt):
        if "Write the plan to" in prompt:
            _first_batch(prompt).write_text(
                _plan_text(
                    _SIDE_GRAPH.replace(
                        "  - id: t\n    type: perf\n    depends_on: [s]\n    side: true\n", ""
                    )
                )
            )
            return
        if "cannot be run as written" in prompt:
            fixes.append(prompt)
            refused.set()
            return
        if _states(prompt).get("land") == "running" and not refused.is_set():
            _decide(prompt, cancel=["land"])
        else:
            await _finish_once_done("a")(agent, prompt)

    workers = _Workers(s=s, land=land)
    workflow = _workflow(tmp_path, tmp_git_repo, FakeFactory(planner=planner, worker=workers))

    assert workflow.run() == RunOutcome.SUCCEEDED
    assert "cannot be abandoned" in fixes[0]
    assert workers.attempts["land"] == 1
