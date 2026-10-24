# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the workflow's lifecycle: preflight, the planned run, delivery, tune.

No agent is ever constructed: ``_make_agent`` is replaced by a fake that
writes the decision file the real role would write -- and, for the planner,
the graph. That is the whole contract between the orchestrator and its
agents, so a fake that honours it exercises the real control flow.
"""

from __future__ import annotations

import itertools
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.onboard import vocabulary as vocab
from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run import graph as graph_mod
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod
from agent_flow.workflows.trtllm_modeling_v2.run import progress as progress_mod
from agent_flow.workflows.trtllm_modeling_v2.run.stages.plan import PlanStage
from agent_flow.workflows.trtllm_modeling_v2.run.state import (
    STAGE_PLANNED,
    STAGE_PREFLIGHT,
    load_state,
)
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from .test_delivery import TARGET_REL

_SCRATCH = "/scratch/mv2flow-run-tmp"
_STATE = ".trtllm_modeling_v2_state.json"


# --------------------------------------------------------------------- fakes


_DECISION_PATH = re.compile(r"Write your decision file to exactly this path[^`]*`([^`]+)`")


class FakeLayer:
    """Stands in for one ``AgentLayer``. Writes the decision file the real role would write."""

    def __init__(self, name, script, calls):
        self.name = name
        self.script = script
        self.calls = calls

    def __call__(self, prompt: str) -> str:
        self.calls.append((self.name, prompt))
        turn = sum(1 for n, _ in self.calls if n == self.name)
        entry = self.script(self.name, turn, prompt)
        if entry is not None:
            match = _DECISION_PATH.search(prompt)
            assert match, f"{self.name}'s dispatch reserved no decision file"
            Path(match.group(1)).write_text(yaml.safe_dump(dict(entry)), encoding="utf-8")
        return f"{self.name} turn {turn} done"

    def __exit__(self, *args) -> None:
        return None


class Script:
    """Per-role scripted decisions, keyed by role and turn number."""

    def __init__(self, ctx, plan=("DRAFT_READY",), review=("APPROVE",), **roles):
        self.ctx = ctx
        self.plan = list(plan)
        self.review = list(review)
        self.roles = roles

    def __call__(self, role, turn, prompt):
        override = self.roles.get(role)
        if role == "planner" and not callable(override):
            return plan_every_unit(turn, prompt)
        if role == "plan_drafter" and not callable(override):
            return {"decision": self.plan[min(turn, len(self.plan)) - 1], "summary": "drafted"}
        if role == "plan_reviewer":
            return {"decision": self.review[min(turn, len(self.review)) - 1], "summary": "reviewed"}
        if callable(override):
            return override(turn, prompt)
        if role == "smith":
            target = _target_from(prompt)
            return {"decision": "LANDED", "summary": f"landed {target}", "entry": target}
        if role == "assembler":
            return {"decision": "UNIT_READY", "summary": "assembled"}
        if role == "reviewer":
            return {"decision": "APPROVE", "summary": "looks right"}
        raise AssertionError(f"unscripted role: {role}")


def _target_from(prompt: str) -> str:
    for token in prompt.split("`"):
        if token and " " not in token and not token.startswith("/"):
            return token
    return "unknown"


def _install(monkeypatch, workflow, script):
    calls: list[tuple[str, str]] = []
    script.ctx = workflow._progress_ctx

    def fake_make_agent(
        self, name, system_prompt, *, requires_decision=True, session_mode="stateless"
    ):
        return FakeLayer(name, script, calls)

    monkeypatch.setattr(TrtllmModelingV2Workflow, "_make_agent", fake_make_agent)
    return calls


def fake_git(*, head: str = "c0ffee1", dirt: str = ""):
    """Answer the git questions the orchestrator asks of the checkout.

    `TrtllmModelingV2Workflow._target_marker` asks which commit last touched
    the target and whether anything is uncommitted under it. It is asked on the orchestrator's
    own host through its ``git_runner`` -- `vcs.git_output`'s shape, stdout or
    ``None`` -- and never through the gate runner, so no shell fake here
    answers git: one that did would be counting a cheap question as a gate.
    """

    def run(repo, *args, stdin=None):
        if args[:2] == ("log", "-1"):
            return head + "\n"
        if args[:2] == ("status", "--porcelain"):
            return dirt
        if args[:1] == ("rev-parse",):
            return head + "\n"
        return ""

    return run


def healthy_smi(count: int = 1):
    rows = "\n".join(f"{i}, 180000, 183359, 0, 10.3" for i in range(count))

    def runner(argv, env=None, timeout=0):
        if argv and argv[0] == "nvidia-smi":
            return CommandResult(returncode=0, stdout=rows + "\n")
        return CommandResult(returncode=0, stdout="ok\n")

    return runner


def driver_records(accepted: int = 320, iterations: int = 100, prompts: int = 32) -> str:
    """What the engine driver prints for ``prompts`` prompts accepting alike."""
    lines = []
    for n in range(prompts):
        record = {
            "index": n,
            "id": f"d{n}",
            "prompt_sha": f"p{n}",
            "tokens": accepted,
            "iterations": iterations,
            "token_ids": list(range(accepted)),
            "text": "Paris is the capital of France",
        }
        lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
    return "\n".join(lines) + "\n"


def other_driver_gate(command: str) -> bool:
    """A driver gate other than boot, which a boot-counting fake passes through."""
    return "engine_driver.py" in command and "/boot." not in command


def passing_shell(score: float = 90.0, accepted: int = 320, iterations: int = 100):
    def runner(command, env=None, timeout=0):
        if "trtllm-eval" in command:
            return CommandResult(returncode=0, stdout=f"lm-eval gsm8k average accuracy: {score}\n")
        # The gates that need per-request structure run the engine driver,
        # which prints one record per prompt; the acceptance gate wants 32 of
        # them. The fake has to print what the real driver prints, or every
        # speculator unit is unpassable for a reason that looks like a broken
        # draft path.
        if "engine_driver.py" in command:
            return CommandResult(returncode=0, stdout=driver_records(accepted, iterations))
        return CommandResult(
            returncode=0,
            stdout=(
                "Paris is the capital of France\n"
                f"[0] Accept rate: {accepted / iterations:.2f} "
                f"(tokens={accepted}, iterations={iterations})\n"
            ),
        )

    return runner


def _workflow(tmp_path, **kwargs):
    kwargs.setdefault("command_runner", healthy_smi())
    kwargs.setdefault("shell_runner", passing_shell())
    # Scripted like every other outside answer: the fixture repo is not a
    # checkout. An explicit `git_runner=None` selects the real, local git.
    kwargs.setdefault("git_runner", fake_git())
    # These tests drive stages other than preflight, and several use a shell
    # runner scripted to fail a specific boot. The substrate gate issues a
    # boot-shaped command of its own, so it would consume those scripts.
    # It has dedicated coverage in test_substrate_gate.py, plus the two
    # end-to-end cases below.
    kwargs.setdefault("skip_substrate_gate", True)
    # The critic adds a turn to every first graph; it has its own test below.
    kwargs.setdefault("graph_critic", False)
    return TrtllmModelingV2Workflow(workspace=tmp_path / "ws", **kwargs)


def _write_vocabulary(workspace: Path, queue):
    doc = {
        "verdict": vocab.VERDICT_AFTER_ADDITIONS if queue else vocab.VERDICT_ASSEMBLABLE,
        "steps": [
            {
                "step": "norm",
                "state": vocab.STATE_OK,
                "entry": "catalog/norm/flashinfer_rmsnorm.py",
                "contract_line": "certified sm_103 bf16",
            }
        ],
        "queue": queue,
    }
    (workspace / "vocabulary.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")


_GRAPH_PATH = re.compile(r"Write the next graph to `([^`]+)`")


def planner_graph_path(prompt: str) -> Path:
    match = _GRAPH_PATH.search(prompt)
    assert match, "the planner's request names no graph path"
    return Path(match.group(1))


def write_graph(prompt: str, tasks: list[dict]) -> None:
    planner_graph_path(prompt).write_text(yaml.safe_dump({"tasks": tasks}), encoding="utf-8")


def unit_tasks(workspace: Path) -> list[dict]:
    """One assemble/feature task per unit, in order, claiming that unit's checks."""
    contract = yaml.safe_load((workspace / contract_mod.CONTRACT_FILENAME).read_text())
    units: dict[str, list[str]] = {}
    for check in contract["checks"]:
        units.setdefault(check["unit"], []).append(
            f"{check['unit']}:{check['gate']}@{check['member']}"
        )
    tasks, previous = [], None
    for unit, checks in units.items():
        tasks.append(
            {
                "id": unit,
                "kind": "assemble" if unit == "core" else "feature",
                "goal": f"build {unit}",
                "ref": unit,
                "depends_on": [previous] if previous else [],
                # The target's own code: no cross-model regression is owed for it.
                "touches": [f"{TARGET_REL}/{unit}.py"],
                "satisfies": checks,
            }
        )
        previous = unit
    return tasks


def plan_every_unit(turn: int, prompt: str) -> dict:
    """The default planner: one task per unit on its first turn, DONE after."""
    if turn == 1:
        workspace = planner_graph_path(prompt).parent
        write_graph(prompt, unit_tasks(workspace))
        return {"decision": "GRAPH_UPDATED", "summary": "one task per unit"}
    return {"decision": "DONE", "summary": "the contract should hold"}


def _git(repo: Path, *args: str) -> str:
    """Run one git command in ``repo``, with an identity the host may not have."""
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=mv2flow",
            "-c",
            "user.email=mv2flow@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(repo),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


#: Job numbers a fake scheduler hands out. Shared by every fake, so no two
#: jobs in a session -- in one process or across a resume -- share a name.
_JOB_NUMBERS = itertools.count(1)


def _unified_task(task_file, members):
    """Rewrite the task spec as a unified run over `members`.

    World size 1 so the harness stays the one every other test here uses: the
    device claim sees one card and the multi-rank substrate probe short-
    circuits. What is under test is the LOOP, not the allocation.
    """
    spec = yaml.safe_load(task_file.read_text())
    spec["parallel"] = "unified"
    spec["members"] = members
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")


def _with_perf(task_file, **perf):
    data = yaml.safe_load(task_file.read_text())
    data["perf"] = {"workload": {"isl": 1024, "osl": 128, "concurrency": [1, 8]}, **perf}
    task_file.write_text(yaml.safe_dump(data))
    return task_file


def _state(tmp_path):
    return load_state(tmp_path / "ws" / _STATE)


def _verdicts(tmp_path) -> list[dict]:
    path = tmp_path / "ws" / contract_mod.STATUS_FILENAME
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------- preflight


def test_preflight_failure_ends_the_run_before_any_agent(tmp_path, task_file, monkeypatch):
    def broken(argv, env=None, timeout=0):
        return CommandResult(returncode=0, stdout="0, 20000, 183359, 99, 10.3\n")

    workflow = _workflow(tmp_path, command_runner=broken)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    assert calls == []
    assert "preflight failed" in (tmp_path / "ws" / "journal.md").read_text()


def test_dry_run_stops_before_the_first_agent_turn(tmp_path, task_file, monkeypatch):
    """And it leaves the checkpoint at preflight, so the real run re-runs it."""
    workflow = _workflow(tmp_path, dry_run=True)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    assert calls == []
    assert _state(tmp_path).stage == STAGE_PREFLIGHT


def test_preflight_resolves_the_target_path_and_the_contract(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path, dry_run=True)
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    assert workflow._preflight.target_rel == TARGET_REL
    contract = yaml.safe_load((tmp_path / "ws" / contract_mod.CONTRACT_FILENAME).read_text())
    assert {(c["unit"], c["gate"]) for c in contract["checks"]} == {
        ("core", "boot"),
        ("core", "accuracy[gsm8k]"),
    }


def _substrate_boot(command: str) -> bool:
    return "TRTLLM_MODELING_V2=off" in command and "quickstart" in command


def test_the_substrate_gate_boots_beside_the_first_turns(tmp_path, task_file, monkeypatch):
    """Phase 0: the planner's first turn does not wait for preflight's GPU check."""
    planner_called, seen = threading.Event(), {}

    def shell(command, env=None, timeout=0):
        if _substrate_boot(command):
            seen["a turn ran while the gate booted"] = planner_called.wait(10)
        return passing_shell()(command, env, timeout)

    def planner(turn, prompt):
        planner_called.set()
        return plan_every_unit(turn, prompt)

    workflow = _workflow(tmp_path, shell_runner=shell, skip_substrate_gate=False)
    _install(monkeypatch, workflow, Script(None, planner=planner))
    with workflow:
        workflow.run(task_file)
    assert seen["a turn ran while the gate booted"]
    assert _state(tmp_path).verdict == "delivered"
    assert "**substrate gate passed**" in (tmp_path / "ws" / "journal.md").read_text()
    # What it measured reaches the host facts once it is in.
    assert "substrate gate: stock path generated" in workflow.host_facts_path.read_text()


def test_a_failing_substrate_gate_stops_the_run_and_leaves_it_to_resume(
    tmp_path, task_file, monkeypatch
):
    planner_called = threading.Event()

    def failing(command, env=None, timeout=0):
        if _substrate_boot(command):
            planner_called.wait(10)
            return CommandResult(returncode=1, stdout="MPI_ERR_SPAWN: could not spawn processes")
        return passing_shell()(command, env, timeout)

    workspace = (tmp_path / "ws").resolve()

    def planner(turn, prompt):
        planner_called.set()
        deadline = time.monotonic() + 10
        while not orchestrator_mod._STOPS[workspace].done():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        return plan_every_unit(turn, prompt)

    workflow = _workflow(
        tmp_path, shell_runner=failing, skip_substrate_gate=False, lessons_dir=tmp_path / "lessons"
    )
    calls = _install(
        monkeypatch,
        workflow,
        Script(None, planner=planner, improver=lambda t, p: {"decision": "NOTHING_NEW"}),
    )
    with workflow:
        workflow.run(task_file)
    # The graph the turn wrote is never adopted: nothing starts after the stop, and
    # the run is not closed, so nothing wakes the improver.
    assert [name for name, _ in calls] == ["planner"]
    state = _state(tmp_path)
    assert not state.done and state.verdict == "" and state.stage == STAGE_PLANNED
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "substrate gate failed" in journal and "stopped: the substrate gate failed" in journal
    assert "run closed" not in journal

    # Fixed outside the run, the same command proves the host again and goes on.
    second = _workflow(tmp_path, skip_substrate_gate=False)
    _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)
    assert _state(tmp_path).verdict == "delivered"


def test_a_new_run_starts_with_the_survey_beside_the_planners_first_turn(
    tmp_path, task_file, monkeypatch
):
    """Its rows become kernel tasks the moment it lands, with no planner turn between."""
    monkeypatch.setattr(PlanStage, "survey_first", True)
    workspace = tmp_path / "ws"
    drafted, seen = threading.Event(), {}

    def drafter(turn, prompt):
        _write_vocabulary(workspace, [{"n": 1, "mode": "onboard", "target": "ops/norm/rms"}])
        drafted.set()
        return {"decision": "DRAFT_READY", "summary": "drafted"}

    def planner(turn, prompt):
        if turn == 1:
            seen["the survey ran during the first turn"] = drafted.wait(10)
            seen["the request named it"] = "the run started with the survey" in prompt
            current = yaml.safe_load((workspace / graph_mod.GRAPH_FILENAME).read_text())
            write_graph(prompt, [*current["tasks"], *unit_tasks(workspace)])
            return {"decision": "GRAPH_UPDATED", "summary": "core beside the survey"}
        return {"decision": "DONE", "summary": "the contract should hold"}

    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None, planner=planner, plan_drafter=drafter))
    with workflow:
        workflow.run(task_file)
    assert seen == {"the survey ran during the first turn": True, "the request named it": True}
    assert {"plan_drafter", "smith", "assembler"} <= {name for name, _ in calls}
    graph = graph_mod.Graph.load(workspace / graph_mod.GRAPH_FILENAME)
    survey, kernel = graph.by_id()["survey"], graph.by_id()["k1-rms"]
    assert survey.ref == TARGET_REL and survey.status == "done"
    assert kernel.touches[0] == f"{TARGET_REL}/ops/norm/rms.py" and kernel.status == "done"
    assert _state(tmp_path).verdict == "delivered"


# ----------------------------------------------------------------- planned run


def test_a_planned_run_is_delivered_once_its_contract_holds(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    assert [name for name, _ in calls] == ["planner", "assembler"]
    state = _state(tmp_path)
    assert state.done and state.verdict == "delivered" and state.stage == STAGE_PLANNED
    graph = graph_mod.Graph.load(tmp_path / "ws" / graph_mod.GRAPH_FILENAME)
    assert [(t.id, t.status) for t in graph.tasks] == [("core", "done")]
    assert {(v["gate"], v["passed"]) for v in _verdicts(tmp_path)} == {
        ("boot", True),
        ("accuracy[gsm8k]", True),
    }
    envelope = json.loads((tmp_path / "ws" / "tasks" / "core.json").read_text())
    assert envelope["satisfies"] == ["core:boot@tp1", "core:accuracy[gsm8k]@tp1"]


def test_the_planner_is_told_what_failed_and_can_escalate(tmp_path, task_file, monkeypatch):
    seen: list[str] = []

    def planner(turn, prompt):
        seen.append(prompt)
        if turn == 1:
            return plan_every_unit(turn, prompt)
        return {"decision": "ESCALATE", "summary": "gsm8k is out of reach on this checkpoint"}

    workflow = _workflow(tmp_path, shell_runner=passing_shell(score=10.0))
    _install(monkeypatch, workflow, Script(None, planner=planner))
    with workflow:
        workflow.run(task_file)
    assert _state(tmp_path).verdict == "escalated"
    assert "checks failed: core:accuracy[gsm8k]@tp1" in seen[1]
    assert "FAILED core:accuracy[gsm8k]@tp1" in seen[1]


def test_a_graph_that_claims_too_little_goes_back_with_the_reason(tmp_path, task_file, monkeypatch):
    seen: list[str] = []

    def planner(turn, prompt):
        seen.append(prompt)
        if turn == 1:
            write_graph(prompt, [{"id": "core", "kind": "assemble", "goal": "core", "ref": "core"}])
            return {"decision": "GRAPH_UPDATED", "summary": "too little"}
        return {"decision": "CEILING", "summary": "nothing serves the attention"}

    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None, planner=planner))
    with workflow:
        workflow.run(task_file)
    assert _state(tmp_path).verdict == "ceiling"
    assert "core:boot@tp1" in seen[1] and "refused" in seen[1]
    assert [name for name, _ in calls] == ["planner", "planner"]


def test_a_survey_that_writes_an_invalid_vocabulary_fails_its_task(
    tmp_path, task_file, monkeypatch
):
    seen: list[str] = []

    def planner(turn, prompt):
        seen.append(prompt)
        if turn == 1:
            write_graph(
                prompt,
                [
                    {"id": "survey", "kind": "survey", "goal": "map it"},
                    {**unit_tasks(tmp_path / "ws")[0], "depends_on": ["survey"]},
                ],
            )
            return {"decision": "GRAPH_UPDATED", "summary": "survey first"}
        return {"decision": "ESCALATE", "summary": "the survey keeps failing"}

    def drafter(turn, prompt):
        (tmp_path / "ws" / "vocabulary.yaml").write_text("not: [a vocabulary\n")
        return {"decision": "DRAFT_READY", "summary": "drafted"}

    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None, planner=planner, plan_drafter=drafter))
    with workflow:
        workflow.run(task_file)
    assert any("vocabulary.yaml does not validate" in p for p in seen[1:])
    assert _state(tmp_path).verdict == "escalated"


def test_a_resumed_run_does_not_redo_finished_work(tmp_path, task_file, monkeypatch):
    first = _workflow(tmp_path)
    _install(monkeypatch, first, Script(None))
    with first:
        first.run(task_file)
    state = _state(tmp_path)
    assert state.verdict == "delivered"

    # Re-open the closed run as though it had crashed before closing.
    from agent_flow.workflows.trtllm_modeling_v2.run.state import save_state

    state.done, state.verdict = False, ""
    save_state(tmp_path / "ws" / _STATE, state)
    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)
    assert [name for name, _ in calls] == []
    assert _state(tmp_path).verdict == "delivered"


def test_host_facts_are_written_and_reach_a_role_prompt(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    facts = (tmp_path / "ws" / "host_facts.md").read_text()
    assert "TRTLLM_SKIP_SYS_PATH_CHECK=1" in facts
    assert _SCRATCH in facts
    for role in ("planner", "assembler"):
        prompt = next(p for name, p in calls if name == role)
        assert str(tmp_path / "ws" / "host_facts.md") in prompt, role


def test_progress_entries_are_typed_and_stage_scoped(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    data = progress_mod.read_progress(tmp_path / "ws" / "progress.yaml")
    assert [e["agent"] for e in data[progress_mod.PLAN_STAGE]] == ["planner"]
    assert [e["agent"] for e in data[progress_mod.BUILD_STAGE]] == ["assembler"]


def test_a_kernel_task_dispatches_smith_for_its_queue_row(tmp_path, task_file, monkeypatch):
    def planner(turn, prompt):
        if turn == 1:
            _write_vocabulary(
                tmp_path / "ws",
                [
                    {
                        "n": 1,
                        "mode": "onboard",
                        "target": "norm/rms",
                        "tier": 1,
                        "candidate": "torch.ops.trtllm.rms",
                    }
                ],  # fmt: skip
            )
            write_graph(
                prompt,
                [
                    {
                        "id": "k1",
                        "kind": "kernel",
                        "goal": "rms",
                        "ref": "1",
                        "touches": [
                            "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/norm/rms.py"
                        ],
                    },  # fmt: skip
                    {**unit_tasks(tmp_path / "ws")[0], "depends_on": ["k1"]},
                    # The catalog is shared code: a cross-model regression review follows it.
                    {
                        "id": "xmodel",
                        "kind": "review",
                        "goal": "other models on the changed catalog entry",
                        "ref": "core",
                        "regression": True,
                        "depends_on": ["k1"],
                    },
                ],
            )
            return {"decision": "GRAPH_UPDATED", "summary": "kernel then core"}
        return {"decision": "DONE", "summary": "done"}

    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None, planner=planner))
    with workflow:
        workflow.run(task_file)
    roles = [name for name, _ in calls]
    # The planner is woken after every finished task; core waits for its kernel.
    assert roles.index("smith") < roles.index("assembler")
    assert "reviewer" in roles
    assert _state(tmp_path).verdict == "delivered"


def test_a_shared_change_without_a_regression_review_is_refused(tmp_path, task_file, monkeypatch):
    """A fresh run binds the cross-model rule: the planner hears why and adds the review."""
    refusals = []

    def planner(turn, prompt):
        if turn == 1:
            assert "## Cross-model regression" in prompt and TARGET_REL in prompt
            core = {
                **unit_tasks(tmp_path / "ws")[0],
                "touches": ["tensorrt_llm/_torch/_experimental/modeling_v2/catalog/norm/rms.py"],
            }
            write_graph(prompt, [core, *unit_tasks(tmp_path / "ws")[1:]])
            return {"decision": "GRAPH_UPDATED", "summary": "no review"}
        if "## Your last graph was refused" in prompt and not refusals:
            refusals.append(prompt)
            return {"decision": "ESCALATE", "summary": "seen"}
        return {"decision": "DONE", "summary": "done"}

    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None, planner=planner))
    with workflow:
        workflow.run(task_file)
    assert refusals and "regression" in refusals[0]


# ----------------------------------------------------------------------- tune


def test_the_tune_entry_starts_a_fresh_workspace_at_tune(tmp_path, task_file, monkeypatch):
    import agent_flow.workflows.trtllm_modeling_v2.tune.stage as stage

    seen = []
    monkeypatch.setattr(
        stage, "run_tune_phase", lambda wf, state, log: seen.append(state.stage) or "tuned"
    )
    workflow = _workflow(tmp_path, entry_stage="tune")
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(_with_perf(task_file))
    state = _state(tmp_path)
    assert seen == ["tune"] and state.done and state.verdict == "tuned"
    assert calls == []
    assert state.tune.get("entry") == "standalone"


def _plan_with_tune(turn: int, prompt: str) -> dict:
    """The core plus a tune task with no dependency: the orchestrator must hold it back."""
    if turn == 1:
        workspace = planner_graph_path(prompt).parent
        tune = {"id": "tune", "kind": "tune", "goal": "tune the core", "resources": {"gpus": 1}}
        write_graph(prompt, [tune, *unit_tasks(workspace)])
        return {"decision": "GRAPH_UPDATED", "summary": "core and tune"}
    return {"decision": "DONE", "summary": "the contract should hold"}


def test_a_tune_task_starts_once_the_core_is_correct(tmp_path, task_file, monkeypatch):
    import agent_flow.workflows.trtllm_modeling_v2.tune.stage as stage

    started_with: list[set] = []

    def fake_tune(wf, state, base, attempt=1, should_stop=lambda: False):
        started_with.append({(v["gate"], v["passed"]) for v in _verdicts(tmp_path)})
        state.tune.update(phase="done", base_commit=base)
        return "TUNED", "tuned from c0ffee1: best branch `tune/x/best`"

    monkeypatch.setattr(stage, "run_tune_task", fake_tune)
    planner_prompts: list[str] = []

    def planner(turn, prompt):
        planner_prompts.append(prompt)
        return _plan_with_tune(turn, prompt)

    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None, planner=planner))
    with workflow:
        workflow.run(_with_perf(task_file, enabled=True))
    assert started_with == [{("boot", True), ("accuracy[gsm8k]", True)}]
    state = _state(tmp_path)
    assert state.verdict == "tuned" and state.tune["phase"] == "done"
    # The campaign's report is an artifact of the run: journaled, with no planner turn to read it.
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "tune reports: tuned from c0ffee1" in journal
    graph = graph_mod.Graph.load(tmp_path / "ws" / graph_mod.GRAPH_FILENAME)
    assert {t.id: t.status for t in graph.tasks} == {"tune": "done", "core": "done"}


def test_perf_enabled_needs_a_tune_task_in_the_graph(tmp_path, task_file, monkeypatch):
    seen: list[str] = []

    def planner(turn, prompt):
        seen.append(prompt)
        if turn == 1:
            return plan_every_unit(turn, prompt)
        return {"decision": "CEILING", "summary": "stop"}

    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None, planner=planner))
    with workflow:
        workflow.run(_with_perf(task_file, enabled=True))
    assert "the graph has no `tune` task" in seen[1]


def test_a_failed_task_teaches_every_later_dispatch(tmp_path, task_file, monkeypatch):
    lessons_dir = tmp_path / "lessons"
    planner_prompts: list[str] = []
    improver_prompts: list[str] = []

    def planner(turn, prompt):
        planner_prompts.append(prompt)
        if turn == 2:
            # The improver reads the failure beside the run; the retry waits for its lesson.
            deadline = time.monotonic() + 10
            while not (lessons_dir / "role_assembler.yaml").is_file():
                assert time.monotonic() < deadline, "the improver never wrote its lesson"
                time.sleep(0.01)
        if turn <= 2:
            tasks = unit_tasks(planner_graph_path(prompt).parent)
            for task in tasks:
                task["id"] += "" if turn == 1 else "-retry"
            write_graph(prompt, tasks)
            return {"decision": "GRAPH_UPDATED", "summary": "core" if turn == 1 else "retry"}
        return {"decision": "DONE", "summary": "the contract should hold"}

    def assembler(turn, prompt):
        if turn == 1:
            return {
                "decision": "BLOCKED",
                "summary": "mpirun hung",
                "blocked_reason": "environment",
            }
        return {"decision": "UNIT_READY", "summary": "assembled"}

    def improver(turn, prompt):
        improver_prompts.append(prompt)
        if turn > 1:
            return {"decision": "NOTHING_NEW", "summary": "nothing"}
        ws = tmp_path / "ws"
        proposal = {
            "lessons": [
                {
                    "scope": "role:assembler",
                    "trigger": "mpirun hangs at startup",
                    "lesson": "launch one task per rank under trtllm-llmapi-launch",
                    "evidence": "journal.md",
                },
                {"scope": "role:nobody", "trigger": "x", "lesson": "y", "evidence": "journal.md"},
            ]
        }
        (ws / "lessons.proposed.yaml").write_text(yaml.safe_dump(proposal), encoding="utf-8")
        return {"decision": "LESSONS_PROPOSED", "summary": "one lesson"}

    workflow = _workflow(tmp_path, lessons_dir=lessons_dir)
    workflow.improver_batch_s = 0.0
    calls = _install(
        monkeypatch, workflow, Script(None, planner=planner, assembler=assembler, improver=improver)
    )
    with workflow:
        # The flow's own measuring run: its improver turns while the run works.
        _measuring(task_file)
        workflow.run(task_file)
    assert _state(tmp_path).verdict == "delivered"
    names = [name for name, _ in calls]
    # The failure wakes the improver beside the run, and the run's close once more.
    assert [n for n in names if n != "improver"] == ["planner", "assembler", "planner", "assembler"]
    assert names.count("improver") == 2 and names[-1] == "improver"
    assert "task `core` (assemble, role assembler) ended BLOCKED" in improver_prompts[0]
    second_assembler = [p for n, p in calls if n == "assembler"][1]
    assert "**When:** mpirun hangs at startup" in second_assembler
    assert "trtllm-llmapi-launch" not in planner_prompts[1]
    assert "verdict `delivered`" in improver_prompts[1]
    stored = yaml.safe_load((lessons_dir / "role_assembler.yaml").read_text())
    assert [s["trigger"] for s in stored] == ["mpirun hangs at startup"]
    assert "role:nobody" in (tmp_path / "ws" / "journal.md").read_text()


def test_without_a_lessons_dir_there_is_no_improver(tmp_path, task_file, monkeypatch):
    def assembler(turn, prompt):
        if turn == 1:
            return {"decision": "BLOCKED", "summary": "x", "blocked_reason": "environment"}
        return {"decision": "UNIT_READY", "summary": "assembled"}

    def planner(turn, prompt):
        if turn <= 2:
            tasks = unit_tasks(planner_graph_path(prompt).parent)
            for task in tasks:
                task["id"] += "" if turn == 1 else "-retry"
            write_graph(prompt, tasks)
            return {"decision": "GRAPH_UPDATED", "summary": "core"}
        return {"decision": "DONE", "summary": "the contract should hold"}

    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None, planner=planner, assembler=assembler))
    with workflow:
        workflow.run(task_file)
    assert "improver" not in [name for name, _ in calls]


def test_an_assemblers_uncertified_calls_reach_the_planner(tmp_path, task_file, monkeypatch):
    """A unit that lands green with calls outside its entries' CELLS tells the planner which.

    The assembler is told to carry on past a correct call no cell covers; the
    certification it needs is the planner's to add. Its summary reaches only the
    receipts of tasks that depend on it, so the call is named in ``uncertified``
    and the task's event carries it.
    """
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import DEFAULT_PROMPTS

    planner_prompts: list[str] = []
    uncertified = (
        "ssm/causal_conv1d_triton_update: 5882 pool slots at width 8448, past its "
        "5600-slot far cell (the MTP verify conv at tp2)"
    )

    def planner(turn, prompt):
        planner_prompts.append(prompt)
        return plan_every_unit(turn, prompt)

    def assembler(turn, prompt):
        return {"decision": "UNIT_READY", "summary": "assembled", "uncertified": [uncertified]}

    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None, planner=planner, assembler=assembler))
    with workflow:
        workflow.run(task_file)
    assert _state(tmp_path).verdict == "delivered"
    later = "\n".join(planner_prompts[1:])
    assert "task core (assemble) reports: " in later
    assert uncertified in later
    assert "  uncertified: ..." in DEFAULT_PROMPTS.assembler


def test_a_delivered_run_is_re_entered_at_tune(tmp_path, task_file, monkeypatch):
    import agent_flow.workflows.trtllm_modeling_v2.tune.stage as stage
    from agent_flow.workflows.trtllm_modeling_v2.run.state import WorkflowState, save_state

    ws = tmp_path / "ws"
    ws.mkdir()
    save_state(
        ws / _STATE,
        WorkflowState(
            task_path=str(ws / "task.yaml"),
            stage=STAGE_PLANNED,
            done=True,
            verdict="delivered",
            tune={"phase": "done", "misses": 3, "kept": ["a"]},
        ),
    )
    seen = []
    monkeypatch.setattr(
        stage, "run_tune_phase", lambda wf, state, log: seen.append(dict(state.tune)) or "tuned"
    )
    workflow = _workflow(tmp_path, entry_stage="tune")
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(_with_perf(task_file))
    assert calls == []
    assert seen and seen[0]["misses"] == 0 and seen[0]["phase"] == "campaign"


def test_the_critic_reads_the_first_graph_and_the_planner_hears_a_rejection(
    tmp_path, task_file, monkeypatch
):
    planner_prompts: list[str] = []

    def planner(turn, prompt):
        planner_prompts.append(prompt)
        if turn == 2:
            return {"decision": "GRAPH_UPDATED", "summary": "keeping the graph"}
        return plan_every_unit(turn, prompt)

    def assembler(turn, prompt):
        # Core runs while the critic reads; it hands back once the planner heard the critic.
        deadline = time.monotonic() + 10
        while len(planner_prompts) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        return {"decision": "UNIT_READY", "summary": "assembled"}

    workflow = _workflow(tmp_path, graph_critic=True)
    calls = _install(
        monkeypatch,
        workflow,
        Script(None, review=("REJECT",), planner=planner, assembler=assembler),
    )
    with workflow:
        workflow.run(task_file)
    # Nothing waits for the critic: core ran beside its read, and finished after the
    # planner heard it.
    assert [name for name, _ in calls][0] == "planner"
    assert {"plan_reviewer", "assembler"} <= {name for name, _ in calls}
    assert _state(tmp_path).verdict == "delivered"
    critic = next(prompt for name, prompt in calls if name == "plan_reviewer")
    assert "graph.critique.yaml" in critic and "paper review" in critic
    assert "critic (advisory) on this graph: REJECT: reviewed" in planner_prompts[1]


def test_an_ordinary_first_failure_does_not_wake_the_improver(tmp_path, task_file, monkeypatch):
    def planner(turn, prompt):
        if turn <= 2:
            tasks = unit_tasks(planner_graph_path(prompt).parent)
            for task in tasks:
                task["id"] += "" if turn == 1 else "-retry"
            write_graph(prompt, tasks)
            return {"decision": "GRAPH_UPDATED", "summary": "core"}
        return {"decision": "DONE", "summary": "the contract should hold"}

    workflow = _workflow(
        tmp_path, lessons_dir=tmp_path / "lessons", shell_runner=passing_shell(score=10.0)
    )
    calls = _install(
        monkeypatch,
        workflow,
        Script(None, planner=planner, improver=lambda t, p: {"decision": "NOTHING_NEW"}),
    )
    with workflow:
        workflow.run(task_file)
    names = [name for name, _ in calls]
    # The first assembler finished but its accuracy failed: the work itself, not a lesson.
    assert names[:3] == ["planner", "assembler", "planner"]


def test_tasks_are_admitted_by_the_live_sessions_capacity(tmp_path, task_file, monkeypatch):
    """A gate filling the nodes for the moment does not hold a task back; reservations do."""
    import time as time_mod

    workflow = _workflow(tmp_path)
    gates, work = tmp_path / "gates", tmp_path / "work"
    gates.mkdir()
    work.mkdir()
    sessions = (("g", str(gates), "gates"), ("w", str(work), "work"), ("b", str(tmp_path), "work"))
    workflow.environment = type("Env", (), {"spool": str(gates), "sessions": sessions})()
    running = graph_mod.Graph(
        tasks=[
            graph_mod.Task(id="core", kind="assemble", goal="x", gpus=2, status="running"),
            graph_mod.Task(id="k1", kind="kernel", goal="y", gpus=4),
        ]
    )
    assert workflow._spool_capacity() is None
    assert workflow._task_capacity(4, running) == 2  # no live session: the site's devices
    busy = {"alive_epoch": time_mod.time(), "devices": ["0", "1", "2", "3"], "free": []}
    (gates / "status.json").write_text(json.dumps({**busy, "quarantined": ["3"]}))
    (work / "status.json").write_text(json.dumps(busy))
    assert workflow._spool_capacity() == 7
    assert workflow._task_capacity(4, running) == 5
    (gates / "status.json").write_text(json.dumps({**busy, "alive_epoch": time_mod.time() - 3600}))
    assert workflow._spool_capacity() == 4
    # A running tune task whose jobs all run on the perf sessions holds none of these devices.
    running.tasks.append(graph_mod.Task(id="tune", kind="tune", goal="t", gpus=4, status="running"))
    assert workflow._task_capacity(4, running) == 2 - 4
    assert workflow._task_capacity(4, running, lambda t: 0 if t.kind == "tune" else t.gpus) == 2


def test_a_run_with_landed_work_keeps_the_rules_it_was_planned_under(tmp_path):
    """The cross-model rule binds a run that starts on it, not one already under way."""
    from types import SimpleNamespace

    workflow = _workflow(tmp_path)
    workflow._preflight = SimpleNamespace(target_rel=TARGET_REL)
    workspace = workflow.workspace
    workspace.mkdir(parents=True, exist_ok=True)
    assert TARGET_REL in workflow._regression_paths()
    graph_mod.Graph(
        tasks=[graph_mod.Task(id="core", kind="assemble", goal="g", status="pending")]
    ).save(workspace / graph_mod.GRAPH_FILENAME)
    assert TARGET_REL in workflow._regression_paths()
    graph_mod.Graph(
        tasks=[graph_mod.Task(id="core", kind="assemble", goal="g", status="done")]
    ).save(workspace / graph_mod.GRAPH_FILENAME)
    assert workflow._regression_paths() == ()


def _measuring(task_file):
    """Mark the task a run that measures the workflow (``knowledge: measurement``)."""
    spec = yaml.safe_load(task_file.read_text(encoding="utf-8"))
    spec["knowledge"] = "measurement"
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
