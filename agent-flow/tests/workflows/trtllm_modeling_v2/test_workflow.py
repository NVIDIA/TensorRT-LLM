# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the workflow's lifecycle: preflight, the execution stage, delivery, tune.

No agent is ever constructed: ``_make_agent`` is replaced by a fake that
writes the decision file the real role would write -- and, for the planner,
the graph. That is the whole contract between the orchestrator and its
agents, so a fake that honours it exercises the real control flow.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as vocab
from agent_flow.workflows.trtllm_modeling_v2.operators import prior_art
from agent_flow.workflows.trtllm_modeling_v2.run import graph as graph_mod
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.records.state import (
    STAGE_EXECUTION,
    load_state,
    save_state,
)
from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from .test_delivery import TARGET_REL

_STATE = ".trtllm_modeling_v2_state.json"


# --------------------------------------------------------------------- fakes


_DECISION_PATH = re.compile(r"Write your decision file to exactly this path[^`]*`([^`]+)`")


class FakeLayer:
    """Stands in for one ``AgentLayer``. Writes the decision file the real role would write."""

    def __init__(self, name, script, calls):
        self.name = name
        self.script = script
        self.calls = calls

    async def aforward(self, prompt: str) -> str:
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

    def __init__(self, ctx, plan=("DRAFT_READY",), **roles):
        self.ctx = ctx
        self.plan = list(plan)
        self.roles = roles

    def __call__(self, role, turn, prompt):
        override = self.roles.get(role)
        if role == "planner" and not callable(override):
            return plan_every_unit(turn, prompt)
        if role == "surveyor" and not callable(override):
            return {"decision": self.plan[min(turn, len(self.plan)) - 1], "summary": "drafted"}
        if callable(override):
            return override(turn, prompt)
        if role == "kernel_writer":
            target = _target_from(prompt)
            return {"decision": "LANDED", "summary": f"landed {target}", "entry": target}
        if role == "assembler":
            return {"decision": "UNIT_READY", "summary": "assembled"}
        raise AssertionError(f"unscripted role: {role}")


def _target_from(prompt: str) -> str:
    for token in prompt.split("`"):
        if token and " " not in token and not token.startswith("/"):
            return token
    return "unknown"


def _install(monkeypatch, workflow, script):
    calls: list[tuple[str, str]] = []

    def fake_make_agent(
        self, name, system_prompt, *, requires_decision=True, session_mode="stateless"
    ):
        return FakeLayer(name, script, calls)

    monkeypatch.setattr(TrtllmModelingV2Workflow, "_make_agent", fake_make_agent)
    return calls


def fake_git(*, head: str = "c0ffee1", dirt: str = ""):
    """Answer the git questions the orchestrator asks of the checkout, as `vcs.git_output` does."""

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


def passing_shell(score: float = 90.0, accepted: int = 320, iterations: int = 100):
    def runner(command, env=None, timeout=0):
        if "trtllm-eval" in command:
            return CommandResult(returncode=0, stdout=f"lm-eval gsm8k average accuracy: {score}\n")
        # The gates that need per-request structure run the engine driver,
        # which prints one record per prompt; the acceptance gate wants 32 of them.
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
    # The substrate gate issues a boot-shaped command of its own, which a shell
    # scripted to fail a specific boot would answer; its test turns it on.
    kwargs.setdefault("skip_substrate_gate", True)
    # A primer turn per role would answer from the same scripted agents.
    kwargs.setdefault("primers", False)
    return TrtllmModelingV2Workflow(workspace=tmp_path / "ws", **kwargs)


def test_the_run_puts_its_skills_where_each_agent_finds_them(tmp_path):
    workflow = _workflow(tmp_path)
    workflow._materialize_skills()
    for skills in (".claude/skills", ".agents/skills"):
        for name in ("trtllm-modeling-v2-entry", "trtllm-modeling-v2-features"):
            skill = workflow.workspace / skills / name / "SKILL.md"
            assert skill.read_text().startswith(f"---\nname: {name}\n")
    assert (
        workflow.workspace / ".agents/skills/trtllm-modeling-v2-features/references/features.md"
    ).is_file()


def test_the_feature_skill_names_only_paths_the_tree_has():
    """The skill and the workers' reuse rule are maps of paths: a path the tree lost is stale."""
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import _common, worker
    from agent_flow.workflows.trtllm_modeling_v2.agents.roles import PACKAGED_SKILLS_DIR

    root = Path(__file__).resolve().parents[4]
    if not (root / "tensorrt_llm").is_dir():
        pytest.skip("not inside a TensorRT-LLM checkout")
    skill = PACKAGED_SKILLS_DIR / "trtllm-modeling-v2-features"
    text = "\n".join(
        [
            *(p.read_text() for p in skill.rglob("*.md")),
            Path(worker.__file__).read_text(),
            Path(_common.__file__).read_text(),
        ]
    )
    named = set()
    for found in re.findall(r"`((?:_torch|tensorrt_llm|tests|cpp|examples)/[^`\s]+)`", text):
        path = found.split("::")[0].rstrip(".,;:)")
        path = "tensorrt_llm/" + path if path.startswith("_torch/") else path
        if not any(ch in path for ch in "*<{["):
            named.add(path)
    missing = sorted(p for p in named if not (root / p).exists())
    assert not missing, f"the feature skill names paths this tree lacks: {missing}"


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


def _substrate_boot(command: str) -> bool:
    return "TRTLLM_MODELING_V2=off" in command and "engine_driver" in command


def test_the_substrate_gate_runs_beside_the_first_turns_and_its_failure_leaves_the_run_open(
    tmp_path, task_file, monkeypatch
):
    planner_called, booted = threading.Event(), []

    def failing(command, env=None, timeout=0):
        if _substrate_boot(command):
            booted.append(command)
            # The gate answers only once the first turn is under way: it does not hold it back.
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

    workflow = _workflow(tmp_path, shell_runner=failing, skip_substrate_gate=False)
    calls = _install(monkeypatch, workflow, Script(None, planner=planner))
    with workflow:
        workflow.run(task_file)
    assert booted, "the substrate gate never ran"
    # The graph the turn wrote is never adopted: nothing starts after the stop.
    assert [name for name, _ in calls] == ["planner"]
    state = _state(tmp_path)
    assert not state.done and state.verdict == "" and state.stage == STAGE_EXECUTION
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "substrate gate failed" in journal and "stopped: the substrate gate failed" in journal
    assert "run closed" not in journal

    # Fixed outside the run, the same command proves the host again and goes on.
    second = _workflow(tmp_path, skip_substrate_gate=False)
    _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)
    assert _state(tmp_path).verdict == "delivered"


def test_a_survey_that_writes_what_the_tree_has_is_sent_back_once(tmp_path, task_file, monkeypatch):
    """The review holds the draft to the dispatch's prior art; the re-draft is taken as it is."""
    monkeypatch.setattr(TrtllmModelingV2Workflow, "survey_first", True)
    lead = prior_art.Candidate(
        path="tensorrt_llm/_torch/attention/backends/sparse/qsa/kernels.py",
        kind="attention backend",
        matched=("gqa", "sparse", "topk"),
        score=9.0,
        names=("triton_qsa_paged_sparse_gqa",),
    )

    def take(workflow, config):
        workflow._prior_art = prior_art.PriorArt(words=lead.matched, candidates=[lead])
        workflow._engine_limits = []
        workflow._prior_art_block = prior_art.survey_block(workflow._prior_art, [])

    monkeypatch.setattr(bootstrap, "_take_prior_art", take)
    workspace = tmp_path / "ws"
    prompts = []

    def surveyor(turn, prompt):
        prompts.append(prompt)
        tried = (
            "no trtllm op or _torch/modules callable"
            if turn == 1
            else ("qsa/kernels.py takes power-of-two heads only")
        )
        _write_vocabulary(workspace, [{"n": 1, "mode": "onboard", "target": "ops/norm/rms"}])
        doc = yaml.safe_load((workspace / "vocabulary.yaml").read_text())
        doc["steps"].append(
            {
                "step": "sparse GQA attention over the selected regions",
                "values": {"topk": 512},
                "state": "GAP",
                "tier": 5,
                "tried": tried,
            }
        )
        (workspace / "vocabulary.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
        return {"decision": "DRAFT_READY", "summary": f"draft {turn}"}

    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None, surveyor=surveyor))
    with workflow:
        workflow.run(task_file)
    assert [name for name, _ in calls].count("surveyor") == 2
    assert "Prior art and engine limits" in prompts[0] and "Survey review" not in prompts[0]
    assert "## Survey review: re-draft" in prompts[1] and "runs once" in prompts[1]
    assert "qsa/kernels.py" in prompts[1]
    assert (workspace / "survey_review.md").is_file()
    assert "survey review: sent back with 1 finding(s)" in (workspace / "journal.md").read_text()
    assert _state(tmp_path).verdict == "delivered"


def test_a_new_run_starts_with_the_survey_beside_the_planners_first_turn(
    tmp_path, task_file, monkeypatch
):
    """Its rows become kernel tasks the moment it lands, with no planner turn between."""
    monkeypatch.setattr(TrtllmModelingV2Workflow, "survey_first", True)
    workspace = tmp_path / "ws"
    drafted, seen = threading.Event(), {}

    def surveyor(turn, prompt):
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
    calls = _install(monkeypatch, workflow, Script(None, planner=planner, surveyor=surveyor))
    with workflow:
        workflow.run(task_file)
    assert seen == {"the survey ran during the first turn": True, "the request named it": True}
    assert {"surveyor", "kernel_writer", "assembler"} <= {name for name, _ in calls}
    graph = graph_mod.Graph.load(workspace / graph_mod.GRAPH_FILENAME)
    survey, kernel = graph.by_id()["survey"], graph.by_id()["k1-rms"]
    assert survey.ref == TARGET_REL and survey.status == "done"
    assert kernel.touches[0] == f"{TARGET_REL}/ops/norm/rms.py" and kernel.status == "done"
    assert _state(tmp_path).verdict == "delivered"


# ----------------------------------------------------------------- execution stage


def test_a_planned_run_is_delivered_once_its_contract_holds_and_a_resume_redoes_nothing(
    tmp_path, task_file, monkeypatch
):
    first = _workflow(tmp_path)
    calls = _install(monkeypatch, first, Script(None))
    with first:
        first.run(task_file)
    assert [name for name, _ in calls] == ["planner", "assembler"]
    state = _state(tmp_path)
    assert state.done and state.verdict == "delivered" and state.stage == STAGE_EXECUTION
    graph = graph_mod.Graph.load(tmp_path / "ws" / graph_mod.GRAPH_FILENAME)
    assert [(t.id, t.status) for t in graph.tasks] == [("core", "done")]
    assert {(v["gate"], v["passed"]) for v in _verdicts(tmp_path)} == {
        ("boot", True),
        ("accuracy[gsm8k]", True),
    }
    envelope = json.loads((tmp_path / "ws" / "tasks" / "core.json").read_text())
    assert envelope["satisfies"] == ["core:boot@tp1", "core:accuracy[gsm8k]@tp1"]

    # Re-open the closed run as though it had crashed before closing.
    state.done, state.verdict = False, ""
    save_state(tmp_path / "ws" / _STATE, state)
    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)
    assert calls == []
    assert _state(tmp_path).verdict == "delivered"


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


def test_a_shared_change_is_followed_by_a_regression_check_with_no_turn(
    tmp_path, task_file, monkeypatch
):
    def planner(turn, prompt):
        if turn == 1:
            assert "## Shared code" in prompt and TARGET_REL in prompt
            core = {
                **unit_tasks(tmp_path / "ws")[0],
                "touches": ["tensorrt_llm/_torch/_experimental/modeling_v2/catalog/norm/rms.py"],
            }
            write_graph(prompt, [core, *unit_tasks(tmp_path / "ws")[1:]])
            return {"decision": "GRAPH_UPDATED", "summary": "core changes a shared entry"}
        return {"decision": "DONE", "summary": "done"}

    checked = []
    workflow = _workflow(tmp_path)
    monkeypatch.setattr(
        ExecutionStage,
        "_run_regression_check",
        lambda self, task: checked.append((task.id, task.covers)) or [],
    )
    calls = _install(monkeypatch, workflow, Script(None, planner=planner))
    with workflow:
        workflow.run(task_file)
    assert _state(tmp_path).verdict == "delivered"
    # It covered what core landed.
    assert checked == [("regression", ["core@1"])]
    graph = yaml.safe_load((tmp_path / "ws" / "graph.yaml").read_text())
    check = next(t for t in graph["tasks"] if t["id"] == "regression")
    assert check["kind"] == "check" and check["regression"] and check["status"] == "done"
    assert check["depends_on"] == ["core"]
    # No agent turn ran for it, and the graph says who added it.
    assert [name for name, _ in calls] == ["planner", "assembler"]
    assert any(
        r["reason"].startswith("orchestrator: regression checks") for r in graph["revisions"]
    )


# ----------------------------------------------------------------------- tune


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
    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None, planner=_plan_with_tune))
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
