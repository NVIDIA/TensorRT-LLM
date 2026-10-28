# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receipts: what a finished task hands the tasks that depend on it, with no planner turn between."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.git import git_output
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.execution import handoff as r
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import decision_file_instruction
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.records.journal import Journal
from agent_flow.workflows.trtllm_modeling_v2.run.validation import runtime_facts as rf
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import Integration

from .test_integration import _git
from .test_workflow import (
    Script,
    _install,
    _workflow,
    passing_shell,
    planner_graph_path,
    unit_tasks,
    write_graph,
)

# ------------------------------------------------------- the turn's files


def _journal(tmp_path: Path) -> Journal:
    journal = Journal(path=tmp_path / "journal.md")
    journal.open(slug="t")
    return journal


def _turn(journal: Journal, role: str, what: str, decision: Path, raw: dict, result: str | None):
    """One dispatched turn as the workflow leaves it: its report, its decision file."""
    report = journal.record_dispatch(
        role, what, f"Do {what}.\n\n{decision_file_instruction(decision)}"
    )
    decision.parent.mkdir(parents=True, exist_ok=True)
    decision.write_text(yaml.safe_dump(raw), encoding="utf-8")
    if result is not None:
        journal.record_result(report, result)
    return report


# ------------------------------------------------------- the execution stage


class _Stage(ExecutionStage):
    """What a task's turn uses of the workflow; the turn itself is scripted."""

    def __init__(self, workspace: Path, integration=None):
        super().__init__(
            SimpleNamespace(
                workspace=workspace,
                _tls=threading.local(),
                journal=_journal(workspace),
                _git=git_output,
            )
        )
        self._integration = integration


def test_a_finished_task_keeps_a_receipt_with_the_commits_it_integrated(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("A = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    ws = tmp_path / "ws"
    ws.mkdir()
    integration = Integration(
        repo, ws, "mv2/run", {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}
    )

    class Worker(_Stage):
        def _work_task_body(self, task):
            tree = integration.tree(task.id)
            for n in (1, 2):
                (tree / "a.py").write_text(f"A = {n + 1}\n")
                _git(tree, "commit", "-q", "-am", f"mtp: step {n}")
            decision = ws / "decisions" / "mtp-assembler-1.yaml"
            raw = {
                "decision": "UNIT_READY",
                "summary": "mtp at tp1.\n\nFor the planner: depth 3 only.",
            }
            report = _turn(
                self.workflow.journal,
                "assembler",
                task.id,
                decision,
                raw,
                "UNIT_READY; mtp at tp1.",
            )
            return r.TaskResult.answered("assembler", decision, report)

    stage = Worker(ws, integration)
    task = g.Task(id="mtp", kind="feature", goal="mtp", ref="mtp")
    assert stage._work_task(task) == "UNIT_READY"
    receipt = r.read(ws / "tasks", "mtp")
    assert receipt["decision"] == "UNIT_READY" and receipt["summary"] == "mtp at tp1."
    assert [c.split(" ", 1)[1] for c in receipt["commits"]] == ["mtp: step 2", "mtp: step 1"]
    assert receipt["commits"][0].split()[0] == _git(repo, "rev-parse", "--short", "HEAD")
    assert receipt["decision_file"] == str(ws / "decisions" / "mtp-assembler-1.yaml")
    assert Path(receipt["result_file"]).name.endswith("-assembler-mtp.md")

    # A process that never integrated it knows only the branch tip.
    later = _Stage(ws, integration)
    [tip] = later._landed_commits("mtp")
    assert tip.endswith("mtp: step 2 (the branch tip)")


def test_a_task_that_did_not_finish_leaves_no_receipt(tmp_path):
    stage = _Stage(tmp_path)
    stage._keep_receipt(
        g.Task(id="core", kind="assemble", goal="g"), r.TaskResult("BLOCKED", "no GPU")
    )
    assert not (tmp_path / "tasks").exists()


# ----------------------------------------------------------- the whole run


_BIND = "modeling_v2 x: bound MambaHybridCacheManagerV2 #1: 897 recurrent slots, 32 per slot\n"


def _binding_shell():
    """Every command passes, and logs the recurrent pool its engine bound."""
    inner = passing_shell()

    def run(command, env=None, timeout=0):
        result = inner(command, env, timeout)
        return CommandResult(returncode=result.returncode, stdout=result.stdout + _BIND)

    return run


def test_a_dependent_starts_with_the_receipt_and_the_measured_facts(
    tmp_path, task_file, monkeypatch
):
    planner_prompts: list[str] = []
    dependent_prompts: list[str] = []

    def planner(turn, prompt):
        planner_prompts.append(prompt)
        if turn > 1:
            return {"decision": "DONE", "summary": "the contract should hold"}
        tasks = unit_tasks(planner_graph_path(prompt).parent)
        tasks.append(
            {"id": "why-core", "kind": "investigate", "goal": "why core binds", "ref": "core",
             "depends_on": ["core"]}
        )  # fmt: skip
        write_graph(prompt, tasks)
        return {"decision": "GRAPH_UPDATED", "summary": "core, then an investigation of it"}

    def assembler(turn, prompt):
        if "## Your task: `why-core`" in prompt:
            dependent_prompts.append(prompt)
            return {"decision": "UNIT_READY", "summary": "found it"}
        summary = "core at tp1.\n\nFor the planner: the conv pool binds 897 slots."
        return {"decision": "UNIT_READY", "summary": summary}

    workflow = _workflow(tmp_path, shell_runner=_binding_shell())
    script = Script(None, planner=planner, assembler=assembler)
    _install(monkeypatch, workflow, script)
    with workflow:
        workflow.run(task_file)

    ws = tmp_path / "ws"
    receipt = r.read(ws / "tasks", "core")
    assert receipt["decision"] == "UNIT_READY" and receipt["summary"] == "core at tp1."
    [prompt] = dependent_prompts
    assert r.HEADING in prompt and "### `core` (assemble): UNIT_READY" in prompt
    assert rf.HEADING in prompt and "- core@tp1:" in prompt and "897 recurrent slots" in prompt
    facts = yaml.safe_load((ws / rf.FACTS_FILENAME).read_text())
    member = facts["members"]["core@tp1"]
    assert member["facts"]["recurrent_slots_per_rank"] == 897
    assert {e["gate"] for e in member["engines"]} == {"boot", "accuracy[gsm8k]"}
    # The first graph is planned before any gate has measured an engine.
    assert rf.HEADING not in planner_prompts[0]
    envelope = json.loads((ws / "tasks" / "why-core.json").read_text())
    assert envelope["receipts"][0]["task"] == "core"
