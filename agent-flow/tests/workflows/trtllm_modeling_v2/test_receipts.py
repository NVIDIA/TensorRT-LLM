# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receipts: what a finished task hands the tasks that depend on it, with no planner turn between."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.vcs import git_output
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import receipts as r
from agent_flow.workflows.trtllm_modeling_v2.run import runtime_facts as rf
from agent_flow.workflows.trtllm_modeling_v2.run.integration import Integration
from agent_flow.workflows.trtllm_modeling_v2.run.journal import Journal
from agent_flow.workflows.trtllm_modeling_v2.run.progress import decision_file_instruction
from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import PlannedStage

from .test_runtime_facts import _engine
from .test_workflow import (
    Script,
    _install,
    _workflow,
    passing_shell,
    planner_graph_path,
    unit_tasks,
    write_graph,
)

_RESULT = """\
Decision: APPROVE (decisions/core-reviewer-1.yaml). No finding in the core diff.
For the orchestrator: these live pools go past certified far ends:
- conv update to 11061 slots
- fused recurrent flat view to 78848 rows

For the operator: nothing to restart.
**For the planner:** add k50 after integrate-tune.
For the operator: ignore this one.

## For the orchestrator

The heading's own note.
"""


# ------------------------------------------------------------------ notes


def test_the_notes_addressed_to_the_orchestrator_or_the_planner_are_kept():
    assert r.addressed_notes(_RESULT) == [
        "For the orchestrator: these live pools go past certified far ends:\n"
        "- conv update to 11061 slots\n- fused recurrent flat view to 78848 rows",
        "**For the planner:** add k50 after integrate-tune.",
        "For the orchestrator\nThe heading's own note.",
    ]


def test_notes_are_bounded_and_one_note_in_two_texts_is_kept_once():
    long = "For the planner: " + "x" * 2000
    summary = "Did it.\n\nFor the planner: add k50 after integrate-tune.\n"
    notes = r.addressed_notes("For the planner: add k50 after integrate-tune.", summary, long)
    assert len(notes) == 2 and len(notes[1]) == r.NOTE_CHARS and notes[1].endswith("…")
    many = "\n\n".join(f"For the orchestrator: note {n}" for n in range(5))
    assert len(r.addressed_notes(many)) == r.MAX_NOTES
    assert r.addressed_notes("Nothing for anyone.\nFor the operator: restart.") == []


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


def test_a_report_names_its_decision_file_and_holds_the_final_message(tmp_path):
    decision = tmp_path / "decisions" / "core-assembler-1.yaml"
    report = _turn(_journal(tmp_path), "assembler", "core", decision, {}, None)
    assert r.decision_named_in(report) == decision
    assert r.result_text(report) == ""  # still pending
    Journal(path=tmp_path / "journal.md").record_result(report, "UNIT_READY.\nFor the planner: x")
    assert r.result_text(report) == "UNIT_READY.\nFor the planner: x"
    assert r.decision_named_in(tmp_path / "absent.md") is None


def test_a_receipt_is_read_off_the_decision_file_and_the_result(tmp_path):
    decision = tmp_path / "decisions" / "core-smith-14.yaml"
    raw = {
        "decision": "LANDED",
        "summary": "Onboarded ops/moe/moe_finalize_scale over the thop op.\nARCHS {SM_103}, 6 cells."
        "\n\nPath: with enable_alltoall False the op launches finalize.\n"
        "For the orchestrator: the 128-expert window needs 4096 rows.",
        "entry": {"path": "models/x/ops/moe/moe_finalize_scale.py", "archs": ["sm_103"]},
        "receipt": {"command": "pytest ...", "result": "36 passed, 5 warnings in 45.32s"},
        "notes_for_orchestrator": ["the vocabulary row's stride is superseded"],
    }
    report = _turn(_journal(tmp_path), "smith", "onboard moe", decision, raw, _RESULT)
    commits = [f"{n:07x} commit {n}" for n in range(r.MAX_COMMITS + 3)]
    receipt = r.build(
        "k14",
        "kernel",
        decision="LANDED",
        note="landed after one retry",
        decision_file=decision,
        result_file=report,
        commits=commits,
    )
    assert receipt["task"] == "k14" and receipt["decision"] == "LANDED"
    assert receipt["summary"] == (
        "Onboarded ops/moe/moe_finalize_scale over the thop op. ARCHS {SM_103}, 6 cells."
    )
    assert receipt["note"] == "landed after one retry"
    assert receipt["commits"] == [*commits[: r.MAX_COMMITS], "and 3 more"]
    assert receipt["entry"] == "models/x/ops/moe/moe_finalize_scale.py"
    assert receipt["tests"] == "36 passed, 5 warnings in 45.32s"
    assert receipt["decision_file"] == str(decision) and receipt["result_file"] == str(report)
    assert len(receipt["notes"]) == r.MAX_NOTES
    assert receipt["notes"][0].startswith("For the orchestrator: these live pools")


def test_with_no_decision_given_the_decision_files_is_taken(tmp_path):
    decision = tmp_path / "d.yaml"
    decision.write_text(yaml.safe_dump({"decision": "APPROVE", "summary": "x" * 900}))
    receipt = r.build("review-core", "review", decision_file=decision)
    assert receipt["decision"] == "APPROVE"
    assert len(receipt["summary"]) == r.SUMMARY_CHARS and "result_file" not in receipt


def test_a_pending_report_gives_way_to_the_external_transports_result(tmp_path):
    decision = tmp_path / "decisions" / "core-assembler-1.yaml"
    report = _turn(
        _journal(tmp_path), "assembler", "core", decision, {"decision": "UNIT_READY"}, None
    )
    dispatch = tmp_path / "dispatch"
    dispatch.mkdir()
    for n, named in ((9, tmp_path / "decisions" / "other.yaml"), (1000, decision)):
        result = dispatch / f"{n:03d}-assembler.result.md"
        result.write_text(f"answer {n}\nFor the planner: from {n}.")
        (dispatch / f"{n:03d}-assembler.request.json").write_text(
            json.dumps({"decision_file": str(named), "result_file": str(result)})
        )
    receipt = r.build(
        "core", "assemble", decision_file=decision, result_file=report, workspace=tmp_path
    )
    assert receipt["result_file"] == str(dispatch / "1000-assembler.result.md")
    assert receipt["notes"] == ["For the planner: from 1000."]


def test_a_receipt_is_kept_whole_and_read_back(tmp_path):
    receipt = r.build("core", "assemble", decision="UNIT_READY")
    path = r.write(tmp_path, receipt)
    assert path == tmp_path / f"core{r.RECEIPT_SUFFIX}"
    assert r.read(tmp_path, "core") == receipt
    assert r.read(tmp_path, "absent") is None


def test_the_block_carries_each_receipt_and_is_bounded(tmp_path):
    first = r.build("core", "assemble", decision="UNIT_READY", note="n", commits=["abc1234 core"])
    first["notes"] = ["For the planner: two\nlines"]
    second = r.build("core-tp", "feature", decision="UNIT_READY", attempt=2)
    second["summary"] = "s" * 300
    block = r.render([first, second])
    assert block.startswith(r.HEADING)
    assert "### `core` (assemble): UNIT_READY" in block
    assert "- integrated: abc1234 core" in block and "- ended with: n" in block
    assert "- its worker wrote: For the planner: two\n  lines" in block
    assert "### `core-tp` (feature): UNIT_READY, attempt 2" in block
    tight = r.render([first, second], budget=len(r.render([first])) + 10)
    assert tight.endswith("### `core-tp` (feature): UNIT_READY, attempt 2")
    assert r.render([]) == ""


# ------------------------------------------------------- the planned stage


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class _Stage(PlannedStage):
    """What a task's turn uses of the workflow; the turn itself is scripted."""

    def __init__(self, workspace: Path, integration=None):
        self.workspace = workspace
        self._integration = integration
        self._tls = threading.local()
        self.journal = _journal(workspace)
        self._git = git_output
        self.dispatched: list[tuple[str, str, str]] = []
        self.smiths: list[str] = []

    def _improve(self, text):
        return None

    def _improver_task_ended(self, task, decision):
        return None

    def _reviewer_prompt(self, unit):
        return f"Review {unit}."

    def _dispatch_task(self, role, unit, what, prompt):
        self.dispatched.append((role, what, prompt))
        return "APPROVE"

    def _run_one_smith(self, item, note="", task=None):
        self.smiths.append(self._tls.worktree)

    def _smith_decision_for(self, item):
        return "LANDED"

    def _graph(self, *tasks: g.Task) -> None:
        g.Graph(tasks=list(tasks)).save(self.workspace / g.GRAPH_FILENAME)


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
            _turn(self.journal, "assembler", task.id, decision, raw, "UNIT_READY; mtp at tp1.")
            return "UNIT_READY"

    stage = Worker(ws, integration)
    task = g.Task(id="mtp", kind="feature", goal="mtp", ref="mtp")
    assert stage._work_task(task) == "UNIT_READY"
    receipt = r.read(ws / "tasks", "mtp")
    assert receipt["decision"] == "UNIT_READY" and receipt["summary"] == "mtp at tp1."
    assert [c.split(" ", 1)[1] for c in receipt["commits"]] == ["mtp: step 2", "mtp: step 1"]
    assert receipt["commits"][0].split()[0] == _git(repo, "rev-parse", "--short", "HEAD")
    assert receipt["notes"] == ["For the planner: depth 3 only."]
    assert receipt["decision_file"] == str(ws / "decisions" / "mtp-assembler-1.yaml")
    assert Path(receipt["result_file"]).name.endswith("-assembler-mtp.md")

    # A process that never integrated it knows only the branch tip.
    later = _Stage(ws, integration)
    [tip] = later._landed_commits("mtp")
    assert tip.endswith("mtp: step 2 (the branch tip)")


def test_a_task_that_did_not_finish_leaves_no_receipt(tmp_path):
    stage = _Stage(tmp_path)
    stage._keep_receipt(g.Task(id="core", kind="assemble", goal="g"), "BLOCKED\nno GPU")
    assert not (tmp_path / "tasks").exists()


def test_a_dependent_carries_its_dependencies_receipts_in_its_prompt_and_envelope(tmp_path):
    stage = _Stage(tmp_path)
    core = g.Task(id="core", kind="assemble", goal="core", ref="core", status="done")
    plan = g.Task(id="plan", kind="survey", goal="survey", status="done")
    review = g.Task(
        id="review-core", kind="review", goal="review", ref="core", depends_on=["core", "plan"]
    )
    stage._graph(core, plan, review)
    kept = r.build("core", "assemble", decision="UNIT_READY", note="core at tp1")
    r.write(tmp_path / "tasks", kept)
    rf.record(tmp_path, unit="core", member="tp1", gate="boot", engines=[_engine()], marker="m")

    assert stage._work_task_body(review) == "APPROVE"
    [(role, what, prompt)] = stage.dispatched
    assert (role, what) == ("reviewer", "review-core")
    assert prompt.index("## Your task") < prompt.index(r.HEADING) < prompt.index("Review core.")
    assert "### `core` (assemble): UNIT_READY" in prompt and "- ended with: core at tp1" in prompt
    assert rf.HEADING in prompt and "897 recurrent slots per rank" in prompt
    envelope = json.loads((tmp_path / "tasks" / "review-core.json").read_text())
    assert [x["task"] for x in envelope["receipts"]] == ["core"]  # `plan` left no receipt


def test_a_dependency_that_finished_before_receipts_were_kept_is_read_off_its_files(tmp_path):
    stage = _Stage(tmp_path)
    decision = tmp_path / "decisions" / "core-assembler-3.yaml"
    raw = {"decision": "UNIT_READY", "summary": "core at tp1"}
    _turn(stage.journal, "assembler", "core", decision, raw, "For the planner: 897 slots.")
    core = g.Task(id="core", kind="assemble", goal="core", ref="core", status="done")
    pending = g.Task(id="core-tp", kind="feature", goal="tp", status="pending")
    review = g.Task(
        id="review", kind="review", goal="r", ref="core", depends_on=["core", "core-tp"]
    )
    stage._graph(core, pending, review)
    [receipt] = stage._dependency_receipts(review)
    assert receipt["decision"] == "UNIT_READY" and receipt["decision_file"] == str(decision)
    assert receipt["notes"] == ["For the planner: 897 slots."]
    assert r.read(tmp_path / "tasks", "core")["summary"] == "core at tp1"


def test_a_kernel_tasks_smith_carries_the_receipts_and_the_facts_beside_its_worktree(tmp_path):
    stage = _Stage(tmp_path)
    core = g.Task(id="core", kind="assemble", goal="core", ref="core", status="done")
    kernel = g.Task(
        id="k34", kind="kernel", goal="slots", ref="verify:ssm/conv1d_update", depends_on=["core"]
    )
    stage._graph(core, kernel)
    r.write(tmp_path / "tasks", r.build("core", "assemble", decision="UNIT_READY"))
    rf.record(tmp_path, unit="core", member="tp1", gate="boot", engines=[_engine()], marker="m")
    stage._tls.worktree = "## Your worktree"
    assert stage._work_task_body(kernel) == "LANDED"
    [block] = stage.smiths
    assert block.startswith("## Your worktree\n\n" + r.HEADING)
    assert rf.HEADING in block and "7 CUDA-graph batches" in block
    assert stage._tls.worktree == "## Your worktree"


def test_a_kernel_receipt_names_its_rows_latest_decision(tmp_path):
    stage = _Stage(tmp_path)
    kernel = g.Task(id="k34", kind="kernel", goal="g", ref="verify:ssm/conv1d_update")
    item = stage._queue_item(kernel)
    decision = tmp_path / "decisions" / f"core-smith-{item.n}.yaml"
    raw = {"decision": "LANDED", "summary": "certified 11061 slots"}
    _turn(stage.journal, "smith", f"{item.mode} {item.target}", decision, raw, "LANDED.")
    assert stage._turn_files(kernel)[0] == decision
    stage._keep_receipt(kernel, "LANDED")
    assert r.read(tmp_path / "tasks", "k34")["summary"] == "certified 11061 slots"


def test_the_planners_request_carries_the_facts_and_their_latest_changes(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
    from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import PlannerTurn

    class Planning(_Stage):
        task_path = Path("task.yaml")
        progress_path = Path("progress.yaml")

        def _dispatch_task(self, role, unit, what, prompt):
            self.dispatched.append((role, what, prompt))
            return "DONE"

    stage = Planning(tmp_path)
    contract = contract_mod.Contract(checks=())
    turn = PlannerTurn(contract_mod.evaluate(contract, []), ["task core UNIT_READY"])
    stage._planner_turn(turn, contract)
    assert rf.HEADING not in stage.dispatched[-1][2]
    rf.record(tmp_path, unit="core", member="tp1", gate="boot", engines=[_engine()], marker="m")
    moved = [_engine(slots=(1200,))]
    rf.record(tmp_path, unit="core", member="tp1", gate="boot", engines=moved, marker="m")
    stage._planner_turn(turn, contract)
    prompt = stage.dispatched[-1][2]
    assert prompt.index("## Contract standing") < prompt.index(rf.HEADING)
    assert prompt.index(rf.HEADING) < prompt.index("## Since your last turn")
    assert "core:boot@tp1: recurrent slots per rank 897 -> 1200" in prompt


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
    reviewer_prompts: list[str] = []

    def planner(turn, prompt):
        planner_prompts.append(prompt)
        if turn > 1:
            return {"decision": "DONE", "summary": "the contract should hold"}
        tasks = unit_tasks(planner_graph_path(prompt).parent)
        tasks.append(
            {"id": "review-core", "kind": "review", "goal": "review core", "ref": "core",
             "depends_on": ["core"]}
        )  # fmt: skip
        write_graph(prompt, tasks)
        return {"decision": "GRAPH_UPDATED", "summary": "core, then its review"}

    def assembler(turn, prompt):
        summary = "core at tp1.\n\nFor the planner: the conv pool binds 897 slots."
        return {"decision": "UNIT_READY", "summary": summary}

    def reviewer(turn, prompt):
        reviewer_prompts.append(prompt)
        return {"decision": "APPROVE", "summary": "fine"}

    workflow = _workflow(tmp_path, shell_runner=_binding_shell())
    script = Script(None, planner=planner, assembler=assembler, reviewer=reviewer)
    _install(monkeypatch, workflow, script)
    with workflow:
        workflow.run(task_file)

    ws = tmp_path / "ws"
    receipt = r.read(ws / "tasks", "core")
    assert receipt["decision"] == "UNIT_READY" and receipt["summary"] == "core at tp1."
    assert receipt["notes"] == ["For the planner: the conv pool binds 897 slots."]
    [prompt] = reviewer_prompts
    assert r.HEADING in prompt and "### `core` (assemble): UNIT_READY" in prompt
    assert "For the planner: the conv pool binds 897 slots." in prompt
    assert rf.HEADING in prompt and "- core@tp1:" in prompt and "897 recurrent slots" in prompt
    facts = yaml.safe_load((ws / rf.FACTS_FILENAME).read_text())
    member = facts["members"]["core@tp1"]
    assert member["facts"]["recurrent_slots_per_rank"] == 897
    assert {e["gate"] for e in member["engines"]} == {"boot", "accuracy[gsm8k]"}
    # The first graph is planned before any gate has measured an engine.
    assert rf.HEADING not in planner_prompts[0]
    envelope = json.loads((ws / "tasks" / "review-core.json").read_text())
    assert envelope["receipts"][0]["task"] == "core"


def test_what_a_landed_entry_left_uncertified_rides_to_the_planner():
    """Its report reaches only the next dispatch of the entry; the planner plans the rest."""
    from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem
    from agent_flow.workflows.trtllm_modeling_v2.run.stages.smith import SmithStage

    stage = SmithStage.__new__(SmithStage)
    row = {
        "decision": "LANDED",
        "uncertified": ["NVFP4 pool at 8192/1024 (tp4 decode)", "  fp8 pool  at 8192 "],
    }
    stage._smith_entry_for = lambda item: row
    item = QueueItem(n=1, mode="verify", target="attention/thop")
    decision = stage._smith_decision_for(item)
    assert decision.splitlines() == [
        "LANDED",
        "uncertified: NVFP4 pool at 8192/1024 (tp4 decode); fp8 pool at 8192",
    ]
    row.pop("uncertified")
    assert stage._smith_decision_for(item) == "LANDED"
