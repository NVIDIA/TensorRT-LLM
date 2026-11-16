# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 workflow behavior and regression checks."""

from __future__ import annotations

import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.operators import prior_art
from agent_flow.workflows.trtllm_modeling_v2.run.records.state import STAGE_EXECUTION
from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from ._helpers import Script, _install, _state, _workflow, _write_vocabulary


def test_preflight_failure_ends_the_run_before_any_agent(tmp_path, task_file, monkeypatch):
    def broken(argv, env=None, timeout=0):
        return CommandResult(returncode=0, stdout="0, 20000, 183359, 99, 10.3\n")

    workflow = _workflow(tmp_path, command_runner=broken)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    assert calls == []
    assert "preflight failed" in (tmp_path / "ws" / "journal.md").read_text()


def test_a_survey_is_not_sent_back_for_what_it_leaves_unsaid(tmp_path, task_file, monkeypatch):
    """A survey is judged by what its rows cause: no review holds its reasoning to a form."""
    monkeypatch.setattr(TrtllmModelingV2Workflow, "survey_first", True)

    def take(workflow, config):
        workflow._prior_art = prior_art.PriorArt(
            architectures=("NewForCausalLM",), registered=False
        )
        workflow._engine_limits = []
        workflow._prior_art_block = prior_art.survey_block(workflow._prior_art, [])

    monkeypatch.setattr(bootstrap, "_take_prior_art", take)
    workspace = tmp_path / "ws"
    prompts = []

    def surveyor(turn, prompt):
        prompts.append(prompt)
        # A row that writes new code beside a lead it does not name, with no call site.
        _write_vocabulary(workspace, [{"n": 1, "mode": "onboard", "target": "ops/norm/rms"}])
        return {"decision": "ready", "summary": f"draft {turn}"}

    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None, surveyor=surveyor))
    with workflow:
        workflow.run(task_file)
    assert len(prompts) == 1, "one survey turn, and no second draft"
    # What the run found of the engine's own model reached the surveyor in its dispatch.
    assert "The engine registers no model for `NewForCausalLM`" in prompts[0]
    assert not (workspace / "survey_review.md").exists()
    assert "survey review" not in (workspace / "journal.md").read_text()
    assert _state(tmp_path).verdict == "delivered"


def test_a_run_with_everything_left_blocked_stops_escalated_and_resumes_once_resolved(
    tmp_path, task_file, monkeypatch, capsys
):
    from agent_flow.workflows.trtllm_modeling_v2.run import escalation, resolve
    from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
    from agent_flow.workflows.trtllm_modeling_v2.run.execution import handoff
    from agent_flow.workflows.trtllm_modeling_v2.run.records import report

    blocked = {
        "decision": "blocked",
        "summary": "the router's weights have no key in the checkpoint",
        "tried": ["the names the HF module uses", "the names the sibling target uses"],
        "command": "python -m pytest tests/x.py::test_router",
        "result": "1 failed",
        "ruled_out": ["a typo in the mapping"],
        "next_question": "is the router tied to the embedding in this export?",
    }
    workspace = tmp_path / "ws"
    first = _workflow(tmp_path)
    calls = _install(monkeypatch, first, Script(None, assembler=lambda turn, prompt: blocked))
    with first:
        first.run(task_file)
    # The run stays open, exits with its own status, and says where it stands.
    assert first._escalated and not _state(tmp_path).done and len(calls) == 1
    saved = g.Graph.load(workspace / g.GRAPH_FILENAME)
    (waits,) = escalation.waiting(saved)
    # What waits behind it: its session, and the task the session's pass starts (the
    # files the delivery requires for CI, which code makes a task for).
    assert waits.startswith("assemble-core (assemble), attempt 1, worker/")
    assert waits.endswith("it holds session-core, delivery-files")
    assert f"it asks: {blocked['next_question']}" in waits
    # What its worker tried and ruled out is in its receipt, written once.
    receipt = handoff.read(workspace / "tasks", "assemble-core", 1)
    assert receipt["command"] == blocked["command"] and receipt["result"] == "1 failed"
    assert receipt["tried"] == blocked["tried"] and receipt["ruled_out"] == blocked["ruled_out"]
    text = (workspace / report.REPORT_FILENAME).read_text()
    assert ": escalated" in text.splitlines()[0] and "**Not onboarded**" in text
    assert "`assemble-core` (blocked)" in text and "(attempt 1 waits for an answer)" in text
    out = " ".join(capsys.readouterr().out.split())
    assert (
        "the run is escalated: 1 block(s) wait for an answer" in out
        and "resolve --workspace" in out
    )
    # The outer session answers with no controller running; the same command then finishes.
    argv = [
        "--workspace",
        str(workspace),
        "retry",
        "assemble-core",
        "--note",
        "it is tied: skip it",
    ]
    assert resolve.main(argv) == 0
    second = _workflow(tmp_path)
    again = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)
    assert _state(tmp_path).verdict == "delivered" and not second._escalated
    # The assembly again. The delivery's task takes no turn: the tree holds the CI files
    # and no landing recorded a finding.
    assert [role for role, _ in again] == ["assembler"]
    text = (workspace / report.REPORT_FILENAME).read_text()
    assert ": delivered" in text.splitlines()[0] and "**Onboarded** at `c0ffee1`" in text
    assert "Nothing: every task of the run is done." in text
    assert "it is tied: skip it" in (workspace / escalation.NOTES_FILENAME).read_text()


def test_a_run_whose_session_could_not_verify_a_check_closes_partial_and_says_which(
    tmp_path, task_file, monkeypatch, capsys
):
    import json

    from agent_flow.workflows.trtllm_modeling_v2.checks import session as session_mod
    from agent_flow.workflows.trtllm_modeling_v2.checks import verdict as check_result
    from agent_flow.workflows.trtllm_modeling_v2.run import escalation
    from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
    from agent_flow.workflows.trtllm_modeling_v2.run.records import report

    why = "hop: 2 of 2 response(s) name no transfer interval"
    loads = []

    def not_observed(deployment, environment, **kwargs):
        # Every check of the session answered; one could not observe what it is for.
        loads.append(deployment.name)
        member = deployment.shape
        boot = check_result.GateResult(name="boot", member=member, passed=True, detail="answered")
        own = check_result.GateResult(
            name="deployment", member=member, passed=True, unverified=why,
            detail="2 check(s): nothing failed, 1 not verified",
        )  # fmt: skip
        return session_mod.SessionResult((boot,), own, loads=1)

    monkeypatch.setattr(session_mod, "run_deployment", not_observed)
    workspace = tmp_path / "ws"
    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    # Closed, with its own verdict: not escalated, nothing repaired, one load.
    state = _state(tmp_path)
    assert state.done and state.verdict == "partial" and not workflow._escalated
    assert escalation.waiting(g.Graph.load(workspace / g.GRAPH_FILENAME)) == []
    assert loads == ["core"]
    # One turn, the assembly's. The delivery's task takes none: the tree already holds
    # the CI files, and no landing recorded a finding.
    assert [role for role, _ in calls] == ["assembler"]
    said = f"core:deployment@tp1: {why}"
    graph = yaml.safe_load((workspace / "graph.yaml").read_text())
    session = next(t for t in graph["tasks"] if t["id"] == "session-core")
    assert session["status"] == "done" and session["unverified"] == [said]
    # Onboarded is pinned, and its one record says what the session could not verify.
    record = json.loads((workspace / "onboarded.json").read_text())
    assert record["session"] == "session-core" and record["unverified"] == [said]
    text = (workspace / report.REPORT_FILENAME).read_text()
    assert ": partial" in text.splitlines()[0]
    assert f"failed nothing on that commit, and could not verify: {said}." in text
    assert f"- `session-core` is done with `core:deployment@tp1` not verified: {why}" in text
    assert "Nothing: every task of the run is done." in text
    journal = (workspace / "journal.md").read_text()
    assert f"; not verified: {said}" in journal and "[partial] every task is done" in journal
    out = " ".join(capsys.readouterr().out.split())
    assert "run closed — verdict partial" in out and "the run is escalated" not in out
    # The same command again finds a closed run and measures nothing more.
    again = _workflow(tmp_path)
    _install(monkeypatch, again, Script(None))
    with again:
        again.run(task_file)
    assert loads == ["core"] and _state(tmp_path).verdict == "partial"


def test_a_workspace_of_an_older_flow_is_refused_in_one_line_before_anything_is_dispatched(
    tmp_path, task_file, monkeypatch, capsys
):
    """A graph with a task of a kind this flow no longer has (`tune`) died with a KeyError."""
    from agent_flow.workflows.trtllm_modeling_v2.run import graph as graph_mod
    from agent_flow.workflows.trtllm_modeling_v2.run.records.state import WorkflowState, save_state

    workspace = tmp_path / "ws"
    workspace.mkdir()
    state_path = workspace / ".trtllm_modeling_v2_state.json"
    save_state(state_path, WorkflowState(task_path=str(task_file), stage=STAGE_EXECUTION))
    tasks = [
        {"id": "t1", "kind": "assemble", "goal": "assemble core", "status": "done"},
        {"id": "t7", "kind": "tune", "goal": "tune the core", "status": "pending"},
    ]
    (workspace / graph_mod.GRAPH_FILENAME).write_text(yaml.safe_dump({"tasks": tasks}))
    assert graph_mod.removed_kinds(workspace / graph_mod.GRAPH_FILENAME) == ["t7 (tune)"]
    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    said = " ".join(capsys.readouterr().out.split())
    assert "was written by an older flow (its graph holds t7 (tune))" in said
    assert "start a new workspace, or pass --clean" in said
    assert calls == [], "nothing was dispatched"
    # The same line for a state file left at a stage this flow no longer has.
    (workspace / graph_mod.GRAPH_FILENAME).unlink()
    state_path.write_text(state_path.read_text().replace(STAGE_EXECUTION, "tune"))
    with _workflow(tmp_path) as again:
        again.run(task_file)
    assert "its state is at stage 'tune'" in " ".join(capsys.readouterr().out.split())
