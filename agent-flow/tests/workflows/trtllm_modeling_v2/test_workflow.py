# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 workflow behavior and regression checks."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.operators import prior_art
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.records.state import STAGE_EXECUTION, save_state
from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from ._helpers import (
    _STATE,
    TARGET_REL,
    Script,
    _install,
    _state,
    _substrate_boot,
    _workflow,
    _write_vocabulary,
    passing_shell,
    plan_every_unit,
    unit_tasks,
    write_graph,
)


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
            *(p.read_text() for p in Path(_common.__file__).parent.rglob("*.j2")),
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


def test_preflight_failure_ends_the_run_before_any_agent(tmp_path, task_file, monkeypatch):
    def broken(argv, env=None, timeout=0):
        return CommandResult(returncode=0, stdout="0, 20000, 183359, 99, 10.3\n")

    workflow = _workflow(tmp_path, command_runner=broken)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    assert calls == []
    assert "preflight failed" in (tmp_path / "ws" / "journal.md").read_text()


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


def test_a_shared_change_is_followed_by_a_regression_check_with_no_turn(
    tmp_path, task_file, monkeypatch
):
    def planner(turn, prompt):
        if turn == 1:
            assert "## Shared code" in prompt and TARGET_REL in prompt
            core = {
                **unit_tasks(tmp_path / "ws")[0],
                "touches": ["tensorrt_llm/_torch/modeling_v2/catalog/norm/rms.py"],
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


def test_a_closed_run_whose_campaign_did_not_end_resumes_only_the_campaign(
    tmp_path, task_file, monkeypatch
):
    import agent_flow.workflows.trtllm_modeling_v2.tune.stage as stage

    first = _workflow(tmp_path)
    _install(monkeypatch, first, Script(None))
    with first:
        first.run(task_file)
    state = _state(tmp_path)
    # The process stopped while the campaign the onboarding closed beside still ran.
    state.tune = {"onboarded": "delivered", "base_commit": "c0ffee1", "attempt": 2}
    save_state(tmp_path / "ws" / _STATE, state)
    resumed: list[dict] = []
    monkeypatch.setattr(
        stage, "resume_campaign", lambda wf, st: resumed.append(dict(st.tune)) or "tuned"
    )
    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)
    assert calls == [] and [r["base_commit"] for r in resumed] == ["c0ffee1"]
    state = _state(tmp_path)
    assert state.done and state.verdict == "delivered" and state.tune["verdict"] == "tuned"
    # Once it ended, a re-run finds the run complete.
    third = _workflow(tmp_path)
    _install(monkeypatch, third, Script(None))
    with third:
        third.run(task_file)
    assert len(resumed) == 1


@pytest.mark.parametrize("outcome", ["TUNE_PENDING", "TUNED"], ids=["pending", "completed"])
def test_a_campaign_pending_before_delivery_resumes_only_on_the_next_run(
    tmp_path, task_file, monkeypatch, outcome
):
    import agent_flow.workflows.trtllm_modeling_v2.tune.stage as stage

    spec = yaml.safe_load(task_file.read_text())
    spec["perf"] = {"enabled": True}
    task_file.write_text(yaml.safe_dump(spec))
    settled = threading.Event()
    finished = orchestrator_mod.GraphOrchestrator._finish
    attempts, resumed = [], []

    def finish(self, task, decision, failed):
        finished(self, task, decision, failed)
        if task.kind == "tune":
            settled.set()

    def tune(wf, state, base_commit, attempt=1):
        attempts.append(attempt)
        state.tune.update(
            phase="done" if outcome == "TUNED" else "campaign",
            base_commit=base_commit,
            attempt=attempt,
        )
        return outcome, "allocation expired" if outcome == "TUNE_PENDING" else "finished"

    def planner(turn, prompt):
        if turn == 1:
            tasks = unit_tasks(tmp_path / "ws")
            tasks.append(
                {
                    "id": "tail",
                    "kind": "feature",
                    "ref": "core",
                    "goal": "finish onboarding",
                    "depends_on": ["core"],
                    "touches": [f"{TARGET_REL}/tail.py"],
                }
            )
            tasks.append(
                {
                    "id": "tune",
                    "kind": "tune",
                    "goal": "tune the core",
                    "depends_on": ["core"],
                    "touches": [],
                }
            )
            write_graph(prompt, tasks)
            return {"decision": "GRAPH_UPDATED", "summary": "core, tail and optional tune"}
        return {"decision": "DONE", "summary": "done"}

    def assemble(turn, prompt):
        if turn == 2:
            assert settled.wait(10), "onboarding must finish after the tune outcome is recorded"
        return {"decision": "UNIT_READY", "summary": "assembled"}

    monkeypatch.setattr(orchestrator_mod.GraphOrchestrator, "_finish", finish)
    monkeypatch.setattr(stage, "run_tune_task", tune)
    monkeypatch.setattr(
        stage, "resume_campaign", lambda wf, st: resumed.append(dict(st.tune)) or "tuned"
    )
    first = _workflow(tmp_path, event_window_s=0)
    _install(monkeypatch, first, Script(None, planner=planner, assembler=assemble))
    with first:
        first.run(task_file)
    state = _state(tmp_path)
    assert state.done and state.verdict == "delivered"
    assert attempts == [1]
    assert not resumed, "a deferred campaign must not be retried during the same run"
    if outcome == "TUNE_PENDING":
        assert state.tune["onboarded"] == "delivered" and not state.tune.get("verdict")
        graph = yaml.safe_load((tmp_path / "ws/graph.yaml").read_text())
        assert next(t for t in graph["tasks"] if t["id"] == "tune")["status"] == "pending"
    else:
        assert "onboarded" not in state.tune

    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)
    assert calls == []
    assert len(resumed) == (1 if outcome == "TUNE_PENDING" else 0)
    if resumed:
        assert resumed[0]["base_commit"] == "c0ffee1" and resumed[0]["attempt"] == 1
        assert _state(tmp_path).tune["verdict"] == "tuned"
    third = _workflow(tmp_path)
    _install(monkeypatch, third, Script(None))
    with third:
        third.run(task_file)
    assert len(resumed) == (1 if outcome == "TUNE_PENDING" else 0)


def test_a_campaign_resumes_on_its_first_pin_as_its_last_attempt(monkeypatch):
    from types import SimpleNamespace

    import agent_flow.workflows.trtllm_modeling_v2.tune.stage as stage

    seen = []

    def stage_for(wf, state, base_commit="", attempt=1):
        seen.append((base_commit, attempt))
        return "tune-failed:spec"

    monkeypatch.setattr(stage, "_stage_for", stage_for)
    state = SimpleNamespace(tune={"base_commit": "c0ffee1", "attempt": 2})
    assert stage.resume_campaign(None, state) == "tune-failed:spec"
    assert seen == [("c0ffee1", 2)]
    assert stage.campaign_verdict("TUNED\ntuned from c0ffee1") == "tuned"
    assert stage.campaign_verdict("TUNE_FAILED (tune-failed:baseline)") == "tune-failed:baseline"
    assert stage.campaign_verdict("TUNE_FAILED (incomplete)") is None
    assert stage.campaign_verdict("raised OSError: the session went away") is None
