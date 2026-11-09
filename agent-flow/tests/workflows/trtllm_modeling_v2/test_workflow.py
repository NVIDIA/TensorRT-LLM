# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 workflow behavior and regression checks."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.operators import prior_art
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod
from agent_flow.workflows.trtllm_modeling_v2.run.records.state import STAGE_EXECUTION
from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from ._helpers import (
    Script,
    _install,
    _state,
    _substrate_boot,
    _workflow,
    _write_vocabulary,
    passing_shell,
    plan_every_unit,
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
            said = {
                "facts": {},
                "problems": ["MPI: the leader sees a world of 1, the launch made 4"],
            }
            return CommandResult(returncode=1, stdout=f"MV2FLOW_SUBSTRATE {json.dumps(said)}")
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
