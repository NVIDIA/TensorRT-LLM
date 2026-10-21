# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine-defect registry: defects keyed by what triggers them, met once, fixed from the start."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.agents import lessons as L
from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import progress
from agent_flow.workflows.trtllm_modeling_v2.run.stages.improve import ImproveStage
from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import PlannedStage
from agent_flow.workflows.trtllm_modeling_v2.run.stages.smith import SmithStage

from .test_orchestrator import PlannerAnswer, _orchestrator, _services, _task

_KEYS = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]
#: An op whose index math overflows int32 once a call addresses 2^31 elements.
_OVERFLOW = L.Lesson(
    scope="all",
    trigger="index math overflows int32 above 2^31 elements",
    lesson="compute the slot offset in int64",
    evidence="decisions/x.yaml",
    kind="engine_defect",
    affects="ssm/state_update",
    regime={"dtypes": ["bfloat16"], "at_least": {"elements": 2**31}},
    detect={"elements": 2**31 + 8, "dtype": "bfloat16"},
    upstream_fix="#1234",
)


def _row(n=3, target="ssm/state_update", **extra):
    return {"n": n, "mode": "onboard", "target": target, "note": "bfloat16 state", **extra}


def test_a_row_is_in_a_defects_regime_by_its_entry_and_every_condition_stated():
    big = _row(cells=[{"elements": 2**32}])
    assert L.in_regime(_OVERFLOW, big)
    assert not L.in_regime(_OVERFLOW, _row(cells=[{"elements": 4096}]))
    assert not L.in_regime(_OVERFLOW, {**big, "note": "float32 state"})
    assert not L.in_regime(_OVERFLOW, {**big, "target": "attention/x"})
    assert L.in_regime(_OVERFLOW, {**big, "target": "ops/ssm/state_update"})
    anchorless = L.Lesson(**{**_OVERFLOW.__dict__, "affects": "engine.py:f", "regime": {}})
    assert not L.in_regime(anchorless, big)  # no entry to anchor on matches nothing
    hits = L.match_defects([_OVERFLOW], [big, _row(n=4, target="gemm/x")])
    assert [(h.row, h.target) for h in hits] == [(3, "ssm/state_update")]


def test_an_open_defect_is_registered_with_its_disposition(tmp_path):
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "x.yaml").write_text("x")
    item = {
        "scope": "all",
        "kind": "engine_defect",
        "trigger": "t",
        "lesson": "l",
        "evidence": "decisions/x.yaml",
        "affects": "a/b",
        "regime": {"targets": ["a/b"], "at_least": {"rows": 8}},
    }
    kept, _, errors = L.validate_proposal({"lessons": [item]}, tmp_path)
    assert not kept and "while it is open" in errors[0]
    kept, _, errors = L.validate_proposal(
        {"lessons": [{**item, "disposition": "fix branch x", "status": "open"}]}, tmp_path
    )
    assert kept and not errors and kept[0].status == "open"
    _, _, errors = L.validate_proposal(
        {"lessons": [{**item, "disposition": "d", "regime": {"sizes": [1]}}]}, tmp_path
    )
    assert "`regime` key 'sizes'" in errors[0]
    _, _, errors = L.validate_proposal(
        {"lessons": [{**item, "kind": "domain", "affects": "a/b", "disposition": "d"}]}, tmp_path
    )
    assert "only an `engine_defect` lesson names `affects`" in errors[0]


def test_a_smith_report_becomes_an_open_registry_entry():
    report = progress.parse_engine_defect(
        {
            "affects": "tensorrt_llm/x.py:f",
            "trigger": "index math overflows int32 above 2^31 elements",
            "detect": {"elements": 2**31},
            "proposal": "widen the offset to int64",
            "unknown": "dropped",
        }
    )
    assert set(report) == {"affects", "trigger", "detect", "proposal"}
    assert progress.parse_engine_defect({"affects": "x"}) == {}
    lesson = L.defect_from_receipt(report, evidence="decisions/d.yaml", where="task k1")
    assert lesson.kind == "engine_defect" and lesson.status == L.DEFECT_OPEN
    assert lesson.disposition == "proposed in task k1" and lesson.detect == {"elements": 2**31}


class _Stage(ImproveStage, PlannedStage, SmithStage):
    def __init__(self, tmp_path: Path, store: Path, knowledge: str = "production"):
        self.task_data = {"trtllm_repo_path": str(tmp_path / "repo"), "knowledge": knowledge}
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.lessons_dir = store
        self._preflight = SimpleNamespace(family="newmodel")
        self.vocabulary_path = self.workspace / "vocabulary.yaml"
        self.events: list[tuple[str, str]] = []
        self.journal = SimpleNamespace(event=lambda t, b: self.events.append((t, b)))


def _store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    family = L.Lesson(scope="family:newmodel", trigger="a family fact", lesson="x", evidence="e")
    L.LessonStore(root).merge([_OVERFLOW, family], run="earlier", evidence_root=tmp_path)
    return root


def test_the_survey_marks_the_rows_a_known_defect_takes_in(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "fix_in_tree", lambda fix, repo: "absent")
    stage = _Stage(tmp_path, _store(tmp_path))
    rows = [_row(cells=[{"elements": 2**32}]), _row(n=4, target="gemm/x")]
    stage.vocabulary_path.write_text(yaml.safe_dump({"queue": rows}))
    said = stage._mark_known_defects()
    assert "row 3 (`ssm/state_update`) is in the regime of known engine defect" in said
    assert "fixed upstream in `#1234`" in said
    written = yaml.safe_load(stage.vocabulary_path.read_text())["queue"]
    assert written[0]["known_defects"][0]["id"] == _OVERFLOW.id
    assert _OVERFLOW.detect in written[0]["cells"]  # the detection is a required cell
    assert "known_defects" not in written[1]
    # The registry is the knowledge of production runs.
    measuring = _Stage(tmp_path, _store(tmp_path), knowledge="measurement")
    assert measuring._mark_known_defects() == ""
    assert measuring._known_defects_warning() == ""
    block = measuring._lessons_block("smith")
    assert "a family fact" not in block and _OVERFLOW.trigger not in block
    assert "a family fact" in stage._lessons_block("smith")


def test_a_failed_receipt_naming_an_engine_defect_registers_it_and_queues_its_fix(
    tmp_path, monkeypatch
):
    store = tmp_path / "store"
    stage = _Stage(tmp_path, store)
    engine = tmp_path / "repo" / "tensorrt_llm" / "x.py"
    engine.parent.mkdir(parents=True)
    engine.write_text("def f(): pass\n")
    report = {
        "affects": "tensorrt_llm/x.py:f",
        "trigger": "index math overflows int32 above 2^31 elements",
        "detect": "tests/test_x.py::test_big",
        "proposal": "widen the offset to int64",
    }
    entry = {"decision": "FAILED_RECEIPT", "decision_file": "core-smith-3.yaml"}
    monkeypatch.setattr(
        type(stage), "_smith_entry_for", lambda self, item: {**entry, "engine_defect": report}
    )
    item = QueueItem(n=3, mode="onboard", target="ssm/state_update")
    task = g.Task(id="k3-state-update", kind="kernel", goal="", ref="3")
    said = stage._engine_defect_reported(item, task, "FAILED_RECEIPT")
    assert "registered as open engine defect" in said and "the orchestrator adds fix-x-" in said
    [kept] = L.LessonStore(store).engine_defects()
    assert kept.status == "open" and kept.affects == "tensorrt_llm/x.py:f"
    [spec] = stage._take_defect_tasks(g.Graph(tasks=[]))
    assert spec["kind"] == "feature" and spec["touches"] == ["tensorrt_llm/x.py"]
    assert "k3-state-update is retried" in spec["goal"]
    assert stage._take_defect_tasks(g.Graph(tasks=[])) == []
    assert stage._engine_defect_reported(item, task, "LANDED") == ""


def test_the_orchestrator_adds_the_fix_task_with_no_planner_turn(tmp_path):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            core = _task("core", "assemble", _KEYS, deps=["k1"])
            tasks = [_task("k1", "kernel", touches=["x.py"]), core]
            return PlannerAnswer("GRAPH_UPDATED", "plan", g.Graph(tasks=tasks))
        if len(turns) == 2:
            # The kernel task is retried once the fix lands.
            graph = orchestrator.graph
            retried = [t for t in graph.tasks if t.id != "k1"]
            again = _task("k1", "kernel", touches=["x.py"], deps=["fix-x"])
            again.attempts = 2
            return PlannerAnswer("GRAPH_UPDATED", "retry", g.Graph(tasks=[*retried, again]))
        return PlannerAnswer("DONE", "green")

    services = _services(plan)
    queued = [{"id": "fix-x", "kind": "feature", "goal": "fix it", "touches": ["x.py"]}]
    services.defect_tasks = lambda graph: [queued.pop()] if queued else []
    services.work = lambda task: (
        "FAILED_RECEIPT" if task.id == "k1" and task.attempts == 1 else "UNIT_READY"
    )
    orchestrator = _orchestrator(tmp_path, services)
    assert orchestrator.run() == "delivered"
    assert {t.id: t.status for t in orchestrator.graph.tasks}["fix-x"] == "done"
    assert any("fix the engine defects" in r.reason for r in orchestrator.graph.revisions)
    heard = [e for t in turns for e in t.events if "fix-x" in e]
    assert heard and "an engine defect a smith's failed receipt reported" in heard[0]


def test_the_fix_task_carries_the_receipts_whole_proposal_and_its_detecting_test(
    tmp_path, monkeypatch
):
    """The registry keeps the words cut to its limit; a fix task given half a proposal is stuck."""
    stage = _Stage(tmp_path, tmp_path / "store")
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm").mkdir(parents=True)
    (repo / "tensorrt_llm" / "op.py").write_text("def safe_exp(x): pass\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_ties.py").write_text("def test_ties(): pass\n")
    proposal = "clamp the exponent at zero, exp(min(x, 0)), instead of zeroing it; " * 30
    report = {
        "affects": "tensorrt_llm/op.py:safe_exp",
        "trigger": "tied gates zero the decay factor",
        "detect": {"test": "tests/test_ties.py::test_ties", "cells": [{"heads": 3}]},
        "proposal": proposal,
    }
    entry = {"decision": "FAILED_RECEIPT", "decision_file": "core-smith-16.yaml"}
    monkeypatch.setattr(
        type(stage), "_smith_entry_for", lambda self, item: {**entry, "engine_defect": report}
    )
    item = QueueItem(n=16, mode="verify", target="ssm/chunk")
    task = g.Task(id="k16", kind="kernel", goal="", ref="16")
    stage._engine_defect_reported(item, task, "FAILED_RECEIPT")
    [spec] = stage._take_defect_tasks(g.Graph(tasks=[]))
    assert " ".join(proposal.split()) in spec["goal"]
    assert "test: tests/test_ties.py::test_ties" in spec["goal"], "detect as yaml, not a repr"
    assert spec["touches"] == ["tensorrt_llm/op.py", "tests/test_ties.py"]
