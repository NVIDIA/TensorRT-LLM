# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""When the survey lands, its vocabulary's queue rows become kernel tasks with no planner turn.

The first graph cannot name kernel tasks before the survey writes its rows;
adding them would take another planner turn and another critic read.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.onboard import delivery
from agent_flow.workflows.trtllm_modeling_v2.onboard import vocabulary as vocab
from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)

_WAIT = 10.0
_TARGET = f"{delivery.MODELING_V2_PACKAGE}/models/fam/ckpt__sm_103__tp1"
_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("mm", "boot", "tp1", c.TIER_INNER, required=False),
    )
)
_ROWS = [
    {"n": 1, "mode": "verify", "target": "norm/rmsnorm", "surface": "hidden 768"},
    {"n": 2, "mode": "onboard", "target": "ops/moe/permute", "tier": 4, "depends_on": [1]},
    {"n": 3, "mode": "verify", "target": "attention/thop", "surface": "mqa", "unit": "mm"},
    {"n": 4, "mode": "verify", "target": "gemm/cublas_mm", "surface": "fp8 kv"},
]


def _write_vocabulary(workspace, rows=_ROWS):
    doc = {"verdict": vocab.VERDICT_AFTER_ADDITIONS, "steps": [], "queue": rows}
    (workspace / vocab.VOCABULARY_FILENAME).write_text(yaml.safe_dump(doc), encoding="utf-8")


def _task(tid, kind="feature", satisfies=(), deps=(), ref=""):
    return g.Task(
        id=tid,
        kind=kind,
        goal=f"do {tid}",
        depends_on=list(deps),
        touches=[f"{_TARGET}/{tid}.py"],
        gpus=1,
        satisfies=list(satisfies),
        ref=ref,
    )


def _result(check):
    return SimpleNamespace(name=check.gate, member=check.member, passed=True, detail="")


def _orchestrator(tmp_path, plan, work, **kw):
    services = Services(plan=plan, work=work, check=_result, free_gpus=lambda: 32)
    status = c.ContractStatus.load(tmp_path)
    kw.setdefault("poll_s", 0.01)
    return GraphOrchestrator(services, _CONTRACT, status, tmp_path, **kw)


def test_a_survey_that_lands_starts_one_kernel_task_per_row_with_no_planner_turn(tmp_path):
    turns, started, seen, orchestrator = [], [], {}, None
    kernels_done = threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            survey = _task("survey", "survey", ref=_TARGET)
            graph = g.Graph(tasks=[survey, _task("core", "assemble", ["core:boot@tp1"])])
            return PlannerAnswer("GRAPH_UPDATED", "survey and core", graph)
        return PlannerAnswer("DONE", "all green")

    def work(task):
        started.append(task.id)
        if task.id == "survey":
            _write_vocabulary(tmp_path)
            return "DRAFT_READY"
        if task.kind == "kernel" and sum(t.startswith("k") for t in started) == 4:
            seen["turns when every row had started"] = len(turns)
            kernels_done.set()
        if task.id == "core":
            kernels_done.wait(_WAIT)
        return "LANDED" if task.kind == "kernel" else "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, plan, work, event_window_s=60)
    assert orchestrator.run() == "delivered"
    assert seen["turns when every row had started"] == 1
    kernels = {t.ref: t for t in orchestrator.graph.tasks if t.kind == "kernel"}
    assert sorted(kernels) == ["1", "2", "3", "4"]
    one, two, three, four = (kernels[n] for n in ("1", "2", "3", "4"))
    # A row on a shared entry lands its local variant in the target's own ops/.
    assert one.id == "k1-rmsnorm" and one.touches == [
        f"{_TARGET}/ops/norm/rmsnorm__r1.py",
        f"{delivery.model_ops_tests_dir(_TARGET)}/norm/test_modeling_v2_rmsnorm__r1.py",
    ]
    # The target's own entry lives under the directory the survey names.
    assert two.touches == [
        f"{_TARGET}/ops/moe/permute.py",
        f"{delivery.model_ops_tests_dir(_TARGET)}/moe/test_modeling_v2_permute.py",
    ]
    assert two.depends_on == [one.id] and two.est_minutes > one.est_minutes
    # A verify row's smith reserves nothing: the jobs its tests run take their own devices.
    assert (one.gpus, two.gpus, four.gpus) == (0, 1, 0)
    # A row only a nice_to_have unit needs is nice_to_have; rows on one entry run side by
    # side, each in a variant of its own, and change only the run's own paths.
    assert three.priority == "nice_to_have" and four.priority == "required"
    assert not g.touches_conflict(three, four)
    assert not g.shared_touches(three, g.own_paths(_TARGET))
    assert orchestrator.graph.revisions[-1].reason.startswith("orchestrator: one kernel task")
    # The planner hears it with its next turn; this run held before one was due.
    said = next(e for e in orchestrator.events if "vocabulary queue row(s)" in e)
    assert "added a kernel task for 4 of them" in said and "k2-permute" in said


def test_rows_a_kernel_task_names_are_left_to_it_and_an_unplaceable_row_is_reported(tmp_path):
    turns, orchestrator = [], None

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(
                tasks=[_task("survey", "survey"), _task("k-mine", "kernel", ref="1"),
                       _task("core", "assemble", ["core:boot@tp1"], deps=["survey"])]
            )  # fmt: skip
            return PlannerAnswer("GRAPH_UPDATED", "survey, my row 1, core", graph)
        return PlannerAnswer("DONE", "all green")

    def work(task):
        if task.id == "survey":
            _write_vocabulary(tmp_path)
            return "DRAFT_READY"
        return "LANDED" if task.kind == "kernel" else "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, plan, work)
    assert orchestrator.run() == "delivered"
    by_ref = {}
    for task in orchestrator.graph.tasks:
        if task.kind == "kernel":
            by_ref.setdefault(task.ref, []).append(task.id)
    # Row 1 has its own task; row 2 is the target's own entry, and no target is known.
    assert by_ref == {"1": ["k-mine"], "3": ["k3-thop"], "4": ["k4-cublas-mm"]}
    said = next(e for t in turns for e in t.events if "vocabulary queue row(s)" in e)
    assert "rows 2 name an entry it could not place" in said


def test_tasks_added_during_a_planner_turn_outlive_a_graph_written_before_them(tmp_path):
    turns, orchestrator = [], None
    landed, in_turn_two = threading.Event(), threading.Event()

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            graph = g.Graph(
                tasks=[_task("survey", "survey", ref=_TARGET), _task("r"),
                       _task("core", "assemble", ["core:boot@tp1"])]
            )  # fmt: skip
            return PlannerAnswer("GRAPH_UPDATED", "survey, r and core", graph)
        if len(turns) == 2:
            stale = g.Graph(tasks=[t for t in orchestrator.graph.tasks if t.kind != "kernel"])
            in_turn_two.set()
            landed.wait(_WAIT)
            time.sleep(0.2)  # the loop records the survey and adds the rows meanwhile
            return PlannerAnswer("GRAPH_UPDATED", "written before the rows", stale)
        return PlannerAnswer("DONE", "all green")

    def work(task):
        if task.id == "r":
            return "UNIT_READY\nthe checkpoint has a second attention variant"  # wakes turn two
        if task.id == "survey":
            in_turn_two.wait(_WAIT)
            _write_vocabulary(tmp_path, _ROWS[:1])
            landed.set()
            return "DRAFT_READY"
        if task.id == "core":
            deadline = time.monotonic() + _WAIT
            while len(turns) < 2 or orchestrator._planner is not None:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        return "LANDED" if task.kind == "kernel" else "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, plan, work, event_window_s=60)
    assert orchestrator.run() == "delivered"
    assert "k1-rmsnorm" in orchestrator.graph.by_id()
    assert orchestrator.graph.by_id()["k1-rmsnorm"].status == "done"


def test_core_rows_soft_wire_into_the_pending_core_assembly_and_start_first(tmp_path):
    contract = c.Contract(
        checks=(c.Check("core", "boot", "tp1", c.TIER_INNER), c.Check("mm", "boot", "tp1", c.TIER_INNER))
    )  # fmt: skip
    services = Services(plan=None, work=None, check=_result, free_gpus=lambda: 32)
    orchestrator = GraphOrchestrator(services, contract, c.ContractStatus.load(tmp_path), tmp_path)
    survey = _task("survey", "survey", ref=_TARGET)
    survey.status = "done"
    core = _task("core", "assemble", ["core:boot@tp1"], deps=["survey"])
    mm = _task("mm", "assemble", ["mm:boot@tp1"], deps=["core"])
    orchestrator.graph = g.Graph(tasks=[survey, core, mm])
    _write_vocabulary(tmp_path)
    orchestrator._add_kernel_tasks(survey)
    kernels = {t.ref: t.id for t in orchestrator.graph.tasks if t.kind == "kernel"}
    # Rows 1, 2 and 4 name no unit, so they are core's; row 3 is the later unit's.
    assert core.depends_on_soft == [kernels["1"], kernels["2"], kernels["4"]]
    assert mm.depends_on_soft == []
    assert "core soft-depends on the core rows" in orchestrator.graph.revisions[-1].reason
    started = [t.id for t in g.next_to_start(orchestrator.graph, free_gpus=8)]
    assert started == [kernels["1"], kernels["4"], kernels["3"]]


def test_dependents_of_a_failed_receipt_that_landed_start_without_a_revision(tmp_path):
    services = Services(
        plan=None, work=None, check=_result, free_gpus=lambda: 32, landed=lambda task: True
    )
    orchestrator = GraphOrchestrator(services, _CONTRACT, c.ContractStatus.load(tmp_path), tmp_path)
    row = _task("k1-row", "kernel", ref="1")
    row.status = "running"
    core = _task("core", "assemble", ["core:boot@tp1"], deps=["k1-row"])
    orchestrator.graph = g.Graph(tasks=[row, core])
    orchestrator._finish(row, "FAILED_RECEIPT\na shared kernel is wrong on ties", [])
    assert row.status == "failed"
    assert [t.id for t in g.ready(orchestrator.graph, orchestrator._receipts_landed)] == ["core"]
    # The planner is still told, and hears which tasks went on.
    said = next(e for e in orchestrator.events if "ended FAILED_RECEIPT" in e)
    assert "its receipt's commits are on the target, so core" in said
    # The landing survives a restart.
    again = GraphOrchestrator(services, _CONTRACT, c.ContractStatus.load(tmp_path), tmp_path)
    assert again._receipts_landed == {"k1-row": 1}


def test_entry_paths_place_a_row_in_the_catalog_or_the_targets_own_ops():
    row = vocab.QueueItem(n=1, mode="verify", target="norm/rmsnorm")
    own = vocab.QueueItem(n=2, mode="onboard", target="ops/moe/permute")
    # With no target there is no local home: a shared entry lands in place.
    assert vocab.entry_paths(row) == (
        f"{delivery.CATALOG_DIR}/norm/rmsnorm.py",
        f"{delivery.CATALOG_TESTS_DIR}/norm/test_modeling_v2_rmsnorm.py",
    )
    assert vocab.entry_paths(row, _TARGET) == (
        f"{_TARGET}/ops/norm/rmsnorm__r1.py",
        f"{delivery.model_ops_tests_dir(_TARGET)}/norm/test_modeling_v2_rmsnorm__r1.py",
    )
    assert vocab.entry_paths(own, _TARGET + "/") == (
        f"{_TARGET}/ops/moe/permute.py",
        f"{delivery.model_ops_tests_dir(_TARGET)}/moe/test_modeling_v2_permute.py",
    )
    assert vocab.entry_paths(own) is None  # the target's own entry, with no target
    for target in ("rmsnorm", "a/b/c", "ops/moe", "norm/"):
        assert (
            vocab.entry_paths(vocab.QueueItem(n=3, mode="verify", target=target), _TARGET) is None
        )


def test_a_kernel_task_id_names_its_row_and_entry_and_never_reuses_one():
    item = vocab.QueueItem(n=7, mode="verify", target="ssm/chunk_gated_delta_rule")
    assert orchestrator_mod._kernel_task_id(item, ()) == "k7-chunk-gated-delta-rule"
    taken = {"k7-chunk-gated-delta-rule", "k7-chunk-gated-delta-rule-2"}
    assert orchestrator_mod._kernel_task_id(item, taken) == "k7-chunk-gated-delta-rule-3"
