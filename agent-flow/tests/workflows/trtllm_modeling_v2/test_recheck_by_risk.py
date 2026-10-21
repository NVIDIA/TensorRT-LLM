# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stale checks run again riskiest first: the tightest memory, then what the last change reached."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)

_LOW = (
    "LOW MARGIN: 0.52 GiB free on device 0 once the engine was up, under the 5.76 GiB "
    "threshold. The next change that adds memory to this configuration runs it out"
)


def _engine(*free_gib):
    return tuple(
        {"leg": "", "memory": {"devices": [{"device": "0", "free_gib": f, "total_gib": 288.0}]}}
        for f in free_gib
    )


def _result(name, member, passed=True, engine=(), warnings=()):
    return SimpleNamespace(
        name=name,
        member=member,
        passed=passed,
        detail="d",
        tree_state="t",
        engine=engine,
        warnings=warnings,
    )


# ------------------------------------------------------------ contract status


def test_a_verdict_keeps_the_least_memory_its_engines_left_and_a_low_margin_warning(tmp_path):
    status = c.ContractStatus.load(tmp_path)
    status.record("mtp", _result("boot", "tp1", engine=_engine(12.5, 0.52), warnings=(_LOW,)))
    status.record("core", _result("boot", "tp1", engine=_engine(40.0)))
    reloaded = c.ContractStatus.load(tmp_path).latest()
    assert (reloaded["mtp:boot@tp1"].min_free_gib, reloaded["mtp:boot@tp1"].low_margin) == (
        0.52,
        True,
    )
    assert (reloaded["core:boot@tp1"].min_free_gib, reloaded["core:boot@tp1"].low_margin) == (
        40.0,
        False,
    )


def test_a_verdict_without_an_engine_reading_is_written_as_before(tmp_path):
    status = c.ContractStatus.load(tmp_path)
    status.record("core", SimpleNamespace(name="boot", member="tp1", passed=True, detail="d"))
    line = json.loads((tmp_path / c.STATUS_FILENAME).read_text())
    assert set(line) == {"unit", "gate", "member", "passed", "detail", "tree_state", "marker"}
    assert c.ContractStatus.load(tmp_path).latest()["core:boot@tp1"].min_free_gib is None


def test_a_status_line_with_a_field_this_version_does_not_know_still_loads(tmp_path):
    line = {"unit": "core", "gate": "boot", "member": "tp1", "passed": True, "later": [1]}
    (tmp_path / c.STATUS_FILENAME).write_text(json.dumps(line) + "\n")
    assert c.ContractStatus.load(tmp_path).latest()["core:boot@tp1"].passed


def test_the_margin_of_a_check_is_its_last_verdict_that_read_one(tmp_path):
    status = c.ContractStatus.load(tmp_path)
    status.record("mtp", _result("boot", "tp1", engine=_engine(0.52), warnings=(_LOW,)), "v1")
    # Out of memory before the engine was up: nothing read, the earlier margin stands.
    status.record("mtp", _result("boot", "tp1", passed=False), "v2")
    margin = status.margins()["mtp:boot@tp1"]
    assert (margin.min_free_gib, margin.low_margin, margin.marker) == (0.52, True, "v1")
    assert "core:boot@tp1" not in status.margins()


# ------------------------------------------------------------------ ordering

_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
        c.Check("core", "boot", "tp4", c.TIER_INNER),
        c.Check("mtp", "boot", "tp1", c.TIER_INNER),
        c.Check("mtp", "acceptance", "tp1", c.TIER_PROOF),
        c.Check("multimodal", "boot", "tp1", c.TIER_INNER),
    )
)


def _task(tid, kind, satisfies=(), deps=(), touches=(), ref=""):
    return g.Task(
        id=tid,
        kind=kind,
        goal=f"do {tid}",
        depends_on=list(deps),
        touches=list(touches),
        satisfies=list(satisfies),
        ref=ref,
        gpus=1,
        est_minutes=10,
    )


def _orchestrator(tmp_path, services=None, **kw):
    services = services or Services(
        plan=lambda turn: PlannerAnswer("DONE"),
        work=lambda task: "",
        check=None,
        free_gpus=lambda: 4,
    )
    status = c.ContractStatus.load(tmp_path)
    return GraphOrchestrator(services, _CONTRACT, status, tmp_path, poll_s=0, **kw)


_KEYS = [check.key for check in _CONTRACT.checks]


def test_without_margins_or_a_landed_change_the_contract_order_stands(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    assert orchestrator._by_risk(list(reversed(_KEYS))) == _KEYS
    assert orchestrator._by_risk([]) == []


def test_low_margin_and_the_tightest_go_first_then_least_free_memory(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    status = orchestrator.status
    status.record("core", _result("boot", "tp1", engine=_engine(30.0)))
    status.record("core", _result("boot", "tp4", engine=_engine(3.0), warnings=(_LOW,)))
    status.record("mtp", _result("boot", "tp1", engine=_engine(1.5)))
    status.record("multimodal", _result("boot", "tp1", engine=_engine(60.0)))
    assert orchestrator._by_risk(_KEYS) == [
        "mtp:boot@tp1",  # the least memory left, warned or not
        "core:boot@tp4",  # LOW MARGIN
        "core:boot@tp1",
        "multimodal:boot@tp1",
        "core:accuracy[gsm8k]@tp1",  # no reading: after those with one, in contract order
        "mtp:acceptance@tp1",
    ]


def test_the_units_the_last_landed_change_reaches_come_next(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    orchestrator.graph = g.Graph(
        tasks=[
            _task("core", "assemble", ["core:boot@tp1"], touches=["models/t/modeling.py"]),
            _task("mtp", "feature", ["mtp:boot@tp1"], touches=["models/t/mtp/", "spec/"]),
            _task("mm", "feature", ["multimodal:boot@tp1"], touches=["models/t/vision.py"]),
            _task("spec-fix", "feature", touches=["spec/utils.py"]),
        ]
    )
    orchestrator.status.record(
        "core", _result("boot", "tp4", engine=_engine(1.0), warnings=(_LOW,))
    )
    # A shared speculative change reaches mtp, the unit whose task declared its directory.
    orchestrator._landed = orchestrator.graph.by_id()["spec-fix"]
    assert orchestrator._by_risk(_KEYS) == [
        "core:boot@tp4",
        "mtp:boot@tp1",
        "mtp:acceptance@tp1",
        "core:boot@tp1",
        "core:accuracy[gsm8k]@tp1",
        "multimodal:boot@tp1",
    ]
    # A landed task reaches its own unit too, by the checks it claims or its ref.
    orchestrator._landed = _task(
        "mm-2", "feature", touches=["models/t/vision2.py"], ref="multimodal"
    )
    assert orchestrator._units_reached() == {"multimodal"}


def test_the_loop_runs_its_stale_checks_again_riskiest_first(tmp_path):
    margins = {"core:boot@tp1": (40.0, ()), "core:boot@tp4": (0.5, (_LOW,))}
    tree = {"marker": "v1"}
    ran: list[str] = []

    def check(chk):
        ran.append(chk.key)
        free, warnings = margins.get(chk.key, (None, ()))
        engine = _engine(free) if free is not None else ()
        return _result(chk.gate, chk.member, engine=engine, warnings=warnings)

    def work(task):
        if task.id == "spec-fix":
            tree["marker"] = "v2"  # every earlier verdict is stale now
        return "UNIT_READY"

    tasks = [
        _task("core", "assemble", _KEYS[:3], touches=["models/t/modeling.py"]),
        _task("mtp", "feature", _KEYS[3:5], ["core"], touches=["models/t/mtp/", "spec/"]),
        _task("mm", "feature", _KEYS[5:], ["mtp"], touches=["models/t/vision.py"]),
        _task("spec-fix", "feature", deps=["mm"], touches=["spec/utils.py"]),
    ]
    answers = iter([PlannerAnswer("GRAPH_UPDATED", "plan", g.Graph(tasks=tasks))])
    services = Services(
        plan=lambda turn: next(answers, PlannerAnswer("DONE", "all green")),
        work=work,
        check=check,
        free_gpus=lambda: 4,
        marker=lambda: tree["marker"],
    )
    orchestrator = _orchestrator(tmp_path, services, max_parallel=1, max_parallel_checks=1)
    assert orchestrator.run() == "delivered"
    assert ran[: len(_KEYS)] == _KEYS
    assert ran[len(_KEYS) :] == [
        "core:boot@tp4",  # LOW MARGIN, and the least memory
        "mtp:boot@tp1",  # the unit spec-fix's path reaches
        "mtp:acceptance@tp1",
        "core:boot@tp1",  # the rest, the one with a reading first
        "core:accuracy[gsm8k]@tp1",
        "multimodal:boot@tp1",
    ]
