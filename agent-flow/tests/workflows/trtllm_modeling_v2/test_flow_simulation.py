# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole runs on the CPU, first graph to delivery: the tail no GPU run has reached yet.

Each scenario drives the real orchestrator with the fakes of ``flow_sim``:
kernels, the core, its width, a feature and its width, a promotion per entry,
the delivery tier and the correctness milestone; a campaign beside them that
full support stops; failures the run must contain; and a target that keeps
changing while it is checked. Each ends in seconds.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g

from .flow_sim import Planner, Simulation, revise, task

_CORE_TP1 = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1"]
_CORE_TEP4 = ["core:boot@tep4"]
_MTP_TP1 = ["mtp:boot@tp1", "mtp:acceptance@tp1"]
_MTP_TEP4 = ["mtp:boot@tep4"]
_DELIVERY = ["mtp:envelope@tp1", "mtp:envelope@tep4"]


def _check(key: str) -> c.Check:
    unit, rest = key.split(":", 1)
    gate, member = rest.rsplit("@", 1)
    tier = (
        c.TIER_DELIVERY
        if gate == "envelope"
        else c.TIER_PROOF
        if gate == "acceptance"
        else c.TIER_INNER
    )
    return c.Check(unit, gate, member, tier)


_CONTRACT = c.Contract(
    checks=tuple(_check(k) for k in (*_CORE_TP1, *_CORE_TEP4, *_MTP_TP1, *_MTP_TEP4, *_DELIVERY))
)


def _graph(*extra: g.Task) -> list[g.Task]:
    """The plan a planner writes for a core, its width, a speculator and its width."""
    return [
        task("survey", "survey", touches=[], gpus=0),
        task("k1", "kernel", touches=["catalog/gemm/a.py"], ref="1"),
        task("k2", "kernel", touches=["catalog/norm/b.py"], ref="2"),
        task("core", "assemble", _CORE_TP1, deps=["k1", "k2"]),
        task("core-tep", "feature", _CORE_TEP4, deps=["core"]),
        task("mtp", "feature", _MTP_TP1, deps=["core"]),
        task("mtp-tep", "feature", [*_MTP_TEP4, *_DELIVERY], deps=["mtp", "core-tep"]),
        *extra,
    ]


@pytest.fixture(autouse=True)
def _quick(monkeypatch):
    monkeypatch.setattr(
        "agent_flow.workflows.trtllm_modeling_v2.run.orchestrator.PLANNER_RETRY_S", 0.01
    )


def _promotion(entries: dict[str, str]):
    """Each shared entry's local variant, landed by its kernel task and promoted on its own."""

    def plan(graph: g.Graph) -> SimpleNamespace:
        by_id = graph.by_id()
        waiting, spec, why = [], None, ""
        for entry, kernel in entries.items():
            promote = f"promote-{entry.rsplit('/', 1)[-1]}"
            if by_id.get(kernel) is None or by_id[kernel].status != "done":
                continue
            if promote in by_id and by_id[promote].status == "done":
                continue
            waiting.append(entry)
            if promote in by_id:
                why = f"(task {promote} promotes it)"
            elif spec is None:
                spec = {
                    "id": promote,
                    "kind": "kernel",
                    "ref": f"promote:{entry}",
                    "goal": f"promote {entry}",
                    "touches": [f"catalog/{entry}.py"],
                    "resources": {"gpus": 0},
                }
        return SimpleNamespace(entries=waiting, task=spec, waiting=why)

    return plan


def test_a_whole_run_ends_delivered_with_its_milestone_on_the_tree_it_delivered(tmp_path):
    planner = Planner(_graph())
    sim = Simulation(tmp_path, _CONTRACT, planner)
    sim.services.promotion = _promotion({"gemm/a": "k1", "norm/b": "k2"})

    assert sim.run() == "delivered"
    statuses = sim.statuses()
    assert all(s == "done" for s in statuses.values()), statuses
    # Each entry was promoted on its own, once its kernel had landed.
    for promote, kernel in (("promote-a", "k1"), ("promote-b", "k2")):
        assert statuses[promote] == "done"
        assert sim.at("start", promote)[0] > sim.at("end", kernel)[0]
    # Every dependency held: a task started after what it depends on ended.
    for t in sim.orchestrator.graph.tasks:
        for dep in t.depends_on:
            assert sim.at("start", t.id)[-1] > sim.at("end", dep)[-1], (t.id, dep)
    # Correctness was delivered on the tree the run ended on, every check passing there.
    final = sim.target.marker
    assert sim.milestones[-1] == final
    passed_there = {key for key, marker, ok in sim.ran if marker == final and ok}
    assert {chk.key for chk in _CONTRACT.required()} <= passed_there | {
        k for k, chk in ((x.key, x) for x in _CONTRACT.checks) if chk.equivalence
    }
    assert [m.marker for m in c.milestones(tmp_path)][-1] == final


def test_a_campaign_runs_beside_the_work_and_stops_once_support_is_complete(tmp_path):
    planner = Planner(_graph(task("tune", "tune", touches=[], gpus=0)))
    sim = Simulation(
        tmp_path,
        _CONTRACT,
        planner,
        required_kinds=("tune",),
        start_after={"tune": _CORE_TP1},
    )

    assert sim.run() == "delivered"
    # It started only once the core's inner checks held, and nothing waited for its budget.
    core_held = max(sim.when(f"check {key} pass")[0] for key in _CORE_TP1)
    assert sim.at("start", "tune")[0] > core_held
    assert sim.stop_reasons == ["full support holds on the target as it is"]
    assert sim.statuses()["tune"] == "done"
    assert sim.at("end", "tune")[0] < 15, "the run waited for the campaign's budget"
    # Its best branch is an artifact: nothing integrated it.
    assert "tune" not in sim.target.landed
    assert sim.milestones and sim.milestones[-1] == sim.target.marker


def test_failures_are_contained_and_no_result_is_lost(tmp_path):
    """A failed receipt naming an engine defect, a check whose session dies, a planner that raises."""
    fix = task("fix-k1", "feature", touches=["engine/x.py"])

    def retry_k1(orchestrator):
        again = task("k1", "kernel", touches=["catalog/gemm/a.py"], deps=["fix-k1"], ref="1")
        again.attempts = 2
        return revise(orchestrator, again, reason="retry k1 on the fix")

    planner = Planner(_graph(), rules=[("k1 (kernel) ended FAILED_RECEIPT", retry_k1)])
    planner.raising[3] = TimeoutError("the planner's turn ran past its bound")
    lost = {"core:boot@tp1": True}

    def raising(chk, version):
        if lost.get(chk.key):
            lost[chk.key] = False
            return True
        return False

    sim = Simulation(
        tmp_path,
        _CONTRACT,
        planner,
        decisions={"k1": ["FAILED_RECEIPT", "LANDED"]},
        raising=raising,
    )
    queued = [fix.as_dict()]
    sim.services.defect_tasks = lambda graph: [queued.pop()] if queued else []

    assert sim.run() == "delivered"
    statuses = sim.statuses()
    assert statuses["fix-k1"] == "done" and statuses["k1"] == "done"
    assert sim.at("start", "k1")[1] > sim.at("end", "fix-k1")[0], "k1 was retried after its fix"
    assert sim.when("raise core:boot@tp1") and statuses["core"] == "done"
    assert planner.heard("your previous turn raised TimeoutError")
    # Every required check has a verdict that passed on the delivered tree.
    final = sim.target.marker
    held = {
        v.key if hasattr(v, "key") else f"{v.unit}:{v.gate}@{v.member}"
        for v in sim.orchestrator.status.verdicts
        if v.passed and v.marker == final
    }
    assert {chk.key for chk in _CONTRACT.required()} <= held


def test_soft_edges_infrastructure_retries_and_an_exclusive_task_all_finish(tmp_path):
    graph = [
        task("k1", "kernel", touches=["catalog/gemm/a.py"], ref="1"),
        task("core", "assemble", _CORE_TP1, soft=["k1"]),
        task("core-tep", "feature", _CORE_TEP4, deps=["core"]),
        task("mtp", "feature", _MTP_TP1, deps=["core"]),
        task("mtp-tep", "feature", [*_MTP_TEP4, *_DELIVERY], deps=["mtp", "core-tep"]),
        task("probe", "investigate", deps=["core"], exclusive=True, priority="nice_to_have"),
    ]
    planner = Planner(graph)
    sim = Simulation(
        tmp_path,
        _CONTRACT,
        planner,
        decisions={"core": ["raised OSError: the session was lost", "UNIT_READY"]},
        durations={"k1": 0.2},
    )

    assert sim.run() == "delivered"
    assert all(s in ("done", "abandoned") for s in sim.statuses().values()), sim.statuses()
    assert len(sim.at("start", "core")) == 2, "the infrastructure failure was retried in place"
    assert not planner.heard("stalled")


def test_the_target_is_checked_while_it_changes_and_delivered_on_its_last_state(tmp_path):
    """A late landing is re-checked beside the work still running; the rehearsal is advisory."""
    graph = [
        task("core", "assemble", [*_CORE_TP1, *_CORE_TEP4]),
        task("mtp", "feature", [*_MTP_TP1, *_MTP_TEP4, *_DELIVERY], deps=["core"]),
        task("polish", "feature", deps=["core"], priority="nice_to_have"),
    ]
    planner = Planner(graph)
    sim = Simulation(tmp_path, _CONTRACT, planner, durations={"mtp": 0.4, "polish": 0.05})

    assert sim.run() == "delivered"
    polish_landed = sim.at("land", "polish")[0]
    mtp_ended = sim.at("end", "mtp")[0]
    rechecked = [t for t in sim.when("check core:") if polish_landed < t < mtp_ended]
    assert rechecked, "nothing re-checked the target after the late landing while mtp ran"
    assert sim.milestones[-1] == sim.target.marker


def test_a_done_that_leaves_full_support_out_of_reach_ends_the_run_and_never_spins(tmp_path):
    """Every check holds and something else does not: DONE again is no progress either."""
    planner = Planner(_graph())
    sim = Simulation(tmp_path, _CONTRACT, planner, revision_cap=3)
    sim.services.delivery_files = lambda: ["tests: INCOMPLETE (1 not run at the head)"]
    # Only the delivery checks on disk fail: the delivery phase reports them as its verdict.
    assert sim.run(timeout=20) == "delivered"
    assert planner.heard("planner said DONE, but full support does not hold")
    assert not sim.milestones, "correctness is not recorded on a tree whose files fail"

    waiting = Planner(_graph())
    (tmp_path / "other").mkdir()
    stuck = Simulation(tmp_path / "other", _CONTRACT, waiting, revision_cap=3)
    stuck.services.promotion = lambda graph: SimpleNamespace(
        entries=["gemm/a"], task=None, waiting="(the target does not import it yet)"
    )
    assert stuck.run(timeout=20) == "planner-stalled"
