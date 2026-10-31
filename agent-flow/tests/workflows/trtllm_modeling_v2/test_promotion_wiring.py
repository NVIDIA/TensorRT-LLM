# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The orchestrator adds the promote task itself, and does not deliver past a variant."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as vocab
from agent_flow.workflows.trtllm_modeling_v2.operators import promotion as promo
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c

TARGET = f"{catalog_paths.MODELING_V2_PACKAGE}/models/newmodel/base__sm_103__tp1"
VARIANT = f"{TARGET}/ops/attention/thop__r3.py"
_CONTRACT = c.Contract(checks=(c.Check("core", "boot", "tp1", c.TIER_INNER),))


def _write(root: Path, rel: str, text: str = "") -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")


def _run(tmp_path, promote_decision="LANDED", passing=True, **kw):
    repo = tmp_path / "repo"
    _write(repo, f"{catalog_paths.CATALOG_DIR}/attention/thop.py")
    kernel = g.Task(id="k3-thop", kind="kernel", goal="verify", ref="3", gpus=1,
                    touches=list(vocab.entry_paths(
                        vocab.QueueItem(n=3, mode="verify", target="attention/thop"), TARGET)))  # fmt: skip
    core = g.Task(id="core", kind="assemble", goal="core", depends_on=["k3-thop"], gpus=1,
                  touches=[f"{TARGET}/modeling.py"], satisfies=["core:boot@tp1"])  # fmt: skip
    turns, worked = [], []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer("GRAPH_UPDATED", "kernel and core", g.Graph(tasks=[kernel, core]))
        return PlannerAnswer("DONE", "all green")

    def work(task):
        worked.append(task.id)
        if task.id == "k3-thop":
            _write(repo, VARIANT)
        elif task.id == "core":
            _write(repo, f"{TARGET}/modeling.py", "from .ops.attention.thop__r3 import thop\n")
        elif task.ref.startswith("promote:") and promote_decision == "LANDED":
            (repo / VARIANT).unlink()
            _write(repo, f"{TARGET}/modeling.py", "from tensorrt_llm._torch._experimental."
                   "modeling_v2.catalog.attention.thop import thop\n")  # fmt: skip
        return promote_decision if task.ref.startswith("promote:") else "LANDED"

    services = Services(
        plan=plan,
        work=work,
        check=lambda check: SimpleNamespace(
            name=check.gate, member=check.member, passed=passing, detail=""
        ),
        free_gpus=lambda: 8,
        promotion=lambda graph: promo.plan(graph.tasks, repo, TARGET),
    )
    orchestrator = GraphOrchestrator(
        services, _CONTRACT, c.ContractStatus.load(tmp_path), tmp_path, poll_s=0.01, **kw
    )
    verdict = orchestrator.run()
    heard = [e for turn in turns for e in turn.events] + [
        event.text for event in orchestrator.events
    ]
    return orchestrator, verdict, worked, heard


def test_no_promote_starts_before_the_contract_holds(tmp_path):
    orchestrator, verdict, worked, heard = _run(tmp_path, passing=False, revision_cap=2)
    assert verdict != "delivered"
    assert "promote" not in worked and "promote" not in orchestrator.graph.by_id()


def test_a_done_promote_carries_the_verdicts_of_the_tree_it_started_from(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import promotions

    promotions.started(tmp_path, "promote", "tree-a")
    assert promotions.carried(tmp_path, "tree-b") == set()  # not done yet
    promotions.ended(tmp_path, "promote", "tree-b")
    promotions.started(tmp_path, "promote-2", "tree-b")
    promotions.ended(tmp_path, "promote-2", "tree-c")
    assert promotions.carried(tmp_path, "tree-c") == {"tree-a", "tree-b"}
    assert promotions.carried(tmp_path, "tree-d") == set()  # the tree moved on since
    promotions.started(tmp_path, "promote-3", "tree-c")
    promotions.ended(tmp_path, "promote-3", None)  # its boot failed: it carries nothing
    assert promotions.carried(tmp_path, "tree-c") == {"tree-a", "tree-b"}


def test_a_failed_promote_keeps_a_holding_contract_from_delivery(tmp_path):
    orchestrator, verdict, worked, heard = _run(tmp_path, "BLOCKED", revision_cap=2)
    assert verdict != "delivered"
    assert worked.count("promote") == 1  # no second promote behind a failed one
    assert any("local variants wait for promotion" in e for e in heard)
