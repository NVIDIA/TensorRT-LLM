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
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c

from .test_workflow import _workflow

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


def test_once_the_target_imports_an_entrys_variants_the_orchestrator_promotes_it(tmp_path):
    orchestrator, verdict, worked, heard = _run(tmp_path)
    assert verdict == "delivered"
    # Due once the core, which imports the variant, has landed: it rewrites code that exists.
    assert worked == ["k3-thop", "core", "promote"]
    promote = orchestrator.graph.by_id()["promote"]
    assert promote.ref == "promote:attention/thop" and promote.status == "done"
    assert VARIANT in promote.touches and f"{TARGET}/modeling.py" in promote.touches
    assert any("the orchestrator added promote" in e for e in heard)
    assert orchestrator.graph.revisions[-1].reason.startswith("orchestrator: promote promotes")
    # Last, once the contract holds; its one check is a boot of the core on the promoted tree.
    assert promote.satisfies == ["core:boot@tp1"]


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


def test_a_planner_added_promote_reaches_its_kernel_writer(tmp_path):
    workflow = _workflow(tmp_path)
    task = g.Task(id="promote", kind="kernel", goal="merge", ref="promote:attention/thop")
    item = ExecutionStage(workflow)._queue_item(task)
    assert item.mode == vocab.MODE_PROMOTE and item.entries == ("attention/thop",)
    site = g.Task(id="k-x", kind="kernel", goal="", ref="verify:attention/thop@decode")
    assert ExecutionStage(workflow)._queue_item(site).variant == "ops/attention/thop__decode"
