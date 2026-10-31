# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The loop keeps moving while the planner thinks.

Every fake that waits for another thread waits on an event with a deadline,
so a loop that serializes what should overlap fails the test instead of
hanging it.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import inbox
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    GraphOrchestrator,
    PlannerAnswer,
    Services,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c

_CONTRACT = c.Contract(
    checks=(
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
    )
)
_KEYS = [check.key for check in _CONTRACT.checks]


def _passed(check):
    return SimpleNamespace(name=check.gate, member=check.member, passed=True, detail="")


def _orchestrator(tmp_path, plan, work, **kw):
    services = Services(plan=plan, work=work, check=_passed, free_gpus=lambda: 8)
    status = c.ContractStatus.load(tmp_path)
    return GraphOrchestrator(services, _CONTRACT, status, tmp_path, poll_s=0.01, **kw)


def test_a_features_own_work_waits_for_the_core_to_boot(tmp_path):
    started, seen = [], {}

    def task(tid, kind, ref, satisfies=()):
        return g.Task(
            id=tid,
            kind=kind,
            ref=ref,
            goal=f"do {tid}",
            touches=[f"models/{tid}.py"],
            gpus=1,
            est_minutes=10,
            satisfies=list(satisfies),
        )

    def plan(turn):
        if not started:
            graph = g.Graph(
                tasks=[
                    task("core", "assemble", "core", satisfies=_KEYS),
                    task("mtp", "feature", "mtp"),
                    task("k1-draft-head", "kernel", "gemm/draft_head"),
                ]
            )
            return PlannerAnswer("GRAPH_UPDATED", "plan", graph)
        return PlannerAnswer("DONE", "all green")

    def work(t):
        started.append(t.id)
        if t.id == "core":
            time.sleep(0.3)  # long enough for the loop to start whatever it would
            seen["beside the core"] = set(started) - {"core"}
        return "UNIT_READY"

    orchestrator = _orchestrator(tmp_path, plan, work, after_core=(["mtp"], [_KEYS[0]]))
    assert orchestrator.run() == "delivered"
    # The kernel the feature needs starts beside the core; the feature waits for its boot.
    assert seen["beside the core"] == {"k1-draft-head"}
    assert started.index("mtp") > started.index("core")


def test_a_burst_of_short_operator_steers_reaches_the_planner_as_one(tmp_path):
    with pytest.raises(ValueError, match="more than 1200"):
        inbox.post(tmp_path, "x" * (inbox.MESSAGE_MAX_CHARS + 1))
    inbox.post(tmp_path, "drop the vision work")
    last = inbox.post(tmp_path, "stop the dep4 boot loop", now=True)
    assert inbox.take(tmp_path, now=last.stat().st_mtime + inbox.SETTLE_S / 2) == []
    taken = inbox.take(tmp_path, now=last.stat().st_mtime + inbox.SETTLE_S + 1)
    assert [text.rsplit(": ", 1)[-1] for text in taken] == [
        "drop the vision work",
        "stop the dep4 boot loop",
    ]
    assert [inbox.is_urgent(e) for e in taken] == [False, True]
    assert all(inbox.is_message(e) for e in taken) and ".now" not in taken[1]
