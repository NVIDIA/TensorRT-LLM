# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A relaunched orchestrator adopts a task's answered turn instead of asking again."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.dispatch import (
    CONSUMED_SUFFIX,
    DISPATCH_DIRNAME,
    READ_SUFFIX,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import TurnRequest
from agent_flow.workflows.trtllm_modeling_v2.infra.external.transport import ExternalAgentTransport
from agent_flow.workflows.trtllm_modeling_v2.run import graph as graph_mod
from agent_flow.workflows.trtllm_modeling_v2.run.progress import (
    decision_value,
    find_entries,
    mark_decision_dispatch,
)

from .test_workflow import _workflow


def _assembler_decision(path: Path) -> bool:
    return decision_value(path, "assembler") is not None


def _dead_turn(workspace: Path, request_id: str, *, what: str = "core-tp") -> Path:
    """The files a dead process and its driver left for one assembler turn; its decision path."""
    role = request_id.split("-", 1)[1]
    dispatch = workspace / DISPATCH_DIRNAME
    dispatch.mkdir(parents=True, exist_ok=True)
    decision_file = mark_decision_dispatch(workspace, role, 5, unit="core")
    (dispatch / f"{request_id}.request.json").write_text(
        json.dumps(
            {
                "id": request_id,
                "role": role,
                "what": what,
                "decision_file": str(decision_file),
                "requested_at": time.time() - 600,
            }
        ),
        encoding="utf-8",
    )
    decision_file.write_text(
        yaml.safe_dump({"decision": "UNIT_READY", "summary": f"{what} assembled"}),
        encoding="utf-8",
    )
    (dispatch / f"{request_id}.result.md").write_text(f"{what}: UNIT_READY", encoding="utf-8")
    (dispatch / f"{request_id}.done").write_text("1\n", encoding="utf-8")
    return decision_file


def _requests(workspace: Path) -> list[str]:
    return sorted(
        p.name[: -len(".request.json")]
        for p in (workspace / DISPATCH_DIRNAME).glob("*.request.json")
    )


def test_a_consumed_turn_is_never_adopted(tmp_path):
    _dead_turn(tmp_path, "066-assembler")
    first = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert first.consume(["core-tp"]) == ["066-assembler"]
    assert (first.root / f"066-assembler{CONSUMED_SUFFIX}").is_file()
    assert first.consume(["core-tp"]) == [], "consumed once"

    resumed = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert resumed.adopt_answered("assembler", "core-tp", _assembler_decision) is None


def test_a_turn_this_process_asked_or_already_adopted_is_never_adopted(tmp_path):
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    decision = tmp_path / "decisions" / "core-assembler-1.yaml"
    decision.parent.mkdir(parents=True)

    def answer():
        while not (transport.root / "001-assembler.request.json").exists():
            time.sleep(0.005)
        decision.write_text("decision: UNIT_READY\n", encoding="utf-8")
        (transport.root / "001-assembler.done").write_text("1\n", encoding="utf-8")

    threading.Thread(target=answer, daemon=True).start()
    transport.run(
        TurnRequest(role="assembler", what="core-tp", prompt="p", decision_file=str(decision))
    )
    assert (transport.root / f"001-assembler{READ_SUFFIX}").is_file(), "hand-back is recorded"
    assert transport.adopt_answered("assembler", "core-tp", _assembler_decision) is None

    _dead_turn(tmp_path, "000-assembler", what="core-dp")
    assert transport.adopt_answered("assembler", "core-dp", _assembler_decision) is not None
    assert transport.adopt_answered("assembler", "core-dp", _assembler_decision) is None


def test_a_relaunch_adopts_the_answered_turn_of_a_task_still_running(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    transport = ExternalAgentTransport(workspace, poll_s=0.01, timeout_s=30)
    workflow = _workflow(tmp_path, agent_transport=transport)
    workflow._turns_by_key = {}
    _dead_turn(workspace, "066-assembler")
    graph_mod.Graph(
        tasks=[graph_mod.Task(id="core-tp", kind="feature", goal="g", status="running", ref="core")]
    ).save(workspace / graph_mod.GRAPH_FILENAME)

    decision = workflow._dispatch_task("assembler", "core", "core-tp", "Assemble core-tp.")

    assert decision == "UNIT_READY"
    assert _requests(workspace) == ["066-assembler"], "no second request for the same turn"
    assert not (workspace / "decisions" / "core-assembler-1.yaml").exists(), "nothing copied"
    entry = find_entries(workflow.progress_path, stage="build_stage", agent="assembler")[-1]
    assert entry["decision_file"] == "core-assembler-5.yaml"
    journal = workflow.journal_path.read_text(encoding="utf-8")
    assert "decision adopted on resume" in journal and "066-assembler" in journal
