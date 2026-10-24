# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The external transport end to end: the orchestrator's side and the driver's."""

from __future__ import annotations

import json
import re
import threading
import time

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.dispatch import (
    ORCHESTRATOR_EXIT,
    ORCHESTRATOR_HEARTBEAT,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.driver import core
from agent_flow.workflows.trtllm_modeling_v2.infra.driver.config import load_config
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import TurnRequest, TurnTimeout
from agent_flow.workflows.trtllm_modeling_v2.infra.external import driver as external_driver
from agent_flow.workflows.trtllm_modeling_v2.infra.external.transport import ExternalAgentTransport
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import write_json_atomic


def _answer(transport, request_id, result):
    """What a driver does: the result, then ``.done`` last."""
    (transport.root / f"{request_id}.result.md").write_text(result, encoding="utf-8")
    (transport.root / f"{request_id}.done").write_text("1\n", encoding="utf-8")


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "repo").mkdir()
    task = {"trtllm_repo_path": str(tmp_path / "repo")}
    (ws / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    return ws


def _ask(
    transport,
    *,
    role="kernel_writer",
    what="onboard x",
    decision_file=None,
    backend=None,
    model=None,
):
    """Send one turn from a thread; return the thread and a box for its outcome."""
    box: dict = {}

    def run():
        try:
            box["outcome"] = transport.run(
                TurnRequest(
                    role=role,
                    what=what,
                    prompt="do it",
                    system_prompt="you are a kernel_writer",
                    decision_file=decision_file,
                    backend=backend,
                    model=model,
                )
            )
        except TurnTimeout as exc:
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def _wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def _request_ids(ws) -> list[str]:
    return sorted(p.name[: -len(".request.json")] for p in (ws / "dispatch").glob("*.request.json"))


def test_a_turn_times_out_on_its_roles_bound(workspace):
    transport = ExternalAgentTransport(
        workspace, poll_s=0.01, role_timeouts={"surveyor": 0.2}, heartbeat_s=None
    )
    with pytest.raises(TurnTimeout, match="surveyor turn timeout"):
        transport.run(TurnRequest(role="surveyor", what="survey", prompt="p"))


def test_a_relaunch_adopts_an_orphaned_turn_whose_decision_path_moved(workspace):
    """A relaunch numbers the same turn differently; it waits on the unanswered request."""
    dispatch = workspace / "dispatch"
    dispatch.mkdir()
    old_decision = workspace / "decisions" / "assembler-1.yaml"
    new_decision = workspace / "decisions" / "assembler-2.yaml"
    write_json_atomic(
        dispatch / "046-assembler.request.json",
        {
            "id": "046-assembler",
            "role": "assembler",
            "what": "dspark",
            "decision_file": str(old_decision),
            "requested_at": time.time() - 600,
        },
    )
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, box = _ask(transport, role="assembler", what="dspark", decision_file=str(new_decision))
    time.sleep(0.3)
    assert _request_ids(workspace) == ["046-assembler"], "no second request for the same turn"
    old_decision.parent.mkdir(parents=True)
    old_decision.write_text("decision: UNIT_READY\n")
    _answer(transport, "046-assembler", "assembled")
    thread.join(timeout=10)
    assert box["outcome"].request_id == "046-assembler"
    assert new_decision.read_text() == "decision: UNIT_READY\n"


def test_a_turn_this_process_asked_is_never_adopted_across_decision_paths(workspace):
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    first, _ = _ask(transport, role="assembler", what="core", decision_file="/d/assembler-1.yaml")
    _wait_for(lambda: len(_request_ids(workspace)) == 1)
    second, _ = _ask(transport, role="assembler", what="core", decision_file="/d/assembler-2.yaml")
    _wait_for(lambda: len(_request_ids(workspace)) == 2)
    for request_id in _request_ids(workspace):
        _answer(transport, request_id, "ok")
    first.join(timeout=10)
    second.join(timeout=10)


def _orchestrate(workspace, *turns):
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    threads = [_ask(transport, **turn)[0] for turn in turns]
    _wait_for(lambda: len(_request_ids(workspace)) == len(turns))
    return transport, threads


def test_claims_are_exclusive(workspace):
    transport, threads = _orchestrate(workspace, {"what": "a"})
    cfg = load_config(workspace)
    (request_id,) = _request_ids(workspace)
    external_driver.claim(cfg, request_id, by="driver-1")
    with pytest.raises(core.DriverError, match="already claimed by driver-1"):
        external_driver.claim(cfg, request_id, by="driver-2")
    assert external_driver.pending(cfg) == [] and len(external_driver.in_flight(cfg)) == 1
    external_driver.release(cfg, request_id)
    assert [r["id"] for r in external_driver.pending(cfg)] == [request_id]
    _answer(transport, request_id, "ok")
    with pytest.raises(core.DriverError, match="already answered"):
        external_driver.claim(cfg, request_id)
    for thread in threads:
        thread.join(timeout=10)


def _request_of(workspace, role, what):
    for request_id in _request_ids(workspace):
        data = json.loads((workspace / "dispatch" / f"{request_id}.request.json").read_text())
        if data["role"] == role and data["what"] == what:
            return request_id
    raise AssertionError(f"no {role} request for {what}")


def test_the_planners_turn_never_waits_behind_workers(workspace):
    """A planner turn is handed out even with every worker slot taken."""
    transport, threads = _orchestrate(
        workspace,
        {"what": "a"},
        {"what": "b"},
        {"role": "planner", "what": "planner"},
    )
    cfg = load_config(workspace, max_inflight=1)
    external_driver.claim(cfg, _request_of(workspace, "kernel_writer", "a"))
    first = core.wait(cfg, timeout_s=5, poll_s=0.01, watch_sessions=False)
    assert first.kind == "NEW" and first.request_id == _request_of(workspace, "planner", "planner")
    external_driver.claim(cfg, first.request_id)
    # b is a worker's turn: it waits at the cap.
    second = core.wait(cfg, timeout_s=0.2, poll_s=0.01, watch_sessions=False)
    assert second.kind == "TIMEOUT" and "in-flight cap, 1/1" in second.detail
    for request_id in _request_ids(workspace):
        _answer(transport, request_id, "ok")
    for thread in threads:
        thread.join(timeout=10)


def test_wait_returns_when_the_orchestrator_exits(workspace):
    """A closed run writes no further request, so the waiter must notice the exit."""
    dispatch = workspace / "dispatch"
    dispatch.mkdir()
    cfg = load_config(workspace)
    assert core.orchestrator_state(cfg)[0] == "unknown"
    write_json_atomic(dispatch / ORCHESTRATOR_HEARTBEAT, {"pid": 1, "host": "h", "at": time.time()})
    assert core.orchestrator_state(cfg)[0] == "running"
    assert core.wait(cfg, timeout_s=0.1, poll_s=0.01, watch_sessions=False).kind == "TIMEOUT"
    write_json_atomic(dispatch / ORCHESTRATOR_EXIT, {"pid": 1, "host": "h", "at": time.time() + 1})
    event = core.wait(cfg, timeout_s=5, poll_s=0.01, watch_sessions=False)
    assert event.kind == "ORCHESTRATOR_EXITED" and "pid 1" in event.detail


def test_the_prompt_resolves_everything_a_subagent_needs(workspace):
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, _ = _ask(
        transport,
        what="onboard topk",
        decision_file=str(workspace / "decisions" / "kernel_writer-3.yaml"),
        backend="claude-code",
        model="claude-opus-5-5",
    )
    _wait_for(lambda: _request_ids(workspace))
    (request_id,) = _request_ids(workspace)
    text = external_driver.render_prompt(load_config(workspace), request_id)
    assert "$" not in re.sub(r"\$[A-Z_]+|\$\(", "", text), "an unrendered template field"
    for fragment in (
        f"export MV2FLOW_JOB_PREFIX=r{request_id.split('-')[0]}",
        str(workspace / "dispatch" / f"{request_id}.system.md"),
        str(workspace / "dispatch" / f"{request_id}.done"),
        str(workspace / "decisions" / "kernel_writer-3.yaml"),
        "model `claude-opus-5-5`",
        "spoolctl.py cancel",
        "6.0 h",
    ):
        assert fragment in text, fragment
    _answer(transport, request_id, "ok")
    thread.join(timeout=10)


def test_the_driver_can_hand_a_turn_back_in_order(workspace):
    transport, threads = _orchestrate(workspace, {"what": "x"})
    (request_id,) = _request_ids(workspace)
    source = workspace / "result.md"
    source.write_text("blocked: no GPU")
    external_driver.answer(load_config(workspace), request_id, source.read_text())
    threads[0].join(timeout=10)
    assert (workspace / "dispatch" / f"{request_id}.result.md").read_text() == "blocked: no GPU"


def test_adoption_never_overwrites_a_decision_that_already_exists(workspace):
    dispatch = workspace / "dispatch"
    dispatch.mkdir()
    theirs = workspace / "decisions" / "core-assembler-1.yaml"
    ours = workspace / "decisions" / "core-assembler-1.2.yaml"
    theirs.parent.mkdir()
    theirs.write_text("decision: UNIT_READY\n")
    ours.write_text("decision: BLOCKED\n")
    write_json_atomic(
        dispatch / "010-assembler.request.json",
        {
            "id": "010-assembler",
            "role": "assembler",
            "what": "core",
            "decision_file": str(theirs),
            "requested_at": time.time() - 60,
        },
    )
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, box = _ask(transport, role="assembler", what="core", decision_file=str(ours))
    time.sleep(0.2)
    _answer(transport, "010-assembler", "ok")
    thread.join(timeout=10)
    assert box["outcome"].request_id == "010-assembler"
    assert ours.read_text() == "decision: BLOCKED\n", (
        "an existing decision file is never overwritten"
    )
    assert theirs.read_text() == "decision: UNIT_READY\n"
