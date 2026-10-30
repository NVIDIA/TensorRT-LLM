# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A relaunched orchestrator adopts a task's answered turn instead of asking again.

On the external transport a process can die after the driver answered a task's
turn and before the task had its outcome: before the answer was read, or while
the checks that run after the turn were going. The relaunched process starts
the task again. Asking its turn again pays for the work twice, and someone has
to answer the new request -- by hand, by copying the old decision to the new
path. These pin the adoption rule: the same task, the same role, the latest
request, answered and not consumed; never a turn whose task already had its
outcome, never another task's, never on a later dispatch in the same process.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.driver import core as driver
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import TurnRequest
from agent_flow.workflows.trtllm_modeling_v2.infra.transport import (
    ADOPTED_SUFFIX,
    CONSUMED_SUFFIX,
    DISPATCH_DIRNAME,
    READ_SUFFIX,
    ExternalAgentTransport,
)
from agent_flow.workflows.trtllm_modeling_v2.run import graph as graph_mod
from agent_flow.workflows.trtllm_modeling_v2.run.progress import (
    decision_value,
    find_entries,
    mark_decision_dispatch,
)

from .test_workflow import _workflow


def _assembler_decision(path: Path) -> bool:
    return decision_value(path, "assembler") is not None


def _dead_turn(
    workspace: Path,
    request_id: str,
    *,
    what: str = "core-tp",
    unit: str = "core",
    turn: int = 5,
    answered: bool = True,
    decision: str | None = "UNIT_READY",
    read_by_pid: int | None = None,
) -> Path:
    """The files a dead process and its driver left for one assembler turn; its decision path."""
    role = request_id.split("-", 1)[1]
    dispatch = workspace / DISPATCH_DIRNAME
    dispatch.mkdir(parents=True, exist_ok=True)
    decision_file = mark_decision_dispatch(workspace, role, turn, unit=unit)
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
    if decision is not None:
        decision_file.write_text(
            yaml.safe_dump({"decision": decision, "summary": f"{what} assembled"}),
            encoding="utf-8",
        )
    if answered:
        (dispatch / f"{request_id}.result.md").write_text(f"{what}: {decision}", encoding="utf-8")
        (dispatch / f"{request_id}.done").write_text("1\n", encoding="utf-8")
    if read_by_pid is not None:
        (dispatch / f"{request_id}{READ_SUFFIX}").write_text(
            json.dumps({"pid": read_by_pid, "host": "gone", "started": 0.0, "at": 1.0}),
            encoding="utf-8",
        )
    return decision_file


def _requests(workspace: Path) -> list[str]:
    return sorted(
        p.name[: -len(".request.json")]
        for p in (workspace / DISPATCH_DIRNAME).glob("*.request.json")
    )


# ------------------------------------------------------------- the transport


def test_an_answer_no_process_read_is_adopted_after_a_restart(tmp_path):
    """core-tp 073: the process died with the turn in flight; the driver answered it later."""
    decision = _dead_turn(tmp_path, "066-assembler")
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)

    adopted = transport.adopt_answered("assembler", "core-tp", _assembler_decision)

    assert adopted is not None
    assert adopted.request_id == "066-assembler"
    assert adopted.decision_file == decision, "read where it was written, not copied"
    assert adopted.text == "core-tp: UNIT_READY"
    assert adopted.read_by is None
    record = json.loads((transport.root / f"066-assembler{ADOPTED_SUFFIX}").read_text())
    assert record["answered"] is True
    assert (transport.root / f"066-assembler{READ_SUFFIX}").is_file()
    assert _requests(tmp_path) == ["066-assembler"]


def test_an_answer_read_by_a_process_that_died_during_the_checks_is_adopted(tmp_path):
    """var-kv-dep4 092: the answer was read, then the process died running the task's checks.

    Reading an answer is not the run being done with it: the task never had
    its outcome, so the next process starts it again.
    """
    _dead_turn(tmp_path, "084-assembler", what="var-kv-dep4", read_by_pid=4242)
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)

    adopted = transport.adopt_answered("assembler", "var-kv-dep4", _assembler_decision)

    assert adopted is not None and adopted.request_id == "084-assembler"
    assert adopted.read_by["pid"] == 4242


def test_a_consumed_turn_is_never_adopted(tmp_path):
    _dead_turn(tmp_path, "066-assembler")
    first = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert first.consume(["core-tp"]) == ["066-assembler"]
    assert (first.root / f"066-assembler{CONSUMED_SUFFIX}").is_file()
    assert first.consume(["core-tp"]) == [], "consumed once"

    resumed = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert resumed.adopt_answered("assembler", "core-tp", _assembler_decision) is None


def test_another_tasks_or_another_roles_turn_is_never_adopted(tmp_path):
    """disagg-2 118: the planner's clone of `disagg` is a different task, asked afresh."""
    _dead_turn(tmp_path, "106-assembler", what="disagg", unit="disagg", turn=1)
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)

    assert transport.adopt_answered("assembler", "disagg-2", _assembler_decision) is None
    assert transport.adopt_answered("reviewer", "disagg", lambda path: True) is None
    assert not (transport.root / f"106-assembler{ADOPTED_SUFFIX}").exists()


def test_only_the_latest_request_for_the_work_is_adopted(tmp_path):
    """An answer followed by another request for the same work was superseded by it."""
    _dead_turn(tmp_path, "066-assembler")
    _dead_turn(tmp_path, "073-assembler", turn=1, answered=False, decision=None)
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert transport.adopt_answered("assembler", "core-tp", _assembler_decision) is None

    (transport.root / "073-assembler.done").write_text("1\n", encoding="utf-8")
    (tmp_path / "decisions" / "core-assembler-1.yaml").write_text(
        "decision: UNIT_READY\n", encoding="utf-8"
    )
    adopted = transport.adopt_answered("assembler", "core-tp", _assembler_decision)
    assert adopted is not None and adopted.request_id == "073-assembler"


def test_a_decision_the_caller_refuses_is_not_adopted(tmp_path):
    """The decision file must exist and parse; otherwise the turn is asked again."""
    _dead_turn(tmp_path, "066-assembler", decision=None)
    _dead_turn(tmp_path, "067-assembler", what="core-dp", turn=6, decision="LANDED")
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)

    assert transport.adopt_answered("assembler", "core-tp", _assembler_decision) is None
    assert transport.adopt_answered("assembler", "core-dp", _assembler_decision) is None
    assert not list(transport.root.glob(f"*{ADOPTED_SUFFIX}"))


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


def test_consume_leaves_an_unanswered_turn_alone(tmp_path):
    _dead_turn(tmp_path, "066-assembler", answered=False)
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert transport.consume(["core-tp"]) == []
    assert transport.consume([]) == []


# ------------------------------------------------------- the planned dispatch


def _graph(workspace: Path, **statuses: str) -> None:
    graph_mod.Graph(
        tasks=[
            graph_mod.Task(id=tid.replace("_", "-"), kind="feature", goal="g", status=s, ref="core")
            for tid, s in statuses.items()
        ]
    ).save(workspace / graph_mod.GRAPH_FILENAME)


def _relaunched(tmp_path: Path):
    """A workflow whose turns go through the external transport, as after a relaunch."""
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    transport = ExternalAgentTransport(workspace, poll_s=0.01, timeout_s=30)
    workflow = _workflow(tmp_path, agent_transport=transport)
    workflow._turns_by_key = {}
    return workflow, transport


def _drive(transport: ExternalAgentTransport, decision: str = "UNIT_READY") -> threading.Event:
    """Play the driver: answer every new request, decision file first and ``.done`` last."""
    stop = threading.Event()
    seen = {p.name for p in transport.root.glob("*.request.json")}

    def run():
        while not stop.is_set():
            for path in sorted(transport.root.glob("*.request.json")):
                if path.name in seen:
                    continue
                seen.add(path.name)
                request = json.loads(path.read_text(encoding="utf-8"))
                Path(request["decision_file"]).write_text(
                    yaml.safe_dump({"decision": decision, "summary": "asked again"}),
                    encoding="utf-8",
                )
                request_id = request["id"]
                (transport.root / f"{request_id}.result.md").write_text("ok", encoding="utf-8")
                (transport.root / f"{request_id}.done").write_text("1\n", encoding="utf-8")
            time.sleep(0.005)

    threading.Thread(target=run, daemon=True).start()
    return stop


def test_a_relaunch_adopts_the_answered_turn_of_a_task_still_running(tmp_path, monkeypatch):
    """Nothing is asked, nothing is copied: the decision is collected where it was written."""
    workflow, transport = _relaunched(tmp_path)
    workspace = workflow.workspace
    _dead_turn(workspace, "066-assembler")
    _graph(workspace, core_tp="running")

    decision = workflow._dispatch_task("assembler", "core", "core-tp", "Assemble core-tp.")

    assert decision == "UNIT_READY"
    assert _requests(workspace) == ["066-assembler"], "no second request for the same turn"
    assert not (workspace / "decisions" / "core-assembler-1.yaml").exists(), "nothing copied"
    entry = find_entries(workflow.progress_path, stage="build_stage", agent="assembler")[-1]
    assert entry["decision_file"] == "core-assembler-5.yaml"
    journal = workflow.journal_path.read_text(encoding="utf-8")
    assert "decision adopted on resume" in journal and "066-assembler" in journal


def test_a_task_with_an_outcome_is_asked_afresh_when_reopened(tmp_path):
    """Its answer is consumed while the graph says it finished; a retry is a new turn."""
    workflow, transport = _relaunched(tmp_path)
    workspace = workflow.workspace
    _dead_turn(workspace, "066-assembler")
    _graph(workspace, core_tp="failed")
    stop = _drive(transport)
    try:
        workflow._dispatch_task("plan_reviewer", "graph", "critic", "Review the graph.")
        assert (transport.root / f"066-assembler{CONSUMED_SUFFIX}").is_file()

        _graph(workspace, core_tp="pending")
        decision = workflow._dispatch_task("assembler", "core", "core-tp", "Retry core-tp.")
    finally:
        stop.set()

    assert decision == "UNIT_READY"
    assert not (transport.root / f"066-assembler{ADOPTED_SUFFIX}").exists()
    assert len([r for r in _requests(workspace) if r.endswith("-assembler")]) == 2


def test_a_later_dispatch_of_the_task_in_the_same_process_is_asked(tmp_path):
    """A rebase dispatches the task again; adopting would hand it the answer it revises."""
    workflow, transport = _relaunched(tmp_path)
    workspace = workflow.workspace
    _dead_turn(workspace, "066-assembler")
    _graph(workspace, core_tp="running")
    assert workflow._dispatch_task("assembler", "core", "core-tp", "Assemble.") == "UNIT_READY"

    stop = _drive(transport, decision="BLOCKED")
    try:
        decision = workflow._dispatch_task("assembler", "core", "core-tp", "Rebase it.")
    finally:
        stop.set()

    assert decision == "BLOCKED"
    assert len(_requests(workspace)) == 2


def test_another_tasks_answer_is_not_adopted_by_the_dispatch(tmp_path):
    workflow, transport = _relaunched(tmp_path)
    workspace = workflow.workspace
    _dead_turn(workspace, "106-assembler", what="disagg", unit="disagg", turn=1)
    _graph(workspace, disagg="failed", disagg_2="running")
    stop = _drive(transport)
    try:
        decision = workflow._dispatch_task("assembler", "disagg", "disagg-2", "Assemble disagg.")
    finally:
        stop.set()

    assert decision == "UNIT_READY"
    assert len(_requests(workspace)) == 2
    assert not (transport.root / f"106-assembler{ADOPTED_SUFFIX}").exists()


def test_an_unanswered_turn_is_still_waited_on(tmp_path):
    """The in-flight path is unchanged: the dead process's request is answered, not re-asked."""
    workflow, transport = _relaunched(tmp_path)
    workspace = workflow.workspace
    old = _dead_turn(workspace, "066-assembler", answered=False, decision=None)
    _graph(workspace, core_tp="running")

    def answer():
        while not (transport.root / f"066-assembler{ADOPTED_SUFFIX}").exists():
            time.sleep(0.005)
        old.write_text("decision: UNIT_READY\nsummary: late\n", encoding="utf-8")
        (transport.root / "066-assembler.done").write_text("1\n", encoding="utf-8")

    threading.Thread(target=answer, daemon=True).start()
    decision = workflow._dispatch_task("assembler", "core", "core-tp", "Assemble core-tp.")

    assert decision == "UNIT_READY"
    assert _requests(workspace) == ["066-assembler"]
    record = json.loads((transport.root / f"066-assembler{ADOPTED_SUFFIX}").read_text())
    assert record["answered"] is False


def test_a_non_task_turn_is_never_adopted(tmp_path):
    """A planner's answer is to a standing that has moved on; it is asked again."""
    workflow, transport = _relaunched(tmp_path)
    workspace = workflow.workspace
    dispatch = workspace / DISPATCH_DIRNAME
    dispatch.mkdir(parents=True, exist_ok=True)
    stale = mark_decision_dispatch(workspace, "planner", 1)
    stale.write_text("decision: DONE\n", encoding="utf-8")
    (dispatch / "125-planner.request.json").write_text(
        json.dumps(
            {
                "id": "125-planner",
                "role": "planner",
                "what": "planner",
                "decision_file": str(stale),
                "requested_at": time.time() - 600,
            }
        ),
        encoding="utf-8",
    )
    (dispatch / "125-planner.done").write_text("1\n", encoding="utf-8")
    # The graph it proposed, set aside by the next planner turn rather than deleted.
    kept = workspace / "graph.proposed.previous.yaml"
    kept.write_text("tasks: []\n", encoding="utf-8")
    _graph(workspace, core_tp="running")
    stop = _drive(transport, decision="GRAPH_UPDATED")
    try:
        decision = workflow._dispatch_task("planner", "plan", "planner", "Plan.")
    finally:
        stop.set()

    assert decision == "GRAPH_UPDATED"
    assert not (dispatch / f"125-planner{ADOPTED_SUFFIX}").exists()
    # Its work is handed to the turn asked in its place, once.
    (asked,) = [r for r in _requests(workspace) if r != "125-planner"]
    prompt = (dispatch / f"{asked}.prompt.md").read_text(encoding="utf-8")
    assert "`125-planner` answered this same work" in prompt and str(stale) in prompt
    assert f"what it proposed at `{kept}`" in prompt
    assert (dispatch / f"125-planner{READ_SUFFIX}").exists()


def test_a_non_task_answer_some_process_read_is_not_handed_on(tmp_path):
    """An answer the previous process read was acted on; the next turn is asked plainly."""
    workflow, transport = _relaunched(tmp_path)
    workspace = workflow.workspace
    _dead_turn(workspace, "125-planner", what="planner", unit="plan", read_by_pid=1)
    _graph(workspace, core_tp="running")
    stop = _drive(transport, decision="GRAPH_UPDATED")
    try:
        assert workflow._dispatch_task("planner", "plan", "planner", "Plan.") == "GRAPH_UPDATED"
    finally:
        stop.set()
    dispatch = workspace / DISPATCH_DIRNAME
    (asked,) = [r for r in _requests(workspace) if r != "125-planner"]
    assert "never read" not in (dispatch / f"{asked}.prompt.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------ the driver


def test_driver_status_shows_an_adopted_turn(tmp_path):
    _dead_turn(tmp_path, "066-assembler")
    _dead_turn(tmp_path, "067-assembler", what="core-dp", turn=6)
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    transport.adopt_answered("assembler", "core-tp", _assembler_decision)
    transport.consume(["core-dp"])

    data = driver.status(driver.load_config(tmp_path))
    rows = {row["id"]: row for row in data["requests"]}
    assert rows["066-assembler"]["adopted"]["answered"] is True
    assert rows["066-assembler"]["read"] and not rows["066-assembler"]["consumed"]
    assert rows["067-assembler"]["adopted"] is None and rows["067-assembler"]["consumed"]
    lines = driver.format_status(data).splitlines()
    assert any(line.startswith("066-assembler") and "ADOPTED" in line for line in lines)
    assert not any(line.startswith("067-assembler") and "ADOPTED" in line for line in lines)


def test_a_finished_task_has_its_answered_turns_consumed_at_once(tmp_path):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import PlannedStage

    _dead_turn(tmp_path, "066-assembler")
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    task = graph_mod.Task(id="core-tp", kind="assemble", goal="g")
    PlannedStage._task_finished(SimpleNamespace(_turns=transport), task)

    assert (transport.root / f"066-assembler{CONSUMED_SUFFIX}").is_file()
    resumed = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert resumed.adopt_answered("assembler", "core-tp", _assembler_decision) is None


def _orphan(transport: ExternalAgentTransport, request_id: str, work: str, decision: Path) -> None:
    """An unanswered smith request a dead process left for ``work``."""
    transport.root.mkdir(parents=True, exist_ok=True)
    (transport.root / f"{request_id}.request.json").write_text(
        json.dumps(
            {
                "id": request_id,
                "role": "smith",
                "what": "verify attention/thop_attention",
                "work": work,
                "decision_file": str(decision),
                "requested_at": time.time() - 600,
            }
        ),
        encoding="utf-8",
    )


def test_two_tasks_asked_alike_each_adopt_only_their_own_orphaned_turn(tmp_path):
    """Two verify tasks on one entry share a ``what``; a turn and its decision stay with its task."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    k1d_old = workspace / "decisions" / "core-smith-1.yaml"
    k2b_old = workspace / "decisions" / "core-smith-2.yaml"
    transport = ExternalAgentTransport(workspace, poll_s=0.01, timeout_s=30)
    _orphan(transport, "090-smith", "k1d", k1d_old)
    _orphan(transport, "091-smith", "k2b", k2b_old)

    def answer(request_id: str, decision: Path, text: str) -> None:
        while not (transport.root / f"{request_id}{ADOPTED_SUFFIX}").exists():
            time.sleep(0.005)
        decision.parent.mkdir(parents=True, exist_ok=True)
        decision.write_text(f"decision: LANDED\nsummary: {text}\n", encoding="utf-8")
        (transport.root / f"{request_id}.result.md").write_text(text, encoding="utf-8")
        (transport.root / f"{request_id}.done").write_text("1\n", encoding="utf-8")

    k2b_new = workspace / "decisions" / "core-smith-2.2.yaml"
    k1d_new = workspace / "decisions" / "core-smith-1.2.yaml"
    for request_id, decision, text in (
        ("091-smith", k2b_old, "k2b"),
        ("090-smith", k1d_old, "k1d"),
    ):
        threading.Thread(target=answer, args=(request_id, decision, text), daemon=True).start()
    # k2b asks first, as after the run-4 reload: it must not take k1d's turn.
    k2b = transport.run(
        TurnRequest(
            role="smith",
            what="verify attention/thop_attention",
            prompt="p",
            decision_file=str(k2b_new),
            work="k2b",
        )
    )
    k1d = transport.run(
        TurnRequest(
            role="smith",
            what="verify attention/thop_attention",
            prompt="p",
            decision_file=str(k1d_new),
            work="k1d",
        )
    )

    assert (k2b.request_id, k2b.text) == ("091-smith", "k2b")
    assert (k1d.request_id, k1d.text) == ("090-smith", "k1d")
    assert "k2b" in k2b_new.read_text(encoding="utf-8")
    assert "k1d" in k1d_new.read_text(encoding="utf-8")
    assert _requests(workspace) == ["090-smith", "091-smith"], "nothing asked twice"


def test_a_planner_turn_sets_the_last_proposed_graph_aside_instead_of_deleting_it(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.stages import planned

    proposed = tmp_path / planned.PROPOSED_GRAPH
    planned._set_aside(proposed)
    assert not planned.previous_proposal(proposed).exists()
    proposed.write_text("tasks: []\n", encoding="utf-8")
    planned._set_aside(proposed)
    assert not proposed.exists()
    assert planned.previous_proposal(proposed).read_text(encoding="utf-8") == "tasks: []\n"
