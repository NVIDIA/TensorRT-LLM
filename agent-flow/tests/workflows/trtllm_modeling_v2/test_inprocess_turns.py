# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What an in-process turn leaves for an operator, and the bound it runs under.

An external turn is files an operator can read while it runs; an in-process
turn used to be a call with nothing on disk until the journal's report, and no
deadline. These hold it to the same: a record per turn written as it streams,
a listing of the turns not yet ended, and the role's timeout, past which the
turn is stopped and raises what an unanswered external turn raises.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import anyio
import pytest

from agent_flow import AgentLayer, AgentLayerConfig, BackendConfig
from agent_flow.backends import claude_code
from agent_flow.backends.base import Backend, BackendClient
from agent_flow.logger import get_logger
from agent_flow.types import AgentTextEvent, ThinkingEvent, ToolCallEvent
from agent_flow.workflows.trtllm_modeling_v2.infra.driver import core as driver_core
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import (
    DEFAULT_ROLE_TURN_TIMEOUTS,
    TURNS_DIRNAME,
    InProcessTurnExecutor,
    TurnRequest,
    TurnTimeout,
    select_turn_executor,
    turns_in_flight,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.transport import (
    DISPATCH_DIRNAME,
    ORCHESTRATOR_EXIT,
    ORCHESTRATOR_HEARTBEAT,
    ExternalAgentTransport,
    start_orchestrator_heartbeat,
)
from agent_flow.workflows.trtllm_modeling_v2.run import drain as drain_mod
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from ...helpers import FakeBackend
from .test_workflow import _workflow


@dataclasses.dataclass(frozen=True)
class _Session:
    mode: str = "stateless"


@dataclasses.dataclass(frozen=True)
class _Config:
    """The parts of ``AgentLayerConfig`` the executor touches."""

    session: _Session = _Session()
    on_activity: Any = None


class StreamingLayer:
    """An ``AgentLayer`` as the executor sees it: a config with an observer, and ``aforward``.

    Each turn streams a tool call, a thinking block and its text through the
    config's observer, as ``AgentLayer._execute`` does, and writes ``stderr``
    as the CLI's stderr reader would. ``hold`` keeps a turn open until set.
    """

    def __init__(self, *, stderr: str = "", hold: threading.Event | None = None) -> None:
        self.config = _Config()
        self.stderr = stderr
        self.hold = hold
        self.cancelled = False

    def _observe(self, kind: str, event: Any) -> None:
        if self.config.on_activity is not None:
            self.config.on_activity(kind, event)

    async def aforward(self, prompt: str) -> str:
        self._observe("tool", ToolCallEvent(name="Bash", input={"command": f"echo {prompt}"}))
        self._observe("thinking", ThinkingEvent(text="first line\nsecond line"))
        if self.stderr:
            claude_code._forward_stderr(self.stderr)
        try:
            while self.hold is not None and not self.hold.is_set():
                await anyio.sleep(0.01)
        except anyio.get_cancelled_exc_class():
            self.cancelled = True
            raise
        self._observe("text", AgentTextEvent(text=f"done with {prompt}"))
        return f"answer to {prompt}"

    def __call__(self, prompt: str) -> str:
        return anyio.run(self.aforward, prompt)


def _request(role: str = "smith", what: str = "k12", prompt: str = "go") -> TurnRequest:
    return TurnRequest(
        role=role,
        what=what,
        prompt=prompt,
        system_prompt="ROLE PROMPT",
        decision_file="decisions/core-smith-1.yaml",
        model="m",
    )


def _wait_for(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "the condition never held"
        time.sleep(0.005)


# ------------------------------------------------------------------ the record


def test_a_turn_leaves_its_prompt_activity_and_answer_as_it_goes(tmp_path):
    root = tmp_path / TURNS_DIRNAME
    layer = StreamingLayer(stderr="claude: a warning on stderr")
    outcome = InProcessTurnExecutor(lambda r: layer, records=root).run(_request())

    assert outcome.request_id == "001-smith" and outcome.text == "answer to go"
    request = json.loads((root / "001-smith.request.json").read_text())
    assert (request["role"], request["what"]) == ("smith", "k12")
    assert request["decision_file"] == "decisions/core-smith-1.yaml"
    assert request["agent"]["model"] == "m"
    assert request["timeout_s"] == DEFAULT_ROLE_TURN_TIMEOUTS["smith"]
    assert request["activity_file"] == str(root / "001-smith.activity.log")
    assert (root / "001-smith.system.md").read_text() == "ROLE PROMPT"
    assert (root / "001-smith.prompt.md").read_text() == "go"
    assert (root / "001-smith.result.md").read_text() == "answer to go"

    activity = (root / "001-smith.activity.log").read_text()
    assert 'tool: Bash {"command": "echo go"}' in activity
    # A multi-line block keeps its lines, indented under its entry.
    assert "thinking: first line\n    second line\n" in activity
    assert "stderr: claude: a warning on stderr" in activity
    assert "text: done with go" in activity
    done = json.loads((root / "001-smith.done").read_text())
    assert done["outcome"] == "answered" and done["error"] == ""


def test_the_layers_own_observer_still_sees_every_event(tmp_path):
    seen: list[str] = []
    layer = StreamingLayer()
    layer.config = dataclasses.replace(
        layer.config, on_activity=lambda kind, event: seen.append(kind)
    )
    executor = InProcessTurnExecutor(lambda r: layer, records=tmp_path)
    executor.run(_request())
    executor.run(_request(what="k13"))
    assert seen == ["tool", "thinking", "text"] * 2
    assert "tool: Bash" in (tmp_path / "002-smith.activity.log").read_text()


def test_concurrent_turns_of_one_layer_each_stream_into_their_own_record(tmp_path):
    """Planned tasks of one role share its layer; each turn's events are its own."""
    gate = threading.Event()
    layer = StreamingLayer(hold=gate)
    executor = InProcessTurnExecutor(lambda r: layer, records=tmp_path)
    threads = [
        threading.Thread(target=executor.run, args=(_request(what=w, prompt=w),), daemon=True)
        for w in ("alpha", "beta")
    ]
    for thread in threads:
        thread.start()
    _wait_for(lambda: len(turns_in_flight(tmp_path)) == 2)
    gate.set()
    for thread in threads:
        thread.join(timeout=5)

    for record in tmp_path.glob("*.request.json"):
        what = json.loads(record.read_text())["what"]
        activity = record.with_name(record.name.replace(".request.json", ".activity.log"))
        text = activity.read_text()
        other = "beta" if what == "alpha" else "alpha"
        assert f"echo {what}" in text and f"done with {what}" in text
        assert other not in text


def test_a_bound_view_numbers_its_turns_with_the_executor(tmp_path):
    executor = InProcessTurnExecutor(lambda r: StreamingLayer(), records=tmp_path)
    executor.run(_request())
    executor.bound(StreamingLayer()).run(_request(role="smith", what="batch"))
    executor.run(_request(role="planner", what="plan"))
    assert sorted(p.name for p in tmp_path.glob("*.request.json")) == [
        "001-smith.request.json",
        "002-smith.request.json",
        "003-planner.request.json",
    ]


def test_ids_continue_across_a_resume(tmp_path):
    InProcessTurnExecutor(lambda r: StreamingLayer(), records=tmp_path).run(_request())
    resumed = InProcessTurnExecutor(lambda r: StreamingLayer())
    resumed.keep_records(tmp_path)
    assert resumed.run(_request()).request_id == "002-smith"


def test_without_a_records_directory_a_turn_writes_nothing(tmp_path):
    outcome = InProcessTurnExecutor(lambda r: StreamingLayer()).run(_request())
    assert outcome.text == "answer to go" and outcome.request_id == ""
    assert not list(tmp_path.iterdir())


def test_a_turn_that_raises_says_so_in_its_record(tmp_path):
    def failing(prompt: str) -> str:
        raise RuntimeError("Claude Code turn failed: unknown")

    with pytest.raises(RuntimeError, match="turn failed"):
        InProcessTurnExecutor(lambda r: failing, records=tmp_path).run(_request())
    done = json.loads((tmp_path / "001-smith.done").read_text())
    assert done["outcome"] == "raised"
    assert done["error"] == "RuntimeError: Claude Code turn failed: unknown"
    assert not (tmp_path / "001-smith.result.md").exists()


# ------------------------------------------------------------------ in flight


def test_turns_in_flight_lists_a_running_turn_with_its_role_task_and_age(tmp_path):
    gate = threading.Event()
    executor = InProcessTurnExecutor(lambda r: StreamingLayer(hold=gate), records=tmp_path)
    thread = threading.Thread(target=executor.run, args=(_request(),), daemon=True)
    thread.start()
    _wait_for(lambda: turns_in_flight(tmp_path))
    [row] = turns_in_flight(tmp_path, now=time.time() + 60)
    assert (row["id"], row["role"], row["what"]) == ("001-smith", "smith", "k12")
    assert row["age_s"] >= 60 and row["alive"] is True and not row["overdue"]
    assert turns_in_flight(tmp_path, now=time.time() + 7 * 3600)[0]["overdue"]
    gate.set()
    thread.join(timeout=5)
    assert turns_in_flight(tmp_path) == []


def test_a_turn_whose_process_is_gone_is_listed_as_not_alive(tmp_path):
    executor = InProcessTurnExecutor(lambda r: StreamingLayer(), records=tmp_path)
    executor.run(_request())
    (tmp_path / "001-smith.done").unlink()
    request = json.loads((tmp_path / "001-smith.request.json").read_text())
    request["pid"] = 2**22 + 12345
    (tmp_path / "001-smith.request.json").write_text(json.dumps(request))
    assert turns_in_flight(tmp_path)[0]["alive"] is False
    request["host"] = "another-host"
    (tmp_path / "001-smith.request.json").write_text(json.dumps(request))
    assert turns_in_flight(tmp_path)[0]["alive"] is None


def test_driver_status_lists_an_in_process_turn_while_it_runs(tmp_path, task_file, monkeypatch):
    """The orchestrator's ledger names the turn; ``driver status`` shows it with its age."""
    seen: dict[str, Any] = {}
    workspace = tmp_path / "ws"

    def observing(prompt: str) -> str:
        cfg = driver_core.load_config(workspace)
        seen["status"] = driver_core.format_status(driver_core.status(cfg))
        seen["turns"] = turns_in_flight(workspace / TURNS_DIRNAME)
        return "done"

    monkeypatch.setattr(
        TrtllmModelingV2Workflow, "_make_agent", lambda self, name, prompt, **kw: observing
    )
    workflow = _workflow(tmp_path)
    workflow._init_state(task_file, get_logger().console)
    workflow._turns.keep_records(workspace / TURNS_DIRNAME)
    workflow._inflight = drain_mod.InFlight(workspace)
    assert workflow._run_turn("smith", "k12", "certify it") == "done"

    assert "orchestrator holds: turn smith k12 0m" in seen["status"]
    [row] = seen["turns"]
    assert (row["role"], row["what"]) == ("smith", "k12")
    assert row["prompt_file"].endswith("001-smith.prompt.md")
    assert turns_in_flight(workspace / TURNS_DIRNAME) == []


# ------------------------------------------------------------------ the bound


def test_every_role_has_its_bound_in_process_too():
    executor = InProcessTurnExecutor(lambda r: None)
    assert executor.timeout_for("assembler") == DEFAULT_ROLE_TURN_TIMEOUTS["assembler"]
    assert executor.timeout_for("planner") == DEFAULT_ROLE_TURN_TIMEOUTS["planner"]
    # One bound for every role, as `--agent-turn-timeout` gives the external transport.
    one = InProcessTurnExecutor(lambda r: None, timeout_s=600)
    assert {one.timeout_for(role) for role in DEFAULT_ROLE_TURN_TIMEOUTS} == {600}
    chosen = select_turn_executor(None, lambda r: None, timeout_s=60, records=None)
    assert isinstance(chosen, InProcessTurnExecutor) and chosen.timeout_for("smith") == 60


def test_a_turn_past_its_bound_is_stopped_and_raises_what_an_unanswered_turn_raises(tmp_path):
    layer = StreamingLayer(hold=threading.Event())
    executor = InProcessTurnExecutor(
        lambda r: layer, role_timeouts={"smith": 0.2}, records=tmp_path
    )
    started = time.monotonic()
    with pytest.raises(TurnTimeout, match=r"001-smith: no answer within 0.2s \(the smith turn"):
        executor.run(_request())
    assert time.monotonic() - started < 5
    # Cancelled, not abandoned: the CLI an AgentLayer started is stopped with it.
    assert layer.cancelled
    done = json.loads((tmp_path / "001-smith.done").read_text())
    assert done["outcome"] == "timed out" and "001-smith.activity.log" in done["error"]


def test_a_timed_out_task_turn_is_retried_as_an_infrastructure_failure(tmp_path):
    """The orchestrator reads a raised turn by its text, for either transport alike."""
    # Imported here: importing the orchestrator first walks into the stages' import cycle.
    from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import _not_the_task

    executor = InProcessTurnExecutor(
        lambda r: StreamingLayer(hold=threading.Event()), timeout_s=0.05
    )
    with pytest.raises(TurnTimeout) as raised:
        executor.run(_request())
    assert _not_the_task(f"raised {type(raised.value).__name__}: {raised.value}")


def test_a_plain_callable_runs_unbounded_as_before():
    """A layer with no ``aforward`` cannot be cancelled, so it is not raced against a clock."""
    calls: list[str] = []

    def plain(prompt: str) -> str:
        time.sleep(0.1)
        calls.append(prompt)
        return "ok"

    executor = InProcessTurnExecutor(lambda r: plain, timeout_s=0.01)
    assert executor.run(_request()).text == "ok" and calls == ["go"]


def test_a_persistent_session_is_called_as_it_is():
    """Its session lives on a loop of its own, which a turn's loop must not borrow."""
    calls: list[str] = []

    class Persistent(StreamingLayer):
        def __call__(self, prompt: str) -> str:
            calls.append(prompt)
            return "persistent"

    layer = Persistent()
    layer.config = dataclasses.replace(layer.config, session=_Session("persistent"))
    assert InProcessTurnExecutor(lambda r: layer).run(_request()).text == "persistent"
    assert calls == ["go"]


# ------------------------------------------------------------ a real AgentLayer


def _agent_layer() -> AgentLayer:
    return AgentLayer(
        AgentLayerConfig(
            name="smith",
            backend=BackendConfig(kind="claude-code", model="m"),
            print_activity=False,
        )
    )


def test_an_agent_layers_events_reach_the_record(tmp_path):
    backend = FakeBackend(
        [{"text": "landed", "tool_calls": [ToolCallEvent(name="Bash", input={"command": "ls"})]}]
    )
    layer = _agent_layer()
    with patch("agent_flow.layers.create_backend", return_value=backend):
        outcome = InProcessTurnExecutor(lambda r: layer, records=tmp_path).run(_request())
    assert outcome.text == "landed"
    assert 'tool: Bash {"command": "ls"}' in (tmp_path / "001-smith.activity.log").read_text()
    assert (tmp_path / "001-smith.result.md").read_text() == "landed"


class _StalledClient(BackendClient):
    async def send_message(self, message: str):
        await anyio.sleep(3600)
        yield  # pragma: no cover - never reached


class _StalledBackend(Backend):
    """A CLI that never answers; its client is closed only when the turn is stopped."""

    def __init__(self) -> None:
        self.closed = False

    @asynccontextmanager
    async def create_client(self, **options):
        try:
            yield _StalledClient()
        finally:
            self.closed = True


def test_an_agent_layer_past_its_bound_has_its_client_closed(tmp_path):
    backend = _StalledBackend()
    layer = _agent_layer()
    executor = InProcessTurnExecutor(lambda r: layer, timeout_s=0.2, records=tmp_path)
    with patch("agent_flow.layers.create_backend", return_value=backend):
        with pytest.raises(TurnTimeout, match="the smith turn timeout"):
            executor.run(_request())
    # The client's exit is where the Claude backend stops its CLI.
    assert backend.closed


# ------------------------------------------------------- the orchestrator's heartbeat


def _beat(workspace) -> dict:
    return json.loads((workspace / DISPATCH_DIRNAME / ORCHESTRATOR_HEARTBEAT).read_text())


def test_the_heartbeat_names_this_process_and_keeps_beating(tmp_path):
    gone = tmp_path / DISPATCH_DIRNAME / ORCHESTRATOR_EXIT
    gone.parent.mkdir(parents=True)
    gone.write_text("{}\n")  # the previous process's exit note
    heartbeat = start_orchestrator_heartbeat(tmp_path, 0.02, started=123.0)
    try:
        _wait_for((tmp_path / DISPATCH_DIRNAME / ORCHESTRATOR_HEARTBEAT).exists)
        first = _beat(tmp_path)
        assert {k: first[k] for k in ("pid", "host", "started")} == {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started": 123.0,
        }
        assert not gone.exists()
        _wait_for(lambda: _beat(tmp_path)["at"] > first["at"])
    finally:
        heartbeat.stop()


def test_an_in_process_orchestrator_that_exits_leaves_its_exit_note(tmp_path):
    """What a driver on any host reads as ORCHESTRATOR_EXITED."""
    script = (
        "import sys, time\n"
        "from agent_flow.workflows.trtllm_modeling_v2.infra.transport import "
        "start_orchestrator_heartbeat\n"
        "start_orchestrator_heartbeat(sys.argv[1], 0.02)\n"
        "time.sleep(0.2)\n"
    )
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert child.returncode == 0, child.stderr
    gone = json.loads((tmp_path / DISPATCH_DIRNAME / ORCHESTRATOR_EXIT).read_text())
    beat = _beat(tmp_path)
    assert gone["pid"] == beat["pid"] != os.getpid()
    assert gone["started"] == beat["started"] and gone["at"] >= beat["at"]
    (tmp_path / "task.yaml").write_text("{}\n")
    state, detail = driver_core.orchestrator_state(driver_core.load_config(tmp_path))
    assert state == "exited" and f"pid {gone['pid']}" in detail


def test_the_external_transport_beats_through_the_same_helper(tmp_path):
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01, heartbeat_s=0.02)
    try:
        _wait_for((tmp_path / DISPATCH_DIRNAME / ORCHESTRATOR_HEARTBEAT).exists)
        assert _beat(tmp_path)["started"] == transport._created
    finally:
        transport.close()


def test_the_cli_starts_the_heartbeat_for_an_in_process_run(tmp_path, task_file, monkeypatch):
    from agent_flow.workflows.trtllm_modeling_v2 import cli as cli_mod
    from agent_flow.workflows.trtllm_modeling_v2.infra.committer import (
        ENV_EMAIL,
        ENV_NAME,
        CommitIdentity,
    )
    from agent_flow.workflows.trtllm_modeling_v2.run import workflow as workflow_mod

    started: list[tuple] = []
    ran: list[str] = []

    class Workflow:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def run(self, task) -> None:
            ran.append(str(task))

    for name in (ENV_NAME, ENV_EMAIL):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(cli_mod, "_HELD_LOCKS", [])
    monkeypatch.setattr(cli_mod, "_commit_identity", lambda task: CommitIdentity("t", "t@x"))
    monkeypatch.setattr(cli_mod, "start_orchestrator_heartbeat", lambda *args: started.append(args))
    monkeypatch.setattr(workflow_mod, "TrtllmModelingV2Workflow", Workflow)
    workspace = tmp_path / "ws"
    cli_mod.main(["--task", str(task_file), "--workspace", str(workspace), "--dry-run"])
    assert ran and started == [(workspace, cli_mod.ORCHESTRATOR_HEARTBEAT_S)]
