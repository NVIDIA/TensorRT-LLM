# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What an in-process turn leaves for an operator, and the bound it runs under."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from typing import Any

import anyio
import pytest

from agent_flow.backends import claude_code
from agent_flow.types import AgentTextEvent, SessionInitEvent, ThinkingEvent, ToolCallEvent
from agent_flow.workflows.trtllm_modeling_v2.agents.turns import (
    DEFAULT_ROLE_TURN_TIMEOUTS,
    TURNS_DIRNAME,
    InProcessTurnExecutor,
    TurnRequest,
    TurnTimeout,
)


@dataclasses.dataclass(frozen=True)
class _Config:
    """The parts of ``AgentLayerConfig`` the executor touches."""

    on_activity: Any = None


class StreamingLayer:
    """An ``AgentLayer`` as the executor sees it: a config with an observer, and ``aforward``.

    Each turn streams its session, a tool call, a thinking block and its text
    through the config's observer, as ``AgentLayer._execute`` does, and writes
    ``stderr`` as the CLI's stderr reader would. ``hold`` keeps a turn open until set.
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
        session = SessionInitEvent(
            backend="codex", model="resolved-model", reasoning_effort="medium", session_id="s1"
        )
        self._observe("session", session)
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


def _request(role: str = "kernel_writer", what: str = "k12", prompt: str = "go") -> TurnRequest:
    return TurnRequest(
        role=role,
        what=what,
        prompt=prompt,
        system_prompt="ROLE PROMPT",
        decision_file="decisions/core-kernel_writer-1.yaml",
        model="m",
    )


def test_a_turn_leaves_its_prompt_activity_and_answer_as_it_goes(tmp_path):
    root = tmp_path / TURNS_DIRNAME
    layer = StreamingLayer(stderr="claude: a warning on stderr")
    outcome = InProcessTurnExecutor(lambda r: layer, records=root).run(_request())

    assert outcome.request_id == "001-kernel_writer" and outcome.text == "answer to go"
    request = json.loads((root / "001-kernel_writer.request.json").read_text())
    assert (request["role"], request["what"]) == ("kernel_writer", "k12")
    assert request["decision_file"] == "decisions/core-kernel_writer-1.yaml"
    assert request["agent"]["model"] == "m"
    assert request["timeout_s"] == DEFAULT_ROLE_TURN_TIMEOUTS["kernel_writer"]
    assert request["activity_file"] == str(root / "001-kernel_writer.activity.log")
    assert (root / "001-kernel_writer.system.md").read_text() == "ROLE PROMPT"
    assert (root / "001-kernel_writer.prompt.md").read_text() == "go"
    assert (root / "001-kernel_writer.result.md").read_text() == "answer to go"

    activity = (root / "001-kernel_writer.activity.log").read_text()
    assert 'tool: Bash {"command": "echo go"}' in activity
    # A multi-line block keeps its lines, indented under its entry.
    assert "thinking: first line\n    second line\n" in activity
    assert "stderr: claude: a warning on stderr" in activity
    assert "text: done with go" in activity
    done = json.loads((root / "001-kernel_writer.done").read_text())
    assert done["outcome"] == "answered" and done["error"] == ""
    # The effective session configuration is retained in the terminal record.
    assert done["sessions"][0]["model"] == "resolved-model"
    assert done["sessions"][0]["reasoning_effort"] == "medium"


def test_a_turn_past_its_bound_is_stopped_and_retried_as_an_infrastructure_failure(tmp_path):
    # Imported here: importing the orchestrator first walks into the stages' import cycle.
    from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import _not_the_task

    layer = StreamingLayer(hold=threading.Event())
    executor = InProcessTurnExecutor(lambda r: layer, timeout_s=0.2, records=tmp_path)
    started = time.monotonic()
    with pytest.raises(
        TurnTimeout, match=r"001-kernel_writer: no answer within 0.2s \(the kernel_writer turn"
    ) as raised:
        executor.run(_request())
    assert time.monotonic() - started < 5
    # Cancelled, not abandoned: the CLI an AgentLayer started is stopped with it.
    assert layer.cancelled
    done = json.loads((tmp_path / "001-kernel_writer.done").read_text())
    assert done["outcome"] == "timed out" and "001-kernel_writer.activity.log" in done["error"]
    # What an unanswered turn raises: the orchestrator retries it as the run's failure.
    assert _not_the_task(f"raised {type(raised.value).__name__}: {raised.value}")
