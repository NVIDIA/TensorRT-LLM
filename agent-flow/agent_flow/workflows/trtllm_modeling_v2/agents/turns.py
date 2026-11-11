# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One agent turn, as a request and an outcome, and the executor that runs it here.

A *turn* is one round trip with one role: the orchestrator hands over a system
prompt and a task, the role does its work, and a text answer comes back (its
typed decision arrives separately, through the decision file or tool). A turn
is bounded by its role's timeout (:data:`DEFAULT_ROLE_TURN_TIMEOUTS`) and raises
:class:`TurnTimeout` past it.

A turn leaves its evidence in the directory the run names
(:meth:`InProcessTurnExecutor.keep_records`, ``<workspace>/turns/``), each file
written as the turn goes, so an operator can tail one while it runs::

    <id>.request.json   role, task, agent, process, start, bound, and the files below
    <id>.system.md      the role's system prompt
    <id>.prompt.md      the task prompt
    <id>.activity.log   each tool call, text and thinking block as it streams, and the CLI's stderr
    <id>.result.md      the answer
    <id>.done           how it ended (answered, raised, timed out, interrupted); written last
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import socket
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import anyio

from agent_flow.backends.claude_code import cli_stderr_to

from ..infra.files import write_json_atomic, write_text_atomic


class TurnTimeout(RuntimeError):
    """Nothing answered the turn inside its deadline."""


#: How long a turn may take when neither the caller nor the role says.
DEFAULT_TURN_TIMEOUT_S = 3 * 60 * 60

#: Per-role turn timeouts, in seconds.
DEFAULT_ROLE_TURN_TIMEOUTS: dict[str, float] = {
    "surveyor": 4 * 3600.0,
    "kernel_writer": 6 * 3600.0,
    "assembler": 16 * 3600.0,
}

#: Where in-process turns leave their files, under the workspace.
TURNS_DIRNAME = "turns"

#: One activity entry's text is cut at this many characters.
ACTIVITY_ENTRY_LIMIT = 4000


@dataclass(frozen=True)
class TurnRequest:
    """What one role is being asked to do.

    ``decision_file`` is the exact path the role must write its typed decision
    to. ``backend``, ``model`` and ``reasoning_effort`` are the agent the role
    resolves to (``agents/roles.py``), kept in the turn's request record.
    """

    role: str
    what: str
    prompt: str
    system_prompt: str = ""
    decision_file: str | None = None
    backend: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class TurnOutcome:
    """The role's prose answer. **Not a control-plane input**: decisions are typed."""

    role: str
    text: str


#: Given a request, the :class:`agent_flow.AgentLayer` that runs it: its
#: ``aforward(prompt)`` answers.
LayerProvider = Callable[[TurnRequest], Any]


def _now_text(epoch: float | None = None) -> str:
    moment = datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc)
    return moment.isoformat(timespec="seconds")


def _process() -> dict:
    return {"pid": os.getpid(), "host": socket.gethostname()}


def _seq(name: str) -> int:
    head = name.split("-", 1)[0]
    return int(head) if head.isdigit() else 0


def _describe(event: Any) -> str:
    """One streamed event as text: a tool call's name and input, a block's text, or its fields."""
    name, payload = getattr(event, "name", None), getattr(event, "input", None)
    if isinstance(name, str) and payload is not None:
        return f"{name} {json.dumps(payload, ensure_ascii=False, default=str)}"
    text = getattr(event, "text", None)
    if isinstance(text, str):
        return text
    if dataclasses.is_dataclass(event) and not isinstance(event, type):
        return json.dumps(dataclasses.asdict(event), ensure_ascii=False, default=str)
    return str(event)


class TurnRecord:
    """The files one in-process turn leaves, written as it goes.

    Bookkeeping only: a write that fails (a full filesystem) never fails the
    turn it describes.
    """

    def __init__(self, root: Path, request_id: str) -> None:
        self.request_id = request_id
        self.request_file = root / f"{request_id}.request.json"
        self.system_file = root / f"{request_id}.system.md"
        self.prompt_file = root / f"{request_id}.prompt.md"
        self.activity_file = root / f"{request_id}.activity.log"
        self.result_file = root / f"{request_id}.result.md"
        self.done_file = root / f"{request_id}.done"
        self.started = time.time()
        self._lock = threading.Lock()
        self.sessions: list[dict] = []

    def begin(self, request: TurnRequest, timeout_s: float) -> None:
        """The request, its prompts and the empty activity log, before the turn starts."""
        try:
            write_text_atomic(self.system_file, request.system_prompt)
            write_text_atomic(self.prompt_file, request.prompt)
            self.activity_file.touch()
            write_json_atomic(
                self.request_file,
                {
                    "id": self.request_id,
                    "role": request.role,
                    "what": request.what,
                    "decision_file": request.decision_file,
                    "agent": {
                        "backend": request.backend,
                        "model": request.model,
                        "reasoning_effort": request.reasoning_effort,
                    },
                    **_process(),
                    "started_at": self.started,
                    "timeout_s": timeout_s,
                    "system_prompt_file": str(self.system_file),
                    "prompt_file": str(self.prompt_file),
                    "activity_file": str(self.activity_file),
                    "result_file": str(self.result_file),
                    "done_file": str(self.done_file),
                },
            )
        except OSError:
            pass

    def note(self, kind: str, text: str) -> None:
        """Append one entry to the activity log: its time and kind, then its text, indented."""
        if len(text) > ACTIVITY_ENTRY_LIMIT:
            cut = len(text) - ACTIVITY_ENTRY_LIMIT
            text = f"{text[:ACTIVITY_ENTRY_LIMIT]} ... ({cut} more characters)"
        first, *rest = text.splitlines() or [""]
        entry = "".join(
            [f"{_now_text()[11:19]} {kind}: {first}\n", *(f"    {line}\n" for line in rest)]
        )
        with self._lock:
            try:
                with self.activity_file.open("a", encoding="utf-8") as handle:
                    handle.write(entry)
            except OSError:
                pass

    def activity(self, kind: str, event: Any) -> None:
        """One event the layer streamed; a subagent's carries its label."""
        if kind == "session" and dataclasses.is_dataclass(event):
            with self._lock:
                self.sessions.append(dataclasses.asdict(event))
        label = getattr(event, "agent_label", None)
        self.note(f"{kind} [{label}]" if label else kind, _describe(event))

    def finish(self, outcome: str, *, text: str | None = None, error: str = "") -> None:
        """The answer, when there is one, then ``.done``: how the turn ended."""
        try:
            if text is not None:
                write_text_atomic(self.result_file, text)
            ended = time.time()
            write_json_atomic(
                self.done_file,
                {
                    "outcome": outcome,
                    "error": error,
                    "ended_at": ended,
                    "elapsed_s": round(ended - self.started, 1),
                    "sessions": self.sessions,
                },
            )
        except OSError:
            pass


#: The record of the turn running in this context. A layer called in a turn's
#: thread streams its events in that thread's context, so each event reaches
#: its own turn's record however many turns of one layer run at once.
_ACTIVE: ContextVar[TurnRecord | None] = ContextVar("in_process_turn_record", default=None)


class _ActivityRouter:
    """A layer's ``on_activity``: each event to the record of the turn it belongs to, then on."""

    def __init__(self, previous: Callable[[str, Any], None] | None) -> None:
        self.previous = previous

    def __call__(self, kind: str, event: Any) -> None:
        record = _ACTIVE.get()
        if record is not None:
            record.activity(kind, event)
        if self.previous is not None:
            self.previous(kind, event)


_ROUTER_LOCK = threading.Lock()


def _route_activity(layer: Any) -> None:
    """Give ``layer`` the observer that writes its events to the running turn's record, once.

    A layer is configured by whoever built it, and its observer is part of
    that configuration; one it already had still sees every event, after the
    record. A layer with no dataclass ``config.on_activity`` streams nothing
    here.
    """
    with _ROUTER_LOCK:
        config = getattr(layer, "config", None)
        if (
            config is None
            or not dataclasses.is_dataclass(config)
            or not hasattr(config, "on_activity")
            or isinstance(config.on_activity, _ActivityRouter)
        ):
            return
        try:
            layer.config = dataclasses.replace(
                config, on_activity=_ActivityRouter(config.on_activity)
            )
        except (AttributeError, TypeError):
            return


#: What a bounded call returns when its deadline cancelled the turn.
_EXPIRED = object()


class InProcessTurnExecutor:
    """Run a turn by calling the role's layer in this process.

    Every request is bounded by its role's timeout: without ``timeout_s``
    each role has its default (:data:`DEFAULT_ROLE_TURN_TIMEOUTS`), with it
    every role has that one bound. A turn past its bound is cancelled -- the
    CLI it started is stopped -- and :class:`TurnTimeout` is raised. Once
    :meth:`keep_records` names a directory, each turn leaves its files there
    (the module docstring).
    """

    def __init__(
        self,
        layer_provider: LayerProvider,
        *,
        timeout_s: float | None = None,
        records: Path | str | None = None,
    ) -> None:
        self._provider = layer_provider
        self.timeout_s = DEFAULT_TURN_TIMEOUT_S if timeout_s is None else timeout_s
        self.role_timeouts = dict(DEFAULT_ROLE_TURN_TIMEOUTS) if timeout_s is None else {}
        #: Where turns leave their files, or ``None`` before :meth:`keep_records`.
        self.records: Path | None = None
        self._counter: Any = None
        if records is not None:
            self.keep_records(records)

    def timeout_for(self, role: str) -> float:
        """How long a turn of ``role`` may take before the run gives up on it."""
        return float(self.role_timeouts.get(role, self.timeout_s))

    def keep_records(self, root: Path | str) -> None:
        """Leave each turn's files in ``root`` from now on, numbered on from those already there."""
        # Absolute, so the paths a record names hold from any working directory.
        self.records = Path(root).absolute()
        self.records.mkdir(parents=True, exist_ok=True)
        seen = [_seq(p.name) for p in self.records.glob("*.request.json")]
        # Continued across a resume, so a later process never overwrites a record.
        self._counter = itertools.count(max(seen, default=0) + 1)

    def run(self, request: TurnRequest) -> TurnOutcome:
        layer = self._provider(request)
        timeout_s = self.timeout_for(request.role)
        record = None
        if self.records is not None:
            record = TurnRecord(self.records, f"{next(self._counter):03d}-{request.role}")
            record.begin(request, timeout_s)
            _route_activity(layer)
        token = _ACTIVE.set(record)
        outcome, text, error = "raised", None, ""
        try:
            with cli_stderr_to(_stderr_sink(record)):
                answer = self._call(layer, request, timeout_s, record)
            text = answer if isinstance(answer, str) else str(answer)
            outcome = "answered"
        except TurnTimeout as exc:
            outcome, error = "timed out", str(exc)
            raise
        except BaseException as exc:
            outcome = "interrupted" if isinstance(exc, KeyboardInterrupt) else "raised"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _ACTIVE.reset(token)
            if record is not None:
                record.finish(outcome, text=text, error=error)
        return TurnOutcome(role=request.role, text=text)

    @staticmethod
    def _call(layer: Any, request: TurnRequest, timeout_s: float, record: TurnRecord | None) -> Any:
        """``layer.aforward(prompt)``, cancelled at ``timeout_s``."""

        async def bounded() -> Any:
            with anyio.move_on_after(timeout_s):
                return await layer.aforward(request.prompt)
            return _EXPIRED

        answer = anyio.run(bounded)
        if answer is _EXPIRED:
            name = record.request_id if record is not None else request.role
            where = f" Its activity is in {record.activity_file}." if record is not None else ""
            raise TurnTimeout(
                f"{name}: no answer within {timeout_s:g}s (the {request.role} turn timeout); "
                f"the turn was stopped.{where}"
            )
        return answer


def _stderr_sink(record: TurnRecord | None) -> Callable[[str], None]:
    """Where a turn's CLI stderr lines go: its activity log, or nowhere without a record."""
    if record is None:
        return lambda line: None
    return lambda line: record.note("stderr", line)
