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

import copy
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
from typing import Any, Callable, Iterable

import anyio

from agent_flow.backends.claude_code import cli_stderr_to

from ..infra.files import read_json, write_json_atomic, write_text_atomic


class TurnExecutionError(RuntimeError):
    """A turn could not be delivered or could not be answered."""


class TurnTimeout(TurnExecutionError):
    """Nothing answered the turn inside its deadline."""


#: How long a turn may take when neither the caller nor the role says.
DEFAULT_TURN_TIMEOUT_S = 3 * 60 * 60

#: Per-role turn timeouts, in seconds.
DEFAULT_ROLE_TURN_TIMEOUTS: dict[str, float] = {
    "surveyor": 4 * 3600.0,
    "kernel_writer": 6 * 3600.0,
    "assembler": 16 * 3600.0,
    # One candidate: implement, test on GPU, micro-bench -- hours of spool jobs.
    "tuner": 8 * 3600.0,
    "planner": 15 * 60.0,
}

#: Where in-process turns leave their files, under the workspace.
TURNS_DIRNAME = "turns"
#: The marker beside an answered turn's files once the run is done with it: no
#: later process adopts that answer.
CONSUMED_SUFFIX = ".consumed"

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
    #: The work the turn is for (a graph task's id), when ``what`` alone does
    #: not name it: two tasks verifying one entry are both asked ``verify
    #: <entry>``. A restarted run adopts a turn a previous process left only
    #: for the same work. Empty: ``what`` names it.
    work: str = ""


@dataclass(frozen=True)
class TurnOutcome:
    """The role's prose answer. **Not a control-plane input**: decisions are typed."""

    role: str
    text: str
    #: The id of the turn's records (``turns/<id>.*``), where it has them.
    request_id: str = ""


@dataclass(frozen=True)
class AnsweredTurn:
    """An earlier process's answered turn, adopted instead of asked again."""

    request_id: str
    #: The answering session's result text.
    text: str
    #: Where the turn wrote its decision; the adopting caller reads it there.
    decision_file: Path
    result_file: Path


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
                    "work": request.work,
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


class _TurnLog:
    """Where an executor and its bound views keep their records, and the ids they share."""

    def __init__(self) -> None:
        self.root: Path | None = None
        self._lock = threading.Lock()
        self._counter: Any = None
        self.created = time.time()
        self.read: set[str] = set()

    def keep_in(self, root: Path | str) -> None:
        # Absolute, so the paths a record names hold from any working directory.
        root = Path(root).absolute()
        with self._lock:
            if self.root == root:
                return
            root.mkdir(parents=True, exist_ok=True)
            seen = [_seq(p.name) for p in root.glob("*.request.json")]
            self.root = root
            # Continued across a resume, so a later process never overwrites a record.
            self._counter = itertools.count(max(seen, default=0) + 1)

    def open(self, request: TurnRequest, timeout_s: float) -> TurnRecord | None:
        with self._lock:
            if self.root is None:
                return None
            request_id = f"{next(self._counter):03d}-{request.role}"
            root = self.root
        record = TurnRecord(root, request_id)
        record.begin(request, timeout_s)
        return record


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
    (the module docstring), and an answered turn can be adopted after a
    restart until its task has an outcome and consumes it.
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
        self._log = _TurnLog()
        if records is not None:
            self._log.keep_in(records)

    def timeout_for(self, role: str) -> float:
        """How long a turn of ``role`` may take before the run gives up on it."""
        return float(self.role_timeouts.get(role, self.timeout_s))

    def keep_records(self, root: Path | str) -> None:
        """Leave each turn's files in ``root`` from now on, numbered on from those already there.

        Shared with every view :meth:`bound` returns.
        """
        self._log.keep_in(root)

    @property
    def records(self) -> Path | None:
        """Where turns leave their files, or ``None`` before :meth:`keep_records`."""
        return self._log.root

    def bound(self, session: Any) -> "InProcessTurnExecutor":
        """Pin every turn to ``session``, whatever the request asks for."""
        view = copy.copy(self)
        view._provider = lambda request: session
        return view

    def _requests(self, role: str | None = None) -> list[tuple[str, dict]]:
        root = self.records
        if root is None:
            return []
        rows = []
        pattern = f"*-{role}.request.json" if role else "*.request.json"
        for path in root.glob(pattern):
            data = read_json(path)
            if data is not None:
                rows.append((path.name.removesuffix(".request.json"), data))
        return sorted(rows, key=lambda row: _seq(row[0]))

    def _answered(self, request_id: str) -> bool:
        root = self.records
        if root is None:
            return False
        done = read_json(root / f"{request_id}.done")
        return bool(done and done.get("outcome") == "answered" and not done.get("error"))

    def adopt_answered(
        self, role: str, what: str, accept: Callable[[Path], bool]
    ) -> AnsweredTurn | None:
        """Recover the latest completed turn, without rerunning its agent or its tools.

        A newer incomplete or failed turn supersedes older answers. Consumed
        answers and turns started or already read by this executor are never
        adopted; a later process may still adopt an unconsumed answer again.
        """
        if self.records is None:
            return None
        with self._log._lock:
            rows = [
                (name, data)
                for name, data in self._requests(role)
                if (data.get("work") or data.get("what")) == what
            ]
            if not rows:
                return None
            name, data = rows[-1]
            try:
                started = float(data.get("started_at") or 0)
            except (TypeError, ValueError):
                return None
            if (
                name in self._log.read
                or started <= 0
                or started >= self._log.created
                or (self.records / f"{name}{CONSUMED_SUFFIX}").exists()
                or not self._answered(name)
            ):
                return None
            decision = data.get("decision_file")
            if (
                not isinstance(decision, str)
                or not decision
                or not Path(decision).is_absolute()
                or not accept(Path(decision))
            ):
                return None
            try:
                text = (self.records / f"{name}.result.md").read_text(encoding="utf-8")
            except OSError:
                return None
            self._log.read.add(name)
            return AnsweredTurn(name, text, Path(decision), self.records / f"{name}.result.md")

    def consume(self, work: Iterable[str]) -> list[str]:
        """Keep a finished or retried task's prior answers from being adopted again."""
        if self.records is None:
            return []
        wanted = set(work)
        if not wanted:
            return []
        consumed = []
        with self._log._lock:
            for name, data in self._requests():
                marker = self.records / f"{name}{CONSUMED_SUFFIX}"
                identity = data.get("work") or data.get("what")
                if not isinstance(identity, str) or identity not in wanted or marker.exists():
                    continue
                if not self._answered(name):
                    continue
                try:
                    write_json_atomic(marker, {**_process(), "at": time.time()})
                except OSError:
                    continue
                consumed.append(name)
        return consumed

    def run(self, request: TurnRequest) -> TurnOutcome:
        layer = self._provider(request)
        if layer is None:
            raise TurnExecutionError(f"no layer available for role {request.role!r}")
        timeout_s = self.timeout_for(request.role)
        record = self._log.open(request, timeout_s)
        if record is not None:
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
        return TurnOutcome(
            role=request.role,
            text=text,
            request_id=record.request_id if record is not None else "",
        )

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


__all__ = [
    "ACTIVITY_ENTRY_LIMIT",
    "CONSUMED_SUFFIX",
    "AnsweredTurn",
    "DEFAULT_ROLE_TURN_TIMEOUTS",
    "DEFAULT_TURN_TIMEOUT_S",
    "TURNS_DIRNAME",
    "InProcessTurnExecutor",
    "LayerProvider",
    "TurnExecutionError",
    "TurnOutcome",
    "TurnRecord",
    "TurnRequest",
    "TurnTimeout",
]
