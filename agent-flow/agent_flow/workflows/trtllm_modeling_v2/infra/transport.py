# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The external agent transport: a role turn as files, answered by a driver session.

A host may be unable to authenticate the agent CLI the orchestrator spawns: the
sandbox makes the credential store read-only, and a staged copy of it holds a
refresh token the parent also holds, so one of the two copies loses on the next
rotation. A role turn is the only thing the orchestrator cannot do there; the
stage machine, the queue, decision validation and every gate are ordinary
Python. So the turn, and only the turn, becomes files::

    orchestrator                         driver (an authenticated session)
    ------------                         ---------------------------------
    write  dispatch/<id>.request.json
    write  dispatch/<id>.system.md
    write  dispatch/<id>.prompt.md
    wait for dispatch/<id>.done    <---  run the turn
                                         write dispatch/<id>.result.md
                                         write dispatch/<id>.done
    read   dispatch/<id>.result.md
    write  dispatch/<id>.read
    read   the decision file the request named

The driver (`infra/driver/`) is trusted with nothing: every decision still arrives
through the validated decision file, and every verdict from a gate this process
runs. Request ids are ``<seq>-<role>`` from a counter that continues across a
resume, and every file is written to a temporary name and renamed, so a reader
that sees one can trust it is complete.

A relaunched process asks for turns its predecessor already asked. One still
unanswered is waited on rather than asked again (:meth:`ExternalAgentTransport.run`).
One already answered is adopted by a caller that knows when the run is done
with an answer (:meth:`ExternalAgentTransport.adopt_answered`): reading an answer
is not that moment, because a task's checks run after its turn and a process
can die during them. The caller says when it is (:meth:`ExternalAgentTransport.consume`),
and a consumed turn is never adopted.
"""

from __future__ import annotations

import atexit
import errno
import itertools
import json
import os
import shutil
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .execution.turns import (
    DEFAULT_ROLE_TURN_TIMEOUTS,
    DEFAULT_TURN_TIMEOUT_S,
    TurnExecutor,
    TurnOutcome,
    TurnRequest,
    TurnTimeout,
)
from .substrate.scheduler import read_json

#: Subdirectory of the workspace the two sides meet in.
DISPATCH_DIRNAME = "dispatch"

#: Markers the orchestrator side leaves beside a request, each a JSON record of
#: the process that wrote it. ``.read``: an answer was handed to a caller.
#: ``.adopted``: a later process took the turn over instead of asking again.
#: ``.consumed``: the run is done with the answer, and nothing adopts it.
READ_SUFFIX = ".read"
ADOPTED_SUFFIX = ".adopted"
CONSUMED_SUFFIX = ".consumed"

#: A driver that has not touched its heartbeat in this long is gone. Checked
#: only if it ever wrote one.
HEARTBEAT_STALE_S = 300.0

#: The orchestrator side's liveness file, and the file it leaves when it exits.
ORCHESTRATOR_HEARTBEAT = "orchestrator.heartbeat"
ORCHESTRATOR_EXIT = "orchestrator.exit"


def job_prefix(request_id: str) -> str:
    """The prefix a turn gives anything it names on shared resources: ``014-smith`` -> ``r014``.

    Unique per request, so two turns can never pick the same spool job name and
    overwrite each other's results.
    """
    return f"r{request_id.split('-', 1)[0]}"


#: How often the orchestrator tells the driver it is alive.
ORCHESTRATOR_HEARTBEAT_S = 30.0


def _write_atomic(path: Path, text: str) -> None:
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(text, encoding="utf-8")
    staging.rename(path)


class OrchestratorHeartbeat:
    """This orchestrator's liveness file in ``dispatch/``, and the note it leaves when it exits.

    A driver waiting for requests or watching sessions cannot otherwise tell an
    orchestrator that is busy running gates from one that has exited: a run that
    closed writes no further request, and the driver would wait for one for
    hours. Every ``interval_s`` a thread writes ``orchestrator.heartbeat``, and
    at exit ``orchestrator.exit`` is written; each holds ``{"pid", "host",
    "started", "at"}``, ``started`` being when this process's orchestrator side
    began. Start one per process, from the orchestrator that holds the
    workspace lock, whichever transport its turns go through.
    """

    def __init__(
        self,
        workspace: Path | str,
        interval_s: float,
        *,
        started: float | None = None,
        write: Callable[[Path, str], None] | None = None,
    ) -> None:
        self.root = Path(workspace) / DISPATCH_DIRNAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.record = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started": time.time() if started is None else started,
        }
        self._write = write or _write_atomic
        self._stop = threading.Event()
        (self.root / ORCHESTRATOR_EXIT).unlink(missing_ok=True)
        threading.Thread(
            target=self._beat, args=(interval_s,), name="dispatch-heartbeat", daemon=True
        ).start()
        atexit.register(self.mark_exit)

    def _beat(self, interval_s: float) -> None:
        while True:
            try:
                self._write(
                    self.root / ORCHESTRATOR_HEARTBEAT,
                    json.dumps({**self.record, "at": time.time()}) + "\n",
                )
            except OSError as exc:
                # A full filesystem is transient (a shared quota); anything
                # else means the dispatch directory is gone.
                if exc.errno not in (errno.ENOSPC, errno.EDQUOT):
                    return
            if self._stop.wait(interval_s):
                return

    def mark_exit(self, record: Mapping | None = None) -> None:
        """Stop beating and write ``orchestrator.exit``; what ``atexit`` calls."""
        self._stop.set()
        try:
            self._write(
                self.root / ORCHESTRATOR_EXIT,
                json.dumps({**(record or self.record), "at": time.time()}) + "\n",
            )
        except OSError:
            pass

    def stop(self) -> None:
        """Stop beating; the exit note is still written when the process exits."""
        self._stop.set()


def start_orchestrator_heartbeat(
    workspace: Path | str,
    interval_s: float,
    *,
    started: float | None = None,
    write: Callable[[Path, str], None] | None = None,
) -> OrchestratorHeartbeat:
    """Start this process's :class:`OrchestratorHeartbeat` in ``workspace``'s ``dispatch/``."""
    return OrchestratorHeartbeat(workspace, interval_s, started=started, write=write)


@dataclass(frozen=True)
class AnsweredTurn:
    """An earlier process's answered turn, adopted instead of asked again."""

    request_id: str
    #: The driver's result text.
    text: str
    #: Where the turn wrote its decision; the adopting caller reads it there.
    decision_file: Path
    #: The process that had read the answer before, or ``None`` if none had.
    read_by: dict | None = None


def _seq(request_id: str) -> int:
    """The request counter in ``<seq>-<role>``; dispatch order past ``999`` too."""
    head = request_id.split("-", 1)[0]
    return int(head) if head.isdigit() else -1


class ExternalAgentTransport(TurnExecutor):
    """Deliver a turn through the filesystem and wait for the answer.

    Thread-safe: concurrent turns are ordinary here, and the only shared state
    is the id counter. Every request names its role's own timeout; without
    ``timeout_s`` each role has its default (:data:`DEFAULT_ROLE_TURN_TIMEOUTS`),
    with it every role has that one bound.
    """

    name = "file-dispatch"

    def __init__(
        self,
        workspace: Path | str,
        *,
        poll_s: float = 5.0,
        timeout_s: float | None = None,
        role_timeouts: Mapping[str, float] | None = None,
        heartbeat_s: float | None = None,
    ) -> None:
        self.root = Path(workspace) / DISPATCH_DIRNAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.poll_s = poll_s
        self.timeout_s = DEFAULT_TURN_TIMEOUT_S if timeout_s is None else timeout_s
        timeouts = dict(DEFAULT_ROLE_TURN_TIMEOUTS) if timeout_s is None else {}
        timeouts.update(role_timeouts or {})
        self.role_timeouts = timeouts
        self._created = time.time()
        self._heartbeat: OrchestratorHeartbeat | None = None
        if heartbeat_s:
            self._heartbeat = start_orchestrator_heartbeat(
                self.root.parent, heartbeat_s, started=self._created, write=self._write_atomic
            )
        self._counter = itertools.count(self._next_seq())
        self._lock = threading.Lock()
        #: Request ids a caller of this executor is waiting on right now. Such
        #: a request is never a twin: its caller will read the answer, and the
        #: next caller's prompt may differ even when the role, ``what`` and
        #: decision file match. Guarded by ``_lock``.
        self._in_flight: set[str] = set()
        #: Request ids this process has waited on or adopted: their answer is
        #: this process's, and is never adopted again here. Guarded by ``_lock``.
        self._read: set[str] = set()

    # ------------------------------------------------------------ helpers

    def timeout_for(self, role: str) -> float:
        """How long a turn of ``role`` may take before the run gives up on it."""
        return float(self.role_timeouts.get(role, self.timeout_s))

    def _mark_exit(self, record: dict) -> None:
        """Leave the exit note now, as the heartbeat does at exit; nothing without one."""
        if self._heartbeat is not None:
            self._heartbeat.mark_exit(record)

    def close(self) -> None:
        """Stop the heartbeat thread, if one runs."""
        if self._heartbeat is not None:
            self._heartbeat.stop()

    def _next_seq(self) -> int:
        """Continue the numbering across a resume rather than restarting it.

        A resume that renumbers from 1 overwrites the previous attempt's
        request files, which is the evidence for what the run asked for.
        """
        seen = [
            int(p.name.split("-", 1)[0])
            for p in self.root.glob("*.request.json")
            if p.name.split("-", 1)[0].isdigit()
        ]
        return (max(seen) + 1) if seen else 1

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        _write_atomic(path, text)

    def _driver_is_gone(self) -> bool:
        beat = self.root / "driver.heartbeat"
        if not beat.is_file():
            return False
        try:
            return (time.time() - beat.stat().st_mtime) > HEARTBEAT_STALE_S
        except OSError:
            return False

    def _process(self) -> dict:
        """This process, as the heartbeat and the request markers name it."""
        return {"pid": os.getpid(), "host": socket.gethostname(), "started": self._created}

    def _mark(self, request_id: str, suffix: str, **detail) -> bool:
        """Leave a marker beside ``request_id``; whether it was written.

        Bookkeeping only: the answer it describes is already in hand, so a
        full filesystem must not turn into a lost turn.
        """
        try:
            self._write_atomic(
                self.root / f"{request_id}{suffix}",
                json.dumps({**self._process(), "at": time.time(), **detail}) + "\n",
            )
        except OSError:
            return False
        return True

    def _requests(self, role: str | None = None) -> list[tuple[str, dict]]:
        """``(id, request JSON)`` of every request, or every one of ``role``, oldest first."""
        out = []
        for path in self.root.glob(f"*-{role}.request.json" if role else "*.request.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                out.append((path.name[: -len(".request.json")], data))
        return sorted(out, key=lambda item: _seq(item[0]))

    def _answered(self, request_id: str) -> bool:
        return (self.root / f"{request_id}.done").exists()

    # -------------------------------------------------------------- public

    def prior_turns(self, role: str) -> list[dict]:
        """Every earlier answered turn for ``role``, oldest first.

        This transport starts a fresh session per turn, which silently
        breaks any role the caller declares **persistent** — one asked to
        remember what it decided so that a later turn revises rather than
        re-derives. Their prompts often say so in as many words ("You are in
        the same session as your first draft"), and here that sentence is
        false.

        Restoring it exactly would mean replaying a conversation. Restoring
        it USEFULLY only needs the artifacts, which are already on disk:
        what the role was asked, and what it said back. A driver hands those
        to the new session as its own prior reasoning.
        """
        out = []
        for request in sorted(self.root.glob(f"*-{role}.request.json")):
            name = request.name[: -len(".request.json")]
            if not (self.root / f"{name}.done").exists():
                continue
            out.append(
                {
                    "id": name,
                    "prompt_file": str(self.root / f"{name}.prompt.md"),
                    "result_file": str(self.root / f"{name}.result.md"),
                }
            )
        return out

    def _unanswered_twin(self, request: TurnRequest) -> str | None:
        """An earlier, still-unanswered request asking exactly this turn.

        A process that dies with turns in flight leaves their request files
        behind, and a driver may be working on them. The resumed process asks
        the same turns again (same role, same ``what``, same decision file);
        writing a second request would duplicate the work -- or, worse, give
        two agents the same decision path. Waiting on the first one is the
        same answer to the same question.

        A request some caller of this executor is still waiting on is not a
        twin. Without that rule, whether two concurrent callers asking alike
        share one id would depend on whether the first caller's request file
        is on disk yet when the second one looks, so they could receive one
        answer between them. An orphan is likewise adopted by one caller at a
        time. Call with ``_lock`` held.
        """
        for name, data in self._requests(request.role):
            if name in self._in_flight or self._answered(name):
                continue
            if (
                data.get("what") == request.what
                and data.get("decision_file") == request.decision_file
                and _same_work(request, data)
            ):
                return name
        return None

    def _orphaned_turn(self, request: TurnRequest) -> tuple[str, str | None] | None:
        """An unanswered request a previous process left for this role and ``what``.

        Consulted when no exact twin exists: after a restart the caller may
        reserve the same turn's decision under a new attempt, so the decision
        file differs while the work asked for does not. Returns the request id
        and the decision file it named; its decision is copied to the path now
        expected. Call with ``_lock`` held.

        ``what`` alone does not name the work: two tasks verifying one entry
        are both asked ``verify <entry>``. Where both requests name the work
        they are for (:attr:`TurnRequest.work`), it must be the same
        (:func:`_same_work`), or one task's turn, and its decision, would be
        taken for the other's.
        """
        for name, data in self._requests(request.role):
            if name in self._in_flight or self._answered(name):
                continue
            if (
                data.get("what") != request.what
                or float(data.get("requested_at", 0)) >= self._created
                or not _same_work(request, data)
            ):
                continue
            return name, data.get("decision_file")
        return None

    def adopt_answered(
        self, role: str, what: str, accept: Callable[[Path], bool]
    ) -> AnsweredTurn | None:
        """Adopt an earlier process's answered turn for ``role`` and ``what``, or ``None``.

        A process can die after the driver answered a turn and before the run
        acted on the answer for good: before reading it, or after reading it
        and before the work it was for had an outcome -- a task's checks run
        after its turn. Asking again would pay for the turn twice. So the
        latest request for ``role`` and ``what`` is adopted when an earlier
        process asked it, it is answered and not consumed, this process has
        not had its answer, and ``accept`` takes the decision file it names.
        Only the latest: an answer followed by another request for the same
        work was superseded by it.

        The turn is marked adopted and read by this process, and stays
        unconsumed: if this process also dies before the run is done with it,
        the next one adopts it again. Its decision is read where it was
        written, and nothing is copied.
        """
        with self._lock:
            mine = [(n, d) for n, d in self._requests(role) if d.get("what") == what]
            if not mine:
                return None
            name, data = mine[-1]
            decision = data.get("decision_file")
            if (
                not decision
                or name in self._in_flight
                or name in self._read
                or float(data.get("requested_at", 0)) >= self._created
                or not self._answered(name)
                or (self.root / f"{name}{CONSUMED_SUFFIX}").exists()
            ):
                return None
            path = Path(decision)
            if not path.is_absolute() and data.get("cwd"):
                path = Path(data["cwd"]) / path
            if not accept(path):
                return None
            read_by = read_json(self.root / f"{name}{READ_SUFFIX}")
            self._read.add(name)
        self._mark(name, ADOPTED_SUFFIX, answered=True)
        self._mark(name, READ_SUFFIX)
        result = self.root / f"{name}.result.md"
        return AnsweredTurn(
            request_id=name,
            text=result.read_text(encoding="utf-8") if result.is_file() else "",
            decision_file=path,
            read_by=read_by,
        )

    def unread_answer(self, role: str, what: str) -> AnsweredTurn | None:
        """The latest answer for ``role`` and ``what`` that no process ever read, or ``None``.

        A process stopped between asking a turn and reading its answer leaves
        that answer behind. A turn that is not a task's (a planner's) is not
        adopted, as its answer is to a standing that has moved on; but the
        work in it is not lost either. The answer is marked read by this
        process, so it is handed on once, and is neither adopted nor consumed.
        """
        with self._lock:
            mine = [(n, d) for n, d in self._requests(role) if d.get("what") == what]
            if not mine:
                return None
            name, data = mine[-1]
            if (
                name in self._in_flight
                or name in self._read
                or float(data.get("requested_at", 0)) >= self._created
                or not self._answered(name)
                or (self.root / f"{name}{READ_SUFFIX}").exists()
                or (self.root / f"{name}{CONSUMED_SUFFIX}").exists()
            ):
                return None
            self._read.add(name)
        self._mark(name, READ_SUFFIX)
        result = self.root / f"{name}.result.md"
        decision = Path(str(data.get("decision_file") or ""))
        if not decision.is_absolute() and data.get("cwd"):
            decision = Path(data["cwd"]) / decision
        return AnsweredTurn(
            request_id=name,
            text=result.read_text(encoding="utf-8") if result.is_file() else "",
            decision_file=decision,
        )

    def consume(self, work: Iterable[str]) -> list[str]:
        """Mark consumed every answered turn whose ``what`` is in ``work``; the ids marked now.

        The caller names work the run is done with, such as a task that has
        its outcome, so no process adopts an answer for it again, however the
        work is reopened later. A turn a caller is still waiting on is left
        alone.
        """
        wanted = set(work)
        marked: list[str] = []
        if not wanted:
            return marked
        with self._lock:
            for name, data in self._requests():
                if (
                    data.get("what") not in wanted
                    or name in self._in_flight
                    or not self._answered(name)
                    or (self.root / f"{name}{CONSUMED_SUFFIX}").exists()
                ):
                    continue
                if self._mark(name, CONSUMED_SUFFIX):
                    marked.append(name)
        return marked

    def run(self, request: TurnRequest) -> TurnOutcome:
        their_decision = None
        with self._lock:
            adopted = self._unanswered_twin(request)
            if adopted is None:
                orphan = self._orphaned_turn(request)
                if orphan is not None:
                    adopted, their_decision = orphan
            request_id = adopted or f"{next(self._counter):03d}-{request.role}"
            self._in_flight.add(request_id)
        if adopted is not None:
            self._mark(adopted, ADOPTED_SUFFIX, answered=False)
        try:
            outcome = self._await(request, request_id, adopted)
        finally:
            with self._lock:
                self._in_flight.discard(request_id)
                self._read.add(request_id)
        self._mark(request_id, READ_SUFFIX)
        if their_decision and request.decision_file and their_decision != request.decision_file:
            source, target = Path(their_decision), Path(request.decision_file)
            # Copied, never moved, and never over an existing file: decision
            # files are evidence and are not overwritten.
            if source.is_file() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return outcome

    def _await(self, request: TurnRequest, request_id: str, adopted: str | None) -> TurnOutcome:
        # The role's system prompt travels WITH the turn. In-process it is
        # set on the session and is easy to forget here -- and a driver that
        # delivers only the user prompt is running a different agent: no
        # role, no skills, none of the invariants the writing roles carry.
        # The result would still look like a plausible answer.
        system = self.root / f"{request_id}.system.md"
        prompt = self.root / f"{request_id}.prompt.md"
        result = self.root / f"{request_id}.result.md"
        done = self.root / f"{request_id}.done"

        if adopted is None:
            self._write_request(request, request_id, system, prompt, result, done)

        timeout_s = self.timeout_for(request.role)
        deadline = time.monotonic() + timeout_s
        while not done.exists():
            if self._driver_is_gone():
                raise TurnTimeout(
                    f"{request_id}: the dispatch driver's heartbeat is stale. "
                    f"Nothing is reading {self.root}; the run would otherwise wait "
                    f"{timeout_s:.0f}s for a turn nobody is taking."
                )
            if time.monotonic() > deadline:
                raise TurnTimeout(
                    f"{request_id}: no answer within {timeout_s:.0f}s (the {request.role} "
                    f"turn timeout). The prompt is at {prompt}."
                )
            time.sleep(self.poll_s)

        return TurnOutcome(
            role=request.role,
            text=result.read_text(encoding="utf-8") if result.is_file() else "",
            transport=self.name,
            request_id=request_id,
        )

    def _write_request(self, request, request_id, system, prompt, result, done) -> None:
        self._write_atomic(system, request.system_prompt)
        self._write_atomic(prompt, request.prompt)
        self._write_atomic(
            self.root / f"{request_id}.request.json",
            json.dumps(
                {
                    "id": request_id,
                    "role": request.role,
                    "what": request.what,
                    "work": request.work,
                    "system_prompt_file": str(system),
                    "prompt_file": str(prompt),
                    "result_file": str(result),
                    "done_file": str(done),
                    "decision_file": request.decision_file,
                    # The agent task.yaml routes the role to; empty means the
                    # driver runs the turn on its own model and effort.
                    "agent": {
                        "backend": request.backend,
                        "model": request.model,
                        "reasoning_effort": request.reasoning_effort,
                    },
                    # Names the turn gives shared resources (spool jobs) start
                    # with this, so two turns never overwrite each other's.
                    "job_prefix": job_prefix(request_id),
                    # Relative paths above are relative to this, the
                    # orchestrator's working directory; a driver elsewhere
                    # resolves them against it, so it writes exactly the
                    # files the orchestrator will read.
                    "cwd": os.getcwd(),
                    "timeout_s": self.timeout_for(request.role),
                    # Named rather than inlined: a driver that wants the
                    # role's own history reads these, and one that does not
                    # is not forced to carry them in the prompt.
                    "prior_turns": self.prior_turns(request.role),
                    "requested_at": time.time(),
                },
                indent=2,
            )
            + "\n",
        )


def _same_work(request: TurnRequest, data: Mapping[str, Any]) -> bool:
    """Whether a request on disk is for ``request``'s work; true when either does not say."""
    theirs = data.get("work")
    return not request.work or not theirs or theirs == request.work


__all__ = [
    "ADOPTED_SUFFIX",
    "CONSUMED_SUFFIX",
    "DEFAULT_ROLE_TURN_TIMEOUTS",
    "DEFAULT_TURN_TIMEOUT_S",
    "DISPATCH_DIRNAME",
    "HEARTBEAT_STALE_S",
    "ORCHESTRATOR_EXIT",
    "ORCHESTRATOR_HEARTBEAT",
    "ORCHESTRATOR_HEARTBEAT_S",
    "READ_SUFFIX",
    "AnsweredTurn",
    "ExternalAgentTransport",
    "OrchestratorHeartbeat",
    "job_prefix",
    "start_orchestrator_heartbeat",
]
