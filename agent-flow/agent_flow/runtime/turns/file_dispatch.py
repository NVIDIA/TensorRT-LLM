# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The turn as a file, for a host that cannot authenticate an agent.

THE PROBLEM, AND WHY IT IS NOT AN AUTHENTICATION PROBLEM

An orchestrator that spawns a fresh CLI per turn needs that CLI to
authenticate itself, from a credential store the sandbox may make read-only.
A staged, writable home solves the *write* half. It does not solve the other
half, and cannot:

    the staged store holds a copy of a refresh token the parent also holds,
    and a refresh rotates it. One of the two copies then loses.

Measured twice on the staircase project. One run recorded "works for about
forty minutes"; the next saw a staged home authenticate cleanly at 12:12 and
fail with `Not logged in · Please run /login` at 12:25, with byte-identical
credentials. Copying harder does not fix a single-use token, and doing it
under ten concurrent agents makes it worse rather than better.

THE ANSWER: MOVE THE TRANSPORT, AND NOTHING ELSE

A role turn is the ONLY thing the orchestrator cannot do here. The stage
machine, the queue walk, the dependency batching, the prompt construction,
the decision-file validation and every gate are ordinary Python and work
fine. So the turn becomes a file:

    orchestrator                         driver (an authenticated session)
    ------------                         ---------------------------------
    write  dispatch/<id>.request.json
    write  dispatch/<id>.system.md
    write  dispatch/<id>.prompt.md
    wait for dispatch/<id>.done    <---  run the turn, however it can
                                         write dispatch/<id>.result.md
                                         write dispatch/<id>.done
    read   dispatch/<id>.result.md
    read   the decision file the request named

The driver can be a human, another agent session, or a service. It is not
trusted with anything: it delivers text and returns text. Every *decision*
still arrives through the caller's own validator, against the same closed
enum an in-process tool schema would have carried, and every *verdict* still
comes from a check the caller ran itself.

NAMING

Request ids are ``<seq>-<role>`` with a monotonic counter, so a directory
listing sorts into dispatch order and a reader can see what the run did
without parsing anything. Concurrency is fine: ids are unique and each file
is written to a temporary name and renamed, so a driver that sees a file can
trust it is complete.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from pathlib import Path
from typing import Callable, ClassVar

from .base import TurnExecutor, TurnOutcome, TurnRequest, TurnTimeout

#: Subdirectory of the workspace the two sides meet in.
DISPATCH_DIRNAME = "dispatch"

#: How long a single role turn may take before the orchestrator gives up on
#: it. Generous: an assembler turn on a large checkpoint is tens of minutes,
#: and a driver that is a human session may be slower still. The failure mode
#: this bounds is a driver that has stopped, not a turn that is slow.
DEFAULT_TURN_TIMEOUT_S = 3 * 60 * 60

#: A driver that has not touched its heartbeat in this long is gone. Checked
#: only if it ever wrote one, so a driver that does not bother is not
#: penalized --- but one that does gets a fast, named failure instead of a
#: three-hour wait.
HEARTBEAT_STALE_S = 300.0

#: Builds the paragraph appended to a role's system prompt saying what this
#: transport cannot honour. Supplied by the caller, because what a role
#: prompt promises is the caller's vocabulary, not this module's.
TransportNote = Callable[[str], str]


class FileDispatchTurnExecutor(TurnExecutor):
    """Deliver a turn through the filesystem and wait for the answer.

    Thread-safe: concurrent turns are ordinary here, and the only shared
    state is the id counter.
    """

    name: ClassVar[str] = "file-dispatch"

    def __init__(
        self,
        workspace: Path | str,
        *,
        poll_s: float = 5.0,
        timeout_s: float = DEFAULT_TURN_TIMEOUT_S,
        dispatch_dirname: str = DISPATCH_DIRNAME,
        transport_note: TransportNote | None = None,
    ) -> None:
        self.root = Path(workspace) / dispatch_dirname
        self.root.mkdir(parents=True, exist_ok=True)
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self._note = transport_note
        self._counter = itertools.count(self._next_seq())
        self._lock = threading.Lock()

    # ------------------------------------------------------------ helpers

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
        staging = path.with_suffix(path.suffix + ".tmp")
        staging.write_text(text, encoding="utf-8")
        staging.rename(path)

    def _driver_is_gone(self) -> bool:
        beat = self.root / "driver.heartbeat"
        if not beat.is_file():
            return False
        try:
            return (time.time() - beat.stat().st_mtime) > HEARTBEAT_STALE_S
        except OSError:
            return False

    def _transport_note(self, role: str) -> str:
        """What this transport cannot honour, said once, by the transport.

        A role prompt written for the in-process path ends by telling the
        agent to call a typed-decision tool. That tool is real there — the
        session it is handed to exists — and absent here, because this
        transport has no session to host it in. The prompt therefore arrives
        carrying two endings and only the second one works.

        Correcting the role prompt would be the wrong repair: the
        instruction is right for the transport it was written for. The
        difference belongs to the transport, so the transport states it.
        """
        return self._note(role) if self._note is not None else ""

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

    def pending(self) -> list[Path]:
        """Request files with no answer yet, in dispatch order. For a driver."""
        return sorted(
            p
            for p in self.root.glob("*.request.json")
            if not (self.root / f"{p.name[: -len('.request.json')]}.done").exists()
        )

    def answer(self, request_id: str, result: str) -> None:
        """Record a turn's output. For a driver; the orchestrator never calls it."""
        self._write_atomic(self.root / f"{request_id}.result.md", result)
        # `.done` last, so a reader that sees it can trust the result beside
        # it is complete. Same rule as a compute spool's `.rc`.
        self._write_atomic(self.root / f"{request_id}.done", "1\n")

    def _unanswered_twin(self, request: TurnRequest) -> str | None:
        """An earlier, still-unanswered request asking exactly this turn.

        A process that dies with turns in flight leaves their request files
        behind, and a driver may be working on them. The resumed process asks
        the same turns again (same role, same ``what``, same decision file);
        writing a second request would duplicate the work -- or, worse, give
        two agents the same decision path. Waiting on the first one is the
        same answer to the same question.
        """
        for path in sorted(self.root.glob(f"*-{request.role}.request.json")):
            name = path.name[: -len(".request.json")]
            if (self.root / f"{name}.done").exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("what") == request.what and data.get("decision_file") == (
                request.decision_file
            ):
                return name
        return None

    def run(self, request: TurnRequest) -> TurnOutcome:
        with self._lock:
            adopted = self._unanswered_twin(request)
            request_id = adopted or f"{next(self._counter):03d}-{request.role}"

        started = time.monotonic()
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

        deadline = time.monotonic() + self.timeout_s
        while not done.exists():
            if self._driver_is_gone():
                raise TurnTimeout(
                    f"{request_id}: the dispatch driver's heartbeat is stale. "
                    f"Nothing is reading {self.root}; the run would otherwise wait "
                    f"{self.timeout_s:.0f}s for a turn nobody is taking."
                )
            if time.monotonic() > deadline:
                raise TurnTimeout(
                    f"{request_id}: no answer within {self.timeout_s:.0f}s. "
                    f"The prompt is at {prompt}."
                )
            time.sleep(self.poll_s)

        return TurnOutcome(
            role=request.role,
            text=result.read_text(encoding="utf-8") if result.is_file() else "",
            transport=self.name,
            request_id=request_id,
            decision_ref=request.decision_file,
            artifacts=(str(system), str(prompt), str(result)),
            duration_s=time.monotonic() - started,
        )

    def _write_request(self, request, request_id, system, prompt, result, done) -> None:
        self._write_atomic(system, request.system_prompt + self._transport_note(request.role))
        self._write_atomic(prompt, request.prompt)
        self._write_atomic(
            self.root / f"{request_id}.request.json",
            json.dumps(
                {
                    "id": request_id,
                    "role": request.role,
                    "what": request.what,
                    "system_prompt_file": str(system),
                    "prompt_file": str(prompt),
                    "result_file": str(result),
                    "done_file": str(done),
                    "decision_file": request.decision_file,
                    "context_refs": list(request.context_refs),
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

    def __call__(
        self,
        *,
        role: str,
        what: str,
        prompt: str,
        system_prompt: str = "",
        decision_file: str | None = None,
    ) -> str:
        """Call it exactly where a layer would be called.

        Kept because a transport that can be dropped in where an
        ``AgentLayer`` was is the cheapest possible migration, and because
        the staircase orchestrator was wired this way before
        :class:`TurnRequest` existed.
        """
        return self.run(
            TurnRequest(
                role=role,
                what=what,
                prompt=prompt,
                system_prompt=system_prompt,
                decision_file=decision_file,
            )
        ).text


__all__ = [
    "DEFAULT_TURN_TIMEOUT_S",
    "DISPATCH_DIRNAME",
    "FileDispatchTurnExecutor",
    "HEARTBEAT_STALE_S",
    "TransportNote",
]
