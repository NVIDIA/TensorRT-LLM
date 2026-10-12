# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One agent turn, as a request and an outcome.

A *turn* is one round trip with one role: the orchestrator hands over a
system prompt and a task, the role does its work, and a text answer plus —
where the caller asked for one — a typed decision comes back. That is the
entire unit of work an orchestrator cannot do in Python, and it is the only
thing a transport moves.

Everything else stays with the caller: which stage runs, which item is next,
whether a decision is valid, whether a gate passed. So the two transports
that exist are interchangeable by construction, and the orchestrator should
not be able to tell them apart from its own source. That is what these types
are for.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, ClassVar


class TurnExecutionError(RuntimeError):
    """A turn could not be delivered or could not be answered."""


class TurnTimeout(TurnExecutionError):
    """Nothing answered the turn inside its deadline."""


@dataclass(frozen=True)
class TurnRequest:
    """What one role is being asked to do.

    ``system_prompt`` travels **with** the request rather than being set once
    on a session. In process it could be either; through a mailbox it has to
    be the former, because a driver handed only the task prompt runs a
    different agent — no role, no skills, none of the invariants the writing
    roles carry — and returns something that still looks like a plausible
    answer.

    ``decision_file`` is the path the caller will read the typed decision
    back from. It is part of the request because the role has to be told
    exactly where to write; a pattern it has to instantiate is one it can
    instantiate differently.
    """

    role: str
    what: str
    prompt: str
    system_prompt: str = ""
    decision_file: str | None = None
    #: The caller's own numbering for this turn (a plan round, a queue item,
    #: an assembler turn). Carried so a transport can name its artifacts
    #: after it; never interpreted here.
    iteration: int | None = None
    #: Paths the role should read as input. Named rather than inlined, so a
    #: transport that cannot afford the prompt budget can hand over pointers.
    context_refs: tuple[str, ...] = ()
    #: The backend and model the caller configured for this role. An
    #: in-process transport already built its session from them; a transport
    #: that hands the turn to someone else passes them on, because otherwise
    #: nothing tells that driver which model the role was meant to run on.
    backend: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class TurnOutcome:
    """What came back, and enough provenance to defend it later.

    ``text`` is the role's prose. **It is not a control-plane input**: a
    caller that branches on it has given the role the ability to talk its way
    past a gate. The typed decision arrives separately, validated against a
    closed enum, and that is what a state machine may read.
    """

    role: str
    text: str
    #: Which executor ran it, for the journal.
    transport: str = ""
    #: The transport's own id for this turn, where it has one.
    request_id: str = ""
    #: Where the typed decision was expected, echoed back for the collector.
    decision_ref: str | None = None
    #: Files this turn produced, oldest first.
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    #: The conversation this turn belonged to, where the transport has one.
    session_ref: str | None = None
    duration_s: float = 0.0


class TurnExecutor(abc.ABC):
    """Deliver one turn and return its answer.

    Deliberately an abstract base rather than a bare protocol, because
    :meth:`bound` has a default that every transport but one wants.
    """

    #: Recorded in :attr:`TurnOutcome.transport`.
    name: ClassVar[str] = "turn-executor"

    @abc.abstractmethod
    def run(self, request: TurnRequest) -> TurnOutcome:
        """Deliver ``request`` and block until it is answered."""

    def bound(self, session: Any) -> "TurnExecutor":
        """A view of this executor pinned to one caller-owned session.

        Concurrent work often builds its own session per item — a batch of
        catalog dispatches is ten independent agents, not ten turns of one —
        and the caller, not the transport, is the thing that knows that.

        The default ignores it, which is correct for every transport with no
        session concept: a mailbox has one directory whatever the caller
        thought it was handing over. Only the in-process executor overrides
        it. Callers can therefore write ``executor.bound(x).run(...)``
        unconditionally and stay free of transport branches.
        """
        return self


__all__ = [
    "TurnExecutionError",
    "TurnExecutor",
    "TurnOutcome",
    "TurnRequest",
    "TurnTimeout",
]
