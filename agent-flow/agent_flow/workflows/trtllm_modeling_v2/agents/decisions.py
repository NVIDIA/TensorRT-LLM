# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The receipt: the two things a turn can end with, and what each carries.

Every role ends its turn the same way, with one file whose ``decision`` is ``ready`` or
``blocked``. One table for the three places that must agree on it: the role's prompt,
whose "Ending your turn" section is rendered from it (:func:`ending_section`); the
decision-file validation (``run/execution/handoff.py``); and the orchestrator, for which
``ready`` ends an attempt done once its work has landed and its check passed, and
``blocked`` opens an escalation for the outer agent session (``run/escalation.py``).

``blocked`` is not a failure to hide: it is how a problem an agent cannot solve reaches
somebody who can. What it carries (:data:`BLOCKED_KEYS`) is what that somebody needs to
not start over.
"""

from __future__ import annotations

from dataclasses import dataclass

READY, BLOCKED = "ready", "blocked"
DECISIONS: tuple[str, ...] = (READY, BLOCKED)
#: What a blocked receipt carries: what was tried, the one command that shows the problem
#: and its result line, what was ruled out, and the one question whose answer unblocks it.
BLOCKED_KEYS: tuple[str, ...] = ("tried", "command", "result", "ruled_out", "next_question")
#: What a ready receipt carries beside its summary: the card's check as it was run, the
#: references actually read, and what the task asked that was left.
READY_KEYS: tuple[str, ...] = ("check", "references", "not_done")
#: The roles a task's turn runs as.
WORKER_ROLES: tuple[str, ...] = ("surveyor", "kernel_writer", "assembler")


@dataclass(frozen=True)
class Decision:
    """One decision of one role."""

    name: str
    #: When the role chooses it, and what its decision file says with it.
    when: str
    #: What the run does next.
    then: str
    #: The attempt ends done (``True``) or blocked for the outer session (``False``).
    finishes: bool | None = None


_READY_WHEN = {
    "surveyor": "both files are written and agree with each other; `summary` says what you "
    "wrote and how many rows of each kind",
    "kernel_writer": "the entry and its test are committed, and the cells that pass are the "
    "ones the entry claims (`CELLS`), `ARCHS` naming this architecture. A cell that fails and "
    "that the first deployment's options do not need is left out of the claim and listed "
    "under `not_done` with its measured numbers; no tolerance is widened",
    "assembler": "the work the card asks is written and committed, and the card's check "
    "passed when you ran it",
}
_READY_THEN = {
    "surveyor": "code makes one task per row and each deployment's tasks; no agent plans",
    "kernel_writer": "your commits land (rebase and fast-forward) and the tasks that wait "
    "for the row start",
    "assembler": "your commits land, and the run measures the deployment's session itself",
}
_BLOCKED = Decision(
    BLOCKED,
    "nothing in this turn can produce the outcome: the entry cannot be used at all, a "
    "prerequisite outside your write scope is missing or wrong, the environment refuses "
    "(a refused job or no holder, never a job still queued), or the task is too large for "
    "one turn. Say so as soon as you know, with `tried`, `command` and its `result` line, "
    "`ruled_out` and `next_question`. Never report a failed check as passed; a failed "
    "check the task can leave out goes under `not_done` of a `ready` receipt instead",
    "the session that watches the run is told at once, with what you wrote; every task "
    "that does not depend on yours goes on. It answers with a retry and a note, a changed "
    "environment or requirement, a split into smaller rows, a wider write scope, or a skip",
    False,
)

DECISIONS_TABLE: dict[str, tuple[Decision, ...]] = {
    role: (Decision(READY, _READY_WHEN[role], _READY_THEN[role], True), _BLOCKED)
    for role in WORKER_ROLES
}


def names(role: str) -> tuple[str, ...]:
    """The decisions ``role`` may write, in the table's order."""
    return tuple(d.name for d in DECISIONS_TABLE[role])


def finishing() -> frozenset[str]:
    """Every decision that ends an attempt done."""
    return frozenset({READY})


def decision_list(role: str) -> str:
    """``role``'s decisions as a list: each with when to choose it and what follows."""
    return "\n".join(f"- `{d.name}`: {d.when}. Then {d.then}." for d in DECISIONS_TABLE[role])


def ending_section(role: str, extra: str = "", *, heading: str = "Ending your turn") -> str:
    """The prompt section that says how ``role`` ends its turn, ``extra`` after the list."""
    body = f"## {heading}\n\nYour decision file carries one of:\n\n{decision_list(role)}"
    return f"{body}\n\n{extra.strip()}" if extra.strip() else body


#: The roles whose decisions are filed under the plan rather than under a unit.
PLAN_ROLES: tuple[str, ...] = ("surveyor",)

#: The one-line reminder at the end of a dispatch; the system prompt carries the protocol.
DECISION_REMINDER = "Before ending your turn, write your decision file to the path below."

#: The closed enum each role's decision file is validated against.
DECISIONS_BY_AGENT: dict[str, tuple[str, ...]] = {role: DECISIONS for role in WORKER_ROLES}

#: The keys besides ``summary`` and ``decision`` a role's decision file carries.
EXTRA_KEYS_BY_AGENT: dict[str, tuple[str, ...]] = {
    "surveyor": BLOCKED_KEYS,
    # `entry` is the committed module's path.
    "kernel_writer": ("entry", *READY_KEYS, *BLOCKED_KEYS),
    "assembler": (*READY_KEYS, *BLOCKED_KEYS),
}


__all__ = [
    "BLOCKED",
    "BLOCKED_KEYS",
    "DECISIONS",
    "DECISIONS_BY_AGENT",
    "DECISIONS_TABLE",
    "DECISION_REMINDER",
    "Decision",
    "EXTRA_KEYS_BY_AGENT",
    "PLAN_ROLES",
    "READY",
    "READY_KEYS",
    "WORKER_ROLES",
    "decision_list",
    "ending_section",
    "finishing",
    "names",
]
