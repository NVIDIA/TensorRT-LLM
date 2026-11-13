# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every decision a role can end its turn with: when to choose it, and what the run does next.

One table for the three places that must agree on it: the role's prompt,
whose "Ending your turn" section is rendered from it (:func:`ending_section`);
the decision-file validation, whose enum per role is its names
(``run/execution/handoff.py``); and the orchestrator, for which a decision that
``finishes`` a task ends it done and any other ends it failed, back to the
planner (``run/orchestrator.FINISHED``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """One decision of one role."""

    name: str
    #: When the role chooses it, and what its decision file says with it.
    when: str
    #: What the run does next.
    then: str
    #: The task the turn worked on ends done (``True``) or failed, to the planner
    #: (``False``); ``None`` for a role whose turn is not a task's.
    finishes: bool | None = None


_TO_PLANNER = "the task fails to the planner"

DECISIONS: dict[str, tuple[Decision, ...]] = {
    "kernel_writer": (
        Decision(
            "LANDED",
            "the entry is committed, its cells driven and passing on this architecture, "
            "`ARCHS` naming it; name `entry` (its path) and `tier` (the sourcing rung its "
            "committed `__call__` wraps, 1 to 5)",
            "the task is done: its commits are integrated and the checks it claims run",
            True,
        ),
        Decision(
            "FAILED_RECEIPT",
            "the op demonstrably fails at a value the call site needs, or the defect is in "
            "shared code outside the entry, handed back as soon as it is measured with the "
            "`engine_defect` block",
            f"{_TO_PLANNER}. Commits that change only the entry's own files are integrated "
            "as they stand, the rest stay on the task's branch for its next attempt; an "
            "`engine_defect` goes to the planner, which plans its fix and retries this task once "
            "the fix lands",
            False,
        ),
        Decision(
            "MACRO",
            "the callable is more than one kernel-launching call and a better replacement "
            "than the native PyTorch composition exists (a fused kernel to source); list "
            "`constituents`. With nothing better, the composition is the entry: land it with "
            "`macro`",
            f"{_TO_PLANNER} with the constituents, and it adds a kernel task per row",
            False,
        ),
        Decision(
            "BLOCKED",
            "nothing in this turn can move the entry: the environment (a refused job or no "
            "session, never a job still queued), or a prerequisite outside it; say which",
            f"{_TO_PLANNER} with your summary",
            False,
        ),
    ),
    "assembler": (
        Decision(
            "UNIT_READY",
            "the work the task asks is written and committed, and you expect its checks to pass",
            "the task is done: its branch is integrated and the checks it claims run; a "
            "check that fails comes back to the planner",
            True,
        ),
        Decision(
            "BLOCKED",
            "the work cannot go on in this turn: a callable no catalog entry covers "
            "(`blocked_reason: vocabulary_gap`, the callable in `missing_entry`), or what "
            "you lack (`knowledge_gap`), an accuracy you cannot close (`accuracy_shortfall`) "
            "or the site (`environment`: a refused job or no session, never a job still "
            "queued). Block as soon as you know",
            f"{_TO_PLANNER} at once with `blocked_reason`, `missing_entry` and your "
            "summary: it adds the missing entry's kernel task and retries yours, or "
            "re-plans",
            False,
        ),
    ),
    "surveyor": (
        Decision(
            "DRAFT_READY",
            "both files are written and coherent; `summary` names what you wrote and the "
            "counts by state",
            "the survey is done: the orchestrator adds a kernel task per queue row, and the "
            "planner wires them",
            True,
        ),
        Decision(
            "CEILING",
            "a computation step has no server anywhere: the step, what you searched and why "
            "go in `vocabulary.yaml`'s `ceiling` block",
            f"{_TO_PLANNER}, which escalates the step or re-plans around it",
            False,
        ),
    ),
    "planner": (
        Decision(
            "GRAPH_UPDATED",
            "the graph is written; `summary` gives the reason for the change in one paragraph",
            "the orchestrator validates the revision and adopts it, or sends it back with why",
        ),
        Decision(
            "DONE",
            "you believe every required check holds",
            "the orchestrator verifies, and delivers once full support holds",
        ),
        Decision(
            "ESCALATE",
            "a requirement cannot be met as confirmed; name it and the evidence",
            "the operator is asked; the work that does not wait on the answer goes on, and the "
            "run ends escalated once nothing else can move",
        ),
        Decision(
            "CEILING",
            "a computation step has no implementation anywhere, with the evidence",
            "the run ends at the ceiling",
        ),
    ),
    "tuner": (
        Decision(
            "CANDIDATE",
            "a change is committed on its branch with its correctness tests and micro-bench",
            "the orchestrator measures it end to end and keeps it only on its own gate",
        ),
        Decision(
            "DROPPED",
            "the micro-bench numbers that killed the hypothesis",
            "the campaign moves on",
        ),
        Decision(
            "BLOCKED",
            "what is broken, with job receipts, or the runtime change or native rebuild "
            "the candidate needs, with its evidence",
            "the campaign records it",
        ),
    ),
}


def names(role: str) -> tuple[str, ...]:
    """The decisions ``role`` may write, in the table's order."""
    return tuple(d.name for d in DECISIONS[role])


def finishing() -> frozenset[str]:
    """Every role decision that ends its task done."""
    return frozenset(d.name for table in DECISIONS.values() for d in table if d.finishes)


def decision_list(role: str) -> str:
    """``role``'s decisions as a list: each with when to choose it and what follows."""
    return "\n".join(f"- `{d.name}`: {d.when}. Then {d.then}." for d in DECISIONS[role])


def ending_section(role: str, extra: str = "", *, heading: str = "Ending your turn") -> str:
    """The prompt section that says how ``role`` ends its turn, ``extra`` after the list."""
    body = f"## {heading}\n\nYour decision file carries one of:\n\n{decision_list(role)}"
    return f"{body}\n\n{extra.strip()}" if extra.strip() else body


#: The roles whose decisions are filed under the plan rather than under a unit.
PLAN_ROLES: tuple[str, ...] = ("surveyor", "planner")

#: The one-line reminder at the end of a dispatch; the system prompt carries the protocol.
DECISION_REMINDER = "Before ending your turn, write your decision file to the path below."

#: The closed enum each role's decision file is validated against: its row of the
#: decision table (``agents/decisions.py``).
DECISIONS_BY_AGENT: dict[str, tuple[str, ...]] = {
    role: names(role) for role in (*PLAN_ROLES, "kernel_writer", "assembler", "tuner")
}

#: The keys besides ``summary`` and ``decision`` a role's decision file carries for the
#: orchestrator to read.
EXTRA_KEYS_BY_AGENT: dict[str, tuple[str, ...]] = {
    # `constituents` is a MACRO's decomposition, the work that comes next.
    # `engine_defect` is a failed receipt's defect outside the entry
    # (`ENGINE_DEFECT_KEYS`): the planner hears it with the task's end and
    # plans the fix.
    # `uncertified` is what a landed entry still does not certify of what the
    # target serves: the planner hears it with the task's end.
    # `macro` is a landed entry that composes native PyTorch calls, nothing better existing:
    # what it composes and where its writer looked.
    "kernel_writer": (
        "entry",
        "receipt",
        "constituents",
        "engine_defect",
        "uncertified",
        "macro",
    ),
    # `uncertified` names the calls a unit makes, correct, outside their entries'
    # CELLS: the planner hears them with the task's end and adds the certification.
    "assembler": ("blocked_reason", "missing_entry", "uncertified"),
    # The orchestrator re-reads the candidate's own decision file for the rest
    # (micro-bench table, projection).
    "tuner": ("slug", "kind", "worktree", "branch", "commits", "projection"),
}


__all__ = [
    "DECISION_REMINDER",
    "PLAN_ROLES",
    "EXTRA_KEYS_BY_AGENT",
    "DECISIONS_BY_AGENT",
    "DECISIONS",
    "Decision",
    "decision_list",
    "ending_section",
    "finishing",
    "names",
]
