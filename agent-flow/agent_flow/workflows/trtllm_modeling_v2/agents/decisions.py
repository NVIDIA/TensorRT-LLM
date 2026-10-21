# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every decision a role can end its turn with: when to choose it, and what the run does next.

One table for the three places that must agree on it: the role's prompt,
whose "Ending your turn" section is rendered from it (:func:`ending_section`);
the decision-file validation, whose enum per role is its names
(``run/progress.py``); and the orchestrator, for which a decision that
``finishes`` a task ends it done and any other ends it failed, back to the
planner (``run/orchestrator.FINISHED``). A prompt that names a decision the
run does not handle, or describes what it no longer does, cannot drift from
it: the text is this table's.
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
    "smith": (
        Decision(
            "LANDED",
            "the entry is committed, its cells driven and passing on this architecture, "
            "`ARCHS` naming it; name `entry` and `tier`",
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
            "`engine_defect` is registered and the orchestrator adds the task that fixes it",
            False,
        ),
        Decision(
            "MACRO",
            "the callable is more than one kernel-launching call; list `constituents`",
            f"{_TO_PLANNER} with the constituents, and it adds a kernel task per row",
            False,
        ),
        Decision(
            "BLOCKED",
            "nothing in this turn can move the entry: the environment, or a prerequisite "
            "outside it; say which",
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
            "or the site (`environment`). Block as soon as you know",
            f"{_TO_PLANNER} at once with `blocked_reason`, `missing_entry` and your "
            "summary: it adds the missing entry's kernel task and retries yours, or "
            "re-plans",
            False,
        ),
    ),
    "reviewer": (
        Decision(
            "APPROVE",
            "a unit review finds nothing a reader can see and a gate cannot; a regression "
            "review's `tests compare` says `CLEAN` and the boots agree (put `cleared: <head "
            "commit>` in the decision file)",
            "the review is done",
            True,
        ),
        Decision(
            "REJECT",
            "a defect, each item citing the file, the symbol and the invariant it violates; "
            "or a regression, each with its test id, its verbatim failure and the change it "
            "traces to",
            f"{_TO_PLANNER} with your items: it adds the fixes",
            False,
        ),
    ),
    "plan_reviewer": (
        Decision(
            "APPROVE",
            "the plan and the vocabulary hold; still list what you noticed",
            "the review is done",
            True,
        ),
        Decision(
            "REJECT",
            "something would waste a task's turn or produce a wrong gate: specific, "
            "actionable items, each naming the file and the row",
            f"{_TO_PLANNER} with your items; it reshapes the work not yet finished, and what "
            "is already running goes on",
            False,
        ),
    ),
    "plan_drafter": (
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
            "the run returns to intake",
        ),
        Decision(
            "CEILING",
            "a computation step has no implementation anywhere, with the evidence",
            "the run ends at the ceiling",
        ),
    ),
    "improver": (
        Decision(
            "LESSONS_PROPOSED",
            "you wrote the proposal file",
            "Python keeps what is admissible and shows your next turn what it refused",
        ),
        Decision(
            "NOTHING_NEW",
            "nothing here would help a later worker or the operator. A good answer: a store "
            "full of noise is worse than an empty one",
            "nothing is stored",
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
            "NEED_SMITH",
            "the op, its tier and the cells it needs",
            "the campaign asks for the entry",
        ),
        Decision(
            "NEED_RUNTIME",
            "the runtime change the candidate needs",
            "the campaign records it",
        ),
        Decision(
            "NEED_REBUILD",
            "the native rebuild the candidate needs",
            "the campaign records it",
        ),
        Decision(
            "BLOCKED",
            "what is broken, with job receipts",
            "the campaign records it",
        ),
        Decision(
            "STOP",
            "the profile evidence that nothing left is worth a candidate",
            "the campaign ends",
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


__all__ = [
    "DECISIONS",
    "Decision",
    "decision_list",
    "ending_section",
    "finishing",
    "names",
]
