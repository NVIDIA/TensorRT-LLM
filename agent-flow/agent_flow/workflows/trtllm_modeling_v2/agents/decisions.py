# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every decision a role can end its turn with, and the file it is written to.

One table for the places that must agree on it: the role's prompt, whose
"Ending your turn" section is rendered from it (:func:`ending_section`); the
decision-file validation, whose enum per role is its names
(:func:`read_decision`); and the pipeline, for which a decision that
``finishes`` a task lets it land and any other ends the attempt
(``run/pipeline.py``).

A turn reports by writing one YAML file at the one path its dispatch names
(:func:`decision_path`: the task and its attempt). The pipeline reads it there
and nowhere else: a turn that wrote no valid decision has none.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DECISIONS_DIRNAME = "decisions"


@dataclass(frozen=True)
class Decision:
    """One decision of one role."""

    name: str
    #: When the role chooses it, and what its decision file says with it.
    when: str
    #: What the run does next.
    then: str
    #: The worker did what the task asked (``True``), or the attempt ends here (``False``).
    finishes: bool = False


_FAILS = (
    "the attempt ends: the task runs again with what you wrote as its note, and once its "
    "attempts are spent it is reported not done and the run goes on without it"
)

DECISIONS: dict[str, tuple[Decision, ...]] = {
    "kernel_writer": (
        Decision(
            "LANDED",
            "the entry is committed, its cells driven and passing on this architecture, "
            "`ARCHS` naming it; name `entry` (its path) and `tier` (the sourcing rung its "
            "committed `__call__` wraps, 1 to 5)",
            "the task is done: its commits are audited and integrated",
            True,
        ),
        Decision(
            "FAILED_RECEIPT",
            "the op demonstrably fails at a value the call site needs, or the defect is in "
            "shared code outside the entry, handed back as soon as it is measured with the "
            "`engine_defect` block",
            f"{_FAILS}. Nothing is integrated; your commits stay on the task's branch for its "
            "next attempt",
            False,
        ),
        Decision(
            "BLOCKED",
            "nothing in this turn can move the entry: the environment, a prerequisite outside "
            "it, or a better fused replacement that must be sourced first (name its entries "
            "in `summary`); say which",
            f"{_FAILS}",
            False,
        ),
    ),
    "assembler": (
        Decision(
            "UNIT_READY",
            "the work the task asks is written and committed, and every check of the unit you "
            "ran yourself passed; list what you ran under `checks`",
            "its branch is audited and lands. For the first deployment the run boots and "
            "scores it on your committed worktree first, and nothing lands unless that "
            "passes; a later deployment's checks run once in the final pass",
            True,
        ),
        Decision(
            "BLOCKED",
            "the work cannot go on in this turn: what you lack (`blocked_reason: "
            "knowledge_gap`), an accuracy you cannot close (`accuracy_shortfall`) or the host "
            "(`environment`). A callable no catalog entry covers is no reason to block: write "
            "its wrapper in the target's own `ops/`. Block as soon as you know",
            f"{_FAILS}",
            False,
        ),
    ),
    "surveyor": (
        Decision(
            "DRAFT_READY",
            "both files are written and coherent; `summary` names what you wrote and the "
            "counts by state",
            "the survey is done: the pipeline adds one kernel task per queue row and one "
            "task per deployment, in the spec's order",
            True,
        ),
        Decision(
            "CEILING",
            "a computation step has no server anywhere: the step, what you searched and why "
            "go in `vocabulary.yaml`'s `ceiling` block",
            f"{_FAILS}",
            False,
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


#: The one-line reminder at the end of a dispatch; the system prompt carries the protocol.
DECISION_REMINDER = "Before ending your turn, write your decision file to the path below."

#: The closed enum each role's decision file is validated against.
DECISIONS_BY_AGENT: dict[str, tuple[str, ...]] = {role: names(role) for role in DECISIONS}

#: The keys besides ``summary`` and ``decision`` a role's decision file carries.
EXTRA_KEYS_BY_AGENT: dict[str, tuple[str, ...]] = {
    # `engine_defect` is a failed receipt's defect outside the entry; `uncertified` is what a
    # landed entry still does not certify of what the target serves; `macro` is a landed
    # entry that composes native PyTorch calls, nothing better existing.
    "kernel_writer": ("entry", "receipt", "engine_defect", "uncertified", "macro"),
    # `checks` are the unit's checks the worker ran itself, each with its result;
    # `uncertified` names the calls a unit makes, correct, outside their entries' CELLS.
    "assembler": ("checks", "blocked_reason", "uncertified"),
}


# -------------------------------------------------------------------------- the file


def decision_path(workspace: Path, task_id: str, attempt: int, turn: int = 1) -> Path:
    """``decisions/<task>-<attempt>[.<turn>].yaml``: where one turn writes its decision.

    A task's attempt has one path, so a restarted run finds the answer an
    earlier process's turn left, and no attempt reads another's. ``turn``
    numbers the short turns after the first within one attempt (a worktree to
    put right before it integrates).
    """
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
    token = f"{attempt}.{turn}" if turn > 1 else str(attempt)
    directory = Path(workspace) / DECISIONS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}-{token}.yaml"


def decision_file_instruction(path: Path) -> str:
    """The block a dispatch prompt ends with: the one exact path this turn writes."""
    return (
        "---\n\n"
        "**Write your decision file to exactly this path when you finish:**\n\n"
        f"`{path}`\n\n"
        "Nothing downstream reads your prose. This file is how this turn "
        "reaches the orchestrator.\n"
    )


def read_decision(path: Path, role: str) -> dict[str, Any] | None:
    """What ``role`` wrote at ``path`` when it holds a decision ``role`` may give; else ``None``."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    if not isinstance(raw, dict):
        return None
    decision = str(raw.get("decision", "")).strip()
    if decision not in DECISIONS_BY_AGENT[role]:
        return None
    return {**raw, "decision": decision}


def decision_problem(path: Path, role: str) -> str:
    """Why ``path`` holds no decision ``role`` may give, in a sentence; ``""`` when it does."""
    allowed = ", ".join(DECISIONS_BY_AGENT[role])
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"there is no file at `{path}`"
    except (OSError, UnicodeError) as exc:
        return f"`{path}` cannot be read: {exc}"
    except yaml.YAMLError as exc:
        return f"`{path}` is not valid YAML: {clip(str(exc), 300)}"
    if not isinstance(raw, dict):
        return f"`{path}` is not a YAML mapping of keys to values"
    if "decision" not in raw:
        return f"`{path}` has no `decision` key; it must be one of {allowed}"
    decision = str(raw.get("decision", "")).strip()
    if decision not in DECISIONS_BY_AGENT[role]:
        return f"`decision` is {decision!r} in `{path}`; it must be one of {allowed}"
    return ""


def decision_repair_block(problem: str) -> str:
    """What a worker whose turn ended without a readable decision is told, for one short turn."""
    return (
        "## Your decision file is not valid\n\n"
        f"Your turn ended, but {problem}. Do not do the task again: your work is in your "
        "worktree. Write the decision file again, as YAML, with `decision:` and the keys your "
        "decision carries (`summary` and the others the ending section of your instructions "
        "names), saying what your earlier turn did. The orchestrator reads only this file."
    )


def clip(text: str, limit: int) -> str:
    """``text`` up to ``limit`` characters, an ellipsis marking what was cut."""
    text = str(text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def first_paragraph(text: Any, limit: int = 400) -> str:
    """The first paragraph of ``text`` as one line, clipped to ``limit``."""
    for paragraph in re.split(r"\n\s*\n", str(text or "").strip()):
        line = " ".join(paragraph.split())
        if line:
            return clip(line, limit)
    return ""


__all__ = [
    "DECISIONS",
    "DECISIONS_BY_AGENT",
    "DECISIONS_DIRNAME",
    "DECISION_REMINDER",
    "EXTRA_KEYS_BY_AGENT",
    "Decision",
    "clip",
    "decision_file_instruction",
    "decision_list",
    "decision_path",
    "ending_section",
    "finishing",
    "first_paragraph",
    "names",
    "read_decision",
]
