# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The card of a task: what one worker's turn is asked, rendered from the task's row.

A card says three things, in under :data:`CARD_WORDS` words: what the task is for (the
row's outcome, the values that matter, what is required and what is optional, and
reference code to search for: the engine's own model of the architecture first, then the
row's leads); what to hand back (commits inside the write scope, and
the receipt); and how it is judged (the one check, by its registry id, with its command
and its job line filled in).

It names no location in the engine: a place somebody wrote into a row is turned into the
name to search for (:func:`~.paths.as_search`). The bound is kept by code: a list is cut
at :data:`CARD_ITEMS` items, a line at :data:`ITEM_WORDS` words, and a cell is shown by
what differs from the first one; the row's file (``work.yaml``) holds everything whole,
and the card says so when it cut.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...operators import prior_art
from ...operators.rows import DELIVERY_ROW, WORK_FILENAME, Row
from ..decisions import BLOCKED_KEYS, READY_KEYS
from .paths import as_search

#: The most items of one list a card shows; the row's file holds the rest.
CARD_ITEMS = 8
#: The most words of one item a card shows.
ITEM_WORDS = 45
#: What a card stays under, whatever its row holds.
CARD_WORDS = 600
#: The words of the check's command that count toward that: the command is never cut.
COMMAND_WORDS = 40
#: The most tasks skipped before this one that a card names.
SKIPPED_ITEMS = 4
#: A card names the engine's own model of the architecture while it shows this many leads.
ENGINE_MODEL_ROOM = 2


@dataclass(frozen=True)
class _Room:
    """How much of each part of a row one rendering of its card shows: (items, words an item)."""

    required: tuple[int, int] = (CARD_ITEMS, ITEM_WORDS)
    optional: tuple[int, int] = (4, ITEM_WORDS)
    leads: tuple[int, int] = (6, 35)
    #: The values that matter, and the prose lines (the outcome, the last attempt's).
    axes: int = ITEM_WORDS
    prose: int = 60


#: How a card that is too long is cut, least important first: the leads and the optional
#: lines give way before a required line loses a word. The last step fits any row.
_LEAST = {"optional": (1, 8), "leads": (1, 8), "axes": 8, "prose": 18}
_TIGHTER: tuple[_Room, ...] = (
    _Room(),
    _Room(optional=(3, 30), leads=(4, 25)),
    _Room(optional=(3, 20), leads=(3, 18), axes=25),
    _Room(optional=(2, 14), leads=(2, 14), axes=15, prose=40),
    # Everything but the required lines at its least, before a required line loses a word.
    _Room(**_LEAST),
    _Room(required=(8, 30), **_LEAST),
    _Room(required=(6, 22), **_LEAST),
    _Room(required=(4, 14), **_LEAST),
    _Room(required=(2, 8), optional=(1, 6), leads=(1, 6), axes=6, prose=12),
)

#: What a row of each kind is told to look for when it states no lead of its own: the
#: behaviour to find, as a search. Never a place.
REFERENCE: dict[str, str] = {
    "operator": "an entry of the same category that wraps the same kind of kernel, with its "
    "test: its `CELLS`, its `reference` and how its test drives them",
    "weights": "the loader of a target whose checkpoint has this weight format and layer "
    "geometry: how it maps, transforms and shares out each key",
    "routing": "how an existing family registers its route and words a `NoTarget` refusal",
    "state": "a model and a single-process transfer test that carry this cache kind between "
    "unequal layouts: its role declarations, its reshaping and its assertions",
    "feature": "the engine's own test that turns this option on, and a target that already "
    "serves it: what its forward reads from the metadata",
    "assemble": "a target of the same family, or one with this attention and MoE kind: its "
    "lifecycle, its forward's sequence of operator calls and its routing",
}

#: The kinds of row whose files are the target's own modules, and the tasks that wire them
#: into the model: a deployment's assembly, or the two sides of a disaggregated one. A row
#: whose task waits for one of those runs after it (``needs: model``, or it waits for such a
#: row); every other row of these kinds is waited for by an assembly.
_WIRED_KINDS = ("weights", "routing", "state", "feature")
_ASSEMBLIES = ("assemble-", "pd-ctx-", "pd-gen-")

#: What "required" asks, said on the card. A kernel row lands what it certified: a cell
#: that fails is left out of the entry's claim, and only one the first deployment needs
#: makes the row blocked. A weak model once read "required" as all or nothing and blocked
#: a row that had certified seventeen of twenty cells.
_MUST_HOLD = "each must hold before you write `ready`"
_REQUIRED: dict[str, str] = {
    "operator": "certify each as written; one that fails is left out of `CELLS` and goes "
    "under `not_done` with its numbers: still `ready`, unless the first deployment needs "
    "that cell",
    # A weak surveyor asked a feature row for five assertions, and its worker spent most of
    # an hour on a harness of its own after the deployment's session had passed.
    "feature": _MUST_HOLD + ", shown by the smallest test that would fail without your change",
}

_PAIR = re.compile(r", (?=[A-Za-z_][\w.]*: )")


def _clip(text: str, words: int) -> tuple[str, bool]:
    parts = text.split()
    if len(parts) <= words:
        return text, False
    return " ".join(parts[:words]) + " …", True


def _pairs(line: str) -> dict[str, str] | None:
    """A cell written as ``key: value, key: value``, as its pairs; ``None`` for anything else.

    Prose, a line that states a key twice, and a kernel row's ``surface: ...`` line are
    not cells: each is shown as it was written.
    """
    if line.lstrip().lower().startswith("surface:"):
        return None
    found: dict[str, str] = {}
    for part in _PAIR.split(line):
        key, sep, value = part.partition(": ")
        key = key.strip()
        if not sep or key in found or not re.fullmatch(r"[A-Za-z_][\w.]*", key):
            return None
        found[key] = value.strip()
    return found if len(found) >= 3 else None


def _by_difference(items: Sequence[str]) -> list[str]:
    """``items`` with each cell after the first shown by the keys that differ from the first.

    Only a cell of the same shape as the first: one that shares fewer than half of the
    first one's keys is something else, and is shown whole.
    """
    base: Mapping[str, str] | None = None
    out = []
    for item in items:
        cell = _pairs(item)
        if cell is None:
            out.append(item)
        elif base is None:
            base = cell
            out.append(item)
        elif 2 * len(set(cell) & set(base)) < len(base):
            out.append(item)
        else:
            changed = [f"{k}: {v}" for k, v in cell.items() if base.get(k) != v]
            dropped = [k for k in base if k not in cell]
            said = ", ".join(changed + [f"no {k}" for k in dropped]) or "nothing"
            out.append(f"as the first cell, but {said}")
    return out


class _Card:
    """The card's lines, and whether anything was cut from them."""

    def __init__(self, scope: Sequence[str], room: _Room) -> None:
        self.lines: list[str] = []
        self.cut = False
        self.scope = tuple(scope)
        self.room = room

    def clip(self, text: str, words: int) -> str:
        text, cut = _clip(as_search(" ".join(str(text).split()), self.scope), words)
        self.cut = self.cut or cut
        return text

    def items(self, title: str, items: Sequence[str], room: tuple[int, int], note: str) -> None:
        if not items:
            return
        most, words = room
        shown = [self.clip(item, words) for item in _by_difference(list(items))[:most]]
        if len(items) > most:
            shown.append(f"and {len(items) - most} more")
            self.cut = True
        self.lines += [f"**{title}** ({note}):", *[f"- {item}" for item in shown], ""]


def _reference(row: Row, card: _Card, engine: str = "") -> list[str]:
    """The reference code a card offers: what to look for, and why each lead may fit.

    ``engine`` is the engine's own model of the checkpoint's architecture, as a search:
    the first lead of a card that has room for it. Then the row's own leads.
    """
    most, words = card.room.leads
    lines = [
        "**Reference code** (leads to search for with `git grep -n`: where to look, not a "
        "verdict; a name not found has moved, it never means the thing is unsupported):",
        f"- look for {REFERENCE.get(row.kind, REFERENCE['assemble'])}",
    ]
    if engine and most >= ENGINE_MODEL_ROOM:
        lines.append(f"- {as_search(engine, card.scope)}")
    lines += [f"- lead: {card.clip(lead, words)}" for lead in row.leads[:most]]
    if len(row.leads) > most:
        card.cut = True
    return [*lines, ""]


def render_card(
    task: Any,
    row: Row | None,
    *,
    command: str,
    notes: str | Path,
    spec: str | Path,
    skipped: Sequence[str] = (),
    engine: str | None = None,
) -> str:
    """The card of ``task`` from ``row``; a task with no row (the survey) gets its outcome and the rest.

    ``engine`` is the engine's own model of the checkpoint's architecture as one lead
    (``prior_art.reference_for``): what to copy from and compare against, and never
    import. Unsaid, it is read from what the run kept beside the notes file
    (``prior_art.PRIOR_ART_DATA``); a card with no room for leads leaves it out. Always
    under :data:`CARD_WORDS` words: a row too long for that is cut further, least
    important first, and the card says where the whole of it is.
    """
    if engine is None and row is not None:
        kept = prior_art.load(Path(notes).parent / prior_art.PRIOR_ART_DATA)
        engine = prior_art.reference_for(kept)
    text = ""
    for room in _TIGHTER:
        text = _render(task, row, command, notes, spec, skipped, room, engine or "")
        if _words(text, command) < CARD_WORDS:
            break
    return text


def _words(text: str, command: str) -> int:
    """The card's words, a command counted as at most :data:`COMMAND_WORDS`.

    The check's command is given whole whatever its length (a row with many test files
    names each): it is run, not read, and is the one part a card never cuts.
    """
    return len(text.split()) - max(0, len(command.split()) - COMMAND_WORDS)


def _render(
    task: Any,
    row: Row | None,
    command: str,
    notes: str | Path,
    spec: str | Path,
    skipped: Sequence[str],
    room: _Room,
    engine: str = "",
) -> str:
    card = _Card(tuple(task.touches or ()), room)
    lines = card.lines
    lines += [
        f"## Your task: `{task.id}` ({task.kind}, attempt {task.attempts} of "
        f"{max(task.budget, task.attempts)})",
        "",
        f"**What it is for.** {card.clip(task.goal, room.prose)}",
        "",
    ]
    inputs = [f"the spec `{spec}`"]
    if row is not None and row.deployment:
        inputs.append(
            f"deployment `{row.deployment}`" + (f", its {row.side} side" if row.side else "")
        )
    if row is not None and row.config_axes:
        axes = ", ".join(f"{k}={v}" for k, v in row.config_axes.items())
        inputs.append(f"the values that matter here: {card.clip(axes, room.axes)}")
    if task.depends_on:
        inputs.append("the receipts of the tasks you depend on, below")
    lines += [f"**Inputs.** {'; '.join(inputs)}.", ""]
    if row is not None:
        # A row with no check runs no test, so it shows nothing by one.
        must = _REQUIRED.get(row.kind, _MUST_HOLD) if task.check else _MUST_HOLD
        card.items("Required", row.required, room.required, must)
        card.items(
            "Optional",
            row.optional,
            room.optional,
            "try each once; a failure goes under `not_done` with its numbers and never "
            "stops or blocks you",
        )
        lines += _reference(row, card, engine)
    back = []
    if task.touches:
        shown = ", ".join(f"`{p}`" for p in task.touches[:CARD_ITEMS])
        if len(task.touches) > CARD_ITEMS:
            shown += f" and {len(task.touches) - CARD_ITEMS} more (the task envelope lists them)"
        owns_init = any(Path(p).name == "__init__.py" for p in task.touches)
        back.append(
            f"Commits that change only {shown}: your write scope. A commit outside is sent "
            "back once. "
            + (
                "An `__init__.py` this scope lists is yours; no other one is. "
                if owns_init
                else "No shared initializer is yours to change: the skeleton is committed. "
            )
            + "A file you need outside: say which and why in a `blocked` receipt."
        )
        before = not any(str(d).startswith(_ASSEMBLIES) for d in task.depends_on or ())
        # Said only of a scope that leaves the model's file out: one that names it, or a
        # directory (the whole target), holds it.
        apart = not any(Path(p).name == "modeling.py" or not Path(p).suffix for p in task.touches)
        wired = row is not None and row.kind in _WIRED_KINDS and row.id != DELIVERY_ROW
        if wired and before and apart:
            which = f"of `{row.deployment}`" if row.deployment else "that follows"
            back.append(
                f"The model's own files are not in it on purpose: the assembly {which} runs "
                "after your row and wires what you wrote into the model. End at your own "
                "check, and say in your receipt what the assembly must call."
            )
    back.append(
        f"A receipt: `ready` carries {', '.join(f'`{k}`' for k in READY_KEYS)} (under "
        "`not_done` only what this card asked and you did not do); `blocked` carries "
        f"{', '.join(f'`{k}`' for k in BLOCKED_KEYS)}."
    )
    lines += [f"**Hand back.** {' '.join(back)}", ""]
    if task.check:
        job = f", planned as `{row.job_line}`" if row is not None else ""
        lines += [
            f"**How it is judged.** By the check `{task.check}`. Run it last, after your last "
            "edit, exactly as given, and watch it end:",
            "",
            f"    {command}",
            "",
            f"The command queues a job of the run's pool on a pin of your tree{job}, waits "
            "for it, and records its result in the run's ledger for that pin: you submit "
            "nothing for it, and your task is accepted only on a pass "
            "for the tree you hand in. `partial` (exit 3: something could not be observed) "
            "is reported as not verified and lands nothing as passed: say it, do not work "
            "around it. Never write a launch line. Anything else that needs the engine's "
            "Python is a pool job of your own with `--tree`: your own shell may not have it.",
            "",
        ]
    elif row is not None:
        # A row with no check (the delivery's files): one read the flow's source to learn this.
        lines += [
            "**How it is judged.** By no command of yours, and `ready` needs none. When you "
            "hand in, code reads the files you changed: an entry is an `OpWrapper` that states "
            "its contract, with its test where CI matches it; the accuracy test for CI, where "
            "it is yours, is there and a test list names it. What does not hold now is listed "
            "in your dispatch (no list: nothing); what does not hold then comes back to you "
            "once, and the delivery asks it again when the run closes.",
            "",
        ]
    own = Path(notes).parent / "tasks" / f"{task.id}.notes.md"
    lines += [
        f"**Notes.** Read `{notes}` first, if it exists, and again before your receipt. Yours, "
        f"to read first and keep: `{own}`."
    ]
    if getattr(task, "last", ""):
        lines += ["", f"**Your previous attempt.** {card.clip(task.last, room.prose)}"]
    if skipped:
        # A skip's note is somebody's free text: bounded, and a place in it becomes a name.
        said = [card.clip(item, max(room.prose // 2, 8)) for item in skipped[:SKIPPED_ITEMS]]
        if len(skipped) > SKIPPED_ITEMS:
            said.append(f"and {len(skipped) - SKIPPED_ITEMS} more")
        lines += [
            "",
            f"**Skipped before you.** {'; '.join(said)}. Go on without it, and list under "
            "`not_done` what you could not do for lack of it.",
        ]
    if card.cut and row is not None:
        lines += [
            "",
            f'**Cut to fit.** What ends in `…` or in "more" is whole in the row `{row.id}` '
            f"of `{Path(notes).parent / WORK_FILENAME}`: read it before you write.",
        ]
    return "\n".join(lines).rstrip()


def scope_note(outside: Sequence[str], scope: Sequence[str]) -> str:
    """What a worker is told when its commits reach outside its write scope."""
    listed = "\n".join(f"- `{path}`" for path in outside)
    return (
        "## Your commits reach outside your write scope\n\n"
        f"Your branch changes these paths, which are not yours:\n\n{listed}\n\n"
        f"Your write scope is {', '.join(f'`{p}`' for p in scope)}. Take those changes off "
        "your branch (amend or add a commit that restores the files as the head has them) "
        "and keep the work inside your scope. If the outcome cannot be produced without "
        "one of them, write a `blocked` receipt whose `next_question` names the path and "
        "why: the session that watches the run can widen your scope."
    )


__all__ = [
    "CARD_ITEMS",
    "CARD_WORDS",
    "COMMAND_WORDS",
    "ITEM_WORDS",
    "REFERENCE",
    "render_card",
    "scope_note",
]
