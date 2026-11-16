# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared doctrine and run-scoped prompt context."""

from __future__ import annotations

import re
from typing import Sequence

from ._render import render_template


def tidy(text: str) -> str:
    """Collapse the blank-line runs that composing blocks leaves behind."""
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


MODELING_V2_NATIVE = render_template("common/modeling_v2_native.md.j2")


#: The form of each key a decision file may carry beside ``decision`` and ``summary``: what a
#: worker writes after the colon. A key with no form here is shown as free text.
KEY_FORMS: dict[str, str] = {
    "entry": "<the committed module's path, as your write scope names it>",
    "check": '{id: "<the card\'s check id>", command: "<as you ran it>", result: "<its last line>"}',
    "references": '[{symbol: "<name>", commit: "<sha>", where: "<the file then>", why: "<why it fits>"}]',
    "not_done": '["<what the card asked that you did not do, with its measured numbers>"]',
    "tried": '["<what you tried, one line each>"]',
    "command": '"<the one command that shows the problem>"',
    "result": '"<its result line>"',
    "ruled_out": '["<a hypothesis and what ruled it out>"]',
    "next_question": '"<the word that names what answers it>: <your one question, and what answers it>"',
}


def decision_file_block(agent: str, extra: Sequence[str] = ()) -> str:
    """Render the decision-file protocol; decisions come from the role's ending section."""
    forms = [(key, KEY_FORMS.get(key, '"<text>"')) for key in extra]
    return render_template("common/decision_file.md.j2", agent=agent, extra=forms)


def notes_block(notes: str) -> str:
    """Render the task's notes: the user's own guidance, verbatim, to every role."""
    if not notes or not notes.strip():
        return ""
    return render_template("common/notes.md.j2", notes=notes.strip())


def scope_block(scope: str) -> str:
    """Render the target's scope when it is narrower than the checkpoint."""
    if scope != "text_only":
        return ""
    return render_template("common/text_only.md.j2")


__all__ = [
    "KEY_FORMS",
    "MODELING_V2_NATIVE",
    "decision_file_block",
    "notes_block",
    "scope_block",
    "tidy",
]
