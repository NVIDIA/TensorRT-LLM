# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The section a dispatch carries for each kind of work, in briefing order."""

from __future__ import annotations

from typing import Iterable

from ._render import render_template
from .survey import CAPTURE_LISTS_HEADING

_SURVEY = render_template("tags/survey.md.j2", CAPTURE_LISTS_HEADING=CAPTURE_LISTS_HEADING)

#: Each tag's section of the dispatch.
TAG_PROMPTS: dict[str, str] = {
    "survey": _SURVEY,
    **{
        tag: render_template(f"tags/{tag}.md.j2")
        for tag in ("kernel", "assemble", "weights", "routing", "state", "feature", "investigate")
    },
}


def task_tags(kind: str, tags: Iterable[str] = ()) -> tuple[str, ...]:
    """A task's tags in briefing order: its kind, then its own, none twice."""
    return tuple(dict.fromkeys(tag for tag in (kind, *tags) if tag in TAG_PROMPTS))


def tag_section(tags: Iterable[str]) -> str:
    """Return dispatch sections in order, or an empty string when none is a worker tag."""
    return "\n".join(TAG_PROMPTS[tag] for tag in tags if tag in TAG_PROMPTS)


__all__ = [
    "TAG_PROMPTS",
    "tag_section",
    "task_tags",
]
