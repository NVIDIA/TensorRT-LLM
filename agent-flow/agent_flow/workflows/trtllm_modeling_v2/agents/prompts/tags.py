# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker tag sections in briefing order: what each kind of task asks of its worker."""

from __future__ import annotations

from typing import Iterable

from ...operators.coverage import MAX_ROW_CELLS
from ._common import TIER_DESCRIPTION
from ._render import render_template
from .survey import CAPTURE_LISTS_HEADING

#: Tags another tag brings with it: a feature is work on the target.
IMPLIED: dict[str, tuple[str, ...]] = {"feature": ("assemble",)}

_LADDER = "\n".join(f"{tier}. {text}" for tier, text in sorted(TIER_DESCRIPTION.items()))

_SURVEY = render_template(
    "tags/survey.md.j2",
    ladder=_LADDER,
    CAPTURE_LISTS_HEADING=CAPTURE_LISTS_HEADING,
    MAX_ROW_CELLS=MAX_ROW_CELLS,
)
_KERNEL = render_template("tags/kernel.md.j2")
_ASSEMBLE = render_template("tags/assemble.md.j2")
_FEATURE = render_template("tags/feature.md.j2")
_FIX = render_template("tags/fix.md.j2")

#: Each tag's section of the dispatch.
TAG_PROMPTS: dict[str, str] = {
    "survey": _SURVEY,
    "kernel": _KERNEL,
    "assemble": _ASSEMBLE,
    "feature": _FEATURE,
    "fix": _FIX,
}


def task_tags(kind: str, tags: Iterable[str] = ()) -> tuple[str, ...]:
    """A task's tags in briefing order: each tag after those it brings with it, none twice."""
    ordered: list[str] = []

    def add(tag: str) -> None:
        for implied in IMPLIED.get(tag, ()):
            add(implied)
        if tag in TAG_PROMPTS and tag not in ordered:
            ordered.append(tag)

    for tag in (kind, *tags):
        add(tag)
    return tuple(ordered)


def tag_section(tags: Iterable[str]) -> str:
    """Return dispatch sections in order, or an empty string when none is a worker tag."""
    return "\n".join(TAG_PROMPTS[tag] for tag in tags if tag in TAG_PROMPTS)


__all__ = [
    "IMPLIED",
    "TAG_PROMPTS",
    "tag_section",
    "task_tags",
]
