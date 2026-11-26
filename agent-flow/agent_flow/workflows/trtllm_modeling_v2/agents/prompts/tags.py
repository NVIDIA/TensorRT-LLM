# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker tag sections in briefing order, and the summaries the planner chooses them by."""

from __future__ import annotations

from typing import Iterable

from ...operators.coverage import MAX_ROW_CELLS, TIER_DESCRIPTION
from ._render import render_template
from .survey import CAPTURE_LISTS_HEADING

#: What the planner chooses a task's kind and tags by: one line per tag.
TAG_SUMMARIES: dict[str, str] = {
    "survey": "map the checkpoint's computation steps onto the catalog: `plan.md` and "
    "`vocabulary.yaml`",
    "kernel": "one catalog entry: onboard a missing one, or verify an existing one at new values",
    "assemble": "the target itself: modeling, weights, routing",
    "feature": "one feature, or one configuration variant, on a target whose core is built",
    "integrate": "land work that did not integrate on its own, resolving its conflicts",
    "investigate": "find the root cause of a failing check; writes a diagnosis, never the target",
}

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
_INTEGRATE = render_template("tags/integrate.md.j2")
_INVESTIGATE = render_template("tags/investigate.md.j2")

#: Each tag's section of the dispatch.
TAG_PROMPTS: dict[str, str] = {
    "survey": _SURVEY,
    "kernel": _KERNEL,
    "assemble": _ASSEMBLE,
    "feature": _FEATURE,
    "integrate": _INTEGRATE,
    "investigate": _INVESTIGATE,
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
    "TAG_SUMMARIES",
    "tag_section",
    "task_tags",
]
