# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A unit's measured anchor: its own first benchmark score, once its reference is a placeholder.

A model with no written score and no oracle that can run it starts from a low placeholder
(e.g. 10 on GSM8K), which lets a working target integrate and catches one that answers
nothing. The first time the unit's benchmark passes on it at a member, the score it
measured there becomes the unit's anchor at that member: every later benchmark of the
unit there (on a later tree, after a fix) is held to it, as a variant is to its unit's own
score (``anchor: relative``). ``anchors.json`` in the workspace keeps them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...infra.files import write_text_atomic

FILENAME = "anchors.json"


@dataclass(frozen=True)
class Anchor:
    score: float
    #: The tree the score was measured on (``git describe``).
    tree: str


def _key(unit: str, gate: str, member: str | None) -> str:
    return f"{unit}:{gate}@{member or ''}"


def _read(workspace: Path) -> dict[str, dict]:
    try:
        data = json.loads((Path(workspace) / FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def measured(workspace: Path, unit: str, gate: str, member: str | None) -> Anchor | None:
    """The unit's measured anchor for ``gate`` at ``member``, if it has one."""
    entry = _read(workspace).get(_key(unit, gate, member))
    if not isinstance(entry, dict) or not isinstance(entry.get("score"), (int, float)):
        return None
    return Anchor(float(entry["score"]), str(entry.get("tree") or ""))


def record(
    workspace: Path, unit: str, gate: str, member: str | None, score: float, tree: str
) -> bool:
    """Keep ``score`` as the anchor, unless one is kept already; whether it was kept now."""
    data = _read(workspace)
    key = _key(unit, gate, member)
    if key in data:
        return False
    data[key] = {"score": float(score), "tree": tree}
    write_text_atomic(Path(workspace) / FILENAME, json.dumps(data, indent=1, sort_keys=True))
    return True


__all__ = ["FILENAME", "Anchor", "measured", "record"]
