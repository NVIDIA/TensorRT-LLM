# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The promotions a run closes with, and the verdicts they carry across.

A promote merges the target's local variants into the shared catalog: the same
kernels, moved, and the target's imports pointed at them. It runs last, once
the contract holds (``orchestrator._promote``), so it interrupts no check, and
its one check is a boot of the core (the promote task's ``satisfies``). Once
the promote is done (integrated, its boot passed on the promoted tree), every
verdict the contract held on the tree it started from holds on the tree it
left: no benchmark runs again for a move.

``promotions.json`` in the workspace keeps, per promote task, the tree it
started from and, once done, the tree it left.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...infra.files import write_text_atomic

FILENAME = "promotions.json"


def _read(workspace: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads((Path(workspace) / FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(workspace: Path, data: dict[str, dict[str, Any]]) -> None:
    write_text_atomic(Path(workspace) / FILENAME, json.dumps(data, indent=1, sort_keys=True))


def started(workspace: Path, task_id: str, marker: str) -> None:
    """Promote ``task_id`` starts from the tree at ``marker`` (the contract holds there)."""
    if not marker:
        return
    data = _read(workspace)
    data[task_id] = {"from": marker}
    _write(workspace, data)


def ended(workspace: Path, task_id: str, marker: str | None) -> None:
    """Promote ``task_id`` ended: done on the tree at ``marker``, or not done (``None``)."""
    data = _read(workspace)
    if task_id not in data:
        return
    if marker:
        data[task_id]["to"] = marker
    else:
        data.pop(task_id)
    _write(workspace, data)


def carried(workspace: Path, current: str) -> set[str]:
    """The trees whose verdicts hold on ``current`` through done promotes, ``current`` aside.

    A chain of promotes carries back to the tree the first one started from.
    """
    if not current:
        return set()
    done = [r for r in _read(workspace).values() if r.get("from") and r.get("to")]
    reached, found = {current}, set()
    changed = True
    while changed:
        changed = False
        for record in done:
            if record["to"] in reached and record["from"] not in reached:
                reached.add(record["from"])
                found.add(record["from"])
                changed = True
    return found


__all__ = ["FILENAME", "carried", "ended", "started"]
