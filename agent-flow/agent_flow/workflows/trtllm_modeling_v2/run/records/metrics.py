# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a run cost and what it proved, in one file: ``metrics.json``.

Read off what the run already keeps in its workspace, so it can be written at
any moment, by the run as it closes or by ``trtllm-modeling-v2 metrics`` while
it goes on.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...infra.files import write_text_atomic
from .. import inbox as inbox_mod
from ..validation import contract as contract_mod

METRICS_FILENAME = "metrics.json"
#: One line per graph the orchestrator refused, in the workspace.
REFUSALS_FILENAME = "refusals.jsonl"

_GATE_KIND = re.compile(r"^([a-z_]+)")


def gate_kind(gate: str) -> str:
    """``accuracy`` for ``accuracy[gsm8k]``; the gate's own name otherwise."""
    match = _GATE_KIND.match(gate)
    return match.group(1) if match else gate


def _ranks(member: str) -> int:
    digits = re.findall(r"\d+", member)
    return int(digits[-1]) if digits else 1


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def _turns(workspace: Path) -> dict[str, int]:
    """Turns by role, from the turns' request files."""
    counts: dict[str, int] = {}
    for path in (workspace / "turns").glob("*.request.json"):
        try:
            role = json.loads(path.read_text(encoding="utf-8")).get("role")
        except (OSError, ValueError, AttributeError):
            continue
        if role:
            counts[str(role)] = counts.get(str(role), 0) + 1
    return dict(sorted(counts.items()))


def _checks(workspace: Path) -> dict[str, dict[str, float]]:
    """Per gate kind: runs, passed, failed, void, and the device-hours they took."""
    status = contract_mod.ContractStatus.load(workspace)
    out: dict[str, dict[str, float]] = {}
    for verdict in status.verdicts:
        row = out.setdefault(
            gate_kind(verdict.gate),
            {"runs": 0, "passed": 0, "failed": 0, "void": 0, "gpu_hours": 0.0},
        )
        row["runs"] += 1
        if verdict.void:
            row["void"] += 1
        elif verdict.passed:
            row["passed"] += 1
        else:
            row["failed"] += 1
        row["gpu_hours"] += verdict.duration_s * _ranks(verdict.member) / 3600
    for row in out.values():
        row["gpu_hours"] = round(row["gpu_hours"], 3)
    return dict(sorted(out.items()))


def _milestones(workspace: Path, opened: datetime | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tier in (contract_mod.MILESTONE_BRINGUP, contract_mod.MILESTONE_CORRECTNESS):
        recorded = contract_mod.milestones(workspace, tier)
        if not recorded:
            continue
        first, last = recorded[0], recorded[-1]

        def hours(m: contract_mod.Milestone) -> float | None:
            if m.elapsed_s is not None:
                return round(m.elapsed_s / 3600, 2)
            return round((m.at - opened.timestamp()) / 3600, 2) if opened and m.at else None

        out[tier] = {"sha": first.sha, "hours": hours(first), "latest_sha": last.sha}
    return out


def collect(workspace: Path, *, opened: datetime | None = None) -> dict[str, Any]:
    """The run's metrics, read off its workspace; ``opened`` is when it started."""
    workspace = Path(workspace)
    now = time.time()
    cells = contract_mod.load_cells(workspace)
    taken = workspace / inbox_mod.INBOX_DIRNAME / inbox_mod.TAKEN_DIRNAME
    return {
        "written_at": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
        "opened_at": opened.isoformat(timespec="seconds") if opened else None,
        "hours": round((now - opened.timestamp()) / 3600, 2) if opened else None,
        "milestones": _milestones(workspace, opened),
        "cells": {"holding": sorted(cells["cells"]), "best": cells["best"]},
        "checks": _checks(workspace),
        "turns": _turns(workspace),
        "planner_refusals": len(_lines(workspace / REFUSALS_FILENAME)),
        "operator_messages": len(list(taken.glob("*.md"))) if taken.is_dir() else 0,
    }


def write(workspace: Path, *, opened: datetime | None = None) -> Path:
    """Write :func:`collect`'s metrics to ``metrics.json``, whole; returns its path."""
    path = Path(workspace) / METRICS_FILENAME
    write_text_atomic(path, json.dumps(collect(workspace, opened=opened), indent=1) + "\n")
    return path


def record_refusal(workspace: Path, refusals: list[str]) -> None:
    """Add one refused graph, and why, to ``refusals.jsonl``."""
    path = Path(workspace) / REFUSALS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": time.time(), "refusals": refusals}) + "\n")


__all__ = [
    "METRICS_FILENAME",
    "REFUSALS_FILENAME",
    "collect",
    "gate_kind",
    "record_refusal",
    "write",
]
