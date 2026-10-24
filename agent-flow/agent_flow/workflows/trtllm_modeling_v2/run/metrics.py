# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a run cost and what it proved, in one file: ``metrics.json``, and the run's budget.

Read off what the run already keeps in its workspace, so it can be written at
any moment, by the run as it closes or by ``trtllm-modeling-v2 metrics`` while
it goes on.

The budget (``task.yaml``'s ``budget``) bounds the run's wall-clock hours and
its device-hours: the devices its sessions hold, times the hours since the
run started.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..spec.task_schema import BUDGET_FIELD
from . import contract as contract_mod
from . import inbox as inbox_mod

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
    """Turns by role, from both transports' request files."""
    counts: dict[str, int] = {}
    for directory in ("turns", "dispatch"):
        for path in (workspace / directory).glob("*.request.json"):
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


def collect(
    workspace: Path,
    *,
    opened: datetime | None = None,
    task_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The run's metrics, read off its workspace; ``opened`` is when it started."""
    workspace = Path(workspace)
    now = time.time()
    cells = contract_mod.load_cells(workspace)
    taken = workspace / inbox_mod.INBOX_DIRNAME / inbox_mod.TAKEN_DIRNAME
    metrics: dict[str, Any] = {
        "written_at": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
        "opened_at": opened.isoformat(timespec="seconds") if opened else None,
        "hours": round((now - opened.timestamp()) / 3600, 2) if opened else None,
        "milestones": _milestones(workspace, opened),
        "cells": {"holding": sorted(cells["cells"]), "best": cells["best"]},
        "checks": _checks(workspace),
        "turns": _turns(workspace),
        "planner_refusals": len(_lines(workspace / REFUSALS_FILENAME)),
        "operator_messages": len(list(taken.glob("*.md"))) if taken.is_dir() else 0,
        "restarts": len(_lines(workspace / "restarts.jsonl")),
    }
    budget = (task_data or {}).get(BUDGET_FIELD)
    if budget:
        metrics["budget"] = dict(budget)
    return metrics


def write(workspace: Path, **kwargs: Any) -> Path:
    """Write :func:`collect`'s metrics to ``metrics.json``, whole; returns its path."""
    path = Path(workspace) / METRICS_FILENAME
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(collect(workspace, **kwargs), indent=1) + "\n", encoding="utf-8")
    os.replace(partial, path)
    return path


def record_refusal(workspace: Path, refusals: list[str]) -> None:
    """Add one refused graph, and why, to ``refusals.jsonl``."""
    path = Path(workspace) / REFUSALS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": time.time(), "refusals": refusals}) + "\n")


def budget_spent(
    budget: Mapping[str, Any] | None, *, opened: datetime | None, devices: int, now: float
) -> str:
    """Why the run is past its ``budget``; ``""`` while it is within it, or has none."""
    if not budget or opened is None:
        return ""
    hours = (now - opened.timestamp()) / 3600
    wall = budget.get("wall_hours")
    if wall is not None and hours >= float(wall):
        return f"{hours:.1f} h of wall clock, the budget's {float(wall):g}"
    gpus = budget.get("gpu_hours")
    if gpus is not None and devices > 0 and devices * hours >= float(gpus):
        return (
            f"{devices * hours:.0f} device-hours held ({devices} devices for {hours:.1f} h), "
            f"the budget's {float(gpus):g}"
        )
    return ""


__all__ = [
    "METRICS_FILENAME",
    "REFUSALS_FILENAME",
    "budget_spent",
    "collect",
    "gate_kind",
    "record_refusal",
    "write",
]
