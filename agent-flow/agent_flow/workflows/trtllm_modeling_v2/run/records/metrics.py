# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a run cost and what it proved, in one file: ``metrics.json``.

Read off what the run already keeps in its workspace, so it can be written at
any moment, by the run as it closes or by ``trtllm-modeling-v2 metrics`` while
it goes on.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from ...infra.files import write_text_atomic
from ...spec.task_schema import TaskSchemaError, parse_parallel_segment
from .. import inbox as inbox_mod
from ..validation import contract as contract_mod

METRICS_FILENAME = "metrics.json"
#: One line per graph the orchestrator refused, in the workspace.
REFUSALS_FILENAME = "refusals.jsonl"


def _ranks(member: str) -> int:
    """How many devices a check at ``member`` ran on; one when the member does not say."""
    try:
        return parse_parallel_segment(member)[1] or 1
    except TaskSchemaError:
        return 1


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
    """Per test: runs, passed, failed, void, and the device-hours they took."""
    status = contract_mod.ContractStatus.load(workspace)
    out: dict[str, dict[str, float]] = {}
    for verdict in status.verdicts:
        row = out.setdefault(
            verdict.gate,
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


#: Who submitted a GPU job, by its spool ``owner``: the orchestrator's checks, a worker's
#: ``trtllm-modeling-v2 gate``, the flow's other jobs (preflight, delivery tests), a worker.
JOB_CHECKS, JOB_WORKER_GATES, JOB_FLOW_OTHER, JOB_WORKER = (
    "checks",
    "worker_gates",
    "flow_other",
    "worker",
)


def job_source(owner: str) -> str:
    """Which of the run's parts submitted a spool job, by its ``owner``."""
    if owner == "gate":
        return JOB_CHECKS
    if owner.startswith("gate-"):
        return JOB_WORKER_GATES
    return JOB_FLOW_OTHER if owner else JOB_WORKER


def _spools(workspace: Path) -> list[Path]:
    """The spools the run's jobs went to, from the workspace's spec; none on a local run."""
    try:
        spec = yaml.safe_load((workspace / "task.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    environment = spec.get("environment") or {}
    if not isinstance(environment, dict) or environment.get("kind") != "session_spool":
        return []
    named = [environment.get("spool")]
    named += [s.get("spool") for s in environment.get("sessions") or () if isinstance(s, dict)]
    return [Path(str(p)) for p in dict.fromkeys(n for n in named if n)]


def _epoch(text: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def gpu_jobs(records: Iterable[dict[str, Any]], since: float | None = None) -> dict[str, Any]:
    """Per source (:func:`job_source`): jobs, failed and device-hours; and the longest wait.

    ``records`` are the spool's finished jobs; one submitted before ``since`` is another run's.
    """
    out: dict[str, Any] = {}
    longest = 0.0
    for record in records:
        submitted, started = _epoch(record.get("submitted")), _epoch(record.get("started"))
        if since is not None and submitted is not None and submitted < since:
            continue
        row = out.setdefault(
            job_source(str(record.get("owner") or "")),
            {"jobs": 0, "failed": 0, "gpu_hours": 0.0},
        )
        row["jobs"] += 1
        row["failed"] += int(record.get("rc") not in (0, None))
        devices = len(record.get("devices") or ()) or 1
        row["gpu_hours"] += float(record.get("duration_s") or 0.0) * devices / 3600
        if submitted is not None and started is not None:
            longest = max(longest, started - submitted)
    for row in out.values():
        row["gpu_hours"] = round(row["gpu_hours"], 3)
    return {**dict(sorted(out.items())), "longest_wait_min": round(longest / 60, 1)}


def _gpu_jobs(workspace: Path, opened: datetime | None) -> dict[str, Any]:
    records = []
    for spool in _spools(workspace):
        for path in sorted((spool / "done").glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    return gpu_jobs(records, opened.timestamp() if opened else None) if records else {}


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
        # What the GPUs ran besides the checks: a worker testing its own work is most of it.
        "gpu_jobs": _gpu_jobs(workspace, opened),
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
    "JOB_CHECKS",
    "JOB_FLOW_OTHER",
    "JOB_WORKER",
    "JOB_WORKER_GATES",
    "METRICS_FILENAME",
    "REFUSALS_FILENAME",
    "collect",
    "gpu_jobs",
    "job_source",
    "record_refusal",
    "write",
]
