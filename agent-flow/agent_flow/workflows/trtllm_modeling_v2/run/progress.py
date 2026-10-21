# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed decisions: the only channel an agent's verdict reaches the orchestrator through.

Each turn writes one decision file whose ``decision`` is a closed enum
(``agents/prompts/_common.decision_file_block`` states the protocol). The
orchestrator validates it (:func:`collect_decision_file`) and appends it to
``progress.yaml``, which holds one list of entries per stage. Concurrent
kernel writers append at once, so every write takes a lock and re-reads first.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.syntax import Syntax

from agent_flow.console import print_layer_panel
from agent_flow.logger import get_logger

from ..agents.decisions import names as decision_names

PLAN_STAGE = "plan_stage"
BUILD_STAGE = "build_stage"
TUNE_STAGE = "tune_stage"
_STAGES = (PLAN_STAGE, BUILD_STAGE, TUNE_STAGE)

_STAGE_BY_AGENT = {
    "surveyor": PLAN_STAGE,
    "kernel_writer": BUILD_STAGE,
    "assembler": BUILD_STAGE,
    "tuner": TUNE_STAGE,
    "planner": PLAN_STAGE,
}

#: The closed enum each role's decision file is validated against: its row of the
#: decision table (``agents/decisions.py``).
DECISIONS_BY_AGENT: dict[str, tuple[str, ...]] = {
    role: decision_names(role) for role in _STAGE_BY_AGENT
}


def _empty() -> dict[str, list[dict[str, Any]]]:
    return {stage: [] for stage in _STAGES}


def _yaml_dump(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)


def read_progress(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load ``progress.yaml``, returning every stage key even when absent."""
    if not path.is_file():
        return _empty()
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return _empty()
    data = yaml.safe_load(text)
    if data is None:
        return _empty()
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping keyed by stage, got {type(data).__name__}"
        )
    result = _empty()
    for stage in _STAGES:
        value = data.get(stage) or []
        if not isinstance(value, list):
            raise ValueError(f"{path}: `{stage}` must be a list, got {type(value).__name__}")
        result[stage] = value
    return result


def write_progress(path: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    """Persist with a fixed key order so diffs stay readable.

    Renamed into place: a reader on another thread never sees a truncated file.
    """
    staged = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    staged.write_text(
        _yaml_dump({stage: data.get(stage, []) for stage in _STAGES}), encoding="utf-8"
    )
    os.replace(staged, path)


def init_progress_file(path: Path) -> None:
    write_progress(path, _empty())


def find_entries(path: Path, *, stage: str, agent: str | None = None) -> list[dict[str, Any]]:
    """Return the entries of one stage, in append order, optionally only ``agent``'s."""
    if stage not in _STAGES:
        raise ValueError(f"unknown stage: {stage!r}")
    if agent is not None and _STAGE_BY_AGENT.get(agent) != stage:
        return []
    entries = list(read_progress(path)[stage])
    if agent is not None:
        entries = [e for e in entries if e.get("agent") == agent]
    return entries


def latest_entry(path: Path, agent: str) -> dict[str, Any] | None:
    stage = _STAGE_BY_AGENT.get(agent)
    if stage is None:
        return None
    matches = find_entries(path, stage=stage, agent=agent)
    return matches[-1] if matches else None


def _log_write(agent: str, entry: dict[str, Any]) -> None:
    body = Syntax(_yaml_dump([entry]), "yaml", theme="ansi_dark", word_wrap=True)
    print_layer_panel(
        agent, f"system · wrote iter {entry.get('iteration')}", body, get_logger().console
    )


@dataclass
class ProgressContext:
    """``progress.yaml``, which the decision collector appends to under ``lock``."""

    path: Path
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def append(self, agent: str, entry: dict[str, Any]) -> None:
        with self.lock:
            data = read_progress(self.path)
            data[_STAGE_BY_AGENT[agent]].append(entry)
            write_progress(self.path, data)
        _log_write(agent, entry)


DECISIONS_DIRNAME = "decisions"

#: The keys besides ``summary`` and ``decision`` that ``progress.yaml`` keeps of a decision.
EXTRA_KEYS_BY_AGENT: dict[str, tuple[str, ...]] = {
    # `constituents` is a MACRO's decomposition, the work that comes next.
    # `engine_defect` is a failed receipt's defect outside the entry
    # (`ENGINE_DEFECT_KEYS`): the orchestrator adds the task that fixes it.
    # `uncertified` is what a landed entry still does not certify of what the
    # target serves: the planner hears it with the task's end.
    "kernel_writer": ("entry", "receipt", "constituents", "engine_defect", "uncertified"),
    # `uncertified` names the calls a unit makes, correct, outside their entries'
    # CELLS: the planner hears them with the task's end and adds the certification.
    "assembler": ("blocked_reason", "missing_entry", "uncertified"),
    # The orchestrator re-reads the candidate's own decision file for the rest
    # (micro-bench table, projection).
    "tuner": ("slug", "kind", "worktree", "branch", "commits", "projection"),
}

#: Keys a MACRO decision's `constituents` entries may carry besides `target` and `tier`.
CONSTITUENT_OPTIONAL: tuple[str, ...] = ("surface", "note", "unit", "depends_on")


#: The keys a kernel writer's ``engine_defect`` report carries; ``detect`` is optional.
ENGINE_DEFECT_KEYS: tuple[str, ...] = ("affects", "trigger", "detect", "proposal")


def parse_engine_defect(raw: Any) -> dict[str, Any]:
    """A kernel writer's ``engine_defect`` report, its known keys only; empty unless it names the defect.

    ``affects`` (where the defect is), ``trigger`` (the parameter regime that
    hits it) and ``proposal`` (the fix) are text; ``detect`` (the failing
    test, or the cell) is text or a mapping.
    """
    if not isinstance(raw, dict):
        return {}
    report: dict[str, Any] = {}
    for key in ENGINE_DEFECT_KEYS:
        value = raw.get(key)
        if key == "detect" and isinstance(value, dict):
            report[key] = dict(value)
        elif value is not None and str(value).strip():
            report[key] = str(value).strip()
    return report if all(report.get(k) for k in ("affects", "trigger", "proposal")) else {}


def parse_constituents(raw: Any) -> list[dict[str, Any]]:
    """Normalize a MACRO decision's child list: one with no ``target`` is dropped, no ``tier`` is 1."""
    if not isinstance(raw, list):
        return []
    children: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        target = str(row.get("target", "")).strip()
        if not target:
            continue
        try:
            tier = int(row.get("tier", 1) or 1)
        except (TypeError, ValueError):
            tier = 1
        child: dict[str, Any] = {"target": target, "tier": tier}
        for key in CONSTITUENT_OPTIONAL:
            value = row.get(key)
            if value is None:
                continue
            if key == "depends_on":
                if isinstance(value, list):
                    child[key] = [int(v) for v in value if str(v).strip().lstrip("-").isdigit()]
            elif str(value).strip():
                child[key] = str(value).strip()
        children.append(child)
    return children


#: The unit a plan-stage decision is filed under: that stage is not per unit.
PLAN_DECISION_UNIT = "plan"
#: The unit a build-stage decision is filed under when none is in progress.
DEFAULT_DECISION_UNIT = "core"


def decision_unit(agent: str, unit: str = "") -> str:
    """The unit ``agent``'s decision file is named for.

    Plan roles file under their stage; build roles (kernel_writer, assembler) under
    the unit they work for.
    """
    stage = _STAGE_BY_AGENT.get(agent)
    if stage == PLAN_STAGE:
        return PLAN_DECISION_UNIT
    return re.sub(r"[^A-Za-z0-9_.]+", "_", unit or DEFAULT_DECISION_UNIT)


def uncertified_report(value: Any) -> str:
    """A decision's ``uncertified`` list as one line for the planner; ``""`` when it names none."""
    if isinstance(value, str):
        value = value.splitlines()
    if not isinstance(value, list):
        return ""
    calls = [" ".join(str(v).split()) for v in value if str(v).strip()]
    if not calls:
        return ""
    listed = "; ".join(calls)
    return "uncertified: " + (listed if len(listed) <= 1200 else listed[:1197] + "...")


def decision_path(
    workspace: Path, agent: str, iteration: int, *, unit: str = "", attempt: int = 1
) -> Path:
    """``decisions/<unit>-<role>-<turn>[.<attempt>].yaml``: where a role writes its decision.

    Turn numbers restart with every unit, so the unit is part of the name. A
    turn number can come round again for the same unit and role (a re-entered
    unit, a retried kernel writer row, a ``--clean`` run in the same
    workspace), so each later dispatch of it gets the next ``attempt``.
    """
    token = f"{iteration}.{attempt}" if attempt > 1 else str(iteration)
    name = f"{decision_unit(agent, unit)}-{agent}-{token}"
    return Path(workspace) / DECISIONS_DIRNAME / f"{name}.yaml"


def _decision_marker_path(decision: Path) -> Path:
    """The dotfile stamped beside a decision when its turn is dispatched.

    Beside the decision, so that the two share a filesystem
    (:func:`mark_decision_dispatch`).
    """
    return decision.with_name(f".{decision.stem}.dispatched")


def mark_decision_dispatch(workspace: Path, agent: str, iteration: int, *, unit: str = "") -> Path:
    """Claim a decision path no earlier dispatch used, stamp its dispatch marker, and return it.

    The first attempt whose decision file and marker both do not exist is
    claimed by creating the marker with ``O_CREAT | O_EXCL``, so two
    concurrent dispatches (or two processes) are never handed the same path,
    and no decision file is ever overwritten.

    The marker also orders the turn against its answer. It is compared by
    **mtime against the decision file on the same filesystem**, never against
    a clock: an agent may answer from a different node whose clock is skewed
    by minutes, and both files live under the workspace.
    """
    attempt = 1
    while True:
        path = decision_path(workspace, agent, iteration, unit=unit, attempt=attempt)
        marker = _decision_marker_path(path)
        if not path.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
            except FileExistsError:
                pass
            else:
                return path
        attempt += 1


def decision_file_instruction(path: Path) -> str:
    """The block a dispatch prompt ends with: the one exact path this turn writes."""
    return (
        "---\n\n"
        "**Write your decision file to exactly this path when you finish:**\n\n"
        f"`{path}`\n\n"
        "Nothing downstream reads your prose. This file is how this turn "
        "reaches the orchestrator.\n"
    )


def current_decision(workspace: Path, agent: str, iteration: int, *, unit: str = "") -> Path:
    """The path the latest dispatch of ``(unit, agent, iteration)`` was told to write.

    The highest reserved attempt; with nothing reserved, the first attempt's path.
    """
    latest = decision_path(workspace, agent, iteration, unit=unit)
    attempt = 1
    while True:
        attempt += 1
        candidate = decision_path(workspace, agent, iteration, unit=unit, attempt=attempt)
        if not (_decision_marker_path(candidate).exists() or candidate.exists()):
            return latest
        latest = candidate


def _predates_dispatch(decision: Path) -> bool:
    """Whether ``decision`` is older than the dispatch marker stamped beside it.

    False when either file is absent: a run that never stamped a marker has
    its decisions accepted.
    """
    marker = _decision_marker_path(decision)
    if not decision.is_file() or not marker.is_file():
        return False
    return decision.stat().st_mtime < marker.stat().st_mtime


def _read_decision(path: Path, agent: str) -> tuple[str, dict[str, Any]] | None:
    """``(decision, the file's mapping)`` when ``path`` holds a decision ``agent`` may give."""
    if not path.is_file():
        return None
    # Older than its dispatch marker: a previous turn's file that reached this path.
    if _predates_dispatch(path):
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(raw, dict):
        return None
    decision = str(raw.get("decision", "")).strip()
    if decision not in DECISIONS_BY_AGENT[agent]:
        return None
    return decision, raw


def decision_value(path: Path, agent: str) -> str | None:
    """The decision ``agent`` wrote at ``path`` if it is a valid one; nothing is recorded."""
    read = _read_decision(Path(path), agent)
    return read[0] if read else None


def collect_decision_file(
    ctx: ProgressContext,
    workspace: Path,
    agent: str,
    iteration: int,
    *,
    unit: str = "",
    path: Path | None = None,
) -> str | None:
    """Read, validate and record the decision file ``agent`` wrote; the decision, or ``None``.

    ``path`` is read instead of the latest dispatch's path for a turn adopted
    from an earlier process, whose decision stays where that dispatch reserved
    it. ``None`` means the role wrote nothing usable: the caller treats it as a
    failed turn.
    """
    if path is None:
        path = current_decision(workspace, agent, iteration, unit=unit)
    path = Path(path)
    read = _read_decision(path, agent)
    if read is None:
        return None
    decision, raw = read

    entry: dict[str, Any] = {
        "iteration": iteration,
        "agent": agent,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "summary": str(raw.get("summary", "")).strip(),
        "decision": decision,
        "decision_file": path.name,
    }
    for key in EXTRA_KEYS_BY_AGENT.get(agent, ()):
        value = raw.get(key)
        if value is None:
            continue
        if key == "constituents":
            children = parse_constituents(value)
            if children:
                entry[key] = children
        elif key == "engine_defect":
            report = parse_engine_defect(value)
            if report:
                entry[key] = report
        elif str(value).strip():
            entry[key] = str(value).strip()
    ctx.append(agent, entry)
    return decision


__all__ = [
    "uncertified_report",
    "ENGINE_DEFECT_KEYS",
    "BUILD_STAGE",
    "DECISIONS_BY_AGENT",
    "parse_engine_defect",
    "PLAN_STAGE",
    "ProgressContext",
    "DEFAULT_DECISION_UNIT",
    "PLAN_DECISION_UNIT",
    "collect_decision_file",
    "current_decision",
    "decision_file_instruction",
    "decision_path",
    "decision_unit",
    "decision_value",
    "find_entries",
    "init_progress_file",
    "latest_entry",
    "mark_decision_dispatch",
    "read_progress",
    "write_progress",
]
