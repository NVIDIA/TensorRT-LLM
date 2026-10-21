# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed decisions — the only channel an agent's verdict reaches the orchestrator through.

Agents never write ``progress.yaml`` directly and the orchestrator never
parses prose. Each turn writes one decision file whose ``decision`` is a
closed enum (``agents/prompts/_common.decision_file_block`` states the protocol), a
``Stop`` hook refuses to let the turn end without a write, and the
orchestrator validates the file (:func:`collect_decision_file`), records it
here and branches on the field. A verdict that cannot be expressed in the
enum is a verdict the workflow does not know how to act on, which is exactly
the thing worth finding out loudly.

The file is a YAML mapping with one list per stage, so iteration counters
stay local to the stage that produced them::

    plan_stage:
      - iteration: 1
        agent: plan_drafter
        timestamp: "2026-09-17T22:41:03"
        summary: |
          ...
        decision: DRAFT_READY
    build_stage:
      - {iteration: 1, agent: smith, ..., decision: LANDED, entry: catalog/norm/x.py}

Concurrency matters here in a way it does not in the other workflows: the
smith batch runs several agents at once, all appending. Every write takes a
lock and re-reads before appending, so a batch of eight cannot lose seven
entries to a read-modify-write race.
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
    "plan_drafter": PLAN_STAGE,
    "plan_reviewer": PLAN_STAGE,
    "smith": BUILD_STAGE,
    "assembler": BUILD_STAGE,
    "reviewer": BUILD_STAGE,
    "tuner": TUNE_STAGE,
    "planner": PLAN_STAGE,
    "improver": PLAN_STAGE,
}

# Decision vocabularies. Each is closed on purpose — see the module
# docstring. ``BLOCKED`` variants carry a reason so the orchestrator's
# triage does not have to read the summary.
# The tuner claims, the orchestrator measures: a CANDIDATE is measured end to
# end and gated by Python (tune/campaign.py), never kept on the tuner's word.
# Each role's enum is its row of the decision table (agents/decisions.py), the
# one its prompt's "Ending your turn" section is rendered from.
TUNER_DECISIONS = decision_names("tuner")
# The planner decides the task graph (orchestrator.py reads these).
PLANNER_DECISIONS = decision_names("planner")
IMPROVER_DECISIONS = decision_names("improver")
PLAN_DRAFTER_DECISIONS = decision_names("plan_drafter")
PLAN_REVIEWER_DECISIONS = decision_names("plan_reviewer")
SMITH_DECISIONS = decision_names("smith")
ASSEMBLER_DECISIONS = decision_names("assembler")
REVIEWER_DECISIONS = decision_names("reviewer")

# The triage table, as an enum. Each maps to a different
# orchestrator action, which is the whole reason it is not free text.
BLOCKED_REASONS = (
    "vocabulary_gap",
    "knowledge_gap",
    "accuracy_shortfall",
    "environment",
)


def _empty() -> dict[str, list[dict[str, Any]]]:
    return {stage: [] for stage in _STAGES}


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
    """Persist with a fixed key order so diffs stay readable."""
    ordered = {stage: data.get(stage, []) for stage in _STAGES}
    path.write_text(
        yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def init_progress_file(path: Path) -> None:
    write_progress(path, _empty())


def find_entries(
    path: Path,
    *,
    stage: str,
    agent: str | None = None,
    last_iterations: int | None = None,
) -> list[dict[str, Any]]:
    """Return entries from one stage, optionally filtered by agent and recency.

    Recency follows **append order**, not iteration number: a smith batch
    stamps several entries with the same iteration, and an assembler's
    re-dispatch after a mid-unit smith carries a number already seen.
    """
    if stage not in _STAGES:
        raise ValueError(f"unknown stage: {stage!r}")
    if last_iterations is not None and last_iterations < 1:
        raise ValueError(f"last_iterations must be >= 1, got {last_iterations}")
    if agent is not None and _STAGE_BY_AGENT.get(agent) != stage:
        return []
    entries = list(read_progress(path)[stage])
    if agent is not None:
        entries = [e for e in entries if e.get("agent") == agent]
    if last_iterations is not None:
        keep_from = len(entries)
        seen: list[Any] = []
        for index in range(len(entries) - 1, -1, -1):
            iteration = entries[index].get("iteration")
            if iteration is None:
                continue
            if iteration not in seen:
                if len(seen) == last_iterations:
                    break
                seen.append(iteration)
            keep_from = index
        entries = entries[keep_from:]
    return entries


def latest_entry(path: Path, agent: str) -> dict[str, Any] | None:
    stage = _STAGE_BY_AGENT.get(agent)
    if stage is None:
        return None
    matches = find_entries(path, stage=stage, agent=agent)
    return matches[-1] if matches else None


def latest_decision(path: Path, agent: str, allowed: tuple[str, ...]) -> str | None:
    """Return the agent's most recent decision, or ``None`` if it is not one we know."""
    entry = latest_entry(path, agent)
    if entry is None:
        return None
    decision = str(entry.get("decision", "")).strip().upper()
    return decision if decision in allowed else None


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _yaml_dump(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _log_write(agent: str, entry: dict[str, Any]) -> None:
    body = Syntax(_yaml_dump([entry]), "yaml", theme="ansi_dark", word_wrap=True)
    print_layer_panel(
        agent, f"system · wrote iter {entry.get('iteration')}", body, get_logger().console
    )


@dataclass
class ProgressContext:
    """Shared mutable context the decision collector writes through.

    ``current_iteration`` is set before each dispatch so an entry is stamped
    without the agent having to pass (or guess) it. ``lock`` serializes the
    read-modify-write, which a concurrent smith batch would otherwise race.

    ``current_unit`` is the assembly unit in progress, or empty outside
    assembly. Decision files and runtime-value dumps are filed under it.
    """

    path: Path
    current_iteration: int = 0
    current_unit: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def append(self, agent: str, entry: dict[str, Any]) -> None:
        with self.lock:
            data = read_progress(self.path)
            data[_STAGE_BY_AGENT[agent]].append(entry)
            write_progress(self.path, data)
        _log_write(agent, entry)


DECISIONS_DIRNAME = "decisions"

# The closed enum each role's decision file is validated against.
DECISIONS_BY_AGENT: dict[str, tuple[str, ...]] = {
    "plan_drafter": PLAN_DRAFTER_DECISIONS,
    "plan_reviewer": PLAN_REVIEWER_DECISIONS,
    "smith": SMITH_DECISIONS,
    "assembler": ASSEMBLER_DECISIONS,
    "reviewer": REVIEWER_DECISIONS,
    "tuner": TUNER_DECISIONS,
    "planner": PLANNER_DECISIONS,
    "improver": IMPROVER_DECISIONS,
}

EXTRA_KEYS_BY_AGENT: dict[str, tuple[str, ...]] = {
    # `constituents` is what makes MACRO actionable. Without it the
    # decomposition arrives as prose inside `summary`, which a human can act
    # on and the orchestrator cannot -- so the one outcome whose entire
    # purpose is "here is the work that comes next" would end the run instead
    # of extending it.
    # `engine_defect` is a failed receipt's defect outside the entry, in the
    # shape the engine-defect registry keeps (`agents/lessons.RECEIPT_KEYS`).
    # `uncertified` is what a landed entry still does not certify of what the
    # target serves: the planner hears it with the task's end.
    "smith": ("entry", "receipt", "constituents", "engine_defect", "uncertified"),
    # `uncertified` names the calls a unit makes, correct, outside their entries'
    # CELLS: the planner hears them with the task's end and adds the certification.
    "assembler": ("blocked_reason", "missing_entry", "uncertified"),
    # The orchestrator re-reads the candidate's own decision file for the rest
    # (micro-bench table, projection); these are what progress.yaml keeps.
    "tuner": ("slug", "kind", "worktree", "branch", "commits", "projection"),
}

#: Keys a MACRO decision's `constituents` entries may carry. `target` and
#: `tier` are required: the orchestrator needs a name to dispatch and a rung
#: to budget turns against.
CONSTITUENT_REQUIRED: tuple[str, ...] = ("target", "tier")
CONSTITUENT_OPTIONAL: tuple[str, ...] = ("surface", "note", "unit", "depends_on")


#: The keys a smith's ``engine_defect`` report carries; ``detect`` is optional.
ENGINE_DEFECT_KEYS: tuple[str, ...] = ("affects", "trigger", "detect", "proposal")


def parse_engine_defect(raw: Any) -> dict[str, Any]:
    """A smith's ``engine_defect`` report, its known keys only; empty unless it names the defect.

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
    """Normalize a MACRO decision's child list, dropping anything unusable.

    Permissive about shape and strict about content: a smith writing YAML by
    hand should not lose a whole decomposition to a formatting slip, but a
    child with no target cannot be dispatched and a child with no tier cannot
    be budgeted, so those are dropped rather than guessed at.
    """
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


#: The unit a plan-stage decision is filed under. That stage is not per unit,
#: and naming it keeps its files apart from every assembly unit's.
PLAN_DECISION_UNIT = "plan"
#: The unit a build-stage decision is filed under when none is in progress.
DEFAULT_DECISION_UNIT = "core"


def decision_unit(agent: str, unit: str = "") -> str:
    """The unit ``agent``'s decision file is named for.

    Plan roles file under their stage; build roles (smith, assembler,
    reviewer) under the unit they work for.
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
    """Where a role writes its decision.

    ``decisions/<unit>-<role>-<turn>.yaml``. The unit is part of the name
    because turn numbers restart with every unit: named by role and turn
    alone, a feature unit's assembler turn 1 would overwrite core's, and each
    unit's reviewers would overwrite the earlier units' decisions.

    A turn number can still come round again for the same unit and role -- a
    re-entered unit restarts its turn count, a smith row is retried under its
    own number, a ``--clean`` run starts over in the same workspace. So a
    second dispatch of the same ``(unit, role, turn)`` gets ``attempt`` 2 and
    the turn token ``<turn>.2``, and so on: :func:`reserve_decision` never
    hands out a name that was used before, so no decision file is ever
    overwritten.
    """
    token = f"{iteration}.{attempt}" if attempt > 1 else str(iteration)
    name = f"{decision_unit(agent, unit)}-{agent}-{token}"
    return Path(workspace) / DECISIONS_DIRNAME / f"{name}.yaml"


def _decision_marker_path(decision: Path) -> Path:
    """The dotfile stamped beside a decision when its turn is dispatched.

    A dotfile, and beside the decision rather than under a directory of its
    own, because the two must share a filesystem — see
    ``reserve_decision`` for why that is the whole mechanism.
    """
    return decision.with_name(f".{decision.stem}.dispatched")


def reserve_decision(workspace: Path, agent: str, iteration: int, *, unit: str = "") -> Path:
    """Claim a decision path no earlier dispatch used, and stamp its dispatch marker.

    The first attempt whose decision file and marker both do not exist is
    claimed by creating the marker with ``O_CREAT | O_EXCL``, so two
    concurrent dispatches (or two processes) can never be handed the same
    path. Returns the path the turn must write.

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


def mark_decision_dispatch(workspace: Path, agent: str, iteration: int, *, unit: str = "") -> Path:
    """Reserve this turn's decision path before it is dispatched. See :func:`reserve_decision`.

    A reserved path is new, so it holds nothing until this turn writes it.
    With ``(agent, iteration)`` as the whole name, a turn that ends *without*
    writing a decision would leave an earlier turn's verdict at its path,
    looking fresh, and an audit loop that reads old verdicts back cannot
    converge.
    """
    return reserve_decision(workspace, agent, iteration, unit=unit)


def decision_file_instruction(path: Path) -> str:
    """The block a dispatch prompt ends with: the one exact path this turn writes.

    Stated in the dispatch rather than left to the system prompt's pattern, so
    the path is exact and un-guessable: the role is told the absolute file to
    write, not a pattern to instantiate.
    """
    return (
        "---\n\n"
        "**Write your decision file to exactly this path when you finish:**\n\n"
        f"`{path}`\n\n"
        "Nothing downstream reads your prose. This file is how this turn "
        "reaches the orchestrator.\n"
    )


def current_decision(workspace: Path, agent: str, iteration: int, *, unit: str = "") -> Path:
    """The path the latest dispatch of ``(unit, agent, iteration)`` was told to write.

    The highest reserved attempt. With nothing reserved -- a run that never
    stamped one -- it is the first attempt's path.
    """
    latest = decision_path(workspace, agent, iteration, unit=unit)
    attempt = 1
    while True:
        attempt += 1
        candidate = decision_path(workspace, agent, iteration, unit=unit, attempt=attempt)
        if not (_decision_marker_path(candidate).exists() or candidate.exists()):
            return latest
        latest = candidate


def decision_is_stale(workspace: Path, agent: str, iteration: int, *, unit: str = "") -> bool:
    """Whether the current decision file predates the dispatch that reserved it.

    False when either file is absent: a missing decision is the caller's
    "no decision" case, and a missing marker means this run never stamped one
    — in which case accepting the decision is the honest fallback, rather
    than silently refusing every decision a run ever writes.
    """
    return _predates_dispatch(current_decision(workspace, agent, iteration, unit=unit))


def _predates_dispatch(decision: Path) -> bool:
    """Whether ``decision`` is older than the dispatch marker stamped beside it."""
    marker = _decision_marker_path(decision)
    if not decision.is_file() or not marker.is_file():
        return False
    return decision.stat().st_mtime < marker.stat().st_mtime


def _read_decision(path: Path, agent: str) -> tuple[str, dict[str, Any]] | None:
    """``(decision, the file's mapping)`` when ``path`` holds a decision ``agent`` may give."""
    if not path.is_file():
        return None
    # A file older than its dispatch marker belongs to a PREVIOUS turn that
    # shared this path. Reservation makes that impossible for paths it hands
    # out; the check covers a file that reached the path some other way.
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
    """Read, validate and record the decision file ``agent`` wrote.

    The decision file is the only channel a verdict reaches the orchestrator
    through, on either transport. The enum is validated here, on read, and a
    decision outside it is rejected, so the orchestrator branches on one
    closed value it did not get from prose.

    ``path`` is read instead of the latest dispatch's path when the turn
    being collected wrote elsewhere: one adopted from an earlier process,
    whose decision stays where that dispatch reserved it.

    Returns the validated decision, or ``None`` if the role wrote nothing
    usable — which the caller must treat as a failed turn rather than as
    silence to interpret.
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
        "timestamp": _now_iso(),
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
    "ASSEMBLER_DECISIONS",
    "BLOCKED_REASONS",
    "BUILD_STAGE",
    "IMPROVER_DECISIONS",
    "PLANNER_DECISIONS",
    "PLAN_DRAFTER_DECISIONS",
    "parse_engine_defect",
    "PLAN_REVIEWER_DECISIONS",
    "PLAN_STAGE",
    "ProgressContext",
    "REVIEWER_DECISIONS",
    "SMITH_DECISIONS",
    "DEFAULT_DECISION_UNIT",
    "PLAN_DECISION_UNIT",
    "collect_decision_file",
    "current_decision",
    "decision_file_instruction",
    "decision_is_stale",
    "decision_path",
    "decision_unit",
    "decision_value",
    "find_entries",
    "init_progress_file",
    "latest_decision",
    "latest_entry",
    "mark_decision_dispatch",
    "read_progress",
    "reserve_decision",
    "write_progress",
]
