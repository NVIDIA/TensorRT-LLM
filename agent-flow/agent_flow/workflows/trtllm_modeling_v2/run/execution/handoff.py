# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task handoffs: validated turn decisions and bounded receipts for dependent tasks.

Each turn writes one decision file whose ``decision`` is a closed enum
(``agents/prompts/_common.decision_file_block`` states the protocol), at the
one path its dispatch reserved (:func:`mark_decision_dispatch`). The
orchestrator reads it there and nowhere else (:func:`read_decision`): a turn
that wrote no valid decision has none, whatever an earlier turn wrote.

A finished task leaves a bounded receipt in ``tasks/<id>.receipt.yaml``.
Dependent tasks receive its decision, landed commits and evidence paths
without a planner turn restating them in a goal.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ...agents.decisions import DECISIONS_BY_AGENT, PLAN_ROLES
from ...infra.files import write_text_atomic
from ..records.journal import utc_now

DECISIONS_DIRNAME = "decisions"

#: Keys a MACRO decision's `constituents` entries may carry besides `target` and `tier`.
CONSTITUENT_OPTIONAL: tuple[str, ...] = ("surface", "note", "unit", "depends_on")


#: The keys a kernel writer's ``engine_defect`` report carries; ``detect`` is optional.
ENGINE_DEFECT_KEYS: tuple[str, ...] = ("affects", "trigger", "detect", "proposal")


def engine_defect_report(raw: Any) -> str:
    """A kernel writer's ``engine_defect`` report as one line for the planner; ``""`` unless it names the defect.

    It names the defect with ``affects`` (where it is), ``trigger`` (the
    parameter regime that hits it) and ``proposal`` (the fix); ``detect`` (the
    failing test, or the cell as a mapping) follows when given.
    """
    if not isinstance(raw, Mapping):
        return ""
    said = {key: _one_line(raw.get(key)) for key in ENGINE_DEFECT_KEYS}
    if not all(said[key] for key in ("affects", "trigger", "proposal")):
        return ""
    line = (
        f"engine defect in `{said['affects']}`, hit by {said['trigger']}; "
        f"proposed fix: {said['proposal']}"
    )
    return line + (f"; detected by {said['detect']}" if said["detect"] else "")


def _one_line(value: Any) -> str:
    if isinstance(value, Mapping):
        value = json.dumps(dict(value), sort_keys=True, default=str)
    return " ".join(str(value).split()) if value is not None else ""


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


#: The unit a plan role's decision is filed under: the plan is not per unit.
PLAN_DECISION_UNIT = "plan"
#: The unit a build role's decision is filed under when none is in progress.
DEFAULT_DECISION_UNIT = "core"


def decision_unit(agent: str, unit: str = "") -> str:
    """The unit ``agent``'s decision file is named for.

    Plan roles (:data:`PLAN_ROLES`) file under the plan; the others under the
    unit they work for.
    """
    if agent in PLAN_ROLES:
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


def mark_decision_dispatch(
    workspace: Path,
    agent: str,
    iteration: int,
    *,
    unit: str = "",
    identity: Mapping[str, str | int] | None = None,
) -> Path:
    """Claim a decision path no earlier dispatch used, stamp its dispatch marker, and return it.

    The first attempt whose decision file and marker both do not exist is
    claimed by creating the marker with ``O_CREAT | O_EXCL``, so two
    concurrent dispatches (or two processes) are never handed the same path,
    and no decision file is ever overwritten.

    The marker also orders the turn against its answer. It is compared by
    **mtime against the decision file on the same filesystem**, never against
    a clock: an agent may answer from a different node whose clock is skewed
    by minutes, and both files live under the workspace.

    ``identity`` records which task attempt owns the dispatch, when supplied.
    A resumed process can distinguish its unfinished attempt from a new retry.
    """
    payload = json.dumps(dict(identity)) if identity is not None else ""
    attempt = 1
    while True:
        path = decision_path(workspace, agent, iteration, unit=unit, attempt=attempt)
        marker = _decision_marker_path(path)
        if not path.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(payload)
                return path
        attempt += 1


def decision_dispatch_matches(decision: Path, identity: Mapping[str, str | int]) -> bool:
    """Whether the dispatch marker names exactly this task attempt.

    Missing, legacy empty, or incomplete markers cannot establish identity.
    Matching identity does not validate the answer; :func:`read_decision` does.
    """
    try:
        recorded = json.loads(_decision_marker_path(decision).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return bool(identity) and recorded == dict(identity)


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


def read_decision(path: Path, agent: str) -> dict[str, Any] | None:
    """What ``agent`` wrote at ``path`` when it holds a decision ``agent`` may give; else ``None``.

    The file's mapping, its ``decision`` stripped. ``None`` also for a file
    older than the dispatch marker stamped beside it: a previous turn's that
    reached this path.
    """
    path = Path(path)
    if not path.is_file() or _predates_dispatch(path):
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
    return {**raw, "decision": decision}


def decision_value(path: Path, agent: str) -> str | None:
    """The decision ``agent`` wrote at ``path`` if it is a valid one (:func:`read_decision`)."""
    raw = read_decision(path, agent)
    return raw["decision"] if raw is not None else None


#: The heading a task's prompt carries its dependencies' receipts under.
HEADING = "## Receipts of the tasks this one depends on"
#: Beside the envelope ``tasks/<id>.json``.
RECEIPT_SUFFIX = ".receipt.yaml"
#: The most of the summary's first paragraph a receipt keeps.
SUMMARY_CHARS = 400
#: The most commits a receipt lists; a count says how many more there were.
MAX_COMMITS = 12
#: Past this many characters of a prompt's receipts, each further one is a single line.
RENDER_CHARS = 12_000

#: A full or abbreviated commit id.
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")


@dataclass
class TaskResult:
    """A dispatch's validated answer and evidence, carried through integration into its receipt."""

    decision: str = ""
    note: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    decision_file: Path | None = None
    result_file: Path | None = None

    @classmethod
    def answered(cls, role: str, decision_file: Path, result_file: Path | None) -> TaskResult:
        data = read_decision(decision_file, role) or {}
        return cls(
            str(data.get("decision") or ""),
            data=data,
            decision_file=decision_file,
            result_file=result_file,
        )

    @property
    def text(self) -> str:
        return f"{self.decision}\n{self.note}" if self.note else self.decision

    def annotate(self, note: str) -> TaskResult:
        if note:
            self.note = f"{self.note}; {note}" if self.note else note
        return self


def clip(text: str, limit: int) -> str:
    """``text`` up to ``limit`` characters, an ellipsis marking what was cut."""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def first_paragraph(text: Any, limit: int = SUMMARY_CHARS) -> str:
    """The first paragraph of ``text`` as one line, clipped to ``limit``."""
    for paragraph in re.split(r"\n\s*\n", str(text or "").strip()):
        line = " ".join(paragraph.split())
        if line:
            return clip(line, limit)
    return ""


def _entry(raw: Mapping[str, Any]) -> str:
    """A kernel writer's entry path, from its decision's ``entry`` (a path, or a mapping with one)."""
    entry = raw.get("entry")
    if isinstance(entry, Mapping):
        entry = entry.get("path")
    return clip(str(entry), 300) if entry else ""


def _tests(raw: Mapping[str, Any]) -> str:
    """A kernel writer's test result, from its decision's ``receipt`` (a mapping with ``result``, or text)."""
    receipt = raw.get("receipt")
    if isinstance(receipt, Mapping):
        receipt = receipt.get("result")
    return first_paragraph(receipt, 200) if receipt else ""


def build(
    task_id: str,
    kind: str,
    *,
    result: TaskResult,
    attempt: int = 1,
    commits: Sequence[str] = (),
    cleared: str = "",
) -> dict[str, Any]:
    """The receipt of task ``task_id``, from the dispatch that actually ran.

    ``cleared`` is the commit a regression check found clean, the next one's
    base (``checks/test_results.default_base``).
    """
    raw = result.data
    receipt: dict[str, Any] = {
        "task": task_id,
        "kind": kind,
        "attempt": int(attempt),
        "decision": result.decision,
        "summary": first_paragraph(raw.get("summary")),
    }
    if result.note.strip():
        receipt["note"] = clip(result.note, SUMMARY_CHARS)
    if commits:
        shown = [clip(str(c), 120) for c in commits[:MAX_COMMITS]]
        if len(commits) > MAX_COMMITS:
            shown.append(f"and {len(commits) - MAX_COMMITS} more")
        receipt["commits"] = shown
    entry, tests = _entry(raw), _tests(raw)
    if entry:
        receipt["entry"] = entry
    if tests:
        receipt["tests"] = tests
    cleared = str(cleared or raw.get("cleared") or "").strip().lower()
    if _COMMIT.match(cleared):
        receipt["cleared"] = cleared
    if result.decision_file is not None:
        receipt["decision_file"] = str(result.decision_file)
    if result.result_file is not None:
        receipt["result_file"] = str(result.result_file)
    receipt["written_at"] = utc_now()
    return receipt


def path_for(tasks_dir: Path, task_id: str) -> Path:
    return Path(tasks_dir) / f"{task_id}{RECEIPT_SUFFIX}"


def write(tasks_dir: Path, receipt: Mapping[str, Any]) -> Path:
    """Keep ``receipt`` as its task's, replacing an earlier attempt's whole."""
    path = path_for(tasks_dir, str(receipt["task"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        path, yaml.safe_dump(dict(receipt), sort_keys=False, allow_unicode=True, width=100)
    )
    return path


def read(tasks_dir: Path, task_id: str) -> dict[str, Any] | None:
    """The receipt kept for ``task_id``, or None when there is none to read."""
    try:
        raw = yaml.safe_load(path_for(tasks_dir, task_id).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return raw if isinstance(raw, dict) and raw.get("task") else None


def _paths(receipt: Mapping[str, Any]) -> str:
    parts = []
    if receipt.get("decision_file"):
        parts.append(f"decision `{receipt['decision_file']}`")
    if receipt.get("result_file"):
        parts.append(f"result `{receipt['result_file']}`")
    return ", ".join(parts)


def _title(receipt: Mapping[str, Any]) -> str:
    attempt = int(receipt.get("attempt") or 1)
    again = f", attempt {attempt}" if attempt > 1 else ""
    return (
        f"### `{receipt.get('task')}` ({receipt.get('kind')}): "
        f"{receipt.get('decision') or 'no decision'}{again}"
    )


def _full(receipt: Mapping[str, Any]) -> str:
    lines = [_title(receipt)]
    if receipt.get("summary"):
        lines += ["", str(receipt["summary"])]
    lines.append("")
    if receipt.get("note"):
        lines.append(f"- ended with: {receipt['note']}")
    commits = receipt.get("commits") or ()
    if commits:
        lines.append(f"- integrated: {'; '.join(str(c) for c in commits)}")
    if receipt.get("entry"):
        lines.append(f"- entry: `{receipt['entry']}`")
    if receipt.get("tests"):
        lines.append(f"- tests: {receipt['tests']}")
    if receipt.get("cleared"):
        lines.append(f"- cleared: `{receipt['cleared']}`")
    if _paths(receipt):
        lines.append(f"- {_paths(receipt)}")
    return "\n".join(lines).rstrip()


def render(receipts: Iterable[Mapping[str, Any]]) -> str:
    """The block a task's prompt carries its dependencies' receipts in; empty without one."""
    receipts = list(receipts)
    if not receipts:
        return ""
    blocks = [
        HEADING,
        "Each task below finished and passed its checks before this one started. The "
        "orchestrator took this from their decision and result files; read those for "
        "anything more. Your goal may have been written before they finished: where what "
        "they found changes your work, say so in your report.",
    ]
    used = sum(len(b) for b in blocks)
    for receipt in receipts:
        block = _full(receipt)
        if used + len(block) > RENDER_CHARS:
            block = " ".join(filter(None, [_title(receipt), _paths(receipt)]))
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


__all__ = [
    "CONSTITUENT_OPTIONAL",
    "DECISIONS_DIRNAME",
    "DEFAULT_DECISION_UNIT",
    "ENGINE_DEFECT_KEYS",
    "PLAN_DECISION_UNIT",
    "current_decision",
    "decision_dispatch_matches",
    "decision_file_instruction",
    "decision_path",
    "decision_unit",
    "decision_value",
    "engine_defect_report",
    "mark_decision_dispatch",
    "parse_constituents",
    "read_decision",
    "uncertified_report",
    "HEADING",
    "MAX_COMMITS",
    "RECEIPT_SUFFIX",
    "RENDER_CHARS",
    "SUMMARY_CHARS",
    "build",
    "clip",
    "TaskResult",
    "first_paragraph",
    "path_for",
    "read",
    "render",
    "write",
]
