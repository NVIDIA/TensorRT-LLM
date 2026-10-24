# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A finished task's receipt: what it hands the tasks that depend on it.

A task that depends on another needs what that one found: the commits it
landed, the numbers its worker measured, the caveats it wrote for whoever
comes next. When a task's worker finishes, :func:`build` reads them off its
decision file and its result, and :func:`write` keeps them beside its
envelope, in ``tasks/<id>.receipt.yaml``. When a task starts, its envelope
and its prompt carry the receipt of each task it depends on (:func:`render`,
under :data:`HEADING`), so what a finished task found reaches the work that
needs it without a planner turn to restate it in a goal.

Every part is bounded (:data:`SUMMARY_CHARS`, :data:`NOTE_CHARS`,
:data:`MAX_NOTES`, :data:`MAX_COMMITS`), and so is a prompt's block of them
(:data:`RENDER_CHARS`): past it, a receipt is one line and its paths.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..infra.dispatch import DISPATCH_DIRNAME
from .progress import decision_file_instruction

#: The heading a task's prompt carries its dependencies' receipts under.
HEADING = "## Receipts of the tasks this one depends on"
#: Beside the envelope ``tasks/<id>.json``.
RECEIPT_SUFFIX = ".receipt.yaml"
#: The most of the summary's first paragraph a receipt keeps.
SUMMARY_CHARS = 400
#: The most of one note for the orchestrator or the planner a receipt keeps, and how many.
NOTE_CHARS = 600
MAX_NOTES = 3
#: The most commits a receipt lists; a count says how many more there were.
MAX_COMMITS = 12
#: Past this many characters of a prompt's receipts, each further one is a single line.
RENDER_CHARS = 12_000

#: A paragraph a worker addressed to the orchestrator or the planner.
_NOTE_START = re.compile(r"^[\s>*_#-]*for the (?:orchestrator|planner)\b", re.IGNORECASE)
#: A paragraph addressed to anyone (the operator, say), which ends the one before it.
_ANY_START = re.compile(r"^[\s>*_#-]*for the [^:\n]{1,40}:", re.IGNORECASE)
#: A full or abbreviated commit id.
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
#: The line a dispatch report's result opens with.
_RETURNED = re.compile(r"^- returned: `[^`]*`\s*$")
_PENDING = "_pending_"
_RESULT_HEADING = "\n## Result\n"
#: What a dispatch prompt says right before the one path its turn writes.
_DECISION_LEAD = decision_file_instruction(Path("<path>")).split("<path>")[0]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def addressed_notes(*texts: Any, limit: int = NOTE_CHARS, count: int = MAX_NOTES) -> list[str]:
    """The paragraphs of ``texts`` addressed to the orchestrator or the planner, in order.

    A note opens with ``For the orchestrator`` or ``For the planner`` (bold,
    a bullet or a heading included) and runs to a blank line, a heading or a
    paragraph addressed to someone else. A heading's own blank line does not
    end it. The same note in two texts is kept once.
    """
    notes: list[str] = []
    seen: set[str] = set()

    def keep(lines: list[str]) -> None:
        text = clip("\n".join(lines), limit)
        key = " ".join(text.lower().split())[:120]
        if text and key not in seen:
            seen.add(key)
            notes.append(text)

    for text in texts:
        current: list[str] | None = None
        heading = False
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if _NOTE_START.match(line):
                if current:
                    keep(current)
                heading = line.startswith("#")
                current = [re.sub(r"^#+\s*", "", line)]
                continue
            if current is None:
                continue
            if not line:
                if heading and len(current) == 1:
                    continue
                keep(current)
                current = None
            elif line.startswith("#") or _ANY_START.match(line):
                keep(current)
                current = None
            else:
                current.append(line)
        if current:
            keep(current)
    return notes[:count]


def decision_named_in(report: Path | None) -> Path | None:
    """The decision file a dispatch report's prompt told its turn to write, or None."""
    text = _read(report)
    at = text.rfind(_DECISION_LEAD)
    if at < 0:
        return None
    path = text[at + len(_DECISION_LEAD) :].split("`", 1)[0].strip()
    return Path(path) if path else None


def result_text(path: Path | None) -> str:
    """The turn's final message: a dispatch report's ``Result`` section, or a whole result file."""
    text = _read(path)
    if _RESULT_HEADING in text:
        text = text.rpartition(_RESULT_HEADING)[2]
        lines = text.strip().splitlines()
        if lines and _RETURNED.match(lines[0]):
            lines = lines[1:]
        text = "\n".join(lines).strip()
        return "" if text == _PENDING else text
    return text.strip()


def dispatch_result(workspace: Path, decision_file: Path) -> Path | None:
    """The external transport's result file for the turn that was to write ``decision_file``."""
    root = Path(workspace) / DISPATCH_DIRNAME
    if not root.is_dir():
        return None
    wanted = os.path.normpath(str(decision_file))
    for request in sorted(root.glob("*.request.json"), key=_dispatch_order, reverse=True):
        try:
            data = json.loads(request.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not data.get("decision_file"):
            continue
        named = Path(str(data["decision_file"]))
        if not named.is_absolute() and data.get("cwd"):
            named = Path(str(data["cwd"])) / named
        if os.path.normpath(str(named)) == wanted:
            result = Path(str(data.get("result_file") or ""))
            return result if result.is_file() else None
    return None


def _dispatch_order(request: Path) -> int:
    """A request's number (``<n>-<role>.request.json``), which runs past three digits."""
    head = request.name.split("-", 1)[0]
    return int(head) if head.isdigit() else -1


def _read(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _decision_mapping(path: Path | None) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(_read(path)) if path is not None else None
    except yaml.YAMLError:
        return {}
    return raw if isinstance(raw, dict) else {}


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
    decision: str = "",
    note: str = "",
    attempt: int = 1,
    decision_file: Path | None = None,
    result_file: Path | None = None,
    commits: Sequence[str] = (),
    workspace: Path | None = None,
    cleared: str = "",
) -> dict[str, Any]:
    """The receipt of task ``task_id``, read off its decision file and its result.

    ``decision`` is what the task ended with, or, when empty, the decision
    file's. A result file whose turn has no final message in it (a dispatch
    report a stopped process left pending) gives way to the external
    transport's result file for the same decision, under ``workspace``.
    ``cleared`` is the commit a regression check found clean, the next one's
    base (``checks/testcache.default_base``).
    """
    raw = _decision_mapping(decision_file)
    result = result_text(result_file)
    if not result and decision_file is not None and workspace is not None:
        answered = dispatch_result(workspace, decision_file)
        if answered is not None:
            result_file, result = answered, result_text(answered)
    receipt: dict[str, Any] = {
        "task": task_id,
        "kind": kind,
        "attempt": int(attempt),
        "decision": decision or str(raw.get("decision", "")).strip(),
        "summary": first_paragraph(raw.get("summary")),
    }
    if note.strip():
        receipt["note"] = clip(note, SUMMARY_CHARS)
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
    if decision_file is not None:
        receipt["decision_file"] = str(decision_file)
    if result_file is not None:
        receipt["result_file"] = str(result_file)
    listed = raw.get("notes_for_orchestrator")
    if isinstance(listed, str):
        listed = [listed]
    if not isinstance(listed, list):
        listed = []
    notes = addressed_notes(result, raw.get("summary"))
    notes += [clip(str(n), NOTE_CHARS) for n in listed if str(n).strip()]
    if notes:
        receipt["notes"] = notes[:MAX_NOTES]
    receipt["written_at"] = _now()
    return receipt


def path_for(tasks_dir: Path, task_id: str) -> Path:
    return Path(tasks_dir) / f"{task_id}{RECEIPT_SUFFIX}"


def write(tasks_dir: Path, receipt: Mapping[str, Any]) -> Path:
    """Keep ``receipt`` as its task's, replacing an earlier attempt's whole."""
    path = path_for(tasks_dir, str(receipt["task"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        yaml.safe_dump(dict(receipt), sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    os.replace(partial, path)
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
    for note in receipt.get("notes") or ():
        body = str(note).replace("\n", "\n  ")
        lines.append(f"- its worker wrote: {body}")
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
    "HEADING",
    "MAX_COMMITS",
    "MAX_NOTES",
    "NOTE_CHARS",
    "RECEIPT_SUFFIX",
    "RENDER_CHARS",
    "SUMMARY_CHARS",
    "addressed_notes",
    "build",
    "clip",
    "decision_named_in",
    "dispatch_result",
    "first_paragraph",
    "path_for",
    "read",
    "render",
    "result_text",
    "write",
]
