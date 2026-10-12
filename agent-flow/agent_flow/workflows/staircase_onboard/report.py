# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reports generated from the ledger, and the check that they agree with it.

The DeepSeek-V4.1 run kept ``progress.json``, the gap-closure report and the
final report by hand. They drifted apart in ten recorded ways. One summary
said "16/16 PASS, no finding" over a regression table that held a FINDING.
The progress page still listed an A/B as pending three hours after an
addendum closed it, and named a branch and a target commit that were no
longer current. Every one of those was a sentence written about evidence
instead of from it.

So the facts in a report are rendered, never typed:

- ``progress.json`` and ``progress.html`` (the latter offline and
  self-contained, ported from that run's ``scripts/render_progress.py``) are
  generated from the ledger, the checkpoint and the journal;
- the final report's gate tables are generated into marked regions of
  ``final-report.md``, and everything outside those regions is the
  operator's narrative.

Narrative still exists, because judgement does: feature notes, workstream
items, prose conclusions. It lives in ``progress.overlay.yaml`` and outside
the generated regions, and :func:`check_consistency` reads it against the
ledger. The check fails on "no finding" while a FINDING verdict exists, an
"N/N PASS" or "all gates pass" while a verdict fails or is stale, a pending
item whose evidence is in fact fresh and passing, a closed item whose
evidence is not, a "final tree" that is not the tree of the last
regression, and a ``progress.json`` that was generated from an older ledger
or was edited after it was generated.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from .freshness import (
    Assessment,
    Binding,
    Regression,
    RerunEntry,
    assess,
    last_regression,
    regression_binding,
    rerun_queue,
)
from .ledger import (
    OUTCOME_FAIL,
    OUTCOME_FINDING,
    OUTCOME_PASS,
    Ledger,
    LedgerPolicy,
    Verdict,
    load_task,
    select,
    utc_now,
)
from .state import (
    STAGE_ASSEMBLE,
    STAGE_DELIVER,
    STAGE_PLAN,
    STAGE_PREFLIGHT,
    STAGE_SMITH,
    STATE_FILENAME,
    load_state,
)
from .task_schema import assembly_units

PROGRESS_JSON = "progress.json"
PROGRESS_HTML = "progress.html"
OVERLAY_FILENAME = "progress.overlay.yaml"
FINAL_REPORT = "final-report.md"
SCHEMA_VERSION = 1

STATUS_DELIVERED = "DELIVERED"
STATUS_STALE = "STALE"
STATUS_BLOCKED = "BLOCKED"
STATUS_IN_PROGRESS = "IN PROGRESS"
STATUS_NOT_STARTED = "NOT STARTED"
STATUS_DONE = "DONE"
STATUS_FRESH = "FRESH"

#: Item statuses that claim the work is finished, and those that claim it is not.
CLOSED_STATUSES = frozenset(
    {"DONE", "CLOSED", "PASS", "GATED-PASS", "DELIVERED", "CONFIRMED", "RESOLVED"}
)
OPEN_STATUSES = frozenset(
    {"PENDING", "IN PROGRESS", "OPEN", "NOT STARTED", "RUNNING", "QUEUED", "PARTIAL", "BLOCKED"}
)

_STAGES = (STAGE_PREFLIGHT, STAGE_PLAN, STAGE_SMITH, STAGE_ASSEMBLE, STAGE_DELIVER)
_GENERATED = re.compile(
    r"<!-- BEGIN GENERATED: (?P<name>[a-z-]+) -->\n.*?<!-- END GENERATED: (?P=name) -->",
    re.S,
)
_JOURNAL_EVENT = re.compile(r"^- `(?P<time>[^`]+)` \*\*(?P<what>.+?)\*\*(?: — (?P<detail>.*))?$")
_TIMELINE_LIMIT = 200


# ------------------------------------------------------------------- facts


@dataclass(frozen=True)
class Facts:
    """Everything a report says, read from the ledger, the checkpoint and the journal."""

    latest: tuple[Verdict, ...] = ()
    assessments: Mapping[tuple[str, str, str], Assessment] = field(default_factory=dict)
    queue: tuple[RerunEntry, ...] = ()
    runs: Mapping[tuple[str, str, str], int] = field(default_factory=dict)
    regression: Regression | None = None
    binding: Binding | None = None
    ledger_digest: str = ""
    verdict_count: int = 0
    state: Mapping[str, Any] = field(default_factory=dict)
    units: tuple[str, ...] = ()
    repo: str = ""
    events: tuple[Mapping[str, str], ...] = ()

    @property
    def findings(self) -> list[Verdict]:
        return [v for v in self.latest if v.outcome == OUTCOME_FINDING]

    @property
    def failing(self) -> list[Verdict]:
        return [v for v in self.latest if v.outcome == OUTCOME_FAIL]

    @property
    def stale(self) -> list[Verdict]:
        return [v for v in self.latest if self.is_stale(v)]

    def is_stale(self, verdict: Verdict) -> bool:
        item = self.assessments.get(verdict.key)
        return bool(item and item.stale)

    def holds(self, verdict: Verdict) -> bool:
        """Passed, and still describes the tree."""
        return verdict.passed and not self.is_stale(verdict)

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.latest),
            "pass": sum(v.outcome == OUTCOME_PASS for v in self.latest),
            "finding": len(self.findings),
            "fail": len(self.failing),
            "stale": len(self.stale),
            "queued": len(self.queue),
        }

    def status(self) -> str:
        """The run's one-word status, derived rather than declared."""
        state = self.state
        if not state:
            return STATUS_NOT_STARTED
        if state.get("done"):
            if state.get("verdict") != "delivered":
                return STATUS_BLOCKED
            if self.stale or self.failing:
                return STATUS_STALE
            if self.binding is not None and self.binding.claim and not self.binding.ok:
                return STATUS_STALE
            return STATUS_DELIVERED
        return STATUS_IN_PROGRESS


def _read_journal_events(path: Path) -> list[dict[str, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, UnicodeDecodeError):
        return []
    events = []
    for line in lines:
        match = _JOURNAL_EVENT.match(line)
        if match:
            events.append(
                {
                    "time": match.group("time"),
                    "event": match.group("what")
                    + (f" — {match.group('detail')}" if match.group("detail") else ""),
                }
            )
    return events[-_TIMELINE_LIMIT:]


def collect_facts(
    workspace: str | Path,
    *,
    repo_root: str | Path | None = None,
    task_data: Mapping[str, Any] | None = None,
    scope: Callable[[Verdict], bool] | None = None,
    policy: LedgerPolicy | None = None,
    check_history: bool = False,
) -> Facts:
    """Read the workspace's ledger, checkpoint and journal into :class:`Facts`.

    ``repo_root`` and ``task_data`` default to the workspace's own
    ``task.yaml``. ``check_history`` adds the git-history switch check, which
    costs a ``git diff`` per verdict commit; delivery wants it, a page
    refresh after every checkpoint does not.
    """
    workspace = Path(workspace)
    if task_data is None:
        try:
            task_data = load_task(workspace / "task.yaml")
        except (OSError, yaml.YAMLError):
            task_data = None
    task_data = task_data or {}
    repo = str(repo_root or task_data.get("trtllm_repo_path") or "")
    policy = policy or LedgerPolicy()

    ledger = Ledger.in_workspace(workspace)
    verdicts = ledger.verdicts()
    runs: dict[tuple[str, str, str], int] = {}
    for verdict in verdicts:
        runs[verdict.key] = runs.get(verdict.key, 0) + 1
    latest = [v for v in ledger.latest().values() if scope is None or scope(v)]
    assessed: list[Assessment] = []
    binding = None
    if repo:
        assessed = assess(
            latest, repo, task_data=task_data or None, policy=policy, check_history=check_history
        )
        binding = regression_binding(ledger, repo, policy=policy)
    state: dict[str, Any] = {}
    state_path = workspace / STATE_FILENAME
    if state_path.is_file():
        try:
            state = dict(load_state(state_path).__dict__)
        except (OSError, ValueError, KeyError):
            state = {}
    return Facts(
        latest=tuple(latest),
        assessments={a.verdict.key: a for a in assessed},
        queue=tuple(rerun_queue(assessed)) if assessed else (),
        runs=runs,
        regression=last_regression(ledger),
        binding=binding,
        ledger_digest=ledger.digest(),
        verdict_count=len(verdicts),
        state=state,
        units=tuple(assembly_units(task_data)) if task_data else (),
        repo=repo,
        events=tuple(_read_journal_events(workspace / "journal.md")),
    )


# -------------------------------------------------------------- progress.json


def headline(facts: Facts) -> str:
    """The one-sentence summary, rendered from counts so it cannot overstate them."""
    c = facts.counts()
    if not c["total"]:
        return "No verdict recorded yet."
    parts = [
        f"{c['total']} verdict(s): {c['pass']} PASS, {c['finding']} FINDING, {c['fail']} FAIL",
        f"{c['stale']} stale" if c["stale"] else "none stale",
    ]
    if facts.regression is not None:
        bound = ""
        if facts.binding is not None and facts.binding.claim:
            bound = ", bound to HEAD" if facts.binding.ok else ", NOT the current HEAD"
        parts.append(
            f"last full regression {len(facts.regression.keys)} verdict(s) on "
            f"{facts.regression.tree.short()}{bound}"
        )
    else:
        parts.append("no full regression recorded")
    return "; ".join(parts) + "."


def _stage_rows(state: Mapping[str, Any]) -> list[dict[str, str]]:
    current = str(state.get("stage") or STAGE_PREFLIGHT)
    done = bool(state.get("done"))
    verdict = str(state.get("verdict") or "")
    position = _STAGES.index(current) if current in _STAGES else 0
    rows = []
    for index, stage in enumerate(_STAGES):
        if index < position:
            status = STATUS_DONE
        elif index > position:
            status = STATUS_NOT_STARTED
        elif done and verdict == "delivered":
            status = STATUS_DELIVERED
        elif done:
            status = STATUS_BLOCKED
        else:
            status = STATUS_IN_PROGRESS
        rows.append({"stage": stage, "status": status})
    return rows


def _unit_rows(facts: Facts) -> list[dict[str, Any]]:
    state = facts.state
    done = set(state.get("units_done") or ())
    in_assembly = state.get("stage") == STAGE_ASSEMBLE and not state.get("done")
    index = int(state.get("unit_index") or 0)
    rows = []
    for position, unit in enumerate(facts.units):
        verdicts = [v for v in facts.latest if v.unit == unit]
        holding = sum(facts.holds(v) for v in verdicts)
        if unit in done:
            status = STATUS_DONE if holding == len(verdicts) else STATUS_STALE
        elif in_assembly and position == index:
            status = STATUS_IN_PROGRESS
        else:
            status = STATUS_NOT_STARTED
        rows.append(
            {
                "unit": unit,
                "status": status,
                "gates": f"{holding}/{len(verdicts)} hold" if verdicts else "no verdict",
            }
        )
    return rows


def gate_row(facts: Facts, verdict: Verdict) -> dict[str, Any]:
    item = facts.assessments.get(verdict.key)
    return {
        "unit": verdict.unit,
        "member": verdict.member,
        "gate": verdict.gate,
        "result": verdict.outcome,
        "fresh": STATUS_STALE if item and item.stale else STATUS_FRESH,
        "value_vs_anchor": verdict.value_text(),
        "tree": verdict.tree.short(),
        "native": verdict.native.short(),
        "timestamp": verdict.timestamp,
        "runs": facts.runs.get(verdict.key, 1),
        "reasons": list(item.reasons) if item else [],
        "id": verdict.id,
    }


def _item_evidence(facts: Facts, item: Mapping[str, Any]) -> tuple[list[Verdict], list[str]]:
    """The verdicts an overlay item cites, and what keeps them from closing it."""
    selectors = item.get("evidence") or []
    if isinstance(selectors, Mapping):
        selectors = [selectors]
    matched: list[Verdict] = []
    for selector in selectors:
        if not isinstance(selector, Mapping):
            continue
        for verdict in select(
            facts.latest,
            gate=str(selector.get("gate", "*")),
            unit=str(selector.get("unit", "*")),
            member=str(selector.get("member", "*")),
        ):
            if verdict not in matched:
                matched.append(verdict)
    gaps = [
        f"{v.label()} is {'stale' if facts.is_stale(v) else v.outcome}"
        for v in matched
        if not facts.holds(v)
    ]
    queued = {e.key for e in facts.queue}
    gaps.extend(f"{v.label()} is queued for a re-run" for v in matched if v.key in queued)
    return matched, gaps


def _items(facts: Facts, overlay: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in overlay.get("items") or ():
        if not isinstance(item, Mapping):
            continue
        row = {str(k): v for k, v in item.items()}
        if item.get("evidence"):
            matched, gaps = _item_evidence(facts, item)
            if not matched:
                row["evidence_status"] = "cites no verdict"
            elif gaps:
                row["evidence_status"] = "open: " + "; ".join(gaps)
            else:
                row["evidence_status"] = f"closed by {len(matched)} fresh passing verdict(s)"
        rows.append(row)
    return rows


def build_progress(
    facts: Facts, overlay: Mapping[str, Any] | None = None, *, workspace: str | Path = ""
) -> dict[str, Any]:
    """The ``progress.json`` document. Every fact in it comes from ``facts``."""
    overlay = dict(overlay or {})
    state = facts.state
    meta = dict(overlay.get("meta") or {})
    meta.update(
        {
            "title": meta.get("title") or f"Onboarding progress — {state.get('slug') or 'run'}",
            "slug": state.get("slug", ""),
            "branch": state.get("branch", ""),
            "repo": facts.repo,
            "stage": state.get("stage", ""),
            "verdict": state.get("verdict", ""),
            "status": facts.status(),
            "last_updated": utc_now(),
            "final_report": FINAL_REPORT
            if workspace and (Path(workspace) / FINAL_REPORT).is_file()
            else meta.get("final_report", ""),
        }
    )
    regression = facts.regression
    doc: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "generated_at": meta["last_updated"],
        "generated_from": {
            "ledger": "ledger.jsonl",
            "ledger_digest": facts.ledger_digest,
            "verdicts": facts.verdict_count,
            "state": STATE_FILENAME,
        },
        "meta": meta,
        "summary": {"headline": headline(facts), "gates": facts.counts()},
        "pipeline_stages": _stage_rows(state),
        "units": _unit_rows(facts),
        "gates": {
            "note": "Latest verdict per (gate, unit, member), generated from the ledger.",
            "rows": [gate_row(facts, v) for v in facts.latest],
        },
        "findings": [
            {
                "gate": v.gate,
                "unit": v.unit,
                "member": v.member,
                "value_vs_anchor": v.value_text(),
                "detail": v.detail[:300],
            }
            for v in facts.findings
        ],
        "rerun_queue": [e.to_json() for e in facts.queue],
        "regression": {
            "last": None
            if regression is None
            else {
                "tree": regression.tree.short(),
                "tree_object": regression.tree.tree,
                "commit": regression.tree.commit,
                "native": regression.native.short(),
                "verdicts": len(regression.keys),
                "timestamp": regression.timestamp,
            },
            "binding": None
            if facts.binding is None
            else {
                "ok": facts.binding.ok,
                "claim": facts.binding.claim,
                "detail": facts.binding.detail,
            },
        },
        "items": _items(facts, overlay),
        "timeline": [dict(e) for e in facts.events] + list(overlay.get("timeline") or ()),
    }
    for key in ("features", "streams", "workflow_feedback", "notes"):
        if overlay.get(key):
            doc[key] = overlay[key]
    return doc


# ------------------------------------------------------------- consistency


@dataclass(frozen=True)
class Contradiction:
    """A statement in a report that the ledger refutes."""

    where: str
    claim: str
    fact: str

    def render(self) -> str:
        return f"{self.where}: says {self.claim!r}, but {self.fact}"


_NO_FINDING = re.compile(
    r"\bno\s+(?:new\s+|open\s+)?findings?\b|\bfindings?\s*[:=]\s*(?:none|0)\b", re.I
)
_ALL_PASS = re.compile(r"\ball\s+(?:\d+\s+)?(?:gates?|verdicts?)\s+(?:pass(?:ed)?|green)\b", re.I)
_N_OF_N = re.compile(r"\b(\d+)\s*/\s*(\d+)\s+(?:gates?\s+)?(?:PASS(?:ED)?|green)\b", re.I)
_TREE_CLAIM = re.compile(
    r"\b(?:final|delivered|regression|merge)\s+(?:tree|head|commit)\b[^0-9a-f`]{0,12}`?"
    r"(?P<sha>[0-9a-f]{7,40})\b",
    re.I,
)


def check_claims(text: str, facts: Facts, *, where: str) -> list[Contradiction]:
    """Prose claims in ``text`` that the ledger refutes."""
    problems: list[Contradiction] = []
    counts = facts.counts()
    for match in _NO_FINDING.finditer(text):
        if facts.findings:
            names = ", ".join(v.label() for v in facts.findings[:5])
            problems.append(
                Contradiction(where, match.group(0), f"the ledger holds a FINDING: {names}")
            )
    not_holding = [v for v in facts.latest if not facts.holds(v)]
    for match in _ALL_PASS.finditer(text):
        if not_holding:
            names = ", ".join(v.label() for v in not_holding[:5])
            problems.append(Contradiction(where, match.group(0), f"these do not hold now: {names}"))
    for match in _N_OF_N.finditer(text):
        claimed, total = int(match.group(1)), int(match.group(2))
        if claimed != total:
            continue
        sizes = {counts["total"]}
        if facts.regression is not None:
            sizes.add(len(facts.regression.keys))
        if total not in sizes:
            problems.append(
                Contradiction(
                    where,
                    match.group(0),
                    f"the ledger's latest verdicts number {counts['total']} and the last "
                    f"regression {len(facts.regression.keys) if facts.regression else 0}",
                )
            )
        elif not_holding:
            names = ", ".join(v.label() for v in not_holding[:5])
            problems.append(Contradiction(where, match.group(0), f"these do not hold now: {names}"))
    regression = facts.regression
    for match in _TREE_CLAIM.finditer(text):
        sha = match.group("sha").lower()
        if regression is None:
            problems.append(Contradiction(where, match.group(0), "no full regression is recorded"))
        elif not (
            regression.tree.commit.startswith(sha)
            or regression.tree.tree.startswith(sha)
            or sha in regression.tree.describe
        ):
            problems.append(
                Contradiction(
                    where,
                    match.group(0),
                    f"the last full regression ran on {regression.tree.short()} "
                    f"({regression.tree.commit[:12]})",
                )
            )
    return problems


def check_items(items: Iterable[Mapping[str, Any]], facts: Facts) -> list[Contradiction]:
    """Overlay items whose stated status their cited evidence refutes."""
    problems: list[Contradiction] = []
    for item in items:
        if not isinstance(item, Mapping) or not item.get("evidence"):
            continue
        name = f"item {item.get('id') or item.get('title') or '?'}"
        status = str(item.get("status") or "").upper()
        matched, gaps = _item_evidence(facts, item)
        if status in CLOSED_STATUSES:
            if not matched:
                problems.append(Contradiction(name, status, "its evidence matches no verdict"))
            elif gaps:
                problems.append(Contradiction(name, status, "; ".join(gaps)))
        elif status in OPEN_STATUSES and matched and not gaps:
            problems.append(
                Contradiction(
                    name,
                    status,
                    f"it is in fact closed: {len(matched)} fresh passing verdict(s) "
                    f"({', '.join(v.label() for v in matched[:3])})",
                )
            )
    return problems


def _strings(value: Any, path: str) -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, inner in value.items():
            yield from _strings(inner, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, inner in enumerate(value):
            yield from _strings(inner, f"{path}[{index}]")


def check_overlay(overlay: Mapping[str, Any], facts: Facts) -> list[Contradiction]:
    """The operator's narrative and items, read against the ledger."""
    problems = check_items(overlay.get("items") or (), facts)
    for key, value in overlay.items():
        if key == "timeline":
            continue
        for where, text in _strings(value, f"overlay.{key}"):
            problems.extend(check_claims(text, facts, where=where))
    return problems


def check_progress(doc: Mapping[str, Any], facts: Facts) -> list[Contradiction]:
    """A ``progress.json`` on disk, read against the ledger it claims to come from."""
    problems: list[Contradiction] = []
    digest = str((doc.get("generated_from") or {}).get("ledger_digest") or "")
    if digest != facts.ledger_digest:
        problems.append(
            Contradiction(
                PROGRESS_JSON,
                f"generated from ledger {digest or '(none)'}",
                f"the ledger is now {facts.ledger_digest or '(empty)'}; the page is stale",
            )
        )
    status = str((doc.get("meta") or {}).get("status") or "")
    if status and status != facts.status():
        problems.append(
            Contradiction(f"{PROGRESS_JSON} meta.status", status, f"the facts say {facts.status()}")
        )
    latest = {v.key: v for v in facts.latest}
    for row in (doc.get("gates") or {}).get("rows") or ():
        key = (str(row.get("gate")), str(row.get("unit")), str(row.get("member")))
        verdict = latest.get(key)
        where = f"{PROGRESS_JSON} gate {key[0]}@{key[2]} [{key[1]}]"
        if verdict is None:
            problems.append(Contradiction(where, str(row.get("result")), "no such verdict"))
            continue
        if str(row.get("result")) != verdict.outcome:
            problems.append(
                Contradiction(where, str(row.get("result")), f"the ledger says {verdict.outcome}")
            )
        fresh = STATUS_STALE if facts.is_stale(verdict) else STATUS_FRESH
        if row.get("fresh") and row.get("fresh") != fresh:
            problems.append(Contradiction(where, str(row.get("fresh")), f"it is {fresh}"))
    queued = {e.key for e in facts.queue}
    for entry in doc.get("rerun_queue") or ():
        key = (str(entry.get("gate")), str(entry.get("unit")), str(entry.get("member")))
        if key not in queued:
            problems.append(
                Contradiction(
                    f"{PROGRESS_JSON} rerun_queue",
                    f"{key[0]}@{key[2]} [{key[1]}] pending",
                    "it is in fact closed: its latest verdict holds",
                )
            )
    summary = doc.get("summary") or {}
    if isinstance(summary, Mapping):
        for where, text in _strings(summary, f"{PROGRESS_JSON} summary"):
            problems.extend(check_claims(text, facts, where=where))
    return problems


def narrative(text: str) -> str:
    """``text`` with every generated region removed: what a human wrote."""
    return _GENERATED.sub("", text)


def check_consistency(
    workspace: str | Path,
    *,
    facts: Facts | None = None,
    documents: Sequence[str | Path] = (),
) -> list[Contradiction]:
    """Every report in the workspace, plus ``documents``, read against the ledger."""
    workspace = Path(workspace)
    facts = facts or collect_facts(workspace)
    problems: list[Contradiction] = []
    overlay = read_overlay(workspace)
    problems.extend(check_overlay(overlay, facts))
    progress = workspace / PROGRESS_JSON
    if progress.is_file():
        try:
            problems.extend(check_progress(json.loads(progress.read_text("utf-8")), facts))
        except json.JSONDecodeError as exc:
            problems.append(Contradiction(PROGRESS_JSON, "a JSON document", f"it is not: {exc}"))
    for path in [workspace / FINAL_REPORT, *map(Path, documents)]:
        if path.is_file():
            text = narrative(path.read_text(encoding="utf-8"))
            problems.extend(check_claims(text, facts, where=path.name))
    unique: dict[str, Contradiction] = {}
    for problem in problems:
        unique.setdefault(problem.render(), problem)
    return list(unique.values())


# ------------------------------------------------------------- final report


def gate_table_markdown(facts: Facts) -> str:
    if not facts.latest:
        return "_No verdict recorded._"
    lines = [
        "| unit | member | gate | result | fresh | value vs threshold | tree | measured | runs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for verdict in facts.latest:
        row = gate_row(facts, verdict)
        cells = [
            row["unit"] or "-",
            row["member"] or "-",
            row["gate"],
            f"**{row['result']}**" if row["result"] != OUTCOME_PASS else row["result"],
            row["fresh"],
            row["value_vs_anchor"].replace("|", "\\|"),
            f"`{row['tree']}`",
            row["timestamp"],
            str(row["runs"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def findings_markdown(facts: Facts) -> str:
    if not facts.findings:
        return "FINDING verdicts: 0."
    return "\n".join(
        f"- **{v.label()}**: {v.value_text()} — {v.detail[:300]}" for v in facts.findings
    )


def freshness_markdown(facts: Facts) -> str:
    stale = [facts.assessments[v.key] for v in facts.stale]
    lines = [f"Stale verdicts: {len(stale)}. Queued for a re-run: {len(facts.queue)}."]
    for item in stale:
        lines.append(f"- {item.verdict.label()}: {'; '.join(item.reasons)}")
    return "\n".join(lines)


def regression_markdown(facts: Facts) -> str:
    regression = facts.regression
    if regression is None:
        lines = ["No full regression is recorded."]
    else:
        lines = [
            f"Last full regression: {len(regression.keys)} verdict(s) on tree "
            f"`{regression.tree.short()}` (commit `{regression.tree.commit[:12]}`, tree object "
            f"`{regression.tree.tree[:12]}`), native build `{regression.native.short()}`, "
            f"recorded {regression.timestamp}."
        ]
    if facts.binding is not None:
        verdict = "holds" if facts.binding.ok else "DOES NOT HOLD"
        lines.append(f"Binding to the tree under merge {verdict}: {facts.binding.detail}.")
    return "\n\n".join(lines)


_SECTIONS: tuple[tuple[str, str, Callable[[Facts], str]], ...] = (
    ("summary", "", headline),
    ("gate-table", "## Gates", gate_table_markdown),
    ("findings", "## Findings", findings_markdown),
    ("freshness", "## Evidence freshness", freshness_markdown),
    ("regression", "## Final regression", regression_markdown),
)


def _region(name: str, body: str) -> str:
    return f"<!-- BEGIN GENERATED: {name} -->\n{body.rstrip()}\n<!-- END GENERATED: {name} -->"


def render_final_report(facts: Facts, existing: str | None = None) -> str:
    """Refresh the generated regions of a final report, or write a new one.

    Text outside the ``BEGIN/END GENERATED`` markers is kept as written; a
    region the existing report lacks is appended.
    """
    if existing is None:
        title = f"# Final report — {facts.state.get('slug') or 'onboarding run'}"
        parts = [title]
        for name, heading, render in _SECTIONS:
            if heading:
                parts.append(heading)
            parts.append(_region(name, render(facts)))
        parts.append(
            "## Narrative\n\nWritten by the operator. Everything outside the generated "
            "regions is read against the ledger by `report.check_consistency`."
        )
        return "\n\n".join(parts) + "\n"
    text = existing
    for name, heading, render in _SECTIONS:
        region = _region(name, render(facts))
        pattern = re.compile(
            rf"<!-- BEGIN GENERATED: {name} -->\n.*?<!-- END GENERATED: {name} -->", re.S
        )
        if pattern.search(text):
            text = pattern.sub(lambda _m, r=region: r, text)
        else:
            text = text.rstrip() + "\n\n" + (f"{heading}\n\n" if heading else "") + region + "\n"
    return text


# -------------------------------------------------------------------- HTML

# Status -> (css class, icon). Color is never the only signal: every badge
# also carries the icon and the literal status text.
STATUS_MAP = {
    "NOT STARTED": ("status-neutral", "○"),
    "RESEARCH": ("status-info", "◐"),
    "IN PROGRESS": ("status-warning", "◑"),
    "GATED-PASS": ("status-good", "◆"),
    "DONE": ("status-good", "●"),
    "PASS": ("status-good", "●"),
    "FRESH": ("status-good", "✓"),
    "BLOCKED": ("status-critical", "✕"),
    "FAIL": ("status-critical", "✕"),
    "FINDING": ("status-serious", "▲"),
    "STALE": ("status-serious", "↻"),
    "QUEUED": ("status-warning", "◇"),
    "DESCOPED": ("status-neutral", "—"),
    "OPEN": ("status-warning", "◇"),
    "PENDING": ("status-warning", "◇"),
    "DELIVERED": ("status-good", "★"),
    "CONFIRMED": ("status-good", "✓"),
    "PARTIAL": ("status-warning", "◑"),
}


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def badge(status: str) -> str:
    status = str(status or "NOT STARTED")
    css_class, icon = STATUS_MAP.get(status.upper(), ("status-neutral", "○"))
    return (
        f'<span class="badge {css_class}"><span class="badge-icon" aria-hidden="true">{icon}'
        f'</span><span class="badge-label">{esc(status)}</span></span>'
    )


def _link(path: Any) -> str:
    if not path:
        return '<span class="muted">&mdash;</span>'
    return f'<a href="{esc(path)}">{esc(path)}</a>'


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]], empty: str) -> str:
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    if not body:
        return f'<p class="muted">{esc(empty)}</p>'
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    return (
        f'<div class="table-scroll"><table class="data-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _card(title: str, body: str) -> str:
    return f'<section class="card"><h2>{esc(title)}</h2>{body}</section>'


def render_html(doc: Mapping[str, Any]) -> str:
    """Render a ``progress.json`` document into one offline page."""
    meta = doc.get("meta") or {}
    summary = doc.get("summary") or {}
    counts = summary.get("gates") or {}
    regression = doc.get("regression") or {}
    binding = regression.get("binding") or {}
    consistency = doc.get("consistency") or {}

    meta_rows = "".join(
        f'<div class="meta-row"><div class="meta-key">{esc(k)}</div>'
        f'<div class="meta-val">{esc(v)}</div></div>'
        for k, v in meta.items()
        if k not in ("title", "status", "last_updated") and v not in ("", None)
    )
    tiles = "".join(
        f'<div class="tile"><div class="tile-label">{esc(label)}</div>'
        f'<div class="tile-value">{esc(counts.get(key, 0))}</div></div>'
        for key, label in (
            ("total", "verdicts"),
            ("pass", "pass"),
            ("finding", "finding"),
            ("fail", "fail"),
            ("stale", "stale"),
            ("queued", "re-run queue"),
        )
    )
    parts = [
        _card(
            "Overview",
            f'<p class="headline">{esc(summary.get("headline"))}</p>'
            f'<div class="tile-row">{tiles}</div>'
            f'<div class="meta-grid" style="margin-top:12px">{meta_rows}</div>',
        )
    ]
    problems = consistency.get("problems") or []
    if problems:
        parts.append(
            _card(
                "Consistency check: contradictions",
                '<ul class="feedback-list">'
                + "".join(f"<li>{badge('BLOCKED')} {esc(p)}</li>" for p in problems)
                + "</ul>",
            )
        )
    parts.append(
        _card(
            "Pipeline stages",
            _table(
                ("Stage", "Status"),
                ((esc(r["stage"]), badge(r["status"])) for r in doc.get("pipeline_stages") or ()),
                "No stages recorded.",
            ),
        )
    )
    parts.append(
        _card(
            "Units",
            _table(
                ("Unit", "Status", "Gates"),
                (
                    (esc(r["unit"]), badge(r["status"]), esc(r["gates"]))
                    for r in doc.get("units") or ()
                ),
                "No units.",
            ),
        )
    )
    gate_rows = (doc.get("gates") or {}).get("rows") or ()
    parts.append(
        _card(
            "Gates",
            _table(
                (
                    "Unit",
                    "Member",
                    "Gate",
                    "Result",
                    "Fresh",
                    "Value vs threshold",
                    "Tree",
                    "Measured",
                    "Runs",
                    "Why stale",
                ),
                (
                    (
                        esc(r.get("unit")),
                        esc(r.get("member")),
                        esc(r.get("gate")),
                        badge(r.get("result")),
                        badge(r.get("fresh")),
                        esc(r.get("value_vs_anchor")),
                        f"<code>{esc(r.get('tree'))}</code>",
                        esc(r.get("timestamp")),
                        esc(r.get("runs")),
                        esc("; ".join(r.get("reasons") or ())),
                    )
                    for r in gate_rows
                ),
                "No verdict recorded yet.",
            ),
        )
    )
    findings = doc.get("findings") or ()
    parts.append(
        _card(
            "Findings",
            _table(
                ("Gate", "Unit", "Member", "Value", "Detail"),
                (
                    (
                        esc(f["gate"]),
                        esc(f["unit"]),
                        esc(f["member"]),
                        esc(f["value_vs_anchor"]),
                        esc(f["detail"]),
                    )
                    for f in findings
                ),
                "FINDING verdicts: 0.",
            ),
        )
    )
    parts.append(
        _card(
            "Re-run queue",
            _table(
                ("Gate", "Unit", "Member", "Why"),
                (
                    (esc(e["gate"]), esc(e["unit"]), esc(e["member"]), esc("; ".join(e["reasons"])))
                    for e in doc.get("rerun_queue") or ()
                ),
                "Nothing queued.",
            ),
        )
    )
    last = regression.get("last") or {}
    regression_body = (
        f"<p>{esc(last.get('verdicts'))} verdict(s) on <code>{esc(last.get('tree'))}"
        f"</code>, native <code>{esc(last.get('native'))}</code>, {esc(last.get('timestamp'))}</p>"
        if last
        else '<p class="muted">No full regression recorded.</p>'
    )
    if binding:
        regression_body += f"<p>{badge('DONE' if binding.get('ok') else 'BLOCKED')} {esc(binding.get('detail'))}</p>"
    parts.append(_card("Final regression", regression_body))
    items = doc.get("items") or ()
    if items:
        parts.append(
            _card(
                "Items",
                _table(
                    ("Item", "Title", "Status", "Evidence"),
                    (
                        (
                            esc(i.get("id")),
                            esc(i.get("title")),
                            badge(i.get("status")),
                            esc(i.get("evidence_status", "narrative only")),
                        )
                        for i in items
                    ),
                    "",
                ),
            )
        )
    if doc.get("features"):
        parts.append(
            _card(
                "Feature scorecard",
                _table(
                    ("Feature", "Meaning", "Mapping", "Status", "Evidence", "Notes"),
                    (
                        (
                            esc(r.get("feature")),
                            esc(r.get("meaning")),
                            esc(r.get("mapping")),
                            badge(r.get("status")),
                            _link(r.get("evidence")),
                            esc(r.get("notes")),
                        )
                        for r in doc["features"]
                    ),
                    "",
                ),
            )
        )
    if doc.get("streams"):
        parts.append(
            _card(
                "Parallel streams",
                _table(
                    ("Stream", "Detail", "Status", "Evidence"),
                    (
                        (
                            esc(r.get("stream")),
                            esc(r.get("detail")),
                            badge(r.get("status")),
                            _link(r.get("evidence")),
                        )
                        for r in doc["streams"]
                    ),
                    "",
                ),
            )
        )
    if doc.get("workflow_feedback"):
        parts.append(
            _card(
                "Workflow feedback",
                '<ul class="feedback-list">'
                + "".join(
                    f"<li><code>{esc(i.get('id'))}</code> <strong>{esc(i.get('title'))}</strong> "
                    f"{badge(i.get('status'))}"
                    + (f" &mdash; {_link(i.get('link'))}" if i.get("link") else "")
                    + "</li>"
                    for i in doc["workflow_feedback"]
                )
                + "</ul>",
            )
        )
    timeline = doc.get("timeline") or ()
    parts.append(
        _card(
            "Timeline",
            '<ul class="timeline-list">'
            + "".join(
                f'<li><span class="timeline-time">{esc(e.get("time"))}</span>'
                f'<span class="timeline-event">{esc(e.get("event"))}</span></li>'
                for e in timeline
            )
            + "</ul>"
            if timeline
            else '<p class="muted">No events recorded.</p>',
        )
    )
    final = meta.get("final_report")
    return _PAGE.format(
        title=esc(meta.get("title", "Onboarding progress")),
        status_badge=badge(meta.get("status")) if meta.get("status") else "",
        last_updated=esc(meta.get("last_updated", "")),
        final_report_link=f' &mdash; <a href="{esc(final)}">final report</a>' if final else "",
        body="\n".join(parts),
    )


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --surface-1: #fcfcfb; --page-plane: #f9f9f7; --text-primary: #0b0b0b;
  --text-secondary: #52514e; --text-muted: #898781; --gridline: #e1e0d9;
  --baseline: #c3c2b7; --border: rgba(11,11,11,0.10); --series-1: #2a78d6;
  --status-good: #0ca30c; --status-warning: #fab219; --status-serious: #ec835a;
  --status-critical: #d03b3b; --status-neutral: #898781;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --surface-1: #1a1a19; --page-plane: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --gridline: #2c2c2a; --baseline: #383835;
    --border: rgba(255,255,255,0.10); --series-1: #3987e5; --status-critical: #e66767;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface-1: #1a1a19; --page-plane: #0d0d0d; --text-primary: #ffffff;
  --text-secondary: #c3c2b7; --gridline: #2c2c2a; --baseline: #383835;
  --border: rgba(255,255,255,0.10); --series-1: #3987e5; --status-critical: #e66767;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 24px; background: var(--page-plane); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 14px; line-height: 1.5; }}
.wrap {{ max-width: 1280px; margin: 0 auto; }}
header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;
  margin-bottom: 20px; flex-wrap: wrap; }}
h1 {{ font-size: 22px; margin: 0 0 4px 0; }}
.subtitle {{ color: var(--text-secondary); font-size: 13px; }}
.headline {{ font-size: 15px; margin: 0 0 12px 0; }}
.theme-toggle {{ border: 1px solid var(--border); background: var(--surface-1);
  color: var(--text-primary); }}
  border-radius: 6px; padding: 6px 12px; font-size: 13px; cursor: pointer; }}
.card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 20px; margin-bottom: 20px; }}
.card h2 {{ font-size: 15px; margin: 0 0 12px 0; padding-bottom: 8px;
  border-bottom: 1px solid var(--gridline); }}
.meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 4px 24px; }}
.meta-row {{ display: flex; gap: 8px; padding: 3px 0; border-bottom: 1px solid var(--gridline); }}
.meta-key {{ color: var(--text-secondary); min-width: 110px; flex-shrink: 0; }}
.meta-val {{ color: var(--text-primary); word-break: break-word; }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.data-table th {{ text-align: left; color: var(--text-secondary); font-weight: 600;
  padding: 6px 10px; }}
  border-bottom: 1px solid var(--baseline); white-space: nowrap; }}
.data-table td {{ padding: 7px 10px; border-bottom: 1px solid var(--gridline); vertical-align: top; }}
.data-table tbody tr:hover {{ background: var(--page-plane); }}
.muted {{ color: var(--text-muted); }}
a {{ color: var(--series-1); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{ background: var(--page-plane); border: 1px solid var(--border); border-radius: 4px;
  padding: 1px 5px; font-size: 12px; }}
.badge {{ display: inline-flex; align-items: center; gap: 5px; border-radius: 999px;
  padding: 2px 9px 2px 7px; }}
  font-size: 12px; font-weight: 600; border: 1px solid currentColor; white-space: nowrap; }}
.badge-icon {{ font-size: 11px; line-height: 1; }}
.status-good {{ color: var(--status-good);
  background: color-mix(in srgb, var(--status-good) 14%, transparent); }}
.status-warning {{ color: var(--status-warning);
  background: color-mix(in srgb, var(--status-warning) 18%, transparent); }}
.status-serious {{ color: var(--status-serious);
  background: color-mix(in srgb, var(--status-serious) 16%, transparent); }}
.status-critical {{ color: var(--status-critical);
  background: color-mix(in srgb, var(--status-critical) 14%, transparent); }}
.status-info {{ color: var(--series-1);
  background: color-mix(in srgb, var(--series-1) 14%, transparent); }}
.status-neutral {{ color: var(--text-secondary);
  background: color-mix(in srgb, var(--text-secondary) 12%, transparent); }}
.tile-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.tile {{ flex: 1 1 120px; border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px;
  background: var(--page-plane); }}
.tile-label {{ font-size: 12px; color: var(--text-secondary); text-transform: uppercase;
  letter-spacing: .04em; }}
.tile-value {{ font-size: 20px; font-weight: 700; margin-top: 4px;
  font-variant-numeric: tabular-nums; }}
.feedback-list, .timeline-list {{ list-style: none; margin: 0; padding: 0; }}
.feedback-list li {{ padding: 6px 0; border-bottom: 1px solid var(--gridline); }}
.timeline-list li {{ display: flex; gap: 14px; padding: 5px 0;
  border-bottom: 1px solid var(--gridline); font-size: 13px; }}
.timeline-time {{ color: var(--text-secondary); min-width: 190px; flex-shrink: 0;
  font-variant-numeric: tabular-nums; }}
.table-scroll {{ overflow-x: auto; }}
footer {{ color: var(--text-muted); font-size: 12px; margin-top: 24px; text-align: center; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>{title} {status_badge}</h1>
    <div class="subtitle">Last updated: {last_updated}{final_report_link}</div>
  </div>
  <button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle light/dark theme">Toggle theme</button>
</header>
{body}
<footer>Generated from ledger.jsonl by staircase_onboard.report — offline, no external resources.</footer>
</div>
<script>
function applyStoredTheme() {{
  var t = null;
  try {{ t = localStorage.getItem('progress-theme'); }} catch (e) {{}}
  if (t === 'light' || t === 'dark') {{ document.documentElement.setAttribute('data-theme', t); }}
}}
function toggleTheme() {{
  var cur = document.documentElement.getAttribute('data-theme');
  var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  var next = (cur || (prefersDark ? 'dark' : 'light')) === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try {{ localStorage.setItem('progress-theme', next); }} catch (e) {{}}
}}
applyStoredTheme();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------- writing

_write_lock = threading.Lock()


def read_overlay(workspace: str | Path) -> dict[str, Any]:
    """The operator's narrative for the progress page, or ``{}``."""
    path = Path(workspace) / OVERLAY_FILENAME
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as exc:
        return {"notes": [f"{OVERLAY_FILENAME} does not parse: {exc}"]}
    return dict(data) if isinstance(data, Mapping) else {}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_reports(
    workspace: str | Path,
    *,
    facts: Facts | None = None,
    final_report: bool = False,
) -> dict[str, Any]:
    """Regenerate ``progress.json``/``progress.html``, and the final report's tables.

    The final report is written when ``final_report`` is set or the file
    already exists; its narrative is kept. The consistency check's result is
    embedded in ``progress.json`` so the page shows its own contradictions.
    """
    workspace = Path(workspace)
    facts = facts or collect_facts(workspace)
    overlay = read_overlay(workspace)
    with _write_lock:
        report = workspace / FINAL_REPORT
        if final_report or report.is_file():
            existing = report.read_text(encoding="utf-8") if report.is_file() else None
            _atomic_write(report, render_final_report(facts, existing))
        doc = build_progress(facts, overlay, workspace=workspace)
        problems = check_overlay(overlay, facts)
        if report.is_file():
            problems.extend(
                check_claims(
                    narrative(report.read_text(encoding="utf-8")), facts, where=FINAL_REPORT
                )
            )
        doc["consistency"] = {"ok": not problems, "problems": [p.render() for p in problems]}
        _atomic_write(
            workspace / PROGRESS_JSON, json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
        )
        _atomic_write(workspace / PROGRESS_HTML, render_html(doc))
    return doc


# ----------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate a workspace's reports, or check them (and other documents) for contradictions."""
    parser = argparse.ArgumentParser(prog="staircase-report", description=main.__doc__)
    parser.add_argument("workspace")
    parser.add_argument("--repo", help="the repository, if not the workspace task.yaml's")
    parser.add_argument("--final-report", action="store_true", help="also write final-report.md")
    parser.add_argument(
        "--check",
        nargs="*",
        metavar="DOC",
        help="do not write; check the workspace's reports and these documents, exit 1 on any "
        "contradiction",
    )
    args = parser.parse_args(argv)
    facts = collect_facts(args.workspace, repo_root=args.repo, check_history=True)
    if args.check is not None:
        problems = check_consistency(args.workspace, facts=facts, documents=args.check)
        for problem in problems:
            sys.stdout.write(f"CONTRADICTION {problem.render()}\n")
        sys.stdout.write(f"{len(problems)} contradiction(s)\n")
        return 1 if problems else 0
    doc = write_reports(args.workspace, facts=facts, final_report=args.final_report)
    sys.stdout.write(f"{doc['summary']['headline']}\n")
    return 0


__all__ = [
    "CLOSED_STATUSES",
    "Contradiction",
    "FINAL_REPORT",
    "Facts",
    "OPEN_STATUSES",
    "OVERLAY_FILENAME",
    "PROGRESS_HTML",
    "PROGRESS_JSON",
    "build_progress",
    "check_claims",
    "check_consistency",
    "check_items",
    "check_overlay",
    "check_progress",
    "collect_facts",
    "gate_table_markdown",
    "headline",
    "narrative",
    "read_overlay",
    "render_final_report",
    "render_html",
    "write_reports",
]


if __name__ == "__main__":
    sys.exit(main())
