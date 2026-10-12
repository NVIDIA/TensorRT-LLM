# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed decisions — the only channel an agent's verdict reaches the orchestrator through.

Agents never write ``progress.yaml`` directly and the orchestrator never
parses prose. Each role gets an ``append_<role>_progress`` MCP tool whose
``decision`` is a closed enum, a ``Stop`` hook refuses to let the turn end
until it was called, and the orchestrator branches on the field. A verdict
that cannot be expressed in the enum is a verdict the workflow does not
know how to act on, which is exactly the thing worth finding out loudly.

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
    deliver_stage:
      - {iteration: 1, agent: auditor, ..., decision: CLEAN}

Concurrency matters here in a way it does not in the other workflows: the
smith batch runs several agents at once, all appending. Every write takes a
lock and re-reads before appending, so a batch of eight cannot lose seven
entries to a read-modify-write race.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from claude_agent_sdk import tool
from rich.syntax import Syntax

from agent_flow.console import print_layer_panel
from agent_flow.logger import get_logger

from .remediation import CERTIFICATION_GAP_KEY, normalize_certification_gap_records

PLAN_STAGE = "plan_stage"
BUILD_STAGE = "build_stage"
DELIVER_STAGE = "deliver_stage"
_STAGES = (PLAN_STAGE, BUILD_STAGE, DELIVER_STAGE)

_STAGE_BY_AGENT = {
    "plan_drafter": PLAN_STAGE,
    "plan_reviewer": PLAN_STAGE,
    "smith": BUILD_STAGE,
    "assembler": BUILD_STAGE,
    "reviewer": BUILD_STAGE,
    "auditor": DELIVER_STAGE,
}

# Decision vocabularies. Each is closed on purpose — see the module
# docstring. ``BLOCKED`` variants carry a reason so the orchestrator's
# triage does not have to read the summary.
PLAN_DRAFTER_DECISIONS = ("DRAFT_READY", "CEILING")
PLAN_REVIEWER_DECISIONS = ("APPROVE", "REJECT")
SMITH_DECISIONS = ("LANDED", "FAILED_RECEIPT", "MACRO", "BLOCKED")
ASSEMBLER_DECISIONS = ("UNIT_READY", "BLOCKED")
REVIEWER_DECISIONS = ("APPROVE", "REJECT")
AUDITOR_DECISIONS = ("CLEAN", "UNCOVERED")

# Staircase's triage table, as an enum. Each maps to a different
# orchestrator action, which is the whole reason it is not free text.
BLOCKED_REASONS = (
    "vocabulary_gap",
    "knowledge_gap",
    "accuracy_shortfall",
    "environment",
    "certification_gap",
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


def _log_read(caller: str, description: str, text: str) -> None:
    body = Syntax(text, "yaml", theme="ansi_dark", word_wrap=True)
    print_layer_panel(caller, f"system · read ({description})", body, get_logger().console)


@dataclass
class ProgressContext:
    """Shared mutable context the tool handlers close over.

    ``current_iteration`` is set before each dispatch so an entry is stamped
    without the agent having to pass (or guess) it. ``lock`` serializes the
    read-modify-write, which a concurrent smith batch would otherwise race.
    """

    path: Path
    current_iteration: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def append(self, agent: str, entry: dict[str, Any]) -> None:
        with self.lock:
            data = read_progress(self.path)
            data[_STAGE_BY_AGENT[agent]].append(entry)
            write_progress(self.path, data)
        _log_write(agent, entry)


def _decision_tool(
    ctx: ProgressContext,
    agent: str,
    *,
    decisions: tuple[str, ...],
    decision_help: str,
    summary_help: str,
    extra_properties: dict[str, Any] | None = None,
    extra_keys: tuple[str, ...] = (),
):
    """Build one role's ``append_<agent>_progress`` tool."""
    properties: dict[str, Any] = {
        "summary": {"type": "string", "description": summary_help},
        "decision": {"type": "string", "enum": list(decisions), "description": decision_help},
    }
    properties.update(extra_properties or {})

    @tool(
        f"append_{agent}_progress",
        (
            f"Record this turn's {agent} entry in progress.yaml. Call it exactly "
            f"once, as the last action of your turn. The orchestrator branches on "
            f"`decision`, so it is the only part of your turn it reads."
        ),
        {"type": "object", "properties": properties, "required": ["summary", "decision"]},
    )
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "iteration": ctx.current_iteration,
            "agent": agent,
            "timestamp": _now_iso(),
            "summary": args["summary"],
            "decision": args["decision"],
        }
        for key in extra_keys:
            if args.get(key) is not None:
                entry[key] = args[key]
        ctx.append(agent, entry)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Recorded {agent} entry for iteration {ctx.current_iteration} "
                        f"(decision={entry['decision']})."
                    ),
                }
            ]
        }

    return handler


def _read_tool(ctx: ProgressContext, caller: str):
    """A stage-scoped read tool, so a role sees its own phase and no more."""
    stage = _STAGE_BY_AGENT[caller]

    @tool(
        "read_latest_progress",
        (
            f"Return the most recent entries from progress.yaml's `{stage}` as YAML. "
            f"Cheaper and better scoped than reading the whole file."
        ),
        {
            "type": "object",
            "properties": {
                "iterations": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "How many of the most recent iterations to include (default 1).",
                }
            },
            "required": [],
        },
    )
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        iterations = int(args.get("iterations") or 1)
        selected = find_entries(ctx.path, stage=stage, last_iterations=iterations)
        text = _yaml_dump(selected) if selected else f"# No {stage} entries yet.\n"
        _log_read(caller, f"{stage}, last {iterations}", text)
        return {"content": [{"type": "text", "text": text}]}

    return handler


DECISIONS_DIRNAME = "decisions"

# The enums, keyed by role, so the file protocol can validate exactly what
# the tool schema would have.
DECISIONS_BY_AGENT: dict[str, tuple[str, ...]] = {
    "plan_drafter": PLAN_DRAFTER_DECISIONS,
    "plan_reviewer": PLAN_REVIEWER_DECISIONS,
    "smith": SMITH_DECISIONS,
    "assembler": ASSEMBLER_DECISIONS,
    "reviewer": REVIEWER_DECISIONS,
    "auditor": AUDITOR_DECISIONS,
}

EXTRA_KEYS_BY_AGENT: dict[str, tuple[str, ...]] = {
    # `constituents` is what makes MACRO actionable. Without it the
    # decomposition arrives as prose inside `summary`, which a human can act
    # on and the orchestrator cannot -- so the one outcome whose entire
    # purpose is "here is the work that comes next" ended the run instead of
    # extending it. Measured on the first real onboarding, on its largest
    # item.
    "smith": ("entry", "receipt", "constituents"),
    "assembler": ("blocked_reason", "missing_entry", CERTIFICATION_GAP_KEY),
    # `certification_gap`: "this entry is correct and must be certified at
    # these values". The orchestrator fans each row out to a verify smith
    # itself (remediation.py), so a certification-only finding no longer
    # re-opens the assembler or waits on a human to relay it.
    "reviewer": (CERTIFICATION_GAP_KEY,),
    # `uncovered` is to the auditor what `constituents` is to a MACRO: the
    # difference between a verdict the orchestrator can act on and one a
    # human has to read. Prose cannot be deduplicated, grouped by entry
    # surface, or compared across rounds -- so an UNCOVERED whose findings
    # live only in `summary` ends the run, which is exactly what it used to
    # do. See remediation.py.
    "auditor": ("uncovered", CERTIFICATION_GAP_KEY),
}

#: Keys an uncovered row may carry. `entry`, `arch` and `axis` are required:
#: without all three a finding cannot be grouped into a dispatch, and a
#: finding that cannot be dispatched is one a human has to re-derive.
UNCOVERED_REQUIRED: tuple[str, ...] = ("entry", "arch", "axis")
UNCOVERED_OPTIONAL: tuple[str, ...] = ("values", "value", "contract_line")

#: Keys a MACRO decision's `constituents` entries may carry. `target` and
#: `tier` are required: the orchestrator needs a name to dispatch and a rung
#: to budget turns against.
CONSTITUENT_REQUIRED: tuple[str, ...] = ("target", "tier")
CONSTITUENT_OPTIONAL: tuple[str, ...] = ("surface", "note", "unit", "depends_on")


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


def parse_uncovered_records(raw: Any) -> list[dict[str, Any]]:
    """Normalize an auditor's uncovered list, dropping anything undispatchable.

    Same bargain as :func:`parse_constituents`, for the same reason: an
    auditor writing YAML by hand should not lose a whole finding to a
    formatting slip, but a row without an entry, an arch or an axis cannot
    be grouped into a verify dispatch. Those are dropped here rather than
    guessed at -- a guessed axis merges two unrelated surfaces into one
    dispatch and certifies neither.

    Kept as plain dicts because this is the persistence boundary;
    :mod:`remediation` turns them into typed rows.
    """
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        record = {key: str(row.get(key, "")).strip() for key in UNCOVERED_REQUIRED}
        if not all(record.values()):
            continue
        values = row.get("values")
        if values is None:
            values = row.get("value")
        if isinstance(values, (str, int, float)):
            values = [values]
        if isinstance(values, list):
            record["values"] = [str(v).strip() for v in values if str(v).strip()]
        line = str(row.get("contract_line", "")).strip()
        if line:
            record["contract_line"] = line
        rows.append(record)
    return rows


def decision_path(workspace: Path, agent: str, iteration: int) -> Path:
    """Where a role writes its decision when MCP tools are unavailable."""
    return Path(workspace) / DECISIONS_DIRNAME / f"{agent}-{iteration}.yaml"


def _decision_marker_path(workspace: Path, agent: str, iteration: int) -> Path:
    """The dotfile stamped beside a decision when its turn is dispatched.

    A dotfile, and beside the decision rather than under a directory of its
    own, because the two must share a filesystem — see
    ``mark_decision_dispatch`` for why that is the whole mechanism.
    """
    return Path(workspace) / DECISIONS_DIRNAME / f".{agent}-{iteration}.dispatched"


def mark_decision_dispatch(workspace: Path, agent: str, iteration: int) -> None:
    """Stamp the moment this turn was dispatched, so a stale answer is detectable.

    ``(agent, iteration)`` is **not unique across a run**: ``iteration`` is 1
    for every turn of the deliver stage, and it restarts per unit in assembly.
    So two different turns routinely write the same path — see
    ``decision_path`` — and a turn that ends *without* writing one leaves the
    **previous** turn's verdict sitting there looking fresh, which
    ``collect_decision_file`` then reports as this turn's answer.

    That is not hypothetical. It happened five times in one onboarding: ten
    smith decisions collected from the wrong place, an assembler told to write
    the previous turn's path, decision paths colliding across units, and —
    worst — three audit verdicts read back as a verdict from two passes
    earlier, which made the audit loop structurally unable to converge,
    because a ``CLEAN`` could not end the run. Every one of those was silent.

    This run saw its cousin: a smith committed 380 lines with a passing
    receipt and then died without writing its decision file, blocking a batch
    for 58 minutes. Staleness detection is what turns that class of failure
    from silent into visible.

    The marker is compared by **mtime against the decision file on the same
    filesystem**, never against a clock. An agent may answer from a different
    node whose clock is skewed by minutes; both files live under the
    workspace, so their mtimes are ordered by the one filesystem that wrote
    them.
    """
    marker = _decision_marker_path(workspace, agent, iteration)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")


def decision_is_stale(workspace: Path, agent: str, iteration: int) -> bool:
    """Whether the decision file at this path predates the turn that wants it.

    False when either file is absent: a missing decision is the caller's
    "no decision" case, and a missing marker means this run never stamped one
    — in which case the old unconditional behaviour is the honest fallback,
    rather than silently refusing every decision a run ever writes.
    """
    decision = decision_path(workspace, agent, iteration)
    marker = _decision_marker_path(workspace, agent, iteration)
    if not decision.is_file() or not marker.is_file():
        return False
    return decision.stat().st_mtime < marker.stat().st_mtime


def collect_decision_file(
    ctx: ProgressContext,
    workspace: Path,
    agent: str,
    iteration: int,
) -> str | None:
    """Read, validate and record the decision file ``agent`` wrote.

    **Why this exists.** The typed-decision MCP tools are the workflow's
    preferred mechanism — a closed enum in a tool schema is checked before
    the model can emit anything else. But an *enterprise* MCP configuration
    (``/etc/claude-code/managed-mcp.json``) makes the SDK refuse every
    dynamically-registered server, including the in-process one those tools
    live in. On such a host the tools cannot exist at all.

    So the decision becomes a file, and the guarantee moves rather than
    disappearing: the enum is validated here, on read, and a decision
    outside it is rejected exactly as the schema would have rejected it.
    The orchestrator still branches on one closed value it did not get from
    prose.

    Returns the validated decision, or ``None`` if the role wrote nothing
    usable — which the caller must treat as a failed turn rather than as
    silence to interpret.
    """
    allowed = DECISIONS_BY_AGENT[agent]
    path = decision_path(workspace, agent, iteration)
    if not path.is_file():
        return None
    # A file older than this turn's dispatch marker belongs to a PREVIOUS turn
    # that happened to share this path. Reading it would report that turn's
    # verdict as this one's — the failure `mark_decision_dispatch` exists to
    # make visible, and one that is silent every other way.
    if decision_is_stale(workspace, agent, iteration):
        return None

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(raw, dict):
        return None

    decision = str(raw.get("decision", "")).strip()
    if decision not in allowed:
        return None

    entry: dict[str, Any] = {
        "iteration": iteration,
        "agent": agent,
        "timestamp": _now_iso(),
        "summary": str(raw.get("summary", "")).strip(),
        "decision": decision,
    }
    for key in EXTRA_KEYS_BY_AGENT.get(agent, ()):
        value = raw.get(key)
        if value is None:
            continue
        if key == "constituents":
            children = parse_constituents(value)
            if children:
                entry[key] = children
        elif key == "uncovered":
            rows = parse_uncovered_records(value)
            if rows:
                entry[key] = rows
        elif key == CERTIFICATION_GAP_KEY:
            gaps = normalize_certification_gap_records(value)
            if gaps:
                entry[key] = gaps
        elif str(value).strip():
            entry[key] = str(value).strip()
    ctx.append(agent, entry)
    return decision


def _certification_gap_property(who: str) -> dict[str, Any]:
    """The ``certification_gap`` tool property, worded for one role."""
    return {
        "type": "array",
        "description": (
            f"{who} One row per catalog entry that is CORRECT but not certified at "
            "values the target reaches inside the serving envelope. The orchestrator "
            "dispatches a verify smith per row itself -- no human relays it -- so a "
            "certification-only finding never needs a REJECT or a BLOCKED."
        ),
        "items": {
            "type": "object",
            "properties": {
                "entry": {"type": "string", "description": "Catalog entry, e.g. `moe/runner`."},
                "values": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The values to certify (row counts, sizes, ...).",
                },
                "axis": {
                    "type": "string",
                    "description": "The certification axis; default `envelope`.",
                },
                "arch": {"type": "string", "description": "Default: this run's architecture."},
                "reason": {
                    "type": "string",
                    "description": "Why the target needs it: the call site, the contract line.",
                },
            },
            "required": ["entry", "values"],
        },
    }


def build_progress_tools(ctx: ProgressContext) -> dict[str, list[Any]]:
    """Build the per-role tool lists for ``BackendConfig(tools=...)``."""
    plan_drafter = _decision_tool(
        ctx,
        "plan_drafter",
        decisions=PLAN_DRAFTER_DECISIONS,
        summary_help=(
            "What you wrote into plan.md and vocabulary.yaml, and the step counts "
            "by state (OK / CERT / GAP)."
        ),
        decision_help=(
            "DRAFT_READY: both files are written and coherent; the PlanReviewer "
            "reads them next. CEILING: a computation step has no server anywhere — "
            "no catalog entry, no op, no binding — so the run stops here. A ceiling "
            "ends the run, so use it only with the evidence in vocabulary.yaml."
        ),
    )
    plan_reviewer = _decision_tool(
        ctx,
        "plan_reviewer",
        decisions=PLAN_REVIEWER_DECISIONS,
        summary_help=(
            "Your rationale. On REJECT, the specific actionable items, naming the "
            "file and row for each. On APPROVE, anything you noticed anyway — it "
            "reaches the assembler as context."
        ),
        decision_help=(
            "APPROVE if the plan is executable and vocabulary.yaml is sound. REJECT "
            "only for something that would waste an assembly turn or produce a wrong "
            "gate; the build is not blocked on you, so a rejection that buys nothing "
            "costs a whole round."
        ),
    )
    smith = _decision_tool(
        ctx,
        "smith",
        decisions=SMITH_DECISIONS,
        summary_help=(
            "The entry path, the wrapped op's qualified name, the receipt exactly as "
            "recorded, what the test covered, and what remains uncertified."
        ),
        decision_help=(
            "LANDED: the entry is written, tested and committed with a passing "
            "receipt. FAILED_RECEIPT: the wrapper and test are right and the op "
            "demonstrably fails at these values — you landed that receipt, which is "
            "knowledge, not a failure. MACRO: the callable is Python-level "
            "composition and cannot be one entry; you wrote the decomposition "
            "report. BLOCKED: you could not proceed; say why."
        ),
        extra_properties={
            "entry": {
                "type": "string",
                "description": "Path of the catalog entry this run produced or verified.",
            },
            "receipt": {
                "type": "string",
                "description": "The receipt exactly as recorded, including the arch key.",
            },
        },
        extra_keys=("entry", "receipt"),
    )
    assembler = _decision_tool(
        ctx,
        "assembler",
        decisions=ASSEMBLER_DECISIONS,
        summary_help=(
            "The derivation table, the vocabulary audit, the dump review, the gate "
            "outcomes verbatim, and the symptoms — stating separately anything you "
            "had to establish by observation and which mechanism it belongs to."
        ),
        decision_help=(
            "UNIT_READY: this unit is written and you believe it passes its gates. "
            "BLOCKED: you cannot proceed — set `blocked_reason`, because the "
            "orchestrator's next action depends on which kind it is."
        ),
        extra_properties={
            "blocked_reason": {
                "type": "string",
                "enum": list(BLOCKED_REASONS),
                "description": (
                    "BLOCKED only. `vocabulary_gap`: a callable you need has no entry "
                    "— the orchestrator dispatches smith and resumes you. "
                    "`knowledge_gap`: a question your sources cannot answer — state "
                    "it exactly; the run ends and records it. `accuracy_shortfall`: a "
                    "shortfall that survived your self-heal — the run ends with the "
                    "audit trail, because the gate is the release criterion and it is "
                    "not yours to relax. `environment`: the launch prefix or the host "
                    "cannot do what the task needs. `certification_gap`: you cannot "
                    "go on until entries are certified at values the target needs — "
                    "list them in `certification_gap`; the orchestrator verifies them "
                    "and resumes you."
                ),
            },
            "missing_entry": {
                "type": "string",
                "description": (
                    "vocabulary_gap only: the callable with no entry, and the candidate "
                    "op if you found one. This becomes smith's dispatch brief."
                ),
            },
            CERTIFICATION_GAP_KEY: _certification_gap_property(
                "Optional with UNIT_READY (the verifies run while the unit is gated), "
                "required with BLOCKED/certification_gap."
            ),
        },
        extra_keys=("blocked_reason", "missing_entry", CERTIFICATION_GAP_KEY),
    )
    reviewer = _decision_tool(
        ctx,
        "reviewer",
        decisions=REVIEWER_DECISIONS,
        summary_help=(
            "What you checked and what you found. On REJECT, cite the file and line "
            "for each item and name the invariant it violates."
        ),
        decision_help=(
            "APPROVE if the diff is consistent with the gate evidence and the "
            "invariants hold. REJECT for a defect the gates cannot see and a reader "
            "can — an unmapped call, a stale op list, a sync in the forward, a "
            "topology that does not match the segment. A value outside an entry's "
            "certified column on a CORRECT target is not a REJECT: APPROVE and list "
            "it in `certification_gap`."
        ),
        extra_properties={
            CERTIFICATION_GAP_KEY: _certification_gap_property(
                "Certification-only findings, with APPROVE or REJECT."
            )
        },
        extra_keys=(CERTIFICATION_GAP_KEY,),
    )
    auditor = _decision_tool(
        ctx,
        "auditor",
        decisions=AUDITOR_DECISIONS,
        summary_help=(
            "How many distinct runtime values you checked against how many entries. "
            "On UNCOVERED, every value that is not covered, naming the entry, the "
            "value and the contract line that excludes it — that list is smith's "
            "dispatch brief."
        ),
        decision_help=(
            "CLEAN: every dumped value sits inside its entry's certified column on "
            "this architecture. UNCOVERED: at least one does not. Do not soften an "
            "UNCOVERED because the gates passed — gates are blind to this by "
            "construction, which is the entire reason this turn exists."
        ),
        extra_properties={
            "uncovered": {
                "type": "array",
                "description": (
                    "REQUIRED on UNCOVERED. One row per (entry, architecture, "
                    "certification axis) — not one row per value. Seven uncovered "
                    "batch sizes on one entry are ONE row whose `values` lists all "
                    "seven, because they are one piece of work: one parameterized "
                    "test, one contract line, one receipt. The orchestrator groups, "
                    "deduplicates and dispatches from this list, so a finding that "
                    "appears only in `summary` is a finding nothing acts on."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "entry": {
                            "type": "string",
                            "description": "Catalog entry path, e.g. `attention/thop_attention`.",
                        },
                        "arch": {
                            "type": "string",
                            "description": "The GPU architecture audited, e.g. `sm_103`.",
                        },
                        "axis": {
                            "type": "string",
                            "description": (
                                "The certification dimension the value varies along — "
                                "`batch_size`, `num_pages`, `local_layer_idx`. This is "
                                "what makes two findings on one entry the same work or "
                                "different work, so be precise."
                            ),
                        },
                        "values": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Every uncovered value on this axis.",
                        },
                        "contract_line": {
                            "type": "string",
                            "description": (
                                "The contract sentence that excludes them, verbatim. It "
                                "is the difference between a dispatch that certifies the "
                                "surface and one that re-derives what was wrong."
                            ),
                        },
                    },
                    "required": ["entry", "arch", "axis", "values"],
                },
            },
            CERTIFICATION_GAP_KEY: _certification_gap_property(
                "Optional, and dispatched exactly like `uncovered`."
            ),
        },
        extra_keys=("uncovered", CERTIFICATION_GAP_KEY),
    )

    return {
        "plan_drafter": [plan_drafter, _read_tool(ctx, "plan_drafter")],
        "plan_reviewer": [plan_reviewer, _read_tool(ctx, "plan_reviewer")],
        "smith": [smith],
        "assembler": [assembler, _read_tool(ctx, "assembler")],
        "reviewer": [reviewer, _read_tool(ctx, "reviewer")],
        # The auditor gets no read tool on purpose: its verdict must be
        # grounded in the dump and the contracts, not in what the agents
        # whose claims it is checking said about themselves.
        "auditor": [auditor],
    }


__all__ = [
    "ASSEMBLER_DECISIONS",
    "AUDITOR_DECISIONS",
    "BLOCKED_REASONS",
    "BUILD_STAGE",
    "DELIVER_STAGE",
    "PLAN_DRAFTER_DECISIONS",
    "PLAN_REVIEWER_DECISIONS",
    "PLAN_STAGE",
    "ProgressContext",
    "REVIEWER_DECISIONS",
    "SMITH_DECISIONS",
    "UNCOVERED_OPTIONAL",
    "UNCOVERED_REQUIRED",
    "build_progress_tools",
    "decision_is_stale",
    "find_entries",
    "init_progress_file",
    "latest_decision",
    "latest_entry",
    "mark_decision_dispatch",
    "read_progress",
    "write_progress",
]
