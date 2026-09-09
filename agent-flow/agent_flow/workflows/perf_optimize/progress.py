"""Shared progress log for the perf-optimize workflow.

Progress is persisted as a YAML mapping at ``workspace/progress.yaml``
with a single top-level list — the workflow has one logical phase
(optimization), with the loop position carried per entry:

    optimization:
      - step: 1
        agent: benchmarker
        round: 0
        timestamp: "2026-07-01T..."
        summary: |
          ...
      - step: 4
        agent: evaluator
        round: 1
        attempt: 1
        item_id: opt-001
        timestamp: "..."
        summary: ...
        decision: APPROVE           # evaluator: APPROVE | REJECT | PUSH_BACK
        reason_category: none       # evaluator: none | code_quality |
                                    #   functionality | perf_shortfall
        measured_gain_pct: 8.4      # evaluator: vs the last ACCEPTED value
        measured_value: 1298.7      # evaluator: absolute target metric
      - step: 9
        agent: qa
        round: 2
        timestamp: "..."
        summary: ...
        cumulative_improvement_pct: 8.4   # qa: final verification vs baseline

Agents never write this file directly. Each agent is given an
``append_*_progress`` MCP tool that records a structured entry, plus a
shared ``read_latest_progress`` tool for fetching an upstream agent's
summary (e.g. the optimizer reading the evaluator's PUSH_BACK feedback
before a retry). The orchestrator branches the optimizer ⇄ evaluator
loop on the evaluator's structured ``decision`` field — no regex over
prose; qa carries no decision (it verifies, the orchestrator owns the
loop). The substantive data still flows through the workspace artifact
files (``roadmap.yaml``, the per-attempt ``.md`` reports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from claude_agent_sdk import tool
from rich.syntax import Syntax

from agent_flow.console import print_layer_panel
from agent_flow.logger import get_logger

OPTIMIZATION_STAGE = "optimization"
_AGENTS = (
    "benchmarker",
    "projector",
    "analyzer",
    "optimizer",
    "evaluator",
    "integrator",
    "qa",
    "reporter",
)
_READABLE_AGENTS = (*_AGENTS, "optimizer_evaluator")

# Structured decision vocabulary the orchestrator branches on. APPROVE
# accepts the attempt; REJECT fails the item terminally (no retry would
# help); PUSH_BACK reverts and retries the optimizer with feedback,
# bounded by ``optimize.max_attempts_per_item``.
EVALUATOR_DECISIONS = ("APPROVE", "REJECT", "PUSH_BACK")
EVALUATOR_REASON_CATEGORIES = ("none", "code_quality", "functionality", "perf_shortfall")
INTEGRATOR_DECISIONS = ("APPROVE", "FALLBACK_BEST", "REJECT")

# Per-point curve measurements (Pareto-curve mode only: when
# ``benchmark.concurrency`` in task.yaml is a list). Shared by the
# evaluator and QA tools; optional in both so scalar runs are untouched.
_CURVE_FIELD_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "concurrency": {"type": "integer", "minimum": 1},
            "value": {"type": "number"},
            "tok_s_user": {"type": "number"},
            "tok_s_gpu": {"type": "number"},
        },
        "required": ["concurrency", "value", "tok_s_user", "tok_s_gpu"],
    },
    "description": "Curve mode only (benchmark.concurrency is a list): one "
    "entry per concurrency point, ascending — the absolute target-metric "
    "value plus tok_s_user (=1000/mean_tpot_ms) and tok_s_gpu "
    "(=output_throughput/num_gpus) measured at that point. Omit in scalar "
    "mode.",
}


def _coerce_curve(curve: Any) -> list[dict[str, Any]]:
    """Coerce a tool-supplied curve into plain int/float entries."""
    coerced: list[dict[str, Any]] = []
    for point in curve:
        coerced.append(
            {
                "concurrency": int(point["concurrency"]),
                "value": float(point["value"]),
                "tok_s_user": float(point["tok_s_user"]),
                "tok_s_gpu": float(point["tok_s_gpu"]),
            }
        )
    return coerced


def _empty_progress() -> dict[str, list[dict[str, Any]]]:
    return {OPTIMIZATION_STAGE: []}


def read_progress(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the YAML mapping at ``path``, returning an empty mapping if it is missing or empty.

    Always returns a dict with exactly the ``optimization`` key, whose
    value is a (possibly empty) list. Raises ``ValueError`` if the file's
    top-level node isn't a mapping or the ``optimization`` value isn't a
    list — that catches malformed / legacy formats early instead of
    letting them silently mis-route entries.
    """
    if not path.is_file():
        return _empty_progress()
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return _empty_progress()
    data = yaml.safe_load(text)
    if data is None:
        return _empty_progress()
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping with an "
            f"`{OPTIMIZATION_STAGE}` key, got {type(data).__name__}"
        )
    v = data.get(OPTIMIZATION_STAGE, [])
    if v is None:
        v = []
    if not isinstance(v, list):
        raise ValueError(f"{path}: `{OPTIMIZATION_STAGE}` must be a list, got {type(v).__name__}")
    return {OPTIMIZATION_STAGE: v}


def write_progress(path: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    """Persist ``data`` with the canonical key so diffs stay stable."""
    ordered = {OPTIMIZATION_STAGE: data.get(OPTIMIZATION_STAGE, [])}
    path.write_text(
        yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def init_progress_file(path: Path) -> None:
    """Write an empty progress.yaml with the canonical key."""
    write_progress(path, _empty_progress())


def find_entries(
    path: Path,
    *,
    agent: str | None = None,
    last_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Return progress entries, optionally filtered by agent and step.

    - ``agent=None`` keeps all agents; otherwise filters by
      ``entry["agent"]``. An unknown agent name returns ``[]``.
    - ``last_steps=None`` keeps all steps; otherwise keeps entries whose
      ``step`` is in ``[max_step - last_steps + 1, max_step]`` where
      ``max_step`` is the highest step in the (already agent-filtered)
      result.
    - Returns ``[]`` when the file is empty or nothing matches.
    """
    if last_steps is not None and last_steps < 1:
        raise ValueError(f"last_steps must be >= 1, got {last_steps}")
    if agent is not None and agent not in _READABLE_AGENTS:
        return []
    entries = list(read_progress(path)[OPTIMIZATION_STAGE])
    if agent is not None:
        entries = [e for e in entries if e.get("agent") == agent]
    if last_steps is not None:
        step_values = [e["step"] for e in entries if "step" in e]
        if not step_values:
            return []
        cutoff = max(step_values) - last_steps + 1
        entries = [e for e in entries if e.get("step", -1) >= cutoff]
    return entries


def latest_entry(path: Path, agent: str) -> dict[str, Any] | None:
    """Return the most recent entry written by ``agent``, or ``None``."""
    if agent not in _READABLE_AGENTS:
        return None
    matches = find_entries(path, agent=agent)
    return matches[-1] if matches else None


def _append(
    path: Path,
    entry: dict[str, Any],
    *,
    lock: Lock | None = None,
    allocate_step: bool = False,
) -> dict[str, Any]:
    """Append one entry, optionally serializing and allocating its step."""

    def _write() -> dict[str, Any]:
        data = read_progress(path)
        stored = dict(entry)
        if allocate_step:
            stored["step"] = len(data[OPTIMIZATION_STAGE]) + 1
        data[OPTIMIZATION_STAGE].append(stored)
        write_progress(path, data)
        return stored

    if lock is None:
        return _write()
    with lock:
        return _write()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _yaml_dump(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _log_progress_write(agent: str, entry: dict[str, Any]) -> None:
    """Log a freshly written progress entry as a styled YAML panel."""
    body = Syntax(_yaml_dump([entry]), "yaml", theme="ansi_dark", word_wrap=True)
    suffix = f"system · wrote step {entry.get('step')}"
    print_layer_panel(agent, suffix, body, get_logger().console)


def _log_progress_read(caller: str, agent_filter: str | None, steps: int, text: str) -> None:
    """Log the content returned by ``read_latest_progress``.

    ``caller`` is the agent making the call and drives the panel style —
    so the log shows *who is reading*, not which agent's entries got
    filtered.
    """
    body = Syntax(text, "yaml", theme="ansi_dark", word_wrap=True)
    filter_part = f"agent={agent_filter}" if agent_filter else "all agents"
    suffix = f"system · read ({filter_part}, last {steps})"
    print_layer_panel(caller, suffix, body, get_logger().console)


@dataclass
class ProgressContext:
    """Shared mutable context captured by the per-agent tool handlers.

    The workflow updates ``current_step`` / ``current_round`` /
    ``current_attempt`` / ``current_item_id`` before each agent run so
    the tools stamp every entry with the right loop position without the
    agent having to pass (or guess) it.
    """

    path: Path
    # Item workers append to the global dashboard first under this shared
    # lock, then append the same structured event to their own progress file.
    global_path: Path | None = None
    global_lock: Lock | None = field(default=None, repr=False, compare=False)
    current_step: int = 0
    current_round: int = 0
    current_attempt: int | None = None
    current_item_id: str = ""
    _tool_cache: list[Any] | None = field(default=None, repr=False, compare=False)


def append_workflow_event(
    path: Path,
    lock: Lock,
    *,
    event: str,
    round_no: int,
    summary: str,
    item_ids: list[str],
) -> dict[str, Any]:
    """Append a main-thread batch lifecycle event to global progress."""
    entry = {
        "agent": "optimizer_evaluator",
        "round": round_no,
        "timestamp": _now_iso(),
        "event": event,
        "item_ids": list(item_ids),
        "summary": summary,
    }
    stored = _append(path, entry, lock=lock, allocate_step=True)
    _log_progress_write("optimizer_evaluator", stored)
    return stored


def build_progress_tools(ctx: ProgressContext) -> dict[str, list[Any]]:
    """Build the per-agent tool lists for ``BackendConfig(tools=...)``.

    Returns a dict keyed by agent name (one key per role in
    ``_AGENTS``). The tool objects are ``SdkMcpTool`` instances, which
    the claude-code backend wraps into an in-process MCP server.
    """

    def _base_entry(agent: str) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "step": ctx.current_step,
            "agent": agent,
            "round": ctx.current_round,
        }
        if ctx.current_attempt is not None:
            entry["attempt"] = ctx.current_attempt
        if ctx.current_item_id:
            entry["item_id"] = ctx.current_item_id
        entry["timestamp"] = _now_iso()
        return entry

    def _append_for_context(entry: dict[str, Any]) -> dict[str, Any]:
        if ctx.global_path is not None:
            global_entry = _append(
                ctx.global_path,
                entry,
                lock=ctx.global_lock,
                allocate_step=True,
            )
            _append(ctx.path, entry)
            return global_entry
        _append(ctx.path, entry)
        return entry

    def _ack(agent: str) -> dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Recorded {agent} entry for step {ctx.current_step}.",
                }
            ]
        }

    def _make_summary_tool(agent: str, summary_description: str):
        @tool(
            f"append_{agent}_progress",
            (
                f"Record a {agent.capitalize()} progress entry in progress.yaml. "
                f"Call this exactly once as the last action of your turn."
            ),
            {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": summary_description,
                    },
                },
                "required": ["summary"],
            },
        )
        async def append_summary_progress(args: dict[str, Any]) -> dict[str, Any]:
            entry = _base_entry(agent)
            entry["summary"] = args["summary"]
            stored = _append_for_context(entry)
            _log_progress_write(agent, stored)
            return _ack(agent)

        return append_summary_progress

    append_benchmarker_progress = _make_summary_tool(
        "benchmarker",
        "Short human-readable summary: the serve + benchmark commands you "
        "ran, the operating point (ISL/OSL/concurrency), and the headline "
        "baseline metrics. Name the result files you wrote.",
    )
    append_projector_progress = _make_summary_tool(
        "projector",
        "Short human-readable summary: the sources you used (skill, peaks "
        "calculator, config.json), the model/device "
        "mapping, the headline SOL ceiling and the baseline-vs-SOL gap (or "
        "the unavailability reason), and the files you wrote.",
    )
    append_analyzer_progress = _make_summary_tool(
        "analyzer",
        "Short human-readable summary: which profilers you ran, the trace "
        "files produced, and the roadmap items you added / reordered / "
        "marked obsolete this round with their expected gains.",
    )
    append_optimizer_progress = _make_summary_tool(
        "optimizer",
        "Short human-readable summary: the roadmap item you implemented, "
        "what you changed (config keys / source files), the smoke-check "
        "result, and any risks or blockers.",
    )
    append_reporter_progress = _make_summary_tool(
        "reporter",
        "Short human-readable summary: the cumulative improvement headline, "
        "the accepted/failed item counts, and confirmation that both "
        "optimization_report.md and optimization_report.html were written.",
    )

    @tool(
        "append_evaluator_progress",
        (
            "Record an Evaluator progress entry (with your verdict) in "
            "progress.yaml. Call this exactly once as the last action of "
            "your turn."
        ),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short human-readable summary: the diff you reviewed, "
                    "the functionality evidence, the measured metrics vs "
                    "the current best, and the reasoning behind your "
                    "verdict.",
                },
                "decision": {
                    "type": "string",
                    "enum": list(EVALUATOR_DECISIONS),
                    "description": "APPROVE if code quality, functionality, AND the "
                    "measured perf gain all pass the expectation gate. "
                    "Otherwise PUSH_BACK when a concrete, actionable fix "
                    "could plausibly pass a retry, or REJECT when the "
                    "item's premise is broken and no retry would help "
                    "(the item is failed and the loop moves on).",
                },
                "reason_category": {
                    "type": "string",
                    "enum": list(EVALUATOR_REASON_CATEGORIES),
                    "description": "The single dominant reason for a REJECT or "
                    'PUSH_BACK ("none" on APPROVE): code_quality | '
                    "functionality | perf_shortfall.",
                },
                "measured_gain_pct": {
                    "type": "number",
                    "description": "Signed % improvement of the target metric vs the "
                    "last ACCEPTED measurement (roadmap current_best), "
                    "normalized so positive = better. Curve mode: the mean "
                    "of the per-point gains — over "
                    "optimize.focus_concurrencies when task.yaml sets it. "
                    "0 if the benchmark could not run.",
                },
                "measured_value": {
                    "type": "number",
                    "description": "The absolute target-metric value you measured (e.g. "
                    "output tok/s). Curve mode: the mean of the per-point "
                    "values — over optimize.focus_concurrencies when "
                    "task.yaml sets it. 0 if the benchmark could not run.",
                },
                "curve": _CURVE_FIELD_SCHEMA,
            },
            "required": [
                "summary",
                "decision",
                "reason_category",
                "measured_gain_pct",
                "measured_value",
            ],
        },
    )
    async def append_evaluator_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry = _base_entry("evaluator")
        entry["summary"] = args["summary"]
        entry["decision"] = args["decision"]
        entry["reason_category"] = args["reason_category"]
        entry["measured_gain_pct"] = float(args["measured_gain_pct"])
        entry["measured_value"] = float(args["measured_value"])
        if args.get("curve"):
            entry["curve"] = _coerce_curve(args["curve"])
        stored = _append_for_context(entry)
        _log_progress_write("evaluator", stored)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Recorded evaluator entry for step {ctx.current_step} "
                        f"(decision={entry['decision']}, "
                        f"reason_category={entry['reason_category']})."
                    ),
                }
            ]
        }

    @tool(
        "append_integrator_progress",
        (
            "Record the Integrator's authoritative combined-candidate verdict. "
            "Call this exactly once as the last action of your turn."
        ),
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "decision": {
                    "type": "string",
                    "enum": list(INTEGRATOR_DECISIONS),
                },
                "included_item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "dropped_item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "remediation_attempts": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 2,
                },
                "measured_gain_pct": {"type": "number"},
                "measured_value": {"type": "number"},
                "required_gain_pct": {"type": "number"},
                "best_candidate_id": {"type": "string"},
                "curve": _CURVE_FIELD_SCHEMA,
            },
            "required": [
                "summary",
                "decision",
                "included_item_ids",
                "dropped_item_ids",
                "remediation_attempts",
                "measured_gain_pct",
                "measured_value",
                "required_gain_pct",
                "best_candidate_id",
            ],
        },
    )
    async def append_integrator_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry = _base_entry("integrator")
        entry.update(
            {
                "summary": args["summary"],
                "decision": args["decision"],
                "included_item_ids": list(args["included_item_ids"]),
                "dropped_item_ids": list(args["dropped_item_ids"]),
                "remediation_attempts": int(args["remediation_attempts"]),
                "measured_gain_pct": float(args["measured_gain_pct"]),
                "measured_value": float(args["measured_value"]),
                "required_gain_pct": float(args["required_gain_pct"]),
                "best_candidate_id": str(args["best_candidate_id"]),
            }
        )
        if args.get("curve"):
            entry["curve"] = _coerce_curve(args["curve"])
        stored = _append_for_context(entry)
        _log_progress_write("integrator", stored)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Recorded authoritative integrator verdict "
                        f"{entry['decision']} for step {ctx.current_step}."
                    ),
                }
            ]
        }

    @tool(
        "append_qa_progress",
        (
            "Record the QA final-verification progress entry in "
            "progress.yaml. Call this exactly once as the last action of "
            "your turn."
        ),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short human-readable summary: your independent "
                    "benchmark numbers, the sanity/accuracy checks you ran, "
                    "and whether they corroborate the loop's numbers.",
                },
                "cumulative_improvement_pct": {
                    "type": "number",
                    "description": "Signed % improvement of the target metric vs the "
                    "roadmap baseline value, normalized so positive = "
                    "better, from your own measurement. Curve mode: the "
                    "mean across concurrency points of the per-point gain "
                    "vs baseline.curve — over optimize.focus_concurrencies "
                    "when task.yaml sets it.",
                },
                "curve": _CURVE_FIELD_SCHEMA,
            },
            "required": ["summary", "cumulative_improvement_pct"],
        },
    )
    async def append_qa_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry = _base_entry("qa")
        entry["summary"] = args["summary"]
        entry["cumulative_improvement_pct"] = float(args["cumulative_improvement_pct"])
        if args.get("curve"):
            entry["curve"] = _coerce_curve(args["curve"])
        stored = _append_for_context(entry)
        _log_progress_write("qa", stored)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Recorded qa entry for step {ctx.current_step} "
                        f"(cumulative_improvement_pct="
                        f"{entry['cumulative_improvement_pct']})."
                    ),
                }
            ]
        }

    # ``read_latest_progress`` is shared; each caller gets a closure so the
    # log attribution shows *who is reading*, not which agent's entries
    # got filtered.
    def _make_read_tool(caller: str):
        @tool(
            "read_latest_progress",
            (
                "Return entries from progress.yaml belonging to the most "
                "recent step(s), as YAML text. Use this to fetch an "
                "upstream agent's summary (e.g. the optimizer reading the "
                "evaluator's REJECT feedback) instead of reading the full "
                "progress.yaml."
            ),
            {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many of the most recent step numbers to "
                        "include. Defaults to 6 (roughly one full round).",
                    },
                    "agent": {
                        "type": "string",
                        "enum": list(_READABLE_AGENTS),
                        "description": "Optional filter: return only entries written by "
                        f"this agent (one of {list(_AGENTS)}). Omit to "
                        "return entries from every agent.",
                    },
                },
                "required": [],
            },
        )
        async def read_latest_progress(args: dict[str, Any]) -> dict[str, Any]:
            steps = int(args.get("steps") or 6)
            agent = args.get("agent") or None
            selected = find_entries(ctx.path, agent=agent, last_steps=steps)
            if not selected:
                text = f"# No optimization entries yet{f' for agent={agent}' if agent else ''}.\n"
            else:
                text = _yaml_dump(selected)
            _log_progress_read(caller, agent, steps, text)
            return {"content": [{"type": "text", "text": text}]}

        return read_latest_progress

    appenders = {
        "benchmarker": append_benchmarker_progress,
        "projector": append_projector_progress,
        "analyzer": append_analyzer_progress,
        "optimizer": append_optimizer_progress,
        "evaluator": append_evaluator_progress,
        "integrator": append_integrator_progress,
        "qa": append_qa_progress,
        "reporter": append_reporter_progress,
    }
    return {agent: [appenders[agent], _make_read_tool(agent)] for agent in _AGENTS}
