"""Shared progress log for the agent-team workflow.

Progress is persisted as a YAML mapping at ``workspace/progress.yaml`` with
two top-level lists, one per phase, so iteration counters stay local to
the phase that produced them, plus a ``human_feedback`` list of
user-authored guidance injected via ``--feedback``:

    plan_stage:
      - iteration: 2
        agent: plan_reviewer
        timestamp: "2026-04-23T14:32:11"
        summary: |
          ...free-form summary...
        decision: APPROVE
    build_stage:
      - iteration: 1
        agent: coder
        timestamp: "..."
        summary: ...
    human_feedback:
      - timestamp: "2026-05-13T10:21:33"
        iteration: 3
        stage: build_stage
        summary: |
          Please consider the multi-GPU edge case.

Each agent entry is a dict with the shape:

    iteration: 3
    agent: coder                # plan_drafter | plan_reviewer  (plan_stage)
                                # coder | reviewer | qa         (build_stage)
    timestamp: "2026-04-23T14:32:11"
    summary: |
      ...
    # Role-specific optional fields:
    decision: APPROVE           # plan_drafter:  DRAFT_READY | POLISHING
                                #                | HUMAN_APPROVED
                                # plan_reviewer: APPROVE | REJECT
                                # reviewer:      APPROVE | REJECT
                                # qa:            APPROVE | REJECT
    weighted_score: 8.5         # qa only (0-10); logged, not gating

Each human-feedback entry has ``timestamp``, ``iteration`` (the iteration
that was about to run when the feedback landed — 0 on a fresh start),
``stage`` (``plan_stage`` or ``build_stage`` based on the resume point),
and ``summary`` (the user-supplied text).

Agents never write this file directly. Each agent is given:

- an ``append_*_progress`` MCP tool that records a structured entry into
  the stage that matches the agent's role, and
- a shared ``read_latest_progress`` tool that returns entries from the
  caller's stage as YAML text — saves the agent from reading the whole
  file with the generic ``Read`` tool just to see the latest state.

Build-phase agents (coder, reviewer, qa) additionally get a
``read_human_feedback`` tool that returns every entry in
``human_feedback`` so they can incorporate user-supplied mid-run
guidance.

The orchestrator reads the resulting YAML to extract reviewer and QA
``decision`` fields, and the PlanDrafter's ``decision`` (HUMAN_APPROVED
ends the plan phase) — no regex over prose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from claude_agent_sdk import tool
from rich.syntax import Syntax

from agent_flow.console import print_layer_panel
from agent_flow.logger import get_logger

PLAN_STAGE = "plan_stage"
BUILD_STAGE = "build_stage"
HUMAN_FEEDBACK = "human_feedback"
_STAGES = (PLAN_STAGE, BUILD_STAGE)
_TOP_LEVEL_KEYS = (PLAN_STAGE, BUILD_STAGE, HUMAN_FEEDBACK)

_STAGE_BY_AGENT = {
    "plan_drafter": PLAN_STAGE,
    "plan_reviewer": PLAN_STAGE,
    "coder": BUILD_STAGE,
    "reviewer": BUILD_STAGE,
    "qa": BUILD_STAGE,
}
_AGENTS_BY_STAGE: dict[str, tuple[str, ...]] = {
    PLAN_STAGE: ("plan_drafter", "plan_reviewer"),
    BUILD_STAGE: ("coder", "reviewer", "qa"),
}


def _empty_progress() -> dict[str, list[dict[str, Any]]]:
    return {key: [] for key in _TOP_LEVEL_KEYS}


def read_progress(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the YAML mapping at ``path``; return an empty mapping if the
    file is missing or empty.

    Always returns a dict with exactly the ``plan_stage``, ``build_stage``,
    and ``human_feedback`` keys, each a (possibly empty) list. Raises
    ``ValueError`` if the file's top-level node isn't a mapping or any
    list-valued key isn't a list — that catches the legacy flat-list
    format early instead of letting it silently mis-route entries.
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
            f"{path} must contain a YAML mapping with `plan_stage`, "
            f"`build_stage`, and `human_feedback` keys, got "
            f"{type(data).__name__}")
    result = _empty_progress()
    for key in _TOP_LEVEL_KEYS:
        v = data.get(key, [])
        if v is None:
            v = []
        if not isinstance(v, list):
            raise ValueError(
                f"{path}: `{key}` must be a list, got {type(v).__name__}")
        result[key] = v
    return result


def write_progress(path: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    """Persist ``data`` with a fixed key order so diffs stay stable."""
    ordered = {key: data.get(key, []) for key in _TOP_LEVEL_KEYS}
    path.write_text(
        yaml.safe_dump(ordered,
                       sort_keys=False,
                       allow_unicode=True,
                       default_flow_style=False),
        encoding="utf-8",
    )


def init_progress_file(path: Path) -> None:
    """Write an empty progress.yaml with all top-level keys."""
    write_progress(path, _empty_progress())


def append_human_feedback(
    path: Path,
    *,
    summary: str,
    iteration: int,
    stage: str,
) -> dict[str, Any]:
    """Append one ``human_feedback`` entry to ``progress.yaml`` and return it.

    Used by the orchestrator when the user passes ``--feedback`` so the
    user's mid-run guidance becomes visible to the build-phase agents on
    the next turn. Also emits a styled log panel so the user can see what
    landed.
    """
    if stage not in _STAGES:
        raise ValueError(f"unknown stage: {stage!r}")
    entry = {
        "timestamp": _now_iso(),
        "iteration": int(iteration),
        "stage": stage,
        "summary": summary,
    }
    data = read_progress(path)
    data[HUMAN_FEEDBACK].append(entry)
    write_progress(path, data)
    _log_human_feedback_write(entry)
    return entry


def find_entries(
    path: Path,
    *,
    stage: str,
    agent: str | None = None,
    last_iterations: int | None = None,
) -> list[dict[str, Any]]:
    """Return progress entries from a single ``stage`` (``"plan_stage"`` or
    ``"build_stage"``), optionally restricted by agent and/or by the most
    recent ``last_iterations`` iteration numbers.

    - ``agent=None`` keeps all agents in the stage; otherwise filters by
      ``entry["agent"]``. If ``agent`` is a name from a different stage,
      returns ``[]``.
    - ``last_iterations=None`` keeps all iterations; otherwise keeps
      entries whose ``iteration`` is in
      ``[max_iter - last_iterations + 1, max_iter]`` where ``max_iter`` is
      the highest iteration in the (already agent-filtered) result.
    - Returns ``[]`` when the file is empty or nothing matches.
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
        iter_values = [e["iteration"] for e in entries if "iteration" in e]
        if not iter_values:
            return []
        cutoff = max(iter_values) - last_iterations + 1
        entries = [e for e in entries if e.get("iteration", -1) >= cutoff]
    return entries


def latest_entry(path: Path, agent: str) -> dict[str, Any] | None:
    """Return the most recent entry written by ``agent``, or ``None``."""
    stage = _STAGE_BY_AGENT.get(agent)
    if stage is None:
        return None
    matches = find_entries(path, stage=stage, agent=agent)
    return matches[-1] if matches else None


def _append(path: Path, agent: str, entry: dict[str, Any]) -> None:
    data = read_progress(path)
    data[_STAGE_BY_AGENT[agent]].append(entry)
    write_progress(path, data)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _yaml_dump(data: Any) -> str:
    return yaml.safe_dump(data,
                          sort_keys=False,
                          allow_unicode=True,
                          default_flow_style=False)


def _log_progress_write(agent: str, entry: dict[str, Any]) -> None:
    """Log a freshly written progress entry as a styled YAML panel."""
    body = Syntax(_yaml_dump([entry]),
                  "yaml",
                  theme="ansi_dark",
                  word_wrap=True)
    suffix = f"system · wrote iter {entry.get('iteration')}"
    print_layer_panel(agent, suffix, body, get_logger().console)


def _log_progress_read(caller: str, agent_filter: str | None, iterations: int,
                       text: str) -> None:
    """Log the content returned by ``read_latest_progress``.

    ``caller`` is the agent making the call (planner/coder/reviewer/qa) and
    drives the panel style — so the log shows *who is reading*, not which
    agent's entries got filtered.
    """
    body = Syntax(text, "yaml", theme="ansi_dark", word_wrap=True)
    filter_part = f"agent={agent_filter}" if agent_filter else "all agents"
    suffix = f"system · read ({filter_part}, last {iterations})"
    print_layer_panel(caller, suffix, body, get_logger().console)


def _log_human_feedback_read(caller: str, count: int, text: str) -> None:
    """Log the content returned by ``read_human_feedback``."""
    body = Syntax(text, "yaml", theme="ansi_dark", word_wrap=True)
    suffix = f"system · read human_feedback ({count} entr{'y' if count == 1 else 'ies'})"
    print_layer_panel(caller, suffix, body, get_logger().console)


def _log_human_feedback_write(entry: dict[str, Any]) -> None:
    """Log an orchestrator-side ``human_feedback`` append."""
    body = Syntax(_yaml_dump([entry]),
                  "yaml",
                  theme="ansi_dark",
                  word_wrap=True)
    suffix = (f"system · appended human_feedback "
              f"(stage={entry.get('stage')}, iter {entry.get('iteration')})")
    print_layer_panel("orchestrator", suffix, body, get_logger().console)


@dataclass
class ProgressContext:
    """Shared mutable context captured by the per-agent tool handlers.

    The workflow updates ``current_iteration`` before each agent run so the
    tool stamps the right iteration number on every entry without the agent
    having to pass (or guess) it.
    """

    path: Path
    current_iteration: int = 0
    _tool_cache: list[Any] | None = field(default=None,
                                          repr=False,
                                          compare=False)


def build_progress_tools(ctx: ProgressContext) -> dict[str, list[Any]]:
    """Build the per-agent tool lists for ``BackendConfig(tools=...)``.

    Returns a dict keyed by agent name (``"plan_drafter"`` /
    ``"plan_reviewer"`` / ``"coder"`` / ``"reviewer"`` / ``"qa"``). The
    tool objects are ``SdkMcpTool`` instances, which the claude-code
    backend wraps into an in-process MCP server.

    """

    @tool(
        "append_plan_drafter_progress",
        ("Record a PlanDrafter progress entry in progress.yaml's "
         "`plan_stage`. Call this exactly once as the last action of your "
         "turn."),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type":
                    "string",
                    "description":
                    "Short human-readable summary of what you drafted or "
                    "polished this turn (and, in the human-review phase, "
                    "what the human asked for).",
                },
                "decision": {
                    "type":
                    "string",
                    "enum": ["DRAFT_READY", "POLISHING", "HUMAN_APPROVED"],
                    "description":
                    "DRAFT_READY: plan.md is ready for the AI PlanReviewer "
                    "(use this in the drafter↔reviewer phase). POLISHING: "
                    "human asked for changes (or you couldn't reach the "
                    "human this turn) and the plan is still being revised. "
                    "HUMAN_APPROVED: the human approved the plan via "
                    "ask_human — workflow advances to the build phase.",
                },
            },
            "required": ["summary", "decision"],
        },
    )
    async def append_plan_drafter_progress(
            args: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "iteration": ctx.current_iteration,
            "agent": "plan_drafter",
            "timestamp": _now_iso(),
            "summary": args["summary"],
            "decision": args["decision"],
        }
        _append(ctx.path, "plan_drafter", entry)
        _log_progress_write("plan_drafter", entry)
        return {
            "content": [{
                "type":
                "text",
                "text": (f"Recorded plan_drafter entry for iteration "
                         f"{ctx.current_iteration} "
                         f"(decision={entry['decision']})."),
            }]
        }

    @tool(
        "append_plan_reviewer_progress",
        ("Record a PlanReviewer progress entry in progress.yaml's "
         "`plan_stage`. Call this exactly once as the last action of your "
         "turn."),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type":
                    "string",
                    "description":
                    "Short rationale. On REJECT include the specific, "
                    "actionable items the PlanDrafter must address.",
                },
                "decision": {
                    "type":
                    "string",
                    "enum": ["APPROVE", "REJECT"],
                    "description":
                    "APPROVE if the plan satisfies task.yaml and is concrete "
                    "enough for the Coder to execute; REJECT to loop back "
                    "to the PlanDrafter.",
                },
            },
            "required": ["summary", "decision"],
        },
    )
    async def append_plan_reviewer_progress(
            args: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "iteration": ctx.current_iteration,
            "agent": "plan_reviewer",
            "timestamp": _now_iso(),
            "summary": args["summary"],
            "decision": args["decision"],
        }
        _append(ctx.path, "plan_reviewer", entry)
        _log_progress_write("plan_reviewer", entry)
        return {
            "content": [{
                "type":
                "text",
                "text": (f"Recorded plan_reviewer entry for iteration "
                         f"{ctx.current_iteration} "
                         f"(decision={entry['decision']})."),
            }]
        }

    @tool(
        "append_coder_progress",
        ("Record a Coder progress entry in progress.yaml's `build_stage`. "
         "Call this exactly once as the last action of your turn."),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type":
                    "string",
                    "description":
                    "Short human-readable summary of what you implemented "
                    "or changed this iteration.",
                },
            },
            "required": ["summary"],
        },
    )
    async def append_coder_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "iteration": ctx.current_iteration,
            "agent": "coder",
            "timestamp": _now_iso(),
            "summary": args["summary"],
        }
        _append(ctx.path, "coder", entry)
        _log_progress_write("coder", entry)
        return {
            "content": [{
                "type":
                "text",
                "text": (f"Recorded coder entry for iteration "
                         f"{ctx.current_iteration}."),
            }]
        }

    @tool(
        "append_reviewer_progress",
        ("Record a Reviewer progress entry in progress.yaml's "
         "`build_stage`. Call this exactly once as the last action of your "
         "turn."),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type":
                    "string",
                    "description":
                    "Short rationale. On REJECT include the specific, "
                    "actionable items the Coder must fix.",
                },
                "decision": {
                    "type":
                    "string",
                    "enum": ["APPROVE", "REJECT"],
                    "description":
                    "APPROVE if the change looks right and is ready for QA; "
                    "REJECT to loop back to the Coder.",
                },
            },
            "required": ["summary", "decision"],
        },
    )
    async def append_reviewer_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "iteration": ctx.current_iteration,
            "agent": "reviewer",
            "timestamp": _now_iso(),
            "summary": args["summary"],
            "decision": args["decision"],
        }
        _append(ctx.path, "reviewer", entry)
        _log_progress_write("reviewer", entry)
        return {
            "content": [{
                "type":
                "text",
                "text": (f"Recorded reviewer entry for iteration "
                         f"{ctx.current_iteration} "
                         f"(decision={entry['decision']})."),
            }]
        }

    @tool(
        "append_qa_progress",
        ("Record a QA progress entry in progress.yaml's `build_stage`. "
         "Call this exactly once as the last action of your turn."),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type":
                    "string",
                    "description":
                    "QA report — per-criterion scores, strengths, weaknesses, "
                    "and recommendation.",
                },
                "decision": {
                    "type":
                    "string",
                    "enum": ["APPROVE", "REJECT"],
                    "description":
                    "APPROVE ends the workflow; REJECT sends the work back to "
                    "the Coder.",
                },
                "weighted_score": {
                    "type":
                    "number",
                    "minimum":
                    0,
                    "maximum":
                    10,
                    "description":
                    "Weighted overall score in [0, 10]. The orchestrator "
                    "applies a score floor: an APPROVE below the floor is "
                    "downgraded to a loop-back, so do not pad the score to "
                    "clear the gate.",
                },
            },
            "required": ["summary", "decision", "weighted_score"],
        },
    )
    async def append_qa_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "iteration": ctx.current_iteration,
            "agent": "qa",
            "timestamp": _now_iso(),
            "summary": args["summary"],
            "decision": args["decision"],
            "weighted_score": float(args["weighted_score"]),
        }
        _append(ctx.path, "qa", entry)
        _log_progress_write("qa", entry)
        return {
            "content": [{
                "type":
                "text",
                "text": (f"Recorded qa entry for iteration "
                         f"{ctx.current_iteration} "
                         f"(decision={entry['decision']}, "
                         f"weighted_score={entry['weighted_score']})."),
            }]
        }

    # ``read_latest_progress`` is shared, but each caller gets a closure
    # that (a) reads only from its own stage and (b) restricts the optional
    # ``agent`` filter to that stage's agents. This keeps the plan and
    # build phases isolated end-to-end and lets the read log be attributed
    # to *who is reading*, not which agent's entries got filtered.
    def _make_read_tool(caller: str):
        stage = _STAGE_BY_AGENT[caller]
        stage_agents = list(_AGENTS_BY_STAGE[stage])

        @tool(
            "read_latest_progress",
            (f"Return entries from progress.yaml's `{stage}` belonging to "
             f"the most recent iteration(s), as YAML text. Use this "
             f"instead of reading the full progress.yaml file when you "
             f"only need the latest state for your own phase."),
            {
                "type": "object",
                "properties": {
                    "iterations": {
                        "type":
                        "integer",
                        "minimum":
                        1,
                        "description":
                        "How many of the most recent iteration numbers to "
                        "include. Defaults to 1 (just the latest).",
                    },
                    "agent": {
                        "type":
                        "string",
                        "enum":
                        stage_agents,
                        "description":
                        (f"Optional filter: return only entries written "
                         f"by this agent (must be one of {stage_agents}, "
                         f"i.e. an agent in the same stage as the caller). "
                         f"Omit to return all of {stage}'s entries."),
                    },
                },
                "required": [],
            },
        )
        async def read_latest_progress(args: dict[str, Any]) -> dict[str, Any]:
            iterations = int(args.get("iterations") or 1)
            agent = args.get("agent") or None
            selected = find_entries(ctx.path,
                                    stage=stage,
                                    agent=agent,
                                    last_iterations=iterations)
            if not selected:
                text = (f"# No {stage} progress entries yet"
                        f"{f' for agent={agent}' if agent else ''}.\n")
            else:
                text = _yaml_dump(selected)
            _log_progress_read(caller, agent, iterations, text)
            return {"content": [{"type": "text", "text": text}]}

        return read_latest_progress

    # ``read_human_feedback`` returns every entry the user has injected via
    # ``--feedback``. Build-phase agents share one schema; the caller name
    # only drives panel styling in the log.
    def _make_human_feedback_tool(caller: str):

        @tool(
            "read_human_feedback",
            ("Return every entry in progress.yaml's `human_feedback` list "
             "as YAML text. These are user-authored notes injected via "
             "`--feedback` on resume; treat them as direct guidance from "
             "the human. Call this at the start of your turn so you do "
             "not miss feedback that arrived between iterations."),
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
        async def read_human_feedback(_args: dict[str, Any]) -> dict[str, Any]:
            entries = read_progress(ctx.path)[HUMAN_FEEDBACK]
            if not entries:
                text = "# No human_feedback entries yet.\n"
            else:
                text = _yaml_dump(entries)
            _log_human_feedback_read(caller, len(entries), text)
            return {"content": [{"type": "text", "text": text}]}

        return read_human_feedback

    # QA intentionally does not get `read_latest_progress`: its verdict
    # must be grounded in ``task.yaml`` and the actual code it builds, not
    # downstream artifacts (plan.md, progress.yaml) that could drift.
    # ``read_human_feedback`` is the exception — human feedback is the
    # user's direct voice, not a downstream agent's rationalization, so
    # QA must be able to see it even though the rest of progress.yaml
    # stays out of reach.
    return {
        "plan_drafter": [
            append_plan_drafter_progress,
            _make_read_tool("plan_drafter"),
        ],
        "plan_reviewer": [
            append_plan_reviewer_progress,
            _make_read_tool("plan_reviewer"),
        ],
        "coder": [
            append_coder_progress,
            _make_read_tool("coder"),
            _make_human_feedback_tool("coder"),
        ],
        "reviewer": [
            append_reviewer_progress,
            _make_read_tool("reviewer"),
            _make_human_feedback_tool("reviewer"),
        ],
        "qa": [append_qa_progress,
               _make_human_feedback_tool("qa")],
    }
