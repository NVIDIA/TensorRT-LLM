"""Shared progress log for the pr-review workflow.

Progress is persisted as a YAML mapping at ``workspace/progress.yaml`` with
one top-level list per review stage, so round counters stay local to the
stage that produced them:

    stage1:
      - round: 1
        agent: reviewer
        timestamp: "2026-06-04T14:32:11"
        summary: |
          ...free-form review...
        decision: REQUEST_CHANGES
      - round: 1
        agent: coder
        timestamp: "..."
        summary: ...
        decision: REVISE
    stage2:
      - ...

Each entry is a dict with the shape:

    round: 3
    agent: reviewer             # reviewer | coder
    timestamp: "2026-06-04T14:32:11"
    summary: |
      ...
    decision: APPROVE           # reviewer: APPROVE | REQUEST_CHANGES
                                # coder:    AGREE | REVISE | STAND_FIRM

The same two roles (``reviewer`` / ``coder``) appear in **both** stages, so
entries are keyed by ``(stage, agent)`` rather than by agent alone. The
orchestrator sets ``ProgressContext.current_stage`` and
``ProgressContext.current_round`` before each agent call so the tools stamp
every entry with the right stage and round without the agent having to pass
(or guess) them.

Agents never write this file directly. Each agent is given:

- an ``append_*_progress`` MCP tool that records a structured entry into the
  current stage, and
- a shared ``read_latest_progress`` tool that returns entries from the current
  stage as YAML text — saving the agent from reading the whole file just to
  see the latest state.

The orchestrator reads the resulting YAML to extract the reviewer and coder
``decision`` fields that drive the loop — no regex over prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from claude_agent_sdk import tool
from rich.syntax import Syntax

from agent_flow.console import print_layer_panel
from agent_flow.logger import get_logger

from .state import STAGE1, STAGE2

_STAGES = (STAGE1, STAGE2)
_AGENTS = ("reviewer", "coder")

REVIEWER_DECISIONS = ("APPROVE", "REQUEST_CHANGES")
CODER_DECISIONS = ("AGREE", "REVISE", "STAND_FIRM")


def _empty_progress() -> dict[str, list[dict[str, Any]]]:
    return {key: [] for key in _STAGES}


def read_progress(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the YAML mapping at ``path``, returning an empty mapping if it is missing or empty.

    Always returns a dict with exactly the ``stage1`` and ``stage2`` keys,
    each a (possibly empty) list. Raises ``ValueError`` if the file's
    top-level node isn't a mapping or any stage value isn't a list — that
    catches a malformed file early instead of letting it silently mis-route
    entries.
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
            f"{path} must contain a YAML mapping with `stage1` and `stage2` "
            f"keys, got {type(data).__name__}"
        )
    result = _empty_progress()
    for key in _STAGES:
        v = data.get(key, [])
        if v is None:
            v = []
        if not isinstance(v, list):
            raise ValueError(f"{path}: `{key}` must be a list, got {type(v).__name__}")
        result[key] = v
    return result


def write_progress(path: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    """Persist ``data`` with a fixed key order so diffs stay stable."""
    ordered = {key: data.get(key, []) for key in _STAGES}
    path.write_text(
        yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def init_progress_file(path: Path) -> None:
    """Write an empty progress.yaml with both stage keys."""
    write_progress(path, _empty_progress())


def find_entries(
    path: Path,
    *,
    stage: str,
    agent: str | None = None,
    last_rounds: int | None = None,
) -> list[dict[str, Any]]:
    """Return entries from a single ``stage``, optionally filtered by agent and round.

    - ``agent=None`` keeps both roles in the stage; otherwise filters by
      ``entry["agent"]``.
    - ``last_rounds=None`` keeps all rounds; otherwise keeps entries whose
      ``round`` is in ``[max_round - last_rounds + 1, max_round]`` where
      ``max_round`` is the highest round in the (already agent-filtered)
      result.
    - Returns ``[]`` when the file is empty or nothing matches.
    """
    if stage not in _STAGES:
        raise ValueError(f"unknown stage: {stage!r}")
    if agent is not None and agent not in _AGENTS:
        raise ValueError(f"unknown agent: {agent!r}")
    if last_rounds is not None and last_rounds < 1:
        raise ValueError(f"last_rounds must be >= 1, got {last_rounds}")
    entries = list(read_progress(path)[stage])
    if agent is not None:
        entries = [e for e in entries if e.get("agent") == agent]
    if last_rounds is not None:
        round_values = [e["round"] for e in entries if "round" in e]
        if not round_values:
            return []
        cutoff = max(round_values) - last_rounds + 1
        entries = [e for e in entries if e.get("round", -1) >= cutoff]
    return entries


def latest_entry(path: Path, stage: str, agent: str) -> dict[str, Any] | None:
    """Return the most recent entry written by ``agent`` in ``stage``, or ``None``."""
    matches = find_entries(path, stage=stage, agent=agent)
    return matches[-1] if matches else None


def _append(path: Path, stage: str, entry: dict[str, Any]) -> None:
    data = read_progress(path)
    data[stage].append(entry)
    write_progress(path, data)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _yaml_dump(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _log_progress_write(agent: str, entry: dict[str, Any]) -> None:
    """Log a freshly written progress entry as a styled YAML panel."""
    body = Syntax(_yaml_dump([entry]), "yaml", theme="ansi_dark", word_wrap=True)
    suffix = f"system · wrote {entry.get('stage')} round {entry.get('round')}"
    print_layer_panel(agent, suffix, body, get_logger().console)


def _log_progress_read(
    caller: str, stage: str, agent_filter: str | None, rounds: int, text: str
) -> None:
    """Log the content returned by ``read_latest_progress``."""
    body = Syntax(text, "yaml", theme="ansi_dark", word_wrap=True)
    filter_part = f"agent={agent_filter}" if agent_filter else "all roles"
    suffix = f"system · read {stage} ({filter_part}, last {rounds})"
    print_layer_panel(caller, suffix, body, get_logger().console)


@dataclass
class ProgressContext:
    """Shared mutable context captured by the per-agent tool handlers.

    The workflow updates ``current_stage`` and ``current_round`` before each
    agent run so the tools stamp the right stage and round on every entry
    without the agent having to pass them.
    """

    path: Path
    current_stage: str = STAGE1
    current_round: int = 0


def build_progress_tools(ctx: ProgressContext) -> dict[str, list[Any]]:
    """Build the per-agent tool lists for ``BackendConfig(tools=...)``.

    Returns a dict keyed by role name (``"reviewer"`` / ``"coder"``). The tool
    objects are ``SdkMcpTool`` instances, which the claude-code backend wraps
    into an in-process MCP server.
    """

    @tool(
        "append_reviewer_progress",
        (
            "Record a Reviewer progress entry for the current review stage in "
            "progress.yaml. Call this exactly once as the last action of your turn."
        ),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short rationale grounded in what you inspected / ran. On "
                    "REQUEST_CHANGES, list the specific, actionable items the "
                    "Coder must address, ordered by importance.",
                },
                "decision": {
                    "type": "string",
                    "enum": list(REVIEWER_DECISIONS),
                    "description": "APPROVE if the change is good and you have no outstanding "
                    "requests; REQUEST_CHANGES to ask the Coder for specific fixes.",
                },
            },
            "required": ["summary", "decision"],
        },
    )
    async def append_reviewer_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "round": ctx.current_round,
            "stage": ctx.current_stage,
            "agent": "reviewer",
            "timestamp": _now_iso(),
            "summary": args["summary"],
            "decision": args["decision"],
        }
        _append(ctx.path, ctx.current_stage, entry)
        _log_progress_write("reviewer", entry)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Recorded reviewer entry for {ctx.current_stage} round "
                        f"{ctx.current_round} (decision={entry['decision']})."
                    ),
                }
            ]
        }

    @tool(
        "append_coder_progress",
        (
            "Record a Coder progress entry for the current review stage in "
            "progress.yaml. Call this exactly once as the last action of your turn."
        ),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What you changed this round to address the review, and — for "
                    "any requested change you decline — the rationale for pushing "
                    "back. Be specific so the Reviewer can re-check.",
                },
                "decision": {
                    "type": "string",
                    "enum": list(CODER_DECISIONS),
                    "description": "REVISE: you made changes and want the Reviewer to re-review. "
                    "AGREE: you addressed everything and believe the change is "
                    "good (no outstanding objections). STAND_FIRM: you addressed "
                    "what you accept and decline the rest with rationale — this is "
                    "your final position and ends the stage in your favor.",
                },
            },
            "required": ["summary", "decision"],
        },
    )
    async def append_coder_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "round": ctx.current_round,
            "stage": ctx.current_stage,
            "agent": "coder",
            "timestamp": _now_iso(),
            "summary": args["summary"],
            "decision": args["decision"],
        }
        _append(ctx.path, ctx.current_stage, entry)
        _log_progress_write("coder", entry)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Recorded coder entry for {ctx.current_stage} round "
                        f"{ctx.current_round} (decision={entry['decision']})."
                    ),
                }
            ]
        }

    # ``read_latest_progress`` is shared. Each caller gets a closure so the
    # read log can be attributed to *who is reading*; the entries returned are
    # always from the orchestrator-set ``current_stage`` so the two stages stay
    # isolated end-to-end.
    def _make_read_tool(caller: str):
        @tool(
            "read_latest_progress",
            (
                "Return entries from the current review stage in progress.yaml "
                "belonging to the most recent round(s), as YAML text. Use this "
                "instead of reading the full progress.yaml file when you only need "
                "the latest state for this stage."
            ),
            {
                "type": "object",
                "properties": {
                    "rounds": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many of the most recent rounds to include. "
                        "Defaults to 1 (just the latest).",
                    },
                    "agent": {
                        "type": "string",
                        "enum": list(_AGENTS),
                        "description": "Optional filter: return only entries written by this "
                        "role (`reviewer` or `coder`). Omit to return both.",
                    },
                },
                "required": [],
            },
        )
        async def read_latest_progress(args: dict[str, Any]) -> dict[str, Any]:
            rounds = int(args.get("rounds") or 1)
            agent = args.get("agent") or None
            selected = find_entries(
                ctx.path, stage=ctx.current_stage, agent=agent, last_rounds=rounds
            )
            if not selected:
                text = (
                    f"# No {ctx.current_stage} progress entries yet"
                    f"{f' for agent={agent}' if agent else ''}.\n"
                )
            else:
                text = _yaml_dump(selected)
            _log_progress_read(caller, ctx.current_stage, agent, rounds, text)
            return {"content": [{"type": "text", "text": text}]}

        return read_latest_progress

    return {
        "reviewer": [append_reviewer_progress, _make_read_tool("reviewer")],
        "coder": [append_coder_progress, _make_read_tool("coder")],
    }
