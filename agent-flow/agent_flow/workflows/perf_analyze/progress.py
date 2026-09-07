"""Shared progress log for the perf-analyze workflow.

Progress is persisted as a YAML mapping at ``workspace/progress.yaml``
with a single top-level list — the workflow has one logical phase
(analysis) so the multi-list shape used by ``agent_team`` would just be
ceremony here:

    analysis:
      - step: 1
        agent: benchmarker
        timestamp: "2026-06-26T..."
        summary: |
          ...
      - step: 2
        agent: projector
        timestamp: "..."
        summary: ...
      - step: 3
        agent: analyzer
        timestamp: "..."
        summary: ...
      - step: 4
        agent: reporter
        timestamp: "..."
        summary: ...

The pipeline is linear and one-shot with fixed per-role step numbers:
``benchmarker`` logs step 1, ``projector`` step 2 (only when the
``sol`` block enables that stage — otherwise step 2 is simply absent),
``analyzer`` step 3, ``reporter`` step 4. Each entry is a dict with the
shape:

    step: 3
    agent: analyzer             # benchmarker | projector | analyzer | reporter
    timestamp: "2026-06-26T14:32:11"
    summary: |
      ...

Agents never write this file directly. Each agent is given:

- an ``append_*_progress`` MCP tool that records a structured entry, and
- a shared ``read_latest_progress`` tool that returns recent entries as
  YAML text — so e.g. the analyzer can pull the benchmarker's summary
  without reading the whole file with the generic ``Read`` tool.

The orchestrator reads the resulting YAML to confirm each stage logged a
summary before advancing — no regex over prose. The substantive data
flows through the workspace artifact files (``benchmark_results.md``,
``profile_findings.md``, the report), not through these summaries.
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

ANALYSIS_STAGE = "analysis"
_AGENTS = ("benchmarker", "projector", "analyzer", "reporter")


def _empty_progress() -> dict[str, list[dict[str, Any]]]:
    return {ANALYSIS_STAGE: []}


def read_progress(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the YAML mapping at ``path``, returning an empty mapping if it is missing or empty.

    Always returns a dict with exactly the ``analysis`` key, whose value
    is a (possibly empty) list. Raises ``ValueError`` if the file's
    top-level node isn't a mapping or the ``analysis`` value isn't a list
    — that catches malformed / legacy formats early instead of letting
    them silently mis-route entries.
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
            f"`{ANALYSIS_STAGE}` key, got {type(data).__name__}"
        )
    v = data.get(ANALYSIS_STAGE, [])
    if v is None:
        v = []
    if not isinstance(v, list):
        raise ValueError(f"{path}: `{ANALYSIS_STAGE}` must be a list, got {type(v).__name__}")
    return {ANALYSIS_STAGE: v}


def write_progress(path: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    """Persist ``data`` with the canonical key so diffs stay stable."""
    ordered = {ANALYSIS_STAGE: data.get(ANALYSIS_STAGE, [])}
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
    if agent is not None and agent not in _AGENTS:
        return []
    entries = list(read_progress(path)[ANALYSIS_STAGE])
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
    if agent not in _AGENTS:
        return None
    matches = find_entries(path, agent=agent)
    return matches[-1] if matches else None


def _append(path: Path, entry: dict[str, Any]) -> None:
    data = read_progress(path)
    data[ANALYSIS_STAGE].append(entry)
    write_progress(path, data)


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

    The workflow updates ``current_step`` before each agent run so the
    tool stamps the right step number on every entry without the agent
    having to pass (or guess) it.
    """

    path: Path
    current_step: int = 0
    _tool_cache: list[Any] | None = field(default=None, repr=False, compare=False)


def build_progress_tools(ctx: ProgressContext) -> dict[str, list[Any]]:
    """Build the per-agent tool lists for ``BackendConfig(tools=...)``.

    Returns a dict keyed by agent name (``"benchmarker"`` / ``"projector"``
    / ``"analyzer"`` / ``"reporter"``). The tool objects are ``SdkMcpTool``
    instances, which the claude-code backend wraps into an in-process MCP
    server.
    """

    @tool(
        "append_benchmarker_progress",
        (
            "Record a Benchmarker progress entry in progress.yaml. Call "
            "this exactly once as the last action of your turn."
        ),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short human-readable summary: the serve + benchmark "
                    "commands you ran, the operating point (ISL/OSL/"
                    "concurrency), and the headline metrics (throughput, "
                    "TTFT/TPOT/ITL/E2EL). Name the result files you wrote.",
                },
            },
            "required": ["summary"],
        },
    )
    async def append_benchmarker_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "step": ctx.current_step,
            "agent": "benchmarker",
            "timestamp": _now_iso(),
            "summary": args["summary"],
        }
        _append(ctx.path, entry)
        _log_progress_write("benchmarker", entry)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (f"Recorded benchmarker entry for step {ctx.current_step}."),
                }
            ]
        }

    @tool(
        "append_projector_progress",
        (
            "Record a Projector progress entry in progress.yaml. Call this "
            "exactly once as the last action of your turn."
        ),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short human-readable summary: the sources you used "
                    "(skill, peaks calculator, config.json), the "
                    "model/device mapping, the headline "
                    "SOL ceiling and the measured-vs-SOL gap (or the "
                    "unavailability reason), and the files you wrote.",
                },
            },
            "required": ["summary"],
        },
    )
    async def append_projector_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "step": ctx.current_step,
            "agent": "projector",
            "timestamp": _now_iso(),
            "summary": args["summary"],
        }
        _append(ctx.path, entry)
        _log_progress_write("projector", entry)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (f"Recorded projector entry for step {ctx.current_step}."),
                }
            ]
        }

    @tool(
        "append_analyzer_progress",
        (
            "Record an Analyzer progress entry in progress.yaml. Call this "
            "exactly once as the last action of your turn."
        ),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short human-readable summary: which profilers you ran "
                    "(nsys / torch / ncu), the trace files you produced, and the "
                    "ranked bottleneck hypotheses with their key evidence.",
                },
            },
            "required": ["summary"],
        },
    )
    async def append_analyzer_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "step": ctx.current_step,
            "agent": "analyzer",
            "timestamp": _now_iso(),
            "summary": args["summary"],
        }
        _append(ctx.path, entry)
        _log_progress_write("analyzer", entry)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (f"Recorded analyzer entry for step {ctx.current_step}."),
                }
            ]
        }

    @tool(
        "append_reporter_progress",
        (
            "Record a Reporter progress entry in progress.yaml. Call this "
            "exactly once as the last action of your turn."
        ),
        {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short human-readable summary: the main bottleneck you "
                    "concluded, the evidence backing it, and confirmation "
                    "that both performance_report.md and "
                    "performance_report.html were written.",
                },
            },
            "required": ["summary"],
        },
    )
    async def append_reporter_progress(args: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "step": ctx.current_step,
            "agent": "reporter",
            "timestamp": _now_iso(),
            "summary": args["summary"],
        }
        _append(ctx.path, entry)
        _log_progress_write("reporter", entry)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (f"Recorded reporter entry for step {ctx.current_step}."),
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
                "upstream agent's summary (e.g. the analyzer reading the "
                "benchmarker) instead of reading the full progress.yaml."
            ),
            {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many of the most recent step numbers to "
                        "include. Defaults to 4 (the whole pipeline so far).",
                    },
                    "agent": {
                        "type": "string",
                        "enum": list(_AGENTS),
                        "description": "Optional filter: return only entries written by "
                        f"this agent (one of {list(_AGENTS)}). Omit to "
                        "return entries from every agent.",
                    },
                },
                "required": [],
            },
        )
        async def read_latest_progress(args: dict[str, Any]) -> dict[str, Any]:
            steps = int(args.get("steps") or 4)
            agent = args.get("agent") or None
            selected = find_entries(ctx.path, agent=agent, last_steps=steps)
            if not selected:
                text = f"# No analysis entries yet{f' for agent={agent}' if agent else ''}.\n"
            else:
                text = _yaml_dump(selected)
            _log_progress_read(caller, agent, steps, text)
            return {"content": [{"type": "text", "text": text}]}

        return read_latest_progress

    return {
        "benchmarker": [append_benchmarker_progress, _make_read_tool("benchmarker")],
        "projector": [append_projector_progress, _make_read_tool("projector")],
        "analyzer": [append_analyzer_progress, _make_read_tool("analyzer")],
        "reporter": [append_reporter_progress, _make_read_tool("reporter")],
    }
