"""Rolling status scratchpad for the agent-team workflow.

Status is persisted as a free-form markdown file at ``workspace/status.md``.
Unlike ``progress.yaml`` (an immutable, append-only audit log), ``status.md``
is **rolling state**: every Coder/Reviewer turn fully overwrites it with a
short, clean snapshot the next agent needs to pick up.

Recommended sections (enforced only by the system prompt, not the schema):

1. Current status — what the artifact looks like right now
2. Execution path — the major steps taken across iterations
3. What's been tried, what worked, what didn't
4. Pointers for the next step — open questions, gotchas, next moves

Agents never edit ``status.md`` directly. The Coder and Reviewer each get:

- an ``update_status`` MCP tool that overwrites the file, and
- a ``read_status`` MCP tool that returns the current contents (or a
  placeholder when the file is empty).

The PlanDrafter and QA do not see status.md: the PlanDrafter runs only
during the plan phase, and QA's verdict must be grounded solely in
``task.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool
from rich.markdown import Markdown

from agent_flow.console import print_layer_panel
from agent_flow.logger import get_logger

EMPTY_STATUS_PLACEHOLDER = (
    "# (status.md is empty — no rolling state yet)\n"
    "The Coder and Reviewer have not written a status snapshot yet.\n")


def read_status_text(path: Path) -> str:
    """Return ``status.md`` contents, or an empty string if missing."""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def write_status_text(path: Path, content: str) -> None:
    """Overwrite ``status.md`` with ``content``.

    ``status.md`` is rolling state, never appended to — every writer fully
    replaces the file with a fresh snapshot.
    """
    path.write_text(content, encoding="utf-8")


def _log_status_write(agent: str, content: str) -> None:
    """Log a freshly written status.md as a styled markdown panel."""
    body = Markdown(content) if content.strip() else Markdown(
        "_(empty status.md written)_")
    print_layer_panel(agent, "system · wrote status.md", body,
                      get_logger().console)


def _log_status_read(caller: str, content: str) -> None:
    """Log the content returned by ``read_status``, attributed to the caller."""
    body = Markdown(content) if content.strip() else Markdown(
        "_(no status.md content)_")
    print_layer_panel(caller, "system · read status.md", body,
                      get_logger().console)


@dataclass
class StatusContext:
    """Shared mutable context captured by the per-agent tool handlers."""

    path: Path


def build_status_tools(ctx: StatusContext) -> dict[str, list[Any]]:
    """Build the per-agent tool lists for ``BackendConfig(tools=...)``.

    Returns a dict keyed by agent name (``"coder"`` / ``"reviewer"``). Only
    those two agents read and write ``status.md``; the PlanDrafter and QA
    do not get the tools.

    """

    def _make_update(caller: str):

        @tool(
            "update_status",
            ("Overwrite status.md with the current rolling status snapshot. "
             "Call this exactly once as part of ending your turn. Keep it "
             "short and clean — current status, execution path, what was "
             "tried, what worked / didn't, and pointers for the next step. "
             "The file is overwritten, so include everything that should "
             "remain visible to the next agent."),
            {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description":
                        "Full new contents of status.md (markdown).",
                    },
                },
                "required": ["content"],
            },
        )
        async def update_status(args: dict[str, Any]) -> dict[str, Any]:
            content = args["content"]
            write_status_text(ctx.path, content)
            _log_status_write(caller, content)
            return {
                "content": [{
                    "type": "text",
                    "text": "status.md updated.",
                }]
            }

        return update_status

    def _make_read(caller: str):

        @tool(
            "read_status",
            ("Return the current contents of status.md (rolling state "
             "maintained by Coder and Reviewer). Returns a placeholder "
             "string when the file is empty."),
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
        async def read_status(_args: dict[str, Any]) -> dict[str, Any]:
            text = read_status_text(ctx.path) or EMPTY_STATUS_PLACEHOLDER
            _log_status_read(caller, text)
            return {"content": [{"type": "text", "text": text}]}

        return read_status

    return {
        "coder": [_make_update("coder"),
                  _make_read("coder")],
        "reviewer": [_make_update("reviewer"),
                     _make_read("reviewer")],
    }
