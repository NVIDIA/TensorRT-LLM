"""Rolling discussion scratchpad for the pr-review workflow.

The discussion is persisted as a free-form markdown file at
``workspace/discussion.md``. Unlike ``progress.yaml`` (an immutable,
append-only audit log), ``discussion.md`` is **rolling state**: every
Reviewer/Coder turn fully overwrites it with a short, clean snapshot the next
agent needs to pick up.

This is where the **local** review conversation lives — the workflow never
posts to the PR/MR, so the open threads, what each side agreed to, and any
declined push-backs are tracked here instead.

Recommended sections (enforced only by the system prompt, not the schema):

1. Open threads — review comments still being worked, with status
   (open / addressed / declined-with-reason / accepted-by-reviewer)
2. Agreed — points both sides have settled
3. Declined push-backs — items the Coder declined, with the rationale
4. Pointers for the next turn — what to look at next

Agents never edit ``discussion.md`` directly. The Reviewer and Coder each get:

- an ``update_discussion`` MCP tool that overwrites the file, and
- a ``read_discussion`` MCP tool that returns the current contents (or a
  placeholder when the file is empty).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool
from rich.markdown import Markdown

from agent_flow.console import print_layer_panel
from agent_flow.logger import get_logger

EMPTY_DISCUSSION_PLACEHOLDER = (
    "# (discussion.md is empty — no review conversation yet)\n"
    "The Reviewer and Coder have not written a discussion snapshot yet.\n"
)


def read_discussion_text(path: Path) -> str:
    """Return ``discussion.md`` contents, or an empty string if missing."""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def write_discussion_text(path: Path, content: str) -> None:
    """Overwrite ``discussion.md`` with ``content``.

    ``discussion.md`` is rolling state, never appended to — every writer fully
    replaces the file with a fresh snapshot.
    """
    path.write_text(content, encoding="utf-8")


def _log_discussion_write(agent: str, content: str) -> None:
    """Log a freshly written discussion.md as a styled markdown panel."""
    body = Markdown(content) if content.strip() else Markdown("_(empty discussion.md written)_")
    print_layer_panel(agent, "system · wrote discussion.md", body, get_logger().console)


def _log_discussion_read(caller: str, content: str) -> None:
    """Log the content returned by ``read_discussion``, attributed to the caller."""
    body = Markdown(content) if content.strip() else Markdown("_(no discussion.md content)_")
    print_layer_panel(caller, "system · read discussion.md", body, get_logger().console)


@dataclass
class DiscussionContext:
    """Shared mutable context captured by the per-agent tool handlers."""

    path: Path


def build_discussion_tools(ctx: DiscussionContext) -> dict[str, list[Any]]:
    """Build the per-agent tool lists for ``BackendConfig(tools=...)``.

    Returns a dict keyed by role name (``"reviewer"`` / ``"coder"``). Both
    roles read and write ``discussion.md`` so the local review conversation
    stays in lock-step across turns.
    """

    def _make_update(caller: str):
        @tool(
            "update_discussion",
            (
                "Overwrite discussion.md with the current rolling snapshot of the "
                "review conversation. Call this exactly once as part of ending your "
                "turn. Keep it short and clean — open threads (with status), points "
                "both sides agreed, declined push-backs (with rationale), and "
                "pointers for the next turn. The file is overwritten, so include "
                "everything that should remain visible to the next agent."
            ),
            {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Full new contents of discussion.md (markdown).",
                    },
                },
                "required": ["content"],
            },
        )
        async def update_discussion(args: dict[str, Any]) -> dict[str, Any]:
            content = args["content"]
            write_discussion_text(ctx.path, content)
            _log_discussion_write(caller, content)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "discussion.md updated.",
                    }
                ]
            }

        return update_discussion

    def _make_read(caller: str):
        @tool(
            "read_discussion",
            (
                "Return the current contents of discussion.md (the rolling local "
                "review conversation maintained by Reviewer and Coder). Returns a "
                "placeholder string when the file is empty."
            ),
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
        async def read_discussion(_args: dict[str, Any]) -> dict[str, Any]:
            text = read_discussion_text(ctx.path) or EMPTY_DISCUSSION_PLACEHOLDER
            _log_discussion_read(caller, text)
            return {"content": [{"type": "text", "text": text}]}

        return read_discussion

    return {
        "reviewer": [_make_update("reviewer"), _make_read("reviewer")],
        "coder": [_make_update("coder"), _make_read("coder")],
    }
