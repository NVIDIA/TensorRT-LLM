"""Sourcing handshake for the pr-review workflow.

Sourcing a PR/MR — checking its branch out into the local repo and reading its
metadata — is done by a dedicated **sourcing agent**, not by the Python
orchestrator. The agent runs ``gh`` / ``glab`` itself (this workflow no longer
shells out to those binaries) and hands the result back through a single MCP
tool, ``report_pr_context``.

The tool fills a :class:`SourcingContext` the orchestrator owns; after the
agent's turn the orchestrator reads ``base`` from it to compute the diff under
review (pure local ``git`` — see :mod:`.vcs`) and the rest to render
``pr_context.md``. Only the base branch is required; everything else is
best-effort display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import tool

from agent_flow.console import print_layer_panel
from agent_flow.logger import get_logger


@dataclass
class SourcingContext:
    """Mutable sink the ``report_pr_context`` tool writes the agent's findings into.

    The orchestrator constructs one, hands its tool to the sourcing agent, then
    reads the fields back after the agent's turn. ``reported`` distinguishes "the
    agent called the tool" from "the agent left without reporting".
    """

    base: str = ""
    head: str = ""
    title: str = ""
    author: str = ""
    url: str = ""
    body: str = ""
    reported: bool = False

    def reset(self) -> None:
        """Clear any prior report so a re-run starts from a clean slate."""
        self.base = ""
        self.head = ""
        self.title = ""
        self.author = ""
        self.url = ""
        self.body = ""
        self.reported = False

    def as_metadata(self) -> dict[str, str]:
        """Return the reported fields as the metadata dict ``vcs.build_pr_context`` expects."""
        return {
            "base": self.base,
            "head": self.head,
            "title": self.title,
            "author": self.author,
            "url": self.url,
            "body": self.body,
        }


def _log_report(ctx: SourcingContext) -> None:
    """Log a freshly reported PR/MR context as a styled panel."""
    body = (
        f"base: {ctx.base or '(none)'}\n"
        f"head: {ctx.head or '(unknown)'}\n"
        f"title: {ctx.title or '(none)'}\n"
        f"author: {ctx.author or '(unknown)'}\n"
        f"url: {ctx.url or '(none)'}"
    )
    print_layer_panel("sourcing", "system · reported PR/MR context", body, get_logger().console)


def build_sourcing_tools(ctx: SourcingContext) -> list[Any]:
    """Build the sourcing agent's tool list for ``BackendConfig(tools=...)``.

    Returns a one-element list holding the ``report_pr_context`` tool bound to
    ``ctx`` — an ``SdkMcpTool`` the claude-code backend wraps into an in-process
    MCP server.
    """

    @tool(
        "report_pr_context",
        (
            "Report the PR/MR you checked out so the review can begin. Call this "
            "exactly once, as the last action of your turn. The base branch is "
            "required (the orchestrator diffs against it); the rest is best-effort "
            "metadata used to describe the change."
        ),
        {
            "type": "object",
            "properties": {
                "base": {
                    "type": "string",
                    "description": "The branch the PR/MR merges INTO (GitHub "
                    "`baseRefName`, GitLab `target_branch`), e.g. `main`.",
                },
                "head": {
                    "type": "string",
                    "description": "The PR/MR's own source branch (GitHub "
                    "`headRefName`, GitLab `source_branch`).",
                },
                "title": {"type": "string", "description": "PR/MR title."},
                "author": {"type": "string", "description": "PR/MR author (login/username)."},
                "url": {"type": "string", "description": "Web URL of the PR/MR."},
                "body": {"type": "string", "description": "PR/MR description / body text."},
            },
            "required": ["base"],
        },
    )
    async def report_pr_context(args: dict[str, Any]) -> dict[str, Any]:
        ctx.base = str(args.get("base") or "").strip()
        ctx.head = str(args.get("head") or "").strip()
        ctx.title = str(args.get("title") or "")
        ctx.author = str(args.get("author") or "")
        ctx.url = str(args.get("url") or "")
        ctx.body = str(args.get("body") or "")
        ctx.reported = True
        _log_report(ctx)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Recorded PR/MR context (base={ctx.base!r}, head={ctx.head!r}). "
                        f"Sourcing complete."
                    ),
                }
            ]
        }

    return [report_pr_context]
