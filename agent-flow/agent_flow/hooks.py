"""Reusable hook helpers for the claude-code backend.

The Claude Agent SDK supports a ``hooks`` option on ``ClaudeAgentOptions``
that lets callers intercept events such as ``Stop`` (the agent trying to end
its turn). This module builds common hook patterns on top of that API.

The primary helper is :func:`require_tool_call_stop_hook`, which produces a
``Stop`` hook that blocks the agent from ending a turn until it has called
at least one of a set of required tools. This upgrades prompt-only "please
call X before stopping" instructions into a runtime guarantee.

Example::

    from agent_flow import BackendConfig, CLAUDE_CODE_DEFAULT_MODEL
    from agent_flow.hooks import require_tool_call_stop_hook

    hooks = require_tool_call_stop_hook(["append_planner_progress"])
    backend = BackendConfig(kind="claude-code",
                            model=CLAUDE_CODE_DEFAULT_MODEL,
                            tools=planner_tools,
                            hooks=hooks)

"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from claude_agent_sdk.types import HookMatcher


def _iter_transcript_entries(transcript_path: str | Path) -> list[dict]:
    """Parse the JSONL transcript and return a list of entries.

    Malformed lines are skipped silently — a half-written tail line shouldn't
    cause the hook to raise inside the SDK.
    """
    path = Path(transcript_path)
    if not path.is_file():
        return []
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _is_human_user_entry(entry: dict) -> bool:
    """Return True if ``entry`` is a turn-starting user message.

    Tool results are also delivered as ``type=="user"`` records whose content
    is a list of ``tool_result`` blocks; those are not real turn boundaries.
    """
    if entry.get("type") != "user":
        return False
    message = entry.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        non_tool_result = [
            c for c in content
            if not (isinstance(c, dict) and c.get("type") == "tool_result")
        ]
        return bool(non_tool_result)
    return False


def _tool_uses_since_last_user(entries: list[dict]) -> list[str]:
    """Return the tool-use names recorded after the most recent human user
    message in ``entries``.
    """
    last_user_idx = -1
    for i, entry in enumerate(entries):
        if _is_human_user_entry(entry):
            last_user_idx = i
    if last_user_idx < 0:
        # No human user message yet — everything in the transcript counts.
        tail = entries
    else:
        tail = entries[last_user_idx + 1:]

    names: list[str] = []
    for entry in tail:
        if entry.get("type") != "assistant":
            continue
        message = entry.get("message") or {}
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name") or ""
                if name:
                    names.append(name)
    return names


def _matches_required(tool_name: str, required: set[str]) -> bool:
    """Match either on the fully-qualified name or on the bare tool name.

    Tools registered through ``create_sdk_mcp_server`` appear in the
    transcript as ``mcp__<server>__<tool>``. Callers usually only know the
    short tool name, so we accept either form.
    """
    if tool_name in required:
        return True
    short = tool_name.rsplit("__", 1)[-1]
    return short in required


def called_required_tool_this_turn(
    transcript_path: str | Path,
    required_tool_names: Iterable[str],
) -> bool:
    """Return True if any ``required_tool_names`` was called this turn.

    "This turn" means: after the most recent human user message in the
    transcript. Matches either the fully-qualified MCP name
    (``mcp__<server>__<tool>``) or the bare tool name.
    """
    required = set(required_tool_names)
    if not required:
        return True
    entries = _iter_transcript_entries(transcript_path)
    for name in _tool_uses_since_last_user(entries):
        if _matches_required(name, required):
            return True
    return False


def require_tool_call_stop_hook(
    required_tool_names: Iterable[str],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build a ``Stop`` hook configuration that enforces a tool call.

    The returned dict is shaped for ``ClaudeAgentOptions(hooks=...)``:

        {"Stop": [HookMatcher(hooks=[<async callable>])]}

    The hook inspects the transcript and, if none of
    ``required_tool_names`` was called during the current turn, returns
    ``{"decision": "block", "reason": ...}``. This makes the SDK resume the
    turn with the ``reason`` surfaced back to the model.

    To avoid infinite loops, the hook respects the SDK's
    ``stop_hook_active`` flag: on the second stop attempt in the same
    flow it gives up and allows the agent to stop. This caps retries at
    one per stop event, which is enough in practice while keeping the
    guarantee bounded.
    """
    required = list(required_tool_names)
    if not required:
        raise ValueError("required_tool_names must not be empty")
    required_set = set(required)
    tool_list = ", ".join(f"`{name}`" for name in required)
    block_reason = reason or (
        f"You must call one of {tool_list} before ending your turn. "
        "Call it now, then stop.")

    async def _stop_hook(input_data, _tool_use_id, _context):
        # The stop hook re-fires after a block. ``stop_hook_active`` is True
        # on that second invocation — give up then to avoid infinite loops.
        if input_data.get("stop_hook_active"):
            return {}
        transcript_path = input_data.get("transcript_path")
        if not transcript_path:
            return {}
        if called_required_tool_this_turn(transcript_path, required_set):
            return {}
        return {
            "decision": "block",
            "reason": block_reason,
        }

    return {"Stop": [HookMatcher(hooks=[_stop_hook])]}
