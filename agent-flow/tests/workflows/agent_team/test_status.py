"""Tests for the agent-team status.md tool flow."""
from __future__ import annotations

import asyncio
from typing import Any

from agent_flow.workflows.agent_team import status as status_module


def _load_status_module():
    return status_module


def _call(handler, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(handler(args))


# ---------------------------------------------------------- file-level helpers


def test_read_status_text_returns_empty_when_missing(tmp_path):
    status = _load_status_module()
    assert status.read_status_text(tmp_path / "missing.md") == ""


def test_read_status_text_returns_file_contents(tmp_path):
    status = _load_status_module()
    path = tmp_path / "status.md"
    path.write_text("# hello\n\nworld\n", encoding="utf-8")
    assert status.read_status_text(path) == "# hello\n\nworld\n"


def test_write_status_text_overwrites(tmp_path):
    """Each write must fully replace the file — never append."""
    status = _load_status_module()
    path = tmp_path / "status.md"

    status.write_status_text(path, "first\n")
    assert path.read_text(encoding="utf-8") == "first\n"

    status.write_status_text(path, "second\n")
    # Second write replaces the first; nothing accumulates.
    assert path.read_text(encoding="utf-8") == "second\n"


# ---------------------------------------------------------------- tool flow


def test_build_status_tools_only_for_coder_and_reviewer(tmp_path):
    """Planner and QA must not get status.md tools."""
    status = _load_status_module()
    ctx = status.StatusContext(path=tmp_path / "status.md")
    tools = status.build_status_tools(ctx)

    assert set(tools.keys()) == {"coder", "reviewer"}
    for role in ("coder", "reviewer"):
        names = [t.name for t in tools[role]]
        assert names == ["update_status", "read_status"]


def test_update_status_tool_overwrites_file(tmp_path):
    status = _load_status_module()
    path = tmp_path / "status.md"
    ctx = status.StatusContext(path=path)
    tools = status.build_status_tools(ctx)
    update = next(t for t in tools["coder"] if t.name == "update_status")

    _call(update.handler, {"content": "# v1\nnotes\n"})
    assert path.read_text(encoding="utf-8") == "# v1\nnotes\n"

    # Overwriting drops the previous content entirely.
    _call(update.handler, {"content": "# v2\n"})
    assert path.read_text(encoding="utf-8") == "# v2\n"


def test_read_status_tool_returns_placeholder_when_missing(tmp_path):
    status = _load_status_module()
    ctx = status.StatusContext(path=tmp_path / "status.md")
    tools = status.build_status_tools(ctx)
    read = next(t for t in tools["coder"] if t.name == "read_status")

    out = _call(read.handler, {})
    assert out["content"][0]["text"] == status.EMPTY_STATUS_PLACEHOLDER


def test_read_status_tool_returns_placeholder_when_empty(tmp_path):
    status = _load_status_module()
    path = tmp_path / "status.md"
    path.write_text("", encoding="utf-8")
    ctx = status.StatusContext(path=path)
    tools = status.build_status_tools(ctx)
    read = next(t for t in tools["reviewer"] if t.name == "read_status")

    out = _call(read.handler, {})
    assert out["content"][0]["text"] == status.EMPTY_STATUS_PLACEHOLDER


def test_read_status_tool_returns_file_contents(tmp_path):
    status = _load_status_module()
    path = tmp_path / "status.md"
    path.write_text("# state\n\nrunning\n", encoding="utf-8")
    ctx = status.StatusContext(path=path)
    tools = status.build_status_tools(ctx)
    read = next(t for t in tools["coder"] if t.name == "read_status")

    out = _call(read.handler, {})
    assert out["content"][0]["text"] == "# state\n\nrunning\n"


def test_coder_and_reviewer_share_same_path(tmp_path):
    """Update from coder must be visible to reviewer's read_status."""
    status = _load_status_module()
    path = tmp_path / "status.md"
    ctx = status.StatusContext(path=path)
    tools = status.build_status_tools(ctx)

    coder_update = next(t for t in tools["coder"] if t.name == "update_status")
    reviewer_read = next(t for t in tools["reviewer"]
                         if t.name == "read_status")

    _call(coder_update.handler, {"content": "coder wrote this\n"})
    out = _call(reviewer_read.handler, {})
    assert out["content"][0]["text"] == "coder wrote this\n"


def test_status_tools_log_panels(tmp_path, monkeypatch):
    """Each tool handler emits a styled markdown panel attributed to caller."""
    status = _load_status_module()
    path = tmp_path / "status.md"
    ctx = status.StatusContext(path=path)

    calls: list[dict[str, Any]] = []

    def _capture(layer_name, title_suffix, body, extra=None):
        calls.append({"layer": layer_name, "suffix": title_suffix})

    monkeypatch.setattr(status, "print_layer_panel", _capture)

    tools = status.build_status_tools(ctx)
    coder_update = next(t for t in tools["coder"] if t.name == "update_status")
    reviewer_update = next(t for t in tools["reviewer"]
                           if t.name == "update_status")
    coder_read = next(t for t in tools["coder"] if t.name == "read_status")
    reviewer_read = next(t for t in tools["reviewer"]
                         if t.name == "read_status")

    _call(coder_update.handler, {"content": "c\n"})
    _call(reviewer_update.handler, {"content": "r\n"})
    _call(coder_read.handler, {})
    _call(reviewer_read.handler, {})

    layers = [c["layer"] for c in calls]
    suffixes = [c["suffix"] for c in calls]

    assert layers == ["coder", "reviewer", "coder", "reviewer"]
    assert suffixes[0] == "system · wrote status.md"
    assert suffixes[1] == "system · wrote status.md"
    assert suffixes[2] == "system · read status.md"
    assert suffixes[3] == "system · read status.md"
