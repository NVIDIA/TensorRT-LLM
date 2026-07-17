from __future__ import annotations

import asyncio

from agent_flow.workflows.pr_review import discussion as discussion_module


def _call(handler, args):
    return asyncio.run(handler(args))


def _tool(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not found in {[t.name for t in tools]}")


def test_tool_sets_per_role(tmp_path):
    ctx = discussion_module.DiscussionContext(path=tmp_path / "discussion.md")
    tools = discussion_module.build_discussion_tools(ctx)
    for role in ("reviewer", "coder"):
        names = sorted(t.name for t in tools[role])
        assert names == ["read_discussion", "update_discussion"]


def test_update_then_read_roundtrips(tmp_path):
    path = tmp_path / "discussion.md"
    ctx = discussion_module.DiscussionContext(path=path)
    tools = discussion_module.build_discussion_tools(ctx)

    _call(
        _tool(tools["reviewer"], "update_discussion").handler, {"content": "## Open\n- thread A\n"}
    )
    assert path.read_text(encoding="utf-8") == "## Open\n- thread A\n"

    # The coder reads what the reviewer wrote — shared scratchpad.
    res = _call(_tool(tools["coder"], "read_discussion").handler, {})
    assert "thread A" in res["content"][0]["text"]


def test_read_returns_placeholder_when_empty(tmp_path):
    ctx = discussion_module.DiscussionContext(path=tmp_path / "discussion.md")
    tools = discussion_module.build_discussion_tools(ctx)
    res = _call(_tool(tools["coder"], "read_discussion").handler, {})
    assert res["content"][0]["text"] == discussion_module.EMPTY_DISCUSSION_PLACEHOLDER


def test_update_overwrites_not_appends(tmp_path):
    path = tmp_path / "discussion.md"
    ctx = discussion_module.DiscussionContext(path=path)
    update = _tool(discussion_module.build_discussion_tools(ctx)["reviewer"], "update_discussion")
    _call(update.handler, {"content": "first\n"})
    _call(update.handler, {"content": "second\n"})
    assert path.read_text(encoding="utf-8") == "second\n"
