from __future__ import annotations

import asyncio

from agent_flow.workflows.pr_review import sourcing as sourcing_module


def _call(handler, args):
    return asyncio.run(handler(args))


def _tool(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not found in {[t.name for t in tools]}")


def test_build_sourcing_tools_exposes_report_only():
    tools = sourcing_module.build_sourcing_tools(sourcing_module.SourcingContext())
    assert [t.name for t in tools] == ["report_pr_context"]


def test_report_fills_context_and_marks_reported():
    ctx = sourcing_module.SourcingContext()
    assert ctx.reported is False
    report = _tool(sourcing_module.build_sourcing_tools(ctx), "report_pr_context")
    res = _call(
        report.handler,
        {
            "base": "main",
            "head": "feature/x",
            "title": "Add x",
            "author": "alice",
            "url": "https://github.com/o/r/pull/7",
            "body": "does x",
        },
    )
    assert ctx.reported is True
    assert ctx.as_metadata() == {
        "base": "main",
        "head": "feature/x",
        "title": "Add x",
        "author": "alice",
        "url": "https://github.com/o/r/pull/7",
        "body": "does x",
    }
    # The tool result echoes the captured base so the agent sees it landed.
    assert "main" in res["content"][0]["text"]


def test_report_defaults_optional_fields_and_strips_base():
    ctx = sourcing_module.SourcingContext()
    report = _tool(sourcing_module.build_sourcing_tools(ctx), "report_pr_context")
    _call(report.handler, {"base": "  release-2.0  "})
    assert ctx.base == "release-2.0"
    assert ctx.head == ""
    assert ctx.title == ""
    assert ctx.author == ""
    assert ctx.url == ""
    assert ctx.body == ""
    assert ctx.reported is True


def test_reset_clears_prior_report():
    ctx = sourcing_module.SourcingContext(
        base="main", head="feat", title="t", author="a", url="u", body="b", reported=True
    )
    ctx.reset()
    assert ctx.reported is False
    assert ctx.as_metadata() == {
        "base": "",
        "head": "",
        "title": "",
        "author": "",
        "url": "",
        "body": "",
    }
