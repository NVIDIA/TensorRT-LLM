"""Tests for the perf-analyze progress.yaml tool flow."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import yaml

from agent_flow.workflows.perf_analyze import progress as progress_module


def _call(handler, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(handler(args))


# --------------------------------------------------------------------- module


def test_read_progress_handles_missing_empty_and_seeded(tmp_path):
    path = tmp_path / "progress.yaml"

    empty = {"analysis": []}
    assert progress_module.read_progress(path) == empty  # missing

    path.write_text("", encoding="utf-8")
    assert progress_module.read_progress(path) == empty  # empty

    path.write_text("analysis: []\n", encoding="utf-8")
    assert progress_module.read_progress(path) == empty  # explicit empty mapping

    path.write_text(
        "analysis:\n  - step: 1\n    agent: benchmarker\n    summary: hi\n",
        encoding="utf-8",
    )
    assert progress_module.read_progress(path) == {
        "analysis": [
            {"step": 1, "agent": "benchmarker", "summary": "hi"},
        ],
    }


def test_read_progress_rejects_legacy_list(tmp_path):
    path = tmp_path / "progress.yaml"
    path.write_text("- step: 1\n  agent: benchmarker\n", encoding="utf-8")
    with pytest.raises(ValueError, match="analysis"):
        progress_module.read_progress(path)


def test_read_progress_rejects_non_list_stage(tmp_path):
    path = tmp_path / "progress.yaml"
    path.write_text("analysis: oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match="analysis"):
        progress_module.read_progress(path)


def test_init_progress_file_writes_canonical_key(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data == {"analysis": []}


def test_tool_handlers_route_entries_with_correct_shape(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)

    ctx = progress_module.ProgressContext(path=path, current_step=1)
    tools = progress_module.build_progress_tools(ctx)

    _call(tools["benchmarker"][0].handler, {"summary": "served + benchmarked"})
    ctx.current_step = 2
    _call(tools["projector"][0].handler, {"summary": "SOL projection + gap"})
    ctx.current_step = 3
    _call(tools["analyzer"][0].handler, {"summary": "nsys + torch traces"})
    ctx.current_step = 4
    _call(tools["reporter"][0].handler, {"summary": "memory-bound verdict"})

    entries = progress_module.read_progress(path)["analysis"]
    assert [e["agent"] for e in entries] == ["benchmarker", "projector", "analyzer", "reporter"]
    assert [e["step"] for e in entries] == [1, 2, 3, 4]

    # No decision field anywhere (this pipeline has no reviewer).
    for e in entries:
        assert "decision" not in e
        assert "T" in e["timestamp"]
        assert len(e["timestamp"]) == len("2026-06-26T14:32:11")


def test_find_entries(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.write_progress(
        path,
        {
            "analysis": [
                {"step": 1, "agent": "benchmarker", "summary": "b"},
                {"step": 2, "agent": "analyzer", "summary": "p"},
                {"step": 3, "agent": "reporter", "summary": "r"},
            ],
        },
    )

    # No filters → all entries.
    assert len(progress_module.find_entries(path)) == 3

    # last_steps=1 → just the reporter (step 3).
    assert [e["agent"] for e in progress_module.find_entries(path, last_steps=1)] == ["reporter"]

    # last_steps=2 → analyzer + reporter.
    assert [e["agent"] for e in progress_module.find_entries(path, last_steps=2)] == [
        "analyzer",
        "reporter",
    ]

    # Agent filter.
    assert [e["step"] for e in progress_module.find_entries(path, agent="benchmarker")] == [1]

    # Empty file → empty list regardless of filters.
    empty = tmp_path / "empty.yaml"
    progress_module.init_progress_file(empty)
    assert progress_module.find_entries(empty) == []
    assert progress_module.find_entries(empty, last_steps=5) == []

    # Invalid last_steps.
    with pytest.raises(ValueError):
        progress_module.find_entries(path, last_steps=0)

    # Unknown agent silently yields empty list.
    assert progress_module.find_entries(path, agent="not_a_role") == []


def test_read_latest_progress_tool_handler(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.write_progress(
        path,
        {
            "analysis": [
                {"step": 1, "agent": "benchmarker", "timestamp": "t1", "summary": "b1"},
                {"step": 2, "agent": "analyzer", "timestamp": "t2", "summary": "p1"},
            ],
        },
    )
    ctx = progress_module.ProgressContext(path=path, current_step=2)
    tools = progress_module.build_progress_tools(ctx)

    # Every role has the read tool alongside the append tool.
    for role in ("benchmarker", "projector", "analyzer", "reporter"):
        assert len(tools[role]) == 2
        names = [t.name for t in tools[role]]
        assert "read_latest_progress" in names

    # The read tool's `agent` enum exposes every role.
    analyzer_read = next(t for t in tools["analyzer"] if t.name == "read_latest_progress")
    enum = analyzer_read.input_schema["properties"]["agent"]["enum"]
    assert sorted(enum) == ["analyzer", "benchmarker", "projector", "reporter"]

    # Analyzer fetches the benchmarker's entry by filter.
    out = _call(analyzer_read.handler, {"agent": "benchmarker"})
    rendered = yaml.safe_load(out["content"][0]["text"])
    assert rendered == [
        {"step": 1, "agent": "benchmarker", "timestamp": "t1", "summary": "b1"},
    ]

    # Empty file gives a human-readable stub.
    empty = tmp_path / "empty.yaml"
    progress_module.init_progress_file(empty)
    ctx2 = progress_module.ProgressContext(path=empty)
    reporter_read2 = next(
        t
        for t in progress_module.build_progress_tools(ctx2)["reporter"]
        if t.name == "read_latest_progress"
    )
    out = _call(reporter_read2.handler, {})
    assert "No analysis entries yet" in out["content"][0]["text"]


def test_read_latest_progress_default_spans_pipeline_with_step_gap(tmp_path):
    """The default window covers all four stages and tolerates a step gap.

    Steps are fixed per role (benchmarker=1, projector=2, analyzer=3,
    reporter=4); a skipped projector leaves step 2 absent. The default
    read must still return every recorded stage — the cutoff math is
    ``max_step - steps + 1``, so with the default of 4 the window reaches
    back to step 1 regardless of the gap.
    """
    path = tmp_path / "progress.yaml"
    progress_module.write_progress(
        path,
        {
            "analysis": [
                {"step": 1, "agent": "benchmarker", "summary": "b"},
                {"step": 3, "agent": "analyzer", "summary": "p"},
                {"step": 4, "agent": "reporter", "summary": "r"},
            ],
        },
    )
    ctx = progress_module.ProgressContext(path=path, current_step=4)
    tools = progress_module.build_progress_tools(ctx)
    reporter_read = next(t for t in tools["reporter"] if t.name == "read_latest_progress")

    out = _call(reporter_read.handler, {})
    rendered = yaml.safe_load(out["content"][0]["text"])
    assert [e["step"] for e in rendered] == [1, 3, 4]


def test_tool_handlers_log_write_and_read(tmp_path, monkeypatch):
    """Each tool handler should emit a styled panel attributed to the caller."""
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)

    calls: list[dict[str, Any]] = []

    def _capture(layer_name, title_suffix, body, extra=None):
        calls.append({"layer": layer_name, "suffix": title_suffix})

    monkeypatch.setattr(progress_module, "print_layer_panel", _capture)

    ctx = progress_module.ProgressContext(path=path, current_step=1)
    tools = progress_module.build_progress_tools(ctx)

    _call(tools["benchmarker"][0].handler, {"summary": "b"})
    ctx.current_step = 2
    _call(tools["analyzer"][0].handler, {"summary": "p"})

    analyzer_read = next(t for t in tools["analyzer"] if t.name == "read_latest_progress")
    # Analyzer reads the benchmarker's entry → styled ANALYZER (caller).
    _call(analyzer_read.handler, {"agent": "benchmarker"})

    layers = [c["layer"] for c in calls]
    suffixes = [c["suffix"] for c in calls]
    assert layers == ["benchmarker", "analyzer", "analyzer"]
    assert suffixes[0].startswith("system · wrote step 1")
    assert suffixes[1].startswith("system · wrote step 2")
    assert "agent=benchmarker" in suffixes[2]


def test_latest_entry_returns_most_recent_for_agent(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.write_progress(
        path,
        {
            "analysis": [
                {"step": 1, "agent": "benchmarker", "summary": "b1"},
                {"step": 2, "agent": "projector", "summary": "d1"},
                {"step": 3, "agent": "analyzer", "summary": "p1"},
            ],
        },
    )
    assert progress_module.latest_entry(path, "projector")["summary"] == "d1"
    assert progress_module.latest_entry(path, "analyzer")["summary"] == "p1"
    assert progress_module.latest_entry(path, "reporter") is None  # never wrote
    assert progress_module.latest_entry(path, "not_a_role") is None
