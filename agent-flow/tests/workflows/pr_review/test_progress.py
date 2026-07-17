from __future__ import annotations

import asyncio

import pytest
import yaml

from agent_flow.workflows.pr_review import progress as progress_module
from agent_flow.workflows.pr_review.state import STAGE1, STAGE2


def _call(handler, args):
    return asyncio.run(handler(args))


def _ctx(tmp_path, stage=STAGE1, rnd=1):
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)
    return progress_module.ProgressContext(path=path, current_stage=stage, current_round=rnd)


def _tool(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not found in {[t.name for t in tools]}")


def test_init_progress_file_has_both_stage_keys(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"stage1": [], "stage2": []}


def test_tool_sets_per_role(tmp_path):
    tools = progress_module.build_progress_tools(_ctx(tmp_path))
    reviewer_names = sorted(t.name for t in tools["reviewer"])
    coder_names = sorted(t.name for t in tools["coder"])
    assert reviewer_names == ["append_reviewer_progress", "read_latest_progress"]
    assert coder_names == ["append_coder_progress", "read_latest_progress"]


def test_append_and_latest_entry_keyed_by_stage_and_agent(tmp_path):
    ctx = _ctx(tmp_path, stage=STAGE1, rnd=1)
    tools = progress_module.build_progress_tools(ctx)

    _call(
        _tool(tools["reviewer"], "append_reviewer_progress").handler,
        {"summary": "looks off", "decision": "REQUEST_CHANGES"},
    )
    _call(
        _tool(tools["coder"], "append_coder_progress").handler,
        {"summary": "fixed it", "decision": "REVISE"},
    )

    # A later stage2 entry must not collide with stage1 entries.
    ctx.current_stage = STAGE2
    _call(
        _tool(tools["reviewer"], "append_reviewer_progress").handler,
        {"summary": "good", "decision": "APPROVE"},
    )

    rev1 = progress_module.latest_entry(ctx.path, STAGE1, "reviewer")
    coder1 = progress_module.latest_entry(ctx.path, STAGE1, "coder")
    rev2 = progress_module.latest_entry(ctx.path, STAGE2, "reviewer")
    assert rev1["decision"] == "REQUEST_CHANGES" and rev1["round"] == 1
    assert rev1["stage"] == STAGE1 and rev1["agent"] == "reviewer"
    assert coder1["decision"] == "REVISE"
    assert rev2["decision"] == "APPROVE"
    assert progress_module.latest_entry(ctx.path, STAGE2, "coder") is None


def test_find_entries_filters_by_agent_and_rounds(tmp_path):
    ctx = _ctx(tmp_path, stage=STAGE1, rnd=1)
    tools = progress_module.build_progress_tools(ctx)
    rev = _tool(tools["reviewer"], "append_reviewer_progress")
    cod = _tool(tools["coder"], "append_coder_progress")

    for rnd, (rdec, cdec) in enumerate(
        [("REQUEST_CHANGES", "REVISE"), ("REQUEST_CHANGES", "REVISE"), ("APPROVE", "AGREE")],
        start=1,
    ):
        ctx.current_round = rnd
        _call(rev.handler, {"summary": f"r{rnd}", "decision": rdec})
        _call(cod.handler, {"summary": f"c{rnd}", "decision": cdec})

    only_reviewer = progress_module.find_entries(ctx.path, stage=STAGE1, agent="reviewer")
    assert [e["round"] for e in only_reviewer] == [1, 2, 3]
    last_round = progress_module.find_entries(ctx.path, stage=STAGE1, last_rounds=1)
    assert {e["round"] for e in last_round} == {3}
    assert {e["agent"] for e in last_round} == {"reviewer", "coder"}


def test_read_latest_progress_reads_current_stage(tmp_path):
    ctx = _ctx(tmp_path, stage=STAGE1, rnd=1)
    tools = progress_module.build_progress_tools(ctx)
    _call(
        _tool(tools["reviewer"], "append_reviewer_progress").handler,
        {"summary": "s1 note", "decision": "APPROVE"},
    )

    read = _tool(tools["coder"], "read_latest_progress").handler
    res = _call(read, {})
    assert "s1 note" in res["content"][0]["text"]

    # Switching the context to stage2 makes the same tool read stage2 (empty).
    ctx.current_stage = STAGE2
    res2 = _call(read, {})
    assert "No stage2 progress entries yet" in res2["content"][0]["text"]


def test_find_entries_rejects_unknown_stage_and_agent(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)
    with pytest.raises(ValueError, match="unknown stage"):
        progress_module.find_entries(path, stage="nope")
    with pytest.raises(ValueError, match="unknown agent"):
        progress_module.find_entries(path, stage=STAGE1, agent="nope")


def test_read_progress_rejects_non_mapping(tmp_path):
    path = tmp_path / "progress.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        progress_module.read_progress(path)
