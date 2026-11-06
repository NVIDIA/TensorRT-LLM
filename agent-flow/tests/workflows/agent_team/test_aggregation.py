"""Tests for the orchestrator-owned global observation window.

Under ``--concurrent`` the top-level ``workspace/status.md`` and
``workspace/progress.yaml`` are DERIVED, single-writer aggregations: a
whole-DAG rollup plus a node-tagged timeline, both rendered ONLY by the
orchestrator from each node's private ``nodes/<slug>/progress.yaml``. These
tests pin the rollup counts, the per-node rows, the node-tagged and ordered
timeline, idempotency, and clean handling of an empty/missing node.
"""

from __future__ import annotations

import itertools

import yaml

from agent_flow.workflows.agent_team import aggregation
from agent_flow.workflows.agent_team import progress as progress_module
from agent_flow.workflows.agent_team.node_workspace import create_node_workspace


def _fixed_clock(monkeypatch, start: int = 1):
    """Make ``record_progress_entry`` stamp deterministic increasing timestamps."""
    counter = itertools.count(start)
    monkeypatch.setattr(
        progress_module,
        "_now_iso",
        lambda: f"2026-08-04T00:00:{next(counter):02d}",
    )


def _build_two_node_workspace(tmp_path, monkeypatch):
    """Seed g1 (coder→reviewer APPROVE) and g2 (coder) with ordered timestamps.

    Timestamps interleave across nodes so the aggregated timeline exercises the
    ``(timestamp, node)`` ordering: g1@01, g2@02, g1@03.
    """
    _fixed_clock(monkeypatch)
    g1 = create_node_workspace(tmp_path, "g1")
    g2 = create_node_workspace(tmp_path, "g2")

    # 01 -> g1 coder, 02 -> g2 coder, 03 -> g1 reviewer APPROVE
    progress_module.record_progress_entry(
        g1.progress_path, "coder", 1, {"summary": "implemented g1 feature\nsecond line"}
    )
    progress_module.record_progress_entry(
        g2.progress_path, "coder", 1, {"summary": "implemented g2 feature"}
    )
    progress_module.record_progress_entry(
        g1.progress_path,
        "reviewer",
        1,
        {"summary": "g1 looks good", "decision": "APPROVE"},
    )
    return g1, g2


# ------------------------------------------------------------- render_global_status


def test_render_global_status_writes_rollup_and_rows(tmp_path, monkeypatch):
    _build_two_node_workspace(tmp_path, monkeypatch)

    aggregation.render_global_status(
        workspace=tmp_path,
        node_states={"g1": "done", "g2": "running"},
        node_ids=["g1", "g2"],
    )

    text = (tmp_path / "status.md").read_text(encoding="utf-8")
    # Header rollup: total + counts by state.
    assert "Total nodes: 2" in text
    assert "done: 1" in text
    assert "running: 1" in text
    assert "pending: 0" in text
    assert "failed: 0" in text
    assert "blocked: 0" in text

    # Per-node rows: id, state, and the latest decision + one-line summary.
    assert "g1" in text
    assert "g2" in text
    assert "done" in text
    assert "APPROVE" in text
    # g1's latest is the reviewer entry (APPROVE, "g1 looks good").
    assert "g1 looks good" in text
    # Only the first line of a multi-line summary is shown.
    assert "second line" not in text
    # g2's latest is the coder entry with no decision.
    assert "implemented g2 feature" in text


def test_render_global_status_empty_node_yields_clean_row(tmp_path, monkeypatch):
    _fixed_clock(monkeypatch)
    create_node_workspace(tmp_path, "g1")  # never records any progress entry

    # Must not crash on an empty-progress node.
    aggregation.render_global_status(
        workspace=tmp_path,
        node_states={"g1": "pending"},
        node_ids=["g1"],
    )
    text = (tmp_path / "status.md").read_text(encoding="utf-8")
    assert "Total nodes: 1" in text
    assert "g1" in text
    assert "no entries" in text.lower()


def test_render_global_status_missing_node_dir_no_crash(tmp_path):
    # A node id whose sub-workspace was never created still yields a clean row.
    aggregation.render_global_status(
        workspace=tmp_path,
        node_states={"ghost": "blocked"},
        node_ids=["ghost"],
    )
    text = (tmp_path / "status.md").read_text(encoding="utf-8")
    assert "Total nodes: 1" in text
    assert "ghost" in text
    assert "no entries" in text.lower()


def test_render_global_status_is_idempotent(tmp_path, monkeypatch):
    _build_two_node_workspace(tmp_path, monkeypatch)
    kwargs = {
        "workspace": tmp_path,
        "node_states": {"g1": "done", "g2": "running"},
        "node_ids": ["g1", "g2"],
    }
    aggregation.render_global_status(**kwargs)
    first = (tmp_path / "status.md").read_text(encoding="utf-8")
    aggregation.render_global_status(**kwargs)
    second = (tmp_path / "status.md").read_text(encoding="utf-8")
    # Overwrite, not append — a second render reproduces the same file.
    assert first == second


# ----------------------------------------------------------- render_global_progress


def test_render_global_progress_tags_and_orders_timeline(tmp_path, monkeypatch):
    _build_two_node_workspace(tmp_path, monkeypatch)

    aggregation.render_global_progress(workspace=tmp_path, node_ids=["g1", "g2"])

    data = yaml.safe_load((tmp_path / "progress.yaml").read_text(encoding="utf-8"))
    assert set(data.keys()) == {"timeline"}
    timeline = data["timeline"]
    assert len(timeline) == 3

    # Every entry is stamped with its node id.
    assert all("node" in entry for entry in timeline)
    # Ordered by timestamp then node: g1@01, g2@02, g1@03.
    assert [e["node"] for e in timeline] == ["g1", "g2", "g1"]
    assert [e["timestamp"] for e in timeline] == [
        "2026-08-04T00:00:01",
        "2026-08-04T00:00:02",
        "2026-08-04T00:00:03",
    ]
    # Original entry fields survive the aggregation.
    assert timeline[0]["agent"] == "coder"
    assert timeline[2]["decision"] == "APPROVE"


def test_render_global_progress_orders_ties_by_node(tmp_path, monkeypatch):
    # Two entries sharing one timestamp must break the tie on node id.
    monkeypatch.setattr(progress_module, "_now_iso", lambda: "2026-08-04T09:00:00")
    gb = create_node_workspace(tmp_path, "b")
    ga = create_node_workspace(tmp_path, "a")
    progress_module.record_progress_entry(gb.progress_path, "coder", 1, {"summary": "b work"})
    progress_module.record_progress_entry(ga.progress_path, "coder", 1, {"summary": "a work"})

    aggregation.render_global_progress(workspace=tmp_path, node_ids=["b", "a"])
    data = yaml.safe_load((tmp_path / "progress.yaml").read_text(encoding="utf-8"))
    assert [e["node"] for e in data["timeline"]] == ["a", "b"]


def test_render_global_progress_handles_empty_and_missing_nodes(tmp_path, monkeypatch):
    _fixed_clock(monkeypatch)
    g1 = create_node_workspace(tmp_path, "g1")
    create_node_workspace(tmp_path, "g2")  # empty progress
    progress_module.record_progress_entry(g1.progress_path, "coder", 1, {"summary": "only g1"})

    # "missing" has no sub-workspace at all; must not crash.
    aggregation.render_global_progress(workspace=tmp_path, node_ids=["g1", "g2", "missing"])
    data = yaml.safe_load((tmp_path / "progress.yaml").read_text(encoding="utf-8"))
    assert data["timeline"] == [
        {
            "node": "g1",
            "iteration": 1,
            "agent": "coder",
            "timestamp": "2026-08-04T00:00:01",
            "summary": "only g1",
        }
    ]


def test_render_global_progress_preserves_plan_and_feedback(tmp_path, monkeypatch):
    """The shared plan_stage + human_feedback audit trail survives aggregation.

    The plan phase and ``--feedback`` write ``plan_stage`` / ``human_feedback``
    into the SHARED ``workspace/progress.yaml`` BEFORE the concurrent build runs.
    Aggregating the node timeline into that same file must not discard them.
    """
    _fixed_clock(monkeypatch)
    # A node contributes one build entry to the aggregated timeline.
    g1 = create_node_workspace(tmp_path, "g1")
    progress_module.record_progress_entry(g1.progress_path, "coder", 1, {"summary": "built g1"})

    # Seed the SHARED top-level progress.yaml with plan-phase + feedback history,
    # exactly as the plan phase / ``--feedback`` would before the build runs.
    shared = tmp_path / "progress.yaml"
    progress_module.init_progress_file(shared)
    progress_module.record_progress_entry(
        shared, "plan_drafter", 1, {"summary": "drafted plan", "decision": "HUMAN_APPROVED"}
    )
    progress_module.append_human_feedback(
        shared, summary="handle the multi-GPU case", iteration=0, stage="build_stage"
    )

    aggregation.render_global_progress(workspace=tmp_path, node_ids=["g1"])

    data = yaml.safe_load(shared.read_text(encoding="utf-8"))
    # The node timeline is present and node-tagged.
    assert [e["node"] for e in data["timeline"]] == ["g1"]
    # The plan-phase and human-feedback sections survived intact.
    assert len(data["plan_stage"]) == 1
    assert data["plan_stage"][0]["agent"] == "plan_drafter"
    assert data["plan_stage"][0]["decision"] == "HUMAN_APPROVED"
    assert len(data["human_feedback"]) == 1
    assert data["human_feedback"][0]["summary"] == "handle the multi-GPU case"


def test_render_global_progress_is_idempotent(tmp_path, monkeypatch):
    _build_two_node_workspace(tmp_path, monkeypatch)
    aggregation.render_global_progress(workspace=tmp_path, node_ids=["g1", "g2"])
    first = (tmp_path / "progress.yaml").read_text(encoding="utf-8")
    aggregation.render_global_progress(workspace=tmp_path, node_ids=["g1", "g2"])
    second = (tmp_path / "progress.yaml").read_text(encoding="utf-8")
    assert first == second


def test_renderers_never_write_into_node_subworkspace(tmp_path, monkeypatch):
    g1, g2 = _build_two_node_workspace(tmp_path, monkeypatch)
    before_g1 = g1.progress_path.read_text(encoding="utf-8")
    before_g2 = g2.progress_path.read_text(encoding="utf-8")

    aggregation.render_global_status(
        workspace=tmp_path,
        node_states={"g1": "done", "g2": "running"},
        node_ids=["g1", "g2"],
    )
    aggregation.render_global_progress(workspace=tmp_path, node_ids=["g1", "g2"])

    # Per-node private files are pure inputs — never mutated by the renderers.
    assert g1.progress_path.read_text(encoding="utf-8") == before_g1
    assert g2.progress_path.read_text(encoding="utf-8") == before_g2
