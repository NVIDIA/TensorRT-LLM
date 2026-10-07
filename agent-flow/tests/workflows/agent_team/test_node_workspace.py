"""Tests for the per-node sub-workspace and node-scoped contexts."""

from __future__ import annotations

from agent_flow.workflows.agent_team import node_workspace as node_workspace_module
from agent_flow.workflows.agent_team.progress import ProgressContext, read_progress
from agent_flow.workflows.agent_team.status import StatusContext, write_status_text


def _load_node_workspace_module():
    return node_workspace_module


# ------------------------------------------------------------------- slug rule


def test_node_dir_slug_replaces_path_separators():
    node_workspace = _load_node_workspace_module()
    assert node_workspace.node_dir_slug("a/b") == "a_b"
    assert node_workspace.node_dir_slug("a\\b") == "a_b"
    assert node_workspace.node_dir_slug("s1/g2/coder") == "s1_g2_coder"


def test_node_dir_slug_passes_through_dotted_ids():
    node_workspace = _load_node_workspace_module()
    # Dots are legal in a single path component, so ``s1.g2`` is unchanged.
    assert node_workspace.node_dir_slug("s1.g2") == "s1.g2"
    assert node_workspace.node_dir_slug("plain") == "plain"


# ---------------------------------------------------------------- creation


def test_create_node_workspace_makes_dir_and_seed_files(tmp_path):
    node_workspace = _load_node_workspace_module()
    ws = node_workspace.create_node_workspace(tmp_path, "s1.g2")

    node_dir = tmp_path / "nodes" / "s1.g2"
    assert node_dir.is_dir()
    assert ws.node_id == "s1.g2"
    assert ws.dir == node_dir
    assert ws.status_path == node_dir / "status.md"
    assert ws.progress_path == node_dir / "progress.yaml"

    # status.md starts empty, matching the shared-workspace convention.
    assert ws.status_path.is_file()
    assert ws.status_path.read_text(encoding="utf-8") == ""

    # progress.yaml is a schema-valid empty progress log.
    assert ws.progress_path.is_file()
    data = read_progress(ws.progress_path)
    assert set(data.keys()) == {"plan_stage", "build_stage", "human_feedback"}
    assert data == {"plan_stage": [], "build_stage": [], "human_feedback": []}


def test_create_node_workspace_slugs_path_separated_ids(tmp_path):
    node_workspace = _load_node_workspace_module()
    ws = node_workspace.create_node_workspace(tmp_path, "s1/g2")

    assert ws.dir == tmp_path / "nodes" / "s1_g2"
    assert ws.dir.is_dir()
    assert ws.status_path.is_file()
    assert ws.progress_path.is_file()


def test_two_node_ids_get_isolated_dirs_and_files(tmp_path):
    node_workspace = _load_node_workspace_module()
    a = node_workspace.create_node_workspace(tmp_path, "s1.g1")
    b = node_workspace.create_node_workspace(tmp_path, "s1.g2")

    assert a.dir != b.dir
    assert a.status_path != b.status_path
    assert a.progress_path != b.progress_path


# --------------------------------------------------------- scoped contexts


def test_progress_context_points_at_node_private_file(tmp_path):
    node_workspace = _load_node_workspace_module()
    ws = node_workspace.create_node_workspace(tmp_path, "s1.g2")

    ctx = ws.progress_context()
    assert isinstance(ctx, ProgressContext)
    assert ctx.path == ws.progress_path


def test_status_context_points_at_node_private_file(tmp_path):
    node_workspace = _load_node_workspace_module()
    ws = node_workspace.create_node_workspace(tmp_path, "s1.g2")

    ctx = ws.status_context()
    assert isinstance(ctx, StatusContext)
    assert ctx.path == ws.status_path


def test_status_write_stays_isolated_between_nodes(tmp_path):
    """Writing through one node's status context must not touch another."""
    node_workspace = _load_node_workspace_module()
    a = node_workspace.create_node_workspace(tmp_path, "s1.g1")
    b = node_workspace.create_node_workspace(tmp_path, "s1.g2")

    write_status_text(a.status_context().path, "node a rolling state\n")

    assert a.status_path.read_text(encoding="utf-8") == "node a rolling state\n"
    # b's private status.md is untouched — still the empty seed.
    assert b.status_path.read_text(encoding="utf-8") == ""


# ------------------------------------------------------------- idempotency


def test_create_node_workspace_preserves_existing_status(tmp_path):
    """A second call on resume must not clobber a written status.md."""
    node_workspace = _load_node_workspace_module()
    first = node_workspace.create_node_workspace(tmp_path, "s1.g2")
    write_status_text(first.status_path, "important rolling state\n")

    second = node_workspace.create_node_workspace(tmp_path, "s1.g2")

    assert second.dir == first.dir
    assert second.status_path.read_text(encoding="utf-8") == "important rolling state\n"


def test_create_node_workspace_preserves_existing_progress(tmp_path):
    """A second call on resume must not clobber recorded progress entries."""
    node_workspace = _load_node_workspace_module()
    first = node_workspace.create_node_workspace(tmp_path, "s1.g2")

    data = read_progress(first.progress_path)
    data["build_stage"].append({"iteration": 1, "agent": "coder", "summary": "did work"})
    from agent_flow.workflows.agent_team.progress import write_progress

    write_progress(first.progress_path, data)

    node_workspace.create_node_workspace(tmp_path, "s1.g2")

    reloaded = read_progress(first.progress_path)
    assert reloaded["build_stage"] == [{"iteration": 1, "agent": "coder", "summary": "did work"}]


def test_create_node_workspace_reseeds_empty_status(tmp_path):
    """An absent/empty status.md is (re)seeded to an empty string, never left missing."""
    node_workspace = _load_node_workspace_module()
    ws = node_workspace.create_node_workspace(tmp_path, "s1.g2")
    # Simulate an interrupted run that left status.md truncated/empty.
    ws.status_path.unlink()

    reopened = node_workspace.create_node_workspace(tmp_path, "s1.g2")
    assert reopened.status_path.is_file()
    assert reopened.status_path.read_text(encoding="utf-8") == ""
