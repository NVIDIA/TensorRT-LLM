"""Tests for the overlapped (``--concurrent-review``) build phase."""

from __future__ import annotations

import asyncio
import subprocess
import threading
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.agent_team import concurrent_review as cr
from agent_flow.workflows.agent_team import progress as progress_module
from agent_flow.workflows.agent_team import workflow as workflow_module

_TIMEOUT = 10.0


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A one-commit git repo standing in for the Coder's checkout."""
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "test")
    (path / "file.txt").write_text("v1\n", encoding="utf-8")
    _git(path, "add", "file.txt")
    _git(path, "commit", "-q", "-m", "initial")
    return path


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_workspace(tmp_path: Path, repo: Path | None) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    body = {"description": "demo"}
    if repo is not None:
        body["review_snapshot_repo"] = str(repo)
    (workspace / "task.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    return workspace


def _append(workflow, agent: str, entry: dict) -> None:
    progress_module._append(workflow.progress_path, agent, {"agent": agent, **entry})


def _reviewer_notices(workflow) -> list[dict]:
    return progress_module.read_notices(workflow.progress_path, cr.REVIEWER_NOTICE_SOURCE)


def _stub_plan(workflow) -> None:
    """Skip the plan phase and neuter session recycling."""
    workflow.plan_path.write_text("plan\n", encoding="utf-8")
    workflow.acceptance_criteria_path.write_text("- [ ] done\n", encoding="utf-8")
    workflow._run_plan_phase = lambda state, log: setattr(
        state, "stage", workflow_module.STAGE_CODER
    )
    workflow._reset_coder = lambda: None
    workflow._reset_reviewer = lambda: None


# --------------------------------------------------------------- flag off


def test_flag_off_keeps_sequential_ordering(tmp_path, repo):
    """With the flag off the build phase is coder -> reviewer -> qa, no addendum."""
    workspace = _make_workspace(tmp_path, repo)
    workflow = workflow_module.AgentTeamWorkflow(workspace=workspace, num_iterations=3)
    trace: list[tuple[int, str]] = []
    _stub_plan(workflow)

    def coder(iteration: int, **kwargs):
        trace.append((iteration, "coder"))
        # The sequential loop must not pass any concurrent-review extras.
        assert kwargs == {}

    def reviewer(iteration: int, **kwargs):
        trace.append((iteration, "reviewer"))
        assert kwargs == {}
        _append(workflow, "reviewer", {"iteration": iteration, "decision": "APPROVE"})

    def qa(iteration: int):
        trace.append((iteration, "qa"))
        _append(
            workflow,
            "qa",
            {"iteration": iteration, "decision": "APPROVE", "weighted_score": 9.0},
        )

    workflow._run_coder = coder
    workflow._run_reviewer = reviewer
    workflow._run_qa = qa
    try:
        workflow.run(workspace / "task.yaml")
    finally:
        workflow.close()

    assert trace == [(1, "coder"), (1, "reviewer"), (1, "qa")]
    # No snapshot ref is created and no reviewer notice is posted.
    assert not _reviewer_notices(workflow)
    assert _git(repo, "for-each-ref", "refs/agent-flow") == ""


# ---------------------------------------------------------------- flag on


def test_concurrent_review_overlaps_and_injects_verdict(tmp_path, repo):
    """reviewer(i) and coder(i+1) run at the same time; the verdict lands mid-turn."""
    workspace = _make_workspace(tmp_path, repo)
    workflow = workflow_module.AgentTeamWorkflow(
        workspace=workspace, num_iterations=4, concurrent_review=True
    )
    _stub_plan(workflow)

    trace: list[tuple[int, str]] = []
    reviewer_started = threading.Event()
    coder2_started = threading.Event()
    observed: dict[str, object] = {}
    # reviewer(1) REJECTs so the pipeline keeps overlapping; reviewer(2)
    # APPROVEs, which drains the pipeline into a fresh sequential review.
    decisions = {1: "REJECT", 2: "APPROVE", 3: "APPROVE"}

    def coder(iteration: int, *, addendum: str = ""):
        trace.append((iteration, "coder"))
        if iteration == 2:
            observed["addendum"] = addendum
            coder2_started.set()
            # Deadlocks (and fails on timeout) unless the reviewer really
            # is running concurrently.
            assert reviewer_started.wait(_TIMEOUT), "reviewer(1) never started"
            # The verdict must reach the coder while its turn is open.
            for _ in range(int(_TIMEOUT * 100)):
                notices = _reviewer_notices(workflow)
                if notices:
                    observed["notice"] = notices[0]["summary"]
                    break
                threading.Event().wait(0.01)

    def reviewer(iteration: int, *, snapshot=None):
        trace.append((iteration, "reviewer"))
        if iteration == 1:
            observed["snapshot_commit"] = snapshot.commit
            reviewer_started.set()
            assert coder2_started.wait(_TIMEOUT), "coder(2) never started"
        _append(
            workflow,
            "reviewer",
            {"iteration": iteration, "decision": decisions[iteration]},
        )

    def qa(iteration: int):
        trace.append((iteration, "qa"))
        _append(
            workflow,
            "qa",
            {"iteration": iteration, "decision": "APPROVE", "weighted_score": 9.0},
        )

    workflow._run_coder = coder
    workflow._run_reviewer = reviewer
    workflow._run_qa = qa
    try:
        workflow.run(workspace / "task.yaml")
    finally:
        workflow.close()

    # coder(1) -> [reviewer(1) || coder(2)] -> [reviewer(2) || coder(3)]
    # -> reviewer(2) APPROVEd, so iteration 3 gets a fresh sequential
    # review before QA is allowed to see it.
    assert trace[0] == (1, "coder")
    assert set(trace[1:3]) == {(1, "reviewer"), (2, "coder")}
    assert trace[-3:] == [(3, "coder"), (3, "reviewer"), (3, "qa")]

    head = _git(repo, "rev-parse", "HEAD")
    assert observed["snapshot_commit"] == head
    assert _git(repo, "rev-parse", cr.review_ref(1)) == head
    assert "FROZEN" not in str(observed["addendum"])  # that wording is the reviewer's
    assert "MUST NOT modify" in str(observed["addendum"])
    assert head in str(observed["addendum"])
    assert "REVIEW VERDICT for iteration 1: REJECT" in str(observed["notice"])
    assert str(workflow._review_status_path(1)) in str(observed["notice"])


def test_reviewer_notice_is_hidden_from_read_human_feedback(tmp_path, repo):
    """A ``source: reviewer`` notice is not served as the human's own voice."""
    workspace = _make_workspace(tmp_path, repo)
    workflow = workflow_module.AgentTeamWorkflow(
        workspace=workspace, num_iterations=2, concurrent_review=True
    )
    try:
        progress_module.append_human_feedback(
            workflow.progress_path, summary="from the user", iteration=1, stage="build_stage"
        )
        workflow._post_verdict_notice(1, "APPROVE")
    finally:
        workflow.close()

    entries = yaml.safe_load(workflow.progress_path.read_text(encoding="utf-8"))
    human = progress_module.read_notices(workflow.progress_path, progress_module.SOURCE_HUMAN)
    reviewer = _reviewer_notices(workflow)
    assert len(entries["human_feedback"]) == 2
    assert [e["summary"] for e in human] == ["from the user"]
    assert len(reviewer) == 1
    # The historical human entry shape is untouched (no ``source`` key).
    assert "source" not in human[0]


# ---------------------------------------------------------- dirty fallback


def test_dirty_tree_falls_back_to_sequential_review(tmp_path, repo, monkeypatch):
    """An uncommitted change means no snapshot, so the review stays sequential."""
    (repo / "file.txt").write_text("uncommitted\n", encoding="utf-8")
    workspace = _make_workspace(tmp_path, repo)
    workflow = workflow_module.AgentTeamWorkflow(
        workspace=workspace, num_iterations=3, concurrent_review=True
    )
    _stub_plan(workflow)
    trace: list[tuple[int, str]] = []
    warnings: list[str] = []
    original = workflow_module.print_message
    monkeypatch.setattr(
        workflow_module,
        "print_message",
        lambda message, log=None: (warnings.append(str(message)), original(message, log))[1],
    )

    def coder(iteration: int, *, addendum: str = ""):
        trace.append((iteration, "coder"))
        assert addendum == "", "no review can be in flight without a snapshot"

    def reviewer(iteration: int, *, snapshot=None):
        trace.append((iteration, "reviewer"))
        assert snapshot is None
        _append(workflow, "reviewer", {"iteration": iteration, "decision": "APPROVE"})

    def qa(iteration: int):
        trace.append((iteration, "qa"))
        _append(
            workflow,
            "qa",
            {"iteration": iteration, "decision": "APPROVE", "weighted_score": 9.0},
        )

    workflow._run_coder = coder
    workflow._run_reviewer = reviewer
    workflow._run_qa = qa
    try:
        workflow.run(workspace / "task.yaml")
    finally:
        workflow.close()

    assert trace == [(1, "coder"), (1, "reviewer"), (1, "qa")]
    assert not _reviewer_notices(workflow)
    assert any("dirty" in w and "SEQUENTIAL" in w for w in warnings), warnings


def test_snapshot_repo_missing_falls_back_to_sequential(tmp_path):
    """No configured repo is a warning plus sequential review, not a crash."""
    workspace = _make_workspace(tmp_path, None)
    workflow = workflow_module.AgentTeamWorkflow(
        workspace=workspace, num_iterations=2, concurrent_review=True
    )
    _stub_plan(workflow)
    trace: list[tuple[int, str]] = []

    workflow._run_coder = lambda iteration, addendum="": trace.append((iteration, "coder"))

    def reviewer(iteration: int, *, snapshot=None):
        trace.append((iteration, "reviewer"))
        assert snapshot is None
        _append(workflow, "reviewer", {"iteration": iteration, "decision": "APPROVE"})

    def qa(iteration: int):
        trace.append((iteration, "qa"))
        _append(
            workflow,
            "qa",
            {"iteration": iteration, "decision": "APPROVE", "weighted_score": 9.0},
        )

    workflow._run_reviewer = reviewer
    workflow._run_qa = qa
    try:
        workflow.run(workspace / "task.yaml")
    finally:
        workflow.close()

    assert trace == [(1, "coder"), (1, "reviewer"), (1, "qa")]


# ------------------------------------------------- approve while coder runs


def test_approve_while_coder_runs_does_not_kill_the_coder(tmp_path, repo):
    """An APPROVE lets coder(i+1) finish, then a fresh review pass gates QA."""
    workspace = _make_workspace(tmp_path, repo)
    workflow = workflow_module.AgentTeamWorkflow(
        workspace=workspace, num_iterations=4, concurrent_review=True
    )
    _stub_plan(workflow)

    trace: list[str] = []
    reviewer_done = threading.Event()
    coder2_finished = threading.Event()

    def coder(iteration: int, *, addendum: str = ""):
        trace.append(f"coder{iteration}:start")
        if iteration == 2:
            # Keep working past the APPROVE — the orchestrator must not
            # cut the turn short.
            assert reviewer_done.wait(_TIMEOUT), "reviewer(1) never finished"
            coder2_finished.set()
        trace.append(f"coder{iteration}:end")

    def reviewer(iteration: int, *, snapshot=None):
        trace.append(f"reviewer{iteration}")
        _append(workflow, "reviewer", {"iteration": iteration, "decision": "APPROVE"})
        if iteration == 1:
            reviewer_done.set()

    def qa(iteration: int):
        trace.append(f"qa{iteration}")
        assert coder2_finished.is_set(), "QA ran before coder(2) finished its turn"
        _append(
            workflow,
            "qa",
            {"iteration": iteration, "decision": "APPROVE", "weighted_score": 9.0},
        )

    workflow._run_coder = coder
    workflow._run_reviewer = reviewer
    workflow._run_qa = qa
    try:
        workflow.run(workspace / "task.yaml")
    finally:
        workflow.close()

    # coder(2) completes its turn, then iteration 2 gets its own review on
    # the new snapshot before QA sees anything.
    assert trace[-3:] == ["coder2:end", "reviewer2", "qa2"]
    assert trace.index("coder2:end") < trace.index("reviewer2")
    assert trace.index("reviewer2") < trace.index("qa2")


# ---------------------------------------------------------------- resume


def test_resume_reruns_the_in_flight_review_from_the_snapshot(tmp_path, repo):
    """A checkpoint with a review in flight re-reviews that exact commit."""
    workspace = _make_workspace(tmp_path, repo)
    snapshot = cr.snapshot_repo(repo, 1)
    workflow = workflow_module.AgentTeamWorkflow(
        workspace=workspace, num_iterations=2, concurrent_review=True
    )
    seen: list = []
    workflow._run_reviewer = lambda iteration, *, snapshot=None: seen.append(
        (iteration, snapshot.commit if snapshot else None)
    )
    workflow._checkpoint = lambda state: None
    state = workflow_module.WorkflowState(
        task_path=str(workspace / "task.yaml"),
        num_iterations=2,
        next_iteration_index=1,
        stage=workflow_module.STAGE_CODER,
        review_in_flight_iteration=1,
        review_snapshot_repo=str(repo),
        review_snapshot_commit=snapshot.commit,
        review_snapshot_ref=snapshot.ref,
    )
    try:
        workflow._resume_in_flight_review(state, None)
    finally:
        workflow.close()

    assert seen == [(1, snapshot.commit)]
    assert state.review_in_flight_iteration == 0
    assert _reviewer_notices(workflow)


def test_resume_with_a_missing_snapshot_ref_skips_the_rerun(tmp_path, repo):
    """A pruned snapshot ref degrades to a warning, not a crash."""
    workspace = _make_workspace(tmp_path, repo)
    workflow = workflow_module.AgentTeamWorkflow(
        workspace=workspace, num_iterations=2, concurrent_review=True
    )
    workflow._run_reviewer = lambda *a, **k: pytest.fail("must not re-review")
    workflow._checkpoint = lambda state: None
    state = workflow_module.WorkflowState(
        task_path=str(workspace / "task.yaml"),
        num_iterations=2,
        review_in_flight_iteration=1,
        review_snapshot_repo=str(repo),
        review_snapshot_ref="refs/agent-flow/review/iter-999",
    )
    try:
        workflow._resume_in_flight_review(state, None)
    finally:
        workflow.close()

    assert state.review_in_flight_iteration == 0


# ------------------------------------------------------------ shared files


def test_reviewer_status_file_is_per_iteration_in_concurrent_mode(tmp_path, repo):
    """Coder and Reviewer never write the same status file concurrently."""
    workspace = _make_workspace(tmp_path, repo)
    workflow = workflow_module.AgentTeamWorkflow(
        workspace=workspace, num_iterations=2, concurrent_review=True
    )
    try:
        assert workflow._reviewer_status_ctx is not workflow._status_ctx
        assert workflow._reviewer_progress_ctx is not workflow._progress_ctx
        assert workflow._review_status_path(3).name == "status-review-3.md"
    finally:
        workflow.close()

    sequential = workflow_module.AgentTeamWorkflow(
        workspace=_make_workspace(_mkdir(tmp_path / "b"), repo), num_iterations=2
    )
    try:
        # Sequential mode keeps the historical single shared context.
        assert sequential._reviewer_status_ctx is sequential._status_ctx
        assert sequential._reviewer_progress_ctx is sequential._progress_ctx
    finally:
        sequential.close()


def test_reviewer_reset_keeps_its_own_status_and_progress_contexts(tmp_path, repo):
    """Recycling the Reviewer session must not re-point it at the Coder's files.

    ``_reset_reviewer`` fires every ``reviewer_context_reset_interval``
    iterations, so rebuilding it from the shared tool lists made every
    later review overwrite the Coder's status.md.
    """
    workspace = _make_workspace(tmp_path, repo)
    workflow = workflow_module.AgentTeamWorkflow(
        workspace=workspace, num_iterations=4, concurrent_review=True
    )
    try:
        workflow._reset_reviewer()
        tools = list(workflow.reviewer.config.backend.tools or [])
        # The rebuilt Reviewer must still hold its own contexts, so
        # ``_run_reviewer``'s per-iteration re-pointing reaches its tools.
        workflow._reviewer_status_ctx.path = workflow._review_status_path(2)
        workflow._reviewer_progress_ctx.current_iteration = 2
        update = next(t for t in tools if t.name == "update_status")
        asyncio.run(update.handler({"content": "review 2"}))
        assert (workspace / "status-review-2.md").read_text(encoding="utf-8") == "review 2"
        assert (workspace / "status.md").read_text(encoding="utf-8") == ""

        append = next(t for t in tools if t.name == "append_reviewer_progress")
        asyncio.run(append.handler({"summary": "s", "decision": "REJECT"}))
        entry = progress_module.latest_entry(workflow.progress_path, "reviewer")
        assert entry["iteration"] == 2
    finally:
        workflow.close()


def test_snapshot_repo_prefers_explicit_key_then_trtllm_path(tmp_path, repo):
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump({"trtllm_repo_path": str(repo)}), encoding="utf-8")
    assert cr.resolve_snapshot_repo(task) == repo

    task.write_text(
        yaml.safe_dump({"trtllm_repo_path": "/nope", "review_snapshot_repo": str(repo)}),
        encoding="utf-8",
    )
    assert cr.resolve_snapshot_repo(task) == repo
    assert cr.resolve_snapshot_repo(task, override=tmp_path) == tmp_path

    task.write_text("description: nothing\n", encoding="utf-8")
    assert cr.resolve_snapshot_repo(task) is None


def test_snapshot_rejects_a_non_repo(tmp_path):
    with pytest.raises(cr.SnapshotError, match="does not exist"):
        cr.snapshot_repo(tmp_path / "missing", 1)


def test_coder_addendum_names_the_configured_worktree_paths(repo):
    snapshot = cr.ReviewSnapshot(iteration=2, repo=repo, commit="abc123", ref="refs/x")
    default = cr.coder_addendum(snapshot)
    assert "/../worktrees" in default

    configured = cr.coder_addendum(
        snapshot, worktrees_dir="/wt", reservations="/wt/RESERVATIONS.md"
    )
    assert "/wt" in configured and "RESERVATIONS.md" in configured


# --- in-place mode (`review_checkout`) -------------------------------------


def _worktree_of(repo: Path, tmp_path: Path, name: str = "wt") -> Path:
    wt = tmp_path / name
    _git(repo, "worktree", "add", "--detach", "-q", str(wt), "HEAD")
    return wt


def test_snapshot_repo_allow_dirty_records_uncommitted_entries(repo):
    (repo / "file.txt").write_text("v2\n", encoding="utf-8")
    with pytest.raises(cr.SnapshotError):
        cr.snapshot_repo(repo, 1)
    snapshot = cr.snapshot_repo(repo, 1, allow_dirty=True)
    assert snapshot.commit == _git(repo, "rev-parse", "HEAD")
    assert [d.strip() for d in snapshot.dirty] == ["M file.txt"]


def test_checkout_snapshot_detaches_the_reviewer_worktree(repo, tmp_path):
    wt = _worktree_of(repo, tmp_path)
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "file.txt").write_text("v2\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "v2")
    snapshot = cr.snapshot_repo(repo, 3)
    cr.checkout_snapshot(wt, snapshot)
    assert _git(wt, "rev-parse", "HEAD") == snapshot.commit != first
    assert (wt / "file.txt").read_text() == "v2\n"
    # coder keeps moving; the reviewer tree does not
    (repo / "file.txt").write_text("v3\n", encoding="utf-8")
    assert (wt / "file.txt").read_text() == "v2\n"


def test_checkout_snapshot_refuses_bad_checkouts(repo, tmp_path):
    snapshot = cr.snapshot_repo(repo, 1)
    with pytest.raises(cr.SnapshotError, match="does not exist"):
        cr.checkout_snapshot(tmp_path / "missing", snapshot)
    with pytest.raises(cr.SnapshotError, match="snapshot repo itself"):
        cr.checkout_snapshot(repo, snapshot)
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    with pytest.raises(cr.SnapshotError, match="not a worktree"):
        cr.checkout_snapshot(other, snapshot)
    wt = _worktree_of(repo, tmp_path)
    (wt / "file.txt").write_text("edited\n", encoding="utf-8")
    with pytest.raises(cr.SnapshotError, match="tracked modifications"):
        cr.checkout_snapshot(wt, snapshot)


def test_in_place_addenda_name_the_reviewer_checkout(repo):
    snapshot = cr.ReviewSnapshot(
        iteration=2, repo=repo, commit="abc123", ref="refs/x", dirty=(" M a.py",)
    )
    reviewer = cr.reviewer_addendum(snapshot, review_checkout="/wt/wt-2")
    assert "/wt/wt-2" in reviewer and "IN PLACE" in reviewer
    assert "1 uncommitted entries" in reviewer and "a.py" in reviewer
    assert "uncommitted" not in cr.reviewer_addendum(
        cr.ReviewSnapshot(iteration=2, repo=repo, commit="abc123", ref="refs/x"),
        review_checkout="/wt/wt-2",
    )
    coder = cr.coder_addendum(snapshot, review_checkout="/wt/wt-2")
    assert "in-place mode" in coder and "NOT frozen" in coder
    assert "worktree under" not in coder
    assert "Commit everything before you end your turn" in coder
    notice = cr.verdict_notice(2, "REJECT", Path("/w/status-review-2.md"), in_place=True)
    assert "fold your worktree" not in notice and "REJECT" in notice
