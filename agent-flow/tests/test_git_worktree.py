"""Tests for the opt-in ``git worktree`` isolation provider.

These exercise :mod:`agent_flow.git_worktree` — the concrete, git-backed
:class:`~agent_flow.orchestration.isolation.IsolationProvider` that lives
*outside* the orchestration core so the DAG engine stays git-free. They run
against a real temp git repo (the shared ``tmp_git_repo`` fixture):

Worktree mechanics
    ``acquire`` creates a worktree on branch ``node/<id>`` seeded from ``HEAD``;
    a file written inside the worktree does not leak into the main checkout;
    ``release`` removes only the worktree directory and keeps the branch (the
    node's durable output), while ``reclaim`` deletes the branch and is
    idempotent; ``acquire`` reuses an already-existing worktree/branch instead
    of raising; ``isolation="shared"`` returns ``repo_path`` with a no-op
    ``release``; ids with path separators map to a filesystem-safe directory
    while keeping a valid branch ref; and a failing git call raises
    ``IsolationError`` carrying git's stderr.

Merge (``prepare``) mechanics
    two dependency branches touching **different** files merge cleanly (both
    files land, ``conflicted_paths`` empty); two touching the **same** lines
    conflict (markers left in place, merge not aborted); a missing dependency
    branch is a genuine error and raises ``IsolationError``; and ``prepare``
    acquires the merge node's worktree itself when it was not acquired yet.

End-to-end
    the :class:`NodeScheduler` acquires a real worktree around ``run_node`` and
    releases it afterward; and a real fan-out (``g1``, ``g2``) → merge (``gm``)
    graph merges cleanly because the dependency branches survive ``release``
    until ``gm`` consumes them, then every branch is reclaimed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import anyio
import pytest

from agent_flow.git_worktree import GitWorktreeIsolation, MergeReport, node_id_to_slug
from agent_flow.orchestration.graph import ExecutionGraph, Node, NodeState
from agent_flow.orchestration.isolation import IsolationError, IsolationProvider
from agent_flow.orchestration.scheduler import GraphResult, NodeOutcome, NodeScheduler


def _git(repo: Path, *args: str) -> str:
    """Run a git command in ``repo``, returning stdout and raising on failure."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _current_branch(worktree: Path) -> str:
    """Return the abbreviated branch name checked out in ``worktree``."""
    return _git(worktree, "rev-parse", "--abbrev-ref", "HEAD").strip()


def _rev(repo: Path, ref: str = "HEAD") -> str:
    """Return the full commit hash of ``ref`` in ``repo``."""
    return _git(repo, "rev-parse", ref).strip()


def _base_branch(repo: Path) -> str:
    """Name of the branch currently checked out in ``repo`` (the base branch)."""
    return _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()


def _branch_exists(repo: Path, branch: str) -> bool:
    """Whether ``branch`` currently exists in ``repo``."""
    return _git(repo, "branch", "--list", branch).strip() != ""


def _make_dep_branch(repo: Path, branch: str, filename: str, content: str) -> None:
    """Create ``branch`` off the base branch with a single file commit.

    Branches are created from the same repo (and thus share its ``.git/config``
    identity), then the base branch is restored so the next branch also forks
    from the base commit — giving two siblings that diverge from a common base.
    """
    base = _base_branch(repo)
    _git(repo, "checkout", "-b", branch)
    (repo / filename).write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", f"{branch}: {filename}")
    _git(repo, "checkout", base)


def _worktree_path(iso: GitWorktreeIsolation, node: Node) -> Path:
    """Where ``iso`` places ``node``'s worktree (mirrors the provider's rule)."""
    return iso.worktrees_root / node_id_to_slug(node.id)


# --------------------------------------------------------------------------- #
# Worktree acquire/release mechanics
# --------------------------------------------------------------------------- #


def test_provider_conforms_to_protocol(tmp_git_repo, tmp_path):
    """GitWorktreeIsolation satisfies the runtime-checkable IsolationProvider."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    assert isinstance(iso, IsolationProvider)


def test_worktree_acquire_isolates_and_release_keeps_branch(tmp_git_repo, tmp_path):
    """Acquire yields an isolated checkout on node/<id>; release removes only the worktree."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="s1.g2", type="goal")

    cwd = anyio.run(iso.acquire, node)

    assert cwd.is_dir()
    assert cwd != tmp_git_repo
    assert _current_branch(cwd) == "node/s1.g2"
    # base_ref defaults to HEAD -> worktree starts at the repo's current commit.
    assert _rev(cwd) == _rev(tmp_git_repo)

    # The whole point: work in the worktree stays out of the main working tree.
    (cwd / "only_here.txt").write_text("secret\n")
    assert not (tmp_git_repo / "only_here.txt").exists()

    anyio.run(iso.release, node)
    # release removes the worktree directory...
    assert not cwd.exists()
    # ...but keeps the branch — the node's durable output survives until reclaimed.
    assert _branch_exists(tmp_git_repo, "node/s1.g2")

    # reclaim frees the durable output.
    anyio.run(iso.reclaim, node)
    assert not _branch_exists(tmp_git_repo, "node/s1.g2")


def test_worktree_dir_is_filesystem_safe_for_slashed_id(tmp_git_repo, tmp_path):
    """A node id with a path separator maps to a safe dir but a valid branch ref."""
    root = tmp_path / "wt"
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=root)
    node = Node(id="g1/sub", type="goal")

    cwd = anyio.run(iso.acquire, node)

    # Leaf directory name carries no separator...
    assert "/" not in cwd.name
    assert cwd.parent == root
    assert cwd.name == node_id_to_slug("g1/sub")
    # ...while the branch keeps the raw id (slashes are legal in git refs).
    assert _current_branch(cwd) == "node/g1/sub"

    anyio.run(iso.release, node)
    assert not cwd.exists()


def test_acquire_absolutizes_relative_worktrees_root(tmp_git_repo, tmp_path, monkeypatch):
    """A RELATIVE worktrees_root must still yield a real, absolute worktree dir.

    Regression (concurrent run crash ``CLIConnectionError: Working directory
    does not exist``): ``git -C <repo> worktree add <relative>`` resolves the
    relative path against the *repo*, creating the worktree under the repo — but
    the scheduler hands ``acquire``'s return value to the agent as its ``cwd``,
    resolved against the *process* cwd. When the two bases differ the agent's
    cwd does not exist. The provider must absolutize ``worktrees_root`` so the
    git-add location and the returned cwd always agree.
    """
    # Process cwd deliberately != repo_path, mirroring the real run: the agent
    # process runs from the agent-flow root while repo_path is the trtllm repo.
    monkeypatch.chdir(tmp_path)
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=Path("rel_wt"))
    node = Node(id="s3.g1", type="goal")

    cwd = anyio.run(iso.acquire, node)

    # acquire's return value is exactly what the agent is told to run in.
    assert cwd.is_absolute(), f"worktree path must be absolute, got {cwd!r}"
    assert cwd.is_dir(), f"worktree dir must exist at the returned path, got {cwd!r}"
    # It must NOT have leaked under the repo (the buggy relative-to-repo location).
    assert tmp_git_repo not in cwd.parents

    anyio.run(iso.release, node)
    anyio.run(iso.reclaim, node)


def test_shared_isolation_returns_repo_and_release_is_noop(tmp_git_repo, tmp_path):
    """isolation='shared' runs in the parent workspace; release changes nothing."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="s1", type="goal", isolation="shared")

    cwd = anyio.run(iso.acquire, node)
    assert cwd == tmp_git_repo

    # No worktree dir is created for shared nodes.
    assert not (tmp_path / "wt").exists()

    anyio.run(iso.release, node)
    assert tmp_git_repo.is_dir()
    assert (tmp_git_repo / "README.md").exists()


def test_acquire_failure_raises_isolation_error_with_stderr(tmp_git_repo, tmp_path):
    """A genuine git failure (a bad base_ref) surfaces as IsolationError with stderr."""
    iso = GitWorktreeIsolation(
        repo_path=tmp_git_repo,
        worktrees_root=tmp_path / "wt",
        base_ref="no-such-ref",
    )
    node = Node(id="bad", type="goal")

    with pytest.raises(IsolationError) as excinfo:
        anyio.run(iso.acquire, node)  # base_ref does not resolve -> git fails

    message = str(excinfo.value)
    assert "git" in message
    assert message.strip() != ""


def test_release_keeps_branch_and_reclaim_deletes_it(tmp_git_repo, tmp_path):
    """Release keeps the durable branch; reclaim deletes it (and prunes)."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="n1", type="goal")

    cwd = anyio.run(iso.acquire, node)
    assert _branch_exists(tmp_git_repo, "node/n1")

    anyio.run(iso.release, node)
    assert not cwd.exists()  # worktree gone
    assert _branch_exists(tmp_git_repo, "node/n1")  # branch survives

    anyio.run(iso.reclaim, node)
    assert not _branch_exists(tmp_git_repo, "node/n1")  # now freed


def test_reclaim_is_idempotent(tmp_git_repo, tmp_path):
    """A second reclaim on an already-gone branch/worktree does not raise."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="n2", type="goal")

    anyio.run(iso.acquire, node)
    anyio.run(iso.release, node)
    anyio.run(iso.reclaim, node)
    # Reclaiming again must be a tolerant no-op, not an error.
    anyio.run(iso.reclaim, node)
    assert not _branch_exists(tmp_git_repo, "node/n2")


def test_reclaim_of_never_acquired_node_is_tolerant(tmp_git_repo, tmp_path):
    """Reclaiming a node whose branch never existed (e.g. a BLOCKED node) is a no-op."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="ghost", type="goal")

    anyio.run(iso.reclaim, node)  # must not raise
    assert not _branch_exists(tmp_git_repo, "node/ghost")


def test_reclaim_shared_node_is_a_noop(tmp_git_repo, tmp_path):
    """A shared node has no durable branch; reclaim leaves the repo untouched."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="s1", type="goal", isolation="shared")

    anyio.run(iso.reclaim, node)  # must not raise
    assert (tmp_git_repo / "README.md").exists()


def test_acquire_reuses_existing_worktree(tmp_git_repo, tmp_path):
    """Re-acquiring the same node reuses its worktree instead of raising."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="dup", type="goal")

    cwd1 = anyio.run(iso.acquire, node)
    (cwd1 / "marker.txt").write_text("x\n")

    cwd2 = anyio.run(iso.acquire, node)  # branch + path already exist -> reuse
    assert cwd2 == cwd1
    assert (cwd2 / "marker.txt").read_text() == "x\n"

    anyio.run(iso.release, node)
    anyio.run(iso.reclaim, node)


def test_acquire_reuses_existing_branch_after_release(tmp_git_repo, tmp_path):
    """After release removes the worktree, re-acquire checks the surviving branch back out."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="resume", type="goal")

    cwd = anyio.run(iso.acquire, node)
    (cwd / "work.txt").write_text("done\n")
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-m", "work")

    anyio.run(iso.release, node)  # worktree removed, branch node/resume kept
    assert not cwd.exists()
    assert _branch_exists(tmp_git_repo, "node/resume")

    cwd2 = anyio.run(iso.acquire, node)  # branch exists, no worktree dir -> re-checkout
    assert cwd2 == cwd
    assert _current_branch(cwd2) == "node/resume"
    # The branch's committed content is restored into the fresh worktree.
    assert (cwd2 / "work.txt").read_text() == "done\n"

    anyio.run(iso.release, node)
    anyio.run(iso.reclaim, node)


async def test_clean_tears_down_worktrees_branches_and_resets_repo(tmp_git_repo, tmp_path):
    """``clean`` removes every worktree + node branch and restores a pristine HEAD.

    The ``--clean`` "start over" teardown: after a concurrent run leaves per-node
    worktrees/branches AND writes land in the main checkout (a shared/merge node,
    or a linear run), ``clean`` must leave the repo at ``HEAD`` with no worktrees,
    no ``node/*`` branches, and no leftover working-tree changes.
    """
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    n1, n2 = Node(id="g1", type="goal"), Node(id="g2", type="goal")
    await iso.acquire(n1)
    await iso.acquire(n2)
    # Simulate work that reached the MAIN checkout (linear run / shared node):
    (tmp_git_repo / "modeling_new.py").write_text("wip\n")  # untracked source
    (tmp_git_repo / "README.md").write_text("locally modified\n")  # tracked edit
    assert _branch_exists(tmp_git_repo, "node/g1")
    assert (tmp_path / "wt" / "g1").is_dir()

    await iso.clean()

    # Worktrees gone (dirs + git registration), node/* branches gone.
    assert not (tmp_path / "wt" / "g1").exists()
    assert not (tmp_path / "wt" / "g2").exists()
    assert _git(tmp_git_repo, "branch", "--list", "node/*").strip() == ""
    assert _git(tmp_git_repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    # Main working tree is pristine at HEAD: tracked edit reverted, untracked removed.
    assert _git(tmp_git_repo, "status", "--porcelain").strip() == ""
    assert (tmp_git_repo / "README.md").read_text() == "init\n"
    assert not (tmp_git_repo / "modeling_new.py").exists()
    # The worktrees root directory itself is removed.
    assert not (tmp_path / "wt").exists()


async def test_clean_with_no_worktrees_still_resets_working_tree(tmp_git_repo, tmp_path):
    """A linear run has no worktrees/branches; ``clean`` still resets the checkout."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    (tmp_git_repo / "stray.py").write_text("x\n")  # untracked
    (tmp_git_repo / "README.md").write_text("changed\n")  # tracked edit

    await iso.clean()  # no worktrees to remove — must be a tolerant reset

    assert _git(tmp_git_repo, "status", "--porcelain").strip() == ""
    assert (tmp_git_repo / "README.md").read_text() == "init\n"
    assert not (tmp_git_repo / "stray.py").exists()


def test_node_id_to_slug_sanitizes_separators():
    """The slug helper strips path separators so ids map to one leaf dir."""
    assert node_id_to_slug("s1.g2") == "s1.g2"
    assert "/" not in node_id_to_slug("a/b/c")
    assert "\\" not in node_id_to_slug("a\\b")


def test_git_and_workspace_slug_rules_stay_equal():
    """Pin the git-worktree slug rule equal to the sub-workspace slug rule.

    ``git_worktree.node_id_to_slug`` (a node's worktree dir) and
    ``node_workspace.node_dir_slug`` (its private sub-workspace dir) are kept
    byte-identical on purpose but deliberately not merged (the layering keeps
    the workflow layer git-free). A future edit that diverges one would silently
    point a node's worktree name away from its sub-workspace name; this guards
    the equality without coupling the two modules.

    ``node_workspace`` is part of the agent-team concurrent integration, which
    this repo has not adopted yet — the DAG engine is currently driven by
    ``perf_optimize``, which owns its own per-item directories. Skip rather than
    fail while that module is absent, so the guard arms itself automatically if
    the agent-team integration is ported later.
    """
    node_workspace = pytest.importorskip(
        "agent_flow.workflows.agent_team.node_workspace",
        reason="agent-team concurrent integration is not ported to this repo",
    )
    node_dir_slug = node_workspace.node_dir_slug

    for node_id in ("a/b", "a\\b", "s1.g2", "plain"):
        assert node_id_to_slug(node_id) == node_dir_slug(node_id)


# --------------------------------------------------------------------------- #
# Commit mechanics (persist a node's work to its branch before release)
# --------------------------------------------------------------------------- #


def test_commit_persists_worktree_changes_to_node_branch(tmp_git_repo, tmp_path):
    """``commit`` stages+commits the worktree's changes onto the node branch.

    The concurrent code-loss defect: the node loop wrote files into the worktree
    but nothing committed them, so ``release`` (which removes the worktree)
    discarded the work and the branch stayed at base. ``commit`` closes that gap
    so the node's output survives release and a dependent can merge it.
    """
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="c1", type="impl")

    cwd = anyio.run(iso.acquire, node)
    # The coder writes files but does NOT commit them.
    (cwd / "impl.py").write_text("x = 1\n")

    sha = anyio.run(iso.commit, node, cwd)
    assert sha  # a commit was recorded on node/c1

    anyio.run(iso.release, node)  # remove the worktree, keep the branch
    assert not cwd.exists()
    assert _branch_exists(tmp_git_repo, "node/c1")

    # Re-acquiring the node restores its committed content from the branch.
    cwd2 = anyio.run(iso.acquire, node)
    assert (cwd2 / "impl.py").read_text() == "x = 1\n"

    anyio.run(iso.release, node)
    anyio.run(iso.reclaim, node)


def test_commit_is_noop_when_worktree_has_no_changes(tmp_git_repo, tmp_path):
    """``commit`` returns ``None`` and creates no commit when nothing changed."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="c2", type="impl")

    cwd = anyio.run(iso.acquire, node)
    head_before = _rev(cwd)

    result = anyio.run(iso.commit, node, cwd)  # nothing was written

    assert result is None
    assert _rev(cwd) == head_before  # no empty commit

    anyio.run(iso.release, node)
    anyio.run(iso.reclaim, node)


def test_commit_shared_node_commits_to_base_branch(tmp_git_repo, tmp_path):
    """A shared node commits its working-tree changes to the base branch.

    A merge node runs shared (directly in the repo, not a worktree). Committing
    its integration to the base branch advances HEAD, so a later stage's fresh
    worktree (forked from HEAD) inherits it — the cross-stage code handoff. This
    is why the merge node's own edits must not be left uncommitted.
    """
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="s1.g7", type="merge", kind="merge", isolation="shared")

    cwd = anyio.run(iso.acquire, node)
    assert cwd == tmp_git_repo
    head_before = _rev(tmp_git_repo)
    (tmp_git_repo / "integrated.py").write_text("z = 3\n")  # the merge node's integration edit

    sha = anyio.run(iso.commit, node, cwd)

    assert sha  # a commit was made on the base branch
    assert _rev(tmp_git_repo) != head_before  # base branch HEAD advanced
    # The change is now committed on the base branch (a fresh worktree inherits it).
    assert "integrated.py" in _git(tmp_git_repo, "show", "--name-only", "--format=", "HEAD")


def test_commit_shared_node_noop_when_clean(tmp_git_repo, tmp_path):
    """A shared node with no changes commits nothing (no empty commit)."""
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    node = Node(id="s1.g7", type="merge", kind="merge", isolation="shared")

    cwd = anyio.run(iso.acquire, node)
    head_before = _rev(tmp_git_repo)

    assert anyio.run(iso.commit, node, cwd) is None  # nothing to commit
    assert _rev(tmp_git_repo) == head_before


# --------------------------------------------------------------------------- #
# Merge (prepare) mechanics
# --------------------------------------------------------------------------- #


def test_prepare_different_files_merges_clean(tmp_git_repo, tmp_path):
    """Deps touching different files merge cleanly; both files land, no conflicts."""
    # Two dependency branches diverging from the base, each adding its own file.
    _make_dep_branch(tmp_git_repo, "node/dep_a", "file_a.txt", "alpha\n")
    _make_dep_branch(tmp_git_repo, "node/dep_b", "file_b.txt", "beta\n")

    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    merge_node = Node(id="merge", type="merge", kind="merge", depends_on=("dep_a", "dep_b"))
    dep_nodes = [Node(id="dep_a", type="impl"), Node(id="dep_b", type="impl")]
    worktree = _worktree_path(iso, merge_node)

    report = anyio.run(iso.prepare, merge_node, dep_nodes, worktree)

    assert isinstance(report, MergeReport)
    assert report.conflicted_paths == []
    # Both dependency branches were recorded as cleanly merged, in order.
    assert report.merged == ["node/dep_a", "node/dep_b"]

    # The merge node's worktree now carries both dependencies' contributions.
    assert (worktree / "file_a.txt").read_text() == "alpha\n"
    assert (worktree / "file_b.txt").read_text() == "beta\n"


def test_prepare_same_lines_records_conflict_without_aborting(tmp_git_repo, tmp_path):
    """Deps editing the same lines conflict; markers stay, merge is not aborted."""
    # Both branches rewrite the same file (README.md) with incompatible content.
    _make_dep_branch(tmp_git_repo, "node/dep_a", "README.md", "alpha\n")
    _make_dep_branch(tmp_git_repo, "node/dep_b", "README.md", "beta\n")

    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    merge_node = Node(id="merge", type="merge", kind="merge", depends_on=("dep_a", "dep_b"))
    dep_nodes = [Node(id="dep_a", type="impl"), Node(id="dep_b", type="impl")]
    worktree = _worktree_path(iso, merge_node)

    report = anyio.run(iso.prepare, merge_node, dep_nodes, worktree)

    # The first dep merged clean (fast-forward); the second conflicts.
    assert report.merged == ["node/dep_a"]
    assert report.conflicted_paths == ["README.md"]

    # Conflict markers are left in place for a later resolver — not aborted.
    conflicted = (worktree / "README.md").read_text()
    assert "<<<<<<<" in conflicted
    assert ">>>>>>>" in conflicted
    assert "alpha" in conflicted and "beta" in conflicted
    # Git still reports the file as unmerged (the merge remains in progress).
    unmerged = _git(worktree, "diff", "--name-only", "--diff-filter=U").split()
    assert unmerged == ["README.md"]


def test_prepare_skips_missing_dependency_branch(tmp_git_repo, tmp_path):
    """A dependency whose branch no longer exists is SKIPPED, not an error.

    In the DAG lifecycle a dependency branch is absent only because it was
    reclaimed after the dependency went DONE — its work is already integrated
    into the base checkout. A dependent re-running after a replan reset therefore
    has nothing to merge for it, so ``prepare`` records the branch in ``skipped``
    and proceeds rather than crashing on the (reclaimed) branch.
    """
    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    merge_node = Node(id="merge", type="merge", kind="merge", depends_on=("ghost",))
    # No branch ``node/ghost`` exists (never created, or reclaimed after DONE).
    dep_nodes = [Node(id="ghost", type="impl")]
    worktree = _worktree_path(iso, merge_node)

    report = anyio.run(iso.prepare, merge_node, dep_nodes, worktree)

    assert report.skipped == ["node/ghost"]
    assert report.merged == []
    assert report.conflicted_paths == []


def test_prepare_acquires_worktree_when_not_already_acquired(tmp_git_repo, tmp_path):
    """``prepare`` creates the merge node's worktree itself when absent."""
    _make_dep_branch(tmp_git_repo, "node/dep_a", "file_a.txt", "alpha\n")

    iso = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    merge_node = Node(id="merge", type="merge", kind="merge", depends_on=("dep_a",))

    worktree = _worktree_path(iso, merge_node)
    assert not worktree.exists()  # nothing acquired yet

    report = anyio.run(iso.prepare, merge_node, [Node(id="dep_a", type="impl")], worktree)

    assert worktree.is_dir()  # prepare acquired it
    assert report.merged == ["node/dep_a"]
    assert (worktree / "file_a.txt").read_text() == "alpha\n"


# --------------------------------------------------------------------------- #
# End-to-end: scheduler over a real worktree
# --------------------------------------------------------------------------- #


async def test_scheduler_runs_inside_real_worktree_and_releases(tmp_git_repo, tmp_path):
    """The scheduler acquires a real worktree around ``run_node`` and releases it."""
    graph = ExecutionGraph(nodes=(Node(id="w1", type="impl"),))
    isolation = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    seen: dict[str, Path] = {}

    async def run_node(node, path):
        seen["path"] = path
        (path / "made.txt").write_text("hi\n")
        return NodeOutcome(NodeState.DONE)

    scheduler = NodeScheduler(graph, run_node, isolation)
    result = await scheduler.run()

    assert result.succeeded
    assert isinstance(result, GraphResult)
    # Ran in an isolated worktree, not the main checkout, and got cleaned up.
    assert seen["path"] != tmp_git_repo
    assert not seen["path"].exists()
    assert not (tmp_git_repo / "made.txt").exists()


async def test_scheduler_fanout_merge_survives_until_consumed(tmp_git_repo, tmp_path):
    """A real fan-out → merge run: dep branches survive release so ``gm`` can merge them.

    ``g1`` and ``g2`` each commit their own file; ``gm`` depends on both and is a
    ``merge`` node, so the scheduler ``prepare``s it by git-merging the dependency
    branches into its worktree. This only works if ``g1``/``g2``'s branches were
    *kept* through their ``release`` — the whole point of dependency-lifetime
    reclamation. After the run, every per-node branch has been reclaimed.
    """
    graph = ExecutionGraph(
        nodes=(
            Node(id="g1", type="impl"),
            Node(id="g2", type="impl"),
            Node(id="gm", type="merge", kind="merge", depends_on=("g1", "g2")),
        )
    )
    isolation = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")

    async def run_node(node, path):
        if node.kind != "merge":
            # Give each dependency branch content the merge node can fold in.
            (path / f"{node.id}.txt").write_text(f"{node.id}\n")
            _git(path, "add", "-A")
            _git(path, "commit", "-m", f"{node.id} work")
        else:
            # ``prepare`` already merged the dependency branches into this worktree.
            assert (path / "g1.txt").read_text() == "g1\n"
            assert (path / "g2.txt").read_text() == "g2\n"
        return NodeOutcome(NodeState.DONE)

    scheduler = NodeScheduler(graph, run_node, isolation)
    result = await scheduler.run()

    assert result.succeeded
    # ``gm``'s prepare cleanly merged BOTH dependency branches — proof they
    # survived their own ``release`` until the merge node consumed them.
    report = result.prepare_results["gm"]
    assert isinstance(report, MergeReport)
    assert report.merged == ["node/g1", "node/g2"]
    assert report.conflicted_paths == []

    # Every per-node branch is reclaimed once all dependents are terminal.
    listed = _git(tmp_git_repo, "branch", "--list", "node/*").strip()
    assert listed == ""
    # No worktree directories linger either.
    assert not (tmp_path / "wt" / "g1").exists()
    assert not (tmp_path / "wt" / "g2").exists()
    assert not (tmp_path / "wt" / "gm").exists()


async def test_impl_node_inherits_dependency_without_manual_commit(tmp_git_repo, tmp_path):
    """The concurrent code-loss regression, end to end.

    An ``impl`` node that depends on a sibling ``impl`` node receives the
    sibling's code even though ``run_node`` never commits — the scheduler
    auto-commits each node's worktree on its terminal state (so the dependency's
    branch is non-empty) and merges dependency branches into *any* dependent (so
    an impl node, not only a merge node, inherits them). This is exactly the
    s1.g5 assembly scenario that failed: modules built but were never committed,
    so the assembly node started from base with none of their code.
    """
    graph = ExecutionGraph(
        nodes=(
            Node(id="mod", type="impl"),
            Node(id="assemble", type="impl", depends_on=("mod",)),
        )
    )
    isolation = GitWorktreeIsolation(repo_path=tmp_git_repo, worktrees_root=tmp_path / "wt")
    seen: dict[str, object] = {}

    async def run_node(node, path):
        if node.id == "mod":
            # A module node writes source but does NOT commit (the real defect).
            (path / "module.py").write_text("VALUE = 42\n")
        else:
            # The dependent must find the dependency's file in its worktree.
            present = (path / "module.py").exists()
            seen["present"] = present
            seen["text"] = (path / "module.py").read_text() if present else None
        return NodeOutcome(NodeState.DONE)

    result = await NodeScheduler(graph, run_node, isolation).run()

    assert result.succeeded
    # The assembly node inherited the (never-manually-committed) module code.
    assert seen["present"] is True
    assert seen["text"] == "VALUE = 42\n"
    # It arrived via prepare merging the dependency branch into an IMPL node.
    report = result.prepare_results["assemble"]
    assert isinstance(report, MergeReport)
    assert report.merged == ["node/mod"]
    # Every per-node branch is reclaimed once all dependents are terminal.
    assert _git(tmp_git_repo, "branch", "--list", "node/*").strip() == ""
