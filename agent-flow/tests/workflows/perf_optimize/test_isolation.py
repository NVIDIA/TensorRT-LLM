"""Tests for perf-optimize's worktree isolation adapter.

The one behavior this module changes from the stock provider is load-bearing:
the Integrator runs *after* ``scheduler.run()`` returns and cherry-picks each
candidate's commit, so a candidate branch must outlive the scheduler's own
teardown. Upstream would have deleted it — a dependent-less node is reclaimed
the moment it goes terminal, and ``run`` sweeps whatever is left on return.
"""

from __future__ import annotations

import subprocess

import anyio
import pytest

from agent_flow.git_worktree import GitWorktreeIsolation
from agent_flow.orchestration import Node
from agent_flow.workflows.perf_optimize.isolation import CandidateWorktreeIsolation, item_node_id


def _git(repo, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "src.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _provider(repo, tmp_path):
    return CandidateWorktreeIsolation(
        repo_path=repo,
        worktrees_root=tmp_path / "worktrees" / "round_1",
        branch_prefix="perf-optimize/ws-round-1-",
        base_ref=_git(repo, "rev-parse", "HEAD"),
    )


# ------------------------------------------------------------------ node ids


@pytest.mark.parametrize(
    ("index", "item_id", "expected"),
    [
        (0, "opt-001", "item_1_opt-001"),
        (2, "opt.007", "item_3_opt.007"),
        # Roadmap ids are author-supplied: path separators and spaces would make
        # an illegal directory component, and a leading/trailing dot or dash an
        # awkward one.
        (0, "a/b c", "item_1_a-b-c"),
        (0, "...", "item_1_item"),
        (0, "", "item_1_item"),
    ],
)
def test_item_node_id_is_filesystem_and_ref_safe(index, item_id, expected):
    assert item_node_id(index, item_id) == expected


def test_item_node_id_is_length_capped():
    """A pathological roadmap id must not blow past a path component's limit."""
    node_id = item_node_id(0, "x" * 300)
    assert node_id == "item_1_" + "x" * 48


# ----------------------------------------------------------- naming agreement


def test_naming_helpers_match_what_the_provider_hands_the_node(repo, tmp_path):
    """The precomputed ledger paths must equal what ``acquire`` actually returns.

    ``_prepare_item_batch`` records ``item_worktree_path``/``item_branch`` before
    the scheduler runs; if those disagreed with the provider, the batch ledger
    would point at a directory no item ever ran in.
    """
    provider = _provider(repo, tmp_path)
    node = Node(id="item_1_opt-001", type="roadmap_item")

    acquired = anyio.run(provider.acquire, node)

    assert acquired == provider.worktree_path("item_1_opt-001")
    assert provider.branch_name("item_1_opt-001") == "perf-optimize/ws-round-1-item_1_opt-001"
    assert "perf-optimize/ws-round-1-item_1_opt-001" in _git(repo, "branch", "--list")


# --------------------------------------------------------- deferred reclaim


def test_reclaim_keeps_the_candidate_branch_for_the_out_of_graph_integrator(repo, tmp_path):
    """``reclaim`` must not delete the branch the Integrator still has to read.

    Guards the defect this subclass exists to prevent: with the stock provider
    the scheduler's end-of-run sweep frees every branch, and ``_integrate_batch``
    — which runs afterwards — would find nothing to cherry-pick.
    """
    provider = _provider(repo, tmp_path)
    node = Node(id="item_1_opt-001", type="roadmap_item")
    worktree = anyio.run(provider.acquire, node)
    (worktree / "src.py").write_text("x = 2\n", encoding="utf-8")
    anyio.run(provider.commit, node, worktree)

    anyio.run(provider.release, node)
    anyio.run(provider.reclaim, node)

    assert not worktree.exists(), "release still frees the transient worktree"
    assert "perf-optimize/ws-round-1-item_1_opt-001" in _git(repo, "branch", "--list")
    assert _git(repo, "show", "perf-optimize/ws-round-1-item_1_opt-001:src.py") == "x = 2", (
        "the candidate's commit is still reachable for integration"
    )


def test_stock_provider_would_have_deleted_that_branch(repo, tmp_path):
    """Pin the upstream behavior the subclass overrides, so a rebase can't erase it.

    If a future upstream sync made ``reclaim`` a no-op for everyone, this test
    fails and the subclass can be retired deliberately rather than silently kept.
    """
    stock = GitWorktreeIsolation(
        repo_path=repo,
        worktrees_root=tmp_path / "worktrees" / "round_1",
        branch_prefix="perf-optimize/ws-round-1-",
        base_ref=_git(repo, "rev-parse", "HEAD"),
    )
    node = Node(id="item_1_opt-001", type="roadmap_item")
    anyio.run(stock.acquire, node)
    anyio.run(stock.release, node)
    anyio.run(stock.reclaim, node)

    assert "perf-optimize/ws-round-1-item_1_opt-001" not in _git(repo, "branch", "--list")


def test_acquire_is_idempotent_across_a_resume(repo, tmp_path):
    """A resumed round re-acquires the same worktree instead of failing on it."""
    provider = _provider(repo, tmp_path)
    node = Node(id="item_1_opt-001", type="roadmap_item")
    first = anyio.run(provider.acquire, node)
    (first / "candidate.txt").write_text("in progress\n", encoding="utf-8")

    second = anyio.run(provider.acquire, node)

    assert second == first
    assert (second / "candidate.txt").exists()
