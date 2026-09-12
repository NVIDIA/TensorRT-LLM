from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _git(repo: Path, *args: str) -> None:
    """Run a git command in ``repo``, raising with captured output on failure."""
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    """Create an initialized git repo with one commit and an explicit identity.

    Shared by the git-worktree provider tests (``tests/test_git_worktree.py``).
    A clean CI/sandbox environment has no global ``user.name``/``user.email``,
    so ``git commit`` would abort. This fixture pins a local identity (and
    disables commit signing defensively) before creating the initial commit, so
    the repo is self-contained and deterministic.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "README.md").write_text("init\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    return repo
