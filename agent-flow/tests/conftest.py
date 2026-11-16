# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Variables that point git at a repository or a work tree whatever the working directory
#: is. Git exports them to what it starts (`git rebase --exec`, `git bisect run`, a hook):
#: a suite started from there would run its fixtures' `git init` and `git config` against
#: the caller's repository instead of the temporary ones it makes.
GIT_LOCATION_VARIABLES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
)


def pytest_configure(config: pytest.Config) -> None:
    """Drop the caller's git location before anything of the suite can run git."""
    for name in GIT_LOCATION_VARIABLES:
        os.environ.pop(name, None)


@pytest.fixture(autouse=True)
def _no_git_location_from_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test runs git against a repository its caller's environment names."""
    for name in GIT_LOCATION_VARIABLES:
        monkeypatch.delenv(name, raising=False)


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
