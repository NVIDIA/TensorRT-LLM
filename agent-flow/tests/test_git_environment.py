# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The suite never runs git against the repository its caller's environment names."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .conftest import GIT_LOCATION_VARIABLES, ROOT

INNER = "test_a_fixtures_repository_is_the_one_it_made"


def test_a_fixtures_repository_is_the_one_it_made(tmp_git_repo: Path) -> None:
    """What the outer test runs with ``GIT_DIR`` set: the fixture's commands stayed in its repository."""
    assert not [name for name in GIT_LOCATION_VARIABLES if name in os.environ]
    said = subprocess.run(
        ["git", "config", "user.name"], cwd=tmp_git_repo, capture_output=True, text=True
    )
    assert said.stdout.strip() == "Test"
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=tmp_git_repo, capture_output=True, text=True
    )
    assert Path(top.stdout.strip()).resolve() == tmp_git_repo.resolve()


def test_a_callers_git_dir_never_receives_a_fixtures_git_commands(tmp_path: Path) -> None:
    """As under ``git rebase --exec``: ``GIT_DIR`` and ``GIT_WORK_TREE`` name the caller's repository."""
    callers = tmp_path / "callers"
    subprocess.run(["git", "init", "-q", str(callers)], check=True, capture_output=True)
    config = callers / ".git" / "config"
    before = config.read_text()
    env = {
        **os.environ,
        "GIT_DIR": str(callers / ".git"),
        "GIT_WORK_TREE": str(callers),
        "GIT_INDEX_FILE": str(callers / ".git" / "index"),
        "GIT_PREFIX": "",
    }
    ran = subprocess.run(
        [
            sys.executable, "-m", "pytest", f"{Path(__file__)}::{INNER}",
            "-q", "-p", "no:cacheprovider", f"--basetemp={tmp_path / 'inner'}",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert ran.returncode == 0, ran.stdout + ran.stderr
    # The fixture ran `git init`, `git config user.name Test`, `git add`, `git commit`:
    # none of it reached the repository the environment named.
    assert config.read_text() == before
    assert "Test" not in config.read_text()
    assert not (callers / "README.md").exists()
    log = subprocess.run(["git", "-C", str(callers), "log", "--oneline"], capture_output=True)
    assert log.returncode != 0, "the caller's repository has no commit"
