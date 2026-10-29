# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The commit rules, audited against a real git repo.

These tests shell out to real ``git`` on purpose: the audit reads what git
wrote, and a mock of git would test the mock.
"""

from __future__ import annotations

import subprocess

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.git import (
    CommitIdentity,
    CommitPolicy,
    audit_commits,
    trial_commit,
)


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.invalid")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "commit.gpgsign", "false")
    (path / "seed.txt").write_text("seed\n")
    _git(path, "add", "seed.txt")
    _git(path, "commit", "-q", "-m", "seed")
    return path


@pytest.fixture
def isolated_git_env(tmp_path, monkeypatch):
    """No global or system git configuration, and no identity override.

    So a test that asks "what identity does git have" gets the repository's
    own answer, not the developer's ~/.gitconfig.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / ".gitconfig"))
    for name in (
        "MV2FLOW_GIT_NAME",
        "MV2FLOW_GIT_EMAIL",
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


def _commit_raw(repo, message, path, content="x\n"):
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(content)
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def test_audit_commits_finds_every_kind_of_violation(repo):
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    good = _commit_raw(repo, "good\n\nSigned-off-by: Test <t@example.invalid>", "good.py")
    unsigned = _commit_raw(repo, "unsigned", "unsigned.py")
    mismatched = _commit_raw(
        repo, "mismatch\n\nSigned-off-by: Operator <op@example.invalid>", "mismatch.py"
    )
    coauthored = _commit_raw(
        repo,
        "co\n\nCo-authored-by: Bot <b@example.invalid>\nSigned-off-by: Test <t@example.invalid>",
        "co.py",
    )
    (repo / "link").symlink_to("good.py")
    _git(repo, "add", "link")
    _git(repo, "commit", "-q", "-s", "-m", "link")
    link = _git(repo, "rev-parse", "HEAD").stdout.strip()
    placeholder = _commit_raw(
        repo, "ph\n\nSigned-off-by: Test <t@example.invalid>", "pkg/.mcp.json", content=""
    )

    findings = audit_commits(repo, f"{base}..HEAD")
    by_sha: dict[str, list[str]] = {}
    for finding in findings:
        by_sha.setdefault(finding.sha, []).append(finding.problem)
    assert good not in by_sha
    assert by_sha[unsigned] == ["no Signed-off-by"]
    assert "no sign-off carries the author email" in by_sha[mismatched][0]
    assert by_sha[coauthored] == ["carries Co-authored-by"]
    assert by_sha[link] == ["adds a symlink: link"]
    assert by_sha[placeholder] == ["adds a sandbox placeholder: pkg/.mcp.json"]
    assert audit_commits(repo, f"{base}..HEAD", CommitPolicy(allow_co_authors=True)) != findings
    scoped = audit_commits(repo, f"{base}..HEAD", paths=("pkg/",))
    assert {f.sha for f in scoped} == {placeholder}


def test_a_tasks_own_commits_are_the_runs_identity_inside_the_allowed_paths(repo):
    """What a task's branch must also hold before integration: who wrote it, and where."""
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _commit_raw(repo, "readme\n\nSigned-off-by: Test <t@example.invalid>", "README.md")
    _commit_raw(repo, "pkg\n\nSigned-off-by: Test <t@example.invalid>", "pkg/a.py")
    policy = CommitPolicy(
        identity=CommitIdentity("Repo User", "t@example.invalid"), allowed_paths=("pkg/",)
    )
    assert audit_commits(repo, f"{base}..HEAD", policy) == []
    problems = sorted(f.problem for f in audit_commits(repo, f"{base}..HEAD", policy, own=True))
    assert problems == [
        "README.md is outside the paths this run may commit: pkg/",
        "authored as 'Test <t@example.invalid>', not the run's 'Repo User <t@example.invalid>'",
        "authored as 'Test <t@example.invalid>', not the run's 'Repo User <t@example.invalid>'",
    ]


def test_the_trial_commit_catches_an_environment_that_rewrites_every_commit(
    tmp_path, isolated_git_env
):
    """A global hook that appends a trailer would break every agent's commit, beyond repair."""
    identity = CommitIdentity("Repo User", "repo.user@example.invalid")
    assert trial_commit(identity) == []
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook = hooks / "commit-msg"
    hook.write_text('#!/bin/sh\necho "Co-authored-by: Bot <b@example.invalid>" >> "$1"\n')
    hook.chmod(0o755)
    _git(tmp_path, "config", "--global", "core.hooksPath", str(hooks))
    assert trial_commit(identity) == ["carries Co-authored-by"]
    assert trial_commit(identity, allow_co_authors=True) == []


def test_the_cli_refuses_to_start_without_an_identity(tmp_path, isolated_git_env):
    from agent_flow.workflows.trtllm_modeling_v2.cli import _commit_identity

    repo = tmp_path / "noid"
    repo.mkdir()
    _git(repo, "init", "-q")
    with pytest.raises(SystemExit) as exc:
        _commit_identity({"trtllm_repo_path": str(repo)})
    assert exc.value.code == 2
