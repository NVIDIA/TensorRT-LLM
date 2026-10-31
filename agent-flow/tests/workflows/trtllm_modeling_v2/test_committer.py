# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The commit rules, audited against a real git repo.

These tests shell out to real ``git`` on purpose: the audit reads what git
wrote, and a mock of git would test the mock.
"""

from __future__ import annotations

import subprocess

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.git import CommitPolicy, audit_commits


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
