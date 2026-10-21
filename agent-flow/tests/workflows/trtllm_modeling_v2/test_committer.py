# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The commit helper, exercised against a real git repo.

These tests shell out to real ``git`` on purpose. The thing under test is a
shell script whose whole job is to behave correctly under concurrency, and a
mock of git would test the mock.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.committer import (
    CommitIdentity,
    CommitPolicy,
    audit_commits,
    install_committer,
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
def commit(tmp_path, repo):
    script = install_committer(tmp_path / "ws", repo)

    def run(message, *paths, timeout=60):
        return subprocess.run(
            [str(script), message, *paths], capture_output=True, text=True, timeout=timeout
        )

    return run


def _files_in(repo, ref="HEAD"):
    out = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", ref).stdout
    return sorted(line for line in out.splitlines() if line)


def test_it_commits_exactly_what_it_was_given(repo, commit):
    (repo / "a.py").write_text("a\n")
    (repo / "b.py").write_text("b\n")
    result = commit("add a", "a.py")
    assert result.returncode == 0, result.stderr
    assert _files_in(repo) == ["a.py"]
    assert (repo / "b.py").exists()  # untouched, still uncommitted


def test_another_agents_already_staged_file_is_not_swept_in(repo, commit):
    (repo / "mine.py").write_text("mine\n")
    (repo / "theirs.py").write_text("theirs\n")
    _git(repo, "add", "theirs.py")  # the other agent staged, has not committed
    assert commit("mine", "mine.py").returncode == 0
    assert _files_in(repo) == ["mine.py"]
    # And theirs is still there for them to commit.
    assert "theirs.py" in _git(repo, "diff", "--cached", "--name-only").stdout


def test_concurrent_commits_all_land_and_none_contaminate(repo, commit):
    """Eight agents, one worktree, one file each.

    Without the lock this is where ``Unable to create '.git/index.lock'``
    appears, and where one agent's files land in another's commit.
    """
    names = [f"entry_{i}.py" for i in range(8)]
    for name in names:
        (repo / name).write_text(f"# {name}\n")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: commit(f"land {n}", n), names))

    assert all(r.returncode == 0 for r in results), [r.stderr for r in results if r.returncode]
    log = _git(repo, "log", "--format=%H").stdout.split()
    assert len(log) == 9  # seed + 8
    landed = {sha: _files_in(repo, sha) for sha in log[:8]}
    assert sorted(f for files in landed.values() for f in files) == sorted(names)
    for files in landed.values():
        assert len(files) == 1  # one file per commit, no contamination


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


def test_paths_outside_the_allowed_prefixes_are_refused(tmp_path, repo):
    policy = CommitPolicy(allowed_paths=("tensorrt_llm/", "tests/"))
    script = install_committer(tmp_path / "ws", repo, policy=policy)
    (repo / "README.md").write_text("readme\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("x\n")

    refused = subprocess.run([str(script), "m", "README.md"], capture_output=True, text=True)
    assert refused.returncode == 2
    assert "outside the paths this run may commit" in refused.stderr

    # One bad path refuses the whole call: nothing partial lands.
    mixed = subprocess.run(
        [str(script), "m", "tests/test_x.py", "README.md"], capture_output=True, text=True
    )
    assert mixed.returncode == 2
    assert _git(repo, "log", "--format=%s", "-1").stdout.strip() == "seed"

    ok = subprocess.run([str(script), "m", "tests/test_x.py"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr


def test_the_message_may_not_carry_co_authors_or_its_own_sign_off(repo, commit):
    (repo / "a.py").write_text("a\n")
    co = commit("[None][feat] x\n\nCo-authored-by: Someone <s@example.invalid>", "a.py")
    assert co.returncode == 2
    assert "Co-authored-by" in co.stderr
    so = commit("[None][feat] x\n\nSigned-off-by: Other <o@example.invalid>", "a.py")
    assert so.returncode == 2
    assert "Signed-off-by" in so.stderr


def test_author_committer_and_sign_off_are_one_baked_identity(tmp_path, repo, monkeypatch):
    """The sign-off email must equal the author email, whatever the shell exports.

    An inherited GIT_AUTHOR_* / GIT_COMMITTER_* beats `-c user.*`, and a
    sign-off is the committer's identity, so an operator environment that
    exports either would otherwise split them.
    """
    identity = CommitIdentity(name="Repo User", email="repo.user@example.invalid")
    script = install_committer(tmp_path / "ws", repo, policy=CommitPolicy(identity=identity))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "login")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "login@host.invalid")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "operator@host.invalid")
    monkeypatch.setenv("MV2FLOW_GIT_EMAIL", "runtime-override@host.invalid")
    (repo / "a.py").write_text("a\n")
    result = subprocess.run([str(script), "m", "a.py"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    fmt = "%an <%ae>|%cn <%ce>|%(trailers:key=Signed-off-by,valueonly)"
    author, committer, signoff = (
        _git(repo, "log", "-1", f"--format={fmt}").stdout.strip().split("|")
    )
    assert author == committer == signoff.strip() == "Repo User <repo.user@example.invalid>"


def _commit_raw(repo, message, path, content="x\n", env=None):
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(content)
    _git(repo, "add", path)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", message],
        check=True,
        capture_output=True,
        env=env,
    )
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def test_audit_commits_finds_every_kind_of_violation(tmp_path, repo):
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
    assert audit_commits(repo, f"{base}..HEAD", allow_co_authors=True) != findings
    scoped = audit_commits(repo, f"{base}..HEAD", paths=("pkg/",))
    assert {f.sha for f in scoped} == {placeholder}


def test_the_cli_refuses_to_start_without_an_identity(tmp_path, isolated_git_env):
    from agent_flow.workflows.trtllm_modeling_v2.cli import _commit_identity

    repo = tmp_path / "noid"
    repo.mkdir()
    _git(repo, "init", "-q")
    with pytest.raises(SystemExit) as exc:
        _commit_identity({"trtllm_repo_path": str(repo)})
    assert exc.value.code == 2


def test_the_run_installs_its_helper_under_the_task_policy(tmp_path, monkeypatch):
    """Preflight writes `bin/commit.sh` with the task's paths and the repo's identity."""
    import json

    import yaml

    from .test_delivery import _build_repo
    from .test_workflow import Script, _install, _workflow

    repo = _build_repo(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Repo User")
    _git(repo, "config", "user.email", "repo.user@example.invalid")
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["GptOssForCausalLM"]}))
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(repo),
        "target": {"family": "gpt_oss", "checkpoint_segment": "gpt_oss_120b"},
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": "/scratch/t", "CLAUDE_CODE_TMPDIR": "/scratch/t"},
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
            "model_key": "openai/gpt-oss-120b",
            "tolerance": 5.0,
        },
        "commit_policy": {"allowed_paths": ["tensorrt_llm/", "tests/"]},
    }
    task_file = tmp_path / "task.yaml"
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    monkeypatch.delenv("MV2FLOW_GIT_NAME", raising=False)
    monkeypatch.delenv("MV2FLOW_GIT_EMAIL", raising=False)

    workflow = _workflow(tmp_path, dry_run=True)
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    body = (tmp_path / "ws" / "bin" / "commit.sh").read_text()
    assert "ALLOWED_PATHS=(tensorrt_llm/ tests/)" in body
    assert "COMMIT_EMAIL=repo.user@example.invalid" in body
