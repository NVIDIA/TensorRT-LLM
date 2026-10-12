# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delivery's tail: the commit audit as a delivery check, and the pre-merge pack."""

from __future__ import annotations

import json
import os
import subprocess

import pytest
import yaml

from agent_flow.workflows.staircase_onboard import delivery
from agent_flow.workflows.staircase_onboard.committer import CommitPolicy
from agent_flow.workflows.staircase_onboard.state import load_state

from .test_delivery import TARGET_REL, _build_repo
from .test_workflow import Script, _install, _workflow, _write_vocabulary


def _git(repo, *args, env=None):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env
    ).stdout


def _signed_repo(tmp_path):
    """`_build_repo`'s trees, committed on `main`, with a `work` branch checked out."""
    repo = _build_repo(tmp_path)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Repo User")
    _git(repo, "config", "user.email", "repo.user@example.invalid")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-s", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "work")
    return repo


def _commit(repo, rel, text, *, signoff=True, env=None):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", *(["-s"] if signoff else []), "-m", f"touch {rel}", env=env)


# ------------------------------------------------------------------ the check


def test_an_unsigned_commit_under_the_runs_paths_fails_delivery(tmp_path):
    repo = _signed_repo(tmp_path)
    _commit(repo, f"{TARGET_REL}/extra.py", "x\n", signoff=False)
    checks = delivery.check_commit_hygiene(repo, CommitPolicy(allowed_paths=("tensorrt_llm/",)))
    assert [c.name for c in checks] == ["git:commit-hygiene"]
    assert not checks[0].passed
    assert "no Signed-off-by" in checks[0].detail and "premerge resign" in checks[0].detail


def test_a_sign_off_that_is_not_the_authors_fails_delivery(tmp_path):
    repo = _signed_repo(tmp_path)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "login",
        "GIT_AUTHOR_EMAIL": "login@host.invalid",
    }
    _commit(
        repo, f"{TARGET_REL}/extra.py", "x\n", env=env
    )  # signed as the committer, not the author
    (check,) = delivery.check_commit_hygiene(repo, CommitPolicy(allowed_paths=("tensorrt_llm/",)))
    assert not check.passed and "author email <login@host.invalid>" in check.detail


def test_commits_outside_the_runs_paths_are_not_the_runs_to_answer_for(tmp_path):
    repo = _signed_repo(tmp_path)
    _commit(repo, "agent-flow/x.py", "x\n", signoff=False)
    _commit(repo, f"{TARGET_REL}/extra.py", "x\n")
    (check,) = delivery.check_commit_hygiene(repo, CommitPolicy(allowed_paths=("tensorrt_llm/",)))
    assert check.passed, check.detail
    # Audited over the whole repository, the same branch fails.
    (whole,) = delivery.check_commit_hygiene(repo, CommitPolicy())
    assert not whole.passed


def test_no_git_or_no_base_makes_no_claim(tmp_path):
    assert delivery.check_commit_hygiene(tmp_path, CommitPolicy()) == []
    repo = _signed_repo(tmp_path / "x")
    assert delivery.check_commit_hygiene(repo, CommitPolicy(base_ref="no-such-ref")) == []


def test_verify_delivery_runs_the_audit_only_when_given_a_policy(tmp_path):
    repo = _signed_repo(tmp_path)
    kwargs = dict(
        repo_root=repo,
        target_rel=TARGET_REL,
        family="gpt_oss",
        reference_yaml=f"{delivery.ACCURACY_REFERENCES_DIR}/gsm8k.yaml",
        model_key="openai/gpt-oss-120b",
        accuracy_test="test_modeling_v2_gpt_oss.py",
    )
    names = [c.name for c in delivery.verify_delivery(**kwargs).checks]
    assert "git:commit-hygiene" not in names
    with_policy = delivery.verify_delivery(**kwargs, commit_policy=CommitPolicy())
    assert "git:commit-hygiene" in [c.name for c in with_policy.checks]


# ------------------------------------------------------------------ the pack


def test_the_pack_is_written_into_the_reports(tmp_path):
    repo = _signed_repo(tmp_path)
    _commit(repo, f"{TARGET_REL}/extra.py", "x = 1\n")
    path, summary = delivery.write_premerge_pack(repo, tmp_path / "reports", family="gpt_oss")
    assert path == tmp_path / "reports" / "premerge.md"
    text = path.read_text()
    assert "## Commit audit (clean)" in text
    assert "premerge lint" in text and "premerge split" in text
    assert any(line.startswith("copyright:") for line in summary)
    # A fixture that is not a repository gets a report that says so.
    plain, _ = delivery.write_premerge_pack(tmp_path / "nogit", tmp_path / "r2", family="gpt_oss")
    assert "not a git work tree" in plain.read_text()


# ------------------------------------------------------------------ the run


@pytest.fixture
def git_task(tmp_path):
    repo = _signed_repo(tmp_path)
    checkpoint = tmp_path / "gpt-oss-120b"
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
    }
    task_file = tmp_path / "task.yaml"
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return repo, task_file


def _run(tmp_path, monkeypatch, task_file):
    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)
    return load_state(tmp_path / "ws" / ".staircase_onboard_state.json"), calls


def test_a_delivered_run_ends_with_the_pre_merge_pack(tmp_path, monkeypatch, git_task):
    _, task_file = git_task
    state, _ = _run(tmp_path, monkeypatch, task_file)
    assert state.verdict == "delivered"
    pack = tmp_path / "ws" / "reports" / "premerge.md"
    assert pack.is_file() and "## Remaining steps" in pack.read_text()
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "pre-merge pack" in journal and "git:commit-hygiene" in journal


def test_an_unsigned_run_commit_fails_delivery(tmp_path, monkeypatch, git_task):
    repo, task_file = git_task
    _commit(repo, f"{TARGET_REL}/unsigned.py", "x\n", signoff=False)
    state, calls = _run(tmp_path, monkeypatch, task_file)
    assert state.verdict == "delivery-failed"
    assert "auditor" not in [n for n, _ in calls]
    assert not (tmp_path / "ws" / "reports" / "premerge.md").exists()
