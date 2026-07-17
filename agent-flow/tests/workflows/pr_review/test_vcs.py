from __future__ import annotations

import types
from pathlib import Path

import pytest

from agent_flow.workflows.pr_review import vcs


def test_resolve_bin_uses_env_override(monkeypatch):
    monkeypatch.setenv(vcs.GIT_BIN_ENV, "/custom/git")
    monkeypatch.setattr(vcs.shutil, "which", lambda c: None)
    # Env override is honored even when PATH lookup fails.
    assert vcs.resolve_bin("git") == "/custom/git"


def test_resolve_bin_falls_back_to_path(monkeypatch):
    monkeypatch.delenv(vcs.GIT_BIN_ENV, raising=False)
    monkeypatch.setattr(vcs.shutil, "which", lambda c: f"/usr/bin/{c}")
    # Defaults to git when no tool is named.
    assert vcs.resolve_bin() == "/usr/bin/git"


def test_resolve_bin_missing_raises(monkeypatch):
    monkeypatch.delenv(vcs.GIT_BIN_ENV, raising=False)
    monkeypatch.setattr(vcs.shutil, "which", lambda c: None)
    with pytest.raises(vcs.VcsError, match="not found on PATH"):
        vcs.resolve_bin("git")


def test_resolve_bin_rejects_gh_glab(monkeypatch):
    # gh / glab are the sourcing agent's job — this module only runs git.
    with pytest.raises(vcs.VcsError, match="only runs `git`"):
        vcs.resolve_bin("github")


def _fake_proc(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_returns_stdout_on_success(monkeypatch):
    monkeypatch.setattr(vcs.subprocess, "run", lambda *a, **k: _fake_proc(0, "hello\n"))
    assert vcs._run(["echo", "hello"]) == "hello\n"


def test_run_raises_on_nonzero(monkeypatch):
    monkeypatch.setattr(vcs.subprocess, "run", lambda *a, **k: _fake_proc(2, "", "boom"))
    with pytest.raises(vcs.VcsError, match="exited 2: boom"):
        vcs._run(["false"])


def test_run_raises_on_missing_binary(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(vcs.subprocess, "run", _raise)
    with pytest.raises(vcs.VcsError, match="command not found"):
        vcs._run(["does-not-exist"])


def test_resolve_diff_base_ref_prefers_origin(monkeypatch):
    monkeypatch.setattr(vcs, "_rev_parse_ok", lambda repo, ref: ref == "origin/main")
    assert vcs.resolve_diff_base_ref(Path("/repo"), "main") == "origin/main"


def test_resolve_diff_base_ref_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(vcs, "_rev_parse_ok", lambda repo, ref: ref == "main")
    assert vcs.resolve_diff_base_ref(Path("/repo"), "main") == "main"


def test_resolve_diff_base_ref_missing_raises(monkeypatch):
    monkeypatch.setattr(vcs, "_rev_parse_ok", lambda repo, ref: False)
    with pytest.raises(vcs.VcsError, match="not found"):
        vcs.resolve_diff_base_ref(Path("/repo"), "main")


def test_diff_command_uses_merge_base_to_working_tree():
    cmd = vcs.diff_command(Path("/repo"), "origin/main")
    assert "merge-base origin/main HEAD" in cmd
    assert "git -C" in cmd and "/repo" in cmd
    # No commit range (three-dot) — the working tree must be included.
    assert "..." not in cmd


def test_build_pr_context_uses_reported_metadata(monkeypatch):
    monkeypatch.setattr(vcs, "resolve_diff_base_ref", lambda repo, base: "origin/main")
    metadata = {
        "base": "main",
        "head": "feat",
        "title": "T",
        "author": "alice",
        "body": "B",
        "url": "U",
    }
    ctx = vcs.build_pr_context(Path("/repo"), "9", metadata, None)
    assert ctx.base_branch == "main"
    assert ctx.diff_base_ref == "origin/main"
    assert ctx.head_branch == "feat"
    assert ctx.title == "T"
    assert ctx.author == "alice"
    assert "merge-base origin/main HEAD" in ctx.diff_command


def test_build_pr_context_base_override_wins(monkeypatch):
    seen = {}

    def _resolve(repo, base):
        seen["base"] = base
        return f"origin/{base}"

    monkeypatch.setattr(vcs, "resolve_diff_base_ref", _resolve)
    metadata = {"base": "main", "head": "f"}
    ctx = vcs.build_pr_context(Path("/repo"), "9", metadata, "release-2.0")
    assert seen["base"] == "release-2.0"
    assert ctx.base_branch == "release-2.0"


def test_build_pr_context_missing_base_raises(monkeypatch):
    # No reported base and no override → actionable error pointing at --base.
    monkeypatch.setattr(vcs, "resolve_diff_base_ref", lambda repo, base: "origin/main")
    with pytest.raises(vcs.VcsError, match="pass --base"):
        vcs.build_pr_context(Path("/repo"), "9", {"head": "f"}, None)


def test_format_pr_context_md_includes_diff_command_and_no_posting_hint(monkeypatch):
    ctx = vcs.PrContext(
        identifier="9",
        repo=Path("/repo"),
        base_branch="main",
        head_branch="feat",
        diff_base_ref="origin/main",
        diff_command='git -C "/repo" diff "$(...)"',
        title="Add a thing",
        author="alice",
        url="https://example/pr/9",
        body="Body text",
    )
    md = vcs.format_pr_context_md(ctx)
    assert "Add a thing" in md
    assert 'git -C "/repo" diff' in md
    assert "Diff under review" in md
    assert "Body text" in md
    # Platform-agnostic heading — we don't distinguish PR from MR.
    assert "# PR/MR under review" in md
