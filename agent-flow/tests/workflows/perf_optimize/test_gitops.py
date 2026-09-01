"""Tests for the perf-optimize git wrappers against a real repository.

The revert semantics are load-bearing: ``discard_uncommitted`` must wipe
a rejected attempt's edits (tracked and untracked) while sparing
gitignored build artifacts — an editable TRT-LLM checkout keeps compiled
outputs the workflow must never destroy.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from agent_flow.workflows.perf_optimize import gitops


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
    (repo / ".gitignore").write_text("build/\n", encoding="utf-8")
    (repo / "src.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_is_git_repo(repo, tmp_path):
    assert gitops.is_git_repo(repo) is True
    plain = tmp_path / "plain"
    plain.mkdir()
    assert gitops.is_git_repo(plain) is False


def test_worktree_clean_ignores_gitignored_artifacts(repo):
    assert gitops.worktree_clean(repo) is True

    # Gitignored build outputs must not read as "dirty".
    (repo / "build").mkdir()
    (repo / "build" / "lib.so").write_text("bin", encoding="utf-8")
    assert gitops.worktree_clean(repo) is True

    (repo / "src.py").write_text("x = 2\n", encoding="utf-8")
    assert gitops.worktree_clean(repo) is False


def test_branch_and_head_helpers(repo):
    head = gitops.rev_parse_head(repo)
    assert len(head) == 40

    gitops.create_branch(repo, "perf-optimize/test")
    assert gitops.current_branch(repo) == "perf-optimize/test"
    # Creating an existing branch is an error, not a silent reset.
    with pytest.raises(gitops.GitOpsError):
        gitops.create_branch(repo, "perf-optimize/test")

    default = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    assert default == "perf-optimize/test"
    gitops.checkout(repo, "perf-optimize/test")  # no-op checkout is fine


def test_commit_all_stages_everything_but_respects_gitignore(repo):
    base = gitops.rev_parse_head(repo)
    (repo / "src.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "new_file.py").write_text("y = 3\n", encoding="utf-8")
    (repo / "build").mkdir()
    (repo / "build" / "lib.so").write_text("bin", encoding="utf-8")

    new_head = gitops.commit_all(repo, "perf-optimize: opt-001 accepted")

    assert new_head != base
    assert "diff --git a/src.py b/src.py" in gitops.format_patch(repo, base, new_head)
    assert gitops.worktree_clean(repo) is True
    assert _git(repo, "log", "-1", "--pretty=%s") == "perf-optimize: opt-001 accepted"
    committed = _git(repo, "show", "--name-only", "--pretty=format:", "HEAD").split()
    assert "src.py" in committed
    assert "new_file.py" in committed
    # The gitignored artifact stayed out of the commit.
    assert not any(name.startswith("build") for name in committed)


def test_commit_all_bypasses_failing_precommit_hook(repo):
    """A developer checkout with `pre-commit install`ed must not abort the accept.

    Real-world failure: TRT-LLM's yapf hook reformatted the optimizer's
    edits and exited non-zero, `git commit` aborted, and the workflow
    crashed right after the evaluator's APPROVE. The orchestrator's
    bookkeeping commits must bypass hooks (`--no-verify`).
    """
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        # Mimic a formatter: mutate the worktree, then fail the commit.
        "echo '# reformatted' >> src.py\n"
        "exit 1\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    commit_msg_hook = repo / ".git" / "hooks" / "commit-msg"
    commit_msg_hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    commit_msg_hook.chmod(0o755)

    base = gitops.rev_parse_head(repo)
    (repo / "src.py").write_text("x = 2\n", encoding="utf-8")

    new_head = gitops.commit_all(repo, "perf-optimize: opt-001 accepted")

    assert new_head != base
    assert gitops.worktree_clean(repo) is True
    # The hook never ran: the worktree holds exactly what was committed.
    assert (repo / "src.py").read_text(encoding="utf-8") == "x = 2\n"


def test_discard_uncommitted_spares_gitignored_artifacts(repo):
    (repo / "build").mkdir()
    (repo / "build" / "lib.so").write_text("bin", encoding="utf-8")
    (repo / "src.py").write_text("x = 999\n", encoding="utf-8")  # tracked edit
    (repo / "junk.py").write_text("z\n", encoding="utf-8")  # untracked add

    gitops.discard_uncommitted(repo)

    # The rejected attempt's edits are gone ...
    assert (repo / "src.py").read_text(encoding="utf-8") == "x = 1\n"
    assert not (repo / "junk.py").exists()
    # ... but gitignored build artifacts survive (clean -fd without -x).
    assert (repo / "build" / "lib.so").is_file()
    assert gitops.worktree_clean(repo) is True


def test_worktree_reset_and_fast_forward(repo, tmp_path):
    base = gitops.rev_parse_head(repo)
    gitops.create_branch(repo, "perf-optimize/campaign")
    item_worktree = tmp_path / "item-worktree"
    gitops.create_worktree(
        repo,
        item_worktree,
        "perf-optimize/campaign-round-1-item-opt-001",
        base,
    )
    (item_worktree / "src.py").write_text("x = 2\n", encoding="utf-8")
    candidate = gitops.commit_all(item_worktree, "candidate")
    assert candidate != base

    gitops.fast_forward(repo, "perf-optimize/campaign-round-1-item-opt-001")
    assert (repo / "src.py").read_text(encoding="utf-8") == "x = 2\n"

    (item_worktree / "src.py").write_text("broken\n", encoding="utf-8")
    gitops.reset_to(item_worktree, candidate)
    assert (item_worktree / "src.py").read_text(encoding="utf-8") == "x = 2\n"
    gitops.remove_worktree(repo, item_worktree)
    assert not item_worktree.exists()


def test_git_errors_raise_with_context(repo):
    with pytest.raises(gitops.GitOpsError, match="checkout"):
        gitops.checkout(repo, "no-such-branch")


def test_worktree_clean_ignores_a_built_worktrees_submodules(repo, tmp_path):
    """A built worktree makes plain `git status` exit 128 on a clean tree.

    Its submodule ``.git`` files point at gitdirs under
    ``<base>/.git/worktrees/<name>/modules/...`` which need not exist. git then
    writes ``fatal: not a git repository: 3rdparty/MSA/...`` to stderr and exits
    128, while stdout — the actual answer — is empty. So the tree is clean and the
    command fails, and the error names a missing repository rather than uncommitted
    work, which sends you looking in the wrong place.

    Hit on the real GB300 debug worktree. Pre-existing rather than introduced by the
    ssh work: it breaks the same way locally, it had just never been pointed at a
    built worktree with submodules.
    """
    sub = tmp_path / "subsrc"
    sub.mkdir()
    _git(sub, "init", "-b", "main")
    _git(sub, "config", "user.email", "t@t")
    _git(sub, "config", "user.name", "Test")
    (sub / "f.txt").write_text("s\n", encoding="utf-8")
    _git(sub, "add", "-A")
    _git(sub, "commit", "-q", "-m", "sub")

    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(sub), "vendor/sub")
    _git(repo, "commit", "-q", "-m", "add submodule")

    wt = tmp_path / "built"
    _git(repo, "worktree", "add", "--detach", str(wt))
    _git(wt, "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive")
    pointer = (wt / "vendor" / "sub" / ".git").read_text(encoding="utf-8").strip()
    gitdir = (wt / "vendor" / "sub" / pointer.removeprefix("gitdir: ")).resolve()
    assert gitdir.exists(), "precondition: the submodule gitdir exists before removal"
    shutil.rmtree(gitdir)

    plain = subprocess.run(
        ["git", "-C", str(wt), "status", "--porcelain"], capture_output=True, text=True
    )
    assert plain.returncode == 128 and plain.stdout == "", (
        "precondition: plain status must fail while reporting a clean tree, or this "
        "test cannot tell the fix from its absence"
    )

    assert gitops.worktree_clean(wt) is True


# ── where the git command runs ───────────────────────────────────────────────
def test_git_runs_locally_by_default(repo):
    """The historical behaviour, and what every other test in this file assumes.

    Asserted explicitly because it is now a *choice* rather than the only option: a
    default that silently became remote would make every existing deployment reach
    for a host it was never told about.
    """
    assert gitops.cluster_ssh() == ""
    assert gitops.rev_parse_head(repo)  # a real local git call


def test_use_cluster_routes_every_command_through_ssh(repo, monkeypatch):
    """One switch has to cover all nine helpers, not just the one under test.

    ``_git`` is the only place a git command is built in this package — there is no
    other ``subprocess`` call here — which is what makes that true. So this asserts
    the *rendering* for a representative sample rather than running ssh in a unit
    test.
    """
    seen: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = "deadbeef\n"
        stderr = ""

    monkeypatch.setattr(
        gitops.subprocess, "run", lambda cmd, **kw: (seen.append(cmd), _Result())[1]
    )
    gitops.use_cluster("me@login-01")
    remote_worktree = repo.parent / "remote-only" / "item"
    try:
        gitops.rev_parse_head(repo)
        gitops.discard_uncommitted(repo)
        gitops.create_worktree(repo, remote_worktree, "candidate", "deadbeef")
    finally:
        # Restore, or the module-level host leaks into every later test in the
        # session and they pass for the wrong reason.
        gitops.use_cluster("")

    assert not remote_worktree.parent.exists()
    assert len(seen) == 4, seen  # rev-parse + reset --hard + clean -fd + worktree add
    for cmd in seen:
        assert cmd[0] == "ssh" and "me@login-01" in cmd, cmd
        # The whole git command is ONE argument, so ssh's own shell cannot re-split
        # a path or a commit message containing spaces.
        assert cmd[-1].startswith("git -C "), cmd[-1]


def test_a_remote_failure_names_the_host_it_ran_on(repo, monkeypatch):
    """Reproducing a remote failure locally gives a different answer.

    So the error carries the ssh line, not the bare git command — otherwise a reader
    copies it, runs it here, and draws a conclusion about the wrong machine.
    """

    class _Fail:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(gitops.subprocess, "run", lambda cmd, **kw: _Fail())
    gitops.use_cluster("me@login-01")
    try:
        with pytest.raises(gitops.GitOpsError, match="me@login-01"):
            gitops.rev_parse_head(repo)
    finally:
        gitops.use_cluster("")
