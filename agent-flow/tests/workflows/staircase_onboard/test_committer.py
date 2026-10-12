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

from agent_flow.workflows.staircase_onboard.committer import (
    SCRIPT_NAME,
    commit_instruction,
    install_committer,
    lock_is_held,
    render_script,
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

    run.script = script
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


def test_an_untracked_file_from_another_agent_is_not_swept_in(repo, commit):
    """The measured failure: another agent's in-progress file in my commit."""
    (repo / "mine.py").write_text("mine\n")
    (repo / "theirs_half_written.py").write_text("half\n")
    assert commit("mine", "mine.py").returncode == 0
    assert _files_in(repo) == ["mine.py"]


def test_another_agents_already_staged_file_is_not_swept_in(repo, commit):
    """The worse variant: they had already run `git add`."""
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


def test_a_commit_that_swept_more_than_asked_is_undone(repo, commit):
    """The verification, exercised by pointing at a directory.

    A directory pathspec is the exact mistake the prose has always warned
    about. Here it stops being advice: the script refuses the result.
    """
    sub = repo / "pkg"
    sub.mkdir()
    (sub / "mine.py").write_text("mine\n")
    (sub / "theirs.py").write_text("theirs\n")
    before = _git(repo, "rev-parse", "HEAD").stdout.strip()

    result = commit("sloppy", "pkg")
    assert result.returncode == 1
    assert "REFUSING" in result.stderr
    assert "pkg/theirs.py" in result.stderr
    # HEAD is back where it was, and the work is still in the tree.
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == before
    assert (sub / "mine.py").exists()


def test_a_path_that_does_not_exist_is_rejected_before_the_lock(repo, commit):
    result = commit("typo", "does_not_exist.py")
    assert result.returncode == 2
    assert "does not exist" in result.stderr


def test_it_refuses_without_a_message(repo, commit):
    assert commit("only-a-message").returncode == 2


def test_nothing_to_commit_fails_loudly_rather_than_silently(repo, commit):
    """An agent that thinks it committed and did not is the worst outcome."""
    result = commit("no change", "seed.txt")
    assert result.returncode == 1
    assert "nothing to commit" in result.stderr


def test_the_sha_and_the_file_list_are_printed(repo, commit):
    """The agent's report has to be checkable against the repo."""
    (repo / "a.py").write_text("a\n")
    result = commit("add a", "a.py")
    sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert sha in result.stdout
    assert "a.py" in result.stdout


def test_waiting_for_the_lock_is_a_distinct_exit_code(tmp_path, repo):
    """So a queue is never mistaken for a broken repo."""
    script = install_committer(tmp_path / "ws", repo, timeout=1)
    (repo / "a.py").write_text("a\n")
    lock = tmp_path / "ws" / ".commit.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "10"])
    try:
        result = subprocess.run(
            [str(script), "blocked", "a.py"], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 75
        assert "did not get it" in result.stderr
        assert "symptom to report" in result.stderr
    finally:
        holder.kill()
        holder.wait()


def test_the_lock_can_be_observed_by_the_orchestrator(tmp_path, repo):
    """Tells "still committing" from "wedged" when a dispatch overruns."""
    install_committer(tmp_path / "ws", repo)
    assert not lock_is_held(tmp_path / "ws")
    lock = tmp_path / "ws" / ".commit.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "10"])
    try:
        import time

        for _ in range(50):
            if lock_is_held(tmp_path / "ws"):
                break
            time.sleep(0.1)
        assert lock_is_held(tmp_path / "ws")
    finally:
        holder.kill()
        holder.wait()


# ------------------------------------------------------------- installation


def test_the_script_is_executable_and_self_contained(tmp_path, repo):
    script = install_committer(tmp_path / "ws", repo)
    assert script.name == SCRIPT_NAME
    assert script.stat().st_mode & 0o111
    body = script.read_text()
    assert str(repo) in body  # baked, so no argument an agent can get wrong
    assert "flock" in body


def test_paths_with_spaces_survive_the_bake(tmp_path):
    body = render_script("/a path/repo", "/a path/ws/.commit.lock")
    assert "'/a path/repo'" in body


def test_the_instruction_names_the_script_and_the_refusal(tmp_path, repo):
    script = install_committer(tmp_path / "ws", repo)
    text = commit_instruction(script)
    assert str(script) in text
    assert "undoes the commit" in text
    assert "git add -A" in text  # still named, still forbidden


# --------------------------------------------------- wired into the prompts


def test_the_committing_roles_are_pointed_at_the_script(tmp_path, repo):
    from agent_flow.workflows.staircase_onboard.prompts import build_staircase_onboard_prompts

    script = install_committer(tmp_path / "ws", repo)
    prompts = build_staircase_onboard_prompts(commit_script=str(script))
    assert str(script) in prompts.smith
    assert str(script) in prompts.assembler


def test_a_role_that_does_not_commit_is_not_handed_a_commit_recipe(tmp_path, repo):
    """It will otherwise find a reason to use it."""
    from agent_flow.workflows.staircase_onboard.prompts import build_staircase_onboard_prompts

    script = install_committer(tmp_path / "ws", repo)
    prompts = build_staircase_onboard_prompts(commit_script=str(script))
    for role in (prompts.plan_drafter, prompts.plan_reviewer, prompts.reviewer, prompts.auditor):
        assert str(script) not in role


def test_the_new_recipe_explicitly_supersedes_the_old_one(tmp_path, repo):
    """Two commit recipes in one prompt is how the wrong one gets picked."""
    from agent_flow.workflows.staircase_onboard.prompts import build_staircase_onboard_prompts

    script = install_committer(tmp_path / "ws", repo)
    smith = build_staircase_onboard_prompts(commit_script=str(script)).smith
    # The baseline block is still there — it is embedded in the role prompt —
    # so the replacement has to say so, and has to come after it.
    assert "Commit **by explicit file path**" in smith
    assert "this replaces the Committing section above" in smith
    assert smith.index("Commit **by explicit file path**") < smith.index("this replaces the")


def test_the_undo_rewinds_exactly_one_commit(repo, commit):
    """Not two.

    The refusal message quoted the reset command inside a double-quoted
    `echo`, where a backtick is a command substitution — so writing the
    example ran it, and the explicit reset below then ran it again. In a
    repo with one commit that fails harmlessly; in a repo with history it
    silently throws away the previous commit.
    """
    for i in range(3):
        (repo / f"history_{i}.txt").write_text(f"{i}\n")
        _git(repo, "add", f"history_{i}.txt")
        _git(repo, "commit", "-q", "-m", f"history {i}")
    log_before = _git(repo, "log", "--format=%H").stdout.split()

    sub = repo / "pkg"
    sub.mkdir()
    (sub / "mine.py").write_text("mine\n")
    (sub / "theirs.py").write_text("theirs\n")
    assert commit("sloppy", "pkg").returncode == 1

    assert _git(repo, "log", "--format=%H").stdout.split() == log_before


# ------------------------------------------- the linter cannot see this code


def test_the_generated_script_is_valid_shell(tmp_path, repo):
    """`bash -n` parses without running.

    Neither `ruff` nor `shellcheck` looks inside a Python string literal, so
    this script gets no static checking at all unless a test asks for it.
    """
    script = install_committer(tmp_path / "ws", repo)
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_script_has_no_command_substitution_in_its_messages(tmp_path, repo):
    """The bug that shipped: a backtick inside a double-quoted `echo`.

    It quoted `git reset --soft HEAD^` as an example and bash *ran* it, so
    the explicit reset below rewound a second, unrelated commit. Static
    rule, because the behavioural test only covers the one message that
    happened to have it.
    """
    script = install_committer(tmp_path / "ws", repo)
    offenders = [
        (n, line)
        for n, line in enumerate(script.read_text().splitlines(), 1)
        if line.strip().startswith("echo ") and '"' in line and "`" in line
    ]
    assert not offenders, f"backtick inside a double-quoted echo: {offenders}"


def test_every_commit_carries_a_dco_sign_off_under_gits_own_identity(repo, commit, monkeypatch):
    """The target repository rejects a commit without `Signed-off-by`.

    A login name is not a sign-off identity either: the name git itself would
    use comes before $USER.
    """
    monkeypatch.delenv("STAIRCASE_GIT_NAME", raising=False)
    monkeypatch.delenv("STAIRCASE_GIT_EMAIL", raising=False)
    (repo / "signed.txt").write_text("x\n")
    result = commit("[None][test] signed", "signed.txt")
    assert result.returncode == 0, result.stderr
    assert (
        _git(repo, "log", "-1", "--format=%an <%ae>").stdout.strip() == "Test <t@example.invalid>"
    )
    trailers = _git(
        repo, "log", "-1", "--format=%(trailers:key=Signed-off-by,valueonly)"
    ).stdout.strip()
    assert trailers == "Test <t@example.invalid>"
