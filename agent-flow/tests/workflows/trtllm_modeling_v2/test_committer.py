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
    DEFAULT_ALLOWED_PATHS,
    SCRIPT_NAME,
    CommitIdentity,
    CommitPolicy,
    CommitPolicyError,
    audit_commits,
    commit_instruction,
    install_committer,
    render_script,
    resolve_commit_identity,
    trial_commit,
    validate_commit_policy_block,
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
    """The failure this guards against: another agent's in-progress file in my commit."""
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


def test_a_directory_is_refused_before_anything_is_staged(repo, commit):
    """A directory pathspec is the exact mistake the prompts warn about.

    It is refused up front, so nothing is staged and nothing has to be
    rewound.
    """
    sub = repo / "pkg"
    sub.mkdir()
    (sub / "mine.py").write_text("mine\n")
    (sub / "theirs.py").write_text("theirs\n")
    before = _git(repo, "rev-parse", "HEAD").stdout.strip()

    result = commit("sloppy", "pkg")
    assert result.returncode == 2
    assert "is a directory" in result.stderr
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == before
    assert _git(repo, "diff", "--cached", "--name-only").stdout.strip() == ""
    assert (sub / "mine.py").exists()


def _sweeping_hook(repo, tmp_path, extra):
    """A pre-commit hook that stages ``extra`` into the commit being made.

    The one way left for a commit to carry more than was asked once every
    argument is an explicit file: something else touches the index while
    git commits. The verification after the commit is what catches it.
    """
    hooks = tmp_path / "hooks"
    hooks.mkdir(exist_ok=True)
    hook = hooks / "pre-commit"
    hook.write_text(f"#!/bin/sh\ngit add -- {extra}\n")
    hook.chmod(0o755)
    _git(repo, "config", "core.hooksPath", str(hooks))


def test_a_commit_that_swept_more_than_asked_is_undone(tmp_path, repo, commit):
    (repo / "mine.py").write_text("mine\n")
    (repo / "theirs.py").write_text("theirs\n")
    _sweeping_hook(repo, tmp_path, "theirs.py")
    before = _git(repo, "rev-parse", "HEAD").stdout.strip()

    result = commit("mine", "mine.py")
    assert result.returncode == 1
    assert "REFUSING" in result.stderr
    assert "theirs.py" in result.stderr
    # HEAD is back where it was, and the work is still in the tree.
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == before
    assert (repo / "mine.py").exists()


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
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import (
        build_trtllm_modeling_v2_prompts,
    )

    script = install_committer(tmp_path / "ws", repo)
    prompts = build_trtllm_modeling_v2_prompts(commit_script=str(script))
    assert str(script) in prompts.smith
    assert str(script) in prompts.assembler


def test_a_role_that_does_not_commit_is_not_handed_a_commit_recipe(tmp_path, repo):
    """It will otherwise find a reason to use it."""
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import (
        build_trtllm_modeling_v2_prompts,
    )

    script = install_committer(tmp_path / "ws", repo)
    prompts = build_trtllm_modeling_v2_prompts(commit_script=str(script))
    for role in (prompts.plan_drafter, prompts.plan_reviewer, prompts.reviewer):
        assert str(script) not in role


def test_a_committing_role_gets_exactly_one_commit_recipe(tmp_path, repo):
    """Two commit recipes in one prompt is how the wrong one gets picked."""
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import (
        build_trtllm_modeling_v2_prompts,
    )

    script = install_committer(tmp_path / "ws", repo)
    smith = build_trtllm_modeling_v2_prompts(commit_script=str(script)).smith
    assert smith.count("## Committing") == 1
    assert "git add -A" not in smith.replace("never `git add -A`", "")


def test_the_undo_rewinds_exactly_one_commit(tmp_path, repo, commit):
    """Not two.

    A reset command quoted inside a double-quoted `echo` is a command
    substitution, so printing the example would run it and the explicit
    reset would then run it again. In a repo with one commit that fails
    harmlessly; in a repo with history it silently throws away the previous
    commit.
    """
    for i in range(3):
        (repo / f"history_{i}.txt").write_text(f"{i}\n")
        _git(repo, "add", f"history_{i}.txt")
        _git(repo, "commit", "-q", "-m", f"history {i}")
    log_before = _git(repo, "log", "--format=%H").stdout.split()

    (repo / "mine.py").write_text("mine\n")
    (repo / "theirs.py").write_text("theirs\n")
    _sweeping_hook(repo, tmp_path, "theirs.py")
    assert commit("sloppy", "mine.py").returncode == 1

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
    """No backtick inside a double-quoted `echo`.

    Quoting `git reset --soft HEAD^` that way makes bash *run* it, and the
    explicit reset then rewinds a second, unrelated commit. Static rule,
    because the behavioural test covers only one message.
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
    monkeypatch.delenv("MV2FLOW_GIT_NAME", raising=False)
    monkeypatch.delenv("MV2FLOW_GIT_EMAIL", raising=False)
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


# ------------------------------------------------- what is refused up front


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


def test_the_bindings_symlink_is_never_committed(tmp_path, repo, commit):
    """The accident this refusal exists for.

    A worktree links the main checkout's native build. `.gitignore` names
    `tensorrt_llm/bindings/` — the directory form — so the link form is
    untracked and visible, and `git add -A` commits it. Naming it, or
    any path beneath it, is refused before anything is staged; a real file
    beside it still commits.
    """
    (repo / ".gitignore").write_text("tensorrt_llm/bindings/\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", "ignore the build output")
    build = tmp_path / "main-checkout" / "tensorrt_llm" / "bindings"
    build.mkdir(parents=True)
    (build / "__init__.pyi").write_text("# stub\n")
    pkg = repo / "tensorrt_llm"
    pkg.mkdir()
    (pkg / "bindings").symlink_to(build)
    (pkg / "modeling.py").write_text("x = 1\n")
    # The premise: git sees the link as an ordinary untracked path.
    assert "tensorrt_llm/bindings" in _git(repo, "status", "--porcelain", "-uall").stdout

    for path in ("tensorrt_llm/bindings", "tensorrt_llm/bindings/__init__.pyi"):
        result = commit("add the link", path)
        assert result.returncode == 2, result.stderr
        assert "symlink git does not track" in result.stderr
        assert _git(repo, "diff", "--cached", "--name-only").stdout.strip() == ""

    result = commit("the real file", "tensorrt_llm/modeling.py")
    assert result.returncode == 0, result.stderr
    assert _files_in(repo) == ["tensorrt_llm/modeling.py"]
    assert "120000" not in _git(repo, "ls-tree", "-r", "HEAD").stdout


def test_a_tracked_symlink_may_still_be_updated(repo, commit):
    """Only links git does not already track are refused."""
    (repo / "target_a.txt").write_text("a\n")
    (repo / "link").symlink_to("target_a.txt")
    _git(repo, "add", "target_a.txt", "link")
    _git(repo, "commit", "-q", "-m", "a tracked link")
    (repo / "link").unlink()
    (repo / "link").symlink_to("seed.txt")
    result = commit("retarget", "link")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "placeholder",
    [".mcp.json", "pkg/.mcp.json", "pkg/.claude/settings.json", "pkg/.claude/.cc-writes/x"],
)
def test_sandbox_placeholders_are_refused(repo, commit, placeholder):
    """The sandbox leaves them wherever its shell goes."""
    path = repo / placeholder
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    result = commit("placeholder", placeholder)
    assert result.returncode == 2
    assert "placeholder" in result.stderr


@pytest.mark.parametrize(
    "path,why",
    [
        ("*.py", "glob"),
        ("./a.py", "plain repo-relative"),
        ("pkg/../a.py", "plain repo-relative"),
        ("/etc/passwd", "absolute"),
    ],
)
def test_only_plain_explicit_paths_are_accepted(repo, commit, path, why):
    (repo / "a.py").write_text("a\n")
    (repo / "pkg").mkdir()
    result = commit("bad path", path)
    assert result.returncode == 2
    assert why in result.stderr


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


def test_the_runs_own_workspace_is_denied_when_it_sits_inside_the_repo(tmp_path, repo):
    workspace = repo / "onboard" / "workspace"
    workspace.mkdir(parents=True)
    policy = CommitPolicy.from_task({"trtllm_repo_path": str(repo)}, workspace=workspace)
    assert policy.denied_paths == ("onboard/workspace/",)
    assert policy.allowed_paths == DEFAULT_ALLOWED_PATHS
    script = install_committer(
        workspace, repo, policy=CommitPolicy(denied_paths=policy.denied_paths)
    )
    (workspace / "journal.md").write_text("notes\n")
    result = subprocess.run(
        [str(script), "m", "onboard/workspace/journal.md"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "its own workspace" in result.stderr


def test_a_deleted_tracked_file_can_be_committed_by_path(repo, commit):
    (repo / "seed.txt").unlink()
    result = commit("drop seed", "seed.txt")
    assert result.returncode == 0, result.stderr
    assert _git(repo, "ls-tree", "-r", "--name-only", "HEAD").stdout.strip() == ""


# ------------------------------------------------- identity and trailers


def test_the_message_may_not_carry_co_authors_or_its_own_sign_off(repo, commit):
    (repo / "a.py").write_text("a\n")
    co = commit("[None][feat] x\n\nCo-authored-by: Someone <s@example.invalid>", "a.py")
    assert co.returncode == 2
    assert "Co-authored-by" in co.stderr
    so = commit("[None][feat] x\n\nSigned-off-by: Other <o@example.invalid>", "a.py")
    assert so.returncode == 2
    assert "Signed-off-by" in so.stderr


def test_co_authors_pass_when_the_policy_allows_them(tmp_path, repo):
    script = install_committer(tmp_path / "ws", repo, policy=CommitPolicy(allow_co_authors=True))
    (repo / "a.py").write_text("a\n")
    result = subprocess.run(
        [str(script), "[None][feat] x\n\nCo-authored-by: Pair <p@example.invalid>", "a.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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


def test_with_no_identity_anywhere_the_commit_is_refused(tmp_path, isolated_git_env):
    """No `$USER@nvidia.com` invented: that is an identity no DCO check matches."""
    repo = tmp_path / "bare-identity"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("a\n")
    script = install_committer(tmp_path / "ws2", repo)
    result = subprocess.run([str(script), "m", "a.py"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "no commit identity" in result.stderr


def test_a_hook_that_appends_a_co_author_gets_the_commit_undone(tmp_path, repo, commit):
    hooks = tmp_path / "msg-hooks"
    hooks.mkdir()
    hook = hooks / "commit-msg"
    hook.write_text('#!/bin/sh\nprintf "\\nCo-authored-by: Bot <bot@example.invalid>\\n" >> "$1"\n')
    hook.chmod(0o755)
    _git(repo, "config", "core.hooksPath", str(hooks))
    before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.py").write_text("a\n")
    result = commit("m", "a.py")
    assert result.returncode == 1
    assert "Co-authored-by" in result.stderr
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == before


def test_identity_resolution_prefers_a_complete_override_then_the_repo(tmp_path, repo):
    assert resolve_commit_identity(repo, environ={}).render() == "Test <t@example.invalid>"
    half = {"MV2FLOW_GIT_NAME": "Only Name"}
    assert resolve_commit_identity(repo, environ=half).render() == "Test <t@example.invalid>"
    both = {"MV2FLOW_GIT_NAME": "Op", "MV2FLOW_GIT_EMAIL": "op@example.invalid"}
    assert resolve_commit_identity(repo, environ=both).render() == "Op <op@example.invalid>"


def test_identity_resolution_fails_loudly_without_one(tmp_path, isolated_git_env):
    repo = tmp_path / "noid"
    repo.mkdir()
    _git(repo, "init", "-q")
    with pytest.raises(CommitPolicyError, match="no commit identity"):
        resolve_commit_identity(repo, environ={})


# ------------------------------------------------- preflight trial


def test_the_trial_commit_is_clean_in_a_clean_environment(isolated_git_env):
    identity = CommitIdentity(name="Repo User", email="repo.user@example.invalid")
    assert trial_commit(CommitPolicy(identity=identity)) == []


def test_the_trial_commit_catches_a_global_hook_that_adds_a_trailer(tmp_path, isolated_git_env):
    """What a template cannot vouch for: the environment's own git configuration."""
    hooks = tmp_path / "global-hooks"
    hooks.mkdir()
    hook = hooks / "commit-msg"
    hook.write_text('#!/bin/sh\nprintf "\\nCo-authored-by: Bot <bot@example.invalid>\\n" >> "$1"\n')
    hook.chmod(0o755)
    (isolated_git_env / ".gitconfig").write_text(f"[core]\n\thooksPath = {hooks}\n")
    identity = CommitIdentity(name="Repo User", email="repo.user@example.invalid")
    problems = trial_commit(CommitPolicy(identity=identity))
    assert problems and "Co-authored-by" in " ".join(problems)


def test_the_trial_without_an_identity_is_a_problem():
    assert trial_commit(CommitPolicy()) != []


# ------------------------------------------------- a whole branch


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


def test_commit_policy_block_validation_reports_every_problem():
    errors: list[str] = []
    validate_commit_policy_block(
        {"allowed_paths": ["/abs", "a/../b", "tests/"], "allow_co_authors": "yes", "nope": 1},
        errors,
    )
    joined = "\n".join(errors)
    assert "unknown key" in joined
    assert "must be repo-relative" in joined
    assert "'..'" in joined
    assert "allow_co_authors" in joined
    errors = []
    validate_commit_policy_block(
        {"allowed_paths": ["tensorrt_llm/"], "base_ref": "origin/main"}, errors
    )
    assert errors == []


def test_the_policy_reads_the_task_block():
    policy = CommitPolicy.from_task(
        {
            "trtllm_repo_path": "",
            "commit_policy": {"allowed_paths": ["tensorrt_llm", "setup.py"], "base_ref": "main"},
        },
        identity=CommitIdentity("A", "a@example.invalid"),
    )
    assert policy.allowed_paths == ("tensorrt_llm/", "setup.py")
    assert policy.base_ref == "main"
    assert policy.identity.email == "a@example.invalid"


# ------------------------------------------------- wired into the run


def test_the_cli_refuses_to_start_without_an_identity(tmp_path, isolated_git_env):
    from agent_flow.workflows.trtllm_modeling_v2.cli import _commit_identity

    repo = tmp_path / "noid"
    repo.mkdir()
    _git(repo, "init", "-q")
    with pytest.raises(SystemExit) as exc:
        _commit_identity({"trtllm_repo_path": str(repo)})
    assert exc.value.code == 2


def test_the_cli_takes_the_repositorys_identity(repo, isolated_git_env):
    from agent_flow.workflows.trtllm_modeling_v2.cli import _commit_identity

    assert _commit_identity({"trtllm_repo_path": str(repo)}).render() == "Test <t@example.invalid>"


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
