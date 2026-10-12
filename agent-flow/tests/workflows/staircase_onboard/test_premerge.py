# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-merge pack, against real git repositories in temporary directories.

pre-commit, pytest and the GitHub API are stood in for by small fakes; git is
real, because every step here is a statement about what git will record.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from agent_flow.workflows.staircase_onboard.committer import CommitIdentity
from agent_flow.workflows.staircase_onboard.premerge import ci_case, pack
from agent_flow.workflows.staircase_onboard.premerge.copyright import check_copyright
from agent_flow.workflows.staircase_onboard.premerge.lint import (
    LintError,
    classify_python_change,
    run_lint,
    writable_pre_commit_home,
)
from agent_flow.workflows.staircase_onboard.premerge.pr_open import (
    PrOpenError,
    PrRequest,
    open_pr,
    read_token,
    validate,
)
from agent_flow.workflows.staircase_onboard.premerge.resign import ResignError, resign_branch
from agent_flow.workflows.staircase_onboard.premerge.split import (
    SplitError,
    parse_stack,
    split_branch,
)

AUTHOR = "Repo User <repo.user@example.invalid>"
IDENTITY = CommitIdentity(name="Repo User", email="repo.user@example.invalid")


def _git(repo, *args, env=None, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check, env=env
    ).stdout


def _env(name="Repo User", email="repo.user@example.invalid", committer=None):
    env = dict(os.environ)
    cn, ce = committer or (name, email)
    env.update(
        GIT_AUTHOR_NAME=name,
        GIT_AUTHOR_EMAIL=email,
        GIT_COMMITTER_NAME=cn,
        GIT_COMMITTER_EMAIL=ce,
    )
    return env


def _commit(repo, message, files, *, signoff=True, env=None):
    for path, content in files.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            _git(repo, "rm", "-q", path)
            continue
        target.write_text(content)
        _git(repo, "add", path)
    args = ["commit", "-q", "-m", message] + (["-s"] if signoff else [])
    _git(repo, *args, env=env or _env())
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", "Repo User")
    _git(path, "config", "user.email", "repo.user@example.invalid")
    _git(path, "config", "commit.gpgsign", "false")
    _commit(path, "seed", {"README.md": "seed\n"})
    _git(path, "checkout", "-q", "-b", "work")
    return path


# ------------------------------------------------------------------ AST proof


@pytest.mark.parametrize(
    "before,after,expected",
    [
        ("x = 1\n", "x = 1\n", "unchanged"),
        ("x=1\n", "x = 1\n", "identical"),
        ("x = 'a'\n", 'x = "a"\n', "identical"),
        ('"""Doc  string\n   here."""\n', '"""Doc string here."""\n', "docstring-whitespace"),
        ("import os\nimport sys\n", "import sys\nimport os\n", "import-order"),
        ("from a import x, y\n", "from a import y\nfrom a import x\n", "import-order"),
        ("import os\nimport sys\nx = 1\n", "import sys\nx = 1\n", "import-removed"),
        ("x = 1\n", "x = 2\n", "different"),
        ("import os\n", "import os\nimport sys\n", "different"),
        ("x = 1\n", "def (:\n", "unparsable"),
    ],
)
def test_the_ast_proof_classifies_what_formatters_do(before, after, expected):
    assert classify_python_change(before, after) == expected


# ------------------------------------------------------------------ lint


def _fake_pre_commit(tmp_path, body):
    """A `pre-commit` that runs ``body`` over every file and records its env."""
    script = tmp_path / "fake-pre-commit"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            echo "PRE_COMMIT_HOME=$PRE_COMMIT_HOME SKIP=${{SKIP:-}}" >> {tmp_path}/pre-commit-env.log
            shift; shift; shift   # run --show-diff-on-failure --files
            rc=0
            for f in "$@"; do
              before=$(cat "$f")
              {body}
              [ "$before" != "$(cat "$f")" ] && rc=1
            done
            exit $rc
            """
        )
    )
    script.chmod(0o755)
    return str(script)


def test_lint_commits_formatting_only_when_pass_two_is_clean(tmp_path, repo):
    _commit(repo, "feature", {"tensorrt_llm/a.py": "x = 1   \n", "notes.txt": "n\n"})
    base = _git(repo, "rev-parse", "main").strip()
    fake = _fake_pre_commit(tmp_path, "sed -i 's/[[:space:]]*$//' \"$f\"")
    result = run_lint(
        repo,
        base=base,
        log_dir=tmp_path / "log",
        pre_commit=fake,
        python="/nonexistent/python",
        identity=IDENTITY,
    )
    assert result.status == "committed", result.render()
    assert result.rewritten == ["tensorrt_llm/a.py"]
    assert result.proof.summary() == "1 identical"
    assert "type-check" in result.skipped_hooks
    message = _git(repo, "log", "-1", "--format=%B")
    assert "AST proof: 1 identical" in message
    assert _git(repo, "log", "-1", "--format=%an <%ae>").strip() == AUTHOR
    assert (
        _git(repo, "log", "-1", "--format=%(trailers:key=Signed-off-by,valueonly)").strip()
        == AUTHOR
    )
    assert _git(repo, "show", "--name-only", "--format=", "HEAD").split() == ["tensorrt_llm/a.py"]
    assert "SKIP=type-check" in (tmp_path / "pre-commit-env.log").read_text()


def test_lint_restores_the_tree_and_commits_nothing_when_pass_two_fails(tmp_path, repo):
    _commit(repo, "feature", {"a.py": "x = 1   \n"})
    head = _git(repo, "rev-parse", "HEAD").strip()
    fake = _fake_pre_commit(
        tmp_path, "sed -i 's/[[:space:]]*$//' \"$f\"; grep -q 'x' \"$f\" && rc=1"
    )
    result = run_lint(repo, base="main", log_dir=tmp_path / "log", pre_commit=fake, python="/x")
    assert result.status == "pass2-failed"
    assert not result.ok
    assert _git(repo, "rev-parse", "HEAD").strip() == head
    assert _git(repo, "status", "--porcelain", "--untracked-files=no").strip() == ""
    assert (tmp_path / "log" / "formatting.patch").read_text().strip()


def test_lint_refuses_a_formatting_commit_that_changes_semantics(tmp_path, repo):
    _commit(repo, "feature", {"a.py": "x = 1\n"})
    head = _git(repo, "rev-parse", "HEAD").strip()
    fake = _fake_pre_commit(tmp_path, "sed -i 's/x = 1/x = 2/' \"$f\"")
    result = run_lint(repo, base="main", log_dir=tmp_path / "log", pre_commit=fake, python="/x")
    assert result.status == "semantic-change"
    assert result.proof.unexplained == [("a.py", "different")]
    assert _git(repo, "rev-parse", "HEAD").strip() == head
    assert (repo / "a.py").read_text() == "x = 1\n"


def test_lint_dry_run_proves_and_restores(tmp_path, repo):
    _commit(repo, "feature", {"a.py": "x=1\n"})
    fake = _fake_pre_commit(tmp_path, "sed -i 's/x=1/x = 1/' \"$f\"")
    result = run_lint(
        repo, base="main", log_dir=tmp_path / "log", pre_commit=fake, python="/x", dry_run=True
    )
    assert result.status == "dry-run"
    assert (repo / "a.py").read_text() == "x=1\n"
    assert "identical" in (tmp_path / "log" / "ast-proof.md").read_text()


def test_lint_refuses_a_dirty_tree(tmp_path, repo):
    _commit(repo, "feature", {"a.py": "x = 1\n"})
    (repo / "a.py").write_text("x = 3\n")
    with pytest.raises(LintError, match="tracked files are modified"):
        run_lint(repo, base="main", log_dir=tmp_path / "log", pre_commit="true")


def test_a_read_only_pre_commit_store_gets_a_writable_copy_of_its_database(tmp_path):
    store = tmp_path / "home" / ".cache" / "pre-commit"
    store.mkdir(parents=True)
    (store / "db.db").write_text("db")
    store.chmod(0o555)
    try:
        if os.access(store, os.W_OK):
            pytest.skip("running as a user that can write a 0555 directory")
        home = writable_pre_commit_home(tmp_path / "log", home=tmp_path / "home")
        assert home == tmp_path / "log" / "pre-commit-home"
        assert (home / "db.db").read_text() == "db"
    finally:
        store.chmod(0o755)
    assert writable_pre_commit_home(tmp_path / "log2", home=tmp_path / "home") == store


# ------------------------------------------------------------------ CI cases


_TEST_DB = """\
version: 0.0.1
l0_fake_multi_gpus:
- condition:
    ranges:
      system_gpu_count:
        gte: 4
        lte: 4
    terms:
      stage: post_merge
      backend: pytorch
  tests:
  - unittest/_torch/attention TIMEOUT (120)
  - accuracy/test_modeling_v2_fam.py::TestFam::test_gsm8k[dep4] TIMEOUT (90)
  - accuracy/test_modeling_v2_fam.py::TestFam::test_gsm8k[tp4]
"""


@pytest.fixture
def ci_repo(repo):
    db = repo / ci_case.TEST_DB_DIR
    db.mkdir(parents=True)
    (db / "l0_fake_multi_gpus.yml").write_text(_TEST_DB)
    (repo / "tests/integration/defs").mkdir(parents=True)
    _commit(repo, "test-db", {f"{ci_case.TEST_DB_DIR}/l0_fake_multi_gpus.yml": _TEST_DB})
    return repo


def test_test_db_lines_are_found_with_their_block(ci_repo):
    entries = ci_case.find_test_db_entries(ci_repo, "test_modeling_v2_fam")
    assert [e.test_id.rsplit("::", 1)[-1] for e in entries] == [
        "test_gsm8k[dep4]",
        "test_gsm8k[tp4]",
    ]
    assert entries[0].timeout_minutes == 90
    assert entries[0].terms == {"stage": "post_merge", "backend": "pytorch"}
    assert entries[0].system_gpu_count == (4, 4)


def test_the_stage_comes_from_the_repositorys_own_mapper(ci_repo):
    mapper = ci_repo / ci_case.STAGE_MAPPER
    mapper.parent.mkdir(parents=True, exist_ok=True)
    mapper.write_text("print('FAKE-4_GPUs-PyTorch-Post-Merge-1')\n")
    stages = ci_case.resolve_stages(ci_repo, "accuracy/x.py::T::t")
    assert stages == ["FAKE-4_GPUs-PyTorch-Post-Merge-1"]


def test_the_ci_script_runs_the_line_as_ci_does_and_records_its_tree(tmp_path, ci_repo):
    cases = ci_case.build_cases(
        ci_repo,
        "test_gsm8k[dep4]",
        out_root=tmp_path / "out",
        stage="FAKE-Stage-1",
        launcher="mpirun -n 4 trtllm-llmapi-launch",
        models_root="/models",
    )
    assert len(cases) == 1
    script = ci_case.write_case(cases[0])
    body = script.read_text()
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert "--test-prefix=FAKE-Stage-1" in body
    assert "--apply-test-list-correction" in body
    assert "TIMEOUT (90)" in body  # the line itself is the test list
    assert "env -u SLURM_PROCID" in body
    assert "export LLM_MODELS_ROOT=/models" in body
    assert f"cd {ci_repo}/tests/integration/defs" in body
    assert body.count("rev-parse --short=10 HEAD") == 2  # at start and at end

    # Run it, with stand-ins for the launcher and pytest that record their argv.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("mpirun", "pytest"):
        stub = bin_dir / name
        stub.write_text(
            f'#!/usr/bin/env bash\necho "{name} $*" >> {tmp_path}/argv.log\n'
            + ('shift 3; exec "$@"\n' if name == "mpirun" else "exit 0\n")
        )
        stub.chmod(0o755)
    (bin_dir / "trtllm-llmapi-launch").write_text('#!/usr/bin/env bash\nexec "$@"\n')
    (bin_dir / "trtllm-llmapi-launch").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    run = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "CI_CASE_START" in run.stdout and "CI_CASE_END" in run.stdout
    record = json.loads((Path(cases[0].out_dir) / "ci_case.json").read_text())
    assert record["rc"] == 0 and record["same_tree"] is True
    argv = (tmp_path / "argv.log").read_text()
    assert "--test-list=" in argv and "--unittest-markexpr=not cpu_only" in argv
    listed = (Path(cases[0].out_dir) / "test_list.txt").read_text().strip()
    assert listed == "accuracy/test_modeling_v2_fam.py::TestFam::test_gsm8k[dep4] TIMEOUT (90)"


def test_premerge_ci_insists_on_exactly_one_line(tmp_path, ci_repo):
    from agent_flow.workflows.staircase_onboard.premerge.cli import main

    assert main(["ci", "--repo", str(ci_repo), "--case", "test_gsm8k", "--stage", "S"]) == 2
    assert (
        main(
            [
                "ci",
                "--repo",
                str(ci_repo),
                "--case",
                "test_gsm8k[tp4]",
                "--stage",
                "S",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )


# ------------------------------------------------------------------ copyright


def test_copyright_finds_missing_and_stale_headers(repo):
    header = "# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.\n"
    _commit(repo, "old file", {"old.py": header + "x = 1\n"})
    _git(repo, "branch", "-f", "main", "HEAD")
    fresh = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.\n"
    _commit(
        repo,
        "changes",
        {
            "old.py": header + "x = 2\n",
            "new_no_header.py": "y = 1\n",
            "new_ok.py": fresh + "z = 1\n",
            "doc.md": "not source\n",
        },
    )
    report = check_copyright(repo, "main", "HEAD", year=2026)
    assert report.missing == [("A", "new_no_header.py")]
    assert [p for _, p, _ in report.stale] == ["old.py"]
    assert not report.passed


# ------------------------------------------------------------------ split


def test_split_by_path_reproduces_the_branch_and_drops_the_bindings_link(tmp_path, repo):
    base = _git(repo, "rev-parse", "HEAD").strip()
    _commit(repo, "workflow", {"agent-flow/w.py": "w = 1\n"})
    _commit(repo, "model", {"tensorrt_llm/m.py": "m = 1\n"})
    _commit(repo, "both", {"agent-flow/w.py": "w = 2\n", "tensorrt_llm/m.py": "m = 2\n"})
    build = tmp_path / "build"
    build.mkdir()
    (repo / "tensorrt_llm" / "bindings").symlink_to(build)
    _git(repo, "add", "tensorrt_llm/bindings")
    _git(repo, "commit", "-q", "-s", "-m", "accident", env=_env())
    _git(repo, "rm", "-q", "--cached", "tensorrt_llm/bindings")
    _git(repo, "commit", "-q", "-s", "-m", "untrack the link", env=_env())

    stacks = [parse_stack("model=pr-a:*"), parse_stack("flow=pr-b:agent-flow/,AGENTS.md")]
    result = split_branch(
        repo, base=base, src="work", stacks=stacks, forbid={"model": "import agent_flow"}
    )
    assert result.ok, result.render()
    assert result.counts == {"model": 2, "flow": 2}
    assert result.split_commits == 1
    assert result.dropped_commits == 2
    assert _git(repo, "ls-tree", "-r", "--name-only", "pr-a").split() == [
        "README.md",
        "tensorrt_llm/m.py",
    ]
    assert _git(repo, "ls-tree", "-r", "--name-only", "pr-b").split() == [
        "README.md",
        "agent-flow/w.py",
    ]
    # Attribution and sign-off survive the replay.
    assert (
        _git(
            repo, "log", "-1", "--format=%an <%ae>|%(trailers:key=Signed-off-by,valueonly)", "pr-b"
        ).strip()
        == f"{AUTHOR}|{AUTHOR}"
    )

    stacked = split_branch(
        repo,
        base=base,
        src="work",
        stacks=[parse_stack("model=pr-a2:*"), parse_stack("flow=pr-b2:agent-flow/")],
        stacked=True,
    )
    assert stacked.ok, stacked.render()
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", "pr-a2", "pr-b2"]
    )
    assert ancestry.returncode == 0
    assert _git(repo, "rev-parse", "pr-b2^{tree}") == _git(repo, "rev-parse", "work^{tree}")


def test_a_forbidden_import_fails_the_split(repo):
    base = _git(repo, "rev-parse", "HEAD").strip()
    _commit(repo, "leak", {"tensorrt_llm/m.py": "import agent_flow\n", "agent-flow/w.py": "w\n"})
    result = split_branch(
        repo,
        base=base,
        src="work",
        stacks=[parse_stack("model=x-a:*"), parse_stack("flow=x-b:agent-flow/")],
        forbid={"model": r"(^|\s)import agent_flow"},
    )
    assert not result.ok
    assert any("no match" in label and not ok for label, ok, _ in result.checks)


def test_a_path_no_stack_claims_is_an_error(repo):
    base = _git(repo, "rev-parse", "HEAD").strip()
    _commit(repo, "x", {"other/z.py": "z\n"})
    with pytest.raises(SplitError, match="no stack claims"):
        split_branch(repo, base=base, src="work", stacks=[parse_stack("flow=y-b:agent-flow/")])


def test_stack_specs_are_validated():
    with pytest.raises(SplitError):
        parse_stack("no-branch")
    with pytest.raises(SplitError):
        parse_stack("a=b:*,x/")


# ------------------------------------------------------------------ resign


def test_resign_signs_each_commit_as_its_own_author_on_a_new_branch(repo):
    base = _git(repo, "rev-parse", "HEAD").strip()
    signed = _commit(repo, "signed", {"a.py": "a\n"})
    unsigned = _commit(
        repo,
        "unsigned by the orchestrator",
        {"b.py": "b\n"},
        signoff=False,
        env=_env("login", "login@host.invalid", committer=("Operator", "op@example.invalid")),
    )
    after = _commit(repo, "after", {"c.py": "c\n"})
    result = resign_branch(repo, src="work", new_branch="work-signed", base="main")
    assert result.ok
    assert result.unsigned_left == 0
    assert [(o, n == o, s) for o, n, s, _ in result.mapping][:1] == [(signed, True, False)]
    assert result.mapping[1][0] == unsigned and result.mapping[1][2] is True
    assert result.mapping[2][0] == after and result.mapping[2][1] != after  # descendant rewritten
    new_unsigned = result.mapping[1][1]
    fmt = "%an <%ae>|%cn <%ce>|%(trailers:key=Signed-off-by,valueonly)"
    an, cn, so = _git(repo, "log", "-1", f"--format={fmt}", new_unsigned).strip().split("|")
    assert an == cn == so.strip() == "login <login@host.invalid>"
    assert _git(repo, "rev-parse", "work").strip() == after  # source untouched
    with pytest.raises(ResignError, match="exists"):
        resign_branch(repo, src="work", new_branch="work-signed", base="main")
    assert result.base == base
    assert "| 2 |" in result.render_markdown("work")


# ------------------------------------------------------------------ PR opener


def _request(**kw):
    defaults = dict(
        upstream="NVIDIA/TensorRT-LLM",
        owner="someone",
        branch="user/someone/x",
        title="[TRTLLM-1][feat] staircase: x",
        body="body\n",
    )
    return PrRequest(**{**defaults, **kw})


def test_the_pr_opener_validates_the_title_and_body():
    assert validate(_request()) == []
    assert validate(_request(title="feat: x"))
    assert validate(_request(body="x\n\nCo-authored-by: Bot <b@x>\n"))
    with pytest.raises(PrOpenError):
        open_pr(_request(title="bad title"))


def test_the_pr_opener_is_a_dry_run_that_touches_no_network_by_default():
    def opener(_request):
        raise AssertionError("the network was touched")

    report = open_pr(_request(), opener=opener)
    assert report["dry_run"] is True
    assert report["payload"]["draft"] is True
    assert report["payload"]["head"] == "someone:user/someone/x"


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_the_pr_opener_posts_only_with_yes_and_never_reports_the_token():
    calls = []

    def opener(request):
        calls.append((request.get_method(), request.full_url, request.data))
        if request.get_method() == "POST":
            return _Response(
                {
                    "html_url": "https://github.com/x/pull/1",
                    "number": 1,
                    "state": "open",
                    "draft": True,
                }
            )
        if "/branches/" in request.full_url:
            return _Response({"commit": {"sha": "abcdef1234567890"}})
        return _Response([])

    checked = open_pr(_request(), check_remote=True, token="SECRET-TOKEN", opener=opener)
    assert [m for m, _, _ in calls] == ["GET", "GET"]
    assert checked["dry_run"] is True
    created = open_pr(_request(), yes=True, token="SECRET-TOKEN", opener=opener)
    assert created["url"] == "https://github.com/x/pull/1"
    assert [m for m, _, _ in calls].count("POST") == 1
    assert json.loads(calls[-1][2])["draft"] is True
    assert "SECRET-TOKEN" not in json.dumps(created) + json.dumps(checked)


def test_the_pr_opener_refuses_when_a_pr_is_already_open():
    def opener(request):
        if "/branches/" in request.full_url:
            return _Response({"commit": {"sha": "abc"}})
        return _Response([{"html_url": "https://github.com/x/pull/9"}])

    with pytest.raises(PrOpenError, match="already"):
        open_pr(_request(), yes=True, token="t", opener=opener)


def test_the_token_comes_from_the_environment_or_ghs_hosts_file(tmp_path):
    assert read_token({"GH_TOKEN": "a"}) == "a"
    config = tmp_path / "gh"
    config.mkdir()
    (config / "hosts.yml").write_text(
        "github.com:\n  user: me\n  users:\n    me:\n      oauth_token: from-hosts\n"
    )
    assert read_token({"GH_CONFIG_DIR": str(config)}) == "from-hosts"
    with pytest.raises(PrOpenError):
        read_token({"GH_CONFIG_DIR": str(tmp_path / "none")})


# ------------------------------------------------------------------ pack and CLI


def test_the_pack_reports_the_audit_the_headers_and_the_ci_lines(tmp_path, ci_repo):
    _commit(ci_repo, "unsigned", {"tensorrt_llm/m.py": "m\n"}, signoff=False)
    report = pack.build_pack(
        ci_repo, test_pattern="test_modeling_v2_fam", base_ref="main", year=2026
    )
    assert len(report.hygiene) == 1 and "no Signed-off-by" in report.hygiene[0].problem
    assert report.copyright is not None and report.copyright.missing == [("A", "tensorrt_llm/m.py")]
    assert len(report.ci_entries) == 2
    path = pack.write_pack(report, tmp_path / "reports")
    text = path.read_text()
    assert "premerge resign" in text  # the remedy is the first remaining step
    assert "test_gsm8k[dep4]" in text


def test_the_pack_without_a_base_says_so(tmp_path):
    lone = tmp_path / "lone"
    lone.mkdir()
    _git(lone, "init", "-q", "-b", "trunk")
    _commit(lone, "seed", {"x": "x\n"})
    report = pack.build_pack(lone, base_ref="nope")
    assert report.base == ""
    assert "no base" in report.render_markdown()


def test_the_cli_dispatches_premerge(repo):
    from agent_flow.workflows.staircase_onboard import cli

    _commit(repo, "unsigned", {"z.py": "z\n"}, signoff=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["premerge", "hygiene", "--repo", str(repo), "--base", "main"])
    assert exc.value.code == 1
    with pytest.raises(SystemExit) as exc:
        cli.main(["premerge", "hygiene", "--repo", str(repo), "--base", "HEAD"])
    assert exc.value.code == 0
