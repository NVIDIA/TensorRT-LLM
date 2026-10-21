# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the improver leaves besides lessons: patch drafts, known engine defects, reported issues.

A patch is kept only when it fixes the workflow package, carries its failing
scenario as a test, touches nothing else and applies to the code the run
executes; each way of missing that is refused. An ``engine_defect`` lesson
names its upstream fix, and a tree without that fix is told so at preflight.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.agents import lessons as L

from .test_improver_wake import Host

PACKAGE = "agent-flow/agent_flow/workflows/trtllm_modeling_v2"
TESTS = "agent-flow/tests/workflows/trtllm_modeling_v2"


def _git(repo, *args):
    identity = ["-c", "user.name=mv2flow", "-c", "user.email=mv2flow@example.invalid"]
    command = ["git", *identity, "-c", "commit.gpgsign=false", "-C", str(repo), *args]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def flow(tmp_path, monkeypatch):
    """The workflow's code as a checkout of its own, which this process runs."""
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    top = tmp_path / "flow"
    (top / PACKAGE / "run").mkdir(parents=True)
    (top / TESTS).mkdir(parents=True)
    (top / PACKAGE / "run" / "graph.py").write_text("def ready(graph):\n    return []\n")
    (top / TESTS / "test_graph.py").write_text("def test_ready():\n    pass\n")
    (top / "README.md").write_text("the tree\n")
    _git(top, "init", "-q")
    _git(top, "add", "-A")
    _git(top, "commit", "-q", "-m", "base")
    monkeypatch.setattr(L, "FLOW_CODE", (top / PACKAGE).resolve())
    monkeypatch.setattr(L, "RUNNING_HEAD", _git(top, "rev-parse", "HEAD"))
    L._toplevel.cache_clear()
    yield top
    L._toplevel.cache_clear()


def _diff(top, files):
    """The diff that writes ``files`` into the checkout, as ``git diff`` makes it; the checkout is left as it was."""
    for name, text in files.items():
        (top / name).parent.mkdir(parents=True, exist_ok=True)
        (top / name).write_text(text)
    _git(top, "add", "-A")
    diff = _git(top, "diff", "--cached") + "\n"
    _git(top, "reset", "-q", "--hard", "HEAD")
    _git(top, "clean", "-fdq")
    return diff


FIX = {f"{PACKAGE}/run/graph.py": "def ready(graph):\n    return list(graph)\n"}
TEST = {f"{TESTS}/test_ready_lists.py": "def test_lists():\n    assert True\n"}


def _workspace(tmp_path, patch_text=None, name="fix.diff"):
    ws = tmp_path / "ws"
    (ws / "improve" / "patches").mkdir(parents=True, exist_ok=True)
    (ws / "journal.md").write_text("x")
    if patch_text is not None:
        (ws / "improve" / "patches" / name).write_text(patch_text)
    return ws


def _proposal(patch="improve/patches/fix.diff"):
    issue = {
        "suspected": "run/graph.py:ready",
        "proposed_fix": "list the graph's tasks rather than none",
    }
    if patch:
        issue["patch"] = patch
    return {
        "lessons": [
            {
                "scope": "role:planner",
                "kind": "workflow_defect",
                "trigger": "no task ever starts",
                "lesson": "not the graph's fault",
                "evidence": "journal.md",
                "issue": issue,
            }
        ]
    }


# ------------------------------------------------------------------ patch drafts


def test_a_patch_with_its_test_that_applies_is_filed_with_its_issue(tmp_path, flow):
    ws = _workspace(tmp_path, _diff(flow, {**FIX, **TEST}))
    [lesson], [issue], errors = L.validate_proposal(_proposal(), ws)
    assert errors == []
    assert issue.patch == str((ws / "improve" / "patches" / "fix.diff").resolve())
    assert issue.patch_base == L.RUNNING_HEAD
    [line] = L.record_issues(ws, [issue])
    assert line.endswith(f"; patch {lesson.id}.diff applies to {L.RUNNING_HEAD[:12]}")
    filed = ws / "improve" / "patches" / f"{lesson.id}.diff"
    assert filed.is_file() and not (ws / "improve" / "patches" / "fix.diff").exists()
    [stored] = L.load_issues(ws)
    assert stored.patch == str(filed.resolve()) and stored.patch_base == L.RUNNING_HEAD
    # Refreshed without a patch, the issue keeps the one it has.
    _, [again], _ = L.validate_proposal(_proposal(patch=""), ws)
    assert L.record_issues(ws, [again])[0].startswith("refreshed")
    assert L.load_issues(ws)[0].patch == str(filed.resolve())


@pytest.mark.parametrize(
    "files, refused",
    [
        ({**FIX, **TEST, "README.md": "edited\n"}, "changes files outside"),
        (FIX, "has no test"),
        ({**FIX, f"{TESTS}/helpers.py": "x = 1\n"}, "has no test"),
        (TEST, f"changes nothing under {PACKAGE}/"),
    ],
)
def test_a_patch_that_is_not_a_fix_with_its_test_is_refused(tmp_path, flow, files, refused):
    ws = _workspace(tmp_path, _diff(flow, files))
    kept, [issue], [error] = L.validate_proposal(_proposal(), ws)
    assert len(kept) == 1 and issue.patch == "" and issue.patch_base == ""
    assert "its issue is kept without its patch" in error and refused in error


def test_a_patch_that_does_not_apply_to_the_running_code_is_refused(tmp_path, flow):
    # Made on a tree that moved on from the commit this process runs.
    (flow / PACKAGE / "run" / "graph.py").write_text("def ready(graph):\n    return None\n")
    _git(flow, "commit", "-qam", "later")
    patch = _diff(flow, {**FIX, **TEST})
    ws = _workspace(tmp_path, patch)
    _, _, [error] = L.validate_proposal(_proposal(), ws)
    assert f"does not apply to the workflow's code at {L.RUNNING_HEAD[:12]}" in error
    # The checkout itself is neither read nor changed by the check.
    assert _git(flow, "status", "--porcelain") == ""


@pytest.mark.parametrize(
    "where, text, refused",
    [
        ("journal.md", None, "is not a `.diff` under improve/patches/"),
        ("improve/patches/missing.diff", None, "is not a file"),
        ("improve/patches/empty.diff", "", "is 0 bytes"),
        (
            "improve/patches/prose.diff",
            "change ready() to list the tasks\n",
            "not a diff git reads",
        ),
    ],
)
def test_a_patch_that_is_no_diff_in_its_place_is_refused(tmp_path, flow, where, text, refused):
    ws = _workspace(tmp_path)
    if text is not None:
        (ws / where).write_text(text)
    _, [issue], [error] = L.validate_proposal(_proposal(patch=where), ws)
    assert issue.patch == "" and refused in error


def test_no_patch_is_kept_when_the_running_code_is_not_a_checkout(tmp_path, flow, monkeypatch):
    ws = _workspace(tmp_path, _diff(flow, {**FIX, **TEST}))
    monkeypatch.setattr(L, "RUNNING_HEAD", "")
    _, [issue], [error] = L.validate_proposal(_proposal(), ws)
    assert issue.patch == "" and "not a git checkout here" in error


def test_the_improver_checks_its_proposal_before_it_ends(tmp_path, flow, capsys):
    ws = _workspace(tmp_path, _diff(flow, {**FIX, **TEST}))
    (ws / L.PROPOSED_FILENAME).write_text(yaml.safe_dump(_proposal()))
    assert L.main(["check-proposal", str(ws)]) == 0
    assert "kept: 1 lesson(s), 1 issue(s), 1 patch(es)" in capsys.readouterr().out
    (ws / "improve" / "patches" / "fix.diff").write_text(_diff(flow, FIX))
    assert L.main(["check-proposal", str(ws)]) == 1
    out = capsys.readouterr().out
    assert "REFUSED lessons[0]: its issue is kept without its patch" in out
    assert L.main(["check"]) == 2


def test_the_check_command_runs_the_code_this_process_runs(tmp_path):
    ws = _workspace(tmp_path)
    proposal = {"lessons": [{"scope": "role:nobody", "trigger": "t", "lesson": "l"}]}
    (ws / L.PROPOSED_FILENAME).write_text(yaml.safe_dump(proposal))
    run = subprocess.run(
        L.check_command(ws), shell=True, capture_output=True, text=True, timeout=300
    )
    assert run.returncode == 1, run.stderr
    assert "REFUSED lessons[0]" in run.stdout and "no role 'nobody'" in run.stdout


# ------------------------------------------------------------- known engine defects


def _engine_defect(trigger, fix, scope="all"):
    return {
        "scope": scope,
        "kind": "engine_defect",
        "trigger": trigger,
        "lesson": "take the upstream fix",
        "evidence": "journal.md",
        "upstream_fix": fix,
    }


def test_an_engine_defect_names_its_upstream_fix(tmp_path):
    (tmp_path / "journal.md").write_text("x")
    raw = {
        "lessons": [
            _engine_defect("a", "#123"),
            _engine_defect("b", "NVIDIA/TensorRT-LLM#123"),
            _engine_defect("c", "https://github.com/NVIDIA/TensorRT-LLM/pull/123"),
            _engine_defect("d", "4a3ce48dd8"),
            _engine_defect("e", ""),
            _engine_defect("f", "the fla fix"),
            {**_engine_defect("g", "#1"), "fixed_by": "4a3ce48dd8"},
            {**_engine_defect("h", "#1"), "kind": "domain"},
        ]
    }
    kept, _, errors = L.validate_proposal(raw, tmp_path)
    assert [k.trigger for k in kept] == ["a", "b", "c", "d"]
    assert L.upstream_ref(kept[2].upstream_fix) == ("pull", "123")
    assert L.upstream_ref(kept[3].upstream_fix) == ("commit", "4a3ce48dd8")
    assert "names the change that fixed it in `upstream_fix`" in errors[0]
    assert "'the fla fix' is not a commit hash or a pull request" in errors[1]
    assert "an engine fix is `upstream_fix`" in errors[2]
    assert "only an `engine_defect` lesson names an `upstream_fix`" in errors[3]


@pytest.fixture
def engine(tmp_path):
    """An engine checkout: a fix merged as a pull request, one on a branch, one cherry-picked."""
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repo = tmp_path / "engine"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "base")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "[None][fix] keep the scan monotonic (#123)")
    merged = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "side")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "use 64-bit offsets")
    picked = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "a fix nobody took")
    unmerged = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-")
    _git(
        repo,
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        f"offsets\n\n(cherry picked from commit {picked})",
    )
    return repo, merged, picked, unmerged


def test_whether_a_tree_has_an_upstream_fix(tmp_path, engine):
    repo, merged, picked, unmerged = engine
    assert L.fix_in_tree(merged[:10], repo) == "present"
    assert L.fix_in_tree("#123", repo) == "present"
    assert L.fix_in_tree("https://github.com/NVIDIA/TensorRT-LLM/pull/123", repo) == "present"
    assert L.fix_in_tree(picked, repo) == "present"  # cherry-picked with -x
    assert L.fix_in_tree(unmerged, repo) == "absent"
    # A commit that only names the fix, as a revert does, does not bring it.
    _git(repo, "commit", "-q", "--allow-empty", "-m", f"Revert\n\nThis reverts commit {unmerged}.")
    assert L.fix_in_tree(unmerged, repo) == "absent"
    assert L.fix_in_tree("#12", repo) == "absent"  # not #123
    assert L.fix_in_tree("#1234", repo) == "absent"
    assert L.fix_in_tree("#123", tmp_path / "nowhere") == "unknown"


def test_preflight_warns_of_a_known_defect_the_tree_lacks(tmp_path, engine):
    repo, merged, _, unmerged = engine
    host = Host(tmp_path)
    host.task_data = {"trtllm_repo_path": str(repo)}
    lessons, _, errors = L.validate_proposal(
        {
            "lessons": [
                _engine_defect("a scan rises across exact zeros", "#123"),
                _engine_defect("an op's offsets wrap past 4 GiB", unmerged[:12]),
                _engine_defect("an op of another family", "#999", scope="family:other"),
            ]
        },
        host.workspace,
    )
    assert errors == []
    L.LessonStore(host.lessons_dir).merge(lessons, run="r1", evidence_root=host.workspace)
    warning = host._known_defects_warning()
    assert "an op's offsets wrap past 4 GiB" in warning and f"`{unmerged[:12]}`" in warning
    assert "scan rises" not in warning and "another family" not in warning
    assert host.events == [("known engine defects", warning)]
    # On a tree git cannot read, the warning says it could not tell.
    host.task_data = {"trtllm_repo_path": str(tmp_path / "nowhere")}
    assert "git could not tell" in host._known_defects_warning()
    # Nothing to warn of: a tree with every fix, or no store.
    host.task_data = {"trtllm_repo_path": str(repo)}
    host.lessons_dir = tmp_path / "empty"
    assert host._known_defects_warning() == ""
    host.lessons_dir = None
    assert host._known_defects_warning() == ""


def test_a_known_defect_is_shown_with_its_fix_and_keeps_it_when_refreshed(tmp_path):
    (tmp_path / "journal.md").write_text("x")
    store = L.LessonStore(tmp_path / "store")
    [lesson], _, _ = L.validate_proposal({"lessons": [_engine_defect("t", "#123")]}, tmp_path)
    store.merge([lesson], run="a", evidence_root=tmp_path)
    refreshed = L.Lesson.from_dict({**_engine_defect("t", ""), "lesson": "better"})
    store.merge([refreshed], run="b", evidence_root=tmp_path)
    [kept] = store.engine_defects()
    assert (kept.upstream_fix, kept.lesson, kept.seen) == ("#123", "better", 2)
    assert "an engine defect fixed upstream in `#123`" in L.render_block([kept])
    assert kept in store.for_dispatch("smith")


# --------------------------------------------------------------- reported issues


def test_a_reported_flow_issue_is_one_line_each(tmp_path):
    with pytest.raises(ValueError):
        L.report_flow_issue(tmp_path, symptom="  ")
    first = L.report_flow_issue(
        tmp_path, symptom="a no-op planner turn", evidence=["journal.md"], suspected="x.py:f"
    )
    L.report_flow_issue(tmp_path, by="smith k-12", symptom="the spool lost a job")
    assert first["by"] == "driver" and first["evidence"] == ["journal.md"]
    assert [r["by"] for r in L.reported_issues(tmp_path)] == ["driver", "smith k-12"]
    assert L.reported_issues(tmp_path / "nowhere") == []
