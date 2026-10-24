# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lessons store: what the improver may propose, what is kept, and who is shown it."""

from __future__ import annotations

import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.agents import lessons as L
from agent_flow.workflows.trtllm_modeling_v2.run import graph as graph_mod

from .test_workflow import Script, _install, _workflow, planner_graph_path, unit_tasks, write_graph


def _lesson(scope="role:smith", trigger="nvcc: out of memory", lesson="build with -j8", **kw):
    return L.Lesson(scope=scope, trigger=trigger, lesson=lesson, evidence="journal.md", **kw)


def _defect(**kw):
    """A proposed workflow_defect lesson and its issue, as the improver writes them."""
    item = {
        "scope": "role:planner",
        "kind": "workflow_defect",
        "trigger": "TUNE_FAILED: server not healthy after 1800s",
        "lesson": "not the target: the bench started a multi-rank server as one process",
        "evidence": "journal.md",
        "issue": {
            "suspected": "tune/bench.py:run_part",
            "proposed_fix": "start a multi-rank server under mpirun and trtllm-llmapi-launch",
            "evidence": ["journal.md", "logs/server.log"],
        },
    }
    item.update(kw)
    return item


def test_a_proposal_keeps_what_is_traceable_and_names_the_rest(tmp_path):
    (tmp_path / "journal.md").write_text("x")
    raw = {
        "lessons": [
            {"scope": "role:smith", "trigger": "t", "lesson": "l", "evidence": "journal.md"},
            {"scope": "role:nobody", "trigger": "t", "lesson": "l", "evidence": "journal.md"},
            {"scope": "all", "trigger": "t", "lesson": "l", "evidence": "missing.log"},
            {"scope": "host:gb200", "trigger": "", "lesson": "x" * 601, "evidence": "journal.md"},
            {"scope": "all", "trigger": "t", "lesson": "l", "evidence": "../outside"},
        ]
    }
    kept, issues, errors = L.validate_proposal(raw, tmp_path)
    assert [k.scope for k in kept] == ["role:smith"] and issues == []
    assert kept[0].kind == "domain"  # a lesson that names no kind
    assert "no role 'nobody'" in errors[0]
    assert "'missing.log' is not a path in the workspace" in errors[1]
    assert "no `trigger`" in errors[2] and "601 characters" in errors[2]
    assert "lessons[4]" in errors[3]


def test_a_workflow_defect_files_an_issue_and_a_domain_lesson_does_not(tmp_path):
    (tmp_path / "journal.md").write_text("x")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "server.log").write_text("x")
    domain = {
        "scope": "family:qwen3",
        "kind": "domain",
        "trigger": "t",
        "lesson": "l",
        "evidence": "journal.md",
    }
    raw = {
        "lessons": [
            _defect(),
            domain,
            {**domain, "kind": "bug"},
            {**domain, "issue": {"suspected": "run/graph.py:validate", "proposed_fix": "x"}},
            _defect(issue=None),
            _defect(issue=None, fixed_by="4a3ce48dd8"),
            _defect(issue={"suspected": "tune/nowhere.py:run", "evidence": ["gone.log"]}),
            _defect(issue={"suspected": "tune/bench.py", "proposed_fix": "x"}),
            _defect(fixed_by="HEAD~1"),
        ]
    }
    kept, issues, errors = L.validate_proposal(raw, tmp_path)
    assert [(k.kind, k.fixed_by) for k in kept] == [
        ("workflow_defect", ""),
        ("domain", ""),
        ("workflow_defect", "4a3ce48dd8"),  # already fixed: no issue to file
    ]
    [issue] = issues
    assert issue.lesson == kept[0].id and issue.scope == "role:planner"
    assert issue.symptom == kept[0].trigger  # the trigger, unless the issue says otherwise
    assert issue.suspected == "tune/bench.py:run_part"
    assert issue.evidence == ["journal.md", "logs/server.log"]
    assert "kind 'bug' is not one of" in errors[0]
    assert "only a `workflow_defect` lesson files an `issue`" in errors[1]
    assert "files an `issue` (suspected, proposed_fix) unless `fixed_by`" in errors[2]
    assert "'gone.log' is not a path" in errors[3]
    assert "'tune/nowhere.py' is not a file of the workflow" in errors[3]
    assert "no `proposed_fix`" in errors[3]
    assert "'tune/bench.py' is not `<file>:<function>`" in errors[4]
    assert "'HEAD~1' is not a commit hash" in errors[5]


def test_flow_issues_are_filed_refreshed_and_marked_fixed(tmp_path, monkeypatch):
    (tmp_path / "journal.md").write_text("x")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "server.log").write_text("x")
    _, [issue], _ = L.validate_proposal({"lessons": [_defect()]}, tmp_path)
    monkeypatch.setattr(L.time, "strftime", lambda fmt: "2026-01-01")
    assert L.record_issues(tmp_path, [issue]) == [
        "filed [role:planner] tune/bench.py:run_part: TUNE_FAILED: server not healthy after 1800s"
    ]
    path = tmp_path / L.FLOW_ISSUES_FILENAME
    [stored] = yaml.safe_load(path.read_text())["issues"]
    assert stored["evidence"] == [
        str((tmp_path / "journal.md").resolve()),
        str((tmp_path / "logs" / "server.log").resolve()),
    ]
    assert stored["proposed_fix"].startswith("start a multi-rank server")
    assert stored["fixed_by"] == "" and stored["first_seen"] == "2026-01-01"

    # The operator records the fix by hand; a later refresh keeps it.
    path.write_text(path.read_text().replace("fixed_by: ''", "fixed_by: 4a3ce48dd8"))
    monkeypatch.setattr(L.time, "strftime", lambda fmt: "2026-01-02")
    _, [again], _ = L.validate_proposal({"lessons": [_defect()]}, tmp_path)
    assert L.record_issues(tmp_path, [again])[0].startswith("refreshed")
    [issue] = L.load_issues(tmp_path)
    assert (issue.fixed_by, issue.first_seen, issue.last_seen) == (
        "4a3ce48dd8",
        "2026-01-01",
        "2026-01-02",
    )
    # A lesson proposed with its fix marks its issue fixed.
    assert L.record_issues(tmp_path, [], {issue.lesson: "be045f9822"}) == [
        "fixed by be045f9822 [role:planner] tune/bench.py:run_part"
    ]
    assert L.load_issues(tmp_path)[0].fixed_by == "be045f9822"
    # A file the operator broke is left for the operator.
    path.write_text("issues: [", encoding="utf-8")
    assert "left as it is" in L.record_issues(tmp_path, [again])[0]
    assert path.read_text() == "issues: ["


def test_scopes_are_checked_by_kind():
    assert L.scope_problem("all") == ""
    assert L.scope_problem("family:qwen3") == ""
    assert "takes no name" in L.scope_problem("all:x")
    assert "kind must be one of" in L.scope_problem("cluster:x")
    assert "needs a name" in L.scope_problem("host:")


def test_seeing_a_lesson_again_refreshes_it_rather_than_duplicating_it(tmp_path):
    store = L.LessonStore(tmp_path / "store")
    assert store.merge([_lesson()], run="run-a", evidence_root=tmp_path)[0].startswith("added")
    again = _lesson(trigger="NVCC:  out of memory", lesson="build with -j4")
    assert store.merge([again], run="run-b", evidence_root=tmp_path)[0].startswith("refreshed")
    [kept] = store.load("role:smith")
    assert kept.lesson == "build with -j4" and kept.seen == 2 and kept.runs == ["run-a", "run-b"]
    assert kept.evidence == str((tmp_path / "journal.md").resolve())


def test_a_full_scope_drops_its_stalest_lesson(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "MAX_PER_SCOPE", 2)
    store = L.LessonStore(tmp_path)
    for i in range(3):
        monkeypatch.setattr(L.time, "strftime", lambda fmt, i=i: f"2026-01-0{i + 1}")
        store.merge([_lesson(trigger=f"symptom {i}")], run="r", evidence_root=tmp_path)
    assert [x.trigger for x in store.load("role:smith")] == ["symptom 2", "symptom 1"]


def test_a_dispatch_sees_its_own_scopes_only(tmp_path):
    store = L.LessonStore(tmp_path)
    store.merge(
        [
            _lesson(scope="all", trigger="a"),
            _lesson(scope="role:smith", trigger="s"),
            _lesson(scope="role:tuner", trigger="t"),
            _lesson(scope="host:eos", trigger="h"),
            _lesson(scope="family:qwen3", trigger="f"),
        ],
        run="r",
        evidence_root=tmp_path,
    )
    seen = {x.trigger for x in store.for_dispatch("smith", host="eos", family="qwen3")}
    assert seen == {"a", "s", "h", "f"}
    assert {x.trigger for x in store.for_dispatch("smith")} == {"a", "s"}


def test_the_block_leads_with_the_symptom():
    block = L.render_block([_lesson()])
    assert "**When:** nvcc: out of memory" in block and "**Do:** build with -j8" in block
    assert "workflow defect" not in block
    defect = L.render_block([_lesson(kind="workflow_defect")])
    assert "(scope `role:smith`, a workflow defect, seen in 1 run(s)" in defect
    assert L.render_block([]) == ""


def test_a_missing_or_broken_proposal(tmp_path):
    assert L.load_proposal(tmp_path) == ([], [], [])
    (tmp_path / L.PROPOSED_FILENAME).write_text("lessons: [", encoding="utf-8")
    assert "is not YAML" in L.load_proposal(tmp_path)[2][0]
    (tmp_path / L.PROPOSED_FILENAME).write_text(yaml.safe_dump({"x": 1}), encoding="utf-8")
    assert "`lessons` list" in L.load_proposal(tmp_path)[2][0]


def test_the_directory_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(L.LESSONS_ENV, str(tmp_path))
    assert L.default_lessons_dir() == tmp_path
    monkeypatch.delenv(L.LESSONS_ENV)
    assert L.default_lessons_dir() == L.DEFAULT_LESSONS_DIR


def test_a_lesson_reported_wrong_stops_being_shown(tmp_path):
    store = L.LessonStore(tmp_path)
    store.merge([_lesson()], run="r", evidence_root=tmp_path)
    [lesson] = store.load("role:smith")
    assert store.feedback({lesson.id: "helped"}) == [f"helped [role:smith] {lesson.trigger}"]
    store.feedback({lesson.id: "wrong", "nonsense": "maybe"})
    assert store.for_dispatch("smith")  # 1 helped, 1 wrong: still shown
    store.feedback({lesson.id: "wrong"})
    assert store.for_dispatch("smith") == []  # wrong twice, more often than helpful
    assert store.load("role:smith")[0].wrong == 2  # kept for the improver
    assert f"`{lesson.id}`" in L.render_block([lesson])


# ------------------------------------------------------------------ retirement


def _git(repo, *args):
    identity = ["-c", "user.name=mv2flow", "-c", "user.email=mv2flow@example.invalid"]
    command = ["git", *identity, "-c", "commit.gpgsign=false", "-C", str(repo), *args]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def flow_checkout(tmp_path, monkeypatch):
    """The flow code as a checkout: a fix in its history, and one on a branch it never merged."""
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repo = tmp_path / "flow"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "the fix")
    fix = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "elsewhere")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "a fix on another branch")
    side = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "later work")
    monkeypatch.setattr(L, "FLOW_CODE", repo)
    monkeypatch.setattr(L, "RUNNING_HEAD", L._git_head(repo))
    L._is_ancestor.cache_clear()
    yield SimpleNamespace(fix=fix, side=side)
    L._is_ancestor.cache_clear()


def test_a_lesson_whose_fix_is_running_is_retired_not_deleted(tmp_path, flow_checkout):
    store = L.LessonStore(tmp_path / "store")
    lessons = [
        _lesson(trigger="fixed here", kind="workflow_defect", fixed_by=flow_checkout.fix[:10]),
        _lesson(trigger="fixed elsewhere", kind="workflow_defect", fixed_by=flow_checkout.side),
        _lesson(trigger="unknown fix", kind="workflow_defect", fixed_by="0123456789abcdef"),
        _lesson(trigger="open"),
    ]
    changes = store.merge(lessons, run="r", evidence_root=tmp_path)
    assert f"fixed by {flow_checkout.fix[:10]} [role:smith] fixed here; retired" in changes
    elsewhere = f"fixed by {flow_checkout.side} [role:smith] fixed elsewhere"
    assert f"{elsewhere}; not in the running code yet" in changes
    shown = {x.trigger for x in store.for_dispatch("smith")}
    assert shown == {"fixed elsewhere", "unknown fix", "open"}
    assert [x.trigger for x in store.retired()] == ["fixed here"]
    assert len(store.load("role:smith")) == 4
    # A lesson whose fix exists but is not running says so.
    pending = next(x for x in store.for_dispatch("smith") if x.trigger == "fixed elsewhere")
    assert f"fixed in `{flow_checkout.side}` but not in the code this run executes" in (
        L.render_block([pending])
    )


def test_the_improver_retires_a_lesson_and_can_reopen_it(tmp_path, flow_checkout):
    store = L.LessonStore(tmp_path / "store")
    workaround = _lesson(kind="workflow_defect")
    store.merge([workaround], run="a", evidence_root=tmp_path)
    assert store.for_dispatch("smith")
    fixed = _lesson(kind="workflow_defect", fixed_by=flow_checkout.fix)
    store.merge([fixed], run="b", evidence_root=tmp_path)
    assert store.for_dispatch("smith") == [] and store.retired()
    # Proposed again without its fix: the symptom came back on code that has it.
    changes = store.merge([_lesson(kind="workflow_defect")], run="c", evidence_root=tmp_path)
    assert any(c.startswith("reopened [role:smith]") for c in changes)
    assert store.for_dispatch("smith") and store.load("role:smith")[0].fixed_by == ""


def test_the_ancestor_check_is_cached_and_shows_the_lesson_without_git(
    tmp_path, flow_checkout, monkeypatch
):
    calls = []
    run = subprocess.run

    def counting(*args, **kwargs):
        calls.append(args[0])
        return run(*args, **kwargs)

    monkeypatch.setattr(L.subprocess, "run", counting)
    assert L.fix_is_running(flow_checkout.fix) and L.fix_is_running(flow_checkout.fix)
    assert not L.fix_is_running(flow_checkout.side) and not L.fix_is_running("")
    assert len(calls) == 2  # once per commit
    L._is_ancestor.cache_clear()
    monkeypatch.setenv("PATH", str(tmp_path / "no-git-here"))
    assert not L.fix_is_running(flow_checkout.fix)
    assert L._git_head(L.FLOW_CODE) == ""
    monkeypatch.setattr(L, "RUNNING_HEAD", "")
    assert not _lesson(fixed_by=flow_checkout.fix).retired


# --------------------------------------------------------------------- the run


def test_a_run_files_its_flow_issues_retries_in_place_and_hides_retired_lessons(
    tmp_path, task_file, monkeypatch
):
    head = L.RUNNING_HEAD
    if not head:
        pytest.skip("the workflow's code is not a git checkout here")
    lessons_dir = tmp_path / "lessons"
    earlier = tmp_path / "earlier-run"
    earlier.mkdir()
    (earlier / "journal.md").write_text("x")
    L.LessonStore(lessons_dir).merge(
        [
            L.Lesson(
                scope="role:assembler",
                trigger="fast-forward failed: index.lock exists",
                lesson="wait and integrate again",
                evidence="journal.md",
                kind="workflow_defect",
                fixed_by=head[:12],  # a commit is in its own history
            ),
            L.Lesson(
                scope="role:assembler",
                trigger="fla chunk cumsum rises across exact zeros",
                lesson="keep the in-chunk cumsum non-increasing",
                evidence="journal.md",
            ),
        ],
        run="earlier-run",
        evidence_root=earlier,
    )
    planner_prompts: list[str] = []
    improver_prompts: list[str] = []

    def planner(turn, prompt):
        planner_prompts.append(prompt)
        if turn == 2:
            # The improver reads the failure beside the run; the retry waits for its issue.
            deadline = time.monotonic() + 10
            while not (tmp_path / "ws" / L.FLOW_ISSUES_FILENAME).is_file():
                assert time.monotonic() < deadline, "the improver never filed its issue"
                time.sleep(0.01)
        if turn <= 2:
            # The second graph is the first one again: the failed task's retry.
            write_graph(prompt, unit_tasks(planner_graph_path(prompt).parent))
            return {"decision": "GRAPH_UPDATED", "summary": "plan" if turn == 1 else "retry core"}
        return {"decision": "DONE", "summary": "the contract should hold"}

    def assembler(turn, prompt):
        if turn == 1:
            return {"decision": "BLOCKED", "summary": "lost", "blocked_reason": "environment"}
        return {"decision": "UNIT_READY", "summary": "assembled"}

    def improver(turn, prompt):
        improver_prompts.append(prompt)
        if turn > 1:
            return {"decision": "NOTHING_NEW", "summary": "nothing"}
        defect = {
            "scope": "role:planner",
            "kind": "workflow_defect",
            "trigger": "task core ended BLOCKED: the spool lost its job",
            "lesson": "not the task's fault: write it back as pending to run it again",
            "evidence": "journal.md",
            "issue": {
                "suspected": "run/graph.py:apply_revision",
                "proposed_fix": "let the orchestrator retry an infrastructure failure itself",
            },
        }
        proposal = yaml.safe_dump({"lessons": [defect]})
        (tmp_path / "ws" / L.PROPOSED_FILENAME).write_text(proposal, encoding="utf-8")
        return {"decision": "LESSONS_PROPOSED", "summary": "one workflow defect"}

    workflow = _workflow(tmp_path, lessons_dir=lessons_dir)
    workflow.improver_batch_s = 0.0
    calls = _install(
        monkeypatch, workflow, Script(None, planner=planner, assembler=assembler, improver=improver)
    )
    with workflow:
        # The flow's own measuring run: its improver turns while the run works.
        _measuring(task_file)
        workflow.run(task_file)

    ws = tmp_path / "ws"
    core = graph_mod.Graph.load(ws / graph_mod.GRAPH_FILENAME).by_id()["core"]
    assert (core.status, core.attempts) == ("done", 2)
    [issue] = L.load_issues(ws)
    assert issue.suspected == "run/graph.py:apply_revision" and issue.scope == "role:planner"
    assert issue.evidence == [str((ws / "journal.md").resolve())]
    journal = (ws / "journal.md").read_text()
    assert "**flow issues** — filed [role:planner] run/graph.py:apply_revision" in journal
    assert journal.count("**lessons retired**") == 1
    assert "fast-forward failed: index.lock exists" in journal
    assemblers = [p for name, p in calls if name == "assembler"]
    assert all("index.lock" not in p and "cumsum rises" in p for p in assemblers)
    assert "a workflow defect" in planner_prompts[-1]
    assert "**When:** task core ended BLOCKED" in planner_prompts[-1]
    assert f"at commit `{head}`" in improver_prompts[0]
    retired = improver_prompts[0].split("## Retired lessons", 1)[1]
    assert "index.lock exists" in retired and f"fixed by `{head[:12]}`" in retired


def _measuring(task_file):
    """Mark the task a run that measures the workflow (``knowledge: measurement``)."""
    spec = yaml.safe_load(task_file.read_text(encoding="utf-8"))
    spec["knowledge"] = "measurement"
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
