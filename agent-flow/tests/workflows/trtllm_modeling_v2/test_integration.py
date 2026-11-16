# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 integration behavior and regression checks."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import session as session_mod
from agent_flow.workflows.trtllm_modeling_v2.checks import verdict as check_result
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import pintree
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.git import (
    CommitIdentity,
    CommitPolicy,
    git_output,
)
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.execution import handoff as r
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import (
    TaskResult,
    decision_file_instruction,
)
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.records.journal import Journal
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import (
    NATIVE_BASE,
    REBASE_RETRIES,
    Integration,
    worktree_block,
)

from ._helpers import TARGET_REL, Script, _git, _install, _workflow, passing_shell, seed_graph

_integration_ENV = {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}


def _integration_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _integration_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _integration_git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("A = 1\n")
    (repo / "b.py").write_text("B = 1\n")
    _integration_git(repo, "add", ".")
    _integration_git(repo, "commit", "-q", "-m", "base")
    return repo


def _integration_commit(tree: Path, name: str, text: str, message: str) -> None:
    (tree / name).write_text(text)
    _integration_git(tree, "commit", "-q", "-am", f"{message}\n\nSigned-off-by: T <t@example.com>")


def test_uncommitted_work_or_a_commit_that_breaks_the_rules_goes_back_to_its_worker(tmp_path):
    """A plain `git commit` in a worktree is audited once, before it can land."""
    repo = _integration_repo(tmp_path)
    policy = CommitPolicy(identity=CommitIdentity("T", "t@example.com"), allowed_paths=("a.py",))
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _integration_ENV, policy=policy)
    tree = integration.open("k1")
    (tree / "a.py").write_text("A = 2\n")
    outcome = integration.land("k1")
    assert not outcome.merged and "uncommitted changes" in outcome.detail
    assert not outcome.base  # nothing for a rebase to resolve
    _integration_git(tree, "commit", "-q", "-am", "unsigned")
    outcome = integration.land("k1")
    assert not outcome.merged and outcome.repairable
    assert "no Signed-off-by" in outcome.detail and "rebase --signoff" in outcome.detail
    assert (repo / "a.py").read_text() == "A = 1\n"
    _integration_git(tree, "rebase", "-q", "--signoff", _integration_git(repo, "rev-parse", "HEAD"))
    assert integration.land("k1").merged


def test_a_branch_that_changes_native_code_does_not_land_and_says_what_the_run_does_instead(
    tmp_path,
):
    """Nothing inside a run rebuilds the one build its trees link: such work ends blocked."""
    repo = _integration_repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _integration_ENV)
    tree = integration.open("k1")
    (tree / "kernel.cu").write_text("// a kernel\n")
    (tree / "CMakeLists.txt").write_text("add_library(k kernel.cu)\n")
    _integration_git(tree, "add", ".")
    _integration_commit(tree, "a.py", "A = 2\n", "a kernel and its caller")
    outcome = integration.land("k1")
    # It goes back to its worker, who takes the native change out or ends blocked on it.
    assert not outcome.merged and outcome.repairable
    assert "change native code (`CMakeLists.txt`, `kernel.cu`)" in outcome.detail
    assert NATIVE_BASE in outcome.detail and (repo / "a.py").read_text() == "A = 1\n"
    # The worker read the same text when its turn began, and so does who reads that prompt.
    assert NATIVE_BASE in worktree_block(tree, repo, scratch=False)
    assert "ends `blocked`" in NATIVE_BASE and "rebuilds the base" in NATIVE_BASE
    _integration_git(tree, "rm", "-q", "kernel.cu", "CMakeLists.txt")
    signed = "python only\n\nSigned-off-by: T <t@example.com>"
    _integration_git(tree, "commit", "-q", "-m", signed)
    assert integration.land("k1").merged and (repo / "a.py").read_text() == "A = 2\n"


class _Journal:
    def __init__(self):
        self.events: list[str] = []

    def event(self, kind, text):
        self.events.append(text)


class _integration_Stage(ExecutionStage):
    """What a task's worktree turn uses of the workflow, with a scripted worker."""

    def __init__(self, workspace, integration, worker):
        super().__init__(
            SimpleNamespace(workspace=workspace, _tls=threading.local(), journal=_Journal())
        )
        self._integration = integration
        self.worker = worker
        #: The worktree block each turn was given, the note on a rebase included.
        self.blocks: list[str] = []

    def _work_task_body(self, task):
        self.blocks.append(self._tls.worktree)
        return TaskResult(self.worker(len(self.blocks)))


def test_a_branch_another_run_left_under_a_tasks_name_is_set_aside_not_landed(tmp_path):
    """A second run, started over in a new workspace, landed two commits of the run that was stopped."""
    repo = _integration_repo(tmp_path)
    # What a stopped run leaves in the checkout: its task's branch, ahead of the head.
    _integration_git(repo, "checkout", "-q", "-b", "mv2/run/k1")
    _integration_commit(repo, "a.py", "A = 'the stopped run'\n", "[None][feat] an attempt")
    left = _integration_git(repo, "rev-parse", "--short", "HEAD")
    _integration_git(repo, "checkout", "-q", "main")
    said: list[str] = []
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _integration_ENV, log=said.append)
    tree = integration.open("k1")
    # This run's task starts at the head, and the other run's commits are kept by name.
    assert (tree / "a.py").read_text() == "A = 1\n"
    assert _integration_git(repo, "rev-parse", "mv2/run/k1") == _integration_git(
        repo, "rev-parse", "main"
    )
    assert _integration_git(repo, "rev-parse", "--short", f"mv2/run/k1.left-{left}") == left
    assert any(f"kept as `mv2/run/k1.left-{left}`" in line for line in said)


def test_a_conflicting_task_goes_back_to_its_worker_and_then_integrates(tmp_path):
    repo = _integration_repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _integration_ENV)
    first, second = integration.open("k1"), integration.open("k2")
    _integration_commit(first, "a.py", "A = 2\n", "a")
    # k2's own change predates k1's landing, so its worktree keeps its base.
    _integration_commit(second, "a.py", "A = 3\n", "a again")
    assert integration.land("k1").merged
    head = _integration_git(repo, "rev-parse", "HEAD")

    def worker(turn):
        if turn == 1:
            return "ready"
        # What the note asks for: rebase onto the head, keep both changes, continue.
        rebased = subprocess.run(["git", "-C", str(second), "rebase", head], capture_output=True)
        assert rebased.returncode != 0
        (second / "a.py").write_text("A = 2\nA3 = 3\n")
        _integration_git(second, "add", "a.py")
        _integration_git(second, "-c", "core.editor=true", "rebase", "--continue")
        return "ready"

    stage = _integration_Stage(tmp_path / "ws", integration, worker)
    assert stage._work_task_turn(g.Task(id="k2", kind="feature", goal="mtp")).decision == "ready"
    assert (repo / "a.py").read_text() == "A = 2\nA3 = 3\n"
    assert _integration_git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"
    assert not second.exists()  # integrated, so closed
    first_block, rebase_turn = stage.blocks
    assert "did not integrate" not in first_block
    assert rebase_turn.startswith(first_block)
    assert "did not integrate" in rebase_turn and f"rebase {head}" in rebase_turn
    assert "`a.py`" in rebase_turn and "`blocked`" in rebase_turn
    events = stage.workflow.journal.events
    assert "conflicts with the integration head" in events[0]
    # The repair is the task's next attempt: one counter, one budget.
    assert events[1] == f"k2: back to its assembler to rebase onto {head[:10]} and resolve " + (
        f"(1 of {REBASE_RETRIES}) (attempt 2 of {g.MAX_ATTEMPTS})"
    )
    assert events[2].startswith("k2: integrated")


TRACKED = (
    "README.md",
    "setup.py",
    "tensorrt_llm/__init__.py",
    "tensorrt_llm/models/modeling.py",
    "tests/unittest/_torch/test_modeling.py",
    "tests/integration/test_lists/l0.yml",
    "examples/llm-api/quickstart_advanced.py",
    "cpp/CMakeLists.txt",
    "cpp/kernels/rmsnorm.cu",
    "3rdparty/lib/lib.h",
    "docs/guide/page.md",
    "scripts/build.py",
)


def _integration_whole_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _integration_git(repo, "init", "-q", "-b", "main")
    for path in TRACKED:
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(f"{path} 1\n")
    (repo / ".gitignore").write_text("tensorrt_llm/*.so\n")
    _integration_git(repo, "add", ".")
    _integration_git(repo, "commit", "-q", "-m", "base")
    (repo / "tensorrt_llm" / "libnative.so").write_text("elf")  # the native build
    return repo


def test_a_task_gets_the_whole_tree_and_lands_by_rebase_while_the_head_moves(tmp_path):
    """No cone of chosen directories: every tracked file is on disk, wherever the engine keeps it."""
    repo = _integration_whole_repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _integration_ENV)
    tree = integration.open("k1")
    assert all((tree / path).read_text() == f"{path} 1\n" for path in TRACKED)
    assert (tree / "tensorrt_llm" / "libnative.so").resolve() == repo / "tensorrt_llm/libnative.so"
    assert _integration_git(tree, "status", "--porcelain", "--untracked-files=no") == ""
    # The checkout moves on while the task works, so landing is a real rebase.
    (repo / "cpp" / "kernels" / "rmsnorm.cu").write_text("rmsnorm 2\n")
    (repo / "docs" / "new.md").write_text("new\n")
    _integration_git(repo, "add", ".")
    _integration_git(repo, "commit", "-q", "-m", "native")
    expected = {p: (repo / p).read_text() for p in _integration_git(repo, "ls-files").split()}
    (tree / "tensorrt_llm" / "models" / "modeling.py").write_text("modeling 2\n")
    _integration_git(tree, "commit", "-q", "-am", "model\n\nSigned-off-by: T <t@example.com>")

    outcome = integration.land("k1")
    assert outcome.merged and outcome.commits == 1
    expected["tensorrt_llm/models/modeling.py"] = "modeling 2\n"
    assert set(_integration_git(repo, "ls-files").split()) == set(expected)
    for path, text in expected.items():
        assert (repo / path).read_text() == text, path
        assert _integration_git(repo, "show", f"HEAD:{path}") + "\n" == text, path
    assert _integration_git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"
    integration.close("k1")
    assert not tree.exists()


def _journal(tmp_path: Path) -> Journal:
    journal = Journal(path=tmp_path / "journal.md")
    journal.open(slug="t")
    return journal


class _receipts_Stage(ExecutionStage):
    """What a task's turn uses of the workflow; the turn itself is scripted."""

    def __init__(self, workspace: Path, integration=None):
        super().__init__(
            SimpleNamespace(
                workspace=workspace,
                _tls=threading.local(),
                journal=_journal(workspace),
                _git=git_output,
            )
        )
        self._integration = integration


def test_a_finished_task_keeps_a_receipt_with_the_commits_it_integrated(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _integration_git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("A = 1\n")
    _integration_git(repo, "add", ".")
    _integration_git(repo, "commit", "-q", "-m", "base")
    ws = tmp_path / "ws"
    ws.mkdir()
    integration = Integration(
        repo, ws, "mv2/run", {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}
    )

    class Worker(_receipts_Stage):
        def _work_task_body(self, task):
            tree = integration.tree(task.id)
            for n in (1, 2):
                (tree / "a.py").write_text(f"A = {n + 1}\n")
                _integration_git(tree, "commit", "-q", "-am", f"mtp: step {n}")
            # The turn as the workflow leaves it: its report, its decision file.
            decision = ws / "decisions" / "mtp-assembler-1.yaml"
            journal = self.workflow.journal
            report = journal.record_dispatch(
                "assembler", task.id, f"Do {task.id}.\n\n{decision_file_instruction(decision)}"
            )
            decision.parent.mkdir(parents=True)
            summary = "mtp at tp1.\n\nFor the planner: depth 3 only."
            answer = {"decision": "ready", "summary": summary,
                      "check": {"id": "deployment:mtp", "command": "gate deployment", "result": "PASS"},
                      "not_done": ["depth 4: 0.61 against a floor of 0.7"]}  # fmt: skip
            decision.write_text(yaml.safe_dump(answer))
            journal.record_result(report, "ready; mtp at tp1.")
            return r.TaskResult.answered("assembler", decision, report)

    stage = Worker(ws, integration)
    task = g.Task(id="mtp", kind="feature", goal="mtp", ref="mtp")
    assert stage._work_task(task) == "ready"
    receipt = r.read(ws / "tasks", "mtp")
    assert receipt["decision"] == "ready" and receipt["summary"] == "mtp at tp1."
    assert receipt["check"] == {
        "id": "deployment:mtp",
        "command": "gate deployment",
        "result": "PASS",
    }
    assert receipt["not_done"] == ["depth 4: 0.61 against a floor of 0.7"]
    # One receipt per attempt, written once.
    assert r.path_for(ws / "tasks", "mtp", 1).is_file() and r.read(ws / "tasks", "mtp", 2) is None
    assert [c.split(" ", 1)[1] for c in receipt["commits"]] == ["mtp: step 2", "mtp: step 1"]
    assert receipt["commits"][0].split()[0] == _integration_git(
        repo, "rev-parse", "--short", "HEAD"
    )
    assert receipt["decision_file"] == str(ws / "decisions" / "mtp-assembler-1.yaml")
    assert Path(receipt["result_file"]).name.endswith("-assembler-mtp.md")

    # A process that never integrated it knows only the branch tip.
    later = _receipts_Stage(ws, integration)
    [tip] = later._landed_commits("mtp")
    assert tip.endswith("mtp: step 2 (the branch tip)")
    # A blocked attempt leaves its receipt too: what its escalation is opened with.
    blocked = r.TaskResult(
        "blocked",
        data={"tried": ["a guard"], "command": "pytest t.py::x", "result": "1 failed",
              "ruled_out": "the test's setup", "next_question": "is int64 meant?"},
    )  # fmt: skip
    later._keep_receipt(g.Task(id="core", kind="assemble", goal="g", attempts=2), blocked)
    kept = later._receipt_of(g.Task(id="core", kind="assemble", goal="g", attempts=2))
    assert kept["decision"] == "blocked" and kept["attempt"] == 2 and "commits" not in kept
    assert kept["tried"] == ["a guard"] and kept["ruled_out"] == ["the test's setup"]
    assert (kept["command"], kept["result"]) == ("pytest t.py::x", "1 failed")
    assert kept["next_question"] == "is int64 meant?" and "note" not in kept
    # One that leaves out what it must carry says so in its note.
    thin = r.TaskResult("blocked", data={"summary": "stuck"})
    later._keep_receipt(g.Task(id="thin", kind="assemble", goal="g"), thin)
    assert "leaves out tried, command, result, ruled_out, next_question" in thin.note
    # A turn that wrote no receipt leaves none.
    later._keep_receipt(g.Task(id="none", kind="assemble", goal="g"), r.TaskResult(""))
    assert r.read(ws / "tasks", "none") is None


_BIND = "modeling_v2 x: bound MambaHybridCacheManagerV2 #1: 897 recurrent slots, 32 per slot"
#: As a session passes it on: the line of its server's log that says which pool it bound.
_BIND_RECORD = f"MV2FLOW_LOG {json.dumps({'instance': 'server', 'line': _BIND})}\n"


def _binding_shell():
    """Every command passes, and logs the recurrent pool its engine bound."""
    inner = passing_shell()

    def run(command, env=None, timeout=0):
        result = inner(command, env, timeout)
        return CommandResult(returncode=result.returncode, stdout=result.stdout + _BIND_RECORD)

    return run


def test_a_dependent_starts_with_the_receipt_of_what_it_depends_on(
    tmp_path, task_file, monkeypatch
):
    dependent_prompts: list[str] = []

    def assembler(turn, prompt):
        if "## Your task: `why-core`" in prompt:
            dependent_prompts.append(prompt)
            return {"decision": "ready", "summary": "found it"}
        summary = "core at tp1.\n\nThe conv pool binds 897 slots."
        return {"decision": "ready", "summary": summary}

    # The run starts from these tasks instead of the list code makes.
    keys = ["core:deployment@tp1"]
    seed_graph(
        tmp_path / "ws",
        [
            {"id": "core", "kind": "assemble", "goal": "build core", "ref": "core",
             "touches": [f"{TARGET_REL}/core.py"], "satisfies": keys},
            {"id": "why-core", "kind": "feature", "goal": "why core binds", "ref": "core",
             "depends_on": ["core"], "touches": [f"{TARGET_REL}/why.py"]},
        ],
    )  # fmt: skip
    workflow = _workflow(tmp_path, shell_runner=_binding_shell())
    script = Script(None, assembler=assembler)
    _install(monkeypatch, workflow, script)
    with workflow:
        workflow.run(task_file)

    ws = tmp_path / "ws"
    receipt = r.read(ws / "tasks", "core")
    assert receipt["decision"] == "ready" and receipt["summary"] == "core at tp1."
    [prompt] = dependent_prompts
    # What it depends on is one line: how it ended, and the file that has the rest.
    assert r.HEADING in prompt and "\n- `core` (assemble): ready; `" in prompt
    assert "The conv pool binds 897 slots." not in prompt and "core at tp1." not in prompt
    # One order for any kind: the card, the kind's text, what it depends on.
    assert prompt.index("## Your task: `why-core`") < prompt.index(r.HEADING)
    # What the engine ran with is in its session's evidence, on no card and in no file
    # of its own.
    assert "897 recurrent slots" not in prompt and not (ws / "runtime_facts.yaml").exists()
    envelope = json.loads((ws / "tasks" / "why-core.json").read_text())
    assert envelope["receipts"][0]["task"] == "core"


def _snapshots_commit(repo: Path, name: str, text: str) -> str:
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"change {name}")
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def snapshot_repo(tmp_path) -> Path:
    """A checkout with a package, native code, a quickstart and a built library git ignores."""
    repo = tmp_path / "checkout"
    (repo / "tensorrt_llm" / "libs").mkdir(parents=True)
    (repo / "tensorrt_llm" / "libs" / "libth_common.so").write_bytes(b"\x7fELF")
    (repo / ".gitignore").write_text("*.so\n")
    _git(repo, "init", "-q", ".")
    for name in ("tensorrt_llm/__init__.py", "cpp/kernel.cu", "examples/llm-api/quickstart.py"):
        _snapshots_commit(repo, name, "x = 1\n")
    return repo


@pytest.mark.parametrize("imported_from", ["snapshot", "checkout"])
def test_a_check_measures_one_commit_while_integrations_move_the_checkout(
    tmp_path, snapshot_repo, monkeypatch, imported_from
):
    repo = snapshot_repo
    head = _git(repo, "rev-parse", "HEAD").strip()
    seen: list[str] = []

    def shell(command, env=None, timeout=0):
        seen.append(command)
        if len(seen) == 1:  # a task lands between the legs
            _snapshots_commit(repo, "tensorrt_llm/landed.py", "z = 3\n")
        root = command.split("cd ", 1)[1].split(" ", 1)[0] if imported_from == "snapshot" else repo
        return CommandResult(
            returncode=0,
            stdout=f"[mv2flow-resolve] pid=7 rank=0 module=tensorrt_llm "
            f"origin={root}/tensorrt_llm/__init__.py\n",
        )

    def two_jobs(deployment, environment, **kwargs):
        # A session and a load of its own after it: two jobs of one check.
        for job in ("session", "eval"):
            kwargs["runner"](f"job {job}", None, 10)
        boot = check_result.GateResult(name="boot", member="tp1", passed=True, detail="answered")
        own = check_result.GateResult(
            name="deployment", member="tp1", passed=True, detail="1 check(s) passed"
        )
        return session_mod.SessionResult((boot,), own, loads=2)

    monkeypatch.setattr(session_mod, "run_deployment", two_jobs)
    workflow = _workflow(tmp_path, shell_runner=shell, git_runner=None)
    workflow.task_data = {
        "trtllm_repo_path": str(repo),
        "deployments": [{"name": "core", "options": {}}],
    }
    workflow.environment = RunEnvironment(kind="local", python="/py", world_size=1)
    workflow._snapshots = Integration(repo, workflow.workspace, "mv2/run", pin_age_s=0)
    check = contract_mod.Check("core", "deployment", "tp1")
    marker = workflow._check_key(check)

    result = CheckRunner(workflow)._run_contract_check(check, None)
    status = workflow._contract_status()
    # Each check of the session has its own line in the ledger, on what the session measured.
    line = status.latest()["core:boot@tp1"]
    assert line.marker == marker and line.tree_state == head[:7]
    assert line.passed == (imported_from == "snapshot") and bool(line.void) != line.passed
    verdict = status.record("core", result, marker="the target when the check was queued")

    snapshot = Path(seen[0].split("cd ", 1)[1].split(" ", 1)[0])
    assert snapshot.parent == pintree.pins_root(workflow._snapshots.place)
    assert snapshot.name.startswith(head[:12] + "-"), "named for the commit and its build"
    assert seen and all(command.startswith(f"cd {snapshot} && ") for command in seen)
    assert verdict.marker == result.marker == marker and verdict.tree_state == head[:7]
    assert not snapshot.exists()  # the checkout moved on and nothing holds it
    journal = workflow.journal_path.read_text()
    if imported_from == "snapshot":
        assert result.passed and verdict.imported == [head, head] and not verdict.void
        return
    # Each leg imported the checkout an integration moved: void, never passed, and not
    # measured a second time (the tree is pinned: the same command imports from the same place).
    assert len(seen) == 2 and "measuring it again" not in journal
    assert not result.passed and not verdict.passed and str(repo) in verdict.void
    assert "[VOID] deployment@tp1" in result.summary()
    rendered = contract_mod.evaluate(contract_mod.Contract(checks=(check,)), status).render()
    assert f"- VOID {check.key}: " in rendered


_FILES = {
    "tensorrt_llm/models/target.py": "T = 1\n",
    "tensorrt_llm/README.md": "the package\n",
    "tests/integration/defs/accuracy/references/gsm8k.yaml": "m: [{accuracy: 50}]\n",
    "examples/llm-api/quickstart_advanced.py": "print('q')\n",
}


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout"
    for name, text in _FILES.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "init", "-q", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _workflow_marker_workflow(repo: Path) -> TrtllmModelingV2Workflow:
    workflow = TrtllmModelingV2Workflow(workspace=repo / "ws")
    workflow.task_data = {
        "trtllm_repo_path": str(repo),
        "accuracy": {"reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml"},
        "deployments": [{"name": "core", "options": {}, "tests": ["tests/unittest/test_x.py"]}],
    }
    return workflow


def _workflow_marker_commit(repo: Path, name: str, text: str) -> None:
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"change {name}")


_SESSION = contract_mod.Check("core", "deployment", "tp1")
_NATIVE = contract_mod.Check("core", "tests", "tp1")


@pytest.mark.parametrize(
    "name,session,native",
    [
        # A test nobody registered: no session loads it; the tests' own key moves.
        ("tests/unittest/test_new.py", False, True),
        ("examples/llm-api/quickstart_advanced.py", False, True),
        # Documentation and a CI test list: no check of either class consumes them.
        ("docs/source/guide.md", False, False),
        ("README.md", False, False),
        # In the runtime package too: a guide or a note edited after a session passed must
        # not cost that session's load again. Markdown a check reads is named by the spec.
        ("tensorrt_llm/README.md", False, False),
        ("tests/integration/test_lists/qa/llm_function_core.txt", False, False),
        # The runtime package: everything that runs it.
        ("tensorrt_llm/models/target.py", True, True),
        # A file the spec names as an input wins over every exclusion, wherever it lives.
        ("tests/integration/defs/accuracy/references/gsm8k.yaml", True, True),
    ],
)
def test_a_commit_changes_the_key_of_the_checks_that_consume_what_it_changed(
    checkout, name, session, native
):
    workflow = _workflow_marker_workflow(checkout)
    custom = contract_mod.Check("core", "a-command", "tp1")
    before = {c.key: workflow._check_key(c) for c in (_SESSION, _NATIVE)}
    whole = workflow._key("a-command:core")
    assert all(key.startswith("key:") for key in before.values())
    assert before[_SESSION.key] != before[_NATIVE.key], "one key per check, by its class"
    _workflow_marker_commit(checkout, name, "edited\n")
    assert (workflow._check_key(_SESSION) != before[_SESSION.key]) is session
    assert (workflow._check_key(_NATIVE) != before[_NATIVE.key]) is native
    # A custom command uses the whole tree; its inputs are unknown, so nothing is reused.
    assert workflow._check_key(custom) == "" and whole == ""


def test_a_key_takes_what_the_spec_names_wherever_it_is_kept_or_stated(checkout, monkeypatch):
    workflow = _workflow_marker_workflow(checkout)
    tests = {"files": ["tests/unittest/test_new.py"], "gpus": 1}
    # The image a Slurm job runs in is stated in the slurm block: the session's key and the
    # final tests' take it there. Where the scheduler runs the job is in neither.
    workflow.task_data["environment"] = {"kind": "slurm", "slurm": {"container_image": "a.sqsh"}}
    before = workflow._check_key(_SESSION), workflow._key("final-tests", extra=tests)
    assert all(key.startswith("key:") for key in before)
    workflow.task_data["environment"]["slurm"]["partition"] = "another"
    assert (workflow._check_key(_SESSION), workflow._key("final-tests", extra=tests)) == before
    # Nor a mount added in the middle of a run: it changes no result, and a key that moved
    # for it would have every deployment loaded again.
    workflow.task_data["environment"]["slurm"]["container_mounts"] = ["/data:/data"]
    assert (workflow._check_key(_SESSION), workflow._key("final-tests", extra=tests)) == before
    workflow.task_data["environment"]["slurm"]["container_image"] = "b.sqsh"
    after = workflow._check_key(_SESSION), workflow._key("final-tests", extra=tests)
    assert after[0] != before[0] and after[1] != before[1]
    # A variable the tests run with is the final tests' runtime too.
    workflow.task_data["environment"]["env"] = {"MY_TEST_BACKEND": "new"}
    assert workflow._key("final-tests", extra=tests) != after[1]
    # What a final job carries of the build and the runtime is in the test driver's own
    # digest of its environment: results recorded under another are not read as these.
    from agent_flow.workflows.trtllm_modeling_v2.checks.harness import test_driver
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import regression

    monkeypatch.setenv(regression.IDENTITY_VAR, "one")
    first = test_driver.environment()["vars"]
    monkeypatch.setenv(regression.IDENTITY_VAR, "two")
    assert first[regression.IDENTITY_VAR] == "one"
    assert test_driver.environment()["vars"] != first


def test_a_directory_the_environment_names_is_its_name_not_what_the_run_writes_there(
    checkout, tmp_path
):
    """A whole run measured both deployments and its final tests again on one candidate.

    Nothing had changed but its own cache of compiled kernels, which its sessions had
    filled between two starts of the controller.
    """
    from agent_flow.workflows.trtllm_modeling_v2.checks import check_runtime

    workflow = _workflow_marker_workflow(checkout)
    tests = {"files": ["tests/unittest/test_new.py"], "gpus": 1}
    scratch, settings = tmp_path / "scratch", tmp_path / "settings.json"
    settings.write_text('{"a": 1}')
    workflow.task_data["environment"] = {
        "env": {"TMPDIR": str(scratch), "MY_SETTINGS": str(settings), "MY_BACKEND": "one"}
    }
    # As every start leaves the spec: the run's own kernel cache, under its workspace.
    check_runtime.default_triton_cache(workflow.task_data, tmp_path / "ws")
    cache = Path(workflow.task_data["environment"]["env"]["TRITON_CACHE_DIR"])
    assert cache == (tmp_path / "ws").resolve() / "triton-cache"

    def keys():
        return workflow._check_key(_SESSION), workflow._key("final-tests", extra=tests)

    before = keys()
    assert all(key.startswith("key:") for key in before)
    # The first session compiles its kernels into the cache, and a job leaves files in
    # the scratch. The tree, the options and what the checks are judged by are as they
    # were: so are the keys, and nothing is measured again.
    for directory in (cache / "VFK2AGW7", scratch):
        directory.mkdir(parents=True)
        (directory / "kernel.cubin").write_bytes(b"compiled")
    assert keys() == before
    (cache / "VFK2AGW7" / "kernel.json").write_text("{}")
    assert keys() == before
    # A file a variable names is still what it holds, and a variable's value is the
    # runtime: either changed is another key for both.
    settings.write_text('{"a": 22}')
    assert all(now != was for now, was in zip(keys(), before))
    settings.write_text('{"a": 1}')
    assert keys() == before
    workflow.task_data["environment"]["env"]["MY_BACKEND"] = "two"
    assert all(now != was for now, was in zip(keys(), before))
    # The models root is such a directory too: another model put there is nothing a
    # check is judged by. The dataset the evaluator finds there is, by its own content.
    root = tmp_path / "models"
    (root / "datasets" / "openai" / "gsm8k").mkdir(parents=True)
    (root / "datasets" / "openai" / "gsm8k" / "test.jsonl").write_text('{"q": 1}\n')
    workflow.task_data["environment"]["env"]["LLM_MODELS_ROOT"] = str(root)
    workflow.task_data["deployments"][0]["benchmarks"] = [{"name": "gsm8k", "anchor": 41.0}]
    scored = workflow._check_key(_SESSION)
    (root / "another-org" / "another-model").mkdir(parents=True)
    (root / "another-org" / "another-model" / "config.json").write_text("{}")
    assert workflow._check_key(_SESSION) == scored
    (root / "datasets" / "openai" / "gsm8k" / "test.jsonl").write_text('{"q": 2, "more": 1}\n')
    assert workflow._check_key(_SESSION) != scored


def test_the_native_tests_a_deployment_names_are_no_input_of_its_session(tmp_path):
    def inputs(gate, tests):
        deployment = {"name": "core", "options": {}, **({"tests": tests} if tests else {})}
        data = {"trtllm_repo_path": str(tmp_path), "deployments": [deployment]}
        return contract_mod.check_inputs(data, "core", gate, "tp1")

    # A node id taken out of the list, or the whole list: the session ran none of them, so
    # what it measured stands and it is not measured again.
    both = ["tests/a.py::one", "tests/a.py::two"]
    assert inputs("deployment", both) == inputs("deployment", both[:1]) == inputs("deployment", [])
    # The check that runs them is judged by the list it ran.
    assert inputs("tests", both) != inputs("tests", both[:1])


def test_a_small_input_is_what_it_holds_wherever_it_is_kept(tmp_path):
    def inputs(prompts, repo):
        data = {
            "trtllm_repo_path": str(repo),
            "deployments": [{"name": "core", "options": {}}],
            "accuracy": {"boot": {"prompts_file": str(prompts)}},
        }
        return contract_mod.check_inputs(data, "core", "deployment", "tp1")

    for name in ("one", "two", "repo-1/docs", "repo-2/docs"):
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "prompts.md").write_text("the prompts\n")
    # Outside the checkout: the same bytes at another place are the same input.
    here = inputs(tmp_path / "one" / "prompts.md", tmp_path / "repo-1")
    assert inputs(tmp_path / "two" / "prompts.md", tmp_path / "repo-1") == here
    (tmp_path / "two" / "prompts.md").write_text("other prompts\n")
    assert inputs(tmp_path / "two" / "prompts.md", tmp_path / "repo-1") != here
    # In the checkout: its place in the checkout, wherever the checkout is (its bytes are
    # the key's source).
    first = inputs(tmp_path / "repo-1" / "docs" / "prompts.md", tmp_path / "repo-1")
    assert inputs(tmp_path / "repo-2" / "docs" / "prompts.md", tmp_path / "repo-2") == first
    assert first != here


def test_a_key_is_content_not_names_times_or_places(checkout, tmp_path):
    with (checkout / ".git" / "info" / "exclude").open("a") as exclude:
        exclude.write("*.so\n")
    library = checkout / "tensorrt_llm" / "libs" / "libth_common.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"\x7fELF-one")
    workflow = _workflow_marker_workflow(checkout)
    before = workflow._check_key(_SESSION)
    # The same bytes written again at another time, as a copy to another node would be.
    stat = library.stat()
    library.unlink()
    library.write_bytes(b"\x7fELF-one")
    os.utime(library, ns=(stat.st_atime_ns + 10**12, stat.st_mtime_ns + 10**12))
    assert workflow._check_key(_SESSION) == before
    # Other bytes under the same name, size and modification time are another build.
    stat = library.stat()
    library.write_bytes(b"\x7fELF-two")
    os.utime(library, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert library.stat().st_size == stat.st_size
    assert library.stat().st_mtime_ns == stat.st_mtime_ns
    assert workflow._check_key(_SESSION) != before
    library.write_bytes(b"\x7fELF-one")
    assert workflow._check_key(_SESSION) == before
    # What the session is judged by is in its key: an option, and the reference it names.
    workflow.task_data["deployments"][0]["options"]["max_batch_size"] = 8
    assert workflow._check_key(_SESSION) != before
    assert workflow._check_key(_NATIVE) != ""
    # What only schedules it is not: how long it may take.
    workflow.task_data["deployments"][0]["options"].pop("max_batch_size")
    workflow.task_data["deployments"][0]["timeout"] = 9999
    assert workflow._check_key(_SESSION) == before


def test_a_checkpoint_is_keyed_by_what_it_holds_not_by_when_it_was_written(checkout, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"hidden_size": 8}')
    (checkpoint / "tokenizer.json").write_text('{"vocab": ["a", "b"]}')
    (checkpoint / "model.safetensors").write_bytes(b"w" * 64)
    workflow = _workflow_marker_workflow(checkout)
    workflow.task_data["checkpoint_path"] = str(checkpoint)
    workflow.task_data["environment"] = {"python": "/usr/bin/python3"}
    before = workflow._check_key(_SESSION)
    assert before.startswith("key:")
    # Touched, not changed: the same checkpoint.
    os.utime(checkpoint / "tokenizer.json", ns=(1, 1))
    assert workflow._check_key(_SESSION) == before
    # The tokenizer's content is the checkpoint's content.
    (checkpoint / "tokenizer.json").write_text('{"vocab": ["a", "c"]}')
    assert workflow._check_key(_SESSION) != before
    (checkpoint / "tokenizer.json").write_text('{"vocab": ["a", "b"]}')
    assert workflow._check_key(_SESSION) == before
    # So is a weight file's size, and a file that appears.
    (checkpoint / "model.safetensors").write_bytes(b"w" * 65)
    assert workflow._check_key(_SESSION) != before
    (checkpoint / "model.safetensors").write_bytes(b"w" * 64)
    (checkpoint / "preprocessor_config.json").write_text("{}")
    assert workflow._check_key(_SESSION) != before
    (checkpoint / "preprocessor_config.json").unlink()
    assert workflow._check_key(_SESSION) == before
    # The runtime the session runs in is in the key; where the scheduler runs it is not.
    workflow.task_data["environment"]["slurm"] = {"account": "a", "partition": "p", "time": "2h"}
    assert workflow._check_key(_SESSION) == before
    workflow.task_data["environment"]["container_image"] = "another-image"
    assert workflow._check_key(_SESSION) != before


@pytest.mark.parametrize("change,measured_again", [("docs", False), ("library", True)])
def test_the_final_pass_leaves_a_session_a_docs_commit_passed_and_measures_one_a_library_changed(
    checkout, tmp_path, change, measured_again
):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orch

    with (checkout / ".git" / "info" / "exclude").open("a") as exclude:
        exclude.write("*.so\n")
    library = checkout / "tensorrt_llm" / "libs" / "libth_common.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"\x7fELF-one")
    workflow = _workflow_marker_workflow(checkout)
    ran, logged = [], []

    def check(chk):
        ran.append(chk.key)
        return SimpleNamespace(name=chk.gate, member=chk.member, passed=True, detail="")

    def work(task):
        # After the session passed, in the middle of the run.
        if change == "docs":
            _workflow_marker_commit(checkout, "docs/source/guide.md", "a new page\n")
            _workflow_marker_commit(
                checkout, "tests/integration/test_lists/qa/core.txt", "a test, registered\n"
            )
        else:
            library.write_bytes(b"\x7fELF-rebuilt")
        return "ready"

    workspace = tmp_path / "run"
    workspace.mkdir()
    orch.seed(
        workspace,
        [
            g.Task("session-core", g.CHECK_KIND, "the session", satisfies=[_SESSION.key]),
            g.Task("late", "feature", "a late row", depends_on=["session-core"]),
        ],
    )
    services = orch.Services(work=work, check=check, key=workflow._check_key, log=logged.append)
    loop = orch.GraphOrchestrator(
        services,
        contract_mod.Contract((_SESSION,)),
        contract_mod.ContractStatus.load(workspace),
        workspace,
        poll_s=0.01,
    )
    assert loop.run() == "delivered"
    # Nothing ran for the landing itself. The final pass asked for the session's key on
    # the candidate: the same after documentation and a test registration, another after
    # the library changed.
    assert ran.count(_SESSION.key) == (2 if measured_again else 1)
    assert loop.final.rows[_SESSION.key].measured is measured_again
    assert loop.final.rows[_SESSION.key].row.marker == workflow._check_key(_SESSION)


def _new_file(tree, name, text, message):
    (tree / name).write_text(text)
    _integration_git(tree, "add", name)
    _integration_git(tree, "commit", "-q", "-s", "-m", message)


def test_a_commit_outside_the_write_scope_goes_back_with_the_files_named_and_then_lands(tmp_path):
    repo = _integration_repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _integration_ENV)
    tree = integration.open("k1")
    _integration_commit(tree, "a.py", "A = 2\n", "the entry")
    _new_file(tree, "shared.py", "S = 1\n", "a shared initializer")

    def puts_it_right(turn):
        if turn == 2:
            # What the note asks for: the file outside the scope leaves the branch.
            _integration_git(tree, "rm", "-q", "shared.py")
            _integration_git(tree, "commit", "-q", "-m", "keep to the scope")
        return "ready"

    stage = _integration_Stage(tmp_path / "ws", integration, puts_it_right)
    task = g.Task(id="k1", kind="kernel", goal="the entry", touches=["a.py"])
    assert stage._work_task_turn(task).decision == "ready"
    assert (repo / "a.py").read_text() == "A = 2\n" and not (repo / "shared.py").exists()
    first, sent_back = stage.blocks
    assert "outside your write scope" not in first
    assert "## Your commits reach outside your write scope" in sent_back
    assert "- `shared.py`" in sent_back and "`a.py`" in sent_back
    assert any(
        "reach outside its write scope: shared.py" in e for e in stage.workflow.journal.events
    )
