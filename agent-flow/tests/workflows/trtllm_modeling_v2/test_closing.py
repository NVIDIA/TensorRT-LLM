# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a run leaves when it stops: the onboarded ref, the skeleton, the closing report, its exit."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.checks import anchors
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import pool as pool_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.git import CommitIdentity, CommitPolicy
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor import heartbeat as heartbeat_mod
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.run import escalation as esc
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.execution import tasks as tasks_mod
from agent_flow.workflows.trtllm_modeling_v2.run.records import report
from agent_flow.workflows.trtllm_modeling_v2.run.records.state import WorkflowState
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import (
    ONBOARDED_FILENAME,
    TrtllmModelingV2Workflow,
)
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import Integration

from ._helpers import _workflow

_IDENTITY = {
    "GIT_AUTHOR_NAME": "T",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "T",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}
_TARGET = f"{catalog_paths.MODELING_V2_PACKAGE}/models/fam/new__sm_100__unified"
_TESTS = catalog_paths.model_ops_tests_dir(_TARGET)


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    """A real checkout whose modeling_v2 package has a family and no target yet."""
    repo = tmp_path / "repo"
    family = repo / catalog_paths.MODELING_V2_PACKAGE / "models" / "fam"
    family.mkdir(parents=True)
    for directory in (family, *family.parents[:5]):
        (directory / "__init__.py").write_text("")
    (repo / "tests").mkdir()
    (repo / "tests" / "keep.py").write_text("")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _real(tmp_path, checkout) -> TrtllmModelingV2Workflow:
    """A workflow that asks the real git of a real checkout."""
    workflow = _workflow(tmp_path, git_runner=None)
    workflow.task_data = {
        "trtllm_repo_path": str(checkout),
        "deployments": [{"name": "tp1", "options": {}}],
    }
    workflow._state = SimpleNamespace(slug="run1", stage="execution")
    return workflow


def test_onboarded_is_a_ref_created_once_and_a_record_written_once(tmp_path, checkout):
    workflow = _real(tmp_path, checkout)
    head = _git(checkout, "rev-parse", "HEAD")
    session = g.Task(id="session-tp1", kind=g.CHECK_KIND, goal="s", ref="tp1",
                     satisfies=["tp1:deployment@tp1"])  # fmt: skip
    record = workflow._mark_onboarded(session)
    ref = "refs/mv2/run1/onboarded"
    assert record["commit"] == head and record["ref"] == ref
    assert (record["deployment"], record["session"]) == ("tp1", "session-tp1")
    # With the key of the session's evidence at that commit: what it was pinned on.
    assert record["checks"] == ["tp1:deployment@tp1"] and record["key"].startswith("key:")
    assert _git(checkout, "rev-parse", ref) == head
    written = (workflow.workspace / ONBOARDED_FILENAME).read_text()
    assert json.loads(written) == record
    # The head moves on (a later repair): nothing moves the ref or the record.
    (checkout / "tests" / "later.py").write_text("")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-q", "-m", "later")
    later = _git(checkout, "rev-parse", "HEAD")
    assert later != head
    assert workflow._mark_onboarded(session) == record
    assert _git(checkout, "rev-parse", ref) == head
    assert (workflow.workspace / ONBOARDED_FILENAME).read_text() == written
    # A start that finds the ref and no record (the write was lost) takes the ref's commit.
    (workflow.workspace / ONBOARDED_FILENAME).unlink()
    # It is written again for that commit, with that commit's key; what only the moment
    # knew is said to be not recorded.
    again = _real(tmp_path, checkout)._mark_onboarded(session)
    assert again["commit"] == head and _git(checkout, "rev-parse", ref) == head
    assert again["key"] == record["key"] and again["recovered_at"]
    assert again["at"] is None and again["requirements"] is None
    # A part of a record (a write that was cut) is no record: it is replaced, whole.
    (workflow.workspace / ONBOARDED_FILENAME).write_text("{")
    whole = _real(tmp_path, checkout)._mark_onboarded(session)
    assert whole["commit"] == head
    assert json.loads((workflow.workspace / ONBOARDED_FILENAME).read_text()) == whole
    assert "onboarded" in workflow.journal_path.read_text()


def test_onboarded_is_not_pinned_on_a_commit_that_landed_after_the_session(tmp_path, checkout):
    """Another task's landing between the session and the pin is not what the session measured."""
    workflow = _real(tmp_path, checkout)
    workflow._contract = contract_mod.compile_contract(workflow.task_data)
    (check,) = workflow._contract.checks
    measured, key = _git(checkout, "rev-parse", "HEAD"), workflow._check_key(check)
    line = SimpleNamespace(
        name=check.gate, member=check.member, passed=True, detail="", imported=(measured,)
    )
    workflow._contract_status().record(check.unit, line, marker=key)
    # A commit the session consumes lands before the publication: the head has another key.
    (checkout / "tensorrt_llm").mkdir(exist_ok=True)
    (checkout / "tensorrt_llm" / "another_runtime.py").write_text("CHANGED = 1\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-q", "-m", "another task's runtime change")
    later = _git(checkout, "rev-parse", "HEAD")
    assert later != measured and workflow._check_key(check) != key
    session = g.Task(id="session-tp1", kind=g.CHECK_KIND, goal="s", ref="tp1",
                     satisfies=[check.key])  # fmt: skip
    record = workflow._mark_onboarded(session)
    # The ref and the record name what was measured, with the key it was measured under.
    assert (record["commit"], record["key"]) == (measured, key)
    assert _git(checkout, "rev-parse", record["ref"]) == measured
    assert "unverified" not in record
    # A row that names no commit cannot say which was measured: pinned at the head, and
    # the record says that the head is not what the session measured.
    other = tmp_path / "other"
    other.mkdir()
    blind = _real(other, checkout)
    blind._state = SimpleNamespace(slug="run2", stage="execution")
    blind._contract = workflow._contract
    blind._contract_status().record(
        check.unit, SimpleNamespace(name=check.gate, member=check.member, passed=True, detail=""),
        marker=key,
    )  # fmt: skip
    said = blind._mark_onboarded(session)
    assert said["commit"] == later and "pinned at the checkout's head" in said["unverified"][0]


def test_what_a_task_says_it_could_not_verify_is_not_listed_again_from_the_ledger(tmp_path):
    why = "tp1:deployment@tp1: hop: 2 of 2 response(s) name no transfer interval"
    session = g.Task(id="session-tp1", kind=g.CHECK_KIND, goal="s", ref="tp1", status="done",
                     satisfies=["tp1:deployment@tp1"], unverified=[why])  # fmt: skip
    g.Graph(tasks=[session]).save(tmp_path / g.GRAPH_FILENAME)
    # The ledger's own line for that check is not listed a second time; another's is.
    with (tmp_path / contract_mod.STATUS_FILENAME).open("a") as ledger:
        for gate in ("deployment", "hop"):
            ledger.write(json.dumps({"unit": "tp1", "gate": gate, "member": "tp1", "passed": True,
                                     "unverified": "no transfer interval"}) + "\n")  # fmt: skip
    listed = report.not_verified(tmp_path)
    assert len(listed) == 2 and listed[1] == "`tp1:hop@tp1`: no transfer interval"


def test_the_skeleton_is_one_signed_commit_that_lands_before_the_rows_tasks(
    tmp_path, checkout, monkeypatch
):
    # The run's commits are made as the identity the run resolved, which the command
    # that starts a run exports; never as whoever this host's git is configured for.
    monkeypatch.setenv("MV2FLOW_GIT_NAME", "T")
    monkeypatch.setenv("MV2FLOW_GIT_EMAIL", "t@example.com")
    workspace = tmp_path / "ws"
    integration = Integration(checkout, workspace, "mv2/run", _IDENTITY)
    journal = SimpleNamespace(event=lambda kind, text: None)
    stage = tasks_mod.ExecutionStage(
        SimpleNamespace(
            workspace=workspace,
            task_data={"trtllm_repo_path": str(checkout)},
            _tls=threading.local(),
            journal=journal,
        )
    )
    stage._integration = integration
    stage._own_paths = g.own_paths(_TARGET)
    files = [f"{_TARGET}/ops/norm/rms.py", f"{_TARGET}/ops/attention/gqa.py",
             f"{_TESTS}/norm/test_modeling_v2_rms.py"]  # fmt: skip
    before = _git(checkout, "rev-parse", "HEAD")
    said = stage._commit_skeleton(files)
    assert said.startswith("6 initializer(s) committed and landed: ")
    # One commit on the head, signed off, holding the initializers and nothing else.
    assert _git(checkout, "rev-list", "--count", f"{before}..HEAD") == "1"
    assert "Signed-off-by: T <t@example.com>" in _git(checkout, "log", "-1", "--format=%B")
    changed = _git(checkout, "diff", "--name-only", before, "HEAD").splitlines()
    assert sorted(changed) == sorted(
        [
            f"{_TESTS}/__init__.py",
            f"{_TARGET}/__init__.py",
            f"{_TARGET}/ops/__init__.py",
            f"{_TARGET}/ops/norm/__init__.py",
            f"{_TARGET}/ops/attention/__init__.py",
            f"{_TESTS}/norm/__init__.py",
        ]
    )
    assert (checkout / _TARGET / "ops" / "norm" / "__init__.py").read_text().startswith("# SPDX")
    # A task opened after it starts with the directories in place; asking again owes nothing.
    tree = integration.open("k1-rms")
    assert (tree / _TARGET / "ops" / "norm" / "__init__.py").is_file()
    assert stage._commit_skeleton(files) == "nothing to commit: every directory on the way exists"
    assert not list(Path(integration.place).glob(f"worktrees/{tasks_mod.SKELETON_TASK}*"))
    assert _git(checkout, "rev-list", "--count", f"{before}..HEAD") == "1"


def test_the_execution_stage_starts_on_a_real_checkout_with_the_runs_own_trees(
    tmp_path, checkout, monkeypatch
):
    """The stage builds the loop's services from the run's trees as they are, and the loop starts.

    A service that named something the trees no longer had (`Integration.alone`, gone with
    the rebuild barrier) ended every real run at this stage's first lines while the suite
    was green: no other case enters ``run_execution`` on real trees.
    """
    workflow = _real(tmp_path, checkout)
    workflow._contract = contract_mod.compile_contract(workflow.task_data)
    workflow.environment = SimpleNamespace(spool="")
    started = []
    monkeypatch.setattr(
        tasks_mod.GraphOrchestrator, "run", lambda loop: started.append(loop) or "stopped: here"
    )
    stage = tasks_mod.ExecutionStage(workflow)
    state = WorkflowState(task_path=str(tmp_path / "task.yaml"), stage="execution", slug="run1")
    assert stage.run_execution(state, None) is False
    (loop,) = started
    assert isinstance(stage._integration, Integration) and loop.workspace == workflow.workspace
    # Every service the loop asks of the stage is one the stage has.
    assert loop.s.freeze() == _git(checkout, "rev-parse", "HEAD")


def test_a_commit_policy_a_resolution_widened_reaches_the_landing_of_the_controller_that_runs(
    tmp_path, checkout, monkeypatch
):
    """Found by rehearsing a run: `allow-shared` was applied, and the row was refused at its landing."""
    monkeypatch.setenv("MV2FLOW_GIT_NAME", "T")
    monkeypatch.setenv("MV2FLOW_GIT_EMAIL", "t@example.com")
    workspace = tmp_path / "ws"
    spec = {
        "trtllm_repo_path": str(checkout),
        "commit_policy": {"allowed_paths": [f"{_TARGET}/ops/norm/"]},
    }
    policy = CommitPolicy.from_task(spec, workspace=workspace)
    integration = Integration(checkout, workspace, "mv2/run", _IDENTITY, policy=policy)
    told, followed = [], []
    stage = tasks_mod.ExecutionStage(
        SimpleNamespace(
            workspace=workspace,
            task_data=spec,
            _tls=threading.local(),
            journal=SimpleNamespace(event=lambda kind, text: None),
            _respec=lambda keys: told.append(list(keys)),
        )
    )
    stage._integration = integration
    stage._allow_paths = followed.append
    tree = integration.open("k3-gqa")
    entry = f"{_TARGET}/ops/attention/gqa.py"
    (tree / entry).parent.mkdir(parents=True)
    (tree / entry).write_text("E = 1\n")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-s", "-m", "[None][feat] modeling_v2: gqa")
    refused = integration.land("k3-gqa")
    assert not refused.merged and "outside the paths this run may commit" in refused.detail
    # `resolve allow-shared`, as a running controller applies it: the spec, then its own copies.
    spec["commit_policy"]["allowed_paths"].append(f"{_TARGET}/ops/attention/")
    stage._hooks().respec(["commit_policy.allowed_paths"])
    assert told == [["commit_policy.allowed_paths"]]
    assert followed == [(f"{_TARGET}/ops/norm/", f"{_TARGET}/ops/attention/")]
    assert integration.land("k3-gqa").merged and (checkout / entry).is_file()
    # A key outside the commit policy leaves both as they are.
    kept = integration.policy
    stage._hooks().respec(["environment.env.X"])
    assert integration.policy is kept and len(followed) == 1 and len(told) == 2


def _closed(workspace: Path) -> None:
    """A workspace as a run leaves it: onboarded, one anchor and one key changed, work left."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ONBOARDED_FILENAME).write_text(
        json.dumps({"commit": "c0ffee1234", "ref": "refs/mv2/run1/onboarded",
                    "session": "session-tp1"})
    )  # fmt: skip
    anchors.change(workspace, "tp1", "gsm8k", reason="as the spec stated it", value=61.77)
    anchors.change(workspace, "tp1", "gsm8k", reason="61.77 was a typing error", value=41.77)
    spec = {"environment": {"env": {"TMPDIR": "/tmp"}}}
    esc.record_change(workspace, spec, "environment.env.TMPDIR", "/var/tmp",
                      why="the node's /tmp is 2 GiB", command="c1", escalation="esc-k1-1")  # fmt: skip
    g.Graph(
        tasks=[
            g.Task(id="k1-rms", kind="kernel", goal="e", status="done", check="kernel:1"),
            g.Task(id="weights", kind="weights", goal="w", status="done", check="cpu:weights"),
            g.Task(id="notes", kind="feature", goal="n", status="done"),
            g.Task(id="k2-fp8", kind="kernel", goal="e", status="skipped",
                   last="skipped (esc-k2-fp8-3): no deployment needs an FP8 cell"),
            g.Task(id="session-pd", kind=g.CHECK_KIND, goal="s", status="blocked",
                   attempts=2, last="its checks failed: pd:deployment@pd2"),
            g.Task(id="assemble-kv", kind="assemble", goal="a", depends_on=["session-pd"]),
        ]
    ).save(workspace / g.GRAPH_FILENAME)  # fmt: skip
    status = contract_mod.ContractStatus.load(workspace)
    status.record("tp1", SimpleNamespace(name="deployment", member="tp1", passed=True, detail="ok"))
    status.record(
        "pd",
        SimpleNamespace(name="hop", member="pd2", passed=True,
                        detail="not verified: 2 of 2 response(s) name no transfer interval"),
    )  # fmt: skip


def test_the_closing_report_says_under_which_requirements_and_what_is_not_done_or_verified(
    tmp_path,
):
    workspace = tmp_path / "ws"
    _closed(workspace)
    text = report.render(workspace, verdict="partial", target="models/fam/new", gates="[PASS] x")
    assert text.splitlines()[0] == "# models/fam/new: partial"
    assert "**Onboarded** at `c0ffee1234` (`refs/mv2/run1/onboarded`)" in text
    # The requirements it is judged by now, and each one changed with from, to and why.
    assert "- `tp1:gsm8k`: declared 41.77 (61.77 was a typing error)" in text
    assert (
        "| anchor `tp1:gsm8k` | requirement | declared 61.77 | declared 41.77 "
        "| 61.77 was a typing error |"
    ) in text
    assert (
        "| `environment.env.TMPDIR` | environment | '/tmp' | '/var/tmp' "
        "| the node's /tmp is 2 GiB |"
    ) in text
    # This record has no time: the changes are attributed to it neither way.
    assert "When `onboarded` was pinned against these changes is not recorded" in text
    assert "holds under the requirements as changed" not in text
    # Not done: skipped, blocked and waiting for an answer, and what never started.
    assert "- `k2-fp8` (skipped): skipped (esc-k2-fp8-3): no deployment needs an FP8 cell" in text
    assert (
        "- `session-pd` (blocked): its checks failed: pd:deployment@pd2 "
        "(attempt 2 waits for an answer)"
    ) in text
    assert "- `assemble-kv` (pending): it did not start: what it waits for is not done" in text
    # Not verified: the session's third outcome, and a row that landed on its receipt alone.
    assert "- `pd:hop@pd2`: not verified: 2 of 2 response(s) name no transfer interval" in text
    # A row's own tests are the flow's to run now: only a task with no check lands unchecked.
    assert "- `notes` landed on its worker's receipt: it names no check" in text
    assert "`k1-rms` landed" not in text and "`weights` landed" not in text
    assert "## Checks" in text
    # The ledger's own field for it is read where a line carries one.
    with (workspace / contract_mod.STATUS_FILENAME).open("a") as ledger:
        ledger.write(json.dumps({"unit": "pd", "gate": "envelope", "member": "pd2", "passed": True,
                                 "unverified": "the engine reports no pool size"}) + "\n")  # fmt: skip
    assert "`pd:envelope@pd2`: the engine reports no pool size" in "\n".join(
        report.not_verified(workspace)
    )
    # What the run itself asks, on no task (a final pass that did not hold), is not done too.
    graph = g.Graph.load(workspace / g.GRAPH_FILENAME)
    graph.run_blocked = {"kind": "not_delivered",
                         "observed": "delivery files: gsm8k.yaml has no entry for m"}  # fmt: skip
    graph.save(workspace / g.GRAPH_FILENAME)
    assert (
        "- the run (blocked, not_delivered): delivery files: gsm8k.yaml has no entry for m"
    ) in report.render(workspace, verdict="escalated")
    # A run that closed short of delivered says what kept it, as the deliver stage wrote it.
    (workspace / "journal.md").write_text(
        "## t0 — delivery blocked\n\nold\n\n## t1 — contract\n\nholds\n\n"
        "## t2 — delivery blocked\n\ndelivery files: tests: INCOMPLETE (7 not run)\n\n"
        "## t3 — run closed\n\n- verdict: **delivery-failed**\n"
    )
    assert report.delivery_gaps(workspace) == ["delivery files: tests: INCOMPLETE (7 not run)"]
    failed = report.render(workspace, verdict="delivery-failed")
    assert "## What kept it from delivered\n\n- delivery files: tests: INCOMPLETE" in failed
    assert "What kept it from delivered" not in text
    # A run with nothing to say says so, in each section.
    empty = report.render(tmp_path / "none", verdict="escalated")
    assert "**Not onboarded**" in empty and "None: the run was judged by" in empty
    assert "Nothing: every task of the run is done." in empty
    # A second run, on a model an earlier run onboarded, pins none: it says whose the
    # milestone is, not that there is none.
    second = tmp_path / "second"
    second.mkdir()
    (second / "task.yaml").write_text("onboarded: refs/mv2/the-run/onboarded\n")
    said = report.render(second, verdict="delivered")
    assert "**Onboarded before this run** (`onboarded: refs/mv2/the-run/onboarded`" in said
    assert "is not pinned by this run" in said and "**Not onboarded**" not in said


def test_closing_a_run_writes_the_report_and_names_it_in_its_journal(tmp_path, capsys):
    workflow = _workflow(tmp_path)
    _closed(workflow.workspace)
    state = WorkflowState(task_path=str(workflow.task_path), stage="execution")
    workflow._close_run(state, None, verdict="partial")
    written = (workflow.workspace / report.REPORT_FILENAME).read_text()
    assert written.startswith("# modeling_v2 target: partial") and "## Not verified" in written
    # One place says where the run stands, and the journal's closing lines name it.
    assert "**Onboarded** at `c0ffee1234` (`refs/mv2/run1/onboarded`)" in written
    assert "| anchor `tp1:gsm8k` | requirement | declared 61.77 | declared 41.77 |" in written
    journal = workflow.journal_path.read_text()
    assert f"- closing report: `{workflow.workspace / report.REPORT_FILENAME}`" in journal
    assert "- verdict: **partial**" in journal and "anchor `tp1:gsm8k`" not in journal
    out = " ".join(capsys.readouterr().out.split())
    assert "run closed — verdict partial; the closing report is" in out and state.done
    # A run with no holder of a scheduler states no node-hours. One with a bound and a
    # holder says the pool's own line: what was asked so far, of how much.
    assert "Asked of the scheduler" not in written
    spool = tmp_path / "spool"
    (spool / "holders").mkdir(parents=True)
    (spool / "holders" / "1.json").write_text(
        json.dumps({"seq": 1, "name": "run-pool-1", "nodes": 1, "time_s": 14400})
    )
    workflow.task_data["environment"] = {
        "kind": "slurm",
        "spool": str(spool),
        "slurm": {"time": "04:00:00", "max_node_hours": 48},
    }
    workflow._write_report("partial")
    written = (workflow.workspace / report.REPORT_FILENAME).read_text()
    line = "Asked of the scheduler: node-hours asked: 4.0 of 48 (1 holder job(s))."
    assert line in written.splitlines()


def test_a_changed_spec_reaches_the_agents_copy_and_the_environment_the_gates_run_in(
    tmp_path, task_file
):
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
        resolve_environment,
    )
    from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import load_and_validate_task_yaml

    workflow = _workflow(tmp_path)
    workflow.task_data = load_and_validate_task_yaml(task_file)
    workflow.environment = before = resolve_environment(workflow.task_data)
    esc.record_change(workflow.workspace, workflow.task_data, "environment.env.MY_SWITCH", "1",
                      why="the engine reads it", command="c1")  # fmt: skip
    workflow._respec(["environment.env.MY_SWITCH"])
    copy = yaml.safe_load(workflow.task_path.read_text())
    assert copy["environment"]["env"]["MY_SWITCH"] == "1"
    assert workflow.environment is not before
    assert "spec changed" in workflow.journal_path.read_text()
    # A key outside the environment rewrites the copy and leaves the environment as it is.
    kept = workflow.environment
    esc.record_change(workflow.workspace, workflow.task_data, "notes", "use the fused norm",
                      why="asked", command="c2")  # fmt: skip
    workflow._respec(["notes"])
    assert yaml.safe_load(workflow.task_path.read_text())["notes"] == "use the fused norm"
    assert workflow.environment is kept


def test_the_pools_verbs_ask_the_runs_own_pool_with_what_its_tasks_need(tmp_path, monkeypatch):
    workflow = _workflow(tmp_path)
    workflow.task_data = {"deployments": [{"name": "tp1", "options": {}, "timeout": 300}]}
    asked = []

    class Pool:
        paths = SimpleNamespace(root=tmp_path / "spool")
        env = "old"

        def close(self, seq, why):
            asked.append(("close", seq, why))
            return {"name": f"holder-{seq}"}

        def reconcile(self, demand, retry_now=False):
            asked.append(("reconcile", dict(demand), retry_now))
            return [pool_mod.Event("POOL_CAPACITY", "1", "holder 4 has waited 20m")]

    # A run that keeps no pool says so and does nothing.
    assert "keeps no pool" in workflow._pool_retry(None, "n") and asked == []
    workflow._pool = Pool()
    workflow._task_demand = lambda: {2: 9000.0}
    said = workflow._pool_retry(3, "squeue shows nothing under that name")
    assert asked == [
        ("close", 3, "squeue shows nothing under that name"),
        ("reconcile", {2: 9000.0}, True),
    ]
    assert said.startswith("record 3 (holder-3) closed; asked again: POOL_CAPACITY")
    # pool-adjust: the pool's environment is built again from the changed spec, then asked.
    built = []

    def build(task_data, **kwargs):
        built.append(dict(task_data))
        return SimpleNamespace(env="new")

    monkeypatch.setattr(pool_mod, "build", build)
    said = workflow._pool_adjust()
    assert built == [workflow.task_data] and workflow._pool.env == "new"
    assert asked[-1] == ("reconcile", {2: 9000.0}, True)
    assert said.startswith("the pool took the new values and asked again: POOL_CAPACITY")
    # Before the run has a task list the demand is the start's: the longest session.
    workflow._task_demand = None
    workflow._pool_retry(None, "again")
    assert asked[-1] == ("reconcile", {1: 900.0}, True)


@pytest.mark.parametrize("escalated, status", [(True, 76), (False, None), ("at its bound", 76)])
def test_a_run_whose_remaining_work_is_all_blocked_exits_76(
    tmp_path, task_file, monkeypatch, escalated, status
):
    # So does a run stopped at its bound of node-hours: it waits for an answer, and the same
    # command run again before the bound is raised would only stop again. 75 is for a run
    # that goes on when the command is run again.
    monkeypatch.setattr(cli, "_commit_identity", lambda task_data: CommitIdentity("T", "t@e.com"))
    # The command pins the run's identity in its own environment for the rest of the
    # process: here that is this suite's, so each variable is put back when the test ends.
    for name in ("MV2FLOW_GIT_NAME", "MV2FLOW_GIT_EMAIL", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL",
                 "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "CLAUDE_CODE_DISABLE_AUTO_MEMORY"):  # fmt: skip
        monkeypatch.setenv(name, "set by the test")
    ran = []

    def run(self, task):
        ran.append(str(task))
        self._escalated = escalated is True
        self._budget_stop = escalated == "at its bound"

    def left(code: int) -> None:
        raise SystemExit(code)  # a run at its bound leaves without waiting for its turns

    monkeypatch.setattr(TrtllmModelingV2Workflow, "run", run)
    monkeypatch.setattr(cli.os, "_exit", left)
    argv = ["--task", str(task_file), "--workspace", str(tmp_path / "ws")]
    if status is None:
        assert cli.main(argv) is None
    else:
        with pytest.raises(SystemExit) as done:
            cli.main(argv)
        assert done.value.code == status == cli.EXIT_ESCALATED
    assert ran == [str(task_file)]


def test_a_path_named_from_where_a_person_stood_is_whole_before_anything_reads_it(
    tmp_path, task_file, monkeypatch, environment_restored
):
    """A job runs in its pin's tree on a node: a workspace named `ws` is not there."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import controller

    monkeypatch.setattr(cli, "_commit_identity", lambda task_data: CommitIdentity("T", "t@e.com"))
    ran, continued, asked = [], [], []
    monkeypatch.setattr(
        TrtllmModelingV2Workflow, "run", lambda self, task: ran.append((self.workspace, task))
    )
    monkeypatch.setattr(
        controller, "watch_limit", lambda workspace, argv, **how: continued.append(list(argv))
    )
    monkeypatch.chdir(tmp_path)
    here = Path.cwd()
    spec = os.path.relpath(task_file, here)
    assert not os.path.isabs(spec)
    assert cli.main(["--task", spec, "--workspace=ws", "--max-attempts", "2"]) is None
    # The run, and with it every job and record it makes, has the workspace by its whole path.
    assert ran == [(here / "ws", here / spec)]
    # So has the command that continues it: it is said to whoever reads the log, anywhere.
    assert continued == [
        ["--task", str(here / spec), f"--workspace={here / 'ws'}", "--max-attempts", "2"]
    ]
    assert cli._parse_args(["--task", spec]).workspace.is_absolute()
    # The other commands take the same workspace. A task's name (`resolve --task`) is no path.
    monkeypatch.setitem(cli.SUBCOMMANDS, "resolve", lambda argv: asked.append(argv) or 0)
    with pytest.raises(SystemExit):
        cli.main(["resolve", "--work", "ws", "retry", "run", "--task", "assemble-core"])
    assert asked == [["--work", str(here / "ws"), "retry", "run", "--task", "assemble-core"]]


_BLOCKED = """
import sys, time
from concurrent.futures import ThreadPoolExecutor
from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.infra.git import CommitIdentity
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

class Writing:
    # Standard output as a signal's handler finds it when the signal came during a write.
    def write(self, text):
        return sys.__stdout__.write(text)

    def flush(self):
        raise RuntimeError("reentrant call inside <_io.BufferedWriter name='<stdout>'>")

def run(self, task):
    # As a run is: the main thread waits on worker threads that are not daemons.
    print("RUNNING", flush=True)
    sys.stdout = Writing()
    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(lambda _: time.sleep(300), range(2)))

cli._commit_identity = lambda task_data: CommitIdentity("T", "t@e.com")
TrtllmModelingV2Workflow.run = run
cli.main(sys.argv[1:])
"""


def test_a_controller_told_to_stop_leaves_at_once(tmp_path, task_file):
    """TERM, as a job's script or the scheduler sends it: it does not wait to be killed.

    Nor does it matter what the signal interrupted: a write under way makes the handler's
    own flush raise, and the controller still leaves with its mark.
    """
    script = tmp_path / "blocked.py"
    script.write_text(_BLOCKED)
    flow = str(Path(cli.__file__).resolve().parents[3])
    proc = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--task",
            str(task_file),
            "--workspace",
            str(tmp_path / "ws"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONPATH": flow},
    )
    try:
        assert proc.stdout.readline().strip() == "RUNNING"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=15) == 143, "128 + TERM: stopped from outside, not closed"
    finally:
        proc.kill()
    mark = tmp_path / "ws" / heartbeat_mod.DISPATCH_DIRNAME / heartbeat_mod.ORCHESTRATOR_EXIT
    # Whoever watches the run reads that its controller left, and why.
    assert json.loads(mark.read_text())["why"] == "told to stop (SIGTERM)"


@pytest.fixture
def environment_restored():
    """The CLI pins the run's commit identity in the process environment: put back what was there."""
    before = dict(os.environ)
    yield
    for name in set(os.environ) - set(before):
        del os.environ[name]
    os.environ.update(before)


def test_a_start_that_did_not_start_exits_with_a_status_that_is_not_zero_and_says_why(
    tmp_path, task_file, monkeypatch, environment_restored
):
    """In the controller's own job a status of 0 reads COMPLETED: a start that did nothing is not that."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.monitor import status
    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

    why = "the session of `tp1` is 16800s, longer than one holder (14400s)"

    def refused(task_data, **kwargs):
        raise bootstrap.PreflightError(why)

    monkeypatch.setattr(bootstrap, "run_preflight", refused)
    for name, more in (("ws-start", []), ("ws-dry", ["--dry-run"])):
        workspace = tmp_path / name
        with pytest.raises(SystemExit) as done:
            cli.main(["--task", str(task_file), "--workspace", str(workspace), *more])
        assert done.value.code == cli.EXIT_NOT_STARTED == 2
        # The exit record carries the run's own reason, and the monitor's line says it.
        state, detail = status.orchestrator_state(status.load_config(workspace))
        assert state == "exited" and f"the start did not start: preflight: {why}" in detail


def test_a_start_that_did_not_start_inside_a_job_of_its_own_ends_its_holder(
    tmp_path, monkeypatch, capsys
):
    """Nobody types the command again there: the first holder would idle to its limit."""
    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

    record = tmp_path / "pool.json"
    record.write_text(json.dumps({"holders": [{"name": "run-pool-1"}]}))

    class Pool:
        env = SimpleNamespace(kind=pool_mod.KIND_SLURM)
        paths = SimpleNamespace(pool=record)

        def __init__(self):
            self.ended = 0

        def done(self):
            self.ended += 1
            return []

    def ended(not_started: str) -> int:
        pool = Pool()
        workflow = SimpleNamespace(
            _pool=pool, _pool_keeper=None, _not_started=not_started, workspace=tmp_path
        )
        bootstrap.end_pool(workflow, SimpleNamespace(done=False), None)
        return pool.ended

    probe = "the agent probe: an agent of the spec cannot run a turn"
    # Typed by a person: the holder stays for the same command.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert ended(probe) == 0
    assert "its holder(s) stay (run-pool-1)" in " ".join(capsys.readouterr().out.split())
    # Inside a job of its own (a site's script submitted the command): ended, and said.
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    assert ended(probe) == 1
    said = " ".join(capsys.readouterr().out.split())
    assert "no command resumes on its holder(s), so they are ended" in said
    # A run that started and stopped short there keeps them: what continues it resumes on them.
    assert ended("") == 0
