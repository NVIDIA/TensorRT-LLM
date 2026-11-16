# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task is done on evidence: its card's check, in the ledger, under the pin of its tree."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import pintree
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.operators import rows as rows_mod
from agent_flow.workflows.trtllm_modeling_v2.run import evidence
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.execution import tasks as tasks_mod
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import TaskResult
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import Integration

from ._helpers import passing_shell

_ENV = {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}
_TARGET = f"{catalog_paths.MODELING_V2_PACKAGE}/models/fam/m__sm_100__unified"
_TEST = f"{catalog_paths.model_ops_tests_dir(_TARGET)}/norm/test_modeling_v2_rms.py"
_NODE = "tests/integration/defs/accuracy/test_disaggregated_serving.py::TestStep5::test_dep4"
_PASS = '<testsuite><testcase name="native"/></testsuite>'
_FAIL = '<testsuite><testcase name="native"><failure/></testcase></testsuite>'


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def run(tmp_path):
    """A run's workspace over a real checkout: its spec, its graph, its vocabulary, one task's tree."""
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm").mkdir(parents=True)
    (repo / "tensorrt_llm" / "__init__.py").write_text("")
    (repo / Path(_TEST).parent).mkdir(parents=True)
    (repo / _TEST).write_text("def test_rms():\n    pass\n")
    (repo / "tests/integration/defs/accuracy").mkdir(parents=True)
    (repo / "tests/integration/defs/accuracy/test_disaggregated_serving.py").write_text("")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"architectures": ["MForCausalLM"]}))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    moves = {"cache_transceiver_config": {"backend": "DEFAULT"}}
    spec = {
        "checkpoint_path": str(ckpt),
        "trtllm_repo_path": str(repo),
        "arch": "sm_100",
        "target": {"family": "fam", "checkpoint_segment": "m"},
        "environment": {"kind": "local", "python": sys.executable},
        "accuracy": {"benchmark": "gsm8k"},
        "deployments": [
            {
                "name": "tep2",
                "options": {"tensor_parallel_size": 2, "moe_expert_parallel_size": 2},
                "benchmarks": [{"name": "gsm8k", "anchor": 92.0, "source": "measured"}],
            },
            {"name": "pd", "ctx": {"options": moves}, "gen": {"options": moves},
             "benchmarks": [], "tests": [_NODE]},
        ],
    }  # fmt: skip
    task_path = workspace / "task.yaml"
    task_path.write_text(yaml.safe_dump(spec))
    g.Graph(tasks=[g.Task(id="survey", kind="survey", goal="s", status="done")]).save(
        workspace / g.GRAPH_FILENAME
    )
    vocabulary = {
        "steps": [],
        "queue": [{"n": 1, "mode": "onboard", "target": "ops/norm/rms"}],
    }
    (workspace / "vocabulary.yaml").write_text(yaml.safe_dump(vocabulary))
    # As the run's start leaves it: the contract the controller compiled.
    contract_mod.write_contract(workspace, contract_mod.compile_contract(spec))
    integration = Integration(repo, workspace, "mv2/run", _ENV)
    tree = integration.open("k1-rms")
    return SimpleNamespace(
        repo=repo, workspace=workspace, task_path=task_path, spec=spec, tree=tree,
        integration=integration,
    )  # fmt: skip


@pytest.fixture
def shell(monkeypatch):
    """Every job the gate submits is answered by a stand-in; a native test writes ``junit``."""
    state = {"seen": [], "junit": _PASS, "score": 92.5}

    def fake_runner_for(executor, **kwargs):
        def answer(command, env=None, timeout=0):
            state["seen"].append(command)
            report = re.search(r"--junitxml=([^\s']+)", command)
            if report and state["junit"] is not None:
                Path(report.group(1)).write_text(state["junit"], encoding="utf-8")
            return passing_shell(score=state["score"])(command, env, timeout)

        return answer

    monkeypatch.setattr(gate_cli, "shell_runner_for", fake_runner_for)
    return state


def _marker(run, check="kernel:1", tree=None):
    """The key ``check``'s evidence is recorded under, on a pin of the worker's tree."""
    commit = pintree.freeze(tree or run.tree, "test")
    return evidence.key_of(run.spec, run.workspace, commit, check)


def _argv(run, check, task="k1-rms", tree=None):
    return tasks_mod.check_argv(
        check,
        task_path=run.task_path,
        tree=str(tree or run.tree),
        workdir=run.workspace / "gates" / task,
    )


def test_each_card_command_is_the_gates_own_and_records_its_result_under_the_pin(
    run, shell, capsys
):
    # kernel:<row>: the row's own tests, on a pin of the worker's tree.
    assert evidence.main(_argv(run, "kernel:1")) == 0
    out, err = capsys.readouterr()
    assert out.startswith("check kernel:1\n[PASS] kernel:1: ")
    assert "recorded in the run's ledger: kernel:1 on the pin" in err
    assert "test_modeling_v2_rms.py" in shell["seen"][-1]
    found = evidence.lookup(run.workspace, "kernel:1", _marker(run))
    assert found is not None and found.passed and found.unit == evidence.ROW_UNIT
    # deployment:<name>: one session, each of its checks a line, as the run's own records them.
    assert evidence.main([*_argv(run, "deployment:tep2"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "deployment:tep2" and payload["passed"] is True
    marker = _marker(run, "deployment:tep2")
    session = evidence.lookup(run.workspace, "deployment:tep2", marker)
    assert session.passed and (session.unit, session.gate, session.member) == (
        "tep2",
        "deployment",
        "tep2",
    )
    kept = {f"{v.unit}:{v.gate}@{v.member}" for v in evidence.verdicts(run.workspace)}
    assert {"tep2:boot@tep2", "tep2:accuracy[gsm8k]@tep2", "tep2:deployment@tep2"} <= kept
    # The contract's own reading of the ledger takes it: on that tree the session is not owed again.
    contract = contract_mod.compile_contract(run.spec)
    standing = contract_mod.evaluate(
        contract, contract_mod.ContractStatus.load(run.workspace), marker=marker
    )
    assert "tep2:deployment@tep2" in {c.key for c in standing.passed}
    assert session.inputs == next(c.inputs for c in contract.checks if c.unit == "tep2")


def test_evidence_is_for_one_pin_a_failure_is_evidence_too_and_a_spec_outside_a_run_records_nothing(
    run, shell, capsys, tmp_path
):
    assert evidence.main(_argv(run, "kernel:1")) == 0
    before = _marker(run)
    # The same files committed are the same evidence; other files are another pin.
    _git(run.tree, "commit", "-q", "--allow-empty", "-m", "nothing changed")
    assert _marker(run) == before
    (run.tree / "tensorrt_llm" / "new.py").write_text("X = 1\n")
    after = _marker(run)
    assert after != before and evidence.lookup(run.workspace, "kernel:1", after) is None
    # A check that fails is recorded as failed, for the pin it failed on.
    shell["junit"] = _FAIL
    assert evidence.main(_argv(run, "kernel:1")) == 1
    capsys.readouterr()
    failed = evidence.lookup(run.workspace, "kernel:1", after)
    assert failed is not None and not failed.passed
    assert evidence.lookup(run.workspace, "kernel:1", before).passed
    # A usage error is the gate's, and leaves nothing.
    lines = len(evidence.verdicts(run.workspace))
    assert evidence.main(_argv(run, "kernel:7")) == 2
    assert "names no row of" in capsys.readouterr().err
    assert len(evidence.verdicts(run.workspace)) == lines
    # Outside a run (no graph, no state) the gate runs as it is and nothing is recorded.
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "task.yaml").write_text(run.task_path.read_text())
    (outside / "vocabulary.yaml").write_text((run.workspace / "vocabulary.yaml").read_text())
    shell["junit"] = _PASS
    argv = ["kernel", "--task", str(outside / "task.yaml"), "--row", "1", "--tree", str(run.tree)]
    assert evidence.main([*argv, "--workdir", str(outside / "w")]) == 0
    assert capsys.readouterr().out.startswith("check kernel:1\n[PASS]")
    assert not (outside / contract_mod.STATUS_FILENAME).exists()


def test_a_row_a_split_wrote_is_found_by_the_gate_in_its_own_vocabulary(run, shell, capsys):
    rows = rows_mod.split_rows(
        [{"id": "norm-pre", "kind": "operator", "mode": "onboard", "target": "ops/norm/pre",
          "outcome": "the pre-norm half", "cells": [{"dtype": "bf16"}]}], taken=["k1-rms"],
    )  # fmt: skip
    (row,) = [rows_mod.write_row_vocabulary(run.workspace, r) for r in rows]
    assert row.check == "kernel:ops/norm/pre"
    test = run.tree / Path(_TEST).parent / "test_modeling_v2_pre.py"
    test.write_text("def test_pre():\n    pass\n")
    argv = tasks_mod.check_argv(
        row.check, task_path=run.task_path, tree=str(run.tree),
        workdir=run.workspace / "gates" / "norm-pre", vocabulary=row.config_axes["vocabulary"],
    )  # fmt: skip
    assert argv[:6] == [
        "kernel",
        "--task",
        str(run.task_path),
        "--row",
        "ops/norm/pre",
        "--vocabulary",
    ]
    assert evidence.main(argv) == 0
    assert capsys.readouterr().out.startswith("check kernel:ops/norm/pre\n[PASS]")
    assert "test_modeling_v2_pre.py" in shell["seen"][-1]
    assert evidence.lookup(run.workspace, row.check, _marker(run, row.check)).passed
    # The survey's own vocabulary is left as it is.
    assert "pre" not in (run.workspace / "vocabulary.yaml").read_text()


class _Journal:
    def __init__(self):
        self.events: list[str] = []

    def event(self, kind, text):
        self.events.append(f"{kind}: {text}")


class _Stage(tasks_mod.ExecutionStage):
    """What a task's landing uses of the workflow; the worker and the coordinator's gate are scripted."""

    def __init__(self, run, worker, gate=None):
        self.ran: list[list[str]] = []

        def process(argv, cli=()):
            self.ran.append(list(argv))
            return gate(argv) if gate is not None else (2, "no gate here")

        super().__init__(
            SimpleNamespace(
                workspace=run.workspace,
                task_path=run.task_path,
                task_data=run.spec,
                _tls=threading.local(),
                journal=_Journal(),
                _gate_process=process,
            )
        )
        self._integration = run.integration
        self.worker = worker
        self.turns = 0

    def _work_task_body(self, task):
        self.turns += 1
        return TaskResult(self.worker(self.turns))


def _commit(tree, name, text):
    (tree / name).write_text(text)
    _git(tree, "add", name)
    _git(tree, "commit", "-q", "-s", "-m", f"write {name}")


def _task(**more):
    return g.Task(id="k1-rms", kind="kernel", goal="the entry", ref="1", check="kernel:1", **more)


def _line(passed, detail="1 test"):
    return [{"name": "kernel:1", "member": None, "passed": passed, "detail": detail}]


def test_a_ready_receipt_lands_on_the_result_its_workers_own_run_left_for_the_pin(run, shell):
    def worker(turn):
        _commit(run.tree, "entry.py", "E = 1\n")
        # The worker runs its card's command, which records itself.
        assert evidence.main(_argv(run, "kernel:1")) == 0
        return "ready"

    stage = _Stage(run, worker)
    assert stage._work_task_turn(_task()).decision == "ready"
    assert (run.repo / "entry.py").read_text() == "E = 1\n"
    # The coordinator ran nothing: the ledger had the result for the pin.
    assert stage.ran == []
    assert any("evidence: k1-rms: `kernel:1` passed on" in e for e in stage.workflow.journal.events)


def test_a_session_whose_checkpoint_is_not_there_has_no_key_and_no_job_is_spent_on_it(run):
    """A name with nothing behind it identifies no content: nothing is reused, and it is said."""
    session = "deployment:pd"
    commit = _git(run.tree, "rev-parse", "HEAD").strip()
    there = evidence.key_of(run.spec, run.workspace, commit, session)
    assert there.startswith("key:")
    run.spec["checkpoint_path"] = str(run.workspace / "a-checkpoint-nobody-mounted")
    assert evidence.key_of(run.spec, run.workspace, commit, session) == ""
    # A row's own tests load no checkpoint: their key stands.
    assert evidence.key_of(run.spec, run.workspace, commit, "kernel:1").startswith("key:")
    stage = _Stage(run, lambda turn: "ready")
    task = g.Task(id="k1-rms", kind="kernel", goal="the entry", ref="1", check=session)
    proof = stage._evidence(task, run.integration, run.tree)
    assert not proof.passed and "has no key on the pin" in proof.said
    assert "the checkpoint's path is not there" in proof.said
    assert stage.ran == [], "no gate is run to find that no result can count"


def test_a_result_for_an_earlier_state_of_the_tree_is_not_evidence_for_what_is_handed_in(
    run, shell
):
    seen = []

    def gate(argv):
        # The same command a card gives; it records what it measured, as the gate does.
        assert argv == _argv(run, "kernel:1")
        seen.append(_marker(run))
        evidence.record(run.workspace, run.spec, "kernel:1", _line(True), _marker(run))
        return 0, ""

    def worker(turn):
        _commit(run.tree, "entry.py", "E = 1\n")
        assert evidence.main(_argv(run, "kernel:1")) == 0
        # Edited after its own run: what it hands in is not what was measured.
        _commit(run.tree, "entry.py", "E = 2\n")
        return "ready"

    stage = _Stage(run, worker, gate)
    assert stage._work_task_turn(_task()).decision == "ready"
    assert len(stage.ran) == 1 and (run.repo / "entry.py").read_text() == "E = 2\n"
    said = stage.workflow.journal.events
    assert any("has no result for the pin" in e and "runs it once" in e for e in said)


def test_a_result_that_failed_nothing_and_could_not_verify_lands_and_is_recorded_as_such(
    run, monkeypatch, capsys
):
    why = "hop: 2 of 2 response(s) name no transfer interval"
    lines = [
        {"name": "boot", "member": "tep2", "passed": True, "detail": "answered"},
        {"name": "deployment", "member": "tep2", "passed": True, "unverified": why,
         "detail": "2 check(s): nothing failed, 1 not verified"},
    ]  # fmt: skip

    def partial(argv):
        # The front door's third outcome: exit 3, `passed` false, `status` partial.
        payload = {"id": "deployment:tep2", "status": "partial", "passed": False,
                   "unverified": why, "lines": lines}  # fmt: skip
        print(json.dumps(payload))
        return gate_cli.EXIT_STATUS[gate_cli.PARTIAL]

    monkeypatch.setattr(gate_cli, "main", partial)
    _commit(run.tree, "model.py", "M = 1\n")
    # The worker's own run: the gate's status, its closing line, and the line in the ledger.
    argv = _argv(run, "deployment:tep2", task="assemble-tep2")
    assert evidence.main(argv) == 3
    out, err = capsys.readouterr()
    closing = f"partial (exit 3): nothing failed, and this does not certify: {why}"
    assert out.rstrip().endswith(closing)
    assert "(not verified: nothing failed)" in err
    found = evidence.lookup(run.workspace, "deployment:tep2", _marker(run, "deployment:tep2"))
    assert found.passed and found.unverified == why
    # The contract reads it as the third outcome: not passed, not failed, not missing.
    standing = contract_mod.evaluate(
        contract_mod.compile_contract(run.spec),
        contract_mod.ContractStatus.load(run.workspace),
        marker=_marker(run, "deployment:tep2"),
    )
    assert [c.key for c, _ in standing.unverified] == ["tep2:deployment@tep2"]
    # Every check of the card ran and none failed: the work lands, with no run of the
    # coordinator's own (the result is there), no attempt spent and no escalation. What was
    # not verified is recorded on the landing, never as passed.
    stage = _Stage(run, lambda turn: "ready")
    marks = []
    stage._mark_integrated = lambda task_id, attempt, said: marks.append((task_id, attempt, said))
    task = g.Task(id="k1-rms", kind="assemble", goal="the target", ref="assemble-tep2",
                  check="deployment:tep2")  # fmt: skip
    result = stage._work_task_turn(task)
    assert result.decision == "ready" and stage.ran == [] and task.attempts == 1
    assert "blocker_kind" not in result.data
    assert (run.repo / "model.py").read_text() == "M = 1\n"
    assert marks == [("k1-rms", 1, f"deployment:tep2: {why}")]
    assert any("failed nothing and could not verify" in e for e in stage.workflow.journal.events)
    assert any("landed NOT VERIFIED" in e for e in stage.workflow.journal.events)


@pytest.mark.parametrize("left", ["a failure", "nothing"])
def test_a_receipt_alone_lands_nothing(run, left):
    def gate(argv):
        if left == "a failure":
            evidence.record(
                run.workspace,
                run.spec,
                "kernel:1",
                _line(False, "1 of 3 cells failed"),
                _marker(run),
            )
            return 1, "[FAIL] kernel:1: 1 of 3 cells failed"
        return 2, "error: row 1 has no test in the tree it measures"

    def worker(turn):
        _commit(run.tree, "entry.py", "E = 1\n")
        return "ready"

    stage = _Stage(run, worker, gate)
    result = stage._work_task_turn(_task())
    assert result.decision == tasks_mod.CHECK_NOT_PASSED and not (run.repo / "entry.py").exists()
    assert len(stage.ran) == 1
    if left == "a failure":
        assert "`kernel:1` did not pass on the pin" in result.note
        assert "1 of 3 cells failed" in result.note
    else:
        assert "left no result for the pin" in result.note and "the gate exited 2" in result.note
        assert "has no test in the tree it measures" in result.note
    # A row's own tests are evidence too: `ready` with no result for them lands nothing.
    plain = g.Task(id="k1-rms", kind="weights", goal="w", ref="weights", check="cpu:weights",
                   touches=["entry.py", "tests/unit/test_w.py"])  # fmt: skip
    stage = _Stage(run, lambda turn: "ready", lambda argv: (2, "no evidence"))
    assert stage._work_task_turn(plain).decision == tasks_mod.CHECK_NOT_PASSED
    assert [argv[0] for argv in stage.ran] == ["cpu"] and "--test" in stage.ran[0]


def test_the_coordinators_own_run_is_the_cards_command_in_a_process_of_its_own(run):
    """The real front door, a real process, a real (empty) test: no stand-in anywhere."""
    _commit(run.tree, "entry.py", "E = 1\n")
    status, tail = evidence.run_gate_process(_argv(run, "kernel:1"))
    assert status == 0, tail
    assert "check kernel:1" in tail and "[PASS] kernel:1" in tail
    found = evidence.lookup(run.workspace, "kernel:1", _marker(run))
    assert found is not None and found.passed
    # And through the stage: one run, then the landing.
    stage = _Stage(run, lambda turn: "ready")
    stage.workflow._gate_process = None
    (run.tree / "entry.py").write_text("E = 3\n")
    _git(run.tree, "commit", "-q", "-s", "-am", "again")
    assert stage._work_task_turn(_task()).decision == "ready"
    assert (run.repo / "entry.py").read_text() == "E = 3\n"


def test_a_rows_own_tests_run_through_the_front_door_in_a_process_of_their_own(run):
    """`gate cpu`, a real process, a real test file, no device and no stand-in anywhere."""
    _commit(run.tree, "entry.py", "E = 1\n")
    argv = tasks_mod.check_argv(
        "cpu:weights", task_path=run.task_path, tree=str(run.tree), tests=(_TEST,),
        workdir=run.workspace / "gates" / "weights", wall=300,
    )  # fmt: skip
    assert argv[:2] == ["cpu", "--task"] and "--gpus" not in argv
    status, tail = evidence.run_gate_process(argv)
    assert status == 0, tail
    assert "check cpu:weights" in tail and "[PASS] cpu:weights" in tail
    found = evidence.lookup(run.workspace, "cpu:weights", _marker(run, "cpu:weights"))
    assert found is not None and found.passed and found.unit == evidence.ROW_UNIT
    # A row that names a test the tree does not have fails: nothing could show its outcome.
    missing = [
        *argv[: argv.index("--test")],
        "--test",
        "tests/none/test_x.py",
        *argv[argv.index("--test") + 2 :],
    ]
    status, tail = evidence.run_gate_process(missing)
    assert status == 1 and "has no test in the tree it measures" in tail
    # Through the landing: the coordinator runs it once, and the row lands on that result.
    stage = _Stage(run, lambda turn: "ready")
    stage.workflow._gate_process = None
    (run.tree / "entry.py").write_text("E = 4\n")
    _git(run.tree, "commit", "-q", "-s", "-am", "again")
    row = g.Task(id="k1-rms", kind="weights", goal="w", ref="weights", check="cpu:weights",
                 touches=["entry.py", _TEST])  # fmt: skip
    assert stage._work_task_turn(row).decision == "ready"
    assert (run.repo / "entry.py").read_text() == "E = 4\n"


def test_the_workspaces_own_command_is_replaced_whole_and_never_written_in_place(tmp_path):
    """A start died with "Text file busy": something was running the file it wrote again."""
    script = gate_cli.install_cli(tmp_path)
    first, text = script.stat().st_ino, script.read_text()
    assert gate_cli.install_cli(tmp_path) == script
    # A new file renamed over the old one: whoever runs the old one keeps it.
    assert script.stat().st_ino != first and script.read_text() == text
    assert script.stat().st_mode & 0o111 == 0o111
    assert [p.name for p in script.parent.iterdir()] == [gate_cli.CLI_NAME]


def test_a_changed_requirement_takes_the_pass_judged_by_the_old_one_with_it(run):
    status = contract_mod.ContractStatus.load(run.workspace)
    marker = "m1"
    line = SimpleNamespace(name="deployment", member="tep2", passed=True, detail="ok")
    status.record("tep2", line, marker=marker, inputs="i1")
    assert evidence.void_passes(
        status, ["tep2:deployment@tep2", "pd:deployment@pd2"], "anchor 92 -> 90"
    ) == ["tep2:deployment@tep2"]
    latest = contract_mod.ContractStatus.load(run.workspace).latest()["tep2:deployment@tep2"]
    assert not latest.passed and latest.void == "anchor 92 -> 90"
    assert (latest.marker, latest.inputs) == (marker, "i1")
    # Nothing is removed: the pass is still in the ledger, behind the line that voids it.
    assert [v.passed for v in evidence.verdicts(run.workspace)] == [True, False]
    assert evidence.void_passes(status, ["tep2:deployment@tep2"], "again") == []
