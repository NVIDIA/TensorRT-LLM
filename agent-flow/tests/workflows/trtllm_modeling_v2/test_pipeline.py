# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run: the fixed pipeline, the planner's repairs, resume, and the final certification."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.git import CommitIdentity, CommitPolicy
from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run import inbox
from agent_flow.workflows.trtllm_modeling_v2.run.integration import (
    Integration,
    load_labels,
    own_paths,
)
from agent_flow.workflows.trtllm_modeling_v2.run.state import (
    STAGE_EXECUTION,
    WorkflowState,
    load_state,
    save_state,
)

from ._helpers import (
    TARGET_REL,
    Script,
    _install,
    _run,
    _state,
    _task_id,
    _tasks,
    _workflow,
    _write_vocabulary,
    passing_shell,
)

FAMILY = str(Path(TARGET_REL).parent)


def _verdicts(tmp_path) -> list[dict]:
    path = tmp_path / "ws" / contract_mod.STATUS_FILENAME
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _mtp_task(tmp_path, task_file, repo, **more) -> Path:
    """The fixture spec with one feature (``mtp``) and one configuration variant."""
    spec = yaml.safe_load(task_file.read_text())
    config = Path(spec["checkpoint_path"]) / "config.json"
    config.write_text(json.dumps({**json.loads(config.read_text()), "num_nextn_predict_layers": 1}))
    options = tmp_path / "mtp.yaml"
    options.write_text(
        yaml.safe_dump({"speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}})
    )
    spec["features"] = ["mtp"]
    spec["accuracy"]["llm_api_options"] = {"mtp": str(options)}
    spec["accuracy"]["discriminators"] = {"default": {}, "mtp": {"spec_dec_algo": "MTP"}}
    spec["variants"] = {"switched": {"env": {"MY_SWITCH": "1"}, "gates": ["accuracy"]}}
    spec.update(more)
    refs = repo / spec["accuracy"]["reference_yaml"]
    refs.write_text(
        yaml.safe_dump(
            {
                "openai/gpt-oss-120b": [
                    {"accuracy": 85.6},
                    {"accuracy": 85.6, "spec_dec_algo": "MTP"},
                ]
            }
        )
    )
    task_file.write_text(yaml.safe_dump(spec))
    return task_file


def _logging(events: list[str], script, shell=None):
    """``script`` and a shell that both append to ``events``: a turn per task, a check per command."""

    def scripted(role, turn, prompt):
        events.append(f"turn:{role}:{_task_id(prompt)}")
        return script(role, turn, prompt)

    inner = shell or passing_shell()

    def run(command, env=None, timeout=0):
        if "trtllm-eval" in command:
            events.append("check:accuracy")
        elif "engine_driver" in command or "quickstart" in command:
            events.append("check:engine")
        return inner(command, env, timeout)

    return scripted, run


# ---------------------------------------------------------------------- the fixed order


def test_the_tasks_run_in_the_fixed_order_and_every_check_runs_once_at_the_end(
    tmp_path, task_file, repo, monkeypatch
):
    """Survey, each unit's kernel rows then the unit, the variants; then one certification."""
    task = _mtp_task(tmp_path, task_file, repo)
    workspace = tmp_path / "ws"
    monkeypatch.setattr(
        "agent_flow.workflows.trtllm_modeling_v2.run.workflow.TrtllmModelingV2Workflow.survey_first",
        True,
    )

    def surveyor(turn, prompt):
        _write_vocabulary(
            workspace,
            [
                {"n": 1, "mode": "onboard", "target": "ops/norm/rms"},
                {"n": 2, "mode": "onboard", "target": "ops/spec/draft", "unit": "mtp"},
                {"n": 3, "mode": "onboard", "target": "ops/norm/scale", "depends_on": [1]},
            ],
        )
        return {"decision": "DRAFT_READY", "summary": "three rows"}

    events: list[str] = []
    script, shell = _logging(events, Script(surveyor=surveyor))
    calls = _run(tmp_path, task, monkeypatch, script, shell_runner=shell)

    assert _state(tmp_path).verdict == "delivered"
    assert [t["id"] for t in _tasks(tmp_path)] == [
        "survey",
        "k1-rms",
        "k3-scale",
        "core",
        "k2-draft",
        "mtp",
        "switched",
    ]
    assert all(t["status"] == "done" and t["attempts"] == 1 for t in _tasks(tmp_path))
    assert "planner" not in {role for role, _ in calls}
    # The workers are trusted while they build: no check runs before the last task ended.
    turns = [i for i, e in enumerate(events) if e.startswith("turn:")]
    checks = [i for i, e in enumerate(events) if e.startswith("check:")]
    assert checks and min(checks) > max(turns)
    # Every required check has exactly one verdict, all on one tree, all passed.
    contract = yaml.safe_load((workspace / "contract.yaml").read_text())["checks"]
    keys = sorted(f"{c['deployment']}:{c['gate']}" for c in contract)
    verdicts = _verdicts(tmp_path)
    assert sorted(f"{v['deployment']}:{v['gate']}" for v in verdicts) == keys
    assert {"mtp:accuracy[gsm8k]", "switched:accuracy[gsm8k]"} <= set(keys)
    assert not [key for key in keys if "acceptance" in key]  # a drafter's is a native test
    assert len({v["tree"] for v in verdicts}) == 1 and all(v["passed"] for v in verdicts)
    certification = yaml.safe_load((workspace / "certification.yaml").read_text())
    assert certification["certified"] and certification["round"] == 1
    assert certification["commit"] == "c0ffee1"
    report = (workspace / "delivery_report.md").read_text()
    assert report.startswith("# Delivery report: delivered") and "`tp1:boot` | PASS" in report
    # A worker is told which checks its unit owes and that it runs them itself.
    core_prompt = next(p for role, p in calls if role == "assembler" and _task_id(p) == "core")
    assert "## The checks `core` owes" in core_prompt and "`tp1:boot`" in core_prompt
    assert "gate <boot|accuracy|serve|tests>" in core_prompt and "--deployment tp1" in core_prompt
    mtp_prompt = next(p for role, p in calls if role == "assembler" and _task_id(p) == "mtp")
    assert "`core` (assemble): done UNIT_READY" in mtp_prompt


def test_a_kernel_row_runs_after_the_rows_it_depends_on_in_the_first_unit_that_needs_it():
    from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem
    from agent_flow.workflows.trtllm_modeling_v2.run.pipeline import _rows_by_unit

    queue = (
        QueueItem(1, "onboard", "ops/a/one", depends_on=(3,)),  # on a later row
        QueueItem(2, "onboard", "ops/a/two", unit="mtp"),
        QueueItem(3, "onboard", "ops/a/three"),
        QueueItem(4, "onboard", "ops/a/four", unit="mtp", depends_on=(2,)),
        QueueItem(5, "onboard", "ops/a/five", depends_on=(6,)),  # on a row of a later unit
        QueueItem(6, "onboard", "ops/a/six", unit="mtp"),
        QueueItem(7, "onboard", "ops/a/seven", unit="mtp", depends_on=(8,)),  # a circle
        QueueItem(8, "onboard", "ops/a/eight", unit="mtp", depends_on=(7,)),
    )
    grouped = _rows_by_unit(queue, ["core", "mtp"])
    assert [q.n for q in grouped["core"]] == [3, 1, 6, 5]
    assert [q.n for q in grouped["mtp"]] == [2, 4, 7, 8]


# -------------------------------------------------------------------------- the planner


def test_a_blocked_task_goes_to_the_planner_and_its_note_reaches_the_next_attempt(
    tmp_path, task_file, monkeypatch
):
    prompts: dict[str, list[str]] = {"assembler": [], "planner": []}

    def assembler(turn, prompt):
        prompts["assembler"].append(prompt)
        if turn == 1:
            return {
                "decision": "BLOCKED",
                "blocked_reason": "knowledge_gap",
                "summary": "the rotary layout is not in the config",
            }
        return {"decision": "UNIT_READY", "summary": "assembled"}

    def planner(turn, prompt):
        prompts["planner"].append(prompt)
        return {
            "decision": "CONTINUE",
            "summary": "a hint",
            "note": "read rope_theta from the HF module",
        }

    _run(tmp_path, task_file, monkeypatch, Script(assembler=assembler, planner=planner))
    assert _state(tmp_path).verdict == "delivered"
    [asked] = prompts["planner"]
    assert (
        "Task `core` (assemble, unit `core`, run by the assembler) failed on attempt 1 of 3"
        in asked
    )
    assert (
        "with the decision `BLOCKED`" in asked and "the rotary layout is not in the config" in asked
    )
    assert "Answer with exactly one of: `CONTINUE`, `REPAIR`, `ESCALATE`." in asked
    retry = prompts["assembler"][1]
    assert "## Why this task runs again" in retry and "Attempt 2 of 3" in retry
    assert "The planner says: read rope_theta from the HF module" in retry
    [core] = _tasks(tmp_path)[1:]
    assert core["status"] == "done" and core["attempts"] == 2


def test_a_repair_runs_its_tasks_before_the_task_it_repairs(tmp_path, task_file, monkeypatch):
    order: list[str] = []

    def assembler(turn, prompt):
        order.append(_task_id(prompt))
        if turn == 1:
            return {
                "decision": "BLOCKED",
                "blocked_reason": "vocabulary_gap",
                "missing_entry": "norm/my_norm",
                "summary": "no entry serves the gated norm",
            }
        return {"decision": "UNIT_READY", "summary": "assembled"}

    def kernel_writer(turn, prompt):
        order.append(_task_id(prompt))
        return {"decision": "LANDED", "summary": "landed"}

    answers = iter(
        [
            # Refused: a kernel task names its entry as onboard:<category>/<name>.
            {"decision": "REPAIR", "summary": "s", "tasks": [{"kind": "kernel", "ref": "my_norm"}]},
            {
                "decision": "REPAIR",
                "summary": "the gated norm needs an entry",
                "tasks": [
                    {"kind": "kernel", "ref": "onboard:norm/my_norm", "unit": "core", "goal": "g"},
                    {"kind": "fix", "unit": "core", "goal": "wire the entry into the block"},
                ],
            },
        ]
    )
    asked: list[str] = []

    def planner(turn, prompt):
        asked.append(prompt)
        return next(answers)

    _run(
        tmp_path,
        task_file,
        monkeypatch,
        Script(assembler=assembler, kernel_writer=kernel_writer, planner=planner),
    )
    assert _state(tmp_path).verdict == "delivered"
    assert order == ["core", "kx-1", "fix-1", "core"]
    # The planner is told what the worker said it lacks, not only that it blocked.
    assert "blocked_reason: vocabulary_gap; missing_entry: norm/my_norm" in asked[0]
    assert [t["id"] for t in _tasks(tmp_path)] == ["survey", "kx-1", "fix-1", "core"]
    assert "## Your last answer was refused" in asked[1] and "onboard:<category>/<name>" in asked[1]


def test_a_turn_with_no_decision_runs_again_by_itself_and_a_spent_task_can_only_be_escalated(
    tmp_path, task_file, monkeypatch
):
    asked: list[str] = []

    def planner(turn, prompt):
        asked.append(prompt)
        # Neither is possible for a task with no attempt left: the default stands.
        return {"decision": "CONTINUE", "summary": "again"}

    calls = _run(
        tmp_path,
        task_file,
        monkeypatch,
        Script(assembler=lambda turn, prompt: None, planner=planner),
        max_attempts=2,
    )
    # Each attempt is its turn and one short turn to write the missing file.
    assert [role for role, _ in calls] == ["assembler"] * 4 + ["planner"] * 2
    assert "It has no attempt left" in asked[0]
    assert "Answer with exactly one of: `ESCALATE`." in asked[0]
    assert "CONTINUE is not possible here" in asked[1]
    state = _state(tmp_path)
    assert state.done and state.verdict == "escalated"
    assert _verdicts(tmp_path) == []  # nothing was certified
    report = (tmp_path / "ws" / "delivery_report.md").read_text()
    assert report.startswith("# Delivery report: escalated") and "No certification ran" in report

    # The operator repaired what it named: the same command re-enters, the task starts over.
    calls = _run(tmp_path, task_file, monkeypatch, Script())
    assert [role for role, _ in calls] == ["assembler"]
    assert _state(tmp_path).verdict == "delivered"


def test_a_failed_kernel_task_can_be_skipped(tmp_path, task_file, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_vocabulary(workspace, [{"n": 1, "mode": "onboard", "target": "ops/norm/rms"}])

    def kernel_writer(turn, prompt):
        return {"decision": "FAILED_RECEIPT", "summary": "overflows above 2^31 elements"}

    def planner(turn, prompt):
        assert "`SKIP`" in prompt
        return {"decision": "SKIP", "summary": "the core composes the norm natively"}

    _run(tmp_path, task_file, monkeypatch, Script(kernel_writer=kernel_writer, planner=planner))
    assert _state(tmp_path).verdict == "delivered"
    assert [(t["id"], t["status"]) for t in _tasks(tmp_path)] == [
        ("survey", "done"),
        ("k1-rms", "skipped"),
        ("core", "done"),
    ]


def test_an_operator_message_is_put_to_the_planner_between_tasks(tmp_path, task_file, monkeypatch):
    (tmp_path / "ws").mkdir()
    inbox.post(tmp_path / "ws", "keep the router in its own module")
    asked: list[str] = []

    def planner(turn, prompt):
        asked.append(prompt)
        return {"decision": "CONTINUE", "summary": "the core's task already does"}

    _run(tmp_path, task_file, monkeypatch, Script(planner=planner))
    assert _state(tmp_path).verdict == "delivered"
    [prompt] = asked
    assert "The operator left a message" in prompt and "keep the router in its own module" in prompt
    assert inbox.take(tmp_path / "ws") == []


def test_work_an_operator_message_asks_for_runs_before_the_next_task(
    tmp_path, task_file, monkeypatch
):
    (tmp_path / "ws").mkdir()
    inbox.post(tmp_path / "ws", "onboard the gated norm first: the core will need it")
    order: list[str] = []

    def worker(turn, prompt):
        order.append(_task_id(prompt))
        return {"decision": "LANDED" if _task_id(prompt).startswith("kx") else "UNIT_READY"}

    def planner(turn, prompt):
        task = {"kind": "kernel", "ref": "onboard:norm/gated_norm", "goal": "the gated norm"}
        return {"decision": "REPAIR", "summary": "added the entry", "tasks": [task]}

    _run(
        tmp_path,
        task_file,
        monkeypatch,
        Script(planner=planner, kernel_writer=worker, assembler=worker),
    )
    assert _state(tmp_path).verdict == "delivered"
    assert order == ["kx-1", "core"]
    assert [(t["id"], t["kind"], t["unit"]) for t in _tasks(tmp_path)] == [
        ("survey", "survey", ""),
        ("kx-1", "kernel", "core"),
        ("core", "assemble", "core"),
    ]


# ------------------------------------------------------------------------ certification


class _Checkout:
    """A scripted checkout: its head and its uncommitted files, as git would say them."""

    def __init__(self, head="c0ffee1"):
        self.head, self.dirt = head, ""

    def __call__(self, repo, *args, stdin=None):
        if args[:2] == ("status", "--porcelain"):
            return self.dirt
        if args[:1] == ("rev-parse",) or args[:2] == ("log", "-1"):
            return self.head + "\n"
        return ""


def _failing_boot(until: dict):
    """A shell whose boot produces no text while ``until['fixed']`` is false."""
    inner = passing_shell()
    counts = {"boot": 0, "accuracy": 0}

    def run(command, env=None, timeout=0):
        if "trtllm-eval" in command:
            counts["accuracy"] += 1
        elif "engine_driver" in command:
            counts["boot"] += 1
            if not until["fixed"]:
                return CommandResult(returncode=1, stdout="RuntimeError: shape mismatch\n")
        return inner(command, env, timeout)

    return run, counts


def test_a_failed_certification_is_repaired_and_certified_again_whole_on_the_new_commit(
    tmp_path, task_file, monkeypatch
):
    checkout, state = _Checkout(), {"fixed": False}
    shell, counts = _failing_boot(state)
    asked: list[str] = []

    def assembler(turn, prompt):
        if _task_id(prompt) == "fix-1":
            state["fixed"], checkout.head = True, "f1x3d02"  # the repair lands a commit
            assert "## Tag `fix`" in prompt and "tp1:boot fails on a shape mismatch" in prompt
        return {"decision": "UNIT_READY", "summary": "done"}

    def planner(turn, prompt):
        asked.append(prompt)
        goal = "tp1:boot fails on a shape mismatch; see the round's files"
        return {
            "decision": "REPAIR",
            "summary": "one cause",
            "tasks": [{"kind": "fix", "goal": goal}],
        }

    _run(
        tmp_path,
        task_file,
        monkeypatch,
        Script(assembler=assembler, planner=planner),
        shell_runner=shell,
        git_runner=checkout,
    )
    assert _state(tmp_path).verdict == "delivered"
    [prompt] = asked
    assert "certification round 1 of 4 did not certify the final commit `c0ffee1`" in prompt
    assert "check tp1:boot FAILED" in prompt
    verdicts = _verdicts(tmp_path)
    first = [v for v in verdicts if v["round"] == 1]
    second = [v for v in verdicts if v["round"] == 2]
    assert [(v["gate"], v["passed"]) for v in first] == [("boot", False), ("accuracy[gsm8k]", True)]
    # The check that failed runs first, and the one that passed runs again: a verdict
    # about the earlier commit says nothing about this one.
    assert [(v["gate"], v["passed"]) for v in second] == [("boot", True), ("accuracy[gsm8k]", True)]
    assert {v["tree"].split("+")[0] for v in first} == {"c0ffee1"}
    assert {v["tree"].split("+")[0] for v in second} == {"f1x3d02"}
    assert counts["accuracy"] == 2
    certification = yaml.safe_load((tmp_path / "ws" / "certification.yaml").read_text())
    assert certification["certified"] and certification["commit"] == "f1x3d02"


def test_a_round_stops_when_a_repaired_check_fails_again_and_the_run_ends_uncertified(
    tmp_path, task_file, monkeypatch
):
    checkout = _Checkout()
    shell, counts = _failing_boot({"fixed": False})

    def assembler(turn, prompt):
        if _task_id(prompt) == "fix-1":
            checkout.head = "f1x3d02"
        return {"decision": "UNIT_READY", "summary": "done"}

    def planner(turn, prompt):
        return {
            "decision": "REPAIR",
            "summary": "s",
            "tasks": [{"kind": "fix", "goal": "fix boot"}],
        }

    _run(
        tmp_path,
        task_file,
        monkeypatch,
        Script(assembler=assembler, planner=planner),
        shell_runner=shell,
        git_runner=checkout,
        max_rounds=2,
    )
    state = _state(tmp_path)
    assert state.done and state.verdict == "certification-failed"
    # Round 2 ran the failed boot first and stopped there: the benchmark was not paid for again.
    assert [(v["round"], v["gate"]) for v in _verdicts(tmp_path)] == [
        (1, "boot"),
        (1, "accuracy[gsm8k]"),
        (2, "boot"),
    ]
    assert counts["accuracy"] == 1
    report = (tmp_path / "ws" / "delivery_report.md").read_text()
    assert report.startswith("# Delivery report: certification-failed")
    assert "check tp1:accuracy[gsm8k] has no verdict on this commit" in report
    certification = yaml.safe_load((tmp_path / "ws" / "certification.yaml").read_text())
    assert not certification["certified"]


@pytest.mark.parametrize("how", ["dirty", "moved", "rebuilt"])
def test_only_a_clean_checkout_that_stays_on_one_commit_is_certified(
    tmp_path, task_file, repo, monkeypatch, how
):
    checkout = _Checkout()
    inner = passing_shell()
    ran: list[str] = []

    def shell(command, env=None, timeout=0):
        ran.append(command)
        if how == "moved" and "engine_driver" in command:
            checkout.head = "0therc0mm1t"  # something moved the checkout under the check
        if how == "rebuilt" and "engine_driver" in command:
            # A native rebuild: no commit, but not the binary the round started on.
            (repo / "tensorrt_llm" / "libtensorrt_llm.so").write_bytes(b"rebuilt")
        return inner(command, env, timeout)

    if how == "dirty":
        checkout.dirt = " M tensorrt_llm/_torch/model.py\n"
    _run(
        tmp_path,
        task_file,
        monkeypatch,
        Script(),
        shell_runner=shell,
        git_runner=checkout,
        max_rounds=1,
    )
    assert _state(tmp_path).verdict == "certification-failed"
    certification = yaml.safe_load((tmp_path / "ws" / "certification.yaml").read_text())
    assert not certification["certified"]
    if how == "dirty":
        assert "uncommitted changes to tracked files" in certification["problems"][0]
        assert not any("engine_driver" in c or "trtllm-eval" in c for c in ran)
    else:
        assert certification["problems"] == [
            "its commit or its native libraries changed while the checks ran"
        ]
        assert not any("trtllm-eval" in c for c in ran)  # the round stopped at once


def test_a_check_that_cannot_be_run_is_a_failed_check_not_a_crash(tmp_path, task_file, monkeypatch):
    """A run that raised inside a check would raise again at that check on every restart."""
    from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli

    real = gate_cli.run_check

    def run_check(task_data, gate, deployment=None, **kwargs):
        if gate == "boot":
            raise FileNotFoundError("the boot prompts file is gone")
        return real(task_data, gate, deployment, **kwargs)

    monkeypatch.setattr(gate_cli, "run_check", run_check)
    _run(tmp_path, task_file, monkeypatch, Script(), max_rounds=1)
    assert _state(tmp_path).verdict == "certification-failed"
    boot, accuracy = _verdicts(tmp_path)
    assert not boot["passed"] and accuracy["passed"]
    assert boot["detail"] == (
        "the check could not be run: FileNotFoundError: the boot prompts file is gone"
    )
    report = (tmp_path / "ws" / "delivery_report.md").read_text()
    assert "check tp1:boot FAILED" in report and "could not be run" in report


def test_a_passing_verdict_counts_only_for_the_tree_it_measured(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm").mkdir(parents=True)
    contract = contract_mod.Contract(
        (contract_mod.Check("tp1", "boot"), contract_mod.Check("tp1", "serve"))
    )
    status = contract_mod.ContractStatus.load(tmp_path)

    class Result:
        name, deployment, detail, passed, infrastructure = "boot", "tp1", "ok", True, ""

    def git(repo, *args):
        return "tree-of-" + args[-1]

    before = contract_mod.tree_id(repo, "aaaa", git=git)
    status.record(contract.checks[0], Result(), tree=before, round=1)
    assert contract_mod.evaluate(contract, status.on(before, 1)).missing == (contract.checks[1],)
    assert contract_mod.evaluate(contract, status.on(before, 1)).passed == (contract.checks[0],)
    # Another tree, another round, or a rebuilt native library: the verdict is not about it.
    other = contract_mod.tree_id(repo, "bbbb", git=git)
    assert not contract_mod.evaluate(contract, status.on(other)).passed
    assert not contract_mod.evaluate(contract, status.on(before, 2)).passed
    library = repo / "tensorrt_llm" / "libtensorrt_llm.so"
    library.write_bytes(b"elf")
    rebuilt = contract_mod.tree_id(repo, "aaaa", git=git)
    assert rebuilt != before
    # No commit, or no git tree for it: nothing is established, so no verdict is reused.
    assert contract_mod.tree_id(repo, "", git=git) == ""
    assert contract_mod.tree_id(repo, "aaaa", git=lambda *args: "") == ""
    assert not contract_mod.evaluate(contract, status.on("", 1)).passed
    # A result the infrastructure ended is never a pass.
    Result.infrastructure = "the session was lost"
    assert not status.record(contract.checks[1], Result(), tree=before, round=1).passed
    assert contract_mod.ContractStatus.load(tmp_path).on(before, 1).keys() == {
        "tp1:boot",
        "tp1:serve",
    }


# -------------------------------------------------------------------------------- resume


def test_an_interrupted_run_goes_on_from_its_tasks_and_from_the_verdicts_of_its_round(
    tmp_path, task_file, repo, monkeypatch
):
    task = _mtp_task(tmp_path, task_file, repo, variants={})
    turns: list[str] = []
    stop = {"at": "mtp"}

    def assembler(turn, prompt):
        turns.append(_task_id(prompt))
        if _task_id(prompt) == stop["at"]:
            stop["at"] = ""
            raise KeyboardInterrupt
        return {"decision": "UNIT_READY", "summary": "done"}

    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, task, monkeypatch, Script(assembler=assembler))
    assert [(t["id"], t["status"]) for t in _tasks(tmp_path)][1:] == [
        ("core", "done"),
        ("mtp", "running"),
    ]

    # The second process stops in the middle of the certification, after its first check.
    inner = passing_shell()
    seen = {"eval": 0}

    def shell(command, env=None, timeout=0):
        if "trtllm-eval" in command:
            seen["eval"] += 1
            if seen["eval"] == 1:
                raise KeyboardInterrupt
        return inner(command, env, timeout)

    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, task, monkeypatch, Script(assembler=assembler), shell_runner=shell)
    assert turns == ["core", "mtp", "mtp"]  # core was not asked again; mtp kept its attempt
    assert [t["attempts"] for t in _tasks(tmp_path)][1:] == [1, 1]
    recorded = [(v["round"], v["deployment"], v["gate"]) for v in _verdicts(tmp_path)]
    assert recorded == [(1, "tp1", "boot")]

    # The third finishes the same round: the boot it already has is not run again.
    booted: list[str] = []

    def last(command, env=None, timeout=0):
        if "engine_driver" in command and "accept" not in command.lower():
            booted.append(command)
        return inner(command, env, timeout)

    calls = _run(tmp_path, task, monkeypatch, Script(), shell_runner=last)
    assert calls == [] and _state(tmp_path).verdict == "delivered"
    verdicts = _verdicts(tmp_path)
    assert {v["round"] for v in verdicts} == {1}
    assert [v["gate"] for v in verdicts].count("boot") == 2  # core's, then mtp's: once each
    assert sum(1 for v in verdicts if (v["deployment"], v["gate"]) == ("tp1", "boot")) == 1


def test_a_spec_edited_between_two_runs_adds_and_drops_tasks(
    tmp_path, task_file, repo, monkeypatch
):
    """A run re-entered after the operator changed the spec follows the spec as it is now."""
    plain = task_file.read_text()
    with_variant = yaml.safe_load(plain)
    with_variant["variants"] = {"switched": {"env": {"MY_SWITCH": "1"}, "gates": ["accuracy"]}}
    task_file.write_text(yaml.safe_dump(with_variant))

    def blocked_variant(turn, prompt):
        if _task_id(prompt) == "switched":
            return {"decision": "BLOCKED", "summary": "the switch is not wired"}
        return {"decision": "UNIT_READY", "summary": "done"}

    _run(tmp_path, task_file, monkeypatch, Script(assembler=blocked_variant))
    assert _state(tmp_path).verdict == "escalated"
    # The operator drops the variant and asks for a feature instead.
    _mtp_task(tmp_path, task_file, repo, variants={})
    calls = _run(tmp_path, task_file, monkeypatch, Script())
    assert [_task_id(p) for role, p in calls] == ["mtp"]
    assert _state(tmp_path).verdict == "delivered"
    assert [(t["id"], t["status"]) for t in _tasks(tmp_path)] == [
        ("survey", "done"),
        ("core", "done"),
        ("switched", "skipped"),
        ("mtp", "done"),
    ]


def test_a_decision_left_by_a_stopped_process_is_taken_not_asked_for_again(
    tmp_path, task_file, monkeypatch
):
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    def stops_after_answering(self, role, what, prompt, *, decision_file=None):
        decision_file.write_text(yaml.safe_dump({"decision": "UNIT_READY", "summary": "written"}))
        raise KeyboardInterrupt

    with monkeypatch.context() as patched:
        patched.setattr(TrtllmModelingV2Workflow, "_run_turn", stops_after_answering)
        with pytest.raises(KeyboardInterrupt):
            _run(tmp_path, task_file, monkeypatch, Script())
    calls = _run(tmp_path, task_file, monkeypatch, Script())
    assert calls == [] and _state(tmp_path).verdict == "delivered"
    assert _tasks(tmp_path)[1]["summary"] == "written"


def test_the_checkpoint_round_trips_and_an_older_one_is_refused(tmp_path):
    path = tmp_path / "state.json"
    original = WorkflowState(
        task_path="/x/task.yaml",
        stage=STAGE_EXECUTION,
        slug="qwen3-30b-a3b-sm_103-tp1",
        branch="trtllm-modeling-v2/qwen3-30b-a3b-sm_103-tp1",
        devices=[2, 3],
        round=2,
        round_open=True,
        failed_checks=["tp1:boot"],
    )
    save_state(path, original)
    assert load_state(path) == original
    path.write_text(json.dumps({"version": 2, "task_path": "t", "stage": "execution"}))
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_state(path)


# --------------------------------------------------------------------- preflight, dry run


def test_a_dry_run_stops_after_preflight_with_no_agent_turn_and_no_engine_command(
    tmp_path, task_file, monkeypatch
):
    ran: list[str] = []

    def shell(command, env=None, timeout=0):
        ran.append(command)
        return CommandResult(returncode=0, stdout="ok\n")

    devices: list[list[str]] = []

    def no_gpu(argv, env=None, timeout=0):
        devices.append(list(argv))
        return CommandResult(returncode=127, stderr="nvidia-smi: command not found")

    workflow = _workflow(tmp_path, shell_runner=shell, command_runner=no_gpu, dry_run=True)
    calls = _install(monkeypatch, workflow, Script())
    with workflow:
        assert workflow.run(task_file) is True
    # No device is queried or claimed, and no agent turn starts.
    assert calls == [] and devices == []
    assert not any("engine_driver" in c or "trtllm-eval" in c or "trtllm-serve" in c for c in ran)
    state = _state(tmp_path)
    assert not state.done and state.stage == "preflight"
    assert (tmp_path / "ws" / "contract.yaml").is_file()
    assert not (tmp_path / "ws" / "tasks.yaml").exists()


def test_a_failed_preflight_ends_the_run_before_any_agent(tmp_path, task_file, monkeypatch):
    def busy_device(argv, env=None, timeout=0):
        return CommandResult(returncode=0, stdout="0, 20000, 183359, 99, 10.3\n")

    calls = _run(tmp_path, task_file, monkeypatch, Script(), command_runner=busy_device)
    assert calls == []
    assert "preflight failed" in (tmp_path / "ws" / "journal.md").read_text()
    assert not _state(tmp_path).done


# ------------------------------------------------------- integration, on a real checkout

_ENV = {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}
_SIGNED = "\n\nSigned-off-by: T <t@example.com>"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkout(tmp_path: Path) -> Path:
    repo = tmp_path / "checkout"
    (repo / FAMILY).mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / FAMILY / "a.py").write_text("A = 1\n")
    (repo / "shared.py").write_text("S = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _integration(repo: Path, tmp_path: Path) -> Integration:
    policy = CommitPolicy.from_task(
        {"trtllm_repo_path": str(repo)},
        identity=CommitIdentity("T", "t@example.com"),
        own_paths=own_paths(TARGET_REL),
    )
    return Integration(repo, tmp_path / "ws", "mv2/run", _ENV, policy=policy)


def test_uncommitted_work_or_a_commit_that_breaks_the_rules_does_not_land(tmp_path):
    repo = _checkout(tmp_path)
    integration = _integration(repo, tmp_path)
    tree = integration.open("core")
    (tree / FAMILY / "a.py").write_text("A = 2\n")
    outcome = integration.integrate("core")
    assert not outcome.merged and outcome.repairable and "uncommitted changes" in outcome.detail
    _git(tree, "commit", "-q", "-am", "unsigned")
    outcome = integration.integrate("core")
    assert not outcome.merged and outcome.repairable
    assert "no Signed-off-by" in outcome.detail and "rebase --signoff" in outcome.detail
    assert (repo / FAMILY / "a.py").read_text() == "A = 1\n"
    _git(tree, "rebase", "-q", "--signoff", _git(repo, "rev-parse", "HEAD"))
    assert integration.integrate("core").merged
    assert (repo / FAMILY / "a.py").read_text() == "A = 2\n"
    assert _git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"


def test_a_commit_lands_labelled_by_its_paths_and_only_a_judge_path_is_refused(tmp_path):
    repo = _checkout(tmp_path)
    integration = _integration(repo, tmp_path)
    assert f"{FAMILY}/" in integration.policy.own_paths
    tree = integration.open("core")
    (tree / "shared.py").write_text("S = 2\n")
    _git(tree, "commit", "-q", "-am", f"change shared code{_SIGNED}")
    (tree / FAMILY / "a.py").write_text("A = 2\n")
    _git(tree, "commit", "-q", "-am", f"the target{_SIGNED}")
    (tree / FAMILY / "a.py").write_text("A = 3\n")
    (tree / "shared.py").write_text("S = 3\n")
    _git(tree, "commit", "-q", "-am", f"both{_SIGNED}")
    # Code other models run is not refused: it lands, and the report can list it.
    assert integration.integrate("core").merged
    assert (repo / "shared.py").read_text() == "S = 3\n"
    rows = load_labels(tmp_path / "ws")
    assert {(r["subject"], r["label"], tuple(r["outside"])) for r in rows} == {
        ("change shared code", "shared", ("shared.py",)),
        ("the target", "target", ()),
        ("both", "mixed", ("shared.py",)),
    }
    assert all(r["task"] == "core" and len(r["sha"]) == 40 for r in rows)
    # What scores the run is not the run's to change: one line naming the path, and nothing lands.
    tree = integration.open("fix")
    judge = tree / "tensorrt_llm/evaluate/lm_eval.py"
    judge.parent.mkdir(parents=True)
    judge.write_text("EASIER = True\n")
    _git(tree, "add", "tensorrt_llm/evaluate/lm_eval.py")
    _git(tree, "commit", "-q", "-m", f"an easier judge{_SIGNED}")
    outcome = integration.integrate("fix")
    assert not outcome.merged and outcome.repairable
    assert (
        "tensorrt_llm/evaluate/lm_eval.py is a judge path (tensorrt_llm/evaluate/)"
        in outcome.detail
    )
    assert not (repo / "tensorrt_llm/evaluate").exists() and len(load_labels(tmp_path / "ws")) == 3
    # A reference file the spec names is the judge's too; `allowed_paths` is read no more.
    spec = {"trtllm_repo_path": str(repo), "accuracy": {"reference_yaml": "refs/gsm8k.yaml"}}
    named = CommitPolicy.from_task(spec, identity=CommitIdentity("T", "t@example.com"))
    assert "refs/gsm8k.yaml is a judge path" in named.path_problem("refs/gsm8k.yaml")
    assert named.path_problem("tests/integration/defs/accuracy/references/gsm8k.yaml")
    assert not named.path_problem("tests/integration/defs/accuracy/test_llm_api_pytorch.py")


def test_a_task_whose_branch_conflicts_goes_back_to_its_worker_once_and_then_lands(
    tmp_path, task_file, repo, monkeypatch
):
    """The pipeline on a real checkout: audit, a rebase conflict, the worker's repair, integration."""
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "user.email", "t@example.com")
    (repo / FAMILY / "a.py").write_text("A = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", f"base{_SIGNED}")
    prompts: list[str] = []

    def assembler(turn, prompt):
        prompts.append(prompt)
        tree = Path(re.search(r"You work in `([^`]+)`", prompt).group(1))
        if turn == 1:
            (tree / FAMILY / "a.py").write_text("A = 3\n")
            _git(tree, "commit", "-q", "-am", f"core: three{_SIGNED}")
            # Another task lands on the checkout meanwhile, in the same file.
            (repo / FAMILY / "a.py").write_text("A = 2\n")
            _git(repo, "commit", "-q", "-am", f"other: two{_SIGNED}")
        else:
            head = re.search(r"git -C \S+ rebase ([0-9a-f]{40})", prompt).group(1)
            assert subprocess.run(["git", "-C", str(tree), "rebase", head]).returncode != 0
            (tree / FAMILY / "a.py").write_text("A = 2\nA3 = 3\n")
            _git(tree, "add", "-A")
            _git(tree, "-c", "core.editor=true", "rebase", "--continue")
        return {"decision": "UNIT_READY", "summary": "done"}

    workflow = _workflow(tmp_path, git_runner=None, max_rounds=1)
    _install(monkeypatch, workflow, Script(assembler=assembler))
    with workflow:
        workflow.run(task_file)
    assert len(prompts) == 2
    assert "## Your finished work did not integrate" not in prompts[0]
    assert "## Your finished work did not integrate" in prompts[1] and "a.py" in prompts[1]
    assert (repo / FAMILY / "a.py").read_text() == "A = 2\nA3 = 3\n"
    assert _git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"
    assert _tasks(tmp_path)[1]["status"] == "done" and _tasks(tmp_path)[1]["attempts"] == 1
    # The certification then measured the commit the integration left, and nothing else.
    head = _git(repo, "rev-parse", "HEAD")
    certification = yaml.safe_load((tmp_path / "ws" / "certification.yaml").read_text())
    assert certification["commit"] == head
    # A verdict names the files it measured: the commit's tree, not the commit.
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    assert {v["tree"].split("+")[0] for v in _verdicts(tmp_path)} == {tree}
    assert not (tmp_path / "ws" / "worktrees" / "core").exists()  # integrated, so closed


def test_a_verdict_is_identified_by_content(tmp_path):
    repo = tmp_path / "repo"
    family = repo / contract_mod.FAMILY_LIBRARY_DIR / "models" / "fam" / "native"
    family.mkdir(parents=True)
    (repo / "tensorrt_llm" / "libs").mkdir(parents=True, exist_ok=True)
    engine = repo / "tensorrt_llm" / "libs" / "libtensorrt_llm.so"
    engine.write_bytes(b"engine")
    kernels = family / "libfam_kernels.so.1"
    kernels.write_bytes(b"kernels")

    def git(repo, *args, stdin=None):
        # Two commits with the same files have one tree.
        return {"aaaa^{tree}": "7ree", "bbbb^{tree}": "7ree", "cccc^{tree}": "0ther"}[args[-1]]

    spec = {"checkpoint_path": "/ckpt", "parallel": "tp1", "notes": "n", "agents": {"a": 1}}
    spec["deployments"] = [{"name": "tp1", "options": {"max_seq_len": 5120}}]

    def identity(commit="aaaa", task=spec, environment="env1"):
        return contract_mod.tree_id(repo, commit, task, environment, git)

    base = identity()
    assert base.split("+")[0] == "7ree" and len(base.split("+")) == 5
    assert identity("bbbb") == base and identity("cccc") != base
    # The same build on another node: other times, the same bytes.
    os.utime(engine, ns=(1, 1))
    assert identity() == base
    # A rebuilt engine library, or a family's own, is another thing measured.
    kernels.write_bytes(b"rebuilt")
    assert identity() != base
    kernels.write_bytes(b"kernels")
    assert identity() == base
    # What the checks read of the spec, and where they run; not the notes or the agents.
    assert identity(task={**spec, "notes": "other", "agents": {}}) == base
    edited = {**spec, "deployments": [{"name": "tp1", "options": {"max_seq_len": 4096}}]}
    assert identity(task=edited) != base
    assert identity(environment="env2") != base
    assert contract_mod.flow_version() in base
    # A file a check reads through the spec counts by its content: edited in place, it is
    # a spec edit.
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("The capital of France is\n")
    filed = {**spec, "accuracy": {"boot": {"prompts_file": str(prompts)}}}
    read = identity(task=filed)
    prompts.write_text("The capital of Peru is\n")
    assert identity(task=filed) != read

    # The latest outcome counts, and every append reaches the disk.
    status = contract_mod.ContractStatus.load(tmp_path)
    check = contract_mod.Check("tp1", "boot")

    class Result:
        name, deployment, detail, passed, infrastructure = "boot", "tp1", "ok", True, ""

    status.record(check, Result(), tree=base)
    Result.passed = False
    status.record(check, Result(), tree=base)
    assert not contract_mod.ContractStatus.load(tmp_path).on(base)["tp1:boot"].passed
    # A line a dead process tore is closed before the next verdict, which stays its own.
    with status.path.open("a") as ledger:
        ledger.write('{"deployment": "tp1", "gate": "bo')
    Result.passed = True
    contract_mod.ContractStatus.load(tmp_path).record(check, Result(), tree=base)
    assert contract_mod.ContractStatus.load(tmp_path).on(base)["tp1:boot"].passed
