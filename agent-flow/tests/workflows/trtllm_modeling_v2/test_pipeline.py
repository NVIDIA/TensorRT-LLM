# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run: the fixed order, the onboard check, not done and go on, the bounds, resume, the report."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.git import CommitIdentity, CommitPolicy
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem, rows_by_unit
from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run import report as report_mod
from agent_flow.workflows.trtllm_modeling_v2.run.integration import (
    Integration,
    load_labels,
    own_paths,
)
from agent_flow.workflows.trtllm_modeling_v2.run.pipeline import NOTES_FILENAME
from agent_flow.workflows.trtllm_modeling_v2.run.state import WorkflowState, load_state, save_state
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

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
    healthy_smi,
    passing_shell,
)

FAMILY = str(Path(TARGET_REL).parent)
SLUG = "gpt-oss-120b-sm_103-tp1"
KV_FP8 = {"name": "kv_fp8", "options": {"kv_cache_config": {"dtype": "fp8"}}, "relative_to": "tp1"}
TEP2 = {"name": "tep2", "options": {"tensor_parallel_size": 2}}
BLOCKED = {"decision": "BLOCKED", "blocked_reason": "environment", "summary": "no route to it"}
READY = {"decision": "UNIT_READY", "summary": "assembled"}


def _spec(task_file, *later: dict, **more) -> Path:
    """The fixture spec stating its deployments: ``tp1`` first, then ``later``."""
    spec = yaml.safe_load(task_file.read_text())
    spec["deployments"] = [{"name": "tp1", "options": {"max_seq_len": 4096}}, *later]
    spec.update(more)
    task_file.write_text(yaml.safe_dump(spec))
    return task_file


def _checks(fail=(), events: list[str] | None = None, score: float = 90.0):
    """A shell whose checks pass but those in ``fail`` (``tp1:boot``, ``kv_fp8:accuracy``).

    ``fail`` is read at each check, so a test changes it while the run goes; every check
    is noted in ``ran`` and, as ``check:<key>``, in ``events``.
    """
    inner = passing_shell(score)
    ran: list[str] = []

    def run(command, env=None, timeout=0):
        found = re.search(r"/(boot|eval)\.([^./]+)\.", command)
        if found is None:
            return inner(command, env, timeout)
        key = f"{found.group(2)}:{'boot' if found.group(1) == 'boot' else 'accuracy'}"
        ran.append(key)
        if events is not None:
            events.append(f"check:{key}")
        if key in run.fail:
            return CommandResult(returncode=1, stdout="it fell over\n")
        return inner(command, env, timeout)

    run.ran, run.fail = ran, set(fail)  # type: ignore[attr-defined]
    return run


def _by_task(script: dict, events: list[str] | None = None):
    """An assembler scripted by task id: a decision, a list of one per attempt, or a callable."""
    turns: dict[str, int] = {}

    def assembler(turn, prompt):
        task = _task_id(prompt)
        turns[task] = turns.get(task, 0) + 1
        if events is not None:
            events.append(f"turn:{task}")
        said = script.get(task, READY)
        if callable(said):
            return said(turns[task], prompt)
        return said[min(turns[task], len(said)) - 1] if isinstance(said, list) else said

    return assembler


def _report(tmp_path) -> str:
    return (tmp_path / "ws" / "delivery_report.md").read_text()


def _prompts(calls, task: str) -> list[str]:
    return [prompt for _, prompt in calls if _task_id(prompt) == task]


def _verdicts(tmp_path) -> list[dict]:
    path = tmp_path / "ws" / contract_mod.STATUS_FILENAME
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ------------------------------------------------- the order, the onboard check, going on


def test_core_is_onboarded_by_code_then_one_deployment_is_not_done_and_the_next_is(
    tmp_path, task_file, monkeypatch
):
    """Survey, core's rows, core, the onboard check; then each deployment, whatever the last came to."""
    task = _spec(task_file, KV_FP8, TEP2)
    workspace = tmp_path / "ws"
    monkeypatch.setattr(TrtllmModelingV2Workflow, "survey_first", True)
    events: list[str] = []

    def surveyor(turn, prompt):
        events.append("turn:survey")
        rows = [
            {"n": 1, "mode": "onboard", "target": "ops/norm/rms"},
            {"n": 2, "mode": "onboard", "target": "ops/kv/quant", "unit": "kv_fp8"},
            {"n": 3, "mode": "onboard", "target": "ops/norm/scale", "depends_on": [1]},
        ]
        _write_vocabulary(workspace, rows)
        return {"decision": "DRAFT_READY", "summary": "three rows"}

    def kernel_writer(turn, prompt):
        events.append(f"turn:{_task_id(prompt)}")
        return {"decision": "LANDED", "summary": "landed", "entry": "x"}

    shell = _checks(events=events)
    script = Script(
        surveyor=surveyor,
        kernel_writer=kernel_writer,
        assembler=_by_task({"kv_fp8": BLOCKED}, events),
    )
    workflow = _workflow(tmp_path, command_runner=healthy_smi(2), shell_runner=shell)
    calls = _install(monkeypatch, workflow, script)
    with workflow:
        assert workflow.run(task) is True  # onboarded, whatever else is not done

    # The order is the code's. Core's checks run by code as soon as core says it is ready,
    # before any later deployment's turn; the others run once, after the last task, and a
    # deployment that is not done owes none.
    assert events == [
        "turn:survey",
        "turn:k1-rms",
        "turn:k3-scale",
        "turn:core",
        "check:tp1:boot",
        "check:tp1:accuracy",
        "turn:k2-quant",
        *["turn:kv_fp8"] * 3,
        "turn:tep2",
        "check:tep2:boot",
        "check:tep2:accuracy",
    ]
    stands = {t["id"]: (t["status"], t["attempts"]) for t in _tasks(tmp_path)}
    assert stands == {
        "survey": ("done", 1),
        "k1-rms": ("done", 1),
        "k3-scale": ("done", 1),
        "core": ("done", 1),
        "k2-quant": ("done", 1),
        "kv_fp8": ("not done", 3),
        "tep2": ("done", 1),
    }
    record = _state(tmp_path).onboarded
    assert record["commit"] == "c0ffee1" and record["accuracy"] == "judged"
    assert record["ref"] == f"refs/mv2/{SLUG}/onboarded" and record["libraries"] and record["time"]
    # A failed attempt runs again with its result line as its note.
    again = _prompts(calls, "kv_fp8")[1]
    assert "## Why this task runs again" in again and "Attempt 2 of 3" in again
    assert "it ended BLOCKED: blocked_reason: environment; no route to it" in again
    # What a later worker is told of the tasks before it, and of its own checks.
    last = _prompts(calls, "tep2")[0]
    assert "`kv_fp8` (deployment): not done BLOCKED" in last
    assert "`core` (core): done UNIT_READY" in last and "Nothing of a task not done landed" in last
    assert "## The checks `tep2` owes" in last and "`tep2:boot`, `tep2:accuracy[gsm8k]`" in last
    assert "gate <boot|accuracy|serve|tests>" in last and "--deployment tep2" in last
    assert "the run runs each of them once on the checkout" in last
    core = _prompts(calls, "core")[0]
    assert "the run itself boots and scores this deployment on your committed worktree" in core
    # The report: its first line, then every deployment where it stands.
    report = _report(tmp_path)
    first = report.splitlines()[0]
    assert first.startswith(f"# onboarded: commit `c0ffee1`, ref `refs/mv2/{SLUG}/onboarded`, 20")
    assert "| `tp1` | `accuracy[gsm8k]` | 90 | reference 85.6 (" in report
    assert "| PASS | in process | `c0ffee1` |" in report
    assert "| `tep2` | `boot` | - | - | PASS | in process | `c0ffee1` |" in report
    assert "| `kv_fp8` |" not in report.split("## Done")[1].split("##")[0]
    not_done = report.split("## Not done")[1].split("##")[0]
    assert "- `kv_fp8` (deployment): it ended BLOCKED: blocked_reason: environment" in not_done
    assert f"evidence: `{workspace}/decisions/kv_fp8-3.yaml`" in not_done
    assert "- native libraries: the head's are the onboarded ones" in report
    unchecked = report.split("## What the run did not check")[1].split("##")[0]
    assert "registered with the repository's CI" in unchecked and "(unknown)" in unchecked
    assert "- assembler: claude-code claude-test-model (default effort)" in report
    assert {role for role, _ in calls} == {"surveyor", "kernel_writer", "assembler"}


def test_a_kernel_row_runs_after_the_rows_it_depends_on_in_the_first_unit_that_needs_it():
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
    grouped = rows_by_unit(queue, ["core", "mtp"])
    assert [q.n for q in grouped["core"]] == [3, 1, 6, 5]
    assert [q.n for q in grouped["mtp"]] == [2, 4, 7, 8]


def test_a_failed_onboard_check_is_the_next_attempts_note_and_a_pass_on_record_is_not_run_again(
    tmp_path, task_file, monkeypatch
):
    task = _spec(task_file, KV_FP8)
    shell = _checks(fail={"tp1:accuracy"})

    def core(attempt, prompt):
        if attempt == 2:
            shell.fail.clear()  # what the second attempt changed makes the benchmark run
        return READY

    script = Script(assembler=_by_task({"core": core}))
    calls = _run(tmp_path, task, monkeypatch, script, shell_runner=shell)
    # The boot passed on this content and is on record: only what failed runs again.
    assert shell.ran[:3] == ["tp1:boot", "tp1:accuracy", "tp1:accuracy"]
    assert _tasks(tmp_path)[1]["attempts"] == 2 and _state(tmp_path).onboarded
    second = _prompts(calls, "core")[1]
    assert "The previous attempt did not finish:\n\n[FAIL] tp1:accuracy[gsm8k]: " in second
    assert "trtllm-eval exited 1" in second
    # Nothing was onboarded on the commit that failed, and no later task ran before it passed.
    assert [_task_id(p) for _, p in calls] == ["core", "core", "kv_fp8"]
    assert [v["passed"] for v in _verdicts(tmp_path)][:3] == [True, False, True]


@pytest.mark.parametrize("stated", ["no row", "no benchmark"])
def test_a_first_deployment_nothing_judges_is_onboarded_on_its_boot_and_says_so(
    tmp_path, task_file, repo, monkeypatch, stated
):
    first = {"name": "tp1", "options": {"max_seq_len": 4096}}
    if stated == "no benchmark":
        first["benchmarks"] = []
    else:
        refs = repo / "tests/integration/defs/accuracy/references/gsm8k.yaml"
        refs.write_text(yaml.safe_dump({"another/model": [{"accuracy": 85.6}]}))
    task = _spec(task_file, deployments=[first])
    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script())
    with workflow:
        assert workflow.run(task) is True
    assert _state(tmp_path).onboarded["accuracy"] == "unjudged (boot only)"
    report = _report(tmp_path)
    assert report.splitlines()[0].endswith("; accuracy unjudged (boot only)")
    assert report.startswith("# onboarded: commit `c0ffee1`")
    if stated == "no row":
        # The score is a line of the report, with no verdict: it failed nothing.
        assert "| `tp1` | `accuracy[gsm8k]` | 90 | none (no declared anchor" in report
        assert "| no verdict | in process |" in report
        assert _verdicts(tmp_path)[1]["unjudged"] and _verdicts(tmp_path)[1]["passed"]
    else:
        assert "tp1: it names no benchmark, so its boot alone judges it" in report


def test_core_spent_ends_the_run_not_onboarded_and_the_report_says_where(
    tmp_path, task_file, monkeypatch
):
    task = _spec(task_file, KV_FP8)
    shell = _checks(fail={"tp1:boot"})
    workflow = _workflow(tmp_path, shell_runner=shell)
    calls = _install(monkeypatch, workflow, Script())
    with workflow:
        assert workflow.run(task) is False
    # Core had its attempts; the run ends there and no later deployment gets a turn.
    assert [_task_id(p) for _, p in calls] == ["core"] * 3
    assert shell.ran == ["tp1:boot"] * 3
    stands = {t["id"]: t["status"] for t in _tasks(tmp_path)}
    assert stands == {"survey": "done", "core": "not done", "kv_fp8": "pending"}
    assert _state(tmp_path).onboarded == {}
    report = _report(tmp_path).splitlines()
    assert report[0] == "# not onboarded: stopped at `core`, not done after 3 attempt(s)"
    assert report[2].startswith("- result: [FAIL] tp1:boot: ")
    workspace = tmp_path / "ws"
    assert report[3] == (
        f"- reproduce: `trtllm-modeling-v2 gate boot --task {workspace}/task.yaml "
        f"--deployment tp1 --tree {workspace}/worktrees/core`"
    ) or report[3].startswith(f"- reproduce: `{workspace}/bin/trtllm-modeling-v2 gate boot ")
    assert report[4] == "- last commit where `tp1:boot` passed: `none`"
    assert "## Done\n\n(none)" in "\n".join(report)


def test_a_worker_that_never_gets_core_ready_stops_the_run_with_its_own_reason(
    tmp_path, task_file, monkeypatch
):
    shell = _checks()
    script = Script(assembler=_by_task({"core": BLOCKED}))
    workflow = _workflow(tmp_path, shell_runner=shell)
    _install(monkeypatch, workflow, script)
    with workflow:
        assert workflow.run(_spec(task_file)) is False
    report = _report(tmp_path)
    assert report.startswith("# not onboarded: stopped at `core`, not done after 3 attempt(s)")
    assert "- result: it ended BLOCKED: blocked_reason: environment; no route to it" in report
    assert "- no check of the run failed: see `" in report and shell.ran == []


def test_the_config_task_runs_first_and_lands_only_when_the_tree_loads_the_config(
    tmp_path, task_file, monkeypatch
):
    monkeypatch.setattr(TrtllmModelingV2Workflow, "survey_first", True)
    broken = {"config_error": "KeyError: 'model_type'"}
    record = json.loads(healthy_smi()("x").stdout.split(" ", 1)[1])

    def runner(command, env=None, timeout=0):
        said = {**record, **broken}
        return CommandResult(returncode=0, stdout=f"MV2FLOW_PREFLIGHT {json.dumps(said)}\n")

    def config(attempt, prompt):
        if attempt == 2:
            broken.clear()  # the second attempt's change makes the tree load it
        return READY

    def surveyor(turn, prompt):
        _write_vocabulary(tmp_path / "ws", [])
        return {"decision": "DRAFT_READY", "summary": "no rows"}

    script = Script(surveyor=surveyor, assembler=_by_task({"config": config}))
    calls = _run(tmp_path, _spec(task_file), monkeypatch, script, command_runner=runner)
    assert [_task_id(p) for _, p in calls] == ["config", "config", "survey", "core"]
    first, second = _prompts(calls, "config")
    assert (
        "make the tree load the checkpoint's config" in first and "KeyError: 'model_type'" in first
    )
    # Judged by preflight's own probe on what the worker committed; its line is the note.
    assert "does not load the config: KeyError: 'model_type' (on node-1)" in second
    assert [t["id"] for t in _tasks(tmp_path)][:2] == ["config", "survey"]
    assert _state(tmp_path).onboarded


def test_what_preflight_left_out_gets_no_task_and_a_base_with_no_score_gives_no_verdict(
    tmp_path, task_file, monkeypatch
):
    mid = {"name": "mid", "options": {"max_batch_size": 8}}
    kv = {**KV_FP8, "relative_to": "mid"}
    # Scored on a benchmark nothing anchors: preflight leaves that one check out.
    unanchored = {"name": "mmlu", "reference_yaml": "refs/none.yaml"}
    wide = {"name": "wide", "options": {}, "benchmarks": [{"name": "gsm8k"}, unanchored]}
    task = _spec(task_file, mid, kv, TEP2, wide)
    shell = _checks()
    script = Script(assembler=_by_task({"mid": BLOCKED}))
    # One free device: the two-rank deployment does not fit, which preflight records.
    calls = _run(tmp_path, task, monkeypatch, script, shell_runner=shell)
    assert "tep2" not in {t["id"] for t in _tasks(tmp_path)}
    assert "tep2" not in {_task_id(p) for _, p in calls}
    report = _report(tmp_path)
    left = report.split("## Not attempted")[1].split("##")[0]
    assert "- `tep2`: `tep2` needs 2 free devices on this host; it has 1" in left
    assert "- `wide` accuracy[mmlu]: " in left
    # `kv_fp8` is held to `mid`'s score, and `mid` is not done: its checks are not run for
    # it, and the score is reported with no verdict. What preflight left out is not run.
    assert shell.ran == [
        "tp1:boot",
        "tp1:accuracy",
        "kv_fp8:boot",
        "kv_fp8:accuracy",
        "wide:boot",
        "wide:accuracy",
    ]
    assert "| `wide` | `accuracy[gsm8k]` |" in report and "`accuracy[mmlu]` |" not in report
    assert "`wide:accuracy[mmlu]`" not in _prompts(calls, "wide")[0]
    assert "| `kv_fp8` | `accuracy[gsm8k]` | 90 | relative (the score of `mid`) | no verdict |" in (
        report
    )


def test_a_check_that_cannot_be_run_is_a_failed_line_not_a_crash(tmp_path, task_file, monkeypatch):
    real = gate_cli.run_check

    def run_check(task_data, gate, deployment=None, **kwargs):
        if deployment == "kv_fp8" and gate == "boot":
            raise FileNotFoundError("no such reference: /refs/x.yaml")
        return real(task_data, gate, deployment, **kwargs)

    monkeypatch.setattr(gate_cli, "run_check", run_check)
    calls = _run(tmp_path, _spec(task_file, KV_FP8), monkeypatch, Script())
    assert _state(tmp_path).onboarded
    said = "the check could not be run: FileNotFoundError: no such reference: /refs/x.yaml"
    assert said in _verdicts(tmp_path)[2]["detail"]
    # It goes to the one fix task with everything else that failed.
    assert said in _prompts(calls, "fix-1")[0]


# ----------------------------------------------------------------------- the final pass


class _Head:
    """The checkout's commit as a scripted git answers it; a test moves it when a task lands."""

    def __init__(self, at: str = "c0ffee1"):
        self.at = at

    def __call__(self, repo, *args, stdin=None):
        return self.at + "\n" if args[:1] == ("rev-parse",) else ""


def test_the_final_pass_runs_once_and_one_fix_task_names_every_failure_and_finding(
    tmp_path, task_file, repo, monkeypatch
):
    """Each landing is read for the conventions; findings hold nothing and go to the fix."""
    task = _spec(task_file, KV_FP8)
    modeling = repo / TARGET_REL / "modeling.py"
    head = _Head()
    shell = _checks(fail={"kv_fp8:boot"})
    v1 = "tensorrt_llm._torch.models.modeling_qwen3"

    def core(attempt, prompt):
        modeling.write_text(f"from {v1} import Qwen3\n" + modeling.read_text())
        return READY

    def kv(attempt, prompt):
        bare = repo / TARGET_REL / "ops" / "kv" / "quant.py"
        bare.parent.mkdir(parents=True)
        bare.write_text("def quant(x):\n    return x\n")
        return READY

    def fix(attempt, prompt):
        modeling.write_text(modeling.read_text().replace(f"from {v1} import Qwen3\n", ""))
        (repo / TARGET_REL / "ops" / "kv" / "extra.py").write_text("def extra(x):\n    return x\n")
        shell.fail.clear()
        head.at = "f1xed00"
        return READY

    script = Script(assembler=_by_task({"core": core, "kv_fp8": kv, "fix-1": fix}))
    workflow = _workflow(tmp_path, shell_runner=shell, git_runner=head)
    calls = _install(monkeypatch, workflow, script)
    with workflow:
        assert workflow.run(task) is True
    # A finding is kept with the task whose landing first showed it, and held neither.
    found = json.loads((tmp_path / "ws" / "conventions.json").read_text())
    assert set(found.values()) == {"core", "kv_fp8", "fix-1"} and all(
        t["status"] == "done" and t["attempts"] <= 1 for t in _tasks(tmp_path)
    )
    assert [_task_id(p) for _, p in calls] == ["core", "kv_fp8", "fix-1"]
    goal = _prompts(calls, "fix-1")[0]
    assert "repair what the final pass found, every item:" in goal
    assert "- check `kv_fp8:boot` failed at c0ffee1: [FAIL] kv_fp8:boot: " in goal
    assert f"- convention: {TARGET_REL}/modeling.py:1 imports Modeling V1 ({v1})" in goal
    assert "(first at `core`)" in goal and "(first at `kv_fp8`)" in goal
    assert "## Tag `fix`" in goal and "A check runs again with `" in goal
    # After the fix: what failed and core's own run once more, on the new commit. What
    # passed before keeps the commit it was measured on.
    assert shell.ran == [
        "tp1:boot",
        "tp1:accuracy",
        "kv_fp8:boot",
        "kv_fp8:accuracy",
        "kv_fp8:boot",
        "tp1:boot",
        "tp1:accuracy",
    ]
    report = _report(tmp_path)
    assert "| `kv_fp8` | `boot` | - | - | PASS | in process | `f1xed00` |" in report
    assert "| PASS | in process | `c0ffee1` |" in report.split("| `kv_fp8` | `accuracy")[1]
    # Nothing of the final pass moves the onboard record.
    assert report.startswith("# onboarded: commit `c0ffee1`")
    left = report.split("## Convention findings left")[1].split("##")[0]
    assert "quant.py" in left and "(first at `kv_fp8`)" in left and "Modeling V1" not in left
    # The next start has its own entry: what is still found gets one more fix task, once.
    calls = _run(tmp_path, task, monkeypatch, Script(), shell_runner=shell, git_runner=head)
    assert [_task_id(p) for _, p in calls] == ["fix-2"]
    assert "Modeling V1" not in calls[0][1] and "quant.py" in calls[0][1]
    assert "extra.py" in calls[0][1] and "(first at `fix-1`)" in calls[0][1]


def test_the_fix_task_belongs_to_its_final_pass(tmp_path, task_file, monkeypatch):
    task = _spec(task_file, KV_FP8)
    shell = _checks(fail={"kv_fp8:boot"})

    def fix(attempt, prompt):
        if attempt == 1:
            raise KeyboardInterrupt
        return BLOCKED

    script = Script(assembler=_by_task({"fix-1": fix, "fix-2": BLOCKED}))
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, task, monkeypatch, script, shell_runner=shell)
    # The cut start was no new entry: the same task goes on, and has all its attempts in
    # this one pass.
    calls = _run(tmp_path, task, monkeypatch, script, shell_runner=shell)
    assert [_task_id(p) for _, p in calls] == ["fix-1"] * 3
    stands = _tasks(tmp_path)[-1]
    assert (stands["id"], stands["status"], stands["attempts"]) == ("fix-1", "not done", 3)
    assert "| `kv_fp8` | `boot` | - | - | FAIL | in process |" in _report(tmp_path)
    # The next start's final pass names what fails then, in a task of its own: the old one
    # is not reopened.
    calls = _run(tmp_path, task, monkeypatch, script, shell_runner=shell)
    assert [_task_id(p) for _, p in calls] == ["fix-2"] * 3
    stands = {t["id"]: (t["status"], t["grants"]) for t in _tasks(tmp_path)}
    assert stands["fix-1"] == ("not done", 0) and stands["fix-2"] == ("not done", 0)


def test_the_survey_is_held_to_its_schema_and_spent_it_ends_the_run(
    tmp_path, task_file, monkeypatch
):
    monkeypatch.setattr(TrtllmModelingV2Workflow, "survey_first", True)
    workspace = tmp_path / "ws"
    good = {"valid": False}

    def surveyor(turn, prompt):
        if good["valid"] and turn == 2:
            _write_vocabulary(workspace, [])
        else:
            (workspace / "vocabulary.yaml").write_text("queue: [{n: 1}]\n")
        return {"decision": "DRAFT_READY", "summary": "drafted"}

    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(surveyor=surveyor))
    with workflow:
        assert workflow.run(_spec(task_file)) is False
    assert [role for role, _ in calls] == ["surveyor"] * 3
    assert "vocabulary.yaml does not validate: " in calls[1][1]
    report = _report(tmp_path)
    assert report.startswith("# not onboarded: stopped at `survey`, not done after 3 attempt(s)")
    assert "- result: vocabulary.yaml does not validate: " in report
    # A run that stopped before core has no final pass.
    assert "## Final pass\n\n(none)" in report
    # The next start gives it fresh attempts; a vocabulary that validates lets the run go on.
    good["valid"] = True
    (workspace / "vocabulary.yaml").unlink()
    calls = _run(tmp_path, task_file, monkeypatch, Script(surveyor=surveyor))
    assert [role for role, _ in calls] == ["surveyor", "surveyor", "assembler"]
    assert _state(tmp_path).onboarded


# ------------------------------------------------------------------ the bounds, resume


def test_a_cut_turn_restarts_free_twice_and_the_third_cut_spends_the_attempt(
    tmp_path, task_file, monkeypatch
):
    task = _spec(task_file)
    cut = {"left": 3}

    def core(attempt, prompt):
        if cut["left"]:
            cut["left"] -= 1
            raise KeyboardInterrupt
        return READY

    def start() -> list:
        return _run(tmp_path, task, monkeypatch, Script(assembler=_by_task({"core": core})))

    for cuts in (0, 1, 2):
        with pytest.raises(KeyboardInterrupt):
            start()
        stands = _tasks(tmp_path)[1]
        # The same attempt each time: its decision file's path does not move.
        assert (stands["status"], stands["attempts"], stands["cuts"]) == ("running", 1, cuts)
    calls = start()
    # Cut a third time, the attempt is spent with why; the next one is a new attempt.
    assert len(calls) == 1 and "decisions/core-2.yaml" in calls[0][1]
    assert "its turn was cut 3 times before it ended" in calls[0][1]
    stands = _tasks(tmp_path)[1]
    assert (stands["status"], stands["attempts"], stands["cuts"]) == ("done", 2, 0)
    assert "run interrupted" in (tmp_path / "ws" / "journal.md").read_text()


def test_a_decision_left_by_a_stopped_process_is_taken_not_asked_for_again(
    tmp_path, task_file, monkeypatch
):
    def stops_after_answering(self, role, what, prompt, *, decision_file=None):
        decision_file.write_text(yaml.safe_dump({"decision": "UNIT_READY", "summary": "written"}))
        raise KeyboardInterrupt

    with monkeypatch.context() as patched:
        patched.setattr(TrtllmModelingV2Workflow, "_run_turn", stops_after_answering)
        with pytest.raises(KeyboardInterrupt):
            _run(tmp_path, task_file, monkeypatch, Script())
    # Stopped again and again in the check that follows the answer: the turn was not cut,
    # so no restart is counted against it however often that happens.
    stopping = _checks()

    def stops(command, env=None, timeout=0):
        if "/boot." in command:
            raise KeyboardInterrupt
        return stopping(command, env, timeout)

    for _ in range(4):
        with pytest.raises(KeyboardInterrupt):
            _run(tmp_path, task_file, monkeypatch, Script(), shell_runner=stops)
        stands = _tasks(tmp_path)[1]
        assert (stands["status"], stands["attempts"], stands["cuts"]) == ("running", 1, 0)
    shell = _checks()
    calls = _run(tmp_path, task_file, monkeypatch, Script(), shell_runner=shell)
    # No turn is asked again; the answer is held to the onboard check as any other.
    assert calls == [] and shell.ran == ["tp1:boot", "tp1:accuracy"]
    assert _tasks(tmp_path)[1]["summary"] == "written" and _state(tmp_path).onboarded


def test_a_stopped_run_goes_on_from_its_tasks_and_from_the_verdicts_on_record(
    tmp_path, task_file, monkeypatch
):
    task = _spec(task_file, KV_FP8, TEP2)
    shell = _checks()

    def stops(attempt, prompt):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        script = Script(assembler=_by_task({"tep2": stops}))
        _run(tmp_path, task, monkeypatch, script, shell_runner=shell, command_runner=healthy_smi(2))
    assert shell.ran == ["tp1:boot", "tp1:accuracy"]
    onboarded = _state(tmp_path).onboarded
    calls = _run(
        tmp_path, task, monkeypatch, Script(), shell_runner=shell, command_runner=healthy_smi(2)
    )
    # Done tasks are not asked again, core is not checked again, the record is as it was.
    assert [_task_id(p) for _, p in calls] == ["tep2"]
    assert shell.ran[2:] == ["kv_fp8:boot", "kv_fp8:accuracy", "tep2:boot", "tep2:accuracy"]
    assert _state(tmp_path).onboarded == onboarded
    # A start on a finished run has nothing left to do: every pass is on record.
    calls = _run(
        tmp_path, task, monkeypatch, Script(), shell_runner=shell, command_runner=healthy_smi(2)
    )
    assert calls == [] and len(shell.ran) == 6
    assert _report(tmp_path).startswith("# onboarded: commit `c0ffee1`")


def test_a_new_start_reopens_what_was_not_done_within_its_grants(tmp_path, task_file, monkeypatch):
    task = _spec(task_file, KV_FP8)
    script = Script(assembler=_by_task({"kv_fp8": BLOCKED}))

    def turns() -> int:
        return len(_prompts(_run(tmp_path, task, monkeypatch, script), "kv_fp8"))

    # Its own three attempts, then fresh ones on each of two more starts, and no more.
    assert [turns() for _ in range(4)] == [3, 3, 3, 0]
    stands = {t["id"]: t for t in _tasks(tmp_path)}
    assert (stands["kv_fp8"]["status"], stands["kv_fp8"]["grants"]) == ("not done", 2)
    assert "`kv_fp8` | deployment | kv_fp8 | not done | 3 | 2 |" in _report(tmp_path)
    assert "kv_fp8: reopened, grant 2" in (tmp_path / "ws" / "journal.md").read_text()


def test_core_is_reopened_four_times_and_the_run_stays_stopped_after(
    tmp_path, task_file, monkeypatch
):
    task = _spec(task_file, KV_FP8)
    script = Script(assembler=_by_task({"core": BLOCKED}))
    seen = [len(_run(tmp_path, task, monkeypatch, script, max_attempts=1)) for _ in range(6)]
    assert seen == [1, 1, 1, 1, 1, 0]
    assert _tasks(tmp_path)[1]["grants"] == 4 and _tasks(tmp_path)[2]["status"] == "pending"
    assert _report(tmp_path).startswith("# not onboarded: stopped at `core`")


def test_a_spec_edited_between_two_starts_adds_and_drops_tasks(tmp_path, task_file, monkeypatch):
    def stops(attempt, prompt):
        raise KeyboardInterrupt

    mid = {"name": "mid", "options": {"max_batch_size": 8}}
    script = Script(assembler=_by_task({"kv_fp8": BLOCKED, "mid": stops}))
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, _spec(task_file, KV_FP8, mid), monkeypatch, script)
    assert [t["status"] for t in _tasks(tmp_path)[2:]] == ["not done", "running"]
    # The spec is read again on every start: two deployments went, one not done and one
    # whose turn was cut, and another came.
    calls = _run(
        tmp_path, _spec(task_file, TEP2), monkeypatch, script, command_runner=healthy_smi(2)
    )
    assert [_task_id(p) for _, p in calls] == ["tep2"]
    stands = {t["id"]: (t["status"], t["note"]) for t in _tasks(tmp_path)}
    assert stands["kv_fp8"] == ("skipped", "the spec or preflight no longer asks for it")
    assert stands["mid"] == ("skipped", "the spec or preflight no longer asks for it")
    assert stands["tep2"][0] == "done"
    # Skipped as it stood: the cut turn is not started again for a deployment that is gone.
    assert "its turn raised" not in (tmp_path / "ws" / "journal.md").read_text()
    assert "- name: tep2" in (tmp_path / "ws" / "task.yaml").read_text()


def test_the_run_state_round_trips_and_another_versions_is_refused(tmp_path):
    path = tmp_path / "state.json"
    record = dict(commit="abc", ref="refs/mv2/s/onboarded", libraries="e3b0", time="t")
    original = WorkflowState(
        task_path="/x/task.yaml",
        slug="qwen3-30b-a3b-sm_103-tp1",
        branch="trtllm-modeling-v2/qwen3-30b-a3b-sm_103-tp1",
        devices=[2, 3],
        onboarded={**record, "accuracy": "judged"},
    )
    save_state(path, original)
    assert load_state(path) == original
    path.write_text(json.dumps({"version": 3, "task_path": "t", "stage": "execution"}))
    with pytest.raises(ValueError, match="version 3, expected 4.*start over with --clean"):
        load_state(path)


# --------------------------------------------------------------------- the operator's notes


def test_every_dispatch_says_to_read_the_operators_notes_and_feedback_appends_to_them(
    tmp_path, task_file, capsys, monkeypatch
):
    workspace = tmp_path / "ws"
    monkeypatch.setattr(TrtllmModelingV2Workflow, "survey_first", True)
    feedback = cli.SUBCOMMANDS["feedback"]
    assert feedback(["--workspace", str(workspace), "kv_fp8: the fp8 path needs the new op"]) == 0
    (tmp_path / "more.md").write_text("tep2: use two devices\n")
    assert feedback(["--workspace", str(workspace), "--file", str(tmp_path / "more.md")]) == 0
    notes = (workspace / NOTES_FILENAME).read_text()
    # Appended, each under its own time; nothing reads it but the workers.
    stamps = re.findall(r"^## (\d{4}-\d\d-\d\dT[\d:]+\+00:00)$", notes, re.M)
    assert len(stamps) == 2 and notes.index("the fp8 path") < notes.index("use two devices")
    assert f"appended to {workspace / NOTES_FILENAME}" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        feedback(["--workspace", str(workspace), "  "])
    assert not (workspace / "inbox").exists()

    def surveyor(turn, prompt):
        _write_vocabulary(workspace, [{"n": 1, "mode": "onboard", "target": "ops/norm/rms"}])
        return {"decision": "DRAFT_READY", "summary": "one row"}

    def kv(attempt, prompt):
        return None if attempt == 1 else READY  # no decision file: one short turn to write it

    script = Script(surveyor=surveyor, assembler=_by_task({"kv_fp8": kv}))
    calls = _run(tmp_path, _spec(task_file, KV_FP8), monkeypatch, script)
    assert [role for role, _ in calls] == [
        "surveyor",
        "kernel_writer",
        "assembler",
        "assembler",
        "assembler",
    ]
    for _, prompt in calls:
        said = prompt.split("## The operator's notes")[1].split("##")[0]
        assert f"Read `{workspace / NOTES_FILENAME}` first" in said
        assert "read it again before you write your decision" in said
        assert "never a reason to change an anchor or a check" in said
    assert "## Your decision file is not valid" in calls[-1][1]
    assert _tasks(tmp_path)[-1]["attempts"] == 1  # the repair turn is no new attempt
    # Each turn's prompt and answer are kept with its activity, one set of files a turn.
    kept = {path.read_text() for path in (workspace / "turns").glob("*.prompt.md")}
    assert kept == {prompt for _, prompt in calls}
    assert len(list((workspace / "turns").glob("*.result.md"))) == len(calls)
    journal = (workspace / "journal.md").read_text()
    assert journal.count("**dispatch: assembler**: kv_fp8, to write ") == 2


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
    assert _state(tmp_path).onboarded == {}
    assert "checks owed: 2" in (tmp_path / "ws" / "journal.md").read_text()
    assert not (tmp_path / "ws" / "tasks.yaml").exists()
    assert not (tmp_path / "ws" / "delivery_report.md").exists()


def test_a_clean_start_removes_the_runs_records_and_keeps_what_is_the_workspaces(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    records = ("preflight.json", "conventions.json", "commit_labels.json", "journal.md")
    # Frozen once per workspace; what is still owned; what the operator wrote.
    kept = ("anchors.yaml", "servers.json", NOTES_FILENAME)
    for name in (*records, *kept, "tasks.yaml", "delivery_report.md"):
        (workspace / name).write_text("{}")
    _workflow(tmp_path, clean=True)
    assert not [name for name in records if (workspace / name).exists()]
    assert (
        not (workspace / "tasks.yaml").exists() and not (workspace / "delivery_report.md").exists()
    )
    assert all((workspace / name).exists() for name in kept)


def test_a_failed_preflight_ends_the_run_before_any_agent(tmp_path, task_file, monkeypatch):
    busy = healthy_smi(free=20000)
    workflow = _workflow(tmp_path, command_runner=busy)
    calls = _install(monkeypatch, workflow, Script())
    with workflow:
        assert workflow.run(task_file) is False
    assert calls == []
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "preflight failed" in journal
    assert "`tp1` needs 1 free sm_103 device(s) on this host; 0 of 1 are free" in journal
    assert not (tmp_path / "ws" / "delivery_report.md").exists()


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
    landed = integration.integrate("core")
    assert landed.merged and landed.commit == _git(repo, "rev-parse", "HEAD")
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
    spec = {
        "trtllm_repo_path": str(repo),
        "accuracy": {"reference_yaml": {"default": "refs/gsm8k.yaml", "mtp": "refs/mtp.yaml"}},
        "deployments": [
            {"name": "tp1", "benchmarks": [{"name": "x", "reference_yaml": "r/x.yaml"}]}
        ],
    }
    named = CommitPolicy.from_task(spec, identity=CommitIdentity("T", "t@example.com"))
    assert "refs/gsm8k.yaml is a judge path" in named.path_problem("refs/gsm8k.yaml")
    assert named.path_problem("refs/mtp.yaml") and named.path_problem("r/x.yaml")
    assert named.path_problem("tests/integration/defs/accuracy/references/gsm8k.yaml")
    assert not named.path_problem("tests/integration/defs/accuracy/test_llm_api_pytorch.py")


def test_on_a_real_checkout_core_lands_only_the_commit_the_onboard_check_passed(
    tmp_path, task_file, repo, monkeypatch
):
    """Worktrees, the audit, a conflict's repair turn, the onboard check, the pinned ref, labels."""
    task = _spec(task_file, KV_FP8, TEP2)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "user.email", "t@example.com")
    (repo / FAMILY / "a.py").write_text("A = 1\n")
    (repo / "shared.py").write_text("S = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", f"base{_SIGNED}")
    base = _git(repo, "rev-parse", "HEAD")
    checked: list[tuple[str, str]] = []
    shell = _checks()

    def judged(command, env=None, timeout=0):
        tree = re.match(r"cd (\S+) &&", command)
        if tree and re.search(r"/(boot|eval)\.", command):
            checked.append((tree.group(1), _git(Path(tree.group(1)), "rev-parse", "HEAD")))
        return shell(command, env, timeout)

    def tree_of(prompt: str) -> Path:
        return Path(re.search(r"You work in `([^`]+)`", prompt).group(1))

    def core(turn, prompt):
        tree = tree_of(prompt)
        if turn == 1:
            (tree / FAMILY / "a.py").write_text("A = 3\n")
            _git(tree, "commit", "-q", "-am", f"core: three{_SIGNED}")
            # This run rebuilt a native library in its worktree; git does not see it.
            (tree / "tensorrt_llm" / "libs").mkdir(parents=True)
            (tree / "tensorrt_llm" / "libs" / "libtensorrt_llm.so").write_bytes(b"rebuilt")
            # Another commit lands on the checkout meanwhile, in the same file.
            (repo / FAMILY / "a.py").write_text("A = 2\n")
            _git(repo, "commit", "-q", "-am", f"other: two{_SIGNED}")
        else:
            head = re.search(r"git -C \S+ rebase ([0-9a-f]{40})", prompt).group(1)
            assert subprocess.run(["git", "-C", str(tree), "rebase", head]).returncode != 0
            (tree / FAMILY / "a.py").write_text("A = 2\nA3 = 3\n")
            _git(tree, "add", f"{FAMILY}/a.py")
            _git(tree, "-c", "core.editor=true", "rebase", "--continue")
        return READY

    def kv(turn, prompt):
        tree = tree_of(prompt)
        (tree / "shared.py").write_text("S = 2\n")
        _git(tree, "commit", "-q", "-am", f"kv: shared code{_SIGNED}")
        return READY

    def tep2(turn, prompt):
        # What scores the run is not the run's to change; asked to put it right, it does not.
        tree = tree_of(prompt)
        if turn > 1:
            return READY
        judge = tree / "tensorrt_llm/evaluate/lm_eval.py"
        judge.parent.mkdir(parents=True, exist_ok=True)
        judge.write_text("EASIER = True\n")
        _git(tree, "add", "tensorrt_llm/evaluate/lm_eval.py")
        _git(tree, "commit", "-q", "-m", f"an easier judge{_SIGNED}")
        return READY

    script = Script(assembler=_by_task({"core": core, "kv_fp8": kv, "tep2": tep2}))
    workflow = _workflow(
        tmp_path,
        git_runner=None,
        shell_runner=judged,
        command_runner=healthy_smi(2),
        max_attempts=1,
    )
    calls = _install(monkeypatch, workflow, script)
    with workflow:
        assert workflow.run(task) is True
    workspace = tmp_path / "ws"
    first, repair = _prompts(calls, "core")
    assert "## Your finished work did not integrate" not in first
    assert "## Your finished work did not integrate" in repair and "a.py" in repair
    assert (repo / FAMILY / "a.py").read_text() == "A = 2\nA3 = 3\n"
    assert _git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"
    # The onboard check ran in core's worktree on the rebased commit, and that commit is
    # what landed, what the record names and what the ref points at.
    record = _state(tmp_path).onboarded
    onboarded = record["commit"]
    assert checked[:2] == [(str(workspace / "worktrees" / "core"), onboarded)] * 2
    assert _git(repo, "rev-parse", f"refs/mv2/{SLUG}/onboarded") == onboarded != base
    assert _git(repo, "merge-base", "--is-ancestor", onboarded, "HEAD") == ""
    assert _tasks(tmp_path)[1]["commit"] == onboarded and _tasks(tmp_path)[1]["attempts"] == 1
    assert not (workspace / "worktrees" / "core").exists()  # landed, so closed
    # A verdict names the files it measured: the commit's tree, not the commit.
    tree = _git(repo, "rev-parse", f"{onboarded}^{{tree}}")
    assert {v["tree"].split("+")[0] for v in _verdicts(tmp_path)[:2]} == {tree}
    assert {v["commit"] for v in _verdicts(tmp_path)[:2]} == {onboarded}
    # The libraries it accepted are not the checkout's: kept beside the record, and said.
    kept = workspace / "onboarded_libraries" / "tensorrt_llm" / "libs" / "libtensorrt_llm.so"
    assert kept.read_bytes() == b"rebuilt"
    later = _prompts(calls, "kv_fp8")[0]
    assert "- libraries were changed by `core`: rebuild before you boot" in later
    assert f"kept in `{workspace / 'onboarded_libraries'}`" in later
    report = _report(tmp_path)
    assert "are not the onboarded ones" in report and record["libraries"] in report
    # Code other models run lands, labelled; a commit on a judge path is refused whole, the
    # task is not done, and the run goes on to its end.
    assert (repo / "shared.py").read_text() == "S = 2\n"
    labels = {(r["task"], r["label"]) for r in load_labels(workspace)}
    assert labels == {("core", "target"), ("kv_fp8", "shared")}
    outside = report.split("## Changed outside the target")[1].split("##")[0]
    assert "shared (`kv_fp8`): kv: shared code: shared.py" in outside
    stands = {t["id"]: t for t in _tasks(tmp_path)}
    assert stands["tep2"]["status"] == "not done" and not stands["tep2"]["commit"]
    assert "tensorrt_llm/evaluate/lm_eval.py is a judge path" in stands["tep2"]["note"]
    assert not (repo / "tensorrt_llm/evaluate").exists()
    assert "an easier judge" not in _git(repo, "log", "--format=%s")
    assert "## Your finished work could not be integrated" in _prompts(calls, "tep2")[1]
    assert report.startswith(f"# onboarded: commit `{onboarded}`")
    final = report.split("## Final pass")[1].split("##")[0]
    assert "[ok] git:target-committed at `" in final
    # Nothing later takes the record back, or writes it again.
    report_mod.record_onboarded(workflow, repo, base)
    assert _state(tmp_path).onboarded == record
    assert _git(repo, "rev-parse", f"refs/mv2/{SLUG}/onboarded") == onboarded


def test_a_worker_that_commits_nothing_is_still_held_to_the_onboard_check(
    tmp_path, task_file, repo, monkeypatch
):
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", f"base{_SIGNED}")
    base = _git(repo, "rev-parse", "HEAD")
    shell = _checks(fail={"tp1:boot"})

    def core(attempt, prompt):
        if attempt == 2:
            shell.fail.clear()
        return READY

    script = Script(assembler=_by_task({"core": core}))
    workflow = _workflow(tmp_path, git_runner=None, shell_runner=shell)
    _install(monkeypatch, workflow, script)
    with workflow:
        assert workflow.run(_spec(task_file)) is True
    # The target was there already: what the check judges is the head as it is. Its
    # failure is the attempt's result line, word for word, with no turn to "put it right".
    assert shell.ran == ["tp1:boot", "tp1:boot", "tp1:accuracy"]
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "core (attempt 1 of 3): pending UNIT_READY: [FAIL] tp1:boot: " in journal
    assert _state(tmp_path).onboarded["commit"] == base == _git(repo, "rev-parse", "HEAD")
    assert _tasks(tmp_path)[1]["attempts"] == 2


def test_a_pass_counts_only_for_the_content_it_measured(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm").mkdir(parents=True)
    boot, serve = contract_mod.Check("tp1", "boot"), contract_mod.Check("tp1", "serve")
    status = contract_mod.ContractStatus.load(tmp_path)

    class Result:
        name, deployment, detail, passed, infrastructure = "boot", "tp1", "ok", True, ""

    def git(repo, *args):
        return "tree-of-" + args[-1]

    before = contract_mod.tree_id(repo, "aaaa", git=git)
    status.record(boot, Result(), tree=before, commit="aaaa")
    assert status.on(before).keys() == {"tp1:boot"} and status.on(before)["tp1:boot"].passed
    # Another tree or a rebuilt native library: the verdict is not about it.
    assert not status.on(contract_mod.tree_id(repo, "bbbb", git=git))
    (repo / "tensorrt_llm" / "libtensorrt_llm.so").write_bytes(b"elf")
    assert contract_mod.tree_id(repo, "aaaa", git=git) != before
    # No commit, or no git tree for it: nothing is established, so no verdict is reused.
    assert contract_mod.tree_id(repo, "", git=git) == ""
    assert contract_mod.tree_id(repo, "aaaa", git=lambda *args: "") == ""
    assert not status.on("")
    # A result the infrastructure ended is never a pass, and never a score with no verdict.
    Result.infrastructure, Result.unjudged = "the session was lost", True
    lost = status.record(serve, Result(), tree=before, commit="aaaa")
    assert not lost.passed and not lost.unjudged and lost.said == "FAIL"
    again = contract_mod.ContractStatus.load(tmp_path)
    assert again.on(before).keys() == {"tp1:boot", "tp1:serve"}
    # The report's lines: the latest of a check on any content, and the last that passed.
    assert again.last("tp1:boot").commit == "aaaa" and again.last("tp1:serve", passed=True) is None
    assert again.last("tp1:tests") is None


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
