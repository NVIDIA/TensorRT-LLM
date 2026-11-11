# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run: the fixed order, the onboard check, not done and go on, the bounds, resume, the report."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.agents import prompts
from agent_flow.workflows.trtllm_modeling_v2.agents.turns import TurnTimeout
from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.git import (
    CommitIdentity,
    CommitPolicy,
    audit_commits,
    git_output,
)
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem, rows_by_unit
from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run import report as report_mod
from agent_flow.workflows.trtllm_modeling_v2.run.integration import (
    Integration,
    behind,
    load_labels,
    own_paths,
)
from agent_flow.workflows.trtllm_modeling_v2.run.pipeline import NOTES_FILENAME
from agent_flow.workflows.trtllm_modeling_v2.run.state import (
    StateError,
    WorkflowState,
    load_state,
    save_state,
)
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
    facts,
    healthy_smi,
    passing_shell,
    scripted_refs,
)
from .conftest import SCRATCH

FAMILY = str(Path(TARGET_REL).parent)
SLUG = "gpt-oss-120b-sm_103-tp1"
KV_FP8 = {"name": "kv_fp8", "options": {"kv_cache_config": {"dtype": "fp8"}}, "relative_to": "tp1"}
TEP2 = {"name": "tep2", "options": {"tensor_parallel_size": 2}}
BLOCKED = {"decision": "blocked", "summary": "no route to it"}
READY = {"decision": "ready", "summary": "assembled"}


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
        return {"decision": "ready", "summary": "three rows"}

    def kernel_writer(turn, prompt):
        events.append(f"turn:{_task_id(prompt)}")
        return {"decision": "ready", "summary": "landed", "entry": "x"}

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
    assert "it ended blocked: no route to it" in again
    # What a later worker is told of the tasks before it, and of its own checks.
    last = _prompts(calls, "tep2")[0]
    assert "`kv_fp8` (deployment): not done blocked" in last
    assert "`core` (core): done ready" in last and "Nothing of a task not done landed" in last
    assert "The checks it owes: `tep2:boot`, `tep2:accuracy[gsm8k]`." in last
    assert "gate <boot|accuracy|tests>" in last and "--deployment tep2" in last
    script = workspace / "bin" / "trtllm-modeling-v2"
    assert os.access(script, os.X_OK) and f"      {script} gate <boot|" in last
    assert "They run once on the checkout after the last task" in last
    core = _prompts(calls, "core")[0]
    assert "### What the onboard check runs after your turn" in core
    assert "Code decides, on your last commit" in core and "attempt 1 of 3" in core
    assert "`tp1:accuracy[gsm8k]`: reference 85.6 (" in core
    # The report: its first line, then every deployment where it stands.
    report = _report(tmp_path)
    first = report.splitlines()[0]
    assert first.startswith(f"# onboarded: commit `c0ffee1`, ref `refs/mv2/{SLUG}/onboarded`, 20")
    assert "| `tp1` | `accuracy[gsm8k]` | 90 | reference 85.6 (" in report
    assert "| PASS | trtllm-eval | `c0ffee1` |" in report
    assert "| `tep2` | `boot` | - | - | PASS | through the endpoint | `c0ffee1` |" in report
    assert "| `kv_fp8` |" not in report.split("## Done")[1].split("##")[0]
    # Every row is of the head here, so nothing is said under the table.
    assert "Not measured on the head" not in report
    # The tasks, in a reader's word for what each belongs to: a deployment.
    tasks = report.split("## Tasks")[1]
    assert "| task | kind | deployment | status | attempts | grants | decision | summary |" in tasks
    assert "| `kv_fp8` | deployment | kv_fp8 | not done |" in tasks and "unit" not in tasks
    # With what each check attested of the engine that ran: every rank's options, the target.
    row = next(line for line in report.splitlines() if line.startswith("| `tep2` | `boot` |"))
    assert row.endswith(
        "| `c0ffee1` | [engine match: 2 rank(s) x 2 option(s); target "
        "tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss.model.GptOssModel] |"
    )
    assert _verdicts(tmp_path)[0]["attested"].startswith("[engine match: 1 rank(s) x 2 option(s);")
    survey = _prompts(calls, "survey")[0]
    assert "## Task: survey" in survey and "## Surface survey" in survey
    assert f"Answer in `{workspace}/plan.md` and `{workspace}/vocabulary.yaml`" in survey
    # The survey has no worktree and runs no check: its message says neither.
    assert "## Worktree and commits" not in survey and " gate <boot|" not in survey
    not_done = report.split("## Not done")[1].split("##")[0]
    assert "- `kv_fp8` (deployment): it ended blocked: no route to it" in not_done
    assert f"evidence: `{workspace}/decisions/kv_fp8-3.yaml`" in not_done
    assert "- native libraries: the head's are the onboarded ones" in report
    unchecked = report.split("## What the run did not check")[1].split("##")[0]
    assert "registered with the repository's CI" in unchecked and "(unknown)" in unchecked
    assert "- assembler: claude-code claude-test-model (default effort)" in report
    assert {role for role, _ in calls} == {"surveyor", "kernel_writer", "assembler"}


def test_a_kernel_row_that_is_not_done_is_skipped_and_the_run_goes_on(
    tmp_path, task_file, monkeypatch
):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rows = [
        {"n": 1, "mode": "onboard", "target": "ops/norm/rms"},
        {"n": 2, "mode": "onboard", "target": "ops/norm/scale"},
    ]
    _write_vocabulary(workspace, rows)

    def kernel_writer(turn, prompt):
        if _task_id(prompt) == "k1-rms" and turn == 1:
            return {"decision": "blocked", "summary": "tensorrt_llm/x.py:f overflows above 2^31"}
        if _task_id(prompt) == "k1-rms":
            return {"decision": "blocked", "summary": "no kernel exports it on sm_103"}
        return {"decision": "ready", "summary": "landed", "entry": "x"}

    script = Script(kernel_writer=kernel_writer)
    calls = _run(tmp_path, _spec(task_file), monkeypatch, script)
    assert [_task_id(p) for _, p in calls] == [*["k1-rms"] * 3, "k2-scale", "core"]
    assert _state(tmp_path).onboarded
    # A kernel task's next attempt is told why the last one ended, in its worker's words.
    again = _prompts(calls, "k1-rms")[1]
    assert "it ended blocked: tensorrt_llm/x.py:f overflows above 2^31" in again
    assert "Attempt 2 of 3 of task `k1-rms`" in again
    # A later worker sees the row as skipped and is told what to do where it needs the entry.
    core = _prompts(calls, "core")[0]
    assert "- `k1-rms` (kernel): skipped blocked. no kernel exports it on sm_103" in core
    assert "write its wrapper in the target's own `ops/`" in core
    report = _report(tmp_path)
    skipped = report.split("## Skipped kernel rows")[1].split("##")[0]
    assert "- `k1-rms` (queue row 1): it ended blocked: no kernel exports it on sm_103" in skipped
    assert "k1-rms" not in report.split("## Not done")[1].split("##")[0]


def test_a_workers_one_gpu_commands_run_behind_the_sites_prefix(
    tmp_path, task_file, repo, monkeypatch
):
    """`environment.launch` is where an entry's test runs, not the orchestrator's host."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_vocabulary(workspace, [{"n": 1, "mode": "onboard", "target": "ops/norm/rms"}])
    spec = yaml.safe_load(_spec(task_file).read_text())
    launch = spec["environment"]["launch"] = "srun -N1 --gpus=1 --container-image=img"
    task_file.write_text(yaml.safe_dump(spec))
    calls = _run(tmp_path, task_file, monkeypatch, Script())
    [kernel] = _prompts(calls, "k1-rms")
    [command] = [line.strip() for line in kernel.splitlines() if " -m pytest " in line]
    assert command.startswith(f"cd {repo} && PYTHONPATH={repo} ")
    assert f" TMPDIR={SCRATCH} " in command and " TLLM_WORKER_USE_SINGLE_PROCESS=1 " in command
    assert f" {launch} /opt/venv/bin/python -m pytest " in command
    # The prefix a worker is given for any other command is the same: the spec's variables
    # and the site's prefix, with no device of this host.
    said = re.search(r"the site's, which brings its own device:\n\n    ([^\n]+)\n", kernel)
    assert said and said.group(1).endswith(f" {launch}") and f"TMPDIR={SCRATCH} " in said.group(1)
    assert "CUDA_VISIBLE_DEVICES" not in kernel


def test_a_task_that_changed_the_libraries_without_landing_is_named_in_the_report(
    tmp_path, task_file, repo, monkeypatch
):
    def kv(attempt, prompt):
        if attempt == 1:
            (repo / "tensorrt_llm" / "libtensorrt_llm.so").write_bytes(b"rebuilt for kv")
        return BLOCKED

    script = Script(assembler=_by_task({"kv_fp8": kv}))
    calls = _run(tmp_path, _spec(task_file, KV_FP8, TEP2), monkeypatch, script, max_attempts=2)
    stands = {t["id"]: t for t in _tasks(tmp_path)}
    # The hashes are the task's first start and its last end: a later attempt that built
    # nothing does not hide what the first one did.
    assert stands["kv_fp8"]["lib_start"] == stands["core"]["lib_end"]
    assert stands["kv_fp8"]["lib_end"] != stands["kv_fp8"]["lib_start"]
    report = _report(tmp_path)
    assert "- native libraries were changed by `kv_fp8`, which did not land" in report
    # What the head holds now is not what was onboarded; nothing was kept to point at.
    onboarded = _state(tmp_path).onboarded["libraries"]
    assert f"are not the onboarded ones ({onboarded})" in report
    assert "the accepted ones are in" not in report
    assert "libraries were changed by" not in _prompts(calls, "kv_fp8")[0]


def test_a_dispatch_lists_the_newest_ended_tasks_and_says_where_the_rest_are(
    tmp_path, task_file, monkeypatch
):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rows = [{"n": n, "mode": "onboard", "target": f"ops/norm/op{n}"} for n in range(1, 34)]
    _write_vocabulary(workspace, rows)
    calls = _run(tmp_path, _spec(task_file), monkeypatch, Script())
    core = _prompts(calls, "core")[0]
    # Thirty-four ended before core (the survey and 33 rows): the newest thirty are listed.
    assert f"- (4 earlier task(s): `{workspace}/tasks.yaml`)" in core
    assert "`k3-op3` (kernel)" not in core and "- `k4-op4` (kernel): done ready" in core
    assert core.count("(kernel): done ready") == 30


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
        assert "| no verdict | trtllm-eval |" in report
        assert _verdicts(tmp_path)[1]["unjudged"] and _verdicts(tmp_path)[1]["passed"]
    else:
        assert "tp1: it names no benchmark, so its boot alone judges it" in report


def test_only_a_benchmark_with_an_anchor_can_refuse_onboarded(
    tmp_path, task_file, repo, monkeypatch
):
    refs = repo / "tests/integration/defs/accuracy/references/gsm8k.yaml"
    refs.write_text(yaml.safe_dump({"another/model": [{"accuracy": 85.6}]}))
    # No anchor, the boot passes, the evaluator fails: nothing judged the score it never gave.
    shell = _checks(fail={"tp1:accuracy"})
    calls = _run(tmp_path, _spec(task_file), monkeypatch, Script(), shell_runner=shell)
    assert len(calls) == 1 and _state(tmp_path).onboarded["accuracy"] == "unjudged (boot only)"
    report = _report(tmp_path)
    assert report.splitlines()[0].endswith("; accuracy unjudged (boot only)")
    # Its failure is a line of its own, and its row says it failed.
    said = "- `tp1:accuracy[gsm8k]` has no anchor, and its evaluation failed: [FAIL] tp1:accuracy"
    assert said in report
    assert "| `tp1` | `accuracy[gsm8k]` | - | none (no declared anchor" in report
    assert "| FAIL | trtllm-eval |" in report


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
    # The command that shows it again, on the tree that was checked: here the checkout itself.
    assert report[3] == (
        f"- reproduce: `{workspace}/bin/trtllm-modeling-v2 gate boot --task {workspace}/task.yaml "
        f"--deployment tp1 --tree {tmp_path}/repo`"
    )
    assert report[4] == "- last commit where `tp1:boot` passed: `none`"
    assert "## Done\n\n(none)" in "\n".join(report)


def test_a_turn_that_raises_or_runs_out_of_time_is_a_failed_attempt_with_why(
    tmp_path, task_file, monkeypatch
):
    def core(attempt, prompt):
        if attempt == 1:
            raise TurnTimeout("003-assembler: no answer within 60s; the turn was stopped.")
        if attempt == 2:
            raise OSError("the CLI could not be started")
        return READY

    script = Script(assembler=_by_task({"core": core}))
    calls = _run(tmp_path, _spec(task_file), monkeypatch, script)
    second, third = _prompts(calls, "core")[1:]
    assert "its turn raised TurnTimeout: 003-assembler: no answer within 60s" in second
    assert "its turn raised OSError: the CLI could not be started" in third
    assert _tasks(tmp_path)[1]["attempts"] == 3 and _state(tmp_path).onboarded


def test_a_worker_that_never_gets_core_ready_stops_the_run_with_its_own_reason(
    tmp_path, task_file, monkeypatch
):
    shell = _checks(fail={"tp1:boot"})
    # Ready once, and the boot failed; then it gave up twice without reaching a check.
    script = Script(assembler=_by_task({"core": [READY, BLOCKED]}))
    workflow = _workflow(tmp_path, shell_runner=shell)
    _install(monkeypatch, workflow, script)
    with workflow:
        assert workflow.run(_spec(task_file)) is False
    report = _report(tmp_path)
    assert report.startswith("# not onboarded: stopped at `core`, not done after 3 attempt(s)")
    assert "- result: it ended blocked: no route to it" in report
    # What ended it is the worker's own word, not the check an earlier attempt failed.
    decisions = tmp_path / "ws" / "decisions"
    assert (
        f"- its last attempt ended at no check of the run: see `{decisions}/core-*.yaml`" in report
    )
    assert "- reproduce:" not in report and shell.ran == ["tp1:boot"]


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
        return {"decision": "ready", "summary": "no rows"}

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


def test_what_preflight_left_out_gets_no_task_and_a_score_nothing_judges_gives_no_verdict(
    tmp_path, task_file, monkeypatch
):
    mid = {"name": "mid", "options": {"max_batch_size": 8}}
    kv = {**KV_FP8, "relative_to": "mid"}
    # Scored on a benchmark nothing anchors: it runs, and its score is a line with no verdict.
    unanchored = {"name": "mmlu", "reference_yaml": "refs/none.yaml"}
    (tmp_path / "engines").mkdir()
    (tmp_path / "engines" / "wide.yaml").write_text("max_batch_size: 4\n")
    wide = {
        "name": "wide",
        "options": "engines/wide.yaml",
        "benchmarks": [{"name": "gsm8k"}, unanchored],
    }
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
    assert "`wide`" not in left
    # `kv_fp8` is held to `mid`'s score, and `mid` is not done: its checks are not run for
    # it, and the score is reported with no verdict. What preflight left out is not run.
    assert shell.ran == [
        "tp1:boot",
        "tp1:accuracy",
        "kv_fp8:boot",
        "kv_fp8:accuracy",
        "wide:boot",
        "wide:accuracy",
        "wide:accuracy",
    ]
    assert "| `wide` | `accuracy[gsm8k]` |" in report
    assert "| `wide` | `accuracy[mmlu]` | 90 | none (no declared anchor" in report
    assert "| no verdict | trtllm-eval |" in report.split("| `wide` | `accuracy[mmlu]` |")[1]
    assert "`wide:accuracy[mmlu]`" in _prompts(calls, "wide")[0]
    # Each deployment's dispatch names its options: a file, or where the spec states them.
    assert f"`{tmp_path}/engines/wide.yaml` state what is asked." in _prompts(calls, "wide")[0]
    asked = "`deployments.kv_fp8` in the spec state what is asked."
    assert asked in _prompts(calls, "kv_fp8")[0]
    assert "| `kv_fp8` | `accuracy[gsm8k]` | 90 | relative (the score of `mid`) | no verdict |" in (
        report
    )
    # It waits for its base: once `mid` is done and scored, the score on record is judged
    # against it, and nothing of `kv_fp8` is loaded again.
    shell = _checks(score=91.0)
    _run(tmp_path, task, monkeypatch, Script(), shell_runner=shell)
    assert shell.ran == ["mid:boot", "mid:accuracy"]
    report = _report(tmp_path)
    assert "| `kv_fp8` | `accuracy[gsm8k]` | 90 | relative (the score of `mid`) | PASS |" in report
    judged = [v for v in _verdicts(tmp_path) if v["gate"].startswith("accuracy")][-1]
    assert (judged["deployment"], judged["passed"], judged["unjudged"]) == ("kv_fp8", True, False)
    assert judged["reference"] == 91.0 and "its score on record, judged now" in judged["detail"]


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


def test_a_check_the_infrastructure_ended_runs_once_more_and_is_no_verdict_of_its_own(
    tmp_path, task_file, monkeypatch
):
    inner = _checks()
    lost = {"left": 1}

    def shell(command, env=None, timeout=0):
        if "/eval." in command and lost["left"]:
            lost["left"] -= 1
            inner.ran.append("tp1:accuracy (session lost)")
            return CommandResult(returncode=1, infrastructure="the allocation ended under it")
        return inner(command, env, timeout)

    calls = _run(tmp_path, _spec(task_file), monkeypatch, Script(), shell_runner=shell)
    # Not the worker's failure and not the target's: the check runs again in the same
    # attempt, and what is recorded is the verdict of the run that measured something.
    assert inner.ran == ["tp1:boot", "tp1:accuracy (session lost)", "tp1:accuracy"]
    assert len(calls) == 1 and _tasks(tmp_path)[1]["attempts"] == 1
    assert [v["passed"] for v in _verdicts(tmp_path)] == [True, True]
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert (
        "check without a verdict**: tp1:accuracy[gsm8k]: the allocation ended under it" in journal
    )
    # Lost twice running, it is recorded as not passed, and the attempt fails on it.
    lost["left"] = 2
    task = _spec(task_file, KV_FP8)
    (tmp_path / "ws" / ".trtllm_modeling_v2_state.json").unlink()
    (tmp_path / "ws" / "tasks.yaml").unlink()
    (tmp_path / "ws" / "contract_status.jsonl").unlink()
    calls = _run(tmp_path, task, monkeypatch, Script(), shell_runner=shell, max_attempts=1)
    assert _state(tmp_path).onboarded == {} and len(calls) == 1
    assert "did not run to a verdict: the allocation ended under it" in _tasks(tmp_path)[1]["note"]


# ----------------------------------------------------------------------- the final pass


class _Head:
    """The checkout's commit as a scripted git answers it; a test moves it when a task lands."""

    def __init__(self, at: str = "c0ffee1"):
        self.at = at
        self.refs: dict[str, str] = {}

    def __call__(self, repo, *args, stdin=None):
        if args[:1] in (("rev-parse",), ("update-ref",)):
            return scripted_refs(self.refs, self.at, args)
        return ""


def test_the_final_pass_runs_once_and_one_fix_task_names_every_failure_and_finding(
    tmp_path, task_file, repo, monkeypatch
):
    """Each landing is read for the conventions; findings hold nothing and go to the fix."""
    task = _spec(task_file, KV_FP8, TEP2)
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
    kwargs = dict(shell_runner=shell, git_runner=head, command_runner=healthy_smi(2))
    workflow = _workflow(tmp_path, **kwargs)
    calls = _install(monkeypatch, workflow, script)
    with workflow:
        assert workflow.run(task) is True
    # A finding is kept with the task whose landing first showed it, and held neither.
    found = json.loads((tmp_path / "ws" / "conventions.json").read_text())
    assert set(found.values()) == {"core", "kv_fp8", "fix-1"} and all(
        t["status"] == "done" and t["attempts"] <= 1 for t in _tasks(tmp_path)
    )
    assert [_task_id(p) for _, p in calls] == ["core", "kv_fp8", "tep2", "fix-1"]
    goal = _prompts(calls, "fix-1")[0]
    assert "repair what the final pass found, every item:" in goal
    assert "- check `kv_fp8:boot` failed at c0ffee1: [FAIL] kv_fp8:boot: " in goal
    assert f"- convention: {TARGET_REL}/modeling.py:1 imports Modeling V1 model code ({v1})" in goal
    assert "(first at `core`)" in goal and "(first at `kv_fp8`)" in goal
    assert "## Task: the final fix" in goal and "Reproduce a failed check with the gate" in goal
    # The text says what ran and what will run, as the list of checks below has it.
    assert "after a failed boot, that deployment's other checks did not run" in goal
    assert "`tp1` passed its checks when it was onboarded, maybe on an earlier commit" in goal
    again = "Afterwards code runs every check of each deployment that had a failure, and those of"
    assert f"{again} `tp1`, once more." in goal and "ran every check" not in goal
    # A deployment whose boot failed is not scored: its benchmark would start the same
    # engine to the same end. After the fix, that deployment's checks and core's own run
    # once more, on the new commit (core's benchmark first, as what `kv_fp8` is held to).
    # What passed before keeps the commit it was measured on.
    assert shell.ran == [
        "tp1:boot",
        "tp1:accuracy",
        "kv_fp8:boot",
        "tep2:boot",
        "tep2:accuracy",
        "kv_fp8:boot",
        "tp1:accuracy",
        "kv_fp8:accuracy",
        "tp1:boot",
    ]
    named = goal.split("## What the run did before this task")[0]
    assert "kv_fp8:accuracy" not in named and "tep2" not in named
    report = _report(tmp_path)
    assert "| `kv_fp8` | `boot` | - | - | PASS | through the endpoint | `f1xed00` |" in report
    assert "| PASS | trtllm-eval | `f1xed00` |" in report.split("| `kv_fp8` | `accuracy")[1]
    assert "| `tep2` | `boot` | - | - | PASS | through the endpoint | `c0ffee1` |" in report
    # Each row names its commit, and a reader still takes the table for the head: one line
    # under it says which checks were not measured there, and why.
    under = report.split("## Done")[1].split("##")[0].strip().splitlines()[-1]
    assert under == (
        "Not measured on the head `f1xed00`: `tep2:boot`, `tep2:accuracy[gsm8k]` (on "
        "`c0ffee1`). Each passed on its commit; what passed is measured again only after a "
        "`fix` task, and then only for the first deployment and for a deployment that had a "
        "failure."
    )
    # Nothing of the final pass moves the onboard record.
    assert report.startswith("# onboarded: commit `c0ffee1`")
    left = report.split("## Convention findings left")[1].split("##")[0]
    assert "quant.py" in left and "(first at `kv_fp8`)" in left and "Modeling V1" not in left
    # The next start has its own entry: what is still found gets one more fix task, once.
    calls = _run(tmp_path, task, monkeypatch, Script(), **kwargs)
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
    assert "| `kv_fp8` | `boot` | - | - | FAIL | through the endpoint |" in _report(tmp_path)
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
        return {"decision": "ready", "summary": "drafted"}

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


def test_a_fix_that_breaks_core_is_reported_and_takes_nothing_back(
    tmp_path, task_file, monkeypatch
):
    head = _Head()
    shell = _checks(fail={"kv_fp8:accuracy"})

    def fix(attempt, prompt):
        shell.fail.clear()
        shell.fail.add("tp1:boot")  # what it changed for the one broke the other
        head.at = "f1xed00"
        return READY

    script = Script(assembler=_by_task({"fix-1": fix}))
    workflow = _workflow(tmp_path, shell_runner=shell, git_runner=head)
    _install(monkeypatch, workflow, script)
    with workflow:
        assert workflow.run(_spec(task_file, KV_FP8)) is True
    report = _report(tmp_path)
    # Onboarded on the commit where core's checks passed; what the head does now is a line.
    assert report.startswith("# onboarded: commit `c0ffee1`")
    assert _state(tmp_path).onboarded["commit"] == "c0ffee1"
    assert "| `tp1` | `boot` | - | - | FAIL | through the endpoint | `f1xed00` |" in report
    assert "| `kv_fp8` | `accuracy[gsm8k]` | 90 | relative (the score of `tp1`) | PASS |" in report
    # Core's benchmark ran on the new commit as what `kv_fp8` is held to, then its boot.
    assert shell.ran[-3:] == ["tp1:accuracy", "kv_fp8:accuracy", "tp1:boot"]


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
        decision_file.write_text(yaml.safe_dump({"decision": "ready", "summary": "written"}))
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
    # What a role appended to the host facts is kept by the starts that follow.
    workspace = tmp_path / "ws"
    facts = workspace / "host_facts.md"
    facts.write_text(facts.read_text() + "- `srun` needs `--mpi=pmix` on this cluster\n")
    # A start on a finished run has nothing left to do: every pass is on record.
    calls = _run(
        tmp_path, task, monkeypatch, Script(), shell_runner=shell, command_runner=healthy_smi(2)
    )
    assert calls == [] and len(shell.ran) == 6
    assert _report(tmp_path).startswith("# onboarded: commit `c0ffee1`")
    assert facts.read_text().endswith("- `srun` needs `--mpi=pmix` on this cluster\n")
    assert facts.read_text().count("Learned during this run") == 1
    # One journal for the workspace, and one set of files a turn: a later start numbers its
    # turns on from the earlier ones' and overwrites none. Three ran, and one was cut.
    journal = (workspace / "journal.md").read_text()
    assert journal.count("# Onboarding journal: ") == 1 and journal.count("**run opened**") == 1
    assert journal.count("**run resumed**") == 2
    ids = sorted(path.name.split("-")[0] for path in (workspace / "turns").glob("*.request.json"))
    assert ids == ["001", "002", "003", "004"]


def test_a_new_start_reopens_what_was_not_done_within_its_grants(tmp_path, task_file, monkeypatch):
    task = _spec(task_file, KV_FP8)
    script = Script(assembler=_by_task({"kv_fp8": BLOCKED}))

    def turns(script=script) -> int:
        return len(_prompts(_run(tmp_path, task, monkeypatch, script), "kv_fp8"))

    # Its own three attempts, then fresh ones on each of two more starts, and no more.
    assert [turns() for _ in range(2)] == [3, 3]
    # A fresh attempt has the same number as one of the earlier set: the decision that one
    # left is not taken for this one's when its worker writes none.
    silent = Script(assembler=_by_task({"kv_fp8": lambda attempt, prompt: None}))
    assert turns(silent) == 6  # each attempt: its turn, then the one asked to write the file
    assert "wrote no valid receipt" in _tasks(tmp_path)[2]["note"]
    assert turns() == 0
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
    # A deployment the spec asks for again gets its task again, not a skip that stays.
    back = _spec(task_file, TEP2, mid)
    calls = _run(tmp_path, back, monkeypatch, Script(), command_runner=healthy_smi(2))
    assert [_task_id(p) for _, p in calls] == ["mid"]
    stands = {t["id"]: (t["status"], t["note"]) for t in _tasks(tmp_path)}
    assert stands["mid"] == ("done", "") and stands["kv_fp8"][0] == "skipped"
    assert "task(s) added: mid" in (tmp_path / "ws" / "journal.md").read_text()


def test_the_library_hash_at_a_tasks_start_survives_a_cut_during_its_turn(
    tmp_path, task_file, repo, monkeypatch
):
    def kv(attempt, prompt):
        if attempt == 1:
            (repo / "tensorrt_llm" / "libtensorrt_llm.so").write_bytes(b"rebuilt for kv")
            raise KeyboardInterrupt
        return BLOCKED

    script = Script(assembler=_by_task({"kv_fp8": kv}))
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, _spec(task_file, KV_FP8), monkeypatch, script)
    cut = {t["id"]: t for t in _tasks(tmp_path)}
    # On disk before the worker was asked: the restart does not take the rebuilt libraries
    # for the ones the task started from.
    assert cut["kv_fp8"]["status"] == "running"
    assert cut["kv_fp8"]["lib_start"] == cut["core"]["lib_end"] != ""
    _run(tmp_path, _spec(task_file, KV_FP8), monkeypatch, script, max_attempts=1)
    assert "- native libraries were changed by `kv_fp8`, which did not land" in _report(tmp_path)


def test_a_task_list_that_cannot_be_read_is_refused_and_left_as_it_is(
    tmp_path, task_file, monkeypatch
):
    _run(tmp_path, _spec(task_file), monkeypatch, Script())
    tasks = tmp_path / "ws" / "tasks.yaml"
    for torn in ("tasks: [unclosed", "- a\n- b\n", "tasks: 3\n"):
        tasks.write_text(torn)
        # It holds the attempts and grants every task has spent: not begun anew over it.
        with pytest.raises(StateError, match="does not hold the task list") as refused:
            _run(tmp_path, _spec(task_file), monkeypatch, Script())
        assert "restore it, or start over with --clean" in str(refused.value)
        assert tasks.read_text() == torn


def test_another_first_deployment_is_not_taken_for_the_one_core_was_built_for(
    tmp_path, task_file, monkeypatch
):
    _run(tmp_path, _spec(task_file, KV_FP8), monkeypatch, Script())
    assert _state(tmp_path).onboarded and _tasks(tmp_path)[1]["ref"] == "tp1"
    # `core` is the first deployment. With another one first, what was built and onboarded
    # would be reported as that one's: the start is refused, with what to do.
    spec = yaml.safe_load(task_file.read_text())
    spec["deployments"] = [{"name": "tep2", "options": {"tensor_parallel_size": 2}}]
    task_file.write_text(yaml.safe_dump(spec))
    workflow = _workflow(tmp_path, command_runner=healthy_smi(2))
    calls = _install(monkeypatch, workflow, Script())
    with workflow, pytest.raises(StateError) as refused:
        workflow.run(task_file)
    said = str(refused.value)
    assert "the first deployment is `tep2` now" in said and "was built for `tp1`" in said
    assert "put `tp1` first again, or start over with --clean" in said and calls == []
    # A deployment added after it changes nothing of that.
    calls = _run(
        tmp_path, _spec(task_file, TEP2), monkeypatch, Script(), command_runner=healthy_smi(2)
    )
    assert [_task_id(p) for _, p in calls] == ["tep2"]


class _Tree(_Head):
    """A scripted git that also says what the tree holds outside its commit."""

    dirt = ""

    def __call__(self, repo, *args, stdin=None):
        return self.dirt if args[:2] == ("status", "--porcelain") else super().__call__(repo, *args)


def test_a_tree_that_is_not_its_commit_gives_no_verdict_on_record_or_not(
    tmp_path, task_file, monkeypatch
):
    git, shell = _Tree(), _checks()
    task = _spec(task_file, KV_FP8)
    run = dict(shell_runner=shell, git_runner=git, max_attempts=1)
    # An untracked source: a check would import it, and the commit that lands would not hold it.
    git.dirt = "?? tensorrt_llm/extra.py\n"
    _run(tmp_path, task, monkeypatch, Script(), **run)
    said = "[NO VERDICT] tp1:boot: 1 file(s) in the tree are in no commit: tensorrt_llm/extra.py"
    assert said in _tasks(tmp_path)[1]["note"]
    assert shell.ran == [] and _verdicts(tmp_path) == [] and not _state(tmp_path).onboarded
    # Said when the run starts, not found at its end: the checkout itself is not its commit.
    told = "**checkout**: the checkout is not its commit: 1 file(s) in the tree are in no commit: "
    told += "tensorrt_llm/extra.py. A check that runs in it gives no verdict until"
    assert told in (tmp_path / "ws" / "journal.md").read_text()
    # What a sandbox drops into a directory is no source, nor what an engine leaves behind.
    git.dirt = "?? .mcp.json\n?? tensorrt_llm/.claude/x.py\n?? core.4242\n?? lm_cache/a.db\n"
    _run(tmp_path, task, monkeypatch, Script(), **run)
    assert _state(tmp_path).onboarded
    assert shell.ran == ["tp1:boot", "tp1:accuracy", "kv_fp8:boot", "kv_fp8:accuracy"]
    # A pass on record is no verdict about a tree that is not the commit: not reused, not run.
    git.dirt = " M tensorrt_llm/model.py\n"
    _run(tmp_path, task, monkeypatch, Script(), **run)
    assert len(shell.ran) == 4 and len(_verdicts(tmp_path)) == 4
    report = _report(tmp_path)
    assert "| `kv_fp8` | `boot` | - | - | no verdict | - | `c0ffee1` |" in report
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "[NO VERDICT] kv_fp8:boot: 1 file(s) in the tree are in no commit" in journal
    # The onboard record stands. After the fix task core's own checks were asked again,
    # of the same tree: no verdict there either, and the row says so.
    assert report.startswith("# onboarded: commit `c0ffee1`")
    assert "| `tp1` | `boot` | - | - | no verdict | - | `c0ffee1` |" in report


def test_the_runs_own_workspace_inside_the_tree_is_no_uncommitted_source(tmp_path):
    listed = "?? ws/turns/probe.py\n?? other/probe.py\n"
    workflow = SimpleNamespace(
        task_data={"trtllm_repo_path": str(tmp_path)},
        _git=lambda repo, *args: "c0ffee1\n" if args[0] == "rev-parse" else listed,
        _commit_policy=lambda: CommitPolicy(denied_paths=("ws/",)),
    )
    said = report_mod.unmeasurable(workflow, tmp_path, "c0ffee1")
    assert said == "1 file(s) in the tree are in no commit: other/probe.py"


def test_a_tree_that_moves_under_a_check_gives_no_verdict(tmp_path, task_file, monkeypatch):
    git, inner = _Tree(), _checks()

    def shell(command, env=None, timeout=0):
        if "/boot." in command:
            git.at = "moved00"  # the branch advanced while the engine was up
        return inner(command, env, timeout)

    run = dict(shell_runner=shell, git_runner=git, max_attempts=1)
    _run(tmp_path, _spec(task_file), monkeypatch, Script(), **run)
    note = _tasks(tmp_path)[1]["note"]
    assert "[NO VERDICT] tp1:boot: the tree changed under the check: the tree is at moved00" in note
    assert _verdicts(tmp_path) == [] and not _state(tmp_path).onboarded


def test_libraries_rebuilt_under_a_check_or_a_tree_git_cannot_name_give_no_verdict(
    tmp_path, task_file, repo, monkeypatch
):
    inner = _checks()

    def rebuilds(command, env=None, timeout=0):
        if "/boot." in command:
            (repo / "tensorrt_llm" / "libtensorrt_llm.so").write_bytes(b"rebuilt meanwhile")
        return inner(command, env, timeout)

    _run(tmp_path, _spec(task_file), monkeypatch, Script(), shell_runner=rebuilds, max_attempts=1)
    said = "the tree changed under the check: its native libraries are not the ones the check"
    assert said in _tasks(tmp_path)[1]["note"]
    assert _verdicts(tmp_path) == [] and not _state(tmp_path).onboarded

    class Unnamed(_Head):
        def __call__(self, repo, *args, stdin=None):
            return "" if args[-1].endswith("^{tree}") else super().__call__(repo, *args)

    # Nothing is established about content git names no tree for: not a pass to onboard on.
    run = dict(shell_runner=_checks(), git_runner=Unnamed(), max_attempts=1)
    _run(tmp_path, _spec(task_file), monkeypatch, Script(), **run)
    assert "the tree's content cannot be identified: git names no commit" in (
        (tmp_path / "ws" / "journal.md").read_text()
    )
    assert run["shell_runner"].ran == [] and not _state(tmp_path).onboarded


def test_the_report_prints_what_this_start_used_or_measured_not_the_ledgers_latest(
    tmp_path, task_file, monkeypatch
):
    git, shell = _Tree(), _checks()
    task = _spec(task_file, KV_FP8)
    run = dict(shell_runner=shell, git_runner=git)
    _run(tmp_path, task, monkeypatch, Script(), **run)  # content A
    git.at = "bbbbbbb"
    _run(tmp_path, task, monkeypatch, Script(), **run)  # content B: measured again
    # With the score it is held to: `tp1`'s, on the same content.
    assert shell.ran[4:] == ["kv_fp8:boot", "tp1:accuracy", "kv_fp8:accuracy"]
    report = _report(tmp_path)
    assert "| `kv_fp8` | `boot` | - | - | PASS | through the endpoint | `bbbbbbb` |" in report
    # Core's boot, which the final pass leaves alone, is the onboarded commit's.
    assert "| `tp1` | `boot` | - | - | PASS | through the endpoint | `c0ffee1` |" in report
    git.at = "c0ffee1"
    _run(tmp_path, task, monkeypatch, Script(), **run)  # content A again: A's verdicts are used
    assert len(shell.ran) == 7
    report = _report(tmp_path)
    assert "| `kv_fp8` | `boot` | - | - | PASS | through the endpoint | `c0ffee1` |" in report
    assert "`bbbbbbb`" not in report.split("## Done")[1].split("##")[0]
    # A boot that fails now: its benchmark is not run, and the row says so, whatever passed before.
    git.at, shell.fail = "ddddddd", {"kv_fp8:boot"}
    _run(tmp_path, task, monkeypatch, Script(), **run)
    report = _report(tmp_path)
    assert "| `kv_fp8` | `boot` | - | - | FAIL | through the endpoint | `ddddddd` |" in report
    row = "| `kv_fp8` | `accuracy[gsm8k]` | - | relative (the score of `tp1`) | not run | - | `-` | - |"
    assert row in report


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
    # A key a later version adds is not this version's to refuse.
    stored = json.loads(path.read_text())
    path.write_text(json.dumps({**stored, "added_later": 1}))
    assert load_state(path) == original
    path.write_text(json.dumps({"version": 3, "task_path": "t", "stage": "execution"}))
    with pytest.raises(StateError, match="version 3, expected 4.*start over with --clean"):
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
        return {"decision": "ready", "summary": "one row"}

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
        said = prompt.split("## Notes file")[1].split("##")[0]
        assert f"Read `{workspace / NOTES_FILENAME}` first" in said
        assert "Read it again before you write your receipt" in said
        assert "It never permits changing an anchor, a tolerance or a check" in said
    assert "## Your receipt is not valid" in calls[-1][1]
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
    assert not any("/boot." in c or "trtllm-eval" in c or "trtllm-serve" in c for c in ran)
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


def test_a_start_that_stops_early_after_onboarding_exits_zero_and_says_why_it_stopped(
    tmp_path, task_file, monkeypatch
):
    _run(tmp_path, _spec(task_file, KV_FP8), monkeypatch, Script())
    before = _report(tmp_path)
    assert before.startswith("# onboarded: commit `c0ffee1`")
    # A later start whose preflight fails: onboarded stays true, and the report of the
    # earlier start does not stand as this one's.
    for _ in range(2):
        workflow = _workflow(tmp_path, command_runner=healthy_smi(free=20000))
        calls = _install(monkeypatch, workflow, Script())
        with workflow:
            assert workflow.run(task_file) is True
        assert calls == []
    lines = _report(tmp_path).splitlines()
    assert lines[0] == before.splitlines()[0] and lines[1] == ""
    assert lines[2].startswith("- this start (20") and "stopped before its first task" in lines[2]
    assert lines[2].endswith(
        "preflight failed: `tp1` needs 1 free sm_103 device(s) on this host; 0 of 1 are free:"
    )
    # One line for the latest such start, then the earlier report as it was.
    assert lines[3:] == before.splitlines()[2:]
    assert _state(tmp_path).onboarded["commit"] == "c0ffee1"


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
    assert not outcome.merged and outcome.repairable
    assert f"files in {tree} that no commit holds" in outcome.detail
    assert f"\n  {FAMILY}/a.py" in outcome.detail
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


def test_what_lands_is_the_commit_that_was_checked_and_nothing_beside_it(tmp_path):
    repo = _checkout(tmp_path)
    integration = _integration(repo, tmp_path)
    tree = integration.open("core")
    (tree / FAMILY / "a.py").write_text("A = 2\n")
    _git(tree, "commit", "-q", "-am", f"two{_SIGNED}")
    # An untracked source: a check imports it, and the commit that lands does not hold it.
    (tree / FAMILY / "extra.py").write_text("E = 1\n")
    (tree / ".mcp.json").write_text("{}")  # what a sandbox drops is no source
    outcome = integration.integrate("core")
    assert not outcome.merged and outcome.repairable
    assert outcome.detail.endswith(f"or remove them:\n  {FAMILY}/extra.py")
    (tree / FAMILY / "extra.py").unlink()
    checked: list[str] = []

    def check(tree, tip):
        checked.append(tip)
        # The branch moves while the check runs: what it adds was not checked.
        (tree / FAMILY / "a.py").write_text("A = 3\n")
        _git(tree, "commit", "-q", "-am", f"three{_SIGNED}")
        return ""

    landed = integration.integrate("core", check)
    assert landed.merged and [landed.commit] == checked
    assert _git(repo, "rev-parse", "HEAD") == landed.commit
    assert (repo / FAMILY / "a.py").read_text() == "A = 2\n"
    assert [row["subject"] for row in load_labels(tmp_path / "ws")] == ["two"]


def test_a_worker_whose_branch_is_behind_the_checkout_is_told_to_rebase_first(tmp_path):
    repo = _checkout(tmp_path)
    integration = _integration(repo, tmp_path)
    tree = integration.open("kv_fp8")
    assert behind(tree, repo) == (_git(repo, "rev-parse", "HEAD"), "")
    # An earlier attempt left a commit of its own; other tasks have landed since.
    (tree / FAMILY / "a.py").write_text("A = 2\n")
    _git(tree, "commit", "-q", "-am", f"kv{_SIGNED}")
    (repo / "shared.py").write_text("S = 2\n")
    _git(repo, "commit", "-q", "-am", f"other{_SIGNED}")
    head = _git(repo, "rev-parse", "HEAD")
    assert behind(integration.open("kv_fp8"), repo) == (head, "1")
    said = prompts.dispatch("deployment", facts(worktree=tree, head=head, behind="1"))
    assert f"Edit only in `{tree}`" in said and "## Your branch is behind the checkout" in said
    assert "1 commit(s) landed on the checkout after your branch was cut" in said
    assert f"    git -C {tree} rebase {head}" in said
    # A branch with nothing of its own is simply moved to the head: nothing to rebase.
    fresh = integration.open("tep2")
    (repo / "shared.py").write_text("S = 3\n")
    _git(repo, "commit", "-q", "-am", f"another{_SIGNED}")
    assert behind(integration.open("tep2"), repo)[1] == ""
    assert _git(fresh, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")
    # Unless an earlier attempt left edits it did not commit: they stay where they are.
    (fresh / "shared.py").write_text("S = 'mine'\n")
    (repo / "shared.py").write_text("S = 4\n")
    _git(repo, "commit", "-q", "-am", f"a third{_SIGNED}")
    integration.open("tep2")
    assert (fresh / "shared.py").read_text() == "S = 'mine'\n"


def test_a_worktree_is_made_whole_and_a_branch_left_in_a_bad_state_is_said_or_put_right(tmp_path):
    repo = _checkout(tmp_path)
    (repo / ".gitignore").write_text("tensorrt_llm/libs/\n")
    (repo / "tensorrt_llm" / "libs").mkdir(parents=True)
    (repo / "tensorrt_llm" / "libs" / "libtensorrt_llm.so").write_bytes(b"built")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", f"ignore the build{_SIGNED}")
    integration = _integration(repo, tmp_path)
    # A directory a cut-short creation left is no worktree: it is made again, whole.
    half = integration.tree("core")
    half.mkdir(parents=True)
    (half / "stray.txt").write_text("left by a process that died\n")
    tree = integration.open("core")
    assert (tree / "shared.py").is_file() and not (tree / "stray.txt").exists()
    assert _git(tree, "symbolic-ref", "HEAD") == "refs/heads/mv2/run/core"
    # The checkout's native build, which git ignores, is linked into it.
    assert (tree / "tensorrt_llm" / "libs").is_symlink()
    assert (tree / "tensorrt_llm" / "libs" / "libtensorrt_llm.so").read_bytes() == b"built"
    # A worker that left its worktree off its branch is told, in a turn of its own.
    (tree / FAMILY / "a.py").write_text("A = 2\n")
    _git(tree, "commit", "-q", "-am", f"core{_SIGNED}")
    _git(tree, "checkout", "-q", "--detach")
    outcome = integration.integrate("core")
    assert not outcome.merged and outcome.repairable and "is not on its branch" in outcome.detail
    _git(tree, "checkout", "-q", "mv2/run/core")
    # A rebase a stopped process left open is put back before anything else: the conflict
    # is then said as a conflict, with the files, for the worker to resolve.
    (repo / FAMILY / "a.py").write_text("A = 3\n")
    _git(repo, "commit", "-q", "-am", f"other{_SIGNED}")
    head = _git(repo, "rev-parse", "HEAD")
    assert subprocess.run(["git", "-C", str(tree), "rebase", head], capture_output=True).returncode
    outcome = integration.integrate("core")
    assert not outcome.merged and outcome.base == head and outcome.conflicts == (f"{FAMILY}/a.py",)
    assert _git(tree, "symbolic-ref", "HEAD") == "refs/heads/mv2/run/core"


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


def test_the_audit_finds_every_kind_of_commit_the_rules_refuse(tmp_path):
    """On a real repository on purpose: the audit reads what git wrote."""
    repo = _checkout(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")

    def commit(message: str, path: str, link: str = "") -> str:
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).symlink_to(link) if link else (repo / path).write_text("x\n")
        _git(repo, "add", path)
        _git(repo, "commit", "-q", "-m", message)
        return _git(repo, "rev-parse", "HEAD")

    good = commit(f"good{_SIGNED}", "good.py")
    unsigned = commit("unsigned", "unsigned.py")
    other = commit("mismatch\n\nSigned-off-by: Operator <op@example.invalid>", "mismatch.py")
    # Trailers are one block: the co-author sits right above the sign-off.
    shared = commit(f"co\n\nCo-authored-by: Bot <b@example.invalid>\n{_SIGNED.strip()}", "co.py")
    link = commit(f"link{_SIGNED}", "link", link="good.py")
    placeholder = commit(f"ph{_SIGNED}", "pkg/.mcp.json")
    findings = audit_commits(repo, f"{base}..HEAD")
    by_sha: dict[str, list[str]] = {}
    for finding in findings:
        by_sha.setdefault(finding.sha, []).append(finding.problem)
    assert good not in by_sha
    assert by_sha[unsigned] == ["no Signed-off-by"]
    assert "no sign-off carries the author email" in by_sha[other][0]
    assert by_sha[shared] == ["carries Co-authored-by"]
    assert by_sha[link] == ["adds a symlink: link"]
    assert by_sha[placeholder] == ["adds a sandbox placeholder: pkg/.mcp.json"]
    assert audit_commits(repo, f"{base}..HEAD", CommitPolicy(allow_co_authors=True)) != findings


def test_the_judge_is_guarded_by_path_as_git_knows_it_and_by_what_lands(tmp_path):
    repo = _checkout(tmp_path)
    (repo / "refs").mkdir()
    (repo / "refs/gsm8k.yaml").write_text("org/toy: []\n")
    (repo / "alias.yaml").symlink_to("refs/gsm8k.yaml")
    # A reference the spec names through `.`, `..` and a link: the link and its file are guarded.
    spec = {
        "trtllm_repo_path": str(repo),
        "accuracy": {"reference_yaml": {"default": "./refs/../alias.yaml"}},
    }
    policy = CommitPolicy.from_task(spec, workspace=repo / "ws", own_paths=own_paths(TARGET_REL))
    assert policy.path_problem("alias.yaml") and policy.path_problem("refs/gsm8k.yaml")
    said = policy.path_problem("tensorrt_llm/commands/eval.py")
    assert said.startswith("tensorrt_llm/commands/eval.py is a judge path")
    assert policy.path_problem("ws/journal.md") == (
        "ws/journal.md is under ws/, which this run never commits (its own workspace)"
    )
    # A file is matched exactly and a directory by path component, never by a shared spelling.
    for free in (
        "refs/gsm8k.yaml.orig",
        "tensorrt_llm/commands/eval.py.backup",
        "tensorrt_llm/evaluate_more/x.py",
        "wsx/journal.md",
    ):
        assert not policy.path_problem(free), free
    assert policy.label([f"{FAMILY}/a b.py"]) == "target"
    assert policy.label([f"{FAMILY}_more/a.py"]) == "shared"
    # A name git would quote is read as it is: the evaluator's directory is still the judge's.
    integration = _integration(repo, tmp_path)
    tree = integration.open("quoted")
    (tree / "tensorrt_llm/evaluate").mkdir(parents=True)
    (tree / "tensorrt_llm/evaluate/na\u00efve eval.py").write_text("EASIER = True\n")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", f"an easier judge{_SIGNED}")
    outcome = integration.integrate("quoted")
    assert not outcome.merged
    assert "tensorrt_llm/evaluate/na\u00efve eval.py is a judge path" in outcome.detail
    # A target file with a space in its name lands, labelled the target's own.
    tree = integration.open("spaced")
    (tree / FAMILY / "a b.py").write_text("B = 1\n")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", f"a spaced name{_SIGNED}")
    assert integration.integrate("spaced").merged
    [row] = load_labels(tmp_path / "ws")
    assert (row["label"], row["outside"]) == ("target", [])
    # An edit only a merge commit carries is in no single commit, and is still seen.
    tree = integration.open("merged")
    _git(tree, "checkout", "-q", "-b", "side")
    (tree / FAMILY / "side.py").write_text("S = 1\n")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", f"a side change{_SIGNED}")
    _git(tree, "checkout", "-q", integration.branch("merged"))
    (tree / FAMILY / "own.py").write_text("O = 1\n")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", f"the task's change{_SIGNED}")
    _git(tree, "merge", "-q", "--no-ff", "--no-commit", "side")
    (tree / "tensorrt_llm/commands").mkdir(parents=True)
    (tree / "tensorrt_llm/commands/eval.py").write_text("EASIER = True\n")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", f"merge side{_SIGNED}")
    outcome = integration.integrate("merged")
    assert not outcome.merged and outcome.repairable
    assert "the branch's net change: tensorrt_llm/commands/eval.py is a judge path" in (
        outcome.detail
    )
    assert not (repo / "tensorrt_llm/commands").exists()


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
            # A check takes as long as a benchmark: the repository's lock is free meanwhile,
            # so another run on this checkout is not kept from opening its worktrees.
            with open(repo / ".git" / "mv2flow-worktree.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return shell(command, env, timeout)

    def tree_of(prompt: str) -> Path:
        return Path(re.search(r"Edit only in `([^`]+)`", prompt).group(1))

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
    # With the libraries of the tree that was measured, which here are not the checkout's.
    assert {v["tree"].split("+")[1] for v in _verdicts(tmp_path)[:2]} == {record["libraries"]}
    assert record["libraries"] != contract_mod.library_digest(repo)
    # The libraries it accepted are not the checkout's: kept beside the record, and said.
    kept = workspace / "onboarded_libraries" / "tensorrt_llm" / "libs" / "libtensorrt_llm.so"
    assert kept.read_bytes() == b"rebuilt"
    later = _prompts(calls, "kv_fp8")[0]
    assert "- libraries were changed by `core`: rebuild before you boot" in later
    said = f"kept and verified in `{workspace / 'onboarded_libraries'}` (1 file(s))"
    assert f"The ones the onboard check accepted: {said}." in later
    report = _report(tmp_path)
    assert "are not the onboarded ones" in report and record["libraries"] in report
    assert f"; the accepted ones: {said}" in report
    # Code other models run lands, labelled; a commit on a judge path is refused whole, the
    # task is not done, and the run goes on to its end.
    assert (repo / "shared.py").read_text() == "S = 2\n"
    labels = {(r["task"], r["label"]) for r in load_labels(workspace)}
    assert labels == {("core", "target"), ("kv_fp8", "shared")}
    outside = report.split("## Changed outside the target")[1].split("##")[0]
    assert "shared (`kv_fp8`): kv: shared code: shared.py" in outside
    assert "core: three" not in outside and outside.count("\n- ") == 1
    stands = {t["id"]: t for t in _tasks(tmp_path)}
    assert stands["tep2"]["status"] == "not done" and not stands["tep2"]["commit"]
    assert "tensorrt_llm/evaluate/lm_eval.py is a judge path" in stands["tep2"]["note"]
    assert not (repo / "tensorrt_llm/evaluate").exists()
    assert "an easier judge" not in _git(repo, "log", "--format=%s")
    # One short turn to put the branch right, and no more.
    assert len(_prompts(calls, "tep2")) == 2
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
            # Left uncommitted in the checkout's own target: no check measured it.
            modeling = repo / TARGET_REL / "modeling.py"
            modeling.write_text(modeling.read_text() + "# edited in place\n")
        return READY

    script = Script(assembler=_by_task({"core": core}))
    workflow = _workflow(tmp_path, git_runner=None, shell_runner=shell)
    calls = _install(monkeypatch, workflow, script)
    with workflow:
        assert workflow.run(_spec(task_file)) is True
    # A line of the final pass that is no deployment's check goes to the one fix task too.
    assert [_task_id(p) for _, p in calls] == ["core", "core", "fix-1"]
    goal = _prompts(calls, "fix-1")[0]
    assert f"- git:target-committed at {base[:12]}: uncommitted, so a clone would not have" in goal
    assert "[FAIL] git:target-committed at `" in _report(tmp_path)
    # The target's own tests ran there too; a run that left no report is no pass.
    assert f"- tests at {base[:12]}: INCOMPLETE (pytest exited 0)" in goal
    final = _report(tmp_path).split("## Final pass")[1].split("##")[0]
    assert "[ok] tests:modules at `" in final and "[FAIL] tests at `" in final
    # The target was there already: what the check judges is the head as it is. Its
    # failure is the attempt's result line, word for word, with no turn to "put it right".
    assert shell.ran == ["tp1:boot", "tp1:boot", "tp1:accuracy"]
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "core (attempt 1 of 3): pending ready: [FAIL] tp1:boot: " in journal
    assert _state(tmp_path).onboarded["commit"] == base == _git(repo, "rev-parse", "HEAD")
    assert _tasks(tmp_path)[1]["attempts"] == 2


def test_a_run_that_is_not_onboarded_says_what_to_run_where_and_the_last_commit_that_passed(
    tmp_path, task_file, repo, monkeypatch
):
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", f"base{_SIGNED}")
    base = _git(repo, "rev-parse", "HEAD")
    shell = _checks(fail={"tp1:boot"})
    commits: list[str] = []

    def core(attempt, prompt):
        tree = Path(re.search(r"Edit only in `([^`]+)`", prompt).group(1))
        (tree / FAMILY / "a.py").write_text(f"A = {len(commits)}\n")
        _git(tree, "add", f"{FAMILY}/a.py")
        _git(tree, "commit", "-q", "-m", f"core {len(commits)}{_SIGNED}")
        commits.append(_git(tree, "rev-parse", "HEAD"))
        if len(commits) == 2:
            shell.fail.clear()
            shell.fail.add("tp1:accuracy")  # the second commit boots, and scores too low
        return READY

    script = Script(assembler=_by_task({"core": core}))
    workflow = _workflow(tmp_path, git_runner=None, shell_runner=shell, max_attempts=2)
    _install(monkeypatch, workflow, script)
    with workflow:
        assert workflow.run(_spec(task_file, KV_FP8)) is False
    # The first commit did not boot; the second booted and failed its benchmark, which is
    # what ended the task: the report names that check, not the older failure. Nothing landed.
    assert shell.ran == ["tp1:boot", "tp1:boot", "tp1:accuracy"]
    assert _git(repo, "rev-parse", "HEAD") == base and _state(tmp_path).onboarded == {}
    workspace = tmp_path / "ws"
    report = _report(tmp_path).splitlines()
    assert report[0] == "# not onboarded: stopped at `core`, not done after 2 attempt(s)"
    assert report[2].startswith("- result: [FAIL] tp1:accuracy[gsm8k]: ")
    assert report[3] == (
        f"- reproduce: `{workspace}/bin/trtllm-modeling-v2 gate 'accuracy[gsm8k]' --task "
        f"{workspace}/task.yaml --deployment tp1 --tree {workspace}/worktrees/core`"
    )
    assert report[4] == "- last commit where `tp1:accuracy[gsm8k]` passed: `none`"
    # A later start, whose commit does not boot: the boot last passed on the second commit.
    shell.fail.clear()
    shell.fail.add("tp1:boot")
    workflow = _workflow(tmp_path, git_runner=None, shell_runner=shell, max_attempts=1)
    _install(monkeypatch, workflow, script)
    with workflow:
        assert workflow.run(task_file) is False
    report = _report(tmp_path).splitlines()
    assert report[2].startswith("- result: [FAIL] tp1:boot: ")
    assert report[3].startswith(
        f"- reproduce: `{workspace}/bin/trtllm-modeling-v2 gate boot --task"
    )
    assert report[4] == f"- last commit where `tp1:boot` passed: `{commits[1]}`"


def test_a_process_that_died_after_the_commit_landed_goes_on_without_checking_again(
    tmp_path, task_file, repo, monkeypatch
):
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", f"base{_SIGNED}")

    def core(attempt, prompt):
        tree = Path(re.search(r"Edit only in `([^`]+)`", prompt).group(1))
        (tree / FAMILY / "a.py").write_text("A = 1\n")
        _git(tree, "add", f"{FAMILY}/a.py")
        _git(tree, "commit", "-q", "-m", f"core{_SIGNED}")
        return READY

    shell = _checks()
    kwargs = dict(git_runner=None, shell_runner=shell)
    script = Script(assembler=_by_task({"core": core}))
    _run(tmp_path, _spec(task_file), monkeypatch, script, **kwargs)
    landed = _git(repo, "rev-parse", "HEAD")
    record = _state(tmp_path).onboarded
    assert record["commit"] == landed and shell.ran == ["tp1:boot", "tp1:accuracy"]
    # As the workspace stood had the process died between the fast-forward and its records:
    # the task still running, no onboard record, the worktree already removed.
    workspace = tmp_path / "ws"
    tasks = yaml.safe_load((workspace / "tasks.yaml").read_text())
    tasks["tasks"][1]["status"], tasks["closed"] = "running", False
    (workspace / "tasks.yaml").write_text(yaml.safe_dump(tasks))
    state = load_state(workspace / ".trtllm_modeling_v2_state.json")
    state.onboarded = {}
    save_state(workspace / ".trtllm_modeling_v2_state.json", state)
    calls = _run(tmp_path, task_file, monkeypatch, script, **kwargs)
    # The worker's answer stands, the branch is not landed twice, and what passed before the
    # commit went in is not measured again: the record is written from the ledger.
    assert calls == [] and shell.ran == ["tp1:boot", "tp1:accuracy"]
    assert _git(repo, "rev-parse", "HEAD") == landed
    again = _state(tmp_path).onboarded
    assert (again["commit"], again["accuracy"], again["ref"]) == (landed, "judged", record["ref"])
    assert _tasks(tmp_path)[1]["status"] == "done" and _tasks(tmp_path)[1]["commit"] == landed
    assert _report(tmp_path).startswith(f"# onboarded: commit `{landed}`")


def _recording(tmp_path, refs: dict, events: list | None = None) -> SimpleNamespace:
    """What the onboard record reads of a workflow: a scripted git with refs, the state, a journal."""
    repo, tree = tmp_path / "repo", tmp_path / "tree"
    for root in (repo, tree):
        (root / "tensorrt_llm").mkdir(parents=True, exist_ok=True)
    (tree / "tensorrt_llm" / "libtensorrt_llm.so").write_bytes(b"rebuilt")
    events = [] if events is None else events
    return SimpleNamespace(
        task_data={"trtllm_repo_path": str(repo), "deployments": [{"name": "tp1", "options": {}}]},
        workspace=tmp_path / "ws",
        # The run began on a checkout with no library of its own: the tree's were rebuilt.
        _state=WorkflowState(task_path="t", slug="s", libraries=contract_mod.library_digest(repo)),
        _contract=contract_mod.Contract((contract_mod.Check("tp1", "boot"),)),
        _contract_status=lambda: contract_mod.ContractStatus.load(tmp_path / "ws"),
        _git=lambda repo, *args: scripted_refs(refs, "abc123", args),
        _checkpoint=lambda state: None,
        journal=SimpleNamespace(event=lambda what, detail="": events.append((what, detail))),
    )


def _failing_once(monkeypatch, error: BaseException) -> None:
    """The first write of a state that holds the onboard record fails with ``error``."""
    real, fired = TrtllmModelingV2Workflow._checkpoint, []

    def checkpoint(self, state):
        if state.onboarded and not fired:
            fired.append(True)
            raise error
        real(self, state)

    monkeypatch.setattr(TrtllmModelingV2Workflow, "_checkpoint", checkpoint)


def test_a_record_that_could_not_be_written_is_not_a_task_done(tmp_path, task_file, monkeypatch):
    _failing_once(monkeypatch, OSError(28, "No space left on device"))
    shell = _checks()
    _run(tmp_path, _spec(task_file), monkeypatch, Script(), shell_runner=shell)
    # The attempt ends with why; the next one finds its pass on record and writes the record.
    core = _tasks(tmp_path)[1]
    assert (core["status"], core["attempts"]) == ("done", 2)
    assert shell.ran == ["tp1:boot", "tp1:accuracy"]
    assert "the onboard record could not be written (OSError" in (
        (tmp_path / "ws" / "journal.md").read_text()
    )
    assert _state(tmp_path).onboarded["commit"] == "c0ffee1"


def test_a_start_that_finds_the_ref_and_no_record_takes_the_record_from_it(
    tmp_path, task_file, monkeypatch
):
    git, shell = _Tree(), _checks()
    run = dict(shell_runner=shell, git_runner=git, max_attempts=1)
    # Cut after the ref was written and before the state.
    _failing_once(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, _spec(task_file), monkeypatch, Script(), **run)
    ref = f"refs/mv2/{SLUG}/onboarded"
    assert git.refs == {ref: "c0ffee1"} and not _state(tmp_path).onboarded
    # The head has moved since and no longer boots: the first accepted commit is the record
    # all the same, and the ref is where it was.
    git.at, shell.fail = "bbbbbbb", {"tp1:boot"}
    workflow = _workflow(tmp_path, **run)
    _install(monkeypatch, workflow, Script())
    with workflow:
        assert workflow.run(task_file) is True
    assert _state(tmp_path).onboarded["commit"] == "c0ffee1" and git.refs == {ref: "c0ffee1"}
    assert _report(tmp_path).startswith(f"# onboarded: commit `c0ffee1`, ref `{ref}`")


def test_the_onboard_record_is_written_once_and_found_again(tmp_path):
    refs: dict[str, str] = {}
    workflow, tree, ref = _recording(tmp_path, refs), tmp_path / "tree", "refs/mv2/s/onboarded"
    assert report_mod.record_onboarded(workflow, tree, "abc123") == ""
    record = dict(workflow._state.onboarded)
    assert (record["commit"], record["ref"], refs) == ("abc123", ref, {ref: "abc123"})
    assert record["accuracy"] == "unjudged (boot only)"
    # Nothing later moves either: not another commit that passes, not a new start.
    assert report_mod.record_onboarded(workflow, tree, "def456") == ""
    report_mod.reconcile_onboarded(workflow)
    assert workflow._state.onboarded == record and refs == {ref: "abc123"}
    # A start cut after the ref and before the state: the ref is the record, whatever the
    # head is by then, and it is not moved.
    workflow._state.onboarded = {}
    report_mod.reconcile_onboarded(workflow)
    assert workflow._state.onboarded["commit"] == "abc123"
    workflow._state.onboarded = {}
    assert report_mod.record_onboarded(workflow, tree, "def456") == ""
    assert workflow._state.onboarded["commit"] == "abc123" and refs == {ref: "abc123"}
    # A state write that fails is not onboarded and is said; the ref holds it for the next start.
    workflow._state.onboarded = {}

    def full(state):
        raise OSError(28, "No space left on device")

    workflow._checkpoint = full
    said = report_mod.record_onboarded(workflow, tree, "abc123")
    assert said.startswith("the onboard record could not be written (OSError: [Errno 28]")
    assert workflow._state.onboarded == {} and refs == {ref: "abc123"}
    # A ref git did not write is not claimed, and is written on the next start.
    refs.clear()
    workflow._checkpoint, written = (lambda state: None), workflow._git
    workflow._git = lambda repo, *args: None
    assert report_mod.record_onboarded(workflow, tree, "abc123") == ""
    assert workflow._state.onboarded["ref"] == f"{ref} (git did not write it)"
    workflow._git = written
    # Created, never moved: a ref that is there for another commit stays, and is not claimed.
    refs[ref] = "another"
    report_mod.reconcile_onboarded(workflow)
    assert workflow._state.onboarded["ref"].endswith("(git did not write it)")
    assert refs == {ref: "another"}
    refs.clear()
    report_mod.reconcile_onboarded(workflow)
    assert workflow._state.onboarded["ref"] == ref and refs == {ref: "abc123"}


def test_git_itself_creates_the_onboard_ref_once_and_never_moves_it(tmp_path):
    """The scripted refs above say what git does with an empty old value; this asks git."""
    repo, ref = _checkout(tmp_path), "refs/mv2/s/onboarded"
    workflow, first = SimpleNamespace(_git=git_output), _git(repo, "rev-parse", "HEAD")
    (repo / "shared.py").write_text("S = 2\n")
    _git(repo, "commit", "-q", "-am", "next")
    assert report_mod._pinned(workflow, repo, ref) == ""
    assert report_mod._pin(workflow, repo, ref, first)
    assert not report_mod._pin(workflow, repo, ref, _git(repo, "rev-parse", "HEAD"))
    assert report_mod._pin(workflow, repo, ref, first)
    assert _git(repo, "rev-parse", ref) == first


def test_the_accepted_libraries_are_kept_with_a_manifest_and_the_report_says_which(
    tmp_path, monkeypatch
):
    events: list[tuple[str, str]] = []
    workflow, tree = _recording(tmp_path, {}, events), tmp_path / "tree"
    kept = tmp_path / "ws" / "onboarded_libraries"
    assert report_mod.record_onboarded(workflow, tree, "abc123") == ""
    # Rebuilt since the run's start: copied, and the manifest, written last, says what.
    assert (kept / "tensorrt_llm" / "libtensorrt_llm.so").read_bytes() == b"rebuilt"
    manifest = json.loads((kept / "MANIFEST.json").read_text())
    assert manifest["digest"] == workflow._state.onboarded["libraries"]
    assert report_mod.kept_libraries(workflow) == f"kept and verified in `{kept}` (1 file(s))"
    # A directory that merely exists says nothing: the bytes are held to the manifest.
    (kept / "tensorrt_llm" / "libtensorrt_llm.so").write_bytes(b"built over")
    said = report_mod.kept_libraries(workflow)
    assert said == f"incomplete in `{kept}`: 1 file(s) are not what was copied"
    (kept / "MANIFEST.json").unlink()
    assert "no manifest of a finished copy" in report_mod.kept_libraries(workflow)
    # A copy that fails takes nothing back, and is not said to be there.
    workflow._state.onboarded = {}

    def full(source, target):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(report_mod.shutil, "copy2", full)
    assert report_mod.record_onboarded(workflow, tree, "abc123") == ""
    assert workflow._state.onboarded["commit"] == "abc123"
    assert events[-1][0] == "onboarded" and "were not kept" in events[-1][1]
    assert report_mod.kept_libraries(workflow).startswith(f"incomplete in `{kept}`")
    # A worktree links the checkout's build: rebuilt there, the checkout holds the same
    # bytes, and a later task may build over them. They are kept all the same.
    monkeypatch.undo()
    shutil.rmtree(kept)
    repo_lib = tmp_path / "repo" / "tensorrt_llm" / "libtensorrt_llm.so"
    repo_lib.write_bytes(b"rebuilt")
    workflow._state.onboarded = {}
    assert report_mod.record_onboarded(workflow, tree, "abc123") == ""
    assert report_mod.kept_libraries(workflow).startswith("kept and verified in ")
    repo_lib.unlink()
    # Libraries that are the ones the run started with need no copy, linked or not.
    workflow._state.onboarded = {}
    workflow._state.libraries = contract_mod.library_digest(tree)
    assert report_mod.record_onboarded(workflow, tree, "abc123") == ""
    assert report_mod.kept_libraries(workflow) == (
        "unchanged since the run's start, so none were kept"
    )


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
