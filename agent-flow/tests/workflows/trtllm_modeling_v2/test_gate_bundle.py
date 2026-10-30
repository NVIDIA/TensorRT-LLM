# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate bundles: one engine for a unit's checks at a member, one verdict per check.

The checks of a unit at a member that are ready together run their workloads
on one engine of one snapshot commit, and each is still judged by its own
gate over its own section of the job's output. These tests hold the bundle to
grouping only gates that build the same engine, to replaying each section to
its own gate (so a bundled check fails on what its own job would fail on), to
re-running singly whatever a failure cut short, to taking one admission slot,
and to naming the bundle in each verdict. Engine output is canned or comes
from a fake ``tensorrt_llm`` package run as a real process; no GPU is used.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import bundle, gates, logit_agreement
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import bundle_driver, workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CheckAdmission, CommandResult
from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run import snapshots as snapshots_mod

from .test_engine_attestation import _FAKE_TRTLLM, _engine_config, _engine_line, _task
from .test_engine_reuse import _FAKE_LLM, _FAKE_QUICKSTART
from .test_workflow import _git, _workflow

BOOT = "core:boot@tp1"
LOGITS = f"core:{logit_agreement.GATE_NAME}@tp1"
ACCURACY = "core:accuracy[gsm8k]@tp1"

#: A ``trtllm-eval`` that sends twelve few-shot prompts and prints ``FAKE_SCORE``, or
#: exits ``FAKE_EVAL_EXIT`` when that is set.
_FAKE_EVAL = """
    import os
    import sys

    import yaml

    from tensorrt_llm import LLM as PyTorchLLM

    SHOTS = "Question: a farmer sows three fields with wheat. Answer: 3. " * 14


    def main():
        options = {}
        for word in sys.argv[1:]:
            if word.startswith("--extra_llm_api_options="):
                with open(word.split("=", 1)[1]) as handle:
                    options = yaml.safe_load(handle) or {}
        llm = PyTorchLLM(
            model="m", kv_cache_config=options.get("kv_cache_config"),
            max_batch_size=options.get("max_batch_size", 2048),
            enable_chunked_prefill=options.get("enable_chunked_prefill", False),
            modeling_v2=options.get("modeling_v2", "off"))
        handles = [llm.generate_async(SHOTS + f"Question {n}: how many?") for n in range(12)]
        for handle in handles:
            handle.result()
        llm.shutdown()
        if os.environ.get("FAKE_EVAL_EXIT"):
            print("the evaluation failed")
            sys.exit(int(os.environ["FAKE_EVAL_EXIT"]))
        print(f"lm-eval gsm8k average accuracy: {os.environ.get('FAKE_SCORE', '90.00')}")
        sys.exit(0)
"""

#: Appended to the fake quickstart: an engine that will not start under ``FAKE_BUILD_FAIL``.
_REFUSING = """
    import os

    _setup = setup_llm


    def setup_llm(args):
        if os.environ.get("FAKE_BUILD_FAIL"):
            raise RuntimeError("the engine refused to start")
        return _setup(args)
"""


def _env(python: str = "/py", **env) -> RunEnvironment:
    return RunEnvironment(kind="local", python=python, world_size=1, env=dict(env))


def _aligned(tmp_path: Path, **accuracy) -> dict:
    """A unit whose accuracy flags state the limits its runtime block states, as an aligned spec does."""
    task = _task(tmp_path)
    task["accuracy"].update(
        {"max_seq_len": 9216, "max_batch_size": 128, "floor": {"rule": "tolerance"}},
        tolerance=5.0,
    )
    task["accuracy"].update(accuracy)
    return task


def _boot(env, task, workdir, **kwargs):
    return bundle.capture(
        BOOT,
        lambda runner: gates.run_boot_gate(
            env, task_data=task, runner=runner, unit="core", member="tp1", workdir=workdir, **kwargs
        ),
    )


def _accuracy(env, task, workdir):
    return bundle.capture(
        ACCURACY,
        lambda runner: gates.run_accuracy_gate(
            env, task_data=task, reference=90.0, runner=runner, unit="core", member="tp1",
            workdir=workdir,
        ),
    )  # fmt: skip


def _logits(env, task, workdir, *, features=(), max_tokens=16):
    """The logit gate's command, as it writes it before comparing anything."""
    argv = gates.quickstart_words(
        task, env, unit="core", features=list(features), max_tokens=max_tokens, member="tp1"
    )
    spec = logit_agreement.write_driver_spec(
        workdir / "logits.core.tp1.json", task_data=task, argv=argv, requests=[],
        max_tokens=max_tokens, top_k=5, out=workdir / "logits.core.tp1.engine.json",
    )  # fmt: skip
    moded = gates.with_kv_cache_config(gates._with_mode(env, gates.MODE_REQUIRE), task, "core")
    return bundle.parse_launch(LOGITS, logit_agreement.driver_command(moded, spec), 1800)


# ------------------------------------------------------------------ grouping


def test_the_gates_of_one_member_that_build_one_engine_form_one_group(tmp_path):
    task, env = _aligned(tmp_path), _env()
    work = tmp_path / "gates"
    boot, accuracy, logits = (
        _boot(env, task, work),
        _accuracy(env, task, work),
        _logits(env, task, work),
    )
    assert boot.kind == bundle.KIND_DRIVER and logits.kind == bundle.KIND_LOGITS
    assert accuracy.kind == bundle.KIND_EVAL and accuracy.argv[0] == "trtllm-eval"
    # --max_tokens sets the sampling budget only: the two build one engine.
    assert "--max_tokens=32" in boot.argv and "--max_tokens=16" in logits.argv
    assert boot.engine == logits.engine
    groups, alone = bundle.form_groups([boot, accuracy, logits])
    assert alone == {} and len(groups) == 1
    [group] = groups
    assert group.keys == [BOOT, LOGITS, ACCURACY], "an evaluation runs last"
    assert group.timeout == gates.BOOT_TIMEOUT_S * 2 + gates.ACCURACY_TIMEOUT_S
    spec = json.loads(bundle.write_spec(tmp_path / "b.json", group, {LOGITS: "/o"}).read_text())
    assert spec["quickstart_argv"] == list(boot.argv)
    assert spec["build_timeout"] == gates.BOOT_TIMEOUT_S
    assert [s["kind"] for s in spec["sections"]] == ["driver", "logits", "eval"]
    assert spec["sections"][1]["out"] == "/o" and spec["sections"][2]["argv"] == list(accuracy.argv)
    command = bundle.bundle_command(group, tmp_path / "b.json")
    assert command.startswith(boot.launch) and str(bundle.BUNDLE_DRIVER) in command
    assert "engine_driver.py" not in command and "trtllm-eval" not in command


def test_a_speculator_on_or_off_is_another_engine(tmp_path):
    task, env = _aligned(tmp_path), _env()
    work = tmp_path / "gates"
    speculating = _boot(env, task, work, features=["mtp"])
    plain = _logits(env, task, work)
    assert "--spec_decode_algo=MTP" in speculating.argv
    groups, _ = bundle.form_groups([speculating, plain])
    assert sorted(len(g.launches) for g in groups) == [1, 1]


@pytest.mark.parametrize(
    "change,named",
    [
        ({"max_batch_size": 32}, "max_batch_size: quickstart 128, trtllm-eval 32"),
        ({"max_seq_len": None}, "max_seq_len: quickstart 9216, trtllm-eval None"),
        ({"options": {"attention_dp_config": {"enable_balance": True}}}, "`attention_dp_config`"),
        (
            {"options": {"moe_config": {"backend": "CUTLASS", "max_num_tokens": 8}}},
            "moe_config.max",
        ),
        ({"environment": {"CUDA_LAUNCH_BLOCKING": "1"}}, "their launches differ"),
    ],
)
def test_an_evaluation_shares_the_engine_only_when_every_setting_agrees(tmp_path, change, named):
    task, change = _aligned(tmp_path), dict(change)
    if "options" in change:
        path = Path(task["accuracy"]["llm_api_options"]["core"])
        options = yaml.safe_load(path.read_text())
        for key, value in change.pop("options").items():
            options[key] = {**options.get(key, {}), **value}
        path.write_text(yaml.safe_dump(options))
    extra = change.pop("environment", {})
    task["accuracy"].update(change)
    work = tmp_path / "gates"
    boot = _boot(_env(), task, work)
    accuracy = _accuracy(_env(**extra), task, work)
    groups, alone = bundle.form_groups([boot, accuracy])
    assert [g.keys for g in groups] == [[BOOT], [ACCURACY]]
    assert named in alone[ACCURACY], alone[ACCURACY]


def test_a_setting_only_the_options_file_states_reaches_the_quickstart_and_the_engine_is_shared(
    tmp_path,
):
    """The KV dtype in the unit's file is handed to the boot as its flag: one engine, not two."""
    task = _aligned(tmp_path)
    path = Path(task["accuracy"]["llm_api_options"]["core"])
    options = yaml.safe_load(path.read_text())
    options["kv_cache_config"] = {**options.get("kv_cache_config", {}), "dtype": "fp8"}
    path.write_text(yaml.safe_dump(options))
    work = tmp_path / "gates"
    boot = _boot(_env(), task, work)
    assert "--kv_cache_dtype=fp8" in boot.argv
    groups, _ = bundle.form_groups([boot, _accuracy(_env(), task, work)])
    assert [g.keys for g in groups] == [[BOOT, ACCURACY]]


def test_a_gate_that_launches_nothing_or_something_else_is_not_captured(tmp_path):
    refused = gates.GateResult(name="boot", member="tp1", passed=False, detail="blocked")
    assert bundle.capture(BOOT, lambda runner: refused) is refused
    assert bundle.capture(BOOT, lambda runner: runner("/py serve.py", None, 10)) is None


# ------------------------------------------------------------- the job's output


def _section(key: str, body: str = "", **end) -> str:
    end = {"ok": True, **end}
    return (
        f"{bundle.SECTION_MARKER} {json.dumps({'key': key, 'begin': True})}\n{body}"
        f"{bundle.SECTION_MARKER} {json.dumps({'key': key, 'end': True, **end})}\n"
    )


_BUILT = "[mv2flow-resolve] pid=1 rank=0 module=tensorrt_llm origin=/t/tensorrt_llm/__init__.py\n"


def _group(tmp_path) -> bundle.Group:
    task, env = _aligned(tmp_path), _env()
    work = tmp_path / "gates"
    launches = [_boot(env, task, work), _logits(env, task, work), _accuracy(env, task, work)]
    [group], _ = bundle.form_groups(launches)
    return group


def test_each_completed_section_is_replayed_with_the_build_output_and_its_own(tmp_path):
    group = _group(tmp_path)
    built = _BUILT + _engine_line()
    out = CommandResult(
        returncode=0,
        stdout=built + _section(BOOT, "boot says Paris\n") + _section(LOGITS, "logits\n")
        + _section(ACCURACY, "lm-eval gsm8k average accuracy: 90.0\n"),
        stderr="a warning\n",
    )  # fmt: skip
    sections = bundle.split(out, group)
    assert all(s.result is not None for s in sections.values()), sections
    boot = sections[BOOT].result
    assert boot.stdout == built + "boot says Paris\n" and boot.stderr == "a warning\n"
    assert "lm-eval" not in boot.stdout and "boot says" not in sections[ACCURACY].result.stdout


def test_a_section_that_failed_ends_the_bundle_there(tmp_path):
    group = _group(tmp_path)
    out = CommandResult(
        returncode=1,
        stdout=_BUILT + _engine_line() + _section(BOOT)
        + _section(LOGITS, "Traceback\n", ok=False, error="RuntimeError: CUDA error"),
    )  # fmt: skip
    sections = bundle.split(out, group)
    assert sections[BOOT].result is not None
    assert sections[LOGITS].result is None and "RuntimeError: CUDA error" in sections[LOGITS].why
    assert sections[ACCURACY].why == f"an earlier section ({LOGITS}) ended the bundle"


def test_a_bundle_cut_short_inside_a_section_or_before_one(tmp_path):
    group = _group(tmp_path)
    begun = f"{bundle.SECTION_MARKER} {json.dumps({'key': LOGITS, 'begin': True})}\n"
    out = CommandResult(
        returncode=124, stdout=_engine_line() + _section(BOOT) + begun, timed_out=True
    )
    sections = bundle.split(out, group)
    assert sections[BOOT].result is not None
    assert sections[LOGITS].why == "the bundle ended inside its section (timed out)"
    assert "earlier section" in sections[ACCURACY].why
    timed = _section(LOGITS, ok=False, timed_out=True)
    late = bundle.split(CommandResult(returncode=124, stdout=_section(BOOT) + timed), group)
    assert late[LOGITS].why == "its section ran past its own time in the bundle"


def test_an_engine_that_was_not_built_fails_each_quickstart_gate_as_its_own_job_would(tmp_path):
    group = _group(tmp_path)
    failed = f"Traceback ...\n{bundle.BUILD_MARKER} {json.dumps({'error': 'RuntimeError: no'})}\n"
    sections = bundle.split(CommandResult(returncode=1, stdout=failed), group)
    for key in (BOOT, LOGITS):
        assert sections[key].result.returncode == 1 and "Traceback" in sections[key].result.stdout
    assert sections[ACCURACY].result is None and "builds its own" in sections[ACCURACY].why
    slow = f"{bundle.BUILD_MARKER} {json.dumps({'timed_out': True})}\n"
    late = bundle.split(CommandResult(returncode=124, stdout=slow, timed_out=True), group)
    assert all(s.result is None for s in late.values())
    assert late[BOOT].why == "the bundle's engine took longer to build than its gates allow"


def test_a_lost_or_silent_bundle_runs_every_check_singly(tmp_path):
    group = _group(tmp_path)
    lost = CommandResult(returncode=1, timed_out=True, infrastructure="session requeued")
    assert {s.why for s in bundle.split(lost, group).values()} == {
        "the bundle did not run to an end: session requeued"
    }
    silent = bundle.split(CommandResult(returncode=0, stdout="hello\n"), group)
    assert all(s.result is None for s in silent.values())
    assert silent[BOOT].why == "the bundle ended before its section (exited 0)"


def test_an_evaluation_keeps_its_section_only_on_the_engine_its_options_declare(tmp_path):
    group = _group(tmp_path)
    other = _engine_config(cuda_graph={"batch_sizes": [1, 2], "enable_padding": False})
    out = CommandResult(
        returncode=0,
        stdout=_engine_line(other) + _section(BOOT) + _section(LOGITS) + _section(ACCURACY),
    )
    sections = bundle.split(out, group)
    assert sections[BOOT].result is not None, "the boot gate attests its own engine"
    assert sections[ACCURACY].result is None
    assert "did not run the declared configuration -- cuda_graph" in sections[ACCURACY].why


def test_a_replayed_gate_runs_any_command_but_its_own_as_before(tmp_path):
    group = _group(tmp_path)
    launch = group.launch_of(LOGITS)
    produced = tmp_path / "bundle.logits.json"
    produced.write_text('{"records": []}')
    seen, heard = [], []
    section = CommandResult(returncode=0, stdout="mine\n")
    runner = bundle.replaying(
        launch, section, lambda c, e, t: seen.append(c) or CommandResult(0), heard.append,
        str(produced),
    )  # fmt: skip
    assert runner(launch.command, None, 10) is section and seen == []
    assert Path(launch.out).read_text() == '{"records": []}' and heard == ["mine\n\n"]
    runner("the =off leg", None, 10)
    assert seen == ["the =off leg"]


def test_a_verdict_measured_in_a_bundle_names_it_and_its_verdict_records_it(tmp_path):
    result = gates.GateResult(name="boot", member="tp1", passed=True, detail="ok", marker="m1")
    marked = bundle.mark(result, "b1: core:boot@tp1 + core:accuracy[gsm8k]@tp1")
    assert marked.summary().endswith("[bundle b1: core:boot@tp1 + core:accuracy[gsm8k]@tp1]")
    assert marked.passed and marked.marker == "m1"
    status = contract_mod.ContractStatus.load(tmp_path)
    status.record("core", marked)
    status.record("core", result)
    lines = [json.loads(line) for line in status.path.read_text().splitlines()]
    assert lines[0]["bundle"].startswith("b1: ") and "bundle" not in lines[1]
    loaded = contract_mod.ContractStatus.load(tmp_path).verdicts
    assert loaded[0].bundle == marked.bundle and loaded[1].bundle == ""


# ------------------------------------------------------------- the formation


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_a_bundle_takes_members_until_its_leader_seals_it(tmp_path):
    clock = _Clock()
    board = bundle.Board(gather_s=2.0, clock=clock, sleep=clock.sleep)
    leader = board.join(("core", "tp1"), "boot")
    follower = board.join(("core", "tp1"), "accuracy")
    elsewhere = board.join(("core", "tp2"), "boot")
    assert leader.leader and not follower.leader and elsewhere.leader
    clock.now = 0.5
    late = board.join(("core", "tp1"), "logits")
    late.leave()  # skipped before the seal: not a member
    assert [s.check for s in leader.seal()] == ["boot", "accuracy"] and clock.now == 2.0
    after = board.join(("core", "tp1"), "acceptance")
    assert after.leader, "a sealed bundle takes no one; the next arrival leads a new one"


def test_a_leader_that_never_runs_sends_its_members_to_run_singly():
    board = bundle.Board(gather_s=0.0)
    leader = board.join(("core", "tp1"), "boot")
    follower = board.join(("core", "tp1"), "accuracy")
    leader.leave()
    assignment = follower.wait()
    assert assignment.role == bundle.SINGLE and "did not run" in assignment.why
    assert board.join(("core", "tp1"), "logits").leader


# ------------------------------------------------------------ the bundle driver


def _fake_site(tmp_path: Path) -> tuple[Path, Path]:
    """``(site, repo)``: a fake ``tensorrt_llm`` with a ``trtllm-eval``, and a tree with its quickstart."""
    site = tmp_path / "site"
    files = {
        **_FAKE_TRTLLM,
        "tensorrt_llm/__init__.py": _FAKE_LLM,
        "tensorrt_llm/commands/__init__.py": "",
        "tensorrt_llm/commands/eval.py": _FAKE_EVAL,
    }
    for name, text in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    repo = tmp_path / "repo"
    script = repo / "examples" / "llm-api" / "quickstart_advanced.py"
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(_FAKE_QUICKSTART) + textwrap.dedent(_REFUSING))
    return site, repo


def _run(command: str, env=None, timeout: int = 1800) -> CommandResult:
    done = subprocess.run(
        command, shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=120
    )
    return CommandResult(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)


def _bundled(tmp_path: Path, **env) -> tuple[dict, dict, CommandResult]:
    """Boot and accuracy run as one bundle job through the real bundle driver, and replayed."""
    site, repo = _fake_site(tmp_path)
    task = {**_aligned(tmp_path), "trtllm_repo_path": str(repo)}
    environment = _env(sys.executable, PYTHONPATH=str(site), **env)
    work = tmp_path / "gates"
    calls = {
        BOOT: lambda runner: gates.run_boot_gate(
            environment, task_data=task, runner=runner, unit="core", member="tp1", workdir=work
        ),
        ACCURACY: lambda runner: gates.run_accuracy_gate(
            environment, task_data=task, reference=90.0, runner=runner, unit="core",
            member="tp1", workdir=work,
        ),
    }  # fmt: skip
    [group], alone = bundle.form_groups([bundle.capture(k, call) for k, call in calls.items()])
    assert alone == {} and group.keys == [BOOT, ACCURACY]
    output = _run(bundle.bundle_command(group, bundle.write_spec(work / "b.json", group, {})))
    sections = bundle.split(output, group)
    singles: list[str] = []

    def single(command, env=None, timeout=1800):
        singles.append(command)
        return _run(command)

    results = {}
    for key, call in calls.items():
        section = sections[key]
        runner = (
            bundle.replaying(group.launch_of(key), section.result, single)
            if section.result is not None
            else single
        )
        results[key] = call(runner)
    return results, {"sections": sections, "singles": singles}, output


def test_the_bundle_driver_runs_every_section_on_one_engine_and_each_gate_judges_its_own(tmp_path):
    results, seen, output = _bundled(tmp_path)
    assert output.returncode == 0, output.stderr
    assert output.stdout.count(f"{workload_io.ENGINE_MARKER} ") == 1, "one engine was built"
    assert seen["singles"] == [], "every check was judged from its section"
    boot, accuracy = results[BOOT], results[ACCURACY]
    assert boot.passed, f"{boot.detail}\n{boot.stdout_tail}"
    assert accuracy.passed and accuracy.measured == 90.0, accuracy.detail
    # One engine record attests both: the same configuration, read once.
    assert boot.engine[0]["attested"] == accuracy.engine[0]["attested"]
    # Each gate read its own records: the evaluation's reuse is its twelve prompts, the
    # boot's its two prompts and the probe.
    assert accuracy.engine[0]["reuse"]["requests"] == 12
    assert boot.engine[0]["reuse"]["requests"] == 2 and boot.engine[0]["reuse"]["probe"]


def test_a_bundled_accuracy_check_still_fails_on_a_low_score(tmp_path):
    results, seen, _ = _bundled(tmp_path, FAKE_SCORE="10.00")
    assert results[BOOT].passed and seen["singles"] == []
    assert not results[ACCURACY].passed
    assert results[ACCURACY].detail.startswith("measured 10.0000 < 85.0000")


def test_an_evaluation_that_fails_in_the_bundle_runs_singly_and_the_boot_verdict_stands(tmp_path):
    results, seen, output = _bundled(tmp_path, FAKE_EVAL_EXIT="3")
    assert output.returncode == 1
    assert "SectionFailed: trtllm-eval exited 3" in seen["sections"][ACCURACY].why
    assert results[BOOT].passed, "the section that completed keeps its verdict"
    [single] = seen["singles"]
    assert str(gates.EVAL_SHIM) in single, "the evaluation ran again, alone"
    assert results[ACCURACY].detail == "trtllm-eval exited 3"


def test_an_engine_that_cannot_be_built_fails_the_boot_check_from_the_bundle(tmp_path):
    results, seen, output = _bundled(tmp_path, FAKE_BUILD_FAIL="1")
    assert output.returncode == 1 and "the engine refused to start" in output.stderr
    assert not results[BOOT].passed and results[BOOT].detail == "exited 1"
    assert [str(gates.EVAL_SHIM) in c for c in seen["singles"]] == [True]


def test_a_section_past_its_time_ends_the_bundle_after_saying_so(capsys):
    ended: list[int] = []
    record = {"key": BOOT, "end": True, "ok": False, "timed_out": True}
    with bundle_driver.Deadline(0.05, bundle_driver.SECTION_MARKER, record, ended.append):
        time.sleep(0.3)
    assert ended == [bundle_driver.TIMED_OUT_EXIT]
    [said] = workload_io.parse_marked(capsys.readouterr().out, bundle_driver.SECTION_MARKER)
    assert said == record
    with bundle_driver.Deadline(None, bundle_driver.SECTION_MARKER, record, ended.append):
        pass
    assert ended == [bundle_driver.TIMED_OUT_EXIT]


def test_a_logits_section_writes_where_the_bundle_says_and_a_failed_one_says_why(tmp_path, capsys):
    from .test_logit_agreement import _FakeLLM

    base = SimpleNamespace(max_tokens=32, ignore_eos=False, prompt_logprobs=None, logprobs=None)
    spec = tmp_path / "logits.core.tp1.json"
    spec.write_text(
        json.dumps(
            {
                "requests": [{"id": "a", "token_ids": [1, 2]}],
                "max_tokens": 2,
                "top_k": 3,
                "out": str(tmp_path / "gate.json"),
            }
        )
    )
    section = {"key": LOGITS, "kind": "logits", "spec": str(spec), "out": str(tmp_path / "b.json")}
    assert bundle_driver.run_section(_FakeLLM(), base, section)
    assert json.loads((tmp_path / "b.json").read_text())["records"][0]["id"] == "a"
    assert not (tmp_path / "gate.json").exists(), "the gate's copy is the replay's to make"
    broken = {"key": BOOT, "kind": "driver", "spec": str(tmp_path / "missing.json")}
    assert not bundle_driver.run_section(_FakeLLM(), base, broken)
    said = workload_io.parse_marked(capsys.readouterr().out, bundle_driver.SECTION_MARKER)
    assert [(m["key"], m.get("ok")) for m in said] == [
        (LOGITS, None), (LOGITS, True), (BOOT, None), (BOOT, False),
    ]  # fmt: skip
    assert said[-1]["error"].startswith("FileNotFoundError")


def test_both_sides_spell_the_markers_alike():
    assert bundle.SECTION_MARKER == bundle_driver.SECTION_MARKER
    assert bundle.BUILD_MARKER == bundle_driver.BUILD_MARKER


# ------------------------------------------------------- the planned stage


def _committed_tree(tmp_path: Path) -> Path:
    """A git checkout holding the fake package, its ``trtllm-eval`` and the quickstart."""
    site, repo = _fake_site(tmp_path)
    for path in site.rglob("*.py"):
        target = repo / path.relative_to(site)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text())
    _git(repo, "init", "-q", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "a tree to measure")
    return repo


class _Counting(CheckAdmission):
    """An admission that counts the slots it gave, and how many were held while each command ran."""

    def __init__(self):
        super().__init__(lambda environment, gpus, wall: 8, floor=8)
        self.held: list[tuple[str, int]] = []
        self.slots = 0
        self._counting = threading.Lock()

    @contextlib.contextmanager
    def slot(self, *args, **kwargs):
        with self._counting:
            self.slots += 1
        with super().slot(*args, **kwargs):
            yield


def _stage(tmp_path: Path, **env):
    repo = _committed_tree(tmp_path)
    admission = _Counting()

    def shell(command, env=None, timeout=1800):
        kind = "bundle" if str(bundle.BUNDLE_DRIVER) in command else "single"
        admission.held.append((kind, admission.in_flight))
        return _run(command)

    workflow = _workflow(tmp_path, shell_runner=shell, git_runner=None)
    workflow.task_data = {**_aligned(tmp_path), "trtllm_repo_path": str(repo)}
    workflow.environment = _env(sys.executable, **env)
    workflow._snapshots = snapshots_mod.Snapshots(repo, workflow.workspace / "snapshots")
    workflow._reference_score = lambda unit=None, spec=None: 90.0
    workflow._accuracy_by_unit = {}
    workflow.__dict__["_check_board"] = bundle.Board(gather_s=0.5)
    return workflow, admission, repo


def _check_concurrently(workflow, admission, gates_: list[str]) -> dict:
    results: dict = {}
    checks = [contract_mod.Check("core", gate, "tp1", contract_mod.TIER_INNER) for gate in gates_]

    def one(check):
        with workflow._admit_check(admission, check):
            results[check.key] = workflow._run_contract_check(check, None)

    threads = []
    for check in checks:  # the first submitted leads
        threads.append(threading.Thread(target=one, args=(check,)))
        threads[-1].start()
        time.sleep(0.05)
    for thread in threads:
        thread.join(timeout=120)
    return results


def test_ready_checks_of_a_member_run_as_one_bundle_holding_one_slot(tmp_path):
    workflow, admission, repo = _stage(tmp_path)
    head = _git(repo, "rev-parse", "HEAD").strip()
    results = _check_concurrently(workflow, admission, ["boot", "accuracy[gsm8k]"])
    assert [kind for kind, _ in admission.held] == ["bundle"]
    assert admission.held[0][1] == 1 and admission.slots == 1, "one slot for the two checks"
    boot, accuracy = results[BOOT], results[ACCURACY]
    assert boot.passed and accuracy.passed, (boot.summary(), accuracy.summary())
    assert boot.bundle == accuracy.bundle and boot.bundle.endswith(f"{BOOT} + {ACCURACY}")
    marker = workflow._commit_markers(head)[0]
    for result in (boot, accuracy):
        assert result.marker == marker and result.imported == (head,) and not result.void
    status = contract_mod.ContractStatus.load(workflow.workspace)
    verdict = status.record("core", boot, marker="the target when it was queued")
    assert verdict.bundle == boot.bundle and verdict.marker == marker
    journal = workflow.journal_path.read_text()
    assert "2 judged from it" in journal and f"[bundle {boot.bundle}]" in journal


def test_a_crash_mid_bundle_keeps_completed_verdicts_and_reruns_the_rest_singly(tmp_path):
    workflow, admission, _ = _stage(tmp_path, FAKE_EVAL_EXIT="3")
    results = _check_concurrently(workflow, admission, ["boot", "accuracy[gsm8k]"])
    assert [kind for kind, _ in admission.held] == ["bundle", "single"]
    assert admission.held[0][1] == 1 and admission.slots == 2, "the single run took its own slot"
    assert results[BOOT].passed and results[BOOT].bundle
    accuracy = results[ACCURACY]
    assert not accuracy.passed and accuracy.detail == "trtllm-eval exited 3"
    assert not getattr(accuracy, "bundle", ""), "it was measured alone"
    assert f"{ACCURACY} runs singly: its section failed in the bundle" in (
        workflow.journal_path.read_text()
    )


def test_a_check_no_bundle_takes_is_admitted_as_before(tmp_path):
    workflow, admission, _ = _stage(tmp_path)
    witness = contract_mod.Check("core", "equivalence", "tp1", contract_mod.TIER_PROOF)
    with workflow._admit_check(admission, witness):
        assert admission.in_flight == 1 and workflow._seat_of(witness) is None
    workflow._snapshots = None
    boot = contract_mod.Check("core", "boot", "tp1", contract_mod.TIER_INNER)
    with workflow._admit_check(admission, boot):
        assert admission.in_flight == 1 and workflow._seat_of(boot) is None
