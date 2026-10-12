# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scheduler-equivalence witness and its optional step-trace hook."""

from __future__ import annotations

import inspect
import json
import shlex
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase_onboard import step_check, workflow
from agent_flow.workflows.staircase_onboard.environment import resolve_environment
from agent_flow.workflows.staircase_onboard.gates import ShellResult
from agent_flow.workflows.staircase_onboard.harness import workload_io
from agent_flow.workflows.staircase_onboard.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

_TASK = {
    "checkpoint_path": "/ckpt/M",
    "parallel": "tep4",
    "trtllm_repo_path": "/repo",
    "features": ["dspark"],
    "runtime": {"overlap_scheduler": True, "extra_flags": ["--max_batch_size=8"]},
    "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml"},
}


def _env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/py", "env": {}}}, devices=(0,), world_size=1
    )


def _records(ids_by_prompt, start=0):
    lines = []
    for n, ids in enumerate(ids_by_prompt):
        record = {
            "index": n,
            "prompt_sha": f"s{n}",
            "token_ids": ids,
            "tokens": len(ids),
            "iterations": len(ids),
            "text": " ".join(map(str, ids)),
        }
        lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
    return "\n".join(lines) + "\n"


def _leg_of(command):
    words = shlex.split(command)
    spec = Path(words[words.index("--spec") + 1])
    env = dict(w.split("=", 1) for w in words if "=" in w and w.split("=", 1)[0].isupper())
    return spec.stem, json.loads(spec.read_text()), env


_GOOD = [list(range(100 + n, 228 + n)) for n in range(3)]


def _engram_bug(ids):
    """The overlap+spec leg reads stale history from its first generation step on."""
    return [t if i < 5 else t + 1 for i, t in enumerate(ids)]


def _runner(outputs, traces=None):
    """``outputs``: leg -> token ids per prompt; ``traces``: leg -> trace records."""
    seen = []

    def run(command, env=None, timeout=0):
        leg, spec, environment = _leg_of(command)
        seen.append((leg, spec, environment))
        trace_dir = environment.get(step_check.STEP_TRACE_ENV)
        if traces and leg in traces and trace_dir:
            Path(trace_dir, "rank0.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in traces[leg]), encoding="utf-8"
            )
        return ShellResult(returncode=0, stdout=_records(outputs[leg]))

    run.seen = seen
    return run


def _run(tmp_path, outputs, *, task=None, traces=None, features=("dspark",), **kwargs):
    run = _runner(outputs, traces)
    result = step_check.run_scheduler_equivalence(
        _env(),
        task_data=task or _TASK,
        runner=run,
        unit="dspark",
        member="tep4",
        features=list(features),
        workdir=tmp_path,
        **kwargs,
    )
    return result, run


# ------------------------------------------------------------------- the legs


def test_identical_legs_pass(tmp_path):
    result, run = _run(tmp_path, {"overlap": _GOOD, "no_overlap": _GOOD, "no_spec": _GOOD})
    assert result.passed, result.detail
    assert [leg for leg, _, _ in run.seen] == ["overlap", "no_overlap", "no_spec"]
    assert "overlap vs no_overlap (exact): 3/3 identical" in result.detail
    assert "step trace: not written by the target" in result.detail


def test_the_engram_class_of_defect_fails_on_the_overlap_pair(tmp_path):
    """Stale state in overlap+speculation generation steps: overlap off disagrees."""
    bad = [_engram_bug(ids) for ids in _GOOD]
    result, _ = _run(tmp_path, {"overlap": bad, "no_overlap": _GOOD, "no_spec": _GOOD})
    assert not result.passed
    assert "the overlap scheduler changed the greedy output at batch 1" in result.detail
    assert "#0 at token 5" in result.detail
    assert result.measured == 5.0
    assert "prompt #0: first differs at token 5" in result.stdout_tail


def test_every_leg_runs_one_request_at_a_time_with_one_tactic_cache(tmp_path):
    _, run = _run(tmp_path, {"overlap": _GOOD, "no_overlap": _GOOD, "no_spec": _GOOD})
    caches = set()
    trace_dirs = set()
    for leg, spec, env in run.seen:
        argv = spec["quickstart_argv"]
        assert argv.count("--max_batch_size=1") == 1 and "--max_batch_size=8" not in argv
        assert spec["concurrency"] == 1 and spec["ignore_eos"] is True
        assert ("--disable_overlap_scheduler" in argv) == (leg == "no_overlap")
        assert ("--spec_decode_algo=DSPARK" in argv) == (leg != "no_spec")
        assert env["TRTLLM_MODELING_V2"] == "require"
        caches.add(env[step_check.AUTOTUNER_CACHE_ENV])
        trace_dirs.add(env[step_check.STEP_TRACE_ENV])
    assert len(caches) == 1 and len(trace_dirs) == 3


def test_a_target_without_a_speculator_runs_two_legs(tmp_path):
    result, run = _run(tmp_path, {"overlap": _GOOD, "no_overlap": _GOOD}, features=())
    assert result.passed
    assert [leg for leg, _, _ in run.seen] == ["overlap", "no_overlap"]


def test_the_spec_pair_is_reported_by_default_and_held_when_declared(tmp_path):
    """A verify step's GEMM shape differs from a plain step's, so a near-tie can flip."""
    flipped = [ids[:40] + [0] + ids[41:] for ids in _GOOD]
    early = [[0] + ids[1:] for ids in _GOOD]
    outputs = {"overlap": _GOOD, "no_overlap": _GOOD}
    result, _ = _run(tmp_path, {**outputs, "no_spec": early})
    assert result.passed and "overlap vs no_spec (report): 0/3 identical" in result.detail
    agreement = step_check.EquivalenceConfig(dataset={"source": "x"}, spec_pair="agreement")
    result, _ = _run(tmp_path, {**outputs, "no_spec": flipped}, config=agreement)
    assert result.passed
    result, _ = _run(tmp_path, {**outputs, "no_spec": early}, config=agreement)
    assert not result.passed and "speculation changed the greedy output" in result.detail
    exact = step_check.EquivalenceConfig(dataset={"source": "x"}, spec_pair="exact")
    result, _ = _run(tmp_path, {**outputs, "no_spec": flipped}, config=exact)
    assert not result.passed


def test_a_short_continuation_fails_as_too_short(tmp_path):
    short = [ids[:20] for ids in _GOOD]
    result, run = _run(tmp_path, {"overlap": short, "no_overlap": short, "no_spec": short})
    assert not result.passed and "fewer than 64 tokens" in result.detail
    assert len(run.seen) == 1


def test_a_leg_that_fails_or_prints_nothing_fails_the_witness(tmp_path):
    def broken(command, env=None, timeout=0):
        leg, _, _ = _leg_of(command)
        if leg == "no_overlap":
            return ShellResult(returncode=1, stderr="boom")
        return ShellResult(returncode=0, stdout=_records(_GOOD))

    result = step_check.run_scheduler_equivalence(
        _env(), task_data=_TASK, runner=broken, features=["dspark"], workdir=tmp_path
    )
    assert not result.passed and "the no_overlap leg exited 1" in result.detail

    result = step_check.run_scheduler_equivalence(
        _env(),
        task_data=_TASK,
        runner=lambda c, e=None, t=0: ShellResult(returncode=0, stdout="Paris\n"),
        features=["dspark"],
        workdir=tmp_path,
    )
    assert not result.passed and "printed no STAIRCASE_RECORD" in result.detail


# ------------------------------------------------------------ the step trace


def _trace(shift=0):
    return [
        {"request": 0, "step": s, "positions": [10 + s + (shift if s > 0 else 0)], "digest": "d"}
        for s in range(4)
    ]


def test_a_step_trace_names_the_first_divergent_step(tmp_path):
    outputs = {"overlap": _GOOD, "no_overlap": _GOOD, "no_spec": _GOOD}
    result, _ = _run(tmp_path, outputs, traces={"overlap": _trace(5), "no_overlap": _trace()})
    assert not result.passed
    assert "step trace diverges at request 0 step 1: positions overlap [16] vs no_overlap [11]" in (
        result.detail
    )
    result, _ = _run(tmp_path, outputs, traces={"overlap": _trace(), "no_overlap": _trace()})
    assert result.passed and "4 step(s) agree" in result.detail


def test_an_extra_overlap_step_is_not_a_divergence():
    a = _trace() + [{"request": 0, "step": 4, "positions": [99]}]
    divergence, compared = step_check.compare_traces(a, _trace())
    assert divergence is None and compared == 4


def test_a_reference_adapter_is_compared_with_the_target(tmp_path):
    def run(command, env=None, timeout=0):
        if "--spec" in command:
            leg, _, environment = _leg_of(command)
            Path(environment[step_check.STEP_TRACE_ENV], "rank0.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in _trace()), encoding="utf-8"
            )
            return ShellResult(returncode=0, stdout=_records(_GOOD))
        words = shlex.split(command)
        target = next(
            w.split("=", 1)[1] for w in words if w.startswith("STAIRCASE_STEP_TRACE_DIR=")
        )
        accepted = next(w for w in words if w.startswith("STAIRCASE_STEP_TRACE_ACCEPTED="))
        assert Path(accepted.split("=", 1)[1], "rank0.jsonl").is_file()
        Path(target, "rank0.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in _trace(1)), encoding="utf-8"
        )
        return ShellResult(returncode=0, stdout="")

    config = step_check.EquivalenceConfig(
        dataset={"source": "x"}, reference_command="/ref/replay.py", legs=("no_overlap",)
    )
    result = step_check.run_scheduler_equivalence(
        _env(), task_data=_TASK, runner=run, features=["dspark"], workdir=tmp_path, config=config
    )
    assert not result.passed
    assert "diverges from the reference at request 0 step 1" in result.detail


# ------------------------------------------------------------ configuration


def test_the_default_workload_is_eight_dataset_questions_at_128_tokens():
    config = step_check.resolve_equivalence_config(_TASK["accuracy"])
    assert config.dataset["source"] == "openai/gsm8k" and config.dataset["num"] == 8
    assert config.max_tokens == 128 and config.ignore_eos
    assert config.legs == ("no_overlap", "no_spec") and config.spec_pair == "report"


def test_units_and_enabled_select_where_it_runs():
    assert step_check.applies({}, "core")
    accuracy = {"scheduler_equivalence": {"units": ["dspark"]}}
    assert step_check.applies(accuracy, "dspark") and not step_check.applies(accuracy, "core")
    assert not step_check.applies({"scheduler_equivalence": {"enabled": False}}, "dspark")


def test_no_workload_is_a_failing_verdict_not_an_exception(tmp_path):
    task = {**_TASK, "accuracy": {"benchmark": "longbench"}}
    result = step_check.run_scheduler_equivalence(
        _env(), task_data=task, runner=_runner({}), features=["dspark"], workdir=tmp_path
    )
    assert not result.passed and "no scheduler-equivalence workload" in result.detail


def _spec(tmp_path, block):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    data = {
        "checkpoint_path": str(ckpt),
        "trtllm_repo_path": str(repo),
        "arch": "sm_100",
        "parallel": "tp1",
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "r.yaml",
            "model_key": "M",
            "scheduler_equivalence": block,
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_schema_accepts_a_full_block(tmp_path):
    block = {
        "units": ["dspark"],
        "num_prompts": 8,
        "max_tokens": 128,
        "ignore_eos": True,
        "legs": ["no_overlap"],
        "spec_pair": "agreement",
        "agreement_tokens": 24,
        "step_trace": {"reference_command": "/ref/replay.py"},
    }
    load_and_validate_task_yaml(_spec(tmp_path, block))


@pytest.mark.parametrize(
    "block,message",
    [
        ({"max_tokens": 32}, "at least|>= 64"),
        ({"legs": ["sideways"]}, "legs"),
        ({"spec_pair": "close"}, "spec_pair"),
        ({"step_trace": {"cmd": "x"}}, "cmd"),
        ({"enabled": "yes"}, "enabled"),
        ({"bogus": 1}, "bogus"),
    ],
)
def test_the_schema_refuses_a_malformed_block(tmp_path, block, message):
    with pytest.raises(TaskSchemaError, match=message):
        load_and_validate_task_yaml(_spec(tmp_path, block))


# ----------------------------------------------------------------- wiring


def test_the_unit_loop_runs_it_over_every_member_before_acceptance():
    source = inspect.getsource(workflow.StaircaseOnboardWorkflow._assemble_one_unit)
    assert "step_check.applies(" in source
    block = source[source.index("run_scheduler_equivalence") - 400 :][:1200]
    assert "_gate_over_members" in block and "features=self._features_through(unit)" in block
    assert source.index("run_scheduler_equivalence") < source.index("run_acceptance_gate")
