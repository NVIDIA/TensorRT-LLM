# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gates — the commands whose exit codes are the verdicts."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import acceptance as acceptance_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import boot as boot_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import check_config, check_runtime
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment

_TASK = {
    "checkpoint_path": "/ckpt/Qwen3-30B-A3B",
    "parallel": "tp1",
    "trtllm_repo_path": "/repo",
    "accuracy": {
        "benchmark": "gsm8k",
        "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
        "tolerance": 5.0,
    },
}


def _env(world_size: int = 1, devices=(0,)):
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/opt/venv/bin/python", "env": {}}},
        devices=devices,
        world_size=world_size,
    )


def _gen(*texts: str, extra: str = "") -> str:
    """What the engine driver prints: one record per prompt, plus any log lines."""
    from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io

    lines = [extra] if extra else []
    for n, text in enumerate(texts):
        record = {"index": n, "text": text, "token_ids": [1, 2], "tokens": 2, "iterations": 2}
        lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
    return "\n".join(lines) + "\n"


_PARIS = _gen("Paris is the capital", "the way plants make food")


def recorder(result: CommandResult | list[CommandResult]):
    """A runner that returns canned results and records the commands it saw."""
    results = result if isinstance(result, list) else [result]
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return results[min(len(seen) - 1, len(results) - 1)]

    run.seen = seen  # type: ignore[attr-defined]
    return run


# --------------------------------------------------------------- topology


@pytest.mark.parametrize(
    "member,ep,attention_dp", [("tp4", 1, False), ("tep4", 4, False), ("dep4", 4, True)]
)
def test_a_member_uses_public_topology_keys(member, ep, attention_dp):
    options = check_config.check_configuration(
        _TASK, _env(4), unit="core", features=[], max_tokens=1, member=member
    )["llm"]
    assert options["tensor_parallel_size"] == 4
    assert options["moe_expert_parallel_size"] == ep
    assert options["enable_attention_dp"] is attention_dp


# ------------------------------------------------------------ score parsing


def test_the_final_score_wins_over_partial_estimates():
    out = (
        "Partial scores after 100/1319 responses (estimate, 0~100): gsm8k,acc ~ 12.00\n"
        "Partial scores after 900/1319 responses (estimate, 0~100): gsm8k,acc ~ 80.00\n"
        "lm-eval gsm8k exact_match,flexible-extract accuracy: 90.67\n"
    )
    assert accuracy_checks.parse_eval_score(out) == pytest.approx(90.67)


# -------------------------------------------------------------- boot gate


def test_boot_gate_passes_on_clean_exit_with_output_and_fails_on_nonzero_exit(tmp_path):
    run = recorder(CommandResult(returncode=0, stdout=_PARIS))
    result = boot_checks.run_boot_gate(_env(), task_data=_TASK, runner=run, workdir=tmp_path)
    assert result.passed, result.detail
    assert "quickstart" not in result.command
    assert str(check_runtime.ENGINE_DRIVER) in result.command
    assert "TRTLLM_MODELING_V2=require" in result.command
    run = recorder(CommandResult(returncode=1, stderr="CUDA error"))
    result = boot_checks.run_boot_gate(_env(), task_data=_TASK, runner=run, workdir=tmp_path)
    assert not result.passed and "exited 1" in result.detail


def test_a_keyword_in_the_log_is_not_a_keyword_in_the_text(tmp_path):
    """`72` matches `CPUs 72-143` in the engine log, and the prompt is echoed too."""
    spec = boot_checks.BootSpec(prompts=("48 + 24 = ? Answer 72 if unsure.",), keywords=("72",))
    log = "[TRT-LLM] [I] Process 1 bound to CPUs 72-143; prompt '48 + 24 = ? Answer 72'"
    common = dict(task_data=_TASK, spec=spec, workdir=tmp_path)
    run = recorder(CommandResult(returncode=0, stdout=_gen("I do not know", extra=log)))
    result = boot_checks.run_boot_gate(_env(), runner=run, **common)
    assert not result.passed and "'72'" in result.detail
    run = recorder(CommandResult(returncode=0, stdout=_gen("The answer is 72.", extra=log)))
    assert boot_checks.run_boot_gate(_env(), runner=run, **common).passed


# ------------------------------------------------------- a lost session

_LOST = "the session watching /spool stopped heart-beating 412s ago; nothing was queued"


def test_a_gate_lost_to_its_session_says_so_rather_than_timed_out(tmp_path):
    lost = CommandResult(returncode=124, timed_out=True, infrastructure=_LOST)
    common = dict(task_data=_TASK, workdir=tmp_path)
    for result in (
        boot_checks.run_boot_gate(_env(), runner=recorder(lost), timeout=1800, **common),
        accuracy_checks.run_accuracy_gate(
            _env(), reference=89.99, runner=recorder(lost), timeout=14400, **common
        ),
        acceptance_checks.run_acceptance_gate(_env(), runner=recorder(lost), **common),
    ):
        assert not result.passed and result.infrastructure == _LOST
        assert "did not run to a verdict" in result.detail and _LOST in result.detail
        assert "timed out" not in result.detail


def test_a_cancelled_gate_has_no_verdict_and_is_not_queued_again_at_once(tmp_path):
    """A gate job cancelled through the spool raises: missing, not FAIL, and no re-run here."""
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import CANCELLED
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import check_runner
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.environment = RunEnvironment(kind="local", python="python3", world_size=1)
    workflow.task_data = dict(_TASK, trtllm_repo_path=str(tmp_path))
    runner = check_runner.CheckRunner(workflow)
    runner._await_session = lambda log: pytest.fail("a cancelled gate waits for no session")
    cancelled = f"{CANCELLED}: cancelled before it started"
    runs = []

    def run():
        runs.append(1)
        return SimpleNamespace(
            name="accuracy[gsm8k]", member="dep4", passed=False, infrastructure=cancelled
        )

    with pytest.raises(check_runner._SessionLost, match="measured nothing"):
        runner._gate_until_verdict(run, None)
    assert runs == [1]


# ----------------------------------------------------------- accuracy gate


def _options_named(command: str) -> dict:
    """The LLM-API options file a ``trtllm-eval`` command line names, loaded."""
    path = shlex.split(command.split("--config=", 1)[1])[0]
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_the_accuracy_gate_hands_require_to_the_engine_and_needs_a_score(tmp_path):
    """Even when the run's environment does not export the switch itself."""
    unit_file = tmp_path / "core.llm_api.yaml"
    unit_file.write_text("max_num_tokens: 8192\n", encoding="utf-8")
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "llm_api_options": str(unit_file)}}
    run = recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 95\n"))
    assert check_runtime.MODELING_V2_ENV not in _env().env
    result = accuracy_checks.run_accuracy_gate(
        _env(), task_data=task, reference=95.0, runner=run, unit="core", member="tp1",
        workdir=tmp_path / "gates",
    )  # fmt: skip
    options = _options_named(run.seen[0])  # type: ignore[attr-defined]
    assert (
        options["modeling_v2"] == check_runtime.MODE_REQUIRE and options["max_num_tokens"] == 8192
    )
    assert result.passed, result.detail
    assert (
        "score validation only" in result.detail and "engine observability UNKNOWN" in result.detail
    )
    assert "eval_shim" not in run.seen[0]
    run = recorder(CommandResult(returncode=0, stdout="done\n"))
    result = accuracy_checks.run_accuracy_gate(
        _env(), task_data=_TASK, reference=89.99, runner=run, workdir=tmp_path
    )
    assert not result.passed and "no score" in result.detail


# --------------------------------------------------------- acceptance gate


def test_no_declared_workload_takes_32_prompts_from_the_accuracy_dataset():
    """Not the boot prompts: their answers end in EOS before any draft is verified."""
    workload = acceptance_checks.resolve_acceptance_workload({"benchmark": "gsm8k"}, "mtp")
    assert workload.prompts == ()
    assert workload.dataset["source"] == "openai/gsm8k"
    assert workload.dataset["num"] == acceptance_checks.DEFAULT_ACCEPTANCE_NUM_PROMPTS == 32
    assert workload.dataset["text_field"] == "question"
    assert workload.max_tokens == acceptance_checks.DEFAULT_ACCEPTANCE_MAX_TOKENS == 256
    assert workload.min_prompts == 32
    assert "openai/gsm8k" in workload.source


def test_the_draft_depth_comes_from_the_options_yaml(tmp_path):
    """All three gates must ask for the depth the unit was built for."""
    options = tmp_path / "mtp_llm_api.yaml"
    speculative = {"decoding_type": "MTP", "num_nextn_predict_layers": 3}
    options.write_text(yaml.safe_dump({"speculative_config": speculative}))
    accuracy = {"llm_api_options": {"mtp": str(options)}}
    assert check_config.resolve_max_draft_len(accuracy, "mtp") == 3
    assert check_config.speculation_flags(["mtp"], max_draft_len=3) == [
        "--spec_decode_algo=MTP",
        "--spec_decode_max_draft_len=3",
    ]


# ---------------------------------------------------------- member ordering


def test_members_are_ordered_cheapest_first():
    assert check_config.order_members_by_cost(["tp4", "dep4", "tep4"]) == ["tep4", "dep4", "tp4"]


@pytest.mark.parametrize(
    "declared,explicit,expected",
    [
        (None, None, 14400),
        (5400, None, 5400),
        (5400, 800, 800),
        (None, 30, 30),
    ],
)
def test_accuracy_deadline_honors_task_and_explicit_override(
    tmp_path, declared, explicit, expected
):
    task = {**_TASK, "accuracy": {**_TASK["accuracy"]}}
    if declared is not None:
        task["accuracy"]["timeout"] = declared
    seen = []

    def run(command, env, timeout):
        seen.append(timeout)
        return CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 95\n")

    result = accuracy_checks.run_accuracy_gate(
        _env(),
        task_data=task,
        reference=95.0,
        runner=run,
        timeout=explicit,
        workdir=tmp_path,
    )
    assert result.passed and seen == [expected]


def test_requested_gsm8k_transcripts_survive_in_unique_gate_output_directories(tmp_path):
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "benchmark_args": ["--log_samples"]}}
    config = {"llm": {"model": "/model", "dtype": "bfloat16"}}
    destinations = []
    for _ in range(2):
        parts = accuracy_checks.eval_command(
            task, config, "gsm8k", unit="core", member="dep4", workdir=tmp_path / "with spaces"
        )
        arguments = shlex.split(" ".join(parts))
        output = next(arg.split("=", 1)[1] for arg in arguments if arg.startswith("--output_path="))
        destinations.append(Path(output))
        assert Path(output).parent == tmp_path / "with spaces"
        assert "--log_samples" in arguments
        assert f"--output_dir={Path(output) / 'inference'}" in arguments
    assert destinations[0] != destinations[1]


@pytest.mark.parametrize("output_argument", ["--output_path=/chosen", "--output_path"])
def test_a_declared_transcript_destination_is_preserved(tmp_path, output_argument):
    arguments = ["--log_samples", output_argument]
    if output_argument == "--output_path":
        arguments.append("/chosen with spaces")
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "benchmark_args": arguments}}
    result = accuracy_checks.eval_command(
        task, {"llm": {"model": "/model"}}, "gsm8k", workdir=tmp_path
    )
    parsed = shlex.split(" ".join(result))
    assert sum(arg.startswith("--output_path") for arg in parsed) == 1
    assert all(arg in parsed for arg in arguments)


def test_an_explicit_inference_token_destination_is_preserved(tmp_path):
    args = ["--log_samples", "--output_dir=/tokens"]
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "benchmark_args": args}}
    parts = accuracy_checks.eval_command(
        task, {"llm": {"model": "/model"}}, "gsm8k", workdir=tmp_path
    )
    parsed = shlex.split(" ".join(parts))
    assert [arg for arg in parsed if arg.startswith("--output_dir=")] == ["--output_dir=/tokens"]
