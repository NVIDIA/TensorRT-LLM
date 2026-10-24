# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gates — the commands whose exit codes are the verdicts."""

from __future__ import annotations

import itertools
import json
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.infra import vcs
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult

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
    "segment,expected",
    [
        ("tp1", ["--tp_size=1"]),
        ("tp8", ["--tp_size=8"]),
        ("tep4", ["--tp_size=4", "--moe_ep_size=4"]),
        ("dep4", ["--tp_size=4", "--moe_ep_size=4", "--enable_attention_dp"]),
    ],
)
def test_topology_flags_follow_the_segment(segment, expected):
    """Every gate is topology-blind; deriving the flags is what keeps the name honest."""
    assert gates.topology_flags(segment, world_size=1) == expected


def test_unified_takes_its_size_from_the_world_size():
    """With no member, ``unified`` yields only ``--tp_size``.

    The documented fallback when no member is passed. The orchestrator always
    passes one; without it a unified target would be gated at pure tensor
    parallelism while claiming three configurations.
    """
    assert gates.topology_flags("unified", world_size=4) == ["--tp_size=4"]


@pytest.mark.parametrize(
    "member,expected",
    [
        ("tp4", ["--tp_size=4"]),
        ("tep4", ["--tp_size=4", "--moe_ep_size=4"]),
        ("dep4", ["--tp_size=4", "--moe_ep_size=4", "--enable_attention_dp"]),
    ],
)
def test_a_member_is_what_gets_translated(member, expected):
    """`unified` names a family; the member names the configuration served.

    Without this, `--moe_ep_size` and `--enable_attention_dp` are emitted by
    no gate for the one segment whose entire purpose is to carry both
    branches.
    """
    assert gates.topology_flags("unified", world_size=4, member=member) == expected


def test_the_member_overrides_the_declared_segment():
    """It is the configuration actually being served, not a hint."""
    assert gates.topology_flags("tp8", world_size=8, member="dep4") == [
        "--tp_size=4",
        "--moe_ep_size=4",
        "--enable_attention_dp",
    ]


def test_the_member_reaches_the_verdict():
    """A gate line that does not name the topology cannot be told apart from the members it did not run."""
    result = gates.GateResult(name="accuracy[gsm8k]", passed=True, detail="x", member="dep4")
    assert "accuracy[gsm8k]@dep4" in result.summary()
    assert "@" not in gates.GateResult(name="boot", passed=True, detail="x").summary()


# ------------------------------------------------------------ score parsing


def test_lm_eval_score_is_parsed():
    out = "lm-eval gsm8k exact_match,flexible-extract accuracy: 90.67\n"
    assert gates.parse_eval_score(out) == pytest.approx(90.67)


def test_lm_eval_average_form_is_parsed():
    assert gates.parse_eval_score("lm-eval gsm8k average accuracy: 88.10\n") == pytest.approx(88.10)


def test_mmlu_form_is_parsed():
    out = "MMLU weighted average accuracy: 74.98 (14042)\n"
    assert gates.parse_eval_score(out) == pytest.approx(74.98)


def test_the_final_score_wins_over_partial_estimates():
    """A long run prints running estimates; only the last line is the result."""
    out = (
        "Partial scores after 100/1319 responses (estimate, 0~100): gsm8k,acc ~ 12.00\n"
        "Partial scores after 900/1319 responses (estimate, 0~100): gsm8k,acc ~ 80.00\n"
        "lm-eval gsm8k exact_match,flexible-extract accuracy: 90.67\n"
    )
    assert gates.parse_eval_score(out) == pytest.approx(90.67)


def test_no_score_returns_none_rather_than_a_guess():
    assert gates.parse_eval_score("engine built, nothing else\n") is None


def test_acceptance_length_is_parsed():
    out = (
        "Spec-dec stats: AL (decoded tokens/step, iteration-weighted) 3.351 "
        "(per-request min 1.000, max 4.000, n=1319)\n"
    )
    assert gates.parse_acceptance_length(out) == pytest.approx(3.351)


def test_acceptance_length_alternate_form_is_parsed():
    assert gates.parse_acceptance_length("acceptance_length: 3.28\n") == pytest.approx(3.28)


# --------------------------------------------------------------- reference


def _reference_file(tmp_path, entries):
    path = tmp_path / "gsm8k.yaml"
    path.write_text(yaml.safe_dump({"org/Model": entries}), encoding="utf-8")
    return path


def test_reference_matches_on_the_exact_discriminator_set(tmp_path):
    path = _reference_file(
        tmp_path,
        [
            {"accuracy": 74.20},
            {"quant_algo": "FP8", "accuracy": 74.30},
            {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "accuracy": 66.03},
        ],
    )
    assert gates.load_reference_score(path, "org/Model") == pytest.approx(74.20)
    assert gates.load_reference_score(
        path, "org/Model", discriminators={"quant_algo": "FP8"}
    ) == pytest.approx(74.30)
    assert gates.load_reference_score(
        path,
        "org/Model",
        discriminators={"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8"},
    ) == pytest.approx(66.03)


def test_an_fp8_anchor_is_not_silently_used_for_bf16(tmp_path):
    path = _reference_file(tmp_path, [{"quant_algo": "FP8", "accuracy": 74.30}])
    with pytest.raises(KeyError, match="no entry matching"):
        gates.load_reference_score(path, "org/Model")


def test_a_missing_model_says_the_anchor_is_never_invented(tmp_path):
    path = _reference_file(tmp_path, [{"accuracy": 1.0}])
    with pytest.raises(KeyError, match="never invented"):
        gates.load_reference_score(path, "org/Other")


def test_an_ambiguous_anchor_is_refused(tmp_path):
    path = _reference_file(tmp_path, [{"accuracy": 74.20}, {"accuracy": 75.00}])
    with pytest.raises(KeyError, match="unambiguous"):
        gates.load_reference_score(path, "org/Model")


def test_resolve_reference_joins_a_relative_path_to_the_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "gsm8k.yaml").write_text(
        yaml.safe_dump({"org/Model": [{"accuracy": 89.99}]}), encoding="utf-8"
    )
    task = {**_TASK, "trtllm_repo_path": str(repo)}
    task["accuracy"] = {**_TASK["accuracy"], "reference_yaml": "refs/gsm8k.yaml"}
    assert gates.resolve_reference(task, model_key="org/Model") == pytest.approx(89.99)


# -------------------------------------------------------------- boot gate


def test_boot_gate_passes_on_clean_exit_with_output():
    run = recorder(CommandResult(returncode=0, stdout=_PARIS))
    result = gates.run_boot_gate(_env(), task_data=_TASK, runner=run)
    assert result.passed, result.detail
    assert "quickstart_advanced.py" in result.command
    assert str(gates.ENGINE_DRIVER) in result.command
    assert "TRTLLM_MODELING_V2=require" in result.command


def test_every_gate_turns_on_the_targets_step_contract_check():
    """Each target's `step_args` probe runs only under the switch; a gate must set it."""
    run = recorder(CommandResult(returncode=0, stdout="ok"))
    gates.run_boot_gate(_env(), task_data=_TASK, runner=run)
    gates.run_accuracy_gate(_env(), task_data=_TASK, reference=90.0, runner=run)
    assert run.seen  # type: ignore[attr-defined]
    for command in run.seen:  # type: ignore[attr-defined]
        assert f"{gates.STEP_CONTRACT_ENV}=1" in command
    assert gates.STEP_CONTRACT_ENV == "TRTLLM_MODELING_V2_VALIDATE"


def test_boot_gate_always_forces_require_mode():
    """Under `auto` the built-in implementation answers and the gate lies."""
    run = recorder(CommandResult(returncode=0, stdout=_PARIS))
    gates.run_boot_gate(_env(), task_data={**_TASK, "parallel": "dep4"}, runner=run)
    command = run.seen[0]  # type: ignore[attr-defined]
    assert "TRTLLM_MODELING_V2=require" in command
    assert "--enable_attention_dp" in command


def test_boot_gate_fails_on_nonzero_exit():
    run = recorder(CommandResult(returncode=1, stderr="CUDA error"))
    result = gates.run_boot_gate(_env(), task_data=_TASK, runner=run)
    assert not result.passed and "exited 1" in result.detail


def test_boot_gate_fails_on_no_records_and_on_empty_generation():
    run = recorder(CommandResult(returncode=0, stdout="   \n"))
    result = gates.run_boot_gate(_env(), task_data=_TASK, runner=run)
    assert not result.passed and "returned no record for 2 of 2" in result.detail
    run = recorder(CommandResult(returncode=0, stdout=_gen("Paris", "  ")))
    result = gates.run_boot_gate(_env(), task_data=_TASK, runner=run)
    assert not result.passed and "generated nothing for b1" in result.detail


def test_boot_gate_checks_keywords_when_the_spec_has_them():
    spec = gates.BootSpec(prompts=("x",), keywords=("Paris", "Berlin"))
    run = recorder(CommandResult(returncode=0, stdout=_gen("Paris only")))
    result = gates.run_boot_gate(_env(), task_data=_TASK, spec=spec, runner=run)
    assert not result.passed and "Berlin" in result.detail


def test_a_keyword_in_the_log_is_not_a_keyword_in_the_text():
    """`72` matches `CPUs 72-143` in the engine log, and the prompt is echoed too."""
    spec = gates.BootSpec(prompts=("48 + 24 = ? Answer 72 if unsure.",), keywords=("72",))
    log = "[TRT-LLM] [I] Process 1 bound to CPUs 72-143; prompt '48 + 24 = ? Answer 72'"
    run = recorder(CommandResult(returncode=0, stdout=_gen("I do not know", extra=log)))
    result = gates.run_boot_gate(_env(), task_data=_TASK, spec=spec, runner=run)
    assert not result.passed and "'72'" in result.detail
    run = recorder(CommandResult(returncode=0, stdout=_gen("The answer is 72.", extra=log)))
    assert gates.run_boot_gate(_env(), task_data=_TASK, spec=spec, runner=run).passed


def test_a_prompts_own_keywords_are_checked_against_its_own_text():
    spec = gates.BootSpec(
        prompts=(),
        items=(
            gates.BootPrompt(prompt="capital?", keywords=("Paris",)),
            gates.BootPrompt(prompt="sum?", keywords_any=("72", "seventy-two")),
        ),
    )
    run = recorder(CommandResult(returncode=0, stdout=_gen("Seventy-two", "paris")))
    result = gates.run_boot_gate(_env(), task_data=_TASK, spec=spec, runner=run)
    assert not result.passed
    run = recorder(CommandResult(returncode=0, stdout=_gen("PARIS.", "seventy-two")))
    assert gates.run_boot_gate(_env(), task_data=_TASK, spec=spec, runner=run).passed


def test_boot_spec_defaults_to_no_keywords():
    """Keywords are frozen only after observing what this model actually says."""
    assert gates.BootSpec().keywords == ()


def test_boot_spec_loads_from_disk(tmp_path):
    path = tmp_path / "boot.yaml"
    path.write_text(yaml.safe_dump({"prompts": ["a"], "keywords": ["k"], "max_tokens": 8}))
    spec = gates.BootSpec.load(path)
    assert spec.prompts == ("a",) and spec.keywords == ("k",) and spec.max_tokens == 8


def test_boot_spec_missing_file_is_the_default(tmp_path):
    assert gates.BootSpec.load(tmp_path / "absent.yaml") == gates.BootSpec()


def test_boot_gate_timeout_says_it_is_no_longer_a_boot_gate():
    run = recorder(CommandResult(returncode=124, timed_out=True))
    result = gates.run_boot_gate(_env(), task_data=_TASK, runner=run, timeout=5)
    assert not result.passed and "not catching catastrophes" in result.detail
    assert result.infrastructure == ""


# ------------------------------------------------------- a lost session

_LOST = "the session watching /spool stopped heart-beating 412s ago; nothing was queued"


def test_a_gate_lost_to_its_session_says_so_rather_than_timed_out():
    """A gate that fails in 0.04s on a lost session must not read `timed out after 14400s`."""
    lost = CommandResult(returncode=124, timed_out=True, infrastructure=_LOST)
    for result in (
        gates.run_boot_gate(_env(), task_data=_TASK, runner=recorder(lost), timeout=1800),
        gates.run_accuracy_gate(
            _env(), task_data=_TASK, reference=89.99, runner=recorder(lost), timeout=14400
        ),
        gates.run_acceptance_gate(_env(), task_data=_TASK, runner=recorder(lost)),
    ):
        assert not result.passed and result.infrastructure == _LOST
        assert "did not run to a verdict" in result.detail and _LOST in result.detail
        assert "timed out" not in result.detail


def test_a_lost_stock_leg_is_not_read_as_an_unsupported_architecture():
    run = recorder(
        [
            CommandResult(returncode=0, stdout="Spec-dec stats: AL (x) 3.3514 (y)\n"),
            CommandResult(returncode=124, timed_out=True, infrastructure=_LOST),
        ]
    )
    result = gates.run_acceptance_gate(_env(), task_data=_TASK, runner=run)
    assert not result.passed and result.infrastructure == _LOST
    assert "TRTLLM_MODELING_V2=off leg did not run to a verdict" in result.detail


def test_a_real_timeout_is_still_a_timeout():
    run = recorder(CommandResult(returncode=124, timed_out=True))
    result = gates.run_accuracy_gate(
        _env(), task_data=_TASK, reference=89.99, runner=run, timeout=14400
    )
    assert result.detail == "timed out after 14400s" and result.infrastructure == ""


# ----------------------------------------------------------- accuracy gate


# The fixed-tolerance rule, which is an explicit choice.
_TOLERANCE_TASK = {**_TASK, "accuracy": {**_TASK["accuracy"], "floor": {"rule": "tolerance"}}}


def test_accuracy_gate_is_one_sided():
    run = recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 85.50\n"))
    result = gates.run_accuracy_gate(_env(), task_data=_TOLERANCE_TASK, reference=89.99, runner=run)
    assert result.passed  # 85.50 >= 89.99 - 5.0
    assert result.measured == pytest.approx(85.50)


def test_accuracy_gate_fails_below_the_floor():
    run = recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 80.00\n"))
    result = gates.run_accuracy_gate(_env(), task_data=_TOLERANCE_TASK, reference=89.99, runner=run)
    assert not result.passed
    assert "80.0000 < 84.9900" in result.detail
    # The SE is recorded under the tolerance rule too, when the count is known.
    assert result.se == pytest.approx(0.8274, abs=1e-3) and result.samples == 1319


def test_accuracy_gate_with_no_parsable_score_is_not_a_pass():
    """A gate with no number is not a pass, however clean the exit."""
    run = recorder(CommandResult(returncode=0, stdout="done\n"))
    result = gates.run_accuracy_gate(_env(), task_data=_TASK, reference=89.99, runner=run)
    assert not result.passed and "no score" in result.detail


def test_accuracy_gate_passes_the_dataset_path_when_declared():
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "dataset_path": "/data/mmlu"}}
    run = recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 99\n"))
    gates.run_accuracy_gate(_env(), task_data=task, reference=1.0, runner=run)
    assert '--dataset_path="/data/mmlu"' in run.seen[0]  # type: ignore[attr-defined]


def test_gate_summary_carries_the_numbers_verbatim():
    run = recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 95.0720\n"))
    result = gates.run_accuracy_gate(_env(), task_data=_TASK, reference=89.9962, runner=run)
    summary = result.summary()
    assert "95.0720" in summary and "89.9962" in summary and "PASS" in summary


# --------------------------------------------------------- acceptance gate


def test_acceptance_gate_compares_require_against_off():
    run = recorder(
        [
            CommandResult(returncode=0, stdout="Spec-dec stats: AL (x) 3.3514 (y)\n"),
            CommandResult(returncode=0, stdout="Spec-dec stats: AL (x) 3.2752 (y)\n"),
        ]
    )
    result = gates.run_acceptance_gate(_env(), task_data=_TASK, runner=run)
    assert result.passed
    assert result.measured == pytest.approx(3.3514)
    assert result.reference == pytest.approx(3.2752)
    seen = run.seen  # type: ignore[attr-defined]
    assert "TRTLLM_MODELING_V2=require" in seen[0]
    assert "TRTLLM_MODELING_V2=off" in seen[1]


def test_acceptance_of_one_means_every_draft_was_rejected():
    run = recorder(
        [
            CommandResult(returncode=0, stdout="Spec-dec stats: AL (x) 1.0000 (y)\n"),
            CommandResult(returncode=0, stdout="Spec-dec stats: AL (x) 3.2752 (y)\n"),
        ]
    )
    result = gates.run_acceptance_gate(_env(), task_data=_TASK, runner=run)
    assert not result.passed and "every draft was rejected" in result.detail


def test_acceptance_below_the_reference_fails():
    """Clearly above 1.0 but below the reference means subtly wrong, not slower."""
    run = recorder(
        [
            CommandResult(returncode=0, stdout="Spec-dec stats: AL (x) 2.0000 (y)\n"),
            CommandResult(returncode=0, stdout="Spec-dec stats: AL (x) 3.2752 (y)\n"),
        ]
    )
    result = gates.run_acceptance_gate(_env(), task_data=_TASK, runner=run)
    assert not result.passed


# ------------------------------------------------ the acceptance workload

# A speculator run on short-answer boot prompts at 256 tokens: every answer
# is 2-3 tokens then EOS, so the rate is 1.00 for any drafter.
_BOOT_WORKLOAD_OUTPUT = (
    "[0] Accept rate: 1.00 (tokens=2, iterations=2)\n"
    "[1] Accept rate: 1.00 (tokens=2, iterations=2)\n"
    "[2] Accept rate: 1.00 (tokens=3, iterations=3)\n"
)
# The same command on three long-answer prompts with a working drafter.
_LONG_WORKLOAD_OUTPUT = (
    "[0] Accept rate: 3.45 (tokens=152, iterations=44)\n"
    "[1] Accept rate: 2.25 (tokens=256, iterations=114)\n"
    "[2] Accept rate: 5.69 (tokens=239, iterations=42)\n"
)
_LONG_PROMPTS = [
    "<BOS>Explain step by step how a refrigerator moves heat.",
    '<BOS>Write a Python module that parses "key = $value" lines, with tests.',
]


def _no_stock_runner(output):
    """`=require` prints `output`; `=off` cannot load a new architecture."""
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        if "TRTLLM_MODELING_V2=off" in command:
            return CommandResult(
                returncode=1,
                stderr="RuntimeError: Unknown model architecture: NewForCausalLM",
            )
        return CommandResult(returncode=0, stdout=output)

    run.seen = seen  # type: ignore[attr-defined]
    return run


def _acceptance_task(**acceptance):
    """A task whose acceptance block carries a floor; ``min_prompts`` defaults to 1 here.

    These tests exercise the floor and the iteration minimum on small canned
    workloads; the prompt minimum has tests of its own.
    """
    acceptance.setdefault("min_prompts", 1)
    return {
        **_TASK,
        "accuracy": {
            **_TASK["accuracy"],
            "acceptance": {"dspark": {"min_absolute": 2.0}, **acceptance},
        },
    }


def _spec_of(command):
    """The engine-driver spec a gate command points at."""
    words = shlex.split(command)
    return json.loads(open(words[words.index("--spec") + 1], encoding="utf-8").read())


def test_a_short_answer_workload_fails_as_too_short_not_as_a_low_number():
    """Short-answer boot prompts read 1.00 whatever the drafter does.

    Reported as a floor miss, 1.00 against 2.0 reads as a broken drafter and
    sends the assembler after a defect the target does not have.
    """
    result = gates.run_acceptance_gate(
        _env(),
        task_data=_acceptance_task(),
        runner=_no_stock_runner(_BOOT_WORKLOAD_OUTPUT),
        unit="dspark",
    )
    assert not result.passed
    assert "workload too short to measure acceptance" in result.detail
    assert "4 generation iteration(s) over 3 prompt(s)" in result.detail
    assert "openai/gsm8k" in result.detail
    # No number is reported as the target's: there was no measurement.
    assert result.measured is None
    assert "[2] 1.00 (3 tok/3 it)" in result.summary()


def test_the_two_leg_comparison_is_held_to_the_same_minimum():
    run = recorder(
        [
            CommandResult(returncode=0, stdout=_LONG_WORKLOAD_OUTPUT),
            CommandResult(returncode=0, stdout=_BOOT_WORKLOAD_OUTPUT),
        ]
    )
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "acceptance": {"min_prompts": 3}}}
    result = gates.run_acceptance_gate(_env(), task_data=task, runner=run)
    assert not result.passed
    assert "workload too short" in result.detail
    assert "TRTLLM_MODELING_V2=off leg" in result.detail


def test_fewer_prompts_than_the_minimum_is_a_workload_too_small():
    """On four prompts a change in generated text reads as a regression."""
    result = gates.run_acceptance_gate(
        _env(),
        task_data=_acceptance_task(prompts=_LONG_PROMPTS, min_prompts=32),
        runner=_no_stock_runner(_LONG_WORKLOAD_OUTPUT),
        unit="dspark",
    )
    assert not result.passed
    assert "workload too small to compare acceptance" in result.detail
    assert "3 prompt(s), fewer than the 32" in result.detail


def test_per_prompt_acceptance_is_reported_beside_the_mean():
    """The verdict is on the iteration-weighted rate; the journal shows every prompt."""
    result = gates.run_acceptance_gate(
        _env(),
        task_data=_acceptance_task(prompts=_LONG_PROMPTS),
        runner=_no_stock_runner(_LONG_WORKLOAD_OUTPUT),
        unit="dspark",
    )
    assert result.passed, result.detail
    assert result.measured == pytest.approx(647 / 200)
    summary = result.summary()
    for expected in ("[0] 3.45 (152 tok/44 it)", "[1] 2.25", "[2] 5.69", "prompt mean 3.80"):
        assert expected in summary, summary
    assert "197 generation iterations" in summary


def test_a_declared_workload_replaces_the_dataset(tmp_path):
    run = _no_stock_runner(_LONG_WORKLOAD_OUTPUT)
    gates.run_acceptance_gate(
        _env(),
        task_data=_acceptance_task(prompts=_LONG_PROMPTS, max_tokens=300),
        runner=run,
        unit="dspark",
        workdir=tmp_path,
    )
    command = run.seen[0]  # type: ignore[attr-defined]
    assert str(gates.ENGINE_DRIVER) in command
    spec = _spec_of(command)
    # The prompts travel in the spec file, whole, quotes and `$` included.
    assert [r["prompt"] for r in spec["requests"]] == _LONG_PROMPTS
    assert {r["max_tokens"] for r in spec["requests"]} == {300}
    assert spec["dataset"] is None
    assert "--max_tokens=300" in spec["quickstart_argv"]
    assert spec["quickstart"].endswith("examples/llm-api/quickstart_advanced.py")


def test_no_declared_workload_takes_32_prompts_from_the_accuracy_dataset():
    """Not the boot prompts: their answers end in EOS before any draft is verified."""
    workload = gates.resolve_acceptance_workload({"benchmark": "gsm8k"}, "mtp")
    assert workload.prompts == ()
    assert workload.dataset["source"] == "openai/gsm8k"
    assert workload.dataset["num"] == gates.DEFAULT_ACCEPTANCE_NUM_PROMPTS == 32
    assert workload.dataset["text_field"] == "question"
    assert workload.max_tokens == gates.DEFAULT_ACCEPTANCE_MAX_TOKENS == 256
    assert workload.min_prompts == 32
    assert "openai/gsm8k" in workload.source


def test_a_local_dataset_path_is_preferred_to_the_cache():
    accuracy = {"benchmark": "gsm8k", "dataset_path": "/data/gsm8k"}
    assert gates.resolve_acceptance_workload(accuracy, None).dataset["source"] == "/data/gsm8k"


def test_a_multimodal_unit_draws_from_the_run_wide_text_benchmark():
    accuracy = {"benchmark": {"default": "gsm8k", "multimodal": "mmmu"}}
    workload = gates.resolve_acceptance_workload(accuracy, "multimodal")
    assert workload.dataset["source"] == "openai/gsm8k"


def test_a_benchmark_with_no_known_dataset_must_declare_a_workload():
    with pytest.raises(gates.TaskSchemaError, match="no acceptance workload"):
        gates.resolve_acceptance_workload({"benchmark": "longbench"}, None)
    workload = gates.resolve_acceptance_workload(
        {"benchmark": "longbench", "acceptance": {"dataset": "/d.jsonl", "num_prompts": 40}}, None
    )
    assert workload.dataset["source"] == "/d.jsonl" and workload.dataset["num"] == 40


def test_a_unit_entry_overrides_the_run_wide_workload():
    accuracy = {
        "acceptance": {
            "prompts": ["run-wide"],
            "max_tokens": 256,
            "min_draft_iterations": 64,
            "dspark": {"min_absolute": 2.0},
            "multimodal": {"min_absolute": 2.0, "prompts": ["image-side"], "max_tokens": 128},
        }
    }
    dspark = gates.resolve_acceptance_workload(accuracy, "dspark")
    assert dspark.prompts == ("run-wide",) and dspark.max_tokens == 256
    assert dspark.min_draft_iterations == 64
    assert dspark.source == "accuracy.acceptance.prompts"
    multimodal = gates.resolve_acceptance_workload(accuracy, "multimodal")
    assert multimodal.prompts == ("image-side",) and multimodal.max_tokens == 128
    # A key the unit does not state still comes from the run-wide level.
    assert multimodal.min_draft_iterations == 64
    assert dspark.fingerprint != multimodal.fingerprint


def test_a_prompts_file_is_read_in_either_shape(tmp_path):
    as_list = tmp_path / "prompts.json"
    as_list.write_text(json.dumps(["a long question", "another"]), encoding="utf-8")
    as_boot = tmp_path / "boot.yaml"
    as_boot.write_text(yaml.safe_dump({"prompts": ["from boot"], "keywords": ["x"]}))
    for path, expected in ((as_list, ("a long question", "another")), (as_boot, ("from boot",))):
        workload = gates.resolve_acceptance_workload(
            {"acceptance": {"prompts_file": str(path)}}, "dspark"
        )
        assert workload.prompts == expected
        assert str(path) in workload.source


def test_an_unreadable_prompts_file_fails_the_gate_before_it_runs(tmp_path):
    run = _no_stock_runner(_LONG_WORKLOAD_OUTPUT)
    result = gates.run_acceptance_gate(
        _env(),
        task_data=_acceptance_task(prompts_file=str(tmp_path / "gone.json")),
        runner=run,
        unit="dspark",
    )
    assert not result.passed
    assert "declared acceptance workload cannot be read" in result.detail
    assert run.seen == []  # type: ignore[attr-defined]


def test_a_declared_minimum_is_the_one_applied():
    result = gates.run_acceptance_gate(
        _env(),
        task_data=_acceptance_task(prompts=_LONG_PROMPTS, min_draft_iterations=500),
        runner=_no_stock_runner(_LONG_WORKLOAD_OUTPUT),
        unit="dspark",
    )
    assert not result.passed
    assert "197 generation iteration(s)" in result.detail and "500" in result.detail


def test_a_long_workload_with_a_dead_drafter_still_fails_the_floor():
    """Long answers are what let the floor see the drafter at all."""
    dead = (
        "[0] Accept rate: 1.00 (tokens=256, iterations=256)\n"
        "[1] Accept rate: 1.00 (tokens=256, iterations=256)\n"
    )
    result = gates.run_acceptance_gate(
        _env(),
        task_data=_acceptance_task(prompts=_LONG_PROMPTS),
        runner=_no_stock_runner(dead),
        unit="dspark",
    )
    assert not result.passed
    assert "too short" not in result.detail
    assert result.measured == pytest.approx(1.0) and result.reference == pytest.approx(2.0)


# ------------------------------------------------------------------- log


def test_gate_log_keeps_order_and_finds_the_latest():
    log = gates.GateLog()
    log.record(gates.GateResult(name="boot", passed=True, detail="a"))
    log.record(gates.GateResult(name="accuracy[gsm8k]", passed=False, detail="b"))
    log.record(gates.GateResult(name="boot", passed=False, detail="c"))
    assert log.last("boot").detail == "c"
    assert log.last("accuracy").passed is False
    assert log.last("acceptance") is None
    assert len(log.render().splitlines()) == 3


def test_the_two_gate_commands_spell_expert_parallelism_differently():
    """`--moe_ep_size` for quickstart, `--ep_size` for trtllm-eval.

    The quickstart spelling handed to `trtllm-eval` fails with::

        Error: No such option: --moe_ep_size
        (Possible options: --ep_size, --pp_size, --tp_size)

    A `tp` member emits only `--tp_size`, which both commands accept, so the
    mismatch is invisible at the one member that does not exercise it.
    """
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import EVAL_DIALECT, topology_flags

    assert topology_flags("unified", 4, "tep4") == ["--tp_size=4", "--moe_ep_size=4"]
    assert topology_flags("unified", 4, "tep4", dialect=EVAL_DIALECT) == [
        "--tp_size=4",
        "--ep_size=4",
    ]


def test_attention_dp_is_not_a_trtllm_eval_flag():
    """`trtllm-eval --help` lists tp/pp/ep and nothing for attention DP.

    So a `dep` member cannot carry it on the argv, and emitting it there
    fails the command as above. It travels in the options overlay
    instead — and the overlay must not be empty, or the gate would serve
    `tep` while the verdict claimed `dep`.
    """
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import (
        EVAL_DIALECT,
        member_llm_api_overlay,
        topology_flags,
    )

    eval_flags = topology_flags("unified", 4, "dep4", dialect=EVAL_DIALECT)
    assert "--enable_attention_dp" not in eval_flags
    assert eval_flags == ["--tp_size=4", "--ep_size=4"]
    assert member_llm_api_overlay("dep4") == {"enable_attention_dp": True}
    # The quickstart dialect keeps the flag, because that command takes it.
    assert "--enable_attention_dp" in topology_flags("unified", 4, "dep4")


def test_only_dep_members_need_an_overlay():
    """An overlay that fires for every member would be a different bug."""
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import member_llm_api_overlay

    assert member_llm_api_overlay("tp4") == {}
    assert member_llm_api_overlay("tep4") == {}
    assert member_llm_api_overlay(None) == {}


def test_the_draft_depth_comes_from_the_options_yaml(tmp_path):
    """All three gates must ask for the depth the unit was built for.

    `trtllm-eval` takes no speculation flags, so the accuracy gate reads
    `speculative_config.num_nextn_predict_layers` from the unit's options
    YAML. The boot and acceptance gates drive `quickstart_advanced.py`, which
    cannot read that file and takes `--spec_decode_max_draft_len` instead.

    If the flag disagrees with the YAML (say 1 against 3), acceptance
    measures a shallower speculator than the unit was built for. That fails
    toward green, because a shallower speculator accepts a larger fraction of
    what it drafts.
    """
    import yaml as _yaml

    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import (
        resolve_max_draft_len,
        speculation_flags,
    )

    options = tmp_path / "mtp_llm_api.yaml"
    options.write_text(
        _yaml.safe_dump(
            {"speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 3}}
        )
    )
    accuracy = {"llm_api_options": {"mtp": str(options)}}

    assert resolve_max_draft_len(accuracy, "mtp") == 3
    assert speculation_flags(["mtp"], max_draft_len=3) == [
        "--spec_decode_algo=MTP",
        "--spec_decode_max_draft_len=3",
    ]


def test_a_unit_with_no_speculator_still_answers_one(tmp_path):
    """Every unit that is not a speculator gets a draft depth of 1."""
    import yaml as _yaml

    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import resolve_max_draft_len

    assert resolve_max_draft_len({}, "core") == 1
    assert resolve_max_draft_len({"llm_api_options": "/nonexistent.yaml"}, "core") == 1

    plain = tmp_path / "core_llm_api.yaml"
    plain.write_text(_yaml.safe_dump({"enable_chunked_prefill": True}))
    assert resolve_max_draft_len({"llm_api_options": str(plain)}, "core") == 1


def test_the_off_leg_can_be_impossible_rather_than_broken(tmp_path):
    """A new architecture has no built-in modeling to compare a speculator to.

    The acceptance gate's `=off` leg runs the same checkpoint under stock
    in-tree modeling. For a genuinely new architecture that path cannot load
    the checkpoint at all — so the gate is unavailable exactly when onboarding
    is doing its job, which is the substrate gate's problem one gate down.

    Skipping it would be worse than failing: it is the ONLY gate that can see
    a miscomputed draft path, because rejection sampling makes a wrong draft
    produce correct text more slowly. So a declared absolute floor stands in,
    and the weaker claim is recorded.
    """
    import yaml as _yaml

    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import resolve_acceptance_floor

    options = tmp_path / "mtp.yaml"
    options.write_text(_yaml.safe_dump({"speculative_config": {"num_nextn_predict_layers": 3}}))
    accuracy = {
        "llm_api_options": {"mtp": str(options)},
        "acceptance": {"mtp": {"min_absolute": 1.5}},
    }
    assert resolve_acceptance_floor(accuracy, "mtp") == 1.5
    # A unit that declares none keeps the two-leg comparison and its hard fail.
    assert resolve_acceptance_floor(accuracy, "core") is None
    assert resolve_acceptance_floor({}, "mtp") is None


def test_the_unsupported_architecture_markers_are_shared_with_preflight():
    """One definition of "the stock path does not know this model".

    The substrate gate and the acceptance gate ask the same question for the
    same reason, so they must not drift apart.
    """
    from agent_flow.workflows.trtllm_modeling_v2.run.preflight import (
        reports_unsupported_architecture,
    )

    assert reports_unsupported_architecture(
        "exited 1", "Unknown architecture for AutoModelForCausalLM"
    )
    assert reports_unsupported_architecture(
        "exited 1",
        "ValueError: The repository /ckpt contains custom code which must be "
        "executed to correctly load the model.",
    )
    # A real fault must still be a hard stop, not a fallback.
    assert not reports_unsupported_architecture(
        "exited 1", "Impossible to fit in any sequence in kvCache"
    )


def test_the_registry_miss_is_also_an_unsupported_architecture():
    """`get_model_architecture` raises its own message, by its own path.

    `Unknown model architecture:` comes from the registry lookup in
    `_torch/models/modeling_utils.py`, not from the `AutoModelFor*` config
    loaders. A checkpoint can fail there without matching any other marker --
    and for the acceptance gate that is the difference between falling back to
    a declared floor and hard-failing a working unit.
    """
    from agent_flow.workflows.trtllm_modeling_v2.run.preflight import (
        reports_unsupported_architecture,
    )

    assert reports_unsupported_architecture(
        "exited 1", "RuntimeError: Unknown model architecture: MiMoV2ForCausalLM"
    )


# ------------------------------------------------- the tree a gate measured


def _git(repo, *args):
    """Run one git command in ``repo``, with an identity the host may not have."""
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=mv2flow",
            "-c",
            "user.email=mv2flow@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(repo),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _repo_with_target(tmp_path, *, family="gpt_oss"):
    """A real one-commit repo holding a modeling_v2 family directory.

    Real git rather than a mocked one on purpose: what this guard has to get
    right is the exact output of `git status --porcelain`, which is the kind
    of thing a stub agrees with and the tool does not.
    """
    repo = tmp_path / "repo"
    family_dir = repo / gates.MODELING_V2_PACKAGE / "models" / family
    family_dir.mkdir(parents=True)
    (family_dir / "routing.py").write_text("# committed\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "note.md").write_text("committed\n", encoding="utf-8")
    _git(repo, "init", "-q", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "target")
    return repo


def _task_for(repo, family="gpt_oss"):
    return {**_TASK, "trtllm_repo_path": str(repo), "target": {"family": family}}


def test_the_tree_state_reaches_the_summary():
    """`progress.yaml` and the journal read this line; attribution rides on it."""
    plain = gates.GateResult(name="boot", passed=True, detail="ok", tree_state="v1-3-gabc1234")
    assert "[tree v1-3-gabc1234]" in plain.summary()
    numeric = gates.GateResult(
        name="accuracy[gsm8k]",
        passed=True,
        detail="ok",
        measured=90.0,
        reference=89.0,
        tolerance=5.0,
        tree_state="abc1234",
    )
    assert "[tree abc1234]" in numeric.summary()


def test_a_gate_records_the_commit_it_measured(tmp_path):
    repo = _repo_with_target(tmp_path)
    run = recorder(CommandResult(returncode=0, stdout=_PARIS))
    result = gates.run_boot_gate(_env(), task_data=_task_for(repo), runner=run)
    assert result.passed
    assert result.tree_state == _git(repo, "describe", "--always")
    assert result.tree_state in result.summary()


def test_the_accuracy_and_acceptance_gates_record_it_too(tmp_path):
    """A number is attributable or it is not; which gate produced it is not the question."""
    repo = _repo_with_target(tmp_path)
    expected = _git(repo, "describe", "--always")
    accuracy = gates.run_accuracy_gate(
        _env(),
        task_data=_task_for(repo),
        reference=89.99,
        runner=recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 90\n")),
    )
    acceptance = gates.run_acceptance_gate(
        _env(),
        task_data=_task_for(repo),
        runner=recorder(
            [
                CommandResult(returncode=0, stdout="Spec-dec stats: AL (x) 3.3514 (y)\n"),
                CommandResult(returncode=0, stdout="Spec-dec stats: AL (x) 3.2752 (y)\n"),
            ]
        ),
    )
    assert accuracy.tree_state == expected
    assert acceptance.tree_state == expected


def test_a_gate_still_runs_where_git_cannot_answer(tmp_path, monkeypatch):
    """No repository, no commits, no git binary — none of them is a modeling defect."""
    # A TMPDIR inside a checkout (a short one is needed for the RPC socket
    # path) would otherwise let git walk up into that checkout and answer.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    run = recorder(CommandResult(returncode=0, stdout=_PARIS))
    result = gates.run_boot_gate(
        _env(), task_data={**_TASK, "trtllm_repo_path": str(tmp_path)}, runner=run
    )
    assert result.passed
    assert result.tree_state == gates.UNKNOWN_TREE_STATE


def test_git_missing_entirely_does_not_fail_a_gate(tmp_path, monkeypatch):
    """The host with no `git` on PATH. A gate adjudicates a model, not a toolchain."""
    repo = _repo_with_target(tmp_path)

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(vcs.subprocess, "run", no_git)
    run = recorder(CommandResult(returncode=0, stdout=_PARIS))
    result = gates.run_boot_gate(_env(), task_data=_task_for(repo), runner=run)
    assert result.passed and result.tree_state == gates.UNKNOWN_TREE_STATE


# --------------------------------------------------- the dirty-target guard


def test_a_dirty_target_path_stops_the_gate_before_it_spends_anything(tmp_path):
    """At gate time the assembler has already committed; dirty means another writer."""
    repo = _repo_with_target(tmp_path)
    modeling = repo / gates.MODELING_V2_PACKAGE / "models" / "gpt_oss" / "modeling.py"
    modeling.write_text("# half-written by somebody else\n", encoding="utf-8")

    run = recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 90\n"))
    result = gates.run_accuracy_gate(_env(), task_data=_task_for(repo), reference=89.99, runner=run)
    assert not result.passed
    assert "modeling.py" in result.detail
    # And it cost nothing: the guard is in front of the command, not after it.
    assert run.seen == []  # type: ignore[attr-defined]


def test_every_gate_refuses_a_dirty_target_path(tmp_path):
    repo = _repo_with_target(tmp_path)
    (repo / gates.MODELING_V2_PACKAGE / "models" / "gpt_oss" / "weights.py").write_text(
        "# in flight\n", encoding="utf-8"
    )
    task = _task_for(repo)
    boot = gates.run_boot_gate(_env(), task_data=task, runner=recorder([]))
    acceptance = gates.run_acceptance_gate(_env(), task_data=task, runner=recorder([]))
    assert not boot.passed and "weights.py" in boot.detail
    assert not acceptance.passed and "weights.py" in acceptance.detail


def test_the_guard_is_scoped_to_the_target_not_to_the_repo(tmp_path):
    """A run has records and workflow fixes in flight constantly.

    A repo-wide check would refuse every gate of every run, which is a guard
    nobody can leave switched on.
    """
    repo = _repo_with_target(tmp_path)
    (repo / "docs" / "note.md").write_text("edited while the gate runs\n", encoding="utf-8")
    (repo / "07-workflow-feedback.md").write_text("new record\n", encoding="utf-8")
    run = recorder(CommandResult(returncode=0, stdout=_PARIS))
    result = gates.run_boot_gate(_env(), task_data=_task_for(repo), runner=run)
    assert result.passed and not result.dirty_override


def test_the_escape_hatch_runs_the_gate_and_marks_the_number(tmp_path):
    repo = _repo_with_target(tmp_path)
    (repo / gates.MODELING_V2_PACKAGE / "models" / "gpt_oss" / "modeling.py").write_text(
        "# debugging by hand\n", encoding="utf-8"
    )
    run = recorder(CommandResult(returncode=0, stdout=_PARIS))
    result = gates.run_boot_gate(_env(), task_data=_task_for(repo), runner=run, allow_dirty=True)
    assert result.passed and result.dirty_override
    assert "DIRTY TARGET PATH" in result.summary()
    assert result.tree_state.endswith("-dirty")


def test_the_escape_hatch_can_come_from_the_environment(tmp_path, monkeypatch):
    repo = _repo_with_target(tmp_path)
    (repo / gates.MODELING_V2_PACKAGE / "models" / "gpt_oss" / "modeling.py").write_text(
        "# debugging by hand\n", encoding="utf-8"
    )
    monkeypatch.setenv(gates.ALLOW_DIRTY_ENV, "1")
    result = gates.run_boot_gate(
        _env(),
        task_data=_task_for(repo),
        runner=recorder(CommandResult(returncode=0, stdout=_PARIS)),
    )
    assert result.passed and result.dirty_override


def test_a_clean_target_is_not_marked_as_overridden(tmp_path, monkeypatch):
    """The override marks a number measured over someone's edits, not every number."""
    monkeypatch.setenv(gates.ALLOW_DIRTY_ENV, "1")
    repo = _repo_with_target(tmp_path)
    result = gates.run_boot_gate(
        _env(),
        task_data=_task_for(repo),
        runner=recorder(CommandResult(returncode=0, stdout=_PARIS)),
    )
    assert result.passed and not result.dirty_override
    assert "DIRTY" not in result.summary()


def test_the_family_path_follows_the_checkpoint_when_nothing_overrides_it(tmp_path):
    """Derived the way preflight derives it, so the guard watches what is written."""
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["GptOssForCausalLM"]}), encoding="utf-8"
    )
    task = {**_TASK, "checkpoint_path": str(checkpoint)}
    assert gates.target_family_rel(task) == f"{gates.MODELING_V2_PACKAGE}/models/gpt_oss"
    assert (
        gates.target_family_rel({**task, "target": {"family": "mimo_v2"}})
        == f"{gates.MODELING_V2_PACKAGE}/models/mimo_v2"
    )


def test_an_underivable_family_guards_nothing_rather_than_guessing(tmp_path):
    assert gates.target_family_rel({"checkpoint_path": str(tmp_path / "absent")}) is None


# ---------------------------------------------------------- member ordering


def test_members_are_ordered_cheapest_first():
    """A unit is gated at every member and a broken unit fails at all of them."""
    assert gates.order_members_by_cost(["tp4", "dep4", "tep4"]) == ["tep4", "dep4", "tp4"]


def test_the_order_is_a_function_of_the_member_set_alone():
    """Two runs of the same unit must gate in the same sequence, or the journals stop comparing line by line."""
    orderings = {
        tuple(gates.order_members_by_cost(list(permutation)))
        for permutation in itertools.permutations(["tp4", "tep4", "dep4"])
    }
    assert orderings == {("tep4", "dep4", "tp4")}


def test_a_larger_world_size_breaks_the_tie():
    assert gates.order_members_by_cost(["tep8", "tep2", "tep4"]) == ["tep2", "tep4", "tep8"]


def test_an_unrecognized_member_sorts_last_and_does_not_raise():
    """Including `unified`, which names a family rather than a configuration.

    This function decides a running order; a gate loop that died while
    choosing one would have failed worse than one that ran in a poor order.
    """
    ordered = gates.order_members_by_cost(["unified", "nonsense", "tp4", "tep4"])
    assert ordered[:2] == ["tep4", "tp4"]
    assert set(ordered[2:]) == {"unified", "nonsense"}


def test_ordering_keeps_every_member():
    members = ["tp4", "tep4", "dep4", "tep4"]
    assert sorted(gates.order_members_by_cost(members)) == sorted(members)


def test_the_degenerate_member_set_is_returned_unchanged():
    """A concrete segment is gated at itself, once."""
    assert gates.order_members_by_cost(["tp8"]) == ["tp8"]
    assert gates.order_members_by_cost([]) == []


def test_a_member_overlay_never_lands_in_the_source_tree(tmp_path, monkeypatch):
    """A generated file in the repo is indistinguishable from a checked-in one.

    With no unit options file the derived copy has nowhere natural to go, and
    the cwd is the one place it must not be: the cwd of a test run is the
    repository.
    """
    from pathlib import Path

    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import _write_member_options

    monkeypatch.chdir(tmp_path)
    written = Path(
        _write_member_options(None, {"enable_attention_dp": True}, unit="core", member="dep4")
    )
    assert written.is_file()
    assert written.parent != tmp_path, "the overlay landed in the working directory"


# ---------------------------------------------- where a gate's job reads from

#: Every kind whose scheduler, not this process, places the job.
_SCHEDULED_KINDS = ("session_spool",)


def _scheduled_env(kind: str, world_size: int = 1):
    return resolve_environment(
        {"environment": {"kind": kind, "python": "/py", "spool": "/spool", "env": {}}},
        world_size=world_size,
    )


@pytest.fixture
def orchestrator_tmp(tmp_path, monkeypatch):
    """This process's temp dir, pointed where a test can see what lands in it."""
    import tempfile

    path = tmp_path / "orchestrator-tmp"
    path.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(path))
    return path


@pytest.mark.parametrize("kind", _SCHEDULED_KINDS)
def test_a_scheduled_gate_without_a_workdir_refuses_the_temp_dir(kind, orchestrator_tmp):
    """The job may run on another host, which cannot read this process's temp dir.

    An orchestrator in a sandbox has a temp dir no compute node sees, and a
    spool job reading a spec written there dies with
    `FileNotFoundError: .../boot.unit.member.json`.
    """
    with pytest.raises(ValueError, match=f"no workdir.*`{kind}`.*shared filesystem"):
        gates.gate_workdir(None, "boot", _TASK, environment=_scheduled_env(kind))
    with pytest.raises(ValueError, match="no workdir"):
        gates.gate_workdir(None, "boot", environment=_scheduled_env(kind))
    assert not any(orchestrator_tmp.iterdir())


@pytest.mark.parametrize("kind", _SCHEDULED_KINDS)
def test_the_runs_own_kind_refuses_the_temp_dir_too(kind, orchestrator_tmp):
    """The run's kind picks the executor, so a local gate launcher still runs elsewhere."""
    task = {**_TASK, "environment": {"kind": kind}}
    with pytest.raises(ValueError, match=f"`{kind}`"):
        gates.gate_workdir(None, "boot", task, environment=_env())
    assert not any(orchestrator_tmp.iterdir())


@pytest.mark.parametrize("kind", _SCHEDULED_KINDS)
def test_a_scheduled_gate_writes_into_the_workdir_it_is_given(kind, tmp_path, orchestrator_tmp):
    workdir = tmp_path / "ws" / "gates" / "core"
    got = gates.gate_workdir(workdir, "boot", _TASK, environment=_scheduled_env(kind))
    assert got == workdir and workdir.is_dir()
    assert not any(orchestrator_tmp.iterdir())


def test_a_local_gate_without_a_workdir_still_uses_the_temp_dir(orchestrator_tmp):
    """Nothing reads it from another host, so the temp-dir fallback applies."""
    keyed = gates.gate_workdir(None, "boot", _TASK, environment=_env())
    assert keyed.parent == orchestrator_tmp and keyed.name.startswith("mv2flow-boot-")
    assert gates.gate_workdir(None, "boot", _TASK) == keyed
    fresh = gates.gate_workdir(None, "boot")
    assert fresh.parent == orchestrator_tmp and fresh != keyed


def test_a_scheduled_boot_gate_without_a_workdir_runs_nothing(orchestrator_tmp):
    run = recorder(CommandResult(returncode=0, stdout=_PARIS))
    with pytest.raises(ValueError, match="no workdir"):
        gates.run_boot_gate(_scheduled_env("session_spool"), task_data=_TASK, runner=run)
    assert run.seen == []  # type: ignore[attr-defined]


def test_a_scheduled_boot_gate_reads_its_spec_from_the_workdir(tmp_path, orchestrator_tmp):
    run = recorder(CommandResult(returncode=0, stdout=_PARIS))
    workdir = tmp_path / "ws" / "gates" / "core"
    result = gates.run_boot_gate(
        _scheduled_env("session_spool"),
        task_data=_TASK,
        runner=run,
        unit="core",
        member="tp1",
        workdir=workdir,
    )
    assert result.passed, result.detail
    spec = workdir / "boot.core.tp1.json"
    assert spec.is_file() and str(spec) in run.seen[0]  # type: ignore[attr-defined]
    assert not any(orchestrator_tmp.iterdir())


_DEP_TASK = {**_TASK, "parallel": "unified"}


@pytest.mark.parametrize("kind", _SCHEDULED_KINDS)
def test_a_dep_members_options_land_in_the_workdir_without_a_unit_file(
    kind, tmp_path, orchestrator_tmp
):
    """With no unit options file the member's overlay has nowhere natural to go.

    The `trtllm-eval` command names it, so it must be where the job can read
    it: the gate's workdir, not this process's temp dir.
    """
    run = recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 95\n"))
    workdir = tmp_path / "ws" / "gates" / "core"
    gates.run_accuracy_gate(
        _scheduled_env(kind, world_size=4),
        task_data=_DEP_TASK,
        reference=95.0,
        runner=run,
        unit="core",
        member="dep4",
        workdir=workdir,
    )
    options = workdir / "core.dep4.llm_api.yaml"
    assert f'--extra_llm_api_options="{options}"' in run.seen[0]  # type: ignore[attr-defined]
    assert yaml.safe_load(options.read_text())["enable_attention_dp"] is True
    assert not any(orchestrator_tmp.iterdir())


@pytest.mark.parametrize("kind", _SCHEDULED_KINDS)
def test_a_scheduled_dep_member_without_a_workdir_refuses_the_temp_dir(kind, orchestrator_tmp):
    run = recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 95\n"))
    with pytest.raises(ValueError, match=f"no workdir.*`{kind}`"):
        gates.run_accuracy_gate(
            _scheduled_env(kind, world_size=4),
            task_data=_DEP_TASK,
            reference=95.0,
            runner=run,
            unit="core",
            member="dep4",
        )
    assert run.seen == [] and not any(orchestrator_tmp.iterdir())  # type: ignore[attr-defined]


def test_a_local_dep_member_without_a_workdir_keeps_the_temp_dir(orchestrator_tmp):
    """The local fallback is the temp dir, never the cwd."""
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import _write_member_options

    written = _write_member_options(
        None,
        {"enable_attention_dp": True},
        unit="core",
        member="dep4",
        environment=_env(),
        task_data=_DEP_TASK,
    )
    assert written == str(orchestrator_tmp / "core.dep4.llm_api.yaml")


def test_a_unit_options_file_still_hosts_its_members_overlay(tmp_path, orchestrator_tmp):
    """Beside the unit's file, whatever the workdir and the kind."""
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import _write_member_options

    unit_file = tmp_path / "scripts" / "core_llm_api.yaml"
    unit_file.parent.mkdir()
    unit_file.write_text("kv_cache_config: {free_gpu_memory_fraction: 0.5}\n", encoding="utf-8")
    written = _write_member_options(
        str(unit_file),
        {"enable_attention_dp": True},
        unit="core",
        member="dep4",
        workdir=tmp_path / "ws" / "gates" / "core",
        environment=_scheduled_env("session_spool"),
        task_data=_DEP_TASK,
    )
    assert written == str(unit_file.parent / "core.dep4.llm_api.yaml")


# ------------------------------------------------- the accuracy gate's options


def _options_named(command: str) -> dict:
    """The LLM-API options file a ``trtllm-eval`` command line names, loaded."""
    path = shlex.split(command.split("--extra_llm_api_options=", 1)[1])[0]
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", [gates.MODE_REQUIRE, gates.MODE_OFF])
def test_the_accuracy_gate_hands_its_own_mode_to_the_engine(mode, tmp_path):
    """Even when the run's environment does not export the switch itself."""
    unit_file = tmp_path / "core.llm_api.yaml"
    unit_file.write_text("max_num_tokens: 8192\n", encoding="utf-8")
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "llm_api_options": str(unit_file)}}
    run = recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 95\n"))
    assert gates.MODELING_V2_ENV not in _env().env
    gates.run_accuracy_gate(
        _env(), task_data=task, reference=95.0, runner=run, mode=mode, unit="core",
        member="tp1", workdir=tmp_path / "gates",
    )  # fmt: skip
    options = _options_named(run.seen[0])  # type: ignore[attr-defined]
    assert options["modeling_v2"] == mode and options["max_num_tokens"] == 8192


def test_member_options_are_replaced_whole(tmp_path, monkeypatch):
    """Concurrent gates of one member rewrite and read one file: never half of it."""
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import _write_member_options

    unit_file = tmp_path / "core.llm_api.yaml"
    unit_file.write_text("max_num_tokens: 8192\n", encoding="utf-8")
    renames: list[tuple[str, str]] = []
    real_replace = gates.os.replace

    def replace(source, target):
        renames.append((str(source), str(target)))
        real_replace(source, target)

    monkeypatch.setattr(gates.os, "replace", replace)
    written = _write_member_options(
        str(unit_file), {"enable_attention_dp": True}, unit="core", member="dep4"
    )
    [(source, target)] = renames
    assert target == written and Path(source).parent == Path(written).parent
    assert yaml.safe_load(Path(written).read_text())["enable_attention_dp"] is True

    def fail(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(gates.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        _write_member_options(str(unit_file), {"modeling_v2": "off"}, unit="core", member="dep4")
    assert "modeling_v2" not in yaml.safe_load(Path(written).read_text()), "the old file stands"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "core.dep4.llm_api.yaml",
        "core.llm_api.yaml",
    ], "no temporary file is left behind"
