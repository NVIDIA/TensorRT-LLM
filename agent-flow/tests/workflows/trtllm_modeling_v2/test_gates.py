# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gates — the commands whose exit codes are the verdicts."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
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
    "member,expected",
    [
        ("tp4", ["--tp_size=4"]),
        ("tep4", ["--tp_size=4", "--moe_ep_size=4"]),
        ("dep4", ["--tp_size=4", "--moe_ep_size=4", "--enable_attention_dp"]),
    ],
)
def test_a_member_is_what_gets_translated(member, expected):
    assert gates.topology_flags("unified", world_size=4, member=member) == expected


def test_attention_dp_is_not_a_trtllm_eval_flag():
    """`trtllm-eval --help` lists tp/pp/ep and nothing for attention DP."""
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


# ------------------------------------------------------------ score parsing


def test_the_final_score_wins_over_partial_estimates():
    out = (
        "Partial scores after 100/1319 responses (estimate, 0~100): gsm8k,acc ~ 12.00\n"
        "Partial scores after 900/1319 responses (estimate, 0~100): gsm8k,acc ~ 80.00\n"
        "lm-eval gsm8k exact_match,flexible-extract accuracy: 90.67\n"
    )
    assert gates.parse_eval_score(out) == pytest.approx(90.67)


# -------------------------------------------------------------- boot gate


def test_boot_gate_passes_on_clean_exit_with_output():
    run = recorder(CommandResult(returncode=0, stdout=_PARIS))
    result = gates.run_boot_gate(_env(), task_data=_TASK, runner=run)
    assert result.passed, result.detail
    assert "quickstart_advanced.py" in result.command
    assert str(gates.ENGINE_DRIVER) in result.command
    assert "TRTLLM_MODELING_V2=require" in result.command


def test_boot_gate_fails_on_nonzero_exit():
    run = recorder(CommandResult(returncode=1, stderr="CUDA error"))
    result = gates.run_boot_gate(_env(), task_data=_TASK, runner=run)
    assert not result.passed and "exited 1" in result.detail


def test_a_keyword_in_the_log_is_not_a_keyword_in_the_text():
    """`72` matches `CPUs 72-143` in the engine log, and the prompt is echoed too."""
    spec = gates.BootSpec(prompts=("48 + 24 = ? Answer 72 if unsure.",), keywords=("72",))
    log = "[TRT-LLM] [I] Process 1 bound to CPUs 72-143; prompt '48 + 24 = ? Answer 72'"
    run = recorder(CommandResult(returncode=0, stdout=_gen("I do not know", extra=log)))
    result = gates.run_boot_gate(_env(), task_data=_TASK, spec=spec, runner=run)
    assert not result.passed and "'72'" in result.detail
    run = recorder(CommandResult(returncode=0, stdout=_gen("The answer is 72.", extra=log)))
    assert gates.run_boot_gate(_env(), task_data=_TASK, spec=spec, runner=run).passed


# ------------------------------------------------------- a lost session

_LOST = "the session watching /spool stopped heart-beating 412s ago; nothing was queued"


def test_a_gate_lost_to_its_session_says_so_rather_than_timed_out():
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


# ----------------------------------------------------------- accuracy gate


def test_accuracy_gate_with_no_parsable_score_is_not_a_pass():
    run = recorder(CommandResult(returncode=0, stdout="done\n"))
    result = gates.run_accuracy_gate(_env(), task_data=_TASK, reference=89.99, runner=run)
    assert not result.passed and "no score" in result.detail


def _options_named(command: str) -> dict:
    """The LLM-API options file a ``trtllm-eval`` command line names, loaded."""
    path = shlex.split(command.split("--extra_llm_api_options=", 1)[1])[0]
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_the_accuracy_gate_hands_require_to_the_engine(tmp_path):
    """Even when the run's environment does not export the switch itself."""
    unit_file = tmp_path / "core.llm_api.yaml"
    unit_file.write_text("max_num_tokens: 8192\n", encoding="utf-8")
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "llm_api_options": str(unit_file)}}
    run = recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 95\n"))
    assert gates.MODELING_V2_ENV not in _env().env
    gates.run_accuracy_gate(
        _env(), task_data=task, reference=95.0, runner=run, unit="core", member="tp1",
        workdir=tmp_path / "gates",
    )  # fmt: skip
    options = _options_named(run.seen[0])  # type: ignore[attr-defined]
    assert options["modeling_v2"] == gates.MODE_REQUIRE and options["max_num_tokens"] == 8192


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
    """A task whose acceptance block carries a floor; ``min_prompts`` defaults to 1 here."""
    acceptance.setdefault("min_prompts", 1)
    return {
        **_TASK,
        "accuracy": {
            **_TASK["accuracy"],
            "acceptance": {"dspark": {"min_absolute": 2.0}, **acceptance},
        },
    }


def test_a_short_answer_workload_fails_as_too_short_not_as_a_low_number():
    """Short-answer boot prompts read 1.00 whatever the drafter does."""
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
    assert result.measured is None
    assert "[2] 1.00 (3 tok/3 it)" in result.summary()


def test_per_prompt_acceptance_is_reported_beside_the_mean():
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


def test_the_draft_depth_comes_from_the_options_yaml(tmp_path):
    """All three gates must ask for the depth the unit was built for."""
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import (
        resolve_max_draft_len,
        speculation_flags,
    )

    options = tmp_path / "mtp_llm_api.yaml"
    options.write_text(
        yaml.safe_dump(
            {"speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 3}}
        )
    )
    accuracy = {"llm_api_options": {"mtp": str(options)}}

    assert resolve_max_draft_len(accuracy, "mtp") == 3
    assert speculation_flags(["mtp"], max_draft_len=3) == [
        "--spec_decode_algo=MTP",
        "--spec_decode_max_draft_len=3",
    ]


# --------------------------------------------------- the dirty-target guard


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
    """A real one-commit repo holding a modeling_v2 family directory."""
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


def test_a_dirty_target_path_stops_the_gate_before_it_spends_anything(tmp_path):
    """At gate time the assembler has already committed; dirty means another writer."""
    repo = _repo_with_target(tmp_path)
    modeling = repo / gates.MODELING_V2_PACKAGE / "models" / "gpt_oss" / "modeling.py"
    modeling.write_text("# half-written by somebody else\n", encoding="utf-8")

    run = recorder(CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 90\n"))
    result = gates.run_accuracy_gate(_env(), task_data=_task_for(repo), reference=89.99, runner=run)
    assert not result.passed
    assert "modeling.py" in result.detail
    assert run.seen == []  # type: ignore[attr-defined]


# ---------------------------------------------------------- member ordering


def test_members_are_ordered_cheapest_first():
    assert gates.order_members_by_cost(["tp4", "dep4", "tep4"]) == ["tep4", "dep4", "tp4"]
