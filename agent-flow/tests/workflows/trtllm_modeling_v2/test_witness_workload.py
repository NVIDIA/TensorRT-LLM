# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The workload a token-comparing witness declares, and the minimum length it is held to."""

from __future__ import annotations

import shlex

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates, witness
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

_LONG = [
    "Explain, step by step, how a heat pump moves heat.",
    "Write a 300-word story about a lighthouse keeper.",
]


def _task(block, **accuracy):
    return {
        "trtllm_repo_path": "/repo",
        "checkpoint_path": "/ckpt",
        "parallel": "tp1",
        "features": ["kv_quant"],
        "runtime": {"extra_flags": []},
        "accuracy": {"witness": {"kv_quant": block}, **accuracy},
    }


_KV = {"contrast_env": {"KV": "bf16"}, "expected_bytes_per_token": 890}


def _whole(ids_list):
    out = ""
    for i, ids in enumerate(ids_list):
        out += f"[{i}] Prompt: 'p'..., Generated text: 'x'...\n"
        out += f"[{i}] Witness token ids: {list(ids)}\n"
        out += f"[{i}] Witness text: 'x'\n"
    return out


def _legs(ids_list):
    seen = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        stored = 3200 if "KV=bf16" in command else 890
        return CommandResult(
            returncode=0,
            stdout=f"MODELING_V2_WITNESS kv_bytes_per_token={stored}\n" + _whole(ids_list),
        )

    run.seen = seen
    return run


def _run(task, ids_list):
    run = _legs(ids_list)
    result = witness.run_declared_witness(
        RunEnvironment(kind="local", python="/py", world_size=1),
        unit="kv_quant",
        member="tp1",
        task_data=task,
        runner=run,
        features=["kv_quant"],
    )
    return result, run


def _prompts_of(command):
    words = shlex.split(command)
    return words[words.index("--prompt") + 1 :]


# ---------------------------------------------------------- the workload


def test_the_block_declares_its_own_workload():
    """A token-comparing witness runs its block's own workload, not the boot prompts."""
    block = _KV | {"prompts": _LONG, "max_tokens": 300}
    result, run = _run(_task(block), [range(64), range(64)])
    assert result.passed, result.detail
    assert _prompts_of(run.seen[0]) == _LONG
    assert "--max_tokens=300" in run.seen[0]


def test_with_none_the_acceptance_prompts_are_the_workload():
    task = _task(_KV, acceptance={"prompts": _LONG[:1]})
    workload = witness.resolve_witness_workload(task, _KV, gates.BootSpec(), default_max_tokens=256)
    assert workload.prompts == tuple(_LONG[:1]) and workload.source == "accuracy.acceptance.prompts"
    plain = witness.resolve_witness_workload(
        _task(_KV), _KV, gates.BootSpec(), default_max_tokens=64
    )
    assert plain.prompts == gates.BootSpec().prompts and plain.max_tokens == 64
    assert plain.min_generated_tokens == witness.DEFAULT_MIN_GENERATED_TOKENS == 32


def test_short_continuations_fail_as_too_short_to_compare():
    """Identity over a 2-token answer rests on kernel certification, not a comparison."""
    result, _ = _run(_task(_KV), [range(3), range(2)])
    assert not result.passed
    assert "too short to compare" in result.detail and "shortest generated 2" in result.detail
    declared = _KV | {"min_generated_tokens": 2}
    result, _ = _run(_task(declared), [range(3), range(2)])
    assert result.passed, result.detail


def test_text_only_legs_have_no_length_and_cannot_pass_at_any():
    out = "[0] Prompt: 'p'..., Generated text: 'a'...\n"
    comparison = witness.compare_continuations(out, out)
    assert witness.too_short(comparison, 32) is None
    refusal = witness.missing_token_ids(comparison, (("a", out), ("b", out)), 32)
    assert refusal and "nor held to min_generated_tokens=32" in refusal


def test_the_minimum_holds_when_the_legs_print_no_token_ids():
    """A tree whose quickstart prints only the 200-character line, over 2-token answers.

    The prefixes agree, and so would any two answers that short: identity
    there rests on nothing end to end.
    """

    def run(command, env=None, timeout=0):
        stored = 3200 if "KV=bf16" in command else 890
        return CommandResult(
            returncode=0,
            stdout=f"MODELING_V2_WITNESS kv_bytes_per_token={stored}\n"
            "[0] Prompt: 'p'..., Generated text: ' 42'...\n",
        )

    result = witness.run_declared_witness(
        RunEnvironment(kind="local", python="/py", world_size=1),
        unit="kv_quant",
        member="tp1",
        task_data=_task(_KV),
        runner=run,
        features=["kv_quant"],
    )
    assert not result.passed
    assert "nor held to min_generated_tokens=32" in result.detail


# --------------------------------------------------------------- schema


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
        "features": ["kv_quant"],
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "r.yaml",
            "model_key": "M",
            "witness": {"kv_quant": block},
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_schema_accepts_a_witness_workload_and_resolves_its_file(tmp_path):
    (tmp_path / "long.yaml").write_text(yaml.safe_dump(_LONG), encoding="utf-8")
    block = _KV | {"prompts_file": "long.yaml", "max_tokens": 256, "min_generated_tokens": 64}
    loaded = load_and_validate_task_yaml(_spec(tmp_path, block))
    assert loaded["accuracy"]["witness"]["kv_quant"]["prompts_file"] == str(tmp_path / "long.yaml")


@pytest.mark.parametrize(
    "extra,message",
    [
        ({"min_generated_tokens": 0}, "min_generated_tokens"),
        ({"prompts": []}, "prompts"),
        ({"prompts": ["a"], "prompts_file": "x"}, "single source"),
    ],
)
def test_the_schema_refuses_a_malformed_witness_workload(tmp_path, extra, message):
    with pytest.raises(TaskSchemaError, match=message):
        load_and_validate_task_yaml(_spec(tmp_path, _KV | extra))


# ------------------------------------------------------------ every member


def _witnessed_run(tmp_path, task_file, monkeypatch, verdicts):
    """A unified host_offload run whose witness answers ``verdicts`` per member."""
    from .test_workflow import _run_unified, _unified_task, passing_shell

    _unified_task(task_file, ["tp1", "tep1", "dep1"])
    spec = yaml.safe_load(task_file.read_text())
    spec["features"] = ["host_offload"]
    spec["accuracy"]["witness"] = {"host_offload": {"contrast_env": {"X": "0"}, "expected_mib": 1}}
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    seen: list[str] = []

    def fake(environment, **kwargs):
        seen.append(kwargs["member"])
        return gates.GateResult(
            name="residency", passed=verdicts.get(kwargs["member"], True), detail="x"
        )

    monkeypatch.setattr(witness, "run_declared_witness", fake)
    workflow, _ = _run_unified(
        tmp_path, task_file, monkeypatch, passing_shell(), max_assembler_turns_per_unit=1
    )
    return seen


def test_expected_mib_may_differ_per_member():
    block = {"expected_mib": {"default": 48300, "dep4": 48344}}
    assert witness.resolve_expected_mib(block, "dep4") == 48344
    assert witness.resolve_expected_mib(block, "tep4") == 48300
    assert witness.resolve_expected_mib({"expected_mib": 7}, "tp4") == 7


def test_the_schema_accepts_expected_mib_per_member(tmp_path):
    path = _spec(tmp_path, _KV)
    data = yaml.safe_load(path.read_text())
    data["features"] = ["host_offload"]
    data["accuracy"]["witness"] = {
        "host_offload": {"contrast_env": {"X": "0"}, "expected_mib": {"default": 1, "dep4": 2}}
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    load_and_validate_task_yaml(path)
    data["accuracy"]["witness"]["host_offload"]["expected_mib"] = {"dep4": -1}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(TaskSchemaError, match="expected_mib"):
        load_and_validate_task_yaml(path)
