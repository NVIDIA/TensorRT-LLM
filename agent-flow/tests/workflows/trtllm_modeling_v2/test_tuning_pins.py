# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tuning a process does for itself, shared by the legs of a gate that compares them.

A leg is a process of its own, and a ``@triton.autotune`` kernel picks its
config by timing in each one. Two legs of one configuration then differ in
the last bit, and a greedy comparison reads that as the feature's defect.
This file holds the helper and the token-comparing witnesses.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.checks import gates, witness
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult

_TASK = {
    "trtllm_repo_path": "/repo",
    "checkpoint_path": "/ckpt",
    "parallel": "tp1",
    "features": [],
    "runtime": {"extra_flags": []},
}

_PINNED = (
    gates.AUTOTUNER_CACHE_ENV,
    gates.TRITON_CACHE_AUTOTUNING_ENV,
    gates.TRITON_CACHE_DIR_ENV,
)


def _env(**env):
    return RunEnvironment(kind="local", python="/py", world_size=1, env=dict(env))


def _assigned(command: str) -> dict[str, str]:
    """Every ``NAME=value`` the command exports, the last one winning."""
    return dict(
        word.split("=", 1)
        for word in shlex.split(command)
        if "=" in word and word.split("=", 1)[0].isidentifier()
    )


def _pins(command: str) -> dict[str, str | None]:
    exported = _assigned(command)
    return {name: exported.get(name) for name in _PINNED}


def _expected(directory: Path) -> dict[str, str]:
    tuning = directory / "tuning"
    return {
        gates.AUTOTUNER_CACHE_ENV: str(tuning / "autotuner.cache"),
        gates.TRITON_CACHE_AUTOTUNING_ENV: "1",
        gates.TRITON_CACHE_DIR_ENV: str(tuning / "triton-cache"),
    }


def _whole(ids_list, bytes_per_token=None) -> str:
    out = ""
    if bytes_per_token is not None:
        out += f"{witness.WITNESS_COUNTER_MARKER} kv_bytes_per_token={bytes_per_token}\n"
    for i, ids in enumerate(ids_list):
        out += f"[{i}] Prompt: 'p'..., Generated text: 'x'...\n"
        out += f"[{i}] Witness token ids: {list(ids)}\n"
        out += f"[{i}] Witness text: 'x'\n"
    out += f"{witness.DEVICE_MEMORY_MARKER}: 1000\n"
    return out


def _recording(stdout):
    calls: list[str] = []

    def run(command, env=None, timeout=0):
        calls.append(command)
        return CommandResult(returncode=0, stdout=stdout(command) if callable(stdout) else stdout)

    run.calls = calls
    return run


_IDS = [list(range(64)), list(range(1, 65))]


# ------------------------------------------------------------------ the helper


def test_pin_tuning_clears_a_previous_runs_choices_and_keeps_compiled_kernels(tmp_path):
    first = gates.pin_tuning(tmp_path)
    assert first == _expected(tmp_path)
    triton = Path(first[gates.TRITON_CACHE_DIR_ENV])
    assert triton.is_dir()
    tuned = triton / "KEY" / "chunk_fwd_kernel_o.autotune.json"
    compiled = triton / "KEY" / "chunk_fwd_kernel_o.cubin"
    tuned.parent.mkdir()
    tuned.write_text("{}", encoding="utf-8")
    compiled.write_text("binary", encoding="utf-8")
    Path(first[gates.AUTOTUNER_CACHE_ENV]).write_text("{}", encoding="utf-8")
    assert gates.triton_tuning_note(first) == (
        "Triton autotuning shared by the legs (1 tuning result(s) cached)"
    )

    assert gates.pin_tuning(tmp_path) == first
    assert not tuned.exists() and not Path(first[gates.AUTOTUNER_CACHE_ENV]).exists()
    assert compiled.exists()
    assert gates.triton_tuning_note(first).startswith("no Triton autotuning was cached")
    assert gates.triton_tuning_note({}) == ""


# ---------------------------------------------------------------- the witnesses


def test_both_residency_legs_share_tuning_under_the_witness_workdir(tmp_path):
    run = _recording(_whole(_IDS))
    result = witness.run_residency_witness(
        _env(TRITON_CACHE_DIR="/task/triton-cache", TRITON_CACHE_AUTOTUNING="0"),
        task_data=_TASK,
        expected_mib=0.0,
        contrast_env={"OFFLOAD": "0"},
        runner=run,
        member="tp1",
        workdir=tmp_path,
    )
    assert result.passed, result.detail
    assert len(run.calls) == 2
    assert [_pins(call) for call in run.calls] == [_expected(tmp_path / "tp1")] * 2


def test_both_kv_quant_legs_share_tuning_and_a_lossy_one_leg_run_pins_nothing(tmp_path):
    run = _recording(lambda command: _whole(_IDS, 3200 if "KV=bf16" in command else 890))
    common = dict(
        task_data=_TASK,
        expected_bytes_per_token=890,
        contrast_env={"KV": "bf16"},
        runner=run,
        member="tp1",
        workdir=tmp_path,
    )
    result = witness.run_kv_quant_witness(_env(), **common)
    assert result.passed, result.detail
    assert [_pins(call) for call in run.calls] == [_expected(tmp_path / "tp1")] * 2

    run.calls.clear()
    result = witness.run_kv_quant_witness(_env(), exact=False, **common)
    assert result.passed and len(run.calls) == 1
    assert not any(_pins(run.calls[0]).values())


def _hop(command: str) -> str:
    """A leg's output; leg B's self-test run destroys its state, and its one continuation moves."""
    if f"{witness.LEG_B_ABLATE_ENV}={witness.ABLATE_STATE}" in command:
        return _whole([list(range(7, 71))]) + "Witness ablated: conv, ssm\n"
    return _whole(_IDS)


def test_both_equivalence_legs_and_leg_bs_engines_share_tuning(tmp_path):
    """Leg B's launcher re-applies ``legs_env`` to its engines, so a pin wins there too."""
    run = _recording(_hop)
    legs_env = {"TRITON_CACHE_DIR": "/declared/triton-cache", "REPLAY": "0"}
    result = witness.run_equivalence_witness(
        _env(),
        task_data=_TASK,
        contrast_env={"DISAGG": "1"},
        contrast_command="/onboard/leg_b.py",
        runner=run,
        member="tp1",
        workdir=tmp_path,
        legs_env=legs_env,
    )
    assert result.passed, result.detail
    expected = _expected(tmp_path / "tp1")
    agg, dis, ablated = run.calls
    assert _pins(agg) == expected and _pins(dis) == expected and _pins(ablated) == expected
    handed = json.loads(_assigned(dis)[witness.LEG_B_LEGS_ENV_ENV])
    assert handed == {"TRITON_CACHE_DIR": expected[gates.TRITON_CACHE_DIR_ENV], "REPLAY": "0"}

    run.calls.clear()
    witness.run_equivalence_witness(
        _env(),
        task_data=_TASK,
        contrast_env={"DISAGG": "1"},
        runner=run,
        member="tp1",
        workdir=tmp_path,
    )
    assert [_pins(call) for call in run.calls] == [expected] * 3


def test_each_witness_run_tunes_afresh_in_its_own_directory(tmp_path):
    """Members and units do not share a cache, and no run inherits the last one's choices."""
    seen_at_first_leg: list[bool] = []
    stale = tmp_path / "residency" / "tp1" / "tuning" / "triton-cache" / "K" / "k.autotune.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")

    def run(command, env=None, timeout=0):
        if not seen_at_first_leg:
            seen_at_first_leg.append(stale.exists())
        run.calls.append(command)
        return CommandResult(returncode=0, stdout=_whole(_IDS))

    run.calls = []
    for member in ("tp1", "tp2"):
        witness.run_residency_witness(
            _env(),
            task_data=_TASK,
            expected_mib=0.0,
            contrast_env={"OFFLOAD": "0"},
            runner=run,
            member=member,
            workdir=tmp_path / "residency",
        )
    witness.run_residency_witness(
        _env(),
        task_data=_TASK,
        expected_mib=0.0,
        contrast_env={"OFFLOAD": "0"},
        runner=run,
        member="tp1",
        workdir=tmp_path / "other_unit",
    )
    assert seen_at_first_leg == [False]
    caches = [_pins(call)[gates.TRITON_CACHE_DIR_ENV] for call in run.calls]
    assert caches == [
        *[str(tmp_path / "residency" / "tp1" / "tuning" / "triton-cache")] * 2,
        *[str(tmp_path / "residency" / "tp2" / "tuning" / "triton-cache")] * 2,
        *[str(tmp_path / "other_unit" / "tp1" / "tuning" / "triton-cache")] * 2,
    ]


def test_the_declared_witness_pins_under_the_workdir_it_is_given(tmp_path):
    task = dict(_TASK) | {
        "features": ["kv_quant"],
        "accuracy": {
            "witness": {
                "kv_quant": {"contrast_env": {"KV": "bf16"}, "expected_bytes_per_token": 890}
            }
        },
    }
    run = _recording(lambda command: _whole(_IDS, 3200 if "KV=bf16" in command else 890))
    result = witness.run_declared_witness(
        _env(),
        unit="kv_quant",
        member="tp1",
        task_data=task,
        runner=run,
        features=["kv_quant"],
        workdir=tmp_path / "witness" / "kv_quant",
    )
    assert result.passed, result.detail
    expected = _expected(tmp_path / "witness" / "kv_quant" / "tp1")
    assert [_pins(call) for call in run.calls] == [expected] * 2
