# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight's refusals: the checks that decide whether the run happens.

Every probe is injected, so none of this needs a GPU.
"""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight
from agent_flow.workflows.trtllm_modeling_v2.run.startup.preflight import PreflightError


def _smi_rows(*rows: str) -> str:
    return "\n".join(rows) + "\n"


def _healthy_row(index: int, cap: str = "10.3") -> str:
    return f"{index}, 180000, 183359, 0, {cap}"


def make_runner(*, smi: str, probe_ok=lambda index: True):
    """Build a runner that answers nvidia-smi, the lone probe, and torchrun."""

    def runner(argv, env=None, timeout=0):
        joined = " ".join(argv)
        if argv and argv[0] == "nvidia-smi":
            return CommandResult(returncode=0, stdout=smi)
        if argv and argv[0] == "torchrun":
            return CommandResult(returncode=0, stdout="[preflight] OK\n")
        if "torch.cuda.set_device" in joined:
            index = int((env or {}).get("CUDA_VISIBLE_DEVICES", "0"))
            if probe_ok(index):
                return CommandResult(returncode=0, stdout="ok\n")
            return CommandResult(returncode=1, stderr="cudaErrorDevicesUnavailable")
        raise AssertionError(f"unexpected command: {argv}")

    return runner


# A declared scratch location that is not under /tmp. It is never created:
# `check_tmpdir` reads the declared value, it does not stat it — and pytest's
# own `tmp_path` lives under /tmp on some hosts, which is exactly what the
# check rejects.
_SCRATCH = "/scratch/mv2flow-run-tmp"


def _task(tmp_path, **overrides):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["Qwen3MoeForCausalLM"]}))
    task = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(tmp_path),
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": _SCRATCH, "CLAUDE_CODE_TMPDIR": _SCRATCH},
        },
    }
    task.update(overrides)
    return task


# -------------------------------------------------------------- end to end


def test_run_preflight_resolves_identity(tmp_path):
    task = _task(tmp_path, parallel="dep4")
    (tmp_path / "ckpt").rename(tmp_path / "Qwen3-30B-A3B")
    task["checkpoint_path"] = str(tmp_path / "Qwen3-30B-A3B")
    runner = make_runner(smi=_smi_rows(*[_healthy_row(i) for i in range(4)]))
    result = preflight.run_preflight(task, runner=runner)
    assert result.slug == "qwen3-30b-a3b-sm_103-dep4"
    assert result.branch == "trtllm-modeling-v2/qwen3-30b-a3b-sm_103-dep4"
    assert result.devices == (0, 1, 2, 3)
    assert result.world_size == 4
    assert result.checkpoint_config["architectures"] == ["Qwen3MoeForCausalLM"]


# --------------------------------------------------------------- a placeholder anchor


def test_a_relative_variant_needs_its_units_benchmark_and_disagg_one_node(tmp_path):
    fp8 = {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}, "members": ["dep4"]}
    task = _task(
        tmp_path,
        parallel="unified",
        members=["tp4", "dep4"],
        accuracy={"benchmark": "gsm8k", "reference_yaml": "x.yaml", "benchmark_members": "tp4"},
        variants={"kv_fp8": {**fp8, "gates": ["accuracy"]}},
    )
    with pytest.raises(PreflightError, match="`core`'s benchmark at none of dep4"):
        preflight.check_variant_anchors(task, 4)
    del task["accuracy"]["benchmark_members"]
    assert "relative, core's own score at dep4" in preflight.check_variant_anchors(task, 4)[0]
    task["features"] = ["disagg"]
    task["environment"]["gpus_per_node"] = 4
    task["accuracy"]["disagg"] = {"test": "tests/test_d.py::TestM::test_auto_dtype", "devices": 8}
    with pytest.raises(PreflightError, match="more than the 4 devices one node holds"):
        preflight.check_disagg_placement(task)


def test_a_gate_longer_than_one_gates_allocation_is_refused_before_it_waits_for_ever():
    task = {
        "environment": {
            "kind": "session_spool",
            "sessions": [
                {"name": "gates", "spool": "/g", "role": "gates", "time_limit": "02:00:00"},
                {"name": "work", "spool": "/w", "role": "work", "time_limit": "08:00:00"},
            ],
        },
        "accuracy": {},
    }
    with pytest.raises(preflight.PreflightError, match="accuracy.timeout is 14400s"):
        preflight.check_gate_budgets(task)
    task["accuracy"]["timeout"] = 5400
    (note,) = preflight.check_gate_budgets(task)
    assert "accuracy.timeout 5400s" in note and "7200s" in note
    assert preflight.check_gate_budgets({"environment": {"kind": "local"}}) == []
