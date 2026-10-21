# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The acceptance gate's per-prompt comparisons, against stock and against the previous tree."""

from __future__ import annotations

import json
import math

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.checks.gate_records import GateRecords
from agent_flow.workflows.trtllm_modeling_v2.checks.gates import TreeState
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult

_TASK = {
    "checkpoint_path": "/ckpt/M",
    "parallel": "tep4",
    "trtllm_repo_path": "/repo",
    "features": ["dspark"],
    "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml"},
}


def _env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/py", "env": {}}}, devices=(0,), world_size=1
    )


def _driver_output(rates, *, prompts=None, iterations=100):
    """What the engine driver prints for prompts accepting at ``rates``."""
    lines = []
    for n, rate in enumerate(rates):
        tokens = int(round(rate * iterations))
        workload_io_line = {
            "index": n,
            "id": f"d{n}",
            "prompt_sha": (prompts or [f"p{i}" for i in range(len(rates))])[n],
            "tokens": tokens,
            "iterations": iterations,
            "token_ids": [],
            "text": "",
        }
        lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(workload_io_line)}")
    return "\n".join(lines) + "\n"


def _legs(require, off=None):
    """A runner: the require leg prints ``require``; the off leg ``off`` or cannot load."""
    seen = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        if "TRTLLM_MODELING_V2=off" in command:
            if off is None:
                return CommandResult(returncode=1, stderr="Unknown model architecture: X")
            return CommandResult(returncode=0, stdout=off)
        return CommandResult(returncode=0, stdout=require)

    run.seen = seen
    return run


def _task(**acceptance):
    return {**_TASK, "accuracy": {**_TASK["accuracy"], "acceptance": acceptance}}


_BASE = [2.6 + 0.5 * math.sin(i) for i in range(32)]


@pytest.fixture
def tree(monkeypatch):
    """Pin the tree a gate believes it measured, so a baseline can come from another."""
    current = {"name": "treeA"}

    def fake(task_data, *, allow_dirty=None):
        return TreeState(description=current["name"])

    monkeypatch.setattr(gates, "inspect_tree", fake)
    return current


def test_a_drop_against_stock_that_the_paired_test_finds_fails():
    worse = [r - 0.4 for r in _BASE]
    result = gates.run_acceptance_gate(
        _env(), task_data=_task(), runner=_legs(_driver_output(worse), _driver_output(_BASE))
    )
    assert not result.passed
    assert "paired over 32 prompt(s)" in result.detail
    assert "subtly wrong" in result.detail


def test_a_ratio_under_the_floor_that_is_noise_passes():
    """Under 0.95 on the weighted mean, but the per-prompt test cannot tell it from noise."""
    base = [b + 1.0 for b in _BASE]
    noisy = [b + (1.4 if i % 2 == 0 else -1.8) for i, b in enumerate(base)]
    paired = gates.RegressionPolicy().compare(base, noisy)
    assert not paired.significant
    result = gates.run_acceptance_gate(
        _env(), task_data=_task(), runner=_legs(_driver_output(noisy), _driver_output(base))
    )
    weighted = sum(noisy) / sum(base)
    assert weighted < 0.95
    assert result.passed, result.detail


def test_a_significant_drop_against_the_previous_tree_fails(tmp_path, tree):
    records = GateRecords(tmp_path / "r.jsonl")
    common = dict(task_data=_task(), member="tep4", records=records)
    gates.run_acceptance_gate(
        _env(), runner=_legs(_driver_output(_BASE), _driver_output(_BASE)), **common
    )
    tree["name"] = "treeB"
    dropped = [r - 0.3 for r in _BASE]
    result = gates.run_acceptance_gate(
        _env(), runner=_legs(_driver_output(dropped), _driver_output(dropped)), **common
    )
    assert not result.passed
    assert "against tree treeA: REGRESSED" in result.detail
    # A failed verdict is recorded, but is not the next tree's baseline.
    tree["name"] = "treeC"
    again = gates.run_acceptance_gate(
        _env(), runner=_legs(_driver_output(_BASE), _driver_output(_BASE)), **common
    )
    assert again.passed and "against tree treeA" in again.detail
