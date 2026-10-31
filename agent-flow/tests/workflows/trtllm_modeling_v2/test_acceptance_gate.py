# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The acceptance gate's per-prompt comparisons, against stock and against the previous tree."""

from __future__ import annotations

import json
import math

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import acceptance as acceptance_checks
from agent_flow.workflows.trtllm_modeling_v2.checks.gate_records import GateRecords
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment

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
    """What the engine driver prints for prompts accepting at ``rates``.

    ``iterations`` is every prompt's, or a list with one per prompt.
    """
    lines = []
    for n, rate in enumerate(rates):
        steps = iterations[n] if isinstance(iterations, list) else iterations
        tokens = int(round(rate * steps))
        workload_io_line = {
            "index": n,
            "id": f"d{n}",
            "prompt_sha": (prompts or [f"p{i}" for i in range(len(rates))])[n],
            "tokens": tokens,
            "iterations": steps,
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
    monkeypatch.setattr(acceptance_checks, "inspect_tree", lambda task_data: current["name"])
    return current


def test_a_significant_drop_against_the_previous_tree_fails(tmp_path, tree):
    records = GateRecords(tmp_path / "r.jsonl")
    common = dict(task_data=_task(), member="tep4", records=records, workdir=tmp_path / "w")
    acceptance_checks.run_acceptance_gate(
        _env(), runner=_legs(_driver_output(_BASE), _driver_output(_BASE)), **common
    )
    tree["name"] = "treeB"
    dropped = [r - 0.3 for r in _BASE]
    result = acceptance_checks.run_acceptance_gate(
        _env(), runner=_legs(_driver_output(dropped), _driver_output(dropped)), **common
    )
    assert not result.passed
    assert "against tree treeA: REGRESSED" in result.detail
    # A failed verdict is recorded, but is not the next tree's baseline.
    tree["name"] = "treeC"
    again = acceptance_checks.run_acceptance_gate(
        _env(), runner=_legs(_driver_output(_BASE), _driver_output(_BASE)), **common
    )
    assert again.passed and "against tree treeA" in again.detail
