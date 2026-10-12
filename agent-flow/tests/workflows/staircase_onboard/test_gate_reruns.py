# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delivery can re-run every gate kind the unit loop fires, the new ones included."""

from __future__ import annotations

import pytest

from agent_flow.workflows.staircase_onboard import gates, launch_gate, step_check, witness
from agent_flow.workflows.staircase_onboard.environment import RunEnvironment
from agent_flow.workflows.staircase_onboard.freshness import RerunEntry

from .test_workflow import _workflow


@pytest.fixture
def wf(tmp_path):
    flow = _workflow(tmp_path)
    flow.task_data = {
        "trtllm_repo_path": str(tmp_path),
        "checkpoint_path": "/ckpt",
        "parallel": "tep4",
        "features": ["dspark", "host_offload"],
        "accuracy": {
            "benchmark": "gsm8k",
            "witness": {"host_offload": {"contrast_env": {"X": "0"}, "expected_mib": 1}},
        },
    }
    flow.environment = RunEnvironment(kind="local", python="/py", world_size=4)
    flow._gate_environment = flow.environment
    return flow


def _stub(monkeypatch, module, function, name):
    seen = {}

    def fake(environment, **kwargs):
        seen.update(kwargs)
        return gates.GateResult(name=name, passed=True, detail="ok", member=kwargs.get("member"))

    monkeypatch.setattr(module, function, fake)
    return seen


@pytest.mark.parametrize(
    "gate,module,function",
    [
        ("scheduler_equivalence", step_check, "run_scheduler_equivalence"),
        ("launch", launch_gate, "run_launch_gate"),
        ("acceptance", gates, "run_acceptance_gate"),
        ("boot", gates, "run_boot_gate"),
    ],
)
def test_the_new_gate_kinds_have_a_rerunner(wf, monkeypatch, gate, module, function):
    seen = _stub(monkeypatch, module, function, gate)
    run = wf.evidence.rerunner(RerunEntry(gate, "dspark", "tep4", ()))
    assert run is not None
    result = run()
    assert result.passed and wf.gate_log.last(gate) is result
    assert seen["unit"] == "dspark" and seen["member"] == "tep4"
    assert seen["features"] == ["dspark"]


def test_a_witness_reruns_at_the_member_it_was_measured_at(wf, monkeypatch):
    seen = _stub(monkeypatch, witness, "run_declared_witness", "residency")
    run = wf.evidence.rerunner(RerunEntry("residency", "host_offload", "dep4", ()))
    assert run is not None and run().passed
    assert seen["member"] == "dep4"


def test_the_statistical_floor_is_the_threshold_the_ledger_records():
    result = gates.GateResult(name="accuracy[gsm8k]", passed=True, detail="", floor=90.83)
    assert result.threshold == 90.83
