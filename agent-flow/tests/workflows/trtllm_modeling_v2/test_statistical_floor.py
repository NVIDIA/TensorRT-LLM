# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The statistical accuracy floor, the cross-member finding, and the statistics behind them."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy_policy, gate_records, stats
from agent_flow.workflows.trtllm_modeling_v2.checks import verdict as check_result
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment

# A GSM8K anchor: 92.30 over 1319 questions.
_ANCHOR = 92.30
_TASK = {
    "checkpoint_path": "/ckpt/M",
    "parallel": "tep4",
    "trtllm_repo_path": "/repo",
    "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml", "tolerance": 5.0},
}


def _env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/py", "env": {}}}, devices=(0,), world_size=1
    )


def _eval(score: float):
    def run(command, env=None, timeout=0):
        return CommandResult(returncode=0, stdout=f"lm-eval gsm8k average accuracy: {score}\n")

    return run


def test_the_gsm8k_standard_error_is_three_quarters_of_a_point():
    """Anchor - 2 SE is ~1.5 points for GSM8K-1319, where anchor - 5 is ~7 SE."""
    se = stats.binomial_se(_ANCHOR, 1319)
    assert se == pytest.approx(0.734, abs=1e-3)
    floor = accuracy_policy.statistical_floor(_ANCHOR, 1319)
    assert _ANCHOR - floor.value == pytest.approx(1.47, abs=0.01)
    assert (5.0 / se) > 6.5


# ----------------------------------------------- the paired acceptance comparison


def test_a_regression_is_a_drop_both_significant_and_large():
    baseline = [2.6 + 0.3 * math.sin(i) for i in range(32)]
    candidate = [b - 0.5 + 0.05 * math.cos(i) for i, b in enumerate(baseline)]
    result = stats.paired_test(baseline, candidate, min_effect=0.1)
    assert result.significant and result.regressed
    assert "paired over 32 prompt(s)" in result.describe()
    # A tiny but consistent drop on a large workload is below the minimum effect.
    baseline = [3.0 + 0.1 * (i % 5) for i in range(200)]
    result = stats.paired_test(baseline, [b - 0.01 for b in baseline], min_effect=0.1)
    assert result.significant and not result.regressed


# ---------------------------------------------------------- the accuracy gate


def test_the_floor_is_statistical_by_default(tmp_path):
    """With no rule stated the floor is anchor - 2 SE, not a fixed tolerance."""

    def gate(score):
        return accuracy_checks.run_accuracy_gate(
            _env(), task_data=_TASK, reference=_ANCHOR, runner=_eval(score), workdir=tmp_path
        )

    result = gate(90.0)
    assert not result.passed
    assert result.floor == pytest.approx(_ANCHOR - 2 * 0.734, abs=0.01)
    assert result.se == pytest.approx(0.734, abs=1e-3) and result.samples == 1319
    assert "2.00 x SE 0.73 (n=1319)" in result.detail
    assert "SE 0.73" in result.summary() and "floor 90.83" in result.summary()
    assert gate(91.2).passed  # inside two SE


# ----------------------------------------------------- the cross-member finding


def _verdict(member, measured, reference=_ANCHOR, se=0.734):
    return check_result.GateResult(
        name="accuracy[gsm8k]",
        passed=True,
        detail="",
        member=member,
        measured=measured,
        reference=reference,
        se=se,
    )


def test_every_member_below_one_se_is_a_finding():
    results = [_verdict("tep4", 91.4), _verdict("dep4", 91.5), _verdict("tp4", 91.2)]
    finding = accuracy_checks.check_systematic_shift(results, _TASK["accuracy"], "dspark")
    assert finding is not None and not finding.acknowledged
    assert "every member is below the anchor" in finding.describe()
    assert "accuracy.floor.acknowledge.dspark" in finding.describe()


def test_a_finding_blocks_delivery_until_acknowledged_or_resolved(tmp_path):
    records = gate_records.records_for(tmp_path)
    shifted = [_verdict("tep4", 91.4), _verdict("tp4", 91.2)]
    accuracy_checks.note_systematic_shift(records, shifted, _TASK["accuracy"], "dspark")
    blocking = accuracy_checks.blocking_findings(records, _TASK["accuracy"])
    assert len(blocking) == 1 and blocking[0].key == {
        "finding": "systematic_shift",
        "unit": "dspark",
    }
    # Acknowledged in the CURRENT task: no longer blocking.
    acknowledged = {**_TASK["accuracy"], "floor": {"acknowledge": {"dspark": "measured noise"}}}
    assert accuracy_checks.blocking_findings(records, acknowledged) == []
    # A later clean gating of the same unit resolves it.
    clean = [_verdict("tep4", 92.4), _verdict("tp4", 91.2)]
    accuracy_checks.note_systematic_shift(records, clean, _TASK["accuracy"], "dspark")
    assert accuracy_checks.blocking_findings(records, _TASK["accuracy"]) == []


def test_delivery_refuses_open_findings_first(tmp_path):
    """Before the delivery report, and as a remediable verdict a relaunch re-enters."""
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import completion, contract

    from .test_workflow import _workflow

    wf = _workflow(tmp_path)
    wf.task_data = {"accuracy": dict(_TASK["accuracy"])}
    accuracy_checks.note_systematic_shift(
        gate_records.records_for(wf.workspace),
        [_verdict("tep4", 91.0), _verdict("tp4", 91.1)],
        wf.task_data["accuracy"],
        "dspark",
    )
    closed = []
    wf._close_run = lambda state, log, verdict: closed.append(verdict)
    wf._contract = contract.Contract(checks=())
    wf.environment = SimpleNamespace(world_size=1)
    wf._target_marker = lambda: ""
    verdict = contract.evaluate(wf._contract, wf._contract_status(), marker="")
    assert (
        completion.run_deliver_phase(
            wf,
            SimpleNamespace(units_done=["core"]),
            None,
            lambda files: completion.assess(verdict, files=files),
        )
        is False
    )
    assert closed == ["delivery-failed:findings"]
    assert "delivery blocked" in wf.journal.path.read_text(encoding="utf-8")
