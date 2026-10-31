# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The statistical accuracy floor, the cross-member finding, and the statistics behind them."""

from __future__ import annotations

from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import gate_records
from agent_flow.workflows.trtllm_modeling_v2.checks import verdict as check_result

# A GSM8K anchor: 92.30 over 1319 questions.
_ANCHOR = 92.30
_TASK = {
    "checkpoint_path": "/ckpt/M",
    "parallel": "tep4",
    "trtllm_repo_path": "/repo",
    "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml", "tolerance": 5.0},
}


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
