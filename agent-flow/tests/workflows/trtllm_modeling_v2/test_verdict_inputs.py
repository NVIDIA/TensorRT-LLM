# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A verdict is about what its check read: the checkout, and every input outside it."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.checks import acceptance as acceptance_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import test_results
from agent_flow.workflows.trtllm_modeling_v2.checks.delivery import Check as DeliveryCheck
from agent_flow.workflows.trtllm_modeling_v2.checks.delivery import DeliveryReport
from agent_flow.workflows.trtllm_modeling_v2.checks.gate_records import (
    KIND_STOCK_UNAVAILABLE,
    GateRecords,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run.validation import regression
from agent_flow.workflows.trtllm_modeling_v2.run.validation.completion import (
    tests_check as delivered_tests,
)

from .test_acceptance_gate import _driver_output, _env, _legs, _task


def _result(name="boot", member="tp1"):
    return SimpleNamespace(name=name, member=member, passed=True, detail="ok", tree_state="t")


def test_a_verdict_on_other_inputs_is_stale(tmp_path):
    contract = c.Contract(checks=(c.Check("core", "boot", "tp1", c.TIER_INNER, inputs="new"),))
    status = c.ContractStatus.load(tmp_path)
    status.record("core", _result(), marker="tree:1", inputs="old")
    assert c.evaluate(contract, status, marker="tree:1").missing == contract.checks
    status.record("core", _result(), marker="tree:1", inputs="new")
    assert c.evaluate(contract, status, marker="tree:1").holds
    reloaded = c.ContractStatus.load(tmp_path)
    assert reloaded.verdicts[-1].inputs == "new"
    # A verdict recorded before verdicts carried inputs counts as the check's.
    legacy = c.ContractStatus.load(tmp_path / "legacy")
    legacy.record("core", _result(), marker="tree:1")
    assert c.evaluate(contract, legacy, marker="tree:1").holds


def test_the_stock_leg_is_asked_once_per_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance_checks, "inspect_tree", lambda task_data: "t")
    task = _task(dspark={"min_absolute": 1.5, "floor_source": "a measured drafter"})
    rates = [2.6 + 0.5 * math.sin(i) for i in range(32)]
    records = GateRecords(tmp_path / "gate_results.jsonl")
    common = dict(task_data=task, records=records, unit="dspark", workdir=tmp_path / "w")
    first = _legs(_driver_output(rates))
    ran = acceptance_checks.run_acceptance_gate(_env(), runner=first, **common)
    assert ran.passed, ran.detail
    assert any("=off" in cmd for cmd in first.seen)
    assert records.all(KIND_STOCK_UNAVAILABLE)
    again = _legs(_driver_output(rates))
    result = acceptance_checks.run_acceptance_gate(_env(), runner=again, **common)
    assert result.passed and not any("=off" in cmd for cmd in again.seen)
    assert "not run again" in result.detail


class _Delivery:
    def __init__(self, tmp_path: Path, head: str = "c0ffee1234567890"):
        self.task_data = {"trtllm_repo_path": str(tmp_path)}
        self.workspace = tmp_path / "ws"
        self._preflight = SimpleNamespace(target_rel="pkg/models/fam/target")
        self._git = lambda repo, *args: head + "\n"


def test_delivery_runs_the_delivered_tests_at_head_and_holds_on_their_passing(
    tmp_path, monkeypatch
):
    report = DeliveryReport(
        checks=(DeliveryCheck("test:gemm/x", True, str(tmp_path / "tests/test_modeling_v2_x.py")),)
    )
    test = "tests/test_modeling_v2_x.py"
    asked = {}

    def list_test_files(repo, head, paths):
        asked["paths"] = list(paths)
        return [test]

    results = test_results.Results("c0ffee1234567890", "env", collected={test: ["test_a"]})
    monkeypatch.setattr(test_results, "list_test_files", list_test_files)
    monkeypatch.setattr(test_results, "recorded", lambda cache, commit, env_digest="": results)
    stage = _Delivery(tmp_path)
    # Not run at HEAD yet: nothing to say before delivery runs it.
    assert delivered_tests(stage, report) is None
    assert asked["paths"][1] == test
    jobs = []

    def record_at(commit, files, **job):
        jobs.append((commit, list(files)))
        results.tests[f"{test}::test_a"] = {"outcome": "passed"}
        return results, 1, 0, ""

    monkeypatch.setattr(regression, "job_runner", lambda workflow, name: (None, None, "python"))
    monkeypatch.setattr(regression, "record_at", record_at)
    check = delivered_tests(stage, report, run=True)
    assert check.passed and jobs == [("c0ffee1234567890", [test])]
    heads = iter(["c0ffee1234567890", "a-new-head"])
    stage._git = lambda *args: next(heads)
    changed = delivered_tests(stage, report, run=True)
    assert not changed.passed and "HEAD changed" in changed.detail
    stage._git = lambda *args: "c0ffee1234567890"
    # One that fails at HEAD fails the cheap read too.
    results.tests[f"{test}::test_a"] = {"outcome": "failed", "message": "wrong"}
    check = delivered_tests(stage, report)
    assert not check.passed and check.detail.startswith("FAILING")
