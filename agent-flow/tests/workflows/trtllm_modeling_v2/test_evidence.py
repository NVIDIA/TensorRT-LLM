# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The orchestrator's side of the ledger: which verdicts it records, and under what key.

Besides the engine gates, a run produces three more kinds of verdict: the
envelope probes (one boot per declared envelope point), the early delivery
checks (warnings after the plan and after every unit), and the delivery
checks, which include the capacity audit. All of them are recorded, and none
may hide another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent_flow.workflows.trtllm_modeling_v2 import envelope as envelope_mod
from agent_flow.workflows.trtllm_modeling_v2.delivery import Check, DeliveryReport
from agent_flow.workflows.trtllm_modeling_v2.early_checks import EarlyFinding, EarlyReport
from agent_flow.workflows.trtllm_modeling_v2.evidence import (
    DELIVERY_CHECK_PREFIX,
    EARLY_CHECK_PREFIX,
    RunEvidence,
    gate_key,
    role_agents,
)
from agent_flow.workflows.trtllm_modeling_v2.freshness import RerunEntry
from agent_flow.workflows.trtllm_modeling_v2.gates import GateLog, GateResult
from agent_flow.workflows.trtllm_modeling_v2.journal import Journal
from agent_flow.workflows.trtllm_modeling_v2.ledger import OUTCOME_FINDING, OUTCOME_PASS
from agent_flow.workflows.trtllm_modeling_v2.progress import ProgressContext

from .test_delivery import TARGET_REL, _build_repo


@dataclass
class _Probe:
    unit: str
    member: str
    label: str

    def point(self) -> str:
        return self.label


@dataclass
class _Plan:
    probes: list[_Probe] = field(default_factory=list)

    def probes_for(self, unit: str) -> list[_Probe]:
        return [p for p in self.probes if p.unit == unit]


@dataclass
class _Preflight:
    target_rel: str = TARGET_REL
    certification: Any = None


class _Workflow:
    """Just the attributes RunEvidence reads."""

    def __init__(self, tmp_path: Path, *, plan: _Plan | None = None, agents: Any = None) -> None:
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir(parents=True, exist_ok=True)
        repo = _build_repo(tmp_path)
        self.task_data: dict[str, Any] = {
            "trtllm_repo_path": str(repo),
            "features": ["mtp"],
            "environment": {"kind": "local", "python": "/py", "env": {"TRTLLM_X": "1"}},
        }
        if agents is not None:
            self.task_data["agents"] = agents
        self.environment = None
        self._preflight = _Preflight(certification=plan)
        self._progress_ctx = ProgressContext(path=self.workspace / "progress.yaml")
        self.journal = Journal(path=self.workspace / "journal.md")
        self.boot_spec_path = self.workspace / "boot.yaml"
        self._shell_runner = None
        self.evidence = RunEvidence(self)
        self.gate_log = GateLog(sink=self.evidence.record)

    def _features_through(self, unit: str) -> list[str]:
        return [] if unit == "core" else ["mtp"]


def _probe_result(point: str, member: str = "tep4", passed: bool = True) -> GateResult:
    return GateResult(
        name=envelope_mod.PROBE_GATE, passed=passed, member=member, detail=f"at {point}: booted"
    )


# --------------------------------------------------------------- envelope


def test_each_envelope_point_is_its_own_ledger_key():
    """Keyed by name alone, a failing point would hide behind the next passing one."""
    assert gate_key(_probe_result("B=8 T=2048")) == "envelope[B=8 T=2048]"
    assert gate_key(GateResult(name="envelope", passed=True, detail="no point")) == "envelope"
    assert gate_key(GateResult(name="boot", passed=True, detail="at x: y")) == "boot"


def test_two_probes_at_one_member_are_both_latest(tmp_path):
    wf = _Workflow(tmp_path)
    wf._progress_ctx.current_unit = "core"
    wf.gate_log.record(_probe_result("B=8 T=2048", passed=False))
    wf.gate_log.record(_probe_result("B=256 T=16384"))
    latest = wf.evidence.ledger.latest()
    assert latest[("envelope[B=8 T=2048]", "core", "tep4")].outcome == "FAIL"
    assert latest[("envelope[B=256 T=16384]", "core", "tep4")].outcome == OUTCOME_PASS


def test_a_stale_probe_is_re_run_at_its_own_point(tmp_path, monkeypatch):
    probe = _Probe("core", "tep4", "B=8 T=2048")
    wf = _Workflow(tmp_path, plan=_Plan([probe]))
    calls: list[Any] = []

    def run_probe(environment, *, task_data, probe, spec, runner, features):
        calls.append((probe, features))
        return _probe_result(probe.point())

    monkeypatch.setattr(envelope_mod, "run_envelope_probe", run_probe)
    monkeypatch.setattr(type(wf), "_gate_env", lambda self: None, raising=False)
    entry = RerunEntry("envelope[B=8 T=2048]", "core", "tep4", ("code changed: x",))
    assert wf.evidence.rerun(entry) is True
    assert calls == [(probe, [])]
    assert ("envelope[B=8 T=2048]", "core", "tep4") in wf.evidence.ledger.latest()


def test_a_probe_the_plan_no_longer_has_is_neither_required_nor_re_run(tmp_path):
    wf = _Workflow(tmp_path, plan=_Plan([_Probe("core", "tep4", "B=256 T=16384")]))
    wf._progress_ctx.current_unit = "core"
    wf.gate_log.record(_probe_result("B=8 T=2048"))
    (verdict,) = wf.evidence.ledger.verdicts()
    assert not wf.evidence.in_scope()(verdict)
    entry = RerunEntry(verdict.gate, "core", "tep4", ())
    assert wf.evidence.rerunner(entry) is None


# ------------------------------------------------------------------ checks


def test_early_check_warnings_are_findings_and_carry_no_unit(tmp_path):
    wf = _Workflow(tmp_path)
    report = EarlyReport(
        stage="after the core unit",
        findings=(
            EarlyFinding("tests:accuracy-file", "warn", "no accuracy test yet"),
            EarlyFinding("capacity:max_num_tokens", "ok", "bound to the engine argument"),
        ),
    )
    assert wf.evidence.record_early_checks(report) is report
    latest = wf.evidence.ledger.latest()
    warn = latest[(f"{EARLY_CHECK_PREFIX}tests:accuracy-file", "", "")]
    ok = latest[(f"{EARLY_CHECK_PREFIX}capacity:max_num_tokens", "", "")]
    assert warn.outcome == OUTCOME_FINDING and ok.outcome == OUTCOME_PASS
    assert warn.dependencies == {}  # about the repository, not about the target's code
    assert not wf.evidence.in_scope()(warn)  # never re-run, never required by the regression
    assert wf.evidence.reportable()(warn)


def test_delivery_checks_including_the_capacity_audit_are_recorded(tmp_path):
    wf = _Workflow(tmp_path)
    report = DeliveryReport(
        checks=(
            Check("capacity:KV_POOL_ROWS", False, "constant 4096 < envelope 8192"),
            Check("matrix:dspark-x-disagg", True, "pass on abc1234"),
        )
    )
    assert wf.evidence.record_delivery_checks(report) is report
    latest = wf.evidence.ledger.latest()
    assert latest[(f"{DELIVERY_CHECK_PREFIX}capacity:KV_POOL_ROWS", "", "")].outcome == "FAIL"
    assert latest[(f"{DELIVERY_CHECK_PREFIX}matrix:dspark-x-disagg", "", "")].outcome == "PASS"


# ------------------------------------------------------------------ agents


def test_each_roles_backend_and_model_are_in_the_config_fingerprint(tmp_path):
    default = _Workflow(tmp_path / "a")
    default._progress_ctx.current_unit = "core"
    first = default.evidence.record(GateResult(name="boot", passed=True, detail="ok"))
    assert first.config["agents"]["plan_reviewer"]["backend"] == "codex"
    assert first.config["agents"]["assembler"]["backend"] == "claude-code"

    routed = _Workflow(tmp_path / "b", agents={"roles": {"assembler": {"model": "other-model"}}})
    routed._progress_ctx.current_unit = "core"
    second = routed.evidence.record(GateResult(name="boot", passed=True, detail="ok"))
    assert second.config["agents"]["assembler"]["model"] == "other-model"
    assert second.config_fingerprint != first.config_fingerprint
    # Which model writes the next change does not invalidate what was measured.
    assert second.spec_fingerprint == first.spec_fingerprint


@pytest.mark.parametrize("role", ["plan_drafter", "smith", "assembler", "reviewer", "auditor"])
def test_role_agents_covers_every_role(role):
    assert role in role_agents({})
