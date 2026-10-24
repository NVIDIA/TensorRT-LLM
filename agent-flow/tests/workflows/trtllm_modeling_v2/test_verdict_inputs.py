# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A verdict is about what its check read: the checkout, and every input outside it."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates, testcache
from agent_flow.workflows.trtllm_modeling_v2.checks.gate_records import (
    KIND_STOCK_UNAVAILABLE,
    GateRecords,
)
from agent_flow.workflows.trtllm_modeling_v2.onboard.delivery import Check as DeliveryCheck
from agent_flow.workflows.trtllm_modeling_v2.onboard.delivery import DeliveryReport
from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run.stages.deliver import DeliverStage

from .test_acceptance_gate import _driver_output, _env, _legs, _task


def _spec(tmp_path: Path, core: Path, mtp: Path, multimodal: Path) -> dict:
    return {
        "checkpoint_path": "/ckpt/m",
        "parallel": "unified",
        "members": ["tp1"],
        "trtllm_repo_path": str(tmp_path / "repo"),
        "features": ["mtp", "multimodal"],
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "x.yaml",
            "llm_api_options": {
                "core": str(core),
                "mtp": str(mtp),
                "multimodal": str(multimodal),
            },
            "envelope": {"enabled": False},
        },
    }


def _options(path: Path, **content) -> Path:
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_options_of_one_content_at_two_paths_share_a_verdict(tmp_path):
    mtp = {"speculative_config": {"max_draft_len": 3}}
    spec = _spec(
        tmp_path,
        _options(tmp_path / "core.yaml"),
        _options(tmp_path / "mtp.yaml", **mtp),
        _options(tmp_path / "multimodal.yaml", **mtp),
    )
    checks = {check.key: check for check in c.compile_contract(spec, world_size=1).checks}
    for gate in ("accuracy[gsm8k]", "acceptance"):
        assert checks[f"mtp:{gate}@tp1"].equivalence == checks[f"multimodal:{gate}@tp1"].equivalence
    assert checks["core:accuracy[gsm8k]@tp1"].equivalence != (
        checks["mtp:accuracy[gsm8k]@tp1"].equivalence
    )
    _options(tmp_path / "multimodal.yaml", speculative_config={"max_draft_len": 2})
    checks = {check.key: check for check in c.compile_contract(spec, world_size=1).checks}
    assert (
        checks["mtp:acceptance@tp1"].equivalence != checks["multimodal:acceptance@tp1"].equivalence
    )


def test_an_input_outside_the_checkout_is_read_by_content(tmp_path):
    options = _options(tmp_path / "mtp.yaml", speculative_config={"max_draft_len": 3})
    spec = _spec(tmp_path, options, options, options)
    before = c.check_inputs(spec, "mtp", "boot", "tp1", ["mtp"])
    assert before == c.check_inputs(spec, "mtp", "boot", "tp1", ["mtp"])
    _options(options, speculative_config={"max_draft_len": 4})
    assert c.check_inputs(spec, "mtp", "boot", "tp1", ["mtp"]) != before
    # A floor of another unit is not this unit's input; its own is.
    floor = dict(spec, accuracy={**spec["accuracy"], "floor": {"core": {"k": 3}}})
    assert c.check_inputs(floor, "mtp", "boot", "tp1", ["mtp"]) == c.check_inputs(
        spec, "mtp", "boot", "tp1", ["mtp"]
    )
    floor["accuracy"]["floor"]["mtp"] = {"k": 3}
    assert c.check_inputs(floor, "mtp", "boot", "tp1", ["mtp"]) != c.check_inputs(
        spec, "mtp", "boot", "tp1", ["mtp"]
    )


def test_a_file_in_the_checkout_is_left_to_the_marker(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    inside = _options(repo / "core.yaml", max_num_tokens=8192)
    spec = _spec(tmp_path, inside, inside, inside)
    before = c.check_inputs(spec, "core", "boot", "tp1", [])
    _options(inside, max_num_tokens=4096)
    # The options' content is still an input; the path is not read a second time.
    assert c.check_inputs(spec, "core", "boot", "tp1", []) != before
    assert c._with_contents(str(inside), repo.resolve()) == str(inside)


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
    monkeypatch.setattr(
        gates, "inspect_tree", lambda task_data, allow_dirty=None: gates.TreeState("t")
    )
    task = _task(dspark={"min_absolute": 1.5, "floor_source": "a measured drafter"})
    rates = [2.6 + 0.5 * math.sin(i) for i in range(32)]
    records = GateRecords(tmp_path / "gate_results.jsonl")
    first = _legs(_driver_output(rates))
    ran = gates.run_acceptance_gate(
        _env(), task_data=task, runner=first, records=records, unit="dspark"
    )
    assert ran.passed, ran.detail
    assert any("=off" in cmd for cmd in first.seen)
    assert records.all(KIND_STOCK_UNAVAILABLE)
    again = _legs(_driver_output(rates))
    result = gates.run_acceptance_gate(
        _env(), task_data=task, runner=again, records=records, unit="dspark"
    )
    assert result.passed and not any("=off" in cmd for cmd in again.seen)
    assert "not run again" in result.detail


class _Delivery(DeliverStage):
    def __init__(self, tmp_path: Path, head: str = "c0ffee1234567890"):
        self.task_data = {"trtllm_repo_path": str(tmp_path)}
        self.workspace = tmp_path / "ws"
        self._preflight = SimpleNamespace(target_rel="pkg/models/fam/target")
        self._git = lambda repo, *args: head + "\n"


def test_delivery_holds_on_the_delivered_tests_passing_at_head(tmp_path, monkeypatch):
    report = DeliveryReport(
        checks=(DeliveryCheck("test:gemm/x", True, str(tmp_path / "tests/test_modeling_v2_x.py")),)
    )
    asked = {}

    def open_cache(cache_dir, repo, head):
        return SimpleNamespace(test_files=lambda head, paths: asked.setdefault("paths", paths))

    monkeypatch.setattr(testcache, "open_cache", open_cache)
    outcome = {"r": testcache.HeadResults("c0ffee12", [], [], [], ["t::a"])}
    monkeypatch.setattr(testcache, "head_results", lambda cache, head, paths: outcome["r"])
    stage = _Delivery(tmp_path)
    check = stage._tests_check(report)
    assert not check.passed and "INCOMPLETE" in check.detail and "tests affected" in check.detail
    assert asked["paths"][1] == "tests/test_modeling_v2_x.py"
    outcome["r"] = testcache.HeadResults("c0ffee12", ["t::a"], [], [], [])
    assert stage._tests_check(report).passed
