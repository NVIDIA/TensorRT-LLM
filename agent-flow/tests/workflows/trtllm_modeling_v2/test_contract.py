# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contract: the checks a run owes, and their evaluation."""

from __future__ import annotations

from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c

_TASK = {
    "checkpoint_path": "/ckpt/m",
    "parallel": "unified",
    "members": ["tp4", "dep4"],
    "trtllm_repo_path": "/repo",
    "features": ["mtp", "host_offload"],
    "accuracy": {"benchmark": "gsm8k", "reference_yaml": "x.yaml"},
}


def _keys(contract, tier=None):
    checks = contract.by_tier(tier) if tier else contract.checks
    return [check.key for check in checks]


def test_every_unit_owes_boot_and_accuracy_at_every_member():
    contract = c.compile_contract(_TASK, world_size=4)
    inner = _keys(contract, c.TIER_INNER)
    for unit in ("core", "mtp", "host_offload"):
        for member in ("tp4", "dep4"):
            assert f"{unit}:boot@{member}" in inner
            assert f"{unit}:accuracy[gsm8k]@{member}" in inner


def test_feature_proofs_follow_the_units_that_need_them():
    proof = _keys(c.compile_contract(_TASK, world_size=4), c.TIER_PROOF)
    # The speculator's acceptance gate runs from its own unit on, never in core.
    assert "mtp:acceptance@tp4" in proof
    assert "host_offload:acceptance@tp4" in proof
    assert not [k for k in proof if k.startswith("core:acceptance")]
    assert "host_offload:residency@dep4" in proof


def test_delivery_owes_the_envelope_on_the_last_unit_unless_disabled():
    delivery = _keys(c.compile_contract(_TASK, world_size=4), c.TIER_DELIVERY)
    assert sorted(delivery) == ["host_offload:envelope@dep4", "host_offload:envelope@tp4"]
    off = {**_TASK, "accuracy": {**_TASK["accuracy"], "envelope": {"enabled": False}}}
    assert _keys(c.compile_contract(off, world_size=4), c.TIER_DELIVERY) == []


def _result(name, member, passed=True):
    return SimpleNamespace(name=name, member=member, passed=passed, detail="d", tree_state="t")


def test_status_survives_a_new_process_and_the_latest_verdict_counts(tmp_path):
    status = c.ContractStatus.load(tmp_path)
    status.record("core", _result("boot", "tp1", passed=False))
    status.record("core", _result("boot", "tp1", passed=True))
    reloaded = c.ContractStatus.load(tmp_path)
    assert reloaded.latest()["core:boot@tp1"].passed


def test_evaluation_names_what_failed_and_what_never_ran(tmp_path):
    task = {
        "checkpoint_path": "/ckpt/m",
        "parallel": "tp1",
        "trtllm_repo_path": "/repo",
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "x.yaml",
            "envelope": {"enabled": False},
        },
    }
    contract = c.compile_contract(task, world_size=1)
    assert _keys(contract) == ["core:boot@tp1", "core:accuracy[gsm8k]@tp1"]
    status = c.ContractStatus.load(tmp_path)
    status.record("core", _result("boot", "tp1"))
    evaluation = c.evaluate(contract, status)
    assert not evaluation.holds
    assert "MISSING core:accuracy[gsm8k]@tp1" in evaluation.render()
    status.record("core", _result("accuracy[gsm8k]", "tp1", passed=False))
    assert "FAILED core:accuracy[gsm8k]@tp1" in c.evaluate(contract, status).render()
    status.record("core", _result("accuracy[gsm8k]", "tp1"))
    assert c.evaluate(contract, status).holds


def test_the_contract_is_written_where_operators_look(tmp_path):
    import yaml

    path = c.write_contract(tmp_path, c.compile_contract(_TASK, world_size=4))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert {"unit", "gate", "member", "tier", "required"} <= set(data["checks"][0])


def test_a_verdict_on_an_earlier_target_is_stale(tmp_path):
    contract = c.Contract(checks=(c.Check("core", "boot", "tp1", c.TIER_INNER),))
    status = c.ContractStatus.load(tmp_path)
    status.record("core", _result("boot", "tp1"), marker="v1")
    assert c.evaluate(contract, status, marker="v1").holds
    assert "MISSING core:boot@tp1" in c.evaluate(contract, status, marker="v2").render()


def test_a_unit_whose_anchor_is_pending_owes_no_accuracy_check():
    """A score compared with nothing proves nothing; the logit gate is the oracle instead."""
    task = {
        **_TASK,
        "accuracy": {
            **_TASK["accuracy"],
            "anchor": {"mtp": "pending"},
            "logits": {"reference": "logits.json.gz"},
        },
    }
    contract = c.compile_contract(task, world_size=4)
    keys = _keys(contract)
    assert not [k for k in keys if k.startswith("mtp:accuracy")]
    assert [k for k in keys if k.startswith("core:accuracy")]
    assert "mtp:logits@tp4" in keys
    assert any("mtp: accuracy anchor pending" in note for note in contract.notes)
