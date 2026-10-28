# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contract: the checks a run owes, and their evaluation."""

from __future__ import annotations

from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c

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


def test_reference_table_at_checkout_root_is_a_whole_file_gate_input(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import gate_inputs

    (tmp_path / "x.yaml").write_text("M: {gsm8k: 0.8}\n")
    assert "x.yaml" in gate_inputs(_TASK, tmp_path)


def test_every_unit_owes_boot_and_accuracy_at_every_member():
    contract = c.compile_contract(_TASK, world_size=4)
    inner = _keys(contract, c.TIER_INNER)
    for unit in ("core", "mtp", "host_offload"):
        for member in ("tp4", "dep4"):
            assert f"{unit}:boot@{member}" in inner
            assert f"{unit}:accuracy[gsm8k]@{member}" in inner


def _with_accuracy(**blocks):
    return {**_TASK, "accuracy": {**_TASK["accuracy"], **blocks}}


def test_feature_proofs_follow_the_units_that_need_them():
    proof = _keys(c.compile_contract(_TASK, world_size=4), c.TIER_PROOF)
    # The speculator's acceptance gate runs from its own unit on, never in core.
    assert "mtp:acceptance@tp4" in proof
    assert "host_offload:acceptance@tp4" in proof
    assert not [k for k in proof if k.startswith("core:acceptance")]
    # Witnesses are off by default; on, the unit that needs one owes it.
    assert not [k for k in proof if "residency" in k]
    on = _with_accuracy(witness={"enabled": True})
    assert "host_offload:residency@dep4" in _keys(
        c.compile_contract(on, world_size=4), c.TIER_PROOF
    )


def test_with_the_logit_gate_a_units_benchmark_runs_at_one_member():
    task = {
        **_with_accuracy(logits={"reference": "/refs/logits.reference.json.gz"}),
        "members": ["tp4", "tep4", "dep4"],
        "features": [],
    }
    keys = _keys(c.compile_contract(task, world_size=4))
    # Attention DP is what the logit gate sees least; it proves the other members.
    assert [k for k in keys if "accuracy" in k] == ["core:accuracy[gsm8k]@dep4"]
    assert {"core:logits@tp4", "core:logits@tep4", "core:boot@tp4", "core:boot@tep4"} <= set(keys)


def test_the_profile_and_the_hop_decide_what_a_unit_owes():
    task = {
        **_TASK,
        "features": ["multimodal", "disagg"],
        "profile": {"default": "standard", "multimodal": "smoke"},
        "variants": {"disagg_kv_fp8": {"unit": "disagg"}},
    }
    keys = _keys(c.compile_contract(task, world_size=4))
    assert sorted(k for k in keys if k.startswith("multimodal:")) == [
        "multimodal:boot@dep4",
        "multimodal:boot@tp4",
    ]
    # Disaggregation's boot and accuracy would run an aggregated engine: the witness proves it.
    assert sorted(k for k in keys if k.startswith("disagg")) == [
        "disagg:equivalence@dep4",
        "disagg:equivalence@tp4",
        "disagg_kv_fp8:equivalence@dep4",
        "disagg_kv_fp8:equivalence@tp4",
    ]


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


def test_a_verdict_on_an_earlier_target_is_stale(tmp_path):
    contract = c.Contract(checks=(c.Check("core", "boot", "tp1", c.TIER_INNER),))
    status = c.ContractStatus.load(tmp_path)
    status.record("core", _result("boot", "tp1"), marker="v1")
    assert c.evaluate(contract, status, marker="v1").holds
    assert "MISSING core:boot@tp1" in c.evaluate(contract, status, marker="v2").render()


#: Three units; mtp and multimodal are gated on the same MTP options file and
#: anchor row, so their accuracy measures one engine. Core has its own.
_SHARED = {
    "checkpoint_path": "/ckpt/m",
    "parallel": "unified",
    "members": ["tp1", "tep4"],
    "trtllm_repo_path": "/repo",
    "features": ["mtp", "multimodal"],
    "accuracy": {
        "benchmark": "gsm8k",
        "reference_yaml": "x.yaml",
        "llm_api_options": {"core": "/opt/core.yaml", "default": "/opt/mtp3.yaml"},
        "discriminators": {
            "core": {"quant_algo": "FP8"},
            "default": {"quant_algo": "FP8", "spec_dec_algo": "MTP"},
        },
        "envelope": {"enabled": False},
    },
}


def test_a_pass_on_this_target_holds_for_every_equivalent_check(tmp_path):
    contract = c.compile_contract(_SHARED, world_size=4)
    status = c.ContractStatus.load(tmp_path)
    status.record("mtp", _result("accuracy[gsm8k]", "tp1"), marker="tree:1")

    evaluation = c.evaluate(contract, status, marker="tree:1")
    passed = {check.key for check in evaluation.passed}
    missing = {check.key for check in evaluation.missing}
    assert {"mtp:accuracy[gsm8k]@tp1", "multimodal:accuracy[gsm8k]@tp1"} <= passed
    assert [(a.key, b.key) for a, b in evaluation.shared] == [
        ("multimodal:accuracy[gsm8k]@tp1", "mtp:accuracy[gsm8k]@tp1")
    ]
    assert "SHARED multimodal:accuracy[gsm8k]@tp1: holds by the pass of mtp" in evaluation.render()
    # Not across units with another engine, nor across members.
    assert {"core:accuracy[gsm8k]@tp1", "multimodal:accuracy[gsm8k]@tep4"} <= missing

    # On another target the pass is stale, for both.
    later = c.evaluate(contract, status, marker="tree:2")
    assert "multimodal:accuracy[gsm8k]@tp1" in {check.key for check in later.missing}
    assert not later.shared


def test_a_failure_is_not_shared_and_a_check_that_failed_here_stays_failed(tmp_path):
    contract = c.compile_contract(_SHARED, world_size=4)
    status = c.ContractStatus.load(tmp_path)
    status.record("mtp", _result("accuracy[gsm8k]", "tp1", passed=False), marker="tree:1")
    evaluation = c.evaluate(contract, status, marker="tree:1")
    assert "multimodal:accuracy[gsm8k]@tp1" in {check.key for check in evaluation.missing}

    status.record("multimodal", _result("accuracy[gsm8k]", "tp1"), marker="tree:1")
    evaluation = c.evaluate(contract, status, marker="tree:1")
    assert "mtp:accuracy[gsm8k]@tp1" in {check.key for check, _ in evaluation.failed}
    assert not evaluation.shared
