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


def test_every_unit_owes_boot_and_accuracy_at_every_member_and_features_their_proof():
    contract = c.compile_contract(_TASK, world_size=4)
    inner = _keys(contract, c.TIER_INNER)
    for unit in ("core", "mtp", "host_offload"):
        for member in ("tp4", "dep4"):
            assert f"{unit}:boot@{member}" in inner
            assert f"{unit}:accuracy[gsm8k]@{member}" in inner
    proof = _keys(contract, c.TIER_PROOF)
    # The speculator's acceptance gate runs from its own unit on, never in core.
    assert "mtp:acceptance@tp4" in proof
    assert "host_offload:acceptance@tp4" in proof
    assert not [k for k in proof if k.startswith("core:acceptance")]


def test_a_benchmark_runs_where_the_spec_names_and_a_variants_at_one_member():
    task = {
        **_TASK,
        "accuracy": {**_TASK["accuracy"], "benchmark_members": "tep4"},
        "members": ["tp4", "tep4", "dep4"],
        "features": [],
    }
    keys = _keys(c.compile_contract(task, world_size=4))
    assert [k for k in keys if "accuracy" in k] == ["core:accuracy[gsm8k]@tep4"]
    assert {"core:boot@tp4", "core:boot@tep4", "core:boot@dep4"} <= set(keys)
    # A variant boots everywhere; its benchmark prefers attention DP, the most paths.
    task = {**_TASK, "features": [], "variants": {"kv_fp8": {"unit": "core"}}}
    keys = _keys(c.compile_contract(task, world_size=4))
    assert [k for k in keys if k.startswith("kv_fp8:")] == [
        "kv_fp8:boot@dep4",
        "kv_fp8:boot@tp4",
        "kv_fp8:accuracy[gsm8k]@dep4",
    ]


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
    # Disaggregation's boot and accuracy would run an aggregated engine: its own
    # test proves it, once; a variant of it owes nothing, its axis proven on core.
    assert [k for k in keys if k.startswith("disagg")] == ["disagg:disagg@dep4"]


def _result(name, member, passed=True):
    return SimpleNamespace(name=name, member=member, passed=passed, detail="d", tree_state="t")


def test_evaluation_names_what_failed_never_ran_or_ran_on_an_earlier_target(tmp_path):
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
    status.record("core", _result("boot", "tp1"), marker="v1")
    evaluation = c.evaluate(contract, status, marker="v1")
    assert not evaluation.holds
    assert "MISSING core:accuracy[gsm8k]@tp1" in evaluation.render()
    status.record("core", _result("accuracy[gsm8k]", "tp1", passed=False), marker="v1")
    assert "FAILED core:accuracy[gsm8k]@tp1" in c.evaluate(contract, status, marker="v1").render()
    status.record("core", _result("accuracy[gsm8k]", "tp1"), marker="v1")
    # The latest verdict counts, and survives a new process.
    reloaded = c.ContractStatus.load(tmp_path)
    assert c.evaluate(contract, reloaded, marker="v1").holds
    assert "MISSING core:boot@tp1" in c.evaluate(contract, reloaded, marker="v2").render()


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


def test_a_pass_on_this_target_holds_for_every_equivalent_check_and_a_failure_for_none(tmp_path):
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

    # A failure is not shared, and a check that failed here stays failed.
    status.record("mtp", _result("accuracy[gsm8k]", "tp1", passed=False), marker="tree:3")
    evaluation = c.evaluate(contract, status, marker="tree:3")
    assert "multimodal:accuracy[gsm8k]@tp1" in {check.key for check in evaluation.missing}
    status.record("multimodal", _result("accuracy[gsm8k]", "tp1"), marker="tree:3")
    evaluation = c.evaluate(contract, status, marker="tree:3")
    assert "mtp:accuracy[gsm8k]@tp1" in {check.key for check, _ in evaluation.failed}
    assert not evaluation.shared


def test_a_relative_variants_benchmark_comes_after_its_units_whose_score_is_kept(tmp_path):
    fp8 = {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}, "gates": ["accuracy"]}
    task = {
        **_TASK,
        "features": [],
        "variants": {"kv_fp8": fp8, "kv_row": {**fp8, "accuracy": {"discriminators": {"x": 1}}}},
    }
    checks = {k.key: k for k in c.compile_contract(task, world_size=4).checks}
    assert checks["kv_fp8:accuracy[gsm8k]@dep4"].after == "core:accuracy[gsm8k]@dep4"
    assert checks["kv_row:accuracy[gsm8k]@dep4"].after == ""
    status = c.ContractStatus.load(tmp_path)
    result = SimpleNamespace(name="accuracy[gsm8k]", member="dep4", passed=False, detail="d")
    status.record("core", SimpleNamespace(**vars(result), measured=88.5), marker="m1")
    again = c.ContractStatus.load(tmp_path)
    assert c.verdict_on(again, "core:accuracy[gsm8k]@dep4", "m1").measured == 88.5
    assert c.verdict_on(again, "core:accuracy[gsm8k]@dep4", "m2") is None
