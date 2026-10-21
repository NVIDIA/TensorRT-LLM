# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks that measure one engine on one workload share a verdict."""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c

#: Three units; mtp and multimodal are gated on the same MTP options file and
#: anchor row, so their accuracy measures one engine. Core has its own.
_TASK = {
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


def _checks(task):
    return {check.key: check for check in c.compile_contract(task, world_size=4).checks}


def _result(name, member, passed=True):
    return SimpleNamespace(name=name, member=member, passed=passed, detail="d", tree_state="t")


def _with_accuracy(**overrides):
    return {**_TASK, "accuracy": {**_TASK["accuracy"], **overrides}}


def test_units_gated_on_one_engine_carry_one_accuracy_key():
    checks = _checks(_TASK)
    mtp, multimodal = checks["mtp:accuracy[gsm8k]@tp1"], checks["multimodal:accuracy[gsm8k]@tp1"]
    assert mtp.equivalence and mtp.equivalence == multimodal.equivalence
    # Another options file and anchor row, or another member: another engine.
    assert checks["core:accuracy[gsm8k]@tp1"].equivalence != mtp.equivalence
    assert checks["mtp:accuracy[gsm8k]@tep4"].equivalence != mtp.equivalence


def test_a_pass_on_this_target_holds_for_every_equivalent_check(tmp_path):
    contract = c.compile_contract(_TASK, world_size=4)
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


def test_nothing_is_shared_where_the_target_is_not_established(tmp_path):
    contract = c.compile_contract(_TASK, world_size=4)
    status = c.ContractStatus.load(tmp_path)
    status.record("mtp", _result("accuracy[gsm8k]", "tp1"), marker="")
    for marker in ("", None):
        evaluation = c.evaluate(contract, status, marker=marker)
        assert not evaluation.shared
        assert "multimodal:accuracy[gsm8k]@tp1" in {check.key for check in evaluation.missing}


def test_a_failure_is_not_shared_and_a_check_that_failed_here_stays_failed(tmp_path):
    contract = c.compile_contract(_TASK, world_size=4)
    status = c.ContractStatus.load(tmp_path)
    status.record("mtp", _result("accuracy[gsm8k]", "tp1", passed=False), marker="tree:1")
    evaluation = c.evaluate(contract, status, marker="tree:1")
    assert "multimodal:accuracy[gsm8k]@tp1" in {check.key for check in evaluation.missing}

    status.record("multimodal", _result("accuracy[gsm8k]", "tp1"), marker="tree:1")
    evaluation = c.evaluate(contract, status, marker="tree:1")
    assert "mtp:accuracy[gsm8k]@tp1" in {check.key for check, _ in evaluation.failed}
    assert not evaluation.shared


def test_a_difference_in_the_command_or_the_criterion_keeps_checks_apart():
    def same(task):
        checks = _checks(task)
        return (
            checks["mtp:accuracy[gsm8k]@tp1"].equivalence
            == checks["multimodal:accuracy[gsm8k]@tp1"].equivalence
        )

    assert same(_TASK)
    assert not same(_with_accuracy(benchmark_args={"multimodal": ["--num_samples=100"]}))
    assert not same(_with_accuracy(reference_yaml={"default": "x.yaml", "multimodal": "y.yaml"}))
    assert not same(_with_accuracy(model_key={"default": "a/b", "multimodal": "a/c"}))
    llm_api_options = {**_TASK["accuracy"]["llm_api_options"], "multimodal": "/opt/mm.yaml"}
    assert not same(_with_accuracy(llm_api_options=llm_api_options))
    # A per-unit value equal to the one the other unit falls back to is the same value.
    assert same(_with_accuracy(benchmark_args={"default": ["--x"], "multimodal": ["--x"]}))
    # The boot workload is the boot gate's alone.
    assert same(_with_accuracy(boot={"multimodal": {"prompts": ["Describe it."]}}))


def test_boot_shares_only_where_the_flags_and_the_workload_agree():
    checks = _checks({**_TASK, "features": ["mtp", "disagg", "multimodal"]})
    boot = {unit: checks[f"{unit}:boot@tp1"].equivalence for unit in _TASK["features"] + ["core"]}
    disagg = checks["disagg:boot@tp1"].equivalence
    # disagg adds no flag and no prompt to what mtp boots with.
    assert boot["mtp"] and boot["mtp"] == disagg
    # Core has no speculator; multimodal sends an image.
    assert boot["core"] != boot["mtp"] != boot["multimodal"]

    split = _checks({**_TASK, "runtime": {"default": {}, "multimodal": {"cuda_graph": True}}})
    assert split["multimodal:boot@tp1"].equivalence != split["mtp:boot@tp1"].equivalence


def test_only_boot_accuracy_and_acceptance_checks_of_units_carry_a_key(tmp_path):
    task = {
        **_TASK,
        "variants": {
            "kv_fp8": {"unit": "mtp", "runtime": {"extra_flags": ["--kv_cache_dtype=fp8"]}}
        },
    }
    contract = c.compile_contract(task, world_size=4)
    keyed = {check.key for check in contract.checks if check.equivalence}
    assert keyed
    assert all(":boot@" in key or ":accuracy[" in key or ":acceptance@" in key for key in keyed)
    assert not [key for key in keyed if key.startswith("kv_fp8:")]
    assert "mtp:acceptance@tp1" in keyed

    # The contract file names the key where there is one, and only there.
    data = yaml.safe_load(c.write_contract(tmp_path, contract).read_text(encoding="utf-8"))
    rows = {f"{r['unit']}:{r['gate']}@{r['member']}": r for r in data["checks"]}
    by_key = {check.key: check for check in contract.checks}
    assert rows["mtp:boot@tp1"]["equivalence"] == by_key["mtp:boot@tp1"].equivalence
    assert "equivalence" not in rows["kv_fp8:boot@tp1"]
    # Every check names what it reads from outside the checkout.
    assert all(r["inputs"] for r in data["checks"])
