# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a benchmark is held to: resolved in order, frozen once, changed only on request."""

from __future__ import annotations

import json

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy, accuracy_policy, anchors
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments

ROWS = {
    "org/toy": [
        {"accuracy": 92.0},
        {"accuracy": 90.0, "kv_cache_quant_algo": "FP8"},
        {"accuracy": 87.0, "quant_algo": "NVFP4", "spec_dec_algo": "MTP", "extra_acc_spec": "mtp3"},
        {"accuracy": 88.0, "quant_algo": "NVFP4", "spec_dec_algo": "MTP", "threshold": 86.0},
        {"accuracy": 89.0, "quant_algo": "NVFP4", "spec_dec_algo": "MTP", "mtp_enabled": True},
    ]
}
POLICY = accuracy_policy.FloorPolicy()


@pytest.fixture
def spec(tmp_path):
    tree = tmp_path / "tree"
    references = tree / anchors.REFERENCES_DIR
    references.mkdir(parents=True)
    (references / "gsm8k.yaml").write_text(yaml.safe_dump(ROWS))
    return {
        "checkpoint_path": str(tmp_path / "org" / "toy"),
        "trtllm_repo_path": str(tree),
        "accuracy": {"model_key": "org/toy"},
        "deployments": [
            {"name": "tp1", "options": {}},
            {"name": "kv_fp8", "options": {"kv_cache_config": {"dtype": "fp8"}}},
            {
                "name": "mtp",
                "options": {"speculative_config": {"decoding_type": "MTP"}},
                "benchmarks": [{"name": "gsm8k", "discriminators": {"quant_algo": "NVFP4"}}],
            },
            {
                "name": "declared",
                "options": {},
                "benchmarks": [
                    {"name": "aa_lcr", "anchor": 0.62, "source": "measured", "unit": "fraction"}
                ],
            },
            {
                "name": "relative",
                "options": {"max_batch_size": 1},
                "relative_to": "tp1",
                "benchmarks": [{"name": "mmlu"}],
            },
            {"name": "nothing", "options": {}, "benchmarks": [{"name": "mmlu"}]},
        ],
    }


def anchor(spec, name):
    deployment = deployments.named(spec, name)
    return anchors.resolve(spec, deployment, deployment.benchmarks[0])


def test_an_anchor_is_declared_then_the_exact_reference_row_then_relative_then_none(spec):
    assert (anchor(spec, "tp1").kind, anchor(spec, "tp1").value) == (anchors.REFERENCE, 92.0)
    # A row matches its tags: what the options imply, and what the benchmark states.
    assert anchor(spec, "kv_fp8").value == 90.0
    # The row the in-tree accuracy test takes: the first that agrees on that test's five
    # selectors. A row's `threshold` does not select; its `extra_acc_spec` does.
    assert anchor(spec, "mtp").value == 88.0
    path = anchors.reference_path(spec, deployments.named(spec, "mtp").benchmarks[0])
    asked = {"quant_algo": "NVFP4", "spec_dec_algo": "MTP"}

    def row(**more):
        return accuracy.load_reference_score(path, "org/toy", discriminators={**asked, **more})

    assert (row(extra_acc_spec="mtp3"), row(mtp_enabled=True)) == (87.0, 89.0)
    with pytest.raises(KeyError, match="no entry matching"):
        row(dtype="float16")
    declared = anchor(spec, "declared")
    assert (declared.kind, declared.value, declared.unit) == (anchors.DECLARED, 0.62, "fraction")
    relative = anchor(spec, "relative")
    assert (relative.kind, relative.relative_to) == (anchors.RELATIVE, "tp1")
    none = anchor(spec, "nothing")
    assert none.kind == anchors.NONE and "no reference row" in none.source
    assert anchor(spec, "tp1").samples == 1319


def test_frozen_anchors_do_not_move_with_the_tree_and_new_ones_are_added(spec, tmp_path):
    workspace = tmp_path / "workspace"
    held = anchors.freeze(workspace, spec)
    assert held["tp1:gsm8k"].value == 92.0 and len(held) == 6
    rows = {**ROWS, "org/toy": [{"accuracy": 95.0}, *ROWS["org/toy"][1:]]}
    reference = tmp_path / "tree" / anchors.REFERENCES_DIR / "gsm8k.yaml"
    reference.write_text(yaml.safe_dump(rows))
    again = anchors.freeze(workspace, spec)
    assert again["tp1:gsm8k"].value == 92.0
    assert anchors.frozen(workspace, "tp1", "gsm8k").value == 92.0
    spec["deployments"].append({"name": "later", "options": {}})
    assert anchors.freeze(workspace, spec)["later:gsm8k"].value == 95.0
    (workspace / anchors.ANCHORS_FILENAME).write_text("- not: [an anchor")
    with pytest.raises(anchors.AnchorError, match="delete it to freeze them again"):
        anchors.load(workspace)


def test_an_anchor_is_changed_by_an_explicit_command_and_the_change_is_recorded(spec, tmp_path):
    workspace = tmp_path / "workspace"
    anchors.freeze(workspace, spec)
    before, after = anchors.change(
        workspace, "nothing", "mmlu", value=71.5, source="stock path, one node", reason="measured"
    )
    assert before.kind == anchors.NONE
    assert (after.kind, after.value, after.source) == (
        anchors.DECLARED,
        71.5,
        "stock path, one node",
    )
    # No requirement is beyond change: a frozen reference row may be corrected too.
    anchors.change(workspace, "tp1", "gsm8k", value=40.0, reason="the row was another protocol's")
    assert anchors.frozen(workspace, "tp1", "gsm8k").value == 40.0
    anchors.change(workspace, "kv_fp8", "gsm8k", relative_to="tp1", reason="no row for this cache")
    assert anchors.frozen(workspace, "kv_fp8", "gsm8k").kind == anchors.RELATIVE
    said = anchors.changes(workspace)
    assert [(c["anchor"], c["reason"]) for c in said] == [
        ("nothing:mmlu", "measured"),
        ("tp1:gsm8k", "the row was another protocol's"),
        ("kv_fp8:gsm8k", "no row for this cache"),
    ]
    assert said[1]["from"]["value"] == 92.0 and said[1]["to"]["value"] == 40.0
    lines = (workspace / anchors.CHANGES_FILENAME).read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["at"]
    with pytest.raises(ValueError, match="needs its reason"):
        anchors.change(workspace, "tp1", "gsm8k", value=41.0, reason=" ")
    # A later start keeps what was changed.
    assert anchors.freeze(workspace, spec)["tp1:gsm8k"].value == 40.0


def test_the_floor_is_the_specs_rule_on_the_benchmarks_own_scale(spec):
    tp1 = deployments.named(spec, "tp1")
    floor, source = anchors.floor(anchor(spec, "tp1"), tp1.benchmarks[0], POLICY)
    assert floor.value == pytest.approx(92.0 - 2 * accuracy_policy.binomial_se(92.0, 1319))
    assert source == "the count fixed with the anchor"
    declared = deployments.named(spec, "declared")
    benchmark = declared.benchmarks[0]
    with_count = anchors.Anchor(
        "declared", "aa_lcr", anchors.DECLARED, 0.62, unit="fraction", samples=400
    )
    fraction, _ = anchors.floor(with_count, benchmark, POLICY)
    assert 0.5 < fraction.value < 0.62 and fraction.reference == 0.62
    # With no count the statistical rule has no floor, which is not a pass.
    none, why = anchors.floor(anchor(spec, "declared"), benchmark, POLICY)
    assert none is None and "state its `samples`" in why
    relative = deployments.named(spec, "relative")
    held, _ = anchors.floor(
        anchor(spec, "relative"), relative.benchmarks[0], POLICY, relative_score=70.0
    )
    assert held.relative and held.value < 70.0
    missing, why = anchors.floor(anchor(spec, "relative"), relative.benchmarks[0], POLICY)
    assert missing is None and "`tp1` has no score" in why


def test_a_score_is_judged_one_sided_against_its_floor(spec):
    tp1 = deployments.named(spec, "tp1")
    benchmark, held = tp1.benchmarks[0], anchor(spec, "tp1")
    passed = accuracy.judged(91.5, held, benchmark, POLICY, how=" through the endpoint")
    assert passed["passed"] and passed["reference"] == 92.0 and passed["samples"] == 1319
    assert "through the endpoint" in passed["detail"] and "anchor: reference" in passed["detail"]
    assert not accuracy.judged(80.0, held, benchmark, POLICY)["passed"]
    fraction = anchors.Anchor("d", "aa_lcr", anchors.DECLARED, 0.62, unit="fraction", samples=400)
    off_scale = accuracy.judged(62.0, fraction, benchmark, POLICY)
    assert not off_scale["passed"] and "is no fraction" in off_scale["detail"]
