# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for DSpark workload-level confidence admission."""

import hashlib
import json

import pytest

from tensorrt_llm._torch.speculative.dspark_planner import (
    EXACT_SPS_SELECTOR_IDENTITY_SHA256,
    evaluate_confidence_workload_admission,
    load_confidence_workload_admission,
)


def _multi_g_sps_payload():
    fingerprint = {
        "gpu": "B300",
        "gpu_count": 8,
        "gpu_snapshot_sha256": "a" * 64,
        "global_graph_batch_sizes": [512, 1024],
        "max_draft_len": 5,
        "rank_local_graph_batch_sizes": [64, 128],
        "runtime_snapshot": "runtime-v2",
        "source_diff_sha256": "b" * 64,
        "source_head": "23b73d8",
        "topology": "DEP8",
    }
    cells = {
        64: ((0, 352), (6.0, 5.2)),
        128: ((0, 704, 736), (8.0, 7.0, 7.2)),
    }
    payload = {
        "schema_version": 2,
        "minimum_predicted_gain": 0.02,
        "cost_tables": {
            str(graph_batch_size): {
                "token_counts": list(verifier_budgets),
                "step_time_ms": list(step_times),
            }
            for graph_batch_size, (verifier_budgets, step_times) in cells.items()
        },
        "engine_fingerprint": fingerprint,
        "measurements": [
            {
                "rank_local_graph_batch_size": graph_batch_size,
                "rank_local_verifier_budget": verifier_budget,
                "step_time_ms": step_time,
                "source_result_sha256": "c" * 64,
            }
            for graph_batch_size, (verifier_budgets, step_times) in cells.items()
            for verifier_budget, step_time in zip(verifier_budgets, step_times)
        ],
    }
    payload["engine_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _write_workload_admission(
    tmp_path,
    *,
    admitted=True,
    policy_steps=100,
    compact_choices=50,
    gross_compact_value_ms_lower_bound=160.0,
    fixed_confidence_path_overhead_ms_upper_bound=1.0,
    safety_margin_ms=10.0,
):
    sps_payload = _multi_g_sps_payload()
    sps_path = tmp_path / "sps.json"
    sps_path.write_text(json.dumps(sps_payload))
    fingerprint_path = tmp_path / "live-fingerprint.json"
    fingerprint_path.write_text(json.dumps(sps_payload["engine_fingerprint"]))
    admission = {
        "admitted": admitted,
        "calibration_result_sha256": "d" * 64,
        "compact_choices": compact_choices,
        "fixed_confidence_path_overhead_ms_upper_bound": (
            fixed_confidence_path_overhead_ms_upper_bound
        ),
        "gross_compact_value_ms_lower_bound": gross_compact_value_ms_lower_bound,
        "live_engine_fingerprint_sha256": hashlib.sha256(fingerprint_path.read_bytes()).hexdigest(),
        "physical_k": 5,
        "policy_steps": policy_steps,
        "runtime_snapshot": sps_payload["engine_fingerprint"]["runtime_snapshot"],
        "safety_margin_ms": safety_margin_ms,
        "selector_identity_sha256": EXACT_SPS_SELECTOR_IDENTITY_SHA256,
        "selector_replay_sha256": "e" * 64,
        "source_diff_sha256": sps_payload["engine_fingerprint"]["source_diff_sha256"],
        "source_head": sps_payload["engine_fingerprint"]["source_head"],
        "sps_cost_table_sha256": hashlib.sha256(sps_path.read_bytes()).hexdigest(),
        "workload_identity_sha256": "f" * 64,
    }
    receipt = {
        "schema_version": 1,
        "admission": admission,
        "admission_sha256": hashlib.sha256(
            json.dumps(admission, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    receipt_path = tmp_path / "admission.json"
    receipt_path.write_text(json.dumps(receipt))
    return (
        receipt_path,
        sps_path,
        fingerprint_path,
        receipt,
        hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )


def test_workload_admission_has_no_policy_steps_per_choice_fixed_point():
    # A selector that charges H*(N/C) back to each candidate has no stable
    # finite C for this profitable workload. Starting from all 50 positive
    # marginal choices gives R=2 and keeps only the ten 8-ms choices; R then
    # becomes 10 and rejects those too. Aggregate accounting admits because
    # 10*8 + 40*1.9 - 100*1 = 56 ms before the explicit margin.
    marginal_values_ms = [8.0] * 10 + [1.9] * 40 + [0.0] * 50

    assert sum(value > 2.0 for value in marginal_values_ms) == 10
    assert sum(value > 10.0 for value in marginal_values_ms) == 0
    admitted, net_value_ms = evaluate_confidence_workload_admission(
        policy_steps=100,
        compact_choices=50,
        gross_compact_value_ms_lower_bound=sum(marginal_values_ms),
        fixed_confidence_path_overhead_ms_upper_bound=1.0,
        safety_margin_ms=0.0,
    )

    assert admitted is True
    assert net_value_ms == pytest.approx(56.0)


def test_workload_admission_rejects_stale_ratio_undercharge():
    # A stale R=2 would admit each 8-ms opportunity. If that filtering leaves
    # only ten compact choices, however, the workload still pays 100 policy
    # steps of overhead: 80 - 100 = -20 ms. The aggregate gate cannot be fooled
    # by the stale ratio because R is not an input.
    assert 8.0 > 1.0 * 2.0
    admitted, net_value_ms = evaluate_confidence_workload_admission(
        policy_steps=100,
        compact_choices=10,
        gross_compact_value_ms_lower_bound=80.0,
        fixed_confidence_path_overhead_ms_upper_bound=1.0,
        safety_margin_ms=0.0,
    )

    assert admitted is False
    assert net_value_ms == pytest.approx(-20.0)


def test_workload_admission_requires_a_compact_choice():
    admitted, net_value_ms = evaluate_confidence_workload_admission(
        policy_steps=100,
        compact_choices=0,
        gross_compact_value_ms_lower_bound=200.0,
        fixed_confidence_path_overhead_ms_upper_bound=1.0,
        safety_margin_ms=10.0,
    )

    assert admitted is False
    assert net_value_ms == pytest.approx(90.0)


def test_workload_admission_authenticates_and_recomputes_positive_decision(tmp_path):
    receipt_path, sps_path, fingerprint_path, receipt, receipt_sha256 = _write_workload_admission(
        tmp_path
    )

    admission = load_confidence_workload_admission(
        receipt_path,
        expected_receipt_sha256=receipt_sha256,
        sps_cost_table_path=sps_path,
        live_engine_fingerprint_path=fingerprint_path,
        physical_k=5,
    )

    assert admission.admitted is True
    assert admission.net_value_ms_lower_bound == pytest.approx(50.0)
    assert admission.admission_identity_sha256 == receipt["admission_sha256"]
    assert admission.receipt_sha256 == receipt_sha256


def test_workload_admission_rejects_unpinned_receipt_bytes(tmp_path):
    receipt_path, sps_path, fingerprint_path, _, _ = _write_workload_admission(tmp_path)

    with pytest.raises(ValueError, match="pinned SHA256"):
        load_confidence_workload_admission(
            receipt_path,
            expected_receipt_sha256="0" * 64,
            sps_cost_table_path=sps_path,
            live_engine_fingerprint_path=fingerprint_path,
            physical_k=5,
        )


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda receipt: receipt["admission"].update({"selector_identity_sha256": "0" * 64}),
            "selector identity",
        ),
        (
            lambda receipt: receipt["admission"].update({"physical_k": 4}),
            "physical_k",
        ),
        (
            lambda receipt: receipt["admission"].update({"source_head": "stale"}),
            "source_head",
        ),
        (
            lambda receipt: receipt["admission"].update({"policy_steps": 1000}),
            "decision does not match",
        ),
    ],
)
def test_workload_admission_rejects_stale_or_inconsistent_receipt(tmp_path, mutation, message):
    receipt_path, sps_path, fingerprint_path, receipt, _ = _write_workload_admission(tmp_path)
    mutation(receipt)
    receipt["admission_sha256"] = hashlib.sha256(
        json.dumps(receipt["admission"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=message):
        load_confidence_workload_admission(
            receipt_path,
            expected_receipt_sha256=receipt_sha256,
            sps_cost_table_path=sps_path,
            live_engine_fingerprint_path=fingerprint_path,
            physical_k=5,
        )
