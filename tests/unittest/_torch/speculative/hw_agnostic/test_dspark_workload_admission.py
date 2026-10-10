# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for DSpark workload-level confidence admission."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

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


@pytest.mark.parametrize("gross_value,expected_net", [(110.0, 0.0), (100.0, -10.0)])
def test_workload_admission_declines_nonpositive_net_value(
    tmp_path: Path, gross_value: float, expected_net: float
) -> None:
    receipt_path, sps_path, fingerprint_path, _, receipt_sha256 = _write_workload_admission(
        tmp_path, admitted=False, gross_compact_value_ms_lower_bound=gross_value
    )

    admission = load_confidence_workload_admission(
        receipt_path,
        expected_receipt_sha256=receipt_sha256,
        sps_cost_table_path=sps_path,
        live_engine_fingerprint_path=fingerprint_path,
        physical_k=5,
    )

    assert admission.admitted is False
    assert admission.net_value_ms_lower_bound == expected_net


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("policy_steps", True, TypeError),
        ("policy_steps", 0, ValueError),
        ("compact_choices", 101, ValueError),
        ("gross_compact_value_ms_lower_bound", float("nan"), ValueError),
        ("fixed_confidence_path_overhead_ms_upper_bound", float("inf"), ValueError),
        ("safety_margin_ms", -1.0, ValueError),
    ],
)
def test_workload_admission_rejects_invalid_economics(
    field: str, value: object, error: type[Exception]
) -> None:
    inputs = dict(
        policy_steps=100,
        compact_choices=50,
        gross_compact_value_ms_lower_bound=160.0,
        fixed_confidence_path_overhead_ms_upper_bound=1.0,
        safety_margin_ms=10.0,
    )
    inputs[field] = value

    with pytest.raises(error):
        evaluate_confidence_workload_admission(**inputs)


@pytest.mark.parametrize("artifact", ["sps", "fingerprint"])
def test_workload_admission_rejects_changed_artifact_bytes(tmp_path: Path, artifact: str) -> None:
    receipt_path, sps_path, fingerprint_path, _, receipt_sha256 = _write_workload_admission(
        tmp_path
    )
    changed = sps_path if artifact == "sps" else fingerprint_path
    # Equivalent JSON is still a different artifact from the pinned bytes.
    changed.write_bytes(changed.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA256 does not match"):
        load_confidence_workload_admission(
            receipt_path,
            expected_receipt_sha256=receipt_sha256,
            sps_cost_table_path=sps_path,
            live_engine_fingerprint_path=fingerprint_path,
            physical_k=5,
        )


def test_workload_admission_parses_the_pinned_receipt_bytes(tmp_path: Path, monkeypatch) -> None:
    receipt_path, sps_path, fingerprint_path, receipt, receipt_sha256 = _write_workload_admission(
        tmp_path
    )
    replacement = deepcopy(receipt)
    replacement["admission"]["gross_compact_value_ms_lower_bound"] = 170.0
    replacement["admission_sha256"] = hashlib.sha256(
        json.dumps(replacement["admission"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    replacement_bytes = json.dumps(replacement).encode()
    original_open = Path.open
    replaced = False

    class ReplaceOnClose:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self.file.__enter__()

        def __exit__(self, *args):
            result = self.file.__exit__(*args)
            with original_open(receipt_path, "wb") as file:
                file.write(replacement_bytes)
            return result

    def open_with_replacement(path, mode="r", *args, **kwargs):
        nonlocal replaced
        file = original_open(path, mode, *args, **kwargs)
        if path == receipt_path and mode == "rb" and not replaced:
            replaced = True
            return ReplaceOnClose(file)
        return file

    monkeypatch.setattr(Path, "open", open_with_replacement)
    admission = load_confidence_workload_admission(
        receipt_path,
        expected_receipt_sha256=receipt_sha256,
        sps_cost_table_path=sps_path,
        live_engine_fingerprint_path=fingerprint_path,
        physical_k=5,
    )

    assert replaced is True
    assert receipt_path.read_bytes() == replacement_bytes
    assert admission.gross_compact_value_ms_lower_bound == 160.0


def test_workload_admission_rejects_cost_table_over_runtime_cell_limit(tmp_path: Path) -> None:
    receipt_path, sps_path, fingerprint_path, receipt, _ = _write_workload_admission(tmp_path)
    sps_payload = json.loads(sps_path.read_text())
    budgets = [0, 64, 128, 192, 256, 320, 352]
    times = [6.0, 5.9, 5.8, 5.7, 5.6, 5.5, 5.2]
    sps_payload["cost_tables"]["64"] = {"token_counts": budgets, "step_time_ms": times}
    sps_payload["measurements"] = [
        {
            "rank_local_graph_batch_size": int(graph_batch_size),
            "rank_local_verifier_budget": budget,
            "step_time_ms": step_time,
            "source_result_sha256": "c" * 64,
        }
        for graph_batch_size, cells in sps_payload["cost_tables"].items()
        for budget, step_time in zip(cells["token_counts"], cells["step_time_ms"])
    ]
    sps_path.write_text(json.dumps(sps_payload))
    receipt["admission"]["sps_cost_table_sha256"] = hashlib.sha256(
        sps_path.read_bytes()
    ).hexdigest()
    receipt["admission_sha256"] = hashlib.sha256(
        json.dumps(receipt["admission"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="production limit.*compact V cells per G"):
        load_confidence_workload_admission(
            receipt_path,
            expected_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            sps_cost_table_path=sps_path,
            live_engine_fingerprint_path=fingerprint_path,
            physical_k=5,
        )


@pytest.mark.parametrize("field", ["schema_version", "admitted"])
def test_workload_admission_rejects_duplicate_json_keys(tmp_path: Path, field: str) -> None:
    receipt_path, sps_path, fingerprint_path, _, _ = _write_workload_admission(tmp_path)
    text = receipt_path.read_text()
    literal = '"schema_version": 1' if field == "schema_version" else '"admitted": true'
    assert text.count(literal) == 1
    receipt_path.write_text(text.replace(literal, literal + ", " + literal))

    with pytest.raises(ValueError, match="duplicate field"):
        load_confidence_workload_admission(
            receipt_path,
            expected_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            sps_cost_table_path=sps_path,
            live_engine_fingerprint_path=fingerprint_path,
            physical_k=5,
        )


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
