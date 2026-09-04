# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark cost-table and scheduling unit tests (hardware-agnostic, CPU)."""

import hashlib
import json

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_planner import (
    ExactSpsCostRow,
    ExactSpsCostTable,
    ExactSpsDrainGuard,
    load_runtime_sps_cost_table,
    select_exact_sps_candidate,
    validate_sps_cost_table_payload,
)
from tensorrt_llm._torch.speculative.dspark_schedule import DSparkScheduleConfig, compute_survival

BLOCK = 7


def _drain_guard(
    *,
    tail_graph_batch_size=128,
    loss_multiplier=1.0,
    mean_output_tokens_per_request_iteration=10.0,
    minimum_group_value_ms=0.0,
    source_result_sha256="d" * 64,
):
    return ExactSpsDrainGuard(
        loss_multiplier=loss_multiplier,
        mean_output_tokens_per_request_iteration=(mean_output_tokens_per_request_iteration),
        minimum_group_value_ms=minimum_group_value_ms,
        tail_graph_batch_size=tail_graph_batch_size,
        source_result_sha256=source_result_sha256,
    )


def _cfg(**kw) -> DSparkScheduleConfig:
    return DSparkScheduleConfig(block_size=BLOCK, **kw)


# --------------------------------------------------------------------------
# survival
# --------------------------------------------------------------------------


def test_survival_is_cumulative_product():
    conf = torch.tensor([[0.9, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0]])
    surv = compute_survival(conf)
    assert torch.allclose(surv[0, :3], torch.tensor([0.9, 0.72, 0.36]), atol=1e-6)


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


def _load_test_exact(path, payload):
    fingerprint_path = path.with_name(f"{path.stem}-live-fingerprint.json")
    fingerprint_path.write_text(json.dumps(payload["engine_fingerprint"]))
    return load_runtime_sps_cost_table(
        path,
        graph_batch_sizes=[64, 128],
        max_draft_len=5,
        live_engine_fingerprint_path=fingerprint_path,
    )


def test_exact_cost_table_never_interpolates(tmp_path):
    payload = _multi_g_sps_payload()
    path = tmp_path / "multi-g-sps.json"
    path.write_text(json.dumps(payload))

    table, _ = _load_test_exact(path, payload)

    assert isinstance(table, ExactSpsCostTable)
    assert table.minimum_predicted_gain == pytest.approx(0.02)
    assert table.candidate_budgets(64) == (352,)
    assert table.candidate_budgets(128) == (704, 736)
    assert table.candidate_budgets(128, include_native=True) == (0, 704, 736)
    assert table.step_time(0, 128) == pytest.approx(8.0)
    assert table.step_time(704, 128) == pytest.approx(7.0)
    with pytest.raises(ValueError, match=r"no direct measurements for G=128, V=\[720\]"):
        table.step_time(720, 128)
    with pytest.raises(ValueError, match="no direct measurements for G=256"):
        table.step_time(704, 256)
    with pytest.raises(TypeError, match="requested verifier budget"):
        table.step_time(704.5, 128)


def test_exact_cost_table_rejects_fractional_programmatic_cells():
    table = ExactSpsCostRow(token_counts=(0, 704), step_time_ms=(8.0, 7.0))
    with pytest.raises(TypeError, match="graph batch size"):
        ExactSpsCostTable(tables={128.5: table}, max_draft_len=5)
    with pytest.raises(TypeError, match="measured verifier budget"):
        ExactSpsCostTable(
            tables={128: ExactSpsCostRow(token_counts=(0, 704.5), step_time_ms=(8.0, 7.0))},
            max_draft_len=5,
        )
    with pytest.raises(ValueError, match="V=0 native static K5"):
        ExactSpsCostTable(
            tables={128: ExactSpsCostRow(token_counts=(704,), step_time_ms=(7.0,))},
            max_draft_len=5,
        )


def test_exact_cost_table_identity_covers_grid_costs_and_policy():
    tables = {
        64: ExactSpsCostRow(token_counts=(0, 352), step_time_ms=(5.0, 4.5)),
        128: ExactSpsCostRow(token_counts=(0, 704), step_time_ms=(8.0, 7.0)),
    }
    first = ExactSpsCostTable(tables=tables, max_draft_len=5, minimum_predicted_gain=0.02)
    reordered = ExactSpsCostTable(
        tables=dict(reversed(tuple(tables.items()))),
        max_draft_len=5,
        minimum_predicted_gain=0.02,
    )
    changed_cost = ExactSpsCostTable(
        tables={
            **tables,
            128: ExactSpsCostRow(token_counts=(0, 704), step_time_ms=(8.0, 7.1)),
        },
        max_draft_len=5,
        minimum_predicted_gain=0.02,
    )
    changed_policy = ExactSpsCostTable(tables=tables, max_draft_len=5, minimum_predicted_gain=0.03)
    guarded = ExactSpsCostTable(
        tables=tables,
        max_draft_len=5,
        minimum_predicted_gain=0.02,
        iteration_drain_guard=_drain_guard(tail_graph_batch_size=64),
    )

    assert first.identity_sha256 == reordered.identity_sha256
    assert first.collective_identity_words == reordered.collective_identity_words
    assert len(first.collective_identity_words) == 8
    assert all(0 <= value <= 0xFFFFFFFF for value in first.collective_identity_words)
    assert first.identity_sha256 != changed_cost.identity_sha256
    assert first.identity_sha256 != changed_policy.identity_sha256
    assert first.identity_sha256 != guarded.identity_sha256


def test_iteration_drain_guard_rejects_invalid_or_unmeasured_metadata():
    with pytest.raises(ValueError, match="loss_multiplier"):
        _drain_guard(loss_multiplier=0.0)
    with pytest.raises(ValueError, match="mean_output_tokens"):
        _drain_guard(mean_output_tokens_per_request_iteration=float("nan"))
    with pytest.raises(ValueError, match="minimum_group_value_ms"):
        _drain_guard(minimum_group_value_ms=-0.1)

    with pytest.raises(ValueError, match="measured native tail table for G=16"):
        ExactSpsCostTable(
            tables={128: ExactSpsCostRow(token_counts=(0,), step_time_ms=(8.0,))},
            max_draft_len=5,
            iteration_drain_guard=_drain_guard(tail_graph_batch_size=16),
        )


@pytest.mark.parametrize("invalid_budget", [127, 769])
def test_exact_cost_table_direct_construction_bounds_positive_cells(invalid_budget):
    with pytest.raises(ValueError, match=r"G <= V <= G\*\(K\+1\)"):
        ExactSpsCostTable(
            tables={
                128: ExactSpsCostRow(token_counts=(0, invalid_budget), step_time_ms=(8.0, 7.0))
            },
            max_draft_len=5,
        )


def test_runtime_loader_rejects_legacy_cost_curve(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "token_counts": [0, 100],
                "step_time_ms": [1.0, 5.0],
                "_meta": {"lookup": "interp"},
            }
        )
    )

    with pytest.raises(ValueError, match="schema_version=2"):
        load_runtime_sps_cost_table(
            path,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
        )


def test_exact_cost_table_loader_requires_schema_v2(tmp_path):
    payload = _multi_g_sps_payload()
    payload["schema_version"] = 1
    path = tmp_path / "wrong-version.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_version=2"):
        _load_test_exact(path, payload)

    payload = _multi_g_sps_payload()
    del payload["cost_tables"]
    path = tmp_path / "missing-tables.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing required fields: cost_tables"):
        _load_test_exact(path, payload)
    payload = _multi_g_sps_payload()
    del payload["schema_version"]
    path = tmp_path / "missing-version.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_version=2"):
        _load_test_exact(path, payload)


def test_runtime_exact_loader_requires_independent_fingerprint_file(tmp_path):
    payload = _multi_g_sps_payload()
    table_path = tmp_path / "exact-sps.json"
    table_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="independently generated"):
        load_runtime_sps_cost_table(
            table_path,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
        )


@pytest.mark.parametrize("marker", ["engine_fingerprint", "measurements"])
def test_partial_v2_payload_cannot_bypass_schema_validation(tmp_path, marker):
    exact_payload = _multi_g_sps_payload()
    mixed_payload = {
        "token_counts": [0, 100],
        "step_time_ms": [1.0, 5.0],
        marker: exact_payload[marker],
    }
    path = tmp_path / f"mixed-{marker}.json"
    path.write_text(json.dumps(mixed_payload))
    fingerprint_path = tmp_path / f"mixed-{marker}-live-fingerprint.json"
    fingerprint_path.write_text(json.dumps(exact_payload["engine_fingerprint"]))

    with pytest.raises(ValueError, match="schema_version=2"):
        load_runtime_sps_cost_table(
            path,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint_path=fingerprint_path,
        )


def test_schema_v2_validation_accepts_exact_runtime_grid():
    payload = _multi_g_sps_payload()

    fingerprint = validate_sps_cost_table_payload(
        payload,
        graph_batch_sizes=[64, 128],
        max_draft_len=5,
        live_engine_fingerprint=dict(payload["engine_fingerprint"]),
    )

    assert fingerprint["source_head"] == "23b73d8"


def test_schema_v2_loads_authenticated_iteration_drain_metadata(tmp_path):
    payload = _multi_g_sps_payload()
    payload["iteration_drain_guard"] = {
        "loss_multiplier": 1.5,
        "mean_output_tokens_per_request_iteration": 4.25,
        "minimum_group_value_ms": 2.0,
        "source_result_sha256": "d" * 64,
        "tail_graph_batch_size": 64,
    }
    path = tmp_path / "multi-g-sps-with-drain-guard.json"
    path.write_text(json.dumps(payload))

    table, _ = _load_test_exact(path, payload)

    assert table.iteration_drain_guard == _drain_guard(
        tail_graph_batch_size=64,
        loss_multiplier=1.5,
        mean_output_tokens_per_request_iteration=4.25,
        minimum_group_value_ms=2.0,
    )


def test_schema_v2_iteration_drain_metadata_requires_measured_tail_g(tmp_path):
    payload = _multi_g_sps_payload()
    payload["iteration_drain_guard"] = {
        "loss_multiplier": 1.5,
        "mean_output_tokens_per_request_iteration": 4.25,
        "minimum_group_value_ms": 2.0,
        "source_result_sha256": "d" * 64,
        "tail_graph_batch_size": 16,
    }
    path = tmp_path / "multi-g-sps-missing-tail-g.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="measured native tail table for G=16"):
        _load_test_exact(path, payload)


def test_schema_v2_validation_requires_the_runtime_graph_grid():
    payload = _multi_g_sps_payload()
    with pytest.raises(ValueError, match="must match exactly"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )


def test_schema_v2_validation_requires_independent_live_fingerprint():
    payload = _multi_g_sps_payload()
    with pytest.raises(ValueError, match="independently supplied"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
        )
    with pytest.raises(ValueError, match="independently of the SPS"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=payload["engine_fingerprint"],
        )


def test_schema_v2_validation_authenticates_and_matches_fingerprint():
    payload = _multi_g_sps_payload()
    live = dict(payload["engine_fingerprint"])
    payload["engine_fingerprint"]["topology"] = "TEP8"
    with pytest.raises(ValueError, match="SHA256"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=live,
        )

    payload = _multi_g_sps_payload()
    live = dict(payload["engine_fingerprint"])
    live["runtime_snapshot"] = "different"
    with pytest.raises(ValueError, match="does not match active runtime"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=live,
        )


def test_schema_v2_validation_requires_native_v0_for_every_g():
    payload = _multi_g_sps_payload()
    payload["cost_tables"]["128"]["token_counts"].pop(0)
    payload["cost_tables"]["128"]["step_time_ms"].pop(0)
    payload["measurements"] = [
        item
        for item in payload["measurements"]
        if not (
            item["rank_local_graph_batch_size"] == 128 and item["rank_local_verifier_budget"] == 0
        )
    ]

    with pytest.raises(ValueError, match="V=0 native static K5"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )


@pytest.mark.parametrize("invalid_budget", [64, 784])
def test_schema_v2_validation_bounds_positive_verifier_budgets(invalid_budget):
    payload = _multi_g_sps_payload()
    original_budget = 704 if invalid_budget == 64 else 736
    table_index = 1 if invalid_budget == 64 else 2
    payload["cost_tables"]["128"]["token_counts"][table_index] = invalid_budget
    for measurement in payload["measurements"]:
        if (
            measurement["rank_local_graph_batch_size"] == 128
            and measurement["rank_local_verifier_budget"] == original_budget
        ):
            measurement["rank_local_verifier_budget"] = invalid_budget

    with pytest.raises(ValueError, match=r"G <= V <= G\*\(K\+1\)"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )


def test_schema_v2_validation_rejects_extra_or_mismatched_provenance():
    payload = _multi_g_sps_payload()
    payload["cost_tables"]["128"]["token_counts"].append(752)
    payload["cost_tables"]["128"]["step_time_ms"].append(7.4)
    with pytest.raises(ValueError, match="cells must match exactly"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )

    payload = _multi_g_sps_payload()
    payload["measurements"][0]["step_time_ms"] += 0.1
    with pytest.raises(ValueError, match="do not match measurement"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda payload: payload.update({"unknown": 1}), "unknown fields"),
        (lambda payload: payload.update({"measurements": None}), "null fields"),
        (lambda payload: payload["cost_tables"]["128"].update({"unknown": 1}), "unknown fields"),
        (lambda payload: payload["engine_fingerprint"].update({"unknown": 1}), "unknown fields"),
    ],
)
def test_schema_v2_rejects_null_and_unknown_fields(tmp_path, mutation, message):
    payload = _multi_g_sps_payload()
    mutation(payload)
    path = tmp_path / "invalid-shape.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        _load_test_exact(path, payload)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda payload: payload["cost_tables"]["128"]["token_counts"].__setitem__(1, 704.0),
            "JSON integer",
        ),
        (
            lambda payload: payload["measurements"][1].update(
                {"rank_local_verifier_budget": 352.5}
            ),
            "JSON integer",
        ),
        (
            lambda payload: payload.update({"minimum_predicted_gain": -0.1}),
            "non-negative and finite",
        ),
        (
            lambda payload: payload.update({"minimum_predicted_gain": float("nan")}),
            "non-negative and finite",
        ),
        (
            lambda payload: payload["cost_tables"]["64"]["step_time_ms"].__setitem__(0, 0.0),
            "positive and finite",
        ),
        (
            lambda payload: payload["cost_tables"]["64"]["step_time_ms"].__setitem__(
                0, float("inf")
            ),
            "positive and finite",
        ),
    ],
)
def test_schema_v2_rejects_fractional_or_invalid_numbers(tmp_path, mutation, message):
    payload = _multi_g_sps_payload()
    mutation(payload)
    path = tmp_path / "invalid-number.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((TypeError, ValueError), match=message):
        _load_test_exact(path, payload)


def test_schema_v2_rejects_fractional_graph_key(tmp_path):
    payload = _multi_g_sps_payload()
    payload["cost_tables"]["128.0"] = payload["cost_tables"].pop("128")
    path = tmp_path / "fractional-g.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match="canonical positive integer string"):
        _load_test_exact(path, payload)


def test_exact_selector_keeps_native_on_tie_or_below_threshold():
    table = ExactSpsCostTable(
        tables={128: ExactSpsCostRow(token_counts=(0, 704), step_time_ms=(10.0, 8.0))},
        max_draft_len=5,
        minimum_predicted_gain=0.02,
        iteration_drain_guard=_drain_guard(),
    )

    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={704: 8.0},
            compact_max_yield_losses_per_request={704: 2.0},
            cost_table=table,
        )
        == 0
    )
    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={704: 8.08},
            compact_max_yield_losses_per_request={704: 1.92},
            cost_table=table,
        )
        == 0
    )
    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={704: 8.24},
            compact_max_yield_losses_per_request={704: 1.76},
            cost_table=table,
        )
        == 704
    )
    with pytest.raises(ValueError, match="must not include native V=0"):
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={0: 10.0, 704: 8.24},
            compact_max_yield_losses_per_request={704: 1.76},
            cost_table=table,
        )


def test_exact_selector_fails_closed_without_iteration_drain_metadata():
    table = ExactSpsCostTable(
        tables={128: ExactSpsCostRow(token_counts=(0, 512), step_time_ms=(100.0, 80.0))},
        max_draft_len=5,
        minimum_predicted_gain=0.0,
    )

    # Compact has positive immediate expected goodput, but the table cannot
    # price its workload-specific iteration loss.
    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={512: 8.5},
            compact_max_yield_losses_per_request={512: 1.5},
            cost_table=table,
        )
        == 0
    )


def test_exact_selector_applies_strict_iteration_drain_group_value():
    table = ExactSpsCostTable(
        tables={
            16: ExactSpsCostRow(token_counts=(0,), step_time_ms=(40.0,)),
            128: ExactSpsCostRow(token_counts=(0, 512), step_time_ms=(100.0, 80.0)),
        },
        max_draft_len=5,
        minimum_predicted_gain=0.0,
        iteration_drain_guard=_drain_guard(
            tail_graph_batch_size=16,
            loss_multiplier=1.5,
            mean_output_tokens_per_request_iteration=5.0,
            minimum_group_value_ms=2.0,
        ),
    )

    # T(128,0)-T(128,512) - 1.5*(10-8.5)*T(16,0)/5 == 2ms.
    # Equality is deliberately native.
    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={512: 8.5},
            compact_max_yield_losses_per_request={512: 1.5},
            cost_table=table,
        )
        == 0
    )
    # A 1.4-token predicted loss has group value 3.2ms and is admitted.
    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={512: 8.6},
            compact_max_yield_losses_per_request={512: 1.4},
            cost_table=table,
        )
        == 512
    )


@pytest.mark.parametrize("graph_batch_size", [16, 32, 64, 128])
def test_exact_selector_reranks_every_g_by_guarded_group_value(graph_batch_size):
    table = ExactSpsCostTable(
        tables={
            graph_batch_size: ExactSpsCostRow(
                token_counts=(
                    0,
                    3 * graph_batch_size,
                    4 * graph_batch_size,
                    5 * graph_batch_size,
                ),
                step_time_ms=(100.0, 60.0, 70.0, 80.0),
            )
        },
        max_draft_len=5,
        minimum_predicted_gain=0.01,
        iteration_drain_guard=_drain_guard(
            tail_graph_batch_size=graph_batch_size,
            mean_output_tokens_per_request_iteration=20.0,
            minimum_group_value_ms=20.0,
        ),
    )

    # V=4G wins aggregate immediate goodput, but its worst active rank loses
    # enough yield per request to miss the E2E guard. V=3G is the best guarded
    # tier. V=5G does not clear the aggregate immediate-goodput threshold.
    assert (
        select_exact_sps_candidate(
            graph_batch_size=graph_batch_size,
            native_expected_yield=10.0,
            compact_expected_yields={
                3 * graph_batch_size: 6.2,
                4 * graph_batch_size: 7.5,
                5 * graph_batch_size: 8.0,
            },
            compact_max_yield_losses_per_request={
                3 * graph_batch_size: 0.5,
                4 * graph_batch_size: 3.0,
                5 * graph_batch_size: 0.0,
            },
            cost_table=table,
        )
        == 3 * graph_batch_size
    )


def test_exact_selector_fails_closed_without_measured_compact_cell():
    graph_batch_size = 32
    table = ExactSpsCostTable(
        tables={
            16: ExactSpsCostRow(token_counts=(0,), step_time_ms=(40.0,)),
            32: ExactSpsCostRow(token_counts=(0,), step_time_ms=(55.0,)),
            64: ExactSpsCostRow(token_counts=(0,), step_time_ms=(70.0,)),
            128: ExactSpsCostRow(token_counts=(0, 512), step_time_ms=(100.0, 80.0)),
        },
        max_draft_len=5,
        minimum_predicted_gain=0.0,
        iteration_drain_guard=_drain_guard(tail_graph_batch_size=16),
    )

    assert (
        select_exact_sps_candidate(
            graph_batch_size=graph_batch_size,
            native_expected_yield=10.0,
            compact_expected_yields={},
            compact_max_yield_losses_per_request={},
            cost_table=table,
        )
        == 0
    )


def test_exact_production_candidates_exclude_full_ragged_control():
    table = ExactSpsCostTable(
        tables={128: ExactSpsCostRow(token_counts=(0, 704, 768), step_time_ms=(8.0, 7.0, 8.1))},
        max_draft_len=5,
    )
    assert table.candidate_budgets(128) == (704, 768)
    assert table.production_candidate_budgets(128) == (704,)
    assert table.candidate_cells() == ((128, 704),)


def test_exact_table_limits_compact_graph_cells_per_g():
    with pytest.raises(ValueError, match="compact V cells per G"):
        ExactSpsCostTable(
            tables={
                128: ExactSpsCostRow(
                    token_counts=(0, 128, 256, 384, 512, 640),
                    step_time_ms=(8.0, 7.9, 7.8, 7.7, 7.6, 7.5),
                )
            },
            max_draft_len=5,
        )
