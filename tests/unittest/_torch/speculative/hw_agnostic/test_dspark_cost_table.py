# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark cost-table and scheduling unit tests (hardware-agnostic, CPU)."""

import json

import pytest
import torch

from tensorrt_llm._torch.speculative import dspark_planner
from tensorrt_llm._torch.speculative.dspark_planner import (
    CostTableSetup,
    ExactSpsCostRow,
    ExactSpsCostTable,
    build_cost_table_payload,
    load_runtime_sps_cost_table,
    merge_cost_table_payloads,
    select_exact_sps_candidate,
)
from tensorrt_llm._torch.speculative.dspark_schedule import compute_survival

SETUP = CostTableSetup(
    gpu="NVIDIA B300",
    world_size=8,
    tensor_parallel_size=8,
    moe_expert_parallel_size=8,
    enable_attention_dp=True,
    tensorrt_llm_version="1.4.0",
)


def _payload():
    """Schema-3 table with K3 and K5 entries."""
    return build_cost_table_payload(
        setup=SETUP,
        costs={
            3: {64: [(0, 44.0), (192, 41.0)], 128: [(0, 60.0), (384, 53.0)]},
            5: {
                16: [(0, 33.6), (48, 30.8)],
                64: [(0, 49.2), (224, 42.7), (256, 43.6)],
                128: [(0, 75.6), (448, 57.2), (512, 59.7), (576, 65.6), (640, 69.6)],
            },
        },
    )


def _write(tmp_path, payload, name="costs.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _load(path, *, k=5, graphs=(64, 128), gain=0.04, live=SETUP):
    return load_runtime_sps_cost_table(
        path,
        graph_batch_sizes=list(graphs),
        max_draft_len=k,
        minimum_predicted_gain=gain,
        live_setup=live,
    )


def _with(setup, **changes):
    values = setup.to_json()
    values.update(changes)
    return CostTableSetup(**values)


def test_survival_is_cumulative_product():
    conf = torch.tensor([[0.9, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0]])
    surv = compute_survival(conf)
    assert torch.allclose(surv[0, :3], torch.tensor([0.9, 0.72, 0.36]), atol=1e-6)


# --------------------------------------------------------------------------
# Loading: one file for several K, captured-ladder coverage, setup checks
# --------------------------------------------------------------------------


def test_loader_selects_the_runtime_k_and_never_interpolates(tmp_path):
    path = _write(tmp_path, _payload())

    table = _load(path, k=5, gain=0.02)

    assert isinstance(table, ExactSpsCostTable)
    assert table.max_draft_len == 5
    assert table.minimum_predicted_gain == pytest.approx(0.02)
    assert table.candidate_budgets(128, include_native=True) == (0, 448, 512, 576, 640)
    assert table.step_time(448, 128) == pytest.approx(57.2)
    with pytest.raises(ValueError, match=r"no direct measurements for G=128, V=\[450\]"):
        table.step_time(450, 128)
    k3 = _load(path, k=3)
    assert k3.max_draft_len == 3
    assert k3.step_time(384, 128) == pytest.approx(53.0)


def test_loader_installs_only_captured_graph_sizes(tmp_path):
    table = _load(_write(tmp_path, _payload()), graphs=(128,))
    assert sorted(table.tables) == [128]
    assert table.candidate_cells() == ((128, 448), (128, 512), (128, 576), (128, 640))


def test_loader_reports_a_missing_k_with_a_profile_command(tmp_path):
    path = _write(tmp_path, _payload())
    with pytest.raises(
        ValueError, match=r"no costs for max_draft_len=4 \(it has K=\[3, 5\]\)"
    ) as error:
        _load(path, k=4)
    assert "dspark_profile_costs" in str(error.value)
    assert "--k 4" in str(error.value)


def test_loader_reports_uncovered_captured_graph_sizes(tmp_path):
    path = _write(tmp_path, _payload())
    with pytest.raises(
        ValueError, match=r"captured graph batch sizes \[32\]; it covers \[16, 64, 128\]"
    ):
        _load(path, graphs=(32, 64, 128))


def test_loader_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist. Generate one with"):
        _load(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "field, value",
    [
        ("gpu", "NVIDIA B200"),
        ("world_size", 4),
        ("tensor_parallel_size", 4),
        ("moe_expert_parallel_size", 4),
        ("enable_attention_dp", False),
    ],
)
def test_loader_rejects_tables_from_another_setup(tmp_path, field, value):
    path = _write(tmp_path, _payload())
    with pytest.raises(ValueError, match=f"different setup \\({field}: "):
        _load(path, live=_with(SETUP, **{field: value}))


def test_loader_warns_on_another_tensorrt_llm_version(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(dspark_planner.logger, "warning", warnings.append)
    _load(_write(tmp_path, _payload()), live=_with(SETUP, tensorrt_llm_version="1.5.0"))
    assert len(warnings) == 1
    assert "measured with TensorRT-LLM 1.4.0; running 1.5.0" in warnings[0]


def _legacy_v2_payload(max_draft_len=5):
    return {
        "schema_version": 2,
        "minimum_predicted_gain": 0.04,
        "cost_tables": {
            "64": {"token_counts": [0, 224], "step_time_ms": [49.2, 42.7]},
            "128": {"token_counts": [0, 448], "step_time_ms": [75.6, 57.2]},
        },
        "engine_fingerprint": {
            "gpu": "NVIDIA B300",
            "gpu_count": 8,
            "max_draft_len": max_draft_len,
            "rank_local_graph_batch_sizes": [64, 128],
            "topology": json.dumps({"attention_dp": True, "cp": 1, "ep": 8, "pp": 1, "tp": 8}),
        },
        "engine_fingerprint_sha256": "0" * 64,
        "measurements": [],
    }


def test_loader_reads_legacy_schema_v2_tables(tmp_path):
    table = _load(_write(tmp_path, _legacy_v2_payload()), gain=0.0)
    assert table.step_time(448, 128) == pytest.approx(57.2)
    # The runtime gain comes from the config, not from the legacy file.
    assert table.minimum_predicted_gain == 0.0


def test_loader_rejects_a_legacy_table_for_another_k(tmp_path):
    path = _write(tmp_path, _legacy_v2_payload(max_draft_len=3))
    with pytest.raises(ValueError, match="measured for max_draft_len=3, but the runtime uses"):
        _load(path)


def test_loader_checks_legacy_topology(tmp_path):
    path = _write(tmp_path, _legacy_v2_payload())
    with pytest.raises(ValueError, match="moe_expert_parallel_size: table=8, runtime=4"):
        _load(path, live=_with(SETUP, moe_expert_parallel_size=4))


def test_loader_rejects_unknown_schema(tmp_path):
    path = _write(tmp_path, {"schema_version": 7})
    with pytest.raises(ValueError, match="unsupported schema_version=7"):
        _load(path)


# --------------------------------------------------------------------------
# Writing and merging tables
# --------------------------------------------------------------------------


def test_build_rejects_cells_the_runtime_would_reject():
    with pytest.raises(ValueError, match=r"G <= V <= G\*\(K\+1\)"):
        build_cost_table_payload(setup=SETUP, costs={5: {128: [(0, 75.0), (900, 70.0)]}})
    with pytest.raises(ValueError, match="V=0 native"):
        build_cost_table_payload(setup=SETUP, costs={5: {128: [(448, 57.0)]}})


def test_merge_replaces_profiled_rows_and_keeps_the_others():
    existing = _payload()
    new = build_cost_table_payload(
        setup=SETUP,
        costs={5: {128: [(0, 70.0), (448, 55.0)], 256: [(0, 120.0)]}, 4: {128: [(0, 66.0)]}},
    )
    merged = merge_cost_table_payloads(existing, new)
    assert list(merged["costs"]) == ["3", "4", "5"]
    assert merged["costs"]["3"] == existing["costs"]["3"]
    assert list(merged["costs"]["5"]) == ["16", "64", "128", "256"]
    assert merged["costs"]["5"]["64"] == existing["costs"]["5"]["64"]
    assert merged["costs"]["5"]["128"] == {"token_counts": [0, 448], "step_time_ms": [70.0, 55.0]}


def test_merge_keeps_per_k_profile_metadata():
    def profiled(k, graph_batch_size, step_ms, samples):
        return build_cost_table_payload(
            setup=SETUP,
            costs={k: {graph_batch_size: [(0, step_ms)]}},
            extra={"profile": {str(k): {"samples_per_cell": {f"{graph_batch_size}/0": samples}}}},
        )

    merged = merge_cost_table_payloads(profiled(4, 128, 66.0, 30), profiled(3, 128, 58.0, 26))
    assert merged["profile"] == {
        "4": {"samples_per_cell": {"128/0": 30}},
        "3": {"samples_per_cell": {"128/0": 26}},
    }
    merged = merge_cost_table_payloads(merged, profiled(4, 256, 120.0, 12))
    assert merged["profile"]["4"] == {"samples_per_cell": {"128/0": 30, "256/0": 12}}


def test_merge_refuses_tables_from_different_setups():
    other = build_cost_table_payload(
        setup=_with(SETUP, gpu="NVIDIA B200"), costs={5: {128: [(0, 70.0)]}}
    )
    with pytest.raises(ValueError, match="different setups: gpu"):
        merge_cost_table_payloads(_payload(), other)


# --------------------------------------------------------------------------
# Table invariants
# --------------------------------------------------------------------------


def test_exact_cost_table_rejects_fractional_programmatic_cells():
    table = ExactSpsCostRow(token_counts=(0, 704), step_time_ms=(8.0, 7.0))
    with pytest.raises(TypeError, match="graph batch size"):
        ExactSpsCostTable(tables={128.5: table}, max_draft_len=5)
    with pytest.raises(TypeError, match="measured verifier budget"):
        ExactSpsCostTable(
            tables={128: ExactSpsCostRow(token_counts=(0, 704.5), step_time_ms=(8.0, 7.0))},
            max_draft_len=5,
        )
    with pytest.raises(ValueError, match="V=0 native static physical-K"):
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

    assert first.identity_sha256 == reordered.identity_sha256
    assert first.collective_identity_words == reordered.collective_identity_words
    assert len(first.collective_identity_words) == 8
    assert all(0 <= value <= 0xFFFFFFFF for value in first.collective_identity_words)
    assert first.identity_sha256 != changed_cost.identity_sha256
    assert first.identity_sha256 != changed_policy.identity_sha256


@pytest.mark.parametrize("invalid_budget", [127, 769])
def test_exact_cost_table_direct_construction_bounds_positive_cells(invalid_budget):
    with pytest.raises(ValueError, match=r"G <= V <= G\*\(K\+1\)"):
        ExactSpsCostTable(
            tables={
                128: ExactSpsCostRow(token_counts=(0, invalid_budget), step_time_ms=(8.0, 7.0))
            },
            max_draft_len=5,
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


def test_exact_table_limits_total_compact_cells():
    graphs = (16, 24, 32, 40, 48, 56, 64, 72, 80)
    with pytest.raises(ValueError, match="limit of 32 compact"):
        ExactSpsCostTable(
            tables={
                graph: ExactSpsCostRow(
                    token_counts=(0, 2 * graph, 3 * graph, 4 * graph, 5 * graph),
                    step_time_ms=(10.0, 6.0, 7.0, 8.0, 9.0),
                )
                for graph in graphs
            },
            max_draft_len=5,
        )


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def _single_cell_table(gain):
    return ExactSpsCostTable(
        tables={128: ExactSpsCostRow(token_counts=(0, 704), step_time_ms=(10.0, 8.0))},
        max_draft_len=5,
        minimum_predicted_gain=gain,
    )


def test_selector_keeps_native_on_tie_or_below_the_gain_margin():
    table = _single_cell_table(0.02)
    # Equal goodput (8 / 8ms == 10 / 10ms) and +1% are both native.
    for compact_yield in (8.0, 8.08):
        assert (
            select_exact_sps_candidate(
                graph_batch_size=128,
                native_expected_yield=10.0,
                compact_expected_yields={704: compact_yield},
                cost_table=table,
            )
            == 0
        )
    # +3% clears the 2% margin.
    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={704: 8.24},
            cost_table=table,
        )
        == 704
    )
    with pytest.raises(ValueError, match="must not include native V=0"):
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={0: 10.0, 704: 8.24},
            cost_table=table,
        )


@pytest.mark.parametrize("graph_batch_size", [16, 64, 128, 256])
def test_selector_picks_the_highest_goodput_tier(graph_batch_size):
    table = ExactSpsCostTable(
        tables={
            graph_batch_size: ExactSpsCostRow(
                token_counts=(0, 3 * graph_batch_size, 4 * graph_batch_size, 5 * graph_batch_size),
                step_time_ms=(100.0, 60.0, 70.0, 80.0),
            )
        },
        max_draft_len=5,
        minimum_predicted_gain=0.04,
    )
    # Goodput: native 0.100, 3G 0.1033, 4G 0.1071, 5G 0.1000 tokens/ms.
    assert (
        select_exact_sps_candidate(
            graph_batch_size=graph_batch_size,
            native_expected_yield=10.0,
            compact_expected_yields={
                3 * graph_batch_size: 6.2,
                4 * graph_batch_size: 7.5,
                5 * graph_batch_size: 8.0,
            },
            cost_table=table,
        )
        == 4 * graph_batch_size
    )


def test_selector_keeps_native_without_a_measured_compact_cell():
    table = ExactSpsCostTable(
        tables={
            32: ExactSpsCostRow(token_counts=(0,), step_time_ms=(55.0,)),
            128: ExactSpsCostRow(token_counts=(0, 512), step_time_ms=(100.0, 80.0)),
        },
        max_draft_len=5,
        minimum_predicted_gain=0.0,
    )
    assert (
        select_exact_sps_candidate(
            graph_batch_size=32,
            native_expected_yield=10.0,
            compact_expected_yields={},
            cost_table=table,
        )
        == 0
    )
