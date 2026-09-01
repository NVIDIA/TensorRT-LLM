# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-independent tests for the DSpark confidence policy coordinator."""

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_planner import ExactSpsCostTable, SpsCostTable
from tensorrt_llm._torch.speculative.dspark_schedule import DSparkScheduleConfig
from tensorrt_llm._torch.speculative.dspark_verify import (
    DSparkVerifyPlanner,
    ExactSpsLocalDecision,
    _exact_cell_geometry,
)


def _config() -> DSparkScheduleConfig:
    return DSparkScheduleConfig(block_size=5, min_verify_len=1)


def _legacy_table() -> SpsCostTable:
    return SpsCostTable(
        token_counts=(0, 8, 24, 48),
        step_time_ms=(2.0, 2.1, 4.0, 9.0),
    )


def _planner(**kwargs) -> DSparkVerifyPlanner:
    kwargs.setdefault("cfg", _config())
    kwargs.setdefault("cost_table", _legacy_table())
    kwargs.setdefault("tiers", [1, 3, 5])
    return DSparkVerifyPlanner(**kwargs)


def _publish_snapshot(planner: DSparkVerifyPlanner, logits: torch.Tensor) -> None:
    planner._host_buffer = logits
    planner._copy_event = None
    planner._snapshot_valid = True


def test_runtime_cost_table_is_authoritative() -> None:
    planner = DSparkVerifyPlanner(cfg=_config(), tiers=[1, 3, 5])
    authoritative = ExactSpsCostTable(
        tables={4: SpsCostTable(token_counts=(0, 14), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )

    planner.install_runtime_cost_table(authoritative)
    planner.install_runtime_cost_table(authoritative)
    assert planner.cost_table is authoritative

    different = ExactSpsCostTable(
        tables={4: SpsCostTable(token_counts=(0, 14), step_time_ms=(8.0, 7.1))},
        max_draft_len=5,
    )
    with pytest.raises(RuntimeError, match="different SPS cost object"):
        planner.install_runtime_cost_table(different)


def test_runtime_cost_table_must_match_the_draft_block() -> None:
    planner = DSparkVerifyPlanner(cfg=_config(), tiers=[1, 3, 5])
    incompatible = ExactSpsCostTable(
        tables={4: SpsCostTable(token_counts=(0, 12), step_time_ms=(8.0, 7.0))},
        max_draft_len=4,
    )

    with pytest.raises(ValueError, match="max_draft_len does not match"):
        planner.install_runtime_cost_table(incompatible)


@pytest.mark.parametrize(
    ("num_real", "graph_batch_size", "verifier_budget", "expected"),
    [
        (100, 128, 704, (4, 592, 392)),
        (0, 128, 704, (5, 0, 0)),
        (4, 4, 24, (0, 24, 16)),
        (4, 4, 25, None),
        (5, 4, 14, None),
    ],
)
def test_exact_cell_geometry_matches_the_executed_layout(
    num_real: int,
    graph_batch_size: int,
    verifier_budget: int,
    expected: tuple[int, int, int] | None,
) -> None:
    assert (
        _exact_cell_geometry(
            num_real=num_real,
            graph_batch_size=graph_batch_size,
            verifier_budget=verifier_budget,
            min_verify_len=1,
            max_verify_len=5,
        )
        == expected
    )


def test_idle_attention_dp_rank_advertises_only_feasible_cells() -> None:
    table = ExactSpsCostTable(
        tables={
            4: SpsCostTable(
                token_counts=(0, 12, 14, 24),
                step_time_ms=(8.0, 6.0, 6.5, 8.1),
            )
        },
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table, tiers=[1, 3, 5])

    decision = planner.prepare_exact_sps_decision(num_gen_requests=0, rows=[])

    assert decision is not None
    assert decision.num_requests == 0
    assert tuple(decision.survival.shape) == (0, 5)
    assert decision.native_expected_yield == 0.0
    assert decision.compact_expected_yields == (0.0, 0.0)
    assert planner.allocate_exact_sps_candidate(
        decision, graph_batch_size=4, verifier_budget=14
    ) == ([], 0, 3)


def test_exact_policy_reads_confidence_by_request_row() -> None:
    table = ExactSpsCostTable(
        tables={4: SpsCostTable(token_counts=(0, 14), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table, tiers=[1, 3, 5])
    logits = torch.full((5, 5), -8.0)
    logits[3] = 8.0
    _publish_snapshot(planner, logits)

    decision = planner.prepare_exact_sps_decision(num_gen_requests=2, rows=[3, 0])

    assert decision is not None
    assert decision.survival[0].sum() > decision.survival[1].sum()


def test_exact_allocator_spends_the_modeled_real_row_target() -> None:
    table = ExactSpsCostTable(
        tables={4: SpsCostTable(token_counts=(0, 22), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table, tiers=[1, 3, 5])
    decision = ExactSpsLocalDecision(
        num_requests=3,
        survival=torch.ones((3, 5)),
        native_expected_yield=18.0,
        compact_expected_yields=(18.0,),
    )

    lens, budget, pad_tokens = planner.allocate_exact_sps_candidate(
        decision, graph_batch_size=4, verifier_budget=22
    )

    assert lens == [5, 5, 5]
    assert budget == 12
    assert pad_tokens == 4
    assert sum(value + 1 for value in lens) + pad_tokens == 22


def test_unmeasured_exact_cell_is_rejected() -> None:
    table = ExactSpsCostTable(
        tables={4: SpsCostTable(token_counts=(0, 14), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table, tiers=[1, 3, 5])
    decision = ExactSpsLocalDecision(
        num_requests=2,
        survival=torch.ones((2, 5)),
        native_expected_yield=12.0,
        compact_expected_yields=(10.0,),
    )

    with pytest.raises(ValueError, match="Unmeasured exact SPS cell"):
        planner.allocate_exact_sps_candidate(decision, graph_batch_size=4, verifier_budget=16)


def test_policy_reassociates_a_lagged_snapshot_by_request_row() -> None:
    planner = _planner()
    logits = torch.full((5, 5), -8.0)
    logits[3] = 8.0
    _publish_snapshot(planner, logits)

    lens = planner.decide_verify_lens(
        num_gen_requests=2,
        rows=[3, 0],
        reduce_across_ranks=False,
        budget_override=3,
    )

    assert lens is not None
    assert lens[0] > lens[1]


def test_policy_fails_closed_on_incomplete_snapshot_mapping() -> None:
    planner = _planner()
    _publish_snapshot(planner, torch.rand(8, 5))

    assert planner.decide_verify_lens(num_gen_requests=4, rows=[0, 1]) is None
    assert planner.stats["fallback_short_snapshot"] == 1


def test_selected_full_budget_uses_the_native_static_path() -> None:
    batch_size = 4
    planner = DSparkVerifyPlanner(
        cfg=_config(),
        cost_table=SpsCostTable(
            token_counts=(0, batch_size * 6),
            step_time_ms=(10.0, 10.01),
            minimum_predicted_gain=0.01,
        ),
        tiers=[1, 3, 5],
    )
    planner._gather_rows = lambda **_: torch.full((batch_size, 5), 8.0)

    assert (
        planner.decide_verify_lens(
            num_gen_requests=batch_size,
            reduce_across_ranks=False,
        )
        is None
    )
    assert planner.stats["fallback_full_k"] == 1


def test_device_window_capacity_uses_the_older_snapshot() -> None:
    planner = _planner(device_windows=True)
    planner._prev_buffer = torch.full((2, 5), -8.0)
    planner._prev_event = None
    planner._prev_valid = True
    planner._host_buffer = torch.full((2, 5), 8.0)
    planner._copy_event = None
    planner._snapshot_valid = True

    expected = planner.decide_verify_budget(num_gen_requests=2)
    planner._prev_buffer = planner._host_buffer
    different = planner.decide_verify_budget(num_gen_requests=2)

    assert expected is not None
    assert different is None or different[0] > expected[0]


def test_policy_records_the_cost_used_for_a_budget_decision() -> None:
    table = SpsCostTable(
        token_counts=(0, 16, 32, 48),
        step_time_ms=(5.0, 6.0, 12.0, 20.0),
        fixed_overhead_ms=1.0,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table, tiers=[1, 3, 5])
    _publish_snapshot(planner, torch.zeros((8, 5)))

    lens = planner.decide_verify_lens(num_gen_requests=8, reduce_across_ranks=False)

    assert lens is not None
    budget = sum(lens) - 8 * planner.cfg.min_verify_len
    submitted_tokens = 8 * (planner.cfg.min_verify_len + 1) + budget
    expected = float(table.step_times(np.asarray([submitted_tokens]), 8)[0])
    assert planner.last_predicted_step_ms == pytest.approx(expected)
    assert planner.stats["predicted_steps"] == 1
    assert planner.stats["predicted_ms_sum"] == pytest.approx(expected)
