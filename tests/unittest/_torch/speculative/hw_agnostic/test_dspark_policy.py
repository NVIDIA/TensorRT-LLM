# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-independent tests for rank-local DSpark confidence planning."""

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_planner import ExactSpsCostRow, ExactSpsCostTable
from tensorrt_llm._torch.speculative.dspark_schedule import DSparkScheduleConfig
from tensorrt_llm._torch.speculative.dspark_verify import (
    DSparkVerifyPlanner,
    ExactSpsLocalDecision,
    _exact_cell_geometry,
)


def _config() -> DSparkScheduleConfig:
    return DSparkScheduleConfig(block_size=5, min_verify_len=1)


def _publish_snapshot(planner: DSparkVerifyPlanner, logits: torch.Tensor) -> None:
    planner._host_buffer = logits
    planner._copy_event = None
    planner._snapshot_valid = True


def test_runtime_cost_table_is_authoritative() -> None:
    planner = DSparkVerifyPlanner(cfg=_config())
    authoritative = ExactSpsCostTable(
        tables={4: ExactSpsCostRow(token_counts=(0, 14), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )

    planner.install_runtime_cost_table(authoritative)
    planner.install_runtime_cost_table(authoritative)
    assert planner.cost_table is authoritative

    different = ExactSpsCostTable(
        tables={4: ExactSpsCostRow(token_counts=(0, 14), step_time_ms=(8.0, 7.1))},
        max_draft_len=5,
    )
    with pytest.raises(RuntimeError, match="different SPS cost object"):
        planner.install_runtime_cost_table(different)


def test_runtime_cost_table_must_match_the_draft_block() -> None:
    planner = DSparkVerifyPlanner(cfg=_config())
    incompatible = ExactSpsCostTable(
        tables={4: ExactSpsCostRow(token_counts=(0, 12), step_time_ms=(8.0, 7.0))},
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
            4: ExactSpsCostRow(
                token_counts=(0, 12, 14, 24),
                step_time_ms=(8.0, 6.0, 6.5, 8.1),
            )
        },
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table)

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
        tables={4: ExactSpsCostRow(token_counts=(0, 14), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table)
    logits = torch.full((5, 5), -8.0)
    logits[3] = 8.0
    _publish_snapshot(planner, logits)

    decision = planner.prepare_exact_sps_decision(num_gen_requests=2, rows=[3, 0])

    assert decision is not None
    assert decision.survival[0].sum() > decision.survival[1].sum()


def test_exact_allocator_spends_the_modeled_real_row_target() -> None:
    table = ExactSpsCostTable(
        tables={4: ExactSpsCostRow(token_counts=(0, 22), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table)
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


def test_device_window_allocator_retains_ranked_host_fallback() -> None:
    table = ExactSpsCostTable(
        tables={3: ExactSpsCostRow(token_counts=(0, 14), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table, device_windows=True)
    decision = ExactSpsLocalDecision(
        num_requests=3,
        # Deliberately favor the last two requests. The host allocation remains
        # a valid ranked fallback whenever the device prologue cannot run.
        survival=torch.tensor([[0.1] * 5, [0.2] * 5, [0.9] * 5], dtype=torch.float32),
        native_expected_yield=9.0,
        compact_expected_yields=(8.0,),
    )

    lens, budget, pad_tokens = planner.allocate_exact_sps_candidate(
        decision, graph_batch_size=3, verifier_budget=14
    )

    assert lens == [1, 5, 5]
    assert budget == 8
    assert pad_tokens == 0
    assert sum(value + 1 for value in lens) == 14


def test_unmeasured_exact_cell_is_rejected() -> None:
    table = ExactSpsCostTable(
        tables={4: ExactSpsCostRow(token_counts=(0, 14), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table)
    decision = ExactSpsLocalDecision(
        num_requests=2,
        survival=torch.ones((2, 5)),
        native_expected_yield=12.0,
        compact_expected_yields=(10.0,),
    )

    with pytest.raises(ValueError, match="Unmeasured exact SPS cell"):
        planner.allocate_exact_sps_candidate(decision, graph_batch_size=4, verifier_budget=16)


def test_policy_fails_closed_on_incomplete_snapshot_mapping() -> None:
    table = ExactSpsCostTable(
        tables={4: ExactSpsCostRow(token_counts=(0, 14), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table)
    _publish_snapshot(planner, torch.rand(8, 5))

    assert planner.prepare_exact_sps_decision(num_gen_requests=4, rows=[0, 1]) is None
    assert planner.stats["fallback_short_snapshot"] == 1


def test_device_window_capacity_uses_the_older_snapshot() -> None:
    table = ExactSpsCostTable(
        tables={4: ExactSpsCostRow(token_counts=(0, 14), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table, device_windows=True)
    planner._prev_buffer = torch.full((2, 5), -8.0)
    planner._prev_event = None
    planner._prev_valid = True
    planner._host_buffer = torch.full((2, 5), 8.0)
    planner._copy_event = None
    planner._snapshot_valid = True

    older = planner.prepare_exact_sps_decision(num_gen_requests=2)
    planner._prev_buffer = planner._host_buffer
    current = planner.prepare_exact_sps_decision(num_gen_requests=2)

    assert older is not None
    assert current is not None
    assert older.survival.sum() < current.survival.sum()


class _CopyEvent:
    """CPU event double; the tests check ownership and polling, not CUDA copies."""

    def __init__(self) -> None:
        self.ready = False

    def record(self) -> None:
        self.ready = False

    def query(self) -> bool:
        return self.ready


@pytest.mark.parametrize("device_windows", [False, True])
def test_confidence_staging_is_lazy_and_polls_before_reading(
    monkeypatch: pytest.MonkeyPatch, device_windows: bool
) -> None:
    events = []

    def new_event() -> _CopyEvent:
        event = _CopyEvent()
        events.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", new_event)
    table = ExactSpsCostTable(
        tables={4: ExactSpsCostRow(token_counts=(0, 14), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(cfg=_config(), cost_table=table, device_windows=device_windows)
    planner.stage_confidence(torch.empty((0, 5)))
    assert events == []
    assert planner._host_buffer is None

    logits = torch.tensor([[8.0] * 5, [-8.0] * 5])
    planner.stage_confidence(logits)
    assert planner.prepare_exact_sps_decision(num_gen_requests=2) is None
    if device_windows:
        # Rotate the first pending copy into the capacity slot. The newer
        # confidence is deliberately different and must not be substituted.
        planner.stage_confidence(torch.zeros_like(logits))
        assert planner.prepare_exact_sps_decision(num_gen_requests=2) is None
    events[0].ready = True

    decision = planner.prepare_exact_sps_decision(num_gen_requests=2, rows=[1, 0])

    assert decision is not None
    torch.testing.assert_close(decision.survival, torch.sigmoid(logits[[1, 0]]).cumprod(dim=1))
    assert planner._host_buffer.data_ptr() != logits.data_ptr()


def test_device_confidence_staging_reuses_two_completed_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "Event", _CopyEvent)
    planner = DSparkVerifyPlanner(cfg=_config(), device_windows=True)
    planner.stage_confidence(torch.full((2, 5), 1.0))
    first_buffer = planner._host_buffer
    first_event = planner._copy_event
    first_event.ready = True
    planner.stage_confidence(torch.full((2, 5), 2.0))
    second_buffer = planner._host_buffer
    planner._copy_event.ready = True
    planner.stage_confidence(torch.full((2, 5), 3.0))

    assert planner._host_buffer is first_buffer
    assert planner._copy_event is first_event
    assert planner._prev_buffer is second_buffer
    assert planner._ready_snapshot() is None
    torch.testing.assert_close(planner._ready_older_snapshot(), torch.full((2, 5), 2.0))
    torch.testing.assert_close(first_buffer, torch.full((2, 5), 3.0))
