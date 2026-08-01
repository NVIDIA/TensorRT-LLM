# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the DSpark SPS cost-table profiler (CPU only, no model load).

Everything here exercises the pure half of the profiler: turning ``get_stats``
rows into aligned samples, summarizing cells, separating ``alpha(bs)`` from
``theta(M)``, and assembling / refusing the emitted JSON. The sweep driver needs
a GPU and a checkpoint and is deliberately not covered.

The load-bearing case is :func:`test_round_trip_reproduces_measured_cells`: it
takes a synthetic machine with a known cost curve, runs it through the whole
fit-and-emit pipeline, and reloads the JSON exactly the way
``dspark.py::_build_verify_planner`` does. If the profiler and the loader ever
disagree about units, keys or the meaning of a token count, that test is where
it shows up.
"""

import json

import numpy as np
import pytest

from tensorrt_llm._torch.speculative.dspark_planner import SpsCostTable, total_verify_tokens
from tensorrt_llm._torch.speculative.dspark_sps_profiler import (
    CellStat,
    FlatCostTableError,
    InertCostTableError,
    InsufficientSamplesError,
    StepSample,
    SweepConfig,
    SweepGeometryError,
    aligned_steps_from_stats,
    build_cost_table_payload,
    check_table_is_informative,
    compress_to_risers,
    fit_additive_cost_model,
    load_cost_table,
    profitability_probe,
    running_max,
    summarize_cells,
)

#: The exact set of keys ``dspark.py::_build_verify_planner`` reads. Anything
#: else in the file is ignored by the loader, so the profiler must not rely on
#: it -- and must not omit any of these.
LOADER_KEYS = {
    "token_counts",
    "step_time_ms",
    "fixed_overhead_ms",
    "batch_sizes",
    "batch_overhead_ms",
}


def _stats_row(*, iteration, num_gen, rank=0, num_ctx=0, host_ms=10.0, gpu_ms=8.0):
    """One /metrics-shaped iteration record."""
    row = {
        "iter": iteration,
        "attentionDpRank": rank,
        "inflightBatchingStats": {
            "numContextRequests": num_ctx,
            "numGenRequests": num_gen,
        },
    }
    if host_ms is not None:
        row["hostStepTimeMS"] = host_ms
    if gpu_ms is not None:
        row["gpuForwardTimeMS"] = gpu_ms
    return row


# --------------------------------------------------------------------------
# aligned step extraction
# --------------------------------------------------------------------------


def test_aligned_steps_single_rank():
    rows = [_stats_row(iteration=i, num_gen=8, host_ms=10.0 + i) for i in range(3)]
    assert aligned_steps_from_stats(rows) == [(0, 8, 10.0), (1, 8, 11.0), (2, 8, 12.0)]


def test_aligned_steps_drops_prefill_iterations():
    """A step with context requests is a different shape, not a cheap decode."""
    rows = [
        _stats_row(iteration=0, num_gen=0, num_ctx=8, host_ms=90.0),
        _stats_row(iteration=1, num_gen=8, num_ctx=2, host_ms=40.0),
        _stats_row(iteration=2, num_gen=8, host_ms=10.0),
    ]
    assert aligned_steps_from_stats(rows) == [(2, 8, 10.0)]


def test_aligned_steps_requires_every_dp_rank():
    """A missing rank means the iteration was not observed batch-wide."""
    rows = [
        _stats_row(iteration=0, num_gen=8, rank=0),
        _stats_row(iteration=0, num_gen=8, rank=1),
        _stats_row(iteration=1, num_gen=8, rank=0),
    ]
    assert aligned_steps_from_stats(rows, expected_ranks=2) == [(0, 8, 10.0)]


def test_aligned_steps_requires_ranks_to_agree_on_batch_size():
    """Ranks at different batch sizes are not one (bs, M) point."""
    rows = [
        _stats_row(iteration=0, num_gen=8, rank=0),
        _stats_row(iteration=0, num_gen=7, rank=1),
    ]
    assert aligned_steps_from_stats(rows, expected_ranks=2) == []


def test_aligned_steps_takes_rank0_timing():
    """ADP fans out rank-local counters with rank-0's clock; do not average."""
    rows = [
        _stats_row(iteration=0, num_gen=8, rank=1, host_ms=99.0),
        _stats_row(iteration=0, num_gen=8, rank=0, host_ms=10.0),
    ]
    assert aligned_steps_from_stats(rows, expected_ranks=2) == [(0, 8, 10.0)]


def test_aligned_steps_skips_missing_timing_field():
    """gpuForwardTimeMS is read without a sync and is absent on some steps."""
    rows = [
        _stats_row(iteration=0, num_gen=8, gpu_ms=None),
        _stats_row(iteration=1, num_gen=8, gpu_ms=7.5),
    ]
    assert aligned_steps_from_stats(rows, timing_key="gpuForwardTimeMS") == [(1, 8, 7.5)]


# --------------------------------------------------------------------------
# warmup + median
# --------------------------------------------------------------------------


def _samples(batch_size, verify_len, times):
    return [
        StepSample(batch_size=batch_size, verify_len=verify_len, step_time_ms=t, iteration=i)
        for i, t in enumerate(times)
    ]


def test_summarize_discards_warmup():
    warm = [500.0] * 4
    steady = [10.0] * 8
    cells = summarize_cells(_samples(8, 3, warm + steady), warmup_steps=4, min_samples=8)
    assert len(cells) == 1
    assert cells[0].step_time_ms == pytest.approx(10.0)
    assert cells[0].num_samples == 8


def test_summarize_reports_median_not_mean():
    """The right tail is real and must not move the point."""
    times = [10.0, 10.0, 10.0, 10.0, 10.0, 1000.0]
    cells = summarize_cells(_samples(8, 3, times), warmup_steps=0, min_samples=1)
    assert cells[0].step_time_ms == pytest.approx(10.0)
    assert cells[0].step_time_ms != pytest.approx(float(np.mean(times)))
    # ...but the tail is still reported, so a pathological cell is visible.
    assert cells[0].p90_ms > cells[0].step_time_ms


def test_summarize_refuses_short_cell():
    """A silently dropped cell can disconnect the grid, not just widen error bars."""
    with pytest.raises(InsufficientSamplesError, match="steady samples"):
        summarize_cells(_samples(8, 3, [10.0] * 5), warmup_steps=4, min_samples=8)


def test_summarize_orders_by_iteration_before_cutting_warmup():
    """Stats arrive per queue, not necessarily in iteration order."""
    out_of_order = [
        StepSample(batch_size=8, verify_len=1, step_time_ms=t, iteration=i)
        for i, t in sorted(enumerate([500.0, 500.0, 10.0, 10.0]), key=lambda p: -p[0])
    ]
    cells = summarize_cells(out_of_order, warmup_steps=2, min_samples=2)
    assert cells[0].step_time_ms == pytest.approx(10.0)


def test_step_sample_json_round_trip():
    sample = StepSample(batch_size=16, verify_len=3, step_time_ms=12.5, iteration=7)
    assert StepSample.from_json(json.loads(json.dumps(sample.to_json()))) == sample
    assert sample.total_verify_tokens == total_verify_tokens(16, 3) == 64


# --------------------------------------------------------------------------
# additive fit
# --------------------------------------------------------------------------


def _synthetic_cells(batch_sizes, verify_lens, *, alpha, theta, noise=None):
    """Cells generated by an exactly additive machine ``alpha(bs) + theta(M)``."""
    cells = []
    for batch_size in batch_sizes:
        for verify_len in verify_lens:
            tokens = total_verify_tokens(batch_size, verify_len)
            value = alpha(batch_size) + theta(tokens)
            if noise is not None:
                value += noise(batch_size, verify_len)
            cells.append(
                CellStat(
                    batch_size=batch_size,
                    verify_len=verify_len,
                    step_time_ms=value,
                    num_samples=32,
                    p10_ms=value,
                    p90_ms=value,
                )
            )
    return cells


def _staircase(edges_and_costs):
    def theta(tokens):
        value = edges_and_costs[0][1]
        for edge, cost in edges_and_costs:
            if tokens >= edge:
                value = cost
        return value

    return theta


BATCH_SIZES = [8, 16, 32, 64]
VERIFY_LENS = [1, 2, 3, 4, 5]
ALPHA = {8: 20.0, 16: 22.0, 32: 26.0, 64: 34.0}
THETA = _staircase([(0, 1.0), (64, 2.0), (128, 4.0), (256, 8.0)])


def test_fit_recovers_relative_structure_exactly():
    """The fit is only identified up to one shared constant -- pin that, then
    every cell must be reproduced to the last bit."""
    cells = _synthetic_cells(BATCH_SIZES, VERIFY_LENS, alpha=ALPHA.get, theta=THETA)
    fit = fit_additive_cost_model(cells)

    for cell in cells:
        predicted = fit.intercept_ms[cell.batch_size] + fit.theta_ms[cell.total_verify_tokens]
        assert predicted == pytest.approx(cell.step_time_ms, abs=1e-6)
    assert fit.max_rel_residual == pytest.approx(0.0, abs=1e-9)

    # The batch axis keeps its measured *differences*, which is what alpha means.
    for smaller, larger in zip(BATCH_SIZES, BATCH_SIZES[1:]):
        measured = ALPHA[larger] - ALPHA[smaller]
        fitted = fit.intercept_ms[larger] - fit.intercept_ms[smaller]
        assert fitted == pytest.approx(measured, abs=1e-6)


def test_fit_surfaces_a_bad_cell_rather_than_hiding_it():
    """A cell the additive model cannot explain is evidence, not noise.

    Dropping it would produce a table that fits beautifully and predicts badly,
    and would delete the one signal that says ``theta`` is not a function of
    ``M`` alone.
    """

    def noise(batch_size, verify_len):
        return 500.0 if (batch_size, verify_len) == (32, 3) else 0.0

    cells = _synthetic_cells(BATCH_SIZES, VERIFY_LENS, alpha=ALPHA.get, theta=THETA, noise=noise)
    fit = fit_additive_cost_model(cells)

    assert fit.max_abs_residual_ms > 100.0
    assert fit.max_rel_residual > 1.0
    report = " ".join(fit.warnings)
    assert "additive model misses" in report
    # The least-squares solve spreads the bad cell over everything sharing its
    # batch size or token count, so the report names a neighbourhood -- the bad
    # cell is in it, but so are innocent ones, which is why it is worded that way.
    assert "bs=32, L=3" in report


def test_fit_stays_quiet_when_the_model_holds():
    cells = _synthetic_cells(BATCH_SIZES, VERIFY_LENS, alpha=ALPHA.get, theta=THETA)
    assert fit_additive_cost_model(cells).warnings == ()


def test_fit_clamps_a_dip_upward():
    """A non-monotone theta would offer the planner a cheaper *longer* verify."""
    dipped = _staircase([(0, 1.0), (64, 5.0), (128, 3.0), (256, 8.0)])
    cells = _synthetic_cells(BATCH_SIZES, VERIFY_LENS, alpha=ALPHA.get, theta=dipped)
    fit = fit_additive_cost_model(cells)

    tokens = sorted(fit.theta_ms)
    values = [fit.theta_ms[m] for m in tokens]
    assert values == running_max(values)
    assert any("monoton" in w for w in fit.warnings)


def test_fit_keeps_theta_positive():
    """SpsCostTable rejects a non-positive step time, so the shift is bounded."""
    cells = _synthetic_cells(BATCH_SIZES, VERIFY_LENS, alpha=ALPHA.get, theta=THETA)
    fit = fit_additive_cost_model(cells)
    assert min(fit.theta_ms.values()) > 0.0


def test_fit_refuses_a_disconnected_sweep():
    """Two islands each carry their own unknown offset; splicing them lies."""
    cells = _synthetic_cells([8], [1, 2], alpha=ALPHA.get, theta=THETA)
    cells += _synthetic_cells([100], [1, 2], alpha=lambda _: 40.0, theta=THETA)
    with pytest.raises(SweepGeometryError, match="disconnected"):
        fit_additive_cost_model(cells)


def test_fit_refuses_a_single_length_sweep():
    """One verify length per batch size confounds alpha and theta completely."""
    cells = _synthetic_cells(BATCH_SIZES, [5], alpha=ALPHA.get, theta=THETA)
    with pytest.raises(SweepGeometryError):
        fit_additive_cost_model(cells)


# --------------------------------------------------------------------------
# staircase shaping
# --------------------------------------------------------------------------


def test_running_max_is_monotone():
    assert running_max([1.0, 3.0, 2.0, 2.5, 9.0]) == [1.0, 3.0, 3.0, 3.0, 9.0]


def test_compress_keeps_only_real_risers():
    tokens = [16, 32, 48, 64, 80]
    times = [1.0, 1.002, 1.004, 2.0, 2.001]
    kept_tokens, kept_times = compress_to_risers(
        tokens, times, min_riser_ms=0.02, max_breakpoints=8
    )
    assert kept_tokens == [16, 64]
    assert kept_times == [1.0, 2.0]


def test_compress_drops_the_shallowest_risers_first():
    tokens = [16, 32, 64, 128]
    times = [1.0, 1.1, 4.0, 9.0]
    kept_tokens, _ = compress_to_risers(tokens, times, min_riser_ms=0.0, max_breakpoints=3)
    assert kept_tokens == [16, 64, 128]


def test_compress_never_drops_the_base_shelf():
    """Index 0 is what every below-range query clamps onto."""
    tokens = [16, 32, 64]
    times = [1.0, 100.0, 101.0]
    kept_tokens, _ = compress_to_risers(tokens, times, min_riser_ms=0.0, max_breakpoints=2)
    assert kept_tokens[0] == 16


# --------------------------------------------------------------------------
# payload assembly and JSON round-trip through SpsCostTable
# --------------------------------------------------------------------------


def _payload():
    cells = _synthetic_cells(BATCH_SIZES, VERIFY_LENS, alpha=ALPHA.get, theta=THETA)
    return build_cost_table_payload(cells, riser_tolerance=0.01, max_breakpoints=8), cells


def test_payload_has_exactly_the_loader_keys():
    payload, _ = _payload()
    assert LOADER_KEYS.issubset(payload)
    # Everything else must live under _meta, where the loader ignores it.
    assert set(payload) - LOADER_KEYS == {"_meta"}


def test_payload_survives_a_json_file_round_trip():
    payload, _ = _payload()
    reloaded = json.loads(json.dumps(payload))
    table = load_cost_table(reloaded)
    assert isinstance(table, SpsCostTable)
    assert not table.is_flat
    # The loader builds SpsCostTable directly, so the profiler's own validation
    # must be the real constructor rather than a copy of its rules.
    assert list(table.token_counts) == payload["token_counts"]
    assert len(table.batch_sizes) == len(table.batch_overhead_ms)


def test_round_trip_reproduces_measured_cells():
    """The whole point: what the planner reads back must be what was measured.

    The tolerance is one shelf width, because the emitted curve is a floor
    staircase by construction -- a cell between two risers is deliberately
    priced at the lower shelf.
    """
    payload, cells = _payload()
    table = load_cost_table(json.loads(json.dumps(payload)))
    for cell in cells:
        predicted = table.step_time(cell.total_verify_tokens, cell.batch_size)
        assert predicted == pytest.approx(cell.step_time_ms, abs=1e-4)


def test_emitted_terms_are_not_double_counted():
    """step_time_ms is theta only; the floor lives in the two overhead fields."""
    payload, cells = _payload()
    table = load_cost_table(payload)
    smallest = min(BATCH_SIZES)
    floor = payload["fixed_overhead_ms"] + table.batch_overhead(smallest)
    # The floor is the M-independent part, i.e. the intercept of the measured
    # row at the smallest batch size -- not the whole step time.
    row = sorted((c for c in cells if c.batch_size == smallest), key=lambda c: c.verify_len)
    assert floor < row[0].step_time_ms
    assert floor == pytest.approx(ALPHA[smallest], abs=0.5)


def test_batch_overhead_grows_with_batch_size():
    payload, _ = _payload()
    overheads = payload["batch_overhead_ms"]
    assert overheads == sorted(overheads)
    assert payload["batch_sizes"] == sorted(BATCH_SIZES)
    assert payload["fixed_overhead_ms"] >= 0.0


def test_meta_carries_the_tier_advice_and_the_cells():
    payload, cells = _payload()
    assert payload["_meta"]["encoding"] == "decomposed"
    assert len(payload["_meta"]["cells"]) == len(cells)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_refuses_a_flat_curve():
    """A machine whose step cost ignores M must not produce a table at all."""
    cells = _synthetic_cells(BATCH_SIZES, VERIFY_LENS, alpha=ALPHA.get, theta=lambda _: 3.0)
    payload = build_cost_table_payload(cells)
    assert load_cost_table(payload).is_flat
    with pytest.raises(FlatCostTableError, match="is_flat"):
        check_table_is_informative(payload, batch_sizes=BATCH_SIZES, tiers=[1, 3, 5])


def test_refuses_a_curve_that_only_moves_by_noise():
    """is_flat is exact float equality, so it cannot catch this one."""
    cells = _synthetic_cells(
        BATCH_SIZES,
        VERIFY_LENS,
        alpha=ALPHA.get,
        theta=lambda tokens: 3.0 + 1e-4 * tokens,
    )
    payload = build_cost_table_payload(cells, riser_tolerance=0.0, max_breakpoints=8)
    assert not load_cost_table(payload).is_flat
    with pytest.raises(FlatCostTableError, match="noise floor"):
        check_table_is_informative(
            payload, batch_sizes=BATCH_SIZES, tiers=[1, 3, 5], min_step_time_spread=0.02
        )


# A real but shallow riser: at bs=32 tier 3 (M=128) sits below it and tier 5
# (M=192) above, so the curve is genuinely non-flat and clears the noise floor
# (2.5%), yet the step only gets 2.5% more expensive while the acceptance yield
# given up between those tiers is ~36% at p=0.9. The argmax therefore never
# leaves max_tier -- non-flat and inert at the same time.
INERT_PAYLOAD = {
    "token_counts": [0, 160],
    "step_time_ms": [2.0, 2.5],
    "fixed_overhead_ms": 18.0,
    "batch_sizes": [32],
    "batch_overhead_ms": [0.0],
}


def test_refuses_an_inert_table():
    """Non-flat, but no tier pair is worth trading yield for.

    This is the failure mode that does *not* trip ``fallback_flat_cost``, so
    nothing downstream would report it.
    """
    with pytest.raises(InertCostTableError, match="inert"):
        check_table_is_informative(
            INERT_PAYLOAD,
            batch_sizes=[32],
            tiers=[1, 3, 5],
            acceptance_rates=[0.9, 0.95],
        )


def test_allow_inert_escape_hatch():
    diagnostics = check_table_is_informative(
        INERT_PAYLOAD,
        batch_sizes=[32],
        tiers=[1, 3, 5],
        acceptance_rates=[0.9, 0.95],
        allow_inert=True,
    )
    assert diagnostics["trims_somewhere"] is False


def test_refuses_a_single_entry_ladder():
    payload, _ = _payload()
    with pytest.raises(InertCostTableError, match="single entry"):
        check_table_is_informative(payload, batch_sizes=BATCH_SIZES, tiers=[5])


def test_informative_table_passes_and_reports_where_trimming_wins():
    payload, _ = _payload()
    diagnostics = check_table_is_informative(
        payload,
        batch_sizes=BATCH_SIZES,
        tiers=[1, 3, 5],
        acceptance_rates=[0.3, 0.6, 0.9],
    )
    assert diagnostics["trims_somewhere"] is True
    assert diagnostics["best_step_time_spread"] > 0.02


# --------------------------------------------------------------------------
# sweep geometry guards (pure; the sweep itself needs a GPU)
# --------------------------------------------------------------------------


def _sweep(**overrides) -> SweepConfig:
    kwargs = dict(
        model="/models/x",
        speculative_model="/models/x",
        batch_sizes=[8, 16],
        verify_lens=[1, 3, 5],
        input_len=1024,
        warmup_steps=8,
        measure_steps=32,
        max_seq_len=4096,
        max_num_tokens=8192,
    )
    kwargs.update(overrides)
    return SweepConfig(**kwargs)


def test_token_budget_is_the_step_count_when_acceptance_is_pinned():
    """Pinned, a step commits exactly one token, so the two are the same number."""
    config = _sweep()
    assert config.max_tokens_for(5) == config.warmup_steps + config.measure_steps + 16


def test_token_budget_is_scaled_when_acceptance_is_not_pinned():
    """Unpinned, a step can commit verify_len + 1 tokens, so the budget must cover it."""
    pinned = _sweep()
    loose = _sweep(pin_acceptance=0.0)
    assert loose.max_tokens_for(5) == pinned.max_tokens_for(5) * 6


def test_validate_rejects_a_sequence_that_would_be_evicted():
    """An evicted request drains the batch mid-cell and shrinks the shape."""
    with pytest.raises(SweepGeometryError, match="max-seq-len"):
        _sweep(input_len=4064).validate()


def test_validate_rejects_a_decode_step_wider_than_max_num_tokens():
    """A split decode step never has the shape its cell is filed under."""
    with pytest.raises(SweepGeometryError, match="widest decode step"):
        _sweep(batch_sizes=[8, 4096], max_num_tokens=1024).validate()


def test_validate_accepts_a_sane_sweep():
    _sweep().validate()


def test_dp_size_only_counts_ranks_under_attention_dp():
    """Non-ADP publishes one stats row per iteration however wide TP is."""
    assert _sweep(tp_size=8).dp_size == 1
    assert _sweep(tp_size=8, enable_attention_dp=True).dp_size == 8


# --------------------------------------------------------------------------
# profitability probe (runs the real planner argmax)
# --------------------------------------------------------------------------


def test_probe_trims_at_low_acceptance_and_not_at_high():
    """Trimming is a trade: it only wins when the drafts are unlikely to survive."""
    payload = {
        # A riser between bs*(3+1)=128 and bs*(5+1)=192 at bs=32, so tiers 3 and
        # 5 sit on different shelves and the argmax has something to choose:
        # T(3) = 3 ms, T(5) = 4 ms, a 33% step for a yield gain that is worth it
        # only when the drafts survive.
        "token_counts": [0, 160],
        "step_time_ms": [2.0, 3.0],
        "fixed_overhead_ms": 0.0,
        "batch_sizes": [32],
        "batch_overhead_ms": [1.0],
    }
    probes = profitability_probe(
        payload, batch_size=32, tiers=[1, 3, 5], acceptance_rates=[0.05, 0.99]
    )
    by_rate = {p["acceptance_rate"]: p for p in probes}
    assert by_rate[0.05]["chosen_verify_len"] < 5
    assert by_rate[0.05]["trims"] is True
    assert by_rate[0.99]["chosen_verify_len"] == 5


def test_probe_uses_the_bonus_token_convention():
    """A tier's cost must be looked up at bs*(L+1), never bs*L.

    The two differ by a whole batch, which is easily a shelf's width; this pins
    the profiler to the same convention as ``total_verify_tokens``.
    """
    payload = {
        # Priced so that tier 5 (M = 8*6 = 48) is expensive and tier 3
        # (M = 8*4 = 32) is cheap. Under the wrong bs*L convention tier 5 would
        # be M = 40 and land on the cheap shelf instead.
        "token_counts": [0, 44],
        "step_time_ms": [1.0, 50.0],
        "fixed_overhead_ms": 0.0,
        "batch_sizes": [8],
        "batch_overhead_ms": [0.0],
    }
    table = load_cost_table(payload)
    assert table.step_time(total_verify_tokens(8, 5), 8) == pytest.approx(50.0)
    assert table.step_time(total_verify_tokens(8, 3), 8) == pytest.approx(1.0)
    probes = profitability_probe(payload, batch_size=8, tiers=[1, 3, 5], acceptance_rates=[0.9])
    assert probes[0]["chosen_verify_len"] == 3
