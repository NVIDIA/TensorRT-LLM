# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two snapshot instruments: stamp-lag histogram and neutral-row count.

They exist because the planner's inputs have never been measured: the argmax
eats a slot-indexed snapshot staged one step earlier, and nothing recorded how
old each gathered row was (relay staleness) or how many rows were still the
+30 neutral fill (unknowns entering as certainties). The live anomaly they
instrument: 445/476 full-batch steps chose the full block while calibrated
pooled survivals favour rung-2 by 36% -- so the per-step snapshot content must
differ from the pooled shards, and these counters say in which way.

Both are measurement-only. A failure inside them must never kill a step.
"""

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_planner import SpsCostTable
from tensorrt_llm._torch.speculative.dspark_schedule import DSparkScheduleConfig
from tensorrt_llm._torch.speculative.dspark_verify import DSparkVerifyPlanner


def _planner():
    table = SpsCostTable(token_counts=(0, 512, 768, 1536),
                         step_time_ms=(5.0, 68.4, 80.2, 150.5),
                         fixed_overhead_ms=1.0)
    cfg = DSparkScheduleConfig(block_size=5, min_verify_len=1)
    return DSparkVerifyPlanner(cfg=cfg, cost_table=table, tiers=[1, 2, 5])


def test_neutral_rows_are_counted_and_real_rows_are_not():
    planner = _planner()
    selected = torch.zeros((8, 5))
    selected[3] = 30.0  # the neutral fill: sigmoid ~ 1.0 at any temperature
    planner._note_snapshot_stats(selected, rows=None)
    assert planner.stats["snap_rows"] == 8
    assert planner.stats["snap_neutral_rows"] == 1


def test_lag_histogram_reads_staged_seq_minus_stamp():
    planner = _planner()
    planner._host_stamps = torch.tensor([7, 6, 5, 0], dtype=torch.int32)
    planner._staged_seq = 7
    selected = torch.zeros((4, 5))
    planner._note_snapshot_stats(selected, rows=[0, 1, 2, 3])
    hist = planner.stats["stamp_lag_hist"]
    # Stamp 0 is "never drafted": lag 7 in truth, and it must land visibly at
    # the histogram's far end rather than vanish -- these are exactly the rows
    # the neutral counter should agree about.
    assert hist == {0: 1, 1: 1, 2: 1, 7: 1}


def test_large_and_negative_lags_clamp_instead_of_exploding_the_dict():
    planner = _planner()
    planner._host_stamps = torch.tensor([0, 500], dtype=torch.int32)
    planner._staged_seq = 400
    planner._note_snapshot_stats(torch.zeros((2, 5)), rows=[0, 1])
    assert planner.stats["stamp_lag_hist"] == {8: 1, -2: 1}


def test_counts_accumulate_across_steps():
    planner = _planner()
    planner._host_stamps = torch.tensor([3, 3], dtype=torch.int32)
    planner._staged_seq = 4
    for _ in range(3):
        planner._note_snapshot_stats(torch.full((2, 5), 30.0), rows=[0, 1])
    assert planner.stats["snap_rows"] == 6
    assert planner.stats["snap_neutral_rows"] == 6
    assert planner.stats["stamp_lag_hist"] == {1: 6}


def test_instrument_failure_is_counted_not_raised():
    planner = _planner()
    planner._host_stamps = torch.tensor([1], dtype=torch.int32)
    planner._staged_seq = 1
    # Rows that cannot index the stamp buffer must clamp, and anything worse
    # must be swallowed into the error counter: instruments never kill steps.
    planner._note_snapshot_stats(torch.zeros((1, 5)), rows=[10_000])
    assert planner.stats.get("snap_stats_errors", 0) == 0
    planner._note_snapshot_stats(object(), rows=None)  # type: ignore[arg-type]
    assert planner.stats["snap_stats_errors"] == 1


def test_decide_path_feeds_the_instruments():
    """The counters must fill from the real decide path, not only direct calls."""
    planner = _planner()
    bs = 8
    rng = np.random.default_rng(3)
    survival = np.sort(rng.random((bs, 5)), axis=1)[:, ::-1].copy()
    planner._gather_rows = lambda **_: torch.tensor(survival,
                                                    dtype=torch.float32)
    lens = planner.decide_verify_lens(num_gen_requests=bs,
                                      reduce_across_ranks=False)
    assert lens is not None
    assert planner.stats["snap_rows"] == bs
    assert planner.stats["snap_neutral_rows"] == 0
    # The local-decision instruments must fill on the same path: one rung per
    # decision, one window per request, and the two must agree -- the windows
    # are the top-k realisation of exactly that rung's budget.
    rung_hist = planner.stats["local_rung_hist"]
    assert sum(rung_hist.values()) == 1
    (rung,) = rung_hist
    assert rung in (1, 2, 5)
    len_hist = planner.stats["local_len_hist"]
    assert sum(len_hist.values()) == bs
    budget = bs * (rung - planner.cfg.min_verify_len)
    extra = sum((k - planner.cfg.min_verify_len) * v for k, v in len_hist.items())
    assert extra <= budget
