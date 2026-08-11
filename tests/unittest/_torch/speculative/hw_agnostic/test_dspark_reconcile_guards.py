# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two reconciliation guards: engine fingerprint and price prediction.

Both exist because this pipeline's characteristic failure is a PREDICTION
recorded as a MEASUREMENT with nothing positioned to contradict it: a cost
table measured on the wrong MoE kernel almost planned MegaMoE steps, the same
nominal cell measured 23% apart across two engine configs, and a planner
argmax'd against a table whose labels described shapes that never ran. The
fingerprint refuses the wrong table at load; the recorded prediction gives
every run a number that reality (hostStepTimeMS) can contradict.
"""

import numpy as np
import pytest

from tensorrt_llm._torch.speculative.dspark_planner import (
    SpsCostTable, check_table_fingerprint)
from tensorrt_llm._torch.speculative.dspark_schedule import DSparkScheduleConfig
from tensorrt_llm._torch.speculative.dspark_verify import DSparkVerifyPlanner

LIVE = {"tp": 8, "ep": 8, "attention_dp": True, "block": 5,
        "max_batch_size": 256}


def _payload(engine):
    return {"token_counts": [512, 1536], "step_time_ms": [60.0, 150.0],
            "_meta": {"engine": engine} if engine is not None else {}}


def test_mismatched_engine_is_refused():
    """The 23%-apart case: same cell, different max_batch_size."""
    wrong = dict(LIVE, max_batch_size=64)
    with pytest.raises(ValueError, match="different engine configuration"):
        check_table_fingerprint(payload=_payload(wrong), live=dict(LIVE))


def test_fingerprints_that_must_load():
    """Every non-contradicted fingerprint loads; only a contradicted fact refuses.

    Matching facts load silently; string facts must not fail on spelling case
    (MEGAMOE vs MegaMoe); facts the consumer cannot see (image hash, geometry)
    are informational and logged, not refused -- refusing on unverifiable keys
    would make every fingerprint addition a breaking change; and an old table
    with no fingerprint warns rather than breaking existing deployments.
    """
    check_table_fingerprint(payload=_payload(dict(LIVE)), live=dict(LIVE))
    check_table_fingerprint(
        payload=_payload(dict(LIVE, moe_backend="megamoe_cutedsl")),
        live=dict(LIVE, moe_backend="MEGAMOE_CUTEDSL"))
    check_table_fingerprint(
        payload=_payload(dict(LIVE, image="faf2c60935",
                              geometry="constant_block")),
        live=dict(LIVE))
    check_table_fingerprint(payload=_payload(None), live=dict(LIVE))


def test_planner_records_the_price_it_paid():
    """Every budget decision must leave behind the step time it assumed.

    Without this the run has no number reality can contradict: the argmax
    consults the table and the table's opinion evaporates. With it, the
    [final] planner block carries predicted_ms_sum / predicted_steps, and a
    systematic gap against measured hostStepTimeMS is the one signal that
    catches a wrong table, a wrong lookup, and an engine mismatch alike.
    """
    table = SpsCostTable(token_counts=(0, 512, 768, 1536),
                         step_time_ms=(5.0, 68.4, 80.2, 150.5),
                         fixed_overhead_ms=1.0)
    cfg = DSparkScheduleConfig(block_size=5, min_verify_len=1)
    planner = DSparkVerifyPlanner(cfg=cfg, cost_table=table, tiers=[1, 2, 5])

    bs = 8
    rng = np.random.default_rng(11)
    survival = np.sort(rng.random((bs, 5)), axis=1)[:, ::-1].copy()

    import torch
    # Only the snapshot is faked; calibration stays the constructor's default
    # sigmoid (setting it to None post-construction would bypass the
    # `apply_calibration or torch.sigmoid` fallback and crash the decide path).
    planner._gather_rows = lambda **_: torch.tensor(survival,
                                                    dtype=torch.float32)

    lens = planner.decide_verify_lens(num_gen_requests=bs,
                                      reduce_across_ranks=False)
    assert lens is not None

    steps = planner.stats.get("predicted_steps", 0)
    total = planner.stats.get("predicted_ms_sum", 0.0)
    assert steps == 1 and total > 0.0

    # The recorded price must be the table's own answer for the tokens the
    # decision implies -- anything else and the reconciliation compares two
    # unrelated numbers.
    budget = sum(lens) - bs * cfg.min_verify_len
    tokens = bs * (cfg.min_verify_len + 1) + budget
    expected = float(table.step_times(np.asarray([tokens]), bs)[0])
    assert planner.last_predicted_step_ms == pytest.approx(expected)
    assert total == pytest.approx(expected)
