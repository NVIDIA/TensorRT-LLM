# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A campaign over several perf configs: a broken constraint, and the goal."""

from __future__ import annotations


def test_the_perf_block_names_no_stock_reference_or_repeats():
    from agent_flow.workflows.trtllm_modeling_v2.tune.schema import validate_perf_block

    perf = {"reference": "stock", "noise_repeats": 2, "goal": {"ratio_to_reference": 1.1}}
    errors = validate_perf_block({"perf": perf})
    assert "perf.reference: removed: no stock reference is measured" in errors[0]
    assert "perf.noise_repeats: removed: the baseline is measured once" in errors[1]
    assert "ratio_to_baseline and/or pct_of_sol" in errors[2]
    assert validate_perf_block({"perf": {"goal": {"ratio_to_baseline": 1.1}}}) == []
    campaign = {"checkpoint_keeps": 0, "micro_noise": -1, "logits": {"min_top1": 2, "k": 1}}
    assert validate_perf_block({"perf": {"campaign": campaign}}) == [
        "perf.campaign.micro_noise: a fraction >= 0 (0.05 is 5%)",
        "perf.campaign.checkpoint_keeps: a positive number of keeps",
        "perf.campaign.logits.k: unknown key",
        "perf.campaign.logits.min_top1: a number in [0, 1]",
    ]
