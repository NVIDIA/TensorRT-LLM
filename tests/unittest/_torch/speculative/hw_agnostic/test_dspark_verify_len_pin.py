# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runtime verify-length pin: queue, agree, adopt.

A cost-table sweep has to hold every request at one verify length while it
measures a cell, or the cell's label describes a shape that never ran (this
branch shipped a sweep where eight of twelve cells were mislabelled and the
fit dutifully concluded the curve was flat). The environment variable that
does this today is read once at planner construction, so walking the ladder
means rebuilding the engine per length -- and on a pre-spawned world it
cannot reach the ranks that are already running.

The runtime pin fixes both, but it must never let one rank pin while its
peers do not: the ranks would derive different token totals, and the
attention-DP shape gate would drop the whole group out of graph replay for as
long as they disagreed. So the pin is queued locally, carried by the step's
existing allgather, and adopted from that agreed payload -- which is what
these tests pin down.
"""

import pytest

from tensorrt_llm._torch.speculative.dspark_planner import SpsCostTable
from tensorrt_llm._torch.speculative.dspark_schedule import DSparkScheduleConfig
from tensorrt_llm._torch.speculative.dspark_verify import DSparkVerifyPlanner


def _planner(tiers=(1, 2, 5)):
    table = SpsCostTable(token_counts=(0, 512, 768, 1536),
                         step_time_ms=(5.0, 68.4, 80.2, 150.5),
                         fixed_overhead_ms=1.0)
    cfg = DSparkScheduleConfig(block_size=5, min_verify_len=1)
    return DSparkVerifyPlanner(cfg=cfg, cost_table=table, tiers=list(tiers))


def test_adopting_the_agreed_value_applies_it():
    """Queuing must not change this rank's behaviour before the group agrees;
    adopting the agreed value applies it and consumes the pending request."""
    planner = _planner()
    assert planner.request_verify_len_pin(2) == 2
    # Queued, not yet in force.
    assert planner._forced_verify_len is None
    assert planner.pending_verify_len_pin() == 2
    planner.adopt_verify_len_pin(2)
    assert planner._forced_verify_len == 2
    # Consumed: the next step must not re-broadcast a pin nobody asked for.
    assert planner.pending_verify_len_pin() == -1


def test_a_rank_that_queued_nothing_still_adopts_the_group_value():
    """The endpoint lands on ONE rank; the others learn it from the payload."""
    peer = _planner()
    assert peer.pending_verify_len_pin() == -1
    peer.adopt_verify_len_pin(5)
    assert peer._forced_verify_len == 5


def test_zero_clears_the_pin():
    """Wire protocol: -1 means nobody asked (leave the pin alone); clearing
    needs its own value, 0, because None cannot travel in an int payload."""
    planner = _planner()
    planner.adopt_verify_len_pin(2)
    planner.adopt_verify_len_pin(-1)
    assert planner._forced_verify_len == 2
    assert planner.request_verify_len_pin(None) is None
    assert planner.pending_verify_len_pin() == 0
    planner.adopt_verify_len_pin(0)
    assert planner._forced_verify_len is None


def test_an_uncaptured_length_is_refused_at_the_call_site():
    """Rejected when requested, not on some later step, and never queued."""
    planner = _planner(tiers=(1, 2, 5))
    with pytest.raises(ValueError, match="captured tier ladder"):
        planner.request_verify_len_pin(3)
    with pytest.raises(ValueError, match="outside"):
        planner.request_verify_len_pin(9)
    assert planner.pending_verify_len_pin() == -1
    assert planner._forced_verify_len is None


def test_pinned_steps_hand_every_request_the_same_window():
    """The whole point: the shape a sweep cell can honestly label."""
    import numpy as np
    import torch

    planner = _planner()
    bs = 8
    rng = np.random.default_rng(5)
    survival = np.sort(rng.random((bs, 5)), axis=1)[:, ::-1].copy()
    planner._gather_rows = lambda **_: torch.tensor(survival,
                                                    dtype=torch.float32)
    planner.adopt_verify_len_pin(2)
    lens = planner.decide_verify_lens(num_gen_requests=bs,
                                      reduce_across_ranks=False)
    assert lens == [2] * bs
    assert planner.stats["forced_steps"] == 1
