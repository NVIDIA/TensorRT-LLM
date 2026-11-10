# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the adaptive EPLB share solver (``trtllm::adaptive_eplb_solve_shares``)."""

import itertools
import math

import numpy as np
import pytest
from _torch.moe.test_adaptive_eplb_policy import GEOMETRY, even_shares

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
    ExpertGeometry,
    slot_rows,
    solve_shares,
)

pytestmark = pytest.mark.cpu_only


def _random_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A home layout with every free slot holding a random expert, and lognormal rows."""
    layout = GEOMETRY.home_layout()
    free = np.flatnonzero(layout < 0)
    layout[free] = rng.integers(0, GEOMETRY.num_experts, size=free.size)
    rows = rng.lognormal(mean=math.log(rng.choice([8.0, 64.0, 512.0])), sigma=1.0, size=16)
    rows[rng.random(16) < 0.15] = 0.0
    return rows, layout


def _gpu_rows(rows, layout, shares, geometry=GEOMETRY) -> np.ndarray:
    """Rows of every GPU: the sum of the rows of its slots."""
    per_slot = slot_rows(rows[None], layout[None], shares[None])[0]
    return per_slot.reshape(geometry.ep_size, -1).sum(1)


@pytest.mark.parametrize("seed", range(40))
def test_shares_split_every_expert_and_beat_even_shares(seed):
    rows, layout = _random_case(np.random.default_rng(seed))
    shares = solve_shares(rows, layout, GEOMETRY)
    assert shares.shape == (24,) and shares.dtype == np.float64
    assert np.all(np.isfinite(shares)) and np.all(shares >= 0.0)
    assert np.all(shares[layout < 0] == 0.0)
    for expert in range(GEOMETRY.num_experts):
        own = shares[layout == expert]
        assert math.fsum(own) == pytest.approx(1.0, abs=1e-9)
        if own.size == 1:
            assert own[0] == 1.0
    even = even_shares(layout[None])[0]
    assert _gpu_rows(rows, layout, shares).max() <= _gpu_rows(rows, layout, even).max() * (1 + 1e-9)


@pytest.mark.parametrize("seed", range(20))
def test_shares_reach_the_grid_optimum(seed):
    """Brute force over shares in steps of 1/16 of every replicated expert of a small layer."""
    geometry = ExpertGeometry(num_experts=4, num_slots=6, ep_size=2)
    rng = np.random.default_rng(seed)
    layout = geometry.home_layout()
    free = np.flatnonzero(layout < 0)
    layout[free] = rng.integers(0, geometry.num_experts, size=free.size)
    rows = rng.lognormal(mean=math.log(100.0), sigma=1.0, size=4)
    replicated = [np.flatnonzero(layout == e) for e in range(4) if np.sum(layout == e) > 1]
    splits = [
        [c for c in itertools.product(range(17), repeat=slots.size) if sum(c) == 16]
        for slots in replicated
    ]
    best = math.inf
    for choice in itertools.product(*splits):
        shares = even_shares(layout[None])[0]
        for slots, split in zip(replicated, choice):
            shares[slots] = np.array(split) / 16.0
        best = min(best, _gpu_rows(rows, layout, shares, geometry).max())
    solved = solve_shares(rows, layout, geometry)
    # The solver's bisection stops at a share error of about 1e-9, so a relative 1e-8.
    assert _gpu_rows(rows, layout, solved, geometry).max() <= best * (1 + 1e-8)


def test_shares_balance_the_gpus_of_a_hot_expert():
    layout = GEOMETRY.home_layout()
    layout[10] = 0  # a copy of expert 0 (home GPU 0) on GPU 1
    rows = np.full(16, 40.0)
    rows[0] = 3000.0
    gpu = _gpu_rows(rows, layout, solve_shares(rows, layout, GEOMETRY))
    # GPUs 0 and 1 carry 3000 + 3 x 40 + 4 x 40 rows between them, split evenly.
    assert gpu[0] == pytest.approx(1640.0, rel=1e-8) and gpu[1] == pytest.approx(1640.0, rel=1e-8)


def test_shares_are_deterministic_and_leave_inputs_untouched():
    rows, layout = _random_case(np.random.default_rng(7))
    rows_copy, layout_copy = rows.copy(), layout.copy()
    first = solve_shares(rows, layout, GEOMETRY)
    second = solve_shares(rows, layout, GEOMETRY)
    assert first.tobytes() == second.tobytes()
    assert np.array_equal(rows, rows_copy) and np.array_equal(layout, layout_copy)


def test_shares_refuse_invalid_inputs():
    rows, layout = _random_case(np.random.default_rng(3))
    rows[0] = 5.0
    solve_shares(rows, layout, GEOMETRY)

    nan = rows.copy()
    nan[3] = math.nan
    inf = rows.copy()
    inf[3] = math.inf
    low = layout.copy()
    low[5] = -2
    high = layout.copy()
    high[5] = GEOMETRY.num_experts
    homeless = layout.copy()
    homeless[layout == 0] = -1  # expert 0 has rows but no slot
    invalid = [
        (rows[:-1], layout),
        (rows, layout.astype(np.float64)),
        (nan, layout),
        (inf, layout),
        (rows, low),
        (rows, high),
        (rows, homeless),
    ]
    for bad_rows, bad_layout in invalid:
        with pytest.raises(ValueError):
            solve_shares(bad_rows, bad_layout, GEOMETRY)
    with pytest.raises(RuntimeError, match="divide evenly"):
        solve_shares(rows, layout[:-1], GEOMETRY)
