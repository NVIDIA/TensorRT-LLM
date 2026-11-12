# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the adaptive EPLB policy: time model, layout planner, install decisions."""

import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import policy
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
    CostGrowth,
    EvaluationScheduler,
    ExpertGeometry,
    TimeModel,
    active_slots,
    agreed_timings,
    choose_layers_to_install,
    evaluate_gains,
    fixed_install_ms,
    plan_layout,
    slot_rows,
)

pytestmark = pytest.mark.cpu_only

GEOMETRY = ExpertGeometry(num_experts=16, num_slots=24, ep_size=4)
MODEL = TimeModel(
    per_active_slot_us=4.645,
    per_extra_tile_us=2.0588531830700743,
    per_row_us=0.008615490581896095,
    hot_slot_extra_tile_us=1.019131452343683,
    scale=0.901323556551518,
    tile_rows=256,
)
# 8 experts on 4 GPUs of 4 slots: expert e is homed on slot 4 * (e // 2) + e % 2, slots 2-3 of
# every GPU are free in the home layout.
SMALL = ExpertGeometry(num_experts=8, num_slots=16, ep_size=4)


def even_shares(layout: np.ndarray) -> np.ndarray:
    """``[L, num_slots]`` shares that split every expert's rows evenly over its copies."""
    shares = np.zeros(layout.shape)
    for row, out in zip(layout, shares):
        active = row >= 0
        out[active] = 1.0 / np.bincount(row[active])[row[active]]
    return shares


# ------------------------------------------------------------------------------------------------
# Time model
# ------------------------------------------------------------------------------------------------


def test_the_policy_imports_with_numpy_alone():
    """The decisions need numpy only, so that they can be computed and replayed without torch."""
    code = (
        "import importlib.util, sys; sys.modules['torch'] = None; "
        f"spec = importlib.util.spec_from_file_location('policy', {policy.__file__!r}); "
        "module = sys.modules['policy'] = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "print(module.plan_layout.__name__)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and done.stdout.split() == ["plan_layout"], done.stderr


def test_home_layout_and_slot_rows():
    home = [0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1, 8, 9, 10, 11, -1, -1, 12, 13, 14, 15, -1, -1]
    assert GEOMETRY.home_layout().tolist() == home
    assert MODEL.slope_us == MODEL.scale * (MODEL.per_extra_tile_us / 256 + MODEL.per_row_us)
    layout = np.array(
        [[0, 1, 2, 3, 0, -1, 4, 5, 6, 7, 0, 4, 8, 9, 10, 11, -1, -1, 12, 13, 14, 15, -1, -1]]
    )
    load = np.arange(16, dtype=np.float64)[None, :] * 30.0
    rows = slot_rows(load, layout, even_shares(layout))
    assert rows[0, 5] == 0.0
    assert rows[0, 6] == pytest.approx(4 * 30.0 * 0.5)
    assert rows.sum() == pytest.approx(load.sum())


def test_balanced_time_is_the_floor():
    load = np.array([[600.0] * 16, [50.0] * 8 + [1150.0] * 8])
    layout = np.stack([GEOMETRY.home_layout()] * 2)
    at_home = MODEL.layer_time_us(slot_rows(load, layout, even_shares(layout)), GEOMETRY.ep_size)
    balanced = MODEL.balanced_time_us(load, GEOMETRY)
    assert at_home[0] == balanced[0]
    assert at_home[1] > balanced[1]
    # One layer, 2 GPUs x 2 slots, rows [[300, 0], [10, 20]]: GPU 0 has F=1 slot with rows, Lt=1
    # extra tile, R=300 rows and a hottest slot of H=1 extra tile; GPU 1 has F=2, Lt=0, R=30, H=0.
    one, two = MODEL, np.array([[300.0, 0.0, 10.0, 20.0]])
    gpu0 = one.scale * (
        one.per_active_slot_us * 1
        + one.per_extra_tile_us * 1
        + one.per_row_us * 300
        + one.hot_slot_extra_tile_us * 1
    )
    gpu1 = one.scale * (one.per_active_slot_us * 2 + one.per_row_us * 30)
    assert gpu0 > gpu1
    assert one.layer_time_us(two, 2)[0] == pytest.approx(max(gpu0, gpu1), rel=1e-12)


# ------------------------------------------------------------------------------------------------
# Layout planner
# ------------------------------------------------------------------------------------------------


def _layout(geometry: ExpertGeometry, replicas: dict[int, int] | list) -> np.ndarray:
    layout = geometry.home_layout()
    for slot, expert in dict(replicas).items():
        layout[slot] = expert
    return layout


def _plan(load, current=None, total=None, geometry=SMALL, keep_first=False):
    home = geometry.home_layout()
    return plan_layout(
        np.asarray(load, dtype=np.float64),
        geometry,
        home,
        home if current is None else current,
        active_slots(geometry) if total is None else total,
        keep_first,
    )


def _check_one_install(plan, geometry: ExpertGeometry, current: np.ndarray) -> None:
    home = geometry.home_layout()
    homes = home >= 0
    assert (plan.layout[homes] == home[homes]).all()
    replica = ~homes & (plan.layout >= 0)
    assert ((plan.layout[replica] == current[replica]) | (current[replica] < 0)).all()
    for expert, row in enumerate(plan.expert_slots):
        assert list(row) == sorted(row)
        assert (plan.layout[list(row)] == expert).all()
    assert sum(len(row) for row in plan.expert_slots) == int((plan.layout >= 0).sum())


def _gpus(row: tuple[int, ...], geometry: ExpertGeometry) -> list[int]:
    return sorted(slot // geometry.local_slots for slot in row)


def _check_steps(
    load: np.ndarray, current: np.ndarray, geometry: ExpertGeometry, keep=False
) -> None:
    """The invariants every step of :func:`plan_layout` keeps (``keep``: keep-first), on its own
    intermediate rows."""
    home = geometry.home_layout()
    homes, is_home = home.tolist(), (home >= 0).tolist()
    home_slot = [homes.index(e) for e in range(geometry.num_experts)]
    rows = load.tolist()
    total = active_slots(geometry)
    spare = geometry.ep_size * (geometry.local_slots - geometry.local_experts)
    assert total - geometry.num_experts <= spare
    installed = policy._expert_slots_of(current.tolist(), geometry.num_experts)
    copies = policy._copies(rows, total)
    proposed = policy._place_replicas(rows, homes, copies, geometry, installed if keep else None)
    fitted = policy._fit_free_slots(installed, proposed, rows, geometry)
    paired = policy._keep_replica_slots(installed, fitted, is_home, geometry)
    targets = policy._one_install(installed, fitted, is_home, geometry)
    for step in (proposed, fitted, paired, targets):
        owners = [slot for row in step for slot in row]
        assert all(step) and len(owners) == len(set(owners))
        assert all(row[0] == home_slot[e] or home_slot[e] in row for e, row in enumerate(step))

    def replicas_per_gpu(step) -> list[int]:
        counts = [0] * geometry.ep_size
        for row in step:
            for slot in row:
                counts[slot // geometry.local_slots] += not is_home[slot]
        return counts

    assert all(a <= b for a, b in zip(replicas_per_gpu(fitted), replicas_per_gpu(proposed)))
    for a, b, c in zip(fitted, paired, targets):
        assert _gpus(a, geometry) == _gpus(b, geometry) == _gpus(c, geometry)


def test_first_plan_fills_the_budget_from_the_homes():
    # The budget keeps half the free slots of every GPU free.
    assert active_slots(ExpertGeometry(num_experts=384, num_slots=512, ep_size=32)) == 448
    assert active_slots(SMALL) == 12
    assert active_slots(ExpertGeometry(num_experts=8, num_slots=20, ep_size=4)) == 16
    assert active_slots(ExpertGeometry(num_experts=8, num_slots=8, ep_size=4)) == 8
    rng = np.random.default_rng(7)
    for _ in range(20):
        load = rng.gamma(0.5, 100.0, SMALL.num_experts)
        plan = _plan(load)
        _check_one_install(plan, SMALL, SMALL.home_layout())
        copies = [len(row) for row in plan.expert_slots]
        assert sum(copies) == active_slots(SMALL)
        hot_first = np.argsort(-load, kind="stable")
        assert all(copies[a] >= copies[b] for a, b in zip(hot_first, hot_first[1:]))


def test_one_replica_goes_to_the_hottest_expert_and_ties_to_the_larger_id():
    plan = _plan([1.0, 2.0, 1.0, 9.0, 1.0, 1.0, 9.0, 1.0], total=SMALL.num_experts + 1)
    assert [len(row) for row in plan.expert_slots] == [1, 1, 1, 1, 1, 1, 2, 1]
    # Rows that one float32 cannot tell apart are no tie: the planner ranks the float64 rows that
    # the share solver and the gains use.
    plan = _plan([1.0, 1.0, 1.0, 10.0000001, 1.0, 1.0, 10.0, 1.0], total=SMALL.num_experts + 1)
    assert len(plan.expert_slots[3]) == 2 and len(plan.expert_slots[6]) == 1


def test_replica_goes_to_the_least_loaded_gpu_and_ties_to_the_lower_gpu():
    plan = _plan([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0], total=9)
    assert plan.layout.tolist() == _layout(SMALL, {2: 7}).tolist()
    plan = _plan([5.0, 5.0, 1.0, 1.0, 3.0, 3.0, 1.0, 10.0], total=9)
    assert plan.layout.tolist() == _layout(SMALL, {6: 7}).tolist()
    # Every GPU with room holds the expert: a second copy on a GPU.
    plan = _plan([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert plan.expert_slots[0] == (0, 2, 6, 10, 14)
    assert plan.layout.tolist() == _layout(SMALL, {2: 0, 6: 0, 10: 0, 14: 0}).tolist()


def test_installed_replicas_keep_their_slots():
    current = _layout(SMALL, {7: 7})
    plan = _plan([5.0, 5.0, 1.0, 1.0, 3.0, 3.0, 1.0, 10.0], current=current, total=9)
    assert plan.layout.tolist() == current.tolist()  # on the same GPU
    # GPU 1's free slots hold replicas of experts 0 and 4; the plan wants a replica of expert 6
    # there, so GPU 1 keeps expert 0's replica (rows per copy tie with expert 4's, smaller id).
    current = _layout(SMALL, {6: 0, 7: 4})
    plan = _plan([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 8.0], current=current)
    assert plan.layout.tolist() == _layout(SMALL, {2: 6, 3: 7, 6: 0, 10: 7}).tolist()
    _check_one_install(plan, SMALL, current)


def test_keep_first_keeps_a_needed_replica_on_its_least_loaded_gpu_and_copies_nothing():
    # Expert 0 needs one replica and holds two, on GPU 1 (4 rows) and GPU 3 (2 rows); the plan
    # copies one to GPU 2 (2 rows, the lower id) instead.
    current, load = _layout(SMALL, {6: 0, 14: 0}), [8.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert _plan(load, current, 9).layout.tolist() == _layout(SMALL, {10: 0}).tolist()
    kept = _plan(load, current, 9, keep_first=True)
    assert kept.layout.tolist() == _layout(SMALL, {14: 0}).tolist()


def test_random_installed_layouts_are_reached_in_one_install():
    geometry = ExpertGeometry(num_experts=48, num_slots=64, ep_size=4)
    home = geometry.home_layout()
    free = np.flatnonzero(home < 0)
    rng = np.random.default_rng(11)
    for _ in range(30):
        current = home.copy()
        taken = rng.choice(free, size=rng.integers(0, len(free) + 1), replace=False)
        current[taken] = rng.integers(0, geometry.num_experts, size=len(taken))
        load = rng.gamma(0.3, 50.0, geometry.num_experts)
        plan = _plan(load, current=current, geometry=geometry)
        _check_one_install(plan, geometry, current)
        kept = _plan(load, current=current, geometry=geometry, keep_first=True)
        _check_one_install(kept, geometry, current)
        assert sum(len(row) for row in kept.expert_slots) <= active_slots(geometry)
        _check_steps(load, current, geometry)
        _check_steps(load, current, geometry, keep=True)
        assert sum(len(row) for row in plan.expert_slots) <= active_slots(geometry)
        again = _plan(load.tolist(), current=current.tolist(), geometry=geometry)
        assert again.expert_slots == plan.expert_slots


# ------------------------------------------------------------------------------------------------
# Install decisions
# ------------------------------------------------------------------------------------------------


def _gains(load, current, new, new_shares=None):
    current, new, load = np.atleast_2d(current), np.atleast_2d(new), np.atleast_2d(load)
    shares = even_shares(new) if new_shares is None else np.atleast_2d(new_shares)
    return evaluate_gains(load, current, even_shares(current), new, shares, MODEL, GEOMETRY)


def _times(load, layout, shares) -> float:
    rows = slot_rows(np.atleast_2d(load), np.atleast_2d(layout), np.atleast_2d(shares))
    return float(MODEL.layer_time_us(rows, GEOMETRY.ep_size)[0])


def test_evaluate_gains_replica_of_hot_expert_pays():
    home = GEOMETRY.home_layout()
    load = np.full(16, 100.0)
    load[0] = 2000.0
    replicated = home.copy()
    replicated[[4, 10]] = 0  # free slots on GPUs 0 and 1
    gains = _gains(np.stack([load, load]), np.stack([home, home]), np.stack([replicated, home]))
    assert gains.shape == (2,) and gains[0] > 0 and gains[1] == 0.0
    t_cur = _times(load, home, even_shares(home[None])[0])
    assert gains[0] == pytest.approx(
        (t_cur - _times(load, replicated, even_shares(replicated[None])[0])) / 1e3
    )
    # The new layout is priced at its own shares: an install applies them together.
    solved = even_shares(replicated[None])[0]
    solved[[0, 4, 10]] = [0.3, 0.2, 0.5]
    t_solved = _times(load, replicated, solved)
    assert _gains(load, home, replicated, solved)[0] == pytest.approx((t_cur - t_solved) / 1e3)
    empty = np.zeros((0, GEOMETRY.num_slots), dtype=np.int64)
    assert _gains(np.zeros((0, 16)), empty, empty).shape == (0,)
    a, b = np.stack([home, replicated]), np.stack([replicated, home])  # a replica pays route_us
    args = (np.stack([load] * 2), a, even_shares(a), b, even_shares(b))
    got = evaluate_gains(*args, replace(MODEL, route_us=7.0), GEOMETRY)
    assert got == pytest.approx([gains[0] - 7e-3, 7e-3 - gains[0]])


def test_choose_layers_to_install():
    gains = {1: 0.05, 2: 0.02, 3: 0.001, 4: 0.03}
    assert choose_layers_to_install(gains, 100.0, dict.fromkeys(gains, 0.5)) == [1, 4, 2]
    assert choose_layers_to_install(gains, None, dict.fromkeys(gains, 0.5)) == []
    assert choose_layers_to_install({1: 0.05, 2: 0.05}, 100.0, {1: 4.9, 2: 4.9}) == [1, 2]
    assert choose_layers_to_install({1: 0.05, 2: 0.05}, 100.0, {1: 5.0, 2: 5.0}) == []
    assert choose_layers_to_install({1: 0.06, 2: 0.05}, 100.0, {1: 6.5, 2: 4.0}) == [2]


def test_group_readings_copy_and_apply():
    rank0 = [(10, 2, 1.0), (20, 3, 2.0), (30, 1, 0.5)]
    rank1 = [(10, 2, 1.5), (20, 3, 1.0)]
    common = agreed_timings([rank0, rank1])
    assert common == [(10, 2, 1.5), (20, 3, 2.0)]
    assert fixed_install_ms([(10, 2, 1.0), (20, 2, 1.2)]) is None
    assert fixed_install_ms([(10, 2, 3.0), (20, 4, 5.0)]) == pytest.approx(1.0)
    assert fixed_install_ms([(10, 2, 0.2), (20, 4, 1.0)]) == 0.0
    with pytest.raises(ValueError):
        agreed_timings([rank0, [(10, 3, 1.0)]])


def test_evaluation_scheduler_cost_growth_and_plan():
    tracker = EvaluationScheduler(MODEL, GEOMETRY)
    layers = [3, 5]
    layout = np.stack([GEOMETRY.home_layout()] * 2)
    shares = even_shares(layout)
    flat = np.full((2, 16), 100.0)
    skew = flat.copy()
    skew[:, 0] = 1000.0
    assert tracker.read_cost([], flat[:0], layout[:0], shares[:0], 5) is None
    assert tracker.read_cost(layers, flat, layout, shares, 10) == CostGrowth(None, 0, 0)
    first = tracker.read_cost(layers, skew, layout, shares, 30)
    balanced = MODEL.balanced_time_us(skew, GEOMETRY)
    cost = (MODEL.layer_time_us(slot_rows(skew, layout, shares), 4) - balanced).sum() / 1e3
    assert (first.layers, first.interval) == (2, 20) and first.rate == pytest.approx(cost / 20)
    # The same shape at twice the rows is no drift of the configuration's cost.
    assert tracker.read_cost(layers, skew * 2.0, layout, shares, 40) == CostGrowth(0.0, 2, 10)
    no_rate = tracker.plan(31, 29, 4, [], [], CostGrowth(None, 0, 0), None)
    assert (no_rate.interval, no_rate.open_after) == (7, 1)
    growth = CostGrowth(1e-4, 1, 1)
    plan = tracker.plan(31, 29, 4, [2.0, 0.5, 1.0], [(1, 2, 3.0), (2, 4, 5.0)], growth, None)
    # sqrt(2 * fixed / rate) calls: the median copy 1.0 ms and the install intercept 1.0 ms.
    assert (plan.interval, plan.open_after) == (200, 194)
    # A falling cost keeps the last rising rate; the steps left cap the interval.
    capped = tracker.plan(31, 29, 4, [1.0], [], CostGrowth(-1.0, 1, 1), 50.5)
    assert (capped.interval, capped.open_after) == (51, 45)
    kept = tracker.plan(31, 29, 4, [1.0], [], CostGrowth(-1.0, 1, 1), None)
    assert kept.interval == math.ceil(math.sqrt(2 * 1.0 / 1e-4))
    # No layer read this window: the interval stays at its floor whatever was learnt before.
    unread = tracker.plan(31, 29, 4, [1.0], [], None, None)
    assert unread.interval == 7
    # An even number of snapshot times: their median is the mean of the middle two.
    even = tracker.plan(31, 29, 4, [1.0, 3.0], [], CostGrowth(1e-4, 1, 1), None)
    assert even.interval == math.ceil(math.sqrt(2 * 2.0 / 1e-4))


def test_rate_floor_follows_the_reading_interval():
    tracker = EvaluationScheduler(MODEL, GEOMETRY)
    per_call = 3 * MODEL.slope_us / 1e3
    assert tracker.min_resolvable_rate(3, 1) == per_call
    assert tracker.min_resolvable_rate(3, 20) == per_call / 20
    # Between the floor of one call and that of the reading's 20 calls: resolved.
    resolved = tracker.plan(10, 10, 2, [1.0], [], CostGrowth(per_call / 2, 3, 20), None)
    assert resolved.interval == math.ceil(math.sqrt(2 * 1.0 / (per_call / 2)))
    # Below the floor of the reading's interval: the last resolved rate stays.
    kept = tracker.plan(20, 20, 2, [1.0], [], CostGrowth(per_call / 40, 3, 20), None)
    assert kept.interval == resolved.interval
    # A negative rate reading is no rate at all (a tracker that has learnt nothing yet).
    fresh = EvaluationScheduler(MODEL, GEOMETRY)
    assert fresh.plan(5, 5, 2, [1.0], [], CostGrowth(-0.5, 1, 1), None).interval == 3


def test_a_step_takes_the_costs_of_the_bucket_it_pads_to():
    buckets = ((40.0, 0.8, 5.0, 1.5, True), (160.0, 1.1, 0.0, 2.0, False))
    at = [TimeModel(1.0, 2.0, 3.0, 4.0, 1.0, 8, buckets).at(r) for r in (1, 40, 41, 160, 1e6)]
    want = [buckets[0][1:]] * 2 + [buckets[1][1:]] * 3
    assert [(m.scale, m.per_active_slot_us, m.route_us, m.time_shares) for m in at] == want
    assert at[2].scales == () and at[2].at(1.0).scale == 1.1
