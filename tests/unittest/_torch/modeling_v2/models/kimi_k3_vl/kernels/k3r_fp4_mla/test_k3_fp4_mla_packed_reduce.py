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
"""K3's FP4 MLA decode task plan and split reduction (``kernels/k3r_fp4_mla/fp4_mla_packed_reduce``).

The host planners (the rule, wave and makespan (LPT) plans, the task order and the modelled
makespan), the CuTe DSL plan kernel against them (optionally fused with the step's generation
lengths), and the split reduction against an FP64 reference, eager and replayed.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires an SM100-or-newer CUDA device",
)


def _plan_mod():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_fp4_mla import (
        fp4_mla_packed_reduce,
    )

    return fp4_mla_packed_reduce


@pytest.mark.parametrize(
    "tiles,capacity,wave_tasks,expected_tiles_per_task,expected_begins",
    [
        # Even batch under one wave (helix16 B4, 6 m-tiles -> 17 wave tasks): one wave of 32-tile tasks.
        ([128, 128, 128, 128], 24, 17, 32, [0, 4, 8, 12, 16]),
        # One long request among short ones takes most of the slots (two-wave target, fit by the fallback share).
        ([1, 79, 3, 176], 24, 17, 13, [0, 1, 8, 9, 23]),
        # Full machine, longest 1.5x above the wave share: two-wave target, the short requests also halve.
        ([128] * 15 + [200], 48, 17, 67, list(range(0, 32, 2)) + [33]),
        # Full machine, longest within 1.5x of the wave share: no split at all.
        ([128] * 15 + [130], 48, 17, 130, list(range(17))),
        # Strong skew in a full machine: the long request gets twelve tasks.
        ([128] * 15 + [1024], 48, 17, 86, list(range(0, 32, 2)) + [42]),
        # Strong skew in a small batch: two-wave target, fit by the fallback share.
        ([128, 128, 128, 1024], 24, 17, 71, [0, 2, 4, 6, 21]),
        # A single request fills one wave with 16 tasks, or honours the minimum.
        ([128], 16, 17, 8, [0, 16]),
        ([8], 16, 17, 2, [0, 4]),
        # No spare slots: one task per request, however long.
        ([5, 900], 2, 17, 900, list(range(3))),
    ],
)
def test_plan_packed_tasks_balances_by_kv_length(
    tiles: list[int],
    capacity: int,
    wave_tasks: int,
    expected_tiles_per_task: int,
    expected_begins: list[int],
) -> None:
    tiles_per_task, begins = _plan_mod().plan_packed_tasks(tiles, capacity, wave_tasks)
    assert tiles_per_task == expected_tiles_per_task
    assert begins == expected_begins
    assert begins[-1] <= capacity
    for request_tiles, begin, end in zip(tiles, begins, begins[1:]):
        assert end > begin
        assert (end - begin - 1) * tiles_per_task < request_tiles <= (end - begin) * tiles_per_task


def test_plan_packed_tasks_one_wave_for_helix() -> None:
    plan = _plan_mod().plan_packed_tasks
    # helix16 bs16 (6 m-tiles, 17 wave tasks), per-rank KV skewed 3.5x: the two-wave
    # target cuts 12-tile pieces (42 tasks); a Helix rank keeps one wave of 28-tile tasks.
    tiles = [16] * 12 + [40, 48, 56, 56]
    assert plan(tiles, 48, 17) == (12, list(range(0, 26, 2)) + [28, 32, 37, 42])
    assert plan(tiles, 48, 17, one_wave=True) == (28, list(range(13)) + [14, 16, 18, 20])
    # Even Helix batches are unchanged.
    assert plan([128] * 4, 24, 17, one_wave=True) == plan([128] * 4, 24, 17)


@pytest.mark.parametrize(
    "tiles,capacity,wave_tasks,expected_tiles_per_task,expected_begins",
    [
        # Even batch under one wave (helix16 B4, 6 m-tiles -> 17 wave tasks): one wave of 32-tile tasks.
        ([128, 128, 128, 128], 24, 17, 32, [0, 4, 8, 12, 16]),
        # One long request among short ones: 16-tile tasks; the one task past the wave is a short tail.
        ([1, 79, 3, 176], 24, 17, 16, [0, 1, 6, 7, 18]),
        # Full machine, longest 1.5x above the wave share: 100-tile tasks, the tails in a second wave.
        ([128] * 15 + [200], 48, 17, 100, list(range(0, 34, 2))),
        # Full machine, longest within 1.5x of the wave share: no split at all.
        ([128] * 15 + [130], 48, 17, 130, list(range(17))),
        # Strong skew in a full machine: eleven 94-tile tasks for the long request, short tails behind.
        ([128] * 15 + [1024], 48, 17, 94, list(range(0, 32, 2)) + [41]),
        # Strong skew in a small batch: 94-tile tasks, 11 for the long request, fill one wave.
        ([128, 128, 128, 1024], 24, 17, 94, [0, 2, 4, 6, 17]),
        # A single request fills one wave with 16 tasks; a short one is not worth splitting.
        ([128], 16, 17, 8, [0, 16]),
        ([8], 16, 17, 8, [0, 1]),
        # No spare slots: one task per request, however long.
        ([5, 900], 2, 17, 900, list(range(3))),
    ],
)
def test_plan_packed_tasks_wave_plan_balances_by_kv_length(
    tiles: list[int],
    capacity: int,
    wave_tasks: int,
    expected_tiles_per_task: int,
    expected_begins: list[int],
) -> None:
    tiles_per_task, begins = _plan_mod().plan_packed_tasks(
        tiles, capacity, wave_tasks, wave_plan=True
    )
    assert tiles_per_task == expected_tiles_per_task
    assert begins == expected_begins
    assert begins[-1] <= capacity
    for request_tiles, begin, end in zip(tiles, begins, begins[1:]):
        assert end > begin
        assert (end - begin - 1) * tiles_per_task < request_tiles <= (end - begin) * tiles_per_task


def test_plan_packed_tasks_simulates_helix_ranks() -> None:
    plan = _plan_mod().plan_packed_tasks
    # helix16 bs16 (6 m-tiles, 17 wave tasks), per-rank KV skewed 3.5x: the Helix rule targets one
    # wave of 28-tile tasks (21 tasks, a second wave); the simulation keeps the 16 requests whole.
    tiles = [16] * 12 + [40, 48, 56, 56]
    assert plan(tiles, 48, 17, one_wave=True, wave_plan=False) == (
        28,
        list(range(13)) + [14, 16, 18, 20],
    )
    assert plan(tiles, 48, 17, one_wave=True, wave_plan=True) == (56, list(range(17)))
    # Even Helix batches are unchanged.
    assert plan([128] * 4, 24, 17, one_wave=True, wave_plan=True) == plan(
        [128] * 4, 24, 17, wave_plan=True
    )


def test_plan_packed_tasks_simulates_the_waves() -> None:
    reduce_mod = _plan_mod()
    plan = reduce_mod.plan_packed_tasks
    # DEP16 rank on Rubin (3 m-tiles, 35 wave tasks), production-like skew: the two-wave target
    # (93 tiles) rounds every request up to 74 tasks, a third wave; 206-tile tasks fill one wave.
    tiles = [1848, 1250, 977, 743, 586, 430, 313, 196, 106]
    assert plan(tiles, 9 * 9, 35, wave_plan=True) == (
        206,
        [0, 9, 16, 21, 25, 28, 31, 33, 34, 35],
    )
    # Nine equal requests: 36 one-wave tasks spill one task; three 239-tile pieces each fit one wave.
    assert plan([715] * 9, 9 * 9, 35, wave_plan=True) == (239, list(range(0, 28, 3)))
    simulate = reduce_mod.plan_simulated_makespan
    cost = reduce_mod.PLAN_TASK_COST_TILES + reduce_mod.PLAN_SPLIT_COST_TILES
    # 36 equal 179-tile tasks on 35 slots: the last one waits for a whole task.
    assert simulate([716] * 9, 179, 35) == 2 * (179 + cost)
    # The 36th task (a 178-tile tail) starts when the first tail frees its slot.
    assert simulate([715] * 9, 179, 35) == (178 + cost) + (178 + cost)
    # 2-tile tails free their slots early, so a spilled tail does not extend the makespan.
    assert simulate([181] * 18, 179, 35) == 179 + cost
    # Requests that keep one task pay no split cost.
    assert simulate([100, 300], 300, 35) == 300 + reduce_mod.PLAN_TASK_COST_TILES


def test_plan_task_order_runs_full_tasks_then_longest_tails() -> None:
    order = _plan_mod().plan_task_order
    # Tails: request 0 -> 4 tiles, request 1 -> 10 (a full task), request 2 -> 3, request 3 -> 4.
    full = [(0, 0), (0, 1), (1, 0), (1, 1), (3, 0)]
    tails = [(1, 2), (0, 2), (3, 1), (2, 0)]
    assert order([24, 30, 3, 14], 10, wave_plan=True) == full + tails
    # One task per request: longest first, ties in request order.
    assert order([5, 9, 5, 2], 10, wave_plan=True) == [(1, 0), (0, 0), (2, 0), (3, 0)]


def test_plan_without_wave_plan_keeps_the_rule_and_request_order() -> None:
    reduce_mod = _plan_mod()
    plan = reduce_mod.plan_packed_tasks
    # Without the wave simulation: the rule's length.
    assert plan([1, 79, 3, 176], 24, 17, wave_plan=False) == (13, [0, 1, 8, 9, 23])
    assert plan([128] * 15 + [1024], 48, 17, wave_plan=False) == (
        86,
        list(range(0, 32, 2)) + [42],
    )
    order = reduce_mod.plan_task_order([24, 30, 3, 14], 10, wave_plan=False)
    assert order == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (3, 0), (3, 1)]


def test_plan_packed_tasks_honours_the_runtime_profile_cap() -> None:
    plan = _plan_mod().plan_packed_tasks
    # An even full machine would not split, but the profile only covers 512 tiles.
    assert plan([1024] * 16, 48, 17, max_tiles_per_task=512) == (512, list(range(0, 34, 2)))
    # The cap never pushes a batch past its slots when it is ceil(longest / (splits - 1)).
    assert plan([1024, 3, 1024, 700], 4 * 6, 17, max_tiles_per_task=205)[1][-1] <= 24


def test_plan_packed_tasks_random_batches_fit_their_budget() -> None:
    import random

    rng = random.Random(1234)
    plan = _plan_mod().plan_packed_tasks
    for _ in range(2000):
        requests = rng.randint(1, 40)
        tiles = [rng.choice([1, 2, 3, 8, 64, 128, 500, 1024, 4096]) for _ in range(requests)]
        splits = rng.randint(1, 16)
        wave_tasks = rng.choice([17, 53, 106])
        tiles_per_task, begins = plan(tiles, requests * splits, wave_tasks)
        assert begins[-1] <= requests * splits
        assert all(end > begin for begin, end in zip(begins, begins[1:]))
        assert min(2, max(tiles)) <= tiles_per_task <= max(tiles)
        assert all(
            math.ceil(t / tiles_per_task) == e - b for t, b, e in zip(tiles, begins, begins[1:])
        )


# A production-like DEP16 rank: 9 requests, 1.65M KV tokens.
_DEP16_RANK_KV_LENS = [473000, 320000, 250000, 190000, 150000, 110000, 80000, 50000, 27000]


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 7) if torch.cuda.is_available() else True,
    reason="the CuTe DSL plan kernel is compiled for Rubin SM107",
)
@pytest.mark.parametrize(
    "kv_lens,splits,wave_tasks,query_len,helix,wave_plan",
    [
        # The rule plan: tasks in request order.
        ([32768] * 15 + [262144], 4, 17, 8, False, False),
        ([131, 20000, 600, 45000], 10, 17, 8, False, False),
        ([131, 20000, 600, 45000], 10, 17, 8, True, False),
        ([300, 70000, 2200], 16, 106, 1, False, False),
        ([5, 5, 5], 2, 106, 4, False, False),
        # The wave plan.
        ([32768] * 15 + [262144], 4, 17, 8, False, True),
        ([131, 20000, 600, 45000], 10, 17, 8, False, True),
        ([131, 20000, 600, 45000], 10, 17, 8, True, True),
        ([300, 70000, 2200], 16, 106, 1, False, True),
        ([5, 5, 5], 2, 106, 4, False, True),
        # DEP16 rank shapes (3 m-tiles, 35 wave tasks): the simulated task length.
        (_DEP16_RANK_KV_LENS, 9, 35, 4, False, True),
        ([183000] * 9, 9, 35, 4, False, True),
        ([103000] * 16, 6, 35, 4, False, True),
        # The rule plan on the same shapes, off and on Helix.
        (_DEP16_RANK_KV_LENS, 9, 35, 4, False, False),
        ([131, 20000, 600, 45000], 10, 17, 8, True, False),
    ],
)
def test_packed_plan_kernel_matches_host_mirror(
    kv_lens: list[int], splits: int, wave_tasks: int, query_len: int, helix: bool, wave_plan: bool
) -> None:
    reduce_mod = _plan_mod()
    requests = len(kv_lens)
    tasks = requests * splits
    kv = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    bounds = None
    bounds_list = None
    if helix:
        # Per-token Helix bounds: ragged, last token of each request decides.
        bounds_list = [
            max(1, kv_len - (query_len - 1 - q) * 37)
            for kv_len in kv_lens
            for q in range(query_len)
        ]
        bounds = torch.tensor(bounds_list, dtype=torch.int32, device="cuda")
    tiles = reduce_mod.request_kv_tiles(kv_lens, query_len, bounds_list)
    cap = max(tiles)
    expected_tiles, expected_begins = reduce_mod.plan_packed_tasks(
        tiles, tasks, wave_tasks, cap, one_wave=helix, wave_plan=wave_plan
    )
    plan = torch.full(
        (reduce_mod.packed_plan_length(requests, tasks),), -7, dtype=torch.int32, device="cuda"
    )
    reduce_mod.launch_packed_plan(
        kv,
        plan,
        num_requests=requests,
        num_splits=splits,
        wave_tasks=wave_tasks,
        max_tiles_per_task=cap,
        query_len=query_len,
        helix_kv_bounds=bounds,
        wave_plan=wave_plan,
    )
    torch.cuda.synchronize()
    plan = plan.tolist()
    assert plan[0] == expected_tiles
    assert plan[1] == expected_begins[-1]
    assert plan[2 : 3 + requests] == expected_begins
    tables = 3 + requests
    request_of, split_of, count_of = (
        plan[tables : tables + tasks],
        plan[tables + tasks : tables + 2 * tasks],
        plan[tables + 2 * tasks : tables + 3 * tasks],
    )
    # Tasks are listed in grid order; each request still owns begins[r + 1] - begins[r] of them.
    order = reduce_mod.plan_task_order(tiles, expected_tiles, wave_plan)
    assert len(order) == expected_begins[-1]
    for task, (request, split) in enumerate(order):
        assert (request_of[task], split_of[task], count_of[task]) == (
            request,
            split,
            expected_begins[request + 1] - expected_begins[request],
        )
    # Unused slots point past every request's tiles and are never scheduled.
    for task in range(expected_begins[-1], tasks):
        assert request_of[task] == requests - 1 and split_of[task] * expected_tiles >= tiles[-1]


@pytest.mark.parametrize("helix", [False, True])
def test_packed_plan_kernel_writes_generation_lengths(helix: bool) -> None:
    reduce_mod = _plan_mod()
    kv_lens = [40000, 1200, 9000, 70000]
    prompt_lens = [1, 3, 1, 8]
    generation_len = 8
    corrected = [kv - prompt + generation_len for kv, prompt in zip(kv_lens, prompt_lens)]
    requests, splits, wave_tasks, query_len = len(kv_lens), 6, 17, generation_len
    tasks = requests * splits
    bounds = None
    bounds_list = None
    if helix:
        bounds_list = [
            max(1, kv_len - (query_len - 1 - q) * 37)
            for kv_len in corrected
            for q in range(query_len)
        ]
        bounds = torch.tensor(bounds_list, dtype=torch.int32, device="cuda")
    tiles = reduce_mod.request_kv_tiles(corrected, query_len, bounds_list)
    cap = max(tiles)
    expected_tiles, expected_begins = reduce_mod.plan_packed_tasks(
        tiles, tasks, wave_tasks, cap, one_wave=helix
    )
    kv = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    prompt = torch.tensor(prompt_lens, dtype=torch.int32, device="cuda")
    corrected_out = torch.full((requests,), -1, dtype=torch.int32, device="cuda")
    generation_out = torch.full((requests,), -1, dtype=torch.int32, device="cuda")
    plan = torch.full(
        (reduce_mod.packed_plan_length(requests, tasks),), -7, dtype=torch.int32, device="cuda"
    )
    reduce_mod.launch_packed_plan(
        kv,
        plan,
        num_requests=requests,
        num_splits=splits,
        wave_tasks=wave_tasks,
        max_tiles_per_task=cap,
        query_len=query_len,
        helix_kv_bounds=bounds,
        generation_lengths=(prompt, corrected_out, generation_out, generation_len),
    )
    torch.cuda.synchronize()
    # One launch writes the lengths every FP4 MLA kernel of the step reads and
    # plans on the corrected ones.
    assert corrected_out.tolist() == corrected
    assert generation_out.tolist() == [generation_len] * requests
    plan = plan.tolist()
    assert plan[0] == expected_tiles
    assert plan[1] == expected_begins[-1]
    assert plan[2 : 3 + requests] == expected_begins


def test_plan_packed_tasks_lpt_launches_longest_first() -> None:
    import random

    rng = random.Random(99)
    mod = _plan_mod()
    for _ in range(300):
        requests = rng.randint(1, 40)
        tiles = [
            rng.choice([1, 2, 3, 8, 43, 64, 128, 500, 704, 1024, 3516]) for _ in range(requests)
        ]
        splits = rng.randint(1, 16)
        m_tiles = rng.choice([1, 5, 6])
        wave_clusters = rng.choice([74, 106])
        capacity = requests * splits
        tiles_per_task, begins, order = mod.plan_packed_tasks_lpt(
            tiles, capacity, max(1, wave_clusters // m_tiles), m_tiles, wave_clusters
        )
        assert begins[-1] == len(order) <= capacity
        assert all(
            math.ceil(t / tiles_per_task) == e - b for t, b, e in zip(tiles, begins, begins[1:])
        )
        # Every task of every request exactly once, longest first.
        assert sorted(order) == [
            (request, split)
            for request in range(requests)
            for split in range(begins[request + 1] - begins[request])
        ]
        lengths = [min(tiles_per_task, tiles[r] - s * tiles_per_task) for r, s in order]
        assert min(lengths) > 0 and lengths == sorted(lengths, reverse=True)


def test_plan_packed_tasks_lpt_shortens_the_modelled_schedule() -> None:
    mod = _plan_mod()
    mixed = [
        math.ceil(kv / 256) for kv in (20000, 40000, 80000, 100000, 150000, 200000, 250000, 600000)
    ]
    for tiles, m_tiles, expected_tiles in (
        # dep16 B8 MTP7 at 180K: two full tasks per request and eight 54-tile
        # tails that fill the ten cluster slots the full tasks leave free.
        ([704] * 8, 6, 325),
        # dep16 B16 MTP5 at 175K: one task per request leaves 26 of 106 slots idle.
        ([684] * 16, 5, 555),
        (mixed, 6, None),
    ):
        capacity = 8 * len(tiles)
        wave_tasks = 106 // m_tiles
        tiles_per_task, _, _ = mod.plan_packed_tasks_lpt(tiles, capacity, wave_tasks, m_tiles, 106)
        balanced, _ = mod.plan_packed_tasks(tiles, capacity, wave_tasks, wave_plan=False)
        chosen = mod.lpt_makespan(tiles, tiles_per_task, m_tiles, 106, capacity)
        assert chosen <= mod.lpt_makespan(tiles, balanced, m_tiles, 106, capacity)
        if expected_tiles is not None:
            assert tiles_per_task == expected_tiles
    # A one-tile gain is inside the margin: helix16 B16 MTP5 keeps whole requests.
    assert mod.plan_packed_tasks_lpt([43] * 16, 128, 21, 5, 106, one_wave=True)[0] == 43
    # Batches past the simulation limit keep the balanced length.
    tiles = [43] * 80
    assert (
        mod.plan_packed_tasks_lpt(tiles, 160, 17, 6, 106)[0]
        == mod.plan_packed_tasks(tiles, 160, 17, wave_plan=False)[0]
    )


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 7) if torch.cuda.is_available() else True,
    reason="the CuTe DSL plan kernel is compiled for Rubin SM107",
)
@pytest.mark.parametrize(
    "kv_lens,splits,query_len,helix,m_tiles",
    [
        ([180000] * 8, 8, 8, False, 6),
        ([175000] * 16, 8, 6, False, 5),
        ([20000, 40000, 80000, 100000, 150000, 200000, 250000, 600000], 8, 8, False, 6),
        ([131, 20000, 600, 45000], 10, 8, True, 6),
        ([11000] * 32, 8, 8, True, 6),
        ([300, 70000, 2200], 16, 1, False, 1),
        # Past the simulation limit: balanced length, longest first.
        ([5000 + 300 * i for i in range(80)], 2, 8, False, 6),
    ],
)
def test_packed_plan_kernel_lpt_matches_host_mirror(
    kv_lens: list[int], splits: int, query_len: int, helix: bool, m_tiles: int
) -> None:
    reduce_mod = _plan_mod()
    requests = len(kv_lens)
    tasks = requests * splits
    wave_clusters = 106
    wave_tasks = wave_clusters // m_tiles
    kv = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    bounds = None
    bounds_list = None
    if helix:
        bounds_list = [
            max(1, kv_len - (query_len - 1 - q) * 37)
            for kv_len in kv_lens
            for q in range(query_len)
        ]
        bounds = torch.tensor(bounds_list, dtype=torch.int32, device="cuda")
    tiles = reduce_mod.request_kv_tiles(kv_lens, query_len, bounds_list)
    cap = 640
    expected_tiles, expected_begins, expected_order = reduce_mod.plan_packed_tasks_lpt(
        tiles, tasks, wave_tasks, m_tiles, wave_clusters, cap, one_wave=helix
    )
    plan = torch.full(
        (reduce_mod.packed_plan_length(requests, tasks),),
        -7,
        dtype=torch.int32,
        device="cuda",
    )
    reduce_mod.launch_packed_plan(
        kv,
        plan,
        num_requests=requests,
        num_splits=splits,
        wave_tasks=wave_tasks,
        max_tiles_per_task=cap,
        query_len=query_len,
        helix_kv_bounds=bounds,
        lpt_m_tiles=m_tiles,
        lpt_wave_clusters=wave_clusters,
    )
    torch.cuda.synchronize()
    plan = plan.tolist()
    assert plan[0] == expected_tiles
    assert plan[1] == expected_begins[-1]
    assert plan[2 : 3 + requests] == expected_begins
    tables = 3 + requests
    request_of, split_of, count_of = (
        plan[tables : tables + tasks],
        plan[tables + tasks : tables + 2 * tasks],
        plan[tables + 2 * tasks : tables + 3 * tasks],
    )
    for task, (request, split) in enumerate(expected_order):
        assert (request_of[task], split_of[task]) == (request, split)
        assert count_of[task] == expected_begins[request + 1] - expected_begins[request]
    for task in range(expected_begins[-1], tasks):
        assert request_of[task] == requests - 1 and split_of[task] == 1 << 15


def test_request_kv_tiles_follow_helix_bounds() -> None:
    tiles = _plan_mod().request_kv_tiles
    assert tiles([1, 256, 257, 0], 1) == [1, 1, 2, 1]
    # Helix: the last query token's bound of each request decides.
    assert tiles([5000, 5000], 2, [100, 512, 5000, 1000]) == [2, 4]


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 7) if torch.cuda.is_available() else True,
    reason="the CuTe DSL reducer is compiled for Rubin SM107",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_stats", [False, True])
@pytest.mark.parametrize(
    "kv_tiles,splits,wave_tasks,num_heads,query_len",
    [
        ([128, 128, 128, 128], 6, 17, 96, 8),
        ([4096], 16, 106, 12, 8),
        ([3, 300, 3], 3, 106, 6, 6),
        ([7, 1024], 2, 26, 128, 4),
        ([1, 1, 1, 1, 1], 1, 17, 96, 1),
        # Skewed batches use an uneven number of tasks per request, and
        # requests that end up with one task are stats-only here.
        ([1, 79, 3, 176], 6, 17, 96, 8),
        ([128] * 15 + [1024], 3, 106, 12, 8),
        # Many single-task requests: stats-only rows across many reducer CTAs.
        ([9] * 40, 2, 17, 96, 8),
    ],
)
def test_packed_split_reduce_matches_reference_and_replays(
    dtype: torch.dtype,
    with_stats: bool,
    kv_tiles: list[int],
    splits: int,
    wave_tasks: int,
    num_heads: int,
    query_len: int,
) -> None:
    reduce_mod = _plan_mod()
    torch.manual_seed(7)
    requests = len(kv_tiles)
    valid_m = num_heads * query_len
    physical_m = (valid_m + 127) // 128 * 128
    tasks = splits * requests
    # KV lengths just below the tile boundary; the kernels' plan is the mirror's plan.
    kv_lens = torch.tensor([t * 256 - 3 for t in kv_tiles], dtype=torch.int32, device="cuda")
    tiles_per_task, begins = reduce_mod.plan_packed_tasks(kv_tiles, tasks, wave_tasks)
    plan = torch.empty(
        reduce_mod.packed_plan_length(requests, tasks), dtype=torch.int32, device="cuda"
    )
    reduce_mod.launch_packed_plan(
        kv_lens,
        plan,
        num_requests=requests,
        num_splits=splits,
        wave_tasks=wave_tasks,
        max_tiles_per_task=max(kv_tiles),
        query_len=query_len,
    )
    torch.cuda.synchronize()
    assert plan[0].item() == tiles_per_task and plan[2 : 3 + requests].tolist() == begins
    assert reduce_mod.request_kv_tiles(kv_lens.tolist(), query_len) == kv_tiles
    partial = torch.randn((tasks, physical_m, 512), device="cuda")
    maxima = torch.randn((tasks, physical_m), device="cuda") * 12
    sums = torch.rand((tasks, physical_m), device="cuda") * 16 + 0.5
    # Each request's first task is empty for every row; row 0 of request 0 is
    # empty everywhere; unplanned task slots hold garbage that must be ignored.
    for begin in begins[:-1]:
        sums[begin] = 0
    sums[begins[0] : begins[1], 0] = 0
    maxima.masked_fill_(sums == 0, -torch.inf)
    partial.masked_fill_((sums == 0).unsqueeze(-1), torch.nan)
    sums[begins[-1] :] = 3.0
    maxima[begins[-1] :] = 1e9
    partial[begins[-1] :] = torch.nan
    q_scale = torch.tensor(1.25, device="cuda")
    kv_scale = torch.tensor(0.75, device="cuda")
    sm_scale, output_scale = 0.1, 0.25
    output = torch.empty((requests * valid_m, 512), dtype=dtype, device="cuda")
    stats = torch.empty((requests * valid_m, 2), device="cuda") if with_stats else None
    single_task = torch.tensor(
        [end - begin == 1 for begin, end in zip(begins, begins[1:])], device="cuda"
    ).repeat_interleave(valid_m)
    direct_output = torch.randn((requests * valid_m, 512), device="cuda").to(dtype)

    def launch() -> None:
        # Rows of single-task requests were stored by the decode kernel itself.
        output.copy_(direct_output)
        reduce_mod.reduce_packed_splits(
            partial,
            maxima,
            sums,
            output,
            q_scale,
            kv_scale,
            num_requests=requests,
            num_splits=splits,
            valid_m=valid_m,
            physical_m=physical_m,
            sm_scale=sm_scale,
            output_scale=output_scale,
            plan=plan,
            stats=stats,
            sm_count=212,
            use_pdl=False,
        )

    def check() -> None:
        scale = sm_scale * math.log2(math.e) / (q_scale.double() * kv_scale.double())
        expected_rows = []
        max_rows = []
        sum_rows = []
        for begin, end in zip(begins, begins[1:]):
            row_sum = sums[begin:end, :valid_m].double()
            active = row_sum != 0
            row_max = maxima[begin:end, :valid_m].double() * scale
            global_max = row_max.masked_fill(~active, -torch.inf).amax(0)
            weights = torch.where(active, torch.exp2(row_max - global_max), 0.0)
            values = partial[begin:end, :valid_m].double()
            numerator = (
                torch.where(active.unsqueeze(-1), values, 0.0) * weights.unsqueeze(-1)
            ).sum(0)
            denominator = (row_sum * weights).sum(0)
            expected_rows.append(
                torch.where(
                    denominator.unsqueeze(-1) != 0,
                    numerator / denominator.unsqueeze(-1) * output_scale,
                    0.0,
                )
            )
            max_rows.append(global_max)
            sum_rows.append(denominator)
        expected = torch.cat(expected_rows).to(dtype)
        expected = torch.where(single_task.unsqueeze(-1), direct_output, expected)
        global_max = torch.cat(max_rows)
        denominator = torch.cat(sum_rows)
        tolerance = torch.finfo(dtype).eps
        torch.testing.assert_close(output, expected, rtol=4 * tolerance, atol=tolerance)
        if stats is not None:
            expected_max = torch.where(denominator != 0, global_max / math.log2(math.e), -torch.inf)
            torch.testing.assert_close(
                stats[:, 0], expected_max.reshape(-1).float(), rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(
                stats[:, 1], denominator.reshape(-1).float(), rtol=2e-5, atol=2e-6
            )

    launch()
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    # Replays follow the live scales and payloads at the same addresses.
    q_scale.fill_(0.5)
    kv_scale.fill_(2.0)
    partial.mul_(0.75)
    sums.mul_(1.5)
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    check()
