# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed FP4 MLA decode: row layout policy, the CuTe DSL split reducer and the step metadata written ahead."""

import functools
import importlib
import math
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.attention.backends.fp4_mla.cache_update as fp4_mla_cache_update
import tensorrt_llm._torch.attention.backends.fp4_mla.config as fp4_mla_config
import tensorrt_llm._torch.attention.backends.fp4_mla.metadata as fp4_mla_metadata
import tensorrt_llm._torch.attention.backends.fp4_mla.packed_plan as fp4_mla_packed_plan
import tensorrt_llm._torch.attention.backends.fp4_mla.state as fp4_mla_state_module
from tensorrt_llm._torch.attention.backends.fp4_mla.state import Fp4MlaState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires an SM100-or-newer CUDA device",
)


@pytest.mark.parametrize(
    "num_heads,query_len,expected",
    [
        (96, 8, 768),
        (12, 8, 128),
        (6, 6, 128),
        (128, 4, 512),
        (128, 8, 1024),
        (128, 9, 0),
        (0, 1, 0),
    ],
)
def test_packed_rows_per_seq_pads_each_request_to_128_rows(
    monkeypatch, num_heads: int, query_len: int, expected: int
) -> None:
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_PACKED_Q_ENV, "1")
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, "0")
    monkeypatch.setattr(fp4_mla_config, "_fp4_mla_attention_backend", lambda: "cutedsl")
    assert fp4_mla_config._fp4_mla_packed_rows_per_seq(num_heads, query_len) == expected
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_PACKED_Q_ENV, "0")
    assert fp4_mla_config._fp4_mla_packed_rows_per_seq(num_heads, query_len) == 0


def test_packed_split_count_fills_one_cluster_wave(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_FP4_MLA_SPLIT_KV", raising=False)
    monkeypatch.setattr(fp4_mla_packed_plan, "_get_sm_count", lambda device: 212)
    device = torch.device("cuda")
    select = fp4_mla_packed_plan._select_packed_split_count
    # helix16 B4 MTP7 (6 m-tiles): two waves = 34 tasks / 4 requests + 1 rounding slot -> 10.
    assert select(4, 6, device, max_kv_len=999424, is_cuda_graph=True) == 10
    # A single short request is capped by the KV length when it is known.
    assert select(1, 6, device, max_kv_len=512, is_cuda_graph=False) == 1
    assert select(1, 6, device, max_kv_len=999424, is_cuda_graph=True) == 16
    # Large batches keep a two-slot budget so one long request can still split.
    assert select(32, 6, device, max_kv_len=999424, is_cuda_graph=True) == 3
    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", "3")
    assert select(4, 6, device, max_kv_len=999424, is_cuda_graph=True) == 3


def _plan_mod():
    return importlib.import_module(
        "tensorrt_llm._torch.attention.backends.fp4_mla.fp4_mla_packed_reduce"
    )


@pytest.mark.parametrize(
    "tiles,capacity,wave_tasks,expected_tiles_per_task,expected_begins",
    [
        # Even batch under one wave (helix16 B4, 6 m-tiles -> 17 wave tasks): one wave of 32-tile tasks.
        ([128, 128, 128, 128], 24, 17, 32, [0, 4, 8, 12, 16]),
        # One long request among short ones: 16-tile tasks; the one task past the wave is a short tail.
        ([1, 79, 3, 176], 24, 17, 16, [0, 1, 6, 7, 18]),
        # Full machine, longest 1.5x above the wave share: the balanced per-slot load (136 tiles)
        # splits only the long request, so its two tasks and the 15 others fill one wave.
        ([128] * 15 + [200], 48, 17, 136, list(range(16)) + [17]),
        # Full machine, longest within 1.5x of the wave share: no split at all.
        ([128] * 15 + [130], 48, 17, 130, list(range(17))),
        # Strong skew in a full machine: twelve 92-tile tasks for the long request, short tails behind.
        ([128] * 15 + [1024], 48, 17, 92, list(range(0, 32, 2)) + [42]),
        # Strong skew in a small batch: 89-tile tasks, 12 for the long request, fill one wave.
        ([128, 128, 128, 1024], 24, 17, 89, [0, 2, 4, 6, 18]),
        # A single request fills one wave with 16 tasks; a short one takes four 2-tile tasks.
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
    plan = functools.partial(_plan_mod().plan_packed_tasks, wave_plan=True)
    # helix16 bs16 (6 m-tiles, 17 wave tasks), per-rank KV skewed 3.5x: the Helix rule targets one
    # wave of 28-tile tasks (21 tasks, a second wave); no candidate beats it by the margin.
    tiles = [16] * 12 + [40, 48, 56, 56]
    rule = (28, list(range(13)) + [14, 16, 18, 20])
    assert plan(tiles, 48, 17, one_wave=True, wave_plan=False) == rule
    assert plan(tiles, 48, 17, one_wave=True) == rule
    assert plan(tiles, 48, 17, one_wave=True, m_tiles=6, wave_clusters=106) == rule
    # Even Helix batches are unchanged.
    assert plan([128] * 4, 24, 17, one_wave=True) == plan([128] * 4, 24, 17)


def test_plan_packed_tasks_simulates_the_waves() -> None:
    reduce_mod = _plan_mod()
    plan = functools.partial(reduce_mod.plan_packed_tasks, wave_plan=True)
    # Skewed request lengths exercise the task-rounding overhead across waves.
    tiles = [1848, 1250, 977, 743, 586, 430, 313, 196, 106]
    assert plan(tiles, 9 * 9, 35) == (188, [0, 10, 17, 23, 27, 31, 34, 36, 38, 39])
    # Nine equal requests: three 239-tile pieces fit one wave, but four 206-tile pieces finish
    # first: eight of the nine 97-tile tails share the first wave, the ninth follows one of them.
    assert plan([715] * 9, 9 * 9, 35) == (206, list(range(0, 37, 4)))
    simulate = reduce_mod.plan_simulated_makespan
    cost = reduce_mod.PLAN_TASK_COST_TILES
    # 36 equal 179-tile tasks on 35 slots: the last one waits for a whole task.
    assert simulate([716] * 9, 179, 35) == 2 * (179 + cost)
    # The 36th task (a 178-tile tail) starts when the first tail frees its slot.
    assert simulate([715] * 9, 179, 35) == (178 + cost) + (178 + cost)
    # 2-tile tails free their slots early, so a spilled tail does not extend the makespan.
    assert simulate([181] * 18, 179, 35) == 179 + cost
    assert simulate([100, 300], 300, 35) == 300 + cost
    # dep16 16 x 1094 tiles, 5 clusters per task on 106 cluster slots: 16 full 883-tile tasks
    # (80 clusters) leave 26 slots to the 211-tile tails; 438-tile tasks take three rounds.
    assert simulate([1094] * 16, 883, 21, m_tiles=5, wave_clusters=106) == 883 + cost
    assert simulate([1094] * 16, 438, 21, m_tiles=5, wave_clusters=106) == 910


@functools.cache
def _device_plan_candidates():
    """A launcher of ``_plan_candidate_tiles`` for every candidate index (one thread each)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from tensorrt_llm._torch.attention.backends.fp4_mla.cute_dsl_utils import (
        _compile_cutedsl,
        _current_cu_stream,
    )

    reduce_mod = _plan_mod()
    count = reduce_mod.PLAN_CANDIDATES

    @cute.kernel
    def candidates_kernel(
        out: cute.Tensor,
        rule_tiles: cutlass.Int32,
        total_tiles: cutlass.Int32,
        longest: cutlass.Int32,
        num_requests: cutlass.Int32,
        m_tiles: cutlass.Int32,
        slots: cutlass.Int32,
        max_tiles_per_task: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx < cutlass.Int32(count):
            out[tidx] = reduce_mod._plan_candidate_tiles(
                tidx,
                rule_tiles,
                total_tiles,
                longest,
                num_requests,
                m_tiles,
                slots,
                max_tiles_per_task,
            )

    @cute.jit
    def candidates_host(
        out_ptr: cute.Pointer,
        stream: cuda.CUstream,
        rule_tiles: cutlass.Int32,
        total_tiles: cutlass.Int32,
        longest: cutlass.Int32,
        num_requests: cutlass.Int32,
        m_tiles: cutlass.Int32,
        slots: cutlass.Int32,
        max_tiles_per_task: cutlass.Int32,
    ):
        out = cute.make_tensor(out_ptr, cute.make_layout((count,), stride=(1,)))
        candidates_kernel(
            out, rule_tiles, total_tiles, longest, num_requests, m_tiles, slots, max_tiles_per_task
        ).launch(grid=(1, 1, 1), block=((count + 31) // 32 * 32, 1, 1), stream=stream)

    compiled = None

    def run(request_tiles, rule_tiles, m_tiles, slots, max_tiles_per_task):
        nonlocal compiled
        out = torch.full((count,), -1, dtype=torch.int32, device="cuda")
        ptr = make_ptr(cutlass.Int32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
        scalars = [
            cutlass.Int32(v)
            for v in (
                rule_tiles,
                sum(request_tiles),
                max(request_tiles),
                len(request_tiles),
                m_tiles,
                slots,
                max_tiles_per_task,
            )
        ]
        if compiled is None:
            compiled = _compile_cutedsl(candidates_host, ptr, _current_cu_stream(), *scalars)
        compiled(ptr, _current_cu_stream(), *scalars)
        return out.tolist()

    return run


def test_plan_candidates_device_match_the_host_mirror() -> None:
    """``_plan_candidate_tiles`` (device) and ``plan_candidate_lengths`` (host) give the same
    task lengths for random batches within the simulated limits."""
    reduce_mod = _plan_mod()
    run = _device_plan_candidates()
    generator = torch.Generator().manual_seed(29)

    def draw(low, high):
        return int(torch.randint(low, high + 1, (1,), generator=generator))

    for _ in range(64):
        num_requests = draw(1, reduce_mod.PLAN_SIM_MAX_REQUESTS)
        request_tiles = [draw(1, 4096) for _ in range(num_requests)]
        m_tiles = draw(1, 8)
        slots = draw(1, reduce_mod.PLAN_SIM_MAX_SLOTS)
        wave_tasks = max(1, slots // m_tiles)
        rule_tiles = draw(1, max(request_tiles))
        max_tiles_per_task = draw(1, 1024)
        expected = reduce_mod.plan_candidate_lengths(
            request_tiles,
            rule_tiles,
            wave_tasks,
            max_tiles_per_task,
            m_tiles=m_tiles,
            wave_clusters=slots,
        )
        got = run(request_tiles, rule_tiles, m_tiles, slots, max_tiles_per_task)
        assert got == expected, (request_tiles, rule_tiles, m_tiles, slots, max_tiles_per_task)


def test_plan_load_candidates_follow_the_balanced_slot_load() -> None:
    reduce_mod = _plan_mod()
    # 2120 tiles + 12 per slot and per request over 17 slots: 136 tiles per slot, then its
    # 1/2..1/4 and 4% longer variants; the even-batch closed form of the 200-tile request.
    tiles = [128] * 15 + [200]
    assert reduce_mod.plan_load_candidates(tiles, 17) == [136, 141, 68, 70, 45, 47, 34, 35, 199]
    # Clamped to the minimum task length, the longest request and the runtime profile.
    assert reduce_mod.plan_load_candidates([3, 5], 106) == [2] * reduce_mod.PLAN_LOAD_CANDIDATES
    assert max(reduce_mod.plan_load_candidates([684] * 16, 21, max_tiles_per_task=300)) == 300
    # dep16 B16 MTP5 at 280K: 16 requests x 5 cluster streams on 106 cluster slots.
    assert reduce_mod.plan_load_candidates([1094] * 16, 21, m_tiles=5, wave_clusters=106) == [
        835,
        868,
        417,
        434,
        278,
        289,
        208,
        217,
        883,
    ]
    candidates = reduce_mod.plan_candidate_lengths(tiles, 100, 17)
    assert len(candidates) == reduce_mod.PLAN_CANDIDATES
    assert candidates[0] == 100 and candidates[1:3] == [200, 100]


def test_plan_packed_tasks_takes_the_shortest_simulated_candidate() -> None:
    """The rule's length unless a candidate beats it by the margin, then the shortest one."""
    import random

    rng = random.Random(4321)
    reduce_mod = _plan_mod()
    for _ in range(500):
        requests = rng.randint(1, 40)
        tiles = [rng.choice([1, 3, 25, 64, 128, 300, 704, 1024, 4096]) for _ in range(requests)]
        capacity = requests * rng.choice([2, 4, 8, 16])
        m_tiles = rng.choice([1, 3, 5, 6])
        wave_tasks = 106 // m_tiles
        geometry = dict(m_tiles=m_tiles, wave_clusters=106)
        tiles_per_task, begins = reduce_mod.plan_packed_tasks(
            tiles, capacity, wave_tasks, wave_plan=True, **geometry
        )
        rule, _ = reduce_mod.plan_packed_tasks(tiles, capacity, wave_tasks, wave_plan=False)
        feasible = [
            candidate
            for candidate in reduce_mod.plan_candidate_lengths(tiles, rule, wave_tasks, **geometry)
            if sum(math.ceil(t / candidate) for t in tiles)
            <= min(capacity, reduce_mod.PLAN_SIM_MAX_TASKS)
        ]
        assert begins[-1] <= capacity
        if requests >= capacity or not feasible:
            assert tiles_per_task == rule
            continue

        def simulate(length: int) -> int:
            return reduce_mod.plan_simulated_makespan(tiles, length, wave_tasks, **geometry)

        baseline = simulate(feasible[0])
        best = min(simulate(candidate) for candidate in feasible)
        limit = baseline - max(
            baseline // reduce_mod.PLAN_MIN_GAIN_DIVISOR, reduce_mod.PLAN_MIN_GAIN_TILES
        )
        if best <= limit:
            assert simulate(tiles_per_task) == best
        else:
            assert tiles_per_task == feasible[0]


@pytest.mark.parametrize(
    "tiles,m_tiles,one_wave,expected",
    [
        # dep16 B16 MTP5 (5 m-tiles) at 280K: 883-tile tasks, 2 per request (makespan 895 tiles
        # against 910 for the 438-tile tasks a per-task slot model picks).
        ([1094] * 16, 5, False, 883),
        ([1094] * 16, 6, False, 1005),
        ([1094] * 8, 6, False, 502),
        # helix8 B8 / B20 and helix16 B4 per-rank KV (35K / 17.5K).
        ([137] * 8, 5, True, 58),
        ([137] * 20, 6, True, 58),
        ([69] * 4, 5, True, 14),
    ],
)
def test_plan_packed_tasks_simulates_cluster_slots(
    tiles: list[int], m_tiles: int, one_wave: bool, expected: int
) -> None:
    """Each task runs m_tiles clusters on the wave's 2-CTA cluster slots (106 on Rubin)."""
    tiles_per_task, _ = _plan_mod().plan_packed_tasks(
        tiles,
        8 * len(tiles),
        106 // m_tiles,
        one_wave=one_wave,
        wave_plan=True,
        m_tiles=m_tiles,
        wave_clusters=106,
    )
    assert tiles_per_task == expected


def test_plan_task_order_runs_full_tasks_then_longest_tails() -> None:
    order = functools.partial(_plan_mod().plan_task_order, wave_plan=True)
    # Tails: request 0 -> 4 tiles, request 1 -> 10 (a full task), request 2 -> 3, request 3 -> 4.
    full = [(0, 0), (0, 1), (1, 0), (1, 1), (3, 0)]
    tails = [(1, 2), (0, 2), (3, 1), (2, 0)]
    assert order([24, 30, 3, 14], 10) == full + tails
    # One task per request: longest first, ties in request order.
    assert order([5, 9, 5, 2], 10) == [(1, 0), (0, 0), (2, 0), (3, 0)]


def test_plan_without_wave_plan_keeps_the_rule_and_request_order() -> None:
    reduce_mod = _plan_mod()
    plan = reduce_mod.plan_packed_tasks
    # TRTLLM_FP4_MLA_PACKED_WAVE_PLAN=0: the rule's length, as before the wave simulation.
    assert plan([1, 79, 3, 176], 24, 17, wave_plan=False) == (13, [0, 1, 8, 9, 23])
    assert plan([128] * 15 + [1024], 48, 17, wave_plan=False) == (
        86,
        list(range(0, 32, 2)) + [42],
    )
    order = reduce_mod.plan_task_order([24, 30, 3, 14], 10, wave_plan=False)
    assert order == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (3, 0), (3, 1)]


def test_plan_packed_tasks_honours_the_runtime_profile_cap() -> None:
    plan = functools.partial(_plan_mod().plan_packed_tasks, wave_plan=True)
    # An even full machine would not split, but the profile only covers 512 tiles.
    assert plan([1024] * 16, 48, 17, max_tiles_per_task=512) == (488, list(range(0, 49, 3)))
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


# Skewed request lengths for planner coverage.
_DEP16_RANK_KV_LENS = [473000, 320000, 250000, 190000, 150000, 110000, 80000, 50000, 27000]


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 7) if torch.cuda.is_available() else True,
    reason="the CuTe DSL plan kernel is compiled for Rubin SM107",
)
@pytest.mark.parametrize(
    "kv_lens,splits,wave_tasks,query_len,helix,wave_plan",
    [
        ([32768] * 15 + [262144], 4, 17, 8, False, True),
        ([131, 20000, 600, 45000], 10, 17, 8, False, True),
        ([131, 20000, 600, 45000], 10, 17, 8, True, True),
        ([300, 70000, 2200], 16, 106, 1, False, True),
        ([5, 5, 5], 2, 106, 4, False, True),
        # DEP16 rank shapes (3 m-tiles, 35 wave tasks): the simulated task length.
        (_DEP16_RANK_KV_LENS, 9, 35, 4, False, True),
        ([183000] * 9, 9, 35, 4, False, True),
        ([103000] * 16, 6, 35, 4, False, True),
        # Balanced per-slot load candidates: dep16 B8 MTP7 at 180K, dep16 B16 MTP5 at 175K.
        ([180000] * 8, 8, 17, 8, False, True),
        ([175000] * 16, 8, 21, 6, False, True),
        # dep16 B16 MTP5 at 280K, helix8 B20 MTP7 at 35K per rank.
        ([280000] * 16, 8, 21, 6, False, True),
        ([35000] * 20, 8, 17, 8, True, True),
        # Two cached requests per lane; larger batches take the existing fallback.
        ([32768] * 31 + [262144], 2, 35, 4, False, True),
        ([32768] * 32 + [262144], 2, 35, 4, False, True),
        ([32768] * 63 + [262144], 2, 35, 4, False, True),
        ([32768] * 64 + [262144], 2, 35, 4, False, True),
        # TRTLLM_FP4_MLA_PACKED_WAVE_PLAN=0, off and on Helix: the rule's length, request order.
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
    # Rubin: 106 2-CTA cluster slots, wave_tasks of them per m_tiles-cluster task.
    geometry = dict(m_tiles=106 // wave_tasks, wave_clusters=106)
    expected_tiles, expected_begins = reduce_mod.plan_packed_tasks(
        tiles, tasks, wave_tasks, cap, one_wave=helix, wave_plan=wave_plan, **geometry
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
        **geometry,
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


def test_uniform_generation_lengths_fuse_the_packed_plan(monkeypatch) -> None:
    corrected = torch.zeros(2, dtype=torch.int32, device="cuda")
    generation_lens = torch.zeros(2, dtype=torch.int32, device="cuda")
    plan = torch.zeros(16, dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(
        num_contexts=0,
        num_seqs=2,
        kv_lens_cuda_runtime=torch.tensor([17, 25], dtype=torch.int32, device="cuda"),
        prompt_lens_cuda_runtime=torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
        is_cuda_graph=False,
        fp4_mla_state=SimpleNamespace(
            generation_kv_lens=corrected,
            generation_append_lens=generation_lens,
            generation_lengths_num_tokens=-1,
            generation_lengths_num_seqs=-1,
            generation_lengths_num_contexts=-1,
            generation_lengths_capture_recorded=False,
            packed_plan_key=None,
        ),
    )
    geometry = fp4_mla_packed_plan._PackedPlanGeometry(
        2, 4, 3, 17, 64, 6, 1024, None, plan, ("key",)
    )
    calls = []
    monkeypatch.setattr(
        fp4_mla_metadata,
        "populate_fp4_mla_generation_lengths",
        lambda *args, **kwargs: calls.append(("lengths", kwargs)),
    )
    monkeypatch.setattr(
        _plan_mod(), "launch_packed_plan", lambda *args, **kwargs: calls.append(("plan", kwargs))
    )
    uniform = fp4_mla_metadata._fp4_mla_uniform_generation_lengths

    # First FP4 MLA kernel of the step: one fused launch writes lengths and plan.
    uniform(metadata, 8, 2, plan_geometry=geometry)
    assert [name for name, _ in calls] == ["plan"]
    assert calls[0][1]["generation_lengths"][3] == 4
    assert metadata.fp4_mla_state.packed_plan_key == ("key",)
    # The other layers of the step launch nothing.
    uniform(metadata, 8, 2, plan_geometry=geometry)
    uniform(metadata, 8, 2)
    assert len(calls) == 1
    # A decode planning on other bounds rewrites the plan alone.
    uniform(metadata, 8, 2, plan_geometry=geometry._replace(key=("other",)))
    assert len(calls) == 2 and calls[1][0] == "plan" and calls[1][1]["generation_lengths"] is None
    # A new step whose first caller cannot plan keeps the Triton lengths kernel,
    # and the decode then plans alone.
    metadata.fp4_mla_state.generation_lengths_num_tokens = -1
    metadata.fp4_mla_state.packed_plan_key = None
    uniform(metadata, 8, 2)
    uniform(metadata, 8, 2, plan_geometry=geometry)
    assert [name for name, _ in calls[2:]] == ["lengths", "plan"]
    assert calls[3][1]["generation_lengths"] is None
    with pytest.raises(RuntimeError):
        uniform(metadata, 12, 3, plan_geometry=geometry)


def test_packed_plan_wave_plan_env_rolls_back(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        _plan_mod(), "launch_packed_plan", lambda *args, **kwargs: calls.append(kwargs["wave_plan"])
    )
    geometry = fp4_mla_packed_plan._PackedPlanGeometry(
        2, 4, 3, 17, 64, 6, 1024, None, None, ("key",)
    )
    kv_lens = torch.zeros(2, dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(fp4_mla_state=SimpleNamespace(packed_plan_key=None))
    monkeypatch.delenv(fp4_mla_config.FP4_MLA_PACKED_WAVE_PLAN_ENV, raising=False)
    fp4_mla_packed_plan._launch_fp4_mla_packed_plan(geometry, kv_lens, metadata)
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_PACKED_WAVE_PLAN_ENV, "0")
    fp4_mla_packed_plan._launch_fp4_mla_packed_plan(geometry, kv_lens, metadata)
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_PACKED_WAVE_PLAN_ENV, "1")
    fp4_mla_packed_plan._launch_fp4_mla_packed_plan(geometry, kv_lens, metadata)
    assert calls == [False, False, True]


def test_packed_plan_geometry_is_shared_by_cache_update_and_decode(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_FP4_MLA_SPLIT_KV", raising=False)
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_PACKED_Q_ENV, "1")
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, "0")
    monkeypatch.setattr(fp4_mla_config, "_fp4_mla_attention_backend", lambda: "cutedsl")
    monkeypatch.setattr(fp4_mla_packed_plan, "_get_sm_count", lambda device: 212)
    # An indexed device, as the kernels' tensors carry: the workspace helper
    # compares devices exactly before reusing a buffer.
    device = torch.device("cuda", torch.cuda.current_device())
    metadata = SimpleNamespace(
        num_contexts=0,
        num_seqs=4,
        page_size=256,
        is_cuda_graph=True,
        fp4_mla_state=SimpleNamespace(
            device_page_table=True,
            page_table_stride=3904,
            generation_kv_lens=torch.zeros(4, dtype=torch.int32, device=device),
            workspaces={},
        ),
    )
    geometry = fp4_mla_packed_plan._fp4_mla_packed_plan_geometry(
        metadata, num_heads=96, query_len=8, num_requests=4, device=device, helix_kv_bounds=None
    )
    # helix16 B4 MTP7: 6 m-tiles, 17 tasks per wave, 10 slots per request.
    assert (geometry.splits, geometry.wave_tasks, geometry.task_capacity) == (10, 17, 40)
    # The wave plan simulates the 6 clusters of each task on 106 2-CTA cluster slots.
    assert (geometry.m_tiles, geometry.wave_clusters) == (6, 106)
    # The plan lives in the metadata workspace (a view of the stored buffer).
    assert (
        geometry.plan.data_ptr()
        == metadata.fp4_mla_state.workspaces["_fp4_mla_packed_plan"].data_ptr()
    )
    assert geometry.plan.numel() == _plan_mod().packed_plan_length(4, 40)
    again = fp4_mla_packed_plan._fp4_mla_packed_plan_geometry(
        metadata, num_heads=96, query_len=8, num_requests=4, device=device, helix_kv_bounds=None
    )
    assert again.key == geometry.key
    bounds = torch.zeros(32, dtype=torch.int32, device=device)
    with_bounds = fp4_mla_packed_plan._fp4_mla_packed_plan_geometry(
        metadata, num_heads=96, query_len=8, num_requests=4, device=device, helix_kv_bounds=bounds
    )
    assert with_bounds.key != geometry.key
    # 434-tile tasks fit the shared-memory page-plan profile, which caps the plan in both
    # cases; only Helix ranks also run its kernel, the others the next bucketed profile.
    module = fp4_mla_config._fp4_mla_cutedsl_kernel_module()
    page_plan_kv = module.SMEM_P4_PAGE_PLAN_PROFILE_KV
    assert geometry.profile_tiles == with_bounds.profile_tiles == page_plan_kv // 256
    assert geometry.split_kv_len == with_bounds.split_kv_len == 434 * 256
    assert with_bounds.use_smem_page_plan and not geometry.use_smem_page_plan
    assert module._select_runtime_kv_profile(geometry.split_kv_len) == page_plan_kv
    assert module._select_runtime_kv_profile(geometry.split_kv_len, False) > page_plan_kv
    # Too many rows for the packed layout: the legacy path, no geometry.
    assert (
        fp4_mla_packed_plan._fp4_mla_packed_plan_geometry(
            metadata,
            num_heads=128,
            query_len=9,
            num_requests=4,
            device=device,
            helix_kv_bounds=None,
        )
        is None
    )
    # One task per request needs no plan.
    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", "1")
    single = fp4_mla_packed_plan._fp4_mla_packed_plan_geometry(
        metadata, num_heads=96, query_len=8, num_requests=4, device=device, helix_kv_bounds=None
    )
    assert single.splits == 1 and single.key is None and single.plan is None


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
        # Many single-task requests: their stats chunks spread over the reducer CTAs.
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


# ----------------------------------------------------------------------------------------------------------------
# A generation-only step's metadata written ahead of its first FP4 MLA layer
# (Fp4MlaState.launch_generation_metadata_ahead / join_generation_metadata)
# ----------------------------------------------------------------------------------------------------------------


def _ahead_metadata(num_contexts=0, num_seqs=2, num_tokens=8, device_page_table=True):
    device = torch.device("cuda", torch.cuda.current_device())
    return SimpleNamespace(
        num_contexts=num_contexts,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        helix_position_offsets=None,
        helix_is_inactive_rank=None,
        fp4_mla_state=Fp4MlaState(
            device_page_table=device_page_table,
            generation_kv_lens=torch.zeros(4, dtype=torch.int32, device=device),
        ),
    )


def test_generation_metadata_ahead_runs_on_its_stream_and_the_layer_joins(monkeypatch) -> None:
    """Planning stays on the current stream, the length / plan and page-table launches run on the given stream after
    the work queued before, and the first generation cache update orders its stream after them."""
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    metadata = _ahead_metadata()
    state = metadata.fp4_mla_state
    calls = []
    geometry = object()
    written = torch.zeros(1, dtype=torch.int64, device="cuda")

    def plan_geometry(md, **kwargs):
        calls.append(("geometry", torch.cuda.current_stream(), kwargs))
        return geometry

    def lengths(md, num_tokens, num_gen, plan_geometry=None):
        calls.append(("lengths", torch.cuda.current_stream(), (num_tokens, num_gen, plan_geometry)))
        # A slow writer: the reader below sees its value only if it was ordered after it.
        torch.cuda._sleep(20_000_000)
        written.fill_(7)
        return state.generation_kv_lens[:num_gen], None

    def page_table(md, generation_kv_lens):
        calls.append(("page_table", torch.cuda.current_stream(), generation_kv_lens))

    monkeypatch.setattr(fp4_mla_state_module, "_fp4_mla_packed_plan_geometry", plan_geometry)
    monkeypatch.setattr(fp4_mla_state_module, "_fp4_mla_uniform_generation_lengths", lengths)
    monkeypatch.setattr(
        fp4_mla_state_module, "_materialize_fp4_mla_device_page_table_for_forward", page_table
    )

    assert state.launch_generation_metadata_ahead(metadata, num_heads=96, stream=side)
    assert [name for name, _, _ in calls] == ["geometry", "lengths", "page_table"]
    assert calls[0][1] == main
    assert calls[0][2] == dict(
        num_heads=96,
        query_len=4,
        num_requests=2,
        device=state.generation_kv_lens.device,
        helix_kv_bounds=None,
    )
    assert calls[1][1] == side and calls[1][2] == (8, 2, geometry)
    assert calls[2][1] == side
    assert calls[2][2].data_ptr() == state.generation_kv_lens.data_ptr()
    assert state.generation_metadata_ready is not None

    # The first FP4 MLA kernel of the step (the generation cache update) joins
    # before anything else; it rejects this empty batch right after.
    with pytest.raises(ValueError, match="at least one latent token"):
        fp4_mla_cache_update.scatter_fp4_mla_kv_cache(
            metadata,
            torch.empty(0, 576, dtype=torch.bfloat16, device="cuda"),
            0,
            token_offset=0,
            phase="generation",
            local_layer=0,
            v_head_dim=512,
        )
    assert state.generation_metadata_ready is None
    seen = written.clone()
    torch.cuda.synchronize()
    assert seen.item() == 7
    # Joined once: later joins are no-ops.
    state.join_generation_metadata()


@pytest.mark.parametrize(
    "num_contexts,num_seqs,num_tokens,device_page_table,num_heads",
    [
        (1, 3, 9, True, 96),  # a context request in the step
        (0, 3, 8, True, 96),  # a non-uniform generation batch
        (0, 0, 0, True, 96),  # no generation request
        (0, 2, 8, False, 96),  # no device page table
        (0, 2, 8, True, 0),  # no query heads
    ],
)
def test_generation_metadata_ahead_declines_other_steps(
    num_contexts, num_seqs, num_tokens, device_page_table, num_heads
) -> None:
    """Such steps keep writing the metadata in the first FP4 MLA layer: nothing is launched or left to join."""
    metadata = _ahead_metadata(num_contexts, num_seqs, num_tokens, device_page_table)
    state = metadata.fp4_mla_state
    assert not state.launch_generation_metadata_ahead(
        metadata, num_heads=num_heads, stream=torch.cuda.Stream()
    )
    assert state.generation_metadata_ready is None and state._generation_metadata_events is None


def _step_metadata(*, is_cuda_graph: bool, num_gen: int, query_len: int, max_pages: int):
    """A generation-only FP4 MLA step: shuffled block offsets in two pools (FP4 cache and HP ring), KV lengths across
    many pages and the overlap scheduler's stale prompt lengths."""
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device="cpu").manual_seed(3)
    max_seqs = num_gen + 2
    page_size = fp4_mla_config.FP4_MLA_TOKENS_PER_BLOCK

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    offsets = torch.randperm(2 * max_seqs * 2 * max_pages, generator=generator)
    block_offsets = (offsets.view(2, max_seqs, 2, max_pages) * 2).to(torch.int32).to(device)
    kv_lens = torch.randint(page_size, (max_pages - 4) * page_size, (num_gen,), generator=generator)
    pages = max_seqs * max_pages
    state = Fp4MlaState(
        generation_kv_lens=torch.full((max_seqs,), -1, dtype=torch.int32, device=device),
        generation_append_lens=torch.full((max_seqs,), -1, dtype=torch.int32, device=device),
        _paged_kv_indices=torch.full((pages,), -1, dtype=torch.int32, device=device),
        hp_page_indices=torch.full((pages,), -1, dtype=torch.int32, device=device),
        _paged_kv_indptr=torch.full((max_seqs + 1,), -1, dtype=torch.int32, device=device),
        paged_kv_indptr_decode=torch.full((max_seqs + 1,), -1, dtype=torch.int32, device=device),
        device_page_table=True,
        page_table_stride=max_pages,
        cache_pool_id=0,
        cache_page_index_scale=2,
        hp_pool_id=1,
        hp_page_index_scale=2,
        # Larger than any plan here, so that the geometry reuses it (also in a
        # graph capture) and its slots past the plan compare equal.
        workspaces={
            "_fp4_mla_packed_plan": torch.full((4096,), -1, dtype=torch.int32, device=device)
        },
    )
    return SimpleNamespace(
        num_contexts=0,
        num_seqs=num_gen,
        num_tokens=num_gen * query_len,
        num_ctx_tokens=0,
        page_size=page_size,
        is_cuda_graph=is_cuda_graph,
        kv_cache_block_offsets=block_offsets,
        kv_lens_cuda_runtime=i32(kv_lens.tolist()),
        prompt_lens_cuda_runtime=i32([1] * num_gen),
        helix_position_offsets=None,
        helix_is_inactive_rank=None,
        _helix_spec_tokens_valid=False,
        fp4_mla_state=state,
    )


def _first_layer_metadata(metadata, num_heads: int) -> None:
    """What the step's first FP4 MLA generation cache update writes (_scatter_fp4_mla_kv_cache_2d_generation)."""
    num_gen = metadata.num_seqs
    geometry = fp4_mla_packed_plan._fp4_mla_packed_plan_geometry(
        metadata,
        num_heads=num_heads,
        query_len=metadata.num_tokens // num_gen,
        num_requests=num_gen,
        device=metadata.fp4_mla_state.generation_kv_lens.device,
        helix_kv_bounds=None,
    )
    kv_lens, _ = fp4_mla_metadata._fp4_mla_uniform_generation_lengths(
        metadata, metadata.num_tokens, num_gen, plan_geometry=geometry
    )
    fp4_mla_metadata._materialize_fp4_mla_device_page_table_for_forward(metadata, kv_lens)
    return geometry


def _step_outputs(metadata) -> dict:
    state = metadata.fp4_mla_state
    return {
        "generation_kv_lens": state.generation_kv_lens,
        "generation_append_lens": state.generation_append_lens,
        "paged_kv_indices": state._paged_kv_indices,
        "hp_page_indices": state.hp_page_indices,
        "paged_kv_indptr": state._paged_kv_indptr,
        "paged_kv_indptr_decode": state.paged_kv_indptr_decode,
        "plan": state.workspaces["_fp4_mla_packed_plan"],
    }


def _assert_step_outputs_equal(got: dict, want: dict) -> None:
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name


def _no_layer_launch(monkeypatch) -> None:
    """The first layer finds the metadata current: any metadata launch fails."""

    def fail(*args, **kwargs):
        raise AssertionError("the first FP4 MLA layer relaunched the step metadata")

    monkeypatch.setattr(fp4_mla_metadata, "populate_fp4_mla_generation_lengths", fail)
    monkeypatch.setattr(fp4_mla_metadata, "_launch_fp4_mla_packed_plan", fail)
    monkeypatch.setattr(fp4_mla_metadata, "materialize_fp4_mla_device_page_table", fail)


@pytest.mark.parametrize("num_gen,query_len,num_heads", [(4, 8, 96), (1, 8, 96), (3, 4, 12)])
def test_generation_metadata_ahead_is_bitwise_eager_and_in_a_graph(
    monkeypatch, num_gen: int, query_len: int, num_heads: int
) -> None:
    """The metadata written ahead on a side stream is bit-identical to the first layer's, eager and in a captured
    and replayed CUDA graph, and the layer then launches no metadata kernel."""
    monkeypatch.delenv("TRTLLM_FP4_MLA_SPLIT_KV", raising=False)
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_PACKED_Q_ENV, "1")
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, "0")
    monkeypatch.setattr(fp4_mla_config, "_fp4_mla_attention_backend", lambda: "cutedsl")
    side = torch.cuda.Stream()
    shape = dict(num_gen=num_gen, query_len=query_len, max_pages=48)

    for is_cuda_graph in (False, True):
        reference = _step_metadata(is_cuda_graph=is_cuda_graph, **shape)
        geometry = _first_layer_metadata(reference, num_heads)
        assert geometry is not None and geometry.plan is not None, "the step must plan its splits"
        ahead = _step_metadata(is_cuda_graph=is_cuda_graph, **shape)
        with monkeypatch.context() as patch:
            assert ahead.fp4_mla_state.launch_generation_metadata_ahead(
                ahead, num_heads=num_heads, stream=side
            )
            ahead.fp4_mla_state.join_generation_metadata()
            _no_layer_launch(patch)
            _first_layer_metadata(ahead, num_heads)
        torch.cuda.synchronize()
        _assert_step_outputs_equal(_step_outputs(ahead), _step_outputs(reference))

    # A CUDA graph (its first eager step above compiled the kernels and sized the
    # plan workspace): the fork and join are captured with the step.
    captured = _step_metadata(is_cuda_graph=True, **shape)
    _first_layer_metadata(captured, num_heads)
    captured.fp4_mla_state.invalidate_generation_lengths()
    captured.fp4_mla_state.device_page_table_valid = False
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert captured.fp4_mla_state.launch_generation_metadata_ahead(
            captured, num_heads=num_heads, stream=side
        )
        captured.fp4_mla_state.join_generation_metadata()
        _first_layer_metadata(captured, num_heads)
    for step in range(3):
        # The next step's lengths at the same addresses; outputs reset to the
        # reference's initial contents.
        captured.kv_lens_cuda_runtime.add_(fp4_mla_config.FP4_MLA_TOKENS_PER_BLOCK // 2 + step)
        for tensor in _step_outputs(captured).values():
            tensor.fill_(-1)
        graph.replay()
        reference = _step_metadata(is_cuda_graph=True, **shape)
        reference.kv_lens_cuda_runtime.copy_(captured.kv_lens_cuda_runtime)
        _first_layer_metadata(reference, num_heads)
        torch.cuda.synchronize()
        _assert_step_outputs_equal(_step_outputs(captured), _step_outputs(reference))


@pytest.mark.parametrize("value,enabled", [(None, False), ("0", False), ("1", True)])
def test_wave_plan_requires_explicit_opt_in(monkeypatch, value: str | None, enabled: bool) -> None:
    if value is None:
        monkeypatch.delenv(fp4_mla_config.FP4_MLA_PACKED_WAVE_PLAN_ENV, raising=False)
    else:
        monkeypatch.setenv(fp4_mla_config.FP4_MLA_PACKED_WAVE_PLAN_ENV, value)
    assert fp4_mla_config._fp4_mla_packed_wave_plan_enabled() is enabled
    plan = _plan_mod()
    assert plan.plan_packed_tasks([1, 79, 3, 176], 24, 17) == (
        plan.plan_packed_tasks([1, 79, 3, 176], 24, 17, wave_plan=enabled)
    )
    assert plan.plan_task_order([24, 30, 3, 14], 10) == (
        plan.plan_task_order([24, 30, 3, 14], 10, wave_plan=enabled)
    )
