# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed FP4 MLA decode: row layout policy and the CuTe DSL split reducer."""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.attention_backend.fp4_mla as fp4_mla_backend

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
    layout = fp4_mla_backend.Fp4MlaVLayout
    rows = fp4_mla_backend._fp4_mla_packed_rows_per_seq
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_Q_ENV, "1")
    monkeypatch.setattr(fp4_mla_backend, "_fp4_mla_attention_backend", lambda: "cutedsl")
    assert rows(num_heads, query_len, layout.DUAL_COPY) == expected
    assert rows(num_heads, query_len, layout.PACKED_SINGLE_COPY) == expected
    # The fused-V module has no packed decode.
    assert rows(num_heads, query_len, layout.FUSED_V) == 0
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_Q_ENV, "0")
    assert rows(num_heads, query_len, layout.DUAL_COPY) == 0


def test_packed_split_count_fills_one_cluster_wave(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_FP4_MLA_SPLIT_KV", raising=False)
    monkeypatch.setattr(fp4_mla_backend, "_get_sm_count", lambda device: 212)
    device = torch.device("cuda")
    select = fp4_mla_backend._select_packed_split_count
    # helix16 B4 MTP7 (6 m-tiles): two waves = 34 tasks / 4 requests + 1 rounding slot -> 10.
    assert select(4, 6, device, max_kv_len=999424, is_cuda_graph=True) == 10
    # A single short request is capped by the KV length when it is known.
    assert select(1, 6, device, max_kv_len=512, is_cuda_graph=False) == 1
    # A lone request may take two waves of tasks (34 + 1 slots at 6 m-tiles); at one
    # m-tile that would be 213, capped by _FP4_MLA_PACKED_MAX_SPLITS.
    assert select(1, 6, device, max_kv_len=999424, is_cuda_graph=True) == 35
    assert select(1, 1, device, max_kv_len=999424, is_cuda_graph=True) == 128
    assert select(2, 1, device, max_kv_len=999424, is_cuda_graph=True) == 107
    # Large batches keep a two-slot budget so one long request can still split.
    assert select(32, 6, device, max_kv_len=999424, is_cuda_graph=True) == 3
    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", "3")
    assert select(4, 6, device, max_kv_len=999424, is_cuda_graph=True) == 3
    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", "500")
    assert select(4, 6, device, max_kv_len=999424, is_cuda_graph=True) == 128


def _plan_mod():
    return importlib.import_module(
        "tensorrt_llm._torch.attention_backend.fp4_mla.fp4_mla_packed_reduce"
    )


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
        # Strong skew in a small batch: 86-tile tasks beat the 71-tile fallback; one short tail spills.
        ([128, 128, 128, 1024], 24, 17, 86, [0, 2, 4, 6, 18]),
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
    # helix16 bs16 (6 m-tiles, 17 wave tasks), per-rank KV skewed 3.5x: off Helix the
    # simulated length cuts 14-tile pieces (39 tasks); a Helix rank keeps one wave of 28-tile tasks.
    tiles = [16] * 12 + [40, 48, 56, 56]
    assert plan(tiles, 48, 17) == (14, list(range(0, 26, 2)) + [27, 31, 35, 39])
    assert plan(tiles, 48, 17, one_wave=True) == (28, list(range(13)) + [14, 16, 18, 20])
    # Even Helix batches are unchanged.
    assert plan([128] * 4, 24, 17, one_wave=True) == plan([128] * 4, 24, 17)


def test_plan_packed_tasks_simulates_the_waves() -> None:
    reduce_mod = _plan_mod()
    plan = reduce_mod.plan_packed_tasks
    # DEP16 rank on Rubin (3 m-tiles, 35 wave tasks), production-like skew: the two-wave target
    # (93 tiles) rounds every request up to 74 tasks, a third wave; 185-tile tasks run one wave of
    # full tasks with the short tails backfilling.
    tiles = [1848, 1250, 977, 743, 586, 430, 313, 196, 106]
    assert plan(tiles, 9 * 9, 35) == (185, [0, 10, 17, 23, 28, 32, 35, 37, 39, 40])
    # Nine equal requests: 36 one-wave tasks spill one task; seven 103-tile pieces fit two waves.
    assert plan([715] * 9, 9 * 9, 35) == (103, list(range(0, 64, 7)))
    simulate = reduce_mod.plan_simulated_makespan
    cost = reduce_mod.PLAN_TASK_COST_TILES
    # 36 equal 179-tile tasks on 35 slots: the last one waits for a whole task.
    assert simulate([716] * 9, 179, 35) == 2 * (179 + cost)
    # The 36th task (a 178-tile tail) starts when the first tail frees its slot.
    assert simulate([715] * 9, 179, 35) == (178 + cost) + (178 + cost)
    # 2-tile tails free their slots early, so a spilled tail does not extend the makespan.
    assert simulate([181] * 18, 179, 35) == 179 + cost


def test_plan_task_order_runs_full_tasks_then_longest_tails() -> None:
    order = _plan_mod().plan_task_order
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


_PLAN_TABLE_UNUSED = (1 << 15, 1 << 15, 1 << 30, 0)


def _read_plan_tables(plan: list[int], requests: int, tasks: int, used: int) -> list[tuple]:
    tables = 3 + requests
    return [tuple(plan[tables + k * tasks + task] for k in range(5)) for task in range(used)]


def _page_stride(kv_lens: list[int]) -> int:
    """A fixed page-table stride covering every request (128-token pages, one spare)."""
    return (max(kv_lens) + 127) // 128 + 1


def _assert_plan_task_records(
    plan: list[int], requests: int, tasks: int, valid_k: list[int], page_stride: int
) -> None:
    """Every used task record repeats its request's fields for the decode prologue: the
    partial slot, the KV length and, with a fixed-stride page table, the page window
    ``[request * stride, +stride)`` (else page count -1); unused slots hold zeros."""
    tables = 3 + requests
    used = plan[1]
    for task in range(tasks):
        record = [plan[tables + k * tasks + task] for k in range(5, 9)]
        if task >= used:
            assert record == [0, 0, 0, 0]
            continue
        request = plan[tables + task]
        split = plan[tables + tasks + task]
        assert record == [
            plan[2 + request] + split,
            valid_k[request],
            request * page_stride,
            page_stride if page_stride > 0 else -1,
        ]


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 7) if torch.cuda.is_available() else True,
    reason="the CuTe DSL plan kernel is compiled for Rubin SM107",
)
@pytest.mark.parametrize("var_plan", [False, True])
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
        # Wave-boundary batches where variable-length wrap plans win (q4: 35 slots, q6: 21).
        ([32768] * 48, 3, 35, 4, False, True),
        ([32768] * 40, 3, 35, 4, False, True),
        ([45056] * 36, 3, 35, 4, False, True),
        ([32768] * 64, 2, 21, 6, False, True),
        ([9000, 15000, 4000, 22000, 7000, 30000, 12000, 5000] * 3, 4, 17, 8, False, True),
        # The decode tests' wrap shapes (test_fp4_mla.py: h96-q8-wrap20-s4, h12-q8-wrap64-s3).
        ([10240] * 20, 4, 17, 8, False, True),
        ([20000] * 64, 3, 106, 8, False, True),
        # DEP16 generation ranks past 64 requests (40K tokens each, two slots per
        # request), up to the simulation's request limit and one past it (the rule's length).
        ([40960] * 72, 2, 35, 4, False, True),
        ([40960] * 96, 2, 35, 4, False, True),
        ([40960] * 128, 2, 35, 4, False, True),
        ([40960] * 129, 2, 35, 4, False, True),
        # TRTLLM_FP4_MLA_PACKED_WAVE_PLAN=0, off and on Helix: the rule's length, request order.
        (_DEP16_RANK_KV_LENS, 9, 35, 4, False, False),
        ([131, 20000, 600, 45000], 10, 17, 8, True, False),
    ],
)
def test_packed_plan_kernel_matches_host_mirror(
    kv_lens: list[int],
    splits: int,
    wave_tasks: int,
    query_len: int,
    helix: bool,
    wave_plan: bool,
    var_plan: bool,
) -> None:
    reduce_mod = _plan_mod()
    requests = len(kv_lens)
    tasks = requests * splits
    m_tiles = max(1, 106 // wave_tasks)
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
    expected = reduce_mod.plan_packed_table(
        tiles,
        tasks,
        wave_tasks,
        cap,
        one_wave=helix,
        wave_plan=wave_plan,
        var_plan=var_plan,
        m_tiles=m_tiles,
    )
    plan = torch.full(
        (reduce_mod.packed_plan_length(requests, tasks),), -7, dtype=torch.int32, device="cuda"
    )
    # Both page-window modes: a fixed stride (var plan cases) and none (the others).
    page_stride = _page_stride(kv_lens) if var_plan else 0
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
        var_plan=var_plan,
        m_tiles=m_tiles,
        page_stride=page_stride,
    )
    torch.cuda.synchronize()
    plan = plan.tolist()
    used = expected.begins[-1]
    assert plan[0] == expected.tiles_per_task
    assert plan[1] == used
    assert plan[2 : 3 + requests] == expected.begins
    # Tasks in grid order: (request, split, count, first KV tile, KV tiles).
    assert _read_plan_tables(plan, requests, tasks, used) == expected.tasks
    _assert_plan_task_records(plan, requests, tasks, kv_lens, page_stride)
    # Unused slots: last request, out-of-range split, no tiles; never scheduled.
    tables = 3 + requests
    for task in range(used, tasks):
        assert plan[tables + task] == requests - 1
        assert tuple(plan[tables + k * tasks + task] for k in range(1, 5)) == _PLAN_TABLE_UNUSED
    if not var_plan:
        # The equal-length plan is unchanged by the variable-length plan's arrival.
        tiles_per_task, begins = reduce_mod.plan_packed_tasks(
            tiles, tasks, wave_tasks, cap, one_wave=helix, wave_plan=wave_plan
        )
        assert (plan[0], plan[2 : 3 + requests]) == (tiles_per_task, begins)


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
    expected = reduce_mod.plan_packed_table(tiles, tasks, wave_tasks, cap, one_wave=helix)
    kv = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    prompt = torch.tensor(prompt_lens, dtype=torch.int32, device="cuda")
    corrected_out = torch.full((requests,), -1, dtype=torch.int32, device="cuda")
    generation_out = torch.full((requests,), -1, dtype=torch.int32, device="cuda")
    plan = torch.full(
        (reduce_mod.packed_plan_length(requests, tasks),), -7, dtype=torch.int32, device="cuda"
    )
    page_stride = _page_stride(corrected)
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
        page_stride=page_stride,
    )
    torch.cuda.synchronize()
    # One launch writes the lengths every FP4 MLA kernel of the step reads and
    # plans on the corrected ones.
    assert corrected_out.tolist() == corrected
    assert generation_out.tolist() == [generation_len] * requests
    plan = plan.tolist()
    assert plan[0] == expected.tiles_per_task
    assert plan[1] == expected.begins[-1]
    assert plan[2 : 3 + requests] == expected.begins
    assert _read_plan_tables(plan, requests, tasks, expected.begins[-1]) == expected.tasks
    # The records carry the corrected KV lengths the decode reads.
    _assert_plan_task_records(plan, requests, tasks, corrected, page_stride)


def test_uniform_generation_lengths_fuse_the_packed_plan(monkeypatch) -> None:
    corrected = torch.zeros(2, dtype=torch.int32, device="cuda")
    generation_lens = torch.zeros(2, dtype=torch.int32, device="cuda")
    plan = torch.zeros(16, dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(
        num_contexts=0,
        num_seqs=2,
        kv_lens_cuda_runtime=torch.tensor([17, 25], dtype=torch.int32, device="cuda"),
        prompt_lens_cuda_runtime=torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
        fp4_mla_generation_kv_lens=corrected,
        fp4_mla_generation_append_lens=generation_lens,
        fp4_mla_generation_lengths_num_tokens=-1,
        fp4_mla_generation_lengths_num_seqs=-1,
        fp4_mla_generation_lengths_num_contexts=-1,
        _fp4_mla_generation_lengths_capture_recorded=False,
        is_cuda_graph=False,
        _fp4_mla_packed_plan_key=None,
    )
    geometry = fp4_mla_backend._PackedPlanGeometry(2, 4, 3, 17, 64, 6, 1024, None, plan, ("key",))
    calls = []
    monkeypatch.setattr(
        fp4_mla_backend,
        "populate_fp4_mla_generation_lengths",
        lambda *args, **kwargs: calls.append(("lengths", kwargs)),
    )
    monkeypatch.setattr(
        _plan_mod(), "launch_packed_plan", lambda *args, **kwargs: calls.append(("plan", kwargs))
    )
    uniform = fp4_mla_backend._fp4_mla_uniform_generation_lengths

    # First FP4 MLA kernel of the step: one fused launch writes lengths and plan.
    uniform(metadata, 8, 2, plan_geometry=geometry)
    assert [name for name, _ in calls] == ["plan"]
    assert calls[0][1]["generation_lengths"][3] == 4
    assert metadata._fp4_mla_packed_plan_key == ("key",)
    assert metadata._fp4_mla_metadata_launch_pending
    # The other layers of the step launch nothing.
    uniform(metadata, 8, 2, plan_geometry=geometry)
    uniform(metadata, 8, 2)
    assert len(calls) == 1
    # A decode planning on other bounds rewrites the plan alone.
    uniform(metadata, 8, 2, plan_geometry=geometry._replace(key=("other",)))
    assert len(calls) == 2 and calls[1][0] == "plan" and calls[1][1]["generation_lengths"] is None
    # A new step whose first caller cannot plan keeps the Triton lengths kernel,
    # and the decode then plans alone.
    metadata.fp4_mla_generation_lengths_num_tokens = -1
    metadata._fp4_mla_packed_plan_key = None
    metadata._fp4_mla_metadata_launch_pending = False
    uniform(metadata, 8, 2)
    assert metadata._fp4_mla_metadata_launch_pending
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
    geometry = fp4_mla_backend._PackedPlanGeometry(2, 4, 3, 17, 64, 6, 1024, None, None, ("key",))
    kv_lens = torch.zeros(2, dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(_fp4_mla_packed_plan_key=None)
    monkeypatch.delenv(fp4_mla_backend.FP4_MLA_PACKED_WAVE_PLAN_ENV, raising=False)
    fp4_mla_backend._launch_fp4_mla_packed_plan(geometry, kv_lens, metadata)
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_WAVE_PLAN_ENV, "0")
    fp4_mla_backend._launch_fp4_mla_packed_plan(geometry, kv_lens, metadata)
    assert calls == [True, False]


def test_packed_decode_pdl_only_follows_the_cache_update(monkeypatch) -> None:
    # The decode reads its plan, lengths and page table before griddepcontrol.wait, so it
    # launches as a programmatic dependent only while no plan, lengths or page-table kernel
    # has been launched since the last generation cache update.
    monkeypatch.setattr(_plan_mod(), "launch_packed_plan", lambda *args, **kwargs: None)
    monkeypatch.delenv(fp4_mla_backend.FP4_MLA_PACKED_DECODE_PDL_ENV, raising=False)
    monkeypatch.delenv("TRTLLM_ENABLE_PDL", raising=False)
    uses_pdl = fp4_mla_backend._packed_decode_uses_pdl
    metadata = SimpleNamespace(_fp4_mla_packed_plan_key=None)
    assert uses_pdl(metadata)
    geometry = fp4_mla_backend._PackedPlanGeometry(2, 4, 3, 17, 64, 6, 1024, None, None, ("key",))
    kv_lens = torch.zeros(2, dtype=torch.int32, device="cuda")
    fp4_mla_backend._launch_fp4_mla_packed_plan(geometry, kv_lens, metadata)
    assert not uses_pdl(metadata)
    # As the generation cache update leaves it.
    metadata._fp4_mla_metadata_launch_pending = False
    assert uses_pdl(metadata)
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_DECODE_PDL_ENV, "0")
    assert not uses_pdl(metadata)
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_DECODE_PDL_ENV, "1")
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    assert not uses_pdl(metadata)


def test_packed_plan_page_stride_follows_the_device_page_table(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        _plan_mod(),
        "launch_packed_plan",
        lambda *args, **kwargs: calls.append(kwargs["page_stride"]),
    )
    geometry = fp4_mla_backend._PackedPlanGeometry(2, 4, 3, 17, 64, 6, 1024, None, None, ("key",))
    kv_lens = torch.zeros(2, dtype=torch.int32, device="cuda")
    # The fixed-stride device page table: request r's decode window is r * stride, which
    # the plan records carry (the table may be materialized after the plan launch).
    fixed = SimpleNamespace(
        _fp4_mla_packed_plan_key=None,
        _fp4_mla_device_page_table=True,
        fp4_mla_page_table_stride=37,
    )
    fp4_mla_backend._launch_fp4_mla_packed_plan(geometry, kv_lens, fixed)
    # Without it the records carry no window and the decode kernel reads its indptr.
    other = SimpleNamespace(_fp4_mla_packed_plan_key=None, fp4_mla_page_table_stride=37)
    fp4_mla_backend._launch_fp4_mla_packed_plan(geometry, kv_lens, other)
    assert calls == [37, 0]


def test_packed_plan_var_plan_env_rolls_back(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        _plan_mod(),
        "launch_packed_plan",
        lambda *args, **kwargs: calls.append((kwargs["var_plan"], kwargs["m_tiles"])),
    )
    geometry = fp4_mla_backend._PackedPlanGeometry(
        2, 4, 3, 17, 64, 6, 1024, None, None, ("key",), 6
    )
    kv_lens = torch.zeros(2, dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(_fp4_mla_packed_plan_key=None)
    monkeypatch.delenv(fp4_mla_backend.FP4_MLA_PACKED_VAR_PLAN_ENV, raising=False)
    fp4_mla_backend._launch_fp4_mla_packed_plan(geometry, kv_lens, metadata)
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_VAR_PLAN_ENV, "0")
    fp4_mla_backend._launch_fp4_mla_packed_plan(geometry, kv_lens, metadata)
    assert calls == [(True, 6), (False, 6)]


def _assert_plan_table_covers_every_tile(
    tiles: list[int], capacity: int, plan, max_tiles_per_task: int
) -> None:
    requests = len(tiles)
    assert len(plan.begins) == requests + 1 and plan.begins[0] == 0
    assert plan.begins[-1] == len(plan.tasks) <= capacity
    pieces = {request: [] for request in range(requests)}
    for request, split, count, first, length in plan.tasks:
        assert count == plan.begins[request + 1] - plan.begins[request]
        assert 0 <= split < count and length >= 1
        assert length <= min(plan.tiles_per_task, max_tiles_per_task)
        pieces[request].append((first, length, split))
    for request, request_tiles in enumerate(tiles):
        position = 0
        for first, length, _ in sorted(pieces[request]):
            assert first == position
            position += length
        assert position == request_tiles
        assert sorted(split for _, _, split in pieces[request]) == list(range(len(pieces[request])))


def test_plan_packed_table_random_batches_cover_every_tile_once() -> None:
    import random

    reduce_mod = _plan_mod()
    rng = random.Random(20261019)
    wrap_plans = 0
    for _ in range(2000):
        requests = rng.randint(1, 70)
        tiles = [rng.choice([1, 2, 3, 8, 64, 128, 175, 500, 1024]) for _ in range(requests)]
        wave_tasks, m_tiles = rng.choice([(17, 6), (21, 5), (35, 3), (106, 1)])
        splits = max(2, min(16, math.ceil(2 * wave_tasks / requests) + 1))
        max_tiles = max(640, math.ceil(max(tiles) / (splits - 1)))
        for var_plan in (False, True):
            plan = reduce_mod.plan_packed_table(
                tiles, requests * splits, wave_tasks, max_tiles, var_plan=var_plan, m_tiles=m_tiles
            )
            _assert_plan_table_covers_every_tile(tiles, requests * splits, plan, max_tiles)
            lengths = [task[4] for task in plan.tasks]
            if any(task[3] % plan.tiles_per_task for task in plan.tasks):
                wrap_plans += 1
                assert lengths == sorted(lengths, reverse=True)
    assert wrap_plans > 0


@pytest.mark.parametrize(
    "tiles,wave_tasks,m_tiles,old_tasks,max_new_tasks",
    [
        # q4 (3 M tiles, 35 slots): 48 even 32K requests no longer split every request in two.
        ([128] * 48, 35, 3, 96, 80),
        ([128] * 40, 35, 3, 120, 72),
        # q6 (5 M tiles, 21 slots): 64 even 32K requests (a shape where the equal-length plan regressed).
        ([128] * 64, 21, 5, 128, 80),
    ],
)
def test_var_plan_fills_the_waves_with_fewer_tasks(
    tiles: list[int], wave_tasks: int, m_tiles: int, old_tasks: int, max_new_tasks: int
) -> None:
    reduce_mod = _plan_mod()
    requests = len(tiles)
    capacity = requests * max(2, min(16, math.ceil(2 * wave_tasks / requests) + 1))
    old = reduce_mod.plan_packed_table(tiles, capacity, wave_tasks, 640, var_plan=False)
    new = reduce_mod.plan_packed_table(tiles, capacity, wave_tasks, 640, m_tiles=m_tiles)
    assert len(old.tasks) == old_tasks
    assert len(new.tasks) <= max_new_tasks
    task_cost, partial_cost64 = reduce_mod.packed_plan_costs(m_tiles)

    def score(plan) -> int:
        counts = [b - a for a, b in zip(plan.begins, plan.begins[1:])]
        partial = sum(n for n in counts if n > 1)
        makespan = reduce_mod.plan_lpt_makespan(
            [task[4] for task in plan.tasks], wave_tasks, task_cost
        )
        return makespan * reduce_mod.PLAN_V_COST_SCALE + partial * partial_cost64

    assert score(new) < score(old)


def test_var_plan_fills_the_third_wave_past_64_requests() -> None:
    """DEP16 generation rank (H96 q4: 3 M tiles, 35 slots), 40K tokens a request:
    one task per request put 72 requests on 35 slots in three waves, the last of two tasks,
    so B72 cost as much as B96. The simulated plans spread the work over the
    waves; past PLAN_SIM_MAX_REQUESTS the rule's length is kept."""
    reduce_mod = _plan_mod()
    wave_tasks, m_tiles = 35, 3
    task_cost, _ = reduce_mod.packed_plan_costs(m_tiles)
    tiles = [160] * 72
    # _select_packed_split_count at 72 requests: two slots per request.
    plan = reduce_mod.plan_packed_table(tiles, 2 * len(tiles), wave_tasks, 1026, m_tiles=m_tiles)
    assert len(plan.tasks) > len(tiles)
    makespan = reduce_mod.plan_lpt_makespan([task[4] for task in plan.tasks], wave_tasks, task_cost)
    assert makespan < reduce_mod.plan_lpt_makespan(tiles, wave_tasks, task_cost)
    past_limit = [160] * (reduce_mod.PLAN_SIM_MAX_REQUESTS + 1)
    rule = reduce_mod.plan_packed_table(
        past_limit, 2 * len(past_limit), wave_tasks, 1026, m_tiles=m_tiles
    )
    assert len(rule.tasks) == len(past_limit)


def test_packed_plan_geometry_is_shared_by_cache_update_and_decode(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_FP4_MLA_SPLIT_KV", raising=False)
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_Q_ENV, "1")
    monkeypatch.setattr(fp4_mla_backend, "_fp4_mla_attention_backend", lambda: "cutedsl")
    monkeypatch.setattr(fp4_mla_backend, "_get_sm_count", lambda device: 212)
    # An indexed device, as the kernels' tensors carry: the workspace helper
    # compares devices exactly before reusing a buffer.
    device = torch.device("cuda", torch.cuda.current_device())
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(fp4_mla_v_layout=fp4_mla_backend.Fp4MlaVLayout.DUAL_COPY),
        num_contexts=0,
        num_seqs=4,
        _fp4_mla_device_page_table=True,
        fp4_mla_page_table_stride=3904,
        page_size=256,
        is_cuda_graph=True,
        fp4_mla_generation_kv_lens=torch.zeros(4, dtype=torch.int32, device=device),
    )
    geometry = fp4_mla_backend._fp4_mla_packed_plan_geometry(
        metadata, num_heads=96, query_len=8, num_requests=4, device=device, helix_kv_bounds=None
    )
    # helix16 B4 MTP7: 6 m-tiles, 17 tasks per wave, 10 slots per request.
    assert (geometry.splits, geometry.wave_tasks, geometry.task_capacity) == (10, 17, 40)
    # The plan lives in the metadata workspace (a view of the stored buffer).
    assert geometry.plan.data_ptr() == metadata._fp4_mla_packed_plan.data_ptr()
    assert geometry.plan.numel() == _plan_mod().packed_plan_length(4, 40)
    again = fp4_mla_backend._fp4_mla_packed_plan_geometry(
        metadata, num_heads=96, query_len=8, num_requests=4, device=device, helix_kv_bounds=None
    )
    assert again.key == geometry.key
    bounds = torch.zeros(32, dtype=torch.int32, device=device)
    with_bounds = fp4_mla_backend._fp4_mla_packed_plan_geometry(
        metadata, num_heads=96, query_len=8, num_requests=4, device=device, helix_kv_bounds=bounds
    )
    assert with_bounds.key != geometry.key
    # Too many rows for the packed layout: the legacy path, no geometry.
    assert (
        fp4_mla_backend._fp4_mla_packed_plan_geometry(
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
    single = fp4_mla_backend._fp4_mla_packed_plan_geometry(
        metadata, num_heads=96, query_len=8, num_requests=4, device=device, helix_kv_bounds=None
    )
    assert single.splits == 1 and single.key is None and single.plan is None


def test_request_kv_tiles_follow_helix_bounds() -> None:
    tiles = _plan_mod().request_kv_tiles
    assert tiles([1, 256, 257, 0], 1) == [1, 1, 2, 1]
    # Helix: the last query token's bound of each request decides.
    assert tiles([5000, 5000], 2, [100, 512, 5000, 1000]) == [2, 4]


def test_reduce_groups_only_long_task_lists_over_few_rows() -> None:
    reduce_mod = _plan_mod()
    groups = reduce_mod.select_reduce_task_groups
    # One or two requests (64+ task slots each) over up to 48 / 96 output rows.
    assert groups(12, 128) == 4
    assert groups(48, 128) == 4
    assert groups(24, 107) == 4
    assert groups(49, 128) == 2
    assert groups(96, 107) == 2
    # More rows, or fewer slots per request (four or more requests), keep one warp.
    assert groups(97, 128) == 1
    assert groups(192, 54) == 1
    assert groups(48, 63) == 1
    ctas = reduce_mod.select_reduce_ctas
    assert ctas(48, 4, 212) == 48
    assert ctas(3072, 1, 212) == 768
    assert ctas(1 << 20, 1, 212) == 4 * 212


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 7) if torch.cuda.is_available() else True,
    reason="the CuTe DSL reducer is compiled for Rubin SM107",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_stats", [False, True])
@pytest.mark.parametrize(
    "kv_tiles,splits,wave_tasks,num_heads,query_len,forced_groups",
    [
        ([128, 128, 128, 128], 6, 17, 96, 8, 0),
        ([4096], 16, 106, 12, 8, 0),
        ([3, 300, 3], 3, 106, 6, 6, 0),
        ([7, 1024], 2, 26, 128, 4, 0),
        ([1, 1, 1, 1, 1], 1, 17, 96, 1, 0),
        # Skewed batches use an uneven number of tasks per request, and
        # requests that end up with one task are stats-only here.
        ([1, 79, 3, 176], 6, 17, 96, 8, 0),
        ([128] * 15 + [1024], 3, 106, 12, 8, 0),
        # Few rows and long task lists: the tasks of an item are split over four
        # (12 rows) or two (96 rows) warps and combined in shared memory, also when
        # short requests leave some of those warps without tasks.
        ([4096], 64, 106, 12, 1, 0),
        ([4096], 96, 106, 12, 1, 0),
        ([4096], 64, 106, 12, 8, 0),
        ([32, 32], 64, 106, 12, 4, 0),
        # Groups the selection never picks (any band of the kernel): two warps per
        # 128- and 256-column item, four per 512-column item.
        ([2048, 2048], 24, 106, 12, 1, 2),
        ([1024, 1024, 1024], 12, 106, 12, 8, 2),
        ([4096], 17, 17, 96, 8, 4),
    ],
)
def test_packed_split_reduce_matches_reference_and_replays(
    monkeypatch,
    dtype: torch.dtype,
    with_stats: bool,
    kv_tiles: list[int],
    splits: int,
    wave_tasks: int,
    num_heads: int,
    query_len: int,
    forced_groups: int,
) -> None:
    reduce_mod = _plan_mod()
    if forced_groups:
        monkeypatch.setattr(reduce_mod, "select_reduce_task_groups", lambda *args: forced_groups)
    torch.manual_seed(7)
    requests = len(kv_tiles)
    valid_m = num_heads * query_len
    physical_m = (valid_m + 127) // 128 * 128
    tasks = splits * requests
    # KV lengths just below the tile boundary; the kernels' plan is the mirror's plan.
    kv_lens = torch.tensor([t * 256 - 3 for t in kv_tiles], dtype=torch.int32, device="cuda")
    expected = reduce_mod.plan_packed_table(kv_tiles, tasks, wave_tasks, max(kv_tiles))
    tiles_per_task, begins = expected.tiles_per_task, expected.begins
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
