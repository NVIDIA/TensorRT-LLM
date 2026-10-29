# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed FP4 MLA decode: row layout policy and the CuTe DSL split reducer."""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.attention.backends.fp4_mla.config as fp4_mla_config
import tensorrt_llm._torch.attention.backends.fp4_mla.metadata as fp4_mla_metadata
import tensorrt_llm._torch.attention.backends.fp4_mla.packed_plan as fp4_mla_packed_plan

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


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 7) if torch.cuda.is_available() else True,
    reason="the CuTe DSL plan kernel is compiled for Rubin SM107",
)
@pytest.mark.parametrize(
    "kv_lens,splits,wave_tasks,query_len,helix",
    [
        ([32768] * 15 + [262144], 4, 17, 8, False),
        ([131, 20000, 600, 45000], 10, 17, 8, False),
        ([131, 20000, 600, 45000], 10, 17, 8, True),
        ([300, 70000, 2200], 16, 106, 1, False),
        ([5, 5, 5], 2, 106, 4, False),
    ],
)
def test_packed_plan_kernel_matches_host_mirror(
    kv_lens: list[int], splits: int, wave_tasks: int, query_len: int, helix: bool
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
        tiles, tasks, wave_tasks, cap, one_wave=helix
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
    for request, (begin, end) in enumerate(zip(expected_begins, expected_begins[1:])):
        for task in range(begin, end):
            assert (request_of[task], split_of[task], count_of[task]) == (
                request,
                task - begin,
                end - begin,
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
