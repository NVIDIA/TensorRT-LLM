# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential boundary tests for the MSA direct-greedy planner fast paths.

The oracle is a CPU translation of the pre-fast-path ``direct_greedy``
algorithm, with full-width integer cost ordering.  Comparing packed planner
output, rather than only coverage, catches changes in tie-breaking, per-SM
ordering, and dropped work.

Only the planner runs: an 8M-token KV cache is not allocated.  The GPU tests
skip unless the patched MSA package and an SM100 CUDA runtime are available.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest
import torch


_PER_ITER_COST = 43
_TILE_COST = 110
_SM_COST = 165
_LEGACY_PACKED_COST_SENTINEL = 0x7FFFFF
_LEGACY_PACKED_COST_MAX = 0xFFFFFF


@lru_cache(maxsize=1)
def _load_patched_msa_api() -> ModuleType:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the MSA planner")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the MSA planner requires an SM100 (Blackwell) GPU")

    try:
        import fmha_sm100.api as msa_api
    except (ImportError, OSError) as error:
        pytest.skip(f"fmha_sm100 (MSA) is unavailable: {error}")

    plan_source = Path(msa_api.__file__).parent / "csrc" / "include" / "plan.cuh"
    if not plan_source.is_file():
        pytest.skip("installed MSA package does not contain the planner source")
    if "Fast path 2 (single head)" not in plan_source.read_text():
        pytest.skip("installed MSA package does not contain the planner fast paths")
    return msa_api


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _compute_kv_iters(
    batch: int,
    qo_tile: int,
    qo_lens: list[int],
    kv_lens: list[int],
    qo_offsets: list[int],
    qo_tile_size: int,
    kv_tile_size: int,
    causal: bool,
) -> int:
    kv_len = kv_lens[batch]
    if not causal:
        return _ceil_div(kv_len, kv_tile_size)
    q_end = (qo_tile + 1) * qo_tile_size
    effective_kv = min(q_end + qo_offsets[batch], kv_len)
    return 0 if effective_kv <= 0 else _ceil_div(effective_kv, kv_tile_size)


def _stock_direct_greedy(
    qo_lens: list[int],
    kv_lens: list[int],
    num_heads: int,
    num_buckets: int,
    causal: bool,
    qo_offsets: list[int],
) -> tuple[list[int], list[int]]:
    """Return the packed ranges/work-info emitted by stock direct_greedy."""

    qo_tile_size = 128 if max(qo_lens) <= 128 else 256
    kv_tile_size = 256 if qo_tile_size == 128 else 128
    max_qo_tiles = max(_ceil_div(qo_len, qo_tile_size) for qo_len in qo_lens)

    costs = [_SM_COST] * num_buckets
    tasks: list[list[tuple[int, int, int]]] = [[] for _ in range(num_buckets)]

    # Stock row order and rank-based assignment.  All cases use one KV split,
    # so each active row has exactly one KV piece.
    for qo_tile in range(max_qo_tiles - 1, -1, -1):
        for batch, qo_len in enumerate(qo_lens):
            if qo_tile >= _ceil_div(qo_len, qo_tile_size):
                continue
            kv_iters = _compute_kv_iters(
                batch,
                qo_tile,
                qo_lens,
                kv_lens,
                qo_offsets,
                qo_tile_size,
                kv_tile_size,
                causal,
            )
            if kv_iters <= 0:
                continue

            tile_cost = _PER_ITER_COST * kv_iters + _TILE_COST
            head_offset = 0
            while head_offset < num_heads:
                batch_count = min(num_heads - head_offset, num_buckets)
                ranked_buckets = sorted(
                    range(num_buckets), key=lambda bucket: (costs[bucket], bucket)
                )
                for rank, bucket in enumerate(ranked_buckets[:batch_count]):
                    costs[bucket] += tile_cost
                    tasks[bucket].append((qo_tile, head_offset + rank, batch))
                head_offset += batch_count

    ranges: list[int] = []
    work_info: list[int] = []
    start = 0
    for bucket_tasks in tasks:
        end = start + len(bucket_tasks)
        ranges.append((end << 32) | start)
        for qo_tile, head, batch in bucket_tasks:
            work_info.append((qo_tile << 32) | (head << 16) | batch)
        start = end
    return ranges, work_info


def _run_gpu_plan(
    qo_lens: list[int],
    kv_lens: list[int],
    num_heads: int,
    num_buckets: int,
    causal: bool = False,
) -> tuple[list[int], list[int]]:
    plan_fn = _load_patched_msa_api()._fmha_sm100_plan
    qo_lens_t = torch.tensor(qo_lens, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    qo_offsets_t = torch.zeros(len(qo_lens), dtype=torch.int32)

    plan = plan_fn(
        qo_lens_t,
        kv_lens_t,
        num_heads,
        num_kv_heads=-1,  # pack_factor=1: direct_greedy sees num_heads unchanged
        qo_offset=qo_offsets_t,
        num_kv_splits=1,
        usable_SM_count=num_buckets,
        causal=causal,
    )
    torch.cuda.synchronize()

    ranges = [int(value) for value in plan["packed_work_range"].cpu().tolist()]
    total_work = (ranges[-1] >> 32) & 0xFFFFFFFF
    work_info = [
        int(value)
        for value in plan["packed_work_info"][:total_work].cpu().tolist()
    ]
    return ranges, work_info


def _run_raw_gpu_plan(
    qo_lens: list[int],
    kv_lens: list[int],
    num_heads: int,
    num_buckets: int,
) -> tuple[list[int], list[int]]:
    """Call the JIT wrapper directly so tests can activate all 256 lanes.

    The public API caps ``num_buckets`` to the physical SM count.  The planner
    itself is one 256-thread block, so using 256 logical buckets is valid and
    ensures that neither reduction stage contains an inactive lane.
    """

    msa_api = _load_patched_msa_api()
    device = torch.device("cuda", torch.cuda.current_device())
    batch_size = len(qo_lens)
    qo_lens_t = torch.tensor(qo_lens, dtype=torch.int32, device=device)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    qo_offsets_t = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    packed_work_range = torch.empty(num_buckets, dtype=torch.int64, device=device)
    packed_work_info = torch.empty(batch_size * num_heads, dtype=torch.int64, device=device)

    msa_api._call_plan(
        qo_offsets_t,
        qo_lens_t,
        kv_lens_t,
        packed_work_range,
        packed_work_info,
        128,  # qo_tile_size
        256,  # kv_tile_size
        num_heads,
        num_buckets,
        False,  # causal
        None,  # qo_offset
        1,  # num_kv_splits
        None,  # kv_tile_begin_indices
        None,  # kv_tile_end_indices
        None,  # kv_split_indices
        0,  # chunk_size
        None,  # out_max_sm_cost
        None,  # num_kv_splits_per_row
        None,  # workspace_lse
        0,  # lse_total_size
        1,  # pack_factor
    )
    torch.cuda.synchronize()

    ranges = [int(value) for value in packed_work_range.cpu().tolist()]
    total_work = (ranges[-1] >> 32) & 0xFFFFFFFF
    work_info = [int(value) for value in packed_work_info[:total_work].cpu().tolist()]
    return ranges, work_info


def _assert_matches_stock(
    qo_lens: list[int],
    kv_lens: list[int],
    num_heads: int,
    num_buckets: int,
) -> None:
    qo_offsets = [0] * len(qo_lens)
    expected = _stock_direct_greedy(
        qo_lens, kv_lens, num_heads, num_buckets, False, qo_offsets
    )
    actual = _run_gpu_plan(qo_lens, kv_lens, num_heads, num_buckets)
    assert actual == expected


@pytest.mark.parametrize("kv_iters_boundary", [32767, 32768])
@pytest.mark.parametrize("seed", [1, 17, 314159])
def test_single_head_fast_path_matches_stock_at_int16_boundary(
    kv_iters_boundary: int, seed: int
) -> None:
    """Fast path 2 must preserve kv_iters on both sides of the int16 limit."""

    rng = random.Random(seed)
    batch_size = rng.randint(3, 8)
    kv_iters = [rng.randint(1, 1024) for _ in range(batch_size)]
    kv_iters[rng.randrange(batch_size)] = kv_iters_boundary

    # qo_tile_size=128 => kv_tile_size=256.  Variation prevents the uniform
    # fast path from intercepting this single-head scenario.
    qo_lens = [1] * batch_size
    kv_lens = [iters * 256 for iters in kv_iters]
    _assert_matches_stock(qo_lens, kv_lens, num_heads=1, num_buckets=8)


@pytest.mark.parametrize("kv_iters_boundary", [32767, 32768])
@pytest.mark.parametrize("seed", [2, 23, 271828])
def test_row_compaction_fast_path_matches_stock_at_int16_boundary(
    kv_iters_boundary: int, seed: int
) -> None:
    """Fast path 3 must preserve kv_iters on both sides of the int16 limit."""

    rng = random.Random(seed)
    batch_size = rng.randint(3, 7)
    boundary_batch = rng.randrange(batch_size)
    qo_lens = [rng.randint(1, 256) for _ in range(batch_size)]
    qo_lens[boundary_batch] = 257  # max_qo_tiles=2: excludes fast paths 1 and 2
    kv_iters = [rng.randint(1, 1024) for _ in range(batch_size)]
    kv_iters[boundary_batch] = kv_iters_boundary

    # max(qo_lens)>128 => kv_tile_size=128.
    kv_lens = [iters * 128 for iters in kv_iters]
    _assert_matches_stock(qo_lens, kv_lens, num_heads=2, num_buckets=8)


@pytest.mark.parametrize("cross_boundary", [False, True])
def test_single_head_fast_path_matches_stock_at_inactive_lane_sentinel(
    cross_boundary: bool,
) -> None:
    """An inactive reduction lane must not beat a full-width active cost."""

    # This varied prefix lands exactly one below the legacy inactive sentinel.
    # Appending two rows makes a subsequent find-min observe a cost above it.
    kv_iters = [21670] * 8 + [21697]
    accumulated_cost = _SM_COST + sum(
        _PER_ITER_COST * iters + _TILE_COST for iters in kv_iters
    )
    assert accumulated_cost == _LEGACY_PACKED_COST_SENTINEL - 1
    if cross_boundary:
        kv_iters += [1, 2]

    qo_lens = [1] * len(kv_iters)
    kv_lens = [iters * 256 for iters in kv_iters]
    _assert_matches_stock(qo_lens, kv_lens, num_heads=1, num_buckets=1)


def test_single_head_fast_path_matches_stock_across_24bit_cost_wrap() -> None:
    """Full-width ordering survives 0xffffff with no inactive lanes involved."""

    num_buckets = 256
    heavy_kv_iters = 32767
    heavy_tile_cost = _PER_ITER_COST * heavy_kv_iters + _TILE_COST

    # Eleven complete rounds leave every bucket below 0xffffff.  Another 128
    # heavy rows push buckets [0, 128) above it, while [128, 256) stay below.
    # The final light row must therefore go to bucket 128.  A 24-bit shift
    # wraps the expensive buckets and incorrectly chooses bucket 0 instead.
    low_cost = _SM_COST + 11 * heavy_tile_cost
    high_cost = low_cost + heavy_tile_cost
    assert low_cost < _LEGACY_PACKED_COST_MAX < high_cost
    assert (high_cost & _LEGACY_PACKED_COST_MAX) < low_cost

    kv_iters = [heavy_kv_iters] * (11 * num_buckets + 128) + [1]
    qo_lens = [1] * len(kv_iters)
    kv_lens = [iters * 256 for iters in kv_iters]
    qo_offsets = [0] * len(kv_iters)

    expected = _stock_direct_greedy(
        qo_lens,
        kv_lens,
        num_heads=1,
        num_buckets=num_buckets,
        causal=False,
        qo_offsets=qo_offsets,
    )
    actual = _run_raw_gpu_plan(
        qo_lens, kv_lens, num_heads=1, num_buckets=num_buckets
    )
    assert actual == expected

    # The last row uniquely identifies the decision made after the split cost
    # state.  Confirm that the stock oracle assigned it to bucket 128.
    last_batch = len(kv_iters) - 1
    bucket_start = expected[0][128] & 0xFFFFFFFF
    bucket_end = (expected[0][128] >> 32) & 0xFFFFFFFF
    assert last_batch in expected[1][bucket_start:bucket_end]
