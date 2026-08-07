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

from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest
import torch

_PER_ITER_COST = 43
_TILE_COST = 110
_SM_COST = 165
_LEGACY_PACKED_COST_SENTINEL = 0x7FFFFF


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
    work_info = [int(value) for value in plan["packed_work_info"][:total_work].cpu().tolist()]
    return ranges, work_info


def _assert_matches_stock(
    qo_lens: list[int],
    kv_lens: list[int],
    num_heads: int,
    num_buckets: int,
) -> None:
    qo_offsets = [0] * len(qo_lens)
    expected = _stock_direct_greedy(qo_lens, kv_lens, num_heads, num_buckets, False, qo_offsets)
    actual = _run_gpu_plan(qo_lens, kv_lens, num_heads, num_buckets)
    assert actual == expected


def test_single_head_fast_path_preserves_full_width_kv_iters() -> None:
    """Fast path 2 must not narrow a 32,768-iteration row to int16."""

    # qo_tile_size=128 => kv_tile_size=256. Variation prevents fast path 1
    # from intercepting this single-head scenario.
    qo_lens = [1, 1, 1, 1]
    kv_iters = [7, 32768, 31, 1024]
    kv_lens = [iters * 256 for iters in kv_iters]
    _assert_matches_stock(qo_lens, kv_lens, num_heads=1, num_buckets=8)


def test_row_compaction_fast_path_preserves_full_width_kv_iters() -> None:
    """Fast path 3 must not narrow a 32,768-iteration row to int16."""

    # max_qo_tiles=2 excludes fast paths 1 and 2. max(qo_lens)>128 selects a
    # 128-token KV tile, making the second request cross the int16 boundary.
    qo_lens = [1, 257, 129, 33]
    kv_iters = [3, 32768, 17, 5]
    kv_lens = [iters * 128 for iters in kv_iters]
    _assert_matches_stock(qo_lens, kv_lens, num_heads=2, num_buckets=8)


def test_single_head_fast_path_falls_back_before_packed_cost_sentinel() -> None:
    """Fast path 2 must not let an inactive lane beat an active bucket."""

    # This varied prefix lands exactly one below the legacy inactive sentinel.
    # Appending two rows makes a subsequent find-min observe a cost above it.
    kv_iters = [21670] * 8 + [21697]
    accumulated_cost = _SM_COST + sum(_PER_ITER_COST * iters + _TILE_COST for iters in kv_iters)
    assert accumulated_cost == _LEGACY_PACKED_COST_SENTINEL - 1
    # The first light row crosses the sentinel. The following row forces
    # another argmin, which was incorrect when costs were packed into 24 bits.
    kv_iters += [1, 2]

    qo_lens = [1] * len(kv_iters)
    kv_lens = [iters * 256 for iters in kv_iters]
    _assert_matches_stock(qo_lens, kv_lens, num_heads=1, num_buckets=1)
