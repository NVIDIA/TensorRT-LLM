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
"""Chunk planner tests: port of bounceTransferPlanTest.cpp.

Packing/alignment/cut rules, scatter-run coalescing, plus a randomized
byte-for-byte cross-check of the coalesced scatter runs against the naive
per-desc expectation.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import load_bounce_v2

_b = load_bounce_v2()
build_plan = _b.build_plan
ALIGNMENT = _b.ALIGNMENT


def plan_of(
    descs: list[tuple[int, int, int]],
    dsts: list[tuple[int, int, int]],
    max_chunk_bytes: int,
    max_descs: int,
):
    """Build a plan from C++-style (addr, len, dev) tuples for src and dst.

    The C++ carried a device id on BOTH sides; the Python planner takes a
    single dst_devs argument (src devices are uniform by construction on the
    sender), so lengths come from the src tuples and devices from the dst.
    """
    if len(descs) != len(dsts):
        raise ValueError("src/dst desc count mismatch")  # exercised via arrays
    src_ptrs = np.array([a for a, _length, _d in descs], dtype=np.uint64)
    dst_ptrs = np.array([a for a, _length, _d in dsts], dtype=np.uint64)
    sizes = np.array([length for _a, length, _d in descs], dtype=np.uint64)
    devs = np.array([d for _a, _length, d in dsts], dtype=np.uint32)
    return build_plan(src_ptrs, dst_ptrs, sizes, max_chunk_bytes, max_descs, devs)


def expand_runs_to_byte_map(runs: np.ndarray) -> dict[int, int]:
    """Expand scatter runs to a {bounce byte offset -> dst byte address} map."""
    mapping: dict[int, int] = {}
    for run in runs:
        bounce_off = int(run["bounce_offset"])
        dst_addr = int(run["dst_addr"])
        dst_stride = int(run["dst_stride"])
        bounce_stride = int(run["bounce_stride"])
        piece_size = int(run["piece_size"])
        count = int(run["count"])
        for p in range(count):
            b0 = bounce_off + p * bounce_stride
            d0 = dst_addr + p * dst_stride
            for i in range(piece_size):
                key = b0 + i
                assert key not in mapping, f"scatter runs overlap at bounce byte {key}"
                mapping[key] = d0 + i
    return mapping


def per_desc_byte_map(
    bounce_offsets: np.ndarray, dst_ptrs: np.ndarray, sizes: np.ndarray
) -> dict[int, int]:
    """The naive per-desc {bounce byte offset -> dst byte address} map."""
    mapping: dict[int, int] = {}
    for off, dst, size in zip(bounce_offsets.tolist(), dst_ptrs.tolist(), sizes.tolist()):
        for i in range(int(size)):
            mapping[int(off) + i] = int(dst) + i
    return mapping


# C++: BounceTransferPlan.EmptyYieldsNoChunks
def test_empty_yields_no_chunks() -> None:
    plan = plan_of([], [], 1024, 64)
    assert plan.num_chunks == 0
    assert plan.total_descs == 0
    assert plan.total_bytes == 0


# C++: BounceTransferPlan.SingleDescOneChunk
def test_single_desc_one_chunk() -> None:
    plan = plan_of([(0x1000, 100, 0)], [(0x9000, 100, 0)], 1024, 64)
    assert plan.num_chunks == 1
    c = plan.chunks[0]
    assert c.num_descs == 1
    assert c.bounce_offsets[0] == 0
    assert c.sizes[0] == 100
    assert c.total_bytes == 100
    assert c.dst_ptrs[0] == 0x9000


# C++: BounceTransferPlan.TwoDescsPackOneChunkWith32ByteAlignedOffsets
def test_two_descs_pack_one_chunk_with_32_byte_aligned_offsets() -> None:
    # len 100 -> next offset aligns up to 128 (multiple of 32).
    plan = plan_of(
        [(0x1000, 100, 0), (0x2000, 50, 0)],
        [(0x9000, 100, 0), (0xA000, 50, 0)],
        1024,
        64,
    )
    assert plan.num_chunks == 1
    c = plan.chunks[0]
    assert c.bounce_offsets[0] == 0
    assert c.bounce_offsets[1] == 128  # align_up(100, 32) = 128
    assert c.sizes.tolist() == [100, 50]
    assert c.total_bytes == 150


# C++: BounceTransferPlan.OverflowSplitsIntoTwoChunks
def test_overflow_splits_into_two_chunks() -> None:
    # With a 256-byte chunk cap, two 200-byte descs cannot share a chunk.
    plan = plan_of(
        [(0x1000, 200, 0), (0x2000, 200, 0)],
        [(0x9000, 200, 0), (0xA000, 200, 0)],
        256,
        64,
    )
    assert plan.num_chunks == 2
    assert plan.chunks[0].num_descs == 1
    assert plan.chunks[1].num_descs == 1


# C++: BounceTransferPlan.DescExactlyChunkSizeIsOneChunk
def test_desc_exactly_chunk_size_is_one_chunk() -> None:
    plan = plan_of([(0x1000, 256, 0)], [(0x9000, 256, 0)], 256, 64)
    assert plan.num_chunks == 1
    assert plan.chunks[0].total_bytes == 256


# C++: BounceTransferPlan.DescLargerThanChunkThrows
def test_desc_larger_than_chunk_raises() -> None:
    with pytest.raises(ValueError):
        plan_of([(0x1000, 257, 0)], [(0x9000, 257, 0)], 256, 64)


# C++: BounceTransferPlan.MaxChunkSizeBytesAboveU32Throws
def test_max_chunk_size_bytes_above_u32_raises() -> None:
    # A chunk's packed size travels in 32-bit wire fields.
    with pytest.raises(ValueError):
        plan_of([(0x1000, 8, 0)], [(0x9000, 8, 0)], 1 << 32, 64)
    # Exactly 4 GiB - 1 is allowed.
    plan = plan_of([(0x1000, 8, 0)], [(0x9000, 8, 0)], (1 << 32) - 1, 64)
    assert plan.num_chunks == 1


# C++: BounceTransferPlan.MaxDescsPerChunkBoundary
def test_max_descs_per_chunk_boundary() -> None:
    # 3 tiny descs, max_descs=2 -> first chunk holds 2, second holds 1.
    plan = plan_of(
        [(0x1000, 8, 0), (0x2000, 8, 0), (0x3000, 8, 0)],
        [(0x9000, 8, 0), (0xA000, 8, 0), (0xB000, 8, 0)],
        4096,
        2,
    )
    assert plan.num_chunks == 2
    assert plan.chunks[0].num_descs == 2
    assert plan.chunks[1].num_descs == 1


# C++: BounceTransferPlan.MixedDestinationDeviceIdsThrow — deviation: the
# Python planner SUPPORTS mixed dst devices by cutting a chunk at every
# device boundary (the C++ rejected them). MixedSourceDeviceIdsThrow is not
# ported: the Python API does not carry per-desc src devices.
def test_mixed_destination_device_ids_cut_chunks() -> None:
    plan = plan_of(
        [(0x1000, 8, 0), (0x2000, 8, 0), (0x3000, 8, 0)],
        [(0x9000, 8, 0), (0xA000, 8, 1), (0xB000, 8, 1)],
        4096,
        64,
    )
    assert plan.num_chunks == 2
    assert plan.chunks[0].num_descs == 1
    assert plan.chunks[0].dst_device_id == 0
    assert plan.chunks[1].num_descs == 2
    assert plan.chunks[1].dst_device_id == 1


# C++: BounceTransferPlan.ZeroLengthDescSkippedButCounted
def test_zero_length_desc_skipped_but_counted() -> None:
    plan = plan_of(
        [(0x1000, 0, 0), (0x2000, 16, 0)],
        [(0x9000, 0, 0), (0xA000, 16, 0)],
        1024,
        64,
    )
    assert plan.num_chunks == 1
    assert plan.chunks[0].num_descs == 1  # zero-len skipped from packing
    assert plan.total_descs == 2  # but still counted as seen
    assert plan.total_bytes == 16


# C++: BounceTransferPlan.CountMismatchThrows
def test_count_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        build_plan(
            np.array([0x1000], dtype=np.uint64),
            np.array([], dtype=np.uint64),
            np.array([8], dtype=np.uint64),
            1024,
            64,
        )


# C++: BounceTransferPlan.ContiguousSrcAndDstDescsMergeInPlace — deviation:
# the Python planner does NOT merge contiguous gather descs in place (per-desc
# arrays keep both entries), but the DATA MOVEMENT is identical: the scatter
# view still collapses the dense pair to ONE run and packed bytes match.
def test_contiguous_src_and_dst_descs_scatter_collapses_to_one_run() -> None:
    plan = plan_of(
        [(0x1000, 32, 0), (0x1020, 32, 0)],
        [(0x9000, 32, 0), (0x9020, 32, 0)],
        1024,
        64,
    )
    assert plan.num_chunks == 1
    c = plan.chunks[0]
    assert c.num_descs == 2  # DEVIATION from C++: no in-place gather merge
    assert c.total_bytes == 64
    assert c.packed_bytes == 64
    assert plan.total_descs == 2
    assert plan.total_bytes == 64
    assert len(c.scatter_runs) == 1  # dense pair -> one count-1 run
    assert int(c.scatter_runs[0]["piece_size"]) == 64
    assert int(c.scatter_runs[0]["count"]) == 1


# C++: BounceTransferPlan.ContiguousSrcOnlyDoesNotMergeDescs
def test_contiguous_src_only_does_not_merge_descs() -> None:
    # src contiguous but dst jumps -> per-desc arrays must stay separate.
    plan = plan_of(
        [(0x1000, 32, 0), (0x1020, 32, 0)],
        [(0x9000, 32, 0), (0xA000, 32, 0)],
        1024,
        64,
    )
    assert plan.num_chunks == 1
    assert plan.chunks[0].num_descs == 2


# C++: BounceTransferPlan.ScatterRunsCoalesceContiguousDst
def test_scatter_runs_coalesce_contiguous_dst() -> None:
    # dst contiguous, src strided: per-desc arrays keep 3 entries for the
    # gather; the scatter view collapses to ONE count==1 run over the extent.
    plan = plan_of(
        [(0x1000, 32, 0), (0x3000, 32, 0), (0x5000, 32, 0)],
        [(0x9000, 32, 0), (0x9020, 32, 0), (0x9040, 32, 0)],
        1024,
        64,
    )
    assert plan.num_chunks == 1
    c = plan.chunks[0]
    assert c.num_descs == 3
    assert len(c.scatter_runs) == 1
    run = c.scatter_runs[0]
    assert int(run["dst_addr"]) == 0x9000
    assert int(run["bounce_offset"]) == 0
    assert int(run["piece_size"]) == 96
    assert int(run["count"]) == 1


# C++: BounceTransferPlan.ScatterRunsCoalesceUniformlyStridedDst
def test_scatter_runs_coalesce_uniformly_strided_dst() -> None:
    # dst uniformly strided: ONE strided run of count 3; the bounce packing
    # steps by exactly 32 (aligned), so bounce_stride == piece_size.
    plan = plan_of(
        [(0x1000, 32, 0), (0x3000, 32, 0), (0x5000, 32, 0)],
        [(0x9000, 32, 0), (0x9080, 32, 0), (0x9100, 32, 0)],
        1024,
        64,
    )
    assert plan.num_chunks == 1
    c = plan.chunks[0]
    assert len(c.scatter_runs) == 1
    run = c.scatter_runs[0]
    assert int(run["dst_addr"]) == 0x9000
    assert int(run["dst_stride"]) == 0x80
    assert int(run["bounce_stride"]) == 32
    assert int(run["piece_size"]) == 32
    assert int(run["count"]) == 3


# C++: BounceTransferPlan.ScatterRunsBreakOnDstHoleOrAlignGap
def test_scatter_runs_break_on_dst_hole_or_align_gap() -> None:
    # First pair: dst steps forward but the second desc's SIZE differs -> no
    # stride latch -> two runs.
    plan_hole = plan_of(
        [(0x1000, 32, 0), (0x3000, 16, 0)],
        [(0x9000, 32, 0), (0xA000, 16, 0)],
        1024,
        64,
    )
    assert plan_hole.num_chunks == 1
    assert len(plan_hole.chunks[0].scatter_runs) == 2

    # Second pair: dst contiguous but the 100-byte desc aligns the cursor up
    # to 128, leaving a bounce gap; sizes differ (100 vs 32) -> two runs.
    plan_gap = plan_of(
        [(0x1000, 100, 0), (0x3000, 32, 0)],
        [(0x9000, 100, 0), (0x9064, 32, 0)],
        1024,
        64,
    )
    assert plan_gap.num_chunks == 1
    c = plan_gap.chunks[0]
    assert len(c.scatter_runs) == 2
    assert int(c.scatter_runs[1]["bounce_offset"]) == 128  # align_up(100, 32)


# C++: BounceTransferPlan.ScatterRunsIrregularStrideBreaks
def test_scatter_runs_irregular_stride_breaks() -> None:
    # Same sizes but NON-uniform dst steps (+0x80 then +0x40): the latch fixes
    # stride 0x80; the third desc doesn't land on it -> a new run (2 runs).
    plan = plan_of(
        [(0x1000, 32, 0), (0x3000, 32, 0), (0x5000, 32, 0)],
        [(0x9000, 32, 0), (0x9080, 32, 0), (0x90C0, 32, 0)],
        1024,
        64,
    )
    assert plan.num_chunks == 1
    c = plan.chunks[0]
    assert len(c.scatter_runs) == 2
    assert int(c.scatter_runs[0]["count"]) == 2
    assert int(c.scatter_runs[1]["count"]) == 1
    assert int(c.scatter_runs[1]["dst_addr"]) == 0x90C0


# ---- deterministic coverage of the coalescer's head-steal branches ----
# (_coalesce_scatter_runs in plan.py: the rule-(b) latch head steal, the
# steal-on-extension after a rule-(c) block, and the synthetic-remainder
# re-latch — previously only hit probabilistically by the randomized test.)


def test_scatter_runs_latch_head_steal_from_dense_chain() -> None:
    """Rule-(b) latch head steal.

    A single piece latches onto the HEAD desc of a following multi-desc dense
    chain; only the head joins (count 2), the chain's dense remainder starts
    a fresh run.
    """
    # Descs (all size 32, bounce offsets 0/32/64):
    #   d0: dst 0x9000                      -> single-piece chain 0
    #   d1: dst 0x9100, d2: dst 0x9120      -> dense chain 1 (multi)
    # d0->d1 latches (size 32 == head size, forward): head steal.
    plan = plan_of(
        [(0x1000, 32, 0), (0x2000, 32, 0), (0x3000, 32, 0)],
        [(0x9000, 32, 0), (0x9100, 32, 0), (0x9120, 32, 0)],
        1024,
        64,
    )
    assert plan.num_chunks == 1
    c = plan.chunks[0]
    runs = c.scatter_runs
    assert len(runs) == 2
    # Run 0: the stolen head — count 2, strides fixed from the first pair.
    assert int(runs[0]["bounce_offset"]) == 0
    assert int(runs[0]["dst_addr"]) == 0x9000
    assert int(runs[0]["dst_stride"]) == 0x100
    assert int(runs[0]["bounce_stride"]) == 32
    assert int(runs[0]["piece_size"]) == 32
    assert int(runs[0]["count"]) == 2
    # Run 1: the chain's remainder (d2) as a fresh count-1 run.
    assert int(runs[1]["bounce_offset"]) == 64
    assert int(runs[1]["dst_addr"]) == 0x9120
    assert int(runs[1]["dst_stride"]) == 0
    assert int(runs[1]["bounce_stride"]) == 0
    assert int(runs[1]["piece_size"]) == 32
    assert int(runs[1]["count"]) == 1
    assert expand_runs_to_byte_map(runs) == per_desc_byte_map(c.bounce_offsets, c.dst_ptrs, c.sizes)


def test_scatter_runs_steal_on_extension_after_stride_block() -> None:
    """Steal-on-extension after a rule-(c) stride block.

    A count>=2 strided run whose NEXT chain is a multi-desc dense chain with
    the head landing exactly one stride further — the head joins
    (count += 1), the remainder starts a fresh run.
    """
    # Descs (all size 32, bounce offsets 0/32/64/96/128):
    #   d0..d2: dst 0x9000/0x9100/0x9200    -> three single pieces, stride 0x100
    #   d3: dst 0x9300, d4: dst 0x9320      -> dense chain, head ON the stride
    plan = plan_of(
        [(0x1000 + i * 0x10000, 32, 0) for i in range(5)],
        [
            (0x9000, 32, 0),
            (0x9100, 32, 0),
            (0x9200, 32, 0),
            (0x9300, 32, 0),
            (0x9320, 32, 0),
        ],
        1024,
        64,
    )
    assert plan.num_chunks == 1
    c = plan.chunks[0]
    runs = c.scatter_runs
    assert len(runs) == 2
    # Run 0: d0..d2 plus the stolen head d3 -> count 4 on stride 0x100/32.
    assert int(runs[0]["bounce_offset"]) == 0
    assert int(runs[0]["dst_addr"]) == 0x9000
    assert int(runs[0]["dst_stride"]) == 0x100
    assert int(runs[0]["bounce_stride"]) == 32
    assert int(runs[0]["piece_size"]) == 32
    assert int(runs[0]["count"]) == 4
    # Run 1: the chain's remainder (d4) as a fresh count-1 run.
    assert int(runs[1]["bounce_offset"]) == 128
    assert int(runs[1]["dst_addr"]) == 0x9320
    assert int(runs[1]["piece_size"]) == 32
    assert int(runs[1]["count"]) == 1
    assert expand_runs_to_byte_map(runs) == per_desc_byte_map(c.bounce_offsets, c.dst_ptrs, c.sizes)


def test_scatter_runs_synthetic_remainder_relatches_and_extends() -> None:
    """Synthetic-remainder re-latch.

    After a head steal, the chain's remainder (a SYNTHETIC piece, not indexed
    by the precomputed deltas) latches onto the next single piece and extends
    through the manual first-extension check into a count-3 strided run.
    """
    # Descs (all size 32, bounce offsets 0/32/64/96/128):
    #   d0: dst 0x9000                      -> single piece
    #   d1: dst 0x9100, d2: dst 0x9120      -> dense chain; d1 stolen by d0
    #   d3: dst 0x9320, d4: dst 0x9520      -> stride 0x200 FROM d2 (synthetic)
    plan = plan_of(
        [(0x1000 + i * 0x10000, 32, 0) for i in range(5)],
        [
            (0x9000, 32, 0),
            (0x9100, 32, 0),
            (0x9120, 32, 0),
            (0x9320, 32, 0),
            (0x9520, 32, 0),
        ],
        1024,
        64,
    )
    assert plan.num_chunks == 1
    c = plan.chunks[0]
    runs = c.scatter_runs
    assert len(runs) == 2
    # Run 0: head steal d0+d1 (count 2, stride 0x100/32).
    assert int(runs[0]["dst_addr"]) == 0x9000
    assert int(runs[0]["dst_stride"]) == 0x100
    assert int(runs[0]["bounce_stride"]) == 32
    assert int(runs[0]["count"]) == 2
    # Run 1: the synthetic remainder d2 latches d3 and extends with d4 ->
    # count-3 strided run at 0x200/32 starting from d2.
    assert int(runs[1]["bounce_offset"]) == 64
    assert int(runs[1]["dst_addr"]) == 0x9120
    assert int(runs[1]["dst_stride"]) == 0x200
    assert int(runs[1]["bounce_stride"]) == 32
    assert int(runs[1]["piece_size"]) == 32
    assert int(runs[1]["count"]) == 3
    assert expand_runs_to_byte_map(runs) == per_desc_byte_map(c.bounce_offsets, c.dst_ptrs, c.sizes)


# ---- randomized cross-check: scatter runs == naive per-desc mapping ----


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_scatter_runs_match_per_desc_expectation_randomized(seed: int) -> None:
    """Cross-check the coalesced scatter runs against the per-desc map.

    Expand the runs back to a per-byte map and compare byte-for-byte against
    the naive per-desc expectation over randomized layouts (mix of dense,
    uniformly-strided, and irregular destinations).
    """
    rng = np.random.default_rng(seed)
    n = 200
    sizes = rng.integers(1, 129, size=n).astype(np.uint64)
    # Some dense stretches: force runs of equal sizes to trigger latching.
    for start in range(0, n, 40):
        stretch = min(10, n - start)
        sizes[start : start + stretch] = int(rng.integers(1, 65))

    src = (0x100000 + np.arange(n, dtype=np.uint64) * 0x1000).astype(np.uint64)
    dst = np.zeros(n, dtype=np.uint64)
    cursor = 0x9000_0000
    for i in range(n):
        mode = int(rng.integers(0, 3))
        if mode == 0:  # dense: continue exactly where the previous desc ended
            dst[i] = cursor
        elif mode == 1:  # forward jump by a uniform-ish stride
            dst[i] = cursor + 128
        else:  # irregular jump
            dst[i] = cursor + int(rng.integers(1, 4096))
        cursor = int(dst[i]) + int(sizes[i])

    plan = build_plan(src, dst, sizes, max_chunk_bytes=4096, max_descs_per_chunk=32)
    assert plan.total_bytes == int(sizes.sum())
    assert plan.total_descs == n

    seen_descs = 0
    for c in plan.chunks:
        expected = per_desc_byte_map(c.bounce_offsets, c.dst_ptrs, c.sizes)
        actual = expand_runs_to_byte_map(c.scatter_runs)
        assert actual == expected, "scatter runs disagree with per-desc mapping"
        seen_descs += c.num_descs
        # Runs never read past the packed extent.
        assert max(actual) < c.packed_bytes
    assert seen_descs == n


def test_scatter_runs_match_per_desc_uniform_stride_layout() -> None:
    """Deterministic uniformly-strided layout (the tp-resharding fast path).

    Many equal descs landing every 128B must still expand correctly.
    """
    n = 64
    sizes = np.full(n, 32, dtype=np.uint64)
    src = (0x1000 + np.arange(n, dtype=np.uint64) * 0x2000).astype(np.uint64)
    dst = (0x9000 + np.arange(n, dtype=np.uint64) * 0x80).astype(np.uint64)
    plan = build_plan(src, dst, sizes, max_chunk_bytes=1 << 20, max_descs_per_chunk=1024)
    assert plan.num_chunks == 1
    c = plan.chunks[0]
    assert len(c.scatter_runs) == 1  # one strided run of count 64
    assert int(c.scatter_runs[0]["count"]) == n
    assert expand_runs_to_byte_map(c.scatter_runs) == per_desc_byte_map(
        c.bounce_offsets, c.dst_ptrs, c.sizes
    )


# Python-specific argument validation.
@pytest.mark.parametrize(
    "max_chunk,max_descs",
    [(0, 64), (-1, 64), (1024, 0), (1024, -1)],
    ids=["zero-chunk", "neg-chunk", "zero-descs", "neg-descs"],
)
def test_non_positive_limits_raise(max_chunk: int, max_descs: int) -> None:
    with pytest.raises(ValueError):
        plan_of([(0x1000, 8, 0)], [(0x9000, 8, 0)], max_chunk, max_descs)
