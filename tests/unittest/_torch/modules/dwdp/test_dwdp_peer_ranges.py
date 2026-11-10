# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for ``ExpertPartition``, ``handle_geometry`` and ``lookup_owner``.

0/0 is the EP partition of the MoE modules (balanced); otherwise a strided table whose last range
ends at N (uniform, redundant) or is cut at N (capped).  ``lookup_owner`` gives the lowest owner.
"""

import itertools
import unittest

import pytest

from tensorrt_llm._torch.modules.dwdp.specs import (
    ExpertPartition,
    PageAlignedLayout,
    handle_geometry,
    lookup_owner,
)
from tensorrt_llm._torch.moe.fused_moe.interface import _compute_ep_partition

pytestmark = pytest.mark.cpu_only

K3_EXPERTS = 896


def _from_config(num_experts: int, dwdp_size: int, size: int, stride: int) -> ExpertPartition:
    return ExpertPartition.from_config(
        num_experts=num_experts,
        dwdp_size=dwdp_size,
        num_experts_per_worker=size,
        num_prefetch_experts=stride,
    )


@pytest.mark.parametrize("num_experts", [7, 72, 73, 128, 160, 256, 384, K3_EXPERTS])
def test_balanced_partition_is_the_ep_partition(num_experts: int) -> None:
    for dwdp_size in range(2, 33):
        if num_experts < dwdp_size:
            with pytest.raises(ValueError):
                ExpertPartition.balanced(num_experts=num_experts, dwdp_size=dwdp_size)
            continue
        partition = ExpertPartition.balanced(num_experts=num_experts, dwdp_size=dwdp_size)
        assert partition.ranges == tuple(
            _compute_ep_partition(num_experts, dwdp_size, rank)[1:] for rank in range(dwdp_size)
        )


def test_strided_partition_accepts_exact_and_capped_tables() -> None:
    """Every (size, stride) on a grid: accepted iff exact or capped, with the strided ranges."""
    mismatches = []
    for num_experts, dwdp_size in itertools.product((8, 9, 10, 12, 72, 96), (2, 3, 4, 5)):
        for size, stride in itertools.product(range(1, num_experts + 2), repeat=2):
            try:
                partition = ExpertPartition.strided(
                    num_experts=num_experts, dwdp_size=dwdp_size, size=size, stride=stride
                )
                got = (partition.kind, partition.peer_ranges)
            except ValueError:
                got = None
            # Accepted: ranges that end exactly at N (stride <= size), or a capped table.
            exact = stride <= size and (dwdp_size - 1) * stride + size == num_experts
            capped = stride == size and (dwdp_size - 1) * size < num_experts < dwdp_size * size
            want = None
            if exact or capped:
                kind = "capped" if capped else ("uniform" if size == stride else "redundant")
                ranges = [
                    (r * stride, min(r * stride + size, num_experts)) for r in range(dwdp_size)
                ]
                want = (kind, ranges)
            if got != want:
                mismatches.append((num_experts, dwdp_size, size, stride))
    assert mismatches == []


@pytest.mark.parametrize(
    "num_experts, dwdp_size, size, stride, counts",
    [
        (K3_EXPERTS, 12, 0, 0, [75] * 8 + [74] * 4),
        (K3_EXPERTS, 12, 82, 74, [82] * 12),
        (72, 2, 40, 40, [40, 32]),
        (K3_EXPERTS, 12, 75, 75, [75] * 11 + [71]),
        # Rejected: the last rank would store nothing, a gap, overlap past the experts, one field
        # set, negative fields, fewer experts than ranks.
        (K3_EXPERTS, 33, 28, 28, None),
        (72, 10, 8, 8, None),
        (K3_EXPERTS, 12, 74, 75, None),
        (K3_EXPERTS, 12, 83, 74, None),
        (K3_EXPERTS, 12, 0, 74, None),
        (K3_EXPERTS, 12, 82, 0, None),
        (K3_EXPERTS, 12, -1, -1, None),
        (7, 8, 0, 0, None),
    ],
)
def test_partition_from_config(
    num_experts: int, dwdp_size: int, size: int, stride: int, counts: list[int] | None
) -> None:
    if counts is None:
        with pytest.raises(ValueError):
            _from_config(num_experts, dwdp_size, size, stride)
        return
    partition = _from_config(num_experts, dwdp_size, size, stride)
    assert [partition.count_of(rank) for rank in range(dwdp_size)] == counts


@pytest.mark.parametrize(
    "ranges",
    [((0, 4), (5, 8)), ((1, 4), (4, 8)), ((0, 4), (4, 7)), ((0, 4), (4, 4), (4, 8))],
    ids=["gap", "late_start", "short_end", "empty_range"],
)
def test_partition_refuses_an_expert_without_owner_or_a_rank_without_expert(
    ranges: tuple[tuple[int, int], ...],
) -> None:
    with pytest.raises(ValueError):
        ExpertPartition(num_experts=8, ranges=ranges, kind="custom")


@pytest.mark.parametrize("expert_bytes", [11010048, 5505024, 1376256, 688128])
def test_handle_geometry_is_the_page_aligned_layout_of_the_range(expert_bytes: int) -> None:
    """The span ``WeightBuffer`` maps whole, at its data offset; Kimi-K3's four expert weights."""
    page = 2 << 20
    for partition in (_from_config(K3_EXPERTS, 12, 0, 0), _from_config(K3_EXPERTS, 12, 82, 74)):
        for start, end in partition.ranges:
            layout = PageAlignedLayout.compute(expert_bytes, K3_EXPERTS, start, end, page, 1 << 62)
            geometry = (layout.mnnvl_size, layout.data_offset)
            assert handle_geometry(expert_bytes, start, end, page) == geometry


class TestLookupOwner(unittest.TestCase):
    def test_uniform_matches_floor_div(self) -> None:
        # The whole point: lookup_owner must agree with the
        # legacy formula on every uniform expert id.
        peer_ranges = _from_config(256, 4, 64, 64).peer_ranges
        for expert_id in range(256):
            self.assertEqual(
                lookup_owner(expert_id, peer_ranges),
                expert_id // 64,
                f"mismatch at expert_id={expert_id}",
            )

    def test_mode_b_dwdp3_overlap(self) -> None:
        # dwdp=3, 256 experts, Mode B (size=86, stride=85): production
        # recipe for non-divisible dwdp_size.  Adjacent ranks overlap by
        # 1 expert; lookup_owner picks the lowest-rank owner for shared
        # experts (deterministic).
        peer_ranges = _from_config(256, 3, 86, 85).peer_ranges
        # Rank 0's exclusive range
        self.assertEqual(lookup_owner(0, peer_ranges), 0)
        self.assertEqual(lookup_owner(84, peer_ranges), 0)
        # Boundary expert 85: in both rank 0 (0..85] inclusive end-1)
        # and rank 1 (start at 85). lookup_owner picks lowest = rank 0.
        self.assertEqual(lookup_owner(85, peer_ranges), 0)
        # Rank 1's exclusive range
        self.assertEqual(lookup_owner(86, peer_ranges), 1)
        self.assertEqual(lookup_owner(169, peer_ranges), 1)
        # Boundary 170 in rank 1 and rank 2 → rank 1.
        self.assertEqual(lookup_owner(170, peer_ranges), 1)
        # Rank 2's exclusive range
        self.assertEqual(lookup_owner(171, peer_ranges), 2)
        self.assertEqual(lookup_owner(255, peer_ranges), 2)

    def test_redundancy_picks_lowest_rank(self) -> None:
        # dwdp=4, size=70, stride=62: experts in the overlap zone
        # [62, 70) are owned by both rank 0 and rank 1; lookup_owner
        # must consistently return the lowest rank (0) for these.
        peer_ranges = _from_config(256, 4, 70, 62).peer_ranges
        # Experts in rank 0's exclusive range
        self.assertEqual(lookup_owner(0, peer_ranges), 0)
        self.assertEqual(lookup_owner(61, peer_ranges), 0)
        # Overlap zone [62, 70): both rank 0 and rank 1 have it.
        # First-match → rank 0.
        for expert_id in range(62, 70):
            self.assertEqual(
                lookup_owner(expert_id, peer_ranges),
                0,
                f"overlap should pick lowest peer at {expert_id}",
            )
        # Past rank 0's range → rank 1 (or its overlap with rank 2)
        self.assertEqual(lookup_owner(70, peer_ranges), 1)
        self.assertEqual(lookup_owner(123, peer_ranges), 1)  # in rank 1's exclusive
        # Overlap [124, 132): rank 1 vs rank 2 → rank 1
        for expert_id in range(124, 132):
            self.assertEqual(lookup_owner(expert_id, peer_ranges), 1)


if __name__ == "__main__":
    unittest.main()
