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
"""Unit tests for ``compute_peer_ranges`` and ``lookup_owner``.

These two helpers are the core of Phase 2 (non-uniform / redundant
partition support).  They replace the legacy
``cursor // experts_per_rank`` formula used everywhere a remote expert
needed to be mapped to its owning peer rank.

Cases covered:

  * Uniform partition (size == stride == num_experts // dwdp_size):
    behavior must match the legacy ``//`` formula exactly.
  * Non-uniform tail padding (size * dwdp_size > num_experts): the tail
    rank's valid range is capped at ``num_experts``.
  * Redundancy (size > stride): adjacent ranges overlap;
    ``lookup_owner`` picks the lowest-rank owner (deterministic).
  * Boundary expert ids (0, num_experts - 1).
  * Out-of-range expert ids raise.
"""

import unittest

from tensorrt_llm._torch.modules.dwdp.specs import compute_peer_ranges, lookup_owner


class TestComputePeerRanges(unittest.TestCase):
    def test_uniform_dwdp4_matches_floor_div(self):
        # 256 experts / 4 ranks = 64 each.  Every rank's range starts at
        # rank * 64 and ends at the next rank's start.
        peer_ranges = compute_peer_ranges(
            dwdp_size=4,
            num_experts_per_worker=64,
            num_prefetch_experts=64,
            num_experts_total=256,
        )
        self.assertEqual(peer_ranges, [(0, 64), (64, 128), (128, 192), (192, 256)])

    def test_uniform_dwdp8_cross_tray(self):
        peer_ranges = compute_peer_ranges(
            dwdp_size=8,
            num_experts_per_worker=32,
            num_prefetch_experts=32,
            num_experts_total=256,
        )
        for r, (start, end) in enumerate(peer_ranges):
            self.assertEqual(start, r * 32)
            self.assertEqual(end, (r + 1) * 32)

    def test_non_uniform_tail_padding_capped(self):
        # Defensive: the function caps the last rank's end at
        # ``num_experts_total`` even when ``(dwdp-1)*stride + size >
        # num_experts``.  In production this configuration is rejected
        # upstream by ``_validate_partition_config`` (tail padding is
        # incompatible with GB200 fabric-handle partial mapping), but
        # ``compute_peer_ranges`` itself stays robust.
        peer_ranges = compute_peer_ranges(
            dwdp_size=3,
            num_experts_per_worker=86,
            num_prefetch_experts=86,
            num_experts_total=256,
        )
        self.assertEqual(peer_ranges, [(0, 86), (86, 172), (172, 256)])

    def test_dwdp5_tail_padding_capped(self):
        # Defensive (rejected upstream): see
        # ``test_non_uniform_tail_padding_capped``.
        peer_ranges = compute_peer_ranges(
            dwdp_size=5,
            num_experts_per_worker=52,
            num_prefetch_experts=52,
            num_experts_total=256,
        )
        # rank 4: start 208, end 260 capped to 256.
        self.assertEqual(peer_ranges[-1], (208, 256))

    def test_mode_b_dwdp3_overlap(self):
        # dwdp=3, 256 experts, Mode B: size=86, stride=85 — the
        # production recipe for ``num_experts not divisible by dwdp_size``.
        # ``(dwdp-1)*stride + size == num_experts`` exactly, with 1
        # expert overlapping between adjacent ranks.
        peer_ranges = compute_peer_ranges(
            dwdp_size=3,
            num_experts_per_worker=86,
            num_prefetch_experts=85,
            num_experts_total=256,
        )
        self.assertEqual(peer_ranges, [(0, 86), (85, 171), (170, 256)])

    def test_mode_b_dwdp7_larger_overlap(self):
        # dwdp=7, 256 experts: ``num_experts % dwdp = 4`` so Mode B
        # requires more overlap.  size=40, stride=36: 6*36+40=256.
        peer_ranges = compute_peer_ranges(
            dwdp_size=7,
            num_experts_per_worker=40,
            num_prefetch_experts=36,
            num_experts_total=256,
        )
        # All ranks have valid range = 40 (last rank ends at exactly 256).
        self.assertEqual(peer_ranges[-1], (216, 256))
        for r, (start, end) in enumerate(peer_ranges):
            self.assertEqual(start, r * 36)
            self.assertEqual(end, min(r * 36 + 40, 256))

    def test_redundancy_overlap(self):
        # dwdp=4, size=70, stride=62: 8-expert overlap between adjacent
        # ranks.  Last rank's valid end is min(186 + 70, 256) = 256.
        peer_ranges = compute_peer_ranges(
            dwdp_size=4,
            num_experts_per_worker=70,
            num_prefetch_experts=62,
            num_experts_total=256,
        )
        self.assertEqual(peer_ranges, [(0, 70), (62, 132), (124, 194), (186, 256)])
        # Verify overlap exists between every adjacent pair.
        for r in range(len(peer_ranges) - 1):
            cur_end = peer_ranges[r][1]
            next_start = peer_ranges[r + 1][0]
            self.assertGreater(
                cur_end, next_start, f"Expected overlap between rank {r} and {r + 1}"
            )


class TestLookupOwner(unittest.TestCase):
    def test_uniform_matches_floor_div(self):
        # The whole point: lookup_owner must agree with the
        # legacy formula on every uniform expert id.
        peer_ranges = compute_peer_ranges(
            dwdp_size=4,
            num_experts_per_worker=64,
            num_prefetch_experts=64,
            num_experts_total=256,
        )
        for expert_id in range(256):
            self.assertEqual(
                lookup_owner(expert_id, peer_ranges),
                expert_id // 64,
                f"mismatch at expert_id={expert_id}",
            )

    def test_mode_b_dwdp3_overlap(self):
        # dwdp=3, 256 experts, Mode B (size=86, stride=85): production
        # recipe for non-divisible dwdp_size.  Adjacent ranks overlap by
        # 1 expert; lookup_owner picks the lowest-rank owner for shared
        # experts (deterministic).
        peer_ranges = compute_peer_ranges(
            dwdp_size=3,
            num_experts_per_worker=86,
            num_prefetch_experts=85,
            num_experts_total=256,
        )
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

    def test_redundancy_picks_lowest_rank(self):
        # dwdp=4, size=70, stride=62: experts in the overlap zone
        # [62, 70) are owned by both rank 0 and rank 1; lookup_owner
        # must consistently return the lowest rank (0) for these.
        peer_ranges = compute_peer_ranges(
            dwdp_size=4,
            num_experts_per_worker=70,
            num_prefetch_experts=62,
            num_experts_total=256,
        )
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

    def test_out_of_range_raises(self):
        peer_ranges = compute_peer_ranges(
            dwdp_size=2,
            num_experts_per_worker=4,
            num_prefetch_experts=4,
            num_experts_total=8,
        )
        with self.assertRaises(ValueError):
            lookup_owner(8, peer_ranges)  # past the end
        with self.assertRaises(ValueError):
            lookup_owner(99, peer_ranges)

    def test_negative_id_raises(self):
        peer_ranges = compute_peer_ranges(
            dwdp_size=2,
            num_experts_per_worker=4,
            num_prefetch_experts=4,
            num_experts_total=8,
        )
        with self.assertRaises(ValueError):
            lookup_owner(-1, peer_ranges)


if __name__ == "__main__":
    unittest.main()
