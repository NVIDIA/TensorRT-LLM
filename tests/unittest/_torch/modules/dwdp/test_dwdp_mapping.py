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
"""Unit tests for DWDP (Distributed Weight Data Parallelism) extensions to Mapping.

Covers:
- Constructor validation (dwdp_size must be 0 or >=2; dwdp_rank range)
- dwdp_enabled / dwdp_size / dwdp_rank accessors
- DWDP override of moe_tp_size / moe_ep_size / moe_cluster_size to 1
  (DWDP takes ownership of the expert layout via
  ``_init_dwdp_expert_layout``; the fused MoE backend sees a single
  full-table partition)
- moe_ep_rank is always 0 when DWDP is enabled
- to_dict/from_dict round-trip preserves DWDP fields
- __eq__ and __hash__ include DWDP fields
"""

import unittest

from tensorrt_llm.mapping import Mapping


class TestMappingDwdp(unittest.TestCase):
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_enabled_basic(self):
        """dwdp_size=4 gives an enabled mapping with DWDP-managed MoE parallelism."""
        m = Mapping(world_size=4, rank=2, tp_size=4, dwdp_size=4, dwdp_rank=2)
        self.assertTrue(m.dwdp_enabled)
        self.assertEqual(m.dwdp_size, 4)
        self.assertEqual(m.dwdp_rank, 2)

    # ------------------------------------------------------------------
    # MoE parallelism override
    # ------------------------------------------------------------------

    def test_override_moe_parallelism(self):
        """DWDP enabled: moe_tp_size=1, moe_ep_size=1, cluster=1.

        ``moe_ep_size`` is forced to 1 so the fused-MoE backend bypasses
        ``num_experts % ep_size == 0`` and sees a single (full) expert
        table; the rank-specific layout is installed by
        ``_init_dwdp_expert_layout`` from ``DwdpManager``.
        """
        m = Mapping(world_size=4, rank=0, tp_size=4, dwdp_size=4, dwdp_rank=0)
        self.assertEqual(m.moe_tp_size, 1)
        self.assertEqual(m.moe_ep_size, 1)
        self.assertEqual(m.moe_cluster_size, 1)

    def test_override_beats_explicit_moe_values(self):
        """Even if user explicitly passes moe_tp/ep sizes, DWDP override wins."""
        m = Mapping(
            world_size=4,
            rank=0,
            tp_size=4,
            moe_tp_size=2,
            moe_ep_size=2,
            moe_cluster_size=1,
            dwdp_size=4,
            dwdp_rank=1,
        )
        self.assertEqual(m.moe_tp_size, 1)
        self.assertEqual(m.moe_ep_size, 1)
        self.assertEqual(m.moe_cluster_size, 1)

    def test_non_dwdp_leaves_moe_alone(self):
        """When DWDP disabled, existing MoE defaulting logic is preserved."""
        m = Mapping(
            world_size=4, rank=0, tp_size=4, moe_tp_size=2, moe_ep_size=2, moe_cluster_size=1
        )
        self.assertEqual(m.moe_tp_size, 2)
        self.assertEqual(m.moe_ep_size, 2)

    def test_moe_tp_cluster_ep_size_dwdp(self):
        """moe_tp_cluster_ep_size collapses to moe_cluster_size with DWDP.

        With ``moe_ep_size = moe_tp_size = 1`` under DWDP, the legacy
        composite product reduces to ``moe_cluster_size`` (the DWDP
        group is orthogonal to the MoE TP/EP world).
        """
        m = Mapping(world_size=4, rank=0, tp_size=4, dwdp_size=4, dwdp_rank=0)
        self.assertEqual(m.moe_tp_cluster_ep_size, 1)

    def test_dwdp_supports_non_divisible_size(self):
        """DWDP must accept dwdp_size values that don't divide num_experts.

        Pre-Phase-1 the override forced ``moe_ep_size = dwdp_size``,
        which then triggered ``num_experts % ep_size == 0`` deep in the
        fused MoE backend.  The Phase 1 contract is that ``Mapping``
        construction with dwdp_size=3 / 5 is unconditionally valid; the
        partition is later validated against the concrete
        ``num_experts`` by ``DwdpManager``/``setup_dwdp``.
        """
        for dwdp_size in (3, 5):
            for dwdp_rank in range(dwdp_size):
                m = Mapping(
                    world_size=1, rank=0, tp_size=1, dwdp_size=dwdp_size, dwdp_rank=dwdp_rank
                )
                self.assertEqual(m.moe_ep_size, 1)
                self.assertEqual(m.dwdp_size, dwdp_size)
                self.assertEqual(m.dwdp_rank, dwdp_rank)

    # ------------------------------------------------------------------
    # moe_ep_rank conditional branch
    # ------------------------------------------------------------------

    def test_moe_ep_rank_zero_when_dwdp_enabled(self):
        """DWDP on: moe_ep_rank is always 0 (matches moe_ep_size = 1).

        The actual rank-specific expert range comes from
        ``DwdpManager.start_expert_id`` via ``_init_dwdp_expert_layout``,
        not from ``moe_ep_rank``.
        """
        for dwdp_rank in range(4):
            m = Mapping(world_size=4, rank=dwdp_rank, tp_size=4, dwdp_size=4, dwdp_rank=dwdp_rank)
            self.assertEqual(m.moe_ep_rank, 0)

    def test_moe_ep_rank_uses_tp_rank_when_disabled(self):
        """DWDP off: moe_ep_rank falls back to tp_rank % moe_ep_size."""
        m = Mapping(
            world_size=4, rank=0, tp_size=4, moe_tp_size=1, moe_ep_size=4, moe_cluster_size=1
        )
        self.assertEqual(m.moe_ep_rank, m.tp_rank % m.moe_ep_size)

    # ------------------------------------------------------------------
    # to_dict / from_dict round-trip
    # ------------------------------------------------------------------

    def test_to_dict_includes_dwdp(self):
        m = Mapping(world_size=4, rank=1, tp_size=4, dwdp_size=4, dwdp_rank=1)
        d = m.to_dict()
        self.assertIn("dwdp_size", d)
        self.assertIn("dwdp_rank", d)
        self.assertEqual(d["dwdp_size"], 4)
        self.assertEqual(d["dwdp_rank"], 1)

    def test_roundtrip_preserves_dwdp(self):
        m1 = Mapping(world_size=4, rank=1, tp_size=4, dwdp_size=4, dwdp_rank=1)
        m2 = Mapping(**m1.to_dict())
        self.assertEqual(m1, m2)
        self.assertEqual(m2.dwdp_size, 4)
        self.assertEqual(m2.dwdp_rank, 1)
        self.assertTrue(m2.dwdp_enabled)

    def test_roundtrip_disabled(self):
        m1 = Mapping(world_size=4, rank=0, tp_size=4)
        m2 = Mapping(**m1.to_dict())
        self.assertEqual(m1, m2)
        self.assertFalse(m2.dwdp_enabled)

    # ------------------------------------------------------------------
    # __eq__ / __hash__ include DWDP fields
    # ------------------------------------------------------------------

    def test_eq_distinguishes_dwdp_rank(self):
        """Two otherwise-identical mappings with different dwdp_rank are not equal."""
        m1 = Mapping(world_size=4, rank=0, tp_size=4, dwdp_size=4, dwdp_rank=0)
        m2 = Mapping(world_size=4, rank=0, tp_size=4, dwdp_size=4, dwdp_rank=1)
        self.assertNotEqual(m1, m2)

    def test_eq_distinguishes_dwdp_enabled(self):
        m_on = Mapping(world_size=4, rank=0, tp_size=4, dwdp_size=4, dwdp_rank=0)
        m_off = Mapping(world_size=4, rank=0, tp_size=4)
        self.assertNotEqual(m_on, m_off)

    def test_hash_distinguishes_dwdp(self):
        """Hash must differ for different dwdp configurations.

        Probabilistic but deterministic here since we only tweak dwdp_rank.
        """
        m1 = Mapping(world_size=4, rank=0, tp_size=4, dwdp_size=4, dwdp_rank=0)
        m2 = Mapping(world_size=4, rank=0, tp_size=4, dwdp_size=4, dwdp_rank=1)
        self.assertNotEqual(hash(m1), hash(m2))


if __name__ == "__main__":
    unittest.main()
