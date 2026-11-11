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
import unittest
from unittest.mock import patch

import pytest

from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _force_mpi_topology_mapping():
    with patch("tensorrt_llm.mapping.mpi_disabled", return_value=False):
        yield


@pytest.mark.parametrize(
    ("tp_size", "cp_size", "attention_dp", "shares_requests"),
    [
        (1, 4, True, True),  # helix + attention DP at tp 1: one DP group
        (2, 2, True, False),  # attention DP across the 2 original tp ranks
        (2, 2, False, True),  # helix without attention DP
    ],
)
def test_repurposed_helix_mapping_for_full_rows(tp_size, cp_size, attention_dp,
                                                shares_requests):
    world = tp_size * cp_size
    helix = Mapping(
        world_size=world,
        rank=world - 1,
        tp_size=tp_size,
        cp_size=cp_size,
        cp_config={"cp_type": "HELIX"},
        enable_attention_dp=attention_dp,
    )
    repurposed = helix.repurpose_helix_cp_to_tp()
    assert repurposed.helix_repurposed_cp_size == cp_size
    assert repurposed.tp_group_shares_requests is shares_requests
    assert Mapping.from_dict(
        repurposed.to_dict()).helix_repurposed_cp_size == cp_size

    full_rows = repurposed.mapping_for_full_rows()
    if attention_dp and shares_requests:
        assert full_rows is not repurposed
        assert not full_rows.enable_attention_dp
        assert (full_rows.tp_size, full_rows.tp_rank) == (world,
                                                          repurposed.tp_rank)
        assert repurposed.enable_attention_dp  # the source mapping is untouched
    else:
        assert full_rows is repurposed


@pytest.mark.parametrize(("world_size", "pp_size"), [(1, 1), (2, 2)])
def test_plain_attention_dp_at_tp1_keeps_attention_dp(world_size, pp_size):
    # Not helix: attention DP at tp_size 1 must not be read as a shared TP group,
    # or the LM head and the vocab embedding disagree on their TP mode.
    mapping = Mapping(world_size=world_size,
                      rank=0,
                      tp_size=1,
                      pp_size=pp_size,
                      enable_attention_dp=True)
    assert not mapping.tp_group_shares_requests
    assert mapping.mapping_for_full_rows() is mapping


class TestMapping(unittest.TestCase):

    def test_mapping(self):
        m = Mapping(world_size=8, rank=0, tp_size=8)
        self.assertEqual(len(m.tp_groups), 1)
        self.assertEqual(len(m.pp_groups), 8)
        self.assertEqual(m.tp_group, [0, 1, 2, 3, 4, 5, 6, 7])

        m = Mapping(world_size=8, rank=0, tp_size=4, pp_size=2)
        self.assertEqual(len(m.tp_groups), 2)
        self.assertEqual(len(m.pp_groups), 4)
        self.assertEqual(m.tp_group, [0, 1, 2, 3])
        self.assertEqual(m.pp_group, [0, 4])
        self.assertTrue(m.is_first_pp_rank())
        self.assertFalse(m.is_last_pp_rank())
        self.assertEqual(m.prev_pp_rank(), 4)
        self.assertEqual(m.next_pp_rank(), 4)

        m = Mapping(world_size=8, rank=6, tp_size=2, pp_size=4)
        self.assertEqual(len(m.tp_groups), 4)
        self.assertEqual(len(m.pp_groups), 2)
        self.assertEqual(m.tp_group, [6, 7])
        self.assertEqual(m.pp_group, [0, 2, 4, 6])
        self.assertFalse(m.is_first_pp_rank())
        self.assertTrue(m.is_last_pp_rank())
        self.assertEqual(m.prev_pp_rank(), 4)
        self.assertEqual(m.next_pp_rank(), 0)

        m = Mapping(world_size=2, rank=0, cp_size=2)
        self.assertEqual(len(m.tp_groups), 2)
        self.assertEqual(len(m.pp_groups), 2)
        self.assertEqual(len(m.cp_groups), 1)
        self.assertEqual(m.tp_group, [0])
        self.assertEqual(m.pp_group, [0])
        self.assertEqual(m.cp_group, [0, 1])

        m = Mapping(world_size=8, rank=3, tp_size=2, pp_size=2, cp_size=2)
        self.assertEqual(len(m.tp_groups), 4)
        self.assertEqual(len(m.pp_groups), 4)
        self.assertEqual(len(m.cp_groups), 4)
        self.assertEqual(m.tp_group, [1, 3])
        self.assertEqual(m.pp_group, [3, 7])
        self.assertEqual(m.cp_group, [2, 3])
        self.assertTrue(m.is_first_pp_rank())
        self.assertFalse(m.is_last_pp_rank())
        self.assertFalse(m.is_first_cp_rank())
        self.assertTrue(m.is_last_cp_rank())
        self.assertEqual(m.prev_pp_rank(), 7)
        self.assertEqual(m.next_pp_rank(), 7)
        self.assertEqual(m.prev_cp_rank(), 2)
        self.assertEqual(m.next_cp_rank(), 2)

        m = Mapping(world_size=16, rank=9, tp_size=2, pp_size=2, cp_size=4)
        self.assertEqual(m.tp_group, [9, 13])
        self.assertEqual(m.pp_group, [1, 9])
        self.assertEqual(m.cp_group, [8, 9, 10, 11])
        self.assertFalse(m.is_first_pp_rank())
        self.assertTrue(m.is_last_pp_rank())
        self.assertFalse(m.is_first_cp_rank())
        self.assertFalse(m.is_last_cp_rank())
        self.assertEqual(m.prev_pp_rank(), 1)
        self.assertEqual(m.next_pp_rank(), 1)
        self.assertEqual(m.prev_cp_rank(), 8)
        self.assertEqual(m.next_cp_rank(), 10)
