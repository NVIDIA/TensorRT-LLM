# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections import namedtuple

from tensorrt_llm.mapping import CpType, Mapping


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
        self.assertEqual(m.tp_group, [2, 3])
        self.assertEqual(m.pp_group, [3, 7])
        self.assertEqual(m.cp_group, [1, 3])
        self.assertTrue(m.is_first_pp_rank())
        self.assertFalse(m.is_last_pp_rank())
        self.assertFalse(m.is_first_cp_rank())
        self.assertTrue(m.is_last_cp_rank())
        self.assertEqual(m.prev_pp_rank(), 7)
        self.assertEqual(m.next_pp_rank(), 7)
        self.assertEqual(m.prev_cp_rank(), 1)
        self.assertEqual(m.next_cp_rank(), 1)

        m = Mapping(world_size=16, rank=9, tp_size=2, pp_size=2, cp_size=4)
        self.assertEqual(m.tp_group, [8, 9])
        self.assertEqual(m.pp_group, [1, 9])
        self.assertEqual(m.cp_group, [9, 11, 13, 15])
        self.assertFalse(m.is_first_pp_rank())
        self.assertTrue(m.is_last_pp_rank())
        self.assertTrue(m.is_first_cp_rank())
        self.assertFalse(m.is_last_cp_rank())
        self.assertEqual(m.prev_pp_rank(), 1)
        self.assertEqual(m.next_pp_rank(), 1)
        self.assertEqual(m.prev_cp_rank(), 15)
        self.assertEqual(m.next_cp_rank(), 11)

    def test_helix_overridden_tp_rank(self):
        # Test case for helix overridden TP rank: (pp_size, tp_size, cp_size, expected_mapping)
        # where expected_mapping is a list of (rank, expected_helix_tp_rank) tuples.
        HelixTestCase = namedtuple(
            'HelixTestCase',
            ['pp_size', 'tp_size', 'cp_size', 'expected_mapping'])
        test_cases = [
            # Case: pp_size=1, tp_size=2, cp_size=2.
            # CP groups: [0, 2], [1, 3] -> helix order: [0, 2, 1, 3].
            HelixTestCase(pp_size=1,
                          tp_size=2,
                          cp_size=2,
                          expected_mapping=[
                              (0, 0),
                              (2, 1),
                              (1, 2),
                              (3, 3),
                          ]),
            # Case: pp_size=1, tp_size=4, cp_size=2.
            # CP groups: [0, 4], [1, 5], [2, 6], [3, 7] -> helix order: [0, 4, 1, 5, 2, 6, 3, 7].
            HelixTestCase(pp_size=1,
                          tp_size=4,
                          cp_size=2,
                          expected_mapping=[
                              (0, 0),
                              (4, 1),
                              (1, 2),
                              (5, 3),
                              (2, 4),
                              (6, 5),
                              (3, 6),
                              (7, 7),
                          ]),
            # Case: pp_size=1, tp_size=2, cp_size=4.
            # CP groups: [0, 2, 4, 6], [1, 3, 5, 7] -> helix order: [0, 2, 4, 6, 1, 3, 5, 7].
            HelixTestCase(pp_size=1,
                          tp_size=2,
                          cp_size=4,
                          expected_mapping=[
                              (0, 0),
                              (2, 1),
                              (4, 2),
                              (6, 3),
                              (1, 4),
                              (3, 5),
                              (5, 6),
                              (7, 7),
                          ]),
            # Case: pp_size=1, tp_size=4, cp_size=4.
            # CP groups: [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15] -> helix order: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15].
            HelixTestCase(pp_size=1,
                          tp_size=4,
                          cp_size=4,
                          expected_mapping=[
                              (0, 0),
                              (4, 1),
                              (8, 2),
                              (12, 3),
                              (1, 4),
                              (5, 5),
                              (9, 6),
                              (13, 7),
                              (2, 8),
                              (6, 9),
                              (10, 10),
                              (14, 11),
                              (3, 12),
                              (7, 13),
                              (11, 14),
                              (15, 15),
                          ]),
            # Case: pp_size=2, tp_size=4, cp_size=2.
            # PP stage 0 CP groups: [0,4], [1,5], [2,6], [3,7] -> helix order: [0, 4, 1, 5, 2, 6, 3, 7].
            # PP stage 1 CP groups: [8,12], [9,13], [10,14], [11,15] -> helix order: [8, 12, 9, 13, 10, 14, 11, 15].
            HelixTestCase(
                pp_size=2,
                tp_size=4,
                cp_size=2,
                expected_mapping=[
                    (0, 0),
                    (4, 1),
                    (1, 2),
                    (5, 3),
                    (2, 4),
                    (6, 5),
                    (3, 6),
                    (7, 7),  # PP stage 0
                    (8, 0),
                    (12, 1),
                    (9, 2),
                    (13, 3),
                    (10, 4),
                    (14, 5),
                    (11, 6),
                    (15, 7),  # PP stage 1
                ]),
            # Case: pp_size=2, tp_size=2, cp_size=4.
            # PP stage 0 CP groups: [0, 2, 4, 6], [1, 3, 5, 7] -> helix order: [0, 2, 4, 6, 1, 3, 5, 7].
            # PP stage 1 CP groups: [8, 10, 12, 14], [9, 11, 13, 15] -> helix order: [8, 10, 12, 14, 9, 11, 13, 15].
            HelixTestCase(
                pp_size=2,
                tp_size=2,
                cp_size=4,
                expected_mapping=[
                    (0, 0),
                    (2, 1),
                    (4, 2),
                    (6, 3),
                    (1, 4),
                    (3, 5),
                    (5, 6),
                    (7, 7),  # PP stage 0
                    (8, 0),
                    (10, 1),
                    (12, 2),
                    (14, 3),
                    (9, 4),
                    (11, 5),
                    (13, 6),
                    (15, 7),  # PP stage 1
                ]),
        ]

        for case in test_cases:
            world_size = case.pp_size * case.tp_size * case.cp_size
            with self.subTest(pp_size=case.pp_size,
                              tp_size=case.tp_size,
                              cp_size=case.cp_size):
                for rank, expected in case.expected_mapping:
                    m = Mapping(world_size=world_size,
                                rank=rank,
                                tp_size=case.tp_size,
                                pp_size=case.pp_size,
                                cp_size=case.cp_size,
                                cp_config={"cp_type": CpType.HELIX})
                    self.assertEqual(
                        m.get_helix_overridden_tp_rank(), expected,
                        f"Failed for rank={rank}: expected {expected}, got {m.get_helix_overridden_tp_rank()}"
                    )
