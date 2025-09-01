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

from tensorrt_llm.mapping import Mapping


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
