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

import torch

from tensorrt_llm.layers import GroupNorm
from tensorrt_llm.module import Module, ModuleList


class Module1(Module):

    def __init__(self, name):
        super(Module1, self).__init__()
        self.name = name

    def forward(self):
        self.register_network_output('o1', 1)


class Module2(Module):

    def __init__(self):
        super(Module2, self).__init__()
        self.name = 'module2'
        self.m1 = Module1('m1')
        self.m2 = Module1('m2')

    def forward(self):
        self.m1.forward()
        self.m2.forward()
        self.register_network_output('o2', 2)
        self.register_network_output('o3', 3)


class Module3(Module):

    def __init__(self):
        super(Module3, self).__init__()
        self.name = 'module3'
        self.m1 = Module2()

    def forward(self):
        self.m1.forward()
        self.register_network_output('o4', 4)


class Module4(Module):

    def __init__(self):
        super(Module4, self).__init__()
        self.layers = ModuleList([Module2(), Module2()])

    def forward(self):
        for l in self.layers:
            l.forward()


class TestModule(unittest.TestCase):

    def test_module(self):
        m = Module3()
        m.forward()

        self.assertEqual(4, len(list(m.named_modules())))
        self.assertEqual(5, len(list(m.named_network_outputs())))

    def test_module_list(self):
        m = Module4()
        m.forward()

        self.assertEqual(8, len(list(m.named_modules())))
        self.assertEqual(8, len(list(m.named_network_outputs())))

    def test_module_named_parameter(self):
        m = GroupNorm(2, 4)
        md = {k: v for k, v in m.named_parameters()}

        tm = torch.nn.GroupNorm(2, 4)
        tmd = {k: v for k, v in tm.named_parameters()}

        self.assertEqual(len(md), len(tmd))

        for k, _ in md.items():
            self.assertIn(k, tmd)

        for k, _ in tmd.items():
            self.assertIn(k, md)
