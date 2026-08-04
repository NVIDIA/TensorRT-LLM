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

import numpy as np
import torch
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner
from torch import nn

import tensorrt_llm
from tensorrt_llm import Module, Tensor


class TorchMLP(nn.Module):

    def __init__(self, hidden_size, ffn_hidden_size, bias=True):
        super().__init__()
        self.fc = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        self.proj = nn.Linear(ffn_hidden_size, hidden_size, bias=bias)

    def forward(self, hidden_states):
        inter = self.fc(hidden_states)
        inter = nn.functional.relu(inter)
        output = self.proj(inter)
        return output, inter


class MLP(Module):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 bias=True,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.fc = tensorrt_llm.layers.ColumnLinear(hidden_size,
                                                   ffn_hidden_size,
                                                   bias=bias,
                                                   tp_group=tp_group,
                                                   tp_size=tp_size,
                                                   gather_output=False)
        self.proj = tensorrt_llm.layers.RowLinear(ffn_hidden_size,
                                                  hidden_size,
                                                  bias=bias,
                                                  tp_group=tp_group,
                                                  tp_size=tp_size)

    def forward(self, hidden_states):
        inter = self.fc(hidden_states)
        inter = tensorrt_llm.functional.relu(inter)
        self.register_network_output('inter', inter)
        output = self.proj(inter)
        return output


class TestDebuggingAPI(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_debugging_api(self):
        # test data
        dtype = 'float32'
        hidden_size = 768
        x_data = torch.randn(2, 16, hidden_size)

        tm = TorchMLP(hidden_size=hidden_size,
                      ffn_hidden_size=hidden_size * 4,
                      bias=False)

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            gm = MLP(hidden_size=hidden_size,
                     ffn_hidden_size=4 * hidden_size,
                     bias=False)
            gm.fc.weight.value = tm.fc.weight.detach().cpu().numpy()
            gm.proj.weight.value = tm.proj.weight.detach().cpu().numpy()

            output = gm.forward(x)
            net._mark_output(output, 'output',
                             tensorrt_llm.str_dtype_to_trt(dtype))

            for k, v in gm.named_network_outputs():
                net._mark_output(v, k, tensorrt_llm.str_dtype_to_trt(dtype))

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        # pytorch run
        with torch.no_grad():
            ref1, ref2 = tm(x_data)

        # compare diff
        np.testing.assert_allclose(ref1.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)
        np.testing.assert_allclose(ref2.cpu().numpy(),
                                   outputs['inter'],
                                   atol=1e-5)
