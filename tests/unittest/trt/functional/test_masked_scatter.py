# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner
from utils.util import unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([
        ('int32', (1, )),
        ('int32', (256, )),
        ('int32', (256, )),
        ('float32', (3, 16)),
        ('float32', (3, 16)),
        ('float32', (3, 16)),
        ('float16', (5, 6, 8)),
        ('float16', (5, 6, 8)),
        ('float16', (5, 6, 8)),
    ],
                          name_func=unittest_name_func)
    def test_masked_select(self, dtype, input_shape):
        dtype = 'float32'
        mask_shape = input_shape

        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        mask_data = torch.randint(2, mask_shape).to(torch.bool)

        source_shape = (mask_data.nonzero().shape[0])
        source_data = torch.rand(
            source_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='input',
                       shape=input_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='mask',
                       shape=mask_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt('bool'))
            source = Tensor(name='source',
                            shape=source_data.shape,
                            dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.masked_scatter(x, y,
                                                            source).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(fp16=(dtype == 'float16')))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(
                feed_dict={
                    'input': input_data.numpy(),
                    'mask': mask_data.numpy(),
                    'source': source_data.numpy()
                })
        input_data = input_data.cuda()
        ref = input_data.masked_scatter_(mask_data.cuda(), source_data.cuda())
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)
