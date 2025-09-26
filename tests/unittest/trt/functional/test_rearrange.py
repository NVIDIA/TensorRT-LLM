# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import einops
import torch
from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor


class TestRearrange(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([
        ('b h w c -> b h w c', None),
        ('b h w c -> b c h w', None),
        ('b h w c -> (b h) w c', None),
        ('b h w c -> h (b w) c', None),
        ('b h w c -> b (c h w)', None),
        ('b (h1 h) (w1 w) c -> (b h1 w1) h w c', {
            "h1": 1,
            "w1": 2
        }),
        ('b (h h1) (w w1) c -> b h w (c h1 w1)', {
            "h1": 2,
            "w1": 2
        }),
    ],
                          name_func=unittest_name_func)
    def test_rearrange(self, pattern, kwargs):
        # test data
        dtype = 'float32'
        x_shape = (32, 30, 40, 3)
        x_data = torch.rand(x_shape,
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):

            x = Tensor(name='x',
                       shape=x_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            if kwargs is None:
                output = tensorrt_llm.functional.rearrange(x, pattern)
            else:
                output = tensorrt_llm.functional.rearrange(x, pattern, **kwargs)
            output.mark_output('output')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        if kwargs is None:
            ref = einops.rearrange(x_data, pattern)
        else:
            ref = einops.rearrange(x_data, pattern, **kwargs)

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])
