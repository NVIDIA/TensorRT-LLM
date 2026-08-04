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
from itertools import product

import torch
from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor


class TestCumsum(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([
        ('int32', (256, ), 0),
        ('int32', (256, ), -1),
        ('float32', (3, 16), 0),
        ('float32', (3, 16), 1),
        ('float32', (3, 16), -2),
        ('float16', (5, 6, 8), 1),
        ('float16', (5, 6, 8), 2),
        ('float16', (5, 6, 8), -3),
        ('float32', (1, 512), -1),
        ('float16', (3, 5, 5, 6), -1),
        ('int32', (1, 33), -1),
        ('int32', (1, 65), -1),
        ('float32', (1, 50000), -1),
        ('float32', (1, 2, 50000), -1),
        ('float32', (3, 5, 5, 50000), -1),
    ],
                          name_func=unittest_name_func)
    def test_cumsum(self, dtype, x_shape, dim):
        torch_dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)
        if 'float' in dtype:
            x_data = torch.rand(x_shape, dtype=torch_dtype, device="cuda")
        else:
            x_data = torch.randint(-100,
                                   100,
                                   x_shape,
                                   dtype=torch_dtype,
                                   device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):

            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.cumsum(x, dim=dim)
            output.mark_output('output', dtype)

        # trt run
        session = create_session(
            builder,
            network,
            precision='float32' if 'int32' in dtype else dtype)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        ref = torch.cumsum(x_data, dim=dim).to(torch_dtype)

        # compare diff
        tols = {
            "float32": {
                "rtol": 1e-05,
                "atol": 1e-05
            },
            "float16": {
                "rtol": 1e-02,
                "atol": 1e-02
            },
            "int32": {
                "rtol": 0,
                "atol": 0
            },
        }
        torch.testing.assert_close(outputs['output'], ref, **tols[dtype])

    @parameterized.expand(
        list(
            product(['float32', 'float16', 'int32'],
                    [(256, ), (3, 16), (5, 6, 8)], [True, False])) +
        list(product(['float32'], [(3, 5, 5, 50000)],
                     [True])),  # False seems to be running into a TRT bug
        name_func=unittest_name_func)
    def test_cumsum_dynamic_last_dim(self, dtype, x_shape, prefer_plugin=True):
        dim = -1
        torch_dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)
        if 'float' in dtype:
            x_data = torch.rand(x_shape, dtype=torch_dtype, device="cuda")
        else:
            x_data = torch.randint(-100,
                                   100,
                                   x_shape,
                                   dtype=torch_dtype,
                                   device="cuda")

        shape_except_last_dim = list(x_data.shape[:-1])
        last_dim_size = x_data.shape[-1]
        assert last_dim_size >= 1
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            x = Tensor(
                name='x',
                shape=shape_except_last_dim + [-1],  # last dim dynamic
                dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.cumsum(x,
                                                    dim=dim,
                                                    prefer_plugin=prefer_plugin)
            output.mark_output('output', dtype)
        # needs profile for dynamic shape
        profile = builder.trt_builder.create_optimization_profile()
        profile.set_shape('x', shape_except_last_dim + [1],
                          shape_except_last_dim + [last_dim_size],
                          shape_except_last_dim + [last_dim_size * 2])
        session = create_session(
            builder,
            network,
            precision='float32' if 'int32' in dtype else dtype,
            optimization_profiles=[profile])
        inputs = {'x': x_data}
        outputs = run_session(session, inputs)

        ref = torch.cumsum(x_data, dim=dim).to(torch_dtype)

        # compare diff
        tols = {
            "float32": {
                "rtol": 1e-05,
                "atol": 1e-05
            },
            "float16": {
                "rtol": 1e-02,
                "atol": 1e-02
            },
            "int32": {
                "rtol": 0,
                "atol": 0
            },
        }
        torch.testing.assert_close(outputs['output'], ref, **tols[dtype])
