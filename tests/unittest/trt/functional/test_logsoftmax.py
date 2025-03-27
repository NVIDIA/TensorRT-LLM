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

# isort: off
import torch
# isort: on
from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor


class TestLogSoftmax(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_lt(self):
        dtype = 'float32'
        t_shape = (2, 3)
        x_data = torch.rand(t_shape,
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device="cuda")
        y_data = torch.rand(t_shape,
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            x = Tensor(name='x',
                       shape=t_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=t_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            output = tensorrt_llm.functional.lt(x, y)
            output.mark_output('output')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {'x': x_data, 'y': y_data}
        outputs = run_session(session, inputs)

        # pytorch run
        ref = torch.lt(x_data, y_data)
        # compare diff
        torch.testing.assert_close(ref, outputs['output'])

    @parameterized.expand(list(product(['float32'])))
    def test_log(self, dtype):
        # test data
        x_shape = (4, 6, 8)
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

            output = tensorrt_llm.functional.log(x)
            output.mark_output('output')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        ref = x_data.log()

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])

    @parameterized.expand(list(product(['float32'], [0, 1, 2], [False, True])),
                          name_func=unittest_name_func)
    def test_sum(self, dtype, dim, keepdim):
        # test data
        x_shape = (4, 6, 8)
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

            output = tensorrt_llm.functional.sum(x, dim, keepdim)
            output.mark_output('output')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        ref = x_data.sum(dim=dim, keepdim=keepdim)

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])

    @parameterized.expand(list(product(['float32'], [0, 1, 2])),
                          name_func=unittest_name_func)
    def test_log_softmax(self, dtype, dim):
        # test data
        x_shape = (4, 6, 8)
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

            output = tensorrt_llm.functional.log_softmax(x, dim=dim)
            output.mark_output('output')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        ref = x_data.log_softmax(dim=dim)

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])
