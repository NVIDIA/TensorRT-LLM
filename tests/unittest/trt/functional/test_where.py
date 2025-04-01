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
from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor


class TestWhere(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    @parameterized.expand([
        (True, ),
        (False, ),
    ], name_func=unittest_name_func)
    def test_where_from_bool(self, condition):
        dtype = 'float32'
        t_data = torch.randn(2, 3, device="cuda")
        f_data = torch.randn(2, 3, device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            t = Tensor(name='t',
                       shape=t_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            f = Tensor(name='f',
                       shape=f_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.where(condition, t, f)
            output.mark_output('output')

        session = create_session(builder, network, precision=dtype)
        inputs = {'t': t_data, 'f': f_data}
        outputs = run_session(session, inputs)

        ref = torch.where(torch.tensor(condition).cuda(), t_data, f_data)
        torch.testing.assert_close(ref, outputs['output'])

    def test_where_from_tensor(self):
        dtype = 'float32'
        t_data = torch.randn(3, 4, device="cuda")
        f_data = torch.randn(3, 4, device="cuda")
        c_data = torch.randint(2, size=(3, 1), dtype=torch.bool, device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            t = Tensor(name='t',
                       shape=t_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            f = Tensor(name='f',
                       shape=f_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            c = Tensor(name='c',
                       shape=c_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt('bool'))
            output = tensorrt_llm.functional.where(c, t, f)
            output.mark_output('output')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {'t': t_data, 'f': f_data, 'c': c_data}
        outputs = run_session(session, inputs)

        # pytorch run
        ref = torch.where(c_data, t_data, f_data)

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])
