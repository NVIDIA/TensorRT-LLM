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

# isort: off
import torch
import tensorrt as trt
# isort: on

from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float32', ), ('float16', )],
                          name_func=unittest_name_func)
    def test_slice_explicit(self, dtype):
        # test data
        x_shape = (1, 256)
        x_data = torch.rand(x_shape,
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device="cuda")
        starts_data = torch.tensor([0, 128]).int()
        sizes_data = torch.tensor([1, 1]).int()

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):

            x = Tensor(name='x',
                       shape=x_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            starts = Tensor(name='starts', shape=(2, ), dtype=trt.int32)

            sizes = Tensor(name='sizes', shape=(2, ), dtype=trt.int32)

            output = tensorrt_llm.functional.slice(x, starts, sizes)
            output.mark_output('output')

        profile = builder.trt_builder.create_optimization_profile()
        profile.set_shape_input('starts', (0, 128), (0, 128), (0, 128))
        profile.set_shape_input('sizes', (1, 1), (1, 1), (1, 1))

        # trt run
        session = create_session(builder,
                                 network,
                                 precision=dtype,
                                 optimization_profiles=[profile])
        inputs = {'x': x_data, 'starts': starts_data, 'sizes': sizes_data}
        outputs = run_session(session, inputs)

        # pytorch run
        ref = x_data[0:1, 128:129]

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])

    def test_slice_implicit(self):
        dtype = 'float32'
        x_shape = (256, )
        slice_length = 128
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
            starts = tensorrt_llm.functional.constant(
                np.array([0], dtype=np.int32))
            output_length = tensorrt_llm.functional.constant(
                np.array([0] * slice_length, dtype=np.int32))
            sizes = tensorrt_llm.functional.shape(output_length, 0)

            output = tensorrt_llm.functional.slice(x, starts, sizes.view([1]))

            output.mark_output('output')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        ref = x_data[0:slice_length]

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])
