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

# isort: off
import torch
import tensorrt as trt
# isort: on

from utils.util import create_session, run_session

import tensorrt_llm
from tensorrt_llm import Tensor


class TestView(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_view_static(self):
        # test data
        dtype = 'float32'
        input_shape = (4, 3)
        output_shape = (12, 1)
        input_data = torch.rand(
            input_shape,
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):

            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.view(input=input,
                                                  shape=output_shape)
            output.mark_output('output')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'input': input_data,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        ref = input_data.view(output_shape)

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])

    def test_view_dynamic(self):
        # test data
        dtype = 'float32'
        input_shape = (4, 3)
        output_shape = (2, 6)
        input_data = torch.rand(
            input_shape,
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device="cuda")
        shape_data = torch.tensor(output_shape).int()

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):

            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            shape = Tensor(name='shape',
                           shape=(len(input_shape), ),
                           dtype=trt.int32)
            output = tensorrt_llm.functional.view(input=input, shape=shape)
            output.mark_output('output')

        # trt run
        profile = builder.trt_builder.create_optimization_profile()
        profile.set_shape_input('shape', output_shape, output_shape,
                                output_shape)
        session = create_session(builder,
                                 network,
                                 precision=dtype,
                                 optimization_profiles=[profile])
        inputs = {'input': input_data, 'shape': shape_data}
        outputs = run_session(session, inputs)

        # pytorch run
        ref = input_data.view(output_shape)

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])
