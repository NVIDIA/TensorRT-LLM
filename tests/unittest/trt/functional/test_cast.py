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
from utils.util import create_session, run_session

import tensorrt_llm
from tensorrt_llm import Tensor


class TestCast(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_cast_fp16_to_fp32(self):
        dtype = 'float16'
        x_data = torch.randn(
            (2, 3, 4, 5),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.cast(x, 'float32')
            output.mark_output('output', 'float32')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        ref = x_data.to(torch.float32)

        # compare diff
        assert ref.dtype == outputs[
            'output'].dtype, "data type after cast is not the same"
        torch.testing.assert_close(ref, outputs['output'])
