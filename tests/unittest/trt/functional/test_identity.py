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


class TestIdentity(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float32', False), ('float32', True),
                           ('float16', False), ('float16', True),
                           ('bfloat16', False), ('bfloat16', True)],
                          name_func=unittest_name_func)
    def test_identity(self, dtype, use_plugin):
        x_data = torch.randn(
            (4, 6, 3, 4),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        if use_plugin:
            network.plugin_config.identity_plugin = dtype

        with tensorrt_llm.net_guard(network):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.identity(x)
            output.mark_output('output', dtype)

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        # compare diff
        torch.testing.assert_close(x_data, outputs['output'])
