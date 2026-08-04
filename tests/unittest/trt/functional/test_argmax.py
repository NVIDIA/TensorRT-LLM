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
from utils.util import create_session, run_session

import tensorrt_llm
from tensorrt_llm import Tensor


class TestArgmax(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand(
        list(product(['float32', 'float16'], [False, True], [0, 1, 2])))
    def test_argmax(self, dtype, keep_dim, dim):
        # test data
        x_shape = (4, 12, 32)
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

            output = tensorrt_llm.functional.argmax(x, dim, keepdim=keep_dim)
            output.mark_output('output')

        # trt run
        inputs = {'x': x_data}
        session = create_session(builder, network, precision=dtype)
        outputs = run_session(session, inputs)

        # pytorch run
        ref = x_data.argmax(dim=dim, keepdim=keep_dim)

        # compare diff
        # ref is torch.int64, outputs is torch.int32
        torch.testing.assert_close(ref.int(), outputs['output'].int())
