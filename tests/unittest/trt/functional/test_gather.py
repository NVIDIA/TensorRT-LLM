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


class TestGather(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_gather(self):
        dtype = 'float32'
        x_data = torch.randn(2, 128, 768, device="cuda")
        y_data = torch.tensor([101, 127], device="cuda").int()

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):

            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=y_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            y = y.view(
                tensorrt_llm.functional.concat(
                    [tensorrt_llm.functional.shape(y, 0), 1, 1]))
            y = tensorrt_llm.functional.expand(
                y,
                tensorrt_llm.functional.concat([
                    tensorrt_llm.functional.shape(y, 0), 1,
                    tensorrt_llm.functional.shape(x, 2)
                ]))
            output = tensorrt_llm.functional.gather(x, dim=1, indices=y).view(
                tensorrt_llm.functional.concat([
                    tensorrt_llm.functional.shape(x, 0),
                    tensorrt_llm.functional.shape(x, 2)
                ]))
            output.mark_output('output', dtype)

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {'x': x_data, 'y': y_data}
        outputs = run_session(session, inputs)

        # pytorch run
        y_data = y_data.reshape(y_data.size(0), 1, 1)
        y_data = y_data.expand(y_data.size(0), 1, x_data.size(-1))
        ref = torch.gather(x_data, dim=1,
                           index=y_data.to(dtype=torch.int64)).view(
                               x_data.size(0), x_data.size(2))

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])
