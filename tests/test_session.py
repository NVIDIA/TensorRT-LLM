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

import tensorrt_llm


class MyAddModule(tensorrt_llm.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class TestSession(unittest.TestCase):

    def test_session_debug_run(self):
        tensorrt_llm.logger.set_level('verbose')
        builder = tensorrt_llm.Builder()
        builder_config = builder.create_builder_config("test", "llmTimingCache")
        model = MyAddModule()

        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            x = tensorrt_llm.Tensor(name='x', dtype=trt.float32, shape=[1, 1])
            y = tensorrt_llm.Tensor(name='y', dtype=trt.float32, shape=[1, 1])
            # Prepare
            network.set_named_parameters(model.named_parameters())
            # Forward
            z = model(x, y)
            z.mark_output('z', trt.float32)

            ### Addtionl debug tensor
            debug_tensor = x * y
            debug_tensor.mark_output('debug_tensor', trt.float32)

        engine = builder.build_engine(network, builder_config)
        assert engine is not None

        # Show to _debug_run can be used
        # You need to mark "z" and "debug_tensor" as output, and then use Session._debug_run
        # to run inference and get the output
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        inputs = {
            'x': torch.rand([1, 1], device='cuda'),
            'y': torch.rand([1, 1], device='cuda')
        }
        outputs = session._debug_run(inputs)
        assert 'z' in outputs and 'debug_tensor' in outputs

        expected_debug_tensor = inputs['x'] * inputs['y']
        expected_z = inputs['x'] + inputs['y']
        self.assertTrue(
            torch.allclose(outputs['debug_tensor'], expected_debug_tensor))
        self.assertTrue(torch.allclose(outputs['z'], expected_z))


if __name__ == '__main__':
    unittest.main()
