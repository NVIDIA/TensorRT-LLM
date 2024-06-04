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
import os
import sys
import unittest

import numpy as np
import torch

import tensorrt_llm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import create_session, run_session


class TestArange(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_arange_int(self):
        # test data
        start = 0
        end = 128

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):

            output = tensorrt_llm.functional.arange(start=start,
                                                    end=end,
                                                    dtype="int32")
            output.mark_output('output', "int32")

        # trt run
        inputs = {}
        session = create_session(builder, network, precision="float32")
        outputs = run_session(session, inputs)

        ref = torch.arange(start, end).int().cuda()
        torch.testing.assert_close(outputs['output'], ref)

    def test_arange_tensor(self):
        # test data
        s = 0
        e = 128
        dtype = 'int32'

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):

            start = tensorrt_llm.functional.constant(np.array(s,
                                                              dtype=np.int32))
            end_tensor = tensorrt_llm.functional.constant(
                np.array([0] * e, dtype=np.int32))

            output = tensorrt_llm.functional.arange(
                start=start,
                end=tensorrt_llm.functional.shape(end_tensor, 0),
                dtype=dtype)

            output.mark_output('output', dtype)

        # trt run
        inputs = {}
        session = create_session(builder, network, precision="float32")
        outputs = run_session(session, inputs)

        # pytorch run
        ref = torch.arange(s, e).int().cuda()

        # compare diff
        torch.testing.assert_close(outputs['output'], ref)
