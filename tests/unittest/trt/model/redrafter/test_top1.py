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
import tensorrt_llm.models.redrafter
import tensorrt_llm.models.redrafter.redrafter_helper
from tensorrt_llm import Tensor

NINF = -50000.0


class TestReDrafter(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')


########################################################################################################################

    def test_top_1_logits(self):
        # test data
        bs = 2
        S = 5
        V = 4
        old_device = torch.get_default_device()
        torch.set_default_device("cuda")
        torch.manual_seed(0)
        logits = torch.rand((bs, S, V), dtype=torch.float32)
        ref_res = torch.tensor(
            [[[NINF, NINF, NINF, -0.], [-0., NINF, NINF, NINF],
              [NINF, -0., NINF, NINF], [NINF, -0., NINF, NINF],
              [-0., NINF, NINF, NINF]],
             [[-0., NINF, NINF, NINF], [NINF, -0., NINF, NINF],
              [-0., NINF, NINF, NINF], [NINF, -0., NINF, NINF],
              [NINF, -0., NINF, NINF]]],
            dtype=torch.float32)

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            logits_t = Tensor(name='logits',
                              shape=logits.shape,
                              dtype=tensorrt_llm.torch_dtype_to_trt(
                                  logits.dtype))

            outputs = tensorrt_llm.models.redrafter.redrafter_helper._top_1_logits(
                logits_t, NINF)
            outputs.mark_output("outputs")
        # trt run
        session = create_session(
            builder,
            network,
            precision='float32',
        )
        inputs = {
            'logits': logits,
        }
        outputs = run_session(session, inputs)
        torch.testing.assert_close(outputs['outputs'], ref_res, rtol=0, atol=0)
        torch.set_default_device(old_device)
        return
