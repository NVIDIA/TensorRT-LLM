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
from utils.util import create_session, run_session, set_input_shape

import tensorrt_llm
import tensorrt_llm.models.redrafter
import tensorrt_llm.models.redrafter.redrafter_helper
from tensorrt_llm import Tensor


class TestReDrafter(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    @parameterized.expand([0, 4])
    def test_gather_beams(self, v=0):
        bs = 2
        nb = 3
        old_device = torch.get_default_device()
        torch.set_default_device("cuda")
        torch.manual_seed(0)
        # test data
        if v is None or v == 0:
            x = torch.randint(100, (bs, nb), dtype=torch.int32)
            ref_res = torch.tensor([[97, 97, 41], [72, 4, 72]],
                                   dtype=torch.int32)
        else:
            x = torch.randint(100, (bs, nb, v), dtype=torch.int32)
            ref_res = torch.tensor(
                [[[53, 4, 3, 33], [53, 4, 3, 33], [41, 97, 91, 72]],
                 [[98, 84, 78, 8], [12, 57, 51, 75], [98, 84, 78, 8]]],
                dtype=torch.int32)
        bi = torch.randint(nb, size=(bs, nb), dtype=torch.int32)
        ind = torch.concat([
            torch.arange(bs * nb, dtype=torch.int32).view(-1, 1) // nb,
            bi.view(-1, 1)
        ],
                           dim=1)
        bs = torch.tensor([bs], dtype=torch.int32, device="cpu")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            x_t = Tensor(name='x',
                         shape=x.shape,
                         dtype=tensorrt_llm.torch_dtype_to_trt(x.dtype))
            ind_t = Tensor(name='ind',
                           shape=ind.shape,
                           dtype=tensorrt_llm.torch_dtype_to_trt(ind.dtype))
            bs_t = Tensor(name='bs',
                          shape=bs.shape,
                          dtype=tensorrt_llm.torch_dtype_to_trt(bs.dtype))

            outputs = tensorrt_llm.models.redrafter.redrafter_helper._gather_beams(
                x_t, ind_t, bs_t, nb)
            outputs.mark_output('res')
        profile = builder.trt_builder.create_optimization_profile()
        set_input_shape(profile, x_t, x.shape, x)
        set_input_shape(profile, ind_t, ind.shape, ind)
        set_input_shape(profile, bs_t, bs.shape, bs)

        # trt run
        session = create_session(builder,
                                 network,
                                 precision='float32',
                                 optimization_profiles=[profile])
        inputs = {
            'x': x,
            'ind': ind,
            'bs': bs,
        }
        outputs = run_session(session, inputs)
        torch.testing.assert_close(outputs['res'], ref_res, rtol=0, atol=0)
        torch.set_default_device(old_device)
        return
