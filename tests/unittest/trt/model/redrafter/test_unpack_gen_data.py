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


########################################################################################################################

    @parameterized.expand([
        ([3], [[[0.3990, 0.5167, 0.0249, 0.9401],
                [0.9459, 0.7967, 0.4150, 0.8203],
                [0.2290, 0.9096, 0.1183, 0.0752]]]),
        ([3, 4], [[[0.3990, 0.5167, 0.0249, 0.9401],
                   [0.9459, 0.7967, 0.4150, 0.8203],
                   [0.2290, 0.9096, 0.1183, 0.0752],
                   [0.4092, 0.9601, 0.2093, 0.1940]],
                  [[0.4092, 0.9601, 0.2093, 0.1940],
                   [0.8909, 0.4387, 0.3570, 0.5454],
                   [0.8299, 0.2099, 0.7684, 0.4290],
                   [0.2117, 0.6606, 0.1654, 0.4250]]]),
        ([4, 3], [[[0.3990, 0.5167, 0.0249, 0.9401],
                   [0.9459, 0.7967, 0.4150, 0.8203],
                   [0.2290, 0.9096, 0.1183, 0.0752],
                   [0.4092, 0.9601, 0.2093, 0.1940]],
                  [[0.8909, 0.4387, 0.3570, 0.5454],
                   [0.8299, 0.2099, 0.7684, 0.4290],
                   [0.2117, 0.6606, 0.1654, 0.4250],
                   [0.2117, 0.6606, 0.1654, 0.4250]]]),
        ([3, 5, 1], [[[0.3990, 0.5167, 0.0249, 0.9401],
                      [0.9459, 0.7967, 0.4150, 0.8203],
                      [0.2290, 0.9096, 0.1183, 0.0752],
                      [0.4092, 0.9601, 0.2093, 0.1940],
                      [0.8909, 0.4387, 0.3570, 0.5454]],
                     [[0.4092, 0.9601, 0.2093, 0.1940],
                      [0.8909, 0.4387, 0.3570, 0.5454],
                      [0.8299, 0.2099, 0.7684, 0.4290],
                      [0.2117, 0.6606, 0.1654, 0.4250],
                      [0.9927, 0.6964, 0.2472, 0.7028]],
                     [[0.7494, 0.9303, 0.0494, 0.0750],
                      [0.7494, 0.9303, 0.0494, 0.0750],
                      [0.7494, 0.9303, 0.0494, 0.0750],
                      [0.7494, 0.9303, 0.0494, 0.0750],
                      [0.7494, 0.9303, 0.0494, 0.0750]]]),
    ])
    def test_unpack_gen_data(self,
                             num_gen_tokens=[3],
                             ref_res=[[[0.3990, 0.5167, 0.0249, 0.9401],
                                       [0.9459, 0.7967, 0.4150, 0.8203],
                                       [0.2290, 0.9096, 0.1183, 0.0752]]]):
        # test data
        V = 4
        nb = 3
        bl = 4
        old_device = torch.get_default_device()
        torch.set_default_device("cuda")
        torch.manual_seed(0)
        num_gen_tokens = torch.tensor(num_gen_tokens, dtype=torch.int32)
        ref_res = torch.tensor(ref_res, dtype=torch.float32)
        assert torch.any(num_gen_tokens <= (nb * (bl - 1) + 1))
        total_tokens = num_gen_tokens.sum()
        max_gen_token = num_gen_tokens.max().cpu()
        lm_logits = torch.rand((total_tokens, V), dtype=torch.float32)
        gen_unpack_indxs = torch.arange(max_gen_token, dtype=torch.int32)
        gen_unpack_indxs = gen_unpack_indxs.unsqueeze(0) + (
            torch.cumsum(num_gen_tokens, dim=0) - num_gen_tokens).unsqueeze(1)
        gen_unpack_indxs = torch.minimum(gen_unpack_indxs, total_tokens - 1)

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            lm_logits_t = Tensor(name='l',
                                 shape=lm_logits.shape,
                                 dtype=tensorrt_llm.torch_dtype_to_trt(
                                     lm_logits.dtype))
            num_gen_tokens_t = Tensor(name='ng',
                                      shape=num_gen_tokens.shape,
                                      dtype=tensorrt_llm.torch_dtype_to_trt(
                                          num_gen_tokens.dtype))
            gen_unpack_indxs_t = Tensor(name='gui',
                                        shape=gen_unpack_indxs.shape,
                                        dtype=tensorrt_llm.torch_dtype_to_trt(
                                            gen_unpack_indxs.dtype))
            max_gen_token_t = Tensor(name='mgt',
                                     shape=max_gen_token.shape,
                                     dtype=tensorrt_llm.torch_dtype_to_trt(
                                         max_gen_token.dtype))

            outputs = tensorrt_llm.models.redrafter.redrafter_helper._unpack_gen_data(
                lm_logits_t, num_gen_tokens_t, gen_unpack_indxs_t,
                max_gen_token_t)
            outputs.mark_output('res')
            # save onnx
            # model_path = 'unpack_gen.onnx'
            # to_onnx(net.trt_network, model_path)

        # needs profile for dynamic shape
        profile = builder.trt_builder.create_optimization_profile()
        set_input_shape(profile, lm_logits_t, lm_logits.shape, lm_logits)
        set_input_shape(profile, num_gen_tokens_t, num_gen_tokens.shape,
                        num_gen_tokens)
        set_input_shape(profile, gen_unpack_indxs_t, gen_unpack_indxs.shape,
                        gen_unpack_indxs)
        set_input_shape(profile, max_gen_token_t, max_gen_token.shape,
                        max_gen_token)

        # trt run
        session = create_session(builder,
                                 network,
                                 precision='float32',
                                 optimization_profiles=[profile])
        inputs = {
            'l': lm_logits,
            'ng': num_gen_tokens,
            'gui': gen_unpack_indxs,
            'mgt': max_gen_token,
        }
        outputs = run_session(session, inputs)
        # print(outputs)
        torch.testing.assert_close(outputs['res'], ref_res, atol=0.01, rtol=0.1)
        torch.set_default_device(old_device)
        return
