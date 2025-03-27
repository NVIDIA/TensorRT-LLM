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
from transformers.models.bloom.modeling_bloom import build_alibi_tensor
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor


class TestAlibi(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def create_random_bool_mask(self, batch_size, seq_len):
        mask = torch.zeros(size=[batch_size, seq_len],
                           dtype=torch.bool,
                           device="cuda")
        seq_lens = torch.randint(low=1,
                                 high=seq_len + 1,
                                 size=[batch_size],
                                 device="cuda")

        for b in range(batch_size):
            mask[b, :seq_lens[b]] = True

        return mask

    # We don't run alibi in FP16, so only check FP32 here.
    @parameterized.expand([(1, 64, 32), (16, 1, 64), (24, 20, 500),
                           (32, 128, 60), (64, 32, 1024), (80, 12, 20),
                           (112, 4, 389)],
                          name_func=unittest_name_func)
    def test_alibi_biases(self, num_heads, batch_size, seq_len):

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            trt_key = Tensor(name='fake_key',
                             shape=(seq_len, ),
                             dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            key_len = tensorrt_llm.functional.shape(trt_key, 0)
            slopes = tensorrt_llm.functional.constant(
                tensorrt_llm.functional.generate_alibi_slopes(
                    num_heads=num_heads))
            output = tensorrt_llm.functional.generate_alibi_biases(
                slopes, key_len)
            output.mark_output('output')

        # trt run
        inputs = {
            'fake_key': torch.empty((seq_len, ),
                                    dtype=torch.int32,
                                    device="cuda")
        }
        session = create_session(builder, network, precision="float32")
        outputs = run_session(session, inputs)

        trt_alibi_output = outputs['output']

        # transformers reference
        binary_mask = self.create_random_bool_mask(batch_size, seq_len)
        ref = build_alibi_tensor(binary_mask, num_heads, torch.float32)
        ref = ref.reshape(batch_size, num_heads, 1, seq_len)

        # We only require that the alibi bias matches in the "valid" regions. Our TRT,
        # implementation differs in this regard for efficiency reasons but it does not matter
        # because these values will get masked before the softmax.
        binary_mask = binary_mask.reshape(batch_size, 1, 1, seq_len)
        ref *= binary_mask

        trt_alibi_output = torch.repeat_interleave(trt_alibi_output,
                                                   batch_size,
                                                   dim=0)
        trt_alibi_output *= binary_mask

        # compare diff
        torch.testing.assert_close(trt_alibi_output, ref, atol=1e-3, rtol=1e-2)
