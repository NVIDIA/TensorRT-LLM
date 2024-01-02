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

import numpy as np
import torch
from parameterized import parameterized
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner
from transformers.models.bloom.modeling_bloom import build_alibi_tensor

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def create_random_bool_mask(self, batch_size, seq_len):
        mask = torch.zeros(size=[batch_size, seq_len], dtype=torch.bool)
        seq_lens = torch.randint(low=1, high=seq_len + 1, size=[batch_size])

        for b in range(batch_size):
            mask[b, :seq_lens[b]] = True

        return mask

    # We don't run alibi in FP16, so only check FP32 here.
    @parameterized.expand([(1, 64, 32), (16, 1, 64), (24, 20, 500),
                           (32, 128, 60), (64, 32, 1024), (80, 12, 20),
                           (112, 4, 389)])
    def test_alibi_biases(self, num_heads, batch_size, seq_len):

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            trt_key = Tensor(name='fake_key',
                             shape=(seq_len, ),
                             dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            key_len = tensorrt_llm.functional.shape(trt_key, 0)
            slopes = tensorrt_llm.functional.generate_alibi_slopes(
                num_heads=num_heads)
            output = tensorrt_llm.functional.generate_alibi_biases(
                slopes, key_len).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            print(seq_len)
            outputs = runner.infer(
                feed_dict={
                    'fake_key': np.empty(shape=(seq_len, ), dtype=np.int32)
                })

        trt_alibi_output = outputs['output']

        # transformers reference
        binary_mask = self.create_random_bool_mask(batch_size, seq_len)
        ref = build_alibi_tensor(binary_mask, num_heads,
                                 torch.float32).cpu().numpy()
        ref = ref.reshape(batch_size, num_heads, 1, seq_len)

        # We only require that the alibi bias matches in the "valid" regions. Our TRT,
        # implementation differs in this regard for efficiency reasons but it does not matter
        # because these values will get masked before the softmax.
        binary_mask = binary_mask.cpu().numpy().reshape(batch_size, 1, 1,
                                                        seq_len)
        ref *= binary_mask

        trt_alibi_output = np.repeat(trt_alibi_output, batch_size, axis=0)
        trt_alibi_output *= binary_mask

        # compare diff
        np.testing.assert_allclose(ref, trt_alibi_output, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
