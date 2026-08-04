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


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        torch.random.manual_seed(0)
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float32', ), ('float16', )],
                          name_func=unittest_name_func)
    def test_embedding(self, dtype):

        # meta data
        batch_size = 10
        vocab_size = 1000
        n_embed = 1024

        # test data
        ## input index
        index_shape = (batch_size, )
        index_data = torch.randint(0,
                                   vocab_size,
                                   index_shape,
                                   dtype=torch.int32,
                                   device="cuda")

        ## weight data
        weight_data = torch.rand(vocab_size,
                                 n_embed,
                                 dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                                 device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()

        with tensorrt_llm.net_guard(network):
            index = Tensor(name='index',
                           shape=index_data.shape,
                           dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            weight = Tensor(name='weight',
                            shape=weight_data.shape,
                            dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            output = tensorrt_llm.functional.embedding(input=index,
                                                       weight=weight)
            output.mark_output('output', dtype)

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'index': index_data,
            'weight': weight_data,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        embedding = torch.nn.Embedding.from_pretrained(weight_data)
        ref = embedding(index_data)

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])
