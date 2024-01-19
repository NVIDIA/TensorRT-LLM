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
import math
import unittest

import numpy as np
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor


def split_vocab_size(vocab_size, tp_size):
    return int(math.ceil(vocab_size / tp_size))


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])
    return None


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([(
        'float32',
        1,
    ), (
        'float32',
        0,
    ), (
        'float16',
        1,
    ), (
        'float16',
        0,
    )])
    def test_embedding(self, dtype, use_lookup_plugin):
        # torch gelu does not support float16
        fp16 = (dtype == 'float16')

        # meta data
        batch_size = 10
        vocab_size = 1000
        n_embed = 1024
        np.random.seed(0)

        # test data
        ## input index
        index_shape = (batch_size)
        index_np = np.random.randint(low=0,
                                     high=vocab_size,
                                     size=index_shape,
                                     dtype=np.int32)
        index_data = torch.from_numpy(index_np)

        ## weight data
        weight_np = np.random.rand(vocab_size, n_embed).astype(dtype)
        weight_data = torch.from_numpy(weight_np)

        # construct trt network
        builder = tensorrt_llm.Builder()
        # builder_config = builder.create_builder_config(
        #         name='embedding',
        #         precision='float16' if fp16 else 'float32',
        #         timing_cache=timing_cache)

        net = builder.create_network()
        if use_lookup_plugin:
            net.plugin_config.set_lookup_plugin(dtype)

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            index = Tensor(name='index',
                           shape=index_data.shape,
                           dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            weight = Tensor(name='weight',
                            shape=weight_data.shape,
                            dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            output = tensorrt_llm.functional.embedding(input=index,
                                                       weight=weight)

            output = output.trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(fp16=(dtype == 'float16')))

        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'index': index_np,
                'weight': weight_np
            })

        # pytorch run
        embedding = torch.nn.Embedding.from_pretrained(weight_data)
        ref = embedding(index_data)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-3)
