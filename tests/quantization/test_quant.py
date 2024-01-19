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

from tensorrt_llm.layers import ColumnLinear, RowLinear
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import GPTLMHeadModel, quantize_model
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.quantization.layers import (SmoothQuantAttention,
                                              SmoothQuantLayerNorm,
                                              SmoothQuantMLP,
                                              WeightOnlyQuantColumnLinear,
                                              WeightOnlyQuantRowLinear)


class TestQuant(unittest.TestCase):

    def test_weight_only_quant(self):
        mode = QuantMode.use_weight_only()

        model = GPTLMHeadModel(num_layers=2,
                               num_heads=12,
                               hidden_size=768,
                               vocab_size=51200,
                               hidden_act='relu',
                               max_position_embeddings=1024,
                               dtype='float16')

        quant_model = quantize_model(model, mode)

        self.assertTrue(hasattr(quant_model, 'quant_mode'))

        self.assertTrue(
            isinstance(quant_model.layers[0].attention.qkv,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.layers[0].attention.dense,
                       WeightOnlyQuantRowLinear))
        self.assertTrue(
            isinstance(quant_model.layers[0].mlp.fc,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.layers[0].mlp.proj,
                       WeightOnlyQuantRowLinear))

        self.assertTrue(
            isinstance(quant_model.layers[1].attention.qkv,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.layers[1].attention.dense,
                       WeightOnlyQuantRowLinear))
        self.assertTrue(
            isinstance(quant_model.layers[1].mlp.fc,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.layers[1].mlp.proj,
                       WeightOnlyQuantRowLinear))

        self.assertTrue(isinstance(quant_model.lm_head, ColumnLinear))

    def test_weight_only_quant_exclude_modules(self):
        mode = QuantMode.use_weight_only()

        model = GPTLMHeadModel(num_layers=1,
                               num_heads=12,
                               hidden_size=768,
                               vocab_size=51200,
                               hidden_act='relu',
                               max_position_embeddings=1024,
                               dtype='float16')

        quant_model = quantize_model(model,
                                     mode,
                                     exclude_modules=['fc', 'dense'])

        self.assertTrue(hasattr(quant_model, 'quant_mode'))

        self.assertTrue(
            isinstance(quant_model.layers[0].attention.qkv,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.layers[0].attention.dense, RowLinear))
        self.assertTrue(isinstance(quant_model.layers[0].mlp.fc, ColumnLinear))
        self.assertTrue(
            isinstance(quant_model.layers[0].mlp.proj,
                       WeightOnlyQuantRowLinear))
        self.assertTrue(
            isinstance(quant_model.lm_head, WeightOnlyQuantColumnLinear))

    def test_convert_GPT_to_smooth_quant(self):
        gpt = GPTLMHeadModel(num_layers=1,
                             num_heads=1,
                             hidden_size=128,
                             vocab_size=1024,
                             hidden_act='gelu',
                             max_position_embeddings=256,
                             dtype='float16',
                             mapping=Mapping(world_size=1, rank=0, tp_size=1))

        quant_mode = QuantMode.use_smooth_quant()
        sq_gpt = quantize_model(gpt, quant_mode)
        for layer in sq_gpt.layers:
            assert isinstance(layer.input_layernorm, SmoothQuantLayerNorm)
            assert isinstance(layer.post_layernorm, SmoothQuantLayerNorm)
            assert isinstance(layer.mlp, SmoothQuantMLP)
            assert isinstance(layer.attention, SmoothQuantAttention)

        assert sq_gpt.quant_mode == quant_mode


if __name__ == '__main__':
    unittest.main()
