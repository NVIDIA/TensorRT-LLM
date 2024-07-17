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
from tensorrt_llm.models import GPTForCausalLM, PretrainedConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.quantization.layers import (SmoothQuantAttention,
                                              SmoothQuantLayerNorm,
                                              SmoothQuantMLP,
                                              WeightOnlyQuantColumnLinear,
                                              WeightOnlyQuantRowLinear)
from tensorrt_llm.quantization.quantize import quantize


class TestQuant(unittest.TestCase):

    def test_weight_only_quant(self):
        quant_algo = QuantAlgo.W8A16
        config = {
            'architecture': 'GPTForCausalLM',
            'dtype': 'float16',
            'num_hidden_layers': 2,
            'num_attention_heads': 12,
            'hidden_size': 768,
            'vocab_size': 51200,
            'max_position_embeddings': 1024,
            'hidden_act': 'relu',
        }
        config = PretrainedConfig.from_dict(config)
        model = GPTForCausalLM(config)

        quant_model = quantize(model, QuantConfig(quant_algo))

        self.assertTrue(hasattr(quant_model, 'quant_mode'))

        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].attention.qkv,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].attention.dense,
                       WeightOnlyQuantRowLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].mlp.fc,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].mlp.proj,
                       WeightOnlyQuantRowLinear))

        self.assertTrue(
            isinstance(quant_model.transformer.layers[1].attention.qkv,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[1].attention.dense,
                       WeightOnlyQuantRowLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[1].mlp.fc,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[1].mlp.proj,
                       WeightOnlyQuantRowLinear))

        self.assertTrue(isinstance(quant_model.lm_head, ColumnLinear))

    def test_weight_only_quant_exclude_modules(self):
        quant_algo = QuantAlgo.W8A16
        config = {
            'architecture': 'GPTForCausalLM',
            'dtype': 'float16',
            'num_hidden_layers': 1,
            'num_attention_heads': 12,
            'hidden_size': 768,
            'vocab_size': 51200,
            'max_position_embeddings': 1024,
            'hidden_act': 'relu',
        }
        config = PretrainedConfig.from_dict(config)
        model = GPTForCausalLM(config)

        quant_model = quantize(
            model,
            QuantConfig(quant_algo,
                        exclude_modules=[
                            'fc', 'dense', 'vocab_embedding',
                            'position_embedding', 'block_embedding'
                        ]))

        self.assertTrue(hasattr(quant_model, 'quant_mode'))

        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].attention.qkv,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].attention.dense,
                       RowLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].mlp.fc, ColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].mlp.proj,
                       WeightOnlyQuantRowLinear))
        self.assertTrue(
            isinstance(quant_model.lm_head, WeightOnlyQuantColumnLinear))

    def test_convert_GPT_to_smooth_quant(self):
        config = {
            'architecture': 'GPTForCausalLM',
            'dtype': 'float16',
            'num_hidden_layers': 1,
            'num_attention_heads': 1,
            'hidden_size': 128,
            'vocab_size': 1024,
            'max_position_embeddings': 256,
            'hidden_act': 'gelu',
        }
        config = PretrainedConfig.from_dict(config)
        model = GPTForCausalLM(config)

        quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN
        quant_config = QuantConfig(quant_algo)
        quant_model = quantize(model, quant_config)
        for layer in quant_model.transformer.layers:
            assert isinstance(layer.input_layernorm, SmoothQuantLayerNorm)
            assert isinstance(layer.post_layernorm, SmoothQuantLayerNorm)
            assert isinstance(layer.mlp, SmoothQuantMLP)
            assert isinstance(layer.attention, SmoothQuantAttention)

        assert quant_model.quant_mode == quant_config.quant_mode


if __name__ == '__main__':
    unittest.main()
